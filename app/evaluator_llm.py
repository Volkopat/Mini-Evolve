import yaml
import re
import httpx
import asyncio
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os
from typing import Optional, Tuple, List, Dict, Any, Callable
from dotenv import load_dotenv

from .logger_setup import get_logger, VERBOSE_LEVEL_NUM
from . import llm_tools # For hierarchical generation

load_dotenv()

# --- Module Level Variables ---
LAST_USED_PROMPT = "No prompt generated yet."
evaluator_templates: Dict[str, Optional[Template]] = {
    "default": None,
    "hierarchical_orchestrator": None,
    "delegated_subtask": None,
}
main_config = {}
problem_config = {}
prompt_context_parts = {} # Problem-specific context for evaluator generation
llm_config = {}

# --- Evaluator Templating Constants ---
DEFAULT_EVALUATOR_TEMPLATE_NAME = "evaluator_generation_prompt.jinja"
HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME = "hierarchical_evaluator_generation_prompt.jinja"
DELEGATED_SUBTASK_TEMPLATE_NAME = "delegated_evaluator_subtask_prompt.jinja"

# --- Configuration Loading ---
MAIN_CONFIG_FILE = "config/config.yaml"
PROBLEM_CONFIG_FILENAME = "problem_config.yaml"
PROMPT_CONTEXT_FILENAME = "prompt_context.yaml" # This is for problem-specific context for evaluator generation

def load_all_configs():
    global main_config, problem_config, prompt_context_parts, llm_config
    logger = get_logger("EvaluatorLLMConfig")
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("CRITICAL: Main configuration file %s not found." % MAIN_CONFIG_FILE)
        raise
    except yaml.YAMLError as e:
        logger.critical("CRITICAL: Error parsing main YAML configuration: %s" % e)
        raise

    llm_config = main_config.get('llm', {})
    llm_config.setdefault('enable_hierarchical_generation', False)
    llm_config.setdefault('max_delegation_depth', 1)
    llm_config.setdefault('max_sub_tasks_per_step', 3)
    llm_config.setdefault('delegation_iteration_limit', 5)

    current_problem_dir = main_config.get('current_problem_directory')
    if not current_problem_dir:
        logger.critical("CRITICAL: current_problem_directory not set in %s" % MAIN_CONFIG_FILE)
        raise ValueError("current_problem_directory not set in %s" % MAIN_CONFIG_FILE)

    problem_config_path = os.path.join(current_problem_dir, PROBLEM_CONFIG_FILENAME)
    try:
        with open(problem_config_path, 'r') as f:
            problem_config = yaml.safe_load(f)
        if not problem_config or not isinstance(problem_config, dict):
            logger.warning("Problem config file %s is empty or not a dictionary. Problem-specific params might be missing." % problem_config_path)
            problem_config = {} 
    except FileNotFoundError:
        logger.warning("Problem config file %s not found. Using defaults if any, or LLM may lack problem-specific params." % problem_config_path)
        problem_config = {}
    except yaml.YAMLError as e:
        logger.warning("Error parsing problem YAML configuration %s: %s. Using defaults." % (problem_config_path, e))
        problem_config = {}

    prompt_context_path = os.path.join(current_problem_dir, PROMPT_CONTEXT_FILENAME)
    try:
        with open(prompt_context_path, 'r') as f:
            prompt_context_parts = yaml.safe_load(f)
        if not prompt_context_parts or not isinstance(prompt_context_parts, dict):
            logger.warning("Prompt context file %s is empty or not a dictionary. Evaluator context might be incomplete." % prompt_context_path)
            prompt_context_parts = {} 
    except FileNotFoundError:
        logger.warning("Prompt context file %s not found. Evaluator generation might be less effective." % prompt_context_path)
        prompt_context_parts = {} 
    except yaml.YAMLError as e:
        logger.warning("Error parsing prompt context YAML %s: %s. Evaluator generation might be less effective." % (prompt_context_path, e))
        prompt_context_parts = {}

load_all_configs()

# Setup Jinja2 environment
template_env: Optional[Environment] = None
try:
    if not os.path.exists("templates"):
        os.makedirs("templates", exist_ok=True)
        
    template_env = Environment(
        loader=FileSystemLoader("templates/"), 
        autoescape=select_autoescape(['html', 'xml', 'jinja'])
    )
except Exception as e:
    get_logger("EvaluatorLLMTemplate").warning("Jinja2 FileSystemLoader for 'templates/' failed: %s. Prompting will fail if template not found." % e)
    template_env = None

def _initialize_evaluator_templates():
    global evaluator_templates, template_env
    logger = get_logger("EvaluatorLLMTemplate")

    if not template_env:
        logger.critical("Jinja2 template_env not initialized. Cannot load evaluator templates from file.")
        return

    template_names = {
        "default": DEFAULT_EVALUATOR_TEMPLATE_NAME,
        "hierarchical_orchestrator": HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME,
        "delegated_subtask": DELEGATED_SUBTASK_TEMPLATE_NAME,
    }

    for key, name in template_names.items():
        try:
            evaluator_templates[key] = template_env.get_template(name)
            logger.info("Successfully loaded evaluator template: %s as %s" % (name, key))
        except Exception as e:
            logger.critical("Failed to load evaluator template '%s' (for %s): %s." % (name, key, e))
            evaluator_templates[key] = None

_initialize_evaluator_templates()

async def _get_llm_completion(prompt_text: str, client: httpx.AsyncClient) -> Tuple[Optional[str], str]:
    global LAST_USED_PROMPT, llm_config
    LAST_USED_PROMPT = prompt_text 
    logger = get_logger("EvaluatorLLMCompletion")
    
    model_name = llm_config.get('model_name')
    provider = llm_config.get('provider')
    base_url = llm_config.get('base_url')

    if not provider or not model_name:
        logger.error("LLM provider or model_name is missing in main_config.")
        return None, prompt_text

    try:
        system_message_content = ("You are a helpful assistant for generating Python evaluation logic. "
                                  "Follow the user's instructions carefully and precisely. "
                                  "Output only the requested Python code or delegation requests.")

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt_text}
        ]
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": llm_config.get('temperature', 0.7),
            "max_tokens": llm_config.get('max_tokens', 2048),
            "stream": False 
        }
        
        headers = {"Content-Type": "application/json"}
        api_key_to_use = None

        if provider == "ollama_local":
            if not base_url:
                logger.error("base_url is required for ollama_local provider.")
                return None, prompt_text
            api_key_to_use = llm_config.get('api_key')
            api_endpoint = base_url.rstrip('/')
            if not api_endpoint.endswith("/v1/chat/completions"):
                 api_endpoint = os.path.join(api_endpoint, "v1/chat/completions")

        elif provider == "openrouter":
            api_key_env_var = llm_config.get('api_key_env_var')
            if not api_key_env_var:
                logger.error("api_key_env_var is required for openrouter provider in config.")
                return None, prompt_text
            api_key_to_use = os.getenv(api_key_env_var)
            if not api_key_to_use:
                logger.error("Could not retrieve API key from environment variable: %s" % api_key_env_var)
                return None, prompt_text
            if not base_url: base_url = "https://openrouter.ai/api/v1"
            api_endpoint = base_url.rstrip('/') + "/chat/completions"
            
            http_referer = llm_config.get('http_referer')
            if http_referer and http_referer != "YOUR_SITE_URL": headers["HTTP-Referer"] = http_referer
            x_title = llm_config.get('x_title')
            if x_title and x_title != "YOUR_SITE_NAME": headers["X-Title"] = x_title
        else:
            logger.error("Unsupported LLM provider: %s" % provider)
            return None, prompt_text

        if api_key_to_use:
            headers["Authorization"] = "Bearer %s" % api_key_to_use
        
        request_timeout = float(llm_config.get('timeout_seconds', 300.0))
        response = await client.post(api_endpoint, json=payload, headers=headers, timeout=request_timeout)
        response.raise_for_status() 
        
        json_response = response.json()
        
        if json_response.get("choices") and json_response["choices"][0].get("message"):
            return json_response["choices"][0]["message"].get("content", "").strip(), prompt_text
        else:
            logger.warning("LLM response structure unexpected: %s" % str(json_response)[:500])
            return None, prompt_text
    
    except httpx.HTTPStatusError as e:
        logger.error("HTTP error calling %s API: %s - Response: %s" % (provider, e, e.response.text[:500]))
        return None, prompt_text
    except httpx.RequestError as e:
        logger.error("Request error calling %s API. Details: %s" % (provider, repr(e)))
        return None, prompt_text
    except Exception as e:
        logger.error("Generic error calling %s API. Details: %s" % (provider, repr(e)))
        return None, prompt_text

def _extract_evaluator_code(response_text: str) -> Optional[str]:
    logger = get_logger("EvaluatorLLMParser")
    if not response_text:
        return None
    
    if "<delegate_subtask>" in response_text: 
        logger.debug("Delegation tags found in response, not extracting as direct evaluator code.")
        return None
    
    # 1. Try to find code within ```python ... ```
    match = re.search(r"^```python\s*\n(.*?)\n```$", response_text, re.DOTALL | re.MULTILINE)
    if match:
        extracted_code = match.group(1).strip()
        logger.debug("Extracted evaluator code using regex from '```python' markdown fence.")
        return extracted_code
    
    # 2. If not found, try to find code within generic ``` ... ```
    match = re.search(r"^```\s*\n(.*?)\n```$", response_text, re.DOTALL | re.MULTILINE)
    if match:
        extracted_code = match.group(1).strip()
        logger.debug("Extracted evaluator code using regex from generic '```' markdown fence.")
        return extracted_code

    # 3. Fallback: If no markdown fences, assume the entire cleaned response is the code.
    cleaned_fallback = re.sub(r"<([^>]+)>.*?<\/\1>", "", response_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_fallback = re.sub(r"<[^>]+\/>", "", cleaned_fallback, flags=re.DOTALL | re.IGNORECASE)
    cleaned_fallback = cleaned_fallback.strip()
 
    if not cleaned_fallback:
        logger.warning("No evaluator code could be extracted. Response was likely empty or only non-code elements after cleaning.")
        return None
    
    logger.debug("Extracted evaluator code using fallback (assuming entire response is code after cleaning).")
    return cleaned_fallback

async def generate_evaluator_logic(
    context_from_evolution_loop: dict,
    client: httpx.AsyncClient,
    current_delegation_depth: int = 0
) -> Tuple[Optional[str], str]:
    global evaluator_templates, prompt_context_parts, problem_config, llm_config
    logger = get_logger("EvaluatorLLMGenerator")

    hierarchical_enabled = llm_config.get('enable_hierarchical_generation', False)
    max_depth = llm_config.get('max_delegation_depth', 1)
    max_iterations = llm_config.get('delegation_iteration_limit', 5)
    enable_self_correction = llm_config.get('enable_self_correction', False) # New: Self-correction flag
    max_correction_attempts = llm_config.get('max_correction_attempts', 3) # New: Max attempts
    
    current_evaluator_template_key = "default"
    if hierarchical_enabled and current_delegation_depth == 0 :
        current_evaluator_template_key = "hierarchical_orchestrator"

    evaluator_template_to_use = evaluator_templates.get(current_evaluator_template_key)
    if evaluator_template_to_use is None:
        logger.critical("Evaluator template '%s' is not available. Cannot generate evaluator." % current_evaluator_template_key)
        return None, "EVALUATOR_TEMPLATE_ERROR"

    # --- Self-correction loop for evaluator generation ---
    current_evaluator_content = context_from_evolution_loop.get("parent_evaluator_code")
    current_error_feedback = context_from_evolution_loop.get("previous_error_feedback")
    current_evaluator_for_db = "N/A (Initial Call)" # To track the prompt that led to the final version

    for attempt in range(max_correction_attempts + 1 if enable_self_correction else 1): # +1 for initial attempt
        if attempt > 0: # Only log for actual correction attempts
            logger.info("Evaluator self-correction attempt %s/%s..." % (attempt, max_correction_attempts))
            # Update context for self-correction
            context_from_evolution_loop["parent_evaluator_code"] = current_evaluator_content
            context_from_evolution_loop["previous_error_feedback"] = current_error_feedback

        # --- Orchestration Loop (if hierarchical) or single generation ---
        if hierarchical_enabled and current_delegation_depth == 0:
            accumulated_delegation_results: List[Dict[str, Any]] = []
            
            for iteration in range(max_iterations):
                logger.info("Evaluator Generation/Orchestration iteration %s/%s" % (iteration + 1, max_iterations))
                
                template_render_context = {
                    "problem": problem_config,
                    "prompt_ctx": prompt_context_parts, # Problem context for evaluator
                    "parent_evaluator_code": context_from_evolution_loop.get("parent_evaluator_code"),
                    "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback"),
                    "llm": llm_config,
                    "previous_delegation_results": accumulated_delegation_results,
                    "best_programs_from_generator": context_from_evolution_loop.get("best_programs_from_generator", []) # Programs to challenge
                }
                
                evaluator_template_to_use = evaluator_templates.get(current_evaluator_template_key)
                if evaluator_template_to_use is None:
                    logger.error("Evaluator template '%s' became unavailable mid-loop." % current_evaluator_template_key)
                    return None, "EVALUATOR_TEMPLATE_ERROR_UNEXPECTED"

            try:
                prompt_text = evaluator_template_to_use.render(template_render_context)
            except Exception as e:
                logger.error("Error rendering evaluator template '%s': %s" % (current_evaluator_template_key,e))
                return None, "EVALUATOR_RENDERING_ERROR: %s" % str(e)

            llm_response, actual_evaluator_used = await _get_llm_completion(prompt_text, client)
            current_evaluator_for_db = actual_evaluator_used # Log the prompt that led to this version

            if not llm_response:
                current_evaluator_content = None
                break # Exit self-correction loop
            
            delegation_requests = llm_tools.extract_delegation_requests(llm_response)
            if delegation_requests and current_delegation_depth < max_depth:
                logger.info("Orchestrator requested %s sub-tasks. Processing..." % len(delegation_requests))
                delegation_requests = delegation_requests[:llm_config.get('max_sub_tasks_per_step', 3)]
                
                sub_task_futures = []
                for req in delegation_requests:
                    logger.info("Delegating sub-task ID: %s, Desc: %s..." % (req['sub_task_id'], req['description'][:50]))
                    sub_task_futures.append(
                        llm_tools.generate_delegated_task_content( # Updated function call
                            req,
                            client,
                            current_prompt_string="", # Not directly relevant for evaluator subtasks, pass empty
                            prompt_templates_ref=evaluator_templates, # Pass evaluator templates
                            problem_config_ref=problem_config,
                            prompt_parts_ref=prompt_context_parts,
                            llm_config_ref=llm_config,
                            get_llm_completion_func=_get_llm_completion,
                            extract_content_func=_extract_evaluator_code # Use evaluator code extraction
                        )
                    )
                
                sub_task_results_tuples = await asyncio.gather(*sub_task_futures)
                
                new_delegation_results = []
                for req, (sub_content, _sub_prompt) in zip(delegation_requests, sub_task_results_tuples):
                    new_delegation_results.append({
                        "sub_task_id": req["sub_task_id"],
                        "description": req["description"],
                        "expected_signature": req["expected_signature"], # This might need to be adapted for evaluators
                        "content": sub_content if sub_content else "# SUB-TASK FAILED OR RETURNED NO CONTENT"
                    })
                accumulated_delegation_results.extend(new_delegation_results)

                if iteration == max_iterations - 1:
                    logger.warning("Max delegation iterations reached. LLM must integrate or provide final evaluator.")
                continue
            
            # If no delegation, assume it's final evaluator code
            logger.info("No further delegation. Attempting to extract final evaluator code.")
            generated_evaluator = _extract_evaluator_code(llm_response)
            if generated_evaluator:
                logger.log(VERBOSE_LEVEL_NUM, "Final Evaluator Output (Iteration: %s):\nPrompt (first 300 chars):\n%s...\nFinal Evaluator:\n%s",
                           iteration + 1, actual_evaluator_used[:300], generated_evaluator)
                current_evaluator_content = generated_evaluator # Update for self-correction loop
                current_evaluator_for_db = actual_evaluator_used
                break # Exit orchestration loop, found an evaluator
            else:
                logger.warning("LLM did not delegate, nor provided extractable evaluator code. Output: %s" % llm_response[:200])
                if iteration == max_iterations - 1:
                    logger.error("Max iterations reached in main loop without extractable final evaluator.")
                    current_evaluator_content = None # Indicate failure
                    current_evaluator_for_db = actual_evaluator_used
                    break
                current_error_feedback = "LLM did not provide extractable evaluator code or delegation. Please try again." # Feedback for next iteration
        else: # If orchestration loop finishes without breaking (max_iterations reached)
            logger.error("Evaluator Generation/Orchestration loop finished (max_iterations) without producing final evaluator.")
            current_evaluator_content = None # Indicate failure
            # current_evaluator_for_db is already set from last iteration
 
        if current_evaluator_content is not None: # If an evaluator was successfully generated (or corrected)
            return current_evaluator_content, current_evaluator_for_db
        # If the loop finishes and no valid content was generated after all attempts
        elif attempt == max_correction_attempts:
            logger.error("Evaluator generation failed after %s attempts." % (attempt + 1))
            # Add debug log for the content that failed
            if current_evaluator_content:
                logger.debug(f"Failed evaluator content: {current_evaluator_content[:500]}...")
            return None, current_evaluator_for_db # Return None to indicate failure
    
    # This return is for the case where the loop completes without a successful generation
    # and without reaching max_correction_attempts (e.g., if enable_self_correction is False)
    return None, current_evaluator_for_db

def get_last_prompt():
    return LAST_USED_PROMPT

# --- Main function for standalone testing (Async) ---
async def _test_main():
    global problem_config, prompt_context_parts, llm_config
    logger = get_logger("EvaluatorLLMTest")
    logger.info("Evaluator LLM Generator Standalone Test Started.")
    
    if not llm_config or not problem_config or not prompt_context_parts: 
        logger.error("Initial configurations not loaded properly. Aborting test.")
        try: 
            load_all_configs()
            _initialize_evaluator_templates()
            logger.info("Configs and templates reloaded for test.")
        except Exception as e:
            logger.error("Failed to reload configs for test: %s" % e)
            return

    required_templates_ok = True
    if not evaluator_templates.get("default"):
        logger.error("Default evaluator template not loaded.")
        required_templates_ok = False
    if llm_config.get('enable_hierarchical_generation') and \
       (not evaluator_templates.get("hierarchical_orchestrator") or not evaluator_templates.get("delegated_subtask")):
        logger.error("Hierarchical/delegated evaluator templates not loaded.")
        required_templates_ok = False

    if not required_templates_ok:
        logger.error("One or more critical evaluator templates are not loaded. Aborting test.")
        return

    logger.info("LLM Config (first 200 chars): %s..." % str(llm_config)[:200])

    example_context = {
        'parent_evaluator_code': "def evaluate_program(p, pc, mc): return {'score': 0.5, 'is_valid': True}",
        'previous_error_feedback': None,
        'best_programs_from_generator': [] # Example: pass some programs here
    }
    
    async with httpx.AsyncClient() as client:
        logger.info("\n--- Test: Generating evaluator (hierarchical: %s) ---" % 
                    (llm_config.get('enable_hierarchical_generation')))
        evaluator_code, prompt_used = await generate_evaluator_logic(example_context, client, current_delegation_depth=0)
        
        print("\nFinal Prompt Used (for last LLM call in the chain):")
        print(prompt_used)
        print("\nGenerated Evaluator Code (final integrated output, if any):")
        print(evaluator_code if evaluator_code else "No final evaluator code generated/extracted.")

    logger.info("Evaluator LLM Generator Standalone Test Finished.")

if __name__ == "__main__":
    print("NOTE: If running standalone, ensure 'current_problem_directory' in config.yaml is set correctly.")
    print("Ensure LLM provider, model, and API keys (if needed) are configured.")
    print("Ensure templates/ directory contains all required .jinja files.")
    
    asyncio.run(_test_main())