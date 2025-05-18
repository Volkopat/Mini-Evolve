import yaml
import re
import httpx # Added for async
import asyncio # Added for async
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os # For checking templates directory
from typing import Optional, Tuple, List, Dict, Any # Added for Python 3.9 compatibility & new types
from dotenv import load_dotenv # Import load_dotenv
from .logger_setup import get_logger, VERBOSE_LEVEL_NUM # Import VERBOSE_LEVEL_NUM
import xml.etree.ElementTree as ET # For parsing delegation tags

load_dotenv() # Load environment variables from .env file

# --- Module Level Variables ---
LAST_USED_PROMPT = "No prompt generated yet."
# Modified to store multiple templates
prompt_templates: Dict[str, Optional[Template]] = {
    "default": None,
    "hierarchical_orchestrator": None,
    "delegated_subtask": None
}
main_config = {}
problem_config = {} # Will hold content from <problem_dir>/problem_config.yaml
prompt_parts = {} # Will hold content from <problem_dir>/prompt_context.txt or .yaml

# --- Prompt Templating Constants ---
DEFAULT_PROMPT_TEMPLATE_NAME = "code_generation_prompt.jinja"
HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME = "hierarchical_code_generation_prompt.jinja"
DELEGATED_SUBTASK_TEMPLATE_NAME = "delegated_subtask_prompt.jinja"

# --- Configuration Loading ---
MAIN_CONFIG_FILE = "config/config.yaml"
PROBLEM_CONFIG_FILENAME = "problem_config.yaml" # Expected in problem directory
PROMPT_CONTEXT_FILENAME = "prompt_context.yaml" # Changed to YAML for easier parsing

def load_all_configs():
    global main_config, problem_config, prompt_parts
    logger = get_logger("LLMGeneratorConfig") # Use a logger for config loading
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("CRITICAL: Main configuration file %s not found." % MAIN_CONFIG_FILE)
        raise
    except yaml.YAMLError as e:
        logger.critical("CRITICAL: Error parsing main YAML configuration: %s" % e)
        raise

    current_problem_dir = main_config.get('current_problem_directory')
    if not current_problem_dir:
        logger.critical("CRITICAL: current_problem_directory not set in %s" % MAIN_CONFIG_FILE)
        raise ValueError("current_problem_directory not set in %s" % MAIN_CONFIG_FILE) # Raise to stop if critical

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
            prompt_parts = yaml.safe_load(f)
        if not prompt_parts or not isinstance(prompt_parts, dict):
            logger.warning("Prompt context file %s is empty or not a dictionary. Prompt context might be incomplete." % prompt_context_path)
            prompt_parts = {}
    except FileNotFoundError:
        logger.critical("CRITICAL: Prompt context file %s not found. LLM generation will likely fail." % prompt_context_path)
        prompt_parts = {} 
        raise # Re-raise so the system stops if this crucial file is missing
    except yaml.YAMLError as e:
        logger.critical("CRITICAL: Error parsing prompt context YAML %s: %s. LLM generation will likely fail." % (prompt_context_path, e))
        prompt_parts = {}
        raise # Re-raise

load_all_configs() # Load them at module import

# References to specific llm_config parts should now use main_config
llm_config = main_config.get('llm', {})
# toy_problem_config is now problem_config.get('toy_problem', {}) or similar, accessed where needed.


# Setup Jinja2 environment
template_env: Optional[Environment] = None # Type hint for template_env
try:
    if not os.path.exists("templates"):
        os.makedirs("templates", exist_ok=True)
        
    template_env = Environment(
        loader=FileSystemLoader("templates/"), 
        autoescape=select_autoescape(['html', 'xml', 'jinja'])
    )
except Exception as e:
    # Use logger for warnings/errors
    get_logger("LLMGeneratorTemplate").warning("Jinja2 FileSystemLoader for 'templates/' failed: %s. Prompting will fail if template not found." % e)
    template_env = None

def _initialize_prompt_templates(): # Renamed to plural
    global prompt_templates, template_env
    logger = get_logger("LLMGeneratorTemplate")

    if not template_env:
        logger.critical("Jinja2 template_env not initialized. Cannot load prompt templates from file.")
        return

    template_names = {
        "default": DEFAULT_PROMPT_TEMPLATE_NAME,
        "hierarchical_orchestrator": HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME,
        "delegated_subtask": DELEGATED_SUBTASK_TEMPLATE_NAME
    }

    for key, name in template_names.items():
        try:
            prompt_templates[key] = template_env.get_template(name)
            logger.info("Successfully loaded prompt template: %s as %s" % (name, key))
        except Exception as e:
            logger.critical("Failed to load prompt template '%s' (for %s): %s." % (name, key, e))
            prompt_templates[key] = None # Ensure it's None if loading failed

_initialize_prompt_templates()

async def _get_llm_completion(prompt_text: str, client: httpx.AsyncClient) -> Tuple[Optional[str], str]: # For Python 3.9
    global LAST_USED_PROMPT, main_config, problem_config
    LAST_USED_PROMPT = prompt_text 
    logger = get_logger("LLMCompletion") # Logger for this function
    
    # Ensure llm_config is up-to-date if main_config was reloaded (e.g., in tests)
    current_llm_config = main_config.get('llm', {})
    model_name = current_llm_config.get('model_name')
    provider = current_llm_config.get('provider')
    base_url = current_llm_config.get('base_url')

    if not provider or not model_name:
        logger.error("LLM provider or model_name is missing in main_config.")
        return None, prompt_text

    try:
        # System message should be specific to the type of call (orchestrator vs. sub-task vs. direct)
        # This might need to be passed in or determined based on context if it varies significantly.
        # For now, the prompts themselves handle the primary instruction.
        # The system message in the payload could be made more generic or removed if prompts are sufficient.
        
        expected_function_name = problem_config.get('function_details', {}).get('name', 'solve') 
        # Generic system message, as prompts are now very detailed
        system_message_content = ("You are a helpful Python coding assistant. "
                                  "Follow the user's instructions carefully and precisely. "
                                  "Output only the requested information (code or delegation requests).")


        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": prompt_text}
        ]
        
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": current_llm_config.get('temperature', 0.7),
            "max_tokens": current_llm_config.get('max_tokens', 2048),
            "stream": False 
        }
        
        headers = {"Content-Type": "application/json"}
        api_key_to_use = None

        if provider == "ollama_local":
            if not base_url:
                logger.error("base_url is required for ollama_local provider.")
                return None, prompt_text
            api_key_to_use = current_llm_config.get('api_key') # Ollama might have an API key
            api_endpoint = base_url.rstrip('/')
            if not api_endpoint.endswith("/v1/chat/completions"):
                 api_endpoint = os.path.join(api_endpoint, "v1/chat/completions")


        elif provider == "openrouter":
            api_key_env_var = current_llm_config.get('api_key_env_var')
            if not api_key_env_var:
                logger.error("api_key_env_var is required for openrouter provider in config.")
                return None, prompt_text
            api_key_to_use = os.getenv(api_key_env_var)
            if not api_key_to_use:
                logger.error("Could not retrieve API key from environment variable: %s" % api_key_env_var)
                return None, prompt_text
            if not base_url: base_url = "https://openrouter.ai/api/v1"
            api_endpoint = base_url.rstrip('/') + "/chat/completions"
            
            # Optional OpenRouter headers
            http_referer = current_llm_config.get('http_referer')
            if http_referer and http_referer != "YOUR_SITE_URL": headers["HTTP-Referer"] = http_referer
            x_title = current_llm_config.get('x_title')
            if x_title and x_title != "YOUR_SITE_NAME": headers["X-Title"] = x_title
        else:
            logger.error("Unsupported LLM provider: %s" % provider)
            return None, prompt_text

        if api_key_to_use: # Add Authorization header if api_key is available
            headers["Authorization"] = "Bearer %s" % api_key_to_use
        
        request_timeout = float(current_llm_config.get('timeout_seconds', 300.0))
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

def _extract_delegation_requests(llm_output: str) -> List[Dict[str, str]]:
    requests = []
    try:
        # Ensure the string is treated as a single block for parsing
        # and that there's a root element if it's partial XML.
        # This is a simplified parser; more robust XML parsing might be needed if LLM output is noisy.
        # A simple string search might be more robust if LLM struggles with perfect XML.
        
        # Using regex to find all <delegate_subtask> blocks
        # This is more robust to malformed XML or surrounding text than ET.parse
        matches = re.finditer(r"<delegate_subtask>(.*?)</delegate_subtask>", llm_output, re.DOTALL)
        for match in matches:
            content = match.group(1)
            try:
                desc_match = re.search(r"<description>(.*?)</description>", content, re.DOTALL)
                sig_match = re.search(r"<expected_signature>(.*?)</expected_signature>", content, re.DOTALL)
                id_match = re.search(r"<sub_task_id>(.*?)</sub_task_id>", content, re.DOTALL)

                if desc_match and sig_match and id_match:
                    requests.append({
                        "description": desc_match.group(1).strip(),
                        "expected_signature": sig_match.group(1).strip(),
                        "sub_task_id": id_match.group(1).strip()
                    })
            except Exception as e_inner:
                 get_logger("LLMGeneratorParser").warning("Failed to parse inner content of a <delegate_subtask> block: %s. Content: '%s'" % (e_inner, content[:100]))
                 continue # Skip this malformed block
    except Exception as e:
        get_logger("LLMGeneratorParser").error("Error parsing delegation requests: %s. LLM output (first 200 chars): '%s'" % (e, llm_output[:200]))
    return requests

def _extract_python_code(response_text: str) -> Optional[str]:
    logger = get_logger("LLMGeneratorParser")
    if not response_text:
        return None
    
    # If delegation tags are present, it's not final code.
    if "<delegate_subtask>" in response_text: 
        logger.debug("Delegation tags found in response, not extracting as direct Python code.")
        return None

    text_to_process = response_text.strip()

    # Regex to find code within ```python ... ``` or just ``` ... ```
    # It handles optional 'python' and surrounding whitespace around the code block.
    match = re.search(r"^```(?:python)?\s*\n(.*?)\n```$", text_to_process, re.DOTALL | re.MULTILINE)
    
    if match:
        extracted_code = match.group(1).strip()
        logger.debug("Extracted code using regex from markdown fence.")
        return extracted_code
    
    # If no markdown fences are found, assume the entire cleaned response is the code.
    # This handles cases where the LLM directly outputs code without fences.
    # Before returning directly, remove any potential leftover XML-like thought/process tags
    # that weren't delegation requests.
    cleaned_fallback = re.sub(r"<([^>]+)>.*?<\/\1>", "", text_to_process, flags=re.DOTALL | re.IGNORECASE)
    cleaned_fallback = re.sub(r"<[^>]+\/>", "", cleaned_fallback, flags=re.DOTALL | re.IGNORECASE) # Handle self-closing tags
    cleaned_fallback = cleaned_fallback.strip()

    if not cleaned_fallback: # If stripping everything leaves nothing
        logger.warning("No code could be extracted. Response was likely empty or only non-code elements after cleaning.")
        return None

    # Heuristic: If the fallback still contains what looks like partial/unterminated markdown, log a warning.
    if cleaned_fallback.startswith("```") or cleaned_fallback.endswith("```"):
        logger.warning("Fallback code extraction still contains markdown-like fences. This might indicate an unusual LLM output format.")
    else:
        logger.debug("Extracted code using fallback (assuming entire response is code after cleaning).")
    
    return cleaned_fallback


async def generate_code_variant(
    context_from_evolution_loop: dict, 
    client: httpx.AsyncClient,
    current_delegation_depth: int = 0 # New parameter
) -> Tuple[Optional[str], str]:
    global prompt_templates, prompt_parts, problem_config, main_config, llm_config # Ensure llm_config is global or passed
    logger = get_logger("LLMGenerator")

    hierarchical_enabled = llm_config.get('enable_hierarchical_generation', False)
    max_depth = llm_config.get('max_delegation_depth', 1)
    max_iterations = llm_config.get('delegation_iteration_limit', 5)
    
    current_prompt_template_key = "default"
    if hierarchical_enabled and current_delegation_depth == 0 : # Top-level call for hierarchical
        current_prompt_template_key = "hierarchical_orchestrator"
    elif hierarchical_enabled and current_delegation_depth > 0: # This is a sub-task call
        # This case is handled by _generate_delegated_task_code, generate_code_variant is for orchestrator or direct.
        # However, if generate_code_variant was called directly for a subtask, it would need a different template.
        # For simplicity, sub-tasks are called via a dedicated helper.
        pass


    prompt_template_to_use = prompt_templates.get(current_prompt_template_key)
    if prompt_template_to_use is None:
        logger.critical("Prompt template '%s' is not available. Cannot generate code." % current_prompt_template_key)
        return None, "PROMPT_TEMPLATE_ERROR"

    # --- Orchestration Loop (if hierarchical_enabled and this is the orchestrator call) ---
    if hierarchical_enabled and current_delegation_depth == 0:
        accumulated_delegation_results: List[Dict[str, Any]] = []
        
        for iteration in range(max_iterations):
            logger.info("Orchestration iteration %s/%s" % (iteration + 1, max_iterations))
            
            template_render_context = {
                "problem": problem_config,
                "prompt_ctx": prompt_parts,
                "parent_code": context_from_evolution_loop.get("parent_code"),
                "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback"),
                "llm": llm_config, # Pass llm_config for template to access max_sub_tasks_per_step
                "previous_delegation_results": accumulated_delegation_results # Results from previous iteration
            }
            try:
                prompt_text = prompt_template_to_use.render(template_render_context)
            except Exception as e:
                logger.error("Error rendering orchestrator prompt template: %s" % e)
                return None, "PROMPT_RENDERING_ERROR: %s" % str(e)

            orchestrator_response, actual_prompt_used = await _get_llm_completion(prompt_text, client)
            if not orchestrator_response:
                return None, actual_prompt_used # Error logged in _get_llm_completion

            delegation_requests = _extract_delegation_requests(orchestrator_response)

            if delegation_requests and current_delegation_depth < max_depth:
                logger.info("Orchestrator requested %s sub-tasks. Processing..." % len(delegation_requests))
                # Limit number of sub-tasks processed in one step
                delegation_requests = delegation_requests[:llm_config.get('max_sub_tasks_per_step', 3)]
                
                sub_task_futures = []
                for req in delegation_requests:
                    logger.info("Delegating sub-task ID: %s, Desc: %s..." % (req['sub_task_id'], req['description'][:50]))
                    # IMPORTANT: Sub-tasks should not themselves delegate further in this simple model (depth+1)
                    # or if they can, current_delegation_depth+1 must be passed.
                    # For now, assume sub-tasks are direct.
                    sub_task_futures.append(
                        _generate_delegated_task_code(req, client, current_delegation_depth + 1)
                    )
                
                sub_task_results_tuples = await asyncio.gather(*sub_task_futures)
                
                new_delegation_results = []
                for req, (sub_code, _sub_prompt) in zip(delegation_requests, sub_task_results_tuples):
                    new_delegation_results.append({
                        "sub_task_id": req["sub_task_id"],
                        "description": req["description"],
                        "expected_signature": req["expected_signature"],
                        "code": sub_code if sub_code else "# SUB-TASK FAILED OR RETURNED NO CODE"
                    })
                accumulated_delegation_results.extend(new_delegation_results) # Add new results for next iteration

                if iteration == max_iterations - 1:
                    logger.warning("Max delegation iterations reached. Attempting to force integration or fail.")
                    # The next loop will use the final accumulated_delegation_results and the LLM
                    # MUST produce final code or the generation fails. Could add specific instruction for this.
                    # For now, the default prompt handles providing results back.

            else: # No delegation requests, or max depth reached for delegation
                logger.info("Orchestrator provided direct code or no further delegation. Extracting final code.")
                final_code = _extract_python_code(orchestrator_response)
                if final_code:
                    logger.log(VERBOSE_LEVEL_NUM, "Orchestrator - Direct Code Output (Iteration: %s):\nPrompt (first 300 chars):\n%s...\nFinal Code:\n%s", 
                               iteration + 1, actual_prompt_used[:300], final_code)
                    return final_code, actual_prompt_used
                else:
                    logger.warning("Orchestrator did not delegate but also did not provide extractable final code. Output: %s" % orchestrator_response[:200])
                    return None, actual_prompt_used # Failed to get final code
        
        # If loop finishes due to max_iterations without returning final code (should have been forced in last iteration)
        logger.error("Orchestration loop finished (max_iterations) without producing final code.")
        return None, actual_prompt_used

    else: # --- Standard (Non-Hierarchical) or Sub-Task Code Generation ---
        # This path is for normal generation or if hierarchical is disabled.
        # Delegated tasks are handled by _generate_delegated_task_code directly.
        
        template_render_context = {
            "problem": problem_config,
            "prompt_ctx": prompt_parts,
            "parent_code": context_from_evolution_loop.get("parent_code"),
            "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback"),
            "llm": llm_config # Pass llm_config for template access
        }
        try:
            if not problem_config.get('function_details', {}).get('name'): # Basic check
                 logger.critical("Missing 'function_details.name' in problem_config for standard prompt.")
                 return None, "PROBLEM_CONFIG_ERROR_MISSING_FUNCTION_DETAILS"
            prompt_text = prompt_template_to_use.render(template_render_context)
        except Exception as e:
            logger.error("Error rendering standard prompt template: %s" % e)
            return None, "PROMPT_RENDERING_ERROR: %s" % str(e)

        llm_response_content, actual_prompt_used = await _get_llm_completion(prompt_text, client)
        if not llm_response_content:
            return None, actual_prompt_used

        generated_code = _extract_python_code(llm_response_content)
        if not generated_code:
            logger.warning("Failed to extract code from LLM response. Response (first 500 chars): %s" % llm_response_content[:500])
            return None, actual_prompt_used
        
        # Basic validation (function name check) - might need to be adjusted for sub-tasks
        # expected_function_name = problem_config.get('function_details', {}).get('name')
        # if expected_function_name and "def %s(" % expected_function_name not in generated_code:
        #     logger.warning("Extracted code does not appear to contain the expected function definition '%s'." % expected_function_name)
        #     # return None, actual_prompt_used # This might be too strict if LLM includes helper functions

        return generated_code, actual_prompt_used

async def _generate_delegated_task_code(
    sub_task_request: Dict[str, str], 
    client: httpx.AsyncClient,
    delegation_depth: int # Current depth for this sub-task
) -> Tuple[Optional[str], str]:
    global prompt_templates, prompt_parts, problem_config, llm_config
    logger = get_logger("LLMGeneratorSubTask")

    # Here, we would check if delegation_depth > llm_config.get('max_delegation_depth')
    # and prevent further delegation if this sub-task itself wanted to delegate.
    # For this version, sub-tasks are assumed to be direct code generators.

    sub_task_prompt_template = prompt_templates.get("delegated_subtask")
    if not sub_task_prompt_template:
        logger.error("Delegated sub-task prompt template not found.")
        return None, "SUB_TASK_PROMPT_ERROR"

    template_render_context = {
        "sub_task": sub_task_request, # Contains description, expected_signature, sub_task_id
        "problem": problem_config,   # Main problem context (e.g. function_details.name for overall goal)
        "prompt_ctx": prompt_parts,  # Main problem constraints
        "llm": llm_config
    }
    try:
        prompt_text = sub_task_prompt_template.render(template_render_context)
    except Exception as e:
        logger.error("Error rendering sub-task prompt: %s" % e)
        return None, "SUB_TASK_PROMPT_RENDERING_ERROR: %s" % str(e)

    llm_response_content, actual_prompt_used = await _get_llm_completion(prompt_text, client)
    if not llm_response_content:
        return None, actual_prompt_used # Error already logged

    # For sub-tasks, we expect direct code. No further delegation parsing here.
    generated_code = _extract_python_code(llm_response_content) 
    if not generated_code:
        logger.warning("Failed to extract code for sub-task ID %s. Response: %s" % (sub_task_request.get("sub_task_id"), llm_response_content[:200]))
        return None, actual_prompt_used
    
    logger.log(VERBOSE_LEVEL_NUM, "Delegated Sub-task (ID: %s) - Generated Code:\nPrompt (first 300 chars):\n%s...\nSub-task Code:\n%s", 
               sub_task_request.get("sub_task_id", "N/A"), actual_prompt_used[:300], generated_code)

    # Basic check: does the generated code define the function expected by sub_task.expected_signature?
    sub_task_func_name = sub_task_request.get("expected_signature", "def FAKE(").split("(")[0].split("def ")[-1].strip()
    if not ("def %s(" % sub_task_func_name in generated_code):
        logger.warning("Sub-task %s generated code does not seem to define expected function %s." % (sub_task_request.get("sub_task_id"), sub_task_func_name))
        # Not returning None here, orchestrator might still find it useful or try to correct it.

    return generated_code, actual_prompt_used


def get_last_prompt():
    return LAST_USED_PROMPT

# --- Main function for standalone testing (Async) ---
async def _test_main():
    global problem_config, prompt_parts, main_config 
    logger = get_logger("LLMGeneratorTest")
    logger.info("LLM Generator Standalone Test Started.")
    
    if not main_config or not problem_config or not prompt_parts: # Check if initial load failed
        logger.error("Initial configurations not loaded. Aborting test. Check for CRITICAL errors above.")
        try: # Attempt re-load for test context
            load_all_configs()
            _initialize_prompt_templates() # Also re-init templates
            logger.info("Configs and templates reloaded for test.")
            # Update llm_config as it might have changed if main_config was reloaded
            global llm_config
            llm_config = main_config.get('llm', {})

        except Exception as e:
            logger.error("Failed to reload configs for test: %s" % e)
            return

    if not prompt_templates["default"] or \
       (llm_config.get('enable_hierarchical_generation') and \
        (not prompt_templates["hierarchical_orchestrator"] or not prompt_templates["delegated_subtask"])):
        logger.error("One or more required prompt templates are not loaded. Aborting test.")
        return

    logger.info("Main Config (first 200 chars): %s..." % str(main_config)[:200])
    logger.info("Problem Config (first 200 chars): %s..." % str(problem_config)[:200])
    logger.info("LLM Config (first 200 chars): %s..." % str(llm_config)[:200])


    # Example context (can be adapted)
    example_context = {
        'parent_code': None,
        'previous_error_feedback': None
    }
    
    async with httpx.AsyncClient() as client:
        logger.info("\n--- Test: Generating code (hierarchical enabled: %s) ---" % llm_config.get('enable_hierarchical_generation'))
        # Pass current_delegation_depth = 0 for a top-level call
        code, prompt_used = await generate_code_variant(example_context, client, current_delegation_depth=0)
        
        print("\nFinal Prompt Used (for last LLM call in the chain):")
        print(prompt_used)
        print("\nGenerated Code (final integrated output, if any):")
        print(code if code else "No final code generated/extracted.")

        if code and problem_config.get('function_details',{}).get('name'):
            expected_main_func_name = problem_config['function_details']['name']
            assert ("def %s(" % expected_main_func_name) in code, "Final code should contain main function."
            logger.info("Test basic assertion for main function presence passed (if code generated).")

    logger.info("LLM Generator Standalone Test Finished.")

if __name__ == "__main__":
    # Ensure that the `current_problem_directory` in `config/config.yaml` points to a valid problem
    # with `problem_config.yaml` and `prompt_context.yaml`.
    # Also, ensure the LLM provider and model in config.yaml are correctly set up.
    
    # Example: To test hierarchical generation, ensure enable_hierarchical_generation: true in config.yaml
    # and pick a problem that might benefit from it.
    # Ensure the new prompt templates exist in the templates/ directory.
    
    print("NOTE: If running standalone, ensure 'current_problem_directory' in config.yaml is set correctly.")
    print("Ensure LLM provider, model, and API keys (if needed) are configured.")
    print("Ensure templates/ directory contains all required .jinja files.")
    
    asyncio.run(_test_main()) 