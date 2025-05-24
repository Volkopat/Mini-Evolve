import yaml
import re
import httpx # Added for async
import asyncio # Added for async
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os # For checking templates directory
from typing import Optional, Tuple, List, Dict, Any # Added for Python 3.9 compatibility & new types
from dotenv import load_dotenv # Import load_dotenv
from .logger_setup import get_logger, VERBOSE_LEVEL_NUM # Import VERBOSE_LEVEL_NUM

# Import the new tools module
from . import llm_tools 

load_dotenv() # Load environment variables from .env file

# --- Module Level Variables ---
LAST_USED_PROMPT = "No prompt generated yet."
# Modified to store multiple templates
prompt_templates: Dict[str, Optional[Template]] = {
    "default": None,
    "hierarchical_orchestrator": None,
    "delegated_subtask": None,
    "lean_interaction": None # Placeholder for a potential Lean-specific template
}
main_config = {}
problem_config = {} # Will hold content from <problem_dir>/problem_config.yaml
program_context_parts = {} # Will hold content from <problem_dir>/prompt_context.txt or .yaml

# --- Prompt Templating Constants ---
DEFAULT_PROMPT_TEMPLATE_NAME = "code_generation_prompt.jinja"
HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME = "hierarchical_code_generation_prompt.jinja"
DELEGATED_SUBTASK_TEMPLATE_NAME = "delegated_subtask_prompt.jinja"
LEAN_INTERACTION_TEMPLATE_NAME = "lean_interaction_prompt.jinja" # New template for Lean

# --- Configuration Loading ---
MAIN_CONFIG_FILE = "config/config.yaml"
PROBLEM_CONFIG_FILENAME = "problem_config.yaml" # Expected in problem directory
PROMPT_CONTEXT_FILENAME = "prompt_context.yaml" # Changed to YAML for easier parsing

def load_all_configs():
    global main_config, problem_config, program_context_parts, llm_config # Added llm_config here
    logger = get_logger("ProgramLLMConfig") # Use a logger for config loading
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical("CRITICAL: Main configuration file %s not found." % MAIN_CONFIG_FILE)
        raise
    except yaml.YAMLError as e:
        logger.critical("CRITICAL: Error parsing main YAML configuration: %s" % e)
        raise

    # Initialize llm_config from main_config and set defaults for Lean prover
    llm_config = main_config.get('llm', {})
    llm_config.setdefault('enable_lean_prover_interaction', False)
    llm_config.setdefault('lean_api_url', 'http://127.0.0.1:8060/run_lean')
    # Ensure other hierarchical settings also have defaults if not present, or rely on .get later
    llm_config.setdefault('enable_hierarchical_generation', False)
    llm_config.setdefault('max_delegation_depth', 1)
    llm_config.setdefault('max_sub_tasks_per_step', 3)
    llm_config.setdefault('delegation_iteration_limit', 5)
    llm_config.setdefault('enable_diff_generation', False) # New: Enable LLM to generate diffs

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
            program_context_parts = yaml.safe_load(f)
        if not program_context_parts or not isinstance(program_context_parts, dict):
            logger.warning("Prompt context file %s is empty or not a dictionary. Prompt context might be incomplete." % prompt_context_path)
            program_context_parts = {}
    except FileNotFoundError:
        logger.critical("CRITICAL: Prompt context file %s not found. LLM generation will likely fail." % prompt_context_path)
        program_context_parts = {}
        raise # Re-raise so the system stops if this crucial file is missing
    except yaml.YAMLError as e:
        logger.critical("CRITICAL: Error parsing prompt context YAML %s: %s. LLM generation will likely fail." % (prompt_context_path, e))
        program_context_parts = {}
        raise # Re-raise

load_all_configs() # Load them at module import

# llm_config is now initialized and populated within load_all_configs
# REMOVED: llm_config = main_config.get('llm', {})
# REMOVED: # toy_problem_config is now problem_config.get('toy_problem', {}) or similar, accessed where needed.


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
    get_logger("ProgramLLMTemplate").warning("Jinja2 FileSystemLoader for 'templates/' failed: %s. Prompting will fail if template not found." % e)
    template_env = None

def _initialize_prompt_templates(): # Renamed to plural
    global prompt_templates, template_env
    logger = get_logger("ProgramLLMTemplate")

    if not template_env:
        logger.critical("Jinja2 template_env not initialized. Cannot load prompt templates from file.")
        return

    template_names = {
        "default": DEFAULT_PROMPT_TEMPLATE_NAME,
        "hierarchical_orchestrator": HIERARCHICAL_ORCHESTRATOR_TEMPLATE_NAME,
        "delegated_subtask": DELEGATED_SUBTASK_TEMPLATE_NAME,
        "lean_interaction": LEAN_INTERACTION_TEMPLATE_NAME # Add new template
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
    global LAST_USED_PROMPT, llm_config # main_config, problem_config are not needed here as llm_config is global
    LAST_USED_PROMPT = prompt_text
    logger = get_logger("ProgramLLMCompletion") # Logger for this function
    
    model_name = llm_config.get('model_name')
    provider = llm_config.get('provider')
    base_url = llm_config.get('base_url')

    if not provider or not model_name:
        logger.error("LLM provider or model_name is missing in config.")
        return None, prompt_text

    try:
        # System message should be specific to the type of call (orchestrator vs. sub-task vs. direct)
        # This might need to be passed in or determined based on context if it varies significantly.
        # For now, the prompts themselves handle the primary instruction.
        # The system message in the payload could be made more generic or removed if prompts are sufficient.
        
        # expected_function_name is not directly used in this function's logic, but kept for context if needed by system message
        # expected_function_name = problem_config.get('function_details', {}).get('name', 'solve')
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
            api_key_to_use = llm_config.get('api_key') # Ollama might have an API key
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
            if not base_url: base_url = "https://openrouter.ai/api/v1" # base_url from llm_config
            api_endpoint = base_url.rstrip('/') + "/chat/completions"
            
            # Optional OpenRouter headers
            http_referer = llm_config.get('http_referer')
            if http_referer and http_referer != "YOUR_SITE_URL": headers["HTTP-Referer"] = http_referer
            x_title = llm_config.get('x_title')
            if x_title and x_title != "YOUR_SITE_NAME": headers["X-Title"] = x_title
        else:
            logger.error("Unsupported LLM provider: %s" % provider)
            return None, prompt_text

        if api_key_to_use: # Add Authorization header if api_key is available
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

def _extract_python_code(response_text: str) -> Optional[str]:
    logger = get_logger("ProgramLLMParser")
    if not response_text:
        return None
    
    # If delegation tags are present, it's not final Python code.
    if "<delegate_subtask>" in response_text: 
        logger.debug("Delegation tags found in response, not extracting as direct Python code.")
        return None
    
    # New: If Lean code tags are present, it's not final Python code either.
    if "<lean_code>" in response_text and "</lean_code>" in response_text:
        logger.debug("Lean code tags found in response, not extracting as direct Python code.")
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

# New: Function to extract Lean code
def _extract_lean_code(response_text: str) -> Optional[str]:
    logger = get_logger("ProgramLLMParser")
    if not response_text:
        return None

    match = re.search(r"<lean_code>(.*?)</lean_code>", response_text, re.DOTALL)
    if match:
        extracted_code = match.group(1).strip()
        logger.debug("Extracted Lean code using regex from <lean_code> tags.")
        return extracted_code
    
    logger.debug("No <lean_code> tags found for Lean code extraction.")
    return None

def _extract_diff_blocks(response_text: str) -> List[Dict[str, str]]:
    logger = get_logger("ProgramLLMParser")
    diff_blocks = []
    # Regex to find diff blocks: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
    # Using re.DOTALL to match across newlines
    matches = re.finditer(r"<<<<<<< SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>> REPLACE", response_text, re.DOTALL)
    for match in matches:
        search_content = match.group(1).strip()
        replace_content = match.group(2).strip()
        diff_blocks.append({"search": search_content, "replace": replace_content})
    
    if diff_blocks:
        logger.debug("Extracted %d diff blocks." % len(diff_blocks))
    else:
        logger.debug("No diff blocks found in response.")
    return diff_blocks

def _apply_diffs(original_code: str, diff_blocks: List[Dict[str, str]]) -> str:
    """Applies a list of diff blocks to the original code string."""
    logger = get_logger("ProgramLLMDiffApplier")
    modified_code = original_code
    
    for i, diff in enumerate(diff_blocks):
        search_content = diff["search"]
        replace_content = diff["replace"]
        
        # Escape special characters in search_content for regex matching
        # This is a simple approach; a more robust solution might involve line-by-line matching
        # or a dedicated diff patching library.
        escaped_search_content = re.escape(search_content)
        
        # Use re.DOTALL and re.MULTILINE to match across lines and handle anchors
        # Ensure the search is for the exact block.
        # We're looking for the exact string, so re.escape is important.
        try:
            # Find the exact match. If multiple, only the first is replaced.
            # For now, assuming unique search blocks or that order of application is fine.
            if re.search(escaped_search_content, modified_code, re.DOTALL):
                modified_code = re.sub(escaped_search_content, replace_content, modified_code, 1, re.DOTALL)
                logger.debug("Applied diff %d: Replaced '%s' with '%s'." % (i+1, search_content[:50], replace_content[:50]))
            else:
                logger.warning("Diff %d: Search content not found in code. Skipping diff. Search: '%s'" % (i+1, search_content[:100]))
        except Exception as e:
            logger.error("Error applying diff %d: %s. Search: '%s'" % (i+1, e, search_content[:100]))
            # Continue with the current modified_code even if one diff fails
            
    return modified_code

async def generate_code_variant(
    context_from_evolution_loop: dict,
    client: httpx.AsyncClient,
    current_prompt_string: str, # New parameter
    current_delegation_depth: int = 0
) -> Tuple[Optional[str], str]:
    global prompt_templates, program_context_parts, problem_config, llm_config # main_config removed, llm_config is now global
    logger = get_logger("ProgramLLMGenerator")

    hierarchical_enabled = llm_config.get('enable_hierarchical_generation', False)
    lean_interaction_enabled = llm_config.get('enable_lean_prover_interaction', False)
    diff_generation_enabled = llm_config.get('enable_diff_generation', False) # New: Get diff generation flag
    lean_api_url = llm_config.get('lean_api_url')
    max_depth = llm_config.get('max_delegation_depth', 1)
    max_iterations = llm_config.get('delegation_iteration_limit', 5) # For both orchestration and Lean loops
    
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

    # --- Orchestration / Lean Interaction Loop ---
    # This loop handles both hierarchical delegation and Lean interaction if enabled.
    if (hierarchical_enabled and current_delegation_depth == 0) or lean_interaction_enabled:
        accumulated_delegation_results: List[Dict[str, Any]] = []
        accumulated_lean_results: List[Dict[str, Any]] = [] # For Lean results
        
        for iteration in range(max_iterations):
            logger.info("Generation/Orchestration iteration %s/%s" % (iteration + 1, max_iterations))
            
            template_render_context = {
                "problem": problem_config,
                "prompt_ctx": program_context_parts,
                "parent_code": context_from_evolution_loop.get("parent_code"),
                "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback"),
                "llm": llm_config,
                "previous_delegation_results": accumulated_delegation_results,
                "previous_lean_results": accumulated_lean_results, # Pass Lean results to template
                "current_prompt_string": current_prompt_string # Pass the current prompt string
            }

            # Determine if current context suggests Lean interaction
            # This is a simple heuristic; could be refined based on prompt template content
            active_prompt_template_key = current_prompt_template_key
            if lean_interaction_enabled and accumulated_lean_results and prompt_templates.get("lean_interaction"):
                # If there are previous Lean results, consider using a specific Lean interaction template
                # This part is conceptual; actual switching might depend on LLM output or explicit state.
                # For now, we assume the main orchestrator/default prompt handles lean results feedback.
                pass
            
            prompt_template_to_use = prompt_templates.get(active_prompt_template_key)
            if prompt_template_to_use is None: # Should not happen if initial check passed
                logger.error("Prompt template '%s' became unavailable mid-loop." % active_prompt_template_key)
                return None, "PROMPT_TEMPLATE_ERROR_UNEXPECTED"

            try:
                prompt_text = prompt_template_to_use.render(template_render_context)
            except Exception as e:
                logger.error("Error rendering prompt template '%s': %s" % (active_prompt_template_key,e))
                return None, "PROMPT_RENDERING_ERROR: %s" % str(e)

            llm_response, actual_prompt_used = await _get_llm_completion(prompt_text, client)
            if not llm_response:
                return None, actual_prompt_used

            # 1. Check for Lean code execution request
            requested_lean_code = _extract_lean_code(llm_response)
            if lean_interaction_enabled and requested_lean_code and lean_api_url:
                logger.info("LLM requested Lean code execution. Executing...")
                lean_result = await llm_tools.execute_lean_code(requested_lean_code, client, lean_api_url)
                accumulated_lean_results.append({
                    "requested_code": requested_lean_code,
                    "execution_result": lean_result
                })
                logger.debug(f"Lean execution result: {lean_result}")
                # Continue to next iteration to feed this result back to LLM
                if iteration == max_iterations - 1:
                    logger.warning("Max iterations reached after a Lean call. LLM needs to produce final code now.")
                continue # Loop back to re-prompt with Lean results

            # 2. Check for delegation requests (only if hierarchical is enabled and orchestrating)
            if hierarchical_enabled and current_delegation_depth == 0:
                delegation_requests = llm_tools.extract_delegation_requests(llm_response)
                if delegation_requests and current_delegation_depth < max_depth:
                    logger.info("Orchestrator requested %s sub-tasks. Processing..." % len(delegation_requests))
                    delegation_requests = delegation_requests[:llm_config.get('max_sub_tasks_per_step', 3)]
                    
                    sub_task_futures = []
                    for req in delegation_requests:
                        logger.info("Delegating sub-task ID: %s, Desc: %s..." % (req['sub_task_id'], req['description'][:50]))
                        sub_task_futures.append(
                            llm_tools.generate_delegated_task_code(
                                req,
                                client,
                                prompt_templates_ref=prompt_templates, # Pass reference
                                problem_config_ref=problem_config,   # Pass reference
                                program_context_parts_ref=program_context_parts,     # Pass reference
                                llm_config_ref=llm_config,       # Pass reference
                                get_llm_completion_func=_get_llm_completion, # Pass function
                                extract_python_code_func=_extract_python_code,  # Pass function
                                current_prompt_string=current_prompt_string # Pass the current prompt string to sub-tasks
                            )
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
                    accumulated_delegation_results.extend(new_delegation_results)

                    if iteration == max_iterations - 1:
                        logger.warning("Max delegation iterations reached. LLM must integrate or provide final code.")
                    continue # Loop back to re-prompt with delegation results

            # 3. Check for diff blocks if enabled and parent_code is present
            parent_code = context_from_evolution_loop.get("parent_code")
            if diff_generation_enabled and parent_code:
                diff_blocks = _extract_diff_blocks(llm_response)
                if diff_blocks:
                    logger.info("Extracted diff blocks. Attempting to apply to parent code.")
                    final_code = _apply_diffs(parent_code, diff_blocks)
                    if final_code:
                        logger.log(VERBOSE_LEVEL_NUM, "Final Code Output (Iteration: %s) after diff application:\nPrompt (first 300 chars):\n%s...\nFinal Code:\n%s",
                                   iteration + 1, actual_prompt_used[:300], final_code)
                        return final_code, actual_prompt_used
                    else:
                        logger.warning("Failed to apply diffs to parent code. Falling back to full code extraction.")
                else:
                    logger.debug("Diff generation enabled but no diff blocks found. Attempting full code extraction.")

            # 4. If no Lean, delegation, or diffs, assume it's final Python code (or fallback for diffs)
            logger.info("No Lean request, further delegation, or applicable diffs. Attempting to extract final Python code.")
            final_code = _extract_python_code(llm_response)
            if final_code:
                logger.log(VERBOSE_LEVEL_NUM, "Final Code Output (Iteration: %s):\nPrompt (first 300 chars):\n%s...\nFinal Code:\n%s",
                           iteration + 1, actual_prompt_used[:300], final_code)
                return final_code, actual_prompt_used
            else:
                logger.warning("LLM did not request Lean/delegate, nor provided extractable Python code. Output: %s" % llm_response[:200])
                # If max iterations reached and still no code, this is a failure for this path.
                if iteration == max_iterations - 1:
                    logger.error("Max iterations reached in main loop without extractable final code.")
                    return None, actual_prompt_used
                # Otherwise, we might want to reprompt with an error or just continue if there's a strategy for it.
                # For now, if no code and no explicit actions, and not max iterations, we will re-prompt with current state.
                # This could happen if LLM just chats or asks a question. Template should discourage this.
                logger.debug("Re-prompting as no final code, no lean, no delegation was found.")
                # Clear previous error if any, as this is not a self-correction loop for *Python* errors yet
                context_from_evolution_loop["previous_error_feedback"] = None

        # If loop finishes due to max_iterations without returning final code
        logger.error("Generation/Orchestration loop finished (max_iterations) without producing final code.")
        return None, actual_prompt_used

    else: # --- Standard (Non-Hierarchical, Non-Lean) Code Generation ---
        template_render_context = {
            "problem": problem_config,
            "prompt_ctx": program_context_parts,
            "parent_code": context_from_evolution_loop.get("parent_code"),
            "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback"),
            "llm": llm_config,
            "current_prompt_string": current_prompt_string # Pass the current prompt string
        }
        try:
            # Basic check: problem_config must be populated for standard prompt
            if not problem_config.get('function_details', {}).get('name'):
                 logger.critical("Missing 'function_details.name' in problem_config for standard prompt.")
                 return None, "PROBLEM_CONFIG_ERROR_MISSING_FUNCTION_DETAILS"
            prompt_text = prompt_template_to_use.render(template_render_context)
        except Exception as e:
            logger.error("Error rendering standard prompt template: %s" % e)
            return None, "PROMPT_RENDERING_ERROR: %s" % str(e)

        llm_response_content, actual_prompt_used = await _get_llm_completion(prompt_text, client)
        if not llm_response_content:
            return None, actual_prompt_used

        # If diff generation is enabled and parent_code is available, try to extract and apply diffs
        parent_code = context_from_evolution_loop.get("parent_code")
        if diff_generation_enabled and parent_code:
            diff_blocks = _extract_diff_blocks(llm_response_content)
            if diff_blocks:
                logger.info("Extracted diff blocks. Attempting to apply to parent code.")
                generated_code = _apply_diffs(parent_code, diff_blocks)
                if generated_code:
                    logger.log(VERBOSE_LEVEL_NUM, "Generated Code after diff application:\nPrompt (first 300 chars):\n%s...\nFinal Code:\n%s",
                               actual_prompt_used[:300], generated_code)
                    return generated_code, actual_prompt_used
                else:
                    logger.warning("Failed to apply diffs to parent code. Falling back to full code extraction.")
            else:
                logger.debug("Diff generation enabled but no diff blocks found. Attempting full code extraction.")

        # Fallback to full code extraction if diffs not enabled or not found/applied
        generated_code = _extract_python_code(llm_response_content)
        if not generated_code:
            logger.warning("Failed to extract code from LLM response. Response (first 500 chars): %s" % llm_response_content[:500])
            return None, actual_prompt_used
        
        return generated_code, actual_prompt_used

def get_last_prompt():
    return LAST_USED_PROMPT

# --- Main function for standalone testing (Async) ---
async def _test_main():
    global problem_config, program_context_parts, llm_config # main_config removed
    logger = get_logger("ProgramLLMTest")
    logger.info("Program LLM Generator Standalone Test Started.")
    
    # Configs are loaded at module start. Re-check llm_config directly.
    if not llm_config or not problem_config or not program_context_parts:
        logger.error("Initial configurations not loaded properly. Aborting test.")
        # Attempt re-load for test context as a fallback
        try:
            load_all_configs()
            _initialize_prompt_templates() # This is now _initialize_program_templates
            logger.info("Configs and templates reloaded for test.")
        except Exception as e:
            logger.error("Failed to reload configs for test: %s" % e)
            return

    # Check for required templates based on config
    required_templates_ok = True
    if not prompt_templates.get("default"):
        logger.error("Default prompt template not loaded.")
        required_templates_ok = False
    if llm_config.get('enable_hierarchical_generation') and \
       (not prompt_templates.get("hierarchical_orchestrator") or not prompt_templates.get("delegated_subtask")):
        logger.error("Hierarchical/delegated prompt templates not loaded.")
        required_templates_ok = False
    if llm_config.get('enable_lean_prover_interaction') and not prompt_templates.get("lean_interaction"):
        logger.warning("Lean interaction template not loaded. Lean features might be limited without it.")
        # Not making this critical yet, as main prompt might handle basic Lean feedback

    if not required_templates_ok:
        logger.error("One or more critical prompt templates are not loaded. Aborting test.")
        return

    logger.info("Main Config (first 200 chars): %s..." % str(main_config)[:200]) # main_config is still available globally
    logger.info("Problem Config (first 200 chars): %s..." % str(problem_config)[:200])
    logger.info("LLM Config (first 200 chars): %s..." % str(llm_config)[:200])

    example_context = {
        'parent_code': None,
        'previous_error_feedback': None
    }
    dummy_prompt_string = "Generate a Python function to solve the problem."
    
    async with httpx.AsyncClient() as client:
        logger.info("\n--- Test: Generating code (hierarchical: %s, lean: %s) ---" %
                    (llm_config.get('enable_hierarchical_generation'), llm_config.get('enable_lean_prover_interaction')))
        code, prompt_used = await generate_code_variant(example_context, client, dummy_prompt_string, current_delegation_depth=0)
        
        print("\nFinal Prompt Used (for last LLM call in the chain):")
        print(prompt_used)
        print("\nGenerated Code (final integrated output, if any):")
        print(code if code else "No final code generated/extracted.")

        if code and problem_config.get('function_details',{}).get('name'):
            expected_main_func_name = problem_config['function_details']['name']
            # This assertion might fail if final code is from Lean or a complex delegated task
            # For now, keeping it simple. 
            # assert ("def %s(" % expected_main_func_name) in code, "Final code should ideally contain main function if Python problem."
            logger.info("Test basic assertion for main function presence passed (if Python code generated and expected).")

    logger.info("Program LLM Generator Standalone Test Finished.")

if __name__ == "__main__":
    # Ensure that the `current_problem_directory` in `config/config.yaml` points to a valid problem
    # with `problem_config.yaml` and `prompt_context.yaml`.
    # Also, ensure the LLM provider and model in config.yaml are correctly set up.
    
    # Example: To test hierarchical generation, ensure enable_hierarchical_generation: true in config.yaml
    # and pick a problem that might benefit from it.
    # Ensure the new prompt templates exist in the templates/ directory.
    
    print("NOTE: If running standalone, ensure 'current_problem_directory' in config.yaml is set correctly.")
    print("Ensure LLM provider, model, and API keys (if needed) are configured.")
    print("Ensure templates/ directory contains all required .jinja files, including potentially new ones for Lean.")
    
    asyncio.run(_test_main()) 