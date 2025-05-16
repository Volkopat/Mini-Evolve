import yaml
import re
import httpx # Added for async
import asyncio # Added for async
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
import os # For checking templates directory
from typing import Optional, Tuple # Added for Python 3.9 compatibility
from dotenv import load_dotenv # Import load_dotenv
from .logger_setup import get_logger

load_dotenv() # Load environment variables from .env file

# --- Module Level Variables ---
LAST_USED_PROMPT = "No prompt generated yet."
prompt_template: Optional[Template] = None # For Python 3.9
main_config = {}
problem_config = {} # Will hold content from <problem_dir>/problem_config.yaml
prompt_parts = {} # Will hold content from <problem_dir>/prompt_context.txt or .yaml

# --- Prompt Templating Constants ---
DEFAULT_PROMPT_TEMPLATE_NAME = "code_generation_prompt.jinja" # This will be the main generic template

# --- Configuration Loading ---
MAIN_CONFIG_FILE = "config/config.yaml"
PROBLEM_CONFIG_FILENAME = "problem_config.yaml" # Expected in problem directory
PROMPT_CONTEXT_FILENAME = "prompt_context.yaml" # Changed to YAML for easier parsing

def load_all_configs():
    global main_config, problem_config, prompt_parts
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError("CRITICAL: Main configuration file %s not found." % MAIN_CONFIG_FILE)
    except yaml.YAMLError as e:
        raise yaml.YAMLError("CRITICAL: Error parsing main YAML configuration: %s" % e)

    current_problem_dir = main_config.get('current_problem_directory')
    if not current_problem_dir:
        raise ValueError("CRITICAL: current_problem_directory not set in %s" % MAIN_CONFIG_FILE)

    problem_config_path = os.path.join(current_problem_dir, PROBLEM_CONFIG_FILENAME)
    try:
        with open(problem_config_path, 'r') as f:
            problem_config = yaml.safe_load(f)
        if not problem_config or not isinstance(problem_config, dict): # ensure it's a dictionary
            print("Warning: Problem config file %s is empty or not a dictionary. Problem-specific params might be missing." % problem_config_path)
            problem_config = {} # Default to empty dict if malformed
    except FileNotFoundError:
        print("Warning: Problem config file %s not found. Using defaults if any, or LLM may lack problem-specific params." % problem_config_path)
        problem_config = {} # Default to empty dict if not found
    except yaml.YAMLError as e:
        print("Warning: Error parsing problem YAML configuration %s: %s. Using defaults." % (problem_config_path, e))
        problem_config = {}

    prompt_context_path = os.path.join(current_problem_dir, PROMPT_CONTEXT_FILENAME)
    try:
        with open(prompt_context_path, 'r') as f:
            prompt_parts = yaml.safe_load(f)
        if not prompt_parts or not isinstance(prompt_parts, dict):
            print("Warning: Prompt context file %s is empty or not a dictionary. Prompt context might be incomplete." % prompt_context_path)
            prompt_parts = {}
    except FileNotFoundError:
        print("CRITICAL: Prompt context file %s not found. LLM generation will likely fail." % prompt_context_path)
        prompt_parts = {} # This is critical, so prompts will be bad
        raise # Re-raise so the system stops if this crucial file is missing
    except yaml.YAMLError as e:
        print("CRITICAL: Error parsing prompt context YAML %s: %s. LLM generation will likely fail." % (prompt_context_path, e))
        prompt_parts = {}
        raise # Re-raise

load_all_configs() # Load them at module import

# References to specific llm_config parts should now use main_config
llm_config = main_config.get('llm', {})
# toy_problem_config is now problem_config.get('toy_problem', {}) or similar, accessed where needed.


# Setup Jinja2 environment
template_env = None
try:
    if not os.path.exists("templates"):
        os.makedirs("templates", exist_ok=True)
        
    template_env = Environment(
        loader=FileSystemLoader("templates/"), 
        autoescape=select_autoescape(['html', 'xml', 'jinja'])
    )
except Exception as e:
    print("Warning: Jinja2 FileSystemLoader for 'templates/' failed: %s. Prompting will fail if template not found." % e)
    template_env = None

def _initialize_prompt_template():
    global prompt_template, template_env
    global DEFAULT_PROMPT_TEMPLATE_NAME

    if not template_env:
        print("CRITICAL: Jinja2 template_env not initialized. Cannot load prompt template from file.")
        prompt_template = None 
        return # Exit if no template environment

    # This try-except block should be outside the 'if not template_env' check
    try:
        prompt_template = template_env.get_template(DEFAULT_PROMPT_TEMPLATE_NAME)
    except Exception as e:
        print("CRITICAL: Failed to load main prompt template '%s': %s. LLM generation will fail." % (DEFAULT_PROMPT_TEMPLATE_NAME, e))
        prompt_template = None

_initialize_prompt_template()

async def _get_llm_completion(prompt_text: str, client: httpx.AsyncClient) -> Tuple[Optional[str], str]: # For Python 3.9
    global LAST_USED_PROMPT, main_config, problem_config
    LAST_USED_PROMPT = prompt_text 
    
    current_llm_config = main_config.get('llm', {})
    model_name = current_llm_config.get('model_name')
    provider = current_llm_config.get('provider')
    base_url = current_llm_config.get('base_url')

    if not provider or not model_name:
        print("LLM provider or model_name is missing in main_config.")
        return None, prompt_text

    try:
        if provider == "ollama_local":
            if not base_url:
                print("Error: base_url is required for ollama_local provider.")
                return None, prompt_text
            
            api_key_ollama = current_llm_config.get('api_key') # Still get this for ollama

            # Dynamically create system message with the expected function name
            expected_function_name = problem_config.get('function_details', {}).get('name', 'solve') 
            system_message_content = (
                "You are a Python code generation assistant. "
                "Your sole task is to output ONLY a valid, complete Python code block "
                "for the function named `%s`. "
                "Do NOT include any explanations, comments outside the code, "
                "or any markdown formatting like ```python ... ```."
            ) % expected_function_name

            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": prompt_text}
            ]
            
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": current_llm_config.get('temperature', 0.7),
                "max_tokens": current_llm_config.get('max_tokens', 2048), # Removed a stale comment here previously
                "stream": False 
            }
            
            headers = {
                "Content-Type": "application/json",
            }
            if api_key_ollama: # Only add Authorization header if api_key is provided for Ollama
                headers["Authorization"] = "Bearer %s" % api_key_ollama

            api_endpoint = base_url.rstrip('/') 
            
            if not api_endpoint.endswith("/v1/chat/completions"):
                if api_endpoint.endswith("/v1"):
                    api_endpoint += "/chat/completions"
                elif not "/v1" in api_endpoint: # Ensure this check is correct for all base_url variations
                    api_endpoint += "/v1/chat/completions"

            request_timeout = current_llm_config.get('timeout_seconds', 300.0) 
            response = await client.post(api_endpoint, json=payload, headers=headers, timeout=request_timeout)
            response.raise_for_status() 
            
            json_response = response.json()
            
            if json_response.get("choices") and json_response["choices"][0].get("message"):
                return json_response["choices"][0]["message"].get("content", "").strip(), prompt_text
            else: # This else corresponds to the if json_response.get("choices")
                return None, prompt_text
        elif provider == "openrouter":
            api_key_env_var = current_llm_config.get('api_key_env_var')
            if not api_key_env_var:
                print("Error: api_key_env_var is required for openrouter provider in config.")
                return None, prompt_text
            
            api_key_openrouter = os.getenv(api_key_env_var)
            if not api_key_openrouter:
                print("Error: Could not retrieve API key from environment variable: %s" % api_key_env_var)
                return None, prompt_text

            if not base_url: # Should be set in config, but as a fallback
                base_url = "https://openrouter.ai/api/v1"
            
            api_endpoint = base_url.rstrip('/') + "/chat/completions"

            # System message (can be adapted if OpenRouter models prefer a different style)
            expected_function_name = problem_config.get('function_details', {}).get('name', 'solve')
            system_message_content = (
                "You are a Python code generation assistant. "
                "Your sole task is to output ONLY a valid, complete Python code block "
                "for the function named `%s`. "
                "Do NOT include any explanations, comments outside the code, "
                "or any markdown formatting like ```python ... ```."
            ) % expected_function_name
            
            messages = [
                {"role": "system", "content": system_message_content},
                {"role": "user", "content": prompt_text} # For text-only, simple content string
            ]

            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": current_llm_config.get('temperature', 0.7),
                "max_tokens": current_llm_config.get('max_tokens', 2048), # Ensure this respects model limits
                # stream is not explicitly in user example for OpenRouter, assuming False
            }

            headers = {
                "Authorization": "Bearer %s" % api_key_openrouter,
                "Content-Type": "application/json"
            }
            
            # Optional headers from config
            http_referer = current_llm_config.get('http_referer')
            if http_referer and http_referer != "YOUR_SITE_URL": # Add if defined and not the placeholder
                headers["HTTP-Referer"] = http_referer
            
            x_title = current_llm_config.get('x_title')
            if x_title and x_title != "YOUR_SITE_NAME": # Add if defined and not the placeholder
                headers["X-Title"] = x_title

            request_timeout = current_llm_config.get('timeout_seconds', 300.0)
            response = await client.post(api_endpoint, json=payload, headers=headers, timeout=request_timeout)
            response.raise_for_status()
            
            json_response = response.json()
            
            if json_response.get("choices") and json_response["choices"][0].get("message"):
                return json_response["choices"][0]["message"].get("content", "").strip(), prompt_text
            else:
                return None, prompt_text
        else: # This else corresponds to the if provider == "ollama_local"
            print("Unsupported LLM provider: %s" % provider)
            return None, prompt_text
    
    except httpx.HTTPStatusError as e:
        print("HTTP error calling %s API: %s - Response: %s" % (provider, e, e.response.text))
        return None, prompt_text
    except httpx.RequestError as e:
        print("Request error calling %s API. Details: %s" % (provider, repr(e)))
        # print(f"Request details: Method={e.request.method}, URL={e.request.url}") # Commented out debug line
        return None, prompt_text
    except Exception as e:
        print("Generic error calling %s API. Details: %s" % (provider, repr(e)))
        # import traceback # Commented out debug line
        # print(traceback.format_exc()) # Commented out debug line
        return None, prompt_text

# --- Code Extraction ---
def _extract_python_code(response_text: str) -> str | None:
    if not response_text:
        return None
    
    cleaned_response_text = response_text
    # Remove <think>...</think> and <tag/> style tags
    cleaned_response_text = re.sub(r"<([^>]+)>.*?<\/\1>", "", cleaned_response_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_response_text = re.sub(r"<[^>]+\/>", "", cleaned_response_text, flags=re.DOTALL | re.IGNORECASE) # Handle self-closing tags
    cleaned_response_text = cleaned_response_text.strip()
    if not cleaned_response_text:
        return None

    # Attempt to find Markdown code blocks first
    match = re.search(r"```python\n(.*?)\n```", cleaned_response_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no markdown is found, and our prompt instructs the LLM to ONLY output code,
    # then the cleaned_response_text is assumed to be the code.
    # The check for `startswith("def ")` was too restrictive if imports are present.
    return cleaned_response_text # Return the whole cleaned text


# --- Main Generator Function (Async) ---
async def generate_code_variant(context_from_evolution_loop: dict, client: httpx.AsyncClient) -> tuple[str | None, str]:
    """Generates a code variant using the LLM based on the provided context.

    Args:
        context_from_evolution_loop: A dictionary containing dynamic elements like parent_code.
        client: An httpx.AsyncClient instance.

    Returns:
        A tuple containing (generated_code_string_or_None, prompt_used_str).
    """
    global prompt_template, prompt_parts, problem_config # Ensure problem_config is accessible
    logger = get_logger("LLMGenerator")

    if prompt_template is None:
        logger.critical("Prompt template is not available. Cannot generate code.")
        return None, "PROMPT_TEMPLATE_ERROR_NOT_INITIALIZED"

    # Combine all context parts: problem-specific static parts, and dynamic parts from evolution loop
    template_render_context = {
        "problem": problem_config,
        "prompt_ctx": prompt_parts,
        "parent_code": context_from_evolution_loop.get("parent_code"),
        "previous_error_feedback": context_from_evolution_loop.get("previous_error_feedback")
    }

    try:
        # Ensure problem_config and its nested keys are present before rendering
        if not problem_config.get('function_details', {}).get('name') or \
           not problem_config.get('function_details', {}).get('input_params_string'):
            logger.critical("Missing 'function_details.name' or 'function_details.input_params_string' in problem_config.")
            return None, "PROBLEM_CONFIG_ERROR_MISSING_FUNCTION_DETAILS"
        
        prompt_text = prompt_template.render(template_render_context)
    except Exception as e:
        logger.error("Error rendering prompt template: %s. Context was: %s" % (e, template_render_context))
        return None, "PROMPT_RENDERING_ERROR: %s" % str(e)

    llm_response_content, actual_prompt_used = await _get_llm_completion(prompt_text, client)

    if not llm_response_content:
        # _get_llm_completion would have logged the error
        return None, actual_prompt_used

    generated_code = _extract_python_code(llm_response_content)

    if not generated_code:
        logger.warning("Failed to extract code from LLM response. Response (first 500 chars): %s" % llm_response_content[:500])
        return None, actual_prompt_used
        
    # Get expected function name from problem_config
    expected_function_name = problem_config.get('function_details', {}).get('name')
    if not expected_function_name:
        logger.error("CRITICAL: Function name not found in problem_config.yaml (function_details.name).")
        expected_function_name = "solve" 
        logger.warning("Falling back to expected function name: %s due to missing config." % expected_function_name)

    func_def_pattern = "def %s(" % expected_function_name

    # Check if the expected function definition is present in the extracted code.
    # We use the entire 'generated_code' block as extracted by _extract_python_code,
    # as it might contain necessary imports before the function definition.
    if func_def_pattern not in generated_code:
        logger.warning("Extracted code does not appear to contain the expected function definition '%s'. Code (first 150 chars): %s..." % (func_def_pattern, generated_code[:150]))
        return None, actual_prompt_used

    # If the definition is present, return the whole extracted code block.
    return generated_code, actual_prompt_used

# --- Utility to get last prompt (for debugging/logging) ---
def get_last_prompt():
    return LAST_USED_PROMPT

# --- Main function for standalone testing (Async) ---
async def _test_main():
    # This test function needs to be updated to reflect the modular config loading
    global problem_config, prompt_parts, main_config # Added main_config for completeness

    logger = get_logger("LLMGeneratorTest")
    logger.info("LLM Generator Standalone Test Started.")
    
    # Ensure configs are loaded (they should be at module import, but double-check)
    if not main_config or not problem_config or not prompt_parts:
        logger.error("Configurations not loaded. Aborting test.")
        # Optionally, try to reload them here for the test
        try:
            load_all_configs() # This will use MAIN_CONFIG_FILE and derive others
            logger.info("Configs reloaded.")
        except Exception as e:
            logger.error("Failed to reload configs: %s" % e)
            return # Cannot proceed if configs are missing

    logger.info("Main Config Loaded: %s..." % str(main_config)[:200])
    logger.info("Problem Config Loaded: %s..." % str(problem_config)[:200])
    logger.info("Prompt Parts Loaded: %s..." % str(prompt_parts)[:200])

    # Get expected function name from problem_config
    current_expected_function_name = problem_config.get('function_details', {}).get('name', 'solve') # Get from problem_config
    logger.info("Expected function name for test: %s" % current_expected_function_name)


    # Example context for generating a new function (no parent code)
    generation_context = {
        'parent_code': None, # Or omit if template handles it
        # 'task_description': "Generate a Python function named 'solve' that adds two numbers." # This should come from prompt_parts
    }

    # Example context for modifying an existing function
    modification_context = {
        'parent_code': "def %s(matrix_a, matrix_b):\n    # A very basic placeholder\n    return matrix_a" % current_expected_function_name,
        # 'task_description': "Modify the provided Python function to correctly perform matrix multiplication." # From prompt_parts
    }
    
    async with httpx.AsyncClient() as client:
        print("\n--- Test: Generating NEW function ---")
        # The prompt template should have enough info from prompt_parts for a new function
        code, prompt_g = await generate_code_variant(generation_context, client)
        print("Prompt Used:")
        print(prompt_g)
        print("\nGenerated Code:")
        print(code if code else "No code generated/extracted.")
        if code:
            assert ("def %s(" % current_expected_function_name) in code # Basic check

        print("\n--- Test: Modifying EXISTING function ---")
        m_code, prompt_m = await generate_code_variant(modification_context, client)
        print("Prompt Used:")
        print(prompt_m)
        print("\nModified Code:")
        print(m_code if m_code else "No code generated/extracted.")
        if m_code:
            assert ("def %s(" % current_expected_function_name) in m_code # Basic check
            assert "# A very basic placeholder" not in m_code # Example check for modification

    logger.info("LLM Generator Standalone Test Finished.")

if __name__ == "__main__":
    # This allows running the test main function directly: python3 -m app.llm_generator
    # Ensure that the `current_problem_directory` in `config/config.yaml` points to a valid problem
    # with `problem_config.yaml` and `prompt_context.yaml`.
    
    # Example: Create a dummy `problems/test_problem/prompt_context.yaml` for this test to run
    # if not os.path.exists("problems/test_problem"):
    #     os.makedirs("problems/test_problem")
    # with open("problems/test_problem/prompt_context.yaml", "w") as f:
    #     f.write("""
# problem_details: |
#   Problem Description: Test problem for llm_generator.
#   Constraints: Must be simple.
# function_name: solve_test
# function_signature_info: |
#   The function MUST be named `solve_test`.
#   Output ONLY the Python code for the `solve_test(a, b)` function.
# """)
    # print("NOTE: If running standalone, ensure 'current_problem_directory' in config.yaml is set,")
    # print("and the problem directory contains 'problem_config.yaml' and 'prompt_context.yaml'.")
    # print("E.g., set current_problem_directory: 'problems/matrix_multiplication_direct'")
    
    asyncio.run(_test_main()) 