import httpx
import asyncio
import re
from typing import Optional, Tuple, List, Dict, Any, Callable
from jinja2 import Template # For type hinting if templates are passed

# Import necessary functions and variables from llm_generator
# This assumes llm_generator provides these.
# If direct import causes circular dependency issues, they might need to be passed as arguments.
# from .llm_generator import _get_llm_completion, _extract_python_code, prompt_templates, problem_config, prompt_parts, llm_config

from .logger_setup import get_logger, VERBOSE_LEVEL_NUM


def extract_delegation_requests(llm_output: str) -> List[Dict[str, str]]:
    logger = get_logger("LLMToolsParser") # Use a distinct logger name
    requests = []
    try:
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
                 logger.warning("Failed to parse inner content of a <delegate_subtask> block: %s. Content: '%s'" % (e_inner, content[:100]))
                 continue 
    except Exception as e:
        logger.error("Error parsing delegation requests: %s. LLM output (first 200 chars): '%s'" % (e, llm_output[:200]))
    return requests

async def generate_delegated_task_code(
    sub_task_request: Dict[str, str], 
    client: httpx.AsyncClient,
    # Dependencies passed as arguments to avoid circular imports
    prompt_templates_ref: Dict[str, Optional[Template]],
    problem_config_ref: Dict[str, Any],
    prompt_parts_ref: Dict[str, Any],
    llm_config_ref: Dict[str, Any],
    get_llm_completion_func: Callable[..., Tuple[Optional[str], str]],
    extract_python_code_func: Callable[[str], Optional[str]]
) -> Tuple[Optional[str], str]:
    logger = get_logger("LLMToolsSubTask") # Use a distinct logger name

    sub_task_prompt_template = prompt_templates_ref.get("delegated_subtask")
    if not sub_task_prompt_template:
        logger.error("Delegated sub-task prompt template not found.")
        return None, "SUB_TASK_PROMPT_ERROR"

    template_render_context = {
        "sub_task": sub_task_request,
        "problem": problem_config_ref, # Use passed problem_config_ref
        "prompt_ctx": prompt_parts_ref,  # Use passed prompt_parts_ref
        "llm": llm_config_ref           # Use passed llm_config_ref
    }
    try:
        prompt_text = sub_task_prompt_template.render(template_render_context)
    except Exception as e:
        logger.error("Error rendering sub-task prompt: %s" % e)
        return None, "SUB_TASK_PROMPT_RENDERING_ERROR: %s" % str(e)

    llm_response_content, actual_prompt_used = await get_llm_completion_func(prompt_text, client)
    if not llm_response_content:
        return None, actual_prompt_used

    generated_code = extract_python_code_func(llm_response_content) 
    if not generated_code:
        logger.warning("Failed to extract code for sub-task ID %s. Response: %s" % (sub_task_request.get("sub_task_id"), llm_response_content[:200]))
        return None, actual_prompt_used

    logger.log(VERBOSE_LEVEL_NUM, "Delegated Sub-task (ID: %s) - Generated Code:\\nPrompt (first 300 chars):\\n%s...\\nSub-task Code:\\n%s", 
               sub_task_request.get("sub_task_id", "N/A"), actual_prompt_used[:300], generated_code)

    sub_task_func_name = sub_task_request.get("expected_signature", "def FAKE(").split("(")[0].split("def ")[-1].strip()
    if not ("def %s(" % sub_task_func_name in generated_code):
        logger.warning("Sub-task %s generated code does not seem to define expected function %s." % (sub_task_request.get("sub_task_id"), sub_task_func_name))

    return generated_code, actual_prompt_used

# Placeholder for Lean execution function to be added later
async def execute_lean_code(lean_code: str, client: httpx.AsyncClient, lean_api_url: str) -> Dict[str, Any]:
    logger = get_logger("LLMToolsLean")
    payload = {"code": lean_code}
    try:
        response = await client.post(lean_api_url, json=payload, timeout=60.0) # Increased timeout for Lean
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error calling Lean API: {e} - Response: {e.response.text[:500]}")
        return {"error": "Lean API HTTP error", "details": e.response.text[:500], "status_code": e.response.status_code}
    except httpx.RequestError as e:
        logger.error(f"Request error calling Lean API: {e}")
        return {"error": "Lean API request error", "details": str(e)}
    except Exception as e:
        logger.error(f"Generic error calling Lean API: {e}")
        return {"error": "Lean API generic error", "details": str(e)} 