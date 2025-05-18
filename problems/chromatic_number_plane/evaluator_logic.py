import math
import time
from typing import Any, Dict

# It's unlikely we can truly "evaluate" solutions to open problems like this automatically yet.
# This evaluator will focus on whether the LLM attempts a reasonable exploration
# and produces outputs in the expected format.

def evaluate_program(
    program_module: Any,
    problem_config: Dict[str, Any],
    main_config: Dict[str, Any],
    timeout_seconds: float = 5.0
) -> Dict[str, Any]:
    
    results = {
        'score': 0.0,
        'is_valid': False,
        'error_message': None,
        'execution_output': "",
        'steps_taken': [],
        'custom_metrics': {}
    }
    
    # 1. Security Check (basic)
    # program_code string is not directly available; this check might need to be re-thought
    # or program_code passed to program_module if evaluator needs it
    # For now, this specific check will be less effective as program_module is already loaded.
    # If security checks on raw code are vital, app/evaluator.py should do it before creating program_module,
    # or pass program_code_string to this function.
    # disallowed_keywords = problem_config.get("disallowed_patterns_python", [])
    # for keyword in disallowed_keywords:
    #     if keyword in program_code: # program_code is not defined here anymore
    #         results['error_message'] = f"Disallowed keyword '{keyword}' found."
    #         return results

    # 2. Attempt to execute the program's main function
    target_function_name = problem_config.get("function_details", {}).get("name", "explore_chromatic_number_plane")
    
    # The program_module is already prepared by app/evaluator.py's exec call.
    # We just need to check if the function exists in the module.

    if not hasattr(program_module, target_function_name):
        error_msg = f"Function '{target_function_name}' not found in program_module."
        results['error_message'] = error_msg
        return results

    try:
        explore_func = getattr(program_module, target_function_name)
        
        default_params = {"task": "analyze_known_bounds"} 
        
        start_time = time.time()
        function_result = explore_func(default_params)
        execution_time = time.time() - start_time
        
        results['custom_metrics']["execution_time_seconds"] = execution_time
        results['steps_taken'].append(f"Executed '{target_function_name}' with default params.")
        results['execution_output'] = str(function_result)

        if not isinstance(function_result, dict):
            error_msg = f"Function '{target_function_name}' did not return a dictionary."
            results['error_message'] = error_msg
            return results

        if "description" not in function_result or "bounds_found" not in function_result:
            error_msg = f"Returned dictionary from '{target_function_name}' is missing expected keys like 'description' or 'bounds_found'."
            results['error_message'] = error_msg
            results['score'] = 0.25
            results['is_valid'] = True # Still consider it valid if it ran, but low score
            results['steps_taken'].append("Partial success: ran but output format unexpected.")
            return results
        
        results['score'] = 1.0 
        results['is_valid'] = True
        results['error_message'] = "Program executed and returned a dictionary with expected basic keys." # This is more of a success message
        results['steps_taken'].append(results['error_message'])
        
        results['custom_metrics']["returned_description_length"] = len(function_result.get("description", ""))
        results['custom_metrics']["lean_code_present"] = function_result.get("lean_code_generated") is not None
        results['custom_metrics']["lower_bound_found"] = function_result.get("bounds_found", {}).get("lower")
        results['custom_metrics']["upper_bound_found"] = function_result.get("bounds_found", {}).get("upper")

        return results

    except SyntaxError as e: # This is less likely here as code is already compiled by app/evaluator.py
        error_msg = f"SyntaxError: {e}"
        results['error_message'] = error_msg
        return results
    except Exception as e:
        error_msg = f"Runtime Error during execution of '{target_function_name}': {type(e).__name__}: {e}"
        results['error_message'] = error_msg
        return results 