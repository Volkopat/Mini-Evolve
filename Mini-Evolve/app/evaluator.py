import yaml
import time
import traceback
import os
import importlib.util
import sys
from types import ModuleType

from .logger_setup import get_logger # Use the central logger

MAIN_CONFIG_FILE = "config/config.yaml" # Keep for loading main_config in __main__
EVALUATOR_LOGIC_FILENAME = "evaluator_logic.py"

# Disallowed keywords that are checked before even attempting to load problem-specific logic
# Problem-specific disallowed patterns are in problem_config.yaml and handled by LLM prompt primarily,
# but can also be checked by problem-specific evaluator_logic if needed.
GLOBAL_DISALLOWED_KEYWORDS = ["subprocess", "os."] # Minimal global set for security

def evaluate(program_code_string: str, main_config: dict, problem_config: dict, current_problem_dir: str) -> dict:
    logger = get_logger("Evaluator")
    start_time = time.perf_counter()

    base_results = {
        'score': 0.0,
        'is_valid': False,
        'error_message': None,
        'execution_time_ms': 0.0,
    }

    # 1. Global preliminary checks on the code string
    for keyword in GLOBAL_DISALLOWED_KEYWORDS:
        if keyword in program_code_string:
            base_results['error_message'] = "Global disallowed keyword: %s" % keyword
            base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
            return base_results
    
    # 2. Prepare a sandboxed module for the program_code_string
    program_module_name = "candidate_program"
    program_module = ModuleType(program_module_name)
    
    # Define a restricted environment for exec
    # Allow common builtins, but not dangerous ones like open, eval, exec itself directly from candidate.
    restricted_globals = {
        "__builtins__": {
            "len": len, "range": range, "list": list, "dict": dict, "tuple": tuple,
            "int": int, "float": float, "str": str, "bool": bool, "None": None,
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "print": print, # Allow print for debugging by LLM, not used for formal evaluation
            "isinstance": isinstance, "hasattr": hasattr, "callable": callable,
            # Math module functions can be added here if deemed safe and broadly useful
            # For example: 'math': math (after import math)
            # However, problem-specific modules should be handled by problem itself if allowed.
        },
        # Can pass main_config and problem_config to the executed code if ever needed, but generally not recommended.
    }

    try:
        compiled_code = compile(program_code_string, f'<{program_module_name}>', 'exec')
        # Populate the module's dictionary
        exec(compiled_code, program_module.__dict__)
        # setattr(program_module, '__dict__', program_module_locals) # Not needed if exec populates __dict__ directly

    except SyntaxError as e:
        base_results['error_message'] = "SyntaxError: %s\n%s" % (e, traceback.format_exc(limit=1))
        base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
        return base_results
    except Exception as e: # Catch other errors during exec like NameError, TypeError from top level
        base_results['error_message'] = "Error during program execution/setup: %s\n%s" % (e, traceback.format_exc(limit=1))
        base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
        return base_results

    # 3. Dynamically load and call the problem-specific evaluator_logic
    evaluator_logic_path = os.path.join(current_problem_dir, EVALUATOR_LOGIC_FILENAME)
    if not os.path.exists(evaluator_logic_path):
        base_results['error_message'] = "Evaluator logic file not found: %s" % evaluator_logic_path
        base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
        return base_results

    try:
        spec = importlib.util.spec_from_file_location("problem_evaluator_module", evaluator_logic_path)
        if spec is None or spec.loader is None:
            raise ImportError("Could not create module spec from %s" % evaluator_logic_path)
        
        problem_eval_module = importlib.util.module_from_spec(spec)
        sys.modules["problem_evaluator_module"] = problem_eval_module # Add to sys.modules before exec
        spec.loader.exec_module(problem_eval_module)
        
        if not hasattr(problem_eval_module, 'evaluate_program') or not callable(problem_eval_module.evaluate_program):
            base_results['error_message'] = "'evaluate_program' function not found or not callable in %s" % evaluator_logic_path
            base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
            # Clean up module from sys.modules if it was added
            if "problem_evaluator_module" in sys.modules:
                del sys.modules["problem_evaluator_module"]
            return base_results

        # Call the problem-specific evaluation function
        # It receives the prepared program_module, and the configs
        specific_eval_results = problem_eval_module.evaluate_program(program_module, problem_config, main_config)
        
        # Clean up module from sys.modules to allow fresh loads if called again with different problem
        if "problem_evaluator_module" in sys.modules:
            del sys.modules["problem_evaluator_module"]

        # Merge base results (like total execution time) with specific results
        # Specific results should contain 'score', 'is_valid', 'error_message', and any other problem-specific metrics.
        final_results = base_results.copy() # Start with a copy of base (which has exec time template)
        final_results.update(specific_eval_results) # Override/add fields from specific evaluator
        final_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000 # Recalculate total time
        return final_results
        
    except ImportError as e:
        logger.error("ImportError loading problem-specific evaluator: %s" % e)
        base_results['error_message'] = "ImportError for %s: %s" % (evaluator_logic_path, e)
    except Exception as e:
        logger.error("Exception in problem-specific evaluator '%s': %s\n%s" % (evaluator_logic_path, e, traceback.format_exc(limit=2)))
        base_results['error_message'] = "Error in problem-specific evaluator %s: %s" % (evaluator_logic_path, e)
    
    # Fallback if errors occurred during dynamic loading or execution of problem-specific eval
    base_results['execution_time_ms'] = (time.perf_counter() - start_time) * 1000
    return base_results

def print_eval_results(results: dict):
    logger = get_logger("EvaluatorPrint") # Use logger for output
    logger.info("--- Evaluation Results ---")
    logger.info("  Score: %s" % results.get('score'))
    logger.info("  Is Valid: %s" % results.get('is_valid'))
    if 'error_message' in results and results['error_message']:
        logger.error("  Error: %s" % results.get('error_message'))
    # Print other problem-specific keys that might be present
    for key, value in results.items():
        if key not in ['score', 'is_valid', 'error_message', 'execution_time_ms']:
            logger.info("  %s: %s" % (key.replace('_', ' ').capitalize(), value))
    logger.info("  Time (ms): %.2f" % results.get('execution_time_ms', 0.0))
    logger.info("------------------------")


if __name__ == "__main__":
    main_logger = get_logger("Evaluator_MainTest")
    main_logger.info("==== Evaluator Standalone Test Started ====")

    # 1. Load main configuration to find the current problem directory
    try:
        with open(MAIN_CONFIG_FILE, 'r') as f:
            test_main_config = yaml.safe_load(f)
    except Exception as e:
        main_logger.critical("Could not load main config %s for test: %s" % (MAIN_CONFIG_FILE, e))
        sys.exit(1)

    test_current_problem_dir = test_main_config.get('current_problem_directory')
    if not test_current_problem_dir or not os.path.isdir(test_current_problem_dir):
        main_logger.critical("'current_problem_directory' (%s) not found or not a directory in %s." % (test_current_problem_dir, MAIN_CONFIG_FILE))
        main_logger.info("Please ensure 'current_problem_directory' points to a valid problem (e.g., problems/matrix_multiplication_direct)")
        sys.exit(1)
    main_logger.info("Testing with current_problem_directory: %s" % test_current_problem_dir)

    # 2. Load problem-specific configuration
    problem_config_path = os.path.join(test_current_problem_dir, "problem_config.yaml")
    try:
        with open(problem_config_path, 'r') as f:
            test_problem_config = yaml.safe_load(f)
        if not test_problem_config:
            raise ValueError("Problem config is empty.")
    except Exception as e:
        main_logger.critical("Could not load problem config %s for test: %s" % (problem_config_path, e))
        sys.exit(1)
    main_logger.info("Problem config loaded from %s" % problem_config_path)

    # --- Test Case 1: Correct Matrix Multiplication (assuming current_problem is matrix_multiplication_direct) ---
    if "matrix_multiplication_direct" in test_current_problem_dir:
        main_logger.info("\n--- Test Case: Correct Matrix Multiplication ---")
        code_correct_matrix_mult = """
def solve(matrix_a, matrix_b):
    if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]: return []
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])
    if cols_a != rows_b: return []
    res = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for r_idx in range(rows_a):
        for c_idx in range(cols_b):
            sum_val = 0
            for k_idx in range(cols_a):
                sum_val += matrix_a[r_idx][k_idx] * matrix_b[k_idx][c_idx]
            res[r_idx][c_idx] = sum_val
    return res
"""
        results = evaluate(code_correct_matrix_mult, test_main_config, test_problem_config, test_current_problem_dir)
        print_eval_results(results)
        assert results.get('score', 0.0) > 0.99 # Expect perfect or near-perfect score
        assert results.get('is_valid') is True

        main_logger.info("\n--- Test Case: Flawed Matrix Multiplication (e.g., wrong dimensions in result) ---")
        code_flawed_matrix_mult = """
def solve(matrix_a, matrix_b):
    return [[1,2],[3,4]] # Always returns 2x2, likely wrong
"""
        results_flawed = evaluate(code_flawed_matrix_mult, test_main_config, test_problem_config, test_current_problem_dir)
        print_eval_results(results_flawed)
        assert results_flawed.get('score', 1.0) < 0.9 # Expect lower score
        # is_valid might still be true if it runs, but score shows failure

    # --- Test Case 2: Code with Syntax Error ---
    main_logger.info("\n--- Test Case: Syntax Error ---")
    code_syntax_error = "def solve(a,b) return a+b" # Missing colon
    results_syntax = evaluate(code_syntax_error, test_main_config, test_problem_config, test_current_problem_dir)
    print_eval_results(results_syntax)
    assert results_syntax['is_valid'] is False
    assert "SyntaxError" in results_syntax.get('error_message', "")

    # --- Test Case 3: Code with Global Disallowed Keyword ---
    main_logger.info("\n--- Test Case: Global Disallowed Keyword (subprocess) ---")
    code_disallowed = "import subprocess\ndef solve(a,b): return subprocess.call(['ls'])"
    results_disallowed = evaluate(code_disallowed, test_main_config, test_problem_config, test_current_problem_dir)
    print_eval_results(results_disallowed)
    assert results_disallowed['is_valid'] is False
    assert "Global disallowed keyword: subprocess" in results_disallowed.get('error_message', "")

    # --- Test Case 4: Target function not found in candidate code ---
    main_logger.info("\n--- Test Case: Target function (e.g. 'solve') not defined ---")
    expected_func_name = test_problem_config.get('function_details',{}).get('name', 'solve')
    code_no_func = "def another_func(a,b): return a+b\n# Expected function %s is missing" % expected_func_name
    # This test requires the problem-specific evaluator to check for the function.
    # The generic evaluator prepares the module, problem-specific logic uses it.
    # The check for function existence is now part of problem-specific evaluator_logic.py files.
    # For this test to be fully effective, we assume the problem-specific evaluator correctly reports it.
    results_no_func = evaluate(code_no_func, test_main_config, test_problem_config, test_current_problem_dir)
    print_eval_results(results_no_func)
    # The error message should ideally come from the problem-specific evaluator.
    # For matrix_multiplication_direct, it checks for 'solve'.
    if "matrix_multiplication_direct" in test_current_problem_dir:
         assert "Function '%s' not found" % expected_func_name in results_no_func.get('error_message', "")
    # For other problems, the error might be different if their evaluator doesn't explicitly check this.

    # --- Test Case 5: Problem-specific evaluator logic file missing ---
    main_logger.info("\n--- Test Case: Missing evaluator_logic.py (simulated) ---")
    if os.path.exists(os.path.join(test_current_problem_dir, EVALUATOR_LOGIC_FILENAME)):
        os.rename(os.path.join(test_current_problem_dir, EVALUATOR_LOGIC_FILENAME), os.path.join(test_current_problem_dir, EVALUATOR_LOGIC_FILENAME + ".bak"))
        results_missing_logic = evaluate("def solve(a,b): return a+b", test_main_config, test_problem_config, test_current_problem_dir)
        print_eval_results(results_missing_logic)
        assert "Evaluator logic file not found" in results_missing_logic.get('error_message', "")
        os.rename(os.path.join(test_current_problem_dir, EVALUATOR_LOGIC_FILENAME + ".bak"), os.path.join(test_current_problem_dir, EVALUATOR_LOGIC_FILENAME)) # Restore
    else:
        main_logger.warning("Skipping missing evaluator_logic.py test as file was already not present (or .bak exists).")


    main_logger.info("==== Evaluator Standalone Test Finished ====") 