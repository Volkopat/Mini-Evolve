# Evaluator logic for direct matrix multiplication problem

import random

def evaluate_program(program_module, problem_config, main_config):
    """
    Evaluates the generated program for the direct matrix multiplication problem.

    Args:
        program_module: The dynamically imported module containing the 'solve' function.
        problem_config: Dictionary loaded from problem_config.yaml for this problem.
        main_config: Dictionary loaded from the main config/config.yaml.

    Returns:
        A dictionary containing evaluation metrics.
        Example: {'score': 1.0, 'is_valid': True, 'error_message': None, 'test_cases_passed': 20, 'total_test_cases': 20}
    """
    
    params = problem_config.get('problem_specific_parameters', {})
    num_test_cases = params.get('num_test_cases_per_eval', 10)
    min_dim = params.get('matrix_min_dim_size', 2)
    max_dim = params.get('matrix_max_dim_size', 5)
    min_val = params.get('matrix_element_min', -10)
    max_val = params.get('matrix_element_max', 10)

    passed_count = 0
    error_message = None
    is_valid_program = True # Assume valid unless a test fails badly or function is missing

    if not hasattr(program_module, 'solve'):
        return {
            'score': 0.0,
            'is_valid': False,
            'error_message': "Function 'solve' not found in program_module.",
            'test_cases_passed': 0,
            'total_test_cases': num_test_cases
        }

    for i in range(num_test_cases):
        try:
            # Generate random matrix dimensions
            rows_a = random.randint(min_dim, max_dim)
            cols_a_rows_b = random.randint(min_dim, max_dim) # Common dimension
            cols_b = random.randint(min_dim, max_dim)

            matrix_a = [[random.randint(min_val, max_val) for _ in range(cols_a_rows_b)] for _ in range(rows_a)]
            matrix_b = [[random.randint(min_val, max_val) for _ in range(cols_b)] for _ in range(cols_a_rows_b)]
            
            # Calculate expected result (simple, direct way for correctness check)
            expected_result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
            for r in range(rows_a):
                for c in range(cols_b):
                    for k_common in range(cols_a_rows_b):
                        expected_result[r][c] += matrix_a[r][k_common] * matrix_b[k_common][c]

            # Test the generated function
            # print(f"Test Case {i+1}: A={matrix_a}, B={matrix_b}") # For debugging
            actual_result = program_module.solve(matrix_a, matrix_b)
            # print(f"Actual: {actual_result}, Expected: {expected_result}") # For debugging

            if actual_result == expected_result:
                passed_count += 1
            else:
                # Optional: Could add more detailed error messages about mismatches
                # For now, just failing the test case is enough
                error_message = error_message or "Output mismatch on test case %s" % (i+1) # Keep first error
                # is_valid_program = False # A single mismatch might not make the program structure invalid, just its logic
                # If we want to penalize any failure harshly, uncomment line above and break

        except Exception as e:
            error_message = error_message or "Runtime error during test case %s: %s" % (i+1, str(e))
            is_valid_program = False # Runtime error makes it invalid for further tests
            break # Stop testing if a runtime error occurs

    score = 0.0
    if num_test_cases > 0:
        score = float(passed_count) / num_test_cases
    
    # The program is considered structurally valid if it ran all tests or failed gracefully.
    # Logical validity (full score) depends on passing all tests.
    is_valid_for_db = is_valid_program and (error_message is None or passed_count > 0) 

    return {
        'score': score,
        'is_valid': is_valid_for_db, # is_valid here means good enough to be added to DB
        'error_message': error_message,
        'test_cases_passed': passed_count,
        'total_test_cases': num_test_cases
    }

# Example usage (for testing this evaluator logic directly)
if __name__ == '__main__':
    class MockSuccessfulProgramModule:
        def solve(self, matrix_a, matrix_b):
            # Correct implementation for testing
            if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]: return []
            rows_a, cols_a = len(matrix_a), len(matrix_a[0])
            rows_b, cols_b = len(matrix_b), len(matrix_b[0])
            if cols_a != rows_b: return []
            res = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
            for r in range(rows_a): 
                for c in range(cols_b):
                    for k in range(cols_a): res[r][c] += matrix_a[r][k] * matrix_b[k][c]
            return res

    class MockFlawedProgramModule:
        def solve(self, matrix_a, matrix_b):
            # Flawed: always returns a fixed incorrect matrix or makes mistakes
            return [[1,1],[1,1]] 
            
    class MockErrorProgramModule:
        def solve(self, matrix_a, matrix_b):
            raise ValueError("Intentional error in solve")

    mock_problem_conf = {
        'problem_specific_parameters': {
            'num_test_cases_per_eval': 5,
            'matrix_min_dim_size': 1,
            'matrix_max_dim_size': 3,
            'matrix_element_min': 1,
            'matrix_element_max': 5
        },
        'function_details': {'name': 'solve'}
    }
    mock_main_conf = {}

    print("--- Testing Successful Module ---")
    results_ok = evaluate_program(MockSuccessfulProgramModule(), mock_problem_conf, mock_main_conf)
    print(results_ok)

    print("\n--- Testing Flawed Module ---")
    results_flawed = evaluate_program(MockFlawedProgramModule(), mock_problem_conf, mock_main_conf)
    print(results_flawed)
    
    print("\n--- Testing Error Module ---")
    results_error = evaluate_program(MockErrorProgramModule(), mock_problem_conf, mock_main_conf)
    print(results_error)

    print("\n--- Testing Module with Missing Function ---")
    results_missing = evaluate_program(object(), mock_problem_conf, mock_main_conf) # object() has no solve
    print(results_missing) 