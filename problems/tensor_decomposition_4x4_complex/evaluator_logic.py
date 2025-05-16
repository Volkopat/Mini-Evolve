import math

# Helper for complex number comparison
def are_complex_numbers_close(c1, c2, epsilon=1e-9):
    return abs(c1 - c2) < epsilon

# Helper for standard 4x4 complex matrix multiplication
def standard_complex_matrix_multiply(A, B):
    C = [[complex(0,0) for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k] * B[k][j]
    return C

def evaluate_program(program_module, problem_config, main_config):
    """
    Evaluates the generated program for the tensor decomposition problem.

    Args:
        program_module: The dynamically imported module containing the 'decompose_tensor' function.
        problem_config: Dictionary loaded from the problem's problem_config.yaml.
        main_config: Dictionary loaded from the main config/config.yaml.

    Returns:
        A dictionary containing evaluation metrics.
        Example: {'score': 0.0, 'is_valid': False, 'error_message': 'Not implemented', 
                  'num_complex_multiplications': -1, 'accuracy': 0.0}
    """
    
    is_valid = False
    score = 0.0
    error_message = None
    num_complex_multiplications = -1 # Use the LLM's claimed number if valid
    accuracy_passed = False

    # Define test matrices (complex numbers)
    test_cases = [
        {
            "A": [[complex(i+j, i-j) for j in range(4)] for i in range(4)],
            "B": [[complex(i*j % 5 - 2, i+j % 3 -1) for j in range(4)] for i in range(4)],
        },
        {
            "A": [[complex(1 if i==j else 0, 0) for j in range(4)] for i in range(4)], # Identity
            "B": [[complex(i*2-j, j*2-i) for j in range(4)] for i in range(4)],
        }
    ]

    if not hasattr(program_module, 'decompose_tensor'):
        error_message = "Function 'decompose_tensor' not found in program_module."
    else:
        try:
            mock_tensor_input_for_discovery = None 
            
            # Expects: (callable_algorithm_function, claimed_multiplications_count)
            returned_value = program_module.decompose_tensor(mock_tensor_input_for_discovery)
            
            if not isinstance(returned_value, tuple) or len(returned_value) != 2:
                error_message = "'decompose_tensor' did not return a tuple of length 2."
            else:
                algorithm_representation, claimed_mults = returned_value
                num_complex_multiplications = claimed_mults

                if not callable(algorithm_representation):
                    error_message = "The first element returned by 'decompose_tensor' (expected algorithm function) was not callable."
                elif not isinstance(claimed_mults, int) or claimed_mults < 0:
                    error_message = "The second element returned by 'decompose_tensor' (claimed_mults) is invalid: %s" % claimed_mults
                else:
                    all_tests_passed = True
                    for i, case in enumerate(test_cases):
                        A, B = case["A"], case["B"]
                        C_expected = standard_complex_matrix_multiply(A, B)
                        
                        try:
                            C_generated_by_llm_method = algorithm_representation(A,B)

                            if not isinstance(C_generated_by_llm_method, list) or \
                               len(C_generated_by_llm_method) != 4 or \
                               any(not isinstance(row, list) or len(row) != 4 for row in C_generated_by_llm_method) or \
                               any(not isinstance(el, complex) for row in C_generated_by_llm_method for el in row):
                                error_message = "Test %s: Generated matrix C has incorrect type, dimensions or element types." % (i+1)
                                all_tests_passed = False
                                break
                            
                            current_test_passed = True
                            for r_idx in range(4):
                                for c_idx in range(4):
                                    if not are_complex_numbers_close(C_generated_by_llm_method[r_idx][c_idx], C_expected[r_idx][c_idx]):
                                        error_message = "Test %s: Mismatch at C[%s][%s]. Expected %s, Got %s" % \
                                                        (i+1, r_idx, c_idx, C_expected[r_idx][c_idx], C_generated_by_llm_method[r_idx][c_idx])
                                        all_tests_passed = False
                                        current_test_passed = False
                                        break
                                if not current_test_passed:
                                    break
                            if not current_test_passed:
                                break
                        except Exception as e_apply:
                            error_message = "Test %s: Error calling the algorithm function provided by 'decompose_tensor': %s" % (i+1, str(e_apply))
                            all_tests_passed = False
                            break
                    
                    if all_tests_passed:
                        is_valid = True
                        accuracy_passed = True
                        if claimed_mults > 0:
                            score = 100.0 / claimed_mults 
                            if claimed_mults < 49:
                                 score *= (49.0 / claimed_mults) 
                        else: 
                            score = 0.01 
                        error_message = None 
                    elif not error_message: 
                         error_message = "One or more test cases failed."

        except Exception as e:
            error_message = "Error during evaluation of decompose_tensor: %s" % str(e)

    return {
        'score': score,
        'is_valid': is_valid,
        'error_message': error_message,
        'num_complex_multiplications': num_complex_multiplications,
        'accuracy_passed': accuracy_passed,
    }

if __name__ == '__main__':
    class MockProgramModuleCorrect:
        def multiply_matrices_correctly(self, A, B):
            # This is a simple correct matrix multiplication
            C = [[complex(0,0) for _ in range(4)] for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        C[i][j] += A[i][k] * B[k][j]
            return C

        def decompose_tensor(self, tensor_input):
            # Returns the 'algorithm' and claimed multiplications
            return self.multiply_matrices_correctly, 64 # Standard mults

    class MockProgramModuleFewerMults:
        def multiply_matrices_fewer(self, A, B):
            # Simulates a more efficient (but still correct) multiplication
            C = [[complex(0,0) for _ in range(4)] for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        C[i][j] += A[i][k] * B[k][j]
            return C # Still does standard for testing, but claims fewer
        
        def decompose_tensor(self, tensor_input):
            return self.multiply_matrices_fewer, 48 # Claiming better than Strassen

    class MockProgramModuleIncorrect:
        def multiply_matrices_incorrectly(self, A, B):
            C = [[complex(0,0) for _ in range(4)] for _ in range(4)]
            for i in range(4):
                for j in range(4):
                    # Intentionally wrong calculation
                    C[i][j] = A[i][j] + B[i][j] 
            return C

        def decompose_tensor(self, tensor_input):
            return self.multiply_matrices_incorrectly, 60
    
    class MockProgramModuleBadReturn:
        def decompose_tensor(self, tensor_input):
            return "not a callable", "not an int"

    mock_problem_config = {}
    mock_main_config = {}

    print("Testing evaluator_logic.py for tensor_decomposition_4x4_complex...")

    print("\n--- Test: Correct Algorithm (standard multiplications) ---")
    results_correct = evaluate_program(MockProgramModuleCorrect(), mock_problem_config, mock_main_config)
    for k, v in results_correct.items(): print("%s: %s" % (k,v))
    assert results_correct['is_valid'] is True
    assert results_correct['score'] > 0
    assert results_correct['num_complex_multiplications'] == 64

    print("\n--- Test: Correct Algorithm (claims fewer multiplications) ---")
    results_fewer = evaluate_program(MockProgramModuleFewerMults(), mock_problem_config, mock_main_config)
    for k, v in results_fewer.items(): print("%s: %s" % (k,v))
    assert results_fewer['is_valid'] is True
    assert results_fewer['score'] > (100.0/48.0) # Should get bonus
    assert results_fewer['num_complex_multiplications'] == 48

    print("\n--- Test: Incorrect Algorithm ---")
    results_incorrect = evaluate_program(MockProgramModuleIncorrect(), mock_problem_config, mock_main_config)
    for k, v in results_incorrect.items(): print("%s: %s" % (k,v))
    assert results_incorrect['is_valid'] is False
    assert results_incorrect['score'] == 0.0

    print("\n--- Test: Bad Return from decompose_tensor ---")
    results_bad_return = evaluate_program(MockProgramModuleBadReturn(), mock_problem_config, mock_main_config)
    for k, v in results_bad_return.items(): print("%s: %s" % (k,v))
    assert results_bad_return['is_valid'] is False

    print("\n--- Test: Module missing decompose_tensor ---")
    class MockProgramModuleMissingFunc: pass
    results_missing_func = evaluate_program(MockProgramModuleMissingFunc(), mock_problem_config, mock_main_config)
    for k, v in results_missing_func.items(): print("%s: %s" % (k,v))
    assert results_missing_func['is_valid'] is False
    assert "not found" in results_missing_func['error_message']

    print("\nAll tests seem to have passed locally if no assertion errors.") 