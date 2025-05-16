import math

def evaluate_program(candidate_module, problem_config, main_config):
    solve_set_cover_func = getattr(candidate_module, problem_config['function_details']['name'])

    test_cases = [
        {
            "universe": {1, 2, 3, 4, 5},
            "subsets": [{1, 2, 5}, {2, 3}, {3, 4}, {4, 5}, {1, 3, 4}],
            "name": "Small Case 1",
            "optimal_known_sets": 2 # e.g., indices 0 and 2 -> {1,2,5} U {3,4}
        },
        {
            "universe": set(range(1, 11)), # 1 to 10
            "subsets": [
                {1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10},
                {1, 4, 7, 10}, {2, 5, 8}, {3, 6, 9},
                {1,2,3,4,5}
            ],
            "name": "Medium Case 1",
            "optimal_known_sets": 3 # e.g., {1,4,7,10}, {2,5,8}, {3,6,9}
        },
        {
            "universe": {'A', 'B', 'C', 'D', 'E', 'F'},
            "subsets": [
                {'A', 'B'}, {'A', 'C'}, {'B', 'D'}, {'C', 'E'}, {'D', 'F'}, {'E', 'F'},
                {'A', 'B', 'C', 'D'}
            ],
            "name": "String Elements Case",
            "optimal_known_sets": 2 # e.g. {'A', 'B', 'C', 'D'} and {'E', 'F'} (indices 6 and 5)
        }
    ]

    total_score = 0
    all_results = []
    overall_success = True

    for i, tc in enumerate(test_cases):
        test_universe = tc["universe"]
        test_subsets = [set(s) for s in tc["subsets"]] # Ensure they are sets

        try:
            chosen_indices = solve_set_cover_func(test_universe.copy(), [s.copy() for s in test_subsets])
            
            if not isinstance(chosen_indices, list) or not all(isinstance(idx, int) for idx in chosen_indices):
                error_message = "Output must be a list of integers (indices)."
                all_results.append({"test_case": tc["name"], "score": 0, "error": error_message, "sets_used": float('inf')})
                overall_success = False
                continue

            # Validate indices
            if any(idx < 0 or idx >= len(test_subsets) for idx in chosen_indices):
                error_message = "Invalid subset index provided."
                all_results.append({"test_case": tc["name"], "score": 0, "error": error_message, "sets_used": float('inf')})
                overall_success = False
                continue
            
            # Check for duplicate indices, though not strictly an error for coverage, it inflates count
            if len(chosen_indices) != len(set(chosen_indices)):
                # We can choose to penalize or just note it. For now, let it pass for coverage check.
                # but it will affect the score naturally due to higher count.
                pass 

            covered_elements = set()
            for idx in chosen_indices:
                covered_elements.update(test_subsets[idx])
            
            if not test_universe.issubset(covered_elements):
                error_message = "The chosen subsets do not cover the entire universe."
                all_results.append({"test_case": tc["name"], "score": 0, "error": error_message, "sets_used": len(chosen_indices)})
                overall_success = False
                continue

            num_sets_used = len(chosen_indices)
            if num_sets_used == 0 and len(test_universe) > 0: # Covered by empty set if universe is empty
                 score = 0 # Cannot cover non-empty universe with 0 sets
                 error_message = "Cannot cover a non-empty universe with zero sets."
                 all_results.append({"test_case": tc["name"], "score": score, "error": error_message, "sets_used": num_sets_used})
                 overall_success = False
            elif num_sets_used == 0 and len(test_universe) == 0:
                score = 1.0 # Perfect score for empty universe and no sets
                all_results.append({"test_case": tc["name"], "score": score, "info": "Correctly covered empty universe.", "sets_used": num_sets_used})
            else:
                score = 1.0 / num_sets_used
                all_results.append({"test_case": tc["name"], "score": score, "info": "Valid cover found.", "sets_used": num_sets_used})
            
            total_score += score

        except Exception as e:
            error_message = "%s: %s" % (type(e).__name__, str(e))
            all_results.append({"test_case": tc["name"], "score": 0, "error": error_message, "sets_used": float('inf')})
            overall_success = False
            # Break on first exception for now to avoid cascading errors from a bad function
            # Remove break if you want to test all cases even if one fails badly
            break 

    average_score = total_score / len(test_cases) if len(test_cases) > 0 else 0

    # Determine final score, validity, and error message
    final_score_to_return = 0.0  # Default score, will be 0.0 if not overall_success
    actual_error_message = None # Default for success, will be set if not overall_success

    if overall_success:
        final_score_to_return = average_score
        # actual_error_message remains None for success
    else:
        # For any failure, score is explicitly 0.0
        final_score_to_return = 0.0
        
        # Attempt to find a specific error message from individual test case results
        specific_error = next((r.get('error') for r in all_results if r.get('error')), None)
        
        if specific_error:
            actual_error_message = specific_error
        else:
            # Fallback error message if no specific error was found in test_results.
            # This can happen if overall_success is False due to other reasons 
            # (e.g., a test case scored 0 but didn't set an r['error']).
            actual_error_message = "Evaluation indicates failure (is_valid=False), but no specific error was pinpointed from individual test case messages. The program might not meet all success criteria across tests, or a test may have failed silently."

    # Heuristic for number of passed tests based on positive score and no error
    num_passed_tests = sum(1 for r in all_results if r.get('score', 0) > 0 and not r.get('error'))

    return {
        "score": final_score_to_return,
        "is_valid": overall_success, # Ensure correct key 'is_valid' is used
        "error_message": actual_error_message,
        "details": { 
            "num_test_cases": len(test_cases),
            "num_passed": num_passed_tests, 
            "average_score_raw": average_score, # The score before being potentially zeroed out
            "all_results": all_results # Detailed results per test case
        }
    } 