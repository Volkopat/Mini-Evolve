import math

# --- Evaluation Logic for TSP Heuristic ---

def evaluate_program(candidate_module: object, problem_config: dict, current_problem_directory: str):
    """Evaluates a given TSP heuristic, provided as an executed module.

    Args:
        candidate_module: The module object containing the executed candidate program code.
        problem_config: The configuration dictionary for the current problem.
        current_problem_directory: The directory path for the current problem. (Note: main_config was passed here previously by app/evaluator.py, but this func expects current_problem_directory as per its old signature. Review app/evaluator.py call if main_config is needed here instead/additionally)

    Returns:
        A dictionary containing:
            'score': A float score (higher is better, e.g., 1/tour_length).
            'is_valid': Boolean, True if the program executes and returns a valid tour.
            'tour_length': The length of the tour found by the heuristic.
            'tour': The tour itself (list of city indices).
            'error_message': String, error message if any, else None.
            'details': Additional details or metrics.
    """
    
    # Define a few test instances (adjacency matrices)
    # More complex instances can be loaded from files if needed.
    test_graphs = {
        "simple_4_city": [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0]
        ],
        "line_5_city": [
            [0, 1, 9, 9, 9],
            [1, 0, 1, 9, 9],
            [9, 1, 0, 1, 9],
            [9, 9, 1, 0, 1],
            [9, 9, 9, 1, 0]
        ]
        # Add more diverse graphs for thorough testing
    }

    total_inverse_tour_length = 0.0
    total_valid_tours = 0
    all_tour_details = {}
    error_messages = []

    # The program_code_string has already been exec'd into candidate_module by app/evaluator.py
    # No need for execution_globals or exec here.

    function_name = problem_config.get('function_details', {}).get('name', 'solve_tsp_heuristic')
    
    if not hasattr(candidate_module, function_name) or not callable(getattr(candidate_module, function_name)):
        return {
            'score': 0.0,
            'is_valid': False,
            'tour_length': float('inf'),
            'tour': None,
            'error_message': "Function '%s' not found or not callable in candidate module." % function_name,
            'details': {}
        }
    
    tsp_heuristic_func = getattr(candidate_module, function_name)

    for graph_name, graph_matrix in test_graphs.items():
        num_cities = len(graph_matrix)
        current_tour = None
        current_tour_length = float('inf')
        is_current_tour_valid = False
        current_error = None

        try:
            # Call the heuristic function from the executed code
            tour_candidate = tsp_heuristic_func(graph_matrix)

            # Validate the tour
            if not isinstance(tour_candidate, list) or len(tour_candidate) != num_cities or \
               sorted(tour_candidate) != list(range(num_cities)):
                current_error = "Invalid tour: Not a permutation of city indices. Tour: %s" % tour_candidate
            else:
                # Calculate tour length
                calculated_length = 0
                for i in range(num_cities):
                    from_city = tour_candidate[i]
                    to_city = tour_candidate[(i + 1) % num_cities] # Connect back to start
                    calculated_length += graph_matrix[from_city][to_city]
                
                current_tour_length = calculated_length
                current_tour = tour_candidate
                is_current_tour_valid = True
                total_valid_tours += 1
                total_inverse_tour_length += (1.0 / current_tour_length if current_tour_length > 0 else 0)
        
        except Exception as e:
            current_error = "Error during heuristic execution for %s: %s" % (graph_name, str(e))
        
        if current_error:
            error_messages.append("Graph '%s': %s" % (graph_name, current_error))
        
        all_tour_details[graph_name] = {
            'tour': current_tour,
            'length': current_tour_length,
            'is_valid': is_current_tour_valid,
            'error': current_error
        }

    final_score = 0.0
    if total_valid_tours > 0:
        # Average inverse tour length as score. Higher is better.
        final_score = total_inverse_tour_length / total_valid_tours 
    
    final_is_valid = total_valid_tours == len(test_graphs) # Consider valid if all test cases produce valid tours
    overall_error_message = "; ".join(error_messages) if error_messages else None

    # For simplicity in reporting, pick the first tour's length and details if any are valid
    # A more sophisticated approach might average lengths or pick the best/worst.
    representative_tour_length = float('inf')
    representative_tour = None
    for details in all_tour_details.values():
        if details['is_valid']:
            representative_tour_length = details['length']
            representative_tour = details['tour']
            break

    return {
        'score': final_score,
        'is_valid': final_is_valid,
        'tour_length': representative_tour_length, # Representative length
        'tour': representative_tour, # Representative tour
        'error_message': overall_error_message,
        'details': all_tour_details # Contains details for all test graphs
    }

# --- Main for testing the evaluator itself ---
if __name__ == '__main__':
    # Example: Test with the seed program from problem_config.yaml
    mock_problem_config = {
        'function_details': {
            'name': 'solve_tsp_heuristic'
        }
    }
    
    seed_program_code = """
def solve_tsp_heuristic(graph_matrix):
    num_cities = len(graph_matrix)
    if num_cities == 0: return []
    return list(range(num_cities))
"""

    print("--- Testing Seed Program --- ")
    results = evaluate_program(seed_program_code, mock_problem_config, ".")
    print("Score: %.6f" % results['score'])
    print("Is Valid: %s" % results['is_valid'])
    print("Tour Length (repr.): %s" % results['tour_length'])
    print("Tour (repr.): %s" % results['tour'])
    print("Error: %s" % results['error_message'])
    print("Details:")
    for graph_name, detail in results.get('details', {}).items():
        print("  %s: Length=%s, Tour=%s, Valid=%s, Error=%s" % 
              (graph_name, detail['length'], detail['tour'], detail['is_valid'], detail['error']))
    print("-------------------------")

    faulty_program_code = """
def solve_tsp_heuristic(graph_matrix):
    # This will cause an error or invalid tour
    return [0, 0, 1] # Invalid for 4 cities, for example
"""
    print("--- Testing Faulty Program --- ")
    results_faulty = evaluate_program(faulty_program_code, mock_problem_config, ".")
    print("Score: %.6f" % results_faulty['score'])
    print("Is Valid: %s" % results_faulty['is_valid'])
    print("Tour Length (repr.): %s" % results_faulty['tour_length'])
    print("Tour (repr.): %s" % results_faulty['tour'])
    print("Error: %s" % results_faulty['error_message'])
    print("Details:")
    for graph_name, detail in results_faulty.get('details', {}).items():
        print("  %s: Length=%s, Tour=%s, Valid=%s, Error=%s" % 
              (graph_name, detail['length'], detail['tour'], detail['is_valid'], detail['error']))
    print("-------------------------")

    better_heuristic_code_example = """
import random
def solve_tsp_heuristic(graph_matrix):
    # Example: Nearest Neighbor (simplified, starts at 0)
    num_cities = len(graph_matrix)
    if num_cities == 0: return []
    
    current_city = 0
    tour = [current_city]
    visited = {current_city}

    while len(tour) < num_cities:
        next_city = -1
        min_dist = float('inf')
        # Find nearest unvisited neighbor
        for city_idx in range(num_cities):
            if city_idx not in visited and graph_matrix[current_city][city_idx] < min_dist:
                min_dist = graph_matrix[current_city][city_idx]
                next_city = city_idx
        
        if next_city == -1: # Should not happen in a complete graph if logic is correct
            # Fallback: if stuck, pick any unvisited city (should refine this)
            for i in range(num_cities):
                if i not in visited:
                    next_city = i
                    break
            if next_city == -1: # All visited, something is wrong, or it's the last city
                 break # Should have been caught by len(tour) < num_cities

        tour.append(next_city)
        visited.add(next_city)
        current_city = next_city
    return tour
"""
    print("--- Testing Example Heuristic Program --- ")
    results_heuristic = evaluate_program(better_heuristic_code_example, mock_problem_config, ".")
    print("Score: %.6f" % results_heuristic['score'])
    print("Is Valid: %s" % results_heuristic['is_valid'])
    print("Tour Length (repr.): %s" % results_heuristic['tour_length'])
    print("Tour (repr.): %s" % results_heuristic['tour'])
    print("Error: %s" % results_heuristic['error_message'])
    print("Details:")
    for graph_name, detail in results_heuristic.get('details', {}).items():
        print("  %s: Length=%s, Tour=%s, Valid=%s, Error=%s" % 
              (graph_name, detail['length'], detail['tour'], detail['is_valid'], detail['error']))
    print("-------------------------") 