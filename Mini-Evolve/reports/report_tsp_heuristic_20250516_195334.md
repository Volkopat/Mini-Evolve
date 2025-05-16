# Mini-Evolve Run Report
Generated: 2025-05-16 19:53:34
Problem: tsp_heuristic
Database: db/program_database.db

---

## I. Overall Statistics
- Total programs in database: 8
- Valid programs: 8
- Invalid programs: 0
- Percentage valid: 100.00%
- Max score (valid programs): 0.0447
- Min score (valid programs): 0.0437
- Average score (valid programs): 0.0446
- Generations spanned: 0 to 2

## II. Best Program(s)
### Top Scorer:
- Program ID: f45b4e01-aa09-4f79-93ce-c6f92acb7226
- Score: 0.0447
- Generation Discovered: 2
- Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
- Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.0056190551258624, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
```python
def solve_tsp_heuristic(graph_matrix):
    if not graph_matrix or not isinstance(graph_matrix, list):
        return []
    
    num_cities = len(graph_matrix)
    if num_cities == 0:
        return []

    if num_cities == 1:
        return [0]
    
    # Validate matrix structure (assuming square and complete based on problem)
    # Example: if not all(isinstance(row, list) and len(row) == num_cities for row in graph_matrix): return list(range(num_cities))

    # Step 1: Generate initial tour using Nearest Neighbor heuristic
    # Start at city 0. This can be randomized or iterated for better results,
    # but for simplicity, a fixed start is used here.
    initial_tour = []
    current_city = 0 
    initial_tour.append(current_city)
    visited = {current_city}

    while len(initial_tour) < num_cities:
        next_city_candidate = -1
        min_dist_to_next = float('inf')

        for neighbor_idx in range(num_cities):
            if neighbor_idx not in visited:
                dist = graph_matrix[current_city][neighbor_idx]
                if dist < min_dist_to_next:
                    min_dist_to_next = dist
                    next_city_candidate = neighbor_idx
        
        if next_city_candidate == -1:
            # This case should ideally not be reached in a complete graph with positive distances.
            # Fallback: add remaining unvisited cities in their numerical order.
            for i_fallback in range(num_cities):
                if i_fallback not in visited:
                    initial_tour.append(i_fallback)
                    visited.add(i_fallback) # Ensure visited set is consistent
            break # Exit while loop as tour is now full (or was supposed to be)
            
        initial_tour.append(next_city_candidate)
        visited.add(next_city_candidate)
        current_city = next_city_candidate
    
    # Step 2: Improve the tour using the 2-opt heuristic

    current_best_tour = list(initial_tour) # Start with the NN tour
    
    improved = True
    while improved:
        improved = False
        
        # Iterate over all pairs of non-adjacent edges for a potential 2-opt swap
        for i_loop_idx in range(num_cities): 
            # This i_loop_idx is an index into current_best_tour.
            # First edge is (current_best_tour[i_loop_idx], current_best_tour[(i_loop_idx + 1) % num_cities])
            
            # k_loop_idx_offset ensures that k_loop_idx defines an edge non-adjacent to the first one.
            # The second edge starts at current_best_tour[k_loop_idx].
            # k_loop_idx must not be i_loop_idx or (i_loop_idx + 1) % num_cities.
            # So, k_loop_idx starts cyclically from (i_loop_idx + 2).
            for k_loop_idx_offset in range(2, num_cities): 
                k_loop_idx = (i_loop_idx + k_loop_idx_offset) % num_cities

                # Define the four critical node indices in the current_best_tour list
                # Edge 1 is (tour[idx_A], tour[idx_B])
                # Edge 2 is (tour[idx_C], tour[idx_D])
                idx_A = i_loop_idx
                idx_B = (i_loop_idx + 1) % num_cities
                idx_C = k_loop_idx
                idx_D = (k_loop_idx + 1) % num_cities
                
                # Retrieve the actual city IDs for these nodes
                node_A = current_best_tour[idx_A]
                node_B = current_best_tour[idx_B]
                node_C = current_best_tour[idx_C]
                node_D = current_best_tour[idx_D]

                # Calculate cost of current edges vs. new edges if swapped
                # Current edges: (A,B) and (C,D)
                cost_original_edges = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                
                # New edges if swapped: (A,C) and (B,D)
                # This swap implies reversing the path segment between B and C (inclusive of B and C if thinking path segments)
                # More precisely, the tour segment from tour[idx_B] to tour[idx_C] is reversed.
                cost_new_edges = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]

                if cost_new_edges < cost_original_edges:
                    # Improvement found, perform the 2-opt swap
                    new_tour_after_swap = list(current_best_tour) # Make a copy to modify
                    
                    # Reverse the segment of the tour from new_tour_after_swap[idx_B] to new_tour_after_swap[idx_C]
                    # Store elements of the segment to be reversed
                    segment_nodes_to_reverse = []
                    current_segment_fill_idx = idx_B
                    while True:
                        segment_nodes_to_reverse.append(new_tour_after_swap[current_segment_fill_idx])
                        if current_segment_fill_idx == idx_C:
                            break
                        current_segment_fill_idx = (current_segment_fill_idx + 1) % num_cities
                    
                    segment_nodes_to_reverse.reverse() # Reverse the collected segment
                    
                    # Place the reversed segment back into the tour
                    current_placement_idx = idx_B
                    for city_node_in_segment in segment_nodes_to_reverse:
                        new_tour_after_swap[current_placement_idx] = city_node_in_segment
                        if current_placement_idx == idx_C: # Stop once the segment is filled
                            break
                        current_placement_idx = (current_placement_idx + 1) % num_cities
                    
                    current_best_tour = new_tour_after_swap # Update the tour
                    improved = True # Mark that an improvement was made
                    
                    # Break from inner loops to restart the 2-opt scan from the beginning with the new tour
                    break # Exit k_loop_idx_offset loop
            if improved:
                break # Exit i_loop_idx loop (and thus restart the 'while improved' loop)
                
    return current_best_tour
```

## III. Top 5 Programs (by Score)

### 1. Program ID: f45b4e01-aa09-4f79-93ce-c6f92acb7226
    - Score: 0.0447
    - Generation: 2
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.0056190551258624, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
        
        # Validate matrix structure (assuming square and complete based on problem)
        # Example: if not all(isinstance(row, list) and len(row) == num_cities for row in graph_matrix): return list(range(num_cities))
    
        # Step 1: Generate initial tour using Nearest Neighbor heuristic
        # Start at city 0. This can be randomized or iterated for better results,
        # but for simplicity, a fixed start is used here.
        initial_tour = []
        current_city = 0 
        initial_tour.append(current_city)
        visited = {current_city}
    
        while len(initial_tour) < num_cities:
            next_city_candidate = -1
            min_dist_to_next = float('inf')
    
            for neighbor_idx in range(num_cities):
                if neighbor_idx not in visited:
                    dist = graph_matrix[current_city][neighbor_idx]
                    if dist < min_dist_to_next:
                        min_dist_to_next = dist
                        next_city_candidate = neighbor_idx
            
            if next_city_candidate == -1:
                # This case should ideally not be reached in a complete graph with positive distances.
                # Fallback: add remaining unvisited cities in their numerical order.
                for i_fallback in range(num_cities):
                    if i_fallback not in visited:
                        initial_tour.append(i_fallback)
                        visited.add(i_fallback) # Ensure visited set is consistent
                break # Exit while loop as tour is now full (or was supposed to be)
                
            initial_tour.append(next_city_candidate)
            visited.add(next_city_candidate)
            current_city = next_city_candidate
        
        # Step 2: Improve the tour using the 2-opt heuristic
    
        current_best_tour = list(initial_tour) # Start with the NN tour
        
        improved = True
        while improved:
            improved = False
            
            # Iterate over all pairs of non-adjacent edges for a potential 2-opt swap
            for i_loop_idx in range(num_cities): 
                # This i_loop_idx is an index into current_best_tour.
                # First edge is (current_best_tour[i_loop_idx], current_best_tour[(i_loop_idx + 1) % num_cities])
                
                # k_loop_idx_offset ensures that k_loop_idx defines an edge non-adjacent to the first one.
                # The second edge starts at current_best_tour[k_loop_idx].
                # k_loop_idx must not be i_loop_idx or (i_loop_idx + 1) % num_cities.
                # So, k_loop_idx starts cyclically from (i_loop_idx + 2).
                for k_loop_idx_offset in range(2, num_cities): 
                    k_loop_idx = (i_loop_idx + k_loop_idx_offset) % num_cities
    
                    # Define the four critical node indices in the current_best_tour list
                    # Edge 1 is (tour[idx_A], tour[idx_B])
                    # Edge 2 is (tour[idx_C], tour[idx_D])
                    idx_A = i_loop_idx
                    idx_B = (i_loop_idx + 1) % num_cities
                    idx_C = k_loop_idx
                    idx_D = (k_loop_idx + 1) % num_cities
                    
                    # Retrieve the actual city IDs for these nodes
                    node_A = current_best_tour[idx_A]
                    node_B = current_best_tour[idx_B]
                    node_C = current_best_tour[idx_C]
                    node_D = current_best_tour[idx_D]
    
                    # Calculate cost of current edges vs. new edges if swapped
                    # Current edges: (A,B) and (C,D)
                    cost_original_edges = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                    
                    # New edges if swapped: (A,C) and (B,D)
                    # This swap implies reversing the path segment between B and C (inclusive of B and C if thinking path segments)
                    # More precisely, the tour segment from tour[idx_B] to tour[idx_C] is reversed.
                    cost_new_edges = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]
    
                    if cost_new_edges < cost_original_edges:
                        # Improvement found, perform the 2-opt swap
                        new_tour_after_swap = list(current_best_tour) # Make a copy to modify
                        
                        # Reverse the segment of the tour from new_tour_after_swap[idx_B] to new_tour_after_swap[idx_C]
                        # Store elements of the segment to be reversed
                        segment_nodes_to_reverse = []
                        current_segment_fill_idx = idx_B
                        while True:
                            segment_nodes_to_reverse.append(new_tour_after_swap[current_segment_fill_idx])
                            if current_segment_fill_idx == idx_C:
                                break
                            current_segment_fill_idx = (current_segment_fill_idx + 1) % num_cities
                        
                        segment_nodes_to_reverse.reverse() # Reverse the collected segment
                        
                        # Place the reversed segment back into the tour
                        current_placement_idx = idx_B
                        for city_node_in_segment in segment_nodes_to_reverse:
                            new_tour_after_swap[current_placement_idx] = city_node_in_segment
                            if current_placement_idx == idx_C: # Stop once the segment is filled
                                break
                            current_placement_idx = (current_placement_idx + 1) % num_cities
                        
                        current_best_tour = new_tour_after_swap # Update the tour
                        improved = True # Mark that an improvement was made
                        
                        # Break from inner loops to restart the 2-opt scan from the beginning with the new tour
                        break # Exit k_loop_idx_offset loop
                if improved:
                    break # Exit i_loop_idx loop (and thus restart the 'while improved' loop)
                    
        return current_best_tour
    ```

### 2. Program ID: 59e329ce-7dfc-4a03-9db2-3765e96d9ea2
    - Score: 0.0447
    - Generation: 2
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.7139550182037055, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # 1. Nearest Neighbor Heuristic to get an initial tour
        # Try starting from each city and take the best initial tour from NN
        # This adds O(N) factor to NN, making it O(N^3) for this part,
        # or O(N^2) if NN is O(N) per start. NN is O(N^2) per start.
        # So this would be O(N^3). Let's stick to a single start for NN (city 0)
        # to keep NN part O(N^2).
        
        current_city_nn = 0  # Start at city 0 for NN
        tour_nn = [current_city_nn]
        visited_nn = {current_city_nn}
    
        temp_current_city = current_city_nn
        while len(tour_nn) < num_cities:
            next_city_candidate = -1
            min_dist_candidate = float('inf')
            
            for city_idx in range(num_cities):
                if city_idx not in visited_nn:
                    dist = graph_matrix[temp_current_city][city_idx]
                    if dist < min_dist_candidate:
                        min_dist_candidate = dist
                        next_city_candidate = city_idx
            
            # In a complete graph, next_city_candidate should always be found
            # if len(tour_nn) < num_cities.
            if next_city_candidate == -1:
                # Fallback if something is wrong (e.g. not a complete graph as assumed)
                # This part should ideally not be reached given problem constraints.
                # Add any remaining unvisited city to complete the tour.
                for city_idx in range(num_cities):
                    if city_idx not in visited_nn:
                        next_city_candidate = city_idx
                        break
                if next_city_candidate == -1: # Should be impossible if len(tour_nn) < num_cities
                    break # Safety break
    
            tour_nn.append(next_city_candidate)
            visited_nn.add(next_city_candidate)
            temp_current_city = next_city_candidate
        
        best_tour = list(tour_nn) 
    
        # 2. Improve the tour using 2-opt heuristic
        improved = True
        while improved:
            improved = False
            for i in range(num_cities):  # Index for the start of the segment to reverse (inclusive)
                for j in range(i + 1, num_cities):  # Index for the end of the segment to reverse (inclusive)
                                                    # Segment is best_tour[i...j]
    
                    # Reversing the whole tour (i=0, j=N-1) doesn't change its length.
                    # It's equivalent to traversing it in the opposite direction.
                    if i == 0 and j == num_cities - 1:
                        continue
    
                    # Current nodes involved in the two edges being swapped by reversing best_tour[i...j]:
                    # Edge 1 (before segment start): ( best_tour[(i-1+N)%N], best_tour[i] )
                    # Edge 2 (after segment end):   ( best_tour[j], best_tour[(j+1+N)%N] )
                    
                    # Values of the nodes:
                    node_before_segment_start_val = best_tour[(i - 1 + num_cities) % num_cities]
                    node_segment_start_val = best_tour[i]
                    node_segment_end_val = best_tour[j]
                    node_after_segment_end_val = best_tour[(j + 1) % num_cities]
                    
                    # Cost change calculation:
                    # (cost of new edges) - (cost of old edges)
                    # After reversing segment best_tour[i...j], the new edges are:
                    # New Edge 1': ( node_before_segment_start_val, node_segment_end_val )
                    # New Edge 2': ( node_segment_start_val, node_after_segment_end_val )
                    
                    current_edge_costs = (graph_matrix[node_before_segment_start_val][node_segment_start_val] + 
                                          graph_matrix[node_segment_end_val][node_after_segment_end_val])
                    
                    new_edge_costs = (graph_matrix[node_before_segment_start_val][node_segment_end_val] + 
                                      graph_matrix[node_segment_start_val][node_after_segment_end_val])
                    
                    cost_change = new_edge_costs - current_edge_costs
    
                    # If improvement is found (cost_change is negative)
                    if cost_change < -1e-9:  # Using a small epsilon for floating point comparisons
                        # Perform the 2-opt swap by reversing the segment best_tour[i...j]
                        segment_to_reverse = best_tour[i : j+1] # Python slice: elements from index i up to, but not including, j+1
                        segment_to_reverse.reverse()
                        
                        # Reconstruct the tour with the reversed segment
                        best_tour = best_tour[:i] + segment_to_reverse + best_tour[j+1:]
                        
                        improved = True
                        # Restart search for improvements from the beginning, as the tour has changed
                        break  # Exit j loop
                if improved:
                    break # Exit i loop, and restart the 'while improved' loop
                    
        return best_tour
    ```

### 3. Program ID: 57b791be-7080-41c3-bd84-5946e6df568a
    - Score: 0.0447
    - Generation: 2
    - Parent ID: 1735484a-caa4-487d-842b-0646d6eabc53
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.897874055430293, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        
        if num_cities == 1:
            # This logic correctly handles graph_matrix = [[0]] (returns [0])
            # and graph_matrix = [[]] (returns []) due to the len check inside.
            if not graph_matrix[0]: # True for graph_matrix = [[]]
                 # This inner check is specific for [[]] where num_cities = 1 but graph_matrix[0] is empty.
                 if len(graph_matrix[0]) != num_cities : 
                     return [] 
            return [0]
    
        # Validate that graph_matrix[0] is a list and all rows have correct length.
        # As per problem spec, graph_matrix is a square matrix, so this should hold.
        # This check is more for robustness against malformed inputs not strictly following spec.
        if not isinstance(graph_matrix[0], list) or len(graph_matrix[0]) != num_cities:
            return [] # Malformed input
    
        # 1. Initial tour using Nearest Neighbor Heuristic
        start_node = 0 
        current_city = start_node
        tour = [current_city]
        visited = [False] * num_cities
        visited[current_city] = True
        num_visited = 1
    
        while num_visited < num_cities:
            next_city = -1
            min_dist = float('inf')
            
            for candidate_city in range(num_cities):
                if not visited[candidate_city]:
                    # Problem states distances are positive, graph_matrix[i][i]=0
                    # graph_matrix[current_city][candidate_city] should be > 0
                    distance = graph_matrix[current_city][candidate_city]
                    if distance < min_dist:
                        min_dist = distance
                        next_city = candidate_city
            
            if next_city == -1:
                # Fallback for incomplete graph or if all remaining distances are inf.
                # Should not happen given problem constraints (complete graph, positive distances).
                found_unvisited = False
                for i in range(num_cities):
                    if not visited[i]:
                        next_city = i
                        found_unvisited = True
                        break
                if not found_unvisited: 
                    break # All cities somehow marked visited, exit.
    
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city
            num_visited += 1
        
        if len(tour) != num_cities or len(set(tour)) != num_cities:
            # Fallback if NN tour construction failed to produce a valid permutation.
            return list(range(num_cities))
    
        # 2. Improve using 2-opt Heuristic
        best_tour = list(tour) 
        
        # Set a limit on the number of full improvement passes for 2-opt.
        # Using num_cities as the limit is a common heuristic.
        # For small N, 2-opt converges very quickly.
        # For N=2,3, 2-opt loops will do minimal/zero work or converge in 1 pass.
        # Limit of N passes for N cities is generally sufficient for significant improvement.
        pass_iteration_limit = num_cities
        
        current_pass = 0
        improvement_found_in_cycle = True 
    
        while improvement_found_in_cycle and current_pass < pass_iteration_limit :
            improvement_found_in_cycle = False 
            current_pass += 1
    
            for i in range(num_cities - 1): 
                for j in range(i + 2, num_cities): 
                    node_pi = best_tour[i]
                    node_pi1 = best_tour[i+1] 
                    node_pj = best_tour[j]   
                    node_pj1 = best_tour[(j + 1) % num_cities] # Handles wrap-around
    
                    cost_current_edges = graph_matrix[node_pi][node_pi1] + graph_matrix[node_pj][node_pj1]
                    cost_new_edges = graph_matrix[node_pi][node_pj] + graph_matrix[node_pi1][node_pj1]
    
                    if cost_new_edges < cost_current_edges:
                        # Improvement found. Reverse the segment best_tour[i+1...j]
                        segment_to_reverse = best_tour[i+1 : j+1]
                        segment_to_reverse.reverse() 
                        
                        best_tour = best_tour[0 : i+1] + segment_to_reverse + best_tour[j+1 : num_cities]
                        
                        improvement_found_in_cycle = True 
                        
                        # "First improvement" strategy: restart scans if an improvement is made.
                        break 
                if improvement_found_in_cycle:
                    break 
    
        return best_tour
    ```

### 4. Program ID: 0fd07314-d331-4ee7-bf84-fb09f6ca230e
    - Score: 0.0447
    - Generation: 2
    - Parent ID: 6871580e-750a-4fa8-8731-3e2ed983494a
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.9905780209228396, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix:
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
    
        # Part 1: Nearest Neighbor to get a good initial tour
        # Try starting from each city and pick the best tour found.
        best_nn_tour = None
        min_nn_tour_length = float('inf')
    
        for start_node in range(num_cities):
            current_city = start_node
            tour = [current_city]
            visited = {current_city}
            current_tour_segment_length = 0.0
    
            while len(tour) < num_cities:
                next_city_candidate = -1
                min_dist_to_next = float('inf')
                
                for neighbor_idx in range(num_cities):
                    if neighbor_idx not in visited:
                        distance = graph_matrix[current_city][neighbor_idx]
                        if distance < min_dist_to_next:
                            min_dist_to_next = distance
                            next_city_candidate = neighbor_idx
                        elif distance == min_dist_to_next:
                            if next_city_candidate == -1 or neighbor_idx < next_city_candidate:
                                next_city_candidate = neighbor_idx
                
                if next_city_candidate == -1:
                    # This case should ideally not be reached in a complete graph
                    # as specified by problem constraints if len(tour) < num_cities.
                    break 
                
                current_tour_segment_length += min_dist_to_next
                tour.append(next_city_candidate)
                visited.add(next_city_candidate)
                current_city = next_city_candidate
            
            if len(tour) == num_cities: 
                final_edge_length = graph_matrix[tour[-1]][tour[0]]
                total_tour_length = current_tour_segment_length + final_edge_length
                
                if total_tour_length < min_nn_tour_length:
                    min_nn_tour_length = total_tour_length
                    best_nn_tour = list(tour) 
    
        if best_nn_tour is None:
            # Fallback if Nearest Neighbor somehow fails (e.g., graph not complete as expected)
            # Given problem constraints, this should not be hit for num_cities > 1.
            return list(range(num_cities)) 
    
        # Part 2: 2-opt refinement using "best improvement" strategy
        tour_to_optimize = list(best_nn_tour) 
        
        while True: 
            min_improvement_delta = 0 
            best_i_swap = -1
            best_k_swap = -1
    
            for i in range(num_cities - 1): 
                node_A = tour_to_optimize[i]
                node_B = tour_to_optimize[i+1] 
    
                for k in range(i + 1, num_cities): 
                    node_C = tour_to_optimize[k]
                    node_D = tour_to_optimize[(k + 1) % num_cities]
    
                    original_edges_len = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                    new_edges_len = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]
                    
                    current_delta = new_edges_len - original_edges_len
    
                    if current_delta < min_improvement_delta:
                        min_improvement_delta = current_delta
                        best_i_swap = i
                        best_k_swap = k
            
            if min_improvement_delta < 0: 
                i_to_swap = best_i_swap
                k_to_swap = best_k_swap
                
                segment_to_reverse = tour_to_optimize[i_to_swap+1 : k_to_swap+1]
                segment_to_reverse.reverse()
                
                tour_to_optimize = tour_to_optimize[0 : i_to_swap+1] + \
                                   segment_to_reverse + \
                                   tour_to_optimize[k_to_swap+1 : num_cities]
            else:
                break 
        
        return tour_to_optimize
    ```

### 5. Program ID: 64bc1a45-2e62-438d-818e-5d0cafb14bba
    - Score: 0.0447
    - Generation: 2
    - Parent ID: 6871580e-750a-4fa8-8731-3e2ed983494a
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.7923150337301195, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix:
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
    
        # Part 1: Nearest Neighbor to get a good initial tour
        # Try starting from each city and pick the best tour found.
        best_nn_tour = None
        min_nn_tour_length = float('inf')
    
        for start_node in range(num_cities):
            current_city = start_node
            tour = [current_city]
            visited = {current_city}
            current_tour_segment_length = 0.0
    
            while len(tour) < num_cities:
                next_city_candidate = -1
                min_dist_to_next = float('inf')
                
                for neighbor_idx in range(num_cities):
                    if neighbor_idx not in visited:
                        distance = graph_matrix[current_city][neighbor_idx]
                        if distance < min_dist_to_next:
                            min_dist_to_next = distance
                            next_city_candidate = neighbor_idx
                        elif distance == min_dist_to_next:
                            if next_city_candidate == -1 or neighbor_idx < next_city_candidate: # Deterministic tie-breaking
                                next_city_candidate = neighbor_idx
                
                if next_city_candidate == -1:
                    # Should only happen if graph is not complete or all cities visited
                    break 
                
                current_tour_segment_length += min_dist_to_next
                tour.append(next_city_candidate)
                visited.add(next_city_candidate)
                current_city = next_city_candidate
            
            if len(tour) == num_cities: 
                final_edge_length = graph_matrix[tour[-1]][tour[0]]
                total_tour_length = current_tour_segment_length + final_edge_length
                
                if total_tour_length < min_nn_tour_length:
                    min_nn_tour_length = total_tour_length
                    best_nn_tour = list(tour) 
    
        if best_nn_tour is None:
            # Fallback for num_cities > 1 if NN somehow fails (e.g., graph not complete as expected)
            # This path should ideally not be hit given problem constraints.
            # Create a naive tour if no tour was found by NN.
            if num_cities > 0 : # Ensure range(num_cities) is not called with 0 if somehow missed earlier checks
                 best_nn_tour = list(range(num_cities))
            else: # Should be caught by initial checks
                 return []
    
    
        # Part 2: 2-opt refinement
        tour_to_optimize = list(best_nn_tour) 
        improved = True
        while improved:
            improved = False
            for i in range(num_cities -1): # Iterate over all possible first edges (A,B)
                # node_A is tour_to_optimize[i]
                # node_B is tour_to_optimize[i+1]
                # These are indices in the tour list, not city indices directly unless tour is [0,1,...,N-1]
                
                # Iterate over all possible second edges (C,D) that don't overlap with (A,B) in a problematic way
                # k is the index for node_C in the tour list
                # (k+1)%num_cities is the index for node_D
                for k in range(i + 1, num_cities): 
                    # node_C is tour_to_optimize[k]
                    # node_D is tour_to_optimize[(k + 1) % num_cities]
                    
                    # Current edges: (A,B) and (C,D)
                    # A = tour_to_optimize[i]
                    # B = tour_to_optimize[i+1] (this index i+1 is always valid as i goes up to num_cities-2)
                    # C = tour_to_optimize[k]
                    # D = tour_to_optimize[(k+1)%num_cities] (this index (k+1)%num_cities is always valid)
    
                    node_A_val = tour_to_optimize[i]
                    node_B_val = tour_to_optimize[i+1]
                    node_C_val = tour_to_optimize[k]
                    node_D_val = tour_to_optimize[(k + 1) % num_cities]
    
                    # Cost of original edges
                    original_edges_len = graph_matrix[node_A_val][node_B_val] + graph_matrix[node_C_val][node_D_val]
                    
                    # Cost of new edges if we swap: (A,C) and (B,D)
                    new_edges_len = graph_matrix[node_A_val][node_C_val] + graph_matrix[node_B_val][node_D_val]
    
                    if new_edges_len < original_edges_len:
                        # Improvement found, apply the swap by reversing the segment tour_to_optimize[i+1...k]
                        segment_to_reverse = tour_to_optimize[i+1 : k+1]
                        segment_to_reverse.reverse()
                        
                        tour_to_optimize = tour_to_optimize[0 : i+1] + segment_to_reverse + tour_to_optimize[k+1 : num_cities]
                        
                        improved = True
                        # Using "first improvement" strategy: restart search from the beginning of the modified tour
                        break 
                if improved:
                    break 
        
        return tour_to_optimize
    ```

## IV. Evolutionary Lineage (Parent-Child)
- Gen: 0, ID: 179d54a3 (Score: 0.044, V)
    - Gen: 1, ID: 1735484a (Score: 0.045, V)
        - Gen: 2, ID: 57b791be (Score: 0.045, V)
    - Gen: 1, ID: 6871580e (Score: 0.045, V)
        - Gen: 2, ID: 64bc1a45 (Score: 0.045, V)
        - Gen: 2, ID: 0fd07314 (Score: 0.045, V)
    - Gen: 2, ID: 59e329ce (Score: 0.045, V)
    - Gen: 2, ID: f45b4e01 (Score: 0.045, V)

## V. All Programs by Generation & Timestamp

### 1. Program ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52 (Gen: 0)
    - Score: 0.0437
    - Valid: True
    - Parent ID: None
    - Timestamp: 1747424800.93
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using a naive heuristic.
        Simply visits cities in the order they appear (0, 1, 2, ...).
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            # Basic validation
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        # Naive tour: visit cities in their original order
        tour = list(range(num_cities))
        return tour
    
    ```

### 2. Program ID: 1735484a-caa4-487d-842b-0646d6eabc53 (Gen: 1)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Timestamp: 1747424953.69
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        
        if num_cities == 1:
            if not graph_matrix[0]: # Handles case like [[]] which is invalid for N=1
                 # Or if graph_matrix = [[0]] but len(graph_matrix[0]) == 0
                 # However, problem implies valid square matrix.
                 # If num_cities is 1, graph_matrix should be [[0]] or similar.
                 if len(graph_matrix[0]) != num_cities :  # e.g. graph_matrix = [[]] -> num_cities = 1, len(graph_matrix[0])=0
                     return [] # Invalid input for N=1
            return [0]
    
        # 1. Initial tour using Nearest Neighbor Heuristic
        start_node = 0 # Start at city 0 by convention
        current_city = start_node
        tour = [current_city]
        visited = [False] * num_cities
        visited[current_city] = True
        num_visited = 1
    
        while num_visited < num_cities:
            next_city = -1
            min_dist = float('inf')
            
            for candidate_city in range(num_cities):
                if not visited[candidate_city]:
                    distance = graph_matrix[current_city][candidate_city]
                    if distance < min_dist:
                        min_dist = distance
                        next_city = candidate_city
            
            if next_city == -1:
                # This should not happen in a complete graph if num_visited < num_cities
                # If it does, it implies an issue or disconnected graph (not per spec)
                # As a fallback, add any remaining unvisited city to complete the tour.
                # This part is a safeguard; for a complete graph, next_city should always be found.
                found_unvisited = False
                for i in range(num_cities):
                    if not visited[i]:
                        next_city = i
                        found_unvisited = True
                        break
                if not found_unvisited: # All cities are somehow visited, but num_visited was wrong
                    break # Exit while loop
    
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city
            num_visited += 1
        
        # Ensure tour is a valid permutation, especially if NN had issues (unlikely for complete graph)
        if len(tour) != num_cities or len(set(tour)) != num_cities:
            # Fallback to a naive tour if construction failed.
            # This should ideally not be reached if graph is complete and distances are numbers.
            return list(range(num_cities))
    
        # 2. Improve using 2-opt Heuristic
        best_tour = list(tour) # Start with the NN tour
    
        # Limit iterations for 2-opt to prevent very long runtimes.
        # A limit related to num_cities is common.
        # Using num_cities as a base for iterations limit.
        # For small N, allow a fixed minimum number of passes if needed.
        # max_passes = num_cities 
        # Let's use a slightly more generous fixed number of passes, or N, whichever is larger.
        # This ensures that for small N, it still tries a few full passes.
        # max_total_passes = max(num_cities, 20) # Example: at least 20 passes or N passes
        # Or simply iterate until no improvement, with a safety break for very large N or pathological cases.
        # For this problem, num_cities iterations should be a reasonable number of passes.
        
        # Using a simple pass limit.
        # Iteration_limit refers to full passes over the 2-opt neighborhood.
        # Set a practical limit on the number of improvement passes.
        # For very small N (e.g. N < 4), 2-opt might not do much or anything.
        # The loops for i and j will handle small N correctly.
        
        # Let's use a number of improvement rounds.
        # Typically, 2-opt converges relatively quickly.
        # Using num_cities as a limit for the number of successful improvement passes.
        
        # The structure: keep trying as long as an improvement was made in the last full pass.
        # Add a counter for total passes to avoid getting stuck too long.
        pass_iteration_limit = num_cities # Max number of full improvement passes
        if num_cities < 5: # For very small N, allow a few more relative passes
            pass_iteration_limit = 10
        
        current_pass = 0
        improvement_in_this_pass_cycle = True # Controls the outer loop of 2-opt passes
    
        while improvement_in_this_pass_cycle and current_pass < pass_iteration_limit :
            improvement_in_this_pass_cycle = False # Reset for the current pass
            current_pass += 1
    
            for i in range(num_cities -1): # Index for the first node of the first edge (0 to N-2)
                # P_i is best_tour[i]
                # P_{i+1} is best_tour[i+1] (first edge is P_i -> P_{i+1})
                
                for j in range(i + 2, num_cities): # Index for the first node of the second edge (i+2 to N-1)
                    # P_j is best_tour[j]
                    # P_{j+1} is best_tour[(j+1)%num_cities] (second edge is P_j -> P_{j+1})
                    # This ensures edges are non-adjacent. Segment to reverse is P_{i+1}...P_j.
    
                    node_pi = best_tour[i]
                    node_pi1 = best_tour[i+1] 
                    node_pj = best_tour[j]   
                    node_pj1 = best_tour[(j + 1) % num_cities] # Handles wrap-around if j is last city
    
                    # Cost of current edges: (P_i, P_{i+1}) and (P_j, P_{j+1})
                    cost_current_edges = graph_matrix[node_pi][node_pi1] + graph_matrix[node_pj][node_pj1]
                    
                    # Cost of new edges if segment P_{i+1}...P_j is reversed: (P_i, P_j) and (P_{i+1}, P_{j+1})
                    cost_new_edges = graph_matrix[node_pi][node_pj] + graph_matrix[node_pi1][node_pj1]
    
                    if cost_new_edges < cost_current_edges:
                        # Improvement found. Reverse the segment best_tour[i+1...j]
                        # The slice best_tour[i+1 : j+1] corresponds to elements from index i+1 to j.
                        segment_to_reverse = best_tour[i+1 : j+1]
                        segment_to_reverse.reverse() # In-place reverse
                        
                        # Reconstruct the tour
                        best_tour = best_tour[0 : i+1] + segment_to_reverse + best_tour[j+1 : num_cities]
                        
                        improvement_in_this_pass_cycle = True # Mark that an improvement was made in this cycle
                        
                        # "First improvement" strategy: if an improvement is found,
                        # break inner loops and restart the pass from the beginning,
                        # as the tour structure has changed.
                        break # Exit j loop (inner loop over second edge)
                if improvement_in_this_pass_cycle:
                    break # Exit i loop (outer loop over first edge), to restart the full pass cycle
    
        return best_tour
    ```

### 3. Program ID: 6871580e-750a-4fa8-8731-3e2ed983494a (Gen: 1)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Timestamp: 1747424953.71
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
    
        # Part 1: Nearest Neighbor to get a good initial tour
        # Try starting from each city and pick the best tour found.
        best_nn_tour = None
        min_nn_tour_length = float('inf')
    
        for start_node in range(num_cities):
            current_city = start_node
            tour = [current_city]
            visited = {current_city}
            current_tour_segment_length = 0.0 # Length of the path segments, not the full cycle yet
    
            while len(tour) < num_cities:
                next_city_candidate = -1
                min_dist_to_next = float('inf')
                
                # Find the nearest unvisited city
                for neighbor_idx in range(num_cities):
                    if neighbor_idx not in visited:
                        distance = graph_matrix[current_city][neighbor_idx]
                        if distance < min_dist_to_next:
                            min_dist_to_next = distance
                            next_city_candidate = neighbor_idx
                        # Tie-breaking: if distances are equal, prefer the city with the smaller index
                        elif distance == min_dist_to_next:
                            if next_city_candidate == -1 or neighbor_idx < next_city_candidate:
                                next_city_candidate = neighbor_idx
                
                if next_city_candidate == -1:
                    # This implies no unvisited city is reachable, or all cities visited.
                    # Given a complete graph, this should only happen if all cities are visited.
                    # The loop `while len(tour) < num_cities` handles normal termination.
                    # If it breaks here, it means an issue (e.g. disconnected graph, not per spec).
                    break 
                
                current_tour_segment_length += min_dist_to_next
                tour.append(next_city_candidate)
                visited.add(next_city_candidate)
                current_city = next_city_candidate
            
            if len(tour) == num_cities: # Ensure a full tour was constructed
                # Complete the cycle by adding the edge from the last city back to the start city
                final_edge_length = graph_matrix[tour[-1]][tour[0]]
                total_tour_length = current_tour_segment_length + final_edge_length
                
                if total_tour_length < min_nn_tour_length:
                    min_nn_tour_length = total_tour_length
                    best_nn_tour = list(tour) # Store a copy of the best tour found so far
    
        if best_nn_tour is None:
            # Fallback if Nearest Neighbor somehow fails to produce a tour
            # (e.g., if graph isn't complete as expected, or other rare edge cases for num_cities > 1)
            # For num_cities > 0, this path should ideally not be hit given problem constraints.
            return list(range(num_cities)) # Naive tour
    
        # Part 2: 2-opt refinement
        # Apply 2-opt to the best tour found by Nearest Neighbor
        tour_to_optimize = list(best_nn_tour) 
        improved = True
        while improved:
            improved = False
            for i in range(num_cities - 1): 
                # i is the index of the first city of the first edge (A)
                # The first edge is (A, B) = (tour[i], tour[i+1])
                for k in range(i + 1, num_cities): 
                    # k is the index of the first city of the second edge (C)
                    # The second edge is (C, D) = (tour[k], tour[(k+1)%num_cities])
                    # The segment to reverse is tour[i+1 ... k]
                    
                    node_A = tour_to_optimize[i]
                    node_B = tour_to_optimize[i+1] 
                    node_C = tour_to_optimize[k]
                    node_D = tour_to_optimize[(k + 1) % num_cities]
    
                    # Calculate change in tour length if edges (A,B) and (C,D)
                    # are replaced by (A,C) and (B,D)
                    original_edges_len = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                    new_edges_len = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]
    
                    if new_edges_len < original_edges_len:
                        # Improvement found, apply the swap
                        # Reverse the segment tour_to_optimize[i+1...k]
                        segment_to_reverse = tour_to_optimize[i+1 : k+1]
                        segment_to_reverse.reverse()
                        
                        tour_to_optimize = tour_to_optimize[0 : i+1] + segment_to_reverse + tour_to_optimize[k+1 : num_cities]
                        
                        improved = True
                        # Using "first improvement" strategy: restart search from the beginning of the modified tour
                        break 
                if improved:
                    break 
        
        return tour_to_optimize
    ```

### 4. Program ID: 64bc1a45-2e62-438d-818e-5d0cafb14bba (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 6871580e-750a-4fa8-8731-3e2ed983494a
    - Timestamp: 1747425144.18
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix:
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
    
        # Part 1: Nearest Neighbor to get a good initial tour
        # Try starting from each city and pick the best tour found.
        best_nn_tour = None
        min_nn_tour_length = float('inf')
    
        for start_node in range(num_cities):
            current_city = start_node
            tour = [current_city]
            visited = {current_city}
            current_tour_segment_length = 0.0
    
            while len(tour) < num_cities:
                next_city_candidate = -1
                min_dist_to_next = float('inf')
                
                for neighbor_idx in range(num_cities):
                    if neighbor_idx not in visited:
                        distance = graph_matrix[current_city][neighbor_idx]
                        if distance < min_dist_to_next:
                            min_dist_to_next = distance
                            next_city_candidate = neighbor_idx
                        elif distance == min_dist_to_next:
                            if next_city_candidate == -1 or neighbor_idx < next_city_candidate: # Deterministic tie-breaking
                                next_city_candidate = neighbor_idx
                
                if next_city_candidate == -1:
                    # Should only happen if graph is not complete or all cities visited
                    break 
                
                current_tour_segment_length += min_dist_to_next
                tour.append(next_city_candidate)
                visited.add(next_city_candidate)
                current_city = next_city_candidate
            
            if len(tour) == num_cities: 
                final_edge_length = graph_matrix[tour[-1]][tour[0]]
                total_tour_length = current_tour_segment_length + final_edge_length
                
                if total_tour_length < min_nn_tour_length:
                    min_nn_tour_length = total_tour_length
                    best_nn_tour = list(tour) 
    
        if best_nn_tour is None:
            # Fallback for num_cities > 1 if NN somehow fails (e.g., graph not complete as expected)
            # This path should ideally not be hit given problem constraints.
            # Create a naive tour if no tour was found by NN.
            if num_cities > 0 : # Ensure range(num_cities) is not called with 0 if somehow missed earlier checks
                 best_nn_tour = list(range(num_cities))
            else: # Should be caught by initial checks
                 return []
    
    
        # Part 2: 2-opt refinement
        tour_to_optimize = list(best_nn_tour) 
        improved = True
        while improved:
            improved = False
            for i in range(num_cities -1): # Iterate over all possible first edges (A,B)
                # node_A is tour_to_optimize[i]
                # node_B is tour_to_optimize[i+1]
                # These are indices in the tour list, not city indices directly unless tour is [0,1,...,N-1]
                
                # Iterate over all possible second edges (C,D) that don't overlap with (A,B) in a problematic way
                # k is the index for node_C in the tour list
                # (k+1)%num_cities is the index for node_D
                for k in range(i + 1, num_cities): 
                    # node_C is tour_to_optimize[k]
                    # node_D is tour_to_optimize[(k + 1) % num_cities]
                    
                    # Current edges: (A,B) and (C,D)
                    # A = tour_to_optimize[i]
                    # B = tour_to_optimize[i+1] (this index i+1 is always valid as i goes up to num_cities-2)
                    # C = tour_to_optimize[k]
                    # D = tour_to_optimize[(k+1)%num_cities] (this index (k+1)%num_cities is always valid)
    
                    node_A_val = tour_to_optimize[i]
                    node_B_val = tour_to_optimize[i+1]
                    node_C_val = tour_to_optimize[k]
                    node_D_val = tour_to_optimize[(k + 1) % num_cities]
    
                    # Cost of original edges
                    original_edges_len = graph_matrix[node_A_val][node_B_val] + graph_matrix[node_C_val][node_D_val]
                    
                    # Cost of new edges if we swap: (A,C) and (B,D)
                    new_edges_len = graph_matrix[node_A_val][node_C_val] + graph_matrix[node_B_val][node_D_val]
    
                    if new_edges_len < original_edges_len:
                        # Improvement found, apply the swap by reversing the segment tour_to_optimize[i+1...k]
                        segment_to_reverse = tour_to_optimize[i+1 : k+1]
                        segment_to_reverse.reverse()
                        
                        tour_to_optimize = tour_to_optimize[0 : i+1] + segment_to_reverse + tour_to_optimize[k+1 : num_cities]
                        
                        improved = True
                        # Using "first improvement" strategy: restart search from the beginning of the modified tour
                        break 
                if improved:
                    break 
        
        return tour_to_optimize
    ```

### 5. Program ID: 0fd07314-d331-4ee7-bf84-fb09f6ca230e (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 6871580e-750a-4fa8-8731-3e2ed983494a
    - Timestamp: 1747425144.19
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix:
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
    
        # Part 1: Nearest Neighbor to get a good initial tour
        # Try starting from each city and pick the best tour found.
        best_nn_tour = None
        min_nn_tour_length = float('inf')
    
        for start_node in range(num_cities):
            current_city = start_node
            tour = [current_city]
            visited = {current_city}
            current_tour_segment_length = 0.0
    
            while len(tour) < num_cities:
                next_city_candidate = -1
                min_dist_to_next = float('inf')
                
                for neighbor_idx in range(num_cities):
                    if neighbor_idx not in visited:
                        distance = graph_matrix[current_city][neighbor_idx]
                        if distance < min_dist_to_next:
                            min_dist_to_next = distance
                            next_city_candidate = neighbor_idx
                        elif distance == min_dist_to_next:
                            if next_city_candidate == -1 or neighbor_idx < next_city_candidate:
                                next_city_candidate = neighbor_idx
                
                if next_city_candidate == -1:
                    # This case should ideally not be reached in a complete graph
                    # as specified by problem constraints if len(tour) < num_cities.
                    break 
                
                current_tour_segment_length += min_dist_to_next
                tour.append(next_city_candidate)
                visited.add(next_city_candidate)
                current_city = next_city_candidate
            
            if len(tour) == num_cities: 
                final_edge_length = graph_matrix[tour[-1]][tour[0]]
                total_tour_length = current_tour_segment_length + final_edge_length
                
                if total_tour_length < min_nn_tour_length:
                    min_nn_tour_length = total_tour_length
                    best_nn_tour = list(tour) 
    
        if best_nn_tour is None:
            # Fallback if Nearest Neighbor somehow fails (e.g., graph not complete as expected)
            # Given problem constraints, this should not be hit for num_cities > 1.
            return list(range(num_cities)) 
    
        # Part 2: 2-opt refinement using "best improvement" strategy
        tour_to_optimize = list(best_nn_tour) 
        
        while True: 
            min_improvement_delta = 0 
            best_i_swap = -1
            best_k_swap = -1
    
            for i in range(num_cities - 1): 
                node_A = tour_to_optimize[i]
                node_B = tour_to_optimize[i+1] 
    
                for k in range(i + 1, num_cities): 
                    node_C = tour_to_optimize[k]
                    node_D = tour_to_optimize[(k + 1) % num_cities]
    
                    original_edges_len = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                    new_edges_len = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]
                    
                    current_delta = new_edges_len - original_edges_len
    
                    if current_delta < min_improvement_delta:
                        min_improvement_delta = current_delta
                        best_i_swap = i
                        best_k_swap = k
            
            if min_improvement_delta < 0: 
                i_to_swap = best_i_swap
                k_to_swap = best_k_swap
                
                segment_to_reverse = tour_to_optimize[i_to_swap+1 : k_to_swap+1]
                segment_to_reverse.reverse()
                
                tour_to_optimize = tour_to_optimize[0 : i_to_swap+1] + \
                                   segment_to_reverse + \
                                   tour_to_optimize[k_to_swap+1 : num_cities]
            else:
                break 
        
        return tour_to_optimize
    ```

### 6. Program ID: 57b791be-7080-41c3-bd84-5946e6df568a (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 1735484a-caa4-487d-842b-0646d6eabc53
    - Timestamp: 1747425144.21
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        
        if num_cities == 1:
            # This logic correctly handles graph_matrix = [[0]] (returns [0])
            # and graph_matrix = [[]] (returns []) due to the len check inside.
            if not graph_matrix[0]: # True for graph_matrix = [[]]
                 # This inner check is specific for [[]] where num_cities = 1 but graph_matrix[0] is empty.
                 if len(graph_matrix[0]) != num_cities : 
                     return [] 
            return [0]
    
        # Validate that graph_matrix[0] is a list and all rows have correct length.
        # As per problem spec, graph_matrix is a square matrix, so this should hold.
        # This check is more for robustness against malformed inputs not strictly following spec.
        if not isinstance(graph_matrix[0], list) or len(graph_matrix[0]) != num_cities:
            return [] # Malformed input
    
        # 1. Initial tour using Nearest Neighbor Heuristic
        start_node = 0 
        current_city = start_node
        tour = [current_city]
        visited = [False] * num_cities
        visited[current_city] = True
        num_visited = 1
    
        while num_visited < num_cities:
            next_city = -1
            min_dist = float('inf')
            
            for candidate_city in range(num_cities):
                if not visited[candidate_city]:
                    # Problem states distances are positive, graph_matrix[i][i]=0
                    # graph_matrix[current_city][candidate_city] should be > 0
                    distance = graph_matrix[current_city][candidate_city]
                    if distance < min_dist:
                        min_dist = distance
                        next_city = candidate_city
            
            if next_city == -1:
                # Fallback for incomplete graph or if all remaining distances are inf.
                # Should not happen given problem constraints (complete graph, positive distances).
                found_unvisited = False
                for i in range(num_cities):
                    if not visited[i]:
                        next_city = i
                        found_unvisited = True
                        break
                if not found_unvisited: 
                    break # All cities somehow marked visited, exit.
    
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city
            num_visited += 1
        
        if len(tour) != num_cities or len(set(tour)) != num_cities:
            # Fallback if NN tour construction failed to produce a valid permutation.
            return list(range(num_cities))
    
        # 2. Improve using 2-opt Heuristic
        best_tour = list(tour) 
        
        # Set a limit on the number of full improvement passes for 2-opt.
        # Using num_cities as the limit is a common heuristic.
        # For small N, 2-opt converges very quickly.
        # For N=2,3, 2-opt loops will do minimal/zero work or converge in 1 pass.
        # Limit of N passes for N cities is generally sufficient for significant improvement.
        pass_iteration_limit = num_cities
        
        current_pass = 0
        improvement_found_in_cycle = True 
    
        while improvement_found_in_cycle and current_pass < pass_iteration_limit :
            improvement_found_in_cycle = False 
            current_pass += 1
    
            for i in range(num_cities - 1): 
                for j in range(i + 2, num_cities): 
                    node_pi = best_tour[i]
                    node_pi1 = best_tour[i+1] 
                    node_pj = best_tour[j]   
                    node_pj1 = best_tour[(j + 1) % num_cities] # Handles wrap-around
    
                    cost_current_edges = graph_matrix[node_pi][node_pi1] + graph_matrix[node_pj][node_pj1]
                    cost_new_edges = graph_matrix[node_pi][node_pj] + graph_matrix[node_pi1][node_pj1]
    
                    if cost_new_edges < cost_current_edges:
                        # Improvement found. Reverse the segment best_tour[i+1...j]
                        segment_to_reverse = best_tour[i+1 : j+1]
                        segment_to_reverse.reverse() 
                        
                        best_tour = best_tour[0 : i+1] + segment_to_reverse + best_tour[j+1 : num_cities]
                        
                        improvement_found_in_cycle = True 
                        
                        # "First improvement" strategy: restart scans if an improvement is made.
                        break 
                if improvement_found_in_cycle:
                    break 
    
        return best_tour
    ```

### 7. Program ID: 59e329ce-7dfc-4a03-9db2-3765e96d9ea2 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Timestamp: 1747425144.22
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # 1. Nearest Neighbor Heuristic to get an initial tour
        # Try starting from each city and take the best initial tour from NN
        # This adds O(N) factor to NN, making it O(N^3) for this part,
        # or O(N^2) if NN is O(N) per start. NN is O(N^2) per start.
        # So this would be O(N^3). Let's stick to a single start for NN (city 0)
        # to keep NN part O(N^2).
        
        current_city_nn = 0  # Start at city 0 for NN
        tour_nn = [current_city_nn]
        visited_nn = {current_city_nn}
    
        temp_current_city = current_city_nn
        while len(tour_nn) < num_cities:
            next_city_candidate = -1
            min_dist_candidate = float('inf')
            
            for city_idx in range(num_cities):
                if city_idx not in visited_nn:
                    dist = graph_matrix[temp_current_city][city_idx]
                    if dist < min_dist_candidate:
                        min_dist_candidate = dist
                        next_city_candidate = city_idx
            
            # In a complete graph, next_city_candidate should always be found
            # if len(tour_nn) < num_cities.
            if next_city_candidate == -1:
                # Fallback if something is wrong (e.g. not a complete graph as assumed)
                # This part should ideally not be reached given problem constraints.
                # Add any remaining unvisited city to complete the tour.
                for city_idx in range(num_cities):
                    if city_idx not in visited_nn:
                        next_city_candidate = city_idx
                        break
                if next_city_candidate == -1: # Should be impossible if len(tour_nn) < num_cities
                    break # Safety break
    
            tour_nn.append(next_city_candidate)
            visited_nn.add(next_city_candidate)
            temp_current_city = next_city_candidate
        
        best_tour = list(tour_nn) 
    
        # 2. Improve the tour using 2-opt heuristic
        improved = True
        while improved:
            improved = False
            for i in range(num_cities):  # Index for the start of the segment to reverse (inclusive)
                for j in range(i + 1, num_cities):  # Index for the end of the segment to reverse (inclusive)
                                                    # Segment is best_tour[i...j]
    
                    # Reversing the whole tour (i=0, j=N-1) doesn't change its length.
                    # It's equivalent to traversing it in the opposite direction.
                    if i == 0 and j == num_cities - 1:
                        continue
    
                    # Current nodes involved in the two edges being swapped by reversing best_tour[i...j]:
                    # Edge 1 (before segment start): ( best_tour[(i-1+N)%N], best_tour[i] )
                    # Edge 2 (after segment end):   ( best_tour[j], best_tour[(j+1+N)%N] )
                    
                    # Values of the nodes:
                    node_before_segment_start_val = best_tour[(i - 1 + num_cities) % num_cities]
                    node_segment_start_val = best_tour[i]
                    node_segment_end_val = best_tour[j]
                    node_after_segment_end_val = best_tour[(j + 1) % num_cities]
                    
                    # Cost change calculation:
                    # (cost of new edges) - (cost of old edges)
                    # After reversing segment best_tour[i...j], the new edges are:
                    # New Edge 1': ( node_before_segment_start_val, node_segment_end_val )
                    # New Edge 2': ( node_segment_start_val, node_after_segment_end_val )
                    
                    current_edge_costs = (graph_matrix[node_before_segment_start_val][node_segment_start_val] + 
                                          graph_matrix[node_segment_end_val][node_after_segment_end_val])
                    
                    new_edge_costs = (graph_matrix[node_before_segment_start_val][node_segment_end_val] + 
                                      graph_matrix[node_segment_start_val][node_after_segment_end_val])
                    
                    cost_change = new_edge_costs - current_edge_costs
    
                    # If improvement is found (cost_change is negative)
                    if cost_change < -1e-9:  # Using a small epsilon for floating point comparisons
                        # Perform the 2-opt swap by reversing the segment best_tour[i...j]
                        segment_to_reverse = best_tour[i : j+1] # Python slice: elements from index i up to, but not including, j+1
                        segment_to_reverse.reverse()
                        
                        # Reconstruct the tour with the reversed segment
                        best_tour = best_tour[:i] + segment_to_reverse + best_tour[j+1:]
                        
                        improved = True
                        # Restart search for improvements from the beginning, as the tour has changed
                        break  # Exit j loop
                if improved:
                    break # Exit i loop, and restart the 'while improved' loop
                    
        return best_tour
    ```

### 8. Program ID: f45b4e01-aa09-4f79-93ce-c6f92acb7226 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 179d54a3-9c2e-49a4-9ff7-6f5d1c99de52
    - Timestamp: 1747425144.23
    - Code:
    ```python
    def solve_tsp_heuristic(graph_matrix):
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
        
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        if num_cities == 1:
            return [0]
        
        # Validate matrix structure (assuming square and complete based on problem)
        # Example: if not all(isinstance(row, list) and len(row) == num_cities for row in graph_matrix): return list(range(num_cities))
    
        # Step 1: Generate initial tour using Nearest Neighbor heuristic
        # Start at city 0. This can be randomized or iterated for better results,
        # but for simplicity, a fixed start is used here.
        initial_tour = []
        current_city = 0 
        initial_tour.append(current_city)
        visited = {current_city}
    
        while len(initial_tour) < num_cities:
            next_city_candidate = -1
            min_dist_to_next = float('inf')
    
            for neighbor_idx in range(num_cities):
                if neighbor_idx not in visited:
                    dist = graph_matrix[current_city][neighbor_idx]
                    if dist < min_dist_to_next:
                        min_dist_to_next = dist
                        next_city_candidate = neighbor_idx
            
            if next_city_candidate == -1:
                # This case should ideally not be reached in a complete graph with positive distances.
                # Fallback: add remaining unvisited cities in their numerical order.
                for i_fallback in range(num_cities):
                    if i_fallback not in visited:
                        initial_tour.append(i_fallback)
                        visited.add(i_fallback) # Ensure visited set is consistent
                break # Exit while loop as tour is now full (or was supposed to be)
                
            initial_tour.append(next_city_candidate)
            visited.add(next_city_candidate)
            current_city = next_city_candidate
        
        # Step 2: Improve the tour using the 2-opt heuristic
    
        current_best_tour = list(initial_tour) # Start with the NN tour
        
        improved = True
        while improved:
            improved = False
            
            # Iterate over all pairs of non-adjacent edges for a potential 2-opt swap
            for i_loop_idx in range(num_cities): 
                # This i_loop_idx is an index into current_best_tour.
                # First edge is (current_best_tour[i_loop_idx], current_best_tour[(i_loop_idx + 1) % num_cities])
                
                # k_loop_idx_offset ensures that k_loop_idx defines an edge non-adjacent to the first one.
                # The second edge starts at current_best_tour[k_loop_idx].
                # k_loop_idx must not be i_loop_idx or (i_loop_idx + 1) % num_cities.
                # So, k_loop_idx starts cyclically from (i_loop_idx + 2).
                for k_loop_idx_offset in range(2, num_cities): 
                    k_loop_idx = (i_loop_idx + k_loop_idx_offset) % num_cities
    
                    # Define the four critical node indices in the current_best_tour list
                    # Edge 1 is (tour[idx_A], tour[idx_B])
                    # Edge 2 is (tour[idx_C], tour[idx_D])
                    idx_A = i_loop_idx
                    idx_B = (i_loop_idx + 1) % num_cities
                    idx_C = k_loop_idx
                    idx_D = (k_loop_idx + 1) % num_cities
                    
                    # Retrieve the actual city IDs for these nodes
                    node_A = current_best_tour[idx_A]
                    node_B = current_best_tour[idx_B]
                    node_C = current_best_tour[idx_C]
                    node_D = current_best_tour[idx_D]
    
                    # Calculate cost of current edges vs. new edges if swapped
                    # Current edges: (A,B) and (C,D)
                    cost_original_edges = graph_matrix[node_A][node_B] + graph_matrix[node_C][node_D]
                    
                    # New edges if swapped: (A,C) and (B,D)
                    # This swap implies reversing the path segment between B and C (inclusive of B and C if thinking path segments)
                    # More precisely, the tour segment from tour[idx_B] to tour[idx_C] is reversed.
                    cost_new_edges = graph_matrix[node_A][node_C] + graph_matrix[node_B][node_D]
    
                    if cost_new_edges < cost_original_edges:
                        # Improvement found, perform the 2-opt swap
                        new_tour_after_swap = list(current_best_tour) # Make a copy to modify
                        
                        # Reverse the segment of the tour from new_tour_after_swap[idx_B] to new_tour_after_swap[idx_C]
                        # Store elements of the segment to be reversed
                        segment_nodes_to_reverse = []
                        current_segment_fill_idx = idx_B
                        while True:
                            segment_nodes_to_reverse.append(new_tour_after_swap[current_segment_fill_idx])
                            if current_segment_fill_idx == idx_C:
                                break
                            current_segment_fill_idx = (current_segment_fill_idx + 1) % num_cities
                        
                        segment_nodes_to_reverse.reverse() # Reverse the collected segment
                        
                        # Place the reversed segment back into the tour
                        current_placement_idx = idx_B
                        for city_node_in_segment in segment_nodes_to_reverse:
                            new_tour_after_swap[current_placement_idx] = city_node_in_segment
                            if current_placement_idx == idx_C: # Stop once the segment is filled
                                break
                            current_placement_idx = (current_placement_idx + 1) % num_cities
                        
                        current_best_tour = new_tour_after_swap # Update the tour
                        improved = True # Mark that an improvement was made
                        
                        # Break from inner loops to restart the 2-opt scan from the beginning with the new tour
                        break # Exit k_loop_idx_offset loop
                if improved:
                    break # Exit i_loop_idx loop (and thus restart the 'while improved' loop)
                    
        return current_best_tour
    ```