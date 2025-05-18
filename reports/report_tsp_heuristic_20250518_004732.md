# Mini-Evolve Run Report
Generated: 2025-05-18 00:47:32
Problem: tsp_heuristic
Database: db/program_database.db

---

## I. Overall Statistics
- Total programs in database: 17
- Valid programs: 17
- Invalid programs: 0
- Percentage valid: 100.00%
- Max score (valid programs): 0.0447
- Min score (valid programs): 0.0437
- Average score (valid programs): 0.0447
- Generations spanned: 0 to 4

## II. Best Program(s)
### Top Scorer:
- Program ID: dfa13b9b-c98d-4002-9547-1327c851815f
- Score: 0.0447
- Generation Discovered: 4
- Parent ID: bfc7e9e3-7712-467c-8b48-055800b2b5b0
- Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.3659640098921955, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
```python
import sys

def calculate_tour_length(graph_matrix, tour):
    """Calculates the total length of a given tour including the return to the start."""
    num_cities = len(graph_matrix)
    if not tour or len(tour) != num_cities:
        # This case should ideally not be reached for a valid tour covering all cities
        # in a complete graph, but return a very large value as a safeguard.
        return sys.float_info.max

    total_length = 0
    try:
        for i in range(num_cities - 1):
            total_length += graph_matrix[tour[i]][tour[i+1]]
        # Add distance back to the start city to complete the cycle
        total_length += graph_matrix[tour[-1]][tour[0]]
    except IndexError:
        # Handle potential index errors if tour indices are out of bounds
        return sys.float_info.max
    except TypeError:
        # Handle potential type errors if graph_matrix or tour contains invalid types
         return sys.float_info.max


    return total_length

def two_opt_swap(tour, i, k):
    """Performs a 2-opt swap on the tour between indices i and k (exclusive of k+1)."""
    # Assumes i < k
    new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
    return new_tour

def two_opt_optimize(graph_matrix, initial_tour):
    """
    Applies the 2-opt local search optimization algorithm to an initial tour.

    Args:
        graph_matrix: The adjacency matrix of distances.
        initial_tour: A list of city indices representing the starting tour.

    Returns:
        A list of city indices representing the optimized tour.
    """
    num_cities = len(graph_matrix)
    current_tour = list(initial_tour) # Create a mutable copy
    improved = True

    # Continue iterating as long as improvements are being made
    while improved:
        improved = False
        best_length = calculate_tour_length(graph_matrix, current_tour)

        # Iterate through all possible 2-opt swaps (pairs of non-adjacent edges)
        # We consider edges (i, i+1) and (k, k+1) for 0 <= i < k-1 <= num_cities-2
        # When swapping, we reverse the segment between i+1 and k.
        # The indices in the tour list are 0-based. An edge is between tour[i] and tour[i+1].
        # The 2-opt swap involves reversing the segment from index i+1 up to k.
        # The indices i and k in the loop below refer to the start of the segments to be swapped.
        # The actual edges are (tour[i], tour[i+1]) and (tour[k], tour[k+1])
        # The segment reversed is from tour[i+1] to tour[k].
        # The new edges become (tour[i], tour[k]) and (tour[i+1], tour[k+1])

        for i in range(num_cities - 1):
            for k in range(i + 2, num_cities): # k must be at least i+2 for a valid segment reversal
                # Perform the 2-opt swap
                new_tour = two_opt_swap(current_tour, i + 1, k) # indices i+1 to k inclusive are reversed

                # Calculate the length of the new tour
                new_length = calculate_tour_length(graph_matrix, new_tour)

                # If the new tour is shorter, update and continue the process
                if new_length < best_length:
                    current_tour = new_tour
                    best_length = new_length
                    improved = True
        # Note: The loop structure here performs one pass of potential swaps.
        # If an improvement is found, 'improved' is set to True, and the while loop continues
        # for another full pass until no improvements are found in a complete pass.

    return current_tour


def solve_tsp_heuristic(graph_matrix):
    """
    Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
    with multiple starting cities, followed by a 2-opt local search optimization.

    Args:
        graph_matrix: A list of lists of numbers (integers or floats) representing
                      the adjacency matrix of a complete graph. graph_matrix[i][j]
                      is the distance from city i to city j. Assumes a square matrix
                      where len(graph_matrix) is the number of cities. Distances
                      are positive, and graph_matrix[i][i] is 0.

    Returns:
        A list of node indices (integers from 0 to N-1) representing the order
        of cities in the found tour. The returned list is a permutation of [0...N-1].
        Returns an empty list for invalid input or zero cities. Returns [0] for a single city.
    """
    # Input validation
    if not graph_matrix or not isinstance(graph_matrix, list):
        return []

    num_cities = len(graph_matrix)

    if num_cities == 0:
        return []

    # Basic check for square matrix (assuming first row determines width)
    if any(len(row) != num_cities for row in graph_matrix):
         # Not a square matrix
         return []

    if num_cities == 1:
        return [0] # Tour is just the single city

    best_initial_tour = []
    min_initial_length = sys.float_info.max

    # --- Step 1: Generate an initial tour using Nearest Neighbor from multiple starts ---
    for start_city in range(num_cities):
        current_tour = [start_city]
        visited = {start_city}
        current_city = start_city

        while len(current_tour) < num_cities:
            next_city = -1
            min_dist = sys.float_info.max

            for city_idx in range(num_cities):
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    # Find the true minimum positive distance
                    if distance < min_dist:
                         min_dist = distance
                         next_city = city_idx

            if next_city != -1:
                current_tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
            else:
                 # Should not happen in a complete graph with positive distances
                 break

        # If a full tour was constructed (visited all cities)
        if len(current_tour) == num_cities:
            current_total_length = calculate_tour_length(graph_matrix, current_tour)

            if current_total_length < min_initial_length:
                min_initial_length = current_total_length
                best_initial_tour = current_tour

    # If no valid initial tour was found (shouldn't happen for complete graph > 1), return empty
    if not best_initial_tour:
        return []

    # --- Step 2: Apply 2-opt optimization to the best initial tour ---
    final_optimized_tour = two_opt_optimize(graph_matrix, best_initial_tour)

    return final_optimized_tour
```

## III. Top 5 Programs (by Score)

### 1. Program ID: dfa13b9b-c98d-4002-9547-1327c851815f
    - Score: 0.0447
    - Generation: 4
    - Parent ID: bfc7e9e3-7712-467c-8b48-055800b2b5b0
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.3659640098921955, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    import sys
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour including the return to the start."""
        num_cities = len(graph_matrix)
        if not tour or len(tour) != num_cities:
            # This case should ideally not be reached for a valid tour covering all cities
            # in a complete graph, but return a very large value as a safeguard.
            return sys.float_info.max
    
        total_length = 0
        try:
            for i in range(num_cities - 1):
                total_length += graph_matrix[tour[i]][tour[i+1]]
            # Add distance back to the start city to complete the cycle
            total_length += graph_matrix[tour[-1]][tour[0]]
        except IndexError:
            # Handle potential index errors if tour indices are out of bounds
            return sys.float_info.max
        except TypeError:
            # Handle potential type errors if graph_matrix or tour contains invalid types
             return sys.float_info.max
    
    
        return total_length
    
    def two_opt_swap(tour, i, k):
        """Performs a 2-opt swap on the tour between indices i and k (exclusive of k+1)."""
        # Assumes i < k
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        return new_tour
    
    def two_opt_optimize(graph_matrix, initial_tour):
        """
        Applies the 2-opt local search optimization algorithm to an initial tour.
    
        Args:
            graph_matrix: The adjacency matrix of distances.
            initial_tour: A list of city indices representing the starting tour.
    
        Returns:
            A list of city indices representing the optimized tour.
        """
        num_cities = len(graph_matrix)
        current_tour = list(initial_tour) # Create a mutable copy
        improved = True
    
        # Continue iterating as long as improvements are being made
        while improved:
            improved = False
            best_length = calculate_tour_length(graph_matrix, current_tour)
    
            # Iterate through all possible 2-opt swaps (pairs of non-adjacent edges)
            # We consider edges (i, i+1) and (k, k+1) for 0 <= i < k-1 <= num_cities-2
            # When swapping, we reverse the segment between i+1 and k.
            # The indices in the tour list are 0-based. An edge is between tour[i] and tour[i+1].
            # The 2-opt swap involves reversing the segment from index i+1 up to k.
            # The indices i and k in the loop below refer to the start of the segments to be swapped.
            # The actual edges are (tour[i], tour[i+1]) and (tour[k], tour[k+1])
            # The segment reversed is from tour[i+1] to tour[k].
            # The new edges become (tour[i], tour[k]) and (tour[i+1], tour[k+1])
    
            for i in range(num_cities - 1):
                for k in range(i + 2, num_cities): # k must be at least i+2 for a valid segment reversal
                    # Perform the 2-opt swap
                    new_tour = two_opt_swap(current_tour, i + 1, k) # indices i+1 to k inclusive are reversed
    
                    # Calculate the length of the new tour
                    new_length = calculate_tour_length(graph_matrix, new_tour)
    
                    # If the new tour is shorter, update and continue the process
                    if new_length < best_length:
                        current_tour = new_tour
                        best_length = new_length
                        improved = True
            # Note: The loop structure here performs one pass of potential swaps.
            # If an improvement is found, 'improved' is set to True, and the while loop continues
            # for another full pass until no improvements are found in a complete pass.
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        with multiple starting cities, followed by a 2-opt local search optimization.
    
        Args:
            graph_matrix: A list of lists of numbers (integers or floats) representing
                          the adjacency matrix of a complete graph. graph_matrix[i][j]
                          is the distance from city i to city j. Assumes a square matrix
                          where len(graph_matrix) is the number of cities. Distances
                          are positive, and graph_matrix[i][i] is 0.
    
        Returns:
            A list of node indices (integers from 0 to N-1) representing the order
            of cities in the found tour. The returned list is a permutation of [0...N-1].
            Returns an empty list for invalid input or zero cities. Returns [0] for a single city.
        """
        # Input validation
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
    
        # Basic check for square matrix (assuming first row determines width)
        if any(len(row) != num_cities for row in graph_matrix):
             # Not a square matrix
             return []
    
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        best_initial_tour = []
        min_initial_length = sys.float_info.max
    
        # --- Step 1: Generate an initial tour using Nearest Neighbor from multiple starts ---
        for start_city in range(num_cities):
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max
    
                for city_idx in range(num_cities):
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Find the true minimum positive distance
                        if distance < min_dist:
                             min_dist = distance
                             next_city = city_idx
    
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # Should not happen in a complete graph with positive distances
                     break
    
            # If a full tour was constructed (visited all cities)
            if len(current_tour) == num_cities:
                current_total_length = calculate_tour_length(graph_matrix, current_tour)
    
                if current_total_length < min_initial_length:
                    min_initial_length = current_total_length
                    best_initial_tour = current_tour
    
        # If no valid initial tour was found (shouldn't happen for complete graph > 1), return empty
        if not best_initial_tour:
            return []
    
        # --- Step 2: Apply 2-opt optimization to the best initial tour ---
        final_optimized_tour = two_opt_optimize(graph_matrix, best_initial_tour)
    
        return final_optimized_tour
    ```

### 2. Program ID: b55e179b-7753-4725-b584-b520f6c70e29
    - Score: 0.0447
    - Generation: 4
    - Parent ID: 2b19831b-d3da-477a-a4c6-56818885d642
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 1.6023510252125561, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            # If num_cities is 1, the length is graph_matrix[0][0] which is 0.
            if num_cities == 1:
                 if graph_matrix and len(graph_matrix) > 0:
                     return graph_matrix[tour[0]][tour[0]] # Should be 0
                 else:
                     return 0.0
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            # Added check for graph_matrix dimensions
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix) and \
               len(graph_matrix[current_city]) > next_city:
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour or graph matrix issue
                 # print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        # Added check for graph_matrix dimensions
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix) and \
           len(graph_matrix[last_city]) > first_city:
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour or graph matrix issue
            # print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        best_tour = list(current_tour)
        best_distance = calculate_tour_length(graph_matrix, current_tour)
    
        improved = True
        while improved:
            improved = False
            for i in range(N - 1): # Index of the first city of the first edge (0 to N-2)
                for j in range(i + 1, N): # Index of the first city of the second edge (i+1 to N-1)
                    # The segment to reverse is between index i+1 and j (inclusive of j).
                    # This swaps edges (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
    
                    # If j is i+1, the segment is only one city long (current_tour[i+1]). Reversing does nothing.
                    # Skip adjacent edges.
                    if j == i + 1:
                        continue
    
                    # Create a new tour by performing the 2-opt swap
                    # Take the tour from the start up to i
                    # Reverse the segment from i+1 up to j
                    # Take the tour from j+1 to the end
                    new_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
    
                    # Calculate the distance of the new tour
                    new_distance = calculate_tour_length(graph_matrix, new_tour)
    
                    # If the new tour is shorter, update the best tour and distance
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improved = True
                        # Found an improvement, restart the search for improvements
                        # Break from inner loops to restart the outer while loop
                        break
                if improved:
                    break
    
            # If an improvement was made in this pass, update current_tour to the best found tour
            # and continue the while loop.
            if improved:
                current_tour = list(best_tour) # Make a new copy for the next iteration
    
        return best_tour
    
    
    def solve_tsp_heuristic(graph_matrix: list[list[float]]) -> list[int]:
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not all(isinstance(row, list) for row in graph_matrix):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        # We return [0, 1] as the sequence.
        if num_cities == 2:
            # Basic check to ensure graph_matrix is at least 2x2
            if len(graph_matrix[0]) < 2 or len(graph_matrix[1]) < 2:
                 # Invalid matrix for 2 cities
                 return []
            # The tour sequence is just the cities in some order, e.g., [0, 1].
            # The calculate_tour_length would add d(0,1) + d(1,0).
            return [0, 1]
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and the distance is valid
                # Added check for matrix column size
                if city_idx not in visited and city_idx < len(graph_matrix[current_city]):
                    distance = graph_matrix[current_city][city_idx]
                    # Assuming distances are positive as per problem description, and not infinity
                    if distance < min_dist and distance >= 0: # Ensure positive or zero distance
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found with a valid distance.
            # This should only happen if all cities are visited or if the graph is not complete/valid.
            # In a complete graph, this implies an issue.
            if next_city == -1:
                 # Could indicate a disconnected graph or invalid matrix entries (e.g., all inf).
                 # Return the partial tour found or an empty list if no progress could be made.
                 # Given the problem assumes a complete graph with positive distances, this is unexpected.
                 # Returning the partial tour is better than crashing or infinite loop.
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 0 (handled 0 already)
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with a valid, complete graph.
             # Return the potentially incomplete tour.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        # For N >= 3, apply 2-opt.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 3. Program ID: b1077a78-71ee-4bc3-be19-18c97b3715cc
    - Score: 0.0447
    - Generation: 4
    - Parent ID: 2b19831b-d3da-477a-a4c6-56818885d642
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.355839009396732, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix):
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour
                 # print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix):
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour
            # print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        current_tour_length = calculate_tour_length(graph_matrix, current_tour)
    
        improved = True
        while improved:
            improved = False
            best_delta = 0 # We are looking for negative delta (improvement)
    
            # Iterate over all pairs of indices i and j in the tour (0 <= i < j <= N-1).
            # These indices define a segment tour[i+1 ... j].
            # Reversing this segment swaps edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The indices for edges must wrap around (modulo N).
            # We iterate through all possible pairs of non-adjacent edges to swap.
            # Edges are defined by their starting city index in the tour.
            # Consider edges starting at index i and index j.
            # Edge 1: (tour[i], tour[(i+1)%N])
            # Edge 2: (tour[j], tour[(j+1)%N])
            # We reverse the segment of the tour *between* these two edges.
            # The segment is from index (i+1)%N up to index j in the original tour list order.
            # The standard 2-opt swap involves selecting two indices i and j (0 <= i < j <= N-1)
            # and reversing the segment of the tour from index i+1 up to index j.
            # This swaps edges (tour[i], tour[i+1]) and (tour[j+1], tour[j]) (indices mod N).
            # Note: The indices i and j in the loops refer to the position in the tour list.
            # The cities are current_tour[i] and current_tour[j].
            # The edges being considered are (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
            # The 2-opt move reverses the segment current_tour[i+1...j].
            # This effectively replaces edges (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N])
            # with (current_tour[i], current_tour[j]) and (current_tour[i+1], current_tour[(j+1)%N]).
            # Let A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
            # The swap replaces AB + CD with AC + BD.
            # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            # These indices define the endpoints of the segment to be reversed.
            # The segment is from index i+1 up to index j (inclusive).
            # The edges involved are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The pairs (i, j) iterate over the potential edges to 'uncross' or swap.
            # i is the index of the city *before* the segment starts.
            # j is the index of the city *at the end* of the segment.
            # The segment to reverse is tour[i+1 ... j].
            # The edges being swapped are (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (using linear indexing for clarity in explanation here, but modulo N for the last edge).
            # Let's use the standard 2-opt indices i and j where 0 <= i < j <= N-1.
            # The points involved are p1=tour[i], p2=tour[i+1], p3=tour[j], p4=tour[(j+1)%N].
            # We swap edges (p1, p2) and (p3, p4) with (p1, p3) and (p2, p4).
            # This is achieved by reversing the segment from index i+1 to j.
    
            for i in range(N - 2): # i goes from 0 to N-3
                for j in range(i + 1, N - 1): # j goes from i+1 to N-2
                    # The segment to reverse is tour[i+1 ... j].
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # Cities involved in swap:
                    # A = current_tour[i]
                    # B = current_tour[i+1]
                    # C = current_tour[j+1] # City *after* the segment end in the original tour
                    # D = current_tour[(j+2)%N] # City after C in the original tour
    
                    # This standard 2-opt loop structure considers pairs of non-adjacent edges
                    # defined by indices i and j (0 <= i < j-1 <= N-2).
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (modulo N).
                    # Let A = current_tour[i], B = current_tour[(i+1)%N], C = current_tour[j], D = current_tour[(j+1)%N].
                    # We check delta for swapping (A, B) and (C, D) with (A, C) and (B, D).
                    # This corresponds to reversing the segment between B and C.
                    # The segment to reverse is from index i+1 up to index j.
                    # The cities are current_tour[i+1], current_tour[i+2], ..., current_tour[j].
    
                    # The loop indices i and j typically refer to the points *before* the edges being swapped.
                    # Edge 1: (tour[i], tour[i+1])
                    # Edge 2: (tour[j], tour[j+1]) (with wrap-around for j+1)
                    # We need to iterate over all pairs of non-adjacent edges.
                    # Let's use indices i and k in the tour list, 0 <= i < k <= N-1.
                    # Edge 1 starts at tour[i]. Edge 2 starts at tour[k].
                    # Edge 1: (tour[i], tour[(i+1)%N])
                    # Edge 2: (tour[k], tour[(k+1)%N])
                    # We consider swapping these edges if they are not adjacent.
                    # Adjacent means (i+1)%N == k or (k+1)%N == i.
    
                    # Let's iterate over all pairs of indices (i, j) in the tour list,
                    # where 0 <= i < j <= N-1.
                    # i and j define the segment to reverse: tour[i+1 ... j].
                    # The edges affected are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                    # Cities: A=tour[i], B=tour[i+1], C=tour[j], D=tour[(j+1)%N].
                    # Check delta for swapping (A, B) and (C, D) with (A, C) and (B, D).
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                    # Correct indices for 2-opt swap:
                    # Choose two indices i and j from the tour, 0 <= i < j <= N-1.
                    # Reverse the segment current_tour[i+1 ... j].
                    # The edges being swapped are (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
                    # Cities: A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
                    # The original edges are (A, B) and (C, D).
                    # The new edges after reversing segment [i+1...j] are (A, C) and (B, D).
                    # Need to handle wrap-around correctly.
                    # The indices i and j define the endpoints of the segment *outside* the reversal.
                    # Reversing tour[i+1 : j+1] swaps edge (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # Let's use indices i and j such that 0 <= i < j-1 <= N-2.
                    # This ensures the segment [i+1...j] has at least one city.
                    # i goes from 0 up to N-3.
                    # j goes from i+2 up to N-1.
                    # This considers pairs of non-consecutive edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
    
                    # Let's refine the loops based on standard 2-opt implementation.
                    # Iterate all pairs of indices i and j such that 0 <= i < j < N.
                    # These indices define the points where the tour is 'cut'.
                    # The segments are tour[0..i], tour[i+1..j], tour[j+1..N-1].
                    # Reversing tour[i+1..j] creates a new tour.
                    # The edges changed are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # New edges are (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
                    # Need to handle wrap-around for the last edge (tour[N-1], tour[0]).
                    # A standard way is to iterate i from 0 to N-2 and j from i+1 to N-1.
                    # The segment to reverse is from index i+1 to j.
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                    # Let i_idx = i, j_idx = j.
                    # Cities: A = current_tour[i_idx], B = current_tour[(i_idx+1)%N], C = current_tour[j_idx], D = current_tour[(j_idx+1)%N].
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                    # Iterate over all possible first points i (0 to N-1)
                    for i_idx in range(N):
                        # Iterate over all possible second points j (i+1 to N-1), handling wrap-around
                        # A standard 2-opt swap considers reversing the segment between two edges.
                        # Let the edges be (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
                        # We iterate over all pairs of indices i and j (0 <= i < j <= N-1)
                        # and consider reversing the segment tour[i+1...j].
                        # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                        # Cities involved: A=tour[i], B=tour[i+1], C=tour[j], D=tour[(j+1)%N].
                        # New edges: (A, C) and (B, D).
                        # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                        # The standard loop structure for 2-opt:
                        # Iterate i from 0 to N-2
                        # Iterate j from i+1 to N-1
                        # This defines the segment tour[i+1...j] to reverse.
                        # The edges are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                        # Cities: A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
                        # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                        for i in range(N - 1): # Edge 1 starts at index i
                            for j in range(i + 1, N): # Edge 2 starts at index j
                                # Indices i and j in the loop define the start points of the two edges being considered for swapping.
                                # Edge 1: (current_tour[i], current_tour[(i+1)%N])
                                # Edge 2: (current_tour[j], current_tour[(j+1)%N])
    
                                # These edges are adjacent if (i+1)%N == j or (j+1)%N == i.
                                # If i < j, adjacency means j == i+1 or (j+1)%N == i (only if j=N-1 and i=0).
                                # Swapping adjacent edges doesn't change the tour (reversing a segment of length 1 or 0).
                                # The standard 2-opt swap reverses the segment between the *end* of the first edge and the *start* of the second edge.
                                # If edges are (A, B) and (C, D), where B is A's successor and D is C's successor in the tour,
                                # the swap considers reversing the path from B to C.
                                # This replaces (A, B) and (C, D) with (A, C) and (B, D).
                                # Cities: A = current_tour[i], B = current_tour[(i+1)%N], C = current_tour[j], D = current_tour[(j+1)%N].
                                # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                                # Let's use the common implementation iterating pairs of indices (i, j)
                                # where 0 <= i < j <= N-1. Reversing segment tour[i+1...j].
                                # Edges swapped: (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                                # Cities: A=tour[i], B=tour[i+1], C=tour[j], D=tour[j+1]. (linear indices)
                                # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                                # Correct indices for reversing segment tour[i+1 ... j] (inclusive endpoints)
                                # Cities involved:
                                # p1 = current_tour[i]
                                # p2 = current_tour[i+1]
                                # p3 = current_tour[j]
                                # p4 = current_tour[(j+1)%N] # Handles wrap around
    
                                # Calculate change in length
                                # Old edges: (p1, p2) and (p3, p4)
                                # New edges: (p1, p3) and (p2, p4)
                                # Delta = d(p1, p3) + d(p2, p4) - d(p1, p2) - d(p3, p4)
                                delta = (graph_matrix[current_tour[i]][current_tour[j]] +
                                         graph_matrix[current_tour[(i+1)%N]][current_tour[(j+1)%N]] -
                                         graph_matrix[current_tour[i]][current_tour[(i+1)%N]] -
                                         graph_matrix[current_tour[j]][current_tour[(j+1)%N]])
    
    
                                # If delta is negative, we found an improvement
                                if delta < -1e-9: # Use tolerance for float comparison
                                    # Perform the swap by reversing the segment from i+1 to j
                                    # current_tour[i+1 ... j] needs to be reversed.
                                    # This corresponds to reversing the slice current_tour[i+1 : j+1].
                                    temp_segment = current_tour[i+1 : j+1]
                                    temp_segment.reverse()
                                    current_tour[i+1 : j+1] = temp_segment
    
                                    current_tour_length += delta
                                    improved = True
                                    # Restart search from the beginning
                                    break # Break j loop
                            if improved:
                                break # Break i loop
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not graph_matrix[0] or not isinstance(graph_matrix[0], list):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        if num_cities == 2:
            # The tour [0, 1] visits both cities. Returning to 0 completes the cycle.
            return [0, 1]
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city from the current city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and not the current city itself
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                     # Ensure distance is valid (non-negative and finite)
                    if distance >= 0 and distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found.
            # This should only happen if all cities are visited, which is checked by the while loop condition.
            # However, as a safeguard, if it happens unexpectedly (e.g., graph not complete or disconnected), break.
            if next_city == -1:
                 # This indicates an issue, likely with graph structure assumptions or logic error.
                 # Return the partial tour found.
                 # print(f"Warning: Nearest Neighbor failed to find next city. Partial tour: {initial_tour}")
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph with positive distances, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with complete graph.
             # Return the potentially incomplete tour.
             # print(f"Warning: Initial tour is incomplete. Expected {num_cities}, got {len(initial_tour)}")
             return initial_tour # Return whatever was constructed
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 4. Program ID: 502cf498-71fa-4e50-9c2e-80b24416eae4
    - Score: 0.0447
    - Generation: 3
    - Parent ID: 5b4559cd-c51d-4b11-9acf-c9b6f8fd0eac
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.2137610358186066, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            total_length += graph_matrix[current_city][next_city]
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        total_length += graph_matrix[last_city][first_city]
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        best_tour = list(current_tour)
        best_distance = calculate_tour_length(graph_matrix, best_tour)
    
        improved = True
        while improved:
            improved = False
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            # These indices define the segment (current_tour[i+1 ... j]) to reverse.
            # Reversing this segment swaps edges (current_tour[i], current_tour[i+1])
            # and (current_tour[j], current_tour[j+1]) (indices mod N).
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # The indices i and j define the segment current_tour[i+1 : j+1]
                    # Note: The standard 2-opt swap involves reversing the path *between*
                    # two edges. If we pick edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]),
                    # the segment reversed is tour[i+1...j].
                    # The indices i and j here correspond to the *start* indices of the two edges
                    # being potentially swapped.
                    # Edge 1: (current_tour[i], current_tour[(i + 1) % N])
                    # Edge 2: (current_tour[j], current_tour[(j + 1) % N])
    
                    # Skip adjacent edges (reversing segment of size 1 or N-1)
                    # For i < j, adjacent edges occur when j == i + 1 or (i==0 and j==N-1 for the cycle edge)
                    # The loop structure i < j covers j=i+1. The wrap-around case (0, N-1)
                    # corresponds to i=0, j=N-1. Reversing tour[1:N] swaps (tour[0],tour[1]) and (tour[N-1],tour[0]).
                    # This is handled by the indices i and j.
                    # We are swapping edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # by reversing the segment tour[i+1...j].
                    # The cities involved are A=tour[i], B=tour[i+1], C=tour[j], D=tour[j+1] (indices mod N).
                    # Old edges: (A, B) and (C, D). New edges: (A, C) and (B, D).
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D)
    
                    # Correct indices for the cities involved in the potential swap:
                    # Edge 1: (current_tour[i], current_tour[(i + 1) % N])
                    # Edge 2: (current_tour[j], current_tour[(j + 1) % N])
                    # The segment to reverse is between index i and index j in the tour list.
                    # The segment is current_tour[i+1 ... j].
                    # The cities are tour[i], tour[i+1], ..., tour[j], tour[j+1] (indices mod N)
                    # The edges being swapped are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
    
                    # Consider indices i and j in the tour list (0 <= i < j <= N-1)
                    # This swap replaces edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # with (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                    # by reversing the segment tour[i+1 ... j].
    
                    # Cities involved in the swap:
                    A = current_tour[i]
                    B = current_tour[(i + 1) % N] # Note: Use modulo for the end of the segment
                    C = current_tour[j]
                    D = current_tour[(j + 1) % N] # Note: Use modulo for the end of the segment
    
                    # Calculate the change in distance
                    # If we swap edges (A, B) and (C, D) with (A, C) and (B, D)
                    # This corresponds to reversing the segment from B to C.
                    # The segment in the list is tour[i+1 ... j].
                    # If we reverse tour[i+1 ... j], the new edges become (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
                    # This is correct for the 2-opt swap defined by reversing the segment tour[i+1...j].
    
                    current_dist_edges = graph_matrix[A][B] + graph_matrix[C][D]
                    new_dist_edges = graph_matrix[A][C] + graph_matrix[B][D]
    
                    delta = new_dist_edges - current_dist_edges
    
                    # Optimization: only consider swaps if delta < 0
                    if delta < 0:
                        # Perform the 2-opt swap by reversing the segment current_tour[i+1 : j+1]
                        # This reverses the list segment from index i+1 up to index j (inclusive).
                        current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
    
                        # Update best distance and flag improvement
                        best_distance += delta
                        improved = True
                        # Since an improvement was made, restart the search for improvements
                        # from the beginning of the outer while loop.
                        # Breaking both loops and letting the while loop condition re-evaluate
                        # is a standard way to implement this.
                        break # Break the inner j loop
                if improved:
                    break # Break the outer i loop
    
            # After iterating through all pairs (i, j) in one pass, if an improvement was made,
            # the `improved` flag is True, and the while loop continues for another pass.
            # If no improvement was made in a full pass, `improved` remains False, and the loop terminates.
    
        return current_tour # Return the locally optimal tour
    
    def solve_tsp_heuristic(graph_matrix: list[list[float]]) -> list[int]:
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Handle case for N=2 specially for NN, although handled by 2-opt N<3 check
        if num_cities == 2:
            # If only two cities, the only tour starting at 0 is [0, 1]
            for city_idx in range(num_cities):
                if city_idx != start_city:
                    initial_tour.append(city_idx)
                    visited.add(city_idx)
                    break
        else: # num_cities > 2, proceed with standard NN
            while len(initial_tour) < num_cities:
                next_city = -1
                min_dist = float('inf')
    
                for city_idx in range(num_cities):
                    # Check if city_idx is unvisited and the distance is the minimum found so far
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Assuming distances are positive as per problem description
                        if distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # This case should not happen in a complete graph with N > len(tour)
                # If somehow stuck (e.g., graph not truly complete), break
                if next_city == -1:
                     break
    
                initial_tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected graph structure or issue.
             # Return the partial tour found.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 5. Program ID: 2b19831b-d3da-477a-a4c6-56818885d642
    - Score: 0.0447
    - Generation: 3
    - Parent ID: 5b4559cd-c51d-4b11-9acf-c9b6f8fd0eac
    - Evaluation Details: `{"score": 0.04471153846153846, "is_valid": true, "error_message": null, "execution_time_ms": 2.3707430227659643, "tour_length": 80, "tour": [0, 1, 3, 2], "details": {"simple_4_city": {"tour": [0, 1, 3, 2], "length": 80, "is_valid": true, "error": null}, "line_5_city": {"tour": [0, 1, 2, 3, 4], "length": 13, "is_valid": true, "error": null}}}`
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix):
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour
                 print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix):
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour
            print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
    
        improved = True
        while improved:
            improved = False
            # Iterate over all pairs of indices i and j in the tour (0 <= i < j <= N-1).
            # These indices define a segment tour[i+1 ... j].
            # Reversing this segment swaps edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The indices for edges must wrap around (modulo N).
            # We iterate through all possible pairs of non-adjacent edges to swap.
            # Edges are defined by their starting city index in the tour.
            # Consider edges starting at index i and index j.
            # Edge 1: (tour[i], tour[(i+1)%N])
            # Edge 2: (tour[j], tour[(j+1)%N])
            # We reverse the segment of the tour *between* these two edges.
            # The segment is from index (i+1)%N up to index j in the original tour list order.
            # The standard 2-opt swap involves selecting two indices i and j (0 <= i < j <= N-1)
            # and reversing the segment of the tour from index i+1 up to index j.
            # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
            # Cities involved: A=tour[i], B=tour[(i+1)%N], C=tour[j], D=tour[(j+1)%N].
            # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # Indices i and j define the segment to reverse (tour[i+1 ... j])
                    # Cities defining the edges being swapped:
                    # Edge 1: (tour[i], tour[i+1]) => city_i, city_i_plus_1
                    # Edge 2: (tour[j], tour[(j+1)%N]) => city_j, city_j_plus_1 (indices mod N)
                    city_i = current_tour[i]
                    city_i_plus_1 = current_tour[(i + 1) % N]
                    city_j = current_tour[j]
                    city_j_plus_1 = current_tour[(j + 1) % N]
    
                    # Check if the edges are adjacent in the tour
                    # Edges (i, i+1) and (j, j+1) are adjacent if (i+1)%N == j or (j+1)%N == i
                    # Given 0 <= i < j <= N-1, (j+1)%N == i is not possible unless N=2 (handled).
                    # (i+1)%N == j is true when j = i+1. This is the case where the segment
                    # tour[i+1 ... j] has only one city (tour[i+1]). Reversing it does nothing.
                    # We only consider non-adjacent edges for 2-opt swap.
                    # The standard 2-opt swap reverses the segment between i+1 and j.
                    # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
                    # The condition `(i + 1) % N == j` correctly skips adjacent segments for i < j.
                    if (i + 1) % N == j:
                        continue
    
                    # Calculate the change in total tour distance if we swap edges (city_i, city_i_plus_1)
                    # and (city_j, city_j_plus_1) with (city_i, city_j) and (city_i_plus_1, city_j_plus_1).
                    # Delta = d(city_i, city_j) + d(city_i_plus_1, city_j_plus_1) - d(city_i, city_i_plus_1) - d(city_j, city_j_plus_1)
                    current_dist_segment = graph_matrix[city_i][city_i_plus_1] + graph_matrix[city_j][city_j_plus_1]
                    new_dist_segment = graph_matrix[city_i][city_j] + graph_matrix[city_i_plus_1][city_j_plus_1]
                    delta = new_dist_segment - current_dist_segment
    
    
                    if delta < 0:
                        # An improvement is found (new tour is shorter)
                        # Perform the 2-opt swap by reversing the segment current_tour[i+1 : j+1]
                        # This reversal is correct for swapping edges (i,i+1) and (j,j+1) (linear indices).
                        # Given i < j, the slice current_tour[i + 1 : j + 1] is valid and contains indices from i+1 to j.
                        current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
                        improved = True
                        # Since an improvement was made, restart the search for improvements
                        # from the beginning of the outer while loop.
                        break # Break the inner j loop
                if improved:
                    break # Break the outer i loop
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not graph_matrix[0] or not isinstance(graph_matrix[0], list):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        if num_cities == 2:
            return [0, 1] # Or [1, 0], doesn't matter for length in undirected graph
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    # Assuming distances are positive as per problem description
                    if distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found.
            # This should only happen if all cities are visited, which is checked by the while loop condition.
            # However, as a safeguard, if it happens unexpectedly (e.g., graph not complete), break.
            if next_city == -1:
                 # This indicates an issue, likely with graph structure assumptions or logic error.
                 # Return the partial tour found.
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with complete graph.
             # Return the potentially incomplete tour.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

## IV. Evolutionary Lineage (Parent-Child)
- Gen: 0, ID: 567e297a (Score: 0.044, V)
    - Gen: 1, ID: ba82bd40 (Score: 0.045, V)
        - Gen: 2, ID: 5b4559cd (Score: 0.045, V)
            - Gen: 3, ID: 2b19831b (Score: 0.045, V)
                - Gen: 4, ID: b1077a78 (Score: 0.045, V)
                - Gen: 4, ID: b55e179b (Score: 0.045, V)
            - Gen: 3, ID: 502cf498 (Score: 0.045, V)
        - Gen: 2, ID: 6d168a77 (Score: 0.045, V)
    - Gen: 1, ID: 160685c7 (Score: 0.045, V)
        - Gen: 2, ID: 7cdc3445 (Score: 0.045, V)
            - Gen: 3, ID: d61187b3 (Score: 0.045, V)
            - Gen: 3, ID: 1a4c61d0 (Score: 0.045, V)
        - Gen: 2, ID: bfc7e9e3 (Score: 0.045, V)
            - Gen: 3, ID: aea12621 (Score: 0.045, V)
            - Gen: 4, ID: dfa13b9b (Score: 0.045, V)
    - Gen: 2, ID: ae2e352e (Score: 0.045, V)
    - Gen: 2, ID: 23414ddb (Score: 0.045, V)

## V. All Programs by Generation & Timestamp

### 1. Program ID: 567e297a-6b5a-43d7-a0a4-4e381e6bce48 (Gen: 0)
    - Score: 0.0437
    - Valid: True
    - Parent ID: None
    - Timestamp: 1747528192.40
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

### 2. Program ID: ba82bd40-13bc-4979-a27d-2830148e6230 (Gen: 1)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 567e297a-6b5a-43d7-a0a4-4e381e6bce48
    - Timestamp: 1747528198.47
    - Code:
    ```python
    import math
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # Implement Nearest Neighbor heuristic
        start_city = 0  # Could be randomized for multiple runs, but fixed for simplicity
        tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        while len(tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and the distance is the minimum found so far
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    # Ensure distance is positive/valid, although problem statement says positive
                    if distance > 0 and distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If no unvisited city was found (should not happen in a complete graph with N>0)
            if next_city == -1:
                 # This case indicates an issue, but we break to avoid infinite loop
                 # in case of unexpected graph properties.
                 break
    
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        return tour
    ```

### 3. Program ID: 160685c7-be8c-461a-9298-2b80cf3f3de9 (Gen: 1)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 567e297a-6b5a-43d7-a0a4-4e381e6bce48
    - Timestamp: 1747528198.47
    - Code:
    ```python
    import sys
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
    
        # Implement the Nearest Neighbor heuristic
        start_city = 0 # Start from city 0
        tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        while len(tour) < num_cities:
            next_city = -1
            min_dist = sys.float_info.max # Use max float value for initial min_dist
    
            for city_idx in range(num_cities):
                # Check if city_idx is not visited and is not the current city itself (distance to self is 0)
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    if distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If a next city was found (should always happen in a complete graph unless num_cities <= 1)
            if next_city != -1:
                tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
            else:
                 # This case should theoretically not be reached in a complete graph with num_cities > 1
                 # but as a safeguard, break the loop if no unvisited city is found.
                 break
    
        return tour
    ```

### 4. Program ID: 7cdc3445-4452-47fd-a5ba-9b1ee47446bd (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 160685c7-be8c-461a-9298-2b80cf3f3de9
    - Timestamp: 1747528470.19
    - Code:
    ```python
    import sys
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour."""
        num_cities = len(tour)
        if num_cities < 2:
            return 0 # Or handle as error/special case
    
        total_length = 0
        for i in range(num_cities - 1):
            total_length += graph_matrix[tour[i]][tour[i+1]]
        # Add distance back to the starting city
        total_length += graph_matrix[tour[-1]][tour[0]]
        return total_length
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic,
        trying each city as a starting point and selecting the best tour found.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list) or not all(isinstance(row, list) for row in graph_matrix):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        best_tour = []
        min_tour_length = sys.float_info.max
    
        # Try starting the Nearest Neighbor algorithm from each city
        for start_city in range(num_cities):
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            # Build the tour using Nearest Neighbor
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max
    
                for city_idx in range(num_cities):
                    # Check if city_idx is not visited and is not the current city itself (distance to self is 0)
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Ensure distance is positive for valid edges in TSP context
                        if distance >= 0 and distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # If a next city was found (should always happen in a complete graph unless num_cities <= 1)
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # This case should theoretically not be reached in a complete graph with num_cities > 1
                     # but as a safeguard, break the loop if no unvisited city is found.
                     # This might happen if graph_matrix is not truly complete or has non-positive distances
                     # for unvisited cities, though constraints say distances are positive.
                     break
    
            # After building a full tour (if possible)
            if len(current_tour) == num_cities:
                current_length = calculate_tour_length(graph_matrix, current_tour)
    
                if current_length < min_tour_length:
                    min_tour_length = current_length
                    best_tour = current_tour
    
        # If no tour was found (e.g., num_cities > 1 but graph not complete), return empty
        if not best_tour and num_cities > 0:
             # This should not happen with a complete valid graph matrix input
             # but returning the tour found from start_city 0 as a fallback
             # or an empty list indicates failure to find a full tour.
             # Given the constraints assume a complete graph, this fallback is unlikely needed.
             # However, if for some reason a full tour wasn't constructed (e.g. graph issues),
             # the loop finishes without finding a best_tour if min_tour_length remains inf.
             # A simple fallback could be to return the tour from start_city 0 if no better one exists,
             # but returning the best found (or empty if none) is more robust.
             pass # best_tour is already [] if nothing better than inf was found
    
        return best_tour
    ```

### 5. Program ID: bfc7e9e3-7712-467c-8b48-055800b2b5b0 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 160685c7-be8c-461a-9298-2b80cf3f3de9
    - Timestamp: 1747528470.20
    - Code:
    ```python
    import sys
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour including the return to the start."""
        num_cities = len(graph_matrix)
        if not tour or len(tour) != num_cities:
            # This case should ideally not be reached for a valid tour covering all cities
            # in a complete graph, but return a very large value as a safeguard.
            return sys.float_info.max
    
        total_length = 0
        try:
            for i in range(num_cities - 1):
                total_length += graph_matrix[tour[i]][tour[i+1]]
            # Add distance back to the start city to complete the cycle
            total_length += graph_matrix[tour[-1]][tour[0]]
        except IndexError:
            # Handle potential index errors if tour indices are out of bounds
            return sys.float_info.max
        except TypeError:
            # Handle potential type errors if graph_matrix or tour contains invalid types
             return sys.float_info.max
    
    
        return total_length
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        with multiple starting cities. It runs the Nearest Neighbor algorithm starting
        from each possible city and returns the best tour found (the one with the minimum total length).
    
        Args:
            graph_matrix: A list of lists of numbers (integers or floats) representing
                          the adjacency matrix of a complete graph. graph_matrix[i][j]
                          is the distance from city i to city j. Assumes a square matrix
                          where len(graph_matrix) is the number of cities. Distances
                          are positive, and graph_matrix[i][i] is 0.
    
        Returns:
            A list of node indices (integers from 0 to N-1) representing the order
            of cities in the found tour. The returned list is a permutation of [0...N-1].
            Returns an empty list for invalid input or zero cities. Returns [0] for a single city.
        """
        # Input validation
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
    
        # Basic check for square matrix (assuming first row determines width)
        if any(len(row) != num_cities for row in graph_matrix):
             # Not a square matrix
             return []
    
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        best_tour = []
        min_total_length = sys.float_info.max
    
        # Iterate through each city to use as a potential starting point for the NN heuristic
        for start_city in range(num_cities):
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            # Build the tour using Nearest Neighbor from the current start_city
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max
    
                # Find the nearest unvisited city from the current city
                for city_idx in range(num_cities):
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Find the true minimum positive distance
                        if distance < min_dist:
                             min_dist = distance
                             next_city = city_idx
    
                # Add the found nearest city to the tour
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # This case indicates that despite having unvisited cities,
                     # no reachable unvisited city was found from the current_city
                     # with a finite distance. This should not happen in a complete graph
                     # with positive distances. Break to prevent infinite loops.
                     break # Should ideally not be reached
    
            # If a full tour was constructed (visited all cities)
            if len(current_tour) == num_cities:
                # Calculate the length of the current tour including return to start
                current_total_length = calculate_tour_length(graph_matrix, current_tour)
    
                # Update best tour if the current one is shorter
                if current_total_length < min_total_length:
                    min_total_length = current_total_length
                    best_tour = current_tour
    
        # Return the best tour found across all starting cities.
        # If num_cities > 1 and no valid tour was found (shouldn't happen for complete graph), best_tour will be empty.
        # If num_cities == 1, it returns [0] from the initial check.
        return best_tour
    ```

### 6. Program ID: 5b4559cd-c51d-4b11-9acf-c9b6f8fd0eac (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: ba82bd40-13bc-4979-a27d-2830148e6230
    - Timestamp: 1747528470.22
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            total_length += graph_matrix[current_city][next_city]
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        total_length += graph_matrix[last_city][first_city]
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
    
        improved = True
        while improved:
            improved = False
            # Iterate over all pairs of indices i and j in the tour (0 <= i < j <= N-1).
            # These indices define a segment tour[i+1 ... j].
            # Reversing this segment swaps edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The indices for edges must wrap around (modulo N).
            # We iterate through all possible pairs of non-adjacent edges to swap.
            # Edges are defined by their starting city index in the tour.
            # Consider edges starting at index i and index j.
            # Edge 1: (tour[i], tour[(i+1)%N])
            # Edge 2: (tour[j], tour[(j+1)%N])
            # We reverse the segment of the tour *between* these two edges.
            # The segment is from index (i+1)%N up to index j in the original tour list order.
            # The standard 2-opt swap involves selecting two indices i and j (0 <= i < j <= N-1)
            # and reversing the segment of the tour from index i+1 up to index j.
            # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
            # Cities involved: A=tour[i], B=tour[(i+1)%N], C=tour[j], D=tour[(j+1)%N].
            # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # Indices i and j define the segment to reverse (tour[i+1 ... j])
                    # Cities defining the edges being swapped:
                    # Edge 1: (tour[i], tour[i+1]) => city_i, city_i_plus_1
                    # Edge 2: (tour[j], tour[j+1]) => city_j, city_j_plus_1 (indices mod N)
                    city_i = current_tour[i]
                    city_i_plus_1 = current_tour[(i + 1) % N]
                    city_j = current_tour[j]
                    city_j_plus_1 = current_tour[(j + 1) % N]
    
                    # Check if the edges are adjacent in the tour
                    # Edges (i, i+1) and (j, j+1) are adjacent if (i+1)%N == j or (j+1)%N == i
                    # Given 0 <= i < j <= N-1, (j+1)%N == i is not possible unless N=2 (handled).
                    # (i+1)%N == j is true when j = i+1. This is the case where the segment
                    # tour[i+1 ... j] has only one city (tour[i+1]). Reversing it does nothing.
                    # We only consider non-adjacent edges for 2-opt swap.
                    # The condition (i + 1) % N == j covers the case j = i+1.
                    # We also need to exclude the case where j = N-1 and i = 0,
                    # as reversing tour[1...N-1] swaps edges (tour[0], tour[1]) and (tour[N-1], tour[0]),
                    # which are adjacent in the cycle. The loop structure naturally handles i<j.
                    # The condition `(i + 1) % N == j` is sufficient to skip adjacent edges for i < j.
                    if (i + 1) % N == j:
                        continue
    
                    # Calculate the change in total tour distance if we swap edges (city_i, city_i_plus_1)
                    # and (city_j, city_j_plus_1) with (city_i, city_j) and (city_i_plus_1, city_j_plus_1).
                    # Delta = d(city_i, city_j) + d(city_i_plus_1, city_j_plus_1) - d(city_i, city_i_plus_1) - d(city_j, city_j_plus_1)
                    current_dist_segment = graph_matrix[city_i][city_i_plus_1] + graph_matrix[city_j][city_j_plus_1]
                    new_dist_segment = graph_matrix[city_i][city_j] + graph_matrix[city_i_plus_1][city_j_plus_1]
                    delta = new_dist_segment - current_dist_segment
    
    
                    if delta < 0:
                        # An improvement is found (new tour is shorter)
                        # Perform the 2-opt swap by reversing the segment current_tour[i+1 : j+1]
                        # This reversal is correct for swapping edges (i,i+1) and (j,j+1) (linear indices).
                        # Given i < j, the slice current_tour[i + 1 : j + 1] is valid and contains indices from i+1 to j.
                        current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
                        improved = True
                        # Since an improvement was made, restart the search for improvements
                        # from the beginning of the outer while loop.
                        break # Break the inner j loop
                if improved:
                    break # Break the outer i loop
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Handle case for N=2 specially for NN, although handled by 2-opt N<3 check
        if num_cities == 2:
            # If only two cities, the only tour is [0, 1] or [1, 0]
            # NN starting at 0 will go to 1.
            for city_idx in range(num_cities):
                if city_idx != start_city:
                    initial_tour.append(city_idx)
                    visited.add(city_idx)
                    break
    
        else: # num_cities > 2, proceed with standard NN
            while len(initial_tour) < num_cities:
                next_city = -1
                min_dist = float('inf')
    
                for city_idx in range(num_cities):
                    # Check if city_idx is unvisited and the distance is the minimum found so far
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Assuming distances are positive as per problem description
                        if distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # This case should not happen in a complete graph with N > len(tour)
                if next_city == -1:
                     # If somehow stuck, break to avoid infinite loop
                     break
    
                initial_tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected graph structure or issue.
             # Return the partial tour found.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 7. Program ID: 6d168a77-796a-41dc-a6a2-0d5f685f4fa1 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: ba82bd40-13bc-4979-a27d-2830148e6230
    - Timestamp: 1747528470.23
    - Code:
    ```python
    import math
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Repeated Nearest Neighbor heuristic.
        Runs the Nearest Neighbor heuristic starting from each city and returns the best tour found.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
            Returns an empty list for empty or invalid input, or [0] for a single city.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        best_tour = None
        min_tour_length = float('inf')
    
        # Try starting the Nearest Neighbor heuristic from each city
        for start_node in range(num_cities):
            current_tour = [start_node]
            visited = {start_node}
            current_city = start_node
    
            # Build the tour using the Nearest Neighbor rule
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = float('inf')
    
                for city_idx in range(num_cities):
                    # Check if city_idx is unvisited and the distance is the minimum found so far
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Ensure distance is non-negative and minimum
                        # Problem statement says positive, but checking >= 0 adds robustness
                        if distance >= 0 and distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # If no unvisited city was found (should not happen in a complete graph with N>0)
                if next_city == -1:
                     # This case indicates an issue with the graph or logic, break to avoid infinite loop
                     # This part of the tour construction might be incomplete
                     break
    
                current_tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
    
            # Check if the generated tour is complete (contains all cities)
            if len(current_tour) == num_cities:
                # Calculate the length of the current tour (including return to start)
                current_tour_length = 0
                # Add distances between consecutive cities in the tour
                for i in range(num_cities - 1):
                    current_tour_length += graph_matrix[current_tour[i]][current_tour[i+1]]
                # Add the distance from the last city back to the starting city
                current_tour_length += graph_matrix[current_tour[-1]][current_tour[0]]
    
                # Update best tour if the current tour is shorter
                if current_tour_length < min_tour_length:
                    min_tour_length = current_tour_length
                    best_tour = current_tour
    
        # Return the best tour found among all starting cities
        return best_tour if best_tour is not None else [] # Return empty list if no complete tour was found (should not happen for valid inputs)
    ```

### 8. Program ID: ae2e352e-db7f-4704-9a63-b2c47870f689 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 567e297a-6b5a-43d7-a0a4-4e381e6bce48
    - Timestamp: 1747528470.23
    - Code:
    ```python
    import sys
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic.
    
        Starts at city 0 and iteratively moves to the nearest unvisited city
        until all cities are visited.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
                          Distances are assumed to be positive.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            # Basic validation
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # Start at city 0 (can be modified to start at a random city or iterate starts)
        start_city = 0
        tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        while len(tour) < num_cities:
            next_city = -1
            min_dist = sys.float_info.max # Use max float value for initial min_dist
    
            for city_idx in range(num_cities):
                # Check if the city is not visited and the distance is smaller than current min_dist
                # Also ensure it's not the current city itself (distance graph_matrix[i][i] is 0)
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    # Ensure positive distance for valid edge
                    if distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found.
            # This should not happen in a complete graph unless num_cities > 1
            # and all nodes are already in the tour (which is the loop termination condition).
            if next_city == -1:
                 # This case theoretically shouldn't be reached in a connected complete graph
                 # when the loop condition `len(tour) < num_cities` is true.
                 # However, as a safeguard, break if no next city is found.
                 break
    
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        return tour
    ```

### 9. Program ID: 23414ddb-242a-4488-8e4c-c7f7d5c21da3 (Gen: 2)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 567e297a-6b5a-43d7-a0a4-4e381e6bce48
    - Timestamp: 1747528470.24
    - Code:
    ```python
    import sys
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            # Basic validation for empty input
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # Use Nearest Neighbor heuristic
        start_city = 0 # Could randomize this for multiple runs, but fixed for deterministic output
        
        tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        while len(tour) < num_cities:
            next_city = -1
            min_dist = sys.float_info.max # Use a large number for initial min_dist
    
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and reachable from current_city
                # Ensure distance is positive (though problem statement says distances are positive)
                if city_idx not in visited and graph_matrix[current_city][city_idx] > 0:
                    if graph_matrix[current_city][city_idx] < min_dist:
                        min_dist = graph_matrix[current_city][city_idx]
                        next_city = city_idx
    
            # If no unvisited city found (should only happen if graph isn't complete/connected,
            # which is contrary to problem description assumptions, but good for robustness)
            if next_city == -1:
                 # This case should ideally not be reached in a complete graph with positive distances
                 # If it somehow is, we might break or try to find any unvisited city.
                 # Given the problem constraints, we expect to always find a next_city until all are visited.
                 break 
    
            tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        return tour
    ```

### 10. Program ID: d61187b3-9e6d-4f21-bdd4-9912e17e8dcc (Gen: 3)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 7cdc3445-4452-47fd-a5ba-9b1ee47446bd
    - Timestamp: 1747528681.69
    - Code:
    ```python
    import sys
    import math
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour."""
        num_cities = len(tour)
        if num_cities < 2:
            return 0
    
        total_length = 0
        for i in range(num_cities - 1):
            total_length += graph_matrix[tour[i]][tour[i+1]]
        # Add distance back to the starting city
        total_length += graph_matrix[tour[-1]][tour[0]]
        return total_length
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement. The Nearest Neighbor phase
        tries each city as a starting point to find a good initial tour.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
                          Assumes graph_matrix[i][j] > 0 for i != j and graph_matrix[i][i] == 0.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list) or not all(isinstance(row, list) and len(row) == len(graph_matrix) for row in graph_matrix):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        # --- Phase 1: Construct initial tour using Nearest Neighbor (all starts) ---
        best_tour = []
        min_tour_length = math.inf
    
        for start_city in range(num_cities):
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = math.inf
    
                for city_idx in range(num_cities):
                    # Find the nearest unvisited city
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        if distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # If a next city was found (should always happen in a complete graph for N > 1)
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # This break should only be reached if the graph is not complete
                     # and there are unvisited cities but no edge with finite positive distance
                     # from the current city. Given constraints, this should not happen for N > 1.
                     break # Safety break
    
            # After building a tour, check if it visited all cities
            if len(current_tour) == num_cities:
                current_length = calculate_tour_length(graph_matrix, current_tour)
    
                if current_length < min_tour_length:
                    min_tour_length = current_length
                    best_tour = list(current_tour) # Use list() to copy
    
        # If no complete tour was found by NN (implies issue with graph input for N>1)
        # With valid input (complete graph, N>1), best_tour will always be populated.
        if not best_tour:
            # This indicates a failure to construct a full tour, likely due to invalid graph input
            return [] # Return empty if NN couldn't build a full tour
    
    
        # --- Phase 2: Improve tour using 2-opt local search ---
        current_tour = list(best_tour) # Start 2-opt from the best NN tour
        # current_tour_length = min_tour_length # Can optionally track length, but 2-opt logic compares edge changes
    
        # 2-opt improvement loop
        improved = True
        while improved:
            improved = False
            # Iterate through all pairs of indices i and j in the tour, 0 <= i < j < num_cities.
            # These indices define the endpoints of the segment [tour[i]...tour[j]] to reverse.
            # The standard 2-opt swap usually considers edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N])
            # and reverses the segment tour[i+1 ... j].
            # The loops for i and j to achieve this swap type are 0 <= i < num_cities - 1 and i + 1 < j < num_cities.
            # This ensures i+1 < j, so the edges are not adjacent.
            # The cities involved for comparison are: A=tour[i], B=tour[i+1], C=tour[j], D=tour[(j+1)%N]
    
            for i in range(num_cities - 1):
                for j in range(i + 2, num_cities):
                    # Cities involved in the potential swap of edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # Note: When j = num_cities - 1, the second edge is (tour[N-1], tour[0]).
                    # The indices i and j here refer to positions in the *current_tour* list.
    
                    city_a = current_tour[i]
                    city_b = current_tour[i+1]
                    city_c = current_tour[j]
                    city_d = current_tour[(j+1) % num_cities] # Handles wrap-around for the edge after city_c
    
                    # Calculate the change in distance if we swap edges (A,B) and (C,D) with (A,C) and (B,D)
                    old_dist = graph_matrix[city_a][city_b] + graph_matrix[city_c][city_d]
                    new_dist = graph_matrix[city_a][city_c] + graph_matrix[city_b][city_d]
    
                    # If swapping improves the tour length
                    if new_dist < old_dist:
                        # Perform the 2-opt swap by reversing the segment from index i+1 up to index j (inclusive)
                        # Example: tour = [t0, t1, t2, t3, t4, t5], i=1, j=4. Reverse tour[2:5] = [t2, t3, t4].
                        # New tour = [t0, t1] + [t4, t3, t2] + [t5] = [t0, t1, t4, t3, t2, t5]
                        current_tour[i+1 : j+1] = current_tour[i+1 : j+1][::-1]
    
                        # An improvement was made, set flag and restart the search from the beginning of the loops
                        # (First-improvement strategy)
                        improved = True
                        break # Break inner j loop
                if improved:
                    break # Break outer i loop
    
        # Return the optimized tour after the 2-opt phase
        return current_tour
    ```

### 11. Program ID: 1a4c61d0-fb62-4c72-85bd-a0fa13084fc8 (Gen: 3)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 7cdc3445-4452-47fd-a5ba-9b1ee47446bd
    - Timestamp: 1747528681.70
    - Code:
    ```python
    import sys
    import random
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour."""
        num_cities = len(tour)
        if num_cities < 2:
            return 0
    
        total_length = 0
        for i in range(num_cities):
            total_length += graph_matrix[tour[i]][tour[(i + 1) % num_cities]]
        return total_length
    
    def two_opt_swap(tour, i, j):
        """
        Performs a 2-opt swap on the tour by reversing the segment tour[i+1...j].
        Assumes 0 <= i < j < len(tour).
        """
        # The new tour is tour[0...i] + reversed(tour[i+1...j]) + tour[j+1...N-1].
        new_tour = tour[:i+1] + tour[i+1:j+1][::-1] + tour[j+1:]
        return new_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        trying each city as a starting point, and then applies the 2-opt improvement heuristic.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
                          Distances are assumed to be positive.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            Example: [0, 1, 2, 3] for 4 cities. Returns [] for 0 cities, [0] for 1 city.
        """
        if not graph_matrix or not isinstance(graph_matrix, list) or not all(isinstance(row, list) for row in graph_matrix):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        # --- Phase 1: Nearest Neighbor (trying all start cities) ---
        best_tour = []
        min_tour_length = sys.float_info.max
    
        # Try starting the Nearest Neighbor algorithm from each city
        # Could add randomization here for start_city for non-deterministic runs if needed,
        # but deterministic is fine for a standard heuristic implementation.
        start_cities = list(range(num_cities))
        # random.shuffle(start_cities) # Optional: uncomment for randomized start cities
    
        for start_city in start_cities:
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            # Build the tour using Nearest Neighbor
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max
    
                for city_idx in range(num_cities):
                    # Check if city_idx is not visited
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Based on constraints, distance is positive, so no need to check distance >= 0
                        if distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # If a next city was found (should always happen in a complete graph for N > 1)
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # This case indicates an issue with the graph (not complete) or logic for N > 1.
                     # Given constraints, we assume this break is not reached for N > 1.
                     break
    
            # After building a full tour (if possible)
            if len(current_tour) == num_cities:
                current_length = calculate_tour_length(graph_matrix, current_tour)
    
                if current_length < min_tour_length:
                    min_tour_length = current_length
                    best_tour = list(current_tour) # Use list() to create a copy
    
        # If num_cities > 1 and no tour was found (implies graph issue), return empty list.
        # Given constraints, best_tour should be populated here for num_cities > 1.
        if not best_tour and num_cities > 1:
            # This branch should theoretically not be hit with valid inputs based on constraints
            # but serves as a safeguard.
            return []
    
        # --- Phase 2: 2-Opt Improvement ---
        # Apply 2-opt to the best tour found by Nearest Neighbor
        current_tour = list(best_tour) # Start 2-opt from a copy of the best NN tour
        num_cities = len(current_tour)
    
        # 2-opt requires at least 3 cities to perform meaningful swaps (reversing a segment of size >= 2)
        if num_cities < 3:
            return current_tour # Cannot apply 2-opt
    
        improved = True
        while improved:
            improved = False
            # Iterate over all pairs of indices (i, j) with 0 <= i < j < num_cities.
            # We consider reversing the segment tour[i+1...j].
            # This corresponds to swapping edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
            # We require j > i + 1 to ensure the segment to reverse has at least 2 elements
            # and the edges being swapped are non-adjacent.
    
            for i in range(num_cities - 2): # i is the index BEFORE the segment to reverse
                for j in range(i + 2, num_cities): # j is the index AT the end of the segment to reverse
    
                    # Get the city indices involved in the potential swap
                    city1_idx = current_tour[i]       # Start node of first edge
                    city2_idx = current_tour[i+1]     # End node of first edge (start of segment to reverse)
                    city3_idx = current_tour[j]       # End node of segment to reverse (start node of second edge)
                    city4_idx = current_tour[(j+1)%num_cities] # End node of second edge (handles wrap-around)
    
                    # Calculate change in tour length if we swap the edges (city1_idx, city2_idx) and (city3_idx, city4_idx)
                    # and add new edges (city1_idx, city3_idx) and (city2_idx, city4_idx).
                    # This corresponds to reversing the segment from city2_idx to city3_idx in the tour array.
                    delta = graph_matrix[city1_idx][city3_idx] + graph_matrix[city2_idx][city4_idx] - \
                            (graph_matrix[city1_idx][city2_idx] + graph_matrix[city3_idx][city4_idx])
    
                    if delta < 0: # If improvement found (new tour is shorter)
                        # Perform the swap by reversing the segment tour[i+1...j]
                        current_tour = two_opt_swap(current_tour, i, j)
                        improved = True
                        # Restart the search from the beginning after an improvement is made.
                        # This ensures that the algorithm continues until a local optimum is reached.
                        break # Break j loop
                if improved:
                    break # Break i loop
    
        # Return the final improved tour
        return current_tour
    ```

### 12. Program ID: aea12621-5794-45b8-a493-4837172542f2 (Gen: 3)
    - Score: 0.0447
    - Valid: True
    - Parent ID: bfc7e9e3-7712-467c-8b48-055800b2b5b0
    - Timestamp: 1747528756.32
    - Code:
    ```python
    import sys
    import random
    
    def calculate_tour_length(graph_matrix, tour):
        """
        Calculates the total length of a given tour including the return to the start.
    
        Args:
            graph_matrix: The adjacency matrix.
            tour: A list of city indices representing the tour order.
    
        Returns:
            The total length of the tour, or sys.float_info.max if the tour is invalid
            (e.g., not a complete tour of all cities).
        """
        num_cities = len(graph_matrix)
        if not tour or len(tour) != num_cities:
            # Invalid tour length
            return sys.float_info.max
    
        total_length = 0
        try:
            for i in range(num_cities):
                # Connect tour[i] to tour[(i+1)%N]
                total_length += graph_matrix[tour[i]][tour[(i + 1) % num_cities]]
        except (IndexError, TypeError):
            # Handle potential errors accessing graph_matrix or invalid tour indices/types
            return sys.float_info.max
    
        return total_length
    
    # Helper function for performing a 2-opt swap (reversing a segment)
    def two_opt_swap(tour, i, j):
        """
        Performs a 2-opt swap on the tour by reversing a segment.
    
        Swapping edges (tour[i], tour[(i+1)%N]) and (tour[j], tour[(j+1)%N])
        corresponds to reversing the segment of the tour between index (i+1)%N and j.
        i and j are the indices in the tour list *before* the edges being swapped.
    
        Args:
            tour: The current tour list of city indices.
            i: The index in the tour list before the first edge (tour[i], tour[(i+1)%N]).
            j: The index in the tour list before the second edge (tour[j], tour[(j+1)%N]).
    
        Returns:
            A new list representing the tour after the 2-opt swap, or None if swap logic fails.
        """
        n = len(tour)
        # Indices defining the segment to reverse in the tour list
        # The segment is from index (i+1)%N up to index j.
        start_idx = (i + 1) % n
        end_idx = j
    
        # Create the new tour list by reversing the segment
        new_tour = tour[:] # Start with a copy
    
        if start_idx <= end_idx:
            # Simple segment reversal: tour[0...start-1] + reversed(tour[start...end]) + tour[end+1...N-1]
            new_tour[start_idx : end_idx + 1] = tour[start_idx : end_idx + 1][::-1]
        else:
            # Wrap-around segment reversal:
            # The segment to reverse consists of tour[start_idx:] and tour[:end_idx + 1].
            # The parts not reversed are tour[(end_idx + 1)%n : start_idx].
            # The new tour is constructed by the non-reversed part followed by the reversed part.
            # The reversed part is (tour[start_idx:] + tour[:end_idx + 1])[::-1]
            # The non-reversed part is tour[(end_idx + 1) % n : start_idx]
    
            # The standard way to construct the new tour list when reversing segment (i+1)%n to j (wrapping)
            # is to take the list from (j+1)%n to i (wrapping), followed by the reversed segment.
            # Cities in order: tour[(j+1)%n], tour[(j+2)%n], ..., tour[i],
            # then tour[j], tour[j-1], ..., tour[(i+1)%n].
    
            # A simpler slice-based approach for wrap-around reversal:
            # Take tour from start_idx to end (inclusive), then tour from 0 to end_idx (inclusive).
            # Reverse this combined segment.
            segment_to_reverse = tour[start_idx:] + tour[:end_idx + 1]
            reversed_segment = segment_to_reverse[::-1]
    
            # Construct the new tour list
            # It starts with the segment from (end_idx + 1)%n up to start_idx-1 (wrapping)
            # followed by the reversed segment.
            # Example: n=6, i=4, j=1. start_idx=5, end_idx=1.
            # Reversed segment indices: 5, 0, 1. Cities: tour[5], tour[0], tour[1]. Reversed: tour[1], tour[0], tour[5].
            # Non-reversed indices: from (1+1)%6=2 up to 4. Indices 2, 3, 4. Cities: tour[2], tour[3], tour[4].
            # New tour list: [tour[2], tour[3], tour[4], tour[1], tour[0], tour[5]]
    
            # This is equivalent to:
            # tour[(end_idx + 1) % n : start_idx] + (tour[start_idx:] + tour[:end_idx + 1])[::-1]
            new_tour = tour[(end_idx + 1) % n : start_idx] + (tour[start_idx:] + tour[:end_idx + 1])[::-1]
    
    
        # Ensure the new tour has the correct number of cities
        if len(new_tour) != n:
             # This indicates an error in the swap logic
             return None # Should not happen with correct indices
    
        return new_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        starting from each city, followed by a 2-opt local search optimization on each resulting tour.
        Returns the best tour found among all starting cities after optimization.
    
        Args:
            graph_matrix: A list of lists of numbers (integers or floats) representing
                          the adjacency matrix of a complete graph. graph_matrix[i][j]
                          is the distance from city i to city j. Assumes a square matrix
                          where len(graph_matrix) is the number of cities. Distances
                          are positive, and graph_matrix[i][i] is 0.
    
        Returns:
            A list of node indices (integers from 0 to N-1) representing the order
            of cities in the found tour. The returned list is a permutation of [0...N-1].
            Returns an empty list for invalid input or zero cities. Returns [0] for a single city.
        """
        # Input validation
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
    
        # Basic check for square matrix
        if any(len(row) != num_cities for row in graph_matrix):
             return []
    
        if num_cities == 1:
            return [0]
    
        best_tour = []
        min_total_length = sys.float_info.max
    
        # Iterate through each city to use as a potential starting point for the NN heuristic
        # Consider iterating through a random subset of starting cities for larger N
        # For simplicity and determinism in testing, iterate through all cities 0 to N-1
        start_cities = list(range(num_cities))
        # Optional: Shuffle start_cities for variety in repeated runs, but for testing determinism, keep ordered.
        # random.shuffle(start_cities)
    
        for start_city in start_cities:
            # --- Nearest Neighbor Construction ---
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max # Use float_info.max for comparison
    
                for city_idx in range(num_cities):
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Ensure distance is valid (e.g., not negative infinity or similar issue)
                        # and is positive (as per problem description assumption)
                        if isinstance(distance, (int, float)) and distance >= 0 and distance < min_dist:
                             min_dist = distance
                             next_city = city_idx
    
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # Should not happen in a complete graph with positive finite distances
                     # If it happens (e.g., invalid input like non-finite distances), break to prevent infinite loop
                     # This tour is incomplete and won't be considered valid.
                     break
    
            # If NN failed to build a complete tour (e.g., graph not complete or invalid data)
            if len(current_tour) != num_cities:
                 continue # Skip this start city, as it didn't yield a full tour
    
            # --- 2-opt Improvement ---
            # Apply 2-opt until no further improvements can be made in a full pass
            improved = True
            while improved:
                improved = False # Assume no improvement in this pass
                num_tour_cities = len(current_tour) # Should be num_cities
    
                # Iterate through all unique pairs of non-adjacent edges (i, (i+1)%N) and (j, (j+1)%N)
                # i and j are indices in the tour list.
                # i ranges from 0 to N-1. j ranges from (i+2)%N to (i-1+N)%N.
                # The original loop structure `for i in range(N): for k in range(2, N): j = (i + k) % N` is correct.
    
                # Flag to indicate if an improvement was made in this pass, used to break outer loop
                improvement_found_in_pass = False
    
                for i in range(num_tour_cities):
                     # If an improvement was found and applied in an inner iteration (k loop),
                     # we set the flag and break out of the outer loop to restart the while loop.
                     if improvement_found_in_pass:
                         break # Break the i loop
    
                     # k is the distance in indices between i and j. k must be at least 2 for non-adjacent edges.
                     for k in range(2, num_tour_cities):
                          j = (i + k) % num_tour_cities
    
                          # Indices of the 4 cities involved in the potential swap
                          city_i = current_tour[i]
                          city_i_plus_1 = current_tour[(i + 1) % num_tour_cities]
                          city_j = current_tour[j]
                          city_j_plus_1 = current_tour[(j + 1) % num_tour_cities]
    
                          # Calculate the change in tour length if we swap edges
                          # (city_i, city_i_plus_1) and (city_j, city_j_plus_1)
                          # with (city_i, city_j) and (city_i_plus_1, city_j_plus_1).
                          # This corresponds to reversing the segment from (i+1)%N to j.
                          original_segment_dist = graph_matrix[city_i][city_i_plus_1] + graph_matrix[city_j][city_j_plus_1]
                          new_segment_dist = graph_matrix[city_i][city_j] + graph_matrix[city_i_plus_1][city_j_plus_1]
                          delta = new_segment_dist - original_segment_dist
    
                          if delta < 0:
                              # An improvement is found. Perform the swap (reverse the segment).
                              next_tour = two_opt_swap(current_tour, i, j)
    
                              if next_tour is not None: # Check if swap was successful
                                  current_tour = next_tour
                                  improved = True # Mark that an improvement was made in this pass (for while condition)
                                  improvement_found_in_pass = True # Flag to break out of outer loops
                                  # Break out of inner loops to restart the while loop
                                  # A single improvement is enough to warrant a new pass.
                                  break # Break the k loop
                     # After the k loop finishes, check if we need to restart the outer while loop
                     if improvement_found_in_pass:
                         break # Break the i loop
    
                # The while loop condition `while improved:` handles the restart.
                # If improved is True after the loops, the while loop continues.
                # If improved is False after the loops, the while loop terminates.
    
    
            # After 2-opt optimization, check if this tour is the best found so far
            final_tour_length = calculate_tour_length(graph_matrix, current_tour)
    
            # Update best tour if the current optimized one is shorter
            if final_tour_length < min_total_length:
                min_total_length = final_tour_length
                best_tour = current_tour[:] # Make a copy of the tour list
    
        # Return the best tour found across all starting cities after 2-opt optimization.
        # If num_cities > 1 and no valid tour was found (shouldn't happen for complete graph), best_tour will be empty initially.
        # If num_cities == 1, it returns [0] from the initial check.
        return best_tour
    ```

### 13. Program ID: 2b19831b-d3da-477a-a4c6-56818885d642 (Gen: 3)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 5b4559cd-c51d-4b11-9acf-c9b6f8fd0eac
    - Timestamp: 1747528756.33
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix):
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour
                 print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix):
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour
            print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
    
        improved = True
        while improved:
            improved = False
            # Iterate over all pairs of indices i and j in the tour (0 <= i < j <= N-1).
            # These indices define a segment tour[i+1 ... j].
            # Reversing this segment swaps edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The indices for edges must wrap around (modulo N).
            # We iterate through all possible pairs of non-adjacent edges to swap.
            # Edges are defined by their starting city index in the tour.
            # Consider edges starting at index i and index j.
            # Edge 1: (tour[i], tour[(i+1)%N])
            # Edge 2: (tour[j], tour[(j+1)%N])
            # We reverse the segment of the tour *between* these two edges.
            # The segment is from index (i+1)%N up to index j in the original tour list order.
            # The standard 2-opt swap involves selecting two indices i and j (0 <= i < j <= N-1)
            # and reversing the segment of the tour from index i+1 up to index j.
            # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
            # Cities involved: A=tour[i], B=tour[(i+1)%N], C=tour[j], D=tour[(j+1)%N].
            # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # Indices i and j define the segment to reverse (tour[i+1 ... j])
                    # Cities defining the edges being swapped:
                    # Edge 1: (tour[i], tour[i+1]) => city_i, city_i_plus_1
                    # Edge 2: (tour[j], tour[(j+1)%N]) => city_j, city_j_plus_1 (indices mod N)
                    city_i = current_tour[i]
                    city_i_plus_1 = current_tour[(i + 1) % N]
                    city_j = current_tour[j]
                    city_j_plus_1 = current_tour[(j + 1) % N]
    
                    # Check if the edges are adjacent in the tour
                    # Edges (i, i+1) and (j, j+1) are adjacent if (i+1)%N == j or (j+1)%N == i
                    # Given 0 <= i < j <= N-1, (j+1)%N == i is not possible unless N=2 (handled).
                    # (i+1)%N == j is true when j = i+1. This is the case where the segment
                    # tour[i+1 ... j] has only one city (tour[i+1]). Reversing it does nothing.
                    # We only consider non-adjacent edges for 2-opt swap.
                    # The standard 2-opt swap reverses the segment between i+1 and j.
                    # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
                    # The condition `(i + 1) % N == j` correctly skips adjacent segments for i < j.
                    if (i + 1) % N == j:
                        continue
    
                    # Calculate the change in total tour distance if we swap edges (city_i, city_i_plus_1)
                    # and (city_j, city_j_plus_1) with (city_i, city_j) and (city_i_plus_1, city_j_plus_1).
                    # Delta = d(city_i, city_j) + d(city_i_plus_1, city_j_plus_1) - d(city_i, city_i_plus_1) - d(city_j, city_j_plus_1)
                    current_dist_segment = graph_matrix[city_i][city_i_plus_1] + graph_matrix[city_j][city_j_plus_1]
                    new_dist_segment = graph_matrix[city_i][city_j] + graph_matrix[city_i_plus_1][city_j_plus_1]
                    delta = new_dist_segment - current_dist_segment
    
    
                    if delta < 0:
                        # An improvement is found (new tour is shorter)
                        # Perform the 2-opt swap by reversing the segment current_tour[i+1 : j+1]
                        # This reversal is correct for swapping edges (i,i+1) and (j,j+1) (linear indices).
                        # Given i < j, the slice current_tour[i + 1 : j + 1] is valid and contains indices from i+1 to j.
                        current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
                        improved = True
                        # Since an improvement was made, restart the search for improvements
                        # from the beginning of the outer while loop.
                        break # Break the inner j loop
                if improved:
                    break # Break the outer i loop
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not graph_matrix[0] or not isinstance(graph_matrix[0], list):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        if num_cities == 2:
            return [0, 1] # Or [1, 0], doesn't matter for length in undirected graph
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                    # Assuming distances are positive as per problem description
                    if distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found.
            # This should only happen if all cities are visited, which is checked by the while loop condition.
            # However, as a safeguard, if it happens unexpectedly (e.g., graph not complete), break.
            if next_city == -1:
                 # This indicates an issue, likely with graph structure assumptions or logic error.
                 # Return the partial tour found.
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with complete graph.
             # Return the potentially incomplete tour.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 14. Program ID: 502cf498-71fa-4e50-9c2e-80b24416eae4 (Gen: 3)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 5b4559cd-c51d-4b11-9acf-c9b6f8fd0eac
    - Timestamp: 1747528756.35
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            total_length += graph_matrix[current_city][next_city]
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        total_length += graph_matrix[last_city][first_city]
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        best_tour = list(current_tour)
        best_distance = calculate_tour_length(graph_matrix, best_tour)
    
        improved = True
        while improved:
            improved = False
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            # These indices define the segment (current_tour[i+1 ... j]) to reverse.
            # Reversing this segment swaps edges (current_tour[i], current_tour[i+1])
            # and (current_tour[j], current_tour[j+1]) (indices mod N).
            for i in range(N - 1):
                for j in range(i + 1, N):
                    # The indices i and j define the segment current_tour[i+1 : j+1]
                    # Note: The standard 2-opt swap involves reversing the path *between*
                    # two edges. If we pick edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]),
                    # the segment reversed is tour[i+1...j].
                    # The indices i and j here correspond to the *start* indices of the two edges
                    # being potentially swapped.
                    # Edge 1: (current_tour[i], current_tour[(i + 1) % N])
                    # Edge 2: (current_tour[j], current_tour[(j + 1) % N])
    
                    # Skip adjacent edges (reversing segment of size 1 or N-1)
                    # For i < j, adjacent edges occur when j == i + 1 or (i==0 and j==N-1 for the cycle edge)
                    # The loop structure i < j covers j=i+1. The wrap-around case (0, N-1)
                    # corresponds to i=0, j=N-1. Reversing tour[1:N] swaps (tour[0],tour[1]) and (tour[N-1],tour[0]).
                    # This is handled by the indices i and j.
                    # We are swapping edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # by reversing the segment tour[i+1...j].
                    # The cities involved are A=tour[i], B=tour[i+1], C=tour[j], D=tour[j+1] (indices mod N).
                    # Old edges: (A, B) and (C, D). New edges: (A, C) and (B, D).
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D)
    
                    # Correct indices for the cities involved in the potential swap:
                    # Edge 1: (current_tour[i], current_tour[(i + 1) % N])
                    # Edge 2: (current_tour[j], current_tour[(j + 1) % N])
                    # The segment to reverse is between index i and index j in the tour list.
                    # The segment is current_tour[i+1 ... j].
                    # The cities are tour[i], tour[i+1], ..., tour[j], tour[j+1] (indices mod N)
                    # The edges being swapped are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
    
                    # Consider indices i and j in the tour list (0 <= i < j <= N-1)
                    # This swap replaces edges (tour[i], tour[i+1]) and (tour[j], tour[j+1])
                    # with (tour[i], tour[j]) and (tour[i+1], tour[j+1])
                    # by reversing the segment tour[i+1 ... j].
    
                    # Cities involved in the swap:
                    A = current_tour[i]
                    B = current_tour[(i + 1) % N] # Note: Use modulo for the end of the segment
                    C = current_tour[j]
                    D = current_tour[(j + 1) % N] # Note: Use modulo for the end of the segment
    
                    # Calculate the change in distance
                    # If we swap edges (A, B) and (C, D) with (A, C) and (B, D)
                    # This corresponds to reversing the segment from B to C.
                    # The segment in the list is tour[i+1 ... j].
                    # If we reverse tour[i+1 ... j], the new edges become (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
                    # This is correct for the 2-opt swap defined by reversing the segment tour[i+1...j].
    
                    current_dist_edges = graph_matrix[A][B] + graph_matrix[C][D]
                    new_dist_edges = graph_matrix[A][C] + graph_matrix[B][D]
    
                    delta = new_dist_edges - current_dist_edges
    
                    # Optimization: only consider swaps if delta < 0
                    if delta < 0:
                        # Perform the 2-opt swap by reversing the segment current_tour[i+1 : j+1]
                        # This reverses the list segment from index i+1 up to index j (inclusive).
                        current_tour[i + 1 : j + 1] = current_tour[i + 1 : j + 1][::-1]
    
                        # Update best distance and flag improvement
                        best_distance += delta
                        improved = True
                        # Since an improvement was made, restart the search for improvements
                        # from the beginning of the outer while loop.
                        # Breaking both loops and letting the while loop condition re-evaluate
                        # is a standard way to implement this.
                        break # Break the inner j loop
                if improved:
                    break # Break the outer i loop
    
            # After iterating through all pairs (i, j) in one pass, if an improvement was made,
            # the `improved` flag is True, and the while loop continues for another pass.
            # If no improvement was made in a full pass, `improved` remains False, and the loop terminates.
    
        return current_tour # Return the locally optimal tour
    
    def solve_tsp_heuristic(graph_matrix: list[list[float]]) -> list[int]:
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Handle case for N=2 specially for NN, although handled by 2-opt N<3 check
        if num_cities == 2:
            # If only two cities, the only tour starting at 0 is [0, 1]
            for city_idx in range(num_cities):
                if city_idx != start_city:
                    initial_tour.append(city_idx)
                    visited.add(city_idx)
                    break
        else: # num_cities > 2, proceed with standard NN
            while len(initial_tour) < num_cities:
                next_city = -1
                min_dist = float('inf')
    
                for city_idx in range(num_cities):
                    # Check if city_idx is unvisited and the distance is the minimum found so far
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Assuming distances are positive as per problem description
                        if distance < min_dist:
                            min_dist = distance
                            next_city = city_idx
    
                # This case should not happen in a complete graph with N > len(tour)
                # If somehow stuck (e.g., graph not truly complete), break
                if next_city == -1:
                     break
    
                initial_tour.append(next_city)
                visited.add(next_city)
                current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected graph structure or issue.
             # Return the partial tour found.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 15. Program ID: b1077a78-71ee-4bc3-be19-18c97b3715cc (Gen: 4)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 2b19831b-d3da-477a-a4c6-56818885d642
    - Timestamp: 1747528858.29
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix):
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour
                 # print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix):
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour
            # print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        current_tour_length = calculate_tour_length(graph_matrix, current_tour)
    
        improved = True
        while improved:
            improved = False
            best_delta = 0 # We are looking for negative delta (improvement)
    
            # Iterate over all pairs of indices i and j in the tour (0 <= i < j <= N-1).
            # These indices define a segment tour[i+1 ... j].
            # Reversing this segment swaps edges (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The indices for edges must wrap around (modulo N).
            # We iterate through all possible pairs of non-adjacent edges to swap.
            # Edges are defined by their starting city index in the tour.
            # Consider edges starting at index i and index j.
            # Edge 1: (tour[i], tour[(i+1)%N])
            # Edge 2: (tour[j], tour[(j+1)%N])
            # We reverse the segment of the tour *between* these two edges.
            # The segment is from index (i+1)%N up to index j in the original tour list order.
            # The standard 2-opt swap involves selecting two indices i and j (0 <= i < j <= N-1)
            # and reversing the segment of the tour from index i+1 up to index j.
            # This swaps edges (tour[i], tour[i+1]) and (tour[j+1], tour[j]) (indices mod N).
            # Note: The indices i and j in the loops refer to the position in the tour list.
            # The cities are current_tour[i] and current_tour[j].
            # The edges being considered are (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
            # The 2-opt move reverses the segment current_tour[i+1...j].
            # This effectively replaces edges (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N])
            # with (current_tour[i], current_tour[j]) and (current_tour[i+1], current_tour[(j+1)%N]).
            # Let A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
            # The swap replaces AB + CD with AC + BD.
            # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
            # Loop through all pairs of indices (i, j) with 0 <= i < j <= N-1
            # These indices define the endpoints of the segment to be reversed.
            # The segment is from index i+1 up to index j (inclusive).
            # The edges involved are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
            # The pairs (i, j) iterate over the potential edges to 'uncross' or swap.
            # i is the index of the city *before* the segment starts.
            # j is the index of the city *at the end* of the segment.
            # The segment to reverse is tour[i+1 ... j].
            # The edges being swapped are (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (using linear indexing for clarity in explanation here, but modulo N for the last edge).
            # Let's use the standard 2-opt indices i and j where 0 <= i < j <= N-1.
            # The points involved are p1=tour[i], p2=tour[i+1], p3=tour[j], p4=tour[(j+1)%N].
            # We swap edges (p1, p2) and (p3, p4) with (p1, p3) and (p2, p4).
            # This is achieved by reversing the segment from index i+1 to j.
    
            for i in range(N - 2): # i goes from 0 to N-3
                for j in range(i + 1, N - 1): # j goes from i+1 to N-2
                    # The segment to reverse is tour[i+1 ... j].
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # Cities involved in swap:
                    # A = current_tour[i]
                    # B = current_tour[i+1]
                    # C = current_tour[j+1] # City *after* the segment end in the original tour
                    # D = current_tour[(j+2)%N] # City after C in the original tour
    
                    # This standard 2-opt loop structure considers pairs of non-adjacent edges
                    # defined by indices i and j (0 <= i < j-1 <= N-2).
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (modulo N).
                    # Let A = current_tour[i], B = current_tour[(i+1)%N], C = current_tour[j], D = current_tour[(j+1)%N].
                    # We check delta for swapping (A, B) and (C, D) with (A, C) and (B, D).
                    # This corresponds to reversing the segment between B and C.
                    # The segment to reverse is from index i+1 up to index j.
                    # The cities are current_tour[i+1], current_tour[i+2], ..., current_tour[j].
    
                    # The loop indices i and j typically refer to the points *before* the edges being swapped.
                    # Edge 1: (tour[i], tour[i+1])
                    # Edge 2: (tour[j], tour[j+1]) (with wrap-around for j+1)
                    # We need to iterate over all pairs of non-adjacent edges.
                    # Let's use indices i and k in the tour list, 0 <= i < k <= N-1.
                    # Edge 1 starts at tour[i]. Edge 2 starts at tour[k].
                    # Edge 1: (tour[i], tour[(i+1)%N])
                    # Edge 2: (tour[k], tour[(k+1)%N])
                    # We consider swapping these edges if they are not adjacent.
                    # Adjacent means (i+1)%N == k or (k+1)%N == i.
    
                    # Let's iterate over all pairs of indices (i, j) in the tour list,
                    # where 0 <= i < j <= N-1.
                    # i and j define the segment to reverse: tour[i+1 ... j].
                    # The edges affected are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                    # Cities: A=tour[i], B=tour[i+1], C=tour[j], D=tour[(j+1)%N].
                    # Check delta for swapping (A, B) and (C, D) with (A, C) and (B, D).
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                    # Correct indices for 2-opt swap:
                    # Choose two indices i and j from the tour, 0 <= i < j <= N-1.
                    # Reverse the segment current_tour[i+1 ... j].
                    # The edges being swapped are (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
                    # Cities: A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
                    # The original edges are (A, B) and (C, D).
                    # The new edges after reversing segment [i+1...j] are (A, C) and (B, D).
                    # Need to handle wrap-around correctly.
                    # The indices i and j define the endpoints of the segment *outside* the reversal.
                    # Reversing tour[i+1 : j+1] swaps edge (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # Let's use indices i and j such that 0 <= i < j-1 <= N-2.
                    # This ensures the segment [i+1...j] has at least one city.
                    # i goes from 0 up to N-3.
                    # j goes from i+2 up to N-1.
                    # This considers pairs of non-consecutive edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
    
                    # Let's refine the loops based on standard 2-opt implementation.
                    # Iterate all pairs of indices i and j such that 0 <= i < j < N.
                    # These indices define the points where the tour is 'cut'.
                    # The segments are tour[0..i], tour[i+1..j], tour[j+1..N-1].
                    # Reversing tour[i+1..j] creates a new tour.
                    # The edges changed are (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                    # New edges are (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
                    # Need to handle wrap-around for the last edge (tour[N-1], tour[0]).
                    # A standard way is to iterate i from 0 to N-2 and j from i+1 to N-1.
                    # The segment to reverse is from index i+1 to j.
                    # The edges are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                    # Let i_idx = i, j_idx = j.
                    # Cities: A = current_tour[i_idx], B = current_tour[(i_idx+1)%N], C = current_tour[j_idx], D = current_tour[(j_idx+1)%N].
                    # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                    # Iterate over all possible first points i (0 to N-1)
                    for i_idx in range(N):
                        # Iterate over all possible second points j (i+1 to N-1), handling wrap-around
                        # A standard 2-opt swap considers reversing the segment between two edges.
                        # Let the edges be (tour[i], tour[i+1]) and (tour[j], tour[j+1]) (indices mod N).
                        # We iterate over all pairs of indices i and j (0 <= i < j <= N-1)
                        # and consider reversing the segment tour[i+1...j].
                        # This swaps edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                        # Cities involved: A=tour[i], B=tour[i+1], C=tour[j], D=tour[(j+1)%N].
                        # New edges: (A, C) and (B, D).
                        # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                        # The standard loop structure for 2-opt:
                        # Iterate i from 0 to N-2
                        # Iterate j from i+1 to N-1
                        # This defines the segment tour[i+1...j] to reverse.
                        # The edges are (tour[i], tour[i+1]) and (tour[j], tour[(j+1)%N]).
                        # Cities: A = current_tour[i], B = current_tour[i+1], C = current_tour[j], D = current_tour[(j+1)%N].
                        # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                        for i in range(N - 1): # Edge 1 starts at index i
                            for j in range(i + 1, N): # Edge 2 starts at index j
                                # Indices i and j in the loop define the start points of the two edges being considered for swapping.
                                # Edge 1: (current_tour[i], current_tour[(i+1)%N])
                                # Edge 2: (current_tour[j], current_tour[(j+1)%N])
    
                                # These edges are adjacent if (i+1)%N == j or (j+1)%N == i.
                                # If i < j, adjacency means j == i+1 or (j+1)%N == i (only if j=N-1 and i=0).
                                # Swapping adjacent edges doesn't change the tour (reversing a segment of length 1 or 0).
                                # The standard 2-opt swap reverses the segment between the *end* of the first edge and the *start* of the second edge.
                                # If edges are (A, B) and (C, D), where B is A's successor and D is C's successor in the tour,
                                # the swap considers reversing the path from B to C.
                                # This replaces (A, B) and (C, D) with (A, C) and (B, D).
                                # Cities: A = current_tour[i], B = current_tour[(i+1)%N], C = current_tour[j], D = current_tour[(j+1)%N].
                                # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                                # Let's use the common implementation iterating pairs of indices (i, j)
                                # where 0 <= i < j <= N-1. Reversing segment tour[i+1...j].
                                # Edges swapped: (tour[i], tour[i+1]) and (tour[j], tour[j+1]).
                                # Cities: A=tour[i], B=tour[i+1], C=tour[j], D=tour[j+1]. (linear indices)
                                # Delta = d(A, C) + d(B, D) - d(A, B) - d(C, D).
    
                                # Correct indices for reversing segment tour[i+1 ... j] (inclusive endpoints)
                                # Cities involved:
                                # p1 = current_tour[i]
                                # p2 = current_tour[i+1]
                                # p3 = current_tour[j]
                                # p4 = current_tour[(j+1)%N] # Handles wrap around
    
                                # Calculate change in length
                                # Old edges: (p1, p2) and (p3, p4)
                                # New edges: (p1, p3) and (p2, p4)
                                # Delta = d(p1, p3) + d(p2, p4) - d(p1, p2) - d(p3, p4)
                                delta = (graph_matrix[current_tour[i]][current_tour[j]] +
                                         graph_matrix[current_tour[(i+1)%N]][current_tour[(j+1)%N]] -
                                         graph_matrix[current_tour[i]][current_tour[(i+1)%N]] -
                                         graph_matrix[current_tour[j]][current_tour[(j+1)%N]])
    
    
                                # If delta is negative, we found an improvement
                                if delta < -1e-9: # Use tolerance for float comparison
                                    # Perform the swap by reversing the segment from i+1 to j
                                    # current_tour[i+1 ... j] needs to be reversed.
                                    # This corresponds to reversing the slice current_tour[i+1 : j+1].
                                    temp_segment = current_tour[i+1 : j+1]
                                    temp_segment.reverse()
                                    current_tour[i+1 : j+1] = temp_segment
    
                                    current_tour_length += delta
                                    improved = True
                                    # Restart search from the beginning
                                    break # Break j loop
                            if improved:
                                break # Break i loop
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not graph_matrix[0] or not isinstance(graph_matrix[0], list):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        if num_cities == 2:
            # The tour [0, 1] visits both cities. Returning to 0 completes the cycle.
            return [0, 1]
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city from the current city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and not the current city itself
                if city_idx not in visited:
                    distance = graph_matrix[current_city][city_idx]
                     # Ensure distance is valid (non-negative and finite)
                    if distance >= 0 and distance < min_dist:
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found.
            # This should only happen if all cities are visited, which is checked by the while loop condition.
            # However, as a safeguard, if it happens unexpectedly (e.g., graph not complete or disconnected), break.
            if next_city == -1:
                 # This indicates an issue, likely with graph structure assumptions or logic error.
                 # Return the partial tour found.
                 # print(f"Warning: Nearest Neighbor failed to find next city. Partial tour: {initial_tour}")
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 1
        # In a complete graph with positive distances, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with complete graph.
             # Return the potentially incomplete tour.
             # print(f"Warning: Initial tour is incomplete. Expected {num_cities}, got {len(initial_tour)}")
             return initial_tour # Return whatever was constructed
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 16. Program ID: b55e179b-7753-4725-b584-b520f6c70e29 (Gen: 4)
    - Score: 0.0447
    - Valid: True
    - Parent ID: 2b19831b-d3da-477a-a4c6-56818885d642
    - Timestamp: 1747528858.30
    - Code:
    ```python
    import math
    
    def calculate_tour_length(graph_matrix: list[list[float]], tour: list[int]) -> float:
        """
        Calculates the total length of a given tour based on the graph matrix.
    
        Args:
            graph_matrix: A square matrix where graph_matrix[i][j] is the distance
                          between city i and city j.
            tour: A list of city indices representing the order of visitation.
    
        Returns:
            The total length of the tour, including the return trip from the last
            city to the first.
        """
        total_length = 0.0
        num_cities = len(tour)
    
        if num_cities < 2:
            # A tour requires at least two cities to have a non-zero length path
            # or a defined loop. For 0 or 1 city, the loop length is 0.
            # If num_cities is 1, the length is graph_matrix[0][0] which is 0.
            if num_cities == 1:
                 if graph_matrix and len(graph_matrix) > 0:
                     return graph_matrix[tour[0]][tour[0]] # Should be 0
                 else:
                     return 0.0
            return 0.0
    
        # Sum distances between consecutive cities in the tour
        for i in range(num_cities - 1):
            current_city = tour[i]
            next_city = tour[i+1]
            # Ensure indices are within bounds (should be guaranteed by valid tour)
            # Added check for graph_matrix dimensions
            if 0 <= current_city < len(graph_matrix) and 0 <= next_city < len(graph_matrix) and \
               len(graph_matrix[current_city]) > next_city:
                 total_length += graph_matrix[current_city][next_city]
            else:
                 # Handle error: invalid city index in tour or graph matrix issue
                 # print(f"Error: Invalid city index in tour: {current_city} or {next_city}")
                 return float('inf') # Indicate invalid tour
    
        # Add the distance from the last city back to the first city
        last_city = tour[-1]
        first_city = tour[0]
        # Added check for graph_matrix dimensions
        if 0 <= last_city < len(graph_matrix) and 0 <= first_city < len(graph_matrix) and \
           len(graph_matrix[last_city]) > first_city:
            total_length += graph_matrix[last_city][first_city]
        else:
            # Handle error: invalid city index in tour or graph matrix issue
            # print(f"Error: Invalid city index in tour: {last_city} or {first_city}")
            return float('inf') # Indicate invalid tour
    
    
        return total_length
    
    def two_opt_improve(graph_matrix: list[list[float]], initial_tour: list[int]) -> list[int]:
        """
        Applies the 2-opt local search improvement heuristic to a tour.
    
        Args:
            graph_matrix: The distance matrix.
            initial_tour: The starting tour (a list of city indices).
    
        Returns:
            An improved tour (a locally optimal tour with respect to 2-opt swaps).
        """
        N = len(graph_matrix)
        if N < 3: # 2-opt requires at least 3 cities to perform a meaningful swap
            return list(initial_tour) # Cannot improve a tour of size 0, 1, or 2
    
        current_tour = list(initial_tour) # Create a mutable copy
        best_tour = list(current_tour)
        best_distance = calculate_tour_length(graph_matrix, current_tour)
    
        improved = True
        while improved:
            improved = False
            for i in range(N - 1): # Index of the first city of the first edge (0 to N-2)
                for j in range(i + 1, N): # Index of the first city of the second edge (i+1 to N-1)
                    # The segment to reverse is between index i+1 and j (inclusive of j).
                    # This swaps edges (current_tour[i], current_tour[i+1]) and (current_tour[j], current_tour[(j+1)%N]).
    
                    # If j is i+1, the segment is only one city long (current_tour[i+1]). Reversing does nothing.
                    # Skip adjacent edges.
                    if j == i + 1:
                        continue
    
                    # Create a new tour by performing the 2-opt swap
                    # Take the tour from the start up to i
                    # Reverse the segment from i+1 up to j
                    # Take the tour from j+1 to the end
                    new_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
    
                    # Calculate the distance of the new tour
                    new_distance = calculate_tour_length(graph_matrix, new_tour)
    
                    # If the new tour is shorter, update the best tour and distance
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improved = True
                        # Found an improvement, restart the search for improvements
                        # Break from inner loops to restart the outer while loop
                        break
                if improved:
                    break
    
            # If an improvement was made in this pass, update current_tour to the best found tour
            # and continue the while loop.
            if improved:
                current_tour = list(best_tour) # Make a new copy for the next iteration
    
        return best_tour
    
    
    def solve_tsp_heuristic(graph_matrix: list[list[float]]) -> list[int]:
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        followed by a 2-opt local search improvement.
    
        Args:
            graph_matrix: A list of lists representing the adjacency matrix of a complete graph.
                          graph_matrix[i][j] is the distance from city i to city j.
                          Assumes a square matrix where len(graph_matrix) is the number of cities.
    
        Returns:
            A list of node indices representing the order of cities to visit.
            The tour implicitly starts and ends at the first city in the list.
            Example: [0, 2, 1, 3] for 4 cities.
        """
        # Basic input validation
        if not graph_matrix or not isinstance(graph_matrix, list) or not all(isinstance(row, list) for row in graph_matrix):
            return []
    
        num_cities = len(graph_matrix)
    
        # Handle edge cases for number of cities
        if num_cities == 0:
            return []
        if num_cities == 1:
            return [0]
        # For 2 cities, the only possible tour visiting each once is trivial [0, 1].
        # NN starting at 0 finds this. 2-opt is not applicable.
        # We return [0, 1] as the sequence.
        if num_cities == 2:
            # Basic check to ensure graph_matrix is at least 2x2
            if len(graph_matrix[0]) < 2 or len(graph_matrix[1]) < 2:
                 # Invalid matrix for 2 cities
                 return []
            # The tour sequence is just the cities in some order, e.g., [0, 1].
            # The calculate_tour_length would add d(0,1) + d(1,0).
            return [0, 1]
    
    
        # 1. Generate initial tour using Nearest Neighbor heuristic
        # Start from city 0 for deterministic output
        start_city = 0
        initial_tour = [start_city]
        visited = {start_city}
    
        current_city = start_city
    
        # Build the tour using Nearest Neighbor
        while len(initial_tour) < num_cities:
            next_city = -1
            min_dist = float('inf')
    
            # Find the nearest unvisited city
            for city_idx in range(num_cities):
                # Check if city_idx is unvisited and the distance is valid
                # Added check for matrix column size
                if city_idx not in visited and city_idx < len(graph_matrix[current_city]):
                    distance = graph_matrix[current_city][city_idx]
                    # Assuming distances are positive as per problem description, and not infinity
                    if distance < min_dist and distance >= 0: # Ensure positive or zero distance
                        min_dist = distance
                        next_city = city_idx
    
            # If next_city is still -1, it means no unvisited city was found with a valid distance.
            # This should only happen if all cities are visited or if the graph is not complete/valid.
            # In a complete graph, this implies an issue.
            if next_city == -1:
                 # Could indicate a disconnected graph or invalid matrix entries (e.g., all inf).
                 # Return the partial tour found or an empty list if no progress could be made.
                 # Given the problem assumes a complete graph with positive distances, this is unexpected.
                 # Returning the partial tour is better than crashing or infinite loop.
                 break
    
            initial_tour.append(next_city)
            visited.add(next_city)
            current_city = next_city
    
        # Ensure the initial tour is complete if num_cities > 0 (handled 0 already)
        # In a complete graph, NN should always find a full tour if num_cities > 0
        if len(initial_tour) != num_cities:
             # This indicates an unexpected state - should not happen with a valid, complete graph.
             # Return the potentially incomplete tour.
             return initial_tour
    
        # 2. Improve the tour using 2-opt local search
        # The two_opt_improve function handles N < 3 internally, but we handled N=0, 1, 2 already.
        # For N >= 3, apply 2-opt.
        improved_tour = two_opt_improve(graph_matrix, initial_tour)
    
        return improved_tour
    ```

### 17. Program ID: dfa13b9b-c98d-4002-9547-1327c851815f (Gen: 4)
    - Score: 0.0447
    - Valid: True
    - Parent ID: bfc7e9e3-7712-467c-8b48-055800b2b5b0
    - Timestamp: 1747528858.30
    - Code:
    ```python
    import sys
    
    def calculate_tour_length(graph_matrix, tour):
        """Calculates the total length of a given tour including the return to the start."""
        num_cities = len(graph_matrix)
        if not tour or len(tour) != num_cities:
            # This case should ideally not be reached for a valid tour covering all cities
            # in a complete graph, but return a very large value as a safeguard.
            return sys.float_info.max
    
        total_length = 0
        try:
            for i in range(num_cities - 1):
                total_length += graph_matrix[tour[i]][tour[i+1]]
            # Add distance back to the start city to complete the cycle
            total_length += graph_matrix[tour[-1]][tour[0]]
        except IndexError:
            # Handle potential index errors if tour indices are out of bounds
            return sys.float_info.max
        except TypeError:
            # Handle potential type errors if graph_matrix or tour contains invalid types
             return sys.float_info.max
    
    
        return total_length
    
    def two_opt_swap(tour, i, k):
        """Performs a 2-opt swap on the tour between indices i and k (exclusive of k+1)."""
        # Assumes i < k
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        return new_tour
    
    def two_opt_optimize(graph_matrix, initial_tour):
        """
        Applies the 2-opt local search optimization algorithm to an initial tour.
    
        Args:
            graph_matrix: The adjacency matrix of distances.
            initial_tour: A list of city indices representing the starting tour.
    
        Returns:
            A list of city indices representing the optimized tour.
        """
        num_cities = len(graph_matrix)
        current_tour = list(initial_tour) # Create a mutable copy
        improved = True
    
        # Continue iterating as long as improvements are being made
        while improved:
            improved = False
            best_length = calculate_tour_length(graph_matrix, current_tour)
    
            # Iterate through all possible 2-opt swaps (pairs of non-adjacent edges)
            # We consider edges (i, i+1) and (k, k+1) for 0 <= i < k-1 <= num_cities-2
            # When swapping, we reverse the segment between i+1 and k.
            # The indices in the tour list are 0-based. An edge is between tour[i] and tour[i+1].
            # The 2-opt swap involves reversing the segment from index i+1 up to k.
            # The indices i and k in the loop below refer to the start of the segments to be swapped.
            # The actual edges are (tour[i], tour[i+1]) and (tour[k], tour[k+1])
            # The segment reversed is from tour[i+1] to tour[k].
            # The new edges become (tour[i], tour[k]) and (tour[i+1], tour[k+1])
    
            for i in range(num_cities - 1):
                for k in range(i + 2, num_cities): # k must be at least i+2 for a valid segment reversal
                    # Perform the 2-opt swap
                    new_tour = two_opt_swap(current_tour, i + 1, k) # indices i+1 to k inclusive are reversed
    
                    # Calculate the length of the new tour
                    new_length = calculate_tour_length(graph_matrix, new_tour)
    
                    # If the new tour is shorter, update and continue the process
                    if new_length < best_length:
                        current_tour = new_tour
                        best_length = new_length
                        improved = True
            # Note: The loop structure here performs one pass of potential swaps.
            # If an improvement is found, 'improved' is set to True, and the while loop continues
            # for another full pass until no improvements are found in a complete pass.
    
        return current_tour
    
    
    def solve_tsp_heuristic(graph_matrix):
        """
        Solves the Traveling Salesperson Problem using the Nearest Neighbor heuristic
        with multiple starting cities, followed by a 2-opt local search optimization.
    
        Args:
            graph_matrix: A list of lists of numbers (integers or floats) representing
                          the adjacency matrix of a complete graph. graph_matrix[i][j]
                          is the distance from city i to city j. Assumes a square matrix
                          where len(graph_matrix) is the number of cities. Distances
                          are positive, and graph_matrix[i][i] is 0.
    
        Returns:
            A list of node indices (integers from 0 to N-1) representing the order
            of cities in the found tour. The returned list is a permutation of [0...N-1].
            Returns an empty list for invalid input or zero cities. Returns [0] for a single city.
        """
        # Input validation
        if not graph_matrix or not isinstance(graph_matrix, list):
            return []
    
        num_cities = len(graph_matrix)
    
        if num_cities == 0:
            return []
    
        # Basic check for square matrix (assuming first row determines width)
        if any(len(row) != num_cities for row in graph_matrix):
             # Not a square matrix
             return []
    
        if num_cities == 1:
            return [0] # Tour is just the single city
    
        best_initial_tour = []
        min_initial_length = sys.float_info.max
    
        # --- Step 1: Generate an initial tour using Nearest Neighbor from multiple starts ---
        for start_city in range(num_cities):
            current_tour = [start_city]
            visited = {start_city}
            current_city = start_city
    
            while len(current_tour) < num_cities:
                next_city = -1
                min_dist = sys.float_info.max
    
                for city_idx in range(num_cities):
                    if city_idx not in visited:
                        distance = graph_matrix[current_city][city_idx]
                        # Find the true minimum positive distance
                        if distance < min_dist:
                             min_dist = distance
                             next_city = city_idx
    
                if next_city != -1:
                    current_tour.append(next_city)
                    visited.add(next_city)
                    current_city = next_city
                else:
                     # Should not happen in a complete graph with positive distances
                     break
    
            # If a full tour was constructed (visited all cities)
            if len(current_tour) == num_cities:
                current_total_length = calculate_tour_length(graph_matrix, current_tour)
    
                if current_total_length < min_initial_length:
                    min_initial_length = current_total_length
                    best_initial_tour = current_tour
    
        # If no valid initial tour was found (shouldn't happen for complete graph > 1), return empty
        if not best_initial_tour:
            return []
    
        # --- Step 2: Apply 2-opt optimization to the best initial tour ---
        final_optimized_tour = two_opt_optimize(graph_matrix, best_initial_tour)
    
        return final_optimized_tour
    ```