def solve(matrix_a, matrix_b):
    # Basic structure, may not be correct or complete
    # This seed is deliberately flawed to give LLM something to fix.
    if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
        return [] # Or None
    
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        # print("Matrices not compatible for multiplication") # Not allowed in final code
        return [] # Or None

    # Initialize result matrix with zeros (or placeholder)
    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    # A very incomplete or incorrect multiplication attempt
    # Example: only operates on the first element or does element-wise which is wrong
    # for r_a in range(rows_a):
    #    for c_b in range(cols_b):
    #        if r_a < rows_b and c_b < cols_a : # incorrect indexing if trying to map directly
    #             result_matrix[r_a][c_b] = matrix_a[r_a][0] * matrix_b[0][c_b] # Still not full dot product logic
                        
    return result_matrix # Return whatever it managed to compute, likely just zeros or a badly computed matrix


# Example usage (not part of the evolved code, just for local testing if needed)
if __name__ == "__main__":
    # This part will be stripped by the evolution loop's loader
    # It's here for potential direct testing of the seed structure.
    matrix1 = [[1, 2], [3, 4]]
    matrix2 = [[2, 0], [1, 2]]
    # Expected for m1 * m2: [[ (1*2+2*1), (1*0+2*2) ], [ (3*2+4*1), (3*0+4*2) ]] = [[4,4],[10,8]]
    
    print("Seed solve function with m1, m2:")
    # Create a dummy version of solve for direct execution here
    # The actual solve will be extracted.
    def solve_local(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
        if cols_a != rows_b:
            return []
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
        # Seed doesn't actually multiply correctly yet.
        return result_matrix

    result = solve_local(matrix1, matrix2)
    print("Result of seed (likely just zeros or template):")
    for row in result:
        print(row)

    matrix3 = [[1,2,3],[4,5,6]] # 2x3
    matrix4 = [[7,8],[9,10],[11,12]] # 3x2
    # Expected for m3 * m4: [[58,64],[139,154]]
    result2 = solve_local(matrix3, matrix4)
    print("Result of seed for 2x3 and 3x2 (likely just zeros or template):")
    for row in result2:
        print(row) 