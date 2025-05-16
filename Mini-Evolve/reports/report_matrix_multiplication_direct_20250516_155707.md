# Mini-Evolve Run Report
Generated: 2025-05-16 15:57:07
Problem: matrix_multiplication_direct
Database: db/program_database.db

---

## I. Overall Statistics
- Total programs in database: 7
- Valid programs: 6
- Invalid programs: 1
- Percentage valid: 85.71%
- Max score (valid programs): 1.0000
- Min score (valid programs): 1.0000
- Average score (valid programs): 1.0000
- Generations spanned: 0 to 3

## II. Best Program(s)
### Top Scorer:
- Program ID: fa9f9182-7877-4a28-8329-bc9dccd99026
- Score: 1.0000
- Generation Discovered: 3
- Parent ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf
- Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 6.067418027669191, "test_cases_passed": 20, "total_test_cases": 20}`
```python
def solve(matrix_a, matrix_b):
    if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
        return None

    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        return None

    result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result_matrix
```

## III. Top 5 Programs (by Score)

### 1. Program ID: fa9f9182-7877-4a28-8329-bc9dccd99026
    - Score: 1.0000
    - Generation: 3
    - Parent ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf
    - Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 6.067418027669191, "test_cases_passed": 20, "total_test_cases": 20}`
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return None
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
        return result_matrix
    ```

### 2. Program ID: 016460e7-f475-4189-9909-03c8f188ef0a
    - Score: 1.0000
    - Generation: 2
    - Parent ID: f06b34ff-3927-4508-83b7-78a9c085ea6e
    - Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 3.2376250019297004, "test_cases_passed": 20, "total_test_cases": 20}`
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return None
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

### 3. Program ID: 18860bb5-1a67-4605-89d8-bab0a442fe5d
    - Score: 1.0000
    - Generation: 2
    - Parent ID: f06b34ff-3927-4508-83b7-78a9c085ea6e
    - Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 5.336413974873722, "test_cases_passed": 20, "total_test_cases": 20}`
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

### 4. Program ID: 8d900d8d-134f-422a-8b85-bbc435a6f003
    - Score: 1.0000
    - Generation: 2
    - Parent ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf
    - Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 5.6019930052571, "test_cases_passed": 20, "total_test_cases": 20}`
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
        return result_matrix
    ```

### 5. Program ID: f06b34ff-3927-4508-83b7-78a9c085ea6e
    - Score: 1.0000
    - Generation: 1
    - Parent ID: 66e708f8-a353-4443-81df-8f784b95a654
    - Evaluation Details: `{"score": 1.0, "is_valid": true, "error_message": null, "execution_time_ms": 2.8529560077004135, "test_cases_passed": 20, "total_test_cases": 20}`
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return []
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

## IV. Evolutionary Lineage (Parent-Child)
- Gen: 0, ID: 66e708f8 (Score: 0.000, I)
    - Gen: 1, ID: 7dafb0cc (Score: 1.000, V)
        - Gen: 2, ID: 8d900d8d (Score: 1.000, V)
        - Gen: 3, ID: fa9f9182 (Score: 1.000, V)
    - Gen: 1, ID: f06b34ff (Score: 1.000, V)
        - Gen: 2, ID: 18860bb5 (Score: 1.000, V)
        - Gen: 2, ID: 016460e7 (Score: 1.000, V)

## V. All Programs by Generation & Timestamp

### 1. Program ID: 66e708f8-a353-4443-81df-8f784b95a654 (Gen: 0)
    - Score: 0.0000
    - Valid: False
    - Parent ID: None
    - Timestamp: 1747410037.45
    - Code:
    ```python
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
    ```

### 2. Program ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf (Gen: 1)
    - Score: 1.0000
    - Valid: True
    - Parent ID: 66e708f8-a353-4443-81df-8f784b95a654
    - Timestamp: 1747410101.29
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return []
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
        return result_matrix
    ```

### 3. Program ID: f06b34ff-3927-4508-83b7-78a9c085ea6e (Gen: 1)
    - Score: 1.0000
    - Valid: True
    - Parent ID: 66e708f8-a353-4443-81df-8f784b95a654
    - Timestamp: 1747410101.30
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return []
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

### 4. Program ID: 8d900d8d-134f-422a-8b85-bbc435a6f003 (Gen: 2)
    - Score: 1.0000
    - Valid: True
    - Parent ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf
    - Timestamp: 1747410169.19
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
        return result_matrix
    ```

### 5. Program ID: 18860bb5-1a67-4605-89d8-bab0a442fe5d (Gen: 2)
    - Score: 1.0000
    - Valid: True
    - Parent ID: f06b34ff-3927-4508-83b7-78a9c085ea6e
    - Timestamp: 1747410169.20
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return []
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

### 6. Program ID: 016460e7-f475-4189-9909-03c8f188ef0a (Gen: 2)
    - Score: 1.0000
    - Valid: True
    - Parent ID: f06b34ff-3927-4508-83b7-78a9c085ea6e
    - Timestamp: 1747410169.21
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return None
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for r_a in range(rows_a):
            for c_b in range(cols_b):
                for k in range(cols_a):
                    result_matrix[r_a][c_b] += matrix_a[r_a][k] * matrix_b[k][c_b]
    
        return result_matrix
    ```

### 7. Program ID: fa9f9182-7877-4a28-8329-bc9dccd99026 (Gen: 3)
    - Score: 1.0000
    - Valid: True
    - Parent ID: 7dafb0cc-e362-4d8c-90fe-88dc56074baf
    - Timestamp: 1747410283.78
    - Code:
    ```python
    def solve(matrix_a, matrix_b):
        if not matrix_a or not matrix_a[0] or not matrix_b or not matrix_b[0]:
            return None
    
        rows_a = len(matrix_a)
        cols_a = len(matrix_a[0])
        rows_b = len(matrix_b)
        cols_b = len(matrix_b[0])
    
        if cols_a != rows_b:
            return None
    
        result_matrix = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    
        return result_matrix
    ```