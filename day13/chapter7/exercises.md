# Text book exercises

1. 8*5 + 2*3 + 5*1 = 51
2. [8, 21, 13, 20, 7]
3. 
    a. Do nothing filter

    b. Shift each element in array to the left
    
    c. Shift each element in array to the right
    
    d. Detect change
    
    e. Smoothing values
4. 
    a. 2 * filter radius (r) (r=(M-1)/2) = M - 1
    
    b. N * M
    
    c. N * M - 2*(1 + 2 + 3 + ... + r) = N*M - r(r+1)
5. See 6 for a more general solution
6. 
    a. Consider r1 = (M1 - 1) / 2, r2 = (M2 - 1) / 2
        
    => Result: (N1 + 2r1) * (N2 + 2r2) - N1*N2 = (N1-M1+1) * (N2-M2+1) - N1*N2
    
    b. N1 * N2 * M1 * M2
    
    c. Number of real operations is them sum of real operation of all cells
        
    Consider each cell (row i, column j), the number of real operations is:
            
    `number of real elements in the row` x `number of real elements in the column`

    Consider a row, each cell in that row has the same number of real elements in the column.

    Consider each column, each cell in that column has the same number of real elements in the row.
        
    => Pseudocode for calculate total number of real operation:
        
    ```python
    result = 0
    for i in range(N1):
        for j in range(N2):
            result += r[i] * c[j]
    ```

    This is the same as

    ```python
    sum_row = 0
    sum_col = 0
    for i in range(N1):
        sum_row += r[i]
    for j in range(N2):
        sum_col += c[j]
    result = sum_row * sum_col
    ```

    From 4c we have 
    
    `sum_col = N1 * M1 - r1(r1+1)`
    
    `sum_row = N2 * M2 - r2(r2+1)`

    => Final result `(N1 * M1 - r1(r1+1))*(N2 * M2 - r2(r2+1))`

7. 
    a. ((N + r -1)/r)^2
    
    b. (r + (M - 1)/2)^2

    c. (r + (M - 1)/2)^2 * element_size

    d. 

    number of blocks: ((N + r -1)/r)^2

    threads per block: r^2

    sharememory size: r^2 * element_size
