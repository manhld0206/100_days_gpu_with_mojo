# Text book exercises

1. 
    4, 6, 7, 1, 2, 8, 5, 2

    4, 10, 13, 8, 3, 10, 13, 7

    4, 10, 17, 18, 16, 18, 16, 17

    4, 10, 17, 18, 20, 28, 33, 35

2. 
    If stride < 32 => only warp 1 has divergence

    If stride >= 32 => stride = 32, 64, 128, ... (multiplier of warp size) => All the warps will either be active or inactive => No divergence

3. 
    Number of adds = (2048 - 1) + (2048 - 2) + (2048 - 4) + ... + (2048-1024)

    = 2048 * 11 - (1 + 2 + 4 + ... + 1024) = 22528 - (2048 - 1) = 20481

6. 
    4, 6, 7, 1, 2, 8, 5, 2

    4, 10, 7, 8, 2, 10, 5, 7

    4, 10, 7, 18, 2, 10, 5, 17

    4, 10, 7, 18, 2, 10, 5, 35

    4, 10, 7, 18, 2, 28, 5, 35

    4, 10, 17, 18, 20, 28, 33, 35



