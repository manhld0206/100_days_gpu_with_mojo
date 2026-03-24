# Text book exercises

1. 
    a. 128 threads / 32 threads per warp = 4
    b. 128 threads / 32 threads per warp * 8 blocks = 32
    c. 
        i. each block has 3 active wrap => 3 * 8 blocks = 24 active wrap in the grid
        ii. 2 warps per block are divergent => 16 divergent blocks in the grid
        iii. 100%
        iv. 8 / 32 = 25%
        v. 24 / 32 = 75%
    d.
        i. all 32 warps
        ii. all 32 warps
        iii. 50%
    e.
        i. 3
        ii. 4 and 5
2. block size = 4 => 4 * 512 = 2048 threads in the grid
3. 1 warp will have divergent, and 1 warp have no data to run
4. 1 + 0.7 + 0 + 0.2 + 0.6 + 1.1 + 0.4 + 0.1 = 4.1 ms wait time => 4.1/2.4 = 17%
5. Technically right for older GPU but might not be future proof if NVIDIA change warps size or introduce other kind of thread scheduling
6. c
7. 
    a. Possible with 50%
    b. Possible with 50%
    c. Possible with 50%
    d. Possible with 100%
    e. Possible with 100%
8. 
    a. Full occupancy
    b. Limited by block
    c. Limited by register
9. Maximum 32 * 32 = 1024 > 512 threads per block of the device so not possible to run.
