# Text book exercises

1. No because each threads just read exactly 1 element from global memory so there is no sharing between threads. Perhaps a vector load or vector save operation could change this.
2. 
3. The first barrier makes sure we do calculation only after all the data has been load. The second barrier make sure the next iteration doesn't accidentially corrupt previous iteration calculation.
4. Shared memory data can be shared among threads on the same SM while data on threads can not.
5. From the formula mentioned in the chapter: 32x32 => 32 times reduction
6. 1000 * 512 = 512000
7. 1000
8. 
    a. N times for each input matrix A and B
    b. N/T times
9. 
    a. 36 / (7 * (32 / 8)) < 200 / 100 => Memory bound
    b. 36 / (7 * (32 / 8)) > 300 / 250 => Compute bound
10. 
    a. Assuming BLOCK_SIZE is the same as BLOCK_WIDTH. Lacks of barrier between loading data from global to shared memory and writing result back from shared to global. So only BLOCK_SIZE = 1 (no synchronize needed) will work.
    b. Add the barrier between loading data from global to shared memory and writing result from shared to global memory.
11. 
    a. 8 * 128 (per thread) = 1024 versions
    b. 8 * 128 (per thread) = 1024 versions
    c. 8 (per block) versions
    d. 8 (per block) versions
    e. 129 * 4 bytes (float32) = 516B
    f. 10 (5 adds, 5 multiplies) OPs / (4 (reads) * 4B + 1 (read) * 4B + 1 (write) * 4B) = 5/12 (OP/B)
12. 
    a. 
        2048 (threads per SM) / 64 (threads) = 32 (blocks) => ok
        27 (registers) * 2048 (threads) = 55296 registers per SM => ok
        4 (KB) * 32 (blocks) = 128 (KB) per SM => not ok
        => Not full occupancy limited by shared memory
    b. 
        2048 (threads per SM) / 256 (threads) = 8 (blocks) => ok
        31 (registers) * 2048 (threads) = 63488 registers per SM => ok
        8 (KB) * 8 (blocks) = 64 (KB per SM) => ok
        => Full occupancy

