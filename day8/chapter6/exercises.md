# Text book exercises

1. 
    a. coalesced
    b. not applicable (shared memory)
    c. coalesced
    d. uncoalesced
    e. not applicable (shared memory)
    f. not applicable (shared memory)
    g. coalesced
    h. not applicable (shared memory)
    i. uncoalesced
2. 
3. Mutiple of 32 (threads per warp). However >= 64 is usually impossible because of the usual max 1024 threads per block
4. 
5. Use 32 / GCD(stride, 32)
    a. 1 bank => 32 ways conflict 
    b. 32 banks => no conflict
    c. 4 banks => 8 ways conflict
    d. 2 banks => 16 ways conflict
    e. 4 banks => 8 ways conflict
    f. 4 banks => 8 ways conflict
    g. 32 banks => no conflict


