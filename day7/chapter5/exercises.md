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
