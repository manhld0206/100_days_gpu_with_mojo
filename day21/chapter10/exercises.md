# Text book exercises

1. block_dim = 1024 / 2 = 512 => number of warp is 512 / 32 = 16 => during 5th iteration, all 16 warps will have divergence
2. Same as above, block_dim = 512 and number of warp is 16. During 5th iteration, stride is 32, which mean all 16 warps won't have divergence. To be more specific, 1st warp is fully active while other warp is not.
