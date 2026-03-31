from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext


def vec_add_kernel[dtype: DType, simd_width: Int](
    a: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    c: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
):
    offset = Int(global_idx.x) * simd_width
    if offset + simd_width < n:
        vec_a = (a+offset).load[simd_width]()
        vec_b = (b+offset).load[simd_width]()
        vec_c = vec_a + vec_b
        (c+offset).store(vec_c)
    else:
        comptime for i in range(simd_width):
            if offset + i < n:
                c[offset + i] = a[offset + i] + b[offset + i]


def vec_add[dtype: DType, simd_width: Int](
    a: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    c: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    # Allocate device memory
    var a_d = ctx.enqueue_create_buffer[dtype](n)
    var b_d = ctx.enqueue_create_buffer[dtype](n)
    var c_d = ctx.enqueue_create_buffer[dtype](n)

    # Copy data from host to device
    ctx.enqueue_copy(a_d, a)
    ctx.enqueue_copy(b_d, b)

    # Calculate launch configuration
    var block_dim = 256
    var grid_dim = ceildiv(n, block_dim * simd_width)

    # Launch kernel
    ctx.enqueue_function[vec_add_kernel[dtype, simd_width], vec_add_kernel[dtype, simd_width]](
        a_d,
        b_d,
        c_d,
        n,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )

    # Copy result from device to host
    ctx.enqueue_copy(c, c_d)


def main() raises:
    comptime simd_width = 8
    comptime dtype = DType.float32
    var n = 1_000_000

    # Allocate host memory
    var a = alloc[Scalar[dtype]](n)
    var b = alloc[Scalar[dtype]](n)
    var c = alloc[Scalar[dtype]](n)
    var c_ref = alloc[Scalar[dtype]](n)

    # Initialize input arrays
    for i in range(n):
        a[i] = Scalar[dtype](i)
        b[i] = Scalar[dtype](i)

    # Perform vector addition on GPU
    with DeviceContext() as ctx:
        vec_add[dtype, simd_width](a, b, c, n, ctx)

    # Compute reference result on CPU
    for i in range(n):
        c_ref[i] = a[i] + b[i]

    # Verify results
    for i in range(n):
        if c[i] != c_ref[i]:
            print("Error at index", i, ":", c[i], "!=", c_ref[i])
            return

    print("Success")

    # Clean up
    a.free()
    b.free()
    c.free()
    c_ref.free()
