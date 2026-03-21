from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext


def vec_add_kernel[dtype: DType](
    a: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    c: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        c[i] = a[i] + b[i]


def vec_add[dtype: DType](
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
    var grid_dim = ceildiv(n, block_dim)

    # Launch kernel
    ctx.enqueue_function[vec_add_kernel[dtype], vec_add_kernel[dtype]](
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
    var n = 1000

    # Allocate host memory
    var a = alloc[Float32](n)
    var b = alloc[Float32](n)
    var c = alloc[Float32](n)
    var c_ref = alloc[Float32](n)

    # Initialize input arrays
    for i in range(n):
        a[i] = Float32(i)
        b[i] = Float32(i)

    # Perform vector addition on GPU
    with DeviceContext() as ctx:
        vec_add(a, b, c, n, ctx)

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
