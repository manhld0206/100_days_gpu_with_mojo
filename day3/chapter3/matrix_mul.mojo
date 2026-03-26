from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext


@always_inline
def _matrix_mul_cal[
    dtype: DType
](
    m: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    row: Int,
    col: Int,
    width: Int
) -> Scalar[dtype]:
    var result: Scalar[dtype] = 0
    for k in range(width):
        result += m[row * width + k] * n[k * width + col]

    return result


def matrix_mul_kernel[
    dtype: DType
](
    m: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
):
    row = Int(global_idx.x)
    col = Int(global_idx.y)

    if row < width and col < width:
        p[row * width + col] = _matrix_mul_cal[dtype](m, n, row, col, width)


def matrix_mul_gpu[
    dtype: DType
](
    m: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    ctx: DeviceContext,
) raises:
    var d_m = ctx.enqueue_create_buffer[dtype](width**2)
    var d_n = ctx.enqueue_create_buffer[dtype](width**2)
    var d_p = ctx.enqueue_create_buffer[dtype](width**2)

    ctx.enqueue_copy(d_m, m)
    ctx.enqueue_copy(d_n, n)

    comptime block_dim_x = 32
    comptime block_dim_y = 32
    var grid_dim_x = ceildiv(width, block_dim_x)
    var grid_dim_y = grid_dim_x

    # Launch kernel
    ctx.enqueue_function[matrix_mul_kernel[dtype], matrix_mul_kernel[dtype]](
        d_m,
        d_n,
        d_p,
        width,
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(block_dim_x, block_dim_y, 1),
    )

    # Copy result from device to host
    ctx.enqueue_copy(p, d_p)

    # Synchronize to ensure completion
    ctx.synchronize()


def matrix_mul_cpu[
    dtype: DType
](
    m: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
):
    for row in range(width):
        for col in range(width):
            p[row * width + col] = _matrix_mul_cal(m, n, row, col, width)


def main() raises:
    var width: Int = 1000

    comptime dtype = DType.float32

    # Allocate host memory
    var m = alloc[Scalar[dtype]](width**2)
    var n = alloc[Scalar[dtype]](width**2)
    var p_gpu = alloc[Scalar[dtype]](width**2)
    var p_cpu = alloc[Scalar[dtype]](width**2)

    # Initialize input_data arrays
    for i in range(width**2):
        m[i] = Scalar[dtype](i) % 100
        n[i] = Scalar[dtype](i+1) % 100

    # Compute on GPU
    with DeviceContext() as ctx:
        matrix_mul_gpu[dtype](m, n, p_gpu, width, ctx)

    # Compute on CPU
    matrix_mul_cpu[dtype](m, n, p_cpu, width)

    var is_correct = True

    for i in range(width**2):
        if p_cpu[i] != p_gpu[i]:
            is_correct = False
            print(p_cpu[i])
            print(p_gpu[i])

    if is_correct:
        print("Success")
    else:
        print("Error")

    # Clean up
    m.free()
    n.free()
    p_cpu.free()
    p_gpu.free()
