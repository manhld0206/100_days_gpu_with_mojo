from std.math import ceildiv
from std.gpu import (
    global_idx,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
    barrier,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation


@always_inline
def _matrix_mul_cal[
    dtype: DType,
    address_space: AddressSpace = AddressSpace.GENERIC,
](
    m: UnsafePointer[
        Scalar[dtype], ImmutExternalOrigin, address_space=address_space
    ],
    n: UnsafePointer[
        Scalar[dtype], ImmutExternalOrigin, address_space=address_space
    ],
    row: Int,
    col: Int,
    width: Int,
) -> Scalar[dtype]:
    var result: Scalar[dtype] = 0
    for k in range(width):
        result += m[row * width + k] * n[k * width + col]

    return result


def matrix_mul_kernel[
    dtype: DType,
    tile_width: Int,
    is_n_tranpose: Bool = False,
](
    m: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
):
    var mds = stack_allocation[
        tile_width**2,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var nds = stack_allocation[
        tile_width**2,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var row = tile_width * Int(block_idx.y) + ty
    var col = tile_width * Int(block_idx.x) + tx

    var p_value = Scalar[dtype](0)

    var num_phases = ceildiv(width, tile_width)

    for ph in range(num_phases):
        if row < width and (ph * tile_width + tx) < width:
            mds[ty * tile_width + tx] = m[row * width + ph * tile_width + tx]
        else:
            mds[ty * tile_width + tx] = 0

        if (ph * tile_width + ty) < width and col < width:
            comptime if is_n_tranpose:
                nds[ty * tile_width + tx] = n[
                    col * width + ph * tile_width + ty
                ]
            else:
                nds[ty * tile_width + tx] = n[
                    (ph * tile_width + ty) * width + col
                ]
        else:
            nds[ty * tile_width + tx] = 0

        barrier()

        p_value += _matrix_mul_cal[dtype, AddressSpace.SHARED](
            mds, nds, ty, tx, tile_width
        )

        barrier()

    # Write result to global memory with boundary check
    if row < width and col < width:
        p[row * width + col] = p_value


def matrix_mul_gpu[
    dtype: DType,
    is_n_transpose: Bool = False,
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
    comptime tile_width = 32
    var grid_dim_x = ceildiv(width, block_dim_x)
    var grid_dim_y = grid_dim_x

    # Launch kernel
    ctx.enqueue_function[
        matrix_mul_kernel[dtype, tile_width, is_n_transpose],
        matrix_mul_kernel[dtype, tile_width, is_n_transpose],
    ](
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
    var n_transpose = alloc[Scalar[dtype]](width**2)
    var p_gpu = alloc[Scalar[dtype]](width**2)
    var p_cpu = alloc[Scalar[dtype]](width**2)

    # Initialize input_data arrays
    for i in range(width**2):
        m[i] = Scalar[dtype](i) % 100
        n[i] = Scalar[dtype](i + 1) % 100

    for i in range(width):
        for j in range(width):
            n_transpose[j * width + i] = n[i * width + j]

    # Compute on GPU
    with DeviceContext() as ctx:
        matrix_mul_gpu[dtype, True](m, n_transpose, p_gpu, width, ctx)

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
