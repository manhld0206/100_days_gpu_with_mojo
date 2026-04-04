from std.math import ceildiv
from std.gpu import (
    global_idx,
    block_idx,
    thread_idx,
    barrier,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation


def conv2d_block_dim_out_tile_kernel[
    dtype: DType,
    block_dim: Int,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[
        Scalar[dtype], (2 * filter_radius + 1) * (2 * filter_radius + 1)
    ],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
):
    comptime filter_width = 2 * filter_radius + 1

    var col = block_idx.x * block_dim + thread_idx.x
    var row = block_idx.y * block_dim + thread_idx.y

    var nds = stack_allocation[
        block_dim**2,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    if row < height and row >= 0 and col < width and col >= 0:
        nds[thread_idx.y * block_dim + thread_idx.x] = n[row * width + col]
    else:
        nds[thread_idx.y * block_dim + thread_idx.x] = Scalar[dtype](0)

    barrier()

    if col < width and col >= 0 and row < height and row >= 0:
        p_value = Scalar[dtype](0)
        comptime for dr in range(-filter_radius, filter_radius + 1):
            comptime for dc in range(-filter_radius, filter_radius + 1):
                var nds_row = thread_idx.y + dr
                var nds_col = thread_idx.x + dc
                if (
                    nds_col < block_dim
                    and nds_col >= 0
                    and nds_row < block_dim
                    and nds_row >= 0
                ):
                    p_value += (
                        nds[nds_row * block_dim + nds_col]
                        * f[
                            (filter_radius + dr) * filter_width
                            + filter_radius
                            + dc
                        ]
                    )
                else:
                    var n_row = row + dr
                    var n_col = col + dc
                    if n_row >= 0 and n_row < height and n_col >=0 and n_col < width:
                        p_value += (
                            n[n_row * width + n_col]
                            * f[
                                (filter_radius + dr) * filter_width
                                + filter_radius
                                + dc
                            ]
                        )

        p[row * width + col] = p_value


def conv2d_gpu[
    dtype: DType,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[
        Scalar[dtype], (2 * filter_radius + 1) * (2 * filter_radius + 1)
    ],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 32

    var d_n = ctx.enqueue_create_buffer[dtype](width * height)
    var d_p = ctx.enqueue_create_buffer[dtype](width * height)

    ctx.enqueue_copy(d_n, n)
    ctx.enqueue_copy(d_p, p)

    var grid_dim_x = ceildiv(width, block_dim)
    var grid_dim_y = ceildiv(height, block_dim)

    # Launch kernel
    ctx.enqueue_function[
        conv2d_block_dim_out_tile_kernel[dtype, block_dim, filter_radius],
        conv2d_block_dim_out_tile_kernel[dtype, block_dim, filter_radius],
    ](
        d_n,
        f,
        d_p,
        width,
        height,
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(block_dim, block_dim, 1),
    )

    # Copy result from device to host
    ctx.enqueue_copy(p, d_p)

    # Synchronize to ensure completion
    ctx.synchronize()


def conv2d_cpu[
    dtype: DType,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[
        Scalar[dtype], (2 * filter_radius + 1) * (2 * filter_radius + 1)
    ],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
):
    comptime filter_width = 2 * filter_radius + 1
    for row in range(height):
        for col in range(width):
            var out = row * width + col
            p[out] = Scalar[dtype](0)
            comptime for dr in range(-filter_radius, filter_radius + 1):
                comptime for dc in range(-filter_radius, filter_radius + 1):
                    var nr = row + dr
                    var fr = dr + filter_radius
                    var nc = col + dc
                    var fc = dc + filter_radius
                    if 0 <= nr and nr < height and 0 <= nc and nc < width:
                        p[out] += f[fr * filter_width + fc] * n[nr * width + nc]


def main() raises:
    var width: Int = 1000
    var height: Int = 1500

    comptime filter_radius = 2
    comptime filter_width = 2 * filter_radius + 1

    comptime dtype = DType.float32

    # Allocate host memory
    var n = alloc[Scalar[dtype]](width * height)
    var p_gpu = alloc[Scalar[dtype]](width * height)
    var p_cpu = alloc[Scalar[dtype]](width * height)

    # Initialize input_data arrays
    for i in range(width * height):
        n[i] = Scalar[dtype]((i + 7) % 100)

    var f = InlineArray[Scalar[dtype], filter_width**2](fill=0)
    for i in range(filter_width**2):
        f[i] = Scalar[dtype](i % 4)

    # Compute on GPU
    with DeviceContext() as ctx:
        conv2d_gpu[dtype, filter_radius](n, f, p_gpu, width, height, ctx)

    # Compute on CPU
    conv2d_cpu[dtype, filter_radius](n, f, p_cpu, width, height)

    var is_correct = True

    for i in range(width * height):
        if p_cpu[i] != p_gpu[i]:
            is_correct = False
            print(p_cpu[i])
            print(p_gpu[i])

    if is_correct:
        print("Success")
    else:
        print("Error")

    # Clean up
    n.free()
    p_cpu.free()
    p_gpu.free()
