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


def conv2d_block_dim_in_tile_kernel[
    dtype: DType,
    block_dim: Int,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[Scalar[dtype], (2 * filter_radius + 1)*(2 * filter_radius + 1)],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
):
    comptime filter_width = 2 * filter_radius + 1
    comptime out_dim = block_dim - 2 * filter_radius

    var col = block_idx.x * out_dim + thread_idx.x - filter_radius
    var row = block_idx.y * out_dim + thread_idx.y - filter_radius

    var nds = stack_allocation[
        block_dim**2,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    if col < width and col >= 0 and row < height and row >=0:
        nds[thread_idx.y * block_dim + thread_idx.x] = n[row * width + col]
    else:
        nds[thread_idx.y * block_dim + thread_idx.x] = Scalar[dtype](0)

    barrier()

    if col < width and col >= 0 and row < height and row >=0:
        tile_col = thread_idx.x - filter_radius
        tile_row = thread_idx.y - filter_radius

        if tile_col >= 0 and tile_col < out_dim and tile_row >= 0 and tile_row < out_dim:
            p_value = Scalar[dtype](0)
            for i in range(filter_width):
                for j in range(filter_width):
                    p_value += nds[(tile_row + i) * block_dim + tile_col + j] * f[i * filter_width + j]

            p[row * width + col] = p_value


def conv2d_gpu[
    dtype: DType,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[Scalar[dtype], (2 * filter_radius + 1)*(2 * filter_radius + 1)],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
    ctx: DeviceContext
) raises:
    comptime block_dim = 32
    comptime out_dim = block_dim - 2 * filter_radius

    var d_n = ctx.enqueue_create_buffer[dtype](width*height)
    var d_p = ctx.enqueue_create_buffer[dtype](width*height)

    ctx.enqueue_copy(d_n, n)
    ctx.enqueue_copy(d_p, p)

    var grid_dim_x = ceildiv(width, out_dim)
    var grid_dim_y = ceildiv(height, out_dim)

    # Launch kernel
    ctx.enqueue_function[
        conv2d_block_dim_in_tile_kernel[dtype, block_dim, filter_radius],
        conv2d_block_dim_in_tile_kernel[dtype, block_dim, filter_radius],
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
    f: InlineArray[Scalar[dtype], (2 * filter_radius + 1)*(2 * filter_radius + 1)],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
):
    comptime filter_width = 2 * filter_radius + 1
    for row in range(height):
        for col in range(width):
            var out = row * width + col
            p[out] = Scalar[dtype](0)
            for dr in range(-filter_radius, filter_radius + 1):
                for dc in range(-filter_radius, filter_radius + 1):
                    var nr = row + dr
                    var fr = dr + filter_radius
                    var nc = col + dc
                    var fc = dc + filter_radius
                    if 0 <= nr and nr < height and 0 <= nc and nc < width:
                        p[out] += (
                            f[fr * filter_width + fc] * n[nr * width + nc]
                        )


def main() raises:
    var width: Int = 1000
    var height: Int = 1500

    comptime filter_radius = 2
    comptime filter_width = 2 * filter_radius + 1

    comptime dtype = DType.float32

    # Allocate host memory
    var n = alloc[Scalar[dtype]](width*height)
    var p_gpu = alloc[Scalar[dtype]](width*height)
    var p_cpu = alloc[Scalar[dtype]](width*height)

    # Initialize input_data arrays
    for i in range(width * height):
        n[i] = Scalar[dtype](i + 7) % 100

    var f = InlineArray[Scalar[dtype], filter_width**2](fill=2)

    # Compute on GPU
    with DeviceContext() as ctx:
        conv2d_gpu[dtype, filter_radius](n, f, p_gpu, width, height, ctx)

    # Compute on CPU
    conv2d_cpu[dtype, filter_radius](n, f, p_cpu, width, height)

    var is_correct = True

    for i in range(width*height):
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
