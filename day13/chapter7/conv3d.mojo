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
from layout import (
    UNKNOWN_VALUE,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
)
from std.utils.index import Index
from std.itertools import product

comptime Layout3D = Layout.row_major(
    UNKNOWN_VALUE, UNKNOWN_VALUE, UNKNOWN_VALUE
)


def conv3d_block_dim_in_tile_kernel[
    dtype: DType,
    block_dim: Int,
    filter_radius: Int,
](
    n: LayoutTensor[dtype, Layout3D, ImmutAnyOrigin],
    p: LayoutTensor[dtype, Layout3D, MutAnyOrigin],
    f: InlineArray[
        Scalar[dtype],
        (2 * filter_radius + 1)
        * (2 * filter_radius + 1)
        * (2 * filter_radius + 1),
    ],
):
    comptime filter_width = 2 * filter_radius + 1
    comptime out_dim = block_dim - 2 * filter_radius

    shape = n.get_shape()
    height = shape[0]
    width = shape[1]
    depth = shape[2]

    var col = block_idx.x * out_dim + thread_idx.x - filter_radius
    var row = block_idx.y * out_dim + thread_idx.y - filter_radius
    var dep = block_idx.z * out_dim + thread_idx.z - filter_radius

    var nds = LayoutTensor[
        dtype,
        Layout.row_major(block_dim, block_dim, block_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if (
        col < width
        and col >= 0
        and row < height
        and row >= 0
        and dep < depth
        and dep >= 0
    ):
        nds[thread_idx.z, thread_idx.y, thread_idx.x] = n[dep, row, col]
    else:
        nds[thread_idx.z, thread_idx.y, thread_idx.x] = 0

    barrier()

    if (
        col < width
        and col >= 0
        and row < height
        and row >= 0
        and dep < depth
        and dep >= 0
    ):
        tile_col = thread_idx.x - filter_radius
        tile_row = thread_idx.y - filter_radius
        tile_dep = thread_idx.z - filter_radius

        if (
            tile_col >= 0
            and tile_col < out_dim
            and tile_row >= 0
            and tile_row < out_dim
            and tile_dep >= 0
            and tile_dep < out_dim
        ):
            p_value = n.element_type(0)
            comptime for i, j, k in product(
                range(filter_width), range(filter_width), range(filter_width)
            ):
                p_value += (
                    nds[(tile_dep + k), (tile_row + i), (tile_col + j)]
                    * f[i * filter_width * filter_width + j * filter_width + k]
                )

            p[dep, row, col] = p_value


def conv3d_gpu[
    dtype: DType,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[
        Scalar[dtype],
        (2 * filter_radius + 1)
        * (2 * filter_radius + 1)
        * (2 * filter_radius + 1),
    ],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
    depth: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 8
    comptime out_dim = block_dim - 2 * filter_radius

    var tensor_3d = RuntimeLayout[Layout3D].row_major(
        Index(depth, height, width)
    )

    var d_n = ctx.enqueue_create_buffer[dtype](width * height * depth)
    var d_p = ctx.enqueue_create_buffer[dtype](width * height * depth)

    ctx.enqueue_copy(d_n, n)
    ctx.enqueue_copy(d_p, p)

    var n_tensor = LayoutTensor[dtype, Layout3D](d_n, tensor_3d)
    var p_tensor = LayoutTensor[dtype, Layout3D, MutAnyOrigin](d_p, tensor_3d)

    var grid_dim_x = ceildiv(width, out_dim)
    var grid_dim_y = ceildiv(height, out_dim)
    var grid_dim_z = ceildiv(depth, out_dim)

    # Launch kernel
    ctx.enqueue_function[
        conv3d_block_dim_in_tile_kernel[dtype, block_dim, filter_radius],
        conv3d_block_dim_in_tile_kernel[dtype, block_dim, filter_radius],
    ](
        n_tensor,
        p_tensor,
        f,
        grid_dim=(grid_dim_x, grid_dim_y, grid_dim_z),
        block_dim=(block_dim, block_dim, block_dim),
    )

    # Copy result from device to host
    ctx.enqueue_copy(p, d_p)

    # Synchronize to ensure completion
    ctx.synchronize()


def conv3d_cpu[
    dtype: DType,
    filter_radius: Int,
](
    n: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    f: InlineArray[
        Scalar[dtype],
        (2 * filter_radius + 1)
        * (2 * filter_radius + 1)
        * (2 * filter_radius + 1),
    ],
    p: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    width: Int,
    height: Int,
    depth: Int,
):
    comptime filter_width = 2 * filter_radius + 1
    for row, col, dep in product(range(height), range(width), range(depth)):
        var out = dep * width * height + row * width + col
        p[out] = Scalar[dtype](0)
        for dr, dc, dd in product(
            range(-filter_radius, filter_radius + 1),
            range(-filter_radius, filter_radius + 1),
            range(-filter_radius, filter_radius + 1),
        ):
            var nr = row + dr
            var fr = dr + filter_radius
            var nc = col + dc
            var fc = dc + filter_radius
            var nd = dep + dd
            var fd = dd + filter_radius
            if 0 <= nr and nr < height and 0 <= nc and nc < width and 0 <= nd and nd < depth:
                p[out] += (
                    f[fd * filter_width * filter_width + fr * filter_width + fc]
                    * n[nd * height * width + nr * width + nc]
                )


def main() raises:
    var width: Int = 100
    var height: Int = 100
    var depth: Int = 100

    comptime filter_radius = 2
    comptime filter_width = 2 * filter_radius + 1

    comptime dtype = DType.float32

    # Allocate host memory
    var n = alloc[Scalar[dtype]](width * height * depth)
    var p_gpu = alloc[Scalar[dtype]](width * height * depth)
    var p_cpu = alloc[Scalar[dtype]](width * height * depth)

    # Initialize input_data arrays
    for i in range(width * height * width):
        n[i] = Scalar[dtype]((i + 7) % 100)

    var f = InlineArray[Scalar[dtype], filter_width**3](fill=0)
    for i in range(filter_width**3):
        f[i] = Scalar[dtype](i % 4)

    # Compute on GPU
    with DeviceContext() as ctx:
        conv3d_gpu[dtype, filter_radius](n, f, p_gpu, width, height, depth, ctx)

    # Compute on CPU
    conv3d_cpu[dtype, filter_radius](n, f, p_cpu, width, height, depth)

    var is_correct = True

    for i in range(width * height * depth):
        if p_cpu[i] != p_gpu[i]:
            is_correct = False
            print("i", i)
            print("cpu", p_cpu[i])
            print("gpu", p_gpu[i])

    if is_correct:
        print("Success")
    else:
        print("Error")

    # Clean up
    n.free()
    p_cpu.free()
    p_gpu.free()
