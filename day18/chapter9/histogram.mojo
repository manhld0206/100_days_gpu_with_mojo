from std.math import ceildiv
from std.gpu import (
    global_idx,
    block_idx,
    block_dim,
    grid_dim,
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
from std.os.atomic import Atomic

comptime Layout2D = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)


def histogram_kernel[
    num_bins: Int,
    coarse_factor: Int,
](
    image: LayoutTensor[DType.uint8, Layout2D, ImmutAnyOrigin],
    bins: LayoutTensor[DType.int32, Layout.row_major(num_bins), MutAnyOrigin],
):
    var bins_s = LayoutTensor[
        DType.int32,
        Layout.row_major(num_bins),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    for i in range(thread_idx.x, num_bins, block_dim.x):
        bins_s[thread_idx.x] = 0

    width = image.dim(1)

    segment = image.tile[coarse_factor, 1](
        global_idx.x // width, global_idx.x % width
    )

    # Check if valid segment
    if segment.dim(0) > 0:
        var current_value = segment[0, 0][0]
        var bin_r: Int32 = 1
        for i in range(1, segment.dim(0)):
            next_value = segment[i, 0][0]
            if next_value == current_value:
                bin_r += 1
            else:
                _ = Atomic.fetch_add(
                    bins_s.ptr_at_offset(Index(current_value)), bin_r
                )
                current_value = next_value
                bin_r = 1
        _ = Atomic.fetch_add(bins_s.ptr_at_offset(Index(current_value)), bin_r)

    barrier()

    for i in range(thread_idx.x, num_bins, block_dim.x):
        bin_value = bins_s[i][0]
        if bin_value > 0:
            _ = Atomic.fetch_add(bins.ptr_at_offset(Index(i)), bin_value)


def histogram_gpu[
    num_bins: Int,
    coarse_factor: Int,
](
    image: UnsafePointer[UInt8, ImmutExternalOrigin],
    bins: UnsafePointer[Int32, MutExternalOrigin],
    width: Int,
    height: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 32

    var image_layout = RuntimeLayout[Layout2D].row_major(Index(height, width))

    var d_image = ctx.enqueue_create_buffer[DType.uint8](height * width)
    var d_bins = ctx.enqueue_create_buffer[DType.int32](num_bins)

    ctx.enqueue_copy(d_image, image)
    ctx.enqueue_copy(d_bins, bins)

    var image_tensor = LayoutTensor[DType.uint8, Layout2D](
        d_image, image_layout
    )
    var bins_tensor = LayoutTensor[
        DType.int32, Layout.row_major(num_bins), MutAnyOrigin
    ](d_bins)

    var grid_dim = ceildiv(width * height, block_dim * coarse_factor)

    # Launch kernel
    ctx.enqueue_function[
        histogram_kernel[num_bins, coarse_factor],
        histogram_kernel[num_bins, coarse_factor],
    ](
        image_tensor,
        bins_tensor,
        grid_dim=(grid_dim),
        block_dim=(block_dim),
    )

    # Copy result from device to host
    ctx.enqueue_copy(bins, d_bins)

    # Synchronize to ensure completion
    ctx.synchronize()


def histogram_cpu[
    num_bins: Int,
](
    image: UnsafePointer[UInt8, ImmutExternalOrigin],
    bins: UnsafePointer[Int32, MutExternalOrigin],
    width: Int,
    height: Int,
) raises:
    for i in range(num_bins):
        bins[i] = 0
    for i, j in product(range(height), range(width)):
        bins[image[i * width + j]] += 1


def main() raises:
    var width: Int = 40
    var height: Int = 40

    comptime coarse_factor = 4
    comptime num_bins = 256

    # Allocate host memory
    var image = alloc[UInt8](width * height)
    var bins_gpu = alloc[Int32](num_bins)
    var bins_cpu = alloc[Int32](num_bins)

    # Initialize image arrays
    for i in range(width * height):
        image[i] = UInt8((i + 7) % 200)

    # Compute on GPU
    with DeviceContext() as ctx:
        histogram_gpu[num_bins, coarse_factor](
            image, bins_gpu, width, height, ctx
        )

    # Compute on CPU
    histogram_cpu[num_bins](image, bins_cpu, width, height)

    var is_correct = True

    for i in range(num_bins):
        if bins_cpu[i] != bins_gpu[i]:
            is_correct = False
            print("i", i)
            print("cpu", bins_cpu[i])
            print("gpu", bins_gpu[i])

    if is_correct:
        print("Success")
    else:
        raise "Error"

    # Clean up
    image.free()
    bins_cpu.free()
    bins_gpu.free()
