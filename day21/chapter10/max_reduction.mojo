from std.math import ceildiv
from std.gpu import (
    global_idx,
    block_idx,
    block_dim,
    grid_dim,
    thread_idx,
    barrier,
    warp_id,
    lane_id,
    WARP_SIZE,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives.warp import shuffle_down
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
from std.atomic import Atomic
from std.bit import log2_floor
from std.math import max
from std.utils.numerics import min_or_neg_inf


def warp_reduce[
    dtype: DType, simd_width: Int
](val: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    var partial_max = val

    # Unrolled warp reduction
    comptime LOG2_WARP_SIZE = log2_floor(WARP_SIZE)

    comptime for i in range(LOG2_WARP_SIZE):
        comptime offset = 1 << (LOG2_WARP_SIZE - 1 - i)
        partial_max = max(
            partial_max, shuffle_down(partial_max, UInt32(offset))
        )

    return partial_max


def max_reduction_kernel[
    dtype: DType,
    block_dim: Int,
    coarse_factor: Int,
](
    input: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
):
    assert block_dim <= WARP_SIZE * WARP_SIZE

    var segment = input.tile[coarse_factor * 2 * block_dim](block_idx.x)
    var segment_len = segment.dim(0)
    var i = thread_idx.x
    var partial_max: segment.element_type
    if i < segment_len:
        partial_max = segment[i]
    else:
        partial_max = min_or_neg_inf[dtype]()

    for c in range(1, coarse_factor * 2):
        if i + c * block_dim < segment_len:
            partial_max = max(partial_max, segment[i + c * block_dim])
        else:
            break

    var warp_idx = warp_id()

    partial_max = warp_reduce(partial_max)

    var partial_max_s = LayoutTensor[
        dtype,
        Layout.row_major(block_dim / WARP_SIZE),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if lane_id() == 0:
        partial_max_s[warp_idx] = partial_max

    barrier()

    if warp_idx == 0:
        if i < block_dim / WARP_SIZE:
            partial_max = partial_max_s[i]
        else:
            partial_max = 0
        partial_max = warp_reduce(partial_max)
        if thread_idx.x == 0:
            _ = Atomic.max(output, partial_max[0])


def max_reduction_gpu[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 512
    comptime coarse_factor = 4

    var layout = RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
        Index(n)
    )

    var d_input = ctx.enqueue_create_buffer[dtype](n)
    var d_output = ctx.enqueue_create_buffer[dtype](1)

    ctx.enqueue_copy(d_input, input)

    var input_tensor = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        d_input, layout
    )

    var grid_dim = ceildiv(n, block_dim * 2 * coarse_factor)

    # Launch kernel
    ctx.enqueue_function[
        max_reduction_kernel[dtype, block_dim, coarse_factor],
        max_reduction_kernel[dtype, block_dim, coarse_factor],
    ](
        input_tensor,
        d_output,
        grid_dim=(grid_dim),
        block_dim=(block_dim),
    )

    # Copy result from device to host
    ctx.enqueue_copy(output, d_output)

    # Synchronize to ensure completion
    ctx.synchronize()


def max_reduction_cpu[
    dtype: DType,
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
) raises:
    var max_val = min_or_neg_inf[dtype]()
    for i in range(n):
        max_val = max(max_val, input[i])
    output[0] = max_val


def main() raises:
    var n = 1000000

    comptime dtype = DType.int32

    # Allocate host memory
    var input = alloc[Scalar[dtype]](n)
    var output_gpu = alloc[Scalar[dtype]](1)
    var output_cpu = alloc[Scalar[dtype]](1)

    # Initialize image arrays
    for i in range(n):
        input[i] = Int32((i + 7) % 700000)

    # Compute on GPU
    with DeviceContext() as ctx:
        max_reduction_gpu[dtype](input, output_gpu, n, ctx)

    # Compute on CPU
    max_reduction_cpu[dtype](input, output_cpu, n)

    var is_correct = True
    if output_cpu[0] != output_gpu[0]:
        is_correct = False
        print("cpu", output_cpu[0])
        print("gpu", output_gpu[0])

    if is_correct:
        print("Success")
    else:
        raise "Error"

    # Clean up
    input.free()
    output_cpu.free()
    output_gpu.free()
