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
from std.gpu.memory import AddressSpace, async_copy_wait_all
from std.gpu.primitives.warp import shuffle_down, shuffle_up
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


@always_inline
def warp_scan[dtype: DType](val: Scalar[dtype]) -> Scalar[dtype]:
    var partial_sum = val

    # Unrolled warp reduction
    comptime LOG2_WARP_SIZE = log2_floor(WARP_SIZE)

    for i in range(LOG2_WARP_SIZE):
        offset = 1 << i
        var left_value = shuffle_up(partial_sum, UInt32(offset))
        if lane_id() >= offset:
            partial_sum += left_value

    return partial_sum


@always_inline
def block_scan[
    dtype: DType, //, num_warps: Int
](val: Scalar[dtype]) -> Scalar[dtype]:
    comptime assert num_warps <= WARP_SIZE

    var warp_idx = warp_id()
    var lane_idx = lane_id()
    var thread_id = thread_idx.x

    var warp_scan_val_1 = warp_scan(val)

    var warp_sum_s = LayoutTensor[
        dtype,
        Layout.row_major(num_warps),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    if lane_idx == WARP_SIZE - 1:
        if warp_idx < num_warps:
            warp_sum_s[warp_idx] = warp_scan_val_1

    barrier()

    if warp_idx == 0:
        var warp_scan_val_2: Scalar[dtype]
        if thread_id < num_warps:
            warp_scan_val_2 = warp_sum_s[thread_id][0]
        else:
            warp_scan_val_2 = 0
        warp_scan_val_2 = warp_scan(warp_scan_val_2)
        if thread_id < num_warps:
            warp_sum_s[thread_id] = warp_scan_val_2

    barrier()

    if warp_idx > 0 and warp_idx < num_warps:
        warp_scan_val_1 += warp_sum_s[warp_idx - 1][0]

    return warp_scan_val_1


def sum_scan_kernel[
    dtype: DType,
    block_dim: Int,
    coarse_factor: Int = 4,
](
    input: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin],
    output: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin],
):
    comptime assert block_dim <= WARP_SIZE * WARP_SIZE
    comptime num_warps = ceildiv(block_dim, WARP_SIZE)

    var global_id = global_idx.x
    var block_id = block_idx.x
    var thread_id = thread_idx.x
    var n = input.dim(0)

    var segment = input.tile[coarse_factor](global_id)
    var output_segment = output.tile[coarse_factor](global_id)

    var buffer_r = InlineArray[Scalar[dtype], coarse_factor](fill=0)
    buffer_r[0] = segment[0][0]
    comptime for c in range(1, coarse_factor):
        if c < segment.dim(0):
            buffer_r[c] = buffer_r[c - 1] + segment[c][0]
        else:
            buffer_r[c] = buffer_r[c - 1]

    barrier()

    var thread_sum = buffer_r[coarse_factor - 1]
    thread_sum = block_scan[num_warps](thread_sum)

    var thread_sums = LayoutTensor[
        dtype,
        Layout.row_major(block_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    thread_sums[thread_id] = thread_sum
    barrier()

    if thread_id > 0:
        previous_thread_sum = thread_sums[thread_id - 1][0]
    else:
        previous_thread_sum = 0

    comptime for c in range(coarse_factor):
        if c < output_segment.dim(0):
            output_segment[c] = buffer_r[c] + previous_thread_sum


def sum_scan_gpu[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 1024
    comptime coarse_factor = 4

    var layout = RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
        Index(n)
    )

    var d_input = ctx.enqueue_create_buffer[dtype](n)
    var d_output = ctx.enqueue_create_buffer[dtype](n)

    ctx.enqueue_copy(d_input, input)

    var input_tensor = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        d_input, layout
    )
    var output_tensor = LayoutTensor[
        mut=True, dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](d_output, layout)

    var grid_dim = ceildiv(n, block_dim)

    # Launch kernel
    ctx.enqueue_function[
        sum_scan_kernel[dtype, block_dim, coarse_factor],
        sum_scan_kernel[dtype, block_dim, coarse_factor],
    ](
        input_tensor,
        output_tensor,
        grid_dim=(grid_dim),
        block_dim=(block_dim),
    )

    # Copy result from device to host
    ctx.enqueue_copy(output, d_output)

    # Synchronize to ensure completion
    ctx.synchronize()


def sum_scan_cpu[
    dtype: DType,
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
) raises:
    output[0] = input[0]
    for i in range(1, n):
        output[i] = output[i - 1] + input[i]


def main() raises:
    var n = 4000

    comptime dtype = DType.int32

    # Allocate host memory
    var input = alloc[Scalar[dtype]](n)
    var output_gpu = alloc[Scalar[dtype]](n)
    var output_cpu = alloc[Scalar[dtype]](n)

    for i in range(n):
        input[i] = Int32((i + 7) % 100)
        output_gpu[i] = -1
        output_cpu[i] = -2

    # Compute on GPU
    with DeviceContext() as ctx:
        sum_scan_gpu[dtype](input, output_gpu, n, ctx)

    # Compute on CPU
    sum_scan_cpu[dtype](input, output_cpu, n)

    var is_correct = True
    for i in range(n):
        if output_cpu[i] != output_gpu[i]:
            is_correct = False
            print("i", i)
            print("cpu", output_cpu[i])
            print("gpu", output_gpu[i])

    if is_correct:
        print("Success")
    else:
        raise "Error"

    # Clean up
    input.free()
    output_cpu.free()
    output_gpu.free()
