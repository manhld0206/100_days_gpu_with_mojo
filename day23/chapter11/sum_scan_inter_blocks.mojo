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
from std.atomic import Atomic, Ordering
from std.bit import log2_floor
from std.time import perf_counter


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


def inter_block_scan[
    dtype: DType,
    block_dim: Int,
](
    val: Scalar[dtype],
    bid: Int,
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ],
    flags: LayoutTensor[
        dtype.int32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ],
) -> Scalar[dtype]:
    previous_block_partial_sum = stack_allocation[
        1, dtype, address_space=AddressSpace.SHARED
    ]()
    if thread_idx.x == block_dim - 1:
        if bid > 0:
            while (
                Atomic.fetch_add[ordering=Ordering.ACQUIRE](
                    flags.ptr_at_offset(Index(bid - 1)), 0
                )
                == 0
            ):
                pass
            previous_block_partial_sum[0] = partial_sums[bid - 1][0]
        else:
            previous_block_partial_sum[0] = 0

        partial_sums[bid] = previous_block_partial_sum[0] + val
        _ = Atomic.fetch_add[ordering=Ordering.RELEASE](
            flags.ptr_at_offset(Index(bid)), 1
        )

    barrier()
    return previous_block_partial_sum[0]


def sum_scan_kernel[
    dtype: DType,
    block_dim: Int,
    coarse_factor: Int = 4,
](
    input: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin],
    output: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin],
    block_counter: UnsafePointer[Scalar[dtype.int32], MutAnyOrigin],
    partial_sums: LayoutTensor[
        dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ],
    flags: LayoutTensor[
        dtype.int32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ],
):
    comptime assert block_dim <= WARP_SIZE * WARP_SIZE
    comptime num_warps = ceildiv(block_dim, WARP_SIZE)

    var bid_s = stack_allocation[
        1, dtype.int32, address_space=AddressSpace.SHARED
    ]()

    var thread_id = thread_idx.x
    var n = input.dim(0)

    if thread_id == 0:
        bid_s[0] = Atomic.fetch_add[ordering=Ordering.RELAXED](block_counter, 1)

    barrier()

    var bid = Int(bid_s[0])
    var global_id = bid * block_dim + thread_id

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
        buffer_r[c] = buffer_r[c] + previous_thread_sum

    previous_block_partial_sum = inter_block_scan[dtype, block_dim](
        buffer_r[coarse_factor - 1], bid, partial_sums, flags
    )

    comptime for c in range(coarse_factor):
        buffer_r[c] = buffer_r[c] + previous_block_partial_sum

    comptime for c in range(coarse_factor):
        if c < output_segment.dim(0):
            output_segment[c] = buffer_r[c]


def sum_scan_gpu[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 1024
    comptime coarse_factor = 16

    var num_blocks = ceildiv(n, block_dim * coarse_factor)

    var layout_n = RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
        Index(n)
    )
    var layout_num_blocks = RuntimeLayout[
        Layout.row_major(UNKNOWN_VALUE)
    ].row_major(Index(num_blocks))

    var d_input = ctx.enqueue_create_buffer[dtype](n)
    var d_output = ctx.enqueue_create_buffer[dtype](n)
    var d_block_counter = ctx.enqueue_create_buffer[DType.int32](1)
    var d_partial_sums = ctx.enqueue_create_buffer[dtype](num_blocks)
    var d_flags = ctx.enqueue_create_buffer[DType.int32](num_blocks)

    ctx.enqueue_copy(d_input, input)

    var h_counter = alloc[Int32](1)
    var h_flags = alloc[Int32](num_blocks)
    h_counter[0] = 0
    for i in range(num_blocks):
        h_flags[i] = 0
    ctx.enqueue_copy(d_block_counter, h_counter)
    ctx.enqueue_copy(d_flags, h_flags)
    h_counter.free()
    h_flags.free()

    ctx.synchronize()

    var input_tensor = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        d_input, layout_n
    )
    var output_tensor = LayoutTensor[
        mut=True, dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](d_output, layout_n)
    var partial_sums_tensor = LayoutTensor[
        mut=True, dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](d_partial_sums, layout_num_blocks)
    var flags_tensor = LayoutTensor[
        mut=True, DType.int32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin
    ](d_flags, layout_num_blocks)

    # Launch kernel
    var kernel_start = perf_counter()
    ctx.enqueue_function[
        sum_scan_kernel[dtype, block_dim, coarse_factor],
        sum_scan_kernel[dtype, block_dim, coarse_factor],
    ](
        input_tensor,
        output_tensor,
        d_block_counter,
        partial_sums_tensor,
        flags_tensor,
        grid_dim=(num_blocks),
        block_dim=(block_dim),
    )
    ctx.synchronize()
    print("Pure GPU Kernel time:", perf_counter() - kernel_start)

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
    var n = 100_000_000

    comptime dtype = DType.int64

    # Allocate host memory
    var input = alloc[Scalar[dtype]](n)
    var output_gpu = alloc[Scalar[dtype]](n)
    var output_cpu = alloc[Scalar[dtype]](n)

    for i in range(n):
        input[i] = Scalar[dtype](1)
        output_gpu[i] = -1
        output_cpu[i] = -2

    # Compute on GPU
    var gpu_start = perf_counter()
    with DeviceContext() as ctx:
        sum_scan_gpu[dtype](input, output_gpu, n, ctx)
    print("GPU time", perf_counter() - gpu_start)

    # Compute on CPU
    cpu_start = perf_counter()
    sum_scan_cpu[dtype](input, output_cpu, n)
    print("CPU time", perf_counter() - cpu_start)

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
