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


def sum_scan_kernel[
    dtype: DType,
    block_dim: Int,
](
    input: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin],
    output: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin],
):
    tid = thread_idx.x
    n = input.dim(0)

    input_s = LayoutTensor[
        dtype,
        Layout.row_major(block_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    if tid >= n:
        return

    input_s[tid] = input[tid]
    barrier()

    # Phase 1: Up-Sweep (Reduction Tree)
    var stride = 1
    while stride < n:
        # This thread mapping avoids thread divergence
        index = (tid + 1) * stride * 2 - 1

        if index < n:
            input_s[index] += input_s[index - stride]

        stride *= 2
        barrier()

    # Phase 2: Down-Sweep (Distribution Tree)
    stride = stride // 4

    while stride > 0:
        # This thread mapping avoids thread divergence
        var index = (tid + 1) * stride * 2 - 1 + stride

        if index < n:
            input_s[index] += input_s[index - stride]

        stride = stride // 2
        barrier()

    output[tid] = input_s[tid]


def sum_scan_gpu[
    dtype: DType, //
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime block_dim = 1024
    assert n <= 1024

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
        sum_scan_kernel[dtype, block_dim],
        sum_scan_kernel[dtype, block_dim],
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
    dtype: DType, //
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
) raises:
    output[0] = input[0]
    for i in range(1, n):
        output[i] = output[i - 1] + input[i]


def main() raises:
    var n = 1000

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
    with DeviceContext() as ctx:
        sum_scan_gpu(input, output_gpu, n, ctx)

    # Compute on CPU
    sum_scan_cpu(input, output_cpu, n)

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
