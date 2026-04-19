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
from std.atomic import Atomic


def sum_reduction_kernel[
    dtype: DType
](
    input: LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
):
    var i = thread_idx.x
    var stride = block_dim.x
    while stride >= 1:
        if thread_idx.x < stride:
            input[i] += input[i + stride]

        stride /= 2
        barrier()

    if thread_idx.x == 0:
        output[0] = input[0][0]


def sum_reduction_gpu[
    dtype: DType
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    # comptime block_dim = 32
    var block_dim = n/2

    var layout = RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
        Index(n)
    )

    var d_input = ctx.enqueue_create_buffer[dtype](n)
    var d_output = ctx.enqueue_create_buffer[dtype](1)

    ctx.enqueue_copy(d_input, input)

    var input_tensor = LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE)](
        d_input, layout
    )

    # var grid_dim = ceildiv(n, block_dim)
    var grid_dim = 1

    # Launch kernel
    ctx.enqueue_function[
        sum_reduction_kernel[dtype],
        sum_reduction_kernel[dtype],
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


def sum_reduction_cpu[
    dtype: DType,
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    n: Int,
) raises:
    var sum = Scalar[dtype](0)
    for i in range(n):
        sum += input[i]
    output[0] = sum


def main() raises:
    var n = 1024

    comptime dtype = DType.int32

    # Allocate host memory
    var input = alloc[Scalar[dtype]](n)
    var output_gpu = alloc[Scalar[dtype]](1)
    var output_cpu = alloc[Scalar[dtype]](1)

    # Initialize image arrays
    for i in range(n):
        input[i] = 2

    # Compute on GPU
    with DeviceContext() as ctx:
        sum_reduction_gpu[dtype](input, output_gpu, n, ctx)

    # Compute on CPU
    sum_reduction_cpu[dtype](input, output_cpu, n)

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
