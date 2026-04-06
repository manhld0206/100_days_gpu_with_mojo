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
comptime STENCIL_WIDTH = 7


def stencil_3d_kernel[
    dtype: DType,
    block_dim: Int,
    stencil_width: Int,
](
    input: LayoutTensor[dtype, Layout3D, ImmutAnyOrigin],
    output: LayoutTensor[dtype, Layout3D, MutAnyOrigin],
    # (center, along z coefficients, along y coeeficients, along x coeeficients)
    coefficients: InlineArray[Scalar[dtype], stencil_width],
):
    assert (stencil_width - 1) % 6 == 0
    comptime stencil_radius: Int = (stencil_width - 1) / 6
    comptime out_dim = block_dim - 2 * stencil_radius

    out_shape = output.get_shape()
    n = out_shape[0]
    assert n == out_shape[1] and n == out_shape[2]

    in_shape = input.get_shape()
    assert n == in_shape[0] and n == in_shape[1] and n == in_shape[2]

    var in_s = LayoutTensor[
        dtype,
        Layout.row_major(block_dim, block_dim, block_dim),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    var col = block_idx.x * out_dim + thread_idx.x - stencil_radius
    var row = block_idx.y * out_dim + thread_idx.y - stencil_radius
    var dep = block_idx.z * out_dim + thread_idx.z - stencil_radius

    var tx = thread_idx.x
    var ty = thread_idx.y
    var tz = thread_idx.z

    if col < n and col >= 0 and row < n and row >= 0 and dep < n and dep >= 0:
        in_s[tz, ty, tx] = input[dep, row, col]

    barrier()

    if (
        stencil_radius <= col
        and col < n - stencil_radius
        and stencil_radius <= row
        and row < n - stencil_radius
        and stencil_radius <= dep
        and dep < n - stencil_radius
    ):
        # Only compute interior elements
        if (
            tx < block_dim - stencil_radius
            and tx >= stencil_radius
            and ty < block_dim - stencil_radius
            and ty >= stencil_radius
            and tz < block_dim - stencil_radius
            and tz >= stencil_radius
        ):
            var value = coefficients[0] * in_s[tz, ty, tx]
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + i]
                    * in_s[tz - (stencil_radius - i), ty, tx]
                )
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + stencil_radius + i] * in_s[tz + i, ty, tx]
                )
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + 2 * stencil_radius + i]
                    * in_s[tz, ty - (stencil_radius - i), tx]
                )
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + 3 * stencil_radius + i]
                    * in_s[tz, ty + i, tx]
                )
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + 4 * stencil_radius + i]
                    * in_s[tz, ty, tx - (stencil_radius - i)]
                )
            comptime for i in range(0, stencil_radius):
                value += (
                    coefficients[1 + 5 * stencil_radius + i]
                    * in_s[tz, ty, tx + i]
                )

            output[dep, row, col] = value


def stencil_3d_gpu[
    dtype: DType,
    stencil_width: Int,
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    coefficients: InlineArray[Scalar[dtype], stencil_width],
    n: Int,
    ctx: DeviceContext,
) raises:
    assert (stencil_width - 1) % 6 == 0
    comptime stencil_radius: Int = (stencil_width - 1) / 6
    comptime block_dim = 8
    comptime out_dim = block_dim - 2 * stencil_radius

    var tensor_3d = RuntimeLayout[Layout3D].row_major(Index(n, n, n))

    var d_input = ctx.enqueue_create_buffer[dtype](n**3)
    var d_output = ctx.enqueue_create_buffer[dtype](n**3)

    ctx.enqueue_copy(d_input, input)
    ctx.enqueue_copy(d_output, output)

    var input_tensor = LayoutTensor[dtype, Layout3D](d_input, tensor_3d)
    var output_tensor = LayoutTensor[dtype, Layout3D, MutAnyOrigin](
        d_output, tensor_3d
    )

    var grid_dim = ceildiv(n, out_dim)

    # Launch kernel
    ctx.enqueue_function[
        stencil_3d_kernel[dtype, block_dim, stencil_width],
        stencil_3d_kernel[dtype, block_dim, stencil_width],
    ](
        input_tensor,
        output_tensor,
        coefficients,
        grid_dim=(grid_dim, grid_dim, grid_dim),
        block_dim=(block_dim, block_dim, block_dim),
    )

    # Copy result from device to host
    ctx.enqueue_copy(output, d_output)

    # Synchronize to ensure completion
    ctx.synchronize()


def stencil_3d_cpu[
    dtype: DType,
    stencil_width: Int,
](
    input: UnsafePointer[Scalar[dtype], ImmutExternalOrigin],
    output: UnsafePointer[Scalar[dtype], MutExternalOrigin],
    coefficients: InlineArray[Scalar[dtype], stencil_width],
    n: Int,
):
    assert (stencil_width - 1) % 6 == 0
    comptime stencil_radius: Int = (stencil_width - 1) / 6

    @parameter
    def _cal_index(z: Int, y: Int, x: Int) -> Int:
        return z * n * n + y * n + x

    for tz, ty, tx in product(
        range(stencil_radius, n - stencil_radius),
        range(stencil_radius, n - stencil_radius),
        range(stencil_radius, n - stencil_radius),
    ):
        var out = _cal_index(tz, ty, tx)
        output[out] = Scalar[dtype](0)
        output[out] = coefficients[0] * input[_cal_index(tz, ty, tx)]
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + i]
                * input[_cal_index(tz - (stencil_radius - i), ty, tx)]
            )
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + stencil_radius + i]
                * input[_cal_index(tz + i, ty, tx)]
            )
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + 2 * stencil_radius + i]
                * input[_cal_index(tz, ty - (stencil_radius - i), tx)]
            )
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + 3 * stencil_radius + i]
                * input[_cal_index(tz, ty + i, tx)]
            )
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + 4 * stencil_radius + i]
                * input[_cal_index(tz, ty, tx - (stencil_radius - i))]
            )
        comptime for i in range(0, stencil_radius):
            output[out] += (
                coefficients[1 + 5 * stencil_radius + i]
                * input[_cal_index(tz, ty, tx + i)]
            )


def main() raises:
    var n: Int = 100

    comptime stencil_radius = 1
    comptime stencil_width = stencil_radius * 6 + 2

    comptime dtype = DType.float32

    # Allocate host memory
    var input = alloc[Scalar[dtype]](n**3)
    var output_gpu = alloc[Scalar[dtype]](n**3)
    var output_cpu = alloc[Scalar[dtype]](n**3)

    # Initialize input_data arrays
    for i in range(n**3):
        input[i] = Scalar[dtype]((i + 7) % 100)

    var coefficients = InlineArray[Scalar[dtype], stencil_width](fill=0)
    for i in range(stencil_width):
        coefficients[i] = Scalar[dtype](i % 4)

    # Compute on GPU
    with DeviceContext() as ctx:
        stencil_3d_gpu[dtype, stencil_width](
            input, output_gpu, coefficients, n, ctx
        )

    # Compute on CPU
    stencil_3d_cpu[dtype, stencil_width](input, output_cpu, coefficients, n)

    var is_correct = True

    for y, x, z in product(
        range(stencil_radius, n - stencil_radius),
        range(stencil_radius, n - stencil_radius),
        range(stencil_radius, n - stencil_radius),
    ):
        var i = z * n * n + y * n + x
        if output_cpu[i] != output_gpu[i]:
            is_correct = False
            print("x, y, z", x, y, z)
            print("cpu", output_cpu[i])
            print("gpu", output_gpu[i])

    if is_correct:
        print("Success")
    else:
        print("Error")

    # Clean up
    input.free()
    output_cpu.free()
    output_gpu.free()
