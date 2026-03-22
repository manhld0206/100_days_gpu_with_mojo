from std.math import ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext


@always_inline
def cal_idx(row: Int, col: Int, h: Int) -> Int:
    return row * h + col


@always_inline
def _compute_pixel[
    blur_size: Int
](
    input_data: UnsafePointer[UInt8, ImmutExternalOrigin],
    w: UInt,
    h: UInt,
    row: Int,
    col: Int,
) -> UInt8:
    var value = 0
    var total_pixels = 0
    comptime for i in range(-blur_size, blur_size + 1):
        comptime for j in range(-blur_size, blur_size + 1):
            var cur_row = row + Int(i)
            var cur_col = col + Int(j)

            if (
                cur_row >= 0
                and cur_row < Int(w)
                and cur_col >= 0
                and cur_col < Int(h)
            ):
                value += Int(input_data[cal_idx(cur_row, cur_col, Int(h))])
                total_pixels += 1

    return UInt8(Float32(value) / Float32(total_pixels))


def blur_kernel[
    blur_size: Int
](
    input_data: UnsafePointer[UInt8, ImmutExternalOrigin],
    output_data: UnsafePointer[UInt8, MutExternalOrigin],
    w: UInt,
    h: UInt,
):
    row = Int(global_idx.x)
    col = Int(global_idx.y)

    idx = cal_idx(row, col, Int(h))

    if row < Int(w) and col < Int(h):
        output_data[idx] = _compute_pixel[blur_size](input_data, w, h, row, col)


def blur_gpu[
    blur_size: Int
](
    input_data: UnsafePointer[UInt8, ImmutExternalOrigin],
    output_data: UnsafePointer[UInt8, MutExternalOrigin],
    w: UInt,
    h: UInt,
    ctx: DeviceContext,
) raises:
    comptime dtype = DType.uint8

    var d_input_data = ctx.enqueue_create_buffer[dtype](Int(w * h))
    var d_output_data = ctx.enqueue_create_buffer[dtype](Int(w * h))

    ctx.enqueue_copy(d_input_data, input_data)

    comptime block_dim_x = 32
    comptime block_dim_y = 32
    var grid_dim_x = ceildiv(w, block_dim_x)
    var grid_dim_y = ceildiv(h, block_dim_y)

    # Launch kernel
    ctx.enqueue_function[blur_kernel[blur_size], blur_kernel[blur_size]](
        d_input_data,
        d_output_data,
        w,
        h,
        grid_dim=(grid_dim_x, grid_dim_y, 1),
        block_dim=(block_dim_x, block_dim_y, 1),
    )

    # Copy result from device to host
    ctx.enqueue_copy(output_data, d_output_data)

    # Synchronize to ensure completion
    ctx.synchronize()


def blur_cpu[
    blur_size: Int
](
    input_data: UnsafePointer[UInt8, ImmutExternalOrigin],
    output_data: UnsafePointer[UInt8, MutExternalOrigin],
    w: UInt,
    h: UInt,
):
    for i in range(w):
        for j in range(h):
            output_data[
                cal_idx(
                    Int(i),
                    Int(j),
                    Int(h),
                )
            ] = _compute_pixel[
                blur_size
            ](input_data, w, h, Int(i), Int(j))


def main() raises:
    var h: UInt = 1920
    var w: UInt = 1080

    num_pixels = Int(h * w)

    comptime blur_size = 3

    # Allocate host memory
    var input_data = alloc[UInt8](num_pixels)
    var output_data_gpu = alloc[UInt8](num_pixels)
    var output_data_cpu = alloc[UInt8](num_pixels)

    # Initialize input_data arrays
    for i in range(h * w):
        input_data[i] = UInt8(i % 256)

    # Compute on GPU
    with DeviceContext() as ctx:
        blur_gpu[blur_size](input_data, output_data_gpu, w, h, ctx)

    # Compute on CPU
    blur_cpu[blur_size](input_data, output_data_cpu, w, h)

    var is_correct = True

    for i in range(num_pixels):
        if output_data_cpu[i] != output_data_gpu[i]:
            is_correct = False
            print(input_data[i])
            print(output_data_cpu[i])

    if is_correct:
        print("Success")
    else:
        print("Error")

    # Clean up
    input_data.free()
    output_data_gpu.free()
    output_data_cpu.free()
