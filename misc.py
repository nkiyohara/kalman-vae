def compute_conv2d_output_size(input_size, kernel_size, stride, padding):
    h, w = input_size
    h_out = (h - kernel_size + 2 * padding) // stride + 1
    w_out = (w - kernel_size + 2 * padding) // stride + 1

    return h_out, w_out
