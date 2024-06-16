
#ifndef CONVOLUTION1D_CUH
#define CONVOLUTION1D_CUH

void convolution1d(float *input, float *output, float *kernels, int input_channels,
                   int kernel_channels, int input_size, int kernel_size,
                   int stride, int padding);

#endif