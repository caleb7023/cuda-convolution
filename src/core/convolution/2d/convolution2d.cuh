
#ifndef CONVOLUTION2D_CUH
#define CONVOLUTION2D_CUH

void convolution2d(float *input, float *output, float *kernels, int input_channels,
                   int kernel_channels, int input_size_x,int input_size_y, int kernel_size_x,
                   int kernel_size_y, int stride_x, int stride_y, int padding_x, int padding_y);

#endif