
#ifndef CONVOLUTION3D_CUH
#define CONVOLUTION3D_CUH

void convolution3d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size_x  , int input_size_y   , int input_size_z ,
    int kernel_size_x , int kernel_size_y  , int kernel_size_z,
    int stride_x      , int stride_y       , int stride_z     ,
    int padding_x     , int padding_y      , int padding_z
);

#endif