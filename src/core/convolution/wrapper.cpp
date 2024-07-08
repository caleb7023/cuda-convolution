// author: caleb7023

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdio.h>

extern "C" void convolve1d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size    , int kernel_size, int output_size,
    int stride        , int padding
);

extern "C" void convolve2d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size_x  , int input_size_y   ,
    int kernel_size_x , int kernel_size_y  ,
    int output_size_x , int output_size_y  ,
    int stride_x      , int stride_y       ,
    int padding_x     , int padding_y
);

extern "C" void convolve3d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size_x  , int input_size_y   , int input_size_z ,
    int kernel_size_x , int kernel_size_y  , int kernel_size_z,
    int output_size_x , int output_size_y  , int output_size_z,
    int stride_x      , int stride_y       , int stride_z     ,
    int padding_x     , int padding_y      , int padding_z
);

namespace py = pybind11;

void convolve1d_py(py::array_t<float> input_arr, py::array_t<float> kernel_arr, py::array_t<float> output_arr, int padding, int stride) {
    
    py::buffer_info  input_buf =  input_arr.request();
    py::buffer_info kernel_buf = kernel_arr.request();
    py::buffer_info output_buf = output_arr.request();

    float *input   = static_cast<float *>( input_buf.ptr);
    float *kernels = static_cast<float *>(kernel_buf.ptr);
    float *output  = static_cast<float *>(output_buf.ptr);

    int input_channels  =  input_buf.shape[0];
    int kernel_channels = kernel_buf.shape[0];
    int input_size      =  input_buf.shape[1];
    int kernel_size     = kernel_buf.shape[2];
    int output_size     = output_buf.shape[1];

    convolve1d(input, output, kernels, input_channels, kernel_channels, input_size, kernel_size, output_size, stride, padding);
}

void convolve2d_py(py::array_t<float> input_arr, py::array_t<float> kernel_arr, py::array_t<float> output_arr, int padding_x, int padding_y, int stride_x, int stride_y) {
    
    py::buffer_info  input_buf =  input_arr.request();
    py::buffer_info kernel_buf = kernel_arr.request();
    py::buffer_info output_buf = output_arr.request();

    float *input   = static_cast<float *>( input_buf.ptr);
    float *kernels = static_cast<float *>(kernel_buf.ptr);
    float *output  = static_cast<float *>(output_buf.ptr);

    int input_channels  =  input_buf.shape[0];
    int kernel_channels = kernel_buf.shape[0];
    int input_size_x    =  input_buf.shape[1];
    int kernel_size_x   = kernel_buf.shape[2];
    int output_size_x   = output_buf.shape[1];
    int input_size_y    =  input_buf.shape[2];
    int kernel_size_y   = kernel_buf.shape[3];
    int output_size_y   = output_buf.shape[2];

    convolve2d(
        input, output, kernels, 
        input_channels, kernel_channels,
        input_size_x  , input_size_y   ,
        kernel_size_x , kernel_size_y  ,
        output_size_x , output_size_y  ,
        stride_x      , stride_y       , 
        padding_y     , padding_y
    );
}

void convolve3d_py(py::array_t<float> input_arr, py::array_t<float> kernel_arr, py::array_t<float> output_arr, int padding_x, int padding_y, int padding_z, int stride_x, int stride_y, int stride_z) {
    
    py::buffer_info  input_buf =  input_arr.request();
    py::buffer_info kernel_buf = kernel_arr.request();
    py::buffer_info output_buf = output_arr.request();

    float *input   = static_cast<float *>( input_buf.ptr);
    float *kernels = static_cast<float *>(kernel_buf.ptr);
    float *output  = static_cast<float *>(output_buf.ptr);

    int input_channels  =  input_buf.shape[0];
    int kernel_channels = kernel_buf.shape[0];
    int input_size_x    =  input_buf.shape[1];
    int kernel_size_x   = kernel_buf.shape[2];
    int output_size_x   = output_buf.shape[1];
    int input_size_y    =  input_buf.shape[2];
    int kernel_size_y   = kernel_buf.shape[3];
    int output_size_y   = output_buf.shape[2];
    int input_size_z    =  input_buf.shape[3];
    int kernel_size_z   = kernel_buf.shape[4];
    int output_size_z   = output_buf.shape[3];

    convolve3d(
        input, output, kernels, 
        input_channels, kernel_channels,
        input_size_x  , input_size_y   , input_size_z ,
        kernel_size_x , kernel_size_y  , kernel_size_z, 
        output_size_x , output_size_y  , output_size_z,
        stride_x      , stride_y       , stride_z     , 
        padding_x     , padding_y      , padding_z
    );
}

PYBIND11_MODULE(convolutions, m) {
    m.def("convolve1d_py", &convolve1d_py, "A function that performs 1D convolution");
    m.def("convolve2d_py", &convolve2d_py, "A function that performs 2D convolution");
    m.def("convolve3d_py", &convolve3d_py, "A function that performs 3D convolution");
}