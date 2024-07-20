// author: caleb7023

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


extern "C" void convolve1d(
    const float *input, float *output, const float *kernels,
    unsigned int input_channels, unsigned int kernel_channels,
    unsigned int input_size    , unsigned int kernel_size, unsigned int output_size,
    unsigned int stride        , unsigned int padding
);

extern "C" void convolve2d(
    const float *input, float *output, const float *kernels,
    unsigned int input_channels, unsigned int kernel_channels,
    unsigned int input_size_x  , unsigned int input_size_y   ,
    unsigned int kernel_size_x , unsigned int kernel_size_y  ,
    unsigned int output_size_x , unsigned int output_size_y  ,
    unsigned int stride_x      , unsigned int stride_y       ,
    unsigned int padding_x     , unsigned int padding_y
);

extern "C" void convolve3d(
    const float *input, float *output, const float *kernels,
    unsigned int input_channels, unsigned int kernel_channels,
    unsigned int input_size_x  , unsigned int input_size_y   , unsigned int input_size_z ,
    unsigned int kernel_size_x , unsigned int kernel_size_y  , unsigned int kernel_size_z,
    unsigned int output_size_x , unsigned int output_size_y  , unsigned int output_size_z,
    unsigned int stride_x      , unsigned int stride_y       , unsigned int stride_z     ,
    unsigned int padding_x     , unsigned int padding_y      , unsigned int padding_z
);

namespace py = pybind11;

void convolve1d_py(
    const py::array_t<float>& input_arr,
    const py::array_t<float>& kernel_arr,
    const py::array_t<float>& output_arr,
    const unsigned int padding,
    const unsigned int stride
)
{
    
    const py::buffer_info input_buf =  input_arr.request();
    const py::buffer_info kernel_buf = kernel_arr.request();
    const py::buffer_info output_buf = output_arr.request();

    const float *input   = static_cast<float*>( input_buf.ptr);
    const float *kernels = static_cast<float*>(kernel_buf.ptr);
    auto         output  = static_cast<float*>(output_buf.ptr);

    const unsigned int input_channels  =  input_buf.shape[0];
    const unsigned int kernel_channels = kernel_buf.shape[0];
    const unsigned int input_size      =  input_buf.shape[1];
    const unsigned int kernel_size     = kernel_buf.shape[2];
    const unsigned int output_size     = output_buf.shape[1];

    convolve1d(input, output, kernels, input_channels, kernel_channels, input_size, kernel_size, output_size, stride, padding);
}

void convolve2d_py(
    const py::array_t<float>& input_arr,
    const py::array_t<float>& kernel_arr,
    const py::array_t<float>& output_arr,
    const unsigned int padding_x, const unsigned int padding_y,
    const unsigned int stride_x , const unsigned int stride_y
)
{
    
    const py::buffer_info  input_buf =  input_arr.request();
    const py::buffer_info kernel_buf = kernel_arr.request();
    const py::buffer_info output_buf = output_arr.request();

    const float *input   = static_cast<float*>( input_buf.ptr);
    const float *kernels = static_cast<float*>(kernel_buf.ptr);
    auto         output  = static_cast<float*>(output_buf.ptr);

    const unsigned int input_channels  =  input_buf.shape[0];
    const unsigned int kernel_channels = kernel_buf.shape[0];
    const unsigned int input_size_x    =  input_buf.shape[1];
    const unsigned int kernel_size_x   = kernel_buf.shape[2];
    const unsigned int output_size_x   = output_buf.shape[1];
    const unsigned int input_size_y    =  input_buf.shape[2];
    const unsigned int kernel_size_y   = kernel_buf.shape[3];
    const unsigned int output_size_y   = output_buf.shape[2];

    convolve2d(
        input, output, kernels, 
        input_channels, kernel_channels,
        input_size_x  , input_size_y   ,
        kernel_size_x , kernel_size_y  ,
        output_size_x , output_size_y  ,
        stride_x      , stride_y       , 
        padding_x     , padding_y
    );
}

void convolve3d_py(
    const py::array_t<float>& input_arr,
    const py::array_t<float>& kernel_arr,
    const py::array_t<float>& output_arr,
    const unsigned int padding_x, const unsigned int padding_y,
    const unsigned int padding_z, const unsigned int stride_x,
    const unsigned int stride_y,  const unsigned int stride_z
)
{
    
    const py::buffer_info  input_buf =  input_arr.request();
    const py::buffer_info kernel_buf = kernel_arr.request();
    const py::buffer_info output_buf = output_arr.request();

    const float *input   = static_cast<float*>( input_buf.ptr);
    const float *kernels = static_cast<float*>(kernel_buf.ptr);
    auto         output  = static_cast<float*>(output_buf.ptr);

    const unsigned int input_channels  =  input_buf.shape[0];
    const unsigned int kernel_channels = kernel_buf.shape[0];
    const unsigned int input_size_x    =  input_buf.shape[1];
    const unsigned int kernel_size_x   = kernel_buf.shape[2];
    const unsigned int output_size_x   = output_buf.shape[1];
    const unsigned int input_size_y    =  input_buf.shape[2];
    const unsigned int kernel_size_y   = kernel_buf.shape[3];
    const unsigned int output_size_y   = output_buf.shape[2];
    const unsigned int input_size_z    =  input_buf.shape[3];
    const unsigned int kernel_size_z   = kernel_buf.shape[4];
    const unsigned int output_size_z   = output_buf.shape[3];

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