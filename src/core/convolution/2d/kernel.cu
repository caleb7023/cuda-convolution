
// author: caleb7023

#include <cuda_runtime.h>

#include <stdio.h>

/**
 * @brief This is the kernel function for the convolution operation.
 *
 * @param input: input image
 * @param kernel: kernel
 * @param output: output image
 * @param ic: input channels
 * @param kc: kernel channels
 * @param isx: input size x
 * @param isy: input size y
 * @param ksx: kernel size x
 * @param ksy: kernel size y
 * @param sx: stride x
 * @param sy: stride y
 * @param px: padding x
 * @param py: padding y
 * @param osx: output size x
 * @param osy: output size y
 */
__global__ void convolution2d_ch(
    float *input, float *kernel, float *output,
    int ic , int kc ,
    int isx, int isy,
    int ksx, int ksy,
    int osx, int osy,
    int sx , int sy ,
    int px , int py
)
{
    
    int ox_ = blockIdx.x; // target x position in the output
    int oy_ = blockIdx.y; // target y position in the output
    int kc_ = blockIdx.z; // kernel/output channel
    int kx_ = threadIdx.x; // kernel x position
    int ky_ = threadIdx.y; // kernel y position
    int ic_ = threadIdx.z; // input channel

    int ix = ox_-px + kx_*sx; // input x position
    int iy = oy_-py + ky_*sy; // input y position

    if (
        kc_ < kc  && // kernel channel
        oy_ < osy && // target y position in the input
        ox_ < osx && // target x position in the input
        ic_ < ic  && // input channel
        ky_ < ksy && // kernel y position
        kx_ < ksx && // kernel x position
        0<=ix && ix<isx && // check if the input x position is valid
        0<=iy && iy<isy    // check if the input y position is valid
    )
    {
        atomicAdd(
            &output[kc_*osx*osy + ox_*osy+ oy_],
             input[ic_*isx*isy + ix*isy+ iy] * kernel[kc_*ic*ksx*ksy + ic_*ksx*ksy + kx_*ksx + ky_]
        );
    }

}



/**
 *
 * @brief Performs a convolution operation on the input image.
 *
 * The input size does not include the channel size
 * The kernel size does not include the channel size too
 *
 *
 * @param input The input image.
 * @param output The output image.
 * @param kernels The convolution kernels.
 * @param input_channels The number of channels in the input image.
 * @param kernel_channels The number of channels in the kernels.
 * @param input_size_x The width of the input image.
 * @param input_size_y The height of the input image.
 * @param kernel_size_x The width of the kernels.
 * @param kernel_size_y The height of the kernels.
 * @param stride_x The stride in the x direction.
 * @param stride_y The stride in the y direction.
 * @param padding_x The padding in the x direction.
 * @param padding_y The padding in the y direction.
 *
 */
extern "C" void convolve2d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size_x  , int input_size_y   ,
    int kernel_size_x , int kernel_size_y  ,
    int output_size_x , int output_size_y  ,
    int stride_x      , int stride_y       ,
    int padding_x     , int padding_y
)
{

    float *input_cuda, *kernel_cuda, *output_cuda;
    
    cudaMalloc(& input_cuda,                    input_channels *  input_size_x *  input_size_y * sizeof(float));
    cudaMalloc(&kernel_cuda,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * sizeof(float));
    cudaMalloc(&output_cuda,  kernel_channels *                  output_size_x * output_size_y * sizeof(float));

    cudaMemcpy( input_cuda,   input,                    input_channels *  input_size_x *  input_size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_cuda, kernels,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * sizeof(float), cudaMemcpyHostToDevice);

    convolution2d_ch<<<dim3(output_size_x, output_size_y, kernel_channels), dim3(kernel_size_x, kernel_size_y, input_channels)>>>(
        input_cuda    , kernel_cuda    , output_cuda,
        input_channels, kernel_channels,
        input_size_x  , input_size_y   ,
        kernel_size_x , kernel_size_y,
        output_size_x , output_size_y,
        stride_x      , stride_y     ,
        padding_x     , padding_y
    );

    cudaMemcpy(output, output_cuda, kernel_channels * output_size_x * output_size_y * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);

}