// author: caleb7023

#ifndef CONVOLUTION1D_CUH
#define CONVOLUTION1D_CUH

#include <cuda_runtime.h>

/**
 * @brief This is the kernel function for the convolution operation.
 *
 * @param input: input image
 * @param kernel: kernel
 * @param output: output image
 * @param ic: input channels
 * @param kc: kernel channels
 * @param is: input size
 * @param ks: kernel size
 * @param s: stride
 * @param p: padding
 * @param os: output size
 */
__global__ void convolution1d_ch(
    float *input, float *kernel, float *output,
    int ic, int kc,
    int is, int ks, int os,
    int s , int p
)
{
    
    int  k_ = threadIdx.x; // kernel position
    int ic_ = threadIdx.y; // input channel
    int  o_ = blockIdx.x; // target position in the output
    int kc_ = blockIdx.y; // kernel/output channel

    int i = o_-p + k_*s; // input position

    if (kc_ < kc && // kernel channel
         o_ < os && // target position in the input
        ic_ < ic && // input channel
         k_ < ks && // kernel position
        0<=i && i<is)// check if the input position is valid
    {
        atomicAdd(&output[kc_*os + o_], input[ic_*is + i] * kernel[kc_*ks + k_]);
    }

}



/**
 *
 * @brief Performs a convolution operation on the input 1D array.
 *
 * The input size does not include the channel size
 * The kernel size does not include the channel size too
 * input, output and kernels are 1D arrays
 *
 *
 * @param input The input array.
 * @param output The output array.
 * @param kernels The convolution kernels.
 * @param input_channels The number of channels in the input array.
 * @param kernel_channels The number of channels in the kernels.
 * @param input_size The length of the input.
 * @param kernel_size The length of the kernel.
 * @param stride The stride.
 * @param padding The padding.
 *
 */
extern "C" void convolve1d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size    , int kernel_size, int output_size,
    int stride        , int padding
)
{

    float *input_cuda, *kernel_cuda, *output_cuda;

    cudaMalloc(& input_cuda,                    input_channels *  input_size * sizeof(float));
    cudaMalloc(&kernel_cuda,  kernel_channels * input_channels * kernel_size * sizeof(float));
    cudaMalloc(&output_cuda,  kernel_channels *                  output_size * sizeof(float));

    cudaMemcpy( input_cuda,   input,                    input_channels *  input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_cuda, kernels,  kernel_channels * input_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    convolution1d_ch<<<dim3(output_size, kernel_channels), dim3(kernel_size, input_channels)>>>(
        input_cuda    , kernel_cuda    , output_cuda,
        input_channels, kernel_channels,
        input_size    , kernel_size    , output_size,
        stride+1      , padding
    );

    cudaMemcpy(output, output_cuda, kernel_channels * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);

}

#endif