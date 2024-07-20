
// author: caleb7023

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
    const float *input, const float *kernel, float *output,
    const unsigned int ic, const  unsigned int kc,
    const unsigned int is, const  unsigned int ks, const unsigned int os,
    const unsigned int s , const  unsigned int p
)
{
    
    const unsigned int  k_ = threadIdx.x; // kernel position
    const unsigned int ic_ = threadIdx.y; // input channel
    const unsigned int  o_ = blockIdx.x; // target position in the output
    const unsigned int kc_ = blockIdx.y; // kernel/output channel

    const int i = o_-p + k_*s; // input position

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
    const float *input, float *output, const float *kernels,
    const unsigned int input_channels, const unsigned int kernel_channels,
    const unsigned int input_size    , const unsigned int kernel_size,     const unsigned int output_size,
    const unsigned int stride        , const unsigned int padding
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
        stride        , padding
    );

    cudaMemcpy(output, output_cuda, kernel_channels * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);

}