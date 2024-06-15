
// author:caleb7023

#include <cuda_runtime.h>




extern "C" {
    void convolution_forward(float *input, float *output, float *kernels, int *input_dim_size, int *kernel_dim_size, int channels, int input_size, int kernel_size);
}


__global__ void convolution_channel_cuda(float *input, float *kernel, float *output, , int *input_dim_size, int *kernel_dim_size, int input_size, int kernel_size){

    int channel =  blockIdx.x;
    int index   = threadIdx.x;
    int input_dim_index = 

    output[index] += 0;
}


__host__ void convolution_forward(float *input, float *output, float *kernels, int *input_dim_size, int *kernel_dim_size, int channels, int input_size, int kernel_size){

    /**
     * The input size does not include the channel size
     * The kernel size does not include the channel size too
     * input, output and kernels are 1D arrays
     */

    float *input_cuda, *kernel_cuda, *output_cuda;

    cudaMalloc(& input_cuda, channels * input_size  * sizeof(float));
    cudaMalloc(&kernel_cuda, channels * kernel_size * sizeof(float));
    cudaMalloc(&output_cuda, channels * input_size  * sizeof(float));

    cudaMemcpy( input_cuda,   input, input_size  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_cuda, kernels, kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    convolution_channel_cuda<<<channels, input_size*kernel_size>>>(input_cuda, kernel_cuda, output_cuda, input_dim_size, kernel_dim_size, input_size, kernel_size);

    cudaMemcpy(output, output_cuda, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);
}