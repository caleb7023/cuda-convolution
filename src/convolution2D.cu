
// author:caleb7023

#include <cuda_runtime.h>



__global__ void convolution2D_channel_cuda(float *input, float *kernel, float *output, int ic, int kc, int isx, int isy, int ksx, int ksy, int sx, int sy, int px, int py, int osx, int osy){
    
    /**
     * input: input image
     * kernel: kernel
     * output: output image
     * ic: input channels
     * kc: kernel channels
     * isx: input size x
     * isy: input size y
     * ksx: kernel size x
     * ksy: kernel size y
     * sx: stride x
     * sy: stride y
     * px: padding x
     * py: padding y
     * osx: output size x
     * osy: output size y
     */
    
    int kx_ = threadIdx.x + blockIdx.x * blockDim.x; // kernel x position
    int ky_ = threadIdx.y + blockIdx.y * blockDim.y; // kernel y position
    int ic_ = threadIdx.z + blockIdx.z * blockDim.z; // input channel
    int ox_ = blockIdx.w; // target x position in the output
    int oy_ = blockIdx.v; // target y position in the output
    int kc_ = blockIdx.u; // kernel/output channel

    int ix = ox_-px + kx_*sx; // input x position
    int iy = oy_-py + ky_*sy; // input y position

    if (kc_ < kc  && // kernel channel
        oy_ < osx && // target y position in the input
        ox_ < osy && // target x position in the input
        ic_ < ic  && // input channel
        ky_ < ksy && // kernel y position
        kx_ < ksx && // kernel x position
        0<=ix && ix<isx && // check if the input x position is valid
        0<=iy && iy<isy   )// check if the input y position is valid
    {
        output[kc_*osx*osy + ox_*osx + oy_] += input[ic_*isx*isy + ix*isx + iy] * kernel[kc_*ksx*ksy + kx_*ksx + ky_];
    }
}


__host__ void convolution2D_forward(float *input, float *output, float *kernels, int input_channels, int kernel_channels, int input_size_x, int input_size_y, int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, int padding_x, int padding_y){

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

    output_size_x = floorf((input_size_x - kernel_size_x + padding_x*2) / stride_x) + 1;
    output_size_y = floorf((input_size_y - kernel_size_y + padding_y*2) / stride_y) + 1;

    convolution2D_channel_cuda<<<kernel_channels*output_size_x*output_size_y, channels*output_size_x*output_size_y>>>(
        input_cuda,
        kernel_cuda,
        output_cuda,
        input_channels,
        kernel_channels,
        input_size_x,
        input_size_y,
        kernel_size_x,
        kernel_size_y,
        stride_x,
        stride_y,
        padding_x,
        padding_y,
        output_size_x,
        output_size_y
    );

    cudaMemcpy(output, output_cuda, input_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);
}