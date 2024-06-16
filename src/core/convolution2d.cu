
// author: caleb7023

#include <cuda_runtime.h>




__global__ void convolution2d_channel(float *input, float *kernel, float *output, int ic, int kc, int isx, int isy, int ksx, int ksy, int sx, int sy, int px, int py, int osx, int osy){
    
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
    
    int kx_ = threadIdx.x; // kernel x position
    int ky_ = threadIdx.y; // kernel y position
    int ic_ = threadIdx.z; // input channel
    int ox_ = blockIdx.x; // target x position in the output
    int oy_ = blockIdx.y; // target y position in the output
    int kc_ = blockIdx.z; // kernel/output channel

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



/**
 *
 * @brief Performs a convolution operation on the input image.
 *
 * The input size does not include the channel size
 * The kernel size does not include the channel size too
 * input, output and kernels are 1D arrays
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
void convolution2d(float *input, float *output, float *kernels, int input_channels, int kernel_channels, int input_size_x, int input_size_y, int kernel_size_x, int kernel_size_y, int stride_x, int stride_y, int padding_x, int padding_y){

    float *input_cuda, *kernel_cuda, *output_cuda;

    int output_size_x = static_cast<int>((input_size_x - kernel_size_x + padding_x*2) / (stride_x+1)) + 1;
    int output_size_y = static_cast<int>((input_size_y - kernel_size_y + padding_y*2) / (stride_y+1)) + 1;

    cudaMalloc(& input_cuda,                    input_channels *  input_size_x *  input_size_y * sizeof(float));
    cudaMalloc(&kernel_cuda,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * sizeof(float));
    cudaMalloc(&output_cuda,  kernel_channels *                  output_size_x * output_size_y * sizeof(float));

    cudaMemcpy( input_cuda,   input,                    input_channels *  input_size_x *  input_size_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_cuda, kernels,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * sizeof(float), cudaMemcpyHostToDevice);

    convolution2d_channel<<<dim3(kernel_channels, output_size_x, output_size_y), dim3(kernel_channels, kernel_size_x, kernel_size_y)>>>(
        input_cuda,
        kernel_cuda,
        output_cuda,
        input_channels,
        kernel_channels,
        input_size_x,
        input_size_y,
        kernel_size_x,
        kernel_size_y,
        stride_x+1,
        stride_y+1,
        padding_x,
        padding_y,
        output_size_x,
        output_size_y
    );

    cudaMemcpy(output, output_cuda, kernel_channels * output_size_x * output_size_y * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);
}