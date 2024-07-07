
// author: caleb7023

#include <cuda_runtime.h>

// Its 3D!

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
 * @param isz: input size z
 * @param ksx: kernel size x
 * @param ksy: kernel size y
 * @param ksz: kernel size z
 * @param osx: output size x
 * @param osy: output size y
 * @param osz: output size z
 * @param sx: stride x
 * @param sy: stride y
 * @param sz: stride z
 * @param px: padding x
 * @param py: padding y
 * @param pz: padding z
 */
__global__ void convolution3d_channel(
    float *input, float *kernel, float *output,
    int ic , int kc ,
    int isx, int isy, int isz,
    int ksx, int ksy, int ksz,
    int osx, int osy, int osz,
    int sx , int sy , int sz ,
    int px , int py , int pz
)
{
    
    int kx_ = threadIdx.x/ksx; // kernel x position
    int ky_ = threadIdx.x%ksx; // kernel y position
    int kz_ = threadIdx.y; // kernel z position
    int ic_ = threadIdx.z; // input channel
    int ox_ = blockIdx.x/osx; // target x position in the output
    int oy_ = blockIdx.x%osx; // target y position in the output
    int oz_ = blockIdx.y; // target z position in the output
    int kc_ = blockIdx.z; // kernel/output channel

    int ix = ox_-px + kx_*sx; // input x position
    int iy = oy_-py + ky_*sy; // input y position
    int iz = oz_-pz + kz_*sz; // input z position

    if (kc_ < kc  && // kernel channel
        oy_ < osy && // target y position in the input
        ox_ < osx && // target x position in the input
        oz_ < osz && // target z position in the input
        ic_ < ic  && // input channel
        ky_ < ksy && // kernel y position
        kx_ < ksx && // kernel x position
        kz_ < ksz && // kernel z position
        0<=ix && ix<isx && // check if the input x position is valid
        0<=iy && iy<isy && // check if the input y position is valid
        0<=iz && iz<isz)   // check if the input z position is valid
    {
        atomicAdd(
            &output[kc_*osx*osy*osz + ox_*osx*osy + oy_*osy + oz_],
              input[ic_*isx*isy*isz + ix *isx*isy + iy *isy + iz ] * kernel[kc_*ic*ksx*ksy*ksz + ic_*ksx*ksy*ksz + kx_*ksx*ksy + ky_*ksy + kz_]
        );
    }

}



/**
 *
 * @brief Performs a convolution operation on the input object.
 *
 * The input size does not include the channel size
 * The kernel size does not include the channel size too
 *
 *
 * @param input The input object.
 * @param output The output object.
 * @param kernels The convolution kernels.
 * @param input_channels The number of channels in the input object.
 * @param kernel_channels The number of channels in the kernels.
 * @param input_size_x The width of the input object.
 * @param input_size_y The height of the input object.
 * @param input_size_z The depth of the input object.
 * @param kernel_size_x The width of the kernels.
 * @param kernel_size_y The height of the kernels.
 * @param kernel_size_z The depth of the kernels.
 * @param stride_x The stride in the x direction.
 * @param stride_y The stride in the y direction.
 * @param stride_z The stride in the z direction.
 * @param padding_x The padding in the x direction.
 * @param padding_y The padding in the y direction.
 * @param padding_z The padding in the z direction.
 *
 */
extern "C" void convolve3d(
    float *input, float *output, float *kernels,
    int input_channels, int kernel_channels,
    int input_size_x  , int input_size_y   , int input_size_z ,
    int kernel_size_x , int kernel_size_y  , int kernel_size_z,
    int output_size_x , int output_size_y  , int output_size_z,
    int stride_x      , int stride_y       , int stride_z     ,
    int padding_x     , int padding_y      , int padding_z
)
{

    float *input_cuda, *kernel_cuda, *output_cuda;

    cudaMalloc(& input_cuda,                    input_channels *  input_size_x *  input_size_y *  input_size_y * sizeof(float));
    cudaMalloc(&kernel_cuda,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * kernel_size_y * sizeof(float));
    cudaMalloc(&output_cuda,  kernel_channels *                  output_size_x * output_size_y * output_size_z * sizeof(float));

    cudaMemcpy( input_cuda,   input,                    input_channels *  input_size_x *  input_size_y *  input_size_z * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(kernel_cuda, kernels,  kernel_channels * input_channels * kernel_size_x * kernel_size_y * kernel_size_z * sizeof(float), cudaMemcpyHostToDevice);

    convolution3d_channel<<<dim3(output_size_x*output_size_y, output_size_z, kernel_channels), dim3(kernel_size_x*kernel_size_y, kernel_size_z, input_channels)>>>(
        input_cuda    , kernel_cuda    , output_cuda,
        input_channels, kernel_channels,
        input_size_x  , input_size_y   , input_size_z ,
        kernel_size_x , kernel_size_y  , kernel_size_z,
        output_size_x , output_size_y  , output_size_z,
        stride_x+1    , stride_y+1     , stride_z+1   ,
        padding_x     , padding_y      , padding_z
    );

    cudaMemcpy(output, output_cuda, kernel_channels * output_size_x * output_size_y * output_size_z * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree( input_cuda);
    cudaFree(output_cuda);
    cudaFree(kernel_cuda);

}