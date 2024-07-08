#!/usr/bin/env python

# author: caleb7023

import numpy as np
from src.core.convolution import convolutions as __conv

def convolve1d(input:np.ndarray, kernel:np.ndarray, padding:int=0, stride:int=0) -> np.ndarray:
    
    """
    # convolve1d

    This function convolves a 1D input with channel with a 1D kernel with input and kernel channels.

    The output will be a 1D array with channels.

    # Parameters
    ```plaintext
    >Required
    [np.ndarray]input  : The input array that is going to be convolved. Must be a 2D array
    [np.ndarray]kernel : The kernel(filter) array that is going to be used to convolve the input. Must be a 3D array
    >Not Required
    [int] stride  : The stride of the convolution layer. If not provided, 0 will be used.
    [int] padding : The padding of the convolution layer. If not provided, 0 will be used. -1 to keep the input size. -2 to fully use the kernel.
    ```
    """

    # Check for errors
    if input.ndim != 2:
        raise ValueError("The input must be a 2D array")
    if kernel.ndim != 3:
        raise ValueError("The kernel must be a 3D array")
    if input.shape[0] != kernel.shape[1]:
        raise ValueError("The input's channel and kernel's input channel must have the same size")
    if padding < -2:
        raise ValueError("The value of the padding is invalid. (%d is the input and it must be greater than -2)" % padding)
    if stride < 0:
        raise ValueError("stride(%d) must not be negative" % padding)
    
    # Set the padding if -1 or -2 is provided
    if padding == -1:
        padding = kernel.shape[2] // 2
    if padding == -2:
        padding = kernel.shape[2] - 1
    
    # Create the output array for storing the output
    output_size = (input.shape[1] - kernel.shape[2] + 2 * padding) // (stride + 1)
    output = np.zeros((kernel.shape[0], output_size), astype=np.float32)

    # Calulate the output and store it in the output array
    __conv.convolve1d_py(input.astype(np.float32), kernel.astype(np.float32), output, padding, stride)

    # Return the output
    return output



def convolve2d(input:np.ndarray, kernel:np.ndarray, padding:int|tuple[int]=0, stride:int|tuple[int]=1) -> np.ndarray:

    """
    # convolve2d

    This function convolves a 2D input with channel with a 2D kernel with input and kernel channels.

    The output will be a 2D array with channels.

    # Parameters
    ```plaintext
    >Required
    [np.ndarray]input  : The input array that is going to be convolved. Must be a 3D array
    [np.ndarray]kernel : The kernel(filter) array that is going to be used to convolve the input. Must be a 4D array
    >Not Required
    [int|tuple[int]] stride  : The stride of the convolution layer. If not provided, 0 will be used.
    [int|tuple[int]] padding : The padding of the convolution layer. If not provided, 0 will be used. -1 to keep the input size. -2 to fully use the kernel.
    ```
    """

    # Set the padding
    if isinstance(padding, int):
        if padding == -1:
            padding_x = kernel.shape[2] // 2
            padding_y = kernel.shape[3] // 2
        elif padding == -2:
            padding_x = kernel.shape[2] - 1
            padding_y = kernel.shape[3] - 1
        elif 0 <= padding:
            padding_x = padding
            padding_y = padding
        else:
            raise ValueError("The padding must be a positive integer (%d was gaven)" % padding)
    elif isinstance(padding, (tuple, list)):
        padding_x = padding[0]
        padding_y = padding[1]
        if padding_x < 0:
            raise ValueError("x's padding must be a positive integer (%d was gaven)" % padding_x)
        if padding_y < 0:
            raise ValueError("y's padding must be a positive integer (%d was gaven)" % padding_y)
    else: 
        raise ValueError("The padding must be an integer or a tuple of integers")
    
    # Set the stride
    if isinstance(stride, int):
        if stride < 1:
            raise ValueError("The stride must be greater than 1 (%d was gaven)" % stride)
        stride_x = stride
        stride_y = stride
    elif isinstance(stride, (tuple, list)):
        stride_x = stride[0]
        stride_y = stride[1]
        if stride_x < 1:
            raise ValueError("x's stride must be greater than 1 (%d was gaven)" % stride_x)
        if stride_y < 1:
            raise ValueError("y's stride must be greater than 1 (%d was gaven)" % stride_y)
    else: 
        raise ValueError("The stride must be an integer or a tuple of integers")

    # Check for errors
    if input.ndim != 3:
        raise ValueError("The input must be a 3D array")
    if kernel.ndim != 4:
        raise ValueError("The kernel must be a 4D array")
    if input.shape[0] != kernel.shape[1]:
        raise ValueError("The input's channel and kernel's input channel must have the same size")
    
    # Create the output array for storing the output
    output_size_x = (input.shape[1] - kernel.shape[2] + 1 + 2 * padding_x) // stride_x 
    output_size_y = (input.shape[2] - kernel.shape[3] + 1 + 2 * padding_y) // stride_y
    output = np.zeros((kernel.shape[0], output_size_x, output_size_y), dtype=np.float32)

    # Calulate the output and store it in the output array
    __conv.convolve2d_py(input.astype(np.float32), kernel.astype(np.float32), output, padding_x, padding_y, stride_x, stride_y)

    # Return the output
    return output