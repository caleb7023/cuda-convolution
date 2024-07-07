cmake_minimum_required(VERSION 3.18)
project(cuda_convolution LANGUAGES CXX CUDA)

set(pybind11_DIR "/home/caleb7023/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

pybind11_add_module(
    convolutions
    ./src/core/convolution/wrapper.cpp
    ./src/core/convolution/1d/kernel.cu
    ./src/core/convolution/2d/kernel.cu
    ./src/core/convolution/3d/kernel.cu
)

target_compile_features(convolutions PRIVATE cxx_std_14)

set_target_properties(
    convolutions PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "./src/core/convolution"
)