cmake_minimum_required(VERSION 3.20)
project(cuda_convolution LANGUAGES CXX CUDA)

cmake_policy(SET CMP0104 OLD)
cmake_policy(SET CMP0146 OLD)

set(pybind11_DIR "~/anaconda3/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(CUDA REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
        convolutions
        src/core/convolution/wrapper.cpp
        src/core/convolution/1d/kernel.cu
        src/core/convolution/2d/kernel.cu
        src/core/convolution/3d/kernel.cu
)

target_compile_features(convolutions PRIVATE cxx_std_14)

set_target_properties(
        convolutions PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "./src/core/convolution"
)
