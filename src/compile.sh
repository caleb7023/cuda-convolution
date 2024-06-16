#!/bin/bash
# execute like ./src/compile.sh convolution2d 2d true


nvcc -c src/core/convolution/$2/$1.cu -o src/core/convolution/$2/$1.o
g++ src/debug/$1_debug.cpp src/core/convolution/$2/$1.o -lcudart -o src/debug/$1_debug.o

if [ $3 = "true" ]; then
    ./src/debug/$1_debug.o
fi