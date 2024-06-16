
// author: caleb7023

#include "../core/convolution2d.cuh"
#include <stdio.h>




int main() {

    float inputs[5*5*3] = {
        // channel 1
        1 , 2 , 3 , 4 , 5 ,
        6 , 7 , 8 , 9 , 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,

        // channel 2
        26, 27, 28, 29, 30,
        31, 32, 33, 34, 35,
        36, 37, 38, 39, 40,
        41, 42, 43, 44, 45,
        46, 47, 48, 49, 50,

        // channel 3
        51, 52, 53, 54, 55,
        56, 57, 58, 59, 60,
        61, 62, 63, 64, 65,
        66, 67, 68, 69, 70,
        71, 72, 73, 74, 75,

    };

    float kernels[1*3*3*3] = {
        // channel 1
            // kernel channel 1
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
            // kernel channel 2
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
            // kernel channel 3
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
    };

    float output[3*3];

    convolution2d(inputs, output, kernels, 3, 1, 5, 5, 3, 3, 0, 0, 0, 0);

    for (int i = 0; i < 9; i++) {
        printf("%f\n", output[i]);
    }

    return 0;
}