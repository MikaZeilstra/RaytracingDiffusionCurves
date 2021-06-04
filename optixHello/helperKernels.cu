#pragma once
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>

extern "C" __global__ void setFloatKernel(float* dest, unsigned int n, float src) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        dest[i] = src;
    }
}

extern "C" __host__ void setFloatDevice(float* dest, unsigned int n, float src) {
   setFloatKernel << <256, 16 >> > (dest, n, src);
}

extern "C" __host__ void gaussianBlur(float4 * dest, float4 * source, float* strengths, int width, int height) {
}