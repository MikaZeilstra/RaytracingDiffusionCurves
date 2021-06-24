/*
   Copyright 2021 Mika Zeilstra

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<curand_kernel.h>

//Pi
#define M_PI           3.14159265358979323846

//Minum added to the sigma value to not get divsion by zero
#define MINUM_SIGMA     1e-6f

//Block and grid sizes need to be manually optimized for your devices
#define BLOCK_SIZE             256
#define GRID_SIZE              512



//Kernel for setFloatDevice
extern "C" __global__ void setFloatKernel(float* dest, unsigned int n, float src) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        dest[i] = src;
    }
}

//Sets a device array to a constant float
extern "C" __host__ void setFloatDevice(float* dest, unsigned int n, float src, CUstream stream) {
   setFloatKernel << <GRID_SIZE, BLOCK_SIZE, 0, stream >> > (dest, n, src);
}

// Horizontal pass for the blur
extern "C" __global__ void gaussHorizontal(float4* source, float4* dest, float* sigma, int image_width,int image_height) {

    //pre-initalalize variables
    int source_loc = 0;
    float gauss_val = 0;
    float k_size = 0;
    float accum = 0;
    float sig_square = 0;



    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_width * image_height; i += blockDim.x * gridDim.x) {
        
        //Reset the accumulator of the weight
        accum = 0;

        //get kernel size such that it is 99 percentile of the sigmae
        k_size = 2 * ceilf(3 * sigma[i]) + 1;

        //Pre-calculate sigma squared
        sig_square = (sigma[i] + MINUM_SIGMA) * (sigma[i] + MINUM_SIGMA);

        //Zero the destination
        dest[i] = { 0,0,0 };

        //Loop for over the kernelsize starting at -1/2 kernel size to +1/2 kernelsize
        for (int k_i = -k_size / 2; k_i <= (k_size / 2); k_i++) {
            //Set the location this which the kernel is currently adding. going over the row left to right with clamping as border policy .
            source_loc = max(0, min(i % image_width + k_i, image_width - 1)) + (i / image_width) * image_width;

            //calculate the weight of this location
            gauss_val = expf(-(k_i * k_i) / sig_square);
            accum += gauss_val;         
            
            //Add the weighted colors to the destination
            dest[i].x += source[source_loc].x * gauss_val;
            dest[i].y += source[source_loc].y * gauss_val;
            dest[i].z += source[source_loc].z * gauss_val;
            dest[i].w += source[source_loc].w * gauss_val;
        }   
        

        //Normalize the color again
        dest[i].x /= accum;
        dest[i].y /= accum;
        dest[i].z /= accum;
        dest[i].w /= accum;
    }
}

//Vertical pass for the blur almost the same as horizontal pass except the source_loc
extern "C" __global__ void gaussVertical(float4 * source, float4 * dest, float* sigma, int image_width, int image_height) {
    int source_loc = 0;
    float gauss_val = 0;
    float k_size = 0;
    float accum = 0;

    float sig_square = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_width * image_height; i += blockDim.x * gridDim.x) {
        accum = 0;
        k_size = 2 * ceilf(3 * sigma[i]) + 1;
        sig_square = (sigma[i] + MINUM_SIGMA) * (sigma[i] + MINUM_SIGMA);

        dest[i] = { 0,0,0 };

        for (int k_i = -k_size / 2; k_i <= (k_size / 2); k_i++) {
            //Set the location this which the kernel is currently adding. going over the column top to bottom with clamping as border policy .

            source_loc = max(i % image_width, min((i + k_i * image_width) , (image_height- 1) * image_width + (i % image_width) ));
            gauss_val = expf(-(k_i * k_i) / sig_square);
            accum += gauss_val;

            dest[i].x += source[source_loc].x * gauss_val;
            dest[i].y += source[source_loc].y * gauss_val;
            dest[i].z += source[source_loc].z * gauss_val;
            dest[i].w += source[source_loc].w * gauss_val;
        }

        
        dest[i].x /= accum;
        dest[i].y /= accum;
        dest[i].z /= accum;
        dest[i].w /= accum;
    }

}

//Blurs the image with the provided blur map as sigmas in a vertical and horizontal pass with a linear gaussian kernel
extern "C" __host__ void gaussianBlur(float4 * dest, float4 * source, float* blur_map, int width, int height, CUstream stream) {
    float4* tmp;
    cudaMalloc(
        reinterpret_cast<void**>(&tmp),
        sizeof(float4) * width * height
    );

    gaussHorizontal << <GRID_SIZE, BLOCK_SIZE, 0, stream >> > (source,tmp, blur_map,width,height);
    gaussVertical << <GRID_SIZE, BLOCK_SIZE, 0, stream >> > (tmp, dest , blur_map, width, height);

    cudaFree(tmp);
}

//Kernel for setupCurand
extern "C" __global__ void setupCurandKernel(curandState_t* states, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += blockDim.x * gridDim.x) {
        curand_init(i, width, height, &states[i]);
    }
}

//Sets the curand states on the device
extern "C" __host__ void setupCurand(curandState_t* states, int width, int height, CUstream stream) {
    setupCurandKernel << <GRID_SIZE, BLOCK_SIZE, 0, stream >> > (states,width,height);
}

//Kernel for zeroImageFlow
extern "C" __global__ void zeroImageFlowKernel(float2 * flow, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += blockDim.x * gridDim.x) {
        flow[i] = {0,0};
    }
}

//Zeros the optical flow of the image
extern "C" __host__ void zeroImageFlow(float2 * flow, int width, int height,CUstream stream) {
    zeroImageFlowKernel << < GRID_SIZE, BLOCK_SIZE, 0, stream >> > (flow, width, height);
}

//Kernel for zoomImageFlow
extern "C" __global__ void zoomImageFlowKernel(float2 * flow,float zoom, float zoom_factor, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += blockDim.x * gridDim.x) {
        flow[i].x += (((int)(i - (width / 2))) % width) * zoom * zoom_factor - (((int)(i - (width / 2))) % width) * zoom_factor;
        flow[i].y += (((int)(i - (height / 2))) / width) * zoom * zoom_factor - (((int)(i - (height / 2))) / width) * zoom_factor;
    }
}

//Sets the optical flow of the image when zooming
extern "C" __host__ void zoomImageFlow(float2 * flow, float zoom, float zoom_factor, int width, int height, CUstream stream) {
    zoomImageFlowKernel << < GRID_SIZE, BLOCK_SIZE, 0, stream >> > (flow, zoom, zoom_factor, width, height);
}

//Kernel for zoomImageFlow translateImageFlow
extern "C" __global__ void translateImageFlowKernel(float2 * flow, float2 translation, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += blockDim.x * gridDim.x) {
        flow[i].x += translation.x;
        flow[i].y += translation.y;
    }
}


//Sets the optical flow of the image when panning
extern "C" __host__ void translateImageFlow(float2 * flow, float2 translation,int width, int height, CUstream stream) {
    translateImageFlowKernel << < GRID_SIZE, BLOCK_SIZE, 0, stream >> > (flow, translation, width, height);
}