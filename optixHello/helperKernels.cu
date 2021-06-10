#pragma once
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#include<curand_kernel.h>

# define M_PI           3.14159265358979323846
#define MINUM_SIGMA     1e-6f

#define BLOCK_SIZE             256
#define GRID_SIZE              512

extern "C" __global__ void setFloatKernel(float* dest, unsigned int n, float src) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x){
        dest[i] = src;
    }
}

extern "C" __host__ void setFloatDevice(float* dest, unsigned int n, float src) {
   setFloatKernel << <GRID_SIZE, BLOCK_SIZE >> > (dest, n, src);
}

extern "C" __global__ void gaussHorizontal(float4* source, float4* dest, float* sigma, int image_width,int image_height) {
    int source_loc = 0;
    float gauss_val = 0;
    float k_size = 0;
    float accum = 0;

    float sig_2_square = 0;



    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_width * image_height; i += blockDim.x * gridDim.x) {
        
        //get kernel size such that it is 99 percentile
        accum = 0;
        k_size = 2 * ceilf(3 * sigma[i]) + 1;
        sig_2_square = (sigma[i] + MINUM_SIGMA) * (sigma[i] + MINUM_SIGMA);

        dest[i] = { 0,0,0 };

        for (int k_i = -k_size / 2; k_i <= (k_size / 2); k_i++) {
            source_loc = max(0, min(i % image_width + k_i, image_width - 1)) + (i / image_width) * image_width;
            gauss_val = expf(-(k_i * k_i) / sig_2_square);
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

        if (dest[i].x < 0 && dest[i].y < 0) {
            printf("%d\n", i);
        }

        //dest[i] = source[i];
    }
}

extern "C" __global__ void gaussVertical(float4 * source, float4 * dest, float* sigma, int image_width, int image_height) {
    int source_loc = 0;
    float gauss_val = 0;
    float k_size = 0;
    float accum = 0;

    float sig_2_square = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < image_width * image_height; i += blockDim.x * gridDim.x) {

        //get kernel size such that it is 99 percentile
        accum = 0;
        k_size = 2 * ceilf(3 * sigma[i]) + 1;
        sig_2_square = (sigma[i] + MINUM_SIGMA) * (sigma[i] + MINUM_SIGMA);

        dest[i] = { 0,0,0 };

        for (int k_i = -k_size / 2; k_i <= (k_size / 2); k_i++) {
            source_loc = max(i % image_width, min((i + k_i * image_width) , (image_height- 1) * image_width + (i % image_width) ));
            gauss_val = expf(-(k_i * k_i) / sig_2_square);
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

        //dest[i] = source[i];

    }

}

extern "C" __host__ void gaussianBlur(float4 * dest, float4 * source, float* blur_map, int width, int height) {
    float4* tmp;
    cudaMalloc(
        reinterpret_cast<void**>(&tmp),
        sizeof(float4) * width * height
    );

    gaussHorizontal << <GRID_SIZE, BLOCK_SIZE >> > (source,tmp, blur_map,width,height);
    gaussVertical << <GRID_SIZE, BLOCK_SIZE >> > (tmp, dest , blur_map, width, height);

    cudaFree(tmp);
}


extern "C" __global__ void setupCurandKernel(curandState_t* states, int width, int height) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < width * height; i += blockDim.x * gridDim.x) {
        curand_init(i, width, height, &states[i]);
    }
}

extern "C" __host__ void setupCurand(curandState_t* states, int width, int height) {
    setupCurandKernel << <GRID_SIZE, BLOCK_SIZE >> > (states,width,height);
}

