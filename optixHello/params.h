#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include<curand_kernel.h>



//Are we using diffusion curves saves ? swaps x and y axis, mirrors y axis and swaps r and b color chanels to produce correct image
#define USE_DIFFUSION_CURVE_SAVE true

//Enables blur, randomization and the optix denoiser respectively
#define USE_BLUR true
#define USE_AA true
#define USE_DENOISER true

//The maximum amount of times a ray can pass through any portal 31 is the max optix allows.
#define MAX_TRACE_DEPTH 2




struct Params
{ 
    //Final location for the image, the image of the previous frame and the optical flow of the image
    float4* image;
    float4* prev_image;
    float2* image_flow;

    //Stream used for all rendering
    CUstream stream;

    //Image width and height
    unsigned int image_width;
    unsigned int image_height;

    //Width * height amount of curand states for quick RNG
    curandState_t* curandStates;

    //Number of rays per pixel
    float number_of_rays_per_pixel;

    //Acceleration structure handle
    OptixTraversableHandle traversable;

    //Vertices and starting point of each spline
    float3* vertices;
    int* segmentIndices;
    
    //Map for converting spline indexes to curves and index specifies how far allong each spline in the curve is
    unsigned int* curve_map;
    unsigned int* curve_index;

    //curve_connects maps curves together and curve map inverse maps curves to the first spline of that curve.
    int* curve_connect;
    unsigned int* curve_map_inverse;

    //The left and right colors with the starting position and number of control points in the index and the control points and color in the corresponding variables
    uint2* color_left_index;
    float3* color_left;
    float* color_left_u;
    uint2* color_right_index;
    float3* color_right;
    float* color_right_u;

    //Following is the same but for blur weight multiplier (weight) and weight degree
    float* blur_map;
    uint2* blur_index;
    float* blur;
    float* blur_u;

    uint2* weight_index;
    float* weight;
    float* weight_u;
    
    uint2* weight_degree_index;
    float* weight_degree;
    float* weight_degree_u;
    
    //The zoom factor and offset
    float zoom_factor;
    float offset_x;
    float offset_y;
    
    //Current frame number
    unsigned int frame;
};

struct RayGenData
{
    // No data needed
};


struct MissData
{
   // No data needed
};


struct HitGroupData
{
    // No data needed
};