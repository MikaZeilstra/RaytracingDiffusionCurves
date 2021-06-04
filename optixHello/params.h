#pragma once
#include <optix.h>
#include <cuda_runtime.h>




#define USE_DIFFUSION_CURVE_SAVE true
#define USE_ENDCAPS              false
#define USE_WEIGHT_INTERPOLATION false



struct Params
{
    float4* image;
    unsigned int image_width;
    unsigned int image_height;

    float* random_numbers;

    float number_of_rays_per_pixel;
    OptixTraversableHandle traversable;

    int* segmentIndices;
    float3* vertices;

    unsigned int* curve_map;
    unsigned int* curve_index;

    uint2* color_left_index;
    float3* color_left;
    float* color_left_u;
    uint2* color_right_index;
    float3* color_right;
    float* color_right_u;


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
    
    float zoom_factor;
    float offset_x;
    float offset_y;
    
    unsigned int frame;
};

struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};