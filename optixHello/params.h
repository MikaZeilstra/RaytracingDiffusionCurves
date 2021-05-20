#pragma once
#define USE_DIFFUSION_CURVE_SAVE true

struct Params
{
    float4* image;
    unsigned int image_width;
    unsigned int image_height;

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

    float* blur;

      
    
    
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