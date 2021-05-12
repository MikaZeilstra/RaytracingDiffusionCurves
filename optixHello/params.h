#pragma once

struct Params
{
    float4* image;
    float3* vertices;
    int* segmentIndices;
    unsigned int image_width;
    unsigned int image_height;
    float number_of_rays_per_pixel;
    unsigned int frame;
    float zoom_factor;
    float offset_x;
    float offset_y;
    OptixTraversableHandle traversable;
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