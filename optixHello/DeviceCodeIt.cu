#pragma once
#include <optix.h>

#include "params.h"
#include <cuda_runtime.h>
#include <math_functions.h>
#include <optix_device.h>


#include <curand.h>
#include <curand_kernel.h>

#define MAX_DISTRIBUTION 1024

extern "C" {
__constant__ Params params;
}

extern "C" {
    struct RayData
    {
        float3 n_origin = {};
        float3 n_direction = {};

        float weight = 0;
        float4 color = {};
        float blur = 0;

        bool terminate = false;
    };

}

static __forceinline__ __device__
RayData* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    RayData* ptr = reinterpret_cast<RayData*>(uptr);
    return ptr;
}

static __forceinline__ __device__
void  packPointer(RayData* ptr, uint32_t& i0, uint32_t& i1)
{
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}


extern "C" __forceinline__ __device__ void interpolate(uint2 & index, float u, float* us, float& ratio, int &ind) {
    ind = index.x;

    //Binary search was slower

    while (ind < index.x + index.y && us[ind + 1] < u) {
        ind++;
    }

    ratio = ((u - us[ind]) / (us[ind + 1] - us[ind]));
}

extern "C" __forceinline__ __device__ float4 getColor(uint2 &colorIndex, float u, float3* colors, float* color_u) {
    float color_ratio;
    int ind;
    interpolate(colorIndex, u, color_u,color_ratio,ind);

    return {
     colors[ind].x * (1 - color_ratio) + colors[ind +1].x * color_ratio,
     colors[ind].y * (1 - color_ratio) + colors[ind +1].y * color_ratio,
     colors[ind].z * (1 - color_ratio) + colors[ind +1].z * color_ratio,
     0
    };
}

//Returns the normal to the right of the curve direction
extern "C" __forceinline__ __device__ void calculateSplineNormal(float t, float3 * v, float3 & result) {
    result.x = (1 / 6.0f) * (3 * t * t * v[3].y + v[0].y * (-3 * t * t + 6 * t - 3) + v[1].y * (9 * t * t - 12 * t) + v[2].y * (-9 * t * t + 6 * t + 3));
    result.y = -(1 / 6.0f) * (3 * t * t * v[3].x + v[0].x * (-3 * t * t + 6 * t - 3) + v[1].x * (9 * t * t - 12 * t) + v[2].x * (-9 * t * t + 6 * t + 3));
    result.z = 0;
}

extern "C" __forceinline__ __device__ void calculateSpline(float t, float3 * v, float3 & result) {
    result.x = (1 / 6.0f) * (t * t * t * v[3].x + v[0].x * (-1 * t * t * t + 3 * t * t - 3 * t + 1) + v[1].x * (3 * t * t * t - 6 * t * t + 4) + v[2].x * (-3 * t * t * t + 3 * t * t + 3 * t + 1));
    result.y = (1 / 6.0f) * (t * t * t * v[3].y + v[0].y * (-1 * t * t * t + 3 * t * t - 3 * t + 1) + v[1].y * (3 * t * t * t - 6 * t * t + 4) + v[2].y * (-3 * t * t * t + 3 * t * t + 3 * t + 1));
    result.z = 0;
}

//Returns true if ray is hitting the right side of curve flips if we're using diffusion curve saves
extern "C" __forceinline__ __device__ bool isRayRight(float t, float3 & ray_direction, float3 * v) {
    float3 curve_normal = {};
    calculateSplineNormal(t, v, curve_normal);
    
    return (((curve_normal.x * ray_direction.x + curve_normal.y * ray_direction.y) <= 0) ^ USE_DIFFUSION_CURVE_SAVE);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int p0, p1, p2, p3,p4,rdp_1,rdp_2;
    float4 color = {0,0,0,0};
    float blur = 0;
    float3 ray_origin = {};
    float3 ray_direction = {};
    float weight_total = 0;
    float rot_cos, rot_sin;
    float rot_cos_r, rot_sin_r;
    RayData rd = {};

    packPointer(&rd, rdp_1, rdp_2);


    sincospif(2 / params.number_of_rays_per_pixel, &rot_sin, &rot_cos);
    

    //Flip y axis if loading diffusion curve save
    ray_origin.x = ((int) (idx.x - (params.image_width / 2))) * params.zoom_factor + params.offset_x;
    ray_origin.y = USE_DIFFUSION_CURVE_SAVE ? 
        ((int) ((params.image_height - idx.y) - (params.image_height / 2))) * params.zoom_factor + params.offset_y :
        ((int) (idx.y - (params.image_height / 2))) * params.zoom_factor + params.offset_y;
    ray_origin.z = 0;

    ray_direction.x = 1;
    ray_direction.y = 0;
    ray_direction.z = 0;  


    

    for (int i = 0; i < params.number_of_rays_per_pixel; i++) {
        sincospif((2 / params.number_of_rays_per_pixel) * curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]), &rot_sin_r, &rot_cos_r);


        rd.n_origin = {
                    ray_origin.x + (USE_AA ? (curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]) * params.zoom_factor) : 0),
                    ray_origin.y + (USE_AA ? (curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]) * params.zoom_factor) : 0),
                    ray_origin.z
        };
        rd.n_direction = {
            USE_AA ? ray_direction.x * rot_cos_r - ray_direction.y * rot_sin_r : ray_direction.x,
            USE_AA ? ray_direction.x * rot_sin_r + ray_direction.y * rot_cos_r : ray_direction.y,
            0
        };
        rd.terminate = false;
        

        unsigned int depth = 0;
        


        while (!rd.terminate && depth < MAX_TRACE_DEPTH) {
            optixTrace(
                //Add uniformly random amount within pixel 
                params.traversable,
                //Add uniformly random amount within angle of this sample 

                rd.n_origin,
                rd.n_direction,
                0.0f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,
                1,
                0,
                rdp_1,rdp_2
            );
            depth++;
        }
        //keep track of color weight
        weight_total += rd.weight;

        if (idx.x == 0 && idx.y == 256) {
           // printf("%.30f \n",  p3);
        }

       

        //Accumulate color
        color.x += rd.color.x * rd.weight;
        color.y += rd.color.y * rd.weight;
        color.z += rd.color.z * rd.weight;
        color.w += rd.color.w * rd.weight;
        blur += rd.blur * rd.weight;

        
        

        //Rotate Ray
        
        ray_direction = {
            ray_direction.x * rot_cos - ray_direction.y * rot_sin,
            ray_direction.x * rot_sin + ray_direction.y * rot_cos,
            0
        };
    }
   
    
    //Save average color
    params.image[idx.y * params.image_width + idx.x].x = color.x / weight_total;
    params.image[idx.y * params.image_width + idx.x].y = color.y / weight_total;
    params.image[idx.y * params.image_width + idx.x].z = color.z / weight_total;
    params.image[idx.y * params.image_width + idx.x].w = color.w / weight_total;
    //Save blur
    //printf("%f\n", blur / weight_total);
    params.blur_map[idx.y * params.image_width + idx.x] = blur / weight_total;
}

extern "C" __global__ void __miss__ms() 
{
    unsigned int rdp_1, rdp_2;
    rdp_1 = optixGetPayload_0();
    rdp_2 = optixGetPayload_1();
    RayData* rd = unpackPointer(rdp_1, rdp_2);

    rd->color = { 0,0,0,0 };
    rd->terminate = true;
    rd->weight = 0;
    rd->blur = 0;
}

extern "C" __global__ void __closesthit__ch()
{
    float segment_u = optixGetCurveParameter();
    float rt = optixGetRayTmax();
    int segment_number = optixGetPrimitiveIndex();
    float curve_u = segment_u  + params.curve_index[segment_number];
    float3 ray_direction = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();

    unsigned int rdp_1, rdp_2;
    rdp_1 = optixGetPayload_0();
    rdp_2 = optixGetPayload_1();
    RayData* rd = unpackPointer(rdp_1, rdp_2);

    //segment_u = segment_u > 1e-3 ? (segment_u < 1 - 1e-3 ? segment_u : 1e-3) : 1e-3;

    if (params.curve_connect[params.curve_map[segment_number]] >= 0) {

       //Calculate the starting point at the end of the portal
        float3 n_ray_origin = {};
        int target_segment_number = params.curve_map_inverse[params.curve_connect[params.curve_map[segment_number]]] + params.curve_index[segment_number];
        calculateSpline(segment_u, &(params.vertices[params.segmentIndices[target_segment_number]]), n_ray_origin);


        //Calculate the direction out of the portal
        float3 ray_normal = {};
        //printf("%d ", segment_number);

        calculateSplineNormal(segment_u, &(params.vertices[params.segmentIndices[segment_number]]), ray_normal);
        float ray_normal_size = sqrtf(ray_normal.x * ray_normal.x + ray_normal.y * ray_normal.y);
        ray_normal.x /= -ray_normal_size;
        ray_normal.y /= -ray_normal_size;


        float ray_cos = ray_normal.x * ray_direction.x + ray_normal.y * ray_direction.y;
        float ray_sin = ray_normal.x * ray_direction.y + ray_normal.y * ray_direction.x;

        float3 target_normal = {};
        float3 n_ray_direction = {};

        calculateSplineNormal(segment_u, &(params.vertices[params.segmentIndices[target_segment_number]]), target_normal);


        float target_normal_size = sqrtf(target_normal.x * target_normal.x + target_normal.y * target_normal.y);
        target_normal.x /= target_normal_size;
        target_normal.y /= target_normal_size;

        n_ray_direction.x = -(target_normal.x * ray_cos - target_normal.y * ray_sin);
        n_ray_direction.y = target_normal.y * ray_cos + target_normal.x * ray_sin;
        n_ray_direction.z = 0;


        //Continue the ray from the new point
       
        
        
        

    }
    else {

        int blur_ind;
        float blur_ratio;
        interpolate(params.blur_index[params.curve_map[segment_number]], curve_u, params.blur_u, blur_ratio, blur_ind);
        float blur = (1 - blur_ratio) * params.blur[blur_ind] + blur_ratio * params.blur[blur_ind + 1];
        rd->blur = blur;

        int weight_ind;
        float weight_ratio;
        interpolate(params.weight_index[params.curve_map[segment_number]], curve_u, params.weight_u, weight_ratio, weight_ind);
        float weight_multiplier = (1 - weight_ratio) * params.weight[weight_ind] + weight_ratio * params.weight[weight_ind + 1];

        int weight_degree_ind;
        float weight_degree_ratio;
        interpolate(params.weight_degree_index[params.curve_map[segment_number]], curve_u, params.weight_degree_u, weight_degree_ratio, weight_degree_ind);
        float weight_degree = (1 - weight_degree_ratio) * params.weight_degree[weight_degree_ind] + weight_degree_ratio * params.weight_degree[weight_degree_ind + 1];

        

        rd->weight = weight_multiplier * powf(rt, -weight_degree);



        if (isRayRight(segment_u, ray_direction, &(params.vertices[params.segmentIndices[segment_number]]))) {
            rd->color = getColor(params.color_right_index[params.curve_map[segment_number]], curve_u, params.color_right, params.color_right_u);
        }
        else {
            rd->color = getColor(params.color_left_index[params.curve_map[segment_number]], curve_u, params.color_left, params.color_left_u);
        }

        rd->terminate = true;
    }

}