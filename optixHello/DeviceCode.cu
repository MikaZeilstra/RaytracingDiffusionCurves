/*
Copyright 2021 Mika Zeilstra

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissionsand
limitations under the License.
*/

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

//Given a set of points us finds the u the interval in which the given u lies and set the ratio  Index is used for offset and length of the search area within us
extern "C" __forceinline__ __device__ void interpolate(uint2 & index, float u, float* us, float& ratio, int &ind) {
    ind = index.x;

    while (ind < index.x + index.y && us[ind + 1] < u) {
        ind++;
    }

    ratio = ((u - us[ind]) / (us[ind + 1] - us[ind]));
}

//Calculates and sets the color payload
extern "C" __forceinline__ __device__ void setColorPayload(uint2 &colorIndex, float u, float3* colors, float* color_u) {
    float color_ratio;
    int ind;
    interpolate(colorIndex, u, color_u,color_ratio,ind);

    float3 color = {
     colors[ind].x * (1 - color_ratio) + colors[ind +1].x * color_ratio,
     colors[ind].y * (1 - color_ratio) + colors[ind +1].y * color_ratio,
     colors[ind].z * (1 - color_ratio) + colors[ind +1].z * color_ratio
    };

    optixSetPayload_0(float_as_int(color.x));
    optixSetPayload_1(float_as_int(color.y));
    optixSetPayload_2(float_as_int(color.z));
}

//Returns the normal to the right of the b-spline direction given 4 vertexes v
extern "C" __forceinline__ __device__ void calculateSplineNormal(float t, float3 * v, float3 & result) {
    result.x = (1 / 6.0f) * (3 * t * t * v[3].y + v[0].y * (-3 * t * t + 6 * t - 3) + v[1].y * (9 * t * t - 12 * t) + v[2].y * (-9 * t * t + 6 * t + 3));
    result.y = -(1 / 6.0f) * (3 * t * t * v[3].x + v[0].x * (-3 * t * t + 6 * t - 3) + v[1].x * (9 * t * t - 12 * t) + v[2].x * (-9 * t * t + 6 * t + 3));
    result.z = 0;
}

//Returns the point specified by t along the given b-spline given 4 vertexes v
extern "C" __forceinline__ __device__ void calculateSpline(float t, float3 * v, float3 & result) {
    result.x = (1 / 6.0f) * (t * t * t * v[3].x + v[0].x * (-1 * t * t * t + 3 * t * t - 3 * t + 1) + v[1].x * (3 * t * t * t - 6 * t * t + 4) + v[2].x * (-3 * t * t * t + 3 * t * t + 3 * t + 1));
    result.y = (1 / 6.0f) * (t * t * t * v[3].y + v[0].y * (-1 * t * t * t + 3 * t * t - 3 * t + 1) + v[1].y * (3 * t * t * t - 6 * t * t + 4) + v[2].y * (-3 * t * t * t + 3 * t * t + 3 * t + 1));
    result.z = 0;
}

//Returns true if ray is hitting the right side of curve flips if we're using diffusion curve saves
extern "C" __forceinline__ __device__ bool isRayRight(float t, float3 & ray_direction, float3 * vertices) {
    float3 curve_normal = {};
    calculateSplineNormal(t, vertices, curve_normal);
    
    return (((curve_normal.x * ray_direction.x + curve_normal.y * ray_direction.y) <= 0) ^ USE_DIFFUSION_CURVE_SAVE);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    //Setup space for the payload values p0,p1,p2 hold the color , p3 holds the weight and p4 holds the blur
    unsigned int p0, p1, p2, p3,p4;
    float4 color = {0,0,0,1};
    float blur = 0;
    float3 ray_origin = {};
    float3 ray_direction = {};
    float weight_total = 0;
    float rot_cos, rot_sin;
    float rot_cos_r, rot_sin_r;

    //Rotation we need to uniformly sample the light for each pixel with N rays
    sincospif(2 / params.number_of_rays_per_pixel, &rot_sin, &rot_cos);
    

    //Flip y axis if loading diffusion curve save
    ray_origin.x = ((int) (idx.x - (params.image_width / 2))) * params.zoom_factor + params.offset_x;
    ray_origin.y = USE_DIFFUSION_CURVE_SAVE ? 
        ((int) ((params.image_height - idx.y) - (params.image_height / 2))) * params.zoom_factor + params.offset_y :
        ((int) (idx.y - (params.image_height / 2))) * params.zoom_factor + params.offset_y;
    ray_origin.z = 0;

    //Setup first direction
    ray_direction.x = 1;
    ray_direction.y = 0;
    ray_direction.z = 0;  


    

    for (int i = 0; i < params.number_of_rays_per_pixel; i++) {

        //Calculate a random rotation within the current angle and apply it
        sincospif((2 / params.number_of_rays_per_pixel) * curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]), &rot_sin_r, &rot_cos_r);

        float3 random_direction = {
                ray_direction.x * rot_cos_r - ray_direction.y * rot_sin_r,
                ray_direction.x * rot_sin_r + ray_direction.y * rot_cos_r,
                0
        };

        //Set the initial trace depth to 0
        unsigned int depth = 0;

        optixTrace(
            //Add uniformly random amount within pixel 
            params.traversable,
            {
                ray_origin.x + (USE_AA ? (curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]) * params.zoom_factor) : 0),
                ray_origin.y + (USE_AA ? (curand_uniform(&params.curandStates[idx.y * params.image_width + idx.x]) * params.zoom_factor) : 0),
                ray_origin.z
            },
            //Add uniformly random amount within angle of this sample 
            USE_AA ? random_direction : ray_direction,
            0.0f,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            p0,p1,p2,p3,p4, depth
        );

        //keep track of color weight
        weight_total += int_as_float(p3);


        //Accumulate color
        color.x += int_as_float(p0) * int_as_float(p3);
        color.y += int_as_float(p1) * int_as_float(p3);
        color.z += int_as_float(p2) * int_as_float(p3);
        blur += int_as_float(p4) * int_as_float(p3);

        
        

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

    //Save blur
    params.blur_map[idx.y * params.image_width + idx.x] = blur / weight_total;
}

//If the ray misses set all payload values to zero such that it does not contribute
extern "C" __global__ void __miss__ms() 
{
    optixSetPayload_0(float_as_int(0));
    optixSetPayload_1(float_as_int(0));
    optixSetPayload_2(float_as_int(0));
    optixSetPayload_3(float_as_int(0));
    optixSetPayload_4(float_as_int(0));
}

extern "C" __global__ void __closesthit__ch()
{
    float segment_u = optixGetCurveParameter();
    float rt = optixGetRayTmax();
    int segment_number = optixGetPrimitiveIndex();
    float curve_u = segment_u  + params.curve_index[segment_number];
    float3 ray_direction = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();

    //Calculate blur and weight
    int blur_ind;
    float blur_ratio;
    interpolate(params.blur_index[params.curve_map[segment_number]], curve_u, params.blur_u, blur_ratio, blur_ind);
    float blur = (1 - blur_ratio) * params.blur[blur_ind] + blur_ratio * params.blur[blur_ind + 1];

    int weight_ind;
    float weight_ratio;
    interpolate(params.weight_index[params.curve_map[segment_number]], curve_u, params.weight_u, weight_ratio, weight_ind);
    float weight_multiplier = (1 - weight_ratio) * params.weight[weight_ind] + weight_ratio * params.weight[weight_ind + 1];

    int weight_degree_ind;
    float weight_degree_ratio;
    interpolate(params.weight_degree_index[params.curve_map[segment_number]], curve_u, params.weight_degree_u, weight_degree_ratio, weight_degree_ind);
    float weight_degree = (1 - weight_degree_ratio) * params.weight_degree[weight_degree_ind] + weight_degree_ratio * params.weight_degree[weight_degree_ind + 1];

    //If the connects part of the curve is set the ray needs to continue otherwise we just assign the color
    if (params.curve_connect[params.curve_map[segment_number]] >= 0) {

        //Increase depth by 1
        unsigned int depth = optixGetPayload_5() + 1;
        
        //Calculate the starting point at the end of the portal
        float3 n_ray_origin = {};
        int target_segment_number = params.curve_map_inverse[params.curve_connect[params.curve_map[segment_number]]] + params.curve_index[segment_number];
        calculateSpline(segment_u, &(params.vertices[params.segmentIndices[target_segment_number]]), n_ray_origin);
        
        
        //Calculate the direction out of the portal
        float3 ray_normal = {};

        //First we calculate the normal of the portal we hit
        calculateSplineNormal(segment_u, &(params.vertices[params.segmentIndices[segment_number]]), ray_normal);
        float ray_normal_size = sqrtf(ray_normal.x * ray_normal.x + ray_normal.y * ray_normal.y);
        ray_normal.x /= -ray_normal_size;
        ray_normal.y /= -ray_normal_size;

        //Now we calculate the sin and cos of the angle between the ray direction and the normal of the portal
        float ray_cos = ray_normal.x * ray_direction.x + ray_normal.y * ray_direction.y;
        float ray_sin = ray_normal.x* ray_direction.y + ray_normal.y * ray_direction.x;

        //Calculate the normal of the target portal
        float3 target_normal = {};
        float3 n_ray_direction = {};
        calculateSplineNormal(segment_u, &(params.vertices[params.segmentIndices[target_segment_number]]), target_normal);

        //Rotate the target normal such that it has the same angle with itself as the direction with the normal of the hit curve and use this as the new direction
        float target_normal_size = sqrtf(target_normal.x * target_normal.x + target_normal.y * target_normal.y);
        target_normal.x /= target_normal_size;
        target_normal.y /= target_normal_size;

        n_ray_direction.x = -(target_normal.x * ray_cos - target_normal.y * ray_sin);
        n_ray_direction.y = target_normal.y * ray_cos + target_normal.x * ray_sin;
        n_ray_direction.z = 0;
  
        
        //Continue the ray from the new point

        //Create space for the payload values p0,p1,p2 are the color, p3 is the weigth and p4 is the blur value
        unsigned int p0, p1, p2, p3, p4;
   
        //If we've not yet reached max trace depth shoot a continuation ray
        if (depth <= MAX_TRACE_DEPTH) {
            optixTrace(
                params.traversable,
                n_ray_origin,
                n_ray_direction,
                0.0f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0,
                1,
                0,
                p0, p1, p2, p3, p4, depth
            );

            float color_ratio;
            int ind;

            //Get the color filter of this portal
            float3 color_filter = {};
            if (isRayRight(segment_u, ray_direction, &(params.vertices[params.segmentIndices[segment_number]]))) {
                interpolate(params.color_right_index[params.curve_map[segment_number]], curve_u, params.color_right_u, color_ratio, ind);
                color_filter = {
                    params.color_right[ind].x * (1 - color_ratio) + params.color_right[ind + 1].x * color_ratio,
                    params.color_right[ind].y * (1 - color_ratio) + params.color_right[ind + 1].y * color_ratio,
                    params.color_right[ind].z * (1 - color_ratio) + params.color_right[ind + 1].z * color_ratio
                };
                
            }
            else {
                interpolate(params.color_right_index[params.curve_map[segment_number]], curve_u, params.color_left_u, color_ratio, ind);
                color_filter = {
                    params.color_left[ind].x * (1 - color_ratio) + params.color_left[ind + 1].x * color_ratio,
                    params.color_left[ind].y * (1 - color_ratio) + params.color_left[ind + 1].y * color_ratio,
                    params.color_left[ind].z * (1 - color_ratio) + params.color_left[ind + 1].z * color_ratio
                };
            }


            //Set all the payload values to the filtered values
            optixSetPayload_0(float_as_int(color_filter.x * int_as_float(p0)));
            optixSetPayload_1(float_as_int(color_filter.y * int_as_float(p1)));
            optixSetPayload_2(float_as_int(color_filter.z * int_as_float(p2)));
            optixSetPayload_3(float_as_int(1/((1/int_as_float(p3)) + 1/ (weight_multiplier * powf(rt, -weight_degree)))));
            optixSetPayload_4(float_as_int(blur * int_as_float(p4)));
        }
        else {
            //If we've reached the maximum trace depth we will regard it as a miss
            optixSetPayload_0(float_as_int(0));
            optixSetPayload_1(float_as_int(0));
            optixSetPayload_2(float_as_int(0));
            optixSetPayload_3(float_as_int(0));
            optixSetPayload_4(float_as_int(0));
        }
        
        
        
        
        

    }
    else {
        //The ray does not need to be continued thus we can simply set the color and blur
        optixSetPayload_3(float_as_int(weight_multiplier * powf(rt, -weight_degree)));

        optixSetPayload_4(float_as_int(blur));

        if (isRayRight(segment_u, ray_direction, &(params.vertices[params.segmentIndices[segment_number]]))) {
            setColorPayload(params.color_right_index[params.curve_map[segment_number]], curve_u, params.color_right, params.color_right_u);
        }
        else {
            setColorPayload(params.color_left_index[params.curve_map[segment_number]], curve_u, params.color_left, params.color_left_u);
        }
    }

}