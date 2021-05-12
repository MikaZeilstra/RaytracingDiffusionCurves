//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once
#include <optix.h>
#include "params.h"
#include <cuda_runtime.h>

#define DU 1e-3

extern "C" {
__constant__ Params params;
}


//Returns the normal to the right of the curve direction
extern "C" __forceinline__ __device__ void calculateSplineNormal(float t, float3 * v, float2 & result) {
    result.x = (1 / 6.0f) * (3 * t * t * v[3].y + v[0].y * (-3 * t * t + 6 * t - 3) + v[1].y * (9 * t * t - 12 * t) + v[2].y * (-9 * t * t + 6 * t + 3));
    result.y = -(1 / 6.0f) * (3 * t * t * v[3].x + v[0].x * (-3 * t * t + 6 * t - 3) + v[1].x * (9 * t * t - 12 * t) + v[2].x * (-9 * t * t + 6 * t + 3));
}

//Returns true if ray is hitting the right side of curve
extern "C" __forceinline__ __device__ bool checkLeftRight(float t, float3 & ray_direction, float3 * v) {
    float2 curve_normal = {};
    calculateSplineNormal(t, v, curve_normal);

    return ((curve_normal.x * ray_direction.x + curve_normal.y * ray_direction.y) > 0);
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int p0, p1, p2, p3;
    float4 color = {0,0,0,1};
    float3 ray_origin = {};
    float3 ray_direction = {};
    float weight_total = 0;
    float rot_cos, rot_sin;

    sincospif(2 / params.number_of_rays_per_pixel, &rot_sin, &rot_cos);


    ray_origin.x = idx.x * params.zoom_factor + params.offset_x;
    ray_origin.y = idx.y * params.zoom_factor + params.offset_y;
    ray_origin.z = 0;
    
    ray_direction.x = 1;
    ray_direction.y = 0;
    ray_direction.z = 0;  


    for (int i = 0; i < params.number_of_rays_per_pixel; i++) {

        
        
        

        optixTrace(
            params.traversable,
            ray_origin,
            ray_direction,
            0.0f,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0,
            1,
            0,
            p0,p1,p2,p3
        );
        
        float4 result = { int_as_float(p0),int_as_float(p1), int_as_float(p2), int_as_float(p3) };

        //keep track of color
        weight_total += result.w;

        //Accumulate color
        color.x += result.x * result.w;
        color.y += result.y * result.w;
        color.z += result.z * result.w;


        if (idx.x == 150 && idx.y == 50) {
            //printf("rayO : %f, %f, %f \n", ray_origin.x, ray_origin.y, ray_origin.z);
            //printf("result : %f, %f, %f, %f \n", result.x, result.y, result.z, result.w);
        }
        //Rotate Ray
        ray_direction = { ray_direction.x * rot_cos - ray_direction.y * rot_sin,
                        ray_direction.x * rot_sin + ray_direction.y * rot_cos,
                        0 };


        
    }

    
    
    //Save average color
    params.image[idx.y * params.image_width + idx.x].x = color.x / weight_total;
    params.image[idx.y * params.image_width + idx.x].y = color.y / weight_total;
    params.image[idx.y * params.image_width + idx.x].z = color.z / weight_total;

}

extern "C" __global__ void __miss__ms() 
{
    const uint3 idx = optixGetLaunchIndex();
    /*
	optixSetPayload_0(float_as_int((idx.x % 256) / 256.0f));
	optixSetPayload_1(float_as_int((idx.y % 256) / 256.0f));
	optixSetPayload_2(float_as_int(((idx.x + idx.y + params.frame)  % 256) /256.0f));

    */

    optixSetPayload_0(float_as_int(0));
    optixSetPayload_1(float_as_int(0));
    optixSetPayload_2(float_as_int(0));
    optixSetPayload_3(float_as_int(0));
}

extern "C" __global__ void __closesthit__ch()
{
    float u = optixGetCurveParameter();
    float rt = optixGetRayTmax();
    int vertex_index = optixGetPrimitiveIndex();
    float3 ray_direction = optixGetWorldRayDirection();
    float3 ray_origin = optixGetWorldRayOrigin();


    float weight = 1;    


    
    if (checkLeftRight(u, ray_direction, &(params.vertices[params.segmentIndices[ vertex_index]]))) {
        optixSetPayload_0(float_as_int(0));
        optixSetPayload_1(float_as_int(0));
        optixSetPayload_2(float_as_int(0));
        optixSetPayload_3(float_as_int(weight));
    }
    else {
        optixSetPayload_0(float_as_int(1));
        optixSetPayload_1(float_as_int(1));
        optixSetPayload_2(float_as_int(1));
        optixSetPayload_3(float_as_int(weight));        
    }
}