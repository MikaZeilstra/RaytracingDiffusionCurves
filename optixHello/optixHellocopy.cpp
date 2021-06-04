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
#define GLFW_INCLUDE_NONE

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>


#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>


#include "optixHello.h"
#include "params.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#ifndef CALL_CHECK
#define CALL_CHECK(call) \
    if(call != 0){          \
        std::cerr << "Error in Optix/CUDA call with number " << call << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        throw std::exception(); \
    }
#endif // !

extern "C" __host__ void setFloatDevice(float* dest, unsigned int n, float src);

 extern "C" char embedded_ptx_code[];


template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

GLuint pbo;

int main(int argc, char* argv[]){

    const int width = 512;
    const int height = 512;
    const float zoom_factor = 1;
    const float offset_x = 0;
    const float offset_y = 0;
    const int number_of_rays = 128;
    
    const float curve_width = 1e-3f;

    auto start_time = std::chrono::high_resolution_clock::now();

    

    Params params{};
    
    GLFWwindow* window;
    
    //printf("ptx string :  %s", embedded_ptx_code);

    //Setup window
    glfwInit();
    window = glfwCreateWindow(width,height,"My first window",NULL,NULL);
    glfwMakeContextCurrent(window);

    gladLoadGL();
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    
    glfwSwapBuffers(window);

    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * width * height, NULL, GL_DYNAMIC_DRAW);
   
    cudaGraphicsResource* d_pbo;

    cudaGraphicsGLRegisterBuffer(&d_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

	//Create Context
	cudaFree(0);
	CUcontext cuCtx = 0;

	CALL_CHECK(optixInit());

    char log[2048];
    size_t sizeof_log = sizeof(log);


	OptixDeviceContextOptions Context_options = {};
    Context_options.logCallbackFunction = &logFunction;
    Context_options.logCallbackLevel = 4;
    
    OptixDeviceContext context = nullptr;

    CUstream stream;
    CALL_CHECK(cudaStreamCreate(&stream));

	optixDeviceContextCreate(cuCtx, &Context_options, &context);
	
    //Load curves 
    std::vector<float3> vertices = {};
    std::vector<unsigned int> curve_map = {};
    std::vector<unsigned int> curve_index = {};

    std::vector<uint2> color_left_index{};
    std::vector<float3> color_left = {};
    std::vector<float> color_left_u = {};

    std::vector<uint2> color_right_index{};
    std::vector<float3> color_right = {};
    std::vector<float> color_right_u = {};
    std::vector<unsigned int> segmentIndices;



    int current_segment = 0;
    int current_curve = 0;
    int current_curve_segment = 0;

    unsigned int n_colors_left = 0;
    unsigned int n_colors_right = 0;

    std::string file = std::string("/arch.xml");

        

    rapidxml::file<> xmlFile((std::filesystem::current_path().generic_string() + file).c_str());
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());
    rapidxml::xml_node<>* curve_set = doc.first_node();
    rapidxml::xml_node<>* set_node;

    for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {
        current_curve_segment = 0;
        set_node = curve->first_node("control_points_set",18);

        //Data is using bezier splines thus we need phantom points.
        rapidxml::xml_node<>* control_point = set_node->first_node();
        float y_pos;
        float x_pos;
        while ( control_point->next_sibling()) {
            vertices.push_back({ 0,0,0 });

            //Insert the 4 vertexes of the spline
            pushPoint(control_point, vertices);
            control_point = control_point->next_sibling();
            pushPoint(control_point, vertices);
            control_point = control_point->next_sibling();
            pushPoint(control_point, vertices);
            control_point = control_point->next_sibling();
            pushPoint(control_point, vertices);

            //Calculate phantom points

            vertices.at(current_segment) = {
                2 * vertices.at(current_segment + 1).x - vertices.at(current_segment + 2).x,
                2 * vertices.at(current_segment + 1).y - vertices.at(current_segment + 2).y,
                0
            };


            vertices.push_back({
                2 * vertices.at(current_segment + 4).x - vertices.at(current_segment + 3).x,
                2 * vertices.at(current_segment + 4).y - vertices.at(current_segment + 3).y,
                0
            });

            segmentIndices.push_back(current_segment++);
            segmentIndices.push_back(current_segment++);
            segmentIndices.push_back(current_segment++);
            current_segment += 3;

            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }

        set_node = curve->first_node("left_colors_set",15);
        color_left_index.push_back({ n_colors_left ,0 });
        rapidxml::xml_node<>* color_node = set_node->first_node();

        while (color_node) {
            float u = 0;
            pushColor(color_node, color_left_index, color_left_u, color_left);
            u = color_left_u.back();
            color_node = color_node->next_sibling();
        }
        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_left.push_back(color_left.back());
            color_left_index.back().y++;
            color_left_u.push_back(current_curve_segment);
        }

        n_colors_left += color_left_index.back().y;

        set_node = curve->first_node("right_colors_set", 16);
        color_right_index.push_back({ n_colors_right ,0 });
        color_node = set_node->first_node();

        while (color_node) {
            float u = 0;
            pushColor(color_node, color_right_index, color_right_u, color_right);
            u = color_right_u.back();
            color_node = color_node->next_sibling();
        }

        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_right.push_back(color_right.back());
            color_right_index.back().y++;
            color_right_u.push_back(current_curve_segment);
        }

        n_colors_right += color_right_index.back().y;

        current_curve++;

    }


    //upload vertex data
    const size_t widths_size = sizeof(float) * vertices.size();
    CUdeviceptr d_widths = 0;
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), widths_size));
    setFloatDevice(reinterpret_cast<float*>(d_widths),vertices.size(),curve_width);

 


    const size_t vertices_size = sizeof(float3) * vertices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.vertices), vertices_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));
    
  

    CUdeviceptr              d_segementIndices = 0;
    const size_t segmentIndices_size = sizeof(unsigned int) * segmentIndices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segementIndices), segmentIndices_size));
    CALL_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_segementIndices), segmentIndices.data(),
        segmentIndices_size, cudaMemcpyHostToDevice));
    params.segmentIndices = reinterpret_cast<int*>(d_segementIndices);

    //Upload curve -> spline map data
    const size_t curve_map_size = sizeof(unsigned int) * curve_map.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.curve_map), curve_map_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.curve_map),
        curve_map.data(),
        curve_map_size,
        cudaMemcpyHostToDevice
    ));

    const size_t curve_index_size = sizeof(unsigned int) * curve_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.curve_index), curve_index_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.curve_index),
        curve_index.data(),
        curve_index_size,
        cudaMemcpyHostToDevice
    ));



    //Upload color data
    const size_t color_left_index_size = sizeof(uint2) * color_left_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left_index), color_left_index_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_left_index),
        color_left_index.data(),
        color_left_index_size,
        cudaMemcpyHostToDevice
    ));


    const size_t color_left_size = sizeof(float3) * color_left.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left), color_left_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_left),
        color_left.data(),
        color_left_size,
        cudaMemcpyHostToDevice
    ));

    const size_t color_left_u_size = sizeof(float) * color_left_u.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left_u), color_left_u_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_left_u),
        color_left_u.data(),
        color_left_u_size,
        cudaMemcpyHostToDevice
    ));

    const size_t color_right_index_size = sizeof(uint2) * color_right_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right_index), color_right_index_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_right_index),
        color_right_index.data(),
        color_right_index_size,
        cudaMemcpyHostToDevice
    ));

    const size_t color_right_size = sizeof(float3) * color_right.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right), color_right_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_right),
        color_right.data(),
        color_right_size,
        cudaMemcpyHostToDevice
    ));

    const size_t color_right_u_size = sizeof(float) * color_right_u.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right_u), color_right_u_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.color_right_u),
        color_right_u.data(),
        color_right_u_size,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput curve_input = {};
    curve_input.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    curve_input.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    curve_input.curveArray.numPrimitives = static_cast<uint32_t>(segmentIndices.size());
    curve_input.curveArray.vertexBuffers = reinterpret_cast<CUdeviceptr*>(&params.vertices);
    curve_input.curveArray.numVertices = static_cast<uint32_t>(vertices.size());;
    curve_input.curveArray.vertexStrideInBytes = 0;
    curve_input.curveArray.widthBuffers = &d_widths;
    curve_input.curveArray.widthStrideInBytes = 0;
    curve_input.curveArray.normalBuffers = 0;
    curve_input.curveArray.normalStrideInBytes = 0;
    curve_input.curveArray.indexBuffer = d_segementIndices;
    curve_input.curveArray.indexStrideInBytes = 0;
    curve_input.curveArray.flag = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    curve_input.curveArray.primitiveIndexOffset = 0;


    //Create acceleration structure
    OptixTraversableHandle gas_handle;
    CUdeviceptr            d_gas_output_buffer;


    OptixAccelBuildOptions accel_options = {};
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;


    OptixAccelBufferSizes gas_buffer_sizes;
    CALL_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &curve_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));
    CUdeviceptr d_temp_buffer_gas;
    CALL_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes
    ));
    CALL_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes
    ));
        
    CALL_CHECK(optixAccelBuild(
        context,
        0,                 
        &accel_options,
        &curve_input,
        1,                  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr, 
        0
    ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

    //Setup Modules
    OptixModule module = nullptr;

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    //Setup pipeline cimpile options
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 4;
    pipeline_compile_options.numAttributeValues = 0;

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

    const std::string ptxCode = embedded_ptx_code;

    CALL_CHECK(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptxCode.c_str(),
        ptxCode.size(),
        log,
        &sizeof_log,
        &module
    ));


    OptixModule is_module;
    OptixBuiltinISOptions ISOptions = {};
    ISOptions.usesMotionBlur = false;
    ISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;

    CALL_CHECK(optixBuiltinISModuleGet(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        &ISOptions,
        &is_module
    ));

    //Make Program groups
    OptixProgramGroup raygen_prog_group = nullptr;
    OptixProgramGroup miss_prog_group = nullptr;
    OptixProgramGroup hitgroup_prog_group = nullptr;
    
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    CALL_CHECK(optixProgramGroupCreate(
        context,
        &raygen_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &raygen_prog_group
    ));


    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    CALL_CHECK(optixProgramGroupCreate(
        context,
        &miss_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &miss_prog_group
    ));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleIS = is_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = 0;

    sizeof_log = sizeof(log);
    CALL_CHECK(optixProgramGroupCreate(
        context,
        &hitgroup_prog_group_desc,
        1,   // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &hitgroup_prog_group
    ));  
    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };



    //Make Pipeline
    OptixPipeline pipeline;
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;


    CALL_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline
    ));


    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups)
    {
        CALL_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    CALL_CHECK(optixUtilComputeStackSizes(&stack_sizes, 1,
        0,  // maxCCDepth
        0,  // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    CALL_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1  // maxTraversableDepth
    ));


    //Make Shader binding table
    OptixShaderBindingTable sbt = {};

    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size);
    RayGenSbtRecord rg_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    cudaMemcpy(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    );


    CUdeviceptr miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord);
    cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size);
    MissSbtRecord miss_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(miss_prog_group, &miss_sbt));
    cudaMemcpy(
        reinterpret_cast<void*>(miss_record),
        &miss_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice
    );

    CUdeviceptr hg_record;
    const size_t  hg_record_size = sizeof(HitGroupSbtRecord);
    cudaMalloc(reinterpret_cast<void**>(&hg_record), hg_record_size);
    HitGroupSbtRecord hg_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    cudaMemcpy(
        reinterpret_cast<void*>(hg_record),
        &hg_sbt,
        hg_record_size,
        cudaMemcpyHostToDevice
    );

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.hitgroupRecordBase = hg_record;
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    
    


    //Setup Parameters

    cudaGraphicsMapResources(1, &d_pbo, NULL);
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&(params.image)), NULL, d_pbo);

    params.image_width = width;
    params.image_height = height;
    params.frame = 0;
    params.traversable = gas_handle;
    params.zoom_factor =  zoom_factor;
    params.offset_x = offset_x;
    params.offset_y = offset_y;
    params.number_of_rays_per_pixel = number_of_rays;

    CUdeviceptr d_param;
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));

    CALL_CHECK(cudaDeviceSynchronize());


    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
    std::cout << "Setup took : " << setup_duration.count() << " ms" << std::endl;


    long long  total_frame_time = 0;

    //Main Loop
    while (!glfwWindowShouldClose(window)) {
        start_time = std::chrono::high_resolution_clock::now();


        cudaGraphicsMapResources(1, &d_pbo, NULL);
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&(params.image)), NULL, d_pbo);


        //Run pipeline
        CALL_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_param),
            &params, sizeof(Params),
            cudaMemcpyHostToDevice
        ));

        //Launch pipeline
        CALL_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, 1))
        CALL_CHECK(cudaStreamSynchronize(stream));


        glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
        cudaGraphicsUnmapResources(1, &d_pbo, NULL);
        glfwSwapBuffers(window);

        params.frame++;
        printf("\rframe : %d", params.frame);

        glfwPollEvents();
        total_frame_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();

        //glfwSwapBuffers(window);
    };
    std::cout << std::endl;
    std::cout << "Average frame time  : " << total_frame_time / (float) params.frame  << " ms" << std::endl;
    

    
    //Cleanup


    cudaGLUnregisterBufferObject(pbo);
    cudaGraphicsUnregisterResource(d_pbo);
    glDeleteBuffers(1, &pbo);

    glfwDestroyWindow(window);
    glfwTerminate();

    CALL_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    

    CALL_CHECK(optixPipelineDestroy(pipeline));
    CALL_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    CALL_CHECK(optixProgramGroupDestroy(miss_prog_group));
    CALL_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    CALL_CHECK(optixModuleDestroy(module));

    CALL_CHECK(optixDeviceContextDestroy(context));

    cudaDeviceReset();
}



static void logFunction(unsigned int level, const char* tag, const char* message, void*) {
    printf("%d - %s %s \n" ,level, tag  , message);
}

static void printUChar4(uchar4* uchar) {
    printf("R: %d, G: %d, B: %d, A: %d \n", uchar->x, uchar->y, uchar->z, uchar->w);
}

static bool loadSource(std::string& dest, const std::string& loc) {
    std::ifstream file(loc, std::ios::binary);
    if (file.good()) {
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        dest.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}

// B and R color channels switched if load diffusion curve xml
static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color) {
    float u = (std::atof(color_node->first_attribute("globalID", 8)->value()) / 10.0f);
    color.push_back({
        std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "B" : "R",1)->value()) / 255.0f,
        std::atoi(color_node->first_attribute("G",1)->value()) / 255.0f,
        std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "R" : "B",1)->value()) / 255.0f
        });
    color_u.push_back(u);
    ind.back().y++;
}

//Switch x y if loading diffusion curve xml
static void pushPoint(rapidxml::xml_node<>* control_node, std::vector<float3>& vertices) {
    vertices.push_back({
        (float) std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value()) ,
        (float) std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) ,
        0 });

}
