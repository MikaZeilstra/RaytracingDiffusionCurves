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
#include "glfw_events.h"
#include "params.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <array>
#include <chrono>
#include <filesystem>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>


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

const float bspline_correction_matrix[] = { 6,-7,2,0,
                                            0,2,-1,0,
                                            0, - 1,2,0,
                                            0,2,-7,6 };

int main(int argc, char* argv[]){
    


    const int width = 512;
    const int height = 512;
    const float zoom_factor = 1;
    const float offset_x = width / 2;
    const float offset_y = height / 2;
    const int number_of_rays = 256;
    const std::string file_name = "/arch.xml";


    const float curve_width = 1e-2f;

    auto start_time = std::chrono::high_resolution_clock::now();

    

    Params params{};
    
    GLFWwindow* window;
    
    //printf("ptx string :  %s", embedded_ptx_code);

    //Setup window
    glfwInit();
    window = glfwCreateWindow(width,height,"My first window",NULL,NULL);
    glfwMakeContextCurrent(window);

    glfwSetWindowUserPointer(window, &params);

    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_cursor_callback);


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

    std::vector<uint2> weight_index = {};
    std::vector<float> weight = {};
    std::vector<float> weight_u = {};

    int current_segment = 0;
    int current_curve = 0;
    int current_curve_segment = 0;

    unsigned int n_colors_left = 0;
    unsigned int n_colors_right = 0;
    unsigned int n_weights = 0;
        

    rapidxml::file<> xmlFile((std::filesystem::current_path().generic_string() + file_name).c_str());
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());
    rapidxml::xml_node<>* curve_set = doc.first_node();
    rapidxml::xml_node<>* set_node;
    rapidxml::xml_node<>* current_node;

    for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {
        //Read control points
        current_curve_segment = 0;
        set_node = curve->first_node("control_points_set",18);

        current_node = set_node->first_node();
        while ( current_node->next_sibling()) {
            //Insert the 4 vertexes of the spline
            push4Points(current_node, vertices);

            segmentIndices.push_back(current_segment++);


            current_segment += 3;

            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }

        //Read left colors
        set_node = curve->first_node("left_colors_set",15);
        color_left_index.push_back({ n_colors_left ,0 });
        current_node = set_node->first_node();

        while (current_node) {
            pushColor(current_node, color_left_index, color_left_u, color_left);
            current_node = current_node->next_sibling();
        }
        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_left.push_back(color_left.back());
            color_left_index.back().y++;
            color_left_u.push_back(current_curve_segment);
        }

        n_colors_left += color_left_index.back().y;


        //Read right colors
        set_node = curve->first_node("right_colors_set", 16);
        color_right_index.push_back({ n_colors_right ,0 });
        current_node = set_node->first_node();

        while (current_node) {
            pushColor(current_node, color_right_index, color_right_u, color_right);
            current_node = current_node->next_sibling();
        }

        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_right.push_back(color_right.back());
            color_right_index.back().y++;
            color_right_u.push_back(current_curve_segment);
        }

        n_colors_right += color_right_index.back().y;

       


        if (USE_WEIGHT_INTERPOLATION) {
            set_node = curve->first_node("weight_set", 10);

            weight_index.push_back({ n_weights , 0 });
            current_node = set_node->first_node();
            
            while (current_node) {
                pushSingle(current_node, weight_index, weight_u, weight, "w");
                current_node = current_node->next_sibling();
            }


            n_weights += color_right_index.back().y;

        }


        current_curve++;
    }

    


    //upload vertex data
    const size_t widths_size = sizeof(float) * vertices.size();
    CUdeviceptr d_widths = 0;
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), widths_size));
    setFloatDevice(reinterpret_cast<float*>(d_widths),vertices.size(),curve_width);

 


    const size_t vertices_size = sizeof(float3) * vertices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.vertices), vertices_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice,
        stream
    ));
    
 
    const size_t segmentIndices_size = sizeof(unsigned int) * segmentIndices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.segmentIndices), segmentIndices_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.segmentIndices),
        segmentIndices.data(),
        segmentIndices_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    //Upload curve -> spline map data
    const size_t curve_map_size = sizeof(unsigned int) * curve_map.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.curve_map), curve_map_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_map),
        curve_map.data(),
        curve_map_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t curve_index_size = sizeof(unsigned int) * curve_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.curve_index), curve_index_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_index),
        curve_index.data(),
        curve_index_size,
        cudaMemcpyHostToDevice,
        stream
    ));



    //Upload color data
    const size_t color_left_index_size = sizeof(uint2) * color_left_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left_index), color_left_index_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left_index),
        color_left_index.data(),
        color_left_index_size,
        cudaMemcpyHostToDevice,
        stream
    ));


    const size_t color_left_size = sizeof(float3) * color_left.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left), color_left_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left),
        color_left.data(),
        color_left_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t color_left_u_size = sizeof(float) * color_left_u.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_left_u), color_left_u_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left_u),
        color_left_u.data(),
        color_left_u_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t color_right_index_size = sizeof(uint2) * color_right_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right_index), color_right_index_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right_index),
        color_right_index.data(),
        color_right_index_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t color_right_size = sizeof(float3) * color_right.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right), color_right_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right),
        color_right.data(),
        color_right_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t color_right_u_size = sizeof(float) * color_right_u.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.color_right_u), color_right_u_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right_u),
        color_right_u.data(),
        color_right_u_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    //Upload Weights
    if (USE_WEIGHT_INTERPOLATION) {
        const size_t weight_index_size = sizeof(uint2) * weight_index.size();
        CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.weight_index), weight_index_size));
        CALL_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(params.weight_index),
            weight_index.data(),
            weight_index_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        const size_t weight_size = sizeof(float) * weight.size();
        CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.weight), weight_size));
        CALL_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(params.weight),
            weight.data(),
            weight_size,
            cudaMemcpyHostToDevice,
            stream
        ));

        const size_t weight_u_size = sizeof(float) * weight_u.size();
        CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.weight_u), weight_u_size));
        CALL_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(params.weight_u),
            weight_u.data(),
            weight_u_size,
            cudaMemcpyHostToDevice,
            stream
        ));
    }



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
    curve_input.curveArray.indexBuffer = reinterpret_cast<CUdeviceptr>(params.segmentIndices);
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
    CALL_CHECK(cudaStreamSynchronize(stream));


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
#pragma inline
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
#pragma inline
static void push4Points(rapidxml::xml_node<>*& control_node, std::vector<float3>& vertices) {
    float* controls_xy = new float[8]; 

    for (int i = 0; i < 6; i+= 2) {
        controls_xy[i] = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value());
        controls_xy[i+1] = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value());
        control_node = control_node->next_sibling();
    }

    controls_xy[6] = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value());
    controls_xy[7] = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value());


    for (int i = 0; i < 4; i++) {
        if (USE_DIFFUSION_CURVE_SAVE) {
            vertices.push_back({
                controls_xy[0] * bspline_correction_matrix[i * 4] + controls_xy[2] * bspline_correction_matrix[i * 4 + 1] + controls_xy[4] * bspline_correction_matrix[i * 4 + 2] + controls_xy[6] * bspline_correction_matrix[i * 4 + 3],
                controls_xy[1] * bspline_correction_matrix[i * 4] + controls_xy[3] * bspline_correction_matrix[i * 4 + 1] + controls_xy[5] * bspline_correction_matrix[i * 4 + 2] + controls_xy[7] * bspline_correction_matrix[i * 4 + 3],
                0
            });
        }
        else {
            vertices.push_back({
                controls_xy[i * 2],
                controls_xy[i * 2 + 1],
                0
            });
        
        }
        

    }

    delete[] controls_xy;
}

#pragma inline
static void pushSingle(rapidxml::xml_node<>* node, std::vector<uint2>& ind, std::vector<float>& us, std::vector<float>& target, const char* name) {
    float u = (std::atof(node->first_attribute("globalID", 8)->value()) / 10.0f);
    target.push_back(std::atof(node->first_attribute(name)->value()));
    us.push_back(u);
    ind.back().y++;
}