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
#include <nvrtc.h>


#include "optixHello.h"
#include "params.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <array>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>




#ifndef CUDA_NVRTC_OPTIONS
#define CUDA_NVRTC_OPTIONS  \
  "-std=c++11", \
  "-arch", \
  "compute_60", \
  "-use_fast_math", \
  "-lineinfo", \
  "-default-device", \
  "-rdc", \
  "true", \
  "-D__x86_64",\
  "-I C:/Users/Mika/source/repos/SDK/optixHello/",\
  "-I C:/Users/Mika/source/repos/SDK/cuda/",\
  "-I C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0/include/",\
  "-I C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include/"
#define NUMBER_OF_CUDA_NVRTC_OPTIONS 13
#endif 
#ifndef CALL_CHECK
#define CALL_CHECK(call) \
    if(call != 0){          \
        std::cerr << "Error in Optix/CUDA call with number " << call << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        throw std::exception(); \
    }
#endif // !






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


    

    Params params{};
    
    GLFWwindow* window;
    

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



	optixDeviceContextCreate(cuCtx, &Context_options, &context);
	

    const std::array<float3, 6> vertices =
    { {
          {  128,-128, 0}, //Phantom point to connect to 1st control point

          {  128, 128, 0},
          {  128, 384, 0},
          {  384, 384, 0},
          {  384, 128, 0},
          {  384,-128, 0} //Phantom point to connect to last control point
    } };

  
    


    const std::array<float, 6> widths = { {0.5,0.5,0.5,0.5,0.5,0.5} };

    const std::array<unsigned int, 3> segmentIndices = { 0,1,2 };    

    const size_t vertices_size = sizeof(float3) * vertices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.vertices), vertices_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice
    ));

    const size_t widths_size = sizeof(float) * widths.size();
    CUdeviceptr d_widths = 0;
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), widths_size));
    CALL_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_widths),
        widths.data(),
        widths_size,
        cudaMemcpyHostToDevice
    ));

    

    CUdeviceptr              d_segementIndices = 0;
    const size_t segmentIndices_size = sizeof(unsigned int) * segmentIndices.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_segementIndices), segmentIndices_size));
    CALL_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_segementIndices), segmentIndices.data(),
        segmentIndices_size, cudaMemcpyHostToDevice));
    params.segmentIndices = reinterpret_cast<int*>(d_segementIndices);


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
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;


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



    size_t      inputSize = 0;
    size_t logsize;

    std::string devicecode;

    std::string devicecodeLocation = "C:/Users/Mika/source/repos/SDK/optixHello/DeviceCode.cu";

    //Read DeviceCode for compilation
    if (loadSource(devicecode, devicecodeLocation)) {
        //std::cout << devicecode << std::endl;
    }
    else {
        std::cerr << "Could Not read devicecode" << std::endl;
    }


    //Compile Devicecode
    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, devicecode.c_str(), "test", 0, NULL, NULL);

    const char* compiler_options[] = { CUDA_NVRTC_OPTIONS };
    nvrtcCompileProgram(prog, NUMBER_OF_CUDA_NVRTC_OPTIONS , compiler_options);
                
    std::string compileLog;
    size_t Compilelog_size = 0;
    nvrtcGetProgramLogSize(prog, &Compilelog_size);
    compileLog.resize(Compilelog_size);
    nvrtcGetProgramLog(prog, &compileLog[0]);

    if (Compilelog_size > 1) {
        std::cout << compileLog << std::endl;
    }
    

    size_t ptxSize;
    std::string ptx;

    nvrtcGetPTXSize(prog, &ptxSize);
    ptx.resize(ptxSize);

    nvrtcGetPTX(prog, &ptx[0]);

    nvrtcDestroyProgram(&prog);


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

    CALL_CHECK(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx.c_str(),
        ptxSize,
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
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;


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
    CUstream stream;
    CALL_CHECK(cudaStreamCreate(&stream));

    
    //Main Loop
    while (!glfwWindowShouldClose(window)) {
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
        //printf("\r frame : %d", params.frame);
        Sleep(50);

        glfwPollEvents();
        //glfwSwapBuffers(window);
    };

    

    
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
