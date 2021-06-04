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
                                            0, -1,2,0,
                                            0,2,-7,6 };

int main(int argc, char* argv[]) {



    int width = 0;
    int height = 0;
    const float zoom_factor = 1;
    const float offset_x = 0;
    const float offset_y = 0;
    const int number_of_rays = 512;
    const float endcap_size = 8;
    const std::string file_name = "/lady_bug.xml";


    const float curve_width = 1e-3f;

    auto start_time = std::chrono::high_resolution_clock::now();

    rapidxml::file<> xmlFile((std::filesystem::current_path().generic_string() + file_name).c_str());
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());
    rapidxml::xml_node<>* curve_set = doc.first_node();
    rapidxml::xml_node<>* set_node;
    rapidxml::xml_node<>* current_node;

    width = std::atoi(curve_set->first_attribute("image_width")->value());
    height = std::atoi(curve_set->first_attribute("image_height")->value());

    Params params{};

    GLFWwindow* window;

    //Setup window
    glfwInit();
    window = glfwCreateWindow(width, height, "My first window", NULL, NULL);
    glfwMakeContextCurrent(window);

    glfwSetWindowUserPointer(window, &params);

    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_cursor_callback);


    gladLoadGL();
    glClearColor(0, 0, 0, 1);
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

    std::vector<uint2> blur_index = {};
    std::vector<float> blur = {};
    std::vector<float> blur_u = {};

    std::vector<uint2> weight_index = {};
    std::vector<float> weight = {};
    std::vector<float> weight_u = {};

    int current_segment = 0;
    int current_curve = 0;
    int current_curve_segment = 0;

    unsigned int n_colors_left = 0;
    unsigned int n_colors_right = 0;
    unsigned int n_weights = 0;
    unsigned int n_blurs = 0;


   
    for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {
        //Read control points
        current_curve_segment = 0;
        set_node = curve->first_node("control_points_set", 18);

        current_node = set_node->first_node();


        //Setup andcap for start of curve
        if (USE_ENDCAPS) {
            float3* endcap = new float3[4];
            float3* first_curve = new float3[4];

            rapidxml::xml_node<>* endcap_node = current_node;
            endcap[0] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width/2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                });
            endcap[3] = endcap[0];

            for (int i = 0; i < 4; i++) {
                first_curve[i] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width / 2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                    });

                endcap_node = endcap_node->next_sibling();
            }

            float3 tan = {};
            getBezierTangent(0, first_curve, tan);

            tan = { -tan.x, -tan.y, tan.z };

            getEndcapPoints(endcap[0], tan, endcap[1], endcap[2],endcap_size);
            correctControlPoints(endcap, vertices);

            delete[] endcap;
            delete[] first_curve;

            segmentIndices.push_back(current_segment++);
            current_segment += 3;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }
        

        while (current_node->next_sibling()) {
            //Insert the 4 vertexes of the spline
            push4Points(current_node, vertices,width,height);

            segmentIndices.push_back(current_segment++);
            current_segment += 3;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }

        if (USE_ENDCAPS) {
            float3* endcap = new float3[4];
            float3* first_curve = new float3[4];

            rapidxml::xml_node<>* endcap_node = current_node;
            endcap[0] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width / 2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                });
            endcap[3] = endcap[0];

            endcap_node = endcap_node->previous_sibling()->previous_sibling()->previous_sibling();

            for (int i = 0; i < 4; i++) {
                first_curve[i] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width / 2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                    });

                endcap_node = endcap_node->next_sibling();
            }

            float3 tan = {};
            getBezierTangent(1, first_curve, tan);

            tan = { tan.x, tan.y, tan.z };

            getEndcapPoints(endcap[0], tan, endcap[1], endcap[2],endcap_size);
            correctControlPoints(endcap, vertices);

            delete[] endcap;
            delete[] first_curve;

            segmentIndices.push_back(current_segment++);
            current_segment += 3;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }

        //Read left colors
        set_node = curve->first_node("left_colors_set", 15);
        color_left_index.push_back({ n_colors_left ,0 });
        current_node = set_node->first_node();

        if (USE_ENDCAPS) {
            color_right.push_back({ 0,0,0 });
            color_left.push_back({ 0,0,0 });
            color_left.push_back({ 0,0,0 });

            color_right_u.push_back(0);
            color_left_u.push_back(0);
            color_left_u.push_back(1);
        }

        while (current_node) {
            pushColor(current_node, color_left_index, color_left_u, color_left);
            current_node = current_node->next_sibling();
        }

        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_left.push_back(color_left.back());
            color_left_index.back().y++;
            color_left_u.push_back(current_curve_segment-1);
        }     


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
            color_right_u.push_back(current_curve_segment-1);
        }

        

        if (USE_ENDCAPS) {
            //First Colors
            color_left.at(color_left_index.back().x) = (color_left.at(color_left_index.back().x + 2));
            color_left.at(color_left_index.back().x + 1) = (color_right.at(color_right_index.back().x + 1));
            color_left_index.back().y += 2;

            color_right.at(color_right_index.back().x) = (color_left.at(color_left_index.back().x + 2));
            color_right_index.back().y++;


            //last Colors
            color_left.push_back(color_right.back());
            color_left.push_back(color_left.at(color_left.size() - 2));
            color_left_index.back().y+=2;
            
            color_right.push_back(color_right.back());
            color_right.push_back(color_left.at(color_left.size() - 3));
            color_right_index.back().y+=2;

            color_right_u.push_back(current_curve_segment-1);
            color_right_u.push_back(current_curve_segment);
            color_left_u.push_back(current_curve_segment-1);
            color_left_u.push_back(current_curve_segment);

        }

        n_colors_left += color_left_index.back().y;
        n_colors_right += color_right_index.back().y;

        set_node = curve->first_node("blur_points_set", 15);
        blur_index.push_back({ n_blurs , 0 });
        current_node = set_node->first_node();

        while (current_node) {
            pushSingle(current_node, blur_index, blur_u, blur, "value");
            current_node = current_node->next_sibling();
        }


        n_blurs += blur_index.back().y;

        if (USE_WEIGHT_INTERPOLATION) {
            set_node = curve->first_node("weight_set", 10);

            weight_index.push_back({ n_weights , 0 });
            current_node = set_node->first_node();

            while (current_node) {
                pushSingle(current_node, weight_index, weight_u, weight, "w");
                current_node = current_node->next_sibling();
            }


            n_weights += weight_index.back().y;

        }


        current_curve++;
    }




    //upload vertex data
    const size_t widths_size = sizeof(float) * vertices.size();
    CUdeviceptr d_widths = 0;
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_widths), widths_size));
    setFloatDevice(reinterpret_cast<float*>(d_widths), vertices.size(), curve_width);




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

    //Upload blur
    const size_t blur_index_size = sizeof(uint2) * blur_index.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.blur_index), blur_index_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur_index),
        blur_index.data(),
        blur_index_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t blur_size = sizeof(float) * blur.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.blur), blur_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur),
        blur.data(),
        blur_size,
        cudaMemcpyHostToDevice,
        stream
    ));

    const size_t blur_u_size = sizeof(float) * blur_u.size();
    CALL_CHECK(cudaMalloc(reinterpret_cast<void**>(&params.blur_u), blur_u_size));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur_u),
        blur_u.data(),
        blur_u_size,
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
    pipeline_compile_options.numPayloadValues = 5;
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

    CALL_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.blur_map),
        sizeof(float) * width * height
    ))

        params.image_width = width;
    params.image_height = height;
    params.frame = 0;
    params.traversable = gas_handle;
    params.zoom_factor = zoom_factor;
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
    std::cout << "Average frame time  : " << total_frame_time / (float)params.frame << " ms" << std::endl;



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
    printf("%d - %s %s \n", level, tag, message);
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
    float u = (std::atof(color_node->first_attribute("globalID", 8)->value()) / 10.0f + (USE_ENDCAPS ? 1.0f : 0.0f));
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
static void push4Points(rapidxml::xml_node<>*& control_node, std::vector<float3>& vertices, int width, int height) {
    float3* controls_xy = new float3[4];

    for (int i = 0; i < 3; i++) {
        controls_xy[i].x = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value()) - (width/2);
        controls_xy[i].y = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) - (height/2);
        controls_xy[i].z = 0;
        control_node = control_node->next_sibling();
    }

    controls_xy[3].x = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1))->value()) - (width / 2);
    controls_xy[3].y = (float)std::atof((control_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1))->value()) - (height / 2);
    controls_xy[3].z = 0;


    correctControlPoints(controls_xy, vertices);

    delete[] controls_xy;
}

static void correctControlPoints(float3* xy_control_points, std::vector<float3>& controls) {
    for (int i = 0; i < 4; i++) {
        controls.push_back({
                xy_control_points[0].x * bspline_correction_matrix[i * 4] + xy_control_points[1].x * bspline_correction_matrix[i * 4 + 1] + xy_control_points[2].x * bspline_correction_matrix[i * 4 + 2] + xy_control_points[3].x * bspline_correction_matrix[i * 4 + 3],
                xy_control_points[0].y * bspline_correction_matrix[i * 4] + xy_control_points[1].y * bspline_correction_matrix[i * 4 + 1] + xy_control_points[2].y * bspline_correction_matrix[i * 4 + 2] + xy_control_points[3].y * bspline_correction_matrix[i * 4 + 3],
                xy_control_points[i].z
            });
    }
}


#pragma inline
static void pushSingle(rapidxml::xml_node<>* node, std::vector<uint2>& ind, std::vector<float>& us, std::vector<float>& target, const char* name) {
    float u = (std::atof(node->first_attribute("globalID", 8)->value()) / 10.0f + (USE_ENDCAPS ? 1.0f : 0.0f));
    target.push_back(std::atof(node->first_attribute(name)->value()));
    us.push_back(u);
    ind.back().y++;
}

#pragma inline
static void getBezierTangent(float t, float3* v, float3& result) {
    result.x = (3 * t * t * v[3].x + v[0].x * (-3 * t * t + 6 * t - 3) + v[1].x * (9 * t * t - 12 * t + 3) + v[2].x * (-9 * t * t + 6 * t));
    result.y = (3 * t * t * v[3].y + v[0].y * (-3 * t * t + 6 * t - 3) + v[1].y * (9 * t * t - 12 * t + 3) + v[2].y * (-9 * t * t + 6 * t));   
}


#pragma inline
static void getEndcapPoints(float3& endpoint, float3& tangent, float3& point1, float3& point2,int endcap_size) {
    //get cos and sin using dot and cross product
    float tangentNormalize = invSqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    float cos = tangent.y * tangentNormalize;
    float sin = tangent.x * tangentNormalize;

    //use rotation matrix on points 1,-1 and 1,1 to get control points
    point2 = { (-cos - sin) * endcap_size  + endpoint.x,  (-sin + cos) * endcap_size + endpoint.y,0 };
    point1 = { (cos - sin ) * endcap_size + endpoint.x, (sin + cos) * endcap_size + endpoint.y, 0 };
}

//Always wanted to use this taken from wikipedia fast inverse square root
float invSqrt(float number) {
    union {
        float f;
        uint32_t i;
    } conv;

    float x2;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    conv.f = number;
    conv.i = 0x5f3759df - (conv.i >> 1);
    conv.f = conv.f * (threehalfs - (x2 * conv.f * conv.f));
    return conv.f;
}