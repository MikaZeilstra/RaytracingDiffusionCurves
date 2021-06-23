#pragma once
#define GLFW_INCLUDE_NONE

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>


#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <curand_kernel.h>


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

//Functions defined in helperkernels.cu
extern "C" __host__ void setFloatDevice(float* dest, unsigned int n, float src, CUstream params);
extern "C" __host__ void gaussianBlur(float4 * dest, float4 * source, float* sigma, int width, int height, CUstream params);
extern "C" __host__ void setupCurand(curandState_t * states, int width, int height, CUstream params);
extern "C" __host__ void zeroImageFlow(float2 * flow, int width, int height, CUstream stream);

//CMake puts DeviceCode.cu PTX here to use for optix
extern "C" char embedded_ptx_code[];


//Template for STB records
template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>     RayGenSbtRecord;
typedef SbtRecord<MissData>       MissSbtRecord;
typedef SbtRecord<HitGroupData>   HitGroupSbtRecord;

//PBO opject
GLuint pbo;

//matrix used to convert bezier control points to b-spline control points
constexpr float bspline_correction_matrix[] = { 6,-7,2,0,
                                            0,2,-1,0,
                                            0, -1,2,0,
                                            0,2,-7,6 };

int main(int argc, char* argv[]) {
    //Check if the xml and raycount are set
    if (argc < 3) {
        std::cout << "Please provide a path to a diffusion curve xml and the number of rays per pixel" << std::endl;
        return 1;    
    }

    //Set the starting point for the zoom and offset
    float zoom_factor = 1  ;
    float offset_x = 0;
    float offset_y = 0;

    //Set several constants for the program
    const float default_weight_degree = 0.5;
    const float curve_width = 1e-3f;
    const float endcap_size = 8;
    //What fraction of the denoised image are we using? 1 for the denoised image 0 for no denoising and in between a mix
    const float corrected_image_mix = 1;

    //Get the argument for ray count and xml
    const int number_of_rays = std::atoi(argv[2]);
    const std::string file_name = "/" + std::string(argv[1]);

    //Setup timer for setup
    auto start_time = std::chrono::high_resolution_clock::now();

    //Load the xml file
    rapidxml::file<> xmlFile((std::filesystem::current_path().generic_string() + file_name).c_str());
    rapidxml::xml_document<> doc;
    doc.parse<0>(xmlFile.data());
    rapidxml::xml_node<>* curve_set = doc.first_node();
    rapidxml::xml_node<>* set_node;
    rapidxml::xml_node<>* current_node;

    //Load image width and height
    int width = std::atoi(curve_set->first_attribute("image_width")->value());
    int height = std::atoi(curve_set->first_attribute("image_height")->value());

    Params params = {};
    GLFWwindow* window;

    //Setup window
    glfwInit();
    window = glfwCreateWindow(width, height, "My first window", NULL, NULL);
    glfwMakeContextCurrent(window);

    //Bind the param point to the window so we can retrieve it.
    glfwSetWindowUserPointer(window, &params);

    //Add callbacks for interaction
    glfwSetKeyCallback(window, key_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, mouse_cursor_callback);

    //Setup OpenGL
    gladLoadGL();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glfwSwapBuffers(window);

    //Create PBO for output
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4 * sizeof(GLfloat) * width * height, NULL, GL_DYNAMIC_DRAW);

    //Register PBO in CUDA for further usage
    cudaGraphicsResource* d_pbo;
    cudaGraphicsGLRegisterBuffer(&d_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    //Create CUDA/Optix Context
    cudaFree(0);
    CUcontext cuCtx = 0;
    CALL_CHECK(optixInit());

    char log[2048];
    size_t sizeof_log = sizeof(log);

    OptixDeviceContextOptions Context_options = {};
    Context_options.logCallbackFunction = &logFunction;
    Context_options.logCallbackLevel = 4;

    OptixDeviceContext context = nullptr;

    CALL_CHECK(cudaStreamCreate(&params.stream));
    optixDeviceContextCreate(cuCtx, &Context_options, &context);

    //Setup all host side variable for the curves
    std::vector<float3> vertices = {};
    std::vector<unsigned int> curve_map = {};
    std::vector<unsigned int> curve_map_inverse = {};
    std::vector<unsigned int> curve_index = {};

    std::vector<int> curve_connect = {};

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

    std::vector<uint2> weight_degree_index = {};
    std::vector<float> weight_degree = {};
    std::vector<float> weight_degree_u = {};

    int current_segment = 0;
    int current_curve = 0;
    int current_curve_segment = 0;

    unsigned int n_colors_left = 0;
    unsigned int n_colors_right = 0;
    unsigned int n_weights = 0;
    unsigned int n_weights_degree = 0;
    unsigned int n_blurs = 0;
    unsigned int n_segments = 0;


    //Load the curves
    for (rapidxml::xml_node<>* curve = curve_set->first_node(); curve; curve = curve->next_sibling()) {

        //Read control points
        current_curve_segment = 0;
        set_node = curve->first_node("control_points_set", 18);

        current_node = set_node->first_node();

        //Check if we need to use an endcap
        bool use_endcap = (curve->first_attribute("use_endcap") ? curve->first_attribute("use_endcap")->value() : "") == std::string("true");

        //Setup portal if neccesary
        curve_connect.push_back(curve->first_attribute("connects") ? std::stoi(curve->first_attribute("connects")->value()) : -1);

        curve_map_inverse.push_back(n_segments);

        //Setup andcap for start of curve
        if (use_endcap) {
            //Setup temporary variables for the endcap spline
            float3* endcap = new float3[4];
            float3* first_curve = new float3[4];

            //Read the first control point of the curve and save it as end and beginning points
            rapidxml::xml_node<>* endcap_node = current_node;
            endcap[0] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width/2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                });
            endcap[3] = endcap[0];

            //Get the control points for the first spline of the curve
            for (int i = 0; i < 4; i++) {
                first_curve[i] = float3({
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "y" : "x", 1)->value()) - (width / 2),
                    (float)std::atof(endcap_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "x" : "y", 1)->value()) - (height / 2),
                    0
                    });

                endcap_node = endcap_node->next_sibling();
            }

            //Calculate the tangent of the curve
            float3 tan = {};
            getBezierTangent(1e-3, first_curve, tan);

            //Reverse the tangent
            tan = { -tan.x, -tan.y, tan.z };

            //Calculate the endcap points and transform the bezier curve points to b-spline points
            getEndcapPoints(endcap[0], tan, endcap[1], endcap[2],endcap_size);
            correctControlPoints(endcap, vertices);

            //cleanup temp variables
            delete[] endcap;
            delete[] first_curve;

            //Update the host variables to the current state
            segmentIndices.push_back(current_segment);
            current_segment += 4;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }
        
        //Read all splines belonging to this curve
        while (current_node->next_sibling()) {
            //Insert the 4 vertexes of the spline
            push4Points(current_node, vertices,width,height);

            //Update the host variables to the current state
            segmentIndices.push_back(current_segment);
            current_segment += 4;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }


        //Same as previous endcap creation only the tangent is not reversed
        if (use_endcap) {
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
            getBezierTangent(1- 1e-3, first_curve, tan);

            tan = { tan.x, tan.y, tan.z };

            getEndcapPoints(endcap[0], tan, endcap[1], endcap[2],endcap_size);
            correctControlPoints(endcap, vertices);

            delete[] endcap;
            delete[] first_curve;

            segmentIndices.push_back(current_segment);
            current_segment += 4;
            curve_map.push_back(current_curve);
            curve_index.push_back(current_curve_segment++);
        }


        //Set the current node to the left color nodes
        set_node = curve->first_node("left_colors_set", 15);
        color_left_index.push_back({ n_colors_left ,0 });
        current_node = set_node->first_node();

        //Reserve space for endcap colors
        if (use_endcap) {
            color_right.push_back({ 0,0,0 });
            color_right.push_back({ 0,0,0 });
            color_left.push_back({ 0,0,0 });
            color_left.push_back({ 0,0,0 });

            color_right_u.push_back(0);
            color_right_u.push_back(1);
            color_left_u.push_back(0);
            color_left_u.push_back(1);
        }


        //Read all the left colors
        while (current_node) {
            pushColor(current_node, color_left_index, color_left_u, color_left,use_endcap);
            current_node = current_node->next_sibling();
        }   


        //set the current node to the right color nodes
        set_node = curve->first_node("right_colors_set", 16);
        color_right_index.push_back({ n_colors_right ,0 });
        current_node = set_node->first_node();

        //Read all the right colors
        while (current_node) {
            pushColor(current_node, color_right_index, color_right_u, color_right,use_endcap);
            current_node = current_node->next_sibling();
        }

        //Make sure there is a color value for the last parameter value of the curve if we are using a diffusion curve save
        if (USE_DIFFUSION_CURVE_SAVE) {
            color_right.push_back(color_right.back());
            color_right_index.back().y++;
            color_right_u.push_back(current_curve_segment - (use_endcap ? 1 : 0));

            color_left.push_back(color_left.back());
            color_left_index.back().y++;
            color_left_u.push_back(current_curve_segment - (use_endcap ? 1 : 0));
        }

        
        //Setup Colors for endcaps
        if (use_endcap) {
            //First Colors
            color_left.at(color_left_index.back().x) = (color_left.at(color_left_index.back().x + 2));
            color_left.at(color_left_index.back().x + 1) = (color_right.at(color_right_index.back().x + 2));
            color_left_index.back().y += 2;

            color_right.at(color_right_index.back().x) = (color_left.at(color_left_index.back().x + 2));
            color_right.at(color_right_index.back().x + 1) = (color_right.at(color_right_index.back().x + 2));
            color_right_index.back().y += 2;


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


        //Read blur
        set_node = curve->first_node("blur_points_set", 15);
        blur_index.push_back({ n_blurs , 0 });

        current_node = set_node->first_node();

        if (use_endcap) {
            blur.push_back(0);
            blur_u.push_back(0);
            blur_index.back().y++;
        }

        while (current_node) {
            pushSingle(current_node, blur_index, blur_u, blur, "value",use_endcap);
            current_node = current_node->next_sibling();
        }

        if (use_endcap) {
            blur.at(blur_index.back().x) = blur.at(blur_index.back().x + 1);
            blur.push_back(blur.back());
            blur_u.push_back(current_curve_segment);
            blur_index.back().y++;
        }

        n_blurs += blur_index.back().y;


        //Read weight multiplier
        set_node = curve->first_node("weight_set", 10);
        weight_index.push_back({ n_weights , 0 });

        if (set_node) {
            if (use_endcap) {
                weight.push_back(0);
                weight_u.push_back(0);
                weight_index.back().y++;
            }


            current_node = set_node->first_node();
            while (current_node) {
                pushSingle(current_node, weight_index, weight_u, weight, "w",use_endcap);
                current_node = current_node->next_sibling();
            }

            if (use_endcap) {
                weight.at(weight_index.back().x) = weight.at(weight_index.back().x + 1);
                weight.push_back(weight.back());
                weight_u.push_back(current_curve_segment);
                weight_index.back().y++;
            }

        }
        else {
            weight.push_back(1);
            weight.push_back(1);
            weight_u.push_back(0);
            weight_u.push_back(current_curve_segment);
            weight_index.back().y += 2;
        }

        n_weights += weight_index.back().y;

        
        //Read weight base
        set_node = curve->first_node("weight_degree_set");
        weight_degree_index.push_back({ n_weights_degree , 0 });

        if (set_node) {
            if (use_endcap) {
                weight_degree.push_back(default_weight_degree);
                weight_degree_u.push_back(0);
                weight_degree_index.back().y++;
            }


            current_node = set_node->first_node();
            while (current_node) {
                pushSingle(current_node, weight_degree_index, weight_degree_u, weight_degree, "w", use_endcap);
                current_node = current_node->next_sibling();
            }

            if (use_endcap) {
                weight_degree.at(weight_degree_index.back().x) = weight_degree.at(weight_degree_index.back().x + 1);
                weight_degree.push_back(weight_degree.back());
                weight_degree_u.push_back(current_curve_segment);
                weight_degree_index.back().y++;
            }

        }
        else {
            weight_degree.push_back(default_weight_degree);
            weight_degree.push_back(default_weight_degree);
            weight_degree_u.push_back(0);
            weight_degree_u.push_back(current_curve_segment);
            weight_degree_index.back().y += 2;
        }

        n_weights_degree += weight_degree_index.back().y;

        current_curve++;
        n_segments += current_curve_segment;
    }
    

    //Setup PRNG states
    const size_t curand_size = sizeof(curandState_t) * width * height;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.curandStates), curand_size, params.stream));
    setupCurand(params.curandStates, width, height,params.stream);

    //Setup space for temporal stability
    const size_t image_size = sizeof(float4) * width * height;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.prev_image), image_size,params.stream));
    
    const size_t flow_size = sizeof(float2) * width * height;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.image_flow), flow_size, params.stream));


    //Setup the constant array with the widths of each curve 
    const size_t widths_size = sizeof(float) * vertices.size();
    CUdeviceptr d_widths = 0;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_widths), widths_size,params.stream));
    setFloatDevice(reinterpret_cast<float*>(d_widths), vertices.size(), curve_width, params.stream);

    //upload spline control points data
    const size_t vertices_size = sizeof(float3) * vertices.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.vertices), vertices_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.vertices),
        vertices.data(),
        vertices_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t segmentIndices_size = sizeof(unsigned int) * segmentIndices.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.segmentIndices), segmentIndices_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.segmentIndices),
        segmentIndices.data(),
        segmentIndices_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload curve -> spline map data
    const size_t curve_map_size = sizeof(unsigned int) * curve_map.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.curve_map), curve_map_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_map),
        curve_map.data(),
        curve_map_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload spline -> spline in curve data
    const size_t curve_index_size = sizeof(unsigned int) * curve_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.curve_index), curve_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_index),
        curve_index.data(),
        curve_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload portal connections
    const size_t curve_connect_size = sizeof(int) * curve_connect.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.curve_connect), curve_connect_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_connect),
        curve_connect.data(),
        curve_connect_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload Curve -> first spline map
    const size_t curve_map_inverse_size = sizeof(int) * curve_map_inverse.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.curve_map_inverse), curve_map_inverse_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.curve_map_inverse),
        curve_map_inverse.data(),
        curve_map_inverse_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload color data
    const size_t color_left_index_size = sizeof(uint2) * color_left_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_left_index), color_left_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left_index),
        color_left_index.data(),
        color_left_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));


    const size_t color_left_size = sizeof(float3) * color_left.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_left), color_left_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left),
        color_left.data(),
        color_left_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t color_left_u_size = sizeof(float) * color_left_u.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_left_u), color_left_u_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_left_u),
        color_left_u.data(),
        color_left_u_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t color_right_index_size = sizeof(uint2) * color_right_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_right_index), color_right_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right_index),
        color_right_index.data(),
        color_right_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t color_right_size = sizeof(float3) * color_right.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_right), color_right_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right),
        color_right.data(),
        color_right_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t color_right_u_size = sizeof(float) * color_right_u.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.color_right_u), color_right_u_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.color_right_u),
        color_right_u.data(),
        color_right_u_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload blur
    const size_t blur_index_size = sizeof(uint2) * blur_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.blur_index), blur_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur_index),
        blur_index.data(),
        blur_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t blur_size = sizeof(float) * blur.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.blur), blur_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur),
        blur.data(),
        blur_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t blur_u_size = sizeof(float) * blur_u.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.blur_u), blur_u_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.blur_u),
        blur_u.data(),
        blur_u_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    CALL_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&params.blur_map),
        sizeof(float)* width* height,
        params.stream
    ))


    //Upload Weights
    const size_t weight_index_size = sizeof(uint2) * weight_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight_index), weight_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight_index),
        weight_index.data(),
        weight_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t weight_size = sizeof(float) * weight.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight), weight_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight),
        weight.data(),
        weight_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t weight_u_size = sizeof(float) * weight_u.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight_u), weight_u_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight_u),
        weight_u.data(),
        weight_u_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Upload Weights degrees
    const size_t weight_degree_index_size = sizeof(uint2) * weight_degree_index.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight_degree_index), weight_degree_index_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight_degree_index),
        weight_degree_index.data(),
        weight_degree_index_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t weight_degree_size = sizeof(float) * weight_degree.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight_degree), weight_degree_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight_degree),
        weight_degree.data(),
        weight_degree_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    const size_t weight_degree_u_size = sizeof(float) * weight_degree_u.size();
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&params.weight_degree_u), weight_degree_u_size, params.stream));
    CALL_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(params.weight_degree_u),
        weight_degree_u.data(),
        weight_degree_u_size,
        cudaMemcpyHostToDevice,
        params.stream
    ));

    //Setup Acceleration structure primitive input
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
    CALL_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes,
        params.stream
    ));
    CALL_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes,
        params.stream
    ));


    CALL_CHECK(optixAccelBuild(
        context,
        params.stream,
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

    //Free the scrap memory used for acceleration structure
    CALL_CHECK(cudaFreeAsync(reinterpret_cast<void*>(d_temp_buffer_gas), params.stream));

    //Setup Modules
    OptixModule module = nullptr;

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;

    //Setup pipeline compile options
    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 7;
    pipeline_compile_options.numAttributeValues = 2;

    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

    const std::string ptxCode = embedded_ptx_code;

    //Create module using PTX provided from DeviceCode by CMAKE
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


    //Setup intersection program
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
    pipeline_link_options.maxTraceDepth = MAX_TRACE_DEPTH;
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
    cudaMallocAsync(reinterpret_cast<void**>(&raygen_record), raygen_record_size, params.stream);
    RayGenSbtRecord rg_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    cudaMemcpyAsync(
        reinterpret_cast<void*>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice,
        params.stream
    );


    CUdeviceptr miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord);
    cudaMallocAsync(reinterpret_cast<void**>(&miss_record), miss_record_size, params.stream);
    MissSbtRecord miss_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(miss_prog_group, &miss_sbt));
    cudaMemcpyAsync(
        reinterpret_cast<void*>(miss_record),
        &miss_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice,
        params.stream
    );

    CUdeviceptr hg_record;
    const size_t  hg_record_size = sizeof(HitGroupSbtRecord);
    cudaMallocAsync(reinterpret_cast<void**>(&hg_record), hg_record_size, params.stream);
    HitGroupSbtRecord hg_sbt;
    CALL_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
    cudaMemcpyAsync(
        reinterpret_cast<void*>(hg_record),
        &hg_sbt,
        hg_record_size,
        cudaMemcpyHostToDevice,
        params.stream
    );

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.hitgroupRecordBase = hg_record;
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);

    //setup params.image as PBO
    cudaGraphicsMapResources(1, &d_pbo, NULL);
    cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&(params.image)), NULL, d_pbo);


    //Setup denoiser
    OptixDenoiser denoiser;
    OptixImage2D denoiser_input = {};
    OptixImage2D denoiser_prev_output = {};
    OptixImage2D denoiser_output = {};
    OptixImage2D denoiser_flow = {};
    OptixDenoiserSizes denoiser_sizes = {};
    OptixDenoiserGuideLayer denoiser_guide = {};
    OptixDenoiserLayer denoiser_layer = {};
    OptixDenoiserParams denoiser_params = {};

    CUdeviceptr denoiser_scratch = 0;
    CUdeviceptr denoiser_state = 0;
    CUdeviceptr denoiser_output_image;
    CUdeviceptr d_hdr;

    if (USE_DENOISER) {
        //Create denoiser if we are going to use it
        OptixDenoiserOptions denoiser_options = {};
        denoiser_options.guideAlbedo = 0;
        denoiser_options.guideNormal = 0;

        CALL_CHECK(optixDenoiserCreate(
            context,
            OPTIX_DENOISER_MODEL_KIND_TEMPORAL,
            &denoiser_options,
            &denoiser
        ));

        

        CALL_CHECK(optixDenoiserComputeMemoryResources(
            denoiser,
            width,
            height,
            &denoiser_sizes
        ));

        cudaMallocAsync(reinterpret_cast<void**>(&denoiser_state), denoiser_sizes.stateSizeInBytes, params.stream);

        cudaMallocAsync(reinterpret_cast<void**>(&denoiser_scratch),
            std::max(denoiser_sizes.withOverlapScratchSizeInBytes,
                denoiser_sizes.withoutOverlapScratchSizeInBytes),
            params.stream
        );

        CALL_CHECK(optixDenoiserSetup(
            denoiser,
            params.stream,
            width,
            height,
            denoiser_state,
            denoiser_sizes.stateSizeInBytes,
            denoiser_scratch,
            std::max(denoiser_sizes.withOverlapScratchSizeInBytes,
                denoiser_sizes.withoutOverlapScratchSizeInBytes)
        ));

       
        //Create input of the denoiser consiting of the current frame the previous frame and the optical flow between them
        denoiser_input.data = reinterpret_cast<CUdeviceptr>(params.image);
        denoiser_input.width = width;
        denoiser_input.height = height;
        denoiser_input.rowStrideInBytes = width * sizeof(float4);
        denoiser_input.pixelStrideInBytes = sizeof(float4);
        denoiser_input.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        denoiser_prev_output.data = reinterpret_cast<CUdeviceptr>(params.prev_image);
        denoiser_prev_output.width = width;
        denoiser_prev_output.height = height;
        denoiser_prev_output.rowStrideInBytes = width * sizeof(float4);
        denoiser_prev_output.pixelStrideInBytes = sizeof(float4);
        denoiser_prev_output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        denoiser_output.data = denoiser_output_image;
        denoiser_output.width = width;
        denoiser_output.height = height;
        denoiser_output.rowStrideInBytes = width * sizeof(float4);
        denoiser_output.pixelStrideInBytes = sizeof(float4);
        denoiser_output.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        denoiser_flow.data = reinterpret_cast<CUdeviceptr>(params.image_flow);
        denoiser_flow.width = width;
        denoiser_flow.height = height;
        denoiser_flow.rowStrideInBytes = width * sizeof(float2);
        denoiser_flow.pixelStrideInBytes = sizeof(float2);
        denoiser_flow.format = OPTIX_PIXEL_FORMAT_FLOAT2;

        //Create space for the output of the denoiser
        CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&denoiser_output_image), width * height * sizeof(float4), params.stream));
        denoiser_layer.input = denoiser_input;
        denoiser_layer.previousOutput = denoiser_prev_output;
        denoiser_layer.output = denoiser_output;
        denoiser_guide.flow = denoiser_flow;
        
        //Setup denoiser params
        cudaMallocAsync(reinterpret_cast<void**>(&d_hdr), sizeof(float), params.stream);
        denoiser_params.blendFactor = 1-corrected_image_mix;
        denoiser_params.denoiseAlpha = 0;
        denoiser_params.hdrIntensity = d_hdr;
    }


    //Setup Parameter
    params.image_width = width;
    params.image_height = height;
    params.frame = 0;
    params.traversable = gas_handle;
    params.zoom_factor = zoom_factor;
    params.offset_x = offset_x;
    params.offset_y = offset_y;
    params.number_of_rays_per_pixel = number_of_rays;

    //Upload params for first frame
    CUdeviceptr d_param;
    CALL_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_param), sizeof(Params),params.stream));

    //Everything has been uploaded and setup by now and we need to make sure all the Async tasks are done
    CALL_CHECK(cudaDeviceSynchronize());
    CALL_CHECK(cudaStreamSynchronize(params.stream));

    //End the timing of the setup
    auto setup_duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time);
    std::cout << "Setup took : " << setup_duration.count() << " ms" << std::endl;

    //Setup the counter for the total frame time
    long long  total_frame_time = 0;

    //Main Loop running while window is not closed
    while (!glfwWindowShouldClose(window)) {
        //Start frame timer
        start_time = std::chrono::high_resolution_clock::now();

        //Procces events from previous frane
        glfwPollEvents();

        //Get the pointer to the pbo into params
        cudaGraphicsMapResources(1, &d_pbo, NULL);
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&(params.image)), NULL, d_pbo);


        //Upload params to device
        CALL_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(d_param),
            &params, sizeof(Params),
            cudaMemcpyHostToDevice,
            params.stream
        ));

        //Launch pipeline to get the image
        CALL_CHECK(optixLaunch(pipeline, params.stream, d_param, sizeof(Params), &sbt, width, height, 1));

        if (USE_DENOISER) {
            //Apply denoiser if required
            CALL_CHECK(optixDenoiserComputeIntensity(
                denoiser,
                params.stream,
                &denoiser_input,
                d_hdr,
                denoiser_scratch,
                std::max(denoiser_sizes.withOverlapScratchSizeInBytes,
                    denoiser_sizes.withoutOverlapScratchSizeInBytes)));

            denoiser_params.hdrIntensity = d_hdr;

            CALL_CHECK(optixDenoiserInvoke(
                denoiser,
                params.stream,
                &denoiser_params,
                denoiser_state,
                denoiser_sizes.stateSizeInBytes,
                &denoiser_guide,
                &denoiser_layer,
                1,
                0,
                0,
                denoiser_scratch,
                std::max(denoiser_sizes.withOverlapScratchSizeInBytes,
                    denoiser_sizes.withoutOverlapScratchSizeInBytes)
            ));

            //Copy the output to the final image location
            cudaMemcpyAsync(
                params.prev_image,
                reinterpret_cast<void*>(denoiser_output_image),
                width* height * sizeof(float4),
                cudaMemcpyDeviceToDevice,
                params.stream
            );

            //Copy the output to the previous output for the next frame
            cudaMemcpyAsync(
                params.image,
                reinterpret_cast<void*>(denoiser_output_image),
                width* height * sizeof(float4),
                cudaMemcpyDeviceToDevice,
                params.stream
            );

            //Zero the image optical flow for the next image
            zeroImageFlow(params.image_flow, params.image_width, params.image_height, params.stream);
        }

        //Apply blur if necessary
        if (USE_BLUR) {
            gaussianBlur(params.image, params.image, params.blur_map, params.image_width, params.image_height,params.stream);
        }

        
        //Sync all things
        CALL_CHECK(cudaStreamSynchronize(params.stream));

        //Display image
        glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
        cudaGraphicsUnmapResources(1, &d_pbo, NULL);
        glfwSwapBuffers(window);

        //update frame number display
        params.frame++;
        printf("\rframe : %d", params.frame);

        

        //Add frame time to total
        total_frame_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
    };

    //When the window is closed print the average frame time
    std::cout << std::endl;
    std::cout << "Average frame time  : " << total_frame_time / (float)params.frame << " ms" << std::endl;



    //Cleanup WARNING I did not pay enough attetion to cleanup all device side memory when using this program as a standalone this should not be a problem however if you want to intergrate it into something you need to look into this
    cudaGLUnregisterBufferObject(pbo);
    cudaGraphicsUnregisterResource(d_pbo);
    glDeleteBuffers(1, &pbo);

    glfwDestroyWindow(window);
    glfwTerminate();

    //CALL_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
    CALL_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));
    
    if (USE_DENOISER) {
        CALL_CHECK(optixDenoiserDestroy(denoiser));
    }
    
    CALL_CHECK(optixPipelineDestroy(pipeline));
    CALL_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    CALL_CHECK(optixProgramGroupDestroy(miss_prog_group));
    CALL_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    CALL_CHECK(optixModuleDestroy(module));

    CALL_CHECK(optixDeviceContextDestroy(context));

    cudaDeviceReset();
}


//Log function used for the log when creating denoiser and modules
static void logFunction(unsigned int level, const char* tag, const char* message, void*) {
    printf("%d - %s %s \n", level, tag, message);
}

//Read and push a color control point B and R color channels switched if load diffusion curve xml
static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color, bool use_endcap) {
    float u = (std::atof(color_node->first_attribute("globalID", 8)->value()) / 10.0f + (use_endcap ? 1.0f : 0.0f));
    color.push_back({
        std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "B" : "R",1)->value()) / 255.0f,
        std::atoi(color_node->first_attribute("G",1)->value()) / 255.0f,
        std::atoi(color_node->first_attribute(USE_DIFFUSION_CURVE_SAVE ? "R" : "B",1)->value()) / 255.0f
        });
    color_u.push_back(u);
    ind.back().y++;
}

//Read and push a spline control point and convert it to b-spline Switch x y if loading diffusion curve xml
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

//Does a matrix multiplication to transform a set of control points from bezier to b-spline
static void correctControlPoints(float3* xy_control_points, std::vector<float3>& controls) {
    for (int i = 0; i < 4; i++) {
        controls.push_back({
                xy_control_points[0].x * bspline_correction_matrix[i * 4] + xy_control_points[1].x * bspline_correction_matrix[i * 4 + 1] + xy_control_points[2].x * bspline_correction_matrix[i * 4 + 2] + xy_control_points[3].x * bspline_correction_matrix[i * 4 + 3],
                xy_control_points[0].y * bspline_correction_matrix[i * 4] + xy_control_points[1].y * bspline_correction_matrix[i * 4 + 1] + xy_control_points[2].y * bspline_correction_matrix[i * 4 + 2] + xy_control_points[3].y * bspline_correction_matrix[i * 4 + 3],
                xy_control_points[i].z
            });
    }
}

//Reads and pushes a single value to the designated location
static void pushSingle(rapidxml::xml_node<>* node, std::vector<uint2>& ind, std::vector<float>& us, std::vector<float>& target, const char* name,bool use_endcap) {
    float u = (std::atof(node->first_attribute("globalID", 8)->value()) / 10.0f + (use_endcap ? 1.0f : 0.0f));
    target.push_back(std::atof(node->first_attribute(name)->value()));
    us.push_back(u);
    ind.back().y++;
}

//Calculates the tangent of a bezier curve
static void getBezierTangent(float t, float3* v, float3& result) {
    result.x = (3 * t * t * v[3].x + v[0].x * (-3 * t * t + 6 * t - 3) + v[1].x * (9 * t * t - 12 * t + 3) + v[2].x * (-9 * t * t + 6 * t));
    result.y = (3 * t * t * v[3].y + v[0].y * (-3 * t * t + 6 * t - 3) + v[1].y * (9 * t * t - 12 * t + 3) + v[2].y * (-9 * t * t + 6 * t));   
}

//Sets up the middle two endcap control points to the correct translated points
static void getEndcapPoints(float3& endpoint, float3& tangent, float3& point1, float3& point2,int endcap_size) {
    //get cos and sin using dot and cross product
    float tangentNormalize = invSqrt(tangent.x * tangent.x + tangent.y * tangent.y);
    float cos = tangent.y * tangentNormalize;
    float sin = -tangent.x * tangentNormalize;

    //use rotation matrix on points -1,1 and 1,1 to get control points
    point1 = { (-cos - sin) * endcap_size  + endpoint.x,  (-sin + cos) * endcap_size + endpoint.y,0 };
    point2 = { (cos - sin ) * endcap_size + endpoint.x, (sin + cos) * endcap_size + endpoint.y, 0 };
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