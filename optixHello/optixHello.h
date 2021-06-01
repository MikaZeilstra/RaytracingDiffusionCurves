#pragma once
#include <rapidxml/rapidxml_utils.hpp>

#include <iostream>

#ifndef CALL_CHECK
#define CALL_CHECK(call) \
    if(call != 0){          \
        std::cerr << "Error in Optix/CUDA call with number " << call << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        throw std::exception(); \
    }
#endif // !


static void printUChar4(uchar4* uchar);
static void logFunction(unsigned int level, const char* tag, const char* message, void*);
static bool loadSource(std::string& dest, const std::string& loc);
static void pushColor(rapidxml::xml_node<>* color_node, std::vector<uint2>& ind, std::vector<float>& color_u, std::vector<float3>& color);
static void push4Points(rapidxml::xml_node<>*&control_node, std::vector<float3>&vertices, int width, int height);
static void correctControlPoints(float3* xy_control_points, std::vector<float3>&controls);
static void pushSingle(rapidxml::xml_node<>*node, std::vector<uint2>&ind, std::vector<float>&us, std::vector<float>&target, const char* name);
static void getBezierTangent(float t, float3 * v, float3 & result);
static void getEndcapPoints(float3 & endpoint, float3 & tangent, float3 & point1, float3 & point2);
float invSqrt(float number);