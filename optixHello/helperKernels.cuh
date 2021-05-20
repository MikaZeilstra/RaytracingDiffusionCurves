#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


extern "C" __global__ void setFloatKernel(float* dest, int n, float src);
extern "C" __host__ void setFloatDevice(float* dest, int n, float src);