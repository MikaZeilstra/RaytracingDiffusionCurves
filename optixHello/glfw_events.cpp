/*
   Copyright 2021 Mika Zeilstra

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

//Setup STBI variables
#pragma once
#define STB_IMAGE_IMPLEMENTATION
//#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "glfw_events.h"
#include "params.h"
#include "optixHello.h"

#include <tinygltf/stb_image_write.h>

#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <iomanip>
#include <sstream>


//File prefix and how much each scrollwheel tick scrolls
#define FILE_PREFIX "screenshot-"
#define ZOOM_STEP 1.5

//Global variable to hold if the mouse button is down and the previous x and y position of the mouse
bool l_mouse_down = false;
double prev_x = 0;
double prev_y = 0;

//Functions defined in helperKernels.cu to update the optical flow on zooming and panning
extern "C" __host__ void zoomImageFlow(float2 * flow, float zoom, float zoom_factor, int width, int height, CUstream stream);
extern "C" __host__ void translateImageFlow(float2 * flow, float2 translation, int width, int height, CUstream stream);

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	//F11 makes a printscreen
	if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {

		//Get the image
		Params* params = reinterpret_cast<Params*>(glfwGetWindowUserPointer(window));

		float4* image = reinterpret_cast<float4*>(malloc(sizeof(float4) * params->image_width * params->image_height));
		unsigned char* final_image = reinterpret_cast<unsigned char*>(malloc(sizeof(unsigned char) * params->image_width * params->image_height * 4));

		CUstream stream;

		int ind = 0;
		CALL_CHECK(cudaMemcpy(
			image,
			params->image,
			sizeof(float4) * params->image_width * params->image_height,
			cudaMemcpyDeviceToHost
		));
		cudaDeviceSynchronize();

		//Convert the image from [0,1] to [0,255]
		for (unsigned int i = 0; i < params->image_height; i++) {
			for (unsigned int j = 0; j < params->image_width; j++) {

				final_image[ind++] = std::min(image[i * params->image_width + j].x * 255, 255.0f);
				final_image[ind++] = std::min(image[i * params->image_width + j].y * 255, 255.0f);
				final_image[ind++] = std::min(image[i * params->image_width + j].z * 255, 255.0f);
				final_image[ind++] = std::min(image[i * params->image_width + j].w * 255, 255.0f);
			}
		}



		//Create timestamp and filename
		time_t t = std::time(nullptr);
		tm tm = *std::localtime(&t);
		std::ostringstream oss;
		oss << FILE_PREFIX << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S") << ".jpg";

		//Write image
		stbi_flip_vertically_on_write(USE_DIFFUSION_CURVE_SAVE);

		stbi_write_jpg(oss.str().c_str(), params->image_width, params->image_height, 4, final_image, params->image_width * 4);

		free(image);
		free(final_image);


	}

}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	//Zoom the image and update optical flow
	Params* params = reinterpret_cast<Params*>(glfwGetWindowUserPointer(window));
	zoomImageFlow(params->image_flow, params->zoom_factor, powf((ZOOM_STEP), -yoffset),params->image_width, params->image_height,params->stream);
    params->zoom_factor *= powf((ZOOM_STEP), -yoffset);
	
}


void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{

	//If the mouse button is down we allow the user to pan the image by dragging
	if (l_mouse_down)
	{
		Params* params = reinterpret_cast<Params*>(glfwGetWindowUserPointer(window));
		params->offset_x -= (xpos - prev_x) * params->zoom_factor;
		params->offset_y -= (ypos - prev_y) * params->zoom_factor;

		prev_x = xpos;
		prev_y = ypos;

		translateImageFlow(params->image_flow, { (float)-(xpos - prev_x),(float) -(ypos - prev_y) }, params->image_width, params->image_height, params->stream);
	}    
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	//Set and reset the l_mouse_down variable
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		l_mouse_down = true;
		glfwGetCursorPos(window, &prev_x, &prev_y);
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		l_mouse_down = false;
	}
		
}
