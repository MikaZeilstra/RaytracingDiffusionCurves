
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



#define FILE_PREFIX "screenshot-"
#define ZOOM_STEP 1.5

bool l_mouse_down = false;
double prev_x = 0;
double prev_y = 0;

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {

	//PrintScreen
	if (key == GLFW_KEY_F11 && action == GLFW_PRESS) {

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

		for (unsigned int i = 0; i < params->image_height; i++) {
			for (unsigned int j = 0; j < params->image_width; j++) {
				
				final_image[ind++] = image[i * params->image_width + j].x * 255;
				final_image[ind++] = image[i * params->image_width + j].y * 255;
				final_image[ind++] = image[i * params->image_width + j].z * 255;
				final_image[ind++] = image[i * params->image_width + j].w * 255;
			}
		}
		
		

		time_t t = std::time(nullptr);
		tm tm = *std::localtime(&t);

		

		std::ostringstream oss;
		oss << FILE_PREFIX << std::put_time(&tm, "%d-%m-%Y-%H-%M-%S") << ".png";

		stbi_flip_vertically_on_write(USE_DIFFUSION_CURVE_SAVE);

		//stbi_write_png(oss.str().c_str(), params->image_width, params->image_height, 4, final_image, params->image_width * 4);
		
		stbi_write_jpg(oss.str().c_str(), params->image_width, params->image_height, 4, final_image, params->image_width * 4);

		free(image);
		free(final_image);


	}

}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	Params* params = reinterpret_cast<Params*>(glfwGetWindowUserPointer(window));
    params->zoom_factor *= powf((ZOOM_STEP), -yoffset);
}


void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{

	if (l_mouse_down)
	{
		Params* params = reinterpret_cast<Params*>(glfwGetWindowUserPointer(window));
		params->offset_x -= (xpos - prev_x) * params->zoom_factor;
		params->offset_y -= (ypos - prev_y) * params->zoom_factor;

		prev_x = xpos;
		prev_y = ypos;
	}
    
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		l_mouse_down = true;
		glfwGetCursorPos(window, &prev_x, &prev_y);
	}
	else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
		l_mouse_down = false;
	}
		
}
