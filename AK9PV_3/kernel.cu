#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <omp.h>
#include "pngio.h"

using namespace std;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN( value ) {							\
	cudaError_t err = value;									\
	if( err != cudaSuccess ) {									\
		fprintf( stderr, "Error %s at line %d in file %s\n",	\
				cudaGetErrorString(err), __LINE__, __FILE__ );	\
		exit( 1 );												\
	}}

string trimQuotes(string input);
bool endsWith(string fullString, string ending);

#define FILTER_SIZE (3u)
#define TILE_SIZE (14u)
#define BLOCK_SIZE (16u)
#define OUTPUT_DIR "../pictures/"

// Masks allocated in constant memory
__constant__ float mask[FILTER_SIZE * FILTER_SIZE];


string getFileFromUser()
{
	string input;
	bool keepAsking = true;
	while (keepAsking)
	{
		std::cout << "Enter the filepath of a .png file (no spaces or diacritics): " << std::endl;
		std::cin >> input;
		input = trimQuotes(input);
		if (!endsWith(input, ".png"))
		{
			std::cout << "That's not a .png file." << std::endl;
			continue;
		}
		fstream file;
		file.open(input, ios::in);
		if (!file)
		{
			std::cout << "File not found." << std::endl;
			continue;
		}
		file.close();
		keepAsking = false;
	}
	return input;
}

string getActionChoiceFromUser() {
	string input;
	bool keepAsking = true;
	while (keepAsking)
	{
		std::cout << "Choose from the following actions:" << std::endl;
		std::cout << "1 - blurring" << std::endl;
		std::cout << "2 - edge detection" << std::endl;
		std::cin >> input;
		if (input != "1" && input != "2") {
			std::cout << "That's not a valid option." << std::endl;
		}
		keepAsking = false;
	}
	return input;
}

bool endsWith(string fullString, string ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	return false;
}

string trimQuotes(string input)
{
	input.erase(std::remove(input.begin(), input.end(), '\"'), input.end());
	return input;
}

string extractFileName(string filepath)
{
	size_t nameStart = filepath.find_last_of("/") + 1;
	if (nameStart == 0)
	{
		nameStart = filepath.find_last_of("\\") + 1;
	}
	size_t substrEnd = filepath.find(".png") - nameStart;
	return filepath.substr(nameStart, substrEnd);
}

__global__ void processWithMask(unsigned char* out, const unsigned char* __restrict__ in, size_t pitch, unsigned int width, unsigned int height)
{
	int x_o = TILE_SIZE * blockIdx.x + threadIdx.x;
	int y_o = TILE_SIZE * blockIdx.y + threadIdx.y;
	int x_i = x_o - FILTER_SIZE / 2;
	int y_i = y_o - FILTER_SIZE / 2;
	unsigned int sum = 0;

	__shared__ unsigned char sBuffer[BLOCK_SIZE][BLOCK_SIZE];

	if ((x_i >= 0) && (x_i < width) && (y_i >= 0) && (y_i < height))
		sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
	else
		sBuffer[threadIdx.y][threadIdx.x] = 0;

	__syncthreads();

	if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {

		float newVal = 0.0;
		for (int i = 0; i < FILTER_SIZE; i++)
		{
			for (int j = 0; j < FILTER_SIZE; j++)
			{
				if (y_i + i >= 0 && y_i + i < height)
				{
					if (x_i + j >= 0 && x_i + j < width)
					{
						newVal += mask[i * FILTER_SIZE + j] * sBuffer[threadIdx.y + i][threadIdx.x + j];
					}
				}
			}
		}

		if (x_o < width && y_o < height) {
			out[y_o * width + x_o] = (unsigned char)(newVal < 0.0 ? 0.0 : newVal > 255 ? 255 : newVal);
		}
	}
}

int main()
{
	string filepath = getFileFromUser();
	string actionChoice = getActionChoiceFromUser();

	string nameAddition = "";
	size_t bytes_m = FILTER_SIZE * FILTER_SIZE * sizeof(float);
	float* h_mask = new float[FILTER_SIZE * FILTER_SIZE];
	
	if (actionChoice == "1") {
		nameAddition = "-blurred";
		for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
		{
			h_mask[i] = 1.0f / 9.0f;
		}
	}
	else {
		nameAddition = "-edgy";
		float array[] = { -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0 };
		for (int i = 0; i < FILTER_SIZE * FILTER_SIZE; i++)
		{
			h_mask[i] = array[i];
		}
	}

	/* Load image from file */
	//png::image<png::rgb_pixel> img("../pictures/lena.png");
	png::image<png::rgb_pixel> img(filepath);

	string justName = extractFileName(filepath);
	string outputTo = OUTPUT_DIR + justName + nameAddition + ".png";
	std::cout << "I'll output to: " << outputTo << std::endl;

	unsigned int width = img.get_width();
	unsigned int height = img.get_height();

	/* Allocate memory buffers for the image processing */
	int size = width * height * sizeof(unsigned char);

	/* Allocate image buffers on the host memory */
	unsigned char* h_r = new unsigned char[size];
	unsigned char* h_g = new unsigned char[size];
	unsigned char* h_b = new unsigned char[size];

	unsigned char* h_r_n = new unsigned char[size];
	unsigned char* h_g_n = new unsigned char[size];
	unsigned char* h_b_n = new unsigned char[size];

	/* Convert PNG image to raw buffer */
	pvg::pngToRgb3(h_r, h_g, h_b, img);

	/* Allocate image buffre on GPGPU */
	unsigned char* d_r = NULL;
	unsigned char* d_g = NULL;
	unsigned char* d_b = NULL;

	size_t pitch_r = 0;
	size_t pitch_g = 0;
	size_t pitch_b = 0;

	unsigned char* d_r_n = NULL;
	unsigned char* d_g_n = NULL;
	unsigned char* d_b_n = NULL;

	CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, &pitch_r, width, height));
	CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &pitch_g, width, height));
	CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, &pitch_b, width, height));

	CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
	CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
	CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));

	/* Copy raw buffer from host memory to device memory */
	CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));

	/* mask to shared memory */
	CUDA_CHECK_RETURN(cudaMemcpyToSymbol(mask, h_mask, bytes_m));

	/* Configure image kernel */
	dim3 grid_size((width + TILE_SIZE - 1) / TILE_SIZE,
		(height + TILE_SIZE - 1) / TILE_SIZE);

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

	/* Run kernel and measure processing time */
	double start = omp_get_wtime();
	processWithMask <<<grid_size, block_size >>> (d_r_n, d_r, pitch_r, width, height);
	processWithMask <<<grid_size, block_size >>> (d_g_n, d_g, pitch_g, width, height);
	processWithMask <<<grid_size, block_size >>> (d_b_n, d_b, pitch_b, width, height);
	
	/* Wait untile the kernel exits */
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	double end = omp_get_wtime();

	/* Copy raw buffer from device memory to host memory */
	CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));

	/* Convert raw buffer to PNG image */
	pvg::rgb3ToPng(img, h_r_n, h_g_n, h_b_n);

	std::cout << "Done in " << end - start << " seconds." << std::endl;

	/* Write modified image to the disk */
	img.write(outputTo);

	/* Free allocated buffers */
	CUDA_CHECK_RETURN(cudaFree(d_r));
	CUDA_CHECK_RETURN(cudaFree(d_r_n));

	CUDA_CHECK_RETURN(cudaFree(d_g));
	CUDA_CHECK_RETURN(cudaFree(d_g_n));

	CUDA_CHECK_RETURN(cudaFree(d_b));
	CUDA_CHECK_RETURN(cudaFree(d_b_n));

	delete[] h_r;
	delete[] h_r_n;

	delete[] h_g;
	delete[] h_g_n;

	delete[] h_b;
	delete[] h_b_n;

	return 0;
}