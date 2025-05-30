#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <string.h>


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
	} }

#define VECT_SIZE (12000u)
#define BLOCK_SIZE (128u)

__global__ void vectFill(float* data)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < VECT_SIZE) data[i] = (float)i + 1;
}

__global__ void VectAdd(float* A, float* B, float* C)
{
	//identifikacni cislo vlakna(v ramci bloku)
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	C[i] = A[i] + B[i];
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {

	//int devices = 0;

	//CUDA_CHECK_RETURN( cudaGetDeviceCount( &devices ) );

	//cudaDeviceProp properties;

	//for (int i = 0; i < devices; ++i) {
	//	CUDA_CHECK_RETURN(cudaGetDeviceProperties(&properties, i));
	//	std::cout << "Device " << i << " name: " << properties.name << std::endl;
	//	std::cout << "Compute capability: " << properties.major << "." << properties.minor << std::endl;
	//	std::cout << "Block dimensions: " << properties.maxThreadsDim[0]
	//		<< ", " << properties.maxThreadsDim[1]
	//		<< ", " << properties.maxThreadsDim[2]
	//		<< std::endl;
	//	std::cout << "Grid dimensions: " << properties.maxGridSize[0]
	//		<< ", " << properties.maxGridSize[1]
	//		<< ", " << properties.maxGridSize[2]
	//		<< std::endl;
	//}

	/* Allocate data buffer in host memory */
	float* h_data_c = (float*)malloc(VECT_SIZE * sizeof(float));
	memset(h_data_c, 0, VECT_SIZE * sizeof(float));

	float* h_data_a = (float*)malloc(VECT_SIZE * sizeof(float));
	memset(h_data_a, 0, VECT_SIZE * sizeof(float));

	float* h_data_b = (float*)malloc(VECT_SIZE * sizeof(float));
	memset(h_data_b, 0, VECT_SIZE * sizeof(float));

	/* Allocate data buffer in device memory */
	float* d_data_c = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&d_data_c, VECT_SIZE * sizeof(float)));

	float* d_data_a = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&d_data_a, VECT_SIZE * sizeof(float)));

	float* d_data_b = NULL;
	CUDA_CHECK_RETURN(cudaMalloc(&d_data_b, VECT_SIZE * sizeof(float)));

	/* Configure kernel */
	int blockSize = BLOCK_SIZE;
	int gridSize = (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	//std::cout << blockSize << ", " << gridSize << std::endl;

	/* Run kernel */
	vectFill <<< gridSize, blockSize >>> (d_data_a);
	vectFill <<< gridSize, blockSize >>> (d_data_b);

	/* Wait until the kernel finishes its work */
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(h_data_a, d_data_a, VECT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(h_data_b, d_data_b, VECT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	VectAdd <<<gridSize, blockSize >>> (d_data_a, d_data_b, d_data_c);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(h_data_c, d_data_c, VECT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

	for( unsigned int i = 0; i < VECT_SIZE; ++i ) std::cout << h_data_a[i] << " + " << h_data_b[i] << " = " << h_data_c[i] << std::endl;
	//for (unsigned int i = 0; i < VECT_SIZE; ++i) std::cout << h_data_c[i] << std::endl;

	CUDA_CHECK_RETURN(cudaFree(d_data_a));
	CUDA_CHECK_RETURN(cudaFree(d_data_b));
	CUDA_CHECK_RETURN(cudaFree(d_data_c));

	free(h_data_a);
	free(h_data_b);
	free(h_data_c);

	return 0;
}