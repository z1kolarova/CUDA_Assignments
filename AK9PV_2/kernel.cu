#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

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

typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

bool isNullPtr(Matrix* ptr);
bool isNullPtr(float* ptr);
void printMatrix(Matrix* m);

Matrix* initMatrix(int width, int height)
{
	Matrix* m = (Matrix*)malloc(sizeof(Matrix));
	if (isNullPtr(m))
		return nullptr;
	m->width = width;
	m->height = height;
	m->elements = (float*)malloc(m->width * m->height * sizeof(float));
	if (isNullPtr(m->elements))
	{
		free(m);
		return nullptr;
	}
	memset(m->elements, 0, m->width * m->height * sizeof(float));
	return m;
}

bool isNullPtr(Matrix* ptr)
{
	if (ptr == nullptr)
	{
		std::cerr << "Matrix memory allocation failed.";
		return true;
	}
	return false;
}

bool isNullPtr(float* ptr)
{
	if (ptr == nullptr)
	{
		std::cerr << "Matrix elements memory allocation failed.";
		return true;
	}
	return false;
}

void printMatrix(Matrix* m)
{
	for (int row = 0; row < m->height; row++)
	{
		for (int col = 0; col < m->width; col++)
		{
			std::cout << m->elements[row * m->width + col] << "   ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void multiplyOnHost(const Matrix* A, const Matrix* B, Matrix* C)
{
	for (int row = 0; row < A->height; row++)
	{
		for (int col = 0; col < B->width; col++)
		{
			float sum = 0;
			for (int i = 0; i < A->width; i++)
			{
				float a = A->elements[row * A->width + 1];
				float b = B->elements[col + i * B->width];
				sum += a * b;
			}
			C->elements[row * C->width + col] = sum;
		}
	}
}

__global__ void multiplyInGlobal(const Matrix* a, const Matrix* b, Matrix* c)
{
	float value = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < c->height && col < c->width)
	{
		for (int i = 0; i < a->width; i++)
		{
			value += a->elements[row * a->width + i] * b->elements[i * b->width + col];
		}
		c->elements[row * c->width + col] = value;
	}
}

#define MATRIX_DIM1 (10u)
#define MATRIX_DIM2 (145u)
#define ELEMENT_COUNT (MATRIX_DIM1 * MATRIX_DIM2)
#define BLOCK_SIZE (16u)

__global__ void matrixFill(Matrix* m)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = row * m->width + col;

	if (i < ELEMENT_COUNT)
		m->elements[i] = (float)i + 1;
}

void fillMatrixLocally(Matrix* m)
{
	unsigned long int matrixSize = m->height * m->width;
	for (int i = 0; i < matrixSize; i++) {
		m->elements[i] = (float)i + 1;
	}
}

int main()
{
	/* Initialize memory and matrixes in host memory */
	Matrix* a = initMatrix(MATRIX_DIM1, MATRIX_DIM2);
	if (isNullPtr(a))
	{
		return 1;
	}

	Matrix* b = initMatrix(MATRIX_DIM2, MATRIX_DIM1);
	if (isNullPtr(b))
	{
		free(a->elements);
		free(a);
		return 1;
	}

	Matrix* c = initMatrix(MATRIX_DIM2, MATRIX_DIM2);
	if (isNullPtr(c))
	{
		free(b->elements);
		free(b);
		free(a->elements);
		free(a);
		return 1;
	}

	//fillMatrixLocally(a);
	//fillMatrixLocally(b);

	//printMatrix(a);
	//printMatrix(b);
	//printMatrix(c);

	/* Allocate data buffer in device memory */
	Matrix* d_a = NULL;
	float* d_a_elements = NULL;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_a, a, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a_elements, ELEMENT_COUNT * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(&(d_a->elements), &(d_a_elements), sizeof(float*), cudaMemcpyHostToDevice));
	//CUDA_CHECK_RETURN(cudaMemcpy(d_a_elements, a->elements, ELEMENT_COUNT * sizeof(float), cudaMemcpyHostToDevice));
	
	Matrix* d_b = NULL;
	float* d_b_elements;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_b, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_b, b, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_b_elements, ELEMENT_COUNT * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(&(d_b->elements), &(d_b_elements), sizeof(float*), cudaMemcpyHostToDevice));
	//CUDA_CHECK_RETURN(cudaMemcpy(d_b_elements, b->elements, ELEMENT_COUNT * sizeof(float), cudaMemcpyHostToDevice));

	Matrix* d_c = NULL;
	float* d_c_elements;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_c, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMemcpy(d_c, c, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_c_elements, MATRIX_DIM2 * MATRIX_DIM2 * sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemcpy(&(d_c->elements), &(d_c_elements), sizeof(float*), cudaMemcpyHostToDevice));

	/* Configure kernel */
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((MATRIX_DIM2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (MATRIX_DIM2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

	/* Run kernel */
	matrixFill <<<gridSize, blockSize >>> (d_a);
	matrixFill <<<gridSize, blockSize >>> (d_b);

	/* Wait until the kernel finishes its work */
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	multiplyInGlobal <<<gridSize, blockSize >>> (d_a, d_b, d_c);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	CUDA_CHECK_RETURN(cudaMemcpy(a->elements, d_a_elements, MATRIX_DIM1 * MATRIX_DIM2 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(b->elements, d_b_elements, MATRIX_DIM2 * MATRIX_DIM1 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(c->elements, d_c_elements, MATRIX_DIM2 * MATRIX_DIM2 * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());

	printMatrix(a);
	printMatrix(b);
	printMatrix(c);

	CUDA_CHECK_RETURN(cudaFree(d_c_elements));
	CUDA_CHECK_RETURN(cudaFree(d_c));
	CUDA_CHECK_RETURN(cudaFree(d_b_elements));
	CUDA_CHECK_RETURN(cudaFree(d_b));
	CUDA_CHECK_RETURN(cudaFree(d_a_elements));
	CUDA_CHECK_RETURN(cudaFree(d_a));

	free(c->elements);
	free(c);
	free(b->elements);
	free(b);
	free(a->elements);
	free(a);
    return 0;
}