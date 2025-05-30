/**
 * Simple CUDA application template.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <string>

#define BLOCK_SIZE (16u)
#define MUL_ON_CPU 0

#define MATRIX_A_W 2
#define MATRIX_A_H 3
#define MATRIX_B_W 3
#define MATRIX_B_H 2

typedef struct {
	int width;
	int height;
	int* elements;
} Matrix;

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


__global__ void MultiplyOnGPU(const Matrix* sourceMatrix1, const Matrix* sourceMatrix2, Matrix* resultMatrix) {
	int result = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < resultMatrix->width && row < resultMatrix->height) {
		for (int i = 0; i < sourceMatrix1->width; i++) {
			result += sourceMatrix1->elements[row * sourceMatrix1->width + i] *
				sourceMatrix2->elements[i * sourceMatrix2->width + col];
		}

		resultMatrix->elements[row * resultMatrix->width + col] = result;
	}

}

void MultiplyOnCPU(const Matrix sourceMatrix1, const Matrix sourceMatrix2, Matrix resultMatrix) {
	for (int row = 0; row < sourceMatrix1.height; row++) {
		for (int col = 0; col < sourceMatrix2.width; col++) {
			int result = 0;
			for (int i = 0; i < sourceMatrix1.width; ++i) {
				result += sourceMatrix1.elements[row * sourceMatrix1.width + i] * sourceMatrix2.elements[i * sourceMatrix2.width + col];
			}
			resultMatrix.elements[row * resultMatrix.width + col] = result;
		}
	}

}

void printMatrix(Matrix* m) {
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

int SetMatrix(int height, int width, Matrix* matrix) {
	matrix->height = height;
	matrix->width = width;
	int matrixSize = matrix->height * matrix->width;
	return matrixSize;
}

void FillMatrix(bool zeroes, Matrix* matrix) {
	unsigned long int matrixSize = matrix->height * matrix->width;
	matrix->elements = (int*)malloc(sizeof(int) * matrixSize);
	for (size_t i = 0; i < matrixSize; ++i) {
		if (zeroes)
			matrix->elements[i] = 0;
		else
			matrix->elements[i] = (int)i + 1;
	}
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(int argc, char** argv) {
	Matrix sourceMatrix1;
	int sourceMatrix1Size = SetMatrix(MATRIX_A_H, MATRIX_A_W, &sourceMatrix1);
	FillMatrix(false, &sourceMatrix1);

	Matrix sourceMatrix2;
	int sourceMatrix2Size = SetMatrix(MATRIX_B_H, MATRIX_B_W, &sourceMatrix2);
	FillMatrix(false, &sourceMatrix2);

	if (sourceMatrix1.width != sourceMatrix2.height) {
		printf("\nMatice nejsou správných rozměrů!\n\n");
		return 1;
	}

	printMatrix(&sourceMatrix1);
	printMatrix(&sourceMatrix2);

	Matrix resultMatrixCPU;
	SetMatrix(sourceMatrix1.height, sourceMatrix2.width, &resultMatrixCPU);
	FillMatrix(true, &resultMatrixCPU);

	Matrix resultMatrixGPU;
	int resultMatrixSize = SetMatrix(sourceMatrix1.height, sourceMatrix2.width, &resultMatrixGPU);
	FillMatrix(true, &resultMatrixGPU);

	if (MUL_ON_CPU)
	{
		double startCPU = omp_get_wtime();
		MultiplyOnCPU(sourceMatrix1, sourceMatrix2, resultMatrixCPU);
		double endCPU = omp_get_wtime();

		printf("\nNa CPU hotovo za %.7f sec.\n", endCPU - startCPU);
		//	printf("\nVýsledná matice CPU:\n");
		//	printMatrix(resultMatrixCPU);
		//	printf("\n");
	}
	double startAlloc = omp_get_wtime();
	// alokace paměti na gpu
	Matrix* deviceSourceMatrix1;
	int* dataMatrix1;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceSourceMatrix1, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dataMatrix1, sizeof(int) * sourceMatrix1Size));

	Matrix* deviceSourceMatrix2;
	int* dataMatrix2;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceSourceMatrix2, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dataMatrix2, sizeof(int) * sourceMatrix2Size));

	Matrix* deviceResultMatrix;
	int* dataMatrixResult;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&deviceResultMatrix, sizeof(Matrix)));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&dataMatrixResult, sizeof(int) * resultMatrixSize));

	double endAlloc = omp_get_wtime();
	printf("Alokace paměti na GPU %.7f sec.\n", endAlloc - startAlloc);

	double startCopy = omp_get_wtime();
	// přesun dat do paměti GPU
	CUDA_CHECK_RETURN(cudaMemcpy(deviceSourceMatrix1, &sourceMatrix1, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dataMatrix1, sourceMatrix1.elements, sizeof(int) * sourceMatrix1Size, cudaMemcpyHostToDevice));
	// nastavení ukazatele elements na pole
	CUDA_CHECK_RETURN(cudaMemcpy(&(deviceSourceMatrix1->elements), &dataMatrix1, sizeof(int*), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(deviceSourceMatrix2, &sourceMatrix2, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(dataMatrix2, sourceMatrix2.elements, sizeof(int) * sourceMatrix2Size, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(&(deviceSourceMatrix2->elements), &dataMatrix2, sizeof(int*), cudaMemcpyHostToDevice));

	CUDA_CHECK_RETURN(cudaMemcpy(deviceResultMatrix, &resultMatrixGPU, sizeof(Matrix), cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(&(deviceResultMatrix->elements), &dataMatrixResult, sizeof(int*), cudaMemcpyHostToDevice));

	double endCopy = omp_get_wtime();
	printf("Přenos dat na GPU %.7f sec.\n", endCopy - startCopy);

	dim3 gridSize((resultMatrixGPU.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(resultMatrixGPU.height + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);


	double startGPU = omp_get_wtime();
	MultiplyOnGPU << <gridSize, blockSize >> > (deviceSourceMatrix1, deviceSourceMatrix2, deviceResultMatrix);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	double endGPU = omp_get_wtime();

	CUDA_CHECK_RETURN(cudaGetLastError());

	// přesun výsledného pole do CPU
	CUDA_CHECK_RETURN(cudaMemcpy(resultMatrixGPU.elements, dataMatrixResult, sizeof(int) * resultMatrixSize, cudaMemcpyDeviceToHost));


	printf("Na GPU hotovo za %.7f sec.\n", endGPU - startGPU);
	printf("\nVýsledná matice GPU:\n");
	printMatrix(&resultMatrixGPU);

	free(sourceMatrix1.elements);
	free(sourceMatrix2.elements);
	free(resultMatrixCPU.elements);
	free(resultMatrixGPU.elements);

	CUDA_CHECK_RETURN(cudaFree(dataMatrix1));
	CUDA_CHECK_RETURN(cudaFree(deviceSourceMatrix1));
	CUDA_CHECK_RETURN(cudaFree(dataMatrix2));
	CUDA_CHECK_RETURN(cudaFree(deviceSourceMatrix2));
	CUDA_CHECK_RETURN(cudaFree(dataMatrixResult));
	CUDA_CHECK_RETURN(cudaFree(deviceResultMatrix));


	CUDA_CHECK_RETURN(cudaDeviceReset());

	return 0;
}