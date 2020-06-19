#ifndef _CUDA_SOBEL_FILTER_H_
#define _CUDA_SOBEL_FILTER_H_
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

//Static values
#define SOBEL_MASK_SIZE 3
#define SOBEL_TILE_WIDTH 30
#define SOBEL_BLOCK_WIDTH (SOBEL_TILE_WIDTH + SOBEL_MASK_SIZE - 1)

int getSobelTileWidth(){
	return SOBEL_TILE_WIDTH;
}

/**
Dynamic Tiled Convolution: Implementation of Convolution 2D using shared memory. 
Uses dynamic parameters such as tile_width or block_size precalculated in host.
**/
__global__ void DynamicSobelConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ gx, const float* __restrict__ gy, int TILE_WIDTH, int BLOCK_WIDTH){
	extern __shared__ float imageDS[]; //Dynamic shared memory
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * TILE_WIDTH + ty;
	int colIn = blockIdx.x * TILE_WIDTH + tx;
	int radio = SOBEL_MASK_SIZE / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;
	
	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[getIndex(ty, tx, BLOCK_WIDTH )] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[getIndex(ty, tx, BLOCK_WIDTH)] = 0.0;

	__syncthreads();

	//Convolve image with Gradient Masks
	float sumGx = 0.0, sumGy = 0.0;
	if (ty < TILE_WIDTH && tx < TILE_WIDTH){
		for (int i = 0; i < SOBEL_MASK_SIZE; ++i){
			for (int j = 0; j < SOBEL_MASK_SIZE; ++j){
				sumGx += gx[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
				sumGy += gy[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sqrt(sumGx * sumGx + sumGy * sumGy);
	}
}

__global__ void DynamicSobelConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ gx, const float* __restrict__ gy, int TILE_WIDTH, int BLOCK_WIDTH, int dim){
	extern __shared__ float imageDS[]; //Dynamic shared memory
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * TILE_WIDTH + ty;
	int colIn = blockIdx.x * TILE_WIDTH + tx;
	int radio = SOBEL_MASK_SIZE / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;
	
	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[getIndex(ty, tx, BLOCK_WIDTH )] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[getIndex(ty, tx, BLOCK_WIDTH)] = 0.0;

	__syncthreads();

	//Convolve image with Gradient Masks
	float sumG = 0.0;
	if (ty < TILE_WIDTH && tx < TILE_WIDTH){
		for (int i = 0; i < SOBEL_MASK_SIZE; ++i){
			for (int j = 0; j < SOBEL_MASK_SIZE; ++j){
				if(dim==0){
					sumG += gx[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
				}else if(dim==1){
					sumG += gy[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
				}
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sumG;
	}
}
/**
Tiled Convolution: Implementation of Sobel Convolution 2D using shared memory.
**/
__global__ void tiledSobelConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ gx, const float* __restrict__ gy){
	__shared__ float imageDS[SOBEL_BLOCK_WIDTH][SOBEL_BLOCK_WIDTH];
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * SOBEL_TILE_WIDTH + ty;
	int colIn = blockIdx.x * SOBEL_TILE_WIDTH + tx;
	int radio = SOBEL_MASK_SIZE / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;

	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[ty][tx] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[ty][tx] = 0.0;

	__syncthreads();

	//Convolve image with Gradient Masks
	float sumGx = 0.0, sumGy = 0.0;
	if (ty < SOBEL_TILE_WIDTH && tx < SOBEL_TILE_WIDTH){
		for (int i = 0; i < SOBEL_MASK_SIZE; ++i){
			for (int j = 0; j < SOBEL_MASK_SIZE; ++j){
				sumGx += gx[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[i + ty][j + tx];
				sumGy += gy[getIndex(i, j, SOBEL_MASK_SIZE)] * imageDS[i + ty][j + tx];
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sqrt(sumGx * sumGx + sumGy * sumGy);
	}
}

/**
Naive implementation of Sobel Convolution
**/
__global__ void naiveSobelConvolution(float *image, float *result, int imageWidth, int imageHeight, float*gx, float*gy, int maskSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < imageHeight && j < imageWidth){
		float sumGx = 0.0, sumGy = 0.0;
		int radio = maskSize / 2;
		for (int x = -radio; x <= radio; ++x){
			for (int y = -radio; y <= radio; ++y){
				int nx = x + i;
				int ny = y + j;
				if (nx >= 0 && ny >= 0 && nx < imageHeight && ny < imageWidth){
					float valueNeighbor = image[nx * imageWidth + ny];
					sumGx += gx[getIndex(x + radio, y + radio, maskSize)] * valueNeighbor;
					sumGy += gy[getIndex(x + radio, y + radio, maskSize)] * valueNeighbor;
				}
			}
		}
		result[i * imageWidth + j] = sqrt(sumGx * sumGx + sumGy * sumGy);
	}
}

#endif
