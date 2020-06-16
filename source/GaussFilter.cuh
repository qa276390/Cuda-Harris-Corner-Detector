#ifndef _CUDA_GAUSS_KERNEL_H_
#define _CUDA_GAUSS_KERNEL_H_
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

/***
Used in static Tiled Convolution
Best Results with 32 threads per block. The following configurations will allow this quantity
MASK_SIZE = [ 3 , 5 , 7 , 9 ... ]
GAUSS_TILE_WIDTH = {30, 28, 26, 24 .. ]
***/
//Static values
#define GAUSS_MASK_SIZE 7
#define GAUSS_TILE_WIDTH 26
#define GAUSS_BLOCK_WIDTH (GAUSS_TILE_WIDTH + GAUSS_MASK_SIZE - 1)

int getGaussTileWidth(){
	return GAUSS_TILE_WIDTH;
}

__device__ int getIndex(int x, int y, int width){
	return x * width + y;
}

/**
Dynamic Tiled Convolution: Implementation of Convolution 2D using shared memory. 
Uses dynamic parameters such as tile_width or block_size precalculated in host.
**/
__global__ void DynamicTiledConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ mask, int maskSize , int TILE_WIDTH, int BLOCK_WIDTH){
	extern __shared__ float imageDS[]; //Dynamic shared memory
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * TILE_WIDTH + ty;
	int colIn = blockIdx.x * TILE_WIDTH + tx;
	int radio = maskSize / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;
	
	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[getIndex(ty, tx, BLOCK_WIDTH )] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[getIndex(ty, tx, BLOCK_WIDTH)] = 0.0;

	__syncthreads();

	//Convolve image with gaussian mask
	float sum = 0.0;
	if (ty < TILE_WIDTH && tx < TILE_WIDTH){
		for (int i = 0; i < maskSize; ++i){
			for (int j = 0; j < maskSize; ++j){
				sum += mask[getIndex(i, j, maskSize)] * imageDS[getIndex(i + ty, j + tx, BLOCK_WIDTH)];
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sum;
	}
}

/**
Tiled Convolution: Implementation of Convolution 2D using shared memory.
**/

__global__ void tiledConvolution(float *image, float *result, int imageWidth, int imageHeight, const float* __restrict__ mask, int maskSize){

	__shared__ float imageDS[GAUSS_BLOCK_WIDTH][GAUSS_BLOCK_WIDTH];
	int tx = threadIdx.x, ty = threadIdx.y;
	int rowIn = blockIdx.y * GAUSS_TILE_WIDTH + ty;
	int colIn = blockIdx.x * GAUSS_TILE_WIDTH + tx;
	int radio = maskSize / 2;
	int rowOut = rowIn - radio;
	int colOut = colIn - radio;

	//Copy from global memory to shared memory
	if (rowOut < imageHeight && colOut < imageWidth && rowOut >= 0 && colOut >= 0)
		imageDS[ty][tx] = image[getIndex(rowOut, colOut, imageWidth)];
	else
		imageDS[ty][tx] = 0.0;

	__syncthreads();

	//Convolve image with gaussian mask
	float sum = 0.0;
	if (ty < GAUSS_TILE_WIDTH && tx < GAUSS_TILE_WIDTH){
		for (int i = 0; i < maskSize; ++i){
			for (int j = 0; j < maskSize; ++j){
				sum += (mask[getIndex(i, j, maskSize)] * imageDS[i + ty][j + tx]);
			}
		}
		if (rowIn < imageHeight && colIn < imageWidth)
			result[getIndex(rowIn, colIn, imageWidth)] = sum;
	}
}

/**
Naive implementation of convolution2D
**/
__global__ void naiveConvolution(float *image, float *result, int imageWidth, int imageHeight, float*mask, int maskSize){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i < imageHeight && j < imageWidth){
		float sum = 0;
		int radio = maskSize / 2;
		for (int x = -radio; x <= radio; ++x){
			for (int y = -radio; y <= radio; ++y){
				int nx = x + i;
				int ny = y + j;
				if (nx >= 0 && ny >= 0 && nx < imageHeight && ny < imageWidth){
					sum += mask[(x + radio) * maskSize + (y + radio)] * image[nx * imageWidth + ny];
				}
			}
		}
		result[i * imageWidth + j] = sum;
	}
}

#endif
