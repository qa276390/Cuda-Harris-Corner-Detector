#ifndef _CUDA_VECTOR_OPERATION_H_
#define _CUDA_VECTOR_OPERATION_H_
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__global__ void VecOperation(const float* A, const float* B, float* C, int N, int OP){


	
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < N){
		if(OP==0) // Addition(+)
		{
			C[i] = A[i] + B[i];
		
		}else if(OP==1){ // Substract(-)
			
			C[i] = A[i] - B[i];
		
		}else if(OP==2){ // Multiplication(*)
			
			C[i] = A[i] * B[i];
	
		}else if(OP==3){ // Division(/)
		
			C[i] = A[i] / (B[i] + 1E-8);
		
		}
	
	
	}
	__syncthreads();

}
#endif
