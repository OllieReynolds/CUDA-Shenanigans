#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

// The device kernel
__global__ void my_first_kernel(float *x) {
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	x[tid] = (float)threadIdx.x;
}

int main() {
	// Setup
	float *hx, *dx;
	int blocks = 2;
	int threads = 8;
	int size = blocks*threads;

	// Allocate host and device memory
	hx = (float*)malloc(size * sizeof(float));
	cudaMalloc((void**)&dx, size * sizeof(float));

	// Execute kernel 
	my_first_kernel << <blocks, threads >> > (dx);

	// Copy device memory back to host memory
	cudaMemcpy(hx, dx, size * sizeof(float), cudaMemcpyDeviceToHost);

	// Output results
	for (int i = 0; i < size; i++) {
		printf(" n, x = %d %f\n", i, hx[i]);
	}

	// Free memory
	cudaFree(dx);
	free(hx);
}