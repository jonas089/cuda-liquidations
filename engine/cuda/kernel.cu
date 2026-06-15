#include <cuda_runtime.h>
#include <stdio.h>


__device__ int add(unsigned int x, unsigned int y){
	return x + y;
}

__device__ int mul(unsigned int x, unsigned int y){
	return x * y;
}

__global__ void add_kernel(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int n){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		out[idx] = add(a[idx], b[idx]);
	}
}

__global__ void mul_kernel(unsigned int* a, unsigned int* b, unsigned int* out, unsigned int n){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n){
		out[idx] = mul(a[idx], b[idx]);
	}
}

__global__ void check_liquidations(
    const unsigned long long* liq,
    const unsigned long long price,
    unsigned char* out,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        out[i] = (price <= liq[i]) ? 1 : 0;
    }
}

extern "C" void launch_add_kernel(
	unsigned int* a,
	unsigned int* b,
	unsigned int* out,
	unsigned int n
) {
	unsigned int threads = 256;
	unsigned int blocks = (n + threads - 1) / threads;
	add_kernel<<<blocks, threads>>>(a, b, out, n);
	cudaDeviceSynchronize();
}


extern "C" void launch_check_liquidations(
    const unsigned long long* liq,
    const unsigned long long price,
    unsigned char* out,
    unsigned int n
) {
    unsigned int threads = 1024;
    unsigned int blocks = (n + threads - 1) / threads;

    check_liquidations<<<blocks, threads>>>(
        liq, price, out, n
    );

    cudaDeviceSynchronize();
}