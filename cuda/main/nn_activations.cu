#include "cuda_nn.h"
#include <cuda_runtime.h>
#include <float.h>
#include <math.h> // For fmaxf

__global__ void ReLU_kernel(float* input, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid<n){
        //fmaxf is a very optimised CUDA function, faster than doing an if here
        input[tid] = fmaxf(0.0f, input[tid]);
    }
} 

void relu_cuda(float* h_input, int n) 
{
    float* d_input;
    size_t bytes = n * sizeof(float);

    cudaMalloc((void**)&d_input, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    ReLU_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, n);

    cudaMemcpy(h_input, d_input, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
}