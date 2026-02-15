#include "cuda_nn.h"
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------
// Matrix Addition
// ---------------------------
__global__ void matAdd(const float* A, const float* B, float* C, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

void mat_add_cuda(const float* h_A, const float* h_B, float* h_C, int rows, int cols)
{
    cudaSetDevice(0);
    size_t bytes = rows * cols * sizeof(float);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    matAdd<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

// ---------------------------
// Dot Product
// ---------------------------
__global__ void dot_kernel(const float* a, const float* b, float* partial_sum, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();
    //"Reduction tree", with every iteration half the array gets summed.x
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sum[blockIdx.x] = sdata[0];
}

void dot_product_cuda(const float* h_a, const float* h_b, float* result, int n)
{
    cudaSetDevice(0);

    float *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_partial, gridSize * sizeof(float));

    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_partial, 0, gridSize * sizeof(float));

    dot_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_b, d_partial, n);
    cudaDeviceSynchronize();
    // h_partial is the array of the sum of the products of all elements in a block
    // so we need to add all of them together.
    float* h_partial = new float[gridSize];
    cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    
    float sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) sum += h_partial[i]; 
    *result = sum;

    delete[] h_partial;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_partial);
}