#include "cuda_nn.h" // Include our header
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------
// Forward Layer (Dense)
// ---------------------------
__global__ void forward_layer_kernel(const float* input, const float* weights, const float* bias, float* output, int n_inputs, int n_outputs)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n_outputs) {
        float sum = 0.0f;
        for (int i = 0; i < n_inputs; ++i) {
            sum += weights[row * n_inputs + i] * input[i];
        }
        output[row] = sum + bias[row];
    }
}

void forward_layer_cuda(const float* h_input, const float* h_weights, const float* h_bias, float* h_output, int n_inputs, int n_outputs)
{
    cudaSetDevice(0);
    size_t size_in = n_inputs * sizeof(float);
    size_t size_out = n_outputs * sizeof(float);
    size_t size_weights = n_inputs * n_outputs * sizeof(float);

    float *d_input, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_input, size_in);
    cudaMalloc(&d_weights, size_weights);
    cudaMalloc(&d_bias, size_out);
    cudaMalloc(&d_output, size_out);

    cudaMemcpy(d_input, h_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, size_out, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_outputs + threadsPerBlock - 1) / threadsPerBlock;

    forward_layer_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_bias, d_output, n_inputs, n_outputs);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size_out, cudaMemcpyDeviceToHost);

    cudaFree(d_input); cudaFree(d_weights); cudaFree(d_bias); cudaFree(d_output);
}

// ---------------------------
// Argmax
// ---------------------------
__global__ void argmax_kernel(const float* a, float* block_max_vals, int* block_max_idxs, int n)
{
    extern __shared__ unsigned char smem[];
    float* svals = reinterpret_cast<float*>(smem);
    int* sidxs = reinterpret_cast<int*>(svals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float v = -FLT_MAX;
    int i = -1;

    if (gid < n) {
        v = a[gid];
        i = gid;
    }

    svals[tid] = v; sidxs[tid] = i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = svals[tid + s]; int i2 = sidxs[tid + s];
            float v1 = svals[tid];     int i1 = sidxs[tid];
            if (v2 > v1 || (v2 == v1 && i2 != -1 && (i1 == -1 || i2 < i1))) {
                svals[tid] = v2; sidxs[tid] = i2;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_max_vals[blockIdx.x] = svals[0];
        block_max_idxs[blockIdx.x] = sidxs[0];
    }
}

void argmax_cuda(const float* h_a, int* result_idx, int n)
{
    cudaSetDevice(0);
    if (!result_idx || n <= 0) { if(result_idx) *result_idx = -1; return; }

    float* d_a = nullptr;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    float* d_block_vals = nullptr;
    int* d_block_idxs = nullptr;
    cudaMalloc(&d_block_vals, gridSize * sizeof(float));
    cudaMalloc(&d_block_idxs, gridSize * sizeof(int));

    size_t shmemBytes = blockSize * (sizeof(float) + sizeof(int));
    argmax_kernel<<<gridSize, blockSize, shmemBytes>>>(d_a, d_block_vals, d_block_idxs, n);
    cudaDeviceSynchronize();

    float* h_block_vals = new float[gridSize];
    int* h_block_idxs = new int[gridSize];

    cudaMemcpy(h_block_vals, d_block_vals, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_idxs, d_block_idxs, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    float bestVal = -FLT_MAX;
    int bestIdx = -1;

    for (int b = 0; b < gridSize; ++b) {
        if (h_block_idxs[b] == -1) continue;
        if (h_block_vals[b] > bestVal || (h_block_vals[b] == bestVal && h_block_idxs[b] < bestIdx)) {
            bestVal = h_block_vals[b];
            bestIdx = h_block_idxs[b];
        }
    }
    *result_idx = bestIdx;

    delete[] h_block_vals; delete[] h_block_idxs;
    cudaFree(d_block_vals); cudaFree(d_block_idxs); cudaFree(d_a);
}