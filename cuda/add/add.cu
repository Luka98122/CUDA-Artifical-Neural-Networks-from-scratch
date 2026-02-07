#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>


__global__ void add_kernel(float *a, float *b, float *c)
{
    *c = *a + *b;
}

__global__ void multiply_kernel(float *a, float *b, float *c)
{
    *c = (*a) * (*b);
}

__global__ void matAdd(const float* A,
                       const float* B,
                       float* C,
                       int rows,
                       int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}


extern "C" __declspec(dllexport)
void add_cuda(float *a, float *b, float *result)
{
    cudaSetDevice(0);
    float* d_a = nullptr;
    float* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));

    cudaMemcpy(d_a,a,sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_c = nullptr;
    cudaMalloc(&d_c, sizeof(float));

    add_kernel<<<1, 1>>>(d_a, d_b, d_c);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
}

extern "C" __declspec(dllexport)
void multiply_cuda(float *a, float *b, float *result)
{
    cudaSetDevice(0);
    float* d_a = nullptr;
    float* d_b = nullptr;
    cudaMalloc(&d_a, sizeof(float));
    cudaMalloc(&d_b, sizeof(float));

    cudaMemcpy(d_a,a,sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_c = nullptr;
    cudaMalloc(&d_c, sizeof(float));

    multiply_kernel<<<1, 1>>>(d_a, d_b, d_c);
    cudaGetLastError();
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_c, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_c);
    cudaFree(d_a);
    cudaFree(d_b);

}

extern "C" __declspec(dllexport)
void mat_add_cuda(const float* h_A,
                  const float* h_B,
                  float* h_C,
                  int rows,
                  int cols)
{
    cudaSetDevice(0);

    size_t bytes = rows * cols * sizeof(float);

    // Device pointers
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    matAdd<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#include <cuda_runtime.h>

__global__ void dot_kernel(const float* a, const float* b, float* partial_sum, int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load element into shared memory
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Write result of this block to global memory
    if (tid == 0)
        partial_sum[blockIdx.x] = sdata[0];
}

extern "C" __declspec(dllexport)
void dot_product_cuda(const float* h_a, const float* h_b, float* result, int n)
{
    cudaError_t err;
    cudaSetDevice(0);
    
    // Check if n is valid
    if (n <= 0) {
        *result = 0.5f;
        return;
    }
    
    float *d_a = nullptr, *d_b = nullptr;
    cudaMalloc(&d_a, n * sizeof(float));
    
    cudaMalloc(&d_b, n * sizeof(float));
    
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    float* d_partial = nullptr;
    err = cudaMalloc(&d_partial, gridSize * sizeof(float));
    
    // Initialize partial sums to 0
    cudaMemset(d_partial, 0, gridSize * sizeof(float));
    
    dot_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_a, d_b, d_partial, n);
    
    err = cudaGetLastError();

    
    cudaDeviceSynchronize();
    
    float* h_partial = new float[gridSize];
    err = cudaMemcpy(h_partial, d_partial, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0.0f;
    for (int i = 0; i < gridSize; ++i) {
        sum += h_partial[i];
    }
    
    *result = sum;
    delete[] h_partial;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partial);
}

<<<<<<< HEAD
__global__ void forward_layer_kernel(const float* input, 
                                     const float* weights, 
                                     const float* bias, 
                                     float* output, 
                                     int n_inputs, 
                                     int n_outputs)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_outputs) {
        float sum = 0.0f;
        
        // Dot product of Input vector and the Weights row corresponding to this neuron
        for (int i = 0; i < n_inputs; ++i) {
            // Weights are assumed flattened [n_outputs * n_inputs]
            // Accessing row 'row' and column 'i'
            sum += weights[row * n_inputs + i] * input[i];
        }

        // Add bias and store result
        // (Optional: You could add ReLU here: fmaxf(0.0f, sum + bias[row]))
        output[row] = sum + bias[row]; 
=======
__global__ void argmax_kernel(const float* a, float* block_max_vals, int* block_max_idxs, int n)
{
    extern __shared__ unsigned char smem[];
    float* svals = reinterpret_cast<float*>(smem);
    int* sidxs   = reinterpret_cast<int*>(svals + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize
    float v = -FLT_MAX;
    int   i = -1;

    if (gid < n) {
        v = a[gid];
        i = gid;
    }

    svals[tid] = v;
    sidxs[tid] = i;
    __syncthreads();


    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float v2 = svals[tid + s];
            int   i2 = sidxs[tid + s];

            float v1 = svals[tid];
            int   i1 = sidxs[tid];

            if (v2 > v1 || (v2 == v1 && i2 != -1 && (i1 == -1 || i2 < i1))) {
                svals[tid] = v2;
                sidxs[tid] = i2;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max_vals[blockIdx.x] = svals[0];
        block_max_idxs[blockIdx.x] = sidxs[0];
>>>>>>> ce3f4d2009155ec8feb10f64442ab9afc2918b2b
    }
}

extern "C" __declspec(dllexport)
<<<<<<< HEAD
void forward_layer_cuda(const float* h_input, 
                        const float* h_weights, 
                        const float* h_bias, 
                        float* h_output, 
                        int n_inputs, 
                        int n_outputs)
{
    cudaSetDevice(0);

    // Calculate sizes
    size_t size_in = n_inputs * sizeof(float);
    size_t size_out = n_outputs * sizeof(float);
    size_t size_weights = n_inputs * n_outputs * sizeof(float);

    // Allocate Device Memory
    float *d_input = nullptr, *d_weights = nullptr, *d_bias = nullptr, *d_output = nullptr;
    
    cudaMalloc(&d_input, size_in);
    cudaMalloc(&d_weights, size_weights);
    cudaMalloc(&d_bias, size_out);
    cudaMalloc(&d_output, size_out);

    // Copy Host -> Device
    cudaMemcpy(d_input, h_input, size_in, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, size_out, cudaMemcpyHostToDevice);

    // Launch Configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_outputs + threadsPerBlock - 1) / threadsPerBlock;

    forward_layer_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weights, d_bias, d_output, n_inputs, n_outputs);
    
    cudaDeviceSynchronize();

    // Copy Device -> Host
    cudaMemcpy(h_output, d_output, size_out, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_output);
=======
void argmax_cuda(const float* h_a, int* result_idx, int n)
{
    cudaSetDevice(0);

    if (!result_idx) return;

    if (n <= 0 || !h_a) {
        *result_idx = -1;
        return;
    }

    float* d_a = nullptr;
    cudaMalloc(&d_a, (size_t)n * sizeof(float));
    cudaMemcpy(d_a, h_a, (size_t)n * sizeof(float), cudaMemcpyHostToDevice);

    const int blockSize = 256;
    const int gridSize  = (n + blockSize - 1) / blockSize;

    float* d_block_vals = nullptr;
    int*   d_block_idxs = nullptr;
    cudaMalloc(&d_block_vals, (size_t)gridSize * sizeof(float));
    cudaMalloc(&d_block_idxs, (size_t)gridSize * sizeof(int));

    size_t shmemBytes = (size_t)blockSize * (sizeof(float) + sizeof(int));

    argmax_kernel<<<gridSize, blockSize, shmemBytes>>>(d_a, d_block_vals, d_block_idxs, n);
    cudaGetLastError();
    cudaDeviceSynchronize();

    float* h_block_vals = new float[gridSize];
    int*   h_block_idxs = new int[gridSize];

    cudaMemcpy(h_block_vals, d_block_vals, (size_t)gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_block_idxs, d_block_idxs, (size_t)gridSize * sizeof(int),   cudaMemcpyDeviceToHost);

    float bestVal = -FLT_MAX;
    int   bestIdx = -1;

    for (int b = 0; b < gridSize; ++b) {
        float v = h_block_vals[b];
        int   i = h_block_idxs[b];

        if (i == -1) continue;

        if (v > bestVal || (v == bestVal && i < bestIdx)) {
            bestVal = v;
            bestIdx = i;
        }
    }

    *result_idx = bestIdx;

    delete[] h_block_vals;
    delete[] h_block_idxs;

    cudaFree(d_block_vals);
    cudaFree(d_block_idxs);
    cudaFree(d_a);
>>>>>>> ce3f4d2009155ec8feb10f64442ab9afc2918b2b
}