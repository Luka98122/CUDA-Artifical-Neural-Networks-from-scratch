#include <cuda_runtime.h>

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
    }
}

extern "C" __declspec(dllexport)
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
}