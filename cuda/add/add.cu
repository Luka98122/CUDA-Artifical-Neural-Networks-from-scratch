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
