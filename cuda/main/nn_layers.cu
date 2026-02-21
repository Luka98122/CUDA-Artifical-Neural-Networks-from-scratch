#include "cuda_nn.h"
#include <cuda_runtime.h>
#include <float.h>

// ---------------------------
// Forward Layer (Dense)
// ---------------------------

__global__ void forward_layer_kernel(const float* input, const float* weights, const float* bias, float* output, int n_inputs, int n_outputs)
{
    // TODO: Implement shared memory cache
    
    // Which output neuron this thread is responsible for
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_outputs) {
        float sum = 0.0f;
        for (int i = 0; i < n_inputs; ++i) {
            sum += weights[row * n_inputs + i] * input[i];
        }
        output[row] = sum + bias[row];
    }
}

// OOP layer

class DenseLayer {
private:
    int n_in;
    int n_out;
    
    float* d_weights;
    float* d_bias;
    float* d_output;

public:
    //constructor (initialise)
    DenseLayer(int n_inputs, int n_outputs, const float* h_weights, const float* h_bias) 
        : n_in(n_inputs), n_out(n_outputs) {
        
        size_t size_out = n_out * sizeof(float);
        size_t size_weights = n_in * n_out * sizeof(float);

        cudaMalloc(&d_weights, size_weights);
        cudaMalloc(&d_bias, size_out);
        cudaMalloc(&d_output, size_out);

        cudaMemcpy(d_weights, h_weights, size_weights, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias, size_out, cudaMemcpyHostToDevice);
    }

    // destructor
    ~DenseLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
    }

    // takes a device pointer for input, returns a device pointer for output
    float* forward(const float* d_input) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_out + threadsPerBlock - 1) / threadsPerBlock;

        forward_layer_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_input, d_weights, d_bias, d_output, n_in, n_out
        );
        cudaDeviceSynchronize();
        
        return d_output;
    }
};

// Old forward layer

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


