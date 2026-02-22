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
    DenseLayer(int n_inputs, int n_outputs, const float* h_weights, const float* h_bias) {
        n_in = n_inputs; 
        n_out = n_outputs;

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

    void forward_host(const float* h_input, float* h_output) {
        float* d_input;
        
        cudaMalloc(&d_input, n_in * sizeof(float));
        cudaMemcpy(d_input, h_input, n_in * sizeof(float), cudaMemcpyHostToDevice);

        this->forward(d_input);

        cudaMemcpy(h_output, d_output, n_out * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_input);
    }
};

void* create_dense_layer(int n_inputs, int n_outputs, const float* h_weights, const float* h_bias) {
    return new DenseLayer(n_inputs, n_outputs, h_weights, h_bias);
}

void destroy_dense_layer(void* layer_ptr) {
    if (layer_ptr != nullptr) {
        delete static_cast<DenseLayer*>(layer_ptr);
    }
}

float* forward_dense_layer(void* layer_ptr, const float* d_input) {
    DenseLayer* layer = static_cast<DenseLayer*>(layer_ptr);
    return layer->forward(d_input);
}

void forward_dense_layer_host(void* layer_ptr, const float* h_input, float* h_output) {
    DenseLayer* layer = static_cast<DenseLayer*>(layer_ptr);
    layer->forward_host(h_input, h_output);
}

void* copy_array_to_device(const float* h_array, int size){
    float* d_array;
    cudaMalloc(&d_array,size*sizeof(float));
    cudaMemcpy(d_array,h_array,size*sizeof(float),cudaMemcpyHostToDevice);
    return static_cast<void*>(d_array);
}

void* copy_device_array_to_host(const float* d_array, int size) {
    float* h_array = (float*)malloc(size * sizeof(float)); 
    
    cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

    return static_cast<void*>(h_array);
}

// TODO: add function to free device array