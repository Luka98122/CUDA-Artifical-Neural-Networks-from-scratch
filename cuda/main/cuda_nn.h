#ifndef CUDA_NN_H
#define CUDA_NN_H

#ifdef _WIN32
    #define CUDA_API extern "C" __declspec(dllexport)
#else
    #define CUDA_API extern "C"
#endif


// Basic Math Operations
CUDA_API void mat_add_cuda(const float* h_A, const float* h_B, float* h_C, int rows, int cols);
CUDA_API void dot_product_cuda(const float* h_a, const float* h_b, float* result, int n);

// Neural Network Specific Operations
CUDA_API void forward_layer_cuda(const float* h_input, const float* h_weights, const float* h_bias, float* h_output, int n_inputs, int n_outputs);
CUDA_API void argmax_cuda(const float* h_a, int* result_idx, int n);

CUDA_API void relu_cuda(float* h_input, int n);
CUDA_API void relu_cuda_100(float* h_input, int n);


// Switch to OOP:
CUDA_API void* create_dense_layer(int n_inputs, int n_outputs, const float* h_weights, const float* h_bias) {
    return new DenseLayer(n_inputs, n_outputs, h_weights, h_bias);
}

CUDA_API void destroy_dense_layer(void* layer_ptr) {
    if (layer_ptr != nullptr) {
        delete static_cast<DenseLayer*>(layer_ptr);
    }
}

CUDA_API float* forward_dense_layer(void* layer_ptr, const float* d_input) {
    DenseLayer* layer = static_cast<DenseLayer*>(layer_ptr);
    return layer->forward(d_input);
}


#endif // CUDA_NN_H