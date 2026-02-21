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
#endif // CUDA_NN_H