#pragma once
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
static __global__ void checkAllEqualKernel(const T* arr, bool* result, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (arr[idx] != value) {
            *result = false;
        }
    }
}

template <typename T>
bool isArrayAllEqualToValue(const T* d_arr, T value, int size) {
    bool* d_result;
    bool host_result = true;
    
    // Allocate device memory for the result
    cudaMalloc(&d_result, sizeof(bool));
    cudaMemcpy(d_result, &host_result, sizeof(bool), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    checkAllEqualKernel<<<blocks, threadsPerBlock>>>(d_arr, d_result, value, size);
    
    // Copy the result back to host
    cudaMemcpy(&host_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);

    return host_result;
}