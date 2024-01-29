#pragma once
#include <iostream>
#include <cuda_runtime.h>

template <typename T>
static __global__ void checkAllEqualKernel(const T* arr, int* result, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (arr[idx] != value) {
            //atomicly set result to false
            atomicCAS(result, 1, 0);
        }
    }
}

template <typename T>
bool isArrayAllEqualToValue(const T* d_arr, int size, T value) {
    if(size == 0)
        return true;
    int* d_result;
    int host_result = 1;
    
    // Allocate device memory for the result
    auto err = cudaMalloc(&d_result, sizeof(int));
    if(err != cudaSuccess) {
        std::cout << "cudaMalloc failed with error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(d_result, &host_result, sizeof(int), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        std::cout << "cudaMemcpy failed with error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    
    int threadsPerBlock = 512;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    // std::cout << "blocks: " << blocks << " threadsPerBlock: " << threadsPerBlock << std::endl;
    // Launch the kernel
    checkAllEqualKernel<<<blocks, threadsPerBlock>>>(d_arr, d_result, value, size);
    cudaDeviceSynchronize();
    
    // Copy the result back to host
    err = cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        std::cout << "cudaMemcpy failed with error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    // Free device memory
    err = cudaFree(d_result);
    if(err != cudaSuccess) {
        std::cout << "cudaFree failed with error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    return host_result;
}

template <typename T>
static __global__ void checkAllNotEqualKernel(T* arr, int* result, T value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (arr[idx] == value) {
            if(*result == true)
                atomicCAS(result, true, false);
        }
    }
}

template <typename T>
bool isArrayAllNotEqualToValue(T* d_arr, int size, T value) {
    if(size == 0){
        return true;
    }
    int* d_result;
    int host_result = true;
    
    // Allocate device memory for the result
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_result, &host_result, sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the kernel
    checkAllNotEqualKernel<<<blocks, threadsPerBlock>>>(d_arr, d_result, value, size);
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(&host_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_result);

    return host_result;
}

bool h_isArrayAllEqualToValue(const int* d_arr, int value, int size) {
    int* h_arr = new int[size];
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++) {
        if(h_arr[i] != value)
            return false;
    }
    delete [] h_arr;
    return true;
}

bool h_isArrayAllNotEqualToValue(const int* d_arr, int value, int size) {
    int* h_arr = new int[size];
    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < size; i++) {
        if(h_arr[i] == value)
            return false;
    }
    delete [] h_arr;
    return true;
}