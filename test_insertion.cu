#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>

int main()
{
    TIME_INIT;
    const uint32_t tableSize = 1 << 25;
    const uint32_t testMaxSize = 1 << 24;
    const uint32_t seed1 = 114514;
    const uint32_t seed2 = 191981;
    const int range = 1 << 30;
    hashTableEntry *d_hashTable = nullptr;
    initHashTable(&d_hashTable, tableSize);

    int *d_keys = nullptr;
    cudaMalloc((void **)&d_keys, sizeof(int) * tableSize);
    cudaMemset(d_keys, 0xff, sizeof(int) * tableSize);
    generateRandomKeys<<<(testMaxSize + 255) / 256, 256>>>(d_keys, tableSize, range);
    cudaDeviceSynchronize();
    //copy keys to host
    std::vector<int> h_keys(tableSize);
    cudaMemcpy(h_keys.data(), d_keys, sizeof(int) * tableSize, cudaMemcpyDeviceToHost);
    //print some
    for (int i = 0; i < 10; ++i)
    {
        std::cout << h_keys[i] << " ";
    }

    int *d_retvals = nullptr;
    cudaMalloc((void **)&d_retvals, sizeof(int) * testMaxSize);
    for (uint32_t s = 11; s <= 24; ++s)
    {
        int testSize = 1 << s;
        HashFunc f1{seed1 * s, tableSize}, f2{seed2 * s, tableSize};
        reuseHashTable(&d_hashTable, tableSize);
        TIME_START;
        insertItemBatch<<<(testSize + 255) / 256, 256>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
        std::cout << "testsize,elapsed_μs,valid | " << testSize << "," << elapsed_μs << "," << valid << std::endl;
    }
    cudaFree(d_keys);
    cudaFree(d_retvals);
    return 0;
}