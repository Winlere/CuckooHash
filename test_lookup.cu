#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <omp.h>

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
    generateRandomKeys<<<(testMaxSize + 255) / 256, 256>>>(d_keys, tableSize, range, seed1);
    cudaDeviceSynchronize();

    HashFunc f1{seed1, tableSize}, f2{seed2, tableSize};

    int *d_retvals = nullptr;
    cudaMalloc((void **)&d_retvals, sizeof(int) * testMaxSize);
    {
        int s = 24;
        int testSize = 1 << s;
        reuseHashTable(&d_hashTable, tableSize);
        TIME_START;
        insertItemBatch<<<(testSize + 255) / 256, 256>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
    }

    // random shuffle the old keys. to ensure uniformity
    std::vector<int> oldKeys(testMaxSize);
    cudaMemcpy(oldKeys.data(), d_keys, sizeof(int) * testMaxSize, cudaMemcpyDeviceToHost);
    std::random_shuffle(oldKeys.begin(), oldKeys.end());
    std::unordered_set<int> oldKeysSet(oldKeys.begin(), oldKeys.end());
    cudaMemcpy(d_keys, oldKeys.data(), sizeof(int) * testMaxSize, cudaMemcpyHostToDevice);

    // prepare randomkeys
    int *d_queries = nullptr;
    cudaMalloc((void **)&d_queries, sizeof(int) * testMaxSize);
    
    std::vector<int> h_newRandomKeys(testMaxSize);
    #pragma omp parallel for
    for (int i = 0; i < testMaxSize; ++i)
    {
        
        while (oldKeysSet.find(key) != oldKeysSet.end())
        {
            key = rand() % range;
        }
        h_newRandomKeys[i] = key;
    }
    
    
    int *d_newRandomKeys = nullptr;
    cudaMalloc((void **)&d_newRandomKeys, sizeof(int) * testMaxSize);
    
    
    for (int i = 0; i <= 10; ++i)
    {
        int existingTestSize = testMaxSize * i / 10;
        int randomTestSize = testMaxSize - existingTestSize;
        // copy the existing keys to the new keys
        cudaMemcpy(d_newRandomKeys, d_keys, sizeof(int) * existingTestSize, cudaMemcpyDeviceToDevice);
        // copy the random keys to the new keys
        cudaMemcpy(d_newRandomKeys + existingTestSize, d_queries + existingTestSize, sizeof(int) * randomTestSize, cudaMemcpyDeviceToDevice);
        // do lookups
        lookupItemBatch<<<(testMaxSize + 255) / 256, 256>>>(d_hashTable, d_newRandomKeys, d_retvals, tableSize, testMaxSize, f1, f2);
        // validate the return values

    }
    
    cudaFree(d_keys);
    cudaFree(d_retvals);
    return 0;
}