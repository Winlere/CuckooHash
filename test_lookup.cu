#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <omp.h>
#include <random>

int retry_times = 900;
int main()
{
    TIME_INIT;
#ifdef TRIHASH
    const uint32_t tableSize = 1 << 25;
#else
    const uint32_t tableSize = 1 << 26;
#endif
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

    HashFunc f1{seed1 + retry_times, tableSize}, f2{seed2 + retry_times, tableSize};

    int *d_retvals = nullptr;
    cudaMalloc((void **)&d_retvals, sizeof(int) * tableSize);
    {
        int testSize;
    retry:
        f1 = {seed1 + retry_times, tableSize};
        f2 = {seed2 + retry_times, tableSize};
        testSize = testMaxSize;
        reuseHashTable(d_hashTable, tableSize);
        TIME_START;
        insertItemBatch<<<(testSize + 255) / 256, 256>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2, MAX_MOVE_TIME);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
        if (valid){
            // std::cout << "construction sucessfull" << std::endl;
        }else
        {
            // std::cout << "failed. reconstructing..." << std::endl;
            ++retry_times;
            goto retry;
        }
        // report sucess hash parameters
        // std::cout << "report sucess hash parameters" << std::endl;
        // std::cout << "f1.seed = " << f1.seed << " f1.tableSize = " << f1.tableSize << std::endl;
        // std::cout << "f2.seed = " << f2.seed << " f2.tableSize = " << f2.tableSize << std::endl;
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
#pragma omp parallel for shared(testMaxSize, range, h_newRandomKeys,oldKeysSet) schedule(dynamic)
    for (uint32_t i = 0; i < testMaxSize; ++i)
    {
        std::mt19937 rng(i);
        std::uniform_int_distribution<int> uni(0, range);
        int key = uni(rng);
        while (oldKeysSet.find(key) != oldKeysSet.end())
        {
            key = uni(rng);
        }
        h_newRandomKeys[i] = key;
    }
    int *d_newRandomKeys = nullptr;
    cudaMalloc((void **)&d_newRandomKeys, sizeof(int) * testMaxSize);
    cudaMemcpy(d_newRandomKeys, h_newRandomKeys.data(), sizeof(int) * testMaxSize, cudaMemcpyHostToDevice);
    for (int i = 0; i <= 10; ++i)
    {
        int existingTestSize = std::max(testMaxSize * i / 10, 1u);
        int randomTestSize = testMaxSize - existingTestSize;
        // copy the existing keys to the new keys
        cudaMemcpy(d_queries, d_keys, sizeof(int) * existingTestSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_queries + existingTestSize, d_newRandomKeys, sizeof(int) * randomTestSize, cudaMemcpyDeviceToDevice);
        // do lookups
        TIME_START;
        lookupItemBatch<<<(testMaxSize + 512 - 1) / 512, 512>>>(d_hashTable, d_queries, d_retvals, tableSize, testMaxSize, f1, f2);
        cudaDeviceSynchronize();
        TIME_END;
        // validate the return values
        // bool valid = isArrayAllNotEqualToValue(d_retvals, existingTestSize, NOT_A_INDEX) && isArrayAllEqualToValue(d_retvals + existingTestSize, randomTestSize, NOT_A_INDEX);
        bool valid1 = isArrayAllNotEqualToValue(d_retvals, existingTestSize, NOT_A_INDEX);
        bool valid2 = isArrayAllEqualToValue(d_retvals + existingTestSize, randomTestSize, NOT_A_INDEX);
        bool valid = valid1 && valid2;
        // std::cout << "valid1,2,#=" << valid1 << ',' << valid2 << ',' << valid << std::endl;
        std::cout << "percentage,elapsed_μs,valid | " << i << ',' << elapsed_μs << "," << valid << std::endl;
    }

    cudaFree(d_keys);
    cudaFree(d_retvals);
    cudaFree(d_queries);
    cudaFree(d_newRandomKeys);
    cudaFree(d_hashTable);
    return 0;
}