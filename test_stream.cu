#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <omp.h>
#include <string>
#include <random>

#define LOGGER(X) std::cerr << __FILE__ << ":" << __LINE__ << " " << (X) << std::endl;

int retry_times = 900;
int main(int argc, char const *argv[])
{
    int seed = argc >=2 ? std::stoi(argv[1]) : 0;
    TIME_INIT;
#ifdef TRIHASH
    const uint32_t tableSize = 1 << 25;
#else
    const uint32_t tableSize = 1 << 26;
#endif
    const uint32_t testMaxSize = 1 << 24;
    uint32_t seed1 = 114514 + seed;
    uint32_t seed2 = 191981 + seed;
    const int range = 1 << 30;
    hashTableEntry *d_hashTable = nullptr;
    initHashTable(&d_hashTable, tableSize);

    int *d_keys = nullptr;
    cudaMalloc((void **)&d_keys, sizeof(int) * tableSize);
    cudaMemset(d_keys, 0xff, sizeof(int) * tableSize);
    generateRandomKeys<<<(testMaxSize + 255) / 256, 256>>>(d_keys, tableSize, range, seed1);
    cudaDeviceSynchronize();
    
    HashFunc f1{seed1 + retry_times, tableSize}, f2{seed2 + retry_times, tableSize};
    LOGGER(1);
    int *d_retvals = nullptr;
    cudaMalloc((void **)&d_retvals, sizeof(int) * tableSize);
    {
        int testSize;
    retry:
        f1 = HashFunc{seed1 + retry_times, tableSize};
        f2 = HashFunc{seed2 + retry_times, tableSize};
        testSize = testMaxSize;
        reuseHashTable(d_hashTable, tableSize);
        TIME_START;
        insertItemBatch<<<(testSize + 255) / 256, 256>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2, MAX_MOVE_TIME);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
        if (valid){
            std::cout << "construction sucessfull" << std::endl;
        }else
        {
            std::cout << "failed. reconstructing..." << std::endl;
            ++retry_times;
            goto retry;
        }
        // report sucess hash parameters
        std::cout << "report sucess hash parameters" << std::endl;
        std::cout << "f1.seed = " << f1.seed << " f1.tableSize = " << f1.tableSize << std::endl;
        std::cout << "f2.seed = " << f2.seed << " f2.tableSize = " << f2.tableSize << std::endl;
    }
    LOGGER(2);

    // random shuffle the old keys. to ensure uniformity
    std::vector<int> oldKeys(testMaxSize);
    cudaMemcpy(oldKeys.data(), d_keys, sizeof(int) * testMaxSize, cudaMemcpyDeviceToHost);
    std::random_shuffle(oldKeys.begin(), oldKeys.end());
    std::unordered_set<int> oldKeysSet(oldKeys.begin(), oldKeys.end());
    cudaMemcpy(d_keys, oldKeys.data(), sizeof(int) * testMaxSize, cudaMemcpyHostToDevice);
    LOGGER(3);

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


    int *h_mixedKeys, *h_retvals; 
    cudaMallocHost(&h_mixedKeys, sizeof(int) * testMaxSize);
    cudaMallocHost(&h_retvals, sizeof(int) * testMaxSize);
    
    
    uint64_t totalμs = 0;
    uint64_t totalops = testMaxSize * 11;
    for (int i = 5; i <= 5; ++i)
    {
        int existingTestSize = std::max(testMaxSize * i / 10, 1u);
        int randomTestSize = testMaxSize - existingTestSize;

        std::copy(oldKeys.begin(), oldKeys.begin() + existingTestSize, h_mixedKeys);
        LOGGER("AFTER OLDKEY COPY");
        std::copy(h_newRandomKeys.begin(), h_newRandomKeys.begin() + randomTestSize, h_mixedKeys + existingTestSize);
        LOGGER("AFTER NOKEY COPY");
        
        std::cout << "here" << std::endl; 
        TIME_START;
        lookupItemBatchStreamed(d_hashTable, h_mixedKeys, h_retvals, testMaxSize, 
                        tableSize, f1, f2);
        TIME_END;
        
        bool valid1 = std::all_of(h_retvals, h_retvals + existingTestSize, [](int x){
                        return x != NOT_A_INDEX;});
        bool valid2 = std::all_of(h_retvals + existingTestSize, h_retvals, [](int x){
                        return x == NOT_A_INDEX;});
        bool valid = valid1 && valid2;
        std::cout << "valid1,2,#=" << valid1 << ',' << valid2 << ',' << valid << std::endl;
        if(argc != 2)
            std::cout << "percentage,elapsed_μs,valid | " << i << ',' << elapsed_μs << "," << valid << std::endl;
        totalμs += elapsed_μs;
    }
    if(argc == 2){
        std::cout << totalμs/1e6 <<' ' << 1.0 * totalops / totalμs << std::endl;
    }
    cudaFree(d_keys);
    cudaFree(d_retvals);
    cudaFree(d_queries);
    cudaFree(d_newRandomKeys);
    cudaFree(d_hashTable);
    cudaFree(h_mixedKeys);
    cudaFree(h_retvals);
    return 0;
}