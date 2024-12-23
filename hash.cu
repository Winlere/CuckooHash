#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <cstring>
#include <map>
#include "hash.cuh"
#include <omp.h>

#define MAX_MOVE_TIME 125
#define NOT_A_KEY -1 // if change this, please change the memset value as well.
#define NOT_A_INDEX -1
class hashTableEntry
{
public:
    int key;
    // int value; no value is needed
};

int initHashTable(hashTableEntry **d_table, int tableSize)
{
    auto err = cudaMalloc(d_table, sizeof(hashTableEntry) * tableSize);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed with error code " << err << std::endl;
        return 1;
    }
    err = cudaMemset(*d_table, 0xff, sizeof(hashTableEntry) * tableSize);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed with error code " << err << std::endl;
        return 1;
    }
    return 0;
}

int reuseHashTable(hashTableEntry *d_table, int tableSize)
{
    auto err = cudaMemset(d_table, 0xff, sizeof(hashTableEntry) * tableSize);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed with error code " << err << std::endl;
        return 1;
    }
    return 0;
}

__device__ void insertItem(hashTableEntry *d_table, int original_key, HashFunc f1, HashFunc f2, int *retval, int OVERRIDE_MAX_MOVE_TIME)
{
    *retval = 0;
    int move_time = 0;
    int h1 = f1(original_key);
    int h2 = f2(original_key);
#ifdef TRIHASH
    HashFunc f3 = {(unsigned)f1(0), f1.tableSize};
    int h3 = f3(original_key);
#endif
    int evicteeIndex = h1;
#ifndef TRIHASH
    if (d_table[h1].key == original_key || d_table[h2].key == original_key)
        return; // Duplicate key
#else
    if (d_table[h1].key == original_key || d_table[h2].key == original_key || d_table[h3].key == original_key)
        return; // Duplicate key
#endif
    // Try to place original_key in the slot
    int current_key = original_key;
    int k1 = atomicExch(&d_table[h1].key, current_key);
    if (k1 == NOT_A_KEY || k1 == current_key)
        return;

    current_key = k1; // Now we work with the evicted key

    do
    { // This block tries to place 'current_key' in the alternative slot
        h1 = f1(current_key);
        h2 = f2(current_key);
#ifndef TRIHASH
        int alternativeIndex = evicteeIndex == h1 ? h2 : h1;
#else
        int h3 = f3(current_key);
        int alternativeIndex = evicteeIndex == h1 ? h2 : (evicteeIndex == h2 ? h3 : h1);
#endif
        k1 = atomicExch(&d_table[alternativeIndex].key, current_key);
        if (k1 == NOT_A_KEY || k1 == current_key)
            return;

        current_key = k1; // Update the current_key with the newly evicted key
        evicteeIndex = alternativeIndex;
        ++move_time;
    } while (move_time < MAX_MOVE_TIME);

    *retval = 1; // Indicate failure after MAX_MOVE_TIME attempts
}

__device__ inline void lookupItem(hashTableEntry *d_table, int key, HashFunc f1, HashFunc f2, int *retval)
{
    *retval = NOT_A_INDEX;
    int h1 = f1(key);
    if (d_table[h1].key == key)
    {
        *retval = d_table[h1].key;
        return;
    }
    int h2 = f2(key);
    if (d_table[h2].key == key)
    {
        *retval = d_table[h2].key;
        return;
    }
#ifdef TRIHASH
    HashFunc f3 = {(unsigned)f1(0), f1.tableSize};
    int h3 = f3(key);
    if (d_table[h3].key == key)
    {
        *retval = d_table[h3].key;
        return;
    }
#endif
    *retval = NOT_A_INDEX;
}

__global__ void insertItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2, int OVERRIDE_MAX_MOVE_TIME)
{
    long long tid = 1ll * blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < batchSize)
    {
        int key = d_keys[tid];
        int tmp = -244;
        insertItem(d_table, key, f1, f2, &tmp, OVERRIDE_MAX_MOVE_TIME);
        d_retvals[tid] = tmp;
    }
}

__global__ void lookupItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batchSize)
        return;
    int key = d_keys[tid];
    int tmp = -233;
    lookupItem(d_table, key, f1, f2, &tmp);
    d_retvals[tid] = tmp;
}

bool validation(int tableSize_, int test_)
{
    std::cout << "[Sanity Check] start sanity check"
              << "tableSize=" << tableSize_ << " test=" << test_ << std::endl;
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int> dist(0, 1e9);
    uint32_t tableSize = tableSize_;
    uint32_t test = test_;
    std::vector<int> keys;
    std::unordered_set<int> keySet;
    for (uint32_t i = 0; i < test; ++i)
    {
        int key = dist(rng);
        if (keySet.find(key) == keySet.end())
        {
            keySet.insert(key);
            keys.push_back(key);
        }
    }
    hashTableEntry *d_table;
    cudaMalloc(&d_table, sizeof(hashTableEntry) * tableSize);
    const int MAX_RETRY = 30;
    int retries = MAX_RETRY;
    HashFunc f1, f2;
    std::map<std::pair<int, int>, int> dup;
    uint32_t a1, a2;
retry:
    cudaMemset(d_table, 0xff, sizeof(hashTableEntry) * tableSize);
    a1 = dist(rng), a2 = dist(rng);
    f1 = HashFunc{a1, tableSize};
    f2 = HashFunc{a2, tableSize};
    dup.clear();
    for (int key : keys)
    {
        int h1 = f1(key);
        int h2 = f2(key);
        dup[{h1, h2}]++;
        // std:: cout << key << " " << h1 << " " << h2 << std::endl;
        if (dup[{h1, h2}] == 3)
        {
            // LOG_DEBUG(("inherent collision detected. h1 = %d h2 = %d",h1,h2));
            for (auto key2 : keys)
            {
                int h1_ = f1(key2); // hash(key2,a1,b1,tableSize);
                int h2_ = f2(key2); // hash(key2,a2,b2,tableSize);
                if (h1 == h1_ && h2 == h2_)
                    std::cout << "[Sanity Check] key,h1,h2 = " << key2 << " " << h1 << " " << h2 << std::endl;
            }
            if (retries--)
            {
                std::cout << "[Sanity Check] retrying..." << std::endl;
                goto retry;
            }
            else
            {
                std::cout << "[Sanity Check] failed after " + std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
        }
    }
    std::cout << "[Sanity Check] no inherent hash collision detected." << std::endl;
    // insert test
    for (int key : keys)
    {
        int *d_retval;
        int *d_key;
        cudaMalloc(&d_key, sizeof(int));
        cudaMemcpy(d_key, &key, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_retval, sizeof(int));
        insertItemBatch<<<1, 1>>>(d_table, d_key, d_retval, tableSize, 1, f1, f2, MAX_MOVE_TIME);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval, d_retval, sizeof(int), cudaMemcpyDeviceToHost);
        if (retval == 1)
        {
            std::cout << "[Sanity Check] insertion failed (or reconstruction needed)" << std::endl;

            if (retries--)
            {
                std::cout << "[Sanity Check] retrying..." << std::endl;
                goto retry;
            }
            else
            {
                std::cout << "[Sanity Check] failed after " + std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
            cudaFree(d_table);
        }
    }
    std::cout << "[Sanity Check] passed insert test." << std::endl;
    // lookup test
    for (int key : keys)
    {
        int *d_retval;
        int *d_key;
        cudaMalloc(&d_key, sizeof(int));
        cudaMemcpy(d_key, &key, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_retval, sizeof(int));
        lookupItemBatch<<<1, 1>>>(d_table, d_key, d_retval, tableSize, 1, f1, f2);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval, d_retval, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_retval);
        if (retval == NOT_A_INDEX)
        {
            std::cout << "[Sanity Check] lookup failed (false negative)" << std::endl;
            std::cout << "[Sanity Check] key = " << key << std::endl;
            std::cout << "[Sanity Check] h1 = " << f1(key) << std::endl;
            std::cout << "[Sanity Check] h2 = " << f2(key) << std::endl;
            std::cout << "[Sanity Check] retval = " << retval << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "[Sanity Check] passed lookup test. no false negative" << std::endl;
    // lookup test
    for (uint32_t i = 0; i < test; ++i)
    {
        int key;
        do
        {
            key = dist(rng);
        } while (keySet.find(key) != keySet.end());
        int *d_retval;
        int *d_key;
        cudaMalloc(&d_key, sizeof(int));
        cudaMemcpy(d_key, &key, sizeof(int), cudaMemcpyHostToDevice);
        cudaMalloc(&d_retval, sizeof(int));
        lookupItemBatch<<<1, 1>>>(d_table, d_key, d_retval, tableSize, 1, f1, f2);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval, d_retval, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_retval);
        if (retval != NOT_A_INDEX)
        {
            std::cout << "[Sanity Check] lookup failed (false positive)" << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "[Sanity Check] passed lookup test. no false positive" << std::endl;
    std::cout << "[Sanity Check] passed all sanity tests!" << std::endl;
    // release resource
    cudaFree(d_table);
    return true;
}

__global__ void generateRandomKeys(int *d_keys, int batchSize, int range, uint32_t seed)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batchSize)
        return;
    HashFunc f{seed + tid, (uint32_t)range};
    d_keys[tid] = f(tid);
}


void lookupItemBatchStreamed(hashTableEntry *d_table, int *h_keys, int* h_retvals, int batchSize, int tableSize, HashFunc f1, HashFunc f2){
    static const int streamCount = 1;
    static const int tileSize = batchSize;
    static const int BLOCKSIZE = 256;
    // tileSize / BLOCKSIZE
    cudaStream_t streams[streamCount];
    int* d_keys_in_stream[streamCount];
    int* d_retvals_in_stream[streamCount];
    for(int i = 0; i < streamCount; ++i){
        cudaStreamCreate(&streams[i]);
        cudaMalloc(&d_keys_in_stream[i], sizeof(int)*tileSize);
        cudaMalloc(&d_retvals_in_stream[i], sizeof(int)*tileSize);
    }
    omp_set_num_threads(streamCount);
    int tileNum = (batchSize + tileSize - 1) / tileSize;
    #pragma omp parallel for
    for(int j = 0; j < tileNum; ++j){
        int i = j * tileSize;
        int stream_id = (i / tileSize) % streamCount;
        int actualTileSize = std::min(tileSize, batchSize - i);
        cudaStream_t this_stream = streams[stream_id];
        int* d_keys_this_tile = d_keys_in_stream[stream_id];
        int* d_retvals_this_tile = d_retvals_in_stream[stream_id];
        int* h_keys_this_tile = h_keys + i;
        int* h_retvals_this_tile = h_retvals + i;
        cudaMemcpyAsync(d_keys_this_tile, h_keys_this_tile, actualTileSize * sizeof(int), 
                    cudaMemcpyHostToDevice, this_stream);
        lookupItemBatch<<<(actualTileSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE, 0 ,this_stream>>>
            (d_table, d_keys_this_tile, d_retvals_this_tile, tableSize, actualTileSize, f1, f2);
        cudaMemcpyAsync(h_retvals_this_tile, d_retvals_this_tile, actualTileSize * sizeof(int), 
                    cudaMemcpyDeviceToHost, this_stream);
    }
    for(int i = 0; i < streamCount; ++i){
        cudaFree(&d_keys_in_stream[i]);
        cudaFree(&d_retvals_in_stream[i]);
    }
    cudaDeviceSynchronize();
}