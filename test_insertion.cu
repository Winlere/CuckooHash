#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>
#include <string>
const int BLOCKSIZE = 512;
int maxRetry = 50;

int main(int argc, char **argv)
{
    int seed = argc >= 2 ? std::stoi(std::string(argv[1])) : 0;
    int targetS = argc >= 3 ? std::stoi(std::string(argv[2])) : -1;

    TIME_INIT;
    const uint32_t tableSize = 1 << 25;
    const uint32_t testMaxSize = 1 << 24;
    const uint32_t seed1 = 114514;
    const uint32_t seed2 = 191981;
    const int range = 1 << 30;
    HashFunc f1, f2;
    hashTableEntry *d_hashTable = nullptr;
    initHashTable(&d_hashTable, tableSize);

    int *d_keys = nullptr;
    cudaMalloc((void **)&d_keys, sizeof(int) * tableSize);
    cudaMemset(d_keys, 0xff, sizeof(int) * tableSize);
    generateRandomKeys<<<(testMaxSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_keys, tableSize, range);
    cudaDeviceSynchronize();

    int *d_retvals = nullptr;
    cudaMalloc((void **)&d_retvals, sizeof(int) * tableSize);

    for (uint32_t s = 11; s <= 24; ++s)
    {
        if (targetS != -1 && s != targetS)
            continue;
        f1.seed = 115838 << 10 | seed;
        f1.tableSize = 4194304;
        f2.seed = 193305 << 10 | seed;
        f2.tableSize = 4194304;
    retry_entry:
        int testSize = 1 << s;
        reuseHashTable(d_hashTable, tableSize);
        TIME_START;
        insertItemBatch<<<(testSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2, MAX_MOVE_TIME);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
        if (!valid)
        {
            ++maxRetry;
            f1 = {seed1 + maxRetry, tableSize};
            f2 = {seed2 + maxRetry, tableSize};
            std::cout << "retrying..." << std::endl;
            goto retry_entry;
        }
        if(argc == 3){
            std::cout << elapsed_μs/1e6 << " " << 1.0 * testSize / elapsed_μs << std::endl;
        }else
            std::cout << "testsize,elapsed_μs,valid | " << testSize << "," << elapsed_μs << "," << valid << std::endl;
    }
    cudaFree(d_keys);
    cudaFree(d_retvals);
    cudaFree(d_hashTable);
    return 0;
}