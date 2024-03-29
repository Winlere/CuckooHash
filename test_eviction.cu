#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <cmath>
#include <vector>
const int BLOCKSIZE = 512;
int maxRetry = 50;

int main(int argc,char **argv)
{
    uint32_t seed = argc >= 2 ? atoi(argv[1]) : 0;
    double percentageS = argc >= 3 ? atof(argv[2]) : -1;
    TIME_INIT;
    const uint32_t tableSize = 1 << 25;
    const uint32_t testMaxSize = 1 << 24;
    uint32_t seed1 = 114514 + seed;
    uint32_t seed2 = 191981 + seed;
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
    int testTableSize = 1.4 * testMaxSize;
    for (int maxMove = 1; maxMove <= 125; ++maxMove)
    {
        if(percentageS > 0){
            maxMove = (int) (percentageS * std::log2(testTableSize));
        }
        f1.seed = 115838 + seed;
        f1.tableSize = testTableSize;
        f2.seed = 193305 + seed;
        f2.tableSize = testTableSize;
    retry_entry:
        int testSize = 1 << 24;
        reuseHashTable(d_hashTable, testTableSize);
        TIME_START;
        insertItemBatch<<<(testSize + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE>>>(d_hashTable, d_keys, d_retvals, testTableSize, testSize, f1, f2, maxMove);
        cudaDeviceSynchronize();
        TIME_END;
        bool valid = isArrayAllEqualToValue(d_retvals, testSize, 0);
        if (!valid)
        {
            ++maxRetry;
            f1 = {seed1 + maxRetry, (unsigned) testTableSize};
            f2 = {seed2 + maxRetry, (unsigned) testTableSize};
            // std::cout << "retrying..." << std::endl;
            goto retry_entry;
        }
        if(percentageS < 0)
            std::cout << "maxMove,elapsed_μs,valid | " << maxMove << "," << elapsed_μs << "," << valid << std::endl;
        else{
            std::cout << elapsed_μs / 1e6 << ' ' << 1.0 * testMaxSize / elapsed_μs << std::endl;
            break;
        }
    }
    cudaFree(d_keys);
    cudaFree(d_retvals);
    cudaFree(d_hashTable);
    return 0;
}