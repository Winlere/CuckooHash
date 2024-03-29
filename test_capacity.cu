#include "hash.cuh"
#include "time.hpp"
#include "helper.cuh"
#include <iostream>
#include <vector>

int main(int argc, char const *argv[])
{
    uint32_t seed = argc >= 2 ? atoi(argv[1]) : 0;
    double percentageS = argc >= 3 ? atof(argv[2]) : -1;
    TIME_INIT;
    const uint32_t maxTableSize = 1 << 25;
    const uint32_t testMaxSize = 1 << 24;
    uint32_t seed1 = 114514 + seed;
    uint32_t seed2 = 191981 + seed;
    const int range = 1 << 30;
    int retries = 0;
    hashTableEntry *d_hashTable = nullptr;
    initHashTable(&d_hashTable, maxTableSize);

    int *d_keys = nullptr;
    cudaMalloc((void **)&d_keys, sizeof(int) * maxTableSize);
    cudaMemset(d_keys, 0xff, sizeof(int) * maxTableSize);
    generateRandomKeys<<<(maxTableSize + 255) / 256, 256>>>(d_keys, maxTableSize, range);
    cudaDeviceSynchronize();
    HashFunc f1, f2;

    int *d_retvals = nullptr;
    cudaMalloc(&d_retvals, sizeof(int) * maxTableSize);
    {

        uint32_t testSize = testMaxSize;
        
        std::vector<int> tableSizeProportions = {110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 101, 102, 105};
        if(percentageS > 0){
            tableSizeProportions.clear();
            tableSizeProportions.push_back(percentageS * 100);
        }
        for (int proportion : tableSizeProportions)
        {
            uint32_t tableSize = testMaxSize * proportion / 100;
            retries = 0;
            f1 = {seed1 + retries, tableSize}, f2 = {seed2 + retries, tableSize};
            reuseHashTable(d_hashTable, tableSize);
            TIME_START;
            insertItemBatch<<<(testSize + 256 - 1) / 256, 256>>>(d_hashTable, d_keys, d_retvals, tableSize, testSize, f1, f2, MAX_MOVE_TIME);
            cudaDeviceSynchronize();
            TIME_END;
            // // print the first 100000 values of d_retvals
            // int *h_retvals = new int[testMaxSize];
            // cudaMemcpy(h_retvals, d_retvals, sizeof(int) * testMaxSize, cudaMemcpyDeviceToHost);
            // std::cout << "d_retvals[0:9] = ";
            // for (int i = 0; i < 128; i++)
            // {
            //     std::cout << h_retvals[i] << " ";
            // }
            // std::cout << std::endl;
            // delete [] h_retvals;
            bool valid = isArrayAllEqualToValue(d_retvals, (int)testMaxSize, 0);
            // bool valid = 1;
            cudaDeviceSynchronize();
            if(tableSizeProportions.size() != 1)
                std::cout << "tableSize,elapsed_μs,valid | " << tableSize << "," << elapsed_μs << "," << valid << std::endl;
            else
                std::cout << elapsed_μs / 1e6 << ' ' << 1.0 * (1<<24) / elapsed_μs << std::endl;
        }
    }
    cudaFree(d_keys);
    cudaFree(d_retvals);
    cudaFree(d_hashTable);
    return 0;
}
