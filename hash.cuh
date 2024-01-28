#pragma once
#include <iostream>
class hashTableEntry;
#define NOT_A_KEY -1 // if change this, please change the memset value as well.
#define NOT_A_INDEX -1

/*The following hash function is adapted from https://github.com/easyaspi314/xxhash-clean/blob/master/xxhash32-ref.c */
static uint32_t const PRIME32_1 = 0x9E3779B1U; /* 0b10011110001101110111100110110001 */
static uint32_t const PRIME32_2 = 0x85EBCA77U; /* 0b10000101111010111100101001110111 */
static uint32_t const PRIME32_3 = 0xC2B2AE3DU; /* 0b11000010101100101010111000111101 */
static uint32_t const PRIME32_4 = 0x27D4EB2FU; /* 0b00100111110101001110101100101111 */
static uint32_t const PRIME32_5 = 0x165667B1U; /* 0b00010110010101100110011110110001 */
class HashFunc
{
public:
    uint32_t seed;
    uint32_t tableSize;
    /*https://github.com/easyaspi314/xxhash-clean/blob/86a04ab3f01277049a23f6c9e2c4a6c174ff50c4/xxhash32-ref.c#L97 */
    __host__ __device__ int operator()(int key)
    {
        uint32_t ret = PRIME32_5 + seed;
        uint8_t const *p = reinterpret_cast<uint8_t const *>(&key);
        for (int i = 0; i < 4; ++i)
        {
            ret += (*p++) * PRIME32_3;
            ret = (ret << 17) | (ret >> 15);
            ret *= PRIME32_4;
        }
        ret ^= ret >> 15;
        ret *= PRIME32_2;
        ret ^= ret >> 13;
        ret *= PRIME32_3;
        ret ^= ret >> 16;
        return ret % tableSize;
    };
};
int initHashTable(hashTableEntry ** d_table, int tableSize);
int reuseHashTable(hashTableEntry ** d_table, int tableSize);
__global__ void generateRandomKeys(int *d_keys, int batchSize, int range, uint32_t seed = 114514);
__device__ inline void insertItem(hashTableEntry *d_table, int original_key, HashFunc f1, HashFunc f2, int *retval);
__device__ inline void lookupItem(hashTableEntry *d_table, int key, HashFunc f1, HashFunc f2, int *retval);
__global__ void insertItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2);
__global__ void lookupItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2);
bool validation(int tableSize_ = 1 << 25, int test_ = 1 << 19);