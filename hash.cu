#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <unordered_set>
#include <cstring>
#include <map>
#define MAX_MOVE_TIME 100
#define NOT_A_KEY -1 //if change this, please change the memset value in validation() and other functions as well
#define NOT_A_INDEX -1 

struct hashTableEntry{
    int key;
    // int value; no value is needed
};

/*The following hash function is adapted from https://github.com/easyaspi314/xxhash-clean/blob/master/xxhash32-ref.c */
static uint32_t const PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static uint32_t const PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */
static uint32_t const PRIME32_3 = 0xC2B2AE3DU;   /* 0b11000010101100101010111000111101 */
static uint32_t const PRIME32_4 = 0x27D4EB2FU;   /* 0b00100111110101001110101100101111 */
static uint32_t const PRIME32_5 = 0x165667B1U;   /* 0b00010110010101100110011110110001 */

struct HashFunc{
    int seed;
    int tableSize;
    /*https://github.com/easyaspi314/xxhash-clean/blob/86a04ab3f01277049a23f6c9e2c4a6c174ff50c4/xxhash32-ref.c#L97 */
    __host__ __device__ int operator ()(int key) {
        uint32_t ret = PRIME32_5 + seed;
        uint8_t const* p = reinterpret_cast<uint8_t const*>(&key);
        for(int i = 0;i < 4;++i){
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


__global__ void insertItem(hashTableEntry* d_table, int key, HashFunc f1,HashFunc f2,int* retval){
    *retval = 0;
    int move_time = 0;
    int tmpBannedSlot = -1;
    do{
        int h1 = f1(key);
        int h2 = f2(key);
        if(d_table[h1].key == key || d_table[h2].key == key) return;
        //try to place key in the slot
        if(tmpBannedSlot != 1 && d_table[h1].key == NOT_A_KEY && atomicCAS(&d_table[h1].key,NOT_A_KEY,key) == NOT_A_KEY)
            return;
        if(tmpBannedSlot != 2 && d_table[h2].key == NOT_A_KEY && atomicCAS(&d_table[h2].key,NOT_A_KEY,key) == NOT_A_KEY)
            return;
        //if there is no available slot. evict k1 in the h1
        int k1 = atomicExch(&d_table[h1].key,key);
        //set a lock to prevent self-looping
        tmpBannedSlot = 1 + (f2(k1) == h1);
        //start over
        key = k1;
        ++move_time;
    }while(move_time < MAX_MOVE_TIME);
    *retval = 1;
}

__global__ void lookupItem(hashTableEntry* d_table, int key, HashFunc f1, HashFunc f2, int* retval){
    *retval = NOT_A_INDEX;
    int h1 = f1(key);
    if(d_table[h1].key == key){
        *retval = d_table[h1].key;
        return;
    }
    int h2 = f2(key);
    if(d_table[h2].key == key){
        *retval = d_table[h2].key;
        return;
    }
    *retval = NOT_A_INDEX;
}

bool validation(int tableSize_ = 1 << 25, int test_  = 1 << 19){
    std::cout << "[Sanity Check] start sanity check" << "tableSize=" << tableSize_ << " test=" << test_ << std::endl;
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<int> dist(0,1e9);
    int tableSize = tableSize_;
    int test = test_;
    std::vector<int> keys;
    std::unordered_set<int> keySet;
    for(int i = 0;i < test;++i){
        int key = dist(rng);
        if(keySet.find(key) == keySet.end()){
            keySet.insert(key);
            keys.push_back(key);
        }
    }
    hashTableEntry* d_table;
    cudaMalloc(&d_table,sizeof(hashTableEntry) * tableSize);
    const int MAX_RETRY = 30;
    int retries = MAX_RETRY;
retry:
    cudaMemset(d_table,0xff,sizeof(hashTableEntry) * tableSize);
    int a1 = dist(rng), a2 = dist(rng);
    HashFunc f1{a1,tableSize},f2{a2,tableSize};
    std::map<std::pair<int,int>,int> dup; 
    for(int key:keys){
        int h1 = f1(key);
        int h2 = f2(key);
        dup[{h1,h2}]++;
        // std:: cout << key << " " << h1 << " " << h2 << std::endl;
        if(dup[{h1,h2}] == 3){
            std::cout << "[Sanity Check] inherent collision detected. h1 = " << h1 << " h2 = " << h2 <<  std::endl;
            for(auto key2 : keys){
                int h1_ = f1(key2); // hash(key2,a1,b1,tableSize);
                int h2_ = f2(key2); // hash(key2,a2,b2,tableSize);
                if(h1 == h1_ && h2 == h2_)
                    std::cout << "[Sanity Check] key,h1,h2 = " <<  key2 << " " << h1 << " " << h2 << std::endl;
            }
            if(retries--){
                std::cout << "[Sanity Check] retrying..." << std::endl;
                goto retry;
            }else{
                std::cout << "[Sanity Check] failed after " +  std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
        }
    }
    std::cout<<"[Sanity Check] no inherent hash collision detected." << std::endl;
    //insert test
    for(int key : keys){
        int* d_retval;
        cudaMalloc(&d_retval,sizeof(int));
        insertItem<<<1,1>>>(d_table,key,f1,f2,d_retval);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval,d_retval,sizeof(int),cudaMemcpyDeviceToHost);
        if(retval == 1){
            std::cout << "[Sanity Check] insertion failed (or reconstruction needed)" << std::endl;

            if(retries--){
                std::cout << "[Sanity Check] retrying..." << std::endl;
                goto retry;
            }else{
                std::cout << "[Sanity Check] failed after " +  std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
            cudaFree(d_table);
        }
    }
    std::cout << "[Sanity Check] passed insert test." << std::endl;
    //lookup test
    for(int key : keys){
        int* d_retval;
        cudaMalloc(&d_retval,sizeof(int));
        lookupItem<<<1,1>>>(d_table,key,f1,f2,d_retval);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval,d_retval,sizeof(int),cudaMemcpyDeviceToHost);
        cudaFree(d_retval);
        if(retval == NOT_A_INDEX){
            std::cout << "[Sanity Check] lookup failed (false negative)" << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "[Sanity Check] passed lookup test. no false negative" << std::endl;
    //lookup test
    for(int i = 0;i < test; ++i){
        int key;
        do{
            key = dist(rng);
        }while(keySet.find(key) != keySet.end());
        int* d_retval;
        cudaMalloc(&d_retval,sizeof(int));
        lookupItem<<<1,1>>>(d_table,key,f1,f2,d_retval);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval,d_retval,sizeof(int),cudaMemcpyDeviceToHost);
        cudaFree(d_retval);
        if(retval != NOT_A_INDEX){
            std::cout << "[Sanity Check] lookup failed (false positive)" << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "[Sanity Check] passed lookup test. no false positive" << std::endl;
    std::cout << "[Sanity Check] passed all sanity tests!" << std::endl;
    //release resource
    cudaFree(d_table);
    return true;
}

int main(){
    const int sanityCheck = 10;
    std::cout<< "[Sanity Check] start " +  std::to_string(sanityCheck) + " rounds of sanity check" << std::endl;
    int success = 0;
    for(int i = 1;i <= sanityCheck;++i){
        success += validation(1 << 25 , 1 << int(10 + 1. * i / sanityCheck * (20 - 10)));
    }
    std::cout << "[Sanity Check] passed " << success << " rounds out of " + std::to_string(sanityCheck) + " sanity check(s)" << std::endl;
    return 0;
}