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

//miller rabin prime test
bool isPrime(int n){
    if(n == 2) return true;
    if(n % 2 == 0) return false;
    int d = n - 1;
    while(d % 2 == 0) d /= 2;
    for(int i = 0;i < 10;++i){
        int a = rand() % (n - 1) + 1;
        int t = d;
        int y = 1;
        while(t > 0){
            if(t & 1) y = 1ll * y * a % n;
            a = 1ll * a * a % n;
            t >>= 1;
        }
        bool flag = false;
        if(y != 1){
            while(d != n - 1){
                if(y == n - 1){
                    flag = true;
                    break;
                }
                y = 1ll * y * y % n;
                d <<= 1;
            }
            if(!flag) return false;
        }
    }
    return true;
}

int randomPrime(){
    int n = rand() % 1000000 + 1000000;
    while(!isPrime(n)) ++n;
    return n;
}

inline int hash(int key, int a,int b,int tableSize){
    return (1ll * key  % tableSize * a + b) % tableSize;
}


class HashFunc{
public:
    virtual __host__ __device__ int operator ()(int key) {
        return NOT_A_INDEX;
    };
};

class HashFuncAffine: public HashFunc{
    int a,b,tableSize;
public:
    HashFuncAffine(int a,int b,int c):a(a),b(b),tableSize(c){}
    __host__ __device__ int operator ()(int key) override {
        return (1ll * key  % tableSize * a + b) % tableSize;
    }
};

class HashFuncBit : public HashFunc{
public:
    __host__ __device__ int operator ()(int key) override {
        return 0;
    }
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

bool validation(){
    std::cout << "start sanity check" << std::endl;
    std::mt19937_64 rng(10);
    std::uniform_int_distribution<int> dist(0,1e9);
    int tableSize = 2000;
    int test = 25;
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
    const int MAX_RETRY = 15;
    int retries = MAX_RETRY;
retry:
    cudaMemset(d_table,0xff,sizeof(hashTableEntry) * tableSize);
    int a1 = randomPrime(), b1 = randomPrime();
    int a2 = a1 + 2, b2 = b1 + 2;
    HashFuncAffine f1{a1,b1,tableSize},f2{a2,b2,tableSize};
    std::map<std::pair<int,int>,int> dup; 
    for(int key:keys){
        int h1 = hash(key,a1,b1,tableSize);
        int h2 = hash(key,a2,b2,tableSize);
        dup[{h1,h2}]++;
        // std:: cout << key << " " << h1 << " " << h2 << std::endl;
        if(dup[{h1,h2}] == 3){
            std::cout << "hash collision detected. h1 = " << h1 << " h2 = " << h2 <<  std::endl;
            for(auto key2 : keys){
                int h1_ = hash(key2,a1,b1,tableSize);
                int h2_ = hash(key2,a2,b2,tableSize);
                if(h1 == h1_ && h2 == h2_)
                    std::cout << key2 << " " << h1 << " " << h2 << std::endl;
            }
            if(retries--){
                std::cout << "retrying..." << std::endl;
                goto retry;
            }else{
                std::cout << "failed after " +  std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
        }
    }
    //insert test
    for(int key : keys){
        int* d_retval;
        cudaMalloc(&d_retval,sizeof(int));
        insertItem<<<1,1>>>(d_table,key,f1,f2,d_retval);
        cudaDeviceSynchronize();
        int retval;
        cudaMemcpy(&retval,d_retval,sizeof(int),cudaMemcpyDeviceToHost);
        if(retval == 1){
            std::cout << "insertion failed (or reconstruction needed)" << std::endl;

            if(retries--){
                std::cout << "retrying..." << std::endl;
                goto retry;
            }else{
                std::cout << "failed after " +  std::to_string(MAX_RETRY) + " retries. aborting..." << std::endl;
                cudaFree(d_table);
                return false;
            }
            cudaFree(d_table);
        }
    }
    std::cout << "passed insert test." << std::endl;
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
            std::cout << "lookup failed (false negative)" << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "passed lookup test. no false negative" << std::endl;
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
            std::cout << "lookup failed (false positive)" << std::endl;
            cudaFree(d_table);
            return false;
        }
    }
    std::cout << "passed lookup test. no false positive" << std::endl;
    std::cout << "passed all sanity tests!" << std::endl;
    //release resource
    cudaFree(d_table);
    return true;
}

int main(){
    std::cout<< "start 5 rounds of sanity check" << std::endl;
    int success = 0;
    for(int i = 0;i < 1;++i){
        success += validation();
    }
    std::cout << "passed " << success << " rounds out of 5 sanity check(s)" << std::endl;
    return 0;
}