#pragma ONCE

class hashTableEntry;
class HashFunc;
__device__ inline void insertItem(hashTableEntry *d_table, int original_key, HashFunc f1, HashFunc f2, int *retval);
__device__ inline void lookupItem(hashTableEntry *d_table, int key, HashFunc f1, HashFunc f2, int *retval);
__global__ void insertItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2);
__global__ void lookupItemBatch(hashTableEntry *d_table, int *d_keys, int *d_retvals, int tableSize, int batchSize, HashFunc f1, HashFunc f2);
bool validation(int tableSize_ = 1 << 25, int test_ = 1 << 19);