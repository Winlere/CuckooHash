#pragma once
#include <chrono>

// Initialize start and end times
#define TIME_INIT \
    auto start = std::chrono::high_resolution_clock::now(); \
    auto end = std::chrono::high_resolution_clock::now(); \
    int64_t elapsed_μs = 0;

// Record the start time
#define TIME_START \
    start = std::chrono::high_resolution_clock::now();

// Record the end time, calculate elapsed time in milliseconds
#define TIME_END                                           \
    end = std::chrono::high_resolution_clock::now();       \
    elapsed_μs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
