#include "hash.cuh"
#include <iostream>
int main()
{
    const int sanityCheck = 10;
    std::cout << "[Sanity Check] start " + std::to_string(sanityCheck) + " rounds of sanity check" << std::endl;
    int success = 0;
    for (int i = 1; i <= sanityCheck; ++i)
    {
        success += validation(1 << 25, 1 << int(10 + 1. * i / sanityCheck * (20 - 10)));
    }
    std::cout << "[Sanity Check] passed " << success << " rounds out of " + std::to_string(sanityCheck) + " sanity check(s)" << std::endl;
    return 0;
}