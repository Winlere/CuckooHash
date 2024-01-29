# Makefile for compiling CUDA test files

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -g -G -Xcompiler -Wall -Xcompiler -fopenmp -DTRIHASH

# CUDA test files
TEST_FILES = test_sanity.cu test_insertion.cu test_lookup.cu test_capacity.cu test_eviction.cu

# other source files
OTHER_CU_FILES = hash.cu

# Object files
OBJECTS = $(TEST_FILES:.cu=.o) $(OTHER_CU_FILES:.cu=.o)

# Executable files
EXECUTABLES = $(TEST_FILES:.cu=)

# Defines
DEFINES = -DTRIHASH

# Default target
all: $(EXECUTABLES)

# Compile .cu files into object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link object files into executables
$(EXECUTABLES): % : %.o $(OTHER_CU_FILES:.cu=.o)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# helper files

$(OBJECTS): helper.cuh

# Clean up
clean:
	rm -f $(OBJECTS) $(EXECUTABLES)

.PHONY: all clean
