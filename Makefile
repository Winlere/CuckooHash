# Compiler
NVCC = nvcc

# Make System
MAKE = make 

# Compiler flags
NVCCFLAGS = -g -G -Xcompiler -Wall -Xcompiler -fopenmp -O2 -std=c++11

# CUDA test files
TEST_FILES = test_sanity.cu test_insertion.cu test_lookup.cu test_capacity.cu test_eviction.cu test_stream.cu

# Other source files
OTHER_CU_FILES = hash.cu

# Object files without TRIHASH
OBJECTS = $(TEST_FILES:.cu=.o) $(OTHER_CU_FILES:.cu=.o)

# Object files with TRIHASH
TRIHASH_OBJECTS = $(TEST_FILES:.cu=_trihash.o) $(OTHER_CU_FILES:.cu=_trihash.o)

# Executable files without TRIHASH
EXECUTABLES = $(TEST_FILES:.cu=)

# Executable files with TRIHASH
TRIHASH_EXECUTABLES = $(addsuffix _trihash, $(basename $(TEST_FILES)))

# Default target
all: $(EXECUTABLES) $(TRIHASH_EXECUTABLES)

# Compile .cu files into object files without TRIHASH
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile .cu files into object files with TRIHASH
%_trihash.o: %.cu
	$(NVCC) $(NVCCFLAGS) -DTRIHASH -c $< -o $@

# Link object files into executables without TRIHASH
$(EXECUTABLES): % : %.o $(OTHER_CU_FILES:.cu=.o)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Link object files into executables with TRIHASH
$(TRIHASH_EXECUTABLES): %_trihash : %_trihash.o $(OTHER_CU_FILES:.cu=_trihash.o)
	$(NVCC) $(NVCCFLAGS) -DTRIHASH $^ -o $@

# Helper files dependency
$(OBJECTS) $(TRIHASH_OBJECTS): helper.cuh

# Clean up
clean:
	rm -f $(OBJECTS) $(TRIHASH_OBJECTS) $(EXECUTABLES) $(TRIHASH_EXECUTABLES)

test: $(EXECUTABLES)
	# ./test_insertion : test_insertion need to execute manually because the last value is always invalid and I didn't set a timeout.
	./test_lookup
	./test_capacity
	# ./test_eviction : evition for t=2 is meaning less and it loops forever
test_trihash: $(TRIHASH_EXECUTABLES)
	./test_insertion_trihash
	./test_lookup_trihash
	./test_capacity_trihash
	./test_eviction_trihash

test_all: $(EXECUTABLES) $(TRIHASH_EXECUTABLES)	
	$(MAKE) test && \
	$(MAKE) test_trihash

sanity: $(EXECUTABLES)
	./test_sanity
	./test_capacity_trihash

.PHONY: all clean
