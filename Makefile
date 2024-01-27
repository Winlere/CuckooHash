# Makefile for compiling main.cu and hash.cu into an executable named main

# Specify the compiler
NVCC = nvcc

# Compiler flags, e.g., for debugging, add -G -g
NVCCFLAGS = -O2

# Specify the target executable
TARGET = main

# List of source files
SOURCES = main.cu hash.cu

# List of object files, replace .cu extension with .o
OBJECTS = $(SOURCES:.cu=.o)

# Default target
all: $(TARGET)

# Rule for linking the final executable
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Rule for compiling source files to object files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean
