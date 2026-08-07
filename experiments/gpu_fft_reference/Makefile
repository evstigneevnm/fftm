PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GPU_FFT_DIR ?= $(PROJECT_DIR)/upstream/GPU_FFT
BUILD_DIR ?= $(PROJECT_DIR)/build
BUILD_DIR := $(abspath $(BUILD_DIR))

CUDA_DIR ?= /usr/local/cuda
MPI_DIR ?= /usr/local/mpi
NVCC ?= $(CUDA_DIR)/bin/nvcc

CPPSTD ?= c++14
OPT_FLAGS ?= -O3
CUDA_ARCH ?= -gencode arch=compute_80,code=sm_80
GPU_FFT_UPSTREAM_COMMIT ?= unknown

INCLUDES := \
	-I$(GPU_FFT_DIR)/src \
	-I$(CUDA_DIR)/include \
	-I$(MPI_DIR)/include

NVCC_FLAGS := \
	$(OPT_FLAGS) \
	-std=$(CPPSTD) \
	$(CUDA_ARCH) \
	-Xcompiler=-fPIC \
	-DGPU_FFT_UPSTREAM_COMMIT=\"$(GPU_FFT_UPSTREAM_COMMIT)\"

LINK_FLAGS := \
	-L$(CUDA_DIR)/lib64 \
	-L$(MPI_DIR)/lib \
	-lcufft \
	-lcudart \
	-lmpi

UPSTREAM_SOURCES := \
	$(GPU_FFT_DIR)/src/initialize_GPU_FFT/GPU_FFT.cu \
	$(GPU_FFT_DIR)/src/initialize_GPU_FFT/GPU_FFT_init.cu \
	$(GPU_FFT_DIR)/src/initialize_GPU_FFT/communications_setup.cu \
	$(GPU_FFT_DIR)/src/FFT_implementation/FFT_R2C.cu \
	$(GPU_FFT_DIR)/src/FFT_implementation/FFT_C2R.cu \
	$(GPU_FFT_DIR)/src/FFT_implementation/FFT_definations.cu \
	$(GPU_FFT_DIR)/src/matrix_transpose/transpose.cu

UPSTREAM_OBJECTS := $(patsubst $(GPU_FFT_DIR)/src/%.cu,$(BUILD_DIR)/upstream/%.o,$(UPSTREAM_SOURCES))
LIBRARY := $(BUILD_DIR)/lib/libGPU_FFT.so
BENCHMARK_OBJECT := $(BUILD_DIR)/gpu_fft_reference_benchmark.o
TARGET := $(BUILD_DIR)/bin/gpu_fft_reference_benchmark.bin

.PHONY: all clean

all: $(TARGET)

$(BUILD_DIR)/upstream/%.o: $(GPU_FFT_DIR)/src/%.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(LIBRARY): $(UPSTREAM_OBJECTS)
	mkdir -p $(dir $@)
	$(NVCC) -shared $^ $(LINK_FLAGS) -o $@

$(BENCHMARK_OBJECT): $(PROJECT_DIR)/src/gpu_fft_reference_benchmark.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(BENCHMARK_OBJECT) $(LIBRARY)
	mkdir -p $(dir $@)
	$(NVCC) $(BENCHMARK_OBJECT) -L$(BUILD_DIR)/lib -lGPU_FFT $(LINK_FLAGS) \
		-Xlinker=-rpath -Xlinker='$$ORIGIN/../lib' -o $@

clean:
	rm -rf $(BUILD_DIR)
