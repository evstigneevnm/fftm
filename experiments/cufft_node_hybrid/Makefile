PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(PROJECT_DIR)/../..)
BUILD_DIR ?= $(PROJECT_DIR)/build
BUILD_DIR := $(abspath $(BUILD_DIR))

CUDA_DIR ?= /usr/local/cuda
MPI_DIR ?= /usr/local/mpi
NVCC ?= $(CUDA_DIR)/bin/nvcc
MPICXX ?= $(MPI_DIR)/bin/mpicxx

CPPSTD ?= c++14
OPT_FLAGS ?= -O3
CUDA_ARCH ?= -gencode arch=compute_80,code=sm_80

INCLUDES := \
	-I$(PROJECT_DIR)/include \
	-I$(REPO_ROOT)/source/contrib/scfd/include \
	-I$(CUDA_DIR)/include \
	-I$(MPI_DIR)/include

NVCC_FLAGS := $(OPT_FLAGS) -std=$(CPPSTD) $(CUDA_ARCH) -Xcompiler -fopenmp
CXX_FLAGS := $(OPT_FLAGS) -std=$(CPPSTD) -fopenmp
LINK_FLAGS := -fopenmp -L$(CUDA_DIR)/lib64 -L$(MPI_DIR)/lib -lcufft -lcudart -lmpi

TARGET := $(BUILD_DIR)/fftm_cufft_node_hybrid.bin
TARGET_4D := $(BUILD_DIR)/fftm_cufft_node_hybrid_4d.bin
BACKEND_OBJECT := $(BUILD_DIR)/cuda_cufft_xt_backend.o
BENCHMARK_OBJECT := $(BUILD_DIR)/node_hybrid_benchmark.o
BACKEND_4D_OBJECT := $(BUILD_DIR)/cuda_cufft_xt_4d_backend.o
BENCHMARK_4D_OBJECT := $(BUILD_DIR)/node_hybrid_4d_benchmark.o

.PHONY: all clean

all: $(TARGET) $(TARGET_4D)

$(BUILD_DIR):
	mkdir -p $@

$(BACKEND_OBJECT): $(PROJECT_DIR)/src/cuda_cufft_xt_backend.cu $(PROJECT_DIR)/include/node_fft_backend.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BENCHMARK_OBJECT): $(PROJECT_DIR)/src/node_hybrid_benchmark.cpp $(PROJECT_DIR)/include/node_fft_backend.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(BACKEND_OBJECT) $(BENCHMARK_OBJECT)
	$(MPICXX) $^ $(LINK_FLAGS) -o $@

$(BACKEND_4D_OBJECT): $(PROJECT_DIR)/src/cuda_cufft_xt_4d_backend.cu $(PROJECT_DIR)/include/node_fft_4d_backend.hpp $(PROJECT_DIR)/include/node_fft_backend.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BENCHMARK_4D_OBJECT): $(PROJECT_DIR)/src/node_hybrid_4d_benchmark.cpp $(PROJECT_DIR)/include/node_fft_4d_backend.hpp $(PROJECT_DIR)/include/node_fft_backend.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET_4D): $(BACKEND_OBJECT) $(BACKEND_4D_OBJECT) $(BENCHMARK_4D_OBJECT)
	$(MPICXX) $^ $(LINK_FLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)
