MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
EXAMPLE_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/../..)
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(EXAMPLE_DIR)/build)
BUILD_DIR := $(abspath $(BUILD_DIR))

TARGET_NVCC = -O3
CPPSTD = c++14

cuda_dir=/usr/local/cuda
mpi_dir=/usr/local/mpi
NVCC ?= $(cuda_dir)/bin/nvcc
MPICXX ?= $(mpi_dir)/bin/mpicxx
source=$(PROJECT_ROOT)/source
contrib=$(source)/contrib/scfd/include
include_cuda=$(cuda_dir)/include
include_mpi=$(mpi_dir)/include
lib_cuda=$(cuda_dir)/lib64
lib_mpi=$(mpi_dir)/lib
OMP=-fopenmp -lpthread
CUDA_ARCH=-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
SCFD_FLAGS = -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI

BINS = \
	poisson_periodic_3d_autotuned.bin \
	poisson_periodic_4d.bin

.PHONY: all $(BINS)

all: $(addprefix $(BUILD_DIR)/,$(BINS))

$(BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/poisson_periodic_3d_autotuned.bin: $(EXAMPLE_DIR)/source/poisson_periodic_3d_autotuned.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(SCFD_FLAGS) $< -c -o $(BUILD_DIR)/poisson_periodic_3d_autotuned.o
	$(MPICXX) -g $(BUILD_DIR)/poisson_periodic_3d_autotuned.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@

$(BUILD_DIR)/poisson_periodic_4d.bin: $(EXAMPLE_DIR)/source/poisson_periodic_4d.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(SCFD_FLAGS) $< -c -o $(BUILD_DIR)/poisson_periodic_4d.o
	$(MPICXX) -g $(BUILD_DIR)/poisson_periodic_4d.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@
