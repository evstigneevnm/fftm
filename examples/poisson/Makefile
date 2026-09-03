MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
EXAMPLE_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/../..)
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(EXAMPLE_DIR)/build)
BUILD_DIR := $(abspath $(BUILD_DIR))

TARGET_NVCC ?= -O3
TARGET_HIP ?= -O3
CPPSTD ?= c++14

cuda_dir ?= /usr/local/cuda
mpi_dir ?= /usr/local/mpi
ROCM_DIR ?= /opt/rocm
NVCC ?= $(cuda_dir)/bin/nvcc
HIPCC ?= $(ROCM_DIR)/bin/hipcc
MPICXX ?= $(mpi_dir)/bin/mpicxx
source=$(PROJECT_ROOT)/source
contrib=$(source)/contrib/scfd/include
include_cuda=$(cuda_dir)/include
include_mpi=$(mpi_dir)/include
lib_cuda=$(cuda_dir)/lib64
lib_mpi=$(mpi_dir)/lib
ROCM_LIB_DIR ?= $(ROCM_DIR)/lib
OMP ?= -fopenmp -lpthread
CUDA_ARCH ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
HIP_ARCH ?=
HIP_ARCH_FLAGS = $(if $(strip $(HIP_ARCH)),--offload-arch=$(HIP_ARCH),)
NVCCFLAGS ?= -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
HIPFLAGS ?= -std=$(CPPSTD) $(TARGET_HIP) $(HIP_ARCH_FLAGS)
SCFD_FLAGS ?= -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t
DEVICE_AWARE_FLAG = -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
HIP_PLATFORM_FLAGS = -DPLATFORM_HIP -DFFTM_PLATFORM_HIP -DSCFD_BACKEND_ENABLE_MPI

CUDA_BINS = \
	poisson_periodic_3d_autotuned.bin \
	poisson_periodic_4d_autotuned.bin
HIP_BINS = \
	poisson_periodic_3d_autotuned_hip.bin \
	poisson_periodic_4d_autotuned_hip.bin
HIP_NCA_BINS = \
	poisson_periodic_3d_autotuned_hip_nca.bin \
	poisson_periodic_4d_autotuned_hip_nca.bin

.PHONY: all cuda hip hip-nca $(CUDA_BINS) $(HIP_BINS) $(HIP_NCA_BINS)

all: cuda

cuda: $(addprefix $(BUILD_DIR)/,$(CUDA_BINS))

hip: $(addprefix $(BUILD_DIR)/,$(HIP_BINS))

hip-nca: $(addprefix $(BUILD_DIR)/,$(HIP_NCA_BINS))

$(CUDA_BINS) $(HIP_BINS) $(HIP_NCA_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

define BUILD_CUDA_POISSON
$(BUILD_DIR)/$1: $(EXAMPLE_DIR)/source/$2 | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -x cu -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) \
		$(SCFD_FLAGS) $(DEVICE_AWARE_FLAG) $$< -c -o $(BUILD_DIR)/$3
	$(MPICXX) -g $(BUILD_DIR)/$3 $(OMP) -L$(lib_cuda) -L$(lib_mpi) \
		-Wl,-rpath,$(lib_cuda) -Wl,-rpath,$(lib_mpi) -lcudart -lcufft -lmpi -o $$@

endef

define BUILD_HIP_POISSON
$(BUILD_DIR)/$1: $(EXAMPLE_DIR)/source/$2 | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) -x hip -I$(source) -I$(include_mpi) -I$(contrib) \
		$(SCFD_FLAGS) $(HIP_PLATFORM_FLAGS) $3 $$< \
		-L$(lib_mpi) -L$(ROCM_LIB_DIR) -Wl,-rpath,$(lib_mpi) -Wl,-rpath,$(ROCM_LIB_DIR) \
		-lmpi -lhipfft -lpthread -o $$@

endef

$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_3d_autotuned.bin,poisson_periodic_3d_autotuned.cpp,poisson_periodic_3d_autotuned.o))
$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_4d_autotuned.bin,poisson_periodic_4d_autotuned.cpp,poisson_periodic_4d_autotuned.o))

$(eval $(call BUILD_HIP_POISSON,poisson_periodic_3d_autotuned_hip.bin,poisson_periodic_3d_autotuned.cpp,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_4d_autotuned_hip.bin,poisson_periodic_4d_autotuned.cpp,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_3d_autotuned_hip_nca.bin,poisson_periodic_3d_autotuned.cpp,))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_4d_autotuned_hip_nca.bin,poisson_periodic_4d_autotuned.cpp,))
