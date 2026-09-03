MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
EXAMPLE_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/..)
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(EXAMPLE_DIR)/build)
BUILD_DIR := $(abspath $(BUILD_DIR))

TARGET_NVCC ?= -O3
CPPSTD ?= c++14

cuda_dir ?= /usr/local/cuda
mpi_dir ?= /usr/local/mpi
ROCM_DIR ?= /opt/rocm
NVCC ?= $(cuda_dir)/bin/nvcc
HIPCC ?= $(ROCM_DIR)/bin/hipcc
HIP_ARCH ?=
MPICXX ?= $(mpi_dir)/bin/mpicxx
source=$(PROJECT_ROOT)/source
contrib=$(source)/contrib/scfd/include
include_cuda=$(cuda_dir)/include
include_mpi=$(mpi_dir)/include
lib_cuda=$(cuda_dir)/lib64
lib_mpi=$(mpi_dir)/lib
OMP ?= -fopenmp -lpthread
CUDA_ARCH ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75
NVCCFLAGS ?= -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
SCFD_FLAGS ?= -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI

ALL_BINS = \
	poisson_periodic_3d_autotuned.bin \
	poisson_periodic_4d_autotuned.bin \
	taylor_green_3d_autotuned.bin \
	taylor_green_spacetime_4d.bin
OPTIONAL_BINS = taylor_green_3d_autotuned_float.bin

.PHONY: all reader-smoke poisson-hip poisson-hip-nca $(ALL_BINS) $(OPTIONAL_BINS)

all: $(addprefix $(BUILD_DIR)/,$(ALL_BINS))

reader-smoke:
	bash $(EXAMPLE_DIR)/tests/run_reader_smoke.sh

poisson-hip:
	$(MAKE) -C $(EXAMPLE_DIR)/poisson hip BUILD_FOLDER=$(BUILD_DIR) \
		ROCM_DIR="$(ROCM_DIR)" HIPCC="$(HIPCC)" HIP_ARCH="$(HIP_ARCH)" mpi_dir="$(mpi_dir)"

poisson-hip-nca:
	$(MAKE) -C $(EXAMPLE_DIR)/poisson hip-nca BUILD_FOLDER=$(BUILD_DIR) \
		ROCM_DIR="$(ROCM_DIR)" HIPCC="$(HIPCC)" HIP_ARCH="$(HIP_ARCH)" mpi_dir="$(mpi_dir)"

$(ALL_BINS): %: $(BUILD_DIR)/%

$(OPTIONAL_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

POISSON_DIR := $(EXAMPLE_DIR)/poisson

$(BUILD_DIR)/poisson_periodic_3d_autotuned.bin: \
		$(POISSON_DIR)/Makefile \
		$(POISSON_DIR)/source/poisson_periodic_3d_autotuned.cpp | $(BUILD_DIR)
	$(MAKE) -C $(POISSON_DIR) \
		poisson_periodic_3d_autotuned.bin \
		BUILD_FOLDER=$(BUILD_DIR) \
		CUDA_ARCH="$(CUDA_ARCH)" \
		NVCC="$(NVCC)" \
		MPICXX="$(MPICXX)"

$(BUILD_DIR)/poisson_periodic_4d_autotuned.bin: \
		$(POISSON_DIR)/Makefile \
		$(POISSON_DIR)/source/poisson_periodic_4d_autotuned.cpp | $(BUILD_DIR)
	$(MAKE) -C $(POISSON_DIR) \
		poisson_periodic_4d_autotuned.bin \
		BUILD_FOLDER=$(BUILD_DIR) \
		CUDA_ARCH="$(CUDA_ARCH)" \
		NVCC="$(NVCC)" \
		MPICXX="$(MPICXX)"

TURBULENCE_SOURCE := $(EXAMPLE_DIR)/turbulence/source
TURBULENCE_HEADERS := \
	$(TURBULENCE_SOURCE)/snapshot_writer.h \
	$(TURBULENCE_SOURCE)/spacetime_kernels.h \
	$(TURBULENCE_SOURCE)/spacetime_options.h \
	$(TURBULENCE_SOURCE)/taylor_green_kernels.h \
	$(TURBULENCE_SOURCE)/taylor_green_solver.h \
	$(TURBULENCE_SOURCE)/turbulence_options.h

$(BUILD_DIR)/taylor_green_3d_autotuned.bin: $(TURBULENCE_SOURCE)/taylor_green_3d_autotuned.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS) $< -c -o $(BUILD_DIR)/taylor_green_3d_autotuned.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_3d_autotuned.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@

$(BUILD_DIR)/taylor_green_3d_autotuned_float.bin: $(TURBULENCE_SOURCE)/taylor_green_3d_autotuned.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -DFFTM_TURBULENCE_SINGLE_PRECISION=1 -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS) $< -c -o $(BUILD_DIR)/taylor_green_3d_autotuned_float.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_3d_autotuned_float.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@

$(BUILD_DIR)/taylor_green_spacetime_4d.bin: $(TURBULENCE_SOURCE)/taylor_green_spacetime_4d.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS) $< -c -o $(BUILD_DIR)/taylor_green_spacetime_4d.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_spacetime_4d.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@
