MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
TEST_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(TEST_DIR)/../..)
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(TEST_DIR)/build)
BUILD_DIR := $(abspath $(BUILD_DIR))

TARGET_GCC = -O3# -g
TARGET_NVCC = -O3#-g -G #-O3
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
CUDA_ARCH=-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75# -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 #-gencode arch=compute_35,code=sm_35
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
GCCFLAGS = -std=$(CPPSTD) $(TARGET_GCC)
SCFD_FLAGS = -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t
SCFD_DEVICE_AWARE = -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
SCFD_FLAGS_CA = $(SCFD_FLAGS) $(SCFD_DEVICE_AWARE)
SCFD_FLAGS_NCA = $(SCFD_FLAGS)
NVCCFLAGS_NO_DEVICE_DEBUG = $(filter-out -G,$(NVCCFLAGS))

ALL_BINS = \
	test_init.bin \
	test_fft_2D.bin \
	test_2D_poisson.bin \
	test_3D_poisson.bin \
	test_4D_poisson.bin \
	test_fft_3D.bin \
	test_mpi_transpose_3D.bin \
	test_mpi_transpose_3D_nca.bin \
	test_mpi_transpose_4D.bin \
	test_mpi_transpose_4D_nca.bin \
	test_fftm_3D_compare.bin \
	test_fftm_3D_compare_nca.bin \
	test_fftm_4D_compare.bin \
	test_fftm_4D_compare_nca.bin \
	test_3D_poisson_mpi.bin \
	test_3D_poisson_mpi_nca.bin \
	test_3D_poisson_mpi_tutorial.bin \
	test_3D_poisson_mpi_tutorial_nca.bin \
	test_4D_poisson_mpi.bin \
	test_4D_poisson_mpi_nca.bin \
	test_4D_poisson_mpi_tutorial.bin \
	test_4D_poisson_mpi_tutorial_nca.bin \
	test_benchmark_ffts_3D.bin \
	test_benchmark_ffts_4D.bin \
	test_benchmark_fftm_3D.bin \
	test_benchmark_fftm_3D_nca.bin \
	test_benchmark_fftm_4D.bin \
	test_benchmark_fftm_4D_nca.bin \
	test_fftm_v0_3D.bin \
	test_fftm_v0_3D_nca.bin \
	test_fftm_v1_3D.bin \
	test_fftm_v1_3D_nca.bin \
	test_fftm_v2_3D.bin \
	test_fftm_v2_3D_nca.bin \
	test_fftm_v3_3D.bin \
	test_fftm_v3_3D_nca.bin \
	test_fftm_v4_3D.bin \
	test_fftm_v4_3D_nca.bin \
	test_fftm_v0_4D.bin \
	test_fftm_v0_4D_nca.bin \
	test_fftm_v1_4D.bin \
	test_fftm_v1_4D_nca.bin \
	test_fftm_v2_4D.bin \
	test_fftm_v2_4D_nca.bin \
	test_fftm_v3_4D.bin \
	test_fftm_v3_4D_nca.bin \
	test_fftm_v4_4D.bin \
	test_fftm_v4_4D_nca.bin \
	test_ffts_v0_3D.bin \
	test_ffts_v1_3D.bin \
	test_ffts_v2_3D.bin \
	test_ffts_v3_3D.bin \
	test_ffts_v4_3D.bin \
	test_ffts_v0_4D.bin \
	test_ffts_v1_4D.bin \
	test_ffts_v2_4D.bin \
	test_ffts_v3_4D.bin \
	test_ffts_v4_4D.bin

BIN_TARGETS = $(ALL_BINS) test_cudaMemcpy3DAsync.bin

.PHONY: all test $(BIN_TARGETS)

all: $(addprefix $(BUILD_DIR)/,$(ALL_BINS))

$(BIN_TARGETS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

define BUILD_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $4 $$< -c -o $(BUILD_DIR)/$5
	$(MPICXX) -g $(BUILD_DIR)/$5 $(OMP) -L$(lib_cuda) -L$(lib_mpi) $6 -o $$@

endef

define BUILD_CUDA_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 -I$(source) -I$(include_cuda) -I$(contrib) $4 $$< -L$(lib_cuda) $5 -o $$@

endef

define BUILD_CUDA_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $4 $$< -L$(lib_cuda) -L$(lib_mpi) $5 -o $$@

endef

$(eval $(call BUILD_MPI_TEST,test_init.bin,test_init.cu,$(NVCCFLAGS),$(SCFD_FLAGS),test_init.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_TEST,test_cudaMemcpy3DAsync.bin,test_cudaMemcpy3DAsync.cu,$(NVCCFLAGS),$(SCFD_FLAGS),))

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_mpi_transpose_3D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D_nca.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_mpi_transpose_3D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_mpi_transpose_4D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D_nca.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_mpi_transpose_4D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_3D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare_nca.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_3D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_4D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare_nca.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_4D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi.bin,test_3D_poisson_mpi.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_3D_poisson_mpi.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_nca.bin,test_3D_poisson_mpi.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_3D_poisson_mpi_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_tutorial.bin,test_3D_poisson_mpi_tutorial.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_3D_poisson_mpi_tutorial.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_tutorial_nca.bin,test_3D_poisson_mpi_tutorial.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_3D_poisson_mpi_tutorial_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi.bin,test_4D_poisson_mpi.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_4D_poisson_mpi.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_nca.bin,test_4D_poisson_mpi.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_4D_poisson_mpi_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_tutorial.bin,test_4D_poisson_mpi_tutorial.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_4D_poisson_mpi_tutorial.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_tutorial_nca.bin,test_4D_poisson_mpi_tutorial.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_4D_poisson_mpi_tutorial_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_benchmark_ffts_3D.bin,test_benchmark_ffts_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_benchmark_ffts_4D.bin,test_benchmark_ffts_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_3D.bin,test_benchmark_fftm_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_benchmark_fftm_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_3D_nca.bin,test_benchmark_fftm_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_benchmark_fftm_3D_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_4D.bin,test_benchmark_fftm_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_benchmark_fftm_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_4D_nca.bin,test_benchmark_fftm_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_benchmark_fftm_4D_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_v0_3D.bin,test_fftm_v0_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_v0_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v0_3D_nca.bin,test_fftm_v0_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_v0_3D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v1_3D.bin,test_fftm_v1_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_v1_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v1_3D_nca.bin,test_fftm_v1_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_v1_3D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v2_3D.bin,test_fftm_v2_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_v2_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v2_3D_nca.bin,test_fftm_v2_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_v2_3D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v3_3D.bin,test_fftm_v3_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_v3_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v3_3D_nca.bin,test_fftm_v3_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_v3_3D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v4_3D.bin,test_fftm_v4_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_v4_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v4_3D_nca.bin,test_fftm_v4_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_v4_3D_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_v0_4D.bin,test_fftm_v0_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_v0_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v0_4D_nca.bin,test_fftm_v0_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_v0_4D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v1_4D.bin,test_fftm_v1_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_v1_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v1_4D_nca.bin,test_fftm_v1_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_v1_4D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v2_4D.bin,test_fftm_v2_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_v2_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v2_4D_nca.bin,test_fftm_v2_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_v2_4D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v3_4D.bin,test_fftm_v3_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_v3_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v3_4D_nca.bin,test_fftm_v3_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_v3_4D_nca.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v4_4D.bin,test_fftm_v4_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_v4_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_v4_4D_nca.bin,test_fftm_v4_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_v4_4D_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v0_3D.bin,test_ffts_v0_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v1_3D.bin,test_ffts_v1_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v2_3D.bin,test_ffts_v2_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v3_3D.bin,test_ffts_v3_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v4_3D.bin,test_ffts_v4_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v0_4D.bin,test_ffts_v0_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v1_4D.bin,test_ffts_v1_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v2_4D.bin,test_ffts_v2_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v3_4D.bin,test_ffts_v3_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_MPI_TEST,test_ffts_v4_4D.bin,test_ffts_v4_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft -lmpi))

test: | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) $(TEST_DIR)/test_cudaMemcpy3DAsync_1.cu -o $(BUILD_DIR)/test_cudaMemcpy3DAsync_1.bin

$(eval $(call BUILD_CUDA_TEST,test_fft_2D.bin,test_fft_2D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_2D_poisson.bin,test_2D_poisson.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_3D_poisson.bin,test_3D_poisson.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_4D_poisson.bin,test_4D_poisson.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_fft_3D.bin,test_fft_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))
