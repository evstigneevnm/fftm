MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
TEST_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(TEST_DIR)/../..)
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(TEST_DIR)/build)
BUILD_DIR := $(abspath $(BUILD_DIR))

TARGET_GCC = -O3# -g
TARGET_NVCC = -O3#-g -G #-O3
TARGET_HIP = -O3
CPPSTD = c++14

cuda_dir ?= /usr/local/cuda
mpi_dir ?= /usr/local/mpi
NVCC ?= $(cuda_dir)/bin/nvcc
MPICXX ?= $(mpi_dir)/bin/mpicxx
CXX ?= g++
ROCM_DIR ?= /opt/rocm
HIPCC ?= $(ROCM_DIR)/bin/hipcc
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
HIP_ARCH ?=
HIP_ARCH_FLAGS = $(if $(strip $(HIP_ARCH)),--offload-arch=$(HIP_ARCH),)
HIPFLAGS ?= -std=$(CPPSTD) $(TARGET_HIP) $(HIP_ARCH_FLAGS)
HIP_PLATFORM_FLAGS = -DPLATFORM_HIP -DFFTM_PLATFORM_HIP -DSCFD_BACKEND_ENABLE_MPI
HIP_DEVICE_AWARE = -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI

FFTM3D_DIR := $(PROJECT_ROOT)/fftm3d
EGGER_ROOT ?= $(PROJECT_ROOT)/fft_Egger
WRAPPED_EGGER_ROOT := $(FFTM3D_DIR)/egger_wrapped
FFTM3D_WRAPPED_INCLUDES := -I$(FFTM3D_DIR)/src -I$(WRAPPED_EGGER_ROOT)/include -I$(EGGER_ROOT)/include
FFTM3D_WRAPPED_INCLUDES += -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib)
FFTM3D_WRAPPED_SRCS := \
	$(WRAPPED_EGGER_ROOT)/src/fftm3d_egger_mpi_facade.cpp \
	$(WRAPPED_EGGER_ROOT)/src/mpicufft.cpp \
	$(WRAPPED_EGGER_ROOT)/src/timer.cpp \
	$(WRAPPED_EGGER_ROOT)/src/pencil/mpicufft_pencil.cpp \
	$(WRAPPED_EGGER_ROOT)/src/pencil/mpicufft_pencil_opt1.cpp
FFTM3D_WRAPPED_OBJS := $(patsubst $(WRAPPED_EGGER_ROOT)/src/%.cpp,$(BUILD_DIR)/fftm3d_egger_wrapped_%.o,$(FFTM3D_WRAPPED_SRCS))
FFTM3D_BENCH_FOR_FFTM_OBJ := $(BUILD_DIR)/fftm3d_benchmark_3d_for_fftm.o
FFTM_ENABLE_FFTM3D_BACKEND ?= 0

ifeq ($(FFTM_ENABLE_FFTM3D_BACKEND),1)
FFTM3D_BENCH_DEPS := $(FFTM3D_BENCH_FOR_FFTM_OBJ) $(FFTM3D_WRAPPED_OBJS)
FFTM3D_BENCH_OBJS := $(FFTM3D_BENCH_FOR_FFTM_OBJ) $(FFTM3D_WRAPPED_OBJS)
FFTM3D_BENCH_FLAGS := -I$(FFTM3D_DIR)/src -DFFTM_ENABLE_FFTM3D_SCFD_FFT_BACKEND=1
else
FFTM3D_BENCH_DEPS :=
FFTM3D_BENCH_OBJS :=
FFTM3D_BENCH_FLAGS :=
endif

ALL_BINS = \
	test_fftm_options.bin \
	test_fft_benchmark_reporting.bin \
	test_fftm_autotune_hardware.bin \
	test_fftm_resource_lifecycle.bin \
	test_fftm_quiet_default.bin \
	test_init.bin \
	test_fft_2D.bin \
	test_2D_poisson.bin \
	test_3D_poisson.bin \
	test_4D_poisson.bin \
	test_fft_3D.bin \
	test_mpi_transpose_3D.bin \
	test_mpi_transpose_3D_nca.bin \
	test_scfd_mpi_transport.bin \
	test_egger_pencil_comm_pattern_3D.bin \
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
HIP_BINS = \
	test_fftm_3D_compare_hip.bin \
	test_fftm_3D_compare_hip_nca.bin \
	test_fftm_4D_compare_hip.bin \
	test_fftm_4D_compare_hip_nca.bin \
	test_benchmark_fftm_3D_hip.bin \
	test_benchmark_fftm_3D_hip_nca.bin \
	test_benchmark_fftm_4D_hip.bin \
	test_benchmark_fftm_4D_hip_nca.bin \
	test_4D_poisson_mpi_hip.bin \
	test_4D_poisson_mpi_hip_nca.bin

.PHONY: all hip test check-abstraction-boundaries $(BIN_TARGETS) $(HIP_BINS)

all: $(addprefix $(BUILD_DIR)/,$(ALL_BINS))

hip: $(addprefix $(BUILD_DIR)/,$(HIP_BINS))

check-abstraction-boundaries:
	$(PROJECT_ROOT)/scripts/check_backend_abstraction_boundaries.sh

$(BIN_TARGETS): %: $(BUILD_DIR)/%

$(HIP_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/test_fftm_options.bin: $(TEST_DIR)/test_fftm_options.cpp $(source)/fftm_options.hpp | $(BUILD_DIR)
	$(CXX) $(GCCFLAGS) -I$(source) $< -o $@

$(BUILD_DIR)/test_fft_benchmark_reporting.bin: $(TEST_DIR)/test_fft_benchmark_reporting.cpp $(TEST_DIR)/detail/fft_benchmark_common.h | $(BUILD_DIR)
	$(MPICXX) $(GCCFLAGS) -I$(source) -I$(include_mpi) -I$(contrib) $< -L$(lib_mpi) -lmpi -o $@

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

define BUILD_HIP_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) -x hip -I$(source) -I$(include_mpi) -I$(contrib) \
		$(SCFD_FLAGS) $(HIP_PLATFORM_FLAGS) $3 $$< \
		-L$(lib_mpi) -L$(ROCM_DIR)/lib -Wl,-rpath,$(lib_mpi) -Wl,-rpath,$(ROCM_DIR)/lib \
		-lmpi -lhipfft -lpthread -o $$@

endef

$(eval $(call BUILD_MPI_TEST,test_init.bin,test_init.cu,$(NVCCFLAGS),$(SCFD_FLAGS),test_init.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_quiet_default.bin,test_fftm_quiet_default.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_quiet_default.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_autotune_hardware.bin,test_fftm_autotune_hardware.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_autotune_hardware.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_resource_lifecycle.bin,test_fftm_resource_lifecycle.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_resource_lifecycle.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_CUDA_TEST,test_cudaMemcpy3DAsync.bin,test_cudaMemcpy3DAsync.cu,$(NVCCFLAGS),$(SCFD_FLAGS),))

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_mpi_transpose_3D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D_nca.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_mpi_transpose_3D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_scfd_mpi_transport.bin,test_scfd_mpi_transport.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS) -DPLATFORM_CUDA -DSCFD_BACKEND_ENABLE_MPI,test_scfd_mpi_transport.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_egger_pencil_comm_pattern_3D.bin,test_egger_pencil_comm_pattern_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_egger_pencil_comm_pattern_3D.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_mpi_transpose_4D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D_nca.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_mpi_transpose_4D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_3D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare_nca.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_3D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_4D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare_nca.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_4D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_3D_compare_hip.bin,test_fftm_3D_compare.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_3D_compare_hip_nca.bin,test_fftm_3D_compare.cu,))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_4D_compare_hip.bin,test_fftm_4D_compare.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_4D_compare_hip_nca.bin,test_fftm_4D_compare.cu,))
$(eval $(call BUILD_HIP_MPI_TEST,test_benchmark_fftm_3D_hip.bin,test_benchmark_fftm_3D.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_benchmark_fftm_3D_hip_nca.bin,test_benchmark_fftm_3D.cu,))
$(eval $(call BUILD_HIP_MPI_TEST,test_benchmark_fftm_4D_hip.bin,test_benchmark_fftm_4D.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_benchmark_fftm_4D_hip_nca.bin,test_benchmark_fftm_4D.cu,))
$(eval $(call BUILD_HIP_MPI_TEST,test_4D_poisson_mpi_hip.bin,test_4D_poisson_mpi.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_4D_poisson_mpi_hip_nca.bin,test_4D_poisson_mpi.cu,))

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

$(BUILD_DIR)/fftm3d_egger_wrapped_%.o: $(WRAPPED_EGGER_ROOT)/src/%.cpp | $(BUILD_DIR)
	mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(FFTM3D_WRAPPED_INCLUDES) $(SCFD_FLAGS) -DFFTM3D_WRAPPED_EGGER=1 -x cu -c $< -o $@

$(FFTM3D_BENCH_FOR_FFTM_OBJ): $(FFTM3D_DIR)/src/fftm3d_benchmark_3d.cu $(FFTM3D_DIR)/src/fftm3d_benchmark_3d_api.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FFTM3D_WRAPPED_INCLUDES) $(SCFD_FLAGS) -DFFTM3D_WRAPPED_EGGER=1 -DFFTM3D_BENCHMARK_NO_MAIN=1 -c $< -o $@

$(BUILD_DIR)/test_benchmark_fftm_3D.bin: $(TEST_DIR)/test_benchmark_fftm_3D.cu $(FFTM3D_BENCH_DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(FFTM3D_BENCH_FLAGS) $(SCFD_FLAGS_CA) $< -c -o $(BUILD_DIR)/test_benchmark_fftm_3D.o
	$(MPICXX) -g $(BUILD_DIR)/test_benchmark_fftm_3D.o $(FFTM3D_BENCH_OBJS) $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@

$(BUILD_DIR)/test_benchmark_fftm_3D_nca.bin: $(TEST_DIR)/test_benchmark_fftm_3D.cu $(FFTM3D_BENCH_DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(FFTM3D_BENCH_FLAGS) $(SCFD_FLAGS_NCA) $< -c -o $(BUILD_DIR)/test_benchmark_fftm_3D_nca.o
	$(MPICXX) -g $(BUILD_DIR)/test_benchmark_fftm_3D_nca.o $(FFTM3D_BENCH_OBJS) $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o $@

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
