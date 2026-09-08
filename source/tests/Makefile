MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
TEST_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(TEST_DIR)/../..)
DEFAULT_BUILD_DIR := $(TEST_DIR)/build
FFTM_BUILD_SCOPE := tests
include $(PROJECT_ROOT)/build_configs/common.mk

FFTM3D_DIR := $(PROJECT_ROOT)/fftm3d
EGGER_ROOT ?= $(PROJECT_ROOT)/fft_Egger
WRAPPED_EGGER_ROOT := $(FFTM3D_DIR)/egger_wrapped
FFTM3D_WRAPPED_INCLUDES := -I$(FFTM3D_DIR)/src -I$(WRAPPED_EGGER_ROOT)/include -I$(EGGER_ROOT)/include
FFTM3D_WRAPPED_INCLUDES += -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib)
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
	test_fftm_autotune_4d.bin \
	test_fftm_autotune_4d_runtime.bin \
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
	test_fftm_autotune_4d_hip.bin \
	test_fftm_autotune_4d_hip_nca.bin \
	test_fftm_autotune_4d_runtime_hip.bin \
	test_fftm_autotune_4d_runtime_hip_nca.bin \
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

.PHONY: all cuda hip hip-nca test check-abstraction-boundaries $(BIN_TARGETS) $(HIP_BINS)

all: $(if $(filter hip,$(FFTM_BUILD_BACKEND)),$(if $(filter 1,$(FFTM_DEVICE_AWARE_MPI)),hip,hip-nca),cuda)
cuda: $(addprefix $(BUILD_DIR)/,$(ALL_BINS))

hip: $(addprefix $(BUILD_DIR)/,$(HIP_BINS))
hip-nca: $(addprefix $(BUILD_DIR)/,$(filter %_nca.bin,$(HIP_BINS)))

check-abstraction-boundaries:
	$(PROJECT_ROOT)/scripts/check_backend_abstraction_boundaries.sh

$(BIN_TARGETS): %: $(BUILD_DIR)/%

$(HIP_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/test_fftm_options.bin: $(TEST_DIR)/test_fftm_options.cpp $(source)/fftm_options.hpp | $(BUILD_DIR)
	$(CXX) $(GCCFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(FFTM_DEPFLAGS) -I$(source) $< $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_fft_benchmark_reporting.bin: $(TEST_DIR)/test_fft_benchmark_reporting.cpp $(TEST_DIR)/detail/fft_benchmark_common.h | $(BUILD_DIR)
	$(MPICXX) $(GCCFLAGS) $(CXXFLAGS) $(CPPFLAGS) $(FFTM_DEPFLAGS) -I$(source) $(MPI_CPPFLAGS) -I$(contrib) $< $(MPI_LDFLAGS) $(LDFLAGS) $(MPI_LIBS) $(LDLIBS) -o $@

define BUILD_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 $(CUDA_HOST_FLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) $4 $$< -c -o $(BUILD_DIR)/$5
	$(MPICXX) -g $(BUILD_DIR)/$5 $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) $(subst -lmpi,$(MPI_LIBS),$6) $(LDLIBS) -o $$@

endef

define BUILD_CUDA_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 $(CUDA_HOST_FLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $4 $$< -L$(lib_cuda) $(NVCC_LDFLAGS) $5 $(LDLIBS) -o $$@

endef

define BUILD_CUDA_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(NVCC) $3 $(CUDA_HOST_FLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) $4 $$< -L$(lib_cuda) $(MPI_LDFLAGS) $(NVCC_LDFLAGS) $(subst -lmpi,$(MPI_LIBS),$5) $(LDLIBS) -o $$@

endef

define BUILD_HIP_MPI_TEST
$(BUILD_DIR)/$1: $(TEST_DIR)/$2 | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -x hip -I$(source) $(MPI_CPPFLAGS) -I$(contrib) \
		$(SCFD_FLAGS) $(HIP_PLATFORM_FLAGS) $3 $$< \
		$(MPI_LDFLAGS) -L$(ROCM_LIB_DIR) -Wl,-rpath,$(lib_mpi) -Wl,-rpath,$(ROCM_LIB_DIR) $(LDFLAGS) \
		$(MPI_LIBS) -lhipfft -lpthread $(LDLIBS) -o $$@

endef

$(eval $(call BUILD_MPI_TEST,test_init.bin,test_init.cu,$(NVCCFLAGS),$(SCFD_FLAGS),test_init.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_quiet_default.bin,test_fftm_quiet_default.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_quiet_default.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_autotune_hardware.bin,test_fftm_autotune_hardware.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_autotune_hardware.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_autotune_4d.bin,test_fftm_autotune_4d.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_autotune_4d.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_autotune_4d_runtime.bin,test_fftm_autotune_4d_runtime.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_autotune_4d_runtime.o,-lcudart -lcufft -lmpi))

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
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_autotune_4d_hip.bin,test_fftm_autotune_4d.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_autotune_4d_hip_nca.bin,test_fftm_autotune_4d.cu,))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_autotune_4d_runtime_hip.bin,test_fftm_autotune_4d_runtime.cu,$(HIP_DEVICE_AWARE)))
$(eval $(call BUILD_HIP_MPI_TEST,test_fftm_autotune_4d_runtime_hip_nca.bin,test_fftm_autotune_4d_runtime.cu,))
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
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) $(FFTM3D_WRAPPED_INCLUDES) $(SCFD_FLAGS) -DFFTM3D_WRAPPED_EGGER=1 -x cu -c $< -o $@

$(FFTM3D_BENCH_FOR_FFTM_OBJ): $(FFTM3D_DIR)/src/fftm3d_benchmark_3d.cu $(FFTM3D_DIR)/src/fftm3d_benchmark_3d_api.hpp | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) $(FFTM3D_WRAPPED_INCLUDES) $(SCFD_FLAGS) -DFFTM3D_WRAPPED_EGGER=1 -DFFTM3D_BENCHMARK_NO_MAIN=1 -c $< -o $@

$(BUILD_DIR)/test_benchmark_fftm_3D.bin: $(TEST_DIR)/test_benchmark_fftm_3D.cu $(FFTM3D_BENCH_DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) $(FFTM3D_BENCH_FLAGS) $(SCFD_FLAGS_CA) $< -c -o $(BUILD_DIR)/test_benchmark_fftm_3D.o
	$(MPICXX) -g $(BUILD_DIR)/test_benchmark_fftm_3D.o $(FFTM3D_BENCH_OBJS) $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/test_benchmark_fftm_3D_nca.bin: $(TEST_DIR)/test_benchmark_fftm_3D.cu $(FFTM3D_BENCH_DEPS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) $(FFTM3D_BENCH_FLAGS) $(SCFD_FLAGS_NCA) $< -c -o $(BUILD_DIR)/test_benchmark_fftm_3D_nca.o
	$(MPICXX) -g $(BUILD_DIR)/test_benchmark_fftm_3D_nca.o $(FFTM3D_BENCH_OBJS) $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $@

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
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) $(TEST_DIR)/test_cudaMemcpy3DAsync_1.cu $(NVCC_LDFLAGS) $(LDLIBS) -o $(BUILD_DIR)/test_cudaMemcpy3DAsync_1.bin

$(eval $(call BUILD_CUDA_TEST,test_fft_2D.bin,test_fft_2D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_2D_poisson.bin,test_2D_poisson.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_3D_poisson.bin,test_3D_poisson.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_4D_poisson.bin,test_4D_poisson.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS),-lcudart -lcufft))

$(eval $(call BUILD_CUDA_TEST,test_fft_3D.bin,test_fft_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS),-lcudart -lcufft))

FFTM_BUILD_TARGETS := $(addprefix $(BUILD_DIR)/,$(BIN_TARGETS) $(HIP_BINS)) $(FFTM3D_BENCH_DEPS)
include $(PROJECT_ROOT)/build_configs/targets.mk
