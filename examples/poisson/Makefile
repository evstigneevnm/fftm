MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
EXAMPLE_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/../..)
DEFAULT_BUILD_DIR := $(EXAMPLE_DIR)/build
FFTM_BUILD_SCOPE := poisson
include $(PROJECT_ROOT)/build_configs/common.mk

CUDA_BINS = \
	poisson_periodic_3d_autotuned.bin \
	poisson_periodic_4d_autotuned.bin
CUDA_NCA_BINS = \
	poisson_periodic_3d_autotuned_nca.bin \
	poisson_periodic_4d_autotuned_nca.bin
HIP_BINS = \
	poisson_periodic_3d_autotuned_hip.bin \
	poisson_periodic_4d_autotuned_hip.bin
HIP_NCA_BINS = \
	poisson_periodic_3d_autotuned_hip_nca.bin \
	poisson_periodic_4d_autotuned_hip_nca.bin

.PHONY: all cuda cuda-nca hip hip-nca $(CUDA_BINS) $(CUDA_NCA_BINS) $(HIP_BINS) $(HIP_NCA_BINS)

all: $(if $(filter hip,$(FFTM_BUILD_BACKEND)),$(if $(filter 1,$(FFTM_DEVICE_AWARE_MPI)),hip,hip-nca),$(if $(filter 1,$(FFTM_DEVICE_AWARE_MPI)),cuda,cuda-nca))

cuda: $(addprefix $(BUILD_DIR)/,$(CUDA_BINS))

cuda-nca: $(addprefix $(BUILD_DIR)/,$(CUDA_NCA_BINS))

hip: $(addprefix $(BUILD_DIR)/,$(HIP_BINS))

hip-nca: $(addprefix $(BUILD_DIR)/,$(HIP_NCA_BINS))

$(CUDA_BINS) $(CUDA_NCA_BINS) $(HIP_BINS) $(HIP_NCA_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

define BUILD_CUDA_POISSON
$(BUILD_DIR)/$1: $(EXAMPLE_DIR)/source/$2 | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -x cu -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) \
		$(SCFD_FLAGS) $4 $$< -c -o $(BUILD_DIR)/$3
	$(MPICXX) -g $(BUILD_DIR)/$3 $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) \
		-Wl,-rpath,$(lib_cuda) -Wl,-rpath,$(lib_mpi) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $$@

endef

define BUILD_HIP_POISSON
$(BUILD_DIR)/$1: $(EXAMPLE_DIR)/source/$2 | $(BUILD_DIR)
	$(HIPCC) $(HIPFLAGS) $$(FFTM_DEPFLAGS) $(CPPFLAGS) -x hip -I$(source) $(MPI_CPPFLAGS) -I$(contrib) \
		$(SCFD_FLAGS) $(HIP_PLATFORM_FLAGS) $3 $$< \
		$(MPI_LDFLAGS) -L$(ROCM_LIB_DIR) -Wl,-rpath,$(lib_mpi) -Wl,-rpath,$(ROCM_LIB_DIR) $(LDFLAGS) \
		$(MPI_LIBS) -lhipfft -lpthread $(LDLIBS) -o $$@

endef

$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_3d_autotuned.bin,poisson_periodic_3d_autotuned.cpp,poisson_periodic_3d_autotuned.o,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_4d_autotuned.bin,poisson_periodic_4d_autotuned.cpp,poisson_periodic_4d_autotuned.o,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_3d_autotuned_nca.bin,poisson_periodic_3d_autotuned.cpp,poisson_periodic_3d_autotuned_nca.o,))
$(eval $(call BUILD_CUDA_POISSON,poisson_periodic_4d_autotuned_nca.bin,poisson_periodic_4d_autotuned.cpp,poisson_periodic_4d_autotuned_nca.o,))

$(eval $(call BUILD_HIP_POISSON,poisson_periodic_3d_autotuned_hip.bin,poisson_periodic_3d_autotuned.cpp,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_4d_autotuned_hip.bin,poisson_periodic_4d_autotuned.cpp,$(DEVICE_AWARE_FLAG)))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_3d_autotuned_hip_nca.bin,poisson_periodic_3d_autotuned.cpp,))
$(eval $(call BUILD_HIP_POISSON,poisson_periodic_4d_autotuned_hip_nca.bin,poisson_periodic_4d_autotuned.cpp,))

FFTM_BUILD_TARGETS := $(addprefix $(BUILD_DIR)/,$(CUDA_BINS) $(CUDA_NCA_BINS) $(HIP_BINS) $(HIP_NCA_BINS))
include $(PROJECT_ROOT)/build_configs/targets.mk
