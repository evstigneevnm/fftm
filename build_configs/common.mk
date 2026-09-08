# Included after PROJECT_ROOT and DEFAULT_BUILD_DIR are set by the entry Makefile.
.DEFAULT_GOAL := all

ifneq ($(origin CONFIG_FILE),undefined)
ifeq ($(strip $(CONFIG_FILE)),)
$(error CONFIG_FILE was explicitly set but is empty)
endif
override CONFIG_FILE := $(abspath $(if $(filter /%,$(CONFIG_FILE)),$(CONFIG_FILE),$(PROJECT_ROOT)/$(CONFIG_FILE)))
ifeq ($(wildcard $(CONFIG_FILE)),)
$(error Configuration file does not exist: $(CONFIG_FILE))
endif
ifeq ($(shell test -s '$(CONFIG_FILE)' && printf yes),)
$(error Configuration file is empty: $(CONFIG_FILE))
endif
include $(CONFIG_FILE)
else
ifneq ($(wildcard $(PROJECT_ROOT)/build_configs/config_local.inc),)
CONFIG_FILE := $(PROJECT_ROOT)/build_configs/config_local.inc
ifeq ($(shell test -s '$(CONFIG_FILE)' && printf yes),)
$(error Configuration file is empty: $(CONFIG_FILE))
endif
include $(CONFIG_FILE)
endif
endif

FFTM_BUILD_BACKEND ?= cuda
ifneq ($(words $(FFTM_BUILD_BACKEND)),1)
$(error FFTM_BUILD_BACKEND must be cuda or hip)
endif
ifeq ($(filter $(FFTM_BUILD_BACKEND),cuda hip),)
$(error FFTM_BUILD_BACKEND must be cuda or hip)
endif
FFTM_DEVICE_AWARE_MPI ?= 1
ifneq ($(words $(FFTM_DEVICE_AWARE_MPI)),1)
$(error FFTM_DEVICE_AWARE_MPI must be 0 or 1)
endif
ifeq ($(filter $(FFTM_DEVICE_AWARE_MPI),0 1),)
$(error FFTM_DEVICE_AWARE_MPI must be 0 or 1)
endif

ifeq ($(origin BUILD_FOLDER),command line)
ifneq ($(origin BUILD_DIR),command line)
BUILD_DIR := $(BUILD_FOLDER)
endif
endif
BUILD_DIR ?= $(if $(BUILD_FOLDER),$(BUILD_FOLDER),$(DEFAULT_BUILD_DIR))
override BUILD_DIR := $(abspath $(BUILD_DIR))
TARGET_GCC ?= -O3
TARGET_NVCC ?= -O3
TARGET_HIP ?= -O3
CPPSTD ?= c++14
cuda_dir ?= /usr/local/cuda
mpi_dir ?= /usr/local/mpi
ROCM_DIR ?= /opt/rocm
NVCC ?= $(cuda_dir)/bin/nvcc
HIPCC ?= $(ROCM_DIR)/bin/hipcc
MPICXX ?= $(mpi_dir)/bin/mpicxx
MPIEXEC ?= $(mpi_dir)/bin/mpiexec
MPIEXEC_FLAGS ?=
CXX ?= g++
CUDA_HOST_CXX ?=
source := $(PROJECT_ROOT)/source
SCFD_INCLUDE_DIR ?= $(source)/contrib/scfd/include
contrib := $(SCFD_INCLUDE_DIR)
include_cuda ?= $(cuda_dir)/include
include_mpi ?= $(mpi_dir)/include
lib_cuda ?= $(cuda_dir)/lib64
lib_mpi ?= $(mpi_dir)/lib
ROCM_LIB_DIR ?= $(ROCM_DIR)/lib
# Override these for MPI installations with multiple include/library directories.
MPI_CPPFLAGS ?= -I$(include_mpi)
MPI_LDFLAGS ?= -L$(lib_mpi)
MPI_LIBS ?= -lmpi
OMP ?= -fopenmp -lpthread
CUDA_ARCH_LIST ?= 70 75
CUDA_ARCH ?= $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))
HIP_ARCH_LIST ?=
HIP_ARCH ?= $(HIP_ARCH_LIST)
HIP_ARCH_FLAGS = $(foreach arch,$(HIP_ARCH),--offload-arch=$(arch))
NVCCFLAGS ?= -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
CUDA_HOST_FLAGS = $(if $(strip $(CUDA_HOST_CXX)),-ccbin $(CUDA_HOST_CXX),)
GCCFLAGS ?= -std=$(CPPSTD) $(TARGET_GCC)
HIPFLAGS ?= -std=$(CPPSTD) $(TARGET_HIP) $(HIP_ARCH_FLAGS)
SCFD_FLAGS ?= -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t
SCFD_DEVICE_AWARE := -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
SCFD_FLAGS_CA = $(SCFD_FLAGS) $(SCFD_DEVICE_AWARE)
SCFD_FLAGS_NCA = $(SCFD_FLAGS)
DEVICE_AWARE_FLAG = $(SCFD_DEVICE_AWARE)
HIP_DEVICE_AWARE = $(SCFD_DEVICE_AWARE)
HIP_PLATFORM_FLAGS := -DPLATFORM_HIP -DFFTM_PLATFORM_HIP -DSCFD_BACKEND_ENABLE_MPI
NVCCFLAGS_NO_DEVICE_DEBUG = $(filter-out -G,$(NVCCFLAGS))
FFTM_DEPFLAGS = -MMD -MP -MF $@.d -MT $@

# Export the effective values, including command-line overrides, to child makes
# and the reader launcher. Do not source Make syntax in a shell script.
FFTM_CONFIG_VARIABLES := CONFIG_FILE FFTM_BUILD_BACKEND FFTM_DEVICE_AWARE_MPI \
    BUILD_DIR TARGET_GCC TARGET_NVCC TARGET_HIP CPPSTD cuda_dir mpi_dir ROCM_DIR \
    NVCC HIPCC MPICXX MPIEXEC MPIEXEC_FLAGS CXX CUDA_HOST_CXX \
    SCFD_INCLUDE_DIR include_cuda include_mpi lib_cuda lib_mpi ROCM_LIB_DIR \
    MPI_CPPFLAGS MPI_LDFLAGS MPI_LIBS OMP CUDA_ARCH_LIST CUDA_ARCH HIP_ARCH_LIST HIP_ARCH \
    NVCCFLAGS GCCFLAGS HIPFLAGS SCFD_FLAGS CPPFLAGS CXXFLAGS LDFLAGS NVCC_LDFLAGS LDLIBS
export $(filter-out CONFIG_FILE,$(FFTM_CONFIG_VARIABLES))
ifneq ($(origin CONFIG_FILE),undefined)
export CONFIG_FILE
endif

fftm_shell_quote = '$(subst ','"'"',$(1))'
.PHONY: print-config check-config check-config-cxx check-config-cuda check-config-hip
print-config:
	@$(foreach var,$(FFTM_CONFIG_VARIABLES),printf '%s\n' $(call fftm_shell_quote,$(var)=$($(var)));)

check-config: check-config-$(FFTM_BUILD_BACKEND)

check-config-cxx:
	@command -v $(firstword $(CXX)) >/dev/null || { echo 'C++ compiler not found: $(CXX)' >&2; exit 1; }

check-config-cuda check-config-hip:
	@command -v $(firstword $(MPICXX)) >/dev/null || { echo 'MPI C++ wrapper not found: $(MPICXX)' >&2; exit 1; }
	@test -d '$(contrib)/scfd' || { echo 'SCFD headers not found: $(contrib)' >&2; exit 1; }
	@command -v $(if $(filter check-config-cuda,$@),$(firstword $(NVCC)),$(firstword $(HIPCC))) >/dev/null || { echo 'Selected GPU compiler not found' >&2; exit 1; }
	@printf '%s\n' 'Configuration check passed: $@; use print-config to inspect flags.'
