MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
EXAMPLE_DIR := $(patsubst %/,%,$(dir $(MAKEFILE_PATH)))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/..)
DEFAULT_BUILD_DIR := $(EXAMPLE_DIR)/build
FFTM_BUILD_SCOPE := examples
include $(PROJECT_ROOT)/build_configs/common.mk

ALL_BINS = \
	poisson_periodic_3d_autotuned.bin \
	poisson_periodic_4d_autotuned.bin \
	taylor_green_3d_autotuned.bin \
	taylor_green_spacetime_4d.bin
OPTIONAL_BINS = taylor_green_3d_autotuned_float.bin

.PHONY: all cuda hip hip-nca reader-smoke poisson-hip poisson-hip-nca $(ALL_BINS) $(OPTIONAL_BINS)

all: $(if $(filter hip,$(FFTM_BUILD_BACKEND)),$(if $(filter 1,$(FFTM_DEVICE_AWARE_MPI)),hip,hip-nca),cuda)
cuda: $(addprefix $(BUILD_DIR)/,$(ALL_BINS))
hip: poisson-hip
hip-nca: poisson-hip-nca

reader-smoke:
	bash $(EXAMPLE_DIR)/tests/run_reader_smoke.sh

poisson-hip:
	$(MAKE) -C $(EXAMPLE_DIR)/poisson hip BUILD_DIR="$(BUILD_DIR)"

poisson-hip-nca:
	$(MAKE) -C $(EXAMPLE_DIR)/poisson hip-nca BUILD_DIR="$(BUILD_DIR)"

$(ALL_BINS): %: $(BUILD_DIR)/%

$(OPTIONAL_BINS): %: $(BUILD_DIR)/%

$(BUILD_DIR):
	mkdir -p $@

POISSON_DIR := $(EXAMPLE_DIR)/poisson

$(BUILD_DIR)/poisson_periodic_3d_autotuned.bin: \
		$(POISSON_DIR)/Makefile \
		$(POISSON_DIR)/source/poisson_periodic_3d_autotuned.cpp fftm-poisson-force | $(BUILD_DIR)
	$(MAKE) -C $(POISSON_DIR) \
		poisson_periodic_3d_autotuned.bin \
		BUILD_DIR="$(BUILD_DIR)"

$(BUILD_DIR)/poisson_periodic_4d_autotuned.bin: \
		$(POISSON_DIR)/Makefile \
		$(POISSON_DIR)/source/poisson_periodic_4d_autotuned.cpp fftm-poisson-force | $(BUILD_DIR)
	$(MAKE) -C $(POISSON_DIR) \
		poisson_periodic_4d_autotuned.bin \
		BUILD_DIR="$(BUILD_DIR)"

.PHONY: fftm-poisson-force
fftm-poisson-force:

TURBULENCE_SOURCE := $(EXAMPLE_DIR)/turbulence/source
TURBULENCE_HEADERS := \
	$(TURBULENCE_SOURCE)/snapshot_writer.h \
	$(TURBULENCE_SOURCE)/spacetime_kernels.h \
	$(TURBULENCE_SOURCE)/spacetime_options.h \
	$(TURBULENCE_SOURCE)/taylor_green_kernels.h \
	$(TURBULENCE_SOURCE)/taylor_green_solver.h \
	$(TURBULENCE_SOURCE)/turbulence_options.h

$(BUILD_DIR)/taylor_green_3d_autotuned.bin: $(TURBULENCE_SOURCE)/taylor_green_3d_autotuned.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS_CA) $< -c -o $(BUILD_DIR)/taylor_green_3d_autotuned.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_3d_autotuned.o $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/taylor_green_3d_autotuned_float.bin: $(TURBULENCE_SOURCE)/taylor_green_3d_autotuned.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) -DFFTM_TURBULENCE_SINGLE_PRECISION=1 -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS_CA) $< -c -o $(BUILD_DIR)/taylor_green_3d_autotuned_float.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_3d_autotuned_float.o $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $@

$(BUILD_DIR)/taylor_green_spacetime_4d.bin: $(TURBULENCE_SOURCE)/taylor_green_spacetime_4d.cu $(TURBULENCE_HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_FLAGS) $(FFTM_DEPFLAGS) $(CPPFLAGS) -I$(source) -I$(include_cuda) $(MPI_CPPFLAGS) -I$(contrib) -I$(TURBULENCE_SOURCE) $(SCFD_FLAGS_CA) $< -c -o $(BUILD_DIR)/taylor_green_spacetime_4d.o
	$(MPICXX) -g $(BUILD_DIR)/taylor_green_spacetime_4d.o $(OMP) -L$(lib_cuda) $(MPI_LDFLAGS) $(LDFLAGS) -lcudart -lcufft $(MPI_LIBS) $(LDLIBS) -o $@

ifeq ($(FFTM_BUILD_BACKEND),hip)
ifneq ($(filter cuda taylor_green%,$(notdir $(MAKECMDGOALS))),)
$(error Taylor-Green examples are CUDA-only; select a CUDA configuration)
endif
endif

FFTM_BUILD_TARGETS := $(addprefix $(BUILD_DIR)/,$(filter taylor_green%,$(ALL_BINS)) $(OPTIONAL_BINS))
include $(PROJECT_ROOT)/build_configs/targets.mk
