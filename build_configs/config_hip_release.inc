FFTM_BUILD_BACKEND = hip
ROCM_DIR = /opt/rocm
mpi_dir = /usr/local/mpi
# Set to the architecture reported by rocminfo, e.g. gfx1102.
HIP_ARCH_LIST =
CPPSTD = c++14
TARGET_GCC = -O3
TARGET_HIP = -O3
# Keep ambient package-manager flags out of release builds; customize here.
CPPFLAGS =
CXXFLAGS =
LDFLAGS =
NVCC_LDFLAGS =
LDLIBS =
# Host-staged is the portable starting point. Set 1 only with ROCm-aware MPI.
FFTM_DEVICE_AWARE_MPI = 0
# BUILD_FOLDER = $(PROJECT_ROOT)/build/hip_release
