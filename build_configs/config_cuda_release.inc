# Copy to config_local.inc and edit for this machine, or select with CONFIG_FILE.
FFTM_BUILD_BACKEND = cuda
cuda_dir = /usr/local/cuda
mpi_dir = /usr/local/mpi
CUDA_ARCH_LIST = 70 75
CPPSTD = c++14
TARGET_GCC = -O3
TARGET_NVCC = -O3
# Keep ambient package-manager flags out of release builds; customize here.
CPPFLAGS =
CXXFLAGS =
LDFLAGS =
NVCC_LDFLAGS =
LDLIBS =
# CUDA_HOST_CXX = /usr/bin/g++-13
# BUILD_FOLDER = $(PROJECT_ROOT)/build/cuda_release
