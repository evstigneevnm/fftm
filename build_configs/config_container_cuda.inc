# Keep the validated container build flags; Docker build arguments may override.
FFTM_BUILD_BACKEND = cuda
cuda_dir = /usr/local/cuda
mpi_dir = /usr/local/mpi
CUDA_ARCH_LIST = 80 70
CPPSTD = c++14
TARGET_GCC = -O3
TARGET_NVCC = -O3
CPPFLAGS =
CXXFLAGS =
LDFLAGS =
NVCC_LDFLAGS =
LDLIBS =
