FFTM_BUILD_BACKEND = cuda
cuda_dir = /usr/local/cuda
mpi_dir = /opt/mpi
CUDA_ARCH_LIST = 60 61 70 75 80 86 89 90 100 103 120
CUDA_ARCH = $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch)) -gencode arch=compute_60,code=compute_60
TARGET_NVCC = -O3
CPPFLAGS =
CXXFLAGS =
LDFLAGS =
LDLIBS =
BUILD_FOLDER = /opt/fftm/bin
