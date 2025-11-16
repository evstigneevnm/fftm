TARGET_GCC = -g -G
TARGET_NVCC = -g -G #-O3
CPPSTD = c++14

cuda_dir=/usr/local/cuda
mpi_dir=/usr/local/mpi
contrib=../contrib/scfd/include
source=../
include_cuda=$(cuda_dir)/include 
include_mpi=$(mpi_dir)/include
lib_cuda=$(cuda_dir)/lib64 
lib_mpi=$(mpi_dir)/lib
OMP=-fopenmp -lpthread
CUDA_ARCH=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr



test_init.bin test_init.cu:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_init.cu -c -o test_init.o
	mpicxx -g test_init.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_init.bin

test_cudaMemcpy3DAsync.bin test_cudaMemcpy3DAsync.cu:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_cudaMemcpy3DAsync.cu -o test_cudaMemcpy3DAsync.bin

test:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_cudaMemcpy3DAsync_1.cu -o test_cudaMemcpy3DAsync_1.bin


test_fft_2D.bin: test_fft_2D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_fft_2D.cu -L$(lib_cuda) -lcudart -lcufft -o test_fft_2D.bin
