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
CUDA_ARCH=-gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 #-gencode arch=compute_35,code=sm_35 
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
NVCCFLAGS_NO_DEVICE_DEBUG = $(filter-out -G,$(NVCCFLAGS))



test_init.bin test_init.cu:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_init.cu -c -o test_init.o
	mpicxx -g test_init.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_init.bin

test_cudaMemcpy3DAsync.bin test_cudaMemcpy3DAsync.cu:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_cudaMemcpy3DAsync.cu -o test_cudaMemcpy3DAsync.bin

test_mpi_transpose_3D.bin: test_mpi_transpose_3D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_mpi_transpose_3D.cu -c -o test_mpi_transpose_3D.o
	mpicxx -g test_mpi_transpose_3D.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lmpi -o test_mpi_transpose_3D.bin

test_mpi_transpose_4D.bin: test_mpi_transpose_4D.cu
	nvcc $(NVCCFLAGS_NO_DEVICE_DEBUG) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_mpi_transpose_4D.cu -c -o test_mpi_transpose_4D.o
	mpicxx -g test_mpi_transpose_4D.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lmpi -o test_mpi_transpose_4D.bin

test_fftm_3D_compare.bin: test_fftm_3D_compare.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_fftm_3D_compare.cu -c -o test_fftm_3D_compare.o
	mpicxx -g test_fftm_3D_compare.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_fftm_3D_compare.bin

test_fftm_4D_compare.bin: test_fftm_4D_compare.cu
	nvcc $(NVCCFLAGS_NO_DEVICE_DEBUG) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_fftm_4D_compare.cu -c -o test_fftm_4D_compare.o
	mpicxx -g test_fftm_4D_compare.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_fftm_4D_compare.bin

test_3D_poisson_mpi.bin: test_3D_poisson_mpi.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_3D_poisson_mpi.cu -c -o test_3D_poisson_mpi.o
	mpicxx -g test_3D_poisson_mpi.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_3D_poisson_mpi.bin

test_3D_poisson_mpi_tutorial.bin: test_3D_poisson_mpi_tutorial.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_3D_poisson_mpi_tutorial.cu -c -o test_3D_poisson_mpi_tutorial.o
	mpicxx -g test_3D_poisson_mpi_tutorial.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_3D_poisson_mpi_tutorial.bin

test:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_cudaMemcpy3DAsync_1.cu -o test_cudaMemcpy3DAsync_1.bin


test_fft_2D.bin: test_fft_2D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_fft_2D.cu -L$(lib_cuda) -lcudart -lcufft -o test_fft_2D.bin

test_2D_poisson.bin: test_2D_poisson.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_2D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_2D_poisson.bin

test_3D_poisson.bin: test_3D_poisson.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_3D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_3D_poisson.bin

test_4D_poisson.bin: test_4D_poisson.cu
	nvcc $(NVCCFLAGS_NO_DEVICE_DEBUG) -I$(source) -I$(include_cuda) -I$(contrib) test_4D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_4D_poisson.bin

test_fft_3D.bin: test_fft_3D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) test_fft_3D.cu -L$(lib_cuda) -lcudart -lcufft -o test_fft_3D.bin
