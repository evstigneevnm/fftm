cuda_dir=/usr/local/cuda
mpi_dir=/usr/local/mpi
contrib=../contrib/scfd/include
source=../
include_cuda=$(cuda_dir)/include 
include_mpi=$(mpi_dir)/include
lib_cuda=$(cuda_dir)/lib64 
lib_mpi=$(mpi_dir)/lib
OMP=-fopenmp -lpthread



test_init.bin test_init.cu:
	nvcc -g -G -std=c++14 -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) test_init.cu -c -o test_init.o
	mpicxx -g test_init.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_init.bin

test_cudaMemcpy3DAsync.bin test_cudaMemcpy3DAsync.cu:
	nvcc -g -G -std=c++14 -I$(source) -I$(include_cuda) -I$(contrib) test_cudaMemcpy3DAsync.cu -o test_cudaMemcpy3DAsync.bin