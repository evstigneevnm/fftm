TARGET_GCC = -O3# -g
TARGET_NVCC = -O3#-g -G #-O3
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
CUDA_ARCH=-gencode arch=compute_70,code=sm_70# -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 #-gencode arch=compute_35,code=sm_35 
NVCCFLAGS = -Wno-deprecated-gpu-targets $(CUDA_ARCH) -std=$(CPPSTD) $(TARGET_NVCC) --expt-relaxed-constexpr
GCCFLAGS = -std=$(CPPSTD) $(TARGET_GCC) 
SCFD_FLAGS = -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t 
SCFD_DEVICE_AWARE = -DSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI
SCFD_FLAGS_CA = $(SCFD_FLAGS) $(SCFD_DEVICE_AWARE)
SCFD_FLAGS_NCA = $(SCFD_FLAGS)
NVCCFLAGS_NO_DEVICE_DEBUG = $(filter-out -G,$(NVCCFLAGS))

define BUILD_MPI_TEST
$1: $2
	nvcc $3 -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $4 $2 -c -o $5
	mpicxx -g $5 $(OMP) -L$(lib_cuda) -L$(lib_mpi) $6 -o $1

endef



test_init.bin: test_init.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(SCFD_FLAGS) test_init.cu -c -o test_init.o
	mpicxx -g test_init.o $(OMP) -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_init.bin

test_cudaMemcpy3DAsync.bin: test_cudaMemcpy3DAsync.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_cudaMemcpy3DAsync.cu -o test_cudaMemcpy3DAsync.bin

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_mpi_transpose_3D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_3D_nca.bin,test_mpi_transpose_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_mpi_transpose_3D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_mpi_transpose_4D.o,-lcudart -lmpi))
$(eval $(call BUILD_MPI_TEST,test_mpi_transpose_4D_nca.bin,test_mpi_transpose_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_mpi_transpose_4D_nca.o,-lcudart -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_fftm_3D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_3D_compare_nca.bin,test_fftm_3D_compare.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_fftm_3D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_fftm_4D_compare.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_fftm_4D_compare_nca.bin,test_fftm_4D_compare.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_fftm_4D_compare_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi.bin,test_3D_poisson_mpi.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_3D_poisson_mpi.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_nca.bin,test_3D_poisson_mpi.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_3D_poisson_mpi_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_tutorial.bin,test_3D_poisson_mpi_tutorial.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_3D_poisson_mpi_tutorial.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_3D_poisson_mpi_tutorial_nca.bin,test_3D_poisson_mpi_tutorial.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_3D_poisson_mpi_tutorial_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi.bin,test_4D_poisson_mpi.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_4D_poisson_mpi.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_nca.bin,test_4D_poisson_mpi.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_4D_poisson_mpi_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_tutorial.bin,test_4D_poisson_mpi_tutorial.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_4D_poisson_mpi_tutorial.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_4D_poisson_mpi_tutorial_nca.bin,test_4D_poisson_mpi_tutorial.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_4D_poisson_mpi_tutorial_nca.o,-lcudart -lcufft -lmpi))

test_benchmark_ffts_3D.bin: test_benchmark_ffts_3D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(SCFD_FLAGS) test_benchmark_ffts_3D.cu -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_benchmark_ffts_3D.bin

test_benchmark_ffts_4D.bin: test_benchmark_ffts_4D.cu
	nvcc $(NVCCFLAGS_NO_DEVICE_DEBUG) -I$(source) -I$(include_cuda) -I$(include_mpi) -I$(contrib) $(SCFD_FLAGS) test_benchmark_ffts_4D.cu -L$(lib_cuda) -L$(lib_mpi) -lcudart -lcufft -lmpi -o test_benchmark_ffts_4D.bin

$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_3D.bin,test_benchmark_fftm_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_CA),test_benchmark_fftm_3D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_3D_nca.bin,test_benchmark_fftm_3D.cu,$(NVCCFLAGS),$(SCFD_FLAGS_NCA),test_benchmark_fftm_3D_nca.o,-lcudart -lcufft -lmpi))

$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_4D.bin,test_benchmark_fftm_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_CA),test_benchmark_fftm_4D.o,-lcudart -lcufft -lmpi))
$(eval $(call BUILD_MPI_TEST,test_benchmark_fftm_4D_nca.bin,test_benchmark_fftm_4D.cu,$(NVCCFLAGS_NO_DEVICE_DEBUG),$(SCFD_FLAGS_NCA),test_benchmark_fftm_4D_nca.o,-lcudart -lcufft -lmpi))

test:
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_cudaMemcpy3DAsync_1.cu -o test_cudaMemcpy3DAsync_1.bin


test_fft_2D.bin: test_fft_2D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_fft_2D.cu -L$(lib_cuda) -lcudart -lcufft -o test_fft_2D.bin

test_2D_poisson.bin: test_2D_poisson.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_2D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_2D_poisson.bin

test_3D_poisson.bin: test_3D_poisson.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_3D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_3D_poisson.bin

test_4D_poisson.bin: test_4D_poisson.cu
	nvcc $(NVCCFLAGS_NO_DEVICE_DEBUG) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_4D_poisson.cu -L$(lib_cuda) -lcudart -lcufft -o test_4D_poisson.bin

test_fft_3D.bin: test_fft_3D.cu
	nvcc $(NVCCFLAGS) -I$(source) -I$(include_cuda) -I$(contrib) $(SCFD_FLAGS) test_fft_3D.cu -L$(lib_cuda) -lcudart -lcufft -o test_fft_3D.bin

all: test_init.bin test_fft_2D.bin test_2D_poisson.bin test_3D_poisson.bin test_4D_poisson.bin test_fft_3D.bin test_mpi_transpose_3D.bin test_mpi_transpose_3D_nca.bin test_mpi_transpose_4D.bin test_mpi_transpose_4D_nca.bin test_fftm_3D_compare.bin test_fftm_3D_compare_nca.bin test_fftm_4D_compare.bin test_fftm_4D_compare_nca.bin test_3D_poisson_mpi.bin test_3D_poisson_mpi_nca.bin test_3D_poisson_mpi_tutorial.bin test_3D_poisson_mpi_tutorial_nca.bin test_4D_poisson_mpi.bin test_4D_poisson_mpi_nca.bin test_4D_poisson_mpi_tutorial.bin test_4D_poisson_mpi_tutorial_nca.bin test_benchmark_ffts_3D.bin test_benchmark_ffts_4D.bin test_benchmark_fftm_3D.bin test_benchmark_fftm_3D_nca.bin test_benchmark_fftm_4D.bin test_benchmark_fftm_4D_nca.bin
