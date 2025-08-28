cuda_dir=/usr/local/cuda
contrib=../contrib/scfd/include
source=../ 
lib=$(cuda_dir)/lib64


deb:
	nvcc -g -G -std=c++14 -I$(source) -I$(contrib) test_init.cu -L$(lib) -lcufft -o test_init.bin
