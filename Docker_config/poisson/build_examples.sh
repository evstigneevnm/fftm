#!/usr/bin/env bash
set -euo pipefail
backend=$1
profile="Docker_config/poisson/config_${backend}.inc"
cd /opt/fftm/src
mkdir -p /opt/fftm/bin /opt/fftm/metadata
cp provenance.json source.sha256 /opt/fftm/metadata/
make -C examples/poisson CONFIG_FILE="$profile" print-config check-config \
    | tee /opt/fftm/metadata/build-config.txt
if [[ $backend == cuda ]]; then
    make -C examples/poisson CONFIG_FILE="$profile" -j"${BUILD_JOBS:-2}" cuda cuda-nca
    /usr/local/cuda/bin/nvcc -std=c++14 -O2 --expt-relaxed-constexpr \
        -Wno-deprecated-gpu-targets -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
        -gencode arch=compute_86,code=sm_86 -x cu \
        -I source -I source/contrib/scfd/include -I/opt/mpi/include \
        -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t Docker_config/poisson/probe.cpp \
        -L/opt/mpi/lib -lmpi -lcufft -Xcompiler=-fopenmp \
        -o /opt/fftm/bin/capsule_probe.bin
    /usr/local/cuda/bin/nvcc --version > /opt/fftm/metadata/compiler.txt
else
    make -C examples/poisson CONFIG_FILE="$profile" -j"${BUILD_JOBS:-2}" hip hip-nca
    /opt/rocm/bin/hipcc -std=c++14 -O2 --offload-arch=gfx1102 -x hip \
        -DFFTM_PLATFORM_HIP -DPLATFORM_HIP -DSCFD_BACKEND_ENABLE_MPI \
        -I source -I source/contrib/scfd/include -I/opt/mpi/include \
        -DSCFD_ARRAYS_ORDINAL_TYPE=ptrdiff_t Docker_config/poisson/probe.cpp \
        -L/opt/mpi/lib -lmpi -lhipfft -o /opt/fftm/bin/capsule_probe.bin
    /opt/rocm/bin/hipcc --version > /opt/fftm/metadata/compiler.txt
fi
printf '%s\n' "$backend" > /opt/fftm/metadata/backend
/opt/mpi/bin/ompi_info --all > /opt/fftm/metadata/ompi-info.txt
/opt/ucx/bin/ucx_info -v > /opt/fftm/metadata/ucx-version.txt
dpkg-query -W > /opt/fftm/metadata/build-packages.txt
rm -f /opt/fftm/bin/*.o /opt/fftm/bin/*.d /opt/fftm/bin/.fftm-*
sha256sum /opt/fftm/bin/* > /opt/fftm/metadata/binaries.sha256
