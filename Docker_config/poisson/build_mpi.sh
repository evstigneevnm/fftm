#!/usr/bin/env bash
set -euo pipefail
backend=$1
jobs=${BUILD_JOBS:-2}
mkdir -p /tmp/mpi-build /opt/fftm/dependencies
cd /deps
sha256sum -c SHA256SUMS
cp ./*.tar.* SHA256SUMS /opt/fftm/dependencies/
cd /tmp/mpi-build
tar -xf /deps/ucx-1.18.1.tar.gz
tar -xf /deps/openmpi-5.0.7.tar.gz
case "$backend" in
    cuda) gpu_args=(--with-cuda=/usr/local/cuda --without-rocm) ;;
    hip) gpu_args=(--with-rocm=/opt/rocm --without-cuda) ;;
    *) exit 2 ;;
esac
cd ucx-1.18.1
./configure --prefix=/opt/ucx --enable-mt --disable-static \
    --without-java --without-go "${gpu_args[@]}"
make -j"$jobs"
make install
cp LICENSE /opt/fftm/dependencies/UCX-LICENSE
cd ../openmpi-5.0.7
mpi_gpu_args=("${gpu_args[@]}")
if [[ $backend == cuda ]]; then
    # CUDA images also contain compat/libcuda.so; Open MPI 5.0.7 cannot infer two directories.
    mpi_gpu_args+=(--with-cuda-libdir=/usr/local/cuda/lib64/stubs)
fi
./configure --prefix=/opt/mpi --with-ucx=/opt/ucx "${mpi_gpu_args[@]}" \
    --with-pmix=internal --with-prrte=internal --with-hwloc=internal \
    --with-libevent=internal --disable-mpi-fortran --disable-oshmem \
    --disable-static --without-ucc
cp config.log /opt/fftm/dependencies/OpenMPI-config.log
if [[ $backend == cuda ]]; then
    grep -Eq '^#define OPAL_CUDA_SUPPORT +1$' opal/include/opal_config.h
else
    grep -Eq '^#define OPAL_ROCM_SUPPORT +1$' opal/include/opal_config.h
fi
make -j"$jobs"
make install
cp LICENSE /opt/fftm/dependencies/OpenMPI-LICENSE
rm -rf /tmp/mpi-build
