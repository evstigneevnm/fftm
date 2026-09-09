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
    /usr/local/cuda/bin/nvcc --version > /opt/fftm/metadata/compiler.txt
else
    test -s /opt/fftm/metadata/rocfft-rebuild.txt
    make -C examples/poisson CONFIG_FILE="$profile" -j"${BUILD_JOBS:-2}" hip hip-nca
    /opt/rocm/bin/hipcc --version > /opt/fftm/metadata/compiler.txt
fi
make -f Docker_config/poisson/Makefile.probe CONFIG_FILE="$profile" all
python3 Docker_config/poisson/check_architectures.py "$backend" \
    --config /opt/fftm/metadata/build-config.txt --bin-dir /opt/fftm/bin \
    --output /opt/fftm/metadata/architectures.json
printf '%s\n' "$backend" > /opt/fftm/metadata/backend
/opt/mpi/bin/ompi_info --all > /opt/fftm/metadata/ompi-info.txt
/opt/ucx/bin/ucx_info -v > /opt/fftm/metadata/ucx-version.txt
dpkg-query -W > /opt/fftm/metadata/build-packages.txt
rm -f /opt/fftm/bin/*.o /opt/fftm/bin/*.d /opt/fftm/bin/.fftm-*
sha256sum /opt/fftm/bin/* > /opt/fftm/metadata/binaries.sha256
