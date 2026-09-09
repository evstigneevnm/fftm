#!/usr/bin/env bash
set -euo pipefail
: "${HIP_ARCH:?Build through Makefile.probe with the HIP capsule profile}"
root=$(mktemp -d /tmp/fftm-rocfft.XXXXXX)
trap 'rm -rf "$root"' EXIT
prefix=$(readlink -f /opt/rocm)
test -s "$prefix/share/rocm/cmake/ROCMConfig.cmake"
mkdir -p "$root/source" /opt/fftm/metadata /opt/fftm/dependencies
tar -xf /deps/rocfft-rocm-6.4.1.tar.gz --strip-components=1 -C "$root/source"
cp /deps/sqlite-amalgamation-3430200.zip "$root/source/LICENSE.md" /opt/fftm/dependencies/
read -ra targets <<< "$HIP_ARCH"
gpu_targets=$(IFS=';'; printf '%s' "${targets[*]}")
# Reuse the shipped cache. Missing entries compile at runtime, not during image build.
cmake -S "$root/source" -B "$root/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER="$prefix/bin/hipcc" -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DROCM_PATH="$prefix" -DROCM_DIR="$prefix/share/rocm/cmake" \
    -DGPU_TARGETS="$gpu_targets" -DBUILD_CLIENTS=OFF -DUSE_HIPRAND=OFF \
    -DROCFFT_RUNTIME_COMPILE_DEFAULT=ON -DROCFFT_KERNEL_CACHE_ENABLE=OFF \
    -DSQLITE_3_43_2_SRC_URL=/deps/sqlite-amalgamation-3430200.zip
cmake --build "$root/build" --parallel "${BUILD_JOBS:-2}"
cp "$root/build/CMakeCache.txt" /opt/fftm/metadata/rocfft-CMakeCache.txt
cmake --install "$root/build"
cp "$root/build/install_manifest.txt" /opt/fftm/metadata/rocfft-install-manifest.txt
printf '%s\n' 'source_commit=058ba87fdcfdae334dbc8dbe048955b248e9328a' \
    'source_version=1.0.32' "gpu_targets=$gpu_targets" \
    'runtime_compile_default=ON' 'shipped_kernel_cache=retained' \
    > /opt/fftm/metadata/rocfft-rebuild.txt
