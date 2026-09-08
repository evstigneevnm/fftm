#!/usr/bin/env bash
set -euo pipefail
dst=/cuda-runtime/usr/local/cuda
mkdir -p "$dst/lib64" "$dst/licenses"
# cuFFT may compile kernels at runtime; retain NVRTC and nvJitLink as well.
for family in cudart cufft nvJitLink nvrtc nvrtc-builtins; do
    cp -a /usr/local/cuda/lib64/lib"$family".so* "$dst/lib64/"
done
find /usr/local/cuda -maxdepth 3 -type f \( -iname '*license*' -o -iname '*eula*' \) \
    -exec cp --parents '{}' "$dst/licenses/" \;
find /usr/share/doc -maxdepth 1 -type d \( -name 'cuda*' -o -name 'libcufft*' \) \
    -exec cp -a '{}' "$dst/licenses/" \;
