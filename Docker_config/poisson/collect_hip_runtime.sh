#!/usr/bin/env bash
set -euo pipefail
src=/opt/rocm-6.4.1
dst=/hip-runtime/opt/rocm-6.4.1
mkdir -p "$dst/lib" "$dst/share" "$dst/lib/llvm/lib/clang"
# rocFFT uses HIP runtime compilation and its shipped kernel cache at runtime.
for family in amdhip64 hiprtc hiprtc-builtins hipfft rocfft amd_comgr \
              hsa-runtime64 rocprofiler-register rocm-core; do
    cp -a "$src"/lib/lib"$family".so* "$dst/lib/"
done
cp -a "$src/lib/rocfft" "$dst/lib/"
cp -a "$src/include" "$dst/"
cp -a "$src/share/doc" "$dst/share/"
cp -a "$src/lib/llvm/lib/clang/19" "$dst/lib/llvm/lib/clang/"
cp -a "$src/amdgcn" "$dst/"
ln -s rocm-6.4.1 /hip-runtime/opt/rocm
mkdir -p /hip-runtime/opt/amdgpu/lib/x86_64-linux-gnu /hip-runtime/usr/share/doc
cp -a /opt/amdgpu/lib/x86_64-linux-gnu/libdrm*.so* /hip-runtime/opt/amdgpu/lib/x86_64-linux-gnu/
cp -a /opt/amdgpu/share /hip-runtime/opt/amdgpu/
cp -a /usr/share/doc/libdrm* /hip-runtime/usr/share/doc/
