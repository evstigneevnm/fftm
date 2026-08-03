#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

status=0

check_forbidden()
{
    local description="$1"
    local pattern="$2"
    shift 2

    local matches
    matches=$(rg -n "$pattern" "$@" || true)
    if [[ -n "$matches" ]]; then
        printf 'ERROR: %s\n%s\n' "$description" "$matches" >&2
        status=1
    fi
}

production_globs=(
    --glob '*.{h,hpp,inc,cu,cpp,cxx}'
    --glob '!source/external_wrap/**'
    --glob '!source/contrib/**'
    --glob '!source/tests/**'
    --glob '!source/fftm_backend.hpp'
)

check_forbidden \
    'vendor runtime/FFT headers escaped the approved backend boundaries' \
    '#[[:space:]]*include[[:space:]]*[<"][^>"]*(cuda|cufft|hip/|hipfft|scfd/(backend|memory)/(cuda|hip)|scfd/utils/(cuda|hip))' \
    "${production_globs[@]}" source examples

check_forbidden \
    'raw vendor runtime/FFT symbols escaped source/external_wrap' \
    '\b(cuda(Stream|Event|Error|Memcpy|Device|HostFn)|CUdeviceptr|cufft(Handle|Result|Type)|hip(Stream|Event|Error|Memcpy|Device|HostFn)|hipfft(Handle|Result|Type))\b|\b(cuda|cufft|hip|hipfft)[A-Z][A-Za-z0-9_]*[[:space:]]*\(|\b(CUDA_SAFE_CALL|CUFFT_SAFE_CALL|HIP_SAFE_CALL|HIPFFT_SAFE_CALL)\b' \
    "${production_globs[@]}" source examples

check_forbidden \
    'backend-selection macros escaped source/fftm_backend.hpp and test/build code' \
    '\b(FFTM_PLATFORM_HIP|PLATFORM_HIP|__HIPCC__|__CUDACC__)\b' \
    "${production_globs[@]}" source examples

check_forbidden \
    'the legacy SCFD CUDA-named MPI capability escaped its compatibility boundary' \
    '\bSCFD_COMMUNICATION_ENABLE_CUDA_AWARE_MPI\b' \
    "${production_globs[@]}" \
    --glob '!source/detail/device_aware_mpi_config.h' source examples

check_forbidden \
    'vendor-shaped 3D-copy fields escaped source/external_wrap' \
    '\.(srcPos|srcPtr|dstPos|dstPtr)\b' \
    "${production_globs[@]}" source examples

check_forbidden \
    'CUDA-specific local-transpose naming remains in backend-neutral code' \
    'cuda_memcpy_4d_slab_transposer' \
    "${production_globs[@]}" source examples

check_forbidden \
    'a wrapper exposes vendor copy descriptors or complex storage as its neutral API' \
    'using[[:space:]]+(memcpy_kind_t|memcpy_3d_params_t|complex)[[:space:]]*=[[:space:]]*(cuda|cufft|hip|hipfft)' \
    --glob '*.{h,hpp}' source/external_wrap

if (( status != 0 )); then
    exit "$status"
fi

echo 'Backend abstraction boundaries: OK'
