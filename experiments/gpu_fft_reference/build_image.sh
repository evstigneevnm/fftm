#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_IMAGE="${GPUFFT_REF_BASE_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.06}"
DOCKER_TAG="${GPUFFT_REF_DOCKER_TAG:-fftm/gpu-fft-reference:a100}"
SQSH_NAME="${GPUFFT_REF_SQSH_NAME:-fftm_gpu_fft_reference_a100}"
GPU_FFT_REPOSITORY="${GPUFFT_REF_REPOSITORY:-https://github.com/Manthan-Verma/GPU_FFT.git}"
GPU_FFT_COMMIT="${GPUFFT_REF_COMMIT:-ff4d84bbc6e0ecbcd52246c9d7e83198dadb2426}"
ENROOT_WORK_ROOT="${GPUFFT_REF_ENROOT_WORKDIR:-${ROOT_DIR}/enroot_workdir}"

if [[ "${ENROOT_WORK_ROOT}" != /* ]]; then
    ENROOT_WORK_ROOT="${ROOT_DIR}/${ENROOT_WORK_ROOT}"
fi
ENROOT_WORK_ROOT="$(realpath -m -- "${ENROOT_WORK_ROOT}")"

case "${ENROOT_WORK_ROOT}" in
    /|"${ROOT_DIR}")
        echo "Unsafe Enroot work directory: ${ENROOT_WORK_ROOT}" >&2
        exit 2
        ;;
esac

case "${GPUFFT_REF_KEEP_ENROOT_WORKDIR:-0}" in
    0|1) ;;
    *)
        echo "GPUFFT_REF_KEEP_ENROOT_WORKDIR must be 0 or 1." >&2
        exit 2
        ;;
esac

[[ ! -e "${SQSH_NAME}.sqsh" ]] || {
    echo "Refusing to overwrite ${SQSH_NAME}.sqsh" >&2
    exit 1
}

mkdir -p "${ENROOT_WORK_ROOT}"
ENROOT_RUN_WORKDIR="$(mktemp -d "${ENROOT_WORK_ROOT}/gpu-fft-reference.XXXXXXXX")"
chmod 0755 "${ENROOT_RUN_WORKDIR}"

cleanup_enroot_workdir()
{
    local status=$?
    if [[ "${GPUFFT_REF_KEEP_ENROOT_WORKDIR:-0}" == "1" ]]; then
        echo "Keeping Enroot work directory: ${ENROOT_RUN_WORKDIR}"
    else
        echo "Cleaning Enroot work directory: ${ENROOT_RUN_WORKDIR}"
        if ! rm -rf -- "${ENROOT_RUN_WORKDIR}" 2>/dev/null; then
            sudo rm -rf -- "${ENROOT_RUN_WORKDIR}"
        fi
        rmdir "${ENROOT_WORK_ROOT}" 2>/dev/null || true
    fi
    trap - EXIT
    exit "${status}"
}
trap cleanup_enroot_workdir EXIT

docker build \
    --build-arg "FFTM_BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "GPU_FFT_REPOSITORY=${GPU_FFT_REPOSITORY}" \
    --build-arg "GPU_FFT_COMMIT=${GPU_FFT_COMMIT}" \
    -f experiments/gpu_fft_reference/Dockerfile \
    -t "${DOCKER_TAG}" \
    .

FFTM_DOCKER_TAG="${DOCKER_TAG}" \
FFTM_SQSH_NAME="${SQSH_NAME}" \
FFTM_ENROOT_WORKDIR="${ENROOT_RUN_WORKDIR}" \
bash make_enroot.sh

sha256sum "${SQSH_NAME}.sqsh" > "${SQSH_NAME}.sqsh.sha256"
printf 'Built %s\n' "${ROOT_DIR}/${SQSH_NAME}.sqsh"
