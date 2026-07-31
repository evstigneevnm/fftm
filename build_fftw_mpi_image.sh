#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

BASE_IMAGE="${FFTW_CPU_BASE_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.06}"
DOCKER_TAG="${FFTW_CPU_DOCKER_TAG:-fftm/fftw-mpi:cpu}"
SQSH_NAME="${FFTW_CPU_SQSH_NAME:-fftm_fftw_mpi_cpu}"
WORK_DIR="${FFTW_CPU_ENROOT_WORKDIR:-enroot_workdir}"
BUILD_JOBS="${FFTW_CPU_BUILD_JOBS:-8}"

if [[ "${WORK_DIR}" != /* ]]; then
    WORK_DIR="${SCRIPT_DIR}/${WORK_DIR}"
fi
WORK_DIR="$(realpath -m -- "${WORK_DIR}")"
SQSH_PATH="${SCRIPT_DIR}/${SQSH_NAME}.sqsh"

case "${WORK_DIR}" in
    /|"${SCRIPT_DIR}")
        echo "Unsafe Enroot work directory: ${WORK_DIR}" >&2
        exit 2
        ;;
esac

if [[ -e "${SQSH_PATH}" ]]; then
    echo "Output already exists: ${SQSH_PATH}" >&2
    echo "Set FFTW_CPU_SQSH_NAME to a fresh name or remove the old image explicitly." >&2
    exit 2
fi

cleanup()
{
    if [[ -d "${WORK_DIR}" ]]; then
        rm -rf -- "${WORK_DIR}"
    fi
}
trap cleanup EXIT

printf 'Building Docker image %s from %s\n' "${DOCKER_TAG}" "${BASE_IMAGE}"
docker build \
    --build-arg "FFTM_BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
    -f Docker_config/Dockerfile.fftw_mpi \
    -t "${DOCKER_TAG}" \
    .

install -d -m 0755 \
    "${WORK_DIR}/tmp" \
    "${WORK_DIR}/cache/enroot" \
    "${WORK_DIR}/data/enroot" \
    "${WORK_DIR}/config/enroot"
install -d -m 0700 \
    "${WORK_DIR}/runtime" \
    "${WORK_DIR}/runtime/enroot"

printf 'Exporting %s\n' "${SQSH_PATH}"
env \
    TMP="${WORK_DIR}/tmp" \
    TEMP="${WORK_DIR}/tmp" \
    TMPDIR="${WORK_DIR}/tmp" \
    XDG_CACHE_HOME="${WORK_DIR}/cache" \
    XDG_CONFIG_HOME="${WORK_DIR}/config" \
    XDG_DATA_HOME="${WORK_DIR}/data" \
    XDG_RUNTIME_DIR="${WORK_DIR}/runtime" \
    ENROOT_CACHE_PATH="${WORK_DIR}/cache/enroot" \
    ENROOT_CONFIG_PATH="${WORK_DIR}/config/enroot" \
    ENROOT_DATA_PATH="${WORK_DIR}/data/enroot" \
    ENROOT_RUNTIME_PATH="${WORK_DIR}/runtime/enroot" \
    ENROOT_TEMP_PATH="${WORK_DIR}/tmp" \
    enroot import -o "${SQSH_PATH}" "dockerd://${DOCKER_TAG}"

chown "$(id -u):$(id -g)" "${SQSH_PATH}"
sha256sum "${SQSH_PATH}" > "${SQSH_PATH}.sha256"
ls -lh "${SQSH_PATH}" "${SQSH_PATH}.sha256"
