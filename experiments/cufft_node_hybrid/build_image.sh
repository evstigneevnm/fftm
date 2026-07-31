#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

BASE_IMAGE="${FFTM_NODE_HYBRID_BASE_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.06}"
DOCKER_TAG="${FFTM_NODE_HYBRID_DOCKER_TAG:-fftm/node-hybrid:a100}"
SQSH_NAME="${FFTM_NODE_HYBRID_SQSH_NAME:-fftm_node_hybrid_a100}"
ENROOT_WORK_ROOT="${FFTM_NODE_HYBRID_ENROOT_WORKDIR:-${ROOT_DIR}/enroot_workdir}"

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

case "${FFTM_NODE_HYBRID_KEEP_ENROOT_WORKDIR:-0}" in
    0|1)
        ;;
    *)
        echo "FFTM_NODE_HYBRID_KEEP_ENROOT_WORKDIR must be 0 or 1." >&2
        exit 2
        ;;
esac

[[ ! -e "${SQSH_NAME}.sqsh" ]] || {
    echo "Refusing to overwrite ${SQSH_NAME}.sqsh" >&2
    exit 1
}

mkdir -p "${ENROOT_WORK_ROOT}"
ENROOT_RUN_WORKDIR="$(
    mktemp -d "${ENROOT_WORK_ROOT}/fftm-node-hybrid.XXXXXXXX"
)"
chmod 0755 "${ENROOT_RUN_WORKDIR}"

cleanup_enroot_workdir()
{
    local status=$?

    if [[ "${FFTM_NODE_HYBRID_KEEP_ENROOT_WORKDIR:-0}" == "1" ]]; then
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
    -f experiments/cufft_node_hybrid/Dockerfile \
    -t "${DOCKER_TAG}" \
    .

FFTM_DOCKER_TAG="${DOCKER_TAG}" \
FFTM_SQSH_NAME="${SQSH_NAME}" \
FFTM_ENROOT_WORKDIR="${ENROOT_RUN_WORKDIR}" \
bash make_enroot.sh

sha256sum "${SQSH_NAME}.sqsh" > "${SQSH_NAME}.sqsh.sha256"
printf 'Built %s\n' "${ROOT_DIR}/${SQSH_NAME}.sqsh"
