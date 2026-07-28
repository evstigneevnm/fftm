#!/bin/bash
set -euo pipefail
DOCKER_TAG="${FFTM_DOCKER_TAG:-fftm/bench:a100}"

# Default: fftm/bench:a100 -> fftm_bench_a100.sqsh.
# Override with FFTM_SQSH_NAME if a custom output name is needed.
DEFAULT_SQSH_NAME="$(
    printf '%s' "${DOCKER_TAG}" |
        sed -E 's/[^[:alnum:]]+/_/g; s/^_+//; s/_+$//'
)"
SQSH_NAME="${FFTM_SQSH_NAME:-${DEFAULT_SQSH_NAME}}"

ENROOT_ENV=()
if [[ -n "${FFTM_ENROOT_WORKDIR:-}" ]]; then
    ENROOT_WORKDIR="${FFTM_ENROOT_WORKDIR}"
    if [[ "${ENROOT_WORKDIR}" != /* ]]; then
        ENROOT_WORKDIR="${PWD}/${ENROOT_WORKDIR}"
    fi
    ENROOT_WORKDIR="$(realpath -m -- "${ENROOT_WORKDIR}")"

    case "${ENROOT_WORKDIR}" in
        /)
            echo "Unsafe Enroot work directory: ${ENROOT_WORKDIR}" >&2
            exit 2
            ;;
    esac

    install -d -m 0755 \
        "${ENROOT_WORKDIR}/tmp" \
        "${ENROOT_WORKDIR}/cache/enroot" \
        "${ENROOT_WORKDIR}/data/enroot" \
        "${ENROOT_WORKDIR}/config/enroot"
    install -d -m 0700 \
        "${ENROOT_WORKDIR}/runtime" \
        "${ENROOT_WORKDIR}/runtime/enroot"

    ENROOT_ENV=(
        "TMP=${ENROOT_WORKDIR}/tmp"
        "TEMP=${ENROOT_WORKDIR}/tmp"
        "TMPDIR=${ENROOT_WORKDIR}/tmp"
        "XDG_CACHE_HOME=${ENROOT_WORKDIR}/cache"
        "XDG_CONFIG_HOME=${ENROOT_WORKDIR}/config"
        "XDG_DATA_HOME=${ENROOT_WORKDIR}/data"
        "XDG_RUNTIME_DIR=${ENROOT_WORKDIR}/runtime"
        "ENROOT_CACHE_PATH=${ENROOT_WORKDIR}/cache/enroot"
        "ENROOT_CONFIG_PATH=${ENROOT_WORKDIR}/config/enroot"
        "ENROOT_DATA_PATH=${ENROOT_WORKDIR}/data/enroot"
        "ENROOT_RUNTIME_PATH=${ENROOT_WORKDIR}/runtime/enroot"
        "ENROOT_TEMP_PATH=${ENROOT_WORKDIR}/tmp"
    )

    echo "Using Enroot work directory: ${ENROOT_WORKDIR}"
fi

env "${ENROOT_ENV[@]}" \
    enroot import -o "${SQSH_NAME}.sqsh" "dockerd://${DOCKER_TAG}"
