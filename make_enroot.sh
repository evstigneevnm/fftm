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

sudo enroot import -o "${SQSH_NAME}.sqsh" "dockerd://${DOCKER_TAG}"
