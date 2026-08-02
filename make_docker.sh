#!/bin/bash

#use FFTM_DOCKER_TAG=name/type:tag
set -euo pipefail
DOCKER_TAG="${FFTM_DOCKER_TAG:-fftm/bench:a100}"
BASE_IMAGE="${FFTM_BASE_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.06}"
GIT_COMMIT="${FFTM_GIT_COMMIT:-$(git rev-parse --verify HEAD 2>/dev/null || printf unknown)}"
GIT_DIRTY="${FFTM_GIT_DIRTY:-0}"

if [[ "${GIT_DIRTY}" == "0" ]] && \
   ! git diff --quiet --ignore-submodules -- 2>/dev/null; then
    GIT_DIRTY=1
fi
if [[ "${GIT_DIRTY}" == "0" ]] && \
   ! git diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
    GIT_DIRTY=1
fi

printf 'Building %s from %s (commit=%s dirty=%s)\n' \
    "${DOCKER_TAG}" "${BASE_IMAGE}" "${GIT_COMMIT}" "${GIT_DIRTY}"
docker build \
    --build-arg "FFTM_BASE_IMAGE=${BASE_IMAGE}" \
    --build-arg "FFTM_GIT_COMMIT=${GIT_COMMIT}" \
    --build-arg "FFTM_GIT_DIRTY=${GIT_DIRTY}" \
    -f Docker_config/Dockerfile \
    . \
    -t "${DOCKER_TAG}"
