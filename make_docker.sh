#!/bin/bash

#use FFTM_DOCKER_TAG=name/type:tag
set -euo pipefail
DOCKER_TAG="${FFTM_DOCKER_TAG:-fftm/bench:a100}"
BASE_IMAGE="${FFTM_BASE_IMAGE:-nvcr.io/nvidia/hpc-benchmarks:24.06}"

printf 'Building %s from %s\n' "${DOCKER_TAG}" "${BASE_IMAGE}"
docker build \
    --build-arg "FFTM_BASE_IMAGE=${BASE_IMAGE}" \
    -f Docker_config/Dockerfile \
    . \
    -t "${DOCKER_TAG}"
