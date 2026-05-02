#!/bin/bash

#use FFTM_DOCKER_TAG=name/type:tag
set -euo pipefail
DOCKER_TAG="${FFTM_DOCKER_TAG:-fftm/bench:a100}"
sudo docker build -f Docker_config/Dockerfile . -t ${DOCKER_TAG}
