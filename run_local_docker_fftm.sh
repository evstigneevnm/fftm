#!/usr/bin/env bash
set -euo pipefail

# Run FFTM benchmarks from a local Docker image and collect all output data
# into a host directory. This script does not run Egger and does not compare
# datasets; use the analysis/comparison scripts later after copying the data.
#
# First run, with image build:
#   FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi FFTM_DATA_DIR=build/paper_data/docker_v100x2 BUILD_IMAGE=1 ./run_local_docker_fftm.sh
#
# Repeat with an already-built image:
#   FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi FFTM_DATA_DIR=build/paper_data/docker_v100x2 ./run_local_docker_fftm.sh
#
# If Docker does not need sudo:
#   DOCKER_CMD=docker FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi FFTM_DATA_DIR=... ./run_local_docker_fftm.sh
#
# If Docker uses --gpus all without an explicit --runtime=nvidia:
#   DOCKER_RUNTIME=none FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi FFTM_DATA_DIR=... ./run_local_docker_fftm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"

IMAGE="${FFTM_DOCKER_IMAGE:-fftm/bench:v100-mpi}"
DATA_DIR="${FFTM_DATA_DIR:-${SCRIPT_DIR}/build/paper_data/docker_fftm_${STAMP}}"
DOCKER_CMD="${DOCKER_CMD:-${FFTM_DOCKER_CMD:-sudo docker}}"
DOCKER_RUNTIME="${DOCKER_RUNTIME:-${FFTM_DOCKER_RUNTIME:-nvidia}}"
DOCKER_GPUS="${DOCKER_GPUS:-${FFTM_DOCKER_GPUS:-all}}"
BUILD_IMAGE="${BUILD_IMAGE:-${FFTM_BUILD_IMAGE:-0}}"

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd)"

echo "FFTM Docker image: ${IMAGE}"
echo "Output data dir:   ${DATA_DIR}"
echo "Docker command:    ${DOCKER_CMD}"
echo "Docker runtime:    ${DOCKER_RUNTIME}"
echo "Docker GPUs:       ${DOCKER_GPUS}"

FFTM_DOCKER_CMD="${DOCKER_CMD}" \
FFTM_DOCKER_RUNTIME="${DOCKER_RUNTIME}" \
FFTM_DOCKER_GPUS="${DOCKER_GPUS}" \
FFTM_DOCKER_IMAGE="${IMAGE}" \
FFTM_BUILD_IMAGE="${BUILD_IMAGE}" \
FFTM_DATA_DIR="${DATA_DIR}" \
FFTM_GPU_COUNTS="${FFTM_GPU_COUNTS:-1,2}" \
FFTM_MAX_GPUS="${FFTM_MAX_GPUS:-2}" \
FFTM_DEVICE_MEMORY_MIB="${FFTM_DEVICE_MEMORY_MIB:-32768,32768}" \
FFTM_AUTO_MEMORY_FRACTION="${FFTM_AUTO_MEMORY_FRACTION:-0.72}" \
FFTM_AUTO_RESERVE_MEMORY_MIB="${FFTM_AUTO_RESERVE_MEMORY_MIB:-2048}" \
FFTM_BENCHMARK_SIZES_3D="${FFTM_BENCHMARK_SIZES_3D:-auto}" \
FFTM_BENCHMARK_SIZES_4D="${FFTM_BENCHMARK_SIZES_4D:-auto}" \
FFTM_EXTRA_SIZES_3D="${FFTM_EXTRA_SIZES_3D:-540,686,729,900}" \
FFTM_FIXED_SCALING_SIZES_3D="${FFTM_FIXED_SCALING_SIZES_3D:-${FFTM_FIXED_SCALING_SIZE_3D:-}}" \
FFTM_FIXED_SCALING_SIZES_4D="${FFTM_FIXED_SCALING_SIZES_4D:-${FFTM_FIXED_SCALING_SIZE_4D:-}}" \
FFTM_BENCHMARK_TIMES="${FFTM_BENCHMARK_TIMES:-5}" \
FFTM_WARMUP="${FFTM_WARMUP:-3}" \
FFTM_VALIDATION_TIMES="${FFTM_VALIDATION_TIMES:-1}" \
FFTM_MODES="${FFTM_MODES:-alltoallv,p2p-waitall,p2p-waitany}" \
FFTM_TRANSPORTS="${FFTM_TRANSPORTS:-cuda_aware,non_cuda_aware}" \
FFTM_INCLUDE_VERSIONED="${FFTM_INCLUDE_VERSIONED:-0}" \
FFTM_USE_DIRECT_BACKWARD_RECEIVE="${FFTM_USE_DIRECT_BACKWARD_RECEIVE:-0}" \
FFTM_DIRECT_P2P_CUDA_AWARE="${FFTM_DIRECT_P2P_CUDA_AWARE:-1}" \
FFTM_USE_P2P_SEND_THREAD="${FFTM_USE_P2P_SEND_THREAD:-0}" \
FFTM_USE_P2P_BYTE_TRANSFER="${FFTM_USE_P2P_BYTE_TRANSFER:-0}" \
FFTM_P2P_VARIANTS="${FFTM_P2P_VARIANTS:-configured}" \
FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS="${FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS:-both}" \
FFTM_CUDA_ARCH="${FFTM_CUDA_ARCH:--gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75}" \
FFTM_BUILD_JOBS="${FFTM_BUILD_JOBS:-$(nproc)}" \
scripts/run_container_local_benchmarks.sh

cat <<EOF

FFTM Docker run complete.

Data directory:
  ${DATA_DIR}

Host-side analysis command:
  python3 scripts/analyze_paper_results.py --data-directory "${DATA_DIR}"

Common overrides:
  FFTM_DOCKER_IMAGE=fftm/bench:v100-mpi
  FFTM_DATA_DIR=build/paper_data/docker_v100x2
  FFTM_GPU_COUNTS=1,2
  FFTM_EXTRA_SIZES_3D=540,686,729,900
  FFTM_FIXED_SCALING_SIZES_3D=1024
  FFTM_BENCHMARK_TIMES=10
  FFTM_WARMUP=3
EOF
