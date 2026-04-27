#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the .sqsh image path or Pyxis image URI}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${FFTM_DATA_DIR:-${PWD}/fftm_cluster_data_${STAMP}}"

NODE_COUNTS="${FFTM_NODE_COUNTS:-1}"
GPU_COUNTS="${FFTM_GPU_COUNTS:-auto}"
MAX_GPUS="${FFTM_MAX_GPUS:-}"
GPUS_PER_NODE="${FFTM_GPUS_PER_NODE:-8}"
SRUN_TIME="${FFTM_SRUN_TIME:-00:20:00}"
SRUN_EXTRA_ARGS="${FFTM_SRUN_EXTRA_ARGS:-}"
CONTAINER_WORKDIR="${FFTM_CONTAINER_WORKDIR:-/opt/fftm/bin}"
CONTAINER_DATA_DIR="${FFTM_CONTAINER_DATA_DIR:-/data}"
CONTAINER_TESTS_ROOT="${FFTM_CONTAINER_TESTS_ROOT:-/opt/fftm/bin}"
CONTAINER_ENV="${FFTM_CONTAINER_ENV:-}"
CONTAINER_MOUNTS="${FFTM_CONTAINER_MOUNTS:-}"

BENCHMARK_SIZES_3D="${FFTM_BENCHMARK_SIZES_3D:-auto}"
BENCHMARK_SIZES_4D="${FFTM_BENCHMARK_SIZES_4D:-auto}"
VERSIONED_SIZE_3D="${FFTM_VERSIONED_SIZE_3D:-128}"
VERSIONED_SIZE_4D="${FFTM_VERSIONED_SIZE_4D:-32}"
BENCHMARK_TIMES="${FFTM_BENCHMARK_TIMES:-3}"
TEST_WARMUP="${FFTM_WARMUP:-${FFTM_BENCHMARK_WARMUP:-3}}"
VALIDATION_TIMES="${FFTM_VALIDATION_TIMES:-1}"
AUTO_MEMORY_FRACTION="${FFTM_AUTO_MEMORY_FRACTION:-0.72}"
AUTO_RESERVE_MEMORY_MIB="${FFTM_AUTO_RESERVE_MEMORY_MIB:-2048}"
AUTO_BYTES_PER_POINT_3D="${FFTM_AUTO_BYTES_PER_POINT_3D:-48}"
AUTO_BYTES_PER_POINT_4D="${FFTM_AUTO_BYTES_PER_POINT_4D:-40}"
AUTO_REFERENCE_SIZE_3D="${FFTM_AUTO_REFERENCE_SIZE_3D:-}"
AUTO_REFERENCE_SIZE_4D="${FFTM_AUTO_REFERENCE_SIZE_4D:-}"
AUTO_MIN_SIZE_3D="${FFTM_AUTO_MIN_SIZE_3D:-64}"
AUTO_MIN_SIZE_4D="${FFTM_AUTO_MIN_SIZE_4D:-16}"
AUTO_MAX_SIZE_3D="${FFTM_AUTO_MAX_SIZE_3D:-}"
AUTO_MAX_SIZE_4D="${FFTM_AUTO_MAX_SIZE_4D:-}"
MODES="${FFTM_MODES:-alltoallv,p2p-waitall,p2p-waitany}"
TRANSPORTS="${FFTM_TRANSPORTS:-cuda_aware,non_cuda_aware}"
INCLUDE_VERSIONED="${FFTM_INCLUDE_VERSIONED:-0}"
VERSIONED_FULL_MATRIX="${FFTM_VERSIONED_FULL_MATRIX:-0}"
GPU_NAME="${FFTM_GPU_NAME:-A100}"
DEVICE_MEMORY_MIB="${FFTM_DEVICE_MEMORY_MIB:-40960}"
TIMEOUT_SECONDS="${FFTM_TIMEOUT_SECONDS:-7200}"

mkdir -p "${DATA_DIR}"

args=(
    python3 "${REPO_ROOT}/scripts/run_cluster_paper_benchmarks.py"
    --executor slurm-pyxis
    --container-image "${FFTM_CONTAINER_IMAGE}"
    --container-workdir "${CONTAINER_WORKDIR}"
    --container-tests-root "${CONTAINER_TESTS_ROOT}"
    --container-data-directory "${CONTAINER_DATA_DIR}"
    --data-directory "${DATA_DIR}"
    --node-counts "${NODE_COUNTS}"
    --gpu-counts "${GPU_COUNTS}"
    --gpus-per-node "${GPUS_PER_NODE}"
    --srun-time "${SRUN_TIME}"
    --srun-extra-args "${SRUN_EXTRA_ARGS}"
    --gpu-name "${GPU_NAME}"
    --device-memory-mib "${DEVICE_MEMORY_MIB}"
    --benchmark-sizes-3d "${BENCHMARK_SIZES_3D}"
    --benchmark-sizes-4d "${BENCHMARK_SIZES_4D}"
    --versioned-size-3d "${VERSIONED_SIZE_3D}"
    --versioned-size-4d "${VERSIONED_SIZE_4D}"
    --benchmark-times "${BENCHMARK_TIMES}"
    --warmup "${TEST_WARMUP}"
    --validation-times "${VALIDATION_TIMES}"
    --auto-memory-fraction "${AUTO_MEMORY_FRACTION}"
    --auto-reserve-memory-mib "${AUTO_RESERVE_MEMORY_MIB}"
    --auto-bytes-per-point-3d "${AUTO_BYTES_PER_POINT_3D}"
    --auto-bytes-per-point-4d "${AUTO_BYTES_PER_POINT_4D}"
    --auto-min-size-3d "${AUTO_MIN_SIZE_3D}"
    --auto-min-size-4d "${AUTO_MIN_SIZE_4D}"
    --modes "${MODES}"
    --transports "${TRANSPORTS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
)

if [[ -n "${MAX_GPUS}" ]]; then
    args+=(--max-gpus "${MAX_GPUS}")
fi
if [[ -n "${AUTO_MAX_SIZE_3D}" ]]; then
    args+=(--auto-max-size-3d "${AUTO_MAX_SIZE_3D}")
fi
if [[ -n "${AUTO_MAX_SIZE_4D}" ]]; then
    args+=(--auto-max-size-4d "${AUTO_MAX_SIZE_4D}")
fi
if [[ -n "${AUTO_REFERENCE_SIZE_3D}" ]]; then
    args+=(--auto-reference-size-3d "${AUTO_REFERENCE_SIZE_3D}")
fi
if [[ -n "${AUTO_REFERENCE_SIZE_4D}" ]]; then
    args+=(--auto-reference-size-4d "${AUTO_REFERENCE_SIZE_4D}")
fi

if [[ -n "${CONTAINER_ENV}" ]]; then
    args+=(--container-env "${CONTAINER_ENV}")
fi

if [[ -n "${CONTAINER_MOUNTS}" ]]; then
    IFS=',' read -r -a mount_array <<< "${CONTAINER_MOUNTS}"
    for mount in "${mount_array[@]}"; do
        [[ -n "${mount}" ]] && args+=(--container-mounts "${mount}")
    done
fi

if [[ "${INCLUDE_VERSIONED}" == "1" ]]; then
    args+=(--include-versioned)
fi
if [[ "${VERSIONED_FULL_MATRIX}" == "1" ]]; then
    args+=(--versioned-full-matrix)
fi

"${args[@]}"

cat <<EOF

Cluster benchmark data:
  ${DATA_DIR}

Copy this directory back locally, then run:
  python3 scripts/analyze_paper_results.py --data-directory /path/to/copied/data
EOF
