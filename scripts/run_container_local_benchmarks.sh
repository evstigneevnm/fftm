#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${FFTM_DOCKER_IMAGE:-fftm/bench:latest}"
DATA_DIR="${FFTM_DATA_DIR:-${REPO_ROOT}/build/paper_data/docker_local_smoke}"
DOCKERFILE="${FFTM_DOCKERFILE:-${REPO_ROOT}/Docker_config/Dockerfile}"
BUILD_IMAGE="${FFTM_BUILD_IMAGE:-0}"
DOCKER_CMD_STRING="${FFTM_DOCKER_CMD:-docker}"
DOCKER_GPUS="${FFTM_DOCKER_GPUS:-all}"
DOCKER_RUNTIME="${FFTM_DOCKER_RUNTIME:-none}"
DOCKER_GPU_FLAGS_STRING="${FFTM_DOCKER_GPU_FLAGS:-}"

BENCHMARK_SIZES_3D="${FFTM_BENCHMARK_SIZES_3D:-64}"
BENCHMARK_SIZES_4D="${FFTM_BENCHMARK_SIZES_4D:-16}"
VERSIONED_SIZE_3D="${FFTM_VERSIONED_SIZE_3D:-64}"
VERSIONED_SIZE_4D="${FFTM_VERSIONED_SIZE_4D:-16}"
BENCHMARK_TIMES="${FFTM_BENCHMARK_TIMES:-1}"
TEST_WARMUP="${FFTM_WARMUP:-${FFTM_BENCHMARK_WARMUP:-3}}"
VALIDATION_TIMES="${FFTM_VALIDATION_TIMES:-1}"
GPU_COUNTS="${FFTM_GPU_COUNTS:-auto}"
MODES="${FFTM_MODES:-p2p-waitany}"
TRANSPORTS="${FFTM_TRANSPORTS:-cuda_aware,non_cuda_aware}"
INCLUDE_VERSIONED="${FFTM_INCLUDE_VERSIONED:-1}"
VERSIONED_FULL_MATRIX="${FFTM_VERSIONED_FULL_MATRIX:-0}"
USE_DIRECT_BACKWARD_RECEIVE="${FFTM_USE_DIRECT_BACKWARD_RECEIVE:-0}"
DIRECT_P2P_CUDA_AWARE="${FFTM_DIRECT_P2P_CUDA_AWARE:-1}"
USE_P2P_SEND_THREAD="${FFTM_USE_P2P_SEND_THREAD:-0}"
USE_P2P_BYTE_TRANSFER="${FFTM_USE_P2P_BYTE_TRANSFER:-0}"
USE_PERSISTENT_P2P="${FFTM_USE_PERSISTENT_P2P:-0}"
USE_READY_P2P_SEND="${FFTM_USE_READY_P2P_SEND:-0}"
USE_DIRECT_FORWARD_BYTE_RECEIVE="${FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE:-0}"
USE_STABLE_FORWARD_BYTE_SEND_BUFFER="${FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER:-0}"
USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER="${FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER:-0}"
USE_CONTIGUOUS_FORWARD_BYTE_SEND="${FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND:-0}"
USE_PHYSICAL_FORWARD_PEER_EXCHANGE="${FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE:-0}"
USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT="${FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT:-0}"
USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY="${FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY:-${FFTM_USE_NATIVE_OPT0_EGGER_Y_BUFFER_TOPOLOGY:-0}}"
USE_NATIVE_OPT0_COMPACT_Y_WORKAREA="${FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA:-0}"
USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE="${FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE:-0}"
USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES="${FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES:-0}"
USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC="${FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC:-0}"
USE_NATIVE_OPT0_Y_NO_SYNC_EXEC="${FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC:-1}"
USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR="${FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR:-0}"
USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE="${FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE:-0}"
USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE="${FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE:-0}"
USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE="${FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE:-0}"
USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST="${FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST:-0}"
USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS="${FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS:-${FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_EGGER_STREAMS:-0}}"
USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT="${FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT:-${FFTM_USE_NATIVE_OPT0_EGGER_LOCAL_PLAN_CONTEXT:-0}}"
ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS="${FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS:-0}"
USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD="${FFTM_USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD:-1}"
NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB="${FFTM_NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB:-512}"
NATIVE_OPT0_Y_CROSS_MICROBENCH="${FFTM_NATIVE_OPT0_Y_CROSS_MICROBENCH:-0}"
NATIVE_OPT0_Y_CROSS_FACTORY_MODES="${FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES:-configured}"
NATIVE_OPT0_Y_MICROBENCH_ITERATIONS="${FFTM_NATIVE_OPT0_Y_MICROBENCH_ITERATIONS:-20}"
NATIVE_OPT0_Y_MICROBENCH_WARMUP="${FFTM_NATIVE_OPT0_Y_MICROBENCH_WARMUP:-3}"
ENABLE_GPU_TELEMETRY="${FFTM_ENABLE_GPU_TELEMETRY:-0}"
CONTIGUOUS_FORWARD_SEND_MODE="${FFTM_CONTIGUOUS_FORWARD_SEND_MODE:-single}"
CONTIGUOUS_FORWARD_SEND_MODES="${FFTM_CONTIGUOUS_FORWARD_SEND_MODES:-configured}"
CONTIGUOUS_FORWARD_SEND_CHUNK_MIB="${FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB:-1024}"
CONTIGUOUS_FORWARD_SEND_REGISTRATION_WARMUPS="${FFTM_CONTIGUOUS_FORWARD_SEND_REGISTRATION_WARMUPS:-0}"
USE_LARGE_COUNT_DATATYPE_CACHE="${FFTM_USE_LARGE_COUNT_DATATYPE_CACHE:-0}"
USE_FFT_EXEC_NO_SYNC="${FFTM_USE_FFT_EXEC_NO_SYNC:-0}"
P2P_VARIANTS="${FFTM_P2P_VARIANTS:-configured}"
P2P_SCHEDULERS="${FFTM_P2P_SCHEDULERS:-configured}"
PENCIL_LAYOUTS="${FFTM_PENCIL_LAYOUTS:-configured}"
PENCIL_PIPELINES="${FFTM_PENCIL_PIPELINES:-configured}"
LARGE_COUNT_P2P_TRANSPORTS="${FFTM_LARGE_COUNT_P2P_TRANSPORTS:-configured}"
PENCIL_PENCIL_GRID_ORIENTATIONS="${FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS:-both}"
EXTRA_SIZES_3D="${FFTM_EXTRA_SIZES_3D:-}"
EXTRA_SIZES_3D_BY_GPU="${FFTM_EXTRA_SIZES_3D_BY_GPU:-}"
FIXED_SCALING_SIZES_3D="${FFTM_FIXED_SCALING_SIZES_3D:-${FFTM_FIXED_SCALING_SIZE_3D:-}}"
FIXED_SCALING_SIZES_4D="${FFTM_FIXED_SCALING_SIZES_4D:-${FFTM_FIXED_SCALING_SIZE_4D:-}}"
MAX_GPUS="${FFTM_MAX_GPUS:-}"
DEVICE_MEMORY_MIB="${FFTM_DEVICE_MEMORY_MIB:-40960}"
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
BUILD_JOBS="${FFTM_BUILD_JOBS:-8}"
CUDA_ARCH="${FFTM_CUDA_ARCH:--gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70}"

read -r -a DOCKER_CMD <<< "${DOCKER_CMD_STRING}"
DOCKER_GPU_FLAGS=()
if [[ -n "${DOCKER_GPU_FLAGS_STRING}" ]]; then
    read -r -a DOCKER_GPU_FLAGS <<< "${DOCKER_GPU_FLAGS_STRING}"
else
    if [[ -n "${DOCKER_RUNTIME}" && "${DOCKER_RUNTIME}" != "none" ]]; then
        DOCKER_GPU_FLAGS+=(--runtime="${DOCKER_RUNTIME}")
    fi
    if [[ -n "${DOCKER_GPUS}" && "${DOCKER_GPUS}" != "none" ]]; then
        DOCKER_GPU_FLAGS+=(--gpus "${DOCKER_GPUS}")
    fi
fi

mkdir -p "${DATA_DIR}"

DEVICE_MEMORY_ARGS=()
read -r -a DEVICE_MEMORY_ARGS <<< "${DEVICE_MEMORY_MIB//,/ }"

if [[ "${BUILD_IMAGE}" == "1" ]]; then
    "${DOCKER_CMD[@]}" build \
        -f "${DOCKERFILE}" \
        --build-arg "CUDA_ARCH=${CUDA_ARCH}" \
        --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
        -t "${IMAGE}" \
        "${REPO_ROOT}"
fi

args=(
    python3 /opt/fftm/scripts/run_cluster_paper_benchmarks.py
    --executor local
    --tests-root /opt/fftm/bin
    --data-directory /data
    --gpu-counts "${GPU_COUNTS}"
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
    --p2p-variants "${P2P_VARIANTS}"
    --p2p-schedulers "${P2P_SCHEDULERS}"
    --pencil-layouts "${PENCIL_LAYOUTS}"
    --pencil-pipelines "${PENCIL_PIPELINES}"
    --large-count-p2p-transports "${LARGE_COUNT_P2P_TRANSPORTS}"
    --pencil-pencil-grid-orientations "${PENCIL_PENCIL_GRID_ORIENTATIONS}"
    --extra-sizes-3d "${EXTRA_SIZES_3D}"
    --extra-sizes-3d-by-gpu "${EXTRA_SIZES_3D_BY_GPU}"
    --fixed-scaling-sizes-3d "${FIXED_SCALING_SIZES_3D}"
    --fixed-scaling-sizes-4d "${FIXED_SCALING_SIZES_4D}"
)

if [[ ${#DEVICE_MEMORY_ARGS[@]} -gt 0 ]]; then
    args+=(--device-memory-mib "${DEVICE_MEMORY_ARGS[@]}")
fi

if [[ -n "${MAX_GPUS}" ]]; then
    args+=(--max-gpus "${MAX_GPUS}")
fi
if [[ -n "${AUTO_REFERENCE_SIZE_3D}" ]]; then
    args+=(--auto-reference-size-3d "${AUTO_REFERENCE_SIZE_3D}")
fi
if [[ -n "${AUTO_REFERENCE_SIZE_4D}" ]]; then
    args+=(--auto-reference-size-4d "${AUTO_REFERENCE_SIZE_4D}")
fi
if [[ -n "${AUTO_MAX_SIZE_3D}" ]]; then
    args+=(--auto-max-size-3d "${AUTO_MAX_SIZE_3D}")
fi
if [[ -n "${AUTO_MAX_SIZE_4D}" ]]; then
    args+=(--auto-max-size-4d "${AUTO_MAX_SIZE_4D}")
fi

case "${USE_DIRECT_BACKWARD_RECEIVE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-direct-backward-receive) ;;
    *) args+=(--no-direct-backward-receive) ;;
esac

case "${DIRECT_P2P_CUDA_AWARE}" in
    0|false|FALSE|no|NO|off|OFF) args+=(--no-direct-p2p-cuda-aware) ;;
    *) args+=(--direct-p2p-cuda-aware) ;;
esac

case "${USE_P2P_SEND_THREAD}" in
    0|false|FALSE|no|NO|off|OFF) args+=(--no-p2p-send-thread) ;;
    *) args+=(--use-p2p-send-thread) ;;
esac

case "${USE_P2P_BYTE_TRANSFER}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-p2p-byte-transfer) ;;
    *) args+=(--no-p2p-byte-transfer) ;;
esac

case "${USE_PERSISTENT_P2P}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-persistent-p2p) ;;
    *) args+=(--no-persistent-p2p) ;;
esac

case "${USE_READY_P2P_SEND}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-ready-p2p-send) ;;
    *) args+=(--no-ready-p2p-send) ;;
esac

case "${USE_DIRECT_FORWARD_BYTE_RECEIVE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-direct-forward-byte-receive) ;;
    *) args+=(--no-direct-forward-byte-receive) ;;
esac

case "${USE_STABLE_FORWARD_BYTE_SEND_BUFFER}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-stable-forward-byte-send-buffer) ;;
    *) args+=(--no-stable-forward-byte-send-buffer) ;;
esac

case "${USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-ready-stable-forward-byte-send-buffer) ;;
    *) args+=(--no-ready-stable-forward-byte-send-buffer) ;;
esac

case "${USE_CONTIGUOUS_FORWARD_BYTE_SEND}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-contiguous-forward-byte-send) ;;
    *) args+=(--no-contiguous-forward-byte-send) ;;
esac

case "${USE_PHYSICAL_FORWARD_PEER_EXCHANGE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-physical-forward-peer-exchange) ;;
    *) args+=(--no-physical-forward-peer-exchange) ;;
esac

case "${USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-default-z-layout) ;;
    *) args+=(--no-native-opt0-default-z-layout) ;;
esac

case "${USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-reference-y-buffer-topology) ;;
    *) args+=(--no-native-opt0-reference-y-buffer-topology) ;;
esac

case "${USE_NATIVE_OPT0_COMPACT_Y_WORKAREA}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-compact-y-workarea) ;;
    *) args+=(--no-native-opt0-compact-y-workarea) ;;
esac

case "${USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-tight-y-plan-sequence) ;;
    *) args+=(--no-native-opt0-tight-y-plan-sequence) ;;
esac

case "${USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-shared-y-plan-handles) ;;
    *) args+=(--no-native-opt0-shared-y-plan-handles) ;;
esac

case "${USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-y-group-device-sync) ;;
    *) args+=(--no-native-opt0-y-group-device-sync) ;;
esac

case "${USE_NATIVE_OPT0_Y_NO_SYNC_EXEC}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-y-no-sync-exec) ;;
    *) args+=(--no-native-opt0-y-no-sync-exec) ;;
esac

case "${USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-raw-y-plan-array-executor) ;;
    *) args+=(--no-native-opt0-raw-y-plan-array-executor) ;;
esac

case "${USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-reference-y-plan-lifecycle) ;;
    *) args+=(--no-native-opt0-reference-y-plan-lifecycle) ;;
esac

case "${USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-reference-y-plan-bundle) ;;
    *) args+=(--no-native-opt0-reference-y-plan-bundle) ;;
esac

case "${USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-raw-y-plan-bundle) ;;
    *) args+=(--no-native-opt0-raw-y-plan-bundle) ;;
esac

case "${USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-y-plan-bundle-stream-first) ;;
    *) args+=(--no-native-opt0-y-plan-bundle-stream-first) ;;
esac

case "${USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-raw-y-plan-bundle-reference-streams) ;;
    *) args+=(--no-native-opt0-raw-y-plan-bundle-reference-streams) ;;
esac

case "${USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-opt0-reference-local-plan-context) ;;
    *) args+=(--no-native-opt0-reference-local-plan-context) ;;
esac

case "${ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--allow-native-opt0-diagnostic-variants) ;;
    *) args+=(--no-native-opt0-diagnostic-variants) ;;
esac

case "${USE_NATIVE_OPT0_MEMORY_FEASIBILITY_GUARD}" in
    0|false|FALSE|no|NO|off|OFF) args+=(--no-native-opt0-memory-feasibility-guard) ;;
    *) args+=(--use-native-opt0-memory-feasibility-guard) ;;
esac
args+=(--native-opt0-memory-feasibility-reserve-mib "${NATIVE_OPT0_MEMORY_FEASIBILITY_RESERVE_MIB}")

case "${NATIVE_OPT0_Y_CROSS_MICROBENCH}" in
    1|true|TRUE|yes|YES|on|ON)
        args+=(
            --native-opt0-y-cross-microbench
            --native-opt0-y-cross-factory-modes "${NATIVE_OPT0_Y_CROSS_FACTORY_MODES}"
            --native-opt0-y-microbench-iterations "${NATIVE_OPT0_Y_MICROBENCH_ITERATIONS}"
            --native-opt0-y-microbench-warmup "${NATIVE_OPT0_Y_MICROBENCH_WARMUP}"
        )
        ;;
esac

case "${ENABLE_GPU_TELEMETRY}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--enable-gpu-telemetry) ;;
    *) args+=(--disable-gpu-telemetry) ;;
esac

args+=(
    --contiguous-forward-send-mode "${CONTIGUOUS_FORWARD_SEND_MODE}"
    --contiguous-forward-send-modes "${CONTIGUOUS_FORWARD_SEND_MODES}"
    --contiguous-forward-send-chunk-mib "${CONTIGUOUS_FORWARD_SEND_CHUNK_MIB}"
    --contiguous-forward-send-registration-warmups "${CONTIGUOUS_FORWARD_SEND_REGISTRATION_WARMUPS}"
)

case "${USE_LARGE_COUNT_DATATYPE_CACHE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-large-count-datatype-cache) ;;
    *) args+=(--no-large-count-datatype-cache) ;;
esac

case "${USE_FFT_EXEC_NO_SYNC}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-fft-exec-no-sync) ;;
    *) args+=(--no-fft-exec-no-sync) ;;
esac

if [[ "${INCLUDE_VERSIONED}" == "1" ]]; then
    args+=(--include-versioned)
fi
if [[ "${VERSIONED_FULL_MATRIX}" == "1" ]]; then
    args+=(--versioned-full-matrix)
fi

"${DOCKER_CMD[@]}" run --rm \
    "${DOCKER_GPU_FLAGS[@]}" \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES="${DOCKER_GPUS}" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e OMPI_ALLOW_RUN_AS_ROOT=1 \
    -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    -v "${DATA_DIR}:/data" \
    "${IMAGE}" \
    "${args[@]}"

cat <<EOF

Local container benchmark data:
  ${DATA_DIR}

To analyze it on the host:
  python3 scripts/analyze_paper_results.py --data-directory "${DATA_DIR}"
EOF
