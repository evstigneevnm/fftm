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
AUTOTUNE_CONFIG="${FFTM_AUTOTUNE_CONFIG:-}"
WRITE_NATIVE_PENCIL_SCHEDULE="${FFTM_WRITE_NATIVE_PENCIL_SCHEDULE:-0}"
NATIVE_PENCIL_REFERENCE_DIR="${FFTM_NATIVE_PENCIL_REFERENCE_DIR:-}"
NATIVE_PENCIL_SCHEDULE_CHECK_ONLY="${FFTM_NATIVE_PENCIL_SCHEDULE_CHECK_ONLY:-0}"
SKIP_NATIVE_PENCIL_RANK_DEVICE_CHECK="${FFTM_SKIP_NATIVE_PENCIL_RANK_DEVICE_CHECK:-0}"
RUN_PREFLIGHT="${FFTM_RUN_PREFLIGHT:-1}"
PREFLIGHT_TIME="${FFTM_PREFLIGHT_TIME:-00:03:00}"
STOP_ON_FAILURE="${FFTM_STOP_ON_FAILURE:-1}"
SKIP_FFTS="${FFTM_SKIP_FFTS:-0}"
SKIP_FFTM="${FFTM_SKIP_FFTM:-0}"
DRY_RUN="${FFTM_DRY_RUN:-0}"

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
STRATEGIES_3D="${FFTM_STRATEGIES_3D:-slab-pencil,pencil-slab,pencil-pencil}"
STRATEGIES_4D="${FFTM_STRATEGIES_4D:-slab-slab,pencil-pencil}"
INCLUDE_VERSIONED="${FFTM_INCLUDE_VERSIONED:-0}"
VERSIONED_FULL_MATRIX="${FFTM_VERSIONED_FULL_MATRIX:-0}"
USE_DIRECT_BACKWARD_RECEIVE="${FFTM_USE_DIRECT_BACKWARD_RECEIVE:-0}"
DIRECT_P2P_CUDA_AWARE="${FFTM_DIRECT_P2P_CUDA_AWARE:-1}"
USE_P2P_SEND_THREAD="${FFTM_USE_P2P_SEND_THREAD:-0}"
USE_P2P_BYTE_TRANSFER="${FFTM_USE_P2P_BYTE_TRANSFER:-0}"
USE_PERSISTENT_P2P="${FFTM_USE_PERSISTENT_P2P:-0}"
USE_READY_P2P_SEND="${FFTM_USE_READY_P2P_SEND:-0}"
PRINT_PENCIL_SCHEDULE="${FFTM_PRINT_PENCIL_SCHEDULE:-0}"
USE_DIRECT_FORWARD_BYTE_RECEIVE="${FFTM_USE_DIRECT_FORWARD_BYTE_RECEIVE:-0}"
USE_STABLE_FORWARD_BYTE_SEND_BUFFER="${FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER:-0}"
USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER="${FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER:-0}"
USE_CONTIGUOUS_FORWARD_BYTE_SEND="${FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND:-0}"
USE_PHYSICAL_FORWARD_PEER_EXCHANGE="${FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE:-0}"
USE_NATIVE_BACKWARD_SECOND_PEER_LOOP="${FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP:-0}"
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
NATIVE_OPT0_Y_EXECUTOR_VARIANTS="${FFTM_NATIVE_OPT0_Y_EXECUTOR_VARIANTS:-configured}"
NATIVE_BACKWARD_SECOND_PEER_LOOP_MODES="${FFTM_NATIVE_BACKWARD_SECOND_PEER_LOOP_MODES:-configured}"
ENABLE_FFTM3D_BACKEND_STAGE_TIMERS="${FFTM_ENABLE_FFTM3D_BACKEND_STAGE_TIMERS:-0}"
ENABLE_LOCAL_FFT_DIAGNOSTICS="${FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS:-0}"
NATIVE_OPT0_Y_MICROBENCH="${FFTM_NATIVE_OPT0_Y_MICROBENCH:-0}"
NATIVE_OPT0_Y_CROSS_MICROBENCH="${FFTM_NATIVE_OPT0_Y_CROSS_MICROBENCH:-0}"
NATIVE_OPT0_Y_CROSS_FACTORY_MODES="${FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES:-configured}"
NATIVE_OPT0_Y_MICROBENCH_ITERATIONS="${FFTM_NATIVE_OPT0_Y_MICROBENCH_ITERATIONS:-20}"
NATIVE_OPT0_Y_MICROBENCH_WARMUP="${FFTM_NATIVE_OPT0_Y_MICROBENCH_WARMUP:-3}"
ENABLE_NATIVE_STAGE_TIMERS="${FFTM_ENABLE_NATIVE_STAGE_TIMERS:-0}"
ENABLE_GPU_TELEMETRY="${FFTM_ENABLE_GPU_TELEMETRY:-0}"
CONTIGUOUS_FORWARD_SEND_MODE="${FFTM_CONTIGUOUS_FORWARD_SEND_MODE:-single}"
CONTIGUOUS_FORWARD_SEND_MODES="${FFTM_CONTIGUOUS_FORWARD_SEND_MODES:-configured}"
FFTM_4D_SLAB_XW_TRANSPOSES="${FFTM_4D_SLAB_XW_TRANSPOSES:-configured}"
FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS="${FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS:-configured}"
FFTM_4D_SLAB_XW_KERNEL_LAYOUTS="${FFTM_4D_SLAB_XW_KERNEL_LAYOUTS:-configured}"
FFTM_4D_SLAB_XW_VECTOR4_KERNELS="${FFTM_4D_SLAB_XW_VECTOR4_KERNELS:-configured}"
FFTM_4D_SLAB_XW_TILED_KERNELS="${FFTM_4D_SLAB_XW_TILED_KERNELS:-configured}"
FFTM_4D_SLAB_XW_LAYOUT_STAGES="${FFTM_4D_SLAB_XW_LAYOUT_STAGES:-configured}"
FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS="${FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS:-configured}"
CONTIGUOUS_FORWARD_SEND_CHUNK_MIB="${FFTM_CONTIGUOUS_FORWARD_SEND_CHUNK_MIB:-1024}"
CONTIGUOUS_FORWARD_SEND_REGISTRATION_WARMUPS="${FFTM_CONTIGUOUS_FORWARD_SEND_REGISTRATION_WARMUPS:-0}"
USE_LARGE_COUNT_DATATYPE_CACHE="${FFTM_USE_LARGE_COUNT_DATATYPE_CACHE:-0}"
USE_FFT_EXEC_NO_SYNC="${FFTM_USE_FFT_EXEC_NO_SYNC:-0}"
P2P_VARIANTS="${FFTM_P2P_VARIANTS:-configured}"
P2P_SCHEDULERS="${FFTM_P2P_SCHEDULERS:-configured}"
PENCIL_LAYOUTS="${FFTM_PENCIL_LAYOUTS:-configured}"
PENCIL_PIPELINES="${FFTM_PENCIL_PIPELINES:-configured}"
LARGE_COUNT_P2P_TRANSPORTS="${FFTM_LARGE_COUNT_P2P_TRANSPORTS:-configured}"
FFTM_3D_BACKENDS="${FFTM_3D_BACKENDS:-${FFTM_3D_BACKEND:-configured}}"
PENCIL_PENCIL_GRID_ORIENTATIONS="${FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS:-both}"
EXTRA_SIZES_3D="${FFTM_EXTRA_SIZES_3D-}"
EXTRA_SIZES_3D_BY_GPU="${FFTM_EXTRA_SIZES_3D_BY_GPU-1:1050;2:1344;3:1536;4:1680;5:1800,2048;6:1920,2048;7:2025,2048;8:2100,2048}"
FIXED_SCALING_SIZES_3D="${FFTM_FIXED_SCALING_SIZES_3D:-${FFTM_FIXED_SCALING_SIZE_3D:-}}"
FIXED_SCALING_SIZES_4D="${FFTM_FIXED_SCALING_SIZES_4D:-${FFTM_FIXED_SCALING_SIZE_4D:-}}"
GPU_NAME="${FFTM_GPU_NAME:-A100}"
DEVICE_MEMORY_MIB="${FFTM_DEVICE_MEMORY_MIB:-40960}"
TIMEOUT_SECONDS="${FFTM_TIMEOUT_SECONDS:-7200}"

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"

if [[ "${FFTM_CONTAINER_IMAGE}" == /* && ! -f "${FFTM_CONTAINER_IMAGE}" ]]; then
    cat >&2 <<EOF
ERROR: FFTM_CONTAINER_IMAGE does not exist:
  ${FFTM_CONTAINER_IMAGE}

Build/copy the .sqsh image first, or set FFTM_CONTAINER_IMAGE to the correct
Pyxis image URI/path before launching the benchmark matrix.
EOF
    exit 2
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: data directory was not created: ${DATA_DIR}" >&2
    exit 2
fi

if ! ( : > "${DATA_DIR}/.fftm_write_test" ) 2>/dev/null; then
    echo "ERROR: data directory is not writable: ${DATA_DIR}" >&2
    exit 2
fi
rm -f "${DATA_DIR}/.fftm_write_test"

if [[ -n "${CONTAINER_MOUNTS}" ]]; then
    IFS=',' read -r -a preflight_mount_array <<< "${CONTAINER_MOUNTS}"
    for mount in "${preflight_mount_array[@]}"; do
        [[ -z "${mount}" ]] && continue
        src="${mount%%:*}"
        if [[ "${src}" == /* && ! -e "${src}" ]]; then
            echo "ERROR: container mount source does not exist: ${src}" >&2
            exit 2
        fi
    done
fi

container_mounts_arg="${DATA_DIR}:${CONTAINER_DATA_DIR}"
if [[ -n "${CONTAINER_MOUNTS}" ]]; then
    container_mounts_arg="${container_mounts_arg},${CONTAINER_MOUNTS}"
fi

contains_csv()
{
    local haystack=",$1,"
    local needle=",$2,"
    [[ "${haystack}" == *"${needle}"* ]]
}

dim_is_active()
{
    local benchmark_sizes="$1"
    local fixed_sizes="$2"
    local extra_sizes="$3"
    local extra_sizes_by_gpu="${4:-}"

    case "${benchmark_sizes}" in
        none|off|skip)
            [[ -n "${fixed_sizes}${extra_sizes}${extra_sizes_by_gpu}" ]]
            ;;
        *)
            return 0
            ;;
    esac
}

selected_preflight_bins=()
if dim_is_active "${BENCHMARK_SIZES_3D}" "${FIXED_SCALING_SIZES_3D}" "${EXTRA_SIZES_3D}" "${EXTRA_SIZES_3D_BY_GPU}"; then
    if [[ "${SKIP_FFTS}" != "1" && "${SKIP_FFTS}" != "true" && "${SKIP_FFTS}" != "TRUE" ]]; then
        selected_preflight_bins+=( "test_benchmark_ffts_3D.bin" )
    fi
    if [[ "${SKIP_FFTM}" != "1" && "${SKIP_FFTM}" != "true" && "${SKIP_FFTM}" != "TRUE" ]]; then
        contains_csv "${TRANSPORTS}" "cuda_aware" && selected_preflight_bins+=( "test_benchmark_fftm_3D.bin" )
        contains_csv "${TRANSPORTS}" "non_cuda_aware" && selected_preflight_bins+=( "test_benchmark_fftm_3D_nca.bin" )
    fi
fi
if dim_is_active "${BENCHMARK_SIZES_4D}" "${FIXED_SCALING_SIZES_4D}" "" ""; then
    if [[ "${SKIP_FFTS}" != "1" && "${SKIP_FFTS}" != "true" && "${SKIP_FFTS}" != "TRUE" ]]; then
        selected_preflight_bins+=( "test_benchmark_ffts_4D.bin" )
    fi
    if [[ "${SKIP_FFTM}" != "1" && "${SKIP_FFTM}" != "true" && "${SKIP_FFTM}" != "TRUE" ]]; then
        contains_csv "${TRANSPORTS}" "cuda_aware" && selected_preflight_bins+=( "test_benchmark_fftm_4D.bin" )
        contains_csv "${TRANSPORTS}" "non_cuda_aware" && selected_preflight_bins+=( "test_benchmark_fftm_4D_nca.bin" )
    fi
fi

preflight_test_script="set -e; test -d '${CONTAINER_DATA_DIR}' || { echo 'missing container data directory: ${CONTAINER_DATA_DIR}' >&2; exit 41; }; test -w '${CONTAINER_DATA_DIR}' || { echo 'container data directory is not writable: ${CONTAINER_DATA_DIR}' >&2; exit 42; }"
for bin_name in "${selected_preflight_bins[@]}"; do
    preflight_test_script="${preflight_test_script}; test -x '${CONTAINER_TESTS_ROOT}/${bin_name}' || { echo 'missing or non-executable benchmark binary: ${CONTAINER_TESTS_ROOT}/${bin_name}' >&2; ls -l '${CONTAINER_TESTS_ROOT}' >&2 || true; exit 43; }"
done

if [[ "${RUN_PREFLIGHT}" != "0" && "${RUN_PREFLIGHT}" != "false" && "${RUN_PREFLIGHT}" != "FALSE" &&
      "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && "${DRY_RUN}" != "TRUE" ]]; then
    preflight_cmd=(
        srun
        -N 1
        -n 1
        -G 1
        --gpus-per-node=1
        --time="${PREFLIGHT_TIME}"
        --container-image "${FFTM_CONTAINER_IMAGE}"
        --container-mounts="${container_mounts_arg}"
        --container-workdir "${CONTAINER_WORKDIR}"
        --container-entrypoint
    )
    if [[ -n "${SRUN_EXTRA_ARGS}" ]]; then
        read -r -a srun_extra_array <<< "${SRUN_EXTRA_ARGS}"
        preflight_cmd=( srun "${srun_extra_array[@]}" "${preflight_cmd[@]:1}" )
    fi
    if [[ -n "${CONTAINER_ENV}" ]]; then
        preflight_cmd+=( "--container-env=${CONTAINER_ENV}" )
    fi
    preflight_cmd+=(
        /bin/bash
        -lc
        "${preflight_test_script}"
    )

    echo "Running Pyxis preflight check..."
    "${preflight_cmd[@]}"
fi

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
    "--srun-extra-args=${SRUN_EXTRA_ARGS}"
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
    --p2p-variants "${P2P_VARIANTS}"
    --p2p-schedulers "${P2P_SCHEDULERS}"
    --pencil-layouts "${PENCIL_LAYOUTS}"
    --pencil-pipelines "${PENCIL_PIPELINES}"
    --large-count-p2p-transports "${LARGE_COUNT_P2P_TRANSPORTS}"
    --fftm-3d-backends "${FFTM_3D_BACKENDS}"
    --extra-sizes-3d "${EXTRA_SIZES_3D}"
    --extra-sizes-3d-by-gpu "${EXTRA_SIZES_3D_BY_GPU}"
    --fixed-scaling-sizes-3d "${FIXED_SCALING_SIZES_3D}"
    --fixed-scaling-sizes-4d "${FIXED_SCALING_SIZES_4D}"
    --modes "${MODES}"
    --transports "${TRANSPORTS}"
    --strategies-3d "${STRATEGIES_3D}"
    --strategies-4d "${STRATEGIES_4D}"
    --pencil-pencil-grid-orientations "${PENCIL_PENCIL_GRID_ORIENTATIONS}"
    --native-opt0-y-executor-variants "${NATIVE_OPT0_Y_EXECUTOR_VARIANTS}"
    --fftm-4d-slab-xw-transposes "${FFTM_4D_SLAB_XW_TRANSPOSES}"
    --fftm-4d-slab-xw-batched-peer-kernels "${FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS}"
    --fftm-4d-slab-xw-kernel-layouts "${FFTM_4D_SLAB_XW_KERNEL_LAYOUTS}"
    --fftm-4d-slab-xw-vector4-kernels "${FFTM_4D_SLAB_XW_VECTOR4_KERNELS}"
    --fftm-4d-slab-xw-tiled-kernels "${FFTM_4D_SLAB_XW_TILED_KERNELS}"
    --fftm-4d-slab-xw-layout-stages "${FFTM_4D_SLAB_XW_LAYOUT_STAGES}"
    --fftm-4d-slab-xw-native-spectral-layouts "${FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS}"
    --timeout-seconds "${TIMEOUT_SECONDS}"
)

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

case "${PRINT_PENCIL_SCHEDULE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--print-pencil-schedule) ;;
    *) args+=(--no-print-pencil-schedule) ;;
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

case "${USE_NATIVE_BACKWARD_SECOND_PEER_LOOP}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--use-native-backward-second-peer-loop) ;;
    *) args+=(--no-native-backward-second-peer-loop) ;;
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

case "${ENABLE_FFTM3D_BACKEND_STAGE_TIMERS}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--enable-fftm3d-backend-stage-timers) ;;
    *) args+=(--disable-fftm3d-backend-stage-timers) ;;
esac

case "${ENABLE_LOCAL_FFT_DIAGNOSTICS}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--enable-local-fft-diagnostics) ;;
    *) args+=(--disable-local-fft-diagnostics) ;;
esac

case "${NATIVE_OPT0_Y_MICROBENCH}" in
    1|true|TRUE|yes|YES|on|ON)
        args+=(
            --native-opt0-y-microbench
            --native-opt0-y-microbench-iterations "${NATIVE_OPT0_Y_MICROBENCH_ITERATIONS}"
            --native-opt0-y-microbench-warmup "${NATIVE_OPT0_Y_MICROBENCH_WARMUP}"
        )
        ;;
esac

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

case "${ENABLE_NATIVE_STAGE_TIMERS}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--enable-native-stage-timers) ;;
    *) args+=(--disable-native-stage-timers) ;;
esac

case "${ENABLE_GPU_TELEMETRY}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--enable-gpu-telemetry) ;;
    *) args+=(--disable-gpu-telemetry) ;;
esac

args+=(
    --native-backward-second-peer-loop-modes "${NATIVE_BACKWARD_SECOND_PEER_LOOP_MODES}"
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

if [[ -n "${AUTOTUNE_CONFIG}" ]]; then
    args+=(--fftm-autotune-config "${AUTOTUNE_CONFIG}")
fi

case "${WRITE_NATIVE_PENCIL_SCHEDULE}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--write-native-pencil-schedule) ;;
esac

if [[ -n "${NATIVE_PENCIL_REFERENCE_DIR}" ]]; then
    args+=(--native-pencil-reference-dir "${NATIVE_PENCIL_REFERENCE_DIR}")
fi

case "${NATIVE_PENCIL_SCHEDULE_CHECK_ONLY}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--native-pencil-schedule-check-only) ;;
esac

case "${SKIP_NATIVE_PENCIL_RANK_DEVICE_CHECK}" in
    1|true|TRUE|yes|YES|on|ON) args+=(--skip-native-pencil-rank-device-check) ;;
esac

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
if [[ "${STOP_ON_FAILURE}" != "0" && "${STOP_ON_FAILURE}" != "false" && "${STOP_ON_FAILURE}" != "FALSE" ]]; then
    args+=(--stop-on-failure)
fi
if [[ "${SKIP_FFTS}" == "1" || "${SKIP_FFTS}" == "true" || "${SKIP_FFTS}" == "TRUE" ]]; then
    args+=(--skip-ffts)
fi
if [[ "${SKIP_FFTM}" == "1" || "${SKIP_FFTM}" == "true" || "${SKIP_FFTM}" == "TRUE" ]]; then
    args+=(--skip-fftm)
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "TRUE" ]]; then
    args+=(--dry-run)
fi

"${args[@]}"

cat <<EOF

Cluster benchmark data:
  ${DATA_DIR}

Copy this directory back locally, then run:
  python3 scripts/analyze_paper_results.py --data-directory /path/to/copied/data
EOF
