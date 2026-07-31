#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_7day_multinode_smoke.sh [target]

Targets:
  all              8G and 16G, 3D and 4D (4 cases)
  single-node      8G, 3D and 4D (2 cases)
  multinode        16G, 3D and 4D (2 cases, default)
  3d-single        8G 3D only (1 case)
  4d-single        8G 4D only (1 case)
  3d-multinode     16G 3D only (1 case)
  4d-multinode     16G 4D only (1 case)
  transport-selection
                   16G CUDA-aware vs host-staged MPI, 3D and 4D (4 cases)
  3d-transport-control
                   16G 3D CA parity, CA reference, and NCA reference (3 cases)
  hca-selection    16G CUDA-aware application check with GPU-local HCA:
                   3D grids 2x8/4x4 and 4D WZ concurrency 4/8 (4 cases)
  strong-scaling   Fixed 2048^3 and 320^4 on 8/16/32 GPUs (6 cases)
  capacity-scaling Memory-scaled 3D and 4D sizes on 8/16/32 GPUs (6 cases)
  paper-scaling    Strong + capacity scaling in one matrix (11 cases)

Deployment overrides:
  FFTM_CONTAINER_IMAGE, FFTM_DATA_DIR, FFTM_TIMEOUT_SECONDS,
  FFTM_STOP_ON_FAILURE, FFTM_SRUN_TIME, FFTM_SRUN_EXTRA_ARGS,
  FFTM_MPI_RANK_AFFINITY_MODE,
  FFTM_TRANSPORTS, FFTM_PENCIL_LAYOUTS, FFTM_PENCIL_PIPELINES,
  FFTM_RUN_PREFLIGHT,
  FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS
EOF
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

TARGET="${1:-multinode}"

BENCHMARK_SIZES_3D="none"
BENCHMARK_SIZES_4D="none"
AUTO_MEMORY_FRACTION="0.82"
AUTO_BYTES_PER_POINT_3D="65.5"
AUTO_BYTES_PER_POINT_4D="32.2"
DEFAULT_BENCHMARK_TIMES="5"
DEFAULT_WARMUP="3"
DEFAULT_TIMEOUT_SECONDS="600"
DEFAULT_SRUN_TIME="00:20:00"
DEFAULT_TRANSPORTS="cuda_aware"
DEFAULT_PENCIL_LAYOUTS="auto"
DEFAULT_PENCIL_PIPELINES="reference-parity"
DEFAULT_PENCIL_GRID_ORIENTATIONS="production"
DEFAULT_RUN_PREFLIGHT="0"
DEFAULT_MPI_RANK_AFFINITY_MODE="auto"
DEFAULT_WZ_PLAN_CONCURRENCIES="4"

case "${TARGET}" in
    -h|--help)
        usage
        exit 0
        ;;
    all)
        NODE_COUNTS="1,2"
        GPU_COUNTS="8,16"
        FIXED_SIZE_3D="1024"
        FIXED_SIZE_4D="192"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_STOP_ON_FAILURE="0"
        ;;
    single-node)
        NODE_COUNTS="1"
        GPU_COUNTS="8"
        FIXED_SIZE_3D="1024"
        FIXED_SIZE_4D="192"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    multinode)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="1024"
        FIXED_SIZE_4D="192"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    3d-single)
        NODE_COUNTS="1"
        GPU_COUNTS="8"
        FIXED_SIZE_3D="1024"
        FIXED_SIZE_4D="none"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D=""
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    4d-single)
        NODE_COUNTS="1"
        GPU_COUNTS="8"
        FIXED_SIZE_3D="none"
        FIXED_SIZE_4D="192"
        STRATEGIES_3D=""
        STRATEGIES_4D="slab-slab"
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    3d-multinode)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="1024"
        FIXED_SIZE_4D="none"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D=""
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    4d-multinode)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="none"
        FIXED_SIZE_4D="192"
        STRATEGIES_3D=""
        STRATEGIES_4D="slab-slab"
        DEFAULT_STOP_ON_FAILURE="1"
        ;;
    transport-selection)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="2048"
        FIXED_SIZE_4D="320"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_TRANSPORTS="cuda_aware,non_cuda_aware"
        DEFAULT_PENCIL_LAYOUTS="opt0"
        DEFAULT_PENCIL_GRID_ORIENTATIONS="2x8"
        DEFAULT_BENCHMARK_TIMES="10"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        DEFAULT_RUN_PREFLIGHT="1"
        ;;
    3d-transport-control)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="2048"
        FIXED_SIZE_4D="none"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D=""
        DEFAULT_TRANSPORTS="cuda_aware,non_cuda_aware"
        DEFAULT_PENCIL_LAYOUTS="opt0"
        DEFAULT_PENCIL_PIPELINES="reference-parity,reference"
        DEFAULT_PENCIL_GRID_ORIENTATIONS="2x8"
        DEFAULT_BENCHMARK_TIMES="10"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        DEFAULT_RUN_PREFLIGHT="1"
        ;;
    hca-selection)
        NODE_COUNTS="2"
        GPU_COUNTS="16"
        FIXED_SIZE_3D="2048"
        FIXED_SIZE_4D="320"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_TRANSPORTS="cuda_aware"
        DEFAULT_PENCIL_LAYOUTS="opt0"
        DEFAULT_PENCIL_GRID_ORIENTATIONS="2x8,4x4"
        DEFAULT_MPI_RANK_AFFINITY_MODE="hca"
        DEFAULT_WZ_PLAN_CONCURRENCIES="4,8"
        DEFAULT_BENCHMARK_TIMES="10"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        DEFAULT_RUN_PREFLIGHT="1"
        ;;
    strong-scaling)
        NODE_COUNTS="1,2,4"
        GPU_COUNTS="8,16,32"
        FIXED_SIZE_3D="2048"
        FIXED_SIZE_4D="320"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_BENCHMARK_TIMES="20"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        ;;
    capacity-scaling)
        NODE_COUNTS="1,2,4"
        GPU_COUNTS="8,16,32"
        BENCHMARK_SIZES_3D="auto"
        BENCHMARK_SIZES_4D="auto"
        FIXED_SIZE_3D="none"
        FIXED_SIZE_4D="none"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_BENCHMARK_TIMES="20"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        ;;
    paper-scaling)
        NODE_COUNTS="1,2,4"
        GPU_COUNTS="8,16,32"
        BENCHMARK_SIZES_3D="auto"
        BENCHMARK_SIZES_4D="auto"
        FIXED_SIZE_3D="2048"
        FIXED_SIZE_4D="320"
        STRATEGIES_3D="pencil-pencil"
        STRATEGIES_4D="slab-slab"
        DEFAULT_BENCHMARK_TIMES="20"
        DEFAULT_WARMUP="5"
        DEFAULT_TIMEOUT_SECONDS="7200"
        DEFAULT_SRUN_TIME="00:45:00"
        DEFAULT_STOP_ON_FAILURE="0"
        ;;
    *)
        echo "Unknown smoke target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}"
DATA_DIR="${FFTM_DATA_DIR:-/scratch/evstigneevnm/fftm/data_3d4d_${TARGET}_$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_TIMES="${FFTM_BENCHMARK_TIMES:-${DEFAULT_BENCHMARK_TIMES}}"
WARMUP="${FFTM_WARMUP:-${DEFAULT_WARMUP}}"
TIMEOUT_SECONDS="${FFTM_TIMEOUT_SECONDS:-${DEFAULT_TIMEOUT_SECONDS}}"
STOP_ON_FAILURE="${FFTM_STOP_ON_FAILURE:-${DEFAULT_STOP_ON_FAILURE}}"
SRUN_TIME="${FFTM_SRUN_TIME:-${DEFAULT_SRUN_TIME}}"
SRUN_EXTRA_ARGS="${FFTM_SRUN_EXTRA_ARGS:---exclude=cn13 --ntasks-per-node=8 --distribution=block:block --kill-on-bad-exit=1}"
TRANSPORTS="${FFTM_TRANSPORTS:-${DEFAULT_TRANSPORTS}}"
PENCIL_LAYOUTS="${FFTM_PENCIL_LAYOUTS:-${DEFAULT_PENCIL_LAYOUTS}}"
PENCIL_PIPELINES="${FFTM_PENCIL_PIPELINES:-${DEFAULT_PENCIL_PIPELINES}}"
PENCIL_GRID_ORIENTATIONS="${FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS:-${DEFAULT_PENCIL_GRID_ORIENTATIONS}}"
RUN_PREFLIGHT="${FFTM_RUN_PREFLIGHT:-${DEFAULT_RUN_PREFLIGHT}}"
DRY_RUN="${FFTM_DRY_RUN:-0}"
MPI_RANK_AFFINITY_MODE="${FFTM_MPI_RANK_AFFINITY_MODE:-${DEFAULT_MPI_RANK_AFFINITY_MODE}}"
WZ_PLAN_CONCURRENCIES="${FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES:-${DEFAULT_WZ_PLAN_CONCURRENCIES}}"

case "${DRY_RUN}" in
    0|1|false|FALSE|true|TRUE)
        ;;
    *)
        echo "FFTM_DRY_RUN must be 0/1 or false/true." >&2
        exit 2
        ;;
esac

printf 'FFTM target: %s; nodes=%s; GPUs=%s; transports=%s; pipelines=%s; affinity=%s; WZ concurrency=%s; 3D=%s+%s; 4D=%s+%s; times=%s; warmup=%s\n' \
    "${TARGET}" "${NODE_COUNTS}" "${GPU_COUNTS}" \
    "${TRANSPORTS}" "${PENCIL_PIPELINES}" "${MPI_RANK_AFFINITY_MODE}" "${WZ_PLAN_CONCURRENCIES}" \
    "${BENCHMARK_SIZES_3D}" "${FIXED_SIZE_3D}" \
    "${BENCHMARK_SIZES_4D}" "${FIXED_SIZE_4D}" \
    "${BENCHMARK_TIMES}" "${WARMUP}"
printf 'Slurm options: %s\n' "${SRUN_EXTRA_ARGS}"
if [[ "${TARGET}" == "transport-selection" ]]; then
    printf '%s\n' \
        'NCA 4D uses native XW/native spectral layout with host staging; CUDA-aware-only direct XW/WZ pipelining is disabled.'
elif [[ "${TARGET}" == "3d-transport-control" ]]; then
    printf '%s\n' \
        '3D control isolates transport from pipeline: CA parity, CA reference, then NCA reference.'
fi

FFTM_CONTAINER_IMAGE="${CONTAINER_IMAGE}" \
FFTM_DATA_DIR="${DATA_DIR}" \
FFTM_NODE_COUNTS="${NODE_COUNTS}" \
FFTM_GPU_COUNTS="${GPU_COUNTS}" \
FFTM_GPUS_PER_NODE=8 \
FFTM_DEVICE_MEMORY_MIB=81920 \
FFTM_BENCHMARK_SIZES_3D="${BENCHMARK_SIZES_3D}" \
FFTM_BENCHMARK_SIZES_4D="${BENCHMARK_SIZES_4D}" \
FFTM_FIXED_SCALING_SIZES_3D="${FIXED_SIZE_3D}" \
FFTM_FIXED_SCALING_SIZES_4D="${FIXED_SIZE_4D}" \
FFTM_EXTRA_SIZES_3D= \
FFTM_EXTRA_SIZES_3D_BY_GPU= \
FFTM_AUTO_MEMORY_FRACTION="${AUTO_MEMORY_FRACTION}" \
FFTM_AUTO_BYTES_PER_POINT_3D="${AUTO_BYTES_PER_POINT_3D}" \
FFTM_AUTO_BYTES_PER_POINT_4D="${AUTO_BYTES_PER_POINT_4D}" \
FFTM_TRANSPORTS="${TRANSPORTS}" \
FFTM_MODES=p2p-waitany \
FFTM_STRATEGIES_3D="${STRATEGIES_3D}" \
FFTM_STRATEGIES_4D="${STRATEGIES_4D}" \
FFTM_SKIP_FFTS=1 \
FFTM_SKIP_FFTM=0 \
FFTM_INCLUDE_VERSIONED=0 \
FFTM_3D_BACKENDS=native \
FFTM_PENCIL_PIPELINES="${PENCIL_PIPELINES}" \
FFTM_PENCIL_LAYOUTS="${PENCIL_LAYOUTS}" \
FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS="${PENCIL_GRID_ORIENTATIONS}" \
FFTM_P2P_VARIANTS=byte-packed \
FFTM_P2P_SCHEDULERS=main \
FFTM_LARGE_COUNT_P2P_TRANSPORTS=hindexed \
FFTM_USE_P2P_BYTE_TRANSFER=1 \
FFTM_DIRECT_P2P_CUDA_AWARE=1 \
FFTM_USE_PERSISTENT_P2P=0 \
FFTM_USE_READY_P2P_SEND=0 \
FFTM_USE_3D_DEFERRED_SEND_COMPLETION=0 \
FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT=1 \
FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY=1 \
FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC=1 \
FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS=0 \
FFTM_USE_FFT_EXEC_NO_SYNC=0 \
FFTM_4D_SLAB_XW_TRANSPOSES=native \
FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS=off \
FFTM_4D_SLAB_XW_KERNEL_LAYOUTS=buffer-contiguous \
FFTM_4D_SLAB_XW_VECTOR4_KERNELS=off \
FFTM_4D_SLAB_XW_TILED_KERNELS=off \
FFTM_4D_SLAB_XW_LAYOUT_STAGES=off \
FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS=native \
FFTM_4D_NATIVE_XW_DIRECT_LAYOUTS=on \
FFTM_4D_NATIVE_XW_PROTOCOLS=chunked \
FFTM_4D_NATIVE_XW_CHUNK_MIB=512 \
FFTM_4D_NATIVE_XW_CHUNK_WINDOWS=1 \
FFTM_4D_NATIVE_XW_COMPACT_STAGINGS=on \
FFTM_4D_SLAB_NATIVE_WORK_AREA_ALIASES=on \
FFTM_4D_SLAB_NATIVE_WZ_COMM_LAYOUTS=on \
FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES="${WZ_PLAN_CONCURRENCIES}" \
FFTM_4D_SLAB_NATIVE_WZ_READY_PIPELINES=on \
FFTM_PRINT_PENCIL_SCHEDULE=1 \
FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
FFTM_ENABLE_GPU_TELEMETRY=1 \
FFTM_BENCHMARK_TIMES="${BENCHMARK_TIMES}" \
FFTM_WARMUP="${WARMUP}" \
FFTM_RUN_PREFLIGHT="${RUN_PREFLIGHT}" \
FFTM_DRY_RUN="${DRY_RUN}" \
FFTM_MPI_RANK_AFFINITY_MODE="${MPI_RANK_AFFINITY_MODE}" \
FFTM_SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}" \
FFTM_SRUN_TIME="${SRUN_TIME}" \
FFTM_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
FFTM_STOP_ON_FAILURE="${STOP_ON_FAILURE}" \
scripts/run_slurm_pyxis_benchmarks.sh
