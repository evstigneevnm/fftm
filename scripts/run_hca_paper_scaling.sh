#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_hca_paper_scaling.sh TARGET

Independently queueable paper-scaling targets:
  fftm-8,  fftm-16,  fftm-32,  fftm-64,  fftm-96,  fftm-120
  egger-8, egger-16, egger-32, egger-64, egger-96, egger-120

Each target runs:
  FFTM: 3D pencil-pencil and 4D slab-slab, strong and weak scaling.
  Egger: 3D pencil-pencil forward and inverse, strong and weak scaling.

The 8-GPU strong and weak sizes are identical and are run only once.

Required images can be overridden with:
  FFTM_CONTAINER_IMAGE
  EGGER_CONTAINER_IMAGE

Useful overrides:
  FFTM_SCALING_DATA_DIR
  FFTM_SCALING_SRUN_EXTRA_ARGS
  FFTM_SCALING_USE_SINGLE_ALLOCATION=auto|0|1
  FFTM_SCALING_ALLOCATION_TIME
  FFTM_SCALING_SALLOC_EXTRA_ARGS
  FFTM_SCALING_BENCHMARK_TIMES
  FFTM_SCALING_WARMUP
  FFTM_SCALING_DRY_RUN=0|1
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

TARGET="${1}"
case "${TARGET}" in
    fftm-8|fftm-16|fftm-32|fftm-64|fftm-96|fftm-120)
        SUITE="fftm"
        GPUS="${TARGET#fftm-}"
        ;;
    egger-8|egger-16|egger-32|egger-64|egger-96|egger-120)
        SUITE="egger"
        GPUS="${TARGET#egger-}"
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown scaling target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

FFTM_IMAGE="${FFTM_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}"
EGGER_IMAGE="${EGGER_CONTAINER_IMAGE:-/scratch/evstigneevnm/egger_fft/eggerfft_bench_a100.sqsh}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${FFTM_SCALING_DATA_DIR:-/scratch/evstigneevnm/fftm/data_scale_${SUITE:0:1}${GPUS}_${STAMP}}"
SRUN_EXTRA_ARGS="${FFTM_SCALING_SRUN_EXTRA_ARGS:---exclude=cn13 --ntasks-per-node=8 --distribution=block:block --kill-on-bad-exit=1}"
BENCHMARK_TIMES="${FFTM_SCALING_BENCHMARK_TIMES:-30}"
WARMUP="${FFTM_SCALING_WARMUP:-5}"
DRY_RUN="${FFTM_SCALING_DRY_RUN:-0}"
TIMEOUT_SECONDS="${FFTM_SCALING_TIMEOUT_SECONDS:-10800}"
SRUN_TIME="${FFTM_SCALING_SRUN_TIME:-02:00:00}"
USE_SINGLE_ALLOCATION="${FFTM_SCALING_USE_SINGLE_ALLOCATION:-auto}"
ALLOCATION_TIME="${FFTM_SCALING_ALLOCATION_TIME:-08:00:00}"
SALLOC_EXTRA_ARGS="${FFTM_SCALING_SALLOC_EXTRA_ARGS:---exclude=cn13}"

case "${DRY_RUN}" in
    0|1)
        ;;
    *)
        echo "FFTM_SCALING_DRY_RUN must be 0 or 1." >&2
        exit 2
        ;;
esac

case "${USE_SINGLE_ALLOCATION}" in
    auto|0|1)
        ;;
    *)
        echo "FFTM_SCALING_USE_SINGLE_ALLOCATION must be auto, 0, or 1." >&2
        exit 2
        ;;
esac

case "${GPUS}" in
    8)
        NODES=1
        SIZE_3D_WEAK=2048
        SIZE_4D_WEAK=320
        GRIDS_3D="4x2"
        WZ_CONCURRENCIES="4"
        WEAK_RATIO_3D="1.000000000"
        WEAK_RATIO_4D="1.000000000"
        ;;
    16)
        NODES=2
        SIZE_3D_WEAK=2560
        SIZE_4D_WEAK=384
        GRIDS_3D="4x4"
        WZ_CONCURRENCIES="4,8"
        WEAK_RATIO_3D="0.976562500"
        WEAK_RATIO_4D="1.036800000"
        ;;
    32)
        NODES=4
        SIZE_3D_WEAK=3200
        SIZE_4D_WEAK=448
        GRIDS_3D="8x4,4x8"
        WZ_CONCURRENCIES="4,8"
        WEAK_RATIO_3D="0.953674316"
        WEAK_RATIO_4D="0.960400000"
        ;;
    64)
        NODES=8
        SIZE_3D_WEAK=4096
        SIZE_4D_WEAK=540
        GRIDS_3D="8x8"
        WZ_CONCURRENCIES="4,8"
        WEAK_RATIO_3D="1.000000000"
        WEAK_RATIO_4D="1.013643265"
        ;;
    96)
        NODES=12
        SIZE_3D_WEAK=4704
        SIZE_4D_WEAK=600
        GRIDS_3D="12x8,8x12"
        WZ_CONCURRENCIES="4"
        WEAK_RATIO_3D="1.009789467"
        WEAK_RATIO_4D="1.029968262"
        ;;
    120)
        NODES=15
        SIZE_3D_WEAK=5040
        SIZE_4D_WEAK=630
        GRIDS_3D="12x10,15x8"
        WZ_CONCURRENCIES="4"
        WEAK_RATIO_3D="0.993597507"
        WEAK_RATIO_4D="1.001546288"
        ;;
esac

allocate_target=0
if [[ "${USE_SINGLE_ALLOCATION}" == "1" ||
      ( "${USE_SINGLE_ALLOCATION}" == "auto" && "${GPUS}" -ge 96 ) ]]; then
    allocate_target=1
fi
if [[ "${DRY_RUN}" == "0" && -z "${SLURM_JOB_ID:-}" && "${allocate_target}" == "1" ]]; then
    read -r -a salloc_extra_args <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting one allocation for target %s: nodes=%s GPUs=%s time=%s\n' \
        "${TARGET}" "${NODES}" "${GPUS}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_SCALING_DATA_DIR="${DATA_DIR}" \
        FFTM_SCALING_USE_SINGLE_ALLOCATION=0 \
        salloc \
        --job-name="scale-${TARGET}" \
        --nodes="${NODES}" \
        --ntasks="${GPUS}" \
        --ntasks-per-node=8 \
        --gpus-per-node=8 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra_args[@]}" \
        bash "${SCRIPT_DIR}/run_hca_paper_scaling.sh" "${TARGET}"
fi

SIZE_3D_STRONG=2048
SIZE_4D_STRONG=320
if [[ "${GPUS}" == "8" ]]; then
    SIZES_3D="${SIZE_3D_STRONG}"
    SIZES_4D="${SIZE_4D_STRONG}"
else
    SIZES_3D="${SIZE_3D_STRONG},${SIZE_3D_WEAK}"
    SIZES_4D="${SIZE_4D_STRONG},${SIZE_4D_WEAK}"
fi

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"

MANIFEST="${DATA_DIR}/scaling_manifest.csv"
{
    printf '%s\n' \
        "suite,dimension,gpus,nodes,size,scaling_role,variant,points_per_gpu_relative_to_8g"
    if [[ "${SUITE}" == "fftm" ]]; then
        IFS=',' read -r -a grids <<< "${GRIDS_3D}"
        for grid in "${grids[@]}"; do
            printf 'fftm,3,%s,%s,%s,strong,grid%s,%.9f\n' \
                "${GPUS}" "${NODES}" "${SIZE_3D_STRONG}" "${grid}" \
                "$(( 8 * 1000000000 / GPUS ))e-9"
            if [[ "${SIZE_3D_WEAK}" != "${SIZE_3D_STRONG}" ]]; then
                printf 'fftm,3,%s,%s,%s,weak,grid%s,%s\n' \
                    "${GPUS}" "${NODES}" "${SIZE_3D_WEAK}" "${grid}" "${WEAK_RATIO_3D}"
            fi
        done
        IFS=',' read -r -a concurrencies <<< "${WZ_CONCURRENCIES}"
        for concurrency in "${concurrencies[@]}"; do
            printf 'fftm,4,%s,%s,%s,strong,wz_c%s,%.9f\n' \
                "${GPUS}" "${NODES}" "${SIZE_4D_STRONG}" "${concurrency}" \
                "$(( 8 * 1000000000 / GPUS ))e-9"
            if [[ "${SIZE_4D_WEAK}" != "${SIZE_4D_STRONG}" ]]; then
                printf 'fftm,4,%s,%s,%s,weak,wz_c%s,%s\n' \
                    "${GPUS}" "${NODES}" "${SIZE_4D_WEAK}" "${concurrency}" "${WEAK_RATIO_4D}"
            fi
        done
    else
        IFS=',' read -r -a grids <<< "${GRIDS_3D}"
        for grid in "${grids[@]}"; do
            for direction in forward inverse; do
                printf 'egger,3,%s,%s,%s,strong,grid%s_%s,%.9f\n' \
                    "${GPUS}" "${NODES}" "${SIZE_3D_STRONG}" "${grid}" "${direction}" \
                    "$(( 8 * 1000000000 / GPUS ))e-9"
                if [[ "${SIZE_3D_WEAK}" != "${SIZE_3D_STRONG}" ]]; then
                    printf 'egger,3,%s,%s,%s,weak,grid%s_%s,%s\n' \
                        "${GPUS}" "${NODES}" "${SIZE_3D_WEAK}" "${grid}" "${direction}" \
                        "${WEAK_RATIO_3D}"
                fi
            done
        done
    fi
} > "${MANIFEST}"

{
    printf 'target=%s\n' "${TARGET}"
    printf 'suite=%s\n' "${SUITE}"
    printf 'gpus=%s\n' "${GPUS}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'sizes_3d=%s\n' "${SIZES_3D}"
    printf 'sizes_4d=%s\n' "${SIZES_4D}"
    printf 'grids_3d=%s\n' "${GRIDS_3D}"
    printf 'wz_concurrencies=%s\n' "${WZ_CONCURRENCIES}"
    printf 'benchmark_times=%s\n' "${BENCHMARK_TIMES}"
    printf 'warmup=%s\n' "${WARMUP}"
    printf 'subprocess_timeout_seconds=%s\n' "${TIMEOUT_SECONDS}"
    printf 'srun_time=%s\n' "${SRUN_TIME}"
    printf 'allocation_time=%s\n' "${ALLOCATION_TIME}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'mpi_rank_affinity_mode=hca\n'
    printf 'srun_extra_args=%s\n' "${SRUN_EXTRA_ARGS}"
    printf 'fftm_container_image=%s\n' "${FFTM_IMAGE}"
    printf 'egger_container_image=%s\n' "${EGGER_IMAGE}"
} > "${DATA_DIR}/scaling_config.env"

if [[ "${SUITE}" == "fftm" ]]; then
    grid_count="$(awk -F, '{ print NF }' <<< "${GRIDS_3D}")"
    concurrency_count="$(awk -F, '{ print NF }' <<< "${WZ_CONCURRENCIES}")"
    size_count=2
    [[ "${GPUS}" == "8" ]] && size_count=1
    expected_runs=$(( size_count * ( grid_count + concurrency_count ) ))
else
    grid_count="$(awk -F, '{ print NF }' <<< "${GRIDS_3D}")"
    size_count=2
    [[ "${GPUS}" == "8" ]] && size_count=1
    expected_runs=$(( size_count * grid_count * 2 ))
fi

printf 'Paper scaling target: %s\n' "${TARGET}"
printf '  nodes=%s GPUs=%s affinity=hca\n' "${NODES}" "${GPUS}"
printf '  3D sizes=%s grids=%s\n' "${SIZES_3D}" "${GRIDS_3D}"
if [[ "${SUITE}" == "fftm" ]]; then
    printf '  4D sizes=%s WZ concurrency=%s\n' "${SIZES_4D}" "${WZ_CONCURRENCIES}"
fi
printf '  expected cases=%s; times=%s; warmup=%s\n' \
    "${expected_runs}" "${BENCHMARK_TIMES}" "${WARMUP}"
printf '  output=%s\n' "${DATA_DIR}"
printf '  Slurm options: %s\n' "${SRUN_EXTRA_ARGS}"

if [[ "${SUITE}" == "fftm" ]]; then
    FFTM_CONTAINER_IMAGE="${FFTM_IMAGE}" \
    FFTM_DATA_DIR="${DATA_DIR}" \
    FFTM_NODE_COUNTS="${NODES}" \
    FFTM_GPU_COUNTS="${GPUS}" \
    FFTM_GPUS_PER_NODE=8 \
    FFTM_DEVICE_MEMORY_MIB=81920 \
    FFTM_BENCHMARK_SIZES_3D=none \
    FFTM_BENCHMARK_SIZES_4D=none \
    FFTM_FIXED_SCALING_SIZES_3D="${SIZES_3D}" \
    FFTM_FIXED_SCALING_SIZES_4D="${SIZES_4D}" \
    FFTM_EXTRA_SIZES_3D= \
    FFTM_EXTRA_SIZES_3D_BY_GPU= \
    FFTM_TRANSPORTS=cuda_aware \
    FFTM_MODES=p2p-waitany \
    FFTM_STRATEGIES_3D=pencil-pencil \
    FFTM_STRATEGIES_4D=slab-slab \
    FFTM_SKIP_FFTS=1 \
    FFTM_SKIP_FFTM=0 \
    FFTM_INCLUDE_VERSIONED=0 \
    FFTM_3D_BACKENDS=native \
    FFTM_PENCIL_PIPELINES=reference-parity \
    FFTM_PENCIL_LAYOUTS=opt0 \
    FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS="${GRIDS_3D}" \
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
    FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES="${WZ_CONCURRENCIES}" \
    FFTM_4D_SLAB_NATIVE_WZ_READY_PIPELINES=on \
    FFTM_PRINT_PENCIL_SCHEDULE=0 \
    FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
    FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
    FFTM_ENABLE_GPU_TELEMETRY=1 \
    FFTM_BENCHMARK_TIMES="${BENCHMARK_TIMES}" \
    FFTM_WARMUP="${WARMUP}" \
    FFTM_RUN_PREFLIGHT=0 \
    FFTM_MPI_RANK_AFFINITY_MODE=hca \
    FFTM_SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}" \
    FFTM_SRUN_TIME="${SRUN_TIME}" \
    FFTM_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
    FFTM_STOP_ON_FAILURE=0 \
    FFTM_DRY_RUN="${DRY_RUN}" \
    scripts/run_slurm_pyxis_benchmarks.sh
else
    EGGER_CONTAINER_IMAGE="${EGGER_IMAGE}" \
    EGGER_DATA_DIR="${DATA_DIR}" \
    EGGER_GPU_COUNTS="${GPUS}" \
    EGGER_MAX_GPUS="${GPUS}" \
    EGGER_GPUS_PER_NODE=8 \
    EGGER_DEVICE_MEMORY_MIB=81920 \
    EGGER_SIZES="${SIZES_3D}" \
    EGGER_EXTRA_SIZES= \
    EGGER_EXTRA_SIZES_BY_GPU= \
    EGGER_PENCIL_GRIDS_BY_GPU="${GPUS}:${GRIDS_3D}" \
    EGGER_FAMILIES=pencil \
    EGGER_OPTS=0 \
    EGGER_METHODS=Peer2Peer:Sync \
    EGGER_TRANSPORTS=cuda_aware \
    EGGER_TESTCASES=0,2 \
    EGGER_ITERATIONS="${BENCHMARK_TIMES}" \
    EGGER_WARMUP="${WARMUP}" \
    EGGER_MPI_RANK_AFFINITY_MODE=hca \
    EGGER_SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}" \
    EGGER_SRUN_TIME="${SRUN_TIME}" \
    EGGER_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
    EGGER_FAIL_ON_ERROR=0 \
    EGGER_DRY_RUN="${DRY_RUN}" \
    fft_Egger/scripts/run_slurm_pyxis_egger_benchmarks.sh
fi
