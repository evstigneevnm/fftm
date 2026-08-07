#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
SELF="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

usage()
{
    cat <<'EOF'
Run the decisive 4D slab-slab communication-granularity A/B in one Slurm allocation.

The wall phase covers 8, 16, 24, and 32 GPUs. The timer phase covers 16, 24,
and 32 GPUs. Each GPU count compares:
  1. peer-coarse direct P2P;
  2. plane-ready WZ P2P (the current production path);
  3. buffered alltoallv.

Required:
  FFTM_CONTAINER_IMAGE=/path/to/fftm_bench_a100.sqsh

Useful overrides:
  FFTM_4D_GRANULARITY_DATA_DIR
  FFTM_4D_GRANULARITY_ALLOCATION_TIME       (default: 06:00:00)
  FFTM_4D_GRANULARITY_SALLOC_EXTRA_ARGS     (default: --exclude=cn13,cn24)
  FFTM_4D_GRANULARITY_WALL_GPU_COUNTS       (default: 8,16,24,32)
  FFTM_4D_GRANULARITY_TIMER_GPU_COUNTS      (default: 16,24,32)
  FFTM_4D_GRANULARITY_WALL_TIMES            (default: 30)
  FFTM_4D_GRANULARITY_WALL_WARMUP           (default: 5)
  FFTM_4D_GRANULARITY_TIMER_TIMES           (default: 5)
  FFTM_4D_GRANULARITY_TIMER_WARMUP          (default: 2)
  FFTM_DRY_RUN=1
EOF
}

inside_allocation=0
case "${1:-}" in
    --inside-allocation)
        inside_allocation=1
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    "")
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the current FFTM .sqsh image}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${FFTM_4D_GRANULARITY_DATA_DIR:-${PWD}/data_4d_slab_granularity_ab_${STAMP}}"
ALLOCATION_TIME="${FFTM_4D_GRANULARITY_ALLOCATION_TIME:-06:00:00}"
SALLOC_EXTRA_TEXT="${FFTM_4D_GRANULARITY_SALLOC_EXTRA_ARGS:---exclude=cn13,cn24}"
USE_SINGLE_ALLOCATION="${FFTM_4D_GRANULARITY_USE_SINGLE_ALLOCATION:-1}"
WALL_GPU_COUNTS="${FFTM_4D_GRANULARITY_WALL_GPU_COUNTS:-8,16,24,32}"
TIMER_GPU_COUNTS="${FFTM_4D_GRANULARITY_TIMER_GPU_COUNTS:-16,24,32}"
WALL_TIMES="${FFTM_4D_GRANULARITY_WALL_TIMES:-30}"
WALL_WARMUP="${FFTM_4D_GRANULARITY_WALL_WARMUP:-5}"
TIMER_TIMES="${FFTM_4D_GRANULARITY_TIMER_TIMES:-5}"
TIMER_WARMUP="${FFTM_4D_GRANULARITY_TIMER_WARMUP:-2}"
SIZE_4D="${FFTM_4D_GRANULARITY_SIZE:-320}"
GPUS_PER_NODE=8
MAX_NODES=4
MAX_TASKS=$((MAX_NODES * GPUS_PER_NODE))
DRY_RUN="${FFTM_DRY_RUN:-0}"

is_true()
{
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

csv_item_count()
{
    local value=$1
    local item
    local count=0
    local -a items=()
    IFS=',' read -r -a items <<< "${value}"
    for item in "${items[@]}"; do
        [[ -n "${item//[[:space:]]/}" ]] && count=$((count + 1))
    done
    printf '%d\n' "${count}"
}

for gpu_list in "${WALL_GPU_COUNTS}" "${TIMER_GPU_COUNTS}"; do
    IFS=',' read -r -a gpu_values <<< "${gpu_list}"
    for gpu_count in "${gpu_values[@]}"; do
        [[ "${gpu_count}" =~ ^[0-9]+$ ]] || {
            echo "Invalid GPU count: ${gpu_count}" >&2
            exit 2
        }
        if (( gpu_count < GPUS_PER_NODE || gpu_count > MAX_TASKS || gpu_count % GPUS_PER_NODE != 0 )); then
            echo "GPU counts must be full-node values from 8 through 32: ${gpu_count}" >&2
            exit 2
        fi
    done
done

WALL_EXPECTED_CASES=$((3 * $(csv_item_count "${WALL_GPU_COUNTS}")))
TIMER_EXPECTED_CASES=$((3 * $(csv_item_count "${TIMER_GPU_COUNTS}")))

if ! is_true "${DRY_RUN}" && [[ ! -f "${FFTM_CONTAINER_IMAGE}" ]]; then
    echo "Missing FFTM container image: ${FFTM_CONTAINER_IMAGE}" >&2
    exit 2
fi

if [[ "${inside_allocation}" == 0 && -e "${ROOT}" &&
      -n "$(find "${ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory is not empty: ${ROOT}" >&2
    exit 2
fi
mkdir -p "${ROOT}"
ROOT="$(cd -- "${ROOT}" && pwd -P)"

if ! is_true "${DRY_RUN}" && [[ -z "${SLURM_JOB_ID:-}" && "${USE_SINGLE_ALLOCATION}" == 1 ]]; then
    command -v salloc >/dev/null || {
        echo "salloc is required outside an existing allocation." >&2
        exit 2
    }
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_TEXT}"
    printf 'Requesting one 4D slab-granularity allocation: nodes=%d GPUs/node=%d time=%s\n' \
        "${MAX_NODES}" "${GPUS_PER_NODE}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_4D_GRANULARITY_DATA_DIR="${ROOT}" \
        FFTM_4D_GRANULARITY_USE_SINGLE_ALLOCATION=0 \
        salloc \
        --job-name=fftm-4d-granularity \
        --nodes="${MAX_NODES}" \
        --ntasks="${MAX_TASKS}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task=4 \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SELF}" --inside-allocation
fi

if ! is_true "${DRY_RUN}" && [[ -n "${SLURM_JOB_NUM_NODES:-}" &&
      "${SLURM_JOB_NUM_NODES}" -lt "${MAX_NODES}" ]]; then
    echo "This matrix requires a four-node allocation; got ${SLURM_JOB_NUM_NODES}." >&2
    exit 2
fi

mkdir -p "${ROOT}/wall" "${ROOT}/timers"

{
    printf 'container_image=%s\n' "${FFTM_CONTAINER_IMAGE}"
    printf 'size_4d=%s\n' "${SIZE_4D}"
    printf 'wall_gpu_counts=%s\n' "${WALL_GPU_COUNTS}"
    printf 'timer_gpu_counts=%s\n' "${TIMER_GPU_COUNTS}"
    printf 'wall_expected_cases=%d\n' "${WALL_EXPECTED_CASES}"
    printf 'timer_expected_cases=%d\n' "${TIMER_EXPECTED_CASES}"
    printf 'variants=peer-coarse-p2p,plane-ready-p2p,buffered-alltoallv\n'
    printf 'allocation_time=%s\n' "${ALLOCATION_TIME}"
    printf 'salloc_extra_args=%s\n' "${SALLOC_EXTRA_TEXT}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-dry-run}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${ROOT}/config.env"

if ! is_true "${DRY_RUN}"; then
    scontrol show job "${SLURM_JOB_ID}" > "${ROOT}/slurm_job.txt"
    scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${ROOT}/hosts.txt"
    if grep -Eq '^(cn13|cn24)$' "${ROOT}/hosts.txt"; then
        echo "The allocation contains an excluded node." >&2
        exit 2
    fi
    sha256sum \
        "${FFTM_CONTAINER_IMAGE}" \
        "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" \
        "${SELF}" \
        > "${ROOT}/provenance.sha256"
    git -C "${REPO_ROOT}" rev-parse HEAD > "${ROOT}/git_commit.txt" 2>/dev/null || true
fi

AFFINITY_WRAPPER="${ROOT}/fftm_mpi_rank_affinity.sh"
install -m 0755 "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" "${AFFINITY_WRAPPER}"

preflight_cmd=(
    srun
    --nodes="${MAX_NODES}"
    --ntasks="${MAX_TASKS}"
    --ntasks-per-node="${GPUS_PER_NODE}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --distribution=block:block
    --kill-on-bad-exit=1
    --time=00:10:00
    --container-image="${FFTM_CONTAINER_IMAGE}"
    --container-mounts="${ROOT}:/data"
    --container-workdir=/opt/fftm/bin
    --container-entrypoint
    /data/fftm_mpi_rank_affinity.sh
    --mode hca
    --
    /bin/bash -lc
    'printf "FFTM_4D_PREFLIGHT host=%s rank=%s local_rank=%s visible=%s\n" "$(hostname)" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" "${CUDA_VISIBLE_DEVICES:-}"'
)

if is_true "${DRY_RUN}"; then
    printf 'DRY RUN preflight:'
    printf ' %q' "${preflight_cmd[@]}"
    printf '\n'
else
    echo "Running 32-rank HCA/GPU-affinity preflight..."
    "${preflight_cmd[@]}" > "${ROOT}/transport_preflight.log" 2>&1
    validation_count="$(grep -c 'validation=active-sysfs+ucx' "${ROOT}/transport_preflight.log" || true)"
    visibility_count="$(grep -c '^FFTM_4D_PREFLIGHT ' "${ROOT}/transport_preflight.log" || true)"
    if [[ "${validation_count}" -ne "${MAX_TASKS}" || "${visibility_count}" -ne "${MAX_TASKS}" ]]; then
        echo "HCA/GPU preflight expected ${MAX_TASKS} ranks; got HCA=${validation_count}, GPU=${visibility_count}." >&2
        exit 1
    fi
    if grep -Eq 'UCX[[:space:]]+(WARN|ERROR)|not visible in ucx_info|not active' \
        "${ROOT}/transport_preflight.log"; then
        echo "The transport preflight reported a degraded UCX/HCA configuration." >&2
        exit 1
    fi
fi

printf 'phase,expected_cases,returncode\n' > "${ROOT}/phase_status.csv"
failed_phases=0

run_phase()
{
    local phase=$1
    local gpu_counts=$2
    local times=$3
    local warmup=$4
    local timers=$5
    local expected_cases=$6
    local phase_dir="${ROOT}/${phase}"
    local rc=0

    echo "Running ${phase} phase: GPUs=${gpu_counts}, expected cases=${expected_cases}, timers=${timers}"
    if env \
        FFTM_CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE}" \
        FFTM_DATA_DIR="${phase_dir}" \
        FFTM_NODE_COUNTS=1,2,3,4 \
        FFTM_GPU_COUNTS="${gpu_counts}" \
        FFTM_GPUS_PER_NODE="${GPUS_PER_NODE}" \
        FFTM_DEVICE_MEMORY_MIB=81920 \
        FFTM_BENCHMARK_SIZES_3D=none \
        FFTM_BENCHMARK_SIZES_4D=none \
        FFTM_FIXED_SCALING_SIZES_3D= \
        FFTM_FIXED_SCALING_SIZES_4D="${SIZE_4D}" \
        FFTM_EXTRA_SIZES_3D= \
        FFTM_EXTRA_SIZES_3D_BY_GPU= \
        FFTM_MODES=p2p-waitany,alltoallv \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_STRATEGIES_3D= \
        FFTM_STRATEGIES_4D=slab-slab \
        FFTM_SKIP_FFTS=1 \
        FFTM_SKIP_FFTM=0 \
        FFTM_INCLUDE_VERSIONED=0 \
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
        FFTM_4D_SLAB_NATIVE_WZ_COMM_LAYOUTS=off,on \
        FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES=4 \
        FFTM_4D_SLAB_NATIVE_WZ_READY_PIPELINES=on \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_SRUN_EXTRA_ARGS='--distribution=block:block' \
        FFTM_PRINT_PENCIL_SCHEDULE=0 \
        FFTM_ENABLE_NATIVE_STAGE_TIMERS="${timers}" \
        FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
        FFTM_ENABLE_GPU_TELEMETRY=1 \
        FFTM_VALIDATION_TIMES=1 \
        FFTM_BENCHMARK_TIMES="${times}" \
        FFTM_WARMUP="${warmup}" \
        FFTM_SRUN_TIME=01:00:00 \
        FFTM_TIMEOUT_SECONDS=3900 \
        FFTM_STOP_ON_FAILURE=0 \
        FFTM_RUN_PREFLIGHT=0 \
        FFTM_DRY_RUN="${DRY_RUN}" \
        "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"; then
        rc=0
    else
        rc=$?
        failed_phases=$((failed_phases + 1))
    fi

    printf '%s,%s,%s\n' "${phase}" "${expected_cases}" "${rc}" >> "${ROOT}/phase_status.csv"

    if [[ -f "${phase_dir}/summary.json" ]]; then
        python3 - "${phase_dir}/summary.json" "${expected_cases}" <<'PY'
import json
import sys

path, expected_text = sys.argv[1:]
expected = int(expected_text)
with open(path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)
planned = int(summary.get("planned_runs", -1))
if planned != expected:
    raise SystemExit(f"{path}: expected {expected} planned runs, found {planned}")
print(f"Verified {planned} planned runs in {path}")
PY
    elif ! is_true "${DRY_RUN}"; then
        echo "Missing phase summary: ${phase_dir}/summary.json" >&2
        failed_phases=$((failed_phases + 1))
    fi
}

run_phase wall "${WALL_GPU_COUNTS}" "${WALL_TIMES}" "${WALL_WARMUP}" 0 "${WALL_EXPECTED_CASES}"
run_phase timers "${TIMER_GPU_COUNTS}" "${TIMER_TIMES}" "${TIMER_WARMUP}" 1 "${TIMER_EXPECTED_CASES}"

if ! is_true "${DRY_RUN}"; then
    python3 "${SCRIPT_DIR}/analyze_fftm_4d_performance.py" \
        --data-dir "${ROOT}" \
        --output-dir "${ROOT}/analysis"
fi

echo "4D slab granularity A/B complete: ${ROOT}"
if [[ "${failed_phases}" -ne 0 ]]; then
    echo "One or more phases reported failures; all planned phases were attempted." >&2
    exit 1
fi
