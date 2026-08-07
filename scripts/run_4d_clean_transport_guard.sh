#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the current FFTM .sqsh image}"

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${FFTM_4D_CLEAN_GUARD_DATA_DIR:-${PWD}/data_4d_clean_transport_guard_${STAMP}}"
NODES=4
GPUS_PER_NODE=8
TASKS=$((NODES * GPUS_PER_NODE))
ALLOCATION_TIME="${FFTM_4D_CLEAN_GUARD_ALLOCATION_TIME:-02:00:00}"
SALLOC_EXTRA_TEXT="${FFTM_4D_CLEAN_GUARD_SALLOC_EXTRA_ARGS:---exclude=cn13,cn24}"
USE_SINGLE_ALLOCATION="${FFTM_4D_CLEAN_GUARD_USE_SINGLE_ALLOCATION:-1}"
DRY_RUN="${FFTM_DRY_RUN:-0}"

if [[ ! -f "${FFTM_CONTAINER_IMAGE}" ]]; then
    echo "Missing FFTM container image: ${FFTM_CONTAINER_IMAGE}" >&2
    exit 2
fi

mkdir -p "${ROOT}"
ROOT="$(cd -- "${ROOT}" && pwd -P)"

if [[ "${DRY_RUN}" == "0" && -z "${SLURM_JOB_ID:-}" && "${USE_SINGLE_ALLOCATION}" == "1" ]]; then
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_TEXT}"
    printf 'Requesting clean 4D transport-guard allocation: nodes=%d GPUs/node=%d time=%s\n' \
        "${NODES}" "${GPUS_PER_NODE}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_4D_CLEAN_GUARD_DATA_DIR="${ROOT}" \
        FFTM_4D_CLEAN_GUARD_USE_SINGLE_ALLOCATION=0 \
        salloc \
        --job-name=fftm-4d-clean \
        --nodes="${NODES}" \
        --ntasks="${TASKS}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${BASH_SOURCE[0]}"
fi

if [[ -n "${SLURM_NNODES:-}" && "${SLURM_NNODES}" -lt "${NODES}" ]]; then
    echo "The clean 4D transport guard requires a four-node allocation." >&2
    exit 2
fi

AFFINITY_WRAPPER="${ROOT}/fftm_mpi_rank_affinity.sh"
install -m 0755 "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" "${AFFINITY_WRAPPER}"

{
    printf 'container_image=%s\n' "${FFTM_CONTAINER_IMAGE}"
    printf 'nodes=%d\n' "${NODES}"
    printf 'gpus_per_node=%d\n' "${GPUS_PER_NODE}"
    printf 'sizes_4d=320\n'
    printf 'gpu_counts=16,32\n'
    printf 'grids=2x8x1,4x8x1\n'
    printf 'pipeline=production-auto-node-aligned-wz\n'
    printf 'hca_validation=active-sysfs+ucx\n'
    printf 'salloc_extra_args=%s\n' "${SALLOC_EXTRA_TEXT}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-none}"
} > "${ROOT}/guard_config.env"

sha256sum \
    "${FFTM_CONTAINER_IMAGE}" \
    "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" \
    > "${ROOT}/guard_provenance.sha256"

preflight_payload='
set -eu
if [[ "${SLURM_LOCALID:-}" == "0" ]]; then
    echo "===== NODE $(hostname) ====="
    nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader
    nvidia-smi topo -m
    nvidia-smi topo -p2p r || true
fi
'

preflight_cmd=(
    srun
    --nodes="${NODES}"
    --ntasks="${TASKS}"
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
    /bin/bash -lc "${preflight_payload}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY RUN preflight:'
    printf ' %q' "${preflight_cmd[@]}"
    printf '\n'
else
    echo "Running four-node HCA and CUDA P2P/NVLink topology preflight..."
    "${preflight_cmd[@]}" > "${ROOT}/transport_preflight.log" 2>&1

    validation_count="$(grep -c 'validation=active-sysfs+ucx' "${ROOT}/transport_preflight.log" || true)"
    if [[ "${validation_count}" -ne "${TASKS}" ]]; then
        echo "Expected ${TASKS} validated affinity records, found ${validation_count}." >&2
        exit 1
    fi
    if grep -Eq 'UCX[[:space:]]+(WARN|ERROR)|not visible in ucx_info|not active' \
        "${ROOT}/transport_preflight.log"; then
        echo "The transport preflight reported a degraded UCX/HCA configuration." >&2
        exit 1
    fi
fi

echo "Running clean 16/32-GPU 4D production-policy guard..."
env \
    FFTM_CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE}" \
    FFTM_DATA_DIR="${ROOT}" \
    FFTM_NODE_COUNTS=2,4 \
    FFTM_GPU_COUNTS=16,32 \
    FFTM_GPUS_PER_NODE=8 \
    FFTM_DEVICE_MEMORY_MIB=81920 \
    FFTM_BENCHMARK_SIZES_3D=none \
    FFTM_BENCHMARK_SIZES_4D=none \
    FFTM_FIXED_SCALING_SIZES_3D= \
    FFTM_FIXED_SCALING_SIZES_4D=320 \
    FFTM_EXTRA_SIZES_3D= \
    FFTM_EXTRA_SIZES_3D_BY_GPU= \
    FFTM_MODES=p2p-waitany \
    FFTM_TRANSPORTS=cuda_aware \
    FFTM_STRATEGIES_3D= \
    FFTM_STRATEGIES_4D=pencil-pencil \
    FFTM_SKIP_FFTS=1 \
    FFTM_SKIP_FFTM=0 \
    FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS=native \
    FFTM_4D_PENCIL_SAME_ZW_NATIVE_LAYOUTS=native \
    FFTM_4D_PENCIL_GRID_ORIENTATIONS=explicit:2x8x1,4x8x1 \
    FFTM_4D_PENCIL_NODE_ALIGNED_WZ_PIPELINES=production \
    FFTM_MPI_RANK_AFFINITY_MODE=hca \
    FFTM_SRUN_EXTRA_ARGS='--distribution=block:block' \
    FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
    FFTM_ENABLE_GPU_TELEMETRY=0 \
    FFTM_BENCHMARK_TIMES=30 \
    FFTM_WARMUP=5 \
    FFTM_SRUN_TIME=01:00:00 \
    FFTM_TIMEOUT_SECONDS=3900 \
    FFTM_STOP_ON_FAILURE=1 \
    FFTM_RUN_PREFLIGHT=0 \
    FFTM_DRY_RUN="${DRY_RUN}" \
    "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"

if [[ "${DRY_RUN}" == "0" ]]; then
    python3 "${SCRIPT_DIR}/analyze_fftm_4d_performance.py" --data-dir "${ROOT}"
fi

echo "Clean 4D transport guard complete: ${ROOT}"
