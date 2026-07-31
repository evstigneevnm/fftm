#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${FFTM_NODE_HYBRID_4D_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_node_hybrid_a100.sqsh}"
DATA_DIR="${FFTM_NODE_HYBRID_4D_DATA_DIR:-/scratch/evstigneevnm/fftm/data_cufft_node_hybrid_4d_$(date +%Y%m%d_%H%M%S)}"
TARGET="${FFTM_NODE_HYBRID_4D_TARGET:-selection}"
TIMES="${FFTM_NODE_HYBRID_4D_TIMES:-20}"
WARMUP="${FFTM_NODE_HYBRID_4D_WARMUP:-5}"
EPSILON="${FFTM_NODE_HYBRID_4D_EPSILON:-1.0e-11}"
SRUN_TIME="${FFTM_NODE_HYBRID_4D_SRUN_TIME:-02:00:00}"
ALLOCATION_TIME="${FFTM_NODE_HYBRID_4D_ALLOCATION_TIME:-06:00:00}"
SALLOC_EXTRA_ARGS="${FFTM_NODE_HYBRID_4D_SALLOC_EXTRA_ARGS:---exclude=cn13}"
STOP_ON_FAILURE="${FFTM_NODE_HYBRID_4D_STOP_ON_FAILURE:-0}"
SKIP_PREFLIGHT="${FFTM_NODE_HYBRID_4D_SKIP_PREFLIGHT:-0}"
DRY_RUN="${FFTM_NODE_HYBRID_4D_DRY_RUN:-0}"
BASELINE_MS="${FFTM_NODE_HYBRID_4D_BASELINE_MS:-217.723}"

case "${TARGET}" in
    smoke)
        ALLOCATION_GPUS=2
        ;;
    selection|production)
        ALLOCATION_GPUS=8
        ;;
    *)
        echo "Unknown FFTM_NODE_HYBRID_4D_TARGET=${TARGET}; use smoke, selection, or production." >&2
        exit 2
        ;;
esac

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" && "${DRY_RUN}" != "1" ]]; then
    [[ -f "${IMAGE}" ]] || {
        echo "Container image does not exist: ${IMAGE}" >&2
        exit 1
    }
    export FFTM_NODE_HYBRID_4D_CONTAINER_IMAGE="${IMAGE}"
    export FFTM_NODE_HYBRID_4D_DATA_DIR="${DATA_DIR}"
    export FFTM_NODE_HYBRID_4D_TARGET="${TARGET}"
    export FFTM_NODE_HYBRID_4D_TIMES="${TIMES}"
    export FFTM_NODE_HYBRID_4D_WARMUP="${WARMUP}"
    export FFTM_NODE_HYBRID_4D_EPSILON="${EPSILON}"
    export FFTM_NODE_HYBRID_4D_SRUN_TIME="${SRUN_TIME}"
    export FFTM_NODE_HYBRID_4D_ALLOCATION_TIME="${ALLOCATION_TIME}"
    export FFTM_NODE_HYBRID_4D_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}"
    export FFTM_NODE_HYBRID_4D_STOP_ON_FAILURE="${STOP_ON_FAILURE}"
    export FFTM_NODE_HYBRID_4D_SKIP_PREFLIGHT="${SKIP_PREFLIGHT}"
    export FFTM_NODE_HYBRID_4D_BASELINE_MS="${BASELINE_MS}"
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting standalone 4D node allocation: GPUs=%s target=%s time=%s\n' \
        "${ALLOCATION_GPUS}" "${TARGET}" "${ALLOCATION_TIME}"
    exec salloc \
        -N 1 \
        --gres="gpu:${ALLOCATION_GPUS}" \
        --ntasks-per-node=1 \
        --cpus-per-task=32 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_4d.sh" --inside-allocation
fi

if [[ "${1:-}" == "--inside-allocation" ]]; then
    shift
fi

mkdir -p "${DATA_DIR}/raw"
STATUS_FILE="${DATA_DIR}/status.csv"
printf 'case_id,label,gpus,size,rc,log\n' > "${STATUS_FILE}"

cat > "${DATA_DIR}/config.env" <<EOF
container_image=${IMAGE}
target=${TARGET}
times=${TIMES}
warmup=${WARMUP}
epsilon=${EPSILON}
srun_time=${SRUN_TIME}
allocation_gpus=${ALLOCATION_GPUS}
allocation_time=${ALLOCATION_TIME}
salloc_extra_args=${SALLOC_EXTRA_ARGS}
slurm_job_id=${SLURM_JOB_ID:-}
slurm_job_nodelist=${SLURM_JOB_NODELIST:-}
git_commit=$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || printf unknown)
image_checksum=$(sha256sum "${IMAGE}" 2>/dev/null | awk '{print $1}' || printf unknown)
dry_run=${DRY_RUN}
baseline_ms=${BASELINE_MS}
EOF

BINARY="/opt/fftm_node_hybrid/bin/fftm_cufft_node_hybrid_4d.bin"
COMMON_SRUN=(
    --exclusive
    --kill-on-bad-exit=1
    --chdir=/tmp
    --cpu-bind=none
    --cpus-per-task=32
    --time="${SRUN_TIME}"
    --container-image "${IMAGE}"
    "--container-mounts=${DATA_DIR}:/data"
    --container-workdir /opt/fftm_node_hybrid
    --container-entrypoint "${BINARY}"
)

if [[ "${SKIP_PREFLIGHT}" != "1" && "${DRY_RUN}" != "1" ]]; then
    echo "Running standalone cuFFT Xt 4D preflight..."
    PREFLIGHT_LOG="${DATA_DIR}/raw/000_preflight.log"
    set +e
    srun \
        -N 1 -n 1 -G 2 \
        --gpus-per-node=2 \
        --gpus-per-task=2 \
        "${COMMON_SRUN[@]}" \
        --size 32 \
        --gpus-per-process 2 \
        --warmup 0 \
        --iterations 1 \
        --epsilon "${EPSILON}" \
        --label preflight \
        --output /data/preflight.csv \
        > "${PREFLIGHT_LOG}" 2>&1
    PREFLIGHT_RC=$?
    set -e
    cat "${PREFLIGHT_LOG}"
    printf '000,preflight,2,32,%s,raw/000_preflight.log\n' \
        "${PREFLIGHT_RC}" >> "${STATUS_FILE}"
    if (( PREFLIGHT_RC != 0 )); then
        echo "FAILED 4D preflight; log=${PREFLIGHT_LOG}" >&2
        exit "${PREFLIGHT_RC}"
    fi
fi

case_id=0

run_case()
{
    local label="$1"
    local gpus="$2"
    local size="$3"
    local warmup="${4:-${WARMUP}}"
    local times="${5:-${TIMES}}"

    case_id=$((case_id + 1))
    local id
    printf -v id '%03d' "${case_id}"
    local stem="${id}_${label}"
    local log="${DATA_DIR}/raw/${stem}.log"
    local csv="/data/raw/${stem}.csv"

    printf '[%s] label=%s GPUs=%s size=%s warmup=%s times=%s\n' \
        "${id}" "${label}" "${gpus}" "${size}" "${warmup}" "${times}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s,%s,%s,%s,0,dry-run\n' \
            "${id}" "${label}" "${gpus}" "${size}" >> "${STATUS_FILE}"
        return
    fi

    set +e
    srun \
        -N 1 -n 1 -G "${gpus}" \
        --gpus-per-node="${gpus}" \
        --gpus-per-task="${gpus}" \
        "${COMMON_SRUN[@]}" \
        --size "${size}" \
        --gpus-per-process "${gpus}" \
        --warmup "${warmup}" \
        --iterations "${times}" \
        --epsilon "${EPSILON}" \
        --label "${label}" \
        --output "${csv}" \
        > "${log}" 2>&1
    local rc=$?
    set -e

    printf '%s,%s,%s,%s,%s,%s\n' \
        "${id}" "${label}" "${gpus}" "${size}" "${rc}" \
        "raw/${stem}.log" >> "${STATUS_FILE}"
    if (( rc != 0 )); then
        echo "FAILED ${label}; log=${log}" >&2
        if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
            exit "${rc}"
        fi
    fi
}

case "${TARGET}" in
    smoke)
        run_case smoke_g2_s64 2 64 1 3
        ;;
    selection)
        run_case check_g2_s64 2 64 1 3
        run_case check_g8_s160 8 160 2 5
        run_case prod_g8_s320 8 320 "${WARMUP}" "${TIMES}"
        ;;
    production)
        run_case prod_g8_s320 8 320 "${WARMUP}" "${TIMES}"
        ;;
esac

python3 "${SCRIPT_DIR}/analyze_4d_results.py" \
    "${DATA_DIR}" \
    --baseline-ms "${BASELINE_MS}"

echo "Standalone 4D node-hybrid matrix complete: ${DATA_DIR}"
