#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${GPUFFT_REF_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_gpu_fft_reference_a100.sqsh}"
TARGET="${GPUFFT_REF_TARGET:-full}"
if [[ -n "${1:-}" && "${1}" != "--inside-allocation" ]]; then
    TARGET="$1"
    shift
fi
DATA_DIR="${GPUFFT_REF_DATA_DIR:-/scratch/evstigneevnm/fftm/data_gpu_fft_reference_${TARGET}_$(date +%Y%m%d_%H%M%S)}"
TIMES="${GPUFFT_REF_TIMES:-30}"
WARMUP="${GPUFFT_REF_WARMUP:-5}"
EPSILON="${GPUFFT_REF_EPSILON:-1.0e-11}"
SRUN_TIME="${GPUFFT_REF_SRUN_TIME:-01:00:00}"
ALLOCATION_TIME="${GPUFFT_REF_ALLOCATION_TIME:-06:00:00}"
SALLOC_EXTRA_ARGS="${GPUFFT_REF_SALLOC_EXTRA_ARGS:---exclude=cn13}"
STOP_ON_FAILURE="${GPUFFT_REF_STOP_ON_FAILURE:-0}"
SKIP_PREFLIGHT="${GPUFFT_REF_SKIP_PREFLIGHT:-0}"
DRY_RUN="${GPUFFT_REF_DRY_RUN:-0}"
GPUS_PER_NODE=8
CPUS_PER_TASK="${GPUFFT_REF_CPUS_PER_TASK:-8}"
AFFINITY_MODE="${GPUFFT_REF_AFFINITY_MODE:-hca}"

case "${TARGET}" in
    smoke|node1) ALLOCATION_NODES=1 ;;
    node2) ALLOCATION_NODES=2 ;;
    node3) ALLOCATION_NODES=3 ;;
    node4|published|full) ALLOCATION_NODES=4 ;;
    *)
        echo "Unknown GPUFFT_REF_TARGET=${TARGET}; use smoke, node1, node2, node3, node4, published, or full." >&2
        exit 2
        ;;
esac

case "${AFFINITY_MODE}" in
    auto|hca) ;;
    *)
        echo "GPUFFT_REF_AFFINITY_MODE must be auto or hca." >&2
        exit 2
        ;;
esac

case "${STOP_ON_FAILURE}" in
    0|1) ;;
    *)
        echo "GPUFFT_REF_STOP_ON_FAILURE must be 0 or 1." >&2
        exit 2
        ;;
esac

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" && "${DRY_RUN}" != "1" ]]; then
    [[ -f "${IMAGE}" ]] || {
        echo "Container image does not exist: ${IMAGE}" >&2
        exit 1
    }
    export GPUFFT_REF_CONTAINER_IMAGE="${IMAGE}"
    export GPUFFT_REF_DATA_DIR="${DATA_DIR}"
    export GPUFFT_REF_TARGET="${TARGET}"
    export GPUFFT_REF_TIMES="${TIMES}"
    export GPUFFT_REF_WARMUP="${WARMUP}"
    export GPUFFT_REF_EPSILON="${EPSILON}"
    export GPUFFT_REF_SRUN_TIME="${SRUN_TIME}"
    export GPUFFT_REF_ALLOCATION_TIME="${ALLOCATION_TIME}"
    export GPUFFT_REF_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}"
    export GPUFFT_REF_STOP_ON_FAILURE="${STOP_ON_FAILURE}"
    export GPUFFT_REF_SKIP_PREFLIGHT="${SKIP_PREFLIGHT}"
    export GPUFFT_REF_CPUS_PER_TASK="${CPUS_PER_TASK}"
    export GPUFFT_REF_AFFINITY_MODE="${AFFINITY_MODE}"
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting GPU-FFT reference allocation: target=%s nodes=%s GPUs/node=%s time=%s\n' \
        "${TARGET}" "${ALLOCATION_NODES}" "${GPUS_PER_NODE}" "${ALLOCATION_TIME}"
    exec salloc \
        -N "${ALLOCATION_NODES}" \
        --gres="gpu:${GPUS_PER_NODE}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SCRIPT_DIR}/run_slurm_pyxis.sh" --inside-allocation
fi

if [[ "${1:-}" == "--inside-allocation" ]]; then
    shift
fi

mkdir -p "${DATA_DIR}/raw"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"
STATUS_FILE="${DATA_DIR}/status.csv"
printf 'case_id,label,size,nodes,gpus,affinity,rc,summary,iterations,log\n' > "${STATUS_FILE}"

IMAGE_CHECKSUM="$(sha256sum "${IMAGE}" 2>/dev/null | awk '{print $1}' || printf unknown)"
cat > "${DATA_DIR}/config.env" <<EOF
container_image=${IMAGE}
image_checksum=${IMAGE_CHECKSUM}
target=${TARGET}
times=${TIMES}
warmup=${WARMUP}
epsilon=${EPSILON}
srun_time=${SRUN_TIME}
allocation_nodes=${ALLOCATION_NODES}
gpus_per_node=${GPUS_PER_NODE}
cpus_per_task=${CPUS_PER_TASK}
allocation_time=${ALLOCATION_TIME}
salloc_extra_args=${SALLOC_EXTRA_ARGS}
affinity_mode=${AFFINITY_MODE}
gpu_visibility=node
slurm_job_id=${SLURM_JOB_ID:-}
slurm_job_nodelist=${SLURM_JOB_NODELIST:-}
git_commit=$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || printf unknown)
gpu_fft_upstream_commit=ff4d84bbc6e0ecbcd52246c9d7e83198dadb2426
matrix_2048_nodes=1,2,4
matrix_1920_nodes=1,2,3,4
matrix_note=2048 is excluded at 24 ranks because 2048 modulo 24 is nonzero
dry_run=${DRY_RUN}
EOF

BINARY="/opt/gpu_fft_reference/bin/gpu_fft_reference_benchmark.bin"
AFFINITY_WRAPPER="/opt/gpu_fft_reference/bin/run_mpi_rank_affinity.sh"

run_srun()
{
    local nodes="$1"
    shift
    local ranks=$((nodes * GPUS_PER_NODE))

    OMPI_MCA_pml=ucx \
    UCX_MAX_RNDV_RAILS=2 \
    UCX_WARN_UNUSED_ENV_VARS=n \
    srun \
        -N "${nodes}" \
        -n "${ranks}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        -G "${ranks}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --distribution=block:block \
        --cpu-bind=none \
        --exclusive \
        --kill-on-bad-exit=1 \
        --time="${SRUN_TIME}" \
        --container-image "${IMAGE}" \
        "--container-mounts=${DATA_DIR}:/data" \
        --container-workdir /opt/gpu_fft_reference \
        --container-entrypoint \
        "${AFFINITY_WRAPPER}" \
        --mode "${AFFINITY_MODE}" \
        -- \
        "${BINARY}" \
        "$@"
}

if [[ "${SKIP_PREFLIGHT}" != "1" && "${DRY_RUN}" != "1" ]]; then
    echo "Running GPU-FFT image, CUDA-aware MPI, affinity, and numerical preflight..."
    PREFLIGHT_LOG="${DATA_DIR}/raw/preflight.log"
    set +e
    run_srun 1 \
        --size 128 \
        --warmup 0 \
        --times 1 \
        --epsilon "${EPSILON}" \
        --label preflight \
        --summary /data/raw/preflight_summary.csv \
        --iterations /data/raw/preflight_iterations.csv \
        > "${PREFLIGHT_LOG}" 2>&1
    PREFLIGHT_RC=$?
    set -e
    cat "${PREFLIGHT_LOG}"
    if (( PREFLIGHT_RC != 0 )); then
        echo "GPU-FFT reference preflight failed; log=${PREFLIGHT_LOG}" >&2
        exit "${PREFLIGHT_RC}"
    fi
fi

case_id=0
run_case()
{
    local size="$1"
    local nodes="$2"
    local ranks=$((nodes * GPUS_PER_NODE))
    case_id=$((case_id + 1))
    local id
    printf -v id 'c%02d' "${case_id}"
    local label="n${size}_g${ranks}"
    local summary="raw/${id}_summary.csv"
    local iterations="raw/${id}_iterations.csv"
    local log="raw/${id}.log"

    printf '[%s] GPU-FFT size=%s^3 nodes=%s ranks/GPUs=%s affinity=%s\n' \
        "${id}" "${size}" "${nodes}" "${ranks}" "${AFFINITY_MODE}"

    if (( size % ranks != 0 )); then
        echo "Internal matrix error: size ${size} does not divide rank count ${ranks}." >&2
        exit 2
    fi
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s,%s,%s,%s,%s,%s,0,%s,%s,%s\n' \
            "${id}" "${label}" "${size}" "${nodes}" "${ranks}" "${AFFINITY_MODE}" \
            "${summary}" "${iterations}" "dry-run" >> "${STATUS_FILE}"
        return
    fi

    set +e
    run_srun "${nodes}" \
        --size "${size}" \
        --warmup "${WARMUP}" \
        --times "${TIMES}" \
        --epsilon "${EPSILON}" \
        --label "${label}" \
        --summary "/data/${summary}" \
        --iterations "/data/${iterations}" \
        > "${DATA_DIR}/${log}" 2>&1
    local rc=$?
    set -e

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${id}" "${label}" "${size}" "${nodes}" "${ranks}" "${AFFINITY_MODE}" \
        "${rc}" "${summary}" "${iterations}" "${log}" >> "${STATUS_FILE}"
    if (( rc != 0 )); then
        echo "FAILED ${label}; log=${DATA_DIR}/${log}" >&2
        if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
            exit "${rc}"
        fi
    fi
}

case "${TARGET}" in
    smoke)
        run_case 128 1
        ;;
    node1)
        run_case 1920 1
        run_case 2048 1
        ;;
    node2)
        run_case 1920 2
        run_case 2048 2
        ;;
    node3)
        run_case 1920 3
        ;;
    node4)
        run_case 1920 4
        run_case 2048 4
        ;;
    published)
        run_case 2048 1
        run_case 2048 2
        run_case 2048 4
        ;;
    full)
        run_case 1920 1
        run_case 1920 2
        run_case 1920 3
        run_case 1920 4
        run_case 2048 1
        run_case 2048 2
        run_case 2048 4
        ;;
esac

if [[ "${DRY_RUN}" != "1" ]]; then
    python3 "${SCRIPT_DIR}/analyze_results.py" --data-dir "${DATA_DIR}"
fi
echo "GPU-FFT reference matrix complete: ${DATA_DIR}"
