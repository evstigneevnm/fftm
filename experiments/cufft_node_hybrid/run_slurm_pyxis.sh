#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE="${FFTM_NODE_HYBRID_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_node_hybrid_a100.sqsh}"
DATA_DIR="${FFTM_NODE_HYBRID_DATA_DIR:-/scratch/evstigneevnm/fftm/data_cufft_node_hybrid_$(date +%Y%m%d_%H%M%S)}"
TARGET="${FFTM_NODE_HYBRID_TARGET:-selection}"
TIMES="${FFTM_NODE_HYBRID_TIMES:-5}"
WARMUP="${FFTM_NODE_HYBRID_WARMUP:-2}"
CHUNK_MIB="${FFTM_NODE_HYBRID_CHUNK_MIB:-512}"
EPSILON="${FFTM_NODE_HYBRID_EPSILON:-1.0e-11}"
SRUN_TIME="${FFTM_NODE_HYBRID_SRUN_TIME:-01:00:00}"
ALLOCATION_TIME="${FFTM_NODE_HYBRID_ALLOCATION_TIME:-08:00:00}"
SALLOC_EXTRA_ARGS="${FFTM_NODE_HYBRID_SALLOC_EXTRA_ARGS:---exclude=cn13}"
STOP_ON_FAILURE="${FFTM_NODE_HYBRID_STOP_ON_FAILURE:-0}"
SKIP_PREFLIGHT="${FFTM_NODE_HYBRID_SKIP_PREFLIGHT:-0}"
DRY_RUN="${FFTM_NODE_HYBRID_DRY_RUN:-0}"

case "${TARGET}" in
    smoke)
        ALLOCATION_NODES=1
        ALLOCATION_GPUS_PER_NODE=2
        ;;
    local)
        ALLOCATION_NODES=1
        ALLOCATION_GPUS_PER_NODE=8
        ;;
    selection|multinode|all)
        ALLOCATION_NODES=2
        ALLOCATION_GPUS_PER_NODE=8
        ;;
    *)
        echo "Unknown FFTM_NODE_HYBRID_TARGET=${TARGET}; use smoke, local, multinode, selection, or all." >&2
        exit 2
        ;;
esac

if [[ "${1:-}" != "--inside-allocation" && -z "${SLURM_JOB_ID:-}" && "${DRY_RUN}" != "1" ]]; then
    [[ -f "${IMAGE}" ]] || {
        echo "Container image does not exist: ${IMAGE}" >&2
        exit 1
    }
    export FFTM_NODE_HYBRID_CONTAINER_IMAGE="${IMAGE}"
    export FFTM_NODE_HYBRID_DATA_DIR="${DATA_DIR}"
    export FFTM_NODE_HYBRID_TARGET="${TARGET}"
    export FFTM_NODE_HYBRID_TIMES="${TIMES}"
    export FFTM_NODE_HYBRID_WARMUP="${WARMUP}"
    export FFTM_NODE_HYBRID_CHUNK_MIB="${CHUNK_MIB}"
    export FFTM_NODE_HYBRID_EPSILON="${EPSILON}"
    export FFTM_NODE_HYBRID_SRUN_TIME="${SRUN_TIME}"
    export FFTM_NODE_HYBRID_ALLOCATION_TIME="${ALLOCATION_TIME}"
    export FFTM_NODE_HYBRID_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}"
    export FFTM_NODE_HYBRID_STOP_ON_FAILURE="${STOP_ON_FAILURE}"
    export FFTM_NODE_HYBRID_SKIP_PREFLIGHT="${SKIP_PREFLIGHT}"
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting node-hybrid allocation: nodes=%s GPUs/node=%s target=%s time=%s\n' \
        "${ALLOCATION_NODES}" "${ALLOCATION_GPUS_PER_NODE}" "${TARGET}" "${ALLOCATION_TIME}"
    exec salloc \
        -N "${ALLOCATION_NODES}" \
        --gres="gpu:${ALLOCATION_GPUS_PER_NODE}" \
        --ntasks-per-node=1 \
        --cpus-per-task=32 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SCRIPT_DIR}/run_slurm_pyxis.sh" --inside-allocation
fi

if [[ "${1:-}" == "--inside-allocation" ]]; then
    shift
fi

mkdir -p "${DATA_DIR}/raw"
STATUS_FILE="${DATA_DIR}/status.csv"
printf 'case_id,label,nodes,gpus_per_process,size,exchange,hca_mode,rc,log\n' > "${STATUS_FILE}"

cat > "${DATA_DIR}/config.env" <<EOF
container_image=${IMAGE}
target=${TARGET}
times=${TIMES}
warmup=${WARMUP}
chunk_mib=${CHUNK_MIB}
epsilon=${EPSILON}
srun_time=${SRUN_TIME}
allocation_nodes=${ALLOCATION_NODES}
allocation_gpus_per_node=${ALLOCATION_GPUS_PER_NODE}
allocation_time=${ALLOCATION_TIME}
salloc_extra_args=${SALLOC_EXTRA_ARGS}
slurm_job_id=${SLURM_JOB_ID:-}
slurm_job_nodelist=${SLURM_JOB_NODELIST:-}
git_commit=$(git -C "${ROOT_DIR}" rev-parse --short HEAD 2>/dev/null || printf unknown)
image_checksum=$(sha256sum "${IMAGE}" 2>/dev/null | awk '{print $1}' || printf unknown)
dry_run=${DRY_RUN}
EOF

BINARY="/opt/fftm_node_hybrid/bin/fftm_cufft_node_hybrid.bin"
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
    echo "Running cuFFT Xt node-hybrid preflight..."
    PREFLIGHT_LOG="${DATA_DIR}/raw/000_preflight.log"
    set +e
    srun \
        -N 1 -n 1 -G 2 \
        --gpus-per-node=2 \
        --gpus-per-task=2 \
        "${COMMON_SRUN[@]}" \
        --size 32 \
        --gpus-per-process 2 \
        --exchange none \
        --warmup 0 \
        --iterations 1 \
        --chunk-mib 64 \
        --epsilon "${EPSILON}" \
        --label preflight \
        --output /data/preflight.csv \
        > "${PREFLIGHT_LOG}" 2>&1
    PREFLIGHT_RC=$?
    set -e
    cat "${PREFLIGHT_LOG}"
    printf '000,preflight,1,2,32,none,auto,%s,raw/000_preflight.log\n' \
        "${PREFLIGHT_RC}" >> "${STATUS_FILE}"
    if (( PREFLIGHT_RC != 0 )); then
        echo "FAILED preflight; log=${PREFLIGHT_LOG}" >&2
        exit "${PREFLIGHT_RC}"
    fi
fi

case_id=0

run_case()
{
    local label="$1"
    local nodes="$2"
    local gpus="$3"
    local size="$4"
    local exchange="$5"
    local hca_mode="$6"

    case_id=$((case_id + 1))
    local id
    printf -v id '%03d' "${case_id}"
    local stem="${id}_${label}"
    local log="${DATA_DIR}/raw/${stem}.log"
    local csv="/data/raw/${stem}.csv"
    local total_gpus=$((nodes * gpus))

    printf '[%s] label=%s nodes=%s GPUs/process=%s size=%s exchange=%s HCA=%s\n' \
        "${id}" "${label}" "${nodes}" "${gpus}" "${size}" "${exchange}" "${hca_mode}"

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s,%s,%s,%s,%s,%s,%s,0,%s\n' \
            "${id}" "${label}" "${nodes}" "${gpus}" "${size}" "${exchange}" \
            "${hca_mode}" "dry-run" >> "${STATUS_FILE}"
        return
    fi

    set +e
    (
        case "${hca_mode}" in
            auto)
                unset UCX_NET_DEVICES
                unset UCX_MAX_RNDV_RAILS
                ;;
            pair)
                export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1"
                export UCX_MAX_RNDV_RAILS=2
                ;;
            all)
                export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1"
                export UCX_MAX_RNDV_RAILS=8
                ;;
            *)
                echo "Unknown HCA mode: ${hca_mode}" >&2
                exit 2
                ;;
        esac

        srun \
            -N "${nodes}" \
            -n "${nodes}" \
            --ntasks-per-node=1 \
            -G "${total_gpus}" \
            --gpus-per-node="${gpus}" \
            --gpus-per-task="${gpus}" \
            "${COMMON_SRUN[@]}" \
            --size "${size}" \
            --gpus-per-process "${gpus}" \
            --exchange "${exchange}" \
            --warmup "${WARMUP}" \
            --iterations "${TIMES}" \
            --chunk-mib "${CHUNK_MIB}" \
            --epsilon "${EPSILON}" \
            --label "${label}" \
            --output "${csv}"
    ) > "${log}" 2>&1
    local rc=$?
    set -e

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${id}" "${label}" "${nodes}" "${gpus}" "${size}" "${exchange}" \
        "${hca_mode}" "${rc}" "raw/${stem}.log" >> "${STATUS_FILE}"

    if (( rc != 0 )); then
        echo "FAILED ${label}; log=${log}" >&2
        if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
            exit "${rc}"
        fi
    fi
}

case "${TARGET}" in
    smoke)
        run_case smoke_2g 1 2 64 none auto
        ;;
    local)
        run_case local_2g_1024 1 2 1024 none auto
        run_case local_4g_1024 1 4 1024 none auto
        run_case local_8g_1024 1 8 1024 none auto
        run_case local_8g_2048 1 8 2048 none auto
        ;;
    multinode)
        run_case ring_8g_2048_auto 2 8 2048 ring auto
        run_case ring_8g_2048_pair 2 8 2048 ring pair
        run_case ring_8g_2048_all 2 8 2048 ring all
        ;;
    selection)
        run_case local_8g_2048 1 8 2048 none auto
        run_case ring_8g_2048_auto 2 8 2048 ring auto
        run_case ring_8g_2048_pair 2 8 2048 ring pair
        run_case ring_8g_2048_all 2 8 2048 ring all
        ;;
    all)
        run_case local_2g_1024 1 2 1024 none auto
        run_case local_4g_1024 1 4 1024 none auto
        run_case local_8g_1024 1 8 1024 none auto
        run_case local_8g_2048 1 8 2048 none auto
        run_case ring_8g_2048_auto 2 8 2048 ring auto
        run_case ring_8g_2048_pair 2 8 2048 ring pair
        run_case ring_8g_2048_all 2 8 2048 ring all
        ;;
esac

echo "Node-hybrid matrix complete: ${DATA_DIR}"
