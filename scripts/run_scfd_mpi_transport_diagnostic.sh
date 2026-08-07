#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_scfd_mpi_transport_diagnostic.sh [target]

Targets:
  all        collect node/GPU/HCA inventory and run the bandwidth matrix (default)
  inventory  collect only node/GPU/HCA/UCX inventory
  bandwidth  run only the SCFD transport matrix

Required:
  FFTM_CONTAINER_IMAGE

Optional:
  FFTM_DATA_DIR
  FFTM_SRUN_EXTRA_ARGS
  FFTM_SRUN_TIME
  FFTM_SCFD_MPI_NODES
  FFTM_SCFD_MPI_GPUS_PER_NODE
  FFTM_SCFD_MPI_ACTIVE_PAIRS
  FFTM_SCFD_MPI_MESSAGE_MIB
  FFTM_SCFD_MPI_TRANSPORTS
  FFTM_SCFD_MPI_ITERATIONS
  FFTM_SCFD_MPI_WARMUP
  FFTM_SCFD_MPI_UCX_LOG_LEVEL
  FFTM_SCFD_MPI_UCX_PROTO_INFO
  FFTM_SCFD_MPI_AFFINITY_MODE
  FFTM_SCFD_MPI_CPU_BIND_MAP
  FFTM_DRY_RUN
EOF
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

TARGET="${1:-all}"
case "${TARGET}" in
    -h|--help)
        usage
        exit 0
        ;;
    all|inventory|bandwidth)
        ;;
    *)
        echo "Unknown target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the FFTM .sqsh image path}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${FFTM_DATA_DIR:-${PWD}/data_scfd_mpi_transport_${STAMP}}"
NODES="${FFTM_SCFD_MPI_NODES:-2}"
GPUS_PER_NODE="${FFTM_SCFD_MPI_GPUS_PER_NODE:-8}"
TASKS="$((NODES * GPUS_PER_NODE))"
ACTIVE_PAIRS="${FFTM_SCFD_MPI_ACTIVE_PAIRS:-1,2,4,8}"
MESSAGE_MIB="${FFTM_SCFD_MPI_MESSAGE_MIB:-64,256,512,1024}"
TRANSPORTS="${FFTM_SCFD_MPI_TRANSPORTS:-device,pinned,staged}"
ITERATIONS="${FFTM_SCFD_MPI_ITERATIONS:-10}"
WARMUP="${FFTM_SCFD_MPI_WARMUP:-3}"
UCX_LOG_LEVEL_VALUE="${FFTM_SCFD_MPI_UCX_LOG_LEVEL:-warn}"
UCX_PROTO_INFO_VALUE="${FFTM_SCFD_MPI_UCX_PROTO_INFO:-n}"
SRUN_TIME="${FFTM_SRUN_TIME:-00:30:00}"
SRUN_EXTRA_TEXT="${FFTM_SRUN_EXTRA_ARGS---exclude=cn13}"
DRY_RUN="${FFTM_DRY_RUN:-0}"
AFFINITY_MODE="${FFTM_SCFD_MPI_AFFINITY_MODE:-auto}"
CPU_BIND_MAP="${FFTM_SCFD_MPI_CPU_BIND_MAP:-1,1,0,0,3,3,2,2,1,1,0,0,3,3,2,2}"
AFFINITY_WRAPPER="${SCRIPT_DIR}/run_mpi_rank_affinity.sh"

if [[ "${NODES}" -ne 1 && "${NODES}" -ne 2 ]]; then
    echo "This diagnostic requires one node or exactly two nodes." >&2
    exit 2
fi

case "${AFFINITY_MODE}" in
    auto|cpu|hca|hca-cpu)
        ;;
    *)
        echo "Unknown FFTM_SCFD_MPI_AFFINITY_MODE: ${AFFINITY_MODE}" >&2
        exit 2
        ;;
esac

if [[ "${GPUS_PER_NODE}" -ne 8 && "${AFFINITY_MODE}" != "auto" ]]; then
    echo "Affinity diagnostics currently require eight GPUs per node." >&2
    exit 2
fi

mkdir -p "${ROOT}"

read -r -a SRUN_EXTRA <<< "${SRUN_EXTRA_TEXT}"

CONTAINER_MOUNTS="${ROOT}:/data"
BANDWIDTH_COMMAND=(/opt/fftm/bin/test_scfd_mpi_transport.bin)
CPU_BIND_ARGS=()

case "${AFFINITY_MODE}" in
    cpu|hca-cpu)
        CPU_BIND_ARGS=("--cpu-bind=verbose,map_ldom:${CPU_BIND_MAP}")
        ;;
esac

case "${AFFINITY_MODE}" in
    hca|hca-cpu)
        test -x "${AFFINITY_WRAPPER}" || {
            echo "Missing affinity wrapper: ${AFFINITY_WRAPPER}" >&2
            exit 1
        }
        install -m 0755 \
            "${AFFINITY_WRAPPER}" \
            "${ROOT}/fftm_mpi_rank_affinity.sh"
        BANDWIDTH_COMMAND=(
            /data/fftm_mpi_rank_affinity.sh
            --mode "${AFFINITY_MODE}"
            --
            /opt/fftm/bin/test_scfd_mpi_transport.bin
        )
        ;;
esac

COMMON_SRUN=(
    srun
    --nodes="${NODES}"
    --gpus="$((NODES * GPUS_PER_NODE))"
    --gpus-per-node="${GPUS_PER_NODE}"
    --time="${SRUN_TIME}"
    --distribution=block:block
    --kill-on-bad-exit=1
    --container-image="${FFTM_CONTAINER_IMAGE}"
    --container-mounts="${CONTAINER_MOUNTS}"
    --container-workdir=/opt/fftm/bin
)

{
    printf 'target=%s\n' "${TARGET}"
    printf 'container=%s\n' "${FFTM_CONTAINER_IMAGE}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
    printf 'tasks=%s\n' "${TASKS}"
    printf 'active_pairs=%s\n' "${ACTIVE_PAIRS}"
    printf 'message_mib=%s\n' "${MESSAGE_MIB}"
    printf 'transports=%s\n' "${TRANSPORTS}"
    printf 'iterations=%s\n' "${ITERATIONS}"
    printf 'warmup=%s\n' "${WARMUP}"
    printf 'ucx_log_level=%s\n' "${UCX_LOG_LEVEL_VALUE}"
    printf 'ucx_proto_info=%s\n' "${UCX_PROTO_INFO_VALUE}"
    printf 'affinity_mode=%s\n' "${AFFINITY_MODE}"
    printf 'cpu_bind_map=%s\n' "${CPU_BIND_MAP}"
    printf 'srun_extra_args=%s\n' "${SRUN_EXTRA_TEXT}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
} > "${ROOT}/config.env"

print_command()
{
    printf 'DRY RUN:'
    printf ' %q' "$@"
    printf '\n'
}

run_inventory()
{
    local inventory_script
    inventory_script='
set -eu
echo "===== HOST $(hostname) ====="
echo "--- container stack ---"
printf "FFTM_BASE_IMAGE=%s\n" "${FFTM_BASE_IMAGE:-unknown}"
if [[ -r /etc/os-release ]]; then
    cat /etc/os-release
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
fi
if command -v mpicxx >/dev/null 2>&1; then
    mpicxx --showme:version 2>/dev/null || mpicxx --version
fi
if command -v ompi_info >/dev/null 2>&1; then
    ompi_info --version
fi
if command -v ucx_info >/dev/null 2>&1; then
    ucx_info -v
fi
if command -v ldd >/dev/null 2>&1; then
    ldd /opt/fftm/bin/test_scfd_mpi_transport.bin |
        grep -E "libmpi|libucp|libuct|libucs|libcuda|libcudart" || true
fi
echo "--- environment ---"
env | sort | grep -E "^(CUDA|NVIDIA|UCX|OMPI|PMI|PMIX|SLURM)_" || true
echo "--- GPU topology ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,driver_version --format=csv,noheader
    nvidia-smi topo -m
fi
echo "--- InfiniBand devices ---"
if command -v ibdev2netdev >/dev/null 2>&1; then
    ibdev2netdev
fi
for device in /sys/class/infiniband/*; do
    if [[ -e "${device}" ]]; then
        echo "device=$(basename "${device}")"
        if [[ -r "${device}/device/numa_node" ]]; then
            printf "numa_node="
            cat "${device}/device/numa_node"
        fi
        for port in "${device}"/ports/*; do
            [[ -d "${port}" ]] || continue
            printf "port=%s state=" "$(basename "${port}")"
            cat "${port}/state"
            printf "rate="
            cat "${port}/rate"
        done
    fi
done
echo "--- UCX devices ---"
if command -v ucx_info >/dev/null 2>&1; then
    ucx_info -d
fi
echo "--- Open MPI transport configuration ---"
if command -v ompi_info >/dev/null 2>&1; then
    ompi_info --parsable --all | grep -E "mca:pml:ucx|cuda_support|coll:ucc" || true
fi
test -x /opt/fftm/bin/test_scfd_mpi_transport.bin
'

    echo "Collecting ${NODES}-node GPU/HCA/UCX inventory..."
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command \
            "${COMMON_SRUN[@]}" \
            "${SRUN_EXTRA[@]}" \
            --ntasks="${NODES}" \
            --ntasks-per-node=1 \
            --container-entrypoint \
            /bin/bash \
            -lc "${inventory_script}"
        return
    fi
    "${COMMON_SRUN[@]}" \
        "${SRUN_EXTRA[@]}" \
        --ntasks="${NODES}" \
        --ntasks-per-node=1 \
        --container-entrypoint \
        /bin/bash \
        -lc "${inventory_script}" \
        > "${ROOT}/inventory.log" 2>&1
}

run_bandwidth()
{
    echo "Running SCFD MPI transport matrix on ${TASKS} ranks..."
    export SPSFD_DEVICE_BIND_DEBUG=1
    export UCX_LOG_LEVEL="${UCX_LOG_LEVEL_VALUE}"
    export UCX_PROTO_INFO="${UCX_PROTO_INFO_VALUE}"
    case "${AFFINITY_MODE}" in
        auto|cpu)
            unset UCX_NET_DEVICES
            ;;
    esac
    if [[ "${DRY_RUN}" == "1" ]]; then
        print_command \
            "${COMMON_SRUN[@]}" \
            "${SRUN_EXTRA[@]}" \
            "${CPU_BIND_ARGS[@]}" \
            --ntasks="${TASKS}" \
            --ntasks-per-node="${GPUS_PER_NODE}" \
            --container-entrypoint \
            "${BANDWIDTH_COMMAND[@]}" \
            --active-pairs "${ACTIVE_PAIRS}" \
            --message-mib "${MESSAGE_MIB}" \
            --transports "${TRANSPORTS}" \
            --iterations "${ITERATIONS}" \
            --warmup "${WARMUP}" \
            --output /data/scfd_mpi_transport.csv
        return
    fi
    "${COMMON_SRUN[@]}" \
        "${SRUN_EXTRA[@]}" \
        "${CPU_BIND_ARGS[@]}" \
        --ntasks="${TASKS}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --container-entrypoint \
        "${BANDWIDTH_COMMAND[@]}" \
        --active-pairs "${ACTIVE_PAIRS}" \
        --message-mib "${MESSAGE_MIB}" \
        --transports "${TRANSPORTS}" \
        --iterations "${ITERATIONS}" \
        --warmup "${WARMUP}" \
        --output /data/scfd_mpi_transport.csv \
        > "${ROOT}/transport.log" 2>&1

    test -s "${ROOT}/scfd_mpi_transport.csv"
    test -s "${ROOT}/scfd_mpi_transport.samples.csv"
    grep -q "SCFD_MPI_TRANSPORT PASSED" "${ROOT}/transport.log"
}

print_summary()
{
    if [[ "${DRY_RUN}" == "1" ]]; then
        return
    fi

    if [[ -s "${ROOT}/inventory.log" ]]; then
        echo
        echo "GPU inventory exposed inside the container:"
        awk '
            /^--- GPU topology ---$/ { printing = 1; next }
            /^--- InfiniBand devices ---$/ { printing = 0 }
            printing
        ' "${ROOT}/inventory.log"
    fi

    if [[ -s "${ROOT}/transport.log" ]]; then
        echo
        echo "SCFD rank-to-device binding and result:"
        grep -E \
            "SCFD_CUDA_BIND|SCFD_MPI_TOPOLOGY|SCFD_MPI_TRANSPORT PASSED" \
            "${ROOT}/transport.log" || true
    fi
}

case "${TARGET}" in
    all)
        run_inventory
        run_bandwidth
        ;;
    inventory)
        run_inventory
        ;;
    bandwidth)
        run_bandwidth
        ;;
esac

print_summary

cat <<EOF

SCFD MPI transport diagnostic completed:
  ${ROOT}
EOF
