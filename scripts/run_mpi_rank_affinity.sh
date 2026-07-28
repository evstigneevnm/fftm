#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: run_mpi_rank_affinity.sh [--mode auto|hca] [--] COMMAND [ARG ...]

The default HCA/NUMA maps describe the tested 8-GPU DGX-A100 nodes. Override
them for another machine with FFTM_MPI_GPU_HCA_MAP and FFTM_MPI_GPU_NUMA_MAP.

FFTM_MPI_GPU_HCA_MAP is a semicolon-separated entry per local GPU/rank. Each
entry is a comma-separated UCX device list.
EOF
}

MODE="${FFTM_MPI_RANK_AFFINITY_MODE:-${FFTM_SCFD_MPI_AFFINITY_MODE:-auto}}"
if [[ "${1:-}" == "--mode" ]]; then
    [[ $# -ge 3 ]] || {
        usage >&2
        exit 2
    }
    MODE="$2"
    shift 2
fi
if [[ "${1:-}" == "--" ]]; then
    shift
fi
if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

case "${MODE}" in
    auto|cpu)
        unset UCX_NET_DEVICES
        ;;
    hca|hca-cpu)
        ;;
    *)
        echo "Unknown MPI rank-affinity mode: ${MODE}" >&2
        exit 2
        ;;
esac

LOCAL_RANK="${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-}}"
GLOBAL_RANK="${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-unknown}}"
if [[ -z "${LOCAL_RANK}" || ! "${LOCAL_RANK}" =~ ^[0-9]+$ ]]; then
    echo "Cannot determine the numeric node-local MPI rank." >&2
    exit 2
fi

HCA_MAP="${FFTM_MPI_GPU_HCA_MAP:-mlx5_2:1,mlx5_3:1;mlx5_2:1,mlx5_3:1;mlx5_0:1,mlx5_1:1;mlx5_0:1,mlx5_1:1;mlx5_8:1,mlx5_9:1;mlx5_8:1,mlx5_9:1;mlx5_6:1,mlx5_7:1;mlx5_6:1,mlx5_7:1}"
NUMA_MAP="${FFTM_MPI_GPU_NUMA_MAP:-1,1,0,0,3,3,2,2}"
SYSFS_ROOT="${FFTM_MPI_AFFINITY_SYSFS_ROOT:-/sys/class/infiniband}"

IFS=';' read -r -a HCA_BY_GPU <<< "${HCA_MAP}"
IFS=',' read -r -a NUMA_BY_GPU <<< "${NUMA_MAP}"
if (( LOCAL_RANK >= ${#HCA_BY_GPU[@]} || LOCAL_RANK >= ${#NUMA_BY_GPU[@]} )); then
    echo "No affinity mapping exists for local rank ${LOCAL_RANK}." >&2
    exit 2
fi

GPU_LOCAL_HCAS="${HCA_BY_GPU[LOCAL_RANK]}"
GPU_NUMA="${NUMA_BY_GPU[LOCAL_RANK]}"
if [[ -z "${GPU_LOCAL_HCAS}" || -z "${GPU_NUMA}" ]]; then
    echo "Incomplete affinity mapping for local rank ${LOCAL_RANK}." >&2
    exit 2
fi

case "${MODE}" in
    hca|hca-cpu)
        IFS=',' read -r -a HCAS <<< "${GPU_LOCAL_HCAS}"
        for hca_port in "${HCAS[@]}"; do
            hca="${hca_port%%:*}"
            if [[ -z "${hca}" || ! -d "${SYSFS_ROOT}/${hca}" ]]; then
                echo "Mapped HCA is unavailable on $(hostname): ${hca:-<empty>}" >&2
                exit 1
            fi
        done
        export UCX_NET_DEVICES="${GPU_LOCAL_HCAS}"
        ;;
esac

printf '[FFTM_MPI_AFFINITY] host=%s rank=%s local_rank=%s gpu=%s gpu_numa=%s mode=%s ucx_net_devices=%s\n' \
    "$(hostname)" \
    "${GLOBAL_RANK}" \
    "${LOCAL_RANK}" \
    "${LOCAL_RANK}" \
    "${GPU_NUMA}" \
    "${MODE}" \
    "${UCX_NET_DEVICES:-auto}" >&2

exec "$@"
