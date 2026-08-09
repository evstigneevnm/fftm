#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_cpp_autotune_policy_guard.sh [--inside-allocation]

Validates the release C++ 3D policy cache on selected GPU counts. Each case creates
the cache, executes a 2048^3 Poisson solve, reuses the cache in a second solve,
and verifies the selected production strategy, grid, and layout. The launcher also runs the
two-rank collective-reporting regression test from the SQSH.

Useful overrides:
  FFTM_POLICY_GUARD_CONTAINER_IMAGE   default: /scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh
  FFTM_POLICY_GUARD_DATA_DIR          default: timestamped directory under /scratch/evstigneevnm/fftm
  FFTM_POLICY_GUARD_GPU_COUNTS        default: 16,32
  FFTM_POLICY_GUARD_SIZE              default: 2048
  FFTM_POLICY_GUARD_GPUS_PER_NODE     default: 8
  FFTM_POLICY_GUARD_APPLICATION_TIMES default: 1
  FFTM_POLICY_GUARD_APPLICATION_WARMUP default: 0
  FFTM_POLICY_GUARD_STEP_TIME         default: 00:30:00
  FFTM_POLICY_GUARD_ALLOCATION_TIME   default: 02:00:00
  FFTM_POLICY_GUARD_FORBIDDEN_NODES   default: cn13,cn24,cn43
  FFTM_POLICY_GUARD_SALLOC_EXTRA_ARGS default: --exclude=<forbidden nodes>
  FFTM_POLICY_GUARD_DRY_RUN=0|1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

inside_allocation=0
if [[ "${1:-}" == "--inside-allocation" ]]; then
    inside_allocation=1
    shift
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SELF=${SCRIPT_DIR}/run_cpp_autotune_policy_guard.sh

IMAGE=${FFTM_POLICY_GUARD_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}
STAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR=${FFTM_POLICY_GUARD_DATA_DIR:-/scratch/evstigneevnm/fftm/data_cpp_policy_guard_${STAMP}}
GPU_COUNTS=${FFTM_POLICY_GUARD_GPU_COUNTS:-16,32}
SIZE=${FFTM_POLICY_GUARD_SIZE:-2048}
GPUS_PER_NODE=${FFTM_POLICY_GUARD_GPUS_PER_NODE:-8}
APPLICATION_TIMES=${FFTM_POLICY_GUARD_APPLICATION_TIMES:-1}
APPLICATION_WARMUP=${FFTM_POLICY_GUARD_APPLICATION_WARMUP:-0}
STEP_TIME=${FFTM_POLICY_GUARD_STEP_TIME:-00:30:00}
ALLOCATION_TIME=${FFTM_POLICY_GUARD_ALLOCATION_TIME:-02:00:00}
FORBIDDEN_NODES=${FFTM_POLICY_GUARD_FORBIDDEN_NODES:-cn13,cn24,cn43}
SALLOC_EXTRA_ARGS=${FFTM_POLICY_GUARD_SALLOC_EXTRA_ARGS:---exclude=${FORBIDDEN_NODES}}
DRY_RUN=${FFTM_POLICY_GUARD_DRY_RUN:-0}

case "${DRY_RUN}" in
    0|1) ;;
    *) echo "FFTM_POLICY_GUARD_DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
for value in "${SIZE}" "${GPUS_PER_NODE}" "${APPLICATION_TIMES}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Size, GPUs per node, and application times must be positive integers." >&2
        exit 2
    }
done
[[ "${APPLICATION_WARMUP}" =~ ^[0-9]+$ ]] || {
    echo "Application warmup must be a nonnegative integer." >&2
    exit 2
}

IFS=',' read -r -a gpu_values <<< "${GPU_COUNTS}"
(( ${#gpu_values[@]} > 0 )) || {
    echo "At least one GPU count is required." >&2
    exit 2
}
max_nodes=0
for gpus in "${gpu_values[@]}"; do
    [[ "${gpus}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid GPU count '${gpus}'." >&2
        exit 2
    }
    (( gpus % GPUS_PER_NODE == 0 )) || {
        echo "GPU count ${gpus} must be divisible by GPUs per node ${GPUS_PER_NODE}." >&2
        exit 2
    }
    nodes=$((gpus / GPUS_PER_NODE))
    (( nodes > max_nodes )) && max_nodes=${nodes}
done

DATA_DIR=$(realpath -m -- "${DATA_DIR}")
if [[ "${DRY_RUN}" == 0 ]]; then
    test -f "${IMAGE}" || {
        echo "Missing SQSH image: ${IMAGE}" >&2
        exit 2
    }
fi

if [[ "${DRY_RUN}" == 0 && -z "${SLURM_JOB_ID:-}" ]]; then
    command -v salloc >/dev/null || {
        echo "salloc is required outside an existing allocation." >&2
        exit 2
    }
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting C++ policy guard allocation: nodes=%s GPUs/node=%s time=%s\n' \
        "${max_nodes}" "${GPUS_PER_NODE}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_POLICY_GUARD_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_POLICY_GUARD_DATA_DIR="${DATA_DIR}" \
        FFTM_POLICY_GUARD_GPU_COUNTS="${GPU_COUNTS}" \
        FFTM_POLICY_GUARD_SIZE="${SIZE}" \
        FFTM_POLICY_GUARD_GPUS_PER_NODE="${GPUS_PER_NODE}" \
        FFTM_POLICY_GUARD_APPLICATION_TIMES="${APPLICATION_TIMES}" \
        FFTM_POLICY_GUARD_APPLICATION_WARMUP="${APPLICATION_WARMUP}" \
        FFTM_POLICY_GUARD_STEP_TIME="${STEP_TIME}" \
        FFTM_POLICY_GUARD_ALLOCATION_TIME="${ALLOCATION_TIME}" \
        FFTM_POLICY_GUARD_FORBIDDEN_NODES="${FORBIDDEN_NODES}" \
        FFTM_POLICY_GUARD_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}" \
        salloc \
        --job-name=fftm-policy-guard \
        --nodes="${max_nodes}" \
        --ntasks=$((max_nodes * GPUS_PER_NODE)) \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task=4 \
        --gres="gpu:${GPUS_PER_NODE}" \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SELF}" --inside-allocation
fi

if [[ "${DRY_RUN}" == 0 && "${inside_allocation}" == 0 && -n "${SLURM_JOB_ID:-}" ]]; then
    inside_allocation=1
fi
if [[ "${DRY_RUN}" == 0 && -n "${SLURM_JOB_NUM_NODES:-}" &&
      "${SLURM_JOB_NUM_NODES}" -lt "${max_nodes}" ]]; then
    echo "Policy guard requires at least ${max_nodes} allocated nodes." >&2
    exit 2
fi

if [[ -e "${DATA_DIR}" && -n "$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory is not empty: ${DATA_DIR}" >&2
    exit 2
fi
mkdir -p "${DATA_DIR}"

allocated_hosts=()
if [[ "${DRY_RUN}" == 0 ]]; then
    scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${DATA_DIR}/hosts.txt"
    mapfile -t allocated_hosts < "${DATA_DIR}/hosts.txt"
    forbidden_pattern="^($(printf '%s' "${FORBIDDEN_NODES}" | sed 's/,/|/g'))$"
    if grep -Eq "${forbidden_pattern}" "${DATA_DIR}/hosts.txt"; then
        echo "Refusing to run on a forbidden node (${FORBIDDEN_NODES})." >&2
        exit 2
    fi
    scontrol show job "${SLURM_JOB_ID}" > "${DATA_DIR}/slurm_job.txt"
    sha256sum "${IMAGE}" > "${DATA_DIR}/image.sha256"
fi

{
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'gpu_counts=%s\n' "${GPU_COUNTS}"
    printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
    printf 'size_3d=%s\n' "${SIZE}"
    printf 'application_times=%s\n' "${APPLICATION_TIMES}"
    printf 'application_warmup=%s\n' "${APPLICATION_WARMUP}"
    printf 'policy_source=cpp-policy-cache-v4\n'
    printf 'policy_version=4\n'
    printf 'forbidden_nodes=%s\n' "${FORBIDDEN_NODES}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DATA_DIR}/config.env"
printf 'gpus,nodes,stage,status,returncode,log\n' > "${DATA_DIR}/status.csv"

run_or_print()
{
    local log_file=$1
    shift
    printf 'Command: '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == 1 ]]; then
        return 0
    fi
    local rc=0
    if "$@" > "${log_file}" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    cat "${log_file}"
    return "${rc}"
}

REPORTING_COMMAND=(
    srun -N 1 -n 2 --ntasks-per-node=2 --cpus-per-task=1
    --exclusive --kill-on-bad-exit=1 --time=00:05:00
    --container-image "${IMAGE}"
    --container-workdir /opt/fftm/bin
    --container-entrypoint /opt/fftm/bin/test_fft_benchmark_reporting.bin
)
echo "Running collective-reporting preflight..."
run_or_print "${DATA_DIR}/collective_reporting.log" "${REPORTING_COMMAND[@]}"

for gpus in "${gpu_values[@]}"; do
    nodes=$((gpus / GPUS_PER_NODE))
    case_dir="${DATA_DIR}/${gpus}g"
    mkdir -p "${case_dir}"
    expected_strategy=slab-pencil
    expected_grid="${gpus}x1"
    expected_mode=p2p-waitany
    expected_layout=auto
    if (( gpus == 32 )); then
        expected_mode=alltoallv
    elif (( gpus >= 64 )); then
        expected_strategy=pencil-pencil
        expected_grid="$((gpus / GPUS_PER_NODE))x${GPUS_PER_NODE}"
        expected_layout=opt0
    fi
    case_node_args=()
    if [[ "${DRY_RUN}" == 0 ]]; then
        case_host_list=$(IFS=,; printf '%s' "${allocated_hosts[*]:0:${nodes}}")
        case_node_args=(--nodelist="${case_host_list}")
    fi

    for stage in create cache; do
        log_file="${case_dir}/${stage}.log"
        POISSON_COMMAND=(
            env
            OMPI_MCA_pml=ucx
            UCX_MAX_RNDV_RAILS=2
            UCX_WARN_UNUSED_ENV_VARS=n
            srun
            "${case_node_args[@]}"
            -N "${nodes}"
            -n "${gpus}"
            -G "${gpus}"
            --ntasks-per-node="${GPUS_PER_NODE}"
            --gpus-per-node="${GPUS_PER_NODE}"
            --cpus-per-task=4
            --distribution=block:block
            --cpu-bind=none
            --exclusive
            --kill-on-bad-exit=1
            --time="${STEP_TIME}"
            --container-image "${IMAGE}"
            --container-mounts="${case_dir}:/data"
            --container-workdir /opt/fftm/bin
            --container-entrypoint /usr/bin/env
            FFTM_WRAP_PROCS_GPUS=0
            FFTM_CPP_AUTOTUNE_MEASURE=0
            FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY=1
            FFTM_MPI_RANK_AFFINITY_MODE=hca
            /opt/fftm/scripts/run_mpi_rank_affinity.sh
            --mode hca
            --
            /opt/fftm/bin/poisson_periodic_3d_autotuned.bin
            "${SIZE}" "${SIZE}" "${SIZE}"
            /data/autotune.env
            "${APPLICATION_TIMES}" "${APPLICATION_WARMUP}"
        )

        echo "[${gpus}G ${stage}]"
        rc=0
        if run_or_print "${log_file}" "${POISSON_COMMAND[@]}"; then
            rc=0
        else
            rc=$?
        fi
        status=passed
        if [[ "${DRY_RUN}" == 1 ]]; then
            status=dry-run
        elif (( rc != 0 )); then
            status=failed
        fi
        printf '%s,%s,%s,%s,%s,%s\n' \
            "${gpus}" "${nodes}" "${stage}" "${status}" "${rc}" "${log_file}" \
            >> "${DATA_DIR}/status.csv"
        (( rc == 0 )) || exit "${rc}"
    done

    if [[ "${DRY_RUN}" == 1 ]]; then
        continue
    fi

    cache_file="${case_dir}/autotune.env"
    grep -Fxq 'FFTM_AUTOTUNE_SOURCE=cpp-policy-cache-v4' "${cache_file}"
    grep -Fxq 'FFTM_AUTOTUNE_POLICY_VERSION=4' "${cache_file}"
    grep -Fxq "FFTM_AUTOTUNE_STRATEGY_3D=${expected_strategy}" "${cache_file}"
    grep -Fxq "FFTM_AUTOTUNE_GRID_3D=${expected_grid}" "${cache_file}"
    grep -Fxq "FFTM_AUTOTUNE_MODE=${expected_mode}" "${cache_file}"
    grep -Fxq "FFTM_AUTOTUNE_PENCIL_LAYOUT=${expected_layout}" "${cache_file}"
    if [[ "${expected_layout}" == opt0 ]]; then
        grep -Fxq 'FFTM_AUTOTUNE_PENCIL_PIPELINE=reference-parity' "${cache_file}"
        grep -Fxq 'FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT=1' "${cache_file}"
        grep -Fxq 'FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY=1' "${cache_file}"
        grep -Fxq 'FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC=1' "${cache_file}"
        grep -Fxq 'FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR=1' "${cache_file}"
    fi
    grep -q "strategy=${expected_strategy} mode=${expected_mode} grid=${expected_grid}.*layout=${expected_layout}.*source=created" "${case_dir}/create.log"
    grep -q "strategy=${expected_strategy} mode=${expected_mode} grid=${expected_grid}.*layout=${expected_layout}.*source=cache" "${case_dir}/cache.log"
    printf '%s,%s,validate,passed,0,%s\n' \
        "${gpus}" "${nodes}" "${case_dir}" >> "${DATA_DIR}/status.csv"
done

echo "C++ policy guard complete: ${DATA_DIR}"
