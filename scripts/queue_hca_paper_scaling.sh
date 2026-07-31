#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/queue_hca_paper_scaling.sh [TARGET ...]

With no targets, launch all paper-scaling matrices concurrently in this order:
  fftm-120 egger-120 fftm-96 egger-96 fftm-64 egger-64
  fftm-32 egger-32 fftm-16 egger-16 fftm-8 egger-8

Pass an explicit subset to limit the run, for example:
  scripts/queue_hca_paper_scaling.sh fftm-120 egger-120 fftm-96 egger-96

Set FFTM_SCALING_ROOT to select the common output directory.

When any target uses 96 or 120 GPUs, the default is to request one shared
allocation and run all targets sequentially inside it. Overrides:
  FFTM_SCALING_USE_SINGLE_ALLOCATION=auto|0|1
  FFTM_SCALING_ALLOCATION_TIME
  FFTM_SCALING_SALLOC_EXTRA_ARGS
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

if [[ $# -eq 0 ]]; then
    TARGETS=(
        fftm-120 egger-120
        fftm-96 egger-96
        fftm-64 egger-64
        fftm-32 egger-32
        fftm-16 egger-16
        fftm-8 egger-8
    )
else
    TARGETS=( "$@" )
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${FFTM_SCALING_ROOT:-/scratch/evstigneevnm/fftm/data_scale_hca_${STAMP}}"
USE_SINGLE_ALLOCATION="${FFTM_SCALING_USE_SINGLE_ALLOCATION:-auto}"
ALLOCATION_TIME="${FFTM_SCALING_ALLOCATION_TIME:-08:00:00}"
SALLOC_EXTRA_ARGS="${FFTM_SCALING_SALLOC_EXTRA_ARGS:---exclude=cn13}"

case "${USE_SINGLE_ALLOCATION}" in
    auto|0|1)
        ;;
    *)
        echo "FFTM_SCALING_USE_SINGLE_ALLOCATION must be auto, 0, or 1." >&2
        exit 2
        ;;
esac

max_gpus=0
for target in "${TARGETS[@]}"; do
    target_gpus="${target##*-}"
    if [[ ! "${target_gpus}" =~ ^[0-9]+$ ]]; then
        echo "Invalid scaling target: ${target}" >&2
        exit 2
    fi
    (( target_gpus > max_gpus )) && max_gpus="${target_gpus}"
done

allocate_shared=0
if [[ "${USE_SINGLE_ALLOCATION}" == "1" ||
      ( "${USE_SINGLE_ALLOCATION}" == "auto" && "${max_gpus}" -ge 96 ) ]]; then
    allocate_shared=1
fi

if [[ "${FFTM_SCALING_DRY_RUN:-0}" == "0" &&
      -z "${SLURM_JOB_ID:-}" &&
      "${FFTM_SCALING_INSIDE_SHARED_ALLOCATION:-0}" != "1" &&
      "${allocate_shared}" == "1" ]]; then
    max_nodes=$(( max_gpus / 8 ))
    if (( max_nodes * 8 != max_gpus )); then
        echo "Shared scaling allocation requires full 8-GPU nodes." >&2
        exit 2
    fi
    mkdir -p "${ROOT}"
    ROOT="$(cd "${ROOT}" && pwd -P)"
    read -r -a salloc_extra_args <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting one shared scaling allocation: nodes=%s GPUs=%s time=%s\n' \
        "${max_nodes}" "${max_gpus}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_SCALING_ROOT="${ROOT}" \
        FFTM_SCALING_INSIDE_SHARED_ALLOCATION=1 \
        FFTM_SCALING_USE_SINGLE_ALLOCATION=0 \
        salloc \
        --job-name="fftm-scale-${max_gpus}" \
        --nodes="${max_nodes}" \
        --ntasks="${max_gpus}" \
        --ntasks-per-node=8 \
        --gpus-per-node=8 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra_args[@]}" \
        bash "${SCRIPT_DIR}/queue_hca_paper_scaling.sh" "${TARGETS[@]}"
fi

mkdir -p "${ROOT}/logs"
ROOT="$(cd "${ROOT}" && pwd -P)"
STATUS_FILE="${ROOT}/status.tsv"
printf 'target\tstatus\toutput\tlog\n' > "${STATUS_FILE}"
if [[ "${FFTM_SCALING_INSIDE_SHARED_ALLOCATION:-0}" == "1" ]]; then
    {
        printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
        printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-}"
        printf 'allocation_time=%s\n' "${ALLOCATION_TIME}"
        printf 'max_gpus=%s\n' "${max_gpus}"
        printf 'execution=sequential\n'
    } > "${ROOT}/shared_allocation.env"
fi

run_target()
{
    local target="$1"
    local output="$2"
    local log="$3"
    FFTM_SCALING_DATA_DIR="${output}" \
        "${SCRIPT_DIR}/run_hca_paper_scaling.sh" "${target}" > "${log}" 2>&1
}

if [[ "${FFTM_SCALING_INSIDE_SHARED_ALLOCATION:-0}" == "1" ]]; then
    overall_status=0
    for target in "${TARGETS[@]}"; do
        output="${ROOT}/${target}"
        log="${ROOT}/logs/${target}.log"
        printf 'Running %-9s in shared allocation -> %s\n' "${target}" "${output}"
        if run_target "${target}" "${output}" "${log}"; then
            status="PASS"
        else
            status="FAIL"
            overall_status=1
        fi
        printf '%s\t%s\t%s\t%s\n' \
            "${target}" "${status}" "${output}" "${log}" >> "${STATUS_FILE}"
        printf '%-9s %s\n' "${target}" "${status}"
    done
    printf 'Scaling queue completed: %s\n' "${ROOT}"
    exit "${overall_status}"
fi

pids=()
outputs=()
logs=()
for target in "${TARGETS[@]}"; do
    output="${ROOT}/${target}"
    log="${ROOT}/logs/${target}.log"
    printf 'Queueing %-9s -> %s\n' "${target}" "${output}"
    (
        run_target "${target}" "${output}" "${log}"
    ) &
    pids+=( "$!" )
    outputs+=( "${output}" )
    logs+=( "${log}" )
done

overall_status=0
for index in "${!pids[@]}"; do
    target="${TARGETS[index]}"
    if wait "${pids[index]}"; then
        status="PASS"
    else
        status="FAIL"
        overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\n' \
        "${target}" "${status}" "${outputs[index]}" "${logs[index]}" >> "${STATUS_FILE}"
    printf '%-9s %s\n' "${target}" "${status}"
done

printf 'Scaling queue completed: %s\n' "${ROOT}"
exit "${overall_status}"
