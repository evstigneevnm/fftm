#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
TARGET=${1:-smoke}

IMAGE=${FFTM_CPP_TUNE_CONTAINER_IMAGE:-${FFTM_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}}
DATA_DIR=${FFTM_CPP_TUNE_DATA_DIR:-${FFTM_DATA_DIR:-${ROOT_DIR}/data_cpp_autotune_${TARGET}_$(date +%Y%m%d_%H%M%S)}}
GPU_COUNTS=${FFTM_CPP_TUNE_GPU_COUNTS:-}
SIZE=${FFTM_CPP_TUNE_SIZE:-}
AUTOTUNE_WARMUP=${FFTM_CPP_TUNE_WARMUP:-}
AUTOTUNE_TIMES=${FFTM_CPP_TUNE_TIMES:-}
APPLICATION_WARMUP=${FFTM_CPP_TUNE_APPLICATION_WARMUP:-}
APPLICATION_TIMES=${FFTM_CPP_TUNE_APPLICATION_TIMES:-}
MEMORY_RECOVERY_TOLERANCE_MIB=${FFTM_CPP_TUNE_MEMORY_RECOVERY_TOLERANCE_MIB:-1024}
STRICT_DEVICE_IDENTITY=${FFTM_CPP_TUNE_STRICT_DEVICE_IDENTITY:-0}
STEP_TIME=${FFTM_CPP_TUNE_STEP_TIME:-}
ALLOCATION_TIME=${FFTM_CPP_TUNE_ALLOCATION_TIME:-}
USE_SINGLE_ALLOCATION=${FFTM_CPP_TUNE_USE_SINGLE_ALLOCATION:-1}
SALLOC_EXTRA_ARGS=${FFTM_CPP_TUNE_SALLOC_EXTRA_ARGS:---exclude=cn13}
SRUN_EXTRA_ARGS=${FFTM_CPP_TUNE_SRUN_EXTRA_ARGS:-}
DRY_RUN=${FFTM_CPP_TUNE_DRY_RUN:-0}

case "${TARGET}" in
    smoke)
        GPU_COUNTS=${GPU_COUNTS:-4}
        SIZE=${SIZE:-256}
        AUTOTUNE_WARMUP=${AUTOTUNE_WARMUP:-1}
        AUTOTUNE_TIMES=${AUTOTUNE_TIMES:-2}
        APPLICATION_WARMUP=${APPLICATION_WARMUP:-0}
        APPLICATION_TIMES=${APPLICATION_TIMES:-1}
        STEP_TIME=${STEP_TIME:-00:20:00}
        ALLOCATION_TIME=${ALLOCATION_TIME:-01:00:00}
        ;;
    production)
        GPU_COUNTS=${GPU_COUNTS:-6,7,8}
        SIZE=${SIZE:-2048}
        AUTOTUNE_WARMUP=${AUTOTUNE_WARMUP:-3}
        AUTOTUNE_TIMES=${AUTOTUNE_TIMES:-10}
        APPLICATION_WARMUP=${APPLICATION_WARMUP:-1}
        APPLICATION_TIMES=${APPLICATION_TIMES:-3}
        STEP_TIME=${STEP_TIME:-02:00:00}
        ALLOCATION_TIME=${ALLOCATION_TIME:-08:00:00}
        ;;
    *)
        echo "Usage: $0 [smoke|production]" >&2
        exit 2
        ;;
esac

for value in "${SIZE}" "${AUTOTUNE_WARMUP}" "${AUTOTUNE_TIMES}" \
             "${APPLICATION_WARMUP}" "${APPLICATION_TIMES}" "${MEMORY_RECOVERY_TOLERANCE_MIB}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        echo "Invalid nonnegative integer: ${value}" >&2
        exit 2
    }
done
(( SIZE > 0 && AUTOTUNE_TIMES > 0 && APPLICATION_TIMES > 0 )) || {
    echo "Size and timing iteration counts must be positive" >&2
    exit 2
}
[[ "${STRICT_DEVICE_IDENTITY}" == "0" || "${STRICT_DEVICE_IDENTITY}" == "1" ]] || {
    echo "FFTM_CPP_TUNE_STRICT_DEVICE_IDENTITY must be 0 or 1" >&2
    exit 2
}

IFS=',' read -r -a GPU_VALUES <<< "${GPU_COUNTS}"
MAX_GPUS=0
for gpus in "${GPU_VALUES[@]}"; do
    [[ "${gpus}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid GPU count: ${gpus}" >&2
        exit 2
    }
    (( gpus > MAX_GPUS )) && MAX_GPUS=${gpus}
done
(( MAX_GPUS <= 8 )) || {
    echo "The one-node C++ autotune validation supports at most 8 GPUs" >&2
    exit 2
}

mkdir -p "${DATA_DIR}"
DATA_DIR=$(cd "${DATA_DIR}" && pwd -P)

if [[ "${DRY_RUN}" == "0" && -z "${SLURM_JOB_ID:-}" && "${USE_SINGLE_ALLOCATION}" == "1" ]]; then
    command -v salloc >/dev/null || {
        echo "Missing required command: salloc" >&2
        exit 2
    }
    read -r -a SALLOC_EXTRA_ITEMS <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting C++ autotune validation allocation: target=%s GPUs=%d time=%s\n' \
        "${TARGET}" "${MAX_GPUS}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_CPP_TUNE_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_CPP_TUNE_DATA_DIR="${DATA_DIR}" \
        FFTM_CPP_TUNE_GPU_COUNTS="${GPU_COUNTS}" \
        FFTM_CPP_TUNE_SIZE="${SIZE}" \
        FFTM_CPP_TUNE_WARMUP="${AUTOTUNE_WARMUP}" \
        FFTM_CPP_TUNE_TIMES="${AUTOTUNE_TIMES}" \
        FFTM_CPP_TUNE_APPLICATION_WARMUP="${APPLICATION_WARMUP}" \
        FFTM_CPP_TUNE_APPLICATION_TIMES="${APPLICATION_TIMES}" \
        FFTM_CPP_TUNE_MEMORY_RECOVERY_TOLERANCE_MIB="${MEMORY_RECOVERY_TOLERANCE_MIB}" \
        FFTM_CPP_TUNE_STRICT_DEVICE_IDENTITY="${STRICT_DEVICE_IDENTITY}" \
        FFTM_CPP_TUNE_STEP_TIME="${STEP_TIME}" \
        FFTM_CPP_TUNE_ALLOCATION_TIME="${ALLOCATION_TIME}" \
        FFTM_CPP_TUNE_USE_SINGLE_ALLOCATION=0 \
        FFTM_CPP_TUNE_SALLOC_EXTRA_ARGS= \
        FFTM_CPP_TUNE_SRUN_EXTRA_ARGS= \
        FFTM_CPP_TUNE_DRY_RUN=0 \
        salloc \
        --job-name="fftm-cpp-tune-${TARGET}" \
        --nodes=1 \
        --ntasks="${MAX_GPUS}" \
        --ntasks-per-node="${MAX_GPUS}" \
        --cpus-per-task=4 \
        --gpus-per-node="${MAX_GPUS}" \
        --exclusive \
        --time="${ALLOCATION_TIME}" \
        "${SALLOC_EXTRA_ITEMS[@]}" \
        bash "${SCRIPT_DIR}/run_cpp_autotune_validation.sh" "${TARGET}"
fi

if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${SLURM_JOB_NUM_NODES}" -ne 1 ]]; then
    echo "C++ autotune validation requires exactly one allocated node" >&2
    exit 2
fi

STATUS_FILE="${DATA_DIR}/status.csv"
CONFIG_FILE="${DATA_DIR}/config.env"
printf '%s\n' 'gpus,size,stage,status,returncode,log' > "${STATUS_FILE}"
{
    printf 'target=%s\n' "${TARGET}"
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'data_dir=%s\n' "${DATA_DIR}"
    printf 'gpu_counts=%s\n' "${GPU_COUNTS}"
    printf 'size=%s\n' "${SIZE}"
    printf 'autotune_warmup=%s\n' "${AUTOTUNE_WARMUP}"
    printf 'autotune_times=%s\n' "${AUTOTUNE_TIMES}"
    printf 'application_warmup=%s\n' "${APPLICATION_WARMUP}"
    printf 'application_times=%s\n' "${APPLICATION_TIMES}"
    printf 'memory_recovery_tolerance_mib=%s\n' "${MEMORY_RECOVERY_TOLERANCE_MIB}"
    printf 'strict_device_identity=%s\n' "${STRICT_DEVICE_IDENTITY}"
    printf 'step_time=%s\n' "${STEP_TIME}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-}"
    printf 'git_commit=%s\n' "$(git -C "${ROOT_DIR}" rev-parse --verify HEAD 2>/dev/null || printf unknown)"
} > "${CONFIG_FILE}"

read -r -a SRUN_EXTRA_ITEMS <<< "${SRUN_EXTRA_ARGS}"

PREFLIGHT_RANKS=1
(( MAX_GPUS >= 2 )) && PREFLIGHT_RANKS=2
PREFLIGHT_COMMAND=(
    srun
    "${SRUN_EXTRA_ITEMS[@]}"
    -N 1
    -n "${PREFLIGHT_RANKS}"
    -G "${PREFLIGHT_RANKS}"
    --ntasks-per-node="${PREFLIGHT_RANKS}"
    --gpus-per-node="${PREFLIGHT_RANKS}"
    --cpus-per-task=1
    --distribution=block:block
    --kill-on-bad-exit=1
    --time=00:05:00
    --container-image "${IMAGE}"
    --container-mounts="${DATA_DIR}:/data"
    --container-workdir /opt/fftm/bin
    --container-entrypoint /opt/fftm/bin/test_fftm_autotune_hardware.bin
    /data/preflight_cache.env
)
printf '[preflight] '
printf '%q ' "${PREFLIGHT_COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
    "${PREFLIGHT_COMMAND[@]}" 2>&1 | tee "${DATA_DIR}/preflight.log"
fi

LIFECYCLE_COMMAND=(
    srun
    "${SRUN_EXTRA_ITEMS[@]}"
    -N 1
    -n 4
    -G 4
    --ntasks-per-node=4
    --gpus-per-node=4
    --cpus-per-task=1
    --distribution=block:block
    --kill-on-bad-exit=1
    --time=00:10:00
    --container-image "${IMAGE}"
    --container-mounts="${DATA_DIR}:/data"
    --container-workdir /opt/fftm/bin
    --container-entrypoint /usr/bin/env
    FFTM_WRAP_PROCS_GPUS=0
    /opt/fftm/bin/test_fftm_resource_lifecycle.bin
)
printf '[resource-lifecycle] '
printf '%q ' "${LIFECYCLE_COMMAND[@]}"
printf '\n'
if [[ "${DRY_RUN}" != "1" ]]; then
    "${LIFECYCLE_COMMAND[@]}" 2>&1 | tee "${DATA_DIR}/resource_lifecycle.log"
fi

make_poisson_command()
{
    local gpus="$1"
    local case_dir="$2"
    shift 2
    POISSON_COMMAND=(
        srun
        "${SRUN_EXTRA_ITEMS[@]}"
        -N 1
        -n "${gpus}"
        -G "${gpus}"
        --ntasks-per-node="${gpus}"
        --gpus-per-node="${gpus}"
        --cpus-per-task=4
        --distribution=block:block
        --kill-on-bad-exit=1
        --time="${STEP_TIME}"
        --container-image "${IMAGE}"
        --container-mounts="${case_dir}:/data"
        --container-workdir /opt/fftm/bin
        --container-entrypoint /usr/bin/env
        FFTM_WRAP_PROCS_GPUS=0
        FFTM_CPP_AUTOTUNE_MEASURE=1
        FFTM_CPP_AUTOTUNE_STRICT_DEVICE_IDENTITY="${STRICT_DEVICE_IDENTITY}"
        FFTM_CPP_AUTOTUNE_LOG_CANDIDATE_MEMORY=1
        FFTM_CPP_AUTOTUNE_VERIFY_MEMORY_RECOVERY=1
        FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY_TOLERANCE_MIB="${MEMORY_RECOVERY_TOLERANCE_MIB}"
        FFTM_CPP_AUTOTUNE_WARMUP="${AUTOTUNE_WARMUP}"
        FFTM_CPP_AUTOTUNE_TIMES="${AUTOTUNE_TIMES}"
        "$@"
        /opt/fftm/bin/poisson_periodic_3d_autotuned.bin
        "${SIZE}" "${SIZE}" "${SIZE}"
        /data/autotune.env
        "${APPLICATION_TIMES}" "${APPLICATION_WARMUP}"
    )
}

run_stage()
{
    local gpus="$1"
    local stage="$2"
    local log_file="$3"
    shift 3
    make_poisson_command "${gpus}" "$(dirname "${log_file}")" "$@"
    printf '[%sG %s] ' "${gpus}" "${stage}"
    printf '%q ' "${POISSON_COMMAND[@]}"
    printf '\n'
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%s,%s,%s,dry-run,0,%s\n' "${gpus}" "${SIZE}" "${stage}" "${log_file}" >> "${STATUS_FILE}"
        return 0
    fi

    set +e
    "${POISSON_COMMAND[@]}" 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e
    local status=passed
    (( rc == 0 )) || status=failed
    printf '%s,%s,%s,%s,%s,%s\n' "${gpus}" "${SIZE}" "${stage}" "${status}" "${rc}" "${log_file}" \
        >> "${STATUS_FILE}"
    return "${rc}"
}

overall_rc=0
for gpus in "${GPU_VALUES[@]}"; do
    CASE_DIR="${DATA_DIR}/${gpus}g"
    mkdir -p "${CASE_DIR}"
    CREATE_LOG="${CASE_DIR}/01_measure.log"
    REUSE_LOG="${CASE_DIR}/02_cache.log"
    CONSTRAINED_LOG="${CASE_DIR}/03_pencil_cache.log"

    if ! run_stage "${gpus}" measure "${CREATE_LOG}"; then
        overall_rc=1
        continue
    fi
    if ! run_stage "${gpus}" cache "${REUSE_LOG}"; then
        overall_rc=1
        continue
    fi
    if ! run_stage \
        "${gpus}" pencil-cache "${CONSTRAINED_LOG}" \
        FFTM_CPP_AUTOTUNE_STRATEGY_3D=pencil-pencil; then
        overall_rc=1
        continue
    fi

    if [[ "${DRY_RUN}" == "0" ]]; then
        CACHE_FILE="${CASE_DIR}/autotune.env"
        RESULT_COUNT=$(grep -c 'FFTM_CPP_AUTOTUNE_RESULT' "${CREATE_LOG}" || true)
        RELEASE_COUNT=$(grep -c 'phase=released' "${CREATE_LOG}" || true)
        RECOVERY_COUNT=$(grep -c 'FFTM_CPP_AUTOTUNE_MEMORY_RECOVERY' "${CREATE_LOG}" || true)
        if ! grep -q 'source=measured' "${CREATE_LOG}" ||
           ! grep -q 'source=cache' "${REUSE_LOG}" ||
           ! grep -q 'strategy=pencil-pencil.*source=cache' "${CONSTRAINED_LOG}" ||
           [[ "${RESULT_COUNT}" -ne 6 ]] ||
           [[ "${RELEASE_COUNT}" -ne 6 ]] ||
           [[ "${RECOVERY_COUNT}" -ne 6 ]] ||
           ! grep -q 'workspace_pool_mib=' "${CREATE_LOG}" ||
           ! grep -q 'input_pool_mib=' "${CREATE_LOG}" ||
           ! grep -q 'spectrum_pool_mib=' "${CREATE_LOG}" ||
           ! grep -q 'retained_pool_mib=' "${CREATE_LOG}" ||
           grep -Eq 'input_pool_mib=[1-9][0-9]*' "${CREATE_LOG}" ||
           ! grep -Eq 'spectrum_pool_mib=[1-9][0-9]*' "${CREATE_LOG}" ||
           grep -q 'retained .*after plan destruction' "${CREATE_LOG}" ||
           ! grep -q '^FFTM_AUTOTUNE_SCHEMA=2$' "${CACHE_FILE}" ||
           ! grep -q '^FFTM_AUTOTUNE_SOURCE=cpp-measured-v2$' "${CACHE_FILE}" ||
           ! grep -q '^FFTM_AUTOTUNE_CANDIDATE_POLICY_VERSION=2$' "${CACHE_FILE}" ||
           ! grep -q '^FFTM_AUTOTUNE_PRODUCTION_CANDIDATES_ONLY=1$' "${CACHE_FILE}" ||
           ! grep -q '^FFTM_AUTOTUNE_CANDIDATE_COUNT=6$' "${CACHE_FILE}" ||
           ! grep -q "^FFTM_AUTOTUNE_RANKS_PER_NODE=${gpus}$" "${CACHE_FILE}" ||
           ! grep -q "^FFTM_AUTOTUNE_DEVICES_PER_NODE=${gpus}$" "${CACHE_FILE}"; then
            echo "Validation failed for ${gpus} GPUs; inspect ${CASE_DIR}" >&2
            printf '%s,%s,validate,failed,1,%s\n' "${gpus}" "${SIZE}" "${CASE_DIR}" >> "${STATUS_FILE}"
            overall_rc=1
        else
            printf '%s,%s,validate,passed,0,%s\n' "${gpus}" "${SIZE}" "${CASE_DIR}" >> "${STATUS_FILE}"
        fi
    fi
done

if [[ "${DRY_RUN}" == "0" ]]; then
    grep -h 'poisson_periodic_3d_autotuned:' "${DATA_DIR}"/*g/*.log > "${DATA_DIR}/selected_configs.log" || true
fi

echo "C++ autotune validation results: ${DATA_DIR}"
exit "${overall_rc}"
