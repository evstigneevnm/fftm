#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/queue_4d_weak_scaling_endpoints.sh [96] [120]

Queues the selected endpoint launchers independently. With no arguments, both
96- and 120-GPU jobs are queued. Output and launcher logs are collected below
FFTM_WEAK4D_QUEUE_ROOT.
EOF
}

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    usage
    exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ $# -eq 0 ]]; then
    TARGETS=( 96 120 )
else
    TARGETS=( "$@" )
fi
for target in "${TARGETS[@]}"; do
    case "${target}" in
        96|120) ;;
        *) echo "Invalid target: ${target}" >&2; usage >&2; exit 2 ;;
    esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
ROOT=${FFTM_WEAK4D_QUEUE_ROOT:-/scratch/evstigneevnm/fftm/data_4d_weak_endpoints_${STAMP}}
mkdir -p "${ROOT}/logs"
ROOT=$(cd "${ROOT}" && pwd -P)
STATUS=${ROOT}/queue_status.tsv
printf 'target\tstatus\toutput\tlog\n' > "${STATUS}"

pids=()
outputs=()
logs=()
for target in "${TARGETS[@]}"; do
    output=${ROOT}/g${target}
    log=${ROOT}/logs/g${target}.log
    printf 'Queueing %s-GPU endpoint -> %s\n' "${target}" "${output}"
    (
        FFTM_WEAK4D_DATA_DIR="${output}" \
            bash "${SCRIPT_DIR}/run_4d_weak_scaling_endpoints.sh" "${target}"
    ) > "${log}" 2>&1 &
    pids+=( "$!" )
    outputs+=( "${output}" )
    logs+=( "${log}" )
done

overall_status=0
for index in "${!pids[@]}"; do
    target=${TARGETS[index]}
    if wait "${pids[index]}"; then
        status=PASS
    else
        status=FAIL
        overall_status=1
    fi
    printf '%s\t%s\t%s\t%s\n' \
        "${target}" "${status}" "${outputs[index]}" "${logs[index]}" >> "${STATUS}"
    printf '%s-GPU endpoint: %s\n' "${target}" "${status}"
done

printf 'Endpoint queue complete: %s\n' "${ROOT}"
exit "${overall_status}"
