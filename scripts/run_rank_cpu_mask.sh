#!/usr/bin/env bash
set -euo pipefail

rank="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-0}}"
mask_list="${FFTM_CPU_MASKS:?FFTM_CPU_MASKS must contain colon-separated CPU masks}"

IFS=':' read -r -a masks <<< "${mask_list}"
if (( rank < 0 || rank >= ${#masks[@]} )); then
    echo "No CPU mask for MPI rank ${rank}; masks=${mask_list}" >&2
    exit 2
fi

exec taskset -c "${masks[rank]}" "$@"
