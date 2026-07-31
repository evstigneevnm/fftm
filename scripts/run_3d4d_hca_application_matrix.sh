#!/usr/bin/env bash
set -u
set -o pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_3d4d_hca_application_matrix.sh

Runs eight application benchmarks on the same two 8-GPU nodes:
  FFTM 3D 2048^3: grids 2x8 and 4x4
  FFTM 4D 320^4: WZ plan concurrency 4 and 8
  Egger 3D 2048^3: forward/inverse on grids 2x8 and 4x4

Required:
  FFTM_CONTAINER_IMAGE
  EGGER_CONTAINER_IMAGE

Optional:
  FFTM_HCA_MATRIX_DATA_DIR
  FFTM_HCA_MATRIX_NODELIST       Two comma-separated nodes; otherwise discovered
  FFTM_HCA_MATRIX_SRUN_EXTRA_ARGS
  FFTM_HCA_MATRIX_DRY_RUN        0 or 1
EOF
}

if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the FFTM .sqsh path}"
: "${EGGER_CONTAINER_IMAGE:?Set EGGER_CONTAINER_IMAGE to the Egger .sqsh path}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}" || exit 1

STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${FFTM_HCA_MATRIX_DATA_DIR:-${PWD}/data_3d4d_hca_application_${STAMP}}"
EGGER_IMAGE="${EGGER_CONTAINER_IMAGE}"
NODELIST="${FFTM_HCA_MATRIX_NODELIST:-}"
BASE_SRUN_EXTRA="${FFTM_HCA_MATRIX_SRUN_EXTRA_ARGS:---exclude=cn13 --distribution=block:block --kill-on-bad-exit=1}"
DRY_RUN="${FFTM_HCA_MATRIX_DRY_RUN:-0}"
STATUS_FILE="${ROOT}/status.tsv"
read -r -a BASE_SRUN_EXTRA_ARRAY <<< "${BASE_SRUN_EXTRA}"

case "${DRY_RUN}" in
    0|1)
        ;;
    *)
        echo "FFTM_HCA_MATRIX_DRY_RUN must be 0 or 1." >&2
        exit 2
        ;;
esac

if [[ "${DRY_RUN}" == "0" ]]; then
    test -r "${FFTM_CONTAINER_IMAGE}" || {
        echo "Missing FFTM image: ${FFTM_CONTAINER_IMAGE}" >&2
        exit 1
    }
    test -r "${EGGER_IMAGE}" || {
        echo "Missing Egger image: ${EGGER_IMAGE}" >&2
        exit 1
    }
fi

mkdir -p "${ROOT}"
ROOT="$(cd "${ROOT}" && pwd -P)"
printf 'suite\tstatus\toutput\n' > "${STATUS_FILE}"

if [[ -z "${NODELIST}" ]]; then
    if [[ "${DRY_RUN}" == "1" ]]; then
        NODELIST="cn-dry0,cn-dry1"
    else
        echo "Selecting two 8-GPU nodes for the complete matrix..."
        node_output="$(
            srun \
                --nodes=2 \
                --ntasks=2 \
                --ntasks-per-node=1 \
                --gpus-per-node=8 \
                --time=00:05:00 \
                "${BASE_SRUN_EXTRA_ARRAY[@]}" \
                hostname
        )" || {
            echo "Node selection failed." >&2
            exit 1
        }
        mapfile -t nodes < <(
            printf '%s\n' "${node_output}" |
                sed 's/\..*$//' |
                sed '/^[[:space:]]*$/d' |
                sort -u
        )
        if [[ "${#nodes[@]}" -ne 2 ]]; then
            echo "Expected two selected nodes, got: ${node_output}" >&2
            exit 1
        fi
        NODELIST="$(IFS=,; printf '%s' "${nodes[*]}")"
    fi
fi

IFS=',' read -r -a selected_nodes <<< "${NODELIST}"
if [[ "${#selected_nodes[@]}" -ne 2 ]]; then
    echo "FFTM_HCA_MATRIX_NODELIST must contain exactly two nodes." >&2
    exit 2
fi

PINNED_SRUN_EXTRA="${BASE_SRUN_EXTRA} --nodelist=${NODELIST} --ntasks-per-node=8"
printf 'Selected nodes: %s\n' "${NODELIST}"
printf 'Pinned Slurm options: %s\n' "${PINNED_SRUN_EXTRA}"
printf 'Expected benchmark cases: FFTM=4, Egger=4, total=8\n'

summary_matches()
{
    local suite="$1"
    local summary_file="$2"
    local expected_runs="$3"

    python3 - "${suite}" "${summary_file}" "${expected_runs}" <<'PY'
import json
import sys
from pathlib import Path

suite, summary_name, expected_text = sys.argv[1:]
summary_path = Path(summary_name)
expected = int(expected_text)
if not summary_path.is_file():
    print(f"ERROR: missing {suite} summary: {summary_path}", file=sys.stderr)
    raise SystemExit(1)

try:
    summary = json.loads(summary_path.read_text())
except (OSError, json.JSONDecodeError) as exc:
    print(f"ERROR: invalid {suite} summary {summary_path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if suite == "fftm":
    planned = int(summary.get("planned_runs", -1))
    measured = int(summary.get("num_measurement_specs", -1))
    failed = int(summary.get("failed_measurement_runs", -1))
    ok = planned == expected and measured == expected and failed == 0
    detail = f"planned={planned}, measured={measured}, failed={failed}"
elif suite == "egger":
    runs = int(summary.get("runs", -1))
    succeeded = int(summary.get("ok", -1))
    failed = int(summary.get("failed", -1))
    ok = runs == expected and succeeded == expected and failed == 0
    detail = f"runs={runs}, ok={succeeded}, failed={failed}"
else:
    print(f"ERROR: unsupported suite summary type: {suite}", file=sys.stderr)
    raise SystemExit(2)

if not ok:
    print(
        f"ERROR: incomplete {suite} matrix ({detail}); expected {expected} successful cases.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

FFTM_OUTPUT="${ROOT}/fftm"
if env \
    FFTM_CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE}" \
    FFTM_DATA_DIR="${FFTM_OUTPUT}" \
    FFTM_SRUN_EXTRA_ARGS="${PINNED_SRUN_EXTRA}" \
    FFTM_MPI_RANK_AFFINITY_MODE=hca \
    FFTM_DRY_RUN="${DRY_RUN}" \
    bash scripts/run_7day_multinode_smoke.sh hca-selection
then
    fftm_runner_status=0
else
    fftm_runner_status=$?
fi
if [[ "${fftm_runner_status}" -eq 0 ]] &&
    { [[ "${DRY_RUN}" == "1" ]] ||
      summary_matches fftm "${FFTM_OUTPUT}/summary.json" 4; }
then
    fftm_status=PASS
else
    fftm_status=FAIL
fi
printf 'fftm\t%s\t%s\n' "${fftm_status}" "${FFTM_OUTPUT}" >> "${STATUS_FILE}"

EGGER_OUTPUT="${ROOT}/egger"
if env \
    FFTM_CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE}" \
    EGGER_CONTAINER_IMAGE="${EGGER_IMAGE}" \
    EGGER_MATRIX_DATA_DIR="${EGGER_OUTPUT}" \
    EGGER_SRUN_EXTRA_ARGS="${PINNED_SRUN_EXTRA}" \
    EGGER_MPI_RANK_AFFINITY_MODE=hca \
    EGGER_TRANSPORTS=cuda_aware \
    EGGER_FAIL_ON_ERROR=1 \
    EGGER_DRY_RUN="${DRY_RUN}" \
    bash fft_Egger/scripts/run_egger_multinode_transport_matrix.sh multinode
then
    egger_runner_status=0
else
    egger_runner_status=$?
fi
if [[ "${egger_runner_status}" -eq 0 ]] &&
    { [[ "${DRY_RUN}" == "1" ]] ||
      summary_matches egger "${EGGER_OUTPUT}/16g_2n/summary.json" 4; }
then
    egger_status=PASS
else
    egger_status=FAIL
fi
printf 'egger\t%s\t%s\n' "${egger_status}" "${EGGER_OUTPUT}" >> "${STATUS_FILE}"

{
    printf 'container_fftm=%s\n' "${FFTM_CONTAINER_IMAGE}"
    printf 'container_egger=%s\n' "${EGGER_IMAGE}"
    printf 'nodes=%s\n' "${NODELIST}"
    printf 'mpi_rank_affinity_mode=hca\n'
    printf 'srun_extra_args=%s\n' "${PINNED_SRUN_EXTRA}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
} > "${ROOT}/config.env"

echo
echo "HCA application matrix completed: ${ROOT}"
if command -v column >/dev/null 2>&1; then
    column -t -s $'\t' "${STATUS_FILE}"
else
    cat "${STATUS_FILE}"
fi

if [[ "${fftm_status}" != "PASS" || "${egger_status}" != "PASS" ]]; then
    exit 1
fi
