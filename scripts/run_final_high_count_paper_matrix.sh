#!/usr/bin/env bash
set -u
set -o pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_final_high_count_paper_matrix.sh [TARGET]

Runs the final high-count paper confirmations in one Slurm allocation. The
default target is "all" and contains six independent cases:

  d4_g120  320^4, 120 GPUs, slab-slab, production auto credit policy
  d3_g120  2048^3, 120 GPUs, pencil-pencil 15x8 opt0
  d4_g96   320^4,  96 GPUs, slab-slab, production auto credit policy
  d3_g96   2048^3,  96 GPUs, pencil-pencil 12x8 opt0
  d4_g64   320^4,  64 GPUs, slab-slab, production auto credit policy
  d3_g24   2048^3,  24 GPUs, slab-pencil p2p-waitany (optional)

TARGET can be:
  all       all six cases (default)
  required  all cases except optional d3_g24
  3d        d3_g120, d3_g96, and d3_g24
  4d        d4_g120, d4_g96, and d4_g64
  g120      d4_g120 and d3_g120
  g96       d4_g96 and d3_g96
  g64       d4_g64 only
  g24       d3_g24 only

The launcher continues after a failed case and writes status.csv. A required
case failure makes the final exit status nonzero; the optional 24-GPU point
does not.

Required:
  FFTM_CONTAINER_IMAGE=/path/to/fftm_bench_a100.sqsh

Useful overrides:
  FFTM_HIGH_COUNT_DATA_DIR
  FFTM_HIGH_COUNT_ALLOCATION_TIME       default: 04:00:00
  FFTM_HIGH_COUNT_SALLOC_EXTRA_ARGS     default: --exclude=cn13,cn24,cn43
  FFTM_HIGH_COUNT_TIMES                 default: 30
  FFTM_HIGH_COUNT_WARMUP                default: 5
  FFTM_HIGH_COUNT_STEP_TIME             default: 00:45:00
  FFTM_HIGH_COUNT_TIMEOUT_SECONDS       default: 2700
  FFTM_HIGH_COUNT_HASH_IMAGE            default: 1
  FFTM_DRY_RUN                          default: 0
EOF
}

TARGET=${1:-all}
case "${TARGET}" in
    all|required|3d|4d|g120|g96|g64|g24) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

: "${FFTM_CONTAINER_IMAGE:?Set FFTM_CONTAINER_IMAGE to the rebuilt FFTM .sqsh image}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
SELF=${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")
STAMP=$(date +%Y%m%d_%H%M%S)
ROOT=${FFTM_HIGH_COUNT_DATA_DIR:-${PWD}/data_final_high_count_${STAMP}}
ALLOCATION_TIME=${FFTM_HIGH_COUNT_ALLOCATION_TIME:-04:00:00}
SALLOC_EXTRA_TEXT=${FFTM_HIGH_COUNT_SALLOC_EXTRA_ARGS:---exclude=cn13,cn24,cn43}
BENCHMARK_TIMES=${FFTM_HIGH_COUNT_TIMES:-30}
WARMUP=${FFTM_HIGH_COUNT_WARMUP:-5}
STEP_TIME=${FFTM_HIGH_COUNT_STEP_TIME:-00:45:00}
TIMEOUT_SECONDS=${FFTM_HIGH_COUNT_TIMEOUT_SECONDS:-2700}
HASH_IMAGE=${FFTM_HIGH_COUNT_HASH_IMAGE:-1}
DRY_RUN=${FFTM_DRY_RUN:-0}
GPUS_PER_NODE=8

is_true()
{
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

is_positive_integer()
{
    [[ "${1:-}" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_integer()
{
    [[ "${1:-}" =~ ^[0-9]+$ ]]
}

if ! is_positive_integer "${BENCHMARK_TIMES}" ||
   ! is_nonnegative_integer "${WARMUP}" ||
   ! is_positive_integer "${TIMEOUT_SECONDS}"; then
    echo "Times and timeout must be positive integers; warmup must be nonnegative." >&2
    exit 2
fi
case "${HASH_IMAGE}" in 0|1) ;; *) echo "FFTM_HIGH_COUNT_HASH_IMAGE must be 0 or 1." >&2; exit 2 ;; esac

ALL_CASES=(d4_g120 d3_g120 d4_g96 d3_g96 d4_g64 d3_g24)

case_selected()
{
    local case_id=$1
    case "${TARGET}" in
        all) return 0 ;;
        required) [[ "${case_id}" != d3_g24 ]] ;;
        3d) [[ "${case_id}" == d3_* ]] ;;
        4d) [[ "${case_id}" == d4_* ]] ;;
        g120) [[ "${case_id}" == *_g120 ]] ;;
        g96) [[ "${case_id}" == *_g96 ]] ;;
        g64) [[ "${case_id}" == *_g64 ]] ;;
        g24) [[ "${case_id}" == *_g24 ]] ;;
    esac
}

case_properties()
{
    CASE_ID=$1
    CASE_REQUIRED=1
    CASE_SIZE_3D=2048
    CASE_SIZE_4D=320
    CASE_GRID=configured
    CASE_LAYOUT=configured
    CASE_PIPELINE=configured
    case "${CASE_ID}" in
        d4_g120) CASE_DIM=4; CASE_GPUS=120; CASE_NODES=15; CASE_STRATEGY=slab-slab; CASE_MODE=p2p-waitany ;;
        d3_g120) CASE_DIM=3; CASE_GPUS=120; CASE_NODES=15; CASE_STRATEGY=pencil-pencil; CASE_MODE=p2p-waitany; CASE_GRID=15x8; CASE_LAYOUT=opt0; CASE_PIPELINE=reference-parity ;;
        d4_g96)  CASE_DIM=4; CASE_GPUS=96;  CASE_NODES=12; CASE_STRATEGY=slab-slab; CASE_MODE=p2p-waitany ;;
        d3_g96)  CASE_DIM=3; CASE_GPUS=96;  CASE_NODES=12; CASE_STRATEGY=pencil-pencil; CASE_MODE=p2p-waitany; CASE_GRID=12x8; CASE_LAYOUT=opt0; CASE_PIPELINE=reference-parity ;;
        d4_g64)  CASE_DIM=4; CASE_GPUS=64;  CASE_NODES=8;  CASE_STRATEGY=slab-slab; CASE_MODE=p2p-waitany ;;
        d3_g24)  CASE_DIM=3; CASE_GPUS=24;  CASE_NODES=3;  CASE_STRATEGY=slab-pencil; CASE_MODE=p2p-waitany; CASE_REQUIRED=0 ;;
        *) echo "Unknown case: ${CASE_ID}" >&2; return 2 ;;
    esac
}

SELECTED_CASES=()
MAX_GPUS=0
MAX_NODES=0
for case_id in "${ALL_CASES[@]}"; do
    if case_selected "${case_id}"; then
        SELECTED_CASES+=("${case_id}")
        case_properties "${case_id}"
        (( CASE_GPUS > MAX_GPUS )) && MAX_GPUS=${CASE_GPUS}
        (( CASE_NODES > MAX_NODES )) && MAX_NODES=${CASE_NODES}
    fi
done

if (( ${#SELECTED_CASES[@]} == 0 )); then
    echo "Target ${TARGET} selected no cases." >&2
    exit 2
fi

if ! is_true "${DRY_RUN}" && [[ ! -r "${FFTM_CONTAINER_IMAGE}" ]]; then
    echo "Missing FFTM container image: ${FFTM_CONTAINER_IMAGE}" >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" && -e "${ROOT}" &&
      -n "$(find "${ROOT}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory is not empty: ${ROOT}" >&2
    exit 2
fi
mkdir -p "${ROOT}"
ROOT=$(cd -- "${ROOT}" && pwd -P)

if ! is_true "${DRY_RUN}" && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    command -v salloc >/dev/null || { echo "salloc is required outside an allocation." >&2; exit 2; }
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_TEXT}"
    printf 'Requesting high-count paper allocation: target=%s nodes=%d GPUs=%d time=%s\n' \
        "${TARGET}" "${MAX_NODES}" "${MAX_GPUS}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_HIGH_COUNT_DATA_DIR="${ROOT}" \
        salloc \
        --job-name=fftm-high-paper \
        --nodes="${MAX_NODES}" \
        --ntasks="${MAX_GPUS}" \
        --ntasks-per-node="${GPUS_PER_NODE}" \
        --cpus-per-task=4 \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SELF}" "${TARGET}"
fi

if ! is_true "${DRY_RUN}" && (( ${SLURM_JOB_NUM_NODES:-0} < MAX_NODES )); then
    echo "Target ${TARGET} requires ${MAX_NODES} nodes; allocation has ${SLURM_JOB_NUM_NODES:-0}." >&2
    exit 2
fi

{
    printf 'target=%s\n' "${TARGET}"
    printf 'container_image=%s\n' "${FFTM_CONTAINER_IMAGE}"
    printf 'selected_cases=%s\n' "$(IFS=,; echo "${SELECTED_CASES[*]}")"
    printf 'max_nodes=%s\n' "${MAX_NODES}"
    printf 'max_gpus=%s\n' "${MAX_GPUS}"
    printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
    printf 'benchmark_times=%s\n' "${BENCHMARK_TIMES}"
    printf 'warmup=%s\n' "${WARMUP}"
    printf 'step_time=%s\n' "${STEP_TIME}"
    printf 'timeout_seconds=%s\n' "${TIMEOUT_SECONDS}"
    printf 'slab_backward_credit_policy=auto\n'
    printf 'slab_backward_credit_high_count_expected=disabled_unless_explicitly_calibrated\n'
    printf 'mpi_rank_affinity_mode=hca\n'
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-dry-run}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${ROOT}/config.env"

printf 'case_id,required,dimension,size,gpus,nodes,strategy,mode,grid,credit_policy\n' > "${ROOT}/matrix.csv"
for case_id in "${SELECTED_CASES[@]}"; do
    case_properties "${case_id}"
    size=$([[ "${CASE_DIM}" == 3 ]] && echo 2048x2048x2048 || echo 320x320x320x320)
    credit=$([[ "${CASE_DIM}" == 4 ]] && echo auto || echo n/a)
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${CASE_ID}" "${CASE_REQUIRED}" "${CASE_DIM}" "${size}" "${CASE_GPUS}" \
        "${CASE_NODES}" "${CASE_STRATEGY}" "${CASE_MODE}" "${CASE_GRID}" "${credit}" \
        >> "${ROOT}/matrix.csv"
done

if ! is_true "${DRY_RUN}"; then
    scontrol show job "${SLURM_JOB_ID}" > "${ROOT}/slurm_job.txt"
    scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${ROOT}/hosts.txt"
    git -C "${REPO_ROOT}" rev-parse HEAD > "${ROOT}/git_commit.txt" 2>/dev/null || true
    git -C "${REPO_ROOT}" status --short > "${ROOT}/git_status.txt" 2>/dev/null || true
    if [[ "${HASH_IMAGE}" == 1 ]]; then
        sha256sum "${FFTM_CONTAINER_IMAGE}" > "${ROOT}/image.sha256"
    fi
    sha256sum \
        "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" \
        "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh" \
        "${SCRIPT_DIR}/run_cluster_paper_benchmarks.py" \
        "${SELF}" \
        > "${ROOT}/launcher.sha256"
fi

AFFINITY_WRAPPER=${ROOT}/fftm_mpi_rank_affinity.sh
install -m 0755 "${SCRIPT_DIR}/run_mpi_rank_affinity.sh" "${AFFINITY_WRAPPER}"

preflight=(
    srun
    --nodes="${MAX_NODES}"
    --ntasks="${MAX_GPUS}"
    --ntasks-per-node="${GPUS_PER_NODE}"
    --cpus-per-task=4
    --gpus-per-node="${GPUS_PER_NODE}"
    --distribution=block:block
    --kill-on-bad-exit=1
    --time=00:15:00
    --container-image="${FFTM_CONTAINER_IMAGE}"
    --container-mounts="${ROOT}:/data"
    --container-workdir=/opt/fftm/bin
    --container-entrypoint
    /data/fftm_mpi_rank_affinity.sh
    --mode hca
    --
    /bin/bash -lc
    'set -e; test -x /opt/fftm/bin/test_benchmark_fftm_3D.bin; test -x /opt/fftm/bin/test_benchmark_fftm_4D.bin; if [[ "${SLURM_PROCID:-0}" == 0 && -r /opt/fftm/bin/fftm_build_info.txt ]]; then cat /opt/fftm/bin/fftm_build_info.txt; fi; printf "FFTM_HIGH_COUNT_PREFLIGHT host=%s rank=%s local_rank=%s visible=%s\n" "$(hostname)" "${SLURM_PROCID:-}" "${SLURM_LOCALID:-}" "${CUDA_VISIBLE_DEVICES:-}"'
)

if is_true "${DRY_RUN}"; then
    printf 'DRY RUN preflight:'
    printf ' %q' "${preflight[@]}"
    printf '\n'
else
    printf 'Running %d-rank image, GPU, and HCA preflight...\n' "${MAX_GPUS}"
    "${preflight[@]}" > "${ROOT}/transport_preflight.log" 2>&1
    hca_count=$(grep -c 'validation=active-sysfs+ucx' "${ROOT}/transport_preflight.log" || true)
    gpu_count=$(grep -c '^FFTM_HIGH_COUNT_PREFLIGHT ' "${ROOT}/transport_preflight.log" || true)
    if [[ "${hca_count}" -ne "${MAX_GPUS}" || "${gpu_count}" -ne "${MAX_GPUS}" ]]; then
        echo "Preflight expected ${MAX_GPUS} ranks; got HCA=${hca_count}, GPU=${gpu_count}." >&2
        exit 1
    fi
    if grep -Eq 'UCX[[:space:]]+(WARN|ERROR)|not visible in ucx_info|not active' \
        "${ROOT}/transport_preflight.log"; then
        echo "The transport preflight reported a degraded UCX/HCA configuration." >&2
        exit 1
    fi
fi

run_case()
{
    local case_id=$1
    case_properties "${case_id}"
    local case_dir=${ROOT}/${case_id}
    local strategies_3d= strategies_4d= grid_orientations=configured
    mkdir -p "${case_dir}"

    if [[ "${CASE_DIM}" == 3 ]]; then
        strategies_3d=${CASE_STRATEGY}
        grid_orientations=${CASE_GRID}
    else
        strategies_4d=${CASE_STRATEGY}
    fi

    env \
        FFTM_CONTAINER_IMAGE="${FFTM_CONTAINER_IMAGE}" \
        FFTM_DATA_DIR="${case_dir}" \
        FFTM_NODE_COUNTS="${CASE_NODES}" \
        FFTM_GPU_COUNTS="${CASE_GPUS}" \
        FFTM_GPUS_PER_NODE="${GPUS_PER_NODE}" \
        FFTM_DEVICE_MEMORY_MIB=81920 \
        FFTM_BENCHMARK_SIZES_3D=none \
        FFTM_BENCHMARK_SIZES_4D=none \
        FFTM_FIXED_SCALING_SIZES_3D="${CASE_SIZE_3D}" \
        FFTM_FIXED_SCALING_SIZES_4D="${CASE_SIZE_4D}" \
        FFTM_EXTRA_SIZES_3D= \
        FFTM_EXTRA_SIZES_3D_BY_GPU= \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_MODES="${CASE_MODE}" \
        FFTM_STRATEGIES_3D="${strategies_3d}" \
        FFTM_STRATEGIES_4D="${strategies_4d}" \
        FFTM_SKIP_FFTS=1 \
        FFTM_SKIP_FFTM=0 \
        FFTM_INCLUDE_VERSIONED=0 \
        FFTM_3D_BACKENDS=native \
        FFTM_PENCIL_PIPELINES=reference-parity \
        FFTM_PENCIL_LAYOUTS=opt0 \
        FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS="${grid_orientations}" \
        FFTM_P2P_VARIANTS=byte-packed \
        FFTM_P2P_SCHEDULERS=main \
        FFTM_LARGE_COUNT_P2P_TRANSPORTS=hindexed \
        FFTM_USE_DIRECT_BACKWARD_RECEIVE=0 \
        FFTM_DIRECT_P2P_CUDA_AWARE=1 \
        FFTM_USE_P2P_SEND_THREAD=0 \
        FFTM_USE_P2P_BYTE_TRANSFER=1 \
        FFTM_USE_PERSISTENT_P2P=0 \
        FFTM_USE_READY_P2P_SEND=0 \
        FFTM_USE_3D_DEFERRED_SEND_COMPLETION=0 \
        FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT=1 \
        FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY=1 \
        FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE=1 \
        FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES=1 \
        FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC=1 \
        FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC=1 \
        FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR=1 \
        FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS=0 \
        FFTM_USE_FFT_EXEC_NO_SYNC=0 \
        FFTM_4D_SLAB_XW_TRANSPOSES=native \
        FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS=off \
        FFTM_4D_SLAB_XW_KERNEL_LAYOUTS=buffer-contiguous \
        FFTM_4D_SLAB_XW_VECTOR4_KERNELS=off \
        FFTM_4D_SLAB_XW_TILED_KERNELS=off \
        FFTM_4D_SLAB_XW_LAYOUT_STAGES=off \
        FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS=native \
        FFTM_4D_NATIVE_XW_DIRECT_LAYOUTS=on \
        FFTM_4D_NATIVE_XW_PROTOCOLS=chunked \
        FFTM_4D_NATIVE_XW_CHUNK_MIB=512 \
        FFTM_4D_NATIVE_XW_CHUNK_WINDOWS=1 \
        FFTM_4D_NATIVE_XW_COMPACT_STAGINGS=on \
        FFTM_4D_SLAB_NATIVE_WORK_AREA_ALIASES=on \
        FFTM_4D_SLAB_NATIVE_WZ_COMM_LAYOUTS=on \
        FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES=4 \
        FFTM_4D_SLAB_NATIVE_WZ_READY_PIPELINES=on \
        FFTM_BINARY_EXTRA_ARGS= \
        FFTM_4D_BINARY_EXTRA_ARGS='--4d-slab-backward-credit-policy auto --4d-procs-per-node 8 --no-4d-slab-native-wz-send-overlap --no-4d-slab-native-wz-send-overlap-nonblocking-stream' \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_SRUN_EXTRA_ARGS='--distribution=block:block --kill-on-bad-exit=1' \
        FFTM_PRINT_PENCIL_SCHEDULE=0 \
        FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
        FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
        FFTM_ENABLE_GPU_TELEMETRY=1 \
        FFTM_VALIDATION_TIMES=1 \
        FFTM_BENCHMARK_TIMES="${BENCHMARK_TIMES}" \
        FFTM_WARMUP="${WARMUP}" \
        FFTM_RUN_PREFLIGHT=0 \
        FFTM_SRUN_TIME="${STEP_TIME}" \
        FFTM_TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
        FFTM_STOP_ON_FAILURE=0 \
        FFTM_DRY_RUN="${DRY_RUN}" \
        "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

validate_case()
{
    local case_id=$1
    case_properties "${case_id}"
    if is_true "${DRY_RUN}"; then
        return 0
    fi
    python3 - "${ROOT}/${case_id}" "${CASE_ID}" "${CASE_DIM}" "${CASE_GPUS}" \
        "${CASE_NODES}" "${CASE_STRATEGY}" "${CASE_MODE}" "${CASE_GRID}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_id, dimension, gpus, nodes, strategy, mode, grid = sys.argv[2:]
dimension = int(dimension)
gpus = int(gpus)
nodes = int(nodes)

summary_path = root / "summary.json"
if not summary_path.is_file():
    raise SystemExit(f"{case_id}: missing summary.json")
summary = json.loads(summary_path.read_text())
if int(summary.get("planned_runs", -1)) != 1:
    raise SystemExit(f"{case_id}: expected one planned run, got {summary.get('planned_runs')}")
if int(summary.get("failed_measurement_runs", -1)) != 0:
    raise SystemExit(f"{case_id}: failed measurement run is present")

csv_name = f"benchmark_fftm_{dimension}d.csv"
csv_path = root / "cpp_csv" / csv_name
if not csv_path.is_file():
    raise SystemExit(f"{case_id}: missing {csv_path}")
with csv_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
rows = [row for row in rows if int(row["num_gpus"]) == gpus]
if len(rows) != 1:
    raise SystemExit(f"{case_id}: expected one {gpus}-GPU row, got {len(rows)}")
row = rows[0]
if row["strategy"] != strategy or row["mode"] != mode:
    raise SystemExit(
        f"{case_id}: observed {row['strategy']}/{row['mode']}, expected {strategy}/{mode}"
    )
if float(row["max_l2_diff"]) > 1.0e-11:
    raise SystemExit(f"{case_id}: numerical error {row['max_l2_diff']}")

expected_sizes = (2048, 2048, 2048) if dimension == 3 else (320, 320, 320, 320)
size_fields = ("nx", "ny", "nz") if dimension == 3 else ("nx", "ny", "nz", "nw")
observed_sizes = tuple(int(row[name]) for name in size_fields)
if observed_sizes != expected_sizes:
    raise SystemExit(f"{case_id}: observed size {observed_sizes}, expected {expected_sizes}")

validation = {
    "case_id": case_id,
    "dimension": dimension,
    "gpus": gpus,
    "nodes": nodes,
    "strategy": strategy,
    "mode": mode,
    "avg_wall_ms": float(row["avg_wall_ms"]),
    "stddev_wall_ms": float(row["stddev_wall_ms"]),
    "max_l2_diff": float(row["max_l2_diff"]),
    "status": "passed",
}

if dimension == 3 and strategy == "pencil-pencil":
    expected_p1, expected_p2 = (int(value) for value in grid.split("x"))
    observed_grid = (int(row["p1"]), int(row["p2"]))
    if observed_grid != (expected_p1, expected_p2):
        raise SystemExit(f"{case_id}: observed grid {observed_grid}, expected {(expected_p1, expected_p2)}")
    if row["pencil_layout"] != "opt0" or row["pencil_pipeline"] != "reference-parity":
        raise SystemExit(f"{case_id}: unexpected pencil layout/pipeline")
    validation["grid"] = grid
    validation["pencil_layout"] = row["pencil_layout"]
    validation["pencil_pipeline"] = row["pencil_pipeline"]

if dimension == 4:
    required_flags = {
        "slab_native_xw": 1,
        "native_xw_direct_layout": 1,
        "native_xw_chunked_transport": 1,
        "native_xw_compact_staging": 1,
        "slab_native_work_area_alias_effective": 1,
        "slab_native_wz_communication_layout": 1,
        "slab_native_wz_ready_pipeline": 1,
        "slab_native_xw_native_spectral_layout": 1,
    }
    for field, expected in required_flags.items():
        if int(row[field]) != expected:
            raise SystemExit(f"{case_id}: {field}={row[field]}, expected {expected}")
    if int(row["slab_native_wz_plan_concurrency"]) != 4:
        raise SystemExit(f"{case_id}: WZ concurrency is not 4")
    credit_window = int(row["slab_native_wz_backward_plane_credit_window"])
    cyclic = int(row["slab_native_wz_backward_cyclic_peer_order"])
    if (credit_window, cyclic) != (0, 0):
        raise SystemExit(
            f"{case_id}: uncalibrated high-count auto policy unexpectedly selected "
            f"credit window/cyclic {(credit_window, cyclic)}"
        )
    validation["slab_backward_credit_policy"] = "auto"
    validation["slab_backward_credit_window"] = credit_window
    validation["slab_backward_cyclic_peer_order"] = cyclic

(root / "case_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
print(
    f"Validated {case_id}: {validation['avg_wall_ms']:.3f} ms, "
    f"L2={validation['max_l2_diff']:.3e}"
)
PY
}

printf 'case_id,required,run_rc,validation_rc,status\n' > "${ROOT}/status.csv"
overall=0

for case_id in "${SELECTED_CASES[@]}"; do
    case_properties "${case_id}"
    run_rc=0
    validation_rc=0
    status=passed
    printf '[%s] dimension=%s size=%s GPUs=%s nodes=%s strategy=%s mode=%s grid=%s\n' \
        "${CASE_ID}" "${CASE_DIM}" \
        "$([[ "${CASE_DIM}" == 3 ]] && echo 2048x2048x2048 || echo 320x320x320x320)" \
        "${CASE_GPUS}" "${CASE_NODES}" "${CASE_STRATEGY}" "${CASE_MODE}" "${CASE_GRID}"
    run_case "${case_id}" || run_rc=$?
    validate_case "${case_id}" || validation_rc=$?
    if (( run_rc != 0 || validation_rc != 0 )); then
        status=failed
        if [[ "${CASE_REQUIRED}" == 1 ]]; then
            overall=1
        fi
    fi
    printf '%s,%s,%s,%s,%s\n' \
        "${CASE_ID}" "${CASE_REQUIRED}" "${run_rc}" "${validation_rc}" "${status}" \
        >> "${ROOT}/status.csv"
done

if ! is_true "${DRY_RUN}"; then
    python3 - "${ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
matrix = {row["case_id"]: row for row in csv.DictReader((root / "matrix.csv").open(newline=""))}
status = {row["case_id"]: row for row in csv.DictReader((root / "status.csv").open(newline=""))}
fields = [
    "case_id", "required", "status", "dimension", "size", "gpus", "nodes",
    "strategy", "mode", "grid", "avg_wall_ms", "stddev_wall_ms", "max_l2_diff",
    "slab_backward_credit_window", "slab_backward_cyclic_peer_order",
]
with (root / "selected_results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for case_id, metadata in matrix.items():
        output = {name: "" for name in fields}
        output.update({name: metadata.get(name, "") for name in fields})
        output["status"] = status.get(case_id, {}).get("status", "missing")
        validation_path = root / case_id / "case_validation.json"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text())
            for name in ("avg_wall_ms", "stddev_wall_ms", "max_l2_diff"):
                output[name] = validation.get(name, "")
            output["slab_backward_credit_window"] = validation.get(
                "slab_backward_credit_window", ""
            )
            output["slab_backward_cyclic_peer_order"] = validation.get(
                "slab_backward_cyclic_peer_order", ""
            )
        writer.writerow(output)
print(f"Wrote {root / 'selected_results.csv'}")
PY
fi

echo "High-count paper matrix: ${ROOT}"
echo "Copy the complete directory back, including each case's summary.json and cpp_csv directory."
exit "${overall}"
