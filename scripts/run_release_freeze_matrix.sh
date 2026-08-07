#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_release_freeze_matrix.sh TARGET

Release-freeze targets (queue each independently):
  fitted       1 node, 2-8 GPUs, fitted 3D sizes, three strategies (21 cases)
  scale-32     4 nodes, 32 GPUs, 2048^3 and 320^4 strong scaling (2 cases)
  scale-64     8 nodes, 64 GPUs, 2048^3 and 320^4 strong scaling (2 cases)
  scale-96    12 nodes, 96 GPUs, 2048^3 and 320^4 strong scaling (2 cases)
  scale-120   15 nodes, 120 GPUs, 2048^3 and 320^4 strong scaling (2 cases)
  host-staged  2 nodes, 16 GPUs, non-CUDA-aware 3D/4D smoke (2 cases)
  manifest     1 node, hardware/software/topology manifest only

Common overrides:
  FFTM_RELEASE_CONTAINER_IMAGE
  FFTM_RELEASE_DATA_DIR
  FFTM_RELEASE_EXPECTED_GIT_COMMIT
  FFTM_RELEASE_EXPECTED_IMAGE_SHA256
  FFTM_RELEASE_SALLOC_EXTRA_ARGS       (default: --exclude=cn13)
  FFTM_RELEASE_ALLOCATION_TIME
  FFTM_RELEASE_HASH_IMAGE=0|1
  FFTM_RELEASE_DRY_RUN=0|1

Measurement overrides:
  FFTM_RELEASE_FITTED_TIMES            (default: 50)
  FFTM_RELEASE_FITTED_WARMUP           (default: 10)
  FFTM_RELEASE_SCALE_TIMES             (default: 30)
  FFTM_RELEASE_SCALE_WARMUP            (default: 5)
  FFTM_RELEASE_HOST_STAGED_TIMES       (default: 5)
  FFTM_RELEASE_HOST_STAGED_WARMUP      (default: 3)
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

TARGET=$1
case "${TARGET}" in
    fitted|scale-32|scale-64|scale-96|scale-120|host-staged|manifest)
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown release-freeze target: ${TARGET}" >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
SELF=${SCRIPT_DIR}/run_release_freeze_matrix.sh

IMAGE=${FFTM_RELEASE_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}
EXPECTED_COMMIT=${FFTM_RELEASE_EXPECTED_GIT_COMMIT:-e06f75b4b728e02d79f7e2193e616059058e6e12}
EXPECTED_IMAGE_SHA256=${FFTM_RELEASE_EXPECTED_IMAGE_SHA256:-eebd94b1e71682a69dcc7b8a65e478c7dc7690ab2b04d0abb217a9dc8aea63cf}
STAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR=${FFTM_RELEASE_DATA_DIR:-/scratch/evstigneevnm/fftm/data_release_${TARGET}_${STAMP}}
SALLOC_EXTRA_ARGS=${FFTM_RELEASE_SALLOC_EXTRA_ARGS:---exclude=cn13}
DRY_RUN=${FFTM_RELEASE_DRY_RUN:-0}
HASH_IMAGE=${FFTM_RELEASE_HASH_IMAGE:-1}

FITTED_TIMES=${FFTM_RELEASE_FITTED_TIMES:-50}
FITTED_WARMUP=${FFTM_RELEASE_FITTED_WARMUP:-10}
SCALE_TIMES=${FFTM_RELEASE_SCALE_TIMES:-30}
SCALE_WARMUP=${FFTM_RELEASE_SCALE_WARMUP:-5}
HOST_STAGED_TIMES=${FFTM_RELEASE_HOST_STAGED_TIMES:-5}
HOST_STAGED_WARMUP=${FFTM_RELEASE_HOST_STAGED_WARMUP:-3}

case "${DRY_RUN}" in
    0|1) ;;
    *) echo "FFTM_RELEASE_DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac
case "${HASH_IMAGE}" in
    0|1) ;;
    *) echo "FFTM_RELEASE_HASH_IMAGE must be 0 or 1" >&2; exit 2 ;;
esac
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FFTM_RELEASE_EXPECTED_GIT_COMMIT must be a 40-character lowercase Git SHA" >&2
    exit 2
fi
if [[ ! "${EXPECTED_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FFTM_RELEASE_EXPECTED_IMAGE_SHA256 must be a 64-character lowercase SHA-256" >&2
    exit 2
fi

GPUS_PER_NODE=8
CPUS_PER_TASK=4
EXPECTED_CASES=0
case "${TARGET}" in
    fitted)
        NODES=1
        TASKS=8
        ALLOCATION_TIME_DEFAULT=04:00:00
        EXPECTED_CASES=21
        ;;
    scale-32)
        NODES=4
        TASKS=32
        GRID_3D=8x4
        ALLOCATION_TIME_DEFAULT=03:00:00
        EXPECTED_CASES=2
        ;;
    scale-64)
        NODES=8
        TASKS=64
        GRID_3D=8x8
        ALLOCATION_TIME_DEFAULT=03:00:00
        EXPECTED_CASES=2
        ;;
    scale-96)
        NODES=12
        TASKS=96
        GRID_3D=12x8
        ALLOCATION_TIME_DEFAULT=03:00:00
        EXPECTED_CASES=2
        ;;
    scale-120)
        NODES=15
        TASKS=120
        GRID_3D=15x8
        ALLOCATION_TIME_DEFAULT=03:00:00
        EXPECTED_CASES=2
        ;;
    host-staged)
        NODES=2
        TASKS=16
        ALLOCATION_TIME_DEFAULT=02:00:00
        EXPECTED_CASES=2
        ;;
    manifest)
        NODES=1
        TASKS=1
        CPUS_PER_TASK=1
        ALLOCATION_TIME_DEFAULT=00:20:00
        ;;
esac
ALLOCATION_TIME=${FFTM_RELEASE_ALLOCATION_TIME:-${ALLOCATION_TIME_DEFAULT}}

DATA_DIR=$(realpath -m -- "${DATA_DIR}")

if [[ "${DRY_RUN}" == 0 && ! -f "${IMAGE}" ]]; then
    echo "Missing SQSH image: ${IMAGE}" >&2
    exit 2
fi

if [[ "${DRY_RUN}" == 0 && -z "${SLURM_JOB_ID:-}" ]]; then
    command -v salloc >/dev/null || {
        echo "salloc is required outside an existing allocation" >&2
        exit 2
    }
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting release-freeze allocation: target=%s nodes=%s tasks=%s GPUs/node=%s time=%s\n' \
        "${TARGET}" "${NODES}" "${TASKS}" "${GPUS_PER_NODE}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_RELEASE_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_RELEASE_DATA_DIR="${DATA_DIR}" \
        FFTM_RELEASE_EXPECTED_GIT_COMMIT="${EXPECTED_COMMIT}" \
        FFTM_RELEASE_EXPECTED_IMAGE_SHA256="${EXPECTED_IMAGE_SHA256}" \
        FFTM_RELEASE_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}" \
        FFTM_RELEASE_ALLOCATION_TIME="${ALLOCATION_TIME}" \
        FFTM_RELEASE_HASH_IMAGE="${HASH_IMAGE}" \
        FFTM_RELEASE_DRY_RUN=0 \
        salloc \
        --job-name="release-${TARGET}" \
        --nodes="${NODES}" \
        --ntasks="${TASKS}" \
        --ntasks-per-node="$(( (TASKS + NODES - 1) / NODES ))" \
        --cpus-per-task="${CPUS_PER_TASK}" \
        --gpus-per-node="${GPUS_PER_NODE}" \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SELF}" "${TARGET}"
fi

if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${SLURM_JOB_NUM_NODES}" -ne "${NODES}" ]]; then
    echo "Target ${TARGET} requires exactly ${NODES} allocated node(s)" >&2
    exit 2
fi

if [[ -e "${DATA_DIR}" && -n "$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory is not empty: ${DATA_DIR}" >&2
    echo "Use a new FFTM_RELEASE_DATA_DIR for each attempt." >&2
    exit 2
fi
mkdir -p "${DATA_DIR}"

{
    printf 'target=%s\n' "${TARGET}"
    printf 'expected_cases=%s\n' "${EXPECTED_CASES}"
    printf 'expected_git_commit=%s\n' "${EXPECTED_COMMIT}"
    printf 'expected_image_sha256=%s\n' "${EXPECTED_IMAGE_SHA256}"
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'tasks=%s\n' "${TASKS}"
    printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
    printf 'allocation_time=%s\n' "${ALLOCATION_TIME}"
    printf 'salloc_extra_args=%s\n' "${SALLOC_EXTRA_ARGS}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-dry-run}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DATA_DIR}/config.env"

if [[ "${DRY_RUN}" == 0 && "${HASH_IMAGE}" == 1 ]]; then
    sha256sum "${IMAGE}" > "${DATA_DIR}/image.sha256"
    actual_image_sha256=$(cut -d' ' -f1 "${DATA_DIR}/image.sha256")
    if [[ "${actual_image_sha256}" != "${EXPECTED_IMAGE_SHA256}" ]]; then
        echo "SQSH SHA-256 does not match the release image" >&2
        echo "expected: ${EXPECTED_IMAGE_SHA256}" >&2
        echo "actual:   ${actual_image_sha256}" >&2
        exit 2
    fi
fi

print_command()
{
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
}

run_image_preflight()
{
    local command=(
        srun -N 1 -n 1 -G 1
        --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=1
        --exclusive --kill-on-bad-exit=1 --time=00:05:00
        --container-image "${IMAGE}"
        --container-workdir /opt/fftm/bin
        --container-entrypoint /bin/bash
        -lc
        "cat /opt/fftm/bin/fftm_build_info.txt; test -x /opt/fftm/bin/test_benchmark_fftm_3D.bin; test -x /opt/fftm/bin/test_benchmark_fftm_4D.bin"
    )
    print_command "${command[@]}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        return 0
    fi
    "${command[@]}" 2>&1 | tee "${DATA_DIR}/image_preflight.log"
    grep -Fx "git_commit=${EXPECTED_COMMIT}" "${DATA_DIR}/image_preflight.log" >/dev/null || {
        echo "SQSH Git commit does not match ${EXPECTED_COMMIT}" >&2
        return 1
    }
    grep -Fx 'git_dirty=0' "${DATA_DIR}/image_preflight.log" >/dev/null || {
        echo "SQSH was built from a dirty tracked worktree" >&2
        return 1
    }
}

collect_manifest()
{
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf 'Manifest: dry-run (compute-node and container probes skipped)\n'
        return 0
    fi

    local host_probe
    host_probe='set -u; date -u +%Y-%m-%dT%H:%M:%SZ; hostname -f || hostname; uname -a; lscpu; free -h; nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total,driver_version --format=csv,noheader; nvidia-smi topo -m; command -v numactl >/dev/null && numactl --hardware || true; command -v ibstat >/dev/null && ibstat || true; command -v ibdev2netdev >/dev/null && ibdev2netdev || true; command -v ucx_info >/dev/null && ucx_info -v || true'
    srun -N 1 -n 1 -G 8 \
        --ntasks-per-node=1 --gpus-per-node=8 --cpus-per-task=1 \
        --exclusive --kill-on-bad-exit=1 --time=00:05:00 \
        /bin/bash -lc "${host_probe}" \
        > "${DATA_DIR}/compute_node_manifest.txt" 2>&1

    local container_probe
    container_probe='set -u; cat /opt/fftm/bin/fftm_build_info.txt; nvidia-smi; command -v nvcc >/dev/null && nvcc --version || true; command -v mpirun >/dev/null && mpirun --version || true; command -v ompi_info >/dev/null && ompi_info --version || true; command -v ompi_info >/dev/null && ompi_info --all | grep -Ei "cuda|ucx|rocm" || true; command -v ucx_info >/dev/null && ucx_info -v || true; command -v ucx_info >/dev/null && ucx_info -d || true; ldd /opt/fftm/bin/test_benchmark_fftm_3D.bin'
    srun -N 1 -n 1 -G 1 \
        --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=1 \
        --exclusive --kill-on-bad-exit=1 --time=00:05:00 \
        --container-image "${IMAGE}" \
        --container-workdir /opt/fftm/bin \
        --container-entrypoint /bin/bash \
        -lc "${container_probe}" \
        > "${DATA_DIR}/container_manifest.txt" 2>&1

    scontrol show job "${SLURM_JOB_ID}" > "${DATA_DIR}/slurm_job.txt"
    scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${DATA_DIR}/hosts.txt"
}

common_production_options()
{
    env \
        FFTM_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_DATA_DIR="${DATA_DIR}/benchmarks" \
        FFTM_GPUS_PER_NODE=8 \
        FFTM_DEVICE_MEMORY_MIB=81920 \
        FFTM_BENCHMARK_SIZES_3D=none \
        FFTM_BENCHMARK_SIZES_4D=none \
        FFTM_EXTRA_SIZES_3D= \
        FFTM_EXTRA_SIZES_3D_BY_GPU= \
        FFTM_SKIP_FFTS=1 \
        FFTM_SKIP_FFTM=0 \
        FFTM_INCLUDE_VERSIONED=0 \
        FFTM_3D_BACKENDS=native \
        FFTM_P2P_VARIANTS=byte-packed \
        FFTM_P2P_SCHEDULERS=main \
        FFTM_LARGE_COUNT_P2P_TRANSPORTS=hindexed \
        FFTM_USE_DIRECT_BACKWARD_RECEIVE=0 \
        FFTM_USE_P2P_SEND_THREAD=0 \
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
        FFTM_PRINT_PENCIL_SCHEDULE=0 \
        FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
        FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
        FFTM_ENABLE_GPU_TELEMETRY=1 \
        FFTM_VALIDATION_TIMES=1 \
        FFTM_RUN_PREFLIGHT=0 \
        FFTM_DRY_RUN="${DRY_RUN}" \
        "$@"
}

run_fitted()
{
    common_production_options \
        FFTM_NODE_COUNTS=1 \
        FFTM_GPU_COUNTS=2,3,4,5,6,7,8 \
        FFTM_FIXED_SCALING_SIZES_3D= \
        FFTM_FIXED_SCALING_SIZES_4D= \
        'FFTM_EXTRA_SIZES_3D_BY_GPU=2:1344;3:1536;4:1680;5:1800;6:1920;7:2025;8:2100' \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_MODES=p2p-waitany \
        FFTM_STRATEGIES_3D=slab-pencil,pencil-slab,pencil-pencil \
        FFTM_STRATEGIES_4D= \
        FFTM_PENCIL_PIPELINES=reference-parity \
        FFTM_PENCIL_LAYOUTS=auto \
        FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS=production \
        FFTM_DIRECT_P2P_CUDA_AWARE=1 \
        FFTM_USE_P2P_BYTE_TRANSFER=1 \
        FFTM_MPI_RANK_AFFINITY_MODE=auto \
        FFTM_BENCHMARK_TIMES="${FITTED_TIMES}" \
        FFTM_WARMUP="${FITTED_WARMUP}" \
        FFTM_SRUN_EXTRA_ARGS='--distribution=block:block --kill-on-bad-exit=1' \
        FFTM_SRUN_TIME=01:30:00 \
        FFTM_TIMEOUT_SECONDS=5400 \
        FFTM_STOP_ON_FAILURE=0 \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

run_scale()
{
    common_production_options \
        FFTM_NODE_COUNTS="${NODES}" \
        FFTM_GPU_COUNTS="${TASKS}" \
        FFTM_FIXED_SCALING_SIZES_3D=2048 \
        FFTM_FIXED_SCALING_SIZES_4D=320 \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_MODES=p2p-waitany \
        FFTM_STRATEGIES_3D=pencil-pencil \
        FFTM_STRATEGIES_4D=slab-slab \
        FFTM_PENCIL_PIPELINES=reference-parity \
        FFTM_PENCIL_LAYOUTS=opt0 \
        FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS="${GRID_3D}" \
        FFTM_DIRECT_P2P_CUDA_AWARE=1 \
        FFTM_USE_P2P_BYTE_TRANSFER=1 \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_BENCHMARK_TIMES="${SCALE_TIMES}" \
        FFTM_WARMUP="${SCALE_WARMUP}" \
        FFTM_SRUN_EXTRA_ARGS='--ntasks-per-node=8 --distribution=block:block --kill-on-bad-exit=1' \
        FFTM_SRUN_TIME=02:00:00 \
        FFTM_TIMEOUT_SECONDS=7200 \
        FFTM_STOP_ON_FAILURE=0 \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

run_host_staged()
{
    common_production_options \
        FFTM_NODE_COUNTS=2 \
        FFTM_GPU_COUNTS=16 \
        FFTM_FIXED_SCALING_SIZES_3D=2048 \
        FFTM_FIXED_SCALING_SIZES_4D=320 \
        FFTM_TRANSPORTS=non_cuda_aware \
        FFTM_MODES=p2p-waitany \
        FFTM_STRATEGIES_3D=pencil-pencil \
        FFTM_STRATEGIES_4D=slab-slab \
        FFTM_PENCIL_PIPELINES=reference-parity \
        FFTM_PENCIL_LAYOUTS=opt0 \
        FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS=4x4 \
        FFTM_DIRECT_P2P_CUDA_AWARE=0 \
        FFTM_USE_P2P_BYTE_TRANSFER=1 \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_BENCHMARK_TIMES="${HOST_STAGED_TIMES}" \
        FFTM_WARMUP="${HOST_STAGED_WARMUP}" \
        FFTM_SRUN_EXTRA_ARGS='--ntasks-per-node=8 --distribution=block:block --kill-on-bad-exit=1' \
        FFTM_SRUN_TIME=01:30:00 \
        FFTM_TIMEOUT_SECONDS=5400 \
        FFTM_STOP_ON_FAILURE=0 \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

printf 'Release-freeze target: %s; nodes=%s; tasks=%s; expected cases=%s; output=%s\n' \
    "${TARGET}" "${NODES}" "${TASKS}" "${EXPECTED_CASES}" "${DATA_DIR}"
run_image_preflight
collect_manifest

if [[ "${TARGET}" == manifest ]]; then
    printf 'Manifest complete: %s\n' "${DATA_DIR}"
    exit 0
fi

mkdir -p "${DATA_DIR}/benchmarks"
case "${TARGET}" in
    fitted) run_fitted ;;
    scale-*) run_scale ;;
    host-staged) run_host_staged ;;
esac

test -s "${DATA_DIR}/benchmarks/runs.jsonl" || {
    echo "Missing benchmark run ledger: ${DATA_DIR}/benchmarks/runs.jsonl" >&2
    exit 1
}
actual_cases=$(wc -l < "${DATA_DIR}/benchmarks/runs.jsonl")
printf 'expected_cases=%s\nactual_cases=%s\n' \
    "${EXPECTED_CASES}" "${actual_cases}" > "${DATA_DIR}/case_count.env"
if [[ "${actual_cases}" -ne "${EXPECTED_CASES}" ]]; then
    echo "Expected ${EXPECTED_CASES} run records, found ${actual_cases}" >&2
    exit 1
fi

printf 'Release-freeze target complete: %s\n' "${DATA_DIR}"
