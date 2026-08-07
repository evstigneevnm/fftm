#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: scripts/run_4d_weak_scaling_endpoints.sh 96|120

Runs the missing 4D weak-scaling endpoint on one reusable Slurm allocation:
  96 GPUs / 12 nodes: 600^4
 120 GPUs / 15 nodes: 630^4

Each candidate first runs a short qualification case. A full 30-attempt case is
run only when qualification succeeds. Candidate failures are recorded and do
not stop the remaining matrix.

Defaults preserve the validated slab/native-spectral production path while
reducing the compact-staging chunk size that caused the previous OOM:
  WZ concurrency : chunk MiB = 4:256,4:128,4:64,2:128

Useful overrides:
  FFTM_WEAK4D_CONTAINER_IMAGE
  FFTM_WEAK4D_DATA_DIR
  FFTM_WEAK4D_CANDIDATES             comma-separated CONCURRENCY:CHUNK_MIB
  FFTM_WEAK4D_BENCHMARK_TIMES        default: 30
  FFTM_WEAK4D_WARMUP                 default: 5
  FFTM_WEAK4D_QUALIFY_TIMES          default: 1
  FFTM_WEAK4D_QUALIFY_WARMUP         default: 0
  FFTM_WEAK4D_QUALIFY_TIMEOUT        default: 1800 seconds
  FFTM_WEAK4D_FULL_TIMEOUT           default: 7200 seconds
  FFTM_WEAK4D_ALLOCATION_TIME        default: 10:00:00
  FFTM_WEAK4D_SALLOC_EXTRA_ARGS      default: --exclude=cn13
  FFTM_WEAK4D_DRY_RUN=0|1
EOF
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

TARGET=$1
case "${TARGET}" in
    96)
        NODES=12
        GPUS=96
        SIZE=600
        WEAK_RATIO=1.029968262
        ;;
    120)
        NODES=15
        GPUS=120
        SIZE=630
        WEAK_RATIO=1.001546288
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Target must be 96 or 120." >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd -P)
SELF=${SCRIPT_DIR}/run_4d_weak_scaling_endpoints.sh
cd "${REPO_ROOT}"

IMAGE=${FFTM_WEAK4D_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}
EXPECTED_COMMIT=${FFTM_WEAK4D_EXPECTED_GIT_COMMIT:-e06f75b4b728e02d79f7e2193e616059058e6e12}
EXPECTED_IMAGE_SHA256=${FFTM_WEAK4D_EXPECTED_IMAGE_SHA256:-eebd94b1e71682a69dcc7b8a65e478c7dc7690ab2b04d0abb217a9dc8aea63cf}
STAMP=$(date +%Y%m%d_%H%M%S)
DATA_DIR=${FFTM_WEAK4D_DATA_DIR:-/scratch/evstigneevnm/fftm/data_4d_weak_${TARGET}_${STAMP}}
CANDIDATES=${FFTM_WEAK4D_CANDIDATES:-4:256,4:128,4:64,2:128}
BENCHMARK_TIMES=${FFTM_WEAK4D_BENCHMARK_TIMES:-30}
WARMUP=${FFTM_WEAK4D_WARMUP:-5}
QUALIFY_TIMES=${FFTM_WEAK4D_QUALIFY_TIMES:-1}
QUALIFY_WARMUP=${FFTM_WEAK4D_QUALIFY_WARMUP:-0}
QUALIFY_TIMEOUT=${FFTM_WEAK4D_QUALIFY_TIMEOUT:-1800}
FULL_TIMEOUT=${FFTM_WEAK4D_FULL_TIMEOUT:-7200}
QUALIFY_SRUN_TIME=${FFTM_WEAK4D_QUALIFY_SRUN_TIME:-00:35:00}
FULL_SRUN_TIME=${FFTM_WEAK4D_FULL_SRUN_TIME:-02:05:00}
ALLOCATION_TIME=${FFTM_WEAK4D_ALLOCATION_TIME:-10:00:00}
SALLOC_EXTRA_ARGS=${FFTM_WEAK4D_SALLOC_EXTRA_ARGS:---exclude=cn13}
DRY_RUN=${FFTM_WEAK4D_DRY_RUN:-0}
IMAGE_VERIFIED=${FFTM_WEAK4D_IMAGE_VERIFIED:-0}

case "${DRY_RUN}" in
    0|1) ;;
    *) echo "FFTM_WEAK4D_DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac
if [[ ! "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FFTM_WEAK4D_EXPECTED_GIT_COMMIT must be a 40-character Git SHA." >&2
    exit 2
fi
if [[ ! "${EXPECTED_IMAGE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "FFTM_WEAK4D_EXPECTED_IMAGE_SHA256 must be a SHA-256 digest." >&2
    exit 2
fi

IFS=',' read -r -a candidate_list <<< "${CANDIDATES}"
if [[ ${#candidate_list[@]} -eq 0 ]]; then
    echo "At least one candidate is required." >&2
    exit 2
fi
for candidate in "${candidate_list[@]}"; do
    if [[ ! "${candidate}" =~ ^[1-9][0-9]*:[1-9][0-9]*$ ]]; then
        echo "Invalid candidate '${candidate}'; expected CONCURRENCY:CHUNK_MIB." >&2
        exit 2
    fi
done

DATA_DIR=$(realpath -m -- "${DATA_DIR}")
if [[ "${DRY_RUN}" == 0 ]]; then
    test -f "${IMAGE}" || {
        echo "Missing SQSH image: ${IMAGE}" >&2
        exit 2
    }
    if [[ "${IMAGE_VERIFIED}" == 0 ]]; then
        actual_image_sha256=$(sha256sum "${IMAGE}" | awk '{ print $1 }')
        if [[ "${actual_image_sha256}" != "${EXPECTED_IMAGE_SHA256}" ]]; then
            echo "SQSH SHA-256 mismatch." >&2
            echo "expected: ${EXPECTED_IMAGE_SHA256}" >&2
            echo "actual:   ${actual_image_sha256}" >&2
            exit 2
        fi
        IMAGE_VERIFIED=1
    fi
fi

if [[ "${DRY_RUN}" == 0 && -z "${SLURM_JOB_ID:-}" ]]; then
    command -v salloc >/dev/null || {
        echo "salloc is required outside an existing allocation." >&2
        exit 2
    }
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting 4D weak-scaling allocation: GPUs=%s nodes=%s size=%s^4 time=%s\n' \
        "${GPUS}" "${NODES}" "${SIZE}" "${ALLOCATION_TIME}"
    exec env \
        FFTM_WEAK4D_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_WEAK4D_DATA_DIR="${DATA_DIR}" \
        FFTM_WEAK4D_CANDIDATES="${CANDIDATES}" \
        FFTM_WEAK4D_BENCHMARK_TIMES="${BENCHMARK_TIMES}" \
        FFTM_WEAK4D_WARMUP="${WARMUP}" \
        FFTM_WEAK4D_QUALIFY_TIMES="${QUALIFY_TIMES}" \
        FFTM_WEAK4D_QUALIFY_WARMUP="${QUALIFY_WARMUP}" \
        FFTM_WEAK4D_QUALIFY_TIMEOUT="${QUALIFY_TIMEOUT}" \
        FFTM_WEAK4D_FULL_TIMEOUT="${FULL_TIMEOUT}" \
        FFTM_WEAK4D_QUALIFY_SRUN_TIME="${QUALIFY_SRUN_TIME}" \
        FFTM_WEAK4D_FULL_SRUN_TIME="${FULL_SRUN_TIME}" \
        FFTM_WEAK4D_ALLOCATION_TIME="${ALLOCATION_TIME}" \
        FFTM_WEAK4D_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}" \
        FFTM_WEAK4D_EXPECTED_GIT_COMMIT="${EXPECTED_COMMIT}" \
        FFTM_WEAK4D_EXPECTED_IMAGE_SHA256="${EXPECTED_IMAGE_SHA256}" \
        FFTM_WEAK4D_IMAGE_VERIFIED="${IMAGE_VERIFIED}" \
        salloc \
        --job-name="weak4d-${TARGET}" \
        --nodes="${NODES}" \
        --ntasks="${GPUS}" \
        --ntasks-per-node=8 \
        --cpus-per-task=4 \
        --gpus-per-node=8 \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SELF}" "${TARGET}"
fi

if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${SLURM_JOB_NUM_NODES}" -ne "${NODES}" ]]; then
    echo "Target ${TARGET} requires exactly ${NODES} allocated nodes." >&2
    exit 2
fi
if [[ -e "${DATA_DIR}" && -n "$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory is not empty: ${DATA_DIR}" >&2
    exit 2
fi
mkdir -p "${DATA_DIR}"

if [[ "${DRY_RUN}" == 0 ]]; then
    scontrol show hostnames "${SLURM_JOB_NODELIST}" > "${DATA_DIR}/hosts.txt"
    if grep -Fxq cn13 "${DATA_DIR}/hosts.txt"; then
        echo "Refusing to run on excluded node cn13." >&2
        exit 2
    fi
    scontrol show job "${SLURM_JOB_ID}" > "${DATA_DIR}/slurm_job.txt"
    printf '%s  %s\n' "${EXPECTED_IMAGE_SHA256}" "${IMAGE}" > "${DATA_DIR}/image.sha256"
fi

{
    printf 'target_gpus=%s\n' "${GPUS}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'size_4d=%s\n' "${SIZE}"
    printf 'local_points_ratio_to_8g_320=%s\n' "${WEAK_RATIO}"
    printf 'candidates=%s\n' "${CANDIDATES}"
    printf 'benchmark_times=%s\n' "${BENCHMARK_TIMES}"
    printf 'warmup=%s\n' "${WARMUP}"
    printf 'expected_git_commit=%s\n' "${EXPECTED_COMMIT}"
    printf 'expected_image_sha256=%s\n' "${EXPECTED_IMAGE_SHA256}"
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${DATA_DIR}/config.env"

run_image_preflight()
{
    if [[ "${DRY_RUN}" == 1 ]]; then
        echo "Image preflight: dry-run"
        return 0
    fi
    srun -N 1 -n 1 -G 1 \
        --ntasks-per-node=1 --gpus-per-node=1 --cpus-per-task=1 \
        --exclusive --kill-on-bad-exit=1 --time=00:05:00 \
        --container-image "${IMAGE}" \
        --container-workdir /opt/fftm/bin \
        --container-entrypoint /bin/bash \
        -lc 'cat /opt/fftm/bin/fftm_build_info.txt; test -x /opt/fftm/bin/test_benchmark_fftm_4D.bin' \
        2>&1 | tee "${DATA_DIR}/image_preflight.log"
    grep -Fx "git_commit=${EXPECTED_COMMIT}" "${DATA_DIR}/image_preflight.log" >/dev/null
    grep -Fx 'git_dirty=0' "${DATA_DIR}/image_preflight.log" >/dev/null
}

run_case()
{
    local phase=$1
    local concurrency=$2
    local chunk_mib=$3
    local times=$4
    local warmup=$5
    local timeout_seconds=$6
    local srun_time=$7
    local case_dir=${DATA_DIR}/${phase}/c${concurrency}_m${chunk_mib}

    printf '[%s] size=%s^4 GPUs=%s WZ concurrency=%s chunk=%s MiB\n' \
        "${phase}" "${SIZE}" "${GPUS}" "${concurrency}" "${chunk_mib}"

    env \
        FFTM_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_DATA_DIR="${case_dir}" \
        FFTM_NODE_COUNTS="${NODES}" \
        FFTM_GPU_COUNTS="${GPUS}" \
        FFTM_GPUS_PER_NODE=8 \
        FFTM_DEVICE_MEMORY_MIB=81920 \
        FFTM_BENCHMARK_SIZES_3D=none \
        FFTM_BENCHMARK_SIZES_4D=none \
        FFTM_FIXED_SCALING_SIZES_3D= \
        FFTM_FIXED_SCALING_SIZES_4D="${SIZE}" \
        FFTM_EXTRA_SIZES_3D= \
        FFTM_EXTRA_SIZES_3D_BY_GPU= \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_MODES=p2p-waitany \
        FFTM_STRATEGIES_3D=pencil-pencil \
        FFTM_STRATEGIES_4D=slab-slab \
        FFTM_SKIP_FFTS=1 \
        FFTM_SKIP_FFTM=0 \
        FFTM_INCLUDE_VERSIONED=0 \
        FFTM_DIRECT_P2P_CUDA_AWARE=1 \
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
        FFTM_4D_NATIVE_XW_CHUNK_MIB="${chunk_mib}" \
        FFTM_4D_NATIVE_XW_CHUNK_WINDOWS=1 \
        FFTM_4D_NATIVE_XW_COMPACT_STAGINGS=on \
        FFTM_4D_SLAB_NATIVE_WORK_AREA_ALIASES=on \
        FFTM_4D_SLAB_NATIVE_WZ_COMM_LAYOUTS=on \
        FFTM_4D_SLAB_NATIVE_WZ_PLAN_CONCURRENCIES="${concurrency}" \
        FFTM_4D_SLAB_NATIVE_WZ_READY_PIPELINES=on \
        FFTM_PRINT_PENCIL_SCHEDULE=0 \
        FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
        FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
        FFTM_ENABLE_GPU_TELEMETRY=1 \
        FFTM_VALIDATION_TIMES=1 \
        FFTM_BENCHMARK_TIMES="${times}" \
        FFTM_WARMUP="${warmup}" \
        FFTM_RUN_PREFLIGHT=0 \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_SRUN_EXTRA_ARGS='--ntasks-per-node=8 --distribution=block:block --kill-on-bad-exit=1' \
        FFTM_SRUN_TIME="${srun_time}" \
        FFTM_TIMEOUT_SECONDS="${timeout_seconds}" \
        FFTM_STOP_ON_FAILURE=1 \
        FFTM_DRY_RUN="${DRY_RUN}" \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

run_image_preflight

STATUS=${DATA_DIR}/status.csv
printf '%s\n' 'phase,wz_concurrency,chunk_mib,staging_cap_mib,returncode,result_dir' > "${STATUS}"
qualified_count=0
passed_count=0

for candidate in "${candidate_list[@]}"; do
    concurrency=${candidate%%:*}
    chunk_mib=${candidate##*:}
    staging_cap_mib=$(( (GPUS - 1) * chunk_mib ))
    candidate_id=c${concurrency}_m${chunk_mib}
    if [[ "${concurrency}" -ne 4 && "${passed_count}" -gt 0 ]]; then
        printf 'skip,%s,%s,%s,0,%s\n' \
            "${concurrency}" "${chunk_mib}" "${staging_cap_mib}" \
            "production-concurrency candidate already passed" >> "${STATUS}"
        echo "Skipping fallback ${candidate_id}; a concurrency-4 candidate passed."
        continue
    fi
    printf 'Candidate %s: compact-staging upper bound=%s MiB (old 512 MiB cap=%s MiB)\n' \
        "${candidate_id}" "${staging_cap_mib}" "$(( (GPUS - 1) * 512 ))"

    qualify_rc=0
    run_case qualify "${concurrency}" "${chunk_mib}" \
        "${QUALIFY_TIMES}" "${QUALIFY_WARMUP}" \
        "${QUALIFY_TIMEOUT}" "${QUALIFY_SRUN_TIME}" || qualify_rc=$?
    printf 'qualify,%s,%s,%s,%s,%s\n' \
        "${concurrency}" "${chunk_mib}" "${staging_cap_mib}" "${qualify_rc}" \
        "${DATA_DIR}/qualify/${candidate_id}" >> "${STATUS}"
    if [[ "${qualify_rc}" -ne 0 ]]; then
        echo "Qualification failed for ${candidate_id}; continuing."
        continue
    fi
    qualified_count=$(( qualified_count + 1 ))

    full_rc=0
    run_case full "${concurrency}" "${chunk_mib}" \
        "${BENCHMARK_TIMES}" "${WARMUP}" \
        "${FULL_TIMEOUT}" "${FULL_SRUN_TIME}" || full_rc=$?
    printf 'full,%s,%s,%s,%s,%s\n' \
        "${concurrency}" "${chunk_mib}" "${staging_cap_mib}" "${full_rc}" \
        "${DATA_DIR}/full/${candidate_id}" >> "${STATUS}"
    if [[ "${full_rc}" -eq 0 ]]; then
        printf '%s\n' "${candidate_id}" >> "${DATA_DIR}/passed_candidates.txt"
        passed_count=$(( passed_count + 1 ))
    else
        echo "Full measurement failed for ${candidate_id}; continuing."
    fi
done

printf 'qualified_candidates=%s\npassed_candidates=%s\n' \
    "${qualified_count}" "${passed_count}" > "${DATA_DIR}/result.env"

if [[ "${DRY_RUN}" == 1 ]]; then
    touch "${DATA_DIR}/DRY_RUN_COMPLETE"
    echo "Dry-run complete: ${DATA_DIR}"
    exit 0
fi
if [[ "${passed_count}" -eq 0 ]]; then
    echo "No full 4D weak-scaling candidate passed. See ${STATUS}." >&2
    exit 1
fi

touch "${DATA_DIR}/PASSED"
echo "4D weak-scaling endpoint complete: ${DATA_DIR}"
