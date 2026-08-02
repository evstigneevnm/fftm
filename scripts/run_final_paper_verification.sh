#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
SELF="${SCRIPT_DIR}/run_final_paper_verification.sh"
TARGET=${1:-all}

IMAGE=${FFTM_FINAL_CONTAINER_IMAGE:-${FFTM_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_bench_a100.sqsh}}
DATA_ROOT=${FFTM_FINAL_DATA_DIR:-/scratch/evstigneevnm/fftm/data_final_paper_$(date +%Y%m%d_%H%M%S)}
SALLOC_EXTRA_ARGS=${FFTM_FINAL_SALLOC_EXTRA_ARGS:---exclude=cn13}
DRY_RUN=${FFTM_FINAL_DRY_RUN:-0}
REQUIRE_CLEAN=${FFTM_FINAL_REQUIRE_CLEAN:-1}
HASH_IMAGE=${FFTM_FINAL_HASH_IMAGE:-1}
EXPECTED_COMMIT=${FFTM_FINAL_EXPECTED_GIT_COMMIT:-$(
    git -C "${ROOT_DIR}" rev-parse --verify HEAD 2>/dev/null || printf unknown
)}

case "${TARGET}" in
    api|production|multinode|all|validate)
        ;;
    *)
        echo "Usage: $0 [api|production|multinode|all|validate]" >&2
        exit 2
        ;;
esac

DATA_ROOT=$(realpath -m -- "${DATA_ROOT}")
mkdir -p "${DATA_ROOT}"

is_true()
{
    case "$1" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${TARGET}" != "validate" ]] && is_true "${REQUIRE_CLEAN}"; then
    if [[ "${EXPECTED_COMMIT}" == "unknown" ]]; then
        echo "ERROR: cannot identify the Git commit for final verification" >&2
        exit 2
    fi
    if ! git -C "${ROOT_DIR}" diff --quiet --ignore-submodules -- ||
       ! git -C "${ROOT_DIR}" diff --cached --quiet --ignore-submodules --; then
        cat >&2 <<EOF
ERROR: tracked source changes are not committed.
The final paper image and verification data must be tied to one Git commit.
Commit the intended source, rebuild the SQSH, and rerun this target.
EOF
        exit 2
    fi
fi

if [[ "${TARGET}" != "validate" ]] && ! is_true "${DRY_RUN}" && [[ ! -f "${IMAGE}" ]]; then
    echo "ERROR: missing SQSH image: ${IMAGE}" >&2
    exit 2
fi

write_provenance()
{
    local provenance="${DATA_ROOT}/provenance.env"
    if [[ ! -f "${provenance}" ]]; then
        {
            printf 'git_commit=%s\n' "${EXPECTED_COMMIT}"
            printf 'container_image=%s\n' "${IMAGE}"
            printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        } > "${provenance}"
    else
        grep -Fx "git_commit=${EXPECTED_COMMIT}" "${provenance}" >/dev/null || {
            echo "ERROR: data root belongs to a different Git commit: ${provenance}" >&2
            return 2
        }
        grep -Fx "container_image=${IMAGE}" "${provenance}" >/dev/null || {
            echo "ERROR: data root belongs to a different SQSH path: ${provenance}" >&2
            return 2
        }
    fi

    if is_true "${HASH_IMAGE}" && ! is_true "${DRY_RUN}"; then
        if [[ ! -s "${DATA_ROOT}/image.sha256" ]]; then
            sha256sum "${IMAGE}" > "${DATA_ROOT}/image.sha256.tmp"
            mv "${DATA_ROOT}/image.sha256.tmp" "${DATA_ROOT}/image.sha256"
        else
            sha256sum --check --status "${DATA_ROOT}/image.sha256" || {
                echo "ERROR: SQSH checksum changed within the final verification root" >&2
                return 2
            }
        fi
    fi
}

run_all()
{
    write_provenance
    local status="${DATA_ROOT}/targets.csv"
    printf '%s\n' 'target,status,returncode' > "${status}"
    local overall=0
    local item rc state
    for item in api production multinode; do
        printf '\n=== Final paper verification target: %s ===\n' "${item}"
        set +e
        env \
            FFTM_FINAL_CONTAINER_IMAGE="${IMAGE}" \
            FFTM_FINAL_DATA_DIR="${DATA_ROOT}" \
            FFTM_FINAL_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}" \
            FFTM_FINAL_DRY_RUN="${DRY_RUN}" \
            FFTM_FINAL_REQUIRE_CLEAN="${REQUIRE_CLEAN}" \
            FFTM_FINAL_HASH_IMAGE="${HASH_IMAGE}" \
            FFTM_FINAL_EXPECTED_GIT_COMMIT="${EXPECTED_COMMIT}" \
            bash "${SELF}" "${item}"
        rc=$?
        set -e
        state=passed
        (( rc == 0 )) || {
            state=failed
            overall=1
        }
        printf '%s,%s,%d\n' "${item}" "${state}" "${rc}" >> "${status}"
    done

    if ! is_true "${DRY_RUN}"; then
        set +e
        env FFTM_FINAL_DATA_DIR="${DATA_ROOT}" bash "${SELF}" validate
        rc=$?
        set -e
        (( rc == 0 )) || overall=1
    fi
    printf 'Final paper verification root: %s\n' "${DATA_ROOT}"
    return "${overall}"
}

validate_all()
{
    local overall=0
    local item
    for item in api production multinode; do
        if [[ ! -f "${DATA_ROOT}/${item}/PASSED" ]]; then
            echo "ERROR: missing pass marker for ${item}: ${DATA_ROOT}/${item}/PASSED" >&2
            overall=1
        fi
    done
    if [[ -d "${DATA_ROOT}/production/benchmarks" ]]; then
        python3 "${SCRIPT_DIR}/validate_final_paper_results.py" \
            --target production \
            --directory "${DATA_ROOT}/production/benchmarks" \
            --output "${DATA_ROOT}/production/final_validation.json" || overall=1
    fi
    if [[ -d "${DATA_ROOT}/multinode/benchmarks" ]]; then
        python3 "${SCRIPT_DIR}/validate_final_paper_results.py" \
            --target multinode \
            --directory "${DATA_ROOT}/multinode/benchmarks" \
            --output "${DATA_ROOT}/multinode/final_validation.json" || overall=1
    fi
    if (( overall == 0 )); then
        printf 'All final paper verification targets passed.\n' | tee "${DATA_ROOT}/FINAL_PASSED"
    fi
    return "${overall}"
}

if [[ "${TARGET}" == "all" ]]; then
    if ! is_true "${DRY_RUN}" && [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "ERROR: the all target must be launched outside an existing allocation" >&2
        echo "Run api, production, and multinode separately inside matching allocations." >&2
        exit 2
    fi
    run_all
    exit $?
fi
if [[ "${TARGET}" == "validate" ]]; then
    validate_all
    exit $?
fi

request_allocation()
{
    local nodes tasks tasks_per_node gpus_per_node allocation_time
    case "${TARGET}" in
        api)
            nodes=1; tasks=4; tasks_per_node=4; gpus_per_node=4; allocation_time=02:00:00 ;;
        production)
            nodes=1; tasks=8; tasks_per_node=8; gpus_per_node=8; allocation_time=04:00:00 ;;
        multinode)
            nodes=2; tasks=16; tasks_per_node=8; gpus_per_node=8; allocation_time=04:00:00 ;;
    esac
    allocation_time=${FFTM_FINAL_ALLOCATION_TIME:-${allocation_time}}
    local extra=()
    if [[ -n "${SALLOC_EXTRA_ARGS}" ]]; then
        read -r -a extra <<< "${SALLOC_EXTRA_ARGS}"
    fi
    printf 'Requesting final %s allocation: nodes=%d tasks=%d GPUs/node=%d time=%s\n' \
        "${TARGET}" "${nodes}" "${tasks}" "${gpus_per_node}" "${allocation_time}"
    exec env \
        FFTM_FINAL_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_FINAL_DATA_DIR="${DATA_ROOT}" \
        FFTM_FINAL_SALLOC_EXTRA_ARGS="${SALLOC_EXTRA_ARGS}" \
        FFTM_FINAL_DRY_RUN=0 \
        FFTM_FINAL_REQUIRE_CLEAN="${REQUIRE_CLEAN}" \
        FFTM_FINAL_HASH_IMAGE="${HASH_IMAGE}" \
        FFTM_FINAL_EXPECTED_GIT_COMMIT="${EXPECTED_COMMIT}" \
        salloc \
        --job-name="fftm-final-${TARGET}" \
        --nodes="${nodes}" \
        --ntasks="${tasks}" \
        --ntasks-per-node="${tasks_per_node}" \
        --cpus-per-task=4 \
        --gpus-per-node="${gpus_per_node}" \
        --exclusive \
        --mem=0 \
        --time="${allocation_time}" \
        "${extra[@]}" \
        bash "${SELF}" "${TARGET}"
}

if ! is_true "${DRY_RUN}" && [[ -z "${SLURM_JOB_ID:-}" ]]; then
    command -v salloc >/dev/null || {
        echo "ERROR: salloc is required outside an existing allocation" >&2
        exit 2
    }
    request_allocation
fi

case "${TARGET}" in
    api|production)
        EXPECTED_NODES=1 ;;
    multinode)
        EXPECTED_NODES=2 ;;
esac
if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${SLURM_JOB_NUM_NODES}" -ne "${EXPECTED_NODES}" ]]; then
    echo "ERROR: ${TARGET} requires exactly ${EXPECTED_NODES} allocated node(s)" >&2
    exit 2
fi

TARGET_DIR="${DATA_ROOT}/${TARGET}"
if [[ -e "${TARGET_DIR}" ]] && [[ -n "$(find "${TARGET_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: target directory is not empty: ${TARGET_DIR}" >&2
    echo "Use a new FFTM_FINAL_DATA_DIR for a new attempt." >&2
    exit 2
fi
mkdir -p "${TARGET_DIR}"
write_provenance

STATUS_FILE="${TARGET_DIR}/status.csv"
printf '%s\n' 'stage,status,returncode,log' > "${STATUS_FILE}"
{
    printf 'target=%s\n' "${TARGET}"
    printf 'git_commit=%s\n' "${EXPECTED_COMMIT}"
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-dry-run}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-dry-run}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
} > "${TARGET_DIR}/config.env"

run_logged()
{
    local stage=$1
    local log_file=$2
    shift 2
    printf '[%s] ' "${stage}"
    printf '%q ' "$@"
    printf '\n'
    if is_true "${DRY_RUN}"; then
        printf '%s,dry-run,0,%s\n' "${stage}" "${log_file}" >> "${STATUS_FILE}"
        return 0
    fi
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e
    local state=passed
    (( rc == 0 )) || state=failed
    printf '%s,%s,%d,%s\n' "${stage}" "${state}" "${rc}" "${log_file}" >> "${STATUS_FILE}"
    return "${rc}"
}

run_matrix_logged()
{
    local stage=$1
    local log_file=$2
    shift 2
    if ! is_true "${DRY_RUN}"; then
        run_logged "${stage}" "${log_file}" "$@"
        return $?
    fi

    printf '[%s dry-run expansion]\n' "${stage}"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local rc=${PIPESTATUS[0]}
    set -e
    local state=dry-run
    (( rc == 0 )) || state=failed
    printf '%s,%s,%d,%s\n' "${stage}" "${state}" "${rc}" "${log_file}" >> "${STATUS_FILE}"
    return "${rc}"
}

verify_image()
{
    local required=(
        test_fftm_options.bin
        test_fftm_quiet_default.bin
        test_fftm_autotune_hardware.bin
        test_fftm_resource_lifecycle.bin
        poisson_periodic_3d_autotuned.bin
        poisson_periodic_4d.bin
    )
    if [[ "${TARGET}" != "api" ]]; then
        required+=( test_benchmark_fftm_3D.bin test_benchmark_fftm_4D.bin )
    fi
    local check="set -e; cat /opt/fftm/bin/fftm_build_info.txt"
    check+="; grep -Fx 'git_commit=${EXPECTED_COMMIT}' /opt/fftm/bin/fftm_build_info.txt"
    check+="; grep -Fx 'git_dirty=0' /opt/fftm/bin/fftm_build_info.txt"
    local binary
    for binary in "${required[@]}"; do
        check+="; test -x '/opt/fftm/bin/${binary}'"
    done
    local command=(
        srun -N 1 -n 1 -G 1 --ntasks-per-node=1 --gpus-per-node=1
        --cpus-per-task=1 --exclusive --kill-on-bad-exit=1 --time=00:05:00
        --container-image "${IMAGE}"
        --container-mounts="${TARGET_DIR}:/data"
        --container-workdir /opt/fftm/bin
        --container-entrypoint /bin/bash -lc "${check}"
    )
    run_logged image-preflight "${TARGET_DIR}/image_preflight.log" "${command[@]}"
}

run_api()
{
    verify_image || return 1

    local options_cmd=(
        srun -N 1 -n 1 -G 1 --ntasks-per-node=1 --gpus-per-node=1
        --cpus-per-task=1 --exclusive --kill-on-bad-exit=1 --time=00:05:00
        --container-image "${IMAGE}" --container-workdir /opt/fftm/bin
        --container-entrypoint /opt/fftm/bin/test_fftm_options.bin
    )
    run_logged public-options "${TARGET_DIR}/public_options.log" "${options_cmd[@]}" || return 1

    local quiet_cmd=(
        srun -N 1 -n 1 -G 1 --ntasks-per-node=1 --gpus-per-node=1
        --cpus-per-task=1 --exclusive --kill-on-bad-exit=1 --time=00:05:00
        --container-image "${IMAGE}" --container-workdir /opt/fftm/bin
        --container-entrypoint /opt/fftm/bin/test_fftm_quiet_default.bin
    )
    run_logged quiet-default "${TARGET_DIR}/quiet_default.log" "${quiet_cmd[@]}" || return 1

    mkdir -p "${TARGET_DIR}/cpp_autotune"
    local tune_cmd=(
        env
        FFTM_CPP_TUNE_CONTAINER_IMAGE="${IMAGE}"
        FFTM_CPP_TUNE_DATA_DIR="${TARGET_DIR}/cpp_autotune"
        FFTM_CPP_TUNE_GPU_COUNTS=4
        FFTM_CPP_TUNE_SIZE=256
        FFTM_CPP_TUNE_WARMUP=1
        FFTM_CPP_TUNE_TIMES=2
        FFTM_CPP_TUNE_APPLICATION_WARMUP=0
        FFTM_CPP_TUNE_APPLICATION_TIMES=1
        FFTM_CPP_TUNE_USE_SINGLE_ALLOCATION=0
        FFTM_CPP_TUNE_SRUN_EXTRA_ARGS=
        bash "${SCRIPT_DIR}/run_cpp_autotune_validation.sh" smoke
    )
    run_logged cpp-autotune "${TARGET_DIR}/cpp_autotune_driver.log" "${tune_cmd[@]}" || return 1

    local reader="${TARGET_DIR}/reader"
    mkdir -p "${reader}"
    local ranks phase case_dir log cache
    for ranks in 1 2; do
        case_dir="${reader}/3d_r${ranks}"
        mkdir -p "${case_dir}"
        cache=/data/autotune.env
        for phase in create reuse; do
            log="${case_dir}/${phase}.log"
            local reader_3d_cmd=(
                srun -N 1 -n "${ranks}" -G "${ranks}"
                --ntasks-per-node="${ranks}" --gpus-per-node="${ranks}"
                --cpus-per-task=2 --exclusive --distribution=block:block
                --kill-on-bad-exit=1 --time=00:10:00
                --container-image "${IMAGE}"
                --container-mounts="${case_dir}:/data"
                --container-workdir /opt/fftm/bin
                --container-entrypoint /usr/bin/env
                FFTM_WRAP_PROCS_GPUS=0 FFTM_CPP_AUTOTUNE_MEASURE=0
                /opt/fftm/bin/poisson_periodic_3d_autotuned.bin
                32 32 32 "${cache}" 1 0
            )
            run_logged "reader-3d-r${ranks}-${phase}" "${log}" "${reader_3d_cmd[@]}" || return 1
        done
        if ! is_true "${DRY_RUN}"; then
            grep -q 'source=created' "${case_dir}/create.log" || return 1
            grep -q 'source=cache' "${case_dir}/reuse.log" || return 1
            grep -q '^FFTM_AUTOTUNE_SCHEMA=2$' "${case_dir}/autotune.env" || return 1
            grep -q "^FFTM_AUTOTUNE_DEVICES_PER_NODE=${ranks}$" "${case_dir}/autotune.env" || return 1
        fi
    done

    for ranks in 1 2; do
        case_dir="${reader}/4d_r${ranks}"
        mkdir -p "${case_dir}"
        log="${case_dir}/run.log"
        local reader_4d_cmd=(
            srun -N 1 -n "${ranks}" -G "${ranks}"
            --ntasks-per-node="${ranks}" --gpus-per-node="${ranks}"
            --cpus-per-task=2 --exclusive --distribution=block:block
            --kill-on-bad-exit=1 --time=00:10:00
            --container-image "${IMAGE}"
            --container-mounts="${case_dir}:/data"
            --container-workdir /opt/fftm/bin
            --container-entrypoint /usr/bin/env
            FFTM_WRAP_PROCS_GPUS=0
            /opt/fftm/bin/poisson_periodic_4d.bin 16 1
        )
        run_logged "reader-4d-r${ranks}" "${log}" "${reader_4d_cmd[@]}" || return 1
        if ! is_true "${DRY_RUN}"; then
            grep -q 'layout=native_xzwy' "${log}" || return 1
            grep -q 'rel_l2=' "${log}" || return 1
        fi
    done

    api_artifact_validation()
    {
        grep -q ',validate,passed,0,' "${TARGET_DIR}/cpp_autotune/status.csv"
        ! grep -q ',failed,' "${TARGET_DIR}/cpp_autotune/status.csv"
        for ranks in 1 2; do
            grep -q 'source=created' "${reader}/3d_r${ranks}/create.log"
            grep -q 'source=cache' "${reader}/3d_r${ranks}/reuse.log"
            grep -q 'rel_l2=' "${reader}/3d_r${ranks}/reuse.log"
            grep -q '^FFTM_AUTOTUNE_SCHEMA=2$' "${reader}/3d_r${ranks}/autotune.env"
            grep -q "^FFTM_AUTOTUNE_DEVICES_PER_NODE=${ranks}$" \
                "${reader}/3d_r${ranks}/autotune.env"
            grep -q 'layout=native_xzwy' "${reader}/4d_r${ranks}/run.log"
            grep -q 'rel_l2=' "${reader}/4d_r${ranks}/run.log"
        done
    }
    if ! is_true "${DRY_RUN}"; then
        run_logged api-artifacts "${TARGET_DIR}/api_artifacts.log" api_artifact_validation || return 1
    fi
}

production_environment()
{
    env \
        FFTM_CLEANUP_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_CLEANUP_DATA_DIR="${TARGET_DIR}/benchmarks" \
        FFTM_CLEANUP_GPU_COUNTS=6,7,8 \
        FFTM_CLEANUP_SRUN_EXTRA_ARGS= \
        FFTM_CLEANUP_RUN_QUIET_SMOKE=0 \
        FFTM_CLEANUP_BENCHMARK_TIMES=10 \
        FFTM_CLEANUP_WARMUP=3 \
        FFTM_CLEANUP_RUN_PREFLIGHT=1 \
        FFTM_CLEANUP_SRUN_TIME=01:00:00 \
        FFTM_CLEANUP_TIMEOUT_SECONDS=3600 \
        FFTM_CLEANUP_DRY_RUN="${DRY_RUN}" \
        bash "${SCRIPT_DIR}/run_cleanup_production_guard.sh"
}

run_production()
{
    verify_image || return 1
    mkdir -p "${TARGET_DIR}/benchmarks"
    run_matrix_logged production-matrix "${TARGET_DIR}/production_driver.log" production_environment || return 1
    if ! is_true "${DRY_RUN}"; then
        run_logged production-validate "${TARGET_DIR}/validation.log" \
            python3 "${SCRIPT_DIR}/validate_final_paper_results.py" \
            --target production --directory "${TARGET_DIR}/benchmarks" \
            --output "${TARGET_DIR}/final_validation.json" || return 1
    fi
}

multinode_environment()
{
    env \
        FFTM_CONTAINER_IMAGE="${IMAGE}" \
        FFTM_DATA_DIR="${TARGET_DIR}/benchmarks" \
        FFTM_NODE_COUNTS=2 \
        FFTM_GPU_COUNTS=16 \
        FFTM_GPUS_PER_NODE=8 \
        FFTM_DEVICE_MEMORY_MIB=81920 \
        FFTM_BENCHMARK_SIZES_3D=none \
        FFTM_BENCHMARK_SIZES_4D=none \
        FFTM_FIXED_SCALING_SIZES_3D=2048 \
        FFTM_FIXED_SCALING_SIZES_4D=320 \
        FFTM_EXTRA_SIZES_3D= \
        FFTM_EXTRA_SIZES_3D_BY_GPU= \
        FFTM_TRANSPORTS=cuda_aware \
        FFTM_MODES=p2p-waitany \
        FFTM_STRATEGIES_3D=pencil-pencil \
        FFTM_STRATEGIES_4D=slab-slab \
        FFTM_SKIP_FFTS=1 \
        FFTM_SKIP_FFTM=0 \
        FFTM_INCLUDE_VERSIONED=0 \
        FFTM_3D_BACKENDS=native \
        FFTM_PENCIL_PIPELINES=reference-parity \
        FFTM_PENCIL_LAYOUTS=opt0 \
        FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS=4x4 \
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
        FFTM_PRINT_PENCIL_SCHEDULE=0 \
        FFTM_ENABLE_NATIVE_STAGE_TIMERS=0 \
        FFTM_ENABLE_LOCAL_FFT_DIAGNOSTICS=0 \
        FFTM_ENABLE_GPU_TELEMETRY=1 \
        FFTM_BENCHMARK_TIMES=10 \
        FFTM_WARMUP=5 \
        FFTM_VALIDATION_TIMES=1 \
        FFTM_RUN_PREFLIGHT=1 \
        FFTM_MPI_RANK_AFFINITY_MODE=hca \
        FFTM_SRUN_EXTRA_ARGS= \
        FFTM_SRUN_TIME=01:30:00 \
        FFTM_TIMEOUT_SECONDS=5400 \
        FFTM_STOP_ON_FAILURE=1 \
        FFTM_DRY_RUN="${DRY_RUN}" \
        bash "${SCRIPT_DIR}/run_slurm_pyxis_benchmarks.sh"
}

run_multinode()
{
    verify_image || return 1
    mkdir -p "${TARGET_DIR}/benchmarks"
    run_matrix_logged multinode-matrix "${TARGET_DIR}/multinode_driver.log" multinode_environment || return 1
    if ! is_true "${DRY_RUN}"; then
        run_logged multinode-validate "${TARGET_DIR}/validation.log" \
            python3 "${SCRIPT_DIR}/validate_final_paper_results.py" \
            --target multinode --directory "${TARGET_DIR}/benchmarks" \
            --output "${TARGET_DIR}/final_validation.json" || return 1
    fi
}

overall=0
case "${TARGET}" in
    api) run_api || overall=1 ;;
    production) run_production || overall=1 ;;
    multinode) run_multinode || overall=1 ;;
esac

if (( overall == 0 )) && ! is_true "${DRY_RUN}"; then
    printf 'target=%s\ngit_commit=%s\ncompleted_utc=%s\n' \
        "${TARGET}" "${EXPECTED_COMMIT}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${TARGET_DIR}/PASSED"
fi
printf 'Final %s verification output: %s\n' "${TARGET}" "${TARGET_DIR}"
exit "${overall}"
