#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Run direct threaded FFTW-MPI CPU benchmarks through Slurm/Pyxis.

Environment:
  FFTW_CPU_CONTAINER_IMAGE       SQSH image path.
  FFTW_CPU_DATA_DIR              Output directory.
  FFTW_CPU_NODE_COUNTS           Comma-separated node counts.
  FFTW_CPU_RANKS_PER_NODE        Comma-separated MPI ranks per node.
  FFTW_CPU_CORES_PER_NODE        Physical CPU cores used per node.
  FFTW_CPU_DIMENSIONS            3, 4, or 3,4.
  FFTW_CPU_SCALING_ROLES         strong, weak, or strong,weak.
  FFTW_CPU_PLANNERS              estimate, measure, patient, exhaustive.
  FFTW_CPU_PLANNER_TIME_LIMIT_SECONDS
                                Limit for each FFTW planner call.
  FFTW_CPU_BENCHMARK_TIMES       Timed pairs.
  FFTW_CPU_WARMUP                Untimed pairs.
  FFTW_CPU_TIMEOUT_SECONDS       Per-case host timeout.
  FFTW_CPU_SRUN_TIME             Per-case Slurm step limit.
  FFTW_CPU_SRUN_EXTRA_ARGS       Additional srun arguments.
  FFTW_CPU_USE_SINGLE_ALLOCATION auto, 0, or 1 (default: auto).
  FFTW_CPU_ALLOCATION_TIME       Wall time for the shared allocation.
  FFTW_CPU_SALLOC_EXTRA_ARGS     Additional shared-allocation arguments.
  FFTW_CPU_SKIP_PREFLIGHT        1 skips the one-rank image/binary check.
  FFTW_CPU_STOP_ON_FAILURE       0 continues the matrix; 1 stops.
  FFTW_CPU_DRY_RUN               1 prints commands without running.

OpenMP initialization loops run unbound inside each Slurm task cpuset.
This preserves the complete task mask for FFTW's pthread worker pool.

Default strong sizes are 2048^3 and 320^4. Weak sizes use the same
per-node mapping as the 8-GPU-per-node FFTM scaling matrix.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${REPO_ROOT}"

IMAGE="${FFTW_CPU_CONTAINER_IMAGE:-/scratch/evstigneevnm/fftm/fftm_fftw_mpi_cpu.sqsh}"
STAMP="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${FFTW_CPU_DATA_DIR:-/scratch/evstigneevnm/fftm/data_fftw_mpi_cpu_${STAMP}}"
NODE_COUNTS="${FFTW_CPU_NODE_COUNTS:-1,2}"
RANKS_PER_NODE="${FFTW_CPU_RANKS_PER_NODE:-1,2,4}"
CORES_PER_NODE="${FFTW_CPU_CORES_PER_NODE:-128}"
DIMENSIONS="${FFTW_CPU_DIMENSIONS:-3,4}"
SCALING_ROLES="${FFTW_CPU_SCALING_ROLES:-strong}"
PLANNERS="${FFTW_CPU_PLANNERS:-measure}"
PLANNER_TIME_LIMIT_SECONDS="${FFTW_CPU_PLANNER_TIME_LIMIT_SECONDS:-120}"
TIMES="${FFTW_CPU_BENCHMARK_TIMES:-10}"
WARMUP="${FFTW_CPU_WARMUP:-3}"
EPSILON="${FFTW_CPU_EPSILON:-1.0e-11}"
TIMEOUT_SECONDS="${FFTW_CPU_TIMEOUT_SECONDS:-3600}"
SRUN_TIME="${FFTW_CPU_SRUN_TIME:-01:00:00}"
if [[ -n "${FFTW_CPU_SRUN_EXTRA_ARGS+x}" ]]; then
    SRUN_EXTRA_ARGS="${FFTW_CPU_SRUN_EXTRA_ARGS}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    SRUN_EXTRA_ARGS=""
else
    SRUN_EXTRA_ARGS="--exclude=cn13"
fi
USE_SINGLE_ALLOCATION="${FFTW_CPU_USE_SINGLE_ALLOCATION:-auto}"
ALLOCATION_TIME="${FFTW_CPU_ALLOCATION_TIME:-12:00:00}"
SALLOC_EXTRA_ARGS="${FFTW_CPU_SALLOC_EXTRA_ARGS---exclude=cn13}"
STOP_ON_FAILURE="${FFTW_CPU_STOP_ON_FAILURE:-0}"
DRY_RUN="${FFTW_CPU_DRY_RUN:-0}"
SKIP_PREFLIGHT="${FFTW_CPU_SKIP_PREFLIGHT:-0}"

STRONG_SIZE_3D="${FFTW_CPU_STRONG_SIZE_3D:-2048}"
STRONG_SIZE_4D="${FFTW_CPU_STRONG_SIZE_4D:-320}"
WEAK_SIZES_3D="${FFTW_CPU_WEAK_SIZES_3D:-1:2048,2:2560,4:3200,8:4096,12:4704,15:5040}"
WEAK_SIZES_4D="${FFTW_CPU_WEAK_SIZES_4D:-1:320,2:384,4:448,8:540,12:600,15:630}"

if [[ "${DRY_RUN}" == "0" ]]; then
    for binary in srun timeout; do
        command -v "${binary}" >/dev/null || {
            echo "Missing required command: ${binary}" >&2
            exit 2
        }
    done
fi
[[ "${DRY_RUN}" == "1" || -f "${IMAGE}" ]] || {
    echo "Missing FFTW MPI SQSH: ${IMAGE}" >&2
    exit 2
}
[[ "${STOP_ON_FAILURE}" == "0" || "${STOP_ON_FAILURE}" == "1" ]] || {
    echo "FFTW_CPU_STOP_ON_FAILURE must be 0 or 1" >&2
    exit 2
}
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
    echo "FFTW_CPU_DRY_RUN must be 0 or 1" >&2
    exit 2
}
[[ "${SKIP_PREFLIGHT}" == "0" || "${SKIP_PREFLIGHT}" == "1" ]] || {
    echo "FFTW_CPU_SKIP_PREFLIGHT must be 0 or 1" >&2
    exit 2
}
case "${USE_SINGLE_ALLOCATION}" in
    auto|0|1)
        ;;
    *)
        echo "FFTW_CPU_USE_SINGLE_ALLOCATION must be auto, 0, or 1" >&2
        exit 2
        ;;
esac
[[ "${CORES_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FFTW_CPU_CORES_PER_NODE must be a positive integer" >&2
    exit 2
}
[[ "${TIMES}" =~ ^[1-9][0-9]*$ && "${WARMUP}" =~ ^[0-9]+$ ]] || {
    echo "FFTW_CPU_BENCHMARK_TIMES must be positive and FFTW_CPU_WARMUP nonnegative" >&2
    exit 2
}
[[ "${PLANNER_TIME_LIMIT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "FFTW_CPU_PLANNER_TIME_LIMIT_SECONDS must be a positive integer" >&2
    exit 2
}

IFS=',' read -r -a node_values <<< "${NODE_COUNTS}"
max_nodes=0
for nodes in "${node_values[@]}"; do
    [[ "${nodes}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid node count: ${nodes}" >&2
        exit 2
    }
    (( nodes > max_nodes )) && max_nodes="${nodes}"
done

allocate_once=0
if [[ "${USE_SINGLE_ALLOCATION}" == "1" || "${USE_SINGLE_ALLOCATION}" == "auto" ]]; then
    allocate_once=1
fi
if [[ "${DRY_RUN}" == "0" && -z "${SLURM_JOB_ID:-}" && "${allocate_once}" == "1" ]]; then
    command -v salloc >/dev/null || {
        echo "Missing required command: salloc" >&2
        exit 2
    }
    mkdir -p "${DATA_DIR}"
    DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"
    read -r -a salloc_extra <<< "${SALLOC_EXTRA_ARGS}"
    printf 'Requesting one FFTW-MPI CPU allocation: nodes=%s physical_cores_per_node=%s time=%s\n' \
        "${max_nodes}" "${CORES_PER_NODE}" "${ALLOCATION_TIME}"
    exec env \
        FFTW_CPU_DATA_DIR="${DATA_DIR}" \
        FFTW_CPU_USE_SINGLE_ALLOCATION=0 \
        FFTW_CPU_SRUN_EXTRA_ARGS="" \
        salloc \
        --job-name="fftw-cpu-${max_nodes}n" \
        --nodes="${max_nodes}" \
        --ntasks="${max_nodes}" \
        --ntasks-per-node=1 \
        --cpus-per-task="${CORES_PER_NODE}" \
        --hint=nomultithread \
        --exclusive \
        --mem=0 \
        --time="${ALLOCATION_TIME}" \
        "${salloc_extra[@]}" \
        bash "${SCRIPT_DIR}/run_fftw_mpi_cpu_benchmarks.sh"
fi
if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NUM_NODES:-}" &&
      "${SLURM_JOB_NUM_NODES}" -lt "${max_nodes}" ]]; then
    echo "Current allocation has ${SLURM_JOB_NUM_NODES} nodes; matrix requires ${max_nodes}" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}/raw"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"
STATUS_PATH="${DATA_DIR}/status.csv"
CONFIG_PATH="${DATA_DIR}/config.env"

printf '%s\n' \
    "case_id,status,returncode,elapsed_seconds,dimension,scaling_role,nodes,ranks_per_node,threads_per_rank,size,planner,log" \
    > "${STATUS_PATH}"

{
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'node_counts=%s\n' "${NODE_COUNTS}"
    printf 'ranks_per_node=%s\n' "${RANKS_PER_NODE}"
    printf 'cores_per_node=%s\n' "${CORES_PER_NODE}"
    printf 'dimensions=%s\n' "${DIMENSIONS}"
    printf 'scaling_roles=%s\n' "${SCALING_ROLES}"
    printf 'planners=%s\n' "${PLANNERS}"
    printf 'planner_time_limit_seconds=%s\n' "${PLANNER_TIME_LIMIT_SECONDS}"
    printf 'times=%s\n' "${TIMES}"
    printf 'warmup=%s\n' "${WARMUP}"
    printf 'epsilon=%s\n' "${EPSILON}"
    printf 'timeout_seconds=%s\n' "${TIMEOUT_SECONDS}"
    printf 'srun_time=%s\n' "${SRUN_TIME}"
    printf 'srun_extra_args=%s\n' "${SRUN_EXTRA_ARGS}"
    printf 'use_single_allocation=%s\n' "${USE_SINGLE_ALLOCATION}"
    printf 'allocation_time=%s\n' "${ALLOCATION_TIME}"
    printf 'salloc_extra_args=%s\n' "${SALLOC_EXTRA_ARGS}"
    printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'slurm_job_nodelist=%s\n' "${SLURM_JOB_NODELIST:-}"
    printf 'skip_preflight=%s\n' "${SKIP_PREFLIGHT}"
    printf 'stop_on_failure=%s\n' "${STOP_ON_FAILURE}"
    printf 'dry_run=%s\n' "${DRY_RUN}"
    printf 'openmp_binding=unbound_within_slurm_cpuset\n'
    printf 'git_commit=%s\n' "$(git rev-parse --verify HEAD 2>/dev/null || printf unknown)"
    printf 'image_checksum=%s\n' "$(
        if [[ -s "${IMAGE}.sha256" ]]; then
            read -r checksum _ < "${IMAGE}.sha256"
            printf '%s' "${checksum}"
        else
            printf unavailable
        fi
    )"
    printf 'strong_size_3d=%s\n' "${STRONG_SIZE_3D}"
    printf 'strong_size_4d=%s\n' "${STRONG_SIZE_4D}"
    printf 'weak_sizes_3d=%s\n' "${WEAK_SIZES_3D}"
    printf 'weak_sizes_4d=%s\n' "${WEAK_SIZES_4D}"
} > "${CONFIG_PATH}"

csv_contains()
{
    local csv="$1"
    local expected="$2"
    local item
    IFS=',' read -r -a items <<< "${csv}"
    for item in "${items[@]}"; do
        [[ "${item}" == "${expected}" ]] && return 0
    done
    return 1
}

mapped_size()
{
    local mapping="$1"
    local expected_nodes="$2"
    local entry
    IFS=',' read -r -a entries <<< "${mapping}"
    for entry in "${entries[@]}"; do
        if [[ "${entry%%:*}" == "${expected_nodes}" ]]; then
            printf '%s\n' "${entry#*:}"
            return 0
        fi
    done
    return 1
}

append_case_size()
{
    local -n destination="$1"
    local candidate="$2"
    local existing
    for existing in "${destination[@]:-}"; do
        [[ "${existing}" == "${candidate}" ]] && return
    done
    destination+=( "${candidate}" )
}

csv_data_rows()
{
    local path="$1"
    local line_count
    if [[ ! -s "${path}" ]]; then
        printf '0\n'
        return
    fi
    line_count="$(wc -l < "${path}")"
    if (( line_count > 0 )); then
        printf '%s\n' "$(( line_count - 1 ))"
    else
        printf '0\n'
    fi
}

summary_matches()
{
    local path="$1"
    local expected_dimension="$2"
    local isotropic_size="$3"
    local expected_ranks="$4"
    local expected_nodes="$5"
    local expected_rpn="$6"
    local expected_threads="$7"
    local expected_planner="$8"
    local expected_warmup="$9"
    local expected_times="${10}"
    local expected_sizes="${isotropic_size}"
    local line
    local axis
    local -a fields

    [[ -s "${path}" ]] || return 1
    for (( axis = 1; axis < expected_dimension; ++axis )); do
        expected_sizes+="x${isotropic_size}"
    done
    line="$(tail -n 1 "${path}")"
    IFS=',' read -r -a fields <<< "${line}"
    (( ${#fields[@]} >= 36 )) || return 1

    [[ "${fields[1]}" == "${expected_dimension}" &&
       "${fields[2]}" == "${expected_sizes}" &&
       "${fields[3]}" == "${expected_ranks}" &&
       "${fields[4]}" == "${expected_nodes}" &&
       "${fields[5]}" == "${expected_rpn}" &&
       "${fields[6]}" == "${expected_rpn}" &&
       "${fields[7]}" == "${expected_threads}" &&
       "${fields[8]}" == "${expected_threads}" &&
       "${fields[9]}" == "${expected_planner}" &&
       "${fields[10]}" == "1" &&
       "${fields[11]}" == "${expected_warmup}" &&
       "${fields[12]}" == "${expected_times}" &&
       -n "${fields[35]}" ]]
}

read -r -a srun_extra <<< "${SRUN_EXTRA_ARGS}"
IFS=',' read -r -a rpn_values <<< "${RANKS_PER_NODE}"
IFS=',' read -r -a planner_values <<< "${PLANNERS}"

if [[ "${SKIP_PREFLIGHT}" == "0" ]]; then
    preflight_dir="${DATA_DIR}/.fftw_cpu_preflight"
    preflight_summary="${preflight_dir}/benchmark_fftw_mpi_cpu.csv"
    preflight_iterations="${preflight_dir}/fftw_mpi_cpu_iterations.csv"
    mkdir -p "${preflight_dir}"
    rm -f "${preflight_summary}" "${preflight_iterations}"
    printf 'Running FFTW-MPI CPU image and affinity preflight checks...\n'
    for preflight_rpn in "${rpn_values[@]}"; do
        [[ "${preflight_rpn}" =~ ^[1-9][0-9]*$ ]] || {
            echo "Invalid preflight ranks-per-node value: ${preflight_rpn}" >&2
            exit 2
        }
        (( CORES_PER_NODE % preflight_rpn == 0 )) || {
            echo "${CORES_PER_NODE} cores are not divisible by ${preflight_rpn} preflight ranks per node" >&2
            exit 2
        }
        preflight_threads=$(( CORES_PER_NODE / preflight_rpn ))
        preflight=(
            srun
            -N 1
            -n "${preflight_rpn}"
            --ntasks-per-node="${preflight_rpn}"
            --cpus-per-task="${preflight_threads}"
            --hint=nomultithread
            --cpu-bind=cores
            --distribution=block:block
            --time=00:03:00
            "${srun_extra[@]}"
            --container-image "${IMAGE}"
            --container-mounts="${DATA_DIR}:/data"
            --container-workdir /opt/fftw_cpu
            --container-entrypoint /opt/fftw_cpu/bin/fftw_mpi_cpu_benchmark.bin
            --dimension 3
            --size 32
            --threads "${preflight_threads}"
            --planner estimate
            --planner-time-limit "${PLANNER_TIME_LIMIT_SECONDS}"
            --warmup 0
            --times 1
            --epsilon "${EPSILON}"
            --output-dir /data/.fftw_cpu_preflight
        )
        printf '  affinity rpn=%s threads=%s: ' "${preflight_rpn}" "${preflight_threads}"
        printf '%q ' env -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY \
            "OMP_NUM_THREADS=${preflight_threads}" MALLOC_ARENA_MAX=2 "${preflight[@]}"
        printf '\n'
        if [[ "${DRY_RUN}" == "0" ]]; then
            timeout --signal=TERM --kill-after=30 300 \
                env -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY \
                "OMP_NUM_THREADS=${preflight_threads}" MALLOC_ARENA_MAX=2 \
                "${preflight[@]}"
            [[ -s "${preflight_summary}" && -s "${preflight_iterations}" ]] || {
                echo "FFTW-MPI preflight did not write the expected CSV files through /data" >&2
                exit 2
            }
            if ! summary_matches \
                "${preflight_summary}" 3 32 "${preflight_rpn}" 1 \
                "${preflight_rpn}" "${preflight_threads}" estimate 0 1
            then
                echo "FFTW-MPI preflight failed its topology, affinity, or numerical validation" >&2
                exit 2
            fi
        fi
    done
    if [[ "${DRY_RUN}" == "0" ]]; then
        rm -f "${preflight_summary}" "${preflight_iterations}"
        rmdir "${preflight_dir}"
    fi
fi

case_index=0
failed=0

for nodes in "${node_values[@]}"; do
    [[ "${nodes}" =~ ^[1-9][0-9]*$ ]] || {
        echo "Invalid node count: ${nodes}" >&2
        exit 2
    }
    for dimension in 3 4; do
        csv_contains "${DIMENSIONS}" "${dimension}" || continue
        sizes=()
        if csv_contains "${SCALING_ROLES}" strong; then
            if [[ "${dimension}" == "3" ]]; then
                append_case_size sizes "${STRONG_SIZE_3D}:strong"
            else
                append_case_size sizes "${STRONG_SIZE_4D}:strong"
            fi
        fi
        if csv_contains "${SCALING_ROLES}" weak; then
            if [[ "${dimension}" == "3" ]]; then
                weak_size="$(mapped_size "${WEAK_SIZES_3D}" "${nodes}")" || {
                    echo "No 3D weak size for ${nodes} nodes" >&2
                    exit 2
                }
            else
                weak_size="$(mapped_size "${WEAK_SIZES_4D}" "${nodes}")" || {
                    echo "No 4D weak size for ${nodes} nodes" >&2
                    exit 2
                }
            fi
            duplicate=0
            for entry in "${sizes[@]:-}"; do
                if [[ "${entry%%:*}" == "${weak_size}" ]]; then
                    duplicate=1
                    break
                fi
            done
            if [[ "${duplicate}" == "0" ]]; then
                sizes+=( "${weak_size}:weak" )
            fi
        fi

        for size_role in "${sizes[@]}"; do
            size="${size_role%%:*}"
            role="${size_role#*:}"
            for rpn in "${rpn_values[@]}"; do
                [[ "${rpn}" =~ ^[1-9][0-9]*$ ]] || {
                    echo "Invalid ranks-per-node value: ${rpn}" >&2
                    exit 2
                }
                if (( CORES_PER_NODE % rpn != 0 )); then
                    echo "${CORES_PER_NODE} cores are not divisible by ${rpn} ranks per node" >&2
                    exit 2
                fi
                threads=$(( CORES_PER_NODE / rpn ))
                ranks=$(( nodes * rpn ))
                for planner in "${planner_values[@]}"; do
                    case_index=$(( case_index + 1 ))
                    case_id="$(printf '%04d_d%s_%s_n%s_rpn%s_t%s_%s' \
                        "${case_index}" "${dimension}" "${size}" "${nodes}" \
                        "${rpn}" "${threads}" "${planner}")"
                    log_path="${DATA_DIR}/raw/${case_id}.log"
                    summary_path="${DATA_DIR}/benchmark_fftw_mpi_cpu.csv"
                    iteration_path="${DATA_DIR}/fftw_mpi_cpu_iterations.csv"
                    summary_rows_before="$(csv_data_rows "${summary_path}")"
                    iteration_rows_before="$(csv_data_rows "${iteration_path}")"
                    command=(
                        srun
                        -N "${nodes}"
                        -n "${ranks}"
                        --ntasks-per-node="${rpn}"
                        --cpus-per-task="${threads}"
                        --hint=nomultithread
                        --cpu-bind=cores
                        --distribution=block:block
                        --exclusive
                        --mem=0
                        --kill-on-bad-exit=1
                        --time="${SRUN_TIME}"
                        "${srun_extra[@]}"
                        --container-image "${IMAGE}"
                        --container-mounts="${DATA_DIR}:/data"
                        --container-workdir /opt/fftw_cpu
                        --container-entrypoint /opt/fftw_cpu/bin/fftw_mpi_cpu_benchmark.bin
                        --dimension "${dimension}"
                        --size "${size}"
                        --threads "${threads}"
                        --planner "${planner}"
                        --planner-time-limit "${PLANNER_TIME_LIMIT_SECONDS}"
                        --warmup "${WARMUP}"
                        --times "${TIMES}"
                        --epsilon "${EPSILON}"
                        --output-dir /data
                    )

                    printf '[%04d] d=%s role=%s nodes=%s ranks=%s rpn=%s threads=%s size=%s planner=%s\n' \
                        "${case_index}" "${dimension}" "${role}" "${nodes}" "${ranks}" \
                        "${rpn}" "${threads}" "${size}" "${planner}"
                    printf '  '
                    printf '%q ' env \
                        -u OMP_PROC_BIND \
                        -u OMP_PLACES \
                        -u GOMP_CPU_AFFINITY \
                        "OMP_NUM_THREADS=${threads}" \
                        MALLOC_ARENA_MAX=2 \
                        "${command[@]}"
                    printf '\n'

                    if [[ "${DRY_RUN}" == "1" ]]; then
                        printf '%s\n' "${case_id},dry-run,0,0,${dimension},${role},${nodes},${rpn},${threads},${size},${planner},${log_path}" \
                            >> "${STATUS_PATH}"
                        continue
                    fi

                    start_epoch="$(date +%s)"
                    set +e
                    env \
                        -u OMP_PROC_BIND \
                        -u OMP_PLACES \
                        -u GOMP_CPU_AFFINITY \
                        OMP_NUM_THREADS="${threads}" \
                        MALLOC_ARENA_MAX=2 \
                        timeout --signal=TERM --kill-after=60 "${TIMEOUT_SECONDS}" \
                        "${command[@]}" > "${log_path}" 2>&1
                    rc=$?
                    set -e
                    if [[ "${rc}" == "0" ]]; then
                        summary_rows_after="$(csv_data_rows "${summary_path}")"
                        iteration_rows_after="$(csv_data_rows "${iteration_path}")"
                        if ! grep -q '^FFTW_MPI_RESULT ' "${log_path}"; then
                            printf 'ERROR: successful process produced no FFTW_MPI_RESULT marker\n' >> "${log_path}"
                            rc=90
                        elif (( summary_rows_after != summary_rows_before + 1 )); then
                            printf 'ERROR: expected one new summary row; before=%s after=%s\n' \
                                "${summary_rows_before}" "${summary_rows_after}" >> "${log_path}"
                            rc=91
                        elif (( iteration_rows_after != iteration_rows_before + TIMES )); then
                            printf 'ERROR: expected %s new iteration rows; before=%s after=%s\n' \
                                "${TIMES}" "${iteration_rows_before}" "${iteration_rows_after}" >> "${log_path}"
                            rc=92
                        elif ! summary_matches \
                            "${summary_path}" "${dimension}" "${size}" "${ranks}" \
                            "${nodes}" "${rpn}" "${threads}" "${planner}" "${WARMUP}" "${TIMES}"
                        then
                            printf 'ERROR: summary metadata, CPU affinity, or numerical validation mismatch\n' \
                                >> "${log_path}"
                            rc=93
                        fi
                    fi
                    elapsed=$(( $(date +%s) - start_epoch ))
                    if [[ "${rc}" == "0" ]]; then
                        status=ok
                    else
                        status=failed
                        failed=$(( failed + 1 ))
                    fi
                    printf '%s\n' "${case_id},${status},${rc},${elapsed},${dimension},${role},${nodes},${rpn},${threads},${size},${planner},${log_path}" \
                        >> "${STATUS_PATH}"
                    if [[ "${rc}" != "0" ]]; then
                        echo "Case ${case_id} failed with rc=${rc}; log=${log_path}" >&2
                        if [[ "${STOP_ON_FAILURE}" == "1" ]]; then
                            exit "${rc}"
                        fi
                    fi
                done
            done
        done
    done
done

printf 'Completed %d cases with %d failures. Data: %s\n' \
    "${case_index}" "${failed}" "${DATA_DIR}"
(( failed == 0 ))
