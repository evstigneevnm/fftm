#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_ROOT}"

OMPI_ROOT="${FFTM_COMPARE_OMPI_ROOT:-${HOME}/opt/rocm-mpi/openmpi-rocm-5.0.7}"
UCX_ROOT="${FFTM_COMPARE_UCX_ROOT:-${HOME}/opt/rocm-mpi/ucx}"
FFTW_ROOT="${FFTM_COMPARE_FFTW_ROOT:-${HOME}/opt/fftw-mpi-3.3.10}"
BUILD_DIR="${FFTM_COMPARE_BUILD_DIR:-${PROJECT_ROOT}/build_hip}"
DATA_DIR="${FFTM_COMPARE_DATA_DIR:-${BUILD_DIR}/fftm_vs_fftw_$(date +%Y%m%d_%H%M%S)}"
TIMES="${FFTM_COMPARE_TIMES:-12}"
WARMUP="${FFTM_COMPARE_WARMUP:-3}"
PLANNER="${FFTM_COMPARE_FFTW_PLANNER:-measure}"
PLANNER_LIMIT="${FFTM_COMPARE_FFTW_PLANNER_LIMIT_SECONDS:-15}"
SIZE_3D="${FFTM_COMPARE_SIZE_3D:-256}"
SIZE_4D="${FFTM_COMPARE_SIZE_4D:-64}"
CPU_MASKS_1="${FFTM_COMPARE_CPU_MASKS_1:-0-39}"
CPU_MASKS_2="${FFTM_COMPARE_CPU_MASKS_2:-0-19:20-39}"
CPU_MASKS_4="${FFTM_COMPARE_CPU_MASKS_4:-0-9:10-19:20-29:30-39}"
CPU_THREADS_1="${FFTM_COMPARE_CPU_THREADS_1:-40}"
CPU_THREADS_2="${FFTM_COMPARE_CPU_THREADS_2:-20}"
CPU_THREADS_4="${FFTM_COMPARE_CPU_THREADS_4:-10}"

MPI_RUN="${OMPI_ROOT}/bin/mpirun"
FFTM_3D="${BUILD_DIR}/test_benchmark_fftm_3D_hip_nca.bin"
FFTM_4D="${BUILD_DIR}/test_benchmark_fftm_4D_hip_nca.bin"
FFTW_CPU="${BUILD_DIR}/fftw_cpu/fftw_mpi_cpu_benchmark.bin"
CPU_WRAPPER="${SCRIPT_DIR}/run_rank_cpu_mask.sh"

for executable in "${MPI_RUN}" "${FFTM_3D}" "${FFTM_4D}" "${FFTW_CPU}" "${CPU_WRAPPER}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Missing executable: ${executable}" >&2
        exit 2
    fi
done

mkdir -p "${DATA_DIR}/raw" "${DATA_DIR}/fftm" "${DATA_DIR}/fftw"
STATUS="${DATA_DIR}/status.csv"
printf 'label,implementation,dimension,size,resources,configuration,rc,log\n' > "${STATUS}"

export PATH="${OMPI_ROOT}/bin:/opt/rocm-6.4.1/bin:${PATH}"
export LD_LIBRARY_PATH="${OMPI_ROOT}/lib:${UCX_ROOT}/lib:${FFTW_ROOT}/lib:/opt/rocm-6.4.1/lib:${LD_LIBRARY_PATH:-}"

run_case()
{
    local label="$1"
    local implementation="$2"
    local dimension="$3"
    local size="$4"
    local resources="$5"
    local configuration="$6"
    shift 6

    local log="${DATA_DIR}/raw/${label}.log"
    printf '[%s] %s\n' "${label}" "$*"
    set +e
    "$@" > "${log}" 2>&1
    local rc=$?
    set -e
    printf '%s,%s,%s,%s,%s,%s,%d,%s\n' \
        "${label}" "${implementation}" "${dimension}" "${size}" "${resources}" "${configuration}" \
        "${rc}" "${log}" >> "${STATUS}"
    if (( rc != 0 )); then
        echo "WARNING: ${label} failed with rc=${rc}; see ${log}" >&2
    fi
}

run_fftm_3d()
{
    local gpus="$1"
    local mode="$2"
    local label="fftm_d3_g${gpus}_${mode}"
    local out="${DATA_DIR}/fftm/${label}"
    mkdir -p "${out}"
    local -a grid=(1 1)
    if (( gpus == 2 )); then
        grid=(2 1)
    fi
    run_case "${label}" fftm 3 "${SIZE_3D}^3" "${gpus}gpu" "slab-pencil:${mode}:host-staged" \
        env ROCR_VISIBLE_DEVICES="$([[ ${gpus} == 1 ]] && echo 0 || echo 0,1)" \
        "${MPI_RUN}" -np "${gpus}" --bind-to none --mca pml ob1 \
        "${FFTM_3D}" --strategy slab-pencil --mode "${mode}" --grid "${grid[@]}" \
        --times "${TIMES}" --warmup "${WARMUP}" --directory "${out}" \
        "${SIZE_3D}" "${SIZE_3D}" "${SIZE_3D}"
}

run_fftm_4d()
{
    local gpus="$1"
    local mode="$2"
    local label="fftm_d4_g${gpus}_${mode}"
    local out="${DATA_DIR}/fftm/${label}"
    mkdir -p "${out}"
    local -a grid=(1 1 1)
    if (( gpus == 2 )); then
        grid=(1 2 1)
    fi
    run_case "${label}" fftm 4 "${SIZE_4D}^4" "${gpus}gpu" "slab-slab:${mode}:host-staged" \
        env ROCR_VISIBLE_DEVICES="$([[ ${gpus} == 1 ]] && echo 0 || echo 0,1)" \
        "${MPI_RUN}" -np "${gpus}" --bind-to none --mca pml ob1 \
        "${FFTM_4D}" --strategy slab-slab --mode "${mode}" --grid "${grid[@]}" \
        --times "${TIMES}" --warmup "${WARMUP}" --directory "${out}" \
        "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}"
}

run_fftw()
{
    local dimension="$1"
    local size="$2"
    local ranks="$3"
    local threads="$4"
    local masks="$5"
    local label="fftw_d${dimension}_r${ranks}_t${threads}"
    local out="${DATA_DIR}/fftw/${label}"
    mkdir -p "${out}"
    run_case "${label}" fftw "${dimension}" "${size}^${dimension}" "${ranks}x${threads}cpu" \
        "transposed:measure" \
        env OMP_NUM_THREADS="${threads}" FFTM_CPU_MASKS="${masks}" \
        "${MPI_RUN}" -np "${ranks}" --bind-to none --mca pml ob1 \
        "${CPU_WRAPPER}" "${FFTW_CPU}" --dimension "${dimension}" --size "${size}" \
        --threads "${threads}" --planner "${PLANNER}" --planner-time-limit "${PLANNER_LIMIT}" \
        --warmup "${WARMUP}" --times "${TIMES}" --output-dir "${out}"
}

{
    echo "date=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "size_3d=${SIZE_3D}"
    echo "size_4d=${SIZE_4D}"
    echo "times=${TIMES}"
    echo "warmup=${WARMUP}"
    echo "fftw_planner=${PLANNER}"
    echo "fftw_planner_limit_seconds=${PLANNER_LIMIT}"
    echo "ompi_root=${OMPI_ROOT}"
    echo "ucx_root=${UCX_ROOT}"
    echo "fftw_root=${FFTW_ROOT}"
    lscpu | grep -E '^(CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|Model name|NUMA node\(s\))' || true
    /opt/rocm-6.4.1/bin/rocminfo 2>/dev/null | grep -E '^  Name:.*gfx' | head -2 || true
} > "${DATA_DIR}/metadata.txt"

run_fftm_3d 1 alltoallv
run_fftm_3d 2 alltoallv
run_fftm_3d 2 p2p-waitany

run_fftm_4d 1 alltoallv
run_fftm_4d 2 alltoallv
run_fftm_4d 2 p2p-waitany

run_fftw 3 "${SIZE_3D}" 1 "${CPU_THREADS_1}" "${CPU_MASKS_1}"
run_fftw 3 "${SIZE_3D}" 2 "${CPU_THREADS_2}" "${CPU_MASKS_2}"
run_fftw 3 "${SIZE_3D}" 4 "${CPU_THREADS_4}" "${CPU_MASKS_4}"

run_fftw 4 "${SIZE_4D}" 1 "${CPU_THREADS_1}" "${CPU_MASKS_1}"
run_fftw 4 "${SIZE_4D}" 2 "${CPU_THREADS_2}" "${CPU_MASKS_2}"
run_fftw 4 "${SIZE_4D}" 4 "${CPU_THREADS_4}" "${CPU_MASKS_4}"

echo "Comparison data: ${DATA_DIR}"
