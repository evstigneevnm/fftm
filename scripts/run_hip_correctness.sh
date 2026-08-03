#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${FFTM_HIP_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROCM_DIR=${FFTM_HIP_ROCM_DIR:-/opt/rocm}
HIPCC=${FFTM_HIP_HIPCC:-${ROCM_DIR}/bin/hipcc}
BUILD_DIR=${FFTM_HIP_BUILD_DIR:-${ROOT_DIR}/build/hip_correctness}
RESULT_DIR=${FFTM_HIP_RESULT_DIR:-${BUILD_DIR}/results}
MPI_DIR=${FFTM_HIP_MPI_DIR:-}
HIP_ARCH=${FFTM_HIP_ARCH:-}
TEST_DEVICE_AWARE=${FFTM_HIP_TEST_DEVICE_AWARE:-0}
TEST_HOST_STAGED=${FFTM_HIP_TEST_HOST_STAGED:-1}
FORCE_REBUILD=${FFTM_HIP_FORCE_REBUILD:-0}
SIZE_3D=${FFTM_HIP_SIZE_3D:-64}
SIZE_4D=${FFTM_HIP_SIZE_4D:-24}
THRESHOLD=${FFTM_HIP_THRESHOLD:-1e-10}
POISSON_L2_THRESHOLD=${FFTM_HIP_POISSON_L2_THRESHOLD:-1e-3}
POISSON_H1_THRESHOLD=${FFTM_HIP_POISSON_H1_THRESHOLD:-3e-3}
TWO_GPU_MODES=${FFTM_HIP_TWO_GPU_MODES:-alltoallv,p2p-waitany}

if [[ "${TEST_HOST_STAGED}" != 1 && "${TEST_DEVICE_AWARE}" != 1 ]]; then
    echo "Enable at least one of FFTM_HIP_TEST_HOST_STAGED or FFTM_HIP_TEST_DEVICE_AWARE." >&2
    exit 1
fi

if [[ ! -x "${HIPCC}" ]]; then
    echo "HIP compiler is not executable: ${HIPCC}" >&2
    exit 1
fi

if [[ -z "${MPI_DIR}" ]]; then
    for candidate in /usr/local/mpi /usr/local/mpi_all/openmpi-*; do
        if [[ -x "${candidate}/bin/mpiexec" ]]; then
            MPI_DIR=${candidate}
        fi
    done
fi
if [[ -z "${MPI_DIR}" || ! -x "${MPI_DIR}/bin/mpiexec" ]]; then
    echo "Unable to find MPI. Set FFTM_HIP_MPI_DIR to its installation prefix." >&2
    exit 1
fi

if [[ -z "${HIP_ARCH}" ]]; then
    HIP_ARCH=$(rocminfo 2>/dev/null | awk '/Name:/ && $2 ~ /^gfx[0-9]+$/ { print $2; exit }')
fi
if [[ -z "${HIP_ARCH}" ]]; then
    echo "Unable to detect a ROCm GPU architecture. Set FFTM_HIP_ARCH." >&2
    exit 1
fi

GPU_COUNT=$("${ROCM_DIR}/bin/rocm-smi" --showid 2>/dev/null | awk '/GPU\[[0-9]+\]/ { count++ } END { print count + 0 }')
if [[ "${GPU_COUNT}" -lt 2 ]]; then
    echo "The full HIP matrix requires at least two visible GPUs; found ${GPU_COUNT}." >&2
    exit 1
fi

mkdir -p "${BUILD_DIR}" "${RESULT_DIR}"
export PATH="${MPI_DIR}/bin:${ROCM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${MPI_DIR}/lib:${ROCM_DIR}/lib:${LD_LIBRARY_PATH:-}"

make_args=(
    -C "${ROOT_DIR}/source/tests"
    "BUILD_FOLDER=${BUILD_DIR}"
    "mpi_dir=${MPI_DIR}"
    "ROCM_DIR=${ROCM_DIR}"
    "HIPCC=${HIPCC}"
    "HIP_ARCH=${HIP_ARCH}"
)
if [[ "${FORCE_REBUILD}" == 1 ]]; then
    make_args+=( -B )
fi

build_targets=()
if [[ "${TEST_HOST_STAGED}" == 1 ]]; then
    build_targets+=(
        test_fftm_3D_compare_hip_nca.bin
        test_fftm_4D_compare_hip_nca.bin
        test_4D_poisson_mpi_hip_nca.bin
    )
fi
if [[ "${TEST_DEVICE_AWARE}" == 1 ]]; then
    build_targets+=(
        test_fftm_3D_compare_hip.bin
        test_fftm_4D_compare_hip.bin
        test_4D_poisson_mpi_hip.bin
    )
fi

echo "Building HIP correctness targets: arch=${HIP_ARCH}; mpi=${MPI_DIR}; build=${BUILD_DIR}"
make "${make_args[@]}" "${build_targets[@]}"

MPIEXEC=${MPI_DIR}/bin/mpiexec
SUMMARY=${RESULT_DIR}/summary.tsv
printf 'case\tbackend\tranks\tstatus\tlog\n' > "${SUMMARY}"

run_case()
{
    local name=$1
    local backend=$2
    local ranks=$3
    local visible_devices=$4
    shift 4

    local log_file=${RESULT_DIR}/${name}.log
    local -a command=( "${MPIEXEC}" --bind-to none -n "${ranks}" "$@" )
    echo "[${name}] HIP_VISIBLE_DEVICES=${visible_devices:-all} ${command[*]}"

    local rc=0
    if [[ -n "${visible_devices}" ]]; then
        HIP_VISIBLE_DEVICES=${visible_devices} "${command[@]}" 2>&1 | tee "${log_file}" || rc=$?
    else
        "${command[@]}" 2>&1 | tee "${log_file}" || rc=$?
    fi
    if [[ "${rc}" -ne 0 ]] || ! grep -q '^INFO:.*PASSED' "${log_file}"; then
        echo "${name}: test did not report PASSED" >&2
        printf '%s\t%s\t%s\tfailed\t%s\n' "${name}" "${backend}" "${ranks}" "${log_file}" >> "${SUMMARY}"
        return 1
    fi
    printf '%s\t%s\t%s\tpassed\t%s\n' "${name}" "${backend}" "${ranks}" "${log_file}" >> "${SUMMARY}"
}

IFS=',' read -r -a two_gpu_modes <<< "${TWO_GPU_MODES}"

run_two_gpu_matrix()
{
    local backend=$1
    local name_prefix=$2
    local bin_3d=$3
    local bin_4d=$4

    local mode
    for mode in "${two_gpu_modes[@]}"; do
        if [[ -z "${mode}" ]]; then
            continue
        fi
        local mode_label=${mode//-/_}
        run_case "${name_prefix}_3d_2g_slab_${mode_label}" "${backend}" 2 "" \
            "${bin_3d}" --strategy slab-pencil --mode "${mode}" --threshold "${THRESHOLD}" \
            "${SIZE_3D}" "${SIZE_3D}" "${SIZE_3D}"
        run_case "${name_prefix}_3d_2g_pencil_${mode_label}" "${backend}" 2 "" \
            "${bin_3d}" --strategy pencil-pencil --mode "${mode}" --grid 1 2 --threshold "${THRESHOLD}" \
            "${SIZE_3D}" "${SIZE_3D}" "${SIZE_3D}"
        run_case "${name_prefix}_4d_2g_slab_${mode_label}" "${backend}" 2 "" \
            "${bin_4d}" --strategy slab-slab --mode "${mode}" --threshold "${THRESHOLD}" \
            "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}"
        run_case "${name_prefix}_4d_2g_pencil_${mode_label}" "${backend}" 2 "" \
            "${bin_4d}" --strategy pencil-pencil --mode "${mode}" --grid 1 1 2 --threshold "${THRESHOLD}" \
            "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}"
    done
}

run_native_spectral_wz_case()
{
    local backend=$1
    local name_prefix=$2
    local binary=$3

    run_case "${name_prefix}_4d_2g_slab_native_wz" "${backend}" 2 "" \
        "${binary}" --strategy slab-slab --mode p2p-waitany \
        --l2-threshold "${POISSON_L2_THRESHOLD}" --h1-threshold "${POISSON_H1_THRESHOLD}" \
        --use-4d-slab-native-xw-native-spectral-layout \
        --use-4d-native-xw-direct-layout \
        --use-4d-native-xw-chunked-transport \
        --4d-native-xw-chunk-window 1 \
        --use-4d-slab-native-work-area-alias \
        --use-4d-slab-native-wz-communication-layout \
        --4d-slab-native-wz-plan-concurrency 2 \
        --use-4d-slab-native-wz-ready-pipeline \
        "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}"
}

if [[ "${TEST_HOST_STAGED}" == 1 ]]; then
    BIN_3D_NCA=${BUILD_DIR}/test_fftm_3D_compare_hip_nca.bin
    BIN_4D_NCA=${BUILD_DIR}/test_fftm_4D_compare_hip_nca.bin

    for gpu in 0 1; do
        run_case "hip_nca_3d_gpu${gpu}" hip-nca 1 "${gpu}" \
            "${BIN_3D_NCA}" --strategy slab-pencil --mode alltoallv --threshold "${THRESHOLD}" \
            "${SIZE_3D}" "${SIZE_3D}" "${SIZE_3D}"
        run_case "hip_nca_4d_gpu${gpu}" hip-nca 1 "${gpu}" \
            "${BIN_4D_NCA}" --strategy slab-slab --mode alltoallv --threshold "${THRESHOLD}" \
            "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}" "${SIZE_4D}"
    done

    run_two_gpu_matrix hip-nca hip_nca "${BIN_3D_NCA}" "${BIN_4D_NCA}"
fi

if [[ "${TEST_DEVICE_AWARE}" == 1 ]]; then
    BIN_3D_CA=${BUILD_DIR}/test_fftm_3D_compare_hip.bin
    BIN_4D_CA=${BUILD_DIR}/test_fftm_4D_compare_hip.bin
    run_two_gpu_matrix hip-device-aware hip_device_aware "${BIN_3D_CA}" "${BIN_4D_CA}"
    run_native_spectral_wz_case \
        hip-device-aware hip_device_aware "${BUILD_DIR}/test_4D_poisson_mpi_hip.bin"
fi

echo "HIP correctness matrix passed. Summary: ${SUMMARY}"
