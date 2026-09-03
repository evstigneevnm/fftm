#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${FFTM_READER_BUILD_DIR:-/tmp/fftm_reader_smoke_build}
WORK_DIR=$(mktemp -d /tmp/fftm_reader_smoke.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

BACKEND=${FFTM_READER_BACKEND:-cuda}
DEVICE_AWARE_MPI=${FFTM_READER_DEVICE_AWARE_MPI:-1}
MPIEXEC=${MPIEXEC:-/usr/local/mpi/bin/mpiexec}
read -r -a MPIEXEC_COMMAND <<< "${MPIEXEC}"

case "${BACKEND}" in
cuda)
    CUDA_ARCH=${CUDA_ARCH:-"-gencode arch=compute_75,code=sm_75"}
    make -C "${ROOT}/examples" \
        poisson_periodic_3d_autotuned.bin \
        poisson_periodic_4d_autotuned.bin \
        BUILD_FOLDER="${BUILD_DIR}" \
        CUDA_ARCH="${CUDA_ARCH}"
    BIN_3D="${BUILD_DIR}/poisson_periodic_3d_autotuned.bin"
    BIN_4D="${BUILD_DIR}/poisson_periodic_4d_autotuned.bin"
    DEVICE_AWARE_MPI=1
    ;;
hip)
    ROCM_DIR=${ROCM_DIR:-/opt/rocm}
    HIP_ARCH=${HIP_ARCH:-}
    if [[ "${DEVICE_AWARE_MPI}" == 1 ]]; then
        HIP_TARGET=hip
        HIP_SUFFIX=hip
    elif [[ "${DEVICE_AWARE_MPI}" == 0 ]]; then
        HIP_TARGET=hip-nca
        HIP_SUFFIX=hip_nca
    else
        echo "FFTM_READER_DEVICE_AWARE_MPI must be 0 or 1" >&2
        exit 2
    fi
    make -C "${ROOT}/examples/poisson" "${HIP_TARGET}" \
        BUILD_FOLDER="${BUILD_DIR}" \
        ROCM_DIR="${ROCM_DIR}" \
        HIP_ARCH="${HIP_ARCH}" \
        mpi_dir="${mpi_dir:-/usr/local/mpi}"
    BIN_3D="${BUILD_DIR}/poisson_periodic_3d_autotuned_${HIP_SUFFIX}.bin"
    BIN_4D="${BUILD_DIR}/poisson_periodic_4d_autotuned_${HIP_SUFFIX}.bin"
    ;;
*)
    echo "FFTM_READER_BACKEND must be cuda or hip" >&2
    exit 2
    ;;
esac

run_mpi()
{
    local ranks=$1
    shift
    FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC_COMMAND[@]}" -n "${ranks}" "$@"
}

run_3d()
{
    local ranks=$1
    local cache="${WORK_DIR}/poisson_3d_r${ranks}.env"
    local create_log="${WORK_DIR}/poisson_3d_r${ranks}_create.log"
    local reuse_log="${WORK_DIR}/poisson_3d_r${ranks}_reuse.log"
    FFTM_CPP_AUTOTUNE_MEASURE=0 \
        run_mpi "${ranks}" "${BIN_3D}" \
        16 16 16 "${cache}" 1 0 \
        | tee "${create_log}"
    grep -q 'poisson_periodic_3d_autotuned:' "${create_log}"
    grep -q 'source=created' "${create_log}"
    grep -q "backend=${BACKEND}" "${create_log}"
    grep -q "device_aware_mpi=${DEVICE_AWARE_MPI}" "${create_log}"
    grep -q 'rel_l2=' "${create_log}"

    FFTM_CPP_AUTOTUNE_MEASURE=0 \
        run_mpi "${ranks}" "${BIN_3D}" \
        16 16 16 "${cache}" 1 0 \
        | tee "${reuse_log}"
    grep -q 'source=cache' "${reuse_log}"
}

run_3d_measured()
{
    local ranks=2
    local cache="${WORK_DIR}/poisson_3d_measured_r${ranks}.env"
    local create_log="${WORK_DIR}/poisson_3d_measured_create.log"
    local reuse_log="${WORK_DIR}/poisson_3d_measured_reuse.log"
    FFTM_CPP_AUTOTUNE_MEASURE=1 \
    FFTM_CPP_AUTOTUNE_WARMUP=0 \
    FFTM_CPP_AUTOTUNE_TIMES=1 \
        run_mpi "${ranks}" "${BIN_3D}" \
        16 16 16 "${cache}" 1 0 \
        | tee "${create_log}"
    grep -q 'source=measured' "${create_log}"
    grep -q '^FFTM_AUTOTUNE_SOURCE=cpp-measured-v2$' "${cache}"
    grep -q '^FFTM_AUTOTUNE_CANDIDATE_COUNT=' "${cache}"
    grep -q "^FFTM_DIRECT_P2P_CUDA_AWARE=${DEVICE_AWARE_MPI}$" "${cache}"

    FFTM_CPP_AUTOTUNE_MEASURE=1 \
        run_mpi "${ranks}" "${BIN_3D}" \
        16 16 16 "${cache}" 1 0 \
        | tee "${reuse_log}"
    grep -q 'source=cache' "${reuse_log}"
}

run_4d()
{
    local ranks=$1
    local cache="${WORK_DIR}/poisson_4d_r${ranks}.env"
    local create_log="${WORK_DIR}/poisson_4d_r${ranks}_create.log"
    local reuse_log="${WORK_DIR}/poisson_4d_r${ranks}_reuse.log"
    FFTM_CPP_AUTOTUNE_MEASURE=1 \
    FFTM_CPP_AUTOTUNE_WARMUP=0 \
    FFTM_CPP_AUTOTUNE_TIMES=1 \
        run_mpi "${ranks}" "${BIN_4D}" 16 20 24 32 "${cache}" 1 0 \
        | tee "${create_log}"
    grep -q 'poisson_periodic_4d_autotuned:' "${create_log}"
    grep -q 'source=measured' "${create_log}"
    grep -q "backend=${BACKEND}" "${create_log}"
    grep -q "device_aware_mpi=${DEVICE_AWARE_MPI}" "${create_log}"
    grep -q 'rel_l2=' "${create_log}"
    grep -q '^FFTM_AUTOTUNE_SOURCE=cpp-measured-4d-v2$' "${cache}"
    grep -q '^FFTM_AUTOTUNE_ACCEPTED_SPECTRAL_LAYOUTS_4D=public-yzwx,native-xzwy$' "${cache}"
    grep -q 'SPECTRAL_LAYOUT_4D=public-yzwx' "${cache}"
    grep -q 'SPECTRAL_LAYOUT_4D=native-xzwy' "${cache}"

    FFTM_CPP_AUTOTUNE_MEASURE=1 \
        run_mpi "${ranks}" "${BIN_4D}" 16 20 24 32 "${cache}" 1 0 \
        | tee "${reuse_log}"
    grep -q 'source=cache' "${reuse_log}"
}

run_3d 1
run_3d 2
run_3d_measured
run_4d 1
run_4d 2

echo "FFTM ${BACKEND} reader example smoke test passed"
