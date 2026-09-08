#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK_DIR=$(mktemp -d /tmp/fftm_reader_smoke.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

make_args=(-C "${ROOT}/examples/poisson")
if [[ ${CONFIG_FILE+x} ]]; then
    make_args+=("CONFIG_FILE=${CONFIG_FILE}")
fi
if [[ ${FFTM_READER_BUILD_DIR+x} ]]; then
    make_args+=("BUILD_DIR=${FFTM_READER_BUILD_DIR}")
elif [[ ! ${CONFIG_FILE+x} && ! -f ${ROOT}/build_configs/config_local.inc && ! ${BUILD_DIR+x} ]]; then
    make_args+=("BUILD_DIR=/tmp/fftm_reader_smoke_build")
fi
if [[ ${FFTM_READER_BACKEND+x} ]]; then
    make_args+=("FFTM_BUILD_BACKEND=${FFTM_READER_BACKEND}")
fi
if [[ ${FFTM_READER_DEVICE_AWARE_MPI+x} ]]; then
    make_args+=("FFTM_DEVICE_AWARE_MPI=${FFTM_READER_DEVICE_AWARE_MPI}")
fi

# Read Make's resolved settings as key/value data, never as executable shell code.
resolved=$(make --no-print-directory -s "${make_args[@]}" print-config)
while IFS='=' read -r key value; do
    case "${key}" in
        BUILD_DIR|FFTM_BUILD_BACKEND|FFTM_DEVICE_AWARE_MPI|MPIEXEC|MPIEXEC_FLAGS|lib_mpi|lib_cuda|ROCM_LIB_DIR)
            printf -v "${key}" '%s' "${value}"
            ;;
    esac
done <<< "${resolved}"
BACKEND=${FFTM_BUILD_BACKEND}
DEVICE_AWARE_MPI=${FFTM_DEVICE_AWARE_MPI}
read -r -a MPIEXEC_COMMAND <<< "${MPIEXEC}"
read -r -a MPIEXEC_EXTRA <<< "${MPIEXEC_FLAGS}"
MPIEXEC_COMMAND+=("${MPIEXEC_EXTRA[@]}")
export LD_LIBRARY_PATH="${lib_mpi}:${lib_cuda}:${ROCM_LIB_DIR}:${LD_LIBRARY_PATH:-}"

case "${BACKEND}" in
cuda)
    CUDA_TARGET=cuda
    CUDA_SUFFIX=
    if [[ "${DEVICE_AWARE_MPI}" == 0 ]]; then
        CUDA_TARGET=cuda-nca
        CUDA_SUFFIX=_nca
    fi
    make "${make_args[@]}" "${CUDA_TARGET}"
    BIN_3D="${BUILD_DIR}/poisson_periodic_3d_autotuned${CUDA_SUFFIX}.bin"
    BIN_4D="${BUILD_DIR}/poisson_periodic_4d_autotuned${CUDA_SUFFIX}.bin"
    ;;
hip)
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
    make "${make_args[@]}" "${HIP_TARGET}"
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
