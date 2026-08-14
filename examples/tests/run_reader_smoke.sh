#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR=${FFTM_READER_BUILD_DIR:-/tmp/fftm_reader_smoke_build}
WORK_DIR=$(mktemp -d /tmp/fftm_reader_smoke.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

CUDA_ARCH=${CUDA_ARCH:-"-gencode arch=compute_75,code=sm_75"}
MPIEXEC=${MPIEXEC:-/usr/local/mpi/bin/mpiexec}

make -C "${ROOT}/examples" \
    poisson_periodic_3d_autotuned.bin \
    poisson_periodic_4d.bin \
    BUILD_FOLDER="${BUILD_DIR}" \
    CUDA_ARCH="${CUDA_ARCH}"

run_3d()
{
    local ranks=$1
    local cache="${WORK_DIR}/poisson_3d_r${ranks}.env"
    local create_log="${WORK_DIR}/poisson_3d_r${ranks}_create.log"
    local reuse_log="${WORK_DIR}/poisson_3d_r${ranks}_reuse.log"
    FFTM_WRAP_PROCS_GPUS=1 \
    FFTM_CPP_AUTOTUNE_MEASURE=0 \
        "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/poisson_periodic_3d_autotuned.bin" \
        16 16 16 "${cache}" 1 0 \
        | tee "${create_log}"
    grep -q 'poisson_periodic_3d_autotuned:' "${create_log}"
    grep -q 'source=created' "${create_log}"
    grep -q 'rel_l2=' "${create_log}"

    FFTM_WRAP_PROCS_GPUS=1 \
    FFTM_CPP_AUTOTUNE_MEASURE=0 \
        "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/poisson_periodic_3d_autotuned.bin" \
        16 16 16 "${cache}" 1 0 \
        | tee "${reuse_log}"
    grep -q 'poisson_periodic_3d_autotuned:' "${reuse_log}"
    grep -q 'source=cache' "${reuse_log}"
    grep -q 'rel_l2=' "${reuse_log}"
}

run_4d()
{
    local ranks=$1
    local log="${WORK_DIR}/poisson_4d_r${ranks}.log"
    FFTM_WRAP_PROCS_GPUS=1 \
        "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/poisson_periodic_4d.bin" 16 1 \
        | tee "${log}"
    grep -q 'poisson_periodic_4d:' "${log}"
    grep -q 'layout=native_xzwy' "${log}"
    grep -q 'rel_l2=' "${log}"
}

run_3d_measured()
{
    local ranks=2
    local cache="${WORK_DIR}/poisson_3d_measured_r${ranks}.env"
    local create_log="${WORK_DIR}/poisson_3d_measured_create.log"
    local reuse_log="${WORK_DIR}/poisson_3d_measured_reuse.log"
    FFTM_WRAP_PROCS_GPUS=1 \
    FFTM_CPP_AUTOTUNE_MEASURE=1 \
    FFTM_CPP_AUTOTUNE_WARMUP=0 \
    FFTM_CPP_AUTOTUNE_TIMES=1 \
        "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/poisson_periodic_3d_autotuned.bin" \
        16 16 16 "${cache}" 1 0 \
        | tee "${create_log}"
    grep -q 'source=measured' "${create_log}"
    grep -q '^FFTM_AUTOTUNE_SOURCE=cpp-measured-v2$' "${cache}"
    grep -q '^FFTM_AUTOTUNE_CANDIDATE_COUNT=' "${cache}"

    FFTM_WRAP_PROCS_GPUS=1 \
    FFTM_CPP_AUTOTUNE_MEASURE=1 \
        "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/poisson_periodic_3d_autotuned.bin" \
        16 16 16 "${cache}" 1 0 \
        | tee "${reuse_log}"
    grep -q 'source=cache' "${reuse_log}"
}

run_3d 1
run_3d 2
run_3d_measured
run_4d 1
run_4d 2

echo "FFTM reader example smoke test passed"
