#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BUILD_DIR=${FFTM_TURBULENCE_BUILD_DIR:-/tmp/fftm_turbulence_smoke_build}
WORK_DIR=$(mktemp -d /tmp/fftm_turbulence_smoke.XXXXXX)
trap 'rm -rf "${WORK_DIR}"' EXIT

CUDA_ARCH=${CUDA_ARCH:-"-gencode arch=compute_75,code=sm_75"}
MPIEXEC=${MPIEXEC:-/usr/local/mpi/bin/mpiexec}
PYTHON=${PYTHON:-python3}

make -C "${ROOT}/examples" \
    taylor_green_3d_autotuned.bin \
    BUILD_FOLDER="${BUILD_DIR}" \
    CUDA_ARCH="${CUDA_ARCH}"

run_case()
{
    local ranks=$1
    local output="${WORK_DIR}/r${ranks}"
    FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/taylor_green_3d_autotuned.bin" \
        --size 16 \
        --reynolds 100 \
        --dt 0.001 \
        --steps 4 \
        --cache "${WORK_DIR}/cache_r${ranks}.env" \
        --output "${output}" \
        --diagnostics-every 1 \
        --snapshot-every 1 \
        --viz-every 1 \
        --snapshot-size 8 \
        --write-initial-snapshot 1
    "${PYTHON}" "${ROOT}/examples/turbulence/tests/validate_snapshot.py" "${output}"
}

run_case 1
run_case 2

cmp "${WORK_DIR}/r1/omega_z_s00000004.raw" "${WORK_DIR}/r2/omega_z_s00000004.raw"
cmp \
    "${WORK_DIR}/r1/vorticity_magnitude_s00000004.raw" \
    "${WORK_DIR}/r2/vorticity_magnitude_s00000004.raw"
cmp \
    "${WORK_DIR}/r1/q_criterion_s00000004.raw" \
    "${WORK_DIR}/r2/q_criterion_s00000004.raw"

ADAPTIVE_OUTPUT="${WORK_DIR}/adaptive"
FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC}" -n 1 \
    "${BUILD_DIR}/taylor_green_3d_autotuned.bin" \
    --size 16 \
    --reynolds 100 \
    --dt 0.001 \
    --dt-max 0.02 \
    --dt-growth 1.1 \
    --target-cfl 0.3 \
    --cfl-fail 0.4 \
    --final-time 0.03 \
    --cache "${WORK_DIR}/cache_adaptive.env" \
    --output "${ADAPTIVE_OUTPUT}" \
    --diagnostics-every 1 \
    --snapshot-period 0.01 \
    --viz-period 0.01 \
    --snapshot-size 8 \
    --write-initial-snapshot 1
"${PYTHON}" "${ROOT}/examples/turbulence/tests/validate_snapshot.py" "${ADAPTIVE_OUTPUT}"
awk -F, 'NR > 1 && $5 > 0.3000001 { exit 1 }' \
    "${ADAPTIVE_OUTPUT}/adaptive_timesteps.csv"

run_4d_case()
{
    local ranks=$1
    local output="${WORK_DIR}/a${ranks}"
    FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC}" -n "${ranks}" \
        "${BUILD_DIR}/taylor_green_spacetime_4d.bin" \
        --input "${WORK_DIR}/r1" \
        --output "${output}" \
        --field omega_z \
        --size 8 \
        --frames 4 \
        --mode-min 1 \
        --mode-max 1 \
        --write-every 1
}

make -C "${ROOT}/examples" \
    taylor_green_spacetime_4d.bin \
    BUILD_FOLDER="${BUILD_DIR}" \
    CUDA_ARCH="${CUDA_ARCH}"

run_4d_case 1
run_4d_case 2

FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC}" -n 1 \
    "${BUILD_DIR}/taylor_green_spacetime_4d.bin" \
    --input "${ADAPTIVE_OUTPUT}" \
    --output "${WORK_DIR}/adaptive_4d" \
    --field omega_z \
    --size 8 \
    --frames 4 \
    --mode-min 1 \
    --mode-max 1 \
    --write-every 1

cmp \
    "${WORK_DIR}/a1/filtered_omega_z_m1_1_s00000000.raw" \
    "${WORK_DIR}/a2/filtered_omega_z_m1_1_s00000000.raw"
cmp \
    "${WORK_DIR}/a1/filtered_omega_z_m1_1_s00000003.raw" \
    "${WORK_DIR}/a2/filtered_omega_z_m1_1_s00000003.raw"

echo "Taylor-Green local MPI smoke test passed"
