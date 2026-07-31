#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SOURCE_DIR=${1:-"${ROOT}/build/local_turbulence_adaptive_15T_cfl035_20260729"}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT=${2:-"${ROOT}/build/local_spacetime_4d_validation_${STAMP}"}
BUILD_DIR=${FFTM_TURBULENCE_BUILD_DIR:-/tmp/fftm_turbulence_spacetime_build}
CUDA_ARCH=${CUDA_ARCH:-"-gencode arch=compute_75,code=sm_75"}
MPIEXEC=${MPIEXEC:-/usr/local/mpi/bin/mpiexec}
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN=${PYTHON}
elif [[ -x /home/noctum/anaconda3/bin/python ]]; then
    PYTHON_BIN=/home/noctum/anaconda3/bin/python
else
    PYTHON_BIN=python3
fi

TARGET_SIZE=${FFTM_SPACETIME_TARGET_SIZE:-32}
FRAMES=${FFTM_SPACETIME_FRAMES:-24}
FRAME_OFFSET=${FFTM_SPACETIME_FRAME_OFFSET:-4}
MODE_MIN=${FFTM_SPACETIME_MODE_MIN:-1}
MODE_MAX=${FFTM_SPACETIME_MODE_MAX:-3}

if [[ ! -f "${SOURCE_DIR}/snapshots.csv" ]]; then
    echo "Missing source snapshot manifest: ${SOURCE_DIR}/snapshots.csv" >&2
    exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
    echo "Output path already exists: ${OUTPUT_ROOT}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/figures"
export MPLCONFIGDIR="${OUTPUT_ROOT}/.matplotlib"

make -C "${ROOT}/examples" \
    taylor_green_spacetime_4d.bin \
    BUILD_FOLDER="${BUILD_DIR}" \
    CUDA_ARCH="${CUDA_ARCH}"

"${PYTHON_BIN}" \
    "${ROOT}/examples/turbulence/scripts/prepare_spacetime_subset.py" \
    "${SOURCE_DIR}" \
    "${OUTPUT_ROOT}/subset" \
    --field omega_z \
    --frame-offset "${FRAME_OFFSET}" \
    --frames "${FRAMES}" \
    --target-size "${TARGET_SIZE}" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/prepare.log"

run_analysis()
{
    local label=$1
    local mode_min=$2
    local mode_max=$3
    local subtract_mean=$4
    local hann_window=$5

    FFTM_WRAP_PROCS_GPUS=1 "${MPIEXEC}" -n 1 \
        "${BUILD_DIR}/taylor_green_spacetime_4d.bin" \
        --input "${OUTPUT_ROOT}/subset" \
        --output "${OUTPUT_ROOT}/${label}" \
        --field omega_z \
        --size "${TARGET_SIZE}" \
        --frames "${FRAMES}" \
        --mode-min "${mode_min}" \
        --mode-max "${mode_max}" \
        --write-every 1 \
        --subtract-mean "${subtract_mean}" \
        --hann-window "${hann_window}" \
        2>&1 | tee "${OUTPUT_ROOT}/logs/${label}.log"
}

run_analysis roundtrip 0 "$(( FRAMES / 2 ))" 0 0
run_analysis low_modes "${MODE_MIN}" "${MODE_MAX}" 1 1

"${PYTHON_BIN}" \
    "${ROOT}/examples/turbulence/scripts/plot_spacetime_analysis.py" \
    --input "${OUTPUT_ROOT}/subset" \
    --roundtrip "${OUTPUT_ROOT}/roundtrip" \
    --filtered "${OUTPUT_ROOT}/low_modes" \
    --output-prefix "${OUTPUT_ROOT}/figures/spacetime_validation" \
    --slice-time 9.0 \
    --dpi 600 \
    --formats png,pdf \
    2>&1 | tee "${OUTPUT_ROOT}/logs/plot.log"

cat > "${OUTPUT_ROOT}/config.env" <<EOF
source_dir=${SOURCE_DIR}
target_size=${TARGET_SIZE}
frames=${FRAMES}
frame_offset=${FRAME_OFFSET}
mode_min=${MODE_MIN}
mode_max=${MODE_MAX}
build_dir=${BUILD_DIR}
python=${PYTHON_BIN}
mpiexec=${MPIEXEC}
EOF

printf 'Local 4D validation complete: %s\n' "${OUTPUT_ROOT}"
