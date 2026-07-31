#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LAUNCHER="${ROOT}/examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh"
IMAGE="${FFTM_TURBULENCE_CONTAINER_IMAGE:-${FFTM_CONTAINER_IMAGE:-}}"
: "${IMAGE:?Set FFTM_TURBULENCE_CONTAINER_IMAGE to the rebuilt FFTM SQSH}"

USE_SINGLE_ALLOCATION="${FFTM_TURBULENCE_USE_SINGLE_ALLOCATION:-0}"
if [[ "${USE_SINGLE_ALLOCATION}" != "0" && "${USE_SINGLE_ALLOCATION}" != "1" ]]; then
    echo "FFTM_TURBULENCE_USE_SINGLE_ALLOCATION must be 0 or 1." >&2
    exit 2
fi
START_STAGE="${FFTM_TURBULENCE_PIPELINE_START_STAGE:-1}"
if [[ ! "${START_STAGE}" =~ ^[1-6]$ ]]; then
    echo "FFTM_TURBULENCE_PIPELINE_START_STAGE must be an integer from 1 to 6." >&2
    exit 2
fi
SHARED_ALLOCATION="${FFTM_TURBULENCE_SHARED_ALLOCATION:-0}"
if [[ "${SHARED_ALLOCATION}" != "0" && "${SHARED_ALLOCATION}" != "1" ]]; then
    echo "FFTM_TURBULENCE_SHARED_ALLOCATION must be 0 or 1." >&2
    exit 2
fi
ALLOW_RESUME_IMAGE_MISMATCH="${FFTM_TURBULENCE_ALLOW_RESUME_IMAGE_MISMATCH:-0}"
if [[ "${ALLOW_RESUME_IMAGE_MISMATCH}" != "0" &&
      "${ALLOW_RESUME_IMAGE_MISMATCH}" != "1" ]]; then
    echo "FFTM_TURBULENCE_ALLOW_RESUME_IMAGE_MISMATCH must be 0 or 1." >&2
    exit 2
fi
if [[ "${ALLOW_RESUME_IMAGE_MISMATCH}" == "1" &&
      "${START_STAGE}" != "6" ]]; then
    echo "A resume image mismatch is allowed only for visualization Stage 6." >&2
    exit 2
fi
GPU_CPUS_PER_TASK="${FFTM_TURBULENCE_GPU_CPUS_PER_TASK:-1}"
PLOT_CPUS_PER_TASK="${FFTM_TURBULENCE_PLOT_CPUS_PER_TASK:-16}"
ALLOCATION_CPUS_PER_TASK="${FFTM_TURBULENCE_ALLOCATION_CPUS_PER_TASK:-2}"
for VALUE_NAME in \
    GPU_CPUS_PER_TASK \
    PLOT_CPUS_PER_TASK \
    ALLOCATION_CPUS_PER_TASK; do
    VALUE="${!VALUE_NAME}"
    if [[ ! "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${VALUE_NAME} must be a positive integer, got '${VALUE}'." >&2
        exit 2
    fi
done
if (( START_STAGE < 4 &&
      8 * ALLOCATION_CPUS_PER_TASK < PLOT_CPUS_PER_TASK )); then
    printf 'The shared allocation provides %d CPUs, but plotting requires %d.\n' \
        "$(( 8 * ALLOCATION_CPUS_PER_TASK ))" "${PLOT_CPUS_PER_TASK}" >&2
    echo "Increase FFTM_TURBULENCE_ALLOCATION_CPUS_PER_TASK." >&2
    exit 2
fi
if (( START_STAGE < 4 &&
      ALLOCATION_CPUS_PER_TASK < GPU_CPUS_PER_TASK )); then
    echo "FFTM_TURBULENCE_ALLOCATION_CPUS_PER_TASK must be at least " \
         "FFTM_TURBULENCE_GPU_CPUS_PER_TASK." >&2
    exit 2
fi
if [[ "${USE_SINGLE_ALLOCATION}" == "1" && -z "${SLURM_JOB_ID:-}" ]]; then
    ALLOCATION_TIME="${FFTM_TURBULENCE_ALLOCATION_TIME:-08:00:00}"
    SALLOC_EXTRA_ARGS="${FFTM_TURBULENCE_SALLOC_EXTRA_ARGS:---exclude=cn13}"
    read -r -a SALLOC_EXTRA_ARRAY <<< "${SALLOC_EXTRA_ARGS}"
    if (( START_STAGE >= 4 )); then
        ALLOCATION_TASKS=1
        ALLOCATION_GPUS=1
        ALLOCATION_GPUS_PER_NODE=1
        ALLOCATION_STEP_CPUS="${PLOT_CPUS_PER_TASK}"
        SHARED_ALLOCATION=0
    else
        ALLOCATION_TASKS=8
        ALLOCATION_GPUS=8
        ALLOCATION_GPUS_PER_NODE=8
        ALLOCATION_STEP_CPUS="${ALLOCATION_CPUS_PER_TASK}"
        SHARED_ALLOCATION=1
    fi
    printf 'Requesting one %d-GPU/%d-CPU allocation from pipeline Stage %d.\n' \
        "${ALLOCATION_GPUS}" \
        "$(( ALLOCATION_TASKS * ALLOCATION_STEP_CPUS ))" \
        "${START_STAGE}"
    export FFTM_TURBULENCE_USE_SINGLE_ALLOCATION=0
    export FFTM_TURBULENCE_SHARED_ALLOCATION="${SHARED_ALLOCATION}"
    exec salloc \
        "${SALLOC_EXTRA_ARRAY[@]}" \
        -N 1 \
        -n "${ALLOCATION_TASKS}" \
        --ntasks-per-node="${ALLOCATION_TASKS}" \
        --cpus-per-task="${ALLOCATION_STEP_CPUS}" \
        -G "${ALLOCATION_GPUS}" \
        --gpus-per-node="${ALLOCATION_GPUS_PER_NODE}" \
        --time="${ALLOCATION_TIME}" \
        bash "${ROOT}/examples/turbulence/scripts/run_512_256x64_cluster_validation.sh"
fi

if [[ "${SHARED_ALLOCATION}" == "1" ]]; then
    PLOT_GPUS_PER_NODE=8
else
    PLOT_GPUS_PER_NODE=1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_ROOT="${FFTM_TURBULENCE_PIPELINE_ROOT:-${ROOT}/data_tg512_4d256x64_${STAMP}}"
if (( START_STAGE == 1 )); then
    if [[ -e "${PIPELINE_ROOT}" ]]; then
        echo "Pipeline output already exists: ${PIPELINE_ROOT}" >&2
        exit 2
    fi
    mkdir -p "${PIPELINE_ROOT}"
else
    if [[ ! -d "${PIPELINE_ROOT}" ]]; then
        echo "Cannot resume: pipeline output does not exist: ${PIPELINE_ROOT}" >&2
        exit 2
    fi
fi
PIPELINE_ROOT="$(cd "${PIPELINE_ROOT}" && pwd -P)"

SIM_ROOT="${PIPELINE_ROOT}/simulation_run"
ROUNDTRIP_ROOT="${PIPELINE_ROOT}/roundtrip_run"
FILTERED_ROOT="${PIPELINE_ROOT}/filtered_run"
VALIDATION_ROOT="${PIPELINE_ROOT}/spacetime_validation"
TG_VALIDATION_ROOT="${PIPELINE_ROOT}/taylor_green_validation"
FLOW_RENDER_ROOT="${PIPELINE_ROOT}/flow_visualization"
SIMULATION_DIR="${SIM_ROOT}/simulation"

export FFTM_TURBULENCE_CONTAINER_IMAGE="${IMAGE}"
export FFTM_TURBULENCE_NODES=1
export FFTM_TURBULENCE_GPUS_PER_NODE=8
export FFTM_TURBULENCE_AFFINITY=auto
export FFTM_TURBULENCE_SRUN_EXTRA_ARGS="${FFTM_TURBULENCE_SRUN_EXTRA_ARGS:---exclude=cn13 --distribution=block:block --kill-on-bad-exit=1}"
export FFTM_TURBULENCE_DRY_RUN=0

if (( START_STAGE == 1 )); then
    CONFIG_PATH="${PIPELINE_ROOT}/pipeline.env"
else
    CONFIG_PATH="${PIPELINE_ROOT}/resume_stage${START_STAGE}_${STAMP}.env"
fi
cat > "${CONFIG_PATH}" <<EOF
container_image=${IMAGE}
pipeline_root=${PIPELINE_ROOT}
start_stage=${START_STAGE}
simulation_size=512
analysis_size=256
analysis_frames=64
snapshot_period=0.25
final_time=15.75
target_cfl=0.35
cfl_fail=0.4
filtered_modes=1:4
gpu_cpus_per_task=${GPU_CPUS_PER_TASK}
plot_cpus_per_task=${PLOT_CPUS_PER_TASK}
allocation_cpus_per_task=${ALLOCATION_CPUS_PER_TASK}
shared_allocation=${SHARED_ALLOCATION}
plot_gpus_per_node=${PLOT_GPUS_PER_NODE}
allow_resume_image_mismatch=${ALLOW_RESUME_IMAGE_MISMATCH}
srun_extra_args=${FFTM_TURBULENCE_SRUN_EXTRA_ARGS}
EOF
if (( START_STAGE == 1 )); then
    sha256sum "${IMAGE}" > "${PIPELINE_ROOT}/container.sha256"
elif [[ -s "${PIPELINE_ROOT}/container.sha256" ]]; then
    EXPECTED_IMAGE_HASH="$(awk 'NR == 1 { print $1 }' "${PIPELINE_ROOT}/container.sha256")"
    CURRENT_IMAGE_HASH="$(sha256sum "${IMAGE}" | awk '{ print $1 }')"
    if [[ "${EXPECTED_IMAGE_HASH}" != "${CURRENT_IMAGE_HASH}" ]]; then
        if [[ "${ALLOW_RESUME_IMAGE_MISMATCH}" != "1" ]]; then
            echo "Cannot resume with a different container image." >&2
            echo "Expected ${EXPECTED_IMAGE_HASH}, got ${CURRENT_IMAGE_HASH}." >&2
            exit 2
        fi
        printf 'WARNING: Stage 6 uses updated rendering image %s; numerical image was %s.\n' \
            "${CURRENT_IMAGE_HASH}" "${EXPECTED_IMAGE_HASH}" >&2
    fi
    sha256sum "${IMAGE}" > \
        "${PIPELINE_ROOT}/resume_stage${START_STAGE}_${STAMP}.container.sha256"
fi

if (( START_STAGE <= 1 )); then
    echo "Stage 1/6: 512^3 Taylor-Green simulation"
FFTM_TURBULENCE_DATA_DIR="${SIM_ROOT}" \
FFTM_TURBULENCE_GPUS=8 \
FFTM_TURBULENCE_SIZE=512 \
FFTM_TURBULENCE_REYNOLDS=1600 \
FFTM_TURBULENCE_DT=0.001 \
FFTM_TURBULENCE_DT_MIN=0 \
FFTM_TURBULENCE_DT_MAX=0.01 \
FFTM_TURBULENCE_DT_GROWTH=1.1 \
FFTM_TURBULENCE_TARGET_CFL=0.35 \
FFTM_TURBULENCE_CFL_FAIL=0.4 \
FFTM_TURBULENCE_FINAL_TIME=15.75 \
FFTM_TURBULENCE_STEPS=0 \
FFTM_TURBULENCE_DIAGNOSTICS_EVERY=5 \
FFTM_TURBULENCE_SNAPSHOT_EVERY=0 \
FFTM_TURBULENCE_VIZ_EVERY=0 \
FFTM_TURBULENCE_SNAPSHOT_PERIOD=0.25 \
FFTM_TURBULENCE_VIZ_PERIOD=0.5 \
FFTM_TURBULENCE_SNAPSHOT_SIZE=256 \
FFTM_TURBULENCE_WRITE_INITIAL_SNAPSHOT=1 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=06:00:00 \
"${LAUNCHER}" simulate
else
    echo "Stage 1/6: reusing existing simulation"
fi

MANIFEST="${SIMULATION_DIR}/snapshots.csv"
if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing simulation snapshot manifest: ${MANIFEST}" >&2
    exit 1
fi

awk -F, '
    $3 == "omega_z" {
        count += 1
        current = $2 + 0
        if (count == 1) {
            first = current
        } else {
            spacing = current - previous
            if (count == 2)
                reference_spacing = spacing
            difference = spacing - reference_spacing
            if (difference < 0)
                difference = -difference
            if (difference > 1.0e-10)
                nonuniform = 1
        }
        previous = current
        last = current
    }
    END {
        printf "SNAPSHOT_CHECK frames=%d first=%.17g last=%.17g dt=%.17g nonuniform=%d\n",
               count, first, last, reference_spacing, nonuniform
        first_error = first
        if (first_error < 0)
            first_error = -first_error
        last_error = last - 15.75
        if (last_error < 0)
            last_error = -last_error
        if (count != 64 || nonuniform || first_error > 1.0e-10 ||
            last_error > 1.0e-10 || reference_spacing != 0.25)
            exit 1
    }
' "${MANIFEST}"

EXPECTED_SNAPSHOT_BYTES=$(( 256 * 256 * 256 * 4 ))
while IFS= read -r filename; do
    path="${SIMULATION_DIR}/${filename}"
    if [[ ! -f "${path}" ]]; then
        echo "Missing snapshot payload: ${path}" >&2
        exit 1
    fi
    actual_bytes=$(stat -c %s "${path}")
    if (( actual_bytes != EXPECTED_SNAPSHOT_BYTES )); then
        echo "Wrong snapshot size: ${path}: ${actual_bytes}, expected ${EXPECTED_SNAPSHOT_BYTES}" >&2
        exit 1
    fi
done < <(awk -F, '$3 == "omega_z" { print $4 }' "${MANIFEST}")

if (( START_STAGE <= 2 )); then
    echo "Stage 2/6: full-band 256^3 x 64 round trip"
FFTM_TURBULENCE_DATA_DIR="${ROUNDTRIP_ROOT}" \
FFTM_TURBULENCE_GPUS=8 \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ANALYSIS_SIZE=256 \
FFTM_TURBULENCE_ANALYSIS_FRAMES=64 \
FFTM_TURBULENCE_FRAME_OFFSET=0 \
FFTM_TURBULENCE_MODE_MIN=0 \
FFTM_TURBULENCE_MODE_MAX=32 \
FFTM_TURBULENCE_SPATIAL_CUTOFF=0 \
FFTM_TURBULENCE_WRITE_EVERY=1 \
FFTM_TURBULENCE_FIELD=omega_z \
FFTM_TURBULENCE_SUBTRACT_MEAN=0 \
FFTM_TURBULENCE_HANN_WINDOW=0 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=01:00:00 \
"${LAUNCHER}" analyze
else
    echo "Stage 2/6: reusing existing round trip"
fi

if (( START_STAGE <= 3 )); then
    echo "Stage 3/6: Hann-windowed temporal modes 1-4"
FFTM_TURBULENCE_DATA_DIR="${FILTERED_ROOT}" \
FFTM_TURBULENCE_GPUS=8 \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ANALYSIS_SIZE=256 \
FFTM_TURBULENCE_ANALYSIS_FRAMES=64 \
FFTM_TURBULENCE_FRAME_OFFSET=0 \
FFTM_TURBULENCE_MODE_MIN=1 \
FFTM_TURBULENCE_MODE_MAX=4 \
FFTM_TURBULENCE_SPATIAL_CUTOFF=0 \
FFTM_TURBULENCE_WRITE_EVERY=1 \
FFTM_TURBULENCE_FIELD=omega_z \
FFTM_TURBULENCE_SUBTRACT_MEAN=1 \
FFTM_TURBULENCE_HANN_WINDOW=1 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=01:00:00 \
"${LAUNCHER}" analyze
else
    echo "Stage 3/6: reusing existing filtered analysis"
fi

if (( START_STAGE <= 4 )); then
    echo "Stage 4/6: independent 4D numerical validation and paper figures"
FFTM_TURBULENCE_DATA_DIR="${VALIDATION_ROOT}" \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE="${PLOT_GPUS_PER_NODE}" \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ROUNDTRIP_DIR="${ROUNDTRIP_ROOT}/analysis" \
FFTM_TURBULENCE_FILTERED_DIR="${FILTERED_ROOT}/analysis" \
FFTM_TURBULENCE_SPACETIME_VALIDATION_PREFIX=spacetime_validation \
FFTM_TURBULENCE_SPACETIME_SLICE_TIME=9.0 \
FFTM_TURBULENCE_SPACETIME_VALIDATION_DPI=600 \
FFTM_TURBULENCE_SPACETIME_VALIDATION_FORMATS=png,pdf \
FFTM_TURBULENCE_CPUS_PER_TASK="${PLOT_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=00:30:00 \
"${LAUNCHER}" spacetime-validate
else
    echo "Stage 4/6: reusing existing numerical validation"
fi

METRICS="${VALIDATION_ROOT}/spacetime_validation_metrics.json"
if [[ ! -s "${METRICS}" ]] || ! grep -q '"validation_status": "passed"' "${METRICS}"; then
    echo "4D numerical validation did not pass: ${METRICS}" >&2
    exit 1
fi

if (( START_STAGE <= 5 )); then
    echo "Stage 5/6: Taylor-Green literature validation figure"
FFTM_TURBULENCE_DATA_DIR="${TG_VALIDATION_ROOT}" \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE="${PLOT_GPUS_PER_NODE}" \
FFTM_TURBULENCE_DIAGNOSTICS="${SIMULATION_DIR}/diagnostics.csv" \
FFTM_TURBULENCE_VALIDATION_PREFIX=taylor_green_validation \
FFTM_TURBULENCE_VALIDATION_DPI=600 \
FFTM_TURBULENCE_VALIDATION_FORMATS=png,pdf,svg \
FFTM_TURBULENCE_CPUS_PER_TASK="${PLOT_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=00:30:00 \
"${LAUNCHER}" validate
else
    echo "Stage 5/6: reusing existing Taylor-Green validation"
fi

echo "Stage 6/6: 600-DPI vorticity and Q-criterion snapshot series"
FFTM_TURBULENCE_DATA_DIR="${FLOW_RENDER_ROOT}" \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE="${PLOT_GPUS_PER_NODE}" \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_RENDER_FIELDS=vorticity_magnitude,q_criterion \
FFTM_TURBULENCE_RENDER_STRIDE=1 \
FFTM_TURBULENCE_RENDER_DPI=600 \
FFTM_TURBULENCE_RENDER_WIDTH_INCHES=7.2 \
FFTM_TURBULENCE_RENDER_HEIGHT_INCHES=5.4 \
FFTM_TURBULENCE_RENDER_CAMERA_ZOOM=0.8 \
FFTM_TURBULENCE_RENDER_CONTACT_SHEET_COLUMNS=4 \
FFTM_TURBULENCE_CPUS_PER_TASK="${PLOT_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME=02:00:00 \
"${LAUNCHER}" visualize-series

RENDER_MANIFEST="${FLOW_RENDER_ROOT}/figures/renders.csv"
if [[ ! -s "${RENDER_MANIFEST}" ]]; then
    echo "Missing flow-render manifest: ${RENDER_MANIFEST}" >&2
    exit 1
fi
RENDER_COUNT="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${RENDER_MANIFEST}")"
if (( RENDER_COUNT != 64 )); then
    echo "Expected 64 rendered flow snapshots, got ${RENDER_COUNT}." >&2
    exit 1
fi
while IFS= read -r RELATIVE_OUTPUT; do
    if [[ ! -s "${FLOW_RENDER_ROOT}/figures/${RELATIVE_OUTPUT}" ]]; then
        echo "Missing rendered flow snapshot: ${RELATIVE_OUTPUT}" >&2
        exit 1
    fi
done < <(awk -F, 'NR > 1 { print $5 }' "${RENDER_MANIFEST}")
for REQUIRED_RENDER_OUTPUT in \
    "${FLOW_RENDER_ROOT}/figures/renders.json" \
    "${FLOW_RENDER_ROOT}/figures/vorticity_contact_sheet.png" \
    "${FLOW_RENDER_ROOT}/figures/q_contact_sheet.png"; do
    if [[ ! -s "${REQUIRED_RENDER_OUTPUT}" ]]; then
        echo "Missing flow-render artifact: ${REQUIRED_RENDER_OUTPUT}" >&2
        exit 1
    fi
done

printf 'Taylor-Green/4D validation pipeline passed: %s\n' "${PIPELINE_ROOT}"
printf '4D metrics: %s\n' "${METRICS}"
printf 'Flow snapshots: %s\n' "${FLOW_RENDER_ROOT}/figures"
