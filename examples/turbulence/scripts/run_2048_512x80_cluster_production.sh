#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
LAUNCHER="${ROOT}/examples/turbulence/scripts/run_slurm_pyxis_turbulence.sh"
IMAGE="${FFTM_TURBULENCE_CONTAINER_IMAGE:-${FFTM_CONTAINER_IMAGE:-}}"
: "${IMAGE:?Set FFTM_TURBULENCE_CONTAINER_IMAGE to the rebuilt FFTM SQSH}"

START_STAGE="${FFTM_TURBULENCE_PIPELINE_START_STAGE:-1}"
STOP_STAGE="${FFTM_TURBULENCE_PIPELINE_STOP_STAGE:-6}"
if [[ ! "${START_STAGE}" =~ ^[1-6]$ ||
      ! "${STOP_STAGE}" =~ ^[1-6]$ ||
      "${START_STAGE}" -gt "${STOP_STAGE}" ]]; then
    echo "Pipeline start/stop stages must satisfy 1 <= start <= stop <= 6." >&2
    exit 2
fi

GPUS_PER_NODE="${FFTM_TURBULENCE_GPUS_PER_NODE:-8}"
SIM_GPUS="${FFTM_TURBULENCE_PRODUCTION_SIM_GPUS:-64}"
ANALYSIS_GPUS="${FFTM_TURBULENCE_PRODUCTION_ANALYSIS_GPUS:-8}"
GPU_CPUS_PER_TASK="${FFTM_TURBULENCE_GPU_CPUS_PER_TASK:-1}"
REPORT_CPUS_PER_TASK="${FFTM_TURBULENCE_REPORT_CPUS_PER_TASK:-128}"
SIM_SRUN_TIME="${FFTM_TURBULENCE_SIM_SRUN_TIME:-18:00:00}"
ANALYSIS_SRUN_TIME="${FFTM_TURBULENCE_ANALYSIS_SRUN_TIME:-03:00:00}"
REPORT_SRUN_TIME="${FFTM_TURBULENCE_REPORT_SRUN_TIME:-04:00:00}"
RENDER_SRUN_TIME="${FFTM_TURBULENCE_RENDER_SRUN_TIME:-04:00:00}"
SPACETIME_Z_CHUNK="${FFTM_TURBULENCE_SPACETIME_Z_CHUNK:-2}"
SPACETIME_SLICE_TIME="${FFTM_TURBULENCE_SPACETIME_SLICE_TIME:-9.0}"
SRUN_EXTRA_ARGS="${FFTM_TURBULENCE_SRUN_EXTRA_ARGS:---exclude=cn13 --distribution=block:block --kill-on-bad-exit=1 --mem=0}"

for NAME in \
    GPUS_PER_NODE \
    SIM_GPUS \
    ANALYSIS_GPUS \
    GPU_CPUS_PER_TASK \
    REPORT_CPUS_PER_TASK \
    SPACETIME_Z_CHUNK; do
    VALUE="${!NAME}"
    if [[ ! "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${NAME} must be a positive integer, got '${VALUE}'." >&2
        exit 2
    fi
done
if (( GPUS_PER_NODE != 8 )); then
    echo "The production HCA-affinity path requires 8 GPUs per node." >&2
    exit 2
fi
if (( SIM_GPUS % GPUS_PER_NODE != 0 ||
      ANALYSIS_GPUS % GPUS_PER_NODE != 0 )); then
    echo "Simulation and analysis GPU counts must be divisible by 8." >&2
    exit 2
fi
if (( SIM_GPUS < 32 )); then
    echo "2048^3 production requires at least 32 A100-80GB GPUs." >&2
    exit 2
fi

SIM_NODES=$(( SIM_GPUS / GPUS_PER_NODE ))
ANALYSIS_NODES=$(( ANALYSIS_GPUS / GPUS_PER_NODE ))
STAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_ROOT="${FFTM_TURBULENCE_PIPELINE_ROOT:-${ROOT}/data_tg2048_4d512x80_${STAMP}}"
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
if (( START_STAGE == 1 )); then
    MIN_FREE_GIB="${FFTM_TURBULENCE_MIN_FREE_GIB:-60}"
    if [[ ! "${MIN_FREE_GIB}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FFTM_TURBULENCE_MIN_FREE_GIB must be a positive integer." >&2
        exit 2
    fi
    AVAILABLE_BYTES="$(df -PB1 "${PIPELINE_ROOT}" | awk 'NR == 2 { print $4 }')"
    REQUIRED_BYTES=$(( MIN_FREE_GIB * 1024 * 1024 * 1024 ))
    if [[ ! "${AVAILABLE_BYTES}" =~ ^[0-9]+$ ]] ||
       (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
        echo "The production pipeline requires at least ${MIN_FREE_GIB} GiB free " \
             "under ${PIPELINE_ROOT}." >&2
        exit 1
    fi
fi

SIM_ROOT="${PIPELINE_ROOT}/simulation_run"
ROUNDTRIP_ROOT="${PIPELINE_ROOT}/roundtrip_run"
FILTERED_ROOT="${PIPELINE_ROOT}/filtered_run"
SPACETIME_ROOT="${PIPELINE_ROOT}/spacetime_production"
TG_VALIDATION_ROOT="${PIPELINE_ROOT}/taylor_green_validation"
FLOW_RENDER_ROOT="${PIPELINE_ROOT}/flow_visualization"
SIMULATION_DIR="${SIM_ROOT}/simulation"
VISUALIZATION_DIR="${SIMULATION_DIR}/visualization"

export FFTM_TURBULENCE_CONTAINER_IMAGE="${IMAGE}"
export FFTM_TURBULENCE_SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS}"
export FFTM_TURBULENCE_DRY_RUN=0

CONFIG_PATH="${PIPELINE_ROOT}/pipeline_stage${START_STAGE}_${STOP_STAGE}_${STAMP}.env"
cat > "${CONFIG_PATH}" <<EOF
container_image=${IMAGE}
pipeline_root=${PIPELINE_ROOT}
start_stage=${START_STAGE}
stop_stage=${STOP_STAGE}
simulation_shape=2048x2048x2048
analysis_shape=512x512x512x80
simulation_gpus=${SIM_GPUS}
simulation_nodes=${SIM_NODES}
analysis_gpus=${ANALYSIS_GPUS}
analysis_nodes=${ANALYSIS_NODES}
gpus_per_node=${GPUS_PER_NODE}
snapshot_period=0.2
visualization_period=0.5
final_time=15.8
analysis_snapshot_size=512
visualization_snapshot_size=256
target_cfl=0.35
cfl_fail=0.4
filtered_modes=1:4
write_every=10
streaming_z_chunk=${SPACETIME_Z_CHUNK}
streaming_fft_workers=${REPORT_CPUS_PER_TASK}
estimated_payload_gib=52
srun_extra_args=${SRUN_EXTRA_ARGS}
EOF

if (( START_STAGE == 1 )); then
    sha256sum "${IMAGE}" > "${PIPELINE_ROOT}/container.sha256"
elif [[ -s "${PIPELINE_ROOT}/container.sha256" ]]; then
    EXPECTED_HASH="$(awk 'NR == 1 { print $1 }' "${PIPELINE_ROOT}/container.sha256")"
    CURRENT_HASH="$(sha256sum "${IMAGE}" | awk '{ print $1 }')"
    if [[ "${EXPECTED_HASH}" != "${CURRENT_HASH}" ]]; then
        echo "Cannot resume with a different container image." >&2
        echo "Expected ${EXPECTED_HASH}, got ${CURRENT_HASH}." >&2
        exit 2
    fi
fi

run_stage()
{
    local stage="$1"
    (( START_STAGE <= stage && stage <= STOP_STAGE ))
}

validate_uniform_series()
{
    local manifest="$1"
    local field="$2"
    local expected_count="$3"
    local expected_first="$4"
    local expected_last="$5"
    local expected_dt="$6"

    awk -F, \
        -v selected_field="${field}" \
        -v expected_count="${expected_count}" \
        -v expected_first="${expected_first}" \
        -v expected_last="${expected_last}" \
        -v expected_dt="${expected_dt}" '
        $3 == selected_field {
            count += 1
            current = $2 + 0
            if (count == 1) {
                first = current
            } else {
                spacing = current - previous
                difference = spacing - expected_dt
                if (difference < 0)
                    difference = -difference
                if (difference > 1.0e-10)
                    nonuniform = 1
            }
            previous = current
            last = current
        }
        END {
            printf "SERIES_CHECK field=%s frames=%d first=%.17g last=%.17g dt=%.17g nonuniform=%d\n",
                   selected_field, count, first, last, expected_dt, nonuniform
            first_error = first - expected_first
            if (first_error < 0)
                first_error = -first_error
            last_error = last - expected_last
            if (last_error < 0)
                last_error = -last_error
            if (count != expected_count || nonuniform ||
                first_error > 1.0e-10 || last_error > 1.0e-10)
                exit 1
        }
    ' "${manifest}"
}

validate_payloads()
{
    local directory="$1"
    local manifest="$2"
    local field="$3"
    local size="$4"
    local expected_bytes=$(( size * size * size * 4 ))

    while IFS= read -r filename; do
        local path="${directory}/${filename}"
        if [[ ! -f "${path}" ]]; then
            echo "Missing snapshot payload: ${path}" >&2
            exit 1
        fi
        local actual_bytes
        actual_bytes=$(stat -c %s "${path}")
        if (( actual_bytes != expected_bytes )); then
            echo "Wrong snapshot size: ${path}: ${actual_bytes}, expected ${expected_bytes}" >&2
            exit 1
        fi
    done < <(awk -F, -v selected_field="${field}" \
        '$3 == selected_field { print $4 }' "${manifest}")
}

finish_if_stopped()
{
    local stage="$1"
    if (( STOP_STAGE == stage )); then
        printf 'Taylor-Green production pipeline stopped successfully after Stage %d: %s\n' \
            "${stage}" "${PIPELINE_ROOT}"
        exit 0
    fi
}

validate_analysis_directory()
{
    local analysis_directory="$1"
    for required in \
        "${analysis_directory}/analysis.json" \
        "${analysis_directory}/layout.json" \
        "${analysis_directory}/snapshots.csv" \
        "${analysis_directory}/selected_frames.csv"; do
        if [[ ! -s "${required}" ]]; then
            echo "Missing 4D analysis artifact: ${required}" >&2
            exit 1
        fi
    done
    local output_count
    local selected_count
    output_count="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' \
        "${analysis_directory}/snapshots.csv")"
    selected_count="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' \
        "${analysis_directory}/selected_frames.csv")"
    if (( output_count != 8 || selected_count != 80 )); then
        echo "Expected 8 sparse outputs and 80 selected frames in ${analysis_directory}; " \
             "got ${output_count} and ${selected_count}." >&2
        exit 1
    fi
    local expected_bytes=$(( 512 * 512 * 512 * 4 ))
    while IFS= read -r filename; do
        local path="${analysis_directory}/${filename}"
        if [[ ! -f "${path}" ]]; then
            echo "Missing reconstructed frame: ${path}" >&2
            exit 1
        fi
        local actual_bytes
        actual_bytes=$(stat -c %s "${path}")
        if (( actual_bytes != expected_bytes )); then
            echo "Wrong reconstructed frame size: ${path}: ${actual_bytes}, " \
                 "expected ${expected_bytes}." >&2
            exit 1
        fi
    done < <(awk -F, 'NR > 1 { print $4 }' "${analysis_directory}/snapshots.csv")
}

if run_stage 1; then
    echo "Stage 1/6: 2048^3 adaptive-CFL Taylor-Green simulation"
FFTM_TURBULENCE_DATA_DIR="${SIM_ROOT}" \
FFTM_TURBULENCE_NODES="${SIM_NODES}" \
FFTM_TURBULENCE_GPUS="${SIM_GPUS}" \
FFTM_TURBULENCE_GPUS_PER_NODE="${GPUS_PER_NODE}" \
FFTM_TURBULENCE_AFFINITY=hca \
FFTM_TURBULENCE_SIZE=2048 \
FFTM_TURBULENCE_REYNOLDS=1600 \
FFTM_TURBULENCE_DT=0.00025 \
FFTM_TURBULENCE_DT_MIN=0 \
FFTM_TURBULENCE_DT_MAX=0.005 \
FFTM_TURBULENCE_DT_GROWTH=1.1 \
FFTM_TURBULENCE_TARGET_CFL=0.35 \
FFTM_TURBULENCE_CFL_FAIL=0.4 \
FFTM_TURBULENCE_FINAL_TIME=15.8 \
FFTM_TURBULENCE_STEPS=0 \
FFTM_TURBULENCE_DIAGNOSTICS_EVERY=10 \
FFTM_TURBULENCE_SNAPSHOT_EVERY=0 \
FFTM_TURBULENCE_VIZ_EVERY=0 \
FFTM_TURBULENCE_SNAPSHOT_PERIOD=0.2 \
FFTM_TURBULENCE_VIZ_PERIOD=0.5 \
FFTM_TURBULENCE_SNAPSHOT_SIZE=512 \
FFTM_TURBULENCE_VIZ_SNAPSHOT_SIZE=256 \
FFTM_TURBULENCE_WRITE_INITIAL_SNAPSHOT=1 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME="${SIM_SRUN_TIME}" \
"${LAUNCHER}" simulate
else
    echo "Stage 1/6: reusing existing simulation"
fi

SIM_MANIFEST="${SIMULATION_DIR}/snapshots.csv"
VIZ_MANIFEST="${VISUALIZATION_DIR}/snapshots.csv"
for REQUIRED in \
    "${SIM_MANIFEST}" \
    "${SIMULATION_DIR}/layout.json" \
    "${SIMULATION_DIR}/diagnostics.csv" \
    "${SIMULATION_DIR}/adaptive_timesteps.csv" \
    "${VIZ_MANIFEST}" \
    "${VISUALIZATION_DIR}/layout.json"; do
    if [[ ! -s "${REQUIRED}" ]]; then
        echo "Missing production simulation artifact: ${REQUIRED}" >&2
        exit 1
    fi
done
validate_uniform_series "${SIM_MANIFEST}" omega_z 80 0 15.8 0.2
validate_uniform_series "${VIZ_MANIFEST}" vorticity_magnitude 32 0 15.5 0.5
validate_uniform_series "${VIZ_MANIFEST}" q_criterion 32 0 15.5 0.5
validate_payloads "${SIMULATION_DIR}" "${SIM_MANIFEST}" omega_z 512
validate_payloads "${VISUALIZATION_DIR}" "${VIZ_MANIFEST}" vorticity_magnitude 256
validate_payloads "${VISUALIZATION_DIR}" "${VIZ_MANIFEST}" q_criterion 256
finish_if_stopped 1

if run_stage 2; then
    echo "Stage 2/6: full-band 512^3 x 80 round trip with sparse output"
FFTM_TURBULENCE_DATA_DIR="${ROUNDTRIP_ROOT}" \
FFTM_TURBULENCE_NODES="${ANALYSIS_NODES}" \
FFTM_TURBULENCE_GPUS="${ANALYSIS_GPUS}" \
FFTM_TURBULENCE_GPUS_PER_NODE="${GPUS_PER_NODE}" \
FFTM_TURBULENCE_AFFINITY=auto \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ANALYSIS_SIZE=512 \
FFTM_TURBULENCE_ANALYSIS_FRAMES=80 \
FFTM_TURBULENCE_FRAME_OFFSET=0 \
FFTM_TURBULENCE_MODE_MIN=0 \
FFTM_TURBULENCE_MODE_MAX=40 \
FFTM_TURBULENCE_SPATIAL_CUTOFF=0 \
FFTM_TURBULENCE_WRITE_EVERY=10 \
FFTM_TURBULENCE_FIELD=omega_z \
FFTM_TURBULENCE_SUBTRACT_MEAN=0 \
FFTM_TURBULENCE_HANN_WINDOW=0 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME="${ANALYSIS_SRUN_TIME}" \
"${LAUNCHER}" analyze
else
    echo "Stage 2/6: reusing existing round trip"
fi
validate_analysis_directory "${ROUNDTRIP_ROOT}/analysis"
finish_if_stopped 2

if run_stage 3; then
    echo "Stage 3/6: Hann-windowed 512^3 x 80 temporal modes 1-4"
FFTM_TURBULENCE_DATA_DIR="${FILTERED_ROOT}" \
FFTM_TURBULENCE_NODES="${ANALYSIS_NODES}" \
FFTM_TURBULENCE_GPUS="${ANALYSIS_GPUS}" \
FFTM_TURBULENCE_GPUS_PER_NODE="${GPUS_PER_NODE}" \
FFTM_TURBULENCE_AFFINITY=auto \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ANALYSIS_SIZE=512 \
FFTM_TURBULENCE_ANALYSIS_FRAMES=80 \
FFTM_TURBULENCE_FRAME_OFFSET=0 \
FFTM_TURBULENCE_MODE_MIN=1 \
FFTM_TURBULENCE_MODE_MAX=4 \
FFTM_TURBULENCE_SPATIAL_CUTOFF=0 \
FFTM_TURBULENCE_WRITE_EVERY=10 \
FFTM_TURBULENCE_FIELD=omega_z \
FFTM_TURBULENCE_SUBTRACT_MEAN=1 \
FFTM_TURBULENCE_HANN_WINDOW=1 \
FFTM_TURBULENCE_CPUS_PER_TASK="${GPU_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME="${ANALYSIS_SRUN_TIME}" \
"${LAUNCHER}" analyze
else
    echo "Stage 3/6: reusing existing filtered analysis"
fi
validate_analysis_directory "${FILTERED_ROOT}/analysis"
finish_if_stopped 3

if run_stage 4; then
    echo "Stage 4/6: exact block-streamed 4D validation and paper figures"
FFTM_TURBULENCE_DATA_DIR="${SPACETIME_ROOT}" \
FFTM_TURBULENCE_NODES=1 \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE=1 \
FFTM_TURBULENCE_AFFINITY=auto \
FFTM_TURBULENCE_INPUT_DIR="${SIMULATION_DIR}" \
FFTM_TURBULENCE_ROUNDTRIP_DIR="${ROUNDTRIP_ROOT}/analysis" \
FFTM_TURBULENCE_FILTERED_DIR="${FILTERED_ROOT}/analysis" \
FFTM_TURBULENCE_SPACETIME_VALIDATION_PREFIX=spacetime_production \
FFTM_TURBULENCE_SPACETIME_SLICE_TIME="${SPACETIME_SLICE_TIME}" \
FFTM_TURBULENCE_SPACETIME_Z_CHUNK="${SPACETIME_Z_CHUNK}" \
FFTM_TURBULENCE_SPACETIME_WORKERS="${REPORT_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SPACETIME_VALIDATION_DPI=600 \
FFTM_TURBULENCE_SPACETIME_VALIDATION_FORMATS=png,pdf \
FFTM_TURBULENCE_CPUS_PER_TASK="${REPORT_CPUS_PER_TASK}" \
FFTM_TURBULENCE_SRUN_TIME="${REPORT_SRUN_TIME}" \
"${LAUNCHER}" spacetime-validate-streaming
else
    echo "Stage 4/6: reusing existing production 4D report"
fi

SPACETIME_METRICS="${SPACETIME_ROOT}/spacetime_production_metrics.json"
if [[ ! -s "${SPACETIME_METRICS}" ]] ||
   ! grep -q '"validation_status": "passed"' "${SPACETIME_METRICS}"; then
    echo "Production 4D validation did not pass: ${SPACETIME_METRICS}" >&2
    exit 1
fi
for REQUIRED in \
    "${SPACETIME_ROOT}/spacetime_production.png" \
    "${SPACETIME_ROOT}/spacetime_production.pdf" \
    "${SPACETIME_ROOT}/spacetime_production_slice.png" \
    "${SPACETIME_ROOT}/spacetime_production_slice.pdf" \
    "${SPACETIME_ROOT}/spacetime_production_frames.csv" \
    "${SPACETIME_ROOT}/spacetime_production_modes.csv" \
    "${SPACETIME_ROOT}/spacetime_production_summary.md"; do
    if [[ ! -s "${REQUIRED}" ]]; then
        echo "Missing production 4D report artifact: ${REQUIRED}" >&2
        exit 1
    fi
done
finish_if_stopped 4

if run_stage 5; then
    echo "Stage 5/6: Taylor-Green literature validation"
FFTM_TURBULENCE_DATA_DIR="${TG_VALIDATION_ROOT}" \
FFTM_TURBULENCE_NODES=1 \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE=1 \
FFTM_TURBULENCE_AFFINITY=auto \
FFTM_TURBULENCE_DIAGNOSTICS="${SIMULATION_DIR}/diagnostics.csv" \
FFTM_TURBULENCE_VALIDATION_PREFIX=taylor_green_validation \
FFTM_TURBULENCE_VALIDATION_DPI=600 \
FFTM_TURBULENCE_VALIDATION_FORMATS=png,pdf,svg \
FFTM_TURBULENCE_CPUS_PER_TASK=16 \
FFTM_TURBULENCE_SRUN_TIME=00:30:00 \
"${LAUNCHER}" validate
else
    echo "Stage 5/6: reusing existing Taylor-Green validation"
fi
for REQUIRED in \
    "${TG_VALIDATION_ROOT}/taylor_green_validation_metrics.json" \
    "${TG_VALIDATION_ROOT}/taylor_green_validation.png" \
    "${TG_VALIDATION_ROOT}/taylor_green_validation.pdf" \
    "${TG_VALIDATION_ROOT}/taylor_green_validation.svg"; do
    if [[ ! -s "${REQUIRED}" ]]; then
        echo "Missing Taylor-Green validation artifact: ${REQUIRED}" >&2
        exit 1
    fi
done
finish_if_stopped 5

if run_stage 6; then
    echo "Stage 6/6: 600-DPI 256^3 vorticity and Q-criterion series"
FFTM_TURBULENCE_DATA_DIR="${FLOW_RENDER_ROOT}" \
FFTM_TURBULENCE_NODES=1 \
FFTM_TURBULENCE_GPUS=1 \
FFTM_TURBULENCE_GPUS_PER_NODE=1 \
FFTM_TURBULENCE_AFFINITY=auto \
FFTM_TURBULENCE_INPUT_DIR="${VISUALIZATION_DIR}" \
FFTM_TURBULENCE_RENDER_FIELDS=vorticity_magnitude,q_criterion \
FFTM_TURBULENCE_RENDER_STRIDE=1 \
FFTM_TURBULENCE_RENDER_DPI=600 \
FFTM_TURBULENCE_RENDER_WIDTH_INCHES=7.2 \
FFTM_TURBULENCE_RENDER_HEIGHT_INCHES=5.4 \
FFTM_TURBULENCE_RENDER_CAMERA_ZOOM=0.8 \
FFTM_TURBULENCE_RENDER_CONTACT_SHEET_COLUMNS=4 \
FFTM_TURBULENCE_CPUS_PER_TASK=16 \
FFTM_TURBULENCE_SRUN_TIME="${RENDER_SRUN_TIME}" \
"${LAUNCHER}" visualize-series
else
    echo "Stage 6/6: reusing existing flow rendering"
fi

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
while IFS= read -r filename; do
    if [[ ! -s "${FLOW_RENDER_ROOT}/figures/${filename}" ]]; then
        echo "Missing rendered flow snapshot: ${filename}" >&2
        exit 1
    fi
done < <(awk -F, 'NR > 1 { print $5 }' "${RENDER_MANIFEST}")
for REQUIRED in \
    "${FLOW_RENDER_ROOT}/figures/renders.json" \
    "${FLOW_RENDER_ROOT}/figures/vorticity_contact_sheet.png" \
    "${FLOW_RENDER_ROOT}/figures/q_contact_sheet.png"; do
    if [[ ! -s "${REQUIRED}" ]]; then
        echo "Missing flow-render artifact: ${REQUIRED}" >&2
        exit 1
    fi
done

printf 'Taylor-Green production pipeline passed: %s\n' "${PIPELINE_ROOT}"
printf '4D metrics: %s\n' "${SPACETIME_METRICS}"
printf 'Flow snapshots: %s\n' "${FLOW_RENDER_ROOT}/figures"
