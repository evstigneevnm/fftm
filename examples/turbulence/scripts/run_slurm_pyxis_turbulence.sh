#!/usr/bin/env bash
set -euo pipefail

usage()
{
    cat <<'EOF'
Usage: run_slurm_pyxis_turbulence.sh simulate|analyze|visualize|visualize-series|validate|spacetime-validate|spacetime-validate-streaming

Required:
  FFTM_TURBULENCE_CONTAINER_IMAGE  FFTM SQSH image visible on compute nodes.

Common optional settings:
  FFTM_TURBULENCE_DATA_DIR         Result directory.
  FFTM_TURBULENCE_NODES            Node count (default: 1).
  FFTM_TURBULENCE_GPUS             MPI rank/GPU count (default: 8).
  FFTM_TURBULENCE_GPUS_PER_NODE    Allocated GPUs per node (default: ranks/node).
  FFTM_TURBULENCE_SRUN_TIME        Slurm step limit.
  FFTM_TURBULENCE_SRUN_EXTRA_ARGS  Extra srun options.
  FFTM_TURBULENCE_AFFINITY         auto or hca.
  FFTM_TURBULENCE_DRY_RUN          Print the command without executing it.

Simulation settings:
  FFTM_TURBULENCE_SIZE, REYNOLDS, DT, FINAL_TIME, STEPS
  FFTM_TURBULENCE_TARGET_CFL, DT_MIN, DT_MAX, DT_GROWTH, CFL_FAIL
  FFTM_TURBULENCE_DIAGNOSTICS_EVERY
  FFTM_TURBULENCE_SNAPSHOT_EVERY
  FFTM_TURBULENCE_VIZ_EVERY
  FFTM_TURBULENCE_SNAPSHOT_PERIOD
  FFTM_TURBULENCE_VIZ_PERIOD
  FFTM_TURBULENCE_SNAPSHOT_SIZE
  FFTM_TURBULENCE_VIZ_SNAPSHOT_SIZE
  FFTM_TURBULENCE_WRITE_INITIAL_SNAPSHOT

Analysis settings:
  FFTM_TURBULENCE_INPUT_DIR        Host directory containing snapshots.csv.
  FFTM_TURBULENCE_ANALYSIS_SIZE
  FFTM_TURBULENCE_ANALYSIS_FRAMES
  FFTM_TURBULENCE_FRAME_OFFSET
  FFTM_TURBULENCE_MODE_MIN, MODE_MAX, SPATIAL_CUTOFF, WRITE_EVERY
  FFTM_TURBULENCE_SUBTRACT_MEAN, HANN_WINDOW

Visualization settings:
  FFTM_TURBULENCE_SNAPSHOT         Host path to a generated .raw snapshot.
  FFTM_TURBULENCE_RENDER_MODE      isosurface, volume, or slice.
  FFTM_TURBULENCE_RENDER_FIELD     auto, vorticity, q, or scalar.
  FFTM_TURBULENCE_RENDER_OUTPUT    Output PNG filename.
  FFTM_TURBULENCE_RENDER_DPI       Raster DPI (default: 600).
  FFTM_TURBULENCE_RENDER_WIDTH_INCHES, RENDER_HEIGHT_INCHES
  FFTM_TURBULENCE_RENDER_CAMERA_ZOOM
  FFTM_TURBULENCE_RENDER_FIELDS     Series fields (default: vorticity_magnitude,q_criterion).
  FFTM_TURBULENCE_RENDER_STRIDE     Series frame stride (default: 1).

Validation settings:
  FFTM_TURBULENCE_DIAGNOSTICS      Host path to diagnostics.csv.
  FFTM_TURBULENCE_VALIDATION_PREFIX
  FFTM_TURBULENCE_VALIDATION_DPI   Raster DPI (default: 600).
  FFTM_TURBULENCE_VALIDATION_FORMATS

4D validation settings:
  FFTM_TURBULENCE_INPUT_DIR        Prepared 4D input snapshot directory.
  FFTM_TURBULENCE_ROUNDTRIP_DIR    Full-band FFTM analysis directory.
  FFTM_TURBULENCE_FILTERED_DIR     Band-filtered FFTM analysis directory.
  FFTM_TURBULENCE_SPACETIME_VALIDATION_PREFIX
  FFTM_TURBULENCE_SPACETIME_SLICE_TIME
  FFTM_TURBULENCE_SPACETIME_Z_CHUNK     Streaming validator z-slab depth.
  FFTM_TURBULENCE_SPACETIME_WORKERS     Streaming validator FFT worker count.
EOF
}

TARGET="${1:-}"
if [[ "${TARGET}" != "simulate" && "${TARGET}" != "analyze" && \
      "${TARGET}" != "visualize" && "${TARGET}" != "visualize-series" && \
      "${TARGET}" != "validate" && \
      "${TARGET}" != "spacetime-validate" && \
      "${TARGET}" != "spacetime-validate-streaming" ]]; then
    usage >&2
    exit 2
fi

IMAGE="${FFTM_TURBULENCE_CONTAINER_IMAGE:-${FFTM_CONTAINER_IMAGE:-}}"
: "${IMAGE:?Set FFTM_TURBULENCE_CONTAINER_IMAGE to the FFTM .sqsh path}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DATA_DIR="${FFTM_TURBULENCE_DATA_DIR:-${PWD}/data_taylor_green_${TARGET}_${STAMP}}"
NODES="${FFTM_TURBULENCE_NODES:-1}"
if [[ "${TARGET}" == "visualize" || "${TARGET}" == "visualize-series" || \
      "${TARGET}" == "validate" || \
      "${TARGET}" == "spacetime-validate" || \
      "${TARGET}" == "spacetime-validate-streaming" ]]; then
    DEFAULT_GPUS=1
    DEFAULT_CPUS_PER_TASK=16
    DEFAULT_SRUN_TIME=00:30:00
else
    DEFAULT_GPUS=8
    DEFAULT_CPUS_PER_TASK=1
    DEFAULT_SRUN_TIME=02:00:00
fi
GPUS="${FFTM_TURBULENCE_GPUS:-${DEFAULT_GPUS}}"
if (( NODES <= 0 || GPUS <= 0 || GPUS % NODES != 0 )); then
    echo "Nodes and GPUs must be positive, and GPUs must be divisible by nodes." >&2
    exit 2
fi

TASKS_PER_NODE=$(( GPUS / NODES ))
GPUS_PER_NODE="${FFTM_TURBULENCE_GPUS_PER_NODE:-${TASKS_PER_NODE}}"
if (( TASKS_PER_NODE > GPUS_PER_NODE )); then
    echo "MPI ranks per node cannot exceed FFTM_TURBULENCE_GPUS_PER_NODE." >&2
    exit 2
fi

if (( NODES > 1 )); then
    DEFAULT_AFFINITY=hca
else
    DEFAULT_AFFINITY=auto
fi
AFFINITY="${FFTM_TURBULENCE_AFFINITY:-${DEFAULT_AFFINITY}}"
case "${AFFINITY}" in
    auto)
        ;;
    hca)
        if (( GPUS_PER_NODE != 8 || TASKS_PER_NODE != 8 )); then
            echo "HCA affinity requires exactly 8 ranks and 8 allocated GPUs per node." >&2
            exit 2
        fi
        ;;
    *)
        echo "FFTM_TURBULENCE_AFFINITY must be auto or hca." >&2
        exit 2
        ;;
esac

CPUS_PER_TASK="${FFTM_TURBULENCE_CPUS_PER_TASK:-${DEFAULT_CPUS_PER_TASK}}"
SRUN_TIME="${FFTM_TURBULENCE_SRUN_TIME:-${DEFAULT_SRUN_TIME}}"
SRUN_EXTRA_ARGS="${FFTM_TURBULENCE_SRUN_EXTRA_ARGS:---exclude=cn13 --distribution=block:block --kill-on-bad-exit=1}"
DRY_RUN="${FFTM_TURBULENCE_DRY_RUN:-0}"
case "${DRY_RUN}" in
    0|1)
        ;;
    *)
        echo "FFTM_TURBULENCE_DRY_RUN must be 0 or 1." >&2
        exit 2
        ;;
esac

if [[ "${IMAGE}" == /* && ! -r "${IMAGE}" ]]; then
    echo "Missing or unreadable container image: ${IMAGE}" >&2
    exit 2
fi

mkdir -p "${DATA_DIR}"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"
read -r -a SRUN_EXTRA_ARRAY <<< "${SRUN_EXTRA_ARGS}"

MOUNTS="${DATA_DIR}:/data"
BINARY=
BINARY_ARGS=()
case "${TARGET}" in
    simulate)
        SIZE="${FFTM_TURBULENCE_SIZE:-512}"
        REYNOLDS="${FFTM_TURBULENCE_REYNOLDS:-1600}"
        DT="${FFTM_TURBULENCE_DT:-0.001}"
        DT_MIN="${FFTM_TURBULENCE_DT_MIN:-0}"
        DT_MAX="${FFTM_TURBULENCE_DT_MAX:-0}"
        DT_GROWTH="${FFTM_TURBULENCE_DT_GROWTH:-1.1}"
        TARGET_CFL="${FFTM_TURBULENCE_TARGET_CFL:-0}"
        CFL_FAIL="${FFTM_TURBULENCE_CFL_FAIL:-1.0}"
        FINAL_TIME="${FFTM_TURBULENCE_FINAL_TIME:-0.01}"
        STEPS="${FFTM_TURBULENCE_STEPS:-0}"
        DIAGNOSTICS_EVERY="${FFTM_TURBULENCE_DIAGNOSTICS_EVERY:-1}"
        SNAPSHOT_EVERY="${FFTM_TURBULENCE_SNAPSHOT_EVERY:-0}"
        VIZ_EVERY="${FFTM_TURBULENCE_VIZ_EVERY:-0}"
        SNAPSHOT_PERIOD="${FFTM_TURBULENCE_SNAPSHOT_PERIOD:-0}"
        VIZ_PERIOD="${FFTM_TURBULENCE_VIZ_PERIOD:-0}"
        SNAPSHOT_SIZE="${FFTM_TURBULENCE_SNAPSHOT_SIZE:-0}"
        VIZ_SNAPSHOT_SIZE="${FFTM_TURBULENCE_VIZ_SNAPSHOT_SIZE:-0}"
        WRITE_INITIAL="${FFTM_TURBULENCE_WRITE_INITIAL_SNAPSHOT:-0}"

        BINARY=/opt/fftm/bin/taylor_green_3d_autotuned.bin
        BINARY_ARGS=(
            --size "${SIZE}"
            --reynolds "${REYNOLDS}"
            --dt "${DT}"
            --dt-min "${DT_MIN}"
            --dt-max "${DT_MAX}"
            --dt-growth "${DT_GROWTH}"
            --target-cfl "${TARGET_CFL}"
            --cfl-fail "${CFL_FAIL}"
            --final-time "${FINAL_TIME}"
            --steps "${STEPS}"
            --cache /data/fftm_taylor_green_autotune.env
            --output /data/simulation
            --diagnostics-every "${DIAGNOSTICS_EVERY}"
            --snapshot-every "${SNAPSHOT_EVERY}"
            --viz-every "${VIZ_EVERY}"
            --snapshot-period "${SNAPSHOT_PERIOD}"
            --viz-period "${VIZ_PERIOD}"
            --snapshot-size "${SNAPSHOT_SIZE}"
            --viz-snapshot-size "${VIZ_SNAPSHOT_SIZE}"
            --write-initial-snapshot "${WRITE_INITIAL}"
        )
        ;;
    analyze)
        INPUT_DIR="${FFTM_TURBULENCE_INPUT_DIR:-}"
        : "${INPUT_DIR:?Set FFTM_TURBULENCE_INPUT_DIR to the simulation output directory}"
        if [[ ! -f "${INPUT_DIR}/snapshots.csv" ]]; then
            echo "Missing snapshot manifest: ${INPUT_DIR}/snapshots.csv" >&2
            exit 2
        fi
        INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"
        MOUNTS="${MOUNTS},${INPUT_DIR}:/input"

        ANALYSIS_SIZE="${FFTM_TURBULENCE_ANALYSIS_SIZE:-512}"
        FRAMES="${FFTM_TURBULENCE_ANALYSIS_FRAMES:-80}"
        FRAME_OFFSET="${FFTM_TURBULENCE_FRAME_OFFSET:-0}"
        MODE_MIN="${FFTM_TURBULENCE_MODE_MIN:-1}"
        MODE_MAX="${FFTM_TURBULENCE_MODE_MAX:-4}"
        SPATIAL_CUTOFF="${FFTM_TURBULENCE_SPATIAL_CUTOFF:-0}"
        WRITE_EVERY="${FFTM_TURBULENCE_WRITE_EVERY:-10}"
        FIELD="${FFTM_TURBULENCE_FIELD:-omega_z}"
        SUBTRACT_MEAN="${FFTM_TURBULENCE_SUBTRACT_MEAN:-1}"
        HANN_WINDOW="${FFTM_TURBULENCE_HANN_WINDOW:-1}"
        case "${SUBTRACT_MEAN}:${HANN_WINDOW}" in
            0:0|0:1|1:0|1:1)
                ;;
            *)
                echo "FFTM_TURBULENCE_SUBTRACT_MEAN and FFTM_TURBULENCE_HANN_WINDOW must be 0 or 1." >&2
                exit 2
                ;;
        esac

        BINARY=/opt/fftm/bin/taylor_green_spacetime_4d.bin
        BINARY_ARGS=(
            --input /input
            --output /data/analysis
            --field "${FIELD}"
            --size "${ANALYSIS_SIZE}"
            --frames "${FRAMES}"
            --frame-offset "${FRAME_OFFSET}"
            --mode-min "${MODE_MIN}"
            --mode-max "${MODE_MAX}"
            --spatial-cutoff "${SPATIAL_CUTOFF}"
            --write-every "${WRITE_EVERY}"
            --subtract-mean "${SUBTRACT_MEAN}"
            --hann-window "${HANN_WINDOW}"
        )
        ;;
    visualize)
        if (( NODES != 1 || GPUS != 1 )); then
            echo "Visualization requires one node and one GPU." >&2
            exit 2
        fi
        SNAPSHOT="${FFTM_TURBULENCE_SNAPSHOT:-}"
        : "${SNAPSHOT:?Set FFTM_TURBULENCE_SNAPSHOT to a generated .raw file}"
        if [[ ! -f "${SNAPSHOT}" ]]; then
            echo "Missing visualization snapshot: ${SNAPSHOT}" >&2
            exit 2
        fi
        SNAPSHOT_DIR="$(cd "$(dirname "${SNAPSHOT}")" && pwd -P)"
        SNAPSHOT_NAME="$(basename "${SNAPSHOT}")"
        if [[ ! -f "${SNAPSHOT_DIR}/layout.json" ]]; then
            echo "Missing snapshot layout: ${SNAPSHOT_DIR}/layout.json" >&2
            exit 2
        fi
        MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:/input"

        RENDER_MODE="${FFTM_TURBULENCE_RENDER_MODE:-isosurface}"
        RENDER_FIELD="${FFTM_TURBULENCE_RENDER_FIELD:-auto}"
        RENDER_OUTPUT="${FFTM_TURBULENCE_RENDER_OUTPUT:-${SNAPSHOT_NAME%.raw}_${RENDER_MODE}.png}"
        RENDER_DPI="${FFTM_TURBULENCE_RENDER_DPI:-600}"
        RENDER_WIDTH="${FFTM_TURBULENCE_RENDER_WIDTH_INCHES:-7.2}"
        RENDER_HEIGHT="${FFTM_TURBULENCE_RENDER_HEIGHT_INCHES:-5.4}"
        RENDER_CAMERA_ZOOM="${FFTM_TURBULENCE_RENDER_CAMERA_ZOOM:-0.8}"
        RENDER_DOMAIN_BOX="${FFTM_TURBULENCE_RENDER_DOMAIN_BOX:-1}"
        case "${RENDER_DOMAIN_BOX}" in
            0|1)
                ;;
            *)
                echo "FFTM_TURBULENCE_RENDER_DOMAIN_BOX must be 0 or 1." >&2
                exit 2
                ;;
        esac

        BINARY=python3
        BINARY_ARGS=(
            /opt/fftm/scripts/render_turbulence.py
            "/input/${SNAPSHOT_NAME}"
            --layout /input/layout.json
            --output "/data/${RENDER_OUTPUT}"
            --mode "${RENDER_MODE}"
            --field-kind "${RENDER_FIELD}"
            --dpi "${RENDER_DPI}"
            --width-inches "${RENDER_WIDTH}"
            --height-inches "${RENDER_HEIGHT}"
            --camera-zoom "${RENDER_CAMERA_ZOOM}"
        )
        if [[ "${RENDER_DOMAIN_BOX}" == "0" ]]; then
            BINARY_ARGS+=( --no-domain-box )
        fi
        ;;
    visualize-series)
        if (( NODES != 1 || GPUS != 1 )); then
            echo "Visualization series requires one node and one GPU." >&2
            exit 2
        fi
        INPUT_DIR="${FFTM_TURBULENCE_INPUT_DIR:-}"
        : "${INPUT_DIR:?Set FFTM_TURBULENCE_INPUT_DIR to the simulation output directory}"
        if [[ ! -f "${INPUT_DIR}/snapshots.csv" ||
              ! -f "${INPUT_DIR}/layout.json" ]]; then
            echo "Missing snapshot series metadata in ${INPUT_DIR}." >&2
            exit 2
        fi
        INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"
        MOUNTS="${MOUNTS},${INPUT_DIR}:/input"

        RENDER_FIELDS="${FFTM_TURBULENCE_RENDER_FIELDS:-vorticity_magnitude,q_criterion}"
        RENDER_STRIDE="${FFTM_TURBULENCE_RENDER_STRIDE:-1}"
        RENDER_DPI="${FFTM_TURBULENCE_RENDER_DPI:-600}"
        RENDER_WIDTH="${FFTM_TURBULENCE_RENDER_WIDTH_INCHES:-7.2}"
        RENDER_HEIGHT="${FFTM_TURBULENCE_RENDER_HEIGHT_INCHES:-5.4}"
        RENDER_CAMERA_ZOOM="${FFTM_TURBULENCE_RENDER_CAMERA_ZOOM:-0.8}"
        RENDER_CONTACT_COLUMNS="${FFTM_TURBULENCE_RENDER_CONTACT_SHEET_COLUMNS:-4}"
        RENDER_OVERWRITE="${FFTM_TURBULENCE_RENDER_OVERWRITE:-0}"
        case "${RENDER_OVERWRITE}" in
            0|1)
                ;;
            *)
                echo "FFTM_TURBULENCE_RENDER_OVERWRITE must be 0 or 1." >&2
                exit 2
                ;;
        esac

        BINARY=python3
        BINARY_ARGS=(
            /opt/fftm/scripts/render_snapshot_series.py
            --input /input
            --output /data/figures
            --fields "${RENDER_FIELDS}"
            --stride "${RENDER_STRIDE}"
            --dpi "${RENDER_DPI}"
            --width-inches "${RENDER_WIDTH}"
            --height-inches "${RENDER_HEIGHT}"
            --camera-zoom "${RENDER_CAMERA_ZOOM}"
            --contact-sheet-columns "${RENDER_CONTACT_COLUMNS}"
        )
        if [[ "${RENDER_OVERWRITE}" == "1" ]]; then
            BINARY_ARGS+=( --overwrite )
        fi
        ;;
    validate)
        if (( NODES != 1 || GPUS != 1 )); then
            echo "Validation plotting requires one node and one GPU allocation." >&2
            exit 2
        fi
        DIAGNOSTICS="${FFTM_TURBULENCE_DIAGNOSTICS:-}"
        : "${DIAGNOSTICS:?Set FFTM_TURBULENCE_DIAGNOSTICS to diagnostics.csv}"
        if [[ ! -f "${DIAGNOSTICS}" ]]; then
            echo "Missing Taylor-Green diagnostics: ${DIAGNOSTICS}" >&2
            exit 2
        fi
        RUN_DIR="$(cd "$(dirname "${DIAGNOSTICS}")" && pwd -P)"
        DIAGNOSTICS_NAME="$(basename "${DIAGNOSTICS}")"
        MOUNTS="${MOUNTS},${RUN_DIR}:/input"

        VALIDATION_PREFIX="${FFTM_TURBULENCE_VALIDATION_PREFIX:-taylor_green_validation}"
        VALIDATION_DPI="${FFTM_TURBULENCE_VALIDATION_DPI:-600}"
        VALIDATION_WIDTH="${FFTM_TURBULENCE_VALIDATION_WIDTH_INCHES:-7.2}"
        VALIDATION_HEIGHT="${FFTM_TURBULENCE_VALIDATION_HEIGHT_INCHES:-3.25}"
        VALIDATION_FORMATS="${FFTM_TURBULENCE_VALIDATION_FORMATS:-png,pdf,svg}"

        BINARY=python3
        BINARY_ARGS=(
            /opt/fftm/scripts/plot_taylor_green_validation.py
            "/input/${DIAGNOSTICS_NAME}"
            --reference-dir /opt/fftm/reference_data
            --output-prefix "/data/${VALIDATION_PREFIX}"
            --dpi "${VALIDATION_DPI}"
            --width-inches "${VALIDATION_WIDTH}"
            --height-inches "${VALIDATION_HEIGHT}"
            --formats "${VALIDATION_FORMATS}"
        )
        ;;
    spacetime-validate|spacetime-validate-streaming)
        if (( NODES != 1 || GPUS != 1 )); then
            echo "4D validation plotting requires one node and one GPU allocation." >&2
            exit 2
        fi
        INPUT_DIR="${FFTM_TURBULENCE_INPUT_DIR:-}"
        ROUNDTRIP_DIR="${FFTM_TURBULENCE_ROUNDTRIP_DIR:-}"
        FILTERED_DIR="${FFTM_TURBULENCE_FILTERED_DIR:-}"
        : "${INPUT_DIR:?Set FFTM_TURBULENCE_INPUT_DIR to the prepared snapshot directory}"
        : "${ROUNDTRIP_DIR:?Set FFTM_TURBULENCE_ROUNDTRIP_DIR to the full-band analysis directory}"
        : "${FILTERED_DIR:?Set FFTM_TURBULENCE_FILTERED_DIR to the filtered analysis directory}"
        for REQUIRED in \
            "${INPUT_DIR}/snapshots.csv" \
            "${INPUT_DIR}/layout.json" \
            "${ROUNDTRIP_DIR}/snapshots.csv" \
            "${ROUNDTRIP_DIR}/analysis.json" \
            "${FILTERED_DIR}/snapshots.csv" \
            "${FILTERED_DIR}/analysis.json"; do
            if [[ ! -f "${REQUIRED}" ]]; then
                echo "Missing 4D validation input: ${REQUIRED}" >&2
                exit 2
            fi
        done
        if [[ "${TARGET}" == "spacetime-validate-streaming" ]]; then
            for REQUIRED in \
                "${ROUNDTRIP_DIR}/selected_frames.csv" \
                "${FILTERED_DIR}/selected_frames.csv"; do
                if [[ ! -f "${REQUIRED}" ]]; then
                    echo "Missing streaming 4D validation input: ${REQUIRED}" >&2
                    exit 2
                fi
            done
        fi
        INPUT_DIR="$(cd "${INPUT_DIR}" && pwd -P)"
        ROUNDTRIP_DIR="$(cd "${ROUNDTRIP_DIR}" && pwd -P)"
        FILTERED_DIR="$(cd "${FILTERED_DIR}" && pwd -P)"
        MOUNTS="${MOUNTS},${INPUT_DIR}:/input,${ROUNDTRIP_DIR}:/roundtrip,${FILTERED_DIR}:/filtered"

        SPACETIME_PREFIX="${FFTM_TURBULENCE_SPACETIME_VALIDATION_PREFIX:-spacetime_validation}"
        SPACETIME_SLICE_TIME="${FFTM_TURBULENCE_SPACETIME_SLICE_TIME:-9.0}"
        SPACETIME_DPI="${FFTM_TURBULENCE_SPACETIME_VALIDATION_DPI:-600}"
        SPACETIME_FORMATS="${FFTM_TURBULENCE_SPACETIME_VALIDATION_FORMATS:-png,pdf}"

        BINARY=python3
        if [[ "${TARGET}" == "spacetime-validate-streaming" ]]; then
            SPACETIME_Z_CHUNK="${FFTM_TURBULENCE_SPACETIME_Z_CHUNK:-2}"
            SPACETIME_WORKERS="${FFTM_TURBULENCE_SPACETIME_WORKERS:-${CPUS_PER_TASK}}"
            BINARY_ARGS=(
                /opt/fftm/scripts/plot_spacetime_analysis_streaming.py
                --input /input
                --roundtrip /roundtrip
                --filtered /filtered
                --output-prefix "/data/${SPACETIME_PREFIX}"
                --slice-time "${SPACETIME_SLICE_TIME}"
                --z-chunk "${SPACETIME_Z_CHUNK}"
                --workers "${SPACETIME_WORKERS}"
                --dpi "${SPACETIME_DPI}"
                --formats "${SPACETIME_FORMATS}"
            )
        else
            BINARY_ARGS=(
                /opt/fftm/scripts/plot_spacetime_analysis.py
                --input /input
                --roundtrip /roundtrip
                --filtered /filtered
                --output-prefix "/data/${SPACETIME_PREFIX}"
                --slice-time "${SPACETIME_SLICE_TIME}"
                --dpi "${SPACETIME_DPI}"
                --formats "${SPACETIME_FORMATS}"
            )
        fi
        ;;
esac

COMMAND=(
    srun
    "${SRUN_EXTRA_ARRAY[@]}"
    -N "${NODES}"
    -n "${GPUS}"
    -G "$(( NODES * GPUS_PER_NODE ))"
    --ntasks-per-node="${TASKS_PER_NODE}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --cpus-per-task="${CPUS_PER_TASK}"
    --time="${SRUN_TIME}"
    --container-image "${IMAGE}"
    --container-mounts="${MOUNTS}"
    --container-workdir /opt/fftm
    --container-entrypoint
)
if [[ "${AFFINITY}" == "hca" ]]; then
    COMMAND+=( /opt/fftm/scripts/run_mpi_rank_affinity.sh --mode hca -- )
fi
COMMAND+=( "${BINARY}" "${BINARY_ARGS[@]}" )

{
    printf 'target=%s\n' "${TARGET}"
    printf 'container_image=%s\n' "${IMAGE}"
    printf 'nodes=%s\n' "${NODES}"
    printf 'gpus=%s\n' "${GPUS}"
    printf 'gpus_per_node=%s\n' "${GPUS_PER_NODE}"
    printf 'cpus_per_task=%s\n' "${CPUS_PER_TASK}"
    printf 'mpi_rank_affinity=%s\n' "${AFFINITY}"
    printf 'srun_time=%s\n' "${SRUN_TIME}"
    printf 'srun_extra_args=%s\n' "${SRUN_EXTRA_ARGS}"
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
} > "${DATA_DIR}/config.env"

printf 'Taylor-Green target: %s; nodes=%s; GPUs=%s; output=%s\n' \
    "${TARGET}" "${NODES}" "${GPUS}" "${DATA_DIR}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

"${COMMAND[@]}" 2>&1 | tee "${DATA_DIR}/${TARGET}.log"
