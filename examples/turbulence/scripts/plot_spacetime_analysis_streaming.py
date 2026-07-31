#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import scipy.fft

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNAPSHOT_FORMAT = "fftm-taylor-green-snapshot-v1"
ANALYSIS_FORMAT = "fftm-taylor-green-spacetime-v1"
OUTPUT_FORMAT = "fftm-taylor-green-spacetime-streaming-validation-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a large FFTM 4D Taylor-Green analysis with a bounded-memory "
            "z-slab NumPy reference."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--slice-time", type=float, default=9.0)
    parser.add_argument("--z-chunk", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--width-inches", type=float, default=7.2)
    parser.add_argument("--height-inches", type=float, default=5.4)
    parser.add_argument("--max-roundtrip-relative-l2", type=float, default=1.0e-6)
    parser.add_argument("--max-filter-relative-l2", type=float, default=1.0e-6)
    parser.add_argument("--max-retained-fraction-delta", type=float, default=1.0e-10)
    args = parser.parse_args()
    if args.z_chunk <= 0:
        parser.error("--z-chunk must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    return args


def load_json(path: Path, expected_format: str):
    value = json.loads(path.read_text())
    if value.get("format") != expected_format:
        raise RuntimeError(f"unsupported metadata format in {path}")
    return value


def load_layout(directory: Path):
    metadata = load_json(directory / "layout.json", SNAPSHOT_FORMAT)
    shape = tuple(int(value) for value in metadata["snapshot_shape"])
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise RuntimeError(f"invalid snapshot shape in {directory / 'layout.json'}")
    return metadata, shape


def load_records(path: Path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "time", "field", "file"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(f"{path} must contain columns {sorted(required)}")
        records = [
            {
                "step": int(row["step"]),
                "time": float(row["time"]),
                "field": row["field"],
                "file": row["file"],
            }
            for row in reader
        ]
    if not records:
        raise RuntimeError(f"no records in {path}")
    return records


def load_selected_records(directory: Path):
    path = directory / "selected_frames.csv"
    if not path.is_file():
        raise RuntimeError(
            f"missing {path}; rebuild the SQSH with selected-frame metadata support"
        )
    records = load_records(path)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        frames = [int(row["frame"]) for row in reader]
    if frames != list(range(len(records))):
        raise RuntimeError(f"{path} has non-contiguous frame indices")
    return records


def select_output_records(directory: Path):
    records = load_records(directory / "snapshots.csv")
    fields = sorted({record["field"] for record in records})
    if len(fields) != 1:
        raise RuntimeError(f"expected one output field in {directory}, found {fields}")
    return records


def record_key(record):
    return record["step"], round(record["time"], 12)


def verify_selected_records(reference, candidate, label):
    if [record_key(record) for record in reference] != [
        record_key(record) for record in candidate
    ]:
        raise RuntimeError(f"{label} selected-frame metadata differs")


def map_outputs(selected, outputs, label):
    selected_by_key = {
        record_key(record): index for index, record in enumerate(selected)
    }
    mapped = []
    for record in outputs:
        key = record_key(record)
        if key not in selected_by_key:
            raise RuntimeError(f"{label} output {key} is not a selected input frame")
        mapped.append((selected_by_key[key], record))
    if not mapped:
        raise RuntimeError(f"{label} has no reconstructed output frames")
    return mapped


def open_frame_maps(directory: Path, records, shape):
    nx, ny, nz = shape
    expected = nx * ny * nz
    mappings = []
    for record in records:
        path = directory / record["file"]
        if not path.is_file():
            raise RuntimeError(f"missing snapshot payload: {path}")
        mapping = np.memmap(path, dtype="<f4", mode="r", shape=(nz, ny, nx))
        if mapping.size != expected:
            raise RuntimeError(f"snapshot size mismatch: {path}")
        mappings.append(mapping)
    return mappings


def normalized_hann(frames):
    window = np.hanning(frames)
    return window * math.sqrt(frames / np.sum(window * window))


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.2,
            "axes.labelsize": 8.7,
            "axes.titlesize": 8.8,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure, prefix: Path, formats, dpi):
    outputs = []
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = prefix.with_suffix(f".{extension}")
        figure.savefig(path, dpi=dpi if extension == "png" else None)
        outputs.append(str(path))
    return outputs


def safe_relative_l2(diff2, reference2):
    return math.sqrt(diff2 / max(reference2, np.finfo(float).tiny))


def correlation_from_sums(count, sum_actual, sum_expected, sum_actual2, sum_expected2, sum_cross):
    covariance = sum_cross - sum_actual * sum_expected / count
    variance_actual = sum_actual2 - sum_actual * sum_actual / count
    variance_expected = sum_expected2 - sum_expected * sum_expected / count
    denominator = math.sqrt(max(variance_actual * variance_expected, 0.0))
    return covariance / denominator if denominator > 0.0 else float("nan")


def plot_summary(
    times,
    sampled_indices,
    roundtrip_errors,
    filter_errors,
    mode_power,
    mode_min,
    mode_max,
    fluctuation_rms,
    processed_rms,
    filtered_rms,
    sample_dt,
    prefix,
    formats,
    dpi,
    width,
    height,
):
    configure_matplotlib()
    figure = plt.figure(figsize=(width, height), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    error_axes = figure.add_subplot(grid[0, 0])
    power_axes = figure.add_subplot(grid[0, 1])
    rms_axes = figure.add_subplot(grid[1, :])

    floor = np.finfo(float).tiny
    sample_times = times[sampled_indices]
    error_axes.semilogy(
        sample_times,
        np.maximum(roundtrip_errors, floor),
        "o-",
        color="#0072B2",
        markersize=3,
        label="FFTM full-band sparse frames",
    )
    error_axes.semilogy(
        sample_times,
        np.maximum(filter_errors, floor),
        "s-",
        color="#D55E00",
        markersize=3,
        label="FFTM modes vs NumPy",
    )
    error_axes.set_xlabel(r"$t$")
    error_axes.set_ylabel("relative L2 error")
    error_axes.set_title("(a) Independent sparse-frame validation")
    error_axes.grid(True, which="both", alpha=0.22)
    error_axes.legend(frameon=False)

    modes = np.arange(mode_power.size)
    fractions = mode_power / np.sum(mode_power)
    power_axes.axvspan(
        mode_min - 0.45,
        mode_max + 0.45,
        color="#56B4E9",
        alpha=0.18,
        label=f"retained modes {mode_min}-{mode_max}",
    )
    power_axes.semilogy(
        modes,
        np.maximum(fractions, floor),
        "o-",
        color="#222222",
        markersize=3,
    )
    power_axes.set_xlabel("temporal Fourier mode")
    power_axes.set_ylabel("fraction of spectral power")
    power_axes.set_title("(b) Full-field temporal spectrum")
    power_axes.grid(True, which="both", alpha=0.22)
    power_axes.legend(frameon=False)

    rms_axes.plot(
        times,
        fluctuation_rms,
        color="#666666",
        linestyle="--",
        label="temporal fluctuation before window",
    )
    rms_axes.plot(
        times,
        processed_rms,
        color="#009E73",
        label="mean-subtracted Hann signal",
    )
    rms_axes.plot(
        times,
        filtered_rms,
        color="#CC79A7",
        label=f"NumPy modes {mode_min}-{mode_max}",
    )
    rms_axes.set_xlabel(r"$t$")
    rms_axes.set_ylabel(r"spatial RMS of $\omega_z$")
    rms_axes.set_title("(c) Low-frequency spatio-temporal reconstruction")
    rms_axes.grid(True, alpha=0.22)
    rms_axes.legend(frameon=False, ncol=3)

    outputs = save_figure(figure, prefix, formats, dpi)
    plt.close(figure)
    return outputs


def plot_slice(time, processed, actual, expected, mode_min, mode_max, prefix, formats, dpi, width):
    configure_matplotlib()
    images = (processed, actual, actual - expected)
    titles = (
        "(a) Preprocessed input",
        f"(b) FFTM modes {mode_min}-{mode_max}",
        "(c) FFTM - NumPy",
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(width, width / 3.0), constrained_layout=True
    )
    main_limit = max(float(np.max(np.abs(images[0]))), float(np.max(np.abs(images[1]))))
    error_limit = max(float(np.max(np.abs(images[2]))), np.finfo(float).tiny)
    for index, (axis, image, title) in enumerate(zip(axes, images, titles)):
        limit = main_limit if index < 2 else error_limit
        plotted = axis.imshow(
            image,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel(r"$x$")
        if index == 0:
            axis.set_ylabel(r"$y$")
        figure.colorbar(plotted, ax=axis, fraction=0.048, pad=0.03)
    figure.suptitle(
        rf"Central-$z$ signed-vorticity slice at $t={time:.3f}", fontsize=9
    )
    outputs = save_figure(figure, prefix, formats, dpi)
    plt.close(figure)
    return outputs


def main():
    args = parse_args()
    input_dir = args.input.resolve()
    roundtrip_dir = args.roundtrip.resolve()
    filtered_dir = args.filtered.resolve()
    output_prefix = args.output_prefix.resolve()

    input_layout, shape = load_layout(input_dir)
    _, roundtrip_shape = load_layout(roundtrip_dir)
    _, filtered_shape = load_layout(filtered_dir)
    if shape != roundtrip_shape or shape != filtered_shape:
        raise RuntimeError("input and reconstructed snapshot shapes differ")
    nx, ny, nz = shape

    roundtrip_analysis = load_json(roundtrip_dir / "analysis.json", ANALYSIS_FORMAT)
    filtered_analysis = load_json(filtered_dir / "analysis.json", ANALYSIS_FORMAT)
    selected = load_selected_records(filtered_dir)
    roundtrip_selected = load_selected_records(roundtrip_dir)
    verify_selected_records(selected, roundtrip_selected, "round-trip")
    if len(selected) != int(filtered_analysis["shape"][3]):
        raise RuntimeError("selected-frame count and filtered analysis shape differ")

    input_manifest = load_records(input_dir / "snapshots.csv")
    input_by_key = {record_key(record): record for record in input_manifest}
    input_records = []
    for record in selected:
        key = record_key(record)
        if key not in input_by_key:
            raise RuntimeError(f"selected input frame {key} is absent from snapshots.csv")
        input_records.append(input_by_key[key])

    roundtrip_outputs = select_output_records(roundtrip_dir)
    filtered_outputs = select_output_records(filtered_dir)
    roundtrip_mapped = map_outputs(selected, roundtrip_outputs, "round-trip")
    filtered_mapped = map_outputs(selected, filtered_outputs, "filtered")
    roundtrip_indices = [index for index, _ in roundtrip_mapped]
    filtered_indices = [index for index, _ in filtered_mapped]
    if roundtrip_indices != filtered_indices:
        raise RuntimeError("round-trip and filtered sparse output indices differ")
    sampled_indices = np.asarray(filtered_indices, dtype=int)

    times = np.asarray([record["time"] for record in selected], dtype=np.float64)
    sample_dt = float(filtered_analysis["sample_dt"])
    if times.size < 2 or not np.allclose(
        np.diff(times), sample_dt, rtol=1.0e-8, atol=1.0e-12
    ):
        raise RuntimeError("selected snapshots are not uniformly spaced")

    mode_min, mode_max = (
        int(value) for value in filtered_analysis["temporal_mode_range"]
    )
    subtract_mean = bool(filtered_analysis["subtract_mean"])
    hann_window = bool(filtered_analysis["hann_window"])
    window = normalized_hann(times.size) if hann_window else np.ones(times.size)

    input_maps = open_frame_maps(input_dir, input_records, shape)
    roundtrip_maps = open_frame_maps(
        roundtrip_dir, [record for _, record in roundtrip_mapped], shape
    )
    filtered_maps = open_frame_maps(
        filtered_dir, [record for _, record in filtered_mapped], shape
    )

    frame_count = times.size
    spatial_points = nx * ny * nz
    mode_power = np.zeros(frame_count // 2 + 1, dtype=np.float64)
    fluctuation_sq = np.zeros(frame_count, dtype=np.float64)
    processed_sq = np.zeros(frame_count, dtype=np.float64)
    filtered_expected_sq = np.zeros(frame_count, dtype=np.float64)
    roundtrip_diff2 = np.zeros(sampled_indices.size, dtype=np.float64)
    roundtrip_ref2 = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_diff2 = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_ref2 = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_sum_actual = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_sum_expected = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_sum_actual2 = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_sum_expected2 = np.zeros(sampled_indices.size, dtype=np.float64)
    filter_sum_cross = np.zeros(sampled_indices.size, dtype=np.float64)

    slice_output_position = int(
        np.argmin(np.abs(times[sampled_indices] - args.slice_time))
    )
    slice_frame = int(sampled_indices[slice_output_position])
    slice_z = nz // 2
    slice_processed = None
    slice_actual = None
    slice_expected = None
    weights = np.full(frame_count // 2 + 1, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if frame_count % 2 == 0:
        weights[-1] = 1.0

    for z0 in range(0, nz, args.z_chunk):
        z1 = min(nz, z0 + args.z_chunk)
        values = np.empty((frame_count, z1 - z0, ny, nx), dtype=np.float64)
        for frame, mapping in enumerate(input_maps):
            values[frame] = mapping[z0:z1]

        mean = np.mean(values, axis=0)
        for frame in range(frame_count):
            fluctuation_sq[frame] += np.sum(
                (values[frame] - mean) ** 2, dtype=np.float64
            )
        if subtract_mean:
            values -= mean[np.newaxis]
        if hann_window:
            values *= window[:, np.newaxis, np.newaxis, np.newaxis]
        processed_sq += np.sum(values * values, axis=(1, 2, 3), dtype=np.float64)

        spectrum = scipy.fft.rfft(values, axis=0, workers=args.workers)
        mode_power += weights * np.sum(
            np.abs(spectrum) ** 2, axis=(1, 2, 3), dtype=np.float64
        )
        spectrum[:mode_min] = 0.0
        spectrum[mode_max + 1 :] = 0.0
        expected = scipy.fft.irfft(
            spectrum, n=frame_count, axis=0, workers=args.workers
        )
        filtered_expected_sq += np.sum(
            expected * expected, axis=(1, 2, 3), dtype=np.float64
        )

        for position, frame in enumerate(sampled_indices):
            roundtrip = np.asarray(
                roundtrip_maps[position][z0:z1], dtype=np.float64
            )
            reference = np.asarray(input_maps[frame][z0:z1], dtype=np.float64)
            delta = roundtrip - reference
            roundtrip_diff2[position] += np.sum(delta * delta, dtype=np.float64)
            roundtrip_ref2[position] += np.sum(reference * reference, dtype=np.float64)

            actual = np.asarray(filtered_maps[position][z0:z1], dtype=np.float64)
            wanted = expected[frame]
            delta = actual - wanted
            filter_diff2[position] += np.sum(delta * delta, dtype=np.float64)
            filter_ref2[position] += np.sum(wanted * wanted, dtype=np.float64)
            filter_sum_actual[position] += np.sum(actual, dtype=np.float64)
            filter_sum_expected[position] += np.sum(wanted, dtype=np.float64)
            filter_sum_actual2[position] += np.sum(actual * actual, dtype=np.float64)
            filter_sum_expected2[position] += np.sum(wanted * wanted, dtype=np.float64)
            filter_sum_cross[position] += np.sum(actual * wanted, dtype=np.float64)

        if z0 <= slice_z < z1:
            local_z = slice_z - z0
            slice_processed = np.array(values[slice_frame, local_z], copy=True)
            slice_actual = np.asarray(
                filtered_maps[slice_output_position][slice_z], dtype=np.float64
            ).copy()
            slice_expected = np.array(
                expected[slice_frame, local_z], copy=True
            )

        print(
            f"SPACETIME_STREAM z={z1}/{nz} "
            f"chunk={z1 - z0} working_shape={values.shape}",
            flush=True,
        )

    if slice_processed is None or slice_actual is None or slice_expected is None:
        raise RuntimeError("failed to capture central validation slice")

    roundtrip_errors = np.sqrt(
        roundtrip_diff2 / np.maximum(roundtrip_ref2, np.finfo(float).tiny)
    )
    filter_errors = np.sqrt(
        filter_diff2 / np.maximum(filter_ref2, np.finfo(float).tiny)
    )
    sample_points = spatial_points
    correlations = np.asarray(
        [
            correlation_from_sums(
                sample_points,
                filter_sum_actual[index],
                filter_sum_expected[index],
                filter_sum_actual2[index],
                filter_sum_expected2[index],
                filter_sum_cross[index],
            )
            for index in range(sampled_indices.size)
        ]
    )
    numpy_retained_fraction = float(
        np.sum(mode_power[mode_min : mode_max + 1]) / np.sum(mode_power)
    )
    fftm_retained_fraction = float(
        filtered_analysis["retained_spectral_power_fraction"]
    )
    retained_delta = abs(numpy_retained_fraction - fftm_retained_fraction)
    roundtrip_global_error = safe_relative_l2(
        float(np.sum(roundtrip_diff2)), float(np.sum(roundtrip_ref2))
    )
    filter_global_error = safe_relative_l2(
        float(np.sum(filter_diff2)), float(np.sum(filter_ref2))
    )

    fluctuation_rms = np.sqrt(fluctuation_sq / spatial_points)
    processed_rms = np.sqrt(processed_sq / spatial_points)
    filtered_rms = np.sqrt(filtered_expected_sq / spatial_points)
    formats = [
        value.strip() for value in args.formats.split(",") if value.strip()
    ]
    figure_outputs = plot_summary(
        times,
        sampled_indices,
        roundtrip_errors,
        filter_errors,
        mode_power,
        mode_min,
        mode_max,
        fluctuation_rms,
        processed_rms,
        filtered_rms,
        sample_dt,
        output_prefix,
        formats,
        args.dpi,
        args.width_inches,
        args.height_inches,
    )
    slice_outputs = plot_slice(
        times[slice_frame],
        slice_processed,
        slice_actual,
        slice_expected,
        mode_min,
        mode_max,
        output_prefix.with_name(output_prefix.name + "_slice"),
        formats,
        args.dpi,
        args.width_inches,
    )

    frames_csv = output_prefix.with_name(output_prefix.name + "_frames.csv")
    frames_csv.parent.mkdir(parents=True, exist_ok=True)
    with frames_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "analysis_frame",
                "step",
                "time",
                "roundtrip_relative_l2",
                "filtered_numpy_relative_l2",
                "filtered_numpy_correlation",
            ]
        )
        for position, frame in enumerate(sampled_indices):
            writer.writerow(
                [
                    int(frame),
                    selected[frame]["step"],
                    f"{times[frame]:.17g}",
                    f"{roundtrip_errors[position]:.17g}",
                    f"{filter_errors[position]:.17g}",
                    f"{correlations[position]:.17g}",
                ]
            )

    modes_csv = output_prefix.with_name(output_prefix.name + "_modes.csv")
    with modes_csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "mode",
                "frequency",
                "angular_frequency",
                "spectral_power",
                "spectral_power_fraction",
                "retained",
            ]
        )
        total_mode_power = np.sum(mode_power)
        for mode, power in enumerate(mode_power):
            frequency = mode / (frame_count * sample_dt)
            writer.writerow(
                [
                    mode,
                    f"{frequency:.17g}",
                    f"{2.0 * math.pi * frequency:.17g}",
                    f"{power:.17g}",
                    f"{power / total_mode_power:.17g}",
                    int(mode_min <= mode <= mode_max),
                ]
            )

    failures = []
    if roundtrip_global_error > args.max_roundtrip_relative_l2:
        failures.append(
            f"round-trip relative L2 {roundtrip_global_error:.9g} "
            f"> {args.max_roundtrip_relative_l2:.9g}"
        )
    if filter_global_error > args.max_filter_relative_l2:
        failures.append(
            f"filter relative L2 {filter_global_error:.9g} "
            f"> {args.max_filter_relative_l2:.9g}"
        )
    if retained_delta > args.max_retained_fraction_delta:
        failures.append(
            f"retained-power delta {retained_delta:.9g} "
            f"> {args.max_retained_fraction_delta:.9g}"
        )

    metrics = {
        "format": OUTPUT_FORMAT,
        "validation_status": "failed" if failures else "passed",
        "failures": failures,
        "shape": [nx, ny, nz, frame_count],
        "sample_dt": sample_dt,
        "time_range": [float(times[0]), float(times[-1])],
        "streaming_z_chunk": args.z_chunk,
        "fft_workers": args.workers,
        "validated_output_frames": sampled_indices.tolist(),
        "roundtrip_sparse_global_relative_l2": roundtrip_global_error,
        "filtered_sparse_global_relative_l2": filter_global_error,
        "minimum_filtered_numpy_correlation": float(np.nanmin(correlations)),
        "fftm_retained_spectral_power_fraction": fftm_retained_fraction,
        "numpy_retained_spectral_power_fraction": numpy_retained_fraction,
        "retained_spectral_power_fraction_delta": retained_delta,
        "dominant_nonzero_temporal_mode": 1 + int(np.argmax(mode_power[1:])),
        "forward_4d_ms": float(filtered_analysis["forward_4d_ms"]),
        "inverse_4d_ms": float(filtered_analysis["inverse_4d_ms"]),
        "figures": figure_outputs + slice_outputs,
        "frames_csv": str(frames_csv),
        "modes_csv": str(modes_csv),
    }
    metrics_path = output_prefix.with_name(output_prefix.name + "_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    period = frame_count * sample_dt
    summary_path = output_prefix.with_name(output_prefix.name + "_summary.md")
    summary_path.write_text(
        "\n".join(
            [
                "# 4D production spatio-temporal analysis",
                "",
                "## Expected",
                "",
                f"- Sparse full-band round-trip relative L2 <= {args.max_roundtrip_relative_l2:.3g}.",
                f"- Sparse modes {mode_min}-{mode_max} relative L2 <= {args.max_filter_relative_l2:.3g} "
                "against a full-field streaming NumPy temporal FFT.",
                f"- Retained-power difference <= {args.max_retained_fraction_delta:.3g}.",
                "",
                "## Obtained",
                "",
                f"- Validation status: **{'FAILED' if failures else 'PASSED'}**.",
                f"- Sparse full-band round-trip relative L2: `{roundtrip_global_error:.9g}`.",
                f"- Sparse filtered reconstruction relative L2: `{filter_global_error:.9g}`.",
                f"- Minimum filtered-frame correlation: `{np.nanmin(correlations):.12g}`.",
                f"- Retained temporal spectral power: `{100.0 * fftm_retained_fraction:.4f}%` "
                f"(NumPy: `{100.0 * numpy_retained_fraction:.4f}%`).",
                f"- Dominant nonzero temporal mode: `{1 + int(np.argmax(mode_power[1:]))}`.",
                "",
                "## Interpretation",
                "",
                f"- The {frame_count}-sample window spans `{period:.6g}` time units.",
                f"- Modes {mode_min}-{mode_max} represent periods from "
                f"`{period / mode_min:.6g}` to `{period / mode_max:.6g}` time units.",
                "- The validator traverses every spatial point in bounded z-slabs; only the "
                "reconstructed output frames are sparse.",
            ]
        )
        + "\n"
    )

    print(
        "SPACETIME_STREAM_RESULT "
        f"status={metrics['validation_status']} shape={nx}x{ny}x{nz}x{frame_count} "
        f"roundtrip_l2={roundtrip_global_error:.9g} filter_l2={filter_global_error:.9g} "
        f"retained_delta={retained_delta:.9g} output={output_prefix}",
        flush=True,
    )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
