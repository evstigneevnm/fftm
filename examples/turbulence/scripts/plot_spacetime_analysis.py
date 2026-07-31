#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNAPSHOT_FORMAT = "fftm-taylor-green-snapshot-v1"
VALIDATION_FORMAT = "fftm-taylor-green-spacetime-validation-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate FFTM's 4D round trip and temporal band reconstruction "
            "against an independent NumPy reference."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--roundtrip", type=Path, required=True)
    parser.add_argument("--filtered", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--slice-time", type=float, default=9.0)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--width-inches", type=float, default=7.2)
    parser.add_argument("--height-inches", type=float, default=5.4)
    parser.add_argument("--max-roundtrip-relative-l2", type=float, default=1.0e-6)
    parser.add_argument("--max-filter-relative-l2", type=float, default=1.0e-6)
    parser.add_argument("--max-retained-fraction-delta", type=float, default=1.0e-10)
    return parser.parse_args()


def load_layout(directory: Path):
    path = directory / "layout.json"
    metadata = json.loads(path.read_text())
    if metadata.get("format") != SNAPSHOT_FORMAT:
        raise RuntimeError(f"unsupported snapshot layout: {path}")
    shape = tuple(int(value) for value in metadata["snapshot_shape"])
    if len(shape) != 3:
        raise RuntimeError(f"invalid snapshot shape in {path}")
    return metadata, shape


def load_manifest(directory: Path):
    path = directory / "snapshots.csv"
    with path.open(newline="") as source:
        records = [
            {
                "step": int(row["step"]),
                "time": float(row["time"]),
                "field": row["field"],
                "file": row["file"],
            }
            for row in csv.DictReader(source)
        ]
    if not records:
        raise RuntimeError(f"no snapshot records in {path}")
    return records


def load_analysis(directory: Path):
    path = directory / "analysis.json"
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "fftm-taylor-green-spacetime-v1":
        raise RuntimeError(f"unsupported analysis metadata: {path}")
    return metadata


def select_single_field(records, requested=None):
    fields = sorted({record["field"] for record in records})
    if requested is not None:
        matching = [record for record in records if record["field"] == requested]
        if not matching:
            raise RuntimeError(f"field '{requested}' is absent from the manifest")
        return matching
    if len(fields) != 1:
        raise RuntimeError(f"expected one output field, found {fields}")
    return [record for record in records if record["field"] == fields[0]]


def load_stack(directory: Path, records, shape):
    frames = []
    expected = int(np.prod(shape))
    for record in records:
        path = directory / record["file"]
        values = np.fromfile(path, dtype="<f4")
        if values.size != expected:
            raise RuntimeError(
                f"{path} contains {values.size} values, expected {expected}"
            )
        nx, ny, nz = shape
        frames.append(values.reshape((nz, ny, nx)))
    return np.asarray(frames, dtype=np.float64)


def align_records(reference, candidate, label):
    reference_keys = [(record["step"], record["time"]) for record in reference]
    candidate_keys = [(record["step"], record["time"]) for record in candidate]
    if reference_keys != candidate_keys:
        raise RuntimeError(f"{label} output records do not align with the input")


def relative_l2(actual, expected):
    denominator = float(np.linalg.norm(expected.ravel()))
    return float(
        np.linalg.norm((actual - expected).ravel())
        / max(denominator, np.finfo(float).tiny)
    )


def correlation(actual, expected):
    actual_flat = actual.ravel() - np.mean(actual)
    expected_flat = expected.ravel() - np.mean(expected)
    denominator = np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(actual_flat, expected_flat) / denominator)


def preprocess_temporal(values, subtract_mean: bool, hann_window: bool):
    processed = np.array(values, dtype=np.float64, copy=True)
    if subtract_mean:
        processed -= np.mean(processed, axis=0, keepdims=True)
    window = np.ones(values.shape[0], dtype=np.float64)
    scale = 1.0
    if hann_window:
        window = np.hanning(values.shape[0])
        scale = math.sqrt(values.shape[0] / np.sum(window * window))
        processed *= (window * scale)[:, None, None, None]
    return processed, window, scale


def temporal_mode_power(processed):
    spectrum = np.fft.rfft(processed, axis=0)
    weights = np.full(spectrum.shape[0], 2.0)
    weights[0] = 1.0
    if processed.shape[0] % 2 == 0:
        weights[-1] = 1.0
    power = weights * np.sum(np.abs(spectrum) ** 2, axis=(1, 2, 3))
    return spectrum, power


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
        outputs.append(path)
    return outputs


def plot_summary(
    times,
    roundtrip_errors,
    filter_errors,
    mode_power,
    mode_min,
    mode_max,
    input_fluctuation,
    processed,
    filtered,
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
    error_axes.semilogy(
        times,
        np.maximum(roundtrip_errors, floor),
        "o-",
        color="#0072B2",
        markersize=3,
        label="FFTM full-band round trip",
    )
    error_axes.semilogy(
        times,
        np.maximum(filter_errors, floor),
        "s-",
        color="#D55E00",
        markersize=2.8,
        label="FFTM modes vs NumPy",
    )
    error_axes.set_xlabel(r"$t$")
    error_axes.set_ylabel("relative L2 error")
    error_axes.set_title("(a) Numerical validation")
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
    power_axes.set_title("(b) Temporal spectrum")
    power_axes.grid(True, which="both", alpha=0.22)
    power_axes.legend(frameon=False)
    frequency = modes / (times.size * sample_dt)
    upper = power_axes.secondary_xaxis(
        "top",
        functions=(
            lambda mode: mode / (times.size * sample_dt),
            lambda value: value * (times.size * sample_dt),
        ),
    )
    upper.set_xlabel("frequency")
    upper.set_xticks(frequency[:: max(1, math.ceil(frequency.size / 5))])

    rms = lambda values: np.sqrt(np.mean(values * values, axis=(1, 2, 3)))
    rms_axes.plot(
        times,
        rms(input_fluctuation),
        color="#666666",
        linestyle="--",
        label="temporal fluctuation before window",
    )
    rms_axes.plot(
        times,
        rms(processed),
        color="#009E73",
        label="mean-subtracted Hann signal",
    )
    rms_axes.plot(
        times,
        rms(filtered),
        color="#CC79A7",
        label=f"FFTM reconstructed modes {mode_min}-{mode_max}",
    )
    rms_axes.set_xlabel(r"$t$")
    rms_axes.set_ylabel(r"spatial RMS of $\omega_z$")
    rms_axes.set_title("(c) Low-frequency spatio-temporal reconstruction")
    rms_axes.grid(True, alpha=0.22)
    rms_axes.legend(frameon=False, ncol=3)

    outputs = save_figure(figure, prefix, formats, dpi)
    plt.close(figure)
    return outputs


def plot_slices(
    time,
    processed,
    filtered,
    expected,
    mode_min,
    mode_max,
    prefix,
    formats,
    dpi,
    width,
):
    configure_matplotlib()
    z_index = processed.shape[0] // 2
    images = (
        processed[z_index],
        filtered[z_index],
        filtered[z_index] - expected[z_index],
    )
    titles = (
        "(a) Preprocessed input",
        f"(b) FFTM modes {mode_min}-{mode_max}",
        "(c) FFTM - NumPy",
    )
    figure, axes = plt.subplots(
        1, 3, figsize=(width, width / 3.0), constrained_layout=True
    )
    main_limit = max(
        float(np.max(np.abs(images[0]))), float(np.max(np.abs(images[1])))
    )
    error_limit = max(float(np.max(np.abs(images[2]))), np.finfo(float).tiny)
    for index, (axes_item, image, title) in enumerate(
        zip(axes, images, titles)
    ):
        limit = main_limit if index < 2 else error_limit
        plotted = axes_item.imshow(
            image,
            origin="lower",
            extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axes_item.set_title(title)
        axes_item.set_xlabel(r"$x$")
        if index == 0:
            axes_item.set_ylabel(r"$y$")
        figure.colorbar(plotted, ax=axes_item, fraction=0.048, pad=0.03)
    figure.suptitle(
        rf"Central-$z$ signed-vorticity slice at $t={time:.2f}$", fontsize=9
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

    roundtrip_analysis = load_analysis(roundtrip_dir)
    filtered_analysis = load_analysis(filtered_dir)
    input_layout, shape = load_layout(input_dir)
    roundtrip_layout, roundtrip_shape = load_layout(roundtrip_dir)
    filtered_layout, filtered_shape = load_layout(filtered_dir)
    if shape != roundtrip_shape or shape != filtered_shape:
        raise RuntimeError("input and reconstructed snapshot shapes differ")

    input_records = select_single_field(
        load_manifest(input_dir), filtered_analysis["input_field"]
    )
    roundtrip_records = select_single_field(load_manifest(roundtrip_dir))
    filtered_records = select_single_field(load_manifest(filtered_dir))
    align_records(input_records, roundtrip_records, "round-trip")
    align_records(input_records, filtered_records, "filtered")

    input_values = load_stack(input_dir, input_records, shape)
    roundtrip_values = load_stack(
        roundtrip_dir, roundtrip_records, roundtrip_shape
    )
    filtered_values = load_stack(
        filtered_dir, filtered_records, filtered_shape
    )
    times = np.asarray([record["time"] for record in input_records])
    sample_dt = float(filtered_analysis["sample_dt"])
    if times.size < 2 or not np.allclose(
        np.diff(times), sample_dt, rtol=1.0e-8, atol=1.0e-12
    ):
        raise RuntimeError("analysis metadata and snapshot times disagree")

    mode_min, mode_max = (
        int(value) for value in filtered_analysis["temporal_mode_range"]
    )
    processed, window, window_scale = preprocess_temporal(
        input_values,
        bool(filtered_analysis["subtract_mean"]),
        bool(filtered_analysis["hann_window"]),
    )
    temporal_spectrum, mode_power = temporal_mode_power(processed)
    selected_spectrum = np.zeros_like(temporal_spectrum)
    selected_spectrum[mode_min : mode_max + 1] = temporal_spectrum[
        mode_min : mode_max + 1
    ]
    expected_filtered = np.fft.irfft(
        selected_spectrum, n=times.size, axis=0
    )

    roundtrip_errors = np.asarray(
        [
            relative_l2(roundtrip_values[index], input_values[index])
            for index in range(times.size)
        ]
    )
    filter_errors = np.asarray(
        [
            relative_l2(filtered_values[index], expected_filtered[index])
            for index in range(times.size)
        ]
    )
    filter_correlations = np.asarray(
        [
            correlation(filtered_values[index], expected_filtered[index])
            for index in range(times.size)
        ]
    )
    numpy_retained_fraction = float(
        np.sum(mode_power[mode_min : mode_max + 1]) / np.sum(mode_power)
    )
    fftm_retained_fraction = float(
        filtered_analysis["retained_spectral_power_fraction"]
    )

    temporal_mean = np.mean(input_values, axis=0, keepdims=True)
    input_fluctuation = input_values - temporal_mean
    rms = lambda values: np.sqrt(np.mean(values * values, axis=(1, 2, 3)))
    input_rms = rms(input_fluctuation)
    processed_rms = rms(processed)
    filtered_rms = rms(filtered_values)

    metrics_csv = output_prefix.with_name(
        output_prefix.name + "_frames.csv"
    )
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "frame",
                "step",
                "time",
                "roundtrip_relative_l2",
                "filtered_numpy_relative_l2",
                "filtered_numpy_correlation",
                "input_fluctuation_rms",
                "preprocessed_rms",
                "filtered_rms",
            )
        )
        for index, record in enumerate(input_records):
            writer.writerow(
                (
                    index,
                    record["step"],
                    f"{times[index]:.17g}",
                    f"{roundtrip_errors[index]:.17g}",
                    f"{filter_errors[index]:.17g}",
                    f"{filter_correlations[index]:.17g}",
                    f"{input_rms[index]:.17g}",
                    f"{processed_rms[index]:.17g}",
                    f"{filtered_rms[index]:.17g}",
                )
            )

    mode_csv = output_prefix.with_name(output_prefix.name + "_modes.csv")
    with mode_csv.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            (
                "mode",
                "frequency",
                "angular_frequency",
                "spectral_power",
                "spectral_power_fraction",
                "retained",
            )
        )
        for mode, power in enumerate(mode_power):
            frequency = mode / (times.size * sample_dt)
            writer.writerow(
                (
                    mode,
                    f"{frequency:.17g}",
                    f"{2.0 * math.pi * frequency:.17g}",
                    f"{power:.17g}",
                    f"{power / np.sum(mode_power):.17g}",
                    int(mode_min <= mode <= mode_max),
                )
            )

    formats = [
        value.strip()
        for value in args.formats.split(",")
        if value.strip()
    ]
    summary_outputs = plot_summary(
        times,
        roundtrip_errors,
        filter_errors,
        mode_power,
        mode_min,
        mode_max,
        input_fluctuation,
        processed,
        filtered_values,
        sample_dt,
        output_prefix,
        formats,
        args.dpi,
        args.width_inches,
        args.height_inches,
    )
    slice_index = int(np.argmin(np.abs(times - args.slice_time)))
    slice_prefix = output_prefix.with_name(output_prefix.name + "_slice")
    slice_outputs = plot_slices(
        times[slice_index],
        processed[slice_index],
        filtered_values[slice_index],
        expected_filtered[slice_index],
        mode_min,
        mode_max,
        slice_prefix,
        formats,
        args.dpi,
        args.width_inches,
    )

    retained_delta = abs(fftm_retained_fraction - numpy_retained_fraction)
    roundtrip_global_error = relative_l2(roundtrip_values, input_values)
    filter_global_error = relative_l2(filtered_values, expected_filtered)
    failures = []
    if roundtrip_global_error > args.max_roundtrip_relative_l2:
        failures.append(
            "round-trip relative L2 "
            f"{roundtrip_global_error:.9g} > {args.max_roundtrip_relative_l2:.9g}"
        )
    if filter_global_error > args.max_filter_relative_l2:
        failures.append(
            "filter relative L2 "
            f"{filter_global_error:.9g} > {args.max_filter_relative_l2:.9g}"
        )
    if retained_delta > args.max_retained_fraction_delta:
        failures.append(
            "retained-power delta "
            f"{retained_delta:.9g} > {args.max_retained_fraction_delta:.9g}"
        )

    subset_metadata_path = input_dir / "subset.json"
    subset_metadata = (
        json.loads(subset_metadata_path.read_text())
        if subset_metadata_path.exists()
        else {}
    )
    spatial_retained_mean = subset_metadata.get(
        "mean_retained_spatial_power_fraction"
    )
    spatial_retained_at_slice = None
    if subset_metadata.get("frame_metrics"):
        closest = min(
            subset_metadata["frame_metrics"],
            key=lambda row: abs(float(row["time"]) - times[slice_index]),
        )
        spatial_retained_at_slice = float(
            closest["retained_spatial_power_fraction"]
        )

    summary_markdown = output_prefix.with_name(
        output_prefix.name + "_summary.md"
    )
    temporal_period = times.size * sample_dt
    lines = [
        "# 4D spatio-temporal validation",
        "",
        "## Expected",
        "",
        f"- Full-band FFTM round-trip relative L2 <= {args.max_roundtrip_relative_l2:.3g}.",
        f"- Modes {mode_min}-{mode_max} reconstruction relative L2 <= {args.max_filter_relative_l2:.3g} "
        "against the independent NumPy temporal filter.",
        f"- FFTM/NumPy retained-power difference <= {args.max_retained_fraction_delta:.3g}.",
        "",
        "## Obtained",
        "",
        f"- Validation status: **{'FAILED' if failures else 'PASSED'}**.",
        f"- Full-band round-trip relative L2: `{roundtrip_global_error:.9g}`.",
        f"- Filtered reconstruction relative L2: `{filter_global_error:.9g}`.",
        f"- Retained temporal spectral power: `{100.0 * fftm_retained_fraction:.4f}%` "
        f"(NumPy: `{100.0 * numpy_retained_fraction:.4f}%`).",
        f"- Dominant nonzero temporal mode: `{1 + int(np.argmax(mode_power[1:]))}`.",
        "",
        "## Interpretation",
        "",
        f"- Modes {mode_min}-{mode_max} represent periods from "
        f"`{temporal_period / mode_min:.4g}` to `{temporal_period / mode_max:.4g}` time units "
        "and recover the slowly evolving signed-vorticity structure.",
        "- The agreement with NumPy shows that FFTM's 4D transform, native spectral layout, "
        "mode mask, inverse transform, and normalization are mutually consistent.",
    ]
    if spatial_retained_mean is not None:
        lines.append(
            f"- The 64-to-{shape[0]} spatial preparation retained "
            f"`{100.0 * float(spatial_retained_mean):.2f}%` of vorticity power on average."
        )
    if spatial_retained_at_slice is not None:
        lines.append(
            f"- At the plotted `t={times[slice_index]:.2f}` frame it retained "
            f"`{100.0 * spatial_retained_at_slice:.2f}%`; this aggressive small-grid result "
            "validates the workflow, not DNS-scale turbulence fidelity."
        )
    summary_markdown.write_text("\n".join(lines) + "\n")

    summary = {
        "format": VALIDATION_FORMAT,
        "input_directory": str(input_dir),
        "roundtrip_directory": str(roundtrip_dir),
        "filtered_directory": str(filtered_dir),
        "shape": [*shape, int(times.size)],
        "time_range": [float(times[0]), float(times[-1])],
        "sample_dt": sample_dt,
        "temporal_mode_range": [mode_min, mode_max],
        "window_scale": window_scale,
        "spatial_low_pass_cutoff": input_layout.get("low_pass_cutoff"),
        "validation_status": "failed" if failures else "passed",
        "validation_failures": failures,
        "roundtrip_global_relative_l2": roundtrip_global_error,
        "roundtrip_max_frame_relative_l2": float(
            np.max(roundtrip_errors)
        ),
        "filtered_numpy_global_relative_l2": filter_global_error,
        "filtered_numpy_max_frame_relative_l2": float(
            np.max(filter_errors)
        ),
        "filtered_numpy_min_frame_correlation": float(
            np.nanmin(filter_correlations)
        ),
        "fftm_retained_spectral_power_fraction": fftm_retained_fraction,
        "numpy_retained_spectral_power_fraction": numpy_retained_fraction,
        "retained_fraction_absolute_difference": retained_delta,
        "dominant_nonzero_temporal_mode": int(
            1 + np.argmax(mode_power[1:])
        ),
        "roundtrip_forward_4d_ms": roundtrip_analysis["forward_4d_ms"],
        "roundtrip_inverse_4d_ms": roundtrip_analysis["inverse_4d_ms"],
        "filtered_forward_4d_ms": filtered_analysis["forward_4d_ms"],
        "filtered_inverse_4d_ms": filtered_analysis["inverse_4d_ms"],
        "figures": [str(path) for path in summary_outputs + slice_outputs],
        "frame_metrics_csv": str(metrics_csv),
        "mode_metrics_csv": str(mode_csv),
        "summary_markdown": str(summary_markdown),
    }
    summary_path = output_prefix.with_name(
        output_prefix.name + "_metrics.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(
        "SPACETIME_VALIDATION_RESULT "
        f"status={summary['validation_status']} "
        f"shape={shape[0]}x{shape[1]}x{shape[2]}x{times.size} "
        f"roundtrip_l2={summary['roundtrip_global_relative_l2']:.9g} "
        f"filter_l2={summary['filtered_numpy_global_relative_l2']:.9g} "
        f"retained_fftm={fftm_retained_fraction:.9g} "
        f"retained_numpy={numpy_retained_fraction:.9g} "
        f"dominant_mode={summary['dominant_nonzero_temporal_mode']}"
    )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
