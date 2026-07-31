#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference_data"
REQUIRED_DIAGNOSTIC_COLUMNS = ("time", "kinetic_energy", "enstrophy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(reference_dir: Path):
    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != "fftm-taylor-green-reference-manifest-v1":
        raise RuntimeError(f"unsupported reference manifest: {manifest_path}")
    return manifest


def select_available_references(reference_dir: Path, entries, requested_ids):
    available = []
    fetch_script = Path(__file__).resolve().with_name("fetch_reference_data.py")
    for entry in entries:
        path = reference_dir / entry["file"]
        if path.is_file():
            available.append(entry)
            continue

        dataset_id = entry["id"]
        explicitly_requested = dataset_id in requested_ids
        required = bool(entry.get("required", entry.get("bundled", True)))
        fetch_command = (
            f"python3 {fetch_script} "
            f"--reference-dir {reference_dir} --dataset {dataset_id}"
        )
        if required or explicitly_requested:
            raise RuntimeError(
                f"missing reference dataset '{dataset_id}': {path}; "
                f"fetch it with: {fetch_command}"
            )
        print(
            f"warning: skipping optional reference dataset '{dataset_id}'; "
            f"fetch it with: {fetch_command}",
            file=sys.stderr,
        )

    if not available:
        raise RuntimeError(f"no reference datasets are available in {reference_dir}")
    return available


def load_named_csv(path: Path, columns):
    time = []
    energy = []
    dissipation = []
    enstrophy = []
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            time.append(float(row[columns["time"]]))
            energy.append(float(row[columns["kinetic_energy"]]))
            if "dissipation" in columns:
                dissipation.append(float(row[columns["dissipation"]]))
            if "enstrophy" in columns:
                enstrophy.append(float(row[columns["enstrophy"]]))
    return {
        "time": np.asarray(time),
        "kinetic_energy": np.asarray(energy),
        "dissipation": np.asarray(dissipation),
        "enstrophy": np.asarray(enstrophy),
    }


def load_reference(reference_dir: Path, entry, reynolds: float, verify: bool):
    path = reference_dir / entry["file"]
    if verify:
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise RuntimeError(
                f"reference checksum mismatch for {path}: "
                f"expected {entry['sha256']}, got {actual}"
            )

    columns = entry["columns"]
    if entry["reader"] == "whitespace":
        usecols = sorted(set(columns.values()))
        values = np.loadtxt(path, comments="#", usecols=usecols)
        column_to_loaded = {column: index for index, column in enumerate(usecols)}
        data = {
            name: values[:, column_to_loaded[column]]
            for name, column in columns.items()
        }
    elif entry["reader"] == "named_csv":
        data = load_named_csv(path, columns)
    else:
        raise RuntimeError(f"unsupported reference reader: {entry['reader']}")

    if entry.get("enstrophy_from_dissipation"):
        data["enstrophy"] = 0.5 * reynolds * data["dissipation"]

    required = ("time", "kinetic_energy", "enstrophy")
    if any(name not in data or data[name].size == 0 for name in required):
        raise RuntimeError(f"incomplete reference data in {path}")

    finite = np.logical_and.reduce([np.isfinite(data[name]) for name in required])
    order = np.argsort(data["time"][finite])
    result = dict(entry)
    for name in required:
        result[name] = np.asarray(data[name][finite][order], dtype=float)
    return result


def load_diagnostics(path: Path):
    values = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    if values.size == 0:
        raise RuntimeError(f"no diagnostics in {path}")
    available = values.dtype.names or ()
    missing = [name for name in REQUIRED_DIAGNOSTIC_COLUMNS if name not in available]
    if missing:
        raise RuntimeError(f"{path} is missing columns: {', '.join(missing)}")
    return {
        name: np.atleast_1d(np.asarray(values[name], dtype=float))
        for name in REQUIRED_DIAGNOSTIC_COLUMNS
    }


def load_run_metadata(diagnostics: Path):
    path = diagnostics.parent / "run.json"
    if not path.exists():
        return {}
    metadata = json.loads(path.read_text())
    if metadata.get("format") != "fftm-taylor-green-run-v1":
        return {}
    return metadata


def relative_l2(actual, reference):
    denominator = np.linalg.norm(reference)
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm(actual - reference) / denominator)


def comparison_metrics(simulation, reference):
    time = simulation["time"]
    common = np.logical_and(
        time >= reference["time"][0], time <= reference["time"][-1]
    )
    if not np.any(common):
        raise RuntimeError(f"no common time interval for {reference['id']}")

    sim_time = time[common]
    row = {
        "dataset_id": reference["id"],
        "label": reference["label"],
        "comparison_points": int(sim_time.size),
        "common_time_min": float(sim_time[0]),
        "common_time_max": float(sim_time[-1]),
    }
    for field in ("kinetic_energy", "enstrophy"):
        actual = simulation[field][common]
        expected = np.interp(sim_time, reference["time"], reference[field])
        error = actual - expected
        row[f"{field}_rmse"] = float(np.sqrt(np.mean(error * error)))
        row[f"{field}_relative_l2"] = relative_l2(actual, expected)
        row[f"{field}_max_abs"] = float(np.max(np.abs(error)))

    ref_visible = reference["time"] <= sim_time[-1]
    ref_peak = int(np.argmax(reference["enstrophy"][ref_visible]))
    ref_times = reference["time"][ref_visible]
    ref_enstrophy = reference["enstrophy"][ref_visible]
    sim_peak = int(np.argmax(simulation["enstrophy"]))
    row["reference_enstrophy_peak"] = float(ref_enstrophy[ref_peak])
    row["reference_enstrophy_peak_time"] = float(ref_times[ref_peak])
    row["simulation_enstrophy_peak"] = float(
        simulation["enstrophy"][sim_peak]
    )
    row["simulation_enstrophy_peak_time"] = float(
        simulation["time"][sim_peak]
    )
    return row


def marker_indices(size: int, maximum: int):
    stride = max(1, int(math.ceil(size / maximum)))
    indices = np.arange(0, size, stride)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def write_metrics(output_prefix: Path, rows):
    csv_path = output_prefix.with_name(output_prefix.name + "_metrics.csv")
    json_path = output_prefix.with_name(output_prefix.name + "_metrics.json")
    fieldnames = list(rows[0])
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    return csv_path, json_path


def plot_validation(
    simulation,
    references,
    label,
    output_prefix,
    formats,
    dpi,
    width_inches,
    height_inches,
    max_markers,
    time_max,
):
    configure_matplotlib()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(width_inches, height_inches),
        constrained_layout=True,
        sharex=True,
    )

    axes[0].plot(
        simulation["time"],
        simulation["kinetic_energy"],
        color="#111111",
        label=label,
        zorder=5,
    )
    axes[1].plot(
        simulation["time"],
        simulation["enstrophy"],
        color="#111111",
        label=label,
        zorder=5,
    )

    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    markers = ("o", "s", "^", "D")
    for index, reference in enumerate(references):
        visible = reference["time"] <= time_max
        ref_time = reference["time"][visible]
        selected = marker_indices(ref_time.size, max_markers)
        style = {
            "s": 17,
            "marker": markers[index % len(markers)],
            "facecolors": "none",
            "edgecolors": colors[index % len(colors)],
            "linewidths": 0.85,
            "label": reference["label"],
            "zorder": 3,
        }
        axes[0].scatter(
            ref_time[selected],
            reference["kinetic_energy"][visible][selected],
            **style,
        )
        axes[1].scatter(
            ref_time[selected],
            reference["enstrophy"][visible][selected],
            **style,
        )

    axes[0].set_title("(a) Kinetic energy")
    axes[1].set_title("(b) Enstrophy")
    axes[0].set_ylabel(r"$E(t)=\langle |\mathbf{u}|^2\rangle/2$")
    axes[1].set_ylabel(r"$\Omega(t)=\langle |\boldsymbol{\omega}|^2\rangle/2$")
    for axis in axes:
        axis.set_xlabel(r"$t$")
        axis.set_xlim(0.0, time_max)
        axis.grid(True, color="#d9d9d9", linewidth=0.55)
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].legend(loc="upper right", frameon=False)

    outputs = []
    for extension in formats:
        output = output_prefix.with_suffix("." + extension)
        save_options = {"dpi": dpi} if extension == "png" else {}
        figure.savefig(output, **save_options)
        outputs.append(output)
    plt.close(figure)
    return outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare FFTM Taylor-Green energy/enstrophy against published "
            "Re=1600 reference data."
        )
    )
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--reference", action="append", dest="reference_ids")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--reynolds", type=float)
    parser.add_argument("--time-max", type=float)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width-inches", type=float, default=7.2)
    parser.add_argument("--height-inches", type=float, default=3.25)
    parser.add_argument("--max-reference-markers", type=int, default=45)
    parser.add_argument(
        "--formats",
        default="png,pdf,svg",
        help="comma-separated output formats (default: png,pdf,svg)",
    )
    parser.add_argument("--no-verify-checksums", action="store_true")
    args = parser.parse_args()

    if args.dpi < 600:
        parser.error("--dpi must be at least 600 for paper output")
    if args.width_inches <= 0.0 or args.height_inches <= 0.0:
        parser.error("figure dimensions must be positive")
    if args.max_reference_markers <= 0:
        parser.error("--max-reference-markers must be positive")
    formats = [item.strip().lower() for item in args.formats.split(",") if item]
    unsupported = sorted(set(formats) - {"png", "pdf", "svg"})
    if unsupported:
        parser.error(f"unsupported output formats: {', '.join(unsupported)}")
    args.formats = formats
    return args


def main():
    args = parse_args()
    simulation = load_diagnostics(args.diagnostics)
    metadata = load_run_metadata(args.diagnostics)
    manifest = load_manifest(args.reference_dir)
    reynolds = args.reynolds or float(metadata.get("reynolds", manifest["reynolds"]))
    if abs(reynolds - float(manifest["reynolds"])) > 1.0e-12:
        raise RuntimeError(
            f"reference data require Re={manifest['reynolds']}, got Re={reynolds:g}"
        )

    entries = manifest["datasets"]
    requested = set(args.reference_ids or [])
    if args.reference_ids:
        entries = [entry for entry in entries if entry["id"] in requested]
        missing = requested - {entry["id"] for entry in entries}
        if missing:
            raise RuntimeError(f"unknown reference ids: {', '.join(sorted(missing))}")
    entries = select_available_references(args.reference_dir, entries, requested)
    references = [
        load_reference(
            args.reference_dir,
            entry,
            reynolds,
            not args.no_verify_checksums,
        )
        for entry in entries
    ]

    shape = metadata.get("shape", [])
    resolution = f"{shape[0]}^3" if len(shape) == 3 and len(set(shape)) == 1 else "run"
    label = f"FFTM ${resolution}$" if resolution != "run" else "FFTM run"
    time_max = args.time_max or float(np.max(simulation["time"]))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = plot_validation(
        simulation,
        references,
        label,
        args.output_prefix,
        args.formats,
        args.dpi,
        args.width_inches,
        args.height_inches,
        args.max_reference_markers,
        time_max,
    )
    rows = [comparison_metrics(simulation, reference) for reference in references]
    metrics = write_metrics(args.output_prefix, rows)

    metadata_path = args.output_prefix.with_name(
        args.output_prefix.name + "_plot.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "format": "fftm-taylor-green-validation-plot-v1",
                "diagnostics": str(args.diagnostics),
                "reynolds": reynolds,
                "simulation_label": label,
                "reference_ids": [reference["id"] for reference in references],
                "dpi": args.dpi,
                "width_inches": args.width_inches,
                "height_inches": args.height_inches,
                "time_max": time_max,
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
        + "\n"
    )

    for output in outputs:
        print(f"Wrote {output}")
    print(f"Wrote {metrics[0]}")
    print(f"Wrote {metrics[1]}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
