#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SNAPSHOT_FORMAT = "fftm-taylor-green-snapshot-v1"
SUBSET_FORMAT = "fftm-taylor-green-spacetime-subset-v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Select uniformly spaced Taylor-Green snapshots and spectrally "
            "downsample them for a 4D FFT analysis."
        )
    )
    parser.add_argument("input", type=Path, help="source snapshot directory")
    parser.add_argument("output", type=Path, help="prepared snapshot directory")
    parser.add_argument("--field", default="omega_z")
    parser.add_argument("--frame-offset", type=int, default=0)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument(
        "--spatial-cutoff",
        type=int,
        default=None,
        help="rectangular Fourier cutoff (default: target_size/3)",
    )
    return parser.parse_args()


def load_layout(directory: Path):
    path = directory / "layout.json"
    metadata = json.loads(path.read_text())
    if metadata.get("format") != SNAPSHOT_FORMAT:
        raise RuntimeError(f"unsupported snapshot layout: {path}")
    if metadata.get("storage_order") != "x-fastest":
        raise RuntimeError(f"unsupported storage order in {path}")
    if metadata.get("scalar_type") != "float32":
        raise RuntimeError(f"unsupported scalar type in {path}")
    shape = tuple(int(value) for value in metadata["snapshot_shape"])
    if len(shape) != 3 or len(set(shape)) != 1:
        raise RuntimeError("the subset tool currently requires a cubic snapshot")
    return metadata, shape[0]


def load_records(directory: Path, field: str):
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
            if row["field"] == field
        ]
    if not records:
        raise RuntimeError(f"no '{field}' records in {path}")
    return records


def select_records(records, offset: int, count: int):
    if offset < 0 or count < 4 or count % 2:
        raise RuntimeError("frame offset must be nonnegative; frame count must be even and >= 4")
    selected = records[offset : offset + count]
    if len(selected) != count:
        raise RuntimeError(
            f"requested {count} frames at offset {offset}, found {len(selected)}"
        )
    times = np.asarray([record["time"] for record in selected], dtype=float)
    spacing = np.diff(times)
    if np.any(spacing <= 0.0):
        raise RuntimeError("snapshot times are not strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=1.0e-12):
        raise RuntimeError("4D FFT input snapshots must be uniformly spaced")
    return selected, float(spacing[0])


def load_snapshot(path: Path, size: int):
    values = np.fromfile(path, dtype="<f4")
    expected = size**3
    if values.size != expected:
        raise RuntimeError(
            f"{path} contains {values.size} float32 values, expected {expected}"
        )
    return values.reshape((size, size, size))


def retained_indices(target_size: int, source_size: int, cutoff: int):
    target_wave_numbers = np.rint(
        np.fft.fftfreq(target_size) * target_size
    ).astype(int)
    target_full = np.flatnonzero(np.abs(target_wave_numbers) <= cutoff)
    source_full = np.mod(target_wave_numbers[target_full], source_size)
    target_half = np.arange(cutoff + 1, dtype=int)
    return target_full, source_full, target_half


def spectral_downsample(values, target_size: int, cutoff: int):
    source_size = values.shape[0]
    source_spectrum = np.fft.rfftn(values)
    target_spectrum = np.zeros(
        (target_size, target_size, target_size // 2 + 1),
        dtype=source_spectrum.dtype,
    )
    target_full, source_full, target_half = retained_indices(
        target_size, source_size, cutoff
    )
    source_selection = np.ix_(source_full, source_full, target_half)
    target_selection = np.ix_(target_full, target_full, target_half)
    amplitude_scale = (target_size / source_size) ** 3
    target_spectrum[target_selection] = (
        source_spectrum[source_selection] * amplitude_scale
    )
    downsampled = np.fft.irfftn(
        target_spectrum,
        s=(target_size, target_size, target_size),
        axes=(0, 1, 2),
    )

    source_lowpass_spectrum = np.zeros_like(source_spectrum)
    source_lowpass_spectrum[source_selection] = source_spectrum[source_selection]
    source_lowpass = np.fft.irfftn(
        source_lowpass_spectrum,
        s=(source_size, source_size, source_size),
        axes=(0, 1, 2),
    )
    source_power = float(np.mean(np.square(values, dtype=np.float64)))
    retained_power = float(
        np.mean(np.square(source_lowpass, dtype=np.float64))
    )

    decimation_error = None
    if source_size % target_size == 0:
        stride = source_size // target_size
        sampled = source_lowpass[::stride, ::stride, ::stride]
        denominator = float(np.linalg.norm(sampled.ravel()))
        decimation_error = float(
            np.linalg.norm((downsampled - sampled).ravel())
            / max(denominator, np.finfo(float).tiny)
        )

    metrics = {
        "source_rms": math.sqrt(source_power),
        "downsampled_rms": float(
            np.sqrt(np.mean(np.square(downsampled, dtype=np.float64)))
        ),
        "retained_spatial_power_fraction": (
            retained_power / source_power if source_power > 0.0 else 0.0
        ),
        "decimation_relative_l2": decimation_error,
    }
    return np.asarray(downsampled, dtype="<f4"), metrics


def write_xdmf(path: Path, raw_name: str, field: str, time: float, size: int):
    spacing = 2.0 * math.pi / size
    path.write_text(
        '<?xml version="1.0" ?>\n'
        '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'
        '<Xdmf Version="3.0">\n'
        "  <Domain>\n"
        f'    <Grid Name="{field}" GridType="Uniform">\n'
        f'      <Time Value="{time:.17g}"/>\n'
        f'      <Topology TopologyType="3DCoRectMesh" Dimensions="{size} {size} {size}"/>\n'
        '      <Geometry GeometryType="ORIGIN_DXDYDZ">\n'
        '        <DataItem Dimensions="3" Format="XML">0 0 0</DataItem>\n'
        f'        <DataItem Dimensions="3" Format="XML">{spacing:.17g} {spacing:.17g} {spacing:.17g}</DataItem>\n'
        "      </Geometry>\n"
        f'      <Attribute Name="{field}" AttributeType="Scalar" Center="Node">\n'
        f'        <DataItem Dimensions="{size} {size} {size}" NumberType="Float" Precision="4" '
        f'Endian="Little" Format="Binary">{raw_name}</DataItem>\n'
        "      </Attribute>\n"
        "    </Grid>\n"
        "  </Domain>\n"
        "</Xdmf>\n"
    )


def main():
    args = parse_args()
    source_dir = args.input.resolve()
    output_dir = args.output.resolve()
    layout, source_size = load_layout(source_dir)
    records = load_records(source_dir, args.field)
    selected, sample_dt = select_records(
        records, args.frame_offset, args.frames
    )

    target_size = args.target_size
    if target_size <= 0 or target_size > source_size:
        raise RuntimeError("target size must be positive and no larger than source size")
    cutoff = (
        target_size // 3
        if args.spatial_cutoff is None
        else args.spatial_cutoff
    )
    maximum_cutoff = min((target_size - 1) // 2, (source_size - 1) // 2)
    if cutoff < 0 or cutoff > maximum_cutoff:
        raise RuntimeError(
            f"spatial cutoff must be in [0, {maximum_cutoff}], got {cutoff}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_metrics = []
    with (output_dir / "snapshots.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("step", "time", "field", "file"),
            lineterminator="\n",
        )
        writer.writeheader()
        for index, record in enumerate(selected):
            source_path = source_dir / record["file"]
            source = load_snapshot(source_path, source_size)
            downsampled, metrics = spectral_downsample(
                source, target_size, cutoff
            )
            raw_name = record["file"]
            downsampled.tofile(output_dir / raw_name)
            write_xdmf(
                output_dir / f"{Path(raw_name).stem}.xdmf",
                raw_name,
                args.field,
                record["time"],
                target_size,
            )
            writer.writerow(record)
            frame_metrics.append(
                {
                    "index": index,
                    "step": record["step"],
                    "time": record["time"],
                    **metrics,
                }
            )

    output_layout = {
        "format": SNAPSHOT_FORMAT,
        "source_shape": list(layout["snapshot_shape"]),
        "snapshot_shape": [target_size, target_size, target_size],
        "storage_order": "x-fastest",
        "scalar_type": "float32",
        "domain": layout["domain"],
        "low_pass_cutoff": cutoff,
        "pieces": [
            {
                "rank": 0,
                "start": [0, 0, 0],
                "size": [target_size, target_size, target_size],
            }
        ],
        "preparation": {
            "method": "rectangular-spectral-low-pass-and-resample",
            "source_directory": str(source_dir),
            "source_snapshot_size": source_size,
        },
    }
    (output_dir / "layout.json").write_text(
        json.dumps(output_layout, indent=2) + "\n"
    )

    retained = [
        row["retained_spatial_power_fraction"] for row in frame_metrics
    ]
    decimation_errors = [
        row["decimation_relative_l2"]
        for row in frame_metrics
        if row["decimation_relative_l2"] is not None
    ]
    summary = {
        "format": SUBSET_FORMAT,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "field": args.field,
        "source_size": source_size,
        "target_size": target_size,
        "spatial_cutoff": cutoff,
        "frame_offset": args.frame_offset,
        "frames": args.frames,
        "time_range": [selected[0]["time"], selected[-1]["time"]],
        "sample_dt": sample_dt,
        "mean_retained_spatial_power_fraction": float(np.mean(retained)),
        "min_retained_spatial_power_fraction": float(np.min(retained)),
        "max_decimation_relative_l2": (
            float(np.max(decimation_errors)) if decimation_errors else None
        ),
        "frame_metrics": frame_metrics,
    }
    (output_dir / "subset.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        "SPACETIME_SUBSET_RESULT "
        f"source={source_size} target={target_size} frames={args.frames} "
        f"time={selected[0]['time']:.9g}:{selected[-1]['time']:.9g} "
        f"cutoff={cutoff} retained={np.mean(retained):.9g} "
        f"max_decimation_l2={summary['max_decimation_relative_l2']}"
    )


if __name__ == "__main__":
    main()
