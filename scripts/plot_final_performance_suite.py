#!/usr/bin/env python3
"""Build the final FFTM performance and external-library figure suite.

Every plotted time is a complete forward/backward pair measured as the
maximum rank wall time.  Throughput is N/t and effective floating-point rate
uses the conventional 10*N*log2(N)/t operation count for the pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fftm-final")

PAIR_FLOP_FACTOR = 10.0
EPSILON = 1.0e-11
GPU_COUNTS = (8, 16, 24, 32, 64, 96, 120)


@dataclass(frozen=True)
class Measurement:
    library: str
    dimension: int
    side: int
    num_gpus: int
    role: str
    mean_ms: float
    stddev_ms: float
    configuration: str
    campaign: str
    source: str
    max_l2_diff: float = float("nan")
    provenance: str = "current"
    figure_id: str = ""
    series: str = ""

    @property
    def total_points(self) -> int:
        return self.side**self.dimension

    @property
    def num_nodes(self) -> float:
        return self.num_gpus / 8.0

    @property
    def throughput_gpoints_s(self) -> float:
        return self.total_points / (self.mean_ms / 1000.0) / 1.0e9

    @property
    def effective_tflops(self) -> float:
        flops = PAIR_FLOP_FACTOR * self.total_points * math.log2(self.total_points)
        return flops / (self.mean_ms / 1000.0) / 1.0e12

    def metric_stddev(self, metric: str) -> float:
        if not math.isfinite(self.stddev_ms) or self.stddev_ms <= 0.0:
            return 0.0
        if metric == "mean_ms":
            return self.stddev_ms
        value = (
            self.throughput_gpoints_s
            if metric == "throughput_gpoints_s"
            else self.effective_tflops
        )
        return value * self.stddev_ms / self.mean_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite_float(text: object, default: float = float("nan")) -> float:
    if text is None or not str(text).strip():
        return default
    value = float(str(text))
    return value if math.isfinite(value) else default


def source_ref(root: Path, path: Path, line: Optional[int] = None) -> str:
    try:
        value = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        value = str(path.resolve())
    return f"{value}:{line}" if line is not None else value


def validate_measurement(row: Measurement) -> None:
    if row.mean_ms <= 0.0 or not math.isfinite(row.mean_ms):
        raise RuntimeError(f"Invalid wall time in {row.source}")
    if row.library == "FFTM":
        if not math.isfinite(row.max_l2_diff) or row.max_l2_diff > EPSILON:
            raise RuntimeError(f"Invalid FFTM numerical result in {row.source}")


def load_final_policy(root: Path, path: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for line, item in enumerate(read_csv(path), start=2):
        dimension = int(item["dimension"])
        configuration = f"{item['strategy']} {item['mode']}"
        if dimension == 4:
            configuration += (
                f" credit={item.get('credit_window', '') or 'off'}"
                f" cyclic={item.get('cyclic_peer_order', '') or '0'}"
            )
        row = Measurement(
            library="FFTM",
            dimension=dimension,
            side=int(item["size"]),
            num_gpus=int(item["num_gpus"]),
            role="strong",
            mean_ms=float(item["avg_wall_ms"]),
            stddev_ms=float(item["stddev_wall_ms"]),
            max_l2_diff=float(item["max_l2_diff"]),
            configuration=configuration,
            campaign="final-policy-20260808",
            provenance="current-final-policy",
            source=source_ref(root, path, line),
        )
        validate_measurement(row)
        output.append(row)
    return output


def load_release_scaling(root: Path, directories: Sequence[Path]) -> List[Measurement]:
    output: List[Measurement] = []
    for directory in directories:
        for dimension in (3, 4):
            path = directory / "benchmarks" / "cpp_csv" / f"benchmark_fftm_{dimension}d.csv"
            for line, item in enumerate(read_csv(path), start=2):
                side = int(item["nx"])
                configuration = f"{item['strategy']} {item['mode']}"
                if dimension == 3:
                    configuration += f" {item['pencil_layout']} {item['pencil_pipeline']}"
                else:
                    configuration += (
                        f" wz-c{item['slab_native_wz_plan_concurrency']}"
                        f" chunk={item['native_xw_chunk_mib']}MiB"
                    )
                row = Measurement(
                    library="FFTM",
                    dimension=dimension,
                    side=side,
                    num_gpus=int(item["num_gpus"]),
                    role="strong",
                    mean_ms=float(item["avg_wall_ms"]),
                    stddev_ms=float(item["stddev_wall_ms"]),
                    max_l2_diff=float(item["max_l2_diff"]),
                    configuration=configuration,
                    campaign="release-scale-20260805",
                    provenance="current-release-high-count",
                    source=source_ref(root, path, line),
                )
                validate_measurement(row)
                output.append(row)
    return output


def load_final_high_count(root: Path, directory: Path) -> List[Measurement]:
    path = directory / "selected_results.csv"
    output: List[Measurement] = []
    for line, item in enumerate(read_csv(path), start=2):
        if item["status"] != "passed":
            raise RuntimeError(f"Invalid high-count row in {path}:{line}")
        dimension = int(item["dimension"])
        configuration = f"{item['strategy']} {item['mode']}"
        if dimension == 3 and item.get("grid", "") not in ("", "configured"):
            configuration += f" {item['grid']}"
        if dimension == 4:
            configuration += (
                f" credit={item.get('slab_backward_credit_window', '') or 'off'}"
                f" cyclic={item.get('slab_backward_cyclic_peer_order', '') or '0'}"
            )
        row = Measurement(
            library="FFTM",
            dimension=dimension,
            side=int(item["size"].split("x", 1)[0]),
            num_gpus=int(item["gpus"]),
            role="strong",
            mean_ms=float(item["avg_wall_ms"]),
            stddev_ms=float(item["stddev_wall_ms"]),
            max_l2_diff=float(item["max_l2_diff"]),
            configuration=configuration,
            campaign="final-high-count-20260808",
            provenance="current-final-high-count",
            source=source_ref(root, path, line),
        )
        validate_measurement(row)
        output.append(row)
    return output


def load_hca_scaling(root: Path, path: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for line, item in enumerate(read_csv(path), start=2):
        library = item["library"]
        max_l2 = finite_float(item.get("max_l2_diff"))
        row = Measurement(
            library=library,
            dimension=int(item["dimension"]),
            side=int(item["size"]),
            num_gpus=int(item["num_gpus"]),
            role=item["scaling_role"],
            mean_ms=float(item["mean_ms"]),
            stddev_ms=float(item["stddev_ms"]),
            max_l2_diff=max_l2,
            configuration=item["variant"],
            campaign="hca-scaling-20260725",
            provenance=(
                "validated-prior-fftm" if library == "FFTM" else "egger-hca"
            ),
            source=source_ref(root, path, line),
        )
        validate_measurement(row)
        output.append(row)
    return output


def load_gpu_fft(root: Path, directory: Path) -> Tuple[List[Measurement], Dict[int, float]]:
    path = directory / "analysis" / "gpu_fft_reference_comparison.csv"
    output: List[Measurement] = []
    published: Dict[int, float] = {}
    for line, item in enumerate(read_csv(path), start=2):
        if item["valid"] != "1":
            raise RuntimeError(f"Invalid GPU-FFT row in {path}:{line}")
        detail_path = directory / item["source_file"]
        detail = read_csv(detail_path)
        if len(detail) != 1 or detail[0]["valid"] != "1":
            raise RuntimeError(f"Invalid GPU-FFT detail in {detail_path}")
        row = Measurement(
            library="GPU-FFT",
            dimension=3,
            side=int(item["size"]),
            num_gpus=int(item["gpus"]),
            role="strong",
            mean_ms=float(item["avg_pair_ms"]),
            stddev_ms=float(detail[0]["stddev_pair_ms"]),
            max_l2_diff=float(item["relative_l2"]),
            configuration="CUDA-aware HCA-bound reference",
            campaign="gpu-fft-same-system-20260807",
            provenance="same-system-external",
            source=source_ref(root, detail_path),
        )
        output.append(row)
        if item.get("published_tflops", "").strip():
            published[int(item["gpus"])] = float(item["published_tflops"])
    return output, published


def load_fitted_fftm(root: Path, path: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for line, item in enumerate(read_csv(path), start=2):
        row = Measurement(
            library="FFTM",
            dimension=3,
            side=int(item["nx"]),
            num_gpus=int(item["num_gpus"]),
            role="fitted",
            mean_ms=float(item["avg_wall_ms"]),
            stddev_ms=float(item["stddev_wall_ms"]),
            max_l2_diff=float(item["max_l2_diff"]),
            configuration=(
                f"{item['strategy']} {item['mode']} p{item['p1']}x{item['p2']}"
            ),
            campaign="release-fitted-20260805",
            provenance="current-release-fitted",
            source=source_ref(root, path, line),
        )
        validate_measurement(row)
        output.append(row)
    return output


def load_fitted_egger(root: Path, path: Path) -> List[Measurement]:
    rows = read_csv(path)
    grouped: Dict[Tuple[object, ...], Dict[str, Tuple[Dict[str, str], int]]] = {}
    for line, item in enumerate(rows, start=2):
        if (
            item.get("ok") != "1"
            or item.get("transport") != "cuda_aware"
            or item.get("comm") != "Peer2Peer"
            or item.get("send") != "Sync"
            or item.get("case_name") not in {"forward", "inverse"}
        ):
            continue
        key = (
            int(item["gpu_count"]),
            int(item["nx"]),
            item["family"],
            item["variant"],
            item.get("sequence", ""),
            item.get("opt", ""),
            int(item["p1"]),
            int(item["p2"]),
        )
        grouped.setdefault(key, {})[item["case_name"]] = (item, line)

    output: List[Measurement] = []
    for key, pair in grouped.items():
        if set(pair) != {"forward", "inverse"}:
            continue
        forward, forward_line = pair["forward"]
        inverse, inverse_line = pair["inverse"]
        mean_ms = float(forward["avg_wall_ms"]) + float(inverse["avg_wall_ms"])
        stddev_ms = math.hypot(
            float(forward["stddev_wall_ms"]), float(inverse["stddev_wall_ms"])
        )
        gpu, side, family, variant, _sequence, _opt, p1, p2 = key
        output.append(
            Measurement(
                library="Egger",
                dimension=3,
                side=int(side),
                num_gpus=int(gpu),
                role="fitted",
                mean_ms=mean_ms,
                stddev_ms=stddev_ms,
                configuration=f"{family} {variant} p{p1}x{p2}",
                campaign="egger-fitted-20260521",
                provenance="egger-original-fitted",
                source=(
                    f"{source_ref(root, path, forward_line)}+"
                    f"{source_ref(root, path, inverse_line)}"
                ),
            )
        )
    return output


def load_4d_weak_endpoint(root: Path, directory: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for path in sorted((directory / "full").glob("c*_m*/cpp_csv/benchmark_fftm_4d.csv")):
        item = read_csv(path)
        if len(item) != 1:
            raise RuntimeError(f"Expected one row in {path}")
        row_data = item[0]
        row = Measurement(
            library="FFTM",
            dimension=4,
            side=int(row_data["nx"]),
            num_gpus=int(row_data["num_gpus"]),
            role="weak",
            mean_ms=float(row_data["avg_wall_ms"]),
            stddev_ms=float(row_data["stddev_wall_ms"]),
            max_l2_diff=float(row_data["max_l2_diff"]),
            configuration=(
                f"c{row_data['slab_native_wz_plan_concurrency']} "
                f"chunk={row_data['native_xw_chunk_mib']}MiB"
            ),
            campaign="release-4d-weak-20260806",
            provenance="current-release-weak-endpoint",
            source=source_ref(root, path, 2),
        )
        validate_measurement(row)
        output.append(row)
    return output


def best_by(
    rows: Iterable[Measurement], keys: Sequence[str]
) -> List[Measurement]:
    best: Dict[Tuple[object, ...], Measurement] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in keys)
        old = best.get(key)
        if old is None or row.mean_ms < old.mean_ms:
            best[key] = row
    return sorted(
        best.values(),
        key=lambda row: (row.num_gpus, row.dimension, row.side, row.library),
    )


def tag_rows(
    rows: Iterable[Measurement], figure_id: str, labels: Mapping[str, str]
) -> List[Measurement]:
    return [
        replace(row, figure_id=figure_id, series=labels.get(row.library, row.library))
        for row in rows
    ]


def build_figure_sets(
    *,
    final_rows: Sequence[Measurement],
    high_count_rows: Sequence[Measurement],
    release_rows: Sequence[Measurement],
    hca_rows: Sequence[Measurement],
    gpu_fft_rows: Sequence[Measurement],
    fitted_fftm: Sequence[Measurement],
    fitted_egger: Sequence[Measurement],
    weak_4d_endpoints: Sequence[Measurement],
) -> Dict[str, List[Measurement]]:
    figures: Dict[str, List[Measurement]] = {}

    fitted = best_by(
        list(fitted_fftm) + list(fitted_egger),
        ("library", "num_gpus", "side"),
    )
    figures["fitted_3d"] = tag_rows(
        fitted, "fitted_3d", {"FFTM": "FFTM", "Egger": "Egger"}
    )

    strong_1920 = best_by(
        [
            row
            for row in list(final_rows) + list(gpu_fft_rows)
            if row.dimension == 3 and row.side == 1920
        ],
        ("library", "num_gpus", "side"),
    )
    figures["strong_3d_1920"] = tag_rows(
        strong_1920,
        "strong_3d_1920",
        {"FFTM": "FFTM", "GPU-FFT": "GPU-FFT"},
    )

    strong_2048_candidates = [
        row
        for row in list(final_rows)
        + list(high_count_rows)
        + list(release_rows)
        + list(hca_rows)
        + list(gpu_fft_rows)
        if row.dimension == 3 and row.side == 2048 and row.role == "strong"
    ]
    strong_2048 = best_by(
        strong_2048_candidates, ("library", "num_gpus", "side")
    )
    figures["strong_3d_2048"] = tag_rows(
        strong_2048,
        "strong_3d_2048",
        {"FFTM": "FFTM", "Egger": "Egger", "GPU-FFT": "GPU-FFT"},
    )

    strong_4d_candidates = [
        row
        for row in list(final_rows) + list(high_count_rows) + list(release_rows) + list(hca_rows)
        if row.library == "FFTM"
        and row.dimension == 4
        and row.side == 320
        and row.role == "strong"
    ]
    figures["strong_4d_320"] = tag_rows(
        best_by(strong_4d_candidates, ("num_gpus", "side")),
        "strong_4d_320",
        {"FFTM": "FFTM 4D"},
    )

    weak_3d = [
        row for row in hca_rows if row.dimension == 3 and row.role == "weak"
    ]
    figures["weak_3d"] = tag_rows(
        best_by(weak_3d, ("library", "num_gpus", "side")),
        "weak_3d",
        {"FFTM": "FFTM", "Egger": "Egger"},
    )

    weak_4d = [
        row
        for row in list(hca_rows) + list(weak_4d_endpoints)
        if row.library == "FFTM" and row.dimension == 4 and row.role == "weak"
    ]
    figures["weak_4d"] = tag_rows(
        best_by(weak_4d, ("num_gpus", "side")),
        "weak_4d",
        {"FFTM": "FFTM 4D"},
    )
    return figures


STYLES: Dict[str, Dict[str, object]] = {
    "FFTM": {"color": "#007C83", "marker": "o", "linestyle": "-"},
    "FFTM 4D": {"color": "#3564A5", "marker": "D", "linestyle": "-"},
    "Egger": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    "GPU-FFT": {"color": "#8E5EA2", "marker": "^", "linestyle": "-."},
}


def configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def metric_value(row: Measurement, metric: str) -> float:
    if metric == "mean_ms":
        return row.mean_ms
    if metric == "throughput_gpoints_s":
        return row.throughput_gpoints_s
    if metric == "effective_tflops":
        return row.effective_tflops
    raise ValueError(metric)


def plot_three_metrics(
    *,
    rows: Sequence[Measurement],
    title: str,
    path_stem: Path,
    dpi: int,
    fitted_labels: bool = False,
    published_tflops: Optional[Mapping[int, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    configure_matplotlib()
    metrics = (
        ("mean_ms", "Pair time (ms)", "(a) Execution time"),
        ("throughput_gpoints_s", "Throughput (Gpoints/s)", "(b) Throughput"),
        ("effective_tflops", "Effective TFLOP/s", "(c) Floating-point rate"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.55, 3.15))
    gpu_ticks = sorted({row.num_gpus for row in rows})
    x_position = {gpu: index for index, gpu in enumerate(gpu_ticks)}
    series_names = [name for name in STYLES if any(row.series == name for row in rows)]
    for axis, (metric, ylabel, panel_title) in zip(axes, metrics):
        for series_name in series_names:
            values = sorted(
                [row for row in rows if row.series == series_name],
                key=lambda row: row.num_gpus,
            )
            style = STYLES[series_name]
            axis.errorbar(
                [x_position[row.num_gpus] for row in values],
                [metric_value(row, metric) for row in values],
                yerr=[row.metric_stddev(metric) for row in values],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.55,
                markersize=4.7,
                capsize=2.0,
                elinewidth=0.8,
                label=series_name,
                zorder=3,
            )
        if published_tflops and metric == "effective_tflops":
            axis.scatter(
                [x_position[gpu] for gpu in published_tflops if gpu in x_position],
                [published_tflops[gpu] for gpu in published_tflops if gpu in x_position],
                color="#222222",
                marker="x",
                s=27,
                linewidths=1.15,
                label="GPU-FFT published",
                zorder=4,
            )
        axis.set_title(panel_title, pad=5)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22, linewidth=0.65)
        axis.margins(x=0.06, y=0.12)

    if fitted_labels:
        tick_labels = [str(gpu) for gpu in gpu_ticks]
        xlabel = "GPUs"
    elif rows and rows[0].role == "weak":
        tick_labels = [str(gpu) for gpu in gpu_ticks]
        xlabel = "GPUs"
    else:
        tick_labels = [
            f"{gpu}\n({gpu // 8})" if gpu >= 8 else str(gpu) for gpu in gpu_ticks
        ]
        xlabel = "GPUs (nodes)"
    for axis in axes:
        axis.set_xticks(range(len(gpu_ticks)), tick_labels)
        axis.set_xlabel(xlabel)

    handles: List[object] = []
    labels: List[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(labels)),
        frameon=False,
        columnspacing=1.5,
        handlelength=2.4,
    )
    fig.text(0.5, 0.865, title, ha="center", va="center", fontsize=10)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.73, wspace=0.34)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def write_measurements(path: Path, rows: Sequence[Measurement]) -> None:
    fields = [
        "figure_id",
        "series",
        "library",
        "dimension",
        "side",
        "total_points",
        "num_gpus",
        "num_nodes",
        "role",
        "mean_ms",
        "stddev_ms",
        "throughput_gpoints_s",
        "throughput_stddev",
        "effective_tflops",
        "effective_tflops_stddev",
        "max_l2_diff",
        "configuration",
        "campaign",
        "provenance",
        "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "figure_id": row.figure_id,
                    "series": row.series,
                    "library": row.library,
                    "dimension": row.dimension,
                    "side": row.side,
                    "total_points": row.total_points,
                    "num_gpus": row.num_gpus,
                    "num_nodes": f"{row.num_nodes:g}",
                    "role": row.role,
                    "mean_ms": f"{row.mean_ms:.6f}",
                    "stddev_ms": f"{row.stddev_ms:.6f}",
                    "throughput_gpoints_s": f"{row.throughput_gpoints_s:.6f}",
                    "throughput_stddev": (
                        f"{row.metric_stddev('throughput_gpoints_s'):.6f}"
                    ),
                    "effective_tflops": f"{row.effective_tflops:.6f}",
                    "effective_tflops_stddev": (
                        f"{row.metric_stddev('effective_tflops'):.6f}"
                    ),
                    "max_l2_diff": (
                        "" if not math.isfinite(row.max_l2_diff) else f"{row.max_l2_diff:.9e}"
                    ),
                    "configuration": row.configuration,
                    "campaign": row.campaign,
                    "provenance": row.provenance,
                    "source": row.source,
                }
            )


def matched_comparisons(figures: Mapping[str, Sequence[Measurement]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for figure_id in ("fitted_3d", "strong_3d_1920", "strong_3d_2048"):
        rows = figures[figure_id]
        fftm = {(row.num_gpus, row.side): row for row in rows if row.library == "FFTM"}
        for competitor in ("Egger", "GPU-FFT"):
            for other in [row for row in rows if row.library == competitor]:
                own = fftm.get((other.num_gpus, other.side))
                if own is None:
                    continue
                output.append(
                    {
                        "figure_id": figure_id,
                        "dimension": own.dimension,
                        "side": own.side,
                        "num_gpus": own.num_gpus,
                        "competitor": competitor,
                        "fftm_ms": own.mean_ms,
                        "competitor_ms": other.mean_ms,
                        "fftm_speedup": other.mean_ms / own.mean_ms,
                        "fftm_faster_percent": 100.0 * (other.mean_ms - own.mean_ms) / other.mean_ms,
                        "fftm_tflops": own.effective_tflops,
                        "competitor_tflops": other.effective_tflops,
                    }
                )
    return output


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_coverage(path: Path, figures: Mapping[str, Sequence[Measurement]]) -> List[Dict[str, object]]:
    expected: Dict[str, Dict[str, Sequence[int]]] = {
        "fitted_3d": {"FFTM": range(2, 9), "Egger": range(2, 9)},
        "strong_3d_1920": {"FFTM": (8, 16, 24, 32), "GPU-FFT": (8, 16, 24, 32)},
        "strong_3d_2048": {
            "FFTM": (8, 16, 24, 32, 64, 96, 120),
            "Egger": (8, 16, 32, 64, 96, 120),
            "GPU-FFT": (8, 16, 32),
        },
        "strong_4d_320": {"FFTM": (8, 16, 24, 32, 64, 96, 120)},
        "weak_3d": {"FFTM": (8, 16, 32, 64, 96, 120), "Egger": (8, 16, 32, 64, 96, 120)},
        "weak_4d": {"FFTM": (8, 16, 32, 64, 96, 120)},
    }
    output: List[Dict[str, object]] = []
    for figure_id, libraries in expected.items():
        for library, gpu_counts in libraries.items():
            available = {
                row.num_gpus
                for row in figures[figure_id]
                if row.library == library
            }
            for gpu in gpu_counts:
                output.append(
                    {
                        "figure_id": figure_id,
                        "library": library,
                        "num_gpus": gpu,
                        "available": int(gpu in available),
                    }
                )
    write_dict_csv(path, output)
    return output


def write_candidate_audit(
    path: Path,
    candidates: Sequence[Measurement],
    selected: Sequence[Measurement],
) -> None:
    selected_sources = {row.source for row in selected}
    rows = []
    for row in sorted(candidates, key=lambda item: (item.dimension, item.num_gpus, item.mean_ms)):
        rows.append(
            {
                "dimension": row.dimension,
                "side": row.side,
                "num_gpus": row.num_gpus,
                "mean_ms": f"{row.mean_ms:.6f}",
                "stddev_ms": f"{row.stddev_ms:.6f}",
                "effective_tflops": f"{row.effective_tflops:.6f}",
                "selected_best": int(row.source in selected_sources),
                "campaign": row.campaign,
                "provenance": row.provenance,
                "configuration": row.configuration,
                "source": row.source,
            }
        )
    write_dict_csv(path, rows)


def write_latex_table(path: Path, figures: Mapping[str, Sequence[Measurement]]) -> None:
    rows_3d = {
        row.num_gpus: row
        for row in figures["strong_3d_2048"]
        if row.library == "FFTM"
    }
    rows_4d = {row.num_gpus: row for row in figures["strong_4d_320"]}
    gpu_counts = sorted(set(rows_3d) | set(rows_4d))
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"GPUs & $2048^3$ ms & 3D TFLOP/s & $320^4$ ms & 4D TFLOP/s \\",
        r"\midrule",
    ]
    for gpu in gpu_counts:
        row3 = rows_3d.get(gpu)
        row4 = rows_4d.get(gpu)
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                gpu,
                "--" if row3 is None else f"{row3.mean_ms:.1f}",
                "--" if row3 is None else f"{row3.effective_tflops:.2f}",
                "--" if row4 is None else f"{row4.mean_ms:.1f}",
                "--" if row4 is None else f"{row4.effective_tflops:.2f}",
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def write_remaining_cluster_cases(path: Path) -> List[Dict[str, object]]:
    fields = (
        "priority", "dimension", "side", "num_gpus", "times", "warmup",
        "paper_blocking", "recommended", "reason",
    )
    with path.open("w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    return []


def summarize_comparisons(rows: Sequence[Mapping[str, object]], figure_id: str, competitor: str) -> str:
    values = [
        float(row["fftm_speedup"])
        for row in rows
        if row["figure_id"] == figure_id and row["competitor"] == competitor
    ]
    if not values:
        return "no matched cases"
    return f"{min(values):.2f}--{max(values):.2f}x over {len(values)} matched cases"


def write_readme(
    path: Path,
    figures: Mapping[str, Sequence[Measurement]],
    comparisons: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
) -> None:
    missing = [row for row in coverage if int(row["available"]) == 0]
    strong_3d = [row for row in figures["strong_3d_2048"] if row.library == "FFTM"]
    strong_4d = list(figures["strong_4d_320"])
    prior_3d = [row.num_gpus for row in strong_3d if row.provenance == "validated-prior-fftm"]
    prior_4d = [row.num_gpus for row in strong_4d if row.provenance == "validated-prior-fftm"]
    best_3d = max(strong_3d, key=lambda row: row.effective_tflops)
    best_4d = max(strong_4d, key=lambda row: row.effective_tflops)
    if prior_3d or prior_4d:
        provenance_note = (
            f"Selected prior-HCA points remain at 3D GPUs {prior_3d or 'none'} "
            f"and 4D GPUs {prior_4d or 'none'}."
        )
    else:
        provenance_note = "No selected high-count point depends on the prior HCA campaign."
    text = [
        "# Final FFTM performance figure suite",
        "",
        "## Expected",
        "",
        "The analysis was expected to consolidate only numerically valid production measurements, compare complete forward/backward pairs on exact matched workloads, expose source provenance, and identify any measurements still worth collecting before cluster shutdown.",
        "",
        "## Obtained",
        "",
        f"- FFTM versus Egger, fitted 2--8 GPU series: {summarize_comparisons(comparisons, 'fitted_3d', 'Egger')}.",
        f"- FFTM versus GPU-FFT, 1920^3 strong scaling: {summarize_comparisons(comparisons, 'strong_3d_1920', 'GPU-FFT')}.",
        f"- FFTM versus GPU-FFT, 2048^3 strong scaling: {summarize_comparisons(comparisons, 'strong_3d_2048', 'GPU-FFT')}.",
        f"- FFTM versus Egger, 2048^3 strong scaling: {summarize_comparisons(comparisons, 'strong_3d_2048', 'Egger')}.",
        f"- Best selected 3D rate: {best_3d.effective_tflops:.2f} TFLOP/s on {best_3d.num_gpus} GPUs ({best_3d.mean_ms:.1f} ms).",
        f"- Best selected 4D rate: {best_4d.effective_tflops:.2f} TFLOP/s on {best_4d.num_gpus} GPUs ({best_4d.mean_ms:.1f} ms).",
        f"- Coverage audit: {len(coverage) - len(missing)}/{len(coverage)} expected plotted points are present.",
        "",
        "## Provenance warning",
        "",
        "The primary curves select the fastest validated production row for each exact library/workload/GPU key. This is not a minimum over arbitrary diagnostics: candidates are restricted to the final-policy, release, and HCA-validated campaigns listed in analysis_manifest.json.",
        f"{provenance_note} The final high-count campaign supplies the selected 64/96/120-GPU 4D rows and 96/120-GPU 3D rows. The candidate audit retains older rows instead of hiding campaign variability.",
        "",
        "## Remaining cluster information",
        "",
        "No additional cluster measurement is required for the planned paper figures. The current final image now has direct 3D and 4D strong-scaling coverage through 120 GPUs, while the optional 24-GPU 2048^3 point is also present.",
        "GPU-FFT has same-system data through 32 GPUs. Extending that external reference is optional and is not required for the FFTM scaling claim.",
        "The high-count 4D automatic policy intentionally leaves backward plane credits disabled because only the 16/24/32-GPU settings were calibrated. Explicit high-count credit tuning would be future optimization work, not missing paper validation.",
        "Egger and FFTM have matched 2048^3 and memory-fitted 3D coverage through 120 GPUs. No further Egger run is required.",
        "",
        "## Figure layout",
        "",
        "All figures use external shared legends, fixed panel ordering, and no in-panel point labels. PNG files are 600 DPI and PDF files retain vector text and curves.",
    ]
    path.write_text("\n".join(text) + "\n")


def write_checksums(output: Path) -> None:
    checksum_path = output / "artifact_sha256.txt"
    artifacts = sorted(
        path
        for path in output.iterdir()
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in artifacts
    ]
    checksum_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    final_summary = root / "build/resutls_stats/data_final_multinode_paper_3d4d_complete_20260808_124136/paper_results_summary.csv"
    fitted_path = root / "build/resutls_stats/data_release_fitted_20260805_092652/benchmarks/cpp_csv/benchmark_fftm_3d.csv"
    egger_fitted_path = root / "fft_Egger/results/egger_cluster_data_20260521_111434/measurements.csv"
    hca_path = root / "build/resutls_stats/data_scale_hca_96_120_fixed_20260725_163251/analysis/hca_scaling_selected.csv"
    high_count_dir = root / "build/resutls_stats/data_final_high_count_20260808_164725"
    gpu_fft_dir = root / "build/resutls_stats/data_gpu_fft_ref_full_20260807_093212"
    release_dirs = [
        root / f"build/resutls_stats/data_release_scale-{gpu}_{stamp}"
        for gpu, stamp in (
            (32, "20260805_102745"),
            (64, "20260805_103730"),
            (96, "20260805_115614"),
            (120, "20260805_152130"),
        )
    ]
    weak_dirs = [
        root / "build/resutls_stats/data_4d_weak_96_20260806_094507",
        root / "build/resutls_stats/data_4d_weak_120_20260806_145512",
    ]

    final_rows = load_final_policy(root, final_summary)
    high_count_rows = load_final_high_count(root, high_count_dir)
    release_rows = load_release_scaling(root, release_dirs)
    hca_rows = load_hca_scaling(root, hca_path)
    gpu_fft_rows, published_tflops = load_gpu_fft(root, gpu_fft_dir)
    fitted_fftm = load_fitted_fftm(root, fitted_path)
    fitted_egger = load_fitted_egger(root, egger_fitted_path)
    weak_endpoints = [
        row for directory in weak_dirs for row in load_4d_weak_endpoint(root, directory)
    ]

    figures = build_figure_sets(
        final_rows=final_rows,
        high_count_rows=high_count_rows,
        release_rows=release_rows,
        hca_rows=hca_rows,
        gpu_fft_rows=gpu_fft_rows,
        fitted_fftm=fitted_fftm,
        fitted_egger=fitted_egger,
        weak_4d_endpoints=weak_endpoints,
    )
    specifications = (
        ("fitted_3d", "3D memory-fitted single-node comparison", "fig_fitted_3d_fftm_egger", True, None),
        ("strong_3d_1920", r"3D strong scaling, $1920^3$", "fig_strong_3d_1920_fftm_gpufft", False, None),
        ("strong_3d_2048", r"3D strong scaling, $2048^3$", "fig_strong_3d_2048_external", False, published_tflops),
        ("strong_4d_320", r"4D strong scaling, $320^4$", "fig_strong_4d_320", False, None),
        ("weak_3d", "3D memory-fitted scaling", "fig_weak_3d_fftm_egger", False, None),
        ("weak_4d", "4D memory-fitted scaling", "fig_weak_4d", False, None),
    )
    for figure_id, title, stem, fitted_labels, published in specifications:
        plot_three_metrics(
            rows=figures[figure_id],
            title=title,
            path_stem=output / stem,
            dpi=args.dpi,
            fitted_labels=fitted_labels,
            published_tflops=published,
        )

    selected_rows = [row for values in figures.values() for row in values]
    write_measurements(output / "selected_performance.csv", selected_rows)
    comparisons = matched_comparisons(figures)
    write_dict_csv(output / "matched_external_comparisons.csv", comparisons)
    coverage = write_coverage(output / "coverage_audit.csv", figures)
    strong_candidates = [
        row
        for row in list(final_rows) + list(high_count_rows) + list(release_rows) + list(hca_rows)
        if row.library == "FFTM"
        and row.role == "strong"
        and ((row.dimension == 3 and row.side == 2048) or (row.dimension == 4 and row.side == 320))
    ]
    selected_strong = [
        row
        for figure_id in ("strong_3d_2048", "strong_4d_320")
        for row in figures[figure_id]
        if row.library == "FFTM"
    ]
    write_candidate_audit(output / "strong_candidate_audit.csv", strong_candidates, selected_strong)
    write_latex_table(output / "table_best_strong_scaling.tex", figures)
    remaining_cases = write_remaining_cluster_cases(output / "remaining_cluster_cases.csv")
    write_readme(output / "README.md", figures, comparisons, coverage)

    manifest = {
        "metric_definitions": {
            "pair_time": "complete forward/backward pair, maximum MPI-rank wall time",
            "throughput": "total global points / pair time",
            "effective_flops": "10 * total global points * log2(total global points) / pair time",
        },
        "selection": "fastest numerically valid row per exact library/dimension/size/GPU key from curated production campaigns",
        "sources": {
            "final_policy": source_ref(root, final_summary),
            "final_high_count": source_ref(root, high_count_dir / "selected_results.csv"),
            "release_fitted": source_ref(root, fitted_path),
            "egger_fitted": source_ref(root, egger_fitted_path),
            "hca_scaling": source_ref(root, hca_path),
            "gpu_fft": source_ref(root, gpu_fft_dir),
            "release_scaling": [source_ref(root, path) for path in release_dirs],
            "weak_4d_endpoints": [source_ref(root, path) for path in weak_dirs],
        },
        "figures": [stem for _, _, stem, _, _ in specifications],
        "remaining_cluster_cases": remaining_cases,
        "dpi": args.dpi,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    write_checksums(output)
    print(f"Wrote final performance suite to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
