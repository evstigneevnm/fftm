#!/usr/bin/env python3
"""Plot matched 3D and 4D pipeline throughput against total transform size."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PAIR_FLOP_FACTOR = 10.0
TARGET_GPUS = (16, 32)


@dataclass
class Measurement:
    library: str
    pipeline: str
    dimension: int
    side: int
    num_gpus: int
    mean_ms: float
    stddev_ms: float
    max_l2_diff: float
    valid: bool
    dataset: str
    source: str

    @property
    def total_points(self) -> int:
        return self.side**self.dimension

    @property
    def effective_tflops(self) -> float:
        pair_flops = (
            PAIR_FLOP_FACTOR
            * float(self.total_points)
            * math.log2(float(self.total_points))
        )
        return pair_flops / (self.mean_ms / 1000.0) / 1.0e12

    @property
    def effective_tflops_stddev(self) -> float:
        if self.stddev_ms <= 0.0:
            return 0.0
        return self.effective_tflops * self.stddev_ms / self.mean_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fftm-3d-dir",
        type=Path,
        required=True,
        help="Current FFTM 3D A/B directory containing slab/ and pencil/.",
    )
    parser.add_argument(
        "--fftm-4d-dir",
        type=Path,
        required=True,
        help="Current FFTM 4D slab/pencil WZ A/B directory.",
    )
    parser.add_argument(
        "--gpu-fft-dir",
        type=Path,
        required=True,
        help="GPU-FFT reference result directory.",
    )
    parser.add_argument(
        "--egger-scaling-csv",
        type=Path,
        required=True,
        help="Validated HCA-pinned FFTM/Egger selected-scaling CSV.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json_lines(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_fftm_3d(root: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for branch in ("slab", "pencil"):
        branch_dir = root / branch
        records = read_json_lines(branch_dir / "runs.jsonl")
        successful: set[Tuple[str, str, int, int]] = set()
        for record in records:
            if int(record.get("returncode", 1)) != 0:
                continue
            spec = record.get("spec", {})
            sizes = record.get("sizes", [])
            if not isinstance(spec, dict) or not isinstance(sizes, list) or not sizes:
                raise RuntimeError(f"Malformed run record in {branch_dir / 'runs.jsonl'}")
            successful.add(
                (
                    str(spec.get("strategy", "")),
                    str(spec.get("mode", "")),
                    int(spec.get("num_gpus", 0)),
                    int(sizes[0]),
                )
            )

        benchmark = branch_dir / "cpp_csv" / "benchmark_fftm_3d.csv"
        for line_number, row in enumerate(read_csv(benchmark), start=2):
            num_gpus = int(row["num_gpus"])
            side = int(row["nx"])
            if num_gpus not in TARGET_GPUS or side not in (1920, 2048):
                continue
            key = (row["strategy"], row["mode"], num_gpus, side)
            max_l2 = float(row["max_l2_diff"])
            valid = (
                key in successful
                and math.isfinite(max_l2)
                and max_l2 <= float(row["epsilon"])
            )
            if row["strategy"] == "slab-pencil" and row["mode"] == "p2p-waitany":
                pipeline = "FFTM 3D slab / P2P"
            elif row["strategy"] == "slab-pencil" and row["mode"] == "alltoallv":
                pipeline = "FFTM 3D slab / Alltoallv"
            elif row["strategy"] == "pencil-pencil":
                pipeline = "FFTM 3D pencil"
            else:
                continue
            output.append(
                Measurement(
                    library="FFTM",
                    pipeline=pipeline,
                    dimension=3,
                    side=side,
                    num_gpus=num_gpus,
                    mean_ms=float(row["avg_wall_ms"]),
                    stddev_ms=float(row["stddev_wall_ms"]),
                    max_l2_diff=max_l2,
                    valid=valid,
                    dataset="current-fftm-3d-pipeline-ab",
                    source=f"{benchmark.resolve()}:{line_number}",
                )
            )
    return output


def load_fftm_4d(root: Path) -> List[Measurement]:
    records = read_json_lines(root / "runs.jsonl")
    successful: set[Tuple[str, int, int]] = set()
    for record in records:
        if int(record.get("returncode", 1)) != 0:
            continue
        spec = record.get("spec", {})
        if not isinstance(spec, dict):
            raise RuntimeError(f"Malformed run record in {root / 'runs.jsonl'}")
        enabled = int(bool(spec.get("fftm_4d_pencil_p3_degenerate_wz_pipeline", False)))
        successful.add((str(spec.get("strategy", "")), int(spec.get("num_gpus", 0)), enabled))

    benchmark = root / "cpp_csv" / "benchmark_fftm_4d.csv"
    output: List[Measurement] = []
    for line_number, row in enumerate(read_csv(benchmark), start=2):
        num_gpus = int(row["num_gpus"])
        side = int(row["nx"])
        if num_gpus not in TARGET_GPUS or side != 320:
            continue
        enabled = int(row["pencil_p3_degenerate_wz_pipeline"])
        key = (row["strategy"], num_gpus, enabled)
        max_l2 = float(row["max_l2_diff"])
        valid = (
            key in successful
            and math.isfinite(max_l2)
            and max_l2 <= float(row["epsilon"])
        )
        if row["strategy"] == "slab-slab":
            pipeline = "FFTM 4D slab"
        elif enabled:
            pipeline = "FFTM 4D pencil / WZ"
        else:
            pipeline = "FFTM 4D pencil / legacy"
        output.append(
            Measurement(
                library="FFTM",
                pipeline=pipeline,
                dimension=4,
                side=side,
                num_gpus=num_gpus,
                mean_ms=float(row["avg_wall_ms"]),
                stddev_ms=float(row["stddev_wall_ms"]),
                max_l2_diff=max_l2,
                valid=valid,
                dataset="current-fftm-4d-pipeline-ab",
                source=f"{benchmark.resolve()}:{line_number}",
            )
        )
    return output


def load_gpu_fft(root: Path) -> List[Measurement]:
    comparison = root / "analysis" / "gpu_fft_reference_comparison.csv"
    output: List[Measurement] = []
    for row in read_csv(comparison):
        num_gpus = int(row["gpus"])
        side = int(row["size"])
        if num_gpus not in TARGET_GPUS or side not in (1920, 2048):
            continue
        summary_path = root / row["source_file"]
        summary = read_csv(summary_path)
        if len(summary) != 1:
            raise RuntimeError(f"Expected one summary row in {summary_path}")
        detail = summary[0]
        output.append(
            Measurement(
                library="GPU-FFT",
                pipeline="GPU-FFT 3D",
                dimension=3,
                side=side,
                num_gpus=num_gpus,
                mean_ms=float(row["avg_pair_ms"]),
                stddev_ms=float(detail["stddev_pair_ms"]),
                max_l2_diff=float(row["relative_l2"]),
                valid=row["valid"] == "1" and detail["valid"] == "1",
                dataset="current-gpu-fft",
                source=str(summary_path.resolve()),
            )
        )
    return output


def load_egger(path: Path) -> List[Measurement]:
    output: List[Measurement] = []
    for line_number, row in enumerate(read_csv(path), start=2):
        if (
            row["library"] != "Egger"
            or row["dimension"] != "3"
            or row["scaling_role"] != "strong"
            or int(row["size"]) != 2048
            or int(row["num_gpus"]) not in TARGET_GPUS
        ):
            continue
        output.append(
            Measurement(
                library="Egger",
                pipeline="Egger 3D",
                dimension=3,
                side=2048,
                num_gpus=int(row["num_gpus"]),
                mean_ms=float(row["mean_ms"]),
                stddev_ms=float(row["stddev_ms"]),
                max_l2_diff=float("nan"),
                valid=True,
                dataset="validated-hca-egger-scaling",
                source=f"{path.resolve()}:{line_number}",
            )
        )
    return output


def configure_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


PIPELINE_ORDER = (
    "FFTM 3D slab / P2P",
    "FFTM 3D slab / Alltoallv",
    "FFTM 3D pencil",
    "GPU-FFT 3D",
    "Egger 3D",
    "FFTM 4D slab",
    "FFTM 4D pencil / legacy",
    "FFTM 4D pencil / WZ",
)


PIPELINE_STYLE: Dict[str, Dict[str, object]] = {
    "FFTM 3D slab / P2P": {"color": "#007C83", "marker": "o", "offset": -0.20},
    "FFTM 3D slab / Alltoallv": {"color": "#4C956C", "marker": "v", "offset": -0.10},
    "FFTM 3D pencil": {"color": "#3564A5", "marker": "D", "offset": 0.00},
    "GPU-FFT 3D": {"color": "#9B4F96", "marker": "^", "offset": 0.10},
    "Egger 3D": {"color": "#D55E00", "marker": "s", "offset": 0.20},
    "FFTM 4D slab": {"color": "#333333", "marker": "P", "offset": -0.11},
    "FFTM 4D pencil / legacy": {"color": "#8A8A8A", "marker": "X", "offset": 0.00},
    "FFTM 4D pencil / WZ": {"color": "#C43C39", "marker": "*", "offset": 0.11},
}


SHAPE_INDEX = {(3, 1920): 0, (3, 2048): 1, (4, 320): 2}
SHAPE_LABELS = (
    "$1920^3$\n$7.078\\times10^9$ points",
    "$2048^3$\n$8.590\\times10^9$ points",
    "$320^4$\n$10.486\\times10^9$ points",
)


def make_plot(rows: Sequence[Measurement], output_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.8), sharey=True)
    for panel_index, (axis, num_gpus) in enumerate(zip(axes, TARGET_GPUS)):
        panel_rows = [row for row in rows if row.num_gpus == num_gpus and row.valid]
        for pipeline in PIPELINE_ORDER:
            values = sorted(
                [row for row in panel_rows if row.pipeline == pipeline],
                key=lambda row: row.total_points,
            )
            if not values:
                continue
            style = PIPELINE_STYLE[pipeline]
            x_values = [
                SHAPE_INDEX[(row.dimension, row.side)] + float(style["offset"])
                for row in values
            ]
            axis.errorbar(
                x_values,
                [row.effective_tflops for row in values],
                yerr=[row.effective_tflops_stddev for row in values],
                color=str(style["color"]),
                marker=str(style["marker"]),
                linestyle="-" if len(values) > 1 else "none",
                linewidth=1.2,
                markersize=5.2 if style["marker"] != "*" else 7.0,
                markeredgewidth=0.8,
                capsize=2.2,
                label=pipeline,
                zorder=4,
            )

        axis.set_xticks(range(len(SHAPE_LABELS)), SHAPE_LABELS)
        axis.set_xlim(-0.43, 2.43)
        axis.set_xlabel(r"Global transform size, $N=\prod_d N_d$", labelpad=8)
        axis.grid(axis="y", alpha=0.24, linewidth=0.7)
        axis.set_title(
            f"({'ab'[panel_index]}) {num_gpus} GPUs ({num_gpus // 8} nodes)",
            loc="left",
            fontweight="bold",
        )
    axes[0].set_ylabel("Effective throughput (TFLOP/s)")
    plotted_rows = [row for row in rows if row.valid]
    y_max = max(
        row.effective_tflops + row.effective_tflops_stddev
        for row in plotted_rows
    )
    axes[0].set_ylim(0.0, 1.08 * y_max)

    handles: List[object] = []
    labels: List[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    ordered = [
        (handles[labels.index(label)], label)
        for label in PIPELINE_ORDER
        if label in labels
    ]
    fig.legend(
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.text(
        0.5,
        0.015,
        r"Effective rate uses $10N\log_2N$ operations per forward+backward pair; "
        "error bars propagate the measured time standard deviation.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.76, wspace=0.13)
    fig.savefig(output_dir / "fig_3d4d_pipeline_tflops.png", dpi=dpi)
    fig.savefig(output_dir / "fig_3d4d_pipeline_tflops.pdf")
    plt.close(fig)


def write_evidence(path: Path, rows: Iterable[Measurement]) -> None:
    fields = (
        "library",
        "pipeline",
        "dimension",
        "shape",
        "total_points",
        "total_points_billion",
        "num_gpus",
        "num_nodes",
        "mean_ms",
        "stddev_ms",
        "effective_tflops",
        "effective_tflops_stddev",
        "max_l2_diff",
        "valid",
        "dataset",
        "source",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item.num_gpus, item.total_points, item.pipeline)):
            writer.writerow(
                {
                    "library": row.library,
                    "pipeline": row.pipeline,
                    "dimension": row.dimension,
                    "shape": f"{row.side}^{row.dimension}",
                    "total_points": row.total_points,
                    "total_points_billion": f"{row.total_points / 1.0e9:.9f}",
                    "num_gpus": row.num_gpus,
                    "num_nodes": row.num_gpus // 8,
                    "mean_ms": f"{row.mean_ms:.6f}",
                    "stddev_ms": f"{row.stddev_ms:.6f}",
                    "effective_tflops": f"{row.effective_tflops:.6f}",
                    "effective_tflops_stddev": f"{row.effective_tflops_stddev:.6f}",
                    "max_l2_diff": (
                        "" if math.isnan(row.max_l2_diff) else f"{row.max_l2_diff:.9e}"
                    ),
                    "valid": int(row.valid),
                    "dataset": row.dataset,
                    "source": row.source,
                }
            )


def write_readme(path: Path, rows: Sequence[Measurement]) -> None:
    excluded = [row for row in rows if not row.valid]
    lines = [
        "# Matched 3D/4D pipeline throughput",
        "",
        "The figure compares matched 16- and 32-GPU measurements by total global transform size.",
        "All plotted FFTM and GPU-FFT rows completed and passed their numerical tolerance.",
        "Egger rows come from the validated HCA-pinned 2048^3 strong-scaling dataset.",
        "",
        "## Selected results",
        "",
    ]
    for num_gpus in TARGET_GPUS:
        selected = [row for row in rows if row.num_gpus == num_gpus and row.valid]
        fastest_3d = max(
            (row for row in selected if row.dimension == 3),
            key=lambda row: row.effective_tflops,
        )
        fastest_4d = max(
            (row for row in selected if row.dimension == 4),
            key=lambda row: row.effective_tflops,
        )
        lines.extend(
            [
                f"- {num_gpus} GPUs: fastest 3D row is {fastest_3d.pipeline} "
                f"at {fastest_3d.effective_tflops:.3f} TFLOP/s.",
                f"- {num_gpus} GPUs: fastest 4D row is {fastest_4d.pipeline} "
                f"at {fastest_4d.effective_tflops:.3f} TFLOP/s.",
            ]
        )
    lines.extend(
        [
            "",
            "Effective throughput uses 10*N*log2(N) operations for a measured forward+backward pair.",
        ]
    )
    if excluded:
        lines.extend(["", "## Excluded evidence", ""])
        for row in excluded:
            lines.append(
                f"- {row.pipeline}, {row.side}^{row.dimension}, {row.num_gpus} GPUs: "
                "launcher did not complete successfully."
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "fftm_3d_dir": args.fftm_3d_dir.resolve(),
        "fftm_4d_dir": args.fftm_4d_dir.resolve(),
        "gpu_fft_dir": args.gpu_fft_dir.resolve(),
        "egger_scaling_csv": args.egger_scaling_csv.resolve(),
    }
    rows = (
        load_fftm_3d(input_paths["fftm_3d_dir"])
        + load_fftm_4d(input_paths["fftm_4d_dir"])
        + load_gpu_fft(input_paths["gpu_fft_dir"])
        + load_egger(input_paths["egger_scaling_csv"])
    )
    expected_counts = {16: 12, 32: 12}
    counts = {num_gpus: sum(row.num_gpus == num_gpus for row in rows) for num_gpus in TARGET_GPUS}
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected evidence counts: {counts}, expected {expected_counts}")
    invalid = [row for row in rows if not row.valid]
    valid_counts = {
        num_gpus: sum(row.num_gpus == num_gpus and row.valid for row in rows)
        for num_gpus in TARGET_GPUS
    }

    evidence_path = output_dir / "pipeline_tflops_evidence.csv"
    write_evidence(evidence_path, rows)
    make_plot(rows, output_dir, args.dpi)
    write_readme(output_dir / "README.md", rows)
    manifest = {
        "inputs": {key: str(value) for key, value in input_paths.items()},
        "target_gpus": list(TARGET_GPUS),
        "evidence_rows": len(rows),
        "evidence_rows_by_gpu": counts,
        "plotted_rows": len(rows) - len(invalid),
        "plotted_rows_by_gpu": valid_counts,
        "excluded_rows": [
            {
                "pipeline": row.pipeline,
                "shape": f"{row.side}^{row.dimension}",
                "num_gpus": row.num_gpus,
                "source": row.source,
            }
            for row in invalid
        ],
        "effective_flops": "10 * total_points * log2(total_points) per forward+backward pair",
        "outputs": [
            evidence_path.name,
            "fig_3d4d_pipeline_tflops.pdf",
            "fig_3d4d_pipeline_tflops.png",
            "README.md",
        ],
        "dpi": args.dpi,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote matched 3D/4D pipeline comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
