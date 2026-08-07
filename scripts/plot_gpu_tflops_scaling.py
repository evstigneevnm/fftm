#!/usr/bin/env python3
"""Plot effective FFT throughput versus GPU count, including multinode runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PAIR_FLOP_FACTOR = 10.0


@dataclass
class ThroughputRow:
    library: str
    dimension: int
    scaling_role: str
    num_gpus: int
    size: int
    variant: str
    mean_ms: float
    stddev_ms: float
    max_l2_diff: float
    source: str

    @property
    def num_nodes(self) -> int:
        if self.num_gpus % 8 != 0:
            raise ValueError(f"GPU count {self.num_gpus} is not divisible by 8")
        return self.num_gpus // 8

    @property
    def effective_tflops(self) -> float:
        points = float(self.size**self.dimension)
        pair_flops = PAIR_FLOP_FACTOR * points * math.log2(points)
        return pair_flops / (self.mean_ms / 1000.0) / 1.0e12

    @property
    def effective_tflops_stddev(self) -> float:
        return self.effective_tflops * self.stddev_ms / self.mean_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-csv",
        type=Path,
        required=True,
        help="Existing hca_scaling_selected.csv with the validated 8-120 GPU rows.",
    )
    parser.add_argument(
        "--fftm-4d-weak-endpoint",
        type=Path,
        action="append",
        default=[],
        help=(
            "Validated 4D weak-scaling endpoint directory. The fastest successful "
            "candidate under full/ replaces the matching selected row."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_selected(path: Path) -> List[ThroughputRow]:
    rows: List[ThroughputRow] = []
    for item in read_csv(path):
        max_l2_text = item.get("max_l2_diff", "").strip()
        rows.append(
            ThroughputRow(
                library=item["library"],
                dimension=int(item["dimension"]),
                scaling_role=item["scaling_role"],
                num_gpus=int(item["num_gpus"]),
                size=int(item["size"]),
                variant=item["variant"],
                mean_ms=float(item["mean_ms"]),
                stddev_ms=float(item["stddev_ms"]),
                max_l2_diff=float(max_l2_text) if max_l2_text else float("nan"),
                source=str(path.resolve()),
            )
        )
    return rows


def load_json_lines(path: Path) -> List[Dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def endpoint_candidate(candidate_dir: Path) -> ThroughputRow:
    benchmark_path = candidate_dir / "cpp_csv" / "benchmark_fftm_4d.csv"
    wall_path = candidate_dir / "cpp_csv" / "wall_times_4d_r0.csv"
    runs_path = candidate_dir / "runs.jsonl"
    benchmark_rows = read_csv(benchmark_path)
    if len(benchmark_rows) != 1:
        raise RuntimeError(f"Expected one benchmark row in {benchmark_path}")
    runs = load_json_lines(runs_path)
    if len(runs) != 1 or int(runs[0].get("returncode", 1)) != 0:
        raise RuntimeError(f"Endpoint candidate did not complete: {candidate_dir}")

    benchmark = benchmark_rows[0]
    samples = [float(item["global_wall_ms"]) for item in read_csv(wall_path)]
    expected_samples = int(benchmark["times"])
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"{candidate_dir} has {len(samples)} samples, expected {expected_samples}"
        )
    mean_ms = statistics.mean(samples)
    reported_mean_ms = float(benchmark["avg_wall_ms"])
    if abs(mean_ms - reported_mean_ms) > 0.01:
        raise RuntimeError(
            f"{candidate_dir} sample mean {mean_ms} != reported {reported_mean_ms}"
        )
    max_l2_diff = float(benchmark["max_l2_diff"])
    if not math.isfinite(max_l2_diff) or max_l2_diff > float(benchmark["epsilon"]):
        raise RuntimeError(f"Invalid numerical result in {candidate_dir}")

    concurrency = int(benchmark["slab_native_wz_plan_concurrency"])
    chunk_mib = int(benchmark["native_xw_chunk_mib"])
    return ThroughputRow(
        library="FFTM",
        dimension=4,
        scaling_role="weak",
        num_gpus=int(benchmark["num_gpus"]),
        size=int(benchmark["nx"]),
        variant=f"c{concurrency}-m{chunk_mib}",
        mean_ms=mean_ms,
        stddev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        max_l2_diff=max_l2_diff,
        source=str(candidate_dir.resolve()),
    )


def load_endpoint(root: Path) -> ThroughputRow:
    root = root.resolve()
    if not (root / "PASSED").is_file():
        raise RuntimeError(f"Endpoint lacks PASSED marker: {root}")
    candidates = [
        endpoint_candidate(path)
        for path in sorted((root / "full").glob("c*_m*"))
        if (path / "cpp_csv" / "benchmark_fftm_4d.csv").is_file()
    ]
    if not candidates:
        raise RuntimeError(f"No complete full candidates found under {root}")
    gpu_counts = {item.num_gpus for item in candidates}
    sizes = {item.size for item in candidates}
    if len(gpu_counts) != 1 or len(sizes) != 1:
        raise RuntimeError(f"Mixed endpoint problem definitions under {root}")
    return min(candidates, key=lambda item: item.mean_ms)


def merge_rows(
    selected: Sequence[ThroughputRow], endpoints: Iterable[ThroughputRow]
) -> List[ThroughputRow]:
    keyed: Dict[Tuple[str, int, str, int], ThroughputRow] = {
        (row.library, row.dimension, row.scaling_role, row.num_gpus): row
        for row in selected
    }
    for row in endpoints:
        keyed[(row.library, row.dimension, row.scaling_role, row.num_gpus)] = row
    return sorted(
        keyed.values(),
        key=lambda row: (
            row.scaling_role,
            row.library,
            row.dimension,
            row.num_gpus,
        ),
    )


def configure_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def series(
    rows: Sequence[ThroughputRow], library: str, dimension: int, role: str
) -> List[ThroughputRow]:
    return sorted(
        [
            row
            for row in rows
            if row.library == library
            and row.dimension == dimension
            and row.scaling_role == role
        ],
        key=lambda row: row.num_gpus,
    )


def plot(rows: Sequence[ThroughputRow], output_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    configure_plot_style()
    styles = {
        ("FFTM", 3): ("#167D8D", "o", "-", "FFTM 3D"),
        ("Egger", 3): ("#D55E00", "s", "--", "Egger 3D"),
        ("FFTM", 4): ("#6A5ACD", "^", "-", "FFTM 4D"),
    }
    gpu_counts = sorted({row.num_gpus for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.25), constrained_layout=True)
    for axis, role in zip(axes, ("strong", "weak")):
        for (library, dimension), (color, marker, linestyle, label) in styles.items():
            values = series(rows, library, dimension, role)
            if not values:
                continue
            axis.errorbar(
                [row.num_gpus for row in values],
                [row.effective_tflops for row in values],
                yerr=[row.effective_tflops_stddev for row in values],
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.6,
                markersize=5,
                capsize=2.5,
                label=label,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks(gpu_counts, [str(value) for value in gpu_counts])
        axis.set_xlabel("GPUs")
        axis.set_ylabel("Effective throughput (TFLOP/s)")
        axis.set_title(f"{role.capitalize()} scaling")
        axis.grid(alpha=0.25)

        nodes_axis = axis.secondary_xaxis("top")
        nodes_axis.set_xticks(gpu_counts, [str(value // 8) for value in gpu_counts])
        nodes_axis.set_xlabel("Nodes (8 GPUs per node)", labelpad=5)
        nodes_axis.tick_params(labelsize=8)

    axes[0].legend(frameon=False, loc="best")
    png_path = output_dir / "fig_tflops_vs_gpus_multinode.png"
    pdf_path = output_dir / "fig_tflops_vs_gpus_multinode.pdf"
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_rows(path: Path, rows: Sequence[ThroughputRow]) -> None:
    fields = [
        "library",
        "dimension",
        "scaling_role",
        "num_gpus",
        "num_nodes",
        "size",
        "variant",
        "mean_ms",
        "stddev_ms",
        "effective_tflops",
        "effective_tflops_stddev",
        "max_l2_diff",
        "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "library": row.library,
                    "dimension": row.dimension,
                    "scaling_role": row.scaling_role,
                    "num_gpus": row.num_gpus,
                    "num_nodes": row.num_nodes,
                    "size": row.size,
                    "variant": row.variant,
                    "mean_ms": f"{row.mean_ms:.6f}",
                    "stddev_ms": f"{row.stddev_ms:.6f}",
                    "effective_tflops": f"{row.effective_tflops:.6f}",
                    "effective_tflops_stddev": (
                        f"{row.effective_tflops_stddev:.6f}"
                    ),
                    "max_l2_diff": (
                        ""
                        if math.isnan(row.max_l2_diff)
                        else f"{row.max_l2_diff:.9e}"
                    ),
                    "source": row.source,
                }
            )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected(args.selected_csv.resolve())
    endpoints = [load_endpoint(path) for path in args.fftm_4d_weak_endpoint]
    rows = merge_rows(selected, endpoints)

    csv_path = output_dir / "tflops_vs_gpus_selected.csv"
    write_rows(csv_path, rows)
    plot(rows, output_dir, args.dpi)
    manifest = {
        "selected_csv": str(args.selected_csv.resolve()),
        "fftm_4d_weak_endpoints": [
            str(path.resolve()) for path in args.fftm_4d_weak_endpoint
        ],
        "endpoint_winners": [asdict(row) for row in endpoints],
        "effective_flops": (
            "10 * total_points * log2(total_points) for a forward+backward pair"
        ),
        "gpus_per_node": 8,
        "dpi": args.dpi,
        "outputs": [
            csv_path.name,
            "fig_tflops_vs_gpus_multinode.pdf",
            "fig_tflops_vs_gpus_multinode.png",
        ],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote TFLOP/s scaling analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
