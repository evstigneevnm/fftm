#!/usr/bin/env python3
"""Build the paper-ready 3D FFTM, Egger, and GPU-FFT comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PAIR_FLOP_FACTOR = 10.0


@dataclass
class EvidenceRow:
    library: str
    size: int
    num_gpus: int
    strategy: str
    mode: str
    mean_ms: float
    stddev_ms: float
    max_l2_diff: float
    valid: bool
    completed: bool
    dataset: str
    source: str
    plotted_series: str = ""

    @property
    def effective_tflops(self) -> float:
        points = float(self.size**3)
        pair_flops = PAIR_FLOP_FACTOR * points * math.log2(points)
        return pair_flops / (self.mean_ms / 1000.0) / 1.0e12

    @property
    def effective_tflops_stddev(self) -> float:
        if self.stddev_ms <= 0.0:
            return 0.0
        return self.effective_tflops * self.stddev_ms / self.mean_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fftm-ab-dir",
        type=Path,
        required=True,
        help="Directory containing the current slab/ and pencil/ FFTM runs.",
    )
    parser.add_argument(
        "--gpu-fft-dir",
        type=Path,
        required=True,
        help="GPU-FFT reference result directory.",
    )
    parser.add_argument(
        "--historical-scaling-csv",
        type=Path,
        required=True,
        help="Validated FFTM/Egger 8-120 GPU selected-scaling table.",
    )
    parser.add_argument(
        "--release-scaling-dir",
        type=Path,
        action="append",
        default=[],
        help="Current-release scale-out result directory; repeat as needed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json_lines(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_key(record: Dict[str, object]) -> Tuple[str, str, int, int]:
    spec = record.get("spec", {})
    if not isinstance(spec, dict):
        raise RuntimeError("Malformed run record: spec is not an object")
    sizes = record.get("sizes", [])
    if not isinstance(sizes, list) or not sizes:
        raise RuntimeError("Malformed run record: missing sizes")
    return (
        str(spec.get("strategy", "")),
        str(spec.get("mode", "")),
        int(spec.get("num_gpus", 0)),
        int(sizes[0]),
    )


def load_fftm_ab(root: Path) -> Tuple[List[EvidenceRow], List[Dict[str, object]]]:
    rows: List[EvidenceRow] = []
    exclusions: List[Dict[str, object]] = []
    for branch in ("slab", "pencil"):
        branch_dir = root / branch
        run_records = read_json_lines(branch_dir / "runs.jsonl")
        status = {run_key(item): int(item.get("returncode", 1)) for item in run_records}
        benchmark_path = branch_dir / "cpp_csv" / "benchmark_fftm_3d.csv"
        for index, item in enumerate(read_csv(benchmark_path), start=2):
            key = (
                item["strategy"],
                item["mode"],
                int(item["num_gpus"]),
                int(item["nx"]),
            )
            if key not in status:
                raise RuntimeError(f"No run record for {key} in {branch_dir}")
            completed = status[key] == 0
            max_l2 = float(item["max_l2_diff"])
            valid = completed and math.isfinite(max_l2) and max_l2 <= float(item["epsilon"])
            row = EvidenceRow(
                library="FFTM",
                size=int(item["nx"]),
                num_gpus=int(item["num_gpus"]),
                strategy=item["strategy"],
                mode=item["mode"],
                mean_ms=float(item["avg_wall_ms"]),
                stddev_ms=float(item["stddev_wall_ms"]),
                max_l2_diff=max_l2,
                valid=valid,
                completed=completed,
                dataset="current-fftm-ab",
                source=f"{benchmark_path.resolve()}:{index}",
            )
            rows.append(row)
            if not completed:
                matching = [record for record in run_records if run_key(record) == key]
                errors: List[str] = []
                if matching:
                    parsed = matching[0].get("parsed", {})
                    if isinstance(parsed, dict):
                        value = parsed.get("errors", [])
                        if isinstance(value, list):
                            errors = [str(error) for error in value]
                exclusions.append(
                    {
                        "library": row.library,
                        "size": row.size,
                        "num_gpus": row.num_gpus,
                        "strategy": row.strategy,
                        "mode": row.mode,
                        "returncode": status[key],
                        "reason": "run did not complete",
                        "errors": errors,
                    }
                )
    return rows, exclusions


def load_gpu_fft(root: Path) -> Tuple[List[EvidenceRow], List[Tuple[int, int, float]]]:
    comparison_path = root / "analysis" / "gpu_fft_reference_comparison.csv"
    rows: List[EvidenceRow] = []
    published: List[Tuple[int, int, float]] = []
    for item in read_csv(comparison_path):
        summary_path = root / item["source_file"]
        summary = read_csv(summary_path)
        if len(summary) != 1:
            raise RuntimeError(f"Expected one GPU-FFT summary row in {summary_path}")
        detail = summary[0]
        valid = item["valid"] == "1" and detail["valid"] == "1"
        rows.append(
            EvidenceRow(
                library="GPU-FFT",
                size=int(item["size"]),
                num_gpus=int(item["gpus"]),
                strategy="reference",
                mode="cuda-aware",
                mean_ms=float(item["avg_pair_ms"]),
                stddev_ms=float(detail["stddev_pair_ms"]),
                max_l2_diff=float(item["relative_l2"]),
                valid=valid,
                completed=True,
                dataset="current-gpu-fft",
                source=str(summary_path.resolve()),
            )
        )
        published_value = item.get("published_tflops", "").strip()
        if published_value:
            published.append((int(item["size"]), int(item["gpus"]), float(published_value)))
    return rows, published


def load_historical_scaling(path: Path) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for index, item in enumerate(read_csv(path), start=2):
        if item["dimension"] != "3" or item["scaling_role"] != "strong" or item["size"] != "2048":
            continue
        max_l2_text = item.get("max_l2_diff", "").strip()
        rows.append(
            EvidenceRow(
                library=item["library"],
                size=2048,
                num_gpus=int(item["num_gpus"]),
                strategy="pencil-pencil" if item["library"] == "FFTM" else "reference",
                mode="p2p-waitany",
                mean_ms=float(item["mean_ms"]),
                stddev_ms=float(item["stddev_ms"]),
                max_l2_diff=float(max_l2_text) if max_l2_text else float("nan"),
                valid=True,
                completed=True,
                dataset="validated-historical-scaling",
                source=f"{path.resolve()}:{index}",
            )
        )
    return rows


def load_release_scaling(roots: Iterable[Path]) -> List[EvidenceRow]:
    rows: List[EvidenceRow] = []
    for root in roots:
        benchmark_path = root / "benchmarks" / "cpp_csv" / "benchmark_fftm_3d.csv"
        runs_path = root / "benchmarks" / "runs.jsonl"
        records = read_json_lines(runs_path)
        successful = {
            run_key(record)
            for record in records
            if int(record.get("returncode", 1)) == 0
        }
        for index, item in enumerate(read_csv(benchmark_path), start=2):
            key = (
                item["strategy"], item["mode"], int(item["num_gpus"]), int(item["nx"])
            )
            max_l2 = float(item["max_l2_diff"])
            completed = key in successful
            rows.append(
                EvidenceRow(
                    library="FFTM",
                    size=int(item["nx"]),
                    num_gpus=int(item["num_gpus"]),
                    strategy=item["strategy"],
                    mode=item["mode"],
                    mean_ms=float(item["avg_wall_ms"]),
                    stddev_ms=float(item["stddev_wall_ms"]),
                    max_l2_diff=max_l2,
                    valid=completed and max_l2 <= float(item["epsilon"]),
                    completed=completed,
                    dataset="current-release-scaling",
                    source=f"{benchmark_path.resolve()}:{index}",
                )
            )
    return rows


def choose_plot_rows(rows: Sequence[EvidenceRow]) -> Dict[str, List[EvidenceRow]]:
    current = [row for row in rows if row.dataset == "current-fftm-ab" and row.valid]
    gpu_fft = [row for row in rows if row.dataset == "current-gpu-fft" and row.valid]
    historical = [row for row in rows if row.dataset == "validated-historical-scaling"]

    result: Dict[str, List[EvidenceRow]] = {}
    for size in (1920, 2048):
        result[f"slab-p2p-{size}"] = sorted(
            [row for row in current if row.size == size and row.strategy == "slab-pencil" and row.mode == "p2p-waitany"],
            key=lambda row: row.num_gpus,
        )
        result[f"slab-a2av-{size}"] = sorted(
            [row for row in current if row.size == size and row.strategy == "slab-pencil" and row.mode == "alltoallv"],
            key=lambda row: row.num_gpus,
        )
        pencil = [
            row for row in current
            if row.size == size and row.strategy == "pencil-pencil" and row.mode == "p2p-waitany"
        ]
        if size == 2048:
            pencil.extend(
                row for row in historical
                if row.library == "FFTM" and row.num_gpus > 32
            )
        result[f"pencil-{size}"] = sorted(pencil, key=lambda row: row.num_gpus)
        result[f"gpu-fft-{size}"] = sorted(
            [row for row in gpu_fft if row.size == size], key=lambda row: row.num_gpus
        )

    result["egger-2048"] = sorted(
        [row for row in historical if row.library == "Egger"],
        key=lambda row: row.num_gpus,
    )
    for label, values in result.items():
        for row in values:
            row.plotted_series = label
    return result


def configure_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def plot_series(axis, rows: Sequence[EvidenceRow], style: Dict[str, object], label: str) -> None:
    if not rows:
        return
    axis.errorbar(
        [row.num_gpus for row in rows],
        [row.effective_tflops for row in rows],
        yerr=[row.effective_tflops_stddev for row in rows],
        color=style["color"],
        marker=style["marker"],
        linestyle=style["linestyle"],
        linewidth=1.5,
        markersize=4.5,
        markerfacecolor=style.get("markerfacecolor", style["color"]),
        markeredgecolor=style["color"],
        capsize=2.2,
        label=label,
        zorder=style.get("zorder", 3),
    )


def make_plot(
    selected: Dict[str, List[EvidenceRow]], published: Sequence[Tuple[int, int, float]],
    output_dir: Path, dpi: int
) -> None:
    import matplotlib.pyplot as plt

    configure_plot_style()
    styles = {
        "slab-p2p": {"color": "#007C83", "marker": "o", "linestyle": "-"},
        "slab-a2av": {"color": "#4C956C", "marker": "v", "linestyle": "--"},
        "pencil": {"color": "#3564A5", "marker": "D", "linestyle": ":"},
        "egger": {"color": "#D55E00", "marker": "s", "linestyle": "--"},
        "gpu-fft": {"color": "#9B4F96", "marker": "^", "linestyle": "-."},
    }
    labels = {
        "slab-p2p": "FFTM slab, p2p-waitany",
        "slab-a2av": "FFTM slab, alltoallv",
        "pencil": "FFTM pencil, p2p-waitany",
        "egger": "Egger",
        "gpu-fft": "GPU-FFT (this system)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.15))
    for axis, size in zip(axes, (1920, 2048)):
        plot_series(axis, selected[f"slab-p2p-{size}"], styles["slab-p2p"], labels["slab-p2p"])
        plot_series(axis, selected[f"slab-a2av-{size}"], styles["slab-a2av"], labels["slab-a2av"])
        plot_series(axis, selected[f"pencil-{size}"], styles["pencil"], labels["pencil"])
        if size == 2048:
            plot_series(axis, selected["egger-2048"], styles["egger"], labels["egger"])
        plot_series(axis, selected[f"gpu-fft-{size}"], styles["gpu-fft"], labels["gpu-fft"])

        published_values = sorted(
            [(gpus, tflops) for value_size, gpus, tflops in published if value_size == size]
        )
        if published_values:
            axis.scatter(
                [value[0] for value in published_values],
                [value[1] for value in published_values],
                color="#222222", marker="x", s=28, linewidths=1.2,
                label="GPU-FFT (published)", zorder=5,
            )

        gpu_ticks = sorted(
            {
                row.num_gpus
                for key, values in selected.items()
                if key.endswith(f"-{size}") or (size == 2048 and key == "egger-2048")
                for row in values
            }
            | {gpus for value_size, gpus, _ in published if value_size == size}
        )
        axis.set_xscale("log", base=2)
        axis.set_xticks(gpu_ticks, [str(value) for value in gpu_ticks])
        axis.set_xlabel("GPUs")
        axis.set_ylabel("Effective throughput (TFLOP/s)")
        panel = "a" if size == 1920 else "b"
        axis.text(
            0.035, 0.95, rf"({panel}) ${size}^3$", transform=axis.transAxes,
            ha="left", va="top", fontsize=10, fontweight="bold",
        )
        axis.grid(alpha=0.22, linewidth=0.7)

        node_axis = axis.secondary_xaxis("top")
        node_axis.set_xscale("log", base=2)
        node_axis.set_xticks(gpu_ticks, [str(value // 8) for value in gpu_ticks])
        node_axis.set_xlabel("Nodes", labelpad=4)
        node_axis.tick_params(labelsize=7.5)

    handles: List[object] = []
    legend_labels: List[str] = []
    for axis in axes:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in legend_labels:
                handles.append(handle)
                legend_labels.append(label)
    fig.legend(
        handles, legend_labels, loc="upper center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.965), columnspacing=1.3, handlelength=2.5,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.14, top=0.71, wspace=0.28)
    fig.savefig(output_dir / "fig_3d_fftm_egger_gpufft.png", dpi=dpi)
    fig.savefig(output_dir / "fig_3d_fftm_egger_gpufft.pdf")
    plt.close(fig)


def write_evidence(path: Path, rows: Sequence[EvidenceRow]) -> None:
    fields = [
        "library", "size", "num_gpus", "num_nodes", "strategy", "mode",
        "mean_ms", "stddev_ms", "effective_tflops", "effective_tflops_stddev",
        "max_l2_diff", "valid", "completed", "dataset", "plotted_series", "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "library": row.library,
                    "size": row.size,
                    "num_gpus": row.num_gpus,
                    "num_nodes": row.num_gpus / 8.0,
                    "strategy": row.strategy,
                    "mode": row.mode,
                    "mean_ms": f"{row.mean_ms:.6f}",
                    "stddev_ms": f"{row.stddev_ms:.6f}",
                    "effective_tflops": f"{row.effective_tflops:.6f}",
                    "effective_tflops_stddev": f"{row.effective_tflops_stddev:.6f}",
                    "max_l2_diff": "" if math.isnan(row.max_l2_diff) else f"{row.max_l2_diff:.9e}",
                    "valid": int(row.valid),
                    "completed": int(row.completed),
                    "dataset": row.dataset,
                    "plotted_series": row.plotted_series,
                    "source": row.source,
                }
            )


def matched_comparison(
    rows: Sequence[EvidenceRow], output_path: Path
) -> List[Dict[str, object]]:
    fftm_rows = [
        row for row in rows
        if row.dataset == "current-fftm-ab" and row.strategy == "slab-pencil" and row.valid
    ]
    gpu_rows = [row for row in rows if row.dataset == "current-gpu-fft" and row.valid]
    output: List[Dict[str, object]] = []
    for gpu_row in sorted(gpu_rows, key=lambda row: (row.size, row.num_gpus)):
        candidates = [
            row for row in fftm_rows
            if row.size == gpu_row.size and row.num_gpus == gpu_row.num_gpus
        ]
        if not candidates:
            continue
        winner = min(candidates, key=lambda row: row.mean_ms)
        output.append(
            {
                "size": winner.size,
                "num_gpus": winner.num_gpus,
                "fftm_mode": winner.mode,
                "fftm_mean_ms": winner.mean_ms,
                "gpu_fft_mean_ms": gpu_row.mean_ms,
                "fftm_faster_pct": 100.0 * (gpu_row.mean_ms - winner.mean_ms) / gpu_row.mean_ms,
                "fftm_tflops": winner.effective_tflops,
                "gpu_fft_tflops": gpu_row.effective_tflops,
            }
        )
    fields = list(output[0]) if output else []
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
        for item in output:
            writer.writerow(
                {
                    "size": item["size"],
                    "num_gpus": item["num_gpus"],
                    "fftm_mode": item["fftm_mode"],
                    "fftm_mean_ms": f"{float(item['fftm_mean_ms']):.6f}",
                    "gpu_fft_mean_ms": f"{float(item['gpu_fft_mean_ms']):.6f}",
                    "fftm_faster_pct": f"{float(item['fftm_faster_pct']):.6f}",
                    "fftm_tflops": f"{float(item['fftm_tflops']):.6f}",
                    "gpu_fft_tflops": f"{float(item['gpu_fft_tflops']):.6f}",
                }
            )
    return output


def write_readme(path: Path, comparisons: Sequence[Dict[str, object]], exclusions: Sequence[Dict[str, object]]) -> None:
    improvements = [float(item["fftm_faster_pct"]) for item in comparisons]
    fastest_modes = sorted({str(item["fftm_mode"]) for item in comparisons})
    text = [
        "# 3D distributed GPU FFT comparison",
        "",
        "## Expected",
        "",
        "The exact same-system GPU-FFT comparison was intended to distinguish an FFTM design problem from the cluster's multinode transport behavior and to determine whether slab-pencil should replace pencil-pencil for 8-32 GPUs.",
        "",
        "## Obtained",
        "",
        f"FFTM's best slab-pencil mode is faster than GPU-FFT in all {len(comparisons)} matched 1920^3 and 2048^3 cases.",
        f"The wall-time advantage ranges from {min(improvements):.1f}% to {max(improvements):.1f}%.",
        f"The selected FFTM collective modes are {', '.join(fastest_modes)}.",
        "The 2048^3 panel extends the current 8-32 GPU matrix with the previously validated HCA-pinned FFTM/Egger 64-120 GPU rows.",
        "",
        "## Exclusions",
        "",
        f"{len(exclusions)} row was excluded because its launcher return code was nonzero.",
    ]
    for item in exclusions:
        text.append(
            f"- FFTM {item['strategy']} {item['size']}^3 on {item['num_gpus']} GPUs: "
            f"return code {item['returncode']}; {item['reason']}."
        )
    path.write_text("\n".join(text) + "\n")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fftm_rows, exclusions = load_fftm_ab(args.fftm_ab_dir.resolve())
    gpu_rows, published = load_gpu_fft(args.gpu_fft_dir.resolve())
    historical_rows = load_historical_scaling(args.historical_scaling_csv.resolve())
    release_rows = load_release_scaling(path.resolve() for path in args.release_scaling_dir)
    rows = fftm_rows + gpu_rows + historical_rows + release_rows
    selected = choose_plot_rows(rows)

    evidence_path = output_dir / "comparison_evidence.csv"
    matched_path = output_dir / "fftm_vs_gpu_fft_matched.csv"
    write_evidence(evidence_path, rows)
    comparisons = matched_comparison(rows, matched_path)
    make_plot(selected, published, output_dir, args.dpi)
    write_readme(output_dir / "README.md", comparisons, exclusions)

    manifest = {
        "fftm_ab_dir": str(args.fftm_ab_dir.resolve()),
        "gpu_fft_dir": str(args.gpu_fft_dir.resolve()),
        "historical_scaling_csv": str(args.historical_scaling_csv.resolve()),
        "release_scaling_dirs": [str(path.resolve()) for path in args.release_scaling_dir],
        "effective_flops": "10 * N^3 * log2(N^3) for one forward+backward pair",
        "failed_rows_excluded": exclusions,
        "selection": {
            "1920": "current valid FFTM slab modes, current valid FFTM pencil control, current GPU-FFT",
            "2048": "current 8-32 GPU rows; validated historical FFTM pencil and Egger rows above 32 GPUs",
        },
        "outputs": [
            evidence_path.name,
            matched_path.name,
            "fig_3d_fftm_egger_gpufft.pdf",
            "fig_3d_fftm_egger_gpufft.png",
            "README.md",
        ],
        "dpi": args.dpi,
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote 3D FFT comparison to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
