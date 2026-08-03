#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def collect(root):
    rows = []
    for path in root.glob("fftm/**/benchmark_fftm_3d.csv"):
        for source in read_rows(path):
            mean = finite(source.get("avg_wall_ms"))
            if mean is None:
                continue
            rows.append({
                "implementation": "FFTM",
                "dimension": 3,
                "size": f"{source['nx']}^3",
                "resources": f"{source['num_gpus']} GPU",
                "resource_count": int(source["num_gpus"]),
                "configuration": f"{source['strategy']}/{source['mode']}/host-staged",
                "mean_ms": mean,
                "stddev_ms": finite(source.get("stddev_wall_ms")) or 0.0,
                "error": finite(source.get("max_l2_diff")),
                "source": str(path),
            })
    for path in root.glob("fftm/**/benchmark_fftm_4d.csv"):
        for source in read_rows(path):
            mean = finite(source.get("avg_wall_ms"))
            if mean is None:
                continue
            rows.append({
                "implementation": "FFTM",
                "dimension": 4,
                "size": f"{source['nx']}^4",
                "resources": f"{source['num_gpus']} GPU",
                "resource_count": int(source["num_gpus"]),
                "configuration": f"{source['strategy']}/{source['mode']}/host-staged",
                "mean_ms": mean,
                "stddev_ms": finite(source.get("stddev_wall_ms")) or 0.0,
                "error": finite(source.get("max_l2_diff")),
                "source": str(path),
            })
    for path in root.glob("fftw/**/benchmark_fftw_mpi_cpu.csv"):
        for source in read_rows(path):
            mean = finite(source.get("avg_pair_ms"))
            if mean is None:
                continue
            ranks = int(source["num_ranks"])
            threads = int(source["threads_per_rank"])
            rows.append({
                "implementation": "FFTW",
                "dimension": int(source["dimension"]),
                "size": f"{source['sizes'].split('x')[0]}^{source['dimension']}",
                "resources": f"{ranks}x{threads} CPU",
                "resource_count": ranks,
                "configuration": f"MPI transposed/{source['planner']}",
                "mean_ms": mean,
                "stddev_ms": finite(source.get("stddev_pair_ms")) or 0.0,
                "error": finite(source.get("relative_l2")),
                "source": str(path),
            })
    return rows


def select(rows):
    selected = []
    for dimension in sorted({row["dimension"] for row in rows}):
        dim_rows = [row for row in rows if row["dimension"] == dimension]
        cpu_rows = [row for row in dim_rows if row["implementation"] == "FFTW"]
        if cpu_rows:
            selected.extend(sorted(cpu_rows, key=lambda row: row["resource_count"]))
        for gpus in sorted({row["resource_count"] for row in dim_rows if row["implementation"] == "FFTM"}):
            candidates = [
                row for row in dim_rows
                if row["implementation"] == "FFTM" and row["resource_count"] == gpus
            ]
            selected.append(min(candidates, key=lambda row: row["mean_ms"]))
    return selected


def write_csv(path, rows):
    fields = [
        "implementation", "dimension", "size", "resources", "configuration",
        "mean_ms", "stddev_ms", "error", "speedup_vs_best_fftw", "source",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def add_speedups(rows):
    for dimension in sorted({row["dimension"] for row in rows}):
        cpu = [
            row["mean_ms"] for row in rows
            if row["dimension"] == dimension and row["implementation"] == "FFTW"
        ]
        if not cpu:
            continue
        baseline = min(cpu)
        for row in rows:
            if row["dimension"] == dimension:
                row["speedup_vs_best_fftw"] = baseline / row["mean_ms"]


def write_markdown(path, rows):
    lines = [
        "| Transform | Implementation | Resources | Configuration | Pair time (ms) | Speedup vs best FFTW | Rel. L2 |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        error = row["error"]
        error_text = "" if error is None else f"{error:.2e}"
        lines.append(
            f"| {row['size']} | {row['implementation']} | {row['resources']} | "
            f"{row['configuration']} | {row['mean_ms']:.3f} +/- {row['stddev_ms']:.3f} | "
            f"{row.get('speedup_vs_best_fftw', float('nan')):.2f}x | {error_text} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot(path_prefix, rows):
    import matplotlib.pyplot as plt

    dimensions = sorted({row["dimension"] for row in rows})
    fig, axes = plt.subplots(1, len(dimensions), figsize=(6.3 * len(dimensions), 4.4), squeeze=False)
    colors = {"FFTW": "#4c78a8", "FFTM": "#e45756"}
    for axis, dimension in zip(axes[0], dimensions):
        subset = [row for row in rows if row["dimension"] == dimension]
        labels = [f"{row['implementation']}\n{row['resources']}" for row in subset]
        values = [row["mean_ms"] for row in subset]
        errors = [row["stddev_ms"] for row in subset]
        bars = axis.bar(
            range(len(subset)), values, yerr=errors,
            color=[colors[row["implementation"]] for row in subset], capsize=3,
        )
        axis.set_yscale("log")
        axis.set_xticks(range(len(subset)), labels, rotation=20, ha="right")
        axis.set_ylabel("Forward + inverse time (ms, log scale)")
        axis.set_title(f"{subset[0]['size']} ({dimension}D)")
        axis.grid(axis="y", which="both", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.1f}",
                ha="center", va="bottom", fontsize=8,
            )
    fig.suptitle("FFTM (HIP) versus FFTW-MPI on one dual-socket host")
    fig.tight_layout()
    fig.savefig(path_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path_prefix.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.data_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    rows = collect(args.data_dir)
    if not rows:
        raise SystemExit(f"No benchmark CSV rows found below {args.data_dir}")
    add_speedups(rows)
    chosen = select(rows)
    add_speedups(chosen)
    write_csv(output / "comparison_all.csv", rows)
    write_csv(output / "comparison_selected.csv", chosen)
    write_markdown(output / "comparison_table.md", chosen)
    plot(output / "fftm_vs_fftw", chosen)
    print(output)


if __name__ == "__main__":
    main()
