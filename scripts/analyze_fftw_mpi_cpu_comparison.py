#!/usr/bin/env python3
"""Compare direct threaded FFTW-MPI CPU results with selected FFTM GPU results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


STRONG_SIZES = {3: 2048, 4: 320}
WEAK_SIZES_BY_NODES = {
    3: {1: 2048, 2: 2560, 4: 3200, 8: 4096, 12: 4704, 15: 5040},
    4: {1: 320, 2: 384, 4: 448, 8: 540, 12: 600, 15: 630},
}


@dataclass
class CpuMeasurement:
    dimension: int
    size: int
    nodes: int
    ranks: int
    ranks_per_node: str
    threads_per_rank: int
    allowed_cpus_per_rank: int
    planner: str
    planner_time_limit_seconds: float
    samples: List[float]
    forward_ms: float
    backward_ms: float
    plan_forward_ms: float
    plan_backward_ms: float
    relative_l2: float
    max_abs_error: float
    max_rank_alloc_mib: float
    total_alloc_gib: float
    cpu_model: str
    node_names: str
    fftw_version: str
    mpi_library: str

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples)

    @property
    def stddev_ms(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples)

    @property
    def minimum_ms(self) -> float:
        return min(self.samples)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.samples)
        index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
        return ordered[index]

    @property
    def variant(self) -> str:
        planner_limit = (
            "unlimited"
            if self.planner_time_limit_seconds < 0.0
            else f"{self.planner_time_limit_seconds:g}s"
        )
        return (
            f"rpn{self.ranks_per_node}-t{self.threads_per_rank}-"
            f"{self.planner}-plan{planner_limit}"
        )


@dataclass
class GpuMeasurement:
    dimension: int
    role: str
    size: int
    nodes: int
    gpus: int
    variant: str
    mean_ms: float
    stddev_ms: float
    median_ms: float
    p95_ms: float
    effective_tflops: float
    node_names: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-data-dir",
        type=Path,
        action="append",
        required=True,
        help="FFTW-MPI result directory; may be passed more than once.",
    )
    parser.add_argument(
        "--gpu-selected-csv",
        type=Path,
        action="append",
        required=True,
        help="hca_scaling_selected.csv from the FFTM scaling analysis.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--max-relative-l2", type=float, default=1.0e-11)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def isotropic_size(text: str, dimension: int) -> int:
    values = [int(value) for value in text.split("x")]
    if len(values) != dimension or len(set(values)) != 1:
        raise ValueError(f"expected an isotropic {dimension}D size, got {text}")
    return values[0]


def merge_text(values: Iterable[str]) -> str:
    result = set()
    for value in values:
        result.update(item for item in value.split(";") if item)
    return ";".join(sorted(result))


def read_cpu_measurements(roots: Sequence[Path]) -> List[CpuMeasurement]:
    grouped: Dict[Tuple[object, ...], Dict[str, object]] = {}
    summary_paths = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.resolve().rglob("benchmark_fftw_mpi_cpu.csv")
        }
    )
    if not summary_paths:
        raise RuntimeError("no benchmark_fftw_mpi_cpu.csv files found")

    for summary_path in summary_paths:
        iteration_path = summary_path.with_name("fftw_mpi_cpu_iterations.csv")
        samples_by_run: Dict[str, List[float]] = {}
        if iteration_path.is_file():
            for row in read_csv(iteration_path):
                samples_by_run.setdefault(row["run_id"], []).append(
                    float(row["pair_ms"])
                )

        for row in read_csv(summary_path):
            dimension = int(row["dimension"])
            size = isotropic_size(row["sizes"], dimension)
            threads_per_rank = int(row["threads_per_rank"])
            planner_time_limit_seconds = float(
                row.get("planner_time_limit_seconds", "-1")
            )
            allowed_cpus_per_rank = int(
                row.get("allowed_cpus_per_rank", threads_per_rank)
            )
            if allowed_cpus_per_rank != threads_per_rank:
                continue
            ranks_per_node = (
                row["min_ranks_per_node"]
                if row["min_ranks_per_node"] == row["max_ranks_per_node"]
                else f"{row['min_ranks_per_node']}:{row['max_ranks_per_node']}"
            )
            key = (
                dimension,
                size,
                int(row["num_nodes"]),
                int(row["num_ranks"]),
                ranks_per_node,
                threads_per_rank,
                row["planner"],
                planner_time_limit_seconds,
            )
            entry = grouped.setdefault(
                key,
                {
                    "samples": [],
                    "forward": [],
                    "backward": [],
                    "plan_forward": [],
                    "plan_backward": [],
                    "relative_l2": [],
                    "max_abs_error": [],
                    "max_rank_alloc_mib": [],
                    "total_alloc_gib": [],
                    "cpu_model": [],
                    "nodes": [],
                    "fftw_version": [],
                    "mpi_library": [],
                    "allowed_cpus_per_rank": [],
                },
            )
            run_samples = samples_by_run.get(row["run_id"])
            if not run_samples:
                run_samples = [float(row["avg_pair_ms"])]
            entry["samples"].extend(run_samples)
            entry["forward"].append(float(row["avg_forward_ms"]))
            entry["backward"].append(float(row["avg_backward_ms"]))
            entry["plan_forward"].append(float(row["plan_forward_ms"]))
            entry["plan_backward"].append(float(row["plan_backward_ms"]))
            entry["relative_l2"].append(float(row["relative_l2"]))
            entry["max_abs_error"].append(float(row["max_abs_error"]))
            entry["max_rank_alloc_mib"].append(float(row["max_rank_alloc_mib"]))
            entry["total_alloc_gib"].append(float(row["total_alloc_gib"]))
            entry["cpu_model"].append(row.get("cpu_model", ""))
            entry["nodes"].append(row.get("nodes", ""))
            entry["fftw_version"].append(row.get("fftw_version", ""))
            entry["mpi_library"].append(row.get("mpi_library", ""))
            entry["allowed_cpus_per_rank"].append(allowed_cpus_per_rank)

    result = []
    for key, entry in grouped.items():
        (
            dimension,
            size,
            nodes,
            ranks,
            ranks_per_node,
            threads,
            planner,
            planner_time_limit_seconds,
        ) = key
        result.append(
            CpuMeasurement(
                dimension=int(dimension),
                size=int(size),
                nodes=int(nodes),
                ranks=int(ranks),
                ranks_per_node=str(ranks_per_node),
                threads_per_rank=int(threads),
                allowed_cpus_per_rank=min(entry["allowed_cpus_per_rank"]),
                planner=str(planner),
                planner_time_limit_seconds=float(planner_time_limit_seconds),
                samples=list(entry["samples"]),
                forward_ms=statistics.mean(entry["forward"]),
                backward_ms=statistics.mean(entry["backward"]),
                plan_forward_ms=statistics.mean(entry["plan_forward"]),
                plan_backward_ms=statistics.mean(entry["plan_backward"]),
                relative_l2=max(entry["relative_l2"]),
                max_abs_error=max(entry["max_abs_error"]),
                max_rank_alloc_mib=max(entry["max_rank_alloc_mib"]),
                total_alloc_gib=max(entry["total_alloc_gib"]),
                cpu_model=merge_text(entry["cpu_model"]),
                node_names=merge_text(entry["nodes"]),
                fftw_version=merge_text(entry["fftw_version"]),
                mpi_library=merge_text(entry["mpi_library"]),
            )
        )
    if not result:
        raise RuntimeError(
            "no FFTW-MPI rows with valid CPU affinity were found"
        )
    return sorted(
        result,
        key=lambda item: (
            item.dimension,
            item.nodes,
            item.size,
            item.mean_ms,
        ),
    )


def roles_for(dimension: int, nodes: int, size: int) -> Tuple[str, ...]:
    roles = []
    if size == STRONG_SIZES[dimension]:
        roles.append("strong")
    if WEAK_SIZES_BY_NODES[dimension].get(nodes) == size:
        roles.append("weak")
    return tuple(roles)


def read_gpu_measurements(
    paths: Sequence[Path], gpus_per_node: int
) -> List[GpuMeasurement]:
    best: Dict[Tuple[int, str, int, int], GpuMeasurement] = {}
    for path in paths:
        for row in read_csv(path.resolve()):
            if row["library"].lower() != "fftm":
                continue
            gpus = int(row["num_gpus"])
            if gpus % gpus_per_node != 0:
                continue
            item = GpuMeasurement(
                dimension=int(row["dimension"]),
                role=row["scaling_role"],
                size=int(row["size"]),
                nodes=gpus // gpus_per_node,
                gpus=gpus,
                variant=row["variant"],
                mean_ms=float(row["mean_ms"]),
                stddev_ms=float(row["stddev_ms"]),
                median_ms=float(row["median_ms"]),
                p95_ms=float(row["p95_ms"]),
                effective_tflops=float(row["effective_tflops"]),
                node_names=row.get("nodes", ""),
            )
            key = (item.dimension, item.role, item.nodes, item.size)
            if key not in best or item.mean_ms < best[key].mean_ms:
                best[key] = item
    if not best:
        raise RuntimeError("no FFTM rows found in the selected GPU CSV input")
    return sorted(best.values(), key=lambda item: (item.dimension, item.role, item.nodes))


def effective_tflops(dimension: int, size: int, milliseconds: float) -> float:
    points = float(size) ** dimension
    pair_flops = 10.0 * points * math.log2(points)
    return pair_flops / (milliseconds / 1000.0) / 1.0e12


def select_cpu(
    measurements: Sequence[CpuMeasurement], max_relative_l2: float
) -> Dict[Tuple[int, str, int, int], CpuMeasurement]:
    selected: Dict[Tuple[int, str, int, int], CpuMeasurement] = {}
    for item in measurements:
        if item.relative_l2 > max_relative_l2:
            continue
        for role in roles_for(item.dimension, item.nodes, item.size):
            key = (item.dimension, role, item.nodes, item.size)
            if key not in selected or item.mean_ms < selected[key].mean_ms:
                selected[key] = item
    return selected


def write_cpu_candidates(
    path: Path,
    measurements: Sequence[CpuMeasurement],
    max_relative_l2: float,
) -> None:
    fields = [
        "dimension", "roles", "nodes", "size", "variant", "ranks",
        "ranks_per_node", "threads_per_rank", "allowed_cpus_per_rank",
        "planner", "planner_time_limit_seconds", "mean_ms",
        "stddev_ms", "median_ms", "min_ms", "p95_ms", "cv_pct", "samples",
        "forward_ms", "backward_ms", "plan_forward_ms", "plan_backward_ms",
        "effective_tflops", "relative_l2", "valid", "max_abs_error",
        "max_rank_alloc_mib", "total_alloc_gib", "fftw_version",
        "cpu_model", "nodes_used",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in measurements:
            writer.writerow(
                {
                    "dimension": item.dimension,
                    "roles": "+".join(roles_for(item.dimension, item.nodes, item.size)),
                    "nodes": item.nodes,
                    "size": item.size,
                    "variant": item.variant,
                    "ranks": item.ranks,
                    "ranks_per_node": item.ranks_per_node,
                    "threads_per_rank": item.threads_per_rank,
                    "allowed_cpus_per_rank": item.allowed_cpus_per_rank,
                    "planner": item.planner,
                    "planner_time_limit_seconds": f"{item.planner_time_limit_seconds:g}",
                    "mean_ms": f"{item.mean_ms:.6f}",
                    "stddev_ms": f"{item.stddev_ms:.6f}",
                    "median_ms": f"{item.median_ms:.6f}",
                    "min_ms": f"{item.minimum_ms:.6f}",
                    "p95_ms": f"{item.p95_ms:.6f}",
                    "cv_pct": f"{100.0 * item.stddev_ms / item.mean_ms:.6f}",
                    "samples": len(item.samples),
                    "forward_ms": f"{item.forward_ms:.6f}",
                    "backward_ms": f"{item.backward_ms:.6f}",
                    "plan_forward_ms": f"{item.plan_forward_ms:.6f}",
                    "plan_backward_ms": f"{item.plan_backward_ms:.6f}",
                    "effective_tflops": f"{effective_tflops(item.dimension, item.size, item.mean_ms):.6f}",
                    "relative_l2": f"{item.relative_l2:.9e}",
                    "valid": int(item.relative_l2 <= max_relative_l2),
                    "max_abs_error": f"{item.max_abs_error:.9e}",
                    "max_rank_alloc_mib": f"{item.max_rank_alloc_mib:.3f}",
                    "total_alloc_gib": f"{item.total_alloc_gib:.3f}",
                    "fftw_version": item.fftw_version,
                    "cpu_model": item.cpu_model,
                    "nodes_used": item.node_names,
                }
            )


def write_comparison(
    path: Path,
    cpu: Dict[Tuple[int, str, int, int], CpuMeasurement],
    gpu: Sequence[GpuMeasurement],
) -> List[Tuple[CpuMeasurement, GpuMeasurement]]:
    pairs = []
    fields = [
        "dimension", "scaling_role", "nodes", "size", "cpu_variant",
        "cpu_mean_ms", "cpu_stddev_ms", "cpu_tflops", "gpu_count",
        "gpu_variant", "gpu_mean_ms", "gpu_stddev_ms", "gpu_tflops",
        "gpu_speedup_over_cpu", "cpu_relative_l2", "cpu_nodes", "gpu_nodes",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for gpu_item in gpu:
            key = (
                gpu_item.dimension,
                gpu_item.role,
                gpu_item.nodes,
                gpu_item.size,
            )
            cpu_item = cpu.get(key)
            if cpu_item is None:
                continue
            pairs.append((cpu_item, gpu_item))
            writer.writerow(
                {
                    "dimension": gpu_item.dimension,
                    "scaling_role": gpu_item.role,
                    "nodes": gpu_item.nodes,
                    "size": gpu_item.size,
                    "cpu_variant": cpu_item.variant,
                    "cpu_mean_ms": f"{cpu_item.mean_ms:.6f}",
                    "cpu_stddev_ms": f"{cpu_item.stddev_ms:.6f}",
                    "cpu_tflops": f"{effective_tflops(cpu_item.dimension, cpu_item.size, cpu_item.mean_ms):.6f}",
                    "gpu_count": gpu_item.gpus,
                    "gpu_variant": gpu_item.variant,
                    "gpu_mean_ms": f"{gpu_item.mean_ms:.6f}",
                    "gpu_stddev_ms": f"{gpu_item.stddev_ms:.6f}",
                    "gpu_tflops": f"{gpu_item.effective_tflops:.6f}",
                    "gpu_speedup_over_cpu": f"{cpu_item.mean_ms / gpu_item.mean_ms:.6f}",
                    "cpu_relative_l2": f"{cpu_item.relative_l2:.9e}",
                    "cpu_nodes": cpu_item.node_names,
                    "gpu_nodes": gpu_item.node_names,
                }
            )
    return pairs


def write_cpu_selected(
    path: Path,
    selected: Dict[Tuple[int, str, int, int], CpuMeasurement],
) -> None:
    fields = [
        "dimension", "scaling_role", "nodes", "size", "variant", "ranks",
        "ranks_per_node", "threads_per_rank", "allowed_cpus_per_rank",
        "planner", "planner_time_limit_seconds", "mean_ms",
        "stddev_ms", "median_ms", "p95_ms", "samples", "effective_tflops",
        "relative_l2", "max_rank_alloc_mib", "total_alloc_gib", "nodes_used",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, item in sorted(selected.items()):
            writer.writerow(
                {
                    "dimension": key[0],
                    "scaling_role": key[1],
                    "nodes": key[2],
                    "size": key[3],
                    "variant": item.variant,
                    "ranks": item.ranks,
                    "ranks_per_node": item.ranks_per_node,
                    "threads_per_rank": item.threads_per_rank,
                    "allowed_cpus_per_rank": item.allowed_cpus_per_rank,
                    "planner": item.planner,
                    "planner_time_limit_seconds": f"{item.planner_time_limit_seconds:g}",
                    "mean_ms": f"{item.mean_ms:.6f}",
                    "stddev_ms": f"{item.stddev_ms:.6f}",
                    "median_ms": f"{item.median_ms:.6f}",
                    "p95_ms": f"{item.p95_ms:.6f}",
                    "samples": len(item.samples),
                    "effective_tflops": f"{effective_tflops(item.dimension, item.size, item.mean_ms):.6f}",
                    "relative_l2": f"{item.relative_l2:.9e}",
                    "max_rank_alloc_mib": f"{item.max_rank_alloc_mib:.3f}",
                    "total_alloc_gib": f"{item.total_alloc_gib:.3f}",
                    "nodes_used": item.node_names,
                }
            )


def plot_wall(
    output_dir: Path,
    pairs: Sequence[Tuple[CpuMeasurement, GpuMeasurement]],
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.5), constrained_layout=True)
    for row_index, dimension in enumerate((3, 4)):
        for column_index, role in enumerate(("strong", "weak")):
            axis = axes[row_index][column_index]
            rows = sorted(
                (
                    (cpu, gpu)
                    for cpu, gpu in pairs
                    if gpu.dimension == dimension and gpu.role == role
                ),
                key=lambda pair: pair[0].nodes,
            )
            if rows:
                nodes = [cpu.nodes for cpu, _ in rows]
                axis.errorbar(
                    nodes,
                    [cpu.mean_ms for cpu, _ in rows],
                    yerr=[cpu.stddev_ms for cpu, _ in rows],
                    marker="s",
                    color="#D55E00",
                    capsize=2.5,
                    label="FFTW-MPI CPU",
                )
                axis.errorbar(
                    nodes,
                    [gpu.mean_ms for _, gpu in rows],
                    yerr=[gpu.stddev_ms for _, gpu in rows],
                    marker="o",
                    color="#167D8D",
                    capsize=2.5,
                    label="FFTM GPU",
                )
                axis.set_xscale("log", base=2)
                axis.set_yscale("log")
                axis.set_xticks(nodes, [str(value) for value in nodes])
            axis.set_title(f"{dimension}D {role}")
            axis.set_xlabel("Nodes (8 GPUs/node for FFTM)")
            axis.set_ylabel("Forward + inverse (ms)")
            axis.grid(alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(handles, labels, frameon=False)
    fig.savefig(output_dir / "fig_fftw_cpu_vs_fftm_gpu_wall.pdf")
    fig.savefig(output_dir / "fig_fftw_cpu_vs_fftm_gpu_wall.png", dpi=300)
    plt.close(fig)


def plot_speedup(
    output_dir: Path,
    pairs: Sequence[Tuple[CpuMeasurement, GpuMeasurement]],
) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.3, 3.4), constrained_layout=True)
    styles = {
        (3, "strong"): ("#167D8D", "o", "3D strong"),
        (3, "weak"): ("#56B4E9", "s", "3D weak"),
        (4, "strong"): ("#6A5ACD", "^", "4D strong"),
        (4, "weak"): ("#CC79A7", "D", "4D weak"),
    }
    all_nodes = set()
    for key, (color, marker, label) in styles.items():
        rows = sorted(
            (
                (cpu, gpu)
                for cpu, gpu in pairs
                if (gpu.dimension, gpu.role) == key
            ),
            key=lambda pair: pair[0].nodes,
        )
        if not rows:
            continue
        nodes = [cpu.nodes for cpu, _ in rows]
        all_nodes.update(nodes)
        axis.plot(
            nodes,
            [cpu.mean_ms / gpu.mean_ms for cpu, gpu in rows],
            color=color,
            marker=marker,
            linewidth=1.6,
            label=label,
        )
    if all_nodes:
        ordered_nodes = sorted(all_nodes)
        axis.set_xscale("log", base=2)
        axis.set_xticks(ordered_nodes, [str(value) for value in ordered_nodes])
    axis.axhline(1.0, color="#555555", linewidth=1.0)
    axis.set_xlabel("Nodes (8 GPUs/node for FFTM)")
    axis.set_ylabel("FFTM speedup over FFTW-MPI CPU")
    axis.set_title("GPU acceleration at equal node count and global size")
    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, frameon=False, ncol=2)
    fig.savefig(output_dir / "fig_fftm_gpu_speedup_over_fftw_cpu.pdf")
    fig.savefig(output_dir / "fig_fftm_gpu_speedup_over_fftw_cpu.png", dpi=300)
    plt.close(fig)


def write_summary(
    path: Path,
    pairs: Sequence[Tuple[CpuMeasurement, GpuMeasurement]],
    failed_cases: int,
) -> None:
    lines = [
        "# Direct FFTW-MPI CPU Versus FFTM GPU",
        "",
        "## Expected Versus Obtained",
        "",
        "- Expected: FFTM should reduce forward-plus-inverse wall time relative to "
        "direct threaded FFTW-MPI at the same node count and global size.",
    ]
    if pairs:
        speedups = [cpu.mean_ms / gpu.mean_ms for cpu, gpu in pairs]
        lines.append(
            f"- Obtained: {len(pairs)} matched comparisons; FFTM speedup ranges "
            f"from {min(speedups):.2f}x to {max(speedups):.2f}x."
        )
    else:
        lines.append("- Obtained: no CPU/GPU rows matched by dimension, role, nodes, and size.")
    lines.extend(
        [
            f"- CPU launcher failures recorded: {failed_cases}. Invalid FFTW rows "
            "are retained as candidates but excluded from selection.",
            "- FFTW planning, allocation, initialization, and validation are excluded "
            "from the measured transform wall time.",
            "",
            "| Dim. | Role | Nodes | Size | CPU config | CPU ms | GPUs | GPU config | GPU ms | Speedup |",
            "|---:|:---|---:|---:|:---|---:|---:|:---|---:|---:|",
        ]
    )
    for cpu, gpu in sorted(
        pairs,
        key=lambda pair: (
            pair[1].dimension,
            pair[1].role,
            pair[1].nodes,
        ),
    ):
        lines.append(
            f"| {gpu.dimension}D | {gpu.role} | {gpu.nodes} | {gpu.size} | "
            f"{cpu.variant} | {cpu.mean_ms:.1f} | {gpu.gpus} | {gpu.variant} | "
            f"{gpu.mean_ms:.1f} | {cpu.mean_ms / gpu.mean_ms:.2f}x |"
        )
    path.write_text("\n".join(lines) + "\n")


def failed_count(roots: Sequence[Path]) -> int:
    count = 0
    for root in roots:
        for path in root.resolve().rglob("status.csv"):
            count += sum(
                row.get("status") == "failed"
                for row in read_csv(path)
            )
    return count


def main() -> int:
    args = parse_args()
    if args.gpus_per_node <= 0:
        raise SystemExit("--gpus-per-node must be positive")
    cpu_roots = [path.resolve() for path in args.cpu_data_dir]
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else cpu_roots[-1] / "analysis" / "cpu_vs_gpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_measurements = read_cpu_measurements(cpu_roots)
    gpu_measurements = read_gpu_measurements(
        args.gpu_selected_csv, args.gpus_per_node
    )
    selected_cpu = select_cpu(cpu_measurements, args.max_relative_l2)

    candidates_path = output_dir / "fftw_mpi_cpu_candidates.csv"
    selected_path = output_dir / "fftw_mpi_cpu_selected.csv"
    comparison_path = output_dir / "fftw_cpu_vs_fftm_gpu.csv"
    summary_path = output_dir / "summary.md"
    write_cpu_candidates(candidates_path, cpu_measurements, args.max_relative_l2)
    write_cpu_selected(selected_path, selected_cpu)
    pairs = write_comparison(comparison_path, selected_cpu, gpu_measurements)
    write_summary(summary_path, pairs, failed_count(cpu_roots))
    plot_wall(output_dir, pairs)
    plot_speedup(output_dir, pairs)

    manifest = {
        "cpu_sources": [str(path) for path in cpu_roots],
        "gpu_sources": [str(path.resolve()) for path in args.gpu_selected_csv],
        "selection": "minimum pooled mean pair wall time among numerically valid CPU configurations",
        "matching": "dimension, scaling role, node count, and global isotropic size",
        "gpu_mapping": f"{args.gpus_per_node} GPUs per node",
        "strong_sizes": STRONG_SIZES,
        "weak_sizes_by_nodes": WEAK_SIZES_BY_NODES,
        "effective_flops": "10 * total_points * log2(total_points) per forward+inverse pair",
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {len(cpu_measurements)} CPU candidates and {len(pairs)} "
        f"CPU/GPU comparisons to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
