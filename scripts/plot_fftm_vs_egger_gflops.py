#!/usr/bin/env python3
"""Build FFTM/native vs fftm3d vs Egger effective-GFLOP/s figures.

The comparison uses the same conventional forward+inverse-pair operation count
as scripts/analyze_paper_results.py:

    10 * N * log2(N)

Raw Egger measurements store forward and inverse runs separately, so this script
pairs them by GPU count, grid, and opt value before computing GFLOP/s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-fftm")

import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Case:
    implementation: str
    gpu_count: int
    grid: str
    layout: str
    avg_wall_ms: float
    stddev_wall_ms: Optional[float]
    gflops: float
    source: str
    note: str = ""
    strategy: str = ""
    n: int = 0


def fopt(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def iopt(value: object) -> Optional[int]:
    v = fopt(value)
    return None if v is None else int(v)


def effective_gflops(nx: int, ny: int, nz: int, pair_ms: float) -> float:
    points = nx * ny * nz
    flops = 10.0 * float(points) * math.log2(float(points))
    return (flops / (pair_ms / 1000.0)) / 1.0e9


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def successful_timed_row(row: dict) -> bool:
    """True only for rows safe to use in paper-performance comparisons."""
    if str(row.get("returncode", "0")).strip() not in {"0", ""}:
        return False
    avg = fopt(row.get("avg_wall_ms"))
    return avg is not None and avg > 0.0


def parse_size_text(text: object) -> Optional[Tuple[int, int, int]]:
    if text is None:
        return None
    parts = str(text).strip().split("x")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def row_size(row: dict) -> Optional[Tuple[int, int, int]]:
    parsed = parse_size_text(row.get("sizes"))
    if parsed is not None:
        return parsed
    nx, ny, nz = iopt(row.get("nx")), iopt(row.get("ny")), iopt(row.get("nz"))
    if nx is None or ny is None or nz is None:
        return None
    return nx, ny, nz


def parse_gpu_size_map(text: str) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        gpu_s, sizes_s = item.split(":", 1)
        gpu = int(gpu_s.strip())
        for size_s in sizes_s.split(","):
            size_s = size_s.strip()
            if size_s:
                pairs.append((gpu, int(size_s)))
    return pairs


def load_fftm_native(data_dir: Path, nx: int, ny: int, nz: int) -> List[Case]:
    path = data_dir / "cpp_csv" / "benchmark_fftm_3d.csv"
    rows = read_csv(path)
    cases: List[Case] = []
    for row in rows:
        if row.get("strategy") != "pencil-pencil":
            continue
        if (iopt(row.get("nx")), iopt(row.get("ny")), iopt(row.get("nz"))) != (nx, ny, nz):
            continue
        avg = fopt(row.get("avg_wall_ms"))
        if not avg or avg <= 0:
            continue
        p1 = iopt(row.get("p1"))
        p2 = iopt(row.get("p2"))
        if p1 is None or p2 is None:
            continue
        cases.append(
            Case(
                implementation="FFTM native",
                gpu_count=iopt(row.get("num_gpus")) or p1 * p2,
                grid=f"{p1}x{p2}",
                layout=row.get("pencil_layout", ""),
                avg_wall_ms=avg,
                stddev_wall_ms=fopt(row.get("stddev_wall_ms")),
                gflops=effective_gflops(nx, ny, nz, avg),
                source=str(path),
                strategy=row.get("strategy", ""),
                n=nx,
            )
        )
    return cases


def load_fftm_flat_cases(data_dir: Path) -> List[Case]:
    """Load successful 3D FFTM/fftm3d rows from analyze_paper_results output.

    Failed/blank rows are intentionally skipped here. They are summarized
    separately by write_failure_summary().
    """
    flat_path = data_dir / "analysis" / "csv" / "measurements_flat.csv"
    rows = read_csv(flat_path)
    if not rows:
        return []

    cases: List[Case] = []
    for row in rows:
        if row.get("dim") != "3":
            continue
        if not successful_timed_row(row):
            continue
        size = row_size(row)
        if size is None or size[0] != size[1] or size[1] != size[2]:
            continue
        avg = fopt(row.get("avg_wall_ms"))
        if avg is None:
            continue
        gpu = iopt(row.get("num_gpus"))
        if gpu is None:
            continue
        grid = row.get("grid") or ""
        if not grid or grid == "configured":
            grid = ""
        backend = row.get("fftm_3d_backend") or ""
        if backend == "fftm3d-scfd-fft-facade":
            implementation = "fftm3d reference"
        else:
            implementation = "FFTM native"
        gflops = fopt(row.get("effective_gflops_s"))
        if gflops is None:
            gflops = effective_gflops(size[0], size[1], size[2], avg)
        cases.append(
            Case(
                implementation=implementation,
                gpu_count=gpu,
                grid=grid,
                layout=row.get("pencil_layout", ""),
                avg_wall_ms=avg,
                stddev_wall_ms=fopt(row.get("stddev_wall_ms")),
                gflops=gflops,
                source=str(flat_path),
                strategy=row.get("strategy", ""),
                n=size[0],
            )
        )
    return cases


def case_identity(case: Case) -> Tuple[str, int, int, str, str, str]:
    return (
        case.implementation,
        case.gpu_count,
        case.n,
        case.strategy,
        case.layout,
        case.grid,
    )


def merge_case_sources(primary: Sequence[Case], supplement: Sequence[Case]) -> List[Case]:
    """Keep authoritative benchmark rows, filling only configs missing there."""
    merged = list(primary)
    seen = {case_identity(case) for case in primary}
    for case in supplement:
        key = case_identity(case)
        if key in seen:
            continue
        merged.append(case)
        seen.add(key)
    return merged


def load_fftm_cpp_cases(data_dir: Path) -> List[Case]:
    """Load completed native FFTM 3D benchmark rows.

    The benchmark CSV is written only by runs that reached the benchmark binary
    summary path, so it is a safer source of successful timings than planned
    matrix rows with returncode/log metadata.
    """
    path = data_dir / "cpp_csv" / "benchmark_fftm_3d.csv"
    cases: List[Case] = []
    for row in read_csv(path):
        size = row_size(row)
        if size is None or size[0] != size[1] or size[1] != size[2]:
            continue
        avg = fopt(row.get("avg_wall_ms"))
        if avg is None or avg <= 0.0:
            continue
        gpu = iopt(row.get("num_gpus"))
        p1 = iopt(row.get("p1"))
        p2 = iopt(row.get("p2"))
        if gpu is None or p1 is None or p2 is None:
            continue
        cases.append(
            Case(
                implementation="FFTM native",
                gpu_count=gpu,
                grid=f"{p1}x{p2}",
                layout=row.get("pencil_layout", ""),
                avg_wall_ms=avg,
                stddev_wall_ms=fopt(row.get("stddev_wall_ms")),
                gflops=effective_gflops(size[0], size[1], size[2], avg),
                source=str(path),
                strategy=row.get("strategy", ""),
                n=size[0],
            )
        )
    return cases


def load_fftm3d_cpp_cases(data_dir: Path) -> List[Case]:
    path = data_dir / "cpp_csv" / "benchmark_fftm3d_3d.csv"
    cases: List[Case] = []
    for row in read_csv(path):
        size = row_size(row)
        if size is None or size[0] != size[1] or size[1] != size[2]:
            continue
        avg = fopt(row.get("avg_wall_ms"))
        if avg is None or avg <= 0.0:
            continue
        gpu = iopt(row.get("num_gpus"))
        p1 = iopt(row.get("p1"))
        p2 = iopt(row.get("p2"))
        if gpu is None or p1 is None or p2 is None:
            continue
        layout = row.get("opt", "")
        if layout and not layout.startswith("opt"):
            layout = f"opt{layout}"
        cases.append(
            Case(
                implementation="fftm3d reference",
                gpu_count=gpu,
                grid=f"{p1}x{p2}",
                layout=layout,
                avg_wall_ms=avg,
                stddev_wall_ms=fopt(row.get("stddev_wall_ms")),
                gflops=effective_gflops(size[0], size[1], size[2], avg),
                source=str(path),
                note=row.get("diagnostic_stage", ""),
                strategy="pencil-pencil",
                n=size[0],
            )
        )
    return cases


def load_paper_fftm_cases(data_dir: Path) -> List[Case]:
    """Load successful FFTM/fftm3d rows for paper comparisons.

    Prefer the benchmark CSVs because they contain actual successful timing
    summaries. Use measurements_flat.csv only to supplement configurations not
    present there, while still excluding failed or blank planned rows.
    """
    benchmark_cases = load_fftm_cpp_cases(data_dir) + load_fftm3d_cpp_cases(data_dir)
    flat_cases = load_fftm_flat_cases(data_dir)
    if benchmark_cases:
        return merge_case_sources(benchmark_cases, flat_cases)
    return flat_cases


def load_fftm3d(data_dirs: Sequence[Path], nx: int, ny: int, nz: int) -> List[Case]:
    cases: List[Case] = []
    for data_dir in data_dirs:
        path = data_dir / "cpp_csv" / "benchmark_fftm3d_3d.csv"
        for row in read_csv(path):
            if (iopt(row.get("nx")), iopt(row.get("ny")), iopt(row.get("nz"))) != (nx, ny, nz):
                continue
            avg = fopt(row.get("avg_wall_ms"))
            if not avg or avg <= 0:
                continue
            p1 = iopt(row.get("p1"))
            p2 = iopt(row.get("p2"))
            if p1 is None or p2 is None:
                continue
            layout = row.get("opt", "")
            if layout and not layout.startswith("opt"):
                layout = f"opt{layout}"
            cases.append(
                Case(
                    implementation="fftm3d reference",
                    gpu_count=iopt(row.get("num_gpus")) or p1 * p2,
                    grid=f"{p1}x{p2}",
                    layout=layout,
                    avg_wall_ms=avg,
                    stddev_wall_ms=fopt(row.get("stddev_wall_ms")),
                    gflops=effective_gflops(nx, ny, nz, avg),
                    source=str(path),
                    note=row.get("diagnostic_stage", ""),
                )
            )
    return cases


def load_egger_pairs(path: Path, nx: int, ny: int, nz: int) -> List[Case]:
    return [
        case
        for case in load_egger_pair_cases(path)
        if case.strategy == "pencil" and case.n == nx
    ]


def load_egger_pair_cases(path: Path) -> List[Case]:
    rows = read_csv(path)
    grouped: Dict[Tuple[int, int, str, str, str, str, str, str, str, str], Dict[str, dict]] = {}
    for row in rows:
        if row.get("ok") != "1":
            continue
        if row.get("transport") != "cuda_aware":
            continue
        if row.get("comm") != "Peer2Peer" or row.get("send") != "Sync":
            continue
        avg = fopt(row.get("avg_wall_ms"))
        if not avg or avg <= 0:
            continue
        gpu = iopt(row.get("gpu_count"))
        nx = iopt(row.get("nx"))
        ny = iopt(row.get("ny"))
        nz = iopt(row.get("nz"))
        p1 = iopt(row.get("p1"))
        p2 = iopt(row.get("p2"))
        opt = row.get("opt", "")
        case_name = row.get("case_name", "")
        if (
            gpu is None
            or nx is None
            or ny is None
            or nz is None
            or nx != ny
            or ny != nz
            or p1 is None
            or p2 is None
            or case_name not in {"forward", "inverse"}
        ):
            continue
        key = (
            gpu,
            nx,
            row.get("family", ""),
            row.get("variant", ""),
            row.get("transport", ""),
            row.get("comm", ""),
            row.get("send", ""),
            row.get("sequence", ""),
            f"opt{opt}",
            f"{p1}x{p2}",
        )
        grouped.setdefault(key, {})[case_name] = row

    cases: List[Case] = []
    for (gpu, nx, family, variant, _transport, _comm, _send, _sequence, layout, grid), pair in grouped.items():
        if "forward" not in pair or "inverse" not in pair:
            continue
        fwd = fopt(pair["forward"].get("avg_wall_ms"))
        inv = fopt(pair["inverse"].get("avg_wall_ms"))
        if fwd is None or inv is None:
            continue
        pair_ms = fwd + inv
        cases.append(
            Case(
                implementation="Egger raw pair",
                gpu_count=gpu,
                grid=grid,
                layout=layout,
                avg_wall_ms=pair_ms,
                stddev_wall_ms=None,
                gflops=effective_gflops(nx, nx, nx, pair_ms),
                source=str(path),
                note=f"forward={fwd:.6g} ms; inverse={inv:.6g} ms",
                strategy=family,
                n=nx,
            )
        )
    return cases


def best_by_gpu(cases: Iterable[Case]) -> List[Case]:
    best: Dict[int, Case] = {}
    for case in cases:
        old = best.get(case.gpu_count)
        if old is None or case.gflops > old.gflops:
            best[case.gpu_count] = case
    return [best[g] for g in sorted(best)]


def best_by_exact_key(cases: Iterable[Case]) -> Dict[Tuple[int, str, str], Case]:
    best: Dict[Tuple[int, str, str], Case] = {}
    for case in cases:
        if case.layout not in {"opt0", "opt1"}:
            continue
        key = (case.gpu_count, case.grid, case.layout)
        old = best.get(key)
        if old is None or case.gflops > old.gflops:
            best[key] = case
    return best


def write_best_csv(path: Path, rows: Sequence[Case]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "implementation",
                "num_gpus",
                "grid",
                "layout",
                "avg_wall_ms",
                "stddev_wall_ms",
                "effective_gflops_s",
                "source",
                "note",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.implementation,
                    row.gpu_count,
                    row.grid,
                    row.layout,
                    f"{row.avg_wall_ms:.9g}",
                    "" if row.stddev_wall_ms is None else f"{row.stddev_wall_ms:.9g}",
                    f"{row.gflops:.9g}",
                    row.source,
                    row.note,
                ]
            )


def write_exact_csv(path: Path, implementations: Sequence[Tuple[str, Dict[Tuple[int, str, str], Case]]]) -> List[Tuple[Tuple[int, str, str], Dict[str, Case]]]:
    keys = sorted({key for _, mapping in implementations for key in mapping})
    records: List[Tuple[Tuple[int, str, str], Dict[str, Case]]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["num_gpus", "grid", "layout"]
        for name, _ in implementations:
            prefix = name.lower().replace(" ", "_")
            header += [f"{prefix}_ms", f"{prefix}_gflops", f"{prefix}_source"]
        writer.writerow(header)
        for key in keys:
            row = [key[0], key[1], key[2]]
            found: Dict[str, Case] = {}
            for name, mapping in implementations:
                case = mapping.get(key)
                if case:
                    found[name] = case
                    row += [f"{case.avg_wall_ms:.9g}", f"{case.gflops:.9g}", case.source]
                else:
                    row += ["", "", ""]
            writer.writerow(row)
            records.append((key, found))
    return records


def plot_best(path_base: Path, series: Sequence[Tuple[str, Sequence[Case]]], metric: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    colors = {
        "FFTM native": "#1f77b4",
        "fftm3d reference": "#2ca02c",
        "Egger raw pair": "#d62728",
    }
    markers = {
        "FFTM native": "o",
        "fftm3d reference": "s",
        "Egger raw pair": "^",
    }
    for name, cases in series:
        xs = [c.gpu_count for c in cases]
        ys = [c.gflops if metric == "gflops" else c.avg_wall_ms for c in cases]
        ax.plot(xs, ys, marker=markers.get(name, "o"), linewidth=2.0, markersize=6, label=name, color=colors.get(name))
    ax.set_xlabel("GPU count")
    if metric == "gflops":
        ax.set_ylabel("Effective GFLOP/s")
        ax.set_title("2048^3 pencil-pencil effective GFLOP/s")
    else:
        ax.set_ylabel("Forward+inverse pair time [ms]")
        ax.set_title("2048^3 pencil-pencil pair time")
        ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_exact(path_base: Path, records: Sequence[Tuple[Tuple[int, str, str], Dict[str, Case]]]) -> None:
    complete = [(key, found) for key, found in records if {"FFTM native", "fftm3d reference", "Egger raw pair"}.issubset(found)]
    if not complete:
        complete = [(key, found) for key, found in records if len(found) >= 2]
    labels = [f"{key[0]}G\n{key[1]} {key[2]}" for key, _ in complete]
    impls = ["FFTM native", "fftm3d reference", "Egger raw pair"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(7.0, 0.85 * len(labels)), 4.2), constrained_layout=True)
    xs = list(range(len(labels)))
    for offset, (impl, color) in enumerate(zip(impls, colors)):
        vals = [found[impl].gflops if impl in found else math.nan for _, found in complete]
        xpos = [x + (offset - 1) * width for x in xs]
        ax.bar(xpos, vals, width=width, label=impl, color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Effective GFLOP/s")
    ax.set_title("Exact-grid 2048^3 pencil-pencil comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def best_case(cases: Iterable[Case]) -> Optional[Case]:
    valid = [case for case in cases if case.avg_wall_ms > 0.0 and math.isfinite(case.avg_wall_ms)]
    if not valid:
        return None
    return min(valid, key=lambda case: case.avg_wall_ms)


def case_cells(prefix: str, case: Optional[Case]) -> Dict[str, str]:
    if case is None:
        return {
            f"{prefix}_ms": "",
            f"{prefix}_gflops": "",
            f"{prefix}_config": "",
            f"{prefix}_source": "",
            f"{prefix}_note": "",
        }
    return {
        f"{prefix}_ms": f"{case.avg_wall_ms:.9g}",
        f"{prefix}_gflops": f"{case.gflops:.9g}",
        f"{prefix}_config": " ".join(
            part
            for part in [case.strategy, case.layout, case.grid, case.implementation]
            if part and part != "configured"
        ),
        f"{prefix}_source": case.source,
        f"{prefix}_note": case.note,
    }


def paper_comparison_rows(
    *,
    native_cases: Sequence[Case],
    fftm3d_cases: Sequence[Case],
    egger_cases: Sequence[Case],
    gpu_sizes: Sequence[Tuple[int, int]],
) -> List[dict]:
    rows: List[dict] = []
    for gpu, n in gpu_sizes:
        egger_overall = best_case(c for c in egger_cases if c.gpu_count == gpu and c.n == n)
        egger_pencil = best_case(c for c in egger_cases if c.gpu_count == gpu and c.n == n and c.strategy == "pencil")
        egger_grid = egger_pencil.grid if egger_pencil is not None else ""
        fftm_overall = best_case(c for c in native_cases if c.gpu_count == gpu and c.n == n)
        fftm_pencil = best_case(c for c in native_cases if c.gpu_count == gpu and c.n == n and c.strategy == "pencil-pencil")
        fftm_pencil_exact = best_case(
            c
            for c in native_cases
            if c.gpu_count == gpu and c.n == n and c.strategy == "pencil-pencil" and c.grid == egger_grid
        )
        fftm3d_ref = best_case(c for c in fftm3d_cases if c.gpu_count == gpu and c.n == n)

        row: Dict[str, str] = {
            "num_gpus": str(gpu),
            "n": str(n),
            "egger_pencil_grid": egger_grid,
        }
        row.update(case_cells("egger_overall", egger_overall))
        row.update(case_cells("egger_pencil", egger_pencil))
        row.update(case_cells("fftm_overall", fftm_overall))
        row.update(case_cells("fftm_pencil_best", fftm_pencil))
        row.update(case_cells("fftm_pencil_exact_grid", fftm_pencil_exact))
        row.update(case_cells("fftm3d_ref", fftm3d_ref))
        row["fftm_overall_speedup_vs_egger_overall"] = (
            f"{egger_overall.avg_wall_ms / fftm_overall.avg_wall_ms:.9g}" if egger_overall and fftm_overall else ""
        )
        row["fftm_pencil_speedup_vs_egger_pencil"] = (
            f"{egger_pencil.avg_wall_ms / fftm_pencil.avg_wall_ms:.9g}" if egger_pencil and fftm_pencil else ""
        )
        rows.append(row)
    return rows


def write_dict_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_native_best_by_size(path: Path, native_cases: Sequence[Case]) -> None:
    grouped: Dict[Tuple[int, int], List[Case]] = {}
    for case in native_cases:
        grouped.setdefault((case.gpu_count, case.n), []).append(case)
    rows: List[dict] = []
    for (gpu, n), cases in sorted(grouped.items()):
        overall = best_case(cases)
        pencil = best_case(c for c in cases if c.strategy == "pencil-pencil")
        row = {"num_gpus": str(gpu), "n": str(n)}
        row.update(case_cells("fftm_overall", overall))
        row.update(case_cells("fftm_pencil_best", pencil))
        rows.append(row)
    write_dict_csv(path, rows)


def classify_failure_log(text: str) -> str:
    low = text.lower()
    if "memory feasibility guard failed" in low:
        return "native opt0 memory feasibility guard"
    if "benchmark memory preflight failed" in low:
        return "benchmark public tensor memory preflight"
    if "forward_sendcounts exceeds mpi int range" in low or "same_z forward sendcounts exceeds mpi int range" in low:
        return "MPI int-count guard"
    if "forward output shape mismatch" in low:
        return "same_z output shape mismatch"
    if "out of memory" in low or "failed memory allocation" in low or "cudamalloc" in low:
        return "CUDA/scfd OOM"
    if "timeout after" in low or "timed out" in low:
        return "timeout"
    if "no such file" in low:
        return "missing file/path"
    if "error" in low:
        return "other error"
    return "unknown rc!=0"


def write_failure_summary(path: Path, data_dir: Path) -> None:
    runs_path = data_dir / "runs.jsonl"
    rows: List[dict] = []
    if not runs_path.exists():
        write_dict_csv(path, rows)
        return
    counts: Dict[Tuple[str, str], int] = {}
    for line in runs_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("returncode", 0)) == 0:
            continue
        spec = record.get("spec", {})
        raw_log = record.get("raw_log", "")
        log_path = data_dir / "raw" / Path(raw_log).name
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        reason = classify_failure_log(text)
        gpu = str(spec.get("num_gpus", ""))
        counts[(reason, gpu)] = counts.get((reason, gpu), 0) + 1
    for (reason, gpu), count in sorted(counts.items()):
        rows.append({"reason": reason, "num_gpus": gpu, "count": str(count)})
    write_dict_csv(path, rows)


def write_flat_row_exclusion_summary(path: Path, data_dir: Path) -> None:
    flat_path = data_dir / "analysis" / "csv" / "measurements_flat.csv"
    counts: Dict[str, int] = {}
    for row in read_csv(flat_path):
        if row.get("dim") != "3":
            reason = "non-3d"
        elif str(row.get("returncode", "0")).strip() not in {"0", ""}:
            reason = "failed-returncode"
        elif fopt(row.get("avg_wall_ms")) is None or (fopt(row.get("avg_wall_ms")) or 0.0) <= 0.0:
            reason = "blank-or-nonpositive-timing"
        else:
            size = row_size(row)
            if size is None:
                reason = "missing-size"
            elif size[0] != size[1] or size[1] != size[2]:
                reason = "non-cube-size"
            else:
                reason = "included-successful-timed-3d-cube"
        counts[reason] = counts.get(reason, 0) + 1
    rows = [{"reason": reason, "count": str(count)} for reason, count in sorted(counts.items())]
    write_dict_csv(path, rows)


def plot_paper_comparison(path_base: Path, rows: Sequence[dict], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    series = [
        ("Egger overall", "egger_overall_gflops", "o", "#222222"),
        ("FFTM overall", "fftm_overall_gflops", "s", "#1f77b4"),
        ("Egger pencil", "egger_pencil_gflops", "^", "#666666"),
        ("FFTM pencil", "fftm_pencil_best_gflops", "D", "#d62728"),
        ("fftm3d ref", "fftm3d_ref_gflops", "x", "#2ca02c"),
    ]
    for label, column, marker, color in series:
        xs: List[int] = []
        ys: List[float] = []
        for row in rows:
            value = fopt(row.get(column))
            gpu = iopt(row.get("num_gpus"))
            if value is not None and gpu is not None:
                xs.append(gpu)
                ys.append(value)
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, linewidth=1.8, markersize=6, label=label)
    ax.set_xlabel("GPU count")
    ax.set_ylabel("Effective GFLOP/s")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def plot_native_all_sizes(path_base: Path, native_cases: Sequence[Case]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    markers = {"slab-pencil": "o", "pencil-slab": "s", "pencil-pencil": "D"}
    for strategy in ["slab-pencil", "pencil-slab", "pencil-pencil"]:
        xs = [case.n for case in native_cases if case.strategy == strategy]
        ys = [case.gflops for case in native_cases if case.strategy == strategy]
        if xs:
            ax.scatter(xs, ys, label=strategy, alpha=0.8, s=28, marker=markers[strategy])
    ax.set_xlabel("N for N^3")
    ax.set_ylabel("Effective GFLOP/s")
    ax.set_title("Completed native FFTM rows")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(path_base.with_suffix(".pdf"))
    fig.savefig(path_base.with_suffix(".png"), dpi=180)
    plt.close(fig)


def run_paper_3d(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or (args.fftm_dir / "analysis" / "comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_cases = load_paper_fftm_cases(args.fftm_dir)
    if not all_cases:
        raise SystemExit(
            f"No successful timed rows found in {args.fftm_dir / 'cpp_csv'} or "
            f"{args.fftm_dir / 'analysis' / 'csv' / 'measurements_flat.csv'}."
        )
    native_cases = [case for case in all_cases if case.implementation == "FFTM native"]
    fftm3d_cases = [case for case in all_cases if case.implementation == "fftm3d reference"]
    egger_fitted = load_egger_pair_cases(args.egger_csv)
    egger_2048 = load_egger_pair_cases(args.egger_2048_csv) if args.egger_2048_csv else []

    fitted_sizes = parse_gpu_size_map(args.fitted_sizes)
    fitted_rows = paper_comparison_rows(
        native_cases=native_cases,
        fftm3d_cases=fftm3d_cases,
        egger_cases=egger_fitted,
        gpu_sizes=fitted_sizes,
    )
    write_dict_csv(output_dir / "fitted_size_fftm_vs_egger.csv", fitted_rows)
    plot_paper_comparison(
        output_dir / "fig_fitted_sizes_fftm_vs_egger_gflops",
        fitted_rows,
        "Fitted-size 3D FFT performance",
    )

    if egger_2048:
        n2048_sizes = parse_gpu_size_map(args.n2048_sizes)
        n2048_rows = paper_comparison_rows(
            native_cases=native_cases,
            fftm3d_cases=fftm3d_cases,
            egger_cases=egger_2048,
            gpu_sizes=n2048_sizes,
        )
        write_dict_csv(output_dir / "n2048_fftm_vs_egger.csv", n2048_rows)
        plot_paper_comparison(
            output_dir / "fig_n2048_fftm_vs_egger_gflops",
            n2048_rows,
            "2048^3 3D FFT performance",
        )

    write_native_best_by_size(output_dir / "all_sizes_fftm_native_best.csv", native_cases)
    write_failure_summary(output_dir / "failure_summary.csv", args.fftm_dir)
    write_flat_row_exclusion_summary(output_dir / "flat_row_exclusion_summary.csv", args.fftm_dir)
    plot_native_all_sizes(output_dir / "fig_all_completed_native_gflops_by_size", native_cases)

    print(f"Wrote paper 3D comparison outputs to {output_dir}")
    print(f"Successful native cases used: {len(native_cases)}")
    print(f"Successful fftm3d reference cases used: {len(fftm3d_cases)}")
    print(f"Paired fitted Egger cases loaded: {len(egger_fitted)}")
    if egger_2048:
        print(f"Paired 2048 Egger cases loaded: {len(egger_2048)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fftm-dir", type=Path, required=True)
    parser.add_argument("--fftm3d-dir", type=Path, action="append", default=[])
    parser.add_argument("--egger-csv", type=Path, required=True)
    parser.add_argument("--egger-2048-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--size", type=int, nargs=3, default=(2048, 2048, 2048), metavar=("NX", "NY", "NZ"))
    parser.add_argument(
        "--paper-3d",
        action="store_true",
        help=(
            "Generate full 3D paper comparison tables/figures from "
            "analysis/csv/measurements_flat.csv, excluding failed/blank rows."
        ),
    )
    parser.add_argument(
        "--fitted-sizes",
        default="2:1344;3:1536;4:1680;5:1800;6:1920;7:2025;8:2100",
        help="GPU-to-N mapping for fitted-size Egger comparisons.",
    )
    parser.add_argument(
        "--n2048-sizes",
        default="5:2048;6:2048;7:2048;8:2048",
        help="GPU-to-N mapping for 2048^3 Egger comparisons.",
    )
    args = parser.parse_args()

    if args.paper_3d:
        return run_paper_3d(args)

    nx, ny, nz = args.size
    output_dir = args.output_dir or (args.fftm_dir / "analysis")
    csv_dir = output_dir / "csv"
    fig_dir = output_dir / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fftm_cases = load_fftm_native(args.fftm_dir, nx, ny, nz)
    fftm3d_cases = load_fftm3d(args.fftm3d_dir, nx, ny, nz)
    egger_cases = load_egger_pairs(args.egger_csv, nx, ny, nz)

    best_rows: List[Case] = []
    best_series = [
        ("FFTM native", best_by_gpu(fftm_cases)),
        ("fftm3d reference", best_by_gpu(fftm3d_cases)),
        ("Egger raw pair", best_by_gpu(egger_cases)),
    ]
    for _, cases in best_series:
        best_rows.extend(cases)
    write_best_csv(csv_dir / "fftm_vs_egger_fftm3d_best_pencil_gflops.csv", best_rows)

    exact_records = write_exact_csv(
        csv_dir / "fftm_vs_egger_fftm3d_exact_grid_gflops.csv",
        [
            ("FFTM native", best_by_exact_key(fftm_cases)),
            ("fftm3d reference", best_by_exact_key(fftm3d_cases)),
            ("Egger raw pair", best_by_exact_key(egger_cases)),
        ],
    )

    plot_best(fig_dir / "fig_fftm_vs_egger_fftm3d_best_pencil_gflops", best_series, "gflops")
    plot_best(fig_dir / "fig_fftm_vs_egger_fftm3d_best_pencil_time", best_series, "time")
    plot_exact(fig_dir / "fig_fftm_vs_egger_fftm3d_exact_grid_gflops", exact_records)

    print(f"Wrote {csv_dir / 'fftm_vs_egger_fftm3d_best_pencil_gflops.csv'}")
    print(f"Wrote {csv_dir / 'fftm_vs_egger_fftm3d_exact_grid_gflops.csv'}")
    print(f"Wrote figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
