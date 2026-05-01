#!/usr/bin/env python3
"""Generate paper-oriented tables and figures from FFT benchmark stdout data.

The script consumes the JSONL output produced by run_local_paper_benchmarks.py.
It intentionally uses only the Python standard library. Tables are written as
LaTeX documents and compiled with pdflatex; figures are written directly as PDF
vector graphics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TRANSPORT_LABELS = {
    "single_gpu": "single GPU",
    "cuda_aware": "CUDA-aware",
    "non_cuda_aware": "non-CUDA-aware",
}

STRATEGY_ORDER = {
    3: ["slab-pencil", "pencil-slab", "pencil-pencil"],
    4: ["slab-slab", "pencil-pencil"],
}

MODE_ORDER = ["alltoallv", "alltoallw", "p2p-waitall", "p2p-waitany"]


@dataclass
class ResultRow:
    phase: str
    suite: str
    dim: int
    case_name: str
    transport: str
    strategy: str
    mode: str
    p2p_variant: str
    num_gpus: int
    returncode: int
    sizes: Tuple[int, ...]
    warmup: int
    avg_wall_ms: Optional[float]
    stddev_wall_ms: Optional[float]
    used_device_peak_max_mib: Optional[float]
    tracked_device_peak_max_mib: Optional[float]
    host_pinned_peak_max_mib: Optional[float]
    internal_device_peak_max_mib: Optional[float]
    external_device_peak_max_mib: Optional[float]
    max_l2: Optional[float]
    max_h1: Optional[float]
    raw_log: str
    profile_summary_entries: List[dict]
    profile_breakdown_entries: List[dict]
    memory_profile_entries: Dict[str, dict]
    memory_category_entries: Dict[str, dict]
    tracked_memory_entries: Dict[str, dict]

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def points(self) -> int:
        if not self.sizes:
            return 0
        return math.prod(self.sizes)

    @property
    def throughput_gpoints_s(self) -> Optional[float]:
        if not self.avg_wall_ms or self.avg_wall_ms <= 0 or not self.points:
            return None
        return (self.points / (self.avg_wall_ms / 1000.0)) / 1.0e9

    @property
    def effective_gflops_s(self) -> Optional[float]:
        """Conventional effective FFT rate for a measured forward+inverse pair.

        The benchmark times one forward transform plus one inverse transform.
        We report the common FFT estimate 5*N*log2(N) operations per transform,
        i.e. 10*N*log2(N) operations for the measured pair.
        """
        if not self.avg_wall_ms or self.avg_wall_ms <= 0 or self.points <= 1:
            return None
        flops = 10.0 * float(self.points) * math.log2(float(self.points))
        return (flops / (self.avg_wall_ms / 1000.0)) / 1.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and direct-PDF figures from FFT paper benchmark data."
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("build/paper_data/local_v100x2"),
        help="Directory containing runs.jsonl, hardware.json, and summary.json.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-directory>/analysis.",
    )
    parser.add_argument("--pdflatex", default="pdflatex", help="pdflatex executable.")
    parser.add_argument(
        "--compile-latex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile generated table LaTeX documents with pdflatex.",
    )
    parser.add_argument("--font-size", type=float, default=9.0, help="Base figure font size in pt.")
    parser.add_argument("--axis-font-size", type=float, default=8.0, help="Axis/tick font size in pt.")
    parser.add_argument("--legend-font-size", type=float, default=7.0, help="Legend font size in pt.")
    parser.add_argument("--title-font-size", type=float, default=10.0, help="Figure title font size in pt.")
    parser.add_argument("--table-font-size", type=float, default=8.0, help="Table font size in pt.")
    parser.add_argument("--figure-width-cm", type=float, default=8.4, help="Figure width in cm.")
    parser.add_argument("--figure-height-cm", type=float, default=5.2, help="Figure height in cm.")
    return parser.parse_args()


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_sizes(spec: dict, summary: dict, rec: dict) -> Tuple[int, ...]:
    sizes = rec.get("sizes") or []
    if not sizes:
        names = ["Nx", "Ny", "Nz", "Nw"]
        sizes = [summary.get(name) for name in names if summary.get(name) is not None]
    dim = int(spec.get("dim") or len(sizes) or 0)
    return tuple(int(v) for v in sizes[:dim] if v is not None)


def memory_entry(parsed: dict, category: str) -> Optional[float]:
    tracked = parsed.get("tracked_memory") or {}
    entries = tracked.get("entries") or {}
    entry = entries.get(category)
    if entry and entry.get("peak_max_mib") is not None:
        return as_float(entry.get("peak_max_mib"))
    categories = parsed.get("memory_categories") or {}
    entries = categories.get("entries") or {}
    entry = entries.get(category)
    if entry and entry.get("peak_max_mib") is not None:
        return as_float(entry.get("peak_max_mib"))
    return None


def max_numeric(summary: dict, names: Sequence[str]) -> Optional[float]:
    values = [as_float(summary.get(name)) for name in names if summary.get(name) is not None]
    values = [v for v in values if v is not None]
    if not values:
        return None
    return max(abs(v) for v in values)


PROFILE_RE = re.compile(
    r"^\[\s*(?P<name>.+):\s*(?P<ms>[0-9.eE+-]+)\s*ms\]\s*\(\s*(?P<pct>[0-9.eE+-]+)%\s*\)"
)

MEMORY_RE = re.compile(
    r"^\s*(?P<name>[^:]+):\s*current\(sum/max/avg\)=[^B]+B\s*"
    r"\((?P<current_sum_mib>[0-9.eE+-]+)/(?P<current_max_mib>[0-9.eE+-]+)/(?P<current_avg_mib>[0-9.eE+-]+)\s*MiB\),\s*"
    r"peak\(sum/max/avg\)=[^B]+B\s*"
    r"\((?P<peak_sum_mib>[0-9.eE+-]+)/(?P<peak_max_mib>[0-9.eE+-]+)/(?P<peak_avg_mib>[0-9.eE+-]+)\s*MiB\)"
)


def resolve_raw_log(data_dir: Path, raw_log: str) -> Optional[Path]:
    if not raw_log:
        return None
    path = Path(raw_log)
    if path.exists():
        return path
    candidate = data_dir / "raw" / path.name
    if candidate.exists():
        return candidate
    return None


def parse_profile_line(line: str) -> Optional[dict]:
    match = PROFILE_RE.match(line.strip())
    if not match:
        return None
    return {
        "name": match.group("name").strip(),
        "ms": float(match.group("ms")),
        "pct": float(match.group("pct")),
    }


def parse_memory_line(line: str) -> Optional[Tuple[str, dict]]:
    match = MEMORY_RE.match(line)
    if not match:
        return None
    name = match.group("name").strip()
    entry = {
        key: float(match.group(key))
        for key in [
            "current_sum_mib",
            "current_max_mib",
            "current_avg_mib",
            "peak_sum_mib",
            "peak_max_mib",
            "peak_avg_mib",
        ]
    }
    return name, entry


def parse_raw_profiler_sections(raw_path: Optional[Path]) -> Tuple[List[dict], List[dict], Dict[str, dict], Dict[str, dict]]:
    if raw_path is None or not raw_path.exists():
        return [], [], {}, {}
    breakdown: List[dict] = []
    summary: List[dict] = []
    memory_profile: Dict[str, dict] = {}
    memory_categories: Dict[str, dict] = {}
    section: Optional[str] = None
    with raw_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Profile breakdown:" in line:
                section = "profile_breakdown"
                continue
            if "Profile summarize:" in line:
                section = "profile_summary"
                continue
            if "Memory profile (MPI reduced):" in line or "Memory profile:" in line:
                section = "memory_profile"
                continue
            if "Memory profile categories" in line:
                section = "memory_categories"
                continue
            if "Memory profile totals" in line or "tracked memory" in line:
                if section in ("memory_profile", "memory_categories"):
                    section = None
                continue
            if section in ("profile_breakdown", "profile_summary"):
                parsed = parse_profile_line(line)
                if parsed:
                    if section == "profile_breakdown":
                        breakdown.append(parsed)
                    else:
                        summary.append(parsed)
                elif line.startswith("INFO:") and "Profile" not in line:
                    section = None
            elif section in ("memory_profile", "memory_categories"):
                parsed_mem = parse_memory_line(line)
                if parsed_mem:
                    name, entry = parsed_mem
                    if section == "memory_profile":
                        memory_profile[name] = entry
                    else:
                        memory_categories[name] = entry
    return summary, breakdown, memory_profile, memory_categories


def entries_from_parsed(parsed: dict, key: str) -> Dict[str, dict]:
    block = parsed.get(key) or {}
    entries = block.get("entries") or {}
    return dict(entries)


def parse_rows(data_dir: Path, phases: Sequence[str] = ("measure",)) -> List[ResultRow]:
    runs_path = data_dir / "runs.jsonl"
    if not runs_path.exists():
        raise FileNotFoundError(f"missing input file: {runs_path}")
    rows: List[ResultRow] = []
    with runs_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            phase = rec.get("phase") or "unknown"
            if phase not in phases:
                continue
            spec = rec.get("spec") or {}
            parsed = rec.get("parsed") or {}
            summary = parsed.get("summary") or {}
            sizes = extract_sizes(spec, summary, rec)
            strategy = summary.get("strategy") or spec.get("strategy") or "-"
            mode = summary.get("mode") or spec.get("mode") or "-"
            p2p_variant = spec.get("p2p_variant") or "configured"
            transport = spec.get("transport") or "unknown"
            suite = spec.get("suite") or "unknown"
            dim = int(spec.get("dim") or len(sizes) or 0)
            raw_log = str(rec.get("raw_log") or "")
            raw_summary, raw_breakdown, raw_memory_profile, raw_memory_categories = parse_raw_profiler_sections(
                resolve_raw_log(data_dir, raw_log)
            )
            profile_summary_entries = list(parsed.get("profile_summary") or []) or raw_summary
            profile_breakdown_entries = list(parsed.get("profile_breakdown") or []) or raw_breakdown
            memory_profile_entries = entries_from_parsed(parsed, "memory_profile") or raw_memory_profile
            memory_category_entries = entries_from_parsed(parsed, "memory_categories") or raw_memory_categories
            tracked_memory_entries = entries_from_parsed(parsed, "tracked_memory")
            host_pinned = (
                memory_entry(parsed, "host_pinned_internal")
                if memory_entry(parsed, "host_pinned_internal") is not None
                else memory_entry(parsed, "host_pinned")
            )
            external_device = (
                memory_entry(parsed, "external_test_owned")
                if memory_entry(parsed, "external_test_owned") is not None
                else memory_entry(parsed, "external_test")
            )
            rows.append(
                ResultRow(
                    phase=phase,
                    suite=suite,
                    dim=dim,
                    case_name=spec.get("case_name") or "unknown",
                    transport=transport,
                    strategy=strategy,
                    mode=mode,
                    p2p_variant=p2p_variant,
                    num_gpus=int(spec.get("num_gpus") or 0),
                    returncode=int(rec.get("returncode") if rec.get("returncode") is not None else -999),
                    sizes=sizes,
                    warmup=int(summary.get("warmup") if summary.get("warmup") is not None else rec.get("warmup") or 0),
                    avg_wall_ms=as_float(summary.get("avg_wall_ms")),
                    stddev_wall_ms=as_float(summary.get("stddev_wall_ms")),
                    used_device_peak_max_mib=as_float(rec.get("used_device_peak_max_mib")),
                    tracked_device_peak_max_mib=memory_entry(parsed, "tracked_device_total")
                    or memory_entry(parsed, "device"),
                    host_pinned_peak_max_mib=host_pinned,
                    internal_device_peak_max_mib=memory_entry(parsed, "internal_device")
                    or memory_entry(parsed, "device"),
                    external_device_peak_max_mib=external_device,
                    max_l2=max_numeric(
                        summary,
                        [
                            "L2",
                            "l2",
                            "rel_l2",
                            "max_l2",
                            "forward_rel_l2",
                            "backward_rel_l2",
                        ],
                    ),
                    max_h1=max_numeric(summary, ["H1", "h1", "max_h1"]),
                    raw_log=raw_log,
                    profile_summary_entries=profile_summary_entries,
                    profile_breakdown_entries=profile_breakdown_entries,
                    memory_profile_entries=memory_profile_entries,
                    memory_category_entries=memory_category_entries,
                    tracked_memory_entries=tracked_memory_entries,
                )
            )
    return rows


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "csv": out_dir / "csv",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def latex_escape(value) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def pt(value: float) -> str:
    return f"{value:.1f}pt"


def fmt_float(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def fmt_size(sizes: Sequence[int]) -> str:
    if not sizes:
        return "-"
    if len(set(sizes)) == 1:
        return f"{sizes[0]}^{len(sizes)}"
    return r"$\times$".join(str(v) for v in sizes)


def transport_label(transport: str) -> str:
    return TRANSPORT_LABELS.get(transport, transport.replace("_", "-"))


def strategy_sort_key(row: ResultRow) -> Tuple[int, str]:
    order = STRATEGY_ORDER.get(row.dim, [])
    try:
        pos = order.index(row.strategy)
    except ValueError:
        pos = len(order)
    return pos, row.strategy


def mode_sort_key(mode: str) -> Tuple[int, str]:
    try:
        return MODE_ORDER.index(mode), mode
    except ValueError:
        return len(MODE_ORDER), mode


def best_by_throughput(rows: Iterable[ResultRow]) -> Optional[ResultRow]:
    candidates = [r for r in rows if r.ok and r.throughput_gpoints_s is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.throughput_gpoints_s or -1.0)


def write_csv_outputs(rows: List[ResultRow], csv_dir: Path) -> List[Path]:
    paths: List[Path] = []
    flat_path = csv_dir / "measurements_flat.csv"
    with flat_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
                [
                    "phase",
                    "suite",
                "dim",
                "case_name",
                "transport",
                "strategy",
                "mode",
                "p2p_variant",
                "num_gpus",
                "returncode",
                "sizes",
                "warmup",
                "avg_wall_ms",
                "stddev_wall_ms",
                "throughput_gpoints_s",
                "effective_gflops_s",
                "used_device_peak_max_mib",
                "tracked_device_peak_max_mib",
                "host_pinned_peak_max_mib",
                "internal_device_peak_max_mib",
                "external_device_peak_max_mib",
                "max_l2",
                "max_h1",
                "raw_log",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.phase,
                    r.suite,
                    r.dim,
                    r.case_name,
                    r.transport,
                    r.strategy,
                    r.mode,
                    r.p2p_variant,
                    r.num_gpus,
                    r.returncode,
                    "x".join(str(v) for v in r.sizes),
                    r.warmup,
                    r.avg_wall_ms,
                    r.stddev_wall_ms,
                    r.throughput_gpoints_s,
                    r.effective_gflops_s,
                    r.used_device_peak_max_mib,
                    r.tracked_device_peak_max_mib,
                    r.host_pinned_peak_max_mib,
                    r.internal_device_peak_max_mib,
                    r.external_device_peak_max_mib,
                    r.max_l2,
                    r.max_h1,
                    r.raw_log,
                ]
            )
    paths.append(flat_path)
    return paths


def table_document(title: str, tabular: str, table_font_size: float) -> str:
    baseline = table_font_size * 1.18
    return f"""\\documentclass{{article}}
\\usepackage[margin=0.6in]{{geometry}}
\\usepackage{{array}}
\\pagestyle{{empty}}
\\begin{{document}}
\\begin{{center}}
{{\\fontsize{{{pt(table_font_size)}}}{{{pt(baseline)}}}\\selectfont
\\textbf{{{latex_escape(title)}}}

\\vspace{{0.6em}}
{tabular}
}}
\\end{{center}}
\\end{{document}}
"""


def compile_tex(tex_path: Path, pdflatex: str) -> Tuple[bool, str]:
    if shutil.which(pdflatex) is None:
        return False, f"{pdflatex!r} not found"
    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    proc = subprocess.run(
        cmd,
        cwd=str(tex_path.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    compile_log = tex_path.with_suffix(".compile.log")
    compile_log.write_text(proc.stdout, encoding="utf-8")
    for suffix in [".aux", ".log"]:
        aux = tex_path.with_suffix(suffix)
        if aux.exists():
            aux.unlink()
    if proc.returncode != 0:
        return False, f"pdflatex failed for {tex_path}; see {compile_log}"
    return True, ""


def write_table(path: Path, title: str, columns: str, header: Sequence[str], rows: Sequence[Sequence[str]], font_size: float) -> Path:
    body = ["\\begin{tabular}{" + columns + "}", "\\hline"]
    body.append(" & ".join(latex_escape(h) for h in header) + r" \\")
    body.append("\\hline")
    for row in rows:
        body.append(" & ".join(str(cell) for cell in row) + r" \\")
    body.append("\\hline")
    body.append("\\end{tabular}")
    path.write_text(table_document(title, "\n".join(body), font_size), encoding="utf-8")
    return path


def overview_table(data_dir: Path, rows: List[ResultRow], hardware: dict, summary: dict, table_dir: Path, font_size: float) -> Path:
    gpus = hardware.get("effective_gpus") or hardware.get("detected_gpus") or []
    gpu_names = sorted(set(str(g.get("name", "unknown")) for g in gpus))
    gpu_mem = sorted(set(str(g.get("memory_total_mib", "?")) for g in gpus))
    ok_count = sum(1 for r in rows if r.ok)
    fail_count = sum(1 for r in rows if not r.ok)
    benchmark_count = sum(1 for r in rows if r.case_name == "benchmark")
    table_rows = [
        ["Data directory", latex_escape(data_dir)],
        ["GPU count", str(len(gpus))],
        ["GPU model(s)", latex_escape(", ".join(gpu_names) if gpu_names else "unknown")],
        ["GPU memory [MiB]", latex_escape(", ".join(gpu_mem) if gpu_mem else "unknown")],
        ["Measurement records", str(len(rows))],
        ["Successful records", str(ok_count)],
        ["Failed records", str(fail_count)],
        ["Benchmark records", str(benchmark_count)],
        [latex_escape("Collector failed_measurement_runs"), str(summary.get("failed_measurement_runs", "-"))],
    ]
    return write_table(
        table_dir / "table_overview.tex",
        "Benchmark Data Overview",
        "ll",
        ["Quantity", "Value"],
        table_rows,
        font_size,
    )


def best_configs_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.ok]
    table_rows: List[List[str]] = []
    for dim in [3, 4]:
        baseline = best_by_throughput(r for r in benchmark if r.suite == "ffts" and r.dim == dim)
        groups = [
            ("FFTS", "single_gpu"),
            ("FFTM", "cuda_aware"),
            ("FFTM", "non_cuda_aware"),
        ]
        for suite_label, transport in groups:
            suite = suite_label.lower()
            best = best_by_throughput(
                r
                for r in benchmark
                if r.suite == suite and r.dim == dim and r.transport == transport
            )
            if not best:
                continue
            rel = None
            if baseline and baseline.throughput_gpoints_s:
                rel = (best.throughput_gpoints_s or 0.0) / baseline.throughput_gpoints_s
            table_rows.append(
                [
                    f"{dim}D",
                    latex_escape(f"{suite_label}, {best.num_gpus} GPU"),
                    latex_escape(transport_label(best.transport)),
                    latex_escape(best.strategy),
                    latex_escape(best.mode),
                    latex_escape(best.p2p_variant),
                    latex_escape(fmt_size(best.sizes)),
                    fmt_float(best.avg_wall_ms, 2),
                    fmt_float(best.throughput_gpoints_s, 3),
                    fmt_float(rel, 3),
                    fmt_int(best.used_device_peak_max_mib),
                ]
            )
    return write_table(
        table_dir / "table_best_configs.tex",
        "Best Benchmark Configurations",
        "lllllllrrrr",
        [
            "Dim",
            "Impl.",
            "Transport",
            "Strategy",
            "Mode",
            "Variant",
            "Size",
            "Time [ms]",
            "Gpts/s",
            "Rel. to FFTS",
            "GPU MiB",
        ],
        table_rows,
        font_size,
    )


def mode_transport_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    benchmark = [r for r in rows if r.case_name == "benchmark"]
    failures = defaultdict(list)
    for r in benchmark:
        if not r.ok:
            failures[(r.dim, r.transport, r.strategy)].append(f"{r.mode}/{r.p2p_variant}")
    table_rows: List[List[str]] = []
    for dim in [3, 4]:
        strategies = STRATEGY_ORDER.get(dim, sorted(set(r.strategy for r in benchmark if r.dim == dim)))
        for strategy in strategies:
            for transport in ["cuda_aware", "non_cuda_aware"]:
                group = [
                    r
                    for r in benchmark
                    if r.suite == "fftm"
                    and r.dim == dim
                    and r.strategy == strategy
                    and r.transport == transport
                    and r.ok
                ]
                if not group:
                    continue
                best = best_by_throughput(group)
                failed_modes = ", ".join(sorted(set(failures[(dim, transport, strategy)]), key=mode_sort_key))
                table_rows.append(
                    [
                        f"{dim}D",
                        latex_escape(strategy),
                        latex_escape(transport_label(transport)),
                        latex_escape(best.mode if best else "-"),
                        latex_escape(best.p2p_variant if best else "-"),
                        fmt_float(best.avg_wall_ms if best else None, 2),
                        fmt_float(best.throughput_gpoints_s if best else None, 3),
                        latex_escape(failed_modes or "-"),
                    ]
                )
    return write_table(
        table_dir / "table_mode_transport.tex",
        "Best FFTM Mode by Strategy and Transport",
        "lllllrrl",
        ["Dim", "Strategy", "Transport", "Best mode", "Best variant", "Time [ms]", "Gpts/s", "Failed modes"],
        table_rows,
        font_size,
    )


def correctness_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    versioned = [r for r in rows if r.case_name != "benchmark"]
    groups = defaultdict(list)
    for r in versioned:
        key = (r.suite, r.dim, r.transport)
        groups[key].append(r)
    table_rows: List[List[str]] = []
    for key in sorted(groups.keys(), key=lambda k: (k[1], k[0], k[2])):
        suite, dim, transport = key
        group = groups[key]
        cases = sorted(set(r.case_name for r in group))
        ok = sum(1 for r in group if r.ok)
        failed = sum(1 for r in group if not r.ok)
        max_l2_vals = [r.max_l2 for r in group if r.ok and r.max_l2 is not None]
        max_h1_vals = [r.max_h1 for r in group if r.ok and r.max_h1 is not None]
        table_rows.append(
            [
                f"{dim}D",
                latex_escape(suite.upper()),
                latex_escape(transport_label(transport)),
                latex_escape(",".join(cases)),
                str(ok),
                str(failed),
                fmt_float(max(max_l2_vals) if max_l2_vals else None, 2),
                fmt_float(max(max_h1_vals) if max_h1_vals else None, 2),
            ]
        )
    return write_table(
        table_dir / "table_correctness.tex",
        "Versioned Test Correctness Summary",
        "llllrrrr",
        ["Dim", "Impl.", "Transport", "Cases", "OK", "Fail", "max L2", "max H1"],
        table_rows,
        font_size,
    )


def failures_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    counter: Counter = Counter()
    case_sets: Dict[Tuple, set] = defaultdict(set)
    for r in rows:
        if r.ok:
            continue
        key = (r.suite, r.dim, r.transport, r.strategy, r.mode, r.p2p_variant)
        counter[key] += 1
        case_sets[key].add(r.case_name)
    if not counter:
        table_rows = [["-", "-", "-", "-", "-", "-", "0", "-"]]
    else:
        table_rows = []
        for key, count in sorted(counter.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2], kv[0][3], kv[0][4], kv[0][5])):
            suite, dim, transport, strategy, mode, p2p_variant = key
            table_rows.append(
                [
                    f"{dim}D",
                    latex_escape(suite.upper()),
                    latex_escape(transport_label(transport)),
                    latex_escape(strategy),
                    latex_escape(mode),
                    latex_escape(p2p_variant),
                    str(count),
                    latex_escape(",".join(sorted(case_sets[key]))),
                ]
            )
    return write_table(
        table_dir / "table_failures.tex",
        "Failed Measurement Summary",
        "llllllrl",
        ["Dim", "Impl.", "Transport", "Strategy", "Mode", "Variant", "Count", "Cases"],
        table_rows,
        font_size,
    )


def impl_label(row: ResultRow) -> str:
    if row.suite == "ffts":
        return f"FFTS/{row.num_gpus}GPU"
    suffix = "CA" if row.transport == "cuda_aware" else "NCA"
    return f"FFTM-{suffix}/{row.num_gpus}GPU"


def config_label(row: ResultRow) -> str:
    parts: List[str] = []
    if row.strategy and row.strategy != "-":
        parts.append(row.strategy)
    if row.mode and row.mode != "-":
        parts.append(row.mode)
    if row.p2p_variant and row.p2p_variant not in {"configured", "nca-staging"}:
        parts.append(row.p2p_variant)
    return ", ".join(parts) if parts else row.strategy


def entry_float(entry: dict, key: str) -> Optional[float]:
    value = entry.get(key)
    if value is None:
        return None
    return as_float(value)


def aggregate_profile_breakdown(entries: Sequence[dict]) -> List[dict]:
    sums: Dict[str, Tuple[float, float]] = {}
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        ms = as_float(entry.get("ms")) or 0.0
        pct_value = as_float(entry.get("pct")) or 0.0
        old_ms, old_pct = sums.get(name, (0.0, 0.0))
        sums[name] = (old_ms + ms, old_pct + pct_value)
    return [{"name": name, "ms": ms, "pct": pct} for name, (ms, pct) in sums.items()]


def temporal_entries_for_table(row: ResultRow, max_entries: int = 6) -> List[dict]:
    entries = row.profile_summary_entries or aggregate_profile_breakdown(row.profile_breakdown_entries)
    filtered = []
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name or name in {"self", "fftm_prof", "ffts_prof"}:
            continue
        if name.endswith("_prof"):
            continue
        filtered.append(entry)
    filtered.sort(key=lambda e: as_float(e.get("ms")) or 0.0, reverse=True)
    return filtered[:max_entries]


def profile_root_ms(row: ResultRow) -> Optional[float]:
    for entry in row.profile_breakdown_entries:
        name = str(entry.get("name", ""))
        if name.endswith("_prof"):
            value = as_float(entry.get("ms"))
            if value is not None and value > 0:
                return value
    values = [as_float(entry.get("ms")) for entry in row.profile_summary_entries]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def is_top_level_transform_entry(row: ResultRow, name: str) -> bool:
    if row.suite == "fftm":
        return bool(re.match(r"^fftm::(forward|backward)_[34]d_", name))
    if row.suite == "ffts":
        return name in {
            "ffts::forward_3d",
            "ffts::backward_3d",
            "ffts::forward_4d",
            "ffts::backward_4d",
        }
    return False


def is_top_level_communication_entry(name: str) -> bool:
    return name.startswith("mpi_transpose_") and "::transpose_" in name


def derived_temporal_entries(row: ResultRow) -> List[dict]:
    entries = row.profile_summary_entries or aggregate_profile_breakdown(row.profile_breakdown_entries)
    root_ms = profile_root_ms(row)
    transform_ms = 0.0
    communication_ms = 0.0
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        ms = as_float(entry.get("ms")) or 0.0
        if is_top_level_transform_entry(row, name):
            transform_ms += ms
        if is_top_level_communication_entry(name):
            communication_ms += ms
    if row.suite == "ffts":
        local_fft_ms = transform_ms
    else:
        local_fft_ms = max(0.0, transform_ms - communication_ms)
    result = []
    if local_fft_ms > 0.0:
        result.append(
            {
                "name": "derived: local FFT applications",
                "ms": local_fft_ms,
                "pct": (100.0 * local_fft_ms / root_ms) if root_ms else None,
            }
        )
    if row.suite == "fftm" and communication_ms > 0.0:
        result.append(
            {
                "name": "derived: total communication/transposes",
                "ms": communication_ms,
                "pct": (100.0 * communication_ms / root_ms) if root_ms else None,
            }
        )
    return result


def temporal_profile_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    table_rows: List[List[str]] = []
    for row in best_benchmark_rows(rows):
        entries = derived_temporal_entries(row) + temporal_entries_for_table(row, max_entries=6)
        if not entries:
            table_rows.append(
                [
                    f"{row.dim}D",
                    latex_escape(impl_label(row)),
                    latex_escape(config_label(row)),
                    latex_escape("no profile entries"),
                    "-",
                    "-",
                ]
            )
            continue
        for entry in entries:
            table_rows.append(
                [
                    f"{row.dim}D",
                    latex_escape(impl_label(row)),
                    latex_escape(config_label(row)),
                    latex_escape(str(entry.get("name", "-"))),
                    fmt_float(as_float(entry.get("ms")), 2),
                    fmt_float(as_float(entry.get("pct")), 2),
                ]
            )
    return write_table(
        table_dir / "table_temporal_profile_breakdown.tex",
        "Accumulated Temporal Profiler Breakdown for Best Benchmark Configurations",
        r"lll p{0.46\linewidth} rr",
        ["Dim", "Impl.", "Config", "Profiler entry", "Time [ms]", "Total [%]"],
        table_rows,
        font_size,
    )


def selected_memory_category_rows(row: ResultRow) -> List[Tuple[str, str, dict]]:
    entries = row.tracked_memory_entries or row.memory_category_entries
    default_scope = "tracked" if row.tracked_memory_entries else "category"
    preferred = [
        ("tracked", "tracked_device_total"),
        ("tracked", "internal_device"),
        ("tracked", "external_test_owned"),
        ("tracked", "host_pinned_internal"),
        ("category", "device"),
        ("category", "external_test"),
        ("category", "host_pinned"),
    ]
    result = []
    seen = set()
    for scope, name in preferred:
        entry = entries.get(name)
        if not entry:
            continue
        peak = entry_float(entry, "peak_max_mib")
        current = entry_float(entry, "current_max_mib")
        if (peak is None or peak == 0.0) and (current is None or current == 0.0):
            continue
        result.append((scope, name, entry))
        seen.add(name)
    for name, entry in sorted(entries.items()):
        if name in seen:
            continue
        peak = entry_float(entry, "peak_max_mib")
        current = entry_float(entry, "current_max_mib")
        if (peak is None or peak == 0.0) and (current is None or current == 0.0):
            continue
        result.append((default_scope, name, entry))
    return result


def selected_memory_component_rows(row: ResultRow, max_entries: int = 5) -> List[Tuple[str, str, dict]]:
    components = []
    for name, entry in row.memory_profile_entries.items():
        peak = entry_float(entry, "peak_max_mib")
        current = entry_float(entry, "current_max_mib")
        if (peak is None or peak == 0.0) and (current is None or current == 0.0):
            continue
        components.append(("component", name, entry))
    components.sort(key=lambda item: entry_float(item[2], "peak_max_mib") or 0.0, reverse=True)
    return components[:max_entries]


def memory_profile_table(rows: List[ResultRow], table_dir: Path, font_size: float) -> Path:
    table_rows: List[List[str]] = []
    for row in best_benchmark_rows(rows):
        entries = selected_memory_category_rows(row) + selected_memory_component_rows(row, max_entries=5)
        if not entries:
            table_rows.append(
                [
                    f"{row.dim}D",
                    latex_escape(impl_label(row)),
                    latex_escape(config_label(row)),
                    "-",
                    latex_escape("no memory entries"),
                    "-",
                    "-",
                    "-",
                ]
            )
            continue
        for scope, name, entry in entries:
            table_rows.append(
                [
                    f"{row.dim}D",
                    latex_escape(impl_label(row)),
                    latex_escape(config_label(row)),
                    latex_escape(scope),
                    latex_escape(name),
                    fmt_float(entry_float(entry, "peak_max_mib"), 1),
                    fmt_float(entry_float(entry, "current_max_mib"), 1),
                    fmt_float(entry_float(entry, "peak_sum_mib"), 1),
                ]
            )
    return write_table(
        table_dir / "table_memory_profile_breakdown.tex",
        "Memory Profiler Breakdown for Best Benchmark Configurations",
        r"llll p{0.35\linewidth} rrr",
        ["Dim", "Impl.", "Config", "Scope", "Component", "Peak max [MiB]", "Current max [MiB]", "Peak sum [MiB]"],
        table_rows,
        font_size,
    )


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def cm_to_pt(value: float) -> float:
    return value * 28.3464566929


class PdfCanvas:
    """Tiny single-page vector-PDF writer sufficient for paper plots."""

    def __init__(self, width_pt: float, height_pt: float):
        self.width = float(width_pt)
        self.height = float(height_pt)
        self.commands: List[str] = []

    def _cmd(self, command: str) -> None:
        self.commands.append(command)

    def fill_rgb(self, color: Tuple[float, float, float]) -> None:
        r, g, b = color
        self._cmd(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def stroke_rgb(self, color: Tuple[float, float, float]) -> None:
        r, g, b = color
        self._cmd(f"{r:.4f} {g:.4f} {b:.4f} RG")

    def line_width(self, width: float) -> None:
        self._cmd(f"{width:.3f} w")

    def line(self, x0: float, y0: float, x1: float, y1: float, color=(0, 0, 0), width: float = 0.6) -> None:
        self.stroke_rgb(color)
        self.line_width(width)
        self._cmd(f"{x0:.3f} {y0:.3f} m {x1:.3f} {y1:.3f} l S")

    def rect(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        fill: Optional[Tuple[float, float, float]] = None,
        stroke: Optional[Tuple[float, float, float]] = None,
        stroke_width: float = 0.4,
    ) -> None:
        if fill is not None:
            self.fill_rgb(fill)
            self._cmd(f"{x:.3f} {y:.3f} {width:.3f} {height:.3f} re f")
        if stroke is not None:
            self.stroke_rgb(stroke)
            self.line_width(stroke_width)
            self._cmd(f"{x:.3f} {y:.3f} {width:.3f} {height:.3f} re S")

    @staticmethod
    def text_width(text: str, size: float) -> float:
        return len(text) * size * 0.52

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: float,
        color=(0, 0, 0),
        align: str = "left",
    ) -> None:
        if align == "center":
            x -= self.text_width(text, size) / 2.0
        elif align == "right":
            x -= self.text_width(text, size)
        self.fill_rgb(color)
        self._cmd(f"BT /F1 {size:.3f} Tf {x:.3f} {y:.3f} Td ({pdf_escape(text)}) Tj ET")

    def multiline_text(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        size: float,
        color=(0, 0, 0),
        align: str = "left",
        leading: Optional[float] = None,
    ) -> None:
        leading = leading if leading is not None else size * 1.15
        for i, line in enumerate(lines):
            self.text(x, y - i * leading, line, size, color=color, align=align)

    def save(self, path: Path) -> None:
        content = ("\n".join(self.commands) + "\n").encode("latin-1", "replace")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.3f} {self.height:.3f}] "
                f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
        ]
        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out.extend(f"{index} 0 obj\n".encode("ascii"))
            out.extend(obj)
            out.extend(b"\nendobj\n")
        xref_pos = len(out)
        out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        out.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        out.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_pos}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(out)


def nice_axis_max(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10.0**exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10.0**exponent)


def figure_canvas(args: argparse.Namespace, min_width_cm: float = 8.4, min_height_cm: float = 5.2) -> PdfCanvas:
    width_cm = max(args.figure_width_cm, min_width_cm)
    height_cm = max(args.figure_height_cm, min_height_cm)
    return PdfCanvas(cm_to_pt(width_cm), cm_to_pt(height_cm))


def wrap_label(label: str) -> List[str]:
    parts = label.replace("non-CUDA-aware", "NCA").replace("CUDA-aware", "CA").split()
    if len(parts) > 1:
        return parts
    if "-" in label:
        return label.split("-")
    return [label]


def draw_legend(canvas: PdfCanvas, items: Sequence[Tuple[str, Tuple[float, float, float]]], x: float, y: float, font_size: float) -> None:
    cursor = x
    for label, color in items:
        canvas.rect(cursor, y - 1.0, 8.0, 6.0, fill=color, stroke=(0.25, 0.25, 0.25), stroke_width=0.2)
        canvas.text(cursor + 11.0, y, label, font_size)
        cursor += 12.0 + PdfCanvas.text_width(label, font_size) + 10.0


def wrap_words(text: str, max_chars: int) -> List[str]:
    if not text:
        return []
    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def legend_item_width(label: str, font_size: float) -> float:
    return 12.0 + PdfCanvas.text_width(label, font_size) + 10.0


def estimate_legend_rows(items: Sequence[Tuple[str, Tuple[float, float, float]]], max_width: float, font_size: float) -> int:
    rows = 1
    cursor = 0.0
    for label, _ in items:
        width = legend_item_width(label, font_size)
        if cursor > 0.0 and cursor + width > max_width:
            rows += 1
            cursor = width
        else:
            cursor += width
    return rows


def draw_legend_wrapped(
    canvas: PdfCanvas,
    items: Sequence[Tuple[str, Tuple[float, float, float]]],
    x: float,
    y: float,
    max_width: float,
    font_size: float,
    row_gap: float = 10.0,
) -> None:
    cursor = x
    row_y = y
    for label, color in items:
        width = legend_item_width(label, font_size)
        if cursor > x and cursor + width > x + max_width:
            cursor = x
            row_y -= row_gap
        canvas.rect(cursor, row_y - 1.0, 8.0, 6.0, fill=color, stroke=(0.25, 0.25, 0.25), stroke_width=0.2)
        canvas.text(cursor + 11.0, row_y, label, font_size)
        cursor += width


def bar_chart_pdf(
    path: Path,
    title: str,
    categories: Sequence[str],
    series: Sequence[Tuple[str, Sequence[Optional[float]], Tuple[float, float, float]]],
    y_label: str,
    args: argparse.Namespace,
    value_digits: int = 2,
    reference_y: Optional[float] = None,
    note: Optional[str] = None,
) -> Path:
    width_pt = max(cm_to_pt(max(args.figure_width_cm, max(8.4, 1.4 * len(categories) + 3.0))), 238.0)
    note_lines = wrap_words(note or "", max(48, int(width_pt / max(args.legend_font_size * 0.52, 1.0)) - 4))[:3]
    legend_items = [(name, color) for name, _, color in series]
    legend_rows = estimate_legend_rows(legend_items, max(width_pt - 60.0, 80.0), args.legend_font_size)
    margin_left, margin_right, margin_bottom = 48.0, 14.0, 58.0
    margin_top = 40.0 + 9.0 * len(note_lines) + 11.0 * legend_rows
    height_pt = max(cm_to_pt(args.figure_height_cm), margin_top + margin_bottom + 105.0)
    canvas = PdfCanvas(width_pt, height_pt)
    plot_x = margin_left
    plot_y = margin_bottom
    plot_w = canvas.width - margin_left - margin_right
    plot_h = canvas.height - margin_bottom - margin_top
    values = [v for _, vals, _ in series for v in vals if v is not None]
    if reference_y is not None:
        values.append(reference_y)
    y_max = nice_axis_max(max(values or [1.0]) * 1.12)

    canvas.text(canvas.width / 2.0, canvas.height - 17.0, title, args.title_font_size, align="center")
    for index, note_line in enumerate(note_lines):
        canvas.text(
            canvas.width / 2.0,
            canvas.height - 31.0 - index * 8.0,
            note_line,
            args.legend_font_size,
            color=(0.25, 0.25, 0.25),
            align="center",
        )
    canvas.line(plot_x, plot_y, plot_x + plot_w, plot_y, color=(0, 0, 0), width=0.7)
    canvas.line(plot_x, plot_y, plot_x, plot_y + plot_h, color=(0, 0, 0), width=0.7)
    for i in range(5):
        value = y_max * i / 4.0
        y = plot_y + plot_h * value / y_max
        canvas.line(plot_x, y, plot_x + plot_w, y, color=(0.86, 0.86, 0.86), width=0.3)
        canvas.text(plot_x - 5.0, y - 3.0, axis_tick_label(value), args.axis_font_size, color=(0.25, 0.25, 0.25), align="right")
    if reference_y is not None:
        y = plot_y + plot_h * reference_y / y_max
        canvas.line(plot_x, y, plot_x + plot_w, y, color=(0.2, 0.2, 0.2), width=0.8)
        canvas.text(plot_x + plot_w - 2.0, y + 3.0, "1.0", args.legend_font_size, color=(0.2, 0.2, 0.2), align="right")

    ncat = max(1, len(categories))
    nseries = max(1, len(series))
    group_w = plot_w / ncat
    bar_w = min(14.0, group_w * 0.68 / nseries)
    for ci, category in enumerate(categories):
        center = plot_x + group_w * (ci + 0.5)
        start = center - (nseries * bar_w + (nseries - 1) * 2.0) / 2.0
        for si, (_, vals, color) in enumerate(series):
            value = vals[ci] if ci < len(vals) else None
            if value is None:
                continue
            x = start + si * (bar_w + 2.0)
            h = plot_h * value / y_max
            canvas.rect(x, plot_y, bar_w, h, fill=color, stroke=(0.2, 0.2, 0.2), stroke_width=0.25)
            canvas.text(x + bar_w / 2.0, plot_y + h + 3.0, f"{value:.{value_digits}f}", args.legend_font_size, align="center")
        label_lines = wrap_label(category)
        canvas.multiline_text(center, plot_y - 14.0, label_lines, args.axis_font_size, align="center", leading=args.axis_font_size * 1.05)
    canvas.text(plot_x, plot_y + plot_h + 5.0, y_label, args.axis_font_size, color=(0.15, 0.15, 0.15), align="left")
    legend_y = canvas.height - 34.0 - 8.0 * len(note_lines)
    draw_legend_wrapped(canvas, legend_items, plot_x, legend_y, plot_w, args.legend_font_size)
    canvas.save(path)
    return path


def best_benchmark_rows(rows: List[ResultRow]) -> List[ResultRow]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.ok]
    result: List[ResultRow] = []
    for dim in [3, 4]:
        for suite, transport in [("ffts", "single_gpu"), ("fftm", "cuda_aware"), ("fftm", "non_cuda_aware")]:
            best = best_by_throughput(
                r for r in benchmark if r.dim == dim and r.suite == suite and r.transport == transport
            )
            if best:
                result.append(best)
    return result


def throughput_figure_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.suite == "fftm" and r.dim == dim and r.ok]
    if not benchmark:
        return None
    strategies = [s for s in STRATEGY_ORDER.get(dim, []) if any(r.strategy == s for r in benchmark)]
    if not strategies:
        strategies = sorted(set(r.strategy for r in benchmark))
    series = []
    for transport, color in [
        ("cuda_aware", (0.12, 0.32, 0.70)),
        ("non_cuda_aware", (0.88, 0.45, 0.10)),
    ]:
        values = []
        for strategy in strategies:
            best = best_by_throughput(r for r in benchmark if r.transport == transport and r.strategy == strategy)
            values.append(best.throughput_gpoints_s if best else None)
        series.append((transport_label(transport), values, color))
    return bar_chart_pdf(
        fig_dir / f"fig_fftm_{dim}d_best_throughput.pdf",
        f"{dim}D FFTM: best throughput by decomposition",
        strategies,
        series,
        "Gpoints/s",
        args,
        value_digits=2,
        note="Best redistribution mode selected for each bar",
    )


def strategy_mode_groups(rows: List[ResultRow], dim: int) -> List[Tuple[str, str, List[ResultRow]]]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.suite == "fftm" and r.dim == dim]
    if not benchmark:
        return []
    configured = STRATEGY_ORDER.get(dim, [])
    extra = sorted(set(r.strategy for r in benchmark if r.strategy not in configured))
    strategies = [s for s in configured if any(r.strategy == s for r in benchmark)] + extra
    result = []
    for strategy in strategies:
        for transport in ["cuda_aware", "non_cuda_aware"]:
            group = [r for r in benchmark if r.strategy == strategy and r.transport == transport]
            if group:
                result.append((strategy, transport, group))
    return result


def best_mode_record(group: Sequence[ResultRow], mode: str) -> Optional[ResultRow]:
    matching = [r for r in group if r.mode == mode]
    if not matching:
        return None
    successful = [r for r in matching if r.ok]
    if successful:
        return min(successful, key=lambda r: r.avg_wall_ms if r.avg_wall_ms is not None else 1.0e300)
    return matching[0]


def heatmap_color(value: float, vmin: float, vmax: float, lower_better: bool = True) -> Tuple[float, float, float]:
    if vmax <= vmin:
        t = 0.0
    elif vmin > 0.0 and vmax / vmin > 6.0:
        lv = math.log10(max(value, 1.0e-300))
        lo = math.log10(vmin)
        hi = math.log10(vmax)
        t = (lv - lo) / (hi - lo)
    else:
        t = (value - vmin) / (vmax - vmin)
    t = min(max(t, 0.0), 1.0)
    if not lower_better:
        t = 1.0 - t
    return (0.86 + 0.14 * t, 0.96 - 0.42 * t, 0.82 - 0.54 * t)


def strategy_mode_heatmap_pdf(
    *,
    rows: List[ResultRow],
    dim: int,
    fig_dir: Path,
    args: argparse.Namespace,
    path_name: str,
    title: str,
    subtitle: str,
    value_getter,
    text_getter,
    lower_better: bool = True,
) -> Optional[Path]:
    rows_present = strategy_mode_groups(rows, dim)
    if not rows_present:
        return None
    mode_values: Dict[Tuple[int, str], float] = {}
    values: List[float] = []
    for ri, (_, _, group) in enumerate(rows_present):
        for mode in MODE_ORDER:
            rec = best_mode_record(group, mode)
            value = value_getter(rec) if rec is not None and rec.ok else None
            if value is None or value <= 0.0:
                continue
            mode_values[(ri, mode)] = float(value)
            values.append(float(value))
    if not values:
        return None
    vmin, vmax = min(values), max(values)

    cell_w, cell_h = 58.0, 21.0
    label_w = 94.0
    top = 57.0
    left = 12.0
    width = max(cm_to_pt(args.figure_width_cm), left + label_w + len(MODE_ORDER) * cell_w + 22.0)
    height = max(cm_to_pt(args.figure_height_cm), top + len(rows_present) * cell_h + 39.0)
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, title, args.title_font_size, align="center")
    canvas.text(width / 2.0, height - 32.0, subtitle, args.legend_font_size, color=(0.25, 0.25, 0.25), align="center")

    x0 = left + label_w
    y0 = height - top
    for mi, mode in enumerate(MODE_ORDER):
        canvas.text(
            x0 + mi * cell_w + cell_w / 2.0,
            y0 + 10.0,
            mode.replace("p2p-", "p2p "),
            args.axis_font_size,
            align="center",
        )
    for ri, (strategy, transport, group) in enumerate(rows_present):
        y = y0 - (ri + 1) * cell_h
        label = f"{strategy} {('CA' if transport == 'cuda_aware' else 'NCA')}"
        canvas.text(left + label_w - 4.0, y + 6.5, label, args.axis_font_size, align="right")
        for mi, mode in enumerate(MODE_ORDER):
            x = x0 + mi * cell_w
            rec = best_mode_record(group, mode)
            value = mode_values.get((ri, mode))
            if rec is None:
                color = (0.92, 0.92, 0.92)
                text = "-"
            elif not rec.ok:
                color = (0.70, 0.70, 0.70)
                text = "fail"
            elif value is None:
                color = (0.92, 0.92, 0.92)
                text = "-"
            else:
                color = heatmap_color(value, vmin, vmax, lower_better=lower_better)
                text = text_getter(value)
            canvas.rect(x, y, cell_w - 2.0, cell_h - 2.0, fill=color, stroke=(0.65, 0.65, 0.65), stroke_width=0.25)
            canvas.text(x + (cell_w - 2.0) / 2.0, y + 6.5, text, args.axis_font_size, align="center")
    canvas.text(
        x0,
        13.0,
        "CA = CUDA-aware MPI; NCA = non-CUDA-aware. Darker cells are slower/larger.",
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
    )
    path = fig_dir / path_name
    canvas.save(path)
    return path


def runtime_absolute_heatmap_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    return strategy_mode_heatmap_pdf(
        rows=rows,
        dim=dim,
        fig_dir=fig_dir,
        args=args,
        path_name=f"fig_fftm_{dim}d_mode_time.pdf",
        title=f"{dim}D FFTM: absolute runtime by strategy and mode",
        subtitle="Cell values are average forward+backward wall time [ms]; lower is better.",
        value_getter=lambda rec: rec.avg_wall_ms if rec else None,
        text_getter=axis_tick_label,
        lower_better=True,
    )


def memory_peak_heatmap_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    return strategy_mode_heatmap_pdf(
        rows=rows,
        dim=dim,
        fig_dir=fig_dir,
        args=args,
        path_name=f"fig_fftm_{dim}d_mode_memory.pdf",
        title=f"{dim}D FFTM: device-memory peak by strategy and mode",
        subtitle="Cell values are measured peak device memory [MiB] per process/GPU; lower is better.",
        value_getter=lambda rec: rec.used_device_peak_max_mib if rec else None,
        text_getter=axis_tick_label,
        lower_better=True,
    )


def runtime_relative_heatmap_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    rows_present = strategy_mode_groups(rows, dim)
    if not rows_present:
        return None

    cell_w, cell_h = 50.0, 21.0
    label_w = 88.0
    top = 53.0
    left = 12.0
    width = max(cm_to_pt(args.figure_width_cm), left + label_w + len(MODE_ORDER) * cell_w + 18.0)
    height = max(cm_to_pt(args.figure_height_cm), top + len(rows_present) * cell_h + 34.0)
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, f"{dim}D FFTM: relative runtime by redistribution mode", args.title_font_size, align="center")
    canvas.text(
        width / 2.0,
        height - 32.0,
        "Runtime is normalized by the best successful mode in each row; lower is better.",
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    x0 = left + label_w
    y0 = height - top
    for mi, mode in enumerate(MODE_ORDER):
        canvas.text(x0 + mi * cell_w + cell_w / 2.0, y0 + 10.0, mode.replace("p2p-", "p2p "), args.axis_font_size, align="center")
    for ri, (strategy, transport, group) in enumerate(rows_present):
        y = y0 - (ri + 1) * cell_h
        label = f"{strategy} {('CA' if transport == 'cuda_aware' else 'NCA')}"
        canvas.text(left + label_w - 4.0, y + 6.5, label, args.axis_font_size, align="right")
        ok_times = [r.avg_wall_ms for r in group if r.ok and r.avg_wall_ms is not None]
        best = min(ok_times) if ok_times else None
        mode_map = {r.mode: r for r in group}
        for mi, mode in enumerate(MODE_ORDER):
            x = x0 + mi * cell_w
            rec = mode_map.get(mode)
            if rec is None:
                color = (0.92, 0.92, 0.92)
                text = "-"
            elif not rec.ok or rec.avg_wall_ms is None or best is None:
                color = (0.70, 0.70, 0.70)
                text = "fail"
            else:
                rel = rec.avg_wall_ms / best
                t = min(max((rel - 1.0) / 1.5, 0.0), 1.0)
                color = (1.0, 0.96 - 0.42 * t, 0.80 - 0.55 * t)
                text = f"{rel:.2f}"
            canvas.rect(x, y, cell_w - 2.0, cell_h - 2.0, fill=color, stroke=(0.65, 0.65, 0.65), stroke_width=0.25)
            canvas.text(x + (cell_w - 2.0) / 2.0, y + 6.5, text, args.axis_font_size, align="center")
    canvas.text(x0, 13.0, "CA = CUDA-aware MPI, NCA = non-CUDA-aware path.", args.legend_font_size, color=(0.25, 0.25, 0.25))
    path = fig_dir / f"fig_fftm_{dim}d_mode_relative.pdf"
    canvas.save(path)
    return path


def transport_speedup_pdf(rows: List[ResultRow], fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.suite == "fftm" and r.ok]
    categories: List[str] = []
    values: List[Optional[float]] = []
    for dim in [3, 4]:
        for strategy in STRATEGY_ORDER.get(dim, []):
            ca = best_by_throughput(r for r in benchmark if r.dim == dim and r.strategy == strategy and r.transport == "cuda_aware")
            nca = best_by_throughput(r for r in benchmark if r.dim == dim and r.strategy == strategy and r.transport == "non_cuda_aware")
            if ca and nca and ca.avg_wall_ms and nca.avg_wall_ms:
                categories.append(f"{dim}D {strategy}")
                values.append(ca.avg_wall_ms / nca.avg_wall_ms)
    if not categories:
        return None
    return bar_chart_pdf(
        fig_dir / "fig_cuda_aware_vs_nca_speedup.pdf",
        "CUDA-aware vs non-CUDA-aware best-mode runtime",
        categories,
        [("CA time / NCA time", values, (0.34, 0.58, 0.18))],
        "Ratio",
        args,
        value_digits=2,
        reference_y=1.0,
        note="Values above 1 indicate faster non-CUDA-aware execution.",
    )


def memory_footprint_pdf(rows: List[ResultRow], fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    best_rows = best_benchmark_rows(rows)
    if not best_rows:
        return None
    categories: List[str] = []
    used: List[Optional[float]] = []
    internal: List[Optional[float]] = []
    host: List[Optional[float]] = []
    for r in best_rows:
        label = f"{r.dim}D "
        if r.suite == "ffts":
            label += "FFTS"
        elif r.transport == "cuda_aware":
            label += "FFTM CA"
        else:
            label += "FFTM NCA"
        categories.append(label)
        used.append(r.used_device_peak_max_mib)
        internal.append(r.internal_device_peak_max_mib)
        host.append(r.host_pinned_peak_max_mib)
    return bar_chart_pdf(
        fig_dir / "fig_memory_footprint.pdf",
        "Memory footprint of best benchmark configurations",
        categories,
        [
            ("device peak", used, (0.13, 0.36, 0.70)),
            ("tracked device", internal, (0.88, 0.45, 0.10)),
            ("host pinned", host, (0.45, 0.45, 0.45)),
        ],
        "MiB per process/GPU",
        args,
        value_digits=0,
        note="Device peak comes from collected device-memory measurements; tracked values come from library profiling.",
    )


def best_scaling_size(rows: List[ResultRow], dim: int) -> Optional[int]:
    by_size: Dict[int, set] = defaultdict(set)
    for row in rows:
        if row.case_name != "benchmark" or row.suite != "fftm" or row.dim != dim:
            continue
        if not row.ok or row.avg_wall_ms is None:
            continue
        n = size_n(row)
        if n is not None:
            by_size[n].add(row.num_gpus)
    candidates = [n for n, gpu_counts in by_size.items() if len(gpu_counts) >= 2]
    return max(candidates) if candidates else None


def gpu_scaling_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    selected_n = best_scaling_size(rows, dim)
    benchmark = [
        row
        for row in rows
        if row.case_name == "benchmark" and row.dim == dim and row.ok and row.avg_wall_ms is not None
    ]
    if selected_n is not None:
        benchmark = [row for row in benchmark if size_n(row) == selected_n]
    fftm = [row for row in benchmark if row.suite == "fftm"]
    if not fftm:
        return None
    x_values = sorted(set(row.num_gpus for row in fftm))
    if len(x_values) < 2:
        return None

    colors = [
        (0.10, 0.32, 0.70),
        (0.88, 0.35, 0.08),
        (0.12, 0.55, 0.20),
        (0.72, 0.20, 0.18),
        (0.52, 0.24, 0.68),
        (0.35, 0.35, 0.35),
    ]
    primary = []
    color_index = 0
    for strategy in STRATEGY_ORDER.get(dim, sorted(set(row.strategy for row in fftm))):
        for transport in ["cuda_aware", "non_cuda_aware"]:
            points = []
            for gpu_count in x_values:
                best = best_by_throughput(
                    row
                    for row in fftm
                    if row.strategy == strategy and row.transport == transport and row.num_gpus == gpu_count
                )
                if best:
                    value = best.avg_wall_ms if selected_n is not None else best.throughput_gpoints_s
                    if value is not None:
                        points.append((gpu_count, value))
            if points:
                label = f"{strategy} {('CA' if transport == 'cuda_aware' else 'NCA')}"
                primary.append((label, points, colors[color_index % len(colors)]))
                color_index += 1

    if selected_n is not None:
        ffts = best_by_throughput(row for row in benchmark if row.suite == "ffts")
        if ffts and ffts.avg_wall_ms is not None:
            primary.append(("FFTS 1GPU", [(gpu_count, ffts.avg_wall_ms) for gpu_count in x_values], (0.78, 0.20, 0.62)))
        secondary = []
        for label, points, color in primary:
            if label == "FFTS 1GPU" or not points:
                continue
            base = points[0][1]
            if base <= 0:
                continue
            secondary.append((label, [(gpu_count, base / value) for gpu_count, value in points if value > 0], color))
        title = f"{dim}D FFTM GPU-count strong scaling at N={selected_n}"
        subtitle = "Best successful mode selected per strategy/transport/GPU count."
        left_title = "Runtime"
        left_label = "Time [ms]"
        right_title = "Strong-scaling speedup"
        right_label = "Speedup"
        left_log = True
    else:
        size_points = []
        for gpu_count in x_values:
            sizes = [size_n(row) for row in fftm if row.num_gpus == gpu_count and size_n(row) is not None]
            if sizes:
                size_points.append((gpu_count, max(sizes)))
        secondary = [("chosen N", size_points, (0.10, 0.32, 0.70))] if size_points else []
        title = f"{dim}D FFTM GPU-count capacity scaling"
        subtitle = "Each GPU count uses its fitted FFT-friendly maximum benchmark size."
        left_title = "Best throughput"
        left_label = "Gpoints/s"
        right_title = "Fitted problem side"
        right_label = "N"
        left_log = False

    width = 535.0
    height = 292.0
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, title, args.title_font_size, align="center")
    canvas.text(
        width / 2.0,
        height - 32.0,
        subtitle,
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    legend_items = [(label, color) for label, _, color in primary]
    draw_legend_wrapped(canvas, legend_items, 58.0, height - 49.0, width - 116.0, args.legend_font_size)
    plot_y = 56.0
    plot_h = height - 145.0
    gap = 58.0
    plot_w = (width - 72.0 - gap - 18.0) / 2.0
    draw_line_panel(
        canvas,
        (58.0, plot_y, plot_w, plot_h),
        left_title,
        x_values,
        primary,
        left_label,
        "MPI ranks / GPUs",
        args,
        log_y=left_log,
    )
    draw_line_panel(
        canvas,
        (58.0 + plot_w + gap, plot_y, plot_w, plot_h),
        right_title,
        x_values,
        secondary,
        right_label,
        "MPI ranks / GPUs",
        args,
        log_y=False,
        y_min_override=0.0,
    )
    path = fig_dir / f"fig_fftm_{dim}d_gpu_scaling.pdf"
    canvas.save(path)
    return path


def gflops_gpu_scaling_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    benchmark = [
        row
        for row in rows
        if row.case_name == "benchmark"
        and row.dim == dim
        and row.suite == "fftm"
        and row.ok
        and row.effective_gflops_s is not None
    ]
    if not benchmark:
        return None
    x_values = sorted(set(row.num_gpus for row in benchmark))
    if len(x_values) < 2:
        return None

    colors = [
        (0.10, 0.32, 0.70),
        (0.88, 0.35, 0.08),
        (0.12, 0.55, 0.20),
        (0.72, 0.20, 0.18),
        (0.52, 0.24, 0.68),
        (0.35, 0.35, 0.35),
    ]
    total_series = []
    per_gpu_series = []
    color_index = 0
    for strategy in STRATEGY_ORDER.get(dim, sorted(set(row.strategy for row in benchmark))):
        for transport in ["cuda_aware", "non_cuda_aware"]:
            total_points: List[Tuple[int, float]] = []
            per_gpu_points: List[Tuple[int, float]] = []
            for gpu_count in x_values:
                best = max(
                    (
                        row
                        for row in benchmark
                        if row.strategy == strategy and row.transport == transport and row.num_gpus == gpu_count
                    ),
                    key=lambda row: row.effective_gflops_s or -1.0,
                    default=None,
                )
                if best and best.effective_gflops_s is not None:
                    total_points.append((gpu_count, best.effective_gflops_s))
                    per_gpu_points.append((gpu_count, best.effective_gflops_s / float(max(1, gpu_count))))
            if total_points:
                label = f"{strategy} {('CA' if transport == 'cuda_aware' else 'NCA')}"
                color = colors[color_index % len(colors)]
                total_series.append((label, total_points, color))
                per_gpu_series.append((label, per_gpu_points, color))
                color_index += 1

    if not total_series:
        return None

    width = 535.0
    height = 292.0
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, f"{dim}D FFTM effective GFLOP/s vs GPU count", args.title_font_size, align="center")
    canvas.text(
        width / 2.0,
        height - 32.0,
        "Effective rate uses 10*N*log2(N) operations for the measured forward+inverse pair.",
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    legend_items = [(label, color) for label, _, color in total_series]
    draw_legend_wrapped(canvas, legend_items, 58.0, height - 49.0, width - 116.0, args.legend_font_size)
    plot_y = 56.0
    plot_h = height - 145.0
    gap = 58.0
    plot_w = (width - 72.0 - gap - 18.0) / 2.0
    draw_line_panel(
        canvas,
        (58.0, plot_y, plot_w, plot_h),
        "Total effective rate",
        x_values,
        total_series,
        "GFLOP/s",
        "MPI ranks / GPUs",
        args,
        log_y=False,
        y_min_override=0.0,
    )
    draw_line_panel(
        canvas,
        (58.0 + plot_w + gap, plot_y, plot_w, plot_h),
        "Effective rate per GPU",
        x_values,
        per_gpu_series,
        "GFLOP/s/GPU",
        "MPI ranks / GPUs",
        args,
        log_y=False,
        y_min_override=0.0,
    )
    path = fig_dir / f"fig_fftm_{dim}d_effective_gflops_gpu_scaling.pdf"
    canvas.save(path)
    return path


def ffts_fftm_relative_pdf(rows: List[ResultRow], fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.ok]
    categories: List[str] = []
    values: List[Optional[float]] = []
    for dim in [3, 4]:
        baseline = best_by_throughput(r for r in benchmark if r.suite == "ffts" and r.dim == dim)
        for transport in ["cuda_aware", "non_cuda_aware"]:
            best = best_by_throughput(r for r in benchmark if r.suite == "fftm" and r.dim == dim and r.transport == transport)
            if baseline and best and baseline.throughput_gpoints_s:
                categories.append(f"{dim}D {('CA' if transport == 'cuda_aware' else 'NCA')}")
                values.append((best.throughput_gpoints_s or 0.0) / baseline.throughput_gpoints_s)
    if not categories:
        return None
    return bar_chart_pdf(
        fig_dir / "fig_fftm_relative_to_ffts.pdf",
        "Best FFTM throughput relative to single-GPU FFTS",
        categories,
        [("relative throughput", values, (0.52, 0.24, 0.68))],
        "FFTM / FFTS",
        args,
        value_digits=2,
        reference_y=1.0,
        note="Throughput normalization is used because maximum fitted sizes can differ.",
    )


def cleanup_stale_figure_sources(fig_dir: Path) -> None:
    for pattern in ("fig_*.tex", "fig_*.compile.log"):
        for path in fig_dir.glob(pattern):
            path.unlink()


def size_n(row: ResultRow) -> Optional[int]:
    if not row.sizes:
        return None
    if len(set(row.sizes)) == 1:
        return int(row.sizes[0])
    # For non-cubic/non-hypercubic future data, use geometric mean as a scalar
    # abscissa while preserving the full size in CSV/table outputs.
    return int(round(row.points ** (1.0 / max(1, row.dim))))


def merge_duplicate_points(points: Iterable[Tuple[int, float]]) -> List[Tuple[int, float]]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for n, value in points:
        grouped[int(n)].append(float(value))
    return sorted((n, sum(values) / len(values)) for n, values in grouped.items())


def benchmark_sweep_series(rows: List[ResultRow], dim: int) -> List[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]]:
    colors = {
        "slab-pencil": (0.10, 0.32, 0.70),
        "pencil-slab": (0.88, 0.35, 0.08),
        "pencil-pencil": (0.12, 0.55, 0.20),
        "slab-slab": (0.10, 0.32, 0.70),
        "FFTS best": (0.78, 0.20, 0.62),
    }
    result: List[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]] = []
    strategies = STRATEGY_ORDER.get(dim, [])
    for strategy in strategies:
        points = []
        for r in rows:
            if (
                r.case_name == "benchmark"
                and r.suite == "fftm"
                and r.dim == dim
                and r.transport == "cuda_aware"
                and r.strategy == strategy
                and r.mode == "p2p-waitany"
                and r.ok
                and r.avg_wall_ms is not None
            ):
                n = size_n(r)
                if n is not None:
                    points.append((n, r.avg_wall_ms))
        points = merge_duplicate_points(points)
        if points:
            result.append((strategy, points, colors.get(strategy, (0.3, 0.3, 0.3))))

    ffts_by_n: Dict[int, List[float]] = defaultdict(list)
    for r in rows:
        if (
            r.case_name == "benchmark"
            and r.suite == "ffts"
            and r.dim == dim
            and r.ok
            and r.avg_wall_ms is not None
        ):
            n = size_n(r)
            if n is not None:
                ffts_by_n[n].append(r.avg_wall_ms)
    ffts_points = sorted((n, min(values)) for n, values in ffts_by_n.items())
    if ffts_points:
        result.append(("FFTS best", ffts_points, colors["FFTS best"]))
    return result


def relative_series(
    absolute: Sequence[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]],
    dim: int,
) -> List[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]]:
    # Match Egger's convention: normalize to the fastest distributed FFTM
    # variant for a given size; single-GPU FFTS is shown relative to that
    # distributed baseline when the same size is available.
    distributed_by_n: Dict[int, List[float]] = defaultdict(list)
    for label, points, _ in absolute:
        if label == "FFTS best":
            continue
        for n, value in points:
            distributed_by_n[n].append(value)
    baseline = {n: min(values) for n, values in distributed_by_n.items() if values}
    result = []
    for label, points, color in absolute:
        rel_points = [(n, value / baseline[n]) for n, value in points if n in baseline and baseline[n] > 0]
        if rel_points:
            result.append((label, rel_points, color))
    return result


def log_ticks(y_min: float, y_max: float) -> List[float]:
    start = math.floor(math.log10(max(y_min, 1.0e-12)))
    end = math.ceil(math.log10(max(y_max, y_min * 1.01)))
    return [10.0**k for k in range(start, end + 1)]


def linear_ticks(y_min: float, y_max: float, count: int = 5) -> List[float]:
    if y_max <= y_min:
        y_max = y_min + 1.0
    return [y_min + (y_max - y_min) * i / (count - 1) for i in range(count)]


def axis_tick_label(value: float) -> str:
    if abs(value) >= 10000.0:
        return f"{value / 1000.0:.0f}k"
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.1f}k"
    if abs(value) >= 100.0:
        return f"{value:.0f}"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    return fmt_float(value, 2)


def draw_line_panel(
    canvas: PdfCanvas,
    rect: Tuple[float, float, float, float],
    title: str,
    x_values: Sequence[int],
    series: Sequence[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]],
    y_label: str,
    x_label: str,
    args: argparse.Namespace,
    log_y: bool = False,
    y_min_override: Optional[float] = None,
    y_max_override: Optional[float] = None,
) -> None:
    x0, y0, w, h = rect
    values = [value for _, points, _ in series for _, value in points if value > 0]
    if not values or not x_values:
        canvas.text(x0 + w / 2.0, y0 + h / 2.0, "no data", args.axis_font_size, align="center")
        return
    y_min = y_min_override if y_min_override is not None else min(values)
    y_max = y_max_override if y_max_override is not None else max(values)
    if log_y:
        y_min = max(y_min * 0.75, 1.0e-6)
        y_max = max(y_max * 1.35, y_min * 10.0)
        transform = lambda v: (math.log10(v) - math.log10(y_min)) / (math.log10(y_max) - math.log10(y_min))
        ticks = [t for t in log_ticks(y_min, y_max) if y_min <= t <= y_max]
    else:
        y_min = min(0.0, y_min_override if y_min_override is not None else y_min * 0.95)
        y_max = nice_axis_max(y_max * 1.12)
        transform = lambda v: (v - y_min) / (y_max - y_min)
        ticks = linear_ticks(y_min, y_max, 5)

    def x_coord(n: int) -> float:
        if len(x_values) == 1:
            return x0 + w / 2.0
        idx = x_values.index(n)
        return x0 + w * idx / (len(x_values) - 1)

    def y_coord(v: float) -> float:
        t = max(0.0, min(1.0, transform(max(v, 1.0e-30))))
        return y0 + h * t

    canvas.text(x0 + w / 2.0, y0 + h + 18.0, title, args.axis_font_size, align="center")
    canvas.line(x0, y0, x0 + w, y0, color=(0, 0, 0), width=0.6)
    canvas.line(x0, y0, x0, y0 + h, color=(0, 0, 0), width=0.6)
    for tick in ticks:
        y = y_coord(tick)
        canvas.line(x0, y, x0 + w, y, color=(0.84, 0.84, 0.84), width=0.25)
        label = f"$10^{int(round(math.log10(tick)))}$" if log_y and tick > 0 else axis_tick_label(tick)
        if log_y:
            label = f"1e{int(round(math.log10(tick)))}"
        canvas.text(x0 - 4.0, y - 2.5, label, args.legend_font_size, color=(0.22, 0.22, 0.22), align="right")
    max_label_width = max((PdfCanvas.text_width(str(n), args.legend_font_size) for n in x_values), default=1.0)
    label_step = max(1, int(math.ceil((len(x_values) * max_label_width) / max(w * 0.72, 1.0))))
    for i, n in enumerate(x_values):
        x = x_coord(n)
        canvas.line(x, y0, x, y0 + h, color=(0.88, 0.88, 0.88), width=0.2)
        if i % label_step == 0 or i == len(x_values) - 1:
            canvas.text(x, y0 - 12.0, str(n), args.legend_font_size, color=(0.22, 0.22, 0.22), align="center")
    canvas.text(x0 + w / 2.0, y0 - 27.0, x_label, args.legend_font_size, color=(0.15, 0.15, 0.15), align="center")
    canvas.text(x0, y0 + h + 5.0, y_label, args.legend_font_size, color=(0.15, 0.15, 0.15), align="left")

    for label, points, color in series:
        point_map = {n: value for n, value in points}
        ordered = [(n, point_map[n]) for n in x_values if n in point_map and point_map[n] > 0]
        for (n0, v0), (n1, v1) in zip(ordered, ordered[1:]):
            canvas.line(x_coord(n0), y_coord(v0), x_coord(n1), y_coord(v1), color=color, width=0.9)
        for n, value in ordered:
            x, y = x_coord(n), y_coord(value)
            canvas.rect(x - 1.6, y - 1.6, 3.2, 3.2, fill=color, stroke=(0.1, 0.1, 0.1), stroke_width=0.15)


def egger_style_runtime_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    absolute = benchmark_sweep_series(rows, dim)
    if len(absolute) < 2:
        return None
    relative = relative_series(absolute, dim)
    x_values = sorted(set(n for _, points, _ in absolute for n, _ in points))
    width = max(cm_to_pt(args.figure_width_cm * 1.75), 475.0)
    height = max(cm_to_pt(args.figure_height_cm * 1.45), 285.0)
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, f"{dim}D benchmark runtime: FFTM decomposition methods and FFTS", args.title_font_size, align="center")
    canvas.text(
        width / 2.0,
        height - 32.0,
        "CUDA-aware p2p-waitany sweep; relative values use best FFTM at the same N.",
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    plot_y = 56.0
    plot_h = height - 153.0
    gap = 58.0
    plot_w = (width - 72.0 - gap - 18.0) / 2.0
    left_rect = (58.0, plot_y, plot_w, plot_h)
    right_rect = (58.0 + plot_w + gap, plot_y, plot_w, plot_h)
    draw_line_panel(
        canvas,
        left_rect,
        "Absolute runtime",
        x_values,
        absolute,
        "Time [ms]",
        f"N for N^{dim}",
        args,
        log_y=True,
    )
    draw_line_panel(
        canvas,
        right_rect,
        "Relative runtime",
        x_values,
        relative,
        "Relative",
        f"N for N^{dim}",
        args,
        log_y=False,
        y_min_override=0.0,
    )
    legend_items = [(label, color) for label, _, color in absolute]
    draw_legend(canvas, legend_items[:4], 58.0, height - 47.0, args.legend_font_size)
    if len(legend_items) > 4:
        draw_legend(canvas, legend_items[4:], 58.0, height - 58.0, args.legend_font_size)
    path = fig_dir / f"fig_{dim}d_fftm_ffts_absolute_relative_runtime.pdf"
    canvas.save(path)
    return path


def rows_by_size_best(
    rows: Iterable[ResultRow],
    *,
    dim: int,
    suite: str,
    transport: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[int, ResultRow]:
    grouped: Dict[int, List[ResultRow]] = defaultdict(list)
    for row in rows:
        if row.phase != "probe":
            continue
        if row.case_name != "benchmark" or row.suite != suite or row.dim != dim:
            continue
        if transport is not None and row.transport != transport:
            continue
        if mode is not None and row.mode != mode:
            continue
        if not row.ok or row.avg_wall_ms is None:
            continue
        n = size_n(row)
        if n is not None:
            grouped[n].append(row)
    if not grouped:
        # Fall back to measurement rows if a dataset was collected without probes.
        for row in rows:
            if row.case_name != "benchmark" or row.suite != suite or row.dim != dim:
                continue
            if transport is not None and row.transport != transport:
                continue
            if mode is not None and row.mode != mode:
                continue
            if not row.ok or row.avg_wall_ms is None:
                continue
            n = size_n(row)
            if n is not None:
                grouped[n].append(row)
    return {n: best_by_throughput(candidates) or min(candidates, key=lambda r: r.avg_wall_ms or 1.0e300) for n, candidates in grouped.items()}


def derived_temporal_map(row: ResultRow) -> Dict[str, float]:
    derived = {entry["name"]: as_float(entry.get("ms")) or 0.0 for entry in derived_temporal_entries(row)}
    root = profile_root_ms(row)
    local = derived.get("derived: local FFT applications", 0.0)
    communication = derived.get("derived: total communication/transposes", 0.0)
    other = max(0.0, (root or 0.0) - local - communication)
    return {
        "local FFT": local,
        "communication": communication,
        "other/profile": other,
        "total": root or (local + communication + other),
    }


def entry_name_ms(entries: Sequence[dict]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for entry in entries:
        name = str(entry.get("name", "")).strip()
        if not name:
            continue
        result[name] = result.get(name, 0.0) + (as_float(entry.get("ms")) or 0.0)
    return result


def transform_direction_map(row: ResultRow) -> Dict[str, float]:
    entries = row.profile_summary_entries or aggregate_profile_breakdown(row.profile_breakdown_entries)
    values = entry_name_ms(entries)
    result = {
        "forward local FFT": 0.0,
        "forward communication": 0.0,
        "forward total": 0.0,
        "backward local FFT": 0.0,
        "backward communication": 0.0,
        "backward total": 0.0,
    }
    for name, ms in values.items():
        if row.suite == "ffts":
            if name in {"ffts::forward_3d", "ffts::forward_4d"}:
                result["forward local FFT"] += ms
                result["forward total"] += ms
            elif name in {"ffts::backward_3d", "ffts::backward_4d"}:
                result["backward local FFT"] += ms
                result["backward total"] += ms
        else:
            forward_total = bool(re.match(r"^fftm::forward_[34]d_", name))
            backward_total = bool(re.match(r"^fftm::backward_[34]d_", name))
            forward_comm = is_top_level_communication_entry(name) and "_to_" in name and (
                "xyz_to_xzy" in name
                or "x_to_y" in name
                or "xyzw_to_xywz" in name
                or "xywz_to_xzwy" in name
                or "xzwy_to_yzwx" in name
            )
            backward_comm = is_top_level_communication_entry(name) and "_to_" in name and (
                "xzy_to_xyz" in name
                or "y_to_x" in name
                or "yzwx_to_xzwy" in name
                or "xzwy_to_xywz" in name
                or "xywz_to_xyzw" in name
            )
            if forward_total:
                result["forward total"] += ms
            if backward_total:
                result["backward total"] += ms
            if forward_comm:
                result["forward communication"] += ms
            if backward_comm:
                result["backward communication"] += ms
    if row.suite == "fftm":
        result["forward local FFT"] = max(0.0, result["forward total"] - result["forward communication"])
        result["backward local FFT"] = max(0.0, result["backward total"] - result["backward communication"])
    return result


def profile_series_from_map(
    by_size: Dict[int, ResultRow],
    component: str,
    value_getter,
) -> List[Tuple[int, float]]:
    points = []
    for n, row in sorted(by_size.items()):
        value = value_getter(row, component)
        if value is not None and value > 0:
            points.append((n, value))
    return points


def line_chart_pdf(
    path: Path,
    title: str,
    subtitle: str,
    x_values: Sequence[int],
    series: Sequence[Tuple[str, List[Tuple[int, float]], Tuple[float, float, float]]],
    y_label: str,
    x_label: str,
    args: argparse.Namespace,
    log_y: bool = False,
    min_width_pt: float = 415.0,
    min_height_pt: float = 260.0,
) -> Optional[Path]:
    if not series or not any(points for _, points, _ in series):
        return None
    width = max(cm_to_pt(args.figure_width_cm * 1.45), min_width_pt)
    height = max(cm_to_pt(args.figure_height_cm * 1.25), min_height_pt)
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, title, args.title_font_size, align="center")
    canvas.text(width / 2.0, height - 32.0, subtitle, args.legend_font_size, color=(0.25, 0.25, 0.25), align="center")
    legend_items = [(label, color) for label, _, color in series]
    draw_legend_wrapped(canvas, legend_items, 54.0, height - 48.0, width - 92.0, args.legend_font_size)
    rect = (58.0, 55.0, width - 78.0, height - 128.0)
    draw_line_panel(canvas, rect, "", x_values, series, y_label, x_label, args, log_y=log_y)
    canvas.save(path)
    return path


def temporal_breakdown_vs_size_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    fftm_by_size = rows_by_size_best(rows, dim=dim, suite="fftm", transport="cuda_aware", mode="p2p-waitany")
    ffts_by_size = rows_by_size_best(rows, dim=dim, suite="ffts", transport="single_gpu")
    if not fftm_by_size and not ffts_by_size:
        return None
    x_values = sorted(set(fftm_by_size.keys()) | set(ffts_by_size.keys()))
    colors = {
        "FFTM local FFT": (0.12, 0.32, 0.70),
        "FFTM communication": (0.86, 0.34, 0.10),
        "FFTM other": (0.45, 0.45, 0.45),
        "FFTS local FFT": (0.56, 0.20, 0.66),
    }
    series = [
        (
            "FFTM local FFT",
            profile_series_from_map(fftm_by_size, "local FFT", lambda row, component: derived_temporal_map(row).get(component)),
            colors["FFTM local FFT"],
        ),
        (
            "FFTM communication",
            profile_series_from_map(fftm_by_size, "communication", lambda row, component: derived_temporal_map(row).get(component)),
            colors["FFTM communication"],
        ),
        (
            "FFTM other",
            profile_series_from_map(fftm_by_size, "other/profile", lambda row, component: derived_temporal_map(row).get(component)),
            colors["FFTM other"],
        ),
        (
            "FFTS local FFT",
            profile_series_from_map(ffts_by_size, "local FFT", lambda row, component: derived_temporal_map(row).get(component)),
            colors["FFTS local FFT"],
        ),
    ]
    return line_chart_pdf(
        fig_dir / f"fig_{dim}d_temporal_breakdown_vs_size.pdf",
        f"{dim}D temporal profile breakdown vs problem size",
        "Best CUDA-aware FFTM p2p-waitany strategy at each N; FFTS best baseline included.",
        x_values,
        series,
        "Accumulated time [ms]",
        f"N for N^{dim}",
        args,
        log_y=True,
    )


def temporal_forward_backward_vs_size_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    fftm_by_size = rows_by_size_best(rows, dim=dim, suite="fftm", transport="cuda_aware", mode="p2p-waitany")
    ffts_by_size = rows_by_size_best(rows, dim=dim, suite="ffts", transport="single_gpu")
    if not fftm_by_size and not ffts_by_size:
        return None
    x_values = sorted(set(fftm_by_size.keys()) | set(ffts_by_size.keys()))

    def direction_value(row: ResultRow, component: str) -> Optional[float]:
        return transform_direction_map(row).get(component)

    forward_series = [
        (
            "FFTM local FFT",
            profile_series_from_map(fftm_by_size, "forward local FFT", direction_value),
            (0.10, 0.32, 0.70),
        ),
        (
            "FFTM communication",
            profile_series_from_map(fftm_by_size, "forward communication", direction_value),
            (0.86, 0.34, 0.10),
        ),
        (
            "FFTS local FFT",
            profile_series_from_map(ffts_by_size, "forward local FFT", direction_value),
            (0.52, 0.22, 0.66),
        ),
    ]
    backward_series = [
        (
            "FFTM local FFT",
            profile_series_from_map(fftm_by_size, "backward local FFT", direction_value),
            (0.10, 0.32, 0.70),
        ),
        (
            "FFTM communication",
            profile_series_from_map(fftm_by_size, "backward communication", direction_value),
            (0.86, 0.34, 0.10),
        ),
        (
            "FFTS local FFT",
            profile_series_from_map(ffts_by_size, "backward local FFT", direction_value),
            (0.52, 0.22, 0.66),
        ),
    ]

    width = 535.0
    height = 292.0
    canvas = PdfCanvas(width, height)
    canvas.text(width / 2.0, height - 18.0, f"{dim}D forward/backward temporal profile vs problem size", args.title_font_size, align="center")
    canvas.text(
        width / 2.0,
        height - 32.0,
        "Direction-specific local FFT time is transform total minus direction-specific transpose time.",
        args.legend_font_size,
        color=(0.25, 0.25, 0.25),
        align="center",
    )
    legend_items = [(label, color) for label, _, color in forward_series]
    draw_legend_wrapped(canvas, legend_items, 58.0, height - 49.0, width - 116.0, args.legend_font_size)
    plot_y = 56.0
    plot_h = height - 145.0
    gap = 58.0
    plot_w = (width - 72.0 - gap - 18.0) / 2.0
    left_rect = (58.0, plot_y, plot_w, plot_h)
    right_rect = (58.0 + plot_w + gap, plot_y, plot_w, plot_h)
    draw_line_panel(
        canvas,
        left_rect,
        "Forward transform",
        x_values,
        forward_series,
        "Accumulated time [ms]",
        f"N for N^{dim}",
        args,
        log_y=True,
    )
    draw_line_panel(
        canvas,
        right_rect,
        "Backward transform",
        x_values,
        backward_series,
        "Accumulated time [ms]",
        f"N for N^{dim}",
        args,
        log_y=True,
    )
    path = fig_dir / f"fig_{dim}d_temporal_forward_backward_vs_size.pdf"
    canvas.save(path)
    return path


def memory_value(row: ResultRow, component: str) -> Optional[float]:
    if component == "FFTM measured device peak":
        return row.used_device_peak_max_mib
    if component == "FFTM tracked device":
        entry = row.tracked_memory_entries.get("tracked_device_total")
        if entry:
            return entry_float(entry, "peak_max_mib")
        return row.tracked_device_peak_max_mib
    if component == "FFTM internal device":
        entry = row.tracked_memory_entries.get("internal_device")
        if entry:
            return entry_float(entry, "peak_max_mib")
        return row.internal_device_peak_max_mib
    if component == "FFTM external/test device":
        entry = row.tracked_memory_entries.get("external_test_owned")
        if entry:
            return entry_float(entry, "peak_max_mib")
        return row.external_device_peak_max_mib
    if component == "FFTM host pinned":
        entry = row.tracked_memory_entries.get("host_pinned_internal")
        if entry:
            return entry_float(entry, "peak_max_mib")
        return row.host_pinned_peak_max_mib
    if component == "FFTS measured device peak":
        return row.used_device_peak_max_mib
    if component == "FFTS tracked device":
        return row.tracked_device_peak_max_mib
    return None


def compact_strategy_label(strategy: str) -> str:
    return {
        "slab-pencil": "slab-p",
        "pencil-slab": "p-slab",
        "pencil-pencil": "p-pencil",
        "slab-slab": "slab-s",
    }.get(strategy, strategy)


def memory_components_by_strategy_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    benchmark = [r for r in rows if r.case_name == "benchmark" and r.suite == "fftm" and r.dim == dim and r.ok]
    if not benchmark:
        return None
    categories: List[str] = []
    best_rows: List[ResultRow] = []
    for strategy in STRATEGY_ORDER.get(dim, sorted(set(r.strategy for r in benchmark))):
        for transport in ["cuda_aware", "non_cuda_aware"]:
            best = best_by_throughput(r for r in benchmark if r.strategy == strategy and r.transport == transport)
            if best is None:
                continue
            categories.append(f"{compact_strategy_label(strategy)} {('CA' if transport == 'cuda_aware' else 'NCA')}")
            best_rows.append(best)
    if not best_rows:
        return None

    components = [
        ("measured device peak", "FFTM measured device peak", (0.10, 0.32, 0.70)),
        ("internal device", "FFTM internal device", (0.86, 0.34, 0.10)),
        ("external/test device", "FFTM external/test device", (0.12, 0.55, 0.20)),
        ("host pinned", "FFTM host pinned", (0.45, 0.45, 0.45)),
    ]
    series = []
    for label, component, color in components:
        values = [memory_value(row, component) for row in best_rows]
        if any(value is not None and value > 0.0 for value in values):
            series.append((label, values, color))
    if not series:
        return None
    return bar_chart_pdf(
        fig_dir / f"fig_{dim}d_memory_components_by_strategy.pdf",
        f"{dim}D FFTM memory components by strategy",
        categories,
        series,
        "Peak memory [MiB] per process/GPU",
        args,
        value_digits=0,
        note="Best successful mode selected. Abbrev.: slab-p=slab-pencil, p-slab=pencil-slab, p-pencil=pencil-pencil.",
    )


def memory_breakdown_vs_size_pdf(rows: List[ResultRow], dim: int, fig_dir: Path, args: argparse.Namespace) -> Optional[Path]:
    fftm_by_size = rows_by_size_best(rows, dim=dim, suite="fftm", transport="cuda_aware", mode="p2p-waitany")
    ffts_by_size = rows_by_size_best(rows, dim=dim, suite="ffts", transport="single_gpu")
    if not fftm_by_size and not ffts_by_size:
        return None
    x_values = sorted(set(fftm_by_size.keys()) | set(ffts_by_size.keys()))
    series = [
        (
            "FFTM measured peak",
            profile_series_from_map(fftm_by_size, "FFTM measured device peak", memory_value),
            (0.10, 0.32, 0.70),
        ),
        (
            "FFTM internal",
            profile_series_from_map(fftm_by_size, "FFTM internal device", memory_value),
            (0.86, 0.34, 0.10),
        ),
        (
            "FFTM external/test",
            profile_series_from_map(fftm_by_size, "FFTM external/test device", memory_value),
            (0.12, 0.55, 0.20),
        ),
        (
            "FFTS measured peak",
            profile_series_from_map(ffts_by_size, "FFTS measured device peak", memory_value),
            (0.56, 0.20, 0.66),
        ),
    ]
    return line_chart_pdf(
        fig_dir / f"fig_{dim}d_memory_breakdown_vs_size.pdf",
        f"{dim}D memory profile breakdown vs problem size",
        "Best CUDA-aware FFTM p2p-waitany strategy at each N; MiB values are peak max per GPU/process.",
        x_values,
        series,
        "Memory [MiB]",
        f"N for N^{dim}",
        args,
        log_y=False,
    )


def write_manifest(out_dir: Path, generated: Sequence[Path], compile_results: Dict[str, str]) -> Path:
    manifest = {
        "generated_files": [str(p) for p in generated],
        "compile_results": compile_results,
    }
    path = out_dir / "analysis_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    data_dir = args.data_directory.resolve()
    out_dir = (args.output_directory or (data_dir / "analysis")).resolve()
    dirs = ensure_dirs(out_dir)
    cleanup_stale_figure_sources(dirs["figures"])

    rows = parse_rows(data_dir, phases=("measure",))
    sweep_rows = parse_rows(data_dir, phases=("probe", "measure"))
    hardware = read_json(data_dir / "hardware.json", {})
    summary = read_json(data_dir / "summary.json", {})

    generated: List[Path] = []
    generated.extend(write_csv_outputs(rows, dirs["csv"]))
    generated.append(overview_table(data_dir, rows, hardware, summary, dirs["tables"], args.table_font_size))
    generated.append(best_configs_table(rows, dirs["tables"], args.table_font_size))
    generated.append(mode_transport_table(rows, dirs["tables"], args.table_font_size))
    generated.append(correctness_table(rows, dirs["tables"], args.table_font_size))
    generated.append(failures_table(rows, dirs["tables"], args.table_font_size))
    generated.append(temporal_profile_table(rows, dirs["tables"], args.table_font_size))
    generated.append(memory_profile_table(rows, dirs["tables"], args.table_font_size))
    for dim in [3, 4]:
        fig = egger_style_runtime_pdf(sweep_rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = temporal_breakdown_vs_size_pdf(sweep_rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = temporal_forward_backward_vs_size_pdf(sweep_rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = memory_breakdown_vs_size_pdf(sweep_rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = throughput_figure_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = gpu_scaling_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = gflops_gpu_scaling_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = runtime_absolute_heatmap_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = runtime_relative_heatmap_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = memory_peak_heatmap_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
        fig = memory_components_by_strategy_pdf(rows, dim, dirs["figures"], args)
        if fig:
            generated.append(fig)
    for fig in [
        ffts_fftm_relative_pdf(rows, dirs["figures"], args),
        transport_speedup_pdf(rows, dirs["figures"], args),
        memory_footprint_pdf(rows, dirs["figures"], args),
    ]:
        if fig:
            generated.append(fig)

    compile_results: Dict[str, str] = {}
    if args.compile_latex:
        for tex_path in [p for p in generated if p.suffix == ".tex"]:
            ok, message = compile_tex(tex_path, args.pdflatex)
            compile_results[str(tex_path)] = "ok" if ok else message
            if not ok:
                print(f"WARNING: {message}", file=sys.stderr)

    manifest = write_manifest(out_dir, generated, compile_results)
    print(f"Wrote analysis to {out_dir}")
    print(f"Wrote manifest to {manifest}")
    if args.compile_latex:
        failed = [msg for msg in compile_results.values() if msg != "ok"]
        if failed:
            print(f"WARNING: {len(failed)} LaTeX document(s) failed to compile", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
