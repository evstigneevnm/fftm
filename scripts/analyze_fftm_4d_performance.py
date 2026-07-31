#!/usr/bin/env python3
"""Compact 4D FFTM benchmark/stage analyzer.

The script consumes copied cluster result directories.  It intentionally uses
only the Python standard library so it can run on the login node or locally.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FFTM 4D benchmark and stage-timer CSV files.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Result directory to analyze.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-dir>/analysis/4d.",
    )
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def effective_gflops(sizes: Sequence[int], wall_ms: float) -> Optional[float]:
    if wall_ms <= 0 or not sizes:
        return None
    points = math.prod(sizes)
    if points <= 1:
        return None
    flops = 10.0 * float(points) * math.log2(float(points))
    return (flops / (wall_ms / 1000.0)) / 1.0e9


def throughput_gpoints(sizes: Sequence[int], wall_ms: float) -> Optional[float]:
    if wall_ms <= 0 or not sizes:
        return None
    return (float(math.prod(sizes)) / (wall_ms / 1000.0)) / 1.0e9


def fmt_float(value: Optional[float]) -> str:
    parsed = as_float(value)
    if parsed is None:
        return ""
    return f"{parsed:.9g}"


def read_dicts(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        yield from csv.DictReader(fh)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
            count += 1
    return count


def benchmark_key(row: Dict[str, object]) -> Tuple[object, ...]:
    return (
        row.get("source_run", ""),
        row.get("num_gpus", ""),
        row.get("strategy", ""),
        row.get("mode", ""),
        row.get("p1", ""),
        row.get("p2", ""),
        row.get("p3", ""),
        row.get("nx", ""),
        row.get("ny", ""),
        row.get("nz", ""),
        row.get("nw", ""),
        row.get("pencil_same_zw_peer_paired", ""),
        row.get("pencil_same_zw_native_layout", ""),
        row.get("pencil_degenerate_xw_slab_path", ""),
        row.get("pencil_degenerate_local_transposes", ""),
        row.get("pencil_degenerate_same_xw_native", ""),
        row.get("pencil_degenerate_wz_sliced_z_fft", ""),
        row.get("native_xw_direct_layout", ""),
        row.get("native_xw_chunked_transport", ""),
        row.get("native_xw_chunk_mib", ""),
        row.get("native_xw_chunk_window", ""),
        row.get("native_xw_compact_staging", ""),
        row.get("slab_native_work_area_alias", ""),
        row.get("slab_native_work_area_alias_effective", ""),
        row.get("slab_native_wz_communication_layout", ""),
        row.get("slab_native_wz_plan_concurrency", ""),
        row.get("slab_native_wz_ready_pipeline", ""),
    )


def load_benchmark_rows(data_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(data_dir.rglob("benchmark_fftm_4d.csv")):
        for raw in read_dicts(path):
            num_gpus = as_int(raw.get("num_gpus"))
            avg_ms = as_float(raw.get("avg_wall_ms"))
            nx = as_int(raw.get("nx"))
            ny = as_int(raw.get("ny"))
            nz = as_int(raw.get("nz"))
            nw = as_int(raw.get("nw"))
            if num_gpus is None or avg_ms is None or nx is None or ny is None or nz is None or nw is None:
                continue
            sizes = (nx, ny, nz, nw)
            row: Dict[str, object] = dict(raw)
            row["source_csv"] = rel(path, data_dir)
            row["source_run"] = rel(path.parent, data_dir)
            row["num_gpus"] = num_gpus
            row["p1"] = as_int(raw.get("p1")) or 0
            row["p2"] = as_int(raw.get("p2")) or 0
            row["p3"] = as_int(raw.get("p3")) or 0
            row["nx"], row["ny"], row["nz"], row["nw"] = sizes
            row["avg_wall_ms"] = avg_ms
            row["stddev_wall_ms"] = as_float(raw.get("stddev_wall_ms"))
            row["max_l2_diff"] = as_float(raw.get("max_l2_diff"))
            row["size_4d"] = "x".join(str(v) for v in sizes)
            row["grid_4d"] = f"{row['p1']}x{row['p2']}x{row['p3']}"
            row["throughput_gpoints_s"] = throughput_gpoints(sizes, avg_ms)
            row["effective_gflops_s"] = effective_gflops(sizes, avg_ms)
            row.setdefault("native_stage_timers", "")
            row.setdefault("slab_native_xw", "")
            row.setdefault("slab_native_xw_batched_peer_kernels", "")
            row.setdefault("slab_native_xw_tensor_coalesced_kernels", "")
            row.setdefault("slab_native_xw_vector4_kernels", "")
            row.setdefault("slab_native_xw_tiled_kernels", "")
            row.setdefault("slab_native_xw_layout_stage", "")
            row.setdefault("slab_native_xw_native_spectral_layout", "")
            row.setdefault("pencil_same_zw_peer_paired", "")
            row.setdefault("pencil_same_zw_native_layout", "")
            row.setdefault("pencil_degenerate_xw_slab_path", "")
            row.setdefault("pencil_degenerate_local_transposes", "")
            row.setdefault("pencil_degenerate_same_xw_native", "")
            row.setdefault("pencil_degenerate_wz_sliced_z_fft", "")
            row.setdefault("native_xw_direct_layout", "")
            row.setdefault("native_xw_chunked_transport", "")
            row.setdefault("native_xw_chunk_mib", "")
            row.setdefault("native_xw_chunk_window", "")
            row.setdefault("native_xw_compact_staging", "")
            row.setdefault("slab_native_work_area_alias", "")
            row.setdefault("slab_native_work_area_alias_effective", "")
            row.setdefault("slab_native_wz_communication_layout", "")
            row.setdefault("slab_native_wz_plan_concurrency", "")
            row.setdefault("slab_native_wz_ready_pipeline", "")
            row.setdefault("fft_work_bytes", "")
            row.setdefault("transpose_work_bytes", "")
            row.setdefault("same_xw_work_bytes", "")
            row.setdefault("sliced_z_work_bytes", "")
            row.setdefault("stage1_alias_bytes", "")
            row.setdefault("shared_work_bytes", "")
            rows.append(row)
    return rows


def load_stage_summary(data_dir: Path) -> List[Dict[str, object]]:
    stage_iter_rank_sum: Dict[Tuple[object, ...], float] = defaultdict(float)
    wall_iter_rank: Dict[Tuple[object, ...], List[float]] = defaultdict(list)

    for path in sorted(data_dir.rglob("fftm_4d_stage_times_r*.csv")):
        run = rel(path.parent, data_dir)
        for raw in read_dicts(path):
            stage_ms = as_float(raw.get("stage_ms"))
            wall_ms = as_float(raw.get("wall_ms"))
            iteration = as_int(raw.get("iteration"))
            rank = as_int(raw.get("rank"))
            if stage_ms is None or wall_ms is None or iteration is None or rank is None:
                continue
            case = (
                run,
                as_int(raw.get("num_gpus")) or 0,
                raw.get("strategy", ""),
                raw.get("mode", ""),
                as_int(raw.get("p1")) or 0,
                as_int(raw.get("p2")) or 0,
                as_int(raw.get("p3")) or 0,
                as_int(raw.get("nx")) or 0,
                as_int(raw.get("ny")) or 0,
                as_int(raw.get("nz")) or 0,
                as_int(raw.get("nw")) or 0,
                as_int(raw.get("slab_native_xw")) or 0,
                as_int(raw.get("slab_native_xw_batched_peer_kernels")) or 0,
                as_int(raw.get("slab_native_xw_tensor_coalesced_kernels")) or 0,
                as_int(raw.get("slab_native_xw_vector4_kernels")) or 0,
                as_int(raw.get("slab_native_xw_tiled_kernels")) or 0,
                as_int(raw.get("slab_native_xw_layout_stage")) or 0,
                as_int(raw.get("slab_native_xw_native_spectral_layout")) or 0,
                as_int(raw.get("pencil_same_zw_peer_paired")) or 0,
                as_int(raw.get("pencil_same_zw_native_layout")) or 0,
                as_int(raw.get("pencil_degenerate_xw_slab_path")) or 0,
                as_int(raw.get("pencil_degenerate_local_transposes")) or 0,
                as_int(raw.get("pencil_degenerate_same_xw_native")) or 0,
                as_int(raw.get("pencil_degenerate_wz_sliced_z_fft")) or 0,
                as_int(raw.get("native_xw_direct_layout")) or 0,
                as_int(raw.get("native_xw_chunked_transport")) or 0,
                as_int(raw.get("native_xw_chunk_mib")) or 0,
                as_int(raw.get("native_xw_chunk_window")) or 0,
                as_int(raw.get("native_xw_compact_staging")) or 0,
                as_int(raw.get("slab_native_work_area_alias")) or 0,
                as_int(raw.get("slab_native_work_area_alias_effective")) or 0,
                as_int(raw.get("slab_native_wz_communication_layout")) or 0,
                as_int(raw.get("slab_native_wz_plan_concurrency")) or 1,
                as_int(raw.get("slab_native_wz_ready_pipeline")) or 0,
            )
            stage = raw.get("stage", "")
            stage_iter_rank_sum[case + (stage, iteration, rank)] += stage_ms
            wall_iter_rank[case + (iteration,)].append(wall_ms)

    wall_by_case: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for key, values in wall_iter_rank.items():
        case = key[:-1]
        wall_by_case[case].append(max(values))

    stage_by_case: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    stage_iter_values: Dict[Tuple[object, ...], List[float]] = defaultdict(list)
    for key, value in stage_iter_rank_sum.items():
        case_stage_iteration = key[:-1]
        stage_iter_values[case_stage_iteration].append(value)
    for key, values in stage_iter_values.items():
        case_stage = key[:-1]
        stage_by_case[case_stage].append(max(values))

    rows: List[Dict[str, object]] = []
    for case_stage, values in sorted(stage_by_case.items()):
        case = case_stage[:-1]
        stage = case_stage[-1]
        wall_values = wall_by_case.get(case, [])
        avg_stage = sum(values) / len(values)
        max_stage = max(values)
        avg_wall = sum(wall_values) / len(wall_values) if wall_values else None
        fraction = avg_stage / avg_wall if avg_wall else None
        (
            source_run,
            num_gpus,
            strategy,
            mode,
            p1,
            p2,
            p3,
            nx,
            ny,
            nz,
            nw,
            slab_native_xw,
            slab_native_xw_batched_peer_kernels,
            slab_native_xw_tensor_coalesced_kernels,
            slab_native_xw_vector4_kernels,
            slab_native_xw_tiled_kernels,
            slab_native_xw_layout_stage,
            slab_native_xw_native_spectral_layout,
            pencil_same_zw_peer_paired,
            pencil_same_zw_native_layout,
            pencil_degenerate_xw_slab_path,
            pencil_degenerate_local_transposes,
            pencil_degenerate_same_xw_native,
            pencil_degenerate_wz_sliced_z_fft,
            native_xw_direct_layout,
            native_xw_chunked_transport,
            native_xw_chunk_mib,
            native_xw_chunk_window,
            native_xw_compact_staging,
            slab_native_work_area_alias,
            slab_native_work_area_alias_effective,
            slab_native_wz_communication_layout,
            slab_native_wz_plan_concurrency,
            slab_native_wz_ready_pipeline,
        ) = case
        rows.append(
            {
                "source_run": source_run,
                "num_gpus": num_gpus,
                "strategy": strategy,
                "mode": mode,
                "grid_4d": f"{p1}x{p2}x{p3}",
                "size_4d": f"{nx}x{ny}x{nz}x{nw}",
                "slab_native_xw": slab_native_xw,
                "slab_native_xw_batched_peer_kernels": slab_native_xw_batched_peer_kernels,
                "slab_native_xw_tensor_coalesced_kernels": slab_native_xw_tensor_coalesced_kernels,
                "slab_native_xw_vector4_kernels": slab_native_xw_vector4_kernels,
                "slab_native_xw_tiled_kernels": slab_native_xw_tiled_kernels,
                "slab_native_xw_layout_stage": slab_native_xw_layout_stage,
                "slab_native_xw_native_spectral_layout": slab_native_xw_native_spectral_layout,
                "pencil_same_zw_peer_paired": pencil_same_zw_peer_paired,
                "pencil_same_zw_native_layout": pencil_same_zw_native_layout,
                "pencil_degenerate_xw_slab_path": pencil_degenerate_xw_slab_path,
                "pencil_degenerate_local_transposes": pencil_degenerate_local_transposes,
                "pencil_degenerate_same_xw_native": pencil_degenerate_same_xw_native,
                "pencil_degenerate_wz_sliced_z_fft": pencil_degenerate_wz_sliced_z_fft,
                "native_xw_direct_layout": native_xw_direct_layout,
                "native_xw_chunked_transport": native_xw_chunked_transport,
                "native_xw_chunk_mib": native_xw_chunk_mib,
                "native_xw_chunk_window": native_xw_chunk_window,
                "native_xw_compact_staging": native_xw_compact_staging,
                "slab_native_work_area_alias": slab_native_work_area_alias,
                "slab_native_work_area_alias_effective": slab_native_work_area_alias_effective,
                "slab_native_wz_communication_layout": slab_native_wz_communication_layout,
                "slab_native_wz_plan_concurrency": slab_native_wz_plan_concurrency,
                "slab_native_wz_ready_pipeline": slab_native_wz_ready_pipeline,
                "stage": stage,
                "iterations": len(values),
                "avg_stage_ms": avg_stage,
                "max_stage_ms": max_stage,
                "avg_wall_ms_from_stage_rows": avg_wall,
                "stage_fraction_of_wall": fraction,
            }
        )
    return rows


def best_rows(rows: Iterable[Dict[str, object]], key_fields: Sequence[str]) -> List[Dict[str, object]]:
    best: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        gflops = as_float(row.get("effective_gflops_s"))
        if gflops is None:
            continue
        key = tuple(row.get(field, "") for field in key_fields)
        old = best.get(key)
        old_gflops = as_float(old.get("effective_gflops_s")) if old is not None else None
        if old is None or old_gflops is None or gflops > old_gflops:
            best[key] = row
    return [dict(best[key]) for key in sorted(best)]


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else data_dir / "analysis" / "4d"

    benchmark_rows = load_benchmark_rows(data_dir)
    stage_rows = load_stage_summary(data_dir)

    bench_fields = [
        "source_run",
        "source_csv",
        "benchmark",
        "num_gpus",
        "strategy",
        "mode",
        "grid_4d",
        "size_4d",
        "fft_exec_no_sync",
        "native_stage_timers",
        "slab_native_xw",
        "slab_native_xw_batched_peer_kernels",
        "slab_native_xw_tensor_coalesced_kernels",
        "slab_native_xw_vector4_kernels",
        "slab_native_xw_tiled_kernels",
        "slab_native_xw_layout_stage",
        "slab_native_xw_native_spectral_layout",
        "pencil_same_zw_peer_paired",
        "pencil_same_zw_native_layout",
        "pencil_degenerate_xw_slab_path",
        "pencil_degenerate_local_transposes",
        "pencil_degenerate_same_xw_native",
        "pencil_degenerate_wz_sliced_z_fft",
        "native_xw_direct_layout",
        "native_xw_chunked_transport",
        "native_xw_chunk_mib",
        "native_xw_chunk_window",
        "native_xw_compact_staging",
        "slab_native_work_area_alias",
        "slab_native_work_area_alias_effective",
        "slab_native_wz_communication_layout",
        "slab_native_wz_plan_concurrency",
        "slab_native_wz_ready_pipeline",
        "fft_work_bytes",
        "transpose_work_bytes",
        "same_xw_work_bytes",
        "sliced_z_work_bytes",
        "stage1_alias_bytes",
        "shared_work_bytes",
        "times",
        "warmup",
        "avg_wall_ms",
        "stddev_wall_ms",
        "max_l2_diff",
        "throughput_gpoints_s",
        "effective_gflops_s",
        "directory",
    ]
    best_fields = bench_fields
    stage_fields = [
        "source_run",
        "num_gpus",
        "strategy",
        "mode",
        "grid_4d",
        "size_4d",
        "slab_native_xw",
        "slab_native_xw_batched_peer_kernels",
        "slab_native_xw_tensor_coalesced_kernels",
        "slab_native_xw_vector4_kernels",
        "slab_native_xw_tiled_kernels",
        "slab_native_xw_layout_stage",
        "slab_native_xw_native_spectral_layout",
        "pencil_same_zw_peer_paired",
        "pencil_same_zw_native_layout",
        "pencil_degenerate_xw_slab_path",
        "pencil_degenerate_local_transposes",
        "pencil_degenerate_same_xw_native",
        "pencil_degenerate_wz_sliced_z_fft",
        "native_xw_direct_layout",
        "native_xw_chunked_transport",
        "native_xw_chunk_mib",
        "native_xw_chunk_window",
        "native_xw_compact_staging",
        "slab_native_work_area_alias",
        "slab_native_work_area_alias_effective",
        "slab_native_wz_communication_layout",
        "slab_native_wz_plan_concurrency",
        "slab_native_wz_ready_pipeline",
        "stage",
        "iterations",
        "avg_stage_ms",
        "max_stage_ms",
        "avg_wall_ms_from_stage_rows",
        "stage_fraction_of_wall",
    ]

    for row in benchmark_rows:
        for field in ["avg_wall_ms", "stddev_wall_ms", "max_l2_diff", "throughput_gpoints_s", "effective_gflops_s"]:
            row[field] = fmt_float(row.get(field))
    display_benchmark_rows = benchmark_rows

    raw_benchmark_rows = load_benchmark_rows(data_dir)
    best_by_gpu = best_rows(raw_benchmark_rows, ["num_gpus"])
    best_by_gpu_strategy = best_rows(raw_benchmark_rows, ["num_gpus", "strategy"])
    for rows in (best_by_gpu, best_by_gpu_strategy):
        for row in rows:
            for field in ["avg_wall_ms", "stddev_wall_ms", "max_l2_diff", "throughput_gpoints_s", "effective_gflops_s"]:
                row[field] = fmt_float(row.get(field))

    for row in stage_rows:
        for field in ["avg_stage_ms", "max_stage_ms", "avg_wall_ms_from_stage_rows", "stage_fraction_of_wall"]:
            row[field] = fmt_float(row.get(field))

    n_bench = write_csv(output_dir / "benchmark_4d_cases.csv", bench_fields, display_benchmark_rows)
    n_best = write_csv(output_dir / "best_by_gpu.csv", best_fields, best_by_gpu)
    n_best_strategy = write_csv(output_dir / "best_by_gpu_strategy.csv", best_fields, best_by_gpu_strategy)
    n_stage = write_csv(output_dir / "stage_summary.csv", stage_fields, stage_rows)

    report = output_dir / "summary.md"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("# FFTM 4D Compact Analysis\n\n")
        fh.write(f"- Data directory: `{data_dir}`\n")
        fh.write(f"- Benchmark rows: {n_bench}\n")
        fh.write(f"- Best-by-GPU rows: {n_best}\n")
        fh.write(f"- Best-by-GPU-strategy rows: {n_best_strategy}\n")
        fh.write(f"- Stage summary rows: {n_stage}\n\n")
        if not stage_rows:
            fh.write("No 4D stage timer CSV files were found.\n")

    print(f"Wrote {output_dir / 'benchmark_4d_cases.csv'} ({n_bench} rows)")
    print(f"Wrote {output_dir / 'best_by_gpu.csv'} ({n_best} rows)")
    print(f"Wrote {output_dir / 'best_by_gpu_strategy.csv'} ({n_best_strategy} rows)")
    print(f"Wrote {output_dir / 'stage_summary.csv'} ({n_stage} rows)")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
