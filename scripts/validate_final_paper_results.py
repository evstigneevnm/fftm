#!/usr/bin/env python3
"""Validate the final FFTM production benchmark directories.

The checks intentionally use only the Python standard library so they can run
on the cluster login node.  Performance limits are regression guards with
substantial headroom; they are not the values reported in the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PRODUCTION_LIMITS_3D = {6: 450.0, 7: 400.0, 8: 350.0}
PRODUCTION_LIMITS_4D = {6: 400.0, 7: 400.0, 8: 350.0}
MULTINODE_LIMITS_3D = {16: 650.0}
MULTINODE_LIMITS_4D = {16: 650.0}


class Validation:
    def __init__(self, target: str, directory: Path) -> None:
        self.target = target
        self.directory = directory
        self.checks: List[str] = []
        self.failures: List[str] = []
        self.warnings: List[str] = []
        self.measurements: List[Dict[str, object]] = []

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.checks.append(message)
        else:
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            self.warnings.append(message)

    def report(self) -> Dict[str, object]:
        return {
            "target": self.target,
            "directory": str(self.directory),
            "passed": not self.failures,
            "checks": self.checks,
            "failures": self.failures,
            "warnings": self.warnings,
            "measurements": self.measurements,
        }


def read_csv(path: Path, validation: Validation) -> List[Dict[str, str]]:
    validation.require(path.is_file(), f"result CSV exists: {path}")
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_int(row: Mapping[str, str], field: str) -> int:
    return int(row[field])


def as_float(row: Mapping[str, str], field: str) -> float:
    return float(row[field])


def require_fields(
    validation: Validation,
    row: Mapping[str, str],
    expected: Mapping[str, str],
    label: str,
) -> None:
    for field, wanted in expected.items():
        validation.require(
            field in row and row[field] == wanted,
            f"{label}: {field}={wanted}",
        )


def load_summary(directory: Path, validation: Validation, affinity: str, expected_runs: int) -> None:
    path = directory / "summary.json"
    validation.require(path.is_file(), f"summary exists: {path}")
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    validation.require(data.get("failed_measurement_runs") == 0, "no benchmark run failed")
    validation.require(data.get("planned_runs") == expected_runs, f"planned run count is {expected_runs}")
    validation.require(
        data.get("num_measurement_specs") == expected_runs,
        f"measurement specification count is {expected_runs}",
    )
    validation.require(data.get("mpi_rank_affinity_mode") == affinity, f"MPI affinity mode is {affinity}")


def validate_telemetry(directory: Path, validation: Validation, expected_ranks: int) -> None:
    telemetry = sorted((directory / "cpp_csv").glob("gpu_telemetry_r*.csv"))
    validation.require(
        len(telemetry) >= expected_ranks,
        f"GPU telemetry contains at least {expected_ranks} rank files",
    )


def validate_hca_affinity(directory: Path, validation: Validation, expected_runs: int) -> None:
    logs = sorted((directory / "raw").glob("*.log"))
    validation.require(len(logs) == expected_runs, f"HCA run log count is {expected_runs}")
    marker = re.compile(r"\[FFTM_MPI_AFFINITY\].*\brank=(\d+)\b.*\bmode=hca\b.*\bucx_net_devices=mlx5_")
    expected_ranks = set(range(16))
    for path in logs:
        ranks = set()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = marker.search(line)
            if match:
                ranks.add(int(match.group(1)))
        validation.require(ranks == expected_ranks, f"{path.name}: all 16 HCA-pinned ranks are recorded")


def validate_numeric_row(
    validation: Validation,
    row: Mapping[str, str],
    label: str,
    limit_ms: float,
    min_warmup: int,
    max_l2: float,
) -> None:
    try:
        wall_ms = as_float(row, "avg_wall_ms")
        stddev_ms = as_float(row, "stddev_wall_ms")
        l2_error = as_float(row, "max_l2_diff")
        times = as_int(row, "times")
        warmup = as_int(row, "warmup")
    except (KeyError, TypeError, ValueError) as exc:
        validation.failures.append(f"{label}: malformed numerical fields ({exc})")
        return

    validation.require(math.isfinite(wall_ms) and wall_ms > 0.0, f"{label}: finite positive wall time")
    validation.require(wall_ms <= limit_ms, f"{label}: wall time <= {limit_ms:.0f} ms")
    validation.require(math.isfinite(l2_error) and l2_error <= max_l2, f"{label}: L2 error <= {max_l2:g}")
    validation.require(times >= 10, f"{label}: at least 10 measured iterations")
    validation.require(warmup >= min_warmup, f"{label}: at least {min_warmup} warmup iterations")
    validation.require(row.get("wall_time_scope") == "mpi-rank-max", f"{label}: MPI-rank-max wall timing")
    coefficient = stddev_ms / wall_ms if wall_ms > 0.0 else math.inf
    validation.warn(coefficient <= 0.20, f"{label}: high timing variation ({coefficient:.1%})")
    validation.measurements.append(
        {
            "case": label,
            "avg_wall_ms": wall_ms,
            "stddev_wall_ms": stddev_ms,
            "coefficient_of_variation": coefficient,
            "max_l2_diff": l2_error,
        }
    )


def rows_by_gpu(
    rows: Sequence[Mapping[str, str]], expected_gpus: Iterable[int], validation: Validation, dimension: str
) -> Dict[int, Mapping[str, str]]:
    result: Dict[int, Mapping[str, str]] = {}
    for row in rows:
        try:
            gpu = as_int(row, "num_gpus")
        except (KeyError, ValueError):
            validation.failures.append(f"{dimension}: row has an invalid num_gpus field")
            continue
        validation.require(gpu not in result, f"{dimension}: one row for {gpu} GPUs")
        result[gpu] = row
    expected = set(expected_gpus)
    validation.require(set(result) == expected, f"{dimension}: GPU set is {sorted(expected)}")
    return result


def validate_3d(
    validation: Validation,
    rows: Sequence[Mapping[str, str]],
    grids: Mapping[int, Sequence[int]],
    limits: Mapping[int, float],
    warmup: int,
) -> None:
    by_gpu = rows_by_gpu(rows, grids, validation, "3D")
    expected_switches = {
        "strategy": "pencil-pencil",
        "mode": "p2p-waitany",
        "pencil_pipeline": "reference-parity",
        "large_count_p2p_transport": "hindexed",
        "fft_exec_no_sync": "0",
        "persistent_p2p": "0",
        "ready_p2p_send": "0",
        "deferred_send_completion": "0",
        "native_opt0_default_z_layout": "1",
        "native_opt0_reference_y_buffer_topology": "1",
        "native_opt0_tight_y_plan_sequence": "1",
        "native_opt0_shared_y_plan_handles": "1",
        "native_opt0_y_group_device_sync": "1",
        "native_opt0_y_no_sync_exec": "1",
        "native_opt0_raw_y_plan_array_executor": "1",
    }
    for gpu, grid in grids.items():
        row = by_gpu.get(gpu)
        if row is None:
            continue
        label = f"3D {gpu}G {grid[0]}x{grid[1]}"
        require_fields(validation, row, expected_switches, label)
        require_fields(
            validation,
            row,
            {
                "p1": str(grid[0]),
                "p2": str(grid[1]),
                "p3": "1",
                "nx": "2048",
                "ny": "2048",
                "nz": "2048",
            },
            label,
        )
        validate_numeric_row(validation, row, label, limits[gpu], warmup, 1.0e-11)


def validate_4d(
    validation: Validation,
    rows: Sequence[Mapping[str, str]],
    expected_gpus: Sequence[int],
    limits: Mapping[int, float],
    warmup: int,
) -> None:
    by_gpu = rows_by_gpu(rows, expected_gpus, validation, "4D")
    expected_switches = {
        "strategy": "slab-slab",
        "mode": "p2p-waitany",
        "fft_exec_no_sync": "0",
        "native_stage_timers": "0",
        "slab_native_xw": "1",
        "native_xw_direct_layout": "1",
        "native_xw_chunked_transport": "1",
        "native_xw_chunk_mib": "512",
        "native_xw_chunk_window": "1",
        "native_xw_compact_staging": "1",
        "slab_native_work_area_alias": "1",
        "slab_native_work_area_alias_effective": "1",
        "slab_native_wz_communication_layout": "1",
        "slab_native_wz_plan_concurrency": "4",
        "slab_native_wz_ready_pipeline": "1",
        "slab_native_xw_native_spectral_layout": "1",
    }
    for gpu in expected_gpus:
        row = by_gpu.get(gpu)
        if row is None:
            continue
        label = f"4D {gpu}G slab-native"
        require_fields(validation, row, expected_switches, label)
        require_fields(
            validation,
            row,
            {
                "p1": "1",
                "p2": str(gpu),
                "p3": "1",
                "nx": "320",
                "ny": "320",
                "nz": "320",
                "nw": "320",
            },
            label,
        )
        validate_numeric_row(validation, row, label, limits[gpu], warmup, 1.0e-11)


def validate_benchmarks(target: str, directory: Path) -> Validation:
    validation = Validation(target, directory)
    if target == "production":
        grids = {6: (2, 3), 7: (7, 1), 8: (4, 2)}
        limits_3d = PRODUCTION_LIMITS_3D
        limits_4d = PRODUCTION_LIMITS_4D
        affinity = "auto"
        warmup = 3
    else:
        grids = {16: (4, 4)}
        limits_3d = MULTINODE_LIMITS_3D
        limits_4d = MULTINODE_LIMITS_4D
        affinity = "hca"
        warmup = 5

    rows_3d = read_csv(directory / "cpp_csv" / "benchmark_fftm_3d.csv", validation)
    rows_4d = read_csv(directory / "cpp_csv" / "benchmark_fftm_4d.csv", validation)
    load_summary(directory, validation, affinity, len(grids) * 2)
    validate_telemetry(directory, validation, max(grids))
    if target == "multinode":
        validate_hca_affinity(directory, validation, len(grids) * 2)
    validate_3d(validation, rows_3d, grids, limits_3d, warmup)
    validate_4d(validation, rows_4d, list(grids), limits_4d, warmup)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("production", "multinode"))
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    directory = args.directory.resolve()
    validation = validate_benchmarks(args.target, directory)
    report = validation.report()
    output = args.output or directory / "final_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Final {args.target} validation: {'PASS' if report['passed'] else 'FAIL'}")
    for measurement in validation.measurements:
        print(
            f"  {measurement['case']}: {measurement['avg_wall_ms']:.3f} ms, "
            f"CV={measurement['coefficient_of_variation']:.1%}, "
            f"L2={measurement['max_l2_diff']:.3e}"
        )
    for warning in validation.warnings:
        print(f"  WARNING: {warning}")
    for failure in validation.failures:
        print(f"  ERROR: {failure}", file=sys.stderr)
    print(f"Report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
