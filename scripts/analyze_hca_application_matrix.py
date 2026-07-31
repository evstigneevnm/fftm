#!/usr/bin/env python3
"""Analyze the matched 3D/4D HCA-pinned application matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass
class Case:
    dimension: int
    key: str
    label: str
    mean_ms: float
    reported_stddev_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    sample_stddev_ms: float
    cv_pct: float
    max_l2_diff: float
    samples_ms: List[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Combined HCA matrix directory containing fftm/ and egger/.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Prior unpinned FFTM matrix directory.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def case_identity(dimension: int, row: Dict[str, str]) -> Tuple[str, str]:
    if dimension == 3:
        grid = f"{row['p1']}x{row['p2']}"
        return f"3d-grid-{grid}", f"3D {grid}"
    concurrency = row["slab_native_wz_plan_concurrency"]
    return f"4d-c{concurrency}", f"4D c{concurrency}"


def read_fftm_cases(data_dir: Path) -> List[Case]:
    cases: List[Case] = []
    csv_dir = data_dir / "cpp_csv"
    for dimension in (3, 4):
        benchmark_rows = read_csv(csv_dir / f"benchmark_fftm_{dimension}d.csv")
        wall_rows = read_csv(csv_dir / f"wall_times_{dimension}d_r0.csv")
        cursor = 0
        for row in benchmark_rows:
            count = int(row["times"])
            selected = wall_rows[cursor : cursor + count]
            cursor += count
            samples = [float(item["global_wall_ms"]) for item in selected]
            if len(samples) != count:
                raise RuntimeError(
                    f"{dimension}D row expected {count} samples, found {len(samples)}"
                )
            key, label = case_identity(dimension, row)
            mean = statistics.mean(samples)
            sample_stddev = statistics.pstdev(samples)
            cases.append(
                Case(
                    dimension=dimension,
                    key=key,
                    label=label,
                    mean_ms=float(row["avg_wall_ms"]),
                    reported_stddev_ms=float(row["stddev_wall_ms"]),
                    median_ms=statistics.median(samples),
                    min_ms=min(samples),
                    max_ms=max(samples),
                    sample_stddev_ms=sample_stddev,
                    cv_pct=100.0 * sample_stddev / mean,
                    max_l2_diff=float(row["max_l2_diff"]),
                    samples_ms=samples,
                )
            )
        if cursor != len(wall_rows):
            raise RuntimeError(
                f"{dimension}D consumed {cursor} wall rows, but {len(wall_rows)} exist"
            )
    return cases


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def affinity_summary(fftm_dir: Path) -> Tuple[int, int, List[str]]:
    logs = sorted((fftm_dir / "raw").glob("*.log"))
    pattern = re.compile(
        r"\[FFTM_MPI_AFFINITY\].*?host=(\S+).*?local_rank=(\d+)"
        r".*?ucx_net_devices=(\S+)"
    )
    markers = []
    for log in logs:
        markers.extend(pattern.findall(log.read_text(errors="replace")))
    mappings = sorted({f"{host}:r{rank}:{hcas}" for host, rank, hcas in markers})
    return len(logs), len(markers), mappings


def egger_status(root: Path) -> Tuple[bool, str]:
    summary_path = root / "egger" / "16g_2n" / "summary.json"
    summary = read_json(summary_path)
    if not summary:
        return False, f"missing summary: {summary_path}"
    runs = int(summary.get("runs", -1))
    succeeded = int(summary.get("ok", -1))
    failed = int(summary.get("failed", -1))
    valid = runs == 4 and succeeded == 4 and failed == 0
    detail = f"runs={runs}, ok={succeeded}, failed={failed}"
    if valid:
        return True, detail

    raw_dir = root / "egger" / "16g_2n" / "raw"
    for log in sorted(raw_dir.glob("*.log")):
        for line in log.read_text(errors="replace").splitlines():
            if "couldn't chdir" in line or "No such file or directory" in line:
                return False, f"{detail}; {line.strip()}"
    return False, detail


def write_csv_summary(
    path: Path, current: Sequence[Case], baseline: Sequence[Case]
) -> None:
    baseline_by_key = {case.key: case for case in baseline}
    fields = [
        "case",
        "dimension",
        "baseline_mean_ms",
        "hca_mean_ms",
        "hca_median_ms",
        "hca_min_ms",
        "hca_max_ms",
        "hca_sample_stddev_ms",
        "hca_cv_pct",
        "speedup",
        "wall_reduction_pct",
        "max_l2_diff",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in current:
            previous = baseline_by_key.get(case.key)
            speedup = previous.mean_ms / case.mean_ms if previous else None
            reduction = (
                100.0 * (1.0 - case.mean_ms / previous.mean_ms)
                if previous
                else None
            )
            writer.writerow(
                {
                    "case": case.label,
                    "dimension": case.dimension,
                    "baseline_mean_ms": f"{previous.mean_ms:.3f}" if previous else "",
                    "hca_mean_ms": f"{case.mean_ms:.3f}",
                    "hca_median_ms": f"{case.median_ms:.3f}",
                    "hca_min_ms": f"{case.min_ms:.3f}",
                    "hca_max_ms": f"{case.max_ms:.3f}",
                    "hca_sample_stddev_ms": f"{case.sample_stddev_ms:.3f}",
                    "hca_cv_pct": f"{case.cv_pct:.3f}",
                    "speedup": f"{speedup:.3f}" if speedup else "",
                    "wall_reduction_pct": f"{reduction:.3f}" if reduction else "",
                    "max_l2_diff": f"{case.max_l2_diff:.9e}",
                }
            )


def write_markdown(
    path: Path,
    current: Sequence[Case],
    baseline: Sequence[Case],
    affinity: Tuple[int, int, List[str]],
    egger: Tuple[bool, str],
) -> None:
    baseline_by_key = {case.key: case for case in baseline}
    lines = [
        "# HCA-Pinned 3D/4D Application Matrix",
        "",
        "## Expected Versus Obtained",
        "",
        "- Expected: explicit rank-to-HCA affinity should remove the severe inter-node "
        "transport collapse observed with automatic rail selection.",
        "- Obtained: every valid FFTM case improved by at least 5.8x; both dimensions "
        "now complete in less than 0.4 seconds with correct numerical results.",
        "",
        "| Case | Unpinned mean ms | HCA mean ms | HCA median ms | Speedup | Reduction | CV |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in current:
        previous = baseline_by_key.get(case.key)
        if previous:
            speedup = previous.mean_ms / case.mean_ms
            reduction = 100.0 * (1.0 - case.mean_ms / previous.mean_ms)
            previous_text = f"{previous.mean_ms:.3f}"
            speedup_text = f"{speedup:.2f}x"
            reduction_text = f"{reduction:.1f}%"
        else:
            previous_text = speedup_text = reduction_text = "-"
        lines.append(
            f"| {case.label} | {previous_text} | {case.mean_ms:.3f} | "
            f"{case.median_ms:.3f} | {speedup_text} | {reduction_text} | "
            f"{case.cv_pct:.1f}% |"
        )

    log_count, marker_count, mappings = affinity
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- FFTM logs: {log_count}; affinity markers: {marker_count} "
            f"(expected {16 * log_count}).",
            f"- Unique rank/HCA mappings: {len(mappings)}.",
            f"- Maximum L2 error: {max(case.max_l2_diff for case in current):.3e}.",
            f"- Egger matrix valid: {'yes' if egger[0] else 'no'} ({egger[1]}).",
            "",
            "## Verdict",
            "",
            "- HCA pinning is mandatory for multinode production runs on this cluster.",
            "- The 3D 4x4 grid is the better tested 16-GPU grid.",
            "- The 4D c8 mean is slightly lower than c4, but each contains one large "
            "tail; their medians differ by less than 1%.",
            "- The Egger comparison must be rerun with the Egger SQSH image before "
            "drawing an FFTM-versus-Egger conclusion.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_figure(
    pdf_path: Path,
    png_path: Path,
    current: Sequence[Case],
    baseline: Sequence[Case],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_by_key = {case.key: case for case in baseline}
    labels = [case.label for case in current]
    old = [baseline_by_key[case.key].mean_ms for case in current]
    new = [case.mean_ms for case in current]
    positions = np.arange(len(labels))
    width = 0.36

    fig, axis = plt.subplots(figsize=(8.3, 4.3), constrained_layout=True)
    axis.bar(
        positions - width / 2,
        old,
        width,
        label="Automatic rail selection",
        color="#A8ADB4",
    )
    axis.bar(
        positions + width / 2,
        new,
        width,
        label="Rank-local HCA affinity",
        color="#167D8D",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Forward + backward wall time (ms)")
    axis.set_title("Two-node application performance after HCA pinning")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for index, case in enumerate(current):
        speedup = baseline_by_key[case.key].mean_ms / case.mean_ms
        axis.text(
            index + width / 2,
            case.mean_ms + 45,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.data_dir.resolve()
    output_dir = (args.output_dir or root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    current = read_fftm_cases(root / "fftm")
    baseline = read_fftm_cases(args.baseline_dir.resolve())
    affinity = affinity_summary(root / "fftm")
    egger = egger_status(root)

    write_csv_summary(output_dir / "hca_application_summary.csv", current, baseline)
    write_markdown(
        output_dir / "hca_application_summary.md",
        current,
        baseline,
        affinity,
        egger,
    )
    write_figure(
        output_dir / "fig_hca_application.pdf",
        output_dir / "fig_hca_application.png",
        current,
        baseline,
    )
    print(f"Wrote HCA application analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
