#!/usr/bin/env python3
"""Summarize same-system GPU-FFT reference measurements."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PUBLISHED_2048_TFLOPS = {
    8: 10.0,
    16: 12.0,
    32: 17.0,
}


def read_rows(data_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((data_dir / "raw").glob("c*_summary.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            current = list(csv.DictReader(stream))
        if len(current) != 1:
            raise RuntimeError(f"expected one row in {path}, found {len(current)}")
        row = current[0]
        row["source_file"] = str(path.relative_to(data_dir))
        rows.append(row)
    return rows


def numeric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()

    data_dir = arguments.data_dir.resolve()
    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(data_dir)
    if not rows:
        raise RuntimeError(f"no raw/c*_summary.csv files found under {data_dir}")

    rows.sort(key=lambda row: (int(row["size"]), int(row["ranks"])))
    comparison_rows: list[dict[str, object]] = []
    baseline_by_size: dict[int, dict[str, str]] = {}
    for row in rows:
        size = int(row["size"])
        ranks = int(row["ranks"])
        nodes = int(row["nodes"])
        current_tflops = numeric(row, "effective_tflops")
        current_ms = numeric(row, "avg_pair_ms")
        if size not in baseline_by_size or ranks < int(baseline_by_size[size]["ranks"]):
            baseline_by_size[size] = row
        published = PUBLISHED_2048_TFLOPS.get(ranks) if size == 2048 else None
        comparison_rows.append(
            {
                "size": size,
                "nodes": nodes,
                "gpus": ranks,
                "avg_pair_ms": current_ms,
                "median_pair_ms": numeric(row, "median_pair_ms"),
                "p95_pair_ms": numeric(row, "p95_pair_ms"),
                "current_tflops": current_tflops,
                "published_tflops": "" if published is None else published,
                "current_over_published": (
                    "" if published is None else current_tflops / published
                ),
                "relative_l2": numeric(row, "relative_l2"),
                "valid": int(row["valid"]),
                "source_file": row["source_file"],
            }
        )

    for output_row in comparison_rows:
        baseline = baseline_by_size[int(output_row["size"])]
        baseline_ms = numeric(baseline, "avg_pair_ms")
        baseline_gpus = int(baseline["ranks"])
        output_row["strong_speedup_from_min_gpu"] = baseline_ms / float(output_row["avg_pair_ms"])
        output_row["strong_efficiency_from_min_gpu"] = (
            (baseline_ms / float(output_row["avg_pair_ms"]))
            / (int(output_row["gpus"]) / baseline_gpus)
        )

    write_csv(analysis_dir / "gpu_fft_reference_comparison.csv", comparison_rows)

    valid_rows = [row for row in comparison_rows if int(row["valid"]) == 1]
    published_rows = [
        row for row in valid_rows if row["published_tflops"] != ""
    ]
    ratios = [float(row["current_over_published"]) for row in published_rows]

    markdown = [
        "# Same-system GPU-FFT reference diagnostic",
        "",
        "The measured quantity is one double-precision R2C/normalize/C2R pair. ",
        "Effective throughput uses `10*N*log2(N) / wall_time`, matching the ",
        "published GPU-FFT convention. Wall time is the maximum rank time.",
        "",
        "| Size | Nodes | GPUs | Mean pair (ms) | Median (ms) | TFLOP/s | Published TFLOP/s | Current/published | Valid |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison_rows:
        published = row["published_tflops"]
        ratio = row["current_over_published"]
        markdown.append(
            "| {size}^3 | {nodes} | {gpus} | {avg:.3f} | {median:.3f} | "
            "{rate:.3f} | {published} | {ratio} | {valid} |".format(
                size=row["size"],
                nodes=row["nodes"],
                gpus=row["gpus"],
                avg=float(row["avg_pair_ms"]),
                median=float(row["median_pair_ms"]),
                rate=float(row["current_tflops"]),
                published=("-" if published == "" else f"{float(published):.3f}"),
                ratio=("-" if ratio == "" else f"{float(ratio):.3f}"),
                valid="yes" if int(row["valid"]) else "no",
            )
        )

    markdown.extend(["", "## Interpretation guardrails", ""])
    if ratios:
        markdown.append(
            f"Across the comparable 2048^3 points, current/published throughput ranges "
            f"from {min(ratios):.3f} to {max(ratios):.3f}."
        )
    markdown.extend(
        [
            "- If this unmodified GPU-FFT algorithm shows the same 8-to-16 GPU collapse as FFTM, the current cluster transport/topology is the dominant common cause.",
            "- If GPU-FFT approaches its published 16/32-GPU rates while FFTM does not, the remaining loss is specific to FFTM's decomposition or schedule.",
            "- The 1920^3 series is the exact 1/2/3/4-node strong-scaling check. The published-size 2048^3 series excludes 24 ranks because GPU-FFT requires the slab size to divide the rank count.",
        ]
    )
    (analysis_dir / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    manifest = {
        "data_directory": str(data_dir),
        "valid_rows": len(valid_rows),
        "total_rows": len(comparison_rows),
        "published_2048_tflops": PUBLISHED_2048_TFLOPS,
        "effective_flops": "10 * total_points * log2(total_points) per R2C/C2R pair",
        "upstream_repository": "https://github.com/Manthan-Verma/GPU_FFT",
        "upstream_commit": rows[0].get("upstream_commit", "unknown"),
    }
    (analysis_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Analyzed {len(comparison_rows)} GPU-FFT reference rows in {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
