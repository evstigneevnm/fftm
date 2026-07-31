#!/usr/bin/env python3

import argparse
import csv
import statistics
from pathlib import Path


def read_single_row(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise RuntimeError(f"{path} contains {len(rows)} summary rows")
    return rows[0]


def mean_column(path, column):
    with path.open(newline="") as stream:
        values = [float(row[column]) for row in csv.DictReader(stream)]
    return statistics.fmean(values) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--baseline-ms", type=float, default=217.723)
    parser.add_argument("--required-improvement-pct", type=float, default=10.0)
    parser.add_argument("--target-gpus", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=320)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    status_path = data_dir / "status.csv"
    if not status_path.is_file():
        raise SystemExit(f"missing status file: {status_path}")

    threshold_ms = args.baseline_ms * (
        1.0 - args.required_improvement_pct / 100.0
    )
    rows = []
    with status_path.open(newline="") as stream:
        statuses = list(csv.DictReader(stream))

    for status in statuses:
        if status["label"] == "preflight":
            continue
        result = {
            "case_id": status["case_id"],
            "label": status["label"],
            "gpus": status["gpus"],
            "size": status["size"],
            "rc": status["rc"],
            "avg_pair_ms": "",
            "min_pair_ms": "",
            "stddev_pair_ms": "",
            "forward_yzw_ms": "",
            "forward_layout_ms": "",
            "forward_x_ms": "",
            "inverse_x_ms": "",
            "inverse_layout_ms": "",
            "inverse_yzw_ms": "",
            "relative_l2": "",
            "allocated_data_gib": "",
            "shared_work_gib": "",
            "max_device_gib": "",
            "improvement_pct": "",
            "speedup": "",
            "passes_10pct_gate": "",
        }
        summary_path = data_dir / "raw" / (
            f"{status['case_id']}_{status['label']}.csv"
        )
        samples_path = summary_path.with_name(
            summary_path.stem + ".samples.csv"
        )
        if status["rc"] == "0" and summary_path.is_file():
            summary = read_single_row(summary_path)
            average = float(summary["avg_pair_ms"])
            is_target = (
                int(status["gpus"]) == args.target_gpus
                and int(status["size"]) == args.target_size
            )
            result.update(
                {
                    "avg_pair_ms": f"{average:.6f}",
                    "min_pair_ms": summary["min_pair_ms"],
                    "stddev_pair_ms": summary["stddev_pair_ms"],
                    "relative_l2": summary["relative_l2"],
                    "allocated_data_gib": (
                        f"{int(summary['allocated_data_bytes']) / 2**30:.6f}"
                    ),
                    "shared_work_gib": (
                        f"{int(summary['shared_work_bytes']) / 2**30:.6f}"
                    ),
                    "max_device_gib": (
                        f"{(int(summary['max_data_bytes_per_device']) + int(summary['max_work_bytes_per_device'])) / 2**30:.6f}"
                    ),
                    "improvement_pct": (
                        f"{100.0 * (args.baseline_ms - average) / args.baseline_ms:.6f}"
                        if is_target
                        else ""
                    ),
                    "speedup": f"{args.baseline_ms / average:.6f}" if is_target else "",
                    "passes_10pct_gate": (
                        ("1" if average <= threshold_ms else "0")
                        if is_target
                        else ""
                    ),
                }
            )
            if samples_path.is_file():
                for column in (
                    "forward_yzw_ms",
                    "forward_layout_ms",
                    "forward_x_ms",
                    "inverse_x_ms",
                    "inverse_layout_ms",
                    "inverse_yzw_ms",
                ):
                    result[column] = f"{mean_column(samples_path, column):.6f}"
        rows.append(result)

    analysis_dir = data_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else [
        "case_id", "label", "gpus", "size", "rc"
    ]
    with (analysis_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with (analysis_dir / "summary.md").open("w") as stream:
        stream.write("# Standalone 4D Node-Hybrid Result\n\n")
        stream.write(f"- FFTM baseline: {args.baseline_ms:.3f} ms\n")
        stream.write(
            f"- Required improvement: {args.required_improvement_pct:.1f}%\n"
        )
        stream.write(f"- Acceptance threshold: {threshold_ms:.3f} ms\n")
        stream.write(
            f"- Performance gate applies only to: {args.target_gpus} GPUs, "
            f"{args.target_size}^4\n\n"
        )
        stream.write(
            "| Case | GPUs | Size | RC | Average ms | Improvement | Gate |\n"
        )
        stream.write("|---|---:|---:|---:|---:|---:|:---:|\n")
        for row in rows:
            improvement = (
                f"{float(row['improvement_pct']):.2f}%"
                if row["improvement_pct"]
                else ""
            )
            if row["passes_10pct_gate"] == "1":
                gate = "PASS"
            elif row["passes_10pct_gate"] == "0":
                gate = "FAIL"
            else:
                gate = "N/A"
            stream.write(
                f"| {row['label']} | {row['gpus']} | {row['size']} | "
                f"{row['rc']} | {row['avg_pair_ms']} | {improvement} | "
                f"{gate} |\n"
            )

    print(f"Wrote {analysis_dir / 'summary.csv'}")
    print(f"Wrote {analysis_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
