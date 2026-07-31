#!/usr/bin/env python3
"""Regression checks for FFTM autotune cache policy selection."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCRIPT = REPO_ROOT / "scripts" / "fftm_autotune_cache.py"


FFTM_FIELDS = [
    "benchmark",
    "num_gpus",
    "nx",
    "ny",
    "nz",
    "p1",
    "p2",
    "strategy",
    "mode",
    "pencil_layout",
    "pencil_pipeline",
    "large_count_p2p_transport",
    "fftm_3d_backend",
    "fft_exec_no_sync",
    "avg_wall_ms",
    "min_wall_ms",
    "stddev_wall_ms",
    "epsilon",
    "max_l2_diff",
    "native_backward_second_peer_loop",
    "native_opt0_default_z_layout",
    "native_opt0_reference_y_buffer_topology",
    "native_opt0_compact_y_workarea",
    "native_opt0_tight_y_plan_sequence",
    "native_opt0_shared_y_plan_handles",
    "native_opt0_y_group_device_sync",
    "native_opt0_y_no_sync_exec",
    "native_opt0_raw_y_plan_array_executor",
    "native_opt0_reference_y_plan_lifecycle",
    "native_opt0_reference_y_plan_bundle",
    "native_opt0_raw_y_plan_bundle",
    "native_opt0_y_plan_bundle_stream_first",
    "native_opt0_raw_y_plan_bundle_reference_streams",
    "native_opt0_reference_local_plan_context",
]


FFTM3D_FIELDS = [
    "num_gpus",
    "nx",
    "ny",
    "nz",
    "p1",
    "p2",
    "opt",
    "egger_variant",
    "device_map",
    "comm1",
    "send1",
    "storage",
    "mpi_backend",
    "runtime_backend",
    "avg_wall_ms",
    "min_wall_ms",
    "stddev_wall_ms",
]


def run_cmd(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CACHE_SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def run_cmd_unchecked(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CACHE_SCRIPT)] + args,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def base_fftm_row(avg_wall_ms: float) -> Dict[str, object]:
    row: Dict[str, object] = {field: "" for field in FFTM_FIELDS}
    row.update(
        {
            "benchmark": "fftm-3d",
            "num_gpus": 8,
            "nx": 2048,
            "ny": 2048,
            "nz": 2048,
            "p1": 4,
            "p2": 2,
            "strategy": "pencil-pencil",
            "mode": "p2p-waitany",
            "pencil_layout": "opt0",
            "pencil_pipeline": "egger-parity",
            "large_count_p2p_transport": "hindexed",
            "fftm_3d_backend": "native",
            "fft_exec_no_sync": 0,
            "avg_wall_ms": avg_wall_ms,
            "min_wall_ms": avg_wall_ms,
            "stddev_wall_ms": 0.1,
            "epsilon": 1.0e-11,
            "max_l2_diff": 1.0e-15,
            "native_backward_second_peer_loop": 0,
            "native_opt0_default_z_layout": 1,
            "native_opt0_reference_y_buffer_topology": 1,
            "native_opt0_compact_y_workarea": 0,
            "native_opt0_tight_y_plan_sequence": 1,
            "native_opt0_shared_y_plan_handles": 1,
            "native_opt0_y_group_device_sync": 1,
            "native_opt0_y_no_sync_exec": 1,
            "native_opt0_raw_y_plan_array_executor": 1,
            "native_opt0_reference_y_plan_lifecycle": 0,
            "native_opt0_reference_y_plan_bundle": 0,
            "native_opt0_raw_y_plan_bundle": 0,
            "native_opt0_y_plan_bundle_stream_first": 0,
            "native_opt0_raw_y_plan_bundle_reference_streams": 0,
            "native_opt0_reference_local_plan_context": 0,
        }
    )
    return row


def write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_result_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps({"args": {"hostname": "testnode", "gpu_name": "A100-SXM4-80GB", "device_memory_mib": 81920}}),
        encoding="utf-8",
    )

    production = base_fftm_row(250.0)
    diagnostic = base_fftm_row(100.0)
    diagnostic["native_opt0_raw_y_plan_bundle"] = 1
    generic_no_sync = base_fftm_row(80.0)
    generic_no_sync["fft_exec_no_sync"] = 1
    invalid_fast = base_fftm_row(50.0)
    invalid_fast["fft_exec_no_sync"] = 1
    invalid_fast["max_l2_diff"] = 1.0

    write_csv(
        root / "cpp_csv" / "benchmark_fftm_3d.csv",
        FFTM_FIELDS,
        [production, diagnostic, generic_no_sync, invalid_fast],
    )

    reference = {
        "num_gpus": 8,
        "nx": 2048,
        "ny": 2048,
        "nz": 2048,
        "p1": 4,
        "p2": 2,
        "opt": "opt0",
        "egger_variant": "scfd-fft-facade",
        "device_map": "rank",
        "comm1": "Peer2Peer",
        "send1": "Sync",
        "storage": "scfd-tensor",
        "mpi_backend": "scfd",
        "runtime_backend": "scfd",
        "avg_wall_ms": 90.0,
        "min_wall_ms": 90.0,
        "stddev_wall_ms": 0.1,
    }
    write_csv(root / "cpp_csv" / "benchmark_fftm3d_3d.csv", FFTM3D_FIELDS, [reference])


def query(cache_path: Path, *extra: str) -> Dict[str, object]:
    result = run_cmd(
        [
            "query",
            "--cache",
            str(cache_path),
            "--library",
            "fftm",
            "--dim",
            "3",
            "--num-gpus",
            "8",
            "--sizes",
            "2048",
            "2048",
            "2048",
            "--strategy",
            "pencil-pencil",
            "--format",
            "json",
        ]
        + list(extra)
    )
    return json.loads(result.stdout)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fftm-autotune-cache-test-") as raw_tmp:
        tmp = Path(raw_tmp)
        result_dir = tmp / "result"
        cache_path = tmp / "cache.json"
        write_result_dir(result_dir)

        update = run_cmd(["update", "--reset", "--output", str(cache_path), str(result_dir)])
        assert "Wrote 5 autotune entries" in update.stdout, update.stdout

        empty_update_path = tmp / "empty_update_cache.json"
        empty_update = run_cmd_unchecked(
            [
                "update",
                "--reset",
                "--output",
                str(empty_update_path),
                str(tmp / "missing_result_dir"),
            ]
        )
        assert empty_update.returncode == 1
        assert "no autotune entries collected" in empty_update.stderr
        assert not empty_update_path.exists()

        missing_cache = run_cmd_unchecked(
            [
                "query",
                "--cache",
                str(tmp / "missing.json"),
                "--library",
                "fftm",
                "--dim",
                "3",
                "--num-gpus",
                "8",
                "--sizes",
                "2048",
                "2048",
                "2048",
            ]
        )
        assert missing_cache.returncode == 2
        assert "Autotune cache does not exist" in missing_cache.stderr

        missing_entry = run_cmd_unchecked(
            [
                "query",
                "--cache",
                str(cache_path),
                "--library",
                "fftm",
                "--dim",
                "3",
                "--num-gpus",
                "5",
                "--sizes",
                "2048",
                "2048",
                "2048",
            ]
        )
        assert missing_entry.returncode == 1
        assert "No autotune cache entry matched" in missing_entry.stderr

        default_entry = query(cache_path)
        default_selected = default_entry["selected"]
        assert default_entry["metrics"]["avg_wall_ms"] == 250.0, json.dumps(default_entry, indent=2)
        assert default_selected["backend"] == "native"
        assert default_selected["pencil_pipeline"] == "reference-parity"
        assert default_selected.get("diagnostic_only", 0) == 0
        assert default_selected.get("reference_only", 0) == 0
        assert default_selected["native_opt0_raw_y_plan_array_executor"] == 1

        legacy_pipeline_query = query(cache_path, "--pencil-pipeline", "egger-parity")
        assert legacy_pipeline_query["selected"]["pencil_pipeline"] == "reference-parity"
        neutral_pipeline_query = query(cache_path, "--pencil-pipeline", "reference-parity")
        assert neutral_pipeline_query["selected"]["pencil_pipeline"] == "reference-parity"

        diagnostic_entry = query(cache_path, "--include-diagnostic")
        assert diagnostic_entry["metrics"]["avg_wall_ms"] == 80.0, json.dumps(diagnostic_entry, indent=2)
        assert diagnostic_entry["selected"]["diagnostic_only"] == 1
        assert diagnostic_entry["selected"]["fft_exec_no_sync"] == 1

        with cache_path.open(encoding="utf-8") as handle:
            cache = json.load(handle)
        assert all(entry["metrics"]["avg_wall_ms"] != 50.0 for entry in cache["entries"])

        legacy_cache_path = tmp / "legacy_cache.json"
        legacy_cache = json.loads(json.dumps(cache))
        for entry in legacy_cache["entries"]:
            selected = entry.get("selected", {})
            if selected.get("pencil_pipeline") == "reference-parity":
                selected["pencil_pipeline"] = "egger-parity"
            if "native_opt0_reference_y_buffer_topology" in selected:
                selected["native_opt0_egger_y_buffer_topology"] = selected.pop(
                    "native_opt0_reference_y_buffer_topology"
                )
        legacy_cache_path.write_text(json.dumps(legacy_cache), encoding="utf-8")
        upgraded_entry = query(legacy_cache_path, "--pencil-pipeline", "reference-parity")
        upgraded_selected = upgraded_entry["selected"]
        assert upgraded_selected["pencil_pipeline"] == "reference-parity"
        assert "native_opt0_egger_y_buffer_topology" not in upgraded_selected
        assert upgraded_selected["native_opt0_reference_y_buffer_topology"] == 1

        reference_entry = query(
            cache_path,
            "--include-reference-only",
            "--backend",
            "fftm3d-scfd-fft-facade",
        )
        assert reference_entry["metrics"]["avg_wall_ms"] == 90.0, json.dumps(reference_entry, indent=2)
        assert reference_entry["selected"]["reference_only"] == 1

        env = run_cmd(
            [
                "query",
                "--cache",
                str(cache_path),
                "--library",
                "fftm",
                "--dim",
                "3",
                "--num-gpus",
                "8",
                "--sizes",
                "2048",
                "2048",
                "2048",
                "--strategy",
                "pencil-pencil",
                "--include-diagnostic",
                "--format",
                "env",
            ]
        )
        assert "FFTM_AUTOTUNE_DIAGNOSTIC_ONLY=1" in env.stdout
        assert "FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS=1" in env.stdout
        assert "FFTM_ALLOW_FFT_EXEC_NO_SYNC=1" in env.stdout
        assert "FFTM_AUTOTUNE_PENCIL_PIPELINE=reference-parity" in env.stdout
        assert "FFTM_PENCIL_PIPELINES=reference-parity" in env.stdout
        assert "egger-parity" not in env.stdout

    print("fftm_autotune_cache regression checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
