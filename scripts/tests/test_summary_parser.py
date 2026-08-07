#!/usr/bin/env python3
"""Regression checks for benchmark summaries containing process-grid tuples."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert( 0, str( REPO_ROOT / "scripts" ) )

import run_local_paper_benchmarks as benchmarks


def main() -> int:
    complete = benchmarks.parse_summary_line(
        "INFO: benchmark=fftm-3d, strategy=pencil-pencil, grid=(2,3), Nx=2048: "
        "avg_wall_ms=346.81, stddev_wall_ms=2.91"
    )
    assert complete is not None
    assert complete["grid"] == [2, 3]
    assert complete["avg_wall_ms"] == 346.81

    truncated = benchmarks.parse_summary_line(
        "INFO: benchmark=fftm-3d, strategy=pencil-pencil, grid=(4,2), Nx=2048, native_option=1"
    )
    assert truncated is not None
    assert truncated["grid"] == [4, 2]
    assert truncated["Nx"] == 2048

    four_dimensional = benchmarks.parse_summary_line(
        "INFO: benchmark=fftm-4d, strategy=slab-slab, grid=(1,8,1), Nx=320: avg_wall_ms=216.97"
    )
    assert four_dimensional is not None
    assert four_dimensional["grid"] == [1, 8, 1]
    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
