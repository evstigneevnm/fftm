#!/usr/bin/env python3
"""Regression checks for transport-compatible 3D pencil pipeline selection."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert( 0, str( REPO_ROOT / "scripts" ) )

import run_local_paper_benchmarks as benchmarks


def main() -> int:
    selected = ["reference-parity", "reference"]

    assert benchmarks.pencil_pipelines_for_spec(
        3, "pencil-pencil", "p2p-waitany", selected, "cuda_aware"
    ) == selected
    assert benchmarks.pencil_pipelines_for_spec(
        3, "pencil-pencil", "p2p-waitany", selected, "non_cuda_aware"
    ) == ["reference"]
    assert benchmarks.pencil_pipelines_for_spec(
        3, "pencil-pencil", "p2p-waitany", ["reference-parity"], "non_cuda_aware"
    ) == ["reference"]
    assert benchmarks.pencil_pipelines_for_spec(
        3, "slab-pencil", "p2p-waitany", selected, "non_cuda_aware"
    ) == [None]
    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
