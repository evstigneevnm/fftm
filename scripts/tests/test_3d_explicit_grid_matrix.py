#!/usr/bin/env python3
"""Regression checks for explicit 3D pencil-grid matrices."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_local_paper_benchmarks import (  # noqa: E402
    grid_orientations_for_spec,
    parse_pencil_grid_orientations,
)


def main() -> int:
    explicit = parse_pencil_grid_orientations("2x8, 4x4, 8x2")
    assert explicit == "explicit:2x8,4x4,8x2"
    assert grid_orientations_for_spec(3, "pencil-pencil", 16, explicit) == [
        (2, 8),
        (4, 4),
        (8, 2),
    ]

    per_gpu = parse_pencil_grid_orientations("1x8,2x8,4x8")
    assert grid_orientations_for_spec(3, "pencil-pencil", 8, per_gpu) == [(1, 8)]
    assert grid_orientations_for_spec(3, "pencil-pencil", 16, per_gpu) == [(2, 8)]
    assert grid_orientations_for_spec(3, "pencil-pencil", 32, per_gpu) == [(4, 8)]

    assert grid_orientations_for_spec(4, "slab-slab", 16, explicit) == [None]

    try:
        grid_orientations_for_spec(3, "pencil-pencil", 32, explicit)
    except ValueError as exc:
        assert "product 32" in str(exc)
    else:
        raise AssertionError("missing explicit-grid product validation")

    for invalid in ("2x0", "2x8x1", "2x8,", "2*8"):
        try:
            parse_pencil_grid_orientations(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid explicit grid accepted: {invalid}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
