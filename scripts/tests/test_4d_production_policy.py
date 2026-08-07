#!/usr/bin/env python3
"""Regression checks for the validated 4D slab production policy."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert( 0, str( REPO_ROOT / "scripts" ) )

import run_local_paper_benchmarks as benchmarks


def main() -> int:
    assert benchmarks.parse_fftm_4d_slab_xw_transposes( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_slab_xw_native_spectral_layouts( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_native_xw_direct_layouts( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_native_xw_protocols( "production" ) == ["chunked"]
    assert benchmarks.parse_fftm_4d_native_xw_chunk_windows( "production" ) == [1]
    assert benchmarks.parse_fftm_4d_native_xw_compact_stagings( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_slab_native_work_area_aliases( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_slab_native_wz_communication_layouts( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_slab_native_wz_plan_concurrencies( "production" ) == [4]
    assert benchmarks.parse_fftm_4d_slab_native_wz_ready_pipelines( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_pencil_p3_degenerate_wz_pipelines( "production" ) == [True]
    assert benchmarks.parse_fftm_4d_pencil_p3_degenerate_wz_pipelines( "standard" ) == [False]
    assert benchmarks.parse_fftm_4d_pencil_p3_degenerate_wz_pipelines( "matrix" ) == [False, True]
    explicit = benchmarks.parse_4d_pencil_grid_orientations( "explicit:2x8x1,4x8x1" )
    assert benchmarks.grid_orientations_4d_for_spec( 16, explicit ) == [(2, 8, 1)]
    assert benchmarks.grid_orientations_4d_for_spec( 32, explicit ) == [(4, 8, 1)]
    assert benchmarks.grid_orientations_4d_for_spec( 24, explicit ) == []
    assert benchmarks.fftm_4d_native_xw_direct_layouts_for_transport(
        "cuda_aware", [True]
    ) == [True]
    assert benchmarks.fftm_4d_native_xw_direct_layouts_for_transport(
        "non_cuda_aware", [True]
    ) == [False]
    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
