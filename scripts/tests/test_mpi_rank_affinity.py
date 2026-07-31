#!/usr/bin/env python3
"""Regression checks for the pre-MPI GPU-local HCA wrapper."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "run_mpi_rank_affinity.sh"


def run_rank(local_rank: int, sysfs_root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_LOCALID": str(local_rank),
            "SLURM_PROCID": str(local_rank),
            "FFTM_MPI_AFFINITY_SYSFS_ROOT": str(sysfs_root),
        }
    )
    return subprocess.run(
        [
            str(WRAPPER),
            "--mode",
            "hca",
            "--",
            "/bin/bash",
            "-c",
            'printf "%s" "${UCX_NET_DEVICES}"',
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        sysfs_root = Path(tmp)
        for index in (0, 1, 2, 3, 6, 7, 8, 9):
            (sysfs_root / f"mlx5_{index}").mkdir()

        expected = {
            0: "mlx5_2:1,mlx5_3:1",
            2: "mlx5_0:1,mlx5_1:1",
            4: "mlx5_8:1,mlx5_9:1",
            6: "mlx5_6:1,mlx5_7:1",
        }
        for local_rank, devices in expected.items():
            completed = run_rank(local_rank, sysfs_root)
            assert completed.returncode == 0, completed.stderr
            assert completed.stdout == devices
            assert "[FFTM_MPI_AFFINITY]" in completed.stderr

        failed = run_rank(8, sysfs_root)
        assert failed.returncode == 2
        assert "No affinity mapping exists" in failed.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
