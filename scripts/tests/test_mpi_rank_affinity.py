#!/usr/bin/env python3
"""Regression checks for the pre-MPI GPU-local HCA wrapper."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "scripts" / "run_mpi_rank_affinity.sh"


def run_rank(
    local_rank: int, sysfs_root: Path, ucx_info: Path
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "SLURM_LOCALID": str(local_rank),
            "SLURM_PROCID": str(local_rank),
            "FFTM_MPI_AFFINITY_SYSFS_ROOT": str(sysfs_root),
            "FFTM_MPI_UCX_INFO_BIN": str(ucx_info),
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
        root = Path(tmp)
        sysfs_root = root / "infiniband"
        for index in (0, 1, 2, 3, 6, 7, 8, 9):
            state_dir = sysfs_root / f"mlx5_{index}" / "ports" / "1"
            state_dir.mkdir(parents=True)
            (state_dir / "state").write_text("4: ACTIVE\n", encoding="ascii")

        ucx_info = root / "ucx_info"
        ucx_info.write_text(
            "#!/usr/bin/env bash\n"
            "cat <<'EOF'\n"
            + "\n".join(
                f"# Device: mlx5_{index}:1" for index in (0, 1, 2, 3, 6, 7, 8, 9)
            )
            + "\nEOF\n",
            encoding="ascii",
        )
        ucx_info.chmod(0o755)

        expected = {
            0: "mlx5_2:1,mlx5_3:1",
            2: "mlx5_0:1,mlx5_1:1",
            4: "mlx5_8:1,mlx5_9:1",
            6: "mlx5_6:1,mlx5_7:1",
        }
        for local_rank, devices in expected.items():
            completed = run_rank(local_rank, sysfs_root, ucx_info)
            assert completed.returncode == 0, completed.stderr
            assert completed.stdout == devices
            assert "[FFTM_MPI_AFFINITY]" in completed.stderr
            assert "validation=active-sysfs+ucx" in completed.stderr

        failed = run_rank(8, sysfs_root, ucx_info)
        assert failed.returncode == 2
        assert "No affinity mapping exists" in failed.stderr

        (sysfs_root / "mlx5_6" / "ports" / "1" / "state").write_text(
            "1: DOWN\n", encoding="ascii"
        )
        failed = run_rank(6, sysfs_root, ucx_info)
        assert failed.returncode == 1
        assert "Mapped HCA port is not active" in failed.stderr

        (sysfs_root / "mlx5_6" / "ports" / "1" / "state").write_text(
            "4: ACTIVE\n", encoding="ascii"
        )
        ucx_info.write_text(
            "#!/usr/bin/env bash\nprintf '# Device: mlx5_7:1\\n'\n",
            encoding="ascii",
        )
        ucx_info.chmod(0o755)
        failed = run_rank(6, sysfs_root, ucx_info)
        assert failed.returncode == 1
        assert "not visible in ucx_info -d" in failed.stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
