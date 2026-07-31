#!/usr/bin/env python3
"""Regression checks for balanced Slurm MPI task placement."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert( 0, str( REPO_ROOT / "scripts" ) )

from run_cluster_paper_benchmarks import PaperClusterRunner, RunSpec


def make_runner( data_dir: Path ) -> PaperClusterRunner:
    runner = PaperClusterRunner.__new__( PaperClusterRunner )
    runner.args = SimpleNamespace(
        gpus_per_node=8,
        container_mounts=[],
        srun_extra_args="--exclude=cn13",
        srun_time="00:20:00",
        container_image="/tmp/fftm.sqsh",
        container_workdir="/opt/fftm/bin",
        container_env="",
        container_writable=False,
        mpi_rank_affinity_mode="auto",
        mpi_rank_affinity_wrapper="",
    )
    runner.data_dir = data_dir
    runner.container_data_dir = Path( "/data" )
    runner.mpi_rank_affinity_mode = "auto"
    runner.mpi_rank_affinity_wrapper = ""
    runner.binary_args = lambda spec, sizes, times: [str( value ) for value in sizes]
    return runner


def make_spec( num_gpus: int ) -> RunSpec:
    return RunSpec(
        suite="fftm",
        dim=3,
        case_name="benchmark",
        binary_name="test_benchmark_fftm_3D.bin",
        binary_path="/opt/fftm/bin/test_benchmark_fftm_3D.bin",
        num_gpus=num_gpus,
        transport="cuda_aware",
        strategy="pencil-pencil",
        mode="p2p-waitany",
        uses_mpi=True,
        supports_directory=True,
        memory_family="test",
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        runner = make_runner( Path( tmp ) )

        one_node = runner.slurm_command( make_spec( 8 ), ( 64, 64, 64 ), 1 )
        assert "-N" in one_node and one_node[one_node.index( "-N" ) + 1] == "1"
        assert "--ntasks-per-node=8" in one_node
        assert "--kill-on-bad-exit=1" in one_node

        two_nodes = runner.slurm_command( make_spec( 16 ), ( 64, 64, 64 ), 1 )
        assert "-N" in two_nodes and two_nodes[two_nodes.index( "-N" ) + 1] == "2"
        assert "--ntasks-per-node=8" in two_nodes
        assert "--gpus-per-node=8" in two_nodes
        assert "--kill-on-bad-exit=1" in two_nodes

        partial_second_node = runner.slurm_command( make_spec( 10 ), ( 64, 64, 64 ), 1 )
        assert "--ntasks-per-node=5" in partial_second_node

        runner.args.srun_extra_args = (
            "--exclude=cn13 --ntasks-per-node=8 "
            "--distribution=block:block --kill-on-bad-exit=1"
        )
        explicit_layout = runner.slurm_command( make_spec( 16 ), ( 64, 64, 64 ), 1 )
        assert explicit_layout.count( "--ntasks-per-node=8" ) == 1
        assert explicit_layout.count( "--kill-on-bad-exit=1" ) == 1
        assert explicit_layout.count( "--distribution=block:block" ) == 1

        runner.mpi_rank_affinity_mode = "hca"
        runner.mpi_rank_affinity_wrapper = "/data/fftm_mpi_rank_affinity.sh"
        affinity = runner.slurm_command( make_spec( 16 ), ( 64, 64, 64 ), 1 )
        entrypoint = affinity.index( "--container-entrypoint" )
        assert affinity[entrypoint + 1 : entrypoint + 5] == [
            "/data/fftm_mpi_rank_affinity.sh",
            "--mode",
            "hca",
            "--",
        ]
        assert affinity[entrypoint + 5] == "/opt/fftm/bin/test_benchmark_fftm_3D.bin"

    return 0


if __name__ == "__main__":
    raise SystemExit( main() )
