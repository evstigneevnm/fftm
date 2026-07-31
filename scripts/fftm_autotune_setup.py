#!/usr/bin/env python3
"""Resolve or create an FFTM autotune selection for one machine/size.

This is the setup-layer companion to fftm_autotune_cache.py. It first queries
the JSON cache. If the entry is absent and --run-missing is passed, it launches
the small production benchmark matrix, updates the cache, and emits the selected
configuration as key=value lines.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import fftm_autotune_cache as cache_mod  # noqa: E402


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def entry_to_env_text(entry: Dict[str, object]) -> str:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        cache_mod.print_env(entry)
    return stream.getvalue()


def query_entry(args: argparse.Namespace) -> Optional[Dict[str, object]]:
    cache = cache_mod.load_cache(Path(args.cache).resolve())
    query_args = argparse.Namespace(
        library=args.library,
        dim=3,
        num_gpus=args.num_gpus,
        sizes=[args.size_3d, args.size_3d, args.size_3d],
        hostname=args.hostname,
        gpu_name=args.gpu_name,
        strategy=args.strategy_3d,
        mode=args.mode,
        backend=args.backend_3d,
        pencil_layout=args.pencil_layout,
        pencil_pipeline=args.pencil_pipeline,
        grid=args.grid_3d,
        grid_orientation=args.grid_orientation,
        large_count_p2p_transport=args.large_count_p2p_transport,
        include_reference_only=False,
        include_diagnostic=False,
    )
    return cache_mod.find_entry(cache, query_args)


def write_or_print_env(text: str, output_env: str) -> None:
    if output_env:
        path = Path(output_env)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote autotune environment to {path}")
    else:
        print(text, end="")


def apply_extra_env(env: Dict[str, str], values: List[str]) -> None:
    for item in values:
        if "=" not in item:
            raise ValueError(f"--extra-env expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        env[key] = value


def production_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir).resolve()
    root = Path(args.data_root).resolve()
    return root / f"data_{args.size_3d}_{args.library}_autotune_{stamp()}"


def build_fftm3d_run(args: argparse.Namespace, data_dir: Path) -> tuple[Dict[str, str], List[str]]:
    env = os.environ.copy()
    env.update(
        {
            "FFTM_CONTAINER_IMAGE": args.container_image,
            "FFTM3D_DATA_DIR": str(data_dir),
            "FFTM3D_NODE_COUNTS": str(args.node_counts),
            "FFTM3D_GPU_COUNTS": str(args.num_gpus),
            "FFTM3D_GPUS_PER_NODE": str(args.gpus_per_node),
            "FFTM3D_BENCHMARK_SIZES_3D": str(args.size_3d),
            "FFTM3D_GRID_SELECTION": "production",
            "FFTM3D_OPTS": args.fftm3d_opts,
            "FFTM3D_COMM_METHODS": "Peer2Peer",
            "FFTM3D_SEND_METHODS": "Sync",
            "FFTM3D_EGGER_VARIANTS": args.fftm3d_variant,
            "FFTM3D_STORAGE_MODES": args.fftm3d_storage,
            "FFTM3D_MPI_BACKENDS": args.fftm3d_mpi_backend,
            "FFTM3D_RUNTIME_BACKENDS": args.fftm3d_runtime_backend,
            "FFTM3D_DEVICE_MAPS": "rank",
            "FFTM3D_DIAGNOSTIC_STAGES": "wall",
            "FFTM3D_WRITE_ITERATION_CSV": "1",
            "FFTM3D_ENABLE_TELEMETRY": "1",
            "FFTM3D_SHORT_FILENAMES": "1",
            "FFTM3D_BENCHMARK_TIMES": str(args.benchmark_times),
            "FFTM3D_WARMUP": str(args.warmup),
            "FFTM3D_SRUN_TIME": args.srun_time,
            "FFTM3D_STOP_ON_FAILURE": "1",
        }
    )
    if args.fftm3d_grids:
        env["FFTM3D_GRIDS"] = args.fftm3d_grids
    apply_extra_env(env, args.extra_env)
    return env, [str(REPO_ROOT / "fftm3d" / "scripts" / "run_slurm_pyxis_fftm3d_benchmarks.sh")]


def build_fftm_run(args: argparse.Namespace, data_dir: Path) -> tuple[Dict[str, str], List[str]]:
    strategy_3d = args.strategy_3d if not cache_mod.inactive_filter(args.strategy_3d) else "slab-pencil,pencil-slab,pencil-pencil"
    modes = args.mode if not cache_mod.inactive_filter(args.mode) else "p2p-waitany,alltoallv"
    pipeline = args.pencil_pipeline if not cache_mod.inactive_filter(args.pencil_pipeline) else args.fftm_pipeline
    pipeline = ",".join(cache_mod.canonical_pencil_pipeline(item) for item in cache_mod.filter_values(pipeline)) or pipeline
    layouts = args.pencil_layout if not cache_mod.inactive_filter(args.pencil_layout) else args.fftm_layouts
    backends = args.backend_3d if not cache_mod.inactive_filter(args.backend_3d) else args.fftm_backends
    large_count = (
        args.large_count_p2p_transport
        if not cache_mod.inactive_filter(args.large_count_p2p_transport)
        else "hindexed"
    )
    env = os.environ.copy()
    env.update(
        {
            "FFTM_CONTAINER_IMAGE": args.container_image,
            "FFTM_DATA_DIR": str(data_dir),
            "FFTM_NODE_COUNTS": str(args.node_counts),
            "FFTM_GPU_COUNTS": str(args.num_gpus),
            "FFTM_GPUS_PER_NODE": str(args.gpus_per_node),
            "FFTM_DEVICE_MEMORY_MIB": str(args.device_memory_mib),
            "FFTM_BENCHMARK_SIZES_3D": "none",
            "FFTM_BENCHMARK_SIZES_4D": "none",
            "FFTM_FIXED_SCALING_SIZES_3D": str(args.size_3d),
            "FFTM_EXTRA_SIZES_3D_BY_GPU": "",
            "FFTM_MODES": modes,
            "FFTM_TRANSPORTS": "cuda_aware",
            "FFTM_STRATEGIES_3D": strategy_3d,
            "FFTM_STRATEGIES_4D": "",
            "FFTM_SKIP_FFTS": "1",
            "FFTM_SKIP_FFTM": "0",
            "FFTM_PENCIL_PIPELINES": pipeline,
            "FFTM_PENCIL_LAYOUTS": layouts,
            "FFTM_3D_BACKENDS": backends,
            "FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS": "production",
            "FFTM_P2P_VARIANTS": "byte-packed",
            "FFTM_P2P_SCHEDULERS": "main",
            "FFTM_LARGE_COUNT_P2P_TRANSPORTS": large_count,
            "FFTM_USE_DIRECT_BACKWARD_RECEIVE": "0",
            "FFTM_DIRECT_P2P_CUDA_AWARE": "1",
            "FFTM_USE_P2P_SEND_THREAD": "0",
            "FFTM_USE_P2P_BYTE_TRANSFER": "1",
            "FFTM_USE_PERSISTENT_P2P": "0",
            "FFTM_USE_READY_P2P_SEND": "0",
            "FFTM_PRINT_PENCIL_SCHEDULE": "0",
            "FFTM_BENCHMARK_TIMES": str(args.benchmark_times),
            "FFTM_WARMUP": str(args.warmup),
            "FFTM_SRUN_TIME": args.srun_time,
            "FFTM_STOP_ON_FAILURE": "1",
        }
    )
    apply_extra_env(env, args.extra_env)
    return env, [str(REPO_ROOT / "scripts" / "run_slurm_pyxis_benchmarks.sh")]


def print_dry_run(env: Dict[str, str], command: List[str]) -> None:
    interesting = sorted(k for k in env if k.startswith("FFTM"))
    for key in interesting:
        print(f"{key}={env[key]}")
    print("COMMAND=" + " ".join(command))


def run_missing_benchmark(args: argparse.Namespace) -> Path:
    if not args.container_image:
        raise RuntimeError("--container-image is required with --run-missing")
    data_dir = production_data_dir(args)
    if args.library == "fftm3d":
        env, command = build_fftm3d_run(args, data_dir)
    else:
        env, command = build_fftm_run(args, data_dir)
    if args.dry_run:
        print_dry_run(env, command)
        return data_dir
    subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
    return data_dir


def update_cache_from_result(args: argparse.Namespace, data_dir: Path) -> None:
    update_args = argparse.Namespace(result_dirs=[str(data_dir)], output=args.cache)
    cache_mod.update_cache(update_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", choices=["fftm", "fftm3d"], default="fftm")
    parser.add_argument("--cache", default="build/resutls_stats/fftm_autotune_cache.json")
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--size-3d", type=int, required=True)
    parser.add_argument("--hostname", default="", help="Optional cache hostname filter.")
    parser.add_argument("--gpu-name", default="", help="Optional cache GPU-name filter.")
    parser.add_argument(
        "--strategy-3d",
        default=os.environ.get("FFTM_AUTOTUNE_STRATEGY_3D", os.environ.get("FFTM_STRATEGIES_3D", "")),
        help="Optional strategy constraint. Use auto/empty for global fastest.",
    )
    parser.add_argument(
        "--mode",
        default=os.environ.get("FFTM_AUTOTUNE_MODE", os.environ.get("FFTM_MODES", "")),
        help="Optional redistribution mode constraint.",
    )
    parser.add_argument(
        "--backend-3d",
        default=os.environ.get("FFTM_AUTOTUNE_BACKEND_3D", os.environ.get("FFTM_3D_BACKENDS", "")),
        help="Optional FFTM 3D backend constraint.",
    )
    parser.add_argument(
        "--pencil-layout",
        default=os.environ.get("FFTM_AUTOTUNE_PENCIL_LAYOUT", os.environ.get("FFTM_PENCIL_LAYOUTS", "")),
        help="Optional pencil layout constraint.",
    )
    parser.add_argument(
        "--pencil-pipeline",
        default=os.environ.get("FFTM_AUTOTUNE_PENCIL_PIPELINE", os.environ.get("FFTM_PENCIL_PIPELINES", "")),
        help="Optional pencil pipeline constraint.",
    )
    parser.add_argument(
        "--grid-3d",
        default=os.environ.get("FFTM_AUTOTUNE_GRID_3D", ""),
        help="Optional grid constraint, for example 4x2.",
    )
    parser.add_argument(
        "--grid-orientation",
        default=os.environ.get("FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS", ""),
        help="Optional grid orientation constraint.",
    )
    parser.add_argument(
        "--large-count-p2p-transport",
        default=os.environ.get("FFTM_LARGE_COUNT_P2P_TRANSPORTS", ""),
        help="Optional large-count P2P transport constraint.",
    )
    parser.add_argument("--output-env", default="", help="Write selected key=value config to this file.")
    parser.add_argument("--run-missing", action="store_true", help="Run a production benchmark if cache is missing.")
    parser.add_argument("--force-run", action="store_true", help="Run benchmark even when a cache entry exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print the benchmark environment without launching.")

    parser.add_argument("--container-image", default=os.environ.get("FFTM_CONTAINER_IMAGE", ""))
    parser.add_argument("--data-root", default="build/resutls_stats")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--node-counts", default="1")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--device-memory-mib", type=int, default=81920)
    parser.add_argument("--benchmark-times", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--srun-time", default="00:30:00")
    parser.add_argument("--extra-env", action="append", default=[], help="Additional KEY=VALUE passed to benchmark run.")

    parser.add_argument(
        "--fftm-pipeline",
        default="reference-parity",
        help="Default FFTM pencil pipeline for missing-cache production runs. Deprecated aliases are accepted.",
    )
    parser.add_argument("--fftm-layouts", default="opt0,opt1")
    parser.add_argument("--fftm-backends", default="native,fftm3d-scfd-fft-facade")

    parser.add_argument("--fftm3d-grids", default="")
    parser.add_argument("--fftm3d-opts", default="0,1")
    parser.add_argument("--fftm3d-variant", default="scfd-fft-facade")
    parser.add_argument("--fftm3d-storage", default="scfd-tensor")
    parser.add_argument("--fftm3d-mpi-backend", default="scfd")
    parser.add_argument("--fftm3d-runtime-backend", default="scfd")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    entry = None if args.force_run else query_entry(args)
    if entry is not None:
        write_or_print_env(entry_to_env_text(entry), args.output_env)
        return 0

    if not args.run_missing:
        print(
            "No autotune cache entry found. Re-run with --run-missing to benchmark and save one.",
            file=sys.stderr,
        )
        return 1

    data_dir = run_missing_benchmark(args)
    if args.dry_run:
        return 0

    update_cache_from_result(args, data_dir)
    entry = query_entry(args)
    if entry is None:
        print(f"Benchmark finished but no matching cache entry was produced from {data_dir}", file=sys.stderr)
        return 1
    write_or_print_env(entry_to_env_text(entry), args.output_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
