#!/usr/bin/env python3
"""Run FFTM/FFTS benchmark matrices locally or through Slurm/Pyxis.

The output format intentionally matches run_local_paper_benchmarks.py:
  data-directory/
    runs.jsonl
    raw/*.log
    cpp_csv/*.csv
    hardware.json
    matrix.json
    config.json
    summary.json

Executor modes:
  local       Run binaries directly, using mpiexec for MPI cases. This is for
              local Docker validation.
  slurm-pyxis Host-side orchestration. Every test is launched through srun with
              --container-image; no mpiexec is invoked inside the container.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_local_paper_benchmarks import (  # noqa: E402
    FFTM_MODES,
    FFTM_STRATEGIES_3D,
    FFTM_STRATEGIES_4D,
    FFTS_STRATEGIES_4D,
    RunSpec,
    bytes_to_mib,
    detect_gpus,
    infer_device_peak_max_bytes,
    mib_to_bytes,
    now_iso,
    parse_run_output,
    slugify,
)


FFTS_VERSIONED_CASES = tuple(f"v{i}" for i in range(5))
FFTM_VERSIONED_CASES = tuple(f"v{i}" for i in range(5))
TRANSPORTS = ("cuda_aware", "non_cuda_aware")


def parse_csv_ints(value: str, *, allow_auto: bool = False) -> List[int]:
    value = (value or "").strip()
    if allow_auto and value == "auto":
        return []
    if not value:
        return []
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


def dense_gpu_counts(max_gpus: int) -> List[int]:
    if max_gpus < 2:
        return []
    return list(range(2, max_gpus + 1))


def parse_csv_strings(value: str, allowed: Sequence[str], name: str) -> List[str]:
    if value == "all":
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"unknown {name}: {', '.join(unknown)}; allowed: {', '.join(allowed)}")
    return selected


def container_binary_map(tests_root: Path) -> Dict[str, Path]:
    names: List[str] = [
        "test_benchmark_ffts_3D.bin",
        "test_benchmark_ffts_4D.bin",
        "test_benchmark_fftm_3D.bin",
        "test_benchmark_fftm_3D_nca.bin",
        "test_benchmark_fftm_4D.bin",
        "test_benchmark_fftm_4D_nca.bin",
    ]
    for index in range(5):
        names.extend(
            [
                f"test_ffts_v{index}_3D.bin",
                f"test_ffts_v{index}_4D.bin",
                f"test_fftm_v{index}_3D.bin",
                f"test_fftm_v{index}_3D_nca.bin",
                f"test_fftm_v{index}_4D.bin",
                f"test_fftm_v{index}_4D_nca.bin",
            ]
        )
    return {name: (tests_root / name) for name in names}


def discover_local_binaries(tests_root: Path) -> Dict[str, Path]:
    return {path.name: path.resolve() for path in sorted(tests_root.glob("*.bin"))}


def binary_name_for_fftm(dim: int, transport: str, case_name: str) -> str:
    suffix = "" if transport == "cuda_aware" else "_nca"
    if case_name == "benchmark":
        return f"test_benchmark_fftm_{dim}D{suffix}.bin"
    return f"test_fftm_{case_name}_{dim}D{suffix}.bin"


def binary_name_for_ffts(dim: int, case_name: str) -> str:
    if case_name == "benchmark":
        return f"test_benchmark_ffts_{dim}D.bin"
    return f"test_ffts_{case_name}_{dim}D.bin"


def build_specs(
    binaries: Dict[str, Path],
    gpu_counts: Sequence[int],
    *,
    include_ffts: bool,
    include_fftm: bool,
    include_versioned: bool,
    versioned_full_matrix: bool,
    transports: Sequence[str],
    modes: Sequence[str],
    strategies_3d: Sequence[str],
    strategies_4d: Sequence[str],
) -> List[RunSpec]:
    specs: List[RunSpec] = []

    if include_ffts:
        for dim in (3, 4):
            name = binary_name_for_ffts(dim, "benchmark")
            if name in binaries:
                strategies = FFTS_STRATEGIES_4D if dim == 4 else (None,)
                for strategy in strategies:
                    specs.append(
                        RunSpec(
                            suite="ffts",
                            dim=dim,
                            case_name="benchmark",
                            binary_name=name,
                            binary_path=str(binaries[name]),
                            num_gpus=1,
                            transport="single_gpu",
                            strategy=strategy,
                            mode=None,
                            uses_mpi=False,
                            supports_directory=True,
                            memory_family=f"ffts:{dim}d:benchmark:{strategy or 'cufft'}",
                        )
                    )

            if include_versioned:
                for case_name in FFTS_VERSIONED_CASES:
                    name = binary_name_for_ffts(dim, case_name)
                    if name not in binaries:
                        continue
                    strategies = FFTS_STRATEGIES_4D if dim == 4 else (None,)
                    for strategy in strategies:
                        specs.append(
                            RunSpec(
                                suite="ffts",
                                dim=dim,
                                case_name=case_name,
                                binary_name=name,
                                binary_path=str(binaries[name]),
                                num_gpus=1,
                                transport="single_gpu",
                                strategy=strategy,
                                mode=None,
                                uses_mpi=False,
                                supports_directory=True,
                                memory_family=f"ffts:{dim}d:{case_name}:{strategy or 'cufft'}",
                            )
                        )

    if include_fftm:
        for num_gpus in gpu_counts:
            if num_gpus < 2:
                continue
            for dim, strategies in [(3, strategies_3d), (4, strategies_4d)]:
                for transport in transports:
                    name = binary_name_for_fftm(dim, transport, "benchmark")
                    if name in binaries:
                        for strategy in strategies:
                            for mode in modes:
                                specs.append(
                                    RunSpec(
                                        suite="fftm",
                                        dim=dim,
                                        case_name="benchmark",
                                        binary_name=name,
                                        binary_path=str(binaries[name]),
                                        num_gpus=num_gpus,
                                        transport=transport,
                                        strategy=strategy,
                                        mode=mode,
                                        uses_mpi=True,
                                        supports_directory=True,
                                        memory_family=f"fftm:{dim}d:benchmark:{num_gpus}:{transport}:{strategy}",
                                    )
                                )

                    if include_versioned:
                        versioned_modes = modes if versioned_full_matrix else ("p2p-waitany",)
                        for case_name in FFTM_VERSIONED_CASES:
                            name = binary_name_for_fftm(dim, transport, case_name)
                            if name not in binaries:
                                continue
                            for strategy in strategies:
                                for mode in versioned_modes:
                                    specs.append(
                                        RunSpec(
                                            suite="fftm",
                                            dim=dim,
                                            case_name=case_name,
                                            binary_name=name,
                                            binary_path=str(binaries[name]),
                                            num_gpus=num_gpus,
                                            transport=transport,
                                            strategy=strategy,
                                            mode=mode,
                                            uses_mpi=True,
                                            supports_directory=False,
                                            memory_family=f"fftm:{dim}d:{case_name}:{num_gpus}:{transport}:{strategy}",
                                        )
                                    )
    return specs


class PaperClusterRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.executor = args.executor
        self.data_dir = Path(args.data_directory).resolve()
        self.raw_dir = self.data_dir / "raw"
        self.cpp_csv_host = self.data_dir / "cpp_csv"
        self.runs_jsonl_path = self.data_dir / "runs.jsonl"
        self.container_data_dir = Path(args.container_data_directory)
        self.container_cpp_csv = self.container_data_dir / "cpp_csv"
        self.run_index = 0

        if self.executor == "local":
            self.tests_root = Path(args.tests_root).resolve()
            self.binaries = discover_local_binaries(self.tests_root)
            self.detected_gpus = self.detect_local_gpus()
            self.gpu_counts = self.resolve_local_gpu_counts()
        else:
            self.tests_root = Path(args.container_tests_root)
            self.binaries = container_binary_map(self.tests_root)
            self.detected_gpus = []
            self.gpu_counts = self.resolve_slurm_gpu_counts()

        self.transports = parse_csv_strings(args.transports, TRANSPORTS, "--transports")
        self.modes = parse_csv_strings(args.modes, FFTM_MODES, "--modes")
        self.strategies_3d = parse_csv_strings(args.strategies_3d, FFTM_STRATEGIES_3D, "--strategies-3d")
        self.strategies_4d = parse_csv_strings(args.strategies_4d, FFTM_STRATEGIES_4D, "--strategies-4d")
        self.benchmark_sizes_3d = parse_csv_ints(args.benchmark_sizes_3d)
        self.benchmark_sizes_4d = parse_csv_ints(args.benchmark_sizes_4d)
        if not self.benchmark_sizes_3d or not self.benchmark_sizes_4d:
            raise ValueError("benchmark size lists cannot be empty")

        self.specs = build_specs(
            self.binaries,
            self.gpu_counts,
            include_ffts=not args.skip_ffts,
            include_fftm=not args.skip_fftm,
            include_versioned=args.include_versioned,
            versioned_full_matrix=args.versioned_full_matrix,
            transports=self.transports,
            modes=self.modes,
            strategies_3d=self.strategies_3d,
            strategies_4d=self.strategies_4d,
        )
        if not self.specs:
            raise RuntimeError("no runnable specifications were selected")

    def detect_local_gpus(self) -> List[Dict[str, Any]]:
        try:
            return detect_gpus()
        except Exception:
            if not self.args.device_memory_mib:
                raise
            return []

    def resolve_local_gpu_counts(self) -> List[int]:
        if self.args.gpu_counts == "auto":
            count = len(self.detected_gpus) if self.detected_gpus else len(self.args.device_memory_mib or [])
            if self.args.max_gpus is not None:
                count = min(count, int(self.args.max_gpus))
            return dense_gpu_counts(count)
        counts = parse_csv_ints(self.args.gpu_counts)
        if self.args.max_gpus is not None:
            counts = [count for count in counts if count <= self.args.max_gpus]
        return [count for count in counts if count >= 1]

    def resolve_slurm_gpu_counts(self) -> List[int]:
        if self.args.gpu_counts == "full-nodes":
            nodes = parse_csv_ints(self.args.node_counts) or [1]
            counts = sorted(set(node_count * self.args.gpus_per_node for node_count in nodes))
            if self.args.max_gpus is not None:
                counts = [count for count in counts if count <= self.args.max_gpus]
            return counts

        explicit = parse_csv_ints(self.args.gpu_counts, allow_auto=True)
        if explicit:
            if self.args.max_gpus is not None:
                explicit = [count for count in explicit if count <= self.args.max_gpus]
            return explicit
        nodes = parse_csv_ints(self.args.node_counts)
        if not nodes:
            nodes = [1]
        max_gpus = max(nodes) * self.args.gpus_per_node
        if self.args.max_gpus is not None:
            max_gpus = min(max_gpus, int(self.args.max_gpus))
        return dense_gpu_counts(max_gpus)

    def effective_gpus(self) -> List[Dict[str, Any]]:
        if self.executor == "local" and self.detected_gpus:
            return self.detected_gpus
        max_gpus = max([1] + self.gpu_counts)
        memory_values = self.args.device_memory_mib or [self.args.default_device_memory_mib]
        if len(memory_values) == 1:
            memory_values = memory_values * max_gpus
        return [
            {
                "index": index,
                "name": self.args.gpu_name,
                "memory_total_mib": int(memory_values[min(index, len(memory_values) - 1)]),
                "memory_total_bytes": mib_to_bytes(int(memory_values[min(index, len(memory_values) - 1)])),
            }
            for index in range(max_gpus)
        ]

    def prepare_output(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cpp_csv_host.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "hardware.json").write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "executor": self.executor,
                    "detected_gpus": self.detected_gpus,
                    "effective_gpus": self.effective_gpus(),
                    "gpu_counts": self.gpu_counts,
                    "gpus_per_node": self.args.gpus_per_node if self.executor == "slurm-pyxis" else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (self.data_dir / "config.json").write_text(
            json.dumps({"created_at": now_iso(), "args": vars(self.args)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        planned = self.plan_runs()
        (self.data_dir / "matrix.json").write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "num_planned_runs": len(planned),
                    "planned_runs": planned,
                    "measurement_specs": [asdict(spec) for spec in self.specs],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def sizes_for_spec(self, spec: RunSpec) -> List[Tuple[int, ...]]:
        if spec.case_name == "benchmark":
            values = self.benchmark_sizes_3d if spec.dim == 3 else self.benchmark_sizes_4d
        else:
            value = self.args.versioned_size_3d if spec.dim == 3 else self.args.versioned_size_4d
            values = [value]
        return [tuple([int(n)] * spec.dim) for n in values]

    def times_for_spec(self, spec: RunSpec) -> int:
        return self.args.benchmark_times if spec.case_name == "benchmark" else self.args.validation_times

    def plan_runs(self) -> List[Dict[str, Any]]:
        planned = []
        for spec in self.specs:
            for sizes in self.sizes_for_spec(spec):
                planned.append({"spec": asdict(spec), "sizes": list(sizes), "times": self.times_for_spec(spec)})
        return planned

    def binary_args(self, spec: RunSpec, sizes: Tuple[int, ...], times: int) -> List[str]:
        args: List[str] = []
        if spec.strategy is not None:
            args.extend(["--strategy", spec.strategy])
        if spec.mode is not None:
            args.extend(["--mode", spec.mode])
        if spec.supports_directory:
            if self.executor == "slurm-pyxis":
                args.extend(["--directory", str(self.container_cpp_csv)])
            else:
                args.extend(["--directory", str(self.cpp_csv_host)])
        if spec.suite == "fftm" and spec.case_name != "benchmark":
            args.extend(["--threshold", str(self.args.validation_threshold)])
        else:
            args.extend(["--epsilon", str(self.args.validation_epsilon)])
        args.extend(["--times", str(times)])
        args.extend(str(size) for size in sizes)
        return args

    def local_command(self, spec: RunSpec, sizes: Tuple[int, ...], times: int) -> List[str]:
        command: List[str] = []
        if spec.uses_mpi:
            command.extend(["mpiexec", "-n", str(spec.num_gpus)])
        command.append(spec.binary_path)
        command.extend(self.binary_args(spec, sizes, times))
        return command

    def slurm_nodes_for_spec(self, spec: RunSpec) -> int:
        if spec.suite == "ffts":
            return 1
        return max(1, int(math.ceil(float(spec.num_gpus) / float(self.args.gpus_per_node))))

    def slurm_command(self, spec: RunSpec, sizes: Tuple[int, ...], times: int) -> List[str]:
        nodes = self.slurm_nodes_for_spec(spec)
        tasks = 1 if spec.suite == "ffts" else spec.num_gpus
        total_gpus = nodes * self.args.gpus_per_node
        mounts = list(self.args.container_mounts)
        data_mount = f"{self.data_dir}:{self.container_data_dir}"
        if not any(mount.split(":", 1)[0] == str(self.data_dir) for mount in mounts):
            mounts.append(data_mount)

        command = ["srun"]
        command.extend(shlex.split(self.args.srun_extra_args))
        command.extend(["-N", str(nodes), "-n", str(tasks), "-G", str(total_gpus)])
        command.append(f"--gpus-per-node={self.args.gpus_per_node}")
        if self.args.srun_time:
            command.append(f"--time={self.args.srun_time}")
        command.extend(["--container-image", self.args.container_image])
        command.append(f"--container-mounts={','.join(mounts)}")
        command.extend(["--container-workdir", self.args.container_workdir])
        if self.args.container_env:
            command.append(f"--container-env={self.args.container_env}")
        if self.args.container_writable:
            command.append("--container-writable")
        command.append("--container-entrypoint")
        command.append(spec.binary_path)
        command.extend(self.binary_args(spec, sizes, times))
        return command

    def command_for_run(self, spec: RunSpec, sizes: Tuple[int, ...], times: int) -> List[str]:
        if self.executor == "local":
            return self.local_command(spec, sizes, times)
        return self.slurm_command(spec, sizes, times)

    def record_run(self, record: Dict[str, Any]) -> None:
        with self.runs_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def execute_run(self, spec: RunSpec, sizes: Tuple[int, ...]) -> Dict[str, Any]:
        times = self.times_for_spec(spec)
        self.run_index += 1
        command = self.command_for_run(spec, sizes, times)
        command_string = " ".join(shlex.quote(arg) for arg in command)
        slug = slugify(
            f"{self.run_index:05d}_measure_{spec.suite}_{spec.case_name}_{spec.dim}d_"
            f"g{spec.num_gpus}_{spec.transport}_{spec.strategy or 'default'}_{spec.mode or 'none'}_"
            f"{'x'.join(str(s) for s in sizes)}"
        )
        raw_path = self.raw_dir / f"{slug}.log"
        started_at = now_iso()
        started_monotonic = time.monotonic()

        if self.args.dry_run:
            stdout = f"DRY RUN: {command_string}\n"
            returncode = 0
            elapsed_seconds = 0.0
        else:
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.tests_root) if self.executor == "local" else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.args.timeout_seconds,
                    check=False,
                )
                stdout = completed.stdout
                returncode = completed.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = (exc.stdout or "") + f"\nTIMEOUT after {self.args.timeout_seconds} seconds\n"
                returncode = 124
            elapsed_seconds = time.monotonic() - started_monotonic

        raw_path.write_text(stdout, encoding="utf-8")
        parsed = parse_run_output(stdout)
        n = sizes[0]
        used_device_peak_max_bytes = infer_device_peak_max_bytes(parsed, spec, n)
        record: Dict[str, Any] = {
            "phase": "measure",
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "returncode": returncode,
            "command": command,
            "command_string": command_string,
            "cwd": str(self.tests_root) if self.executor == "local" else None,
            "raw_log": str(raw_path),
            "spec": asdict(spec),
            "sizes": list(sizes),
            "parsed": parsed,
            "used_device_peak_max_bytes": used_device_peak_max_bytes,
            "used_device_peak_max_mib": bytes_to_mib(used_device_peak_max_bytes)
            if used_device_peak_max_bytes is not None
            else None,
        }
        self.record_run(record)
        print(
            f"[{self.run_index:05d}] rc={returncode} g={spec.num_gpus} {spec.suite} {spec.case_name} "
            f"{spec.dim}D {spec.transport} {spec.strategy or '-'} {spec.mode or '-'} sizes={sizes}",
            flush=True,
        )
        return record

    def run(self) -> int:
        self.prepare_output()
        failed_runs = 0
        planned_runs = 0
        for spec in self.specs:
            for sizes in self.sizes_for_spec(spec):
                planned_runs += 1
                record = self.execute_run(spec, sizes)
                if record["returncode"] != 0:
                    failed_runs += 1
                    if self.args.stop_on_failure:
                        break
            if failed_runs and self.args.stop_on_failure:
                break

        summary = {
            "created_at": now_iso(),
            "executor": self.executor,
            "num_measurement_specs": len(self.specs),
            "planned_runs": planned_runs,
            "failed_measurement_runs": failed_runs,
            "data_directory": str(self.data_dir),
        }
        (self.data_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0 if failed_runs == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executor", choices=("local", "slurm-pyxis"), required=True)
    parser.add_argument("--data-directory", required=True)
    parser.add_argument("--tests-root", default="/opt/fftm/bin", help="Local executor test binary directory.")
    parser.add_argument("--container-tests-root", default="/opt/fftm/bin")
    parser.add_argument("--container-data-directory", default="/data")
    parser.add_argument("--container-image", default="")
    parser.add_argument("--container-workdir", default="/opt/fftm/bin")
    parser.add_argument("--container-mounts", action="append", default=[])
    parser.add_argument("--container-env", default="")
    parser.add_argument("--container-writable", action="store_true")
    parser.add_argument("--srun-extra-args", default="")
    parser.add_argument("--srun-time", default="00:20:00")
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--node-counts", default="1")
    parser.add_argument(
        "--gpu-counts",
        default="auto",
        help=(
            "Comma-separated FFTM GPU counts, 'auto' for dense 2..max sweep, or "
            "'full-nodes' for only node_count*gpus_per_node counts. FFTS supplies the 1-GPU baseline."
        ),
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Optional cap for automatic or explicit FFTM GPU-count sweeps.",
    )
    parser.add_argument("--gpu-name", default="A100")
    parser.add_argument("--default-device-memory-mib", type=int, default=40960)
    parser.add_argument("--device-memory-mib", type=int, nargs="*", default=None)

    parser.add_argument("--benchmark-sizes-3d", default="256,512")
    parser.add_argument("--benchmark-sizes-4d", default="40,64")
    parser.add_argument("--versioned-size-3d", type=int, default=128)
    parser.add_argument("--versioned-size-4d", type=int, default=32)
    parser.add_argument("--benchmark-times", type=int, default=3)
    parser.add_argument("--validation-times", type=int, default=1)
    parser.add_argument("--epsilon", dest="validation_epsilon", default="1.0e-11")
    parser.add_argument("--threshold", dest="validation_threshold", default="1.0e-11")

    parser.add_argument("--transports", default="cuda_aware,non_cuda_aware")
    parser.add_argument("--modes", default="alltoallv,alltoallw,p2p-waitall,p2p-waitany")
    parser.add_argument("--strategies-3d", default="slab-pencil,pencil-slab,pencil-pencil")
    parser.add_argument("--strategies-4d", default="slab-slab,pencil-pencil")
    parser.add_argument("--skip-ffts", action="store_true")
    parser.add_argument("--skip-fftm", action="store_true")
    parser.add_argument("--include-versioned", action="store_true")
    parser.add_argument(
        "--versioned-full-matrix",
        action="store_true",
        help="Run versioned FFTM tests for every selected mode. Default validates all strategies with p2p-waitany only.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.executor == "slurm-pyxis" and not args.container_image:
        parser.error("--container-image is required with --executor slurm-pyxis")
    runner = PaperClusterRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
