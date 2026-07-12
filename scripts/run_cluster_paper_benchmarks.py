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
    FFTM_P2P_VARIANTS,
    FFTM_CONTIGUOUS_FORWARD_SEND_MODES,
    FFTM_LARGE_COUNT_P2P_TRANSPORTS,
    FFTM_3D_BACKENDS,
    FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES,
    FFTM_STRATEGIES_3D,
    FFTM_STRATEGIES_4D,
    FFTS_STRATEGIES_4D,
    RunSpec,
    bytes_to_mib,
    detect_gpus,
    grid_orientations_for_spec,
    grid_slug,
    infer_device_peak_max_bytes,
    mib_to_bytes,
    now_iso,
    parse_run_output,
    p2p_variant_flags,
    p2p_scheduler_flags,
    p2p_schedulers_for_spec,
    p2p_variants_for_spec,
    contiguous_forward_send_modes_for_spec,
    fftm_3d_backends_for_spec,
    large_count_p2p_transports_for_spec,
    pencil_layouts_for_spec,
    pencil_pipelines_for_spec,
    parse_large_count_p2p_transports,
    parse_contiguous_forward_send_modes,
    parse_fftm_3d_backends,
    parse_gpu_size_map,
    parse_native_backward_second_peer_loop_modes,
    parse_native_opt0_y_executor_variants,
    parse_p2p_schedulers,
    parse_pencil_layouts,
    parse_pencil_pipelines,
    parse_p2p_variants,
    parse_pencil_grid_orientations,
    slugify,
    short_log_slug,
    native_backward_second_peer_loop_modes_for_spec,
    native_opt0_y_executor_flags_are_diagnostic,
    native_opt0_y_executor_variant_flags,
    native_opt0_y_executor_variants_for_spec,
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


def parse_size_list_or_auto(value: str) -> Optional[List[int]]:
    value = (value or "").strip()
    if value == "auto":
        return None
    if value in ("none", "off", "skip"):
        return []
    parsed = parse_csv_ints(value)
    if not parsed:
        raise ValueError("size list cannot be empty unless it is 'auto' or 'none'")
    return parsed


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


def parse_native_opt0_y_cross_factory_modes(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value in ("all", "matrix"):
        return list(FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES)
    selected: List[Optional[str]] = []
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in ("single", "configured"):
            selected.append(None)
        elif key in FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES:
            selected.append(key)
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "single", "all", "matrix", *FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES]
        raise ValueError(
            "unknown --native-opt0-y-cross-factory-modes value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def generate_fft_friendly_sizes(limit: int, minimum: int) -> List[int]:
    """Return even 7-smooth sizes up to limit.

    FFT libraries are generally happiest with factorizations over small primes.
    We keep the side length even for real-to-complex transforms.
    """
    if limit < minimum:
        return []
    values = {1}
    for prime in (2, 3, 5, 7):
        next_values = set(values)
        for value in values:
            current = value * prime
            while current <= limit:
                next_values.add(current)
                current *= prime
        values = next_values
    return sorted(value for value in values if minimum <= value <= limit and value % 2 == 0)


def largest_fft_friendly_size(limit: int, minimum: int) -> int:
    sizes = generate_fft_friendly_sizes(limit, minimum)
    if not sizes:
        return max(minimum, limit)
    return sizes[-1]


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
    p2p_variants: Sequence[Optional[str]],
    p2p_schedulers: Sequence[Optional[str]],
    pencil_layouts: Sequence[Optional[str]],
    pencil_pipelines: Sequence[Optional[str]],
    large_count_p2p_transports: Sequence[Optional[str]],
    contiguous_forward_send_modes: Sequence[Optional[str]],
    native_backward_second_peer_loop_modes: Sequence[Optional[bool]],
    native_opt0_y_executor_variants: Sequence[Optional[str]],
    native_opt0_y_cross_factory_modes: Sequence[Optional[str]],
    fftm_3d_backends: Sequence[Optional[str]],
    use_contiguous_forward_byte_send: bool,
    pencil_grid_orientations: str = "both",
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
                                for variant in p2p_variants_for_spec(dim, transport, strategy, mode, p2p_variants):
                                    for scheduler in p2p_schedulers_for_spec(dim, transport, mode, p2p_schedulers):
                                            for pencil_layout in pencil_layouts_for_spec(dim, strategy, pencil_layouts):
                                                for pencil_pipeline in pencil_pipelines_for_spec(
                                                    dim, strategy, mode, pencil_pipelines
                                                ):
                                                    for large_count_transport in large_count_p2p_transports_for_spec(
                                                        dim,
                                                        transport,
                                                        strategy,
                                                        mode,
                                                        pencil_pipeline,
                                                        large_count_p2p_transports,
                                                    ):
                                                        for contiguous_forward_send_mode in contiguous_forward_send_modes_for_spec(
                                                            dim,
                                                            transport,
                                                            strategy,
                                                            mode,
                                                            use_contiguous_forward_byte_send,
                                                            contiguous_forward_send_modes,
                                                        ):
                                                            for grid in grid_orientations_for_spec(
                                                                dim,
                                                                strategy,
                                                                num_gpus,
                                                                pencil_grid_orientations,
                                                                pencil_layout,
                                                            ):
                                                                for fftm_3d_backend in fftm_3d_backends_for_spec(
                                                                    dim,
                                                                    transport,
                                                                    strategy,
                                                                    mode,
                                                                    "benchmark",
                                                                    grid,
                                                                    fftm_3d_backends,
                                                                ):
                                                                    for native_backward_second_peer_loop in native_backward_second_peer_loop_modes_for_spec(
                                                                        dim,
                                                                        transport,
                                                                        strategy,
                                                                        mode,
                                                                        pencil_pipeline,
                                                                        fftm_3d_backend,
                                                                        native_backward_second_peer_loop_modes,
                                                                    ):
                                                                        for native_opt0_y_executor_variant in native_opt0_y_executor_variants_for_spec(
                                                                            dim,
                                                                            transport,
                                                                            strategy,
                                                                            mode,
                                                                            pencil_layout,
                                                                            pencil_pipeline,
                                                                            fftm_3d_backend,
                                                                            native_opt0_y_executor_variants,
                                                                        ):
                                                                            factory_modes = native_opt0_y_cross_factory_modes
                                                                            if native_opt0_y_cross_factory_modes != [None]:
                                                                                if not (
                                                                                    dim == 3
                                                                                    and transport == "cuda_aware"
                                                                                    and strategy == "pencil-pencil"
                                                                                    and mode.startswith("p2p-")
                                                                                    and pencil_layout == "opt0"
                                                                                    and pencil_pipeline in ("reference", "reference-parity")
                                                                                    and fftm_3d_backend in (None, "native")
                                                                                ):
                                                                                    factory_modes = [None]
                                                                            for native_opt0_y_cross_factory_mode in factory_modes:
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
                                                                                        memory_family=f"fftm:{dim}d:benchmark:{num_gpus}:{transport}:{strategy}:{fftm_3d_backend or 'configured'}:{native_opt0_y_executor_variant or 'y-configured'}:{native_opt0_y_cross_factory_mode or 'factory-configured'}",
                                                                                        p2p_variant=variant,
                                                                                        p2p_scheduler=scheduler,
                                                                                        pencil_layout=pencil_layout,
                                                                                        pencil_pipeline=pencil_pipeline,
                                                                                        large_count_p2p_transport=large_count_transport,
                                                                                        contiguous_forward_send_mode=contiguous_forward_send_mode,
                                                                                        native_backward_second_peer_loop=native_backward_second_peer_loop,
                                                                                        fftm_3d_backend=fftm_3d_backend,
                                                                                        native_opt0_y_executor_variant=native_opt0_y_executor_variant,
                                                                                        native_opt0_y_cross_factory_mode=native_opt0_y_cross_factory_mode,
                                                                                        grid=grid,
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
                                    for variant in p2p_variants_for_spec(dim, transport, strategy, mode, p2p_variants):
                                        for scheduler in p2p_schedulers_for_spec(dim, transport, mode, p2p_schedulers):
                                            for pencil_layout in pencil_layouts_for_spec(dim, strategy, pencil_layouts):
                                                for pencil_pipeline in pencil_pipelines_for_spec(
                                                    dim, strategy, mode, pencil_pipelines
                                                ):
                                                    for large_count_transport in large_count_p2p_transports_for_spec(
                                                        dim,
                                                        transport,
                                                        strategy,
                                                        mode,
                                                        pencil_pipeline,
                                                        large_count_p2p_transports,
                                                    ):
                                                        for grid in grid_orientations_for_spec(
                                                            dim,
                                                            strategy,
                                                            num_gpus,
                                                            pencil_grid_orientations,
                                                            pencil_layout,
                                                        ):
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
                                                                    p2p_variant=variant,
                                                                    p2p_scheduler=scheduler,
                                                                    pencil_layout=pencil_layout,
                                                                    pencil_pipeline=pencil_pipeline,
                                                                    large_count_p2p_transport=large_count_transport,
                                                                    grid=grid,
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
        self.p2p_variants = parse_p2p_variants(args.p2p_variants)
        self.p2p_schedulers = parse_p2p_schedulers(args.p2p_schedulers)
        self.pencil_layouts = parse_pencil_layouts(args.pencil_layouts)
        self.pencil_pipelines = parse_pencil_pipelines(args.pencil_pipelines)
        self.large_count_p2p_transports = parse_large_count_p2p_transports(args.large_count_p2p_transports)
        self.contiguous_forward_send_modes = parse_contiguous_forward_send_modes(
            args.contiguous_forward_send_modes
        )
        self.native_backward_second_peer_loop_modes = parse_native_backward_second_peer_loop_modes(
            args.native_backward_second_peer_loop_modes
        )
        self.native_opt0_y_executor_variants = parse_native_opt0_y_executor_variants(
            args.native_opt0_y_executor_variants
        )
        self.native_opt0_y_cross_factory_modes = (
            parse_native_opt0_y_cross_factory_modes(args.native_opt0_y_cross_factory_modes)
            if args.native_opt0_y_cross_microbench
            else [None]
        )
        self.fftm_3d_backends = parse_fftm_3d_backends(args.fftm_3d_backends)
        self.pencil_grid_orientations = parse_pencil_grid_orientations(args.pencil_pencil_grid_orientations)
        self.extra_sizes_3d = parse_csv_ints(args.extra_sizes_3d)
        self.extra_sizes_3d_by_gpu = parse_gpu_size_map(args.extra_sizes_3d_by_gpu)
        self.fixed_scaling_sizes_3d = parse_csv_ints(args.fixed_scaling_sizes_3d)
        self.fixed_scaling_sizes_4d = parse_csv_ints(args.fixed_scaling_sizes_4d)
        self.benchmark_sizes_3d = parse_size_list_or_auto(args.benchmark_sizes_3d)
        self.benchmark_sizes_4d = parse_size_list_or_auto(args.benchmark_sizes_4d)
        self.size_plan = self.build_size_plan()

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
            p2p_variants=self.p2p_variants,
            p2p_schedulers=self.p2p_schedulers,
            pencil_layouts=self.pencil_layouts,
            pencil_pipelines=self.pencil_pipelines,
            large_count_p2p_transports=self.large_count_p2p_transports,
            contiguous_forward_send_modes=self.contiguous_forward_send_modes,
            native_backward_second_peer_loop_modes=self.native_backward_second_peer_loop_modes,
            native_opt0_y_executor_variants=self.native_opt0_y_executor_variants,
            native_opt0_y_cross_factory_modes=self.native_opt0_y_cross_factory_modes,
            fftm_3d_backends=self.fftm_3d_backends,
            use_contiguous_forward_byte_send=args.use_contiguous_forward_byte_send,
            pencil_grid_orientations=self.pencil_grid_orientations,
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

    def per_rank_memory_total_bytes(self, num_gpus: int) -> int:
        effective = self.effective_gpus()
        selected = effective[: max(1, min(num_gpus, len(effective)))]
        return min(gpu["memory_total_bytes"] for gpu in selected)

    def per_rank_memory_target_bytes(self, num_gpus: int) -> int:
        total = self.per_rank_memory_total_bytes(num_gpus)
        target = int(total * self.args.auto_memory_fraction)
        if self.args.auto_reserve_memory_mib > 0:
            reserve_cap = total - mib_to_bytes(self.args.auto_reserve_memory_mib)
            target = min(target, reserve_cap)
        return max(0, target)

    def effective_auto_bytes_per_point(self, dim: int) -> float:
        reference_size = self.args.auto_reference_size_3d if dim == 3 else self.args.auto_reference_size_4d
        if reference_size is None:
            return self.args.auto_bytes_per_point_3d if dim == 3 else self.args.auto_bytes_per_point_4d
        target = self.per_rank_memory_target_bytes(1)
        if target <= 0:
            return self.args.auto_bytes_per_point_3d if dim == 3 else self.args.auto_bytes_per_point_4d
        return float(target) / float(int(reference_size) ** dim)

    def max_auto_n(self, dim: int, num_gpus: int) -> int:
        bytes_per_point = self.effective_auto_bytes_per_point(dim)
        target = self.per_rank_memory_target_bytes(num_gpus)
        if target <= 0 or bytes_per_point <= 0:
            return self.args.auto_min_size_3d if dim == 3 else self.args.auto_min_size_4d
        raw = int(math.floor(((float(target) * float(num_gpus)) / float(bytes_per_point)) ** (1.0 / float(dim))))
        maximum = self.args.auto_max_size_3d if dim == 3 else self.args.auto_max_size_4d
        minimum = self.args.auto_min_size_3d if dim == 3 else self.args.auto_min_size_4d
        if maximum is not None:
            raw = min(raw, int(maximum))
        return largest_fft_friendly_size(raw, minimum)

    def estimated_per_rank_memory_bytes(self, dim: int, num_gpus: int, side_length: int) -> int:
        bytes_per_point = self.effective_auto_bytes_per_point(dim)
        total_points = float(int(side_length) ** dim)
        return int(math.ceil(total_points * bytes_per_point / float(max(1, num_gpus))))

    def build_size_plan(self) -> Dict[str, Any]:
        gpu_counts = sorted(set([1] + list(self.gpu_counts)))
        plan: Dict[str, Any] = {
            "mode_3d": "auto" if self.benchmark_sizes_3d is None else "explicit",
            "mode_4d": "auto" if self.benchmark_sizes_4d is None else "explicit",
            "assumptions": {
                "auto_memory_fraction": self.args.auto_memory_fraction,
                "auto_reserve_memory_mib": self.args.auto_reserve_memory_mib,
                "auto_bytes_per_point_3d": self.args.auto_bytes_per_point_3d,
                "auto_bytes_per_point_4d": self.args.auto_bytes_per_point_4d,
                "auto_reference_size_3d": self.args.auto_reference_size_3d,
                "auto_reference_size_4d": self.args.auto_reference_size_4d,
                "auto_min_size_3d": self.args.auto_min_size_3d,
                "auto_min_size_4d": self.args.auto_min_size_4d,
                "auto_max_size_3d": self.args.auto_max_size_3d,
                "auto_max_size_4d": self.args.auto_max_size_4d,
                "extra_sizes_3d": self.extra_sizes_3d,
                "extra_sizes_3d_by_gpu": self.extra_sizes_3d_by_gpu,
                "fixed_scaling_sizes_3d": self.fixed_scaling_sizes_3d,
                "fixed_scaling_sizes_4d": self.fixed_scaling_sizes_4d,
                "fft_friendly": "largest even 7-smooth N below the memory target",
            },
            "sizes": {},
        }
        for dim in (3, 4):
            dim_key = f"{dim}d"
            explicit = self.benchmark_sizes_3d if dim == 3 else self.benchmark_sizes_4d
            plan["sizes"][dim_key] = {}
            for gpu_count in gpu_counts:
                if explicit is None:
                    values = [self.max_auto_n(dim, gpu_count)]
                    mode = "auto"
                else:
                    values = list(explicit)
                    mode = "none" if not values else "explicit"
                if dim == 3:
                    values = sorted(set(values + self.extra_sizes_3d))
                    values = sorted(set(values + self.extra_sizes_3d_by_gpu.get(gpu_count, [])))
                fixed_values = self.fixed_scaling_sizes_3d if dim == 3 else self.fixed_scaling_sizes_4d
                estimate_values = values + list(fixed_values)
                if estimate_values:
                    estimated = self.estimated_per_rank_memory_bytes(dim, gpu_count, max(estimate_values))
                else:
                    estimated = 0
                target = self.per_rank_memory_target_bytes(gpu_count)
                plan["sizes"][dim_key][str(gpu_count)] = {
                    "mode": mode,
                    "num_gpus": gpu_count,
                    "per_rank_total_memory_mib": bytes_to_mib(self.per_rank_memory_total_bytes(gpu_count)),
                    "per_rank_target_memory_mib": bytes_to_mib(target),
                    "effective_bytes_per_point": self.effective_auto_bytes_per_point(dim),
                    "estimated_per_rank_memory_mib": bytes_to_mib(estimated),
                    "estimated_target_ratio": (float(estimated) / float(target)) if target > 0 else 0.0,
                    "side_lengths": values,
                    "fixed_scaling_side_lengths": (
                        self.fixed_scaling_sizes_3d if dim == 3 else self.fixed_scaling_sizes_4d
                    ),
                }
        return plan

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
        (self.data_dir / "size_plan.json").write_text(
            json.dumps({"created_at": now_iso(), **self.size_plan}, indent=2, sort_keys=True) + "\n",
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
            dim_key = f"{spec.dim}d"
            gpu_key = str(spec.num_gpus)
            values = self.size_plan["sizes"][dim_key][gpu_key]["side_lengths"]
            if spec.suite == "fftm":
                fixed_values = self.size_plan["sizes"][dim_key][gpu_key].get("fixed_scaling_side_lengths", [])
                values = sorted(set(list(values) + list(fixed_values)))
        else:
            value = self.args.versioned_size_3d if spec.dim == 3 else self.args.versioned_size_4d
            values = [value]
        return [tuple([int(n)] * spec.dim) for n in values]

    def times_for_spec(self, spec: RunSpec) -> int:
        return self.args.benchmark_times if spec.case_name == "benchmark" else self.args.validation_times

    def warmup_for_spec(self, spec: RunSpec) -> int:
        return self.args.test_warmup

    def plan_runs(self) -> List[Dict[str, Any]]:
        planned = []
        for spec in self.specs:
            for sizes in self.sizes_for_spec(spec):
                planned.append(
                    {
                        "spec": asdict(spec),
                        "sizes": list(sizes),
                        "times": self.times_for_spec(spec),
                        "warmup": self.warmup_for_spec(spec),
                    }
                )
        return planned

    def binary_args(self, spec: RunSpec, sizes: Tuple[int, ...], times: int) -> List[str]:
        args: List[str] = []
        if spec.strategy is not None:
            args.extend(["--strategy", spec.strategy])
        if spec.mode is not None:
            args.extend(["--mode", spec.mode])
        if spec.grid is not None:
            args.extend(["--grid", *(str(v) for v in spec.grid)])
        if spec.supports_directory:
            if self.executor == "slurm-pyxis":
                args.extend(["--directory", str(self.container_cpp_csv)])
            else:
                args.extend(["--directory", str(self.cpp_csv_host)])
        if spec.suite == "fftm" and spec.case_name != "benchmark":
            args.extend(["--threshold", str(self.args.validation_threshold)])
        else:
            args.extend(["--epsilon", str(self.args.validation_epsilon)])
        if spec.suite == "fftm":
            use_direct_backward_receive, direct_p2p_cuda_aware, use_p2p_byte_transfer = p2p_variant_flags(
                spec.p2p_variant,
                self.args.use_direct_backward_receive,
                self.args.direct_p2p_cuda_aware,
                self.args.use_p2p_byte_transfer,
            )
            args.append(
                "--use-direct-backward-receive"
                if use_direct_backward_receive
                else "--no-direct-backward-receive"
            )
            args.append(
                "--direct-p2p-cuda-aware"
                if direct_p2p_cuda_aware
                else "--no-direct-p2p-cuda-aware"
            )
            args.append(
                "--use-fft-exec-no-sync"
                if self.args.use_fft_exec_no_sync
                else "--no-fft-exec-no-sync"
            )
            if spec.dim == 3:
                if self.args.fftm_autotune_config:
                    args.extend(["--autotune-config", self.args.fftm_autotune_config])
                use_p2p_send_thread, use_persistent_p2p = p2p_scheduler_flags(
                    spec.p2p_scheduler,
                    self.args.use_p2p_send_thread,
                    self.args.use_persistent_p2p,
                )
                args.append(
                    "--use-p2p-send-thread" if use_p2p_send_thread else "--no-p2p-send-thread"
                )
                args.append(
                    "--use-p2p-byte-transfer"
                    if use_p2p_byte_transfer
                    else "--no-p2p-byte-transfer"
                )
                args.append("--use-persistent-p2p" if use_persistent_p2p else "--no-persistent-p2p")
                args.append(
                    "--use-ready-p2p-send"
                    if self.args.use_ready_p2p_send
                    else "--no-ready-p2p-send"
                )
                args.append(
                    "--print-pencil-schedule"
                    if self.args.print_pencil_schedule
                    else "--no-print-pencil-schedule"
                )
                args.append(
                    "--use-direct-forward-byte-receive"
                    if self.args.use_direct_forward_byte_receive
                    else "--no-direct-forward-byte-receive"
                )
                args.append(
                    "--use-stable-forward-byte-send-buffer"
                    if self.args.use_stable_forward_byte_send_buffer
                    else "--no-stable-forward-byte-send-buffer"
                )
                args.append(
                    "--use-ready-stable-forward-byte-send-buffer"
                    if self.args.use_ready_stable_forward_byte_send_buffer
                    else "--no-ready-stable-forward-byte-send-buffer"
                )
                args.append(
                    "--use-contiguous-forward-byte-send"
                    if self.args.use_contiguous_forward_byte_send
                    else "--no-contiguous-forward-byte-send"
                )
                args.append(
                    "--use-physical-forward-peer-exchange"
                    if self.args.use_physical_forward_peer_exchange
                    else "--no-physical-forward-peer-exchange"
                )
                if spec.native_backward_second_peer_loop is None:
                    args.append(
                        "--use-native-backward-second-peer-loop"
                        if self.args.use_native_backward_second_peer_loop
                        else "--no-native-backward-second-peer-loop"
                    )
                else:
                    args.append(
                        "--use-native-backward-second-peer-loop"
                        if spec.native_backward_second_peer_loop
                        else "--no-native-backward-second-peer-loop"
                    )
                native_opt0_y_flags = native_opt0_y_executor_variant_flags(
                    spec.native_opt0_y_executor_variant,
                    default_tight=self.args.use_native_opt0_tight_y_plan_sequence,
                    default_shared=self.args.use_native_opt0_shared_y_plan_handles,
                    default_device_sync=self.args.use_native_opt0_y_group_device_sync,
                    default_opaque=self.args.use_native_opt0_raw_y_plan_array_executor,
                    default_reference_lifecycle=self.args.use_native_opt0_reference_y_plan_lifecycle,
                    default_reference_bundle=self.args.use_native_opt0_reference_y_plan_bundle,
                    default_raw_bundle=self.args.use_native_opt0_raw_y_plan_bundle,
                    default_stream_first=self.args.use_native_opt0_y_plan_bundle_stream_first,
                    default_reference_streams=self.args.use_native_opt0_raw_y_plan_bundle_reference_streams,
                    default_local_context=self.args.use_native_opt0_reference_local_plan_context,
                    default_no_sync_exec=self.args.use_native_opt0_y_no_sync_exec,
                )
                args.append(
                    "--use-native-opt0-default-z-layout"
                    if self.args.use_native_opt0_default_z_layout
                    else "--no-native-opt0-default-z-layout"
                )
                args.append(
                    "--use-native-opt0-reference-y-buffer-topology"
                    if self.args.use_native_opt0_reference_y_buffer_topology
                    else "--no-native-opt0-reference-y-buffer-topology"
                )
                args.append(
                    "--use-native-opt0-compact-y-workarea"
                    if self.args.use_native_opt0_compact_y_workarea
                    else "--no-native-opt0-compact-y-workarea"
                )
                args.append(
                    "--use-native-opt0-tight-y-plan-sequence"
                    if native_opt0_y_flags["tight"]
                    else "--no-native-opt0-tight-y-plan-sequence"
                )
                args.append(
                    "--use-native-opt0-shared-y-plan-handles"
                    if native_opt0_y_flags["shared"]
                    else "--no-native-opt0-shared-y-plan-handles"
                )
                args.append(
                    "--use-native-opt0-y-group-device-sync"
                    if native_opt0_y_flags["device_sync"]
                    else "--no-native-opt0-y-group-device-sync"
                )
                args.append(
                    "--use-native-opt0-y-no-sync-exec"
                    if native_opt0_y_flags["no_sync_exec"]
                    else "--no-native-opt0-y-no-sync-exec"
                )
                args.append(
                    "--use-native-opt0-raw-y-plan-array-executor"
                    if native_opt0_y_flags["opaque"]
                    else "--no-native-opt0-raw-y-plan-array-executor"
                )
                args.append(
                    "--use-native-opt0-reference-y-plan-lifecycle"
                    if native_opt0_y_flags["reference_lifecycle"]
                    else "--no-native-opt0-reference-y-plan-lifecycle"
                )
                args.append(
                    "--use-native-opt0-reference-y-plan-bundle"
                    if native_opt0_y_flags["reference_bundle"]
                    else "--no-native-opt0-reference-y-plan-bundle"
                )
                args.append(
                    "--use-native-opt0-raw-y-plan-bundle"
                    if native_opt0_y_flags["raw_bundle"]
                    else "--no-native-opt0-raw-y-plan-bundle"
                )
                args.append(
                    "--use-native-opt0-y-plan-bundle-stream-first"
                    if native_opt0_y_flags["stream_first"]
                    else "--no-native-opt0-y-plan-bundle-stream-first"
                )
                args.append(
                    "--use-native-opt0-raw-y-plan-bundle-reference-streams"
                    if native_opt0_y_flags["reference_streams"]
                    else "--no-native-opt0-raw-y-plan-bundle-reference-streams"
                )
                args.append(
                    "--use-native-opt0-reference-local-plan-context"
                    if native_opt0_y_flags["local_context"]
                    else "--no-native-opt0-reference-local-plan-context"
                )
                allow_native_opt0_diagnostics = self.args.allow_native_opt0_diagnostic_variants
                args.append(
                    "--allow-native-opt0-diagnostic-variants"
                    if allow_native_opt0_diagnostics
                    else "--no-native-opt0-diagnostic-variants"
                )
                args.append(
                    "--use-native-opt0-memory-feasibility-guard"
                    if self.args.use_native_opt0_memory_feasibility_guard
                    else "--no-native-opt0-memory-feasibility-guard"
                )
                args.extend(
                    [
                        "--native-opt0-memory-feasibility-reserve-mib",
                        str(self.args.native_opt0_memory_feasibility_reserve_mib),
                    ]
                )
                if spec.contiguous_forward_send_mode is not None:
                    args.extend(["--contiguous-forward-send-mode", spec.contiguous_forward_send_mode])
                else:
                    args.extend(["--contiguous-forward-send-mode", self.args.contiguous_forward_send_mode])
                args.extend(
                    [
                        "--contiguous-forward-send-chunk-mib",
                        str(self.args.contiguous_forward_send_chunk_mib),
                    ]
                )
                if spec.case_name == "benchmark":
                    args.extend(
                        [
                            "--contiguous-forward-send-registration-warmups",
                            str(self.args.contiguous_forward_send_registration_warmups),
                        ]
                    )
                if spec.pencil_layout is not None:
                    args.extend(["--pencil-layout", spec.pencil_layout])
                if spec.pencil_pipeline is not None:
                    args.extend(["--pencil-pipeline", spec.pencil_pipeline])
                if spec.large_count_p2p_transport is not None:
                    args.extend(["--large-count-p2p-transport", spec.large_count_p2p_transport])
                if spec.fftm_3d_backend is not None:
                    args.extend(["--fftm-3d-backend", spec.fftm_3d_backend])
                args.append(
                    "--enable-fftm3d-backend-stage-timers"
                    if self.args.enable_fftm3d_backend_stage_timers
                    else "--disable-fftm3d-backend-stage-timers"
                )
                args.append(
                    "--enable-local-fft-diagnostics"
                    if self.args.enable_local_fft_diagnostics
                    else "--disable-local-fft-diagnostics"
                )
                if self.args.native_opt0_y_microbench:
                    args.extend(
                        [
                            "--native-opt0-y-microbench",
                            "--native-opt0-y-microbench-iterations",
                            str(self.args.native_opt0_y_microbench_iterations),
                            "--native-opt0-y-microbench-warmup",
                            str(self.args.native_opt0_y_microbench_warmup),
                        ]
                    )
                if self.args.native_opt0_y_cross_microbench:
                    args.extend(
                        [
                            "--native-opt0-y-cross-microbench",
                            "--native-opt0-y-cross-factory-mode",
                            spec.native_opt0_y_cross_factory_mode or "single",
                            "--native-opt0-y-microbench-iterations",
                            str(self.args.native_opt0_y_microbench_iterations),
                            "--native-opt0-y-microbench-warmup",
                            str(self.args.native_opt0_y_microbench_warmup),
                        ]
                    )
                args.append(
                    "--enable-native-stage-timers"
                    if self.args.enable_native_stage_timers
                    else "--disable-native-stage-timers"
                )
                if self.args.write_native_pencil_schedule:
                    args.append("--write-native-pencil-schedule")
                if self.args.native_pencil_reference_dir:
                    args.extend(
                        [
                            "--check-native-pencil-reference-dir",
                            self.args.native_pencil_reference_dir,
                        ]
                    )
                if self.args.native_pencil_schedule_check_only:
                    args.append("--native-pencil-schedule-check-only")
                if self.args.skip_native_pencil_rank_device_check:
                    args.append("--skip-native-pencil-rank-device-check")
                args.append(
                    "--use-large-count-datatype-cache"
                    if self.args.use_large_count_datatype_cache
                    else "--no-large-count-datatype-cache"
                )
                args.append(
                    "--enable-gpu-telemetry"
                    if self.args.enable_gpu_telemetry
                    else "--disable-gpu-telemetry"
                )
        args.extend(["--times", str(times)])
        warmup = self.warmup_for_spec(spec)
        if warmup > 0:
            args.extend(["--warmup", str(warmup)])
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

    @staticmethod
    def subprocess_output_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def execute_run(self, spec: RunSpec, sizes: Tuple[int, ...]) -> Dict[str, Any]:
        times = self.times_for_spec(spec)
        self.run_index += 1
        command = self.command_for_run(spec, sizes, times)
        command_string = " ".join(shlex.quote(arg) for arg in command)
        slug = short_log_slug(
            f"{self.run_index:05d}_measure_{spec.suite}_{spec.case_name}_{spec.dim}d_"
            f"g{spec.num_gpus}_{spec.transport}_{spec.strategy or 'default'}_{spec.mode or 'none'}_"
            f"{spec.p2p_variant or 'configured'}_{spec.p2p_scheduler or 'sched-configured'}_"
            f"{spec.pencil_layout or 'layout-configured'}_{spec.pencil_pipeline or 'pipe-configured'}_"
            f"{spec.large_count_p2p_transport or 'large-configured'}_"
            f"{spec.contiguous_forward_send_mode or 'contig-configured'}_"
            f"bwd2peer-{('on' if spec.native_backward_second_peer_loop else 'off') if spec.native_backward_second_peer_loop is not None else 'configured'}_"
            f"{spec.fftm_3d_backend or 'backend-configured'}_"
            f"{spec.native_opt0_y_executor_variant or 'y-configured'}_"
            f"{spec.native_opt0_y_cross_factory_mode or 'factory-configured'}_"
            f"{grid_slug(spec.grid)}_{'x'.join(str(s) for s in sizes)}"
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
                stdout = self.subprocess_output_text(exc.stdout)
                stderr = self.subprocess_output_text(exc.stderr)
                if stderr:
                    stdout += stderr
                stdout += f"\nTIMEOUT after {self.args.timeout_seconds} seconds\n"
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
            "times": times,
            "warmup": self.warmup_for_spec(spec),
            "parsed": parsed,
            "used_device_peak_max_bytes": used_device_peak_max_bytes,
            "used_device_peak_max_mib": bytes_to_mib(used_device_peak_max_bytes)
            if used_device_peak_max_bytes is not None
            else None,
        }
        self.record_run(record)
        print(
            f"[{self.run_index:05d}] rc={returncode} g={spec.num_gpus} {spec.suite} {spec.case_name} "
            f"{spec.dim}D {spec.transport} {spec.strategy or '-'} {spec.mode or '-'} "
            f"{spec.p2p_variant or 'configured'} {spec.p2p_scheduler or 'sched-configured'} "
            f"{spec.pencil_layout or 'layout-configured'} {spec.pencil_pipeline or 'pipe-configured'} "
            f"{spec.fftm_3d_backend or 'backend-configured'} "
            f"yexec={spec.native_opt0_y_executor_variant or 'configured'} "
            f"yfactory={spec.native_opt0_y_cross_factory_mode or 'configured'} "
            f"{spec.large_count_p2p_transport or 'large-configured'} "
            f"{spec.contiguous_forward_send_mode or 'contig-configured'} "
            f"bwd2peer={('on' if spec.native_backward_second_peer_loop else 'off') if spec.native_backward_second_peer_loop is not None else 'configured'} "
            f"{grid_slug(spec.grid)} sizes={sizes}",
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

    parser.add_argument(
        "--benchmark-sizes-3d",
        default="auto",
        help=(
            "Comma-separated 3D benchmark side lengths, 'auto' for per-GPU-count FFT-friendly fitting, "
            "or 'none' to run only fixed scaling/versioned sizes. Default: auto"
        ),
    )
    parser.add_argument(
        "--benchmark-sizes-4d",
        default="auto",
        help=(
            "Comma-separated 4D benchmark side lengths, 'auto' for per-GPU-count FFT-friendly fitting, "
            "or 'none' to run only fixed scaling/versioned sizes. Default: auto"
        ),
    )
    parser.add_argument(
        "--auto-memory-fraction",
        type=float,
        default=0.72,
        help="Fraction of per-GPU memory used by automatic benchmark size fitting. Default: 0.72",
    )
    parser.add_argument(
        "--auto-reserve-memory-mib",
        type=int,
        default=2048,
        help=(
            "Hard per-GPU reserve cap for automatic fitting. The target is device_memory*auto_memory_fraction, "
            "capped to device_memory-reserve. Default: 2048"
        ),
    )
    parser.add_argument(
        "--auto-bytes-per-point-3d",
        type=float,
        default=48.0,
        help="Conservative per-global-point device-memory coefficient for 3D fitting. Default: 48",
    )
    parser.add_argument(
        "--auto-bytes-per-point-4d",
        type=float,
        default=40.0,
        help="Conservative per-global-point device-memory coefficient for 4D fitting. Default: 40",
    )
    parser.add_argument(
        "--auto-reference-size-3d",
        type=int,
        default=None,
        help=(
            "Optional measured 1-GPU maximum 3D side length. When set, the runner infers the effective "
            "bytes-per-point from device_memory*auto_memory_fraction and scales that size with GPU count."
        ),
    )
    parser.add_argument(
        "--auto-reference-size-4d",
        type=int,
        default=None,
        help=(
            "Optional measured 1-GPU maximum 4D side length. When set, the runner infers the effective "
            "bytes-per-point from device_memory*auto_memory_fraction and scales that size with GPU count."
        ),
    )
    parser.add_argument("--auto-min-size-3d", type=int, default=64)
    parser.add_argument("--auto-min-size-4d", type=int, default=16)
    parser.add_argument("--auto-max-size-3d", type=int, default=None)
    parser.add_argument("--auto-max-size-4d", type=int, default=None)
    parser.add_argument("--versioned-size-3d", type=int, default=128)
    parser.add_argument("--versioned-size-4d", type=int, default=32)
    parser.add_argument("--benchmark-times", type=int, default=3)
    parser.add_argument(
        "--warmup",
        "--benchmark-warmup",
        dest="test_warmup",
        type=int,
        default=3,
        help="Untimed warmup iterations for every scheduled test binary. Default: 3",
    )
    parser.add_argument(
        "--use-direct-backward-receive",
        action="store_true",
        default=False,
        help="Enable optional CUDA-aware direct backward peer receives in FFTM where supported. Default: disabled",
    )
    parser.add_argument(
        "--no-direct-backward-receive",
        action="store_false",
        dest="use_direct_backward_receive",
        help="Disable optional CUDA-aware direct backward peer receives.",
    )
    parser.add_argument(
        "--direct-p2p-cuda-aware",
        action="store_true",
        default=True,
        help="Allow direct CUDA-aware peer receive targets in FFTM p2p paths. Default: enabled",
    )
    parser.add_argument(
        "--no-direct-p2p-cuda-aware",
        action="store_false",
        dest="direct_p2p_cuda_aware",
        help="Force CUDA-aware FFTM p2p paths through packed receive buffers.",
    )
    parser.add_argument(
        "--use-p2p-send-thread",
        action="store_true",
        default=False,
        help="Enable MPI sender-thread posting for optimized FFTM 3D p2p paths when MPI_THREAD_MULTIPLE is available. Default: disabled",
    )
    parser.add_argument(
        "--no-p2p-send-thread",
        action="store_false",
        dest="use_p2p_send_thread",
        help="Disable MPI sender-thread posting for optimized FFTM 3D p2p paths.",
    )
    parser.add_argument(
        "--use-p2p-byte-transfer",
        action="store_true",
        default=False,
        help="Use MPI_BYTE chunked peer transfers for optimized FFTM 3D CUDA-aware p2p paths. Default: disabled",
    )
    parser.add_argument(
        "--no-p2p-byte-transfer",
        action="store_false",
        dest="use_p2p_byte_transfer",
        help="Disable MPI_BYTE chunked peer transfers.",
    )
    parser.add_argument(
        "--use-direct-forward-byte-receive",
        action="store_true",
        default=False,
        help=(
            "Experimental: receive reference-parity opt1 forward byte messages directly into strided "
            "stage storage. Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-direct-forward-byte-receive",
        action="store_false",
        dest="use_direct_forward_byte_receive",
        help="Disable experimental direct strided forward byte receives.",
    )
    parser.add_argument(
        "--fftm-autotune-config",
        default="",
        help=(
            "Path passed to FFTM 3D binaries as --autotune-config. For Slurm/Pyxis this must be a path "
            "visible inside the container, for example /data/fftm_8g_2048.env."
        ),
    )
    parser.add_argument(
        "--p2p-variants",
        default="configured",
        help=(
            "3D CUDA-aware p2p variant matrix: configured, all, or comma-separated subset of "
            f"{','.join(FFTM_P2P_VARIANTS)}. Default: configured"
        ),
    )
    parser.add_argument(
        "--p2p-schedulers",
        default="configured",
        help=(
            "3D CUDA-aware p2p scheduler matrix: configured, all, or comma-separated subset of "
            "main,send-thread,persistent,send-thread-persistent. Default: configured"
        ),
    )
    parser.add_argument(
        "--use-persistent-p2p",
        action="store_true",
        default=False,
        help="Use persistent MPI send requests for supported optimized FFTM 3D p2p paths. Default: disabled",
    )
    parser.add_argument(
        "--no-persistent-p2p",
        action="store_false",
        dest="use_persistent_p2p",
        help="Disable persistent MPI send requests.",
    )
    parser.add_argument(
        "--use-ready-p2p-send",
        action="store_true",
        default=False,
        help="Use experimental ready-polled P2P send posting in the owned reference 3D pencil path. Default: disabled",
    )
    parser.add_argument(
        "--no-ready-p2p-send",
        action="store_false",
        dest="use_ready_p2p_send",
        help="Disable experimental ready-polled P2P send posting.",
    )
    parser.add_argument(
        "--print-pencil-schedule",
        action="store_true",
        default=False,
        help="Print owned reference pencil-pipeline peer schedules during initialization. Default: disabled",
    )
    parser.add_argument(
        "--no-print-pencil-schedule",
        action="store_false",
        dest="print_pencil_schedule",
        help="Disable owned reference pencil-pipeline schedule dumps.",
    )
    parser.add_argument(
        "--write-native-pencil-schedule",
        action="store_true",
        default=False,
        help=(
            "Ask the 3D FFTM benchmark to write native FFTM pencil plan, schedule, and rank-device CSVs. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--native-pencil-reference-dir",
        default="",
        help=(
            "Container-visible directory containing frozen pencil_schedule.csv and rank_device_map.csv. "
            "When set, the 3D FFTM benchmark compares the native schedule against this reference."
        ),
    )
    parser.add_argument(
        "--native-pencil-schedule-check-only",
        action="store_true",
        default=False,
        help=(
            "Exit the 3D FFTM benchmark after native pencil schedule checking/writing. "
            "This is intended for diagnostics and does not run the FFT."
        ),
    )
    parser.add_argument(
        "--skip-native-pencil-rank-device-check",
        action="store_true",
        default=False,
        help=(
            "Skip rank-device map comparison in native pencil schedule checks. "
            "Use this only for local wrapped-MPI diagnostics."
        ),
    )
    parser.add_argument(
        "--use-stable-forward-byte-send-buffer",
        action="store_true",
        default=False,
        help=(
            "Copy forward reference-parity CUDA-aware byte sends into the stable communication buffer before MPI_Isend. "
            "Default: disabled"
        ),
    )
    parser.add_argument(
        "--no-stable-forward-byte-send-buffer",
        action="store_false",
        dest="use_stable_forward_byte_send_buffer",
        help="Disable stable-buffer forward byte sends.",
    )
    parser.add_argument(
        "--use-ready-stable-forward-byte-send-buffer",
        action="store_true",
        default=False,
        help=(
            "Experimental: after posting receives, enqueue all stable-buffer forward byte copies and post each send "
            "when that peer stream becomes ready. Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-ready-stable-forward-byte-send-buffer",
        action="store_false",
        dest="use_ready_stable_forward_byte_send_buffer",
        help="Disable experimental ready-polled stable-buffer forward byte sends.",
    )
    parser.add_argument(
        "--use-contiguous-forward-byte-send",
        action="store_true",
        default=False,
        help=(
            "Post stable forward CUDA-aware sends from contiguous communication buffers. "
            "Use --contiguous-forward-send-mode to select single or chunked posting. Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-contiguous-forward-byte-send",
        action="store_false",
        dest="use_contiguous_forward_byte_send",
        help="Disable contiguous posting for stable forward byte sends.",
    )
    parser.add_argument(
        "--use-physical-forward-peer-exchange",
        action="store_true",
        default=False,
        help=(
            "Experimental: use stable contiguous forward peer exchange buffers and value-count MPI "
            "for reference-parity CUDA-aware byte transfers when the per-peer value count fits MPI_INT. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-physical-forward-peer-exchange",
        action="store_false",
        dest="use_physical_forward_peer_exchange",
        help="Disable experimental physical forward peer-exchange path.",
    )
    parser.add_argument(
        "--use-native-backward-second-peer-loop",
        action="store_true",
        default=False,
        help=(
            "Experimental: route backward second transpose through the native schedule-slot peer loop. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-native-backward-second-peer-loop",
        action="store_false",
        dest="use_native_backward_second_peer_loop",
        help="Disable native backward-second peer-loop execution.",
    )
    parser.add_argument(
        "--native-backward-second-peer-loop-modes",
        default="configured",
        help=(
            "Native backward-second peer-loop matrix for 3D CUDA-aware native pencil-pencil P2P runs: "
            "configured, both, or comma-separated off,on."
        ),
    )
    parser.add_argument(
        "--use-native-opt0-default-z-layout",
        action="store_true",
        default=False,
        help=(
            "Experimental: use the FFTM-native opt0 default-Z bridge path for native 3D "
            "pencil-pencil reference-owned runs. Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-default-z-layout",
        action="store_false",
        dest="use_native_opt0_default_z_layout",
        help="Disable the native opt0 default-Z bridge path.",
    )
    parser.add_argument(
        "--use-native-opt0-egger-y-buffer-topology",
        "--use-native-opt0-reference-y-buffer-topology",
        action="store_true",
        dest="use_native_opt0_reference_y_buffer_topology",
        default=False,
        help=(
            "Experimental: bind native opt0 Y input/output buffers to the reference-style active "
            "complex/mem_d slot topology. Requires --use-native-opt0-default-z-layout."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-egger-y-buffer-topology",
        "--no-native-opt0-reference-y-buffer-topology",
        action="store_false",
        dest="use_native_opt0_reference_y_buffer_topology",
        help="Disable the native opt0 reference Y-buffer topology experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-compact-y-workarea",
        action="store_true",
        default=False,
        help=(
            "Experimental: reuse the native opt0 reference-style receive slot as the Y workarea "
            "when communication buffers are inactive, reducing the opt0 workspace lifetime."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-compact-y-workarea",
        action="store_false",
        dest="use_native_opt0_compact_y_workarea",
        help="Disable the native opt0 compact Y workarea layout.",
    )
    parser.add_argument(
        "--use-native-opt0-tight-y-plan-sequence",
        action="store_true",
        default=False,
        help=(
            "Experimental: execute native opt0 Y C2C plan arrays through the typed tight "
            "plan-sequence executor. Requires --use-native-opt0-default-z-layout."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-tight-y-plan-sequence",
        action="store_false",
        dest="use_native_opt0_tight_y_plan_sequence",
        help="Disable the native opt0 tight Y plan-sequence executor.",
    )
    parser.add_argument(
        "--use-native-opt0-shared-y-plan-handles",
        action="store_true",
        default=False,
        help=(
            "Experimental: create one native opt0 Y C2C plan handle array and execute it in "
            "forward or inverse direction, matching the reference opt0 plan-handle lifecycle."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-shared-y-plan-handles",
        action="store_false",
        dest="use_native_opt0_shared_y_plan_handles",
        help="Disable the native opt0 shared Y C2C plan-handle experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-y-group-device-sync",
        action="store_true",
        default=False,
        help=(
            "Experimental: after native opt0 Y plan-sequence launch, synchronize once with "
            "device_synchronize instead of synchronizing each Y plan stream."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-y-group-device-sync",
        action="store_false",
        dest="use_native_opt0_y_group_device_sync",
        help="Disable the native opt0 grouped Y device synchronization experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-y-no-sync-exec",
        action="store_true",
        default=True,
        help=(
            "Experimental: check native opt0 Y cuFFT exec calls without synchronizing each launch; "
            "the executor still synchronizes after the whole Y plan group."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-y-no-sync-exec",
        action="store_false",
        dest="use_native_opt0_y_no_sync_exec",
        help="Use the synchronized native opt0 Y cuFFT exec check.",
    )
    parser.add_argument(
        "--use-native-opt0-raw-y-plan-array-executor",
        action="store_true",
        default=False,
        help=(
            "Experimental: execute native opt0 Y plans through an FFT-wrapper-owned opaque "
            "C2C plan-handle array."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-raw-y-plan-array-executor",
        action="store_false",
        dest="use_native_opt0_raw_y_plan_array_executor",
        help="Disable the native opt0 opaque/raw Y plan-array executor.",
    )
    parser.add_argument(
        "--use-native-opt0-reference-y-plan-lifecycle",
        action="store_true",
        default=False,
        help=(
            "Experimental: recreate native opt0 Y plans with stream/work-area binding in the "
            "reference/reference-style lifecycle inside the FFT abstraction."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-reference-y-plan-lifecycle",
        action="store_false",
        dest="use_native_opt0_reference_y_plan_lifecycle",
        help="Disable the native opt0 reference-style Y plan lifecycle experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-reference-y-plan-bundle",
        action="store_true",
        default=False,
        help=(
            "Experimental: create an FFT-abstraction-owned native opt0 Y plan-array bundle "
            "with bundle-owned streams, offsets, work areas, and direct opaque execution."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-reference-y-plan-bundle",
        action="store_false",
        dest="use_native_opt0_reference_y_plan_bundle",
        help="Disable the native opt0 reference-style Y plan-array bundle experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-raw-y-plan-bundle",
        action="store_true",
        default=False,
        help=(
            "Experimental: create native opt0 Y C2C plan handles through a raw reference-style "
            "plan-array path inside the FFT abstraction, then bind them to the pencil executor streams."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-raw-y-plan-bundle",
        action="store_false",
        dest="use_native_opt0_raw_y_plan_bundle",
        help="Disable the native opt0 raw reference-style Y plan-array bundle experiment.",
    )
    parser.add_argument(
        "--use-native-opt0-y-plan-bundle-stream-first",
        action="store_true",
        default=False,
        help="Experimental: bind native opt0 Y plan-array streams before work areas during bundle activation.",
    )
    parser.add_argument(
        "--no-native-opt0-y-plan-bundle-stream-first",
        action="store_false",
        dest="use_native_opt0_y_plan_bundle_stream_first",
        help="Disable stream-before-work native opt0 Y plan-array bundle activation.",
    )
    parser.add_argument(
        "--use-native-opt0-raw-y-plan-bundle-egger-streams",
        "--use-native-opt0-raw-y-plan-bundle-reference-streams",
        action="store_true",
        dest="use_native_opt0_raw_y_plan_bundle_reference_streams",
        default=False,
        help=(
            "Experimental: create raw native opt0 Y plan-array handles with pencil executor streams bound "
            "immediately after plan creation, matching reference's lifecycle more closely."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-raw-y-plan-bundle-egger-streams",
        "--no-native-opt0-raw-y-plan-bundle-reference-streams",
        action="store_false",
        dest="use_native_opt0_raw_y_plan_bundle_reference_streams",
        help="Disable immediate reference-style stream binding during raw native opt0 Y plan-array creation.",
    )
    parser.add_argument(
        "--use-native-opt0-egger-local-plan-context",
        "--use-native-opt0-reference-local-plan-context",
        action="store_true",
        dest="use_native_opt0_reference_local_plan_context",
        default=False,
        help=(
            "Experimental: create an FFT-abstraction-owned opt0 local plan context in reference order "
            "and use its Y plan array for native opt0 execution."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-egger-local-plan-context",
        "--no-native-opt0-reference-local-plan-context",
        action="store_false",
        dest="use_native_opt0_reference_local_plan_context",
        help="Disable the native opt0 reference local plan-context bundle experiment.",
    )
    parser.add_argument(
        "--allow-native-opt0-diagnostic-variants",
        action="store_true",
        default=False,
        help=(
            "Allow native opt0 diagnostic-only Y executor variants. Production runs leave this disabled."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-diagnostic-variants",
        action="store_false",
        dest="allow_native_opt0_diagnostic_variants",
        help="Reject native opt0 diagnostic-only Y executor variants.",
    )
    parser.add_argument(
        "--use-native-opt0-memory-feasibility-guard",
        action="store_true",
        default=True,
        help=(
            "Check native opt0 reference-style shared workspace against runtime free device memory before "
            "allocating it. Default: enabled."
        ),
    )
    parser.add_argument(
        "--no-native-opt0-memory-feasibility-guard",
        action="store_false",
        dest="use_native_opt0_memory_feasibility_guard",
        help="Disable the native opt0 shared-workspace memory feasibility guard.",
    )
    parser.add_argument(
        "--native-opt0-memory-feasibility-reserve-mib",
        type=int,
        default=512,
        help="Device-memory reserve kept by the native opt0 memory feasibility guard. Default: 512.",
    )
    parser.add_argument(
        "--native-opt0-y-executor-variants",
        default="configured",
        help=(
            "Sweep native opt0 Y C2C executor variants for 3D CUDA-aware native opt0 "
            "pencil-pencil reference runs. Values: configured, all, virtual-stream, "
            "virtual-device, tight-stream, tight-device, opaque-stream, opaque-device, "
            "shared-opaque-stream, shared-opaque-device, ref-life-stream, ref-life-device, "
            "ref-bundle-stream, ref-bundle-device, raw-bundle-stream, raw-bundle-device, "
            "raw-bundle-streamfirst-stream, raw-bundle-streamfirst-device, "
            "raw-bundle-reference-streams-stream, raw-bundle-reference-streams-device, "
            "raw-bundle-reference-streamfirst-stream, raw-bundle-reference-streamfirst-device, "
            "context-bundle-stream, context-bundle-device."
        ),
    )
    parser.add_argument(
        "--enable-fftm3d-backend-stage-timers",
        action="store_true",
        default=False,
        help=(
            "Enable wrapped reference stage-timer CSVs when the fftm3d-scfd-fft-facade backend "
            "is selected. Default: disabled."
        ),
    )
    parser.add_argument(
        "--disable-fftm3d-backend-stage-timers",
        action="store_false",
        dest="enable_fftm3d_backend_stage_timers",
        help="Disable wrapped reference stage-timer CSVs for the fftm3d backend.",
    )
    parser.add_argument(
        "--enable-local-fft-diagnostics",
        action="store_true",
        default=False,
        help=(
            "Enable per-rank local FFT plan descriptor and device-event timing CSVs for native 3D FFTM runs. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--disable-local-fft-diagnostics",
        action="store_false",
        dest="enable_local_fft_diagnostics",
        help="Disable local FFT diagnostics.",
    )
    parser.add_argument(
        "--native-opt0-y-microbench",
        action="store_true",
        default=False,
        help=(
            "Run the native opt0 same-buffer Y-only microbenchmark and exit the 3D FFTM binary after diagnostics. "
            "This is a check-only Step 8 diagnostic."
        ),
    )
    parser.add_argument(
        "--native-opt0-y-cross-microbench",
        action="store_true",
        default=False,
        help=(
            "Run the focused native/reference opt0 Y plan/buffer cross microbenchmark and exit the 3D FFTM binary "
            "after diagnostics. Reuses --native-opt0-y-microbench-iterations and warmup."
        ),
    )
    parser.add_argument(
        "--native-opt0-y-cross-factory-modes",
        default="configured",
        help=(
            "Factory/order modes for --native-opt0-y-cross-microbench. Use configured/single for the historical "
            "single case, matrix/all for all process-clean cases, or a comma list of ref-only, native-current-only, "
            "native-current-nosync-only, native-exact-only, native-owned-only, native-direct-only, "
            "native-direct-nosync-only, native-minimal-only, native-minimal-nosync-only, ref-then-native-current, "
            "native-current-then-ref, ref-then-native-current-nosync, native-current-nosync-then-ref, "
            "ref-then-native-exact, native-exact-then-ref, ref-then-native-owned, native-owned-then-ref, "
            "ref-then-native-direct, native-direct-then-ref, ref-then-native-direct-nosync, "
            "native-direct-nosync-then-ref, ref-then-native-minimal, native-minimal-then-ref, "
            "ref-then-native-minimal-nosync, native-minimal-nosync-then-ref."
        ),
    )
    parser.add_argument(
        "--native-opt0-y-microbench-iterations",
        type=int,
        default=20,
        help="Measured iterations for --native-opt0-y-microbench. Default: 20.",
    )
    parser.add_argument(
        "--native-opt0-y-microbench-warmup",
        type=int,
        default=3,
        help="Warmup iterations for --native-opt0-y-microbench. Default: 3.",
    )
    parser.add_argument(
        "--enable-native-stage-timers",
        action="store_true",
        default=False,
        help=(
            "Enable per-rank measured-loop native stage timing CSVs for native 3D FFTM runs. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--disable-native-stage-timers",
        action="store_false",
        dest="enable_native_stage_timers",
        help="Disable native measured-loop stage timing CSVs.",
    )
    parser.add_argument(
        "--enable-gpu-telemetry",
        action="store_true",
        default=False,
        help=(
            "Write per-rank pre/post GPU telemetry CSVs from inside the benchmark process. "
            "Includes selected device, PCI bus id, clocks, temperature, power, memory, and utilization."
        ),
    )
    parser.add_argument(
        "--disable-gpu-telemetry",
        action="store_false",
        dest="enable_gpu_telemetry",
        help="Disable per-rank GPU telemetry CSVs.",
    )
    parser.add_argument(
        "--contiguous-forward-send-mode",
        default="single",
        choices=FFTM_CONTIGUOUS_FORWARD_SEND_MODES,
        help="Configured contiguous forward send mode when not sweeping. Default: single.",
    )
    parser.add_argument(
        "--contiguous-forward-send-modes",
        default="configured",
        help=(
            "Contiguous forward send mode matrix for 3D CUDA-aware pencil-pencil P2P runs: "
            "configured, all, or comma-separated subset of single,chunked."
        ),
    )
    parser.add_argument(
        "--contiguous-forward-send-chunk-mib",
        type=int,
        default=1024,
        help="Chunk size in MiB for contiguous-forward-send mode=chunked. Default: 1024.",
    )
    parser.add_argument(
        "--contiguous-forward-send-registration-warmups",
        type=int,
        default=0,
        help="Extra untimed benchmark warmup iterations before normal warmups. Default: 0.",
    )
    parser.add_argument(
        "--pencil-layouts",
        default="configured",
        help="3D pencil-pencil layout matrix: configured, all, or comma-separated subset of auto,opt0,opt1,legacy.",
    )
    parser.add_argument(
        "--pencil-pipelines",
        default="configured",
        help=(
            "3D pencil-pencil pipeline matrix: configured, all, or comma-separated subset of "
            "staged,fused,reference,reference-parity. Deprecated aliases: egger,egger-parity."
        ),
    )
    parser.add_argument(
        "--large-count-p2p-transports",
        default="configured",
        help=(
            "Large-count transport matrix for 3D CUDA-aware pencil-pencil P2P runs: "
            "configured, all, or comma-separated subset of "
            f"{','.join(FFTM_LARGE_COUNT_P2P_TRANSPORTS)}."
        ),
    )
    parser.add_argument(
        "--fftm-3d-backends",
        default="configured",
        help=(
            "3D FFTM benchmark backend matrix: configured, all, or comma-separated subset of "
            f"{','.join(FFTM_3D_BACKENDS)}. The fftm3d backend is only emitted for CUDA-aware "
            "3D pencil-pencil benchmark runs."
        ),
    )
    parser.add_argument(
        "--use-large-count-datatype-cache",
        action="store_true",
        default=False,
        help="Cache hindexed large-count MPI datatypes per peer/stage. Default: disabled.",
    )
    parser.add_argument(
        "--no-large-count-datatype-cache",
        action="store_false",
        dest="use_large_count_datatype_cache",
        help="Disable hindexed large-count MPI datatype caching.",
    )
    parser.add_argument(
        "--use-fft-exec-no-sync",
        action="store_true",
        default=False,
        help=(
            "Experimental: check hot cuFFT exec launches without synchronizing inside the FFT abstraction. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--no-fft-exec-no-sync",
        action="store_false",
        dest="use_fft_exec_no_sync",
        help="Use the synchronized FFT exec check in the generic FFT abstraction path.",
    )
    parser.add_argument(
        "--pencil-pencil-grid-orientations",
        default="both",
        help=(
            "Grid orientations scheduled for 3D pencil-pencil FFTM runs: configured, both, production, default, or reversed. "
            "configured passes no --grid argument so an autotune config can choose the grid. "
            "production uses known safe choices, currently 8G opt0/auto -> 4x2 and 8G opt1 -> 2x4. "
            "Default: both"
        ),
    )
    parser.add_argument(
        "--extra-sizes-3d",
        default="",
        help="Extra 3D benchmark side lengths to run in addition to auto or explicit sizes.",
    )
    parser.add_argument(
        "--extra-sizes-3d-by-gpu",
        default="",
        help=(
            "Per-GPU-count 3D benchmark side lengths appended to auto/global sizes, "
            "e.g. '1:1050;2:1344;5:1800,2048'."
        ),
    )
    parser.add_argument(
        "--fixed-scaling-sizes-3d",
        default="",
        help=(
            "Comma-separated 3D side lengths appended to every FFTM benchmark GPU count. "
            "Use this for fixed-problem strong-scaling/acceleration runs across nodes."
        ),
    )
    parser.add_argument(
        "--fixed-scaling-sizes-4d",
        default="",
        help=(
            "Comma-separated 4D side lengths appended to every FFTM benchmark GPU count. "
            "Use this for fixed-problem strong-scaling/acceleration runs across nodes."
        ),
    )
    parser.add_argument("--validation-times", type=int, default=1)
    parser.add_argument("--epsilon", dest="validation_epsilon", default="1.0e-11")
    parser.add_argument("--threshold", dest="validation_threshold", default="1.0e-11")

    parser.add_argument("--transports", default="cuda_aware,non_cuda_aware")
    parser.add_argument("--modes", default="alltoallv,p2p-waitall,p2p-waitany")
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
    if args.test_warmup < 0:
        parser.error("--warmup must be non-negative")
    runner = PaperClusterRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
