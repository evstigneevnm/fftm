#!/usr/bin/env python3
"""Build and query a small FFTM autotune cache from benchmark result directories.

The cache is intentionally simple JSON. It is meant to be produced by an
installation/setup run, reviewed, and then used by launch scripts or applications
to select known-good FFTM options for a machine, GPU count, and transform size.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = 1

DIAGNOSTIC_ONLY_SELECTED_KEYS = (
    "fft_exec_no_sync",
    "native_stage_timers",
    "native_opt0_reference_y_plan_lifecycle",
    "native_opt0_reference_y_plan_bundle",
    "native_opt0_raw_y_plan_bundle",
    "native_opt0_y_plan_bundle_stream_first",
    "native_opt0_raw_y_plan_bundle_reference_streams",
    "native_opt0_reference_local_plan_context",
    "slab_native_xw_batched_peer_kernels",
    "slab_native_xw_tensor_coalesced_kernels",
    "slab_native_xw_vector4_kernels",
    "slab_native_xw_tiled_kernels",
    "slab_native_xw_layout_stage",
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def as_float(value: Any, default: float = math.inf) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def row_value(row: Dict[str, str], primary: str, *aliases: str) -> Any:
    for key in (primary,) + aliases:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return row.get(primary)


def canonical_pencil_pipeline(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "native-compatible": "reference",
        "reference-compatible": "reference",
        "compatible": "reference",
        "egger": "reference",
        "native-parity": "reference-parity",
        "native_parity": "reference-parity",
        "reference_parity": "reference-parity",
        "compatible-parity": "reference-parity",
        "compatible_parity": "reference-parity",
        "egger-parity": "reference-parity",
        "egger_parity": "reference-parity",
    }
    return aliases.get(text, text)


def canonical_filter(value: Any, *, pipeline: bool = False) -> Any:
    if not pipeline:
        return value
    values = filter_values(value)
    if not values:
        return value
    return ",".join(canonical_pencil_pipeline(item) for item in values)


def read_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def machine_info(result_dir: Path) -> Dict[str, Any]:
    env = read_env_file(result_dir / "run_config.env")
    config: Dict[str, Any] = {}
    hardware: Dict[str, Any] = {}
    config_path = result_dir / "config.json"
    hardware_path = result_dir / "hardware.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            config = {}
    if hardware_path.exists():
        try:
            hardware = json.loads(hardware_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            hardware = {}
    config_args = config.get("args") if isinstance(config.get("args"), dict) else {}
    effective_gpus = hardware.get("effective_gpus") if isinstance(hardware.get("effective_gpus"), list) else []
    first_gpu = effective_gpus[0] if effective_gpus and isinstance(effective_gpus[0], dict) else {}
    telemetry = sorted((result_dir / "telemetry").glob("*_pre.txt"))
    info: Dict[str, Any] = {
        "hostname": env.get("FFTM_HOSTNAME") or env.get("FFTM3D_HOSTNAME") or config_args.get("hostname", ""),
        "gpu_name": (
            env.get("FFTM_GPU_NAME")
            or env.get("FFTM3D_GPU_NAME")
            or config_args.get("gpu_name", "")
            or first_gpu.get("name", "")
        ),
        "device_memory_mib": (
            env.get("FFTM_DEVICE_MEMORY_MIB")
            or env.get("FFTM3D_DEVICE_MEMORY_MIB")
            or config_args.get("device_memory_mib", "")
            or first_gpu.get("memory_total_mib", "")
        ),
    }
    if not info["hostname"] and not hardware:
        info["hostname"] = socket.gethostname()
    if telemetry:
        for line in telemetry[0].read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("hostname="):
                info["hostname"] = line.split("=", 1)[1].strip()
            elif "NVIDIA" in line and "GPU" in line and not info.get("gpu_name"):
                payload = line.split(":", 1)[-1].strip()
                info["gpu_name"] = payload.split(" (UUID:", 1)[0].strip()
    return info


def choose_pencil_grid_3d(num_gpus: int) -> Tuple[int, int]:
    p1 = 1
    d = 1
    while d * d <= num_gpus:
        if num_gpus % d == 0:
            p1 = d
        d += 1
    return p1, num_gpus // p1


def grid_orientation(num_gpus: int, p1: int, p2: int) -> str:
    default_grid = choose_pencil_grid_3d(num_gpus)
    reversed_grid = (default_grid[1], default_grid[0])
    grid = (p1, p2)
    if grid == default_grid:
        return "default"
    if grid == reversed_grid:
        return "reversed"
    return "explicit"


def make_key(machine: Dict[str, Any], library: str, dim: int, num_gpus: int, sizes: Tuple[int, ...]) -> Dict[str, Any]:
    key: Dict[str, Any] = {
        "machine": {
            "hostname": machine.get("hostname", ""),
            "gpu_name": machine.get("gpu_name", ""),
        },
        "library": library,
        "dim": dim,
        "num_gpus": num_gpus,
        "sizes": list(sizes),
    }
    return key


def key_token(key: Dict[str, Any]) -> str:
    machine = key.get("machine", {})
    sizes = "x".join(str(v) for v in key.get("sizes", []))
    return "|".join(
        [
            str(machine.get("hostname", "")),
            str(machine.get("gpu_name", "")),
            str(key.get("library", "")),
            str(key.get("dim", "")),
            str(key.get("num_gpus", "")),
            sizes,
        ]
    )


def candidate_token(entry: Dict[str, Any]) -> str:
    """Stable identity for one measured configuration.

    A cache key identifies the machine, library, GPU count, and size. Multiple
    measured configurations can share that key, so the selected option set must
    also participate in de-duplication.
    """
    selected = entry.get("selected") or {}
    return key_token(entry.get("key") or {}) + "|" + json.dumps(selected, sort_keys=True, separators=(",", ":"))


def selected_has_diagnostic_only_flags(selected: Dict[str, Any]) -> bool:
    return any(as_int(selected.get(key)) != 0 for key in DIAGNOSTIC_ONLY_SELECTED_KEYS)


def canonicalize_selected(selected: Dict[str, Any]) -> Dict[str, Any]:
    if "pencil_pipeline" in selected:
        selected["pencil_pipeline"] = canonical_pencil_pipeline(selected.get("pencil_pipeline"))

    legacy_key_pairs = [
        ("native_opt0_egger_y_buffer_topology", "native_opt0_reference_y_buffer_topology"),
        ("native_opt0_raw_y_plan_bundle_egger_streams", "native_opt0_raw_y_plan_bundle_reference_streams"),
        ("native_opt0_egger_local_plan_context", "native_opt0_reference_local_plan_context"),
    ]
    for legacy, current in legacy_key_pairs:
        if current not in selected and legacy in selected:
            selected[current] = selected[legacy]
        selected.pop(legacy, None)
    return selected


def mark_selected_policy(entry: Dict[str, Any]) -> Dict[str, Any]:
    selected = entry.setdefault("selected", {})
    canonicalize_selected(selected)
    if selected.get("backend") == "fftm3d-scfd-fft-facade" or entry.get("source_library") == "fftm3d":
        selected["reference_only"] = 1
    if (
        selected.get("backend", "native") == "native"
        and selected.get("strategy") == "pencil-pencil"
        and selected_pencil_layout(selected) == "opt0"
        and selected_has_diagnostic_only_flags(selected)
    ):
        selected["diagnostic_only"] = 1
    return entry


def remember_best_candidate(entries: Dict[str, Dict[str, Any]], entry: Dict[str, Any]) -> None:
    entry = mark_selected_policy(entry)
    token = candidate_token(entry)
    current = entries.get(token)
    if current is None or entry["metrics"]["avg_wall_ms"] < current["metrics"]["avg_wall_ms"]:
        entries[token] = entry


def selected_fftm_3d(row: Dict[str, str]) -> Dict[str, Any]:
    num_gpus = as_int(row.get("num_gpus"))
    p1 = as_int(row.get("p1"))
    p2 = as_int(row.get("p2"))
    selected: Dict[str, Any] = {
        "strategy": row.get("strategy", ""),
        "mode": row.get("mode", ""),
        "grid": [p1, p2],
        "grid_orientation": grid_orientation(num_gpus, p1, p2),
        "pencil_layout": row.get("pencil_layout", ""),
        "pencil_pipeline": canonical_pencil_pipeline(row.get("pencil_pipeline", "")),
        "large_count_p2p_transport": row.get("large_count_p2p_transport", ""),
        "fft_exec_no_sync": as_int(row.get("fft_exec_no_sync")),
        "stable_forward_byte_send_buffer": as_int(row.get("stable_forward_byte_send_buffer")),
        "ready_stable_forward_byte_send_buffer": as_int(row.get("ready_stable_forward_byte_send_buffer")),
        "contiguous_forward_byte_send": as_int(row.get("contiguous_forward_byte_send")),
        "physical_forward_peer_exchange": as_int(row.get("physical_forward_peer_exchange")),
        "contiguous_forward_send_mode": row.get("contiguous_forward_send_mode", ""),
        "contiguous_forward_send_chunk_mib": as_int(row.get("contiguous_forward_send_chunk_mib")),
        "persistent_p2p": as_int(row.get("persistent_p2p")),
        "ready_p2p_send": as_int(row.get("ready_p2p_send")),
        "backend": row.get("fftm_3d_backend", "") or "native",
        "native_backward_second_peer_loop": as_int(row.get("native_backward_second_peer_loop")),
        "native_opt0_default_z_layout": as_int(row.get("native_opt0_default_z_layout")),
        "native_opt0_reference_y_buffer_topology": as_int(
            row_value(row, "native_opt0_reference_y_buffer_topology", "native_opt0_egger_y_buffer_topology")
        ),
        "native_opt0_compact_y_workarea": as_int(row.get("native_opt0_compact_y_workarea")),
        "native_opt0_compact_y_workarea_effective": as_int(row.get("native_opt0_compact_y_workarea_effective")),
        "native_opt0_auto_compact_y_workarea": as_int(row.get("native_opt0_auto_compact_y_workarea")),
        "native_opt0_default_z_scratch_aliased": as_int(row.get("native_opt0_default_z_scratch_aliased")),
        "native_opt0_tight_y_plan_sequence": as_int(row.get("native_opt0_tight_y_plan_sequence")),
        "native_opt0_shared_y_plan_handles": as_int(row.get("native_opt0_shared_y_plan_handles")),
        "native_opt0_y_group_device_sync": as_int(row.get("native_opt0_y_group_device_sync")),
        "native_opt0_y_no_sync_exec": as_int(row.get("native_opt0_y_no_sync_exec")),
        "native_opt0_raw_y_plan_array_executor": as_int(row.get("native_opt0_raw_y_plan_array_executor")),
        "native_opt0_reference_y_plan_lifecycle": as_int(row.get("native_opt0_reference_y_plan_lifecycle")),
        "native_opt0_reference_y_plan_bundle": as_int(row.get("native_opt0_reference_y_plan_bundle")),
        "native_opt0_raw_y_plan_bundle": as_int(row.get("native_opt0_raw_y_plan_bundle")),
        "native_opt0_y_plan_bundle_stream_first": as_int(row.get("native_opt0_y_plan_bundle_stream_first")),
        "native_opt0_raw_y_plan_bundle_reference_streams": as_int(
            row_value(
                row,
                "native_opt0_raw_y_plan_bundle_reference_streams",
                "native_opt0_raw_y_plan_bundle_egger_streams",
            )
        ),
        "native_opt0_reference_local_plan_context": as_int(
            row_value(row, "native_opt0_reference_local_plan_context", "native_opt0_egger_local_plan_context")
        ),
    }
    return selected


def selected_fftm_4d(row: Dict[str, str]) -> Dict[str, Any]:
    p1 = as_int(row.get("p1"))
    p2 = as_int(row.get("p2"))
    p3 = as_int(row.get("p3"))
    return {
        "strategy": row.get("strategy", ""),
        "mode": row.get("mode", ""),
        "grid": [p1, p2, p3],
        "fft_exec_no_sync": as_int(row.get("fft_exec_no_sync")),
        "native_stage_timers": as_int(row.get("native_stage_timers")),
        "slab_native_xw": as_int(row.get("slab_native_xw")),
        "slab_native_xw_batched_peer_kernels": as_int(row.get("slab_native_xw_batched_peer_kernels")),
        "slab_native_xw_tensor_coalesced_kernels": as_int(row.get("slab_native_xw_tensor_coalesced_kernels")),
        "slab_native_xw_vector4_kernels": as_int(row.get("slab_native_xw_vector4_kernels")),
        "slab_native_xw_tiled_kernels": as_int(row.get("slab_native_xw_tiled_kernels")),
        "slab_native_xw_layout_stage": as_int(row.get("slab_native_xw_layout_stage")),
        "slab_native_xw_native_spectral_layout": as_int(row.get("slab_native_xw_native_spectral_layout")),
        "pencil_same_zw_peer_paired": as_int(row.get("pencil_same_zw_peer_paired")),
        "pencil_same_zw_native_layout": as_int(row.get("pencil_same_zw_native_layout")),
        "pencil_degenerate_xw_slab_path": as_int(row.get("pencil_degenerate_xw_slab_path")),
        "pencil_degenerate_local_transposes": as_int(row.get("pencil_degenerate_local_transposes")),
        "pencil_degenerate_same_xw_native": as_int(row.get("pencil_degenerate_same_xw_native")),
        "pencil_degenerate_wz_sliced_z_fft": as_int(row.get("pencil_degenerate_wz_sliced_z_fft")),
        "native_xw_direct_layout": as_int(row.get("native_xw_direct_layout")),
        "native_xw_chunked_transport": as_int(row.get("native_xw_chunked_transport")),
        "native_xw_chunk_mib": as_int(row.get("native_xw_chunk_mib"), 1024),
    }


def selected_fftm3d(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "grid": [as_int(row.get("p1")), as_int(row.get("p2"))],
        "opt": row.get("opt", ""),
        "egger_variant": row.get("egger_variant", ""),
        "device_map": row.get("device_map", ""),
        "comm": row.get("comm1", ""),
        "send": row.get("send1", ""),
        "storage": row.get("storage", ""),
        "mpi_backend": row.get("mpi_backend", ""),
        "runtime_backend": row.get("runtime_backend", ""),
    }


def selected_fftm3d_as_fftm_backend(row: Dict[str, str]) -> Dict[str, Any]:
    num_gpus = as_int(row.get("num_gpus"))
    p1 = as_int(row.get("p1"))
    p2 = as_int(row.get("p2"))
    opt = row.get("opt", "")
    return {
        "strategy": "pencil-pencil",
        "mode": "p2p-waitany",
        "grid": [p1, p2],
        "grid_orientation": grid_orientation(num_gpus, p1, p2),
        "pencil_layout": opt,
        "pencil_pipeline": "reference-parity",
        "large_count_p2p_transport": "",
        "stable_forward_byte_send_buffer": 0,
        "ready_stable_forward_byte_send_buffer": 0,
        "contiguous_forward_byte_send": 0,
        "physical_forward_peer_exchange": 0,
        "contiguous_forward_send_mode": "",
        "contiguous_forward_send_chunk_mib": 0,
        "persistent_p2p": 0,
        "ready_p2p_send": 0,
        "backend": "fftm3d-scfd-fft-facade",
        "reference_only": 1,
    }


def metrics(row: Dict[str, str]) -> Dict[str, Any]:
    return {
        "avg_wall_ms": as_float(row.get("avg_wall_ms"), math.inf),
        "min_wall_ms": as_float(row.get("min_wall_ms"), None),
        "stddev_wall_ms": as_float(row.get("stddev_wall_ms"), None),
        "max_l2_diff": as_float(row.get("max_l2_diff"), None),
        "epsilon": as_float(row.get("epsilon"), None),
    }


def row_is_numerically_valid(row: Dict[str, str]) -> bool:
    max_l2 = as_float(row.get("max_l2_diff"), None)
    epsilon = as_float(row.get("epsilon"), None)
    if max_l2 is None or epsilon is None:
        return True
    if not math.isfinite(max_l2) or not math.isfinite(epsilon):
        return False
    return max_l2 <= epsilon


def collect_fftm_entries(result_dir: Path, machine: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}

    csv_3d = first_existing(
        [
            result_dir / "cpp_csv" / "benchmark_fftm_3d.csv",
            result_dir / "benchmark_fftm_3d.csv",
        ]
    )
    if csv_3d is not None:
        with csv_3d.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("benchmark") != "fftm-3d":
                    continue
                if not row_is_numerically_valid(row):
                    continue
                num_gpus = as_int(row.get("num_gpus"))
                sizes = (as_int(row.get("nx")), as_int(row.get("ny")), as_int(row.get("nz")))
                key = make_key(machine, "fftm", 3, num_gpus, sizes)
                entry = {
                    "key": key,
                    "selected": selected_fftm_3d(row),
                    "metrics": metrics(row),
                    "source": str(result_dir),
                    "source_csv": str(csv_3d),
                }
                remember_best_candidate(entries, entry)

    csv_4d = first_existing(
        [
            result_dir / "cpp_csv" / "benchmark_fftm_4d.csv",
            result_dir / "benchmark_fftm_4d.csv",
        ]
    )
    if csv_4d is not None:
        with csv_4d.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if row.get("benchmark") != "fftm-4d":
                    continue
                if not row_is_numerically_valid(row):
                    continue
                num_gpus = as_int(row.get("num_gpus"))
                sizes = (
                    as_int(row.get("nx")),
                    as_int(row.get("ny")),
                    as_int(row.get("nz")),
                    as_int(row.get("nw")),
                )
                key = make_key(machine, "fftm", 4, num_gpus, sizes)
                entry = {
                    "key": key,
                    "selected": selected_fftm_4d(row),
                    "metrics": metrics(row),
                    "source": str(result_dir),
                    "source_csv": str(csv_4d),
                }
                remember_best_candidate(entries, entry)
    return list(entries.values())


def collect_fftm3d_entries(result_dir: Path, machine: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_path = first_existing(
        [
            result_dir / "cpp_csv" / "benchmark_fftm3d_3d.csv",
            result_dir / "benchmark_fftm3d_3d.csv",
        ]
    )
    if csv_path is None:
        return []

    entries: Dict[str, Dict[str, Any]] = {}
    fftm_backend_entries: Dict[str, Dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if not row_is_numerically_valid(row):
                continue
            num_gpus = as_int(row.get("num_gpus"))
            sizes = (as_int(row.get("nx")), as_int(row.get("ny")), as_int(row.get("nz")))
            key = make_key(machine, "fftm3d", 3, num_gpus, sizes)
            entry = {
                "key": key,
                "selected": selected_fftm3d(row),
                "metrics": metrics(row),
                "source": str(result_dir),
                "source_csv": str(csv_path),
            }
            remember_best_candidate(entries, entry)

            if row.get("egger_variant") == "scfd-fft-facade":
                fftm_key = make_key(machine, "fftm", 3, num_gpus, sizes)
                fftm_entry = {
                    "key": fftm_key,
                    "selected": selected_fftm3d_as_fftm_backend(row),
                    "metrics": metrics(row),
                    "source": str(result_dir),
                    "source_csv": str(csv_path),
                    "source_library": "fftm3d",
                }
                remember_best_candidate(fftm_backend_entries, fftm_entry)
    return list(entries.values()) + list(fftm_backend_entries.values())


def load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported autotune cache schema in {path}")
    data.setdefault("entries", [])
    data["entries"] = [normalize_entry(entry) for entry in data["entries"]]
    return data


def normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    return mark_selected_policy(entry)


def update_cache(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    cache = {"schema_version": SCHEMA_VERSION, "entries": []} if getattr(args, "reset", False) else load_cache(output)
    entries_by_candidate = {
        candidate_token(normalize_entry(entry)): normalize_entry(entry) for entry in cache.get("entries", [])
    }
    include_strategies = {
        item.strip()
        for item in str(getattr(args, "include_strategies", "") or "").split(",")
        if item.strip()
    }
    exclude_strategies = {
        item.strip()
        for item in str(getattr(args, "exclude_strategies", "") or "").split(",")
        if item.strip()
    }

    def update_filter_entry(entry: Dict[str, Any]) -> bool:
        selected = entry.get("selected") or {}
        strategy = str(selected.get("strategy") or "")
        if include_strategies and strategy not in include_strategies:
            return False
        if exclude_strategies and strategy in exclude_strategies:
            return False
        return True

    added_sources: List[str] = []
    total_new_entries = 0
    for raw_dir in args.result_dirs:
        result_dir = Path(raw_dir).resolve()
        if not result_dir.exists():
            print(f"WARNING: autotune source directory does not exist: {result_dir}", file=sys.stderr)
            added_sources.append(str(result_dir))
            continue
        machine = machine_info(result_dir)
        new_entries = [
            entry
            for entry in collect_fftm_entries(result_dir, machine) + collect_fftm3d_entries(result_dir, machine)
            if update_filter_entry(entry)
        ]
        if not new_entries:
            print(
                "WARNING: no autotune entries collected from "
                f"{result_dir}; expected benchmark_fftm_3d.csv, benchmark_fftm_4d.csv, or benchmark_fftm3d_3d.csv",
                file=sys.stderr,
            )
        total_new_entries += len(new_entries)
        for entry in new_entries:
            remember_best_candidate(entries_by_candidate, entry)
        added_sources.append(str(result_dir))

    if args.result_dirs and total_new_entries == 0:
        print("ERROR: no autotune entries collected from any input directory; cache was not updated.", file=sys.stderr)
        return 1

    cache = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_iso(),
        "source_directories": sorted(set(cache.get("source_directories", []) + added_sources)),
        "entries": sorted(entries_by_candidate.values(), key=lambda item: candidate_token(item)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(cache['entries'])} autotune entries to {output}")
    return 0


def inactive_filter(value: Any) -> bool:
    text = str(value or "").strip()
    return text in {"", "auto", "all", "any", "both", "configured", "none", "production"}


def filter_values(value: Any) -> List[str]:
    if inactive_filter(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def value_matches(actual: Any, expected: Any) -> bool:
    values = filter_values(expected)
    if not values:
        return True
    return str(actual or "") in values


def bool_value_matches(actual: Any, expected: Any) -> bool:
    values = filter_values(expected)
    if not values:
        return True
    aliases = {
        "0": "0",
        "false": "0",
        "off": "0",
        "base": "0",
        "baseline": "0",
        "legacy": "0",
        "1": "1",
        "true": "1",
        "on": "1",
        "native": "1",
        "native-layout": "1",
        "layout-native": "1",
        "peer": "1",
        "paired": "1",
        "peer-paired": "1",
        "peer_paired": "1",
    }
    expected_values = {aliases.get(item.strip().lower(), item.strip()) for item in values}
    return str(as_int(actual)) in expected_values


def pipeline_value_matches(actual: Any, expected: Any) -> bool:
    values = filter_values(expected)
    if not values:
        return True
    canonical_actual = canonical_pencil_pipeline(actual)
    return canonical_actual in {canonical_pencil_pipeline(value) for value in values}


def selected_grid_text(selected: Dict[str, Any]) -> str:
    grid = selected.get("grid") or []
    if isinstance(grid, list) and len(grid) == 2:
        return f"{grid[0]}x{grid[1]}"
    if isinstance(grid, list) and len(grid) == 3:
        return f"{grid[0]}x{grid[1]}x{grid[2]}"
    return str(grid or "")


def selected_pencil_layout(selected: Dict[str, Any]) -> str:
    return str(selected.get("pencil_layout") or selected.get("opt") or "")


def entry_matches_filters(entry: Dict[str, Any], args: argparse.Namespace) -> bool:
    selected = entry.get("selected") or {}
    reference_only = (
        selected.get("reference_only")
        or selected.get("backend") == "fftm3d-scfd-fft-facade"
        or entry.get("source_library") == "fftm3d"
    )
    if reference_only and not getattr(args, "include_reference_only", False):
        return False
    if selected.get("diagnostic_only") and not getattr(args, "include_diagnostic", False):
        return False
    checks = [
        (selected.get("strategy"), getattr(args, "strategy", "")),
        (selected.get("mode"), getattr(args, "mode", "")),
        (selected.get("backend"), getattr(args, "backend", "")),
        (selected_pencil_layout(selected), getattr(args, "pencil_layout", "")),
        (selected.get("pencil_pipeline"), canonical_filter(getattr(args, "pencil_pipeline", ""), pipeline=True)),
        (selected_grid_text(selected), getattr(args, "grid", "")),
        (selected.get("grid_orientation"), getattr(args, "grid_orientation", "")),
        (selected.get("large_count_p2p_transport"), getattr(args, "large_count_p2p_transport", "")),
        (selected.get("slab_native_xw"), getattr(args, "slab_native_xw", "")),
        (
            selected.get("slab_native_xw_native_spectral_layout"),
            getattr(args, "slab_native_xw_native_spectral_layout", ""),
        ),
        (selected.get("slab_native_xw_layout_stage"), getattr(args, "slab_native_xw_layout_stage", "")),
    ]
    for index, (actual, expected) in enumerate(checks):
        if index == 4:
            if not pipeline_value_matches(actual, expected):
                return False
            continue
        if not value_matches(actual, expected):
            return False
    if not bool_value_matches(
        selected.get("pencil_same_zw_peer_paired"), getattr(args, "pencil_same_zw_peer_paired", "")
    ):
        return False
    if not bool_value_matches(
        selected.get("pencil_same_zw_native_layout"), getattr(args, "pencil_same_zw_native_layout", "")
    ):
        return False
    if not bool_value_matches(
        selected.get("pencil_degenerate_xw_slab_path"), getattr(args, "pencil_degenerate_xw_slab_path", "")
    ):
        return False
    if not bool_value_matches(
        selected.get("pencil_degenerate_local_transposes"),
        getattr(args, "pencil_degenerate_local_transposes", ""),
    ):
        return False
    if not bool_value_matches(
        selected.get("pencil_degenerate_same_xw_native"),
        getattr(args, "pencil_degenerate_same_xw_native", ""),
    ):
        return False
    if not bool_value_matches(
        selected.get("pencil_degenerate_wz_sliced_z_fft"),
        getattr(args, "pencil_degenerate_wz_sliced_z_fft", ""),
    ):
        return False
    return True


def find_entry(cache: Dict[str, Any], args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for entry in cache.get("entries", []):
        ekey = entry["key"]
        emachine = ekey.get("machine", {})
        if (
            ekey.get("library") == args.library
            and ekey.get("dim") == args.dim
            and ekey.get("num_gpus") == args.num_gpus
            and ekey.get("sizes") == list(args.sizes)
            and (not args.hostname or emachine.get("hostname") == args.hostname)
            and (not args.gpu_name or emachine.get("gpu_name") == args.gpu_name)
            and entry_matches_filters(entry, args)
        ):
            candidates.append(entry)
    if candidates:
        return min(candidates, key=lambda item: item.get("metrics", {}).get("avg_wall_ms", math.inf))
    return None


def print_env(entry: Dict[str, Any]) -> None:
    key = entry["key"]
    selected = entry["selected"]
    sizes = key["sizes"]
    metrics = entry.get("metrics", {})
    grid = selected.get("grid", [])
    dim = as_int(key.get("dim"))
    if len(grid) == 2:
        print(f"FFTM_AUTOTUNE_GRID_3D={grid[0]}x{grid[1]}")
    elif len(grid) == 3:
        print(f"FFTM_AUTOTUNE_GRID_4D={grid[0]}x{grid[1]}x{grid[2]}")
    print(f"FFTM_AUTOTUNE_LIBRARY={key['library']}")
    print(f"FFTM_AUTOTUNE_NUM_GPUS={key['num_gpus']}")
    print(f"FFTM_AUTOTUNE_SIZE_{dim}D={'x'.join(str(v) for v in sizes)}")
    if selected.get("strategy"):
        print(f"FFTM_AUTOTUNE_STRATEGY_{dim}D={selected['strategy']}")
    if selected.get("mode"):
        print(f"FFTM_AUTOTUNE_MODE={selected['mode']}")
    if selected.get("pencil_layout"):
        print(f"FFTM_AUTOTUNE_PENCIL_LAYOUT={selected['pencil_layout']}")
    elif selected.get("opt"):
        print(f"FFTM_AUTOTUNE_PENCIL_LAYOUT={selected['opt']}")
    if selected.get("pencil_pipeline"):
        print(f"FFTM_AUTOTUNE_PENCIL_PIPELINE={selected['pencil_pipeline']}")
    if selected.get("backend"):
        print(f"FFTM_AUTOTUNE_BACKEND_3D={selected['backend']}")
    if dim == 4:
        print(f"FFTM_AUTOTUNE_SLAB_NATIVE_XW={selected.get('slab_native_xw', 0)}")
        print(
            "FFTM_AUTOTUNE_SLAB_XW_NATIVE_SPECTRAL_LAYOUT="
            f"{selected.get('slab_native_xw_native_spectral_layout', 0)}"
        )
        print(f"FFTM_AUTOTUNE_4D_PENCIL_SAME_ZW_PEER_PAIRED={selected.get('pencil_same_zw_peer_paired', 0)}")
        print(f"FFTM_AUTOTUNE_4D_PENCIL_SAME_ZW_NATIVE_LAYOUT={selected.get('pencil_same_zw_native_layout', 0)}")
        print(f"FFTM_AUTOTUNE_4D_PENCIL_DEGENERATE_XW_SLAB_PATH={selected.get('pencil_degenerate_xw_slab_path', 0)}")
        print(
            "FFTM_AUTOTUNE_4D_PENCIL_DEGENERATE_LOCAL_TRANSPOSES="
            f"{selected.get('pencil_degenerate_local_transposes', 0)}"
        )
        print(
            "FFTM_AUTOTUNE_4D_PENCIL_DEGENERATE_SAME_XW_NATIVE="
            f"{selected.get('pencil_degenerate_same_xw_native', 0)}"
        )
        print(
            "FFTM_AUTOTUNE_4D_PENCIL_DEGENERATE_WZ_SLICED_Z_FFT="
            f"{selected.get('pencil_degenerate_wz_sliced_z_fft', 0)}"
        )
        print(f"FFTM_AUTOTUNE_4D_NATIVE_XW_DIRECT_LAYOUT={selected.get('native_xw_direct_layout', 0)}")
        print(
            "FFTM_AUTOTUNE_4D_NATIVE_XW_PROTOCOL="
            f"{'chunked' if selected.get('native_xw_chunked_transport') else 'single'}"
        )
        print(f"FFTM_AUTOTUNE_4D_NATIVE_XW_CHUNK_MIB={selected.get('native_xw_chunk_mib', 1024)}")
    if metrics.get("avg_wall_ms") is not None:
        print(f"FFTM_AUTOTUNE_AVG_WALL_MS={metrics['avg_wall_ms']}")
    if metrics.get("min_wall_ms") is not None:
        print(f"FFTM_AUTOTUNE_MIN_WALL_MS={metrics['min_wall_ms']}")
    if entry.get("source"):
        print(f"FFTM_AUTOTUNE_SOURCE={entry['source']}")

    if key["library"] == "fftm":
        print(f"FFTM_GPU_COUNTS={key['num_gpus']}")
        if dim == 4:
            print("FFTM_BENCHMARK_SIZES_3D=none")
            print("FFTM_BENCHMARK_SIZES_4D=none")
            if len(set(sizes)) == 1:
                print(f"FFTM_FIXED_SCALING_SIZES_4D={sizes[0]}")
            print("FFTM_STRATEGIES_3D=")
            if selected.get("strategy"):
                print(f"FFTM_STRATEGIES_4D={selected['strategy']}")
            if selected.get("mode"):
                print(f"FFTM_MODES={selected['mode']}")
            print(f"FFTM_4D_SLAB_XW_TRANSPOSES={'native' if selected.get('slab_native_xw') else 'staged'}")
            print(
                "FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS="
                f"{'native' if selected.get('slab_native_xw_native_spectral_layout') else 'public'}"
            )
            print(
                "FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS="
                f"{'batched' if selected.get('slab_native_xw_batched_peer_kernels') else 'legacy'}"
            )
            print(
                "FFTM_4D_SLAB_XW_KERNEL_LAYOUTS="
                f"{'tensor' if selected.get('slab_native_xw_tensor_coalesced_kernels') else 'buffer'}"
            )
            print(
                "FFTM_4D_SLAB_XW_VECTOR4_KERNELS="
                f"{'on' if selected.get('slab_native_xw_vector4_kernels') else 'off'}"
            )
            print(
                "FFTM_4D_SLAB_XW_TILED_KERNELS="
                f"{'on' if selected.get('slab_native_xw_tiled_kernels') else 'off'}"
            )
            print(
                "FFTM_4D_SLAB_XW_LAYOUT_STAGES="
                f"{'stage' if selected.get('slab_native_xw_layout_stage') else 'direct'}"
            )
            print(
                "FFTM_4D_PENCIL_SAME_ZW_PEER_PAIRED="
                f"{'on' if selected.get('pencil_same_zw_peer_paired') else 'off'}"
            )
            print(
                "FFTM_4D_PENCIL_SAME_ZW_NATIVE_LAYOUTS="
                f"{'native' if selected.get('pencil_same_zw_native_layout') else 'base'}"
            )
            print(
                "FFTM_4D_PENCIL_DEGENERATE_XW_SLAB_PATHS="
                f"{'on' if selected.get('pencil_degenerate_xw_slab_path') else 'off'}"
            )
            print(
                "FFTM_4D_PENCIL_DEGENERATE_LOCAL_TRANSPOSES="
                f"{'on' if selected.get('pencil_degenerate_local_transposes') else 'off'}"
            )
            print(
                "FFTM_4D_PENCIL_DEGENERATE_SAME_XW_NATIVE="
                f"{'on' if selected.get('pencil_degenerate_same_xw_native') else 'off'}"
            )
            print(
                "FFTM_4D_PENCIL_DEGENERATE_WZ_SLICED_Z_FFT="
                f"{'on' if selected.get('pencil_degenerate_wz_sliced_z_fft') else 'off'}"
            )
            print(
                "FFTM_4D_NATIVE_XW_DIRECT_LAYOUTS="
                f"{'on' if selected.get('native_xw_direct_layout') else 'off'}"
            )
            print(
                "FFTM_4D_NATIVE_XW_PROTOCOLS="
                f"{'chunked' if selected.get('native_xw_chunked_transport') else 'single'}"
            )
            print(f"FFTM_4D_NATIVE_XW_CHUNK_MIB={selected.get('native_xw_chunk_mib', 1024)}")
            print(f"FFTM_USE_FFT_EXEC_NO_SYNC={selected.get('fft_exec_no_sync', 0)}")
            if selected.get("diagnostic_only"):
                print("FFTM_AUTOTUNE_DIAGNOSTIC_ONLY=1")
            return

        print("FFTM_BENCHMARK_SIZES_3D=none")
        print(f"FFTM_FIXED_SCALING_SIZES_3D={sizes[0]}")
        if selected.get("strategy"):
            print(f"FFTM_STRATEGIES_3D={selected['strategy']}")
        if selected.get("mode"):
            print(f"FFTM_MODES={selected['mode']}")
        if selected.get("pencil_layout"):
            print(f"FFTM_PENCIL_LAYOUTS={selected['pencil_layout']}")
        if selected.get("pencil_pipeline"):
            print(f"FFTM_PENCIL_PIPELINES={selected['pencil_pipeline']}")
        if selected.get("backend"):
            print(f"FFTM_3D_BACKENDS={selected['backend']}")
        if selected.get("grid_orientation") in {"default", "reversed"}:
            print(f"FFTM_PENCIL_PENCIL_GRID_ORIENTATIONS={selected['grid_orientation']}")
        if selected.get("large_count_p2p_transport"):
            print(f"FFTM_LARGE_COUNT_P2P_TRANSPORTS={selected['large_count_p2p_transport']}")
        print(f"FFTM_USE_FFT_EXEC_NO_SYNC={selected.get('fft_exec_no_sync', 0)}")
        print(f"FFTM_USE_PERSISTENT_P2P={selected.get('persistent_p2p', 0)}")
        print(f"FFTM_USE_READY_P2P_SEND={selected.get('ready_p2p_send', 0)}")
        print(f"FFTM_USE_STABLE_FORWARD_BYTE_SEND_BUFFER={selected.get('stable_forward_byte_send_buffer', 0)}")
        print(
            "FFTM_USE_READY_STABLE_FORWARD_BYTE_SEND_BUFFER="
            f"{selected.get('ready_stable_forward_byte_send_buffer', 0)}"
        )
        print(f"FFTM_USE_CONTIGUOUS_FORWARD_BYTE_SEND={selected.get('contiguous_forward_byte_send', 0)}")
        if selected.get("contiguous_forward_send_mode"):
            print(f"FFTM_CONTIGUOUS_FORWARD_SEND_MODES={selected['contiguous_forward_send_mode']}")
        print(f"FFTM_USE_PHYSICAL_FORWARD_PEER_EXCHANGE={selected.get('physical_forward_peer_exchange', 0)}")
        print(f"FFTM_USE_NATIVE_BACKWARD_SECOND_PEER_LOOP={selected.get('native_backward_second_peer_loop', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_DEFAULT_Z_LAYOUT={selected.get('native_opt0_default_z_layout', 0)}")
        print(
            "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_BUFFER_TOPOLOGY="
            f"{selected.get('native_opt0_reference_y_buffer_topology', 0)}"
        )
        print(f"FFTM_USE_NATIVE_OPT0_COMPACT_Y_WORKAREA={selected.get('native_opt0_compact_y_workarea', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_TIGHT_Y_PLAN_SEQUENCE={selected.get('native_opt0_tight_y_plan_sequence', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_SHARED_Y_PLAN_HANDLES={selected.get('native_opt0_shared_y_plan_handles', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_Y_GROUP_DEVICE_SYNC={selected.get('native_opt0_y_group_device_sync', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_Y_NO_SYNC_EXEC={selected.get('native_opt0_y_no_sync_exec', 1)}")
        print(
            "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_ARRAY_EXECUTOR="
            f"{selected.get('native_opt0_raw_y_plan_array_executor', 0)}"
        )
        print(
            "FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_LIFECYCLE="
            f"{selected.get('native_opt0_reference_y_plan_lifecycle', 0)}"
        )
        print(f"FFTM_USE_NATIVE_OPT0_REFERENCE_Y_PLAN_BUNDLE={selected.get('native_opt0_reference_y_plan_bundle', 0)}")
        print(f"FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE={selected.get('native_opt0_raw_y_plan_bundle', 0)}")
        print(
            "FFTM_USE_NATIVE_OPT0_Y_PLAN_BUNDLE_STREAM_FIRST="
            f"{selected.get('native_opt0_y_plan_bundle_stream_first', 0)}"
        )
        print(
            "FFTM_USE_NATIVE_OPT0_RAW_Y_PLAN_BUNDLE_REFERENCE_STREAMS="
            f"{selected.get('native_opt0_raw_y_plan_bundle_reference_streams', 0)}"
        )
        print(
            "FFTM_USE_NATIVE_OPT0_REFERENCE_LOCAL_PLAN_CONTEXT="
            f"{selected.get('native_opt0_reference_local_plan_context', 0)}"
        )
        if selected.get("reference_only"):
            print("FFTM_AUTOTUNE_REFERENCE_ONLY=1")
        if selected.get("diagnostic_only"):
            print("FFTM_AUTOTUNE_DIAGNOSTIC_ONLY=1")
            print("FFTM_ALLOW_NATIVE_OPT0_DIAGNOSTIC_VARIANTS=1")
        if selected.get("fft_exec_no_sync"):
            print("FFTM_ALLOW_FFT_EXEC_NO_SYNC=1")
    elif key["library"] == "fftm3d":
        print(f"FFTM3D_GPU_COUNTS={key['num_gpus']}")
        print(f"FFTM3D_BENCHMARK_SIZES_3D={sizes[0]}")
        if len(grid) == 2:
            print(f"FFTM3D_GRIDS={key['num_gpus']}:{grid[0]}x{grid[1]}")
        if selected.get("opt"):
            print(f"FFTM3D_OPTS={str(selected['opt']).replace('opt', '')}")
        if selected.get("egger_variant"):
            print(f"FFTM3D_EGGER_VARIANTS={selected['egger_variant']}")
        for env_key, selected_key in [
            ("FFTM3D_DEVICE_MAPS", "device_map"),
            ("FFTM3D_COMM_METHODS", "comm"),
            ("FFTM3D_SEND_METHODS", "send"),
            ("FFTM3D_STORAGE_MODES", "storage"),
            ("FFTM3D_MPI_BACKENDS", "mpi_backend"),
            ("FFTM3D_RUNTIME_BACKENDS", "runtime_backend"),
        ]:
            if selected.get(selected_key):
                print(f"{env_key}={selected[selected_key]}")


def query_cache(args: argparse.Namespace) -> int:
    cache_path = Path(args.cache).resolve()
    if not cache_path.exists():
        print(f"Autotune cache does not exist: {cache_path}", file=sys.stderr)
        return 2
    cache = load_cache(cache_path)
    entry = find_entry(cache, args)
    if entry is None:
        print(
            "No autotune cache entry matched "
            f"library={args.library} dim={args.dim} num_gpus={args.num_gpus} "
            f"sizes={'x'.join(str(v) for v in args.sizes)} "
            f"strategy={args.strategy or 'any'} mode={args.mode or 'any'} "
            f"backend={args.backend or 'any'} layout={args.pencil_layout or 'any'} "
            f"pipeline={args.pencil_pipeline or 'any'} grid={args.grid or 'any'} "
            f"grid_orientation={args.grid_orientation or 'any'} "
            f"large_count={args.large_count_p2p_transport or 'any'} "
            f"slab_xw={args.slab_native_xw or 'any'} "
            f"slab_xw_spectral={args.slab_native_xw_native_spectral_layout or 'any'} "
            f"pencil_same_zw_peer={args.pencil_same_zw_peer_paired or 'any'} "
            f"pencil_same_zw_layout={args.pencil_same_zw_native_layout or 'any'} "
            f"pencil_degen_xw={args.pencil_degenerate_xw_slab_path or 'any'} "
            f"pencil_degen_local={args.pencil_degenerate_local_transposes or 'any'} "
            f"pencil_degen_same_xw={args.pencil_degenerate_same_xw_native or 'any'} "
            f"pencil_degen_wz={args.pencil_degenerate_wz_sliced_z_fft or 'any'}",
            file=sys.stderr,
        )
        return 1
    if args.format == "env":
        print_env(entry)
    else:
        print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    update = subparsers.add_parser("update", help="update a cache from one or more benchmark result directories")
    update.add_argument("result_dirs", nargs="+")
    update.add_argument("--output", default="build/resutls_stats/fftm_autotune_cache.json")
    update.add_argument("--reset", action="store_true", help="Start from an empty cache before adding result_dirs.")
    update.add_argument(
        "--include-strategies",
        default="",
        help="Comma-separated strategy allow-list for entries collected from result_dirs.",
    )
    update.add_argument(
        "--exclude-strategies",
        default="",
        help="Comma-separated strategy deny-list for entries collected from result_dirs.",
    )
    update.set_defaults(func=update_cache)

    query = subparsers.add_parser("query", help="query the best cache entry matching optional policy filters")
    query.add_argument("--cache", default="build/resutls_stats/fftm_autotune_cache.json")
    query.add_argument("--library", choices=["fftm", "fftm3d"], required=True)
    query.add_argument("--dim", type=int, default=3)
    query.add_argument("--num-gpus", type=int, required=True)
    query.add_argument("--sizes", type=int, nargs="+", required=True)
    query.add_argument("--hostname", default="")
    query.add_argument("--gpu-name", default="")
    query.add_argument("--strategy", default="", help="Optional selected strategy filter, e.g. pencil-pencil.")
    query.add_argument("--mode", default="", help="Optional selected redistribution mode filter.")
    query.add_argument("--backend", default="", help="Optional FFTM 3D backend filter, e.g. native.")
    query.add_argument("--pencil-layout", default="", help="Optional pencil layout/fftm3d opt filter.")
    query.add_argument("--pencil-pipeline", default="", help="Optional pencil pipeline filter.")
    query.add_argument("--grid", default="", help="Optional grid filter, e.g. 4x2.")
    query.add_argument("--grid-orientation", default="", help="Optional grid orientation filter.")
    query.add_argument("--large-count-p2p-transport", default="", help="Optional large-count transport filter.")
    query.add_argument("--slab-native-xw", default="", help="Optional 4D slab-slab native-XW filter: 0 or 1.")
    query.add_argument(
        "--slab-native-xw-native-spectral-layout",
        default="",
        help="Optional 4D slab-slab native spectral layout filter: 0 or 1.",
    )
    query.add_argument("--slab-native-xw-layout-stage", default="", help="Optional 4D XW layout-stage filter: 0 or 1.")
    query.add_argument(
        "--pencil-same-zw-peer-paired",
        default="",
        help="Optional 4D pencil same_zw p2p-waitany peer-paired filter: 0/1 or off/on.",
    )
    query.add_argument(
        "--pencil-same-zw-native-layout",
        default="",
        help="Optional 4D pencil same_zw native message-layout filter: 0/1 or base/native.",
    )
    query.add_argument(
        "--pencil-degenerate-xw-slab-path",
        default="",
        help="Optional 4D pencil degenerate p1=1,p3=1 native slab-path filter: 0/1 or off/on.",
    )
    query.add_argument(
        "--pencil-degenerate-local-transposes",
        default="",
        help="Optional 4D pencil degenerate p1=1,p3=1 local transpose/alias filter: 0/1 or off/on.",
    )
    query.add_argument(
        "--pencil-degenerate-same-xw-native",
        default="",
        help="Optional 4D pencil degenerate p1=1,p3=1 same_xw native p2p scheduling filter: 0/1 or off/on.",
    )
    query.add_argument(
        "--pencil-degenerate-wz-sliced-z-fft",
        default="",
        help="Optional 4D pencil degenerate p1=1,p3=1 WZ sliced-Z FFT filter: 0/1 or off/on.",
    )
    query.add_argument(
        "--include-reference-only",
        action="store_true",
        help="Allow reference-only fftm3d-as-FFTM candidates in FFTM cache queries.",
    )
    query.add_argument(
        "--include-diagnostic",
        action="store_true",
        help="Allow diagnostic-only native candidates in FFTM cache queries.",
    )
    query.add_argument("--format", choices=["json", "env"], default="json")
    query.set_defaults(func=query_cache)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
