#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


FFTM_STRATEGIES_3D = ("slab-pencil", "pencil-slab", "pencil-pencil")
FFTM_STRATEGIES_4D = ("pencil-pencil", "slab-slab")
FFTM_MODES = ("p2p-waitall", "p2p-waitany", "alltoallv")
FFTM_P2P_VARIANTS = ("value-packed", "datatype-direct", "byte-packed", "byte-direct")
FFTM_P2P_SCHEDULERS = ("main", "send-thread", "persistent", "send-thread-persistent")
FFTM_PENCIL_LAYOUTS = ("auto", "opt0", "opt1", "legacy")
FFTM_PENCIL_PIPELINES = ("staged", "fused", "reference", "reference-parity")
FFTM_PENCIL_PIPELINE_ALIASES = {
    "native-compatible": "reference",
    "reference-compatible": "reference",
    "compatible": "reference",
    "egger": "reference",
    "native-parity": "reference-parity",
    "reference_parity": "reference-parity",
    "compatible-parity": "reference-parity",
    "native_parity": "reference-parity",
    "compatible_parity": "reference-parity",
    "egger-parity": "reference-parity",
    "egger_parity": "reference-parity",
}
FFTM_LARGE_COUNT_P2P_TRANSPORTS = ("hindexed", "mpi-count", "element-count", "chunked")
FFTM_CONTIGUOUS_FORWARD_SEND_MODES = ("single", "chunked")
FFTM_4D_SLAB_XW_TRANSPOSES = ("staged", "native")
FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS = ("legacy", "batched")
FFTM_4D_SLAB_XW_KERNEL_LAYOUTS = ("buffer", "tensor")
FFTM_4D_SLAB_XW_VECTOR4_KERNELS = ("off", "on")
FFTM_4D_SLAB_XW_TILED_KERNELS = ("off", "on")
FFTM_4D_SLAB_XW_LAYOUT_STAGES = ("direct", "stage")
FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS = ("public", "native")
FFTM_4D_NATIVE_XW_DIRECT_LAYOUTS = ("off", "on")
FFTM_4D_NATIVE_XW_PROTOCOLS = ("single", "chunked")
FFTM_4D_NATIVE_XW_CHUNK_WINDOWS = (0, 1, 2, 4)
FFTM_4D_NATIVE_XW_COMPACT_STAGINGS = ("off", "on")
FFTM_4D_PENCIL_SAME_ZW_PEER_PAIRED = ("off", "on")
FFTM_4D_PENCIL_SAME_ZW_NATIVE_LAYOUTS = ("base", "native")
FFTM_4D_PENCIL_DEGENERATE_XW_SLAB_PATHS = ("off", "on")
FFTM_4D_PENCIL_DEGENERATE_LOCAL_TRANSPOSES = ("off", "on")
FFTM_4D_PENCIL_DEGENERATE_SAME_XW_NATIVE = ("off", "on")
FFTM_4D_PENCIL_DEGENERATE_WZ_SLICED_Z_FFT = ("off", "on")
FFTM_4D_PENCIL_GRID_ORIENTATIONS = ("configured", "default", "x-heavy", "y-heavy", "z-heavy", "all")
FFTM_NATIVE_BACKWARD_SECOND_PEER_LOOP_MODES = ("off", "on")
FFTM_3D_BACKENDS = ("native", "fftm3d-scfd-fft-facade")
FFTM_NATIVE_OPT0_Y_CROSS_FACTORY_MODES = (
    "ref-only",
    "native-current-only",
    "native-current-nosync-only",
    "native-exact-only",
    "native-owned-only",
    "native-direct-only",
    "native-direct-nosync-only",
    "native-minimal-only",
    "native-minimal-nosync-only",
    "ref-then-native-current",
    "native-current-then-ref",
    "ref-then-native-current-nosync",
    "native-current-nosync-then-ref",
    "ref-then-native-exact",
    "native-exact-then-ref",
    "ref-then-native-owned",
    "native-owned-then-ref",
    "ref-then-native-direct",
    "native-direct-then-ref",
    "ref-then-native-direct-nosync",
    "native-direct-nosync-then-ref",
    "ref-then-native-minimal",
    "native-minimal-then-ref",
    "ref-then-native-minimal-nosync",
    "native-minimal-nosync-then-ref",
)
FFTM_NATIVE_OPT0_Y_EXECUTOR_VARIANTS = (
    "virtual-stream",
    "virtual-device",
    "tight-stream",
    "tight-device",
    "opaque-stream",
    "opaque-device",
    "shared-opaque-stream",
    "shared-opaque-device",
    "shared-opaque-nosync-stream",
    "shared-opaque-nosync-device",
    "ref-life-stream",
    "ref-life-device",
    "ref-bundle-stream",
    "ref-bundle-device",
    "raw-bundle-stream",
    "raw-bundle-device",
    "raw-bundle-streamfirst-stream",
    "raw-bundle-streamfirst-device",
    "raw-bundle-reference-streams-stream",
    "raw-bundle-reference-streams-device",
    "raw-bundle-reference-streams-nosync-stream",
    "raw-bundle-reference-streams-nosync-device",
    "raw-bundle-reference-streamfirst-stream",
    "raw-bundle-reference-streamfirst-device",
    "context-bundle-stream",
    "context-bundle-device",
    "context-bundle-nosync-stream",
    "context-bundle-nosync-device",
)
FFTS_STRATEGIES_4D = ("pencil-direct", "pencil-memcpy", "slab-direct", "slab-memcpy")

SUMMARY_KEY_RE = re.compile(r"([A-Za-z0-9_]+)=((?:\([^)]*\))|[^,]+)")
SUMMARY_TOKEN_RE = re.compile(r"(?<!\S)([A-Za-z0-9_]+)=([^\s,]+)")
PROFILE_LINE_RE = re.compile(r"^\[\s*(?P<name>.+?):\s+(?P<ms>[0-9.]+)\s+ms\]\s+\((?P<pct>[0-9.]+)%\)$")
MEMORY_LINE_RE = re.compile(
    r"^\s{2,}(?P<name>[^:]+): current\(sum/max/avg\)="
    r"(?P<current_sum>[0-9]+)/(?P<current_max>[0-9]+)/(?P<current_avg>[0-9]+) B "
    r"\((?P<current_sum_mib>[0-9.]+)/(?P<current_max_mib>[0-9.]+)/(?P<current_avg_mib>[0-9.]+) MiB\), "
    r"peak\(sum/max/avg\)="
    r"(?P<peak_sum>[0-9]+)/(?P<peak_max>[0-9]+)/(?P<peak_avg>[0-9]+) B "
    r"\((?P<peak_sum_mib>[0-9.]+)/(?P<peak_max_mib>[0-9.]+)/(?P<peak_avg_mib>[0-9.]+) MiB\)$"
)

PHASE_PROBE = "probe"
PHASE_MEASURE = "measure"


def mib_to_bytes(value_mib: float) -> int:
    return int(value_mib * 1024.0 * 1024.0)


def bytes_to_mib(value_bytes: int) -> float:
    return float(value_bytes) / (1024.0 * 1024.0)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def short_log_slug(value: str, max_len: int = 96) -> str:
    slug = slugify(value)
    if len(slug) <= max_len:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:12]
    head_len = max(1, max_len - len(digest) - 1)
    return f"{slug[:head_len].rstrip('-')}_{digest}"


def split_top_level_csv(payload: str) -> List[str]:
    items: List[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(payload):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            items.append(payload[start:idx].strip())
            start = idx + 1
    tail = payload[start:].strip()
    if tail:
        items.append(tail)
    return items


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith("(") and value.endswith(")"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in split_top_level_csv(inner)]
    if re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
        return float(value)
    return value


def parse_int_list(value: str) -> List[int]:
    value = (value or "").strip().lower()
    if not value or value in ("none", "off", "skip"):
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_gpu_size_map(value: str) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    if not value:
        return result
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            gpu_text, size_text = item.split(":", 1)
        elif "=" in item:
            gpu_text, size_text = item.split("=", 1)
        else:
            raise ValueError(
                "--extra-sizes-3d-by-gpu entries must have the form GPU:SIZES, "
                "for example '1:1050;2:1344' or '1:540,729;2:686,900'"
            )
        gpu_count = int(gpu_text.strip())
        sizes = parse_int_list(size_text)
        if not sizes:
            continue
        result.setdefault(gpu_count, [])
        result[gpu_count].extend(sizes)
    return {gpu: sorted(set(sizes)) for gpu, sizes in result.items()}


def parse_p2p_variants(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_P2P_VARIANTS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in FFTM_P2P_VARIANTS]
    if unknown:
        raise ValueError(
            f"unknown --p2p-variants value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_P2P_VARIANTS)}"
        )
    return selected or [None]


def parse_p2p_schedulers(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_P2P_SCHEDULERS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in FFTM_P2P_SCHEDULERS]
    if unknown:
        raise ValueError(
            f"unknown --p2p-schedulers value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_P2P_SCHEDULERS)}"
        )
    return selected or [None]


def parse_pencil_layouts(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_PENCIL_LAYOUTS)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in FFTM_PENCIL_LAYOUTS]
    if unknown:
        raise ValueError(
            f"unknown --pencil-layouts value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_PENCIL_LAYOUTS)}"
        )
    return selected or [None]


def parse_pencil_pipelines(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_PENCIL_PIPELINES)
    selected = []
    for item in value.split(","):
        key = item.strip()
        if not key:
            continue
        selected.append(FFTM_PENCIL_PIPELINE_ALIASES.get(key, key))
    unknown = [item for item in selected if item not in FFTM_PENCIL_PIPELINES]
    if unknown:
        allowed_aliases = sorted(FFTM_PENCIL_PIPELINE_ALIASES)
        raise ValueError(
            f"unknown --pencil-pipelines value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_PENCIL_PIPELINES)}, "
            f"{', '.join(allowed_aliases)}"
        )
    return selected or [None]


def parse_large_count_p2p_transports(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_LARGE_COUNT_P2P_TRANSPORTS)
    aliases = {
        "complex-count": "element-count",
        "complex_count": "element-count",
        "element_count": "element-count",
        "mpi_count": "mpi-count",
    }
    selected = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        selected.append(aliases.get(item, item.replace("_", "-")))
    unknown = [item for item in selected if item not in FFTM_LARGE_COUNT_P2P_TRANSPORTS]
    if unknown:
        raise ValueError(
            f"unknown --large-count-p2p-transports value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_LARGE_COUNT_P2P_TRANSPORTS)}"
        )
    return selected or [None]


def parse_contiguous_forward_send_modes(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_CONTIGUOUS_FORWARD_SEND_MODES)
    selected = []
    for item in value.split(","):
        item = item.strip().replace("_", "-")
        if item:
            selected.append(item)
    unknown = [item for item in selected if item not in FFTM_CONTIGUOUS_FORWARD_SEND_MODES]
    if unknown:
        raise ValueError(
            f"unknown --contiguous-forward-send-modes value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_CONTIGUOUS_FORWARD_SEND_MODES)}"
        )
    return selected or [None]


def parse_native_backward_second_peer_loop_modes(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value in ("all", "both"):
        return [False, True]
    aliases = {
        "0": False,
        "false": False,
        "no": False,
        "off": False,
        "disabled": False,
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "enabled": True,
    }
    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key not in aliases:
            unknown.append(item.strip())
            continue
        selected.append(aliases[key])
    if unknown:
        raise ValueError(
            "unknown --native-backward-second-peer-loop-modes value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, both, off,on"
        )
    return selected or [None]


def parse_fftm_3d_backends(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_3D_BACKENDS)
    aliases = {
        "fftm": "native",
        "default": "native",
        "scfd-fft-facade": "fftm3d-scfd-fft-facade",
        "fftm3d_scfd_fft_facade": "fftm3d-scfd-fft-facade",
    }
    selected = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        selected.append(aliases.get(item, item))
    unknown = [item for item in selected if item not in FFTM_3D_BACKENDS]
    if unknown:
        raise ValueError(
            f"unknown --fftm-3d-backends value(s): {', '.join(unknown)}; "
            f"allowed: configured, all, {', '.join(FFTM_3D_BACKENDS)}"
        )
    return selected or [None]


NATIVE_OPT0_Y_EXECUTOR_VARIANT_ALIASES = {
    "configured": None,
    "default": None,
    "current": "shared-opaque-nosync-device",
    "best": "shared-opaque-nosync-device",
    "production": "shared-opaque-nosync-device",
    "validated": "shared-opaque-nosync-device",
    "seq-virtual-stream": "virtual-stream",
    "sequence-virtual-stream": "virtual-stream",
    "virtual-per-stream": "virtual-stream",
    "seq-virtual-device": "virtual-device",
    "sequence-virtual-device": "virtual-device",
    "seq-tight-stream": "tight-stream",
    "sequence-tight-stream": "tight-stream",
    "tight-per-stream": "tight-stream",
    "seq-tight-device": "tight-device",
    "sequence-tight-device": "tight-device",
    "seq-opaque-stream": "opaque-stream",
    "sequence-opaque-stream": "opaque-stream",
    "opaque-per-stream": "opaque-stream",
    "seq-opaque-device": "opaque-device",
    "sequence-opaque-device": "opaque-device",
    "shared-stream": "shared-opaque-stream",
    "shared-opaque-per-stream": "shared-opaque-stream",
    "shared-device": "shared-opaque-device",
    "shared-opaque": "shared-opaque-device",
    "sync": "shared-opaque-device",
    "no-sync": "shared-opaque-nosync-device",
    "nosync": "shared-opaque-nosync-device",
    "shared-nosync-stream": "shared-opaque-nosync-stream",
    "shared-nosync-device": "shared-opaque-nosync-device",
    "ref-lifecycle-stream": "ref-life-stream",
    "reference-lifecycle-stream": "ref-life-stream",
    "ref-lifecycle-device": "ref-life-device",
    "reference-lifecycle-device": "ref-life-device",
    "reference-bundle-stream": "ref-bundle-stream",
    "reference-bundle-device": "ref-bundle-device",
    "raw-reference-bundle-stream": "raw-bundle-stream",
    "raw-reference-bundle-device": "raw-bundle-device",
    "raw-streamfirst-stream": "raw-bundle-streamfirst-stream",
    "raw-streamfirst-device": "raw-bundle-streamfirst-device",
    "raw-reference-streams-stream": "raw-bundle-reference-streams-stream",
    "raw-reference-streams-device": "raw-bundle-reference-streams-device",
    "raw-reference-streams-nosync-stream": "raw-bundle-reference-streams-nosync-stream",
    "raw-reference-streams-nosync-device": "raw-bundle-reference-streams-nosync-device",
    "raw-reference-streamfirst-stream": "raw-bundle-reference-streamfirst-stream",
    "raw-reference-streamfirst-device": "raw-bundle-reference-streamfirst-device",
    "raw-egger-streamfirst-stream": "raw-bundle-reference-streamfirst-stream",
    "raw-egger-streamfirst-device": "raw-bundle-reference-streamfirst-device",
    "context-stream": "context-bundle-stream",
    "context-device": "context-bundle-device",
    "context-nosync-stream": "context-bundle-nosync-stream",
    "context-nosync-device": "context-bundle-nosync-device",
    "reference-context-stream": "context-bundle-stream",
    "reference-context-device": "context-bundle-device",
    "local-context-stream": "context-bundle-stream",
    "local-context-device": "context-bundle-device",
}


NATIVE_OPT0_Y_EXECUTOR_VARIANT_FLAGS: Dict[str, Dict[str, bool]] = {
    "virtual-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "virtual-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "tight-stream": {
        "tight": True,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "tight-device": {
        "tight": True,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "opaque-stream": {
        "tight": True,
        "shared": False,
        "device_sync": False,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "opaque-device": {
        "tight": True,
        "shared": False,
        "device_sync": True,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "shared-opaque-stream": {
        "tight": True,
        "shared": True,
        "device_sync": False,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "shared-opaque-device": {
        "tight": True,
        "shared": True,
        "device_sync": True,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "shared-opaque-nosync-stream": {
        "tight": True,
        "shared": True,
        "device_sync": False,
        "no_sync_exec": True,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "shared-opaque-nosync-device": {
        "tight": True,
        "shared": True,
        "device_sync": True,
        "no_sync_exec": True,
        "opaque": True,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "ref-life-stream": {
        "tight": True,
        "shared": True,
        "device_sync": False,
        "opaque": True,
        "reference_lifecycle": True,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "ref-life-device": {
        "tight": True,
        "shared": True,
        "device_sync": True,
        "opaque": True,
        "reference_lifecycle": True,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "ref-bundle-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": True,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "ref-bundle-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": True,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
    },
    "raw-bundle-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": False,
    },
    "raw-bundle-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": False,
    },
    "raw-bundle-streamfirst-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": True,
        "reference_streams": False,
    },
    "raw-bundle-streamfirst-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": True,
        "reference_streams": False,
    },
    "raw-bundle-reference-streams-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": True,
    },
    "raw-bundle-reference-streams-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": True,
    },
    "raw-bundle-reference-streams-nosync-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "no_sync_exec": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": True,
    },
    "raw-bundle-reference-streams-nosync-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "no_sync_exec": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": False,
        "reference_streams": True,
    },
    "raw-bundle-reference-streamfirst-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": True,
        "reference_streams": True,
    },
    "raw-bundle-reference-streamfirst-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": True,
        "stream_first": True,
        "reference_streams": True,
    },
    "context-bundle-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
        "local_context": True,
    },
    "context-bundle-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
        "local_context": True,
    },
    "context-bundle-nosync-stream": {
        "tight": False,
        "shared": False,
        "device_sync": False,
        "no_sync_exec": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
        "local_context": True,
    },
    "context-bundle-nosync-device": {
        "tight": False,
        "shared": False,
        "device_sync": True,
        "no_sync_exec": True,
        "opaque": False,
        "reference_lifecycle": False,
        "reference_bundle": False,
        "raw_bundle": False,
        "stream_first": False,
        "reference_streams": False,
        "local_context": True,
    },
}

for _native_opt0_y_variant_flags in NATIVE_OPT0_Y_EXECUTOR_VARIANT_FLAGS.values():
    _native_opt0_y_variant_flags.setdefault("local_context", False)
    _native_opt0_y_variant_flags.setdefault("no_sync_exec", False)


def parse_native_opt0_y_executor_variants(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "all":
        return list(FFTM_NATIVE_OPT0_Y_EXECUTOR_VARIANTS)
    selected: List[Optional[str]] = []
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        mapped = NATIVE_OPT0_Y_EXECUTOR_VARIANT_ALIASES.get(key, key)
        if mapped is None:
            selected.append(None)
        elif mapped in FFTM_NATIVE_OPT0_Y_EXECUTOR_VARIANTS:
            selected.append(mapped)
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "all", *FFTM_NATIVE_OPT0_Y_EXECUTOR_VARIANTS]
        raise ValueError(
            "unknown --native-opt0-y-executor-variants value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def native_opt0_y_executor_variants_for_spec(
    dim: int,
    transport: str,
    strategy: str,
    mode: str,
    pencil_layout: Optional[str],
    pencil_pipeline: Optional[str],
    fftm_3d_backend: Optional[str],
    selected: Sequence[Optional[str]],
) -> List[Optional[str]]:
    if (
        dim == 3
        and transport == "cuda_aware"
        and strategy == "pencil-pencil"
        and mode.startswith("p2p-")
        and pencil_layout == "opt0"
        and pencil_pipeline in ("reference", "reference-parity")
        and fftm_3d_backend in (None, "native")
    ):
        return list(selected)
    return [None]


def native_opt0_y_executor_variant_flags(
    variant: Optional[str],
    *,
    default_tight: bool,
    default_shared: bool,
    default_device_sync: bool,
    default_opaque: bool,
    default_reference_lifecycle: bool,
    default_reference_bundle: bool,
    default_raw_bundle: bool,
    default_stream_first: bool,
    default_reference_streams: bool,
    default_local_context: bool,
    default_no_sync_exec: bool,
) -> Dict[str, bool]:
    if variant is None:
        return {
            "tight": default_tight,
            "shared": default_shared,
            "device_sync": default_device_sync,
            "opaque": default_opaque,
            "reference_lifecycle": default_reference_lifecycle,
            "reference_bundle": default_reference_bundle,
            "raw_bundle": default_raw_bundle,
            "stream_first": default_stream_first,
            "reference_streams": default_reference_streams,
            "local_context": default_local_context,
            "no_sync_exec": default_no_sync_exec,
        }
    result = dict(NATIVE_OPT0_Y_EXECUTOR_VARIANT_FLAGS[variant])
    result.setdefault("local_context", False)
    result.setdefault("no_sync_exec", False)
    return result


NATIVE_OPT0_DIAGNOSTIC_Y_EXECUTOR_FLAG_KEYS = (
    "reference_lifecycle",
    "reference_bundle",
    "raw_bundle",
    "stream_first",
    "reference_streams",
    "local_context",
)


def native_opt0_y_executor_flags_are_diagnostic(flags: Dict[str, bool]) -> bool:
    return any(bool(flags.get(key, False)) for key in NATIVE_OPT0_DIAGNOSTIC_Y_EXECUTOR_FLAG_KEYS)


def p2p_variant_flags(
    variant: Optional[str],
    default_direct_backward_receive: bool,
    default_direct_p2p_cuda_aware: bool,
    default_p2p_byte_transfer: bool,
) -> Tuple[bool, bool, bool]:
    if variant is None or variant == "nca-staging":
        return default_direct_backward_receive, default_direct_p2p_cuda_aware, default_p2p_byte_transfer
    if variant == "value-packed":
        return False, True, False
    if variant == "datatype-direct":
        return True, True, False
    if variant == "byte-packed":
        return False, True, True
    if variant == "byte-direct":
        return True, True, True
    raise ValueError(f"unknown p2p variant: {variant}")


def p2p_variants_for_spec(
    dim: int, transport: str, strategy: str, mode: str, selected: Sequence[Optional[str]]
) -> List[Optional[str]]:
    if dim == 3 and transport == "cuda_aware" and mode.startswith("p2p-"):
        return list(selected)
    if dim == 3 and transport == "non_cuda_aware" and mode.startswith("p2p-"):
        return ["nca-staging"] if selected != [None] else [None]
    return [None]


def p2p_schedulers_for_spec(
    dim: int, transport: str, mode: str, selected: Sequence[Optional[str]]
) -> List[Optional[str]]:
    if dim == 3 and transport == "cuda_aware" and mode.startswith("p2p-"):
        return list(selected)
    return [None]


def pencil_layouts_for_spec(dim: int, strategy: str, selected: Sequence[Optional[str]]) -> List[Optional[str]]:
    if dim == 3 and strategy == "pencil-pencil":
        return list(selected)
    return [None]


def pencil_pipelines_for_spec(
    dim: int,
    strategy: str,
    mode: Optional[str],
    selected: Sequence[Optional[str]],
    transport: Optional[str] = None,
) -> List[Optional[str]]:
    if dim == 3 and strategy == "pencil-pencil":
        if mode is not None and not mode.startswith("p2p-"):
            return [None]
        pipelines = list(selected)
        if transport == "non_cuda_aware":
            # reference-parity is the device-aware byte path.  Use the
            # host-staging-capable reference implementation for NCA
            # matrices and remove duplicates when both were requested.
            pipelines = ["reference" if item == "reference-parity" else item for item in pipelines]
            pipelines = list(dict.fromkeys(pipelines))
        return pipelines
    return [None]


def large_count_p2p_transports_for_spec(
    dim: int,
    transport: str,
    strategy: str,
    mode: str,
    pencil_pipeline: Optional[str],
    selected: Sequence[Optional[str]],
) -> List[Optional[str]]:
    if (
        dim == 3
        and transport == "cuda_aware"
        and strategy == "pencil-pencil"
        and mode.startswith("p2p-")
        and (pencil_pipeline is None or pencil_pipeline in ("reference", "reference-parity"))
    ):
        return list(selected)
    return [None]


def contiguous_forward_send_modes_for_spec(
    dim: int,
    transport: str,
    strategy: str,
    mode: str,
    use_contiguous_forward_byte_send: bool,
    selected: Sequence[Optional[str]],
) -> List[Optional[str]]:
    if (
        use_contiguous_forward_byte_send
        and dim == 3
        and transport == "cuda_aware"
        and strategy == "pencil-pencil"
        and mode.startswith("p2p-")
    ):
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_transposes(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "staged": False,
        "stage": False,
        "old": False,
        "native": True,
        "direct": True,
        "native-xw": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_TRANSPOSES]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-transposes value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_transposes_for_spec(
    dim: int,
    strategy: Optional[str],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab":
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_batched_peer_kernels(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "legacy": False,
        "old": False,
        "per-peer": False,
        "perpeer": False,
        "sync": False,
        "off": False,
        "0": False,
        "false": False,
        "batched": True,
        "batch": True,
        "optimized": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_BATCHED_PEER_KERNELS]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-batched-peer-kernels value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_batched_peer_kernels_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_kernel_layouts(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "buffer": False,
        "buffer-contiguous": False,
        "mpi-buffer": False,
        "mpi": False,
        "old": False,
        "legacy": False,
        "off": False,
        "0": False,
        "false": False,
        "tensor": True,
        "tensor-contiguous": True,
        "tensor-coalesced": True,
        "coalesced": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_KERNEL_LAYOUTS]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-kernel-layouts value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_kernel_layouts_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_vector4_kernels(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "off": False,
        "0": False,
        "false": False,
        "legacy": False,
        "buffer": False,
        "on": True,
        "1": True,
        "true": True,
        "vector": True,
        "vector4": True,
        "vec4": True,
        "x4": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_VECTOR4_KERNELS]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-vector4-kernels value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_vector4_kernels_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    slab_xw_tensor_coalesced: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        if slab_xw_tensor_coalesced:
            return [False]
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_tiled_kernels(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "off": False,
        "0": False,
        "false": False,
        "legacy": False,
        "scalar": False,
        "on": True,
        "1": True,
        "true": True,
        "tile": True,
        "tiled": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_TILED_KERNELS]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-tiled-kernels value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_tiled_kernels_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    slab_xw_tensor_coalesced: Optional[bool],
    slab_xw_vector4: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        if slab_xw_tensor_coalesced or slab_xw_vector4:
            return [False]
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_layout_stages(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "direct": False,
        "public": False,
        "off": False,
        "0": False,
        "false": False,
        "stage": True,
        "staged": True,
        "layout-stage": True,
        "xzwy": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_LAYOUT_STAGES]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-layout-stages value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_layout_stages_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        return list(selected)
    return [None]


def parse_fftm_4d_slab_xw_native_spectral_layouts(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "public": False,
        "yzwx": False,
        "off": False,
        "0": False,
        "false": False,
        "native": True,
        "xzwy": True,
        "native-spectral": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_SLAB_XW_NATIVE_SPECTRAL_LAYOUTS]
        raise ValueError(
            "unknown --fftm-4d-slab-xw-native-spectral-layouts value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_slab_xw_native_spectral_layouts_for_spec(
    dim: int,
    strategy: Optional[str],
    slab_native_xw: Optional[bool],
    slab_xw_layout_stage: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil":
        return list(selected)
    if dim == 4 and strategy == "slab-slab" and slab_native_xw is not False:
        if slab_xw_layout_stage:
            return [False]
        return list(selected)
    return [None]


def parse_fftm_4d_pencil_same_zw_peer_paired(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "legacy": False,
        "off": False,
        "0": False,
        "false": False,
        "peer-paired": True,
        "peerpaired": True,
        "paired": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_SAME_ZW_PEER_PAIRED]
        raise ValueError(
            "unknown --fftm-4d-pencil-same-zw-peer-paired value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_pencil_same_zw_native_layouts(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "base": False,
        "legacy": False,
        "off": False,
        "0": False,
        "false": False,
        "native": True,
        "layout-native": True,
        "native-layout": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_SAME_ZW_NATIVE_LAYOUTS]
        raise ValueError(
            "unknown --fftm-4d-pencil-same-zw-native-layouts value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_pencil_degenerate_xw_slab_paths(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "base": False,
        "off": False,
        "0": False,
        "false": False,
        "degen": True,
        "degenerate": True,
        "slab": True,
        "slab-path": True,
        "native-slab": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_DEGENERATE_XW_SLAB_PATHS]
        raise ValueError(
            "unknown --fftm-4d-pencil-degenerate-xw-slab-paths value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_pencil_degenerate_local_transposes(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "base": False,
        "off": False,
        "0": False,
        "false": False,
        "local": True,
        "local-transpose": True,
        "local-transposes": True,
        "degen-local": True,
        "degenerate-local": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_DEGENERATE_LOCAL_TRANSPOSES]
        raise ValueError(
            "unknown --fftm-4d-pencil-degenerate-local-transposes value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_pencil_degenerate_same_xw_native(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [False]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "base": False,
        "off": False,
        "0": False,
        "false": False,
        "native": True,
        "same-xw": True,
        "xw-native": True,
        "degen-same-xw": True,
        "degenerate-same-xw": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_DEGENERATE_SAME_XW_NATIVE]
        raise ValueError(
            "unknown --fftm-4d-pencil-degenerate-same-xw-native value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_pencil_degenerate_wz_sliced_z_fft(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [None]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "base": False,
        "off": False,
        "0": False,
        "false": False,
        "sliced": True,
        "sliced-z": True,
        "wz-sliced-z": True,
        "fused": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_PENCIL_DEGENERATE_WZ_SLICED_Z_FFT]
        raise ValueError(
            "unknown --fftm-4d-pencil-degenerate-wz-sliced-z-fft value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_native_xw_direct_layouts(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "baseline": False,
        "buffered": False,
        "off": False,
        "0": False,
        "false": False,
        "direct": True,
        "direct-layout": True,
        "native-direct": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_NATIVE_XW_DIRECT_LAYOUTS]
        raise ValueError(
            "unknown --fftm-4d-native-xw-direct-layouts value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def fftm_4d_native_xw_direct_layouts_for_transport(
    transport: str, selected: Sequence[Optional[bool]]
) -> List[Optional[bool]]:
    if transport == "non_cuda_aware":
        # The direct XW and WZ-ready paths pass device pointers to MPI. Keep
        # native XW/native spectral layout for NCA runs, but force the
        # host-staged transpose implementation.
        return [False]
    return list(selected)


def parse_fftm_4d_native_xw_protocols(value: str) -> List[Optional[str]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return ["chunked"]
    if value in ("all", "both"):
        return list(FFTM_4D_NATIVE_XW_PROTOCOLS)

    selected: List[Optional[str]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "single": "single",
        "one": "single",
        "chunked": "chunked",
        "chunks": "chunked",
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        allowed = ["configured", "production", "all", "both", *FFTM_4D_NATIVE_XW_PROTOCOLS]
        raise ValueError(
            "unknown --fftm-4d-native-xw-protocols value(s): "
            + ", ".join(unknown)
            + "; allowed: "
            + ", ".join(allowed)
        )
    return selected or [None]


def parse_fftm_4d_native_xw_chunk_windows(value: str) -> List[Optional[int]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [1]
    if value == "matrix":
        return list(FFTM_4D_NATIVE_XW_CHUNK_WINDOWS)

    selected: List[Optional[int]] = []
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in ("configured", "default"):
            selected.append(None)
        elif key in ("all", "full", "unbounded", "0"):
            selected.append(0)
        else:
            try:
                window = int(key)
            except ValueError:
                unknown.append(item.strip())
                continue
            if window <= 0:
                unknown.append(item.strip())
            else:
                selected.append(window)
    if unknown:
        raise ValueError(
            "unknown --fftm-4d-native-xw-chunk-windows value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, production, matrix, all, or positive integers"
        )
    return selected or [None]


def parse_fftm_4d_native_xw_compact_stagings(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "full": False,
        "off": False,
        "0": False,
        "false": False,
        "compact": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        raise ValueError(
            "unknown --fftm-4d-native-xw-compact-stagings value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, production, all, both, off, on"
        )
    return selected or [None]


def parse_fftm_4d_slab_native_work_area_aliases(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    aliases = {
        "configured": None,
        "default": None,
        "off": False,
        "0": False,
        "false": False,
        "alias": True,
        "on": True,
        "1": True,
        "true": True,
    }
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        raise ValueError(
            "unknown --fftm-4d-slab-native-work-area-aliases value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, production, all, both, off, on"
        )
    return selected or [None]


def parse_fftm_4d_slab_native_wz_communication_layouts(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value == "configured":
        return [None]
    if value == "production":
        return [True]
    if value in ("all", "both"):
        return [False, True]

    selected: List[Optional[bool]] = []
    aliases = {
        "configured": None, "default": None,
        "off": False, "0": False, "false": False,
        "on": True, "1": True, "true": True, "communication": True, "wz-communication": True,
    }
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        raise ValueError(
            "unknown --fftm-4d-slab-native-wz-communication-layouts value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, production, all, both, off, on"
        )
    return selected or [None]


def parse_fftm_4d_slab_native_wz_plan_concurrencies(value: str) -> List[Optional[int]]:
    value = (value or "configured").strip().lower()
    if value in ("configured", "default"):
        return [None]
    if value == "production":
        return [4]
    if value in ("matrix", "all"):
        return [1, 2, 4, 8]
    selected: List[Optional[int]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            concurrency = int(text)
        except ValueError as exc:
            raise ValueError(
                f"invalid --fftm-4d-slab-native-wz-plan-concurrencies value: {text}"
            ) from exc
        if concurrency <= 0:
            raise ValueError("4D slab WZ plan concurrency must be positive")
        selected.append(concurrency)
    return selected or [None]


def parse_fftm_4d_slab_native_wz_ready_pipelines(value: str) -> List[Optional[bool]]:
    value = (value or "configured").strip()
    if value in ("configured", "default"):
        return [None]
    if value == "production":
        return [True]
    if value in ("matrix", "all", "both"):
        return [False, True]
    aliases = {
        "off": False, "0": False, "false": False,
        "on": True, "1": True, "true": True, "ready": True, "pipeline": True,
    }
    selected: List[Optional[bool]] = []
    unknown: List[str] = []
    for item in value.split(","):
        key = item.strip().lower().replace("_", "-")
        if not key:
            continue
        if key in aliases:
            selected.append(aliases[key])
        else:
            unknown.append(item.strip())
    if unknown:
        raise ValueError(
            "unknown --fftm-4d-slab-native-wz-ready-pipelines value(s): "
            + ", ".join(unknown)
            + "; allowed: configured, production, matrix, off, on"
        )
    return selected or [None]


def fftm_4d_pencil_same_zw_peer_paired_for_spec(
    dim: int,
    strategy: Optional[str],
    mode: Optional[str],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and mode == "p2p-waitany":
        return list(selected)
    return [None]


def fftm_4d_pencil_same_zw_native_layouts_for_spec(
    dim: int,
    strategy: Optional[str],
    mode: Optional[str],
    native_spectral_layout: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and native_spectral_layout is True:
        return list(selected)
    return [None]


def fftm_4d_pencil_degenerate_xw_slab_paths_for_spec(
    dim: int,
    strategy: Optional[str],
    native_spectral_layout: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and native_spectral_layout is True:
        return list(selected)
    return [None]


def fftm_4d_pencil_degenerate_local_transposes_for_spec(
    dim: int,
    strategy: Optional[str],
    native_spectral_layout: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and native_spectral_layout is True:
        return list(selected)
    return [None]


def fftm_4d_pencil_degenerate_same_xw_native_for_spec(
    dim: int,
    strategy: Optional[str],
    mode: Optional[str],
    native_spectral_layout: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and mode == "p2p-waitany" and native_spectral_layout is True:
        return list(selected)
    return [None]


def fftm_4d_pencil_degenerate_wz_sliced_z_fft_for_spec(
    dim: int,
    strategy: Optional[str],
    native_spectral_layout: Optional[bool],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if dim == 4 and strategy == "pencil-pencil" and native_spectral_layout is True:
        return list(selected)
    return [None]


def fftm_4d_pencil_matrix_cases_for_spec(
    dim: int,
    strategy: Optional[str],
    mode: Optional[str],
    native_spectral_layout: Optional[bool],
    fftm_4d_pencil_same_zw_peer_paired: Sequence[Optional[bool]],
    fftm_4d_pencil_same_zw_native_layouts: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_xw_slab_paths: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_local_transposes: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_same_xw_native: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_wz_sliced_z_fft: Sequence[Optional[bool]],
    fftm_4d_native_xw_direct_layouts: Sequence[Optional[bool]],
    fftm_4d_native_xw_protocols: Sequence[Optional[str]],
    fftm_4d_native_xw_chunk_windows: Sequence[Optional[int]],
    fftm_4d_native_xw_compact_stagings: Sequence[Optional[bool]],
    fftm_4d_slab_native_work_area_aliases: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_communication_layouts: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_plan_concurrencies: Sequence[Optional[int]],
    fftm_4d_slab_native_wz_ready_pipelines: Sequence[Optional[bool]],
) -> List[
    Tuple[
        Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool],
        Optional[bool], Optional[str], Optional[int], Optional[bool], Optional[bool], Optional[bool],
        Optional[int], Optional[bool]
    ]
]:
    cases: List[
        Tuple[
            Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool],
            Optional[bool], Optional[str], Optional[int], Optional[bool], Optional[bool], Optional[bool],
            Optional[int], Optional[bool]
        ]
    ] = []
    same_zw_peer_values = fftm_4d_pencil_same_zw_peer_paired_for_spec(
        dim, strategy, mode, fftm_4d_pencil_same_zw_peer_paired
    )
    same_zw_native_values = fftm_4d_pencil_same_zw_native_layouts_for_spec(
        dim, strategy, mode, native_spectral_layout, fftm_4d_pencil_same_zw_native_layouts
    )
    degen_xw_values = fftm_4d_pencil_degenerate_xw_slab_paths_for_spec(
        dim, strategy, native_spectral_layout, fftm_4d_pencil_degenerate_xw_slab_paths
    )
    degen_local_values = fftm_4d_pencil_degenerate_local_transposes_for_spec(
        dim, strategy, native_spectral_layout, fftm_4d_pencil_degenerate_local_transposes
    )
    degen_same_xw_base_values = fftm_4d_pencil_degenerate_same_xw_native_for_spec(
        dim, strategy, mode, native_spectral_layout, fftm_4d_pencil_degenerate_same_xw_native
    )
    degen_wz_base_values = fftm_4d_pencil_degenerate_wz_sliced_z_fft_for_spec(
        dim, strategy, native_spectral_layout, fftm_4d_pencil_degenerate_wz_sliced_z_fft
    )
    for same_zw_peer in same_zw_peer_values:
        for same_zw_native in same_zw_native_values:
            for degen_xw in degen_xw_values:
                for degen_local in degen_local_values:
                    degen_same_xw_values = degen_same_xw_base_values if degen_local is True else [None]
                    degen_wz_values = degen_wz_base_values if degen_local is True else [None]
                    for degen_same_xw in degen_same_xw_values:
                        for degen_wz in degen_wz_values:
                            direct_values: Sequence[Optional[bool]] = [None]
                            if (
                                dim == 4
                                and mode == "p2p-waitany"
                                and native_spectral_layout is True
                                and (
                                    strategy == "slab-slab"
                                    or (
                                        strategy == "pencil-pencil"
                                        and degen_local is True
                                        and degen_same_xw is True
                                    )
                                )
                            ):
                                direct_values = fftm_4d_native_xw_direct_layouts
                            for direct_layout in direct_values:
                                protocol_values = fftm_4d_native_xw_protocols if direct_layout is True else [None]
                                for protocol in protocol_values:
                                    window_values = fftm_4d_native_xw_chunk_windows if protocol == "chunked" else [None]
                                    for chunk_window in window_values:
                                        compact_values = (
                                            fftm_4d_native_xw_compact_stagings
                                            if direct_layout is True
                                            and protocol == "chunked"
                                            and chunk_window is not None
                                            and chunk_window > 0
                                            else [None]
                                        )
                                        for compact_staging in compact_values:
                                            work_alias_values = (
                                                fftm_4d_slab_native_work_area_aliases
                                                if strategy == "slab-slab"
                                                and native_spectral_layout is True
                                                and direct_layout is True
                                                else [None]
                                            )
                                            for work_alias in work_alias_values:
                                                wz_communication_values = (
                                                    fftm_4d_slab_native_wz_communication_layouts
                                                    if strategy == "slab-slab"
                                                    and mode == "p2p-waitany"
                                                    and native_spectral_layout is True
                                                    and direct_layout is True
                                                    and protocol == "chunked"
                                                    and chunk_window is not None
                                                    and chunk_window > 0
                                                    else [None]
                                                )
                                                for wz_communication in wz_communication_values:
                                                    concurrency_values = (
                                                        fftm_4d_slab_native_wz_plan_concurrencies
                                                        if wz_communication is True else [None]
                                                    )
                                                    pipeline_values = (
                                                        fftm_4d_slab_native_wz_ready_pipelines
                                                        if wz_communication is True else [None]
                                                    )
                                                    for wz_concurrency in concurrency_values:
                                                        for wz_pipeline in pipeline_values:
                                                            cases.append(
                                                                (
                                                                    same_zw_peer, same_zw_native, degen_xw, degen_local,
                                                                    degen_same_xw, degen_wz, direct_layout, protocol,
                                                                    chunk_window, compact_staging, work_alias, wz_communication,
                                                                    wz_concurrency, wz_pipeline,
                                                                )
                                                            )
    return cases


def native_backward_second_peer_loop_modes_for_spec(
    dim: int,
    transport: str,
    strategy: str,
    mode: str,
    pencil_pipeline: Optional[str],
    fftm_3d_backend: Optional[str],
    selected: Sequence[Optional[bool]],
) -> List[Optional[bool]]:
    if (
        dim == 3
        and transport == "cuda_aware"
        and strategy == "pencil-pencil"
        and mode.startswith("p2p-")
        and pencil_pipeline in ("reference", "reference-parity")
        and fftm_3d_backend in (None, "native")
    ):
        return list(selected)
    return [None]


def fftm_3d_backends_for_spec(
    dim: int,
    transport: str,
    strategy: str,
    mode: str,
    case_name: str,
    grid: Optional[Tuple[int, ...]],
    selected: Sequence[Optional[str]],
) -> List[Optional[str]]:
    if dim != 3 or case_name != "benchmark":
        return [None]
    if selected == [None]:
        return [None]
    result: List[Optional[str]] = []
    for backend in selected:
        if backend in (None, "native"):
            result.append(backend)
        elif (
            transport == "cuda_aware"
            and strategy == "pencil-pencil"
            and mode.startswith("p2p-")
            and grid is not None
            and len(grid) == 2
            and grid[0] > 1
            and grid[1] > 1
        ):
            result.append(backend)
    return result or [None]


def p2p_scheduler_flags(
    scheduler: Optional[str], default_send_thread: bool, default_persistent_p2p: bool
) -> Tuple[bool, bool]:
    if scheduler is None:
        return default_send_thread, default_persistent_p2p
    if scheduler == "main":
        return False, False
    if scheduler == "send-thread":
        return True, False
    if scheduler == "persistent":
        return False, True
    if scheduler == "send-thread-persistent":
        return True, True
    raise ValueError(f"unknown p2p scheduler: {scheduler}")


def parse_summary_line(line: str) -> Optional[Dict[str, Any]]:
    payload = line.strip()
    if payload.startswith("INFO:"):
        payload = payload.split("INFO:", 1)[1].strip()
    elif payload.startswith("WARNING:"):
        payload = payload.split("WARNING:", 1)[1].strip()
    elif payload.startswith("ERROR"):
        return None

    if "tracked memory incl. external/test-owned" in payload:
        return None

    if not (payload.startswith("benchmark=") or payload.startswith("test=")):
        return None

    data: Dict[str, Any] = {}
    timing_separator = ": avg_wall_ms="
    if timing_separator in payload:
        left, right_tail = payload.rsplit(timing_separator, 1)
        right = "avg_wall_ms=" + right_tail
        for part in split_top_level_csv(left) + split_top_level_csv(right):
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            data[key.strip()] = parse_scalar(raw_value)
    elif ": " in payload:
        left, right = payload.split(": ", 1)
        for part in split_top_level_csv(left) + split_top_level_csv(right):
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            data[key.strip()] = parse_scalar(raw_value)
    else:
        parts = split_top_level_csv(payload)
        if len(parts) > 1:
            for part in parts:
                if "=" not in part:
                    continue
                key, raw_value = part.split("=", 1)
                data[key.strip()] = parse_scalar(raw_value)
        else:
            for match in SUMMARY_TOKEN_RE.finditer(payload):
                data[match.group(1)] = parse_scalar(match.group(2))
    return data or None


def parse_profile_entries(lines: Sequence[str], start_index: int) -> Tuple[List[Dict[str, Any]], int]:
    entries: List[Dict[str, Any]] = []
    index = start_index + 1
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        match = PROFILE_LINE_RE.match(stripped)
        if not match:
            break
        entries.append(
            {
                "name": match.group("name").strip(),
                "ms": float(match.group("ms")),
                "pct": float(match.group("pct")),
            }
        )
        index += 1
    return entries, index


def parse_memory_block(lines: Sequence[str], start_index: int) -> Tuple[Dict[str, Any], int]:
    header = lines[start_index].strip()
    entries: Dict[str, Dict[str, Any]] = {}
    index = start_index + 1
    while index < len(lines):
        stripped = lines[index].rstrip()
        if not stripped:
            index += 1
            continue
        match = MEMORY_LINE_RE.match(stripped)
        if not match:
            break
        raw = match.groupdict()
        entry: Dict[str, Any] = {}
        for key, value in raw.items():
            if key == "name":
                continue
            entry[key] = float(value) if key.endswith("_mib") else int(value)
        entries[raw["name"].strip()] = entry
        index += 1
    return {"header": header, "entries": entries}, index


def parse_run_output(stdout: str) -> Dict[str, Any]:
    lines = stdout.splitlines()
    result: Dict[str, Any] = {
        "summary": None,
        "profile_breakdown": [],
        "profile_summary": [],
        "memory_profile": None,
        "memory_categories": None,
        "memory_totals": None,
        "tracked_memory": None,
        "errors": [],
    }

    for line in lines:
        summary = parse_summary_line(line)
        if summary is not None:
            result["summary"] = summary
        if line.lstrip().startswith("ERROR"):
            result["errors"].append(line.strip())

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.endswith("Profile breakdown:"):
            entries, next_index = parse_profile_entries(lines, index)
            result["profile_breakdown"] = entries
            index = next_index
            continue
        if stripped.endswith("Profile summarize:"):
            entries, next_index = parse_profile_entries(lines, index)
            result["profile_summary"] = entries
            index = next_index
            continue
        if "tracked memory incl. external/test-owned" in stripped:
            block, next_index = parse_memory_block(lines, index)
            result["tracked_memory"] = block
            index = next_index
            continue
        if stripped.startswith("Memory profile totals"):
            block, next_index = parse_memory_block(lines, index)
            result["memory_totals"] = block
            index = next_index
            continue
        if stripped.startswith("Memory profile categories"):
            block, next_index = parse_memory_block(lines, index)
            result["memory_categories"] = block
            index = next_index
            continue
        if stripped.startswith("Memory profile"):
            block, next_index = parse_memory_block(lines, index)
            result["memory_profile"] = block
            index = next_index
            continue
        index += 1

    return result


def tracked_device_peak_max_bytes(parsed: Dict[str, Any]) -> Optional[int]:
    tracked = parsed.get("tracked_memory")
    if not tracked:
        return None
    entry = tracked["entries"].get("tracked_device_total")
    if entry is None:
        return None
    return int(entry["peak_max"])


def internal_device_peak_max_bytes(parsed: Dict[str, Any]) -> Optional[int]:
    categories = parsed.get("memory_categories")
    if not categories:
        return None
    entry = categories["entries"].get("device")
    if entry is None:
        return None
    return int(entry["peak_max"])


def ffts_external_peak_bytes(case_name: str, dim: int, n: int) -> int:
    if dim == 3:
        real_elems = n * n * n
        hat_elems = n * n * (n // 2 + 1)
        real_bytes = real_elems * 8
        hat_bytes = hat_elems * 16
        hat_scalar_bytes = hat_elems * 8
        if case_name in ("benchmark", "v0", "v2", "v3"):
            return real_bytes + hat_bytes
        if case_name == "v1":
            return 2 * real_bytes + 2 * hat_bytes + 2 * hat_scalar_bytes
        if case_name == "v4":
            return 11 * real_bytes + 5 * hat_bytes
    elif dim == 4:
        real_elems = n * n * n * n
        hat_elems = n * n * n * (n // 2 + 1)
        real_bytes = real_elems * 8
        hat_bytes = hat_elems * 16
        hat_scalar_bytes = hat_elems * 8
        if case_name in ("benchmark", "v0", "v2", "v3"):
            return real_bytes + hat_bytes
        if case_name == "v1":
            return 2 * real_bytes + 2 * hat_bytes + 2 * hat_scalar_bytes
        if case_name == "v4":
            return 13 * real_bytes + 6 * hat_bytes
    raise ValueError(f"Unsupported FFTS external memory estimate for case={case_name}, dim={dim}")


def infer_device_peak_max_bytes(parsed: Dict[str, Any], spec: "RunSpec", n: int) -> Optional[int]:
    tracked_bytes = tracked_device_peak_max_bytes(parsed)
    if tracked_bytes is not None:
        return tracked_bytes

    internal_device = internal_device_peak_max_bytes(parsed)
    if internal_device is None:
        return None

    if spec.suite == "ffts":
        return internal_device + ffts_external_peak_bytes(spec.case_name, spec.dim, n)

    return internal_device


def parse_nvidia_smi_inventory(text: str) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            gpu_index = int(parts[0])
            memory_total_mib = int(parts[2])
        except ValueError:
            continue
        gpus.append(
            {
                "index": gpu_index,
                "name": parts[1],
                "memory_total_mib": memory_total_mib,
                "memory_total_bytes": mib_to_bytes(memory_total_mib),
            }
        )
    return sorted(gpus, key=lambda item: item["index"])


def detect_gpus() -> List[Dict[str, Any]]:
    query = ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"]
    completed = subprocess.run(query, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to query GPUs with nvidia-smi: {completed.stderr.strip()}")
    return parse_nvidia_smi_inventory(completed.stdout)


def effective_gpu_inventory(
    detected: List[Dict[str, Any]], override_mib: Optional[Sequence[int]]
) -> List[Dict[str, Any]]:
    if not override_mib:
        return detected

    if detected and len(override_mib) != len(detected):
        raise ValueError(
            "--device-memory-mib must provide exactly one value per detected GPU "
            f"({len(detected)} detected, {len(override_mib)} provided)"
        )

    if not detected:
        return [
            {
                "index": idx,
                "name": f"override_gpu_{idx}",
                "memory_total_mib": int(mib),
                "memory_total_bytes": mib_to_bytes(int(mib)),
            }
            for idx, mib in enumerate(override_mib)
        ]

    effective: List[Dict[str, Any]] = []
    for gpu, override in zip(detected, override_mib):
        effective.append(
            {
                **gpu,
                "memory_total_mib": int(override),
                "memory_total_bytes": mib_to_bytes(int(override)),
            }
        )
    return effective


@dataclass(frozen=True)
class RunSpec:
    suite: str
    dim: int
    case_name: str
    binary_name: str
    binary_path: str
    num_gpus: int
    transport: str
    strategy: Optional[str]
    mode: Optional[str]
    uses_mpi: bool
    supports_directory: bool
    memory_family: str
    p2p_variant: Optional[str] = None
    p2p_scheduler: Optional[str] = None
    pencil_layout: Optional[str] = None
    pencil_pipeline: Optional[str] = None
    large_count_p2p_transport: Optional[str] = None
    contiguous_forward_send_mode: Optional[str] = None
    native_backward_second_peer_loop: Optional[bool] = None
    fftm_4d_slab_native_xw: Optional[bool] = None
    fftm_4d_slab_xw_batched_peer_kernels: Optional[bool] = None
    fftm_4d_slab_xw_tensor_coalesced_kernels: Optional[bool] = None
    fftm_4d_slab_xw_vector4_kernels: Optional[bool] = None
    fftm_4d_slab_xw_tiled_kernels: Optional[bool] = None
    fftm_4d_slab_xw_layout_stage: Optional[bool] = None
    fftm_4d_slab_xw_native_spectral_layout: Optional[bool] = None
    fftm_4d_pencil_same_zw_peer_paired: Optional[bool] = None
    fftm_4d_pencil_same_zw_native_layout: Optional[bool] = None
    fftm_4d_pencil_degenerate_xw_slab_path: Optional[bool] = None
    fftm_4d_pencil_degenerate_local_transposes: Optional[bool] = None
    fftm_4d_pencil_degenerate_same_xw_native: Optional[bool] = None
    fftm_4d_pencil_degenerate_wz_sliced_z_fft: Optional[bool] = None
    fftm_4d_native_xw_direct_layout: Optional[bool] = None
    fftm_4d_native_xw_protocol: Optional[str] = None
    fftm_4d_native_xw_chunk_window: Optional[int] = None
    fftm_4d_native_xw_compact_staging: Optional[bool] = None
    fftm_4d_slab_native_work_area_alias: Optional[bool] = None
    fftm_4d_slab_native_wz_communication_layout: Optional[bool] = None
    fftm_4d_slab_native_wz_plan_concurrency: Optional[int] = None
    fftm_4d_slab_native_wz_ready_pipeline: Optional[bool] = None
    fftm_3d_backend: Optional[str] = None
    native_opt0_y_executor_variant: Optional[str] = None
    native_opt0_y_cross_factory_mode: Optional[str] = None
    grid: Optional[Tuple[int, ...]] = None

    def size_arity(self) -> int:
        return 3 if self.dim == 3 else 4


def choose_pencil_grid_3d(num_gpus: int) -> Tuple[int, int]:
    p1 = 1
    d = 1
    while d * d <= num_gpus:
        if num_gpus % d == 0:
            p1 = d
        d += 1
    return p1, num_gpus // p1


def factor_grids_4d(num_gpus: int) -> List[Tuple[int, int, int]]:
    grids: List[Tuple[int, int, int]] = []
    for p1 in range(1, num_gpus + 1):
        if num_gpus % p1 != 0:
            continue
        rem1 = num_gpus // p1
        for p2 in range(1, rem1 + 1):
            if rem1 % p2 != 0:
                continue
            grids.append((p1, p2, rem1 // p2))
    return grids


def choose_pencil_grid_4d(num_gpus: int) -> Tuple[int, int, int]:
    best = (1, 1, num_gpus)
    best_span = num_gpus - 1
    for grid in factor_grids_4d(num_gpus):
        max_dim = max(grid)
        min_dim = min(grid)
        span = max_dim - min_dim
        if span < best_span:
            best_span = span
            best = grid
    return best


def _dedupe_grids(grids: Sequence[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    seen = set()
    result: List[Tuple[int, ...]] = []
    for grid in grids:
        if grid in seen:
            continue
        seen.add(grid)
        result.append(grid)
    return result


def grid_orientations_4d_for_spec(num_gpus: int, orientation_mode: str) -> List[Optional[Tuple[int, ...]]]:
    if orientation_mode in ("configured", "autotune"):
        return [None]

    default_grid = choose_pencil_grid_4d(num_gpus)
    x_heavy = (num_gpus, 1, 1)
    y_heavy = (1, num_gpus, 1)
    z_heavy = (1, 1, num_gpus)

    if orientation_mode == "production":
        return [default_grid]
    if orientation_mode == "default":
        return [default_grid]
    if orientation_mode == "x-heavy":
        return [x_heavy]
    if orientation_mode == "y-heavy":
        return [y_heavy]
    if orientation_mode == "z-heavy":
        return [z_heavy]
    if orientation_mode in ("axis", "axes"):
        return _dedupe_grids([default_grid, y_heavy, x_heavy, z_heavy])

    all_grids = factor_grids_4d(num_gpus)
    ordered = [default_grid, y_heavy, x_heavy, z_heavy]
    ordered.extend(
        sorted(
            all_grids,
            key=lambda g: (
                max(g) - min(g),
                g[0] == 1,
                g[1] == 1,
                g[2] == 1,
                g,
            ),
        )
    )
    return _dedupe_grids(ordered)


def grid_orientations_for_spec(
    dim: int,
    strategy: Optional[str],
    num_gpus: int,
    orientation_mode: str = "both",
    pencil_layout: Optional[str] = None,
    orientation_mode_4d: str = "configured",
) -> List[Optional[Tuple[int, ...]]]:
    if strategy != "pencil-pencil" or num_gpus < 2:
        return [None]

    if dim == 4:
        return grid_orientations_4d_for_spec(num_gpus, orientation_mode_4d)

    if dim != 3:
        return [None]

    if orientation_mode.startswith("explicit:"):
        grids = []
        for item in orientation_mode.removeprefix("explicit:").split(","):
            p1_text, p2_text = item.split("x", 1)
            grid = (int(p1_text), int(p2_text))
            if grid[0] * grid[1] == num_gpus:
                grids.append(grid)
        if not grids:
            raise ValueError(
                "No explicit 3D pencil grid in "
                f"{orientation_mode.removeprefix('explicit:')!r} has product {num_gpus}"
            )
        return _dedupe_grids(grids)

    if orientation_mode in ("configured", "autotune"):
        return [None]

    default_grid = choose_pencil_grid_3d(num_gpus)
    reversed_grid = (default_grid[1], default_grid[0])

    if orientation_mode == "production":
        if num_gpus == 7:
            if pencil_layout == "opt1":
                return [default_grid]
            return [reversed_grid]
        if num_gpus == 8:
            if pencil_layout == "opt1":
                return [default_grid]
            return [reversed_grid]
        return [default_grid]

    if orientation_mode == "default":
        return [default_grid]
    if orientation_mode == "reversed":
        return [reversed_grid]

    grids: List[Tuple[int, ...]] = [default_grid]
    if reversed_grid != default_grid:
        grids.append(reversed_grid)
    return grids


def grid_slug(grid: Optional[Tuple[int, ...]]) -> str:
    if not grid:
        return "gridauto"
    return "grid" + "x".join(str(v) for v in grid)


def parse_pencil_grid_orientations(value: str) -> str:
    value = re.sub(r"\s+", "", (value or "both").strip().lower())
    if "x" in value:
        items = value.split(",")
        if not items or any(not re.fullmatch(r"[1-9][0-9]*x[1-9][0-9]*", item) for item in items):
            raise ValueError(
                "Explicit 3D pencil grids must be a comma-separated list such as 2x8,4x4"
            )
        return "explicit:" + ",".join(items)

    aliases = {
        "configured": "configured",
        "config": "configured",
        "autotune": "configured",
        "tuned": "configured",
        "auto": "both",
        "all": "both",
        "both": "both",
        "production": "production",
        "prod": "production",
        "known-best": "production",
        "best-known": "production",
        "default": "default",
        "primary": "default",
        "reversed": "reversed",
        "reverse": "reversed",
    }
    if value not in aliases:
        raise ValueError(
            "--pencil-pencil-grid-orientations must be one of: configured, both, production, "
            "default, reversed, or an explicit comma-separated grid list such as 2x8,4x4"
        )
    return aliases[value]


def parse_4d_pencil_grid_orientations(value: str) -> str:
    value = (value or "configured").strip().lower().replace("_", "-")
    aliases = {
        "configured": "configured",
        "config": "configured",
        "autotune": "configured",
        "tuned": "configured",
        "auto": "configured",
        "production": "production",
        "prod": "production",
        "known-best": "production",
        "best-known": "production",
        "default": "default",
        "balanced": "default",
        "primary": "default",
        "x-heavy": "x-heavy",
        "p1-heavy": "x-heavy",
        "i-heavy": "x-heavy",
        "first-heavy": "x-heavy",
        "y-heavy": "y-heavy",
        "p2-heavy": "y-heavy",
        "j-heavy": "y-heavy",
        "second-heavy": "y-heavy",
        "z-heavy": "z-heavy",
        "p3-heavy": "z-heavy",
        "k-heavy": "z-heavy",
        "third-heavy": "z-heavy",
        "axis": "axis",
        "axes": "axis",
        "axis-heavy": "axis",
        "all": "all",
        "both": "all",
        "matrix": "all",
        "all-4d": "all",
    }
    if value not in aliases:
        raise ValueError(
            "--fftm-4d-pencil-grid-orientations must be one of: "
            "configured, production, default, x-heavy, y-heavy, z-heavy, axis, all"
        )
    return aliases[value]


@dataclass(frozen=True)
class ProbeFamily:
    key: str
    dim: int
    num_gpus: int
    representative: RunSpec


def discover_binaries(tests_root: Path) -> Dict[str, Path]:
    binaries: Dict[str, Path] = {}
    for path in sorted(tests_root.glob("*.bin")):
        if path.is_file():
            binaries[path.name] = path.resolve()
    return binaries


def add_ffts_specs(specs: List[RunSpec], binaries: Dict[str, Path]) -> None:
    if "test_benchmark_ffts_3D.bin" in binaries:
        specs.append(
            RunSpec(
                suite="ffts",
                dim=3,
                case_name="benchmark",
                binary_name="test_benchmark_ffts_3D.bin",
                binary_path=str(binaries["test_benchmark_ffts_3D.bin"]),
                num_gpus=1,
                transport="single_gpu",
                strategy=None,
                mode=None,
                uses_mpi=False,
                supports_directory=True,
                memory_family="ffts:3d:small:cufft-3d",
            )
        )

    for testcase in range(5):
        name = f"test_ffts_v{testcase}_3D.bin"
        if name not in binaries:
            continue
        family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
        specs.append(
            RunSpec(
                suite="ffts",
                dim=3,
                case_name=f"v{testcase}",
                binary_name=name,
                binary_path=str(binaries[name]),
                num_gpus=1,
                transport="single_gpu",
                strategy=None,
                mode=None,
                uses_mpi=False,
                supports_directory=True,
                memory_family=f"ffts:3d:{family_kind}:cufft-3d",
            )
        )

    if "test_benchmark_ffts_4D.bin" in binaries:
        for strategy in FFTS_STRATEGIES_4D:
            specs.append(
                RunSpec(
                    suite="ffts",
                    dim=4,
                    case_name="benchmark",
                    binary_name="test_benchmark_ffts_4D.bin",
                    binary_path=str(binaries["test_benchmark_ffts_4D.bin"]),
                    num_gpus=1,
                    transport="single_gpu",
                    strategy=strategy,
                    mode=None,
                    uses_mpi=False,
                    supports_directory=True,
                    memory_family=f"ffts:4d:small:{strategy}",
                )
            )

    for testcase in range(5):
        name = f"test_ffts_v{testcase}_4D.bin"
        if name not in binaries:
            continue
        family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
        for strategy in FFTS_STRATEGIES_4D:
            specs.append(
                RunSpec(
                    suite="ffts",
                    dim=4,
                    case_name=f"v{testcase}",
                    binary_name=name,
                    binary_path=str(binaries[name]),
                    num_gpus=1,
                    transport="single_gpu",
                    strategy=strategy,
                    mode=None,
                    uses_mpi=False,
                    supports_directory=True,
                    memory_family=f"ffts:4d:{family_kind}:{strategy}",
                )
            )


def add_fftm_specs(
    specs: List[RunSpec],
    binaries: Dict[str, Path],
    max_gpus: int,
    include_nca: bool,
    p2p_variants: Sequence[Optional[str]],
    p2p_schedulers: Sequence[Optional[str]],
    pencil_layouts: Sequence[Optional[str]],
    pencil_pipelines: Sequence[Optional[str]],
    large_count_p2p_transports: Sequence[Optional[str]],
    contiguous_forward_send_modes: Sequence[Optional[str]],
    fftm_4d_slab_xw_transposes: Sequence[Optional[bool]],
    fftm_4d_slab_xw_batched_peer_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_kernel_layouts: Sequence[Optional[bool]],
    fftm_4d_slab_xw_vector4_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_tiled_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_layout_stages: Sequence[Optional[bool]],
    fftm_4d_slab_xw_native_spectral_layouts: Sequence[Optional[bool]],
    fftm_4d_pencil_same_zw_peer_paired: Sequence[Optional[bool]],
    fftm_4d_pencil_same_zw_native_layouts: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_xw_slab_paths: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_local_transposes: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_same_xw_native: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_wz_sliced_z_fft: Sequence[Optional[bool]],
    fftm_4d_native_xw_direct_layouts: Sequence[Optional[bool]],
    fftm_4d_native_xw_protocols: Sequence[Optional[str]],
    fftm_4d_native_xw_chunk_windows: Sequence[Optional[int]],
    fftm_4d_native_xw_compact_stagings: Sequence[Optional[bool]],
    fftm_4d_slab_native_work_area_aliases: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_communication_layouts: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_plan_concurrencies: Sequence[Optional[int]],
    fftm_4d_slab_native_wz_ready_pipelines: Sequence[Optional[bool]],
    native_backward_second_peer_loop_modes: Sequence[Optional[bool]],
    fftm_3d_backends: Sequence[Optional[str]],
    use_contiguous_forward_byte_send: bool,
    pencil_grid_orientations: str = "both",
    pencil_grid_orientations_4d: str = "configured",
) -> None:
    transports: List[Tuple[str, str]] = [("cuda_aware", ".bin")]
    if include_nca:
        transports.append(("non_cuda_aware", "_nca.bin"))

    for num_gpus in range(2, max_gpus + 1):
        for transport_name, suffix in transports:
            benchmark_3d = f"test_benchmark_fftm_3D{suffix}"
            benchmark_4d = f"test_benchmark_fftm_4D{suffix}"

            if benchmark_3d in binaries:
                for strategy in FFTM_STRATEGIES_3D:
                    for mode in FFTM_MODES:
                        for variant in p2p_variants_for_spec(3, transport_name, strategy, mode, p2p_variants):
                            for scheduler in p2p_schedulers_for_spec(3, transport_name, mode, p2p_schedulers):
                                for pencil_layout in pencil_layouts_for_spec(3, strategy, pencil_layouts):
                                    for pencil_pipeline in pencil_pipelines_for_spec(
                                        3, strategy, mode, pencil_pipelines, transport_name
                                    ):
                                        for large_count_transport in large_count_p2p_transports_for_spec(
                                            3, transport_name, strategy, mode, pencil_pipeline, large_count_p2p_transports
                                        ):
                                            for contiguous_forward_send_mode in contiguous_forward_send_modes_for_spec(
                                                3,
                                                transport_name,
                                                strategy,
                                                mode,
                                                use_contiguous_forward_byte_send,
                                                contiguous_forward_send_modes,
                                            ):
                                                for grid in grid_orientations_for_spec(
                                                    3,
                                                    strategy,
                                                    num_gpus,
                                                    pencil_grid_orientations,
                                                    pencil_layout,
                                                    pencil_grid_orientations_4d,
                                                ):
                                                    for fftm_3d_backend in fftm_3d_backends_for_spec(
                                                        3,
                                                        transport_name,
                                                        strategy,
                                                        mode,
                                                        "benchmark",
                                                        grid,
                                                        fftm_3d_backends,
                                                    ):
                                                        for native_backward_second_peer_loop in native_backward_second_peer_loop_modes_for_spec(
                                                            3,
                                                            transport_name,
                                                            strategy,
                                                            mode,
                                                            pencil_pipeline,
                                                            fftm_3d_backend,
                                                            native_backward_second_peer_loop_modes,
                                                        ):
                                                            specs.append(
                                                                RunSpec(
                                                                    suite="fftm",
                                                                    dim=3,
                                                                    case_name="benchmark",
                                                                    binary_name=benchmark_3d,
                                                                    binary_path=str(binaries[benchmark_3d]),
                                                                    num_gpus=num_gpus,
                                                                    transport=transport_name,
                                                                    strategy=strategy,
                                                                    mode=mode,
                                                                    uses_mpi=True,
                                                                    supports_directory=True,
                                                                    memory_family=f"fftm:3d:small:{num_gpus}:{strategy}:{fftm_3d_backend or 'configured'}",
                                                                    p2p_variant=variant,
                                                                    p2p_scheduler=scheduler,
                                                                    pencil_layout=pencil_layout,
                                                                    pencil_pipeline=pencil_pipeline,
                                                                    large_count_p2p_transport=large_count_transport,
                                                                    contiguous_forward_send_mode=contiguous_forward_send_mode,
                                                                    native_backward_second_peer_loop=native_backward_second_peer_loop,
                                                                    fftm_3d_backend=fftm_3d_backend,
                                                                    grid=grid,
                                                                )
                                                            )

            if benchmark_4d in binaries:
                for strategy in FFTM_STRATEGIES_4D:
                    for mode in FFTM_MODES:
                        for slab_native_xw in fftm_4d_slab_xw_transposes_for_spec(
                            4, strategy, fftm_4d_slab_xw_transposes
                        ):
                            for slab_xw_tensor in fftm_4d_slab_xw_kernel_layouts_for_spec(
                                4, strategy, slab_native_xw, fftm_4d_slab_xw_kernel_layouts
                            ):
                                for slab_xw_vector4 in fftm_4d_slab_xw_vector4_kernels_for_spec(
                                    4, strategy, slab_native_xw, slab_xw_tensor, fftm_4d_slab_xw_vector4_kernels
                                ):
                                    for slab_xw_tiled in fftm_4d_slab_xw_tiled_kernels_for_spec(
                                        4, strategy, slab_native_xw, slab_xw_tensor, slab_xw_vector4,
                                        fftm_4d_slab_xw_tiled_kernels
                                    ):
                                        for slab_xw_layout_stage in fftm_4d_slab_xw_layout_stages_for_spec(
                                            4, strategy, slab_native_xw, fftm_4d_slab_xw_layout_stages
                                        ):
                                            for slab_xw_native_spectral in fftm_4d_slab_xw_native_spectral_layouts_for_spec(
                                                4, strategy, slab_native_xw, slab_xw_layout_stage,
                                                fftm_4d_slab_xw_native_spectral_layouts
                                            ):
                                                for slab_xw_batched in fftm_4d_slab_xw_batched_peer_kernels_for_spec(
                                                    4, strategy, slab_native_xw, fftm_4d_slab_xw_batched_peer_kernels
                                                ):
                                                    for (
                                                        pencil_same_zw_peer,
                                                        pencil_same_zw_native,
                                                        pencil_degenerate_xw_slab,
                                                        pencil_degenerate_local,
                                                        pencil_degenerate_same_xw,
                                                        pencil_degenerate_wz_sliced_z,
                                                        native_xw_direct_layout,
                                                        native_xw_protocol,
                                                        native_xw_chunk_window,
                                                        native_xw_compact_staging,
                                                        slab_native_work_area_alias,
                                                        slab_native_wz_communication_layout,
                                                        slab_native_wz_plan_concurrency,
                                                        slab_native_wz_ready_pipeline,
                                                    ) in fftm_4d_pencil_matrix_cases_for_spec(
                                                        4,
                                                        strategy,
                                                        mode,
                                                        slab_xw_native_spectral,
                                                        fftm_4d_pencil_same_zw_peer_paired,
                                                        fftm_4d_pencil_same_zw_native_layouts,
                                                        fftm_4d_pencil_degenerate_xw_slab_paths,
                                                        fftm_4d_pencil_degenerate_local_transposes,
                                                        fftm_4d_pencil_degenerate_same_xw_native,
                                                        fftm_4d_pencil_degenerate_wz_sliced_z_fft,
                                                        fftm_4d_native_xw_direct_layouts_for_transport(
                                                            transport_name, fftm_4d_native_xw_direct_layouts
                                                        ),
                                                        fftm_4d_native_xw_protocols,
                                                        fftm_4d_native_xw_chunk_windows,
                                                        fftm_4d_native_xw_compact_stagings,
                                                        fftm_4d_slab_native_work_area_aliases,
                                                        fftm_4d_slab_native_wz_communication_layouts,
                                                        fftm_4d_slab_native_wz_plan_concurrencies,
                                                        fftm_4d_slab_native_wz_ready_pipelines,
                                                    ):
                                                        for grid in grid_orientations_for_spec(
                                                            4,
                                                            strategy,
                                                            num_gpus,
                                                            pencil_grid_orientations,
                                                            None,
                                                            pencil_grid_orientations_4d,
                                                        ):
                                                            specs.append(
                                                                RunSpec(
                                                                    suite="fftm",
                                                                    dim=4,
                                                                    case_name="benchmark",
                                                                    binary_name=benchmark_4d,
                                                                    binary_path=str(binaries[benchmark_4d]),
                                                                    num_gpus=num_gpus,
                                                                    transport=transport_name,
                                                                    strategy=strategy,
                                                                    mode=mode,
                                                                    uses_mpi=True,
                                                                    supports_directory=True,
                                                                    memory_family=(
                                                                        f"fftm:4d:small:{num_gpus}:{strategy}:"
                                                                        f"{'native-xw' if slab_native_xw else 'staged-xw' if slab_native_xw is False else 'configured-xw'}:"
                                                                        f"{'tensor' if slab_xw_tensor else 'buffer' if slab_xw_tensor is False else 'kernel-configured'}:"
                                                                        f"{'vector4' if slab_xw_vector4 else 'scalar' if slab_xw_vector4 is False else 'vector-configured'}:"
                                                                        f"{'tiled' if slab_xw_tiled else 'untiled' if slab_xw_tiled is False else 'tile-configured'}:"
                                                                        f"{'layout-stage' if slab_xw_layout_stage else 'direct-layout' if slab_xw_layout_stage is False else 'layout-configured'}:"
                                                                        f"{'native-spectral' if slab_xw_native_spectral else 'public-spectral' if slab_xw_native_spectral is False else 'spectral-configured'}:"
                                                                        f"{'batched' if slab_xw_batched else 'legacy' if slab_xw_batched is False else 'batch-configured'}:"
                                                                        f"{'samezw-peer' if pencil_same_zw_peer else 'samezw-base' if pencil_same_zw_peer is False else 'samezw-configured'}:"
                                                                        f"{'samezw-native-layout' if pencil_same_zw_native else 'samezw-base-layout' if pencil_same_zw_native is False else 'samezw-layout-configured'}:"
                                                                        f"{'degen-xw-slab' if pencil_degenerate_xw_slab else 'degen-xw-off' if pencil_degenerate_xw_slab is False else 'degen-xw-configured'}:"
                                                                        f"{'degen-local' if pencil_degenerate_local else 'degen-local-off' if pencil_degenerate_local is False else 'degen-local-configured'}:"
                                                                        f"{'degen-samexw-native' if pencil_degenerate_same_xw else 'degen-samexw-base' if pencil_degenerate_same_xw is False else 'degen-samexw-configured'}:"
                                                                        f"{'degen-wz-sliced-z' if pencil_degenerate_wz_sliced_z else 'degen-wz-base' if pencil_degenerate_wz_sliced_z is False else 'degen-wz-configured'}:"
                                                                        f"{'xw-direct' if native_xw_direct_layout else 'xw-buffered' if native_xw_direct_layout is False else 'xw-direct-configured'}:"
                                                                        f"xw-proto-{native_xw_protocol or 'configured'}:"
                                                                        f"xw-window-{native_xw_chunk_window if native_xw_chunk_window is not None else 'configured'}:"
                                                                        f"xw-compact-{'on' if native_xw_compact_staging else 'off' if native_xw_compact_staging is False else 'configured'}:"
                                                                        f"work-alias-{'on' if slab_native_work_area_alias else 'off' if slab_native_work_area_alias is False else 'configured'}:"
                                                                        f"wz-communication-{'on' if slab_native_wz_communication_layout else 'off' if slab_native_wz_communication_layout is False else 'configured'}:"
                                                                        f"wz-concurrency-{slab_native_wz_plan_concurrency if slab_native_wz_plan_concurrency is not None else 'configured'}:"
                                                                        f"wz-pipeline-{'on' if slab_native_wz_ready_pipeline else 'off' if slab_native_wz_ready_pipeline is False else 'configured'}:"
                                                                        f"{grid_slug(grid)}"
                                                                    ),
                                                                    fftm_4d_slab_native_xw=slab_native_xw,
                                                                    fftm_4d_slab_xw_batched_peer_kernels=slab_xw_batched,
                                                                    fftm_4d_slab_xw_tensor_coalesced_kernels=slab_xw_tensor,
                                                                    fftm_4d_slab_xw_vector4_kernels=slab_xw_vector4,
                                                                    fftm_4d_slab_xw_tiled_kernels=slab_xw_tiled,
                                                                    fftm_4d_slab_xw_layout_stage=slab_xw_layout_stage,
                                                                    fftm_4d_slab_xw_native_spectral_layout=slab_xw_native_spectral,
                                                                    fftm_4d_pencil_same_zw_peer_paired=pencil_same_zw_peer,
                                                                    fftm_4d_pencil_same_zw_native_layout=pencil_same_zw_native,
                                                                    fftm_4d_pencil_degenerate_xw_slab_path=pencil_degenerate_xw_slab,
                                                                    fftm_4d_pencil_degenerate_local_transposes=pencil_degenerate_local,
                                                                    fftm_4d_pencil_degenerate_same_xw_native=pencil_degenerate_same_xw,
                                                                    fftm_4d_pencil_degenerate_wz_sliced_z_fft=pencil_degenerate_wz_sliced_z,
                                                                    fftm_4d_native_xw_direct_layout=native_xw_direct_layout,
                                                                    fftm_4d_native_xw_protocol=native_xw_protocol,
                                                                    fftm_4d_native_xw_chunk_window=native_xw_chunk_window,
                                                                    fftm_4d_native_xw_compact_staging=native_xw_compact_staging,
                                                                    fftm_4d_slab_native_work_area_alias=slab_native_work_area_alias,
                                                                    fftm_4d_slab_native_wz_communication_layout=slab_native_wz_communication_layout,
                                                                    fftm_4d_slab_native_wz_plan_concurrency=slab_native_wz_plan_concurrency,
                                                                    fftm_4d_slab_native_wz_ready_pipeline=slab_native_wz_ready_pipeline,
                                                                    grid=grid,
                                                                )
                                                            )

            for testcase in range(5):
                name_3d = f"test_fftm_v{testcase}_3D{suffix}"
                if name_3d in binaries:
                    family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
                    for strategy in FFTM_STRATEGIES_3D:
                        for mode in FFTM_MODES:
                            for variant in p2p_variants_for_spec(3, transport_name, strategy, mode, p2p_variants):
                                for scheduler in p2p_schedulers_for_spec(3, transport_name, mode, p2p_schedulers):
                                    for pencil_layout in pencil_layouts_for_spec(3, strategy, pencil_layouts):
                                        for pencil_pipeline in pencil_pipelines_for_spec(
                                            3, strategy, mode, pencil_pipelines, transport_name
                                        ):
                                            for large_count_transport in large_count_p2p_transports_for_spec(
                                                3,
                                                transport_name,
                                                strategy,
                                                mode,
                                                pencil_pipeline,
                                                large_count_p2p_transports,
                                            ):
                                                for grid in grid_orientations_for_spec(
                                                    3,
                                                    strategy,
                                                    num_gpus,
                                                    pencil_grid_orientations,
                                                    pencil_layout,
                                                ):
                                                    specs.append(
                                                        RunSpec(
                                                            suite="fftm",
                                                            dim=3,
                                                            case_name=f"v{testcase}",
                                                            binary_name=name_3d,
                                                            binary_path=str(binaries[name_3d]),
                                                            num_gpus=num_gpus,
                                                            transport=transport_name,
                                                            strategy=strategy,
                                                            mode=mode,
                                                            uses_mpi=True,
                                                            supports_directory=False,
                                                            memory_family=f"fftm:3d:{family_kind}:{num_gpus}:{strategy}",
                                                            p2p_variant=variant,
                                                            p2p_scheduler=scheduler,
                                                            pencil_layout=pencil_layout,
                                                            pencil_pipeline=pencil_pipeline,
                                                            large_count_p2p_transport=large_count_transport,
                                                            grid=grid,
                                                        )
                                                    )

                name_4d = f"test_fftm_v{testcase}_4D{suffix}"
                if name_4d in binaries:
                    family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
                    for strategy in FFTM_STRATEGIES_4D:
                        for mode in FFTM_MODES:
                            for slab_native_xw in fftm_4d_slab_xw_transposes_for_spec(
                                4, strategy, fftm_4d_slab_xw_transposes
                            ):
                                for slab_xw_tensor in fftm_4d_slab_xw_kernel_layouts_for_spec(
                                    4, strategy, slab_native_xw, fftm_4d_slab_xw_kernel_layouts
                                ):
                                    for slab_xw_vector4 in fftm_4d_slab_xw_vector4_kernels_for_spec(
                                        4, strategy, slab_native_xw, slab_xw_tensor, fftm_4d_slab_xw_vector4_kernels
                                    ):
                                        for slab_xw_batched in fftm_4d_slab_xw_batched_peer_kernels_for_spec(
                                            4, strategy, slab_native_xw, fftm_4d_slab_xw_batched_peer_kernels
                                        ):
                                            specs.append(
                                                RunSpec(
                                                    suite="fftm",
                                                    dim=4,
                                                    case_name=f"v{testcase}",
                                                    binary_name=name_4d,
                                                    binary_path=str(binaries[name_4d]),
                                                    num_gpus=num_gpus,
                                                    transport=transport_name,
                                                    strategy=strategy,
                                                    mode=mode,
                                                    uses_mpi=True,
                                                    supports_directory=False,
                                                    memory_family=(
                                                        f"fftm:4d:{family_kind}:{num_gpus}:{strategy}:"
                                                        f"{'native-xw' if slab_native_xw else 'staged-xw' if slab_native_xw is False else 'configured-xw'}:"
                                                        f"{'tensor' if slab_xw_tensor else 'buffer' if slab_xw_tensor is False else 'kernel-configured'}:"
                                                        f"{'vector4' if slab_xw_vector4 else 'scalar' if slab_xw_vector4 is False else 'vector-configured'}:"
                                                        f"{'batched' if slab_xw_batched else 'legacy' if slab_xw_batched is False else 'batch-configured'}"
                                                    ),
                                                    fftm_4d_slab_native_xw=slab_native_xw,
                                                    fftm_4d_slab_xw_batched_peer_kernels=slab_xw_batched,
                                                    fftm_4d_slab_xw_tensor_coalesced_kernels=slab_xw_tensor,
                                                    fftm_4d_slab_xw_vector4_kernels=slab_xw_vector4,
                                                )
                                            )


def build_measurement_specs(
    binaries: Dict[str, Path],
    max_gpus: int,
    include_nca: bool,
    p2p_variants: Sequence[Optional[str]],
    p2p_schedulers: Sequence[Optional[str]],
    pencil_layouts: Sequence[Optional[str]],
    pencil_pipelines: Sequence[Optional[str]],
    large_count_p2p_transports: Sequence[Optional[str]],
    contiguous_forward_send_modes: Sequence[Optional[str]],
    fftm_4d_slab_xw_transposes: Sequence[Optional[bool]],
    fftm_4d_slab_xw_batched_peer_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_kernel_layouts: Sequence[Optional[bool]],
    fftm_4d_slab_xw_vector4_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_tiled_kernels: Sequence[Optional[bool]],
    fftm_4d_slab_xw_layout_stages: Sequence[Optional[bool]],
    fftm_4d_slab_xw_native_spectral_layouts: Sequence[Optional[bool]],
    fftm_4d_pencil_same_zw_peer_paired: Sequence[Optional[bool]],
    fftm_4d_pencil_same_zw_native_layouts: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_xw_slab_paths: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_local_transposes: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_same_xw_native: Sequence[Optional[bool]],
    fftm_4d_pencil_degenerate_wz_sliced_z_fft: Sequence[Optional[bool]],
    fftm_4d_native_xw_direct_layouts: Sequence[Optional[bool]],
    fftm_4d_native_xw_protocols: Sequence[Optional[str]],
    fftm_4d_native_xw_chunk_windows: Sequence[Optional[int]],
    fftm_4d_native_xw_compact_stagings: Sequence[Optional[bool]],
    fftm_4d_slab_native_work_area_aliases: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_communication_layouts: Sequence[Optional[bool]],
    fftm_4d_slab_native_wz_plan_concurrencies: Sequence[Optional[int]],
    fftm_4d_slab_native_wz_ready_pipelines: Sequence[Optional[bool]],
    native_backward_second_peer_loop_modes: Sequence[Optional[bool]],
    fftm_3d_backends: Sequence[Optional[str]],
    use_contiguous_forward_byte_send: bool,
    pencil_grid_orientations: str = "both",
    pencil_grid_orientations_4d: str = "configured",
) -> List[RunSpec]:
    specs: List[RunSpec] = []
    add_ffts_specs(specs, binaries)
    add_fftm_specs(
        specs,
        binaries,
        max_gpus=max_gpus,
        include_nca=include_nca,
        p2p_variants=p2p_variants,
        p2p_schedulers=p2p_schedulers,
        pencil_layouts=pencil_layouts,
        pencil_pipelines=pencil_pipelines,
        large_count_p2p_transports=large_count_p2p_transports,
        contiguous_forward_send_modes=contiguous_forward_send_modes,
        fftm_4d_slab_xw_transposes=fftm_4d_slab_xw_transposes,
        fftm_4d_slab_xw_batched_peer_kernels=fftm_4d_slab_xw_batched_peer_kernels,
        fftm_4d_slab_xw_kernel_layouts=fftm_4d_slab_xw_kernel_layouts,
        fftm_4d_slab_xw_vector4_kernels=fftm_4d_slab_xw_vector4_kernels,
        fftm_4d_slab_xw_tiled_kernels=fftm_4d_slab_xw_tiled_kernels,
        fftm_4d_slab_xw_layout_stages=fftm_4d_slab_xw_layout_stages,
        fftm_4d_slab_xw_native_spectral_layouts=fftm_4d_slab_xw_native_spectral_layouts,
        fftm_4d_pencil_same_zw_peer_paired=fftm_4d_pencil_same_zw_peer_paired,
        fftm_4d_pencil_same_zw_native_layouts=fftm_4d_pencil_same_zw_native_layouts,
        fftm_4d_pencil_degenerate_xw_slab_paths=fftm_4d_pencil_degenerate_xw_slab_paths,
        fftm_4d_pencil_degenerate_local_transposes=fftm_4d_pencil_degenerate_local_transposes,
        fftm_4d_pencil_degenerate_same_xw_native=fftm_4d_pencil_degenerate_same_xw_native,
        fftm_4d_pencil_degenerate_wz_sliced_z_fft=fftm_4d_pencil_degenerate_wz_sliced_z_fft,
        fftm_4d_native_xw_direct_layouts=fftm_4d_native_xw_direct_layouts,
        fftm_4d_native_xw_protocols=fftm_4d_native_xw_protocols,
        fftm_4d_native_xw_chunk_windows=fftm_4d_native_xw_chunk_windows,
        fftm_4d_native_xw_compact_stagings=fftm_4d_native_xw_compact_stagings,
        fftm_4d_slab_native_work_area_aliases=fftm_4d_slab_native_work_area_aliases,
        fftm_4d_slab_native_wz_communication_layouts=fftm_4d_slab_native_wz_communication_layouts,
        fftm_4d_slab_native_wz_plan_concurrencies=fftm_4d_slab_native_wz_plan_concurrencies,
        fftm_4d_slab_native_wz_ready_pipelines=fftm_4d_slab_native_wz_ready_pipelines,
        native_backward_second_peer_loop_modes=native_backward_second_peer_loop_modes,
        fftm_3d_backends=fftm_3d_backends,
        use_contiguous_forward_byte_send=use_contiguous_forward_byte_send,
        pencil_grid_orientations=pencil_grid_orientations,
        pencil_grid_orientations_4d=pencil_grid_orientations_4d,
    )
    return specs


def build_probe_families(specs: Sequence[RunSpec], probe_mode: str) -> List[ProbeFamily]:
    by_key: Dict[str, RunSpec] = {}
    for spec in specs:
        existing = by_key.get(spec.memory_family)
        if existing is None:
            by_key[spec.memory_family] = spec
            continue
        if existing.transport == "non_cuda_aware" and spec.transport == "cuda_aware":
            by_key[spec.memory_family] = spec

    families: List[ProbeFamily] = []
    for key in sorted(by_key):
        spec = by_key[key]
        if spec.uses_mpi:
            representative = RunSpec(
                suite=spec.suite,
                dim=spec.dim,
                case_name=spec.case_name,
                binary_name=spec.binary_name,
                binary_path=spec.binary_path,
                num_gpus=spec.num_gpus,
                transport=spec.transport,
                strategy=spec.strategy,
                mode=probe_mode,
                uses_mpi=spec.uses_mpi,
                supports_directory=spec.supports_directory,
                memory_family=spec.memory_family,
                p2p_variant=spec.p2p_variant,
                p2p_scheduler=spec.p2p_scheduler,
                pencil_layout=spec.pencil_layout,
                pencil_pipeline=spec.pencil_pipeline,
                large_count_p2p_transport=spec.large_count_p2p_transport,
                contiguous_forward_send_mode=spec.contiguous_forward_send_mode,
                native_backward_second_peer_loop=spec.native_backward_second_peer_loop,
                fftm_4d_slab_native_xw=spec.fftm_4d_slab_native_xw,
                fftm_4d_slab_xw_batched_peer_kernels=spec.fftm_4d_slab_xw_batched_peer_kernels,
                fftm_4d_slab_xw_tensor_coalesced_kernels=spec.fftm_4d_slab_xw_tensor_coalesced_kernels,
                fftm_4d_slab_xw_vector4_kernels=spec.fftm_4d_slab_xw_vector4_kernels,
                fftm_4d_slab_xw_tiled_kernels=spec.fftm_4d_slab_xw_tiled_kernels,
                fftm_4d_slab_xw_layout_stage=spec.fftm_4d_slab_xw_layout_stage,
                fftm_4d_slab_xw_native_spectral_layout=spec.fftm_4d_slab_xw_native_spectral_layout,
                fftm_4d_pencil_same_zw_peer_paired=spec.fftm_4d_pencil_same_zw_peer_paired,
                fftm_4d_pencil_same_zw_native_layout=spec.fftm_4d_pencil_same_zw_native_layout,
                fftm_4d_pencil_degenerate_xw_slab_path=spec.fftm_4d_pencil_degenerate_xw_slab_path,
                fftm_4d_pencil_degenerate_local_transposes=spec.fftm_4d_pencil_degenerate_local_transposes,
                fftm_4d_pencil_degenerate_same_xw_native=spec.fftm_4d_pencil_degenerate_same_xw_native,
                fftm_4d_pencil_degenerate_wz_sliced_z_fft=spec.fftm_4d_pencil_degenerate_wz_sliced_z_fft,
                fftm_4d_native_xw_direct_layout=spec.fftm_4d_native_xw_direct_layout,
                fftm_4d_native_xw_protocol=spec.fftm_4d_native_xw_protocol,
                fftm_4d_native_xw_chunk_window=spec.fftm_4d_native_xw_chunk_window,
                fftm_4d_native_xw_compact_staging=spec.fftm_4d_native_xw_compact_staging,
                fftm_3d_backend=spec.fftm_3d_backend,
                grid=spec.grid,
            )
        else:
            representative = spec
        families.append(ProbeFamily(key=key, dim=spec.dim, num_gpus=spec.num_gpus, representative=representative))
    return families


def round_size(value: int, dim: int, step_3d: int, step_4d: int, minimum: int) -> int:
    step = step_3d if dim == 3 else step_4d
    rounded = max(minimum, int(math.floor(float(value) / float(step)) * step))
    if dim == 4 and rounded % 2 != 0:
        rounded = max(minimum, rounded - 1)
    return max(minimum, rounded)


def midpoint_size(low: int, high: int, dim: int, step_3d: int, step_4d: int, minimum: int) -> int:
    if high <= low:
        return low
    return round_size((low + high) // 2, dim=dim, step_3d=step_3d, step_4d=step_4d, minimum=minimum)


class LocalPaperBenchmarkRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.tests_root = Path(args.tests_root).resolve()
        self.data_dir = Path(args.data_directory).resolve()
        self.raw_dir = self.data_dir / "raw"
        self.cpp_csv_dir = self.data_dir / "cpp_csv"
        self.runs_jsonl_path = self.data_dir / "runs.jsonl"
        self.size_limits_path = self.data_dir / "size_limits.json"
        self.hardware_path = self.data_dir / "hardware.json"
        self.config_path = self.data_dir / "config.json"
        self.matrix_path = self.data_dir / "matrix.json"
        self.detected_gpus = detect_gpus()
        self.effective_gpus = effective_gpu_inventory(self.detected_gpus, self.args.device_memory_mib)
        if not self.effective_gpus:
            raise RuntimeError("No GPUs detected. Use --device-memory-mib to override if needed.")
        self.max_gpus = min(self.args.max_gpus or len(self.effective_gpus), len(self.effective_gpus))
        self.binaries = discover_binaries(self.tests_root)
        self.p2p_variants = parse_p2p_variants(self.args.p2p_variants)
        self.p2p_schedulers = parse_p2p_schedulers(self.args.p2p_schedulers)
        self.pencil_layouts = parse_pencil_layouts(self.args.pencil_layouts)
        self.pencil_pipelines = parse_pencil_pipelines(self.args.pencil_pipelines)
        self.large_count_p2p_transports = parse_large_count_p2p_transports(
            self.args.large_count_p2p_transports
        )
        self.contiguous_forward_send_modes = parse_contiguous_forward_send_modes(
            self.args.contiguous_forward_send_modes
        )
        self.fftm_4d_slab_xw_transposes = parse_fftm_4d_slab_xw_transposes(
            self.args.fftm_4d_slab_xw_transposes
        )
        self.fftm_4d_slab_xw_batched_peer_kernels = parse_fftm_4d_slab_xw_batched_peer_kernels(
            self.args.fftm_4d_slab_xw_batched_peer_kernels
        )
        self.fftm_4d_slab_xw_kernel_layouts = parse_fftm_4d_slab_xw_kernel_layouts(
            self.args.fftm_4d_slab_xw_kernel_layouts
        )
        self.fftm_4d_slab_xw_vector4_kernels = parse_fftm_4d_slab_xw_vector4_kernels(
            self.args.fftm_4d_slab_xw_vector4_kernels
        )
        self.fftm_4d_slab_xw_tiled_kernels = parse_fftm_4d_slab_xw_tiled_kernels(
            self.args.fftm_4d_slab_xw_tiled_kernels
        )
        self.fftm_4d_slab_xw_layout_stages = parse_fftm_4d_slab_xw_layout_stages(
            self.args.fftm_4d_slab_xw_layout_stages
        )
        self.fftm_4d_slab_xw_native_spectral_layouts = parse_fftm_4d_slab_xw_native_spectral_layouts(
            self.args.fftm_4d_slab_xw_native_spectral_layouts
        )
        self.fftm_4d_pencil_same_zw_peer_paired = parse_fftm_4d_pencil_same_zw_peer_paired(
            self.args.fftm_4d_pencil_same_zw_peer_paired
        )
        self.fftm_4d_pencil_same_zw_native_layouts = parse_fftm_4d_pencil_same_zw_native_layouts(
            self.args.fftm_4d_pencil_same_zw_native_layouts
        )
        self.fftm_4d_pencil_degenerate_xw_slab_paths = parse_fftm_4d_pencil_degenerate_xw_slab_paths(
            self.args.fftm_4d_pencil_degenerate_xw_slab_paths
        )
        self.fftm_4d_pencil_degenerate_local_transposes = parse_fftm_4d_pencil_degenerate_local_transposes(
            self.args.fftm_4d_pencil_degenerate_local_transposes
        )
        self.fftm_4d_pencil_degenerate_same_xw_native = parse_fftm_4d_pencil_degenerate_same_xw_native(
            self.args.fftm_4d_pencil_degenerate_same_xw_native
        )
        self.fftm_4d_pencil_degenerate_wz_sliced_z_fft = parse_fftm_4d_pencil_degenerate_wz_sliced_z_fft(
            self.args.fftm_4d_pencil_degenerate_wz_sliced_z_fft
        )
        self.fftm_4d_native_xw_direct_layouts = parse_fftm_4d_native_xw_direct_layouts(
            self.args.fftm_4d_native_xw_direct_layouts
        )
        self.fftm_4d_native_xw_protocols = parse_fftm_4d_native_xw_protocols(
            self.args.fftm_4d_native_xw_protocols
        )
        self.fftm_4d_native_xw_chunk_windows = parse_fftm_4d_native_xw_chunk_windows(
            self.args.fftm_4d_native_xw_chunk_windows
        )
        self.fftm_4d_native_xw_compact_stagings = parse_fftm_4d_native_xw_compact_stagings(
            self.args.fftm_4d_native_xw_compact_stagings
        )
        self.fftm_4d_slab_native_work_area_aliases = parse_fftm_4d_slab_native_work_area_aliases(
            self.args.fftm_4d_slab_native_work_area_aliases
        )
        self.fftm_4d_slab_native_wz_communication_layouts = (
            parse_fftm_4d_slab_native_wz_communication_layouts(
                self.args.fftm_4d_slab_native_wz_communication_layouts
            )
        )
        self.fftm_4d_slab_native_wz_plan_concurrencies = (
            parse_fftm_4d_slab_native_wz_plan_concurrencies(
                self.args.fftm_4d_slab_native_wz_plan_concurrencies
            )
        )
        self.fftm_4d_slab_native_wz_ready_pipelines = (
            parse_fftm_4d_slab_native_wz_ready_pipelines(
                self.args.fftm_4d_slab_native_wz_ready_pipelines
            )
        )
        if any(protocol is not None for protocol in self.fftm_4d_native_xw_protocols) and True not in self.fftm_4d_native_xw_direct_layouts:
            raise ValueError(
                "--fftm-4d-native-xw-protocols requires --fftm-4d-native-xw-direct-layouts=on"
            )
        if any(window is not None for window in self.fftm_4d_native_xw_chunk_windows) and "chunked" not in self.fftm_4d_native_xw_protocols:
            raise ValueError(
                "--fftm-4d-native-xw-chunk-windows requires --fftm-4d-native-xw-protocols=chunked"
            )
        if True in self.fftm_4d_native_xw_compact_stagings:
            if True not in self.fftm_4d_native_xw_direct_layouts:
                raise ValueError(
                    "--fftm-4d-native-xw-compact-stagings=on requires "
                    "--fftm-4d-native-xw-direct-layouts=on"
                )
            if "chunked" not in self.fftm_4d_native_xw_protocols:
                raise ValueError(
                    "--fftm-4d-native-xw-compact-stagings=on requires "
                    "--fftm-4d-native-xw-protocols=chunked"
                )
            if not any(window is not None and window > 0 for window in self.fftm_4d_native_xw_chunk_windows):
                raise ValueError(
                    "--fftm-4d-native-xw-compact-stagings=on requires a positive bounded chunk window"
                )
        if True in self.fftm_4d_slab_native_work_area_aliases:
            if True not in self.fftm_4d_slab_xw_native_spectral_layouts:
                raise ValueError(
                    "--fftm-4d-slab-native-work-area-aliases=on requires "
                    "--fftm-4d-slab-xw-native-spectral-layouts=on"
                )
            if True not in self.fftm_4d_native_xw_direct_layouts:
                raise ValueError(
                    "--fftm-4d-slab-native-work-area-aliases=on requires "
                    "--fftm-4d-native-xw-direct-layouts=on"
                )
        if True in self.fftm_4d_slab_native_wz_communication_layouts:
            if True not in self.fftm_4d_slab_xw_native_spectral_layouts:
                raise ValueError(
                    "--fftm-4d-slab-native-wz-communication-layouts=on requires native spectral layout"
                )
            if True not in self.fftm_4d_native_xw_direct_layouts or "chunked" not in self.fftm_4d_native_xw_protocols:
                raise ValueError(
                    "--fftm-4d-slab-native-wz-communication-layouts=on requires direct chunked XW transport"
                )
            if not any(window is not None and window > 0 for window in self.fftm_4d_native_xw_chunk_windows):
                raise ValueError(
                    "--fftm-4d-slab-native-wz-communication-layouts=on requires a positive bounded chunk window"
                )
        if (
            any(value not in (None, 1) for value in self.fftm_4d_slab_native_wz_plan_concurrencies)
            or True in self.fftm_4d_slab_native_wz_ready_pipelines
        ) and True not in self.fftm_4d_slab_native_wz_communication_layouts:
            raise ValueError(
                "4D slab WZ concurrency/pipeline matrices require "
                "--fftm-4d-slab-native-wz-communication-layouts=on"
            )
        self.native_backward_second_peer_loop_modes = parse_native_backward_second_peer_loop_modes(
            self.args.native_backward_second_peer_loop_modes
        )
        self.fftm_3d_backends = parse_fftm_3d_backends(self.args.fftm_3d_backends)
        self.pencil_grid_orientations = parse_pencil_grid_orientations(self.args.pencil_pencil_grid_orientations)
        self.pencil_grid_orientations_4d = parse_4d_pencil_grid_orientations(
            self.args.fftm_4d_pencil_grid_orientations
        )
        self.extra_sizes_3d = parse_int_list(self.args.extra_sizes_3d)
        self.extra_sizes_3d_by_gpu = parse_gpu_size_map(self.args.extra_sizes_3d_by_gpu)
        self.fixed_scaling_sizes_3d = parse_int_list(self.args.fixed_scaling_sizes_3d)
        self.fixed_scaling_sizes_4d = parse_int_list(self.args.fixed_scaling_sizes_4d)
        self.specs = build_measurement_specs(
            self.binaries,
            max_gpus=self.max_gpus,
            include_nca=not self.args.skip_nca,
            p2p_variants=self.p2p_variants,
            p2p_schedulers=self.p2p_schedulers,
            pencil_layouts=self.pencil_layouts,
            pencil_pipelines=self.pencil_pipelines,
            large_count_p2p_transports=self.large_count_p2p_transports,
            contiguous_forward_send_modes=self.contiguous_forward_send_modes,
            fftm_4d_slab_xw_transposes=self.fftm_4d_slab_xw_transposes,
            fftm_4d_slab_xw_batched_peer_kernels=self.fftm_4d_slab_xw_batched_peer_kernels,
            fftm_4d_slab_xw_kernel_layouts=self.fftm_4d_slab_xw_kernel_layouts,
            fftm_4d_slab_xw_vector4_kernels=self.fftm_4d_slab_xw_vector4_kernels,
            fftm_4d_slab_xw_tiled_kernels=self.fftm_4d_slab_xw_tiled_kernels,
            fftm_4d_slab_xw_layout_stages=self.fftm_4d_slab_xw_layout_stages,
            fftm_4d_slab_xw_native_spectral_layouts=self.fftm_4d_slab_xw_native_spectral_layouts,
            fftm_4d_pencil_same_zw_peer_paired=self.fftm_4d_pencil_same_zw_peer_paired,
            fftm_4d_pencil_same_zw_native_layouts=self.fftm_4d_pencil_same_zw_native_layouts,
            fftm_4d_pencil_degenerate_xw_slab_paths=self.fftm_4d_pencil_degenerate_xw_slab_paths,
            fftm_4d_pencil_degenerate_local_transposes=self.fftm_4d_pencil_degenerate_local_transposes,
            fftm_4d_pencil_degenerate_same_xw_native=self.fftm_4d_pencil_degenerate_same_xw_native,
            fftm_4d_pencil_degenerate_wz_sliced_z_fft=self.fftm_4d_pencil_degenerate_wz_sliced_z_fft,
            fftm_4d_native_xw_direct_layouts=self.fftm_4d_native_xw_direct_layouts,
            fftm_4d_native_xw_protocols=self.fftm_4d_native_xw_protocols,
            fftm_4d_native_xw_chunk_windows=self.fftm_4d_native_xw_chunk_windows,
            fftm_4d_native_xw_compact_stagings=self.fftm_4d_native_xw_compact_stagings,
            fftm_4d_slab_native_work_area_aliases=self.fftm_4d_slab_native_work_area_aliases,
            fftm_4d_slab_native_wz_communication_layouts=
                self.fftm_4d_slab_native_wz_communication_layouts,
            fftm_4d_slab_native_wz_plan_concurrencies=self.fftm_4d_slab_native_wz_plan_concurrencies,
            fftm_4d_slab_native_wz_ready_pipelines=self.fftm_4d_slab_native_wz_ready_pipelines,
            native_backward_second_peer_loop_modes=self.native_backward_second_peer_loop_modes,
            fftm_3d_backends=self.fftm_3d_backends,
            use_contiguous_forward_byte_send=self.args.use_contiguous_forward_byte_send,
            pencil_grid_orientations=self.pencil_grid_orientations,
            pencil_grid_orientations_4d=self.pencil_grid_orientations_4d,
        )
        if not self.specs:
            raise RuntimeError(f"No runnable benchmark/test binaries were found under {self.tests_root}")
        self.probe_families = build_probe_families(self.specs, probe_mode=self.args.probe_mode)
        self.run_index = 0
        self.child_env = self.build_child_env()

    def build_child_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        paths: List[str] = []
        for path in self.args.library_path:
            if path:
                paths.append(path)
        for path in ("/usr/local/mpi/lib", "/usr/local/cuda/lib64"):
            if Path(path).is_dir():
                paths.append(path)
        existing = env.get("LD_LIBRARY_PATH")
        if existing:
            paths.append(existing)
        if paths:
            env["LD_LIBRARY_PATH"] = ":".join(paths)
        return env

    def prepare_output(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cpp_csv_dir.mkdir(parents=True, exist_ok=True)
        self.hardware_path.write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "detected_gpus": self.detected_gpus,
                    "effective_gpus": self.effective_gpus,
                    "max_gpus_used": self.max_gpus,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.config_path.write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "args": vars(self.args),
                    "tests_root": str(self.tests_root),
                    "repo_root": str(self.repo_root),
                    "data_directory": str(self.data_dir),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.matrix_path.write_text(
            json.dumps(
                {
                    "created_at": now_iso(),
                    "probe_families": [
                        {"key": family.key, "dim": family.dim, "num_gpus": family.num_gpus, "representative": asdict(family.representative)}
                        for family in self.probe_families
                    ],
                    "measurement_specs": [asdict(spec) for spec in self.specs],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def per_gpu_budget_bytes(self, num_gpus: int) -> int:
        selected = self.effective_gpus[:num_gpus]
        min_total = min(gpu["memory_total_bytes"] for gpu in selected)
        usable = int(min_total * self.args.usable_memory_fraction) - mib_to_bytes(self.args.reserve_memory_mib)
        return max(0, usable)

    def size_tuple(self, dim: int, n: int) -> Tuple[int, ...]:
        if dim == 3:
            return (n, n, n)
        return (n, n, n, n)

    def build_command(self, spec: RunSpec, sizes: Tuple[int, ...], times: int, warmup: int = 0) -> List[str]:
        command: List[str] = []
        if spec.uses_mpi:
            command.extend(["mpiexec", "-n", str(spec.num_gpus)])
        command.append(spec.binary_path)
        if spec.strategy is not None:
            command.extend(["--strategy", spec.strategy])
        if spec.mode is not None:
            command.extend(["--mode", spec.mode])
        if spec.grid is not None:
            command.extend(["--grid", *(str(v) for v in spec.grid)])
        if spec.supports_directory:
            command.extend(["--directory", str(self.cpp_csv_dir)])
        if spec.suite == "fftm" and spec.case_name != "benchmark":
            command.extend(["--threshold", str(self.args.validation_threshold)])
        else:
            command.extend(["--epsilon", str(self.args.validation_epsilon)])
        if spec.suite == "fftm":
            use_direct_backward_receive, direct_p2p_cuda_aware, use_p2p_byte_transfer = p2p_variant_flags(
                spec.p2p_variant,
                self.args.use_direct_backward_receive,
                self.args.direct_p2p_cuda_aware,
                self.args.use_p2p_byte_transfer,
            )
            command.append(
                "--use-direct-backward-receive"
                if use_direct_backward_receive
                else "--no-direct-backward-receive"
            )
            command.append(
                "--direct-p2p-cuda-aware"
                if direct_p2p_cuda_aware
                else "--no-direct-p2p-cuda-aware"
            )
            command.append(
                "--use-fft-exec-no-sync"
                if self.args.use_fft_exec_no_sync
                else "--no-fft-exec-no-sync"
            )
            if spec.dim == 3:
                if self.args.fftm_autotune_config:
                    command.extend(["--autotune-config", self.args.fftm_autotune_config])
                use_p2p_send_thread, use_persistent_p2p = p2p_scheduler_flags(
                    spec.p2p_scheduler,
                    self.args.use_p2p_send_thread,
                    self.args.use_persistent_p2p,
                )
                command.append(
                    "--use-p2p-send-thread" if use_p2p_send_thread else "--no-p2p-send-thread"
                )
                command.append(
                    "--use-p2p-byte-transfer"
                    if use_p2p_byte_transfer
                    else "--no-p2p-byte-transfer"
                )
                command.append(
                    "--use-persistent-p2p" if use_persistent_p2p else "--no-persistent-p2p"
                )
                command.append(
                    "--use-ready-p2p-send"
                    if self.args.use_ready_p2p_send
                    else "--no-ready-p2p-send"
                )
                command.append(
                    "--print-pencil-schedule"
                    if self.args.print_pencil_schedule
                    else "--no-print-pencil-schedule"
                )
                command.append(
                    "--use-direct-forward-byte-receive"
                    if self.args.use_direct_forward_byte_receive
                    else "--no-direct-forward-byte-receive"
                )
                command.append(
                    "--use-stable-forward-byte-send-buffer"
                    if self.args.use_stable_forward_byte_send_buffer
                    else "--no-stable-forward-byte-send-buffer"
                )
                command.append(
                    "--use-ready-stable-forward-byte-send-buffer"
                    if self.args.use_ready_stable_forward_byte_send_buffer
                    else "--no-ready-stable-forward-byte-send-buffer"
                )
                command.append(
                    "--use-contiguous-forward-byte-send"
                    if self.args.use_contiguous_forward_byte_send
                    else "--no-contiguous-forward-byte-send"
                )
                command.append(
                    "--use-physical-forward-peer-exchange"
                    if self.args.use_physical_forward_peer_exchange
                    else "--no-physical-forward-peer-exchange"
                )
                if spec.native_backward_second_peer_loop is None:
                    command.append(
                        "--use-native-backward-second-peer-loop"
                        if self.args.use_native_backward_second_peer_loop
                        else "--no-native-backward-second-peer-loop"
                    )
                else:
                    command.append(
                        "--use-native-backward-second-peer-loop"
                        if spec.native_backward_second_peer_loop
                        else "--no-native-backward-second-peer-loop"
                    )
                command.append(
                    "--use-3d-deferred-send-completion"
                    if self.args.use_3d_deferred_send_completion
                    else "--no-3d-deferred-send-completion"
                )
                command.append(
                    "--use-native-opt0-default-z-layout"
                    if self.args.use_native_opt0_default_z_layout
                    else "--no-native-opt0-default-z-layout"
                )
                command.append(
                    "--use-native-opt0-reference-y-buffer-topology"
                    if self.args.use_native_opt0_reference_y_buffer_topology
                    else "--no-native-opt0-reference-y-buffer-topology"
                )
                command.append(
                    "--use-native-opt0-compact-y-workarea"
                    if self.args.use_native_opt0_compact_y_workarea
                    else "--no-native-opt0-compact-y-workarea"
                )
                command.append(
                    "--use-native-opt0-tight-y-plan-sequence"
                    if self.args.use_native_opt0_tight_y_plan_sequence
                    else "--no-native-opt0-tight-y-plan-sequence"
                )
                command.append(
                    "--use-native-opt0-shared-y-plan-handles"
                    if self.args.use_native_opt0_shared_y_plan_handles
                    else "--no-native-opt0-shared-y-plan-handles"
                )
                command.append(
                    "--use-native-opt0-y-group-device-sync"
                    if self.args.use_native_opt0_y_group_device_sync
                    else "--no-native-opt0-y-group-device-sync"
                )
                command.append(
                    "--use-native-opt0-y-no-sync-exec"
                    if self.args.use_native_opt0_y_no_sync_exec
                    else "--no-native-opt0-y-no-sync-exec"
                )
                command.append(
                    "--use-native-opt0-raw-y-plan-array-executor"
                    if self.args.use_native_opt0_raw_y_plan_array_executor
                    else "--no-native-opt0-raw-y-plan-array-executor"
                )
                command.append(
                    "--use-native-opt0-reference-y-plan-lifecycle"
                    if self.args.use_native_opt0_reference_y_plan_lifecycle
                    else "--no-native-opt0-reference-y-plan-lifecycle"
                )
                command.append(
                    "--use-native-opt0-reference-y-plan-bundle"
                    if self.args.use_native_opt0_reference_y_plan_bundle
                    else "--no-native-opt0-reference-y-plan-bundle"
                )
                command.append(
                    "--use-native-opt0-raw-y-plan-bundle"
                    if self.args.use_native_opt0_raw_y_plan_bundle
                    else "--no-native-opt0-raw-y-plan-bundle"
                )
                command.append(
                    "--use-native-opt0-y-plan-bundle-stream-first"
                    if self.args.use_native_opt0_y_plan_bundle_stream_first
                    else "--no-native-opt0-y-plan-bundle-stream-first"
                )
                command.append(
                    "--use-native-opt0-raw-y-plan-bundle-reference-streams"
                    if self.args.use_native_opt0_raw_y_plan_bundle_reference_streams
                    else "--no-native-opt0-raw-y-plan-bundle-reference-streams"
                )
                command.append(
                    "--use-native-opt0-reference-local-plan-context"
                    if self.args.use_native_opt0_reference_local_plan_context
                    else "--no-native-opt0-reference-local-plan-context"
                )
                native_opt0_y_flags = {
                    "reference_lifecycle": self.args.use_native_opt0_reference_y_plan_lifecycle,
                    "reference_bundle": self.args.use_native_opt0_reference_y_plan_bundle,
                    "raw_bundle": self.args.use_native_opt0_raw_y_plan_bundle,
                    "stream_first": self.args.use_native_opt0_y_plan_bundle_stream_first,
                    "reference_streams": self.args.use_native_opt0_raw_y_plan_bundle_reference_streams,
                    "local_context": self.args.use_native_opt0_reference_local_plan_context,
                }
                allow_native_opt0_diagnostics = self.args.allow_native_opt0_diagnostic_variants
                command.append(
                    "--allow-native-opt0-diagnostic-variants"
                    if allow_native_opt0_diagnostics
                    else "--no-native-opt0-diagnostic-variants"
                )
                command.append(
                    "--use-native-opt0-memory-feasibility-guard"
                    if self.args.use_native_opt0_memory_feasibility_guard
                    else "--no-native-opt0-memory-feasibility-guard"
                )
                command.extend(
                    [
                        "--native-opt0-memory-feasibility-reserve-mib",
                        str(self.args.native_opt0_memory_feasibility_reserve_mib),
                    ]
                )
                if spec.contiguous_forward_send_mode is not None:
                    command.extend(["--contiguous-forward-send-mode", spec.contiguous_forward_send_mode])
                else:
                    command.extend(
                        ["--contiguous-forward-send-mode", self.args.contiguous_forward_send_mode]
                    )
                command.extend(
                    [
                        "--contiguous-forward-send-chunk-mib",
                        str(self.args.contiguous_forward_send_chunk_mib),
                    ]
                )
                if spec.case_name == "benchmark":
                    command.extend(
                        [
                            "--contiguous-forward-send-registration-warmups",
                            str(self.args.contiguous_forward_send_registration_warmups),
                        ]
                    )
                if spec.pencil_layout is not None:
                    command.extend(["--pencil-layout", spec.pencil_layout])
                if spec.pencil_pipeline is not None:
                    command.extend(["--pencil-pipeline", spec.pencil_pipeline])
                if spec.large_count_p2p_transport is not None:
                    command.extend(["--large-count-p2p-transport", spec.large_count_p2p_transport])
                if spec.fftm_3d_backend is not None:
                    command.extend(["--fftm-3d-backend", spec.fftm_3d_backend])
                command.append(
                    "--enable-fftm3d-backend-stage-timers"
                    if self.args.enable_fftm3d_backend_stage_timers
                    else "--disable-fftm3d-backend-stage-timers"
                )
                command.append(
                    "--enable-local-fft-diagnostics"
                    if self.args.enable_local_fft_diagnostics
                    else "--disable-local-fft-diagnostics"
                )
                command.append(
                    "--enable-native-stage-timers"
                    if self.args.enable_native_stage_timers
                    else "--disable-native-stage-timers"
                )
                command.append(
                    "--use-large-count-datatype-cache"
                    if self.args.use_large_count_datatype_cache
                    else "--no-large-count-datatype-cache"
                )
                command.append(
                    "--enable-gpu-telemetry"
                    if self.args.enable_gpu_telemetry
                    else "--disable-gpu-telemetry"
                )
            else:
                command.append(
                    "--enable-native-stage-timers"
                    if self.args.enable_native_stage_timers
                    else "--disable-native-stage-timers"
                )
                if spec.fftm_4d_slab_native_xw is not None:
                    command.append(
                        "--use-4d-slab-native-xw-transpose"
                        if spec.fftm_4d_slab_native_xw
                        else "--no-4d-slab-native-xw-transpose"
                    )
                if spec.fftm_4d_slab_xw_batched_peer_kernels is not None:
                    command.append(
                        "--use-4d-slab-native-xw-batched-peer-kernels"
                        if spec.fftm_4d_slab_xw_batched_peer_kernels
                        else "--no-4d-slab-native-xw-batched-peer-kernels"
                    )
                if spec.fftm_4d_slab_xw_tensor_coalesced_kernels is not None:
                    command.append(
                        "--use-4d-slab-native-xw-tensor-coalesced-kernels"
                        if spec.fftm_4d_slab_xw_tensor_coalesced_kernels
                        else "--no-4d-slab-native-xw-tensor-coalesced-kernels"
                    )
                if spec.fftm_4d_slab_xw_vector4_kernels is not None:
                    command.append(
                        "--use-4d-slab-native-xw-vector4-kernels"
                        if spec.fftm_4d_slab_xw_vector4_kernels
                        else "--no-4d-slab-native-xw-vector4-kernels"
                    )
                if spec.fftm_4d_slab_xw_tiled_kernels is not None:
                    command.append(
                        "--use-4d-slab-native-xw-tiled-kernels"
                        if spec.fftm_4d_slab_xw_tiled_kernels
                        else "--no-4d-slab-native-xw-tiled-kernels"
                    )
                if spec.fftm_4d_slab_xw_layout_stage is not None:
                    command.append(
                        "--use-4d-slab-native-xw-layout-stage"
                        if spec.fftm_4d_slab_xw_layout_stage
                        else "--no-4d-slab-native-xw-layout-stage"
                    )
                if spec.fftm_4d_slab_xw_native_spectral_layout is not None:
                    command.append(
                        "--use-4d-slab-native-xw-native-spectral-layout"
                        if spec.fftm_4d_slab_xw_native_spectral_layout
                        else "--no-4d-slab-native-xw-native-spectral-layout"
                    )
                if spec.fftm_4d_native_xw_direct_layout is not None:
                    command.append(
                        "--use-4d-native-xw-direct-layout"
                        if spec.fftm_4d_native_xw_direct_layout
                        else "--no-4d-native-xw-direct-layout"
                    )
                if spec.fftm_4d_native_xw_protocol is not None:
                    command.append(
                        "--use-4d-native-xw-chunked-transport"
                        if spec.fftm_4d_native_xw_protocol == "chunked"
                        else "--no-4d-native-xw-chunked-transport"
                    )
                    if spec.fftm_4d_native_xw_protocol == "chunked":
                        command.extend(["--4d-native-xw-chunk-mib", str(self.args.fftm_4d_native_xw_chunk_mib)])
                        if spec.fftm_4d_native_xw_chunk_window is not None:
                            command.extend(
                                [
                                    "--4d-native-xw-chunk-window",
                                    "all"
                                    if spec.fftm_4d_native_xw_chunk_window == 0
                                    else str(spec.fftm_4d_native_xw_chunk_window),
                                ]
                            )
                if spec.fftm_4d_native_xw_compact_staging is not None:
                    command.append(
                        "--use-4d-native-xw-compact-staging"
                        if spec.fftm_4d_native_xw_compact_staging
                        else "--no-4d-native-xw-compact-staging"
                    )
                if spec.fftm_4d_slab_native_work_area_alias is not None:
                    command.append(
                        "--use-4d-slab-native-work-area-alias"
                        if spec.fftm_4d_slab_native_work_area_alias
                        else "--no-4d-slab-native-work-area-alias"
                    )
                if spec.fftm_4d_slab_native_wz_communication_layout is not None:
                    command.append(
                        "--use-4d-slab-native-wz-communication-layout"
                        if spec.fftm_4d_slab_native_wz_communication_layout
                        else "--no-4d-slab-native-wz-communication-layout"
                    )
                if spec.fftm_4d_slab_native_wz_plan_concurrency is not None:
                    command.extend(
                        ["--4d-slab-native-wz-plan-concurrency", str(spec.fftm_4d_slab_native_wz_plan_concurrency)]
                    )
                if spec.fftm_4d_slab_native_wz_ready_pipeline is not None:
                    command.append(
                        "--use-4d-slab-native-wz-ready-pipeline"
                        if spec.fftm_4d_slab_native_wz_ready_pipeline
                        else "--no-4d-slab-native-wz-ready-pipeline"
                    )
                if spec.fftm_4d_pencil_same_zw_peer_paired is not None:
                    command.append(
                        "--use-4d-pencil-same-zw-peer-paired"
                        if spec.fftm_4d_pencil_same_zw_peer_paired
                        else "--no-4d-pencil-same-zw-peer-paired"
                    )
                if spec.fftm_4d_pencil_same_zw_native_layout is not None:
                    command.append(
                        "--use-4d-pencil-same-zw-native-layout"
                        if spec.fftm_4d_pencil_same_zw_native_layout
                        else "--no-4d-pencil-same-zw-native-layout"
                    )
                if spec.fftm_4d_pencil_degenerate_xw_slab_path is not None:
                    command.append(
                        "--use-4d-pencil-degenerate-xw-slab-path"
                        if spec.fftm_4d_pencil_degenerate_xw_slab_path
                        else "--no-4d-pencil-degenerate-xw-slab-path"
                    )
                if spec.fftm_4d_pencil_degenerate_local_transposes is not None:
                    command.append(
                        "--use-4d-pencil-degenerate-local-transposes"
                        if spec.fftm_4d_pencil_degenerate_local_transposes
                        else "--no-4d-pencil-degenerate-local-transposes"
                    )
                if spec.fftm_4d_pencil_degenerate_same_xw_native is not None:
                    command.append(
                        "--use-4d-pencil-degenerate-same-xw-native"
                        if spec.fftm_4d_pencil_degenerate_same_xw_native
                        else "--no-4d-pencil-degenerate-same-xw-native"
                    )
                if spec.fftm_4d_pencil_degenerate_wz_sliced_z_fft is not None:
                    command.append(
                        "--use-4d-pencil-degenerate-wz-sliced-z-fft"
                        if spec.fftm_4d_pencil_degenerate_wz_sliced_z_fft
                        else "--no-4d-pencil-degenerate-wz-sliced-z-fft"
                    )
        command.extend(["--times", str(times)])
        if warmup > 0:
            command.extend(["--warmup", str(warmup)])
        command.extend(str(size) for size in sizes)
        return command

    def record_run(self, record: Dict[str, Any]) -> None:
        with self.runs_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def execute_run(self, spec: RunSpec, sizes: Tuple[int, ...], times: int, phase: str) -> Dict[str, Any]:
        self.run_index += 1
        warmup = self.args.test_warmup if phase == PHASE_MEASURE else 0
        command = self.build_command(spec, sizes, times, warmup)
        command_string = " ".join(shlex.quote(arg) for arg in command)
        slug = short_log_slug(
            f"{self.run_index:05d}_{phase}_{spec.suite}_{spec.case_name}_{spec.dim}d_"
            f"g{spec.num_gpus}_{spec.transport}_{spec.strategy or 'default'}_{spec.mode or 'none'}_"
            f"{spec.p2p_variant or 'configured'}_{spec.p2p_scheduler or 'sched-configured'}_"
            f"{spec.pencil_layout or 'layout-configured'}_{spec.pencil_pipeline or 'pipe-configured'}_"
            f"{spec.large_count_p2p_transport or 'large-configured'}_"
            f"{spec.contiguous_forward_send_mode or 'contig-configured'}_"
            f"4dxw-{('native' if spec.fftm_4d_slab_native_xw else 'staged') if spec.fftm_4d_slab_native_xw is not None else 'configured'}_"
            f"xwkernel-{('tensor' if spec.fftm_4d_slab_xw_tensor_coalesced_kernels else 'buffer') if spec.fftm_4d_slab_xw_tensor_coalesced_kernels is not None else 'configured'}_"
            f"xwvec4-{('on' if spec.fftm_4d_slab_xw_vector4_kernels else 'off') if spec.fftm_4d_slab_xw_vector4_kernels is not None else 'configured'}_"
            f"xwtile-{('on' if spec.fftm_4d_slab_xw_tiled_kernels else 'off') if spec.fftm_4d_slab_xw_tiled_kernels is not None else 'configured'}_"
            f"xwstage-{('on' if spec.fftm_4d_slab_xw_layout_stage else 'off') if spec.fftm_4d_slab_xw_layout_stage is not None else 'configured'}_"
            f"xwspec-{('native' if spec.fftm_4d_slab_xw_native_spectral_layout else 'public') if spec.fftm_4d_slab_xw_native_spectral_layout is not None else 'configured'}_"
            f"xwdirect-{('on' if spec.fftm_4d_native_xw_direct_layout else 'off') if spec.fftm_4d_native_xw_direct_layout is not None else 'configured'}_"
            f"xwproto-{spec.fftm_4d_native_xw_protocol or 'configured'}_"
            f"xwwindow-{spec.fftm_4d_native_xw_chunk_window if spec.fftm_4d_native_xw_chunk_window is not None else 'configured'}_"
            f"xwcompact-{('on' if spec.fftm_4d_native_xw_compact_staging else 'off') if spec.fftm_4d_native_xw_compact_staging is not None else 'configured'}_"
            f"workalias-{('on' if spec.fftm_4d_slab_native_work_area_alias else 'off') if spec.fftm_4d_slab_native_work_area_alias is not None else 'configured'}_"
            f"wzcomm-{('on' if spec.fftm_4d_slab_native_wz_communication_layout else 'off') if spec.fftm_4d_slab_native_wz_communication_layout is not None else 'configured'}_"
            f"wzconc-{spec.fftm_4d_slab_native_wz_plan_concurrency if spec.fftm_4d_slab_native_wz_plan_concurrency is not None else 'configured'}_"
            f"wzpipe-{('on' if spec.fftm_4d_slab_native_wz_ready_pipeline else 'off') if spec.fftm_4d_slab_native_wz_ready_pipeline is not None else 'configured'}_"
            f"xwbatch-{('on' if spec.fftm_4d_slab_xw_batched_peer_kernels else 'off') if spec.fftm_4d_slab_xw_batched_peer_kernels is not None else 'configured'}_"
            f"samezw-{('peer' if spec.fftm_4d_pencil_same_zw_peer_paired else 'base') if spec.fftm_4d_pencil_same_zw_peer_paired is not None else 'configured'}_"
            f"samezwlayout-{('native' if spec.fftm_4d_pencil_same_zw_native_layout else 'base') if spec.fftm_4d_pencil_same_zw_native_layout is not None else 'configured'}_"
            f"degenxw-{('on' if spec.fftm_4d_pencil_degenerate_xw_slab_path else 'off') if spec.fftm_4d_pencil_degenerate_xw_slab_path is not None else 'configured'}_"
            f"degenlocal-{('on' if spec.fftm_4d_pencil_degenerate_local_transposes else 'off') if spec.fftm_4d_pencil_degenerate_local_transposes is not None else 'configured'}_"
            f"degensamexw-{('native' if spec.fftm_4d_pencil_degenerate_same_xw_native else 'base') if spec.fftm_4d_pencil_degenerate_same_xw_native is not None else 'configured'}_"
            f"degenwz-{('on' if spec.fftm_4d_pencil_degenerate_wz_sliced_z_fft else 'off') if spec.fftm_4d_pencil_degenerate_wz_sliced_z_fft is not None else 'configured'}_"
            f"bwd2peer-{('on' if spec.native_backward_second_peer_loop else 'off') if spec.native_backward_second_peer_loop is not None else 'configured'}_"
            f"{spec.fftm_3d_backend or 'backend-configured'}_"
            f"{grid_slug(spec.grid)}_{'x'.join(str(s) for s in sizes)}"
        )
        raw_path = self.raw_dir / f"{slug}.log"
        started_at = now_iso()
        started_monotonic = time.monotonic()
        completed = subprocess.run(
            command,
            cwd=str(self.tests_root),
            env=self.child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=self.args.timeout_seconds,
            check=False,
        )
        elapsed_seconds = time.monotonic() - started_monotonic
        stdout = completed.stdout
        raw_path.write_text(stdout, encoding="utf-8")
        parsed = parse_run_output(stdout)
        n = sizes[0]
        used_device_peak_max_bytes = infer_device_peak_max_bytes(parsed, spec, n)
        record: Dict[str, Any] = {
            "phase": phase,
            "started_at": started_at,
            "finished_at": now_iso(),
            "elapsed_seconds": elapsed_seconds,
            "returncode": completed.returncode,
            "command": command,
            "command_string": command_string,
            "cwd": str(self.tests_root),
            "raw_log": str(raw_path),
            "spec": asdict(spec),
            "sizes": list(sizes),
            "times": times,
            "warmup": warmup,
            "parsed": parsed,
            "used_device_peak_max_bytes": used_device_peak_max_bytes,
            "used_device_peak_max_mib": bytes_to_mib(used_device_peak_max_bytes)
            if used_device_peak_max_bytes is not None
            else None,
        }
        self.record_run(record)
        return record

    def fit_family(self, family: ProbeFamily) -> Dict[str, Any]:
        dim = family.dim
        minimum = self.args.min_size_3d if dim == 3 else self.args.min_size_4d
        step = self.args.size_step_3d if dim == 3 else self.args.size_step_4d
        budget_bytes = self.per_gpu_budget_bytes(family.num_gpus)
        if budget_bytes <= 0:
            return {
                "family": family.key,
                "status": "no_budget",
                "budget_bytes": budget_bytes,
                "budget_mib": bytes_to_mib(budget_bytes),
                "chosen_n": None,
                "history": [],
            }

        n = round_size(
            self.args.initial_size_3d if dim == 3 else self.args.initial_size_4d,
            dim=dim,
            step_3d=self.args.size_step_3d,
            step_4d=self.args.size_step_4d,
            minimum=minimum,
        )
        low_fit: Optional[int] = None
        high_fail: Optional[int] = None
        history: List[Dict[str, Any]] = []
        tried: set[int] = set()

        for _ in range(self.args.probe_max_runs):
            if n in tried:
                break
            tried.add(n)
            sizes = self.size_tuple(dim, n)
            record = self.execute_run(family.representative, sizes, self.args.probe_times, PHASE_PROBE)
            used_bytes = record["used_device_peak_max_bytes"]
            fit = record["returncode"] == 0 and used_bytes is not None and used_bytes <= budget_bytes
            history.append(
                {
                    "n": n,
                    "sizes": list(sizes),
                    "fit": fit,
                    "returncode": record["returncode"],
                    "used_device_peak_max_bytes": used_bytes,
                    "used_device_peak_max_mib": record["used_device_peak_max_mib"],
                    "raw_log": record["raw_log"],
                }
            )

            if fit:
                low_fit = n
                if used_bytes is None:
                    break
                headroom_ratio = float(budget_bytes) / float(max(1, used_bytes))
                if headroom_ratio <= 1.02:
                    break
                if high_fail is not None and high_fail - low_fit <= step:
                    break
                predicted = int(math.floor(n * (headroom_ratio ** (1.0 / float(dim))) * self.args.probe_safety_factor))
                next_n = round_size(
                    max(n + step, predicted),
                    dim=dim,
                    step_3d=self.args.size_step_3d,
                    step_4d=self.args.size_step_4d,
                    minimum=minimum,
                )
                if high_fail is not None and next_n >= high_fail:
                    next_n = midpoint_size(
                        low_fit,
                        high_fail,
                        dim=dim,
                        step_3d=self.args.size_step_3d,
                        step_4d=self.args.size_step_4d,
                        minimum=minimum,
                    )
                if next_n <= n:
                    break
                n = next_n
            else:
                high_fail = n
                if low_fit is None:
                    next_n = round_size(
                        max(minimum, n // 2),
                        dim=dim,
                        step_3d=self.args.size_step_3d,
                        step_4d=self.args.size_step_4d,
                        minimum=minimum,
                    )
                    if next_n >= n:
                        break
                    n = next_n
                else:
                    if high_fail - low_fit <= step:
                        break
                    next_n = midpoint_size(
                        low_fit,
                        high_fail,
                        dim=dim,
                        step_3d=self.args.size_step_3d,
                        step_4d=self.args.size_step_4d,
                        minimum=minimum,
                    )
                    if next_n <= low_fit or next_n >= high_fail:
                        break
                    n = next_n

        return {
            "family": family.key,
            "status": "ok" if low_fit is not None else "no_fit",
            "budget_bytes": budget_bytes,
            "budget_mib": bytes_to_mib(budget_bytes),
            "chosen_n": low_fit,
            "chosen_sizes": list(self.size_tuple(dim, low_fit)) if low_fit is not None else None,
            "history": history,
            "representative": asdict(family.representative),
        }

    def run(self) -> int:
        self.prepare_output()

        family_results: Dict[str, Dict[str, Any]] = {}
        for family in self.probe_families:
            family_results[family.key] = self.fit_family(family)

        self.size_limits_path.write_text(
            json.dumps({"created_at": now_iso(), "families": family_results}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        failed_runs = 0
        skipped_runs = 0
        for spec in self.specs:
            family_result = family_results.get(spec.memory_family)
            chosen_n = None if family_result is None else family_result.get("chosen_n")
            if chosen_n is None:
                skipped_runs += 1
                continue
            side_lengths = [int(chosen_n)]
            if spec.dim == 3 and spec.case_name == "benchmark":
                side_lengths.extend(self.extra_sizes_3d)
                side_lengths.extend(self.extra_sizes_3d_by_gpu.get(spec.num_gpus, []))
            if spec.suite == "fftm" and spec.case_name == "benchmark":
                if spec.dim == 3:
                    side_lengths.extend(self.fixed_scaling_sizes_3d)
                elif spec.dim == 4:
                    side_lengths.extend(self.fixed_scaling_sizes_4d)
            for side_length in sorted(set(side_lengths)):
                record = self.execute_run(
                    spec, self.size_tuple(spec.dim, int(side_length)), self.args.measure_times, PHASE_MEASURE
                )
                if record["returncode"] != 0:
                    failed_runs += 1

        summary = {
            "created_at": now_iso(),
            "num_probe_families": len(self.probe_families),
            "num_measurement_specs": len(self.specs),
            "skipped_measurement_specs": skipped_runs,
            "failed_measurement_runs": failed_runs,
        }
        (self.data_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        return 0 if failed_runs == 0 else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run local FFTS/FFTM performance and validation matrices, estimate economical max problem sizes from "
            "stdout memory data, and save raw + parsed results for later visualization."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--tests-root",
        default="source/tests",
        help="Directory containing built test binaries. Default: source/tests",
    )
    parser.add_argument(
        "--data-directory",
        required=True,
        help="Directory where raw logs and parsed data will be written.",
    )
    parser.add_argument(
        "--device-memory-mib",
        type=int,
        nargs="*",
        default=None,
        help="Optional per-GPU VRAM override in MiB, one value per detected GPU.",
    )
    parser.add_argument(
        "--usable-memory-fraction",
        type=float,
        default=0.82,
        help="Per-GPU memory budget fraction used during size fitting. Default: 0.82",
    )
    parser.add_argument(
        "--reserve-memory-mib",
        type=int,
        default=512,
        help="Per-GPU memory reserve subtracted from the usable budget. Default: 512",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Optional cap on the number of local GPUs to include.",
    )
    parser.add_argument(
        "--skip-nca",
        action="store_true",
        help="Skip *_nca.bin FFTM runs.",
    )
    parser.add_argument(
        "--probe-mode",
        choices=FFTM_MODES,
        default="p2p-waitany",
        help="Representative FFTM mode used during size probing. Default: p2p-waitany",
    )
    parser.add_argument(
        "--probe-times",
        type=int,
        default=1,
        help="Iteration count passed to probe runs. Default: 1",
    )
    parser.add_argument(
        "--measure-times",
        type=int,
        default=10,
        help="Iteration count passed to the final measurement runs. Default: 10",
    )
    parser.add_argument(
        "--warmup",
        "--benchmark-warmup",
        dest="test_warmup",
        type=int,
        default=3,
        help="Untimed warmup iterations for final measurement runs. Probe runs remain un-warmed. Default: 3",
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
        help="Path passed to FFTM 3D binaries as --autotune-config.",
    )
    parser.add_argument(
        "--p2p-variants",
        default="configured",
        help=(
            "3D CUDA-aware p2p variant matrix: configured, all, or comma-separated subset of "
            "value-packed,datatype-direct,byte-packed,byte-direct. Default: configured"
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
        "--use-3d-deferred-send-completion",
        action="store_true",
        default=False,
        help=(
            "Experimental: defer CUDA-aware 3D pencil send completion across the following local FFT. "
            "Restricted to native reference-parity p2p-waitany byte transfers."
        ),
    )
    parser.add_argument(
        "--no-3d-deferred-send-completion",
        action="store_false",
        dest="use_3d_deferred_send_completion",
        help="Disable deferred 3D pencil send completion.",
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
        "--fftm-4d-slab-xw-transposes",
        default="configured",
        help=(
            "4D slab-slab XW transpose matrix: configured, all/both, or comma-separated subset "
            "of staged,native. staged is the current local-reorder path; native skips the local "
            "xywz/xzwy reorder pair around same_xw."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-batched-peer-kernels",
        default="configured",
        help=(
            "4D slab-slab native-XW pack/unpack kernel matrix: configured, all/both, or comma-separated subset "
            "of legacy,batched. legacy waits after each peer kernel; batched waits once per pack/unpack group."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-kernel-layouts",
        default="configured",
        help=(
            "4D slab-slab native-XW kernel memory-layout matrix: configured, all/both, or comma-separated subset "
            "of buffer,tensor. buffer keeps MPI-buffer-contiguous lanes; tensor keeps tensor-contiguous lanes."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-vector4-kernels",
        default="configured",
        help=(
            "4D slab-slab native-XW vectorized-buffer kernel matrix: configured, all/both, or comma-separated "
            "subset of off,on. on processes four X-contiguous MPI-buffer elements per work item."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-tiled-kernels",
        default="configured",
        help=(
            "4D slab-slab native-XW tiled kernel matrix: configured, all/both, or comma-separated "
            "subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-layout-stages",
        default="configured",
        help=(
            "4D slab-slab native-XW output layout matrix: configured, all/both, or comma-separated "
            "subset of direct,stage. stage unpacks to xzwy scratch before the public yzwx layout conversion."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-xw-native-spectral-layouts",
        default="configured",
        help=(
            "4D slab-slab native-XW spectral buffer layout matrix: configured, all/both, or comma-separated "
            "subset of public,native. native keeps the spectral buffer in xzwy layout for forward/backward."
        ),
    )
    parser.add_argument(
        "--fftm-4d-native-xw-direct-layouts",
        default="configured",
        help=(
            "4D native-xzwy same_xw direct receive/send matrix for p2p-waitany: "
            "configured, production, all/both, or comma-separated subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-native-xw-protocols",
        default="configured",
        help=(
            "4D native-xzwy direct-transfer protocol matrix: configured, production, all/both, "
            "or comma-separated subset of single,chunked. Requires direct layout."
        ),
    )
    parser.add_argument(
        "--fftm-4d-native-xw-chunk-mib",
        type=int,
        default=512,
        help="Chunk size in MiB for the 4D native-XW chunked protocol. Default: 512.",
    )
    parser.add_argument(
        "--fftm-4d-native-xw-chunk-windows",
        default="configured",
        help=(
            "4D native-XW per-peer in-flight chunk window matrix: configured, production, matrix, "
            "all, or comma-separated positive integers. 'all' preserves the unbounded baseline."
        ),
    )
    parser.add_argument(
        "--fftm-4d-native-xw-compact-stagings",
        default="configured",
        help=(
            "4D native-XW bounded-window staging matrix: configured, production, all/both, "
            "or comma-separated subset of off,on. Compact mode requires direct layout, chunked transport, "
            "and a positive chunk window."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-native-work-area-aliases",
        default="configured",
        help=(
            "4D native-spectral slab shared-work/stage-1 alias matrix: configured, production, all/both, "
            "or comma-separated subset of off,on. The alias requires native XW direct layout."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-native-wz-communication-layouts",
        default="configured",
        help=(
            "4D slab WZ communication-native layout matrix: configured, production, all/both, or off,on. "
            "Requires native spectral layout and bounded chunked direct XW transport."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-native-wz-plan-concurrencies",
        default="configured",
        help=(
            "4D slab WZ FFT plan/stream concurrency matrix: configured, production, matrix/all, "
            "or comma-separated positive integers. Matrix selects 1,2,4,8."
        ),
    )
    parser.add_argument(
        "--fftm-4d-slab-native-wz-ready-pipelines",
        default="configured",
        help=(
            "4D slab WZ FFT/plane-transfer readiness pipeline matrix: configured, production, "
            "matrix/all/both, or off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-same-zw-peer-paired",
        default="configured",
        help=(
            "4D pencil-pencil same_zw p2p-waitany scheduling matrix: configured, production, all/both, "
            "or comma-separated subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-same-zw-native-layouts",
        default="configured",
        help=(
            "4D pencil-pencil same_zw native message-layout matrix for native spectral runs: "
            "configured, production, all/both, or comma-separated subset of base,native."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-degenerate-xw-slab-paths",
        default="configured",
        help=(
            "4D pencil-pencil native-spectral degenerate-grid path matrix for p1=1,p3=1 grids: "
            "configured, production, all/both, or comma-separated subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-degenerate-local-transposes",
        default="configured",
        help=(
            "4D pencil-pencil native-spectral degenerate-grid local transpose/alias matrix for p1=1,p3=1 "
            "grids: configured, production, all/both, or comma-separated subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-degenerate-same-xw-native",
        default="configured",
        help=(
            "4D pencil-pencil native-spectral degenerate-grid same_xw native p2p scheduling matrix for "
            "p1=1,p3=1 grids: configured, production, all/both, or comma-separated subset of off,on."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-degenerate-wz-sliced-z-fft",
        default="configured",
        help=(
            "4D pencil-pencil native-spectral degenerate-grid WZ sliced-Z FFT matrix for p1=1,p3=1 grids: "
            "configured, production, all/both, or comma-separated subset of off,on."
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
            "Pencil-pencil 3D process-grid orientations to run: configured, both, production, default, or reversed. "
            "configured passes no --grid argument so an autotune config can choose the grid. "
            "production uses known safe choices, currently 7G opt0/auto -> 7x1, "
            "8G opt0/auto -> 4x2, and 8G opt1 -> 2x4. "
            "Default: both, so e.g. 8 GPUs runs both 2x4 and 4x2."
        ),
    )
    parser.add_argument(
        "--fftm-4d-pencil-grid-orientations",
        default="configured",
        help=(
            "4D pencil-pencil process-grid orientations to run: configured, production, default, "
            "x-heavy, y-heavy, z-heavy, axis, or all. configured passes no --grid argument. "
            "For 7 GPUs, all expands to 1x1x7, 1x7x1, and 7x1x1."
        ),
    )
    parser.add_argument(
        "--extra-sizes-3d",
        default="",
        help="Extra 3D benchmark side lengths to run in addition to fitted sizes, e.g. 540,686,729,900.",
    )
    parser.add_argument(
        "--extra-sizes-3d-by-gpu",
        default="",
        help=(
            "Per-GPU-count 3D benchmark side lengths appended to fitted/global sizes, "
            "e.g. '1:1050;2:1344;3:1536' or '1:540,729;2:686,900'."
        ),
    )
    parser.add_argument(
        "--fixed-scaling-sizes-3d",
        default="",
        help=(
            "Comma-separated 3D side lengths appended to every FFTM benchmark GPU count. "
            "Use this for fixed-problem strong-scaling/acceleration runs."
        ),
    )
    parser.add_argument(
        "--fixed-scaling-sizes-4d",
        default="",
        help=(
            "Comma-separated 4D side lengths appended to every FFTM benchmark GPU count. "
            "Use this for fixed-problem strong-scaling/acceleration runs."
        ),
    )
    parser.add_argument(
        "--epsilon",
        dest="validation_epsilon",
        default="1.0e-11",
        help="Validation epsilon passed to FFTS tests and benchmark binaries. Default: 1.0e-11",
    )
    parser.add_argument(
        "--threshold",
        dest="validation_threshold",
        default="1.0e-11",
        help="Validation threshold passed to FFTM versioned tests. Default: 1.0e-11",
    )
    parser.add_argument(
        "--probe-max-runs",
        type=int,
        default=8,
        help="Maximum probe executions per memory family. Default: 8",
    )
    parser.add_argument(
        "--probe-safety-factor",
        type=float,
        default=0.92,
        help="Safety factor applied to memory-ratio-based size growth. Default: 0.92",
    )
    parser.add_argument(
        "--initial-size-3d",
        type=int,
        default=64,
        help="Initial equal-side size used for 3D probing. Default: 64",
    )
    parser.add_argument(
        "--initial-size-4d",
        type=int,
        default=24,
        help="Initial equal-side size used for 4D probing. Default: 24",
    )
    parser.add_argument(
        "--min-size-3d",
        type=int,
        default=16,
        help="Minimum 3D side length used by the probe search. Default: 16",
    )
    parser.add_argument(
        "--min-size-4d",
        type=int,
        default=8,
        help="Minimum 4D side length used by the probe search. Default: 8",
    )
    parser.add_argument(
        "--size-step-3d",
        type=int,
        default=8,
        help="Side-length rounding step for 3D probes. Default: 8",
    )
    parser.add_argument(
        "--size-step-4d",
        type=int,
        default=2,
        help="Side-length rounding step for 4D probes. Default: 2",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=7200,
        help="Per-process timeout in seconds. Default: 7200",
    )
    parser.add_argument(
        "--library-path",
        action="append",
        default=[],
        help=(
            "Additional runtime library directory prepended to LD_LIBRARY_PATH for test binaries. "
            "Can be specified more than once. Common /usr/local/mpi/lib and /usr/local/cuda/lib64 paths are added automatically when present."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.test_warmup < 0:
        parser.error("--warmup must be non-negative")
    runner = LocalPaperBenchmarkRunner(args)
    return runner.run()


if __name__ == "__main__":
    sys.exit(main())
