#!/usr/bin/env python3

import argparse
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
FFTS_STRATEGIES_4D = ("pencil-direct", "pencil-memcpy", "slab-direct", "slab-memcpy")

SUMMARY_KEY_RE = re.compile(r"([A-Za-z0-9_]+)=((?:\([^)]*\))|[^,]+)")
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

    if ": " not in payload:
        return None

    left, right = payload.split(": ", 1)
    data: Dict[str, Any] = {}
    for part in split_top_level_csv(left) + split_top_level_csv(right):
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        data[key.strip()] = parse_scalar(raw_value)
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

    def size_arity(self) -> int:
        return 3 if self.dim == 3 else 4


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


def add_fftm_specs(specs: List[RunSpec], binaries: Dict[str, Path], max_gpus: int, include_nca: bool) -> None:
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
                                memory_family=f"fftm:3d:small:{num_gpus}:{strategy}",
                            )
                        )

            if benchmark_4d in binaries:
                for strategy in FFTM_STRATEGIES_4D:
                    for mode in FFTM_MODES:
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
                                memory_family=f"fftm:4d:small:{num_gpus}:{strategy}",
                            )
                        )

            for testcase in range(5):
                name_3d = f"test_fftm_v{testcase}_3D{suffix}"
                if name_3d in binaries:
                    family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
                    for strategy in FFTM_STRATEGIES_3D:
                        for mode in FFTM_MODES:
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
                                )
                            )

                name_4d = f"test_fftm_v{testcase}_4D{suffix}"
                if name_4d in binaries:
                    family_kind = "compare" if testcase == 1 else "poisson" if testcase == 4 else "small"
                    for strategy in FFTM_STRATEGIES_4D:
                        for mode in FFTM_MODES:
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
                                    memory_family=f"fftm:4d:{family_kind}:{num_gpus}:{strategy}",
                                )
                            )


def build_measurement_specs(binaries: Dict[str, Path], max_gpus: int, include_nca: bool) -> List[RunSpec]:
    specs: List[RunSpec] = []
    add_ffts_specs(specs, binaries)
    add_fftm_specs(specs, binaries, max_gpus=max_gpus, include_nca=include_nca)
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
        self.specs = build_measurement_specs(
            self.binaries, max_gpus=self.max_gpus, include_nca=not self.args.skip_nca
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
        if spec.supports_directory:
            command.extend(["--directory", str(self.cpp_csv_dir)])
        if spec.suite == "fftm" and spec.case_name != "benchmark":
            command.extend(["--threshold", str(self.args.validation_threshold)])
        else:
            command.extend(["--epsilon", str(self.args.validation_epsilon)])
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
        slug = slugify(
            f"{self.run_index:05d}_{phase}_{spec.suite}_{spec.case_name}_{spec.dim}d_"
            f"g{spec.num_gpus}_{spec.transport}_{spec.strategy or 'default'}_{spec.mode or 'none'}_"
            f"{'x'.join(str(s) for s in sizes)}"
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
            record = self.execute_run(spec, self.size_tuple(spec.dim, int(chosen_n)), self.args.measure_times, PHASE_MEASURE)
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
