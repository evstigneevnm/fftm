#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path


DEFAULT_REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "fftm-taylor-green-reference-manifest-v1":
        raise RuntimeError(f"unsupported reference manifest: {path}")
    return manifest


def verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"reference checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def download(entry, reference_dir: Path, timeout: float, force: bool) -> None:
    target = reference_dir / entry["file"]
    expected = entry["sha256"]
    if target.is_file() and not force:
        verify(target, expected)
        print(f"verified existing {entry['id']}: {target}")
        return

    source_url = entry.get("source_url", "")
    if not source_url.startswith("https://"):
        raise RuntimeError(
            f"dataset '{entry['id']}' has no supported HTTPS source URL"
        )

    reference_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".download", dir=reference_dir
    )
    temporary = Path(temporary_name)
    try:
        request = urllib.request.Request(
            source_url, headers={"User-Agent": "FFTM-reference-fetch/1"}
        )
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            output.flush()
            os.fsync(output.fileno())
        verify(temporary, expected)
        os.replace(temporary, target)
        print(f"downloaded and verified {entry['id']}: {target}")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch optional Taylor-Green reference data and verify SHA-256."
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Manifest path; defaults to REFERENCE_DIR/manifest.json.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help=(
            "Dataset id to fetch; repeat for multiple datasets. "
            "Defaults to all external datasets."
        ),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    reference_dir = args.reference_dir.resolve()
    manifest_path = (args.manifest or reference_dir / "manifest.json").resolve()
    manifest = load_manifest(manifest_path)
    entries = manifest["datasets"]
    by_id = {entry["id"]: entry for entry in entries}

    if args.list_only:
        for entry in entries:
            target = reference_dir / entry["file"]
            print(
                f"{entry['id']} bundled={int(bool(entry.get('bundled', True)))} "
                f"present={int(target.is_file())} file={target}"
            )
        return 0

    requested = set(args.dataset)
    unknown = requested - set(by_id)
    if unknown:
        raise RuntimeError(f"unknown dataset ids: {', '.join(sorted(unknown))}")

    selected = (
        [by_id[dataset_id] for dataset_id in args.dataset]
        if args.dataset
        else [entry for entry in entries if not entry.get("bundled", True)]
    )
    if not selected:
        raise RuntimeError("no external reference datasets selected")

    for entry in selected:
        download(entry, reference_dir, args.timeout, args.force)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
