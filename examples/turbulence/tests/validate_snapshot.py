#!/usr/bin/env python3

import argparse
import array
import json
import math
from pathlib import Path


def load_field(path: Path, size: int):
    values = array.array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, size**3)
    if values.itemsize != 4:
        raise RuntimeError("snapshot validator requires four-byte float storage")
    import sys

    if sys.byteorder != "little":
        values.byteswap()
    expected = size**3
    if len(values) != expected:
        raise RuntimeError(f"{path}: expected {expected} values, found {len(values)}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    metadata = json.loads((args.output / "layout.json").read_text())
    size = int(metadata["snapshot_shape"][0])
    omega_z = load_field(args.output / "omega_z_s00000000.raw", size)
    magnitude = load_field(
        args.output / "vorticity_magnitude_s00000000.raw", size
    )
    q_criterion = load_field(args.output / "q_criterion_s00000000.raw", size)
    omega_error = 0.0
    magnitude_error = 0.0
    q_error = 0.0
    for z_index in range(size):
        z = 2.0 * math.pi * z_index / size
        for y_index in range(size):
            y = 2.0 * math.pi * y_index / size
            for x_index in range(size):
                x = 2.0 * math.pi * x_index / size
                offset = x_index + size * (y_index + size * z_index)
                expected_omega_z = 2.0 * math.sin(x) * math.sin(y) * math.cos(z)
                expected_magnitude = math.sqrt(
                    math.cos(x) ** 2 * math.sin(y) ** 2 * math.sin(z) ** 2
                    + math.sin(x) ** 2 * math.cos(y) ** 2 * math.sin(z) ** 2
                    + 4.0 * math.sin(x) ** 2 * math.sin(y) ** 2 * math.cos(z) ** 2
                )
                expected_q = math.cos(z) ** 2 * (
                    math.sin(x) ** 2 * math.sin(y) ** 2
                    - math.cos(x) ** 2 * math.cos(y) ** 2
                )
                omega_error = max(
                    omega_error, abs(float(omega_z[offset]) - expected_omega_z)
                )
                magnitude_error = max(
                    magnitude_error,
                    abs(float(magnitude[offset]) - expected_magnitude),
                )
                q_error = max(
                    q_error, abs(float(q_criterion[offset]) - expected_q)
                )
    tolerance = 2.0e-6
    if omega_error > tolerance or magnitude_error > tolerance or q_error > tolerance:
        raise RuntimeError(
            f"snapshot validation failed: omega_z={omega_error:.6e}, "
            f"magnitude={magnitude_error:.6e}, q={q_error:.6e}"
        )
    print(
        f"snapshot validation passed: omega_z={omega_error:.6e}, "
        f"magnitude={magnitude_error:.6e}, q={q_error:.6e}"
    )


if __name__ == "__main__":
    main()
