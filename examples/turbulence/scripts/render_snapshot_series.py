#!/usr/bin/env python3

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_turbulence import (
    field_style,
    load_snapshot,
    publication_window_size,
    render_isosurfaces,
    selected_levels,
)


FIELD_KINDS = {
    "vorticity_magnitude": "vorticity",
    "q_criterion": "q",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a reproducible Taylor-Green vorticity/Q snapshot series."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="directory containing snapshots.csv, layout.json, and raw fields",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fields",
        default="vorticity_magnitude,q_criterion",
        help="comma-separated manifest fields",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="render every Nth frame of each field (default: 1)",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--width-inches", type=float, default=7.2)
    parser.add_argument("--height-inches", type=float, default=5.4)
    parser.add_argument("--camera-zoom", type=float, default=0.8)
    parser.add_argument(
        "--vorticity-quantiles",
        type=float,
        nargs="+",
        default=(0.92, 0.97),
    )
    parser.add_argument(
        "--q-quantiles",
        type=float,
        nargs="+",
        default=(0.85, 0.95),
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=4,
        help="zero disables per-field contact sheets",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.width_inches <= 0.0 or args.height_inches <= 0.0:
        parser.error("publication dimensions must be positive")
    if args.camera_zoom <= 0.0:
        parser.error("--camera-zoom must be positive")
    if args.contact_sheet_columns < 0:
        parser.error("--contact-sheet-columns cannot be negative")
    for option, values in (
        ("--vorticity-quantiles", args.vorticity_quantiles),
        ("--q-quantiles", args.q_quantiles),
    ):
        if len(values) != 2:
            parser.error(f"{option} requires exactly two values")
        if any(value <= 0.0 or value >= 1.0 for value in values):
            parser.error(f"{option} values must lie strictly between zero and one")
    return args


def read_manifest(input_directory: Path, requested_fields, stride: int):
    manifest = input_directory / "snapshots.csv"
    if not manifest.is_file():
        raise RuntimeError(f"missing snapshot manifest: {manifest}")
    if not (input_directory / "layout.json").is_file():
        raise RuntimeError(f"missing snapshot layout: {input_directory / 'layout.json'}")

    by_field = {field: [] for field in requested_fields}
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "time", "field", "file"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"{manifest} must contain columns {sorted(required)}"
            )
        for row in reader:
            field = row["field"]
            if field not in by_field:
                continue
            source = input_directory / row["file"]
            if not source.is_file():
                raise RuntimeError(f"missing snapshot payload: {source}")
            by_field[field].append(
                {
                    "step": int(row["step"]),
                    "time": float(row["time"]),
                    "field": field,
                    "file": row["file"],
                    "source": source,
                }
            )

    selected = []
    for field in requested_fields:
        records = by_field[field]
        if not records:
            raise RuntimeError(f"manifest contains no '{field}' snapshots")
        records.sort(key=lambda record: (record["time"], record["step"]))
        selected.extend(records[::stride])
    selected.sort(key=lambda record: (record["field"], record["time"]))
    return selected


def output_subdirectory(field: str):
    return "vorticity" if field == "vorticity_magnitude" else "q"


def make_contact_sheet(records, output: Path, columns: int, dpi: int):
    if not records:
        return
    columns = min(columns, len(records))
    rows = int(math.ceil(len(records) / columns))
    cell_width = 720
    cell_height = 570
    label_height = 36
    sheet = Image.new(
        "RGB",
        (columns * cell_width, rows * (cell_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)

    for index, record in enumerate(records):
        column = index % columns
        row = index // columns
        x0 = column * cell_width
        y0 = row * (cell_height + label_height)
        with Image.open(record["output"]) as image:
            rendered = image.convert("RGB")
            rendered.thumbnail((cell_width, cell_height))
            x = x0 + (cell_width - rendered.width) // 2
            y = y0 + (cell_height - rendered.height) // 2
            sheet.paste(rendered, (x, y))
        draw.text(
            (x0 + 12, y0 + cell_height + 8),
            f"t = {record['time']:.2f}",
            fill="black",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, dpi=(dpi, dpi))


def main():
    args = parse_args()
    requested_fields = [
        field.strip() for field in args.fields.split(",") if field.strip()
    ]
    if not requested_fields:
        raise RuntimeError("--fields selected no fields")
    unsupported = sorted(set(requested_fields) - set(FIELD_KINDS))
    if unsupported:
        raise RuntimeError(f"unsupported fields: {', '.join(unsupported)}")

    input_directory = args.input.resolve()
    output_directory = args.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    layout = input_directory / "layout.json"
    records = read_manifest(input_directory, requested_fields, args.stride)
    window_size = publication_window_size(
        args.dpi, args.width_inches, args.height_inches
    )

    rendered_records = []
    for index, record in enumerate(records, start=1):
        field = record["field"]
        field_kind = FIELD_KINDS[field]
        scalar_name, cmap = field_style(field_kind)
        quantiles = (
            args.vorticity_quantiles
            if field_kind == "vorticity"
            else args.q_quantiles
        )
        output = (
            output_directory
            / output_subdirectory(field)
            / f"{Path(record['file']).stem}.png"
        )

        data, _ = load_snapshot(record["source"], layout)
        levels = selected_levels(data, field_kind, None, quantiles)
        if args.overwrite or not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            render_isosurfaces(
                data,
                output,
                levels,
                None,
                f"Taylor-Green {scalar_name}, t = {record['time']:.2f}",
                field_kind,
                scalar_name,
                cmap,
                True,
                args.camera_zoom,
                window_size,
                args.dpi,
            )

        rendered = {
            "step": record["step"],
            "time": record["time"],
            "field": field,
            "source": record["file"],
            "output": output,
            "levels": levels,
            "minimum": float(np.nanmin(data)),
            "maximum": float(np.nanmax(data)),
        }
        rendered_records.append(rendered)
        print(
            f"RENDER_SERIES_FRAME index={index}/{len(records)} "
            f"field={field} time={record['time']:.17g} output={output}"
        )
        del data
        gc.collect()

    manifest = output_directory / "renders.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "step",
                "time",
                "field",
                "source",
                "output",
                "level_0",
                "level_1",
                "minimum",
                "maximum",
            ]
        )
        for record in rendered_records:
            levels = list(record["levels"]) + ["", ""]
            writer.writerow(
                [
                    record["step"],
                    f"{record['time']:.17g}",
                    record["field"],
                    record["source"],
                    record["output"].relative_to(output_directory),
                    levels[0],
                    levels[1],
                    record["minimum"],
                    record["maximum"],
                ]
            )

    contact_sheets = []
    if args.contact_sheet_columns:
        for field in requested_fields:
            field_records = [
                record for record in rendered_records if record["field"] == field
            ]
            output = output_directory / f"{output_subdirectory(field)}_contact_sheet.png"
            make_contact_sheet(
                field_records,
                output,
                args.contact_sheet_columns,
                args.dpi,
            )
            contact_sheets.append(str(output))

    metadata = {
        "format": "fftm-taylor-green-render-series-v1",
        "input_directory": str(input_directory),
        "fields": requested_fields,
        "stride": args.stride,
        "frames": len(rendered_records),
        "dpi": args.dpi,
        "pixel_size": list(window_size),
        "camera_zoom": args.camera_zoom,
        "domain_box": True,
        "vorticity_quantiles": list(args.vorticity_quantiles),
        "q_quantiles": list(args.q_quantiles),
        "manifest": str(manifest),
        "contact_sheets": contact_sheets,
    }
    (output_directory / "renders.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(
        f"RENDER_SERIES_RESULT frames={len(rendered_records)} "
        f"fields={','.join(requested_fields)} output={output_directory}"
    )


if __name__ == "__main__":
    main()
