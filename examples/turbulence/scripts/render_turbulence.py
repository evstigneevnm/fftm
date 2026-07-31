#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_snapshot(raw_path: Path, layout_path: Path):
    metadata = json.loads(layout_path.read_text())
    if metadata.get("format") != "fftm-taylor-green-snapshot-v1":
        raise RuntimeError(f"unsupported snapshot metadata in {layout_path}")
    nx, ny, nz = (int(value) for value in metadata["snapshot_shape"])
    expected = nx * ny * nz
    data = np.memmap(raw_path, dtype="<f4", mode="r")
    if data.size != expected:
        raise RuntimeError(
            f"{raw_path} contains {data.size} float32 values, expected {expected}"
        )
    # FFTM writes x as the contiguous dimension. NumPy's last axis is contiguous.
    return np.asarray(data).reshape((nz, ny, nx)), metadata


def infer_field_kind(path: Path, requested: str):
    if requested != "auto":
        return requested
    name = path.stem.lower()
    if "q_criterion" in name:
        return "q"
    if "vorticity" in name:
        return "vorticity"
    return "scalar"


def field_style(field_kind: str):
    if field_kind == "q":
        return "Q criterion", "viridis"
    if field_kind == "vorticity":
        return "Vorticity magnitude", "inferno"
    return "Scalar", "plasma"


def render_slice(
    data,
    output: Path,
    axis: str,
    index: int | None,
    title: str,
    cmap: str,
    dpi: int,
    width_inches: float,
    height_inches: float,
):
    import matplotlib.pyplot as plt

    axis_to_number = {"x": 2, "y": 1, "z": 0}
    selected_axis = axis_to_number[axis]
    if index is None:
        index = data.shape[selected_axis] // 2
    image = np.take(data, index, axis=selected_axis)

    figure, axes = plt.subplots(
        figsize=(width_inches, height_inches), constrained_layout=True
    )
    plotted = axes.imshow(
        image,
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        extent=(0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi),
    )
    axes.set_xlabel("coordinate")
    axes.set_ylabel("coordinate")
    axes.set_title(title)
    figure.colorbar(plotted, ax=axes, shrink=0.86)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)


def make_grid(data, scalar_name: str):
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        import pyvista as pv
    except ImportError as error:
        raise RuntimeError(
            "3D rendering requires PyVista/VTK; rebuild the FFTM SQSH with "
            "the visualization dependencies or install "
            "examples/turbulence/requirements-visualization.txt"
        ) from error

    spacing = 2.0 * np.pi / float(data.shape[2])
    grid = pv.ImageData(
        dimensions=(data.shape[2], data.shape[1], data.shape[0]),
        spacing=(spacing, spacing, spacing),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data[scalar_name] = np.ascontiguousarray(data).ravel(order="C")
    return pv, grid


def selected_levels(data, field_kind: str, levels, quantiles):
    if levels is not None:
        candidates = levels
    else:
        finite = data[np.isfinite(data)]
        if field_kind == "q":
            finite = finite[finite > 0.0]
        if finite.size == 0:
            raise RuntimeError("the selected field has no finite positive values")
        if quantiles is None:
            quantiles = (0.85, 0.95) if field_kind == "q" else (0.92, 0.97)
        candidates = [float(np.quantile(finite, quantile)) for quantile in quantiles]
    result = sorted({float(value) for value in candidates if np.isfinite(value)})
    if not result:
        raise RuntimeError("no finite isosurface levels were selected")
    return result


def publication_window_size(
    dpi: int, width_inches: float, height_inches: float
) -> tuple[int, int]:
    return (
        max(1, int(round(dpi * width_inches))),
        max(1, int(round(dpi * height_inches))),
    )


def render_scale(dpi: int) -> float:
    return max(1.0, float(dpi) / 250.0)


def scalar_bar_style(dpi: int):
    scale = max(1.0, float(dpi) / 100.0)
    return {
        "title_font_size": int(round(11 * scale)),
        "label_font_size": int(round(9 * scale)),
        "position_x": 0.25,
        "position_y": 0.04,
        "width": 0.5,
        "height": 0.08,
    }


def configure_plotter(
    pv, title: str, window_size: tuple[int, int], dpi: int
):
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.add_text(
        title,
        color="black",
        font_size=int(round(11 * render_scale(dpi))),
    )
    return plotter


def set_png_dpi(output: Path, dpi: int):
    if output.suffix.lower() != ".png":
        return
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "PNG DPI metadata requires Pillow; install "
            "examples/turbulence/requirements-visualization.txt"
        ) from error

    with Image.open(output) as image:
        image.save(output, dpi=(dpi, dpi))


def add_domain_box(pv, plotter, dpi: int):
    domain_box = pv.Box(
        bounds=(
            0.0,
            2.0 * np.pi,
            0.0,
            2.0 * np.pi,
            0.0,
            2.0 * np.pi,
        )
    )
    plotter.add_mesh(
        domain_box,
        color="#303030",
        style="wireframe",
        line_width=max(2.0, 1.5 * render_scale(dpi)),
        lighting=False,
        show_scalar_bar=False,
    )


def configure_3d_camera(
    pv, plotter, show_domain_box: bool, camera_zoom: float, dpi: int
):
    if show_domain_box:
        add_domain_box(pv, plotter, dpi)
    plotter.camera_position = "iso"
    plotter.reset_camera()
    plotter.camera.zoom(camera_zoom)


def render_isosurfaces(
    data,
    output: Path,
    levels,
    quantiles,
    title: str,
    field_kind: str,
    scalar_name: str,
    cmap: str,
    show_domain_box: bool,
    camera_zoom: float,
    window_size: tuple[int, int],
    dpi: int,
):
    pv, grid = make_grid(data, scalar_name)
    levels = selected_levels(data, field_kind, levels, quantiles)
    surfaces = grid.contour(isosurfaces=levels, scalars=scalar_name)

    plotter = configure_plotter(pv, title, window_size, dpi)
    if len(levels) == 1:
        plotter.add_mesh(
            surfaces,
            color="#238b45" if field_kind == "q" else "#d95f0e",
            opacity=0.72,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(
            surfaces,
            scalars=scalar_name,
            cmap=cmap,
            opacity=0.72,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=scalar_bar_style(dpi),
        )
    configure_3d_camera(pv, plotter, show_domain_box, camera_zoom, dpi)
    plotter.show(screenshot=str(output), auto_close=True)
    set_png_dpi(output, dpi)


def render_volume(
    data,
    output: Path,
    title: str,
    field_kind: str,
    scalar_name: str,
    cmap: str,
    show_domain_box: bool,
    camera_zoom: float,
    window_size: tuple[int, int],
    dpi: int,
):
    render_data = np.maximum(data, 0.0) if field_kind == "q" else data
    pv, grid = make_grid(render_data, scalar_name)
    plotter = configure_plotter(pv, title, window_size, dpi)
    plotter.add_volume(
        grid,
        scalars=scalar_name,
        cmap=cmap,
        opacity="sigmoid_6",
        shade=True,
        show_scalar_bar=True,
        scalar_bar_args=scalar_bar_style(dpi),
    )
    configure_3d_camera(pv, plotter, show_domain_box, camera_zoom, dpi)
    plotter.show(screenshot=str(output), auto_close=True)
    set_png_dpi(output, dpi)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render FFTM Taylor-Green vorticity or Q-criterion snapshots."
    )
    parser.add_argument("snapshot", type=Path, help="float32 .raw snapshot")
    parser.add_argument(
        "--layout",
        type=Path,
        help="layout.json (defaults to the snapshot directory)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("slice", "isosurface", "volume"), default="isosurface"
    )
    parser.add_argument(
        "--field-kind",
        choices=("auto", "vorticity", "q", "scalar"),
        default="auto",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--index", type=int)
    parser.add_argument("--level", type=float, action="append", dest="levels")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        help="isosurface quantiles used when --level is omitted",
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=0.8,
        help="3D camera zoom factor; values below one zoom out (default: 0.8)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="raster resolution in dots per inch (default: 600)",
    )
    parser.add_argument(
        "--width-inches",
        type=float,
        default=7.2,
        help="publication width in inches (default: 7.2)",
    )
    parser.add_argument(
        "--height-inches",
        type=float,
        default=5.4,
        help="publication height in inches (default: 5.4)",
    )
    parser.add_argument(
        "--no-domain-box",
        action="store_true",
        help="omit the [0, 2*pi]^3 wireframe box from 3D renders",
    )
    parser.add_argument("--title")
    args = parser.parse_args()
    if args.camera_zoom <= 0.0:
        parser.error("--camera-zoom must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if args.width_inches <= 0.0 or args.height_inches <= 0.0:
        parser.error("--width-inches and --height-inches must be positive")
    if args.mode != "slice" and args.output.suffix.lower() != ".png":
        parser.error("3D isosurface and volume output must use the .png extension")
    return args


def main():
    args = parse_args()
    layout = args.layout or args.snapshot.parent / "layout.json"
    data, _ = load_snapshot(args.snapshot, layout)
    field_kind = infer_field_kind(args.snapshot, args.field_kind)
    scalar_name, cmap = field_style(field_kind)
    title = args.title or f"Taylor-Green {scalar_name}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    window_size = publication_window_size(
        args.dpi, args.width_inches, args.height_inches
    )

    if args.mode == "slice":
        render_slice(
            data,
            args.output,
            args.axis,
            args.index,
            title,
            cmap,
            args.dpi,
            args.width_inches,
            args.height_inches,
        )
    elif args.mode == "volume":
        render_volume(
            data,
            args.output,
            title,
            field_kind,
            scalar_name,
            cmap,
            not args.no_domain_box,
            args.camera_zoom,
            window_size,
            args.dpi,
        )
    else:
        render_isosurfaces(
            data,
            args.output,
            args.levels,
            args.quantiles,
            title,
            field_kind,
            scalar_name,
            cmap,
            not args.no_domain_box,
            args.camera_zoom,
            window_size,
            args.dpi,
        )

    print(
        f"Wrote {args.output} at {window_size[0]}x{window_size[1]} pixels "
        f"({args.width_inches:g}x{args.height_inches:g} inches, {args.dpi} DPI)"
    )


if __name__ == "__main__":
    main()
