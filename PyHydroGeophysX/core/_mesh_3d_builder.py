"""Qt-free 3D ERT mesh builder shared by the desktop studio.

Turns a plain ``config`` dict (the same keys the Streamlit 3D mesh builder uses)
into electrode positions and a PyGIMLi 3D mesh:

* surface grids with topography -> a PyGIMLi prism mesh (``Mesh3DCreator``);
* box and borehole layouts -> a Gmsh-free PyGIMLi structured grid.

Nothing here imports Qt, so it can run inside a worker thread and be unit-tested
without a display. ``generate_mesh`` is the single high-level entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Topography
# ---------------------------------------------------------------------------
def topography_function(config: Dict[str, Any]) -> Callable[[float, float], float]:
    """Build a ``z = f(x, y)`` topography callable from the config."""
    topo_type = config.get("topography_type", "Flat")
    if topo_type == "Flat":
        z_flat = float(config.get("z_flat", 0.0))
        return lambda x, y: float(z_flat)

    if topo_type == "Linear tilt":
        z_base = float(config.get("z_base", 0.0))
        tilt_x = float(config.get("tilt_x", 0.0))
        tilt_y = float(config.get("tilt_y", 0.0))
        return lambda x, y: z_base + tilt_x * float(x) + tilt_y * float(y)

    if topo_type == "Gaussian hill":
        z_base = float(config.get("hill_base", 0.0))
        amp = float(config.get("hill_amp", 5.0))
        sigma = max(float(config.get("hill_sigma", 10.0)), 1.0e-9)
        cx = float(config.get("hill_cx", 0.0))
        cy = float(config.get("hill_cy", 0.0))
        return lambda x, y: z_base + amp * np.exp(
            -((float(x) - cx) ** 2 + (float(y) - cy) ** 2) / (2.0 * sigma**2)
        )

    if topo_type.startswith("From file"):
        return _topography_from_points(config.get("topography_points"))

    expr = str(config.get("topography_expr", "0.0"))
    allowed = {
        "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
        "sqrt": np.sqrt, "abs": abs, "pi": np.pi,
    }

    def _custom_topography(x, y):
        try:
            return float(eval(expr, {"__builtins__": {}}, {**allowed, "x": x, "y": y}))  # noqa: S307
        except Exception:
            return 0.0

    return _custom_topography


def _topography_from_points(points: Any) -> Callable[[float, float], float]:
    """Interpolate ``z = f(x, y)`` from loaded ``(x, y, z)`` points.

    Linear interpolation inside the data hull, nearest-neighbour outside it (so
    sensors near the survey edge still get a sensible elevation).
    """
    pts = np.asarray(points, dtype=float) if points is not None else None
    if pts is None or pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return lambda x, y: 0.0
    xy = pts[:, :2]
    z = pts[:, 2]
    if len(pts) < 3:
        z_const = float(np.nanmean(z))
        return lambda x, y: z_const
    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        lin = LinearNDInterpolator(xy, z)
        near = NearestNDInterpolator(xy, z)

        def _topo(x, y):
            val = lin(float(x), float(y))
            return float(val) if np.isfinite(val) else float(near(float(x), float(y)))

        return _topo
    except Exception:  # noqa: BLE001 - scipy missing: nearest-neighbour fallback
        def _topo_nn(x, y):
            d2 = (xy[:, 0] - float(x)) ** 2 + (xy[:, 1] - float(y)) ** 2
            return float(z[int(np.argmin(d2))])

        return _topo_nn


# ---------------------------------------------------------------------------
# Electrodes
# ---------------------------------------------------------------------------
def build_electrodes(config: Dict[str, Any]):
    """Create a ``Mesh3DCreator`` and an electrode DataFrame from the config.

    Returns ``(creator, electrodes_df)`` where the DataFrame has columns
    ``x, y, z, n`` (electrode number).
    """
    import pandas as pd

    from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator

    creator = Mesh3DCreator(
        mesh_directory=str(config.get("output_dir", ".")),
        elec_refinement=float(config["electrode_refinement"]),
        node_refinement=float(config["boundary_refinement"]),
        attractor_distance=float(config["attractor_distance"]),
    )
    array_type = config["array_type"]
    mesh_type = config["mesh_type"]

    if array_type == "Surface grid":
        electrodes = creator.create_surface_electrode_array(
            nx=int(config["nx"]), ny=int(config["ny"]),
            dx=float(config["dx"]), dy=float(config["dy"]),
            x_offset=float(config["x_offset"]), y_offset=float(config["y_offset"]),
            z=0.0,
        )
        if mesh_type == "Surface with topography":
            topo_func = topography_function(config)
            electrodes["z"] = [
                topo_func(x_val, y_val) for x_val, y_val in zip(electrodes["x"], electrodes["y"])
            ]
    elif array_type == "Single borehole":
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        electrodes = creator.create_borehole_electrode_array(
            float(config["bh_x"]), float(config["bh_y"]), z_values,
        )
    elif array_type == "Crosshole":
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        electrodes = creator.create_crosshole_electrode_array(list(config["boreholes"]), z_values)
    else:  # Surface-to-borehole
        surface = creator.create_surface_electrode_array(
            nx=int(config["n_surface_elec"]), ny=1,
            dx=float(config["surface_dx"]), dy=1.0,
            x_offset=float(config["surface_x0"]), y_offset=float(config["surface_y"]),
            z=float(config["surface_z"]),
        )
        z_values = np.linspace(float(config["z_start"]), float(config["z_end"]), int(config["n_bh_elec"]))
        borehole = creator.create_borehole_electrode_array(
            float(config["bh_x"]), float(config["bh_y"]), z_values,
            electrode_start_number=len(surface) + 1,
        )
        electrodes = pd.concat([surface, borehole], ignore_index=True)
        electrodes["n"] = np.arange(1, len(electrodes) + 1, dtype=int)

    return creator, electrodes


# ---------------------------------------------------------------------------
# Structured (Gmsh-free) mesh for box / borehole layouts
# ---------------------------------------------------------------------------
def _axis_with_points(lower: float, upper: float, spacing: float, required_points: Any) -> np.ndarray:
    """Float axis spanning ``[lower, upper]`` that also includes electrode coords."""
    lower, upper = float(lower), float(upper)
    if upper < lower:
        lower, upper = upper, lower
    if np.isclose(lower, upper):
        pad = max(abs(lower) * 0.05, float(spacing), 1.0)
        lower -= pad
        upper += pad
    spacing = max(float(spacing), 1.0e-6)
    intervals = max(1, int(np.ceil((upper - lower) / spacing)))
    base = np.linspace(lower, upper, intervals + 1, dtype=float)
    points = np.asarray(required_points, dtype=float).ravel()
    points = points[np.isfinite(points)]
    points = points[(points >= lower - 1.0e-9) & (points <= upper + 1.0e-9)]
    axis = np.unique(np.round(np.concatenate([base, points, [lower, upper]]), 8)).astype(float)
    axis.sort()
    if axis.size < 2:
        axis = np.asarray([lower, upper], dtype=float)
    return axis


def _structured_bounds(electrodes_df: Any, config: Dict[str, Any]) -> Dict[str, float]:
    """Estimate a structured 3D mesh domain for box and borehole-style surveys."""
    xs = np.asarray(electrodes_df["x"], dtype=float)
    ys = np.asarray(electrodes_df["y"], dtype=float)
    zs = np.asarray(electrodes_df["z"], dtype=float)
    array_type = config.get("array_type", "Surface grid")
    mesh_type = config.get("mesh_type", "Surface with topography")

    if mesh_type == "Box mesh":
        x_min = min(0.0, float(np.nanmin(xs)))
        x_max = max(float(config.get("box_length", 50.0)), float(np.nanmax(xs)))
        y_min = min(0.0, float(np.nanmin(ys)))
        y_max = max(float(config.get("box_width", 30.0)), float(np.nanmax(ys)))
        z_top = max(0.0, float(np.nanmax(zs)))
        z_bottom = min(-float(config.get("box_height", 25.0)), float(np.nanmin(zs)))
    elif array_type != "Surface grid":
        lateral_padding = float(config.get("borehole_lateral_padding", 10.0))
        top_padding = float(config.get("borehole_top_padding", 2.0))
        bottom_padding = float(config.get("borehole_bottom_padding", 5.0))
        x_min = float(np.nanmin(xs)) - lateral_padding
        x_max = float(np.nanmax(xs)) + lateral_padding
        y_min = float(np.nanmin(ys)) - lateral_padding
        y_max = float(np.nanmax(ys)) + lateral_padding
        z_top = max(0.0, float(np.nanmax(zs)) + top_padding)
        z_bottom = float(np.nanmin(zs)) - bottom_padding
    else:
        spacing = max(float(config.get("dx", 5.0)), float(config.get("dy", 5.0)), 1.0)
        extension = max(float(config.get("boundary_extension", 1.4)) - 1.0, 0.1)
        x_pad = max(spacing, (float(np.nanmax(xs)) - float(np.nanmin(xs))) * extension * 0.5)
        y_pad = max(spacing, (float(np.nanmax(ys)) - float(np.nanmin(ys))) * extension * 0.5)
        x_min = float(np.nanmin(xs)) - x_pad
        x_max = float(np.nanmax(xs)) + x_pad
        y_min = float(np.nanmin(ys)) - y_pad
        y_max = float(np.nanmax(ys)) + y_pad
        z_top = float(np.nanmax(zs))
        z_bottom = float(np.nanmin(zs)) - float(config.get("para_depth", 20.0))

    min_span = max(float(config.get("borehole_horizontal_cell", config.get("boundary_refinement", 2.0))), 1.0)
    if (x_max - x_min) < min_span:
        center = 0.5 * (x_min + x_max)
        x_min, x_max = center - 0.5 * min_span, center + 0.5 * min_span
    if (y_max - y_min) < min_span:
        center = 0.5 * (y_min + y_max)
        y_min, y_max = center - 0.5 * min_span, center + 0.5 * min_span
    if not z_bottom < z_top:
        z_bottom = z_top - float(config.get("para_depth", 20.0))

    return {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "z_bottom": float(z_bottom), "z_top": float(z_top),
    }


def create_structured_mesh(electrodes_df: Any, config: Dict[str, Any]) -> Any:
    """Create a Gmsh-free PyGIMLi structured 3D mesh for box/borehole layouts."""
    import pygimli as pg

    bounds = _structured_bounds(electrodes_df, config)
    if config.get("array_type") == "Surface grid":
        xy_spacing = float(config.get("boundary_refinement", 2.0))
        z_spacing = float(config.get("dz_fine", 0.5))
    else:
        xy_spacing = float(config.get("borehole_horizontal_cell", 2.0))
        z_spacing = float(config.get("borehole_vertical_cell", 1.0))

    x_axis = _axis_with_points(bounds["x_min"], bounds["x_max"], xy_spacing, electrodes_df["x"])
    y_axis = _axis_with_points(bounds["y_min"], bounds["y_max"], xy_spacing, electrodes_df["y"])
    z_axis = _axis_with_points(bounds["z_bottom"], bounds["z_top"], z_spacing, electrodes_df["z"])

    mesh = pg.createGrid(x=x_axis.astype(float), y=y_axis.astype(float), z=z_axis.astype(float), marker=2)
    para_depth = float(config.get("para_depth", abs(bounds["z_top"] - bounds["z_bottom"])))
    for cell in mesh.cells():
        depth = bounds["z_top"] - float(cell.center().z())
        cell.setMarker(1 if depth > para_depth else 2)
    return mesh


# ---------------------------------------------------------------------------
# Summary / export / high-level entry
# ---------------------------------------------------------------------------
def mesh_summary(mesh: Any) -> Dict[str, Any]:
    """Robust mesh summary metrics (cells/nodes/boundaries/dim)."""
    summary: Dict[str, Any] = {}
    for label, method_name in (
        ("Cells", "cellCount"), ("Nodes", "nodeCount"),
        ("Boundaries", "boundaryCount"), ("Dimension", "dim"),
    ):
        try:
            summary[label] = getattr(mesh, method_name)()
        except Exception:  # noqa: BLE001
            continue
    return summary


def save_outputs(
    mesh: Any, electrodes_df: Any, output_dir: Path, mesh_name: str, formats: List[str]
) -> Dict[str, str]:
    """Save requested outputs (BMS / VTK / electrode CSV). Returns {key: path}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    if any("bms" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}.bms"
        from PyHydroGeophysX.core.mesh_serialization import save_mesh_artifact

        path, sidecar = save_mesh_artifact(mesh, path)
        outputs["bms"] = str(path)
        outputs["mesh_structure"] = str(sidecar)
    if any("vtk" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}.vtk"
        from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

        # exportVTK opens its own narrow path, so it fails on the same folders
        # that defeat mesh.save; see via_ascii_path for what that means.
        via_ascii_path(mesh.exportVTK, path, mode="write")
        outputs["vtk"] = str(path)
    if any("csv" in f.lower() or "sensor" in f.lower() or "electrode" in f.lower() for f in formats):
        path = output_dir / f"{mesh_name}_sensors.csv"
        electrodes_df.to_csv(path, index=False)
        outputs["sensors_csv"] = str(path)
    return outputs


def find_gmsh_binary() -> Optional[str]:
    """Locate a usable Gmsh executable.

    The pip ``gmsh`` package ships only the Python API (no console binary), so we
    fall back to the ``gmsh.exe`` bundled with resipy when it is installed.
    """
    import shutil

    found = shutil.which("gmsh")
    if found:
        return found
    try:
        import os

        import resipy

        candidate = os.path.join(os.path.dirname(resipy.__file__), "exe", "gmsh.exe")
        if os.path.exists(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        pass
    return None


def _gmsh_box_mesh(creator: Any, electrodes: Any, config: Dict[str, Any]) -> Any:
    """Refined tetrahedral mesh via Gmsh: a box domain with the sensors embedded.

    High mesh quality with local refinement at the sensors (single region). The
    top is flat, so it suits box / borehole / flat-terrain surveys; strong
    topography is better served by the prism engine.
    """
    binary = find_gmsh_binary()
    if not binary:
        raise RuntimeError(
            "Gmsh executable not found (the pip 'gmsh' package provides only the "
            "Python API). Install Gmsh or resipy, which bundles gmsh.exe.")
    creator.gmsh_path = binary
    xs = np.asarray(electrodes["x"], dtype=float)
    ys = np.asarray(electrodes["y"], dtype=float)
    zs = np.asarray(electrodes["z"], dtype=float)

    if config.get("mesh_type") == "Box mesh":
        length = float(config.get("box_length", 50.0))
        width = float(config.get("box_width", 30.0))
        height = float(config.get("box_height", 25.0))
        origin = (min(0.0, float(xs.min())), min(0.0, float(ys.min())), float(zs.max()) - height)
    else:
        if config.get("array_type") == "Surface grid":
            ext = max(float(config.get("boundary_extension", 1.4)) - 1.0, 0.1)
            lateral = max(float(config.get("boundary_refinement", 2.0)),
                          (float(xs.max()) - float(xs.min())) * ext * 0.5,
                          (float(ys.max()) - float(ys.min())) * ext * 0.5)
        else:
            lateral = float(config.get("borehole_lateral_padding", 10.0))
        depth = float(config.get("para_depth", 20.0)) + float(config.get("borehole_bottom_padding", 5.0))
        origin = (float(xs.min()) - lateral, float(ys.min()) - lateral, float(zs.min()) - depth)
        length = (float(xs.max()) - float(xs.min())) + 2.0 * lateral
        width = (float(ys.max()) - float(ys.min())) + 2.0 * lateral
        height = float(zs.max()) - origin[2]

    return creator.create_box_mesh(length, width, height, electrodes, output_name="gmsh_mesh", origin=origin)


def generate_mesh(config: Dict[str, Any], log: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """Build sensors and a 3D mesh from ``config``.

    Honours ``config['mesh_engine']`` (Auto / Gmsh (tetrahedral) / PyGIMLi prism /
    Structured grid) with a graceful fallback to the structured grid if Gmsh
    fails, and ``config['single_region']`` to collapse the markers to one region.
    Returns ``{"mesh", "electrodes", "generator"}``. Safe to call from a worker.
    """
    say = log or (lambda *_: None)
    say("Building sensor array…")
    creator, electrodes = build_electrodes(config)
    engine = config.get("mesh_engine", "Auto")
    mesh_type = config.get("mesh_type")
    array_type = config.get("array_type")
    surface_topo = mesh_type == "Surface with topography" and array_type == "Surface grid"

    def _prism():
        return creator.create_3d_mesh_with_topography(
            electrode_positions=electrodes,
            topography_func=topography_function(config),
            para_depth=float(config["para_depth"]),
            dz_fine=float(config["dz_fine"]),
            dz_coarse=float(config["dz_coarse"]),
            boundary_extension=float(config["boundary_extension"]),
            use_prism_mesh=True,
        ), "PyGIMLi topography prism"

    try:
        if engine == "Gmsh (tetrahedral)":
            say("Generating Gmsh tetrahedral mesh…")
            mesh, label = _gmsh_box_mesh(creator, electrodes, config), "Gmsh tetrahedral"
        elif engine == "Structured grid":
            say("Creating PyGIMLi structured grid…")
            mesh, label = create_structured_mesh(electrodes, config), "PyGIMLi structured grid"
        elif engine == "PyGIMLi prism" and surface_topo:
            say("Creating PyGIMLi topography prism mesh…")
            mesh, label = _prism()
        elif engine == "Auto" and surface_topo:
            say("Creating PyGIMLi topography prism mesh…")
            mesh, label = _prism()
        else:
            say("Creating PyGIMLi structured grid…")
            mesh, label = create_structured_mesh(electrodes, config), "PyGIMLi structured grid"
    except Exception as exc:  # noqa: BLE001
        if engine == "Gmsh (tetrahedral)":
            say(f"Gmsh failed ({exc}); falling back to structured grid.")
            mesh, label = create_structured_mesh(electrodes, config), "PyGIMLi structured grid (Gmsh fallback)"
        else:
            raise

    if config.get("single_region"):
        for cell in mesh.cells():
            cell.setMarker(2)
        label += " · single region"

    say(f"Mesh ready ({label}).")
    return {"mesh": mesh, "electrodes": electrodes, "generator": label}
