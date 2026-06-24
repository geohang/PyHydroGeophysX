"""Seismic -> 3D subsurface model pipeline for the desktop workbench.

Takes one or more 2D seismic velocity sections (each an inverted velocity model on
a pyGIMLi mesh, positioned in map coordinates) and builds a 3D subsurface model:
a kriged/interpolated velocity volume and a bedrock-interface surface across the
survey area. It is a thin, parameterized re-use of
``PyHydroGeophysX.Geophy_modular.seismic_processor.extract_velocity_structure``
(2D velocity -> interface) and ``PyHydroGeophysX.core.kriging_3d`` (3D structured
grid + optional ordinary kriging). It is deliberately Qt-free so it can run inside
a worker thread (or be unit-tested) without a QApplication.

Two layers:

* ``build_seismic3d_config`` -- numpy only. Serializes the line list + grid
  settings so a configuration can be exported without any heavy backend.
* ``build_3d_model`` -- the real run. It imports pygimli (to load the section
  meshes) and pyvista (for the 3D volume + VTK export) lazily; if these are
  missing it raises ``BackendUnavailable`` so the caller can fall back to config
  export. 3D velocity interpolation uses scipy ``griddata`` by default and
  ``gstools`` ordinary kriging when that package is installed and requested.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

#: Per-line velocity-model bundle: a folder with these two files.
VELOCITY_FILES = {"mesh": "velmesh.bms", "velocity": "Vinvmodel.npy"}

#: Ordinary kriging solves an N x N system; cap the conditioning set accordingly.
_KRIGING_MAX_POINTS = 2500

#: Supported gstools variogram models (name -> gstools class attribute name).
VARIOGRAM_MODELS = ("Exponential", "Gaussian", "Spherical", "Matern")

#: Default 3D variogram (anisotropic): horizontal range 100 m, vertical 10 m.
DEFAULT_KRIGING = {
    "model": "Exponential",
    "len_scale_x": 100.0, "len_scale_y": 100.0, "len_scale_z": 10.0,
    "var": 0.5, "nugget": 0.0,
}


class BackendUnavailable(RuntimeError):
    """Raised when pygimli / pyvista cannot be used (mirror of hydro_pipeline)."""


def _noop(_msg: str) -> None:
    return None


def _ensure_vtk_matplotlib_shim() -> None:
    """Stub the optional ``vtkmodules.vtkRenderingMatplotlib`` module when absent.

    Some conda-forge VTK builds omit it, yet pyvista imports it unconditionally.
    Mirrors the shim in ``qt_apps.modules.mesh3d_processing``.
    """
    import sys
    import types

    if "vtkmodules.vtkRenderingMatplotlib" in sys.modules:
        return
    try:
        import vtkmodules.vtkRenderingMatplotlib  # noqa: F401 - real module present
    except Exception:  # noqa: BLE001 - missing optional submodule; provide a stub
        sys.modules["vtkmodules.vtkRenderingMatplotlib"] = types.ModuleType(
            "vtkmodules.vtkRenderingMatplotlib")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# File resolution (numpy only)
# ---------------------------------------------------------------------------
def find_velocity_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Return the resolved path for each expected velocity-bundle file (or None)."""
    found: Dict[str, Optional[Path]] = {}
    for key, name in VELOCITY_FILES.items():
        candidate = Path(data_dir) / name
        found[key] = candidate if candidate.exists() else None
    return found


def _load_velocity(path: Path) -> np.ndarray:
    """Load a per-cell velocity vector (1D). 2D (n_cells, n_time) uses column 0."""
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 2:
        arr = arr[:, 0]
    return arr.ravel()


def _output_root(context: Dict[str, Any], params: Dict[str, Any]) -> Path:
    base = params.get("output_dir") or context.get("output_dir") or "."
    return io_utils.ensure_dir(Path(base) / "qt_seismic3d")


def _z_cells(depth: float, n_layers: int) -> np.ndarray:
    """Uniform layer-thickness vector for ``create_3d_structured_grid``."""
    n_layers = max(1, int(n_layers))
    return np.r_[0.0, np.full(n_layers, float(depth) / n_layers)]


# ---------------------------------------------------------------------------
# Per-line structure extraction (requires pygimli)
# ---------------------------------------------------------------------------
def _map_to_3d(local_x: np.ndarray, x_min: float, x_max: float,
               x0: float, y0: float, x1: float, y1: float) -> Tuple[np.ndarray, np.ndarray]:
    """Map local profile distance to map (x, y) along the line endpoints."""
    span = float(x_max - x_min) or 1.0
    t = (np.asarray(local_x, dtype=float) - x_min) / span
    return x0 + t * (x1 - x0), y0 + t * (y1 - y0)


def extract_line_structure(
    line: Dict[str, Any],
    threshold: float,
    interval: float,
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Load one velocity section and extract its surface, interface and velocity points.

    Returns dict with ``surface_pts`` (n,3), ``interface_pts`` (m,3), ``vel_pts``
    (k,4) [x,y,z,vel] mapped into map coordinates, and the raw 2D interface.
    """
    import pygimli as pg
    from PyHydroGeophysX.Geophy_modular.seismic_processor import extract_velocity_structure

    mesh = pg.load(str(line["mesh"]))
    velocity = _load_velocity(Path(line["velocity"]))
    if velocity.size != mesh.cellCount():
        raise ValueError(
            f"Velocity length ({velocity.size}) != mesh cell count "
            f"({mesh.cellCount()}) for line {line.get('name', '?')}.")

    x0, y0 = float(line["x0"]), float(line["y0"])
    x1, y1 = float(line["x1"]), float(line["y1"])

    # Bedrock interface from the velocity threshold (smoothed 500-pt curve).
    x_dense, z_dense, info = extract_velocity_structure(mesh, velocity, threshold, interval)
    x_min, x_max = float(info["min_x"]), float(info["max_x"])
    ix, iy = _map_to_3d(x_dense, x_min, x_max, x0, y0, x1, y1)
    interface_pts = np.column_stack([ix, iy, np.asarray(z_dense, dtype=float)])

    # Surface (top) elevation: max cell-center z per horizontal bin.
    cc = np.asarray(mesh.cellCenters(), dtype=float)
    cx, cz = cc[:, 0], cc[:, 1]
    bins = np.arange(x_min, x_max + interval, interval)
    sx, sz = [], []
    for i in range(len(bins) - 1):
        sel = (cx >= bins[i]) & (cx < bins[i + 1])
        if np.any(sel):
            sx.append(0.5 * (bins[i] + bins[i + 1]))
            sz.append(float(np.max(cz[sel])))
    sx = np.asarray(sx, dtype=float)
    sz = np.asarray(sz, dtype=float)
    tx, ty = _map_to_3d(sx, x_min, x_max, x0, y0, x1, y1)
    surface_pts = np.column_stack([tx, ty, sz])

    # 3D velocity samples from every cell center mapped into map coordinates.
    vx, vy = _map_to_3d(cx, x_min, x_max, x0, y0, x1, y1)
    vel_pts = np.column_stack([vx, vy, cz, velocity])

    log(f"  line {line.get('name', '?')}: {mesh.cellCount()} cells, "
        f"interface z [{np.min(z_dense):.1f}, {np.max(z_dense):.1f}]")
    return {
        "surface_pts": surface_pts,
        "interface_pts": interface_pts,
        "vel_pts": vel_pts,
        "raw_interface": np.column_stack([x_dense, z_dense]),
        "n_cells": int(mesh.cellCount()),
        "vel_range": [float(np.min(velocity)), float(np.max(velocity))],
    }


def summarize_line(line: Dict[str, Any]) -> Dict[str, Any]:
    """Light per-line summary (cell count, velocity range, x-extent). Needs pygimli."""
    import pygimli as pg

    mesh = pg.load(str(line["mesh"]))
    velocity = _load_velocity(Path(line["velocity"]))
    cc = np.asarray(mesh.cellCenters(), dtype=float)
    return {
        "n_cells": int(mesh.cellCount()),
        "vel_min": float(np.min(velocity)),
        "vel_max": float(np.max(velocity)),
        "x_extent": [float(cc[:, 0].min()), float(cc[:, 0].max())],
        "z_extent": [float(cc[:, 1].min()), float(cc[:, 1].max())],
    }


# ---------------------------------------------------------------------------
# Config export (numpy only -- no pygimli required)
# ---------------------------------------------------------------------------
def build_seismic3d_config(context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a complete, JSON-serializable 3D-model configuration."""
    lines = [{"name": ln.get("name", f"line{i}"),
              "mesh": str(ln.get("mesh", "")), "velocity": str(ln.get("velocity", "")),
              "x0": float(ln.get("x0", 0.0)), "y0": float(ln.get("y0", 0.0)),
              "x1": float(ln.get("x1", 0.0)), "y1": float(ln.get("y1", 0.0))}
             for i, ln in enumerate(params.get("lines", []))]
    return {
        "created_time": _utc_now(),
        "direction": "seismic_to_3d_model",
        "n_lines": len(lines),
        "lines": lines,
        "threshold": float(params.get("threshold", 1200.0)),
        "interval": float(params.get("interval", 4.0)),
        "grid_resolution": int(params.get("grid_resolution", 50)),
        "depth": float(params.get("depth", 50.0)),
        "n_layers": int(params.get("n_layers", 19)),
        "interp_method": str(params.get("interp_method", "griddata")),
        "z_scale": float(params.get("z_scale", 1.0)),
        "max_velocity_points": int(params.get("max_velocity_points", 40000)),
        "kriging": {**DEFAULT_KRIGING, **(params.get("kriging") or {})},
    }


# ---------------------------------------------------------------------------
# 3D velocity interpolation
# ---------------------------------------------------------------------------
def _subsample(vel_pts: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Randomly thin scattered points so the 3D interpolation stays tractable."""
    n = vel_pts.shape[0]
    if max_points <= 0 or n <= max_points:
        return vel_pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(max_points), replace=False)
    return vel_pts[np.sort(idx)]


def _build_variogram(gs: Any, cfg: Dict[str, Any]) -> Any:
    """Construct a gstools 3D variogram model from a config dict."""
    cfg = {**DEFAULT_KRIGING, **(cfg or {})}
    name = str(cfg.get("model", "Exponential"))
    if name not in VARIOGRAM_MODELS:
        name = "Exponential"
    model_cls = getattr(gs, name)
    return model_cls(
        dim=3,
        len_scale=[float(cfg["len_scale_x"]), float(cfg["len_scale_y"]), float(cfg["len_scale_z"])],
        var=float(cfg["var"]),
        nugget=float(cfg["nugget"]),
    )


def _interp_volume(vel_pts: np.ndarray, grid_points: np.ndarray, z_scale: float,
                   method: str, log: LogFn, max_points: int = 40000,
                   kriging: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Interpolate scattered (x,y,z,vel) onto grid points.

    Uses ``gstools`` ordinary kriging when requested and available; otherwise
    scipy ``griddata``. When the input lines are near-coplanar (few parallel
    sections, so an axis has very few distinct values) a 3D linear Delaunay is
    both ill-posed and very slow, so a fast KDTree ``nearest`` interpolation is
    used instead.
    """
    if vel_pts.shape[0] > max_points:
        log(f"  thinning {vel_pts.shape[0]} velocity points to {max_points} for interpolation")
        vel_pts = _subsample(vel_pts, max_points)

    from scipy.interpolate import griddata
    src = vel_pts[:, :3].astype(float)
    vals = vel_pts[:, 3].astype(float)
    dst = np.asarray(grid_points, dtype=float)

    if method == "kriging":
        try:
            from PyHydroGeophysX.core.kriging_3d import GSTOOLS_AVAILABLE
            if GSTOOLS_AVAILABLE:
                import gstools as gs
                # Ordinary kriging solves an N x N system, so the conditioning set
                # must stay small (memory/time are O(N^2)/O(N^3)).
                k_src, k_vals = src, vals
                if k_src.shape[0] > _KRIGING_MAX_POINTS:
                    kp = _subsample(np.column_stack([k_src, k_vals]), _KRIGING_MAX_POINTS)
                    k_src, k_vals = kp[:, :3], kp[:, 3]
                model = _build_variogram(gs, kriging)
                kcfg = {**DEFAULT_KRIGING, **(kriging or {})}
                log(f"  3D ordinary kriging (gstools, {kcfg['model']}, "
                    f"len_scale=[{kcfg['len_scale_x']:.0f}, {kcfg['len_scale_y']:.0f}, "
                    f"{kcfg['len_scale_z']:.0f}], var={kcfg['var']}, nugget={kcfg['nugget']}) "
                    f"on {k_src.shape[0]} points…")
                krig = gs.krige.Ordinary(model, k_src.T, k_vals)
                field, _ = krig(dst.T)
                return np.asarray(field, dtype=float)
            log("  gstools unavailable; falling back to griddata.")
        except Exception as exc:  # noqa: BLE001 - fall back to griddata
            log(f"  kriging failed ({exc}); falling back to griddata.")

    unique_per_axis = [int(np.unique(np.round(src[:, k], 3)).size) for k in range(3)]
    if min(unique_per_axis) < 4:
        log(f"  near-coplanar lines (unique/axis={unique_per_axis}); nearest-neighbour interpolation")
        return np.asarray(griddata(src, vals, dst, method="nearest"), dtype=float)

    log("  3D velocity interpolation (scipy griddata, linear)…")
    s = src.copy(); d = dst.copy()
    s[:, 2] *= z_scale
    d[:, 2] *= z_scale
    field = griddata(s, vals, d, method="linear")
    nan = ~np.isfinite(field)
    if np.any(nan):
        field[nan] = griddata(s, vals, d[nan], method="nearest")
    return np.asarray(field, dtype=float)


# ---------------------------------------------------------------------------
# Full 3D model build (requires pygimli + pyvista)
# ---------------------------------------------------------------------------
def build_3d_model(
    context: Dict[str, Any],
    params: Dict[str, Any],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Build the 3D subsurface model from the configured seismic lines.

    Raises ``BackendUnavailable`` if pygimli or pyvista cannot be imported, and
    propagates other exceptions so the caller can fall back to config export.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        import pyvista as pv  # noqa: F401 - needed for the structured grid + VTK
        from PyHydroGeophysX.core.kriging_3d import create_3d_structured_grid
    except Exception as exc:  # ImportError or init failure
        raise BackendUnavailable(str(exc))

    lines = params.get("lines", [])
    if not lines:
        raise ValueError("No seismic lines configured.")
    threshold = float(params.get("threshold", 1200.0))
    interval = float(params.get("interval", 4.0))
    grid_res = int(params.get("grid_resolution", 50))
    depth = float(params.get("depth", 50.0))
    n_layers = int(params.get("n_layers", 19))
    method = str(params.get("interp_method", "griddata"))
    z_scale = float(params.get("z_scale", 1.0))
    max_points = int(params.get("max_velocity_points", 40000))
    kriging_cfg = params.get("kriging") or {}

    surface_all, interface_all, vel_all = [], [], []
    line_traces = []
    for line in lines:
        res = extract_line_structure(line, threshold, interval, log=log)
        surface_all.append(res["surface_pts"])
        interface_all.append(res["interface_pts"])
        vel_all.append(res["vel_pts"])
        line_traces.append((float(line["x0"]), float(line["y0"]),
                            float(line["x1"]), float(line["y1"])))
    surface_pts = np.vstack(surface_all)
    interface_pts = np.vstack(interface_all)
    vel_pts = np.vstack(vel_all)
    log(f"Collected {len(surface_pts)} surface, {len(interface_pts)} interface, "
        f"{len(vel_pts)} velocity points from {len(lines)} line(s).")

    # 3D structured grid frame from the surface topography (scipy griddata-based).
    grid = create_3d_structured_grid(surface_pts, grid_res, _z_cells(depth, n_layers))
    grid_points = np.asarray(grid.points, dtype=float)

    # Velocity volume.
    velocity_field = _interp_volume(vel_pts, grid_points, z_scale, method, log, max_points,
                                    kriging=kriging_cfg)
    grid["Velocity"] = velocity_field

    # Map-view surfaces (top + bedrock interface) on a regular xy grid.
    x = np.linspace(surface_pts[:, 0].min(), surface_pts[:, 0].max(), grid_res)
    y = np.linspace(surface_pts[:, 1].min(), surface_pts[:, 1].max(), grid_res)
    xx, yy = np.meshgrid(x, y)
    from scipy.interpolate import griddata
    top_surface = griddata(surface_pts[:, :2], surface_pts[:, 2], (xx, yy), method="nearest")
    iface_surface = griddata(interface_pts[:, :2], interface_pts[:, 2], (xx, yy), method="linear")
    nan = ~np.isfinite(iface_surface)
    if np.any(nan):
        iface_surface[nan] = griddata(interface_pts[:, :2], interface_pts[:, 2],
                                      (xx[nan], yy[nan]), method="nearest")
    bedrock_depth = top_surface - iface_surface  # positive = depth to bedrock

    out_dir = _output_root(context, params)
    figure_paths: List[str] = []
    data_paths: List[str] = []

    # 1. Bedrock depth map (always works).
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    pcm = ax.contourf(xx, yy, bedrock_depth, levels=20, cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Depth to bedrock (m)")
    for (lx0, ly0, lx1, ly1) in line_traces:
        ax.plot([lx0, lx1], [ly0, ly1], "w-", lw=1.5)
        ax.plot([lx0, lx1], [ly0, ly1], "k--", lw=0.8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title("Depth to bedrock interface")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    p = out_dir / "bedrock_depth_map.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    figure_paths.append(str(p))

    # 2. 3D structure surfaces (matplotlib mplot3d -- always works).
    try:
        fig = plt.figure(figsize=(8.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(xx, yy, top_surface, cmap="gist_earth", alpha=0.6, linewidth=0)
        ax.plot_surface(xx, yy, iface_surface, cmap="copper", alpha=0.9, linewidth=0)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Elevation (m)")
        ax.set_title("Top surface and bedrock interface")
        fig.tight_layout()
        p = out_dir / "structure_3d.png"
        fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
        figure_paths.append(str(p))
    except Exception as exc:  # noqa: BLE001 - 3D surface is best-effort
        log(f"  3D surface render skipped: {exc}")

    # 3. Exports: VTK volume + numpy arrays.
    vtk_path = out_dir / "seismic_3d_model.vtk"
    try:
        grid.save(str(vtk_path))
        data_paths.append(str(vtk_path))
    except Exception as exc:  # noqa: BLE001
        log(f"  VTK export skipped: {exc}")
        vtk_path = None
    for name, arr in (("top_surface", top_surface), ("bedrock_interface", iface_surface),
                      ("bedrock_depth", bedrock_depth), ("velocity_volume", velocity_field)):
        ap = out_dir / f"{name}.npy"
        np.save(ap, arr); data_paths.append(str(ap))

    # 4. Optional pyvista off-screen render (best-effort; needs a GL context).
    try:
        _ensure_vtk_matplotlib_shim()
        import pyvista as pv
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(off_screen=True, window_size=(900, 650))
        plotter.add_mesh(grid.outline(), color="grey")
        plotter.add_mesh(grid, scalars="Velocity", cmap="turbo", opacity=0.35,
                         scalar_bar_args={"title": "Velocity (m/s)"})
        isurf = pv.StructuredGrid(xx, yy, iface_surface)
        plotter.add_mesh(isurf, color="saddlebrown")
        plotter.add_axes(); plotter.view_isometric()
        shot = out_dir / "volume_3d.png"
        plotter.screenshot(str(shot)); plotter.close()
        figure_paths.append(str(shot))
    except Exception as exc:  # noqa: BLE001 - 3D GL render is best-effort
        log(f"  pyvista 3D render skipped: {exc}")

    config = build_seismic3d_config(context, params)
    config_path = out_dir / "seismic3d_config.json"
    io_utils.write_json(config_path, config)

    return {
        "status": "ok",
        "direction": "seismic_to_3d_model",
        "n_lines": len(lines),
        "grid_resolution": grid_res,
        "n_grid_points": int(grid_points.shape[0]),
        "velocity_range": [float(np.nanmin(velocity_field)), float(np.nanmax(velocity_field))],
        "bedrock_depth_range": [float(np.nanmin(bedrock_depth)), float(np.nanmax(bedrock_depth))],
        "interp_method": method,
        "figure_paths": figure_paths,
        "data_paths": data_paths,
        "vtk_path": str(vtk_path) if vtk_path else "",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
    }
