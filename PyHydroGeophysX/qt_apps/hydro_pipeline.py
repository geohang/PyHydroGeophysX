"""Hydro -> multi-geophysics forward pipeline for the desktop workbench.

This module is a thin, parameterized re-use of
``examples/Ex_hydro_to_multigeophys.py``. It is deliberately Qt-free so it can run
inside a worker thread (or be unit-tested) without a QApplication.

Two layers:

* ``extract_profile`` / ``build_survey_config`` -- numpy + scipy only. These work
  even when pygimli is not installed, so "Generate profile" and "Export survey
  config" never depend on the heavy backend.
* ``run_hydro_forward`` -- the real forward run. It imports pygimli and the
  ``hydro_to_*`` wrappers lazily; if anything is missing or fails it raises
  ``BackendUnavailable`` so the caller can fall back to config export.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

HYDRO_FILES = {
    "water_content": "Watercontent.npy",
    "porosity": "Porosity.npy",
    "top": "top.npy",
    "bot": "bot.npy",
}

# SRT velocity parameters are not exposed in the Pass-1 UI; use the same
# three-layer defaults as the reference example.
DEFAULT_VEL_PARAMETERS: Dict[str, Dict[str, float]] = {
    "top": {"bulk_modulus": 30.0, "shear_modulus": 20.0, "mineral_density": 2650, "depth": 1.0},
    "mid": {"bulk_modulus": 50.0, "shear_modulus": 35.0, "mineral_density": 2670, "aspect_ratio": 0.05},
    "bot": {"bulk_modulus": 55.0, "shear_modulus": 50.0, "mineral_density": 2680, "aspect_ratio": 0.03},
}

ALL_METHODS = ["Profile", "ERT", "SRT", "TDEM", "FDEM", "Gravity"]


class BackendUnavailable(RuntimeError):
    """Raised when pygimli / the hydro_to_* wrappers cannot be used."""


def _noop(_msg: str) -> None:
    return None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Helpers copied from Ex_hydro_to_multigeophys.py (not part of the package API)
# ---------------------------------------------------------------------------
def fill_profile_nans(values: Any) -> np.ndarray:
    """Fill NaNs along the profile direction for each layer."""
    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
    x = np.arange(arr.shape[1], dtype=float)
    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = np.isfinite(row)
        if np.any(valid):
            if np.count_nonzero(valid) == 1:
                row[~valid] = row[valid][0]
            else:
                row[~valid] = np.interp(x[~valid], x[valid], row[valid])
        else:
            raise RuntimeError("Profile interpolation failed: one layer is all NaN.")
        arr[i, :] = row
    return arr


def get_mesh_xy(mesh: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return mesh cell-center x/y arrays."""
    centers = np.asarray(mesh.cellCenters(), dtype=float)
    if centers.ndim == 2 and centers.shape[1] >= 2:
        return centers[:, 0], centers[:, 1]
    x = np.array([float(c[0]) for c in mesh.cellCenters()], dtype=float)
    y = np.array([float(c[1]) for c in mesh.cellCenters()], dtype=float)
    return x, y


def assign_three_layer_markers(
    mesh: Any,
    line1: np.ndarray,
    line2: np.ndarray,
    top_marker: int = 0,
    mid_marker: int = 3,
    bot_marker: int = 2,
) -> np.ndarray:
    """Assign top/middle/bottom markers from two interface lines."""
    x_cell, y_cell = get_mesh_xy(mesh)
    y_line1 = np.interp(x_cell, line1[:, 0], line1[:, 1])
    y_line2 = np.interp(x_cell, line2[:, 0], line2[:, 1])
    markers = np.full(mesh.cellCount(), bot_marker, dtype=int)
    markers[y_cell >= y_line2] = mid_marker
    markers[y_cell >= y_line1] = top_marker
    mesh.setCellMarkers(markers)
    return markers


def interpolate_profile_to_mesh(
    profile_values: Any,
    layer_boundaries: Any,
    x_profile: Any,
    mesh: Any,
) -> np.ndarray:
    """Interpolate a (layers x distance) profile matrix onto mesh cells."""
    from scipy.interpolate import griddata

    values = np.asarray(profile_values, dtype=float)
    bounds = np.asarray(layer_boundaries, dtype=float)
    n_layers, n_profile = values.shape
    if bounds.shape != (n_layers + 1, n_profile):
        raise ValueError(
            f"layer_boundaries shape must be {(n_layers + 1, n_profile)}, got {bounds.shape}."
        )
    layer_centers = 0.5 * (bounds[:-1, :] + bounds[1:, :])
    x2d = np.repeat(np.asarray(x_profile, dtype=float)[np.newaxis, :], n_layers, axis=0)
    points = np.column_stack((x2d.ravel(), layer_centers.ravel()))
    vals = values.ravel()
    x_cell, y_cell = get_mesh_xy(mesh)
    query = np.column_stack((x_cell, y_cell))
    interp_linear = griddata(points, vals, query, method="linear")
    interp_nearest = griddata(points, vals, query, method="nearest")
    out = np.asarray(interp_linear, dtype=float)
    nan_mask = ~np.isfinite(out)
    out[nan_mask] = interp_nearest[nan_mask]
    return out


# ---------------------------------------------------------------------------
# Parameter access helpers
# ---------------------------------------------------------------------------
def _resolve_hydro_dir(context: Dict[str, Any], params: Dict[str, Any]) -> Path:
    """Pick the hydro data directory from params override or bridge context."""
    candidate = params.get("hydro_data_dir") or context.get("hydro_data_dir")
    if not candidate:
        raise ValueError(
            "No hydro data directory available. Select a folder containing "
            "Watercontent.npy, Porosity.npy, top.npy, bot.npy."
        )
    return Path(candidate)


def find_hydro_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Return the resolved path for each expected hydro file (or None)."""
    found: Dict[str, Optional[Path]] = {}
    for key, name in HYDRO_FILES.items():
        candidate = Path(data_dir) / name
        found[key] = candidate if candidate.exists() else None
    return found


def _output_root(context: Dict[str, Any], params: Dict[str, Any]) -> Path:
    base = (
        params.get("output_dir")
        or context.get("hydro_output_dir")
        or context.get("output_dir")
        or "."
    )
    return io_utils.ensure_dir(Path(base) / "qt_hydro_forward")


# ---------------------------------------------------------------------------
# Profile extraction (numpy/scipy only -- no pygimli required)
# ---------------------------------------------------------------------------
def extract_profile(
    context: Dict[str, Any],
    params: Dict[str, Any],
    point1: Sequence[float],
    point2: Sequence[float],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Extract the 2D hydro profile between two clicked points.

    Returns a dict with ``L_profile``, ``structure`` (layer boundaries),
    ``water_content_profile`` and ``porosity_profile`` plus bookkeeping. This
    only needs numpy + scipy via ``ProfileInterpolator`` and works without
    pygimli.
    """
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator

    data_dir = _resolve_hydro_dir(context, params)
    files = find_hydro_files(data_dir)
    missing = [HYDRO_FILES[k] for k, v in files.items() if v is None]
    if missing:
        raise ValueError(
            f"Missing hydro file(s) in {data_dir}: {', '.join(missing)}."
        )

    log(f"Loading hydro arrays from {data_dir}")
    water_content_4d = np.load(files["water_content"])
    porosity_3d = np.asarray(np.load(files["porosity"]), dtype=float)
    top = np.load(files["top"])
    bot = np.load(files["bot"])

    # Water content may be 4D (time, layer, row, col) or already 3D.
    snapshot_index = int(params.get("snapshot_index", 0))
    if water_content_4d.ndim == 4:
        snapshot_index = max(0, min(snapshot_index, water_content_4d.shape[0] - 1))
        water_content_3d = np.asarray(water_content_4d[snapshot_index], dtype=float)
    else:
        water_content_3d = np.asarray(water_content_4d, dtype=float)

    p1 = [int(round(point1[0])), int(round(point1[1]))]
    p2 = [int(round(point2[0])), int(round(point2[1]))]
    num_points = int(params.get("num_samples", 220))
    log(f"Building profile {p1} -> {p2} with {num_points} samples")

    interpolator = ProfileInterpolator(
        point1=p1,
        point2=p2,
        surface_data=top,
        origin_x=0.0,
        origin_y=0.0,
        pixel_width=1.0,
        pixel_height=-1.0,
        num_points=num_points,
    )

    structure = interpolator.interpolate_layer_data(
        [top] + [bot[i] for i in range(bot.shape[0])]
    )
    water_content_profile = interpolator.interpolate_3d_data(water_content_3d)
    porosity_profile = interpolator.interpolate_3d_data(porosity_3d)

    structure = fill_profile_nans(structure)
    water_content_profile = np.clip(fill_profile_nans(water_content_profile), 0.0, 0.8)
    porosity_profile = np.clip(fill_profile_nans(porosity_profile), 0.01, 0.6)

    return {
        "interpolator": interpolator,
        "L_profile": np.asarray(interpolator.L_profile, dtype=float),
        "structure": structure,
        "water_content_profile": water_content_profile,
        "porosity_profile": porosity_profile,
        "snapshot_index": snapshot_index,
        "data_dir": str(data_dir),
        "point1": p1,
        "point2": p2,
    }


def _rho_parameters(params: Dict[str, Any]) -> Dict[str, List[float]]:
    petro = params.get("petro", {})
    return {
        "rho_sat": list(petro.get("rho_sat", [100.0, 500.0, 2400.0])),
        "n": list(petro.get("archie_n", [2.2, 1.8, 2.5])),
        "sigma_s": list(petro.get("sigma_s", [1.0 / 500.0, 0.0, 0.0])),
    }


def build_survey_config(
    context: Dict[str, Any],
    params: Dict[str, Any],
    methods: Sequence[str],
    point1: Sequence[float],
    point2: Sequence[float],
    profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a complete, JSON-serializable survey configuration.

    If a previously extracted ``profile`` dict is supplied, a downsampled copy of
    the sampled profile is embedded so the config is useful even without a
    forward run.
    """
    ert = params.get("ert", {})
    srt = params.get("srt", {})
    config: Dict[str, Any] = {
        "created_time": _utc_now(),
        "point1": [float(point1[0]), float(point1[1])],
        "point2": [float(point2[0]), float(point2[1])],
        "methods": list(methods),
        "snapshot_index": int(params.get("snapshot_index", 0)),
        "num_samples": int(params.get("num_samples", 220)),
        "hydro_data_dir": str(params.get("hydro_data_dir") or context.get("hydro_data_dir", "")),
        "seed": int(params.get("seed", 7)),
        "mesh_settings": {
            "quality": float(params.get("mesh", {}).get("quality", 32)),
            "area": float(params.get("mesh", {}).get("area", 1.0)),
        },
        "em_stations": int(params.get("em_stations", 24)),
        "ert_settings": {
            "num_electrodes": int(ert.get("num_electrodes", 72)),
            "electrode_spacing": float(ert.get("electrode_spacing", 1.0)),
            "electrode_start": float(ert.get("electrode_start", 0.0)),
            "scheme_name": str(ert.get("scheme_name", "wa")),
            "noise_level": float(ert.get("noise_level", 0.03)),
            "rel_error": float(ert.get("rel_error", 0.03)),
            "abs_error": float(ert.get("abs_error", 0.0)),
        },
        "srt_settings": {
            "num_sensors": int(srt.get("num_sensors", 72)),
            "sensor_spacing": float(srt.get("sensor_spacing", 1.0)),
            "shot_distance": float(srt.get("shot_distance", 5.0)),
            "sensor_start": float(srt.get("sensor_start", 0.0)),
            "noise_level": float(srt.get("noise_level", 0.01)),
            "noise_abs": float(srt.get("noise_abs", 1e-5)),
        },
        "petrophysics": _rho_parameters(params),
        "srt_velocity": params.get("srt_vel", DEFAULT_VEL_PARAMETERS),
        "tdem_settings": params.get("tdem", {}),
        "fdem_settings": params.get("fdem", {}),
        "gravity_settings": params.get("gravity", {}),
    }
    if profile is not None:
        L = np.asarray(profile["L_profile"], dtype=float)
        wc = np.asarray(profile["water_content_profile"], dtype=float)
        # Downsample to keep the JSON small.
        step = max(1, L.size // 60)
        config["profile_preview"] = {
            "distance_m": L[::step].tolist(),
            "n_layers": int(wc.shape[0]),
            "n_profile_points": int(wc.shape[1]),
            "water_content_top_layer": wc[0, ::step].tolist(),
        }
    return config


def _fit_array_start(start: float, spacing: float, count: int, l_max: float) -> float:
    """Clamp the survey start so the whole array stays on the profile."""
    span = max(0.0, (count - 1) * spacing)
    if span >= l_max:
        return 0.0
    return float(min(max(0.0, start), l_max - span))


# ---------------------------------------------------------------------------
# Full forward run (requires pygimli)
# ---------------------------------------------------------------------------
def run_hydro_forward(
    context: Dict[str, Any],
    params: Dict[str, Any],
    methods: Sequence[str],
    point1: Sequence[float],
    point2: Sequence[float],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Run the real hydro->geophysics forward pipeline.

    Raises ``BackendUnavailable`` if pygimli / the hydro_to_* wrappers cannot be
    imported, and propagates any other exception from the forward run so the
    caller can fall back to config export.
    """
    try:
        import matplotlib
        # Force the non-interactive Agg backend so pygimli's internal ``plt.show()``
        # cannot try to open a Qt window (which would block forever when a
        # QApplication is live and the forward runs on a worker thread).
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        import pygimli as pg
        import pygimli.physics.traveltime as tt
        from pygimli.physics import ert as pg_ert
        from PyHydroGeophysX.core.interpolation import create_surface_lines
        from PyHydroGeophysX.core.mesh_utils import MeshCreator
        from PyHydroGeophysX.Hydro_modular import (
            hydro_to_ert,
            hydro_to_fdem,
            hydro_to_gravity,
            hydro_to_srt,
            hydro_to_tdem,
        )
    except Exception as exc:  # ImportError or pygimli init failure
        raise BackendUnavailable(str(exc))

    methods = [m for m in methods if m in ALL_METHODS]
    profile = extract_profile(context, params, point1, point2, log=log)
    interpolator = profile["interpolator"]
    L_profile = profile["L_profile"]
    structure = profile["structure"]
    wc_profile = profile["water_content_profile"]
    por_profile = profile["porosity_profile"]
    n_layers, n_profile = wc_profile.shape

    out_dir = _output_root(context, params)
    figure_paths: List[str] = []
    data_paths: List[str] = []
    data_summary: Dict[str, Any] = {}
    display_paths: Dict[str, Dict[str, str]] = {}

    def _save_data(fn: Callable[[Path], None], name: str) -> None:
        """Best-effort save of a synthetic-data artifact; never break the run."""
        try:
            path = out_dir / name
            fn(path)
            data_paths.append(str(path))
        except Exception as exc:  # noqa: BLE001
            log(f"Could not save {name}: {exc}")

    def _save_display(method: str, panel: str, name: str, render: Callable[[Any], None]) -> None:
        """Render one method-specific model or measurement panel for the Results UI."""
        fig, ax = plt.subplots(figsize=(6.2, 4.5))
        try:
            render(ax)
            fig.tight_layout()
            path = out_dir / name
            fig.savefig(path, dpi=180, bbox_inches="tight")
            display_paths.setdefault(method, {})[panel] = str(path)
        finally:
            plt.close(fig)

    # Build the mesh once; ERT and SRT share it.
    n_bounds = structure.shape[0]
    mid_idx = max(1, min(4, n_bounds // 3))
    bot_idx = max(mid_idx + 1, min(12, n_bounds - 2))
    log("Creating 2D mesh from layer boundaries")
    surface, line1, line2 = create_surface_lines(
        L_profile=L_profile, structure=structure, top_idx=0, mid_idx=mid_idx, bot_idx=bot_idx
    )
    mesh_cfg = params.get("mesh", {})
    mesh_creator = MeshCreator(
        quality=float(mesh_cfg.get("quality", 32)),
        area=float(mesh_cfg.get("area", 1.0)),
    )
    mesh, _ = mesh_creator.create_from_layers(
        surface=surface, layers=[line1, line2], bottom_depth=float(np.min(line2[:, 1]) - 10.0)
    )
    mesh_markers = assign_three_layer_markers(mesh, line1, line2, 0, 3, 2)
    wc_mesh = interpolate_profile_to_mesh(wc_profile, structure, L_profile, mesh)
    porosity_mesh = interpolate_profile_to_mesh(por_profile, structure, L_profile, mesh)
    layer_markers = [0, 3, 2]
    layer_idx = [0, mid_idx, bot_idx]
    l_max = float(L_profile[-1])

    rho_parameters = _rho_parameters(params)
    ert_cfg = params.get("ert", {})
    srt_cfg = params.get("srt", {})
    seed = int(params.get("seed", 7))

    resistivity_model = velocity_model = None
    ert_data = srt_data = None

    if "ERT" in methods:
        count = int(ert_cfg.get("num_electrodes", 72))
        spacing = float(ert_cfg.get("electrode_spacing", 1.0))
        start = _fit_array_start(float(ert_cfg.get("electrode_start", 0.0)), spacing, count, l_max)
        log(f"Running ERT forward ({count} electrodes, scheme {ert_cfg.get('scheme_name', 'wa')})")
        ert_data, resistivity_model = hydro_to_ert(
            water_content=wc_mesh, porosity=porosity_mesh, mesh=mesh,
            profile_interpolator=interpolator, layer_idx=layer_idx, structure=structure,
            marker_labels=layer_markers, rho_parameters=rho_parameters,
            electrode_spacing=spacing, electrode_start=start, num_electrodes=count,
            scheme_name=str(ert_cfg.get("scheme_name", "wa")),
            noise_level=float(ert_cfg.get("noise_level", 0.03)),
            abs_error=float(ert_cfg.get("abs_error", 0.0)),
            rel_error=float(ert_cfg.get("rel_error", 0.03)),
            mesh_markers=mesh_markers, verbose=False, seed=seed,
        )
        data_summary["ert_count"] = int(ert_data.size())
        _save_data(lambda p: ert_data.save(str(p)), "ert_data.dat")
        _save_data(lambda p: np.save(p, np.asarray(resistivity_model, dtype=float)),
                   "resistivity_model.npy")

    if "SRT" in methods:
        count = int(srt_cfg.get("num_sensors", 72))
        spacing = float(srt_cfg.get("sensor_spacing", 1.0))
        start = _fit_array_start(float(srt_cfg.get("sensor_start", 0.0)), spacing, count, l_max)
        log(f"Running SRT forward ({count} sensors)")
        vel_parameters = params.get("srt_vel") or DEFAULT_VEL_PARAMETERS
        srt_data, velocity_model = hydro_to_srt(
            water_content=wc_mesh, porosity=porosity_mesh, mesh=mesh,
            profile_interpolator=interpolator, layer_idx=layer_idx, structure=structure,
            marker_labels=layer_markers, vel_parameters=vel_parameters,
            sensor_spacing=spacing, sensor_start=start, num_sensors=count,
            shot_distance=int(round(float(srt_cfg.get("shot_distance", 5)))),
            noise_level=float(srt_cfg.get("noise_level", 0.01)),
            noise_abs=float(srt_cfg.get("noise_abs", 1e-5)),
            mesh_markers=mesh_markers, verbose=False, seed=seed,
        )
        data_summary["srt_count"] = int(srt_data.size())
        _save_data(lambda p: srt_data.save(str(p)), "srt_data.dat")
        _save_data(lambda p: np.save(p, np.asarray(velocity_model, dtype=float)),
                   "velocity_model.npy")

    if resistivity_model is not None or velocity_model is not None:
        if resistivity_model is not None:
            def render_ert_model(ax) -> None:
                pg.show(mesh, resistivity_model, ax=ax, cMap="Spectral_r", label="Resistivity (ohm m)")
                ax.set_title("ERT resistivity model")

            def render_ert_measurement(ax) -> None:
                pg_ert.show(ert_data, ax=ax)
                ax.set_title("Synthetic ERT measurements")

            _save_display("ERT", "model", "ert_model.png", render_ert_model)
            _save_display("ERT", "measurement", "ert_measurements.png", render_ert_measurement)
        if velocity_model is not None:
            def render_srt_model(ax) -> None:
                pg.show(mesh, velocity_model, ax=ax, cMap="turbo", label="Velocity (m/s)")
                ax.set_title("Seismic velocity model")

            def render_srt_measurement(ax) -> None:
                tt.drawFirstPicks(ax, srt_data)
                ax.set_title("Synthetic seismic first arrivals")

            _save_display("SRT", "model", "srt_velocity_model.png", render_srt_model)
            _save_display("SRT", "measurement", "srt_first_arrivals.png", render_srt_measurement)
        fig = plt.figure(figsize=(13, 7))
        if resistivity_model is not None:
            ax1 = fig.add_subplot(2, 2, 1)
            pg.show(mesh, resistivity_model, ax=ax1, cMap="Spectral_r", label="Resistivity (ohm m)")
            ax1.set_title("2D resistivity model")
            ax3 = fig.add_subplot(2, 2, 3)
            pg_ert.show(ert_data, ax=ax3)
            ax3.set_title("Synthetic ERT response")
        if velocity_model is not None:
            ax2 = fig.add_subplot(2, 2, 2)
            pg.show(mesh, velocity_model, ax=ax2, cMap="turbo", label="Velocity (m/s)")
            ax2.set_title("2D velocity model")
            ax4 = fig.add_subplot(2, 2, 4)
            tt.drawFirstPicks(ax4, srt_data)
            ax4.set_title("Synthetic SRT first arrivals")
        fig.tight_layout()
        ert_srt_path = out_dir / "ert_srt_models.png"
        fig.savefig(ert_srt_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(ert_srt_path))
        _save_data(lambda p: mesh.save(str(p)), "model_mesh.bms")

    # Pseudo-2D EM / gravity along the same profile.
    em_methods = [m for m in methods if m in ("TDEM", "FDEM", "Gravity")]
    if em_methods:
        target_stations = int(params.get("em_stations", 24))
        step = max(1, n_profile // max(1, target_stations))
        station_idx = np.unique(np.r_[np.arange(0, n_profile, step), n_profile - 1])
        x_station = L_profile[station_idx]
        wc_station = wc_profile[:, station_idx]
        por_station = por_profile[:, station_idx]
        structure_station = structure[:, station_idx]

        fig_model, ax_model = plt.subplots(figsize=(6.2, 4.5))
        im_model = ax_model.imshow(np.asarray(wc_profile, dtype=float), aspect="auto", origin="upper",
                                   extent=[float(L_profile[0]), float(L_profile[-1]), n_layers, 0],
                                   cmap="Blues")
        ax_model.set_xlabel("Distance along profile (m)")
        ax_model.set_ylabel("Layer index (surface \u2192 depth)")
        ax_model.set_title("Hydrologic profile input")
        fig_model.colorbar(im_model, ax=ax_model, label="Water content")
        fig_model.tight_layout()
        hydro_model_path = out_dir / "hydrologic_profile_model.png"
        fig_model.savefig(hydro_model_path, dpi=180, bbox_inches="tight")
        plt.close(fig_model)
        for method in em_methods:
            display_paths.setdefault(method, {})["model"] = str(hydro_model_path)

        fig, axes = plt.subplots(1, max(1, len(em_methods)), figsize=(5.2 * len(em_methods), 4.4), squeeze=False)
        col = 0
        if "TDEM" in methods:
            tdem_cfg = params.get("tdem", {})
            log("Running TDEM forward")
            times = np.logspace(
                float(np.log10(float(tdem_cfg.get("t_min_s", 1e-5)))),
                float(np.log10(float(tdem_cfg.get("t_max_s", 1e-2)))),
                int(tdem_cfg.get("n_gates", 28)),
            )
            tdem_noisy, tdem_clean, _, _ = hydro_to_tdem(
                water_content=wc_station, porosity=por_station, layer_boundaries=structure_station,
                times=times,
                sigma_w=float(tdem_cfg.get("sigma_w", 0.05)),
                m=float(tdem_cfg.get("m", 1.5)),
                n=float(tdem_cfg.get("n", 2.0)),
                sigma_s=float(tdem_cfg.get("sigma_s", 0.0)),
                source_radius=float(tdem_cfg.get("source_radius", 10.0)),
                noise_level=float(tdem_cfg.get("noise_level", 0.03)),
                seed=seed, verbose=False,
            )
            ax = axes[0][col]; col += 1
            im = ax.pcolormesh(x_station, times, np.abs(tdem_clean).T, shading="auto", cmap="magma")
            ax.set_yscale("log"); ax.set_xlabel("Distance (m)"); ax.set_ylabel("Time (s)")
            ax.set_title("Pseudo-2D TDEM |response|"); fig.colorbar(im, ax=ax)
            def render_tdem_measurement(ax) -> None:
                image = ax.pcolormesh(x_station, times, np.abs(tdem_clean).T, shading="auto", cmap="magma")
                ax.set_yscale("log"); ax.set_xlabel("Distance (m)"); ax.set_ylabel("Time (s)")
                ax.set_title("Synthetic TDEM measurements")
                ax.figure.colorbar(image, ax=ax, label="|response|")

            _save_display("TDEM", "measurement", "tdem_measurements.png", render_tdem_measurement)
            data_summary["tdem_shape"] = list(np.asarray(tdem_clean).shape)
            _save_data(lambda p: np.savez(p, times=times, clean=np.asarray(tdem_clean),
                                          noisy=np.asarray(tdem_noisy), x_station=x_station),
                       "tdem_response.npz")
        if "FDEM" in methods:
            fdem_cfg = params.get("fdem", {})
            log("Running FDEM forward")
            freqs = np.logspace(
                float(np.log10(float(fdem_cfg.get("f_min_hz", 10.0)))),
                float(np.log10(float(fdem_cfg.get("f_max_hz", 1e4)))),
                int(fdem_cfg.get("n_freqs", 18)),
            )
            _, fdem_clean, _, _ = hydro_to_fdem(
                water_content=wc_station, porosity=por_station, layer_boundaries=structure_station,
                frequencies=freqs,
                sigma_w=float(fdem_cfg.get("sigma_w", 0.05)),
                m=float(fdem_cfg.get("m", 1.5)),
                n=float(fdem_cfg.get("n", 2.0)),
                sigma_s=float(fdem_cfg.get("sigma_s", 0.0)),
                source_location=np.array([float(fdem_cfg.get("source_x", 0.0)), 0.0, 0.0]),
                receiver_location=np.array([float(fdem_cfg.get("receiver_x", 12.0)), 0.0, 0.0]),
                source_radius=float(fdem_cfg.get("source_radius", 10.0)),
                receiver_orientation=str(fdem_cfg.get("receiver_orientation", "z")),
                receiver_component=str(fdem_cfg.get("receiver_component", "secondary")),
                waveform_type=str(fdem_cfg.get("waveform_type", "dipole")),
                noise_level=float(fdem_cfg.get("noise_level", 0.03)),
                seed=seed, verbose=False,
            )
            ax = axes[0][col]; col += 1
            im = ax.pcolormesh(x_station, freqs, np.abs(np.imag(fdem_clean)).T, shading="auto", cmap="viridis")
            ax.set_yscale("log"); ax.set_xlabel("Distance (m)"); ax.set_ylabel("Frequency (Hz)")
            ax.set_title("Pseudo-2D FDEM |imag|"); fig.colorbar(im, ax=ax)
            def render_fdem_measurement(ax) -> None:
                image = ax.pcolormesh(x_station, freqs, np.abs(np.imag(fdem_clean)).T,
                                      shading="auto", cmap="viridis")
                ax.set_yscale("log"); ax.set_xlabel("Distance (m)"); ax.set_ylabel("Frequency (Hz)")
                ax.set_title("Synthetic FDEM measurements")
                ax.figure.colorbar(image, ax=ax, label="|imaginary response|")

            _save_display("FDEM", "measurement", "fdem_measurements.png", render_fdem_measurement)
            data_summary["fdem_shape"] = list(np.asarray(fdem_clean).shape)
            _save_data(lambda p: np.savez(p, frequencies=freqs, clean=np.asarray(fdem_clean),
                                          x_station=x_station), "fdem_response.npz")
        if "Gravity" in methods:
            grav_cfg = params.get("gravity", {})
            log("Running gravity forward")
            grav_noisy, grav_clean, _, _ = hydro_to_gravity(
                water_content=wc_station, porosity=por_station, layer_boundaries=structure_station,
                station_positions=x_station,
                rho_matrix=float(grav_cfg.get("rho_matrix", 2650.0)),
                rho_water=float(grav_cfg.get("rho_water", 1000.0)),
                rho_air=float(grav_cfg.get("rho_air", 1.225)),
                sensor_height=float(grav_cfg.get("sensor_height", 1.0)),
                noise_level=float(grav_cfg.get("noise_level", 0.02)),
                mesh_nx=int(grav_cfg.get("mesh_nx", 80)),
                mesh_nz=int(grav_cfg.get("mesh_nz", 60)),
                model_width_y=float(grav_cfg.get("model_width_y", 12.0)),
                seed=seed, verbose=False,
            )
            ax = axes[0][col]; col += 1
            ax.plot(x_station, grav_clean, "k-", lw=1.6, label="clean")
            ax.plot(x_station, grav_noisy, "o", ms=3.5, alpha=0.7, label="noisy")
            ax.set_xlabel("Distance (m)"); ax.set_ylabel("Gravity anomaly (mGal)")
            ax.set_title("Pseudo-2D gravity"); ax.grid(True, alpha=0.25); ax.legend(loc="best")
            def render_gravity_measurement(ax) -> None:
                ax.plot(x_station, grav_clean, "k-", lw=1.6, label="clean")
                ax.plot(x_station, grav_noisy, "o", ms=3.5, alpha=0.7, label="noisy")
                ax.set_xlabel("Distance (m)"); ax.set_ylabel("Gravity anomaly (mGal)")
                ax.set_title("Synthetic gravity measurements")
                ax.grid(True, alpha=0.25); ax.legend(loc="best")

            _save_display("Gravity", "measurement", "gravity_measurements.png", render_gravity_measurement)
            data_summary["gravity_range_mGal"] = [float(np.min(grav_clean)), float(np.max(grav_clean))]
            _save_data(lambda p: np.savez(p, station_x=x_station, clean=np.asarray(grav_clean),
                                          noisy=np.asarray(grav_noisy)), "gravity_response.npz")
        fig.tight_layout()
        em_path = out_dir / "em_gravity_profile.png"
        fig.savefig(em_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(em_path))

    # Always write the survey config alongside the figures.
    config = build_survey_config(context, params, methods, point1, point2, profile=profile)
    config_path = out_dir / "survey_config.json"
    io_utils.write_json(config_path, config)

    return {
        "status": "ok",
        "methods": list(methods),
        "selected_points": [profile["point1"], profile["point2"]],
        "snapshot_index": profile["snapshot_index"],
        "num_profile_points": int(n_profile),
        "mesh_cells": int(mesh.cellCount()),
        "config_path": str(config_path),
        "figure_paths": figure_paths,
        "display_paths": display_paths,
        "data_paths": data_paths,
        "data_summary": data_summary,
        "output_dir": str(out_dir),
    }
