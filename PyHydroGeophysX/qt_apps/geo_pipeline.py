"""Geophysics -> hydrology inverse pipeline for the desktop workbench.

This is the inverse counterpart to ``hydro_pipeline.py``. It takes an
already-inverted (time-lapse) ERT resistivity model on a mesh and converts it to
volumetric water content (and, optionally, saturated-zone porosity) with
Monte Carlo uncertainty. It is a thin, parameterized re-use of
``examples/Ex_MC_Hydro.py`` and the petrophysics in
``PyHydroGeophysX.petrophysics.resistivity_models``. It is deliberately Qt-free so
it can run inside a worker thread (or be unit-tested) without a QApplication.

Two layers:

* ``extract_model_summary`` / ``build_petro_config`` / ``extract_point_series`` --
  numpy only. These work even when pygimli is not installed, so loading the
  inverted model bundle, inspecting its layers, exporting the petrophysics
  configuration, and reading monitoring-point time series never depend on the
  heavy backend.
* ``run_ert_to_wc`` -- the real Monte Carlo run. It imports pygimli lazily to load
  the mesh and render the section figures; if anything is missing or fails it
  raises ``BackendUnavailable`` so the caller can fall back to config export.

The petrophysical conversion itself (``resistivity_to_saturation`` /
``resistivity_to_porosity``) is pure numpy/scipy and is reused directly from the
library rather than re-implemented here.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

#: The four files of an inverted-model bundle. ``markers`` and ``coverage`` are
#: optional; everything else is required for a real run.
MODEL_FILES = {
    "mesh": "mesh_res.bms",
    "resistivity": "resmodel.npy",
    "markers": "index_marker.npy",
    "coverage": "all_coverage.npy",
}

PRODUCTS = ("water_content", "porosity")

#: Fallback per-layer petrophysics distributions, matching the two layers of the
#: bundled Treeline demo in ``Ex_MC_Hydro.py`` (marker 3 = regolith, 2 = bedrock).
DEFAULT_LAYER_DISTRIBUTIONS: Dict[int, Dict[str, Any]] = {
    3: {
        "name": "Regolith",
        "m": {"mean": 1.3, "std": 0.1},
        "rho_fluid": {"mean": 20.0, "std": 0.0},
        "n": {"mean": 2.1, "std": 0.1},
        "sigma_sur": {"mean": 1.0 / 200.0, "std": 1.0 / 200.0},
        "porosity": {"mean": 0.42, "std": 0.05},
    },
    2: {
        "name": "Bedrock",
        "m": {"mean": 1.9, "std": 0.2},
        "rho_fluid": {"mean": 20.0, "std": 0.0},
        "n": {"mean": 1.7, "std": 0.2},
        "sigma_sur": {"mean": 0.0, "std": 0.0},
        "porosity": {"mean": 0.25, "std": 0.15},
    },
}

# Petrophysics parameter keys sampled per layer (porosity only for water content).
_DIST_KEYS = ("m", "rho_fluid", "n", "sigma_sur", "porosity")


class BackendUnavailable(RuntimeError):
    """Raised when pygimli cannot be used (mirror of hydro_pipeline)."""


def _noop(_msg: str) -> None:
    return None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Directory / file resolution (numpy only)
# ---------------------------------------------------------------------------
def _resolve_model_dir(context: Dict[str, Any], params: Dict[str, Any]) -> Path:
    """Pick the inverted-model directory from params override or bridge context."""
    candidate = (
        params.get("model_data_dir")
        or context.get("geo_data_dir")
        or context.get("model_data_dir")
    )
    if not candidate:
        raise ValueError(
            "No inverted-model directory available. Select a folder containing "
            "mesh_res.bms, resmodel.npy, index_marker.npy."
        )
    return Path(candidate)


def find_model_files(data_dir: Path) -> Dict[str, Optional[Path]]:
    """Return the resolved path for each expected bundle file (or None)."""
    found: Dict[str, Optional[Path]] = {}
    for key, name in MODEL_FILES.items():
        candidate = Path(data_dir) / name
        found[key] = candidate if candidate.exists() else None
    return found


def derive_markers_from_interface(
    context: Dict[str, Any],
    params: Dict[str, Any],
    interface_xz: Any,
    top_marker: int = 3,
    bot_marker: int = 2,
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Classify the model mesh into layers using a (seismic) bedrock interface.

    Cells whose center elevation is at or above the interface get ``top_marker``
    (e.g. regolith); cells below get ``bot_marker`` (e.g. bedrock). The result is
    written as ``index_marker.npy`` in the model directory so the rest of the
    Geophysics -> Hydro workflow can use it. Requires pygimli to load the
    mesh.
    """
    try:
        import pygimli as pg
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))

    data_dir = _resolve_model_dir(context, params)
    files = find_model_files(data_dir)
    if files["mesh"] is None:
        raise ValueError(f"Missing {MODEL_FILES['mesh']} in {data_dir}; cannot classify cells.")

    iface = np.asarray(interface_xz, dtype=float)
    if iface.ndim != 2 or iface.shape[1] < 2:
        raise ValueError("interface_xz must be an (N, 2) array of [x, z] points.")
    order = np.argsort(iface[:, 0])
    ix, iz = iface[order, 0], iface[order, 1]

    mesh = pg.load(str(files["mesh"]))
    centers = np.asarray(mesh.cellCenters(), dtype=float)
    cx, cz = centers[:, 0], centers[:, 1]
    iface_z = np.interp(cx, ix, iz)  # clamps outside the interface x-range
    markers = np.where(cz >= iface_z, int(top_marker), int(bot_marker)).astype(int)

    # Write a sidecar artifact in the output dir (never overwrite the bundle's own
    # index_marker.npy); the module keeps the markers in memory as an override.
    out_path = _output_root(context, params) / "index_marker_from_structure.npy"
    np.save(out_path, markers)
    unique, counts = np.unique(markers, return_counts=True)
    log(f"Derived markers from interface: "
        + ", ".join(f"{int(m)}={int(c)}" for m, c in zip(unique, counts)))
    return {
        "markers": markers.astype(int).tolist(),
        "markers_path": str(out_path),
        "n_cells": int(markers.size),
        "counts": {int(m): int(c) for m, c in zip(unique, counts)},
        "top_marker": int(top_marker),
        "bot_marker": int(bot_marker),
    }


def _output_root(context: Dict[str, Any], params: Dict[str, Any]) -> Path:
    base = (
        params.get("output_dir")
        or context.get("output_dir")
        or context.get("geo_output_dir")
        or "."
    )
    return io_utils.ensure_dir(Path(base) / "qt_geo_inverse")


# ---------------------------------------------------------------------------
# Model summary (numpy only -- no pygimli required)
# ---------------------------------------------------------------------------
def _load_resistivity(path: Path) -> np.ndarray:
    """Load the inverted resistivity model as a 2D ``(n_cells, n_time)`` array."""
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2:
        raise ValueError(
            f"Resistivity model must be 1D or 2D (n_cells, n_time); got shape {arr.shape}."
        )
    return arr


def extract_model_summary(
    context: Dict[str, Any],
    params: Dict[str, Any],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Inspect the inverted-model bundle without loading the mesh.

    Returns shapes, the unique layer markers with per-marker cell counts, and a
    few resistivity statistics. Powers the Data and Layers steps with no pygimli
    dependency (the inverse analog of ``hydro_pipeline.extract_profile``).
    """
    data_dir = _resolve_model_dir(context, params)
    files = find_model_files(data_dir)
    if files["resistivity"] is None:
        raise ValueError(
            f"Missing {MODEL_FILES['resistivity']} in {data_dir}."
        )

    log(f"Reading inverted resistivity from {data_dir}")
    resistivity = _load_resistivity(files["resistivity"])
    n_cells, n_time = resistivity.shape

    override = params.get("markers_override")
    if override is not None:
        markers = np.asarray(override, dtype=int).ravel()
        markers_source = "derived from structure"
    elif files["markers"] is not None:
        markers = np.asarray(np.load(files["markers"]), dtype=int).ravel()
        markers_source = MODEL_FILES["markers"]
    else:
        # Fallback: a single layer covering every cell.
        markers = np.zeros(n_cells, dtype=int)
        markers_source = "fallback (single layer)"
    if markers.size != n_cells:
        raise ValueError(
            f"Marker count ({markers.size}) does not match resistivity cell count "
            f"({n_cells})."
        )

    unique, counts = np.unique(markers, return_counts=True)
    layers = [
        {"marker": int(m), "n_cells": int(c),
         "name": DEFAULT_LAYER_DISTRIBUTIONS.get(int(m), {}).get("name", f"Layer {int(m)}")}
        for m, c in zip(unique.tolist(), counts.tolist())
    ]

    coverage_shape = None
    if files["coverage"] is not None:
        try:
            coverage_shape = list(np.asarray(np.load(files["coverage"])).shape)
        except Exception:  # noqa: BLE001 - coverage is optional
            coverage_shape = None

    finite = resistivity[np.isfinite(resistivity)]
    res_stats = {
        "min": float(np.min(finite)) if finite.size else float("nan"),
        "max": float(np.max(finite)) if finite.size else float("nan"),
        "p10": float(np.percentile(finite, 10)) if finite.size else float("nan"),
        "p50": float(np.percentile(finite, 50)) if finite.size else float("nan"),
        "p90": float(np.percentile(finite, 90)) if finite.size else float("nan"),
    }

    return {
        "data_dir": str(data_dir),
        "n_cells": int(n_cells),
        "n_timesteps": int(n_time),
        "markers_source": markers_source,
        "layers": layers,
        "coverage_shape": coverage_shape,
        "resistivity_stats": res_stats,
    }


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------
def _layers_from_params(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize the per-layer distribution list from the UI params dict.

    Falls back to :data:`DEFAULT_LAYER_DISTRIBUTIONS` when none are supplied.
    """
    raw = params.get("layers")
    if not raw:
        return [
            {"marker": int(marker), **{k: dict(v) if isinstance(v, dict) else v
                                       for k, v in dist.items()}}
            for marker, dist in DEFAULT_LAYER_DISTRIBUTIONS.items()
        ]
    layers: List[Dict[str, Any]] = []
    for entry in raw:
        layer = {"marker": int(entry["marker"]),
                 "name": entry.get("name", f"Layer {int(entry['marker'])}")}
        for key in _DIST_KEYS:
            dist = entry.get(key, {"mean": 0.0, "std": 0.0})
            layer[key] = {"mean": float(dist.get("mean", 0.0)),
                          "std": float(dist.get("std", 0.0))}
        layers.append(layer)
    return layers


def _sample_layer(rng: Any, layer: Dict[str, Any]) -> Dict[str, float]:
    """Draw one Monte Carlo sample of a layer's petrophysics parameters."""
    out: Dict[str, float] = {}
    for key in _DIST_KEYS:
        dist = layer.get(key, {"mean": 0.0, "std": 0.0})
        value = float(rng.normal(dist.get("mean", 0.0), dist.get("std", 0.0)))
        # Keep parameters physical.
        if key in ("m", "n", "rho_fluid"):
            value = max(1e-6, value)
        elif key == "sigma_sur":
            value = max(0.0, value)
        elif key == "porosity":
            value = float(np.clip(value, 0.01, 0.9))
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Monitoring-point time series (numpy only)
# ---------------------------------------------------------------------------
def extract_point_series(
    cell_centers: np.ndarray,
    values: np.ndarray,
    positions: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, List[int]]:
    """Nearest-cell time series at monitoring ``(x, y)`` positions.

    Args:
        cell_centers: ``(n_cells, >=2)`` mesh cell-center coordinates.
        values: ``(n_cells, n_time)`` per-cell field (e.g. mean water content).
        positions: list of ``(x, y)`` tuples.

    Returns:
        ``(series, cell_indices)`` where ``series`` is ``(n_positions, n_time)``.
    """
    centers = np.asarray(cell_centers, dtype=float)
    field = np.atleast_2d(np.asarray(values, dtype=float))
    if field.shape[0] != centers.shape[0] and field.shape[1] == centers.shape[0]:
        field = field.T
    indices: List[int] = []
    for x_pos, y_pos in positions:
        d = (centers[:, 0] - float(x_pos)) ** 2 + (centers[:, 1] - float(y_pos)) ** 2
        indices.append(int(np.argmin(d)))
    series = np.array([field[i, :] for i in indices], dtype=float)
    return series, indices


# ---------------------------------------------------------------------------
# Config export (numpy only -- no pygimli required)
# ---------------------------------------------------------------------------
def build_petro_config(
    context: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a complete, JSON-serializable petrophysics inversion configuration."""
    layers = _layers_from_params(params)
    products = [p for p in params.get("products", ["water_content"]) if p in PRODUCTS]
    return {
        "created_time": _utc_now(),
        "model_data_dir": str(params.get("model_data_dir") or context.get("geo_data_dir", "")),
        "direction": "geophysics_to_hydrology",
        "method": "ERT",
        "products": products,
        "n_realizations": int(params.get("n_realizations", 100)),
        "seed": int(params.get("seed", 7)),
        "saturation_value": float(params.get("saturation_value", 1.0)),
        "tortuosity_a": float(params.get("tortuosity_a", 1.0)),
        "coverage_threshold": float(params.get("coverage_threshold", -1.0)),
        "preview_timestep": int(params.get("preview_timestep", 0)),
        "wc_color_range": [float(params.get("wc_cmin", 0.0)),
                           float(params.get("wc_cmax", 0.32))],
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Monte Carlo conversion core (numpy/scipy only)
# ---------------------------------------------------------------------------
def _run_monte_carlo(
    resistivity: np.ndarray,
    markers: np.ndarray,
    layers: List[Dict[str, Any]],
    products: Sequence[str],
    n_realizations: int,
    seed: int,
    saturation_value: float,
    tortuosity_a: float,
    timestep_indices: Optional[Sequence[int]],
    log: LogFn,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Sample petrophysics and convert resistivity to water content / porosity.

    Returns ``{product: {"mean":..., "std":..., "p10":..., "p50":..., "p90":...}}``
    with each statistic an ``(n_cells, n_time)`` array. Reuses the library's
    ``resistivity_to_saturation`` / ``resistivity_to_porosity``.
    """
    from PyHydroGeophysX.petrophysics.resistivity_models import (
        resistivity_to_porosity,
        resistivity_to_saturation,
    )

    n_cells, n_time_all = resistivity.shape
    t_idx = list(range(n_time_all)) if timestep_indices is None else list(timestep_indices)
    n_time = len(t_idx)
    rng = np.random.default_rng(int(seed))

    want_wc = "water_content" in products
    want_por = "porosity" in products
    wc_all = np.zeros((n_realizations, n_cells, n_time)) if want_wc else None
    por_all = np.zeros((n_realizations, n_cells, n_time)) if want_por else None

    masks = {int(layer["marker"]): (markers == int(layer["marker"])) for layer in layers}

    for r in range(n_realizations):
        if r % max(1, n_realizations // 10) == 0:
            log(f"Monte Carlo realization {r + 1}/{n_realizations}")
        sampled = {int(layer["marker"]): _sample_layer(rng, layer) for layer in layers}
        porosity_cell = np.zeros(n_cells, dtype=float)
        for marker, p in sampled.items():
            porosity_cell[masks[marker]] = p["porosity"]

        for col, t in enumerate(t_idx):
            res_t = resistivity[:, t]
            for marker, mask in masks.items():
                if not np.any(mask):
                    continue
                p = sampled[marker]
                if want_wc:
                    sat = resistivity_to_saturation(
                        resistivity=res_t[mask], porosity=p["porosity"],
                        m=p["m"], rho_fluid=p["rho_fluid"], n=p["n"],
                        sigma_sur=p["sigma_sur"], a=tortuosity_a,
                    )
                    wc_all[r, mask, col] = np.asarray(sat, dtype=float) * p["porosity"]
                if want_por:
                    por = resistivity_to_porosity(
                        resistivity=res_t[mask], saturation=saturation_value,
                        m=p["m"], rho_fluid=p["rho_fluid"], n=p["n"],
                        sigma_sur=p["sigma_sur"], a=tortuosity_a,
                    )
                    por_all[r, mask, col] = np.asarray(por, dtype=float)

    def _stats(values_all: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "mean": np.mean(values_all, axis=0),
            "std": np.std(values_all, axis=0),
            "p10": np.percentile(values_all, 10, axis=0),
            "p50": np.percentile(values_all, 50, axis=0),
            "p90": np.percentile(values_all, 90, axis=0),
        }

    out: Dict[str, Dict[str, np.ndarray]] = {}
    if want_wc:
        out["water_content"] = _stats(wc_all)
    if want_por:
        out["porosity"] = _stats(por_all)
    return out


# ---------------------------------------------------------------------------
# Full inverse run (requires pygimli for the mesh + figures)
# ---------------------------------------------------------------------------
def run_ert_to_wc(
    context: Dict[str, Any],
    params: Dict[str, Any],
    log: LogFn = _noop,
) -> Dict[str, Any]:
    """Run the real ERT -> water content / porosity Monte Carlo inversion.

    Raises ``BackendUnavailable`` if pygimli cannot be imported (needed to load
    the ``.bms`` mesh and render the section figures), and propagates any other
    exception so the caller can fall back to config export.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.ioff()
        import pygimli as pg
    except Exception as exc:  # ImportError or pygimli init failure
        raise BackendUnavailable(str(exc))

    data_dir = _resolve_model_dir(context, params)
    files = find_model_files(data_dir)
    missing = [MODEL_FILES[k] for k in ("mesh", "resistivity") if files[k] is None]
    if missing:
        raise ValueError(f"Missing required file(s) in {data_dir}: {', '.join(missing)}.")

    log(f"Loading inverted model from {data_dir}")
    mesh = pg.load(str(files["mesh"]))
    resistivity = _load_resistivity(files["resistivity"])
    n_cells, n_time = resistivity.shape
    if mesh.cellCount() != n_cells:
        raise ValueError(
            f"Mesh cell count ({mesh.cellCount()}) does not match resistivity "
            f"({n_cells})."
        )

    override = params.get("markers_override")
    if override is not None:
        markers = np.asarray(override, dtype=int).ravel()
    elif files["markers"] is not None:
        markers = np.asarray(np.load(files["markers"]), dtype=int).ravel()
    else:
        markers = np.zeros(n_cells, dtype=int)
    coverage = None
    if files["coverage"] is not None:
        try:
            coverage = np.asarray(np.load(files["coverage"]), dtype=float)
        except Exception:  # noqa: BLE001 - coverage is optional
            coverage = None

    layers = _layers_from_params(params)
    products = [p for p in params.get("products", ["water_content"]) if p in PRODUCTS]
    if not products:
        products = ["water_content"]
    n_real = int(params.get("n_realizations", 100))
    seed = int(params.get("seed", 7))
    sat_value = float(params.get("saturation_value", 1.0))
    a = float(params.get("tortuosity_a", 1.0))
    cov_thr = float(params.get("coverage_threshold", -1.0))
    preview_t = int(np.clip(int(params.get("preview_timestep", 0)), 0, n_time - 1))
    cmin = float(params.get("wc_cmin", 0.0))
    cmax = float(params.get("wc_cmax", 0.32))
    timestep_indices = params.get("timestep_indices")  # optional speed knob

    log(f"Monte Carlo: {n_real} realizations, products={products}, "
        f"{len(layers)} layer(s)")
    stats = _run_monte_carlo(
        resistivity, markers, layers, products, n_real, seed, sat_value, a,
        timestep_indices, log,
    )

    out_dir = _output_root(context, params)
    figure_paths: List[str] = []
    stats_paths: List[str] = []

    # When a timestep subset was used, the preview index refers to the subset.
    eff_t = list(range(n_time)) if timestep_indices is None else list(timestep_indices)
    preview_col = eff_t.index(preview_t) if preview_t in eff_t else 0
    cov_row = None
    if coverage is not None:
        if coverage.ndim == 2 and coverage.shape[0] > preview_t:
            cov_row = coverage[preview_t] > cov_thr
        elif coverage.ndim == 1:
            cov_row = coverage > cov_thr

    def _mesh_map(values: np.ndarray, title: str, label: str, fname: str,
                  cmap: str, vmin: float, vmax: float) -> None:
        fig = plt.figure(figsize=(8.2, 4.4))
        ax = fig.add_subplot(111)
        show_kw = dict(ax=ax, cMap=cmap, cMin=vmin, cMax=vmax, label=label,
                       logScale=False)
        if cov_row is not None:
            show_kw["coverage"] = cov_row
        try:
            pg.show(mesh, values, **show_kw)
        except Exception:  # noqa: BLE001 - coverage masking can fail; retry plain
            show_kw.pop("coverage", None)
            ax.clear()
            pg.show(mesh, values, **show_kw)
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(str(path))

    if "water_content" in stats:
        wc = stats["water_content"]
        _mesh_map(wc["mean"][:, preview_col],
                  f"Water content (mean), t={preview_t}", "Water content (-)",
                  "wc_mean_map.png", "viridis", cmin, cmax)
        _mesh_map(wc["std"][:, preview_col],
                  f"Water content (std), t={preview_t}", "Std water content (-)",
                  "wc_std_map.png", "magma", 0.0, float(max(1e-3, np.nanmax(wc["std"][:, preview_col]))))
        for key in ("mean", "std", "p10", "p50", "p90"):
            p = out_dir / f"water_content_{key}.npy"
            np.save(p, wc[key]); stats_paths.append(str(p))

    if "porosity" in stats:
        por = stats["porosity"]
        _mesh_map(por["mean"][:, preview_col],
                  f"Porosity (mean), t={preview_t}", "Porosity (-)",
                  "porosity_mean_map.png", "viridis", 0.0, 0.5)
        for key in ("mean", "std", "p10", "p50", "p90"):
            p = out_dir / f"porosity_{key}.npy"
            np.save(p, por[key]); stats_paths.append(str(p))

    # Always write the petrophysics config alongside the figures.
    config = build_petro_config(context, params)
    config_path = out_dir / "petro_config.json"
    io_utils.write_json(config_path, config)

    cell_centers = np.asarray(mesh.cellCenters(), dtype=float)

    result: Dict[str, Any] = {
        "status": "ok",
        "direction": "geophysics_to_hydrology",
        "products": products,
        "n_realizations": n_real,
        "mesh_cells": int(mesh.cellCount()),
        "n_timesteps": int(n_time),
        "preview_timestep": preview_t,
        "layers": [{"marker": int(l["marker"]), "name": l.get("name", "")} for l in layers],
        "figure_paths": figure_paths,
        "stats_paths": stats_paths,
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        # In-process payload for the Results monitoring-point feature (numpy
        # arrays; not serialized into module_results JSON).
        "_cell_centers": cell_centers,
        "_wc_mean": stats.get("water_content", {}).get("mean"),
        "_wc_std": stats.get("water_content", {}).get("std"),
        "_timestep_indices": eff_t,
    }
    return result
