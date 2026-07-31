"""Generate synthetic *Hydro -> Geophysics* example datasets for the Qt workbench.

The desktop workbench ships three coupled modules whose bundled "Use example"
button loads a small dataset:

* ``hydro_geophysics``  (navigator "Hydro -> Geophysics")  -> a gridded hydrology model
* ``geo_hydrology``     (navigator "ERT -> Water Content")  -> an inverted ERT model bundle
* ``seismic3d``         (navigator "Seismic -> Structure")  -> two or more 2D velocity sections

A **single** synthetic hydrology model (the ``timelapse_infiltration`` scenario)
drives all three products, run through the **real Hydro -> Geophysics forward
workflow** the app itself uses:

* ``PyHydroGeophysX.Hydro_modular.hydro_to_ert`` -> synthetic ERT + the per-cell
  resistivity model saved as ``resmodel.npy`` (for ``geo_hydrology``);
* ``PyHydroGeophysX.Hydro_modular.hydro_to_srt`` -> synthetic SRT + the per-cell
  velocity model saved as ``Vinvmodel.npy`` (for ``seismic3d``).

The 3D "truth" volume is built from the same 3D hydrology model using the exact
rock-physics that ``hydro_to_srt`` applies internally (Hertz-Mindlin for the
regolith, Differential Effective Medium for the bedrock), so the 2D lines are
consistent slices of it. Two extra hydrology-only scenarios (``wet_shallow``,
``dry_deep``) are also written as additional "Select folder..." examples.

Outputs (match each module's documented input contract):

    examples/data/synthetic/<scenario>/          Watercontent.npy, Porosity.npy, top.npy, bot.npy
    examples/results/synthetic_Structure_WC/     mesh_res.bms, resmodel.npy, index_marker.npy, all_coverage.npy
    examples/results/synthetic_seismic/lineN/    velmesh.bms, Vinvmodel.npy                (2D velocity sections)
    examples/results/synthetic_seismic3d/        velocity_true.{npy,vtk} (3D truth),
                                                 velocity_kriged.{npy,vtk} (Kriging from the 2D lines),
                                                 bedrock_depth_true_vs_kriged.png

The seismic example is provided in both 2D (per-line velocity sections) and 3D
form: a true 3D volume, plus the ordinary-Kriging reconstruction interpolated
from the 2D lines (gstools), so the "if 2D, krige to 3D" path can be compared
against ground truth. Velocities are tuned so the regolith stays below and the
bedrock above the seismic module's default 1200 m/s interface threshold.

The forward runs use pygimli (``hydro_to_ert`` / ``hydro_to_srt`` simulate real
data), so the full run takes a few minutes. Use ``--quick`` for a fast low-res
pass, e.g.::

    conda run -n pg python examples/generate_synthetic_examples.py            # full
    conda run -n pg python examples/generate_synthetic_examples.py --quick    # fast low-res
    conda run -n pg python examples/generate_synthetic_examples.py --scenario wet_shallow --skip-ert --skip-seismic

The seismic line map endpoints (``SEISMIC_LINES``) must stay in sync with
``_SYNTHETIC_LINES`` in ``PyHydroGeophysX/qt_apps/modules/seismic3d.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

# Headless matplotlib must be selected before pygimli imports it.
import matplotlib
matplotlib.use("Agg", force=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyHydroGeophysX.qt_apps.hydro_pipeline import (  # noqa: E402
    assign_three_layer_markers,
    extract_profile,
    interpolate_profile_to_mesh,
)
from PyHydroGeophysX.core.interpolation import create_surface_lines  # noqa: E402
from PyHydroGeophysX.core.mesh_utils import MeshCreator  # noqa: E402
from PyHydroGeophysX.Hydro_modular import hydro_to_ert, hydro_to_srt  # noqa: E402
from PyHydroGeophysX.petrophysics.velocity_models import DEMModel, HertzMindlinModel  # noqa: E402

# ---------------------------------------------------------------------------
# Geometry / petrophysics constants.
# ---------------------------------------------------------------------------
N_LAYERS = 6                        # model sub-layers -> N_LAYERS + 1 boundaries
BASE_ELEV = 1650.0                  # surface base elevation (m)

# Per-layer porosity: regolith (porous) -> transition -> bedrock (tight).
LAYER_POROSITY = np.array([0.42, 0.40, 0.28, 0.22, 0.10, 0.08])
# Fractions of the local bedrock depth D(x, y) at which each layer bottom sits.
LAYER_DEPTH_FRAC = np.array([0.15, 0.35, 0.55, 0.75, 1.00, 1.25])

# ``assign_three_layer_markers`` labels cells 0 (shallowest) / 3 (middle) / 2
# (deepest bedrock); the regolith/bedrock interface is the 3->2 boundary.
MARKER_LABELS = [0, 3, 2]           # top / middle / bottom, as hydro_to_* expect
GEO_MARKER_REMAP = {0: 3}           # collapse to {3: regolith, 2: bedrock} for the ERT bundle

# ERT petrophysics for hydro_to_ert (per layer: top / middle / bottom).
RHO_PARAMETERS: Dict[str, List[float]] = {
    "rho_sat": [200.0, 250.0, 2500.0],
    "n": [2.0, 2.0, 1.9],
    "sigma_s": [1.0 / 150.0, 1.0 / 150.0, 0.0],
}
# SRT velocity parameters for hydro_to_srt. Tuned (soft regolith moduli) so the
# regolith stays below and the bedrock above the module's default 1200 m/s
# threshold: top -> Hertz-Mindlin, mid/bot -> DEM (same as hydro_to_srt).
VEL_PARAMETERS: Dict[str, Dict[str, float]] = {
    "top": {"bulk_modulus": 2.0, "shear_modulus": 1.0, "mineral_density": 2350, "depth": 0.5},
    "mid": {"bulk_modulus": 6.0, "shear_modulus": 3.5, "mineral_density": 2600, "aspect_ratio": 0.03},
    "bot": {"bulk_modulus": 55.0, "shear_modulus": 50.0, "mineral_density": 2680, "aspect_ratio": 0.03},
}
# The two rock-physics models hydro_to_srt uses, reused here for the 3D truth.
_HM = HertzMindlinModel(critical_porosity=0.4, coordination_number=6.0)
_DEM = DEMModel()

ERT_NUM_ELECTRODES = 48
ERT_SCHEME = "dd"                   # dipole-dipole
SRT_NUM_SENSORS = 48

# The single scenario that drives all three modules.
DRIVER = "timelapse_infiltration"
GEO_PROFILE = {"p1": [20, 45], "p2": [100, 45]}
SEISMIC_LINES: List[Dict[str, object]] = [
    {"name": "L1", "p1": [15, 20], "p2": [105, 25]},
    {"name": "L2", "p1": [20, 70], "p2": [100, 65]},
    {"name": "L3", "p1": [30, 15], "p2": [35, 75]},
]

# Survey bounding box (grid coords) shared by the 2D lines, the true 3D volume,
# and the kriged reconstruction: (x_min, x_max, y_min, y_max).
SEIS_BBOX = (
    min(int(p) for s in SEISMIC_LINES for p in (s["p1"][0], s["p2"][0])),
    max(int(p) for s in SEISMIC_LINES for p in (s["p1"][0], s["p2"][0])),
    min(int(p) for s in SEISMIC_LINES for p in (s["p1"][1], s["p2"][1])),
    max(int(p) for s in SEISMIC_LINES for p in (s["p1"][1], s["p2"][1])),
)

SCENARIOS: Dict[str, Dict] = {
    "timelapse_infiltration": {
        "mean_depth": 18.0, "amp": 4.0, "timelapse": True,
        "sat_dry": [0.35, 0.30, 0.25, 0.20, 0.12, 0.08],
        "sat_wet": [0.92, 0.85, 0.70, 0.55, 0.25, 0.15],
    },
    "wet_shallow": {
        "mean_depth": 9.0, "amp": 2.5, "timelapse": False,
        "sat": [0.92, 0.86, 0.72, 0.55, 0.30, 0.20],
    },
    "dry_deep": {
        "mean_depth": 26.0, "amp": 5.0, "timelapse": False,
        "sat": [0.45, 0.40, 0.30, 0.22, 0.12, 0.08],
    },
}


# ---------------------------------------------------------------------------
# 1. Synthetic hydrology grids (pure numpy).
# ---------------------------------------------------------------------------
def _surface_elevation(nx: int, ny: int) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    return (BASE_ELEV - 0.06 * xx - 0.03 * yy
            + 2.5 * np.sin(2.0 * np.pi * xx / nx)
            + 1.5 * np.cos(2.0 * np.pi * yy / ny)).astype(float)


def _bedrock_depth(nx: int, ny: int, mean_depth: float, amp: float) -> np.ndarray:
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    d = mean_depth + amp * np.sin(2.0 * np.pi * xx / nx) * np.cos(1.5 * np.pi * yy / ny)
    return np.clip(d, 2.0, None)


def _build_layers(nx: int, ny: int, mean_depth: float, amp: float) -> Tuple[np.ndarray, np.ndarray]:
    top = _surface_elevation(nx, ny)
    depth = _bedrock_depth(nx, ny, mean_depth, amp)
    bot = np.empty((N_LAYERS, ny, nx), dtype=float)
    for i, frac in enumerate(LAYER_DEPTH_FRAC):
        bot[i] = top - frac * depth
    return top, bot


def _porosity_grid(nx: int, ny: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = LAYER_POROSITY[:, None, None] * np.ones((N_LAYERS, ny, nx))
    base += 0.01 * rng.standard_normal((N_LAYERS, ny, nx))
    return np.clip(base, 0.02, 0.6)


def _water_content(scenario: Dict, porosity: np.ndarray, n_time: int) -> np.ndarray:
    n_layers, ny, nx = porosity.shape
    if scenario.get("timelapse"):
        sat_dry = np.asarray(scenario["sat_dry"], dtype=float)
        sat_wet = np.asarray(scenario["sat_wet"], dtype=float)
        wc = np.empty((n_time, n_layers, ny, nx), dtype=float)
        for t in range(n_time):
            frac = t / max(1, n_time - 1)          # infiltration progress 0..1
            front = frac * n_layers                 # how many layers are wetted
            for L in range(n_layers):
                wet = float(np.clip(front - L, 0.0, 1.0))
                sat = sat_dry[L] + wet * (sat_wet[L] - sat_dry[L])
                wc[t, L] = sat * porosity[L]
        return wc
    sat = np.asarray(scenario["sat"], dtype=float)
    return (sat[:, None, None] * porosity)[None]   # (1, n_layers, ny, nx)


def write_hydro_bundle(key: str, cfg: Dict, data_root: Path) -> Path:
    scenario = SCENARIOS[key]
    out = data_root / key
    out.mkdir(parents=True, exist_ok=True)
    top, bot = _build_layers(cfg["nx"], cfg["ny"], scenario["mean_depth"], scenario["amp"])
    porosity = _porosity_grid(cfg["nx"], cfg["ny"])
    wc = _water_content(scenario, porosity, cfg["n_time"])
    wc_to_save = wc if scenario.get("timelapse") else wc[0]
    np.save(out / "top.npy", top)
    np.save(out / "bot.npy", bot)
    np.save(out / "Porosity.npy", porosity)
    np.save(out / "Watercontent.npy", wc_to_save)
    print(f"  [hydro] {key}: top{top.shape} bot{bot.shape} por{porosity.shape} "
          f"wc{wc_to_save.shape} theta[{wc.min():.3f},{wc.max():.3f}]")
    return out


# ---------------------------------------------------------------------------
# 2. Shared mesh construction (mirrors hydro_pipeline.run_hydro_forward).
# ---------------------------------------------------------------------------
def _build_mesh(profile: Dict, cfg: Dict):
    """Build the 2D mesh, per-cell markers, and the layer-boundary indices."""
    L_profile = profile["L_profile"]
    structure = profile["structure"]
    n_bounds = structure.shape[0]
    mid_idx = max(1, min(4, n_bounds // 3))
    bot_idx = max(mid_idx + 1, min(12, n_bounds - 2))
    surface, line1, line2 = create_surface_lines(
        L_profile=L_profile, structure=structure,
        top_idx=0, mid_idx=mid_idx, bot_idx=bot_idx)
    mesh, _ = MeshCreator(quality=32, area=cfg["mesh_area"]).create_from_layers(
        surface=surface, layers=[line1, line2],
        bottom_depth=float(np.min(line2[:, 1]) - 10.0))
    markers = assign_three_layer_markers(mesh, line1, line2, 0, 3, 2)
    return mesh, markers, [0, mid_idx, bot_idx]


# ---------------------------------------------------------------------------
# 3. ERT -> Water Content bundle: the real hydro_to_ert forward, per time step.
# ---------------------------------------------------------------------------
def build_geo_bundle(hydro_dir: Path, cfg: Dict, geo_dir: Path, log: Callable) -> None:
    ctx = {"hydro_data_dir": str(hydro_dir)}
    p1, p2 = GEO_PROFILE["p1"], GEO_PROFILE["p2"]
    n_time = cfg["n_time"]

    prof0 = extract_profile(ctx, {"snapshot_index": 0, "num_samples": cfg["num_samples"]}, p1, p2)
    mesh, markers, layer_idx = _build_mesh(prof0, cfg)
    por_mesh = interpolate_profile_to_mesh(
        prof0["porosity_profile"], prof0["structure"], prof0["L_profile"], mesh)

    columns: List[np.ndarray] = []
    for t in range(n_time):
        prof = extract_profile(ctx, {"snapshot_index": t, "num_samples": cfg["num_samples"]}, p1, p2)
        wc_mesh = interpolate_profile_to_mesh(
            prof["water_content_profile"], prof["structure"], prof["L_profile"], mesh)
        ert_data, resistivity_model = hydro_to_ert(
            water_content=wc_mesh, porosity=por_mesh, mesh=mesh,
            profile_interpolator=prof["interpolator"], layer_idx=layer_idx,
            structure=prof["structure"], marker_labels=MARKER_LABELS,
            rho_parameters=RHO_PARAMETERS, electrode_spacing=1.0, electrode_start=0.0,
            num_electrodes=ERT_NUM_ELECTRODES, scheme_name=ERT_SCHEME, noise_level=0.03,
            abs_error=0.0, rel_error=0.03, mesh_markers=markers, verbose=False, seed=7 + t)
        columns.append(np.asarray(resistivity_model, dtype=float))
        log(f"  [ert]  t={t}: {ert_data.size()} data, "
            f"rho[{columns[-1].min():.0f},{columns[-1].max():.0f}] ohm.m")

    resmodel = np.column_stack(columns)                       # (n_cells, n_time)
    index_marker = np.asarray(markers, dtype=int).copy()
    for src, dst in GEO_MARKER_REMAP.items():
        index_marker[index_marker == src] = dst

    y_cell = np.asarray(mesh.cellCenters(), dtype=float)[:, 1]
    depth = y_cell.max() - y_cell
    span = float(np.ptp(depth)) or 1.0
    coverage = np.repeat(np.exp(-3.0 * depth / span)[None, :], n_time, axis=0)

    geo_dir.mkdir(parents=True, exist_ok=True)
    mesh.save(str(geo_dir / "mesh_res.bms"))
    np.save(geo_dir / "resmodel.npy", resmodel)
    np.save(geo_dir / "index_marker.npy", index_marker)
    np.save(geo_dir / "all_coverage.npy", coverage)
    uniq, counts = np.unique(index_marker, return_counts=True)
    print(f"  [geo]  {geo_dir.name}: mesh {mesh.cellCount()} cells, resmodel{resmodel.shape}, "
          f"rho[{resmodel.min():.0f},{resmodel.max():.0f}] ohm.m, "
          f"markers {dict(zip(uniq.tolist(), counts.tolist()))}, coverage{coverage.shape}")


# ---------------------------------------------------------------------------
# 4. Seismic -> Structure velocity sections: the real hydro_to_srt forward.
# ---------------------------------------------------------------------------
def build_seismic_lines(hydro_dir: Path, cfg: Dict, seismic_root: Path, log: Callable) -> None:
    ctx = {"hydro_data_dir": str(hydro_dir)}
    snapshot = max(0, cfg["n_time"] // 2)
    for i, spec in enumerate(SEISMIC_LINES, start=1):
        prof = extract_profile(
            ctx, {"snapshot_index": snapshot, "num_samples": cfg["num_samples"]},
            spec["p1"], spec["p2"])
        mesh, markers, layer_idx = _build_mesh(prof, cfg)
        por_mesh = interpolate_profile_to_mesh(
            prof["porosity_profile"], prof["structure"], prof["L_profile"], mesh)
        wc_mesh = interpolate_profile_to_mesh(
            prof["water_content_profile"], prof["structure"], prof["L_profile"], mesh)
        srt_data, velocity_model = hydro_to_srt(
            water_content=wc_mesh, porosity=por_mesh, mesh=mesh,
            profile_interpolator=prof["interpolator"], layer_idx=layer_idx,
            structure=prof["structure"], marker_labels=MARKER_LABELS,
            vel_parameters=VEL_PARAMETERS, sensor_spacing=1.0, sensor_start=0.0,
            num_sensors=SRT_NUM_SENSORS, shot_distance=5, noise_level=0.02,
            noise_abs=1e-5, mesh_markers=markers, verbose=False, seed=7)
        vel = np.asarray(velocity_model, dtype=float)
        line_dir = seismic_root / f"line{i}"
        line_dir.mkdir(parents=True, exist_ok=True)
        mesh.save(str(line_dir / "velmesh.bms"))
        np.save(line_dir / "Vinvmodel.npy", vel)
        reg = vel[np.isin(markers, [0, 3])]
        bed = vel[markers == 2]
        reg_max = float(reg.max()) if reg.size else float("nan")
        bed_min = float(bed.min()) if bed.size else float("nan")
        print(f"  [srt]  line{i} ({spec['name']}): {srt_data.size()} data, mesh {mesh.cellCount()} cells, "
              f"Vp[{vel.min():.0f},{vel.max():.0f}] m/s (regolith<= {reg_max:.0f} < 1200 < {bed_min:.0f} =bedrock)")


# ---------------------------------------------------------------------------
# 5. Seismic 3D volumes: the true model (same rock-physics as hydro_to_srt),
#    plus the Kriging reconstruction from the 2D lines.
# ---------------------------------------------------------------------------
def _velocity_from_rock_physics(sat: np.ndarray, por: np.ndarray, marker: np.ndarray) -> np.ndarray:
    """Per-cell velocity via the same models hydro_to_srt uses: Hertz-Mindlin for
    the top marker, DEM for the middle/bottom markers."""
    v = np.zeros_like(sat, dtype=float)
    m0 = marker == 0
    if np.any(m0):
        p = VEL_PARAMETERS["top"]
        vh, vl = _HM.calculate_velocity(
            porosity=por[m0], saturation=sat[m0], bulk_modulus=p["bulk_modulus"],
            shear_modulus=p["shear_modulus"], mineral_density=p["mineral_density"], depth=p["depth"])
        v[m0] = (np.asarray(vh, float) + np.asarray(vl, float)) / 2.0
    for mk, key in ((3, "mid"), (2, "bot")):
        mm = marker == mk
        if np.any(mm):
            p = VEL_PARAMETERS[key]
            _, _, vp = _DEM.calculate_velocity(
                porosity=por[mm], saturation=sat[mm], bulk_modulus=p["bulk_modulus"],
                shear_modulus=p["shear_modulus"], mineral_density=p["mineral_density"],
                aspect_ratio=p["aspect_ratio"])
            v[mm] = np.asarray(vp, float)
    return v


def build_true_3d_volume(hydro_dir: Path, cfg: Dict, out_dir: Path, snapshot: int) -> np.ndarray:
    """Build the ground-truth 3D velocity volume from the 3D hydrology model over
    the seismic survey box; the 2D lines are slices of it. Saves a VTK volume
    (viewable in Mesh 3D) + .npy, and returns the true bedrock-depth map."""
    top = np.load(hydro_dir / "top.npy")
    bot = np.load(hydro_dir / "bot.npy")
    por = np.load(hydro_dir / "Porosity.npy")
    wc = np.load(hydro_dir / "Watercontent.npy")
    wc_snap = wc[snapshot] if wc.ndim == 4 else wc                 # (n_layers, ny, nx)

    x0, x1, y0, y1 = SEIS_BBOX
    sy, sx = slice(y0, y1 + 1), slice(x0, x1 + 1)
    top_b, bot_b = top[sy, sx], bot[:, sy, sx]
    por_b, wc_b = por[:, sy, sx], wc_snap[:, sy, sx]
    ny_b, nx_b = top_b.shape
    bounds = np.concatenate([top_b[None], bot_b], axis=0)          # (n_layers+1, ny_b, nx_b)
    line1, line2 = bounds[2], bounds[5]                            # mid + bedrock interfaces

    z_min, z_max = float(bot_b[-1].min()) - 2.0, float(top_b.max())
    z = np.linspace(z_min, z_max, cfg["vol_nz"])
    dz = float(z[1] - z[0])
    vol = np.full((cfg["vol_nz"], ny_b, nx_b), np.nan, dtype=float)
    for k, zk in enumerate(z):
        marker = np.where(zk >= line1, 0, np.where(zk >= line2, 3, 2))
        L = np.clip((bounds >= zk).sum(axis=0) - 1, 0, N_LAYERS - 1)
        wc_L = np.take_along_axis(wc_b, L[None], 0)[0]
        por_L = np.take_along_axis(por_b, L[None], 0)[0]
        sat = np.clip(wc_L / np.maximum(por_L, 1e-6), 0.0, 1.0)
        v = _velocity_from_rock_physics(sat, por_L, marker)
        v[zk > top_b] = np.nan                                     # air above the surface
        vol[k] = v

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "velocity_true.npy", vol)
    bedrock_depth_true = top_b - line2                             # positive = depth to bedrock
    np.save(out_dir / "bedrock_depth_true.npy", bedrock_depth_true)
    try:
        import pyvista as pv
        grid = pv.ImageData(dimensions=(nx_b, ny_b, cfg["vol_nz"]),
                            spacing=(1.0, 1.0, dz), origin=(float(x0), float(y0), z_min))
        grid.point_data["Velocity"] = vol.ravel(order="C")
        grid.save(str(out_dir / "velocity_true.vtk"))
    except Exception as exc:  # noqa: BLE001 - VTK export is best-effort
        print(f"  [3d] true VTK export skipped: {exc}")
    print(f"  [3d] true volume {vol.shape}: Vp[{np.nanmin(vol):.0f},{np.nanmax(vol):.0f}] m/s, "
          f"bedrock depth[{bedrock_depth_true.min():.1f},{bedrock_depth_true.max():.1f}] m")
    return bedrock_depth_true


def build_kriged_3d(seismic_root: Path, cfg: Dict, out_dir: Path, log: Callable) -> Dict:
    """Reconstruct a 3D velocity volume from the three 2D lines via ordinary
    Kriging (gstools), using the same engine the Seismic -> Structure module runs
    when 'kriging' is selected. Copies the outputs into ``out_dir``."""
    import shutil
    from PyHydroGeophysX.qt_apps.seismic3d_pipeline import build_3d_model, DEFAULT_KRIGING

    lines = []
    for i, spec in enumerate(SEISMIC_LINES, start=1):
        d = seismic_root / f"line{i}"
        lines.append({"name": spec["name"], "mesh": str(d / "velmesh.bms"),
                      "velocity": str(d / "Vinvmodel.npy"),
                      "x0": float(spec["p1"][0]), "y0": float(spec["p1"][1]),
                      "x1": float(spec["p2"][0]), "y1": float(spec["p2"][1])})
    params = {
        "output_dir": str(out_dir), "lines": lines, "threshold": 1200.0, "interval": 4.0,
        "grid_resolution": cfg["grid_res"], "depth": 50.0,
        "n_layers": max(8, cfg["vol_nz"] // 2), "interp_method": "kriging",
        "z_scale": 1.0, "max_velocity_points": 40000, "kriging": DEFAULT_KRIGING,
    }
    result = build_3d_model({"output_dir": str(out_dir)}, params, log=log)
    src = Path(result["output_dir"])
    for src_name, dst_name in (("velocity_volume.npy", "velocity_kriged.npy"),
                               ("seismic_3d_model.vtk", "velocity_kriged.vtk"),
                               ("bedrock_depth.npy", "bedrock_depth_kriged.npy")):
        sp = src / src_name
        if sp.exists():
            shutil.copy2(sp, out_dir / dst_name)
    print(f"  [3d] kriged volume ({result['interp_method']}): "
          f"Vp[{result['velocity_range'][0]:.0f},{result['velocity_range'][1]:.0f}] m/s, "
          f"bedrock depth[{result['bedrock_depth_range'][0]:.1f},"
          f"{result['bedrock_depth_range'][1]:.1f}] m")
    return result


def build_seismic3d(hydro_dir: Path, seismic_root: Path, cfg: Dict,
                    out_dir: Path, snapshot: int, log: Callable) -> None:
    bd_true = build_true_3d_volume(hydro_dir, cfg, out_dir, snapshot)
    build_kriged_3d(seismic_root, cfg, out_dir, log)
    try:
        import matplotlib.pyplot as plt
        bd_kriged = np.load(out_dir / "bedrock_depth_kriged.npy")
        x0, x1, y0, y1 = SEIS_BBOX
        vmin = float(min(np.nanmin(bd_true), np.nanmin(bd_kriged)))
        vmax = float(max(np.nanmax(bd_true), np.nanmax(bd_kriged)))
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
        for ax, arr, title in ((axes[0], bd_true, "True (from 3D model)"),
                               (axes[1], bd_kriged, "Kriged (from 2D lines)")):
            im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
                           extent=[x0, x1, y0, y1], aspect="auto")
            ax.set_title(f"Bedrock depth - {title}")
            ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        fig.colorbar(im, ax=axes, label="Depth to bedrock (m)")
        p = out_dir / "bedrock_depth_true_vs_kriged.png"
        fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"  [3d] comparison figure: {p.name}")
    except Exception as exc:  # noqa: BLE001 - comparison figure is best-effort
        print(f"  [3d] comparison figure skipped: {exc}")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all",
                   help="Which hydrology scenario(s) to build (default: all).")
    p.add_argument("--n-time", type=int, default=None,
                   help="Override the number of time steps for the time-lapse scenario.")
    p.add_argument("--quick", action="store_true",
                   help="Fast low-resolution run for debugging / CI.")
    p.add_argument("--output-dir", default=None,
                   help="Base output directory (default: the repo's examples/).")
    p.add_argument("--skip-ert", action="store_true", help="Do not build the ERT bundle.")
    p.add_argument("--skip-seismic", action="store_true", help="Do not build the seismic lines.")
    p.add_argument("--skip-3d", action="store_true",
                   help="Build the 2D seismic lines but skip the 3D volumes (true + kriged).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # The grid (nx, ny) is cheap and the profile endpoints index into it, so keep
    # it fixed; --quick only relaxes the knobs that cost time (profile samples,
    # mesh density -> forward cost, time steps, 3D grid).
    cfg = ({"nx": 120, "ny": 90, "num_samples": 110, "mesh_area": 3.0, "n_time": 3,
            "grid_res": 25, "vol_nz": 20} if args.quick
           else {"nx": 120, "ny": 90, "num_samples": 200, "mesh_area": 1.0, "n_time": 8,
                 "grid_res": 40, "vol_nz": 40})
    if args.n_time is not None:
        cfg["n_time"] = max(1, args.n_time)

    base = Path(args.output_dir).resolve() if args.output_dir else (REPO_ROOT / "examples")
    data_root = base / "data" / "synthetic"
    geo_dir = base / "results" / "synthetic_Structure_WC"
    seismic_root = base / "results" / "synthetic_seismic"
    seismic3d_dir = base / "results" / "synthetic_seismic3d"

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    print(f"Repository root: {REPO_ROOT}")
    print(f"Config: {cfg}; output base: {base}")
    print("Building synthetic hydrology grids ...")
    built: Dict[str, Path] = {key: write_hydro_bundle(key, cfg, data_root) for key in keys}

    if DRIVER not in built:
        print(f"\nDriver scenario '{DRIVER}' not selected; skipping ERT/seismic "
              f"(they are derived from it). Use --scenario all or --scenario {DRIVER}.")
        return 0

    if not args.skip_ert:
        print("\nBuilding ERT -> Water Content bundle (hydro_to_ert, time-lapse) ...")
        build_geo_bundle(built[DRIVER], cfg, geo_dir, print)
    if not args.skip_seismic:
        print("\nBuilding Seismic -> Structure velocity sections (hydro_to_srt, 2D) ...")
        build_seismic_lines(built[DRIVER], cfg, seismic_root, print)
        if not args.skip_3d:
            print("\nBuilding Seismic 3D volumes (true model + Kriging reconstruction) ...")
            build_seismic3d(built[DRIVER], seismic_root, cfg, seismic3d_dir,
                            max(0, cfg["n_time"] // 2), print)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
