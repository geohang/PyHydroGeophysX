"""High-level 1D electromagnetic workflows and compatibility facade."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop, utc_now as _utc_now
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.data_processing.em1d import (
    TEMCOMPANY_MOMENTS,
    _normalise_temcompany_moment,
    _response_on_times,
    is_temcompany_source,
    load_line_geometry,
    load_sounding,
    load_temcompany_sounding,
)
from PyHydroGeophysX.forward.em1d import (
    DEFAULT_FDEM,
    DEFAULT_MODEL,
    DEFAULT_TDEM,
    _fdem_config,
    _tdem_config,
    fdem_forward,
    model_arrays,
    model_depth_profile,
    tdem_forward,
)
from PyHydroGeophysX.inversion.em1d import (
    DEFAULT_INVERSION,
    _inversion_layer_thicknesses,
    fdem_invert,
    tdem_invert,
    tdem_joint_invert,
)

LogFn = Callable[[str], None]

METHODS = ("FDEM", "TDEM")


def backend_status(method: Optional[str] = None) -> Dict[str, Any]:
    """Report whether the requested EM forward/inversion backend is usable.

    The check imports the same method-specific forward class used by inversion.
    This prevents the UI and AQUAH from announcing a background inversion that
    cannot start because SimPEG or one of its runtime dependencies is missing.
    """
    methods = (method,) if method is not None else METHODS
    result: Dict[str, Dict[str, Any]] = {}
    for selected in methods:
        if selected not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {selected!r}.")
        try:
            if selected == "FDEM":
                from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling  # noqa: F401
            else:
                from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling  # noqa: F401
            result[selected] = {"available": True, "error": ""}
        except Exception as exc:  # noqa: BLE001 - optional numerical backend
            result[selected] = {"available": False, "error": str(exc)}
    if method is not None:
        return result[method]
    return {
        "available": all(item["available"] for item in result.values()),
        "methods": result,
    }


def example_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the desktop EM examples and their documented settings.

    Prefer the source checkout for development and fall back to the compact
    package-data copy installed from a wheel.
    """
    package_root = Path(__file__).resolve().parents[1]
    checkout_root = package_root.parent / "examples" / "data" / "EM"
    bundled_root = package_root / "data" / "em_examples"
    root = checkout_root if checkout_root.is_dir() else bundled_root
    east_river = root / "EastRiver_VTEM"
    return {
        "east_river_vtem": {
            "label": "East River VTEM (recommended)",
            "method": "TDEM",
            "path": east_river / "eastriver_vtem_line22030.csv",
            "geometry_path": east_river / "eastriver_vtem_line22030_geometry.csv",
            "params": {
                "source_radius": 13.0, "height": 82.0, "orientation": "z",
                "waveform": "step_off", "n_layers": 13, "min_thickness": 8.0,
                "max_thickness": 55.0, "smoothness": 0.5, "rel_error": 0.08,
                "max_iterations": 10, "ref_resistivity": 320.0,
                "auto_scale": False, "max_soundings": 22,
            },
            "note": "Configured VTEM line with companion geometry and reference calibration.",
        },
        "skytem_bhmar": {
            "label": "SkyTEM BHMAR (quick preview)",
            "method": "TDEM",
            "path": root / "skytem_bhmar_tdem.csv",
            "geometry_path": root / "skytem_bhmar_geometry.csv",
            "params": {"auto_scale": True, "ref_resistivity": 0.0, "max_soundings": 5},
            "note": "Relative airborne TDEM preview; system calibration is not supplied.",
        },
        "synthetic_fdem": {
            "label": "Synthetic FDEM (1D)",
            "method": "FDEM",
            "path": root / "synthetic_fdem.csv",
            "params": {
                "source_radius": 10.0, "tx_rx_sep": 10.0, "height": 30.0,
                "orientation": "z", "component": "secondary", "waveform": "dipole",
                "auto_scale": False, "ref_resistivity": 0.0,
            },
            "note": "Deterministic 3% noisy response of 50/200/20 ohm-m layers (10/20 m).",
        },
        "synthetic_tem_lci": {
            "label": "Synthetic LM+HM line (LCI)",
            "method": "TDEM",
            "path": root / "synthetic_tem_lci",
            "params": {
                "tem_moment": "LM+HM", "data_scale": 1.0,
                "auto_scale": False, "ref_resistivity": 0.0,
                "max_iterations": 6, "max_soundings": 9,
                "lateral_smoothness": 1.3, "lci_passes": 1,
            },
            "note": (
                "Nine-station synthetic LM+HM line with 3% deterministic noise "
                "and a known smooth lateral resistivity trend."
            ),
        },
    }


def estimate_data_scale(path: str, method: str, geom: Dict[str, Any], *,
                        max_soundings: int = 8, log: LogFn = _noop) -> float:
    """Estimate the amplitude calibration (``data_scale``) for normalized data.

    Normalized airborne responses (e.g. moment-normalized dB/dt) differ from the
    workbench's 1D forward by a near-constant amplitude factor. This fits each
    sounding's decay SHAPE to a grid of half-space forward responses at the current
    geometry and takes the geometric-mean amplitude ratio ``forward/observed`` at
    the best-fitting resistivity. Returns ``1.0`` if it cannot be estimated (so the
    caller can fall back to no scaling).
    """
    moment = str(geom.get("tem_moment", "HM"))
    use_flags = bool(geom.get("use_project_flags", True))
    tail_cut = geom.get("tail_max_relative_std")
    try:
        head = load_sounding(path, method, sounding=0, moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
    except Exception as exc:  # noqa: BLE001
        log(f"Auto-calibration skipped ({exc}); using data_scale = 1.0")
        return 1.0
    n_total = int(head.get("n_soundings", 1))
    probe = np.unique(np.linspace(0, n_total - 1, min(int(max_soundings), n_total)).astype(int))

    try:
        if method == "TDEM":
            from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
            abscissa = np.asarray(head["times"], dtype=float).ravel()
            cfg = _tdem_config(geom, abscissa)
            grid = []
            for R in np.geomspace(25.0, 3000.0, 20):
                md = TDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=cfg)
                grid.append(
                    float(geom.get("response_sign", 1.0))
                    * np.asarray(md.forward(np.array([1.0 / R, 1.0 / R])),
                                 dtype=float).ravel()[:abscissa.size]
                )
            preds = np.asarray(grid)

            def observed(s):
                return _response_on_times(
                    load_sounding(path, method, sounding=int(s), moment=moment, use_flags=use_flags, max_relative_std=tail_cut),
                    abscissa,
                )
        else:
            from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
            abscissa = np.asarray(head["frequencies"], dtype=float).ravel()
            cfg = _fdem_config(geom, abscissa)
            grid = []
            for R in np.geomspace(25.0, 3000.0, 20):
                md = FDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=cfg)
                resp = np.asarray(md.forward(np.array([1.0 / R, 1.0 / R]))).ravel()
                if resp.size == 2 * abscissa.size and not np.iscomplexobj(resp):
                    resp = resp[0::2] + 1j * resp[1::2]
                grid.append(np.abs(np.asarray(resp, dtype=complex).ravel()[:abscissa.size]))
            preds = np.asarray(grid)

            def observed(s):
                d = load_sounding(path, method, sounding=int(s), moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
                return np.abs(np.asarray(d["real"], float) + 1j * np.asarray(d["imag"], float))
    except Exception as exc:  # noqa: BLE001
        log(f"Auto-calibration skipped ({exc}); using data_scale = 1.0")
        return 1.0

    ks: List[float] = []
    for s in probe:
        obs = observed(s)
        finite = obs > 0
        best = None
        for pr in preds:
            mm = finite & (pr > 0)
            if mm.sum() < 5:
                continue
            lr = np.log10(pr[mm]) - np.log10(obs[mm])  # = log10(scale) at this half-space R
            resid = float(lr.std())
            if best is None or resid < best[0]:
                best = (resid, 10.0 ** float(lr.mean()))
        if best is not None:
            ks.append(best[1])
    if not ks:
        return 1.0
    k = float(np.exp(np.mean(np.log(ks))))
    log(f"Estimated data_scale = {k:.4g} from {len(ks)} soundings.")
    return k


def calibrate_to_reference(path: str, method: str, geom: Dict[str, Any], inv: Dict[str, Any],
                           ref_resistivity: float, *, max_probe: int = 6,
                           log: LogFn = _noop) -> float:
    """Find the ``data_scale`` that makes the recovered near-surface resistivity match
    a known/expected value.

    The amplitude scale and the absolute resistivity level are degenerate — the EM
    data alone cannot fix the level (any ``data_scale`` fits the data, with the
    resistivity shifting to compensate). This breaks the degeneracy with EXTERNAL
    information: the user supplies ``ref_resistivity`` (a known background, e.g. from
    a borehole or regional geology), and this inverts a few probe soundings at two
    trial scales, fits the near-surface resistivity's log-linear response to the
    scale, and solves for the scale that yields ``ref_resistivity``. Returns the
    current ``data_scale`` unchanged if calibration is not possible.
    """
    ref = float(ref_resistivity)
    current = float(inv.get("data_scale", 1.0))
    if ref <= 0:
        return current
    moment = str(geom.get("tem_moment", "HM"))
    use_flags = bool(geom.get("use_project_flags", True))
    tail_cut = geom.get("tail_max_relative_std")
    try:
        head = load_sounding(path, method, sounding=0, moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
        n_total = int(head.get("n_soundings", 1))
        probe = np.unique(np.linspace(0, n_total - 1, min(int(max_probe), n_total)).astype(int))
        if method == "TDEM":
            from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
            abscissa = np.asarray(head["times"], dtype=float).ravel()
            md = TDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=_tdem_config(geom, abscissa))
            pred = (
                float(geom.get("response_sign", 1.0))
                * np.asarray(md.forward(np.array([1.0 / ref, 1.0 / ref])),
                             dtype=float).ravel()[:abscissa.size]
            )

            def observed(s):
                return _response_on_times(
                    load_sounding(path, method, sounding=int(s), moment=moment, use_flags=use_flags, max_relative_std=tail_cut),
                    abscissa,
                )
        else:
            from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
            abscissa = np.asarray(head["frequencies"], dtype=float).ravel()
            md = FDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=_fdem_config(geom, abscissa))
            resp = np.asarray(md.forward(np.array([1.0 / ref, 1.0 / ref]))).ravel()
            if resp.size == 2 * abscissa.size and not np.iscomplexobj(resp):
                resp = resp[0::2] + 1j * resp[1::2]
            pred = np.abs(np.asarray(resp, dtype=complex).ravel()[:abscissa.size])

            def observed(s):
                d = load_sounding(path, method, sounding=int(s), moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
                return np.abs(np.asarray(d["real"], float) + 1j * np.asarray(d["imag"], float))
    except Exception as exc:  # noqa: BLE001
        log(f"Reference calibration unavailable ({exc}); kept data_scale.")
        return current

    # Tie the data AMPLITUDE to a half-space at ``ref``: data_scale = geomean(pred/obs).
    # Deterministic and stable (one forward, no inversion). The recovered model is not
    # forced exactly to ``ref`` (a half-space differs from the layered earth), but the
    # absolute level is pinned to a known value the same way for every dataset.
    ks = []
    for s in probe:
        obs = observed(s)
        m = (pred > 0) & (obs > 0)
        if m.sum() >= 5:
            ks.append(10.0 ** float((np.log10(pred[m]) - np.log10(obs[m])).mean()))
    if not ks:
        return current
    k = float(np.clip(np.exp(np.mean(np.log(ks))), 1e-4, 1e4))
    log(f"Reference calibration to a half-space at {ref:.0f} ohm-m: data_scale = {k:.4g}.")
    return k


def _latest_gate(data: Dict[str, Any]) -> Optional[float]:
    """The last time channel this station actually carries, over all moments.

    A joint station's ``times`` entry holds one preview moment only, so reading
    it would understate a station whose latest gates are in the other moment,
    and would say nothing at all about stations that differ from the first one
    on the line.
    """
    moments = data.get("moments") or {}
    times = ([np.asarray(item.get("times", []), dtype=float).ravel()
              for item in moments.values()]
             if moments else [np.asarray(data.get("times", []), dtype=float).ravel()])
    usable = [array.max() for array in times if array.size]
    return float(max(usable)) if usable else None


def _sounding_data_count(data: Dict[str, Any], method: str) -> int:
    """Number of residual entries one sounding contributes.

    FDEM counts real and imaginary parts separately; a joint TDEM station counts
    every gate of every moment it carries.
    """
    moments = data.get("moments", {})
    if moments:
        return int(sum(np.asarray(item.get("times", [])).size
                       for item in moments.values()))
    if str(method).upper() == "FDEM":
        return int(2 * np.asarray(data.get("frequencies", [])).size)
    return int(np.asarray(data.get("times", [])).size)


def invert_line(path: str, method: str, geom: Dict[str, Any], inv: Dict[str, Any],
                *, spacing: float = 50.0, positions: Optional[np.ndarray] = None,
                heights: Optional[np.ndarray] = None, max_soundings: int = 12,
                doi_blank: bool = True, doi_factor: float = 0.5, ref_resistivity: float = 0.0,
                out_dir: Optional[Path] = None,
                initial_models: Optional[np.ndarray] = None,
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert a line on a shared fixed-layer grid.

    ``inv["lci_mode"]`` selects how the soundings are coupled:

    ``simultaneous`` (the default whenever ``lateral_smoothness`` is positive)
        Solves the whole line as one system, with the lateral constraint part of
        what is being minimized. See
        :mod:`PyHydroGeophysX.inversion.em1d_lci`.
    ``sequential``
        The older block-coordinate passes: each station is re-inverted on its
        own against the distance-weighted model its neighbours had at the end of
        the previous pass. Kept because it needs no analytic Jacobian, so it
        still runs against a forward operator that cannot supply one.
    ``off``
        Independent 1D inversion per sounding, no lateral coupling.

    The models are laid side by side to form a ``resistivity(position, depth)``
    section ready for :meth:`Model3DView.show_model`: ``edges = (ex, ey, ez)``
    (``ez`` is elevation, increasing upward) and ``model3d`` of shape
    ``(n_pos, 1, n_depth)``.

    ``positions`` gives the along-line distance of each sounding (the section
    x-axis); ``heights`` overrides the sensor height per sounding. When
    ``doi_blank`` is set, cells below a per-sounding depth of investigation
    (a diffusion-depth estimate scaled by ``doi_factor``) are blanked (NaN) so the
    unconstrained deep part of an early-time sounding is not shown as railed.

    Two settings answer a chi-squared that stays high, the same pair the ERT
    module offers. ``inv["auto_lambda"]`` re-solves the line at other smoothness
    weights to reach ``target_chi2``. ``inv["reject_outliers"]`` drops the gates
    the converged model cannot explain (beyond ``outlier_threshold`` sigma, over
    ``outlier_passes`` cycles, never below ``min_data_fraction`` of the gates)
    and solves again; what it removed is reported under ``result["outliers"]``.
    They address different causes, so they can be used together: relaxing the
    smoothness helps when the model is too stiff for the data, rejection helps
    when a minority of gates are simply wrong.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    moment = (
        _normalise_temcompany_moment(str(geom.get("tem_moment", "HM")))
        if is_temcompany_source(path) else str(geom.get("tem_moment", "HM"))
    )
    use_flags = bool(geom.get("use_project_flags", True))
    tail_cut = geom.get("tail_max_relative_std")
    head = load_sounding(path, method, sounding=0, moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
    joint = method == "TDEM" and bool(head.get("moments"))
    invert = (
        fdem_invert if method == "FDEM"
        else tdem_joint_invert if joint
        else tdem_invert
    )
    n_total = int(head.get("n_soundings", 1))
    n_pos = min(int(max_soundings), max(1, n_total))

    # The coupled solve needs at least two stations to tie together, and a
    # lateral weight to tie them with.
    lci_mode = str(inv.get("lci_mode", "simultaneous")).strip().lower()
    if lci_mode not in {"simultaneous", "sequential", "off"}:
        lci_mode = "simultaneous"
    simultaneous = (
        lci_mode == "simultaneous"
        and float(inv.get("lateral_smoothness", 0.0)) > 0.0
        and n_pos >= 2
    )
    mode = ("simultaneous LCI" if simultaneous
            else "block-coordinate LCI" if (lci_mode == "sequential" and joint)
            else f"{method} independent 1D")
    log(f"Line inversion: {n_pos} of {n_total} soundings ({mode})")

    # Calibrate the amplitude scale to a known reference resistivity if requested
    # (breaks the data_scale <-> resistivity-level degeneracy with external info).
    if ref_resistivity and float(ref_resistivity) > 0:
        inv = {**inv, "data_scale": calibrate_to_reference(
            path, method, geom, inv, float(ref_resistivity), log=log)}
    data_scale_used = float(inv.get("data_scale", 1.0))

    # Shared layer grid (identical for every sounding).
    n_layers = int(inv.get("n_layers", 15))
    thick = _inversion_layer_thicknesses(inv)
    pad = float(inv.get("max_thickness", 40.0))
    depth_edges = np.concatenate([[0.0], np.cumsum(thick), [float(np.sum(thick)) + pad]])
    ez = (-depth_edges)[::-1]  # elevation edges, increasing upward (surface at 0)

    hts = np.asarray(heights, dtype=float).ravel() if heights is not None else None
    model = np.full((n_pos, 1, n_layers), np.nan, dtype=float)
    surface_models = np.full((n_pos, n_layers), np.nan, dtype=float)
    chi2_list: List[float] = []
    data_count_list: List[int] = []
    datasets: List[Optional[Dict[str, Any]]] = [None] * n_pos
    geometries: List[Dict[str, Any]] = [geom] * n_pos
    lateral_requested = float(inv.get("lateral_smoothness", 0.0))
    warm_models = np.asarray(initial_models, dtype=float) if initial_models is not None else None
    use_warm_models = (
        warm_models is not None
        and warm_models.shape == (n_pos, n_layers)
        and np.all(np.isfinite(warm_models))
        and np.all(warm_models > 0.0)
    )
    if use_warm_models:
        surface_models[:, :] = warm_models
        log("Using supplied line models as the LCI warm start.")
    use_common_lci_start = (
        not use_warm_models
        and lateral_requested > 0.0
        and (joint or simultaneous)
    )
    if use_common_lci_start:
        start = float(inv.get("starting_resistivity", 100.0))
        surface_models[:, :] = max(start, 1.0)
        log(f"Using a common {start:g} ohm-m starting model for the LCI.")
    t_ref = f_ref = None  # last time / min frequency, for the DOI estimate
    for s in range(n_pos):
        try:
            data = load_sounding(path, method, sounding=s, moment=moment, use_flags=use_flags, max_relative_std=tail_cut)
            datasets[s] = data
            if t_ref is None and "times" in data and np.size(data["times"]):
                t_ref = float(np.asarray(data["times"]).ravel()[-1])
            if f_ref is None and "frequencies" in data and np.size(data["frequencies"]):
                f_ref = float(np.asarray(data["frequencies"]).ravel().min())
            geom_s = geom
            if hts is not None and s < hts.size and np.isfinite(hts[s]):
                geom_s = {**geom, "height": float(hts[s])}
            geometries[s] = geom_s
            if use_warm_models or use_common_lci_start:
                # The LCI supplies the model, so the per-sounding inversion is
                # skipped; only the data count is needed here.
                chi2_list.append(float("nan"))
                data_count_list.append(_sounding_data_count(data, method))
                continue
            result = invert(data, geom_s, inv, log=log)
            res = np.asarray(result["resistivity"], dtype=float).ravel()
            surface_models[s, :] = res
            model[s, 0, :] = res[::-1]  # deepest layer first to match ez ordering
            chi2_list.append(float(result.get("chi2", np.nan)))
            data_count_list.append(int(result.get("n_data", 0)))
            log(f"  sounding {s + 1}/{n_pos}: chi2={result.get('chi2', float('nan')):.3f}")
        except Exception as exc:  # noqa: BLE001 - keep the line going if one sounding fails
            chi2_list.append(float("nan"))
            data_count_list.append(0)
            log(f"  sounding {s + 1}/{n_pos} failed: {exc}")

    embedded_positions = np.asarray(head.get("positions", []), dtype=float).ravel()
    requested_positions = (
        np.asarray(positions, dtype=float).ravel()
        if positions is not None else embedded_positions
    )
    if requested_positions.size >= n_pos:
        pos_lci = requested_positions[:n_pos]
    else:
        pos_lci = np.arange(n_pos, dtype=float) * float(spacing)
    # Ground level per sounding, carried through for plotting only.
    embedded_elevation = np.asarray(head.get("elevation", []), dtype=float).ravel()
    surface_elevation = (
        embedded_elevation[:n_pos] if embedded_elevation.size >= n_pos
        else np.full(n_pos, np.nan, dtype=float)
    )
    embedded_lines = np.asarray(head.get("line_numbers", []), dtype=int).ravel()
    line_numbers = (
        embedded_lines[:n_pos]
        if embedded_lines.size >= n_pos else np.zeros(n_pos, dtype=int)
    )

    # LCI keeps model nodes even where the local gate set is too sparse for the
    # SimPEG time spline. Seed those nodes by log-resistivity interpolation along
    # their own survey line; subsequent passes update them from their neighbors.
    for line in np.unique(line_numbers):
        indices = np.flatnonzero(line_numbers == line)
        valid = indices[np.all(np.isfinite(surface_models[indices]), axis=1)]
        missing = indices[~np.all(np.isfinite(surface_models[indices]), axis=1)]
        if not valid.size or not missing.size:
            continue
        order = np.argsort(pos_lci[valid])
        xp = pos_lci[valid][order]
        for layer in range(n_layers):
            fp = np.log10(surface_models[valid, layer][order])
            surface_models[missing, layer] = np.power(
                10.0, np.interp(pos_lci[missing], xp, fp))
    model[:, 0, :] = surface_models[:, ::-1]

    lateral = lateral_requested
    lateral_weight_scale = max(float(inv.get("lateral_weight_scale", 1.0)), 0.0)
    lci_passes = max(0, int(inv.get("lci_passes", 1)))
    reference_distance = max(float(inv.get("reference_distance", 10.0)), 1e-6)
    lci_report: Dict[str, Any] = {}
    outlier_info: Dict[str, Any] = {"enabled": False}
    # Kept for the depth-of-investigation pass below, which reads the same
    # analytic Jacobian the coupled solver used.
    doi_blocks: Dict[int, Any] = {}
    if simultaneous:
        from PyHydroGeophysX.inversion.em1d import build_sounding_block
        from PyHydroGeophysX.inversion.em1d_lci import (
            invert_lci,
            invert_lci_rejecting_outliers,
        )

        usable = [s for s in range(n_pos) if datasets[s] is not None]
        sounding_blocks = []
        kept: List[int] = []
        for s in usable:
            try:
                sounding_blocks.append(build_sounding_block(
                    datasets[s], geometries[s], inv, method,
                    position=float(pos_lci[s]), line=int(line_numbers[s]),
                    label=f"sounding {s + 1}"))
                kept.append(s)
            except Exception as exc:  # noqa: BLE001 - one bad station is not fatal
                log(f"  sounding {s + 1} excluded from the LCI: {exc}")
        if len(kept) < 2:
            # The per-sounding pass was skipped on the assumption the LCI would
            # supply the models, so it has to run now or nothing is inverted.
            log("  Fewer than two usable soundings; falling back to independent 1D.")
            simultaneous = False
            for s in usable:
                try:
                    result = invert(datasets[s], geometries[s], inv, log=log)
                    surface_models[s, :] = np.asarray(
                        result["resistivity"], dtype=float).ravel()
                    chi2_list[s] = float(result.get("chi2", np.nan))
                    log(f"  sounding {s + 1}/{n_pos}: "
                        f"chi2={result.get('chi2', float('nan')):.3f}")
                except Exception as exc:  # noqa: BLE001
                    log(f"  sounding {s + 1}/{n_pos} failed: {exc}")
            model[:, 0, :] = surface_models[:, ::-1]
        else:
            log(f"Simultaneous LCI: {len(kept)} soundings, lateral="
                f"{lateral:g}, vertical={float(inv.get('smoothness', 0.3)):g}")
            warm = (surface_models[kept] if use_warm_models else None)
            lci_kwargs = dict(
                smoothness=float(inv.get("smoothness", 0.3)),
                lateral_smoothness=lateral * lateral_weight_scale,
                reference_distance=reference_distance,
                starting_resistivity=float(inv.get("starting_resistivity", 100.0)),
                max_iterations=int(inv.get("max_iterations", 20)),
                convergence_tolerance=float(inv.get("convergence_tolerance", 0.02)),
                min_iterations=int(inv.get("min_iterations", 2)),
                auto_lambda=bool(inv.get("auto_lambda", True)),
                target_chi2=float(inv.get("target_chi2", 1.0)),
                chi2_tolerance=float(inv.get("chi2_tolerance", 0.2)),
                max_lambda_trials=int(inv.get("max_lambda_trials", 5)),
                verbose=bool(inv.get("verbose", True)),
            )
            doi_blocks.update(zip(kept, sounding_blocks))
            if bool(inv.get("reject_outliers", False)):
                log(f"Outlier rejection: cut beyond "
                    f"{float(inv.get('outlier_threshold', 3.0)):g} sigma, "
                    f"{int(inv.get('outlier_passes', 2))} pass(es), keeping at least "
                    f"{int(float(inv.get('min_data_fraction', 0.5)) * 100)} % of the gates "
                    f"and {int(inv.get('min_gates_per_sounding', 3))} per sounding.")
                outcome, sounding_blocks, outlier_info = invert_lci_rejecting_outliers(
                    sounding_blocks, n_layers,
                    threshold=float(inv.get("outlier_threshold", 3.0)),
                    passes=int(inv.get("outlier_passes", 2)),
                    min_fraction=float(inv.get("min_data_fraction", 0.5)),
                    min_gates=int(inv.get("min_gates_per_sounding", 3)),
                    initial_model=warm, log=log, **lci_kwargs)
                log(f"  Rejection finished: {outlier_info['kept']} of "
                    f"{outlier_info['n_start']} gates kept "
                    f"({outlier_info['stopped_because']}).")
            else:
                outcome = invert_lci(sounding_blocks, n_layers,
                                     initial_model=warm, log=log, **lci_kwargs)
            surface_models[kept] = outcome.models
            model[:, 0, :] = surface_models[:, ::-1]
            for index, s in enumerate(kept):
                chi2_list[s] = float(outcome.chi2_per_sounding[index])
                # The blocks are what was actually fitted, so they, not the file,
                # carry the gate count and the sensitivity once rejection has run.
                data_count_list[s] = int(sounding_blocks[index].dobs.size)
                doi_blocks[s] = sounding_blocks[index]
            # Every stage the solver actually ran, laid end to end. With
            # rejection on, the final run's own history is a couple of points
            # and hides the two solves before it.
            track = [{
                "stage": "solve",
                "lambda": float(outlier_info.get("initial", {}).get(
                    "smoothness_scale", outcome.smoothness_scale)),
                "chi2": list(outlier_info.get("initial", {}).get(
                    "convergence", outcome.chi2_history)),
                "n_data": int(outlier_info.get("initial", {}).get(
                    "n_data", sum(b.dobs.size for b in sounding_blocks))),
            }]
            for entry in outlier_info.get("passes") or []:
                track.append({
                    "stage": f"reject {entry['pass']}",
                    "lambda": float(outcome.smoothness_scale),
                    "chi2": list(entry.get("convergence") or []),
                    "n_data": int(entry.get("kept", 0)),
                })
            lci_report = {
                "mode": "simultaneous",
                "chi2": outcome.chi2,
                "chi2_history": outcome.chi2_history,
                "convergence_track": track,
                "iterations": outcome.iterations,
                "stop_reason": outcome.stop_reason,
                "smoothness_scale": outcome.smoothness_scale,
                "lambda_search": outcome.lambda_search,
                "seconds": outcome.seconds,
                "n_soundings": len(kept),
                "n_lateral_ties": max(len(kept) - 1, 0),
            }
            log(f"  LCI done: chi2={outcome.chi2:.3f} after "
                f"{outcome.iterations} iteration(s) ({outcome.stop_reason}), "
                f"{outcome.seconds:.1f}s")
    if not simultaneous and joint and lateral > 0.0 and lci_passes > 0:
        lci_report = {"mode": "sequential", "lci_passes": lci_passes}
        log(
            f"LCI refinement: {lci_passes} pass(es), lateral smoothness={lateral:g}, "
            f"vertical smoothness={float(inv.get('smoothness', 0.3)):g}, "
            f"auto-scale={lateral_weight_scale:g}"
        )
        for pass_index in range(lci_passes):
            previous = surface_models.copy()
            updated = previous.copy()
            for s in range(n_pos):
                if datasets[s] is None or not np.all(np.isfinite(previous[s])):
                    continue
                same_line = np.flatnonzero(line_numbers == line_numbers[s])
                before = same_line[same_line < s]
                after = same_line[same_line > s]
                neighbors = []
                if before.size:
                    neighbors.append(int(before[-1]))
                if after.size:
                    neighbors.append(int(after[0]))
                neighbors = [
                    index for index in neighbors
                    if np.all(np.isfinite(previous[index]))
                ]
                if not neighbors:
                    continue
                distances = np.asarray([
                    max(abs(float(pos_lci[index] - pos_lci[s])), reference_distance)
                    for index in neighbors
                ])
                weights = reference_distance / distances
                reference_log = np.average(
                    np.log10(previous[neighbors]), axis=0, weights=weights)
                local_inv = {
                    **inv,
                    "starting_model": previous[s],
                    "lateral_reference": np.power(10.0, reference_log),
                    "lateral_weight": (
                        lateral * lateral_weight_scale
                        * math.sqrt(float(np.sum(weights)))
                    ),
                }
                usable_local_data = (
                    not joint
                    or any(
                        np.asarray(item.get("times", [])).size >= 1
                        for item in datasets[s].get("moments", {}).values()
                    )
                )
                if not usable_local_data:
                    updated[s] = np.power(10.0, reference_log)
                    continue
                try:
                    result = invert(
                        datasets[s], geometries[s], local_inv, log=log)
                    updated[s] = np.asarray(
                        result["resistivity"], dtype=float).ravel()
                    chi2_list[s] = float(result.get("chi2", np.nan))
                except Exception as exc:  # noqa: BLE001
                    log(
                        f"  LCI pass {pass_index + 1}, sounding {s + 1} "
                        f"kept previous model: {exc}"
                    )
            surface_models = updated
            model[:, 0, :] = surface_models[:, ::-1]
            finite_pair = np.isfinite(previous) & np.isfinite(updated)
            change = (
                float(np.sqrt(np.mean(
                    (np.log10(updated[finite_pair])
                     - np.log10(previous[finite_pair])) ** 2
                )))
                if np.any(finite_pair) else float("nan")
            )
            log(f"  LCI pass {pass_index + 1}/{lci_passes}: model change={change:.4g}")

    # How far down the data still constrain each sounding, and what to hide.
    #
    # Where the analytic Jacobian is available the reach comes from the cumulated
    # sensitivity, which is what the depth of investigation actually means: below
    # it, moving the whole remaining column by a decade would not move the
    # predicted response out of its error bars. The diffusion-depth rule is the
    # fallback for solvers that supply no Jacobian; it is a rule of thumb about
    # the latest gate, so it uses each sounding's OWN latest gate rather than a
    # single time borrowed from the first sounding on the line.
    from PyHydroGeophysX.inversion.em1d_lci import (
        DOI_SENSITIVITY_THRESHOLD,
        cumulated_sensitivity,
        sensitivity_doi,
    )

    depth_ctr = 0.5 * (depth_edges[:-1] + depth_edges[1:])  # surface-ordered
    doi_threshold = float(inv.get("doi_threshold", DOI_SENSITIVITY_THRESHOLD))
    sensitivity = np.full((n_pos, n_layers), np.nan, dtype=float)
    doi = np.full(n_pos, np.nan, dtype=float)
    mu0 = 4e-7 * np.pi
    for s in range(n_pos):
        row = surface_models[s]
        if not np.all(np.isfinite(row)):
            continue
        block = doi_blocks.get(s)
        if block is not None:
            sensitivity[s] = cumulated_sensitivity(block, row)
            doi[s] = sensitivity_doi(block, row, depth_edges,
                                     threshold=doi_threshold)
            continue
        rho_ref = float(np.nanpercentile(row, 40))
        last_time = _latest_gate(datasets[s]) if datasets[s] is not None else t_ref
        if method == "TDEM" and last_time:
            doi[s] = doi_factor * math.sqrt(2.0 * last_time * rho_ref / mu0)
        elif method == "FDEM" and f_ref:
            doi[s] = doi_factor * 503.0 * math.sqrt(rho_ref / f_ref)

    # A cell stuck at the resistivity bound is a railed, meaningless value
    # whatever the sensitivity says, so that mask is applied either way.
    rail = 10 ** (5.0 - 0.2)  # near the _occam_1d resistivity upper bound (1e5 Ω·m)
    for s in range(n_pos):
        col = model[s, 0, :]  # deepest-first
        if not np.isfinite(col).any():
            continue
        if doi_blank and np.isfinite(doi[s]):
            col[~(depth_ctr <= doi[s])[::-1]] = np.nan
        col[col >= rail] = np.nan

    pos = pos_lci
    if pos.size >= 2:
        step = float(np.median(np.diff(pos)))
    else:
        step = float(spacing)
    ex = np.concatenate([[pos[0] - step / 2.0],
                         0.5 * (pos[:-1] + pos[1:]) if pos.size >= 2 else [],
                         [pos[-1] + step / 2.0]])
    ey = np.array([-step / 2.0, step / 2.0], dtype=float)

    finite = np.isfinite(model)
    # The coupled solve reports a data-weighted whole-line chi-squared, which is
    # the honest number for a single fit; averaging per-sounding values would
    # weight a 6-gate station the same as a 40-gate one.
    chi2_mean = (
        float(lci_report["chi2"]) if "chi2" in lci_report
        else float(np.nanmean(chi2_list)) if np.any(np.isfinite(chi2_list))
        else float("nan")
    )
    result = {
        "method": method, "edges": (ex, ey, ez), "model3d": model,
        "label": "resistivity (Ω·m)", "cmap": "turbo", "log_scale": True,
        "positions": pos, "depth_edges": depth_edges, "thickness": thick,
        "sensitivity": sensitivity, "doi": doi, "doi_threshold": doi_threshold,
        # Ground level at each sounding, so a section can be drawn against
        # elevation instead of depth. The inversion itself is per sounding and
        # does not use it: each 1D model starts at its own ground surface.
        "surface_elevation": surface_elevation,
        "chi2": chi2_mean, "chi2_list": chi2_list, "n_soundings": n_pos,
        "n_layers": n_layers, "n_data": int(sum(data_count_list)),
        "data_count_list": data_count_list, "data_scale": data_scale_used,
        "joint_moments": joint, "lci": bool(lci_report),
        "lci_mode": lci_report.get("mode", "off"), "lci_report": lci_report,
        "outliers": outlier_info,
        "lateral_smoothness": lateral, "lci_passes": lci_passes,
        "lateral_weight_scale": lateral_weight_scale,
        "line_numbers": line_numbers,
        "model_range": (float(np.nanmin(model)) if finite.any() else float("nan"),
                        float(np.nanmax(model)) if finite.any() else float("nan")),
    }
    if out_dir is not None:
        out = table_io.ensure_dir(out_dir)
        np.savez(out / "resistivity_section.npz",
                 positions=pos, elevation_edges=ez, position_edges=ex,
                 resistivity=model[:, 0, :], chi2=np.asarray(chi2_list, dtype=float),
                 # Saved so the depth cut can be reproduced, or moved, without
                 # re-running the inversion.
                 sensitivity=sensitivity, doi=doi, depth_edges=depth_edges,
                 line_numbers=np.asarray(line_numbers, dtype=int),
                 surface_elevation=surface_elevation)
        result["saved"] = [str(out / "resistivity_section.npz")]
        log(f"  saved {out / 'resistivity_section.npz'}")
    return result


def build_em_config(method: str, model: Dict[str, Any], geom: Dict[str, Any],
                    inv: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_time": _utc_now(),
        "method": method,
        "model": {"thickness": list(model.get("thickness", [])),
                  "resistivity": list(model.get("resistivity", []))},
        "geometry": dict(geom),
        "inversion": dict(inv),
    }


def save_inversion(result: Dict[str, Any], out_dir: Path) -> List[str]:
    """Save recovered model + data fit to npy/csv; return written paths."""
    out = table_io.ensure_dir(out_dir)
    paths: List[str] = []
    res = np.asarray(result["resistivity"], dtype=float)
    thick = np.asarray(result["thickness"], dtype=float)
    np.save(out / "recovered_resistivity.npy", res); paths.append(str(out / "recovered_resistivity.npy"))
    rows = [(float(t),) for t in thick]
    table_io.write_csv(out / "recovered_thickness.csv", rows, header=["thickness_m"])
    paths.append(str(out / "recovered_thickness.csv"))
    depth = np.asarray(result["depth"], dtype=float)
    rstep = np.asarray(result["resistivity_step"], dtype=float)
    table_io.write_csv(out / "model_depth_resistivity.csv",
                       list(zip(depth.tolist(), rstep.tolist())),
                       header=["depth_m", "resistivity_ohm_m"])
    paths.append(str(out / "model_depth_resistivity.csv"))
    return paths

__all__ = [
    "BackendUnavailable",
    "METHODS",
    "TEMCOMPANY_MOMENTS",
    "DEFAULT_MODEL",
    "DEFAULT_FDEM",
    "DEFAULT_TDEM",
    "DEFAULT_INVERSION",
    "backend_status",
    "example_catalog",
    "model_arrays",
    "model_depth_profile",
    "is_temcompany_source",
    "load_temcompany_sounding",
    "load_sounding",
    "load_line_geometry",
    "fdem_forward",
    "tdem_forward",
    "fdem_invert",
    "tdem_invert",
    "tdem_joint_invert",
    "estimate_data_scale",
    "calibrate_to_reference",
    "invert_line",
    "build_em_config",
    "save_inversion",
]
