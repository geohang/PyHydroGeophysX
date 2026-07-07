"""1D electromagnetic (FDEM / TDEM) forward + inversion pipeline for the workbench.

A thin, Qt-free wrapper around the SimPEG-based classes already in the package:

* ``PyHydroGeophysX.forward.fdem_forward.FDEMForwardModeling`` /
  ``PyHydroGeophysX.forward.tdem_forward.TDEMForwardModeling`` -- 1D layered-earth
  forward responses.
* ``PyHydroGeophysX.inversion.fdem_inversion.FDEMInversion`` /
  ``PyHydroGeophysX.inversion.tdem_inversion.run_tdem_inversion`` -- 1D Occam-style
  layered inversion.

Models are described by ``thickness`` (N-1 layer thicknesses, m) and
``resistivity`` (N layer resistivities, ohm-m; the last is the half-space).
SimPEG is imported lazily; if it is missing the functions raise
``BackendUnavailable`` so the UI can degrade gracefully.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from PyHydroGeophysX.qt_apps import io_utils

LogFn = Callable[[str], None]

METHODS = ("FDEM", "TDEM")

DEFAULT_MODEL = {"thickness": [10.0, 20.0], "resistivity": [50.0, 200.0, 20.0]}

DEFAULT_FDEM = {
    "freq_min": 100.0, "freq_max": 100000.0, "n_freq": 16,
    "source_radius": 10.0, "tx_rx_sep": 10.0, "height": 30.0,
    "orientation": "z", "component": "secondary", "waveform": "dipole",
    "noise_level": 0.03,
}
DEFAULT_TDEM = {
    "t_min": 1e-5, "t_max": 1e-2, "n_times": 25,
    "source_radius": 10.0, "height": 30.0, "orientation": "z",
    "waveform": "step_off", "noise_level": 0.03,
}
DEFAULT_INVERSION = {
    "n_layers": 15, "min_thickness": 1.0, "max_thickness": 40.0,
    "starting_resistivity": 100.0, "max_iterations": 30,
    "rel_error": 0.05, "noise_floor": 1e-14, "smoothness": 0.3,
}


class BackendUnavailable(RuntimeError):
    """Raised when SimPEG cannot be imported."""


def _noop(_msg: str) -> None:
    return None


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Model helpers (numpy only)
# ---------------------------------------------------------------------------
def model_arrays(model: Dict[str, Any]) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return (thicknesses, resistivity, conductivity) from a model dict."""
    thick = np.asarray(model.get("thickness", []), dtype=float).ravel()
    res = np.asarray(model.get("resistivity", []), dtype=float).ravel()
    if res.size != thick.size + 1:
        raise ValueError(
            f"resistivity needs thickness+1 entries ({thick.size + 1}), got {res.size}.")
    res = np.clip(res, 1e-6, 1e12)
    return thick, res, 1.0 / res


def model_depth_profile(thicknesses: np.ndarray, resistivity: np.ndarray,
                        pad: float = 20.0) -> "tuple[np.ndarray, np.ndarray]":
    """Step profile (depth, resistivity) for plotting a layered model."""
    thick = np.asarray(thicknesses, dtype=float).ravel()
    res = np.asarray(resistivity, dtype=float).ravel()
    tops = np.concatenate([[0.0], np.cumsum(thick)])
    bottoms = np.concatenate([np.cumsum(thick), [tops[-1] + pad]])
    depth, value = [], []
    for i in range(res.size):
        depth.extend([tops[i], bottoms[i]])
        value.extend([res[i], res[i]])
    return np.asarray(depth, dtype=float), np.asarray(value, dtype=float)


# ---------------------------------------------------------------------------
# Sounding IO (numpy only)
# ---------------------------------------------------------------------------
def load_sounding(path: str, method: str, sounding: int = 0) -> Dict[str, np.ndarray]:
    """Load one sounding from a sounding file.

    The first column is the abscissa (FDEM: frequency Hz; TDEM: time s). The
    remaining columns hold the response(s) — a single sounding, or several stacked
    side by side so one file can carry a whole survey line (common for airborne EM
    exports). ``sounding`` picks which one (0-based):

    - **TDEM**: each extra column is one sounding's response → column ``1 + sounding``.
    - **FDEM**: response columns come in ``(real, imag)`` pairs → one sounding is the
      pair starting at ``1 + 2*sounding``; a lone trailing real column gives imag = 0.

    The returned dict also reports ``n_soundings`` so the caller can offer a picker.
    """
    table = np.atleast_2d(io_utils.load_2d_array(path)).astype(float)
    if table.shape[1] < 2:
        raise ValueError(f"Expected >= 2 columns, got shape {table.shape}.")
    x = table[:, 0]
    n_resp = table.shape[1] - 1
    if method == "FDEM":
        n_soundings = max(1, (n_resp + 1) // 2)
        s = max(0, min(int(sounding), n_soundings - 1))
        ri = 1 + 2 * s
        real = table[:, ri]
        imag = table[:, ri + 1] if ri + 1 < table.shape[1] else np.zeros_like(x)
        return {"frequencies": x, "real": real, "imag": imag,
                "n_soundings": n_soundings, "sounding": s}
    n_soundings = n_resp
    s = max(0, min(int(sounding), n_soundings - 1))
    return {"times": x, "response": table[:, 1 + s],
            "n_soundings": n_soundings, "sounding": s}


# ---------------------------------------------------------------------------
# Forward modeling (SimPEG)
# ---------------------------------------------------------------------------
def _fdem_config(geom: Dict[str, Any], frequencies: np.ndarray):
    from PyHydroGeophysX.forward.fdem_forward import FDEMSurveyConfig

    h = float(geom.get("height", 30.0))
    sep = float(geom.get("tx_rx_sep", 10.0))
    return FDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, h]),
        source_radius=float(geom.get("source_radius", 10.0)),
        receiver_location=np.array([sep, 0.0, h]),
        receiver_orientation=str(geom.get("orientation", "z")),
        receiver_component=str(geom.get("component", "secondary")),
        frequencies=np.asarray(frequencies, dtype=float),
        waveform_type=str(geom.get("waveform", "dipole")),
    )


def fdem_forward(model: Dict[str, Any], geom: Dict[str, Any], log: LogFn = _noop) -> Dict[str, Any]:
    """1D FDEM forward response (secondary field, real/imag per frequency)."""
    try:
        from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    thick, res, sigma = model_arrays(model)
    freqs = np.logspace(np.log10(float(geom.get("freq_min", 100.0))),
                        np.log10(float(geom.get("freq_max", 1e5))),
                        int(geom.get("n_freq", 16)))
    log(f"FDEM forward: {sigma.size} layers, {freqs.size} frequencies")
    modeler = FDEMForwardModeling(thicknesses=thick, survey_config=_fdem_config(geom, freqs))
    resp = np.asarray(modeler.forward(sigma))
    resp = resp.ravel()
    if resp.size == 2 * freqs.size and not np.iscomplexobj(resp):
        resp = resp[0::2] + 1j * resp[1::2]
    resp = np.asarray(resp, dtype=complex).ravel()[: freqs.size]
    return {"frequencies": freqs, "real": resp.real, "imag": resp.imag,
            "amplitude": np.abs(resp), "phase_deg": np.degrees(np.angle(resp)),
            "resistivity": res, "thickness": thick}


def _tdem_config(geom: Dict[str, Any], times: np.ndarray):
    from PyHydroGeophysX.forward.tdem_forward import TDEMSurveyConfig

    h = float(geom.get("height", 30.0))
    return TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, h]),
        source_radius=float(geom.get("source_radius", 10.0)),
        receiver_location=np.array([0.0, 0.0, h]),
        receiver_orientation=str(geom.get("orientation", "z")),
        times=np.asarray(times, dtype=float),
        waveform_type=str(geom.get("waveform", "step_off")),
    )


def tdem_forward(model: Dict[str, Any], geom: Dict[str, Any], log: LogFn = _noop) -> Dict[str, Any]:
    """1D TDEM forward response (dB/dt or H per time channel)."""
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    thick, res, sigma = model_arrays(model)
    times = np.logspace(np.log10(float(geom.get("t_min", 1e-5))),
                        np.log10(float(geom.get("t_max", 1e-2))),
                        int(geom.get("n_times", 25)))
    log(f"TDEM forward: {sigma.size} layers, {times.size} time channels")
    modeler = TDEMForwardModeling(thicknesses=thick, survey_config=_tdem_config(geom, times))
    resp = np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]
    return {"times": times, "response": resp, "resistivity": res, "thickness": thick}


# ---------------------------------------------------------------------------
# Inversion (Occam-style 1D least-squares fit reusing the SimPEG forward)
# ---------------------------------------------------------------------------
# SimPEG's own 1D inversion classes crash natively with SimPEG 0.24 on this
# platform, so the inversion is performed here with scipy.optimize.least_squares
# over a fixed-layer model, calling the (working) SimPEG forward operator.
# Unknowns are log10(resistivity) per layer; a first-difference smoothness term
# regularizes the recovered model.
def _layer_thicknesses(n_layers: int, min_thickness: float, max_thickness: float) -> np.ndarray:
    n_layers = max(2, int(n_layers))
    if n_layers == 2:
        return np.array([float(min_thickness)], dtype=float)
    return np.geomspace(float(min_thickness), float(max_thickness), n_layers - 1)


def _occam_1d(forward_vec: Callable[[np.ndarray], np.ndarray], dobs_vec: np.ndarray,
              unc_vec: np.ndarray, n_layers: int, inv: Dict[str, Any], log: LogFn):
    """Smooth fixed-layer fit. ``forward_vec(sigma)`` returns a data vector."""
    from scipy.optimize import least_squares
    lam = float(inv.get("smoothness", 0.3))
    start_res = float(inv.get("starting_resistivity", 100.0))
    lo, hi = 0.0, 5.0  # log10 resistivity bounds (1 .. 1e5 ohm-m)
    x0 = float(np.clip(np.log10(max(start_res, 1.0)), lo, hi)) * np.ones(n_layers)
    unc_vec = np.clip(unc_vec, 1e-30, None)

    def residual(logres: np.ndarray) -> np.ndarray:
        sigma = 1.0 / np.power(10.0, logres)
        pred = np.asarray(forward_vec(sigma), dtype=float)
        data_res = (pred - dobs_vec) / unc_vec
        smooth = lam * np.diff(logres)
        return np.concatenate([data_res, smooth])

    max_nfev = max(40, int(inv.get("max_iterations", 30)) * (n_layers + 1))
    sol = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                        max_nfev=max_nfev, xtol=1e-8, ftol=1e-8)
    res = np.power(10.0, sol.x)
    data_res = residual(sol.x)[: dobs_vec.size]
    chi2 = float(np.mean(data_res ** 2))
    log(f"  inversion done: {sol.nfev} forward evals, chi2={chi2:.3f}")
    return res, chi2, int(sol.nfev)


def fdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert an FDEM sounding for a layered resistivity model (Occam 1D)."""
    try:
        from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    freqs = np.asarray(data["frequencies"], dtype=float).ravel()
    scale = float(inv.get("data_scale", 1.0))
    obs_r = np.asarray(data["real"], dtype=float).ravel() * scale
    obs_i = np.asarray(data["imag"], dtype=float).ravel() * scale
    dobs_vec = np.concatenate([obs_r, obs_i])
    rel = float(inv.get("rel_error", 0.05)); floor = float(inv.get("noise_floor", 1e-14)) * scale
    amp = np.abs(obs_r + 1j * obs_i)
    unc_vec = np.concatenate([rel * amp + floor, rel * amp + floor])
    n_layers = int(inv.get("n_layers", 15))
    thick = _layer_thicknesses(n_layers, float(inv.get("min_thickness", 1.0)),
                               float(inv.get("max_thickness", 40.0)))
    modeler = FDEMForwardModeling(thicknesses=thick, survey_config=_fdem_config(geom, freqs))
    log(f"FDEM inversion: {freqs.size} freqs, {n_layers} layers")

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        resp = np.asarray(modeler.forward(sigma)).ravel()
        if resp.size == 2 * freqs.size and not np.iscomplexobj(resp):
            resp = resp[0::2] + 1j * resp[1::2]
        resp = np.asarray(resp, dtype=complex).ravel()[: freqs.size]
        return np.concatenate([resp.real, resp.imag])

    res, chi2, nfev = _occam_1d(forward_vec, dobs_vec, unc_vec, n_layers, inv, log)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    pred_r, pred_i = pred[: freqs.size], pred[freqs.size:]
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "FDEM", "frequencies": freqs,
            "obs_real": obs_r, "obs_imag": obs_i,
            "pred_real": pred_r, "pred_imag": pred_i,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(dobs_vec.size), "nfev": nfev, "n_layers": n_layers}


def tdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert a TDEM sounding for a layered resistivity model (Occam 1D)."""
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    times = np.asarray(data["times"], dtype=float).ravel()
    scale = float(inv.get("data_scale", 1.0))
    dobs = np.asarray(data["response"], dtype=float).ravel() * scale
    rel = float(inv.get("rel_error", 0.05)); floor = float(inv.get("noise_floor", 1e-18)) * scale
    unc = rel * np.abs(dobs) + floor
    n_layers = int(inv.get("n_layers", 15))
    thick = _layer_thicknesses(n_layers, float(inv.get("min_thickness", 1.0)),
                               float(inv.get("max_thickness", 40.0)))
    modeler = TDEMForwardModeling(thicknesses=thick, survey_config=_tdem_config(geom, times))
    log(f"TDEM inversion: {times.size} times, {n_layers} layers")

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]

    res, chi2, nfev = _occam_1d(forward_vec, dobs, unc, n_layers, inv, log)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "TDEM", "times": times, "obs": dobs, "pred": pred,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2,
            "n_data": int(dobs.size), "nfev": nfev, "n_layers": n_layers}


# ---------------------------------------------------------------------------
# Auto-calibration + per-sounding geometry
# ---------------------------------------------------------------------------
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
    try:
        head = load_sounding(path, method, sounding=0)
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
                grid.append(np.asarray(md.forward(np.array([1.0 / R, 1.0 / R])),
                                       dtype=float).ravel()[:abscissa.size])
            preds = np.asarray(grid)

            def observed(s):
                return np.asarray(load_sounding(path, method, sounding=int(s))["response"], float)
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
                d = load_sounding(path, method, sounding=int(s))
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
    try:
        head = load_sounding(path, method, sounding=0)
        n_total = int(head.get("n_soundings", 1))
        probe = np.unique(np.linspace(0, n_total - 1, min(int(max_probe), n_total)).astype(int))
        if method == "TDEM":
            from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
            abscissa = np.asarray(head["times"], dtype=float).ravel()
            md = TDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=_tdem_config(geom, abscissa))
            pred = np.asarray(md.forward(np.array([1.0 / ref, 1.0 / ref])),
                              dtype=float).ravel()[:abscissa.size]

            def observed(s):
                return np.asarray(load_sounding(path, method, sounding=int(s))["response"], float)
        else:
            from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
            abscissa = np.asarray(head["frequencies"], dtype=float).ravel()
            md = FDEMForwardModeling(thicknesses=np.array([50.0]), survey_config=_fdem_config(geom, abscissa))
            resp = np.asarray(md.forward(np.array([1.0 / ref, 1.0 / ref]))).ravel()
            if resp.size == 2 * abscissa.size and not np.iscomplexobj(resp):
                resp = resp[0::2] + 1j * resp[1::2]
            pred = np.abs(np.asarray(resp, dtype=complex).ravel()[:abscissa.size])

            def observed(s):
                d = load_sounding(path, method, sounding=int(s))
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


def load_line_geometry(path: str) -> Dict[str, Any]:
    """Load per-sounding line geometry: along-line ``positions`` (m), optional
    sensor ``heights`` (m), and optional map coordinates ``x``/``y`` (e.g.
    easting/northing) for plan-view depth slices. Recognizes header names
    (distance/position for the position; alt/height for the height;
    easting/northing for the map coordinates, which also derive the distance when
    no distance column is present). A header-less file is read by column order
    (1 column = position; 2+ = position, height). ``positions`` is shifted to
    start at 0.
    """
    positions = heights = xs = ys = None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.shape[1] >= 1 and any(not str(c).replace(".", "").lstrip("-").isdigit()
                                    for c in df.columns):
            cols = {str(c).lower().strip(): c for c in df.columns}

            def pick(names):
                for nm in names:
                    if nm in cols:
                        return np.asarray(df[cols[nm]], dtype=float).ravel()
                return None

            positions = pick(["dist_m", "distance", "position", "dist", "offset"])
            heights = pick(["sensor_alt_m", "alt", "altitude", "height", "sensor_height", "clearance"])
            xs = pick(["e_utm13n", "easting_m", "easting", "east", "e", "x_utm", "x_m", "x"])
            ys = pick(["n_utm13n", "northing_m", "northing", "north", "n", "y_utm", "y_m", "y"])
            if positions is None and xs is not None and ys is not None:
                positions = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
            elif positions is None and xs is not None:
                positions = xs
    except Exception:  # noqa: BLE001 - fall back to a plain numeric table
        positions = None
    if positions is None:
        arr = np.atleast_2d(io_utils.load_2d_array(path)).astype(float)
        positions = arr[:, 0]
        heights = arr[:, 1] if arr.shape[1] >= 2 else None
    positions = np.asarray(positions, dtype=float).ravel()
    positions = positions - float(np.nanmin(positions))
    if heights is not None:
        heights = np.asarray(heights, dtype=float).ravel()
    has_xy = xs is not None and ys is not None
    return {"positions": positions, "heights": heights,
            "x": np.asarray(xs, dtype=float).ravel() if xs is not None else None,
            "y": np.asarray(ys, dtype=float).ravel() if ys is not None else None,
            "n": int(positions.size), "has_heights": heights is not None, "has_xy": has_xy}


# ---------------------------------------------------------------------------
# Stitched line inversion (position along line x depth)
# ---------------------------------------------------------------------------
def invert_line(path: str, method: str, geom: Dict[str, Any], inv: Dict[str, Any],
                *, spacing: float = 50.0, positions: Optional[np.ndarray] = None,
                heights: Optional[np.ndarray] = None, max_soundings: int = 12,
                doi_blank: bool = True, doi_factor: float = 0.5, ref_resistivity: float = 0.0,
                out_dir: Optional[Path] = None, log: LogFn = _noop) -> Dict[str, Any]:
    """Invert every sounding in a multi-sounding file into a stitched 2D section.

    Each sounding is inverted independently with the 1D Occam routine
    (:func:`fdem_invert` / :func:`tdem_invert`) on a shared fixed-layer grid, then
    the recovered layer resistivities are laid side by side to form a
    ``resistivity(position, depth)`` section. Returns a grid ready for
    :meth:`Model3DView.show_model`: ``edges = (ex, ey, ez)`` (``ez`` is elevation,
    increasing upward) and ``model3d`` of shape ``(n_pos, 1, n_depth)``.

    ``positions`` gives the along-line distance of each sounding (the section
    x-axis); ``heights`` overrides the sensor height per sounding. When
    ``doi_blank`` is set, cells below a per-sounding depth of investigation
    (a diffusion-depth estimate scaled by ``doi_factor``) are blanked (NaN) so the
    unconstrained deep part of an early-time sounding is not shown as railed.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    invert = fdem_invert if method == "FDEM" else tdem_invert
    n_total = int(load_sounding(path, method, sounding=0).get("n_soundings", 1))
    n_pos = min(int(max_soundings), max(1, n_total))
    log(f"Line inversion: {n_pos} of {n_total} soundings ({method})")

    # Calibrate the amplitude scale to a known reference resistivity if requested
    # (breaks the data_scale <-> resistivity-level degeneracy with external info).
    if ref_resistivity and float(ref_resistivity) > 0:
        inv = {**inv, "data_scale": calibrate_to_reference(
            path, method, geom, inv, float(ref_resistivity), log=log)}
    data_scale_used = float(inv.get("data_scale", 1.0))

    # Shared layer grid (identical for every sounding).
    n_layers = int(inv.get("n_layers", 15))
    thick = _layer_thicknesses(n_layers, float(inv.get("min_thickness", 1.0)),
                               float(inv.get("max_thickness", 40.0)))
    pad = float(inv.get("max_thickness", 40.0))
    depth_edges = np.concatenate([[0.0], np.cumsum(thick), [float(np.sum(thick)) + pad]])
    ez = (-depth_edges)[::-1]  # elevation edges, increasing upward (surface at 0)

    hts = np.asarray(heights, dtype=float).ravel() if heights is not None else None
    model = np.full((n_pos, 1, n_layers), np.nan, dtype=float)
    chi2_list: List[float] = []
    t_ref = f_ref = None  # last time / min frequency, for the DOI estimate
    for s in range(n_pos):
        try:
            data = load_sounding(path, method, sounding=s)
            if t_ref is None and "times" in data and np.size(data["times"]):
                t_ref = float(np.asarray(data["times"]).ravel()[-1])
            if f_ref is None and "frequencies" in data and np.size(data["frequencies"]):
                f_ref = float(np.asarray(data["frequencies"]).ravel().min())
            geom_s = geom
            if hts is not None and s < hts.size and np.isfinite(hts[s]):
                geom_s = {**geom, "height": float(hts[s])}
            result = invert(data, geom_s, inv, log=log)
            res = np.asarray(result["resistivity"], dtype=float).ravel()
            model[s, 0, :] = res[::-1]  # deepest layer first to match ez ordering
            chi2_list.append(float(result.get("chi2", np.nan)))
            log(f"  sounding {s + 1}/{n_pos}: chi2={result.get('chi2', float('nan')):.3f}")
        except Exception as exc:  # noqa: BLE001 - keep the line going if one sounding fails
            chi2_list.append(float("nan"))
            log(f"  sounding {s + 1}/{n_pos} failed: {exc}")

    # Blank cells below the depth of investigation (unconstrained by early time)
    # and any cell stuck at the resistivity bound (a railed, meaningless value).
    if doi_blank and np.isfinite(model).any():
        depth_ctr = 0.5 * (depth_edges[:-1] + depth_edges[1:])  # surface-ordered
        mu0 = 4e-7 * np.pi
        rail = 10 ** (5.0 - 0.2)  # near the _occam_1d resistivity upper bound (1e5 Ω·m)
        for s in range(n_pos):
            col = model[s, 0, :]  # deepest-first
            if not np.isfinite(col).any():
                continue
            # A low percentile ignores the railed high tail when gauging depth reach.
            rho_ref = float(np.nanpercentile(col, 40))
            if method == "TDEM" and t_ref:
                doi = doi_factor * np.sqrt(2.0 * t_ref * rho_ref / mu0)
            elif method == "FDEM" and f_ref:
                doi = doi_factor * 503.0 * np.sqrt(rho_ref / f_ref)
            else:
                doi = np.inf
            keep = (depth_ctr <= doi)[::-1]  # deepest-first to match col
            col[~keep] = np.nan
            col[col >= rail] = np.nan

    if positions is not None:
        pos = np.asarray(positions, dtype=float).ravel()[:n_pos]
    else:
        pos = np.arange(n_pos, dtype=float) * float(spacing)
    if pos.size >= 2:
        step = float(np.median(np.diff(pos)))
    else:
        step = float(spacing)
    ex = np.concatenate([[pos[0] - step / 2.0],
                         0.5 * (pos[:-1] + pos[1:]) if pos.size >= 2 else [],
                         [pos[-1] + step / 2.0]])
    ey = np.array([-step / 2.0, step / 2.0], dtype=float)

    finite = np.isfinite(model)
    chi2_mean = float(np.nanmean(chi2_list)) if np.any(np.isfinite(chi2_list)) else float("nan")
    result = {
        "method": method, "edges": (ex, ey, ez), "model3d": model,
        "label": "resistivity (Ω·m)", "cmap": "turbo", "log_scale": True,
        "positions": pos, "depth_edges": depth_edges, "thickness": thick,
        "chi2": chi2_mean, "chi2_list": chi2_list, "n_soundings": n_pos,
        "n_layers": n_layers, "n_data": int(finite.sum()), "data_scale": data_scale_used,
        "model_range": (float(np.nanmin(model)) if finite.any() else float("nan"),
                        float(np.nanmax(model)) if finite.any() else float("nan")),
    }
    if out_dir is not None:
        out = io_utils.ensure_dir(out_dir)
        np.savez(out / "resistivity_section.npz",
                 positions=pos, elevation_edges=ez, position_edges=ex,
                 resistivity=model[:, 0, :], chi2=np.asarray(chi2_list, dtype=float))
        result["saved"] = [str(out / "resistivity_section.npz")]
        log(f"  saved {out / 'resistivity_section.npz'}")
    return result


# ---------------------------------------------------------------------------
# Config export (numpy only)
# ---------------------------------------------------------------------------
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
    out = io_utils.ensure_dir(out_dir)
    paths: List[str] = []
    res = np.asarray(result["resistivity"], dtype=float)
    thick = np.asarray(result["thickness"], dtype=float)
    np.save(out / "recovered_resistivity.npy", res); paths.append(str(out / "recovered_resistivity.npy"))
    rows = [(float(t),) for t in thick]
    io_utils.write_csv(out / "recovered_thickness.csv", rows, header=["thickness_m"])
    paths.append(str(out / "recovered_thickness.csv"))
    depth = np.asarray(result["depth"], dtype=float)
    rstep = np.asarray(result["resistivity_step"], dtype=float)
    io_utils.write_csv(out / "model_depth_resistivity.csv",
                       list(zip(depth.tolist(), rstep.tolist())),
                       header=["depth_m", "resistivity_ohm_m"])
    paths.append(str(out / "model_depth_resistivity.csv"))
    return paths
