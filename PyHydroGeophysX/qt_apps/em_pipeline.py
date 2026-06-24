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
def load_sounding(path: str, method: str) -> Dict[str, np.ndarray]:
    """Load a sounding file. FDEM: x, real, imag. TDEM: x, response."""
    table = np.atleast_2d(io_utils.load_2d_array(path)).astype(float)
    if table.shape[1] < 2:
        raise ValueError(f"Expected >= 2 columns, got shape {table.shape}.")
    x = table[:, 0]
    if method == "FDEM":
        if table.shape[1] >= 3:
            return {"frequencies": x, "real": table[:, 1], "imag": table[:, 2]}
        # Single response column: treat as real, imag = 0.
        return {"frequencies": x, "real": table[:, 1], "imag": np.zeros_like(x)}
    return {"times": x, "response": table[:, 1]}


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
    return res, chi2


def fdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert an FDEM sounding for a layered resistivity model (Occam 1D)."""
    try:
        from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    freqs = np.asarray(data["frequencies"], dtype=float).ravel()
    obs_r = np.asarray(data["real"], dtype=float).ravel()
    obs_i = np.asarray(data["imag"], dtype=float).ravel()
    dobs_vec = np.concatenate([obs_r, obs_i])
    rel = float(inv.get("rel_error", 0.05)); floor = float(inv.get("noise_floor", 1e-14))
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

    res, chi2 = _occam_1d(forward_vec, dobs_vec, unc_vec, n_layers, inv, log)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    pred_r, pred_i = pred[: freqs.size], pred[freqs.size:]
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "FDEM", "frequencies": freqs,
            "obs_real": obs_r, "obs_imag": obs_i,
            "pred_real": pred_r, "pred_imag": pred_i,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2}


def tdem_invert(data: Dict[str, Any], geom: Dict[str, Any], inv: Dict[str, Any],
                log: LogFn = _noop) -> Dict[str, Any]:
    """Invert a TDEM sounding for a layered resistivity model (Occam 1D)."""
    try:
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
    except Exception as exc:  # noqa: BLE001
        raise BackendUnavailable(str(exc))
    times = np.asarray(data["times"], dtype=float).ravel()
    dobs = np.asarray(data["response"], dtype=float).ravel()
    rel = float(inv.get("rel_error", 0.05)); floor = float(inv.get("noise_floor", 1e-18))
    unc = rel * np.abs(dobs) + floor
    n_layers = int(inv.get("n_layers", 15))
    thick = _layer_thicknesses(n_layers, float(inv.get("min_thickness", 1.0)),
                               float(inv.get("max_thickness", 40.0)))
    modeler = TDEMForwardModeling(thicknesses=thick, survey_config=_tdem_config(geom, times))
    log(f"TDEM inversion: {times.size} times, {n_layers} layers")

    def forward_vec(sigma: np.ndarray) -> np.ndarray:
        return np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]

    res, chi2 = _occam_1d(forward_vec, dobs, unc, n_layers, inv, log)
    sigma = 1.0 / np.clip(res, 1e-12, None)
    pred = forward_vec(sigma)
    depth, res_step = model_depth_profile(thick, res)
    return {"method": "TDEM", "times": times, "obs": dobs, "pred": pred,
            "thickness": thick, "resistivity": res, "conductivity": sigma,
            "depth": depth, "resistivity_step": res_step, "chi2": chi2}


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
