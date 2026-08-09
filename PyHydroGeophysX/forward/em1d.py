"""Qt-free 1D FDEM/TDEM forward helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop

LogFn = Callable[[str], None]

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
    sep = float(geom.get("tx_rx_sep", 0.0))
    moment = geom.get("source_moment")
    gates = geom.get("gate_windows") or {}
    # Gate windows only apply when they belong to these very time channels; a
    # sounding whose gates were partly rejected carries a shorter time vector.
    opens = np.asarray(gates.get("open", []), dtype=float).ravel()
    closes = np.asarray(gates.get("close", []), dtype=float).ravel()
    centres = np.asarray(gates.get("centre", []), dtype=float).ravel()
    times = np.asarray(times, dtype=float)
    if centres.size == opens.size == closes.size and centres.size:
        picked = [int(np.argmin(np.abs(centres - value))) for value in times]
        if np.allclose(centres[picked], times, rtol=1e-6):
            opens, closes = opens[picked], closes[picked]
        else:
            opens = closes = np.array([], dtype=float)
    return TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, h]),
        source_radius=float(geom.get("source_radius", 10.0)),
        source_turns=int(geom.get("loop_turns", 1)),
        source_moment=None if moment is None else float(moment),
        waveform_times=geom.get("waveform_times"),
        waveform_currents=geom.get("waveform_currents"),
        gate_open=opens if opens.size == times.size else None,
        gate_close=closes if closes.size == times.size else None,
        # Three nodes per window costs about 5 % and removes the last
        # per-moment discrepancy against the vendor forward.
        gate_samples=int(geom.get("gate_samples", 3)),
        receiver_location=np.array([sep, 0.0, h]),
        receiver_orientation=str(geom.get("orientation", "z")),
        receiver_type=str(geom.get("receiver_type", "b")),
        times=times,
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
    sign = float(geom.get("response_sign", 1.0))
    resp = sign * np.asarray(modeler.forward(sigma), dtype=float).ravel()[: times.size]
    return {"times": times, "response": resp, "resistivity": res, "thickness": thick}

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_FDEM",
    "DEFAULT_TDEM",
    "model_arrays",
    "model_depth_profile",
    "fdem_forward",
    "tdem_forward",
]
