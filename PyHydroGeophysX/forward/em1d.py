"""Qt-free 1D FDEM/TDEM forward helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

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


def _tdem_geometry(
    data: Dict[str, Any],
    geom: Dict[str, Any],
    transmitter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Complete TDEM geometry, with explicit caller settings taking precedence.

    Instrument readers keep station geometry under ``data["system"]`` and the
    moment-specific waveform, filters and gate windows under ``transmitter``.
    Public forward/inversion entry points historically accepted only ``geom``;
    merging here prevents a caller that passes a loaded sounding plus its system
    dictionary from silently falling back to the bare SimPEG receiver path.

    ``geom`` is applied last so notebook/GUI overrides remain effective.
    """
    merged: Dict[str, Any] = {}
    system = data.get("system")
    if isinstance(system, dict):
        merged.update(system)
    source = transmitter if transmitter is not None else data.get("transmitter")
    if isinstance(source, dict):
        merged.update(source)
    merged.update(geom)
    return merged


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


def _gate_window_name(geom: Dict[str, Any]) -> str:
    """The gate window to integrate with, named or read off the instrument.

    A TEMcompany project records the gate window as ``GateShape``. An explicit
    ``gate_window`` wins, so a caller with data that describes its gating some
    other way can say so.

    A shape the mapping does not cover raises rather than falling back. Falling
    back would put a window in the forward that the instrument did not use and
    leave nothing to notice it; a dataset with no recorded shape at all is a
    different case and gets the gate-centre reading.
    """
    from PyHydroGeophysX.forward.tdem_forward import GATE_SHAPE_NAMES

    named = geom.get("gate_window")
    if named:
        return str(named)
    shape = geom.get("gate_window_shape")
    if shape is None:
        return "centre"
    try:
        index = int(round(float(shape)))
    except (TypeError, ValueError):
        return "centre"
    if index not in GATE_SHAPE_NAMES:
        raise ValueError(
            f"GateShape = {index} is not a window this forward implements "
            f"(known: {sorted(GATE_SHAPE_NAMES)}). Set gate_window explicitly "
            "to model this survey.")
    return GATE_SHAPE_NAMES[index]


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
        if np.allclose(centres[picked], times, rtol=1e-6, atol=0.0):
            opens, closes = opens[picked], closes[picked]
        else:
            opens = closes = np.array([], dtype=float)
    shift = float(geom.get("gate_time_shift", 0.0) or 0.0)
    if shift:
        times = times + shift
        if opens.size:
            opens, closes = opens + shift, closes + shift
    # The loop and the coil can sit at different heights. They are equal on
    # every TEMcompany project seen so far, so ``height`` remains the single
    # number a caller that does not care has to supply. A reader that has the
    # key but no value writes NaN rather than omitting it, which ``.get`` cannot
    # catch, and a NaN source location fails much later and less legibly.
    def _height(key: str) -> float:
        try:
            value = float(geom.get(key, h))
        except (TypeError, ValueError):
            return h
        return value if np.isfinite(value) else h

    tx_h = _height("tx_height")
    rx_h = _height("rx_height")
    return TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, tx_h]),
        source_radius=float(geom.get("source_radius", 10.0)),
        source_turns=int(geom.get("loop_turns", 1)),
        source_moment=None if moment is None else float(moment),
        waveform_times=geom.get("waveform_times"),
        waveform_currents=geom.get("waveform_currents"),
        waveform_period=(float(geom["waveform_period"])
                         if geom.get("waveform_period") else None),
        # Whether the transmitter train is superposed is not recorded in the
        # project file, so it cannot be read from one. A bipolar transmitter is
        # running either way, so modelling it is the default; set
        # ``waveform_repeat`` false, or ``waveform_repetitions`` to zero, for a
        # survey processed without it.
        waveform_repetitions=(
            int(geom.get("waveform_repetitions", 3))
            if bool(geom.get("waveform_repeat", True)) else 0),
        gate_open=opens if opens.size == times.size else None,
        gate_close=closes if closes.size == times.size else None,
        # The instrument's own gate window; see GATE_WINDOWS in
        # forward.tdem_forward for the measurement that identified it.
        gate_window=_gate_window_name(geom),
        gate_window_par=float(geom.get("gate_window_par", 0.667) or 0.667),
        analog_lowpass=geom.get("analog_lowpass"),
        analog_points_per_decade=int(
            geom.get("analog_points_per_decade", 150)),
        analog_model_points_per_decade=int(
            geom.get("analog_model_points_per_decade", 40)),
        instrument_points_per_decade=int(
            geom.get("instrument_points_per_decade", 10)),
        instrument_model_points_per_decade=int(
            geom.get("instrument_model_points_per_decade", 10)),
        gate_quadrature_order=int(geom.get("gate_quadrature_order", 8)),
        receiver_location=np.array([sep, 0.0, rx_h]),
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
