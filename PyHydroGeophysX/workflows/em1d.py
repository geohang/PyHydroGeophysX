"""High-level 1D electromagnetic workflows and compatibility facade."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
from PyHydroGeophysX._internal.utils import noop as _noop, utc_now as _utc_now
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.data_processing.em1d import (
    TEMCOMPANY_MOMENTS,
    _normalise_temcompany_moment,
    _response_on_times,
    is_temcompany_source,
    is_ttem_source,
    gate_report,
    load_line_geometry,
    load_sounding,
    load_sounding_container,
    load_temcompany_sounding,
    load_ttem_sounding,
    save_sounding_container,
    survey_summary,
)
from PyHydroGeophysX.forward.em1d import (
    DEFAULT_FDEM,
    DEFAULT_MODEL,
    DEFAULT_TDEM,
    _fdem_config,
    _tdem_config,
    _tdem_geometry,
    fdem_forward,
    model_arrays,
    model_depth_profile,
    tdem_forward,
)
from PyHydroGeophysX.inversion.em1d import (
    DEFAULT_INVERSION,
    INVERSION_PRESETS,
    preset_inversion,
    _inversion_layer_thicknesses,
    _log_resistivity_bounds,
    fdem_invert,
    tdem_invert,
    tdem_joint_invert,
)

LogFn = Callable[[str], None]


def _scale_bounds(inv: Dict[str, Any]) -> "tuple[float, float]":
    """How far auto-lambda may scale the smoothness, as ``(low, high)``.

    Kept beside :func:`_log_resistivity_bounds` so both box constraints reach the
    coupled solver by the same route, and so a bad pair fails at the call rather
    than inside the search.
    """
    pair = inv.get("scale_bounds")
    pair = (1e-4, 1e4) if pair is None else tuple(pair)
    if len(pair) != 2:
        raise ValueError(
            f"scale_bounds must hold exactly two values; got {len(pair)}.")
    low, high = (float(value) for value in pair)
    if not (0.0 < low <= high):
        raise ValueError(
            f"scale_bounds must be positive and ordered; got {low} and {high}.")
    return low, high

METHODS = ("FDEM", "TDEM")


def _tdem_calibration_view(
    data: Dict[str, Any], geom: Dict[str, Any]
) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """Data block and complete instrument geometry used for TDEM calibration."""
    moments = dict(data.get("moments", {}))
    if not moments:
        return data, _tdem_geometry(data, geom)
    requested = _normalise_temcompany_moment(str(geom.get("tem_moment", "LM+HM")))
    if requested in moments:
        name = requested
    elif "HM" in moments:
        # The joint reader exposes HM as its preview whenever HM exists.
        name = "HM"
    else:
        name = next(iter(moments))
    item = dict(moments[name])
    return item, _tdem_geometry(data, geom, item.get("transmitter"))


def _line_block(head: Dict[str, Any],
                lines: Optional[Sequence[int]]) -> "tuple[int, int]":
    """First station and count for a line selection, as an offset into the file.

    ``None`` means the whole file from its first station. Otherwise the stations
    on the named lines, which are contiguous because the reader orders them by
    line. A gap means the request would have to span a line nobody asked for, and
    that is refused: inverting an unrequested line under settings chosen for its
    neighbours is worse than declining.
    """
    n_total = int(head.get("n_soundings", 1))
    if lines is None:
        return 0, n_total
    wanted = {int(v) for v in np.asarray(lines, dtype=int).ravel()}
    if not wanted:
        raise ValueError("lines must name at least one survey line.")
    numbers = np.asarray(head.get("line_numbers", []), dtype=int).ravel()
    if numbers.size < n_total:
        raise ValueError(
            "this source does not record a line number per station, so it "
            "cannot be inverted one line at a time.")
    found = np.flatnonzero(np.isin(numbers[:n_total], sorted(wanted)))
    if not found.size:
        raise ValueError(
            f"no station is on line {sorted(wanted)}; the survey holds "
            f"{sorted(set(numbers[:n_total].tolist()))}.")
    if found.size != int(found[-1] - found[0] + 1):
        raise ValueError(
            f"lines {sorted(wanted)} are not adjacent in this survey, so they "
            "cannot be run as one block. Invert them one at a time.")
    return int(found[0]), int(found.size)


def _line_chi2_summary(
    chi2_per_sounding, data_counts, *, objective_chi2=None,
) -> Dict[str, float]:
    """Summarise line misfit without confusing gate and sounding weighting.

    Each sounding value is the mean squared uncertainty-normalised residual for
    that sounding.  The line objective therefore weights it by the number of
    retained gates.  Equal-sounding mean and median values are useful QC
    summaries, but they are not substitutes for the objective used by the fit.
    Aarhus software's per-model ``data residual`` is also reported on a
    residual/RMS scale, hence the square-root values returned here as a useful
    scale comparison. Its documented tTEM calculation uses log-data space, so
    the two values need not be numerically identical.
    """
    values = np.asarray(chi2_per_sounding, dtype=float).ravel()
    counts = np.asarray(data_counts, dtype=float).ravel()
    n = min(values.size, counts.size)
    values, counts = values[:n], counts[:n]
    valid = np.isfinite(values) & np.isfinite(counts) & (counts > 0)

    weighted = float("nan")
    if valid.any():
        weighted = float(np.sum(values[valid] * counts[valid]) / np.sum(counts[valid]))
    try:
        reported = float(objective_chi2)
    except (TypeError, ValueError):
        reported = float("nan")
    if np.isfinite(reported):
        weighted = reported

    finite_values = values[valid]
    sounding_mean = (float(np.mean(finite_values)) if finite_values.size
                     else float("nan"))
    sounding_median = (float(np.median(finite_values)) if finite_values.size
                       else float("nan"))
    return {
        "global": weighted,
        "sounding_mean": sounding_mean,
        "sounding_median": sounding_median,
        "data_residual_global": (float(math.sqrt(weighted))
                                 if np.isfinite(weighted) and weighted >= 0
                                 else float("nan")),
        "data_residual_sounding_median": (float(math.sqrt(sounding_median))
                                           if np.isfinite(sounding_median)
                                           and sounding_median >= 0
                                           else float("nan")),
    }


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
    studio's 1D forward by a near-constant amplitude factor. This fits each
    sounding's decay SHAPE to a grid of half-space forward responses at the current
    geometry and takes the geometric-mean amplitude ratio ``forward/observed`` at
    the best-fitting resistivity. Returns ``1.0`` if it cannot be estimated (so the
    caller can fall back to no scaling).
    """
    moment = str(geom.get("tem_moment", "HM"))
    use_flags = bool(geom.get("use_project_flags", True))
    tail_cut = geom.get("tail_max_relative_std")
    gate_rejection = str(geom.get("gate_rejection", "truncate"))
    reject_negative = bool(geom.get("reject_negative", False))
    min_gates_per_moment = geom.get("min_gates_per_moment")
    try:
        head = load_sounding(
            path, method, sounding=0, moment=moment, use_flags=use_flags,
            max_relative_std=tail_cut, gate_rejection=gate_rejection,
            reject_negative=reject_negative,
            min_gates_per_moment=min_gates_per_moment,
            ttem_loop_area=geom.get("loop_area"),
            ttem_gex_path=geom.get("ttem_gex_path"),
            ttem_tfi_path=geom.get("ttem_tfi_path"),
        )
    except Exception as exc:  # noqa: BLE001
        log(f"Auto-calibration skipped ({exc}); using data_scale = 1.0")
        return 1.0
    n_total = int(head.get("n_soundings", 1))
    probe = np.unique(np.linspace(0, n_total - 1, min(int(max_soundings), n_total)).astype(int))

    try:
        if method == "TDEM":
            from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
            calibration_data, calibration_geom = _tdem_calibration_view(head, geom)
            abscissa = np.asarray(calibration_data["times"], dtype=float).ravel()
            cfg = _tdem_config(calibration_geom, abscissa)
            md = TDEMForwardModeling(
                thicknesses=np.array([50.0]), survey_config=cfg)
            grid = []
            for R in np.geomspace(25.0, 3000.0, 20):
                grid.append(
                    float(calibration_geom.get("response_sign", 1.0))
                    * np.asarray(md.forward(np.array([1.0 / R, 1.0 / R])),
                                 dtype=float).ravel()[:abscissa.size]
                )
            preds = np.asarray(grid)

            def observed(s):
                return _response_on_times(
                    load_sounding(
                        path, method, sounding=int(s), moment=moment,
                        use_flags=use_flags, max_relative_std=tail_cut,
                        gate_rejection=gate_rejection,
                        reject_negative=reject_negative,
                        min_gates_per_moment=min_gates_per_moment,
                        ttem_loop_area=geom.get("loop_area"),
                        ttem_gex_path=geom.get("ttem_gex_path"),
                        ttem_tfi_path=geom.get("ttem_tfi_path"),
                    ),
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
                d = load_sounding(
                    path, method, sounding=int(s), moment=moment,
                    use_flags=use_flags, max_relative_std=tail_cut,
                        gate_rejection=gate_rejection,
                        reject_negative=reject_negative,
                        min_gates_per_moment=min_gates_per_moment,
                    ttem_loop_area=geom.get("loop_area"),
                    ttem_gex_path=geom.get("ttem_gex_path"),
                    ttem_tfi_path=geom.get("ttem_tfi_path"),
                )
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
    gate_rejection = str(geom.get("gate_rejection", "truncate"))
    reject_negative = bool(geom.get("reject_negative", False))
    min_gates_per_moment = geom.get("min_gates_per_moment")
    try:
        head = load_sounding(
            path, method, sounding=0, moment=moment, use_flags=use_flags,
            max_relative_std=tail_cut, gate_rejection=gate_rejection,
            reject_negative=reject_negative,
            min_gates_per_moment=min_gates_per_moment,
            ttem_loop_area=geom.get("loop_area"),
            ttem_gex_path=geom.get("ttem_gex_path"),
            ttem_tfi_path=geom.get("ttem_tfi_path"),
        )
        n_total = int(head.get("n_soundings", 1))
        probe = np.unique(np.linspace(0, n_total - 1, min(int(max_probe), n_total)).astype(int))
        if method == "TDEM":
            from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling
            calibration_data, calibration_geom = _tdem_calibration_view(head, geom)
            abscissa = np.asarray(calibration_data["times"], dtype=float).ravel()
            md = TDEMForwardModeling(
                thicknesses=np.array([50.0]),
                survey_config=_tdem_config(calibration_geom, abscissa))
            pred = (
                float(calibration_geom.get("response_sign", 1.0))
                * np.asarray(md.forward(np.array([1.0 / ref, 1.0 / ref])),
                             dtype=float).ravel()[:abscissa.size]
            )

            def observed(s):
                return _response_on_times(
                    load_sounding(
                        path, method, sounding=int(s), moment=moment,
                        use_flags=use_flags, max_relative_std=tail_cut,
                        gate_rejection=gate_rejection,
                        reject_negative=reject_negative,
                        min_gates_per_moment=min_gates_per_moment,
                        ttem_loop_area=geom.get("loop_area"),
                        ttem_gex_path=geom.get("ttem_gex_path"),
                        ttem_tfi_path=geom.get("ttem_tfi_path"),
                    ),
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
                d = load_sounding(
                    path, method, sounding=int(s), moment=moment,
                    use_flags=use_flags, max_relative_std=tail_cut,
                        gate_rejection=gate_rejection,
                        reject_negative=reject_negative,
                        min_gates_per_moment=min_gates_per_moment,
                    ttem_loop_area=geom.get("loop_area"),
                    ttem_gex_path=geom.get("ttem_gex_path"),
                    ttem_tfi_path=geom.get("ttem_tfi_path"),
                )
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


#: Geometry a station measures for itself rather than inheriting from the survey.
#:
#: A TEMcompany project records the transmitter-receiver distance and the two
#: heights per station. Only the distance actually varies on a walking ground
#: system, and it varies by more than the nominal layout suggests: one survey
#: spans 11.58 to 17.63 m against a spec that states 15.0 m for every station.
#: It is a per-station quantity.
_STATION_GEOMETRY_KEYS = ("tx_rx_sep", "height", "rx_height", "tx_height")

#: Distance bin the per-station transmitter-receiver separation is rounded to.
#:
#: Only used when ``per_station_geometry`` is switched on; see
#: :func:`_station_geometry` for why that is off by default. A quarter of a
#: metre is 1.7 percent of a typical 15 m offset, which is well inside what the
#: response can tell apart, and it takes one survey's 794 distinct distances
#: down to 25.
STATION_DISTANCE_BIN_M = 0.25


def _station_geometry(geom: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay one station's measured geometry on the survey-wide dictionary.

    On by default. The project records the distance per station, and a walking
    ground survey genuinely records a different one at nearly every station: 794
    distinct values over 929 stations on one line, spanning 11.58 to 17.63 m
    against a nominal 15.0.

    It was briefly off, because with the earlier forward path an operator took
    about fourteen seconds to build and a distinct distance per station turned a
    line inversion from minutes into hours. The native-order instrument chain
    removed that: SimPEG now models a compact step response, a build costs about
    twenty milliseconds, and the reason to switch it off went with it.

    It is also worth more than an earlier measurement suggested, because that
    measurement predated the instrument model above. Replacing the measured
    column with the nominal 15 m moves one survey's low-moment response by 1.4
    percent at the median and 18 percent at its worst gate.

    ``tx_rx_sep`` is still rounded to ``tx_rx_sep_bin`` metres, defaulting to
    :data:`STATION_DISTANCE_BIN_M`, which keeps the operator cache useful for
    little cost; set it to zero to pass the measured value through.

    A value the station did not record, or recorded as non-positive, leaves the
    survey-wide entry alone. That matters for ``tx_rx_sep``, where zero is how a
    failed measurement is stored rather than a coincident loop and coil.
    """
    if not bool(geom.get("per_station_geometry", True)):
        return geom
    system = data.get("system")
    if not isinstance(system, dict):
        return geom
    try:
        bin_m = float(geom.get("tx_rx_sep_bin", STATION_DISTANCE_BIN_M))
    except (TypeError, ValueError):
        bin_m = STATION_DISTANCE_BIN_M
    updates: Dict[str, Any] = {}
    for key in _STATION_GEOMETRY_KEYS:
        value = system.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        if key == "tx_rx_sep":
            if number <= 0.0:
                continue
            if bin_m > 0.0:
                number = round(number / bin_m) * bin_m
        updates[key] = number
    return {**geom, **updates} if updates else geom


def _with_sensor_height(geom: Dict[str, Any], height: Any) -> Dict[str, Any]:
    """Put a caller's own sensor height in charge of the whole geometry.

    All three keys, or the override does nothing. The forward reads
    ``rx_height`` and ``tx_height`` in preference to ``height``, and the station
    dictionary carries both, so setting ``height`` alone leaves a caller's
    heights silently ignored while looking as though they were applied.

    The loop and the coil end up at the same height, which is what a single
    number can say. Every TEMcompany project seen so far records them equal
    anyway; a survey that does not should pass its own geometry rather than one
    height per station.
    """
    try:
        value = float(height)
    except (TypeError, ValueError):
        return geom
    if not np.isfinite(value):
        return geom
    return {**geom, "height": value, "rx_height": value, "tx_height": value}


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


#: Half-spaces the automatic starting model is chosen from, in ohm-m. Twelve
#: values over three and a half decades put the grid about a third of a decade
#: apart, which is finer than the starting model needs to be: the inversion
#: moves from wherever it starts, and what matters is not landing a decade away.
_STARTING_HALF_SPACES = np.geomspace(3.0, 5000.0, 12)

#: Soundings the search evaluates. The starting model is one number for the
#: whole line, so it does not need every station to choose it, and a dozen
#: spread along the survey rank the candidates the same way the full set does.
_STARTING_SAMPLE = 12

def _best_starting_resistivity(blocks, n_layers: int, workers: int, *,
                               default: float, log: LogFn = _noop) -> float:
    """Pick the half-space whose forward response best matches the data.

    A starting model far from the ground costs more than iterations. The
    Gauss-Newton step is built from a linearization about the current model, and
    from a decade and a half away that linearization describes a different
    problem; the line search then shortens the step, the run spends its budget
    crossing the gap, and where it stops depends on where it started. On one
    ground survey the project's own 40 ohm-m default begins at a chi-squared of
    3.1e6 while the best half-space begins at 164.

    Cheap because it runs after the blocks are built: the first candidate warms
    the forward operators the inversion is about to use anyway, and every
    candidate after it is one forward per sampled sounding. Returns ``default``
    if nothing can be evaluated, so a forward that will not run here fails in
    the inversion rather than in the search.

    A half-space is a poor start for a layered conductive site, and two richer
    searches were built, measured and removed. Both are recorded here because
    neither failure is visible from the idea.

    **Layered candidates, ranked the same way, changed nothing.** On one
    conductive site running about 126, 27, 300 and 21 ohm-m with depth, a
    22.7 ohm-m half-space still scored best on initial misfit, 117 against 145
    for the two- and three-layer shapes, and the run ended identically. Initial
    misfit says how close a model already is, which on a multi-minimum problem
    is not where the solver goes from it: after four iterations those same
    layered candidates reached 37.6 while the half-space reached 46.2.

    **Deciding by trial worked where it was aimed and broke everything else.**
    Ranking cheaply and giving the best four a short run found the better
    minimum: that survey went from DataFit 3.39 to 2.83, and its median deep
    resistivity from a tenth of the reference model's to a half. But full-line
    trials cost more than the inversion they prepare, running past ten minutes
    on a 518-station line against a forty-second run. Sampling them to thirty
    soundings restored the cost and destroyed the answer, because a sampled
    ranking is not the full-line ranking: the same survey then chose the
    half-space again and lost the gain, while another regressed from
    chi-squared 1.6 to 9.5. A search that helps one survey and ruins another is
    worse than no search.

    What does work, on the same survey, is starting from an existing model:
    DataFit 1.79, better than the reference's own 1.98. Until a search can be
    made both cheap and representative of the whole line, pass
    ``initial_models`` rather than extending the scan here.
    """
    from PyHydroGeophysX.inversion.em1d_lci import (
        _forward_line, _misfit, _worker_pool, resolve_worker_count,
    )

    if not blocks:
        return default
    step = max(1, len(blocks) // _STARTING_SAMPLE)
    sampled = list(blocks)[::step][:_STARTING_SAMPLE]
    n_data = int(sum(block.dobs.size for block in sampled))
    if n_data <= 0:
        return default
    best_rho, best_chi2 = float(default), float("inf")
    try:
        with _worker_pool(resolve_worker_count(len(sampled), workers)) as pool:
            for rho in _STARTING_HALF_SPACES:
                x = np.full(len(sampled) * n_layers, math.log10(float(rho)))
                residual, _ = _misfit(sampled, _forward_line(sampled, x, n_layers, pool))
                chi2 = float(residual @ residual) / n_data
                if np.isfinite(chi2) and chi2 < best_chi2:
                    best_rho, best_chi2 = float(rho), chi2
    except Exception as exc:  # noqa: BLE001 - the inversion is the thing that must run
        log(f"  Starting-model search skipped ({exc}); using {default:g} ohm-m.")
        return default
    log(f"  Starting model: {best_rho:.0f} ohm-m, chosen from "
        f"{_STARTING_HALF_SPACES.size} half-spaces on {len(sampled)} soundings "
        f"(initial chi2 {best_chi2:.3g}).")
    return best_rho


def invert_line(path: str, method: str, geom: Dict[str, Any], inv: Dict[str, Any],
                *, spacing: float = 50.0, positions: Optional[np.ndarray] = None,
                heights: Optional[np.ndarray] = None, max_soundings: int = 12,
                lines: Optional[Sequence[int]] = None,
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

    ``inv["robust_errors"]`` retains all imported gates and iteratively inflates
    effective errors for large residuals. It overrides hard rejection. The main
    chi2 uses ORIGINAL errors; ``result["robust"]`` records effective errors and
    a separate effective chi2. Import-time flags and QC still apply.

    ``inv["auto_lambda"]`` re-solves the line at other smoothness
    weights to reach ``target_chi2``. ``inv["reject_outliers"]`` drops the gates
    the converged model cannot explain (beyond ``outlier_threshold`` sigma, over
    ``outlier_passes`` cycles, never below ``min_data_fraction`` of the gates)
    and solves again; what it removed is reported under ``result["outliers"]``.
    They address different causes, so they can be used together: relaxing the
    smoothness helps when the model is too stiff for the data, rejection helps
    when a minority of gates are simply wrong.

    ``lines`` restricts the run to the named survey lines, so a line whose data
    is thinner than the rest can be given its own settings instead of one set
    having to suit every line. Passing ``None`` runs from the first station, as
    before. The lateral constraint already groups by line, so a line inverted on
    its own is tied exactly as it would be inside a whole-survey run; what
    changes is which settings reach it, and that the other lines are not
    re-solved. ``max_soundings`` then counts within the selection.

    Stations arrive ordered by line, so a selection is a contiguous block. A set
    of lines that is not contiguous is refused rather than quietly widened to
    the span that encloses it, which would invert the lines in between under
    settings chosen for their neighbours.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}.")
    moment = (
        _normalise_temcompany_moment(str(geom.get("tem_moment", "HM")))
        if (is_temcompany_source(path) or is_ttem_source(path))
        else str(geom.get("tem_moment", "HM"))
    )
    use_flags = bool(geom.get("use_project_flags", True))
    tail_cut = geom.get("tail_max_relative_std")
    gate_rejection = str(geom.get("gate_rejection", "truncate"))
    reject_negative = bool(geom.get("reject_negative", False))
    min_gates_per_moment = geom.get("min_gates_per_moment")
    head = load_sounding(
        path, method, sounding=0, moment=moment, use_flags=use_flags,
        max_relative_std=tail_cut, gate_rejection=gate_rejection,
        reject_negative=reject_negative,
        min_gates_per_moment=min_gates_per_moment,
        ttem_loop_area=geom.get("loop_area"),
        ttem_gex_path=geom.get("ttem_gex_path"),
        ttem_tfi_path=geom.get("ttem_tfi_path"),
    )
    joint = method == "TDEM" and bool(head.get("moments"))
    invert = (
        fdem_invert if method == "FDEM"
        else tdem_joint_invert if joint
        else tdem_invert
    )
    n_total = int(head.get("n_soundings", 1))
    offset, n_available = _line_block(head, lines)
    n_pos = min(int(max_soundings), max(1, n_available))

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
    sequential = (
        lci_mode == "sequential" and joint and n_pos >= 2
        and float(inv.get("lateral_smoothness", 0.0)) > 0.0
        and int(inv.get("lci_passes", 1)) > 0
    )
    mode = ("simultaneous LCI" if simultaneous
            else "block-coordinate LCI" if (lci_mode == "sequential" and joint)
            else f"{method} independent 1D")
    selected = ("" if lines is None
                else f", line{'s' if len(set(lines)) > 1 else ''} "
                     f"{','.join(str(v) for v in sorted(set(lines)))}")
    log(f"Line inversion: {n_pos} of {n_total} soundings{selected} ({mode})")

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
    per_sounding_outliers: Dict[int, Dict[str, Any]] = {}
    per_sounding_robust: Dict[int, Dict[str, Any]] = {}
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
        and (simultaneous or sequential)
    )
    if use_common_lci_start:
        start = float(inv.get("starting_resistivity", 100.0))
        surface_models[:, :] = max(start, 1.0)
        log(f"Using a common {start:g} ohm-m starting model for the LCI.")
    t_ref = f_ref = None  # last time / min frequency, for the DOI estimate
    # Imported here rather than at module scope: this module is imported by the
    # CLI and the Qt app, and the LCI module pulls in SciPy sparse.
    from PyHydroGeophysX.inversion.em1d_lci import _worker_pool, resolve_worker_count

    workers = resolve_worker_count(n_pos, int(inv.get("parallel_workers", 0)))
    lci_supplies_model = (simultaneous or sequential) and (use_warm_models or use_common_lci_start)
    prior_context = bool(inv.get("shallow_prior_enabled", False))

    def prepare(s: int):
        """Read one station, and fit it unless the LCI will supply its model.

        Runs on a worker thread, so it touches nothing shared: the station's
        data, its geometry and its own result go back to the caller, which does
        the ordered bookkeeping. Its own exception travels with it for the same
        reason, since one station failing must not stop the line.
        """
        try:
            data = load_sounding(
                path, method, sounding=offset + s, moment=moment,
                use_flags=use_flags,
                max_relative_std=tail_cut, gate_rejection=gate_rejection,
                reject_negative=reject_negative,
                min_gates_per_moment=min_gates_per_moment,
                ttem_loop_area=geom.get("loop_area"),
                ttem_gex_path=geom.get("ttem_gex_path"),
                ttem_tfi_path=geom.get("ttem_tfi_path"),
            )
            geom_s = _station_geometry(geom, data)
            if hts is not None and s < hts.size:
                geom_s = _with_sensor_height(geom_s, hts[s])
            if lci_supplies_model or prior_context:
                return s, data, geom_s, None, None
            # Quiet inside the worker: the inner per-iteration lines would
            # interleave across stations. The caller logs one line per station,
            # in order, below.
            local_inv = {**inv, "starting_model": warm_models[s]} if use_warm_models else inv
            return s, data, geom_s, invert(data, geom_s, local_inv, log=_noop), None
        except Exception as exc:  # noqa: BLE001 - keep the line going
            return s, None, geom, None, exc

    if workers > 1:
        log(f"Reading and fitting {n_pos} soundings on {workers} threads")
    with _worker_pool(workers) as executor:
        prepared = ([prepare(s) for s in range(n_pos)] if executor is None
                    else list(executor.map(prepare, range(n_pos))))

    for s, data, geom_s, result, failure in prepared:
        if failure is not None:
            chi2_list.append(float("nan"))
            data_count_list.append(0)
            log(f"  sounding {s + 1}/{n_pos} failed: {failure}")
            continue
        datasets[s] = data
        geometries[s] = geom_s
        if t_ref is None and "times" in data and np.size(data["times"]):
            t_ref = float(np.asarray(data["times"]).ravel()[-1])
        if f_ref is None and "frequencies" in data and np.size(data["frequencies"]):
            f_ref = float(np.asarray(data["frequencies"]).ravel().min())
        if result is None:
            # The LCI supplies the model, so the per-sounding inversion is
            # skipped; only the data count is needed here.
            chi2_list.append(float("nan"))
            data_count_list.append(_sounding_data_count(data, method))
            continue
        res = np.asarray(result["resistivity"], dtype=float).ravel()
        surface_models[s, :] = res
        model[s, 0, :] = res[::-1]  # deepest layer first to match ez ordering
        chi2_list.append(float(result.get("chi2", np.nan)))
        data_count_list.append(int(result.get("n_data", 0)))
        if bool(result.get("outliers", {}).get("enabled", False)):
            per_sounding_outliers[s] = dict(result["outliers"])
        if result.get("robust", {}).get("enabled"):
            per_sounding_robust[s] = result["robust"]
        log(f"  sounding {s + 1}/{n_pos}: chi2={result.get('chi2', float('nan')):.3f}")

    embedded_positions = np.asarray(head.get("positions", []), dtype=float).ravel()
    requested_positions = (
        np.asarray(positions, dtype=float).ravel()
        if positions is not None else embedded_positions
    )
    if requested_positions.size >= offset + n_pos:
        pos_lci = requested_positions[offset:offset + n_pos]
    else:
        pos_lci = np.arange(n_pos, dtype=float) * float(spacing)
    # Ground level per sounding, carried through for plotting only.
    embedded_elevation = np.asarray(head.get("elevation", []), dtype=float).ravel()
    surface_elevation = (
        embedded_elevation[offset:offset + n_pos]
        if embedded_elevation.size >= offset + n_pos
        else np.full(n_pos, np.nan, dtype=float)
    )
    embedded_lines = np.asarray(head.get("line_numbers", []), dtype=int).ravel()
    line_numbers = (
        embedded_lines[offset:offset + n_pos]
        if embedded_lines.size >= offset + n_pos
        else np.zeros(n_pos, dtype=int)
    )

    def _per_sounding(key: str, dtype=float):
        """A per-station column from the source, cut to the inverted stations."""
        values = np.asarray(head.get(key, []), dtype=dtype).ravel()
        if values.size >= offset + n_pos:
            return values[offset:offset + n_pos]
        return np.full(n_pos, np.nan if dtype is float else "", dtype=dtype)

    # Map coordinates travel with the section so an export can place each model
    # in the ground rather than only along the line.
    easting, northing = _per_sounding("x"), _per_sounding("y")
    longitude, latitude = _per_sounding("longitude"), _per_sounding("latitude")
    station_ids = _per_sounding("station_ids", dtype=object)

    from PyHydroGeophysX.inversion.em1d_priors import shallow_prior_scores
    quality_rows = None
    if prior_context and any(data and "raw_lm_quality" in data for data in datasets):
        from PyHydroGeophysX.inversion.em1d_priors import raw_lm_quality_rows
        quality_rows = raw_lm_quality_rows(
            datasets, int(inv.get("shallow_prior_reference_gate", 2)))
    signal_limits = None
    if prior_context and inv.get("shallow_prior_mode", "quality_trend") == "signal_threshold":
        from PyHydroGeophysX.inversion.em1d_priors import shallow_signal_thresholds
        log("Calibrating the resistive-background LM signal limit using the instrument forward model.")
        signal_limits = shallow_signal_thresholds(datasets, geometries, inv)
        available = signal_limits[np.isfinite(signal_limits)]
        if available.size:
            log(f"  LM signal threshold: {available.min():.4g} .. {available.max():.4g} "
                "(stored project response units; homogeneous reference, not a depth estimate).")
        else:
            log("  No raw LM diagnostics available: absolute-signal prior cannot activate. Re-import the project.")
    prior_scores, prior_report = shallow_prior_scores(
        datasets, pos_lci, line_numbers, inv, quality_rows, signal_limits)

    def station_inv(s):
        options = {**inv, "_shallow_prior_score": float(prior_scores[s])}
        if use_warm_models:
            # The automatic soft target follows the model that actually starts
            # this station, not a stale project fallback value.
            valid = warm_models[s][np.isfinite(warm_models[s]) & (warm_models[s] > 0.)]
            if valid.size:
                options["_resistive_prior_reference_resistivity"] = float(
                    10. ** np.mean(np.log10(valid)))
        return options

    if prior_context:
        if quality_rows is None and inv.get("shallow_prior_mode", "quality_trend") == "quality_trend":
            log("  Resistive-background prior uses imported LM quality; raw fixed-gate signal/noise "
                "checks are unavailable for this input. Re-import a TEMcompany project "
                "to preserve the raw quality diagnostics.")
        log(f"Empirical resistive-background prior: "
            f"{prior_report['active_soundings']}/{n_pos} stations activated; whole-model "
            f"one-sided tendency (not a shallow-depth interpretation), weight "
            f"{prior_report['weight']:g}.")
        if not lci_supplies_model:
            # Spatial quality needs the read-only first pass over the line before
            # independent fits can receive their individual prior weights.
            def fit_with_prior(s):
                try:
                    options = station_inv(s)
                    if use_warm_models:
                        options["starting_model"] = warm_models[s]
                    return s, invert(datasets[s], geometries[s], options, log=_noop), None
                except Exception as exc:
                    return s, None, exc
            usable_prior = [s for s in range(n_pos) if datasets[s] is not None]
            with _worker_pool(workers) as pool:
                fits = (list(map(fit_with_prior, usable_prior)) if pool is None
                        else list(pool.map(fit_with_prior, usable_prior)))
            for s, fit, failure in fits:
                if failure:
                    log(f"  sounding {s+1} failed: {failure}")
                    data_count_list[s] = 0
                    continue
                surface_models[s] = fit["resistivity"]
                chi2_list[s], data_count_list[s] = float(fit["chi2"]), int(fit["n_data"])
                if fit.get("robust", {}).get("enabled"):
                    per_sounding_robust[s] = fit["robust"]
                if fit.get("outliers", {}).get("enabled"):
                    per_sounding_outliers[s] = fit["outliers"]

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
    lateral_distance_power = max(
        float(inv.get("lateral_distance_power", 1.0)), 0.0)
    lci_report: Dict[str, Any] = {}
    outlier_info: Dict[str, Any] = {"enabled": False}
    robust_info: Dict[str, Any] = {"enabled": False}
    # Kept for the depth-of-investigation pass below, which reads the same
    # analytic Jacobian the coupled solver used.
    doi_blocks: Dict[int, Any] = {}
    if simultaneous:
        from PyHydroGeophysX.inversion.em1d import build_sounding_block
        from PyHydroGeophysX.inversion.em1d_lci import (
            invert_lci,
            invert_lci_rejecting_outliers,
            invert_lci_with_robust_errors,
        )

        usable = [s for s in range(n_pos) if datasets[s] is not None]
        def build(s: int):
            """Assemble one station's block, or hand back why it could not be."""
            try:
                return s, build_sounding_block(
                    datasets[s], geometries[s], station_inv(s), method,
                    position=float(pos_lci[s]), line=int(line_numbers[s]),
                    label=f"sounding {s + 1}"), None
            except Exception as exc:  # noqa: BLE001 - one bad station is not fatal
                return s, None, exc

        # Worth parallelizing in its own right: each block constructs a SimPEG
        # simulation and pays that operator's one-time setup, which on a long
        # line adds up to more than the coupled solve it feeds.
        with _worker_pool(resolve_worker_count(len(usable), workers)) as executor:
            built = ([build(s) for s in usable] if executor is None
                     else list(executor.map(build, usable)))
        sounding_blocks = []
        kept: List[int] = []
        for s, block, failure in built:
            if failure is not None:
                log(f"  sounding {s + 1} excluded from the LCI: {failure}")
                continue
            sounding_blocks.append(block)
            kept.append(s)
        if len(kept) < 2:
            # The per-sounding pass was skipped on the assumption the LCI would
            # supply the models, so it has to run now or nothing is inverted.
            log("  Fewer than two usable soundings; falling back to independent 1D.")
            simultaneous = False
            for s in usable:
                try:
                    result = invert(datasets[s], geometries[s], station_inv(s), log=log)
                    surface_models[s, :] = np.asarray(
                        result["resistivity"], dtype=float).ravel()
                    chi2_list[s] = float(result.get("chi2", np.nan))
                    data_count_list[s] = int(result.get("n_data", 0))
                    if bool(result.get("outliers", {}).get("enabled", False)):
                        per_sounding_outliers[s] = dict(result["outliers"])
                    if result.get("robust", {}).get("enabled"):
                        per_sounding_robust[s] = result["robust"]
                    log(f"  sounding {s + 1}/{n_pos}: "
                        f"chi2={result.get('chi2', float('nan')):.3f}")
                except Exception as exc:  # noqa: BLE001
                    log(f"  sounding {s + 1}/{n_pos} failed: {exc}")
            model[:, 0, :] = surface_models[:, ::-1]
        else:
            log(f"Simultaneous LCI: {len(kept)} soundings, lateral="
                f"{lateral:g}, vertical={float(inv.get('smoothness', 0.3)):g}")
            warm = (surface_models[kept] if use_warm_models else None)
            start_resistivity = float(inv.get("starting_resistivity", 100.0))
            if warm is None and bool(inv.get("auto_starting_model", True)):
                start_resistivity = _best_starting_resistivity(
                    sounding_blocks, n_layers, workers,
                    default=start_resistivity, log=log)
            if prior_context:
                # Block construction precedes the data-driven starting-model
                # search. Rebuild only the cheap prior vectors here, using the
                # half-space that the optimiser will actually start from; the
                # expensive forward operators are retained unchanged.
                from PyHydroGeophysX.inversion.em1d_priors import (
                    resistive_prior_target,
                    shallow_prior_terms,
                )
                targets = []
                references = []
                for block, s in zip(sounding_blocks, kept):
                    options = station_inv(s)
                    if warm is None:
                        options["_resistive_prior_reference_resistivity"] = start_resistivity
                    block.prior_lower, block.prior_weights = shallow_prior_terms(options, thick)
                    reference, target, _, source = resistive_prior_target(options)
                    references.append(reference)
                    targets.append(target)
                prior_report["reference_resistivity"] = float(np.median(references))
                prior_report["target_resistivity"] = float(np.median(targets))
                prior_report["minimum_resistivity"] = prior_report["target_resistivity"]
                prior_report["target_source"] = source
                if source == "explicit":
                    target_description = (
                        f"explicit {prior_report['target_resistivity']:.0f} ohm-m")
                else:
                    target_description = (
                        f"effective starting model "
                        f"{prior_report['reference_resistivity']:.0f} ohm-m × "
                        f"{prior_report['resistivity_factor']:g} → "
                        f"{prior_report['target_resistivity']:.0f} ohm-m")
                log(f"  Background soft tendency: {target_description} "
                    "(capped by rho_max; all layers, not a depth estimate).")
            lci_kwargs = dict(
                solver=str(inv.get("lci_solver", "trf")),
                trf_max_nfev=int(inv.get("lci_max_nfev", 90)),
                trf_ftol=float(inv.get("lci_ftol", 1e-4)),
                trf_xtol=float(inv.get("lci_xtol", 1e-6)),
                trf_gtol=float(inv.get("lci_gtol", 1e-5)),
                smoothness=float(inv.get("smoothness", 0.3)),
                lateral_smoothness=lateral * lateral_weight_scale,
                reference_distance=reference_distance,
                lateral_distance_power=lateral_distance_power,
                model_damping=float(inv.get("model_damping", 0.0)),
                starting_resistivity=start_resistivity,
                max_iterations=int(inv.get("max_iterations", 20)),
                convergence_tolerance=float(inv.get("convergence_tolerance", 0.02)),
                min_iterations=int(inv.get("min_iterations", 2)),
                auto_lambda=bool(inv.get("auto_lambda", True)),
                target_chi2=float(inv.get("target_chi2", 1.0)),
                chi2_tolerance=float(inv.get("chi2_tolerance", 0.2)),
                max_lambda_trials=int(inv.get("max_lambda_trials", 5)),
                # How far auto-lambda may move the smoothness. The default span
                # is four decades either way, which on a station carrying four
                # or five gates buys a chi-squared of 1 with a model that swings
                # to match noise. A caller that wants the search available but
                # bounded passes something like (0.5, 2.0).
                scale_bounds=_scale_bounds(inv),
                bounds=_log_resistivity_bounds(inv),
                parallel_workers=workers,
                verbose=bool(inv.get("verbose", True)),
            )
            doi_blocks.update(zip(kept, sounding_blocks))
            if bool(inv.get("robust_errors", False)):
                from PyHydroGeophysX.inversion.robust_errors import robust_error_options

                log("Robust error weighting: retain every imported gate; hard rejection bypassed.")
                error_options = robust_error_options(inv)
                error_options["error_target_chi2"] = error_options.pop("target_chi2")
                outcome, sounding_blocks, robust_info = invert_lci_with_robust_errors(
                    sounding_blocks, n_layers, initial_model=warm, log=log,
                    **error_options, **lci_kwargs)
                robust_info["sounding_indices"] = list(kept)
                log(f"  Robust weighting finished: {robust_info['kept']} gates retained; "
                    f"{robust_info['downweighted']} downweighted; "
                    f"{robust_info['unchanged_fraction']:.1%} errors unchanged.")
                if robust_info.get("target_chi2", 0) > 0:
                    log(f"  Effective chi2 target {robust_info['target_chi2']:g} "
                        f"± {robust_info['target_tolerance']:g}: "
                        f"{'reached' if robust_info['target_reached'] else 'not reached'}; "
                        f"current-model lower bound under error limits="
                        f"{robust_info['final_model_error_limits']['fixed_model_min_chi2']:.3f}.")
            elif bool(inv.get("reject_outliers", False)):
                log(f"Outlier rejection: cut beyond "
                    f"{float(inv.get('outlier_threshold', 3.0)):g} sigma, "
                    f"{int(inv.get('outlier_passes', 2))} pass(es), keeping at least "
                    f"{int(float(inv.get('min_data_fraction', 0.8)) * 100)} % of the gates "
                    f"and {int(inv.get('min_gates_per_sounding', 3))} per sounding.")
                outcome, sounding_blocks, outlier_info = invert_lci_rejecting_outliers(
                    sounding_blocks, n_layers,
                    threshold=float(inv.get("outlier_threshold", 3.0)),
                    passes=int(inv.get("outlier_passes", 2)),
                    min_fraction=float(inv.get("min_data_fraction", 0.8)),
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
                chi2_list[s] = float(
                    robust_info["chi2_per_sounding_original"][index]
                    if robust_info["enabled"] else outcome.chi2_per_sounding[index])
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
                "chi2_median": list(outlier_info.get("initial", {}).get(
                    "convergence_median", outcome.chi2_median_history)),
                "n_data": int(outlier_info.get("initial", {}).get(
                    "n_data", sum(b.dobs.size for b in sounding_blocks))),
            }]
            for entry in outlier_info.get("passes") or []:
                track.append({
                    "stage": f"reject {entry['pass']}",
                    "lambda": float(outcome.smoothness_scale),
                    "chi2": list(entry.get("convergence") or []),
                    "chi2_median": list(entry.get("convergence_median") or []),
                    "n_data": int(entry.get("kept", 0)),
                })
            if robust_info["enabled"]:
                # Each stage uses its own effective errors. The main chi2 below
                # always uses original errors and all original gates.
                track = [{"stage": "initial" if entry["pass"] == 0 else f"reweight {entry['pass']}",
                          "lambda": float(outcome.smoothness_scale),
                          "chi2": entry["convergence"], "n_data": entry["kept"],
                          "chi2_median": list(entry.get("convergence_median") or []),
                          "chi2_original_median": entry.get("chi2_original_median")}
                         for entry in [robust_info["initial"], *robust_info["passes"]]]
            lci_report = {
                "mode": "simultaneous",
                "chi2": robust_info.get("chi2_original", outcome.chi2),
                "chi2_effective": outcome.chi2,
                "chi2_history": outcome.chi2_history,
                "chi2_median_history": outcome.chi2_median_history,
                "chi2_effective_sounding_median": float(np.nanmedian(outcome.chi2_per_sounding)),
                "convergence_track": track,
                "iterations": robust_info.get("total_iterations", outcome.iterations),
                "stop_reason": outcome.stop_reason,
                "diagnostics": outcome.diagnostics,
                "smoothness_scale": outcome.smoothness_scale,
                "lambda_search": robust_info.get("initial_lambda_search", outcome.lambda_search),
                "seconds": robust_info.get("solve_seconds", outcome.seconds),
                "n_soundings": len(kept),
                "n_lateral_ties": int(sum(max(count - 1, 0) for count in
                    np.unique([b.line for b in sounding_blocks], return_counts=True)[1])),
            }
            log(f"  LCI done: chi2={lci_report['chi2']:.3f} after "
                f"{lci_report['iterations']} total iteration(s) ({outcome.stop_reason}), "
                f"{lci_report['seconds']:.1f}s")
    if not simultaneous and sequential:
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
                distances = np.asarray([
                    max(abs(float(pos_lci[index] - pos_lci[s])), reference_distance)
                    for index in neighbors
                ])
                weights = (reference_distance / distances) ** lateral_distance_power
                # A one-station survey line still needs a real independent fit,
                # even when other lines make the overall selection sequential.
                reference_log = (np.average(
                    np.log10(previous[neighbors]), axis=0, weights=weights)
                    if neighbors else np.log10(previous[s]))
                local_inv = {
                    **station_inv(s),
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
                    data_count_list[s] = int(result.get("n_data", 0))
                    if bool(result.get("outliers", {}).get("enabled", False)):
                        per_sounding_outliers[s] = dict(result["outliers"])
                    if result.get("robust", {}).get("enabled"):
                        per_sounding_robust[s] = result["robust"]
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

    if not simultaneous and bool(inv.get("robust_errors", False)):
        entries = [{"sounding": s, **report} for s, report in sorted(per_sounding_robust.items())]
        total = sum(entry["kept"] for entry in entries)
        robust_info = {
            "enabled": True, "mode": "per_sounding", "soundings": entries,
            "n_start": total, "kept": total, "dropped": 0,
            "downweighted": sum(entry["downweighted"] for entry in entries),
            "unchanged": sum(entry["unchanged"] for entry in entries),
            "unchanged_fraction": (sum(entry["unchanged"] for entry in entries) / total
                                   if total else float("nan")),
            "min_unchanged_fraction": float(inv.get("robust_min_unchanged_fraction", 0.0)),
            "fraction_scope": "per_sounding",
            "target_chi2": float(inv.get("robust_target_chi2", 0.0)),
            "target_tolerance": float(inv.get("robust_target_tolerance", .25)),
            "chi2_original": (sum(e["chi2_original"] * e["kept"] for e in entries) / total
                              if total else float("nan")),
            "chi2_effective": (sum(e["chi2_effective"] * e["kept"] for e in entries) / total
                               if total else float("nan")),
        }
        robust_info["target_reached"] = (
            abs(robust_info["chi2_effective"] - robust_info["target_chi2"]) <= robust_info["target_tolerance"]
            if robust_info["target_chi2"] > 0 else None)
        # Carry the effective errors into sensitivity/DOI, not just the fit.
        from PyHydroGeophysX.inversion.em1d import build_sounding_block
        for s, report in per_sounding_robust.items():
            try:
                block = build_sounding_block(datasets[s], geometries[s], station_inv(s), method,
                                            position=float(pos_lci[s]), line=int(line_numbers[s]))
                block.uncertainty = np.asarray(report["uncertainty_effective"], dtype=float)
                doi_blocks[s] = block
            except Exception as exc:
                log(f"  Robust sensitivity unavailable at sounding {s + 1}: {exc}")
    elif not simultaneous and bool(inv.get("reject_outliers", False)):
        entries = [
            {"sounding": index, **per_sounding_outliers[index]}
            for index in sorted(per_sounding_outliers)
        ]
        outlier_info = {
            "enabled": True,
            "mode": "per_sounding",
            "soundings": entries,
            "n_start": int(sum(item.get("n_start", 0) for item in entries)),
            "kept": int(sum(item.get("kept", 0) for item in entries)),
            "dropped": int(sum(item.get("dropped", 0) for item in entries)),
        }

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
    # Keep the optimizer's gate-weighted objective as the headline value.  The
    # equal-sounding mean and median remain available for spatial QC.
    chi2_summary = _line_chi2_summary(
        chi2_list, data_count_list, objective_chi2=lci_report.get("chi2"),
    )
    chi2_global = chi2_summary["global"]
    chi2_effective_list = list(chi2_list)
    if robust_info.get("enabled"):
        chi2_effective_list = [float("nan")] * n_pos
        if robust_info.get("mode") == "per_sounding":
            for entry in robust_info.get("soundings", []):
                chi2_effective_list[int(entry["sounding"])] = float(entry["chi2_effective"])
        else:
            weighted = (np.asarray(robust_info["residual_original"], float)
                        / np.asarray(robust_info["error_factor"], float))
            cursor = 0
            for s, count in enumerate(data_count_list):
                if count:
                    chi2_effective_list[s] = float(np.mean(weighted[cursor:cursor+count]**2))
                cursor += count
    data_residual_list = [
        float(math.sqrt(value)) if np.isfinite(value) and value >= 0 else float("nan")
        for value in chi2_list
    ]
    result = {
        "method": method, "edges": (ex, ey, ez), "model3d": model,
        "label": "resistivity (Ω·m)", "cmap": "turbo", "log_scale": True,
        "positions": pos, "depth_edges": depth_edges, "thickness": thick,
        "sensitivity": sensitivity, "doi": doi, "doi_threshold": doi_threshold,
        # Ground level at each sounding, so a section can be drawn against
        # elevation instead of depth. The inversion itself is per sounding and
        # does not use it: each 1D model starts at its own ground surface.
        "surface_elevation": surface_elevation,
        "x": easting, "y": northing,
        "longitude": longitude, "latitude": latitude,
        "station_ids": station_ids,
        "coordinate_system": str(head.get("coordinate_system", "")),
        # ``chi2`` remains an alias for compatibility. It is the whole-line,
        # gate-weighted mean squared normalized residual for every solve mode.
        "chi2": chi2_global, "chi2_global": chi2_global,
        "chi2_sounding_mean": chi2_summary["sounding_mean"],
        "chi2_sounding_median": chi2_summary["sounding_median"],
        "data_residual_global": chi2_summary["data_residual_global"],
        "data_residual_sounding_median": chi2_summary["data_residual_sounding_median"],
        "chi2_list": chi2_list, "data_residual_list": data_residual_list,
        "chi2_effective_list": chi2_effective_list,
        "n_soundings": n_pos,
        "n_layers": n_layers, "n_data": int(sum(data_count_list)),
        "data_count_list": data_count_list, "data_scale": data_scale_used,
        "joint_moments": joint, "lci": bool(lci_report),
        "lci_mode": lci_report.get("mode", "off"), "lci_report": lci_report,
        "outliers": outlier_info,
        "robust": robust_info,
        "shallow_prior": prior_report,
        "chi2_effective": robust_info.get("chi2_effective", chi2_global),
        "lateral_smoothness": lateral, "lci_passes": lci_passes,
        "lateral_weight_scale": lateral_weight_scale,
        "lateral_distance_power": lateral_distance_power,
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
                 surface_elevation=surface_elevation,
                 x=easting, y=northing, longitude=longitude, latitude=latitude)
        result["saved"] = [str(out / "resistivity_section.npz")]
        if lci_report:
            result["saved"].append(str(table_io.write_json(out / "lci_report.json", lci_report)))
        if prior_report.get("enabled"):
            result["saved"].append(str(table_io.write_json(out / "shallow_prior.json", prior_report)))
            result["saved"].append(str(table_io.write_csv(
                out / "shallow_prior.csv",
                zip(station_ids, line_numbers, prior_report["line_distance_m"],
                    prior_report["early_lm_snr"], prior_report["smoothed_snr_ratio"],
                    prior_report["signal_ratio"], prior_report["noise_ratio"],
                    prior_report["signal_threshold"], prior_report["signal_to_threshold"], prior_report["score"]),
                header=["station", "line", "line_distance_m", "early_lm_snr",
                        "smoothed_snr_ratio", "signal_ratio", "noise_ratio",
                        "signal_threshold", "signal_to_threshold", "prior_score"])))
        log(f"  saved {out / 'resistivity_section.npz'}")
        for written in save_line_csv(result, out):
            result["saved"].append(written)
            log(f"  saved {written}")
        if robust_info.get("enabled"):
            # Gate order is LM then HM for joint TDEM, exactly as the block
            # assembler uses it. Keep identifiers so sparse early gates can be audited.
            rows = []
            offsets = robust_info.get("block_offsets", [])
            reports = ([(entry["sounding"], entry, 0, entry["kept"])
                        for entry in robust_info["soundings"]]
                       if robust_info.get("mode") == "per_sounding" else
                       [(s, robust_info, offsets[i], offsets[i + 1])
                        for i, s in enumerate(robust_info["sounding_indices"])])
            for s, report, begin, end in reports:
                data = datasets[s]
                if method == "TDEM":
                    moments = data.get("moments") or {"TDEM": data}
                    labels = [(name, i, float(t)) for name in ("LM", "HM", "TDEM")
                              if name in moments for i, t in enumerate(moments[name]["times"])]
                else:
                    labels = [(name, i, float(f)) for name in ("real", "imag")
                              for i, f in enumerate(data["frequencies"])]
                for j, k in enumerate(range(begin, end)):
                    name, gate, coordinate = labels[j]
                    rows.append((str(station_ids[s]), int(line_numbers[s]), name, gate, coordinate,
                                 report["observed"][k], report["predicted"][k],
                                 report["uncertainty_original"][k], report["uncertainty_effective"][k],
                                 report["error_factor"][k], report["weights"][k],
                                 report["residual_original"][k]))
            result["saved"].append(str(table_io.write_csv(
                out / "robust_gate_errors.csv", rows,
                header=["station", "line", "moment", "gate_index", "time_s_or_frequency_hz",
                        "observed", "predicted", "error_original", "error_effective",
                        "error_factor", "inverse_variance_weight", "residual_original"])))
            result["saved"].append(str(table_io.write_json(out / "robust_errors.json", robust_info)))
    return result


def save_line_csv(result: Dict[str, Any], out_dir: Path) -> List[str]:
    """Write the section as two flat tables; return the paths written.

    ``model_cells.csv`` is one row per layer per sounding, which is the form a
    GIS or a gridding package wants: every row carries its own map coordinate
    and its own elevation, so the section can be reconstructed without knowing
    anything about the layer grid. ``soundings.csv`` is the per-station summary
    that would otherwise have to be recovered by grouping the first table.

    Depths are below each station's own ground level, and ``z`` is the elevation
    of the cell centre where the survey carries ground elevations. A cell below
    the depth of investigation is written out with its resistivity and flagged
    rather than dropped: what the inversion produced there is still the answer
    to a question the data cannot settle, and a reader filtering on the flag can
    decide for themselves.
    """
    out = table_io.ensure_dir(out_dir)
    res = np.asarray(result["model3d"], dtype=float)[:, 0, :][:, ::-1]  # surface first
    depth_edges = np.asarray(result["depth_edges"], dtype=float).ravel()
    n_pos, n_layers = res.shape
    top, bottom = depth_edges[:n_layers], depth_edges[1:n_layers + 1]
    centre = 0.5 * (top + bottom)

    def column(key: str, fill=np.nan) -> np.ndarray:
        values = np.asarray(result.get(key, []), dtype=float).ravel()
        return values[:n_pos] if values.size >= n_pos else np.full(n_pos, fill)

    lines = np.asarray(result.get("line_numbers", []), dtype=int).ravel()
    lines = lines[:n_pos] if lines.size >= n_pos else np.zeros(n_pos, dtype=int)
    stations = np.asarray(result.get("station_ids", []), dtype=object).ravel()
    stations = stations[:n_pos] if stations.size >= n_pos else np.arange(1, n_pos + 1)
    surface = column("surface_elevation")
    x, y = column("x"), column("y")
    longitude, latitude = column("longitude"), column("latitude")
    position, chi2 = column("positions"), column("chi2_list")
    doi = column("doi", fill=np.inf)
    counts = np.asarray(result.get("data_count_list", []), dtype=float).ravel()
    counts = counts[:n_pos] if counts.size >= n_pos else np.zeros(n_pos)
    sensitivity = np.asarray(result.get("sensitivity", []), dtype=float)
    has_sensitivity = sensitivity.shape == res.shape

    cells = []
    for s in range(n_pos):
        for k in range(n_layers):
            cells.append((
                int(lines[s]), stations[s], _round(x[s], 3), _round(y[s], 3),
                _round(longitude[s], 8), _round(latitude[s], 8),
                _round(surface[s], 3), _round(position[s], 3),
                _round(top[k], 3), _round(bottom[k], 3), _round(centre[k], 3),
                _round(surface[s] - centre[k], 3),
                _round(res[s, k], 6),
                _round(sensitivity[s, k], 6) if has_sensitivity else "",
                int(centre[k] > doi[s]),
                _round(chi2[s], 4),
            ))
    paths = [str(table_io.write_csv(
        out / "model_cells.csv", cells,
        header=["line", "station", "x", "y", "longitude", "latitude",
                "surface_elevation", "distance_m", "depth_top_m",
                "depth_bottom_m", "depth_center_m", "z", "resistivity_ohm_m",
                "sensitivity", "below_doi", "chi2"]))]
    summary = [(
        int(lines[s]), stations[s], _round(x[s], 3), _round(y[s], 3),
        _round(longitude[s], 8), _round(latitude[s], 8),
        _round(surface[s], 3), _round(position[s], 3), _round(chi2[s], 4),
        int(counts[s]), _round(doi[s], 3) if np.isfinite(doi[s]) else "",
    ) for s in range(n_pos)]
    paths.append(str(table_io.write_csv(
        out / "soundings.csv", summary,
        header=["line", "station", "x", "y", "longitude", "latitude",
                "surface_elevation", "distance_m", "chi2", "n_data", "doi_m"])))
    return paths


def _round(value: float, digits: int):
    """A finite number rounded for a table; an empty field for anything else."""
    number = float(value)
    return round(number, digits) if np.isfinite(number) else ""


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
    robust = result.get("robust") or {}
    if robust.get("enabled"):
        keys = ("observed", "predicted", "uncertainty_original", "uncertainty_effective",
                "error_factor", "weights", "residual_original")
        if result["method"] == "FDEM":
            labels = [(name, i, float(f)) for name in ("real", "imag")
                      for i, f in enumerate(result["frequencies"])]
        else:
            moments = result.get("moments") or {"TDEM": result}
            labels = [(name, i, float(t)) for name, item in moments.items()
                      for i, t in enumerate(item["times"])]
        rows = [(*labels[i], *(robust[key][i] for key in keys))
                for i in range(robust["kept"])]
        paths.append(str(table_io.write_csv(
            out / "robust_gate_errors.csv", rows,
            header=["moment", "gate_index", "time_s_or_frequency_hz", "observed", "predicted",
                    "error_original", "error_effective", "error_factor",
                    "inverse_variance_weight", "residual_original"])))
        paths.append(str(table_io.write_json(out / "robust_errors.json", robust)))
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
    "is_ttem_source",
    "load_temcompany_sounding",
    "load_ttem_sounding",
    "load_sounding",
    "load_sounding_container",
    "save_sounding_container",
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
    "save_line_csv",
]
