"""Line inversion of 1D EM soundings, and the amplitude calibration it can start from.

:func:`invert_line` inverts a line of FDEM or TDEM soundings on one shared layer
grid: each sounding on its own, in the older block-coordinate passes, or as one
laterally constrained system (:mod:`~PyHydroGeophysX.inversion.em1d_lci`). The
single-sounding solvers it calls are in :mod:`~PyHydroGeophysX.inversion.em1d`,
the readers in :mod:`~PyHydroGeophysX.data_processing.em1d`.

:func:`estimate_data_scale` and :func:`calibrate_to_reference` fix the amplitude
scale of normalised airborne data before a run, against the forward model or
against a known background resistivity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop
from PyHydroGeophysX.data_processing import table_io
from PyHydroGeophysX.data_processing.em1d import (
    _normalise_temcompany_moment,
    _response_on_times,
    is_temcompany_source,
    is_ttem_source,
    load_sounding,
    save_line_csv,
    sounding_options,
)
from PyHydroGeophysX.forward.em1d import _fdem_config, _tdem_config, _tdem_geometry
from PyHydroGeophysX.inversion.em1d import (
    DEFAULT_INVERSION,
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


def _station_name(station_id: Any, index: int) -> str:
    """A station's own id, or its 1-based place in the file when it has none."""
    text = "" if station_id is None else str(station_id).strip()
    return text if text and text.lower() != "nan" else f"sounding {int(index) + 1}"


#: Failed soundings named in a line's warning; the rest are counted, and all of
#: them are listed in the result's ``failed_soundings``.
_FAILURES_NAMED = 10


def _failure_summary(n_inverted: int, n_total: int,
                     failed: Sequence[Dict[str, Any]]) -> str:
    """One sentence naming the soundings a line inversion could not fit."""
    named = []
    for item in list(failed)[:_FAILURES_NAMED]:
        where = f"line {item['line']}" + (
            f", {item['position_m']:.0f} m" if item.get("position_m") is not None else "")
        reason = str(item.get("reason", "")).strip().splitlines()
        reason = reason[0][:120].rstrip(". ") if reason else "no reason given"
        named.append(f"{item['station']} ({where}): {reason}")
    more = len(failed) - len(named)
    tail = f"; and {more} more" if more > 0 else ""
    verb = "is" if len(failed) == 1 else "are"
    return (f"Inverted {n_inverted} of {n_total} soundings; {len(failed)} failed "
            f"and {verb} left blank: " + "; ".join(named) + tail + ".")


def _line_chi2_summary(
    chi2_per_sounding, data_counts, *, objective_chi2=None,
) -> Dict[str, float]:
    """Summarise line misfit without confusing gate and sounding weighting.

    Each sounding value is the mean squared uncertainty-normalised residual for
    that sounding.  The line objective therefore weights it by the number of
    retained gates.  Equal-sounding mean and median values are useful QC
    summaries, but they are not substitutes for the objective used by the fit.
    Square-root values are returned alongside because misfit is often quoted on
    a residual/RMS scale, which is the easier one to read against a target of
    one. Any external number on that scale may use a different residual
    definition, log-data space among them, so the two need not agree exactly.
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


#: The refusal for fitting data with ``component="both"``. The forward then
#: returns two fields per frequency, [secondary, total], while an observed
#: sounding holds one - and which one is not something to guess.
FDEM_BOTH_REFUSAL = (
    "FDEM inversion fits one field per frequency; component='both' does not say "
    "which one the data hold. Choose 'secondary' or 'total'.")


def _one_fdem_field(geom: Dict[str, Any]) -> None:
    """Refuse ``component="both"`` wherever a response is compared with data."""
    if str(geom.get("component", "secondary")).strip().lower() == "both":
        raise ValueError(FDEM_BOTH_REFUSAL)


def _fdem_halfspace_amplitude(geom: Dict[str, Any], frequencies: np.ndarray,
                              resistivity: float) -> np.ndarray:
    """``|H|`` of a half-space at ``resistivity``, one value per frequency.

    The solver returns one complex value per frequency and field, frequency
    first. Cutting that to ``frequencies.size``, as this used to, read a "both"
    response - [secondary(f1), total(f1), secondary(f2), ...] - as the band
    itself, mixing the two fields and dropping its upper half. It is reshaped
    as :func:`PyHydroGeophysX.forward.em1d.fdem_forward` does instead.
    """
    _one_fdem_field(geom)
    from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling

    freqs = np.asarray(frequencies, dtype=float).ravel()
    md = FDEMForwardModeling(thicknesses=np.array([50.0]),
                             survey_config=_fdem_config(geom, freqs))
    resp = np.asarray(md.forward(np.array([1.0 / resistivity, 1.0 / resistivity])))
    resp = resp.ravel()
    if not np.iscomplexobj(resp):
        resp = resp[0::2] + 1j * resp[1::2]
    if resp.size != freqs.size:
        raise ValueError(f"FDEM forward returned {resp.size} values for "
                         f"{freqs.size} frequencies.")
    return np.abs(resp)


def _observed_amplitudes(path: str, method: str, moment: str, options: Dict[str, Any],
                         abscissa: np.ndarray) -> Callable[[int], np.ndarray]:
    """Read one station's observed amplitudes on ``abscissa``, by station index.

    What the two calibrations compare with the forward response: the TDEM
    response interpolated onto the calibration gate times, or the FDEM ``|H|``
    per frequency.
    """
    def observed(station: int) -> np.ndarray:
        data = load_sounding(path, method, sounding=int(station), moment=moment, **options)
        if method == "TDEM":
            return _response_on_times(data, abscissa)
        return np.abs(np.asarray(data["real"], float) + 1j * np.asarray(data["imag"], float))

    return observed


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
    options = sounding_options(geom)
    try:
        head = load_sounding(path, method, sounding=0, moment=moment, **options)
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
        else:
            abscissa = np.asarray(head["frequencies"], dtype=float).ravel()
            preds = np.asarray([_fdem_halfspace_amplitude(geom, abscissa, R)
                                for R in np.geomspace(25.0, 3000.0, 20)])
        observed = _observed_amplitudes(path, method, moment, options, abscissa)
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
    a borehole or regional geology), and the scale returned is the one that ties
    the observed amplitudes of a few probe soundings to the forward response of a
    half-space at that resistivity - the geometric mean of ``forward/observed``
    over their gates. That is one forward model and no inversion, so it is
    deterministic; the layered model recovered afterwards is not forced to
    ``ref_resistivity`` exactly, but its absolute level is pinned the same way for
    every dataset. Returns the current ``data_scale`` unchanged if calibration is
    not possible.
    """
    ref = float(ref_resistivity)
    current = float(inv.get("data_scale", 1.0))
    if ref <= 0:
        return current
    moment = str(geom.get("tem_moment", "HM"))
    options = sounding_options(geom)
    try:
        head = load_sounding(path, method, sounding=0, moment=moment, **options)
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
        else:
            abscissa = np.asarray(head["frequencies"], dtype=float).ravel()
            pred = _fdem_halfspace_amplitude(geom, abscissa, ref)
        observed = _observed_amplitudes(path, method, moment, options, abscissa)
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
#: Metres to round a station's measured transmitter-receiver distance to, or
#: zero to model the distance the file records.
#:
#: Zero is the default. Rounding existed to keep the forward-operator cache
#: small, and it is no longer needed for that: a warmed operator is about 50 kB,
#: so even a survey presenting 1,600 distinct ones costs under 80 MB, and the
#: chunked scheduling in
#: :func:`~PyHydroGeophysX.inversion.em1d_lci._map_soundings` keeps each worker
#: to its own share in any case.
#:
#: The rounding was not free. Half a bin at 16.6 m moved the modelled response
#: by 0.4 percent at the median and 1.8 percent at its worst gate, and at 30.5 m
#: by 2.8 percent at its worst. Those are small next to the gate errors, and
#: they are also an offset the instrument did not report.
STATION_DISTANCE_BIN_M = 0.0


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

    It is also where this package and TEMimage part ways. TEMimage 3 (Lupus
    3.0.2) hands its solver the spec's nominal ``RxCoilXYZPos`` for every
    station, as the run's stored ``InversionInput`` shows. Given that same 15 m,
    this forward reproduces TEMimage's stored ``ForwardData`` for TEMimage's own
    models to a median 0.09 (LM) and 0.15 (HM) data errors over 627 stations and
    12,763 gates, the largest systematic gap 2 percent; given each station's own
    distance it differs by a median 2.0 data errors at the low moment, almost
    all of it at the early gates. The measured distance is the right one. On
    that survey an inversion with it fits equally well at every station
    (median chi-squared 0.65 to 0.93 whatever the station's distance), while
    with the nominal 15 m the fit falls apart as the distance departs from it:
    chi-squared 0.77 within half a metre, 5.7 at 2 to 4 m and 16 beyond 4 m,
    and TEMimage's own data fit worsens the same way, 1.3 to 3.9 (correlation
    0.61 with the departure). The distance also accounts for most of the
    difference between the two programs' models: 0.20 decades at the median
    where the station is 2 m or more off nominal, 0.10 where it is within 1 m,
    and 0.10 in both groups once this package is given the nominal distance too.

    ``tx_rx_sep`` is passed through as the file records it. It used to be
    rounded to :data:`STATION_DISTANCE_BIN_M`, which now defaults to zero; set
    ``tx_rx_sep_bin`` to a positive number of metres to round again.

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
        _forward_line, _map_soundings, _misfit, _worker_pool, resolve_worker_count,
    )

    if not blocks:
        return default
    step = max(1, len(blocks) // _STARTING_SAMPLE)
    sampled = list(blocks)[::step][:_STARTING_SAMPLE]
    n_data = int(sum(block.dobs.size for block in sampled))
    if n_data <= 0:
        return default
    # A block that can model a half-space directly does it in one layer.
    direct = all(block.halfspace is not None for block in sampled)
    best_rho, best_chi2 = float(default), float("inf")
    try:
        with _worker_pool(resolve_worker_count(len(sampled), workers)) as pool:
            for rho in _STARTING_HALF_SPACES:
                if direct:
                    predicted = _map_soundings(
                        pool, lambda s: sampled[s].halfspace(1.0 / float(rho)), len(sampled))
                else:
                    x = np.full(len(sampled) * n_layers, math.log10(float(rho)))
                    predicted = _forward_line(sampled, x, n_layers, pool)
                residual, _ = _misfit(sampled, predicted)
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


def _neighboring_starts(raw, positions, lines, bounds, window: int = 2):
    """Local log-median starts, without crossing lines or large spatial gaps.

    ``window`` is how many stations on each side join the median, so the default
    of 2 medians over five. A wider window is what the reference model wants and
    the starting model does not: a start only has to be close, while every
    wiggle left in the reference is written into the model below the depth of
    investigation, where nothing else decides the level.
    """
    window = max(int(window), 0)
    raw = np.asarray(raw, dtype=float)
    result = raw.copy()
    for line in np.unique(lines):
        ids = np.flatnonzero(np.asarray(lines) == line)
        ids = ids[np.argsort(np.asarray(positions)[ids], kind="stable")]
        gaps = np.diff(np.asarray(positions)[ids])
        positive = gaps[gaps > 0]
        limit = 3 * np.median(positive) if positive.size else np.inf
        for group in np.split(ids, np.flatnonzero(gaps > limit) + 1):
            for j, index in enumerate(group):
                local = raw[group[max(0, j-window):j+window+1]]
                valid = local[np.isfinite(local) & (local > bounds[0]) & (local < bounds[1])]
                if valid.size >= 2:
                    result[index] = 10 ** np.median(np.log10(valid))
    return result


def _reference_starts(selected, raw, positions, lines, bounds, inv,
                      log: LogFn = _noop):
    """The half-space per station that the zeroth-order damping pulls toward.

    Kept apart from the starting model on purpose, although the two default to
    the same array. A start only decides where the optimiser begins, so a
    per-station data-driven value costs nothing once the run converges. The
    reference is a prior: below the depth of investigation it is the only thing
    setting the level, so whatever station-to-station scatter it carries is
    written into the section whether the data support it or not.

    ``reference_model_mode`` chooses how much of that scatter to keep.
    ``neighbor`` (the default) reuses the starting model's own local median,
    widened by ``reference_window``; ``line`` collapses each line to one
    half-space; ``global`` collapses the survey to one. A positive
    ``reference_resistivity`` overrides all three, which is the way to pin the
    reference to a level measured from an earlier inversion rather than to
    whatever the half-space scan ranked first.
    """
    selected = np.asarray(selected, dtype=float)
    explicit = float(inv.get("reference_resistivity", 0.0) or 0.0)
    if explicit > 0:
        log(f"Reference model: explicit {explicit:g} ohm-m at every station.")
        return np.full(selected.shape, explicit)

    mode = str(inv.get("reference_model_mode", "neighbor")).lower()
    if mode not in {"neighbor", "line", "global"}:
        raise ValueError("reference_model_mode must be neighbor, line or global.")

    if mode == "neighbor":
        window = max(int(inv.get("reference_window", 2)), 0)
        if window == 2:
            return selected
        widened = _neighboring_starts(raw, positions, lines, bounds, window=window)
        usable = np.isfinite(widened) & (widened > 0)
        log(f"Reference model: local log-median over +/-{window} same-line stations.")
        return np.where(usable, widened, selected)

    usable = np.isfinite(selected) & (selected > 0)
    if not usable.any():
        log("Reference model: no usable starts; falling back to the half-space.")
        return selected

    if mode == "global":
        value = float(10.0 ** np.median(np.log10(selected[usable])))
        log(f"Reference model: one {value:.6g} ohm-m half-space for the survey.")
        return np.full(selected.shape, value)

    reference = selected.copy()
    lines = np.asarray(lines)
    for line in np.unique(lines):
        ids = np.flatnonzero(lines == line)
        good = ids[usable[ids]]
        if good.size:
            reference[ids] = 10.0 ** np.median(np.log10(selected[good]))
    log("Reference model: one half-space per line.")
    return reference


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

    In simultaneous LCI, a positive ``inv["reference_resistivity"]`` sets the
    damping reference independently of ``auto_starting_model`` and
    ``initial_models``, for both TDEM and FDEM.

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
    if method == "FDEM":
        # Refused once, here. Per sounding, the solver's own refusal is caught
        # by the keep-the-line-going handlers, so the line would finish with
        # every station failed and the reason only in the log.
        _one_fdem_field(geom)
    moment = (
        _normalise_temcompany_moment(str(geom.get("tem_moment", "HM")))
        if (is_temcompany_source(path) or is_ttem_source(path))
        else str(geom.get("tem_moment", "HM"))
    )
    options = sounding_options(geom)
    head = load_sounding(path, method, sounding=0, moment=moment, **options)
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
    # Which stations a solve actually fitted, and why each other one was not.
    # A station that failed keeps a starting or interpolated model in the
    # arrays below so the coupled solvers have a node there; without this it
    # was counted as inverted and drawn in the section as if it were data.
    fitted = np.zeros(n_pos, dtype=bool)
    failures: Dict[int, str] = {}
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
    neighbor_start = method.upper() == "TDEM" and bool(inv.get("auto_starting_model", True))

    def prepare(s: int):
        """Read one station, and fit it unless the LCI will supply its model.

        Runs on a worker thread, so it touches nothing shared: the station's
        data, its geometry and its own result go back to the caller, which does
        the ordered bookkeeping. Its own exception travels with it for the same
        reason, since one station failing must not stop the line.
        """
        try:
            data = load_sounding(path, method, sounding=offset + s, moment=moment, **options)
            geom_s = _station_geometry(geom, data)
            if hts is not None and s < hts.size:
                geom_s = _with_sensor_height(geom_s, hts[s])
            if lci_supplies_model or prior_context or neighbor_start:
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
            failures[s] = str(failure)
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
        fitted[s] = True
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

    neighbor_starts = None
    if neighbor_start:
        from PyHydroGeophysX.inversion.em1d import (
            _moment_forward, _moment_halfspace, _moment_jacobian, tdem_moment_blocks,
        )
        from PyHydroGeophysX.inversion.em1d_lci import SoundingBlock, _map_soundings
        raw_starts = np.full(n_pos, np.nan)
        scan_default = float(inv.get("starting_resistivity", 100.))

        def scan(s: int):
            """One station's half-space ranking, on a worker thread.

            The candidates are independent per station, so this is the same
            embarrassingly parallel shape as the block build below. ``workers=1``
            inside: the pool is out here, one station per thread, and a nested
            pool would only oversubscribe the cores.

            Returns its failure rather than raising, so one bad station cannot
            stop the line, and stays quiet so the caller can log in order.
            """
            if datasets[s] is None:
                return s, float("nan"), None
            try:
                blocks = tdem_moment_blocks(datasets[s], geometries[s], inv, thick)
                block = SoundingBlock(
                    forward=_moment_forward(blocks),
                    jacobian=_moment_jacobian(blocks),
                    dobs=np.concatenate([b["observed"] for b in blocks]),
                    uncertainty=np.concatenate([b["uncertainty"] for b in blocks]),
                    position=float(pos_lci[s]), line=int(line_numbers[s]),
                    halfspace=_moment_halfspace(blocks))
                return s, _best_starting_resistivity(
                    [block], n_layers, 1, default=scan_default, log=_noop), None
            except Exception as exc:  # noqa: BLE001 - keep the line going
                return s, float("nan"), exc

        if workers > 1:
            log(f"Ranking starting half-spaces for {n_pos} soundings on {workers} threads")
        with _worker_pool(workers) as scan_pool:
            scanned = _map_soundings(scan_pool, scan, n_pos)
        for s, value, failure in scanned:
            if failure is not None:
                log(f"  sounding {s+1} initial scan unavailable: {failure}")
                continue
            raw_starts[s] = value
        bounds = (float(inv.get("rho_min", 1.)), float(inv.get("rho_max", 1e5)))
        neighbor_starts = _neighboring_starts(raw_starts, pos_lci, line_numbers, bounds)
        if sequential and not use_warm_models:
            valid_starts = np.isfinite(neighbor_starts) & (neighbor_starts > 0)
            surface_models[valid_starts] = neighbor_starts[valid_starts, None]
        log("Automatic starts use local log-medians of up to five same-line stations; large gaps split neighborhoods.")

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
        if neighbor_starts is not None and np.isfinite(neighbor_starts[s]):
            options["neighbor_starting_resistivity"] = float(neighbor_starts[s])
        if use_warm_models:
            # The automatic soft target follows the model that actually starts
            # this station, not a stale project fallback value.
            valid = warm_models[s][np.isfinite(warm_models[s]) & (warm_models[s] > 0.)]
            if valid.size:
                options["_resistive_prior_reference_resistivity"] = float(
                    10. ** np.mean(np.log10(valid)))
        return options

    if prior_context or neighbor_start:
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
                    failures[s] = str(failure)
                    data_count_list[s] = 0
                    continue
                surface_models[s] = fit["resistivity"]
                fitted[s] = True
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
                block = build_sounding_block(
                    datasets[s], geometries[s], station_inv(s), method,
                    position=float(pos_lci[s]), line=int(line_numbers[s]),
                    label=f"sounding {s + 1}")
                # Checked here because the coupled solve cannot: one station
                # with a NaN gate made every residual non-finite and the whole
                # line failed, where it is that one station that cannot be fitted.
                values = np.concatenate([np.ravel(block.dobs), np.ravel(block.uncertainty)])
                bad = int(np.count_nonzero(~np.isfinite(values)))
                if bad or not np.size(block.dobs):
                    raise ValueError(
                        f"{bad} of {values.size} data and error values are not finite numbers"
                        if bad else "no data left to fit")
                return s, block, None
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
                failures[s] = f"excluded from the coupled solve: {failure}"
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
                    fitted[s] = True
                    failures.pop(s, None)
                    chi2_list[s] = float(result.get("chi2", np.nan))
                    data_count_list[s] = int(result.get("n_data", 0))
                    if bool(result.get("outliers", {}).get("enabled", False)):
                        per_sounding_outliers[s] = dict(result["outliers"])
                    if result.get("robust", {}).get("enabled"):
                        per_sounding_robust[s] = result["robust"]
                    log(f"  sounding {s + 1}/{n_pos}: "
                        f"chi2={result.get('chi2', float('nan')):.3f}")
                except Exception as exc:  # noqa: BLE001
                    failures[s] = str(exc)
                    log(f"  sounding {s + 1}/{n_pos} failed: {exc}")
            model[:, 0, :] = surface_models[:, ::-1]
        else:
            log(f"Simultaneous LCI: {len(kept)} soundings, lateral="
                f"{lateral:g}, vertical={float(inv.get('smoothness', 0.3)):g}")
            warm = (surface_models[kept] if use_warm_models else None)
            start_resistivity = float(inv.get("starting_resistivity", 100.0))
            if bool(inv.get("auto_starting_model", True)):
                start_resistivity = _best_starting_resistivity(
                    sounding_blocks, n_layers, workers,
                    default=start_resistivity, log=log)
            if neighbor_starts is not None:
                selected_starts = neighbor_starts[kept]
                valid_starts = np.isfinite(selected_starts) & (selected_starts > 0)
                selected_starts = np.where(valid_starts, selected_starts, start_resistivity)
                if warm is None:
                    warm = np.repeat(selected_starts[:, None], n_layers, axis=1)
            else:
                selected_starts = np.full(len(kept), start_resistivity)
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
            # Resolve the damping prior even without the TDEM half-space scan:
            # manual starts and FDEM must honor an explicit reference too.
            # A warm start affects initialization only, never this prior.
            reference_starts = _reference_starts(
                selected_starts,
                raw_starts[kept] if neighbor_starts is not None else selected_starts,
                pos_lci[kept], line_numbers[kept],
                (float(inv.get("rho_min", 1.)), float(inv.get("rho_max", 1e5))),
                inv, log=log)
            reference_model = np.repeat(reference_starts[:, None], n_layers, axis=1)
            finite = reference_starts[np.isfinite(reference_starts)]
            if not finite.size:
                reference_description = f"{start_resistivity:.6g} ohm-m"
            elif np.allclose(finite, finite[0]):
                reference_description = f"{finite[0]:.6g} ohm-m at every station"
            else:
                reference_description = (
                    f"per station, {finite.min():.6g} to {finite.max():.6g} ohm-m "
                    f"(median {np.median(finite):.6g})")
            log(f"Model damping ratio: {float(inv.get('model_damping', DEFAULT_INVERSION['model_damping'])):g}; "
                f"reference: {reference_description}.")
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
                model_damping=float(inv.get("model_damping", DEFAULT_INVERSION["model_damping"])),
                reference_model=reference_model,
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
            fitted[kept] = True
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
                    # Its neighbours' model, so later passes have a node here;
                    # it has no gates of its own, so it is not reported as data.
                    updated[s] = np.power(10.0, reference_log)
                    failures.setdefault(s, "no usable gates after selection")
                    continue
                try:
                    result = invert(
                        datasets[s], geometries[s], local_inv, log=log)
                    updated[s] = np.asarray(
                        result["resistivity"], dtype=float).ravel()
                    fitted[s] = True
                    failures.pop(s, None)
                    chi2_list[s] = float(result.get("chi2", np.nan))
                    data_count_list[s] = int(result.get("n_data", 0))
                    if bool(result.get("outliers", {}).get("enabled", False)):
                        per_sounding_outliers[s] = dict(result["outliers"])
                    if result.get("robust", {}).get("enabled"):
                        per_sounding_robust[s] = result["robust"]
                except Exception as exc:  # noqa: BLE001
                    if not fitted[s]:
                        # The previous model is only where it started.
                        failures[s] = str(exc)
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

    # A station no solve fitted is blanked rather than left holding the
    # starting or interpolated model it was given as a node, so the section,
    # the saved tables and the misfit summaries carry only what was inverted.
    failed = [s for s in range(n_pos) if not fitted[s]]
    for s in failed:
        failures.setdefault(s, "no model was fitted")
        surface_models[s, :] = np.nan
        model[s, 0, :] = np.nan
        chi2_list[s] = float("nan")
        data_count_list[s] = 0
        doi_blocks.pop(s, None)
    n_inverted = n_pos - len(failed)

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
        _map_soundings,
        cumulated_sensitivity,
        sensitivity_doi,
    )

    depth_ctr = 0.5 * (depth_edges[:-1] + depth_edges[1:])  # surface-ordered
    doi_threshold = float(inv.get("doi_threshold", DOI_SENSITIVITY_THRESHOLD))
    sensitivity = np.full((n_pos, n_layers), np.nan, dtype=float)
    doi = np.full(n_pos, np.nan, dtype=float)
    mu0 = 4e-7 * np.pi

    def reach(s: int):
        """One station's cumulated sensitivity and depth of investigation, or None.

        On the worker pool, and the Jacobian once: this loop ran on one
        thread and computed each station's Jacobian twice, which on a
        627-station TEM2Go survey was about two minutes after the inversion.
        """
        row, block = surface_models[s], doi_blocks.get(s)
        if block is None or not np.all(np.isfinite(row)):
            return None
        cumulated = cumulated_sensitivity(block, row)
        return cumulated, sensitivity_doi(block, row, depth_edges, threshold=doi_threshold,
                                          cumulated=cumulated)

    with _worker_pool(resolve_worker_count(n_pos, int(inv.get("parallel_workers", 0)))) as pool:
        reached = _map_soundings(pool, reach, n_pos)
    for s in range(n_pos):
        row = surface_models[s]
        if not np.all(np.isfinite(row)):
            continue
        if reached[s] is not None:
            sensitivity[s], doi[s] = reached[s]
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
    failed_soundings = [
        {"index": int(offset + s), "station": _station_name(station_ids[s], offset + s),
         "line": int(line_numbers[s]),
         "position_m": float(pos_lci[s]) if np.isfinite(pos_lci[s]) else None,
         "reason": failures[s]}
        for s in failed
    ]
    warnings: List[str] = []
    if failed_soundings:
        warnings.append(_failure_summary(n_inverted, n_pos, failed_soundings))
        log(f"{len(failed_soundings)} of {n_pos} soundings failed and are left "
            "blank in the section.")
    result = {
        "initialization": {
            "neighbor_starting_resistivity": (neighbor_starts.tolist()
                                               if neighbor_starts is not None else None),
            "raw_starting_resistivity": (raw_starts.tolist() if neighbor_start else None),
            "strategy": "same_line_local_log_median" if neighbor_start else "configured",
        },
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
        # ``n_soundings`` is the section's width, failed stations included as
        # blank columns; ``n_inverted`` is how many of them a solve fitted.
        "n_soundings": n_pos,
        "n_inverted": n_inverted, "n_failed": len(failed_soundings),
        "failed_soundings": failed_soundings, "warnings": warnings,
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


__all__ = [
    "FDEM_BOTH_REFUSAL",
    "METHODS",
    "STATION_DISTANCE_BIN_M",
    "backend_status",
    "calibrate_to_reference",
    "estimate_data_scale",
    "invert_line",
]
