"""From time series to transfer functions: the impedance and tipper of a site.

:func:`process_mt` takes a site's runs - any instrument, any number, any
sample rates - and, optionally, a remote reference's, and estimates the
impedance ``E = Z H`` and the tipper ``Hz = T H`` band by band:

1. runs are grouped by sample rate; with a remote reference, each run is cut
   to the stretch the remote site recorded at the same rate;
2. :mod:`.spectra` gives each run's calibrated Fourier coefficients at the
   bands' decimation levels, the remote's on the same windows;
3. the coefficients are expressed in the frame of the Hx azimuth (an
   electric or magnetic pair that is not orthogonal, or turned from the
   other, is resolved into it);
4. for each band, :func:`.robust.robust_regression` estimates each output's
   row from every window's coefficients at the band's harmonics; the error
   covariance is scaled by :func:`.spectra.variance_inflation`, because a
   band's neighbouring harmonics are correlated through the taper;
5. bands from different sample rates are merged, the better estimate kept
   where two overlap.

The defaults are EMTF's (Egbert 1997): 128-point windows overlapping by 32,
decimation by 4, its band set, Huber then redescending weights.

Egbert, G. D. (1997). Robust multiple-station magnetotelluric data processing.
Geophysical Journal International, 130(2), 475-496.
https://doi.org/10.1111/j.1365-246X.1997.tb05663.x
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .robust import RegressionConfig, robust_regression
from .spectra import (
    Band,
    SpectraConfig,
    band_data,
    band_harmonics,
    default_bands,
    n_levels_for,
    run_spectra,
    shared_windows,
    variance_inflation,
)
from .timeseries import TimeSeriesRun, seconds_between
from .transfer_function import FIELD_TO_OHM, TransferFunction

_REQUIRED = ("ex", "ey", "hx", "hy")


@dataclass
class ProcessingConfig:
    """Every choice of :func:`process_mt`; the defaults are EMTF's."""

    window_length: int = 128
    overlap: int = 32
    decimation_factor: int = 4
    n_levels: Optional[int] = None
    min_windows: int = 5
    bands: Optional[List[Band]] = None
    taper: str = "hamming"
    prewhiten: bool = True
    huber: float = 1.5
    redescend: float = 2.8
    max_iterations: int = 10
    redescend_iterations: int = 2
    tolerance: float = 0.005
    leverage_cutoff: Optional[float] = None
    min_points: int = 10
    allow_uncalibrated: bool = False
    tipper: bool = True

    def spectra(self) -> SpectraConfig:
        return SpectraConfig(self.window_length, self.overlap, self.decimation_factor,
                             self.taper, self.prewhiten, self.min_windows)

    def regression(self) -> RegressionConfig:
        return RegressionConfig(self.huber, self.redescend, self.max_iterations,
                                self.redescend_iterations, self.tolerance, self.leverage_cutoff)


def _as_runs(runs: Union[TimeSeriesRun, Sequence[TimeSeriesRun], None]) -> List[TimeSeriesRun]:
    if runs is None:
        return []
    return [runs] if isinstance(runs, TimeSeriesRun) else list(runs)


def _check(run: TimeSeriesRun, config: ProcessingConfig, label: str) -> None:
    missing = [c for c in _REQUIRED if c not in run]
    if missing:
        raise ValueError(f"{label} run has no {missing}; it has {run.components}")
    if not config.allow_uncalibrated:
        bad = [c for c in _REQUIRED + (("hz",) if "hz" in run else ()) if not run[c].is_calibrated]
        if bad:
            raise ValueError(
                f"{label} channels {bad} cannot be brought to mV/km and nT (no calibration or "
                "dipole length); give the calibration to the reader, or allow_uncalibrated=True "
                "for transfer functions in the recorded units")


def _frame_matrix(azimuths: Tuple[float, float], frame: float) -> np.ndarray:
    """``U`` with ``measured = U @ field`` for a pair of channels and a frame azimuth."""
    a = np.radians(np.asarray(azimuths) - frame)
    return np.column_stack([np.cos(a), np.sin(a)])


def _pairs(local: List[TimeSeriesRun], remote: List[TimeSeriesRun], rate: float,
           window_length: int) -> List[Tuple[TimeSeriesRun, Optional[TimeSeriesRun]]]:
    if not remote:
        return [(run, None) for run in local]
    pairs = []
    for run in local:
        for other in remote:
            if other.sample_rate != rate:
                continue
            start, end = max(run.start, other.start), min(run.end, other.end)
            if seconds_between(start, end) * rate < window_length:
                continue
            a, b = run.window(start, end), other.window(start, end)
            n = min(a.n_samples, b.n_samples)
            pairs.append((a.slice(0, n), b.slice(0, n)))
    return pairs


def process_mt(runs: Union[TimeSeriesRun, Sequence[TimeSeriesRun]], *,
               remote: Union[TimeSeriesRun, Sequence[TimeSeriesRun], None] = None,
               config: Optional[ProcessingConfig] = None, station: Optional[str] = None,
               log: Optional[Callable[[str], None]] = None,
               **options: Any) -> TransferFunction:
    """Estimate a site's impedance (and tipper, when it has Hz) from its runs.

    ``remote`` holds the reference site's runs (its Hx and Hy are used).
    ``config`` or keyword ``options`` set :class:`ProcessingConfig` fields;
    ``log`` receives a line per sample rate.
    """
    cfg = config or ProcessingConfig()
    if options:
        cfg = ProcessingConfig(**{**asdict(cfg), **options})
    local_runs, remote_runs = _as_runs(runs), _as_runs(remote)
    if not local_runs:
        raise ValueError("no runs to process")
    for run in local_runs:
        _check(run, cfg, "site")
    for run in remote_runs:
        missing = [c for c in ("hx", "hy") if c not in run]
        if missing:
            raise ValueError(f"remote run has no {missing}")
    estimates = []
    for rate in sorted({run.sample_rate for run in local_runs}, reverse=True):
        group = [run for run in local_runs if run.sample_rate == rate]
        pairs = _pairs(group, remote_runs, rate, cfg.window_length)
        if not pairs:
            continue
        if log is not None:
            log(f"Processing {len(pairs)} run(s) at {rate:g} Hz"
                + (" with remote reference" if remote_runs else ""))
        estimate = _process_rate(pairs, rate, cfg)
        if estimate is not None:
            estimates.append(estimate)
            if log is not None:
                log(f"  {len(estimate['rows'])} bands from {estimate['n_levels']} decimation levels")
    if not estimates:
        raise ValueError("no band had enough data; the runs are too short for the windows")
    tf = merge_estimates(estimates)
    first = local_runs[0]
    tf.station = station or first.station
    tf.latitude, tf.longitude, tf.elevation = first.latitude, first.longitude, first.elevation
    tf.declination = first.declination
    tf.metadata.update({"source_format": "processing", "instrument": first.instrument,
                        "remote_reference": bool(remote_runs),
                        "remote_station": remote_runs[0].station if remote_runs else "",
                        "config": {k: v for k, v in asdict(cfg).items() if k != "bands"}})
    return tf


def _process_rate(pairs, rate: float, cfg: ProcessingConfig) -> Optional[Dict[str, Any]]:
    spectra_cfg = cfg.spectra()
    longest = max(local.n_samples for local, _ in pairs)
    n_levels = cfg.n_levels or n_levels_for(longest, spectra_cfg)
    if n_levels == 0:
        return None
    bands = [b for b in (cfg.bands or default_bands(n_levels, cfg.window_length)) if b.level < n_levels]
    harmonics = band_harmonics(bands)
    first_local = pairs[0][0]
    use_tipper = cfg.tipper and all("hz" in local for local, _ in pairs)
    outputs = ["ex", "ey"] + (["hz"] if use_tipper else [])
    frame = float(first_local["hx"].azimuth) if np.isfinite(first_local["hx"].azimuth) else 0.0

    local_levels, remote_levels = [], []
    for local, other in pairs:
        components = outputs + ["hx", "hy"]
        found = run_spectra(local, components, harmonics, spectra_cfg)
        _to_frame(found, local, frame)
        if other is None:
            local_levels += found
            continue
        reference = run_spectra(other, ["hx", "hy"], harmonics, spectra_cfg)
        by_level = {s.level: s for s in reference}
        for spectra in found:
            if spectra.level in by_level:
                a, b = shared_windows(spectra, by_level[spectra.level])
                local_levels.append(a)
                remote_levels.append(b)

    regression = cfg.regression()
    rows = []
    for band in bands:
        level_rate = rate / cfg.decimation_factor ** band.level
        data, _ = band_data(local_levels, band, outputs + ["hx", "hy"])
        X = np.column_stack([data["hx"], data["hy"]])
        if remote_levels:
            ref, _ = band_data(remote_levels, band, ["hx", "hy"])
            R = np.column_stack([ref["hx"], ref["hy"]])
        else:
            R = None
        if X.shape[0] < max(cfg.min_points, 3):
            continue
        fits = {c: robust_regression(data[c], X, R, regression) for c in outputs}
        inflation = variance_inflation(spectra_cfg, band.last - band.first + 1)
        rows.append((band, band.center(level_rate, cfg.window_length), X.shape[0], fits, inflation))
    if not rows:
        return None
    return {"rows": rows, "outputs": outputs, "frame": frame, "rate": rate, "n_levels": n_levels}


def _to_frame(levels, run: TimeSeriesRun, frame: float) -> None:
    """Express the E and H pairs' coefficients in the frame at azimuth ``frame``."""
    for first, second in (("ex", "ey"), ("hx", "hy")):
        U = _frame_matrix((run[first].azimuth, run[second].azimuth), frame)
        if np.allclose(U, np.eye(2), atol=1e-9):
            continue
        if abs(np.linalg.det(U)) < 0.2:
            raise ValueError(f"{first} and {second} are nearly parallel "
                             f"({run[first].azimuth} and {run[second].azimuth} degrees)")
        inverse = np.linalg.inv(U)
        for spectra in levels:
            a, b = spectra.coefficients[first], spectra.coefficients[second]
            spectra.coefficients[first] = inverse[0, 0] * a + inverse[0, 1] * b
            spectra.coefficients[second] = inverse[1, 0] * a + inverse[1, 1] * b


def merge_estimates(estimates: List[Dict[str, Any]]) -> TransferFunction:
    """One :class:`TransferFunction` from the bands of every sample rate.

    Where bands of two rates fall within 10% in frequency, the one whose
    impedance errors are smaller is kept.
    """
    entries = []
    for estimate in estimates:
        for band, frequency, n_points, fits, inflation in estimate["rows"]:
            entries.append(_band_entry(band, frequency, n_points, fits, inflation, estimate))
    entries.sort(key=lambda e: -e["frequency"])
    kept: List[Dict[str, Any]] = []
    for entry in entries:
        clash = [k for k in kept if abs(np.log(k["frequency"] / entry["frequency"])) < np.log(1.1)
                 and k["rate"] != entry["rate"]]
        if clash:
            if entry["quality"] < clash[0]["quality"]:
                kept[kept.index(clash[0])] = entry
            continue
        kept.append(entry)
    kept.sort(key=lambda e: -e["frequency"])
    n = len(kept)
    stack = lambda key: np.stack([e[key] for e in kept]) if n else None
    has_tipper = all(e["tipper"] is not None for e in kept)
    tf = TransferFunction(
        frequency=np.array([e["frequency"] for e in kept]),
        z=stack("z"), inverse_signal_power=stack("S"), z_residual_covariance=stack("residual"),
        tipper=stack("tipper") if has_tipper else None,
        tipper_residual_covariance=stack("tipper_residual") if has_tipper else None,
        rotation=np.array([e["frame"] for e in kept]),
        metadata={"bands": [e["band"] for e in kept], "n_points": [e["n_points"] for e in kept],
                  "coherence": [e["coherence"] for e in kept],
                  "variance_inflation": [e["inflation"] for e in kept],
                  "sample_rates": [e["rate"] for e in kept], "notes": []},
    )
    return tf


def _band_entry(band, frequency, n_points, fits, inflation, estimate) -> Dict[str, Any]:
    ex, ey = fits["ex"], fits["ey"]
    z = np.array([ex.coefficients, ey.coefficients]) * FIELD_TO_OHM
    S = (ex.inverse_signal_power + ey.inverse_signal_power) / 2 * inflation
    weights = np.sqrt(ex.weights * ey.weights)
    cross = np.sum(weights * ex.residuals * np.conj(ey.residuals)) / max(np.sum(weights), 1e-300)
    var_x, var_y = ex.residual_variance, ey.residual_variance
    raw_x = np.sum(weights * np.abs(ex.residuals) ** 2) / max(np.sum(weights), 1e-300)
    raw_y = np.sum(weights * np.abs(ey.residuals) ** 2) / max(np.sum(weights), 1e-300)
    cross *= np.sqrt(var_x * var_y / max(raw_x * raw_y, 1e-300))
    residual = np.array([[var_x, cross], [np.conj(cross), var_y]]) * FIELD_TO_OHM**2
    tipper = tipper_residual = None
    if "hz" in fits:
        tipper = fits["hz"].coefficients.reshape(1, 2)
        tipper_residual = np.array([[fits["hz"].residual_variance]])
    errors = np.sqrt(np.abs(np.diag(residual))[:, None] * np.abs(np.diag(S))[None, :])
    quality = float(np.nanmedian(errors / np.maximum(np.abs(z), 1e-300)))
    return {"band": (band.level + 1, band.first, band.last), "frequency": frequency,
            "n_points": n_points, "z": z, "S": S, "residual": residual, "tipper": tipper,
            "tipper_residual": tipper_residual, "frame": estimate["frame"], "rate": estimate["rate"],
            "coherence": {c: f.coherence for c, f in fits.items()}, "quality": quality,
            "inflation": inflation}

