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
   band's neighbouring harmonics are correlated through the taper (with
   ``tapers`` above one, every Slepian taper's coefficients are points);
5. bands from different sample rates are merged, the better estimate kept
   where two overlap.

Before step 1 every run can be rid of power-line hum (``powerline``) and of
interference repeating on the receiver's clock (``periodic_noise``).

The defaults are EMTF's (Egbert 1997): 128-point windows overlapping by 32,
decimation by 4, its band set, Huber then redescending weights.

Egbert, G. D. (1997). Robust multiple-station magnetotelluric data processing.
Geophysical Journal International, 130(2), 475-496.
https://doi.org/10.1111/j.1365-246X.1997.tb05663.x

Egbert, G. D. & Livelybrooks, D. W. (1996). Single station magnetotelluric
impedance estimation: Coherence weighting and the regression M-estimate.
Geophysics, 61(4), 964-970. https://doi.org/10.1190/1.1444045 (presorting)

Ernst, T., Sokolova, E. Yu., Varentsov, Iv. M. & Golubev, N. G. (2001).
Comparison of two techniques for magnetotelluric data processing using
synthetic data sets. Acta Geophysica Polonica, 49(2), 213-243 (multi-window
stacking)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .robust import RegressionConfig, robust_regression
from .spectra import (
    Band,
    LevelSpectra,
    SpectraConfig,
    band_data,
    band_harmonics,
    default_bands,
    log_bands,
    n_levels_for,
    run_spectra,
    shared_windows,
    variance_inflation,
)
from .timeseries import (TimeSeriesRun, periodic_interference, remove_periodic, remove_periodic_bursts,
                         remove_powerline, saturated,
                         seconds_between, without_samples)
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
    #: Overlapping bands, this many to a decade (:func:`.spectra.log_bands`),
    #: in place of EMTF's band set: continuous curves, errors correlated
    #: between neighbours.
    bands_per_decade: Optional[int] = None
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
    #: Periods (s) of interference repeating exactly on the receiver's clock,
    #: subtracted from every run before its spectra (:func:`.timeseries.remove_periodic`).
    periodic_noise: Tuple[float, ...] = ()
    #: Cycles each interference waveform is averaged over: fewer follow
    #: interference whose shape drifts, each notch then a little wider.
    periodic_cycles: int = 31
    #: Remove a period from a run only where :func:`.timeseries.periodic_interference`
    #: finds it in that run (when the run holds 30 cycles or more, enough to tell).
    #: The removal also takes the field at every harmonic of the period, and
    #: where a level's windows cannot resolve those notches it biases the band:
    #: so one site's interference is not taken out of another site's data.
    #: Bursts too short to fold are cleaned for every period.
    periodic_where_found: bool = True
    #: Leave out of the bands the harmonics inside those notches, where windows
    #: are long enough to resolve them (twice the period or more): the notch
    #: takes the field there too, and what is left is the interference's drift.
    skip_notched: bool = True
    #: A band whose Zxy and Zyx relative errors average above this is
    #: estimated again from a band about twice as wide (Q 1.5 instead of 3,
    #: same level and centre) and the wider one kept if its errors are smaller:
    #: more data where the field is weakest, at the price of resolution. 0 is off.
    widen_above: float = 0.0
    #: Grid frequency (50 or 60 Hz) whose hum and harmonics are notched out of
    #: every run before its spectra (:func:`.timeseries.remove_powerline`); None is off.
    powerline: Optional[float] = None
    #: Leave out the windows holding a sample at the recorder's full scale
    #: (:func:`.timeseries.saturated`), where the reader knows it.
    drop_saturated: bool = True
    #: Weight points by their leverage too, every iteration (:mod:`.robust`):
    #: noise in the magnetic channels then cannot pull the estimate.
    bounded_influence: bool = False
    #: Errors from the delete-a-group jackknife (20 groups) instead of the
    #: parametric model (:mod:`.robust`).
    jackknife: bool = False
    #: Coherence presorting: in each band, stretches of ``presort_windows``
    #: consecutive windows whose multiple coherence (squared) of the output
    #: with Hx, Hy is below this are left out of that output's regression.
    min_coherence: float = 0.0
    presort_windows: int = 10
    #: Remote reference: where a band's Hx or Hy is coherent with the remote's
    #: same channel below this (squared coherence), the reference holds almost
    #: no natural signal; the elements estimated from that input (Hx: Zxx, Zyx;
    #: Hy: Zxy, Zyy; and the tipper's) are then a ratio of cross-spectra near
    #: zero, and the jackknife understates their error. Their relative errors
    #: are raised to at least 100 %. 0 is off.
    min_reference_coherence: float = 0.0
    #: Slepian tapers per window (multitaper, :mod:`.spectra`); 1 is ``taper`` alone.
    tapers: int = 1
    #: An output (Ex for Zxy, Ey for Zyx, Hz) whose relative error in a band is
    #: above this with one taper is estimated again there with
    #: ``multitaper_tapers`` Slepian tapers, and that estimate kept. Where the
    #: field is weakest the tapers' extra points cut the scatter; where the
    #: errors are a few per cent, the tapers' smoothing over neighbouring
    #: harmonics (a bias of a few per cent where the curve is steep, which the
    #: errors do not show) would be the larger error, so those bands keep one
    #: taper. Where the two estimates differ, which is right is not known: the
    #: difference is added to the errors (in quadrature), so the curve is the
    #: steadier one and its errors hold the other. 0 is off.
    multitaper_above: float = 0.0
    multitaper_tapers: int = 3
    #: The periods (s, lowest and highest) whose bands ``multitaper_above``
    #: may estimate again; None is every band. Tapers help where few windows
    #: are the limit, as in a weak dead band; in short bursts at high rates
    #: they can make the curve rougher.
    multitaper_periods: Optional[Tuple[float, float]] = None
    #: Multi-window stacking (Ernst et al. 2001): each band also takes the
    #: coefficients at its frequencies of the windows one level up or down -
    #: windows ``decimation_factor`` times longer or shorter - in one regression.
    #: Those overlap the band's own data in time, so the errors are the jackknife's.
    multi_window: bool = False
    #: Noise classes (:mod:`.robust`): up to this many classes of noise that
    #: grows with the signal, each normalised before the M-estimate; 0 is off.
    noise_classes: int = 0

    def spectra(self) -> SpectraConfig:
        return SpectraConfig(self.window_length, self.overlap, self.decimation_factor,
                             self.taper, self.prewhiten, self.min_windows, self.tapers)

    def regression(self) -> RegressionConfig:
        return RegressionConfig(self.huber, self.redescend, self.max_iterations,
                                self.redescend_iterations, self.tolerance, self.leverage_cutoff,
                                bounded_influence=self.bounded_influence,
                                jackknife_groups=20 if self.jackknife or self.multi_window else 0,
                                noise_classes=self.noise_classes)


def _as_runs(runs: Union[TimeSeriesRun, Sequence[TimeSeriesRun], None]) -> List[TimeSeriesRun]:
    if runs is None:
        return []
    return [runs] if isinstance(runs, TimeSeriesRun) else list(runs)


def _check(run: TimeSeriesRun, config: ProcessingConfig, label: str) -> None:
    missing = [c for c in _REQUIRED if c not in run]
    if missing:
        raise ValueError(f"{label} run has no {missing}; it has {run.components}")
    if not config.allow_uncalibrated:
        # Hz only gives the tipper: without its calibration the tipper is left
        # out (_has_tipper) and the impedance still estimated.
        bad = [c for c in _REQUIRED if not run[c].is_calibrated]
        if bad:
            raise ValueError(
                f"{label} channels {bad} cannot be brought to mV/km and nT (no calibration or "
                "dipole length); give the calibration to the reader, or allow_uncalibrated=True "
                "for transfer functions in the recorded units")


def _without_periodic(runs: List[TimeSeriesRun], periods: Sequence[float], cycles: int = 31,
                      where_found: bool = False):
    """``runs`` with each period's interference subtracted: folded on itself in a run of three
    periods or more, and with the bursts at the same phase in a shorter one. Also returns how
    many short runs had no such bursts and were kept as recorded; with ``where_found``, how
    many long runs each period was not found in and so left in; and the periods taken out of
    each run (whose notches the bands may then skip)."""
    runs = list(runs)
    untouched = {k for k, run in enumerate(runs) if run.duration < 3 * max(periods)}
    found = {k: periodic_interference(run, periods) for k, run in enumerate(runs)
             if where_found and run.duration >= 3 * min(periods)}
    absent: Dict[float, int] = {}
    removed: List[set] = [set() for _ in runs]

    def tellable(run: TimeSeriesRun, period: float) -> bool:
        cycle = period * run.sample_rate
        return abs(cycle - round(cycle)) < 1e-6 and run.n_samples >= 30 * round(cycle)

    for period in periods:
        period = float(period)
        short = [k for k, run in enumerate(runs) if run.duration < 3 * period]
        for k, run in enumerate(runs):
            if run.duration >= 3 * period:
                if k in found and period not in found[k] and tellable(run, period):
                    absent[period] = absent.get(period, 0) + 1
                    continue
                runs[k] = remove_periodic(run, period, cycles=int(cycles))
                removed[k].add(period)
                untouched.discard(k)
        if short:
            cleaned = remove_periodic_bursts([runs[k] for k in short], period, cycles=int(cycles))
            for k, run in zip(short, cleaned):
                if run is not runs[k]:
                    untouched.discard(k)
                    removed[k].add(period)
                runs[k] = run
    return runs, len(untouched), absent, removed


def _has_tipper(run: TimeSeriesRun, config: ProcessingConfig) -> bool:
    """Whether ``run`` can give a tipper: an Hz that is calibrated, or allowed not to be."""
    return "hz" in run and (config.allow_uncalibrated or run["hz"].is_calibrated)


def _workers(tasks: int) -> int:
    """Threads for ``tasks`` independent pieces of work, at most eight."""
    return max(1, min(tasks, os.cpu_count() or 1, 8))


def _frame_matrix(azimuths: Tuple[float, float], frame: float) -> np.ndarray:
    """``U`` with ``measured = U @ field`` for a pair of channels and a frame azimuth."""
    a = np.radians(np.asarray(azimuths) - frame)
    return np.column_stack([np.cos(a), np.sin(a)])


def _pairs(local: List[TimeSeriesRun], remote: List[TimeSeriesRun], rate: float,
           window_length: int) -> Tuple[List[Tuple[TimeSeriesRun, Optional[TimeSeriesRun]]], int]:
    """Each local run cut to the stretches a remote run at the same rate recorded too, and how many
    local runs had none."""
    if not remote:
        return [(run, None) for run in local], 0
    pairs = []
    unmatched = 0
    for run in local:
        found = len(pairs)
        for other in remote:
            if other.sample_rate != rate:
                continue
            start, end = max(run.start, other.start), min(run.end, other.end)
            if seconds_between(start, end) * rate < window_length:
                continue
            a, b = run.window(start, end), other.window(start, end)
            n = min(a.n_samples, b.n_samples)
            pairs.append((a.slice(0, n), b.slice(0, n)))
        unmatched += len(pairs) == found
    return pairs, unmatched


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
        cfg = replace(cfg, **options)
    local_runs, remote_runs = _as_runs(runs), _as_runs(remote)
    if not local_runs:
        raise ValueError("no runs to process")
    for run in local_runs:
        _check(run, cfg, "site")
    for run in remote_runs:
        missing = [c for c in ("hx", "hy") if c not in run]
        if missing:
            raise ValueError(f"remote run has no {missing}")
    # Clipped samples are found on the runs as recorded - cleaning moves them off the full scale - and
    # set aside after cleaning, so their windows are dropped at every level.
    clipped_local = clipped_remote = None
    if cfg.drop_saturated:
        outputs = list(_REQUIRED) + (["hz"] if cfg.tipper else [])
        clipped_local = [saturated(run, outputs) for run in local_runs]
        clipped_remote = [saturated(run, ["hx", "hy"]) for run in remote_runs]
    if cfg.powerline:
        hum = float(cfg.powerline)
        # Every run on its own, so many runs use the cores (the transforms let go of the interpreter).
        with ThreadPoolExecutor(_workers(len(local_runs) + len(remote_runs))) as pool:
            local_runs = list(pool.map(lambda run: remove_powerline(run, hum), local_runs))
            remote_runs = list(pool.map(lambda run: remove_powerline(run, hum), remote_runs))
        if log is not None:
            log(f"Power-line hum at {hum:g} Hz and its harmonics removed (notched; fitted in runs under a minute)")
    removed_by_rate: Dict[float, set] = {}
    if cfg.periodic_noise:
        periods = [float(p) for p in cfg.periodic_noise]
        bursts = sum(1 for run in local_runs if run.duration < 3 * max(periods))
        local_runs, kept, absent, removed = _without_periodic(local_runs, periods, cfg.periodic_cycles,
                                                              cfg.periodic_where_found)
        remote_runs, _, absent_remote, removed_remote = _without_periodic(remote_runs, periods, cfg.periodic_cycles,
                                                                          cfg.periodic_where_found)
        for run, taken in zip(local_runs + remote_runs, removed + removed_remote):
            removed_by_rate.setdefault(run.sample_rate, set()).update(taken)
        if log is not None:
            left = [f"{p:g} s is not in {n} of the site's long run(s)" for p, n in sorted(absent.items())]
            left += [f"{p:g} s is not in {n} of the remote's long run(s)" for p, n in sorted(absent_remote.items())]
            log(f"Interference repeating every {', '.join(f'{p:g}' for p in periods)} s removed"
                + (f"; {bursts} short burst(s) cleaned with the bursts that start at the same phase"
                   if bursts else "")
                + (f"; {kept} with no such bursts kept as recorded" if kept else "")
                + (f"; {', '.join(left)}, so it is left in them" if left else ""))
    if clipped_local is not None:
        local_runs = [without_samples(run, m) for run, m in zip(local_runs, clipped_local)]
        remote_runs = [without_samples(run, m) for run, m in zip(remote_runs, clipped_remote)]
        hit = [(int(m.sum()), m.size) for m in clipped_local + clipped_remote if m is not None and m.any()]
        if hit and log is not None:
            share = sum(a for a, _ in hit) / max(sum(b for _, b in hit), 1)
            log(f"Saturated (at full scale) in {len(hit)} run(s), {100 * share:.2g} % of their samples: "
                "the windows holding them are left out")
    if cfg.tipper and log is not None and any("hz" in run and not _has_tipper(run, cfg) for run in local_runs):
        log("Hz has no calibration, so no tipper is estimated (the impedance is); give its coil "
            "to the reader, or allow uncalibrated channels")
    estimates = []
    for rate in sorted({run.sample_rate for run in local_runs}, reverse=True):
        group = [run for run in local_runs if run.sample_rate == rate]
        pairs, unmatched = _pairs(group, remote_runs, rate, cfg.window_length)
        if not pairs:
            if log is not None:
                log(f"No run at {rate:g} Hz overlaps a remote run at that rate; that rate is left out")
            continue
        if log is not None:
            log(f"Processing {len(pairs)} run(s) at {rate:g} Hz"
                + (" with remote reference" if remote_runs else "")
                + (f"; {unmatched} run(s) with no remote at the same time left out" if unmatched else ""))
        # The bands skip the notches of the periods taken out of this rate's runs only.
        rate_cfg = replace(cfg, periodic_noise=tuple(sorted(removed_by_rate.get(rate, ())))) if cfg.periodic_noise else cfg
        estimate = _process_rate(pairs, rate, rate_cfg)
        if estimate is not None:
            estimates.append(estimate)
            if log is not None:
                log(f"  {len(estimate['rows'])} bands from {estimate['n_levels']} decimation levels"
                    + (f"; {estimate['tapered_bands']} of them, with errors above {cfg.multitaper_above:.0%},"
                       f" estimated again with {cfg.multitaper_tapers} tapers" if estimate.get("tapered_bands") else ""))
    if not estimates:
        raise ValueError("no band had enough data; the runs are too short for the windows")
    tf = merge_estimates(estimates)
    weak = tf.metadata.get("reference_too_weak") or []
    if log is not None and any(weak):
        counts = {c: sum(c in w for w in weak) for c in ("hx", "hy")}
        log(f"The remote shares too little of the field (coherence under {cfg.min_reference_coherence:g}) in "
            + " and ".join(f"{n} band(s) for {c.upper()}" for c, n in counts.items() if n)
            + ": the elements estimated from it are marked unreliable (errors of 100 % or more)")
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
    # The spectra the noisiest bands are estimated again from (multitaper_above).
    tapered_cfg = (replace(spectra_cfg, tapers=int(cfg.multitaper_tapers))
                   if cfg.multitaper_above > 0 and cfg.tapers <= 1 and cfg.multitaper_tapers > 1 else None)
    longest = max(local.n_samples for local, _ in pairs)
    n_levels = cfg.n_levels or n_levels_for(longest, spectra_cfg)
    if n_levels == 0:
        return None
    if cfg.bands:
        bands = [b for b in cfg.bands if b.level < n_levels]
    elif cfg.bands_per_decade:
        bands = log_bands(rate, n_levels, cfg.window_length, cfg.decimation_factor, int(cfg.bands_per_decade))
    else:
        bands = default_bands(n_levels, cfg.window_length)
    if cfg.periodic_noise and cfg.skip_notched:
        bands = [_without_notched(b, rate, cfg) for b in bands]
    pieces = {id(b): [b] + (_companions(b, n_levels, cfg) if cfg.multi_window else []) for b in bands}
    wider = {id(b): _widened(b, rate, cfg) for b in bands} if cfg.widen_above > 0 else {}
    harmonics = band_harmonics([p for b in bands for p in pieces[id(b)]] + [w for w in wider.values() if w])
    first_local = pairs[0][0]
    use_tipper = cfg.tipper and all(_has_tipper(local, cfg) for local, _ in pairs)
    outputs = ["ex", "ey"] + (["hz"] if use_tipper else [])
    frame = float(first_local["hx"].azimuth) if np.isfinite(first_local["hx"].azimuth) else 0.0

    def spectra_of(pair, scfg, wanted):
        local, other = pair
        found = run_spectra(local, outputs + ["hx", "hy"], wanted, scfg)
        _to_frame(found, local, frame)
        if other is None:
            return found, []
        reference = run_spectra(other, ["hx", "hy"], wanted, scfg)
        # The impedance does not depend on the remote's frame; its coherence
        # with this site's Hx and Hy does, so they are compared in one frame.
        _to_frame(reference, other, frame, pairs=(("hx", "hy"),))
        by_level = {s.level: s for s in reference}
        shared = [shared_windows(s, by_level[s.level]) for s in found if s.level in by_level]
        return [a for a, _ in shared], [b for _, b in shared]

    # Runs, and a band's outputs, are independent, and numpy and scipy let go
    # of the interpreter while they compute, so many runs use the cores.
    # The results are gathered in run order, so they do not depend on timing.
    def all_spectra(scfg, wanted):
        local_levels, remote_levels = [], []
        with ThreadPoolExecutor(_workers(len(pairs))) as pool:
            for found, reference in pool.map(lambda pair: spectra_of(pair, scfg, wanted), pairs):
                local_levels += found
                remote_levels += reference
        return _merged(local_levels), _merged(remote_levels)

    single = all_spectra(spectra_cfg, harmonics)

    regression = cfg.regression()
    rows = []
    n_tapered = 0
    with ThreadPoolExecutor(_workers(len(outputs))) as pool:

        def fit(band_pieces, levels=single, scfg=spectra_cfg, which=outputs):
            data, times, ref = _band_points(levels[0], levels[1], band_pieces, outputs + ["hx", "hy"])
            X = np.column_stack([data["hx"], data["hy"]])
            R = None if ref is None else np.column_stack([ref["hx"], ref["hy"]])
            if cfg.min_coherence > 0 and X.shape[0]:
                keep = _coherent_stretches([data["ex"], data["ey"]], X, times, cfg.min_coherence,
                                           cfg.presort_windows)
                # Too little left coherent is no estimate: the band is left out below.
                if keep.sum() < keep.size:
                    data = {c: v[keep] for c, v in data.items()}
                    X = X[keep]
                    R = None if R is None else R[keep]
                    times = times[keep]
            if X.shape[0] < max(cfg.min_points, 3):
                return None
            fitted = dict(zip(which, pool.map(lambda c, y=data, X=X, R=R, t=times:
                                              robust_regression(y[c], X, R, regression, t), which)))
            inflation = variance_inflation(scfg, band_pieces[0].harmonics().size)
            reference = (_reference_coherence(X, R, [fitted["ex"], fitted["ey"]])
                         if "ex" in fitted and "ey" in fitted else None)
            return X.shape[0], fitted, inflation, reference

        chosen = []
        for band in bands:
            used = pieces[id(band)]
            found = fit(used)
            wide = wider.get(id(band))
            if wide is not None and (found is None or _off_diagonal_error(*found[1:3]) > cfg.widen_above):
                wide_pieces = [wide] + (_companions(wide, n_levels, cfg) if cfg.multi_window else [])
                again = fit(wide_pieces)
                if again is not None and (found is None or
                                          _off_diagonal_error(*again[1:3]) < _off_diagonal_error(*found[1:3])):
                    band, found, used = wide, again, wide_pieces
            chosen.append([band, found, used, {}])
        if tapered_cfg is not None:
            # Output by output - Ex gives Zxy, Ey Zyx - so a noisy Zxy does not
            # take Zyx's resolution too. The tapered spectra are made only at the
            # harmonics those bands use. The tapered estimate is kept whatever its
            # errors: choosing the smaller would always favour the tapers' smoothing.
            def within(band):
                if cfg.multitaper_periods is None:
                    return True
                period = 1.0 / band.center(rate / cfg.decimation_factor ** band.level, cfg.window_length)
                return min(cfg.multitaper_periods) <= period <= max(cfg.multitaper_periods)

            noisy = [[c for c in outputs if found is None or _relative_error(found[1][c], found[2], c)
                      > cfg.multitaper_above] if within(band) else [] for band, found, _, _ in chosen]
            needed = [p for (_, _, used, _), outs in zip(chosen, noisy) if outs for p in used]
            tapered = all_spectra(tapered_cfg, band_harmonics(needed)) if needed else None
            for entry, outs in zip(chosen, noisy):
                again = fit(entry[2], tapered, tapered_cfg, outs) if outs else None
                if again is None:
                    continue
                if entry[1] is None:
                    entry[1] = again
                else:
                    n_points, fits, inflation, reference = entry[1]
                    fits, inflation = dict(fits), {c: inflation for c in outputs}
                    for c in outs:
                        # Which of the two is nearer the truth is not known (the
                        # tapers' smoothing, or one taper's scatter), so the errors
                        # take in their difference (_band_entry).
                        entry[3][c] = again[1][c].coefficients - fits[c].coefficients
                        fits[c], inflation[c] = again[1][c], again[2]
                    entry[1] = (n_points, fits, inflation, reference)
                n_tapered += 1
        for band, found, _, spread in chosen:
            if found is None:
                continue
            level_rate = rate / cfg.decimation_factor ** band.level
            n_points, fits, inflation, reference = found
            rows.append((band, band.center(level_rate, cfg.window_length), n_points, fits, inflation, reference,
                         spread))
    if not rows:
        return None
    return {"rows": rows, "outputs": outputs, "frame": frame, "rate": rate, "n_levels": n_levels,
            "min_reference_coherence": float(cfg.min_reference_coherence), "tapered_bands": n_tapered}


def _reference_coherence(X: np.ndarray, R: Optional[np.ndarray], fits) -> Optional[Dict[str, float]]:
    """Squared coherence of the band's Hx and Hy with the remote's, at the impedance fits' weights.

    The weights (their geometric mean over Ex and Ey) keep a burst the
    regression set aside from deciding it. None without a remote.
    """
    if R is None:
        return None
    w = np.sqrt(fits[0].weights * fits[1].weights)
    out = {}
    for k, name in enumerate(("hx", "hy")):
        cross = np.sum(w * X[:, k] * np.conj(R[:, k]))
        power = np.sum(w * np.abs(X[:, k]) ** 2) * np.sum(w * np.abs(R[:, k]) ** 2)
        out[name] = float(np.abs(cross) ** 2 / power) if power > 0 else 0.0
    return out


def _without_notched(band: Band, rate: float, cfg: ProcessingConfig) -> Band:
    """``band`` less its harmonics inside the notches of the periodic-interference removal.

    A notch is ``1 / (cycles * period)`` wide at each ``k / period``; a level's
    windows resolve it when they last two periods or more. A band left with
    fewer than half its harmonics keeps them all.
    """
    level_rate = rate / cfg.decimation_factor ** band.level
    duration = cfg.window_length / level_rate
    h = np.arange(band.first, band.last + 1)
    f = h * level_rate / cfg.window_length
    skip = np.zeros(h.size, dtype=bool)
    for period in cfg.periodic_noise:
        period = float(period)
        if duration < 2 * period:
            continue
        k = np.round(f * period)
        skip |= (k >= 1) & (np.abs(f - k / period) <= 0.5 / (max(int(cfg.periodic_cycles), 1) * period))
    if not skip.any() or skip.sum() > h.size / 2:
        return band
    return Band(band.level, band.first, band.last, tuple(int(x) for x in h[skip]))


def _widened(band: Band, rate: float, cfg: ProcessingConfig) -> Optional[Band]:
    """``band`` about twice as wide around the same centre (Q 1.5), within the harmonics a level uses.

    None when it cannot widen, or only to one side so far that its centre
    would move by more than 15 % (a band at a level's lowest harmonics).
    """
    scale = cfg.window_length / 128.0
    lowest = max(1, int(round(5 * scale)))
    highest = max(1, int(round(30 * scale))) if band.level == 0 else max(4, cfg.decimation_factor) * lowest
    centre = 0.5 * (band.first + band.last)
    first = max(lowest, int(np.floor(centre / 3.0)) + 1)
    last = min(max(highest, band.last), int(np.ceil(centre * 5.0 / 3.0)) - 1)
    if first >= band.first and last <= band.last:
        return None
    wide = Band(band.level, min(first, band.first), max(last, band.last))
    if cfg.periodic_noise and cfg.skip_notched:
        wide = _without_notched(wide, rate, cfg)
    if abs(np.log(wide.harmonics().mean() / band.harmonics().mean())) > np.log(1.15):
        return None
    return wide


def _inflation(inflation: Union[float, Dict[str, float]], output: str) -> float:
    """A band's variance inflation for one output: one number, or one per output where they differ."""
    return float(inflation[output]) if isinstance(inflation, dict) else float(inflation)


def _relative_error(fit: Any, inflation: Union[float, Dict[str, float]], output: str) -> float:
    """An output's relative error (the jackknife's where there): Zxy's for Ex, Zyx's for Ey, the
    tipper's mean for Hz."""
    columns = {"ex": (1,), "ey": (0,)}.get(output, (0, 1))
    shares = []
    for j in columns:
        var = (fit.jackknife_variance[j] if fit.jackknife_variance is not None
               and np.isfinite(fit.jackknife_variance[j]) else
               fit.residual_variance * np.real(fit.inverse_signal_power[j, j]) * _inflation(inflation, output))
        shares.append(np.sqrt(abs(var) / 2) / max(abs(fit.coefficients[j]), 1e-300))
    return float(np.mean(shares))


def _off_diagonal_error(fits: Dict[str, Any], inflation: Union[float, Dict[str, float]]) -> float:
    """The mean of Zxy's and Zyx's relative errors (the jackknife's where there), for choosing a width."""
    return float(np.mean([_relative_error(fits[c], inflation, c) for c in ("ex", "ey")]))


def _companions(band: Band, n_levels: int, cfg: ProcessingConfig) -> List[Band]:
    """The band's frequencies on the windows one level up and one down (multi-window stacking).

    A harmonic ``h`` of level ``k`` is harmonic ``h * D^(j - k)`` of level
    ``j``; only harmonics within the span the first level's bands use (5 to
    30 of 128) are taken, away from the trend a window's detrending leaves
    and from the decimation filter's edge.
    """
    scale = cfg.window_length / 128.0
    lowest, highest = max(1, int(round(5 * scale))), max(1, int(round(30 * scale)))
    found = []
    for level in (band.level - 1, band.level + 1):
        if not 0 <= level < n_levels:
            continue
        factor = float(cfg.decimation_factor) ** (level - band.level)
        first = max(lowest, int(np.ceil(band.first * factor - 1e-9)))
        last = min(highest, int(np.floor(band.last * factor + 1e-9)))
        if first <= last:
            found.append(Band(level, first, last))
    return found


def _band_points(local_levels, remote_levels, bands: Sequence[Band], components: Sequence[str]):
    """A band's points, the remote's on the same rows, and their window times.

    With companions (multi-window stacking) every piece's points are joined
    and put in time order, so presorting's stretches and the jackknife's
    blocks are spans of time holding every window length.
    """
    if len(bands) == 1:
        data, times = band_data(local_levels, bands[0], components)
        ref = band_data(remote_levels, bands[0], ["hx", "hy"])[0] if remote_levels else None
        return data, times, ref
    parts = [band_data(local_levels, b, components) for b in bands]
    order = np.argsort(np.concatenate([t for _, t in parts]), kind="stable")
    data = {c: np.concatenate([d[c] for d, _ in parts])[order] for c in components}
    times = np.concatenate([t for _, t in parts])[order]
    ref = None
    if remote_levels:
        remote = [band_data(remote_levels, b, ["hx", "hy"])[0] for b in bands]
        ref = {c: np.concatenate([d[c] for d in remote])[order] for c in ("hx", "hy")}
    return data, times, ref


def _coherent_stretches(outputs: Sequence[np.ndarray], X: np.ndarray, times: np.ndarray,
                        minimum: float, windows: int) -> np.ndarray:
    """Which points lie in stretches where every output is coherent with the inputs.

    The band's points are grouped by ``windows`` consecutive windows; in each
    group the squared multiple coherence of an output ``y`` with the inputs is
    the share of its power that least squares on them explains,
    ``b^H A^-1 b / sum |y|^2`` with ``A = X^H X`` and ``b = X^H y``. A group
    is kept when every output's reaches ``minimum`` (coherence presorting,
    after Egbert & Livelybrooks 1996).
    """
    _, window = np.unique(times, return_inverse=True)
    group = window // max(int(windows), 1)
    n = int(group.max()) + 1

    def total(values: np.ndarray) -> np.ndarray:
        real = np.bincount(group, weights=values.real, minlength=n)
        return real + 1j * np.bincount(group, weights=values.imag, minlength=n) if np.iscomplexobj(values) else real

    x1, x2 = X[:, 0], X[:, 1]
    a11, a22 = total(np.abs(x1) ** 2), total(np.abs(x2) ** 2)
    a12 = total(np.conj(x1) * x2)
    det = a11 * a22 - np.abs(a12) ** 2
    keep = np.ones(n, dtype=bool)
    for y in outputs:
        b1, b2 = total(np.conj(x1) * y), total(np.conj(x2) * y)
        explained = (a22 * np.abs(b1) ** 2 + a11 * np.abs(b2) ** 2 - 2 * np.real(np.conj(b1) * a12 * b2))
        with np.errstate(divide="ignore", invalid="ignore"):
            coherence = explained / det / total(np.abs(y) ** 2)
        keep &= np.nan_to_num(coherence, nan=0.0) >= minimum
    return keep[group]


def _merged(levels: List[LevelSpectra]) -> List[LevelSpectra]:
    """Every run's windows of a level as one, in run order.

    A band then takes its harmonics from one array per level instead of
    gathering them from each of hundreds of runs; the points come out in the
    same order. The runs' own arrays are let go one channel at a time, so the
    copy never holds the whole of a level twice.
    """
    groups: Dict[Tuple[int, bytes], List[LevelSpectra]] = {}
    for spectra in levels:
        groups.setdefault((spectra.level, spectra.harmonics.tobytes()), []).append(spectra)
    merged = []
    for group in groups.values():
        if len(group) == 1:
            merged.append(group[0])
            continue
        first = group[0]
        joined = LevelSpectra(first.level, first.sample_rate,
                              np.concatenate([s.window_starts for s in group]), first.harmonics)
        for c in list(first.coefficients):
            joined.coefficients[c] = np.concatenate([s.coefficients.pop(c) for s in group])
        merged.append(joined)
    return merged


def _to_frame(levels, run: TimeSeriesRun, frame: float,
              pairs=(("ex", "ey"), ("hx", "hy"))) -> None:
    """Express the E and H pairs' coefficients in the frame at azimuth ``frame``."""
    for first, second in pairs:
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

    Where two rates cover the same frequencies, the one whose impedance
    errors are smaller there, as a whole, takes the stretch - so the curve
    does not hop between rates band by band. Bands that still fall within
    10% in frequency keep the one with the smaller errors.
    """
    by_rate: Dict[float, List[Dict[str, Any]]] = {}
    for estimate in estimates:
        for band, frequency, n_points, fits, inflation, reference, spread in estimate["rows"]:
            entry = _band_entry(band, frequency, n_points, fits, inflation, estimate, reference, spread)
            by_rate.setdefault(entry["rate"], []).append(entry)
    rates = sorted(by_rate, reverse=True)
    for i, high in enumerate(rates):
        for low in rates[i + 1:]:
            if not by_rate[high] or not by_rate[low]:
                continue
            lo = max(min(e["frequency"] for e in by_rate[high]), min(e["frequency"] for e in by_rate[low]))
            hi = min(max(e["frequency"] for e in by_rate[high]), max(e["frequency"] for e in by_rate[low]))
            if lo >= hi:
                continue
            inside = lambda e: lo <= e["frequency"] <= hi
            scores = {r: [e["quality"] for e in by_rate[r] if inside(e) and np.isfinite(e["quality"])]
                      for r in (high, low)}
            if not scores[high] or not scores[low]:
                continue
            loser = low if np.median(scores[high]) <= np.median(scores[low]) else high
            by_rate[loser] = [e for e in by_rate[loser] if not inside(e)]
    entries = [e for r in rates for e in by_rate[r]]
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
        z_err=stack("z_err"),
        tipper=stack("tipper") if has_tipper else None,
        tipper_err=stack("tipper_err") if has_tipper else None,
        tipper_residual_covariance=stack("tipper_residual") if has_tipper else None,
        rotation=np.array([e["frame"] for e in kept]),
        metadata={"bands": [e["band"] for e in kept], "n_points": [e["n_points"] for e in kept],
                  "coherence": [e["coherence"] for e in kept],
                  "residual_alpha": [e["alpha"] for e in kept],
                  "noise_classes": [e["noise_classes"] for e in kept],
                  "variance_inflation": [e["inflation"] for e in kept],
                  "sample_rates": [e["rate"] for e in kept],
                  "jackknife_errors": [e["jackknifed"] for e in kept],
                  "reference_coherence": [e["reference_coherence"] for e in kept],
                  "reference_too_weak": [e["reference_too_weak"] for e in kept],
                  "tapered": [e["tapered"] for e in kept], "notes": []},
    )
    return tf


def _band_entry(band, frequency, n_points, fits, inflation, estimate, reference=None,
                spread=None) -> Dict[str, Any]:
    """One band's estimate, its errors those of the real and of the imaginary part alike.

    The regression's variances are of the complex coefficient, ``E|dZ|^2``,
    the sum of the two parts' variances; the transfer function's convention
    (Gamble's, the EDI's) is their average, so they are halved here, once.
    The jackknife's errors are used where the band has them, the covariance's
    where it has not - band by band, so one short band does not decide for
    the rest. An output estimated again with tapers (``multitaper_above``)
    has its difference from the one-taper estimate, ``spread``, added to its
    errors in quadrature, so the one-taper estimate lies within them. Elements
    whose input the remote barely shares (``min_reference_coherence``) get
    relative errors of 100 % or more. Both widen the covariance as they do
    the errors.
    """
    spread = spread or {}
    ex, ey = fits["ex"], fits["ey"]
    z = np.array([ex.coefficients, ey.coefficients]) * FIELD_TO_OHM
    S = (ex.inverse_signal_power * _inflation(inflation, "ex")
         + ey.inverse_signal_power * _inflation(inflation, "ey")) / 2 / 2
    var_x, var_y = ex.residual_variance, ey.residual_variance
    if ex.weights.shape == ey.weights.shape:
        weights = np.sqrt(ex.weights * ey.weights)
        cross = np.sum(weights * ex.residuals * np.conj(ey.residuals)) / max(np.sum(weights), 1e-300)
        raw_x = np.sum(weights * np.abs(ex.residuals) ** 2) / max(np.sum(weights), 1e-300)
        raw_y = np.sum(weights * np.abs(ey.residuals) ** 2) / max(np.sum(weights), 1e-300)
        cross *= np.sqrt(var_x * var_y / max(raw_x * raw_y, 1e-300))
    else:
        # One output from one taper's points, the other from several tapers': their
        # residuals are not point for point, and the covariance between them is left out.
        cross = 0.0
    residual = np.array([[var_x, cross], [np.conj(cross), var_y]]) * FIELD_TO_OHM**2
    tipper = tipper_residual = tipper_err = None
    if "hz" in fits:
        hz = fits["hz"]
        tipper = hz.coefficients.reshape(1, 2)
        tipper_residual = np.array([[hz.residual_variance]])
        own = np.sqrt(np.abs(hz.residual_variance * np.real(np.diag(hz.inverse_signal_power))
                             * _inflation(inflation, "hz") / 2))
        tipper_err = own.reshape(1, 2)
        if hz.jackknife_variance is not None and np.all(np.isfinite(hz.jackknife_variance)):
            tipper_err = np.sqrt(hz.jackknife_variance / 2).reshape(1, 2)
        if "hz" in spread:
            wider = np.hypot(tipper_err, np.abs(spread["hz"]).reshape(1, 2))
            tipper_residual = tipper_residual * float(np.max((wider / np.maximum(tipper_err, 1e-300)) ** 2))
            tipper_err = wider
    jack = [fits[c].jackknife_variance for c in ("ex", "ey")]
    jackknifed = all(v is not None and np.all(np.isfinite(v)) for v in jack)
    if jackknifed:
        # The jackknife's errors are the errors; the covariance keeps EMTF's
        # form, its inverse signal power scaled to give them on average, so a
        # file holding either says the same.
        variance = np.array(jack) / 2 * FIELD_TO_OHM**2
        z_err = np.sqrt(variance)
        needed = variance / np.maximum(np.real(np.diag(residual)), 1e-300)[:, None]
        scale = needed.mean(axis=0) / np.maximum(np.real(np.diag(S)), 1e-300)
        S = S * np.sqrt(np.outer(scale, scale))
    elif ex.weights.shape == ey.weights.shape:
        z_err = np.sqrt(np.abs(np.diag(residual))[:, None] * np.abs(np.diag(S))[None, :])
    else:
        # Outputs from different points (one taper's, several tapers'): each its own signal power.
        own = np.array([np.real(np.diag(f.inverse_signal_power)) * _inflation(inflation, c) / 2
                        for c, f in (("ex", ex), ("ey", ey))])
        z_err = np.sqrt(np.abs(np.diag(residual))[:, None] * np.abs(own))
    for i, c in enumerate(("ex", "ey")):
        if c in spread:
            z_err[i] = np.hypot(z_err[i], np.abs(spread[c]) * FIELD_TO_OHM)
    if spread or ex.weights.shape != ey.weights.shape:
        # The covariance says the same: an output's row and column scaled until both its elements' errors reach it.
        implied = np.abs(np.diag(residual))[:, None] * np.maximum(np.abs(np.real(np.diag(S))), 1e-300)[None, :]
        D = np.sqrt(np.maximum(np.max(z_err ** 2 / np.maximum(implied, 1e-300), axis=1), 1.0))
        residual = residual * np.outer(D, D)
    weak = []
    limit = float(estimate.get("min_reference_coherence", 0.0) or 0.0)
    if reference is not None and limit > 0:
        for j, name in enumerate(("hx", "hy")):
            if reference[name] >= limit:
                continue
            weak.append(name)
            z_err[:, j] = np.maximum(z_err[:, j], np.abs(z[:, j]))
            # The covariance says the same: its input-j row and column scaled until both outputs' errors reach it.
            implied = np.abs(np.diag(residual)) * max(abs(np.real(S[j, j])), 1e-300)
            factor = float(np.max(z_err[:, j] ** 2 / np.maximum(implied, 1e-300)))
            if factor > 1:
                D = np.ones(2)
                D[j] = np.sqrt(factor)
                S = S * np.outer(D, D)
            if tipper_err is not None:
                tipper_err[0, j] = max(tipper_err[0, j], abs(tipper[0, j]))
    # How good a band is, by the off-diagonal elements: Zxx and Zyy are small
    # at a 1D or 2D site and their relative errors near one whatever the data.
    off = (np.array([0, 1]), np.array([1, 0]))
    quality = float(np.nanmedian(z_err[off] / np.maximum(np.abs(z[off]), 1e-300)))
    return {"band": (band.level + 1, band.first, band.last), "frequency": frequency,
            "n_points": n_points, "z": z, "S": S, "residual": residual, "tipper": tipper,
            "tipper_residual": tipper_residual, "frame": estimate["frame"], "rate": estimate["rate"],
            "coherence": {c: f.coherence for c, f in fits.items()}, "quality": quality,
            "alpha": {c: f.alpha for c, f in fits.items()},
            "noise_classes": {c: f.noise_classes for c, f in fits.items()},
            "inflation": float(np.mean([_inflation(inflation, c) for c in ("ex", "ey")])),
            "z_err": z_err, "tipper_err": tipper_err, "jackknifed": jackknifed,
            "reference_coherence": reference, "reference_too_weak": weak, "tapered": sorted(spread)}
