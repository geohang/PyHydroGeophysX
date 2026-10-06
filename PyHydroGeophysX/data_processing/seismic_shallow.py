"""Shallow seismic line processing: trace QC, timing, conditioning, CMP stacking, refraction branches.

Written for hammer-source lines a few tens of metres long, recorded at tens of
kHz (a Geometrics Geode with 24 channels at 0.5-1 m spacing, say), where the
target is the first few metres and every millisecond counts. The steps and their
defaults come from processing such a line by hand, and each one guards against a
result that looks like geology and is not:

* **Causal filters only.** A zero-phase band-pass (``sosfiltfilt``) spreads the
  high-amplitude surface wave backwards in time and fakes early arrivals. Every
  filter here is a causal Butterworth (``scipy.signal.sosfilt``). Pure time
  shifts (statics) change no waveform and are used freely.
* **The air wave is the clock.** The hammer's air wave (c ~ 340 m/s) dominates
  everything above ~250 Hz and is the first break at short offsets. Fitting
  ``t = tau_record + |x| / c`` to its onsets gives every record's time-zero error
  (:func:`airwave_statics`), which is then removed as a record static.
* **Trace QC first.** Clipped traces (a run of identical samples: the recorder
  saturated) and traces with energy before any arrival can reach them are left
  out, with the counts reported (:func:`trace_qc`).
* **Mutes shape a shallow stack.** With an air-wave mute and a surface-wave mute
  the live corridor is narrow and moves down with offset, so a stacked "event"
  can be the edge of a mute. :func:`stack_line` therefore always returns the
  NMO-corrected supergather with an offset-binned flatness test
  (:func:`nmo_supergather`) and near/far offset stacks next to the stack. There
  is no migration here.
* **Refraction velocity trades off against intercept.** Over short offset ranges
  different pickers gave 600-1600 m/s on the same data. :func:`refraction_branches`
  fits slopes per offset band with each record's intercept as a fixed effect,
  bootstraps over records, checks reciprocity, reports picker disagreement and
  converts intercept to depth over an explicit bracket of the top-layer velocity,
  which the air wave hides (:func:`depth_bracket`).

Arrays follow :class:`~PyHydroGeophysX.data_processing.seismic.SeismicDataset`:
``traces`` is ``(n_samples, n_traces)``, times are in seconds and lengths in
metres. Positions are along the line (:class:`LineGeometry`).

References
----------
Dix, C. H. (1955). Seismic velocities from surface measurements. Geophysics,
20(1), 68-86. https://doi.org/10.1190/1.1438126 - interval velocities from RMS
picks (:class:`VelocityFunction`).

Maeda, N. (1985). A method for reading and checking phase time in
auto-processing system of seismic wave data. Zisin (Journal of the
Seismological Society of Japan, 2nd ser.), 38(3), 365-379.
https://doi.org/10.4294/zisin1948.38.3_365 - the AIC onset picker used on the
air wave (:func:`airwave_statics`).

Mayne, W. H. (1962). Common reflection point horizontal data stacking
techniques. Geophysics, 27(6), 927-938. https://doi.org/10.1190/1.1439118 - CMP
stacking (:func:`cmp_stack`).

Neidell, N. S., & Taner, M. T. (1971). Semblance and other coherency measures
for multichannel data. Geophysics, 36(3), 482-497.
https://doi.org/10.1190/1.1440186 - semblance (:func:`semblance`).

Steeples, D. W., & Miller, R. D. (1998). Avoiding pitfalls in shallow seismic
reflection surveys. Geophysics, 63(4), 1213-1224.
https://doi.org/10.1190/1.1444422 - why coherent noise and mute edges must be
ruled out before a shallow stacked event is read as a reflection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.data_processing.seismic import SeismicDataset, _aic_minimum

__all__ = [
    "LineGeometry",
    "line_geometry",
    "headers_have_positions",
    "TraceQC",
    "trace_qc",
    "causal_bandpass",
    "causal_highpass",
    "AirwaveStatics",
    "airwave_statics",
    "airwave_record_stacks",
    "apply_statics",
    "suppress_airwave",
    "airwave_mute",
    "surface_wave_mute",
    "balance_traces",
    "VelocityFunction",
    "velocity_from_section",
    "nmo_correct",
    "cmp_bin_edges",
    "CMPStack",
    "cmp_stack",
    "offset_split_stacks",
    "semblance",
    "semblance_peaks",
    "Supergather",
    "nmo_supergather",
    "RefractionPicks",
    "refraction_picks",
    "band_slope",
    "reciprocity_check",
    "intercept_depth",
    "depth_bracket",
    "compare_pickers",
    "RefractionResult",
    "refraction_branches",
    "PreparedLine",
    "prepare_line",
    "StackResult",
    "stack_line",
]

LogFn = Optional[Callable[[str], None]]


def _say(log: LogFn, message: str) -> None:
    if log is not None:
        log(message)


def _rows(traces: np.ndarray) -> np.ndarray:
    """``(n_traces, n_samples)`` float copy of a ``(n_samples, n_traces)`` array."""
    arr = np.asarray(traces, dtype=float)
    if arr.ndim != 2:
        raise ValueError("traces must be a 2-D array of samples by traces.")
    return np.array(arr.T, dtype=float, copy=True)


def _cos_down(n: int) -> np.ndarray:
    """A cosine ramp from 1 to 0 over ``n`` samples."""
    if n <= 0:
        return np.zeros(0)
    return 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n)))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass
class LineGeometry:
    """Each trace's record, channel and source/receiver positions along the line (m)."""

    field_record: np.ndarray
    channel: np.ndarray
    source_x: np.ndarray
    receiver_x: np.ndarray
    source_z: np.ndarray
    receiver_z: np.ndarray
    origin: str = "headers"

    def __post_init__(self) -> None:
        self.field_record = np.asarray(self.field_record, dtype=int).ravel()
        self.channel = np.asarray(self.channel, dtype=int).ravel()
        n = self.field_record.size
        for name in ("source_x", "receiver_x", "source_z", "receiver_z"):
            value = np.asarray(getattr(self, name), dtype=float).ravel()
            if value.size == 1 and n != 1:
                value = np.full(n, float(value[0]))
            if value.size != n:
                raise ValueError(f"{name} has {value.size} values for {n} traces.")
            setattr(self, name, value)
        if self.channel.size != n:
            raise ValueError(f"channel has {self.channel.size} values for {n} traces.")

    def __len__(self) -> int:
        return int(self.field_record.size)

    @property
    def offset(self) -> np.ndarray:
        """Signed offset, receiver minus source (m)."""
        return self.receiver_x - self.source_x

    @property
    def abs_offset(self) -> np.ndarray:
        return np.abs(self.offset)

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.receiver_x + self.source_x)

    @property
    def records(self) -> List[int]:
        return sorted({int(r) for r in self.field_record})

    def side(self) -> np.ndarray:
        """+1 for a receiver ahead of its source along the line, -1 behind, 0 on it."""
        return np.sign(np.round(self.offset, 6)).astype(int)


def headers_have_positions(dataset: SeismicDataset) -> bool:
    """Whether the trace headers place the receivers: two positions differ within a record."""
    by_record: Dict[int, set] = {}
    for header in dataset.headers:
        key = (round(float(header.receiver_x), 6), round(float(header.receiver_y), 6))
        by_record.setdefault(int(header.field_record), set()).add(key)
    return any(len(keys) > 1 for keys in by_record.values())


def line_geometry(dataset: SeismicDataset, *, source_x: Optional[Sequence[float]] = None,
                  receiver_x: Optional[Sequence[float]] = None,
                  origin: Optional[str] = None) -> LineGeometry:
    """Along-line positions of every trace of ``dataset``.

    From the trace headers unless ``source_x``/``receiver_x`` (one value per
    trace) are given. Headers with northings that vary are projected onto the
    straight line through the receivers, distance measured from the first
    receiver, so a line laid out in map coordinates reads as a 2-D profile.
    """
    headers = dataset.headers
    if not headers:
        raise ValueError("The dataset has no trace headers.")
    record = [int(h.field_record) for h in headers]
    channel = [int(h.trace_number) for h in headers]
    sx = np.array([h.source_x for h in headers], dtype=float)
    sy = np.array([h.source_y for h in headers], dtype=float)
    gx = np.array([h.receiver_x for h in headers], dtype=float)
    gy = np.array([h.receiver_y for h in headers], dtype=float)
    sz = np.array([h.source_z for h in headers], dtype=float)
    gz = np.array([h.receiver_z for h in headers], dtype=float)
    how = origin or "headers"
    span = max(float(np.ptp(np.r_[gx, sx])), 1.0)
    if source_x is None and receiver_x is None and float(np.ptp(np.r_[gy, sy])) > 1e-6 * span:
        # Map coordinates: project onto the receivers' principal direction.
        pts = np.column_stack([gx, gy])
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
        u = vt[0]
        along_g = (pts - pts[0]) @ u
        if np.corrcoef(np.arange(along_g.size), along_g)[0, 1] < 0:
            u = -u
            along_g = -along_g
        along_s = (np.column_stack([sx, sy]) - pts[0]) @ u
        sx, gx = along_s, along_g
        how = origin or "headers, projected onto the line through the geophones"
    if source_x is not None:
        sx = np.asarray(source_x, dtype=float)
    if receiver_x is not None:
        gx = np.asarray(receiver_x, dtype=float)
    return LineGeometry(record, channel, sx, gx, sz, gz, origin=how)


# ---------------------------------------------------------------------------
# Trace QC
# ---------------------------------------------------------------------------

def _longest_flat_runs(rows: np.ndarray) -> np.ndarray:
    """Per trace, the longest run of identical consecutive non-zero samples."""
    out = np.ones(rows.shape[0], dtype=int)
    for i, trace in enumerate(rows):
        same = (trace[1:] == trace[:-1]) & (trace[1:] != 0.0)
        if not same.any():
            continue
        padded = np.concatenate(([0], same.astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        out[i] = int((edges[1::2] - edges[::2]).max()) + 1
    return out


@dataclass
class TraceQC:
    """Which traces were left out of the processing, and why.

    ``status`` codes, in priority order when a trace has several faults:
    0 kept, 1 in an excluded record, 2 dead, 3 clipped, 4 energy before the
    first arrival.
    """

    clipped: np.ndarray
    early_energy: np.ndarray
    dead: np.ndarray
    excluded: np.ndarray
    flat_run: np.ndarray
    early_rms: np.ndarray
    noise_floor: float
    settings: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    STATUS_LABELS = ("kept", "excluded record", "dead", "clipped", "early energy")

    @property
    def keep(self) -> np.ndarray:
        return ~(self.clipped | self.early_energy | self.dead | self.excluded)

    @property
    def status(self) -> np.ndarray:
        code = np.zeros(self.clipped.size, dtype=int)
        code[self.early_energy] = 4
        code[self.clipped] = 3
        code[self.dead] = 2
        code[self.excluded] = 1
        return code

    def counts(self) -> Dict[str, int]:
        """Traces kept, and traces removed per reason (each counted once, by priority)."""
        code = self.status
        return {"total": int(code.size), "kept": int(np.sum(code == 0)),
                "excluded record": int(np.sum(code == 1)), "dead": int(np.sum(code == 2)),
                "clipped": int(np.sum(code == 3)), "early energy": int(np.sum(code == 4))}

    def summary(self) -> str:
        c = self.counts()
        parts = [f"{c[k]} {k}" for k in ("clipped", "early energy", "dead", "excluded record") if c[k]]
        removed = "; removed " + ", ".join(parts) if parts else "; none removed"
        return f"Kept {c['kept']} of {c['total']} traces{removed}."


def trace_qc(traces: np.ndarray, dt: float, geometry: LineGeometry, *, clip_run: int = 8,
             early_window: float = 1.5e-3, demean_window: float = 2.0e-3,
             early_min_offset: float = 2.0, noise_min_offset: float = 8.0,
             early_factor: float = 10.0, exclude_records: Iterable[int] = ()) -> TraceQC:
    """Flag the traces that would corrupt timing, stacking or picking.

    * **clipped**: at least ``clip_run`` identical consecutive non-zero samples -
      the recorder saturated (typically the 0-1.5 m offsets of a hammer line).
    * **early energy**: the RMS of the first ``early_window`` (after removing the
      mean of the first ``demean_window``) at ``|offset| >= early_min_offset``
      exceeds ``early_factor`` times the line's noise floor, the median of the
      same quantity at ``|offset| >= noise_min_offset``. Nothing physical reaches
      a 2 m offset within 1.5 ms, so such energy is a trigger or crosstalk fault.
    * **dead**: constant or non-finite.
    * **excluded**: the record is in ``exclude_records``, the user's list.
    """
    rows = _rows(traces)
    n_traces, n_samples = rows.shape
    if len(geometry) != n_traces:
        raise ValueError(f"geometry has {len(geometry)} traces, the data {n_traces}.")
    finite = np.all(np.isfinite(rows), axis=1)
    rows = np.where(np.isfinite(rows), rows, 0.0)
    dead = ~finite | (np.ptp(rows, axis=1) == 0.0)
    flat = _longest_flat_runs(rows)
    clipped = (flat >= int(clip_run)) & ~dead
    n_mean = max(1, min(n_samples, int(round(demean_window / dt))))
    n_early = max(1, min(n_samples, int(round(early_window / dt))))
    centred = rows[:, :max(n_mean, n_early)] - rows[:, :n_mean].mean(axis=1, keepdims=True)
    early_rms = np.sqrt(np.mean(centred[:, :n_early] ** 2, axis=1))
    off = geometry.abs_offset
    notes: List[str] = []
    usable = ~dead & ~clipped
    ref = usable & (off >= noise_min_offset)
    if ref.sum() < 3:
        cut = float(np.percentile(off[usable], 75)) if usable.any() else 0.0
        ref = usable & (off >= cut)
        notes.append(f"Fewer than 3 usable traces at {noise_min_offset:g} m or more; the noise "
                     f"floor was taken at offsets of {cut:.1f} m or more instead.")
    noise_floor = float(np.median(early_rms[ref])) if ref.any() else float("nan")
    early = usable & (off >= early_min_offset) & (early_rms > early_factor * noise_floor)
    excluded = np.isin(geometry.field_record, [int(r) for r in exclude_records])
    return TraceQC(clipped=clipped, early_energy=early, dead=dead, excluded=excluded,
                   flat_run=flat, early_rms=early_rms, noise_floor=noise_floor,
                   settings=dict(clip_run=int(clip_run), early_window=early_window,
                                 early_min_offset=early_min_offset,
                                 noise_min_offset=noise_min_offset, early_factor=early_factor,
                                 exclude_records=sorted({int(r) for r in exclude_records})),
                   notes=notes)


# ---------------------------------------------------------------------------
# Causal filters
# ---------------------------------------------------------------------------

def causal_bandpass(traces: np.ndarray, dt: float, low: Optional[float] = None,
                    high: Optional[float] = None, order: int = 4) -> np.ndarray:
    """Causal Butterworth filter along the time axis (``scipy.signal.sosfilt``).

    A band-pass with both corners, a high-pass with ``high`` None (or at the
    Nyquist frequency), a low-pass with ``low`` None or 0. Applied forward only,
    so no energy moves earlier in time: a zero-phase filter smears the surface
    wave ahead of itself and fakes early arrivals on a shallow line.
    """
    from scipy.signal import butter, sosfilt

    if dt <= 0:
        raise ValueError("dt must be positive.")
    nyquist = 0.5 / float(dt)
    lo = float(low) if low else None
    hi = float(high) if high and float(high) < 0.98 * nyquist else None
    if lo is not None and hi is not None:
        if not 0.0 < lo < hi:
            raise ValueError(f"Band {lo:g}-{hi:g} Hz is not a valid band.")
        sos = butter(int(order), [lo, hi], btype="band", fs=1.0 / dt, output="sos")
    elif lo is not None:
        sos = butter(int(order), lo, btype="high", fs=1.0 / dt, output="sos")
    elif hi is not None:
        sos = butter(int(order), hi, btype="low", fs=1.0 / dt, output="sos")
    else:
        return np.array(traces, dtype=float, copy=True)
    return sosfilt(sos, np.asarray(traces, dtype=float), axis=0)


def causal_highpass(traces: np.ndarray, dt: float, corner: float = 8.0, order: int = 2) -> np.ndarray:
    """Causal Butterworth high-pass: removes DC and drift without precursors."""
    return causal_bandpass(traces, dt, low=corner, high=None, order=order)


# ---------------------------------------------------------------------------
# Air-wave statics
# ---------------------------------------------------------------------------

def _robust_lsq(design: np.ndarray, data: np.ndarray, iterations: int = 5,
                k: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least squares with Huber-like reweighting: ``(solution, residuals, weights)``."""
    weights = np.ones(data.size)
    sol = np.zeros(design.shape[1])
    for _ in range(int(iterations)):
        sol = np.linalg.lstsq(design * weights[:, None], data * weights, rcond=None)[0]
        resid = data - design @ sol
        scale = 1.4826 * float(np.median(np.abs(resid))) + 1e-6
        weights = np.where(np.abs(resid) < k * scale, 1.0, k * scale / np.maximum(np.abs(resid), 1e-30))
    resid = data - design @ sol
    return sol, resid, weights


def _aic_onset_bias(dt: float, band: Tuple[float, float], order: int, window: int,
                    onset: int, draws: int = 8, seed: int = 0) -> float:
    """How late (s) the AIC picker reads an impulsive onset after the causal band-pass.

    A unit spike at a known sample, with filtered white noise 60 dB under the
    filtered peak, is filtered and picked the way the field traces are; the
    mean pick error over ``draws`` noise draws is the bias.
    """
    from scipy.signal import butter, sosfilt

    sos = butter(int(order), list(band), btype="band", fs=1.0 / dt, output="sos")
    spike = np.zeros(window)
    spike[onset] = 1.0
    clean = sosfilt(sos, spike)
    peak = float(np.max(np.abs(clean))) or 1.0
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(int(draws)):
        noise = sosfilt(sos, rng.standard_normal(window + 512))[512:]
        noise *= 1e-3 * peak / (float(np.sqrt(np.mean(noise ** 2))) or 1.0)
        errors.append(_aic_minimum(clean + noise) - onset)
    return float(np.mean(errors)) * float(dt)


def _main_onset(segment: np.ndarray, margin: int, ratio: float = 10.0) -> int:
    """AIC onset of the burst holding the envelope peak, not of weak energy ahead of it.

    From the envelope peak, walk back to where the envelope falls under
    1/``ratio`` of the peak - the quiet ahead of the main arrival - and pick
    with AIC between ``margin`` samples before there and the peak. Energy ahead
    of that quiet stretch is a precursor; energy that never falls under it is
    part of the arrival.
    """
    from scipy.signal import hilbert

    env = np.abs(hilbert(segment))
    peak = int(np.argmax(env))
    j = peak
    while j > 0 and env[j] > env[peak] / float(ratio):
        j -= 1
    a = max(j - int(margin), 0)
    b = min(peak + max(int(margin) // 2, 2), segment.size)
    if b - a < 8:
        return _aic_minimum(segment)
    return a + _aic_minimum(segment[a:b])


def _fit_record_intercepts(rec: np.ndarray, xs: np.ndarray, ts: np.ndarray,
                           records: Sequence[int], fit_velocity: bool, velocity: float
                           ) -> Tuple[Dict[int, float], float, np.ndarray, np.ndarray]:
    """Robust ``t = tau_record + |x| s``: ``(taus, c, residuals_s, weights)``."""
    col = {r: k for k, r in enumerate(records)}
    n_rec = len(records)
    data_ms = ts * 1e3
    if fit_velocity:
        design = np.zeros((ts.size, n_rec + 1))
        design[np.arange(ts.size), [col[int(r)] for r in rec]] = 1.0
        design[:, -1] = xs
        sol, resid, weights = _robust_lsq(design, data_ms)
        slowness = sol[-1] * 1e-3
        taus = sol[:-1] * 1e-3
    else:
        slowness = 1.0 / float(velocity)
        design = np.zeros((ts.size, n_rec))
        design[np.arange(ts.size), [col[int(r)] for r in rec]] = 1.0
        sol, resid, weights = _robust_lsq(design, data_ms - xs * slowness * 1e3)
        taus = sol * 1e-3
    if not slowness > 0:
        raise ValueError("The air-wave fit gave a non-positive slowness; the picks are not on "
                         "the air wave.")
    return {r: float(taus[col[r]]) for r in records}, 1.0 / slowness, resid * 1e-3, weights


@dataclass
class AirwaveStatics:
    """Each record's time-zero error from the air wave, ``t = tau_record + |x| / c``.

    ``statics[record]`` is tau in seconds: positive means the record's arrivals
    are late (the trigger fired early). The correction moves the record by
    ``-tau`` (:meth:`shifts`).
    """

    statics: Dict[int, float]
    velocity: float
    velocity_fitted: bool
    n_picks: Dict[int, int]
    median_records: List[int]
    median_static: float
    onset_bias: float
    rms: float
    picks: Dict[str, np.ndarray]
    settings: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: The line's air-wave pilot (band-passed, unit norm) and its time axis (s),
    #: 0 at the onset every pick was referred to; None without alignment.
    pilot: Optional[np.ndarray] = None
    pilot_time: Optional[np.ndarray] = None

    def shifts(self, geometry: LineGeometry) -> np.ndarray:
        """Per-trace time shift (s, positive = later) that removes the record statics."""
        return np.array([-self.statics.get(int(r), self.median_static)
                         for r in geometry.field_record], dtype=float)

    def table(self) -> List[Tuple[int, int, float, str]]:
        """``(record, picks, static_ms, how)`` rows, one per record."""
        out = []
        for record in sorted(self.statics):
            how = "line median" if record in self.median_records else "fitted"
            out.append((int(record), int(self.n_picks.get(record, 0)),
                        1e3 * float(self.statics[record]), how))
        return out


def airwave_statics(traces: np.ndarray, dt: float, geometry: LineGeometry, *,
                    keep: Optional[np.ndarray] = None, velocity: float = 343.0,
                    fit_velocity: bool = True, band: Tuple[float, float] = (250.0, 1500.0),
                    order: int = 4, min_offset: float = 1.0, max_offset: Optional[float] = None,
                    search: Tuple[float, float] = (-4.0e-3, 4.0e-3), min_picks: int = 5,
                    min_contrast: float = 3.0, align: bool = True,
                    align_search: float = 2.5e-3, min_correlation: float = 0.5,
                    precursor_ratio: float = 10.0,
                    exclude_records: Iterable[int] = ()) -> AirwaveStatics:
    """Record time-zero errors from the air wave.

    1. The traces are band-passed causally to ``band`` (the air wave dominates
       above ~250 Hz) and the onset in a window of ``search`` around
       ``|x| / velocity`` is picked with the AIC picker (Maeda, 1985). A pick
       whose RMS in the millisecond after it is less than ``min_contrast``
       times the RMS before it is dropped.
    2. ``t = tau_record + |x| / c`` is fitted by robust least squares, with one
       c for the line (``fit_velocity``) or c fixed at ``velocity``.
    3. With ``align`` the picks are made consistent: the air-wave windows are
       stacked along that fit into a pilot, the pilot's onset is picked with
       AIC, and every trace is re-timed by its lag against the pilot (envelope
       correlation within ``align_search``, refined on the waveform; a trace
       correlating below ``min_correlation`` is dropped), and 2 is repeated
       (twice). Weak energy ahead of the main arrival otherwise draws the AIC
       split to itself on some records and not others, and those records'
       statics come out a millisecond or more apart. The pilot's onset is that
       of its main arrival: energy ahead of it whose RMS is under
       1/``precursor_ratio`` of the arrival's is passed over (0 picks the first
       AIC split), so time zero does not depend on how often a precursor shows.

    The AIC picker's onset bias after the causal filter, measured on a
    synthetic spike (:func:`_aic_onset_bias`), is subtracted from the picks. A
    record with fewer than ``min_picks`` picks, an excluded record, or one with
    no usable trace gets the median of the fitted records' taus, and the notes
    say so.
    """
    from scipy.signal import hilbert

    rows = _rows(traces)
    n_traces, n_samples = rows.shape
    n_mean = max(1, min(n_samples, int(round(2e-3 / dt))))
    rows -= rows[:, :n_mean].mean(axis=1, keepdims=True)
    filtered = causal_bandpass(rows.T, dt, band[0], band[1], order).T
    off = geometry.abs_offset
    excluded = {int(r) for r in exclude_records}
    usable = np.ones(n_traces, dtype=bool) if keep is None else np.asarray(keep, dtype=bool).copy()
    usable &= off >= float(min_offset)
    if max_offset is not None:
        usable &= off <= float(max_offset)
    usable &= ~np.isin(geometry.field_record, list(excluded))
    window = max(32, int(round((search[1] - search[0]) / dt)))
    bias = _aic_onset_bias(dt, band, order, window,
                           min(max(int(round(-search[0] / dt)), 4), window - 4))
    one_ms = max(4, int(round(1e-3 / dt)))

    # 1. AIC picks in a window around the guessed air-wave time.
    rec_list, idx_list, t_list = [], [], []
    for i in np.flatnonzero(usable):
        t_guess = off[i] / float(velocity)
        a = max(int(round((t_guess + search[0]) / dt)), 0)
        b = min(int(round((t_guess + search[1]) / dt)), n_samples)
        if b - a < 16:
            continue
        segment = filtered[i, a:b]
        k = _aic_minimum(segment)
        before = segment[:k]
        after = filtered[i, a + k:a + k + one_ms]
        rms_before = float(np.sqrt(np.mean(before ** 2))) if before.size >= 4 else 0.0
        rms_after = float(np.sqrt(np.mean(after ** 2))) if after.size else 0.0
        if rms_before > 0 and rms_after < min_contrast * rms_before:
            continue
        rec_list.append(int(geometry.field_record[i]))
        idx_list.append(int(i))
        t_list.append((a + k) * dt - bias)
    rec = np.array(rec_list, dtype=int)
    tidx = np.array(idx_list, dtype=int)
    ts = np.array(t_list, dtype=float)
    all_records = geometry.records

    def fit(rec: np.ndarray, tidx: np.ndarray, ts: np.ndarray):
        counts = {r: int(np.sum(rec == r)) for r in all_records}
        fitted = [r for r in all_records if counts[r] >= int(min_picks) and r not in excluded]
        if not fitted:
            raise ValueError(
                f"No record has {min_picks} or more air-wave picks; check the air-wave velocity "
                f"({velocity:g} m/s), the band ({band[0]:g}-{band[1]:g} Hz) and the QC.")
        in_fit = np.isin(rec, fitted)
        taus, c, resid, weights = _fit_record_intercepts(
            rec[in_fit], off[tidx[in_fit]], ts[in_fit], fitted, fit_velocity, velocity)
        return counts, fitted, in_fit, taus, c, resid, weights

    counts, fitted, in_fit, taus, c, resid, weights = fit(rec, tidx, ts)

    # 3. Re-time every trace against the line's air-wave pilot. The lags are
    # measured against the pilot's centre and fitted; the pilot's onset is
    # added once at the end, so where it falls moves every record alike and
    # changes no static relative to another.
    aligned = False
    pilot_out: Optional[np.ndarray] = None
    pilot_time: Optional[np.ndarray] = None
    if align:
        envelope = np.abs(hilbert(filtered, axis=1))
        # The pilot reaches as far ahead as the first pass searched, so the
        # onset is inside it even where the first pass picked late.
        pre, post = int(round(max(1.5e-3, -search[0]) / dt)), int(round(6.0e-3 / dt))
        lag_max = max(1, int(round(align_search / dt)))
        fine = max(1, int(round(0.5e-3 / dt)))
        state = (rec, tidx, ts, counts, fitted, in_fit, taus, c, resid, weights)
        pilot = None
        for _ in range(2):
            med_tau = float(np.median(list(taus.values())))
            good = set(int(i) for i in tidx[in_fit][weights >= 0.5])
            centre = {int(i): int(round((taus.get(int(geometry.field_record[i]), med_tau)
                                         + off[i] / c) / dt)) for i in np.flatnonzero(usable)}
            stack = []
            for i in good:
                p = centre[i]
                if p - pre >= 0 and p + post <= n_samples:
                    seg = filtered[i, p - pre:p + post]
                    norm = float(np.linalg.norm(seg))
                    if norm > 0:
                        stack.append(seg / norm)
            if len(stack) < 3:
                break
            pilot = np.mean(stack, axis=0)
            pilot_env = np.abs(hilbert(pilot))
            new_rec, new_idx, new_t = [], [], []
            for i, p in centre.items():
                if p - pre - lag_max - fine < 0 or p + post + lag_max + fine > n_samples:
                    continue
                best, best_cc = 0, -np.inf
                for lag in range(-lag_max, lag_max + 1):
                    seg = envelope[i, p - pre + lag:p + post + lag]
                    denom = float(np.linalg.norm(seg) * np.linalg.norm(pilot_env))
                    cc = float(seg @ pilot_env) / denom if denom > 0 else -np.inf
                    if cc > best_cc:
                        best, best_cc = lag, cc
                coarse, best_cc = best, -np.inf
                for lag in range(coarse - fine, coarse + fine + 1):
                    seg = filtered[i, p - pre + lag:p + post + lag]
                    denom = float(np.linalg.norm(seg) * np.linalg.norm(pilot))
                    cc = float(seg @ pilot) / denom if denom > 0 else -np.inf
                    if cc > best_cc:
                        best, best_cc = lag, cc
                if best_cc < min_correlation:
                    continue
                new_rec.append(int(geometry.field_record[i]))
                new_idx.append(int(i))
                new_t.append((p + best) * dt)           # the pilot's centre on this trace
            if not new_rec:
                break
            rec, tidx, ts = np.array(new_rec, dtype=int), np.array(new_idx, dtype=int), np.array(new_t)
            counts, fitted, in_fit, taus, c, resid, weights = fit(rec, tidx, ts)
            aligned = True
        if aligned and pilot is not None:
            k_on = (_main_onset(pilot, max(4, int(round(0.5e-3 / dt))), precursor_ratio)
                    if precursor_ratio and precursor_ratio > 0 else _aic_minimum(pilot))
            shift = (k_on - pre) * dt - bias
            ts = ts + shift
            taus = {r: v + shift for r, v in taus.items()}
            pilot_out = pilot.copy()
            pilot_time = (np.arange(pilot.size) - k_on) * dt
        else:
            rec, tidx, ts, counts, fitted, in_fit, taus, c, resid, weights = state
            aligned = False

    notes: List[str] = []
    if fit_velocity and not 300.0 <= c <= 380.0:
        notes.append(f"The fitted air-wave velocity is {c:.0f} m/s, outside the 300-380 m/s of "
                     "sound in air: the picks may be on a ground arrival; check the band.")
    statics = dict(taus)
    median = float(np.median(list(taus.values())))
    median_records = [r for r in all_records if r not in statics]
    for r in median_records:
        statics[r] = median
    few = [r for r in median_records if r not in excluded]
    if few:
        notes.append(f"Record{'s' if len(few) > 1 else ''} {', '.join(map(str, few))} had fewer "
                     f"than {min_picks} air-wave picks and {'were' if len(few) > 1 else 'was'} given "
                     f"the line median static ({median * 1e3:+.2f} ms).")
    full_resid = np.full(ts.size, np.nan)
    full_w = np.zeros(ts.size)
    full_resid[in_fit] = resid
    full_w[in_fit] = weights
    used = full_w > 0
    rms = float(np.sqrt(np.sum(full_w[used] * full_resid[used] ** 2) / max(np.sum(full_w[used]), 1e-30))) \
        if used.any() else float("nan")
    for r in fitted:
        mine = (rec == r) & used
        if mine.sum() >= 4 and np.mean(full_w[mine] < 0.5) > 0.3:
            notes.append(f"Record {r}: more than 30 % of its air-wave picks were down-weighted; "
                         "its static is uncertain - check that record by eye.")
    return AirwaveStatics(
        statics=statics, velocity=float(c), velocity_fitted=bool(fit_velocity), n_picks=counts,
        median_records=median_records, median_static=median, onset_bias=float(bias), rms=rms,
        picks={"trace": tidx, "record": rec, "offset": off[tidx] if tidx.size else np.zeros(0),
               "time": ts, "residual": full_resid, "weight": full_w},
        settings=dict(velocity=float(velocity), fit_velocity=bool(fit_velocity), band=tuple(band),
                      order=int(order), min_offset=float(min_offset), max_offset=max_offset,
                      search=tuple(search), min_picks=int(min_picks), align=bool(aligned),
                      align_search=float(align_search), min_correlation=float(min_correlation),
                      precursor_ratio=float(precursor_ratio or 0.0)),
        notes=notes, pilot=pilot_out if aligned else None,
        pilot_time=pilot_time if aligned else None)


def airwave_record_stacks(traces: np.ndarray, dt: float, geometry: LineGeometry,
                          statics: AirwaveStatics, *, keep: Optional[np.ndarray] = None,
                          window: Tuple[float, float] = (-4.0e-3, 8.0e-3),
                          min_offset: float = 1.0) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Each record's air wave after its static: ``(records, t, stacks)``.

    Every kept trace at ``min_offset`` or more is band-passed as the statics
    were picked, read along ``statics.statics[record] + |x| / c``, normalised
    and stacked per record; ``t`` (s) is relative to that time, so with good
    statics every record's onset sits at 0. ``stacks`` is
    ``(n_records, n_t)``, each row scaled to a peak of 1.
    """
    band = tuple(statics.settings.get("band", (250.0, 1500.0)))
    order = int(statics.settings.get("order", 4))
    rows = _rows(traces)
    n_mean = max(1, min(rows.shape[1], int(round(2e-3 / dt))))
    rows -= rows[:, :n_mean].mean(axis=1, keepdims=True)
    filtered = causal_bandpass(rows.T, dt, band[0], band[1], order).T
    t_rec = np.arange(rows.shape[1]) * dt
    t = np.arange(window[0], window[1], dt)
    off = geometry.abs_offset
    usable = np.ones(len(geometry), dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    records, out = [], []
    for record in geometry.records:
        idx = np.flatnonzero((geometry.field_record == record) & usable & (off >= min_offset))
        if idx.size == 0:
            continue
        tau = statics.statics.get(record, statics.median_static)
        total = np.zeros(t.size)
        for i in idx:
            seg = np.interp(tau + off[i] / statics.velocity + t, t_rec, filtered[i], left=0.0, right=0.0)
            norm = float(np.linalg.norm(seg))
            if norm > 0:
                total += seg / norm
        peak = float(np.max(np.abs(total)))
        records.append(int(record))
        out.append(total / peak if peak > 0 else total)
    return records, t, np.array(out).reshape(len(records), t.size)


def apply_statics(traces: np.ndarray, dt: float, shifts: Sequence[float]) -> np.ndarray:
    """Shift every trace by ``shifts[i]`` seconds (positive = later) by linear interpolation.

    A pure time shift: samples moved in from outside the record are zero and no
    waveform is filtered beyond the interpolation.
    """
    arr = np.asarray(traces, dtype=float)
    shifts = np.asarray(shifts, dtype=float).ravel()
    if shifts.size != arr.shape[1]:
        raise ValueError(f"{shifts.size} shifts for {arr.shape[1]} traces.")
    t = np.arange(arr.shape[0]) * float(dt)
    out = np.empty_like(arr)
    for i in range(arr.shape[1]):
        out[:, i] = np.interp(t - shifts[i], t, arr[:, i], left=0.0, right=0.0)
    return out


# ---------------------------------------------------------------------------
# Air-wave suppression and mutes
# ---------------------------------------------------------------------------

def suppress_airwave(traces: np.ndarray, dt: float, geometry: LineGeometry, velocity: float, *,
                     keep: Optional[np.ndarray] = None, before: float = 1.0e-3,
                     length: float = 10.0e-3, min_offset: float = 1.5,
                     reference_offset: float = 3.0, min_traces: int = 4
                     ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Subtract each record's air wave, modelled along its moveout ``|x| / velocity``.

    Per record, a window of ``length`` starting ``before`` the air-wave time is
    taken from every kept trace at ``|offset| >= min_offset``; the model is the
    median of the unit-norm windows of the traces at ``reference_offset`` or more
    (all of them when fewer than three are that far), and each trace's window
    loses its least-squares multiple of the model. Expects record statics
    already applied, so the air wave has zero intercept. Returns the traces and
    ``{"records", "traces", "reduction_db"}`` (median RMS reduction in the windows).
    """
    rows = _rows(traces)
    n_traces, n_samples = rows.shape
    off = geometry.abs_offset
    usable = np.ones(n_traces, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    n_win = max(4, int(round(length / dt)))
    n_pre = int(round(before / dt))
    before_rms, after_rms, n_records = [], [], 0
    for record in geometry.records:
        idx = np.flatnonzero((geometry.field_record == record) & usable & (off >= min_offset))
        starts = np.array([int(off[i] / velocity / dt) - n_pre for i in idx], dtype=int)
        fits = (starts >= 0) & (starts + n_win <= n_samples)
        idx, starts = idx[fits], starts[fits]
        if idx.size < int(min_traces):
            continue
        windows = np.array([rows[i, s:s + n_win] for i, s in zip(idx, starts)])
        far = off[idx] >= reference_offset
        ref = windows[far] if far.sum() >= 3 else windows
        unit = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-30)
        model = np.median(unit, axis=0)
        power = float(model @ model)
        if power <= 0:
            continue
        n_records += 1
        for i, s, w in zip(idx, starts, windows):
            amp = float(w @ model) / power
            before_rms.append(float(np.sqrt(np.mean(w ** 2))))
            rows[i, s:s + n_win] = w - amp * model
            after_rms.append(float(np.sqrt(np.mean(rows[i, s:s + n_win] ** 2))))
    reduction = float("nan")
    if after_rms and np.median(after_rms) > 0:
        reduction = 20.0 * math.log10(float(np.median(before_rms)) / float(np.median(after_rms)))
    return rows.T.copy(), {"records": n_records, "traces": len(after_rms), "reduction_db": reduction}


def airwave_mute(traces: np.ndarray, dt: float, geometry: LineGeometry, velocity: float, *,
                 before: float = 0.75e-3, after: float = 5.0e-3, taper: float = 0.5e-3) -> np.ndarray:
    """Zero the window from ``before`` ahead of to ``after`` behind ``|x| / velocity``.

    Cosine tapers of ``taper`` on both sides. Applied after the causal filter so
    it also covers the filter's ringing behind the air wave.
    """
    rows = _rows(traces)
    n_samples = rows.shape[1]
    n_taper = max(1, int(round(taper / dt)))
    ramp = _cos_down(n_taper)
    for i, x in enumerate(geometry.abs_offset):
        ta = x / velocity
        a = int((ta - before) / dt)
        b = int((ta + after) / dt)
        if b <= 0 or a >= n_samples:
            continue
        a0, b0 = max(a, 0), min(b, n_samples)
        w = np.ones(n_samples)
        w[a0:b0] = 0.0
        lo = max(a0 - n_taper, 0)
        if a0 > lo:
            w[lo:a0] = ramp[:a0 - lo]
        hi = min(b0 + n_taper, n_samples)
        if hi > b0:
            w[b0:hi] = ramp[::-1][:hi - b0]
        rows[i] *= w
    return rows.T.copy()


def surface_wave_mute(traces: np.ndarray, dt: float, geometry: LineGeometry,
                      velocity: float = 220.0, *, pad: float = 3.0e-3,
                      taper: float = 1.0e-3) -> np.ndarray:
    """Bottom mute below the surface-wave cone: zero after ``|x| / velocity + pad``.

    A cosine taper of ``taper`` ends at the mute time.
    """
    rows = _rows(traces)
    n_samples = rows.shape[1]
    t = np.arange(n_samples) * dt
    for i, x in enumerate(geometry.abs_offset):
        tm = x / velocity + pad
        w = np.ones(n_samples)
        w[t > tm] = 0.0
        ramp = (t > tm - taper) & (t <= tm)
        w[ramp] = 0.5 * (1.0 + np.cos(np.pi * (t[ramp] - (tm - taper)) / taper))
        rows[i] *= w
    return rows.T.copy()


def balance_traces(traces: np.ndarray, dt: float, start: Sequence[float],
                   end: Sequence[float], *, min_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Divide every trace by its RMS between ``start[i]`` and ``end[i]`` (s).

    A trace with ``min_samples`` or fewer samples in its window has nothing left
    to balance on and is zeroed; returns ``(traces, zeroed)``, ``zeroed`` True
    for each trace that had data and was zeroed.
    """
    rows = _rows(traces)
    t = np.arange(rows.shape[1]) * dt
    start = np.broadcast_to(np.asarray(start, dtype=float), (rows.shape[0],))
    end = np.broadcast_to(np.asarray(end, dtype=float), (rows.shape[0],))
    zeroed = np.zeros(rows.shape[0], dtype=bool)
    for i in range(rows.shape[0]):
        live = (t > start[i]) & (t < end[i])
        rms = float(np.sqrt(np.mean(rows[i, live] ** 2))) if live.sum() > min_samples else 0.0
        if rms > 0:
            rows[i] /= rms
        else:
            zeroed[i] = bool(np.any(rows[i]))
            rows[i] = 0.0
    return rows.T.copy(), zeroed


# ---------------------------------------------------------------------------
# Velocity, NMO and CMP stacking
# ---------------------------------------------------------------------------

def _piecewise_integral(knots: np.ndarray, rates: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Integral from 0 to ``t`` of a function equal to ``rates[k]`` from ``knots[k]`` on."""
    knots = np.asarray(knots, dtype=float)
    rates = np.asarray(rates, dtype=float)
    cum = np.concatenate(([0.0], np.cumsum(rates[:-1] * np.diff(knots))))
    t = np.asarray(t, dtype=float)
    k = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, knots.size - 1)
    return cum[k] + rates[k] * (t - knots[k])


@dataclass
class VelocityFunction:
    """Velocity as a function of zero-offset two-way time.

    ``kind='interval'``: ``velocities[k]`` is the interval velocity from
    ``times[k]`` (the first is 0) to the next time - a layered model.
    ``kind='rms'``: ``(times, velocities)`` are RMS-velocity picks, such as
    semblance peaks; RMS velocity is interpolated linearly between them and held
    constant outside, and interval velocities follow from Dix (1955).
    """

    times: np.ndarray
    velocities: np.ndarray
    kind: str = "interval"
    label: str = ""

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).ravel()
        vel = np.asarray(self.velocities, dtype=float).ravel()
        if times.size != vel.size or times.size == 0:
            raise ValueError("times and velocities must have the same, non-zero length.")
        if np.any(vel <= 0) or not np.all(np.isfinite(vel)):
            raise ValueError("Velocities must be positive.")
        order = np.argsort(times, kind="stable")
        times, vel = times[order], vel[order]
        if self.kind not in ("interval", "rms"):
            raise ValueError("kind must be 'interval' or 'rms'.")
        if self.kind == "interval":
            if times[0] > 0:
                times = np.r_[0.0, times]
                vel = np.r_[vel[0], vel]
            times[0] = 0.0
        self.times, self.velocities = times, vel

    @classmethod
    def layered(cls, v1: float, t_ref: Optional[float] = None, v2: Optional[float] = None,
                t_wt: Optional[float] = None, v3: Optional[float] = None,
                label: str = "") -> "VelocityFunction":
        """V1 above two-way time ``t_ref``, V2 to ``t_wt``, V3 below."""
        times, vel = [0.0], [float(v1)]
        if t_ref and v2:
            times.append(float(t_ref))
            vel.append(float(v2))
            if t_wt and v3 and float(t_wt) > float(t_ref):
                times.append(float(t_wt))
                vel.append(float(v3))
        return cls(np.array(times), np.array(vel), "interval", label)

    def _intervals(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(knots, interval velocities)``, the first knot at 0."""
        if self.kind == "interval":
            return self.times, self.velocities
        t, v = self.times, self.velocities
        knots, vint = [0.0], [float(v[0])]
        for k in range(1, t.size):
            dt_k = t[k] - t[k - 1]
            if dt_k <= 0:
                continue
            sq = (v[k] ** 2 * t[k] - v[k - 1] ** 2 * t[k - 1]) / dt_k
            knots.append(float(t[k - 1]))
            vint.append(float(np.sqrt(sq)) if sq > 0 else vint[-1])
        knots.append(float(t[-1]))
        vint.append(float(v[-1]))
        knots_arr = np.array(knots)
        keep = np.r_[True, np.diff(knots_arr) > 0]
        return knots_arr[keep], np.array(vint)[keep]

    def interval(self, t0: Any) -> np.ndarray:
        knots, vint = self._intervals()
        k = np.clip(np.searchsorted(knots, np.asarray(t0, dtype=float), side="right") - 1,
                    0, knots.size - 1)
        return vint[k]

    def rms(self, t0: Any) -> np.ndarray:
        t0 = np.asarray(t0, dtype=float)
        if self.kind == "rms":
            return np.interp(t0, self.times, self.velocities)
        knots, vint = self._intervals()
        energy = _piecewise_integral(knots, vint ** 2, t0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.sqrt(np.where(t0 > 0, energy / np.where(t0 > 0, t0, 1.0), vint[0] ** 2))
        return out

    def depth(self, t0: Any) -> np.ndarray:
        """Depth (m) of a zero-offset two-way time, integrating interval velocity / 2."""
        knots, vint = self._intervals()
        return _piecewise_integral(knots, 0.5 * vint, np.asarray(t0, dtype=float))

    def describe(self) -> str:
        if self.label:
            return self.label
        if self.kind == "rms":
            picks = ", ".join(f"{t * 1e3:.1f} ms {v:.0f} m/s" for t, v in zip(self.times, self.velocities))
            return f"RMS velocity picks: {picks}"
        parts = [f"{v:.0f} m/s from {t * 1e3:.1f} ms" for t, v in zip(self.times, self.velocities)]
        return "Interval velocities: " + "; ".join(parts)


def velocity_from_section(x: Sequence[float], depth: Sequence[float], velocity: Sequence[float],
                          x_range: Tuple[float, float], *, dz: Optional[float] = None,
                          label: str = "") -> VelocityFunction:
    """A layered stacking velocity from a 2-D velocity section, such as a refraction model.

    The cells whose centres lie within ``x_range`` (the stacked CMPs, say) are
    averaged in depth slices ``dz`` thick, by slowness, since what the NMO
    correction needs is the time a wave spends in each slice; each slice then
    becomes an interval velocity from the two-way time at its top,
    ``t = 2 * sum(dz / v)``. A slice no cell centre falls in takes the velocity
    of the slice above, and below the deepest cell the last velocity
    continues. Cells to leave out - outside the rays' coverage, say - are
    simply not passed. One function for the whole range: a stack takes one
    velocity, so lateral changes average out.
    """
    x = np.asarray(x, dtype=float).ravel()
    z = np.asarray(depth, dtype=float).ravel()
    v = np.asarray(velocity, dtype=float).ravel()
    if not (x.size == z.size == v.size):
        raise ValueError("x, depth and velocity must have one value per cell.")
    lo, hi = sorted(float(a) for a in x_range)
    inside = (x >= lo) & (x <= hi) & np.isfinite(z) & (z >= 0) & np.isfinite(v) & (v > 0)
    if not inside.any():
        raise ValueError(f"The velocity section has no cells between x = {lo:g} and {hi:g} m.")
    z, v = z[inside], v[inside]
    z_max = float(z.max())
    if dz is None:
        dz = max(0.05, z_max / 60.0)
    n = max(1, int(math.ceil(z_max / dz + 1e-9)))
    slices = np.minimum((z / dz).astype(int), n - 1)
    slowness = np.bincount(slices, weights=1.0 / v, minlength=n)
    counts = np.bincount(slices, minlength=n)
    vel = np.full(n, np.nan)
    vel[counts > 0] = counts[counts > 0] / slowness[counts > 0]
    first = int(np.flatnonzero(counts > 0)[0])
    vel[:first] = vel[first]
    for k in range(first + 1, n):
        if not np.isfinite(vel[k]):
            vel[k] = vel[k - 1]
    times = np.r_[0.0, np.cumsum(2.0 * dz / vel)[:-1]]
    return VelocityFunction(times, vel, "interval",
                            label or f"From a velocity section, x = {lo:g}-{hi:g} m")


def _vrms_of(velocity: Any, t0: np.ndarray) -> np.ndarray:
    if isinstance(velocity, VelocityFunction):
        return velocity.rms(t0)
    if callable(velocity):
        return np.asarray(velocity(t0), dtype=float)
    return np.full(t0.shape, float(velocity))


def nmo_correct(traces: np.ndarray, dt: float, offsets: Sequence[float], velocity: Any, *,
                t0: Optional[np.ndarray] = None, tmax: Optional[float] = None,
                stretch: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NMO-correct traces to zero offset: ``t(x) = sqrt(t0^2 + (x / Vrms(t0))^2)``.

    Samples stretched by more than ``stretch`` ((t - t0) / t0) are muted. Returns
    ``(corrected, t0, live)``, ``corrected`` and ``live`` as ``(n_t0, n_traces)``.
    """
    arr = np.asarray(traces, dtype=float)
    n_samples = arr.shape[0]
    if t0 is None:
        n_t0 = n_samples if tmax is None else min(n_samples, int(round(tmax / dt)))
        t0 = np.arange(n_t0) * dt
    t0 = np.asarray(t0, dtype=float)
    v = _vrms_of(velocity, t0)
    x = np.abs(np.asarray(offsets, dtype=float))
    t_in = np.arange(n_samples) * dt
    tx = np.sqrt(t0[None, :] ** 2 + (x[:, None] / v[None, :]) ** 2)
    ok = (tx < t_in[-1]) & ((tx - t0[None, :]) / np.maximum(t0[None, :], 1e-4) <= stretch)
    out = np.zeros((t0.size, x.size))
    for i in range(x.size):
        out[:, i] = np.interp(tx[i], t_in, arr[:, i]) * ok[i]
    return out, t0, ok.T & (out != 0.0)


def cmp_bin_edges(midpoints: Sequence[float], bin_width: float) -> np.ndarray:
    """Bin edges on a grid of ``bin_width`` whose centres are multiples of it."""
    mid = np.asarray(midpoints, dtype=float)
    lo = math.floor(float(mid.min()) / bin_width) * bin_width - bin_width / 2.0
    return np.arange(lo, float(mid.max()) + bin_width, bin_width)


@dataclass
class CMPStack:
    """A CMP stack: ``stack`` and ``fold`` are ``(n_t0, n_bins)``."""

    cmp_x: np.ndarray
    t0: np.ndarray
    stack: np.ndarray
    fold: np.ndarray
    traces_per_bin: np.ndarray
    bin_edges: np.ndarray
    bin_width: float
    offset_range: Tuple[float, float]
    stretch: float
    min_fold: int
    label: str = ""

    @property
    def max_fold(self) -> int:
        return int(self.traces_per_bin.max()) if self.traces_per_bin.size else 0


def _selection(geometry: LineGeometry, keep: Optional[np.ndarray],
               offset_range: Tuple[float, float], cmp_range: Optional[Tuple[float, float]]) -> np.ndarray:
    sel = np.ones(len(geometry), dtype=bool) if keep is None else np.asarray(keep, dtype=bool).copy()
    off = geometry.abs_offset
    sel &= (off >= float(offset_range[0])) & (off <= float(offset_range[1]))
    if cmp_range is not None:
        mid = geometry.midpoint
        sel &= (mid >= float(cmp_range[0])) & (mid <= float(cmp_range[1]))
    return sel


def cmp_stack(traces: np.ndarray, dt: float, geometry: LineGeometry, velocity: Any, *,
              keep: Optional[np.ndarray] = None, bin_width: float = 0.5,
              offset_range: Tuple[float, float] = (0.0, np.inf),
              cmp_range: Optional[Tuple[float, float]] = None, tmax: float = 0.05,
              stretch: float = 0.3, min_fold: int = 3, edges: Optional[np.ndarray] = None,
              label: str = "") -> CMPStack:
    """NMO-correct and stack the selected traces in CMP bins of ``bin_width``.

    Each output sample is the mean of the live (unmuted, non-zero) NMO-corrected
    samples in its bin; a sample with fewer than ``min_fold`` of them is zero.
    """
    sel = _selection(geometry, keep, offset_range, cmp_range)
    if not sel.any():
        raise ValueError("No traces fall in the chosen offset and CMP ranges.")
    mid = geometry.midpoint
    if edges is None:
        edges = cmp_bin_edges(mid[sel], bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.flatnonzero(sel)
    corrected, t0, live = nmo_correct(np.asarray(traces, dtype=float)[:, idx], dt,
                                      geometry.abs_offset[idx], velocity, tmax=tmax, stretch=stretch)
    bins = np.searchsorted(edges, mid[idx]) - 1
    inside = (bins >= 0) & (bins < centres.size)
    total = np.zeros((t0.size, centres.size))
    fold = np.zeros_like(total)
    np.add.at(total.T, bins[inside], corrected[:, inside].T)
    np.add.at(fold.T, bins[inside], live[:, inside].T.astype(float))
    stack = np.where(fold >= int(min_fold), total / np.maximum(fold, 1.0), 0.0)
    per_bin = np.bincount(bins[inside], minlength=centres.size)
    return CMPStack(cmp_x=centres, t0=t0, stack=stack, fold=fold, traces_per_bin=per_bin,
                    bin_edges=np.asarray(edges, dtype=float), bin_width=float(bin_width),
                    offset_range=(float(offset_range[0]), float(offset_range[1])),
                    stretch=float(stretch), min_fold=int(min_fold), label=label)


def offset_split_stacks(traces: np.ndarray, dt: float, geometry: LineGeometry, velocity: Any, *,
                        keep: Optional[np.ndarray] = None,
                        offset_range: Tuple[float, float] = (0.0, np.inf),
                        split_offset: Optional[float] = None, bin_width: float = 0.5,
                        cmp_range: Optional[Tuple[float, float]] = None,
                        **kwargs: Any) -> Dict[str, Any]:
    """Near- and far-offset stacks on the same CMP bins, split at ``split_offset``.

    A reflection appears in both at the same t0; an event made by a mute edge or
    by residual moveout appears in one, or at different times. The split
    defaults to the median offset of the selection.
    """
    sel = _selection(geometry, keep, offset_range, cmp_range)
    if not sel.any():
        raise ValueError("No traces fall in the chosen offset and CMP ranges.")
    off = geometry.abs_offset
    split = float(np.median(off[sel])) if split_offset is None else float(split_offset)
    edges = cmp_bin_edges(geometry.midpoint[sel], bin_width)
    out: Dict[str, Any] = {"split_offset": split}
    for name, rng in (("near", (offset_range[0], split)), ("far", (split, offset_range[1]))):
        try:
            out[name] = cmp_stack(traces, dt, geometry, velocity, keep=keep, bin_width=bin_width,
                                  offset_range=rng, cmp_range=cmp_range, edges=edges,
                                  label=f"{name} offsets {rng[0]:g}-{rng[1]:g} m", **kwargs)
        except ValueError:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Semblance
# ---------------------------------------------------------------------------

def semblance(traces: np.ndarray, dt: float, offsets: Sequence[float], t0: Sequence[float],
              velocities: Sequence[float], *, kind: str = "hyperbolic", window: float = 2.0e-3,
              min_live: int = 6) -> np.ndarray:
    """Semblance (Neidell & Taner, 1971) over ``(velocity, t0)``: ``(n_v, n_t0)``.

    ``kind='hyperbolic'`` scans ``t = sqrt(t0^2 + (x / v)^2)``; ``'linear'``
    scans ``t = t0 + |x| / v`` (surface waves, refractions), for comparison:
    coherent energy that is more linear than hyperbolic is not a reflection.
    Muted (zero) samples do not count: the numerator is the smoothed square of
    the sum, the denominator the smoothed product of the live-trace count and the
    sum of squares, and a point needs ``min_live`` live traces on average over
    the window. ``window`` is a length of t0 (s), whatever the t0 step.
    """
    from scipy.ndimage import uniform_filter1d

    rows = _rows(traces)
    n_samples = rows.shape[1]
    x = np.abs(np.asarray(offsets, dtype=float))
    t0 = np.asarray(t0, dtype=float)
    vels = np.asarray(velocities, dtype=float)
    # The window is smoothed along the t0 axis, so it is counted in t0 steps,
    # not in samples of the traces.
    step = float(np.median(np.diff(t0))) if t0.size > 1 else float(dt)
    w = max(1, int(round(window / step)))
    t_end = (n_samples - 1) * dt
    out = np.zeros((vels.size, t0.size))
    for a, v in enumerate(vels):
        if kind == "hyperbolic":
            times = np.sqrt(t0[None, :] ** 2 + (x[:, None] / v) ** 2)
        elif kind == "linear":
            times = t0[None, :] + x[:, None] / v
        else:
            raise ValueError("kind must be 'hyperbolic' or 'linear'.")
        index = np.clip(np.round(times / dt).astype(int), 0, n_samples - 1)
        picked = np.take_along_axis(rows, index, axis=1)
        valid = (times < t_end) & (picked != 0.0)
        picked = np.where(valid, picked, 0.0)
        count = valid.sum(axis=0).astype(float)
        num = uniform_filter1d(picked.sum(axis=0) ** 2, w)
        den = uniform_filter1d(count * (picked ** 2).sum(axis=0), w)
        # Where the window holds (almost) no energy the ratio is rounding noise.
        live = (uniform_filter1d(count, w) >= min_live) & (den > 1e-6 * max(float(den.max()), 1e-300))
        out[a] = np.where(live, np.minimum(num / (den + 1e-300), 1.0), 0.0)
    return out


def semblance_peaks(panel: np.ndarray, t0: Sequence[float], velocities: Sequence[float], *,
                    count: int = 6, t_separation: int = 12, v_separation: int = 8
                    ) -> List[Tuple[float, float, float]]:
    """The ``count`` strongest separated maxima: ``(t0_s, velocity, semblance)``."""
    t0 = np.asarray(t0, dtype=float)
    vels = np.asarray(velocities, dtype=float)
    seen: List[Tuple[int, int]] = []
    for flat in np.argsort(panel.ravel())[::-1]:
        a, b = np.unravel_index(flat, panel.shape)
        if panel[a, b] <= 0:
            break
        if all(abs(b - bb) > t_separation or abs(a - aa) > v_separation for aa, bb in seen):
            seen.append((int(a), int(b)))
        if len(seen) == count:
            break
    return [(float(t0[b]), float(vels[a]), float(panel[a, b])) for a, b in seen]


# ---------------------------------------------------------------------------
# Supergather and the flatness test
# ---------------------------------------------------------------------------

@dataclass
class Supergather:
    """NMO-corrected traces of a CMP range sorted by offset, and their offset bins.

    ``traces`` is ``(n_t0, n_traces)``; ``binned`` is the mean of the live
    samples per offset bin, ``(n_t0, n_bins)``; ``flatness`` is the zero-moveout
    semblance across the bins at each t0; ``events`` are the strongest events of
    the bin average with the outcome of the flatness test.
    """

    t0: np.ndarray
    traces: np.ndarray
    offsets: np.ndarray
    midpoints: np.ndarray
    binned: np.ndarray
    bin_offsets: np.ndarray
    bin_counts: np.ndarray
    flatness: np.ndarray
    live_bins: np.ndarray
    events: List[Dict[str, Any]]
    cmp_range: Tuple[float, float]
    offset_range: Tuple[float, float]
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def passing(self) -> List[Dict[str, Any]]:
        return [e for e in self.events if e.get("passes")]


def _bin_semblance(binned: np.ndarray, w: int, min_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    from scipy.ndimage import uniform_filter1d

    live = (binned != 0.0)
    count = live.sum(axis=1).astype(float)
    num = uniform_filter1d(binned.sum(axis=1) ** 2, w)
    den = uniform_filter1d(count * (binned ** 2).sum(axis=1), w)
    smooth_count = uniform_filter1d(count, w)
    live = (smooth_count >= min_bins) & (den > 1e-6 * max(float(den.max()), 1e-300))
    return np.where(live, np.minimum(num / (den + 1e-300), 1.0), 0.0), count


def nmo_supergather(traces: np.ndarray, dt: float, geometry: LineGeometry, velocity: Any, *,
                    keep: Optional[np.ndarray] = None,
                    cmp_range: Optional[Tuple[float, float]] = None,
                    offset_range: Tuple[float, float] = (0.0, np.inf), tmax: float = 0.05,
                    stretch: float = 0.3, bin_width: float = 1.0, min_per_bin: int = 2,
                    window: float = 2.0e-3, threshold: float = 0.5,
                    max_residual_moveout: float = 1.0e-3, min_bins: int = 3,
                    min_offset_span: float = 4.0, min_moveout: float = 2.0e-3,
                    mute_edges: Optional[Callable[[np.ndarray], Sequence[np.ndarray]]] = None,
                    max_events: int = 6) -> Supergather:
    """The NMO-corrected supergather of a CMP range, with an offset-binned flatness test.

    The corrected traces are averaged in ``bin_width`` offset bins (a sample
    needs ``min_per_bin`` live traces). For the ``max_events`` strongest peaks of
    the bins' average, the test measures: the zero-moveout semblance across the
    bins (``>= threshold`` to pass), the residual moveout - the lag of each bin
    against the average, fitted as ``a + q x^2`` and evaluated across the offsets
    (``<= max_residual_moveout`` to pass) - the number of bins carrying it
    (``>= min_bins``), the offset span of those bins (``>= min_offset_span``) and
    the uncorrected NMO moveout across that span at the event's velocity
    (``>= min_moveout``), and, given ``mute_edges`` (``|x| -> [edge times]`` in
    record time), whether it sits on an NMO-corrected mute edge in at least half
    of the bins, which fails it. The span and moveout checks matter on short
    shallow lines: when the mutes leave an event live in only a few far-offset
    bins, any energy there looks flat after NMO, because a reflection, a wrong
    velocity and a linear event differ by less than the residual tolerance over
    so short a span. An event that passes is a reflection candidate, not a
    reflection.
    """
    from scipy.signal import find_peaks, hilbert

    if cmp_range is None:
        mid_all = geometry.midpoint[_selection(geometry, keep, offset_range, None)]
        if mid_all.size == 0:
            raise ValueError("No traces fall in the chosen offset range.")
        lo, hi = float(mid_all.min()), float(mid_all.max())
        cmp_range = (lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
    sel = _selection(geometry, keep, offset_range, cmp_range)
    if sel.sum() < 2:
        raise ValueError("Fewer than two traces fall in the supergather's CMP and offset ranges.")
    idx = np.flatnonzero(sel)
    off = geometry.abs_offset[idx]
    order = np.argsort(off + 1e-3 * geometry.midpoint[idx], kind="stable")
    idx, off = idx[order], off[order]
    corrected, t0, live = nmo_correct(np.asarray(traces, dtype=float)[:, idx], dt, off, velocity,
                                      tmax=tmax, stretch=stretch)
    bin_index = np.round(off / bin_width).astype(int)
    unique = np.unique(bin_index)
    binned = np.zeros((t0.size, unique.size))
    counts = np.zeros(unique.size, dtype=int)
    for k, b in enumerate(unique):
        members = bin_index == b
        counts[k] = int(members.sum())
        n_live = live[:, members].sum(axis=1)
        total = corrected[:, members].sum(axis=1)
        binned[:, k] = np.where(n_live >= min_per_bin, total / np.maximum(n_live, 1), 0.0)
    bin_offsets = unique * bin_width
    w = max(1, int(round(window / dt)))
    flatness, live_bins = _bin_semblance(binned, w, min_bins)
    average = np.where(live_bins > 0, binned.sum(axis=1) / np.maximum(live_bins, 1), 0.0)
    envelope = np.abs(hilbert(average)) if np.any(average) else np.zeros_like(average)
    events: List[Dict[str, Any]] = []
    if envelope.max() > 0:
        peaks, props = find_peaks(envelope, distance=2 * w, height=0.15 * envelope.max())
        strongest = peaks[np.argsort(props["peak_heights"])[::-1][:int(max_events)]]
        for p in sorted(strongest):
            events.append(_flatness_event(int(p), t0, binned, average, bin_offsets, flatness,
                                          w, dt, velocity, threshold, max_residual_moveout,
                                          min_bins, mute_edges, min_offset_span, min_moveout))
    return Supergather(t0=t0, traces=corrected, offsets=off, midpoints=geometry.midpoint[idx],
                       binned=binned, bin_offsets=bin_offsets, bin_counts=counts,
                       flatness=flatness, live_bins=live_bins, events=events,
                       cmp_range=(float(cmp_range[0]), float(cmp_range[1])),
                       offset_range=(float(offset_range[0]), float(offset_range[1])),
                       settings=dict(bin_width=bin_width, window=window, threshold=threshold,
                                     max_residual_moveout=max_residual_moveout,
                                     min_bins=min_bins, min_offset_span=min_offset_span,
                                     min_moveout=min_moveout, stretch=stretch))


def _flatness_event(p: int, t0: np.ndarray, binned: np.ndarray, average: np.ndarray,
                    bin_offsets: np.ndarray, flatness: np.ndarray, w: int, dt: float,
                    velocity: Any, threshold: float, max_rmo: float, min_bins: int,
                    mute_edges: Optional[Callable[[np.ndarray], Sequence[np.ndarray]]],
                    min_span: float = 0.0, min_moveout: float = 0.0
                    ) -> Dict[str, Any]:
    n = t0.size
    a, b = max(p - w, 0), min(p + w + 1, n)
    ref = average[a:b]
    lags, xs = [], []
    for k in range(binned.shape[1]):
        trace = binned[:, k]
        if np.count_nonzero(trace[a:b]) < (b - a) // 2:
            continue
        best, best_cc = 0, -np.inf
        for lag in range(-w, w + 1):
            lo, hi = a + lag, b + lag
            if lo < 0 or hi > n:
                continue
            seg = trace[lo:hi]
            denom = float(np.linalg.norm(seg) * np.linalg.norm(ref))
            cc = float(seg @ ref) / denom if denom > 0 else -np.inf
            if cc > best_cc:
                best, best_cc = lag, cc
        if best_cc > 0.3:
            lags.append(best * dt)
            xs.append(float(bin_offsets[k]))
    xs_arr, lags_arr = np.array(xs), np.array(lags)
    rmo = float("nan")
    if xs_arr.size >= 3 and np.ptp(xs_arr) > 0:
        design = np.column_stack([np.ones(xs_arr.size), xs_arr ** 2])
        q = np.linalg.lstsq(design, lags_arr, rcond=None)[0][1]
        rmo = float(q * (xs_arr.max() ** 2 - xs_arr.min() ** 2))
    on_edge = False
    if mute_edges is not None and xs_arr.size:
        v = float(_vrms_of(velocity, np.array([t0[p]]))[0])
        hits = 0
        for x in xs_arr:
            for edge in mute_edges(np.array([x])):
                te = float(np.asarray(edge).ravel()[0])
                t0_edge = math.sqrt(max(te ** 2 - (x / v) ** 2, 0.0))
                if abs(t0_edge - t0[p]) <= w * dt:
                    hits += 1
                    break
        on_edge = hits >= max(1, xs_arr.size / 2.0)
    semb = float(flatness[p])
    reasons = []
    if semb < threshold:
        reasons.append(f"semblance across offset bins {semb:.2f} < {threshold:g}")
    if xs_arr.size < min_bins:
        reasons.append(f"carried by {xs_arr.size} offset bins (< {min_bins})")
    span = float(np.ptp(xs_arr)) if xs_arr.size else 0.0
    moveout = 0.0
    if xs_arr.size:
        v_event = float(_vrms_of(velocity, np.array([t0[p]]))[0])
        moveout = (math.hypot(t0[p], xs_arr.max() / v_event)
                   - math.hypot(t0[p], xs_arr.min() / v_event))
    if span < min_span:
        reasons.append(f"offset span {span:.1f} m (< {min_span:g} m)")
    if moveout < min_moveout:
        reasons.append(f"NMO moveout across the carried offsets {moveout * 1e3:.1f} ms "
                       f"(< {min_moveout * 1e3:g} ms), too little to tell a reflection from other energy")
    if not np.isfinite(rmo) or abs(rmo) > max_rmo:
        reasons.append("residual moveout " + (f"{rmo * 1e3:+.2f} ms" if np.isfinite(rmo) else "not measurable"))
    if on_edge:
        reasons.append("sits on an NMO-corrected mute edge")
    return {"t0": float(t0[p]), "semblance": semb, "residual_moveout": rmo,
            "bins": int(xs_arr.size), "offset_span": span, "nmo_moveout": float(moveout),
            "mute_edge": bool(on_edge), "passes": not reasons,
            "reason": "; ".join(reasons) if reasons else "flat across offsets"}


# ---------------------------------------------------------------------------
# Refraction branches
# ---------------------------------------------------------------------------

@dataclass
class RefractionPicks:
    """First-arrival picks of two pickers on the same traces (times in s, NaN = none)."""

    trace: np.ndarray
    record: np.ndarray
    source_x: np.ndarray
    receiver_x: np.ndarray
    offset: np.ndarray
    onset: np.ndarray
    phase: np.ndarray
    settings: Dict[str, Any] = field(default_factory=dict)

    PICKERS = {"onset": "onset", "phase": "first strong phase"}

    def __len__(self) -> int:
        return int(self.trace.size)

    def column(self, which: str) -> np.ndarray:
        if which not in self.PICKERS:
            raise ValueError("which must be 'onset' or 'phase'.")
        return getattr(self, which)


def refraction_picks(traces: np.ndarray, dt: float, geometry: LineGeometry, *,
                     keep: Optional[np.ndarray] = None, airwave_velocity: float = 343.0,
                     guide_intercept: float = 0.010, guide_velocity: float = 700.0,
                     offset_range: Tuple[float, float] = (5.0, 18.0),
                     half_window: float = 7.0e-3, air_clearance: float = 1.5e-3,
                     min_window: float = 4.0e-3, noise_window: float = 6.0e-3,
                     onset_factor: float = 5.0, sustain: float = 1.0e-3,
                     walk_back: float = 1.5, phase_fraction: float = 0.25,
                     smoothing: float = 0.5e-3) -> RefractionPicks:
    """Pick the refraction on each trace with two pickers, clear of the air wave.

    The window is ``half_window`` either side of the guide ``t = guide_intercept
    + |x| / guide_velocity`` and ends ``air_clearance`` before the air wave; a
    trace whose window is shorter than ``min_window`` is skipped. Expects record
    statics applied and a causal high-pass, nothing else.

    * **onset**: the first sample whose envelope (RMS over ``smoothing``) exceeds
      ``onset_factor`` times the noise before the window and stays above 3x the
      noise for 80 % of the following ``sustain``, walked back to where the trace
      falls under ``walk_back`` times the noise.
    * **phase**: the first sample reaching ``phase_fraction`` of the window's
      peak amplitude - the first strong phase.
    """
    from scipy.ndimage import uniform_filter1d

    rows = _rows(traces)
    n_samples = rows.shape[1]
    off = geometry.abs_offset
    sel = np.ones(len(geometry), dtype=bool) if keep is None else np.asarray(keep, dtype=bool).copy()
    sel &= (off >= offset_range[0]) & (off <= offset_range[1])
    n_noise = max(4, int(round(noise_window / dt)))
    n_smooth = max(1, int(round(smoothing / dt)))
    n_sustain = max(1, int(round(sustain / dt)))
    out = {k: [] for k in ("trace", "record", "sx", "gx", "off", "onset", "phase")}
    for i in np.flatnonzero(sel):
        tp = guide_intercept + off[i] / guide_velocity
        ta = off[i] / airwave_velocity
        a = max(int((tp - half_window) / dt), 1)
        b = min(int(min(tp + half_window, ta - air_clearance) / dt), n_samples - n_sustain - 1)
        if b - a < int(min_window / dt):
            continue
        trace = rows[i]
        noise = float(np.sqrt(np.mean(trace[max(a - n_noise, 0):a] ** 2))) + 1e-30
        env = np.sqrt(uniform_filter1d(trace ** 2, n_smooth))
        loud = env > 3.0 * noise
        csum = np.concatenate(([0], np.cumsum(loud)))
        j_range = np.arange(a, b)
        frac = (csum[j_range + n_sustain] - csum[j_range]) / n_sustain
        hits = j_range[(env[a:b] > onset_factor * noise) & (frac > 0.8)]
        onset = np.nan
        if hits.size:
            j = int(hits[0])
            while j > a and abs(trace[j]) > walk_back * noise:
                j -= 1
            onset = j * dt
        segment = trace[a:b]
        strong = np.flatnonzero(np.abs(segment) >= phase_fraction * np.abs(segment).max())
        phase = (a + int(strong[0])) * dt if strong.size else np.nan
        out["trace"].append(int(i))
        out["record"].append(int(geometry.field_record[i]))
        out["sx"].append(float(geometry.source_x[i]))
        out["gx"].append(float(geometry.receiver_x[i]))
        out["off"].append(float(off[i]))
        out["onset"].append(onset)
        out["phase"].append(phase)
    return RefractionPicks(
        trace=np.array(out["trace"], dtype=int), record=np.array(out["record"], dtype=int),
        source_x=np.array(out["sx"]), receiver_x=np.array(out["gx"]), offset=np.array(out["off"]),
        onset=np.array(out["onset"], dtype=float), phase=np.array(out["phase"], dtype=float),
        settings=dict(airwave_velocity=airwave_velocity, guide_intercept=guide_intercept,
                      guide_velocity=guide_velocity, offset_range=tuple(offset_range),
                      half_window=half_window, air_clearance=air_clearance,
                      phase_fraction=phase_fraction, onset_factor=onset_factor))


def band_slope(picks: RefractionPicks, which: str = "phase",
               offset_band: Tuple[float, float] = (5.0, 10.0), *, n_boot: int = 300,
               seed: int = 0, min_per_side: int = 3, min_span: float = 1.5) -> Optional[Dict[str, Any]]:
    """One velocity for an offset band, with each record-side's intercept a fixed effect.

    ``t = intercept[record, side] + |x| * slowness``: a record-side needs
    ``min_per_side`` picks spanning ``min_span`` m. Robust (Huber-like) least
    squares, and a bootstrap over record-sides for the 5-95 % velocity range.
    None with fewer than two usable record-sides.
    """
    times = picks.column(which)
    x_all = picks.offset
    m = (x_all >= offset_band[0]) & (x_all <= offset_band[1]) & np.isfinite(times)
    side = np.sign(np.round(picks.receiver_x - picks.source_x, 6)).astype(int)
    keys = np.array([f"{r}:{s:+d}" for r, s in zip(picks.record[m], side[m])])
    x = x_all[m]
    t_ms = times[m] * 1e3
    usable = [k for k in np.unique(keys)
              if np.sum(keys == k) >= min_per_side and np.ptp(x[keys == k]) >= min_span]
    if len(usable) < 2:
        return None

    def fit(chosen: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chosen = list(chosen)
        rows = np.isin(keys, chosen)
        design = np.zeros((int(rows.sum()), len(chosen) + 1))
        position = {k: n for n, k in enumerate(chosen)}
        for n, (k, xx) in enumerate(zip(keys[rows], x[rows])):
            design[n, position[k]] = 1.0
            design[n, -1] = xx
        return _robust_lsq(design, t_ms[rows])

    sol, resid, weights = fit(usable)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(int(n_boot)):
        draw = list(dict.fromkeys(rng.choice(usable, len(usable), replace=True)))
        if len(draw) >= 2:
            boots.append(fit(draw)[0][-1])
    boots = np.array(boots) if boots else np.array([sol[-1]])
    slowness_ms = float(sol[-1])
    with np.errstate(divide="ignore"):
        velocity = 1e3 / slowness_ms if slowness_ms > 0 else float("nan")
        v_lo = 1e3 / float(np.percentile(boots, 95)) if np.percentile(boots, 95) > 0 else float("nan")
        v_hi = 1e3 / float(np.percentile(boots, 5)) if np.percentile(boots, 5) > 0 else float("inf")
    sides = []
    for k, ic in zip(usable, sol[:-1]):
        members = keys == k
        rec = int(k.split(":")[0])
        sgn = int(k.split(":")[1])
        sx = float(picks.source_x[m][members][0])
        sides.append({"record": rec, "side": sgn, "source_x": sx, "intercept": float(ic) * 1e-3,
                      "offset_min": float(x[members].min()), "offset_max": float(x[members].max()),
                      "n": int(members.sum())})
    return {"which": which, "band": (float(offset_band[0]), float(offset_band[1])),
            "velocity": float(velocity), "velocity_low": float(v_lo), "velocity_high": float(v_hi),
            "intercept": float(np.median(sol[:-1])) * 1e-3,
            "intercept_min": float(np.min(sol[:-1])) * 1e-3,
            "intercept_max": float(np.max(sol[:-1])) * 1e-3,
            "sides": sides, "n": int(resid.size), "n_sides": len(usable),
            "rms": float(np.sqrt(np.sum(weights * resid ** 2) / np.sum(weights))) * 1e-3}


def reciprocity_check(picks: RefractionPicks, which: str = "phase",
                      tolerance: float = 1e-3) -> Dict[str, Any]:
    """Compare each pick with its reciprocal (source and receiver swapped).

    Returns ``{"pairs": (n, 3) [source_x, receiver_x, t - t_reciprocal (s)],
    "n", "median_abs", "p90_abs"}``. Reciprocal times must agree; a large
    difference is a pick or timing error that no velocity model can fit.
    """
    times = picks.column(which)
    ok = np.isfinite(times)
    sx, gx, t = picks.source_x[ok], picks.receiver_x[ok], times[ok]
    pairs = []
    for k in range(t.size):
        match = np.flatnonzero(np.isclose(sx, gx[k], atol=tolerance) & np.isclose(gx, sx[k], atol=tolerance))
        if match.size and k < match[0]:
            pairs.append((sx[k], gx[k], t[k] - t[match[0]]))
    arr = np.array(pairs, dtype=float).reshape(-1, 3)
    diffs = np.abs(arr[:, 2]) if arr.size else np.array([])
    return {"pairs": arr, "n": int(arr.shape[0]),
            "median_abs": float(np.median(diffs)) if diffs.size else float("nan"),
            "p90_abs": float(np.percentile(diffs, 90)) if diffs.size else float("nan")}


def intercept_depth(intercept: Any, v1: Any, v2: Any) -> Any:
    """Depth to a refractor from its intercept time: ``h = ti V1 V2 / (2 sqrt(V2^2 - V1^2))``.

    NaN where ``V2 <= V1`` (no head wave).
    """
    ti = np.asarray(intercept, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        h = np.where(v2 > v1, ti * v1 * v2 / (2.0 * np.sqrt(np.maximum(v2 ** 2 - v1 ** 2, 1e-30))), np.nan)
    return float(h) if np.ndim(h) == 0 else h


def depth_bracket(intercept: float, v2: float,
                  v1: Sequence[float] = (250.0, 300.0, 343.0)) -> Dict[str, Any]:
    """Refractor depth over a bracket of top-layer velocities.

    The top layer's velocity is hidden behind the air wave on a hammer line, so
    the depth is given for ``v1 = (low, nominal, high)`` rather than one value:
    ``{"low", "nominal", "high"}`` depths (m) and the two-way times ``2 h / V1``
    at which a reflection from the refractor would stack. A non-positive
    intercept has no depth (NaN).
    """
    lo, nom, hi = (float(v) for v in v1)
    if not float(intercept) > 0:
        depths = [float("nan")] * 3
    else:
        depths = [intercept_depth(intercept, v, v2) for v in (lo, nom, hi)]
    twts = [2.0 * d / v if np.isfinite(d) else float("nan") for d, v in zip(depths, (lo, nom, hi))]
    finite = [d for d in depths if np.isfinite(d)]
    return {"intercept": float(intercept), "v2": float(v2), "v1": (lo, nom, hi),
            "depths": depths, "low": min(finite) if finite else float("nan"),
            "nominal": depths[1], "high": max(finite) if finite else float("nan"),
            "twt": twts}


def compare_pickers(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]],
                    tolerance: float = 0.2) -> Optional[str]:
    """A warning when two pickers' band velocities disagree, else None.

    They disagree when the velocities differ by more than ``tolerance`` of their
    mean, or their bootstrap ranges do not overlap.
    """
    if not a or not b:
        return None
    va, vb = a["velocity"], b["velocity"]
    if not (np.isfinite(va) and np.isfinite(vb)):
        return None
    rel = abs(va - vb) / (0.5 * (va + vb))
    apart = a["velocity_low"] > b["velocity_high"] or b["velocity_low"] > a["velocity_high"]
    if rel <= tolerance and not apart:
        return None
    lo, hi = a["band"]
    names = RefractionPicks.PICKERS
    return (f"Offsets {lo:g}-{hi:g} m: the {names[a['which']]} picks give {va:.0f} m/s "
            f"({a['velocity_low']:.0f}-{a['velocity_high']:.0f}) and the {names[b['which']]} picks "
            f"{vb:.0f} m/s ({b['velocity_low']:.0f}-{b['velocity_high']:.0f}); the pickers disagree "
            f"by {100 * rel:.0f} %, so this velocity is uncertain.")


@dataclass
class RefractionResult:
    """Band slopes of both pickers, reciprocity and the depth bracket."""

    picks: RefractionPicks
    bands: List[Dict[str, Any]]
    reciprocity: Dict[str, Dict[str, Any]]
    bracket: Optional[Dict[str, Any]]
    depth_band: Optional[Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)


def refraction_branches(traces: np.ndarray, dt: float, geometry: LineGeometry, *,
                        keep: Optional[np.ndarray] = None, airwave_velocity: float = 343.0,
                        guide_intercept: float = 0.010, guide_velocity: float = 700.0,
                        offset_range: Tuple[float, float] = (5.0, 18.0),
                        bands: Sequence[Tuple[float, float]] = ((5.0, 10.0), (10.0, 18.0)),
                        n_boot: int = 300, v1: Sequence[float] = (250.0, 300.0, 343.0),
                        depth_picker: str = "phase", depth_band: Optional[int] = None,
                        disagreement: float = 0.2, **pick_options: Any) -> RefractionResult:
    """Refraction velocities by offset band, from two pickers, with a depth bracket.

    The slopes are fitted per band (:func:`band_slope`) for the onset and the
    first-strong-phase picks; a band where the two disagree gets a warning
    (:func:`compare_pickers`). Reciprocity is checked for both pickers. The
    depth bracket (:func:`depth_bracket`) uses band ``depth_band`` (default: the
    farthest band with a positive intercept, preferring one where the pickers
    agree) and ``depth_picker``'s velocity and median intercept.
    """
    picks = refraction_picks(traces, dt, geometry, keep=keep, airwave_velocity=airwave_velocity,
                             guide_intercept=guide_intercept, guide_velocity=guide_velocity,
                             offset_range=offset_range, **pick_options)
    warnings: List[str] = []
    if len(picks) == 0:
        raise ValueError("No trace has a pick window clear of the air wave; lower the guide "
                         "intercept, raise the guide velocity or widen the offsets.")
    results = []
    for band in bands:
        entry = {"band": (float(band[0]), float(band[1]))}
        for which in ("onset", "phase"):
            entry[which] = band_slope(picks, which, band, n_boot=n_boot)
        entry["warning"] = compare_pickers(entry["onset"], entry["phase"], disagreement)
        if entry["warning"]:
            warnings.append(entry["warning"])
        if entry["onset"] is None and entry["phase"] is None:
            warnings.append(f"Offsets {band[0]:g}-{band[1]:g} m: too few record-sides with picks "
                            "for a slope.")
        results.append(entry)
    recip = {which: reciprocity_check(picks, which) for which in ("onset", "phase")}
    for which, r in recip.items():
        if r["n"] >= 5 and r["median_abs"] > 1.0e-3:
            warnings.append(f"Reciprocal {RefractionPicks.PICKERS[which]} times differ by a median "
                            f"of {r['median_abs'] * 1e3:.2f} ms over {r['n']} pairs: a timing or "
                            "picking error no velocity can fit.")
    chosen = None
    if depth_band is not None and 0 <= int(depth_band) < len(results):
        chosen = results[int(depth_band)].get(depth_picker)
    else:
        # The farthest band whose slope is usable - head waves lead at the far
        # offsets - preferring one where the two pickers agree.
        usable = [e for e in results if e.get(depth_picker)
                  and np.isfinite(e[depth_picker]["velocity"]) and e[depth_picker]["intercept"] > 0]
        agreed = [e for e in usable if not e.get("warning")]
        pool = agreed or usable
        if pool:
            best = max(pool, key=lambda e: (e["band"][0], -(e["band"][1] - e["band"][0])))
            chosen = best[depth_picker]
    bracket = None
    if chosen is not None and np.isfinite(chosen["velocity"]):
        bracket = depth_bracket(chosen["intercept"], chosen["velocity"], v1)
        if not chosen["intercept"] > 0:
            warnings.append(f"Offsets {chosen['band'][0]:g}-{chosen['band'][1]:g} m give a "
                            f"non-positive intercept ({chosen['intercept'] * 1e3:.2f} ms): no depth.")
        elif not all(np.isfinite(d) for d in bracket["depths"]):
            warnings.append(f"The refractor velocity {chosen['velocity']:.0f} m/s is not above the "
                            f"top-layer bracket {v1[0]:g}-{v1[2]:g} m/s; no depth for those values.")
    else:
        warnings.append("No offset band gave a refractor velocity and a positive intercept; "
                        "no depth bracket.")
    return RefractionResult(picks=picks, bands=results, reciprocity=recip, bracket=bracket,
                            depth_band=chosen, warnings=warnings,
                            settings=dict(airwave_velocity=airwave_velocity,
                                          guide_intercept=guide_intercept,
                                          guide_velocity=guide_velocity,
                                          offset_range=tuple(offset_range),
                                          bands=[tuple(b) for b in bands], n_boot=n_boot,
                                          v1=tuple(v1), depth_picker=depth_picker))


# ---------------------------------------------------------------------------
# The two pipelines the studio runs
# ---------------------------------------------------------------------------

@dataclass
class PreparedLine:
    """A line conditioned for stacking, and for refraction picking.

    ``traces``: record statics, causal high-pass, air-wave subtraction, causal
    band-pass and air-wave mute. ``refraction_traces``: record statics and the
    causal high-pass only. Both ``(n_samples, n_traces)``.
    """

    traces: np.ndarray
    refraction_traces: np.ndarray
    dt: float
    geometry: LineGeometry
    qc: TraceQC
    statics: Optional[AirwaveStatics]
    airwave_velocity: float
    settings: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def keep(self) -> np.ndarray:
        return self.qc.keep

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.traces.shape[0]) * self.dt


def prepare_line(traces: np.ndarray, dt: float, geometry: LineGeometry, *,
                 qc: Optional[TraceQC] = None, qc_options: Optional[Dict[str, Any]] = None,
                 statics: bool = True, statics_options: Optional[Dict[str, Any]] = None,
                 airwave_velocity: float = 343.0, highpass: float = 8.0,
                 suppress: bool = True, suppress_options: Optional[Dict[str, Any]] = None,
                 band: Tuple[float, float] = (50.0, 250.0), band_order: int = 4,
                 mute: Optional[Tuple[float, float]] = (0.75e-3, 5.0e-3), mute_taper: float = 0.5e-3,
                 log: LogFn = None) -> PreparedLine:
    """Trace QC, air-wave record statics and causal conditioning, in that order.

    1. QC on the raw traces (:func:`trace_qc`), unless ``qc`` is given.
    2. Record statics from the air wave (:func:`airwave_statics`), applied as a
       pure time shift; with ``statics=False`` none, and ``airwave_velocity`` is
       the air-wave velocity used below.
    3. Causal high-pass at ``highpass`` Hz -> ``refraction_traces``.
    4. Air-wave subtraction (:func:`suppress_airwave`) when ``suppress``.
    5. Causal band-pass ``band`` (:func:`causal_bandpass`).
    6. Air-wave mute ``mute = (before, after)`` (s) around ``|x| / c``.
    """
    notes: List[str] = []
    warnings: List[str] = []
    arr = np.asarray(traces, dtype=float)
    n_mean = max(1, min(arr.shape[0], int(round(2e-3 / dt))))
    arr = arr - arr[:n_mean].mean(axis=0, keepdims=True)
    if qc is None:
        qc = trace_qc(traces, dt, geometry, **(qc_options or {}))
    notes.append(qc.summary())
    notes.extend(qc.notes)
    _say(log, qc.summary())
    keep = qc.keep
    if not keep.any():
        raise ValueError("Trace QC left no traces; relax the QC or the record exclusions.")
    stat: Optional[AirwaveStatics] = None
    c = float(airwave_velocity)
    if statics:
        options = dict(statics_options or {})
        options.setdefault("velocity", float(airwave_velocity))
        options.setdefault("exclude_records", qc.settings.get("exclude_records", ()))
        stat = airwave_statics(traces, dt, geometry, keep=keep, **options)
        c = stat.velocity
        arr = apply_statics(arr, dt, stat.shifts(geometry))
        values = [v for r, v in stat.statics.items() if r not in stat.median_records]
        message = (f"Record statics from the air wave: c = {c:.1f} m/s, statics "
                   f"{min(values) * 1e3:+.2f} to {max(values) * 1e3:+.2f} ms, pick RMS "
                   f"{stat.rms * 1e3:.2f} ms.")
        notes.append(message)
        warnings.extend(stat.notes)
        _say(log, message)
    if highpass:
        arr = causal_highpass(arr, dt, highpass)
    refraction = arr.copy()
    db = float("nan")
    if suppress:
        arr, report = suppress_airwave(arr, dt, geometry, c, keep=keep, **(suppress_options or {}))
        db = report["reduction_db"]
        message = (f"Air wave subtracted on {report['traces']} traces of {report['records']} records"
                   + (f", {db:.1f} dB median reduction in its window." if np.isfinite(db) else "."))
        notes.append(message)
        _say(log, message)
    if band and (band[0] or band[1]):
        arr = causal_bandpass(arr, dt, band[0], band[1], band_order)
        notes.append(f"Causal band-pass {band[0]:g}-{band[1]:g} Hz (order {band_order}).")
    if mute is not None:
        arr = airwave_mute(arr, dt, geometry, c, before=mute[0], after=mute[1], taper=mute_taper)
    return PreparedLine(traces=arr, refraction_traces=refraction, dt=float(dt), geometry=geometry,
                        qc=qc, statics=stat, airwave_velocity=c,
                        settings=dict(statics=bool(statics), highpass=highpass, suppress=suppress,
                                      band=tuple(band) if band else None, band_order=band_order,
                                      mute=tuple(mute) if mute else None, mute_taper=mute_taper),
                        notes=notes, warnings=warnings)


@dataclass
class StackResult:
    """A CMP stack with the views that say whether its events are reflections."""

    stack: CMPStack
    near: Optional[CMPStack]
    far: Optional[CMPStack]
    split_offset: float
    supergather: Optional[Supergather]
    semblance_t0: np.ndarray
    semblance_velocities: np.ndarray
    semblance_hyperbolic: np.ndarray
    semblance_linear: np.ndarray
    peaks_hyperbolic: List[Tuple[float, float, float]]
    peaks_linear: List[Tuple[float, float, float]]
    velocity: VelocityFunction
    conditioned: np.ndarray
    traces_used: int
    zeroed: int
    settings: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def stack_line(prepared: PreparedLine, velocity: VelocityFunction, *,
               surface_wave_velocity: float = 220.0, surface_wave_pad: float = 3.0e-3,
               offset_range: Tuple[float, float] = (1.9, 10.0),
               cmp_range: Optional[Tuple[float, float]] = None, bin_width: float = 0.5,
               tmax: float = 0.05, stretch: float = 0.3, min_fold: int = 3,
               split_offset: Optional[float] = None,
               semblance_velocities: Tuple[float, float, int] = (120.0, 1500.0, 140),
               semblance_t0: Tuple[float, float] = (4.0e-3, 0.25e-3),
               supergather_cmp_range: Optional[Tuple[float, float]] = None,
               supergather_bin: float = 1.0, flatness_threshold: float = 0.5,
               max_residual_moveout: float = 1.0e-3, log: LogFn = None) -> StackResult:
    """Surface-wave mute, trace balance, CMP stack - and the checks next to it.

    The stack (:func:`cmp_stack`) comes with near/far offset stacks
    (:func:`offset_split_stacks`), hyperbolic and linear semblance on the
    balanced, un-muted traces (:func:`semblance`) and the NMO-corrected
    supergather with its flatness test (:func:`nmo_supergather`), whose events
    are also checked against the NMO-corrected mute edges.
    """
    dt = prepared.dt
    geometry = prepared.geometry
    keep = prepared.keep
    c = prepared.airwave_velocity
    mute_after = (prepared.settings.get("mute") or (0.75e-3, 5.0e-3))[1]
    off = geometry.abs_offset
    muted = surface_wave_mute(prepared.traces, dt, geometry, surface_wave_velocity,
                              pad=surface_wave_pad)
    start = off / c + mute_after + prepared.settings.get("mute_taper", 0.5e-3)
    end = off / surface_wave_velocity + surface_wave_pad
    conditioned, zeroed_mask = balance_traces(muted, dt, start, end)
    sel_all = _selection(geometry, keep, offset_range, cmp_range)
    zeroed = int(np.sum(zeroed_mask & sel_all))
    stack = cmp_stack(conditioned, dt, geometry, velocity, keep=keep, bin_width=bin_width,
                      offset_range=offset_range, cmp_range=cmp_range, tmax=tmax,
                      stretch=stretch, min_fold=min_fold, label="all offsets")
    sel = _selection(geometry, keep, offset_range, cmp_range)
    _say(log, f"Stacked {int(sel.sum())} traces in {stack.cmp_x.size} CMP bins of {bin_width:g} m "
              f"(fold up to {stack.max_fold}).")
    split = offset_split_stacks(conditioned, dt, geometry, velocity, keep=keep,
                                offset_range=offset_range, split_offset=split_offset,
                                bin_width=bin_width, cmp_range=cmp_range, tmax=tmax,
                                stretch=stretch, min_fold=max(1, min_fold - 1))
    # Semblance on the balanced traces without the surface-wave mute, so linear
    # energy can show itself for what it is.
    sem_traces, _ = balance_traces(prepared.traces, dt, start, np.full(off.size, tmax))
    t0s = np.arange(semblance_t0[0], tmax - 1e-12, semblance_t0[1])
    vels = np.linspace(semblance_velocities[0], semblance_velocities[1], int(semblance_velocities[2]))
    idx = np.flatnonzero(sel)
    hyp = semblance(sem_traces[:, idx], dt, off[idx], t0s, vels, kind="hyperbolic")
    lin = semblance(sem_traces[:, idx], dt, off[idx], t0s, vels, kind="linear")
    peaks_h = semblance_peaks(hyp, t0s, vels)
    peaks_l = semblance_peaks(lin, t0s, vels)

    def edges(x: np.ndarray) -> List[np.ndarray]:
        return [x / c + mute_after, x / surface_wave_velocity + surface_wave_pad]

    supergather = None
    warnings: List[str] = []
    try:
        supergather = nmo_supergather(conditioned, dt, geometry, velocity, keep=keep,
                                      cmp_range=supergather_cmp_range, offset_range=offset_range,
                                      tmax=tmax, stretch=stretch, bin_width=supergather_bin,
                                      threshold=flatness_threshold,
                                      max_residual_moveout=max_residual_moveout, mute_edges=edges)
    except ValueError as exc:
        warnings.append(f"No supergather: {exc}")
    notes = [f"Surface-wave mute at {surface_wave_velocity:g} m/s + {surface_wave_pad * 1e3:g} ms; "
             f"trace balance on the live corridor ({zeroed} of the stacked traces had none "
             "and were zeroed)."]
    if supergather is not None:
        passing = supergather.passing
        if not passing:
            warnings.append("No stacked event passes the NMO flatness test: read the stack as "
                            "the mute corridor's energy, not as reflections.")
        else:
            notes.append("Events passing the flatness test (candidates, not proof): "
                         + ", ".join(f"{e['t0'] * 1e3:.1f} ms" for e in passing) + ".")
        on_edge = [e for e in supergather.events if e["mute_edge"]]
        if on_edge:
            plural = len(on_edge) > 1
            warnings.append(("Events" if plural else "Event") + " at "
                            + ", ".join(f"{e['t0'] * 1e3:.1f} ms" for e in on_edge)
                            + (" sit" if plural else " sits")
                            + " on an NMO-corrected mute edge: likely mute artifacts.")
    if peaks_l and peaks_h and peaks_l[0][2] > peaks_h[0][2]:
        warnings.append(f"The strongest coherent energy is linear (semblance {peaks_l[0][2]:.2f} at "
                        f"{peaks_l[0][1]:.0f} m/s) rather than hyperbolic ({peaks_h[0][2]:.2f}): "
                        "surface waves or refractions dominate the window.")
    near, far = split.get("near"), split.get("far")
    for e in (near, far):
        if e is None:
            warnings.append("One of the near/far offset stacks has no traces; widen the offset range.")
            break
    return StackResult(stack=stack, near=near, far=far, split_offset=float(split["split_offset"]),
                       supergather=supergather, semblance_t0=t0s, semblance_velocities=vels,
                       semblance_hyperbolic=hyp, semblance_linear=lin, peaks_hyperbolic=peaks_h,
                       peaks_linear=peaks_l, velocity=velocity, conditioned=conditioned,
                       traces_used=int(sel.sum()), zeroed=int(zeroed),
                       settings=dict(surface_wave_velocity=surface_wave_velocity,
                                     surface_wave_pad=surface_wave_pad,
                                     offset_range=tuple(offset_range), cmp_range=cmp_range,
                                     bin_width=bin_width, tmax=tmax, stretch=stretch,
                                     min_fold=min_fold),
                       notes=notes, warnings=warnings)
