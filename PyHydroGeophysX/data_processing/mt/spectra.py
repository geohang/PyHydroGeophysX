"""Fourier coefficients of MT time series, level by level, in field units.

The cascade EMTF made standard (Egbert & Booker 1986; Egbert 1997): level 0
is the recorded series; each further level low-passes and decimates the one
before by ``decimation_factor`` (a zero-phase FIR, so every channel keeps its
timing), and drops the samples at either end that the filter computed partly
from beyond the record; the first and last windows of each decimated level are
left out too. At every level the series is cut into windows of ``window_length``
samples overlapping by ``overlap``; each window is detrended, first
differenced (prewhitening, undone after the transform), tapered and
transformed, and every Fourier coefficient is divided by its channel's
response at that frequency - so the coefficients are mV/km and nT, whatever
the instrument. Windows sit on a grid counted from each level's first
sample, so two runs cut to the same span - a site and its remote reference -
share their windows.

A frequency band is a set of harmonics of one level's windows, as in an EMTF
band-setup file (``level first last``, level counted from 1).

With ``tapers`` above one, each window is instead tapered by that many
Slepian sequences (Thomson's multitaper method, time-bandwidth
``(tapers + 1) / 2``): their coefficients at a harmonic are uncorrelated
estimates of the same spectrum, so a window gives a band ``tapers`` points
per harmonic, where one taper would give one - most worth it in the narrow
bands at the long-period end of each level. The price is resolution: a
coefficient averages the spectrum over ``±(tapers + 1) / 2`` harmonics.

Egbert, G. D. & Booker, J. R. (1986). Robust estimation of geomagnetic
transfer functions. Geophysical Journal of the Royal Astronomical Society,
87(1), 173-194. https://doi.org/10.1111/j.1365-246X.1986.tb04552.x

Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
Proceedings of the IEEE, 70(9), 1055-1096. https://doi.org/10.1109/PROC.1982.12433
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .timeseries import TimeSeriesRun


@dataclass
class Band:
    """Harmonics ``first..last`` of the windows of decimation ``level`` (0-based), less any in ``skip``."""

    level: int
    first: int
    last: int
    skip: Tuple[int, ...] = ()

    def harmonics(self) -> np.ndarray:
        h = np.arange(self.first, self.last + 1)
        return h[~np.isin(h, self.skip)] if self.skip else h

    def frequencies(self, sample_rate: float, window_length: int) -> np.ndarray:
        return self.harmonics() * sample_rate / window_length

    def center(self, sample_rate: float, window_length: int) -> float:
        """The band's frequency: the mean of its harmonics' (EMTF's convention)."""
        return float(np.mean(self.frequencies(sample_rate, window_length)))


#: EMTF's bands for 128-point windows: the first level, the levels between, the last.
_FIRST_LEVEL = [(25, 30), (20, 24), (16, 19), (13, 15), (10, 12), (8, 9), (6, 7), (5, 5)]
_MIDDLE_LEVEL = [(14, 17), (11, 13), (9, 10), (7, 8), (6, 6), (5, 5)]
_LAST_LEVEL = [(18, 22), (14, 17), (10, 13), (7, 9), (5, 6)]


def default_bands(n_levels: int, window_length: int = 128) -> List[Band]:
    """EMTF's standard band set for ``n_levels`` levels of 128-point windows.

    For other window lengths the harmonics are scaled by ``window_length / 128``.
    """
    scale = window_length / 128.0
    bands: List[Band] = []
    for level in range(n_levels):
        if level == 0:
            table = _FIRST_LEVEL
        elif level == n_levels - 1:
            table = _LAST_LEVEL
        else:
            table = _MIDDLE_LEVEL
        for first, last in table:
            a = max(1, int(round(first * scale)))
            b = max(a, int(round(last * scale)))
            bands.append(Band(level, a, b))
    return bands


def log_bands(sample_rate: float, n_levels: int, window_length: int = 128, decimation_factor: int = 4,
              per_decade: int = 10, width: float = 1.0 / 3.0) -> List[Band]:
    """Overlapping bands centred on frequencies evenly spaced in log, ``per_decade`` to a decade.

    After Chen (2020): a band holds the harmonics within ``1 +- width`` of
    its centre (``width = 1/Q``, Q = 3), so it spans a factor of about 2 and
    neighbouring bands share most of their harmonics - the curve is
    continuous, as in commercial processing that smooths over frequency, at
    the price of resolution and of errors correlated between neighbours.
    Each centre is taken at the deepest level that holds it at harmonic 20
    (of 128) or below - the longest windows that resolve it - and at level 0
    up to harmonic 30; never below harmonic 5. The centres are the same for
    every sample rate, so two rates meet on the same grid.
    """
    scale = window_length / 128.0
    lowest, highest = max(1, int(round(5 * scale))), max(1, int(round(30 * scale)))
    spacing = sample_rate / window_length                          # level 0's harmonic spacing
    top = highest * spacing
    bottom = lowest * spacing / float(decimation_factor) ** (n_levels - 1)
    bands: List[Band] = []
    seen = set()
    for k in range(int(np.floor(per_decade * np.log10(top) + 1e-9)),
                   int(np.ceil(per_decade * np.log10(bottom) - 1e-9)) - 1, -1):
        frequency = 10.0 ** (k / per_decade)
        level = harmonic = None
        for candidate in range(n_levels - 1, -1, -1):
            h = frequency * float(decimation_factor) ** candidate / spacing
            if h <= (highest if candidate == 0 else max(4, decimation_factor) * lowest) + 1e-9:
                level, harmonic = candidate, h
                break
        if level is None or harmonic < lowest - 1e-9:
            continue
        first = max(lowest, int(np.floor(harmonic * (1.0 - width))) + 1)
        last = max(first, int(np.ceil(harmonic * (1.0 + width))) - 1)
        last = min(last, highest)
        if (level, first, last) not in seen:
            seen.add((level, first, last))
            bands.append(Band(level, first, last))
    return bands


def read_band_setup(path: Any) -> List[Band]:
    """An EMTF band-setup file: the number of bands, then ``level first last`` rows."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            try:
                rows.append(Band(int(parts[0]) - 1, int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"{path} has no 'level first last' rows")
    return rows


@dataclass
class SpectraConfig:
    window_length: int = 128
    overlap: int = 32
    decimation_factor: int = 4
    taper: str = "hamming"
    prewhiten: bool = True
    min_windows: int = 5
    #: Slepian tapers per window; above one they replace ``taper`` (multitaper).
    tapers: int = 1


@dataclass
class LevelSpectra:
    """One level's Fourier coefficients: ``coefficients[c]`` is ``(windows, harmonics)``.

    With several tapers, ``harmonics`` holds each harmonic once per taper
    (taper by taper), and so do the coefficients' columns.
    """

    level: int
    sample_rate: float
    window_starts: np.ndarray
    harmonics: np.ndarray
    coefficients: Dict[str, np.ndarray] = field(default_factory=dict)


#: Outputs at each end of a decimated series whose filter reaches past the
#: input: scipy's ``decimate`` uses ``20 * factor + 1`` taps, ``10 * factor``
#: inputs either side of each output, which is 10 outputs.
_EDGE = 10


def _decimate(x: np.ndarray, factor: int) -> Tuple[np.ndarray, int]:
    """Low-pass and decimate along the last axis, every channel of a run at once.

    The filter runs over zeros beyond the ends, so the outputs it computed
    partly from them are dropped: a step from a channel's offset, or from hum
    cut off mid-cycle, would otherwise ring into the first and last windows -
    in a 2 s burst, a third of the windows of the deepest level. Returns the
    series and how many outputs it lost at the start.
    """
    if factor == 1:
        return x, 0
    from scipy import signal

    y = signal.decimate(x, factor, ftype="fir", zero_phase=True, axis=-1)
    last = (x.shape[-1] - 1) // factor - _EDGE
    return y[..., _EDGE:max(last + 1, _EDGE)], _EDGE


@lru_cache(maxsize=16)
def _tapers(window_length: int, taper: str, tapers: int) -> np.ndarray:
    """The data windows, ``(tapers, window_length)``: ``taper`` alone, or Slepian sequences."""
    from scipy import signal

    if tapers <= 1:
        return signal.get_window(taper, window_length, fftbins=False)[None, :]
    return signal.windows.dpss(window_length, (tapers + 1) / 2.0, Kmax=tapers, sym=True)


@lru_cache(maxsize=32)
def _window_operator(window_length: int, taper: str, prewhiten: bool,
                     harmonics: Tuple[int, ...], tapers: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """What a window's samples become at ``harmonics``, as one matrix (real and imaginary parts).

    Detrending, first differencing, tapering and the Fourier transform are all
    linear, so their chain is a matrix: each row is what it makes of a unit
    impulse at that sample. A run's windows times this matrix are their
    coefficients - one product in place of four passes and an FFT per channel.
    Each taper gives a block of columns, normalised to its own power.
    """
    from scipy import signal

    impulses = signal.detrend(np.eye(window_length), axis=1, type="linear")
    if prewhiten:
        impulses = np.diff(impulses, axis=1, prepend=impulses[:, :1])
    weights = _tapers(window_length, taper, tapers)
    norm = np.sqrt(2.0 / np.sum(weights**2, axis=1))
    blocks = [np.fft.rfft(impulses * w, axis=1)[:, list(harmonics)] * s for w, s in zip(weights, norm)]
    operator = np.concatenate(blocks, axis=1)
    return np.ascontiguousarray(operator.real), np.ascontiguousarray(operator.imag)


def n_levels_for(n_samples: int, config: SpectraConfig) -> int:
    """How many levels leave at least ``min_windows`` windows that are used.

    Counted as :func:`run_spectra` makes them: each decimation keeps
    ``(n - 1) // D + 1`` samples less the ``_EDGE`` trimmed at each end, and a
    decimated level leaves out its first and last window.
    """
    stride = config.window_length - config.overlap
    levels, n = 0, int(n_samples)
    while True:
        dropped = 2 if levels else 0
        if n < config.window_length + (config.min_windows + dropped - 1) * stride:
            return levels
        levels += 1
        n = (n - 1) // config.decimation_factor + 1 - 2 * _EDGE


def run_spectra(run: TimeSeriesRun, components: Sequence[str], harmonics: Dict[int, np.ndarray],
                config: SpectraConfig) -> List[LevelSpectra]:
    """Calibrated Fourier coefficients of ``components`` at the levels in ``harmonics``.

    ``harmonics`` maps a level to the harmonic indices its bands use. Windows
    holding a non-finite sample are dropped.
    """
    L, stride = config.window_length, config.window_length - config.overlap
    n_tapers = max(int(config.tapers), 1)
    components = list(components)
    series = np.stack([run[c].data.astype(float) for c in components])
    results: List[LevelSpectra] = []
    rate = run.sample_rate
    origin = 0.0      # seconds from the run's start to the level's first sample
    for level in range(max(harmonics) + 1 if harmonics else 0):
        if level:
            series, lost = _decimate(series, config.decimation_factor)
            rate /= config.decimation_factor
            origin += lost / rate
        n = series.shape[1]
        if n < L or level not in harmonics:
            if n < L:
                break
            continue
        index = np.asarray(harmonics[level], dtype=int)
        frequency = index * rate / L
        n_windows = (n - L) // stride + 1
        # A record's ends - a burst's start above all - carry the instrument's own
        # transient for tens of milliseconds, which the decimation spreads over the
        # first and last windows of every level below the first: those are left out
        # (a decimated level of two windows or fewer gives none).
        kept = slice(1, max(n_windows - 1, 1)) if level else slice(0, n_windows)
        starts = (np.arange(n_windows) * stride)[kept]
        offsets = np.round((origin + starts / rate) * 1e9).astype(np.int64).astype("timedelta64[ns]")
        spectra = LevelSpectra(level, rate, run.start + offsets, np.tile(index, n_tapers))
        finite = np.isfinite(series)
        clean = series if finite.all() else np.nan_to_num(series)
        windows = np.lib.stride_tricks.sliding_window_view(clean, L, axis=1)[:, ::stride][:, :n_windows][:, kept]
        real, imag = _window_operator(L, config.taper, bool(config.prewhiten),
                                      tuple(int(k) for k in index), n_tapers)
        fc = (windows @ real + 1j * (windows @ imag)) / np.sqrt(rate)
        recolor = 1.0 - np.exp(-2j * np.pi * frequency / rate) if config.prewhiten else 1.0
        for k, c in enumerate(components):
            spectra.coefficients[c] = fc[k] / np.tile(recolor * run[c].response_at(frequency), n_tapers)
        if clean is not series:
            # A window holding a non-finite sample, in any channel, is dropped.
            spans = np.lib.stride_tricks.sliding_window_view(finite.all(axis=0), L)[::stride][:n_windows][kept]
            good = spans.all(axis=1)
            spectra.window_starts = spectra.window_starts[good]
            spectra.coefficients = {c: v[good] for c, v in spectra.coefficients.items()}
        results.append(spectra)
    return results


def variance_inflation(config: SpectraConfig, n_harmonics: int) -> float:
    """How much a band of ``n_harmonics`` adjacent harmonics overstates its information.

    The taper makes the coefficients of neighbouring harmonics - and of
    overlapping windows - correlated, for the noise and the fields alike, so a
    regression on them has ``(1/m) sum_kl |rho_(k-l)|^2`` times the variance it
    would have on independent points, ``rho_k`` the taper's correlation at a
    lag of ``k`` harmonics; overlapping windows add the same sum over the
    pairs of points one window and the next share.

    With several tapers the sums run over every pair of points, ``rho`` then
    the correlation of taper ``j`` at one harmonic with taper ``l`` at
    another: zero between two tapers at the same harmonic, which is why they
    add information. (A Monte Carlo of white noise agrees within a few per
    cent, for one taper and for several.)
    """
    L = config.window_length
    weights = _tapers(L, config.taper, max(int(config.tapers), 1))
    power = np.sum(weights**2, axis=1)
    n = np.arange(L)
    m = max(int(n_harmonics), 1)
    lags = np.arange(-(m - 1), m)
    shifts = np.exp(-2j * np.pi * np.outer(lags, n) / L)
    scale = np.sqrt(np.outer(power, power))[..., None]
    stride = L - config.overlap
    total, shift = 0.0, 0
    while shift < L:
        # The tapers of the window ``shift`` samples on, on this window's samples.
        other = np.zeros_like(weights)
        other[:, shift:] = weights[:, :L - shift]
        rho = np.einsum("jn,ln,dn->jld", weights, other, shifts) / scale
        total += (1.0 if shift == 0 else 2.0) * float(np.sum((m - np.abs(lags)) * np.abs(rho) ** 2))
        if stride <= 0:
            break
        shift += stride
    return total / (m * len(weights))


def band_harmonics(bands: Sequence[Band]) -> Dict[int, np.ndarray]:
    found: Dict[int, set] = {}
    for band in bands:
        found.setdefault(band.level, set()).update(range(band.first, band.last + 1))
    return {level: np.array(sorted(h)) for level, h in found.items()}


def band_data(level_spectra: Sequence[LevelSpectra], band: Band,
              components: Sequence[str]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """One band's coefficients, flattened over windows and harmonics, with window times."""
    out: Dict[str, List[np.ndarray]] = {c: [] for c in components}
    times: List[np.ndarray] = []
    for spectra in level_spectra:
        if spectra.level != band.level:
            continue
        columns = np.flatnonzero((spectra.harmonics >= band.first) & (spectra.harmonics <= band.last)
                                 & ~np.isin(spectra.harmonics, band.skip))
        if columns.size == 0:
            continue
        for c in components:
            out[c].append(spectra.coefficients[c][:, columns])
        times.append(np.repeat(spectra.window_starts, columns.size))
    if not times:
        return {c: np.zeros((0,), complex) for c in components}, np.zeros(0, "datetime64[ns]")
    return ({c: np.concatenate([v.ravel() for v in out[c]]) for c in components},
            np.concatenate(times))


def shared_windows(local: LevelSpectra, remote: LevelSpectra) -> Tuple[LevelSpectra, LevelSpectra]:
    """The two levels cut to the windows they share, matched by start time.

    A remote window is the local one's when it starts within half a sample
    of it at that level: start times rounded to the nanosecond, or a remote
    clock a fraction of a sample off, still pair (a constant offset of the
    remote changes the reference's phase only, which cancels).
    """
    a = local.window_starts.astype("datetime64[ns]").astype(np.int64)
    b = remote.window_starts.astype("datetime64[ns]").astype(np.int64)
    if a.size == 0 or b.size == 0:
        i = j = np.zeros(0, dtype=int)
    else:
        order = np.argsort(b, kind="stable")
        sorted_b = b[order]
        pos = np.clip(np.searchsorted(sorted_b, a), 1, sorted_b.size) - 1
        after = np.minimum(pos + 1, sorted_b.size - 1)
        nearest = np.where(np.abs(sorted_b[after] - a) < np.abs(sorted_b[pos] - a), after, pos)
        close = np.abs(sorted_b[nearest] - a) <= 0.5e9 / local.sample_rate
        i, j = np.flatnonzero(close), order[nearest[close]]
        # one remote window for one local window
        j, first = np.unique(j, return_index=True)
        i = i[first]
    cut = lambda s, k: LevelSpectra(s.level, s.sample_rate, s.window_starts[k], s.harmonics,
                                    {c: v[k] for c, v in s.coefficients.items()})
    return cut(local, i), cut(remote, j)
