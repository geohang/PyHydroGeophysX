"""Fourier coefficients of MT time series, level by level, in field units.

The cascade EMTF made standard (Egbert & Booker 1986; Egbert 1997): level 0
is the recorded series; each further level low-passes and decimates the one
before by ``decimation_factor`` (a zero-phase FIR, so every channel keeps its
timing). At every level the series is cut into windows of ``window_length``
samples overlapping by ``overlap``; each window is detrended, first
differenced (prewhitening, undone after the transform), tapered and
transformed, and every Fourier coefficient is divided by its channel's
response at that frequency - so the coefficients are mV/km and nT, whatever
the instrument. Windows sit on a grid counted from the run's first sample,
so two runs cut to the same start - a site and its remote reference - share
their windows.

A frequency band is a set of harmonics of one level's windows, as in an EMTF
band-setup file (``level first last``, level counted from 1).

Egbert, G. D. & Booker, J. R. (1986). Robust estimation of geomagnetic
transfer functions. Geophysical Journal of the Royal Astronomical Society,
87(1), 173-194. https://doi.org/10.1111/j.1365-246X.1986.tb04552.x
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal

from .timeseries import TimeSeriesRun, shifted


@dataclass
class Band:
    """Harmonics ``first..last`` of the windows of decimation ``level`` (0-based)."""

    level: int
    first: int
    last: int

    def frequencies(self, sample_rate: float, window_length: int) -> np.ndarray:
        return np.arange(self.first, self.last + 1) * sample_rate / window_length

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


@dataclass
class LevelSpectra:
    """One level's Fourier coefficients: ``coefficients[c]`` is ``(windows, harmonics)``."""

    level: int
    sample_rate: float
    window_starts: np.ndarray
    harmonics: np.ndarray
    coefficients: Dict[str, np.ndarray] = field(default_factory=dict)


def _decimate(x: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return x
    return signal.decimate(x, factor, ftype="fir", zero_phase=True)


def n_levels_for(n_samples: int, config: SpectraConfig) -> int:
    """How many levels leave at least ``min_windows`` windows."""
    stride = config.window_length - config.overlap
    levels, n = 0, n_samples
    while n >= config.window_length + (config.min_windows - 1) * stride:
        levels += 1
        n //= config.decimation_factor
    return levels


def run_spectra(run: TimeSeriesRun, components: Sequence[str], harmonics: Dict[int, np.ndarray],
                config: SpectraConfig) -> List[LevelSpectra]:
    """Calibrated Fourier coefficients of ``components`` at the levels in ``harmonics``.

    ``harmonics`` maps a level to the harmonic indices its bands use. Windows
    holding a non-finite sample are dropped.
    """
    L, stride = config.window_length, config.window_length - config.overlap
    taper = signal.get_window(config.taper, L, fftbins=False)
    norm = np.sqrt(2.0 / np.sum(taper**2))
    series = {c: run[c].data.astype(float) for c in components}
    results: List[LevelSpectra] = []
    rate = run.sample_rate
    for level in range(max(harmonics) + 1 if harmonics else 0):
        if level:
            series = {c: _decimate(x, config.decimation_factor) for c, x in series.items()}
            rate /= config.decimation_factor
        n = next(iter(series.values())).size
        if n < L or level not in harmonics:
            if n < L:
                break
            continue
        index = np.asarray(harmonics[level], dtype=int)
        frequency = index * rate / L
        n_windows = (n - L) // stride + 1
        starts = np.arange(n_windows) * stride
        spectra = LevelSpectra(level, rate, np.array([shifted(run.start, s / rate) for s in starts],
                                                     dtype="datetime64[ns]"), index)
        good = np.ones(n_windows, dtype=bool)
        recolor = 1.0 - np.exp(-2j * np.pi * frequency / rate) if config.prewhiten else 1.0
        for c, x in series.items():
            windows = np.lib.stride_tricks.sliding_window_view(x, L)[::stride][:n_windows]
            good &= np.all(np.isfinite(windows), axis=1)
            windows = signal.detrend(np.nan_to_num(windows), axis=1, type="linear")
            if config.prewhiten:
                windows = np.diff(windows, axis=1, prepend=windows[:, :1])
            fc = np.fft.rfft(windows * taper, axis=1)[:, index] * norm / np.sqrt(rate)
            fc = fc / recolor / run[c].response_at(frequency)
            spectra.coefficients[c] = fc
        if not np.all(good):
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
    lag of ``k`` harmonics. The overlap's correlation multiplies it.
    """
    L = config.window_length
    taper = signal.get_window(config.taper, L, fftbins=False)
    power = np.sum(taper**2)
    n = np.arange(L)
    m = max(int(n_harmonics), 1)
    rho = [abs(np.sum(taper**2 * np.exp(-2j * np.pi * k * n / L))) / power for k in range(1, m)]
    harmonic = 1.0 + 2.0 / m * sum((m - k) * r**2 for k, r in enumerate(rho, start=1))
    stride = L - config.overlap
    overlap = float(np.sum(taper[stride:] * taper[:L - stride]) / power) if stride < L else 0.0
    return harmonic * (1.0 + 2.0 * overlap**2)


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
        columns = np.flatnonzero((spectra.harmonics >= band.first) & (spectra.harmonics <= band.last))
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
    """The two levels cut to the windows they share, matched by start time."""
    _, i, j = np.intersect1d(local.window_starts, remote.window_starts, return_indices=True)
    cut = lambda s, k: LevelSpectra(s.level, s.sample_rate, s.window_starts[k], s.harmonics,
                                    {c: v[k] for c, v in s.coefficients.items()})
    return cut(local, i), cut(remote, j)
