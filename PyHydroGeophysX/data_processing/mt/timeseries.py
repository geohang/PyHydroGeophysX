"""MT time series as the instruments record them, with their responses.

Every instrument reader returns the same objects:

- :class:`TimeSeriesRun` - one stretch of simultaneous, gap-free samples of
  several channels at one sample rate: a continuous recording, or one burst of
  a segmented (scheduled) one. ``start`` is UTC as ``numpy.datetime64``.
- :class:`Channel` - one component's samples in the units the logger stored
  them in (volts, millivolts, counts, or nT for a fluxgate), with what is
  needed to reach the field: its :class:`ResponseStage` chain, orientation,
  dipole length and sensor.

**Response convention.** Each stage maps its input to its output, physical
side first: a coil is ``nT -> mV`` with a table of mV/nT, a dipole
``mV/km -> mV`` with gain ``L / 1000``, a receiver's anti-alias filter
``V -> V``. The product of the stages is what the recorded spectrum is per
unit of field, so the field's spectrum is the recorded one divided by
:meth:`Channel.response`. Electric fields come out in mV/km and magnetic ones
in nT, the units every MT code and file uses.

**Precision.** Data stay in the precision of the source: 24-bit samples and
float32 files as float32 (exact for 24-bit counts), 32-bit counts and text as
float64. A fluxgate's total field (~50 000 nT at pT resolution) needs float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: GPS - UTC, in seconds, from each date on (IERS Bulletin C).
_LEAP_SECONDS = (
    ("1999-01-01", 13), ("2006-01-01", 14), ("2009-01-01", 15),
    ("2012-07-01", 16), ("2015-07-01", 17), ("2017-01-01", 18),
)
_GPS_EPOCH = np.datetime64("1980-01-06T00:00:00", "ns")
_ONE_SECOND = np.timedelta64(1_000_000_000, "ns")

#: What a logger's channel name means, lower-cased: E1 is Ex, H3 is Hz...
COMPONENT_ALIASES = {
    "e1": "ex", "e2": "ey", "h1": "hx", "h2": "hy", "h3": "hz",
    "bx": "hx", "by": "hy", "bz": "hz",
}


def gps_leap_seconds(when: Any) -> int:
    """GPS - UTC at a date, from the leap-second table (13 s before 1999)."""
    moment = np.datetime64(when, "s")
    offset = 13
    for date, seconds in _LEAP_SECONDS:
        if moment >= np.datetime64(date, "s"):
            offset = seconds
    return offset


def gps_seconds_to_utc(seconds: Any, *, epoch: str = "unix",
                       leap_seconds: Optional[int] = None) -> np.datetime64:
    """UTC of a GPS-scale time stamp.

    ``epoch="unix"`` reads ``seconds`` as GPS time counted from 1970-01-01, as
    Phoenix writes it; ``"gps"`` counts from 1980-01-06. ``leap_seconds`` is
    GPS - UTC; by default it comes from the table for that date.
    """
    base = np.datetime64("1970-01-01T00:00:00", "ns") if epoch == "unix" else _GPS_EPOCH
    gps = base + np.timedelta64(int(round(float(seconds) * 1e9)), "ns")
    offset = gps_leap_seconds(gps) if leap_seconds is None else int(leap_seconds)
    return gps - offset * _ONE_SECOND


def as_datetime64(value: Any) -> Optional[np.datetime64]:
    """``value`` (string, datetime, datetime64) as UTC ``datetime64[ns]``; None stays None."""
    if value is None:
        return None
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[ns]")
    text = str(value).replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1]
    if "+" in text[10:]:
        text = text[:10] + text[10:].split("+")[0]
    return np.datetime64(text, "ns")


def seconds_between(start: np.datetime64, end: np.datetime64) -> float:
    return float((end - start) / np.timedelta64(1, "ns")) * 1e-9


def shifted(start: np.datetime64, seconds: float) -> np.datetime64:
    return start + np.timedelta64(int(round(seconds * 1e9)), "ns")


def _ns(when: np.datetime64) -> int:
    """Nanoseconds since 1970, for sorting and bisecting times."""
    return int(np.datetime64(when, "ns").astype(np.int64))


def normalize_component(name: str) -> str:
    key = str(name).strip().lower()
    return COMPONENT_ALIASES.get(key, key)


def _default_orientation(component: str) -> Tuple[float, float]:
    """(azimuth, tilt) of a component name: x north, y east, z down."""
    return {"ex": (0.0, 0.0), "hx": (0.0, 0.0), "ey": (90.0, 0.0), "hy": (90.0, 0.0),
            "hz": (0.0, 90.0)}.get(component, (float("nan"), 0.0))


@dataclass
class ResponseStage:
    """One stage of an instrument's response: a gain, a table, or both.

    ``frequency`` (Hz) and complex ``values`` tabulate the stage; between
    frequencies the amplitude is interpolated in log-log and the unwrapped
    phase in log frequency. Outside the table the amplitude follows the slope
    of its last three points in log-log (a coil's response falls as f below
    its corner) and the phase is held; :attr:`frequency_range` is what was
    measured. ``gain`` multiplies the table (or stands alone).
    """

    name: str
    input_units: str
    output_units: str
    gain: float = 1.0
    frequency: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frequency is None:
            if self.values is not None:
                raise ValueError(f"response stage {self.name!r} has values but no frequencies")
            return
        frequency = np.asarray(self.frequency, dtype=float).reshape(-1)
        values = np.asarray(self.values, dtype=complex).reshape(-1)
        if frequency.size != values.size or frequency.size == 0:
            raise ValueError(f"response stage {self.name!r}: {frequency.size} frequencies "
                             f"for {values.size} values")
        keep = np.isfinite(frequency) & (frequency > 0) & np.isfinite(values) & (np.abs(values) > 0)
        order = np.argsort(frequency[keep])
        self.frequency = frequency[keep][order]
        self.values = values[keep][order]

    @classmethod
    def from_amplitude_phase(cls, name: str, input_units: str, output_units: str,
                             frequency: Any, amplitude: Any, phase: Any, *,
                             phase_units: str = "deg", gain: float = 1.0,
                             metadata: Optional[Dict[str, Any]] = None) -> "ResponseStage":
        scale = {"deg": np.pi / 180.0, "rad": 1.0, "mrad": 1e-3}[phase_units]
        values = np.asarray(amplitude, dtype=float) * np.exp(1j * scale * np.asarray(phase, dtype=float))
        return cls(name, input_units, output_units, gain=gain, frequency=frequency,
                   values=values, metadata=dict(metadata or {}))

    @property
    def is_table(self) -> bool:
        return self.frequency is not None

    @property
    def frequency_range(self) -> Tuple[float, float]:
        """The tabulated band in Hz; a scalar stage holds at every frequency."""
        if not self.is_table:
            return 0.0, float("inf")
        return float(self.frequency[0]), float(self.frequency[-1])

    def _log_table(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Tuple[float, float]]]:
        """The table in log-log, its unwrapped phase and its end slopes.

        Fitted once per table rather than at every call: processing asks every
        channel's response at every level of every run, hundreds of thousands
        of times for a night of bursts.
        """
        cached = self.__dict__.get("_fitted")
        if cached is not None and cached[0] is self.frequency and cached[1] is self.values:
            return cached[2]
        table_f = np.log(self.frequency)
        table_a = np.log(np.abs(self.values))
        slopes = None
        if table_f.size >= 2:
            k = min(3, table_f.size)
            slopes = (np.polyfit(table_f[:k], table_a[:k], 1)[0],
                      np.polyfit(table_f[-k:], table_a[-k:], 1)[0])
        fitted = (table_f, table_a, np.unwrap(np.angle(self.values)), slopes)
        self.__dict__["_fitted"] = (self.frequency, self.values, fitted)
        return fitted

    def __call__(self, frequency: Any) -> np.ndarray:
        f = np.asarray(frequency, dtype=float)
        if not self.is_table:
            return np.full(f.shape, complex(self.gain))
        table_f, table_a, table_phase, slopes = self._log_table()
        log_f = np.log(np.maximum(f, 1e-300))
        inside = np.clip(log_f, table_f[0], table_f[-1])
        log_a = np.interp(inside, table_f, table_a)
        if slopes is not None:
            low, high = slopes
            log_a = log_a + np.clip(low, -4, 4) * np.minimum(log_f - table_f[0], 0.0)
            log_a = log_a + np.clip(high, -4, 4) * np.maximum(log_f - table_f[-1], 0.0)
        phase = np.interp(inside, table_f, table_phase)
        return self.gain * np.exp(log_a) * np.exp(1j * phase)


def read_response_table(path: Any, *, name: Optional[str] = None, input_units: str = "nT",
                        output_units: str = "mV", phase_units: str = "deg",
                        amplitude_per_hz: bool = False) -> ResponseStage:
    """A calibration table from a text file of frequency, amplitude, phase rows.

    Lines that do not start with three numbers are skipped, so headers and
    comments may be anywhere. ``amplitude_per_hz`` multiplies the amplitude by
    frequency, for tables normalized that way (Metronix V/(nT Hz)).
    """
    from pathlib import Path

    source = Path(path)
    rows = []
    for line in source.read_text(encoding="latin-1", errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        try:
            rows.append([float(v) for v in parts[:3]])
        except ValueError:
            continue
        if len(rows[-1]) < 3:
            rows.pop()
    if not rows:
        raise ValueError(f"{source.name} has no frequency, amplitude, phase rows")
    table = np.asarray(rows)
    amplitude = table[:, 1] * (table[:, 0] if amplitude_per_hz else 1.0)
    return ResponseStage.from_amplitude_phase(
        name or source.stem, input_units, output_units, table[:, 0], amplitude, table[:, 2],
        phase_units=phase_units, metadata={"source_file": str(source)})


def dipole_stage(length_m: float) -> ResponseStage:
    """The dipole: ``mV/km -> mV``, gain ``L / 1000``."""
    return ResponseStage(f"dipole_{length_m:g}m", "mV/km", "mV", gain=float(length_m) / 1000.0)


def unit_stage(input_units: str, output_units: str) -> ResponseStage:
    factors = {("mV", "V"): 1e-3, ("V", "mV"): 1e3, ("mV", "mV"): 1.0, ("V", "V"): 1.0}
    return ResponseStage(f"{input_units}_to_{output_units}", input_units, output_units,
                         gain=factors[(input_units, output_units)])


@dataclass
class Channel:
    """One component's samples, in ``units``, with the chain that reaches the field.

    ``metadata["full_scale"]``, where the reader knows it, is the level in
    ``units`` at which the recorder clips (Phoenix: the A/D's 5 V over the gain).
    """

    component: str
    data: np.ndarray
    units: str
    response: List[ResponseStage] = field(default_factory=list)
    azimuth: float = float("nan")
    tilt: float = 0.0
    dipole_length: float = float("nan")
    sensor: str = ""
    sensor_serial: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.component = normalize_component(self.component)
        data = np.asarray(self.data)
        if data.ndim != 1:
            raise ValueError(f"channel {self.component} data must be 1-D, got {data.shape}")
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(float)
        self.data = data
        azimuth, tilt = _default_orientation(self.component)
        if not np.isfinite(self.azimuth):
            self.azimuth = azimuth
        if self.tilt == 0.0 and tilt:
            self.tilt = tilt

    @property
    def kind(self) -> str:
        first = self.component[:1]
        return {"e": "electric", "h": "magnetic"}.get(first, "auxiliary")

    @property
    def physical_units(self) -> str:
        """The units of the field once the response is removed."""
        return self.response[0].input_units if self.response else self.units

    @property
    def is_calibrated(self) -> bool:
        """Whether removing the response gives mV/km (E) or nT (H)."""
        return self.physical_units == {"electric": "mV/km", "magnetic": "nT"}.get(self.kind, self.units)

    def response_at(self, frequency: Any) -> np.ndarray:
        """Recorded units per physical unit at ``frequency``: the stages' product."""
        f = np.asarray(frequency, dtype=float)
        total = np.ones(f.shape, dtype=complex)
        for stage in self.response:
            total = total * stage(f)
        return total

    def copy(self, data: Optional[np.ndarray] = None) -> "Channel":
        return replace(self, data=self.data.copy() if data is None else data,
                       response=list(self.response), metadata=dict(self.metadata))


@dataclass
class TimeSeriesRun:
    """Simultaneous, gap-free samples of several channels at one rate."""

    channels: Dict[str, Channel]
    sample_rate: float
    start: Any
    station: str = ""
    latitude: float = float("nan")
    longitude: float = float("nan")
    elevation: float = float("nan")
    declination: float = 0.0
    instrument: str = ""
    serial: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.channels, (list, tuple)):
            self.channels = {ch.component: ch for ch in self.channels}
        self.channels = {ch.component: ch for ch in self.channels.values()}
        if not self.channels:
            raise ValueError("a run needs at least one channel")
        lengths = {ch.data.size for ch in self.channels.values()}
        if len(lengths) != 1:
            sizes = {name: ch.data.size for name, ch in self.channels.items()}
            raise ValueError(f"channels of a run must have equal lengths: {sizes}")
        self.sample_rate = float(self.sample_rate)
        if not self.sample_rate > 0:
            raise ValueError("sample_rate must be positive")
        self.start = as_datetime64(self.start)
        self.metadata.setdefault("notes", [])

    def __getitem__(self, component: str) -> Channel:
        return self.channels[normalize_component(component)]

    def __contains__(self, component: str) -> bool:
        return normalize_component(component) in self.channels

    @property
    def components(self) -> List[str]:
        return list(self.channels)

    @property
    def n_samples(self) -> int:
        return next(iter(self.channels.values())).data.size

    @property
    def duration(self) -> float:
        """Seconds the samples span, ``n / sample_rate``."""
        return self.n_samples / self.sample_rate

    @property
    def end(self) -> np.datetime64:
        """The time just after the last sample."""
        return shifted(self.start, self.duration)

    def times(self) -> np.ndarray:
        """Each sample's UTC time as ``datetime64[ns]``."""
        step = 1e9 / self.sample_rate
        return self.start + (np.arange(self.n_samples) * step).round().astype("timedelta64[ns]")

    def array(self, components: Optional[Sequence[str]] = None, dtype: Any = float) -> np.ndarray:
        """The chosen channels stacked as ``(n_channels, n_samples)``."""
        names = self.components if components is None else [normalize_component(c) for c in components]
        return np.stack([self.channels[name].data.astype(dtype, copy=False) for name in names])

    def select(self, components: Iterable[str]) -> "TimeSeriesRun":
        names = [normalize_component(c) for c in components]
        missing = [name for name in names if name not in self.channels]
        if missing:
            raise KeyError(f"run has no {missing}; it has {self.components}")
        return replace(self, channels={name: self.channels[name] for name in names},
                       metadata=dict(self.metadata))

    def slice(self, first: int, last: Optional[int] = None) -> "TimeSeriesRun":
        """Samples ``first:last`` (views, not copies)."""
        first, last, _ = slice(first, last).indices(self.n_samples)
        channels = {name: replace(ch, data=ch.data[first:last]) for name, ch in self.channels.items()}
        return replace(self, channels=channels, start=shifted(self.start, first / self.sample_rate),
                       metadata=dict(self.metadata))

    def window(self, start: Any = None, end: Any = None) -> "TimeSeriesRun":
        """The samples between two UTC times (either may be None)."""
        first = 0 if start is None else int(np.ceil(
            seconds_between(self.start, as_datetime64(start)) * self.sample_rate - 0.01))
        last = self.n_samples if end is None else int(np.ceil(
            seconds_between(self.start, as_datetime64(end)) * self.sample_rate - 0.01))
        return self.slice(max(first, 0), max(min(last, self.n_samples), max(first, 0)))

    def calibrated(self, components: Optional[Sequence[str]] = None, *,
                   band: Optional[Tuple[float, float]] = None) -> "TimeSeriesRun":
        """The channels in field units (mV/km, nT), the response removed by FFT.

        Each channel is demeaned and divided by its response at every Fourier
        frequency; ``band`` (Hz) zeroes the spectrum outside it, which a coil's
        vanishing low-frequency response needs. This holds the whole series in
        memory as complex numbers: for long, fast recordings select a window first.
        """
        run = self if components is None else self.select(components)
        frequency = np.fft.rfftfreq(run.n_samples, d=1.0 / run.sample_rate)
        channels = {}
        for name, ch in run.channels.items():
            if not ch.response:
                channels[name] = ch.copy()
                continue
            only_gains = all(not stage.is_table for stage in ch.response)
            if only_gains and band is None:
                gain = ch.response_at(0.0).real
                data = ch.data.astype(float) / gain
            else:
                spectrum = np.fft.rfft(ch.data.astype(float) - np.nanmean(ch.data))
                response = ch.response_at(np.maximum(frequency, frequency[1] if frequency.size > 1 else 1.0))
                spectrum = spectrum / response
                spectrum[0] = 0.0
                if band is not None:
                    spectrum[(frequency < band[0]) | (frequency > band[1])] = 0.0
                data = np.fft.irfft(spectrum, n=run.n_samples)
            # A reader's ``full_scale`` is in the recorded units, which this leaves.
            channels[name] = replace(ch, data=data, units=ch.physical_units, response=[],
                                     metadata={k: v for k, v in ch.metadata.items() if k != "full_scale"})
        return replace(run, channels=channels, metadata=dict(run.metadata))

    def summary(self) -> str:
        names = ", ".join(f"{name} [{ch.units}{'' if ch.is_calibrated else ', uncalibrated'}]"
                          for name, ch in self.channels.items())
        label = self.station or "run"
        return (f"{label}: {self.sample_rate:g} Hz, {self.n_samples} samples "
                f"({self.duration:g} s) from {self.start} - {names}")


def _cycles_of(run: TimeSeriesRun, period: float) -> Optional[int]:
    """Samples in one ``period``, when the period is a whole number of them."""
    cycle = float(period) * run.sample_rate
    return int(round(cycle)) if cycle >= 2 and abs(cycle - round(cycle)) < 1e-6 else None


#: Share of a channel's full scale read as clipped: a decimated stream's filter
#: rounds the flat top of a clipped stretch off a little below it.
CLIP_LEVEL = 0.995


def saturated(run: TimeSeriesRun, components: Optional[Sequence[str]] = None, *,
              level: float = CLIP_LEVEL, guard: int = 2) -> Optional[np.ndarray]:
    """Which samples of ``run`` have one of ``components`` at the recorder's full scale.

    Only channels whose reader knows the full scale (``metadata["full_scale"]``)
    count; ``None`` when none does. Each clipped sample is widened by ``guard``
    samples either side. A clipped stretch is the recorder's limit, not the
    field: windows holding it give an impedance that is plausible and wrong.
    """
    names = [c for c in (components or list(run.channels)) if c in run.channels]
    mask, known = np.zeros(run.n_samples, dtype=bool), False
    for name in names:
        full = run.channels[name].metadata.get("full_scale")
        if full:
            known = True
            mask |= np.abs(np.asarray(run.channels[name].data, dtype=float)) >= level * float(full)
    if not known:
        return None
    if guard and mask.any():
        mask = np.convolve(mask, np.ones(2 * guard + 1), mode="same") > 0
    return mask


def without_samples(run: TimeSeriesRun, mask: np.ndarray) -> TimeSeriesRun:
    """``run`` with the samples ``mask`` marks set to NaN in every channel (their windows are then dropped)."""
    if mask is None or not np.any(mask):
        return run
    channels = {name: replace(ch, data=np.where(mask, np.nan, np.asarray(ch.data, dtype=float)),
                              metadata=dict(ch.metadata))
                for name, ch in run.channels.items()}
    return replace(run, channels=channels, metadata=dict(run.metadata))


def remove_periodic(run: TimeSeriesRun, period: float, *, cycles: int = 31) -> TimeSeriesRun:
    """``run`` less whatever repeats exactly every ``period`` seconds - a receiver's own interference.

    Each channel is first high-passed by subtracting its centred moving mean
    over one period, which passes the harmonics ``k / period`` whole and
    takes out everything slower; the waveform of each cycle is then the mean
    of that over the ``cycles`` cycles around it, and is subtracted from the
    channel. The whole is one time-invariant, zero-phase comb filter, the
    same on every channel, so an impedance is left as it was: notches about
    ``1 / (cycles * period)`` wide at the harmonics, the frequencies between
    them kept. (Taking each cycle's own mean off instead turned the field's
    slow drift into a staircase with steps at every cycle - a near-real
    "impedance" carrying the site's static shift.) Interference timed by the
    receiver's GPS clock repeats exactly; the field does not. ``period`` must
    be a whole number of samples; a run shorter than three periods is
    returned as it is.
    """
    cycle = _cycles_of(run, period)
    if cycle is None:
        raise ValueError(f"a period of {period:g} s is not a whole number of samples at {run.sample_rate:g} Hz")
    if run.n_samples // cycle < 3:
        return run
    channels = {name: replace(ch, data=_subtract_periodic(ch.data.astype(float), cycle, cycles)
                              .astype(ch.data.dtype, copy=False), metadata=dict(ch.metadata))
                for name, ch in run.channels.items()}
    notes = list(run.metadata.get("notes", [])) + [f"interference repeating every {period:g} s removed"]
    return replace(run, channels=channels, metadata={**run.metadata, "notes": notes})


def _subtract_periodic(data: np.ndarray, cycle: int, cycles: int = 31) -> np.ndarray:
    """``data`` (float) less what repeats every ``cycle`` samples, in place; see :func:`remove_periodic`."""
    n_cycles = data.size // cycle
    if n_cycles < 3:
        return data
    # Centred mean over one period (ends held at the first and last sample), by running sums; for an
    # even cycle, cycle + 1 samples with half weights at both ends.
    half = cycle // 2
    padded = np.pad(data, half, mode="edge")
    sums = np.concatenate([[0.0], np.cumsum(padded)])
    i = np.arange(data.size)
    if cycle % 2:
        mean = (sums[i + cycle] - sums[i]) / cycle
    else:
        mean = (sums[i + cycle + 1] - sums[i] - 0.5 * (padded[i] + padded[i + cycle])) / cycle
    fast = data - mean
    k = np.arange(n_cycles)
    low = np.clip(k - cycles // 2, 0, n_cycles)
    high = np.clip(k + cycles // 2 + 1, 0, n_cycles)
    running = np.cumsum(np.vstack([np.zeros((1, cycle)), fast[: n_cycles * cycle].reshape(n_cycles, cycle)]), axis=0)
    waveform = (running[high] - running[low]) / (high - low)[:, None]
    folds = data[: n_cycles * cycle].reshape(n_cycles, cycle)
    folds -= waveform
    tail = data.size - n_cycles * cycle
    if tail:
        data[n_cycles * cycle:] -= waveform[-1, :tail]
    return data


def remove_periodic_bursts(runs: Sequence[TimeSeriesRun], period: float, *,
                           cycles: int = 31) -> List[TimeSeriesRun]:
    """Bursts of a segmented recording less what repeats every ``period`` seconds.

    A burst shorter than a few periods cannot be folded on itself, but bursts
    that start at the same phase of the period - and have the same rate and
    length - carry the interference at the same place. Each such burst has
    the mean of the other bursts of that group among the ``cycles`` around
    it subtracted (its own field is left out of its template), less its own
    mean; the field, which does not repeat, averages out of it.
    A group of fewer than three bursts is left as recorded. This also takes
    off whatever every burst of the group starts with alike.
    """
    out = list(runs)
    groups: Dict[Tuple[float, int, int], List[int]] = {}
    period_ns = int(round(period * 1e9))
    for k, run in enumerate(runs):
        samples = int(round(period * run.sample_rate))
        phase = int(round((_ns(run.start) % period_ns) * run.sample_rate / 1e9)) % max(samples, 1)
        groups.setdefault((run.sample_rate, run.n_samples, phase), []).append(k)
    for members in groups.values():
        if len(members) < 3:
            continue
        members = sorted(members, key=lambda k: _ns(runs[k].start))
        m = len(members)
        j = np.arange(m)
        low, high = np.clip(j - cycles // 2, 0, m), np.clip(j + cycles // 2 + 1, 0, m)
        # At a group's ends a window of cycles holds fewer bursts; at least two others are needed.
        low = np.minimum(low, np.maximum(high - 3, 0))
        high = np.maximum(high, np.minimum(low + 3, m))
        first = runs[members[0]]
        cleaned: Dict[int, Dict[str, Channel]] = {k: {} for k in members}
        for name in first.channels:
            stack = np.stack([np.asarray(runs[k].channels[name].data, dtype=float) for k in members])
            centred = stack - stack.mean(axis=1, keepdims=True)
            running = np.cumsum(np.vstack([np.zeros((1, stack.shape[1])), centred]), axis=0)
            template = (running[high] - running[low] - centred) / (high - low - 1)[:, None]
            template -= template.mean(axis=1, keepdims=True)
            for row, k in enumerate(members):
                ch = runs[k].channels[name]
                cleaned[k][name] = replace(ch, data=(stack[row] - template[row]).astype(ch.data.dtype, copy=False),
                                           metadata=dict(ch.metadata))
        note = f"interference repeating every {period:g} s removed with the bursts at the same phase"
        for k in members:
            run = runs[k]
            out[k] = replace(run, channels={**run.channels, **cleaned[k]},
                             metadata={**run.metadata, "notes": list(run.metadata.get("notes", [])) + [note]})
    return out


def periodic_interference(run: TimeSeriesRun, periods: Sequence[float], *,
                          threshold: float = 10.0) -> Dict[float, List[str]]:
    """Which channels carry something repeating exactly every one of ``periods`` seconds.

    Each channel is band-passed to 0.1-20 Hz, so neither the field's long
    periods nor power-line hum swamps a pulse that lasts a fraction of a
    second, and folded on the period; a mean waveform standing more than
    ``threshold`` times above what chance leaves (about 4) is interference.
    The periods are tried shortest first, each one found removed before the
    next is tried: folded on 30 s, a pulse every 4 s still stands out (at half
    its height, since 30 s is not a multiple of 4), and would pass for a
    second interference. Returns ``{period: [channel, ...]}`` for the periods found.
    """
    from scipy import signal

    high = min(20.0, 0.4 * run.sample_rate)
    sos = signal.butter(4, [0.1, high], btype="band", fs=run.sample_rate, output="sos")
    passed = {name: signal.sosfilt(sos, np.asarray(ch.data, dtype=float)) for name, ch in run.channels.items()}
    found: Dict[float, List[str]] = {}
    for period in sorted(float(p) for p in periods):
        cycle = _cycles_of(run, period)
        n_cycles = run.n_samples // cycle if cycle else 0
        if n_cycles < 30:
            continue
        hits = []
        for name, data in passed.items():
            folds = data[: n_cycles * cycle].reshape(n_cycles, cycle)
            noise = folds.std(axis=0).mean() / np.sqrt(n_cycles)
            if noise > 0 and np.abs(folds.mean(axis=0)).max() > threshold * noise:
                hits.append(name)
        if hits:
            found[period] = hits
            for data in passed.values():
                _subtract_periodic(data, cycle)
    return found


def _powerline_lines(spectra: np.ndarray, df: float, nyquist: float, frequency: float,
                     threshold: float) -> List[Tuple[int, int]]:
    """Bin ranges, ``(first, last)``, of the hum's lines in ``spectra`` (channels, bins); see :func:`remove_powerline`."""
    from scipy import ndimage

    level = np.median(np.abs(spectra) ** 2, axis=1, keepdims=True)
    power = np.sum(np.abs(spectra) ** 2 / np.where(level > 0, level, 1.0), axis=0)
    smooth = max(1, int(round(0.01 / df)))
    power = ndimage.uniform_filter1d(power, smooth, mode="nearest")
    n_bins = power.size

    def peak(centre: float, half: float) -> int:
        lo, hi = max(1, int(np.floor((centre - half) / df))), min(n_bins - 1, int(np.ceil((centre + half) / df)))
        return lo + int(np.argmax(power[lo:hi + 1])) if hi >= lo else int(round(centre / df))

    fundamental = peak(frequency, 0.5) * df
    lines = []
    k = 1
    while k * fundamental < nyquist - df:
        centre = peak(k * fundamental, max(2 * df, 0.05 * k))
        near, far = max(3, int(round(1.0 / df))), max(10, int(round(3.0 / df)))
        side = np.r_[max(1, centre - far):max(1, centre - near), centre + near + 1:min(n_bins, centre + far + 1)]
        floor = np.median(power[side]) if side.size else 0.0
        if floor > 0 and power[centre] > threshold * floor:
            reach = max(3, int(round((0.5 + 0.05 * k) / df)))
            first, last = centre, centre
            while first > max(1, centre - reach) and power[first - 1] > threshold * floor:
                first -= 1
            while last < min(n_bins - 1, centre + reach) and power[last + 1] > threshold * floor:
                last += 1
            lines.append((max(1, first - smooth), min(n_bins - 1, last + smooth)))
        k += 1
    return lines


def remove_powerline(run: TimeSeriesRun, frequency: float = 60.0, *, threshold: float = 10.0) -> TimeSeriesRun:
    """``run`` with the power-line hum notched out: ``frequency`` and its harmonics below the Nyquist.

    Hum is many orders of magnitude stronger than the field, and in a record
    whose Nyquist lies above it - 60 Hz in 150 Hz data - it leaks through the
    taper's sidelobes into bands well away from it. There it passes for an
    impedance with apparent resistivity growing as the period and a phase of
    0 or 180 degrees, and as the grid is the same at a remote site, remote
    referencing keeps it. Each line is found where the channels' power,
    smoothed over 0.01 Hz, stands ``threshold`` times above the floor 1-3 Hz
    either side, and every bin of it - as wide as the grid's frequency
    wandered over the run - is set to zero, in one transform of the whole run:
    a zero-phase notch that leaves the frequencies between the harmonics as
    they were. A harmonic that does not stand out is left alone.

    A run shorter than a minute - a burst of a segmented recording - has bins
    too wide for that: hum that is not on a bin leaks across many of them and
    stays. There each line is fitted instead, as a sinusoid at its own
    frequency (the peak of the four-times zero-padded spectrum) with an
    amplitude that may change linearly over the run, and subtracted.
    """
    nyquist = run.sample_rate / 2.0
    if not 0 < frequency < nyquist or run.n_samples < 16:
        return run
    names = list(run.channels)
    data = np.vstack([np.asarray(run.channels[name].data, dtype=float) for name in names])
    t = np.linspace(-1.0, 1.0, run.n_samples)
    trend = np.polynomial.polynomial.polyfit(t, data.T, 1)                     # (2, channels)
    line = trend[0][:, None] + trend[1][:, None] * t[None, :]
    spectra = np.fft.rfft(data - line, axis=1)
    lines = _powerline_lines(spectra, run.sample_rate / run.n_samples, nyquist, float(frequency), threshold)
    if not lines:
        return run
    if run.duration < 60.0:
        cleaned = _subtract_tones(data, run.sample_rate, lines, float(frequency))
        how = "fitted and subtracted"
    else:
        for first, last in lines:
            spectra[:, first:last + 1] = 0.0
        cleaned = np.fft.irfft(spectra, n=run.n_samples, axis=1) + line
        how = "notched out"
    channels = {name: replace(run.channels[name], data=cleaned[k].astype(run.channels[name].data.dtype, copy=False),
                              metadata=dict(run.channels[name].metadata))
                for k, name in enumerate(names)}
    notes = list(run.metadata.get("notes", [])) + [
        f"power-line hum at {frequency:g} Hz {how} ({len(lines)} line(s) below the Nyquist)"]
    return replace(run, channels=channels, metadata={**run.metadata, "notes": notes})


def _subtract_tones(data: np.ndarray, rate: float, lines: Sequence[Tuple[int, int]], frequency: float,
                    pad: int = 4, iterations: int = 30) -> np.ndarray:
    """``data`` (channels, samples) less the hum fitted at each line; see :func:`remove_powerline`.

    Hum is one periodic waveform, its lines exact harmonics ``k f0`` of the
    grid's frequency. ``f0`` starts from the lines' peaks in the Hann-tapered
    spectrum zero-padded ``pad`` times (each refined by a parabola through
    the log power), ``k`` from ``frequency``. At each line, cos and sin and
    both times ``t`` (time from the middle) are fitted to every channel by
    least squares, with an offset and a trend that are not subtracted (hum of
    a fractional number of cycles has a trend of its own). The harmonics'
    projections all come from one chirp-z transform, and the fitted hum is
    put back together by another; the fit is iterated, since over a burst
    the harmonics and the trend are not quite orthogonal. The ``t`` terms take
    up a slowly changing amplitude and tell how far ``f0`` is off -
    ``Re[(a + b t) e^{i w t}]`` with ``b / a = i dw`` to first order - so it is
    refined, and the lines fitted again.
    """
    from scipy.signal import czt

    data = np.asarray(data, dtype=float)
    n = data.shape[1]
    centre = (n - 1) / 2.0
    t = (np.arange(n) - centre) / rate
    trend = np.column_stack([np.ones(n), t])
    flat = data - (trend @ np.linalg.lstsq(trend, data.T, rcond=None)[0]).T
    spectra = np.fft.rfft(flat * np.hanning(n), n=pad * n, axis=1)
    level = np.median(np.abs(spectra) ** 2, axis=1, keepdims=True)
    power = np.sum(np.abs(spectra) ** 2 / np.where(level > 0, level, 1.0), axis=0)
    peaks, strength = [], []
    for first, last in lines:
        lo, hi = max(1, first * pad), min(power.size - 2, last * pad + pad - 1)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(power[lo:hi + 1]))
        a, b, c = np.log(np.maximum(power[k - 1:k + 2], 1e-300))
        bend = a - 2 * b + c
        shift = 0.5 * (a - c) / bend if bend < 0 else 0.0
        peaks.append((k + float(np.clip(shift, -0.5, 0.5))) * rate / (pad * n))
        strength.append(power[k])
    if not peaks:
        return data
    peaks, strength = np.array(peaks), np.array(strength)
    order = np.maximum(np.round(peaks / frequency), 1.0)
    f0 = float(np.sum(strength * order * peaks) / np.sum(strength * order**2))
    harmonics = np.unique(np.round(peaks / f0).astype(int))
    harmonics = harmonics[(harmonics >= 1) & (harmonics * f0 < 0.49 * rate)]
    if harmonics.size == 0:
        return data
    K = int(harmonics.max())
    fitted = np.zeros(K, dtype=bool)
    fitted[harmonics - 1] = True
    k = np.arange(1, K + 1)
    sum_t2 = float(np.sum(t * t))

    def analyse(x: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
        """``sum x e^{-i k w t}`` and ``sum t x e^{-i k w t}`` for k = 1..K, ``step = w / rate``."""
        w, a = np.exp(-1j * step), np.exp(1j * step)
        phase = np.exp(1j * step * k * centre)
        return (czt(x, m=K, w=w, a=a, axis=-1) * phase, czt(x * t, m=K, w=w, a=a, axis=-1) * phase)

    def synthesise(alpha: np.ndarray, beta: np.ndarray, step: float) -> np.ndarray:
        """``Re sum_k (alpha_k + beta_k t) e^{i k w t}``."""
        phase = np.exp(-1j * step * k * centre)
        rotate = np.exp(1j * step * np.arange(n))
        w = np.exp(1j * step)
        first = czt(alpha * phase, m=n, w=w, a=1.0, axis=-1) * rotate
        second = czt(beta * phase, m=n, w=w, a=1.0, axis=-1) * rotate
        return np.real(first + t * second)

    step = 2 * np.pi * f0 / rate
    hum = np.zeros_like(data)
    for _ in range(4):
        alpha = np.zeros((data.shape[0], K), dtype=complex)
        beta = np.zeros_like(alpha)
        hum = np.zeros_like(data)
        for _ in range(iterations):
            rest = data - hum
            rest -= (trend @ np.linalg.lstsq(trend, rest.T, rcond=None)[0]).T
            A, B = analyse(rest, step)
            d_alpha, d_beta = 2 * A / n * fitted, 2 * B / sum_t2 * fitted
            alpha += d_alpha
            beta += d_beta
            hum = synthesise(alpha, beta, step)
            if np.max(np.abs(d_alpha)) <= 1e-7 * max(np.max(np.abs(alpha)), 1e-300):
                break
        weight = np.abs(alpha) ** 2 * k**2 * fitted
        if not weight.sum() > 0:
            break
        drift = np.imag(beta / np.where(np.abs(alpha) > 0, alpha, 1.0))        # rad/s, per harmonic
        d_step = float(np.sum(weight * drift / k) / weight.sum()) / rate
        if abs(d_step) * rate * t[-1] * K < 1e-4:
            break
        step += d_step
    return data - hum


def powerline_frequency(run: TimeSeriesRun, candidates: Sequence[float] = (50.0, 60.0), *,
                        threshold: float = 100.0) -> Optional[float]:
    """The grid frequency whose hum ``run`` carries, of ``candidates``: the one whose line stands
    highest above its floor, if more than ``threshold`` times; ``None`` when none does or none
    lies below the Nyquist."""
    nyquist = run.sample_rate / 2.0
    usable = [f for f in candidates if 0 < f < nyquist - 1.0]
    if not usable or run.n_samples < 16:
        return None
    n = min(run.n_samples, int(2 ** np.ceil(np.log2(min(run.n_samples, 600 * run.sample_rate)))))
    data = np.vstack([np.asarray(ch.data[:n], dtype=float) for ch in run.channels.values()])
    spectra = np.fft.rfft(data - data.mean(axis=1, keepdims=True), axis=1)
    level = np.median(np.abs(spectra) ** 2, axis=1, keepdims=True)
    power = np.sum(np.abs(spectra) ** 2 / np.where(level > 0, level, 1.0), axis=0)
    df = run.sample_rate / n
    best, height = None, threshold
    for f in usable:
        k = int(round(f / df))
        near, far = max(3, int(round(1.0 / df))), max(10, int(round(3.0 / df)))
        side = np.r_[max(1, k - far):max(1, k - near), k + near + 1:min(power.size, k + far + 1)]
        top = power[max(1, k - max(1, int(round(0.3 / df)))):k + max(1, int(round(0.3 / df))) + 1].max()
        floor = np.median(power[side]) if side.size else 0.0
        if floor > 0 and top / floor > height:
            best, height = f, top / floor
    return best


def contiguous_blocks(blocks: Sequence[Tuple[np.datetime64, np.ndarray]], sample_rate: float,
                      *, tolerance: float = 0.25) -> List[Tuple[np.datetime64, np.ndarray]]:
    """Join time-stamped blocks that follow each other without a gap.

    A block continues the previous one when it starts within ``tolerance``
    samples of where that one ended; otherwise a new segment starts.
    """
    segments: List[Tuple[np.datetime64, List[np.ndarray]]] = []
    end = None
    for start, data in sorted(blocks, key=lambda item: item[0]):
        if data.size == 0:
            continue
        if end is not None and abs(seconds_between(end, start)) * sample_rate <= tolerance:
            segments[-1][1].append(data)
            end = shifted(end, data.size / sample_rate)
        else:
            segments.append((start, [data]))
            end = shifted(start, data.size / sample_rate)
    return [(start, np.concatenate(parts)) for start, parts in segments]


def assemble_runs(pieces: Dict[str, List[Tuple[np.datetime64, np.ndarray]]],
                  sample_rate: float, make_channel, *, min_samples: int = 1,
                  **run_fields: Any) -> List[TimeSeriesRun]:
    """Runs from each component's contiguous segments, where all of them overlap.

    ``pieces`` maps a component to its ``(start, data)`` segments;
    ``make_channel(component, data)`` builds the :class:`Channel`. Overlaps
    are cut to whole samples on a common grid; offsets of a fraction of a
    sample between channels are reported in the run's notes.
    """
    names = list(pieces)
    if not names:
        return []
    # The other components' segments sorted by start (ns), with the latest end
    # reached so far: a segment's overlaps are then found by bisection. A
    # segmented recording has hundreds of bursts per channel, and comparing
    # every burst with every other is quadratic.
    sorted_spans = {}
    for name in names[1:]:
        starts = np.array([_ns(start) for start, _ in pieces[name]], dtype=np.int64)
        ends = np.array([_ns(shifted(start, data.size / sample_rate)) for start, data in pieces[name]],
                        dtype=np.int64)
        order = np.argsort(starts, kind="stable")
        sorted_spans[name] = (order, starts[order], ends[order], np.maximum.accumulate(ends[order]))
    runs = []
    for start0, data0 in pieces[names[0]]:
        end0 = shifted(start0, data0.size / sample_rate)
        first_ns, end_ns = _ns(start0), _ns(end0)
        spans = {names[0]: (start0, data0)}
        for name in names[1:]:
            order, starts, ends, reach = sorted_spans[name]
            lo = int(np.searchsorted(reach, first_ns, side="right"))
            hi = int(np.searchsorted(starts, end_ns, side="left"))
            overlap = np.minimum(end_ns, ends[lo:hi]) - np.maximum(first_ns, starts[lo:hi])
            if not (overlap > 0).any():
                break
            # The longest overlap; of equal ones, the segment listed first.
            best = int(order[lo:hi][overlap == overlap.max()].min())
            spans[name] = pieces[name][best]
        if len(spans) != len(names):
            continue
        common_start = max(start for start, _ in spans.values())
        common_end = min(shifted(start, data.size / sample_rate) for start, data in spans.values())
        n = int(np.floor(seconds_between(common_start, common_end) * sample_rate + 0.01))
        if n < min_samples:
            continue
        channels, worst = [], 0.0
        for name, (start, data) in spans.items():
            offset = seconds_between(start, common_start) * sample_rate
            first = int(round(offset))
            worst = max(worst, abs(offset - first))
            channels.append(make_channel(name, data[first:first + n]))
        run = TimeSeriesRun(channels=channels, sample_rate=sample_rate, start=common_start,
                            **{k: v for k, v in run_fields.items() if k != "metadata"},
                            metadata=dict(run_fields.get("metadata", {})))
        run.metadata["notes"] = list(run.metadata.get("notes", []))
        if worst > 0.01:
            run.metadata["notes"].append(
                f"channel clocks differ by up to {worst:.2f} samples; aligned to the nearest sample")
        runs.append(run)
    return runs


_SCALARS = ("station", "latitude", "longitude", "elevation", "declination", "instrument", "serial")
_CHANNEL_SCALARS = ("units", "azimuth", "tilt", "dipole_length", "sensor", "sensor_serial")


def _jsonable(value: Any) -> Any:
    """Metadata as JSON: numbers, strings, lists and dicts of them; anything else as text."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def runs_to_payload(runs: Sequence[TimeSeriesRun]) -> List[Dict[str, Any]]:
    """Runs as plain values and arrays, for :func:`PyHydroGeophysX.data_processing.run_inputs.save_container`."""
    payload = []
    for run in runs:
        channels = []
        for ch in run.channels.values():
            stages = [{"name": s.name, "input_units": s.input_units, "output_units": s.output_units,
                       "gain": float(s.gain),
                       "frequency": s.frequency if s.is_table else np.zeros(0),
                       "values": s.values if s.is_table else np.zeros(0, dtype=complex)}
                      for s in ch.response]
            channels.append({"component": ch.component, "data": ch.data, "response": stages,
                             **{k: _jsonable(getattr(ch, k)) for k in _CHANNEL_SCALARS},
                             "metadata": _jsonable(ch.metadata)})
        payload.append({"sample_rate": float(run.sample_rate), "start": str(run.start),
                        "channels": channels, "metadata": _jsonable(run.metadata),
                        **{k: _jsonable(getattr(run, k)) for k in _SCALARS}})
    return payload


def runs_from_payload(payload: Sequence[Dict[str, Any]]) -> List[TimeSeriesRun]:
    """The runs :func:`runs_to_payload` took apart."""
    runs = []
    nan = lambda v: float("nan") if v is None else v
    for item in payload:
        channels = []
        for ch in item["channels"]:
            stages = [ResponseStage(s["name"], s["input_units"], s["output_units"], gain=float(s["gain"]),
                                    frequency=np.asarray(s["frequency"]) if np.size(s["frequency"]) else None,
                                    values=np.asarray(s["values"]) if np.size(s["values"]) else None)
                      for s in ch.get("response", [])]
            channels.append(Channel(component=ch["component"], data=np.asarray(ch["data"]),
                                    units=ch["units"], response=stages, azimuth=nan(ch.get("azimuth")),
                                    tilt=nan(ch.get("tilt")) if ch.get("tilt") is not None else 0.0,
                                    dipole_length=nan(ch.get("dipole_length")),
                                    sensor=ch.get("sensor") or "", sensor_serial=ch.get("sensor_serial") or "",
                                    metadata=dict(ch.get("metadata") or {})))
        runs.append(TimeSeriesRun(channels=channels, sample_rate=item["sample_rate"], start=item["start"],
                                  station=item.get("station") or "",
                                  latitude=nan(item.get("latitude")), longitude=nan(item.get("longitude")),
                                  elevation=nan(item.get("elevation")),
                                  declination=item.get("declination") or 0.0,
                                  instrument=item.get("instrument") or "", serial=item.get("serial") or "",
                                  metadata=dict(item.get("metadata") or {})))
    return runs


def save_runs(destination: Any, runs: Sequence[TimeSeriesRun]):
    """Write runs as one compressed container (``.npz``); returns its path."""
    from PyHydroGeophysX.data_processing.run_inputs import save_container

    return save_container(destination, runs_to_payload(runs), kind="mt_timeseries",
                          meta={"n_runs": len(runs)})


def load_runs(source: Any) -> List[TimeSeriesRun]:
    """The runs a :func:`save_runs` container holds."""
    from PyHydroGeophysX.data_processing.run_inputs import load_container

    return runs_from_payload(load_container(source, kind="mt_timeseries"))
