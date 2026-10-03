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

    def __call__(self, frequency: Any) -> np.ndarray:
        f = np.asarray(frequency, dtype=float)
        if not self.is_table:
            return np.full(f.shape, complex(self.gain))
        table_f = np.log(self.frequency)
        table_a = np.log(np.abs(self.values))
        log_f = np.log(np.maximum(f, 1e-300))
        inside = np.clip(log_f, table_f[0], table_f[-1])
        log_a = np.interp(inside, table_f, table_a)
        if table_f.size >= 2:
            k = min(3, table_f.size)
            low = np.polyfit(table_f[:k], table_a[:k], 1)[0]
            high = np.polyfit(table_f[-k:], table_a[-k:], 1)[0]
            log_a = log_a + np.clip(low, -4, 4) * np.minimum(log_f - table_f[0], 0.0)
            log_a = log_a + np.clip(high, -4, 4) * np.maximum(log_f - table_f[-1], 0.0)
        phase = np.interp(inside, table_f, np.unwrap(np.angle(self.values)))
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
    """One component's samples, in ``units``, with the chain that reaches the field."""

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
            channels[name] = replace(ch, data=data, units=ch.physical_units, response=[],
                                     metadata=dict(ch.metadata))
        return replace(run, channels=channels, metadata=dict(run.metadata))

    def summary(self) -> str:
        names = ", ".join(f"{name} [{ch.units}{'' if ch.is_calibrated else ', uncalibrated'}]"
                          for name, ch in self.channels.items())
        label = self.station or "run"
        return (f"{label}: {self.sample_rate:g} Hz, {self.n_samples} samples "
                f"({self.duration:g} s) from {self.start} - {names}")


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
    runs = []
    for start0, data0 in pieces[names[0]]:
        end0 = shifted(start0, data0.size / sample_rate)
        spans = {names[0]: (start0, data0)}
        for name in names[1:]:
            best = None
            for start, data in pieces[name]:
                end = shifted(start, data.size / sample_rate)
                overlap = seconds_between(max(start0, start), min(end0, end))
                if overlap > 0 and (best is None or overlap > best[0]):
                    best = (overlap, start, data)
            if best is None:
                break
            spans[name] = best[1:]
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
