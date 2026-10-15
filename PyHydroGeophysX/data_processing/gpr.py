"""Ground-penetrating radar profiles: read, process, and convert time to depth.

A small toolkit for common-offset GPR lines (B-scans). A profile is held as one
:class:`GPRProfile` - samples by traces, the sample interval in nanoseconds and
each trace's position along the line in metres - and every processing step
takes a profile and returns a new one, with the step written into its
``history``. The page that drives it is the studio's GPR page
(``qt_apps/modules/gpr_processing.py``).

The steps and their usual order follow the open-source GPR codes - GPRPy
(Plattner, 2020), readgssi (Nesbitt et al.), ImpDAR (Lilien et al., 2020) and
RGPR (Huber and Hans, 2018) - and are written here from their descriptions:

1. :func:`dewow` - remove the slow drift ("wow") each trace starts with;
2. :func:`set_time_zero` - put time zero on the first arrival;
3. :func:`bandpass` - zero-phase Butterworth band-pass;
4. :func:`remove_background` - subtract the mean trace (horizontal banding);
5. :func:`migrate` - constant-velocity F-K (Stolt) migration;
6. :func:`gain` - power-of-time or automatic gain control, for display.

:func:`process` runs a list of such steps in that order. Time converts to depth
with a single velocity (:func:`time_to_depth`, ``depth = v t / 2``), picked by
fitting a diffraction hyperbola (:func:`hyperbola_time`) or taken from a
permittivity (:func:`velocity_from_permittivity`).

Files
-----
:func:`read_gpr` reads GSSI ``.DZT``, Sensors & Software pulseEKKO ``.DT1``
(with its ``.HD``), MALA ``.rd3``/``.rd7`` (with its ``.rad``), SEG-Y and this
module's own ``.npz`` (:func:`save_profile`).

Example data
------------
:func:`example_profile` simulates a 100 MHz line over the MODFLOW catchment
model shipped in ``examples/data``: the model's water content and porosity are
turned into permittivity with the CRIM mixing law (:func:`crim_permittivity`)
and a reflection is drawn at each layer boundary (:func:`synthetic_profile`).
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]

#: Speed of light in vacuum, m/ns.
C0 = 0.299792458

#: The processing steps :func:`process` knows, in the order it applies them.
STEP_ORDER: Tuple[str, ...] = ("dewow", "time_zero", "bandpass", "background",
                               "migrate", "gain")

#: File suffixes :func:`read_gpr` reads, for a file dialog.
FILE_FILTER = ("GPR data (*.dzt *.DZT *.dt1 *.DT1 *.hd *.HD *.rd3 *.rd7 *.rad *.RAD "
               "*.sgy *.segy *.SGY *.SEGY *.npz);;All files (*)")

#: The suffixes that only a GPR line has: a file ending in one is radar data
#: whatever a request calls it. SEG-Y and ``.npz`` are not among them.
GPR_ONLY_SUFFIXES: Tuple[str, ...] = (".dzt", ".dt1", ".hd", ".rd3", ".rd7", ".rad")


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------

@dataclass
class GPRProfile:
    """One GPR line: samples by traces, with its time and position axes.

    Attributes
    ----------
    data : ndarray
        ``(n_samples, n_traces)`` amplitudes, float.
    dt_ns : float
        Sample interval, ns.
    positions : ndarray
        ``(n_traces,)`` distance of each trace along the line, m.
    t_start_ns : float
        Time of the first sample, ns. 0 as recorded; negative once time zero
        has been moved down onto the first arrival with the samples above it
        kept.
    frequency_mhz : float
        Nominal antenna frequency, MHz (NaN when unknown).
    antenna_separation : float
        Transmitter-receiver offset, m (NaN when unknown).
    elevation : ndarray or None
        ``(n_traces,)`` ground elevation under each trace, m, for a
        topographically corrected section.
    source : str
        The file read, or what made the profile.
    instrument : str
        The system or format: ``"GSSI DZT"``, ``"pulseEKKO DT1"``, ...
    history : list of str
        The processing applied, one line per step, in order.
    metadata : dict
        Header values kept as read.
    """

    data: np.ndarray
    dt_ns: float
    positions: np.ndarray
    t_start_ns: float = 0.0
    frequency_mhz: float = float("nan")
    antenna_separation: float = float("nan")
    elevation: Optional[np.ndarray] = None
    source: str = ""
    instrument: str = ""
    history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"GPR data must be 2-D (samples x traces); got {self.data.shape}.")
        self.dt_ns = float(self.dt_ns)
        if not (self.dt_ns > 0.0 and np.isfinite(self.dt_ns)):
            raise ValueError(f"The sample interval must be positive; got {self.dt_ns} ns.")
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1)
        if self.positions.size != self.data.shape[1]:
            raise ValueError(f"{self.positions.size} positions for {self.data.shape[1]} traces.")
        if self.elevation is not None:
            self.elevation = np.asarray(self.elevation, dtype=float).reshape(-1)
            if self.elevation.size != self.data.shape[1]:
                raise ValueError(f"{self.elevation.size} elevations for "
                                 f"{self.data.shape[1]} traces.")

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_traces(self) -> int:
        return int(self.data.shape[1])

    @property
    def time_ns(self) -> np.ndarray:
        """Two-way travel time of each sample, ns."""
        return self.t_start_ns + np.arange(self.n_samples) * self.dt_ns

    @property
    def trace_spacing(self) -> float:
        """The median distance between neighbouring traces, m (0 for one trace)."""
        if self.n_traces < 2:
            return 0.0
        return float(np.median(np.abs(np.diff(self.positions))))

    @property
    def sampling_mhz(self) -> float:
        """The sampling frequency, MHz."""
        return 1000.0 / self.dt_ns

    def with_data(self, data: np.ndarray, step: str = "", **changes: Any) -> "GPRProfile":
        """A copy holding ``data`` (and any other ``changes``), ``step`` added to the history."""
        history = list(self.history) + ([step] if step else [])
        return replace(self, data=np.asarray(data, dtype=float), history=history,
                       metadata=dict(self.metadata), **changes)

    def describe(self) -> str:
        """One plain sentence on what the profile holds."""
        parts = [f"{self.n_traces} traces x {self.n_samples} samples",
                 f"{self.time_ns[-1] - self.time_ns[0] + self.dt_ns:.0f} ns window "
                 f"at {self.dt_ns:.3g} ns"]
        if self.n_traces > 1:
            parts.append(f"{abs(self.positions[-1] - self.positions[0]):.1f} m line, "
                         f"{self.trace_spacing:.3g} m between traces")
        if np.isfinite(self.frequency_mhz):
            parts.append(f"{self.frequency_mhz:.0f} MHz antenna")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Velocity, permittivity and depth
# ---------------------------------------------------------------------------

def velocity_from_permittivity(permittivity: Any) -> Any:
    """Radar velocity in a low-loss material, m/ns: ``c / sqrt(eps_r)``.

    Examples
    --------
    >>> round(velocity_from_permittivity(9.0), 4)
    0.0999
    """
    eps = np.asarray(permittivity, dtype=float)
    out = C0 / np.sqrt(np.clip(eps, 1.0, None))
    return float(out) if out.ndim == 0 else out


def permittivity_from_velocity(velocity: Any) -> Any:
    """The relative permittivity a velocity in m/ns stands for: ``(c / v)^2``."""
    v = np.asarray(velocity, dtype=float)
    out = (C0 / v) ** 2
    return float(out) if out.ndim == 0 else out


def time_to_depth(time_ns: Any, velocity: float) -> Any:
    """Depth of a reflector at two-way time ``time_ns``, m: ``v t / 2``.

    Examples
    --------
    >>> time_to_depth(100.0, 0.1)
    5.0
    """
    t = np.asarray(time_ns, dtype=float)
    out = 0.5 * float(velocity) * t
    return float(out) if out.ndim == 0 else out


def depth_to_time(depth: Any, velocity: float) -> Any:
    """Two-way time to a reflector at ``depth`` m, ns: ``2 d / v``."""
    d = np.asarray(depth, dtype=float)
    out = 2.0 * d / float(velocity)
    return float(out) if out.ndim == 0 else out


def hyperbola_time(x: Any, x0: float, t0_ns: float, velocity: float) -> np.ndarray:
    """Two-way time of the diffraction from a point at ``(x0, t0)``, ns.

    A small object at depth ``d = v t0 / 2`` under position ``x0`` is reached
    from position ``x`` along a slant path, so its echo arrives at
    ``sqrt(t0^2 + (2 (x - x0) / v)^2)``. Drawn over a hyperbola on the
    radargram it checks a velocity: too slow a velocity gives a curve
    narrower than the hyperbola, too fast a wider one.
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(float(t0_ns) ** 2 + (2.0 * (x - float(x0)) / float(velocity)) ** 2)


def crim_permittivity(porosity: Any, water_content: Any, *, eps_solid: float = 4.6,
                      eps_water: float = 80.0, eps_air: float = 1.0) -> np.ndarray:
    """Bulk relative permittivity of a soil by the CRIM mixing law.

    ``sqrt(eps) = (1 - phi) sqrt(eps_s) + theta sqrt(eps_w) + (phi - theta) sqrt(eps_a)``
    (Birchak et al., 1974; Roth et al., 1990), with the water content clipped
    to the porosity.

    Examples
    --------
    >>> round(float(crim_permittivity(0.3, 0.3)), 2)
    14.43
    """
    phi = np.clip(np.asarray(porosity, dtype=float), 0.0, 1.0)
    theta = np.clip(np.asarray(water_content, dtype=float), 0.0, None)
    theta = np.minimum(theta, phi)
    root = ((1.0 - phi) * np.sqrt(eps_solid) + theta * np.sqrt(eps_water)
            + (phi - theta) * np.sqrt(eps_air))
    return root ** 2


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def _running_mean(values: np.ndarray, width: int, axis: int) -> np.ndarray:
    """Centred running mean over ``width`` samples along ``axis``; edges reflect."""
    from scipy.ndimage import uniform_filter1d

    width = int(max(1, width))
    if width <= 1:
        return values.copy()
    return uniform_filter1d(values, size=width, axis=axis, mode="reflect")


def default_dewow_window(profile: GPRProfile) -> float:
    """A dewow window of three antenna periods, ns (30 ns when the frequency is unknown)."""
    if np.isfinite(profile.frequency_mhz) and profile.frequency_mhz > 0:
        return float(3000.0 / profile.frequency_mhz)
    return 30.0


def dewow(profile: GPRProfile, window_ns: Optional[float] = None) -> GPRProfile:
    """Remove the slow drift each trace starts with: subtract its running mean.

    The receiver's coupling to the transmitter puts a decaying low-frequency
    swing under the early part of every trace (the "wow"). A running mean over
    ``window_ns`` - a few periods of the antenna, so the signal's cycles
    average out of it - follows the drift and not the signal, and is
    subtracted. Shorter windows cut into the signal's low frequencies.
    """
    window = float(window_ns) if window_ns else default_dewow_window(profile)
    width = max(3, int(round(window / profile.dt_ns)))
    data = profile.data - _running_mean(profile.data, width, axis=0)
    return profile.with_data(data, f"Dewow: running mean over {window:.3g} ns removed")


def pick_time_zero(profile: GPRProfile, threshold: float = 0.1) -> float:
    """The time of the first arrival on the line, ns.

    Taken on the mean of the traces' absolute amplitudes, after their own
    median is removed: the first sample that rises above ``threshold`` of the
    largest value. One time for the line, since the antennas' delay is the
    same for every trace.
    """
    data = profile.data - np.median(profile.data, axis=0, keepdims=True)
    envelope = np.mean(np.abs(data), axis=1)
    envelope = envelope - np.median(envelope[: max(3, envelope.size // 20)])
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if not peak > 0.0:
        return float(profile.time_ns[0])
    index = int(np.argmax(envelope >= float(threshold) * peak))
    return float(profile.time_ns[index])


def set_time_zero(profile: GPRProfile, time_ns: Optional[float] = None, *,
                  threshold: float = 0.1, crop: bool = True) -> GPRProfile:
    """Move time zero onto ``time_ns`` (the picked first arrival when None).

    With ``crop`` the samples before the new zero are dropped, as GPRPy and
    RGPR do; without it they stay at negative times. When the antenna
    separation is known the zero is put that much earlier than the first
    break, which is the direct air wave that crossed it.
    """
    if time_ns is None:
        picked = pick_time_zero(profile, threshold)
        offset = profile.antenna_separation / C0 if np.isfinite(profile.antenna_separation) else 0.0
        zero = picked - offset
        how = f"first arrival picked at {picked:.3g} ns"
        if offset:
            how += f", less {offset:.3g} ns for the {profile.antenna_separation:.3g} m antenna offset"
    else:
        zero = float(time_ns)
        how = "set by hand"
    time = profile.time_ns
    if crop:
        keep = time >= zero - 0.5 * profile.dt_ns
        if not np.any(keep):
            raise ValueError(f"Time zero {zero:.3g} ns is after the last sample.")
        first = int(np.argmax(keep))
        data = profile.data[first:]
        start = float(time[first] - zero)
    else:
        data = profile.data
        start = float(time[0] - zero)
    return profile.with_data(data, f"Time zero at {zero:.3g} ns ({how})", t_start_ns=start)


def bandpass(profile: GPRProfile, low_mhz: float, high_mhz: float, order: int = 4) -> GPRProfile:
    """Zero-phase Butterworth band-pass between ``low_mhz`` and ``high_mhz``."""
    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * profile.sampling_mhz
    low, high = float(low_mhz), float(high_mhz)
    if not (0.0 < low < high):
        raise ValueError(f"The band must run from a low to a higher frequency; got {low}-{high} MHz.")
    if high >= nyquist:
        raise ValueError(f"The upper corner ({high:g} MHz) must be below half the sampling "
                         f"frequency ({nyquist:.4g} MHz).")
    sos = butter(int(order), [low, high], btype="band", fs=profile.sampling_mhz, output="sos")
    padlen = min(profile.n_samples - 1, 3 * (2 * len(sos) + 1))
    data = sosfiltfilt(sos, profile.data, axis=0, padlen=max(0, padlen))
    return profile.with_data(data, f"Band-pass {low:g}-{high:g} MHz (Butterworth, order {order})")


def remove_background(profile: GPRProfile, window_traces: int = 0) -> GPRProfile:
    """Subtract the mean trace: the horizontal bands antenna ringing leaves.

    ``window_traces`` 0 subtracts the mean of the whole line; a number takes
    the mean of that many neighbouring traces instead, which keeps gently
    dipping reflectors that a whole-line mean would also wash out.
    """
    window = int(window_traces or 0)
    if window <= 0 or window >= profile.n_traces:
        mean = np.mean(profile.data, axis=1, keepdims=True)
        how = "mean trace of the line removed"
    else:
        mean = _running_mean(profile.data, window, axis=1)
        how = f"running mean of {window} traces removed"
    return profile.with_data(profile.data - mean, f"Background: {how}")


def gain(profile: GPRProfile, kind: str = "agc", *, power: float = 1.0,
         window_ns: float = 25.0) -> GPRProfile:
    """Lift the late, weak part of the traces, for display.

    ``kind``:

    ``"tpow"``
        Multiply by ``t ** power`` (``power`` 1-2 makes up for spreading).
    ``"agc"``
        Automatic gain control: divide each sample by the RMS amplitude in a
        ``window_ns`` window around it, so every part of the trace is shown at
        a similar strength. Amplitudes are no longer comparable afterwards.
    """
    kind = str(kind or "").lower()
    if kind in ("tpow", "power", "time"):
        t = np.clip(profile.time_ns, profile.dt_ns, None)
        curve = (t / t.max()) ** float(power)
        data = profile.data * curve[:, None]
        return profile.with_data(data, f"Gain: amplitude x t^{float(power):g}")
    if kind == "agc":
        width = max(3, int(round(float(window_ns) / profile.dt_ns)))
        energy = _running_mean(profile.data ** 2, width, axis=0)
        rms = np.sqrt(np.clip(energy, 0.0, None))
        floor = 1e-6 * max(float(np.max(rms)), 1e-30)
        data = profile.data / np.maximum(rms, floor)
        return profile.with_data(data, f"Gain: automatic gain control over {float(window_ns):g} ns")
    raise ValueError(f"Unknown gain {kind!r}; use 'tpow' or 'agc'.")


def migrate(profile: GPRProfile, velocity: float) -> GPRProfile:
    """Constant-velocity F-K (Stolt) migration.

    Collapses diffraction hyperbolas onto the points that made them and moves
    dipping reflectors to where they are (Stolt, 1978): the line is
    transformed to frequency and wavenumber, each frequency is mapped to the
    vertical wavenumber of the exploding-reflector field at velocity ``v/2``,
    and transformed back. Assumes evenly spaced traces (the median spacing is
    used) and time zero at the surface; the samples before time zero are
    left as they are.
    """
    v = float(velocity)
    if not v > 0.0:
        raise ValueError(f"Migration needs a positive velocity; got {velocity}.")
    dx = profile.trace_spacing
    if not dx > 0.0:
        raise ValueError("Migration needs at least two traces at distinct positions.")
    time = profile.time_ns
    first = int(np.argmax(time >= -0.5 * profile.dt_ns))
    section = profile.data[first:]
    nx = section.shape[1]
    image = np.zeros_like(section)
    # A long line is migrated in overlapping blocks of traces, each keeping its
    # middle: one transform of the whole line would need gigabytes. The overlap
    # is wider than the widest hyperbola the record can hold.
    reach = int(np.ceil(time_to_depth(time[-1], v) / dx)) if time[-1] > 0 else 0
    margin = int(min(max(64, reach), 512))
    block = max(4 * margin, 1024)
    start = 0
    while start < nx:
        stop = min(nx, start + block)
        lo, hi = max(0, start - margin), min(nx, stop + margin)
        part = _stolt(section[:, lo:hi], profile.dt_ns, dx, 0.5 * v)
        image[:, start:stop] = part[:, start - lo:stop - lo]
        start = stop
    data = profile.data.copy()
    data[first:] = image
    return profile.with_data(data, f"Migration: F-K (Stolt) at {v:.3g} m/ns")


def _stolt(section: np.ndarray, dt: float, dx: float, ve: float) -> np.ndarray:
    """Stolt migration of ``section`` (time zero in its first row) at velocity ``ve``.

    ``ve`` is the exploding-reflector velocity, half the radar velocity. Both
    axes are padded by half and more, to the next power of two, so the
    transform's wrap-around falls in the padding.
    """
    nt, nx = section.shape
    nt_pad = int(2 ** np.ceil(np.log2(max(1.5 * nt, 8))))
    nx_pad = int(2 ** np.ceil(np.log2(max(1.5 * nx, 8))))
    spectrum = np.fft.fft2(section, s=(nt_pad, nx_pad))
    f = np.fft.fftfreq(nt_pad, d=dt)                     # cycles/ns
    kx = np.fft.fftfreq(nx_pad, d=dx)                    # cycles/m
    order = np.argsort(f)
    f_sorted = f[order]
    migrated = np.zeros_like(spectrum)
    for j, k in enumerate(kx):
        # The output frequency f_tau stands for kz = f_tau / ve; its energy
        # was recorded at f = sign(f_tau) sqrt(f_tau^2 + (ve kx)^2).
        f_in = np.sign(f) * np.sqrt(f ** 2 + (ve * k) ** 2)
        column = spectrum[order, j]
        real = np.interp(f_in, f_sorted, column.real, left=0.0, right=0.0)
        imag = np.interp(f_in, f_sorted, column.imag, left=0.0, right=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(f_in != 0.0, np.abs(f) / np.abs(f_in), 0.0)
        migrated[:, j] = (real + 1j * imag) * scale
    return np.real(np.fft.ifft2(migrated))[:nt, :nx]


def crop_time(profile: GPRProfile, max_time_ns: float) -> GPRProfile:
    """Keep the samples up to ``max_time_ns``."""
    keep = profile.time_ns <= float(max_time_ns) + 0.5 * profile.dt_ns
    if not np.any(keep):
        raise ValueError(f"No sample is earlier than {max_time_ns} ns.")
    last = int(np.max(np.nonzero(keep)[0])) + 1
    return profile.with_data(profile.data[:last], f"Cropped at {float(max_time_ns):g} ns")


def process(profile: GPRProfile, steps: Iterable[Mapping[str, Any]]) -> GPRProfile:
    """Run the processing ``steps`` on ``profile``, in :data:`STEP_ORDER`.

    Each step is a mapping with ``name`` (one of :data:`STEP_ORDER`) and that
    step's keyword arguments; a step with ``enabled`` False is skipped. The
    order is the standard one whatever order the list is in, so a recipe
    cannot gain the traces before removing their background.

    Examples
    --------
    >>> recipe = [{"name": "gain", "kind": "agc", "window_ns": 20},
    ...           {"name": "dewow"}, {"name": "time_zero"}]
    >>> [s["name"] for s in sorted_steps(recipe)]
    ['dewow', 'time_zero', 'gain']
    """
    out = profile
    for step in sorted_steps(steps):
        kwargs = {k: v for k, v in step.items() if k not in ("name", "enabled")}
        name = step["name"]
        if name == "dewow":
            out = dewow(out, **kwargs)
        elif name == "time_zero":
            out = set_time_zero(out, **kwargs)
        elif name == "bandpass":
            out = bandpass(out, **kwargs)
        elif name == "background":
            out = remove_background(out, **kwargs)
        elif name == "migrate":
            out = migrate(out, **kwargs)
        elif name == "gain":
            out = gain(out, **kwargs)
    return out


def sorted_steps(steps: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """The enabled steps of a recipe, checked and put in :data:`STEP_ORDER`."""
    kept = []
    for step in steps or []:
        step = dict(step)
        name = str(step.get("name", ""))
        if name not in STEP_ORDER:
            raise ValueError(f"Unknown GPR processing step {name!r}; "
                             f"known: {', '.join(STEP_ORDER)}.")
        if step.get("enabled", True):
            kept.append(step)
    return sorted(kept, key=lambda s: STEP_ORDER.index(s["name"]))


# ---------------------------------------------------------------------------
# Sections in depth and elevation
# ---------------------------------------------------------------------------

def elevation_section(profile: GPRProfile, velocity: float, *, dz: Optional[float] = None,
                      elevation: Optional[Sequence[float]] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """The profile resampled onto elevation, each trace hung from its ground surface.

    Returns ``(section, z)``: ``(len(z), n_traces)`` amplitudes, NaN above
    the ground and below the record, and the elevations of its rows, top
    first. ``dz`` defaults to the depth one sample spans.
    """
    surface = np.asarray(elevation if elevation is not None else profile.elevation, dtype=float)
    if surface.size != profile.n_traces or not np.all(np.isfinite(surface)):
        raise ValueError("An elevation for every trace is needed for a section in elevation.")
    depth = time_to_depth(profile.time_ns, velocity)
    step = float(dz) if dz else max(time_to_depth(profile.dt_ns, velocity), 1e-3)
    top, bottom = float(np.max(surface) - min(depth[0], 0.0)), float(np.min(surface) - depth[-1])
    z = np.arange(top, bottom - step, -step)
    section = np.full((z.size, profile.n_traces), np.nan)
    for j in range(profile.n_traces):
        trace_z = surface[j] - depth                    # descending
        section[:, j] = np.interp(-z, -trace_z, profile.data[:, j], left=np.nan, right=np.nan)
    return section, z


# ---------------------------------------------------------------------------
# Synthetic radargrams and the example line
# ---------------------------------------------------------------------------

def ricker(time_ns: Any, frequency_mhz: float) -> np.ndarray:
    """A Ricker wavelet of peak frequency ``frequency_mhz`` centred on time 0."""
    t = np.asarray(time_ns, dtype=float)
    a = (np.pi * float(frequency_mhz) * 1e-3 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def synthetic_profile(positions: Sequence[float], boundary_depths: Any, permittivity: Any, *,
                      frequency_mhz: float = 100.0, dt_ns: float = 0.2,
                      time_window_ns: float = 250.0, attenuation_db_per_m: float = 1.0,
                      delay_ns: float = 12.0, wow: float = 0.3, ringing: float = 0.05,
                      noise: float = 2e-4, elevation: Optional[Sequence[float]] = None,
                      seed: Optional[int] = 0, source: str = "synthetic") -> GPRProfile:
    """A common-offset radargram over a layered ground, by 1-D convolution per trace.

    Parameters
    ----------
    positions : sequence of float
        ``(n_traces,)`` distance along the line, m.
    boundary_depths : array_like
        ``(n_layers + 1, n_traces)`` depth below ground of each layer's top
        (the first row is the surface, 0) and of the last layer's base, m.
    permittivity : array_like
        ``(n_layers, n_traces)`` relative permittivity of each layer.
    frequency_mhz, dt_ns, time_window_ns : float
        The antenna's peak frequency and the recording.
    attenuation_db_per_m : float
        Loss along the path, dB per metre of travel.
    delay_ns, wow, ringing, noise : float
        What a real system adds and processing takes away: the time before
        the direct wave the record starts, a slow drift (relative to the
        direct wave), the antenna's ringing (a band repeated on every trace)
        and random noise.
    elevation : sequence of float, optional
        Ground elevation under each trace, kept on the profile.

    Returns
    -------
    GPRProfile
        Each layer boundary reflects with the normal-incidence coefficient
        ``(n1 - n2) / (n1 + n2)``, ``n = sqrt(eps)``, reduced by transmission
        through the boundaries above, spreading (``1 / t``) and attenuation.
    """
    x = np.asarray(positions, dtype=float)
    bounds = np.asarray(boundary_depths, dtype=float)
    eps = np.clip(np.asarray(permittivity, dtype=float), 1.0, None)
    n_layers = eps.shape[0]
    if bounds.shape != (n_layers + 1, x.size) or eps.shape[1] != x.size:
        raise ValueError(f"boundary_depths must be ({n_layers + 1}, {x.size}) and permittivity "
                         f"({n_layers}, {x.size}); got {bounds.shape} and {eps.shape}.")
    n_samples = int(round(float(time_window_ns) / float(dt_ns)))
    time = np.arange(n_samples) * float(dt_ns)
    record = time - float(delay_ns)                      # time after the direct wave
    period = 1000.0 / float(frequency_mhz)
    rng = np.random.default_rng(seed)
    data = np.zeros((n_samples, x.size))
    index = np.sqrt(eps)
    thickness = np.clip(np.diff(bounds, axis=0), 0.0, None)
    one_way = np.cumsum(thickness * index / C0, axis=0)  # ns to the base of each layer
    path = np.cumsum(thickness, axis=0)                  # m of ground to the base
    loss = 10.0 ** (-float(attenuation_db_per_m) * 2.0 * path / 20.0)
    for j in range(x.size):
        trace = ricker(record, frequency_mhz)            # direct air/ground wave
        transmitted = 1.0
        for k in range(n_layers - 1):
            n1, n2 = index[k, j], index[k + 1, j]
            r = (n1 - n2) / (n1 + n2)
            t_k = 2.0 * one_way[k, j]
            if thickness[k, j] <= 0.0 or t_k >= record[-1]:
                transmitted *= 1.0 - r * r
                continue
            spread = period / max(t_k, period)
            trace += transmitted * r * spread * loss[k, j] * ricker(record - t_k, frequency_mhz)
            transmitted *= 1.0 - r * r
        data[:, j] = trace
    after = np.clip(record, 0.0, None)
    data += float(wow) * np.where(record >= 0.0, np.exp(-after / (8.0 * period)), 0.0)[:, None]
    ring = float(ringing) * np.where(record > period,
                                     np.sin(2 * np.pi * after / (1.3 * period))
                                     * np.exp(-after / (6.0 * period)), 0.0)
    data += ring[:, None]
    data += float(noise) * rng.standard_normal(data.shape)
    return GPRProfile(data=data, dt_ns=dt_ns, positions=x, frequency_mhz=frequency_mhz,
                      elevation=None if elevation is None else np.asarray(elevation, dtype=float),
                      source=source, instrument="Synthetic",
                      metadata={"simulated": True, "delay_ns": float(delay_ns)})


#: The line Ex_ERT_workflow takes through the example catchment, as [col, row],
#: and the water-content snapshot it uses.
EXAMPLE_LINE = ((115.0, 70.0), (95.0, 180.0))
EXAMPLE_SNAPSHOT = 5


def example_profile(data_dir: PathLike, *, trace_spacing: float = 0.25,
                    frequency_mhz: float = 100.0, snapshot: int = EXAMPLE_SNAPSHOT) -> GPRProfile:
    """A simulated 100 MHz line over the MODFLOW catchment model in ``examples/data``.

    The model's water content (time step ``snapshot``) and porosity are
    sampled along the Ex_ERT_workflow line, turned into permittivity by CRIM
    and drawn as a radargram by :func:`synthetic_profile`; the ground surface
    is kept as each trace's elevation.
    """
    from PyHydroGeophysX.core.hydro_profile import sample_profile

    folder = Path(data_dir)
    water = np.load(folder / "Watercontent.npy")
    if water.ndim == 4:
        water = water[int(snapshot)]
    porosity = np.load(folder / "Porosity.npy")
    top = np.loadtxt(folder / "top.txt") if (folder / "top.txt").is_file() \
        else np.load(folder / "top.npy")
    bot = np.load(folder / "bot.npy")
    (c1, r1), (c2, r2) = EXAMPLE_LINE
    length = float(np.hypot(c2 - c1, r2 - r1))
    n_points = max(10, int(round(length / float(trace_spacing))) + 1)
    sampled = sample_profile(water, porosity, top, bot, (c1, r1), (c2, r2), num_points=n_points)
    structure = np.asarray(sampled["structure"], dtype=float)
    surface = structure[0]
    depths = surface[None, :] - structure
    eps = crim_permittivity(sampled["porosity_profile"], sampled["water_content_profile"])
    profile = synthetic_profile(sampled["L_profile"], depths, eps, frequency_mhz=frequency_mhz,
                                elevation=surface,
                                source=f"MODFLOW example model (water content step {snapshot})")
    profile.metadata.update({"permittivity": eps, "boundary_depths": depths})
    return profile


# ---------------------------------------------------------------------------
# Reading and saving
# ---------------------------------------------------------------------------

#: GSSI antenna model numbers and their centre frequencies, MHz.
_GSSI_ANTENNAS = {
    "3207": 100, "5106": 200, "50270": 270, "50300": 300, "350": 350, "5103": 400,
    "50400": 400, "3102": 500, "3101": 900, "51600": 1600, "4105": 2000, "42000": 2000,
    "62000": 2000, "62300": 2300, "52600": 2600,
}

#: Spacing given to the traces of a line recorded by time, until one is set, m.
UNKNOWN_SPACING = 0.05


def _frequency_from_name(name: str) -> float:
    """The centre frequency an antenna name gives, MHz; NaN when it gives none.

    ``"400MHz"`` and ``"1.6 GHz"`` say it; a GSSI model number (``"5103A"``,
    ``"62000-003"``) is looked up.

    Examples
    --------
    >>> _frequency_from_name("5103A"), _frequency_from_name("250 MHz Shielded")
    (400.0, 250.0)
    """
    text = str(name or "").strip()
    found = re.search(r"(\d+(?:\.\d+)?)\s*(MHz|GHz)", text, re.IGNORECASE)
    if found:
        value = float(found.group(1))
        return value * 1000.0 if found.group(2).lower() == "ghz" else value
    model = re.match(r"[A-Za-z]?(\d+)", text)
    if model:
        digits = model.group(1)
        for length in range(len(digits), 2, -1):
            if digits[:length] in _GSSI_ANTENNAS:
                return float(_GSSI_ANTENNAS[digits[:length]])
    return float("nan")


def _gssi_date(packed: int) -> str:
    """A DZT header date: seconds/2, minutes, hours, day, month, years since 1980 in bits."""
    if not packed:
        return ""
    sec, minute, hour = 2 * (packed & 31), (packed >> 5) & 63, (packed >> 11) & 31
    day, month, year = (packed >> 16) & 31, (packed >> 21) & 15, 1980 + (packed >> 25)
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return ""
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{sec:02d}"


def read_dzt(path: PathLike, channel: int = 0) -> GPRProfile:
    """Read a GSSI ``.DZT`` line (one channel of it).

    The header is 1024 bytes a channel, little-endian; the samples follow scan
    by scan, each scan holding every channel's samples in turn. 8- and 16-bit
    samples are unsigned with the middle of their range as zero, which is
    subtracted; 32-bit samples are signed. The first two samples of every
    scan carry the scan number and the user marks, not radar data, and are
    replaced by the third. The time window divided by the samples per scan is
    the sample interval; a line recorded by distance has its scans per metre,
    one recorded by time has none and its traces are spaced evenly at
    :data:`UNKNOWN_SPACING` until a spacing is set.
    """
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < 1024:
        raise ValueError(f"{p.name} is too short to be a GSSI DZT file.")
    tag, rh_data, nsamp, bits = struct.unpack_from("<4H", raw, 0)
    zero = struct.unpack_from("<h", raw, 8)[0]
    sps, spm, mpm, position, window = struct.unpack_from("<5f", raw, 10)
    created = struct.unpack_from("<I", raw, 32)[0]
    nchan = struct.unpack_from("<H", raw, 52)[0] or 1
    epsr, top_m, depth_m = struct.unpack_from("<3f", raw, 54)
    antenna = raw[98:112].split(b"\0")[0].decode("ascii", "replace").strip()
    if nsamp < 3 or bits not in (8, 16, 32) or not (window > 0 and np.isfinite(window)):
        raise ValueError(f"{p.name}: not a GSSI DZT header (samples {nsamp}, {bits} bits, "
                         f"range {window} ns).")
    if not 0 <= int(channel) < nchan:
        raise ValueError(f"{p.name} has {nchan} channel(s); channel {channel} asked for.")
    header = 1024 * rh_data if rh_data < 1024 else 1024 * nchan
    dtype = {8: np.uint8, 16: np.uint16, 32: np.int32}[bits]
    width = bits // 8
    scan = nsamp * nchan
    payload = raw[header:]
    n_traces = len(payload) // (scan * width)
    if n_traces < 1:
        raise ValueError(f"{p.name} holds no complete scan after its header.")
    warnings: List[str] = []
    if len(payload) % (scan * width):
        warnings.append(f"{p.name}: the last {len(payload) % (scan * width)} bytes are not a "
                        "whole scan and were left out.")
    values = np.frombuffer(payload, dtype=dtype, count=n_traces * scan)
    data = values.reshape(n_traces, nchan, nsamp)[:, int(channel), :].T.astype(float)
    if bits in (8, 16):
        data -= 2.0 ** (bits - 1)
    data[:2] = data[2]
    if spm > 0:
        positions = np.arange(n_traces) / float(spm)
    else:
        positions = np.arange(n_traces) * UNKNOWN_SPACING
        warnings.append(f"{p.name} was recorded by time ({sps:g} scans a second), not by "
                        f"distance: its traces are spaced {UNKNOWN_SPACING} m apart until a "
                        "spacing is set.")
    return GPRProfile(
        data=data, dt_ns=float(window) / nsamp, positions=positions,
        frequency_mhz=_frequency_from_name(antenna), source=str(p), instrument="GSSI DZT",
        metadata={"antenna": antenna, "channels": int(nchan), "channel": int(channel),
                  "bits": int(bits), "scans_per_second": float(sps),
                  "scans_per_metre": float(spm), "metres_per_mark": float(mpm),
                  "position_ns": float(position), "range_ns": float(window),
                  "header_zero": int(zero), "permittivity": float(epsr),
                  "depth_m": float(depth_m), "top_m": float(top_m), "tag": int(tag),
                  "created": _gssi_date(created), "warnings": warnings})


def _key_values(text: str, separator: str) -> Dict[str, str]:
    """``KEY <separator> value`` lines, keys upper-cased; split at the first separator."""
    values: Dict[str, str] = {}
    for line in text.splitlines():
        if separator in line:
            key, value = line.split(separator, 1)
            values[key.strip().upper()] = value.strip()
    return values


def _number(values: Mapping[str, str], *keys: str, default: float = float("nan")) -> float:
    for key in keys:
        found = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", values.get(key, ""))
        if found:
            return float(found.group(0))
    return float(default)


def _sibling(path: Path, suffixes: Sequence[str]) -> Optional[Path]:
    """The file beside ``path`` with the same stem and one of ``suffixes`` (any case)."""
    for suffix in suffixes:
        for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
            if candidate.is_file():
                return candidate
    for candidate in path.parent.glob(path.stem + ".*"):
        if candidate.suffix.lower() in suffixes:
            return candidate
    return None


def read_dt1(path: PathLike) -> GPRProfile:
    """Read a Sensors & Software pulseEKKO ``.DT1`` line with its ``.HD`` header.

    The ``.HD`` is text, ``KEY = value``; the ``.DT1`` holds the traces one
    after another, each a 128-byte header (25 float32 words and a comment)
    and then its samples - 16-bit integers, or 32-bit floats when the header
    says four bytes a point. Each trace's own position and topography are
    used; the sample interval is the time window over the points per trace,
    and the time axis starts so that ``TIMEZERO AT POINT`` (counted from 1)
    is time zero.
    """
    p = Path(path)
    hd = p if p.suffix.lower() == ".hd" else _sibling(p, [".hd"])
    dt1 = p if p.suffix.lower() == ".dt1" else _sibling(p, [".dt1"])
    if hd is None or dt1 is None:
        raise ValueError(f"{p.name}: a pulseEKKO line needs both its .HD and .DT1 files, "
                         "side by side with the same name.")
    header = _key_values(hd.read_text(encoding="latin-1", errors="replace"), "=")
    npts = int(_number(header, "NUMBER OF PTS/TRC", default=0))
    window = _number(header, "TOTAL TIME WINDOW")
    if npts < 2 or not window > 0:
        raise ValueError(f"{hd.name}: no NUMBER OF PTS/TRC or TOTAL TIME WINDOW in the header.")
    raw = dt1.read_bytes()
    if len(raw) < 128:
        raise ValueError(f"{dt1.name} is too short to hold a trace.")
    first = struct.unpack_from("<25f", raw, 0)
    width = 4 if int(round(first[5])) == 4 else 2
    if int(round(first[2])) not in (0, npts):
        raise ValueError(f"{dt1.name}: its traces have {first[2]:g} points, the header "
                         f"{npts}.")
    record = 128 + width * npts
    n_traces = len(raw) // record
    dtype = np.dtype("<f4") if width == 4 else np.dtype("<i2")
    traces = np.frombuffer(raw[:n_traces * record], dtype=np.uint8).reshape(n_traces, record)
    heads = traces[:, :100].copy().view("<f4").reshape(n_traces, 25)
    data = traces[:, 128:].copy().view(dtype).reshape(n_traces, npts).T.astype(float)
    positions = heads[:, 1].astype(float)
    units = header.get("POSITION UNITS", "m").lower()
    scale = 0.3048 if units.startswith("f") else 1.0
    positions = positions * scale
    if n_traces > 1 and np.allclose(positions, positions[0]):
        step = _number(header, "STEP SIZE USED", default=UNKNOWN_SPACING) * scale
        positions = positions[0] + np.arange(n_traces) * step
    topography = heads[:, 3].astype(float) * scale
    elevation = topography if np.ptp(topography) > 0 else None
    dt = window / npts
    zero_point = _number(header, "TIMEZERO AT POINT", default=1.0)
    warnings = []
    if len(raw) % record:
        warnings.append(f"{dt1.name}: {len(raw) % record} bytes after the last whole trace "
                        "were left out.")
    return GPRProfile(
        data=data, dt_ns=dt, positions=positions, t_start_ns=-(zero_point - 1.0) * dt,
        frequency_mhz=_number(header, "NOMINAL FREQUENCY"),
        antenna_separation=_number(header, "ANTENNA SEPARATION") * scale,
        elevation=elevation, source=str(dt1), instrument="pulseEKKO DT1",
        metadata={"header": dict(header), "bytes_per_point": width, "warnings": warnings})


def read_mala(path: PathLike) -> GPRProfile:
    """Read a MALA ``.rd3`` (16-bit) or ``.rd7`` (32-bit) line with its ``.rad`` header.

    The ``.rad`` is text, ``KEY:value``; ``FREQUENCY`` there is the sampling
    frequency in MHz, so the sample interval is ``1000 / FREQUENCY`` ns. The
    data file has no header: the traces follow one another, little-endian.
    """
    p = Path(path)
    rad = p if p.suffix.lower() == ".rad" else _sibling(p, [".rad"])
    if rad is None:
        raise ValueError(f"{p.name}: a MALA line needs its .rad header beside it.")
    data_file = p if p.suffix.lower() in (".rd3", ".rd7") else _sibling(p, [".rd3", ".rd7"])
    if data_file is None:
        raise ValueError(f"{p.name}: no .rd3 or .rd7 data file beside the .rad header.")
    header = _key_values(rad.read_text(encoding="latin-1", errors="replace"), ":")
    nsamp = int(_number(header, "SAMPLES", default=0))
    sampling = _number(header, "FREQUENCY")
    if nsamp < 2:
        raise ValueError(f"{rad.name}: no SAMPLES in the header.")
    if sampling > 0:
        dt = 1000.0 / sampling
    else:
        window = _number(header, "TIMEWINDOW")
        if not window > 0:
            raise ValueError(f"{rad.name}: neither FREQUENCY nor TIMEWINDOW in the header.")
        dt = window / nsamp
    dtype = np.dtype("<i4") if data_file.suffix.lower() == ".rd7" else np.dtype("<i2")
    raw = data_file.read_bytes()
    n_traces = len(raw) // (nsamp * dtype.itemsize)
    if n_traces < 1:
        raise ValueError(f"{data_file.name} holds no complete trace.")
    data = np.frombuffer(raw, dtype=dtype, count=n_traces * nsamp).reshape(
        n_traces, nsamp).T.astype(float)
    step = _number(header, "DISTANCE INTERVAL", default=0.0)
    warnings: List[str] = []
    if not step > 0:
        start = _number(header, "START POSITION", default=0.0)
        stop = _number(header, "STOP POSITION", default=float("nan"))
        if np.isfinite(stop) and stop != start and n_traces > 1:
            step = abs(stop - start) / (n_traces - 1)
        else:
            step = UNKNOWN_SPACING
            warnings.append(f"{rad.name} gives no distance between traces: they are spaced "
                            f"{UNKNOWN_SPACING} m apart until a spacing is set.")
    return GPRProfile(
        data=data, dt_ns=dt, positions=np.arange(n_traces) * step,
        frequency_mhz=_frequency_from_name(header.get("ANTENNAS", "")),
        antenna_separation=_number(header, "ANTENNA SEPARATION"),
        source=str(data_file), instrument=f"MALA {data_file.suffix.lstrip('.').upper()}",
        metadata={"header": dict(header), "warnings": warnings})


def read_segy_gpr(path: PathLike, dt_ns: Optional[float] = None,
                  trace_spacing: Optional[float] = None) -> GPRProfile:
    """Read a GPR line exported as SEG-Y.

    SEG-Y keeps the sample interval in whole microseconds, so GPR software
    writes it in picoseconds; that is how the header's value is read unless
    ``dt_ns`` is given. Traces are spaced by their receiver coordinates when
    those change along the line, else by ``trace_spacing``.
    """
    from PyHydroGeophysX.data_processing.seismic import read_segy

    dataset = read_segy(str(path))
    data = np.asarray(dataset.traces, dtype=float)
    warnings: List[str] = []
    if dt_ns is None:
        header_value = float(dataset.metadata.sample_interval_us)
        dt_ns = header_value / 1000.0
        warnings.append(f"{Path(path).name}: the header's sample interval ({header_value:g}) "
                        f"was read as picoseconds, {dt_ns:g} ns; check the time window.")
    x = np.array([h.receiver_x for h in dataset.headers], dtype=float)
    y = np.array([h.receiver_y for h in dataset.headers], dtype=float)
    if x.size == data.shape[1] and (np.ptp(x) > 0 or np.ptp(y) > 0):
        positions = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    else:
        step = float(trace_spacing) if trace_spacing else UNKNOWN_SPACING
        positions = np.arange(data.shape[1]) * step
        if not trace_spacing:
            warnings.append(f"{Path(path).name} has no trace coordinates: traces are spaced "
                            f"{UNKNOWN_SPACING} m apart until a spacing is set.")
    return GPRProfile(data=data, dt_ns=dt_ns, positions=positions, source=str(path),
                      instrument="SEG-Y", metadata={"warnings": warnings})


def read_gpr(path: PathLike, **options: Any) -> GPRProfile:
    """Read a GPR line, choosing the reader by the file's suffix."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".dzt":
        return read_dzt(p, **options)
    if suffix in (".dt1", ".hd"):
        return read_dt1(p, **options)
    if suffix in (".rd3", ".rd7", ".rad"):
        return read_mala(p, **options)
    if suffix in (".sgy", ".segy"):
        return read_segy_gpr(p, **options)
    if suffix == ".npz":
        return load_profile(p)
    raise ValueError(f"{p.name}: not a GPR format this reader knows "
                     "(.DZT, .DT1/.HD, .rd3/.rd7/.rad, .sgy, .npz).")


def save_profile(profile: GPRProfile, path: PathLike) -> Path:
    """Write ``profile`` to ``.npz``; :func:`load_profile` reads it back."""
    out = Path(path).with_suffix(".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in profile.metadata.items()
            if isinstance(v, (str, int, float, bool)) or v is None}
    np.savez(out, data=profile.data, dt_ns=profile.dt_ns, positions=profile.positions,
             t_start_ns=profile.t_start_ns, frequency_mhz=profile.frequency_mhz,
             antenna_separation=profile.antenna_separation,
             elevation=(profile.elevation if profile.elevation is not None
                        else np.array([], dtype=float)),
             info=json.dumps({"source": profile.source, "instrument": profile.instrument,
                              "history": profile.history, "metadata": meta}))
    return out


def load_profile(path: PathLike) -> GPRProfile:
    """Read a profile written by :func:`save_profile`."""
    with np.load(Path(path), allow_pickle=False) as z:
        info = json.loads(str(z["info"])) if "info" in z else {}
        elevation = np.asarray(z["elevation"], dtype=float) if "elevation" in z else np.array([])
        return GPRProfile(
            data=z["data"], dt_ns=float(z["dt_ns"]), positions=z["positions"],
            t_start_ns=float(z["t_start_ns"]), frequency_mhz=float(z["frequency_mhz"]),
            antenna_separation=float(z["antenna_separation"]),
            elevation=elevation if elevation.size else None,
            source=str(info.get("source") or Path(path).name),
            instrument=str(info.get("instrument") or "PyHydroGeophysX"),
            history=list(info.get("history") or []), metadata=dict(info.get("metadata") or {}))


__all__ = [
    "C0", "FILE_FILTER", "GPRProfile", "GPR_ONLY_SUFFIXES", "STEP_ORDER", "bandpass",
    "crim_permittivity",
    "crop_time", "depth_to_time", "dewow", "elevation_section", "example_profile", "gain",
    "hyperbola_time", "load_profile", "migrate", "permittivity_from_velocity",
    "pick_time_zero", "process", "read_dt1", "read_dzt", "read_gpr", "read_mala",
    "read_segy_gpr", "remove_background", "ricker", "save_profile", "set_time_zero",
    "sorted_steps", "synthetic_profile", "time_to_depth", "velocity_from_permittivity",
]
