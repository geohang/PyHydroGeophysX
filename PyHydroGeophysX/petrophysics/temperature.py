"""Temperature correction of electrical resistivity.

Bulk resistivity falls by roughly 2 % per degree of warming, because the mobility
of the ions carrying the current rises with temperature. Over a monitoring season
that is the same order as the change moisture produces, so an uncorrected
time-lapse section shows the ground "drying" through autumn simply because it is
cooling. Correcting every survey to one reference temperature is what leaves the
remaining change attributable to water.

Two standard laws are implemented:

``hayley``
    :math:`\\rho_{ref} = \\rho_T (mT + c) / (mT_{ref} + c)` with
    :math:`m = 0.0183`, :math:`c = 0.4470` (Hayley et al., 2007, GRL), fitted on
    soils and pore fluids between 0 and 25 degC. About 2.0 %/degC near 25 degC.

``linear``
    :math:`\\rho_{ref} = \\rho_T (1 + \\alpha (T - T_{ref}))`, the Campbell/Arps
    form, with :math:`\\alpha = 0.025` per degC by default. About 2.5 %/degC.

Both leave the model untouched at ``T == reference`` and raise the resistivity of a
survey acquired warmer than the reference, which is the direction that matters when
reading a section: without it, a summer survey looks wetter than it is.

The temperature itself comes from one of four sources - a constant, a measured
depth profile, a 1-D heat-conduction simulation driven by a measured surface
record, or the analytical damped sine of the seasonal wave - built by
:func:`temperature_field` into the ``(cells, times)`` array the correction consumes.
The conduction mode is the one most sites can actually use: a logger at the ground
surface, or an air-temperature record from a nearby station, is usually all there
is, and the ground's own diffusion supplies the depth structure. A record of one
temperature per survey belongs there too - applied uniformly over depth it would
correct the deep ground by what happened at the surface, which it never felt.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MODELS",
    "REFERENCE_TEMPERATURE_C",
    "DEFAULT_TEMPERATURE_SPEC",
    "DEFAULT_DIFFUSIVITY",
    "correction_factor",
    "to_reference",
    "from_reference",
    "sensitivity_percent_per_degree",
    "seasonal_temperature",
    "profile_temperature",
    "simulate_temperature_1d",
    "annual_damping_depth",
    "load_surface_temperature",
    "temperature_field",
    "correct_time_lapse_models",
]

#: Reference temperature laboratory and literature values are quoted at.
REFERENCE_TEMPERATURE_C = 25.0

#: Hayley et al. (2007) fit coefficients.
_HAYLEY_M = 0.0183
_HAYLEY_C = 0.4470

#: Campbell (1948) / Arps (1953) linear coefficient, per degC.
_LINEAR_ALPHA = 0.025

MODELS = ("hayley", "linear")

_MODEL_ALIASES = {
    "hayley": "hayley", "hayley2007": "hayley",
    "linear": "linear", "campbell": "linear", "arps": "linear",
}

#: Period of the seasonal ground-temperature wave, in days.
_YEAR_DAYS = 365.25

#: Thermal diffusivity of the ground, m2/day. Soils run about 0.04-0.13 (roughly
#: 0.5-1.5e-6 m2/s); 0.06 puts the annual damping depth near 2.6 m, which is
#: typical of a moist mineral soil.
DEFAULT_DIFFUSIVITY = 0.06

DEFAULT_TEMPERATURE_SPEC: Dict[str, Any] = {
    "enabled": False,
    "model": "hayley",
    "alpha": _LINEAR_ALPHA,           # 'linear' only
    "reference": REFERENCE_TEMPERATURE_C,
    "mode": "constant",               # constant | profile | surface | seasonal
    "value": 15.0,                    # 'constant'
    "profile": None,                  # 'profile': [[depth_m, temperature_C], ...]
    # 'surface': 1-D conduction driven by a measured surface record. The usual
    # case in the field: one logger at the surface, and the ground's own diffusion
    # supplies the depth structure.
    "surface_times": None,            # dates or days, matching surface_temperature
    "surface_temperature": None,      # degC at the ground surface
    "diffusivity": DEFAULT_DIFFUSIVITY,   # m2/day
    "bottom_depth": None,             # insulated base; default is 3 damping depths
    "spin_up_cycles": 2,
    # 'seasonal': damped sine of the annual surface wave
    "mean_temperature": 12.0,
    "amplitude": 10.0,
    "damping_depth": 2.5,
    "peak_day": 200.0,                # day of year of the warmest surface day
    "start_day": None,                # day of year of the first survey, when the
}                                     # surveys carry elapsed days but no dates


def _resolve_model(name: Any) -> str:
    key = str(name or "hayley").strip().lower().replace(" ", "").replace("_", "")
    if key not in _MODEL_ALIASES:
        raise ValueError(
            f"unknown temperature model {name!r}; expected one of {MODELS}")
    return _MODEL_ALIASES[key]


def correction_factor(temperature: Any,
                      reference: float = REFERENCE_TEMPERATURE_C,
                      model: str = "hayley",
                      alpha: float = _LINEAR_ALPHA) -> np.ndarray:
    """Multiplier taking a resistivity measured at ``temperature`` to ``reference``.

    Args:
        temperature: in-situ temperature, degC. Scalar or array.
        reference: temperature the corrected model is reported at, degC.
        model: ``'hayley'`` or ``'linear'``.
        alpha: fractional resistivity change per degC, ``'linear'`` only.

    Returns:
        Factor ``f`` with ``rho_reference = rho_measured * f``. Greater than one
        where the ground was warmer than the reference.
    """
    temp = np.asarray(temperature, dtype=float)
    ref = float(reference)
    kind = _resolve_model(model)
    if kind == "hayley":
        denominator = _HAYLEY_M * ref + _HAYLEY_C
        if denominator <= 0.0:
            raise ValueError(
                f"reference temperature {ref} degC is outside the Hayley fit")
        factor = (_HAYLEY_M * temp + _HAYLEY_C) / denominator
    else:
        factor = 1.0 + float(alpha) * (temp - ref)
    # A factor at or below zero is unphysical and would flip the sign of the
    # model; it only happens far outside the range either law was fitted over.
    return np.clip(factor, 1.0e-3, 1.0e3)


def to_reference(resistivity: Any, temperature: Any,
                 reference: float = REFERENCE_TEMPERATURE_C,
                 model: str = "hayley",
                 alpha: float = _LINEAR_ALPHA) -> np.ndarray:
    """Resistivity measured at ``temperature``, reported at ``reference``."""
    return np.asarray(resistivity, dtype=float) * correction_factor(
        temperature, reference=reference, model=model, alpha=alpha)


def from_reference(resistivity: Any, temperature: Any,
                   reference: float = REFERENCE_TEMPERATURE_C,
                   model: str = "hayley",
                   alpha: float = _LINEAR_ALPHA) -> np.ndarray:
    """Inverse of :func:`to_reference`: predict the reading at ``temperature``.

    Used when a petrophysical relation calibrated at the reference temperature has
    to be compared against a field survey in its own conditions.
    """
    return np.asarray(resistivity, dtype=float) / correction_factor(
        temperature, reference=reference, model=model, alpha=alpha)


def sensitivity_percent_per_degree(temperature: float = REFERENCE_TEMPERATURE_C,
                                   model: str = "hayley",
                                   alpha: float = _LINEAR_ALPHA) -> float:
    """Percent change in resistivity per degC at ``temperature``.

    The number to quote when deciding whether a correction is worth applying: a
    2 % per degC ground against a 5 % resistivity change says the temperature
    swing has to be known to about a degree before the change means moisture.
    """
    kind = _resolve_model(model)
    if kind == "hayley":
        return 100.0 * _HAYLEY_M / (_HAYLEY_M * float(temperature) + _HAYLEY_C)
    return 100.0 * float(alpha)


def seasonal_temperature(depth: Any, day_of_year: Any,
                         mean_temperature: float = 12.0,
                         amplitude: float = 10.0,
                         damping_depth: float = 2.5,
                         peak_day: float = 200.0) -> np.ndarray:
    """Analytical ground temperature of the annual surface wave.

    The classic solution of 1-D heat conduction with a sinusoidal surface forcing:
    the annual swing decays as ``exp(-z/d)`` and lags by ``z/d`` radians, so below
    a few damping depths the ground sits at the annual mean and needs no
    correction at all.

    Args:
        depth: depth below ground, m (positive downward). Broadcast against
            ``day_of_year``.
        day_of_year: day of the year of the survey (1-365), fractional days fine.
        mean_temperature: annual mean ground temperature, degC.
        amplitude: half the peak-to-peak annual swing at the surface, degC.
        damping_depth: e-folding depth of the annual wave, m. Typically 2-3 m for
            soils; larger for dry, less conductive ground.
        peak_day: day of year the surface is warmest (about 200 in the northern
            hemisphere, about 20 in the southern).

    Returns:
        Temperature in degC, broadcast over ``depth`` and ``day_of_year``.
    """
    z = np.asarray(depth, dtype=float)
    t = np.asarray(day_of_year, dtype=float)
    d = float(damping_depth)
    if d <= 0.0:
        raise ValueError("damping_depth must be positive")
    phase = 2.0 * math.pi * (t - (float(peak_day) - _YEAR_DAYS / 4.0)) / _YEAR_DAYS
    return float(mean_temperature) + float(amplitude) * np.exp(-z / d) * np.sin(phase - z / d)


def profile_temperature(depth: Any, profile: Sequence[Sequence[float]]) -> np.ndarray:
    """Interpolate a measured ``(depth, temperature)`` profile onto ``depth``.

    Outside the measured interval the nearest measured value is held, rather than
    extrapolated: a borehole thermistor string says nothing about the ground below
    its deepest sensor, and a linear extrapolation there invents a trend.
    """
    table = np.asarray(profile, dtype=float)
    if table.ndim != 2 or table.shape[1] < 2 or table.shape[0] < 1:
        raise ValueError("profile must be an (n, 2) array of depth, temperature")
    order = np.argsort(table[:, 0])
    return np.interp(np.asarray(depth, dtype=float), table[order, 0], table[order, 1])


def annual_damping_depth(diffusivity: float = None,
                         period_days: float = _YEAR_DAYS) -> float:
    """Depth at which the annual temperature swing falls to 1/e of the surface.

    ``sqrt(2 * alpha / omega)``, the natural length scale of a periodic surface
    forcing diffusing into the ground. Reported alongside a simulation so the
    thermal diffusivity - which nobody has an intuition for - can be checked
    against a depth, which everybody does.
    """
    alpha = float(DEFAULT_DIFFUSIVITY if diffusivity is None else diffusivity)
    if alpha <= 0.0:
        raise ValueError("diffusivity must be positive")
    omega = 2.0 * math.pi / float(period_days)
    return math.sqrt(2.0 * alpha / omega)


def simulate_temperature_1d(surface_times: Sequence[float],
                            surface_temperature: Sequence[float],
                            depths: Any,
                            output_times: Optional[Sequence[float]] = None,
                            *,
                            diffusivity: float = None,
                            bottom_depth: Optional[float] = None,
                            n_layers: int = 120,
                            max_step: float = 1.0,
                            spin_up_cycles: int = 2,
                            initial_temperature: Optional[float] = None
                            ) -> np.ndarray:
    """Ground temperature with depth and time, from a surface record.

    Solves 1-D heat conduction, ``dT/dt = alpha d2T/dz2``, driven by the measured
    surface temperature and insulated at the base. This is the mode most monitoring
    sites can actually use: a logger at the surface, or an air-temperature record
    from a nearby station, is usually all there is, and the analytical seasonal
    wave needs an amplitude and a phase that a real year does not obey.

    The ground's own physics does the rest: the diffusion damps and delays the
    surface signal, so a cold snap that matters at 0.3 m has all but vanished at
    3 m. That is exactly the depth structure a time-lapse correction needs and the
    part a single bulk temperature per survey cannot represent.

    Args:
        surface_times: times of the surface record, in days, increasing.
        surface_temperature: surface temperature at those times, degC.
        depths: depths below ground to report, m (positive downward).
        output_times: times to report, in days on the same clock as
            ``surface_times``. Defaults to the surface record's own times.
        diffusivity: thermal diffusivity in m2/day. Soils are roughly 0.04-0.13;
            the default is 0.06, an annual damping depth of about 2.6 m.
        bottom_depth: depth of the insulated base. Defaults to three annual
            damping depths below the deepest requested depth, far enough that the
            boundary does not reach back into the answer.
        n_layers: number of depth intervals in the solution grid.
        max_step: largest internal time step, in days.
        spin_up_cycles: how many times the record is replayed before the reported
            pass, to let the deep ground forget the initial guess. The deep
            profile lags the surface by months, so a run started from a uniform
            temperature is wrong at depth for as long as it takes to diffuse.
        initial_temperature: starting temperature everywhere. Defaults to the mean
            of the surface record, which is where the deep ground sits.

    Returns:
        ``(len(depths), len(output_times))`` array in degC.
    """
    from scipy.linalg import solve_banded

    times = np.asarray(surface_times, dtype=float).ravel()
    surface = np.asarray(surface_temperature, dtype=float).ravel()
    if times.size != surface.size or times.size < 2:
        raise ValueError("the surface record needs at least two (time, temperature) pairs")
    order = np.argsort(times)
    times, surface = times[order], surface[order]
    if not np.isfinite(surface).all():
        raise ValueError("the surface temperature record contains non-finite values")

    z_out = np.asarray(depths, dtype=float).ravel()
    sample_times = (np.asarray(output_times, dtype=float).ravel()
                    if output_times is not None else times)
    alpha = float(DEFAULT_DIFFUSIVITY if diffusivity is None else diffusivity)
    if alpha <= 0.0:
        raise ValueError("diffusivity must be positive")

    damping = annual_damping_depth(alpha)
    base = float(bottom_depth if bottom_depth is not None
                 else max(float(np.nanmax(z_out)) if z_out.size else 0.0, damping)
                 + 3.0 * damping)
    n_layers = max(int(n_layers), 8)
    dz = base / n_layers
    grid = np.linspace(0.0, base, n_layers + 1)

    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError("the surface record must span a positive time interval")
    n_steps = max(int(np.ceil(span / max(float(max_step), 1.0e-6))), 1)
    dt = span / n_steps
    r = alpha * dt / (dz * dz)

    # Backward Euler: unconditionally stable, so the step is chosen for accuracy
    # rather than to keep the solution from exploding.
    n = grid.size
    ab = np.zeros((3, n))
    ab[0, 1:] = -r                     # upper diagonal
    ab[1, :] = 1.0 + 2.0 * r           # main diagonal
    ab[2, :-1] = -r                    # lower diagonal
    ab[1, 0], ab[0, 1] = 1.0, 0.0      # node 0 is the prescribed surface
    ab[2, n - 2] = -2.0 * r            # insulated base, by mirroring

    profile = np.full(n, float(initial_temperature if initial_temperature is not None
                               else np.mean(surface)))

    def march(record: bool) -> Optional[np.ndarray]:
        nonlocal profile
        history = np.empty((n_steps + 1, n)) if record else None
        if record:
            history[0] = profile
        for step in range(1, n_steps + 1):
            rhs = profile.copy()
            rhs[0] = float(np.interp(times[0] + step * dt, times, surface))
            profile = solve_banded((1, 1), ab, rhs)
            if record:
                history[step] = profile
        return history

    for _ in range(max(int(spin_up_cycles), 0)):
        march(record=False)
    history = march(record=True)

    step_times = times[0] + dt * np.arange(n_steps + 1)
    # Interpolate in time first (the grid of stored profiles), then in depth.
    out = np.empty((z_out.size, sample_times.size))
    for column, when in enumerate(sample_times):
        at_time = np.array([np.interp(when, step_times, history[:, node])
                            for node in range(n)])
        out[:, column] = np.interp(z_out, grid, at_time)
    return out


def load_surface_temperature(path: Any) -> Tuple[list, list, bool]:
    """Read a two-column surface temperature record from a text file.

    The first column is a date (any layout
    :mod:`PyHydroGeophysX.data_processing.survey_timing` reads) or a number of
    days; the second is the temperature in degC. A header line is skipped when it
    does not parse.

    Returns ``(times, temperatures, dated)``, where ``times`` are datetimes when
    the file carried dates and floats (days) when it did not.
    """
    from PyHydroGeophysX.data_processing.survey_timing import parse_timestamp

    times: list = []
    values: list = []
    dated = None
    with open(str(path), "r", errors="ignore") as handle:
        for line in handle:
            row = [token for token in line.replace(",", " ").replace(";", " ").split()
                   if token]
            if len(row) < 2:
                continue
            try:
                temperature = float(row[-1])
            except ValueError:
                continue                      # a header, or a comment
            stamp = parse_timestamp(" ".join(row[:-1]))
            if stamp is not None:
                when: Any = stamp[0]
                is_dated = True
            else:
                try:
                    when = float(row[0])
                except ValueError:
                    continue
                is_dated = False
            if dated is None:
                dated = is_dated
            elif dated != is_dated:
                raise ValueError(
                    "the surface temperature record mixes dates and plain numbers "
                    "in its first column")
            times.append(when)
            values.append(temperature)
    if len(times) < 2:
        raise ValueError(
            "could not read a surface temperature record: expected two columns, "
            "a date or a number of days and a temperature in degC")
    return times, values, bool(dated)


def _to_days(values: Sequence[Any]) -> Tuple[np.ndarray, Optional[_dt.datetime]]:
    """Times as days from the first entry, plus the datetime that entry stands for."""
    parsed = [_coerce_time(value) for value in values]
    if any(item is None for item in parsed):
        raise ValueError("a time in the record could not be read")
    if isinstance(parsed[0], _dt.datetime):
        origin = min(parsed)
        return (np.asarray([(item - origin).total_seconds() / 86400.0
                            for item in parsed], dtype=float), origin)
    return np.asarray(parsed, dtype=float), None


def _coerce_time(value: Any):
    """A datetime or a float from whatever a caller stored in a spec."""
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, _dt.date):
        return _dt.datetime(value.year, value.month, value.day)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return _dt.datetime.fromisoformat(value.strip())
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _survey_days_on(origin: Optional[_dt.datetime], n_times: int,
                    days: Optional[Sequence[float]],
                    dates: Optional[Sequence[Any]]) -> np.ndarray:
    """Survey times on the surface record's own clock.

    A dated surface record and dated surveys are placed on one absolute timeline;
    two plain numeric series are assumed to share an origin. Mixing the two is
    refused rather than guessed, because an offset of a few months in the forcing
    is a sign error in the correction, not a small one.
    """
    if origin is not None:
        if dates is None or len(dates) != n_times:
            raise ValueError(
                "the surface temperature record carries dates, so the surveys need "
                "acquisition dates too; name the survey files with their date or "
                "enter the times in the interface")
        parsed = [_coerce_time(value) for value in dates]
        if any(not isinstance(item, _dt.datetime) for item in parsed):
            raise ValueError("a survey acquisition date could not be read")
        return np.asarray([(item - origin).total_seconds() / 86400.0
                           for item in parsed], dtype=float)
    if days is None or len(days) != n_times:
        raise ValueError(
            "the surface temperature record is in plain days, so the surveys need "
            "numeric measurement times on the same clock")
    return np.asarray(days, dtype=float)


def _days_of_year(n_times: int, days: Optional[Sequence[float]],
                  dates: Optional[Sequence[Any]],
                  start_day: Optional[float]) -> Optional[np.ndarray]:
    """Day-of-year per survey, from acquisition dates or elapsed days."""
    if dates is not None and len(dates) == n_times and all(d is not None for d in dates):
        out = []
        for value in dates:
            when = value
            if isinstance(when, str):
                try:
                    when = _dt.datetime.fromisoformat(when)
                except ValueError:
                    return None
            if not isinstance(when, (_dt.datetime, _dt.date)):
                return None
            if not isinstance(when, _dt.datetime):
                when = _dt.datetime(when.year, when.month, when.day)
            day = when.timetuple().tm_yday
            fraction = (when.hour * 3600 + when.minute * 60 + when.second) / 86400.0
            out.append(day + fraction)
        return np.asarray(out, dtype=float)
    if days is not None and len(days) == n_times and start_day is not None:
        return np.mod(np.asarray(days, dtype=float) + float(start_day) - 1.0,
                      _YEAR_DAYS) + 1.0
    return None


def temperature_field(spec: Dict[str, Any], depths: Any, n_times: int,
                      days: Optional[Sequence[float]] = None,
                      dates: Optional[Sequence[Any]] = None) -> Tuple[np.ndarray, str]:
    """Build the ``(cells, times)`` temperature array a correction needs.

    Args:
        spec: options following :data:`DEFAULT_TEMPERATURE_SPEC`.
        depths: depth below ground of every cell, m.
        n_times: number of surveys.
        days: elapsed days of each survey, when known.
        dates: acquisition datetimes of each survey, when known. Required by the
            seasonal mode unless ``spec['start_day']`` says where in the year the
            sequence begins - a damped annual wave is meaningless without a
            position in the year.

    Returns:
        ``(temperature, description)``: the field in degC and a one-line summary
        naming the source, for the log and the run report.

    Raises:
        ValueError: if the requested mode cannot be built from what was supplied.
            The caller is expected to surface that rather than silently skip the
            correction the user asked for.
    """
    options = {**DEFAULT_TEMPERATURE_SPEC, **(spec or {})}
    z = np.asarray(depths, dtype=float).ravel()
    n_cells = z.size
    mode = str(options.get("mode", "constant")).strip().lower()

    if mode == "constant":
        value = float(options.get("value", 15.0))
        field = np.full((n_cells, n_times), value)
        return field, f"constant {value:g} degC"

    if mode == "profile":
        profile = options.get("profile")
        if profile is None or len(profile) == 0:
            raise ValueError("temperature mode 'profile' needs a depth/temperature table")
        column = profile_temperature(z, profile)
        field = np.tile(column.reshape(n_cells, 1), (1, n_times))
        return field, (f"measured profile, {column.min():.1f} to {column.max():.1f} "
                       f"degC over depth (held constant in time)")

    if mode in ("surface", "surface_series", "conduction"):
        record_times = options.get("surface_times")
        record_values = options.get("surface_temperature")
        if (not record_times or not record_values
                or len(record_times) != len(record_values) or len(record_times) < 2):
            raise ValueError(
                "temperature mode 'surface' needs a surface temperature record: "
                "matching lists of times and temperatures, at least two long")
        surface_days, origin = _to_days(record_times)
        survey_days = _survey_days_on(origin, n_times, days, dates)
        diffusivity = float(options.get("diffusivity", DEFAULT_DIFFUSIVITY))
        field = simulate_temperature_1d(
            surface_days, record_values, z, survey_days,
            diffusivity=diffusivity,
            bottom_depth=options.get("bottom_depth"),
            spin_up_cycles=int(options.get("spin_up_cycles", 2)),
        )
        return field, (
            f"1-D conduction from a surface record of {len(record_times)} points "
            f"(diffusivity {diffusivity:g} m2/day, annual damping depth "
            f"{annual_damping_depth(diffusivity):.1f} m), {field.min():.1f} to "
            f"{field.max():.1f} degC over the section")

    if mode == "seasonal":
        doy = _days_of_year(n_times, days, dates, options.get("start_day"))
        if doy is None:
            raise ValueError(
                "temperature mode 'seasonal' needs acquisition dates, or "
                "'start_day' (day of year of the first survey) alongside the "
                "elapsed times")
        field = seasonal_temperature(
            z.reshape(n_cells, 1), np.asarray(doy, dtype=float).reshape(1, n_times),
            mean_temperature=float(options.get("mean_temperature", 12.0)),
            amplitude=float(options.get("amplitude", 10.0)),
            damping_depth=float(options.get("damping_depth", 2.5)),
            peak_day=float(options.get("peak_day", 200.0)),
        )
        return field, (
            f"seasonal wave (mean {float(options.get('mean_temperature', 12.0)):g} "
            f"degC, amplitude {float(options.get('amplitude', 10.0)):g} degC, "
            f"damping depth {float(options.get('damping_depth', 2.5)):g} m), "
            f"{field.min():.1f} to {field.max():.1f} degC over the section")

    if mode == "series":
        # Deliberately not supported. One temperature per survey applied over the
        # whole depth corrects a 10 m cell by what happened at the surface, and
        # the deep ground does not follow the surface at all - so it invents a
        # change at depth that was never there. That same record belongs in
        # 'surface', where the conduction damps it with depth.
        raise ValueError(
            "temperature mode 'series' has been removed: one temperature per "
            "survey applied uniformly over depth over-corrects the deep part of "
            "the section. Pass the same values as a surface record with "
            "mode='surface' and let the 1-D conduction supply the depth structure")
    raise ValueError(
        f"unknown temperature mode {mode!r}; expected constant, profile, surface "
        f"or seasonal")


def correct_time_lapse_models(models: Any, spec: Dict[str, Any], depths: Any,
                              days: Optional[Sequence[float]] = None,
                              dates: Optional[Sequence[Any]] = None
                              ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct a ``(cells, times)`` resistivity series to one reference temperature.

    Returns the corrected models and a report dict describing what was applied -
    the model, the reference, the temperature source and the size of the
    correction - so the run can state it rather than leave the reader to wonder
    which temperature the section is at.
    """
    rho = np.asarray(models, dtype=float)
    if rho.ndim != 2:
        raise ValueError("models must be a (cells, times) array")
    n_cells, n_times = rho.shape
    z = np.asarray(depths, dtype=float).ravel()
    if z.size != n_cells:
        raise ValueError(
            f"depths has {z.size} entries but the models have {n_cells} cells")

    options = {**DEFAULT_TEMPERATURE_SPEC, **(spec or {})}
    model = _resolve_model(options.get("model"))
    reference = float(options.get("reference", REFERENCE_TEMPERATURE_C))
    alpha = float(options.get("alpha", _LINEAR_ALPHA))

    temperature, source = temperature_field(options, z, n_times, days=days, dates=dates)
    factor = correction_factor(temperature, reference=reference, model=model, alpha=alpha)
    corrected = rho * factor

    change = 100.0 * (factor - 1.0)
    report = {
        "applied": True,
        "model": model,
        "reference_temperature_C": reference,
        "alpha": alpha if model == "linear" else None,
        "mode": str(options.get("mode", "constant")).strip().lower(),
        "temperature_source": source,
        "temperature_range_C": [float(np.nanmin(temperature)), float(np.nanmax(temperature))],
        "factor_range": [float(np.nanmin(factor)), float(np.nanmax(factor))],
        "median_change_percent": float(np.nanmedian(change)),
        "max_abs_change_percent": float(np.nanmax(np.abs(change))),
        "sensitivity_percent_per_degree": sensitivity_percent_per_degree(
            float(np.nanmean(temperature)), model=model, alpha=alpha),
        "note": (
            f"{model} correction to {reference:g} degC from {source}; resistivity "
            f"scaled by {float(np.nanmin(factor)):.3f} to {float(np.nanmax(factor)):.3f} "
            f"(up to {float(np.nanmax(np.abs(change))):.1f} %)"),
    }
    return corrected, report
