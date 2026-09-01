"""1D electromagnetic sounding and line-geometry readers."""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from PyHydroGeophysX.data_processing import run_inputs, table_io
from PyHydroGeophysX.data_processing.ttem import is_ttem_source, load_ttem_sounding

TEMCOMPANY_MOMENTS = ("LM+HM", "HM", "LM")

#: The uniform relative error already folded into a stored station-stack STD.
#:
#: The per-gate error in these files is not the bare scatter of the repeat
#: transients. It is that scatter combined in quadrature with a uniform term
#: covering everything the repeats cannot see, calibration and the departure of
#: the ground from a layered earth among it, and the same term stands in for the
#: scatter on a gate stacked only once.
#:
#: The acquisition protocol records the term as ``UniStd``, and
#: :func:`_temcompany_uniform_error` prefers that value; this is the fallback for
#: a project whose protocol file did not travel with it. Two further lines of
#: evidence put it at 0.03. One survey's protocol states exactly that, and across
#: 25,485 enabled gates in three projects not one stored error falls below
#: 0.030000, the smallest sits exactly there, and the distribution piles up just
#: above it. A quadrature sum cannot go below either of its parts, so a hard
#: floor at 0.03 with nothing beneath it is what a folded-in 3% looks like, and a
#: bare scatter, which is free to be arbitrarily small, is not.
#:
#: :func:`PyHydroGeophysX.inversion.em1d._tdem_uncertainty` subtracts this from
#: the uniform error the inversion asks for, so the two do not stack.
TEMCOMPANY_UNIFORM_ERROR = 0.03


def _temcompany_uniform_error(protocol: Optional[Mapping[str, Any]]) -> float:
    """The uniform term folded into this survey's stored errors.

    Taken from the protocol's ``UniStd`` where the file travelled with the
    project, since a survey is free to have been acquired with another value.
    A protocol that is missing, or that states something outside (0, 1), falls
    back to :data:`TEMCOMPANY_UNIFORM_ERROR`; a term of 1.0 or more is not an
    error budget and a term of 0.0 would mean the reader had confirmed there is
    nothing folded in, which a missing key does not confirm.
    """
    value = (protocol or {}).get("uniform_std")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return TEMCOMPANY_UNIFORM_ERROR
    if not (0.0 < value < 1.0):
        return TEMCOMPANY_UNIFORM_ERROR
    return value

#: What :func:`save_sounding_container` writes and :func:`load_sounding` reads
#: back, so a recorded run carries the soundings it imported rather than the
#: acquisition folder they were parsed out of.
SOUNDING_CONTAINER_KIND = "em_soundings"


def is_temcompany_source(path: str) -> bool:
    """Return whether *path* looks like a TEMcompany/TEM2Go export.

    Both complete project directories and the self-describing ``*.xyz`` exports
    written by TEMImage are accepted.
    """
    source = Path(path)
    if source.is_dir():
        names = {item.name.lower() for item in source.iterdir() if item.is_file()}
        return ("project.db" in names
                or any(name.endswith("_stationdata.xyz") for name in names)
                or any(name.endswith("_rawdata.xyz") for name in names))
    if source.suffix.lower() != ".xyz" or not source.is_file():
        return False
    try:
        with source.open("r", encoding="utf-8-sig", errors="replace") as handle:
            return "temcompany" in "".join(handle.readline() for _ in range(4)).lower()
    except OSError:
        return False


def _temcompany_json_array(value: Any, *, dtype=float) -> np.ndarray:
    """Decode a numeric array stored by TEMcompany in SQLite."""
    if value in (None, ""):
        return np.array([], dtype=dtype)
    parsed = json.loads(value) if isinstance(value, str) else value
    return np.asarray(parsed, dtype=dtype).ravel()


#: How ``max_relative_std`` removes the gates it condemns.
#:
#: ``truncate`` ends the decay at the first bad gate and drops every later one.
#: ``individual`` drops only the gates that fail and keeps the rest, so a late
#: gate that stands on its own merits survives a bad neighbour.
GATE_REJECTION_MODES: Tuple[str, ...] = ("truncate", "individual")


def _check_gate_rejection(mode: str) -> str:
    """Validate and normalise a gate-rejection mode name.

    Called once per survey before the per-station loop, because that loop treats
    a ``ValueError`` from a station as "this station has no usable gates" and
    moves on. A misspelled mode would otherwise surface as an empty survey.
    """
    normalised = str(mode).strip().lower()
    if normalised not in GATE_REJECTION_MODES:
        raise ValueError(
            "gate_rejection must be one of "
            f"{', '.join(GATE_REJECTION_MODES)}; got {mode!r}.")
    return normalised


def _temcompany_valid_channels(
    times: np.ndarray,
    response: np.ndarray,
    relative_std: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    use_flags: bool = True,
    max_relative_std: Optional[float] = None,
    gate_rejection: str = "truncate",
    reject_negative: bool = True,
) -> "tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]":
    """Drop the dummy values a file carries and, unless waived, its unused gates.

    ``use_flags=False`` keeps every gate the file holds a finite, non-dummy value
    for, including the ones its stored flags mark as unused. That is a deliberate
    choice to re-do the gate editing here rather than inherit it: on a noisy
    ground survey a file may leave only a quarter of its gates flagged in use,
    and the inversion's own outlier rejection, which judges a gate by whether a
    model can explain it, can be a better test than one applied before any model
    existed. The gates come back with their recorded stack errors, so a gate that
    was flagged out for being noisy still carries that noise in its weight.

    ``max_relative_std`` condemns a gate that is negative, or whose relative
    stack error exceeds the cut. The stack error is the scatter of the repeat
    transients averaged into that gate divided by their mean, so a value near or
    above 0.3 says the gate's own repeats disagree by as much as the quantity
    being measured. Such a gate constrains nothing and only supplies a direction
    for the model to follow. ``None`` keeps every flagged gate.

    ``reject_negative`` decides whether a negative gate is condemned along with a
    noisy one, and the two tests are separate because they answer different
    questions: one asks whether the gate was measured repeatably, the other
    whether its sign is physical. Tying them together throws away good
    measurements on an offset-loop system. Measured against one project's stored
    inversion inputs, every gate this reader dropped that the reference kept was
    dropped by the sign test and none by the error cut, and those gates carried a
    median relative error of 0.16, well inside any sensible threshold.

    The sign half of the test needs care. An offset-loop configuration genuinely
    reverses sign at early time: while the diffusing current system is still
    inside the transmitter-receiver offset, the vertical dB/dt at the receiver
    carries the opposite sign, and it crosses over once that system has spread
    past the offset. Whether that reversal reaches the gates being inverted is a
    question with a number attached, because the crossing sits near an induction
    number of order one,

        theta * r ~ 1,   theta = sqrt(mu0 / (4 rho t)),

    so a gate at time ``t`` sees the reversal only once the ground is more
    conductive than ``rho ~ mu0 r**2 / (4 t)``. On a 15 m offset that is about
    6 ohm-m at 12 us and about 1 ohm-m at 61 us; sweeping resistivity through
    this operator puts the onset between 10 and 5 ohm-m at the earliest gate,
    which is where the estimate says it should be. Weathered bedrock is two to
    three orders of magnitude above that.

    So before turning the cut off (``None``) on the grounds that the reversal is
    real, put the site's resistivity and the gate time into that expression.
    Where it says the crossing falls far earlier than the first gate, a negative
    gate is not the early reversal: no layered earth the site could plausibly
    have will produce one there, and an inversion handed such a gate can only
    trade the rest of the sounding against a value it cannot reach.

    ``gate_rejection`` decides what the noise cut removes, and the two answers
    pull in opposite directions. It governs the noise test alone. A sign
    reversal always removes its own gate and no other, because the argument for
    truncation is about the decay having reached the noise floor and an
    early-time reversal makes no claim about the gates after it. Coupling the
    two costs whole soundings: the reversals on one ground survey fell between
    12 and 61 us, so on any station where the first gate was reversed the
    truncating rule discarded every gate the sounding had.

    ``truncate`` ends the sounding at the first condemned gate and drops every
    later one. The argument for it is that the transient decays monotonically
    into the noise floor, so once one gate has crossed that floor the later ones
    are below it too; a later gate that still looks clean is then a fluctuation,
    and keeping it invites the inversion to fit noise at the depth that gate
    appears to probe. This is the default, because it cannot keep a gate the
    noise floor has already swallowed.

    ``individual`` drops only the condemned gates and keeps the later ones. The
    argument for it is that diffusion depth grows with time, so the latest usable
    gate is what sets how deep the sounding can see at all, and discarding it for
    a neighbour's fault costs depth no other gate can supply. It also fits the
    case where one gate is spoiled by a local interference spike rather than by
    the decay reaching the noise floor. Measured over one 929-station ground TDEM
    survey the two rules differ little in volume, a mean of 4.68 gates per
    station against 4.46, so the choice is about which gates survive rather than
    how many.
    """
    status, std, _ = _gate_disposition(
        times, response, relative_std, flags, use_flags, max_relative_std,
        gate_rejection, reject_negative)
    mask = status == "kept"
    if not np.any(mask):
        raise ValueError("The selected TEMcompany sounding has no enabled, finite time gates.")
    t = np.asarray(times, dtype=float).ravel()[:status.size]
    d = np.asarray(response, dtype=float).ravel()[:status.size]
    return t[mask], d[mask], (None if std is None else std[mask])


#: Verdicts :func:`_gate_disposition` can return, worst-first after ``kept``.
#: A gate carries exactly one, the first test it fails.
GATE_STATUS = (
    "kept",
    "dummy",             # non-finite, non-positive time, or a fill value
    "flagged out",       # the file's own InUseFlags say the gate is unused
    "noisy",             # relative stack error above the cut
    "after a noisy one",  # truncation carried it out with an earlier gate
    "reversed sign",     # negative, with the sign test on
)


def _gate_disposition(
    times: np.ndarray,
    response: np.ndarray,
    relative_std: Optional[np.ndarray] = None,
    flags: Optional[np.ndarray] = None,
    use_flags: bool = True,
    max_relative_std: Optional[float] = None,
    gate_rejection: str = "truncate",
    reject_negative: bool = True,
) -> "tuple[np.ndarray, Optional[np.ndarray], np.ndarray]":
    """Per-gate verdict of the selection, in file order.

    The selection itself is a mask, which is all an inversion needs. A reader
    looking at a sounding wants the other half of it: which gates the file holds
    that the run will not see, and which test removed each one. Both come from
    here so that the two cannot drift apart, and the tests are applied in the
    order :data:`GATE_STATUS` lists, each gate keeping the first verdict it earns.

    Returns the verdicts, the cleaned relative errors over every gate (``None``
    when the file carries none), and the raw values as read.
    """
    mode = _check_gate_rejection(gate_rejection)
    n = min(np.size(times), np.size(response))
    t = np.asarray(times, dtype=float).ravel()[:n]
    d = np.asarray(response, dtype=float).ravel()[:n]
    status = np.full(n, "kept", dtype=object)
    dummy = ~(np.isfinite(t) & (t > 0.0) & np.isfinite(d) & (np.abs(d) < 9_000.0))
    status[dummy] = "dummy"
    if use_flags and flags is not None and np.size(flags):
        use = np.asarray(flags, dtype=float).ravel()
        off = ~np.pad(use[:n] > 0, (0, max(0, n - use.size)),
                      constant_values=False)[:n]
        status[(status == "kept") & off] = "flagged out"
    std = None
    if relative_std is not None and np.size(relative_std):
        std = np.asarray(relative_std, dtype=float).ravel()
        std = np.pad(std[:n], (0, max(0, n - std.size)), constant_values=np.nan)[:n]
        std[~np.isfinite(std) | (std < 0.0) | (std >= 9_000.0)] = np.nan
    live = status == "kept"
    if max_relative_std is not None and live.any():
        kept = np.flatnonzero(live)                     # gate centres ascend
        errors = std[kept] if std is not None else np.full(kept.size, np.nan)
        noisy = np.isfinite(errors) & (errors > float(max_relative_std))
        reversed_sign = ((d[kept] < 0.0) if reject_negative
                         else np.zeros(kept.size, dtype=bool))
        carried = np.zeros(kept.size, dtype=bool)
        if mode == "truncate" and noisy.any():
            # Truncation is an argument about the noise floor: the decay has
            # reached it, and a later gate that still looks clean is a
            # fluctuation above it rather than signal. So the noise test, and
            # only the noise test, carries the rest of the sounding with it.
            first = int(np.argmax(noisy))
            carried[first:] = ~noisy[first:]
            noisy[first:] = True
        status[kept[noisy & ~carried]] = "noisy"
        status[kept[carried]] = "after a noisy one"
        # Applied last so that a gate failing both tests reads as reversed,
        # which is the more specific of the two and the one worth naming.
        status[kept[reversed_sign]] = "reversed sign"
    return status, std, d


def _temcompany_positions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return cumulative survey distance for map coordinates."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2:
        return np.zeros(x.size, dtype=float)
    steps = np.hypot(np.diff(x), np.diff(y))
    steps[~np.isfinite(steps)] = 0.0
    return np.concatenate([[0.0], np.cumsum(steps)])


def _normalise_temcompany_moment(moment: str) -> str:
    selected = str(moment).upper().replace(" ", "")
    aliases = {"BOTH": "LM+HM", "JOINT": "LM+HM", "HM+LM": "LM+HM"}
    selected = aliases.get(selected, selected)
    if selected not in TEMCOMPANY_MOMENTS:
        raise ValueError(
            f"TEMcompany moment must be one of {TEMCOMPANY_MOMENTS}, got {moment!r}.")
    return selected


def _geometric_layer_thicknesses(
    n_layers: int, first_layer: float, last_depth: float
) -> np.ndarray:
    """Layer thicknesses in geometric progression, from a first layer and a total depth.

    A fixed-layer 1D inversion needs its layers to thicken with depth, because
    the resolving power of a diffusive method falls off the same way: equal
    layers would over-parameterise the top and under-parameterise the bottom.
    Given ``n_layers - 1`` finite layers, a first thickness and the depth to the
    base of the deepest finite layer, the common ratio is fixed by

        first * (ratio**count - 1) / (ratio - 1) = last_depth

    which has no closed form in ``ratio``, so it is bisected. The sum is
    monotone in ``ratio``, so 80 halvings of the bracket reach machine precision.
    """
    count = max(1, int(n_layers) - 1)
    first = max(float(first_layer), 1e-6)
    target = max(float(last_depth), first * count)
    if count == 1:
        return np.asarray([target], dtype=float)
    if np.isclose(target, first * count):
        return np.full(count, first, dtype=float)
    lo, hi = 1.0, 10.0
    for _ in range(80):
        ratio = 0.5 * (lo + hi)
        total = first * (ratio ** count - 1.0) / (ratio - 1.0)
        if total < target:
            lo = ratio
        else:
            hi = ratio
    ratio = 0.5 * (lo + hi)
    return first * ratio ** np.arange(count, dtype=float)


#: Reference resistivity for the default depth scale, in ohm-m.
#:
#: The scale below needs a resistivity before the ground has been inverted for
#: one. Estimating it from the data was tried and dropped: our own operator
#: reproduces the late-time dependence on resistivity well (an exponent of
#: -1.4904 against the theoretical -1.5) but not on time (-3.08 against -2.5),
#: and a single power law leaves 20% residual over the window, which is not
#: solid enough to read a resistivity off one gate. A fixed reference is honest
#: about that, and the calibration below absorbs it.
_DEPTH_SCALE_RESISTIVITY = 100.0

#: How much of the diffusion depth at the latest gate the default model spans.
#:
#: The diffusion depth ``sqrt(2 rho t / mu0)`` is the natural length scale of a
#: transient, but it is not a resolved depth: it keeps growing while the signal
#: sinks into the noise. Measured over three ground TDEM surveys sharing one
#: instrument, the depth of investigation reached 0.246 of the diffusion depth at
#: the 90th percentile, and the median sat between 0.065 and 0.197. So the ratio
#: is not a constant, and a default has to clear the top of that range rather
#: than sit at its middle. 0.6 leaves roughly a factor of two over the deepest
#: value measured.
#:
#: Erring deep is the safe direction, though not free. A model that stops above
#: the depth of investigation cannot express structure the data does constrain,
#: and nothing downstream can recover it. A model that runs past it is reported
#: as unresolved by the DOI curve, at the cost of coarser layers higher up for a
#: given layer count.
_DEPTH_SCALE_FRACTION = 0.6


def suggest_layer_grid(times: np.ndarray, n_layers: int = 20) -> Dict[str, Any]:
    """A layer grid scaled to the gate range, for data that brought no settings.

    What a sounding can see is set by how late it was recorded, so a fixed grid
    is wrong at both ends: too shallow for a survey that recorded to
    milliseconds, and needlessly coarse for one that stopped at tens of
    microseconds. The depth here follows ``sqrt(t)`` through the diffusion depth
    at the latest gate, taken at :data:`_DEPTH_SCALE_RESISTIVITY` and scaled by
    :data:`_DEPTH_SCALE_FRACTION`.

    The first thickness is a hundredth of that depth, floored at 1 m. It is a
    parameterisation choice rather than something the physics fixes: the layers
    grow geometrically to reach the total depth, so tying the first to the last
    keeps the growth ratio, and with it the conditioning of the deep layers,
    roughly constant however deep the model runs.

    This is a starting point for a project that saved no inversion settings. Any
    grid the file does record is read instead; see
    :func:`_temcompany_inversion_defaults`.
    """
    t = np.asarray(times, dtype=float).ravel()
    t = t[np.isfinite(t) & (t > 0.0)]
    if not t.size:
        raise ValueError("A layer grid needs at least one positive gate time.")
    count = max(2, int(n_layers))
    mu0 = 4.0e-7 * math.pi
    diffusion = math.sqrt(2.0 * _DEPTH_SCALE_RESISTIVITY * float(t.max()) / mu0)
    depth = _DEPTH_SCALE_FRACTION * diffusion
    first = max(1.0, depth / 100.0)
    thickness = _geometric_layer_thicknesses(count, first, depth)
    return {
        "n_layers": count,
        "min_thickness": float(thickness[0]),
        "max_thickness": float(thickness[-1]),
        "layer_thicknesses": thickness,
        "last_depth": float(thickness.sum()),
    }


def _temcompany_stored_thicknesses(
    con: sqlite3.Connection, inversion_name: Optional[str], n_layers: int
) -> Optional[np.ndarray]:
    """The layer grid a project's own inversion ran on, if one is stored.

    The settings carry a layer count, a first thickness and a total depth, and a
    grid can be rebuilt from those. Rebuilding means reproducing a rounding rule
    that is not written down anywhere, so it is worth reading the answer where
    the file already holds it: each row of the model table stores the thicknesses
    that row was solved on.

    Comparing the two on one project, the rebuilt grid agrees with the stored one
    to four decimals but not exactly, because the stored values are rounded to
    two and the shortfall is carried by the deepest layer. Reading avoids having
    to know that, and stays right if the rule ever changes.
    """
    if int(n_layers) < 2:
        return None
    try:
        row = None
        if inversion_name:
            row = con.execute(
                "SELECT Thickness FROM InversionModel WHERE InversionName = ? "
                "LIMIT 1", (str(inversion_name),)).fetchone()
        if row is None:
            row = con.execute(
                "SELECT Thickness FROM InversionModel LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None or not row[0]:
        return None
    try:
        stored = np.asarray(json.loads(row[0]), dtype=float).ravel()
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    # One thickness per layer but the last, which a fixed-layer model treats as
    # a halfspace. Anything else is a grid this settings block does not describe.
    if stored.size != int(n_layers) - 1:
        return None
    if not (np.all(np.isfinite(stored)) and np.all(stored > 0.0)):
        return None
    return stored


#: What a saved setting falls back to when its mode reads ``"Auto"``.
#:
#: The project's ``InverseSettings`` block stores a number and, beside it, a
#: mode. When the mode reads ``"Auto"`` the stored number is inert and a default
#: applies instead, so reading only the number reports a constraint the survey
#: never ran under. The gap is not small: one project stores
#: ``LcRefDistance = 10.0`` and ``LcAutoScalePower = 1.0`` while the run used
#: 50.0 and 0.75.
TEMCOMPANY_AUTO_SETTINGS: Dict[str, float] = {
    "LcRefDistance": 50.0,
    "LcAutoScalePower": 0.75,
    "SciMaxDistance": 300.0,
}


def _temcompany_auto_setting(
    settings: Mapping[str, Any], key: str, fallback: float
) -> "tuple[float, Optional[float]]":
    """The value a setting takes, and the one the project stored beside it.

    Returns ``(effective, stored)``. ``stored`` is ``None`` when the mode is not
    ``"Auto"``, because there is then nothing to distinguish: the stored number
    is the effective one. A caller reporting both can show the reader that the
    two differ without having to know the rule.
    """
    try:
        stored = float(settings.get(key, fallback) or fallback)
    except (TypeError, ValueError):
        stored = float(fallback)
    mode = str(settings.get(f"{key}Mode", "") or "").strip().lower()
    if mode == "auto" and key in TEMCOMPANY_AUTO_SETTINGS:
        return TEMCOMPANY_AUTO_SETTINGS[key], stored
    return stored, None


def _temcompany_inversion_defaults(con: sqlite3.Connection) -> Dict[str, Any]:
    """Read the inversion settings saved inside a TEMcompany project."""
    try:
        row = con.execute("SELECT * FROM UserSettingsJson ORDER BY 1 DESC LIMIT 1").fetchone()
        if row is None:
            return {}
        raw = next(
            (value for value in row if isinstance(value, str) and value.lstrip().startswith("{")),
            None,
        )
        settings = json.loads(raw) if raw else {}
        inverse = dict(settings.get("InverseSettings", {}))
        saved = dict(settings.get("SavedInversions", {}))
        last_name = inverse.get("LastInversionName")
        if last_name in saved:
            inverse = {**inverse, **dict(saved[last_name])}
        n_layers = int(inverse.get("Nlayers", 0) or 0)
        first = float(inverse.get("FirstLayer", 0.0) or 0.0)
        last_depth = float(inverse.get("LastDepth", 0.0) or 0.0)
        if n_layers < 2 or first <= 0.0 or last_depth <= 0.0:
            return {}
        thickness = _temcompany_stored_thicknesses(con, last_name, n_layers)
        if thickness is None:
            thickness = _geometric_layer_thicknesses(n_layers, first, last_depth)
        moment = str(inverse.get("InversionMoment", "Both"))
        reference_distance = _temcompany_auto_setting(
            inverse, "LcRefDistance", 10.0)
        auto_scale_power = _temcompany_auto_setting(
            inverse, "LcAutoScalePower", 1.0)
        sci_max_distance = _temcompany_auto_setting(
            inverse, "SciMaxDistance", 150.0)
        return {
            "n_layers": n_layers,
            "min_thickness": float(thickness[0]),
            "max_thickness": float(thickness[-1]),
            "layer_thicknesses": thickness,
            "last_depth": last_depth,
            "starting_resistivity": float(
                inverse.get("ManualStartModelResistivity", 40.0) or 40.0),
            # ``StartModelMode`` reads "Auto" on every project seen so far, in
            # which case the manual resistivity above is a fallback the
            # recorded run never used. Reported so a caller can switch on it
            # rather than silently starting somewhere else.
            "start_model_mode": str(inverse.get("StartModelMode", "") or ""),
            "smoothness": float(inverse.get("VerticalSmoothness", 2.0)),
            "lateral_smoothness": float(inverse.get("LateralSmoothness", 1.3)),
            "tem_moment": _normalise_temcompany_moment(moment),
            "constraint": str(inverse.get("InversionConstraint", "LCI")),
            "norm": str(inverse.get("InversionType", "L2")),
            "reference_distance": reference_distance[0],
            "reference_distance_stored": reference_distance[1],
            "sci_max_distance": sci_max_distance[0],
            "sci_max_distance_stored": sci_max_distance[1],
            # ``LcAutoScale`` switches on the distance fall-off of the lateral
            # constraint, and the exponent sits beside it as a number. Both are
            # about the separation between soundings, so nothing here scales
            # with the layer count. An earlier reading multiplied the lateral
            # weight by sqrt(n_layers) whenever the switch was on, which on a
            # 20-layer model tied neighbours 4.5 times too tightly and returned
            # a section smoother along the line than the data asks for.
            "lateral_weight_scale": 1.0,
            # The penalty between neighbours scales as
            # (reference_distance / separation) ** this. The switch being off
            # means no fall-off, which is a power of zero rather than a
            # different multiplier.
            "lateral_distance_power": (
                auto_scale_power[0]
                if bool(inverse.get("LcAutoScale", False)) else 0.0
            ),
            "lateral_distance_power_stored": auto_scale_power[1],
        }
    except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _response_on_times(data: Dict[str, Any], reference_times: np.ndarray) -> np.ndarray:
    """Place a TDEM response on a reference gate vector, filling absent gates."""
    reference = np.asarray(reference_times, dtype=float).ravel()
    times = np.asarray(data["times"], dtype=float).ravel()
    response = np.asarray(data["response"], dtype=float).ravel()
    if times.size == reference.size and np.allclose(times, reference, rtol=1e-7, atol=0.0):
        return response
    aligned = np.full(reference.size, np.nan, dtype=float)
    for index, value in enumerate(reference):
        matches = np.flatnonzero(np.isclose(times, value, rtol=1e-7, atol=1e-15))
        if matches.size:
            aligned[index] = response[matches[0]]
    return aligned


def _temcompany_protocol(folder: Path) -> Dict[str, Any]:
    """Acquisition settings from the ``.sts`` protocol beside a project.

    Almost everything the forward needs is duplicated in ``project.db``: loop
    geometry, turn-off waveform, gate windows, filter corners, currents. Three
    things are only here.

    ``UniStd`` is the uniform relative uncertainty applied to every gate on top
    of its own stack error, covering what the stack cannot see: system
    calibration, and the error in representing the ground as 1D layers. Reading
    it means the number comes from the protocol that was actually run rather than
    from a default that happens to match this one. ``LM_StackSize`` /
    ``HM_StackSize`` are the transient counts, which an absolute noise model
    would need. ``ChA_WinFunc`` and ``ChA_GateOverlap`` describe the gate taper,
    which the forward now applies: the same description is in the project
    database as ``GateShape`` and ``GateShapePar1``, and it is read from there
    rather than from here so that a project copied without its protocol still
    gets it. See ``GATE_WINDOWS`` in ``forward.tdem_forward``.

    Returns an empty dict when no protocol file is present, which is the normal
    case for a project copied without it.
    """
    try:
        files = sorted(Path(folder).glob("*.sts"))
    except OSError:
        return {}
    if not files:
        return {}
    values: Dict[str, str] = {}
    try:
        for line in files[0].read_text(encoding="utf-8-sig",
                                       errors="replace").splitlines():
            body = line.split(";", 1)[0].strip()
            key, sep, value = body.partition("=")
            if sep and key.strip():
                values[key.strip()] = value.strip()
    except OSError:
        return {}

    def number(name: str) -> Optional[float]:
        try:
            return float(values[name].split(",")[0])
        except (KeyError, ValueError):
            return None

    result: Dict[str, Any] = {"protocol_file": files[0].name}
    for key, name in (("uniform_std", "UniStd"),
                      ("stack_size_lm", "LM_StackSize"),
                      ("stack_size_hm", "HM_StackSize"),
                      ("gate_overlap", "ChA_GateOverlap"),
                      ("gates_per_decade", "ChA_GatePerDecade"),
                      ("gate_window_shape", "ChA_WinFunc"),
                      ("gate_window_par", "ChA_WinFuncPar1"),
                      ("powerline_hz", "PowerLineMonitorFreq")):
        value = number(name)
        if value is not None:
            result[key] = value
    if "AutoSignDetection" in values:
        result["auto_sign_detection"] = values["AutoSignDetection"]
    return result


def _temcompany_number(spec: Dict[str, Any], key: str,
                       default: Optional[float] = None) -> Optional[float]:
    """One finite scalar out of a spec block, or *default*."""
    try:
        value = float(spec[key])
    except (KeyError, TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _temcompany_transmitter(spec: Dict[str, Any], moment: str) -> Dict[str, Any]:
    """Turn-off waveform and gate windows for one moment, as the file records them.

    TEMcompany stores the measured ramp as (time, amplitude) nodes and every
    gate's open and close time. Both matter for a ground system: the first gate
    opens a microsecond or two after the ramp ends, where a step-off stand-in is
    at its worst, and the late gates are wide enough that the centre value is not
    the window average.

    Four further fields describe how the instrument turns a modelled decay into
    a gate value, and all four are read here so the forward can apply them
    rather than approximate them.

    ``{moment}WaveformPeriod`` is the half-period of the bipolar cycle, 400 us
    for the low moment and 800 us for the high one on a TEM2Go. Superposing the
    earlier pulses of the train matters
    only matters once a gate sits an appreciable fraction of a half-period after
    turn-off. On one 929-station survey the latest gate surviving quality
    control was 95.5 us against an 800 us half-period, so the correction was
    below the noise there; on a quiet site holding late gates it is not.

    ``GateShape`` and ``GateShapePar1`` describe the window each gate is
    integrated over: the shape, and its total tapered fraction. The integral
    runs in linear time over the response grid, which is interpolated locally
    with a cubic Hermite rule; see
    :func:`PyHydroGeophysX.forward.tdem_forward._gate_integration_operator`.

    ``{moment}_GateTimeShift`` offsets every gate time. It is zero in every
    project seen so far, and it is read rather than assumed so that a project
    which sets it is not silently modelled at the wrong times.
    """
    times = np.asarray(spec.get(f"{moment}WaveformTime", []), dtype=float).ravel()
    currents = np.asarray(spec.get(f"{moment}WaveformAmplitude", []),
                          dtype=float).ravel()
    usable = times.size >= 2 and times.size == currents.size
    centre = np.asarray(spec.get(f"{moment}_GateCentreTime", []), dtype=float).ravel()
    opens = np.asarray(spec.get(f"{moment}_GateOpenTime", []), dtype=float).ravel()
    closes = np.asarray(spec.get(f"{moment}_GateCloseTime", []), dtype=float).ravel()
    windows = (centre.size and centre.size == opens.size == closes.size)
    period = _temcompany_number(spec, f"{moment}WaveformPeriod")
    return {
        "waveform_times": times if usable else None,
        "waveform_currents": currents if usable else None,
        "waveform_period": period if (period or 0.0) > 0.0 else None,
        "gate_windows": ({"centre": centre, "open": opens, "close": closes}
                         if windows else None),
        "gate_window_shape": _temcompany_number(spec, "GateShape"),
        "gate_window_par": _temcompany_number(spec, "GateShapePar1"),
        "gate_time_shift": _temcompany_number(
            spec, f"{moment}_GateTimeShift", 0.0),
        "analog_lowpass": _temcompany_lowpass(spec),
        # Recorded, never applied. See _temcompany_system for why.
        "data_factor": _temcompany_number(spec, f"{moment}_DataFactor"),
        "target_current": _temcompany_number(spec, f"{moment}_Tx_TargetCurrent"),
    }


def _temcompany_lowpass(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Receiver low-pass corners, as first-order stages.

    The receiver electronics roll off well above the signal band, which sounds
    ignorable and is not. dB/dt falls steeply at early time, and a causal
    low-pass output is a weighted average of the input over the preceding time
    constant, where the signal was larger. Filtering therefore *raises* the
    modelled value at the earliest gates, by more the steeper the decay, and the
    effect dies away once the signal changes little over a time constant.

    On this instrument the corners are a few hundred kilohertz, so the time
    constants are a fraction of a microsecond, and the first gate opens one or
    two microseconds after the ramp. Leaving the filter out costs about 30 % of
    the amplitude at the first low-moment gate and nothing by the late gates; an
    inversion absorbs that tilt by making the near surface conductive, because a
    conductive near surface is what raises the early-time response.

    The field lists corner frequencies for cascaded first-order stages, unlike
    the damped second-order pair a GEX file gives, so it is reported under its
    own key.
    """
    values = np.asarray(spec.get("LPFilter_1order", []), dtype=float).ravel()
    cutoffs = [float(v) for v in values if np.isfinite(v) and v > 0.0]
    return {"first_order_cutoffs_hz": tuple(cutoffs)} if cutoffs else {}


def _temcompany_column(rows, key: str, dtype=float, default=np.nan) -> np.ndarray:
    """One column of ``StationStackData``, or the default where it is absent.

    The schema has grown across TEMcompany releases, and the synthetic exporter
    writes only what its example needs, so a project may carry no Longitude or
    Latitude at all. ``sqlite3.Row`` raises ``IndexError`` for a name it does
    not hold rather than returning None, which turns a missing optional column
    into a failure to open the survey.
    """
    values = []
    for row in rows:
        try:
            value = row[key]
        except (IndexError, KeyError):
            value = None
        values.append(default if value is None else value)
    return np.asarray(values, dtype=dtype)


def _temcompany_row_value(row: Optional[sqlite3.Row], key: str) -> Optional[float]:
    """One finite scalar out of a ``StationStackData`` row, or ``None``."""
    if row is None:
        return None
    try:
        value = row[key]
    except (IndexError, KeyError):
        return None
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _temcompany_system(spec: Dict[str, Any], row: Optional[sqlite3.Row] = None) -> Dict[str, Any]:
    """Map TEMcompany loop/receiver metadata to the studio geometry.

    The transmitter-receiver separation comes from the station rather than from
    the spec wherever the station records one. The spec states a single nominal
    layout, ``RxCoilXYZPos`` at 15 m on a TEM2Go, while the measured column
    spans 11.58 to 17.63 m over one survey, so it is a per-station quantity and
    is read as one. Of the three recorded columns
    only ``RxTxDistance`` is the edited one: ``RxTxDistanceBField`` carries
    zeros where the field estimate did not converge, and
    ``RxTxDistanceGPSBased`` carries GPS scatter.

    ``data_factor`` is recorded and never applied. Measured over 882 stations
    and 4,884 gates of one project, the inversion inputs the project stores
    (``InversionModel.Datasets.InputData``) are identical to the stored
    ``VoltageValues`` with a ratio of exactly 1.000000 and no variance, while
    ``LM_DataFactor`` is 1.05 and ``HM_DataFactor`` is 1.08. The calibration is
    therefore already inside the stored voltages, and applying it here would
    introduce a 5 to 8 percent error that an inversion would absorb into
    resistivity.
    """
    area = float(spec.get("TxLoopArea", 0.0) or 0.0)
    xy = np.asarray(spec.get("TxLoopXYlength", []), dtype=float).ravel()
    if area <= 0.0 and xy.size >= 2:
        area = float(abs(xy[0] * xy[1]))
    radius = math.sqrt(area / math.pi) if area > 0.0 else 10.0
    tx = np.asarray(spec.get("TxLoopXYZPos", [0.0, 0.0, 0.0]), dtype=float).ravel()
    rx = np.asarray(spec.get("RxCoilXYZPos", [0.0, 0.0, 0.0]), dtype=float).ravel()
    nominal = (float(np.linalg.norm(rx[:2] - tx[:2]))
               if tx.size >= 2 and rx.size >= 2 else 0.0)
    measured = _temcompany_row_value(row, "RxTxDistance")
    sep = measured if (measured or 0.0) > 0.0 else nominal
    spec_z = [arr[2] for arr in (tx, rx) if arr.size >= 3 and np.isfinite(arr[2])]
    fallback = float(np.mean(spec_z)) if spec_z else 0.0
    rx_height = _temcompany_row_value(row, "RxCoilHeight")
    tx_height = _temcompany_row_value(row, "TxCoilHeight")
    if rx_height is None:
        rx_height = float(rx[2]) if rx.size >= 3 and np.isfinite(rx[2]) else fallback
    if tx_height is None:
        tx_height = float(tx[2]) if tx.size >= 3 and np.isfinite(tx[2]) else fallback
    corners_x, corners_y = _temcompany_loop_corners(tx, xy)
    return {
        "source_radius": radius,
        "tx_rx_sep": float(sep),
        # ``height`` stays the single number the rest of the pipeline reads.
        # The two below let a caller that cares place the loop and the coil
        # independently; they are equal on every TEM2Go project seen so far.
        "height": float(rx_height),
        "rx_height": float(rx_height),
        "tx_height": float(tx_height),
        "tx_rx_sep_nominal": nominal,
        "tx_rx_sep_bfield": _temcompany_row_value(row, "RxTxDistanceBField"),
        "tx_rx_sep_gps": _temcompany_row_value(row, "RxTxDistanceGPSBased"),
        "orientation": "z",
        "waveform": "step_off",
        "receiver_type": "dbdt",
        "response_sign": -1.0,
        "data_scale": 1.0,
        "auto_scale": False,
        "loop_area": area,
        "loop_turns": int(spec.get("NTurnsTxLoop", 1) or 1),
        "loop_corners_x": corners_x,
        "loop_corners_y": corners_y,
        "rx_coil_area": _temcompany_number(spec, "RxCoilAreaChA"),
        "instrument": str(spec.get("InstrumentType", "") or ""),
        # The export is dB/dt already divided by the transmitter moment
        # (V/A/m^4), so the forward has to model a UNIT moment. Modelling the
        # real moment on top counts it twice: on this instrument that is a
        # factor turns * area = 1.59, and the inversion pays for it by raising
        # every recovered resistivity to bring the amplitude back down.
        "source_moment": 1.0,
    }


def _temcompany_loop_corners(
    centre: np.ndarray, xy_length: np.ndarray
) -> "tuple[Optional[np.ndarray], Optional[np.ndarray]]":
    """Corner coordinates of the rectangular transmitter loop.

    Recorded so a run can state the loop it modelled, and so a segmented-loop
    forward has the geometry ready should one ever be wanted. It is not wanted
    on this instrument: a 0.63 m square at a 15 m offset departs from its
    equivalent dipole by order ``(a / r) ** 2``, about 0.2 percent, which sits
    an order of magnitude below the gate errors.
    """
    xy = np.asarray(xy_length, dtype=float).ravel()
    if xy.size < 2 or not np.all(np.isfinite(xy[:2])) or not np.all(xy[:2] > 0.0):
        return None, None
    origin = np.asarray(centre, dtype=float).ravel()
    x0 = float(origin[0]) if origin.size >= 1 and np.isfinite(origin[0]) else 0.0
    y0 = float(origin[1]) if origin.size >= 2 and np.isfinite(origin[1]) else 0.0
    half_x, half_y = 0.5 * float(xy[0]), 0.5 * float(xy[1])
    return (np.asarray([x0 - half_x, x0 + half_x, x0 + half_x, x0 - half_x]),
            np.asarray([y0 - half_y, y0 - half_y, y0 + half_y, y0 + half_y]))


def _temcompany_forward_metadata(
    spec: Dict[str, Any], system: Dict[str, Any], moments: "tuple[str, ...]",
    protocol: Mapping[str, Any], qc: Mapping[str, Any],
) -> Dict[str, Any]:
    """Everything a forward run needs to be reproducible from the file alone.

    A figure of the modelled response is only checkable if the reader states
    what it modelled. This is that statement: the geometry,
    the waveform, the gate windows, the electronics, the units convention and
    the gate selection that was applied, all as the project records them.

    It is descriptive. Nothing here is consumed by the forward, which reads the
    same fields directly, so a change to one does not silently fail to appear in
    the other.
    """
    def _listed(value: Any) -> List[float]:
        """A JSON-safe list, treating a missing array as an empty one.

        ``value or []`` cannot do this: a numpy array has no truth value.
        """
        if value is None:
            return []
        return np.asarray(value, dtype=float).ravel().tolist()

    per_moment: Dict[str, Any] = {}
    for name in moments:
        transmitter = _temcompany_transmitter(spec, name)
        windows = transmitter.get("gate_windows") or {}
        per_moment[name] = {
            "waveform_times": _listed(transmitter.get("waveform_times")),
            "waveform_currents": _listed(transmitter.get("waveform_currents")),
            "waveform_period": transmitter.get("waveform_period"),
            "gate_open": _listed(windows.get("open")),
            "gate_centre": _listed(windows.get("centre")),
            "gate_close": _listed(windows.get("close")),
            "gate_time_shift": transmitter.get("gate_time_shift"),
            "data_factor": transmitter.get("data_factor"),
            "target_current": transmitter.get("target_current"),
        }
    corners_x = system.get("loop_corners_x")
    corners_y = system.get("loop_corners_y")
    return {
        "instrument": system.get("instrument", ""),
        "gate_window_shape": _temcompany_number(spec, "GateShape"),
        "gate_window_par": _temcompany_number(spec, "GateShapePar1"),
        "analog_lowpass_hz": _listed(spec.get("LPFilter_1order")),
        "rx_coil_lowpass_hz": _listed(spec.get("LowPassFilterRxCoil")),
        "tx_position": _listed(spec.get("TxLoopXYZPos")),
        "rx_position": _listed(spec.get("RxCoilXYZPos")),
        "tx_rx_sep": system.get("tx_rx_sep"),
        "tx_rx_sep_nominal": system.get("tx_rx_sep_nominal"),
        "tx_height": system.get("tx_height"),
        "rx_height": system.get("rx_height"),
        "loop_area": system.get("loop_area"),
        "loop_turns": system.get("loop_turns"),
        "loop_corners_x": None if corners_x is None else _listed(corners_x),
        "loop_corners_y": None if corners_y is None else _listed(corners_y),
        "rx_coil_area": system.get("rx_coil_area"),
        "source_moment": system.get("source_moment"),
        "response_sign": system.get("response_sign"),
        "data_scale": system.get("data_scale"),
        # Recorded so a reader can see it was read and deliberately not applied;
        # see _temcompany_system for the measurement behind that.
        "data_factor_applied": False,
        "moments": per_moment,
        "protocol": dict(protocol),
        "gate_qc": dict(qc),
    }


def _temcompany_raw_lm_quality(row, spec):
    """Unflagged LM diagnostics, not extra observations for the data misfit.

    Keep all raw gates so a saved Qt input container and a direct project read
    can use the same user-selected reference gate for the empirical prior.
    """
    return {
        "times": np.asarray(spec.get("LM_GateCentreTime", []), dtype=float),
        "transmitter": _temcompany_transmitter(spec, "LM"),
        "response": _temcompany_json_array(row["LM_VoltageValues"] if "LM_VoltageValues" in row.keys() else None),
        "relative_std": _temcompany_json_array(row["LM_VoltageValues_STD"] if "LM_VoltageValues_STD" in row.keys() else None),
    }


def _load_temcompany_database(path: Path, sounding: int, moment: str,
                              use_flags: bool = True,
                              max_relative_std: Optional[float] = None,
                              gate_rejection: str = "truncate",
                              reject_negative: bool = False) -> Dict[str, Any]:
    """Load one stacked sounding and project geometry from ``project.db``."""
    gate_rejection = _check_gate_rejection(gate_rejection)
    uri = path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"StationStackData", "RxTxSpecs"}
        if not required.issubset(tables):
            raise ValueError(
                f"{path.name} is not a supported TEMcompany database "
                f"(missing {', '.join(sorted(required - tables))}).")
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in con.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        all_rows = list(con.execute(
            "SELECT * FROM StationStackData "
            "ORDER BY LineNumber, AveragedDataId"))
        value_key = f"{moment}_VoltageValues"
        rows = []
        for candidate in all_rows:
            if candidate[value_key] in (None, "", "[]"):
                continue
            candidate_spec = specs.get(
                candidate["RxTxSpecsId"], next(iter(specs.values()), {}))
            try:
                _temcompany_valid_channels(
                    np.asarray(candidate_spec.get(f"{moment}_GateCentreTime", []), dtype=float),
                    _temcompany_json_array(candidate[value_key]),
                    _temcompany_json_array(candidate[f"{moment}_VoltageValues_STD"]),
                    _temcompany_json_array(candidate[f"{moment}_InUseFlags"]),
                    use_flags, max_relative_std, gate_rejection,
                    reject_negative,
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
            rows.append(candidate)
        if not rows:
            raise ValueError(f"No enabled {moment} station stacks were found in {path.name}.")
        s = max(0, min(int(sounding), len(rows) - 1))
        row = rows[s]
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
        times = np.asarray(spec.get(f"{moment}_GateCentreTime", []), dtype=float)
        response = _temcompany_json_array(row[value_key])
        std = _temcompany_json_array(row[f"{moment}_VoltageValues_STD"])
        flags = _temcompany_json_array(row[f"{moment}_InUseFlags"])
        times, response, std = _temcompany_valid_channels(
            times, response, std, flags, use_flags, max_relative_std,
            gate_rejection, reject_negative)

        x = np.asarray([item["UtmX"] for item in rows], dtype=float)
        y = np.asarray([item["UtmY"] for item in rows], dtype=float)
        elevation = _temcompany_column(rows, "Elevation")
        heights = _temcompany_column(rows, "RxCoilHeight")
        zone = spec.get("UTMZone")
        zone_letter = str(spec.get("UTMZoneLetter", "") or "").strip()
        coordinate_system = (
            f"UTM zone {zone:g}{zone_letter}"
            if isinstance(zone, (int, float)) else
            f"UTM zone {zone}{zone_letter}" if zone not in (None, "") else "UTM"
        )
        protocol = _temcompany_protocol(path.parent)
        system = _temcompany_system(spec, row)
        result: Dict[str, Any] = {
            "times": times,
            "response": response,
            "n_soundings": len(rows),
            "sounding": s,
            "relative_std": std,
            "uniform_error": _temcompany_uniform_error(protocol),
            "raw_lm_quality": _temcompany_raw_lm_quality(row, spec),
            "positions": _temcompany_positions(x, y),
            "x": x,
            "y": y,
            # Kept alongside the UTM pair so a map view can place imagery
            # without a projection library (see visualization.basemap).
            "longitude": _temcompany_column(rows, "Longitude"),
            "latitude": _temcompany_column(rows, "Latitude"),
            "elevation": elevation,
            "heights": heights,
            "line_numbers": np.asarray([item["LineNumber"] for item in rows], dtype=int),
            "station_ids": np.asarray([str(item["StationId"]) for item in rows]),
            "transmitter": _temcompany_transmitter(spec, moment),
            "temcompany": True,
            "tem_moment": moment,
            "source_format": "TEMcompany project database",
            "coordinate_system": coordinate_system,
            "system": system,
            "rx_tx_distances": _temcompany_column(rows, "RxTxDistance"),
            "rx_heights": _temcompany_column(rows, "RxCoilHeight"),
            "tx_heights": _temcompany_column(rows, "TxCoilHeight"),
            "inversion_defaults": (_temcompany_inversion_defaults(con)
                                   or suggest_layer_grid(times)),
            "protocol": protocol,
            "forward_metadata": _temcompany_forward_metadata(
                spec, system, (moment,), protocol,
                {"use_flags": bool(use_flags),
                 "max_relative_std": max_relative_std,
                 "gate_rejection": gate_rejection,
                 "reject_negative": bool(reject_negative),
                 "gates_kept": int(np.size(times))}),
        }
        return result
    finally:
        con.close()


def _load_temcompany_joint_database(path: Path, sounding: int,
                                    use_flags: bool = True,
                                    max_relative_std: Optional[float] = None,
                                    gate_rejection: str = "truncate",
                                    reject_negative: bool = False,
                                    min_gates_per_moment: Optional[
                                        Mapping[str, int]] = None) -> Dict[str, Any]:
    """Load all usable LM/HM gates for one station, preserving common station order.

    ``min_gates_per_moment`` drops a moment that survived gate selection with
    too few gates to say anything, as ``{"LM": 1, "HM": 3}``. The argument for a
    floor is that a moment reduced to one or two gates still costs a forward
    call and still enters the misfit, while two points cannot separate the level
    of a decay from its slope; what it adds is a degree of freedom the fit can
    absorb rather than a constraint on the model. The argument against is that
    the deep moment is the only thing that sees deep, so a station stripped of
    its HM keeps its shallow model and loses its depth.

    Which argument wins depends on the survey, so this is off by default. On one
    929-station survey a ``{"HM": 3}`` floor removed HM from 91 stations, 163
    gates, and left 7 of them with no data at all.
    """
    gate_rejection = _check_gate_rejection(gate_rejection)
    uri = path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in con.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        protocol = _temcompany_protocol(path.parent)
        uniform_error = _temcompany_uniform_error(protocol)
        entries: List["tuple[sqlite3.Row, Dict[str, Dict[str, np.ndarray]]]"] = []
        for row in con.execute(
            "SELECT * FROM StationStackData ORDER BY LineNumber, AveragedDataId"
        ):
            spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
            moments: Dict[str, Dict[str, np.ndarray]] = {}
            for selected in ("LM", "HM"):
                try:
                    times, response, std = _temcompany_valid_channels(
                        np.asarray(spec.get(f"{selected}_GateCentreTime", []), dtype=float),
                        _temcompany_json_array(row[f"{selected}_VoltageValues"]),
                        _temcompany_json_array(row[f"{selected}_VoltageValues_STD"]),
                        _temcompany_json_array(row[f"{selected}_InUseFlags"]),
                        use_flags, max_relative_std, gate_rejection,
                        reject_negative,
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
                if times.size < int((min_gates_per_moment or {}).get(selected, 0)):
                    continue
                moments[selected] = {
                    "times": times,
                    "response": response,
                    "transmitter": _temcompany_transmitter(spec, selected),
                    "relative_std": (
                        np.asarray(std, dtype=float)
                        if std is not None else np.array([], dtype=float)
                    ),
                    # The joint inversion reads its errors from the moment, not
                    # from the station, so the moment has to say what they mean.
                    "uniform_error": uniform_error,
                }
            if moments:
                entries.append((row, moments))
        if not entries:
            raise ValueError(f"No enabled LM/HM station stacks were found in {path.name}.")

        index = max(0, min(int(sounding), len(entries) - 1))
        row, moments = entries[index]
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
        station_system = _temcompany_system(spec, row)
        preview_name = "HM" if "HM" in moments else "LM"
        preview = moments[preview_name]
        rows = [entry[0] for entry in entries]
        x = np.asarray([item["UtmX"] for item in rows], dtype=float)
        y = np.asarray([item["UtmY"] for item in rows], dtype=float)
        zone = spec.get("UTMZone")
        zone_letter = str(spec.get("UTMZoneLetter", "") or "").strip()
        coordinate_system = (
            f"UTM zone {zone:g}{zone_letter}"
            if isinstance(zone, (int, float)) else
            f"UTM zone {zone}{zone_letter}" if zone not in (None, "") else "UTM"
        )
        return {
            "times": preview["times"],
            "response": preview["response"],
            "relative_std": preview["relative_std"],
            "uniform_error": uniform_error,
            "moments": moments,
            "raw_lm_quality": _temcompany_raw_lm_quality(row, spec),
            "available_moments": tuple(moments),
            "n_soundings": len(entries),
            "sounding": index,
            "positions": _temcompany_positions(x, y),
            "x": x,
            "y": y,
            "longitude": _temcompany_column(rows, "Longitude"),
            "latitude": _temcompany_column(rows, "Latitude"),
            "elevation": _temcompany_column(rows, "Elevation"),
            "heights": _temcompany_column(rows, "RxCoilHeight"),
            "line_numbers": np.asarray([item["LineNumber"] for item in rows], dtype=int),
            "station_ids": np.asarray([str(item["StationId"]) for item in rows]),
            "average_data_ids": np.asarray(
                [item["AveragedDataId"] for item in rows], dtype=int),
            "temcompany": True,
            "tem_moment": "LM+HM",
            "source_format": "TEMcompany project database",
            "coordinate_system": coordinate_system,
            "system": station_system,
            "rx_tx_distances": _temcompany_column(rows, "RxTxDistance"),
            "rx_heights": _temcompany_column(rows, "RxCoilHeight"),
            "tx_heights": _temcompany_column(rows, "TxCoilHeight"),
            # The instrument's whole gate set, not this station's surviving
            # subset, because the default grid is a property of the survey.
            "inversion_defaults": (
                _temcompany_inversion_defaults(con)
                or suggest_layer_grid(np.concatenate([
                    np.asarray(spec.get(f"{name}_GateCentreTime", []), dtype=float)
                    for name in ("LM", "HM")
                ] + [preview["times"]]))),
            "protocol": protocol,
            "forward_metadata": _temcompany_forward_metadata(
                spec, station_system, tuple(moments), protocol,
                {"use_flags": bool(use_flags),
                 "max_relative_std": max_relative_std,
                 "gate_rejection": gate_rejection,
                 "reject_negative": bool(reject_negative),
                 "gates_kept": {name: int(np.size(block["times"]))
                                for name, block in moments.items()}}),
        }
    finally:
        con.close()


def gate_report(
    path: str,
    sounding: int = 0,
    *,
    moment: str = "LM+HM",
    use_flags: bool = True,
    max_relative_std: Optional[float] = None,
    gate_rejection: str = "truncate",
    reject_negative: bool = False,
) -> Dict[str, Any]:
    """Every gate one station holds, and what the current selection does with it.

    :func:`load_sounding` returns the gates that survive, which is what an
    inversion needs and less than a reader needs. Working out why a station came
    back with four gates out of twenty-five otherwise means changing a setting
    and watching the count move. This returns the whole list instead, each gate
    carrying its verdict from :func:`_gate_disposition`, so the same question is
    answered by looking.

    ``sounding`` indexes the stations the loader would return under these same
    settings, so a station number here and one there mean the same station. A
    station the settings empty is absent from both.
    """
    source = Path(path)
    if source.is_dir():
        source = source / "project.db"
    if not source.exists():
        raise FileNotFoundError(f"No TEMcompany project database at {source}.")
    selected_moment = _normalise_temcompany_moment(moment)
    wanted = ("LM", "HM") if selected_moment == "LM+HM" else (selected_moment,)
    gate_rejection = _check_gate_rejection(gate_rejection)

    uri = source.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in con.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        kept_rows: List[Any] = []
        for row in con.execute(
            "SELECT * FROM StationStackData ORDER BY LineNumber, AveragedDataId"
        ):
            spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
            if any(_gate_verdicts(row, spec, name, use_flags, max_relative_std,
                                  gate_rejection, reject_negative)[0] is not None
                   for name in wanted):
                kept_rows.append(row)
        if not kept_rows:
            raise ValueError(f"No enabled station stacks were found in {source.name}.")
        index = max(0, min(int(sounding), len(kept_rows) - 1))
        row = kept_rows[index]
        spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))

        moments: Dict[str, Dict[str, Any]] = {}
        for name in wanted:
            report, values = _gate_verdicts(
                row, spec, name, use_flags, max_relative_std, gate_rejection,
                reject_negative)
            if report is None:
                continue
            status, std = report
            moments[name] = {
                "times": np.asarray(spec.get(f"{name}_GateCentreTime", []),
                                    dtype=float)[:status.size],
                "open": np.asarray(spec.get(f"{name}_GateOpenTime", []),
                                   dtype=float)[:status.size],
                "close": np.asarray(spec.get(f"{name}_GateCloseTime", []),
                                    dtype=float)[:status.size],
                "values": values,
                "relative_std": (std if std is not None
                                 else np.full(status.size, np.nan)),
                "flags": _temcompany_json_array(row[f"{name}_InUseFlags"]),
                "status": status,
                "held": int(status.size),
                "kept": int(np.count_nonzero(status == "kept")),
            }
        return {
            "station": str(row["StationId"]),
            "line": int(row["LineNumber"]),
            "sounding": index,
            "n_soundings": len(kept_rows),
            "moments": moments,
            "settings": {
                "moment": selected_moment,
                "use_flags": bool(use_flags),
                "max_relative_std": max_relative_std,
                "gate_rejection": gate_rejection,
                "reject_negative": bool(reject_negative),
            },
        }
    finally:
        con.close()


def _gate_verdicts(row, spec, moment, use_flags, max_relative_std,
                   gate_rejection, reject_negative):
    """Verdicts for one station and moment, or ``(None, None)`` if it holds none.

    A station that carries no record for a moment, or whose record does not
    decode, is not an error here: the survey mixes stations that stacked both
    moments with stations that stacked one.
    """
    column = f"{moment}_VoltageValues"
    if column not in row.keys() or row[column] in (None, "", "[]"):
        return None, None
    try:
        values = _temcompany_json_array(row[column])
        status, std, _ = _gate_disposition(
            np.asarray(spec.get(f"{moment}_GateCentreTime", []), dtype=float),
            values,
            _temcompany_json_array(row[f"{moment}_VoltageValues_STD"]),
            _temcompany_json_array(row[f"{moment}_InUseFlags"]),
            use_flags, max_relative_std, gate_rejection, reject_negative,
        )
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None, None
    if not np.any(status == "kept"):
        return None, None
    return (status, std), np.asarray(values, dtype=float)[:status.size]


def _reference_gate_signal(values, relative_std, times, index):
    """Signal, absolute noise and gate time at one index of a moment's table.

    The absolute noise is the stacked voltage times the relative scatter of the
    transients averaged into it, which is what the file records the second of.
    A value that is missing, a dummy or non-finite gives NaN for all three,
    because a substitute here would read as a measurement.
    """
    values = np.asarray(values, dtype=float).ravel()
    errors = np.asarray(relative_std, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    index = int(index)
    nan = float("nan")
    if not (0 <= index < values.size):
        return nan, nan, nan
    value = float(values[index])
    if not np.isfinite(value) or abs(value) >= 9_000.0:
        return nan, nan, nan
    error = float(errors[index]) if index < errors.size else nan
    if not np.isfinite(error) or error < 0.0 or error >= 9_000.0:
        error = nan
    at = float(times[index]) if index < times.size else nan
    return abs(value), abs(value) * error, at


def survey_summary(
    path: str, *, moment: str = "LM+HM", use_flags: bool = True,
    max_relative_std: Optional[float] = None, gate_rejection: str = "truncate",
    reject_negative: bool = False, reference_gate: int = 2,
) -> Dict[str, Any]:
    """One row per station: where it is, and how much of it survives the QC.

    A line inversion reads every station anyway, so the question "what will this
    quality-control setting actually cost me" has an answer that costs one pass
    over the file rather than a run. That is what this returns: per station, the
    gates the file holds, the gates the given settings keep, and the stack error
    of what is left, alongside the coordinates and per-station geometry.

    ``rows`` is a list of dicts, one per station, in file order. ``totals``
    carries the survey-wide counts so a caller can report the cost of a setting
    without summing the table itself.

    Each row also carries ``{moment}_signal`` and ``{moment}_noise``, the
    stacked voltage and the absolute noise in volts at one gate common to every
    station, along with the ``{moment}_reference_time`` that gate sits at. They
    answer a question the gate counts cannot: whether a station returns fewer
    gates because the ground is resistive, which makes the response genuinely
    smaller, or because something raised the noise floor. The stored relative
    error is the second divided by the first, so it moves either way and cannot
    tell them apart; the two recorded separately can.

    ``reference_gate`` is the index of that gate within the moment's own table,
    counted from the earliest. The default of 2 is an early gate, which is where
    the shallow ground shows, while being late enough to sit clear of the
    turn-off. The gate table is fixed per project, so one index is one physical
    time at every station, which is what makes the column comparable along a
    line. A station whose value there is a dummy records NaN rather than a
    substitute.
    """
    gate_rejection = _check_gate_rejection(gate_rejection)
    selected_moment = _normalise_temcompany_moment(moment)
    wanted = ("LM", "HM") if selected_moment == "LM+HM" else (selected_moment,)
    source = Path(path)
    database = source if source.is_file() else source / "project.db"
    if not database.is_file():
        raise ValueError(f"{source} holds no project.db to summarise.")

    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows: List[Dict[str, Any]] = []
    held = kept = 0
    try:
        specs = {
            item["RxTxSpecsId"]: json.loads(item["RxTxSpecsJson"])
            for item in connection.execute("SELECT * FROM RxTxSpecs")
            if item["RxTxSpecsJson"]
        }
        for row in connection.execute(
            "SELECT * FROM StationStackData ORDER BY LineNumber, AveragedDataId"
        ):
            spec = specs.get(row["RxTxSpecsId"], next(iter(specs.values()), {}))
            entry: Dict[str, Any] = {
                "station": str(row["StationId"]),
                "line": int(row["LineNumber"]),
                "average_data_id": int(row["AveragedDataId"]),
                "x": _temcompany_row_value(row, "UtmX"),
                "y": _temcompany_row_value(row, "UtmY"),
                "longitude": _temcompany_row_value(row, "Longitude"),
                "latitude": _temcompany_row_value(row, "Latitude"),
                "elevation": _temcompany_row_value(row, "Elevation"),
                "rx_tx_distance": _temcompany_row_value(row, "RxTxDistance"),
                "rx_height": _temcompany_row_value(row, "RxCoilHeight"),
                "tx_height": _temcompany_row_value(row, "TxCoilHeight"),
            }
            total_kept = 0
            for name in wanted:
                stored = _temcompany_json_array(row[f"{name}_VoltageValues"])
                entry[f"{name}_gates_held"] = int(stored.size)
                held += int(stored.size)
                gate_times = np.asarray(
                    spec.get(f"{name}_GateCentreTime", []), dtype=float)
                signal, noise, at = _reference_gate_signal(
                    stored, _temcompany_json_array(row[f"{name}_VoltageValues_STD"]),
                    gate_times, reference_gate)
                entry[f"{name}_signal"] = signal
                entry[f"{name}_noise"] = noise
                entry[f"{name}_reference_time"] = at
                try:
                    times, _, std = _temcompany_valid_channels(
                        np.asarray(spec.get(f"{name}_GateCentreTime", []), dtype=float),
                        stored,
                        _temcompany_json_array(row[f"{name}_VoltageValues_STD"]),
                        _temcompany_json_array(row[f"{name}_InUseFlags"]),
                        use_flags, max_relative_std, gate_rejection,
                        reject_negative,
                    )
                except (ValueError, TypeError, json.JSONDecodeError):
                    entry[f"{name}_gates_kept"] = 0
                    entry[f"{name}_median_std"] = float("nan")
                    continue
                errors = np.asarray(std, dtype=float) if std is not None else np.array([])
                errors = errors[np.isfinite(errors)]
                entry[f"{name}_gates_kept"] = int(times.size)
                entry[f"{name}_median_std"] = (float(np.median(errors))
                                               if errors.size else float("nan"))
                total_kept += int(times.size)
                kept += int(times.size)
            entry["gates_kept"] = total_kept
            rows.append(entry)
    finally:
        connection.close()

    usable = sum(1 for item in rows if item["gates_kept"] > 0)
    return {
        "rows": rows,
        "moments": wanted,
        "totals": {
            "stations": len(rows),
            "stations_with_data": usable,
            "stations_emptied": len(rows) - usable,
            "gates_held": held,
            "gates_kept": kept,
        },
        "settings": {
            "moment": selected_moment, "use_flags": bool(use_flags),
            "max_relative_std": max_relative_std,
            "gate_rejection": gate_rejection,
            "reject_negative": bool(reject_negative),
        },
    }


def _temcompany_comment_values(comments: List[str], key: str) -> np.ndarray:
    wanted = key.lower()
    for line in comments:
        body = line.lstrip("/").strip()
        label, separator, values = body.partition(":")
        if separator and label.strip().lower() == wanted:
            return np.fromstring(values.replace(",", " "), sep=" ", dtype=float)
    return np.array([], dtype=float)


def _temcompany_comment_scalar(comments: List[str], key: str, default: float = 0.0) -> float:
    values = _temcompany_comment_values(comments, key)
    return float(values[0]) if values.size else float(default)


def _temcompany_baked_uniform_error(std: Optional[np.ndarray]) -> float:
    """How much uniform error a column of recorded errors already carries.

    The database tables say which product they hold, so a reader of those can
    assert the answer. A text export cannot: the same writer emits both the
    station-stacked errors, which carry the uniform term, and the raw ones,
    which do not. So the column is asked instead.

    A quadrature sum cannot fall below either of its parts, so a folded-in term
    leaves a hard floor at its own size with nothing beneath it, and a bare
    scatter is free to be arbitrarily small. Positive evidence is required, a
    value sitting on the floor as well as none below it, because the two ways of
    being wrong here are not equal: missing a term that is present inflates the
    errors by a few percent, while claiming one that is absent shrinks them and
    invites the inversion to fit noise. A short column of uniformly clean gates
    is therefore left unmarked.
    """
    if std is None:
        return 0.0
    values = np.asarray(std, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0.0) & (values < 9_000.0)]
    if values.size < 8:
        return 0.0
    floor = TEMCOMPANY_UNIFORM_ERROR
    if values.min() < floor * (1.0 - 1e-6):
        return 0.0
    return floor if values.min() <= floor * (1.0 + 1e-3) else 0.0


def _load_temcompany_xyz(path: Path, sounding: int, moment: str) -> Dict[str, Any]:
    """Load TEMcompany station-stacked or raw ``*.xyz`` text exports."""
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    comments = [line.strip() for line in lines if line.lstrip().startswith("/")]
    header: Optional[List[str]] = None
    for line in comments:
        fields = line.lstrip("/").split()
        lowered = {field.lower() for field in fields}
        if "project" in lowered and "date" in lowered and "line" in lowered:
            header = fields
    if not header:
        raise ValueError(f"Could not find the TEMcompany column header in {path.name}.")
    records = [
        line.split() for line in lines
        if line.strip() and not line.lstrip().startswith("/")
    ]
    records = [row for row in records if len(row) >= len(header)]
    if not records:
        raise ValueError(f"No TEMcompany data records were found in {path.name}.")
    columns = {name.lower(): index for index, name in enumerate(header)}
    raw_export = "channel" in columns
    if raw_export:
        channel_index = columns["channel"]
        records = [row for row in records if row[channel_index].upper() == moment]
        gate_names = [name for name in header if name.lower().startswith("gate")]
        std_names: List[str] = []
        station_name = "station-id"
    else:
        gate_names = [name for name in header if name.lower().startswith(moment.lower() + "gate")]
        std_names = [name for name in header if name.lower().startswith(moment.lower() + "std")]
        station_name = "station"
    if not records or not gate_names:
        raise ValueError(f"No {moment} gates were found in {path.name}.")
    gate_names.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
    std_names.sort(key=lambda name: int("".join(filter(str.isdigit, name)) or 0))
    times = _temcompany_comment_values(comments, f"{moment}_GateCentreTime")
    if not times.size:
        raise ValueError(f"{moment} gate centre times are missing from {path.name}.")

    def numeric(name: str, row: List[str], default: float = np.nan) -> float:
        index = columns.get(name.lower())
        try:
            return float(row[index]) if index is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    lat = np.asarray([numeric("latitude", row) for row in records])
    lon = np.asarray([numeric("longitude", row) for row in records])
    elevation = np.asarray([numeric("elevation", row) for row in records])
    # Standalone XYZ exports carry geographic coordinates but no CRS transform.
    # Use a local tangent-plane approximation so plan/section distances remain
    # metric without introducing a pyproj dependency.
    lat0 = float(np.nanmean(lat))
    lon0 = float(np.nanmean(lon))
    earth_radius = 6_371_000.0
    x = np.deg2rad(lon - lon0) * earth_radius * math.cos(math.radians(lat0))
    y = np.deg2rad(lat - lat0) * earth_radius
    s = max(0, min(int(sounding), len(records) - 1))
    row = records[s]
    response = np.asarray([numeric(name, row) for name in gate_names], dtype=float)
    std = (np.asarray([numeric(name, row) for name in std_names], dtype=float)
           if std_names else None)
    times, response, std = _temcompany_valid_channels(times, response, std)

    area = _temcompany_comment_scalar(comments, "LoopArea (m2)", 0.0)
    loop_x = _temcompany_comment_scalar(comments, "LoopX (m)", 0.0)
    loop_y = _temcompany_comment_scalar(comments, "LoopY (m)", 0.0)
    if area <= 0.0:
        area = abs(loop_x * loop_y)
    rx_x = _temcompany_comment_scalar(comments, "RXcoil X-Position (m)", 0.0)
    height = _temcompany_comment_scalar(comments, "LoopZ (m)", 0.0)
    system = {
        "source_radius": math.sqrt(area / math.pi) if area > 0.0 else 10.0,
        "tx_rx_sep": abs(rx_x),
        "height": height,
        "orientation": "z",
        "waveform": "step_off",
        "receiver_type": "dbdt",
        "response_sign": -1.0,
        "data_scale": 1.0,
        "auto_scale": False,
        "loop_area": area,
        "loop_turns": int(_temcompany_comment_scalar(comments, "LoopTurns", 1.0)),
        # The export is dB/dt already divided by the transmitter moment
        # (V/A/m^4), so the forward has to model a UNIT moment. Modelling the
        # real moment on top counts it twice: on this instrument that is a
        # factor turns * area = 1.59, and the inversion pays for it by raising
        # every recovered resistivity to bring the amplitude back down.
        "source_moment": 1.0,
    }
    station_index = columns.get(station_name)
    return {
        "times": times,
        "response": response,
        "n_soundings": len(records),
        "sounding": s,
        "relative_std": std,
        "uniform_error": _temcompany_baked_uniform_error(std),
        "positions": _temcompany_positions(x, y),
        "x": x,
        "y": y,
        "longitude": lon,
        "latitude": lat,
        "elevation": elevation,
        "heights": np.full(len(records), height, dtype=float),
        "line_numbers": np.asarray(
            [int(numeric("line", item, 0)) for item in records], dtype=int),
        "station_ids": np.asarray([
            item[station_index] if station_index is not None else str(index + 1)
            for index, item in enumerate(records)
        ]),
        "temcompany": True,
        "tem_moment": moment,
        "source_format": "TEMcompany raw XYZ" if raw_export else "TEMcompany station XYZ",
        "coordinate_system": "local metric (from latitude/longitude)",
        "system": system,
        # A text export carries no inversion settings, so the grid is always the
        # one the gate range suggests.
        "inversion_defaults": suggest_layer_grid(times),
    }


def load_temcompany_sounding(
    path: str, sounding: int = 0, moment: str = "HM", *, use_flags: bool = True,
    max_relative_std: Optional[float] = None, gate_rejection: str = "truncate",
    reject_negative: bool = False,
    min_gates_per_moment: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    """Load a TEMcompany/TEM2Go sounding from a project folder or XYZ export.

    The defaults reproduce the gate selection the project itself records, which
    was measured rather than assumed. Over 1,503 station-moment datasets of one
    project, the
    gates the stored inversion used (``InversionModel.Datasets``) are exactly
    the gates whose ``InUseFlags`` are set and whose value is finite and not a
    dummy, with 100 percent agreement. There is no further sign test: 87 low-
    moment and 251 high-moment datasets keep a non-positive gate. There is no
    further error cut either, and none is needed, because the largest relative
    error among the kept gates is exactly 0.250, so TEMImage applied that cut
    upstream when it wrote the flags. The selection is not even contiguous, so
    nor truncation: only 36 percent of the high-moment selections
    are a single run of gates.

    So ``max_relative_std=None`` and ``reject_negative=False`` are the defaults.
    Both arguments remain, because a survey whose flags were written by an older
    an older acquisition release, or one being deliberately treated more
    strictly, still needs them. Note that the sign test only runs when
    ``max_relative_std`` is set: it condemns a gate alongside a noisy one rather
    than on its own.

    ``use_flags=False`` ignores the project's in-use flags and returns every gate
    with a finite, non-dummy value. Only a project database records those flags,
    so it makes no difference to an XYZ export.

    ``min_gates_per_moment`` applies only to a joint ``LM+HM`` read of a project
    database, which is the only path where dropping one moment still leaves a
    sounding; see :func:`_load_temcompany_joint_database`.
    """
    selected = _normalise_temcompany_moment(moment)
    source = Path(path)
    if source.is_dir():
        databases = sorted(source.glob("*.db"))
        project_db = next(
            (item for item in databases if item.name.lower() == "project.db"),
            databases[0] if len(databases) == 1 else None,
        )
        if project_db is not None:
            if selected == "LM+HM":
                return _load_temcompany_joint_database(
                    project_db, sounding, use_flags, max_relative_std,
                    gate_rejection, reject_negative, min_gates_per_moment)
            return _load_temcompany_database(
                project_db, sounding, selected, use_flags, max_relative_std,
                gate_rejection, reject_negative)
        xyz = sorted(source.glob("*_StationData.xyz"))
        if not xyz:
            xyz = sorted(source.glob("*_RawData.xyz"))
        if len(xyz) != 1:
            raise ValueError(
                "Select a TEMcompany project directory containing project.db or "
                "one *_StationData.xyz file.")
        source = xyz[0]
    if not source.exists():
        raise ValueError(f"File not found: {source}")
    if selected == "LM+HM":
        raise ValueError(
            "Joint LM+HM loading currently requires a TEMcompany project folder "
            "containing project.db; select HM or LM for a standalone XYZ export.")
    return _load_temcompany_xyz(source, sounding, selected)


def load_sounding(
    path: str, method: str, sounding: int = 0, *, moment: str = "HM",
    use_flags: bool = True, max_relative_std: Optional[float] = None,
    gate_rejection: str = "truncate", reject_negative: bool = False,
    min_gates_per_moment: Optional[Mapping[str, int]] = None,
    ttem_loop_area: Optional[float] = None, ttem_gex_path: Optional[str] = None,
    ttem_tfi_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one sounding from a sounding file.

    The first column is the abscissa (FDEM: frequency Hz; TDEM: time s). The
    remaining columns hold the response(s) — a single sounding, or several stacked
    side by side so one file can carry a whole survey line (common for airborne EM
    exports). ``sounding`` picks which one (0-based):

    - **TDEM**: each extra column is one sounding's response → column ``1 + sounding``.
    - **FDEM**: response columns come in ``(real, imag)`` pairs → one sounding is the
      pair starting at ``1 + 2*sounding``; a lone trailing real column gives imag = 0.

    The returned dict also reports ``n_soundings`` so the caller can offer a picker.
    ``use_flags`` applies only to TEMcompany project databases; see
    :func:`load_temcompany_sounding`.
    """
    if run_inputs.is_container(path):
        return load_sounding_container(path, method, sounding=sounding)
    if is_ttem_source(path):
        if method != "TDEM":
            raise ValueError("TEMcompany tTEM raw data are time-domain EM data; select TDEM.")
        return load_ttem_sounding(path, sounding=sounding, moment=moment,
                                  max_relative_std=max_relative_std,
                                  loop_area=ttem_loop_area,
                                  gex_path=ttem_gex_path,
                                  tfi_path=ttem_tfi_path)
    if is_temcompany_source(path):
        if method != "TDEM":
            raise ValueError("TEMcompany/TEM2Go exports are time-domain EM data; select TDEM.")
        return load_temcompany_sounding(path, sounding=sounding, moment=moment,
                                        use_flags=use_flags,
                                        max_relative_std=max_relative_std,
                                        gate_rejection=gate_rejection,
                                        reject_negative=reject_negative,
                                        min_gates_per_moment=min_gates_per_moment)
    table = np.atleast_2d(table_io.load_2d_array(path)).astype(float)
    if table.shape[1] < 2:
        raise ValueError(f"Expected >= 2 columns, got shape {table.shape}.")
    x = table[:, 0]
    n_resp = table.shape[1] - 1
    if method == "FDEM":
        n_soundings = max(1, (n_resp + 1) // 2)
        s = max(0, min(int(sounding), n_soundings - 1))
        ri = 1 + 2 * s
        real = table[:, ri]
        imag = table[:, ri + 1] if ri + 1 < table.shape[1] else np.zeros_like(x)
        return {"frequencies": x, "real": real, "imag": imag,
                "n_soundings": n_soundings, "sounding": s}
    n_soundings = n_resp
    s = max(0, min(int(sounding), n_soundings - 1))
    return {"times": x, "response": table[:, 1 + s],
            "n_soundings": n_soundings, "sounding": s}


def save_sounding_container(
    destination: str | Path, path: str, method: str, *, moment: str = "HM",
    use_flags: bool = True, max_relative_std: Optional[float] = None,
    gate_rejection: str = "truncate", reject_negative: bool = False,
    min_gates_per_moment: Optional[Mapping[str, int]] = None,
    ttem_loop_area: Optional[float] = None, ttem_gex_path: Optional[str] = None,
    ttem_tfi_path: Optional[str] = None, progress=None,
) -> Path:
    """Materialize every sounding in ``path`` into one compressed container.

    A recorded run used to keep its input by copying the acquisition folder,
    which for a TEMcompany project is hundreds of megabytes per inversion. The
    soundings themselves are a few hundred kilobytes, and they are the only part
    the inversion reads, so this parses the survey once and stores the result.

    The load settings are baked in, because they decide what the arrays contain:
    a container written for ``LM+HM`` holds joint moments, and re-reading it
    under another moment would silently return the wrong gates. They travel in
    the manifest so a reader can report what it was given.
    """
    settings = dict(
        moment=moment, use_flags=use_flags, max_relative_std=max_relative_std,
        gate_rejection=gate_rejection, reject_negative=reject_negative,
        min_gates_per_moment=(dict(min_gates_per_moment)
                              if min_gates_per_moment else None),
        ttem_loop_area=ttem_loop_area, ttem_gex_path=ttem_gex_path,
        ttem_tfi_path=ttem_tfi_path,
    )
    head = load_sounding(path, method, sounding=0, **settings)
    total = max(1, int(head.get("n_soundings", 1)))
    soundings: List[Dict[str, Any]] = [head]
    for index in range(1, total):
        soundings.append(load_sounding(path, method, sounding=index, **settings))
        if progress is not None and index % 100 == 0:
            progress(f"  materialized {index + 1}/{total} soundings")
    return run_inputs.save_sequence_container(
        destination,
        soundings,
        kind=SOUNDING_CONTAINER_KIND,
        meta={
            "method": str(method).upper(),
            "n_soundings": total,
            "source_name": Path(path).name,
            "source_format": str(head.get("source_format", "")),
            **{key: value for key, value in settings.items() if value is not None},
        },
    )


def load_sounding_container(
    path: str | Path, method: str, sounding: int = 0
) -> Dict[str, Any]:
    """Return one sounding from a container written by :func:`save_sounding_container`.

    The moment and flag settings are not arguments here: they were applied when
    the container was written, and the stored arrays are what they produced.
    """
    manifest = run_inputs.read_manifest(path)
    stored_method = str(manifest.get("meta", {}).get("method", "")).upper()
    if stored_method and str(method).upper() != stored_method:
        raise ValueError(
            f"{Path(path).name} holds {stored_method} soundings; {str(method).upper()} "
            "was requested. Re-import the survey to invert it as the other method."
        )
    if run_inputs.sequence_length(path) < 1:
        raise ValueError(f"{Path(path).name} holds no soundings.")
    return dict(
        run_inputs.load_sequence_item(
            path, int(sounding), kind=SOUNDING_CONTAINER_KIND
        )
    )


def load_line_geometry(path: str) -> Dict[str, Any]:
    """Load per-sounding line geometry: along-line ``positions`` (m), optional
    sensor ``heights`` (m), and optional map coordinates ``x``/``y`` (e.g.
    easting/northing) for plan-view depth slices. Recognizes header names
    (distance/position for the position; alt/height for the height;
    easting/northing for the map coordinates, which also derive the distance when
    no distance column is present). A header-less file is read by column order
    (1 column = position; 2+ = position, height). ``positions`` is shifted to
    start at 0.
    """
    positions = heights = xs = ys = None
    try:
        import pandas as pd
        df = pd.read_csv(path)
        if df.shape[1] >= 1 and any(not str(c).replace(".", "").lstrip("-").isdigit()
                                    for c in df.columns):
            cols = {str(c).lower().strip(): c for c in df.columns}

            def pick(names):
                for nm in names:
                    if nm in cols:
                        return np.asarray(df[cols[nm]], dtype=float).ravel()
                return None

            positions = pick(["dist_m", "distance", "position", "dist", "offset"])
            heights = pick(["sensor_alt_m", "alt", "altitude", "height", "sensor_height", "clearance"])
            xs = pick(["e_utm13n", "easting_m", "easting", "east", "e", "x_utm", "x_m", "x"])
            ys = pick(["n_utm13n", "northing_m", "northing", "north", "n", "y_utm", "y_m", "y"])
            if positions is None and xs is not None and ys is not None:
                positions = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
            elif positions is None and xs is not None:
                positions = xs
    except Exception:  # noqa: BLE001 - fall back to a plain numeric table
        positions = None
    if positions is None:
        arr = np.atleast_2d(table_io.load_2d_array(path)).astype(float)
        positions = arr[:, 0]
        heights = arr[:, 1] if arr.shape[1] >= 2 else None
    positions = np.asarray(positions, dtype=float).ravel()
    positions = positions - float(np.nanmin(positions))
    if heights is not None:
        heights = np.asarray(heights, dtype=float).ravel()
    has_xy = xs is not None and ys is not None
    return {"positions": positions, "heights": heights,
            "x": np.asarray(xs, dtype=float).ravel() if xs is not None else None,
            "y": np.asarray(ys, dtype=float).ravel() if ys is not None else None,
            "n": int(positions.size), "has_heights": heights is not None, "has_xy": has_xy}

__all__ = [
    "SOUNDING_CONTAINER_KIND",
    "TEMCOMPANY_MOMENTS",
    "is_temcompany_source",
    "is_ttem_source",
    "load_temcompany_sounding",
    "load_ttem_sounding",
    "load_sounding",
    "load_sounding_container",
    "load_line_geometry",
    "save_sounding_container",
]
