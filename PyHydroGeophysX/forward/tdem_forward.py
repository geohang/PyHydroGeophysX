"""Forward modelling for time-domain electromagnetic (TDEM) soundings.

The layered-earth response comes from SimPEG. On top of it this module applies
the instrument model that turns a modelled decay into the numbers a receiver
records: convolution with the transmitter's turn-off waveform, a cascade of
first-order receiver low-pass stages, superposition of the bipolar transmitter
train, and integration of each gate over its recorded window. Each of those is
a standard step, and their parameters are read from the acquisition files
rather than assumed; see :mod:`PyHydroGeophysX.data_processing.em1d`.

The whole instrument model is linear in the modelled response, so it is built
once as a matrix and applied to the forward and the Jacobian alike. A dataset
that does not describe an instrument keeps SimPEG's direct receiver path.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import numpy as np
import simpeg.electromagnetics.time_domain as tdem

# SimPEG imports
from discretize import TensorMesh
from simpeg import maps
from simpeg.utils import mkvc


# ---------------------------------------------------------------------------
# TDEMSurvey Config
# ---------------------------------------------------------------------------
@dataclass
class TDEMSurveyConfig:
    """Configuration for TDEM survey geometry.
    
    Attributes:
        source_location: [x, y, z] location of source center (m)
        source_radius: Radius of circular loop source (m)
        source_current: Peak current amplitude (A)
        receiver_location: [x, y, z] location of receiver (m)
        receiver_orientation: Component to measure ('x', 'y', or 'z')
        times: Time channels for measurement (s)
        waveform_type: Type of waveform ('step_off', 'ramp_off', 'custom')
    """
    source_location: np.ndarray = None
    source_radius: float = 10.0
    source_current: float = 1.0
    source_turns: int = 1
    #: Transmitter moment in A m^2, overriding ``current * turns * area``. Set it
    #: to 1.0 for data already normalized by the transmitter moment, which is how
    #: TEMcompany instruments report dB/dt (V/A/m^4): dividing the measurement by
    #: the moment and then modelling with that moment counts it twice.
    source_moment: Optional[float] = None
    #: Turn-off waveform as (times, currents) nodes. A real ramp is not a step,
    #: and the earliest gates of a ground system sit only microseconds after it
    #: ends, which is exactly where the difference bites.
    waveform_times: Optional[np.ndarray] = None
    waveform_currents: Optional[np.ndarray] = None
    #: Gate windows, for averaging the response over each one instead of reading
    #: it at the gate centre.
    gate_open: Optional[np.ndarray] = None
    gate_close: Optional[np.ndarray] = None
    #: How a modelled decay becomes a gate value. See :func:`_gate_sampling`.
    gate_window: str = "centre"
    #: Total cosine-taper fraction of the gate window, as stored in the
    #: project's ``GateShapePar1``. 0.667 leaves a 0.333 flat top.
    gate_window_par: float = 0.667
    #: Half-period of the bipolar transmitter cycle, in seconds. Set it to model
    #: the earlier pulses of the train the way the reference implementation does.
    waveform_period: Optional[float] = None
    #: How many earlier half-cycles the repetition sums, three in the reference.
    waveform_repetitions: int = 3
    #: Resolution of the internal grid the analog filter is integrated on, in
    #: samples per decade of time. See :func:`_analog_sampling`.
    analog_points_per_decade: int = 150
    #: Resolution of the grid SimPEG is asked to model, in samples per decade.
    #: Kept separate because SimPEG's setup cost grows with it while the
    #: filter's accuracy does not depend on it.
    analog_model_points_per_decade: int = 40
    #: Analog receiver electronics parsed from a GEX file.  The supported
    #: fields are ``receiver_damping`` / ``receiver_cutoff_hz`` for the
    #: receiver-coil two-pole filter and ``tib_order`` / ``tib_cutoff_hz`` for
    #: the transmitter-interface-board low-pass filter.
    analog_lowpass: Optional[dict] = None
    #: Response grid density, in samples per decade. Ten matches the grid the
    #: instrument's own processing works on, with local interpolation between
    #: those nodes.
    instrument_points_per_decade: int = 10
    #: SimPEG step-response density, in samples per decade. Ten matches the
    #: response grid and avoids asking SimPEG for redundant samples; raise it
    #: independently for convergence checks.
    instrument_model_points_per_decade: int = 10
    #: Gauss points per smooth panel of the gate window. Eight integrates it to
    #: a few parts per million, negligible beside one SimPEG call.
    gate_quadrature_order: int = 8
    receiver_location: np.ndarray = None
    receiver_orientation: str = "z"
    receiver_type: str = "b"
    times: np.ndarray = None
    waveform_type: str = "step_off"

    def __post_init__(self):
        if self.source_location is None:
            self.source_location = np.array([0.0, 0.0, 0.0])
        if self.receiver_location is None:
            self.receiver_location = np.array([0.0, 0.0, 0.0])
        if self.times is None:
            self.times = np.logspace(-5, -2, 31)


#: How a modelled decay becomes one gate value.
#:
#: A recorded gate is an integral of the transient over a window, so the model
#: of it is the same integral. The acquisition files name the window:
#: ``GateShape`` selects its shape and ``GateShapePar1`` gives the fraction of
#: the width that is cosine-tapered, so the usual 0.667 leaves a flat top over
#: the middle third.
#:
#: ``tukey`` is that window. The integral runs in linear time, and the response
#: between the samples of its own grid is interpolated with the local cubic
#: Hermite rule described in :func:`_local_log_hermite_matrix`. Reading
#: ``GateShapePar1`` as the tapered fraction rather than as the flat top changes
#: the result by 1.5 to 4 percent; responses stored in two surveys pick the
#: reading used here.
#:
#: Two things separate it from a flat average of the window, and they pull the
#: same way. The taper draws weight in from the edges, and the window is
#: symmetric in linear time while the gate centre a file records is the
#: geometric mean of open and close, so it sits slightly late relative to it.
#: Neither rule is trying to reproduce the centre value.
#:
#: ``square`` is the same integral with no taper, for a survey whose files ask
#: for the plain window. ``centre`` reads the response at the gate centre and is
#: the fallback for data that records no gate windows at all.
GATE_WINDOWS: Tuple[str, ...] = ("tukey", "square", "centre")

#: ``GateShape`` values, as the acquisition files record them.
#:
#: Only the shapes this module implements appear. A file asking for one that
#: does not is refused rather than quietly given the nearest thing, because a
#: wrong window in the forward has nothing downstream to catch it. See
#: :func:`PyHydroGeophysX.forward.em1d._gate_window_name`.
GATE_SHAPE_NAMES = {1: "tukey", 2: "square"}


def _tukey_quadrature(alpha: float, per_panel: int = 3):
    """Nodes and weights for a normalised Tukey window on [0, 1].

    ``alpha`` follows the project ``GateShapePar1`` convention: it is the total
    cosine-tapered fraction, split between the two ends. Its
    derivative jumps where the taper meets the flat top, so the interval is
    split there and each panel gets its own Gauss-Legendre rule.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    edges = ([0.0, 1.0] if alpha <= 0.0
             else [0.0, alpha / 2.0, 1.0 - alpha / 2.0, 1.0])
    nodes, weights = np.polynomial.legendre.leggauss(max(2, int(per_panel)))
    x_all, w_all = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        if high <= low + 1e-15:
            continue
        mid, half = 0.5 * (high + low), 0.5 * (high - low)
        x_all.append(mid + half * nodes)
        w_all.append(half * weights)
    x = np.concatenate(x_all)
    w = np.concatenate(w_all)
    if alpha > 0.0:
        half = alpha / 2.0
        shape = np.ones_like(x)
        low = x < half
        high = x > 1.0 - half
        shape[low] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[low] / alpha - 1.0)))
        shape[high] = 0.5 * (
            1.0 + np.cos(np.pi * (2.0 * (x[high] - 1.0) / alpha + 1.0)))
        w = w * shape
    return x, w / w.sum()


def _gate_sampling(config: "TDEMSurveyConfig"):
    """Times to model, and the matrix that reduces them onto the gates.

    Returns ``(times, None)`` when the response is read at the gate centre,
    which needs no reduction, and ``(sample_times, matrix)`` otherwise.
    """
    times = np.asarray(config.times, dtype=float).ravel()
    window = str(getattr(config, "gate_window", "centre")).strip().lower()
    if window not in GATE_WINDOWS:
        raise ValueError(
            f"gate_window must be one of {', '.join(GATE_WINDOWS)}; "
            f"got {config.gate_window!r}.")
    opens, closes = config.gate_open, config.gate_close
    if window == "centre" or opens is None or closes is None:
        return times, None
    opens = np.asarray(opens, dtype=float).ravel()
    closes = np.asarray(closes, dtype=float).ravel()
    if opens.size != times.size or closes.size != times.size:
        return times, None
    alpha = (0.0 if window == "square"
             else float(getattr(config, "gate_window_par", 0.667) or 0.0))
    offsets, weights = _tukey_quadrature(alpha)
    count = offsets.size
    sample = np.empty(times.size * count, dtype=float)
    matrix = np.zeros((times.size, times.size * count), dtype=float)
    for gate, (low, high) in enumerate(zip(opens, closes)):
        block = slice(gate * count, (gate + 1) * count)
        # Linear in time, because a gate is a window on the recorded transient
        # and the instrument integrates it as one.
        sample[block] = low + (high - low) * offsets
        matrix[gate, block] = weights
    return sample, matrix


def _valid_positive(value, default=0.0) -> float:
    """Return a finite positive float or *default*."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) and parsed > 0.0 else float(default)


def _analog_parameters(
    spec: Optional[dict],
) -> "tuple[float, float, int, float, tuple]":
    """Analog-filter parameters for caching and modelling.

    Two shapes are supported because instruments describe their electronics
    differently. A GEX file gives one damped second-order receiver stage
    (``receiver_damping`` with ``receiver_cutoff_hz``) and an optional
    order-N front-end stage at a single corner. Other systems list the corner
    frequencies of cascaded first-order stages instead, which is what
    ``first_order_cutoffs_hz`` carries.
    """
    values = dict(spec or {})
    damping = _valid_positive(values.get("receiver_damping"))
    receiver_cutoff = _valid_positive(values.get("receiver_cutoff_hz"))
    tib_cutoff = _valid_positive(values.get("tib_cutoff_hz"))
    try:
        tib_order = max(0, int(round(float(values.get("tib_order", 0)))))
    except (TypeError, ValueError):
        tib_order = 0
    if not (damping and receiver_cutoff):
        damping = receiver_cutoff = 0.0
    if not tib_cutoff:
        tib_order = 0
    first_order = tuple(
        float(value)
        for value in np.asarray(
            values.get("first_order_cutoffs_hz", ()), dtype=float).ravel()
        if np.isfinite(value) and value > 0.0
    )
    return damping, receiver_cutoff, tib_order, tib_cutoff, first_order


def _analog_sampling(config: "TDEMSurveyConfig"):
    """Return SimPEG times and the combined analog-filter/repetition/gate operator.

    The internal grid is uniform in log time across the whole modelled range.
    That is a change from an earlier two-region grid which was dense only until
    the filter had settled and then fell back to the receiver times themselves,
    roughly 0.05 decade apart. The first-order-hold reconstruction was then
    integrating across steps twenty times longer than the filter's own time
    constant, and the per-step lag that leaves does not die away with time the
    way the continuous operator does: on one station it added 2.2 percent at
    95 us, where a 0.55 us group delay can account for at most 1.4 percent.

    The error is set by how well a piecewise-linear reconstruction represents
    the decay, not by the filter, so the density belongs in log time and the
    same density serves every decade. Measured against a 600-points-per-decade
    reference on one station, the largest gate moves by 0.79 percent going from
    20 to 40 points per decade, 0.31 percent from 40 to 80, 0.08 percent from
    80 to 150 and 0.04 percent from 150 to 300. The default of 150 sits at the
    0.1 percent level this was asked to reach.
    """
    targets, weights = _gate_sampling(config)
    damping, receiver_cutoff, tib_order, tib_cutoff, first_order = _analog_parameters(
        config.analog_lowpass
    )
    if receiver_cutoff <= 0.0 and tib_order <= 0 and not first_order:
        return targets, weights

    targets = np.asarray(targets, dtype=float).ravel()
    positive = targets[np.isfinite(targets) & (targets > 0.0)]
    if positive.size != targets.size:
        raise ValueError("Analog-filtered TDEM receiver times must be finite and positive.")
    active_cutoffs = [value for value in
                      (receiver_cutoff, tib_cutoff if tib_order else 0.0, *first_order)
                      if value > 0.0]
    filter_scale = 1.0 / (2.0 * np.pi * max(active_cutoffs))
    # Simulation1DLayered loses useful accuracy extremely close to t=0 while
    # these electronics have 0.2--0.4 microsecond time constants.  Starting at
    # 0.05 microsecond is still well inside both filters and avoids making the
    # Hankel/time transform chase an irrelevant nanosecond singularity. What is
    # dropped is the response before that instant, which the filter weights by
    # exp(-(t - start) / tau); at the earliest gate of a ground system that is
    # of order 1e-6, so the omission is far below the modelling error.
    start = max(5e-8, min(float(positive.min()) / 100.0, filter_scale / 40.0))
    stop = float(positive.max())
    decades = max(np.log10(stop / start), 1e-6)

    def _grid(per_decade: int) -> np.ndarray:
        count = int(min(np.ceil(decades * max(4, per_decade)) + 1, 8192))
        return np.geomspace(start, stop, count)

    # Two grids, because they are paying for different things. SimPEG models the
    # coarse one, and its setup cost grows with how many times it is asked for;
    # the filter is integrated on the dense one, and its accuracy is set by how
    # well a piecewise-linear reconstruction follows the decay. Tying the two
    # together made the filter converge only by making every forward operator
    # four times more expensive to build. A cubic spline in log time carries the
    # coarse response onto the dense grid, which costs a matrix of a few hundred
    # by a few hundred and is exact to far better than either grid's own error:
    # the response is smooth in log-log and changes by about 15 percent per step
    # at 40 points per decade, where a cubic spline errs in the seventh digit.
    #
    # The gate quadrature nodes go on the dense grid alone. There are nine of
    # them per gate, so anchoring SimPEG to them instead would quadruple the
    # times it is asked for and buy nothing: they sit inside a 0.2 decade window
    # that the coarse grid already spans, and the spline reaches them.
    anchor = np.asarray(config.times, dtype=float).ravel()
    modelled = np.unique(np.concatenate([
        _grid(int(getattr(config, "analog_model_points_per_decade", 40))),
        anchor[np.isfinite(anchor) & (anchor > 0.0)],
    ]))
    dense = np.unique(np.concatenate([
        _grid(int(getattr(config, "analog_points_per_decade", 150))),
        modelled, positive,
    ]))
    filtered = _filter_operator(
        dense, _log_spline_matrix(modelled, dense),
        (damping, receiver_cutoff, tib_order, tib_cutoff, first_order))
    lookup = np.searchsorted(dense, targets)
    reduction = filtered[lookup]
    if weights is not None:
        reduction = weights @ reduction
    return modelled, reduction


@lru_cache(maxsize=32)
def _log_spline_matrix_cached(source: tuple, target: tuple) -> np.ndarray:
    """Cubic-spline interpolation from *source* times onto *target*, in log time.

    Linear in the sampled values, so it composes with the filter and gate
    matrices and leaves the Jacobian exact.
    """
    from scipy.interpolate import make_interp_spline

    x = np.log(np.asarray(source, dtype=float))
    order = 3 if x.size > 3 else max(1, x.size - 1)
    spline = make_interp_spline(x, np.eye(x.size), k=order)
    matrix = np.asarray(spline(np.log(np.asarray(target, dtype=float))), dtype=float)
    matrix.setflags(write=False)
    return matrix


def _log_spline_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _log_spline_matrix_cached(
        tuple(float(value) for value in source),
        tuple(float(value) for value in target),
    )


@lru_cache(maxsize=32)
def _local_log_hermite_matrix_cached(source: tuple, target: tuple) -> np.ndarray:
    """Local cubic Hermite interpolation in log time, as a linear operator.

    Deliberately not SciPy's global cubic spline. A response tabulated at ten
    points per decade is conventionally interpolated with a local cubic Hermite
    polynomial whose nodal slopes are centred secants and whose endpoint slopes
    come from the quadratic through the first three points; the same rule is
    used by the open TEM1D reference implementation
    (https://github.com/hydrogeophysicsgroup/TEM1D, MIT).

    Keeping it as a matrix applies the identical interpolation to the response
    and to the Jacobian. Values outside the source interval are zero.
    """
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if source_array.size < 2 or np.any(source_array <= 0.0):
        raise ValueError("Local log-time interpolation needs at least two positive times.")
    if np.any(np.diff(source_array) <= 0.0):
        raise ValueError("Local log-time interpolation source times must increase.")
    x = np.log(source_array)
    slope = np.zeros((x.size, x.size), dtype=float)
    if x.size == 2:
        secant = np.array([-1.0, 1.0]) / (x[1] - x[0])
        slope[0] = secant
        slope[1] = secant
    else:
        for index in range(1, x.size - 1):
            width = x[index + 1] - x[index - 1]
            slope[index, index - 1] = -1.0 / width
            slope[index, index + 1] = 1.0 / width
        first = np.zeros(x.size, dtype=float)
        first[:2] = (-1.0, 1.0)
        first /= x[1] - x[0]
        last = np.zeros(x.size, dtype=float)
        last[-2:] = (-1.0, 1.0)
        last /= x[-1] - x[-2]
        slope[0] = 2.0 * first - slope[1]
        slope[-1] = 2.0 * last - slope[-2]

    matrix = np.zeros((target_array.size, source_array.size), dtype=float)
    valid = ((target_array >= source_array[0])
             & (target_array <= source_array[-1])
             & np.isfinite(target_array))
    for row in np.flatnonzero(valid):
        z = float(np.log(target_array[row]))
        index = min(max(int(np.searchsorted(x, z, side="right") - 1), 0), x.size - 2)
        width = x[index + 1] - x[index]
        u = float(np.clip((z - x[index]) / width, 0.0, 1.0))
        h00 = 2.0 * u ** 3 - 3.0 * u ** 2 + 1.0
        h10 = u ** 3 - 2.0 * u ** 2 + u
        h01 = -2.0 * u ** 3 + 3.0 * u ** 2
        h11 = u ** 3 - u ** 2
        matrix[row, index] += h00
        matrix[row, index + 1] += h01
        matrix[row] += h10 * width * slope[index]
        matrix[row] += h11 * width * slope[index + 1]
    matrix.setflags(write=False)
    return matrix


def _local_log_hermite_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _local_log_hermite_matrix_cached(
        tuple(float(value) for value in source),
        tuple(float(value) for value in target),
    )


def _first_order_filter_operator(times: np.ndarray, values: np.ndarray,
                                 cutoff_hz: float) -> np.ndarray:
    """Apply one continuous first-order low-pass to every operator column.

    The recurrence integrates a first-order-hold input exactly.  Unlike forming
    and multiplying dense N-by-N filter matrices, its work is O(N*M), where M
    is the much smaller number of SimPEG receiver samples.  NumPy performs each
    row update in compiled code, so this is both faster and numerically cleaner
    than moving the operation to a custom C++ extension.
    """
    sample_times = np.asarray(times, dtype=float)
    source = np.asarray(values, dtype=float)
    output = np.zeros_like(source)
    omega = 2.0 * np.pi * float(cutoff_hz)
    for index in range(1, sample_times.size):
        q = omega * float(sample_times[index] - sample_times[index - 1])
        decay = float(np.exp(-q))
        if abs(q) < 1e-5:
            to_next = q / 2.0 - q * q / 6.0 + q ** 3 / 24.0
        else:
            to_next = 1.0 + np.expm1(-q) / q
        to_previous = 1.0 - decay - to_next
        output[index] = (
            decay * output[index - 1]
            + to_previous * source[index - 1]
            + to_next * source[index]
        )
    return output


def _state_space_filter_operator(times: np.ndarray, values: np.ndarray,
                                 a: np.ndarray, b: np.ndarray,
                                 c: np.ndarray) -> np.ndarray:
    """First-order-hold state-space filtering without a dense N-by-N matrix."""
    from scipy.linalg import expm

    sample_times = np.asarray(times, dtype=float)
    source = np.asarray(values, dtype=float)
    states = np.zeros((a.shape[0], source.shape[1]), dtype=float)
    output = np.zeros_like(source)
    for index in range(1, sample_times.size):
        dt = float(sample_times[index] - sample_times[index - 1])
        augmented = np.zeros((a.shape[0] + 2, a.shape[0] + 2), dtype=float)
        augmented[:a.shape[0], :a.shape[0]] = a
        augmented[:a.shape[0], a.shape[0]] = b.ravel()
        augmented[a.shape[0], a.shape[0] + 1] = 1.0
        transition = expm(augmented * dt)
        from_value = transition[:a.shape[0], a.shape[0]]
        from_slope = transition[:a.shape[0], a.shape[0] + 1]
        to_next = from_slope / dt
        to_previous = from_value - to_next
        states = (
            transition[:a.shape[0], :a.shape[0]] @ states
            + to_previous[:, None] * source[index - 1][None, :]
            + to_next[:, None] * source[index][None, :]
        )
        output[index] = (c @ states).ravel()
    return output


def _filter_operator(times: np.ndarray, values: np.ndarray,
                     parameters: tuple) -> np.ndarray:
    """Apply the parsed receiver electronics to an existing linear operator."""
    damping, receiver_cutoff, tib_order, tib_cutoff, first_order = parameters
    output = np.asarray(values, dtype=float)
    if receiver_cutoff > 0.0:
        omega = 2.0 * np.pi * receiver_cutoff
        output = _state_space_filter_operator(
            times, output,
            np.array([[0.0, 1.0], [-omega ** 2, -2.0 * damping * omega]]),
            np.array([[0.0], [omega ** 2]]), np.array([[1.0, 0.0]]),
        )
    for cutoff in first_order:
        output = _first_order_filter_operator(times, output, cutoff)
    for _ in range(tib_order):
        output = _first_order_filter_operator(times, output, tib_cutoff)
    return output


def _gate_integration_operator(raw_times: np.ndarray, times: np.ndarray,
                              opens: np.ndarray, closes: np.ndarray,
                              window: str, parameter: float,
                              quadrature_order: int) -> np.ndarray:
    """Reduce the response grid onto the recorded gates."""
    window = str(window).strip().lower()
    if window == "centre" or opens.size != times.size or closes.size != times.size:
        return _local_log_hermite_matrix(raw_times, times)
    if window == "simpson_log":
        samples = np.column_stack((opens, times, closes)).ravel()
        reduction = np.zeros((times.size, samples.size), dtype=float)
        for gate in range(times.size):
            reduction[gate, gate * 3:gate * 3 + 3] = (1.0 / 6.0, 4.0 / 6.0, 1.0 / 6.0)
        return reduction @ _local_log_hermite_matrix(raw_times, samples)

    if window == "tukey":
        offsets, weights = _tukey_quadrature(parameter, quadrature_order)
    elif window == "square" or window == "linear_time":
        nodes, gauss_weights = np.polynomial.legendre.leggauss(max(2, quadrature_order))
        offsets, weights = 0.5 * (nodes + 1.0), 0.5 * gauss_weights
        weights /= weights.sum()
    else:
        raise ValueError(f"TEMcompany gate operator does not support {window!r}.")
    samples = np.concatenate([
        low + (high - low) * offsets for low, high in zip(opens, closes)
    ])
    reduction = np.zeros((times.size, samples.size), dtype=float)
    count = offsets.size
    for gate in range(times.size):
        reduction[gate, gate * count:(gate + 1) * count] = weights
    return reduction @ _local_log_hermite_matrix(raw_times, samples)


@lru_cache(maxsize=16)
def _instrument_chain_cached(
    times_key: tuple, opens_key: tuple, closes_key: tuple,
    waveform_times_key: tuple, waveform_currents_key: tuple,
    half_period: float, repetitions: int, window: str, window_parameter: float,
    filter_parameters: tuple, dense_points_per_decade: int,
    model_points_per_decade: int, native_points_per_decade: int,
    quadrature_order: int,
) -> tuple:
    """Cached instrument chain, from SimPEG step-response samples to gates."""
    times = np.asarray(times_key, dtype=float)
    opens = np.asarray(opens_key, dtype=float)
    closes = np.asarray(closes_key, dtype=float)
    waveform_times = np.asarray(waveform_times_key, dtype=float)
    waveform_currents = np.asarray(waveform_currents_key, dtype=float)
    if waveform_times.size < 2 or waveform_times.size != waveform_currents.size:
        raise ValueError(
            "Waveform convolution needs matching time and current nodes.")
    if np.any(np.diff(waveform_times) <= 0.0):
        raise ValueError("Waveform times must be strictly increasing.")
    if not np.all(np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError("Gate times must be finite and positive.")

    terms = max(0, int(repetitions)) if half_period > 0.0 else 0
    latest_gate = float(np.max(closes)) if closes.size == times.size else float(np.max(times))
    earliest_waveform = float(np.min(waveform_times))
    # Enough tail for all four bipolar pulses and for the earliest (negative)
    # waveform node used by the convolution at the last gate.
    required_stop = latest_gate - earliest_waveform + terms * max(half_period, 0.0)
    start = 1.0e-8
    native_ppd = max(4, int(native_points_per_decade))
    native_stop_log = np.ceil(np.log10(required_stop) * native_ppd) / native_ppd
    raw = 10.0 ** np.arange(np.log10(start), native_stop_log + 0.5 / native_ppd,
                            1.0 / native_ppd)

    def log_grid(points_per_decade: int) -> np.ndarray:
        ppd = max(4, int(points_per_decade))
        stop_log = np.ceil(np.log10(raw[-1]) * ppd) / ppd
        return 10.0 ** np.arange(np.log10(start), stop_log + 0.5 / ppd, 1.0 / ppd)

    modelled = np.unique(np.concatenate((log_grid(model_points_per_decade), raw)))
    dense = np.unique(np.concatenate((log_grid(dense_points_per_decade), modelled)))
    model_to_dense = _log_spline_matrix(modelled, dense)
    filtered_dense = _filter_operator(dense, model_to_dense, filter_parameters)
    raw_lookup = np.searchsorted(dense, raw)
    if not np.allclose(dense[raw_lookup], raw, rtol=1e-12, atol=0.0):
        raise RuntimeError("Native response grid was not retained in the filter grid.")
    filtered_raw = filtered_dense[raw_lookup]

    if terms:
        repeated_stop = raw[-1] - terms * half_period
        repeated_times = raw[raw <= repeated_stop * (1.0 + 1e-14)]
        repetition = np.zeros((repeated_times.size, raw.size), dtype=float)
        for term in range(terms + 1):
            repetition += ((-1.0) ** term) * _log_spline_matrix(
                raw, repeated_times + term * half_period)
    else:
        repeated_times = raw
        repetition = np.eye(raw.size, dtype=float)

    # Convolve the slope changes of the recorded current waveform with the
    # filtered, repeated step response. For a piecewise-linear waveform the
    # second derivative is a train of impulses at the nodes, so the integral
    # collapses to this weighted sum.
    slopes = np.diff(waveform_currents) / np.diff(waveform_times)
    slope_changes = np.empty(waveform_times.size, dtype=float)
    slope_changes[0] = slopes[0]
    slope_changes[1:-1] = np.diff(slopes)
    slope_changes[-1] = -slopes[-1]
    convolved_stop = repeated_times[-1] + earliest_waveform
    convolved_times = repeated_times[repeated_times <= convolved_stop * (1.0 + 1e-14)]
    convolution = np.zeros((convolved_times.size, repeated_times.size), dtype=float)
    for waveform_time, coefficient in zip(waveform_times, slope_changes):
        delayed = convolved_times - waveform_time
        valid = ((delayed >= repeated_times[0])
                 & (delayed <= repeated_times[-1]))
        if np.any(valid) and coefficient != 0.0:
            convolution[valid] += coefficient * _log_spline_matrix(
                repeated_times, delayed[valid])

    gate = _gate_integration_operator(
        convolved_times, times, opens, closes, window, window_parameter,
        quadrature_order,
    )
    # SimPEG's PointMagneticFluxTimeDerivative uses the opposite sign to the
    # positive derivative convention of the waveform wrapper. Preserve the
    # public modeler's existing dB/dt sign so callers need no special case.
    reduction = -gate @ convolution @ repetition @ filtered_raw
    modelled.setflags(write=False)
    reduction.setflags(write=False)
    return modelled, reduction


def _instrument_sampling(config: "TDEMSurveyConfig"):
    """Return receiver times and the cached native-order instrument matrix."""
    times = np.asarray(config.times, dtype=float).ravel()
    opens = np.asarray(config.gate_open if config.gate_open is not None else (), dtype=float)
    closes = np.asarray(config.gate_close if config.gate_close is not None else (), dtype=float)
    waveform_times = np.asarray(config.waveform_times, dtype=float).ravel()
    waveform_currents = np.asarray(config.waveform_currents, dtype=float).ravel()
    parameters = _analog_parameters(config.analog_lowpass)
    return _instrument_chain_cached(
        tuple(times), tuple(opens), tuple(closes), tuple(waveform_times),
        tuple(waveform_currents), float(config.waveform_period or 0.0),
        int(config.waveform_repetitions), str(config.gate_window).lower(),
        float(config.gate_window_par), parameters,
        int(config.analog_points_per_decade),
        int(config.instrument_model_points_per_decade),
        int(config.instrument_points_per_decade),
        int(config.gate_quadrature_order),
    )


def _has_instrument_model(config: "TDEMSurveyConfig") -> bool:
    """Whether the dataset describes the instrument completely enough to model it.

    There is one instrument chain, so this is a capability test rather than a
    choice: a dataset carrying the turn-off waveform, the bipolar period, the
        receiver-electronics description (which may explicitly contain no
        filters) and the gate windows gets the native order, and a
    dataset that does not gets the direct SimPEG receiver path, which is all
    that can be done with what it supplies.

    The test is deliberately strict. A custom waveform on its own does not
    describe an instrument, and treating it as one would apply a transmitter
    train and a gate window the data never mentioned. ``None`` means the
    electronics are unknown; an empty dictionary means they were described and
    contain no filter stages.
    """
    nodes = np.asarray(config.waveform_times if config.waveform_times is not None else ())
    currents = np.asarray(
        config.waveform_currents if config.waveform_currents is not None else ())
    derivative = str(config.receiver_type).lower() in {"dbdt", "db/dt", "time_derivative"}
    return bool(
        derivative and nodes.size >= 2 and nodes.size == currents.size
        and config.waveform_period and config.analog_lowpass is not None
        and config.gate_open is not None and config.gate_close is not None
    )


# ---------------------------------------------------------------------------
# TDEMForward Modeling
# ---------------------------------------------------------------------------
class TDEMForwardModeling:
    """Class for forward modeling of Time-Domain Electromagnetic (TDEM) data.
    
    This class provides functionality for 1D layered Earth TDEM forward modeling
    using SimPEG's time_domain module.
    
    Example:
        >>> # Define layer model
        >>> thicknesses = np.array([10.0, 30.0])
        >>> conductivity = np.array([0.01, 0.1, 0.001])  # S/m
        >>> 
        >>> # Create forward modeler
        >>> fwd = TDEMForwardModeling(thicknesses=thicknesses)
        >>> 
        >>> # Compute response
        >>> response = fwd.forward(conductivity)
    """
    
    def __init__(
        self,
        thicknesses: np.ndarray,
        survey_config: Optional[TDEMSurveyConfig] = None,
        survey: Optional[tdem.Survey] = None
    ):
        """
        Initialize TDEM forward modeling.
        
        Args:
            thicknesses: Layer thicknesses (m), N-1 values for N layers
            survey_config: Survey configuration (creates survey if survey not provided)
            survey: Pre-defined SimPEG TDEM survey (optional)
        """
        self.thicknesses = np.asarray(thicknesses)
        self.n_layers = len(thicknesses) + 1
        
        # Create or use provided survey
        if survey is not None:
            self.survey = survey
        else:
            if survey_config is None:
                survey_config = TDEMSurveyConfig()
            self.survey_config = survey_config
            self.survey = self._create_survey(survey_config)
        
        # Create simulation
        self.model_mapping = maps.IdentityMap(nP=self.n_layers)
        self.simulation = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.thicknesses,
            sigmaMap=self.model_mapping,
        )
    
    def _create_survey(self, config: TDEMSurveyConfig) -> tdem.Survey:
        """Create TDEM survey from configuration.
        
        Args:
            config: Survey configuration
            
        Returns:
            SimPEG TDEM Survey object
        """
        # The instrument model starts from the step-off magnetic-flux response
        # and applies the waveform, electronics and gating afterwards. That
        # order matters at the earliest low-moment gates, and it is cheaper:
        # SimPEG only sees the compact step-response grid. A dataset that does
        # not describe an instrument keeps the direct SimPEG path.
        self._instrument_model = _has_instrument_model(config)
        if self._instrument_model:
            receiver_cls = tdem.receivers.PointMagneticFluxDensity
            self._sample_times, self._gate_weights = _instrument_sampling(config)
        else:
            receiver_cls = (
                tdem.receivers.PointMagneticFluxTimeDerivative
                if str(config.receiver_type).lower() in {"dbdt", "db/dt", "time_derivative"}
                else tdem.receivers.PointMagneticFluxDensity
            )
            self._sample_times, self._gate_weights = _analog_sampling(config)
        receiver_list = [
            receiver_cls(
                config.receiver_location,
                self._sample_times,
                orientation=config.receiver_orientation
            )
        ]

        # Create waveform. Measured turn-off nodes win over the named shapes:
        # a step is only a stand-in for a ramp nobody recorded.
        nodes = np.asarray(config.waveform_times if config.waveform_times is not None
                           else [], dtype=float).ravel()
        currents = np.asarray(config.waveform_currents if config.waveform_currents
                              is not None else [], dtype=float).ravel()
        if self._instrument_model:
            waveform = tdem.sources.StepOffWaveform()
        elif nodes.size >= 2 and nodes.size == currents.size:
            waveform = tdem.sources.PiecewiseLinearWaveform(
                times=nodes, currents=currents)
        elif config.waveform_type == "ramp_off":
            waveform = tdem.sources.RampOffWaveform()
        else:
            waveform = tdem.sources.StepOffWaveform()

        # SimPEG's 1D CircularLoop implementation only supports a central-loop
        # receiver. Ground TEM systems such as TEM2Go use a small transmitter loop
        # and an offset receiver; represent that loop by its equivalent magnetic
        # dipole moment (I * N * area) when an offset is present.
        offset = np.linalg.norm(
            np.asarray(config.receiver_location, dtype=float)[:2]
            - np.asarray(config.source_location, dtype=float)[:2]
        )
        if offset > 1e-9:
            magnetic_moment = (
                float(config.source_moment)
                if config.source_moment is not None
                else float(config.source_current)
                * max(1, int(config.source_turns))
                * np.pi
                * float(config.source_radius) ** 2
            )
            source = tdem.sources.MagDipole(
                receiver_list=receiver_list,
                location=config.source_location,
                waveform=waveform,
                moment=magnetic_moment,
                orientation=config.receiver_orientation,
            )
        else:
            source = tdem.sources.CircularLoop(
                receiver_list=receiver_list,
                location=config.source_location,
                waveform=waveform,
                current=config.source_current,
                radius=config.source_radius,
                n_turns=max(1, int(config.source_turns)),
            )
        source_list = [source]
        
        return tdem.Survey(source_list)
    
    def forward(
        self,
        conductivity: np.ndarray,
        log_input: bool = False
    ) -> np.ndarray:
        """
        Compute forward response for a given conductivity model.
        
        Args:
            conductivity: Conductivity values for each layer (S/m)
            log_input: If True, conductivity is log-transformed
            
        Returns:
            Forward response (magnetic flux density, T)
        """
        if log_input:
            sigma = np.exp(conductivity)
        else:
            sigma = np.asarray(conductivity)

        predicted = np.asarray(self.simulation.dpred(sigma), dtype=float).ravel()
        weights = getattr(self, "_gate_weights", None)
        return predicted if weights is None else weights @ predicted

    def sensitivity(self, conductivity: np.ndarray) -> np.ndarray:
        """Analytic d(response)/d(conductivity), averaged over the gate windows.

        The same reduction the forward applies has to be applied to the
        Jacobian, or the two describe different data.
        """
        jacobian = np.asarray(self.simulation.getJ(np.asarray(conductivity)),
                              dtype=float)
        weights = getattr(self, "_gate_weights", None)
        return jacobian if weights is None else weights @ jacobian
    
    def forward_with_noise(
        self,
        conductivity: np.ndarray,
        noise_level: float = 0.05,
        seed: Optional[int] = None,
        log_input: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute forward response with added Gaussian noise.
        
        Args:
            conductivity: Conductivity values for each layer (S/m)
            noise_level: Relative noise level (default 5%)
            seed: Random seed for reproducibility
            log_input: If True, conductivity is log-transformed
            
        Returns:
            Tuple of (noisy_data, clean_data, uncertainties)
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Compute clean response
        clean_data = self.forward(conductivity, log_input=log_input)
        
        # Add noise
        noise = noise_level * np.abs(clean_data) * np.random.randn(len(clean_data))
        noisy_data = clean_data + noise
        
        # Compute uncertainties
        uncertainties = noise_level * np.abs(noisy_data)
        
        return noisy_data, clean_data, uncertainties
    
    def get_times(self) -> np.ndarray:
        """Get the time channels from the survey."""
        return self.survey_config.times if hasattr(self, 'survey_config') else None
    
    @property
    def n_data(self) -> int:
        """Number of data points."""
        if hasattr(self, "survey_config"):
            return int(np.asarray(self.survey_config.times).size)
        return self.survey.nD


# ---------------------------------------------------------------------------
# create tdem survey
# ---------------------------------------------------------------------------
def create_tdem_survey(
    times: np.ndarray,
    source_radius: float = 10.0,
    source_current: float = 1.0,
    source_location: Optional[np.ndarray] = None,
    receiver_location: Optional[np.ndarray] = None,
    receiver_orientation: str = "z",
    waveform_type: str = "step_off"
) -> tdem.Survey:
    """
    Create a TDEM survey for 1D sounding.
    
    Args:
        times: Time channels (s)
        source_radius: Loop radius (m)
        source_current: Peak current (A)
        source_location: Source center [x, y, z] (m)
        receiver_location: Receiver position [x, y, z] (m)
        receiver_orientation: Measurement component ('x', 'y', 'z')
        waveform_type: Waveform type ('step_off', 'ramp_off')
        
    Returns:
        SimPEG TDEM Survey object
    """
    if source_location is None:
        source_location = np.array([0.0, 0.0, 0.0])
    if receiver_location is None:
        receiver_location = np.array([0.0, 0.0, 0.0])
    
    config = TDEMSurveyConfig(
        source_location=source_location,
        source_radius=source_radius,
        source_current=source_current,
        receiver_location=receiver_location,
        receiver_orientation=receiver_orientation,
        times=times,
        waveform_type=waveform_type
    )
    
    fwd = TDEMForwardModeling(thicknesses=np.array([1.0]), survey_config=config)
    return fwd.survey


# ---------------------------------------------------------------------------
# hydro to tdem
# ---------------------------------------------------------------------------
def simulate_tdem_sounding_from_hydro(
    water_content: np.ndarray,
    porosity: np.ndarray,
    layer_thicknesses: np.ndarray,
    sigma_w: Union[float, np.ndarray] = 0.05,
    m: Union[float, np.ndarray] = 1.5,
    n: Union[float, np.ndarray] = 2.0,
    sigma_s: Union[float, np.ndarray] = 0.0,
    times: Optional[np.ndarray] = None,
    source_radius: float = 10.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert hydrological properties to TDEM response.
    
    This function takes water content and porosity from hydrological models
    and computes the expected TDEM response using petrophysical relationships.
    
    Args:
        water_content: Water content for each layer (-)
        porosity: Porosity for each layer (-)
        layer_thicknesses: Thickness of each layer except bottom (m)
        sigma_w: Pore water conductivity (S/m)
        m: Cementation exponent
        n: Saturation exponent
        sigma_s: Surface conductivity (S/m)
        times: Time channels (s), default is logspace(-5, -2, 31)
        source_radius: Loop radius (m)
        noise_level: Relative noise level for synthetic data
        seed: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        Tuple of (noisy_data, clean_data, uncertainties, conductivity)
    """
    from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model
    
    n_layers = len(water_content)
    
    # Ensure arrays
    water_content = np.atleast_1d(water_content)
    porosity = np.atleast_1d(porosity)
    
    if np.isscalar(sigma_w):
        sigma_w = np.full(n_layers, sigma_w)
    if np.isscalar(m):
        m = np.full(n_layers, m)
    if np.isscalar(n):
        n = np.full(n_layers, n)
    if np.isscalar(sigma_s):
        sigma_s = np.full(n_layers, sigma_s)
    
    # Calculate saturation
    saturation = water_content / porosity
    
    if verbose:
        print(f"Computing conductivity for {n_layers} layers...")
    
    # Convert to conductivity using Waxman-Smits model
    resistivity = np.zeros(n_layers)
    for i in range(n_layers):
        resistivity[i] = WS_Model(
            saturation[i],
            porosity[i],
            sigma_w[i],
            m[i],
            n[i],
            sigma_s[i]
        )
    
    conductivity = 1.0 / resistivity
    
    if verbose:
        print(f"Conductivity range: {conductivity.min():.4f} - {conductivity.max():.4f} S/m")
    
    # Set up times
    if times is None:
        times = np.logspace(-5, -2, 31)
    
    # Create survey configuration
    survey_config = TDEMSurveyConfig(
        times=times,
        source_radius=source_radius
    )
    
    # Create forward modeler
    fwd = TDEMForwardModeling(
        thicknesses=layer_thicknesses,
        survey_config=survey_config
    )
    
    # Compute response
    noisy_data, clean_data, uncertainties = fwd.forward_with_noise(
        conductivity,
        noise_level=noise_level,
        seed=seed
    )
    
    if verbose:
        print(f"Generated {len(noisy_data)} data points")
    
    return noisy_data, clean_data, uncertainties, conductivity


def hydro_to_tdem(*args, **kwargs):
    """Deprecated alias for :func:`simulate_tdem_sounding_from_hydro`."""
    import warnings

    warnings.warn(
        "forward.tdem_forward.hydro_to_tdem is deprecated; use "
        "simulate_tdem_sounding_from_hydro for a single column or "
        "Hydro_modular.hydro_to_tdem for a profile. This compatibility shim is "
        "deprecated in 0.4.0 and will be removed in 0.5.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return simulate_tdem_sounding_from_hydro(*args, **kwargs)
