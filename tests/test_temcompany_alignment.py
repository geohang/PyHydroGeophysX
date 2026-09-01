"""Tests for the TEMcompany reader and the instrument model it feeds.

Each constant here was chosen by measurement against a real acquisition before
it was written down, so the tests pin behaviour that would otherwise drift back
silently: a change that reverts one shows up as a failure rather than as a few
percent in a section.

The last test runs only where a project folder is present. It checks the claim
that matters most for a reader, that the gates, values and errors it returns are
exactly those the project file holds, so it asserts equality rather than a
tolerance.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from PyHydroGeophysX.data_processing.em1d import (
    TEMCOMPANY_AUTO_SETTINGS,
    _temcompany_auto_setting,
    _temcompany_inversion_defaults,
    _temcompany_loop_corners,
    _temcompany_system,
    _temcompany_transmitter,
)
from PyHydroGeophysX.forward.tdem_forward import (
    GATE_WINDOWS,
    TDEMSurveyConfig,
    _analog_sampling,
    _gate_sampling,
    _has_instrument_model,
    _gate_integration_operator,
    _instrument_sampling,
)
from PyHydroGeophysX.forward.em1d import _tdem_config, _tdem_geometry
from PyHydroGeophysX.workflows.em1d import _station_geometry, _tdem_calibration_view

#: A gate set shaped like the instrument's: 0.2 decade wide, 0.1 decade apart.
CENTRES = np.geomspace(1.0e-5, 1.0e-4, 11)
OPENS = CENTRES / 10.0 ** 0.1
CLOSES = CENTRES * 10.0 ** 0.1

#: The two first-order corners a TEM2Go records.
CUTOFFS = (450_000.0, 800_000.0)


def _config(**overrides) -> TDEMSurveyConfig:
    settings = dict(
        times=CENTRES, gate_open=OPENS, gate_close=CLOSES,
        receiver_type="dbdt", waveform_type="step_off",
    )
    settings.update(overrides)
    return TDEMSurveyConfig(**settings)


# ---------------------------------------------------------------------------
# Gate windows
# ---------------------------------------------------------------------------
def test_centre_window_reads_the_gate_centre_and_reduces_nothing() -> None:
    """What the reference implementation does, so it needs no reduction matrix."""
    times, matrix = _gate_sampling(_config(gate_window="centre"))

    assert matrix is None
    np.testing.assert_allclose(times, CENTRES)


def test_the_tukey_window_is_what_the_instrument_records() -> None:
    """``GateShape`` names the gate window; ``GateShapePar1`` is its taper.

    Read as a Tukey window with a 0.667 tapered fraction, integrated in linear
    time and normalised by the window, this reproduces the responses stored
    alongside the data to about a percent.
    """
    from PyHydroGeophysX.forward.tdem_forward import GATE_SHAPE_NAMES

    assert GATE_SHAPE_NAMES[1] == "tukey"
    times, matrix = _gate_sampling(_config(gate_window="tukey",
                                           gate_window_par=0.667))

    per_gate = times.size // CENTRES.size
    assert matrix.shape == (CENTRES.size, times.size)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    # Sampled inside the window, in linear time, and spanning it.
    block = times[:per_gate]
    assert OPENS[0] < block.min() and block.max() < CLOSES[0]
    weights = matrix[0, :per_gate]
    assert np.all(weights >= 0.0)
    # Two thirds of the width is tapered, so the middle carries more than its
    # share of a flat window would.
    middle = (block > OPENS[0] + 0.4 * (CLOSES[0] - OPENS[0])) & (
        block < OPENS[0] + 0.6 * (CLOSES[0] - OPENS[0]))
    assert weights[middle].sum() > 0.2 * middle.sum() / per_gate


def test_the_taper_concentrates_the_window_and_keeps_it_symmetric() -> None:
    """``GateShape`` can also select a plain window, so both must be available.

    The two cannot be told apart by their weights, which are quadrature rules
    with different node counts. What separates them is the shape they describe:
    a taper pulls weight in from the edges, so the window's spread about its own
    midpoint is smaller.

    Both are symmetric in linear time, and that is worth pinning because the
    gate centre the file records is the geometric mean of open and close, which
    is not the linear midpoint. A window symmetric in linear time therefore sits
    slightly late relative to the recorded centre, and neither rule is trying to
    reproduce the centre value.
    """
    def spread(times: np.ndarray, matrix: np.ndarray, gate: int) -> float:
        block = slice(gate * (times.size // CENTRES.size),
                      (gate + 1) * (times.size // CENTRES.size))
        t, w = times[block], matrix[gate, block]
        mean = float(np.sum(w * t))
        width = float(CLOSES[gate] - OPENS[gate])
        return float(np.sqrt(np.sum(w * (t - mean) ** 2))) / width

    tukey_times, tukey = _gate_sampling(
        _config(gate_window="tukey", gate_window_par=0.667))
    square_times, square = _gate_sampling(_config(gate_window="square"))

    np.testing.assert_allclose(tukey.sum(axis=1), 1.0)
    np.testing.assert_allclose(square.sum(axis=1), 1.0)
    assert spread(tukey_times, tukey, 0) < spread(square_times, square, 0)
    # A flat window on [0, 1] has a standard deviation of 1/sqrt(12).
    assert spread(square_times, square, 0) == pytest.approx(1.0 / np.sqrt(12.0),
                                                            abs=1e-6)
    # Symmetric, so the weighted mean is the linear midpoint of the gate.
    per_gate = tukey_times.size // CENTRES.size
    midpoint = 0.5 * (OPENS[0] + CLOSES[0])
    assert float(np.sum(tukey[0, :per_gate] * tukey_times[:per_gate])) == (
        pytest.approx(midpoint, rel=1e-9))


def test_gate_integration_weights_are_pinned() -> None:
    """Unit impulses pin the interpolation and the window together.

    These are the weights for one low-moment gate on a ten-points-per-decade
    response grid. The negative edge weights are expected: they come from the
    local cubic Hermite interpolation, not from the Tukey window, which is
    positive everywhere.
    """
    raw = 10.0 ** np.arange(-8.0, -3.9, 0.1)
    operator = _gate_integration_operator(
        raw,
        np.array([5.6588373e-6]),
        np.array([4.4999995e-6]),
        np.array([7.1160994e-6]),
        "tukey", 0.667, 12,
    )

    np.testing.assert_allclose(
        operator[0, 25:31],
        [-0.00060600, -0.02263166, 0.41725682,
         0.61152065, -0.00276100, -0.00277880],
        atol=2.0e-5,
    )
    assert operator.sum() == pytest.approx(1.0, abs=2e-12)


def test_temcompany_instrument_chain_is_cached_and_compact() -> None:
    """Building one native chain twice reuses it instead of repeating setup."""
    config = _config(
        waveform_times=np.array([-2.0e-4, -1.0e-5, 0.0, 3.5e-6]),
        waveform_currents=np.array([0.0, 1.0, 1.0, 0.0]),
        waveform_period=4.0e-4,
        waveform_repetitions=3,
        gate_window="tukey",
        analog_lowpass={"first_order_cutoffs_hz": (450_000.0, 800_000.0)},
    )

    model_times, reduction = _instrument_sampling(config)
    model_times_again, reduction_again = _instrument_sampling(config)

    assert model_times is model_times_again
    assert reduction is reduction_again
    assert reduction.shape == (CENTRES.size, model_times.size)
    assert model_times.size < 400
    assert np.all(np.isfinite(reduction))


def test_temcompany_chain_reduces_the_jacobian_identically_to_the_forward() -> None:
    """The cached instrument matrix must not make inversion gradients approximate."""
    from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling

    config = TDEMSurveyConfig(
        source_location=np.array([0.0, 0.0, 0.9]),
        receiver_location=np.array([15.0, 0.0, 0.9]),
        source_moment=1.0,
        receiver_type="dbdt",
        times=np.array([5.66e-6, 7.12e-6, 1.0e-5]),
        gate_open=np.array([4.5e-6, 5.66e-6, 8.0e-6]),
        gate_close=np.array([7.12e-6, 8.95e-6, 1.25e-5]),
        gate_window="tukey",
        waveform_times=np.array([-2.0e-4, -1.0e-5, 0.0, 3.5e-6]),
        waveform_currents=np.array([0.0, 1.0, 1.0, 0.0]),
        waveform_period=4.0e-4,
        analog_lowpass={"first_order_cutoffs_hz": (450_000.0, 800_000.0)},
    )
    modeler = TDEMForwardModeling(np.array([5.0, 15.0]), config)
    conductivity = np.array([0.02, 0.01, 0.005])
    direction = np.array([0.3, -0.2, 0.1])
    step = 1.0e-3

    finite_difference = (
        modeler.forward(conductivity + step * direction)
        - modeler.forward(conductivity - step * direction)
    ) / (2.0 * step)
    analytic = modeler.sensitivity(conductivity) @ direction

    np.testing.assert_allclose(analytic, finite_difference, rtol=5e-4, atol=1e-10)


def test_gate_shape_reads_off_the_project_when_no_window_is_named() -> None:
    from PyHydroGeophysX.forward.em1d import _gate_window_name

    assert _gate_window_name({"gate_window_shape": 1.0}) == "tukey"
    assert _gate_window_name({"gate_window_shape": 2}) == "square"
    # An explicit choice wins, so a comparison does not need the file edited.
    assert _gate_window_name({"gate_window_shape": 1, "gate_window": "centre"}) == "centre"
    # A shape the mapping does not cover raises rather than guessing: falling
    # back would model a window the instrument did not use with nothing to
    # notice it. A dataset recording no shape at all is a different case.
    with pytest.raises(ValueError, match="GateShape = 7"):
        _gate_window_name({"gate_window_shape": 7})
    assert _gate_window_name({}) == "centre"


def test_an_unknown_gate_window_is_refused() -> None:
    with pytest.raises(ValueError, match="gate_window must be one of"):
        _gate_sampling(_config(gate_window="simpson_log"))
    assert "centre" in GATE_WINDOWS


# ---------------------------------------------------------------------------
# Waveform repetition
# ---------------------------------------------------------------------------
NATIVE = dict(
    waveform_times=np.array([-2.0e-4, -1.0e-5, 0.0, 3.5e-6]),
    waveform_currents=np.array([0.0, 1.0, 1.0, 0.0]),
    gate_window="tukey",
    analog_lowpass={"first_order_cutoffs_hz": CUTOFFS},
)


def _native_gates(period, repetitions):
    """Gate values for a synthetic power-law step response, native chain."""
    config = _config(waveform_period=period, waveform_repetitions=repetitions,
                     **NATIVE)
    times, reduction = _instrument_sampling(config)
    return reduction @ (times ** -1.5)


def test_repetition_subtracts_the_previous_pulse() -> None:
    """The reference sums R(t) - R(t+T) + R(t+2T) - R(t+3T).

    The first term subtracted is the largest, so switching repetition on lowers
    every gate, and it lowers the late ones most: those sit closest to the
    previous half-cycle and so see the most of it.
    """
    period = 8.0e-4
    without = np.abs(_native_gates(period, 0))
    with_terms = np.abs(_native_gates(period, 3))

    assert np.all(with_terms < without)
    change = 1.0 - with_terms / without
    assert change[-1] > change[0]
    assert change[-1] > 1.0e-3


def test_repetition_shrinks_as_the_half_period_grows() -> None:
    """A transmitter that waits longer leaves less behind."""
    reference = np.abs(_native_gates(8.0e-4, 0))
    near = 1.0 - np.abs(_native_gates(4.0e-4, 3)) / reference
    far = 1.0 - np.abs(_native_gates(6.4e-3, 3)) / reference

    assert np.all(np.abs(far) < np.abs(near))


def test_a_dataset_without_a_period_keeps_the_direct_path() -> None:
    """Repetition needs a period, and so does the native chain as a whole.

    Nothing else can supply it: the bipolar half-period is a property of the
    instrument, and a dataset that does not record one is not describing a
    TEMcompany acquisition.
    """
    assert not _has_instrument_model(_config(**NATIVE))
    assert _has_instrument_model(
        _config(waveform_period=8.0e-4, **NATIVE))


def test_an_explicitly_filterless_instrument_keeps_the_native_chain() -> None:
    """An empty filter description means no stages, not unknown electronics."""
    config = _config(
        waveform_period=8.0e-4,
        waveform_times=NATIVE["waveform_times"],
        waveform_currents=NATIVE["waveform_currents"],
        gate_window="tukey",
        analog_lowpass={},
    )
    assert _has_instrument_model(config)
    model_times, reduction = _instrument_sampling(config)
    assert reduction.shape == (CENTRES.size, model_times.size)


def test_loaded_transmitter_is_merged_but_explicit_geometry_still_wins() -> None:
    data = {
        "system": {"receiver_type": "dbdt", "tx_rx_sep": 15.0},
        "transmitter": {
            "waveform_times": NATIVE["waveform_times"],
            "waveform_currents": NATIVE["waveform_currents"],
            "waveform_period": 8.0e-4,
            "gate_windows": {
                "centre": CENTRES, "open": OPENS, "close": CLOSES,
            },
            "gate_window": "tukey",
            "analog_lowpass": {},
        },
    }
    merged = _tdem_geometry(data, {"tx_rx_sep": 14.5})
    assert merged["tx_rx_sep"] == 14.5
    assert _has_instrument_model(_tdem_config(merged, CENTRES))


def test_joint_calibration_uses_the_preview_moments_transmitter() -> None:
    low = {"times": CENTRES[:3], "response": np.ones(3),
           "transmitter": {"waveform_period": 4.0e-4}}
    high = {"times": CENTRES[3:], "response": np.ones(CENTRES.size - 3),
            "transmitter": {"waveform_period": 8.0e-4}}
    item, geometry = _tdem_calibration_view(
        {"moments": {"LM": low, "HM": high}, "system": {"receiver_type": "dbdt"}},
        {"tem_moment": "LM+HM"},
    )
    assert item is not high
    np.testing.assert_array_equal(item["times"], high["times"])
    assert geometry["waveform_period"] == pytest.approx(8.0e-4)


def test_rebinned_gate_is_not_mistaken_for_a_native_gate_by_absolute_tolerance() -> None:
    geometry = {
        "gate_windows": {
            "centre": np.array([5.0e-6]),
            "open": np.array([4.0e-6]),
            "close": np.array([6.0e-6]),
        },
    }
    config = _tdem_config(geometry, np.array([5.009e-6]))
    assert config.gate_open is None
    assert config.gate_close is None


# ---------------------------------------------------------------------------
# Analog filter resolution
# ---------------------------------------------------------------------------
def _filtered(per_decade: int):
    """Gate values for a synthetic power-law decay at one filter resolution.

    Only the dense grid changes, so the modelled grid, and therefore the
    response being filtered, is identical between calls and the two results are
    directly comparable.
    """
    config = _config(
        analog_lowpass={"first_order_cutoffs_hz": CUTOFFS},
        analog_points_per_decade=per_decade,
        analog_model_points_per_decade=40,
    )
    times, reduction = _analog_sampling(config)
    # t**-2.5 is the late-time slope of a layered dB/dt, which is what the
    # reconstruction has to follow.
    return times, reduction @ (times ** -2.5)


def test_the_analog_filter_converges_with_resolution() -> None:
    """Doubling the internal resolution must not move a gate by 0.1 percent.

    The two-region grid this replaced failed here. It was dense only until the
    filter had settled and then fell back to the receiver times themselves, so
    the first-order-hold reconstruction integrated across steps twenty times
    longer than the filter's own time constant. The lag that leaves does not die
    away with time the way the continuous operator does.
    """
    times, coarse = _filtered(150)
    times_fine, fine = _filtered(300)

    np.testing.assert_allclose(times, times_fine)
    assert np.max(np.abs(fine / coarse - 1.0)) < 1.0e-3


def test_the_filter_gain_follows_the_group_delay() -> None:
    """A causal low pass averages over the preceding time constant.

    On a decaying signal that means it reads high, by more the steeper the
    decay. To leading order the excess is ``p * kappa / t`` for a decay ``t**-p``
    and a cascade whose group delay is ``kappa``, so it falls off like ``1 / t``
    and the higher-order terms show up only where ``t`` approaches ``kappa``.
    Checking against that rather than against a stored number is what catches a
    discretisation artefact, which has no reason to follow the same law: the
    grid this replaced added 2.2 percent at 95 us where the delay allows 1.4.
    """
    _, filtered = _filtered(150)
    excess = filtered / (CENTRES ** -2.5) - 1.0
    group_delay = sum(1.0 / (2.0 * np.pi * cutoff) for cutoff in CUTOFFS)
    leading = 2.5 * group_delay / CENTRES

    assert np.all(np.diff(excess) < 0.0)              # falls away with time
    # Late, where t is 180 group delays out, the leading term is the whole of it.
    assert excess[-1] / leading[-1] == pytest.approx(1.0, abs=0.05)
    # Early, at 18 group delays, the curvature of the decay adds a few percent.
    assert 1.1 < excess[0] / leading[0] < 1.3


def test_no_filter_means_no_reduction() -> None:
    times, reduction = _analog_sampling(_config())

    assert reduction is None
    np.testing.assert_allclose(times, CENTRES)


# ---------------------------------------------------------------------------
# Settings whose stored value is overridden by its mode
# ---------------------------------------------------------------------------
def _settings_connection(inverse: dict) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE UserSettingsJson (SettingsJson TEXT)")
    connection.execute(
        "INSERT INTO UserSettingsJson VALUES (?)",
        (json.dumps({"InverseSettings": inverse}),))
    return connection


BASE_INVERSE = {
    "Nlayers": 20, "FirstLayer": 1.0, "LastDepth": 100.0,
    "LcAutoScale": True, "InversionMoment": "Both",
}


def test_auto_mode_takes_the_fallback_and_reports_the_stored_value() -> None:
    """"Auto" makes the stored number inert; reading it reports a constraint
    the survey never ran under."""
    connection = _settings_connection({
        **BASE_INVERSE,
        "LcRefDistance": 10.0, "LcRefDistanceMode": "Auto",
        "LcAutoScalePower": 1.0, "LcAutoScalePowerMode": "Auto",
        "SciMaxDistance": 150.0, "SciMaxDistanceMode": "Auto",
    })

    defaults = _temcompany_inversion_defaults(connection)

    assert defaults["reference_distance"] == TEMCOMPANY_AUTO_SETTINGS["LcRefDistance"]
    assert defaults["reference_distance_stored"] == 10.0
    assert defaults["lateral_distance_power"] == TEMCOMPANY_AUTO_SETTINGS["LcAutoScalePower"]
    assert defaults["lateral_distance_power_stored"] == 1.0
    assert defaults["sci_max_distance"] == TEMCOMPANY_AUTO_SETTINGS["SciMaxDistance"]
    assert defaults["sci_max_distance_stored"] == 150.0


def test_a_manual_mode_keeps_the_stored_number() -> None:
    connection = _settings_connection({
        **BASE_INVERSE,
        "LcRefDistance": 10.0, "LcRefDistanceMode": "Manual",
    })

    defaults = _temcompany_inversion_defaults(connection)

    assert defaults["reference_distance"] == 10.0
    # Nothing to distinguish, so nothing is reported beside it.
    assert defaults["reference_distance_stored"] is None


def test_a_missing_mode_key_keeps_the_stored_number() -> None:
    effective, stored = _temcompany_auto_setting({"LcRefDistance": 7.0},
                                                 "LcRefDistance", 10.0)
    assert (effective, stored) == (7.0, None)


# ---------------------------------------------------------------------------
# Geometry and units
# ---------------------------------------------------------------------------
SPEC = {
    "TxLoopArea": 0.3969, "TxLoopXYlength": [0.63, 0.63], "NTurnsTxLoop": 4,
    "TxLoopXYZPos": [0.0, 0.0, 0.9], "RxCoilXYZPos": [-15.0, 0.0, 0.9],
    "RxCoilAreaChA": 441.0, "LPFilter_1order": list(CUTOFFS),
    "GateShape": 1, "GateShapePar1": 0.667, "InstrumentType": "TEM2Go",
    "LM_DataFactor": 1.05, "HM_DataFactor": 1.08,
    "LMWaveformTime": [-2.0e-4, 0.0, 3.5e-6],
    "LMWaveformAmplitude": [0.0, 1.0, 0.0],
    "LMWaveformPeriod": 4.0e-4, "LM_GateTimeShift": 0.0,
    "LM_Tx_TargetCurrent": 1.0,
}


class _Row(dict):
    """A stand-in for ``sqlite3.Row``, which raises rather than returning None."""

    def __getitem__(self, key):
        if key not in self:
            raise IndexError(key)
        return super().__getitem__(key)


def test_the_station_distance_wins_over_the_nominal_layout() -> None:
    row = _Row(RxTxDistance=14.734, RxCoilHeight=0.9, TxCoilHeight=0.9)

    system = _temcompany_system(SPEC, row)

    assert system["tx_rx_sep"] == pytest.approx(14.734)
    assert system["tx_rx_sep_nominal"] == pytest.approx(15.0)


def test_a_missing_or_zero_station_distance_falls_back_to_the_layout() -> None:
    """Zero is how a failed measurement is stored, not a coincident pair."""
    assert _temcompany_system(SPEC, _Row(RxTxDistance=0.0))["tx_rx_sep"] == pytest.approx(15.0)
    assert _temcompany_system(SPEC, None)["tx_rx_sep"] == pytest.approx(15.0)


def test_the_data_factor_is_recorded_and_never_applied() -> None:
    """Measured: the inversion inputs stored in the project equal the stored
    voltages exactly while the factor reads 1.05, so it is already applied."""
    system = _temcompany_system(SPEC, None)
    transmitter = _temcompany_transmitter(SPEC, "LM")

    assert system["data_scale"] == 1.0
    assert system["source_moment"] == 1.0
    assert transmitter["data_factor"] == pytest.approx(1.05)


def test_the_transmitter_reports_the_period_and_the_gate_shape() -> None:
    transmitter = _temcompany_transmitter(SPEC, "LM")

    assert transmitter["waveform_period"] == pytest.approx(4.0e-4)
    assert transmitter["gate_window_par"] == pytest.approx(0.667)
    assert transmitter["gate_time_shift"] == 0.0


def test_loop_corners_describe_the_recorded_rectangle() -> None:
    x, y = _temcompany_loop_corners(np.array([0.0, 0.0, 0.9]), np.array([0.63, 0.63]))

    np.testing.assert_allclose(sorted(set(np.round(x, 6))), [-0.315, 0.315])
    np.testing.assert_allclose(sorted(set(np.round(y, 6))), [-0.315, 0.315])
    assert _temcompany_loop_corners(np.zeros(3), np.array([0.0])) == (None, None)


def test_per_station_geometry_can_be_switched_off() -> None:
    geom = {"tx_rx_sep": 15.0, "per_station_geometry": False}
    data = {"system": {"tx_rx_sep": 12.4}}

    assert _station_geometry(geom, data)["tx_rx_sep"] == 15.0


def test_per_station_geometry_is_on_and_bins_the_distance() -> None:
    """On by default: the project records it per station and it costs nothing.

    Replacing the measured column with the nominal 15 m moves one survey's
    low-moment response by 1.4 percent at the median and 18 percent at its
    worst gate. Binning keeps the operator cache useful; a walking survey records
    794 distinct distances over 929 stations.
    """
    geom = {"tx_rx_sep": 15.0, "height": 0.9}
    data = {"system": {"tx_rx_sep": 12.44, "rx_height": 0.8, "tx_height": 0.85}}

    updated = _station_geometry(geom, data)

    assert updated["tx_rx_sep"] == pytest.approx(12.5)      # quarter-metre bin
    assert updated["rx_height"] == pytest.approx(0.8)
    assert geom["tx_rx_sep"] == 15.0          # the caller's dict is untouched

    exact = _station_geometry({**geom, "tx_rx_sep_bin": 0.0}, data)
    assert exact["tx_rx_sep"] == pytest.approx(12.44)


def test_a_caller_supplied_height_reaches_the_forward() -> None:
    """Setting ``height`` alone would be silently ignored.

    The forward reads ``rx_height`` and ``tx_height`` in preference, and the
    station dictionary carries both, so a caller passing its own heights array
    to a line inversion would have seen the file's heights modelled instead.
    """
    from PyHydroGeophysX.forward.em1d import _tdem_config
    from PyHydroGeophysX.workflows.em1d import _with_sensor_height

    station = {"height": 0.9, "rx_height": 0.9, "tx_height": 0.9,
               "tx_rx_sep": 15.0, "receiver_type": "dbdt"}
    updated = _with_sensor_height(station, 30.0)

    assert (updated["height"], updated["rx_height"], updated["tx_height"]) == (
        30.0, 30.0, 30.0)
    config = _tdem_config(updated, np.geomspace(1e-5, 1e-4, 5))
    assert config.source_location[2] == pytest.approx(30.0)
    assert config.receiver_location[2] == pytest.approx(30.0)
    # A height that is not a number leaves the geometry alone rather than
    # writing NaN into the source location.
    assert _with_sensor_height(station, float("nan")) == station
    assert _with_sensor_height(station, None) == station


def test_the_repetition_switch_can_be_turned_off() -> None:
    """Whether the transmitter train is superposed is not in the project file,
    so the pipeline exposes it rather than reading it."""
    from PyHydroGeophysX.forward.em1d import _tdem_config

    base = {"waveform_period": 8.0e-4, "receiver_type": "dbdt"}
    gates = np.geomspace(1e-5, 1e-4, 5)

    assert _tdem_config(base, gates).waveform_repetitions == 3
    assert _tdem_config({**base, "waveform_repeat": False},
                        gates).waveform_repetitions == 0


# ---------------------------------------------------------------------------
# Against a real project, when one is present
# ---------------------------------------------------------------------------
def _project_folder() -> Path | None:
    """A TEMcompany project to check against, named by the environment."""
    candidate = os.environ.get("PYHYDRO_TEMCOMPANY_PROJECT")
    if not candidate:
        return None
    path = Path(candidate)
    return path if path.exists() else None


def test_the_current_inversion_is_the_one_the_project_points_at() -> None:
    """A project can hold several runs, and row count is a poor way to pick.

    ``InverseSettings.LastInversionName`` is the project's own answer: it is the
    run the application would show. Ordering by row count is only the fallback
    for a project that names none.
    """
    from PyHydroGeophysX.data_processing import temcompany_reference as reference

    database = tmp_project_with_two_inversions()
    assert reference.reference_inversion_names(database)[0] == "LCI_current"


def tmp_project_with_two_inversions() -> Path:
    """A project holding a big old run and a small current one."""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "project.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE InversionModel (InversionName TEXT)")
    connection.executemany("INSERT INTO InversionModel VALUES (?)",
                           [("LCI_old",)] * 5 + [("LCI_current",)] * 2)
    connection.execute("CREATE TABLE UserSettingsJson (SettingsJson TEXT)")
    connection.execute(
        "INSERT INTO UserSettingsJson VALUES (?)",
        (json.dumps({"InverseSettings": {"LastInversionName": "LCI_current"}}),))
    connection.commit()
    connection.close()
    return path


def test_row_count_orders_the_runs_when_the_project_names_none() -> None:
    import tempfile

    from PyHydroGeophysX.data_processing import temcompany_reference as reference

    path = Path(tempfile.mkdtemp()) / "project.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE InversionModel (InversionName TEXT)")
    connection.executemany("INSERT INTO InversionModel VALUES (?)",
                           [("LCI_big",)] * 5 + [("LCI_small",)] * 2)
    connection.commit()
    connection.close()

    assert reference.reference_inversion_names(path)[0] == "LCI_big"


@pytest.mark.skipif(_project_folder() is None,
                    reason="set PYHYDRO_TEMCOMPANY_PROJECT to a project folder")
def test_the_reader_reproduces_the_stored_inversion_input_exactly() -> None:
    """Gates, values and errors, all three identical for every station.

    Not a tolerance: ``InputData`` in the project is the stored ``VoltageValues``
    unchanged and its ``InputSTD`` is the stored error column unchanged, so any
    scaling, sign change or extra gate test on our side shows up here.
    """
    from PyHydroGeophysX.data_processing.temcompany_reference import (
        has_reference_models,
        load_reference_models,
    )

    project = _project_folder()
    if not has_reference_models(project):
        # A raw acquisition folder carries the same protocol and line files as an
        # imported project but no database, so there is nothing to compare
        # against rather than a disagreement to report.
        pytest.skip(f"{project} carries no TEMcompany inversion")

    reference = load_reference_models(project)
    assert reference["n_stations"] > 0

    checked = 0
    for station in reference["stations"]:
        for block in station["moments"].values():
            index = block["spec_gate_indices"]
            if index.size != block["times"].size:
                continue
            np.testing.assert_array_equal(
                block["observed"], block["stored_response"][index])
            np.testing.assert_array_equal(
                block["relative_std"], block["stored_std"][index])
            flags = block["stored_flags"] > 0
            finite = (np.isfinite(block["stored_response"])
                      & (np.abs(block["stored_response"]) < 9_000.0))
            selected = np.zeros(flags.size, dtype=bool)
            selected[index] = True
            np.testing.assert_array_equal(selected, flags & finite)
            checked += 1
    assert checked > 0
