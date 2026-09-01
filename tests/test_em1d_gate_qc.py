"""Tests for the 1D EM data-QC and model-bound options.

Three knobs, each guarding a way a sparse ground TDEM inversion goes wrong: the
gate-rejection mode decides whether one bad gate costs the sounding its depth,
the stack-error floor stops an accidentally quiet gate from deciding the model,
and the resistivity bounds cap how far an unresolved deep layer can rail.
"""

import json
import sqlite3

import numpy as np
import pytest

import PyHydroGeophysX.inversion.em1d as inversion_em1d

from PyHydroGeophysX.data_processing.em1d import (
    GATE_REJECTION_MODES,
    TEMCOMPANY_UNIFORM_ERROR,
    _check_gate_rejection,
    _temcompany_baked_uniform_error,
    _temcompany_inversion_defaults,
    _temcompany_stored_thicknesses,
    _temcompany_uniform_error,
    _temcompany_valid_channels,
    suggest_layer_grid,
)
from PyHydroGeophysX.inversion.em1d_lci import lateral_edges
from PyHydroGeophysX.inversion.em1d import (
    _log_resistivity_bounds,
    _occam_with_optional_rejection,
    _tdem_uncertainty,
)
from PyHydroGeophysX.workflows.em1d import _scale_bounds


# A five-gate decay whose third gate is too noisy for a 0.30 cut. Gates 4 and 5
# are clean, which is the case the two modes disagree about.
TIMES = np.array([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4])
RESPONSE = np.array([1e-7, 5e-8, 2e-8, 8e-9, 3e-9])
STD = np.array([0.05, 0.08, 0.42, 0.12, 0.18])


def test_truncate_drops_the_bad_gate_and_everything_after_it() -> None:
    times, response, std = _temcompany_valid_channels(
        TIMES, RESPONSE, STD, use_flags=False, max_relative_std=0.30,
        gate_rejection="truncate")

    assert times.size == 2
    np.testing.assert_allclose(times, TIMES[:2])
    np.testing.assert_allclose(response, RESPONSE[:2])
    np.testing.assert_allclose(std, STD[:2])


def test_individual_keeps_the_clean_gates_after_a_bad_one() -> None:
    times, response, std = _temcompany_valid_channels(
        TIMES, RESPONSE, STD, use_flags=False, max_relative_std=0.30,
        gate_rejection="individual")

    assert times.size == 4
    np.testing.assert_allclose(times, TIMES[[0, 1, 3, 4]])
    np.testing.assert_allclose(response, RESPONSE[[0, 1, 3, 4]])
    np.testing.assert_allclose(std, STD[[0, 1, 3, 4]])


def test_a_negative_gate_costs_its_own_gate_in_either_mode() -> None:
    """A sign reversal never truncates, because it says nothing about later gates.

    Every stack error here is well inside the cut, so the sign is the only thing
    that can remove a gate. Truncation is an argument about the decay reaching
    the noise floor, and an early-time reversal is not that argument: on an
    offset-loop system the reversals cluster in the earliest gates, so carrying
    the rest of the sounding away with one costs every gate the station had.
    """
    response = RESPONSE.copy()
    response[2] = -2e-8
    quiet = np.full(TIMES.size, 0.05)

    truncated, _, _ = _temcompany_valid_channels(
        TIMES, response, quiet, use_flags=False, max_relative_std=0.30,
        gate_rejection="truncate")
    individual, _, _ = _temcompany_valid_channels(
        TIMES, response, quiet, use_flags=False, max_relative_std=0.30,
        gate_rejection="individual")

    np.testing.assert_allclose(truncated, TIMES[[0, 1, 3, 4]])
    np.testing.assert_allclose(individual, truncated)


def test_a_reversed_first_gate_does_not_empty_the_sounding() -> None:
    """The failure the coupled rule caused, stated as its own case."""
    response = RESPONSE.copy()
    response[0] = -1e-7
    quiet = np.full(TIMES.size, 0.05)

    kept, _, _ = _temcompany_valid_channels(
        TIMES, response, quiet, use_flags=False, max_relative_std=0.30,
        gate_rejection="truncate")

    np.testing.assert_allclose(kept, TIMES[1:])


def test_the_noise_cut_still_truncates_alongside_a_reversal() -> None:
    """One rule per test: the noise half keeps its tail-cutting behaviour."""
    response = RESPONSE.copy()
    response[0] = -1e-7                          # reversed, removed alone
    std = np.array([0.05, 0.08, 0.42, 0.12, 0.18])   # gate 3 is noisy

    kept, _, _ = _temcompany_valid_channels(
        TIMES, response, std, use_flags=False, max_relative_std=0.30,
        gate_rejection="truncate")

    np.testing.assert_allclose(kept, TIMES[[1]])


def test_truncate_is_the_default_so_existing_callers_are_unchanged() -> None:
    explicit, _, _ = _temcompany_valid_channels(
        TIMES, RESPONSE, STD, use_flags=False, max_relative_std=0.30,
        gate_rejection="truncate")
    implied, _, _ = _temcompany_valid_channels(
        TIMES, RESPONSE, STD, use_flags=False, max_relative_std=0.30)

    np.testing.assert_allclose(explicit, implied)


@pytest.mark.parametrize("mode", GATE_REJECTION_MODES)
def test_every_advertised_mode_is_accepted(mode: str) -> None:
    assert _check_gate_rejection(mode.upper()) == mode


def test_an_unknown_mode_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="truncate, individual"):
        _check_gate_rejection("tail")


def test_the_error_floor_lifts_a_quiet_gate_without_touching_a_noisy_one() -> None:
    """A gate that stacked unusually quietly must not outweigh the rest of the sounding."""
    observed = np.array([1e-7, 1e-7])
    item = {"relative_std": np.array([0.001, 0.20])}

    raw = _tdem_uncertainty(observed, item, rel=0.0, floor=0.0)
    floored = _tdem_uncertainty(observed, item, rel=0.0, floor=0.0, min_rel=0.03)

    assert raw[1] / raw[0] == pytest.approx(200.0)
    assert floored[0] == pytest.approx(0.03 * 1e-7)
    assert floored[1] == pytest.approx(raw[1])          # the noisy gate is untouched
    assert floored[1] / floored[0] == pytest.approx(20.0 / 3.0)


def test_the_error_ceiling_caps_a_noisy_gate() -> None:
    observed = np.array([1e-7])
    item = {"relative_std": np.array([0.42])}

    capped = _tdem_uncertainty(observed, item, rel=0.0, floor=0.0, max_rel=0.25)

    assert capped[0] == pytest.approx(0.25 * 1e-7)


def test_independent_occam_rejection_honours_the_eighty_percent_floor(
    monkeypatch,
) -> None:
    """Independent 1D uses the same fit/drop/refit contract as the line solver."""
    calls = []

    def fake_occam(forward, observed, uncertainty, n_layers, inv, log, jacobian):
        calls.append(np.asarray(observed).copy())
        assert np.asarray(forward(np.ones(n_layers))).size == observed.size
        assert np.asarray(jacobian(np.ones(n_layers))).shape == (
            observed.size, n_layers)
        return np.full(n_layers, 100.0), 1.0, 2, [1.0]

    monkeypatch.setattr(inversion_em1d, "_occam_1d", fake_occam)
    predicted = np.array([0.0, 0.0, 0.0, 0.0, 10.0])
    forward = lambda sigma: predicted
    jacobian = lambda sigma: np.ones((predicted.size, 2))
    result = _occam_with_optional_rejection(
        forward, np.zeros(5), np.ones(5), 2,
        {
            "reject_outliers": True,
            "outlier_threshold": 3.0,
            "outlier_passes": 2,
            "min_data_fraction": 0.8,
            "min_gates_per_sounding": 3,
        },
        lambda message: None,
        jacobian,
    )

    _, _, _, _, fit_mask, info = result
    np.testing.assert_array_equal(fit_mask, [True, True, True, True, False])
    assert [item.size for item in calls] == [5, 4]
    assert info["kept"] == 4
    assert info["dropped"] == 1
    assert info["floor"] == 4


def test_a_gate_with_no_recorded_error_is_not_invented_one() -> None:
    """0.0 in the column means "not recorded", so the floor must leave it alone."""
    observed = np.array([1e-7, 1e-7])
    item = {"relative_std": np.array([0.0, 0.10])}

    floored = _tdem_uncertainty(observed, item, rel=0.06, floor=0.0, min_rel=0.03)

    assert floored[0] == pytest.approx(0.06 * 1e-7)     # rel_error is the whole budget
    assert floored[1] == pytest.approx(np.hypot(0.10, 0.06) * 1e-7)


def test_resistivity_bounds_default_to_the_historical_pair() -> None:
    assert _log_resistivity_bounds({}) == (0.0, 5.0)
    assert _log_resistivity_bounds({"rho_min": 1.0, "rho_max": 1e4}) == (0.0, 4.0)


@pytest.mark.parametrize("pair", [
    {"rho_min": 0.0, "rho_max": 1e4},       # a zero lower bound has no log10
    {"rho_min": 1e4, "rho_max": 1.0},       # inverted
    {"rho_min": 100.0, "rho_max": 100.0},   # empty box
])
def test_a_degenerate_resistivity_box_is_refused(pair: dict) -> None:
    """A bad bound has to raise rather than quietly become the default."""
    with pytest.raises(ValueError, match="rho_min"):
        _log_resistivity_bounds(pair)


def test_an_explicit_none_bound_falls_back_to_the_default() -> None:
    assert _log_resistivity_bounds({"rho_min": None, "rho_max": None}) == (0.0, 5.0)


def test_scale_bounds_default_to_four_decades_either_way() -> None:
    assert _scale_bounds({}) == (1e-4, 1e4)
    assert _scale_bounds({"scale_bounds": None}) == (1e-4, 1e4)
    assert _scale_bounds({"scale_bounds": (0.5, 2.0)}) == (0.5, 2.0)
    assert _scale_bounds({"scale_bounds": [1.0, 1.0]}) == (1.0, 1.0)


@pytest.mark.parametrize("pair", [(0.0, 10.0), (-1.0, 10.0), (10.0, 1.0)])
def test_a_degenerate_scale_range_is_refused(pair: tuple) -> None:
    with pytest.raises(ValueError, match="scale_bounds"):
        _scale_bounds({"scale_bounds": pair})


def test_a_scale_range_of_the_wrong_length_is_refused() -> None:
    """A three-value list used to lose its tail silently to a [:2] slice."""
    with pytest.raises(ValueError, match="exactly two"):
        _scale_bounds({"scale_bounds": (0.5, 1.0, 2.0)})


def test_the_sign_test_and_the_noise_test_are_independent() -> None:
    """A negative gate is not a noisy gate, and the two need separate answers.

    An offset-loop system genuinely reverses sign at early time, so condemning a
    gate for its sign discards real signal. Whether it was measured repeatably is
    a different question, answered by its stack error.
    """
    times = np.array([1e-5, 2e-5, 4e-5, 8e-5])
    response = np.array([-1e-7, 5e-8, 2e-8, 8e-9])   # the first gate is reversed
    std = np.array([0.04, 0.05, 0.06, 0.42])         # the last is genuinely noisy

    strict, _, _ = _temcompany_valid_channels(
        times, response, std, use_flags=False, max_relative_std=0.30,
        gate_rejection="individual", reject_negative=True)
    signed, _, _ = _temcompany_valid_channels(
        times, response, std, use_flags=False, max_relative_std=0.30,
        gate_rejection="individual", reject_negative=False)

    # The noisy gate goes either way; only the reversed one is in dispute.
    np.testing.assert_allclose(strict, times[[1, 2]])
    np.testing.assert_allclose(signed, times[[0, 1, 2]])


def test_rejecting_negatives_stays_the_default() -> None:
    """Existing callers must not silently change behaviour."""
    times = np.array([1e-5, 2e-5])
    response = np.array([-1e-7, 5e-8])
    std = np.array([0.04, 0.05])

    implied, _, _ = _temcompany_valid_channels(
        times, response, std, use_flags=False, max_relative_std=0.30,
        gate_rejection="individual")

    np.testing.assert_allclose(implied, times[[1]])


def test_a_baked_in_uniform_error_is_not_added_twice() -> None:
    """A file whose stored error already carries the uniform term gets it once.

    ``rel`` states the size of the uniform term, so a reader that reports the
    term is already inside the recorded value leaves the total unchanged. The
    old rule squared it in a second time and inflated every error bar.
    """
    observed = np.array([1e-7])
    stack, uniform = 0.10, 0.03
    stored = float(np.hypot(stack, uniform))
    item = {"relative_std": np.array([stored]), "uniform_error": uniform}

    honest = _tdem_uncertainty(observed, item, rel=uniform, floor=0.0)
    doubled = _tdem_uncertainty(observed, {"relative_std": np.array([stored])},
                                rel=uniform, floor=0.0)

    assert honest[0] == pytest.approx(stored * 1e-7)
    assert doubled[0] == pytest.approx(np.hypot(stored, uniform) * 1e-7)
    assert doubled[0] > honest[0]


def test_asking_for_more_uniform_error_than_is_baked_in_adds_the_shortfall() -> None:
    observed = np.array([1e-7])
    stack, uniform = 0.10, 0.03
    stored = float(np.hypot(stack, uniform))
    item = {"relative_std": np.array([stored]), "uniform_error": uniform}

    total = _tdem_uncertainty(observed, item, rel=0.05, floor=0.0)

    # The 5% budget already holds 3%; only the remaining 4% joins in quadrature.
    assert total[0] == pytest.approx(np.hypot(stored, 0.04) * 1e-7)
    # And the result is what a 5% budget on the bare stack error would give.
    assert total[0] == pytest.approx(np.hypot(stack, 0.05) * 1e-7)


def test_asking_for_less_than_is_baked_in_never_shrinks_the_error() -> None:
    """The stored value is a measurement, so a smaller budget cannot undo it."""
    observed = np.array([1e-7])
    item = {"relative_std": np.array([0.10]), "uniform_error": 0.03}

    assert _tdem_uncertainty(observed, item, rel=0.0, floor=0.0)[0] == pytest.approx(1e-8)
    assert _tdem_uncertainty(observed, item, rel=0.01, floor=0.0)[0] == pytest.approx(1e-8)


def test_a_gate_with_no_recorded_error_still_gets_the_whole_budget() -> None:
    """Nothing is baked into a gate whose error was never recorded."""
    observed = np.array([1e-7, 1e-7])
    item = {"relative_std": np.array([0.0, 0.10]), "uniform_error": 0.03}

    total = _tdem_uncertainty(observed, item, rel=0.05, floor=0.0)

    assert total[0] == pytest.approx(0.05 * 1e-7)
    assert total[1] == pytest.approx(np.hypot(0.10, 0.04) * 1e-7)


def test_a_file_that_declares_nothing_behaves_exactly_as_before() -> None:
    observed = np.array([1e-7, 2e-7])
    stored = np.array([0.08, 0.12])

    plain = _tdem_uncertainty(observed, {"relative_std": stored}, rel=0.05, floor=0.0)
    zeroed = _tdem_uncertainty(observed, {"relative_std": stored, "uniform_error": 0.0},
                               rel=0.05, floor=0.0)

    np.testing.assert_allclose(plain, np.hypot(stored, 0.05) * np.abs(observed))
    np.testing.assert_allclose(plain, zeroed)


def test_a_text_export_is_marked_only_on_positive_evidence() -> None:
    """A column is asked whether it carries the term; silence means no."""
    floor = 0.03
    stacked = np.concatenate([[floor], np.linspace(0.031, 0.4, 20)])
    raw = np.linspace(0.004, 0.4, 21)
    clean = np.full(20, 0.10)                    # no value near the floor

    assert _temcompany_baked_uniform_error(stacked) == pytest.approx(floor)
    assert _temcompany_baked_uniform_error(raw) == 0.0
    assert _temcompany_baked_uniform_error(clean) == 0.0
    assert _temcompany_baked_uniform_error(None) == 0.0
    assert _temcompany_baked_uniform_error(np.array([floor, 0.1])) == 0.0   # too few


def _model_table(rows) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE InversionModel "
                "(InversionName TEXT, Thickness TEXT)")
    con.executemany("INSERT INTO InversionModel VALUES (?, ?)", rows)
    return con


def test_the_stored_layer_grid_is_read_rather_than_rebuilt() -> None:
    """Reading avoids reproducing a rounding rule the file never states."""
    stored = [1.0, 1.16, 1.35, 1.57, 1.82]
    con = _model_table([("LCI_1", json.dumps(stored))])

    got = _temcompany_stored_thicknesses(con, "LCI_1", 6)

    np.testing.assert_array_equal(got, stored)


def test_the_named_inversion_wins_over_the_first_row() -> None:
    con = _model_table([("LCI_1", json.dumps([1.0, 2.0])),
                        ("LCI_2", json.dumps([3.0, 4.0]))])

    np.testing.assert_array_equal(_temcompany_stored_thicknesses(con, "LCI_2", 3),
                                  [3.0, 4.0])
    # No name, or a name that is not there, falls back to whatever is stored.
    np.testing.assert_array_equal(_temcompany_stored_thicknesses(con, None, 3),
                                  [1.0, 2.0])
    np.testing.assert_array_equal(_temcompany_stored_thicknesses(con, "LCI_9", 3),
                                  [1.0, 2.0])


@pytest.mark.parametrize("stored", [
    [1.0, 2.0, 3.0],                # one too many for a 3-layer model
    [1.0],                          # one too few
    [1.0, 0.0],                     # a zero-thickness layer
    [1.0, -2.0],                    # negative
    [1.0, float("nan")],
])
def test_a_grid_that_does_not_describe_the_model_is_refused(stored: list) -> None:
    """Refusing sends the caller to the rebuilt grid instead of a wrong one."""
    con = _model_table([("LCI_1", json.dumps(stored))])

    assert _temcompany_stored_thicknesses(con, "LCI_1", 3) is None


def test_a_project_with_no_stored_model_falls_back() -> None:
    empty = sqlite3.connect(":memory:")
    assert _temcompany_stored_thicknesses(empty, "LCI_1", 3) is None

    con = _model_table([("LCI_1", "")])
    assert _temcompany_stored_thicknesses(con, "LCI_1", 3) is None

    con = _model_table([("LCI_1", "not json")])
    assert _temcompany_stored_thicknesses(con, "LCI_1", 3) is None


def test_a_single_layer_model_has_no_grid_to_read() -> None:
    con = _model_table([("LCI_1", json.dumps([1.0]))])
    assert _temcompany_stored_thicknesses(con, "LCI_1", 1) is None


def test_the_uniform_term_is_taken_from_the_protocol_when_it_states_one() -> None:
    """A survey is free to have been acquired with something other than 3%."""
    assert _temcompany_uniform_error({"uniform_std": 0.05}) == pytest.approx(0.05)
    assert _temcompany_uniform_error({"uniform_std": "0.02"}) == pytest.approx(0.02)


@pytest.mark.parametrize("protocol", [
    None,
    {},                                  # no protocol file travelled with it
    {"uniform_std": None},
    {"uniform_std": "not a number"},
    {"uniform_std": 0.0},                # silence is not a confirmed zero
    {"uniform_std": 1.0},                # not an error budget
    {"uniform_std": -0.03},
])
def test_a_protocol_that_states_nothing_usable_falls_back(protocol) -> None:
    assert _temcompany_uniform_error(protocol) == TEMCOMPANY_UNIFORM_ERROR


def test_the_default_grid_deepens_with_the_gate_range() -> None:
    """A survey that recorded to milliseconds sees deeper than one that stopped
    at tens of microseconds, so a fixed grid is wrong at one end or the other."""
    shallow = suggest_layer_grid(np.geomspace(5e-6, 4e-4, 20))
    deep = suggest_layer_grid(np.geomspace(5e-6, 4e-3, 24))

    # Depth follows sqrt(t), so a decade later is sqrt(10) deeper.
    assert deep["last_depth"] / shallow["last_depth"] == pytest.approx(
        np.sqrt(10.0), rel=1e-3)
    assert deep["min_thickness"] > shallow["min_thickness"]


def test_the_default_grid_clears_the_depth_this_instrument_resolves() -> None:
    """Calibration check: measured over three surveys, the deepest DOI reached
    114 m at the 90th percentile on a gate range ending at 3.7e-4 s."""
    grid = suggest_layer_grid(np.geomspace(5e-6, 3.701e-4, 25))

    assert grid["last_depth"] > 114.0
    assert grid["last_depth"] < 3 * 114.0          # deep, but not absurdly so


def test_the_default_grid_is_consistent_with_itself() -> None:
    grid = suggest_layer_grid(np.geomspace(1e-5, 1e-3, 20), n_layers=20)
    thick = np.asarray(grid["layer_thicknesses"], dtype=float)

    assert thick.size == grid["n_layers"] - 1
    assert thick[0] == pytest.approx(grid["min_thickness"])
    assert thick[-1] == pytest.approx(grid["max_thickness"])
    assert thick.sum() == pytest.approx(grid["last_depth"])
    assert np.all(np.diff(thick) > 0)              # thickens with depth


def test_the_first_layer_never_goes_below_a_metre() -> None:
    """A very early gate range would otherwise ask for centimetre layers that
    no TDEM sounding can resolve."""
    grid = suggest_layer_grid(np.geomspace(1e-7, 1e-6, 8))

    assert grid["min_thickness"] == pytest.approx(1.0)


def test_a_grid_cannot_be_suggested_from_nothing() -> None:
    for empty in ([], [0.0, -1.0], [np.nan, np.inf]):
        with pytest.raises(ValueError, match="positive gate time"):
            suggest_layer_grid(np.asarray(empty, dtype=float))


def test_the_lateral_tie_falls_off_at_the_power_it_is_given() -> None:
    """The penalty, which is the square of the weight, is what carries the power."""
    positions = [0.0, 10.0, 30.0, 70.0]          # gaps of 10, 20 and 40 m

    for power in (0.0, 0.75, 1.0, 2.0):
        edges = lateral_edges(positions, reference_distance=10.0,
                              distance_power=power)
        penalties = [w * w for _, _, w in edges]
        expected = [(10.0 / gap) ** power for gap in (10.0, 20.0, 40.0)]
        np.testing.assert_allclose(penalties, expected, rtol=1e-12)


def test_the_default_power_keeps_the_previous_weights() -> None:
    """A power of one is the sqrt(ref/d) rule this replaced."""
    positions = [0.0, 10.0, 30.0, 70.0]
    edges = lateral_edges(positions, reference_distance=10.0)

    np.testing.assert_allclose(
        [w for _, _, w in edges],
        [np.sqrt(10.0 / gap) for gap in (10.0, 20.0, 40.0)], rtol=1e-12)


def test_a_pair_closer_than_the_reference_is_not_tied_harder() -> None:
    edges = lateral_edges([0.0, 2.0], reference_distance=10.0, distance_power=1.0)

    assert edges[0][2] == pytest.approx(1.0)


def test_the_auto_scale_switch_reads_as_a_distance_power_not_a_layer_count() -> None:
    """The switch and its exponent both concern the separation between
    soundings, so a 20-layer model must not multiply the lateral weight by
    sqrt(20). Doing so tied neighbours 4.5 times too tightly."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE UserSettingsJson (SettingsJson TEXT)")
    settings = {"InverseSettings": {
        "Nlayers": 20, "FirstLayer": 1.0, "LastDepth": 100.0,
        "LcAutoScale": True, "LcAutoScalePower": 1.0, "LcRefDistance": 10.0}}
    con.execute("INSERT INTO UserSettingsJson VALUES (?)", (json.dumps(settings),))

    got = _temcompany_inversion_defaults(con)

    assert got["lateral_weight_scale"] == 1.0
    assert got["lateral_distance_power"] == pytest.approx(1.0)
    assert got["reference_distance"] == pytest.approx(10.0)


def test_the_auto_scale_switch_off_means_no_fall_off() -> None:
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE UserSettingsJson (SettingsJson TEXT)")
    settings = {"InverseSettings": {
        "Nlayers": 20, "FirstLayer": 1.0, "LastDepth": 100.0,
        "LcAutoScale": False, "LcAutoScalePower": 0.75}}
    con.execute("INSERT INTO UserSettingsJson VALUES (?)", (json.dumps(settings),))

    got = _temcompany_inversion_defaults(con)

    assert got["lateral_distance_power"] == 0.0
    assert got["lateral_weight_scale"] == 1.0
