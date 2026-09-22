"""Temperature correction of resistivity, and its time-lapse entry point."""

import datetime as dt

import numpy as np
import pytest

from PyHydroGeophysX.petrophysics import temperature as tc


def test_the_reference_temperature_leaves_the_model_alone():
    for model in tc.MODELS:
        assert float(tc.correction_factor(25.0, model=model)) == pytest.approx(1.0)


def test_warm_ground_corrects_upward_and_cold_ground_downward():
    """A survey in warm ground reads too conductive; 25 degC is more resistive."""
    warm = float(tc.correction_factor(35.0))
    cold = float(tc.correction_factor(5.0))
    assert warm > 1.0 > cold > 0.0
    assert float(tc.to_reference(100.0, 35.0)) > 100.0
    assert float(tc.to_reference(100.0, 5.0)) < 100.0


def test_both_laws_agree_on_the_two_percent_per_degree_rule():
    assert tc.sensitivity_percent_per_degree(model="hayley") == pytest.approx(2.0, abs=0.1)
    assert tc.sensitivity_percent_per_degree(model="linear") == pytest.approx(2.5)


def test_correcting_and_uncorrecting_is_a_round_trip():
    rho = np.array([50.0, 120.0, 900.0])
    back = tc.from_reference(tc.to_reference(rho, 8.0), 8.0)
    assert back == pytest.approx(rho)


def test_an_unknown_law_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="unknown temperature model"):
        tc.correction_factor(10.0, model="whatever")


def test_the_seasonal_wave_damps_and_lags_with_depth():
    summer, winter = 200.0, 15.0
    shallow_summer = float(tc.seasonal_temperature(0.1, summer, damping_depth=2.5))
    shallow_winter = float(tc.seasonal_temperature(0.1, winter, damping_depth=2.5))
    deep_summer = float(tc.seasonal_temperature(10.0, summer, damping_depth=2.5))
    assert shallow_summer > shallow_winter                       # the wave is there
    assert abs(deep_summer - 12.0) < abs(shallow_summer - 12.0)  # and it damps out
    # Half a wavelength down, the seasons are reversed.
    reversed_depth = 2.5 * np.pi
    assert (float(tc.seasonal_temperature(reversed_depth, summer, damping_depth=2.5))
            < float(tc.seasonal_temperature(reversed_depth, winter, damping_depth=2.5)))


def test_a_measured_profile_is_held_rather_than_extrapolated():
    profile = [[1.0, 10.0], [5.0, 12.0]]
    assert float(tc.profile_temperature(0.0, profile)) == pytest.approx(10.0)
    assert float(tc.profile_temperature(50.0, profile)) == pytest.approx(12.0)
    assert float(tc.profile_temperature(3.0, profile)) == pytest.approx(11.0)


def test_a_constant_temperature_cannot_manufacture_a_change():
    """It rescales every step identically, so the time-lapse ratio is untouched."""
    rho = np.array([[100.0, 110.0], [200.0, 180.0]])
    corrected, report = tc.correct_time_lapse_models(
        rho, {"mode": "constant", "value": 10.0}, depths=[1.0, 4.0])
    assert report["applied"]
    assert corrected[:, 1] / corrected[:, 0] == pytest.approx(rho[:, 1] / rho[:, 0])


def test_the_seasonal_mode_removes_a_purely_thermal_change():
    """Ground that only warmed should read as no change once corrected."""
    depths = np.array([0.5, 1.5, 3.0])
    dates = [dt.datetime(2024, 4, 1), dt.datetime(2024, 7, 20)]
    options = {"mode": "seasonal", "mean_temperature": 12.0, "amplitude": 10.0,
               "damping_depth": 2.5, "reference": 25.0}
    field, _ = tc.temperature_field(options, depths, 2, dates=dates)
    # Synthesize the survey a constant-resistivity ground would return.
    truth = 150.0
    observed = tc.from_reference(np.full_like(field, truth), field, reference=25.0)
    corrected, report = tc.correct_time_lapse_models(observed, options, depths, dates=dates)
    assert corrected == pytest.approx(np.full_like(field, truth))
    assert report["applied"] and report["mode"] == "seasonal"


def test_the_seasonal_mode_refuses_to_guess_the_time_of_year():
    with pytest.raises(ValueError, match="seasonal"):
        tc.temperature_field({"mode": "seasonal"}, [1.0], 2, days=[0.0, 30.0])
    # Told where in the year the sequence starts, it proceeds.
    field, note = tc.temperature_field({"mode": "seasonal", "start_day": 100.0},
                                       [1.0], 2, days=[0.0, 30.0])
    assert field.shape == (1, 2) and "seasonal wave" in note


def test_the_1d_simulation_reproduces_the_analytical_damped_wave():
    """Driven by a clean annual sinusoid it must recover the textbook solution."""
    alpha = 0.06
    damping = tc.annual_damping_depth(alpha)
    days = np.arange(0.0, 4 * 365.25, 1.0)
    omega = 2.0 * np.pi / 365.25
    surface = 12.0 + 10.0 * np.sin(omega * days)
    depths = np.array([0.5, 1.0, 2.0, 3.0])
    sample = days[days >= 3 * 365.25]          # the last year, after spin-up

    simulated = tc.simulate_temperature_1d(days, surface, depths, sample,
                                           diffusivity=alpha)
    analytic = 12.0 + 10.0 * np.exp(-depths[:, None] / damping) * np.sin(
        omega * sample[None, :] - depths[:, None] / damping)
    assert np.abs(simulated - analytic).max() < 0.1

    # The two things the depth structure is for: damping and lag.
    amplitude = (simulated.max(axis=1) - simulated.min(axis=1)) / 2.0
    assert np.all(np.diff(amplitude) < 0.0)
    peak_day = sample[np.argmax(simulated, axis=1)]
    assert np.all(np.diff(peak_day) > 0.0)


def test_the_damping_depth_follows_the_diffusivity():
    assert tc.annual_damping_depth(0.06) == pytest.approx(2.64, abs=0.05)
    assert tc.annual_damping_depth(0.24) == pytest.approx(2.0 * tc.annual_damping_depth(0.06))
    with pytest.raises(ValueError):
        tc.annual_damping_depth(0.0)


def test_a_surface_record_is_read_with_dates_or_with_days(tmp_path):
    dated = tmp_path / "dated.csv"
    dated.write_text("date,temperature_C\n2024-01-01,3.5\n2024-01-02,4.0\n2024-01-03,5.5\n")
    times, values, is_dated = tc.load_surface_temperature(dated)
    assert is_dated and values == [3.5, 4.0, 5.5]
    assert times[0] == dt.datetime(2024, 1, 1)

    plain = tmp_path / "plain.csv"
    plain.write_text("0 3.5\n1 4.0\n2 5.5\n")
    times, values, is_dated = tc.load_surface_temperature(plain)
    assert not is_dated and times == [0.0, 1.0, 2.0]

    bad = tmp_path / "bad.csv"
    bad.write_text("just one column\n")
    with pytest.raises(ValueError, match="surface temperature record"):
        tc.load_surface_temperature(bad)


def test_the_surface_mode_builds_a_field_that_damps_with_depth():
    days = np.arange(0.0, 800.0, 1.0)
    surface = 10.0 + 12.0 * np.sin(2.0 * np.pi * days / 365.25)
    depths = np.array([0.2, 1.0, 6.0])
    options = {"mode": "surface", "surface_times": list(days),
               "surface_temperature": list(surface), "diffusivity": 0.06}
    # Two surveys half a year apart, on the same numeric clock as the record.
    field, note = tc.temperature_field(options, depths, 2, days=[500.0, 680.0])
    assert field.shape == (3, 2)
    assert "1-D conduction" in note and "damping depth" in note
    swing = np.abs(field[:, 0] - field[:, 1])
    assert swing[0] > swing[1] > swing[2]        # damped with depth
    assert swing[2] < 2.0                        # and nearly gone at 6 m


def test_the_surface_mode_refuses_to_line_up_a_dated_record_with_undated_surveys():
    options = {"mode": "surface",
               "surface_times": ["2024-01-01", "2024-06-01"],
               "surface_temperature": [2.0, 18.0]}
    with pytest.raises(ValueError, match="acquisition dates"):
        tc.temperature_field(options, [1.0], 2, days=[0.0, 150.0])
    # With dates on both sides it goes through.
    field, _ = tc.temperature_field(
        options, [1.0], 2, dates=[dt.datetime(2024, 2, 1), dt.datetime(2024, 5, 1)])
    assert field.shape == (1, 2)


def test_the_surface_mode_needs_a_record():
    with pytest.raises(ValueError, match="surface temperature record"):
        tc.temperature_field({"mode": "surface"}, [1.0], 2, days=[0.0, 1.0])


def test_one_temperature_per_survey_is_refused_and_points_at_the_surface_mode():
    """Applied uniformly over depth it would invent a change the ground never had."""
    with pytest.raises(ValueError, match="mode='surface'"):
        tc.temperature_field({"mode": "series", "series": [10.0, 12.0]}, [1.0, 2.0], 2)


def test_mismatched_depths_are_refused():
    with pytest.raises(ValueError, match="depths has"):
        tc.correct_time_lapse_models(np.ones((3, 2)), {"mode": "constant"}, [1.0])


def test_the_report_says_what_was_done():
    corrected, report = tc.correct_time_lapse_models(
        np.full((2, 2), 100.0), {"mode": "constant", "value": 5.0, "model": "linear",
                                 "alpha": 0.025, "reference": 25.0},
        depths=[1.0, 2.0])
    assert corrected == pytest.approx(50.0)          # 1 + 0.025 * (5 - 25) = 0.5
    assert report["reference_temperature_C"] == 25.0
    assert report["median_change_percent"] == pytest.approx(-50.0)
    assert "linear correction to 25" in report["note"]
