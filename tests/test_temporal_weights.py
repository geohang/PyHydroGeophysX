"""Weighting of the temporal constraint by the interval between surveys."""

import numpy as np
import pytest

from PyHydroGeophysX.inversion.temporal_weights import temporal_weights


def test_even_sampling_is_left_exactly_alone():
    """The whole point of normalizing by the median: alpha keeps its meaning."""
    weights, report = temporal_weights([0.0, 1.0, 2.0, 3.0])
    assert weights == pytest.approx(np.ones(3))
    assert report["applied"] and "sampling is even" in report["note"]


def test_the_unit_of_the_times_does_not_matter():
    days = temporal_weights([0.0, 1.0, 3.0])[0]
    hours = temporal_weights([0.0, 24.0, 72.0])[0]
    assert days == pytest.approx(hours)


def test_a_long_gap_is_constrained_less_than_a_short_one():
    # One day, then three: the second pair spans three times as much time.
    weights, report = temporal_weights([0.0, 1.0, 4.0])
    assert weights[0] > weights[1]
    # The penalty is on the rate, so the weight is inversely proportional to the
    # gap - and normalized by the median, which here is the mean of the two.
    assert weights[0] / weights[1] == pytest.approx(3.0)
    assert report["capped_pairs"] == 0        # nothing clipped at the default 10x
    assert report["weight_range"][1] > 1.0 > report["weight_range"][0]


def test_hourly_and_monthly_in_one_series_is_capped_not_ignored():
    """The direction stays right even where the raw ratio is absurd."""
    weights, report = temporal_weights([0.0, 1 / 24, 2 / 24, 30.0])
    assert weights[0] == weights[1] > weights[2]
    assert report["capped_pairs"] > 0


def test_extreme_ratios_are_capped():
    times = [0.0, 1 / 1440, 30.0, 60.0]       # a minute, then months
    weights, report = temporal_weights(times, limit=10.0)
    assert report["capped_pairs"] >= 1
    assert max(weights) / min(weights) <= 100.0 + 1e-9   # 10x either side
    assert "cap" in report["note"]


def test_the_cap_can_be_removed():
    times = [0.0, 1 / 1440, 30.0, 60.0]
    _, report = temporal_weights(times, limit=None)
    assert report["limit"] is None and report["capped_pairs"] == 0


def test_uniform_mode_reproduces_the_old_behaviour():
    weights, report = temporal_weights([0.0, 1 / 24, 30.0], mode="uniform")
    assert weights == pytest.approx(np.ones(2))
    assert not report["applied"] and report["mode"] == "uniform"


def test_decay_rate_still_applies_on_top():
    times = [0.0, 1.0, 2.0]
    plain, _ = temporal_weights(times)
    decayed, _ = temporal_weights(times, decay_rate=0.5)
    assert decayed == pytest.approx(plain * np.exp(-0.5))


def test_times_that_do_not_increase_fall_back_and_say_so():
    weights, report = temporal_weights([0.0, 1.0, 1.0])     # a repeat
    assert weights == pytest.approx(np.ones(2))
    assert not report["applied"] and "do not increase" in report["note"]

    weights, report = temporal_weights([0.0, 5.0, 2.0])     # out of order
    assert weights == pytest.approx(np.ones(2))
    assert not report["applied"]


def test_a_single_survey_has_no_pairs():
    weights, report = temporal_weights([0.0])
    assert weights.size == 0 and not report["applied"]


def test_the_report_is_json_safe():
    import json

    _, report = temporal_weights([0.0, 1 / 24, 30.0])
    json.dumps(report)          # raises on numpy scalars
