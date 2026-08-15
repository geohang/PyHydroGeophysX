"""Regression tests for line-level EM misfit aggregation."""

import pytest

from PyHydroGeophysX.workflows.em1d import _line_chi2_summary


def test_line_chi2_uses_gate_weighting_not_equal_sounding_mean() -> None:
    summary = _line_chi2_summary([1.0, 9.0], [40, 10])

    assert summary["global"] == pytest.approx(2.6)
    assert summary["sounding_mean"] == pytest.approx(5.0)
    assert summary["sounding_median"] == pytest.approx(5.0)
    assert summary["data_residual_global"] == pytest.approx(2.6 ** 0.5)


def test_line_chi2_prefers_simultaneous_objective_and_ignores_empty_soundings() -> None:
    summary = _line_chi2_summary([1.0, 100.0, float("nan")], [20, 0, 0],
                                 objective_chi2=1.25)

    assert summary["global"] == pytest.approx(1.25)
    assert summary["sounding_mean"] == pytest.approx(1.0)
    assert summary["sounding_median"] == pytest.approx(1.0)
