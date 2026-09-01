"""Weighted chi-square is primary in Qt; original errors remain a dashed reference."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PyHydroGeophysX.qt_apps.widgets.quality_view import InversionQualityView


@pytest.fixture
def view():
    app = QApplication.instance() or QApplication([])
    view = InversionQualityView()
    yield view
    view.close()


def report():
    return {"enabled": True, "chi2_original": 27.7, "chi2_effective": 1.69,
            "target_chi2": 1.75, "target_tolerance": .25,
            "initial": {"convergence": [98., 30., 16.5], "chi2_original": 16.5},
            "passes": [{"convergence": [1.75, 1.6, 1.69], "chi2_original": 27.7}]}


def test_weighted_header_and_solid_curve_raw_only_dashed(view):
    view.show_quality({"chi2": 1.69, "robust": report()},
                      per_item={"values": [1., 2., 1.5], "reference_values": [10., 30., 40.],
                                "value_label": "Weighted χ²"})
    assert "Weighted χ² = 1.69" in view._metrics.text()
    assert "27.70" not in view._metrics.text()
    for ax in view._fig.axes:
        raw = [line for line in ax.lines if line.get_label().startswith("Original-error")]
        weighted = [line for line in ax.lines if line.get_label() == "Weighted χ²"]
        assert raw and weighted
        assert all(line.get_linestyle() == "--" for line in raw)
        assert all(line.get_linestyle() == "-" for line in weighted)
    raw = [line for line in view._fig.axes[0].lines if line.get_label().startswith("Original-error")][0]
    np.testing.assert_array_equal(raw.get_xdata(), [3, 6])  # only real solve endpoints
    np.testing.assert_array_equal(raw.get_ydata(), [16.5, 27.7])


def test_independent_robust_run_has_dashed_final_reference(view):
    data = report()
    data.pop("initial")
    data.pop("passes")
    view.show_quality({"chi2": 1.69, "robust": data})
    raw = [line for line in view._fig.axes[0].lines if line.get_label().startswith("Original-error")]
    assert len(raw) == 1 and raw[0].get_linestyle() == "--"


def test_unweighted_other_modules_keep_the_existing_display(view):
    view.show_quality({"chi2": 2.0}, convergence=[10., 4., 2.])
    assert "Weighted" not in view._metrics.text()
    assert "χ² = 2.00" in view._metrics.text()


def test_median_history_is_primary_and_global_only_reference(view):
    view.show_quality({"chi2": 2.5, "chi2_label": "Median χ²",
                       "convergence_track": [{"chi2": [100., 9., 4.],
                                               "chi2_median": [20., 5., 2.5]}]})
    assert "Median χ² = 2.50" in view._metrics.text()
    curves = {line.get_label(): line for line in view._fig.axes[0].lines}
    np.testing.assert_array_equal(curves["Median χ²"].get_ydata(), [20., 5., 2.5])
    assert curves["Median χ²"].get_linestyle() == "-"
    assert curves["Global χ²"].get_linestyle() == ":"


def test_weighted_median_uses_actual_raw_endpoints_only(view):
    track = [{"chi2": [100., 10.], "chi2_median": [50., 5.], "chi2_original_median": 5.},
             {"chi2": [3., 2.], "chi2_median": [2., 1.5], "chi2_original_median": 7.}]
    view.show_quality({"chi2": 1.5, "chi2_label": "Weighted median χ²",
                       "robust": {"enabled": True}, "convergence_track": track})
    curves = {line.get_label(): line for line in view._fig.axes[0].lines}
    raw = curves["Original-error median χ² (solve endpoints)"]
    assert raw.get_linestyle() == "--"
    np.testing.assert_array_equal(raw.get_xdata(), [1, 3])
    np.testing.assert_array_equal(raw.get_ydata(), [5., 7.])
