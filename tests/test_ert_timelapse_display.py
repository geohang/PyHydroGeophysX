"""How a time-lapse ERT result is displayed: one colour scale, change mode, titles.

A time-lapse is read by comparing steps. Autoscaling each step to its own
extremes and heading the panel with a bare number are both ways of losing that
comparison, so both are covered here.
"""

import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_app():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qt stack unavailable: {exc}")
    app = QApplication.instance() or QApplication([])
    yield app


def _grid_and_models():
    pg = pytest.importorskip("pygimli")
    mesh = pg.createGrid(x=np.linspace(0.0, 10.0, 11), y=np.linspace(-5.0, 0.0, 6))
    n = mesh.cellCount()
    baseline = 100.0 * np.exp(np.linspace(-0.4, 0.4, n))
    models = np.column_stack([
        baseline,
        baseline * 1.2,   # +20 % everywhere
        baseline * 0.75,  # -25 % everywhere
    ])
    return mesh, models


def _drawn_limits(view):
    for collection in view._fig.axes[0].collections:
        if collection.get_array() is not None:
            return collection.get_clim()
    raise AssertionError("nothing with a colour array was drawn")


def _timelapse_module(mesh, models, titles):
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    module = ERTProcessingModule(StudioState(output_dir=Path.cwd()), lambda *_a: None)
    module._tl_mesh = mesh
    module._tl_models = models
    module._tl_step_titles = list(titles)
    module._populate_tl_steps()
    return module


def test_steps_share_one_colour_scale(qt_app) -> None:
    """Stepping through the series must not rescale the colours under the reader."""
    mesh, models = _grid_and_models()
    module = _timelapse_module(mesh, models, ["2021-10-08", "2021-10-09", "2021-10-10"])
    try:
        first = _drawn_limits(module._model_view)
        module._show_tl_step(2)
        third = _drawn_limits(module._model_view)

        assert module._model_view._lock_range.isChecked()
        assert np.allclose(first, third)
        # The locked range spans the whole series, not just one step.
        assert first[0] <= models.min() * 1.05
        assert first[1] >= models.max() * 0.95
    finally:
        module.stop_workers()
        module.close()
        qt_app.processEvents()


def test_percent_change_mode_is_signed_and_zero_centred(qt_app) -> None:
    mesh, models = _grid_and_models()
    module = _timelapse_module(mesh, models, ["2021-10-08", "2021-10-09", "2021-10-10"])
    try:
        module._tl_view_mode.setCurrentIndex(1)  # "% change from baseline"
        assert module._tl_view_mode.currentData() == "change"

        assert np.allclose(module._percent_change(models, 0), 0.0)
        assert np.allclose(module._percent_change(models, 1), 20.0)
        assert np.allclose(module._percent_change(models, 2), -25.0)

        module._show_tl_step(2)
        low, high = _drawn_limits(module._model_view)
        assert low == pytest.approx(-high)  # a signed change needs a centred scale
        assert high > 0.0
        assert "Change from baseline (%)" == module._model_view._fig.axes[-1].get_ylabel()

        # Switching back restores the resistivity scale rather than keeping the
        # percentage limits.
        module._tl_view_mode.setCurrentIndex(0)
        low, high = _drawn_limits(module._model_view)
        assert low > 0.0 and high > low
    finally:
        module.stop_workers()
        module.close()
        qt_app.processEvents()


def test_percent_change_skips_a_zero_baseline() -> None:
    from PyHydroGeophysX.qt_apps.modules.ert_processing import ERTProcessingModule

    models = np.array([[100.0, 120.0], [0.0, 5.0]])
    change = ERTProcessingModule._percent_change(models, 1)
    assert change[0] == pytest.approx(20.0)
    assert np.isnan(change[1])  # a division artefact would read as real change


def test_step_titles_keep_the_dates_the_caller_parsed() -> None:
    """The Studio stages files under generated names, so labels must travel."""
    from PyHydroGeophysX.data_processing.ert_io import measurement_times_for
    from PyHydroGeophysX.inversion._time_lapse_workflow import _step_titles

    files = ["ert_2021-10-08_1400.ohm", "ert_2021-10-08_2037.ohm"]
    times, labels = measurement_times_for(files)

    assert _step_titles(labels, times, 2) == labels
    # Without the labels the panel falls back to the elapsed time, which is only
    # readable if it says what it counts.
    numeric = [f"{t:g}" for t in times]
    assert _step_titles(numeric, times, 2, "d") == ["t = 0 d", "t = 0.275694 d"]
    assert _step_titles(["1", "2"], [1.0, 2.0], 2) == ["Time step 1", "Time step 2"]


def test_sensitivity_view_does_not_clobber_the_locked_model_range(qt_app) -> None:
    """The coverage plot is in log10 units; its range must not become the model's."""
    mesh, models = _grid_and_models()
    module = _timelapse_module(mesh, models, ["2021-10-08", "2021-10-09", "2021-10-10"])
    try:
        view = module._model_view
        module._tl_coverage = np.tile(
            np.linspace(-3.0, 1.0, mesh.cellCount()), (models.shape[1], 1))
        module._show_tl_step(1)
        locked = (view._cmin.value(), view._cmax.value())

        view._show_cov.setChecked(True)
        view._show_cov.setChecked(False)

        assert (view._cmin.value(), view._cmax.value()) == locked
        assert np.allclose(_drawn_limits(view), locked)
    finally:
        module.stop_workers()
        module.close()
        qt_app.processEvents()
