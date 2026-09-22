"""The shared time-lapse series view: stepping, percentage change, one scale."""

import numpy as np
import pytest

pytest.importorskip("PySide6")
pg = pytest.importorskip("pygimli")

from PyHydroGeophysX.qt_apps.widgets.series_view import (  # noqa: E402
    TimeLapseSeriesView,
    percent_change,
    series_color_limits,
)


@pytest.fixture(scope="module")
def mesh():
    return pg.createGrid(x=np.linspace(0.0, 10.0, 11), y=np.linspace(-4.0, 0.0, 5))


@pytest.fixture
def models(mesh):
    """A baseline and two steps: the second is 10 % higher everywhere."""
    n = mesh.cellCount()
    base = np.linspace(50.0, 200.0, n)
    return np.column_stack([base, base * 1.1, base * 0.9])


def test_percent_change_is_relative_to_the_baseline(models):
    assert percent_change(models, 0) == pytest.approx(np.zeros(models.shape[0]))
    assert percent_change(models, 1) == pytest.approx(np.full(models.shape[0], 10.0))
    assert percent_change(models, 2) == pytest.approx(np.full(models.shape[0], -10.0))


def test_a_zero_baseline_becomes_nan_rather_than_a_spike():
    models = np.array([[0.0, 5.0], [10.0, 11.0]])
    change = percent_change(models, 1)
    assert np.isnan(change[0]) and change[1] == pytest.approx(10.0)


def test_the_change_scale_is_symmetric_and_the_model_scale_is_not(models):
    low, high = series_color_limits(models, "model")
    assert 0.0 < low < high
    change_limits = series_color_limits(models, "change")
    assert change_limits[0] == pytest.approx(-change_limits[1])


def test_series_color_limits_survives_an_empty_or_odd_series():
    assert series_color_limits(np.zeros((3, 0)), "model") is None
    assert series_color_limits(np.full((3, 2), np.nan), "model") is None


def test_the_view_steps_through_the_surveys_and_switches_mode(qt_application, mesh, models):
    view = TimeLapseSeriesView()
    view.set_series(mesh, models, titles=["2024-01-01", "2024-02-01", "2024-03-01"])
    assert view._step.count() == 3
    assert view.current_index() == 0 and view.current_mode() == "model"
    assert view.current_values() == pytest.approx(models[:, 0])

    view._step.setCurrentIndex(1)
    assert view.current_title() == "2024-02-01"
    assert view.current_values() == pytest.approx(models[:, 1])

    view._mode.setCurrentIndex(1)
    assert view.current_mode() == "change"
    assert view.current_values() == pytest.approx(np.full(mesh.cellCount(), 10.0))


def test_either_array_orientation_is_accepted(qt_application, mesh, models):
    view = TimeLapseSeriesView()
    view.set_series(mesh, models.T)
    assert view.current_values().size == mesh.cellCount()
    assert view._step.count() == 3


def test_a_single_model_hides_the_series_controls(qt_application, mesh, models):
    view = TimeLapseSeriesView()
    view.set_series(mesh, models[:, 0])
    assert view._step.count() == 1
    assert not view._bar_row.isVisibleTo(view)
    assert view.current_values() == pytest.approx(models[:, 0])


def test_coverage_follows_the_step_whichever_way_round_it_arrives(
        qt_application, mesh, models):
    n_cells, n_times = models.shape
    coverage = np.tile(np.arange(n_times, dtype=float), (n_cells, 1))
    view = TimeLapseSeriesView()
    view.set_series(mesh, models, coverage=coverage)
    view._step.setCurrentIndex(2)
    assert view.current_coverage() == pytest.approx(np.full(n_cells, 2.0))

    view.set_series(mesh, models, coverage=coverage.T)
    view._step.setCurrentIndex(1)
    assert view.current_coverage() == pytest.approx(np.full(n_cells, 1.0))


def test_a_saved_run_opens_with_the_series_controls_and_maps(
        qt_application, tmp_path, mesh, models):
    """A result someone else inverted is read, and mapped, without re-running it."""
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.results_store import ResultsStore
    from PyHydroGeophysX.qt_apps.state import StudioState

    store = ResultsStore(tmp_path)
    handle = store.begin_run("ert_processing", "ert.timelapse_inversion",
                             workflow_id="ert.timelapse_inversion")
    mesh_path = handle.outputs_dir / "mesh.bms"
    mesh.save(str(mesh_path))
    models_path = handle.outputs_dir / "final_models.npy"
    np.save(models_path, models)
    record = store.finish_run(handle, {"status": "ok", "summary": {
        "model_bundle": {"mesh": str(mesh_path), "models": str(models_path)},
        "time_labels": ["2024-01-01", "2024-02-01", "2024-03-01"]}})

    viewer = ModelViewerModule(StudioState(), lambda *a, **k: None)
    viewer._store = store
    viewer._current = record
    viewer._populate_artifacts(record)

    series = viewer._series
    assert series is not None, "the saved model bundle did not open in the series view"
    assert series._step.count() == 3
    # The dates the run recorded, not "Time step 1".
    assert series.current_title() == "2024-01-01"

    series._mode.setCurrentIndex(1)
    series._step.setCurrentIndex(1)
    meta, arrays = viewer.map_snapshot()
    # The map gets what is on screen, in the units of what is on screen.
    assert meta["units"] == "%"
    assert arrays["values"] == pytest.approx(np.full(mesh.cellCount(), 10.0))

    series._mode.setCurrentIndex(0)
    meta, _ = viewer.map_snapshot()
    assert meta["units"] != "%"


def test_a_saved_result_can_be_temperature_corrected_and_put_back(
        qt_application, tmp_path, mesh):
    """The correction applies to a run that was inverted somewhere else."""
    import datetime as dt

    from PyHydroGeophysX.core import section_geometry
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.results_store import ResultsStore
    from PyHydroGeophysX.qt_apps.state import StudioState

    store = ResultsStore(tmp_path)
    handle = store.begin_run("ert_processing", "ert.timelapse_inversion",
                             workflow_id="ert.timelapse_inversion")
    flat = np.full((mesh.cellCount(), 2), 100.0)
    mesh_path = handle.outputs_dir / "mesh.bms"
    mesh.save(str(mesh_path))
    models_path = handle.outputs_dir / "final_models.npy"
    np.save(models_path, flat)
    stamps = ["2022-01-15 12:00", "2022-07-15 12:00"]
    record = store.finish_run(handle, {"status": "ok", "summary": {
        "model_bundle": {"mesh": str(mesh_path), "models": str(models_path)},
        "time_labels": stamps, "measurement_times": [0.0, 181.0],
        "survey_timing": {"timestamps": stamps}}})

    viewer = ModelViewerModule(StudioState(), lambda *a, **k: None)
    viewer._store = store
    viewer._current = record
    viewer._populate_artifacts(record)
    assert viewer._series is not None

    # The dates the run recorded are what the correction lines the surveys up on.
    days, dates = viewer._series_survey_times(2)
    assert days == [0.0, 181.0]
    assert dates == [dt.datetime(2022, 1, 15, 12), dt.datetime(2022, 7, 15, 12)]

    # A surface record whose warmest day really is mid-July, so "winter" and
    # "summer" in this test mean what they say.
    origin = dt.datetime(2021, 6, 1)
    record_days = [origin + dt.timedelta(days=float(d)) for d in range(900)]
    surface = [10.0 + 12.0 * np.sin(2.0 * np.pi
                                    * (when.timetuple().tm_yday - 105.0) / 365.25)
               for when in record_days]
    spec = {
        "enabled": True, "mode": "surface", "reference": 25.0, "diffusivity": 0.06,
        "surface_times": [when.isoformat(sep=" ") for when in record_days],
        "surface_temperature": surface,
    }
    ok, message = viewer.apply_temperature_spec(spec)
    assert ok, message
    assert "1-D conduction" in message
    assert "25" in viewer._correction_note.text()

    depth = section_geometry.cell_depths(mesh)
    shallow, deep = depth < depth.min() + 0.5, depth > depth.max() - 0.5
    winter = viewer._series.current_values()
    viewer._series._step.setCurrentIndex(1)
    summer = viewer._series.current_values()
    # Cold ground corrects further down than warm ground, and the swing is bigger
    # near the surface than at depth - which is the whole reason for a depth field.
    assert winter[shallow].mean() < summer[shallow].mean()
    assert (abs(winter[shallow].mean() - summer[shallow].mean())
            > abs(winter[deep].mean() - summer[deep].mean()))

    ok, _ = viewer.apply_temperature_spec(None)
    assert ok
    assert viewer._series.current_values() == pytest.approx(flat[:, 1])
    assert viewer._correction_note.text() == ""


@pytest.mark.parametrize("size", [(13.6, 4.5), (13.6, 3.4), (9.0, 3.0), (6.0, 2.8)])
def test_the_section_title_is_not_cut_off_by_the_top_of_the_panel(qt_application, size):
    """pg.show freezes the axes at 96 % of the figure; the title lands outside.

    An equal-aspect section in a wide panel is the normal case, so this was every
    saved result: the date above the section was clipped by the edge of the canvas.
    """
    from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView

    wide = pg.createGrid(x=np.linspace(0.0, 250.0, 26), y=np.linspace(-170.0, 0.0, 18))
    view = MeshResultView()
    view.show_field(wide, np.linspace(50.0, 500.0, wide.cellCount()),
                    kind="ert", title="2022-03-08")
    view._fig.set_size_inches(*size)
    view._canvas.draw()
    title = view._fig.axes[0].title.get_window_extent(view._canvas.get_renderer())
    assert title.y1 <= view._fig.bbox.height, (
        f"the title overflows the canvas by {title.y1 - view._fig.bbox.height:.1f} px")
    # The colorbar has to stay beside the section, not drift off to the margin.
    section = view._fig.axes[0].get_window_extent(view._canvas.get_renderer())
    bar = view._fig.axes[1].get_window_extent(view._canvas.get_renderer())
    assert 0 < bar.x0 - section.x1 < 0.15 * view._fig.bbox.width


def test_the_contour_rendering_draws_and_keeps_a_colorbar(qt_application, mesh, models):
    """The smooth, contourf-style rendering, on the real drawing path."""
    view = TimeLapseSeriesView()
    view.set_series(mesh, models)
    section = view.mesh_view
    section._smooth.setCurrentIndex(section._smooth.count() - 1)   # Contour
    assert section._levels.isVisibleTo(section)
    assert section._fig.axes, "the contour draw produced no axes"
    # Two axes: the section and its colorbar. A failed draw leaves the message
    # axis alone, which is how a silent fallback would show up here.
    assert len(section._fig.axes) == 2
