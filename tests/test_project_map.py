"""Project Map persistence, coordinate meaning and UI integration regressions."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PyHydroGeophysX.qt_apps.project_map import (
    ProjectMapStore, em_snapshot, grid_snapshot, point_snapshot, map_coordinates, place_profile,
)


def em_example():
    return {'model3d': np.array([[[300., 100.]], [[400., 200.]], [[500., 250.]], [[600., 300.]]]),
            'positions': np.array([0., 10., 20., 30.]), 'depth_edges': np.array([0., 5., 15.]),
            'line_numbers': [1, 1, 2, 2], 'sensitivity': np.array([[1., 0.1]] * 4),
            'doi_threshold': .8, 'longitude': [-91.54, -91.5399, -91.5398, -91.5397],
            'latitude': [41.66, 41.6601, 41.6602, 41.6603], 'method': 'TDEM'}


def add_example(store, name='TEM survey'):
    meta, arrays = em_snapshot(em_example())
    return store.add(meta, arrays, arrays['survey_xy'], 'EPSG:4326', name)


def test_project_snapshot_survives_reopen_move_and_source_changes(tmp_path):
    root = tmp_path / 'project'
    store = ProjectMapStore(root)
    meta, arrays = em_snapshot(em_example())
    original = arrays['model3d'].copy()
    entry = store.add(meta, arrays, arrays['survey_xy'], 'EPSG:4326', 'Line A')
    arrays['model3d'][:] = 999
    np.testing.assert_array_equal(store.load(entry)['model3d'], original)
    moved = tmp_path / 'moved'
    root.rename(moved)
    reopened = ProjectMapStore(moved)
    assert reopened.entries()[0]['name'] == 'Line A'
    np.testing.assert_array_equal(reopened.load(entry)['model3d'], original)
    assert not Path(entry['data']).is_absolute()


def test_duplicate_add_rename_visibility_and_remove(tmp_path):
    store = ProjectMapStore(tmp_path)
    entry = add_example(store)
    assert add_example(store)['id'] == entry['id']
    assert len(store.entries()) == 1
    store.update(entry['id'], name='Renamed', visible=False)
    assert store.entries()[0]['name'] == 'Renamed'
    assert not store.entries()[0]['visible']
    store.remove(entry['id'])
    assert store.entries() == []
    assert (store.directory / entry['data']).exists()


def test_read_only_and_snapshot_path_escape(tmp_path):
    store = ProjectMapStore(tmp_path)
    entry = add_example(store)
    readonly = ProjectMapStore(tmp_path, read_only=True)
    for action in (lambda: add_example(readonly), lambda: readonly.remove(entry['id']),
                   lambda: readonly.update(entry['id'], name='other')):
        with pytest.raises(PermissionError):
            action()
    with pytest.raises(ValueError, match='path'):
        store.load({**entry, 'data': '../outside.npz'})


def test_coordinate_frames_are_explicit_and_epsg_transform_agrees():
    pytest.importorskip('pyproj')
    from pyproj import Transformer
    lonlat = np.array([[-91.54, 41.66], [-91.539, 41.661]])
    east, north = Transformer.from_crs(4326, 32615, always_xy=True).transform(*lonlat.T)
    projected, frame = map_coordinates(np.column_stack([east, north]), 'EPSG:32615')
    geographic, _ = map_coordinates(lonlat, 'EPSG:4326')
    np.testing.assert_allclose(projected, geographic, atol=1e-6)
    assert frame == 'geographic'
    local, frame = map_coordinates(lonlat, 'LOCAL')
    np.testing.assert_array_equal(local, lonlat)
    assert frame == 'local'
    with pytest.raises(ValueError):
        map_coordinates([[500000, 4600000]], 'EPSG:4326')


def test_straight_and_bent_profile_preserve_chainage():
    distance = [100., 110., 120.]
    np.testing.assert_allclose(place_profile(distance, [500, 1000], 90), [[500,1000],[510,1000],[520,1000]])
    np.testing.assert_allclose(place_profile(distance, None, 0, [[100,0,0],[110,10,0],[120,10,10]]),
                               [[0,0],[10,0],[10,10]])
    with pytest.raises(ValueError, match='span'):
        place_profile(distance, None, 0, [[101,0,0],[120,10,10]])


def test_em_uses_result_order_and_missing_coordinate_column_is_safe():
    result = em_example()
    result['longitude'] = result['longitude'][::-1]
    meta, arrays = em_snapshot(result)
    assert meta['suggested_crs'] == 'EPSG:4326'
    np.testing.assert_array_equal(arrays['survey_xy'][:, 0], result['longitude'])
    result.pop('latitude')
    meta, arrays = em_snapshot(result)
    assert 'survey_xy' not in arrays


@pytest.mark.parametrize('method,unit', [('Gravity', 'g/cc'), ('Magnetics', 'SI'), ('GPR', 'ns')])
def test_generic_grids_and_points_keep_values_and_units(tmp_path, method, unit):
    store = ProjectMapStore(tmp_path)
    edges = [np.arange(3.), np.arange(4.), np.arange(5.)]
    model = np.arange(24.).reshape(2,3,4) - 5
    meta, arrays = grid_snapshot(edges, model, method, unit)
    entry = store.add(meta, arrays, arrays['survey_xy'], 'LOCAL', method)
    np.testing.assert_array_equal(store.load(entry)['model3d'], model)
    assert entry['units'] == unit
    meta, arrays = point_snapshot([[0,0],[1,1]], [-2,3], method, unit)
    entry = store.add(meta, arrays, arrays['survey_xy'], 'LOCAL', method + ' anomaly')
    np.testing.assert_array_equal(store.load(entry)['values'], [-2,3])


@pytest.fixture(scope='module')
def app():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_map_reopens_all_methods_filters_and_handles_missing_snapshot(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    store = ProjectMapStore(tmp_path)
    em = add_example(store)
    for method, unit in [('ERT', 'Ω·m'), ('Seismic', 'm/s'), ('Gravity', 'mGal'), ('Magnetics', 'nT')]:
        meta, arrays = point_snapshot([[-91.54,41.66],[-91.539,41.661]], [1,2], method, unit)
        store.add(meta, arrays, arrays['survey_xy'], 'EPSG:4326', method)
    page = ProjectMapModule(state, lambda *_: None)
    page.resize(1300, 850)
    page.refresh()
    assert page._list.topLevelItemCount() == 5
    page._method.setCurrentText('EM')
    assert page._list.topLevelItemCount() == 1
    assert page._result_stack.currentWidget() is page._em
    page._depth.setCurrentIndex(2)
    assert 'no values above' in page._note.text()
    (store.directory / em['data']).unlink()
    page.refresh()
    assert 'Cannot load' in page._empty.text()
    page.close()


def test_em_line_colours_and_selection_follow_section_and_map_click(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    add_example(ProjectMapStore(tmp_path))
    page = ProjectMapModule(state, lambda *_: None)
    page.refresh()
    colours = [line.get_color() for line in page._ax.lines]
    assert colours[0] != colours[1]
    assert page._ax.lines[0].get_linewidth() > page._ax.lines[1].get_linewidth()
    page._em._line.setCurrentIndex(page._em._line.findData(2))
    assert [line.get_color() for line in page._ax.lines] == colours
    assert page._ax.lines[1].get_linewidth() > page._ax.lines[0].get_linewidth()
    assert any('Line 2 (selected)' in t.get_text() for t in page._ax.get_legend().get_texts())
    assert len(page._ax.get_legend().get_texts()) == 1
    assert not page._ax.get_legend().get_in_layout()
    xlim, ylim = page._ax.get_xlim(), page._ax.get_ylim()
    centre = (sum(xlim) / 2, sum(ylim) / 2)
    page._zoom_map(SimpleNamespace(inaxes=page._ax, xdata=centre[0], ydata=centre[1], step=1))
    assert np.diff(page._ax.get_xlim())[0] == pytest.approx(np.diff(xlim)[0] / 1.25)
    page._zoom_map(SimpleNamespace(inaxes=page._ax, xdata=centre[0], ydata=centre[1], step=-1))
    np.testing.assert_allclose(page._ax.get_xlim(), xlim)
    np.testing.assert_allclose(page._ax.get_ylim(), ylim)
    page._pick(SimpleNamespace(artist=page._ax.lines[0]))
    assert page._em._line.currentData() == 1
    assert any(t.get_text() == 'Line 1' for t in page._ax.texts)
    page.close()


def test_map_project_switch_does_not_reuse_old_surveys(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    state = StudioState(output_dir=tmp_path/'one')
    state.set_results_store(tmp_path/'one')
    add_example(ProjectMapStore(tmp_path/'one'))
    page = ProjectMapModule(state, lambda *_: None)
    page.refresh()
    assert len(page._entries) == 1
    state.set_results_store(tmp_path/'two')
    page.refresh()
    assert not page._entries and not page._arrays
    assert page._result_stack.currentWidget() is page._empty
    page.close()


def test_em_result_page_no_longer_offers_maps(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.em_processing import EMProcessingModule
    page = EMProcessingModule(StudioState(output_dir=tmp_path), lambda *_: None)
    assert page._overview_view._section_only
    assert not hasattr(page, '_plan_view')
    page._overview_view.show_result(em_example(), x=np.arange(4.), y=np.arange(4.))
    assert page._overview_view._map_ax is None
    assert page._overview_view._basemap.isHidden()
    page._on_view_mode(1)
    assert page._model_stack.currentWidget() is page._section_view
    page.close()


def test_grid_ui_preserves_signed_values_and_separate_units(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    model = np.arange(8.).reshape(2,2,2)-4
    meta, arrays = grid_snapshot([np.arange(3.)]*3, model, 'Gravity', 'g/cc')
    entry = ProjectMapStore(tmp_path).add(meta, arrays, arrays['survey_xy'], 'LOCAL', 'Density')
    state.map_selected_id = entry['id']
    page = ProjectMapModule(state, lambda *_: None)
    page.refresh()
    page._depth.setCurrentIndex(1)
    assert page._frame.currentData() == 'local'
    assert page._result_stack.currentWidget() is page._mesh_view
    assert page._fig.axes[-1].get_ylabel() == 'g/cc'
    assert page._section_fig.axes[-1].get_ylabel() == 'g/cc'
    assert not page._load_tiles.isEnabled()
    page.close()


def test_map_placement_dialog_rejects_geographic_bearing(app):
    from PyHydroGeophysX.qt_apps.widgets.map_export import MapExportDialog
    meta, arrays = em_snapshot(em_example())
    arrays['line_numbers'][:] = 1
    dialog = MapExportDialog(meta, arrays)
    dialog.mode.setCurrentIndex(dialog.mode.findData('straight'))
    dialog.crs.setCurrentText('EPSG:4326')
    with pytest.raises(ValueError, match='projected'):
        dialog.placement()
    dialog.close()


@pytest.mark.parametrize('add', [False, True])
def test_completion_prompt_is_opt_in_and_keeps_not_now_as_default(app, tmp_path, add):
    from PySide6.QtWidgets import QMessageBox
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    page = BaseModule(StudioState(output_dir=tmp_path), lambda *_: None)
    choices = []
    page.add_to_map = lambda: choices.append('add')
    page.offer_map_export()
    app.processEvents()
    prompt = page.findChild(QMessageBox)
    assert prompt is not None
    assert prompt.defaultButton().text() == 'Not now'
    assert not choices and not (tmp_path/'project_map').exists()
    next(b for b in prompt.buttons() if b.text() == ('Add to Map…' if add else 'Not now')).click()
    app.processEvents()
    assert choices == (['add'] if add else [])
    page.close()


def test_the_dismissed_completion_prompt_stops_belonging_to_the_page(app, tmp_path):
    """A dismissed prompt must leave the page before it is queued for deletion.

    deleteLater only runs when an event loop does, so a dialog still parented to
    the page when the page is torn down leaves a queued deletion naming freed
    memory. Nothing fails at that moment: the next event loop to run - a process
    worker's, a modal dialog's - aborts the interpreter with no Python
    traceback, in whatever test happens to be running then.
    """
    from PySide6.QtWidgets import QMessageBox
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    page = BaseModule(StudioState(output_dir=tmp_path), lambda *_: None)
    page.add_to_map = lambda: None
    page.offer_map_export()
    app.processEvents()
    prompt = page.findChild(QMessageBox)
    next(b for b in prompt.buttons() if b.text() == 'Not now').click()
    app.processEvents()
    assert prompt.parent() is None
    assert page.findChild(QMessageBox) is None
    page.close()


def test_basemap_fetch_is_off_ui_thread_and_failure_leaves_survey_visible(app, tmp_path, monkeypatch):
    import threading
    import time
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules import project_map
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    add_example(ProjectMapStore(tmp_path))
    threads = []
    def offline(*args, **kwargs):
        threads.append(threading.get_ident())
        return None
    monkeypatch.setattr(project_map, 'basemap_image', offline)
    page = project_map.ProjectMapModule(state, lambda *_: None)
    page.refresh()
    page._fetch_tiles()
    # Wait for the worker thread, not for a duration. A fixed budget made this
    # fail whenever the machine was busy, which says nothing about whether the
    # fetch ran off the UI thread; the deadline is only so a genuinely stuck
    # worker fails instead of hanging the suite.
    deadline = time.monotonic() + 120
    while page._tile_worker is not None and time.monotonic() < deadline:
        app.processEvents()
        worker = page._tile_worker
        if worker is not None and worker.isRunning():
            worker.wait(200)
        time.sleep(0.02)
    assert page._tile_worker is None
    assert threads and threads[0] != threading.get_ident()
    assert page._artists and 'unavailable' in page._note.text()
    page.close()


def em_grid_example(lines=5, stations=7, layers=4, spacing=60., seed=2):
    """A TEM survey that actually spreads in 2D, so a plan slice is meaningful."""
    generator = np.random.default_rng(seed)
    x, y = np.meshgrid(np.arange(lines) * spacing * 2, np.arange(stations) * spacing,
                       indexing='ij')
    x, y = x.ravel(), y.ravel()
    depth = np.arange(layers) * 8.
    rho = 40 * 10 ** (.5 * np.sin(x / 200.) * np.cos(y / 180.))[:, None] * (1 + depth / 40.)
    return {'model3d': rho[:, ::-1][:, None, :], 'positions': np.arange(float(len(x))),
            'depth_edges': np.arange(layers + 1) * 8., 'line_numbers': np.repeat(
                np.arange(lines), stations).astype(float),
            'sensitivity': np.ones((len(x), layers)),
            'doi_threshold': .5, 'x': x, 'y': y, 'method': 'TDEM'}


def map_page(state_root, snapshot, name):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    state = StudioState(output_dir=state_root)
    state.set_results_store(state_root)
    meta, arrays = snapshot
    entry = ProjectMapStore(state_root).add(meta, arrays, arrays['survey_xy'], 'LOCAL', name)
    state.map_selected_id = entry['id']
    page = ProjectMapModule(state, lambda *_: None)
    page.refresh()
    return page, entry


def surfaces(page):
    from matplotlib.collections import QuadMesh
    return [a for a in page._ax.collections if isinstance(a, QuadMesh)]


def test_plan_interpolation_fills_an_em_depth_slice_and_replaces_its_markers(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(2)
    assert not surfaces(page), 'Points only must not interpolate anything'
    assert page._ax.lines and page._ax.collections, 'Points only draws lines and stations'
    assert not page._export_grid.isEnabled() and not page._variogram_button.isEnabled()
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    mesh = surfaces(page)
    assert len(mesh) == 1
    result = page._surface[1]
    assert result['method'] == 'kriging' and result['log_values']
    # The surface carries these values, so stamping the same survey's stations
    # and line traces on top of it would only hide the image.
    assert not page._ax.lines
    assert list(page._ax.collections) == mesh
    assert page._fig.axes[-1].get_ylabel().startswith('Resistivity')
    assert page._export_grid.isEnabled() and page._variogram_button.isEnabled()
    assert 'Kriging' in page._note.text() and 'variogram' in page._note.text()
    page.close()


def test_stations_can_be_put_back_over_the_surface_and_share_its_colour_scale(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(2)
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    page._stations.setChecked(True)
    mesh = surfaces(page)
    assert len(mesh) == 1 and page._ax.lines
    coloured = [a for a in page._ax.collections if a is not mesh[0] and a.get_array() is not None]
    # One colour bar has to read the same for interpolated cells and soundings.
    assert coloured and coloured[-1].norm is mesh[0].norm
    assert len([a for a in page._fig.axes if a is not page._ax]) == 1
    page._stations.setChecked(False)
    assert not page._ax.lines
    page.close()


def test_an_unselected_survey_keeps_its_line_traces_while_another_is_interpolated(app, tmp_path):
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.project_map import ProjectMapModule
    store = ProjectMapStore(tmp_path)
    meta, arrays = em_snapshot(em_grid_example())
    chosen = store.add(meta, arrays, arrays['survey_xy'], 'LOCAL', 'TEM block')
    other_meta, other = em_snapshot(em_grid_example(seed=9))
    store.add(other_meta, other, other['survey_xy'] + 2000., 'LOCAL', 'Neighbour block')
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path)
    state.map_selected_id = chosen['id']
    page = ProjectMapModule(state, lambda *_: None)
    page.refresh()
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('idw'))
    # Hiding markers is about the survey the surface belongs to, not the map.
    assert len(surfaces(page)) == 1
    assert page._ax.lines and page._artists
    assert all(page._artists[a][0] != chosen['id'] for a in page._artists)
    page.close()


def test_plan_interpolation_serves_any_method_not_just_em(app, tmp_path):
    generator = np.random.default_rng(4)
    xy = generator.uniform(0, 400, (60, 2))
    page, _ = map_page(tmp_path, point_snapshot(xy, generator.normal(0, 3, 60), 'Gravity', 'mGal'),
                       'Bouguer stations')
    page._interp.setCurrentIndex(page._interp.findData('idw'))
    assert len(surfaces(page)) == 1
    # A signed potential-field product must not be pushed through a log scale.
    assert page._surface[1]['log_values'] is False
    assert page._fig.axes[-1].get_ylabel() == 'mGal'
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    assert page._variogram_button.isEnabled()
    page.close()


def test_blanking_distance_and_resolution_reach_the_drawn_surface(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('idw'))
    wide = page._surface[1]
    page._interp_res.setValue(60)
    assert page._surface[1]['cell_size'] > wide['cell_size']
    page._interp_blank.setValue(20)
    assert page._surface[1]['coverage'] < wide['coverage']
    assert page._surface[1]['max_distance'] == 20.
    page.close()


def test_a_blanking_radius_below_the_station_reach_says_what_it_is_costing(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('idw'))
    needed = page._surface[1]['gap']
    assert 'cutting inside' not in page._note.text()
    page._interp_blank.setValue(max(1, round(needed / 8)))
    # Ribbons along the lines are the visible symptom; the caption has to name
    # the number that would fill the outline instead of leaving it to guesswork.
    assert 'cutting inside the survey' in page._note.text()
    assert f'{needed:.0f}' in page._note.text()
    page._interp_blank.setValue(round(needed) + 1)
    assert 'cutting inside' not in page._note.text()
    page.close()


def test_gridding_settings_ignore_a_wheel_that_drifts_off_the_zooming_map(app, tmp_path):
    from PySide6.QtCore import Qt, QPoint, QPointF
    from PySide6.QtGui import QWheelEvent
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('idw'))
    for widget in (page._interp_res, page._interp_blank):
        widget.clearFocus()
        before = widget.value()
        widget.wheelEvent(QWheelEvent(
            QPointF(4, 4), QPointF(4, 4), QPoint(0, 0), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        assert widget.value() == before, 'an unfocused wheel must not re-grid the slice'
    page.close()


def test_zero_blanking_reads_as_the_off_switch_it_is(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    assert page._interp_blank.value() == 0
    assert page._interp_blank.text() == 'no blanking'
    assert not page._interp_blank.keyboardTracking()
    page.close()


def test_a_single_em_line_is_refused_without_losing_the_survey(app, tmp_path):
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example(lines=1, stations=12)), 'One line')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    assert not surfaces(page)
    assert 'single line' in page._note.text()
    assert page._artists and not page._export_grid.isEnabled()
    page.close()


def test_the_drawn_plan_grid_exports_to_an_ascii_raster(app, tmp_path, monkeypatch):
    from PyHydroGeophysX.qt_apps.modules import project_map
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    target = tmp_path / 'slice.asc'
    monkeypatch.setattr(project_map.QFileDialog, 'getSaveFileName',
                        staticmethod(lambda *a, **k: (str(target), '')))
    page._export_surface()
    header = dict(line.split() for line in target.read_text().splitlines()[:6])
    assert int(header['ncols']) == page._surface[1]['x'].size
    assert float(header['cellsize']) > 0
    assert str(target) in page._note.text()
    page.close()


def test_the_variogram_dialog_plots_the_experimental_cloud_against_the_model(app, tmp_path):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from PyHydroGeophysX.qt_apps.modules.project_map import VariogramDialog
    page, _ = map_page(tmp_path, em_snapshot(em_grid_example()), 'TEM block')
    page._depth.setCurrentIndex(1)
    page._interp.setCurrentIndex(page._interp.findData('kriging'))
    result = page._surface[1]
    fit = result['variogram']
    assert fit['model'] in ('spherical', 'exponential', 'gaussian')
    assert fit['range'] > 0 and fit['sill'] > 0
    dialog = VariogramDialog(result, page._entry(), page._depth.currentText(), page)
    axes = dialog.findChild(FigureCanvasQTAgg).figure.axes[0]
    # Resistivity is kriged in log space, and the axis has to say so.
    assert 'log10' in axes.get_ylabel()
    experimental, model = axes.lines[0], axes.lines[1]
    np.testing.assert_allclose(experimental.get_ydata(), fit['gamma'])
    # The fitted curve has to level off at the sill it reports.
    assert model.get_ydata()[-1] <= fit['sill'] * 1.001
    assert fit['model'] in model.get_label()
    dialog.close()
    page.close()
