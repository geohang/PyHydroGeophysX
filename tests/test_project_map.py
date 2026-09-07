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
    deadline = time.monotonic() + 5
    while page._tile_worker is not None and time.monotonic() < deadline:
        app.processEvents()
    assert page._tile_worker is None
    assert threads and threads[0] != threading.get_ident()
    assert page._artists and 'unavailable' in page._note.text()
    page.close()
