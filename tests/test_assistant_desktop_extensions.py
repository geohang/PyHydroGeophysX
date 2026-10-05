"""Exercise optional desktop contracts without requiring an OpenGL context."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest


def test_vtk_property_selection_preserves_sparse_labels_and_camera(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pv = pytest.importorskip('pyvista')
    import numpy as np
    from PySide6.QtWidgets import QApplication, QWidget
    from PyHydroGeophysX.qt_apps.widgets import model3d_view as views

    app = QApplication.instance() or QApplication([])

    class Plotter:
        def __init__(self, *a, **kw):
            self.interactor = QWidget()
            self.camera_position = 'original'
            self.mesh_calls = []

        def add_mesh(self, mesh, **kwargs):
            self.mesh_calls.append(kwargs)

        add_mesh_clip_plane = add_mesh

        def reset_camera(self):
            self.camera_position = 'reset'

        def __getattr__(self, name):
            return lambda *a, **kw: None

    monkeypatch.setattr(views, 'try_import_pyvista', lambda: (True, pv, Plotter, ''))
    grid = pv.RectilinearGrid([10, 12, 19], [20, 23], [-10, -2])
    grid.cell_data['Density contrast (g/cm3)'] = [0.1, 0.4]
    grid.cell_data['Geo ID'] = [1, 7]
    path = tmp_path / 'models.vtk'
    grid.save(path)
    widget = views.VTKVolumeView()
    assert widget.show_file(str(path))
    assert widget._scalar == 'Density contrast (g/cm3)'
    widget._plotter.camera_position = 'user viewpoint'
    widget._field.setCurrentText('Geo ID')
    assert widget._plotter.camera_position == 'user viewpoint'
    np.testing.assert_array_equal(widget._mesh['Geo ID'], [1, 7])
    plotted = widget._plotter.mesh_calls[-2]
    np.testing.assert_array_equal(plotted['scalars'], [0, 1])
    assert plotted['annotations'] == {0.0: '1', 1.0: '7'}
    assert plotted['scalar_bar_args']['title'] == 'Geo ID'
    widget.close()
    app.processEvents()


def test_offline_button_is_opt_in_and_launch_is_explicit(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from dataclasses import replace
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.agents.assistants import get_assistant
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    app = QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(output_dir=tmp_path), lambda *a: None)
    assistant = get_assistant('aquah')
    page.set_assistant(assistant)
    assert page.run.isHidden()
    page.set_assistant(replace(assistant, offline_workflow=True))
    assert not page.run.isHidden()
    captured = []
    monkeypatch.setattr(page, '_start', lambda **kw: captured.append(kw))
    page.run.click()
    assert captured == [{'step_mode': page.step_through.isChecked(), 'offline': True}]
    page.close()
    app.processEvents()


def test_domain_launcher_overrides_restored_assistant(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    pytest.importorskip('geosage.pyhydrogeophysx')
    import json
    from PyHydroGeophysX.agents.assistants import get_assistant, set_active
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio
    from PyHydroGeophysX.qt_apps.launcher import main

    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'output_dir': str(tmp_path / 'project')}))
    monkeypatch.setattr(PyHydroGeophysXStudio, '_restore_assistant',
                        lambda self: self._apply_assistant(get_assistant('aquah')))
    shown = []

    def show(self):
        assert self._chat._assistant_combo.currentData() == 'geosage'
        assert self._pages['one_click']._assistant.key == 'geosage'
        shown.append(True)

    monkeypatch.setattr(PyHydroGeophysXStudio, 'show', show)
    previous = get_assistant().key
    try:
        assert main(['--self-test', '--context', str(context), '--module', 'one_click'],
                    assistant='geosage') == 0
    finally:
        set_active(previous)
    assert shown == [True]
