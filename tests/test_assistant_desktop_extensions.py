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


def test_normal_studio_start_always_uses_native_aquah():
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.agents.assistants import get_assistant, set_active
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio

    previous = get_assistant().key
    applied = []
    try:
        set_active('geosage')
        window = PyHydroGeophysXStudio.__new__(PyHydroGeophysXStudio)
        window._apply_assistant = applied.append
        window._restore_assistant()
    finally:
        set_active(previous)
    assert [agent.key for agent in applied] == ['aquah']


def test_focused_assistant_keeps_chat_dock_open():
    pytest.importorskip('PySide6')
    from types import SimpleNamespace
    from PyHydroGeophysX.agents.assistants import get_assistant, set_active
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio

    class Dock:
        def __init__(self):
            self.visible = None

        def show(self):
            self.visible = True

        def hide(self):
            self.visible = False

    previous = get_assistant().key
    try:
        set_active('geosage')
        window = PyHydroGeophysXStudio.__new__(PyHydroGeophysXStudio)
        window._all_tools_action = SimpleNamespace(setChecked=lambda _value: None)
        window._toggle_all_tools = lambda _value: None
        window._main_toolbar = SimpleNamespace(actions=lambda: [])
        window._pages = {}
        window._log_dock = Dock()
        window._properties_dock = Dock()
        window._focus_workspace()
    finally:
        set_active(previous)
    assert window._log_dock.visible is False
    assert window._properties_dock.visible is True


def test_output_environment_override_beats_remembered_project(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio
    from PyHydroGeophysX.qt_apps.state import StudioState

    chosen = tmp_path / 'automatic-runs'
    monkeypatch.setenv('PYHYDROGEOPHYSX_OUTPUT_DIR', str(chosen))
    window = PyHydroGeophysXStudio.__new__(PyHydroGeophysXStudio)
    window.state = StudioState(output_dir=tmp_path / 'default')
    window.state.context = {}
    window._refresh_output_label = lambda: None
    window._restore_output_dir()
    assert window.state.output_dir == chosen


def test_tool_requested_workflow_waits_for_paired_result(monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AssistantChatPanel
    from PyHydroGeophysX.qt_apps.agent.controller import StudioController
    from PyHydroGeophysX.llm.providers import make_provider

    app = QApplication.instance() or QApplication([])
    provider = make_provider('openai')
    monkeypatch.setattr(provider, 'available', lambda: (True, ''))
    panel = AssistantChatPanel(StudioController(None), provider=provider)
    started = []
    panel._busy = True
    panel.start_workflow('Run the confirmed inversion')
    assert panel._pending_workflow_text == 'Run the confirmed inversion'
    monkeypatch.setattr(panel, 'start_workflow', started.append)
    panel._tool_queue = []
    panel._executed_in_turn = True
    panel._process_next_tool()
    assert started == ['Run the confirmed inversion']
    assert panel._pending_workflow_text is None
    panel.close()
    app.processEvents()
