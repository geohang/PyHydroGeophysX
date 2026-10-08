"""Saved figures and reports stay readable after switching away from a model."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
pytest.importorskip('PySide6')
from PySide6.QtGui import QImage, QColor, QPalette
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def isolated_appearance(app):
    stylesheet = app.styleSheet()
    palette = QPalette(app.palette())
    # Other desktop tests may apply a global theme. Test this widget's palette
    # handling in isolation, then restore the caller's appearance.
    app.setStyleSheet('')
    try:
        yield
    finally:
        app.setStyleSheet(stylesheet)
        app.setPalette(palette)


def test_figure_uses_window_background_and_fits_after_show(app, tmp_path):
    from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
    image = QImage(600, 200, QImage.Format_RGB32)
    image.fill(QColor('red'))
    path = tmp_path / 'figure.png'
    image.save(str(path))
    view = ZoomableImageView()
    palette = view.palette()
    palette.setColor(QPalette.Base, QColor('#f4f5f7'))
    view.setPalette(palette)
    assert view.set_image_file(path)
    view.resize(800, 500)
    view.show()
    app.processEvents()
    assert view._glw.backgroundBrush().color().name() == '#f4f5f7'
    bounds = view._vb.viewRange()
    assert bounds[0][0] <= 0 and bounds[0][1] >= 600
    assert bounds[1][0] <= 0 and bounds[1][1] >= 200
    palette.setColor(QPalette.Base, QColor('#202020'))
    view.setPalette(palette)
    assert view._glw.backgroundBrush().color().name() == '#202020'
    view.close()


def test_saved_report_recovered_and_metadata_kept_in_details(app, tmp_path):
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.results_store import RunRecord, ResultsStore
    run_dir = tmp_path / 'runs' / 'example'
    (run_dir / 'outputs').mkdir(parents=True)
    (run_dir / 'outputs/summary.md').write_text('# Numerical evidence\nNo LLM interpretation was performed.')
    (run_dir / 'result.json').write_text(json.dumps({'report_files': {'report_markdown': 'outputs/summary.md'}}))
    image = QImage(20, 20, QImage.Format_RGB32)
    image.fill(QColor('white'))
    image.save(str(run_dir / 'outputs/plot.png'))
    record = RunRecord('example', run_dir, 'one_click', 'unified', artifacts=[
        {'path': 'outputs/plot.png', 'kind': 'figure', 'format': 'png'}])
    page = ModelViewerModule(StudioState(output_dir=tmp_path), lambda *args: None)
    page.set_compact(True)
    page._store = ResultsStore(tmp_path, read_only=True)
    page._current = record
    page._populate_artifacts(record)
    assert page._tabs.isTabVisible(page._tabs.indexOf(page._report))
    assert 'No LLM interpretation' in page._report.toPlainText()
    assert page._report.document().baseUrl().toLocalFile().rstrip('/') == str(run_dir / 'outputs').replace('\\', '/')
    assert page._artifact.count() == 1
    assert [page._tabs.tabText(i).lower() for i in range(page._tabs.count())] == list(page._AGENT_TABS)
    page._details_button.setChecked(True)
    assert page._artifact.count() > 1
    page._details_button.setChecked(False)
    assert page._artifact.count() == 1
    page._reset_details()
    assert not page._tabs.isTabVisible(page._tabs.indexOf(page._report))
    page.close()


def test_returning_to_model_keeps_renderer_and_view_state(app, tmp_path, monkeypatch):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.widgets import model3d_view
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.results_store import RunRecord, ResultsStore

    class Renderer(QWidget):
        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            self.loads = 0
            self.selected_field = 'density'

        def show_file(self, path, **kwargs):
            self.loads += 1
            self.selected_field = 'density'
            return True

    monkeypatch.setattr(model3d_view, 'VTKVolumeView', Renderer)
    run_dir = tmp_path / 'runs/example'
    run_dir.mkdir(parents=True)
    vtk = run_dir / 'model.vtk'
    vtk.write_text('initial model')
    (run_dir / 'metadata.json').write_text('{}')
    record = RunRecord('example', run_dir, 'one_click', 'unified', artifacts=[
        {'path': 'model.vtk', 'kind': 'model', 'format': 'vtk'},
        {'path': 'metadata.json', 'kind': 'attachment', 'format': 'json'}])
    page = ModelViewerModule(StudioState(output_dir=tmp_path), lambda *args: None)
    page._store = ResultsStore(tmp_path, read_only=True)
    page._current = record
    page._populate_artifacts(record)
    renderer = page._visual_layout.itemAt(0).widget()
    renderer.selected_field = 'susceptibility'
    page._artifact.setCurrentIndex(1)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert isValid(renderer), 'Switching away must not destroy the OpenGL child'
    page._artifact.setCurrentIndex(0)
    assert page._visual_layout.itemAt(0).widget() is renderer
    assert renderer.selected_field == 'susceptibility'
    assert renderer.loads == 1
    page._artifact.setCurrentIndex(1)
    vtk.write_text('updated model with different content')
    page._artifact.setCurrentIndex(0)
    assert renderer.loads == 2, 'Changed model files must not show a stale cached volume'
    page._search.setText('old project search')
    page._size_cache['example'] = 123
    page.state.set_results_store(tmp_path / 'another-project')
    page.reset_project()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert isValid(renderer), 'Opening a project must retain the OpenGL child'
    assert page._current is None and not page._records
    assert page._vtk_cache_key is None and not page._size_cache
    assert page._search.text() == '' and page._artifact.count() == 0
    assert page._store.root == (tmp_path / 'another-project').resolve()
    page.close()
    page.deleteLater()
    QApplication.sendPostedEvents(page, QEvent.DeferredDelete)
    assert not isValid(renderer), 'The bounded renderer cache belongs to the page lifetime'


def test_runs_named_from_their_data_and_renamed_without_moving_folders(app, tmp_path):
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.results_store import run_label_from_files
    from PyHydroGeophysX.qt_apps.state import StudioState
    files = [f'C:/data/wennerv2_64_{i:03d}.dat' for i in range(1, 421)]
    label = run_label_from_files(files, unit='surveys')
    assert label == 'wennerv2_64 · 420 surveys'
    state = StudioState()
    store = state.set_results_store(tmp_path / 'project')
    runs = []
    for _ in range(2):
        runs.append(state.begin_run('ert_processing', 'ert.timelapse_inversion', label=label))
        state.finish_run('ert_processing', {'status': 'success'}, 'ert.timelapse_inversion')
    label = f'project · {label}'
    store.save_run(runs[1].run_id)
    page = ModelViewerModule(state, lambda *args: None)
    items = page._agent_items([run.run_id for run in runs])
    # Two runs of the same surveys share a name; the list still tells them apart.
    assert {items[run.run_id].text(0) for run in runs} == {
        f"{label}  ({run.run_id.rpartition('_')[2]})" for run in runs}
    assert page._rename_run(runs[1].run_id, 'Line A dry')
    folder = runs[1].run_dir
    assert folder.is_dir() and 'Line A dry' in (folder / 'run.json').read_text(encoding='utf-8')
    assert items[runs[1].run_id].text(0) == 'Line A dry'
    assert items[runs[0].run_id].text(0) == label
    # Naming an unsaved run must not save it as a side effect.
    assert page._rename_run(runs[0].run_id, 'Line A wet') and store.is_unsaved(runs[0].run_id)
    page.close()


def test_window_reset_retains_viewer_but_discards_other_project_pages(app, tmp_path):
    from types import SimpleNamespace
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QStackedWidget, QWidget
    from shiboken6 import isValid
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    state = StudioState(output_dir=tmp_path)
    viewer = ModelViewerModule(state, lambda *args: None)
    other = QWidget()
    other.stop_workers = lambda: None
    stack = QStackedWidget()
    stack.addWidget(other)
    stack.addWidget(viewer)
    window = SimpleNamespace(_pages={'model_viewer': viewer, 'one_click': other}, _stack=stack, state=state)
    state.set_results_store(tmp_path / 'new-project')
    PyHydroGeophysXStudio._reset_pages(window, clear_session=True)
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert window._pages == {'model_viewer': viewer}
    assert stack.count() == 1 and stack.widget(0) is viewer
    assert isValid(viewer) and not isValid(other)
    assert viewer._store.root == (tmp_path / 'new-project').resolve()
    stack.close()
    stack.deleteLater()


def test_project_names_follow_create_save_reopen_and_switch(app, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QLabel, QMainWindow
    from PyHydroGeophysX.qt_apps import main_window
    from PyHydroGeophysX.qt_apps.home_screen import StudioHome
    from PyHydroGeophysX.qt_apps.modules.model_viewer import ModelViewerModule
    from PyHydroGeophysX.qt_apps.results_store import ResultsStore
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.widgets.project_dialogs import NewProjectDialog, SaveRunsDialog

    # Exercise the actual project transition without starting the assistant or
    # changing the user's remembered Project in the system settings.
    monkeypatch.setattr(main_window, 'QSettings', lambda *args: SimpleNamespace(
        setValue=lambda *args: None))
    class Window(QMainWindow):
        _project_name = main_window.PyHydroGeophysXStudio._project_name
        _refresh_output_label = main_window.PyHydroGeophysXStudio._refresh_output_label
        _activate_results_store = main_window.PyHydroGeophysXStudio._activate_results_store
        _switch_project = main_window.PyHydroGeophysXStudio._switch_project
        _create_project = main_window.PyHydroGeophysXStudio._create_project
        _reset_pages = main_window.PyHydroGeophysXStudio._reset_pages

    window = Window()
    window.state = StudioState(output_dir=tmp_path / 'fallback', default_project=True)
    window._output_label = QLabel()
    window.log = lambda *args: None
    window._resolve_unsaved_runs = lambda *args: True
    window._offer_to_clear_abandoned = lambda: None
    window._refresh_unsaved_state = lambda: None
    window._refresh_properties = lambda: None
    home = StudioHome(window.state)
    viewer = ModelViewerModule(window.state, lambda *args: None)
    window._pages = {'home': home, 'model_viewer': viewer}
    assert home._workspace_name.text() == 'Default folder'
    assert viewer._path.text() == 'Project: Default folder'

    dialog = NewProjectDialog(window, tmp_path, name='Site A 2026')
    assert not dialog.problem()
    assert window._create_project(dialog, None)
    project = dialog.project_path().resolve()
    assert window.state.project_directory == project
    assert window.windowTitle().startswith('Site A 2026 — ')
    assert window._output_label.text() == 'Project: Site A 2026'
    assert home._workspace_name.text() == 'Site A 2026'
    assert viewer._path.text() == 'Project: Site A 2026'
    assert str(project) in viewer._path.toolTip()

    run = window.state.begin_run('ert_processing', 'ert.single_inversion', label='Line 01')
    window.state.finish_run('ert_processing', {'status': 'success'}, 'ert.single_inversion')
    expected = 'Site A 2026 · Line 01'
    save = SaveRunsDialog(window, [run.record])
    assert save._edits[run.run_id].text() == expected
    window.state.save_all_runs()
    viewer.refresh()
    assert viewer._agent_items([run.run_id])[run.run_id].text(0) == expected
    assert ResultsStore(project).get_run(run.run_id).label == expected

    # A new project uses its own name and cannot show the previous history.
    window.state.set_results_store(tmp_path / 'Site B')
    viewer.reset_project()
    home.refresh()
    window._refresh_output_label()
    assert viewer._path.text() == 'Project: Site B'
    assert home._workspace_name.text() == 'Site B'
    assert not viewer._records
    other = window.state.begin_run('em_processing', 'em.inversion')
    assert other.record.label == 'Site B'
    assert other.run_dir.parent.parent == (tmp_path / 'Site B').resolve()

    # Reopening the first project preserves an explicit user rename exactly.
    window.state.cancel_run('em_processing', operation_id='em.inversion')
    window.state.discard_all_runs()
    window.state.set_results_store(project)
    window.state.name_runs({run.run_id: 'Dry baseline'})
    viewer.reset_project()
    assert viewer._agent_items([run.run_id])[run.run_id].text(0) == 'Dry baseline'
    assert ResultsStore(project).get_run(run.run_id).label == 'Dry baseline'
    assert run.run_dir.is_dir()
    viewer.close()
    home.close()
    window.close()


def test_browsing_other_project_keeps_name_and_read_only_status(app, tmp_path, monkeypatch):
    from PyHydroGeophysX.qt_apps.modules import model_viewer
    from PyHydroGeophysX.qt_apps.results_store import ResultsStore
    from PyHydroGeophysX.qt_apps.state import StudioState
    state = StudioState(output_dir=tmp_path / 'Active')
    page = model_viewer.ModelViewerModule(state, lambda *args: None)
    other = ResultsStore(tmp_path / 'Other')
    handle = other.begin_run('ert', 'ert.single_inversion', label='Original')
    other.finish_run(handle, {'status': 'success'})
    other.save_run(handle.run_id)
    monkeypatch.setattr(model_viewer.QFileDialog, 'getExistingDirectory',
                        lambda *args: str(other.root))
    page.browse_store()
    for compact in (True, False):
        page.set_compact(compact)
        assert page._path.text() == 'Project: Other — read-only'
        assert str(other.root) in page._path.toolTip()
        assert not page._rename_run(handle.run_id, 'Not editable')
    other.update_run(handle.run_id, label='Changed on disk')
    page.refresh()
    assert page._records[handle.run_id].label == 'Changed on disk'
    assert state.project_name == 'Active'
    page.use_current_store()
    assert page._path.text() == 'Project: Active'
    assert not page._records
    page.close()


def test_default_folder_and_existing_project_prefix(app, tmp_path):
    from PyHydroGeophysX.qt_apps.results_store import run_title
    from PyHydroGeophysX.qt_apps.state import StudioState
    state = StudioState(output_dir=tmp_path / 'fallback', default_project=True)
    handle = state.begin_run('ert', 'ert.single_inversion', label='Line 01')
    assert handle.record.label == 'Line 01'
    with pytest.raises(RuntimeError, match='Cannot switch Project'):
        state.set_results_store(tmp_path / 'Site B')
    assert state.project_name == 'Default folder'
    state.cancel_run('ert', operation_id='ert.single_inversion')
    state.discard_all_runs()
    state.set_results_store(tmp_path / 'Site B')
    state.default_project = False
    handle = state.begin_run('ert', 'ert.single_inversion', label='Site B · Line 01')
    assert handle.record.label == 'Site B · Line 01'
    state.cancel_run('ert', operation_id='ert.single_inversion')
    state.discard_all_runs()
    state.set_results_store(tmp_path / 'ert')
    handle = state.begin_run('ert', 'ert.single_inversion', label='Line 01')
    assert run_title(handle.record) == 'ert · Line 01'
