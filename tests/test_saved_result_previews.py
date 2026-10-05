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
