"""Windowed Python must retain exceptions without a console."""
import io
import sys

from PyHydroGeophysX.qt_apps import launcher


def test_windowed_exception_hook_records_error_without_stderr(monkeypatch):
    record = io.StringIO()
    monkeypatch.setattr(launcher, '_DIAGNOSTIC_STREAM', record)
    monkeypatch.setattr(sys, 'stderr', None)
    monkeypatch.setattr(sys, 'excepthook', sys.excepthook)
    launcher._install_excepthook(show_dialog=False)
    try:
        raise ValueError('Synthetic windowed error')
    except ValueError:
        sys.excepthook(*sys.exc_info())
    assert 'ValueError: Synthetic windowed error' in record.getvalue()


def test_unavailable_diagnostic_stream_does_not_raise(monkeypatch):
    record = io.StringIO()
    record.close()
    monkeypatch.setattr(launcher, '_DIAGNOSTIC_STREAM', record)
    monkeypatch.setattr(sys, 'stderr', None)
    launcher._write_diagnostic('A closed log must not cause another exception')


def test_run_log_keeps_the_summary_a_page_logs_after_recording_its_run(tmp_path, monkeypatch):
    """A page records a run's result and then logs its summary, in one slot.

    Those lines are the run's (the time-lapse page logs "Time-lapse inversion
    complete" after the run has closed), and so is the preparation logged just
    before the run folder existed; a line logged afterwards is not. The run's
    warnings are kept with its record.
    """
    import pytest

    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.base import BaseModule
    from PyHydroGeophysX.qt_apps.state import StudioState

    app = QApplication.instance() or QApplication([])
    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path / 'project')
    page = BaseModule(state, lambda *args: None)

    def next_turns():
        for _ in range(3):
            app.processEvents()

    page.log('Checking the data before the run')
    run = page.begin_persisted_run('test.run', 'test.run')
    page.log('a line the workflow printed')
    next_turns()
    page.finish_persisted_run({'status': 'ok'}, 'test.run')
    page.log('Inversion complete: 420 steps.', 'success')
    page.log('The engine fell back to another solver.', 'warn')
    next_turns()
    page.log('A line from whatever the user does next')
    next_turns()
    text = (run.logs_dir / 'run_log.txt').read_text(encoding='utf-8-sig')
    assert 'INFO    Checking the data before the run' in text
    assert 'INFO    a line the workflow printed' in text
    assert 'SUCCESS Inversion complete: 420 steps.' in text
    assert 'whatever the user does next' not in text
    assert run.record.warnings == ['The engine fell back to another solver.']
    page.deleteLater()


def test_a_failed_run_is_reported_as_itself_not_as_a_missing_backend(tmp_path, monkeypatch):
    """Every failure of the ERT -> water content run read "backend was not found".

    An out-of-memory or a shape error in the workflow process was blamed on
    pygimli and the page jumped to Results showing an exported configuration.
    Only an engine that is really missing - BackendUnavailable or an import
    error, as the process's last exception - may take that fallback; anything
    else is shown as itself on the Run step and recorded as the run's failure.
    """
    import pytest

    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.geo_hydrology import _STEPS, GeoHydrologyModule
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker

    app = QApplication.instance() or QApplication([])

    def ended_on(line):
        """A workflow process that printed ``line`` last and exited with code 1."""
        worker = ProcessWorkflowWorker(tmp_path / 'recipe.json', tmp_path, tmp_path,
                                       tmp_path / 'result.json')
        said = []
        worker.failed.connect(said.append)
        worker._emit_output(f'Traceback (most recent call last):\n{line}\n'.encode(), 'stderr')
        worker._on_finished(1, QProcess.ExitStatus.NormalExit)
        return worker, said[0]

    shape, message = ended_on('ValueError: operands could not be broadcast together '
                              'with shapes (1200,) (1187,)')
    assert not shape.missing_backend
    assert message.startswith('ValueError: operands could not be broadcast')
    missing, _ = ended_on("PyHydroGeophysX._internal.optional_dependencies."
                          "BackendUnavailable: No module named 'pygimli'")
    assert missing.missing_backend

    state = StudioState(output_dir=tmp_path)
    state.set_results_store(tmp_path / 'project')
    page = GeoHydrologyModule(state, lambda *args: None)
    run = page.begin_persisted_run('geo_hydrology.ert_to_wc', 'geo_hydrology.ert_to_wc')
    page._go_to(_STEPS.index('Run'))
    page._on_run_failed(message, shape.missing_backend)
    app.processEvents()
    assert _STEPS[page._current] == 'Run'
    assert message in page._run_status.text()
    assert 'backend' not in page._run_status.text().lower()
    assert (run.record.status, run.record.error) == ('failed', message)
    assert state.module_results.get('geo_hydrology', {}).get('status') != 'config_exported'
    page.deleteLater()
