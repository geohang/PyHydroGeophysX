import json
import os
from pathlib import Path
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PyHydroGeophysX.qt_apps.agent.one_click_runner import execute


class Context:
    def __init__(self, **kwargs):
        pass

    def parse_request(self, request, available_data=None):
        assert available_data
        return {'ert_file': 'invented.dat'}


def test_selected_inputs_win_and_results_are_saved_without_key(tmp_path):
    def run(config, key, model, provider, output, progress_callback):
        assert config['ert_file'] == 'actual.dat'
        assert key == 'private-key'
        return {'model': [10, 20]}, [], 'Interpretation', {}
    result = execute(dict(request='Invert ERT', inputs={'data_file': 'actual.dat'},
                          output_dir=str(tmp_path), api_key='private-key'),
                     lambda *args: None, context_factory=Context, run_fn=run)
    assert result['interpretation'] == 'Interpretation'
    assert json.loads((tmp_path / 'numerical_results.json').read_text())['model'] == [10, 20]
    assert all('private-key' not in p.read_text() for p in tmp_path.iterdir())


def test_engine_failure_is_not_published_as_success(tmp_path):
    with pytest.raises(RuntimeError, match='bad data'):
        execute(dict(request='Invert', inputs={'data_file': 'x'}, output_dir=str(tmp_path)),
                lambda *args: None, context_factory=Context,
                run_fn=lambda *args, **kw: ({'status': 'failed', 'error': 'bad data'}, [], None, {}))
    assert not (tmp_path / 'workflow_result.json').exists()


def test_page_validation_completion_and_cancellation(tmp_path):
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    app = QApplication.instance() or QApplication([])
    state = StudioState()
    state.output_dir = tmp_path
    page = OneClickModule(state, lambda *args: None)
    page._start()
    assert page._worker is None
    handle = page.begin_persisted_run('unified')
    page._output = str(handle.outputs_dir)
    handle.outputs_dir.mkdir(exist_ok=True)
    report = handle.outputs_dir / 'report.md'
    report.write_text('# Finished\nInterpretation', encoding='utf-8')
    page._succeeded({'report_files': {'report_markdown': str(report)}})
    assert page.progress.value() == 100
    assert 'Interpretation' in page.report.toPlainText()
    assert page.files.count() == 1
    assert state.has_unsaved_runs()
    page.begin_persisted_run('unified')
    page._worker = type('Cancelled', (), {'is_cancelled': lambda self: True})()
    page._finished()
    assert page.run.isEnabled()
    assert state.active_run('one_click', 'unified') is None
    page.close()
    page.deleteLater()


def test_workflow_has_no_second_ai_form_and_retains_goal(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QLineEdit
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    app = QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *args: None)
    settings = {'provider': 'openai', 'api_key': 'session-secret', 'model': 'model'}
    response = page.submit_request('Invert ERT and report', settings)
    assert 'Add your data' in response
    assert not page._ai_settings
    assert not page.findChildren(QLineEdit)
    page._inputs = {'data_file': 'survey.dat'}
    seen = []
    monkeypatch.setattr(page, '_start', lambda step_mode=False: seen.append(
        (page._request_text, dict(page._ai_settings), step_mode)))
    page.submit_request('continue', settings)
    # The third value is the pacing: straight through unless the Workflow page's
    # "Approve each step" box is ticked.
    assert seen == [('Invert ERT and report', settings, False)]
    page.step_through.setChecked(True)
    page.submit_request('continue', settings)
    assert seen[-1][2] is True
    page.step_through.setChecked(False)
    page.submit_request('Use time-lapse instead', settings)
    assert 'Invert ERT and report' in seen[-1][0]
    assert 'Use time-lapse instead' in seen[-1][0]
    page.reset_request()
    page.submit_request('New seismic analysis', settings)
    assert seen[-1][0] == 'New seismic analysis'
    assert not page._ai_settings
    page.close()
    page.deleteLater()


def test_chat_auto_mode_uses_shared_settings_and_receives_completion():
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.chat_panel import AquahChatPanel
    from PyHydroGeophysX.llm.providers import make_provider
    app = QApplication.instance() or QApplication([])
    class Controller:
        def capabilities_summary(self):
            return ''
        def run_to_report(self, text, settings, callback):
            self.request, self.settings, self.callback = text, settings, callback
            return 'Workflow started. Follow progress in Workflow.'
    controller = Controller()
    panel = AquahChatPanel(controller, provider=make_provider('openai', model='gpt-4o', api_key='session-secret'))
    panel._execution_mode.setCurrentIndex(1)
    panel._input.setPlainText('Invert my ERT data')
    panel._on_send()
    assert controller.request == 'Invert my ERT data'
    assert controller.settings['api_key'] == 'session-secret'
    assert panel._busy
    controller.callback('Complete: report ready')
    assert not panel._busy
    assert 'report ready' in panel._transcript.toPlainText()
    assert 'session-secret' not in panel._transcript.toPlainText()
    panel.close()
    panel.deleteLater()


def test_catalog_role_edits_are_applied_without_resetting_time_order(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.state import StudioState
    from PyHydroGeophysX.qt_apps.modules.one_click import OneClickModule
    app = QApplication.instance() or QApplication([])
    page = OneClickModule(StudioState(), lambda *args: None)
    page._show_catalog({'files': [dict(name=name, path=name, role='time_lapse_files') for name in ['a.dat', 'b.dat']]})
    monkeypatch.setattr(page, '_start', lambda step_mode=False: None)
    page.submit_request('Invert time lapse', {})
    assert page._inputs['time_lapse_files'] == ['a.dat', 'b.dat']
    page._inputs['time_lapse_files'].reverse()
    page.submit_request('continue', {})
    assert page._inputs['time_lapse_files'] == ['b.dat', 'a.dat']
    selector = page.catalog_table.cellWidget(1, 1)
    selector.setCurrentIndex(selector.findData('ignore'))
    page.submit_request('continue', {})
    assert page._inputs['time_lapse_files'] == ['a.dat']
    page.close()
    page.deleteLater()


def test_losing_the_channel_mid_run_is_reported_rather_than_hung(tmp_path):
    """The input channel now stays open all run so the child can be answered.

    A failed write to it means the child is waiting for an answer it will never
    get, and nothing used to notice: only `FailedToStart` was handled, so the
    run sat there with no message and no result until somebody gave up. Every
    path out of the worker must end in either a result or an error.
    """
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.one_click_worker import OneClickWorker
    QApplication.instance() or QApplication([])
    worker = OneClickWorker({'output_dir': str(tmp_path), 'api_key': 'secret'})
    results, errors, done = [], [], []
    worker.succeeded.connect(results.append)
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: done.append(True))

    worker._on_process_error(QProcess.ProcessError.WriteError)
    assert errors and 'Lost contact' in errors[0] and 'sending to' in errors[0]
    assert done, 'the run has to end, not hang'
    assert not results

    # And a second error afterwards changes nothing, so a run cannot be
    # reported as failed twice.
    worker._on_process_error(QProcess.ProcessError.ReadError)
    assert len(errors) == 1 and len(done) == 1


def test_an_answer_is_written_to_the_child_not_left_in_a_buffer():
    """`QProcess.write` only buffers; the bytes leave when the event loop says.

    An answer is written from inside the read handler that delivered the
    question - that is where the desktop's button press runs - and the flush was
    then deferred for the rest of the run. The child sat blocked on its stdin
    while the answer the user had already given waited on this side, which is
    indistinguishable from the agent having frozen.
    """
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.one_click_worker import OneClickWorker
    QApplication.instance() or QApplication([])
    worker = OneClickWorker({'output_dir': str(Path.cwd()), 'api_key': 'x'})
    waited = []

    class Recording:
        def write(self, payload):
            waited.append(['write', bytes(payload)])

        def waitForBytesWritten(self, msecs):  # noqa: N802 - Qt naming
            waited.append(['flush', msecs])
            return True

    worker.process = Recording()
    worker.answer('segy')
    assert worker.answer('proceed') is True, 'the caller is told it was sent'
    waited.clear()
    worker.answer('segy')
    assert [step[0] for step in waited] == ['write', 'flush'], (
        'the answer has to be flushed, not just queued')
    assert waited[0][1] == b'{"decision": "segy"}\n'
    assert waited[1][1] == worker.WRITE_TIMEOUT_MS

    # And nothing is written or waited on for an empty payload, so a second
    # `_send_input` after the payload is cleared costs nothing.
    waited.clear()
    worker._payload = b''
    worker._send_input()
    assert waited == []


def test_real_process_protocol_progress_and_stdin(tmp_path):
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.agent.one_click_worker import OneClickWorker
    app = QApplication.instance() or QApplication([])
    script = tmp_path / 'child.py'
    # The child reads ONE line and keeps reading: the payload arrives first,
    # then any answer to a question it asks mid-run. Consuming the whole
    # stream - which this did - makes every later question read end of input.
    script.write_text("import sys,json\np=json.loads(sys.stdin.readline())\nassert p['api_key']=='secret'\nprint(json.dumps({'event':'progress','step':'Report','progress':.9}),flush=True)\nprint(json.dumps({'event':'question','question':'Which origin?','options':[{'id':'segy'}]}),flush=True)\nanswer=json.loads(sys.stdin.readline())['decision']\nprint(json.dumps({'ok':True,'result':{'interpretation':'done','answer':answer}}),flush=True)\n", encoding='utf-8')
    worker = OneClickWorker({'output_dir': str(tmp_path), 'api_key': 'secret'})
    worker.process.setArguments([str(script)])
    results, errors, steps = [], [], []
    worker.succeeded.connect(results.append)
    worker.failed.connect(errors.append)
    worker.progress.connect(lambda *args: steps.append(args))
    import time

    asked, sends = [], []
    started_at = time.monotonic()
    # Answering is what the desktop's prompt buttons do, and it happens from
    # inside the read handler that delivered the question - which is exactly
    # where a buffered write can fail to leave this process. The child blocks on
    # its stdin until it arrives, so if the answer is only buffered this test
    # sits here until the safety timer fires and finds the child still running.
    # `sends` records whether the bytes were reported as written, and when, so a
    # failure says which of the two it was rather than leaving it to be guessed.
    worker.asked.connect(lambda event: (
        asked.append(event),
        sends.append((worker.answer('segy'), round(time.monotonic() - started_at, 2)))))
    loop = QEventLoop()
    worker.finished.connect(loop.quit)
    # `finished` is what ends this. The timer exists only so a genuinely stuck
    # child cannot hang the suite forever, so it is set far beyond any real
    # duration: this spawns a fresh interpreter, and inside the full suite that
    # has taken over three minutes just to reach its first line. Three earlier
    # attempts at a "reasonable" cap all produced the same false failure, each
    # time looking like a different bug in the protocol - the child was simply
    # still starting up.
    QTimer.singleShot(600000, loop.quit)
    worker.start()
    loop.exec()
    reached_the_child = not worker.isRunning()
    if worker.isRunning():
        worker.cancel()
        worker.wait(3000)
    # This spawns a real interpreter, so a failure has to carry enough to tell
    # why: whether the child started, what it wrote, and how it ended.
    evidence = (f'\nchild state: {worker.process.state()}'
                f'\nexit: {worker.process.exitCode()}/{worker.process.exitStatus()}'
                f'\nflags: finished={worker._finished} '
                f'cancelled={worker._cancelled} timed_out={worker._timed_out}'
                f'\nanswer (flushed?, at what second): {sends}'
                f'\nloop returned at: {round(time.monotonic() - started_at, 2)}s'
                f'\nasked: {asked}'
                f'\nprogress: {steps}'
                f'\nstdout: {bytes(worker._stdout)[-2000:]!r}'
                f'\nstderr: {bytes(worker._stderr)[-2000:]!r}')
    assert not errors, errors
    assert reached_the_child, (
        'the safety net fired, which means the child never finished - it was '
        'either stuck or the net is still too short for this machine'
        + evidence)
    assert results == [{'interpretation': 'done', 'answer': 'segy'}], (
        'the answer did not make it back through the protocol' + evidence)
    assert steps[0][0] == 'Report'
    assert asked and asked[0]['question'] == 'Which origin?'
    assert 'secret' not in (tmp_path / 'activity.log').read_text()


def test_report_figures_are_scaled_to_the_pane_not_their_own_width(tmp_path):
    """A print-resolution figure must fit the preview, and re-fit on resize."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import _FittedReportBrowser
    app = QApplication.instance() or QApplication([])
    # What a five-panel time-lapse section actually is: figsize=(20, 5) at 300 dpi.
    QImage(6000, 1500, QImage.Format_RGB32).save(str(tmp_path / 'wide.png'))
    QImage(400, 300, QImage.Format_RGB32).save(str(tmp_path / 'small.png'))

    browser = _FittedReportBrowser()
    browser.resize(1280, 700)
    browser.document().setBaseUrl(QUrl.fromLocalFile(str(tmp_path) + '/'))
    browser.setMarkdown('# Report\n\n![wide](wide.png)\n\n![small](small.png)\n')
    app.processEvents()

    def sizes():
        found = {}
        block = browser.document().begin()
        while block.isValid():
            it = block.begin()
            while not it.atEnd():
                image = it.fragment().charFormat().toImageFormat()
                if image.isValid() and image.name():
                    found[image.name()] = (image.width(), image.height())
                it += 1
            block = block.next()
        return found

    sized = sizes()
    wide_w, wide_h = sized['wide.png']
    assert wide_w <= browser.viewport().width(), 'figure still wider than the pane'
    assert wide_h == pytest.approx(wide_w / 4, rel=0.01), "aspect ratio not kept"
    # An image that already fits is left alone rather than blown up to the pane.
    assert sized['small.png'] == (400, 300)

    browser.resize(700, 700)
    app.processEvents()
    assert sizes()['wide.png'][0] <= browser.viewport().width()
    browser.close()
    browser.deleteLater()


def test_a_long_code_line_does_not_drag_the_report_sideways(tmp_path):
    """Qt marks code blocks non-breakable; one long JSON line widened the pane."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from PyHydroGeophysX.qt_apps.modules.one_click import _FittedReportBrowser
    app = QApplication.instance() or QApplication([])
    QImage(6000, 1500, QImage.Format_RGB32).save(str(tmp_path / 'wide.png'))

    browser = _FittedReportBrowser()
    browser.resize(1280, 700)
    browser.document().setBaseUrl(QUrl.fromLocalFile(str(tmp_path) + '/'))
    browser.setMarkdown('# Report\n\n![wide](wide.png)\n\n```\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n```\n')
    browser.show()
    app.processEvents()

    assert browser.document().idealWidth() <= browser.viewport().width(), (
        'the report still scrolls horizontally')
    browser.close()
    browser.deleteLater()
