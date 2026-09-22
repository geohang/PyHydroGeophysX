"""Cancellable process with live progress and credentials passed via stdin."""
import codecs
import json
from pathlib import Path
from PySide6.QtCore import Signal, QProcess
from PyHydroGeophysX.qt_apps.workers import ProcessProbeWorker


class OneClickWorker(ProcessProbeWorker):
    """Runs one workflow in its own process, reporting where it has got to.

    ``progress`` carries a fourth value the earlier protocol did not: the studio
    module the current step belongs to, or ``''``. The workflow runs out of
    process, so this is how "show me where it is working" crosses the boundary -
    the runtime reports a module key and the desktop decides what to do with it.
    """

    progress = Signal(str, float, str, str)
    logged = Signal(str)
    #: The run has paused and needs an answer: either "may I run this step?"
    #: or a question with options. Carries the event dict; reply with
    #: :meth:`answer`.
    asked = Signal(dict)

    def __init__(self, payload, parent=None):
        super().__init__('PyHydroGeophysX.qt_apps.agent.one_click_runner',
                         timeout_ms=2147483647, parent=parent)
        # One line, and the write channel stays open: the child reads this
        # line to start, then blocks on the same stream whenever it needs an
        # answer. Closing it after the payload - which is what this did - makes
        # every later question read end-of-input and give up.
        self._payload = (json.dumps(payload) + '\n').encode('utf-8')
        self._log_path = Path(payload['output_dir']) / 'activity.log'
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = self.process.processEnvironment()
        environment.insert('PYTHONUNBUFFERED', '1')
        self.process.setProcessEnvironment(environment)
        self._decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self._pending = ''
        self.process.started.connect(self._send_input)

    #: How long to wait for a line to actually reach the child. Short, because
    #: it is a few dozen bytes down a pipe that the child is already blocked
    #: reading; long enough that a busy machine does not lose it.
    WRITE_TIMEOUT_MS = 5000

    def _send_input(self):
        self._write(self._payload)
        self._payload = b''

    def answer(self, decision):
        """Reply to the pending question.

        Parameters
        ----------
        decision : str
            For a step approval, ``'proceed'``, ``'skip'`` or ``'stop'``. For a
            question, the id of the chosen option.
        """
        return self._write(
            (json.dumps({'decision': str(decision)}) + '\n').encode('utf-8'))

    def _write(self, payload):
        """Write to the child and make sure it actually leaves this process.

        ``QProcess.write`` only buffers; the bytes go out when the event loop
        services the write notifier. That is normally the next iteration, but
        an answer is written from inside the *read* handler that delivered the
        question - the desktop's button press runs there - and the flush could
        then be deferred indefinitely. The child sat blocked on its stdin for
        the whole run while the answer the user had already given waited in a
        buffer on this side, which looks exactly like the agent freezing.

        Waiting for the bytes is cheap here: it is one short line down a pipe
        the child is already blocked reading.
        """
        if not payload:
            return True
        self.process.write(payload)
        return bool(self.process.waitForBytesWritten(self.WRITE_TIMEOUT_MS))

    def _read_stdout(self):
        raw = bytes(self.process.readAllStandardOutput())
        self._stdout.extend(raw)
        if len(self._stdout) > 4 * 1024 * 1024:
            del self._stdout[:-4 * 1024 * 1024]
        if raw:
            with self._log_path.open('ab') as stream:
                stream.write(raw)
        self._pending += self._decoder.decode(raw)
        while '\n' in self._pending:
            line, self._pending = self._pending.split('\n', 1)
            try:
                event = json.loads(line)
            except ValueError:
                self.logged.emit(line)
                continue
            if isinstance(event, dict) and event.get('event') == 'progress':
                self.progress.emit(str(event['step']), float(event['progress']),
                                   str(event.get('details', '')),
                                   str(event.get('module', '')))
            elif isinstance(event, dict) and event.get('event') in ('approve', 'question'):
                self.asked.emit(event)
            elif not isinstance(event, dict) or 'ok' not in event:
                self.logged.emit(line)

    def _read_stderr(self):
        raw = bytes(self.process.readAllStandardError())
        self._stderr.extend(raw)
        if len(self._stderr) > 65536:
            del self._stderr[:-65536]
        if raw:
            with self._log_path.open('ab') as stream:
                stream.write(raw)
            self.logged.emit(raw.decode('utf-8', errors='replace'))

    def _on_finished(self, code, status):
        self._read_stdout()
        if status == QProcess.ExitStatus.CrashExit and not self._cancelled:
            self._finish_with_error(f'Workflow process crashed (exit code {code}). See the run log.')
            return
        if code and not self._cancelled:
            try:
                payload = self._last_json_object(bytes(self._stdout))
                if payload.get('error'):
                    self._finish_with_error(str(payload['error']))
                    return
            except ValueError:
                pass
        super()._on_finished(code, status)
