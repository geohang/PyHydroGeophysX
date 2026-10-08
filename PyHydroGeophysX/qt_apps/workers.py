"""Generic background worker for the studio.

``TaskWorker`` handles non-workflow support tasks such as file parsing and
preview generation. ``ProcessProbeWorker`` and ``ProcessWorkflowWorker``
isolate native-library probes and workflows in separate Python processes;
every studio page runs its workflows that way, the live objects of a result
coming back through ``workflows.objects`` and read on a thread of their own.
The process a run needs is usually already started and waiting, its libraries
loaded (:class:`_Standby`). ``WorkflowWorker`` runs a workflow in a Qt thread
of this process, for a caller that wants it there.
"""

from __future__ import annotations

import codecs
import datetime
import json
from concurrent.futures import CancelledError
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Callable, Sequence

from PySide6.QtCore import (
    QCoreApplication,
    QObject,
    QProcess,
    QProcessEnvironment,
    QThread,
    QTimer,
    Signal,
)

from PyHydroGeophysX._internal.optional_dependencies import BackendUnavailable
# Where a run's ``logs`` folder keeps everything its workflow process printed.
from PyHydroGeophysX.qt_apps.run_records import LOG_ENCODING, OUTPUT_LOG_NAME
from PyHydroGeophysX.workflows import (
    RunContext,
    WorkflowRunResult,
    WorkflowSpec,
    run_workflow,
)
from PyHydroGeophysX.workflows.objects import load_objects, objects_folder


def _error_message(exc: Exception) -> str:
    if isinstance(exc, BackendUnavailable):
        return f"Backend unavailable: {exc}"
    return str(exc)


#: Terminal colour and cursor codes, as in ``ESC[0;32;49m``.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

#: Windows exit codes of a process ended by the system rather than by Python:
#: the code a native solver dies with, which on its own tells a user nothing.
_NATIVE_CRASHES = {
    0xC0000005: "a Windows access violation",
    0xC0000409: "a fatal Windows runtime error",
    0xC00000FD: "a stack overflow",
}
#: SIGSEGV as QProcess reports it (-11) and as a shell does (128 + 11).
_SEGFAULT_CODES = {-11, 139}
#: What a native solver or Python prints when memory runs out. CHOLMOD, the
#: sparse Cholesky solver behind the inversions, prints its errors as ordinary
#: output and the process then dies on a native exception.
_MEMORY_LINE = re.compile(
    r"CHOLMOD error|out of memory|MemoryError|bad_alloc|unable to allocate|"
    r"cannot allocate memory|not enough memory|insufficient memory",
    re.IGNORECASE,
)
#: The last exception of a workflow process that could not run for want of an
#: engine - an optional backend not installed, or one whose libraries would not
#: load - as Python prints it, with or without the module path. A subclass such
#: as gravmag's ``InversionBackendUnavailable`` counts.
_MISSING_BACKEND = re.compile(
    r"^(\w+\.)*\w*(BackendUnavailable|ModuleNotFoundError|ImportError)(: |$)")

#: The line a Python traceback opens with; an exception group's is indented
#: and marked ("  + Exception Group Traceback (most recent call last):").
_TRACEBACK_START = re.compile(
    r"^\s*(\+\s*)?(Exception Group )?Traceback \(most recent call last\):$")
#: What Python prints between the tracebacks of chained exceptions.
_TRACEBACK_CHAIN = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)
#: The line a traceback ends on: the exception, with or without its module
#: path and message - "IndexError:  1 [0..1)", "KeyboardInterrupt".
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.<>]*(:( |$)|$)")
#: The lines that can continue an exception's message after its first: an
#: indented line, pyGIMLi's C++ source location ("/vector.h:588  const
#: ValueType& GIMLI::Vector<ValueType>::getVal(...)"), a caret line ("~~~^^^").
_EXCEPTION_TAIL = re.compile(r"^(\s|\S*\.(h|hh|hpp|hxx|c|cc|cpp|cxx)(:\d+)?\b|[~^]+$)")


class _TracebackFilter:
    """Tell the lines of a Python traceback in a process's output from the rest.

    A workflow that fails prints its traceback, a dozen lines of file names,
    source lines and carets, and in the Log panel they buried the one line
    that says what went wrong - which the failure message gives anyway, with
    where the whole output is kept (``logs/workflow_output.log``, which keeps
    every line). One filter per output stream, whose lines arrive on their
    own. A block runs from "Traceback (most recent call last):" through the
    exception line and the lines that continue its message (see
    ``_EXCEPTION_TAIL``). A block that does not look like Python's ends at the
    first line that is neither indented nor an exception, at a blank line, or
    after ``MAX_LINES``, so it cannot hide the output that follows it.
    """

    MAX_LINES = 400       # a traceback's frames; Python folds repeated ones
    MAX_TAIL = 20         # lines continuing an exception's message

    def __init__(self) -> None:
        self._state = ""          # "", "body" (in a traceback) or "tail" (after its exception)
        self._count = 0
        self._chained = False
        #: The latest traceback's lines, chained ones together, for
        #: :func:`_plain_cause` to tell what its exception was about.
        self.block: list = []

    def hides(self, line: str) -> bool:
        """Whether ``line`` (its colour codes and trailing space removed) is
        part of a traceback, and so kept from the Log panel."""
        text = line.strip()
        if not text:                    # the end of a block; blank lines are not shown
            self._state = ""
            return True
        if _TRACEBACK_START.match(line):
            if not self._chained:
                self.block = []
            self._chained = False
            self._state, self._count = "body", 0
            return self._keep(text)
        if text in _TRACEBACK_CHAIN:
            self._state, self._chained = "", True
            return self._keep(text)
        self._count += 1
        if self._state == "body" and self._count <= self.MAX_LINES:
            if line[:1].isspace():      # File "...", line n / its source / carets
                return self._keep(text)
            if _EXCEPTION_LINE.match(text):
                self._state, self._count = "tail", 0
                return self._keep(text)
        elif self._state == "tail" and self._count <= self.MAX_TAIL \
                and _EXCEPTION_TAIL.match(line):
            return self._keep(text)
        self._state, self._chained = "", False
        return False

    def _keep(self, text: str) -> bool:
        if len(self.block) < self.MAX_LINES + self.MAX_TAIL:
            self.block.append(text)
        return True


#: Well-known failures in plain words: the exception, what its traceback must
#: mention as well (or None), and what it means. The exception follows them.
_PLAIN_CAUSES = (
    # pyGIMLi spaces the inversion mesh by the distance between the first two
    # electrodes; a file read with the wrong format can yield one or none.
    (re.compile(r"^IndexError\b"), re.compile(r"createParaMesh|sensors\[1\]"),
     "The data have fewer than two electrodes - check the Instrument / format setting"),
    (re.compile(r"No space left on device|\[Errno 28\]|not enough space on the disk",
                re.IGNORECASE), None,
     "The disk is full - free some space and run again"),
    (re.compile(r"^PermissionError\b"), None,
     "A file could not be opened or written - it may be open in another program, "
     "or the folder may be read-only"),
)


def _plain_cause(exception: str, traceback_text: str = "") -> str:
    """What ``exception`` most likely means, in plain words, or "" if unknown."""
    for said, context, meaning in _PLAIN_CAUSES:
        if said.search(exception) and (context is None or context.search(traceback_text)):
            return meaning
    return ""


def _process_failure_message(
    exit_code: int,
    crashed: bool,
    last_exception: str = "",
    memory_hint: str = "",
    log_path: Path | None = None,
    traceback_text: str = "",
) -> str:
    """Say in one plain sentence why a workflow process ended, what to try, and
    where everything it printed is kept.

    A native crash is reported as one, most likely a lack of memory (the
    solvers' usual way of dying), with the line the process printed about it if
    it printed one; an error the workflow raised is reported as that error -
    after what it means, when it is a well-known one (``_PLAIN_CAUSES``, which
    may look at the ``traceback_text``) - and only a process that ended
    without saying why is left with its exit code. The page says what failed
    ("ERT inversion failed: ...").
    """
    code = int(exit_code)
    unsigned = code & 0xFFFFFFFF
    where = f"; the full output is in {log_path}" if log_path else ""
    advice = ("close other programs to free memory, or use a coarser mesh or fewer "
              "cells, and run again")

    def sentence(text: str) -> str:
        text = text.rstrip()
        if where:
            return f"{text.rstrip('.')}{where}."
        return text if text.endswith((".", "!", "?")) else f"{text}."

    # The error itself: the exit code of a Python error is always 1. The module
    # path its class is printed with, and pyGIMLi's doubled spaces, are noise.
    said = re.sub(r"^(\w+\.)+(?=\w+(: |:?$))", "", last_exception.strip())
    said = " ".join(said.split())
    if len(said) > 300:
        said = said[:299] + "…"
    native = _NATIVE_CRASHES.get(unsigned)
    if native or code in _SEGFAULT_CODES or crashed:
        how = (f"{native} (0x{unsigned:08X})" if native
               else "a segmentation fault" if code in _SEGFAULT_CODES
               else f"exit code {code} (0x{unsigned:08X})")
        cause = (f"after reporting \"{memory_hint}\", so it ran out of memory" if memory_hint
                 else "most likely because it ran out of memory")
        return sentence(f"The solver process crashed with {how}, {cause} - {advice}")
    if memory_hint or _MEMORY_LINE.search(said):
        return sentence(f"The computer ran out of memory ({(said or memory_hint).rstrip('.')})"
                        f" - {advice}")
    if said:
        meaning = _plain_cause(said, traceback_text)
        return sentence(f"{meaning} ({said.rstrip('.')})" if meaning else said)
    return sentence(f"Workflow process exited with code {code} (0x{unsigned:08X})")


def _active_console_python() -> Path:
    """Return the console Python launcher belonging to the active environment."""
    prefix_python = Path(sys.prefix) / "Scripts" / "python.exe"
    executable = prefix_python if prefix_python.is_file() else Path(sys.executable)
    if executable.name.lower().startswith("pythonw"):
        console_python = executable.with_name(
            executable.name.replace("pythonw", "python", 1)
        )
        if console_python.is_file():
            executable = console_python
    return executable


def _package_root() -> Path:
    """The folder holding the PyHydroGeophysX package this studio runs."""
    return Path(__file__).resolve().parents[2]


def _workflow_environment() -> QProcessEnvironment:
    """The environment a workflow process runs in.

    A GUI process on Chinese Windows can give redirected Python streams the
    system GBK encoding. Workflow logs legitimately contain Unicode text such as
    ``chi²`` and an ellipsis; printing either would then raise
    UnicodeEncodeError after a successful inversion. UTF-8 matches the decoder
    ``ProcessWorkflowWorker._emit_output`` reads the streams with.
    """
    environment = QProcessEnvironment.systemEnvironment()
    environment.insert("PYTHONUTF8", "1")
    environment.insert("PYTHONIOENCODING", "utf-8")
    # The process must import the PyHydroGeophysX this studio runs. Started from
    # a checkout, the studio found the package through its own working folder
    # or sys.path, which a process started in the project folder does not
    # share, and that process imported whatever other copy the environment
    # held: an older checkout lacked workflow_standby and every run failed. An
    # installed package is already on the child's path and is left alone.
    package_root = _package_root()
    if package_root.name.lower() not in ("site-packages", "dist-packages"):
        existing = [entry for entry in environment.value("PYTHONPATH", "").split(os.pathsep)
                    if entry and os.path.normcase(os.path.abspath(entry))
                    != os.path.normcase(str(package_root))]
        environment.insert("PYTHONPATH", os.pathsep.join([str(package_root), *existing]))
    return environment


#: Set to 0 to start every workflow process when its run starts, as before
#: :class:`_Standby` kept one waiting.
WARM_WORKER_ENV = "PHGX_WARM_WORKER"


class _Standby:
    """The workflow process started ahead of the next run.

    Starting a workflow process cost about a second before the workflow ran -
    the interpreter and the numerical libraries it imports - and every run paid
    it while the user watched. One process is kept started and waiting instead
    (``PyHydroGeophysX._internal.workflow_standby``), having loaded those
    libraries while nobody was waiting; a run takes it over, and the next one is
    started at once. Each still serves one run only, so a run is cancelled by
    ending its process and cannot inherit anything from the one before.

    Idle, it holds the libraries in memory, about 200 MB; ``PHGX_WARM_WORKER=0``
    turns it off. One that dies while waiting is replaced, and after three such
    deaths no more are started this session: every run then starts its own
    process, as it always did.
    """

    MODULE = "PyHydroGeophysX._internal.workflow_standby"
    MAX_FAILURES = 3

    def __init__(self) -> None:
        self._process: QProcess | None = None
        self._slots: list = []            # what the waiting process is connected to
        self._failures = 0
        self._quit_hooked = False

    @staticmethod
    def enabled() -> bool:
        flag = str(os.environ.get(WARM_WORKER_ENV, "1")).strip().lower()
        return flag not in {"0", "false", "no", "off"}

    def prepare(self) -> None:
        """Start a process for the next run, unless one is already waiting."""
        app = QCoreApplication.instance()
        if (app is None or not self.enabled() or self._failures >= self.MAX_FAILURES
                or (self._process is not None
                    and self._process.state() != QProcess.ProcessState.NotRunning)):
            return
        if not self._quit_hooked:
            app.aboutToQuit.connect(self.shutdown)
            self._quit_hooked = True
        # It waits in the folder holding the package it runs, which the studio
        # never deletes; a run moves it to the run's own folder. Windows will not
        # delete a folder that is a process's working directory, and waiting in
        # the folder of the run that started it left that run impossible to
        # discard (WinError 32) until the next run took the process away.
        process = QProcess(app)
        process.setWorkingDirectory(str(_package_root()))
        process.setProcessEnvironment(_workflow_environment())
        process.setProgram(str(_active_console_python()))
        process.setArguments(["-m", self.MODULE])
        # Nothing it says while it waits is anyone's: the warm-up is quiet, and
        # a run's output begins once a run owns the process.
        self._slots = [
            (process.readyReadStandardOutput, lambda: process.readAllStandardOutput()),
            (process.readyReadStandardError, lambda: process.readAllStandardError()),
            (process.finished, lambda *_args: self._on_ended(process)),
        ]
        for signal, slot in self._slots:
            signal.connect(slot)
        process.start()
        self._process = process

    def take(self) -> QProcess | None:
        """The waiting process, handed over; None when there is none to hand."""
        process, self._process = self._process, None
        slots, self._slots = self._slots, []
        if process is None:
            return None
        for signal, slot in slots:
            try:
                signal.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        if process.state() == QProcess.ProcessState.NotRunning:
            self._failures += 1           # it died waiting, before its end was heard
            process.deleteLater()
            return None
        return process

    def _on_ended(self, process: QProcess) -> None:
        """A waiting process ended before any run took it."""
        if process is self._process:
            self._process = None
            self._slots = []
            self._failures += 1
        try:
            process.deleteLater()
        except RuntimeError:
            # Being destroyed with the application, which ended the process.
            pass

    def shutdown(self) -> None:
        """End the waiting process: end of input tells it no run is coming."""
        process, self._process = self._process, None
        try:
            if process is None or process.state() == QProcess.ProcessState.NotRunning:
                return
            process.closeWriteChannel()
            if not process.waitForFinished(1500):
                process.kill()
                process.waitForFinished(1500)
        except RuntimeError:
            pass                          # already destroyed, with the application


_STANDBY = _Standby()


def prepare_workflow_process() -> None:
    """Have a workflow process started and waiting for the next run.

    Called once the studio is up, so its first run does not wait either; each
    run that takes the process starts the next one itself.
    """
    _STANDBY.prepare()


def _kill_process_tree(pid: int) -> None:
    """Freeze the task tree, then kill descendants before their parent.

    Keeping psutil Process objects also guards against PID reuse. Freezing each
    parent before enumerating its children prevents it from starting workers
    while cancellation is collecting the tree. No waits block the Qt event loop.
    """
    try:
        import psutil
    except ImportError as exc:
        raise OSError("Cancelling a process tree requires the desktop dependency psutil") from exc

    tree = []
    try:
        try:
            pending = [psutil.Process(int(pid))]
        except psutil.NoSuchProcess:
            return
        while pending:
            process = pending.pop()
            try:
                process.suspend()
                tree.append(process)
                pending.extend(process.children())
            except psutil.NoSuchProcess:
                continue
        # Parent-before-child collection gives a child-before-parent kill order.
        for process in reversed(tree):
            try:
                process.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.Error as exc:
        raise OSError(f"Could not end the workflow process tree {pid}: {exc}") from exc
    finally:
        # A permission error must not strand any surviving process suspended.
        for process in reversed(tree):
            try:
                process.resume()
            except psutil.Error:
                pass


class TaskWorker(QThread):
    """Run ``fn(*args, **kwargs)`` off the UI thread.

    ``succeeded`` carries the return value, ``failed`` the error text, and
    ``logged`` optional progress strings. When ``with_log=True`` the callable is
    given a ``log`` keyword (a function taking one string) so it can report
    progress. :meth:`cancel` requests interruption; it does not forcibly stop
    the callable. The supplied log callback checks cancellation and raises
    CancelledError at that checkpoint. With ``with_cancel=True``, the callable
    also receives ``cancelled()``, which it can poll between expensive steps.
    Completion/error and new progress emissions are suppressed after cancellation
    is observed; signals already queued in Qt may still be delivered.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args: Any, with_log: bool = False,
                 with_cancel: bool = False, **kwargs: Any) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self._with_log = with_log
        self._with_cancel = with_cancel
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancelled or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        if self.is_cancelled():
            return
        try:
            kwargs = dict(self._kwargs)
            if self._with_log:
                callback = kwargs.get("log", lambda m: self.logged.emit(str(m)))
                def log(message):
                    if self.is_cancelled():
                        raise CancelledError()
                    callback(message)
                kwargs["log"] = log
            if self._with_cancel:
                kwargs.setdefault("cancelled", self.is_cancelled)
            result = self._fn(*self._args, **kwargs)
            if self.is_cancelled():
                return
            self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            if not self.is_cancelled():
                self.failed.emit(_error_message(exc))


class WorkflowWorker(QThread):
    """Execute a serializable workflow without coupling its handler to Qt."""

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)

    def __init__(self, spec: WorkflowSpec, context: RunContext) -> None:
        super().__init__()
        self._spec = spec
        self._context = context
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True
        self.requestInterruption()

    def is_cancelled(self) -> bool:
        return self._cancelled or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: D401 - QThread entry point
        if self.is_cancelled():
            return
        try:
            def progress(message):
                if self.is_cancelled():
                    raise CancelledError()
                self.logged.emit(str(message))
            self._context.progress = progress
            self._context.cancelled = self.is_cancelled
            result = run_workflow(self._spec, self._context)
            if not self.is_cancelled():
                self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001
            if not self.is_cancelled():
                self.failed.emit(_error_message(exc))


class ProcessProbeWorker(QObject):
    """Run a JSON-emitting diagnostic module in a clean Python process.

    GPU libraries load several native Windows DLLs. Importing them from a
    ``QThread`` still uses the studio process and can therefore inherit a
    conflicting OpenMP DLL that another visualization dependency loaded first.
    A fresh interpreter has its own DLL namespace and matches the isolation used
    by the actual ERT inversion workflow.

    The child module must print a final JSON object with either
    ``{"ok": true, "result": {...}}`` or ``{"ok": false, "error": "..."}``.
    Earlier stdout/stderr is retained only as a diagnostic if the contract is
    not satisfied.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        module: str,
        *,
        timeout_ms: int = 60000,
        working_directory: str | Path | None = None,
        arguments: list[str] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.module = str(module)
        self.timeout_ms = max(1000, int(timeout_ms))
        self._cancelled = False
        self._finished = False
        self._timed_out = False
        self._stdout = bytearray()
        self._stderr = bytearray()

        self.process = QProcess(self)
        if working_directory is not None:
            self.process.setWorkingDirectory(str(Path(working_directory).resolve()))
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUTF8", "1")
        environment.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(environment)
        self.process.setProgram(str(_active_console_python()))
        self.process.setArguments(["-m", self.module, *(arguments or [])])
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)

    def start(self) -> None:
        self._timer.start(self.timeout_ms)
        self.process.start()

    def cancel(self) -> None:
        self._cancelled = True
        self._timer.stop()
        if self.isRunning():
            self.process.terminate()
            QTimer.singleShot(2000, self._kill_if_running)

    def quit(self) -> None:
        """QThread-compatible shutdown hook used by ``BaseModule``."""
        self.cancel()

    def wait(self, timeout_ms: int = 30000) -> bool:
        return bool(self.process.waitForFinished(int(timeout_ms)))

    def isRunning(self) -> bool:  # noqa: N802 - mirror QThread's public API
        return self.process.state() != QProcess.ProcessState.NotRunning

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _kill_if_running(self) -> None:
        if self.isRunning():
            self.process.kill()

    def _read_stdout(self) -> None:
        self._stdout.extend(bytes(self.process.readAllStandardOutput()))

    def _read_stderr(self) -> None:
        self._stderr.extend(bytes(self.process.readAllStandardError()))

    def _on_timeout(self) -> None:
        if self._finished:
            return
        self._timed_out = True
        if self.isRunning():
            self.process.kill()
        else:
            self._finish_with_error(
                f"Probe process timed out after {self.timeout_ms / 1000:g} seconds."
            )

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if self._finished:
            return
        if error == QProcess.ProcessError.FailedToStart:
            self._finish_with_error(
                f"Could not start probe process: {self.process.errorString()}"
            )
            return
        # A read or write error does not end the process, and nothing else here
        # would have noticed: the run hung with no message and no result. That
        # became reachable when the input channel started staying open for the
        # whole run so the child could be answered mid-run - a failed write to
        # that channel means the child is waiting for an answer it will never
        # get. Killing it turns a silent hang into a reported failure, which is
        # the difference between a bug report and a mystery.
        if error in (QProcess.ProcessError.WriteError,
                     QProcess.ProcessError.ReadError):
            channel = "sending to" if error == QProcess.ProcessError.WriteError \
                else "reading from"
            if self.process.state() != QProcess.ProcessState.NotRunning:
                self.process.kill()
            self._finish_with_error(
                f"Lost contact with the workflow process while {channel} it: "
                f"{self.process.errorString()}")

    def _finish_with_error(self, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        if not self._cancelled:
            self.failed.emit(message)
        self.finished.emit()

    @staticmethod
    def _last_json_object(raw: bytes) -> dict[str, Any]:
        text = raw.decode("utf-8", errors="replace")
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "ok" in payload:
                return payload
        raise ValueError("probe did not return its JSON result")

    def _on_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        if self._finished:
            return
        self._timer.stop()
        self._read_stdout()
        self._read_stderr()
        if self._cancelled:
            self._finished = True
            self.finished.emit()
            return
        if self._timed_out:
            self._finish_with_error(
                f"Probe process timed out after {self.timeout_ms / 1000:g} seconds."
            )
            return
        if int(exit_code) != 0:
            unsigned = int(exit_code) & 0xFFFFFFFF
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            suffix = f" {detail[-2000:]}" if detail else ""
            self._finish_with_error(
                f"Probe process exited with code {int(exit_code)} "
                f"(0x{unsigned:08X}).{suffix}"
            )
            return
        try:
            payload = self._last_json_object(bytes(self._stdout))
        except Exception as exc:  # noqa: BLE001 - malformed child output
            detail = self._stderr.decode("utf-8", errors="replace").strip()
            suffix = f" Stderr: {detail[-2000:]}" if detail else ""
            self._finish_with_error(f"Could not read probe result: {exc}.{suffix}")
            return
        if not bool(payload.get("ok")):
            self._finish_with_error(str(payload.get("error") or "GPU probe failed."))
            return
        result = payload.get("result")
        if not isinstance(result, dict):
            self._finish_with_error("GPU probe returned an invalid result object.")
            return
        self._finished = True
        self.succeeded.emit(result)
        self.finished.emit()


class _ResultLoader(QThread):
    """Read a finished run's result, and the objects it sent back, off the UI thread.

    Reading a large result took the window half a second: the JSON, then the
    mesh and the arrays the page shows (``workflows.objects``). Here that work
    no longer stops the window painting - except while PyGIMLi builds a mesh,
    which holds the interpreter wherever it runs.
    """

    loaded = Signal(object, object)       # the result, what did not come back
    failed = Signal(str)

    def __init__(self, result_path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.result_path = Path(result_path)

    def run(self) -> None:  # noqa: D401 - QThread entry point
        try:
            payload = json.loads(self.result_path.read_text(encoding="utf-8"))
            result = WorkflowRunResult.from_dict(payload)
        except Exception as exc:  # noqa: BLE001 - report malformed child output
            self.failed.emit(f"Could not read workflow result {self.result_path}: {exc}")
            return
        problems: list = []
        # The meshes, arrays and models the page shows, which a result read
        # from JSON would otherwise lack; one that did not come back is said.
        manifest = payload.get("objects") if isinstance(payload, dict) else None
        if manifest:
            folder = objects_folder(self.result_path)
            objects, problems = load_objects(manifest, folder)
            result.objects.update(objects)
            # Read into memory: the copies need not stay in the Project, whose
            # record of the run is the outputs the workflow saved.
            shutil.rmtree(folder, ignore_errors=True)
        self.loaded.emit(result, problems)


class ProcessWorkflowWorker(QObject):
    """Execute a recipe in an isolated Python process.

    A ``QThread`` still shares the interpreter and GIL with Qt's event loop.
    Some long-running extension calls do not release that GIL often enough, so
    the window stops painting even though the workflow is nominally off-thread.
    ``QProcess`` gives the workflow its own interpreter and makes cancellation
    enforceable without terminating the studio: :meth:`cancel` ends the process
    and everything it started where it stands.
    """

    succeeded = Signal(object)
    failed = Signal(str)
    logged = Signal(str)
    progressed = Signal(int, int, str)
    finished = Signal()

    def __init__(
        self,
        recipe_path: str | Path,
        project_root: str | Path,
        output_dir: str | Path,
        result_path: str | Path,
        parent: QObject | None = None,
        *,
        objects: Sequence[str] = (),
    ) -> None:
        """``objects`` names what of the result's live objects - a mesh, an
        array, a table - the page needs back in ``result.objects``; the
        workflow process writes those beside its result and they are read in
        here (``workflows.objects``). A run in a thread handed over all of
        them; asking for what is shown saves loading what is not."""
        super().__init__(parent)
        self.recipe_path = Path(recipe_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.result_path = Path(result_path).resolve()
        self.objects = tuple(str(name) for name in objects)
        self._cancelled = False
        self._finished = False
        self._output_decoders = {
            stream: codecs.getincrementaldecoder("utf-8")(errors="replace")
            for stream in ("stdout", "stderr")
        }
        self._output_pending = {"stdout": "", "stderr": ""}
        #: What of each stream is a traceback, kept from the Log panel.
        self._tracebacks = {stream: _TracebackFilter() for stream in ("stdout", "stderr")}
        #: The child's last exception line, e.g. "ValueError: 'x.dat' could not
        #: be read as DAS-1: ...". A failed run reported only its exit code, and
        #: the reason was left somewhere in the log above.
        self._last_exception = ""
        #: The first line the child printed about running out of memory, which a
        #: native solver prints as ordinary output just before it crashes.
        self._memory_hint = ""
        #: Whether the run failed because an engine it needs is missing, rather
        #: than in its own work. Pages that fall back to exporting their
        #: configuration when a backend is absent did so for every failure, and
        #: a run that ran out of memory or was handed arrays of the wrong shape
        #: was reported as "backend not found". Set when the failure is.
        self.missing_backend = False
        #: The child's lines not yet appended to the run's output log.
        self._output_unwritten: list = []
        self._output_log_started = False
        self._loader: _ResultLoader | None = None

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.project_root))
        self.process.setProcessEnvironment(_workflow_environment())
        # ``sys.executable`` can resolve to the Windows Store base executable
        # even when the studio was launched through ``.venv\Scripts``.  A
        # direct QProcess launch of that MSIX executable bypasses the venv
        # launcher and has crashed in extension DLL initialization (notably
        # pyarrow/arrow.dll).  Prefer the console launcher belonging to the
        # active prefix; QProcess supplies pipes so it does not open a console.
        self.process.setProgram(str(_active_console_python()))
        self.process.setArguments([
            "-m",
            "PyHydroGeophysX.workflows.cli",
            "run",
            str(self.recipe_path),
            "--project-root",
            str(self.project_root),
            "--output-dir",
            str(self.output_dir),
            "--result-file",
            str(self.result_path),
            *(["--objects", ",".join(self.objects)] if self.objects else []),
        ])
        self.process.started.connect(self._cancel_if_requested)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.errorOccurred.connect(self._on_process_error)
        self.process.finished.connect(self._on_finished)

    def start(self) -> None:
        try:
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            self.result_path.unlink(missing_ok=True)
            shutil.rmtree(objects_folder(self.result_path), ignore_errors=True)
        except OSError as exc:
            self._finish_with_error(f"Could not prepare workflow result {self.result_path}: {exc}")
            return
        waiting = _STANDBY.take()
        if waiting is None:
            self.process.start()
        else:
            self._run_in(waiting)
        # The next run's process, loading its libraries while this one runs.
        _STANDBY.prepare()

    def _run_in(self, process: QProcess) -> None:
        """Run on a process started ahead of this run (see :class:`_Standby`).

        The command line is the one a process of its own would have been given,
        handed over on standard input; from there the process is this run's as
        if it had been started for it.
        """
        own, self.process = self.process, process
        process.setParent(self)
        process.started.connect(self._cancel_if_requested)
        process.readyReadStandardOutput.connect(self._read_stdout)
        process.readyReadStandardError.connect(self._read_stderr)
        process.errorOccurred.connect(self._on_process_error)
        process.finished.connect(self._on_finished)
        arguments = list(own.arguments())
        job = {"module": arguments[1], "argv": arguments[2:], "cwd": str(self.project_root)}
        own.deleteLater()
        line = (json.dumps(job) + "\n").encode("utf-8")
        if process.state() == QProcess.ProcessState.Running:
            process.write(line)
        else:                                   # still starting
            process.started.connect(lambda: process.write(line))

    def cancel(self) -> None:
        self._cancelled = True
        if self.isRunning():
            self._kill_if_running()
            QTimer.singleShot(2000, self._kill_if_running)

    def _cancel_if_requested(self) -> None:
        # Cancellation can arrive while QProcess is Starting and has no PID yet.
        if self._cancelled:
            self._kill_if_running()

    def quit(self) -> None:
        """QThread-compatible shutdown hook used by ``BaseModule``."""
        self.cancel()

    def wait(self, timeout_ms: int = 30000) -> bool:
        done = bool(self.process.waitForFinished(int(timeout_ms)))
        if self._loader is not None:
            self._loader.wait(int(timeout_ms))
        return done

    def isRunning(self) -> bool:  # noqa: N802 - mirror QThread's public API
        # Still running while its result is read: nothing has been delivered.
        return (self.process.state() != QProcess.ProcessState.NotRunning
                or (self._loader is not None and self._loader.isRunning()))

    def is_cancelled(self) -> bool:
        return self._cancelled

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            pid = int(self.process.processId())
            if pid > 0:
                try:
                    _kill_process_tree(pid)
                except OSError as exc:
                    self.logged.emit(str(exc))
            self.process.kill()

    def _emit_output(self, raw: bytes, stream: str = "stdout", *, final: bool = False) -> None:
        if self._cancelled:
            return
        text = self._output_pending[stream] + self._output_decoders[stream].decode(raw, final=final)
        lines = text.splitlines(keepends=True)
        self._output_pending[stream] = ""
        if lines and not final and not lines[-1].endswith(("\n", "\r")):
            self._output_pending[stream] = lines.pop()
        for line in lines:
            # pyGIMLi colours its log lines for a terminal; in the log panel and
            # the run's logs the codes are only noise ("[0;32;49mINFO[0m").
            rendered = _ANSI_ESCAPE.sub("", line).rstrip()
            # A traceback goes to the output log only; the failure message
            # names its exception and where that log is.
            traceback_line = self._tracebacks[stream].hides(rendered)
            if not rendered.strip():
                continue
            self._output_unwritten.append(
                rendered if stream == "stdout" else f"[stderr] {rendered}")
            if not self._memory_hint and _MEMORY_LINE.search(rendered):
                self._memory_hint = rendered.strip()[:300]
            if stream == "stderr" and re.match(
                    r"^\w+(\.\w+)*(Error|Exception|Unavailable)(: |$)", rendered.strip()):
                self._last_exception = rendered.strip()
            if traceback_line:
                continue
            match = re.match(
                r"^\[progress\s+(\d+)/(\d+)\]\s*(.*)$", rendered.strip()
            )
            if match is not None:
                current, total = int(match.group(1)), int(match.group(2))
                label = match.group(3).strip()
                if total > 0 and 0 <= current <= total:
                    self.progressed.emit(current, total, label)
                rendered = label or rendered
            self.logged.emit(rendered)
        # As it arrives rather than at the end: a crash of the studio itself
        # still leaves everything printed so far.
        self._append_output_log()

    def _read_stdout(self) -> None:
        self._emit_output(bytes(self.process.readAllStandardOutput()))

    def _read_stderr(self) -> None:
        self._emit_output(bytes(self.process.readAllStandardError()), "stderr")

    def _append_output_log(self, closing: str = "") -> Path | None:
        """Append the child's new lines to its run's ``logs/workflow_output.log``.

        The page's log panel is gone with the session, and a run that crashed
        otherwise left an empty ``logs`` folder and an exit code behind. All of
        the output is kept: a 2000-line tail lost the start of a long windowed
        run - its settings, its data checks - which is where a failure is
        usually explained. ``closing`` is a last line, the exit code. Written
        only where the result sits in a run folder that has a ``logs`` folder;
        lines a write could not take (a file held a moment by OneDrive) wait
        for the next.
        """
        folder = self.result_path.parent / "logs"
        if closing:
            self._output_unwritten.append(closing)
        if not self._output_unwritten or not folder.is_dir():
            return None
        path = folder / OUTPUT_LOG_NAME
        lines = list(self._output_unwritten)
        if not self._output_log_started:
            # One run folder can see more than one workflow process; each says
            # where its own output begins.
            lines.insert(0, f"[{self.recipe_path.name}, started "
                            f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}]")
        try:
            with open(path, "a", encoding=LOG_ENCODING, newline="\n") as handle:
                handle.write("\n".join(lines) + "\n")
        except OSError:
            return path if path.is_file() else None
        self._output_log_started = True
        self._output_unwritten.clear()
        return path

    @staticmethod
    def _exit_line(exit_code: int, how: str = "exit code") -> str:
        return f"[{how} {exit_code} (0x{exit_code & 0xFFFFFFFF:08X})]"

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart and not self._finished:
            self._finish_with_error(
                f"Could not start workflow process: {self.process.errorString()}"
            )

    def _finish_with_error(self, message: str) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._cancelled:
            self.failed.emit(message)
        self.finished.emit()

    def _on_finished(
        self, exit_code: int, _exit_status: QProcess.ExitStatus
    ) -> None:
        if self._finished:
            return
        self._read_stdout()
        self._read_stderr()
        self._emit_output(b"", "stdout", final=True)
        self._emit_output(b"", "stderr", final=True)
        if self._cancelled:
            self._append_output_log(self._exit_line(int(exit_code), "stopped by the user; exit code"))
            self._finished = True
            self.finished.emit()
            return
        log_path = self._append_output_log(self._exit_line(int(exit_code)))
        crashed = _exit_status == QProcess.ExitStatus.CrashExit
        if int(exit_code) != 0 or crashed:
            # Only an ordinary Python error exit (code 1) ended on the exception
            # it names; a crash after some stray import warning did not.
            self.missing_backend = (
                int(exit_code) == 1 and not crashed and not self._memory_hint
                and bool(_MISSING_BACKEND.match(self._last_exception)))
            block = self._tracebacks["stderr"].block or self._tracebacks["stdout"].block
            self._finish_with_error(_process_failure_message(
                int(exit_code), crashed, self._last_exception, self._memory_hint, log_path,
                "\n".join(block)))
            return
        self.logged.emit("Loading the results…")
        loader = _ResultLoader(self.result_path, self)
        loader.loaded.connect(self._on_loaded)
        loader.failed.connect(self._on_load_failed)
        self._loader = loader
        loader.start()

    def _on_loaded(self, result: WorkflowRunResult, problems) -> None:
        if self._loader is not None:
            self._loader.wait()           # at the end of run(); it returns at once
        for problem in problems or ():
            self.logged.emit(f"Not brought back from the workflow process: {problem}")
        if self._finished:
            return
        self._finished = True
        if not self._cancelled:
            self.succeeded.emit(result)
        self.finished.emit()

    def _on_load_failed(self, message: str) -> None:
        if self._loader is not None:
            self._loader.wait()
        self._finish_with_error(message)


__all__ = [
    "ProcessProbeWorker",
    "ProcessWorkflowWorker",
    "TaskWorker",
    "WorkflowWorker",
]
