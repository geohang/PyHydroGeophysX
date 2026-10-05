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
