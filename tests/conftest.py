"""Suite-wide Qt setup.

Qt allows exactly one application object per process, and the first test to ask
for one decides its type for every test that follows. That is a trap in a mixed
suite: ``test_gpu_probe_process.py`` needs only a ``QCoreApplication`` (it drives
a ``QProcess``, no widgets), but if it runs first its bare core application is
what the later widget tests get, and constructing a ``QWidget`` without a
``QApplication`` aborts the interpreter with no Python traceback.

Creating the widget-capable application here, once, before any test module is
imported, settles the type up front so no individual test's choice can starve
another. Tests keep their own ``QApplication.instance() or QApplication([])``
lines; those now always find this instance.

Nothing here is required when PySide6 is absent — the Qt tests skip themselves,
and this fixture stays out of the way.
"""

import os

import pytest

# Must be set before the first Qt import, not just before the first widget:
# the platform plugin is chosen when QtGui initialises.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """The one QApplication for the whole session, or None without PySide6."""
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - environment without desktop extras
        yield None
        return
    app = QApplication.instance() or QApplication([])
    yield app
    # Deliberately not destroyed: tearing down the application object while
    # widgets from finished tests are still awaiting collection crashes Qt, and
    # the process is about to exit anyway.


@pytest.fixture(autouse=True)
def drain_deferred_deletions(qt_application):
    """Run the ``deleteLater`` calls a test made before the next test starts.

    ``close()`` only hides a widget; the object lives until something deletes
    it. Left alone that falls to Python's cyclic collector - and a QWidget is
    always in a cycle, since its own signal connections hold it - so the C++
    object is destroyed from inside a collection pass at an arbitrary later
    moment. PySide turns some of those into a silent ``abort()``: no failure,
    no traceback, just a dead interpreter in whatever test was running at the
    time, which is rarely the one that leaked the widget.

    A test that creates a widget should therefore call ``deleteLater()`` on it,
    and this fixture drains the queue afterwards - ``processEvents`` does not,
    deferred deletion needs asking for by name.
    """
    yield
    if qt_application is None:
        return
    from PySide6.QtCore import QEvent

    qt_application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
