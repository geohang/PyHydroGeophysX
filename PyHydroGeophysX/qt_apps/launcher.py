"""Launch the PyHydroGeophysX professional desktop workbench.

Usage::

    python -m PyHydroGeophysX.qt_apps.launcher --context <context.json> --module <key>

``--self-test`` constructs the window and exits immediately; combined with
``QT_QPA_PLATFORM=offscreen`` it provides a headless smoke test that catches
import/wiring errors without needing a display.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyHydroGeophysX Professional Workbench")
    parser.add_argument("--context", default=None, help="Path to the bridge context JSON.")
    parser.add_argument("--module", default="home", help="Initial module key to open.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Construct the window and exit (headless smoke test).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # This is a PySide6 app; tell qtpy-based libraries (e.g. pyvistaqt) to use
    # PySide6 too, so they do not bind to a PyQt5 install and clash with our
    # widgets. Must be set before anything imports qtpy.
    import os
    os.environ["QT_API"] = "pyside6"

    # Heavy GUI imports happen only here, inside the dedicated desktop process.
    from PySide6.QtWidgets import QApplication

    from PyHydroGeophysX.qt_apps import theme
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXWorkbench

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("PyHydroGeophysX Professional Workbench")
    # Apply the brand theme before building widgets (also sets pyqtgraph colors).
    theme.apply_theme(app)

    window = PyHydroGeophysXWorkbench(
        context_path=args.context,
        initial_module=(args.module or "home"),
    )
    window.setWindowIcon(theme.window_icon())
    window.resize(1500, 900)
    window.show()

    if args.self_test:
        # Build, pump the event loop briefly, then quit cleanly.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(300, app.quit)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
