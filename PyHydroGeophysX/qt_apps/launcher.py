"""Launch the PyHydroGeophysX professional desktop studio.

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


def _install_excepthook(show_dialog: bool) -> None:
    """Route uncaught exceptions to stderr plus an error dialog.

    Without this, an exception escaping a Qt slot or worker callback kills the
    process with a console traceback that windowed users never see.
    """
    import traceback

    def _hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        sys.stderr.write(text)
        if show_dialog:
            try:
                from PySide6.QtWidgets import QApplication, QMessageBox

                if QApplication.instance() is not None:
                    box = QMessageBox()
                    box.setIcon(QMessageBox.Icon.Critical)
                    box.setWindowTitle("PyHydroGeophysX Studio: unexpected error")
                    box.setText(
                        "An unexpected error occurred. The application will try to "
                        "keep running; details are below."
                    )
                    box.setDetailedText(text)
                    box.exec()
            except Exception:
                pass

    sys.excepthook = _hook


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyHydroGeophysX Professional Studio")
    parser.add_argument("--context", default=None, help="Path to the bridge context JSON.")
    parser.add_argument("--module", default="home", help="Initial module key to open.")
    parser.add_argument(
        "--em-data",
        default=None,
        help="Open the EM module and preload a sounding file or TEMcompany project folder.",
    )
    parser.add_argument(
        "--tem-moment",
        choices=("LM+HM", "HM", "LM"),
        default="LM+HM",
        help="TEMcompany moment(s) to preload with --em-data (default: LM+HM).",
    )
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
    from PyHydroGeophysX.qt_apps.main_window import PyHydroGeophysXStudio

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("PyHydroGeophysX Professional Studio")
    _install_excepthook(show_dialog=not args.self_test)
    # Apply the brand theme before building widgets (also sets pyqtgraph colors).
    theme.apply_theme(app)

    window = PyHydroGeophysXStudio(
        context_path=args.context,
        initial_module=("em" if args.em_data else (args.module or "home")),
    )
    window.setWindowIcon(theme.window_icon())
    if not getattr(window, "_geometry_restored", False):
        window.resize(1500, 900)
    window.show()

    if args.em_data:
        from PySide6.QtCore import QTimer

        def load_em_data() -> None:
            page = window.current_module()
            if page is None or not hasattr(page, "agent_apply"):
                window.log("Could not open the EM data loader.", "error")
                return
            result = page.agent_apply(
                "load_data",
                {"path": args.em_data, "moment": args.tem_moment},
            )
            if result.get("status") != "ok":
                window.log(f"Could not preload EM data: {result.get('error', result)}", "error")

        QTimer.singleShot(0, load_em_data)

    if args.self_test:
        # Build, pump the event loop briefly, then quit cleanly.
        from PySide6.QtCore import QTimer

        QTimer.singleShot(300, app.quit)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
