"""Timestamped log panel (a read-only QTextEdit with colored levels)."""

from __future__ import annotations

import datetime
import html

from PySide6.QtWidgets import QTextEdit

#: Level colours that read on both the light and the dark appearance, so the
#: lines already logged stay legible when the appearance changes. Plain
#: messages take no colour at all and follow the panel's text colour.
_LEVEL_COLORS = {
    "info": None,
    "success": "#34c759",
    "warn": "#ff9500",
    "warning": "#ff9500",
    "error": "#ff3b30",
    "debug": "#8e8e93",
}
_TIME_COLOR = "#8e8e93"


class LogPanel(QTextEdit):
    """Append-only log with ``HH:MM:SS`` timestamps and per-level colors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("logPanel")
        self.setReadOnly(True)
        self.setMinimumHeight(110)
        self.document().setMaximumBlockCount(2000)  # cap memory

    def log(self, message: str, level: str = "info") -> None:
        """Append ``message`` with a timestamp and a color for ``level``."""
        level = (level or "info").lower()
        color = _LEVEL_COLORS.get(level)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        safe = html.escape(str(message))
        tint = f' style="color:{color}"' if color else ""
        self.append(
            f'<span style="color:{_TIME_COLOR}">[{ts}]</span> '
            f'<b{tint}>{level.upper():7}</b> '
            f'<span{tint}>{safe}</span>'
        )
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
