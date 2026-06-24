"""Timestamped log panel (a read-only QTextEdit with colored levels)."""

from __future__ import annotations

import datetime
import html

from PySide6.QtWidgets import QTextEdit

_LEVEL_COLORS = {
    "info": "#d4d4d4",
    "success": "#4caf50",
    "warn": "#e0a800",
    "warning": "#e0a800",
    "error": "#f44336",
    "debug": "#888888",
}


class LogPanel(QTextEdit):
    """Append-only log with ``HH:MM:SS`` timestamps and per-level colors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(110)
        self.document().setMaximumBlockCount(2000)  # cap memory
        self.setStyleSheet(
            "QTextEdit { background:#1e1e1e; color:#d4d4d4; "
            "font-family: Consolas, 'Courier New', monospace; font-size:12px; }"
        )

    def log(self, message: str, level: str = "info") -> None:
        """Append ``message`` with a timestamp and a color for ``level``."""
        level = (level or "info").lower()
        color = _LEVEL_COLORS.get(level, _LEVEL_COLORS["info"])
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        safe = html.escape(str(message))
        self.append(
            f'<span style="color:#6a9955">[{ts}]</span> '
            f'<span style="color:{color}; font-weight:bold">{level.upper():7}</span> '
            f'<span style="color:{color}">{safe}</span>'
        )
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
