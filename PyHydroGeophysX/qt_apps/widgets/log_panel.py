"""Timestamped log panel (a read-only QTextEdit with colored levels)."""

from __future__ import annotations

from collections import deque
import datetime
import html

from PySide6.QtCore import QTimer
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
        # The display keeps a 2000-line tail; complete run logs are
        # persisted independently. Bound queued text as well as the document.
        self._pending = deque(maxlen=2000)
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(50)
        self._flush_timer.timeout.connect(self.flush_redraw)

    def log(self, message: str, level: str = "info") -> None:
        """Append ``message`` with a timestamp and a color for ``level``."""
        level = (level or "info").lower()
        color = _LEVEL_COLORS.get(level)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        safe = html.escape(str(message))
        tint = f' style="color:{color}"' if color else ""
        self._pending.append(
            f'<span style="color:{_TIME_COLOR}">[{ts}]</span> '
            f'<b{tint}>{level.upper():7}</b> '
            f'<span{tint}>{safe}</span>'
        )
        if not self._flush_timer.isActive():
            self._flush_timer.start()

    def flush_redraw(self) -> None:
        """Insert a burst in one document edit, preserving a reader's scroll."""
        self._flush_timer.stop()
        if not self._pending:
            return
        scrollbar = self.verticalScrollBar()
        position = scrollbar.value()
        at_bottom = position >= scrollbar.maximum() - 1
        text = ''.join('<p style="margin-top:0; margin-bottom:0">' + line + '</p>'
                       for line in self._pending)
        self._pending.clear()
        self.append(text)
        scrollbar.setValue(scrollbar.maximum() if at_bottom else position)

    def toPlainText(self) -> str:  # noqa: N802 - Qt override
        self.flush_redraw()
        return super().toPlainText()

    def toHtml(self) -> str:  # noqa: N802 - Qt override
        self.flush_redraw()
        return super().toHtml()

    def clear(self) -> None:
        self._flush_timer.stop()
        self._pending.clear()
        super().clear()
