"""One-line readouts that change all the time: the cursor's position over a plot.

A ``QLabel`` re-lays out its window whenever its text changes. ``setText``
calls ``updateGeometry``, and every layout that activates passes that on to its
parent, up to the window, unless a widget on the way has a fixed size. Under the
mouse that happens on every move: 11 ms a move over the ERT page's Mesh tab and
20 ms over the Hydro -> Geophysics profile, with eight to ten layout passes
each, which is what made moving the mouse over a plot feel sluggish.

:class:`ReadoutLabel` paints its text itself and never changes its size hint,
so a new reading only repaints it. :func:`navigation_toolbar` gives a matplotlib
toolbar one in place of the ``QLabel`` it writes its coordinates into.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["ReadoutLabel", "navigation_toolbar"]


class ReadoutLabel(QWidget):
    """A readout with ``QLabel``'s ``text``/``setText``, repainted and never re-laid out.

    Its width comes from the layout: the preferred width is that of ``sample``
    (the text given, by default), whatever it shows later, and a long reading
    is elided rather than widening the row. Multi-line messages are shown on
    one line.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None, *,
                 alignment: Any = Qt.AlignLeft | Qt.AlignVCenter,
                 sample: Optional[str] = None) -> None:
        super().__init__(parent)
        self._text = ""
        self._alignment = alignment
        self._sample = str(sample if sample is not None else text)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setText(text)

    def text(self) -> str:
        return self._text

    def setText(self, text: Any) -> None:  # noqa: N802 - QLabel's API
        text = " ".join(str(text).split("\n"))
        if text != self._text:
            self._text = text
            self.update()

    def setAlignment(self, alignment: Any) -> None:  # noqa: N802 - QLabel's API
        self._alignment = alignment
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self._sample) + 8, metrics.height() + 4)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return QSize(0, self.fontMetrics().height() + 4)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        rect = self.contentsRect().adjusted(2, 0, -2, 0)
        text = self.fontMetrics().elidedText(self._text, Qt.ElideRight, max(rect.width(), 1))
        painter.setPen(self.palette().color(self.foregroundRole()))
        painter.drawText(rect, int(self._alignment), text)
        painter.end()


_TOOLBAR_CLASS: Any = None


def _toolbar_class() -> Any:
    """``NavigationToolbar2QT`` with its messages sent to a readout, made once."""
    global _TOOLBAR_CLASS
    if _TOOLBAR_CLASS is None:
        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

        class ReadoutToolbar(NavigationToolbar2QT):
            """A navigation toolbar without a label of its own for its messages."""

            def set_message(self, s: str) -> None:
                readout = getattr(self, "readout", None)
                if readout is not None:
                    readout.setText(s)

        _TOOLBAR_CLASS = ReadoutToolbar
    return _TOOLBAR_CLASS


def navigation_toolbar(canvas: Any, parent: Optional[QWidget] = None) -> Tuple[Any, ReadoutLabel]:
    """A matplotlib navigation toolbar, and the readout its cursor position goes to.

    The toolbar is made without its own coordinates label, which it would
    rewrite on every mouse move; its messages - the position, a zoom or pan
    hint - go to the returned :class:`ReadoutLabel`, for the caller to place
    beside it.
    """
    toolbar = _toolbar_class()(canvas, parent, coordinates=False)
    toolbar.readout = ReadoutLabel("", alignment=Qt.AlignRight | Qt.AlignVCenter,
                                   sample="x=0000.00 y=-000.00 [0000.0]")
    return toolbar, toolbar.readout
