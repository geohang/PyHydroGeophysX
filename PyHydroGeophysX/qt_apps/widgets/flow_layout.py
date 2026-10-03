"""A layout that lines its items up left to right and wraps them onto new lines.

A row of view controls in a ``QHBoxLayout`` cannot be narrower than all of
them side by side, and that sum becomes the page's minimum width, then the
window's: the shared section view's twelve controls alone held every ERT and
seismic page at 1600 px, so the studio could not fit a 1920 px screen beside
its project tree and assistant panel. In a :class:`FlowLayout` the same row is
one line when there is room and wraps when there is not, so its minimum width
is that of its widest item.

Hidden items take no space, and a widget shown or hidden later re-flows the
row (Qt invalidates a layout when one of its widgets changes visibility).
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLayout, QLayoutItem, QWidget


def group(*parts, spacing: int = 4) -> QWidget:
    """Controls that belong together as one item of a :class:`FlowLayout`.

    A label and the control it names - "Max time" and its spin box - must not
    be split across two lines when a row wraps; grouped, they wrap as one.
    Strings become labels. A control hidden inside the group takes no space.
    """
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(spacing)
    for part in parts:
        row.addWidget(QLabel(part) if isinstance(part, str) else part)
    return holder


class FlowLayout(QLayout):
    """Items left to right, wrapping onto a new line when the width runs out.

    Parameters
    ----------
    parent : QWidget, optional
        The widget to lay out, as for any layout.
    spacing : int
        Gap between items on a line, in pixels.
    line_spacing : int, optional
        Gap between lines; ``spacing`` when not given.

    Notes
    -----
    Items on one line are centred vertically, so a check box sits level with
    the spin box beside it. The preferred size is everything on one line; the
    minimum width is the widest item's.

    Examples
    --------
    >>> from PySide6.QtWidgets import QApplication, QPushButton, QWidget
    >>> app = QApplication.instance() or QApplication([])
    >>> host = QWidget()
    >>> flow = FlowLayout(host, spacing=4)
    >>> for text in ('One', 'Two', 'Three'):
    ...     flow.addWidget(QPushButton(text))
    >>> widest = max(flow.itemAt(i).minimumSize().width() for i in range(flow.count()))
    >>> flow.minimumSize().width() == widest
    True
    >>> flow.heightForWidth(10_000) < flow.heightForWidth(widest)    # one line vs. three
    True
    """

    def __init__(self, parent: Optional[QWidget] = None, spacing: int = 6,
                 line_spacing: Optional[int] = None) -> None:
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._gap = int(spacing)
        self._line_gap = int(spacing if line_spacing is None else line_spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # -- the QLayout interface -------------------------------------------------
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 - Qt naming
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt naming
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt naming
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt naming
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming
        return self._arrange(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:  # noqa: N802 - Qt naming
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        visible = self._visible()
        margins = self.contentsMargins()
        width = sum(self._width(item) for item in visible)
        width += self._gap * max(0, len(visible) - 1)
        height = max((self._height(item) for item in visible), default=0)
        return QSize(width + margins.left() + margins.right(),
                     height + margins.top() + margins.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt naming
        size = QSize()
        for item in self._visible():
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    # -- internals ---------------------------------------------------------------
    @staticmethod
    def _width(item: QLayoutItem) -> int:
        return max(item.sizeHint().width(), item.minimumSize().width())

    @staticmethod
    def _height(item: QLayoutItem) -> int:
        return max(item.sizeHint().height(), item.minimumSize().height())

    def _visible(self) -> List[QLayoutItem]:
        return [item for item in self._items if not item.isEmpty()]

    def _arrange(self, rect: QRect, apply: bool) -> int:
        """Place the items within ``rect`` (or only measure); returns the height used."""
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        lines: List[List[QLayoutItem]] = [[]]
        x = 0
        for item in self._visible():
            width = self._width(item)
            if lines[-1] and x + width > area.width():
                lines.append([])
                x = 0
            lines[-1].append(item)
            x += width + self._gap
        y = area.y()
        for line in (line for line in lines if line):
            height = max(self._height(item) for item in line)
            if apply:
                x = area.x()
                for item in line:
                    item_height = self._height(item)
                    # Never below the item's own minimum; never past the line
                    # unless that minimum is wider than the line itself.
                    width = max(item.minimumSize().width(), min(self._width(item), area.width()))
                    item.setGeometry(QRect(QPoint(x, y + (height - item_height) // 2),
                                           QSize(width, item_height)))
                    x += width + self._gap
            y += height + self._line_gap
        used = y - self._line_gap - area.y() if any(lines) else 0
        return max(0, used) + margins.top() + margins.bottom()
