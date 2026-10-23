"""Pieces of a guided studio page: where to start, what to do next, folded settings.

The Hydro → Geophysics and ERT → Water Content pages are laid out like the ERT
page and lead someone new through it: a start card where the view would
otherwise be empty, a one-line strip above the views saying what to do next,
the expert settings folded under "▸" lines, and number boxes that show a value
the way it is written.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.qt_utils import PlainDoubleSpinBox


def number_box(value: float, lo: float, hi: float, step: float, decimals: int,
               suffix: str = "", tip: str = "") -> QDoubleSpinBox:
    """A number box that shows 1 rather than 1.00."""
    box = PlainDoubleSpinBox()
    box.setRange(float(lo), float(hi))
    box.setDecimals(int(decimals))
    box.setSingleStep(float(step))
    box.setValue(float(value))
    if suffix:
        box.setSuffix(suffix)
    if tip:
        box.setToolTip(tip)
    return box


def hint(text: str = "") -> QLabel:
    """A small grey explanatory line that wraps."""
    label = QLabel(text)
    label.setWordWrap(True)
    theme.set_tone(label, "hint")
    return label


def fold(title: str = "More options") -> Tuple[QWidget, QFormLayout]:
    """Settings folded under a "▸ title" line, as the Home page's Session details.

    Returns the widget to place and the form to fill. A checkable group box is
    not used: folded, it still draws an empty card under its title, which reads
    as a box to type in.
    """
    holder = QWidget()
    outer = QVBoxLayout(holder)
    outer.setContentsMargins(0, 2, 0, 0)
    outer.setSpacing(0)
    toggle = QToolButton()
    toggle.setText(title)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setCheckable(True)
    toggle.setAutoRaise(True)
    toggle.setCursor(Qt.PointingHandCursor)
    body = QWidget()
    form = QFormLayout(body)
    form.setContentsMargins(14, 4, 0, 0)
    body.setVisible(False)

    def unfold(open_: bool) -> None:
        toggle.setArrowType(Qt.DownArrow if open_ else Qt.RightArrow)
        body.setVisible(open_)

    toggle.toggled.connect(unfold)
    outer.addWidget(toggle, alignment=Qt.AlignLeft)
    outer.addWidget(body)
    return holder, form


class NextStepStrip(QFrame):
    """The one line above a page's views that says what to do now."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("noteCard")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        icon = QLabel()
        icon.setPixmap(theme.icon("fa5s.arrow-circle-right",
                                  color=theme.PALETTE["primary"]).pixmap(18, 18))
        self._text = QLabel("")
        self._text.setWordWrap(True)
        self._text.setStyleSheet("font-weight: 600;")
        row.addWidget(icon, alignment=Qt.AlignTop)
        row.addWidget(self._text, stretch=1)

    def setText(self, text: str) -> None:  # noqa: N802 - QLabel's API
        self._text.setText(text)

    def text(self) -> str:
        return self._text.text()


def start_card(title: str, intro: str, steps: Sequence[str],
               buttons: Iterable[QPushButton], note: str = "") -> QWidget:
    """What a page shows before anything is loaded: what it does, and how to start.

    ``steps`` are numbered in order; ``buttons`` (the first is the primary
    one) sit under them, and ``note`` under the buttons.
    """
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.addStretch(1)
    # A group box is the studio's card; the Home page's card has a rule that
    # makes every child transparent, which would empty the blue button.
    card = QGroupBox()
    # A fixed width, so each wrapped line can be given the height it needs at
    # that width (below); sized from its hint, a wrapped QLabel comes out a line
    # short and squashes the buttons under it.
    width, margin = 600, 18
    card.setFixedWidth(width)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(margin, 4, margin, 12)
    layout.setSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("HomeCardTitle")
    heading.setWordWrap(True)
    layout.addWidget(heading)
    description = QLabel(intro)
    description.setObjectName("HomeDescription")
    description.setWordWrap(True)
    layout.addWidget(description)
    for number, text in enumerate(steps, start=1):
        step = QLabel(f"<b>{number}</b>&nbsp;&nbsp;&nbsp;{text}")
        step.setWordWrap(True)
        layout.addWidget(step)
    row = QHBoxLayout()
    row.setContentsMargins(0, 8, 0, 0)
    buttons = list(buttons)
    for button in buttons:
        row.addWidget(button)
    row.addStretch(1)
    layout.addLayout(row)
    if note:
        layout.addWidget(hint(note))
    for label in card.findChildren(QLabel):
        if label.wordWrap():
            label.ensurePolished()      # the stylesheet's font, not the default
            label.setMinimumHeight(label.heightForWidth(width - 2 * margin - 24))
    for button in buttons:
        button.setMinimumHeight(button.sizeHint().height())
    outer.addWidget(card, alignment=Qt.AlignHCenter)
    outer.addStretch(2)
    return page


__all__ = ["NextStepStrip", "fold", "hint", "number_box", "start_card"]
