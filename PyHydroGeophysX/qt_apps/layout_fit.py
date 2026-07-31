"""Stop a module page from demanding more width than the screen has.

Qt propagates a widget's content width upward as a hard minimum. A paragraph of
guidance text in a QLabel that forgot ``setWordWrap`` therefore becomes a floor
on the whole window: the workbench asked for 2699 px on a 1920 px monitor, which
Windows cannot grant, so every layout pass logged ``QWindowsWindow::setGeometry:
Unable to set geometry`` and the panels were squeezed anyway.

:func:`relax_minimum_width` walks a finished page once and makes its widgets
shrinkable, choosing per widget rather than uniformly:

- prose wraps, because eliding a sentence of instructions destroys it;
- short status readouts elide, because they update constantly and a wrapping
  readout would make the layout jump on every mouse move;
- combo boxes and long-captioned buttons elide, keeping the full text in a
  tooltip.

Nothing here changes what a widget displays at a comfortable width. It only
removes the claim that the widget can never be narrower.
"""

from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QSizePolicy,
    QWidget,
)

#: A plain label at least this long is treated as prose and wrapped.
PROSE_CHARS = 45

#: Labels/buttons narrower than this are left alone; they cost nothing.
RELAX_FLOOR_PX = 130

#: Width a shrinkable widget may fall to.
ELIDED_MIN_PX = 48

#: Floor for a button. Unlike a label, a button clips its caption rather than
#: eliding it, so a short one must keep enough room to stay readable while a
#: long one is still free to give up most of its width.
BUTTON_MIN_PX = 120

#: Characters a combo box guarantees to show before eliding.
COMBO_CHARS = 12

#: Width a spin box may shrink to; enough for a value plus its arrows.
SPIN_MIN_PX = 76

#: Chrome around a caption: frame, icon slot and padding. Approximate on
#: purpose, since the exact value depends on the active style.
BUTTON_PADDING_PX = 40


class _ElideFilter(QObject):
    """Paint a label's text elided instead of forcing the layout to fit it."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Paint or not isinstance(obj, QLabel):
            return False
        painter = QPainter(obj)
        metrics = QFontMetrics(obj.font())
        margin = obj.margin() * 2 + 2
        text = metrics.elidedText(obj.text(), Qt.ElideRight,
                                  max(obj.width() - margin, 1))
        painter.setPen(obj.palette().color(obj.foregroundRole()))
        painter.drawText(obj.contentsRect(), int(obj.alignment()), text)
        painter.end()
        return True


#: One filter instance for the whole application; QObject parents keep it alive.
_ELIDE_FILTER = _ElideFilter()


def _is_rich(label: QLabel) -> bool:
    """Whether the label renders markup rather than plain characters.

    Rich text must never reach the elide filter: that filter measures and draws
    ``label.text()`` verbatim, so a heading would appear on screen as the literal
    string ``<h3>Step 1 …</h3>``.
    """
    if label.textFormat() == Qt.RichText:
        return True
    if label.textFormat() == Qt.PlainText:
        return False
    text = label.text()
    return "<" in text and ">" in text


def _elide(widget: QWidget) -> None:
    widget.setMinimumWidth(0)
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    widget.setSizePolicy(policy)


def elide_label(label: QLabel) -> QLabel:
    """Make one label paint elided and stop claiming its full text width.

    For a label outside a module page (the header, a fixed-height bar) where the
    generic pass does not reach. A rich-text label is only relaxed, never
    elided, since the filter would draw its markup verbatim.
    """
    if not label.toolTip():
        label.setToolTip(label.text())
    if not _is_rich(label):
        label.installEventFilter(_ELIDE_FILTER)
    _elide(label)
    return label


def _text_width(widget: QWidget, text: str) -> int:
    """Width of ``text`` in the widget's font.

    Deliberately not ``minimumSizeHint``: this pass runs when a page is built,
    which for the first page happens before the window is shown. An unpolished
    widget reports a placeholder hint, so gating on it silently relaxed nothing.
    Font metrics are valid from construction.
    """
    return QFontMetrics(widget.font()).horizontalAdvance(text)


def relax_minimum_width(page: QWidget) -> Dict[str, int]:
    """Make ``page`` shrinkable. Returns a count per action, for tests and logs."""
    counts = {"wrapped": 0, "elided_label": 0, "elided_button": 0,
              "combo": 0, "spin": 0}
    page.ensurePolished()

    for label in page.findChildren(QLabel):
        if label.wordWrap():
            continue
        text = label.text()
        if not text:
            continue
        if _is_rich(label) or len(text) >= PROSE_CHARS:
            # Wrapping drops the minimum to the longest single word, and it is the
            # only safe option for markup. A sentence of guidance must stay
            # readable, so it wraps rather than losing its tail to an ellipsis.
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            counts["wrapped"] += 1
            continue
        # What is left is a short, frequently-updated readout. Wrapping one of
        # those would make the layout jump on every mouse move, so it elides.
        if _text_width(label, text) < RELAX_FLOOR_PX:
            continue
        if not label.toolTip():
            label.setToolTip(text)
        label.installEventFilter(_ELIDE_FILTER)
        _elide(label)
        counts["elided_label"] += 1

    for button in page.findChildren(QAbstractButton):
        text = button.text()
        if not text:
            continue
        natural = _text_width(button, text) + BUTTON_PADDING_PX
        if natural < RELAX_FLOOR_PX:
            continue
        if not button.toolTip():
            button.setToolTip(text)
        # An explicit minimum replaces the content-derived one in Qt's layout
        # maths, so the caption still sets the preferred size and the button
        # looks unchanged until the window is genuinely too narrow for it.
        button.setMinimumWidth(min(natural, BUTTON_MIN_PX))
        counts["elided_button"] += 1

    for combo in page.findChildren(QComboBox):
        widest = max((len(combo.itemText(i)) for i in range(combo.count())), default=0)
        # Capping at COMBO_CHARS would widen a combo whose entries are all
        # shorter than that, so only the ones that actually overflow are capped.
        if widest <= COMBO_CHARS:
            continue
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(COMBO_CHARS)
        if not combo.toolTip():
            combo.setToolTip(combo.currentText())
        counts["combo"] += 1

    for spin in page.findChildren(QAbstractSpinBox):
        # A spin box sizes itself to its widest possible value, so a range of
        # 0..1e9 with four decimals reserves room for "1000000000.0000" even
        # though no reading is ever that long.
        top = getattr(spin, "maximum", None)
        if not callable(top):
            continue
        sample = f"{top():.4f}{spin.suffix()}" if hasattr(spin, "decimals") else f"{top()}"
        if _text_width(spin, sample) + BUTTON_PADDING_PX <= SPIN_MIN_PX:
            continue
        spin.setMinimumWidth(SPIN_MIN_PX)
        counts["spin"] += 1

    return counts


def relaxed_report(page: QWidget) -> List[str]:
    """Widgets still forcing a wide minimum, for diagnosing a stubborn page."""
    rows = []
    for widget in page.findChildren(QWidget):
        width = max(widget.minimumSizeHint().width(), widget.minimumWidth())
        if width >= 400:
            rows.append(f"{width:>5}  {type(widget).__name__} "
                        f"{widget.objectName() or ''}".rstrip())
    return sorted(rows, reverse=True)
