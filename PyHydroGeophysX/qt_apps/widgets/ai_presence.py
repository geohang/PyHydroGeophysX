"""How the studio looks while an AI assistant, rather than the user, is doing the work.

An automatic run used to look like any long computation: a status line, a
progress bar whose fraction is a guess, and a log. Nothing on screen said that
a model was choosing each step, so a run read as a script someone had started.
These widgets give the agent a visual presence of its own, in one consistent
language wherever it appears.

The rest of the studio is neutral (:mod:`..theme`), so colour here means the
agent. An assistant at work shows as a soft glow of blue, purple, pink and
orange flowing round the edge of the area the agent is working in, and an orb whose motion says what it is doing - the gradient
breathing while it decides, a ring turning while a step runs, a steady orange
while it waits for you, a green check or a red cross when a step ends.

- :class:`AiOrb` - the agent's orb.
- :class:`ShimmerText` - a status line with a light passing over it, the
  convention for "a model is working on this".
- :class:`TypewriterLabel` - text that writes itself out, used for what the
  agent said: why it chose a step, and what the step found.
- :class:`AgentGlowFrame` - the glow round the studio's central area for as
  long as the agent is in control of it.
- :class:`AgentTimeline` - the run as a sequence of step cards: the reason the
  controller gave, the module it worked in, how long it took and what it found.
- :class:`AgentHeader` - the Workflow page's banner: orb, headline, clock.
- :class:`RunRoute` - the run as a route: steps taken, the one running, and
  the ones still ahead as the run itself projects them.
- :class:`LiveCanvas` - the newest figure the run has written, large, revealed
  as it appears, with the earlier ones in a strip.
- :class:`FinishCard` - the end of a run: how it went, what it made, what to
  check, and where to go next.
- :class:`NoteCard` and :class:`SteerBar` - what the user tells a run while it
  works (pause, resume, a note), and what the run made of it.

Everything shown is the run's own record - the controller's reason, the step's
summary, figures the run has written - never an animation of work that is not
happening. There is deliberately no simulated cursor clicking the panels: the
workflow runs headless in its own process, and showing it pressing buttons it
never pressed would misdescribe what the studio is doing.

Colours are read from :data:`..theme.PALETTE` when painting and labels are
styled through the studio's stylesheet, so all of this follows the Light and
Dark appearances. Animation runs only while a widget is visible and in an
animated state, and the glow repaints only the frame's margin, so a run costs
the UI almost nothing.
"""

from __future__ import annotations

import html
import math
import os
import time
from typing import Dict, List, Optional, Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QFont,
                           QFontMetrics, QLinearGradient, QPainter, QPainterPath,
                           QPen, QRadialGradient, QRegion)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QScrollArea,
                               QSizePolicy, QStyle, QStyleOption, QVBoxLayout,
                               QWidget)

from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.widgets.flow_layout import FlowLayout

#: What the agent is doing. One vocabulary for every widget in this module.
IDLE, THINKING, WORKING, WAITING, DONE, FAILED = (
    "idle", "thinking", "working", "waiting", "done", "failed")
STATES = (IDLE, THINKING, WORKING, WAITING, DONE, FAILED)

_ANIMATED = frozenset({THINKING, WORKING, WAITING})
#: States in which the agent is actively doing something, which is what the
#: passing light on text means. Waiting is shown by a steady pulse instead.
_BUSY = frozenset({THINKING, WORKING})
#: The palette entry each state's text is drawn in.
STATE_TONE = {THINKING: "text", WORKING: "text", WAITING: "amber",
              DONE: "green", FAILED: "red", IDLE: "muted"}
#: The full-strength colour each finished or paused state is marked with.
_STATE_VIVID = {WAITING: "vivid_orange", DONE: "vivid_green",
                FAILED: "vivid_red", IDLE: "vivid_gray"}
#: Neutral grey that reads on both appearances, for marks inside rich text.
CARET_GRAY = "#8e8e93"
_FRAME_MS = 33


def _color(name: str, alpha: float = 1.0) -> QColor:
    color = QColor(name)
    color.setAlphaF(max(0.0, min(1.0, alpha)))
    return color


def _paint_styled_background(widget: QWidget, painter: QPainter) -> None:
    """Paint the widget's style-sheet background, which a paintEvent replaces."""
    option = QStyleOption()
    option.initFrom(widget)
    widget.style().drawPrimitive(QStyle.PE_Widget, option, painter, widget)


def _ai_gradient(centre: QPointF, angle: float, alpha: float = 1.0) -> QConicalGradient:
    """The agent's colours - blue, purple, pink, orange - swept round ``centre``."""
    blue, purple, pink, orange = theme.ai_colors()
    gradient = QConicalGradient(centre, angle)
    for stop, name in ((0.0, blue), (0.27, purple), (0.52, pink),
                       (0.76, orange), (1.0, blue)):
        gradient.setColorAt(stop, _color(name, alpha))
    return gradient


class _Animated(QWidget):
    """A widget that redraws itself on a timer while it has something to show.

    The timer runs only while the widget is visible and :meth:`_animating`
    says so; a hidden tab or a finished run costs nothing.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._frame)

    def _animating(self) -> bool:
        return False

    def _frame(self) -> None:
        self.update()

    def _sync_timer(self) -> None:
        if self._animating() and self.isVisible():
            if not self._timer.isActive():
                self._timer.start()
        elif self._timer.isActive():
            self._timer.stop()

    def _elapsed(self) -> float:
        return time.monotonic() - self._t0

    def showEvent(self, event) -> None:  # noqa: N802 - Qt event override
        super().showEvent(event)
        self._sync_timer()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt event override
        super().hideEvent(event)
        self._sync_timer()


class AiOrb(_Animated):
    """The agent's orb: its motion says what the agent is doing.

    Parameters
    ----------
    diameter : int
        Size in pixels; the orb is square.
    """

    def __init__(self, diameter: int = 28, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = IDLE
        self.setFixedSize(diameter, diameter)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        state = state if state in STATES else IDLE
        if state == self._state:
            return
        self._state = state
        self._sync_timer()
        self.update()

    def _animating(self) -> bool:
        return self._state in _ANIMATED

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        t = self._elapsed()
        side = min(self.width(), self.height())
        c = QPointF(self.width() / 2.0, self.height() / 2.0)
        r = side / 2.0 - 0.5
        state = self._state
        painter.setPen(Qt.NoPen)

        if state in _BUSY:
            thinking = state == THINKING
            breathe = 0.5 + 0.5 * math.sin(t * math.tau / (1.6 if thinking else 2.4))
            blue, purple, pink, _orange = theme.ai_colors()
            # A soft halo, the colours bleeding past the edge of the orb.
            halo = QRadialGradient(c, r)
            halo.setColorAt(0.45, _color(purple, 0.22 + 0.20 * breathe))
            halo.setColorAt(0.75, _color(blue, 0.10 + 0.08 * breathe))
            halo.setColorAt(1.0, _color(pink, 0.0))
            painter.setBrush(halo)
            painter.drawEllipse(c, r, r)
            # The orb: the gradient turning, breathing while it decides.
            angle = (t * (0.55 if thinking else 0.3) * 360.0) % 360.0
            core_r = r * ((0.58 + 0.06 * breathe) if thinking else 0.46)
            painter.setBrush(QBrush(_ai_gradient(c, -angle)))
            painter.drawEllipse(c, core_r, core_r)
            sheen = QRadialGradient(QPointF(c.x() - core_r * 0.30, c.y() - core_r * 0.38),
                                    core_r * 1.05)
            sheen.setColorAt(0.0, QColor(255, 255, 255, 170))
            sheen.setColorAt(0.55, QColor(255, 255, 255, 40))
            sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
            painter.setBrush(sheen)
            painter.drawEllipse(c, core_r, core_r)
            if not thinking:
                # A ring turning round it: a step is running.
                ring_r = r * 0.80
                start = (t * 1.1 * 360.0) % 360.0
                pen = QPen(QBrush(_ai_gradient(c, start)), max(1.6, r * 0.13))
                pen.setCapStyle(Qt.RoundCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawArc(QRectF(c.x() - ring_r, c.y() - ring_r, 2 * ring_r, 2 * ring_r),
                                int(start * 16), int(260 * 16))
            return

        vivid = theme.PALETTE[_STATE_VIVID[state]]
        if state == WAITING:
            pulse = 0.5 + 0.5 * math.sin(t * math.tau / 2.0)
            halo = QRadialGradient(c, r)
            halo.setColorAt(0.55, _color(vivid, 0.18 + 0.22 * pulse))
            halo.setColorAt(1.0, _color(vivid, 0.0))
            painter.setBrush(halo)
            painter.drawEllipse(c, r, r)
        body_r = r * (0.48 if state == IDLE else (0.62 if state == WAITING else 0.72))
        painter.setBrush(_color(vivid, 0.55 if state == IDLE else 1.0))
        painter.drawEllipse(c, body_r, body_r)
        if state == IDLE:
            return
        glyph = QPen(QColor(255, 255, 255), max(1.4, r * 0.13))
        glyph.setCapStyle(Qt.RoundCap)
        glyph.setJoinStyle(Qt.RoundJoin)
        painter.setPen(glyph)
        painter.setBrush(Qt.NoBrush)
        g = body_r * 0.48
        if state == DONE:
            path = QPainterPath(QPointF(c.x() - g, c.y() + g * 0.05))
            path.lineTo(QPointF(c.x() - g * 0.25, c.y() + g * 0.72))
            path.lineTo(QPointF(c.x() + g, c.y() - g * 0.62))
            painter.drawPath(path)
        elif state == FAILED:
            painter.drawLine(QPointF(c.x() - g * 0.7, c.y() - g * 0.7),
                             QPointF(c.x() + g * 0.7, c.y() + g * 0.7))
            painter.drawLine(QPointF(c.x() + g * 0.7, c.y() - g * 0.7),
                             QPointF(c.x() - g * 0.7, c.y() + g * 0.7))
        else:  # waiting: pause bars
            painter.drawLine(QPointF(c.x() - g * 0.38, c.y() - g * 0.62),
                             QPointF(c.x() - g * 0.38, c.y() + g * 0.62))
            painter.drawLine(QPointF(c.x() + g * 0.38, c.y() - g * 0.62),
                             QPointF(c.x() + g * 0.38, c.y() + g * 0.62))


class ShimmerText(_Animated):
    """One line of text with a light passing over it while the agent works.

    Elided rather than wrapped: a status line that changes height every time
    its wording changes makes the whole page jump.
    """

    def __init__(self, text: str = "", point_size: Optional[float] = None,
                 bold: bool = True, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._text = str(text)
        self._shimmer = False
        self._tone = "text"
        self.setAttribute(Qt.WA_TranslucentBackground)
        font = QFont(self.font())
        font.setFamilies([theme.UI_FONT_FAMILY, "Helvetica Neue", "Arial"])
        if point_size:
            font.setPointSizeF(point_size)
        font.setWeight(QFont.DemiBold if bold else QFont.Normal)
        # Kept apart from the widget's own font: the studio's style sheet sets
        # a font on every widget, which replaces anything set with setFont.
        self._font = font
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        text = str(text or "")
        if text != self._text:
            self._text = text
            self.setToolTip(text)
            self.updateGeometry()
            self.update()

    def set_shimmer(self, on: bool, tone: Optional[str] = None) -> None:
        """Start or stop the passing light; ``tone`` is the palette entry the
        text is drawn in (``'text'``, ``'muted'``, ``'amber'``...)."""
        self._shimmer = bool(on)
        if tone:
            self._tone = tone
        self._sync_timer()
        self.update()

    def _animating(self) -> bool:
        return self._shimmer

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        metrics = QFontMetrics(self._font)
        return QSize(metrics.horizontalAdvance(self._text) + 6, metrics.height() + 4)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt naming
        return QSize(40, QFontMetrics(self._font).height() + 4)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self._font)
        metrics = QFontMetrics(self._font)
        text = metrics.elidedText(self._text, Qt.ElideRight, max(0, self.width() - 2))
        base = QColor(theme.PALETTE.get(self._tone, theme.PALETTE["text"]))
        if self._shimmer:
            light = QColor(theme.PALETTE["text" if self._tone == "muted" else "tertiary"])
            width = max(1, metrics.horizontalAdvance(text))
            band = max(60.0, width * 0.28)
            sweep = (self._elapsed() % 2.2) / 2.2
            x = -band + sweep * (width + 2 * band)
            gradient = QLinearGradient(x - band, 0, x + band, 0)
            gradient.setColorAt(0.0, base)
            gradient.setColorAt(0.5, light)
            gradient.setColorAt(1.0, base)
            painter.setPen(QPen(QBrush(gradient), 1))
        else:
            painter.setPen(base)
        painter.drawText(self.rect().adjusted(1, 0, -1, 0),
                         Qt.AlignLeft | Qt.AlignVCenter, text)


class TypewriterLabel(QLabel):
    """A word-wrapped label whose text writes itself out.

    The full text is laid out from the first frame, with the part not yet
    written drawn transparent, so the label is its final height at once and the
    page under it does not reflow on every character. Setting the same text
    again is a no-op, which matters because the Workflow page refreshes its
    displays once a second. Its colour comes from the stylesheet, by object
    name, so it follows the appearance.
    """

    #: Longest a line takes to write out, however long it is: the display must
    #: never fall behind the run it is describing.
    MAX_MS = 1100

    def __init__(self, text: str = "", name: str = "", italic: bool = False,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        if name:
            self.setObjectName(name)
        self.setWordWrap(True)
        self.setTextFormat(Qt.RichText)
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._full = ""
        self._shown = 0
        self._italic = italic
        self._step = 1
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._advance)
        if text:
            self.type_text(text, animate=False)

    def full_text(self) -> str:
        return self._full

    def type_text(self, text: str, animate: bool = True) -> None:
        text = str(text or "")
        if text == self._full:
            return
        self._full = text
        if not animate or not text:
            self._shown = len(text)
            self._timer.stop()
        else:
            self._shown = 0
            ticks = max(1, min(self.MAX_MS, max(250, 12 * len(text))) // 30)
            self._step = max(1, math.ceil(len(text) / ticks))
            self._timer.start()
        self._render()

    def _advance(self) -> None:
        self._shown = min(len(self._full), self._shown + self._step)
        if self._shown >= len(self._full):
            self._timer.stop()
        self._render()

    def _render(self) -> None:
        typed = html.escape(self._full[:self._shown])
        rest = self._full[self._shown:]
        body = typed
        if rest:
            caret = f"<span style='color:{CARET_GRAY};'>&#9613;</span>"
            body += caret + (f"<span style='color:transparent;'>"
                             f"{html.escape(rest[1:])}</span>" if len(rest) > 1 else "")
        body = body.replace(chr(10), "<br>")
        super().setText(f"<i>{body}</i>" if self._italic else body)


class AgentGlowFrame(QWidget):
    """A container whose margin glows while the agent is in control.

    The agent's colours flow slowly round the central area; waiting is a
    steady orange pulse, and a run that ends flashes green (or red) and fades.

    The glow is drawn in the frame's own margin, outside every child widget,
    so it never covers a control, never takes a click, and does not depend on
    how a child draws - a VTK or OpenGL view underneath would hide an overlay.
    Only the margin is repainted on each frame.

    Parameters
    ----------
    margin : int
        Width of the glow, which is also the layout margin the frame's content
        should keep clear.
    """

    #: How long a finished run's glow lingers, then fades.
    HOLD_S, FADE_S = 0.8, 1.6

    def __init__(self, margin: int = 8, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._margin = int(margin)
        self._state = IDLE
        self._ended_at: Optional[float] = None
        self._t0 = time.monotonic()
        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._frame)

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        """Glow for ``state``; DONE and FAILED linger and then fade."""
        state = state if state in STATES else IDLE
        if state == self._state:
            return
        self._state = state
        self._ended_at = time.monotonic() if state in (DONE, FAILED) else None
        if state == IDLE:
            self._timer.stop()
        elif not self._timer.isActive():
            self._timer.start()
        self.update()

    def _strength(self) -> float:
        """Opacity of the glow now: pulsing while waiting, fading once ended."""
        if self._state == IDLE:
            return 0.0
        if self._state == WAITING:
            return 0.55 + 0.45 * math.sin((time.monotonic() - self._t0) * math.tau / 2.0)
        if self._ended_at is not None:
            gone = time.monotonic() - self._ended_at - self.HOLD_S
            return 1.0 if gone <= 0 else max(0.0, 1.0 - gone / self.FADE_S)
        return 1.0

    def _frame(self) -> None:
        if self._ended_at is not None and self._strength() <= 0.0:
            self._state = IDLE
            self._ended_at = None
            self._timer.stop()
            self.update()
            return
        if self.isVisible():
            outer = QRegion(self.rect())
            inner = QRegion(self.rect().adjusted(self._margin, self._margin,
                                                 -self._margin, -self._margin))
            self.update(outer.subtracted(inner))

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        _paint_styled_background(self, painter)
        strength = self._strength()
        if strength <= 0.0:
            return
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect())
        centre = rect.center()
        m = float(self._margin)
        angle = ((time.monotonic() - self._t0) * 0.12 * 360.0) % 360.0
        painter.setBrush(Qt.NoBrush)
        # Soft outer light, a brighter middle, and a crisp line at the core.
        for width, inset, alpha in ((m * 1.1, m * 0.5, 0.20), (m * 0.6, m * 0.45, 0.42),
                                    (2.0, m * 0.42, 0.95)):
            if self._state in _BUSY:
                brush = QBrush(_ai_gradient(centre, angle, alpha * strength))
            else:
                brush = QBrush(_color(theme.PALETTE[_STATE_VIVID[self._state]],
                                      alpha * strength))
            painter.setPen(QPen(brush, width))
            painter.drawRoundedRect(rect.adjusted(inset, inset, -inset, -inset), 12.0, 12.0)


# -- the run as a timeline ---------------------------------------------------

def module_title(key: str) -> str:
    """The studio's own name for a module key, or the key itself."""
    try:
        from PyHydroGeophysX.qt_apps.modules import MODULE_SPECS
    except Exception:  # noqa: BLE001 - a name is cosmetic
        return str(key)
    spec = MODULE_SPECS.get(str(key))
    return spec[2] if spec else str(key)


def usage_text(tokens: int, cost_usd: float) -> str:
    """Tokens and estimated cost, the way a status line says them.

    >>> usage_text(850, 0.0004), usage_text(12400, 0.031), usage_text(2_300_000, 4.2)
    ('850 tokens · ≈$0.0004', '12.4k tokens · ≈$0.03', '2.3M tokens · ≈$4.20')
    """
    tokens = int(tokens or 0)
    if tokens >= 1_000_000:
        count = f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1000:
        count = f"{tokens / 1000:.1f}k"
    else:
        count = str(tokens)
    cost = float(cost_usd or 0.0)
    price = f"${cost:.4f}" if 0 < cost < 0.01 else f"${cost:.2f}"
    return f"{count} tokens · ≈{price}"


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds} s"


def _label(text: str = "", name: str = "") -> QLabel:
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    return label


class _Card(QFrame):
    """A rounded card; its outline takes the state's colour while it waits."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = IDLE
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

    def set_accent(self, state: str) -> None:
        self._state = state
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        p = theme.PALETTE
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        outline = (_color(p["vivid_orange"], 0.65) if self._state == WAITING
                   else QColor(p["border"]))
        painter.setPen(QPen(outline, 1))
        painter.setBrush(QColor(p["card"]))
        painter.drawRoundedRect(rect, 12, 12)


class StepCard(_Card):
    """One step of the run: why it was chosen, where, how long, what it found."""

    def __init__(self, label: str, module: str = "", reason: str = "",
                 parent: Optional[QWidget] = None, typed: bool = True) -> None:
        super().__init__(parent)
        self.label = str(label)
        self.module = str(module or "")
        self.status = "running"
        self._started = time.monotonic()
        self._ended: Optional[float] = None
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 12, 14, 12)
        row.setSpacing(12)
        self.orb = AiOrb(26)
        self.orb.set_state(WORKING)
        row.addWidget(self.orb, 0, Qt.AlignTop)
        column = QVBoxLayout()
        column.setSpacing(3)
        row.addLayout(column, 1)
        top = QHBoxLayout()
        top.setSpacing(8)
        self._title = _label(html.escape(self.label), "agentTitle")
        self._title.setWordWrap(True)
        top.addWidget(self._title, 1)
        if self.module:
            top.addWidget(_label(html.escape(module_title(self.module)), "agentChip"),
                          0, Qt.AlignTop)
        self._clock = _label("", "agentClock")
        top.addWidget(self._clock, 0, Qt.AlignTop)
        column.addLayout(top)
        self.reason = TypewriterLabel(name="agentReason")
        self.reason.setVisible(bool(reason))
        if reason:
            # Not typed out again when the user has just watched the model
            # write it (the thinking card streams it as it is written).
            self.reason.type_text(f"Why: {reason}", animate=typed)
        column.addWidget(self.reason)
        where = f"in {module_title(self.module)}" if self.module else "in its own process"
        self.activity = ShimmerText(f"Running {where}…", bold=False)
        self.activity.set_shimmer(True, tone="muted")
        column.addWidget(self.activity)
        self.summary = TypewriterLabel(name="agentSummary")
        self.summary.setVisible(False)
        column.addWidget(self.summary)
        self._figures_host = QWidget()
        self._figures_host.setObjectName("stepFigures")
        self._figures = QHBoxLayout(self._figures_host)
        self._figures.setContentsMargins(0, 4, 0, 0)
        self._figures.setSpacing(6)
        self._figures_host.setVisible(False)
        column.addWidget(self._figures_host)
        self.tick()

    def elapsed(self) -> float:
        return (self._ended or time.monotonic()) - self._started

    def tick(self) -> None:
        self._clock.setText(_clock(self.elapsed()))

    def finish(self, status: str, summary: str = "",
               figures: Sequence[str] = (), seconds: Optional[float] = None) -> None:
        """Mark the step ended, with what it found and the figures it wrote.

        ``seconds`` is how long the step took by the run's own clock, when
        known; a replay shows the real duration, not the time it was on screen.
        """
        self.status = str(status or "ok")
        self._ended = time.monotonic()
        if seconds is not None:
            self._started = self._ended - max(0.0, float(seconds))
        state = {"ok": DONE, "skipped": IDLE, "stopped": IDLE}.get(self.status, FAILED)
        self.orb.set_state(state)
        self.activity.set_shimmer(False)
        self.activity.setVisible(False)
        prefix = {"ok": "Found: ", "skipped": "Skipped: ",
                  "stopped": "Stopped: "}.get(self.status, "Failed: ")
        if summary:
            if state == FAILED:
                theme.set_tone(self.summary, None)
                self.summary.setProperty("failed", True)
                self.summary.style().unpolish(self.summary)
                self.summary.style().polish(self.summary)
            self.summary.type_text(prefix + str(summary))
            self.summary.setVisible(True)
        self._show_figures(figures)
        self.tick()

    def _show_figures(self, figures: Sequence[str]) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from PyHydroGeophysX.qt_apps.modules.base import thumbnail_pixmap

        shown = 0
        for path in list(figures)[:4]:
            pixmap = thumbnail_pixmap(path, 72)
            if pixmap is None:
                continue
            thumb = _label("", "stepThumb")
            thumb.setPixmap(pixmap)
            thumb.setToolTip(f"{path}\nClick to open full size.")
            thumb.setCursor(Qt.PointingHandCursor)
            thumb.mousePressEvent = (lambda _e, p=path: QDesktopServices.openUrl(
                QUrl.fromLocalFile(p)))
            self._figures.addWidget(thumb)
            shown += 1
        if shown:
            self._figures.addStretch(1)
        self._figures_host.setVisible(bool(shown))


class _ThinkingCard(_Card):
    """Between steps: the agent deciding what to do next, and for how long.

    It is also where a paused run asks its question, with the choices as
    buttons, so a decision is made in the timeline the user is watching.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        from PySide6.QtWidgets import QPlainTextEdit

        super().__init__(parent)
        self._since = time.monotonic()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(12)
        outer.addLayout(row)
        self.orb = AiOrb(26)
        row.addWidget(self.orb, 0, Qt.AlignVCenter)
        self.text = ShimmerText("", bold=True)
        row.addWidget(self.text, 1)
        self._clock = _label("", "agentClock")
        row.addWidget(self._clock)
        # The model's reasoning, as it writes it (``phase="thought"``).
        self._thought = _label("", "agentThought")
        self._thought.setWordWrap(True)
        self._thought.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._thought.setContentsMargins(38, 0, 0, 0)
        self._thought.setVisible(False)
        outer.addWidget(self._thought)
        self._question = _label("", "agentQuestion")
        self._question.setWordWrap(True)
        self._question.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._question.setContentsMargins(38, 0, 0, 0)
        self._question.setVisible(False)
        outer.addWidget(self._question)
        # Code a question asks about goes in a box that can be read and copied,
        # not into prose: approving code means having read it.
        self._code = QPlainTextEdit()
        self._code.setObjectName("agentCode")
        self._code.setReadOnly(True)
        self._code.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._code.setMaximumHeight(200)
        self._code.setVisible(False)
        outer.addWidget(self._code)
        self._choices_host = QWidget()
        self._choices_host.setObjectName("thinkingChoices")
        self._choices = QHBoxLayout(self._choices_host)
        self._choices.setContentsMargins(38, 2, 0, 0)
        self._choices.setSpacing(8)
        self._choices_host.setVisible(False)
        outer.addWidget(self._choices_host)

    def show_text(self, text: str, state: str = THINKING) -> None:
        if text != self.text.text() or state != self.orb.state():
            self._since = time.monotonic()
        self.text.setText(text)
        self.orb.set_state(state)
        self.set_accent(state)
        self.text.set_shimmer(state in _BUSY, tone=STATE_TONE.get(state, "text"))
        self.tick()
        self.setVisible(True)

    def show_thought(self, text: str) -> None:
        """The reasoning so far, with a caret where the model is writing."""
        text = str(text or "")
        caret = f"<span style='color:{CARET_GRAY};'>&#9613;</span>"
        self._thought.setText(html.escape(text).replace(chr(10), "<br>") + caret)
        self._thought.setVisible(bool(text))

    def ask(self, prompt: str, options: Sequence[Dict[str, str]], on_choice) -> bool:
        """Show a question with its options as buttons; False if none has an id."""
        from PySide6.QtWidgets import QPushButton

        from PyHydroGeophysX.qt_apps.modules.base import _split_code

        self.clear_question()
        shown = 0
        for option in options or []:
            decision = str(option.get("id") or "")
            if not decision:
                continue
            button = QPushButton(str(option.get("label") or decision))
            if not shown:
                button.setProperty("primary", True)
            if option.get("detail"):
                button.setToolTip(str(option["detail"]))
            button.clicked.connect(lambda _checked=False, d=decision: on_choice(d))
            self._choices.addWidget(button)
            shown += 1
        if not shown:
            return False
        self._choices.addStretch(1)
        head, code = _split_code(prompt)
        self._question.setText(html.escape(head).replace("\n", "<br>"))
        self._question.setVisible(bool(head))
        self._code.setPlainText(code)
        self._code.setVisible(bool(code))
        self._choices_host.setVisible(True)
        return True

    def clear_question(self) -> None:
        while self._choices.count():
            item = self._choices.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._choices_host.setVisible(False)
        self._question.setVisible(False)
        self._code.setVisible(False)
        self._code.clear()

    def tick(self) -> None:
        self._clock.setText(_clock(time.monotonic() - self._since))

    def stop(self) -> None:
        self.clear_question()
        self._thought.clear()
        self._thought.setVisible(False)
        self.orb.set_state(IDLE)
        self.text.set_shimmer(False)
        self.setVisible(False)


class AgentTimeline(QScrollArea):
    """The run as the assistant sees it: the goal, then one card per step it took.

    Follows the newest card unless the user has scrolled up to read an older
    one, so reading back is never fought by the auto-scroll.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("agentTimelineBody")
        self._layout = QVBoxLayout(body)
        self._layout.setContentsMargins(16, 16, 16, 16)
        self._layout.setSpacing(10)
        self._goal = _label("", "agentGoal")
        self._goal.setWordWrap(True)
        self._goal.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._layout.addWidget(self._goal)
        self._cards = QVBoxLayout()
        self._cards.setSpacing(10)
        self._layout.addLayout(self._cards)
        self._thinking = _ThinkingCard()
        self._thinking.setVisible(False)
        self._layout.addWidget(self._thinking)
        self._layout.addStretch(1)
        self.setWidget(body)
        self._steps: List[StepCard] = []
        self._finish_card: Optional[QWidget] = None
        self._notes: List["NoteCard"] = []
        self._streamed = ""
        self._follow = True
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        self.verticalScrollBar().valueChanged.connect(self._on_scrolled)
        self.verticalScrollBar().rangeChanged.connect(self._on_range)
        self.reset("")

    # -- what the page calls --------------------------------------------------
    def reset(self, goal: str) -> None:
        """Start a new run's timeline, headed by the request it is carrying out."""
        for card in self._steps:
            card.deleteLater()
        self._steps = []
        if self._finish_card is not None:
            self._finish_card.deleteLater()
            self._finish_card = None
        for card in self._notes:
            card.deleteLater()
        self._notes = []
        self._streamed = ""
        self._follow = True
        if goal:
            self._goal.setText(
                f"<span style='color:{CARET_GRAY}; font-size:8pt; font-weight:600;"
                " letter-spacing:1px;'>YOUR GOAL</span><br>"
                + html.escape(goal).replace("\n", "<br>"))
        else:
            self._goal.setText(
                "Describe a goal in the assistant panel on the right and choose <b>Auto "
                "to report</b>. Each step the assistant decides on appears here as it happens: "
                "why it chose it, where it worked, and what it found.")
        self._thinking.setVisible(False)
        self._clock.stop()

    def thinking(self, text: str, state: str = THINKING) -> None:
        """Show the agent busy between steps - deciding, reading, waiting."""
        self._thinking.show_text(text, state)
        self._clock.start()

    def ask(self, text: str, prompt: str, options: Sequence[Dict[str, str]],
            on_choice) -> bool:
        """Pause the timeline on a question: ``text`` as the heading, the
        ``prompt`` beneath it, ``options`` as buttons calling ``on_choice(id)``.

        Returns False, showing nothing, when no option carries an id.
        """
        self._thinking.show_text(text, WAITING)
        if not self._thinking.ask(prompt, options, on_choice):
            return False
        self._follow = True
        self._clock.start()
        return True

    def clear_question(self) -> None:
        self._thinking.clear_question()

    def thought(self, text: str) -> None:
        """The model's reasoning while it decides, as far as it has written it."""
        self._streamed = str(text or "")
        self._thinking.show_thought(self._streamed)
        self._thinking.setVisible(True)
        self._clock.start()

    def note(self, text: str) -> "NoteCard":
        """Something the user told the run; it waits for the next decision."""
        card = NoteCard(text)
        self._cards.addWidget(card)
        self._notes.append(card)
        self._follow = True
        return card

    def notes_read(self, notes: Sequence[str], why: str = "",
                   changes: Sequence[str] = (), heard: bool = True) -> None:
        """The run read ``notes``: say what it made of them on their cards."""
        waiting = [card for card in self._notes if card.pending]
        for text in notes or []:
            card = next((c for c in waiting if c.text == text), None)
            if card is None:
                card = self.note(text)
            else:
                waiting.remove(card)
            card.read(why, changes, heard)

    def step_started(self, label: str, module: str = "", reason: str = "") -> StepCard:
        """A step began: add its card and stop showing the agent as deciding."""
        self._thinking.stop()
        streamed, self._streamed = self._streamed, ""
        card = StepCard(label, module, reason,
                        typed=not (reason and streamed and reason.strip() == streamed.strip()))
        self._cards.addWidget(card)
        self._steps.append(card)
        self._clock.start()
        return card

    def current(self) -> Optional[StepCard]:
        """The step still running, or None."""
        if self._steps and self._steps[-1].status == "running":
            return self._steps[-1]
        return None

    def step_done(self, label: str, status: str, summary: str = "",
                  figures: Sequence[str] = (), module: str = "",
                  seconds: Optional[float] = None) -> Optional[StepCard]:
        """A step ended. A card is added for it if its start was never seen."""
        card = self.current()
        if card is None or (label and card.label != label):
            card = self.step_started(label, module)
        card.finish(status, summary, figures, seconds)
        return card

    def step_skipped(self, label: str) -> None:
        card = self.step_started(label)
        card.finish("skipped", "you chose to leave this step out.")

    def finish(self) -> None:
        """The run is over: nothing is deciding or running any more."""
        self._thinking.stop()
        card = self.current()
        if card is not None:
            card.finish("stopped", "the run ended before this step reported back.")
        self._clock.stop()

    def steps(self) -> List[StepCard]:
        return list(self._steps)

    def add_finish(self, card: QWidget) -> None:
        """End the timeline with ``card`` - the run's outcome - in view."""
        if self._finish_card is not None:
            self._finish_card.deleteLater()
        self._finish_card = card
        self._cards.addWidget(card)
        self._follow = True

    def finish_card(self) -> Optional[QWidget]:
        return self._finish_card

    def scroll_to_step(self, index: int) -> None:
        """Bring the card of the ``index``-th step taken into view."""
        if 0 <= index < len(self._steps):
            self._follow = False
            self.ensureWidgetVisible(self._steps[index], 0, 24)

    # -- internals -----------------------------------------------------------
    def _tick(self) -> None:
        card = self.current()
        if card is not None:
            card.tick()
        if self._thinking.isVisible():
            self._thinking.tick()

    def _on_scrolled(self, value: int) -> None:
        bar = self.verticalScrollBar()
        self._follow = value >= bar.maximum() - 24

    def _on_range(self, _low: int, high: int) -> None:
        if self._follow:
            self.verticalScrollBar().setValue(high)


class AgentHeader(_Card):
    """The Workflow page's banner while an assistant runs: orb, what it is doing, clock."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(14, 10, 18, 10)
        row.setSpacing(14)
        self.orb = AiOrb(46)
        row.addWidget(self.orb, 0, Qt.AlignVCenter)
        column = QVBoxLayout()
        column.setSpacing(2)
        self.headline = ShimmerText("", point_size=13)
        column.addWidget(self.headline)
        self.detail = TypewriterLabel(name="agentDetail")
        column.addWidget(self.detail)
        row.addLayout(column, 1)
        right = QVBoxLayout()
        right.setSpacing(0)
        self.clock = _label("0 s", "agentBigClock")
        self.clock.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        right.addWidget(self.clock)
        self.counter = _label("", "agentCounter")
        self.counter.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        right.addWidget(self.counter)
        self.usage = _label("", "agentCounter")
        self.usage.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.usage.setToolTip("Tokens the run's model calls have used, and their cost "
                              "estimated from the provider's list prices.")
        self.usage.setVisible(False)
        right.addWidget(self.usage)
        row.addLayout(right)

    def show_state(self, state: str, headline: str, detail: str = "") -> None:
        self.orb.set_state(state)
        self.set_accent(state)
        self.headline.setText(headline)
        self.headline.set_shimmer(state in _BUSY, tone=STATE_TONE.get(state, "text"))
        if detail is not None:
            self.detail.type_text(detail)
        self.setVisible(True)

    def set_usage(self, tokens: int, cost_usd: float, calls: int = 0) -> None:
        """What the run's model calls have used so far, and roughly cost."""
        self.usage.setText(usage_text(tokens, cost_usd))
        self.usage.setVisible(bool(tokens or calls))

    def clear_usage(self) -> None:
        self.usage.clear()
        self.usage.setVisible(False)

    def set_clock(self, seconds: float, steps_done: int = 0, running: bool = True,
                  ahead: Optional[int] = None) -> None:
        """The clock, and how far along: steps done and, while running, about
        how many the run's own route still puts ahead."""
        self.clock.setText(_clock(seconds))
        noun = "step" if steps_done == 1 else "steps"
        tail = ""
        if running:
            tail = f" · about {ahead} to go" if ahead else " · running"
        self.counter.setText(f"{steps_done} {noun} done{tail}")


# -- the route, the newest output, the end of a run --------------------------

def _gradient_color(fraction: float) -> QColor:
    """The agent's colours sampled ``fraction`` of the way along, 0 to 1."""
    stops = [QColor(name) for name in theme.ai_colors()]
    if len(stops) < 2:
        return stops[0] if stops else QColor(theme.PALETTE["primary"])
    f = max(0.0, min(1.0, float(fraction))) * (len(stops) - 1)
    i = min(int(f), len(stops) - 2)
    a, b, t = stops[i], stops[i + 1], f - i
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * t,
                           a.greenF() + (b.greenF() - a.greenF()) * t,
                           a.blueF() + (b.blueF() - a.blueF()) * t)


def _linear_ai(x0: float, x1: float, alpha: float = 1.0) -> QLinearGradient:
    """The agent's colours laid along a line from ``x0`` to ``x1``."""
    gradient = QLinearGradient(x0, 0, max(x1, x0 + 1.0), 0)
    names = theme.ai_colors()
    for index, name in enumerate(names):
        gradient.setColorAt(index / max(1, len(names) - 1), _color(name, alpha))
    return gradient


def _small_font(point_size: float = 8.5, weight=QFont.Normal) -> QFont:
    font = QFont()
    font.setFamilies([theme.UI_FONT_FAMILY, "Helvetica Neue", "Arial"])
    font.setPointSizeF(point_size)
    font.setWeight(weight)
    return font


class RunRoute(_Animated):
    """The run as a route: the steps taken, the one running, the ones ahead.

    The steps ahead are the run's own projection - the ``phase="route"`` events
    of :func:`PyHydroGeophysX.agents.runtime.entry.drive`, worked out from what
    the run has and recomputed after every step - so when the controller takes
    a turn the dependency order did not predict, the route redraws itself.
    Steps taken are drawn in the agent's colours, the running one with a ring
    turning round it, and while the agent decides a light travels toward the
    next stop. Stops still ahead are hollow, on a dashed line: plainly not done.

    Signals
    -------
    nodeClicked(int)
        The index, among the steps taken, of a stop the user clicked.
    """

    nodeClicked = Signal(int)

    #: How a step's status is drawn; anything else is a failure.
    _DRAWN = {"ok": "ok", "skipped": "skipped", "stopped": "skipped",
              "blocked": "skipped", "running": "running"}
    _NODE_Y = 46.0

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._taken: List[Dict[str, str]] = []
        self._ahead: List[Dict[str, str]] = []
        self._thinking = False
        self._over = False
        self._fill = 0.0
        self.setFixedHeight(94)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    # -- what the page calls --------------------------------------------------
    def reset(self) -> None:
        self._taken, self._ahead = [], []
        self._thinking = self._over = False
        self._fill = 0.0
        self._sync_timer()
        self.update()

    def set_ahead(self, ahead: Sequence[Dict[str, str]]) -> None:
        """The steps still ahead, as the run last projected them."""
        self._ahead = [{"tool": str(step.get("tool") or ""),
                        "label": str(step.get("label") or step.get("tool") or "Step")}
                       for step in ahead or [] if isinstance(step, dict)]
        self.update()

    def set_thinking(self, on: bool) -> None:
        """The agent is choosing the next step: light travels toward it."""
        self._thinking = bool(on) and not self._over
        self._sync_timer()
        self.update()

    def step_started(self, tool: str, label: str) -> None:
        self._drop_ahead(tool, label)
        self._taken.append({"tool": str(tool or ""), "label": str(label or tool or "Step"),
                            "status": "running"})
        self._thinking = False
        self._sync_timer()
        self.update()

    def step_done(self, tool: str, label: str, status: str) -> None:
        """A step ended; a stop is added for it if its start was never seen."""
        node = self._taken[-1] if self._taken and self._taken[-1]["status"] == "running" else None
        if node is None or (node["tool"] != tool and node["label"] != label):
            self._drop_ahead(tool, label)
            node = {"tool": str(tool or ""), "label": str(label or tool or "Step")}
            self._taken.append(node)
        node["status"] = self._DRAWN.get(str(status or "ok"), "failed")
        if node["status"] == "running":
            node["status"] = "ok"
        self._sync_timer()
        self.update()

    def finish(self) -> None:
        """The run is over: nothing runs or decides any more."""
        for node in self._taken:
            if node["status"] == "running":
                node["status"] = "skipped"
        self._thinking = False
        self._over = True
        self._sync_timer()
        self.update()

    def counts(self) -> tuple:
        """``(steps taken and ended, steps projected ahead)``."""
        return (sum(1 for node in self._taken if node["status"] != "running"),
                len(self._ahead))

    def nodes(self) -> List[Dict[str, str]]:
        """Every stop in order: taken ones with their status, then ``ahead``."""
        return [dict(node) for node in self._taken] + [
            dict(step, status="ahead") for step in self._ahead]

    # -- internals -----------------------------------------------------------
    def _drop_ahead(self, tool: str, label: str) -> None:
        for index, step in enumerate(self._ahead):
            if (tool and step["tool"] == tool) or (not tool and label and step["label"] == label):
                del self._ahead[index]
                return

    def _target(self) -> float:
        return float(max(0, len(self._taken) - 1))

    def _animating(self) -> bool:
        return (self._thinking or abs(self._target() - self._fill) > 0.004
                or any(node["status"] == "running" for node in self._taken))

    def _frame(self) -> None:
        target = self._target()
        self._fill += (target - self._fill) * 0.16
        if abs(target - self._fill) < 0.004:
            self._fill = target
        self.update()
        self._sync_timer()

    def _positions(self, count: int) -> List[float]:
        left, right = 34.0, self.width() - 34.0
        span = max(0.0, right - left)
        gap = min(210.0, span / (count - 1)) if count > 1 else 0.0
        start = left + (span - gap * (count - 1)) / 2.0
        return [start + index * gap for index in range(count)]

    @staticmethod
    def _x_at(xs: List[float], position: float) -> float:
        if not xs:
            return 0.0
        position = max(0.0, min(len(xs) - 1.0, position))
        low = int(position)
        if low >= len(xs) - 1:
            return xs[-1]
        return xs[low] + (xs[low + 1] - xs[low]) * (position - low)

    def _caption(self) -> str:
        # How far along is the banner's to say; this says what the route is.
        done, _ahead = self.counts()
        if self._over:
            return f"{done} step{'' if done == 1 else 's'} taken"
        return "redrawn after every step"

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        p = theme.PALETTE
        w, h = float(self.width()), float(self.height())
        painter.fillRect(self.rect(), QColor(p["bg"]))
        painter.setPen(QPen(QColor(p["border"]), 1))
        painter.drawLine(QPointF(0, h - 0.5), QPointF(w, h - 0.5))
        tag = _small_font(7.5, QFont.DemiBold)
        tag.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        painter.setFont(tag)
        painter.setPen(QColor(p["tertiary"]))
        painter.drawText(QRectF(16, 6, w / 2, 16), Qt.AlignLeft | Qt.AlignVCenter,
                         "ROUTE · AS THINGS STAND")
        painter.setFont(_small_font(8))
        painter.setPen(QColor(p["muted"]))
        painter.drawText(QRectF(w / 2, 6, w / 2 - 16, 16), Qt.AlignRight | Qt.AlignVCenter,
                         self._caption())
        nodes = self.nodes()
        if not nodes:
            painter.setPen(QColor(p["tertiary"]))
            painter.drawText(QRectF(0, 26, w, 40), Qt.AlignCenter,
                             "The route appears once the assistant has read your request.")
            return
        xs = self._positions(len(nodes))
        y = self._NODE_Y
        t = self._elapsed()
        # The way ahead: a dashed line through every stop.
        dashed = QPen(_color(p["tertiary"], 0.7), 1.5, Qt.CustomDashLine)
        dashed.setDashPattern([2.5, 3.5])
        dashed.setCapStyle(Qt.RoundCap)
        painter.setPen(dashed)
        painter.drawLine(QPointF(xs[0], y), QPointF(xs[-1], y))
        # The way taken, in the agent's colours, growing as steps are taken.
        if self._taken and len(xs) > 1:
            solid = QPen(QBrush(_linear_ai(xs[0], xs[-1])), 3.0)
            solid.setCapStyle(Qt.RoundCap)
            painter.setPen(solid)
            painter.drawLine(QPointF(xs[0], y), QPointF(self._x_at(xs, self._fill), y))
        # Deciding: a light travels from the last stop toward the next.
        if self._thinking and len(self._taken) < len(nodes):
            k = len(self._taken)
            x_from = xs[k - 1] if k else xs[0] - 26.0
            x_to = xs[k]
            phase = (t % 1.3) / 1.3
            head = x_from + (x_to - x_from) * phase
            tail = max(x_from, head - 46.0)
            colors = theme.ai_colors()
            comet = QLinearGradient(tail, 0, max(head, tail + 1.0), 0)
            comet.setColorAt(0.0, _color(colors[1], 0.0))
            comet.setColorAt(1.0, _color(colors[2], 0.95))
            pen = QPen(QBrush(comet), 3.0)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(tail, y), QPointF(head, y))
            painter.setPen(Qt.NoPen)
            painter.setBrush(_color(colors[2], 0.9))
            painter.drawEllipse(QPointF(head, y), 2.6, 2.6)
        next_index = len(self._taken) if self._thinking else -1
        for index, (node, x) in enumerate(zip(nodes, xs)):
            self._paint_node(painter, node["status"], QPointF(x, y),
                             (x - xs[0]) / max(1.0, xs[-1] - xs[0]), t,
                             is_next=index == next_index)
        self._paint_labels(painter, nodes, xs, next_index)

    def _paint_node(self, painter: QPainter, status: str, c: QPointF, fraction: float,
                    t: float, is_next: bool = False) -> None:
        p = theme.PALETTE
        colors = theme.ai_colors()
        painter.setPen(Qt.NoPen)
        if status == "running":
            pulse = 0.5 + 0.5 * math.sin(t * math.tau / 1.8)
            halo = QRadialGradient(c, 19)
            halo.setColorAt(0.4, _color(colors[1], 0.28 + 0.18 * pulse))
            halo.setColorAt(1.0, _color(colors[2], 0.0))
            painter.setBrush(halo)
            painter.drawEllipse(c, 19, 19)
            painter.setBrush(QColor(p["card"]))
            painter.drawEllipse(c, 10.5, 10.5)
            start = (t * 1.1 * 360.0) % 360.0
            ring = QPen(QBrush(_ai_gradient(c, start)), 2.6)
            ring.setCapStyle(Qt.RoundCap)
            painter.setPen(ring)
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(QRectF(c.x() - 9, c.y() - 9, 18, 18), int(start * 16), 270 * 16)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(_ai_gradient(c, -start)))
            core = 3.4 + 0.8 * pulse
            painter.drawEllipse(c, core, core)
            return
        if status == "ahead":
            painter.setBrush(QColor(p["card"]))
            if is_next:
                pulse = 0.5 + 0.5 * math.sin(t * math.tau / 1.3)
                painter.setPen(QPen(QBrush(_ai_gradient(c, t * 200.0, 0.45 + 0.5 * pulse)), 2.0))
            else:
                painter.setPen(QPen(_color(p["tertiary"], 0.9), 1.5))
            painter.drawEllipse(c, 7.0, 7.0)
            return
        fill = {"ok": _gradient_color(fraction), "failed": QColor(p["vivid_red"]),
                "skipped": _color(p["vivid_gray"], 0.6)}.get(status, QColor(p["vivid_red"]))
        painter.setBrush(fill)
        painter.drawEllipse(c, 9.0, 9.0)
        glyph = QPen(QColor(255, 255, 255), 1.8)
        glyph.setCapStyle(Qt.RoundCap)
        glyph.setJoinStyle(Qt.RoundJoin)
        painter.setPen(glyph)
        painter.setBrush(Qt.NoBrush)
        if status == "ok":
            path = QPainterPath(QPointF(c.x() - 4.0, c.y() + 0.2))
            path.lineTo(QPointF(c.x() - 1.2, c.y() + 3.0))
            path.lineTo(QPointF(c.x() + 4.2, c.y() - 2.8))
            painter.drawPath(path)
        elif status == "skipped":
            painter.drawLine(QPointF(c.x() - 3.5, c.y()), QPointF(c.x() + 3.5, c.y()))
        else:
            painter.drawLine(QPointF(c.x() - 3, c.y() - 3), QPointF(c.x() + 3, c.y() + 3))
            painter.drawLine(QPointF(c.x() + 3, c.y() - 3), QPointF(c.x() - 3, c.y() + 3))

    def _paint_labels(self, painter: QPainter, nodes: List[Dict[str, str]],
                      xs: List[float], next_index: int) -> None:
        p = theme.PALETTE
        gap = (xs[1] - xs[0]) if len(xs) > 1 else 200.0
        focus = next((i for i, node in enumerate(nodes) if node["status"] == "running"),
                     next_index)
        roomy = gap >= 64.0
        for index, (node, x) in enumerate(zip(nodes, xs)):
            if not roomy and index != focus:
                continue
            width = max(64.0, gap - 8.0) if roomy else 180.0
            status = node["status"]
            font = _small_font(8.5, QFont.DemiBold if index == focus else QFont.Normal)
            painter.setFont(font)
            tone = "text" if index == focus else {"ok": "muted", "failed": "red"}.get(
                status, "tertiary")
            painter.setPen(QColor(p[tone]))
            text = QFontMetrics(font).elidedText(node["label"], Qt.ElideRight, int(width))
            left = max(4.0, min(self.width() - 4.0 - width, x - width / 2.0))
            painter.drawText(QRectF(left, self._NODE_Y + 16, width, 18),
                             Qt.AlignHCenter | Qt.AlignTop, text)

    def _node_at(self, x: float) -> int:
        nodes = self.nodes()
        xs = self._positions(len(nodes))
        best = min(range(len(xs)), key=lambda i: abs(xs[i] - x), default=-1)
        return best if best >= 0 and abs(xs[best] - x) <= 16 else -1

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt event override
        from PySide6.QtWidgets import QToolTip

        index = self._node_at(event.position().x())
        if index < 0:
            QToolTip.hideText()
            self.setCursor(Qt.ArrowCursor)
            return
        node = self.nodes()[index]
        state = {"ok": "Done", "failed": "Did not complete", "skipped": "Left out",
                 "running": "Running now",
                 "ahead": "Ahead, as things stand - not started"}.get(node["status"], "")
        QToolTip.showText(event.globalPosition().toPoint(), f"{node['label']}\n{state}", self)
        self.setCursor(Qt.PointingHandCursor if index < len(self._taken) else Qt.ArrowCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt event override
        index = self._node_at(event.position().x())
        if 0 <= index < len(self._taken):
            self.nodeClicked.emit(index)


class _FigureView(_Animated):
    """A figure drawn to fit, revealed top to bottom behind a band of the
    agent's colours when it is new - the moment the run wrote it."""

    REVEAL_S = 0.9

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = None
        self._previous = None
        self._revealed_at: Optional[float] = None
        self.setMinimumSize(180, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def show_pixmap(self, pixmap, reveal: bool = True) -> None:
        self._previous = self._pixmap if reveal else None
        self._pixmap = pixmap
        self._revealed_at = time.monotonic() if (reveal and pixmap is not None) else None
        self._sync_timer()
        self.update()

    def _progress(self) -> float:
        if self._revealed_at is None:
            return 1.0
        x = min(1.0, (time.monotonic() - self._revealed_at) / self.REVEAL_S)
        return 1.0 - (1.0 - x) ** 3

    def _animating(self) -> bool:
        return self._revealed_at is not None

    def _frame(self) -> None:
        if self._progress() >= 1.0:
            self._revealed_at = None
            self._previous = None
            self._sync_timer()
        self.update()

    def _fit(self, pixmap) -> QRectF:
        area = QRectF(self.rect()).adjusted(10, 10, -10, -10)
        if pixmap is None or pixmap.isNull() or area.width() <= 0 or area.height() <= 0:
            return area
        scale = min(area.width() / pixmap.width(), area.height() / pixmap.height())
        width, height = pixmap.width() * scale, pixmap.height() * scale
        return QRectF(area.center().x() - width / 2, area.center().y() - height / 2,
                      width, height)

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        p = theme.PALETTE
        painter.setPen(QPen(QColor(p["border"]), 1))
        painter.setBrush(QColor(p["canvas"]))
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
        if self._pixmap is None:
            return
        progress = self._progress()
        if self._previous is not None and progress < 1.0:
            painter.setOpacity(1.0 - progress)
            painter.drawPixmap(self._fit(self._previous), self._previous,
                               QRectF(self._previous.rect()))
            painter.setOpacity(1.0)
        rect = self._fit(self._pixmap)
        shown = QRectF(rect.left(), rect.top(), rect.width(), rect.height() * progress)
        painter.save()
        painter.setClipRect(shown)
        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))
        painter.restore()
        if progress < 1.0:
            edge = shown.bottom()
            purple = theme.ai_colors()[1]
            glow = QLinearGradient(0, edge - 26, 0, edge)
            glow.setColorAt(0.0, _color(purple, 0.0))
            glow.setColorAt(1.0, _color(purple, 0.22))
            painter.fillRect(QRectF(rect.left(), edge - 26, rect.width(), 26), glow)
            painter.fillRect(QRectF(rect.left(), edge - 1.5, rect.width(), 3),
                             QBrush(_linear_ai(rect.left(), rect.right(), 0.95)))


class LiveCanvas(_Card):
    """The newest figure the run has written, large, as soon as it is written.

    A run's evidence is its figures; a thumbnail in a step card is a promise of
    one. This shows each new figure at full size the moment it appears in the
    run's folder, with every earlier one in a strip beneath - click one to look
    at it again, double-click to open it. Hidden until the run writes its first.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(_label("LATEST OUTPUT", "agentGoalTag"))
        top.addStretch(1)
        self._step = _label("", "agentChip")
        self._step.setVisible(False)
        top.addWidget(self._step)
        self._when = _label("", "agentClock")
        top.addWidget(self._when)
        outer.addLayout(top)
        self.view = _FigureView()
        self.view.setCursor(Qt.PointingHandCursor)
        self.view.setToolTip("Double-click to open the figure full size.")
        self.view.mouseDoubleClickEvent = lambda _e: self._open(self._current)
        outer.addWidget(self.view, 1)
        self._caption = _label("", "agentReason")
        self._caption.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self._caption)
        self._strip = QScrollArea()
        self._strip.setObjectName("agentFilmstrip")
        self._strip.setFrameShape(QFrame.NoFrame)
        self._strip.setWidgetResizable(True)
        self._strip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._strip.setFixedHeight(80)
        body = QWidget()
        body.setObjectName("agentFilmstrip")
        self._thumbs = QHBoxLayout(body)
        self._thumbs.setContentsMargins(0, 2, 0, 2)
        self._thumbs.setSpacing(6)
        self._thumbs.addStretch(1)
        self._strip.setWidget(body)
        outer.addWidget(self._strip)
        self._labels: Dict[str, QLabel] = {}
        self._steps: Dict[str, str] = {}
        self._current = ""
        self._show_age = True
        self.setVisible(False)

    def set_show_age(self, show: bool) -> None:
        """Say how long ago a figure was written - true live, not in a replay."""
        self._show_age = bool(show)
        self._when.setVisible(self._show_age)

    def reset(self) -> None:
        while self._thumbs.count() > 1:
            item = self._thumbs.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._labels, self._steps = {}, {}
        self._current = ""
        self.view.show_pixmap(None, reveal=False)
        self.setVisible(False)

    def figures(self) -> List[str]:
        return list(self._labels)

    def current(self) -> str:
        return self._current

    def add_figure(self, path: str, step: str = "") -> bool:
        """Show a figure the run has just written; False if known or unreadable."""
        from PySide6.QtGui import QPixmap

        from PyHydroGeophysX.qt_apps.modules.base import thumbnail_pixmap

        path = str(path)
        if path in self._labels:
            return False
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return False
        if max(pixmap.width(), pixmap.height()) > 2000:
            pixmap = pixmap.scaled(2000, 2000, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumb = _label("", "filmThumb")
        small = thumbnail_pixmap(path, 60)
        if small is not None:
            thumb.setPixmap(small)
        thumb.setToolTip(f"{os.path.basename(path)}\nClick to show, double-click to open.")
        thumb.setCursor(Qt.PointingHandCursor)
        thumb.mousePressEvent = lambda _e, p=path: self._select(p)
        thumb.mouseDoubleClickEvent = lambda _e, p=path: self._open(p)
        self._thumbs.insertWidget(self._thumbs.count() - 1, thumb)
        self._labels[path] = thumb
        self._steps[path] = str(step or "")
        self._show(path, pixmap, reveal=True)
        self.setVisible(True)
        bar = self._strip.horizontalScrollBar()
        QTimer.singleShot(0, lambda: bar.setValue(bar.maximum()))
        return True

    def tick(self) -> None:
        """Keep "written 12 s ago" true."""
        if not self._current:
            return
        try:
            age = time.time() - os.path.getmtime(self._current)
        except OSError:
            self._when.setText("")
            return
        self._when.setText("written just now" if age < 5 else f"written {_clock(age)} ago")

    def _select(self, path: str) -> None:
        from PySide6.QtGui import QPixmap

        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self._show(path, pixmap, reveal=False)

    def _show(self, path: str, pixmap, reveal: bool) -> None:
        self._current = path
        self.view.show_pixmap(pixmap, reveal=reveal)
        step = self._steps.get(path, "")
        self._step.setText(html.escape(step))
        self._step.setVisible(bool(step))
        self._caption.setText(html.escape(os.path.basename(path)))
        self._caption.setToolTip(path)
        for other, thumb in self._labels.items():
            thumb.setProperty("selected", other == path)
            thumb.style().unpolish(thumb)
            thumb.style().polish(thumb)
        self.tick()

    @staticmethod
    def _open(path: str) -> None:
        if not path:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(path))


class FinishCard(_Card):
    """The end of a run as the user needs it: how it went, how long it took,
    what it made, what to check before relying on it, and where to go next.

    Parameters
    ----------
    state : str
        DONE, WAITING (finished, with things to review) or FAILED.
    title : str
        One line: "Report ready", "Could not finish"...
    seconds : float
        How long the run worked.
    stats : sequence of str
        Short facts shown as chips: "6 steps", "9 figures".
    gaps : sequence of str
        What the user should check: warnings, steps that did not complete.
    actions : sequence of (label, callable)
        Buttons; the first is the primary one.
    """

    def __init__(self, state: str, title: str, seconds: float,
                 stats: Sequence[str] = (), gaps: Sequence[str] = (),
                 actions: Sequence = (), parent: Optional[QWidget] = None) -> None:
        from PySide6.QtWidgets import QPushButton

        super().__init__(parent)
        self._finish_state = state if state in (DONE, WAITING, FAILED) else DONE
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(8)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.orb = AiOrb(34)
        self.orb.set_state(DONE if self._finish_state == WAITING else self._finish_state)
        top.addWidget(self.orb, 0, Qt.AlignVCenter)
        column = QVBoxLayout()
        column.setSpacing(1)
        self.title = _label(html.escape(title), "agentFinishTitle")
        self.title.setWordWrap(True)
        column.addWidget(self.title)
        column.addWidget(_label(f"Worked for {_clock(seconds)}", "agentDetail"))
        top.addLayout(column, 1)
        outer.addLayout(top)
        indent = 46
        if stats:
            chips = FlowLayout()
            chips.setContentsMargins(indent, 0, 0, 0)
            for fact in stats:
                chips.addWidget(_label(html.escape(str(fact)), "agentChip"))
            outer.addLayout(chips)
        gaps = [str(gap) for gap in gaps if str(gap).strip()]
        if gaps:
            tag = _label("CHECK BEFORE YOU RELY ON IT", "agentGoalTag")
            tag.setContentsMargins(indent, 6, 0, 0)
            outer.addWidget(tag)
            for gap in gaps[:4]:
                line = _label("• " + html.escape(gap), "agentReason")
                line.setWordWrap(True)
                line.setContentsMargins(indent, 0, 0, 0)
                outer.addWidget(line)
            if len(gaps) > 4:
                more = _label(f"and {len(gaps) - 4} more in the report", "agentClock")
                more.setContentsMargins(indent, 0, 0, 0)
                outer.addWidget(more)
        if actions:
            row = FlowLayout(spacing=8)
            row.setContentsMargins(indent, 4, 0, 0)
            for index, (label, callback) in enumerate(actions):
                button = QPushButton(str(label))
                if index == 0:
                    button.setProperty("primary", True)
                button.clicked.connect(lambda _checked=False, f=callback: f())
                row.addWidget(button)
            outer.addLayout(row)
        self._faded_in = False

    def finish_state(self) -> str:
        return self._finish_state

    def showEvent(self, event) -> None:  # noqa: N802 - Qt event override
        super().showEvent(event)
        if self._faded_in:
            return
        self._faded_in = True
        from PySide6.QtCore import QPropertyAnimation
        from PySide6.QtWidgets import QGraphicsOpacityEffect

        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.0)
        self.setGraphicsEffect(effect)
        fade = QPropertyAnimation(effect, b"opacity", self)
        fade.setDuration(450)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        # Dropped once the card is in: under an opacity effect every repaint of
        # the orb inside it goes through an offscreen buffer.
        fade.finished.connect(lambda: self.setGraphicsEffect(None))
        fade.start()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt event override
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        p = theme.PALETTE
        rect = QRectF(self.rect()).adjusted(0.75, 0.75, -0.75, -0.75)
        if self._finish_state == DONE:
            pen = QPen(QBrush(_linear_ai(rect.left(), rect.right(), 0.9)), 1.5)
        else:
            pen = QPen(_color(p[_STATE_VIVID[self._finish_state]], 0.75), 1.5)
        painter.setPen(pen)
        painter.setBrush(QColor(p["card"]))
        painter.drawRoundedRect(rect, 12, 12)


class NoteCard(QFrame):
    """Something the user told the run while it worked, and what came of it.

    Drawn as the user's own message - tinted, without the agent's orb - so the
    timeline keeps clear what the assistant did and what it was told.
    """

    def __init__(self, text: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("noteCard")
        self.text = str(text)
        self.pending = True
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        column = QVBoxLayout(self)
        column.setContentsMargins(14, 10, 14, 10)
        column.setSpacing(3)
        column.addWidget(_label("YOU, DURING THE RUN", "agentGoalTag"))
        body = _label(html.escape(self.text), "agentSummary")
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        column.addWidget(body)
        self._status = ShimmerText("Waiting for the next decision…", bold=False)
        self._status.set_shimmer(True, tone="muted")
        column.addWidget(self._status)
        self._answer = TypewriterLabel(name="agentReason")
        self._answer.setVisible(False)
        column.addWidget(self._answer)

    def read(self, why: str = "", changes: Sequence[str] = (), heard: bool = True) -> None:
        """The run has read this note; ``why`` is what it decided with it."""
        self.pending = False
        self._status.set_shimmer(False)
        if not heard:
            self._status.setText("This run has no model to read notes; "
                                 "pause and stop still work.")
            return
        self._status.setText("Read before the next decision")
        lines = []
        if why:
            lines.append(f"Decided: {why}")
        for change in changes or []:
            lines.append(f"Changed {change}")
        if lines:
            self._answer.type_text("\n".join(lines))
            self._answer.setVisible(True)


class SteerBar(QFrame):
    """Under a running workflow: pause it, or tell it something.

    A note reaches the run with its next decision; pausing takes effect when
    the step that is running ends, because an inversion stopped half-way is no
    use to anyone. Both are the run's to act on - the bar only says so.

    Signals
    -------
    noteSent(str)
        The user's note, as typed.
    pauseRequested(bool)
        True to pause, False to resume.
    """

    noteSent = Signal(str)
    pauseRequested = Signal(bool)

    RUNNING, PAUSING, PAUSED = "running", "pausing", "paused"

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        from PySide6.QtWidgets import QLineEdit, QPushButton

        super().__init__(parent)
        self.setObjectName("steerBar")
        row = QHBoxLayout(self)
        row.setContentsMargins(16, 8, 16, 10)
        row.setSpacing(8)
        self.field = QLineEdit()
        self.field.setObjectName("steerField")
        self.field.returnPressed.connect(self._send)
        row.addWidget(self.field, 1)
        self.send = QPushButton("Send note")
        self.send.clicked.connect(self._send)
        row.addWidget(self.send)
        self.pause = QPushButton("Pause after this step")
        self.pause.clicked.connect(self._toggle)
        row.addWidget(self.pause)
        self._state = self.RUNNING
        self.set_name("the assistant")

    def set_name(self, name: str) -> None:
        self.field.setPlaceholderText(
            f"Tell {name} something while it works - e.g. use lambda 20, "
            "or leave the seismic line out")

    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        """``running``, ``pausing`` (asked, the step is finishing) or ``paused``."""
        self._state = state
        self.pause.setEnabled(state != self.PAUSING)
        self.pause.setText({self.RUNNING: "Pause after this step",
                            self.PAUSING: "Pausing after this step…",
                            self.PAUSED: "Resume"}.get(state, "Pause after this step"))
        self.pause.setProperty("primary", state == self.PAUSED)
        self.pause.style().unpolish(self.pause)
        self.pause.style().polish(self.pause)

    def _send(self) -> None:
        text = self.field.text().strip()
        if text:
            self.field.clear()
            self.noteSent.emit(text)

    def _toggle(self) -> None:
        if self._state == self.PAUSED:
            self.pauseRequested.emit(False)
        elif self._state == self.RUNNING:
            self.set_state(self.PAUSING)
            self.pauseRequested.emit(True)
