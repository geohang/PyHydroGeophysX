"""Native, adaptive welcome screen for the geophysics studio.

Home mirrors the navigator: the hero opens the Workflow group, one card per
group of the model-data loop opens that group's pages, and the workspace card
carries the Project group. A page added to the tree appears here too.
"""

from __future__ import annotations

import json

from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QLayout, QPushButton, QScrollArea,
    QSizePolicy, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from PyHydroGeophysX.agents import assistants as assistant_registry
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.layout_fit import elide_label
from PyHydroGeophysX.qt_apps.widgets.project_tree import TREE_STRUCTURE, item_icon

#: The model-data loop, one card per navigator group, in the order the loop
#: runs: (direction, title, what it does, tree group).
LOOP_TASKS = (
    ("Forward", "Simulate a survey",
     "Hydrologic model → petrophysics → predicted response.",
     "Hydro → Geophysics"),
    ("Inverse", "Invert field data",
     "Load, invert and evaluate measured data.",
     "Geophysical Data Processing"),
    ("Inverse", "Estimate hydrology",
     "Structure, water content and porosity, with uncertainty.",
     "Geophy → Hydrology"),
)

#: Below these widths of Home the illustration goes, the title steps down and
#: the project actions move under the project name.
_ART_MIN_WIDTH = 640
_COMPACT_WIDTH = 900
_ACTIONS_INLINE_WIDTH = 760


def _group_pages(group: str):
    """The pages a navigator group opens, as ``(label, key)``.

    A group whose items are all sections of one page opens that page once,
    under the group's own name.
    """
    items = list(dict(TREE_STRUCTURE).get(group, ()))
    if len(items) > 1 and len({key for _label, key in items}) == 1:
        return [(group, items[0][1])]
    return items


def _label(text: str, role: str, *, wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setObjectName(role)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(wrap)
    label.setMinimumWidth(0)
    # One line by design: the page-wide relax pass must not elide it to nothing.
    label.setProperty("fitted", not wrap)
    return label


def _mix(base: str, tint: str, amount: float) -> QColor:
    """``base`` moved ``amount`` of the way towards ``tint``."""
    a, b = QColor(base), QColor(tint)
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * amount,
                           a.greenF() + (b.greenF() - a.greenF()) * amount,
                           a.blueF() + (b.blueF() - a.blueF()) * amount)


class _FlowLayout(QLayout):
    """Lay items left to right and wrap them, like words; for links of any length."""

    def __init__(self, parent=None, spacing: int = 8) -> None:
        super().__init__(parent)
        self._items = []
        self._gap = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):  # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):  # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, width, 0), move=False)

    def setGeometry(self, rect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, move=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        # One line when there is room, so a row of links sits on one line.
        hints = [item.sizeHint() for item in self._items]
        width = sum(h.width() for h in hints) + self._gap * max(0, len(hints) - 1)
        height = max((h.height() for h in hints), default=0)
        return self._with_margins(QSize(width, height))

    def minimumSize(self) -> QSize:  # noqa: N802
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return self._with_margins(size)

    def _with_margins(self, size: QSize) -> QSize:
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _arrange(self, rect: QRect, *, move: bool) -> int:
        m = self.contentsMargins()
        area = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x, y, line = area.x(), y + line + self._gap, 0
            if move:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._gap
            line = max(line, hint.height())
        return y + line - rect.y() + m.bottom()


class _AdaptiveGrid(QWidget):
    """Reflow the same controls without rebuilding them or losing focus."""

    def __init__(self, widgets, cell_width: int, max_columns: int) -> None:
        super().__init__()
        self._widgets = widgets
        self._cell_width = cell_width
        self._max_columns = max_columns
        self._columns = 0
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(12)
        self._reflow()

    def _reflow(self) -> None:
        columns = max(1, min(self._max_columns,
                            (self.width() + 12) // (self._cell_width + 12)))
        if columns == self._columns:
            return
        for index in range(self._max_columns):
            self._grid.setColumnStretch(index, 0)
        for widget in self._widgets:
            self._grid.removeWidget(widget)
        for index, widget in enumerate(self._widgets):
            # A final solitary card uses the row rather than leaving a hole.
            span = columns if index == len(self._widgets) - 1 and index % columns == 0 else 1
            self._grid.addWidget(widget, index // columns, index % columns, 1, span)
        for index in range(columns):
            self._grid.setColumnStretch(index, 1)
        self._columns = columns

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._reflow()


class _AgentMark(QWidget):
    """A dot in the active assistant's colours: on Home, colour means the agent."""

    def __init__(self, diameter: int = 12) -> None:
        super().__init__()
        self.setFixedSize(diameter, diameter)
        theme.notifier().changed.connect(self.update)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        centre = QPointF(self.width() / 2, self.height() / 2)
        colors = theme.ai_colors()
        gradient = QConicalGradient(centre, 90)
        for index, name in enumerate(colors + colors[:1]):
            gradient.setColorAt(index / len(colors), QColor(name))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(centre, self.width() / 2 - 0.5, self.height() / 2 - 0.5)
        painter.end()


class _SubsurfaceIllustration(QWidget):
    """A decorative earth section; intentionally not a scientific data plot."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("HomeIllustration")
        self.setAccessibleName("Schematic of a survey above layered ground and groundwater")
        self.setToolTip("Conceptual subsurface illustration — not measured data")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        theme.notifier().changed.connect(self.update)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(300, 210)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scale = min(self.width() / 320, self.height() / 225)
        painter.translate((self.width() - 320 * scale) / 2,
                          (self.height() - 225 * scale) / 2)
        painter.scale(scale, scale)
        # Tints of the card it sits on, so both appearances need no colours of their own.
        card, ground = theme.color("card"), theme.color("tertiary")
        water, grass = theme.color("primary"), theme.color("vivid_green")

        def polygon(points, color):
            painter.setPen(Qt.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(QPolygonF([QPointF(x, y) for x, y in points]))

        # The oblique surface and the side give the section its depth.
        polygon([(30, 90), (125, 47), (291, 80), (203, 128)], _mix(card, grass, 0.16))
        polygon([(203, 128), (291, 80), (291, 150), (203, 198)], _mix(card, ground, 0.30))
        polygon([(30, 90), (203, 128), (203, 198), (30, 158)], _mix(card, ground, 0.16))
        bed = QPainterPath(QPointF(30, 111))
        bed.cubicTo(92, 109, 142, 145, 203, 147)
        bed.lineTo(203, 198)
        bed.lineTo(30, 158)
        bed.closeSubpath()
        painter.setBrush(_mix(card, water, 0.22))
        painter.drawPath(bed)
        bed = QPainterPath(QPointF(30, 142))
        bed.cubicTo(92, 132, 149, 169, 203, 175)
        bed.lineTo(203, 198)
        bed.lineTo(30, 158)
        bed.closeSubpath()
        painter.setBrush(_mix(card, ground, 0.30))
        painter.drawPath(bed)

        # Survey lines on the surface, with six acquisition points.
        painter.setPen(QPen(_mix(card, grass, 0.55), 0.8))
        for fraction in (0.25, 0.5, 0.75):
            painter.drawLine(QPointF(30 + 95 * fraction, 90 - 43 * fraction),
                             QPointF(203 + 88 * fraction, 128 - 48 * fraction))
            painter.drawLine(QPointF(30 + 173 * fraction, 90 + 38 * fraction),
                             QPointF(125 + 166 * fraction, 47 + 33 * fraction))
        painter.setPen(QPen(QColor(water), 1.6))
        painter.setBrush(QColor(card))
        for index in range(6):
            x, y = 54 + index * 27, 79 + index * 5.7
            painter.drawLine(QPointF(x, y), QPointF(x, y - 13))
            painter.drawEllipse(QPointF(x, y - 15), 3, 3)
        wave = QPainterPath(QPointF(55, 60))
        wave.cubicTo(90, 14, 168, 23, 189, 88)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(_mix(card, water, 0.70), 1.4, Qt.DashLine))
        painter.drawPath(wave)
        painter.setPen(QPen(_mix(card, water, 0.65), 1.2))
        for index in range(3):
            stream = QPainterPath(QPointF(57 + index * 42, 131 + index * 8))
            stream.cubicTo(75 + index * 42, 123 + index * 8,
                           80 + index * 42, 139 + index * 8,
                           99 + index * 42, 137 + index * 8)
            painter.drawPath(stream)
        painter.end()


class StudioHome(QWidget):
    navigateRequested = Signal(str)
    newProjectRequested = Signal()
    openProjectRequested = Signal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.setObjectName("StudioHome")
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setObjectName("HomeScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self._scroll)
        content = QWidget()
        content.setObjectName("HomeContent")
        self._scroll.setWidget(content)
        self._body = QVBoxLayout(content)
        self._body.setContentsMargins(24, 22, 24, 22)
        self._body.setSpacing(20)

        brand = QHBoxLayout()
        brand.addWidget(_label("PyHydroGeophysX", "HomeBrand", wrap=False))
        brand.addStretch()
        brand.addWidget(_label("PROFESSIONAL STUDIO", "HomeEyebrow", wrap=False))
        self._body.addLayout(brand)
        self._body.addWidget(self._build_workspace())
        self._body.addWidget(self._build_hero())

        self._body.addWidget(_label("Explore your research tasks", "HomeSection"))
        # A group the navigator no longer has drops its card, not the studio.
        cards = [self._task_card(*task) for task in LOOP_TASKS if _group_pages(task[-1])]
        self._body.addWidget(_AdaptiveGrid(cards, 245, 3))

        self._details = QToolButton()
        self._details.setObjectName("HomeDetails")
        self._details.setText("Session details")
        self._details.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._details.setArrowType(Qt.RightArrow)
        self._details.setCheckable(True)
        self._details.setCursor(Qt.PointingHandCursor)
        self._details.toggled.connect(self._toggle_details)
        self._body.addWidget(self._details, alignment=Qt.AlignLeft)
        self._summary = QTextEdit()
        self._summary.setObjectName("HomeSummary")
        self._summary.setAccessibleName("Session context")
        self._summary.setReadOnly(True)
        self._summary.setMinimumHeight(160)
        self._summary.setMaximumHeight(200)
        self._summary.hide()
        self._body.addWidget(self._summary)
        self._body.addStretch(1)
        self.refresh()

    # -- building ------------------------------------------------------------
    def _build_workspace(self) -> QFrame:
        """The active Project, its pages from the navigator, and New / Open."""
        card = QFrame()
        card.setObjectName("HomeWorkspace")
        self._workspace_grid = QGridLayout(card)
        self._workspace_grid.setContentsMargins(16, 12, 12, 12)
        self._workspace_grid.setHorizontalSpacing(12)
        self._workspace_grid.setVerticalSpacing(6)
        icon = QLabel()
        icon.setPixmap(theme.icon(item_icon("Project")).pixmap(QSize(20, 20)))
        self._workspace_grid.addWidget(icon, 0, 0, Qt.AlignVCenter)
        names = QVBoxLayout()
        names.setSpacing(2)
        self._workspace_name = elide_label(_label("", "HomeCardTitle", wrap=False))
        self._workspace_path = elide_label(_label("", "HomeDescription", wrap=False))
        names.addWidget(self._workspace_name)
        names.addWidget(self._workspace_path)
        self._workspace_grid.addLayout(names, 0, 1)
        self._workspace_grid.setColumnStretch(1, 1)

        self._workspace_actions = QWidget()
        actions = _FlowLayout(self._workspace_actions, spacing=2)
        for label, key in _group_pages("Project"):
            actions.addWidget(self._navigate_button(label, key, role="quiet"))
        for text, signal in (("New Project…", self.newProjectRequested),
                             ("Open Project…", self.openProjectRequested)):
            button = QPushButton(text)
            button.setProperty("homeRole", "quiet")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(signal.emit)
            actions.addWidget(button)
            if signal is self.newProjectRequested:
                self._new_project_button = button
        self._actions_below = None
        self._place_workspace_actions(below=False)
        return card

    def _build_hero(self) -> QFrame:
        hero = QFrame()
        hero.setObjectName("HomeHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 24, 24)
        hero_layout.setSpacing(16)
        words = QVBoxLayout()
        words.setSpacing(12)
        agent_line = QHBoxLayout()
        agent_line.setSpacing(8)
        self._agent_mark = _AgentMark()
        agent_line.addWidget(self._agent_mark, 0, Qt.AlignVCenter)
        self._agent_domain = elide_label(_label("", "HomeAccent", wrap=False))
        agent_line.addWidget(self._agent_domain, 1)
        words.addLayout(agent_line)
        self._title = _label("From data to insight.\nWith Agentic AI.", "HomeTitle")
        words.addWidget(self._title)
        self._pitch = _label("", "HomeDescription")
        words.addWidget(self._pitch)
        actions = QHBoxLayout()
        actions.addWidget(self._navigate_button("Start an AI workflow", "one_click",
                                                primary=True))
        actions.addStretch()
        words.addLayout(actions)
        words.addWidget(_label("Plan  →  Process  →  Evaluate  →  Report", "HomeAIStages"))
        hero_layout.addLayout(words, 3)
        self._art = _SubsurfaceIllustration()
        hero_layout.addWidget(self._art, 2)
        return hero

    def _navigate_button(self, text, key, *, primary=False, role=""):
        button = QPushButton(text)
        button.setProperty("primary", primary)
        button.setProperty("homeRole", role)
        button.setProperty("moduleKey", key)
        if role == "chip":
            # The navigator's own icon, so a card's link reads as that tree item;
            # the space keeps the caption off the icon, which QSS cannot pad.
            button.setIcon(theme.icon(item_icon(text)))
            button.setIconSize(QSize(13, 13))
            button.setText(" " + text)
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(lambda _checked=False: self.navigateRequested.emit(key))
        return button

    def _task_card(self, direction, title, description, group):
        card = QFrame()
        card.setObjectName("HomeTaskCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        layout.addWidget(_label(direction.upper(), "HomeDirection"))
        layout.addWidget(_label(title, "HomeCardTitle"))
        layout.addWidget(_label(description, "HomeDescription"))
        layout.addStretch(1)
        layout.addSpacing(4)
        links = _FlowLayout(spacing=6)
        for label, key in _group_pages(group):
            links.addWidget(self._navigate_button(label, key, role="chip"))
        layout.addLayout(links)
        return card

    # -- state ---------------------------------------------------------------
    def _place_workspace_actions(self, *, below: bool) -> None:
        """Beside the project name when there is room, under it when not."""
        if below == self._actions_below:
            return
        grid = self._workspace_grid
        grid.removeWidget(self._workspace_actions)
        if below:
            grid.addWidget(self._workspace_actions, 1, 1, 1, 2)
        else:
            grid.addWidget(self._workspace_actions, 0, 2, Qt.AlignVCenter)
        self._actions_below = below

    def _set_compact(self, compact: bool) -> None:
        if bool(self._title.property("compact")) == compact:
            return
        self._title.setProperty("compact", compact)
        self._title.style().unpolish(self._title)
        self._title.style().polish(self._title)

    def _toggle_details(self, checked):
        self._details.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._summary.setVisible(checked)

    def refresh(self) -> None:
        agent = assistant_registry.active()
        self._agent_domain.setText(f"Agentic AI  /  {agent.domain}".upper())
        self._pitch.setText(f"Let {agent.name} plan, run and explain your research workflow.")
        self._agent_mark.update()

        summary = self.state.context_summary()
        self._summary.setPlainText(json.dumps(summary, indent=2, default=str))
        folder = str(self.state.project_directory or "")
        # The fallback folder is named for what it is, and New Project stands out
        # while it is in use: it is shared by every session and every survey.
        default = bool(getattr(self.state, "default_project", False))
        name = self.state.project_name
        self._workspace_name.setText(name)
        self._workspace_path.setText(folder or "Choose a project folder to organise your research.")
        self._workspace_name.setToolTip(
            (folder or "") + ("\nShared by every session until you create a project."
                              if default else ""))
        self._workspace_path.setToolTip(folder or "")
        button = self._new_project_button
        if bool(button.property("primary")) != default:
            button.setProperty("primary", default)
            button.setProperty("homeRole", "" if default else "quiet")
            button.style().unpolish(button)
            button.style().polish(button)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        width = self.width()
        self._art.setVisible(width >= _ART_MIN_WIDTH)
        self._set_compact(width < _COMPACT_WIDTH)
        self._place_workspace_actions(below=width < _ACTIONS_INLINE_WIDTH)
        margin = 24 if width >= _ART_MIN_WIDTH else 14
        self._body.setContentsMargins(margin, 22, margin, 22)
