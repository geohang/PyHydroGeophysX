"""Central theme for the PyHydroGeophysX studio, in a light and a dark appearance.

One module owns the whole look so it stays consistent: the two palettes, a
hand-crafted QSS stylesheet, pyqtgraph and Matplotlib plot colors, a
``qtawesome`` icon helper (with a graceful no-op fallback when qtawesome is
absent), and the window icon.

The design is neutral greys and white (near-black in the dark appearance) for
everything structural, one blue accent for what can be acted
on, and the system green, orange and red kept for meaning - success, caution,
failure. Colour beyond that belongs to the agent alone: the edge glow and orb
that show AQUAH at work (:mod:`.widgets.ai_presence`).

Appearance is Light, Dark, or System (follow the operating system), chosen in
View > Appearance and remembered. ``PALETTE`` is one dictionary updated in
place when the appearance changes, so code that reads ``theme.PALETTE[...]``
when it draws always gets the current colours; anything styled through the
stylesheet below is restyled automatically.

Plot canvases stay light in the dark appearance, as a printed page would. Many
plots draw black traces and markers that would vanish on a dark
ground, and a colormap's colours carry meaning; the canvas is only dimmed a
little so it does not glare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPalette

# -- the two palettes --------------------------------------------------------
LIGHT: Dict[str, str] = {
    # accent: what can be acted on
    "primary": "#007aff",
    "primary_dark": "#0062cc",
    "accent": "#0071eb",
    "on_primary": "#ffffff",
    # meaning, in shades that read as text on the background
    "green": "#248a3d",
    "red": "#d70015",
    "amber": "#c93400",
    # surfaces
    "bg": "#f5f5f7",
    "card": "#ffffff",
    "field": "#ffffff",
    "menu_bg": "#ffffff",
    "material": "rgba(255, 255, 255, 240)",
    # text
    "text": "#1d1d1f",
    "muted": "#6e6e73",
    "tertiary": "#8e8e93",
    "disabled_text": "#aeaeb2",
    # lines and controls
    "border": "#e5e5ea",
    "border_blue": "#d1d1d6",      # control outline; name kept for callers
    "control": "#ffffff",
    "control_hover": "#f2f2f7",
    "control_pressed": "#e5e5ea",
    "tab_selected": "#e8e8ed",
    "track": "#e5e5ea",
    "scroll": "#c7c7cc",
    "select_bg": "#dceafe",
    "select_text": "#0058cc",
    "hover": "#efeff2",
    "tooltip_bg": "#ffffff",
    "tooltip_text": "#1d1d1f",
    # plot canvases
    "canvas": "#ffffff",
    "canvas_text": "#1d1d1f",
    # the system colours at full strength, for the agent and for status marks
    "vivid_blue": "#007aff",
    "vivid_purple": "#af52de",
    "vivid_pink": "#ff2d55",
    "vivid_orange": "#ff9500",
    "vivid_green": "#34c759",
    "vivid_red": "#ff3b30",
    "vivid_gray": "#8e8e93",
}

DARK: Dict[str, str] = {
    "primary": "#0a84ff",
    "primary_dark": "#0060df",
    "accent": "#409cff",
    "on_primary": "#ffffff",
    "green": "#30d158",
    "red": "#ff453a",
    "amber": "#ff9f0a",
    "bg": "#161618",
    "card": "#1f1f22",
    "field": "#2a2a2d",
    "menu_bg": "#2c2c2e",
    "material": "rgba(36, 36, 38, 240)",
    "text": "#f5f5f7",
    "muted": "#98989d",
    "tertiary": "#7c7c80",
    "disabled_text": "#5a5a5e",
    "border": "#333336",
    "border_blue": "#48484a",
    "control": "#2c2c2e",
    "control_hover": "#3a3a3c",
    "control_pressed": "#48484a",
    "tab_selected": "#3a3a3c",
    "track": "#3a3a3c",
    "scroll": "#58585c",
    "select_bg": "#163a63",
    "select_text": "#e6f0ff",
    "hover": "#2a2a2d",
    "tooltip_bg": "#2c2c2e",
    "tooltip_text": "#f5f5f7",
    "canvas": "#f2f2f4",
    "canvas_text": "#1d1d1f",
    "vivid_blue": "#0a84ff",
    "vivid_purple": "#bf5af2",
    "vivid_pink": "#ff375f",
    "vivid_orange": "#ff9f0a",
    "vivid_green": "#30d158",
    "vivid_red": "#ff453a",
    "vivid_gray": "#98989d",
}

#: The palette in force. Updated in place, never rebound, so every module that
#: imported it sees a change of appearance.
PALETTE: Dict[str, str] = dict(LIGHT)

#: Several data series at once, on a plot canvas, led by the blue a single
#: series is drawn in.
SERIES_COLORS: Tuple[str, ...] = ("#007aff", "#ff9500", "#34c759", "#ff3b30",
                                  "#af52de", "#30b0c7", "#a2845e", "#ff2d55")
#: One data series on a plot canvas.
DATA_COLOR = SERIES_COLORS[0]

#: What View > Appearance offers, and the setting it is remembered under.
APPEARANCES = ("system", "light", "dark")
_SETTING = "appearance/mode"

UI_FONT_FAMILY = "Segoe UI"
MONO_FONT_FAMILY = "Consolas"

_state = {"mode": "light", "appearance": "system"}


class _Notifier(QObject):
    """Tells widgets that keep their own colours that the appearance changed."""

    changed = Signal(str)


_notifier: Optional[_Notifier] = None


def notifier() -> _Notifier:
    """The object whose ``changed(mode)`` signal fires after a change."""
    global _notifier
    if _notifier is None:
        _notifier = _Notifier()
    return _notifier


def mode() -> str:
    """The appearance in force: ``'light'`` or ``'dark'``."""
    return _state["mode"]


def is_dark() -> bool:
    return _state["mode"] == "dark"


def appearance() -> str:
    """The chosen appearance: ``'system'``, ``'light'`` or ``'dark'``."""
    return _state["appearance"]


def color(key: str) -> str:
    """The palette entry ``key`` in the appearance now in force.

    For colour written into rich text, read when the text is set: such text
    keeps its colour until it is next set, unlike a styled widget.
    """
    return PALETTE.get(key, key)


def ai_colors() -> Tuple[str, str, str, str]:
    """The agent's gradient: the assistant's own colours when it has them,
    otherwise blue, purple, pink and orange in this appearance."""
    own = _state.get("ai_colors")
    if own:
        return own
    p = PALETTE
    return (p["vivid_blue"], p["vivid_purple"], p["vivid_pink"], p["vivid_orange"])


def set_ai_colors(colors) -> None:
    """Give the agent glow and orb an assistant's four colours; empty for the default."""
    colors = tuple(str(c) for c in (colors or ()))
    _state["ai_colors"] = (colors * 4)[:4] if colors else None


def set_tone(widget, tone: Optional[str]) -> None:
    """Colour a label by what it says - ``'hint'``, ``'ok'``, ``'warn'``,
    ``'error'`` - through the stylesheet, so it follows the appearance.

    A colour written into the widget's own style sheet stays the colour it was
    when written; a tone is restyled whenever the appearance changes.
    """
    widget.setProperty("tone", tone or "")
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _saved_appearance() -> str:
    try:
        value = str(QSettings("PyHydroGeophysX", "Studio").value(_SETTING, "system"))
    except Exception:  # noqa: BLE001 - a missing setting is the default
        value = "system"
    return value if value in APPEARANCES else "system"


def _system_mode(app) -> str:
    """Light or dark, as the operating system is set."""
    try:
        from PySide6.QtCore import Qt

        if app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
            return "dark"
    except Exception:  # noqa: BLE001 - an older Qt cannot say: light
        pass
    return "light"


def _logo_path() -> Optional[Path]:
    """Locate logo.png at the repository root (parents[2]) or in the package.

    None when neither has it: the studio then runs without a window icon or a
    header logo, and says nothing about it.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "logo.png",          # repo root
        here.parents[1] / "logo.png",          # the package's own copy (wheels)
    ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def window_icon() -> QIcon:
    """Return the application/window icon (logo.png) or an empty icon."""
    path = _logo_path()
    return QIcon(str(path)) if path else QIcon()


def qtawesome_available() -> bool:
    try:
        import qtawesome  # noqa: F401

        return True
    except Exception:
        return False


def icon(name: str, color: Optional[str] = None) -> QIcon:
    """Return a qtawesome icon, or an empty QIcon if qtawesome/name is missing.

    ``name`` uses qtawesome conventions, e.g. ``fa5s.folder-open`` or ``mdi.cog``.
    Icons default to the accent blue, which reads on both appearances, so an
    icon drawn before the appearance changes does not have to be redrawn.
    """
    try:
        import qtawesome as qta

        result = qta.icon(name, color=color or PALETTE["primary"])
        # qtawesome can return a non-QIcon if it was initialized before the
        # QApplication; never hand that to ``setIcon``.
        return result if isinstance(result, QIcon) else QIcon()
    except Exception:
        return QIcon()


#: Small glyphs the stylesheet draws with: a check for a ticked box, and the
#: chevrons of combo and spin boxes. A style sheet that restyles those controls
#: loses the platform's own marks, so these replace them.
_GLYPHS = {
    "check": ('<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" '
              'viewBox="0 0 16 16"><path d="M4 8.4l2.6 2.6 5.4-6" fill="none" '
              'stroke="{color}" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round"/></svg>'),
    "down": ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" '
             'viewBox="0 0 10 10"><path d="M2 3.6l3 3 3-3" fill="none" '
             'stroke="{color}" stroke-width="1.5" stroke-linecap="round" '
             'stroke-linejoin="round"/></svg>'),
    "up": ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" '
           'viewBox="0 0 10 10"><path d="M2 6.4l3-3 3 3" fill="none" '
           'stroke="{color}" stroke-width="1.5" stroke-linecap="round" '
           'stroke-linejoin="round"/></svg>'),
}


def _glyph(name: str, color: str) -> str:
    """Path of the glyph ``name`` drawn in ``color``, written once to a cache folder.

    Returns "" when it cannot be written; the stylesheet then simply goes
    without the mark.
    """
    import tempfile

    folder = Path(tempfile.gettempdir()) / "pyhydrogeophysx_theme"
    path = folder / f"{name}_{color.lstrip('#')}.svg"
    try:
        if not path.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            path.write_text(_GLYPHS[name].format(color=color), encoding="utf-8")
    except OSError:
        return ""
    return path.as_posix()


def _url(path: str) -> str:
    return f'url("{path}")' if path else "none"


def build_qss() -> str:
    """Return the full QSS stylesheet for the palette in force."""
    p = PALETTE
    check = _url(_glyph("check", "#ffffff"))
    down = _url(_glyph("down", p["muted"]))
    up = _url(_glyph("up", p["muted"]))
    return f"""
    QWidget {{
        background-color: {p['bg']};
        color: {p['text']};
        font-family: "{UI_FONT_FAMILY}", "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
    }}
    QMainWindow, QDialog {{ background-color: {p['bg']}; }}

    /* Welcome page: quiet surfaces, readable hierarchy and one main action.
       Blue is what can be clicked; the agent's colours are its mark alone. */
    QWidget#StudioHome QWidget {{ font-family: "{UI_FONT_FAMILY}"; }}
    QWidget#HomeContent, QScrollArea#HomeScroll {{ background: {p['bg']}; border: none; }}
    QFrame#HomeHero {{ background: {p['card']}; border: 1px solid {p['border']}; border-radius: 20px; }}
    QFrame#HomeTaskCard, QFrame#HomeWorkspace {{
        background: {p['card']}; border: 1px solid {p['border']}; border-radius: 14px;
    }}
    QFrame#HomeHero QWidget, QFrame#HomeTaskCard QWidget, QFrame#HomeWorkspace QWidget {{ background: transparent; }}
    QLabel#HomeBrand {{ font-size: 16px; font-weight: 600; }}
    QLabel#HomeEyebrow {{ color: {p['muted']}; font-size: 11px; font-weight: 500; }}
    QLabel#HomeAccent {{ color: {p['muted']}; font-size: 11px; font-weight: 600; }}
    QLabel#HomeAIStages {{ color: {p['muted']}; font-size: 12px; }}
    QLabel#HomeTitle {{ color: {p['text']}; font-size: 34px; font-weight: 600; }}
    QLabel#HomeTitle[compact="true"] {{ font-size: 28px; }}
    QLabel#HomeDescription {{ color: {p['muted']}; font-size: 13px; }}
    QLabel#HomeSection {{ font-size: 16px; font-weight: 600; }}
    QLabel#HomeCardTitle {{ font-size: 16px; font-weight: 600; }}
    QLabel#HomeDirection {{ color: {p['muted']}; font-size: 11px; font-weight: 600; }}
    QWidget#StudioHome QPushButton {{ font-size: 13px; padding: 8px 14px; }}
    QWidget#StudioHome QPushButton[primary="true"] {{
        background: {p['primary']}; color: {p['on_primary']}; border-color: {p['primary']};
    }}
    QWidget#StudioHome QPushButton[primary="true"]:hover {{ background: {p['accent']}; }}
    QWidget#StudioHome QPushButton[homeRole="chip"] {{
        background: transparent; color: {p['primary']}; border: 1px solid {p['border']};
        border-radius: 13px; padding: 4px 10px; font-size: 12px;
    }}
    QWidget#StudioHome QPushButton[homeRole="quiet"] {{
        background: transparent; color: {p['primary']}; border: none; padding: 5px 8px;
    }}
    QWidget#StudioHome QPushButton[homeRole="chip"]:hover,
    QWidget#StudioHome QPushButton[homeRole="quiet"]:hover {{ background: {p['hover']}; }}
    QToolButton#HomeDetails {{ background: transparent; color: {p['muted']}; border: none; padding: 4px 0; }}
    QToolButton#HomeDetails:hover {{ color: {p['primary']}; }}
    QTextEdit#HomeSummary {{ font-family: "{MONO_FONT_FAMILY}"; font-size: 12px; border-color: {p['border']}; }}

    /* Cards / group boxes */
    QGroupBox {{
        background-color: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        margin-top: 16px;
        padding: 12px 10px 8px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 12px;
        padding: 1px 4px;
        color: {p['text']};
        background-color: transparent;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {p['control']};
        color: {p['text']};
        border: 1px solid {p['border_blue']};
        border-radius: 8px;
        padding: 6px 14px;
        font-weight: 500;
    }}
    QPushButton:hover {{ background-color: {p['control_hover']}; }}
    QPushButton:pressed {{ background-color: {p['control_pressed']}; }}
    QPushButton:checked {{ background-color: {p['select_bg']}; color: {p['select_text']}; border-color: {p['select_bg']}; }}
    QPushButton:disabled {{ color: {p['disabled_text']}; border-color: {p['border']}; background-color: {p['bg']}; }}
    QPushButton[primary="true"] {{
        background-color: {p['primary']};
        color: {p['on_primary']};
        border: 1px solid {p['primary']};
    }}
    QPushButton[primary="true"]:hover {{ background-color: {p['accent']}; border-color: {p['accent']}; }}
    QPushButton[primary="true"]:pressed {{ background-color: {p['primary_dark']}; }}
    QPushButton[primary="true"]:disabled {{ background-color: {p['track']}; border-color: {p['track']}; color: {p['disabled_text']}; }}
    /* A segmented choice: checkable buttons joined side by side, the chosen one
       filled and bold, the others plain - as clear as a choice can be at a
       glance, where a radio button's dot was not. */
    QPushButton[segment] {{ border-radius: 0px; padding: 5px 16px; color: {p['muted']}; }}
    QPushButton[segment="first"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
    QPushButton[segment="last"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-left: none; }}
    QPushButton[segment]:checked {{ background-color: {p['select_bg']}; color: {p['select_text']}; border: 1px solid {p['primary']}; font-weight: 600; }}

    /* Inputs */
    QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {p['field']};
        border: 1px solid {p['border_blue']};
        border-radius: 7px;
        padding: 4px 7px;
        selection-background-color: {p['primary']};
        selection-color: {p['on_primary']};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {p['primary']}; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; border: none; width: 20px; }}
    QComboBox::down-arrow {{ image: {down}; width: 10px; height: 10px; }}
    QAbstractSpinBox {{ padding-right: 18px; }}
    QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 18px; border: none; background: transparent; }}
    QAbstractSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border: none; background: transparent; }}
    QAbstractSpinBox::up-arrow {{ image: {up}; width: 9px; height: 9px; }}
    QAbstractSpinBox::down-arrow {{ image: {down}; width: 9px; height: 9px; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {p['hover']}; border-radius: 4px; }}
    QComboBox QAbstractItemView {{
        background-color: {p['menu_bg']};
        border: 1px solid {p['border_blue']};
        selection-background-color: {p['primary']};
        selection-color: {p['on_primary']};
        outline: 0;
    }}
    QComboBox QAbstractItemView::item:disabled {{ color: {p['disabled_text']}; }}
    QTextBrowser {{ background-color: {p['card']}; border: 1px solid {p['border']}; border-radius: 8px; }}

    QCheckBox, QRadioButton {{ spacing: 6px; background: transparent; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {p['border_blue']}; border-radius: 4px; background: {p['field']}; }}
    QCheckBox::indicator:checked {{ background-color: {p['primary']}; border-color: {p['primary']}; image: {check}; }}
    QCheckBox::indicator:disabled {{ background: {p['track']}; border-color: {p['border']}; }}

    /* Sliders: a thin track filled in the accent, a round white knob */
    QSlider::groove:horizontal {{ height: 4px; background: {p['track']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {p['primary']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: #ffffff; border: 1px solid {p['border_blue']}; width: 16px; height: 16px; margin: -7px 0; border-radius: 8px; }}
    QSlider::groove:vertical {{ width: 4px; background: {p['track']}; border-radius: 2px; }}
    QSlider::add-page:vertical {{ background: {p['primary']}; border-radius: 2px; }}
    QSlider::handle:vertical {{ background: #ffffff; border: 1px solid {p['border_blue']}; width: 16px; height: 16px; margin: 0 -7px; border-radius: 8px; }}

    QLabel {{ background: transparent; }}
    /* Plain containers inside a card or a tab page show the card, not the
       window behind it. (.QWidget is that class exactly, not its subclasses.) */
    QGroupBox .QWidget, QTabWidget .QWidget, QFrame#ConfirmCard .QWidget {{ background: transparent; }}
    QLabel[tone="hint"] {{ color: {p['muted']}; font-size: 8pt; }}
    QLabel[tone="ok"] {{ color: {p['green']}; font-size: 8pt; }}
    QLabel[tone="warn"] {{ color: {p['amber']}; font-size: 8pt; }}
    QLabel[tone="error"] {{ color: {p['red']}; font-size: 8pt; }}
    QLabel[tone="muted"] {{ color: {p['muted']}; }}
    QLabel[tone="mono"] {{ color: {p['muted']}; font-family: "{MONO_FONT_FAMILY}", monospace; font-size: 9pt; }}

    /* Tabs: a segmented control */
    QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 10px; background: {p['card']}; top: -1px; }}
    QTabBar {{ background: transparent; }}
    QTabBar::tab {{
        background: transparent;
        color: {p['muted']};
        padding: 6px 14px;
        margin: 3px 2px;
        border: none;
        border-radius: 7px;
        font-weight: 500;
    }}
    QTabBar::tab:selected {{ background: {p['tab_selected']}; color: {p['text']}; }}
    QTabBar::tab:hover:!selected {{ background: {p['hover']}; color: {p['text']}; }}

    /* Docks */
    QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
    QDockWidget::title {{
        background-color: {p['bg']};
        color: {p['muted']};
        padding: 6px 10px;
        font-weight: 600;
        border: none;
    }}

    /* Tree / lists / tables */
    QTreeWidget, QTreeView, QListView, QTableView {{
        background-color: {p['card']};
        alternate-background-color: {p['bg']};
        border: 1px solid {p['border']};
        border-radius: 10px;
        outline: 0;
        selection-background-color: {p['select_bg']};
        selection-color: {p['select_text']};
    }}
    QTreeWidget::item, QListView::item {{ padding: 5px 4px; border-radius: 6px; }}
    QTreeWidget::item:selected, QListView::item:selected {{ background-color: {p['select_bg']}; color: {p['select_text']}; }}
    QTreeWidget::item:hover, QListView::item:hover {{ background-color: {p['hover']}; }}
    QHeaderView::section {{ background-color: {p['card']}; color: {p['muted']}; padding: 5px; border: none; border-bottom: 1px solid {p['border']}; font-weight: 600; }}
    QTableCornerButton::section {{ background-color: {p['card']}; border: none; }}

    /* Toolbar / menus */
    QToolBar {{ background-color: {p['card']}; border: none; border-bottom: 1px solid {p['border']}; spacing: 4px; padding: 4px; }}
    QToolButton {{ background: transparent; color: {p['text']}; border-radius: 6px; padding: 5px 8px; font-weight: 500; }}
    QToolBar#main_toolbar {{ spacing: 2px; padding: 2px 4px; }}
    QToolBar#main_toolbar QToolButton {{ padding: 3px 8px; }}
    QToolButton:hover {{ background-color: {p['hover']}; }}
    QToolButton:pressed, QToolButton:checked {{ background-color: {p['select_bg']}; color: {p['select_text']}; }}
    QMenuBar {{ background-color: {p['card']}; border: none; }}
    QMenuBar::item {{ background: transparent; padding: 3px 9px; border-radius: 5px; }}
    QMenuBar::item:selected {{ background-color: {p['hover']}; }}
    QMenu {{ background-color: {p['menu_bg']}; border: 1px solid {p['border_blue']}; border-radius: 6px; padding: 5px; }}
    QMenu::item {{ padding: 5px 24px 5px 22px; border-radius: 5px; }}
    QMenu::item:selected {{ background-color: {p['primary']}; color: {p['on_primary']}; }}
    QMenu::item:disabled {{ color: {p['disabled_text']}; }}
    QMenu::separator {{ height: 1px; background: {p['border']}; margin: 4px 8px; }}

    /* Status bar */
    QStatusBar {{ background-color: {p['bg']}; color: {p['muted']}; border-top: 1px solid {p['border']}; }}
    QStatusBar QLabel {{ background: transparent; color: {p['muted']}; }}
    /* Something is still running: full-strength text, not a colour, so it
       reads at a glance without borrowing the meaning of green or orange. */
    QStatusBar QLabel[tone="busy"] {{ color: {p['text']}; font-weight: 600; }}

    /* Progress */
    QProgressBar {{ background-color: {p['track']}; border: none; border-radius: 4px; height: 12px; text-align: center; color: {p['text']}; }}
    QProgressBar::chunk {{ background-color: {p['primary']}; border-radius: 4px; }}
    QProgressBar#agentProgress {{ height: 4px; max-height: 4px; min-height: 4px; border-radius: 2px; }}
    QProgressBar#agentProgress::chunk {{ border-radius: 2px; }}

    /* Scrollbars: thin, as on a Mac */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p['scroll']}; border-radius: 3px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {p['tertiary']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {p['scroll']}; border-radius: 3px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {p['tertiary']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{ background-color: {p['tooltip_bg']}; color: {p['tooltip_text']}; border: 1px solid {p['border_blue']}; padding: 5px 8px; border-radius: 6px; }}
    QScrollArea {{ border: none; background: transparent; }}
    QSplitter::handle {{ background-color: {p['bg']}; }}

    /* The log */
    QTextEdit#logPanel {{
        background-color: {p['card']};
        color: {p['text']};
        border: none;
        border-radius: 0;
        font-family: "{MONO_FONT_FAMILY}", "Courier New", monospace;
        font-size: 12px;
    }}

    /* The project map's panels */
    QWidget#MapLayersPanel, QWidget#ProjectMapPanel {{ background: {p['card']}; border: 1px solid {p['border']}; border-radius: 10px; }}
    QLabel#MapSectionTitle {{ color: {p['muted']}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}

    /* The AQUAH assistant */
    QFrame#ConfirmCard {{ background: {p['select_bg']}; border: 1px solid {p['border_blue']}; border-radius: 10px; }}
    QLabel#aquahThinking {{ color: {p['muted']}; font-style: italic; }}

    /* The agent at work (widgets/ai_presence.py and the strip over a module) */
    QWidget#agentTimelineBody {{ background: {p['bg']}; }}
    QLabel#agentGoal {{ background: {p['card']}; border: 1px solid {p['border']}; border-radius: 12px; padding: 12px 14px; color: {p['text']}; }}
    QLabel#agentGoalTag {{ color: {p['tertiary']}; font-size: 8pt; font-weight: 600; letter-spacing: 1px; }}
    QLabel#agentTitle {{ color: {p['text']}; font-weight: 600; }}
    QLabel#agentChip {{ background: {p['tab_selected']}; color: {p['muted']}; border-radius: 9px; padding: 1px 8px; font-size: 8pt; font-weight: 500; }}
    QLabel#agentClock {{ color: {p['tertiary']}; font-family: "{MONO_FONT_FAMILY}", monospace; font-size: 9pt; }}
    QLabel#agentBigClock {{ color: {p['text']}; font-size: 18pt; font-weight: 300; }}
    QLabel#agentCounter {{ color: {p['muted']}; font-size: 8pt; }}
    QLabel#agentReason {{ color: {p['muted']}; }}
    QLabel#agentDetail {{ color: {p['muted']}; }}
    QLabel#agentSummary {{ color: {p['text']}; }}
    QLabel#agentSummary[failed="true"] {{ color: {p['red']}; }}
    QLabel#agentQuestion {{ color: {p['text']}; }}
    QLabel#stepThumb {{ border: 1px solid {p['border']}; border-radius: 6px; background: {p['canvas']}; padding: 2px; }}
    QLabel#filmThumb {{ border: 1px solid {p['border']}; border-radius: 6px; background: {p['canvas']}; padding: 2px; }}
    QLabel#filmThumb[selected="true"] {{ border: 2px solid {p['primary']}; padding: 1px; }}
    QLabel#agentFinishTitle {{ color: {p['text']}; font-size: 13pt; font-weight: 600; }}
    QLabel#agentThought {{ color: {p['muted']}; font-style: italic; }}
    QFrame#noteCard {{ background: {p['tab_selected']}; border: none; border-radius: 12px; }}
    QFrame#noteCard QLabel {{ background: transparent; }}
    QFrame#steerBar {{ background: {p['bg']}; border: none; border-top: 1px solid {p['border']}; }}
    QFrame#replayBar {{ background: {p['material']}; border: none; border-bottom: 1px solid {p['border']}; }}
    QFrame#replayBar QLabel {{ background: transparent; }}
    QScrollArea#agentFilmstrip, QWidget#agentFilmstrip {{ background: transparent; border: none; }}
    QWidget#agentLive {{ background: {p['bg']}; }}
    QSplitter#agentLiveSplit::handle {{ background: transparent; }}
    QPlainTextEdit#agentCode, QPlainTextEdit#runCode {{
        background: {p['field']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: 6px;
        font-family: "{MONO_FONT_FAMILY}", "DejaVu Sans Mono", monospace; font-size: 11px;
    }}
    QFrame#runActivity {{ background: {p['material']}; border: none; border-bottom: 1px solid {p['border']}; }}
    QFrame#runActivity QLabel {{ color: {p['text']}; background: transparent; }}
    QFrame#runActivity QLabel#runActivityNote {{ color: {p['muted']}; font-size: 8pt; }}
    #runStrip, #runOpens, #runChoices, #stepFigures, #thinkingChoices {{ background: transparent; }}
    """


def _qpalette() -> QPalette:
    p = PALETTE
    pal = QPalette()
    for role, key in ((QPalette.Window, "bg"), (QPalette.Base, "field"),
                      (QPalette.AlternateBase, "bg"), (QPalette.Text, "text"),
                      (QPalette.WindowText, "text"), (QPalette.Button, "control"),
                      (QPalette.ButtonText, "text"), (QPalette.Highlight, "primary"),
                      (QPalette.HighlightedText, "on_primary"),
                      (QPalette.ToolTipBase, "tooltip_bg"),
                      (QPalette.ToolTipText, "tooltip_text"),
                      (QPalette.PlaceholderText, "tertiary"),
                      (QPalette.Link, "primary"), (QPalette.Mid, "border_blue"),
                      (QPalette.Midlight, "border"), (QPalette.Light, "card"),
                      (QPalette.Dark, "scroll")):
        pal.setColor(role, QColor(p[key]))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(p["disabled_text"]))
    return pal


def apply_theme(app, appearance: Optional[str] = None) -> str:
    """Apply an appearance to the whole application, before or after widgets exist.

    Parameters
    ----------
    app : QApplication
    appearance : str, optional
        ``'system'``, ``'light'`` or ``'dark'``. Omitted, the remembered choice
        is used (System by default).

    Returns
    -------
    str
        The mode applied, ``'light'`` or ``'dark'``.
    """
    choice = appearance if appearance in APPEARANCES else _saved_appearance()
    resolved = _system_mode(app) if choice == "system" else choice
    _state["appearance"], _state["mode"] = choice, resolved
    PALETTE.clear()
    PALETTE.update(DARK if resolved == "dark" else LIGHT)

    app.setStyle("Fusion")
    app.setPalette(_qpalette())
    app.setFont(QFont(UI_FONT_FAMILY, 10))
    app.setStyleSheet(build_qss())

    # pyqtgraph: a light canvas in either appearance (see the module notes).
    try:
        import pyqtgraph as pg

        pg.setConfigOption("background", PALETTE["canvas"])
        pg.setConfigOption("foreground", PALETTE["canvas_text"])
        pg.setConfigOption("antialias", True)
    except Exception:
        pass
    apply_matplotlib_style()
    _restyle_open_plots(app)
    _follow_system(app)
    try:
        for widget in app.allWidgets():
            widget.update()
    except Exception:  # noqa: BLE001 - a repaint is cosmetic
        pass
    notifier().changed.emit(resolved)
    return resolved


def set_appearance(app, appearance: str) -> str:
    """Remember ``appearance`` and apply it now; returns the mode applied."""
    if appearance not in APPEARANCES:
        appearance = "system"
    try:
        QSettings("PyHydroGeophysX", "Studio").setValue(_SETTING, appearance)
    except Exception:  # noqa: BLE001 - not remembered, still applied
        pass
    return apply_theme(app, appearance)


def _follow_system(app) -> None:
    """While the choice is System, re-apply when the operating system switches."""
    if _state.get("following"):
        return
    try:
        app.styleHints().colorSchemeChanged.connect(
            lambda _scheme: apply_theme(app, "system") if appearance() == "system" else None)
        _state["following"] = True
    except Exception:  # noqa: BLE001 - an older Qt cannot tell us
        pass


def _restyle_open_plots(app) -> None:
    """Give plots already on screen the canvas of the appearance now in force."""
    try:
        import pyqtgraph as pg
    except Exception:  # noqa: BLE001
        return
    try:
        for widget in app.allWidgets():
            if isinstance(widget, pg.GraphicsView):
                widget.setBackground(PALETTE["canvas"])
    except Exception:  # noqa: BLE001 - a canvas tint is cosmetic
        pass


def apply_matplotlib_style() -> None:
    """Draw the studio's Matplotlib figures on the canvas colour, in the
    system colours.

    Only how a figure looks on screen changes: a saved figure keeps a white
    ground, and every plot that names its own colours - colormaps above all -
    keeps them.
    """
    try:
        import matplotlib as mpl
        from cycler import cycler
    except Exception:  # noqa: BLE001 - the studio runs without Matplotlib
        return
    ink, soft = PALETTE["canvas_text"], "#6e6e73"
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["canvas"],
        "axes.facecolor": PALETTE["canvas"],
        "savefig.facecolor": "white",
        "axes.edgecolor": "#c7c7cc",
        "axes.labelcolor": ink,
        "text.color": ink,
        "xtick.color": soft,
        "ytick.color": soft,
        "grid.color": "#e5e5ea",
        "legend.edgecolor": "#e5e5ea",
        "axes.prop_cycle": cycler(color=list(SERIES_COLORS)),
    })
