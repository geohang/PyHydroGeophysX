"""Central light theme for the PyHydroGeophysX workbench (brand-matched).

One module owns the whole look so it stays consistent: a brand color palette, a
hand-crafted QSS stylesheet, pyqtgraph plot colors, a ``qtawesome`` icon helper
(with a graceful no-op fallback when qtawesome is absent), and the window icon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor, QFont, QIcon, QPalette

# Brand palette (matches the Streamlit app identity).
PALETTE = {
    "primary": "#0f4c75",
    "primary_dark": "#0a3a5c",
    "accent": "#3d6cb9",
    "green": "#2d9c5b",
    "red": "#d64545",
    "amber": "#e0a800",
    "bg": "#f5f7fb",
    "card": "#ffffff",
    "text": "#1b262c",
    "muted": "#5a6a7a",
    "border": "#e1e5ec",
    "border_blue": "#c8dce8",
    "select_bg": "#e2f0ff",
    "select_text": "#174ea6",
    "hover": "#e8f4f8",
}

UI_FONT_FAMILY = "Segoe UI"
MONO_FONT_FAMILY = "Consolas"


def _logo_path() -> Optional[Path]:
    """Locate logo.png at the repository root (parents[2]) or package data."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "logo.png",          # repo root
        here.parents[1] / "data" / "logo.png",  # packaged data
        here.parents[1] / "logo.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
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
    """
    try:
        import qtawesome as qta

        result = qta.icon(name, color=color or PALETTE["primary"])
        # qtawesome can return a non-QIcon if it was initialized before the
        # QApplication; never hand that to ``setIcon``.
        return result if isinstance(result, QIcon) else QIcon()
    except Exception:
        return QIcon()


def build_qss() -> str:
    """Return the full light QSS stylesheet using the brand palette."""
    p = PALETTE
    return f"""
    QWidget {{
        background-color: {p['bg']};
        color: {p['text']};
        font-family: "{UI_FONT_FAMILY}", "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
    }}
    QMainWindow, QDialog {{ background-color: {p['bg']}; }}

    /* Header banner */
    QFrame#HeaderBar {{
        background-color: {p['primary']};
        border: none;
    }}
    QLabel#HeaderTitle {{ color: #ffffff; font-size: 16pt; font-weight: 700; background: transparent; }}
    QLabel#HeaderSubtitle {{ color: #cfe0f0; font-size: 9pt; background: transparent; }}

    /* Cards / group boxes */
    QGroupBox {{
        background-color: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        margin-top: 14px;
        padding: 10px 10px 8px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 1px 6px;
        color: {p['primary']};
        background-color: {p['card']};
    }}

    /* Buttons */
    QPushButton {{
        background-color: {p['card']};
        color: {p['primary']};
        border: 1px solid {p['border_blue']};
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: {p['hover']}; border-color: {p['accent']}; }}
    QPushButton:pressed {{ background-color: {p['select_bg']}; }}
    QPushButton:disabled {{ color: #9bb0c2; border-color: {p['border']}; background-color: #f0f2f7; }}
    QPushButton[primary="true"] {{
        background-color: {p['primary']};
        color: #ffffff;
        border: 1px solid {p['primary']};
    }}
    QPushButton[primary="true"]:hover {{ background-color: {p['accent']}; border-color: {p['accent']}; }}
    QPushButton[primary="true"]:pressed {{ background-color: {p['primary_dark']}; }}
    QPushButton[primary="true"]:disabled {{ background-color: #9bb0c2; border-color: #9bb0c2; color: #eef2f6; }}

    /* Inputs */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {p['card']};
        border: 1px solid {p['border_blue']};
        border-radius: 6px;
        padding: 4px 6px;
        selection-background-color: {p['accent']};
        selection-color: #ffffff;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {p['accent']}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background-color: {p['card']};
        border: 1px solid {p['border_blue']};
        selection-background-color: {p['select_bg']};
        selection-color: {p['select_text']};
        outline: 0;
    }}

    QCheckBox {{ spacing: 6px; background: transparent; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {p['border_blue']}; border-radius: 4px; background: {p['card']}; }}
    QCheckBox::indicator:checked {{ background-color: {p['accent']}; border-color: {p['accent']}; }}

    QLabel {{ background: transparent; }}

    /* Tabs */
    QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: 8px; background: {p['card']}; top: -1px; }}
    QTabBar::tab {{
        background: transparent;
        color: {p['muted']};
        padding: 7px 14px;
        margin-right: 2px;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 600;
    }}
    QTabBar::tab:selected {{ color: {p['primary']}; border-bottom: 2px solid {p['primary']}; }}
    QTabBar::tab:hover {{ color: {p['accent']}; }}

    /* Docks */
    QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
    QDockWidget::title {{
        background-color: {p['hover']};
        color: {p['primary']};
        padding: 6px 10px;
        font-weight: 700;
        border-bottom: 1px solid {p['border']};
    }}

    /* Tree / headers */
    QTreeWidget, QTreeView, QListView, QTableView {{
        background-color: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 8px;
        outline: 0;
    }}
    QTreeWidget::item {{ padding: 5px 4px; border-radius: 4px; }}
    QTreeWidget::item:selected {{ background-color: {p['select_bg']}; color: {p['select_text']}; }}
    QTreeWidget::item:hover {{ background-color: {p['hover']}; }}
    QHeaderView::section {{ background-color: {p['hover']}; color: {p['primary']}; padding: 5px; border: none; font-weight: 600; }}

    /* Toolbar / menus */
    QToolBar {{ background-color: {p['card']}; border-bottom: 1px solid {p['border']}; spacing: 4px; padding: 4px; }}
    QToolButton {{ background: transparent; color: {p['primary']}; border-radius: 6px; padding: 5px 8px; font-weight: 600; }}
    QToolButton:hover {{ background-color: {p['hover']}; }}
    QToolButton:pressed, QToolButton:checked {{ background-color: {p['select_bg']}; }}
    QMenuBar {{ background-color: {p['card']}; border-bottom: 1px solid {p['border']}; }}
    QMenuBar::item {{ background: transparent; padding: 6px 10px; }}
    QMenuBar::item:selected {{ background-color: {p['hover']}; border-radius: 4px; }}
    QMenu {{ background-color: {p['card']}; border: 1px solid {p['border_blue']}; padding: 4px; }}
    QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background-color: {p['select_bg']}; color: {p['select_text']}; }}

    /* Status bar */
    QStatusBar {{ background-color: {p['hover']}; color: {p['text']}; border-top: 1px solid {p['border']}; }}
    QStatusBar QLabel {{ background: transparent; }}

    /* Progress */
    QProgressBar {{ background-color: {p['border']}; border: none; border-radius: 6px; height: 12px; text-align: center; color: {p['text']}; }}
    QProgressBar::chunk {{ background-color: {p['accent']}; border-radius: 6px; }}

    /* Scrollbars */
    QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {p['border_blue']}; border-radius: 5px; min-height: 24px; }}
    QScrollBar::handle:vertical:hover {{ background: {p['accent']}; }}
    QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
    QScrollBar::handle:horizontal {{ background: {p['border_blue']}; border-radius: 5px; min-width: 24px; }}
    QScrollBar::handle:horizontal:hover {{ background: {p['accent']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QToolTip {{ background-color: {p['text']}; color: #ffffff; border: none; padding: 5px 8px; border-radius: 4px; }}
    QScrollArea {{ border: none; background: transparent; }}
    """


def apply_theme(app) -> None:
    """Apply the light theme to the whole application."""
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(PALETTE["bg"]))
    pal.setColor(QPalette.Base, QColor(PALETTE["card"]))
    pal.setColor(QPalette.AlternateBase, QColor(PALETTE["bg"]))
    pal.setColor(QPalette.Text, QColor(PALETTE["text"]))
    pal.setColor(QPalette.WindowText, QColor(PALETTE["text"]))
    pal.setColor(QPalette.Button, QColor(PALETTE["card"]))
    pal.setColor(QPalette.ButtonText, QColor(PALETTE["primary"]))
    pal.setColor(QPalette.Highlight, QColor(PALETTE["accent"]))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor(PALETTE["text"]))
    pal.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    app.setPalette(pal)

    app.setFont(QFont(UI_FONT_FAMILY, 10))
    app.setStyleSheet(build_qss())

    # pyqtgraph plot colors (light background, dark axes/labels).
    try:
        import pyqtgraph as pg

        pg.setConfigOption("background", PALETTE["card"])
        pg.setConfigOption("foreground", PALETTE["text"])
        pg.setConfigOption("antialias", True)
    except Exception:
        pass
