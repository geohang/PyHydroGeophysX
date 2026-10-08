"""Naming things the user keeps: a Project when it is made, runs when they are saved.

Results used to go to a fallback folder nobody chose, and every run was listed
as ``Run`` plus four random characters, so after a few surveys nobody could
tell which run belonged to which. These dialogs are where a name is asked for:
a Project's name and place when one is created, or before data first goes into
the fallback folder, and the runs' names when they are saved to the Project.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core import mesh_serialization
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.results_store import (
    INDEX_FILENAME,
    is_placeholder_label,
    run_title,
)

__all__ = [
    "NewProjectDialog",
    "SaveRunsDialog",
    "confirm_project_for_data",
    "project_name_problem",
]

#: Characters Windows refuses in a folder name, and the names it reserves.
_BAD_CHARACTERS = '<>:"/\\|?*'
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL",
                   *(f"COM{n}" for n in range(1, 10)), *(f"LPT{n}" for n in range(1, 10))}


def project_name_problem(name: str) -> str:
    """Why ``name`` cannot name a Project folder, or ``""`` when it can.

    The rules are Windows' (the strictest of the three systems), so a Project
    made on a Mac or Linux machine can still be copied to a Windows one.
    """
    text = str(name or "").strip()
    if not text:
        return "Type a name for the project."
    bad = [ch for ch in _BAD_CHARACTERS if ch in text]
    if bad:
        return "A project name cannot contain " + "  ".join(bad)
    if any(ord(ch) < 32 for ch in text):
        return "A project name cannot contain control characters."
    if text.endswith("."):
        return "A project name cannot end with a full stop."
    if text.split(".")[0].upper() in _RESERVED_NAMES:
        return f"Windows reserves the name {text}. Choose another."
    if len(text) > 100:
        return "Use a name of 100 characters or fewer."
    return ""


def _folder_problem(path: Path) -> str:
    """Why a new Project cannot be made at ``path``, or ``""``.

    A new Project goes into a new or empty folder. One already holding a
    Project is opened with Open Project instead, so two surveys never end up
    in one history by accident.
    """
    if not path.exists():
        return ""
    if not path.is_dir():
        return "A file with this name is already there. Choose another name."
    try:
        occupied = any(path.iterdir())
    except OSError as exc:
        return f"This folder cannot be read: {exc.strerror or exc}"
    if not occupied:
        return ""
    if (path / INDEX_FILENAME).exists() or (path / "runs").is_dir():
        return "A project with this name is already there. Choose another name, or use Open Project."
    return "A folder with this name is already there and is not empty. Choose another name."


class NewProjectDialog(QDialog):
    """A Project's name and the folder it goes in; creates ``<location>/<name>``.

    With ``offer_default`` it is the question asked before data first goes into
    the fallback folder, and also offers to keep using that folder. ``exec()``
    then returns :attr:`USE_DEFAULT` for that answer, besides ``Accepted`` and
    ``Rejected``.
    """

    USE_DEFAULT = 2

    def __init__(self, parent: Optional[QWidget], location: Path, *, name: str = "",
                 title: str = "New Project", message: str = "",
                 offer_default: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        if message:
            intro = QLabel(message)
            intro.setWordWrap(True)
            layout.addWidget(intro)

        form = QFormLayout()
        self._name = QLineEdit(name)
        self._name.setPlaceholderText("For example: Site A 2026")
        form.addRow("Name", self._name)
        place = QHBoxLayout()
        self._location = QLineEdit(str(location))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        place.addWidget(self._location, stretch=1)
        place.addWidget(browse)
        form.addRow("Location", place)
        layout.addLayout(form)

        # Where the folder will be, or why it cannot be made there yet.
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self._note)

        self._ask_again = QCheckBox("Don't ask again")
        self._ask_again.setToolTip("Keep using the default folder without asking. "
                                   "File › New Project… still makes a project.")
        self._ask_again.setVisible(offer_default)
        layout.addWidget(self._ask_again)

        buttons = QDialogButtonBox()
        if offer_default:
            keep = buttons.addButton("Use Default Folder", QDialogButtonBox.ActionRole)
            keep.clicked.connect(lambda: self.done(self.USE_DEFAULT))
        buttons.addButton(QDialogButtonBox.Cancel)
        self._create = buttons.addButton("Create Project", QDialogButtonBox.AcceptRole)
        self._create.setProperty("primary", True)
        self._create.setDefault(True)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._name.textChanged.connect(self._check)
        self._location.textChanged.connect(self._check)
        self._check()
        self._name.setFocus()

    # -- answers -------------------------------------------------------------
    def location(self) -> Path:
        return Path(self._location.text().strip()).expanduser()

    def project_name(self) -> str:
        return self._name.text().strip()

    def project_path(self) -> Path:
        return self.location() / self.project_name()

    def dont_ask_again(self) -> bool:
        return self._ask_again.isChecked()

    def problem(self) -> str:
        """Why Create Project cannot go ahead with what is typed, or ``""``."""
        issue = project_name_problem(self._name.text())
        if issue:
            return issue
        if not self._location.text().strip():
            return "Choose where to keep the project."
        if not self.location().is_absolute():
            return "Choose a full folder path for the location, or use Browse…"
        return _folder_problem(self.project_path())

    # -- behaviour -----------------------------------------------------------
    def _browse(self) -> None:
        start = self._location.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "Choose where to keep the project", start)
        if chosen:
            self._location.setText(str(Path(chosen)))

    def _check(self) -> None:
        issue = self.problem()
        self._create.setEnabled(not issue)
        if issue:
            # An empty name is where every dialog starts; that is not an error.
            self._note.setText(issue)
            theme.set_tone(self._note, "hint" if not self._name.text().strip() else "error")
            return
        target = str(self.project_path())
        if not mesh_serialization.ansi_safe(target):
            self._note.setText(
                f"Creates {target}\nThis path has characters Windows' code page cannot "
                "show. Runs still work, but mesh files are written through a temporary "
                "folder, which is slower.")
            theme.set_tone(self._note, "warn")
            return
        self._note.setText(f"Creates {target}")
        theme.set_tone(self._note, "hint")

    def _accept_if_valid(self) -> None:
        if not self.problem():
            self.accept()


def _started(record: Any) -> str:
    try:
        return datetime.fromisoformat(str(record.created_at)).astimezone().strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return ""


def _what(record: Any) -> str:
    """``Timelapse Inversion · 2026-09-22 14:03``: which computation, and when."""
    raw = str(record.workflow_id or record.operation_id or "").rpartition(".")[2]
    parts = [raw.replace("_", " ").title() or "Run", _started(record)]
    if record.status == "running":
        parts.append("still running, stops first")
    elif record.status not in ("success", ""):
        parts.append(str(record.status).replace("_", " "))
    return " · ".join(part for part in parts if part)


class SaveRunsDialog(QDialog):
    """Name the runs being saved to the Project, all in one place.

    Each run's name is filled in already - the one it was given, or the one
    its module made from its data - so saving many runs is one press of Enter;
    a run nobody named shows its short name greyed until something is typed.
    With ``allow_discard`` it is the question asked before leaving the
    Project or closing, and :attr:`DISCARD` is returned for Discard.
    """

    DISCARD = 2

    def __init__(self, parent: Optional[QWidget], records: Iterable[Any], *,
                 message: str = "", allow_discard: bool = False,
                 title: str = "Save Runs to Project") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(600)
        self._records: List[Any] = list(records)
        self._edits: Dict[str, QLineEdit] = {}
        self._initial: Dict[str, str] = {}
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        intro = QLabel(message or (
            "Name the runs you are saving, so you can tell them apart later. "
            "You can rename a run at any time in Saved Results."))
        intro.setWordWrap(True)
        layout.addWidget(intro)

        rows = QWidget()
        grid = QGridLayout(rows)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 2)
        for row, record in enumerate(self._records):
            named = not is_placeholder_label(record)
            edit = QLineEdit(str(record.label).strip() if named else "")
            edit.setPlaceholderText(run_title(record))
            edit.setToolTip(f"Run ID: {record.run_id}\nFolder: {record.run_dir}")
            # The name is what is being edited, so it gets the room; the
            # description wraps rather than squeezing it.
            edit.setMinimumWidth(240)
            what = QLabel(_what(record))
            what.setWordWrap(True)
            theme.set_tone(what, "muted")
            grid.addWidget(edit, row, 0)
            grid.addWidget(what, row, 1)
            self._edits[record.run_id] = edit
            self._initial[record.run_id] = edit.text().strip()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(rows)
        # Room for a handful of rows; a long list scrolls instead of growing
        # the dialog off the screen.
        scroll.setMinimumHeight(min(6, max(1, len(self._records))) * 34)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox()
        if allow_discard:
            discard = buttons.addButton("Discard", QDialogButtonBox.DestructiveRole)
            discard.setToolTip("Delete these runs' folders.")
            discard.clicked.connect(lambda: self.done(self.DISCARD))
        buttons.addButton(QDialogButtonBox.Cancel)
        save = buttons.addButton("Save", QDialogButtonBox.AcceptRole)
        save.setProperty("primary", True)
        save.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        if self._records:
            first = self._edits[self._records[0].run_id]
            first.setFocus()
            first.selectAll()

    def names(self) -> Dict[str, str]:
        """The names typed or changed, by run id; untouched runs keep theirs."""
        changed: Dict[str, str] = {}
        for run_id, edit in self._edits.items():
            text = edit.text().strip()
            if text and text != self._initial.get(run_id, ""):
                changed[run_id] = text
        return changed


def confirm_project_for_data(widget: Optional[QWidget]) -> bool:
    """Before a page takes in data, give the user one chance to name a Project.

    Call it first thing in a page's own "add data" action, before its file
    dialog. While results still go to the fallback folder nobody chose, the
    window asks once whether to make a named Project for this work or keep the
    default; making one keeps every page as it is, so nothing the user set up
    is lost. Returns False when the user cancelled and the page should stop.
    A page outside the studio window (a test, a standalone widget) is never
    asked.
    """
    window = widget.window() if widget is not None else None
    ask = getattr(window, "ask_for_project_before_data", None)
    return bool(ask()) if callable(ask) else True
