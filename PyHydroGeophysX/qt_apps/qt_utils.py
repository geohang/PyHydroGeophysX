"""Small Qt UI helpers shared across workbench modules."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractButton,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QWidget,
)


def noop(*_args, **_kwargs) -> None:
    return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_output_dir(state, suffix: str = "") -> Path:
    """Return and create one stable module output directory."""
    base = Path(getattr(state, "output_dir", None) or Path.cwd())
    target = base / suffix if suffix else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def select_directory(
    parent: Optional[QWidget],
    title: str,
    initial: str | Path,
) -> Optional[Path]:
    selected = QFileDialog.getExistingDirectory(parent, title, str(initial))
    return Path(selected) if selected else None


def make_double_spinbox(
    value: float,
    minimum: float,
    maximum: float,
    step: float,
    decimals: int = 3,
    *,
    suffix: str = "",
) -> QDoubleSpinBox:
    widget = QDoubleSpinBox()
    widget.setRange(float(minimum), float(maximum))
    widget.setSingleStep(float(step))
    widget.setDecimals(int(decimals))
    widget.setValue(float(value))
    if suffix:
        widget.setSuffix(suffix)
    return widget


def make_spinbox(value: int, minimum: int, maximum: int, *,
                 suffix: str = "", tooltip: str = "") -> QSpinBox:
    widget = QSpinBox()
    widget.setRange(int(minimum), int(maximum))
    widget.setValue(int(value))
    if suffix:
        widget.setSuffix(suffix)
    if tooltip:
        widget.setToolTip(tooltip)
    return widget


def merged_row(*parts: QWidget | str) -> QWidget:
    """Pack several controls onto one form row, with plain text between them.

    Two numbers that are two ends of one setting read as one setting when they
    share a row and as two unrelated options when they do not. Strings become
    labels, so a row can be written the way it is spoken: ``merged_row(target,
    "±", tolerance)``.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    for part in parts:
        row.addWidget(QLabel(part) if isinstance(part, str) else part)
    row.addStretch(1)  # pack left; stretched spin boxes leave odd gaps
    holder = QWidget()
    holder.setLayout(row)
    return holder


def row_label(widget: QWidget) -> Optional[QLabel]:
    """The QFormLayout label paired with ``widget``, or None.

    Resolved through the widget's own parent rather than a stored layout, so
    moving a row between group boxes cannot silently leave its label behind.
    """
    parent = widget.parentWidget()
    form = parent.layout() if parent is not None else None
    return form.labelForField(widget) if isinstance(form, QFormLayout) else None


def set_rows_visible(widgets: Iterable[QWidget], visible: bool) -> None:
    """Show or hide whole form rows; QFormLayout keeps label and field apart."""
    for widget in widgets:
        widget.setVisible(bool(visible))
        label = row_label(widget)
        if label is not None:
            label.setVisible(bool(visible))


def set_rows_enabled(widgets: Iterable[QWidget], enabled: bool) -> None:
    """Grey out whole form rows, label included.

    Preferred over hiding when the row still explains a setting that exists but
    does not apply, since a control that vanishes reads as one that was removed.
    """
    for widget in widgets:
        widget.setEnabled(bool(enabled))
        label = row_label(widget)
        if label is not None:
            label.setEnabled(bool(enabled))


class ReproduceBar(QWidget):
    """Footer that surfaces the code written after a successful run.

    Every module exports four files next to its results: a recipe, a runner
    that reproduces the run byte for byte, and a walkthrough as both a script
    and a notebook. Nothing told the user any of them existed, so the whole
    reproducible-code path was invisible. The bar stays hidden until a run
    produces a bundle, then offers the two files people actually take away, the
    walkthrough script and the notebook, plus the folder holding the rest.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._script: Optional[Path] = None
        self._recipe: Optional[Path] = None
        self._walkthrough: Optional[Path] = None
        self._notebook: Optional[Path] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        self._label = QLabel()
        self._label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._script_button = QPushButton("Open .py")
        self._notebook_button = QPushButton("Open .ipynb")
        self._folder_button = QPushButton("Open folder")
        self._script_button.setToolTip(
            "Step-by-step Python for this run: named parameters, the domain "
            "calls, and a figure."
        )
        self._notebook_button.setToolTip("The same walkthrough as a Jupyter notebook.")
        self._folder_button.setToolTip(
            "Reveal the recipe and the exact-rerun script alongside the walkthrough."
        )
        self._script_button.clicked.connect(self.open_script)
        self._notebook_button.clicked.connect(self.open_notebook)
        self._folder_button.clicked.connect(self.open_folder)
        layout.addWidget(self._label, 1)
        layout.addWidget(self._script_button)
        layout.addWidget(self._notebook_button)
        layout.addWidget(self._folder_button)
        self.setVisible(False)

    def set_bundle(self, recipe_path, script_path) -> None:
        """Show the bundle from the run that just finished."""
        from PyHydroGeophysX.workflows import teaching_paths

        self._recipe = Path(recipe_path) if recipe_path else None
        self._script = Path(script_path) if script_path else None
        target = self._script or self._recipe
        if target is None:
            self.setVisible(False)
            return
        self._walkthrough, self._notebook = (
            teaching_paths(self._script) if self._script else (None, None)
        )
        self._script_button.setEnabled(self._walkthrough is not None)
        self._notebook_button.setEnabled(self._notebook is not None)

        name = (self._walkthrough or target).name
        self._label.setText(f"Reproduce this run outside the GUI: <b>{name}</b>")
        lines = []
        if self._walkthrough is not None:
            lines.append(f"Walkthrough: {self._walkthrough}")
            lines.append(f"Run it with:  python {self._walkthrough.name}")
        if self._notebook is not None:
            lines.append(f"Notebook: {self._notebook}")
        if self._script is not None:
            lines.append(f"Exact rerun: {self._script}")
        if self._recipe is not None:
            lines.append(f"Recipe: {self._recipe}")
        self.setToolTip("\n".join(lines))
        self.setVisible(True)

    def clear(self) -> None:
        self._script = None
        self._recipe = None
        self._walkthrough = None
        self._notebook = None
        self.setVisible(False)

    @staticmethod
    def _open(target: Optional[Path]) -> None:
        if target is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def open_script(self) -> None:
        self._open(self._walkthrough)

    def open_notebook(self) -> None:
        self._open(self._notebook)

    def open_folder(self) -> None:
        target = self._walkthrough or self._script or self._recipe
        if target is not None:
            self._open(target.parent)


class BusyStateController:
    """Apply and reliably restore a busy state to a set of buttons."""

    def __init__(self, buttons: Iterable[QAbstractButton]) -> None:
        self._buttons = list(buttons)
        self._previous: list[bool] = []

    def start(
        self,
        *,
        enabled_while_busy: Iterable[QAbstractButton] = (),
    ) -> None:
        self._previous = [button.isEnabled() for button in self._buttons]
        for button in self._buttons:
            button.setEnabled(False)
        for button in enabled_while_busy:
            button.setEnabled(True)

    def finish(self) -> None:
        previous = self._previous or [True] * len(self._buttons)
        for button, enabled in zip(self._buttons, previous):
            button.setEnabled(enabled)
        self._previous = []


class WizardNavigator(QObject):
    """Shared bounds-checked navigation for stacked wizard pages."""

    changed = Signal(int)

    def __init__(
        self,
        stack: QStackedWidget,
        *,
        previous_button: Optional[QAbstractButton] = None,
        next_button: Optional[QAbstractButton] = None,
        on_changed: Optional[Callable[[int], None]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.stack = stack
        self.previous_button = previous_button
        self.next_button = next_button
        self.on_changed = on_changed
        if previous_button is not None:
            previous_button.clicked.connect(self.previous)
        if next_button is not None:
            next_button.clicked.connect(self.next)
        self.go_to(stack.currentIndex())

    def go_to(self, index: int) -> int:
        bounded = max(0, min(int(index), max(0, self.stack.count() - 1)))
        self.stack.setCurrentIndex(bounded)
        if self.previous_button is not None:
            self.previous_button.setEnabled(bounded > 0)
        if self.next_button is not None:
            self.next_button.setEnabled(bounded < self.stack.count() - 1)
        if self.on_changed is not None:
            self.on_changed(bounded)
        self.changed.emit(bounded)
        return bounded

    def previous(self) -> None:
        self.go_to(self.stack.currentIndex() - 1)

    def next(self) -> None:
        self.go_to(self.stack.currentIndex() + 1)


class Debouncer(QObject):
    """Coalesce rapid signals into a single delayed callback.

    Connect a widget's ``valueChanged`` / ``toggled`` signal to :meth:`trigger`;
    the wrapped ``callback`` runs once, ``interval_ms`` after the last trigger.
    A slider drag or fast spin therefore stops firing the heavy recompute on
    every step and only runs it when the user pauses.
    """

    def __init__(self, callback: Callable[[], None], interval_ms: int = 80, parent=None) -> None:
        super().__init__(parent)
        self._callback = callback
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._fire)

    def trigger(self, *_args) -> None:
        """(Re)start the timer. Extra signal arguments are ignored."""
        self._timer.start()

    def flush(self) -> None:
        """Run the callback immediately if a call is pending."""
        if self._timer.isActive():
            self._timer.stop()
            self._fire()

    def _fire(self) -> None:
        self._callback()
