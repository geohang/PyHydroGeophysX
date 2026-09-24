"""Base classes shared by all studio module pages."""

from __future__ import annotations

import html
import json
import os
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPlainTextEdit,
                               QTextEdit, QVBoxLayout, QWidget)

LogFn = Callable[..., None]
#: One offer in a module's export menu: what it writes, and how to write it.
ExportAction = Tuple[str, Callable[[], Any]]
#: What counts as something the user can be shown mid-run.
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".svg")

#: Scaled figure thumbnails, newest use last, keyed by (path, mtime, size,
#: height). A run's figures are saved at print resolution - five to six thousand
#: pixels across - and the activity strip rebuilt every thumbnail whenever its
#: figure list changed, decoding each full-size file again to show it 96 px
#: high: about 135 ms of UI-thread time per new figure with four on the strip.
_THUMBNAILS: "OrderedDict[Tuple[str, int, int, int], QPixmap]" = OrderedDict()
#: Enough for several runs' strips; one entry is at most a few hundred KB.
_THUMBNAIL_LIMIT = 32


def thumbnail_pixmap(path: str, height: int = 96) -> Optional[QPixmap]:
    """The image at ``path`` scaled to ``height`` pixels, decoded once.

    Parameters
    ----------
    path : str
        An image file.
    height : int
        Target height in pixels; the aspect ratio is kept.

    Returns
    -------
    QPixmap or None
        None when the file cannot be read as an image - an SVG, or a figure
        still being written - which is not cached, so the next call tries
        again. A file rewritten since has a new modification time or size and
        is decoded afresh.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size, int(height))
    cached = _THUMBNAILS.get(key)
    if cached is not None:
        _THUMBNAILS.move_to_end(key)
        return cached
    full = QPixmap(path)
    if full.isNull():
        return None
    scaled = full.scaledToHeight(int(height), Qt.SmoothTransformation)
    _THUMBNAILS[key] = scaled
    while len(_THUMBNAILS) > _THUMBNAIL_LIMIT:
        _THUMBNAILS.popitem(last=False)
    return scaled


class RunActivity(QFrame):
    """A strip across the top of a module saying what the workflow is doing here.

    The studio's modules are interactive tools holding their own state, while an
    automatic workflow runs headless in its own process and writes into a run
    directory. So bringing a module to the front while the run works there
    showed an *empty* tool - "No files added", "No data loaded" - which reads as
    nothing having happened, which is worse than not navigating at all.

    This says plainly which step is running here, and shows the figures the run
    has just written so the work is visible as it happens. It is an overlay
    rather than a row in each module's layout: modules lay themselves out in
    several different ways, and none of them should have to know about this.
    """

    def __init__(self, page: QWidget) -> None:
        super().__init__(page)
        self.setObjectName("runActivity")
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "#runActivity { background: rgba(23, 92, 145, 235); border: none; }"
            "#runActivity QLabel { color: #ffffff; }")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)
        self._headline = QLabel()
        self._headline.setWordWrap(True)
        layout.addWidget(self._headline)
        self._note = QLabel(
            "The workflow runs in its own process and writes to the run folder; "
            "this panel's own controls are untouched.")
        self._note.setWordWrap(True)
        self._note.setStyleSheet("font-size: 11px; color: #d8e6f2;")
        layout.addWidget(self._note)
        # A container widget rather than a bare nested layout: a QVBoxLayout's
        # heightForWidth does not account for a child layout, so the strip was
        # measured as though it held nothing and the figures were clipped away.
        self._strip_host = QWidget()
        self._strip = QHBoxLayout(self._strip_host)
        self._strip.setContentsMargins(0, 0, 0, 0)
        self._strip.setSpacing(8)
        self._strip_host.setVisible(False)
        layout.addWidget(self._strip_host)
        # Where a question's code goes. Read-only and selectable: approving
        # generated code means having read it, so it must be legible and
        # copyable, not summarised into a label.
        self._code = QPlainTextEdit()
        self._code.setReadOnly(True)
        self._code.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._code.setMaximumHeight(200)
        self._code.setStyleSheet(
            "background: #0f2b40; color: #e6f1f8; border: 1px solid #3d6a8c;"
            "font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 11px;")
        self._code.setVisible(False)
        layout.addWidget(self._code)
        self._opens = QHBoxLayout()
        self._opens.setSpacing(8)
        self._open_host = QWidget()
        self._open_host.setLayout(self._opens)
        self._open_host.setVisible(False)
        layout.addWidget(self._open_host)
        self._open_actions: List[Tuple[str, str]] = []
        self._choices = QHBoxLayout()
        self._choices.setSpacing(8)
        self._choice_host = QWidget()
        self._choice_host.setLayout(self._choices)
        self._choice_host.setVisible(False)
        layout.addWidget(self._choice_host)
        self._shown: List[str] = []
        #: Whether a question is on screen here awaiting an answer.
        self._pending = False
        page.installEventFilter(self)
        self.hide()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() in (QEvent.Resize, QEvent.Show) and watched is self.parent():
            self._reposition()
        return False

    def _reposition(self) -> None:
        """Span the page's width, using only the height that needs.

        Asked for ``sizeHint()`` directly a word-wrapped label answers for a
        width nobody has set yet, and the strip came out several hundred pixels
        tall - it covered the panel it was supposed to annotate. The height is
        taken for the width it will actually have, and capped, because this sits
        on top of somebody's controls.
        """
        parent = self.parentWidget()
        if parent is None:
            return
        width = parent.width()
        layout = self.layout()
        if layout is not None:
            # Thumbnails added a moment ago are not in the layout's measurements
            # until it is activated, and the strip then came out as tall as an
            # empty one with the figures clipped away.
            layout.activate()
        if layout is not None and layout.hasHeightForWidth():
            height = layout.heightForWidth(width)
        else:
            height = self.sizeHint().height()
        self.setGeometry(0, 0, width, min(max(int(height), 48), 220))
        self.raise_()

    def set_open_actions(self, actions: List[Tuple[str, str]]) -> None:
        """Buttons that open what this step produced, on disk.

        The strip must never be only words. Wherever the run takes the user,
        there has to be something to look at and a way to open it - the run
        folder at the very least, and this step's own outputs when it has a
        folder of its own.
        """
        from PySide6.QtWidgets import QPushButton

        wanted = [(str(label), str(path)) for label, path in (actions or [])
                  if label and path]
        if wanted == self._open_actions:
            return
        self._open_actions = wanted
        while self._opens.count():
            item = self._opens.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for label, path in wanted:
            button = QPushButton(label)
            button.setToolTip(path)
            button.clicked.connect(
                lambda _checked=False, p=path: QDesktopServices.openUrl(
                    QUrl.fromLocalFile(p)))
            self._opens.addWidget(button)
        self._opens.addStretch(1)
        self._open_host.setVisible(bool(wanted))

    def show_step(self, step: str, details: str = "",
                  figures: Optional[List[str]] = None) -> None:
        """Say what is running here and show what it has produced so far."""
        # A pending question owns the headline. The Workflow page refreshes this
        # strip once a second to pick up new figures, which would otherwise
        # replace the question the user is being asked with the last step's
        # description - leaving buttons under text that no longer explains them.
        if not self._pending:
            self._headline.setText(f"<b>AQUAH is working here</b> — {step}"
                                   + (f"<br>{details}" if details else ""))
        figures = list(figures or [])
        if figures != self._shown:
            self._shown = figures
            while self._strip.count():
                item = self._strip.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            for path in figures:
                self._strip.addWidget(_Thumbnail(path))
            self._strip.addStretch(1)
            self._strip_host.setVisible(bool(figures))
        self.show()
        self._reposition()

    def ask(self, prompt: str, options: List[Dict[str, str]],
            on_choice: Callable[[str], None]) -> bool:
        """Put a decision to the user here, where they are already looking.

        The run pauses on the panel doing the work, so the buttons belong on
        that panel. Putting them back on the Workflow page - which the run has
        just navigated away from - would leave the user watching a stopped run
        with no visible way to answer it.

        Returns
        -------
        bool
            Whether the prompt was shown. ``False`` when no option carries an
            id, which the caller must treat as unanswered rather than as a
            decision.
        """
        from PySide6.QtWidgets import QPushButton

        self._clear_choices()
        shown = False
        for option in options or []:
            decision = str(option.get("id") or "")
            if not decision:
                continue
            button = QPushButton(str(option.get("label") or decision))
            if option.get("detail"):
                button.setToolTip(str(option["detail"]))
            button.clicked.connect(
                lambda _checked=False, d=decision: on_choice(d))
            self._choices.addWidget(button)
            shown = True
        if not shown:
            return False
        self._choices.addStretch(1)
        self._choice_host.setVisible(True)
        self._pending = True
        # A question that carries code is not a sentence with a tooltip. The
        # whole point of asking is that the user reads it, so it goes in a
        # monospaced, scrollable, selectable box rather than into a label that
        # would wrap it into prose or a button that would hide it.
        head, code = _split_code(prompt)
        self._headline.setText(f"<b>AQUAH is waiting for you</b><br>"
                               f"{html.escape(head).replace(chr(10), '<br>')}")
        if code:
            self._code.setPlainText(code)
            self._code.setVisible(True)
        else:
            self._code.setVisible(False)
        self.show()
        self._reposition()
        return True

    def clear_question(self) -> None:
        """Take the buttons away once answered, keeping the activity strip."""
        self._pending = False
        self._clear_choices()
        self._reposition()

    def _clear_choices(self) -> None:
        self._pending = False
        self._code.setVisible(False)
        self._code.clear()
        while self._choices.count():
            item = self._choices.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._choice_host.setVisible(False)


def _split_code(prompt: str) -> Tuple[str, str]:
    """Separate a question's words from the code it is asking about.

    A question that carries source says everything up to the first ``def`` and
    then the source. Splitting here rather than sending two fields keeps the
    question one string across the process boundary, which is what the desktop
    and the Streamlit app both receive.

    Examples
    --------
    >>> head, code = _split_code(chr(10).join(
    ...     ['Read it:', '', 'def adapt(text):', '    return []']))
    >>> head
    'Read it:'
    >>> print(code)
    def adapt(text):
        return []
    >>> _split_code('Which origin?')
    ('Which origin?', '')
    """
    text = str(prompt or "")
    marker = text.find("def ")
    if marker < 0:
        return text.strip(), ""
    return text[:marker].strip(), text[marker:].strip()


class _Thumbnail(QLabel):
    """One figure the run has written, opening full size when clicked."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self._path = path
        pixmap = thumbnail_pixmap(path, 96)
        if pixmap is not None:
            self.setPixmap(pixmap)
            self.setToolTip(f"{path}\nClick to open full size.")
            self.setCursor(Qt.PointingHandCursor)
        else:
            # An SVG, or a file still being written. Name it rather than
            # showing a blank box.
            from pathlib import Path as _Path
            self.setText(_Path(path).name)
            self.setStyleSheet("font-size: 11px;")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt event override
        QDesktopServices.openUrl(QUrl.fromLocalFile(self._path))


class BaseModule(QWidget):
    """A page in the central stacked widget.

    Subclasses get ``self.state`` (the shared :class:`StudioState`) and a
    ``self.log(message, level)`` helper wired to the bottom log panel. They call
    ``self.report_result(dict)`` to publish their result into the bridge state.
    """

    resultsUpdated = Signal()
    #: Ask the main window to open the Mesh 3D module and load a mesh/volume file.
    viewMeshRequested = Signal(str)
    #: Ask the main window to switch to another module by key (cross-module handoff).
    navigateRequested = Signal(str)
    module_key = "base"
    module_title = "Module"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self._log_fn = log
        self._workers: list = []
        self._run_activity: Optional[RunActivity] = None

    # -- live workflow activity ---------------------------------------------
    def show_run_activity(self, step: str, details: str = "",
                          figures: Optional[List[str]] = None,
                          opens: Optional[List[Tuple[str, str]]] = None) -> None:
        """Say that the automatic workflow is working on this page right now.

        Called by the Workflow page as the run moves between modules, so a panel
        the run has brought to the front explains itself instead of showing an
        empty tool. ``opens`` are the buttons that open what the step produced;
        there is always at least the run folder, because a panel the run visits
        must never leave the user with nothing to look at.
        """
        if getattr(self, "_run_activity", None) is None:
            self._run_activity = RunActivity(self)
        self._run_activity.set_open_actions(opens or [])
        self._run_activity.show_step(step, details, figures)

    def hide_run_activity(self) -> None:
        """Take the strip down when the run is over or has moved elsewhere."""
        banner = getattr(self, "_run_activity", None)
        if banner is not None:
            banner.hide()

    def show_run_inputs(self, inputs: Dict[str, Any]) -> str:
        """Load the automatic run's data into this panel, for display only.

        A strip saying what is happening was not enough: the panel underneath
        still read "No files added" and "No data loaded", with an empty plot,
        so there was nothing to actually look at. The run's inputs are ordinary
        files this module can already open, so the page opens them and plots
        them while the workflow gets on with its own copy in its own process.

        Nothing here computes: a module shows what went in, it does not invert
        it a second time. Nothing here is required either - a module that has no
        sensible view of these inputs returns "" and the strip is all there is.

        Parameters
        ----------
        inputs : dict
            Role to path, as the Workflow page holds them - ``data_file``,
            ``time_lapse_files``, ``electrode_file`` and so on.

        Returns
        -------
        str
            One phrase naming what was opened, for the activity log, or ""
            when nothing was.
        """
        return ""

    def show_run_stage(self, tool: str) -> str:
        """Bring up the view that matches the step the run is on.

        Opening a module's data is not the whole of following a run: loading
        surveys belongs on the data view and a finished inversion belongs on the
        model view, and a page left on whichever tab it happened to be showing
        makes the run look like it did nothing.

        Parameters
        ----------
        tool : str
            The runtime tool's registered name - ``load_ert_surveys``,
            ``invert_time_lapse``, ``convert_water_content`` and so on. Given
            rather than inferred, so a module matches on an identity the
            registry defines instead of on a display string.

        Returns
        -------
        str
            The view it moved to, for the activity log, or "" when the module
            has one view or does not recognise the step.
        """
        return ""

    def log(self, message: str, level: str = "info") -> None:
        try:
            self._log_fn(message, level)
        except Exception:
            pass

    def report_result(self, data: Dict[str, Any]) -> None:
        self.state.update_module_result(self.module_key, data)
        self.resultsUpdated.emit()

    def export_actions(self) -> List[ExportAction]:
        """What ``File > Export Results…`` should offer while this page is open.

        Every module writes its own products in its own formats, and before this
        hook the only way to reach them was to know which button on which tab
        did it. A module returns the exports that make sense *right now* — an
        entry it leaves out is one there is no result for yet — and the window
        runs the single offer directly or asks when there is more than one.
        """
        return []

    def add_to_map(self) -> None:
        """Save and open an explicitly located survey snapshot in Project Map."""
        from PyHydroGeophysX.qt_apps.widgets.map_export import add_result_to_map
        add_result_to_map(self)

    def map_export_button(self):
        from PySide6.QtWidgets import QPushButton
        button = QPushButton("Add to Map…")
        button.setToolTip("Save this recovered result as a survey in Project Map. Confirm its location before adding.")
        button.clicked.connect(self.add_to_map)
        return button

    def offer_map_export(self) -> None:
        """Offer opt-in map placement after a successful inversion is finalized."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QMessageBox
        def offer():
            dialog = QMessageBox(self)
            dialog.setWindowTitle("Result ready")
            dialog.setIcon(QMessageBox.Information)
            dialog.setText("Add this result to Project Map?")
            dialog.setInformativeText(
                "Keep surveys together and reopen their results from the map. "
                "You can confirm the location before adding.\n\n"
                "Not now keeps the result here; Add to Map remains available on this page.")
            add = dialog.addButton("Add to Map…", QMessageBox.AcceptRole)
            later = dialog.addButton("Not now", QMessageBox.RejectRole)
            dialog.setDefaultButton(later)
            dialog.setEscapeButton(later)
            def closed(_result: int) -> None:
                # Take the dialog off this page before handing it to Qt to
                # delete. A queued deleteLater is only drained by a running
                # event loop, so a dialog still parented here when the page is
                # destroyed - the user switches module, or the studio tears the
                # page down - leaves a deletion naming freed memory, and the
                # next event loop to run crashes on it.
                chosen = dialog.clickedButton()
                dialog.setParent(None)
                dialog.deleteLater()
                if chosen is add:
                    self.add_to_map()
            dialog.finished.connect(closed)
            dialog.open()
        # self as the context object: Qt drops the callback if the page goes
        # away before the timer fires, and leaves nothing parented to the page
        # waiting to be collected.
        QTimer.singleShot(0, self, offer)

    def begin_persisted_run(
        self, operation_id: str, workflow_id: str = "", *, label: str = ""
    ):
        """Allocate the sole durable directory for a module operation."""
        return self.state.begin_run(
            self.module_key, operation_id, workflow_id, label=label
        )

    def finish_persisted_run(self, result: Any, operation_id: str = "") -> None:
        self.state.finish_run(self.module_key, result, operation_id)

    def fail_persisted_run(self, error: str, operation_id: str = "") -> None:
        self.state.fail_run(self.module_key, error, operation_id)

    def cancel_persisted_run(self, error: str = "", operation_id: str = "") -> None:
        self.state.cancel_run(self.module_key, error, operation_id)

    # -- agent command interface --------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        """Describe this module for the AQUAH assistant.

        Subclasses override this to advertise the actions they accept and their
        current parameter values. The default reports no actions, so the agent
        can still navigate to the module and read state, but cannot drive it.
        """
        return {
            "module": self.module_key,
            "title": self.module_title,
            "actions": [],
            "note": "This module has no agent actions yet; navigation and status only.",
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run an agent action on this module. Override in subclasses."""
        return {
            "status": "failed",
            "error": f"Module '{self.module_key}' does not support action '{action}'.",
        }

    def agent_view_context(self, view: str) -> Optional[Dict[str, Any]]:
        """Numbers that belong with a captured picture of ``view``, or ``None``.

        A model reading a plot has to match a marker against an axis label far
        away from it, and it gets the index wrong often enough to matter. The
        module already holds those values exactly, so a view that plots
        identifiable per-item values should return them here. The picture then
        answers what only vision can ("is this pick on the first arrival or on
        noise") while the numbers answer which item it was.
        """
        return None

    # -- worker lifecycle ----------------------------------------------------
    def register_worker(self, worker: Any) -> Any:
        """Track a QThread so the window can join it on shutdown.

        The reference is kept until the thread finishes, which both prevents the
        worker from being garbage-collected mid-run and lets :meth:`stop_workers`
        cancel/join anything still running when the app closes.
        """
        self._workers.append(worker)
        worker.finished.connect(lambda: self._drop_worker(worker))
        return worker

    def _drop_worker(self, worker: Any) -> None:
        if worker in self._workers:
            self._workers.remove(worker)

    def stop_workers(self, wait_ms: int = 5000) -> None:
        """Cancel cooperatively-interruptible workers and join running threads."""
        had_workers = bool(self._workers)
        for worker in list(self._workers):
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
                if worker.isRunning():
                    worker.quit()
                    worker.wait(wait_ms)
            except Exception:  # noqa: BLE001 - shutdown best effort
                pass
        if had_workers:
            self.state.cancel_module_runs(
                self.module_key, "Studio closed during computation"
            )


class HomePage(BaseModule):
    """Landing page shown when no specific module is selected."""

    module_key = "home"
    module_title = "Home"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        layout = QVBoxLayout(self)
        title = QLabel("<h2>PyHydroGeophysX Professional Studio</h2>")
        intro = QLabel(
            "Select a module from the project tree on the left.<br><br>"
            "<b>Geophysical Data Processing</b>: Seismic, ERT, Mesh 3D, EM, "
            "Gravity / Magnetics, and Joint Inversion.<br>"
            "<b>Hydro → Geophysics</b>: load hydrologic model outputs, pick a "
            "profile, set survey geometry, and run forward modeling.<br>"
            "<b>Geophy → Hydrology</b>: derive subsurface structure and "
            "hydrology from geophysics. <b>Seismic → Structure</b> builds a 3D "
            "structure (bedrock interface + velocity volume) from velocity "
            "sections; that structure can be handed to <b>ERT → Water Content</b>, "
            "which estimates water content and porosity per layer with Monte "
            "Carlo uncertainty."
        )
        intro.setWordWrap(True)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        layout.addWidget(title)
        from PySide6.QtWidgets import QPushButton
        one_click = QPushButton("Open Workflow · Data, progress and reports")
        one_click.clicked.connect(lambda: self.navigateRequested.emit("one_click"))
        layout.addWidget(one_click)
        layout.addWidget(intro)
        layout.addWidget(QLabel("<b>Session context</b>"))
        layout.addWidget(self._summary, stretch=1)
        self.refresh()

    def refresh(self) -> None:
        try:
            summary = self.state.context_summary()
        except Exception:
            summary = {}
        self._summary.setPlainText(json.dumps(summary, indent=2, default=str))


class PlaceholderModule(BaseModule):
    """A clean page used for not-yet-implemented modules and import failures."""

    def __init__(
        self,
        state: Any,
        log: LogFn,
        title: str,
        message: str,
        key: str = "placeholder",
        parent=None,
    ) -> None:
        super().__init__(state, log, parent)
        self.module_key = key
        self.module_title = title
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel(f"<h3>{title}</h3>"))
        body = QLabel(message)
        body.setWordWrap(True)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(body)
        layout.addStretch(1)
