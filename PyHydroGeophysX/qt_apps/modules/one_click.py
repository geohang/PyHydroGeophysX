"""A single workspace for request, execution, and report delivery."""
from pathlib import Path
import time

from PySide6.QtCore import Qt, QUrl, Signal, QTimer, QElapsedTimer
from PySide6.QtGui import QDesktopServices, QImage, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSplitter, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView)

from .base import BaseModule
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.agents import assistants as assistant_registry
from PyHydroGeophysX.qt_apps.agent.one_click_worker import OneClickWorker
from PyHydroGeophysX.qt_apps.widgets import ai_presence as presence
from PyHydroGeophysX.qt_apps.widgets.ai_presence import (AgentHeader, AgentTimeline,
                                                          FinishCard, LiveCanvas, RunRoute,
                                                          SteerBar, usage_text)
from PyHydroGeophysX.qt_apps.widgets import run_replay

#: How many times a module is offered the run's data before giving up on it.
_OPEN_ATTEMPTS = 3

#: Subfolders of the run directory each studio module's work lands in, so the
#: activity strip can offer a button straight to this step's own outputs rather
#: than only to the run folder.
_MODULE_OUTPUTS = {
    'ert': ('inversion', 'ert_model'),
    'geo_hydrology': ('petrophysics',),
    'seismic': ('seismic',),
    'seismic3d': ('structure',),
    'em': ('tdem',),
    'joint_inversion': ('fusion',),
    'hydro_geophysics': ('forward',),
    'one_click': ('figures',),
}


def _count(n, noun):
    """``'1 thing'``, ``'3 things'``."""
    return f'{n} {noun}' + ('' if n == 1 else 's')


def _finish_message(status, warnings, incomplete):
    """What the assistant's conversation is told when a run ends: the outcome,
    and every warning on a line of its own.

    >>> print(_finish_message('Complete · ready.', ['a', 'b'], False))
    The run finished; 2 things to check before relying on the results:
    • a
    • b
    """
    if not warnings:
        return status
    head = ('The run did not complete:' if incomplete else
            f'The run finished; {_count(len(warnings), "thing")} to check before relying '
            'on the results:')
    return head + ''.join(f'\n• {warning}' for warning in warnings)


class _FittedReportBrowser(QTextBrowser):
    """A report view that scales its figures down to the pane width.

    The report's figures are saved at print resolution - a five-panel section
    is 6000 pixels across - and Qt's Markdown importer renders an image at its
    native size, so the preview showed a single panel, a horizontal scrollbar,
    and axis labels running off the edge. Qt also drops inline HTML from
    Markdown, so an ``<img width=...>`` written into the document does nothing:
    the fit has to happen in the viewer.

    Scaling at display time rather than shrinking the files keeps every figure
    usable at its own resolution for papers and slides, and re-fits them when
    the dock is resized.

    Code blocks are the other thing that will not fit: Qt marks them
    non-breakable, so a single long line - a JSON string holding the request,
    or a Windows path - drags the whole report sideways. They are allowed to
    wrap here for the same reason the figures are scaled.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Reading each PNG back off disk for its size, on every event of a
        # resize drag, is the one expensive part of this; sizes are stable
        # within a report.
        self._native_sizes = {}
        self._fitting = False

    def setMarkdown(self, text):  # noqa: N802 - Qt naming
        # A re-run writes new figures under the same filenames, and a section
        # with a different number of time steps is a different width.
        self._native_sizes.clear()
        super().setMarkdown(text)
        self._wrap_code_blocks()
        self._fit_images()

    def resizeEvent(self, event):  # noqa: N802 - Qt event override
        super().resizeEvent(event)
        self._fit_images()

    def _wrap_code_blocks(self):
        """Let pre-formatted blocks wrap instead of widening the document.

        Independent of the pane width, so it is done once per report rather
        than on every resize.
        """
        cursor = QTextCursor(self.document())
        cursor.beginEditBlock()
        block = self.document().begin()
        while block.isValid():
            block_format = block.blockFormat()
            if block_format.nonBreakableLines():
                block_format.setNonBreakableLines(False)
                cursor.setPosition(block.position())
                cursor.setBlockFormat(block_format)
            block = block.next()
        cursor.endEditBlock()

    def _native_size(self, name):
        """Pixel size of the image ``name`` refers to, or None."""
        if name not in self._native_sizes:
            path = self.document().baseUrl().resolved(QUrl(name)).toLocalFile()
            image = QImage(path)
            self._native_sizes[name] = (
                (image.width(), image.height()) if not image.isNull() else None)
        return self._native_sizes[name]

    def _fit_images(self):
        """Hold every image to the pane width, keeping its aspect ratio."""
        # Narrowing an image can retire the horizontal scrollbar, which resizes
        # the viewport, which lands back here.
        if self._fitting:
            return
        self._fitting = True
        try:
            available = max(200, self.viewport().width() - 24)
            cursor = QTextCursor(self.document())
            cursor.beginEditBlock()
            block = self.document().begin()
            while block.isValid():
                for fragment in (it.fragment() for it in _iter_fragments(block)):
                    image = fragment.charFormat().toImageFormat()
                    if not image.isValid() or not image.name():
                        continue
                    native = self._native_size(image.name())
                    width, height = native or (image.width(), image.height())
                    if width <= 0 or height <= 0:
                        continue
                    target = min(float(width), float(available))
                    if abs(image.width() - target) < 1.0:
                        continue
                    image.setWidth(target)
                    image.setHeight(height * target / width)
                    cursor.setPosition(fragment.position())
                    cursor.setPosition(fragment.position() + fragment.length(),
                                       QTextCursor.KeepAnchor)
                    cursor.setCharFormat(image)
                block = block.next()
            cursor.endEditBlock()
        finally:
            self._fitting = False


def _iter_fragments(block):
    """Every fragment in ``block`` - QTextBlock's iterator is not a Python one."""
    iterator = block.begin()
    while not iterator.atEnd():
        yield iterator
        iterator += 1


def _without_clock(details):
    """A progress detail without the runner's ``[12.3s]`` elapsed prefix."""
    text = str(details or '')
    if text.startswith('[') and 's] ' in text[:12]:
        return text.split('s] ', 1)[1]
    return text


class OneClickModule(BaseModule):
    startAIRequested = Signal(str)
    viewRunRequested = Signal(str)
    viewArtifactRequested = Signal(str, str)
    module_key = 'one_click'
    module_title = 'Workflow'
    workflowFinished = Signal(str)
    #: Each step as the run reports it - ``phase`` 'start' with the
    #: controller's reason, 'done' with its summary and status - so the chat
    #: can narrate the run as it goes.
    stepEvent = Signal(dict)

    def __init__(self, state, log, parent=None):
        super().__init__(state, log, parent)
        self._worker = None
        self._output = None
        self._inputs = {}
        self._data_folder = None
        self._catalog = None
        self._last_catalog_roles = None
        self._catalog_inputs = {}
        self._elapsed = QElapsedTimer()
        # What the run is doing and when it started, so a module it visits can
        # say so and show only the figures this run wrote.
        self._latest_step = ''
        self._latest_detail = ''
        self._run_started_at = 0.0
        # The page currently hosting a pending question, so it can be cleared
        # wherever the run happened to put it.
        self._asked_on = None
        # The step a pending approval is about, so the answer can name it.
        self._asked_step = None
        # Module key to how many times it has been offered the run's data.
        # Counted rather than flagged: a page can be visited before the step
        # that produces what it opens has finished, and re-offering the same
        # files after it has opened them would throw away anything the user
        # changed in the panel meanwhile.
        self._inputs_shown = {}
        # Module key to the tool its view was last put on, so a page is not
        # re-navigated on every progress event of the same step.
        self._staged = {}
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick)
        # When the step now running began, so the figures it writes can be put
        # on its own card in the timeline.
        self._step_started_at = 0.0
        layout = QVBoxLayout(self)
        # The assistant this page works for. It is the one active when the page
        # is built and changes with set_assistant(); a run keeps the one it
        # started with.
        self._assistant = assistant_registry.active()
        self._title_label = QLabel()
        layout.addWidget(self._title_label)
        # The agent's own banner: what the assistant is doing, in its colours, with a
        # clock. Hidden until the first run so an idle page stays plain.
        self.header = AgentHeader()
        self.header.setVisible(False)
        layout.addWidget(self.header)
        self.tabs = QTabWidget()
        setup = QWidget()
        form = QVBoxLayout(setup)
        self._setup_layout = form
        self._workflow_setup = None
        self._request_text = ''
        self._ai_settings = {}
        self.goal = QLabel()
        self.goal.setWordWrap(True)
        form.addWidget(self.goal)
        folder_row = QHBoxLayout()
        choose_folder = QPushButton('Choose data folder · AI classification…')
        choose_folder.clicked.connect(self._choose_folder)
        folder_row.addWidget(choose_folder)
        self._choose_folder_btn = choose_folder
        self.folder_label = QLabel('No data folder selected')
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label, 1)
        form.addLayout(folder_row)
        self._folder_note = QLabel()
        self._folder_note.setWordWrap(True)
        form.addWidget(self._folder_note)
        self.catalog_table = QTableWidget(0, 4)
        self.catalog_table.setHorizontalHeaderLabels(['File', 'Role (editable)', 'Confidence', 'Evidence'])
        self.catalog_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.catalog_table.horizontalHeader().setStretchLastSection(True)
        self.catalog_table.setVisible(False)
        form.addWidget(self.catalog_table)
        # The manual path, kept as a fallback rather than an equal alternative.
        # Choosing a folder and letting the assistant classify it is how this is meant
        # to be used, but classification needs a model - without an API key it
        # cannot run at all - and files are not always gathered in one folder.
        manual = QLabel('<b>Or add files yourself</b><br>'
                        'For data spread across several folders, or when no AI '
                        'model is configured (folder classification needs one). '
                        'Pick what the file is, then Add data.')
        manual.setWordWrap(True)
        form.addWidget(manual)
        row = QHBoxLayout()
        # What the files can be is the assistant's to say (Assistant.input_roles).
        self.role = QComboBox()
        row.addWidget(self.role)
        add = QPushButton('Add data…')
        self._add_input_button = add
        add.clicked.connect(self._choose_files)
        row.addWidget(add)
        remove = QPushButton('Remove selected')
        self._remove_input_button = remove
        remove.clicked.connect(self._remove_input)
        row.addWidget(remove)
        form.addLayout(row)
        self.inputs = QListWidget()
        form.addWidget(self.inputs)
        order = QHBoxLayout()
        self._ordered_buttons = []
        for label, offset in [('Move survey up', -1), ('Move survey down', 1)]:
            button = QPushButton(label)
            self._ordered_buttons.append(button)
            button.clicked.connect(lambda checked=False, delta=offset: self._move_survey(delta))
            order.addWidget(button)
        form.addLayout(order)
        form.addStretch(1)
        note = QLabel('Time-lapse surveys run in the displayed order; select a survey and move it up or down to reorder. Your request and workflow context are sent to the selected AI provider.')
        note.setWordWrap(True)
        form.addWidget(note)
        self._generic_intro = [choose_folder, self.folder_label, self._folder_note, manual, note]
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(setup)
        self._data_tab = scroll
        scroll.setAlignment(Qt.AlignHCenter)
        self.tabs.addTab(scroll, '1 · Data')
        # The run as the assistant sees it: one card per step it decided on, with its
        # reason, its module, its time and what it found. This is where a run
        # is watched; the raw event log stays in its own tab.
        self.timeline = AgentTimeline()
        # Above it, the route the run is on, as the run itself projects it;
        # beside it, the newest figure the run has written, shown large as it
        # appears. The canvas stays hidden until there is a figure to show.
        self.route = RunRoute()
        self.route.nodeClicked.connect(self.timeline.scroll_to_step)
        self.route.setVisible(False)
        self.canvas = LiveCanvas()
        self._canvas_host = QWidget()
        self._canvas_host.setObjectName('agentLive')
        host_layout = QVBoxLayout(self._canvas_host)
        host_layout.setContentsMargins(0, 16, 16, 16)
        host_layout.addWidget(self.canvas)
        self._canvas_host.setVisible(False)
        self._live_split = QSplitter(Qt.Horizontal)
        self._live_split.setObjectName('agentLiveSplit')
        self._live_split.setChildrenCollapsible(False)
        self._live_split.addWidget(self.timeline)
        self._live_split.addWidget(self._canvas_host)
        self._live_tab = QWidget()
        self._live_tab.setObjectName('agentLive')
        live_layout = QVBoxLayout(self._live_tab)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setSpacing(0)
        # A finished run can be watched again here (run_replay); the bar is
        # its transport, shown only while a replay is open.
        self.replay_bar = run_replay.ReplayBar()
        self.replay_bar.setVisible(False)
        self.replay_bar.playToggled.connect(self._replay_play)
        self.replay_bar.speedChanged.connect(self._replay_speed)
        self.replay_bar.scrubbed.connect(self._replay_seek)
        self.replay_bar.frameRequested.connect(self._save_frame)
        self.replay_bar.closed.connect(self._end_replay)
        live_layout.addWidget(self.replay_bar)
        live_layout.addWidget(self.route)
        live_layout.addWidget(self._live_split, 1)
        # While a run goes: pause it before its next step, or tell it something.
        self.steer = SteerBar()
        self.steer.setVisible(False)
        self.steer.noteSent.connect(self._send_note)
        self.steer.pauseRequested.connect(self._request_pause)
        live_layout.addWidget(self.steer)
        self.tabs.addTab(self._live_tab, '2 · Live')
        self.report = _FittedReportBrowser()
        self.report.setOpenLinks(False)
        self.report.anchorClicked.connect(self._open_link)
        self.report.setPlainText('Your interpretation and report will appear here after the workflow finishes.')
        # '&&': a single ampersand is a Qt mnemonic, which rendered the tab as
        # "Results _report".
        self._result_page = QWidget()
        self._result_layout = QVBoxLayout(self._result_page)
        self._result_layout.setContentsMargins(0, 0, 0, 0)
        self.result_summary = QLabel()
        self.result_summary.setWordWrap(True)
        self.result_summary.hide()
        self._result_layout.addWidget(self.result_summary)
        self._result_layout.addWidget(self.report, 1)
        self.tabs.addTab(self._result_page, '3 · Results && report')
        self.files = QListWidget()
        self.files.itemDoubleClicked.connect(lambda item: self._preview(item.data(Qt.UserRole)))
        self.tabs.addTab(self.files, 'Output files')
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.document().setMaximumBlockCount(3000)
        self.tabs.addTab(self.details, 'Raw log')
        layout.addWidget(self.tabs, 1)
        self.status = QLabel('Ready · Select data and describe your goal.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.live_detail = QLabel('Actual workflow events will appear here.')
        self.live_detail.setWordWrap(True)
        theme.set_tone(self.live_detail, 'mono')
        layout.addWidget(self.live_detail)
        self.elapsed_label = QLabel('')
        theme.set_tone(self.elapsed_label, 'muted')
        layout.addWidget(self.elapsed_label)
        # Thin. Its fraction is a monotone guess - the loop does not know how
        # many steps it will take - so it is not the thing to read; the banner
        # and the timeline are.
        self.progress = QProgressBar()
        self.progress.setObjectName('agentProgress')
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        layout.addWidget(self.progress)
        # Off by default. The run used to visit the module doing each piece of
        # work, so the user would watch it happen rather than a bar; now the
        # live timeline shows each step's reason, result and figures on
        # this page, and a window that keeps switching panels under the user
        # was the worse way to show it. Kept as an option for anyone who wants
        # to see each module's own panel as well.
        self.follow = QCheckBox('Also bring each module to the front as it runs')
        self.follow.setChecked(False)
        self.follow.setToolTip(
            'The run is followed in the Live tab on this page. Tick this to have '
            'the studio also switch to the module doing each step.')
        self._follow_enabled = False
        self._current_followed = None
        self.follow.toggled.connect(self._on_follow_toggled)
        layout.addWidget(self.follow)
        # Pacing, not a second mode: the same run either approves each step or
        # goes straight through. It belongs here rather than in the chat's mode
        # list because this is where the approval appears, and because a run is
        # always started from chat - a button here could not start one, since
        # the chat clears the API key from this page as soon as it has used it.
        self.step_through = QCheckBox('Approve each step before it runs')
        self.step_through.setToolTip(
            'Pause before every step so you can approve it, skip it, or stop '
            'the run. The steps and their order are the same either way.')
        layout.addWidget(self.step_through)
        # Where the run stops and waits: for approval of the next step, or for
        # an answer to something it cannot decide. Hidden until it is needed,
        # because a permanently visible empty prompt bar reads as broken.
        self.pause_box = QWidget()
        pause_layout = QVBoxLayout(self.pause_box)
        pause_layout.setContentsMargins(0, 0, 0, 0)
        self.pause_label = QLabel()
        self.pause_label.setWordWrap(True)
        pause_layout.addWidget(self.pause_label)
        self.pause_buttons = QHBoxLayout()
        pause_layout.addLayout(self.pause_buttons)
        self.pause_box.setVisible(False)
        layout.addWidget(self.pause_box)
        actions = QHBoxLayout()
        actions.addStretch(1)
        self.run = QPushButton('Run without AI')
        self.run.setToolTip('Run numerical processing and inspect the evidence without contacting an AI provider.')
        self.run.clicked.connect(self._start_offline)
        self.stop = QPushButton('Stop')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._cancel)
        self.folder = QPushButton('Open output folder')
        self.folder.setEnabled(False)
        self.folder.clicked.connect(lambda: self._open_path(self._output))
        self.replay_button = QPushButton('Replay a run…')
        self.replay_button.setToolTip('Watch a finished run again, from the live_replay.json '
                                      'in its output folder. Nothing is recomputed.')
        self.replay_button.clicked.connect(self._choose_replay)
        self.run.hide()
        for button in (self.run, self.stop, self.folder, self.replay_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        result_actions = QHBoxLayout()
        self.save_result = QPushButton('Save this run to Project')
        self.save_result.clicked.connect(self._save_current_run)
        self.view_result = QPushButton('View models && compare runs')
        self.view_result.clicked.connect(lambda: self.viewRunRequested.emit(self._current_run_id or ''))
        self.view_fit = QPushButton('View data fit')
        self.view_fit.clicked.connect(lambda: self.viewArtifactRequested.emit(self._current_run_id or '', 'fit'))
        self.read_report = QPushButton('Read interpretation')
        self.read_report.clicked.connect(self._read_current_report)
        self.view_result.setProperty('primary', True)
        self.result_state = QLabel()
        self.result_state.setWordWrap(True)
        self._result_buttons = (self.view_result, self.view_fit, self.read_report, self.save_result)
        for widget in self._result_buttons:
            widget.hide()
            result_actions.addWidget(widget)
        self.result_state.hide()
        self._result_layout.insertLayout(1, result_actions)
        self._result_layout.insertWidget(2, self.result_state)
        self._current_run_id = None
        self._result_capabilities = {}
        self._continuation_result = None
        self._continuation_assistant = None
        self.next_step = QWidget()
        next_layout = QHBoxLayout(self.next_step)
        next_layout.setContentsMargins(0, 0, 0, 0)
        self.continue_result = QPushButton('Interpret these models…')
        self.continue_result.clicked.connect(self._prepare_continuation)
        next_layout.addWidget(self.continue_result)
        next_note = QLabel('Reuse the completed numerical models. Review the next task before starting.')
        self._next_note = next_note
        next_note.setWordWrap(True)
        next_layout.addWidget(next_note, 1)
        self.next_step.hide()
        self._result_layout.insertWidget(3, self.next_step)
        self._setup = setup
        self.set_assistant(self._assistant)
        # Everything the Live views are told is recorded, with its time, so the
        # run can be watched again; see run_replay.
        self._recorder = run_replay.LiveRecorder({
            'timeline': (self.timeline, ('reset', 'thinking', 'thought', 'ask',
                                         'clear_question', 'step_started', 'step_done',
                                         'step_skipped', 'finish', 'note', 'notes_read')),
            'route': (self.route, ('reset', 'set_ahead', 'set_thinking', 'step_started',
                                   'step_done', 'finish')),
            'canvas': (self.canvas, ('reset', 'add_figure')),
            'header': (self.header, ('show_state', 'set_clock', 'set_usage', 'clear_usage')),
            'page': (self, ('_place_finish', '_reveal_canvas')),
        })
        self._player = None
        self._replaying = False
        self._replay_dir = ''
        self._last_replay = ''
        #: How the last run ended, for the assistant to read (agent_describe).
        self._last_result = None
        self._usage_total = {}
        self._step_clock = None

    def _name(self):
        """The assistant's name, for what this page says."""
        return self._assistant.name

    def _goal_hint(self):
        return (f'Describe your goal to {self._name()} in the assistant panel on the '
                'right and select Auto to report.')

    def set_assistant(self, assistant):
        """Work for ``assistant``: its input roles, its name, its folder sorting.

        Inputs added for a role the new assistant does not take are dropped, so
        a run never receives a file it cannot place. Refused while a run is
        going: the run belongs to the assistant that started it.
        """
        if self._worker is not None:
            return False
        self._assistant = assistant
        self.next_step.hide()
        self._request_text = ''
        if self._workflow_setup is not None:
            self._setup_layout.removeWidget(self._workflow_setup)
            self._workflow_setup.hide()
            self._workflow_setup.deleteLater()
            self._workflow_setup = None
        if getattr(assistant, 'workflow_setup', ''):
            factory = assistant_registry._load(assistant.workflow_setup)
            self._workflow_setup = factory(self)
            self._setup_layout.insertWidget(0, self._workflow_setup)
            self._workflow_setup.changed.connect(self._sync_setup_task)
            if hasattr(self._workflow_setup, 'detailsChanged'):
                self._workflow_setup.detailsChanged.connect(self._set_compact_details)
            self._workflow_setup.update_inputs(self._inputs)
        for widget in self._generic_intro:
            widget.setVisible(self._workflow_setup is None)
        for widget in self._ordered_buttons:
            widget.setVisible(bool(assistant.ordered_roles))
        self.run.setVisible(getattr(assistant, 'offline_workflow', False))
        name = assistant.name
        self._title_label.setText(
            f'<h2>Workflow</h2>Data, progress, results and report · {name} '
            f'({assistant.domain}) works from the assistant panel on the right')
        if not self._request_text:
            self.goal.setText(self._goal_hint())
        self.role.clear()
        for label, key in assistant.input_roles:
            self.role.addItem(label, key)
        roles = {key for _label, key in assistant.input_roles}
        dropped = [key for key in self._inputs if key not in roles]
        for key in dropped:
            self._inputs.pop(key, None)
        if dropped:
            self._refresh_inputs()
        classify = bool(assistant.folder_classifier)
        self._choose_folder_btn.setEnabled(classify)
        self._choose_folder_btn.setToolTip(
            '' if classify else f'{name} does not sort folders; add files below.')
        self._folder_note.setText(
            f'{name} scans filenames and short text previews in this folder using your '
            'selected AI model. Review file roles below before sending “continue”. '
            'No files are moved.' if classify else
            f'{name} does not sort a folder for you: add each file below with its role.')
        self.header.headline.setText(name)
        self.steer.set_name(name)
        self._sync_setup_task()
        self._set_compact_details(False)
        return True

    def _set_compact_details(self, expanded=False):
        compact = self._workflow_setup is not None and hasattr(self._workflow_setup, 'detailsChanged')
        detailed = not compact or expanded
        for widget in (self.role, self._remove_input_button, self.follow, self.step_through,
                       self.folder, self.replay_button, self.live_detail, self.elapsed_label):
            widget.setVisible(detailed)
        for widget in (self.files, self.details):
            self.tabs.setTabVisible(self.tabs.indexOf(widget), detailed)
        self._setup.setMaximumWidth(800 if compact else 16777215)
        self.inputs.setMaximumHeight(110 if compact else 16777215)
        self.inputs.setVisible(detailed or bool(self._inputs))
        if compact:
            self.header.hide()
            self._title_label.setText('<h1>GeoSAGE</h1>')
            self.tabs.setTabText(0, 'Start')
            self.tabs.setTabText(1, 'Activity')
            self.tabs.setTabText(2, 'Results')
            self._add_input_button.setText('Add input…' if expanded else
                                            'Choose configuration…' if self._workflow_setup.primary_role == 'config_file'
                                            else 'Choose results folder…')
            self._add_input_button.setMinimumHeight(44)
            self.stop.setVisible(self._worker is not None)
            self.run.setProperty('primary', True)
            self.run.style().unpolish(self.run)
            self.run.style().polish(self.run)
            self.run.setMaximumWidth(200)
            self.run.setMinimumHeight(42)
            self.view_result.setText('Explore model')
            self.view_fit.setText('Data fit')
            self.save_result.setText('Save')
            self._next_note.hide()
            if not expanded:
                self.role.setCurrentIndex(self.role.findData(self._workflow_setup.primary_role))
        else:
            self._add_input_button.setText('Add data…')
            self.stop.show()
            self.run.setMaximumWidth(16777215)

    def _sync_setup_task(self):
        setup = self._workflow_setup
        if setup is None:
            self.run.setText('Run without AI')
            self.goal.show()
            return
        self.role.clear()
        for label, key in self._assistant.input_roles:
            if key in setup.allowed_roles():
                self.role.addItem(label, key)
        # An existing-results task starts with its primary input, not a JSON file.
        index = self.role.findData(getattr(setup, 'primary_role', ''))
        if index >= 0:
            self.role.setCurrentIndex(index)
        self.run.setText(setup.action_label)
        self.run.setToolTip('Use the provider configured in Assistant settings to interpret these results.'
                            if setup.needs_ai else 'Process locally without contacting an AI provider.')
        if self._worker is None:
            self.status.setText('Choose your inputs to begin.')
        self.goal.hide()
        self._request_text = ''
        self._refresh_inputs()
        self._title_label.setText(f'<h2>{self._name()}</h2>Choose a task, check the inputs, then explore the results.')
        self._set_compact_details(bool(getattr(setup, 'options', None) and setup.options.isChecked()))

    def _read_current_report(self):
        self.tabs.setCurrentWidget(self._result_page)
        self.report.verticalScrollBar().setValue(0)
        self.report.setFocus()

    def _prepare_continuation(self):
        if self._worker is not None or self._continuation_assistant is not self._assistant:
            return
        self._sync_result_storage()
        if not self.continue_result.isEnabled():
            return
        prepare = getattr(self._workflow_setup, 'prepare_continuation', None)
        if not callable(prepare) or not self._continuation_result:
            return
        try:
            inputs = prepare(self._continuation_result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.result_state.setText(f'Could not prepare the next task: {exc}')
            return
        self._inputs = dict(inputs)
        self._catalog_inputs = {}
        self._request_text = ''
        self._refresh_inputs()
        self.tabs.setCurrentWidget(self._data_tab)
        self.status.setText('Existing models selected · Check the inputs and AI settings, then start when ready.')
        self._presence(presence.IDLE, 'Next task ready to configure',
                       'Completed models are selected. Nothing new has started.')
        self.header.set_clock(0, 0, running=False)
        self.header.clear_usage()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.elapsed_label.clear()
        self.live_detail.setText('The previous run remains available in Results and Project history.')

    def _sync_result_storage(self):
        if not self._current_run_id:
            return
        store = self.state.results_store
        record = store.get_run(self._current_run_id) if store else None
        if record is None:
            self.result_state.setText('This run is not in the current Project. Reopen its Project to view or save it.')
            for button in self._result_buttons:
                button.setEnabled(False)
            self.continue_result.setEnabled(False)
            return
        pending = store.is_unsaved(self._current_run_id)
        for button, available in self._result_capabilities.items():
            button.setEnabled(available)
        self.save_result.setEnabled(pending and record.status != 'running')
        self.continue_result.setEnabled(bool(self._continuation_result) and
                                        self._continuation_assistant is self._assistant and
                                        record.status != 'running')
        self.result_state.setText('Files are local · Not saved to Project history yet' if pending else
                                  'Saved locally in Project history')

    def showEvent(self, event):  # noqa: N802 - Qt override
        super().showEvent(event)
        self._sync_result_storage()

    def _save_current_run(self):
        if self._current_run_id:
            self._sync_result_storage()
            if not self.save_result.isEnabled():
                return
            try:
                self.state.results_store.save_run(self._current_run_id)
                self._sync_result_storage()
                if self.state.on_runs_changed:
                    self.state.on_runs_changed()
            except (OSError, KeyError, RuntimeError, ValueError, TypeError) as exc:
                self.result_state.setText(f'Could not save: {exc}. Your output files remain available.')

    def _choose_folder(self):
        from PyHydroGeophysX.qt_apps.widgets.project_dialogs import confirm_project_for_data
        if not confirm_project_for_data(self):   # name a Project before the first data
            return
        folder = QFileDialog.getExistingDirectory(self, 'Select the folder containing survey data and supporting files')
        if folder:
            for key, value in self._catalog_inputs.items():
                if self._inputs.get(key) == value:
                    self._inputs.pop(key, None)
            self._catalog_inputs = {}
            self._last_catalog_roles = None
            self._refresh_inputs()
            self._data_folder = folder
            self._catalog = None
            self.catalog_table.setRowCount(0)
            self.catalog_table.setVisible(False)
            self.folder_label.setText(folder)
            self.status.setText(f'Folder selected · Send your goal to {self._name()} to scan and classify files.')

    def _catalog_rows(self):
        rows = []
        for index, row in enumerate(self._catalog['files']):
            rows.append({**row, 'role': self.catalog_table.cellWidget(index, 1).currentData()})
        return rows

    def _tick(self):
        self._refresh_activity()
        seconds = self._elapsed.elapsed() / 1000
        self.elapsed_label.setText(f'Running · {seconds:.0f}s elapsed · Latest backend event shown above')
        self.header.set_clock(seconds, self._steps_done(), running=True,
                              ahead=self.route.counts()[1])
        self._collect_figures()

    def _steps_done(self):
        return sum(1 for card in self.timeline.steps() if card.status != 'running')

    def _presence(self, state, headline, detail=None, glow_state=None):
        """Say what the agent is doing, everywhere it is shown at once.

        The banner here and the glow round the studio's central area describe
        one state, so they are set together and cannot disagree. ``glow_state``
        differs only when the run is over but has left the user something to
        do: the banner keeps asking, while the edge stops pulsing.
        """
        self.header.show_state(state, headline, detail)
        if getattr(self._assistant, 'focused_workspace', False):
            self.header.hide()
        glow = getattr(self.window(), 'set_agent_presence', None)
        if callable(glow):
            try:
                glow(glow_state or state)
            except Exception:  # noqa: BLE001 - a display must not stop a run
                pass

    def _show_catalog(self, catalog):
        from PyHydroGeophysX.agents.folder_catalog import ROLES, ROLE_LABELS
        self._catalog = catalog
        self.catalog_table.setRowCount(len(catalog['files']))
        for index, row in enumerate(catalog['files']):
            for column, text in [(0, row['name']), (2, f"{row.get('confidence', 0):.0%}"), (3, row.get('reason', ''))]:
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                item.setToolTip(row['path'] if column == 0 else text)
                self.catalog_table.setItem(index, column, item)
            role = QComboBox()
            for key in ROLES:
                role.addItem(ROLE_LABELS[key], key)
            role.setCurrentIndex(role.findData(row['role']))
            self.catalog_table.setCellWidget(index, 1, role)
        self.catalog_table.setVisible(True)

    def _choose_files(self):
        from PyHydroGeophysX.qt_apps.widgets.project_dialogs import confirm_project_for_data
        if not confirm_project_for_data(self):   # name a Project before the first data
            return
        role = self.role.currentData()
        if role.endswith('_dir'):
            path = QFileDialog.getExistingDirectory(self, 'Select model output folder')
            paths = [path] if path else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Select data files')
        if not paths:
            return
        problem = self._add_inputs(role, paths)
        if problem:
            self.status.setText(problem)

    def _add_inputs(self, role, paths):
        """Give ``role`` these files; returns why not, or "".

        A role the assistant takes in order (time-lapse surveys) collects them;
        any other takes one, replacing what it had.
        """
        if role in self._assistant.ordered_roles:
            paths = list(dict.fromkeys(list(self._inputs.get(role, [])) + list(paths)))
            if role == 'time_lapse_files':
                self._inputs.pop('data_file', None)
            self._inputs[role] = paths
        else:
            if len(paths) > 1:
                return ('Select one file for this role, or a role that takes '
                        'several files in order.')
            if role == 'data_file':
                self._inputs.pop('time_lapse_files', None)
            self._inputs[role] = paths[0]
        self._refresh_inputs()
        return ''

    def _refresh_inputs(self):
        self.inputs.clear()
        if self._workflow_setup is not None:
            self._workflow_setup.update_inputs(self._inputs)
            if hasattr(self._workflow_setup, 'detailsChanged'):
                self.inputs.setVisible(bool(self._inputs) or self._workflow_setup.options.isChecked())
        for role, value in self._inputs.items():
            if self._workflow_setup is not None and role not in self._workflow_setup.allowed_roles():
                continue
            for index, path in enumerate(value if isinstance(value, list) else [value], 1):
                compact = self._workflow_setup is not None and hasattr(self._workflow_setup, 'detailsChanged')
                item = QListWidgetItem(Path(path).name if compact else f'{role} · {index} · {path}')
                item.setToolTip(f'{role}\n{path}')
                item.setData(Qt.UserRole, (role, path))
                self.inputs.addItem(item)

    def _remove_input(self):
        item = self.inputs.currentItem()
        if item:
            role, path = item.data(Qt.UserRole)
            if isinstance(self._inputs[role], list) and len(self._inputs[role]) > 1:
                self._inputs[role].remove(path)
            else:
                self._inputs.pop(role)
            self._refresh_inputs()

    def _move_survey(self, delta):
        item = self.inputs.currentItem()
        if not item:
            return
        role, path = item.data(Qt.UserRole)
        if role not in self._assistant.ordered_roles:
            return
        paths = self._inputs[role]
        index = paths.index(path)
        target = index + delta
        if 0 <= target < len(paths):
            paths[index], paths[target] = paths[target], paths[index]
            self._refresh_inputs()
            for row in range(self.inputs.count()):
                if self.inputs.item(row).data(Qt.UserRole) == (role, path):
                    self.inputs.setCurrentRow(row)
                    break

    def reset_request(self):
        """Start a new conversational goal while keeping the selected data."""
        if self._worker is None:
            self._request_text = ''
            self.goal.setText(self._goal_hint())

    def submit_request(self, text, settings):
        if self._worker is not None:
            return 'A workflow is already running. Follow its progress in the center or use Stop.'
        if text.strip() != self._request_text and (text.strip().lower() not in {'continue', 'run', '继续', '开始'} or not self._request_text):
            self._request_text = (self._request_text + '\n\nUser clarification (overrides earlier details):\n'
                                  + text.strip()) if self._request_text else text.strip()
        self._ai_settings = dict(settings)
        self.goal.setText('Goal: ' + self._request_text)
        if self._catalog is not None:
            try:
                from PyHydroGeophysX.agents.folder_catalog import catalog_inputs
                rows = self._catalog_rows()
                signature = [(r['path'], r['role']) for r in rows]
                if signature != self._last_catalog_roles:
                    classified = catalog_inputs(rows)
                    for key, value in self._catalog_inputs.items():
                        if self._inputs.get(key) == value:
                            self._inputs.pop(key, None)
                    self._inputs.update(classified)
                    self._catalog_inputs = {k: list(v) if isinstance(v, list) else v for k, v in classified.items()}
                    self._last_catalog_roles = signature
                    self._refresh_inputs()
            except ValueError as exc:
                self._ai_settings = {}
                return str(exc)
        if not self._inputs and not self._data_folder:
            self._ai_settings = {}
            self.status.setText(f'Waiting for data · Add files here, then send “continue” to {self._name()}.')
            self.tabs.setCurrentWidget(self._data_tab)
            return 'Add your data in the Workflow page, then send “continue” here. I have kept your goal.'
        self._start(step_mode=self.step_through.isChecked())
        self._ai_settings = {}
        if not self._worker:
            return self.status.text()
        return ('Workflow started, pausing before each step. Approve, skip or stop each one in the center.'
                if self.step_through.isChecked() else
                'Workflow started. Progress and the report appear in the center; use Stop there to cancel.')

    def _start_offline(self):
        if not getattr(self._assistant, 'offline_workflow', False):
            return
        if self._workflow_setup is not None:
            self._request_text = self._workflow_setup.request()
            if self._workflow_setup.needs_ai:
                self.startAIRequested.emit(self._request_text)
                return
        if not self._request_text:
            self._request_text = 'Process the supplied configuration or results and summarize the numerical evidence without AI.'
            self.goal.setText(self._request_text)
        self._start(step_mode=self.step_through.isChecked(), offline=True)

    def _start(self, step_mode=False, *, offline=False):
        if self._worker is not None:
            return
        request = self._request_text
        if not request or (not self._inputs and not self._data_folder):
            self.status.setText('Describe your goal and add the data to analyze first.')
            return
        for value in (self._inputs.values() if self._workflow_setup is None else ()):
            for path in value if isinstance(value, list) else [value]:
                if not Path(path).exists():
                    self.status.setText(f'Data no longer exists: {path}')
                    return
        settings = {} if offline else self._ai_settings
        provider = settings.get('provider', 'openai')
        key = settings.get('api_key')
        from PyHydroGeophysX.llm.providers import CLI_PROVIDER_IDS
        if not key and provider not in CLI_PROVIDER_IDS and not (offline and getattr(self._assistant, 'offline_workflow', False)):
            self.status.setText('Configure an API provider or sign in with a CLI provider in Assistant settings.')
            return
        try:
            run_label = request[:120]
            if self._workflow_setup is not None:
                try:
                    prepared = self._workflow_setup.prepare_payload(dict(
                        request=request, inputs=dict(self._inputs), api_key=key,
                        provider=provider, model=settings.get('model'),
                        use_ai=not offline,
                        output_dir=str(Path(self.state.output_dir) / 'runs' / '_preflight')))
                    run_label = str(prepared.get('run_label') or run_label)
                except (ValueError, KeyError, OSError) as exc:
                    self.status.setText(f'Action needed · {exc}')
                    show_error = getattr(self._workflow_setup, 'show_error', None)
                    if callable(show_error):
                        show_error(str(exc))
                    self.tabs.setCurrentWidget(self._data_tab)
                    return
            handle = self.begin_persisted_run('unified', label=run_label)
            self._current_run_id = handle.run_id
            self._output = str(handle.outputs_dir)
            payload = dict(assistant=self._assistant.key,
                           request=request, inputs=dict(self._inputs), provider=provider,
                           model=settings.get('model'), api_key=key, use_ai=not offline,
                           output_dir=self._output)
            payload.update({k: settings.get(k) for k in ('reasoning_effort', 'use_rag', 'use_mcp')})
            payload['step_mode'] = bool(step_mode)
            if self._workflow_setup is not None:
                payload = self._workflow_setup.prepare_payload(payload)
                show_configuration = getattr(self._workflow_setup, 'show_configuration', None)
                if callable(show_configuration):
                    show_configuration(payload)
            if (self._data_folder and self._catalog is None
                    and self._assistant.folder_classifier):
                payload.update(mode='classify', data_folder=self._data_folder)
            if self._catalog:
                payload['classification'] = self._catalog_rows()
            if settings.get('chat_context'):
                payload['request'] += '\n\n' + settings['chat_context']
            payload['chat_reference_sources'] = settings.get('chat_reference_sources', [])
            worker = OneClickWorker(payload, self)
            self._worker = self.register_worker(worker)
            worker.progress.connect(self._on_progress)
            worker.stepped.connect(self._on_step_event)
            worker.asked.connect(self._on_asked)
            worker.logged.connect(self._on_log)
            worker.succeeded.connect(self._succeeded)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._finished)
            self._setup.setEnabled(False)
            self.run.setEnabled(False)
            self.step_through.setEnabled(False)
            self.stop.setEnabled(True)
            self.stop.show()
            self.folder.setEnabled(True)
            self.details.clear()
            self.files.clear()
            self.report.setPlainText('Workflow running. Follow it in the Live tab, or the raw events in Raw log.')
            self.progress.setValue(0)
            self.progress.setRange(0, 0) if self._workflow_setup is not None else self.progress.setRange(0, 100)
            for widget in (*self._result_buttons, self.result_state, self.result_summary):
                widget.hide()
            self.next_step.hide()
            self._continuation_result = None
            self.status.setText('Starting · You can continue using other Studio modules.')
            classify = payload.get('mode') == 'classify'
            self._end_replay()
            self._recorder.start()
            self._recording_meta = {'assistant': self._assistant.key, 'name': self._name(),
                                    'goal': request, 'started': time.strftime('%Y-%m-%d %H:%M')}
            self.timeline.reset(request)
            self.route.reset()
            self.route.setVisible(not classify)
            self.route.set_thinking(True)
            self.canvas.reset()
            self._reveal_canvas(False)
            self._usage_total = {}
            self._step_clock = None
            self.header.clear_usage()
            self.steer.set_name(self._name())
            self.steer.set_state(SteerBar.RUNNING)
            self.steer.setVisible(not classify)
            self.replay_button.setEnabled(False)
            reading = f'{self._name()} is reading your folder' if classify else f'{self._name()} is reading your request'
            self.timeline.thinking(reading)
            self._presence(presence.THINKING, reading,
                           'Working out what to do from your goal and your data.')
            self.header.set_clock(0, 0)
            self.tabs.setCurrentWidget(self._live_tab)
            self._run_started_at = time.time()
            self._inputs_shown = {}
            self._staged = {}
            self._elapsed.start()
            self._clock.start()
            worker.start()
        except Exception as exc:
            self._failed(str(exc))
            self._finished()

    def _on_progress(self, step, fraction, details, module=''):
        if self.progress.maximum():
            self.progress.setValue(max(self.progress.value(), min(99, int(fraction * 100))))
        self.status.setText(f'{step} · {details}')
        self.details.appendPlainText(f'{step}: {details}')
        self._latest_step, self._latest_detail = step, details
        # Stages that are not one of the controller's steps - reading the
        # request, retrieving references, writing the audit - are the agent
        # busy between steps, and are shown as such.
        if self.timeline.current() is None and not module:
            self.timeline.thinking(f'{self._name()} · {step}')
            self.header.show_state(presence.THINKING, f'{self._name()} · {step}',
                                   _without_clock(details))
        self._follow_along(module, step, details)

    def _on_step_event(self, event):
        """A step began or ended: put it on the timeline and the banner."""
        phase = str(event.get('phase') or '')
        label = str(event.get('label') or event.get('tool') or 'Step')
        module = str(event.get('module') or '')
        tool = str(event.get('tool') or '')
        if phase == 'route':
            self.route.set_ahead(event.get('ahead') or [])
        elif phase == 'thought':
            self.timeline.thought(str(event.get('text') or ''))
        elif phase == 'usage':
            self._usage_total = dict(event)
            self.header.set_usage(int(event.get('tokens') or 0),
                                  float(event.get('cost_usd') or 0.0),
                                  int(event.get('calls') or 0))
        elif phase == 'steer':
            self._on_notes_read(event)
        elif phase == 'paused':
            self._on_paused(str(event.get('label') or ''))
        elif phase == 'resumed':
            self.steer.set_state(SteerBar.RUNNING)
            self.route.set_thinking(True)
            self.timeline.thinking(f'{self._name()} is choosing the next step')
            self._presence(presence.THINKING, f'{self._name()} is carrying on', '')
        elif phase == 'start':
            self._step_started_at = time.time()
            self._step_clock = event.get('elapsed_seconds')
            reason = str(event.get('reason') or '')
            self.timeline.step_started(label, module, reason)
            self.route.step_started(tool, label)
            where = f' in {presence.module_title(module)}' if module else ''
            self._presence(presence.WORKING, f'{self._name()} · {label}',
                           f'Why: {reason}' if reason else f'Working{where}.')
        elif phase == 'done':
            status = str(event.get('status') or 'ok')
            summary = str(event.get('summary') or '')
            figures = self._recent_figures(since=self._step_started_at or self._run_started_at)
            seconds = None
            try:
                seconds = max(0.0, float(event['elapsed_seconds']) - float(self._step_clock))
            except (KeyError, TypeError, ValueError):
                pass
            self.timeline.step_done(label, status, summary, figures, module, seconds=seconds)
            self.route.step_done(tool, label, status)
            self.route.set_thinking(True)
            self._collect_figures(label)
            self.timeline.thinking(f'{self._name()} is choosing the next step')
            verb = {'ok': 'finished', 'failed': 'could not finish'}.get(status, status)
            self._presence(presence.THINKING, f'{self._name()} is choosing the next step',
                           f'{label} {verb}' + (f': {summary}' if summary else '.'))
        if self._elapsed.isValid():
            self.header.set_clock(self._elapsed.elapsed() / 1000, self._steps_done(),
                                  ahead=self.route.counts()[1])
        self.stepEvent.emit(dict(event))

    # -- telling a run something while it works ------------------------------------
    def _send_note(self, text):
        """Pass the user's note to the run; it is read before the next decision."""
        if self._worker is None:
            return
        self._worker.steer(text)
        self.timeline.note(text)
        self.details.appendPlainText(f'-> note to the run: {text}')
        when = ('when it carries on' if self.steer.state() == SteerBar.PAUSED
                else 'before its next decision')
        self.status.setText(f'Note sent · {self._name()} reads it {when}.')

    def _request_pause(self, pause):
        """Hold the run before its next step, or let it carry on."""
        if self._worker is None:
            return
        if pause:
            self._worker.pause()
            self.details.appendPlainText('-> pause requested')
            self.status.setText(f'Pausing · {self._name()} stops before its next step; '
                                'the step running now finishes first.')
        else:
            self._worker.resume()
            self.steer.set_state(SteerBar.RUNNING)
            self.details.appendPlainText('-> resume')

    def _on_paused(self, after):
        self.steer.set_state(SteerBar.PAUSED)
        self.route.set_thinking(False)
        where = f' after {after}' if after else ''
        self.timeline.thinking(f'Paused{where}', presence.WAITING)
        self._presence(presence.WAITING, f'{self._name()} is paused{where}',
                       'Send it a note, then Resume - or Stop.')
        self.status.setText(f'Paused · {self._name()} is waiting before its next step.')

    def _on_notes_read(self, event):
        notes = [str(n) for n in event.get('notes') or []]
        heard = bool(event.get('heard', True))
        changes = [str(c) for c in event.get('changes') or []]
        self.timeline.notes_read(notes, str(event.get('why') or ''), changes, heard)
        self.status.setText(
            (f'{self._name()} read your note' + (f' and changed {len(changes)} setting(s)'
                                                  if changes else '') + '.')
            if heard else 'This run has no model to read notes; pause and stop still work.')

    def _on_follow_toggled(self, enabled):
        self._follow_enabled = bool(enabled)

    def _follow_along(self, module, step, details=''):
        """Bring the module doing the current work to the front, and say so.

        A run that only moves a progress bar gives the user no way to see what
        it is doing; the studio already has a panel for each kind of work, so
        the run visits them as it goes. Deliberately reversible: the checkbox
        turns it off, because a window that yanks itself away from someone
        reading a panel is worse than one that never moved.

        Navigating alone was not enough. The workflow runs headless in its own
        process while each module holds its own state, so arriving at one showed
        an empty tool - "No files added", "No data loaded" - which reads as
        nothing having happened. The page is therefore also told what is running
        on it and handed the figures the run has written, which is the part the
        user can actually see.
        """
        if not module or not getattr(self, '_follow_enabled', True):
            return
        if module != getattr(self, '_current_followed', None):
            previous = self._module_page(getattr(self, '_current_followed', None))
            if previous is not None and hasattr(previous, 'hide_run_activity'):
                previous.hide_run_activity()
            show = getattr(self.window(), 'show_module', None)
            if not callable(show):
                return
            try:
                # Navigate before looking the page up: the studio builds a
                # module page the first time it is shown, so on a first visit
                # there is nothing to annotate until this has run.
                show(module)
            except Exception:  # noqa: BLE001 - navigation must not stop a run
                return
            self._current_followed = module
            self.details.appendPlainText(f'-> showing {module} for: {step}')
        page = self._module_page(module)
        if page is not None and hasattr(page, 'show_run_activity'):
            self._show_inputs_on(page, module)
            self._show_stage_on(page, module, step)
            page.show_run_activity(step, details, self._recent_figures(),
                                   self._open_actions(module))

    def _show_inputs_on(self, page, module):
        """Ask a module to open the run's data, once, the first time it is visited.

        The strip alone said what was happening over an empty tool. The files
        the run is working on are ordinary files these panels can already open,
        so they open them - for display; the run keeps inverting its own copy in
        its own process.

        Best effort in every direction: a module with no view of these inputs
        says so by returning nothing, and a module that raises is a display
        problem, which must never disturb a running workflow.
        """
        attempts = self._inputs_shown.get(module, 0)
        # A few attempts, not one and not unlimited. One is too few because a
        # page can be visited before the step that produces what it opens has
        # finished - the water-content page wants the model the inversion is
        # still writing. Unlimited is too many because a file this panel cannot
        # read would be re-read on every progress event for the rest of the run.
        if not module or attempts >= _OPEN_ATTEMPTS:
            return
        self._inputs_shown[module] = attempts + 1
        show = getattr(page, 'show_run_inputs', None)
        if not callable(show):
            self._inputs_shown[module] = _OPEN_ATTEMPTS
            return
        try:
            opened = show({**self._run_config(), **self._inputs})
        except Exception as exc:  # noqa: BLE001 - a preview must not stop a run
            self._inputs_shown[module] = _OPEN_ATTEMPTS
            self.details.appendPlainText(f'-> could not open the data in {module}: {exc}')
            return
        if opened:
            self._inputs_shown[module] = _OPEN_ATTEMPTS
            self.details.appendPlainText(f'-> opened in {module}: {opened}')

    def show_run_inputs(self, inputs):
        """This page's own turn: the run comes back here to write the report.

        It holds the request, the activity log and the report itself, so what it
        opens is its own record rather than a data file. Stated explicitly all
        the same, because the rule is that every panel the run visits shows the
        user something, and a page that quietly relied on being the one the run
        started from would be the first to drift.
        """
        folder = inputs.get('output_dir') or self._output
        if not folder:
            return ''
        return f'the run folder ({Path(str(folder)).name})'

    def show_run_stage(self, tool):
        """Show the report as it is being written, then the report itself."""
        if str(tool) != 'write_report':
            return ''
        self.tabs.setCurrentWidget(self._live_tab)   # while it is being written
        return 'Live'

    def _current_tool(self, step):
        """The assistant's tool whose label is ``step``, or "".

        Looked up in the registry the run is choosing from, rather than matched
        against the wording of a progress line: the label is what the tool calls
        itself, so this is an identity, not a guess.
        """
        return self._assistant.tool_for_label(step)

    def _show_stage_on(self, page, module, step):
        """Put the module on the view that matches the step now running."""
        stage = getattr(page, 'show_run_stage', None)
        tool = self._current_tool(step)
        if not callable(stage) or not tool:
            return
        if self._staged.get(module) == tool:
            return
        try:
            view = stage(tool)
        except Exception as exc:  # noqa: BLE001 - navigation must not stop a run
            self.details.appendPlainText(f'-> could not switch view in {module}: {exc}')
            return
        self._staged[module] = tool
        if view:
            self.details.appendPlainText(f'-> {module} showing: {view}')

    def _open_actions(self, module):
        """Somewhere to click on every panel the run visits.

        The rule this enforces: a module the run brings to the front must show
        the user a result and a way to open it. The run folder always exists, so
        there is always at least one button; the folders a step writes into and
        the newest figure are offered when they are there.
        """
        if not self._output:
            return []
        actions = [('Open run folder', str(self._output))]
        for name in _MODULE_OUTPUTS.get(module, ()):
            folder = Path(self._output) / name
            if folder.is_dir():
                actions.append((f'Open {name}', str(folder)))
        figures = self._recent_figures(limit=1)
        if figures:
            actions.append((f'Open {Path(figures[0]).name}', figures[0]))
        return actions

    def _run_config(self):
        """The configuration the running workflow is actually using.

        The child writes ``workflow_config.json`` into the run folder before it
        starts, so this is read off disk rather than carried back over the
        progress protocol. It matters because a panel opening the run's data has
        to open it the way the run does - reading an E4D file as a unified one
        produces a file list and an empty plot, which is what a panel showing
        "the same data" must not do.
        """
        from PyHydroGeophysX.agents.runtime.catalog import MODEL_BUNDLE_DIR
        # No instrument unless the run names one. When nobody chose, the run
        # reads it from each file's header; filling in E4D here had the studio
        # open, say, a DAS-1 run's files with the E4D reader.
        config = {}
        if not self._output:
            return config
        try:
            import json
            path = Path(self._output) / 'workflow_config.json'
            written = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            written = None
        if isinstance(written, dict):
            config.update({k: v for k, v in written.items() if v is not None})
        # What the run has produced so far, not only what went into it: a page
        # reached halfway through a run wants the model the earlier steps
        # recovered, and the inversion writes it to a known place.
        bundle = Path(self._output) / MODEL_BUNDLE_DIR
        if bundle.is_dir():
            config['model_directory'] = str(bundle)
        return config

    def _module_page(self, module):
        """The studio page for ``module``, or None when it is not loaded."""
        if not module:
            return None
        pages = getattr(self.window(), '_pages', None)
        return pages.get(module) if isinstance(pages, dict) else None

    def _collect_figures(self, step=None):
        """Put each figure the run writes on the Live canvas as it appears.

        The canvas opens beside the timeline with the run's first figure;
        before that it would only be an empty frame.
        """
        if step is None:
            steps = self.timeline.steps()
            step = steps[-1].label if steps else ''
        added = False
        for path in reversed(self._recent_figures(limit=24)):
            added = self.canvas.add_figure(path, step) or added
        if added and self._canvas_host.isHidden():
            self._reveal_canvas(True)
        self.canvas.tick()

    def _reveal_canvas(self, show):
        """Open the canvas beside the timeline, or put it away."""
        if bool(show) != self._canvas_host.isHidden():
            return
        self._canvas_host.setVisible(bool(show))
        if show:
            total = max(2, self._live_split.width())
            self._live_split.setSizes([int(total * 0.56), int(total * 0.44)])

    def _show_finish(self, result=None, error='', stopped=False):
        """End the Live tab with the run's outcome: how it went, what it made,
        what to check, and the way to the report."""
        if self.timeline.finish_card() is not None and not stopped:
            return
        if self._output:
            self._collect_figures()
        self.route.finish()
        result = result or {}
        seconds = self._elapsed.elapsed() / 1000 if self._elapsed.isValid() else 0.0
        taken = sum(1 for card in self.timeline.steps() if card.status == 'ok')
        warnings = [str(w) for w in result.get('warnings') or []]
        reports = result.get('report_files') or {}

        def count(n, noun):
            return f'{n} {noun}' + ('' if n == 1 else 's')

        stats = [count(taken, 'step'), count(len(self.canvas.figures()), 'figure')]
        if self.files.count():
            stats.append(count(self.files.count(), 'file'))
        if self._usage_total.get('tokens'):
            stats.append(usage_text(self._usage_total['tokens'],
                                    self._usage_total.get('cost_usd') or 0.0))
        if warnings:
            stats.append(f'{len(warnings)} to check')
        actions = []
        if reports:
            actions.append('report')
        if self._recorder.recording:
            actions.append('replay')
        if self._output:
            actions.append('folder')
        gaps = warnings
        if error:
            state, title, gaps = presence.FAILED, f'{self._name()} could not complete the run', [error]
            actions.insert(0, 'log')
        elif stopped:
            state, title = presence.WAITING, f'Stopped · {self._name()} handed control back to you'
            gaps = ['The run was stopped before it finished; partial files remain in the output folder.']
        elif result.get('status') == 'incomplete':
            state, title = presence.FAILED, f'{self._name()} could not finish the run'
        elif result.get('completion', {}).get('interpretation') == 'not_run':
            state, title = presence.WAITING, 'Numerical results ready · AI interpretation not run'
        elif warnings:
            state, title = presence.WAITING, ('Report ready · needs your review' if reports
                                              else 'Finished · needs your review')
        else:
            state, title = presence.DONE, 'Report ready' if reports else 'Finished'
        report = str((reports or {}).get('report_markdown') or '')
        self._last_result = {'outcome': title, 'status': result.get('status') or
                             ('failed' if error else 'stopped' if stopped else 'success'),
                             'warnings': list(gaps), 'report': report or None,
                             'seconds': round(seconds, 1)}
        self._place_finish(state, title, seconds, stats, gaps, actions, report)
        self.tabs.setCurrentWidget(self._live_tab)

    def _place_finish(self, state, title, seconds, stats, gaps, actions, report=''):
        """Put the outcome card at the foot of the timeline.

        One call with plain arguments, so a replay can place the same card:
        ``actions`` are keys - report, replay, folder, log - not callbacks.
        """
        folder = self._replay_dir if self._replaying else self._output
        if self._replaying and report and not Path(report).is_file():
            # A recording names the report where the run wrote it; a run
            # folder moved or copied since keeps it beside the recording.
            beside = Path(self._replay_dir) / Path(report).name
            report = str(beside) if beside.is_file() else report
        # In the page's own report tab, for a replayed run too: handing the
        # Markdown file to the system sent the reader to whatever program the
        # machine associates with .md, without the figures laid out.
        read = ('Read the report', lambda: self._show_report(report))
        known = {'report': read,
                 'replay': ('Replay this run', lambda: self.start_replay(
                     self._last_replay if not self._replaying else '')),
                 'folder': ('Open output folder', lambda: self._open_path(folder)),
                 'log': ('Show raw log', lambda: self.tabs.setCurrentWidget(self.details))}
        buttons = [known[key] for key in actions or [] if key in known]
        self.timeline.add_finish(FinishCard(state, title, seconds, stats, gaps, buttons))

    # -- watching a finished run again -------------------------------------------------
    def _save_replay(self):
        """Keep the run's recording beside its results, as live_replay.json."""
        self._recorder.stop()
        if not self._output or not self._recorder.events:
            return ''
        meta = dict(getattr(self, '_recording_meta', {}) or {})
        if self._elapsed.isValid():
            meta['run_seconds'] = round(self._elapsed.elapsed() / 1000, 1)
        try:
            path = self._recorder.save(str(Path(self._output) / run_replay.FILE_NAME), meta)
        except OSError as exc:
            self.details.appendPlainText(f'Could not save the run recording: {exc}')
            return ''
        self._last_replay = path
        return path

    def _choose_replay(self):
        start = self._output or ''
        path, _ = QFileDialog.getOpenFileName(
            self, 'Replay a run', start,
            f'Run recording ({run_replay.FILE_NAME});;JSON files (*.json)')
        if path:
            self.start_replay(path)

    def start_replay(self, path=''):
        """Watch a finished run again in the Live tab, from its recording.

        Returns False, saying why, while a run is going or when the file is not
        a recording. Nothing is recomputed and no model is asked anything.
        """
        if self._worker is not None:
            self.status.setText('A run is going · Replay it once it has finished.')
            return False
        path = path or self._last_replay
        try:
            recording = run_replay.load(path)
        except (OSError, ValueError) as exc:
            self.status.setText(f'Could not open the recording · {exc}')
            return False
        self._end_replay()
        self._replaying = True
        self._replay_dir = str(Path(path).parent)
        self.canvas.set_show_age(False)
        player = run_replay.ReplayPlayer(recording, self._recorder.objects(),
                                         self._replay_substitute, self)
        player.moved.connect(self.replay_bar.show_position)
        player.ended.connect(lambda: self.replay_bar.set_playing(False))
        player.set_speed(self.replay_bar.speed_value())
        self._player = player
        meta = recording.get('meta') or {}
        self.replay_bar.set_run_length(float(meta.get('run_seconds')
                                             or player.real_time(player.length())))
        self.replay_bar.setVisible(True)
        self.route.setVisible(True)
        self.steer.setVisible(False)
        self.tabs.setCurrentWidget(self._live_tab)
        player.seek(0.0)
        player.play()
        self.replay_bar.set_playing(True)
        who = meta.get('name') or 'The assistant'
        when = f" on {meta['started']}" if meta.get('started') else ''
        self.status.setText(f"Replaying {who}'s run{when} · nothing is recomputed.")
        return True

    @staticmethod
    def _replay_substitute(target, call, args, kwargs):
        """Stand-ins for what a recording cannot hold: a question's buttons do nothing."""
        if target == 'timeline' and call == 'ask':
            args = list(args) + [None] * (4 - len(args))
            args[3] = lambda _decision: None
        return args, kwargs

    def _replay_play(self, play):
        if self._player is None:
            return
        if play:
            self._player.play()
        else:
            self._player.pause()
        self.replay_bar.set_playing(bool(play))

    def _replay_speed(self, speed):
        if self._player is not None:
            self._player.set_speed(speed)

    def _replay_seek(self, position):
        if self._player is None:
            return
        self._player.pause()
        self.replay_bar.set_playing(False)
        self._player.seek(position)

    def _save_frame(self):
        """Save the Live tab as it looks now - without the replay bar - as a PNG."""
        from PySide6.QtCore import QRect

        folder = Path(self._replay_dir or self._output or '.')
        top = self.replay_bar.height() if self.replay_bar.isVisible() else 0
        area = QRect(0, top, self._live_tab.width(), self._live_tab.height() - top)
        index = 1
        while (folder / f'replay_frame_{index:02d}.png').exists():
            index += 1
        path = folder / f'replay_frame_{index:02d}.png'
        if self._live_tab.grab(area).save(str(path)):
            self.status.setText(f'Saved the frame · {path}')
        else:
            self.status.setText(f'Could not save the frame to {folder}.')

    def _end_replay(self):
        """Close the replay, leaving the run shown as it ended."""
        if self._player is not None:
            self._player.finish()
            self._player.deleteLater()
            self._player = None
        self._replaying = False
        self.canvas.set_show_age(True)
        self.replay_bar.setVisible(False)

    def _recent_figures(self, limit=4, since=None):
        """The newest figures this run has written, most recent first.

        Found by looking at the run directory rather than by asking the child
        what it produced: the layout differs per method and per branch, while
        "an image file that did not exist when this run started" is the same
        question everywhere and cannot fall out of date. ``since`` narrows it to
        what was written after a given moment - one step's figures, for its card.
        """
        if not self._output:
            return []
        since = self._run_started_at if since is None else since
        from PyHydroGeophysX.qt_apps.modules.base import FIGURE_SUFFIXES
        found = []
        try:
            for path in Path(self._output).rglob('*'):
                if path.suffix.lower() not in FIGURE_SUFFIXES or not path.is_file():
                    continue
                stat = path.stat()
                if stat.st_mtime < since or not stat.st_size:
                    continue
                found.append((stat.st_mtime, str(path)))
        except OSError:
            return []
        found.sort(reverse=True)
        return [path for _, path in found[:limit]]

    def _refresh_activity(self):
        """Put any figures written since the last tick in front of the user."""
        module = getattr(self, '_current_followed', None)
        page = self._module_page(module)
        if page is not None and hasattr(page, 'show_run_activity'):
            page.show_run_activity(self._latest_step, self._latest_detail,
                                   self._recent_figures(),
                                   self._open_actions(module))

    def _on_asked(self, event):
        """The run has stopped and needs an answer: show the choices.

        Two things arrive here and they are deliberately rendered the same way,
        because to the user they are the same experience - the run paused and
        wants a decision:

        - ``approve``: step-by-step is about to run a step. Approve it, leave it
          out, or stop the run here.
        - ``question``: a tool found something it cannot decide for itself - two
          files disagreeing about their origin, a conversion with no
          calibration - and is offering the defensible options.

        The question is asked in the Live timeline, where the run is
        being watched, and the Workflow page is brought back to the front if
        the user has gone elsewhere: the run cannot go on until it is answered.
        With follow-along switched on it is asked on the module doing the work
        instead, so the choice is made while looking at the panel it concerns.
        """
        kind = str(event.get('event') or 'approve')
        self._clear_pause()
        if kind == 'question':
            options = [dict(option) for option in (event.get('options') or [])]
            prompt = str(event.get('question') or 'The workflow needs a decision.')
        else:
            label = str(event.get('label') or event.get('tool') or 'the next step')
            reason = str(event.get('reason') or '')
            prompt = f'Next: {label}' + (f' — {reason}' if reason else '')
            # Kept for the answer, which names the step it set running.
            self._asked_step = label
            options = [
                {'id': 'proceed', 'label': 'Run this step',
                 'detail': 'Carry out this step and pause again before the next one.'},
                {'id': 'skip', 'label': 'Skip it',
                 'detail': 'Leave this step out; the report will record it as skipped.'},
                {'id': 'stop', 'label': 'Stop the run',
                 'detail': 'End here and keep whatever has been produced so far.'}]
        module = str(event.get('module') or '')
        self._follow_along(module, prompt)
        waiting = (f'{self._name()} is waiting for your approval · {self._asked_step}'
                   if kind != 'question' and self._asked_step else
                   f'{self._name()} is waiting for your answer')
        self.timeline.thinking(waiting, presence.WAITING)
        self._presence(presence.WAITING, f'{self._name()} is waiting for you', prompt)
        self.stepEvent.emit({'phase': 'waiting', 'label': self._asked_step or '',
                             'question': prompt})
        # A question with no options would leave the run wedged behind a bar
        # with nothing to press; the default answer is better than a deadlock.
        if not options:
            self._answer(str(event.get('default') or 'stop'))
            return
        self.status.setText('Paused · ' + prompt)
        self.details.appendPlainText(f'? {prompt}')
        # Ask where the user is looking. With follow-along on that is the panel
        # the run has just moved to - buttons back here would leave the user
        # watching a stopped run on one page with the only way to answer it on
        # another. Otherwise it is the timeline on this page.
        page = self._module_page(module)
        activity = getattr(page, '_run_activity', None) if page is not None else None
        if (module and module == self._current_followed and activity is not None
                and activity.ask(prompt, options, self._answer)):
            self._asked_on = page
            return
        if self.timeline.ask(waiting, prompt, options, self._answer):
            show = getattr(self.window(), 'show_module', None)
            if callable(show):
                try:
                    show(self.module_key)
                except Exception:  # noqa: BLE001 - navigation must not stop a run
                    pass
            self.tabs.setCurrentWidget(self._live_tab)
            return
        self.pause_label.setText('<b>Waiting for you</b><br>' + prompt)
        for option in options:
            button = QPushButton(str(option.get('label') or option.get('id')))
            if option.get('detail'):
                button.setToolTip(str(option['detail']))
            decision = str(option.get('id') or '')
            button.clicked.connect(lambda _checked=False, d=decision: self._answer(d))
            self.pause_buttons.addWidget(button)
        self.pause_buttons.addStretch(1)
        self.pause_box.setVisible(True)

    def _answer(self, decision):
        """Send the user's decision back to the running workflow."""
        step = getattr(self, '_asked_step', None)
        self._clear_pause()
        if self._worker is None:
            return
        self._worker.answer(decision)
        self.details.appendPlainText(f'-> answered: {decision}')
        self.status.setText(self._answered_status(step, decision))
        if step and decision == 'skip':
            self.timeline.step_skipped(step)
            self.route.step_done('', step, 'skipped')
        self.route.set_thinking(decision != 'stop')
        self.timeline.thinking(f'{self._name()} is stopping the run' if decision == 'stop'
                               else f'{self._name()} is carrying on')
        self._presence(presence.THINKING, self._answered_status(step, decision), '')

    @staticmethod
    def _answered_status(step, decision):
        """The status line once the user has answered, naming what happens now.

        It read "Continuing · proceed" until the approved step had finished -
        through a whole inversion and the choice of the step after it - which
        looks as though nothing started. The step is named as running at once;
        the workflow's own announcement of it follows a moment later.
        """
        if step and decision == 'proceed':
            return f'Running · {step}'
        if step and decision == 'skip':
            return f'Skipped · {step} · choosing the next step'
        if step and decision == 'stop':
            return f'Stopping · the run ends before {step}'
        return f'Continuing · {decision}'

    def _clear_pause(self):
        """Take the prompt down, wherever it was put, and forget its buttons."""
        self._asked_step = None
        asked_on = getattr(self, '_asked_on', None)
        activity = getattr(asked_on, '_run_activity', None) if asked_on else None
        if activity is not None:
            activity.clear_question()
        self._asked_on = None
        self.timeline.clear_question()
        while self.pause_buttons.count():
            item = self.pause_buttons.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.pause_label.clear()
        self.pause_box.setVisible(False)

    def _on_log(self, text):
        self.details.appendPlainText(text)
        if text.strip():
            self.live_detail.setText(text.strip()[-220:])

    def _succeeded(self, result):
        self.next_step.hide()
        self._continuation_result = None
        if result.get('status') == 'classified':
            self.finish_persisted_run(result, 'unified')
            self._show_catalog(result['catalog'])
            self.tabs.setCurrentWidget(self._data_tab)
            self.progress.setValue(100)
            self.timeline.finish()
            self.route.finish()
            self.route.setVisible(False)
            self._presence(presence.WAITING, f'{self._name()} has sorted your files',
                           f'Check the roles in Data, then send \u201ccontinue\u201d to {self._name()}.',
                           glow_state=presence.DONE)
            message = 'File classification ready. Review/edit the roles in Workflow, then send “continue”. Unknown files must be assigned or ignored.'
            if result['catalog'].get('warnings'):
                message += '\n' + '\n'.join(result['catalog']['warnings'])
            self.status.setText(message)
            self.workflowFinished.emit(message)
            return
        self.finish_persisted_run(result, 'unified')
        self.report_result(result)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        reports = result.get('report_files') or {}
        markdown = reports.get('report_markdown')
        self.report.setPlainText(str(result.get('interpretation') or 'Computation finished. See output files for available results.'))
        self._show_report(markdown, switch=False)
        for path in sorted(Path(self._output).rglob('*')):
            if path.is_file():
                item = QListWidgetItem(str(path.relative_to(self._output)))
                item.setData(Qt.UserRole, str(path))
                self.files.addItem(item)
        self.status.setText('Complete · Report and output files are ready.' if reports else
                            'Computation complete · No report was generated; inspect Activity and output files.')
        # The page says how many things need checking and where they are; the
        # list itself goes to the assistant's conversation. Spelled out here,
        # four warnings filled the banner and the line under the report.
        warnings = [str(w) for w in result.get('warnings') or []]
        incomplete = result.get('status') == 'incomplete'
        where = 'listed in the assistant panel'
        if incomplete:
            self.status.setText(f'Did not complete · {_count(max(1, len(warnings)), "problem")}, '
                                f'{where}.')
        elif warnings:
            self.status.setText(f'Complete · {_count(len(warnings), "thing")} to check before '
                                f'relying on the results, {where}.')
        self.timeline.finish()
        self._presence(presence.FAILED if incomplete else presence.WAITING if warnings else presence.DONE,
                       f'{self._name()} could not finish the run' if incomplete else
                       (f'{self._name()} finished · needs your review' if warnings
                        else f'{self._name()} finished the report'),
                       self.status.text().split(' · ', 1)[-1])
        # The outcome is shown where the run was watched, with the report one
        # click away, rather than swapping the Live tab out from under the user.
        self._show_finish(result)
        if result.get('completion'):
            completion = result['completion']
            self.live_detail.setText('Source files preserved · Explore the models or save this run to Project history.'
                                     if result.get('source_files_unchanged') is True else
                                     'Explore the available outputs or save this run to Project history.')
            wording = {'complete': 'Complete', 'incomplete': 'Incomplete', 'generated': 'Generated',
                       'not_run': 'Not run', 'NOT_REVIEWED': 'Not run', 'ACCEPT': 'Accepted',
                       'REVISE_REPORT': 'Changes required', 'INSUFFICIENT_EVIDENCE': 'Insufficient evidence'}
            self.status.setText(' · '.join(f'{name.title()}: {wording.get(value, value)}' for name, value in completion.items()))
            if incomplete:
                self.status.setText('Workflow incomplete · ' + self.status.text())
            self.result_summary.setText(self.status.text())
            exports = result.get('exports') or {}
            figures = exports.get('figures') or {}
            self.view_result.setEnabled(bool(exports.get('model')))
            self.view_fit.setEnabled(any('data fit' in str(k).lower() for k in figures))
            self.view_fit.setToolTip('Open the observation, prediction and residual maps.' if self.view_fit.isEnabled()
                                     else 'No observation/prediction files are available for this run.')
            self.read_report.setEnabled(bool(markdown))
            self._result_capabilities = {button: button.isEnabled() for button in
                                         (self.view_result, self.view_fit, self.read_report)}
            self.read_report.setText('Read interpretation' if completion.get('interpretation') == 'generated'
                                    else 'Read numerical summary')
            if result.get('continuation') and callable(getattr(self._workflow_setup, 'prepare_continuation', None)):
                self._continuation_result = result
                self._continuation_assistant = self._assistant
                self.next_step.show()
            self._sync_result_storage()
            for widget in (*self._result_buttons, self.result_state, self.result_summary):
                widget.show()
            if getattr(self._assistant, 'focused_workspace', False):
                self.read_report.hide()
            if self._workflow_setup is not None:
                self.tabs.setCurrentWidget(self._result_page)
        summary = str(result.get('interpretation') or '')[:1500]
        self.workflowFinished.emit(_finish_message(self.status.text(), warnings, incomplete)
                                   + ('\n\n' + summary if summary else ''))

    def _failed(self, error):
        self.progress.setRange(0, 100)
        self.fail_persisted_run(error, 'unified')
        self.status.setText(f'Could not complete · {error}')
        self.timeline.finish()
        self._show_finish(error=str(error))
        self._presence(presence.FAILED, f'{self._name()} could not complete the run', str(error))
        self.details.appendPlainText(error)
        self.report.setPlainText(f'The workflow did not complete.\n\n{error}\n\nYour inputs are retained. Adjust them and run again.')
        self.log(error, 'error')
        self.workflowFinished.emit(f'Workflow failed: {error}')

    def _cancel(self):
        if self._worker:
            self.status.setText('Stopping workflow…')
            self.stop.setEnabled(False)
            self._worker.cancel()

    def _finished(self):
        self.progress.setRange(0, 100)
        self._clock.stop()
        # Nothing is left to answer once the child is gone; a prompt bar that
        # outlives its run would send a decision nowhere, and a module still
        # claiming to be worked on would be claiming it forever.
        self._clear_pause()
        page = self._module_page(getattr(self, '_current_followed', None))
        if page is not None and hasattr(page, 'hide_run_activity'):
            page.hide_run_activity()
        self._current_followed = None
        if self._elapsed.isValid():
            self.elapsed_label.setText(f'Finished · {self._elapsed.elapsed() / 1000:.1f}s elapsed')
            self.header.set_clock(self._elapsed.elapsed() / 1000, self._steps_done(), running=False)
        self.timeline.finish()
        self.route.finish()
        if self._worker and self._worker.is_cancelled():
            self._show_finish(stopped=True)
            self._presence(presence.IDLE, f'Stopped · {self._name()} handed control back to you',
                           'Partial files remain in the output folder.')
            self.cancel_persisted_run('Stopped by user', 'unified')
            self.status.setText('Stopped · Partial files remain in the output folder. You can edit and retry.')
            self.report.setPlainText('Workflow stopped. Partial outputs may be available in the output folder.')
            self.workflowFinished.emit('Workflow stopped. Inputs are retained for retry.')
        self.steer.setVisible(False)
        if self._recorder.recording:
            self._save_replay()
        self.replay_button.setEnabled(True)
        self._worker = None
        self._setup.setEnabled(True)
        self.run.setEnabled(True)
        self.step_through.setEnabled(True)
        self.stop.setEnabled(False)
        if self._workflow_setup is not None and hasattr(self._workflow_setup, 'detailsChanged'):
            self.stop.hide()

    #: Files the page shows itself rather than handing to another program.
    _REPORT_SUFFIXES = {'.md', '.markdown'}
    _IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg'}

    def _show_report(self, path, switch=True):
        """Show a Markdown report in the report tab, its figures resolved beside it."""
        report = Path(path) if path else None
        if report is not None and report.is_file():
            try:
                text = report.read_text(encoding='utf-8', errors='replace')
            except OSError as exc:
                self.details.appendPlainText(
                    f'Could not preview report: {exc}. Open the output folder to inspect results.')
            else:
                self.report.document().setBaseUrl(
                    QUrl.fromLocalFile(str(report.resolve().parent) + '/'))
                self.report.setMarkdown(text)
        if switch:
            self.tabs.setCurrentWidget(self._result_page)

    def _show_image(self, path):
        """A figure in a window of the studio's own, scaled to fit the screen."""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QDialog

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._open_path(path)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(Path(path).name)
        screen = (self.screen() or dialog.screen()).availableGeometry()
        fitted = pixmap.scaled(int(screen.width() * 0.85), int(screen.height() * 0.8),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation) \
            if pixmap.width() > screen.width() * 0.85 or pixmap.height() > screen.height() * 0.8 \
            else pixmap
        label = QLabel()
        label.setPixmap(fitted)
        label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        box = QVBoxLayout(dialog)
        box.addWidget(scroll)
        dialog.resize(fitted.width() + 40, fitted.height() + 40)
        dialog.show()

    def _preview(self, path):
        """A report or figure inside the studio; any other file in its own program."""
        suffix = Path(str(path)).suffix.lower()
        if suffix in self._REPORT_SUFFIXES:
            self._show_report(path)
        elif suffix in self._IMAGE_SUFFIXES:
            self._show_image(path)
        else:
            self._open_path(path)

    def _open_link(self, url):
        if url.isLocalFile():
            self._preview(url.toLocalFile())

    def _open_path(self, path):
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    # -- agent command interface ----------------------------------------------
    #: The page's tabs by the names the assistant uses.
    _AGENT_TABS = ('data', 'live', 'report', 'files', 'log')

    def agent_describe(self):
        roles = [{'role': key, 'label': label,
                  'takes_several_in_order': key in self._assistant.ordered_roles}
                 for label, key in self._assistant.input_roles]
        setup_actions = []
        if self._workflow_setup is not None:
            describe = getattr(self._workflow_setup, 'agent_actions', None)
            if callable(describe):
                setup_actions = list(describe() or [])
        return {
            'module': self.module_key,
            'title': self.module_title,
            'state': self._agent_status(),
            'input_roles': roles,
            'actions': [
                {'name': 'get_status', 'args': {},
                 'desc': ('The run: whether one is going or paused, its request and inputs, '
                          'the steps taken with their status and time, the steps still '
                          'ahead, tokens and cost so far, and how the last run ended '
                          '(its warnings and report path).')},
                {'name': 'get_report', 'args': {'max_chars': 'int (default 6000)'},
                 'desc': 'The text of the report the last run wrote, as shown in Results & report.'},
                {'name': 'add_input', 'args': {'role': [r['role'] for r in roles],
                                               'path': 'str', 'paths': 'list of str'},
                 'desc': ('Give an input role a file (or, for a role that takes several '
                          'in order, files appended in order).')},
                {'name': 'remove_input', 'args': {'role': 'str', 'path': 'str (optional)'},
                 'desc': 'Remove one file from a role, or the whole role without a path.'},
                {'name': 'clear_inputs', 'args': {}, 'desc': 'Remove every input.'},
                {'name': 'set_options',
                 'args': {'follow_along': 'bool', 'approve_each_step': 'bool'},
                 'desc': ('follow_along brings each module to the front as the run works '
                          'in it; approve_each_step makes the next run wait for approval '
                          'before every step.')},
                {'name': 'show_tab', 'args': {'tab': list(self._AGENT_TABS)},
                 'desc': 'Show Data, the Live timeline, Results & report, Output files or Raw log.'},
                {'name': 'pause_run', 'args': {},
                 'desc': 'Hold the running workflow before its next step (the running step finishes).'},
                {'name': 'resume_run', 'args': {}, 'desc': 'Let a paused run carry on.'},
                {'name': 'send_note', 'args': {'text': 'str'},
                 'desc': ('Tell the running workflow something; it is read before the next '
                          'decision and may change an adjustable setting.')},
                {'name': 'stop_run', 'args': {},
                 'desc': 'Stop the running workflow. Inputs and partial files are kept.'},
                {'name': 'replay_run', 'args': {'path': 'str (optional live_replay.json)'},
                 'desc': 'Play a finished run back in the Live tab; the last run by default.'},
            ] + setup_actions,
            'note': ('Runs are started from the assistant panel (Auto to report, or '
                     'Step-by-step); this page holds their inputs, follows them live and '
                     'shows the report.'),
        }

    def agent_apply(self, action, args):
        args = args or {}
        handlers = {
            'get_status': self._agent_status,
            'get_report': lambda: self._agent_report(args.get('max_chars', 6000)),
            'add_input': lambda: self._agent_add_input(
                args.get('role'), args.get('paths') or ([args['path']] if args.get('path') else [])),
            'remove_input': lambda: self._agent_remove_input(args.get('role'), args.get('path')),
            'clear_inputs': self._agent_clear_inputs,
            'set_options': lambda: self._agent_set_options(args),
            'show_tab': lambda: self._agent_show_tab(args.get('tab')),
            'pause_run': lambda: self._agent_steer('pause'),
            'resume_run': lambda: self._agent_steer('resume'),
            'send_note': lambda: self._agent_steer('note', args.get('text')),
            'stop_run': self._agent_stop,
            'replay_run': lambda: self._agent_replay(args.get('path')),
        }
        handler = handlers.get(action)
        if handler is not None:
            return handler()
        extension = getattr(self._workflow_setup, 'agent_apply', None)
        if not callable(extension):
            return {'status': 'failed', 'error': f"Unknown action '{action}'.",
                    'valid_actions': list(handlers)}
        result = extension(action, args)
        if not isinstance(result, dict):
            result = {'status': 'ok', 'result': result}
        supplied = result.get('inputs')
        if isinstance(supplied, dict):
            self._inputs = dict(supplied)
            self._refresh_inputs()
        if result.pop('start_workflow', False):
            request = str(result.get('request') or self._workflow_setup.request()).strip()
            self._request_text = request
            self.goal.setText('Goal: ' + request)
            self.startAIRequested.emit(request)
            result['launch_requested'] = True
        return result

    def _agent_status(self):
        usage = self._usage_total or {}
        return {
            'status': 'ok',
            'assistant': self._assistant.key,
            'assistant_name': self._name(),
            'running': self._worker is not None,
            'paused': self._worker is not None and self.steer.state() == SteerBar.PAUSED,
            'replaying': self._replaying,
            'request': self._request_text or '',
            'inputs': {role: (list(value) if isinstance(value, list) else value)
                       for role, value in self._inputs.items()},
            'output_dir': str(self._output or ''),
            'progress_percent': self.progress.value() if self.progress.maximum() else None,
            'status_line': self.status.text(),
            'steps': [{'label': card.label, 'status': card.status,
                       'seconds': round(card.elapsed(), 1)} for card in self.timeline.steps()],
            'ahead': [node['label'] for node in self.route.nodes() if node['status'] == 'ahead'],
            'usage': ({'tokens': int(usage.get('tokens') or 0),
                       'cost_usd': float(usage.get('cost_usd') or 0.0),
                       'calls': int(usage.get('calls') or 0)} if usage else None),
            'last_result': self._last_result,
            'recording': self._last_replay or None,
            'tab': self._agent_tab_name(),
            'options': {'follow_along': self.follow.isChecked(),
                        'approve_each_step': self.step_through.isChecked()},
        }

    def _agent_tab_name(self):
        current = self.tabs.currentWidget()
        widgets = (self._data_tab, self._live_tab, self._result_page, self.files, self.details)
        return next((name for name, widget in zip(self._AGENT_TABS, widgets)
                     if widget is current), '')

    def _agent_report(self, max_chars):
        try:
            limit = max(200, int(max_chars))
        except (TypeError, ValueError):
            limit = 6000
        text = self.report.toPlainText()
        return {'status': 'ok', 'report': text[:limit], 'truncated': len(text) > limit,
                'report_file': (self._last_result or {}).get('report')}

    def _agent_add_input(self, role, paths):
        if self._worker is not None:
            return {'status': 'failed', 'error': 'A run is going; add inputs once it has finished.'}
        roles = [key for _label, key in self._assistant.input_roles]
        if role not in roles:
            return {'status': 'failed', 'error': f"Unknown role '{role}'.", 'roles': roles}
        paths = [str(p) for p in paths or [] if str(p).strip()]
        if not paths:
            return {'status': 'failed', 'error': "Provide 'path' or 'paths'."}
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            return {'status': 'failed', 'error': f'Not found: {", ".join(missing)}'}
        if role.endswith('_dir') and not all(Path(p).is_dir() for p in paths):
            return {'status': 'failed', 'error': f"Role '{role}' takes a folder."}
        problem = self._add_inputs(role, paths)
        if problem:
            return {'status': 'failed', 'error': problem}
        return {'status': 'ok', 'inputs': self._agent_status()['inputs']}

    def _agent_remove_input(self, role, path=None):
        if self._worker is not None:
            return {'status': 'failed', 'error': 'A run is going; change inputs once it has finished.'}
        if role not in self._inputs:
            return {'status': 'failed', 'error': f"No input for role '{role}'.",
                    'inputs': list(self._inputs)}
        value = self._inputs[role]
        if path and isinstance(value, list) and len(value) > 1:
            if path not in value:
                return {'status': 'failed', 'error': f'{path} is not among {role}.'}
            value.remove(path)
        elif path and not isinstance(value, list) and str(value) != str(path):
            return {'status': 'failed', 'error': f'{role} holds {value}, not {path}.'}
        else:
            self._inputs.pop(role)
        self._refresh_inputs()
        return {'status': 'ok', 'inputs': self._agent_status()['inputs']}

    def _agent_clear_inputs(self):
        if self._worker is not None:
            return {'status': 'failed', 'error': 'A run is going; change inputs once it has finished.'}
        self._inputs.clear()
        self._refresh_inputs()
        return {'status': 'ok', 'inputs': {}}

    def _agent_set_options(self, args):
        changed = {}
        if 'follow_along' in args:
            self.follow.setChecked(bool(args['follow_along']))
            changed['follow_along'] = self.follow.isChecked()
        if 'approve_each_step' in args:
            if self._worker is not None:
                return {'status': 'failed',
                        'error': 'approve_each_step applies to the next run; one is going now.'}
            self.step_through.setChecked(bool(args['approve_each_step']))
            changed['approve_each_step'] = self.step_through.isChecked()
        if not changed:
            return {'status': 'failed', 'error': 'Give follow_along and/or approve_each_step.'}
        return {'status': 'ok', 'options': changed}

    def _agent_show_tab(self, tab):
        widgets = dict(zip(self._AGENT_TABS, (self._data_tab, self._live_tab, self._result_page,
                                              self.files, self.details)))
        widget = widgets.get(str(tab or '').lower())
        if widget is None:
            return {'status': 'failed', 'error': f"Unknown tab '{tab}'.",
                    'tabs': list(self._AGENT_TABS)}
        self.tabs.setCurrentWidget(widget)
        return {'status': 'ok', 'tab': tab}

    def _agent_steer(self, what, text=None):
        if self._worker is None:
            return {'status': 'failed', 'error': 'No run is going.'}
        if what == 'note':
            text = str(text or '').strip()
            if not text:
                return {'status': 'failed', 'error': "Provide 'text'."}
            self._send_note(text)
            return {'status': 'ok', 'note': text,
                    'detail': 'Read before the next decision; the timeline shows what came of it.'}
        if what == 'pause':
            if self.steer.state() != SteerBar.RUNNING:
                return {'status': 'ok', 'detail': 'Already pausing or paused.'}
            self.steer.set_state(SteerBar.PAUSING)
            self._request_pause(True)
            return {'status': 'ok', 'detail': 'Pausing after the step that is running.'}
        if self.steer.state() != SteerBar.PAUSED:
            return {'status': 'failed', 'error': 'The run is not paused.'}
        self._request_pause(False)
        return {'status': 'ok', 'detail': 'Carrying on.'}

    def _agent_stop(self):
        if self._worker is None:
            return {'status': 'failed', 'error': 'No run is going.'}
        self._cancel()
        return {'status': 'ok', 'detail': 'Stopping; inputs and partial files are kept.'}

    def _agent_replay(self, path=None):
        if not (path or self._last_replay):
            return {'status': 'failed', 'error': 'No recorded run yet; give the path of a live_replay.json.'}
        if not self.start_replay(str(path or '')):
            return {'status': 'failed', 'error': self.status.text()}
        return {'status': 'ok', 'detail': self.status.text()}
