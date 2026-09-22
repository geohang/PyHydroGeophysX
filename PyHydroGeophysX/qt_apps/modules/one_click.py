"""A single workspace for request, execution, and report delivery."""
from pathlib import Path
import time

from PySide6.QtCore import Qt, QUrl, Signal, QTimer, QElapsedTimer
from PySide6.QtGui import QDesktopServices, QImage, QTextCursor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView)

from .base import BaseModule
from PyHydroGeophysX.qt_apps.agent.one_click_worker import OneClickWorker

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


class OneClickModule(BaseModule):
    module_key = 'one_click'
    module_title = 'Workflow'
    workflowFinished = Signal(str)

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
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('<h2>Workflow</h2>Data, progress, results and report · Controlled by AQUAH on the right'))
        self.tabs = QTabWidget()
        setup = QWidget()
        form = QVBoxLayout(setup)
        self._request_text = ''
        self._ai_settings = {}
        self.goal = QLabel('Describe your goal in AQUAH on the right and select Auto to report.')
        self.goal.setWordWrap(True)
        form.addWidget(self.goal)
        folder_row = QHBoxLayout()
        choose_folder = QPushButton('Choose data folder · AI classification…')
        choose_folder.clicked.connect(self._choose_folder)
        folder_row.addWidget(choose_folder)
        self.folder_label = QLabel('No data folder selected')
        self.folder_label.setWordWrap(True)
        folder_row.addWidget(self.folder_label, 1)
        form.addLayout(folder_row)
        folder_note = QLabel('AQUAH scans filenames and short text previews in this folder using your selected AI model. Review file roles below before sending “continue”. No files are moved.')
        folder_note.setWordWrap(True)
        form.addWidget(folder_note)
        self.catalog_table = QTableWidget(0, 4)
        self.catalog_table.setHorizontalHeaderLabels(['File', 'Role (editable)', 'Confidence', 'Evidence'])
        self.catalog_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.catalog_table.horizontalHeader().setStretchLastSection(True)
        self.catalog_table.setVisible(False)
        form.addWidget(self.catalog_table)
        # The manual path, kept as a fallback rather than an equal alternative.
        # Choosing a folder and letting AQUAH classify it is how this is meant
        # to be used, but classification needs a model - without an API key it
        # cannot run at all - and files are not always gathered in one folder.
        manual = QLabel('<b>Or add files yourself</b><br>'
                        'For data spread across several folders, or when no AI '
                        'model is configured (folder classification needs one). '
                        'Pick what the file is, then Add data.')
        manual.setWordWrap(True)
        form.addWidget(manual)
        row = QHBoxLayout()
        self.role = QComboBox()
        for label, key in [('ERT survey', 'data_file'), ('Time-lapse ERT (ordered surveys)', 'time_lapse_files'),
                           ('Electrode coordinates', 'electrode_file'), ('Seismic travel times', 'seismic_file'),
                           ('Raw seismic SEG-Y', 'raw_seismic_file'), ('TDEM survey', 'tdem_file'),
                           ('Terrain / topography', 'topography_file'), ('Geophone coordinates', 'geophone_file'),
                           ('Reference document', 'reference_file'),
                           ('MODFLOW folder', 'modflow_dir'), ('ParFlow folder', 'parflow_dir')]:
            self.role.addItem(label, key)
        row.addWidget(self.role)
        add = QPushButton('Add data…')
        add.clicked.connect(self._choose_files)
        row.addWidget(add)
        remove = QPushButton('Remove selected')
        remove.clicked.connect(self._remove_input)
        row.addWidget(remove)
        form.addLayout(row)
        self.inputs = QListWidget()
        form.addWidget(self.inputs)
        order = QHBoxLayout()
        for label, offset in [('Move survey up', -1), ('Move survey down', 1)]:
            button = QPushButton(label)
            button.clicked.connect(lambda checked=False, delta=offset: self._move_survey(delta))
            order.addWidget(button)
        form.addLayout(order)
        note = QLabel('Time-lapse surveys run in the displayed order; select a survey and move it up or down to reorder. Your request and workflow context are sent to the selected AI provider.')
        note.setWordWrap(True)
        form.addWidget(note)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(setup)
        self.tabs.addTab(scroll, 'Data')
        self.report = _FittedReportBrowser()
        self.report.setOpenLinks(False)
        self.report.anchorClicked.connect(self._open_link)
        self.report.setPlainText('Your interpretation and report will appear here after the workflow finishes.')
        self.tabs.addTab(self.report, '2 · Results & report')
        self.files = QListWidget()
        self.files.itemDoubleClicked.connect(lambda item: self._open_path(item.data(Qt.UserRole)))
        self.tabs.addTab(self.files, 'Output files')
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.document().setMaximumBlockCount(3000)
        self.tabs.addTab(self.details, 'Activity')
        layout.addWidget(self.tabs, 1)
        self.status = QLabel('Ready · Select data and describe your goal.')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.live_detail = QLabel('Actual workflow events will appear here.')
        self.live_detail.setWordWrap(True)
        layout.addWidget(self.live_detail)
        self.elapsed_label = QLabel('')
        layout.addWidget(self.elapsed_label)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        # The run visits the module doing each piece of work, so the user
        # watches it happen instead of watching a bar. Off-switch included: a
        # window that moves under someone reading a panel is worse than one
        # that never moves.
        self.follow = QCheckBox('Follow along: show each module as it runs')
        self.follow.setChecked(True)
        self.follow.setToolTip(
            'Bring the studio module doing the current step to the front. '
            'Switching module yourself turns this off for the rest of the run.')
        self._follow_enabled = True
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
        self.run = QPushButton('Run through to report')
        self.run.clicked.connect(self._start)
        self.stop = QPushButton('Stop')
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self._cancel)
        self.folder = QPushButton('Open output folder')
        self.folder.setEnabled(False)
        self.folder.clicked.connect(lambda: self._open_path(self._output))
        self.run.hide()
        for button in (self.stop, self.folder):
            actions.addWidget(button)
        layout.addLayout(actions)
        self._setup = setup

    def _choose_folder(self):
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
            self.status.setText('Folder selected · Send your goal in AQUAH to scan and classify files.')

    def _catalog_rows(self):
        rows = []
        for index, row in enumerate(self._catalog['files']):
            rows.append({**row, 'role': self.catalog_table.cellWidget(index, 1).currentData()})
        return rows

    def _tick(self):
        self._refresh_activity()
        self.elapsed_label.setText(f'Running · {self._elapsed.elapsed() / 1000:.0f}s elapsed · Latest backend event shown above')

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
        role = self.role.currentData()
        if role.endswith('_dir'):
            path = QFileDialog.getExistingDirectory(self, 'Select model output folder')
            paths = [path] if path else []
        else:
            paths, _ = QFileDialog.getOpenFileNames(self, 'Select data files')
        if not paths:
            return
        if role == 'time_lapse_files':
            paths = list(dict.fromkeys(self._inputs.get(role, []) + paths))
            self._inputs.pop('data_file', None)
            self._inputs[role] = paths
        else:
            if len(paths) > 1:
                self.status.setText('Select one file for this role, or choose Time-lapse ERT for multiple surveys.')
                return
            if role == 'data_file':
                self._inputs.pop('time_lapse_files', None)
            self._inputs[role] = paths[0]
        self._refresh_inputs()

    def _refresh_inputs(self):
        self.inputs.clear()
        for role, value in self._inputs.items():
            for index, path in enumerate(value if isinstance(value, list) else [value], 1):
                item = QListWidgetItem(f'{role} · {index} · {path}')
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
        if role != 'time_lapse_files':
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
            self.goal.setText('Describe your goal in AQUAH on the right and select Auto to report.')

    def submit_request(self, text, settings):
        if self._worker is not None:
            return 'A workflow is already running. Follow its progress in the center or use Stop.'
        if text.strip().lower() not in {'continue', 'run', '继续', '开始'} or not self._request_text:
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
            self.status.setText('Waiting for data · Add files here, then send “continue” in AQUAH.')
            self.tabs.setCurrentIndex(0)
            return 'Add your data in the Workflow page, then send “continue” here. I have kept your goal.'
        self._start(step_mode=self.step_through.isChecked())
        self._ai_settings = {}
        if not self._worker:
            return self.status.text()
        return ('Workflow started, pausing before each step. Approve, skip or stop each one in the center.'
                if self.step_through.isChecked() else
                'Workflow started. Progress and the report appear in the center; use Stop there to cancel.')

    def _start(self, step_mode=False):
        if self._worker is not None:
            return
        request = self._request_text
        if not request or (not self._inputs and not self._data_folder):
            self.status.setText('Describe your goal and add the data to analyze first.')
            return
        for value in self._inputs.values():
            for path in value if isinstance(value, list) else [value]:
                if not Path(path).exists():
                    self.status.setText(f'Data no longer exists: {path}')
                    return
        provider = self._ai_settings.get('provider', 'openai')
        key = self._ai_settings.get('api_key')
        if not key:
            self.status.setText('Enter an API key or set the provider environment variable before running.')
            return
        try:
            handle = self.begin_persisted_run('unified', label=request[:120])
            self._output = str(handle.outputs_dir)
            payload = dict(request=request, inputs=dict(self._inputs), provider=provider,
                           model=self._ai_settings.get('model'), api_key=key, output_dir=self._output)
            payload.update({k: self._ai_settings.get(k) for k in ('reasoning_effort', 'use_rag', 'use_mcp')})
            payload['step_mode'] = bool(step_mode)
            if self._data_folder and self._catalog is None:
                payload.update(mode='classify', data_folder=self._data_folder)
            if self._catalog:
                payload['classification'] = self._catalog_rows()
            worker = OneClickWorker(payload, self)
            self._worker = self.register_worker(worker)
            worker.progress.connect(self._on_progress)
            worker.asked.connect(self._on_asked)
            worker.logged.connect(self._on_log)
            worker.succeeded.connect(self._succeeded)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._finished)
            self._setup.setEnabled(False)
            self.run.setEnabled(False)
            self.step_through.setEnabled(False)
            self.stop.setEnabled(True)
            self.folder.setEnabled(True)
            self.details.clear()
            self.files.clear()
            self.report.setPlainText('Workflow running. You can follow progress below or open Activity.')
            self.progress.setValue(0)
            self.status.setText('Starting · You can continue using other Studio modules.')
            self.tabs.setCurrentIndex(3)
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
        self.progress.setValue(max(self.progress.value(), min(99, int(fraction * 100))))
        self.status.setText(f'{step} · {details}')
        self.details.appendPlainText(f'{step}: {details}')
        self._latest_step, self._latest_detail = step, details
        self._follow_along(module, step, details)

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
        self.tabs.setCurrentIndex(3)     # Activity, while it is being written
        return 'Activity'

    def _current_tool(self, step):
        """The registered tool whose label is ``step``, or "".

        Looked up in the registry the runtime and the studio already share,
        rather than matched against the wording of a progress line: the label is
        what the tool calls itself, so this is an identity, not a guess.
        """
        try:
            from PyHydroGeophysX.agents.runtime import TOOLS
            from PyHydroGeophysX.agents.runtime import catalog  # noqa: F401
        except Exception:  # noqa: BLE001 - the studio runs without the runtime
            return ''
        for name, tool in TOOLS.items():
            if (getattr(tool, 'label', '') or name) == step:
                return name
        return ''

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
        from PyHydroGeophysX.agents.runtime.catalog import (DEFAULT_INSTRUMENT,
                                                            MODEL_BUNDLE_DIR)
        # The same default the run itself applies, taken from the run's own
        # code rather than written out a second time here.
        config = {'instrument': DEFAULT_INSTRUMENT}
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

    def _recent_figures(self, limit=4):
        """The newest figures this run has written, most recent first.

        Found by looking at the run directory rather than by asking the child
        what it produced: the layout differs per method and per branch, while
        "an image file that did not exist when this run started" is the same
        question everywhere and cannot fall out of date.
        """
        if not self._output:
            return []
        from PyHydroGeophysX.qt_apps.modules.base import FIGURE_SUFFIXES
        found = []
        try:
            for path in Path(self._output).rglob('*'):
                if path.suffix.lower() not in FIGURE_SUFFIXES or not path.is_file():
                    continue
                stat = path.stat()
                if stat.st_mtime < self._run_started_at or not stat.st_size:
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

        The module doing the work is brought to the front first, so the choice
        is made while looking at the panel it concerns rather than at a bar.
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
            options = [
                {'id': 'proceed', 'label': 'Run this step',
                 'detail': 'Carry out this step and pause again before the next one.'},
                {'id': 'skip', 'label': 'Skip it',
                 'detail': 'Leave this step out; the report will record it as skipped.'},
                {'id': 'stop', 'label': 'Stop the run',
                 'detail': 'End here and keep whatever has been produced so far.'}]
        module = str(event.get('module') or '')
        self._follow_along(module, prompt)
        # A question with no options would leave the run wedged behind a bar
        # with nothing to press; the default answer is better than a deadlock.
        if not options:
            self._answer(str(event.get('default') or 'stop'))
            return
        # Ask on the panel the run has just moved to. Putting the buttons back
        # here would leave the user watching a stopped run on one page with the
        # only way to answer it on another.
        page = self._module_page(module)
        activity = getattr(page, '_run_activity', None) if page is not None else None
        if activity is not None and activity.ask(prompt, options, self._answer):
            self._asked_on = page
            self.status.setText('Paused · ' + prompt)
            self.details.appendPlainText(f'? {prompt}')
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
        self.status.setText('Paused · ' + prompt)
        self.details.appendPlainText(f'? {prompt}')

    def _answer(self, decision):
        """Send the user's decision back to the running workflow."""
        self._clear_pause()
        if self._worker is None:
            return
        self._worker.answer(decision)
        self.details.appendPlainText(f'-> answered: {decision}')
        self.status.setText(f'Continuing · {decision}')

    def _clear_pause(self):
        """Take the prompt down, wherever it was put, and forget its buttons."""
        asked_on = getattr(self, '_asked_on', None)
        activity = getattr(asked_on, '_run_activity', None) if asked_on else None
        if activity is not None:
            activity.clear_question()
        self._asked_on = None
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
        if result.get('status') == 'classified':
            self.finish_persisted_run(result, 'unified')
            self._show_catalog(result['catalog'])
            self.tabs.setCurrentIndex(0)
            self.progress.setValue(100)
            message = 'File classification ready. Review/edit the roles in Workflow, then send “continue”. Unknown files must be assigned or ignored.'
            if result['catalog'].get('warnings'):
                message += '\n' + '\n'.join(result['catalog']['warnings'])
            self.status.setText(message)
            self.workflowFinished.emit(message)
            return
        self.finish_persisted_run(result, 'unified')
        self.report_result(result)
        self.progress.setValue(100)
        reports = result.get('report_files') or {}
        markdown = reports.get('report_markdown')
        self.report.setPlainText(str(result.get('interpretation') or 'Computation finished. See output files for available results.'))
        try:
            if markdown and Path(markdown).is_file():
                self.report.document().setBaseUrl(QUrl.fromLocalFile(str(Path(markdown).resolve().parent) + '/'))
                self.report.setMarkdown(Path(markdown).read_text(encoding='utf-8', errors='replace'))
        except OSError as exc:
            self.details.appendPlainText(f'Could not preview report: {exc}. Open the output folder to inspect results.')
        for path in sorted(Path(self._output).rglob('*')):
            if path.is_file():
                item = QListWidgetItem(str(path.relative_to(self._output)))
                item.setData(Qt.UserRole, str(path))
                self.files.addItem(item)
        self.status.setText('Complete · Report and output files are ready.' if reports else
                            'Computation complete · No report was generated; inspect Activity and output files.')
        if result.get('warnings'):
            self.status.setText('Complete · Needs review: ' + '; '.join(result['warnings']))
        self.tabs.setCurrentIndex(1)
        summary = str(result.get('interpretation') or '')[:1500]
        self.workflowFinished.emit(self.status.text() + ('\n\n' + summary if summary else ''))

    def _failed(self, error):
        self.fail_persisted_run(error, 'unified')
        self.status.setText(f'Could not complete · {error}')
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
        if self._worker and self._worker.is_cancelled():
            self.cancel_persisted_run('Stopped by user', 'unified')
            self.status.setText('Stopped · Partial files remain in the output folder. You can edit and retry.')
            self.report.setPlainText('Workflow stopped. Partial outputs may be available in the output folder.')
            self.workflowFinished.emit('Workflow stopped. Inputs are retained for retry.')
        self._worker = None
        self._setup.setEnabled(True)
        self.run.setEnabled(True)
        self.step_through.setEnabled(True)
        self.stop.setEnabled(False)

    def _open_link(self, url):
        if url.isLocalFile():
            self._open_path(url.toLocalFile())

    def _open_path(self, path):
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
