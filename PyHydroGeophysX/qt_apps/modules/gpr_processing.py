"""Ground-Penetrating Radar page: read a GPR line, clean it up, see it in depth.

Laid out like the ERT page: Data opens on the raw radargram, with loading and
signal filters on the right. Processed radargram has velocity, depth and
migration controls; Trace and Spectrum keep the filters beside their diagnostics.
Every change updates the processed line from the data as read; the Data tab
keeps the original recording visible.

Four views: Data as recorded, the processed radargram (in time, depth or
elevation), one trace before and after processing, and the line's amplitude
spectrum. The velocity is fitted by clicking the top of a diffraction
hyperbola and changing the velocity until the drawn curve lies on it.

The numerics are the package's (:mod:`PyHydroGeophysX.data_processing.gpr`).
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import gpr
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    ContentWidthScrollArea,
    Debouncer,
    select_directory,
    set_rows_enabled,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.color_range import ColorRange
from PyHydroGeophysX.qt_apps.widgets.flow_layout import FlowLayout, group
from PyHydroGeophysX.qt_apps.widgets.guided import (
    fold,
    hint,
    number_box,
)
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row
from PyHydroGeophysX.qt_apps.workers import TaskWorker
from PyHydroGeophysX.visualization.axis_units import set_length_axis, to_display_length

DATA, RADARGRAM, TRACE, SPECTRUM = "Data", "Processed radargram", "Trace", "Spectrum"
_TIME, _DEPTH, _ELEVATION = "Time", "Depth", "Elevation"
#: Where the "Use example" model lives under the source repository's examples/.
_EXAMPLE_DATA = "data"

_FORMAT_HELP = """<h3>GPR files this page reads</h3>
<ul>
<li><b>GSSI</b> <code>.DZT</code> (SIR-3000, SIR-4000, StructureScan, UtilityScan...).
Several channels: the first is read. A file recorded by time rather than by distance
has no trace spacing; set it under <i>Line geometry</i>.</li>
<li><b>Sensors &amp; Software pulseEKKO</b> <code>.DT1</code> with its <code>.HD</code>
header beside it (open either).</li>
<li><b>MALA</b> <code>.rd3</code> or <code>.rd7</code> with its <code>.rad</code>
header beside it (open any of them).</li>
<li><b>SEG-Y</b> <code>.sgy</code> exported by GPR software. SEG-Y keeps the sample
interval in whole microseconds, so GPR exports write it in picoseconds; check the
time window it reports.</li>
<li><code>.npz</code> saved by this page (<i>Save results</i>).</li>
</ul>
<p>Ground elevations for a section in elevation: a CSV or text file of
<code>distance, elevation</code> in metres, one row per point along the line.</p>
"""


class _FigurePane(QWidget):
    """A matplotlib figure with the studio's toolbar row: one of the page's views.

    ``_fig`` is the name the assistant's capture looks for, so a captured view
    is re-rendered at print resolution rather than grabbed off the screen.
    """

    def __init__(self, *extras: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(8, 5))
        self.canvas = FigureCanvasQTAgg(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        bar, self.toolbar = toolbar_row(self.canvas, self, *extras)
        layout.addWidget(bar)
        layout.addWidget(self.canvas, stretch=1)

    @property
    def figure(self):
        return self._fig

    def drawn(self) -> None:
        self.canvas.draw_idle()
        self.toolbar.update()

    def empty(self, text: str) -> None:
        self._fig.clear()
        self._fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=11,
                       color="#8e8e93", wrap=True)
        self.drawn()


class GPRProcessingModule(BaseModule):
    module_key = "gpr"
    module_title = "Ground-Penetrating Radar"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._raw: Optional[gpr.GPRProfile] = None
        self._processed: Optional[gpr.GPRProfile] = None
        self._generation = 0                 # which processing run a result belongs to
        self._apex: Optional[tuple] = None   # (x m, t ns) of the hyperbola being fitted
        self._trace = 0
        self._last_seconds = 0.0
        self._file_elevation: Optional[np.ndarray] = None
        self._elevation_source = ""
        self._debounce = Debouncer(self._reprocess, 300, self)

        root = QHBoxLayout(self)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self._tabs = self._build_views()
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._display_controls)
        root.addWidget(center, stretch=1)
        self._controls = self._build_controls()
        root.addWidget(self._controls)
        self.setAcceptDrops(True)

        length_units.notifier().changed.connect(lambda *_: self._redraw())
        self._sync_rows()
        self._sync_tab_controls()
        self._update_guide()
        self._redraw()

    # -- the views ------------------------------------------------------------
    def _build_views(self) -> QTabWidget:
        tabs = QTabWidget()
        self._vertical = QComboBox()
        self._vertical.addItems([_TIME, _DEPTH, _ELEVATION])
        self._vertical.setToolTip("Depth and elevation use the velocity on the right; "
                                  "elevation also needs the ground elevations.")
        self._vertical.currentTextChanged.connect(lambda *_: self._redraw())
        self._colormap = cmaps.ColormapChooser(cmaps.GPR, "gray",
                                               shared=cmaps.colormap_settings(self.state))
        self._colormap.colormapChanged.connect(lambda *_: self._redraw())
        self._range = ColorRange(decimals=4, what="the amplitude colours")
        self._range.changed.connect(self._redraw)
        self._data_view = _FigurePane()
        self._radargram = _FigurePane()
        # Shared display controls below the tabs, like ERT's colour controls.
        # Keep them off the navigation toolbar so its actions never overflow.
        self._display_controls = QWidget()
        display = FlowLayout(self._display_controls, spacing=10)
        display.setContentsMargins(6, 0, 6, 4)
        display.addWidget(group("Vertical", self._vertical))
        display.addWidget(group(self._colormap, self._range))
        for pane in (self._data_view, self._radargram):
            pane.canvas.mpl_connect("button_press_event", self._on_radargram_click)

        self._trace_box = QSpinBox()
        self._trace_box.setRange(1, 1)
        self._trace_box.setToolTip("The trace shown; clicking the radargram picks one too.")
        self._trace_box.valueChanged.connect(self._on_trace_box)
        self._trace_view = _FigurePane()
        trace_row = FlowLayout(spacing=10)
        trace_row.setContentsMargins(6, 0, 6, 4)
        trace_row.addWidget(group("Trace", self._trace_box))
        self._trace_view.layout().addLayout(trace_row)
        self._spectrum = _FigurePane()
        tabs.addTab(self._data_view, DATA)
        tabs.addTab(self._radargram, RADARGRAM)
        tabs.addTab(self._trace_view, TRACE)
        tabs.addTab(self._spectrum, SPECTRUM)
        tabs.currentChanged.connect(self._on_tab_changed)
        return tabs

    # -- the right column -----------------------------------------------------
    def _build_controls(self) -> QWidget:
        controls = QWidget()
        outer = QVBoxLayout(controls)
        outer.setContentsMargins(0, 0, 0, 0)
        self._controls_scroll = ContentWidthScrollArea(minimum=450, maximum=500)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self._load_group = self._build_load_group()
        self._cleanup_group = self._build_cleanup_group()
        self._velocity_group = self._build_velocity_group()
        self._migration_group = self._build_migration_group()
        self._gain_group = self._build_gain_group()
        for box in (self._load_group, self._cleanup_group, self._velocity_group,
                    self._migration_group, self._gain_group):
            layout.addWidget(box)
        layout.addStretch(1)
        self._controls_scroll.setWidget(panel)
        outer.addWidget(self._controls_scroll, stretch=1)
        outer.addWidget(self._build_save_group())
        return controls

    def _on_tab_changed(self, *_args: Any) -> None:
        if not hasattr(self, "_controls"):
            return
        if self._tabs.currentWidget() is not self._radargram:
            self._pick.setChecked(False)
        self._sync_tab_controls()
        self._redraw()

    def _sync_tab_controls(self) -> None:
        current = self._tabs.currentWidget()
        data = current is self._data_view
        section = current is self._radargram
        for box, visible in (
            (self._load_group, data),
            (self._cleanup_group, not section),
            (self._velocity_group, section),
            (self._migration_group, section),
        ):
            box.setVisible(visible)
        self._display_controls.setVisible(data or section)
        self._display_controls.setEnabled(self._raw is not None)
        self._result_btn.setVisible(not section)
        self._result_btn.setEnabled(self._processed is not None)
        for box in (self._cleanup_group, self._velocity_group,
                    self._migration_group, self._gain_group):
            box.setEnabled(self._raw is not None)
        self._controls_scroll.fit_to_content()

    def _build_load_group(self) -> QGroupBox:
        box = QGroupBox("Load GPR data")
        layout = QVBoxLayout(box)
        self._source = QLineEdit()
        self._source.setReadOnly(True)
        self._source.setPlaceholderText("No GPR line loaded")
        layout.addWidget(self._source)
        row = QHBoxLayout()
        open_btn = QPushButton("Open file…")
        open_btn.setProperty("primary", True)
        open_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        open_btn.clicked.connect(self._open_file)
        example_btn = QPushButton("Use example")
        example_btn.setIcon(theme.icon("fa5s.box-open"))
        example_btn.setToolTip("Simulate a 100 MHz GPR line over the MODFLOW catchment model.")
        example_btn.clicked.connect(self._use_example)
        row.addWidget(open_btn, stretch=1)
        row.addWidget(example_btn, stretch=1)
        layout.addLayout(row)
        format_btn = QPushButton("Data format")
        format_btn.setIcon(theme.icon("fa5s.file-alt"))
        format_btn.clicked.connect(self._show_format_help)
        layout.addWidget(format_btn)
        self._info = QLabel("No line loaded.")
        self._info.setWordWrap(True)
        layout.addWidget(self._info)
        geometry, form = fold("Line geometry")
        self._spacing = number_box(0.05, 0.001, 100.0, 0.01, 4, " m",
                                   "Distance between traces. Change it for a line recorded by "
                                   "time, whose file has no distances: the traces are then "
                                   "spaced evenly.")
        self._spacing.setKeyboardTracking(False)
        self._spacing.valueChanged.connect(self._on_spacing_changed)
        form.addRow("Trace spacing", self._spacing)
        self._frequency = number_box(100.0, 1.0, 5000.0, 10.0, 0, " MHz",
                                     "The antenna's centre frequency. It sets the default "
                                     "dewow window and band-pass.")
        self._frequency.setKeyboardTracking(False)
        self._frequency.valueChanged.connect(self._on_frequency_changed)
        form.addRow("Antenna frequency", self._frequency)
        layout.addWidget(geometry)
        return box

    def _build_cleanup_group(self) -> QGroupBox:
        box = QGroupBox("Data QC / filter")
        form = QFormLayout(box)
        self._dewow = QCheckBox("Dewow (remove slow drift)")
        self._dewow.setChecked(True)
        self._dewow.setToolTip("Subtracts each trace's running mean: the low-frequency "
                               "swing a receiver adds under the first arrivals.")
        form.addRow(self._dewow)
        self._dewow_window = number_box(30.0, 0.5, 1000.0, 1.0, 1, " ns",
                                        "About three periods of the antenna (30 ns at 100 MHz). "
                                        "A shorter window cuts into the signal.")
        form.addRow("    Window", self._dewow_window)

        self._time_zero = QCheckBox("Time-zero correction")
        self._time_zero.setChecked(True)
        self._time_zero.setToolTip("Moves time zero onto the direct wave, so times and depths "
                                   "count from the ground surface. The samples before it are "
                                   "dropped.")
        form.addRow(self._time_zero)
        self._zero_mode = QComboBox()
        self._zero_mode.addItem("Pick automatically", "auto")
        self._zero_mode.addItem("Type it in", "manual")
        form.addRow("    Time zero", self._zero_mode)
        self._zero_time = number_box(0.0, -1000.0, 10000.0, 0.5, 2, " ns",
                                     "The time on the record that becomes zero.")
        form.addRow("    At", self._zero_time)

        self._bandpass = QCheckBox("Band-pass filter")
        self._bandpass.setToolTip("Keeps the frequencies the antenna sends and removes the "
                                  "rest. Check the band on the Spectrum view.")
        form.addRow(self._bandpass)
        self._low = number_box(25.0, 0.1, 10000.0, 5.0, 1, " MHz")
        form.addRow("    Low", self._low)
        self._high = number_box(250.0, 1.0, 20000.0, 10.0, 1, " MHz")
        form.addRow("    High", self._high)

        self._background = QCheckBox("Background removal")
        self._background.setChecked(True)
        self._background.setToolTip("Subtracts the average trace: removes the antenna's "
                                    "ringing and the direct wave, which repeat on every trace. "
                                    "It also removes flat reflectors as long as the line.")
        form.addRow(self._background)
        self._bg_window = QSpinBox()
        self._bg_window.setRange(0, 100000)
        self._bg_window.setSpecialValueText("whole line")
        self._bg_window.setToolTip("How many neighbouring traces are averaged. The whole line "
                                   "is usual; a few hundred traces keep long dipping layers.")
        form.addRow("    Average over", self._bg_window)
        for check in (self._dewow, self._time_zero, self._bandpass, self._background):
            check.toggled.connect(self._on_settings_changed)
        for widget in (self._dewow_window, self._zero_time, self._low, self._high):
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(self._on_settings_changed)
        self._zero_mode.currentIndexChanged.connect(self._on_settings_changed)
        self._bg_window.valueChanged.connect(self._on_settings_changed)
        self._zero_note = hint("")
        form.addRow(self._zero_note)
        form.addRow(hint("Changes update the processed radargram automatically."))
        return box

    def _build_velocity_group(self) -> QGroupBox:
        box = QGroupBox("Velocity / depth")
        form = QFormLayout(box)
        self._velocity = number_box(0.1, 0.01, 0.3, 0.002, 4, " m/ns",
                                    "Radar velocity in the ground; depth = velocity x time / 2. "
                                    "Dry sand is about 0.15, wet sand 0.06, clay 0.07-0.09.")
        self._velocity.valueChanged.connect(self._on_velocity_changed)
        form.addRow("Velocity", self._velocity)
        self._permittivity = number_box(9.0, 1.0, 81.0, 0.5, 2, "",
                                        "The relative permittivity the velocity stands for.")
        self._permittivity.valueChanged.connect(self._on_permittivity_changed)
        form.addRow("Relative permittivity", self._permittivity)
        self._pick = QPushButton("Fit a hyperbola")
        self._pick.setCheckable(True)
        self._pick.setIcon(theme.icon("fa5s.crosshairs"))
        self._pick.setToolTip("Then click the top of a hyperbola on the radargram.")
        self._pick.toggled.connect(self._on_pick_toggled)
        form.addRow(self._pick)
        self._pick_note = hint("")
        form.addRow(self._pick_note)
        elevations = QPushButton("Ground elevations…")
        elevations.setToolTip("A CSV of distance and elevation (m) along the line, for the "
                              "Elevation view.")
        elevations.clicked.connect(self._load_elevations)
        form.addRow(elevations)
        self._elevation_note = hint("No ground elevations: the Elevation view is off.")
        form.addRow(self._elevation_note)
        return box

    def _build_migration_group(self) -> QGroupBox:
        box = QGroupBox("Migration")
        form = QFormLayout(box)
        self._migrate = QCheckBox("F-K migration")
        self._migrate.setToolTip("Moves each diffraction back to the point that made it, with "
                                 "the velocity above. Fit the velocity first: a wrong one "
                                 "leaves smiles or frowns.")
        self._migrate.toggled.connect(self._on_settings_changed)
        form.addRow(self._migrate)
        form.addRow(hint("Uses the velocity above, the same everywhere on the line."))
        return box

    def _build_gain_group(self) -> QGroupBox:
        box = QGroupBox("Gain")
        form = QFormLayout(box)
        self._gain = QComboBox()
        self._gain.addItem("None", "none")
        self._gain.addItem("Power of time", "tpow")
        self._gain.addItem("Automatic gain control", "agc")
        self._gain.setCurrentIndex(2)
        self._gain.setToolTip("Brings up the late, weak reflections so they can be seen. "
                              "Automatic gain makes everything about as strong, so amplitudes "
                              "are no longer comparable.")
        self._gain.currentIndexChanged.connect(self._on_settings_changed)
        form.addRow("Gain", self._gain)
        self._power = number_box(1.5, 0.1, 4.0, 0.1, 2, "", "Amplitude x time^power.")
        form.addRow("    Power", self._power)
        self._agc_window = number_box(25.0, 1.0, 2000.0, 5.0, 1, " ns",
                                      "Each sample is divided by the strength around it, "
                                      "over this window.")
        form.addRow("    Window", self._agc_window)
        for widget in (self._power, self._agc_window):
            widget.setKeyboardTracking(False)
            widget.valueChanged.connect(self._on_settings_changed)
        return box

    def _build_save_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout(box)
        self._result_btn = QPushButton("View processed radargram")
        self._result_btn.setIcon(theme.icon("fa5s.chart-area", color="#ffffff"))
        self._result_btn.setProperty("primary", True)
        self._result_btn.clicked.connect(lambda: self._tabs.setCurrentWidget(self._radargram))
        layout.addWidget(self._result_btn)
        self._save_btn = QPushButton("Save results…")
        self._save_btn.setIcon(theme.icon("fa5s.save"))
        self._save_btn.setToolTip("Writes the processed line (.npz), the radargram as shown "
                                  "(.png) and the processing recipe (.json) to a folder.")
        self._save_btn.clicked.connect(lambda: self._save_results())
        self._save_btn.setEnabled(False)
        layout.addWidget(self._save_btn)
        self._status = hint("")
        layout.addWidget(self._status)
        return box

    # -- settings -------------------------------------------------------------
    def _sync_rows(self) -> None:
        set_rows_enabled([self._dewow_window], self._dewow.isChecked())
        set_rows_enabled([self._zero_mode], self._time_zero.isChecked())
        set_rows_enabled([self._zero_time], self._time_zero.isChecked()
                         and self._zero_mode.currentData() == "manual")
        set_rows_enabled([self._low, self._high], self._bandpass.isChecked())
        set_rows_enabled([self._bg_window], self._background.isChecked())
        kind = self._gain.currentData()
        set_rows_enabled([self._power], kind == "tpow")
        set_rows_enabled([self._agc_window], kind == "agc")
        self._vertical.model().item(2).setEnabled(self._elevation() is not None)
        if self._vertical.currentText() == _ELEVATION and self._elevation() is None:
            self._vertical.setCurrentText(_DEPTH)

    def steps(self) -> List[Dict[str, Any]]:
        """The recipe the right column describes, as :func:`gpr.process` takes it."""
        steps: List[Dict[str, Any]] = []
        if self._dewow.isChecked():
            steps.append({"name": "dewow", "window_ns": self._dewow_window.value()})
        if self._time_zero.isChecked():
            manual = self._zero_mode.currentData() == "manual"
            steps.append({"name": "time_zero",
                          "time_ns": self._zero_time.value() if manual else None})
        if self._bandpass.isChecked():
            steps.append({"name": "bandpass", "low_mhz": self._low.value(),
                          "high_mhz": self._high.value()})
        if self._background.isChecked():
            steps.append({"name": "background", "window_traces": self._bg_window.value()})
        if self._migrate.isChecked():
            steps.append({"name": "migrate", "velocity": self._velocity.value()})
        kind = self._gain.currentData()
        if kind == "tpow":
            steps.append({"name": "gain", "kind": "tpow", "power": self._power.value()})
        elif kind == "agc":
            steps.append({"name": "gain", "kind": "agc", "window_ns": self._agc_window.value()})
        return steps

    def _on_settings_changed(self, *_args: Any) -> None:
        self._sync_rows()
        if self._raw is not None:
            self._status.setText("Processing…")
            self._debounce.trigger()

    def _on_velocity_changed(self, value: float) -> None:
        eps = gpr.permittivity_from_velocity(value)
        if abs(eps - self._permittivity.value()) > 0.005:
            self._permittivity.blockSignals(True)
            self._permittivity.setValue(eps)
            self._permittivity.blockSignals(False)
        if self._migrate.isChecked():
            self._on_settings_changed()
        else:
            self._redraw()

    def _on_permittivity_changed(self, value: float) -> None:
        self._velocity.setValue(gpr.velocity_from_permittivity(value))

    def _on_pick_toggled(self, on: bool) -> None:
        if on:
            self._tabs.setCurrentWidget(self._radargram)
            if self._vertical.currentText() == _ELEVATION:
                self._vertical.setCurrentText(_TIME)
        self._update_guide()

    def _on_spacing_changed(self, value: float) -> None:
        if self._raw is None:
            return
        start = float(self._raw.positions[0])
        positions = start + np.arange(self._raw.n_traces) * float(value)
        self._raw = self._raw.with_data(self._raw.data, positions=positions)
        self.log(f"Traces spaced evenly every {value:g} m.", "info")
        self._apply_file_elevation()
        self._on_settings_changed()

    def _on_frequency_changed(self, value: float) -> None:
        if self._raw is None:
            return
        self._raw = self._raw.with_data(self._raw.data, frequency_mhz=float(value))
        self._set_band_defaults(float(value))
        self._on_settings_changed()

    def _set_band_defaults(self, frequency: float) -> None:
        """Dewow over three periods, and a band from a quarter to 2.5 times the antenna's."""
        nyquist = 0.5 * self._raw.sampling_mhz if self._raw is not None else 1e9
        for box, value in ((self._dewow_window, 3000.0 / frequency),
                           (self._low, 0.25 * frequency),
                           (self._high, min(2.5 * frequency, 0.9 * nyquist))):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)

    # -- loading --------------------------------------------------------------
    def _open_file(self) -> None:
        source = Path(str(self._raw.source)) if self._raw is not None else None
        start = str(source.parent if source is not None and source.is_file()
                    else self.state.output_dir or Path.cwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open a GPR line", start, gpr.FILE_FILTER)
        if path:
            self._load(path)

    @staticmethod
    def _drop_paths(event: Any) -> List[str]:
        suffixes = set(gpr.GPR_ONLY_SUFFIXES) | {".sgy", ".segy", ".npz"}
        return [url.toLocalFile() for url in event.mimeData().urls()
                if url.isLocalFile() and Path(url.toLocalFile()).is_file()
                and Path(url.toLocalFile()).suffix.lower() in suffixes]

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self._drop_paths(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        paths = self._drop_paths(event)
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        if len(paths) > 1:
            self.log("GPR opens one line at a time; loading the first dropped file.", "info")
        self._tabs.setCurrentWidget(self._data_view)
        self._load(paths[0])

    def _load(self, path: str) -> None:
        self.log(f"Reading {Path(path).name}…", "info")
        worker = TaskWorker(gpr.read_gpr, path)
        worker.succeeded.connect(self._set_raw)
        worker.failed.connect(lambda message: self._load_failed(path, message))
        self.register_worker(worker, activity="Reading the GPR line")
        worker.start()

    def _load_failed(self, path: str, message: str) -> None:
        self.log(f"Could not read {Path(path).name}: {message}", "error")
        self._info.setText(f"Could not read {Path(path).name}: {message}")
        theme.set_tone(self._info, "error")

    def _example_dir(self) -> Optional[Path]:
        return io_utils.find_example(
            _EXAMPLE_DATA, accept=lambda p: (p / "Watercontent.npy").is_file()
            and (p / "Porosity.npy").is_file() and (p / "bot.npy").is_file())

    def _use_example(self) -> None:
        folder = self._example_dir()
        if folder is None:
            message = io_utils.missing_example_message(
                "The MODFLOW catchment example", f"{_EXAMPLE_DATA}/Watercontent.npy")
            self.log(message, "warn")
            self._info.setText(message)
            return
        self._simulate_example(folder)

    def _simulate_example(self, folder: Path) -> None:
        self.log("Simulating the example line over the MODFLOW catchment model…", "info")
        worker = TaskWorker(gpr.example_profile, str(folder))
        worker.succeeded.connect(self._set_raw)
        worker.failed.connect(lambda message: self._load_failed("the example", message))
        self.register_worker(worker, activity="Simulating the example GPR line")
        worker.start()

    def _set_raw(self, profile: gpr.GPRProfile) -> None:
        self._raw = profile
        self._processed = None
        self._apex = None
        self._file_elevation = profile.elevation
        self._elevation_source = "the file" if profile.elevation is not None else ""
        if profile.metadata.get("simulated"):
            self._elevation_source = "the model's ground surface"
        self._trace = profile.n_traces // 2
        self._trace_box.blockSignals(True)
        self._trace_box.setRange(1, profile.n_traces)
        self._trace_box.setValue(self._trace + 1)
        self._trace_box.blockSignals(False)
        spacing = profile.trace_spacing or 0.05
        frequency = profile.frequency_mhz if np.isfinite(profile.frequency_mhz) else 100.0
        for box, value in ((self._spacing, spacing), (self._frequency, frequency)):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        if not np.isfinite(profile.frequency_mhz):
            self._raw = profile.with_data(profile.data, frequency_mhz=frequency)
        self._set_band_defaults(frequency)
        self._range.unlock()
        source = f"{profile.instrument}: {profile.source}" if profile.instrument else profile.source
        self._source.setText(Path(str(profile.source)).name or str(profile.source))
        self._source.setToolTip(str(profile.source))
        self._source.setCursorPosition(0)
        self._info.setText(f"{profile.instrument or 'GPR line'}: {profile.describe()}.")
        theme.set_tone(self._info, None)
        if profile.metadata.get("warnings"):
            for warning in profile.metadata["warnings"]:
                self.log(warning, "warn")
        self.log(f"Loaded {source} - {profile.describe()}.", "success")
        self._update_elevation_note()
        self._tabs.setCurrentWidget(self._data_view)
        self._save_btn.setEnabled(False)
        self._sync_rows()
        self._sync_tab_controls()
        self._redraw()
        self._reprocess()

    def _load_elevations(self) -> None:
        if self._raw is None:
            self.log("Load a GPR line first.", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Ground elevations along the line",
                                              str(self.state.output_dir or Path.cwd()),
                                              "Tables (*.csv *.txt *.dat);;All files (*)")
        if not path:
            return
        try:
            table = np.genfromtxt(path, delimiter=None if path.endswith((".txt", ".dat"))
                                  else ",", dtype=float)
            table = table[np.all(np.isfinite(table[:, :2]), axis=1)]
            if table.shape[0] < 2:
                raise ValueError("it needs two columns, distance and elevation, and two rows.")
            order = np.argsort(table[:, 0])
            self._file_elevation = np.column_stack([table[order, 0], table[order, -1]])
        except Exception as exc:  # noqa: BLE001 - a bad table is reported, not raised
            self.log(f"Could not read the elevations: {exc}", "error")
            return
        self._elevation_source = Path(path).name
        self._apply_file_elevation()
        self._update_elevation_note()
        self._sync_rows()
        self._reprocess()

    def _apply_file_elevation(self) -> None:
        """Put the loaded distance-elevation table onto the traces."""
        table = self._file_elevation
        if self._raw is None or table is None or np.ndim(table) != 2:
            return
        elevation = np.interp(self._raw.positions, table[:, 0], table[:, 1])
        self._raw = self._raw.with_data(self._raw.data, elevation=elevation)

    def _elevation(self) -> Optional[np.ndarray]:
        profile = self._processed or self._raw
        return None if profile is None else profile.elevation

    def _update_elevation_note(self) -> None:
        if self._elevation() is not None:
            self._elevation_note.setText(f"Ground elevations from {self._elevation_source}.")
        else:
            self._elevation_note.setText("No ground elevations: the Elevation view is off.")

    # -- processing -----------------------------------------------------------
    def _reprocess(self) -> None:
        if self._raw is None:
            return
        self._generation += 1
        generation = self._generation
        steps = self.steps()
        started = time.perf_counter()
        worker = TaskWorker(gpr.process, self._raw, steps)
        worker.succeeded.connect(
            lambda result, g=generation: self._on_processed(g, result, started))
        worker.failed.connect(lambda message, g=generation: self._on_process_failed(g, message))
        self.register_worker(worker, activity="Processing the GPR line")
        self._status.setText("Processing…")
        worker.start()

    def _on_processed(self, generation: int, result: gpr.GPRProfile, started: float) -> None:
        if generation != self._generation:
            return                                   # a newer recipe is on its way
        self._processed = result
        self._last_seconds = time.perf_counter() - started
        picked = [line for line in result.history if line.startswith("Time zero")]
        self._zero_note.setText(picked[0] if picked else "")
        if picked and self._zero_mode.currentData() == "auto":
            try:
                zero = float(picked[0].split(" at ")[1].split(" ns")[0])
                self._zero_time.blockSignals(True)
                self._zero_time.setValue(zero)
                self._zero_time.blockSignals(False)
            except (IndexError, ValueError):
                pass
        steps = len(result.history) - len(self._raw.history)
        self._status.setText(f"{steps} step{'s' if steps != 1 else ''} applied in "
                             f"{self._last_seconds:.2f} s.")
        self._save_btn.setEnabled(True)
        self._sync_tab_controls()
        theme.set_tone(self._status, "hint")
        self._redraw()
        self._update_guide()
        self._publish()

    def _on_process_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._status.setText(f"Processing stopped: {message}")
        theme.set_tone(self._status, "error")
        self.log(f"GPR processing failed: {message}", "error")

    def _publish(self) -> None:
        profile = self._processed
        if profile is None:
            return
        velocity = self._velocity.value()
        self.report_result({
            "source": profile.source,
            "instrument": profile.instrument,
            "n_traces": profile.n_traces,
            "n_samples": profile.n_samples,
            "time_window_ns": float(profile.time_ns[-1]),
            "velocity_m_per_ns": velocity,
            "max_depth_m": float(gpr.time_to_depth(profile.time_ns[-1], velocity)),
            "processing": list(profile.history),
        })

    # -- drawing --------------------------------------------------------------
    def _shown_profile(self) -> Optional[gpr.GPRProfile]:
        if self._tabs.currentWidget() is self._data_view:
            return self._raw
        return self._processed or self._raw

    def _redraw(self) -> None:
        if self._raw is None:
            current = self._tabs.currentWidget()
            message = ("Drop a GPR file here,\nor use Open file… on the right."
                       if current is self._data_view else "Load a GPR line on the Data tab.")
            current.empty(message)
            return
        view = self._tabs.tabText(self._tabs.currentIndex())
        if view in (DATA, RADARGRAM):
            self._draw_radargram()
        elif view == TRACE:
            self._draw_trace()
        else:
            self._draw_spectrum()

    def _draw_radargram(self) -> None:
        profile = self._shown_profile()
        pane = (self._data_view if self._tabs.currentWidget() is self._data_view
                else self._radargram)
        fig = pane.figure
        fig.clear()
        ax = fig.add_subplot(111)
        unit = length_units.current()
        velocity = self._velocity.value()
        data = profile.data
        x = profile.positions
        mode = self._vertical.currentText()
        if mode == _ELEVATION and profile.elevation is None:
            mode = _DEPTH
        if mode == _ELEVATION:
            data, z = gpr.elevation_section(profile, velocity)
            extent = [x[0], x[-1], z[-1], z[0]]
        else:
            t = profile.time_ns
            y = gpr.time_to_depth(t, velocity) if mode == _DEPTH else t
            half = 0.5 * (y[1] - y[0]) if y.size > 1 else 0.5
            extent = [x[0], x[-1], y[-1] + half, y[0] - half]
        finite = np.abs(data[np.isfinite(data)])
        auto = float(np.percentile(finite, 98)) if finite.size else 1.0
        auto = auto if auto > 0 else 1.0
        lo, hi = self._range.limits(-auto, auto)
        image = ax.imshow(data, aspect="auto", extent=extent, interpolation="nearest",
                          cmap=cmaps.to_matplotlib(self._colormap.colormap()), vmin=lo, vmax=hi)
        if mode == _ELEVATION:
            ax.set_ylim(extent[2], extent[3])
            set_length_axis(ax, "y", "Elevation", unit=unit)
            ax.plot(x, profile.elevation, color=theme.PALETTE["canvas_text"], lw=0.8)
        elif mode == _DEPTH:
            set_length_axis(ax, "y", "Depth", unit=unit)
        else:
            ax.set_ylabel("Two-way time (ns)")
        set_length_axis(ax, "x", "Distance", unit=unit)
        if pane is self._radargram:
            self._draw_hyperbola(ax, profile, mode, velocity)
        ax.axvline(x[self._trace], color=theme.DATA_COLOR, lw=0.6, alpha=0.6)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.04, label="Amplitude")
        ax.set_title("\n".join(textwrap.wrap(self._title(profile, mode, velocity), 90)),
                     fontsize=9)
        fig.tight_layout()
        pane.drawn()

    def _title(self, profile: gpr.GPRProfile, mode: str, velocity: float) -> str:
        if profile is self._raw:
            text = "As recorded"
        else:
            done = [re.split(r"[:(]| at | \d", line)[0].strip().lower()
                    for line in profile.history[len(self._raw.history):]]
            text = "Processed: " + ", ".join(done) if done else "No processing chosen"
        if mode != _TIME:
            text += f" - depth at {velocity:.3g} m/ns"
        return text

    def _draw_hyperbola(self, ax: Any, profile: gpr.GPRProfile, mode: str,
                        velocity: float) -> None:
        """The diffraction curve for the picked apex at the current velocity."""
        if self._apex is None or mode == _ELEVATION:
            return
        x0, t0 = self._apex
        x = profile.positions
        half_width = max(2.0 * gpr.time_to_depth(max(t0, profile.dt_ns), velocity), 1.0)
        near = x[(x >= x0 - 3 * half_width) & (x <= x0 + 3 * half_width)]
        if near.size < 2:
            near = np.linspace(x0 - half_width, x0 + half_width, 50)
        t = gpr.hyperbola_time(near, x0, t0, velocity)
        y = gpr.time_to_depth(t, velocity) if mode == _DEPTH else t
        y0 = gpr.time_to_depth(t0, velocity) if mode == _DEPTH else t0
        ink = theme.SERIES_COLORS[1]
        ax.plot(near, y, color=ink, lw=1.6, ls="--")
        ax.plot([x0], [y0], marker="o", color=ink, ms=5)
        unit = length_units.current()
        depth = to_display_length(gpr.time_to_depth(t0, velocity), unit)
        ax.annotate(f"{velocity:.3g} m/ns, {depth:.2f} {unit} deep", (x0, y0),
                    xytext=(6, -8), textcoords="offset points", color=ink, fontsize=9)

    def _draw_trace(self) -> None:
        pane = self._trace_view
        fig = pane.figure
        fig.clear()
        raw, done = self._raw, self._processed
        j = int(np.clip(self._trace, 0, raw.n_traces - 1))
        axes = fig.subplots(2, 1, sharex=False)
        axes[0].plot(raw.time_ns, raw.data[:, j], color=theme.PALETTE["canvas_text"], lw=0.8)
        unit = length_units.current()
        axes[0].set_title(f"Trace {j + 1} at {to_display_length(raw.positions[j], unit):.2f} "
                          f"{unit} - as recorded", fontsize=10)
        if done is not None:
            jj = int(np.clip(j, 0, done.n_traces - 1))
            axes[1].plot(done.time_ns, done.data[:, jj], color=theme.DATA_COLOR, lw=0.8)
            axes[1].set_title("Processed", fontsize=10)
        for ax in axes:
            ax.set_xlabel("Two-way time (ns)")
            ax.set_ylabel("Amplitude")
            ax.axhline(0.0, color="#8e8e93", lw=0.5)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        pane.drawn()

    def _draw_spectrum(self) -> None:
        pane = self._spectrum
        fig = pane.figure
        fig.clear()
        ax = fig.add_subplot(111)
        series = [(self._raw, "As recorded", theme.PALETTE["canvas_text"])]
        if self._processed is not None:
            series.append((self._processed, "Processed", theme.DATA_COLOR))
        top = 0.0
        for profile, label, colour in series:
            freq, amp = _mean_spectrum(profile)
            ax.plot(freq, amp, color=colour, lw=1.0, label=label)
            top = max(top, float(freq[-1]))
        if self._bandpass.isChecked():
            for corner in (self._low.value(), self._high.value()):
                ax.axvline(corner, color=theme.SERIES_COLORS[1], lw=1.0, ls="--")
        frequency = self._frequency.value()
        ax.set_xlim(0.0, min(top, 5.0 * frequency) if frequency > 0 else top)
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (dB, each curve's peak at 0)")
        ax.set_ylim(-60.0, 3.0)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title("Average amplitude spectrum of the traces"
                     + (" - dashed: the band-pass" if self._bandpass.isChecked() else ""),
                     fontsize=10)
        fig.tight_layout()
        pane.drawn()

    # -- clicks ---------------------------------------------------------------
    def _on_radargram_click(self, event: Any) -> None:
        pane = self._tabs.currentWidget()
        if pane not in (self._data_view, self._radargram):
            return
        if pane.toolbar.mode or event.inaxes is None or event.xdata is None:
            return
        profile = self._shown_profile()
        if profile is None or not pane.figure.axes or event.inaxes is not pane.figure.axes[0]:
            return
        mode = self._vertical.currentText()
        if pane is self._radargram and self._pick.isChecked() and mode != _ELEVATION:
            y = float(event.ydata)
            t0 = gpr.depth_to_time(y, self._velocity.value()) if mode == _DEPTH else y
            self._apex = (float(event.xdata), max(t0, 0.0))
            self._redraw()
            self._update_guide()
            return
        self._trace = int(np.argmin(np.abs(profile.positions - float(event.xdata))))
        self._trace_box.blockSignals(True)
        self._trace_box.setValue(self._trace + 1)
        self._trace_box.blockSignals(False)
        self._redraw()

    def _on_trace_box(self, value: int) -> None:
        self._trace = int(value) - 1
        self._redraw()

    # -- velocity picking hint ------------------------------------------------
    def _update_guide(self) -> None:
        if self._pick.isChecked() and self._apex is None:
            self._pick_note.setText("Click the top of a hyperbola on the processed radargram.")
        elif self._pick.isChecked():
            self._pick_note.setText("Adjust velocity until the dashed curve follows the "
                                    "hyperbola, then choose Depth below the plot.")
        else:
            self._pick_note.setText("Fit a hyperbola, then choose Depth below the plot. "
                                    "A curve that is too narrow needs a higher velocity.")

    # -- saving ---------------------------------------------------------------
    def _save_results(self, folder: Optional[str] = None) -> Optional[Path]:
        if self._processed is None:
            self.log("Load and process a GPR line first.", "warn")
            return None
        if not folder:
            folder = select_directory(self, "Save the GPR results to folder",
                                      self.state.output_dir or Path.cwd())
        if not folder:
            return None
        out = io_utils.ensure_dir(folder)
        stem = Path(str(self._raw.source)).stem or "gpr_line"
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60] or "gpr_line"
        npz = gpr.save_profile(self._processed, out / f"{stem}_processed.npz")
        figure = out / f"{stem}_radargram.png"
        self._tabs.setCurrentWidget(self._radargram)
        self._draw_radargram()
        self._radargram.figure.savefig(figure, dpi=200)
        recipe = out / f"{stem}_processing.json"
        recipe.write_text(json.dumps({
            "source": self._raw.source, "instrument": self._raw.instrument,
            "steps": self.steps(), "velocity_m_per_ns": self._velocity.value(),
            "applied": self._processed.history}, indent=2), encoding="utf-8")
        self.log(f"Saved {npz.name}, {figure.name} and {recipe.name} in {out}", "success")
        return out

    def export_actions(self):
        if self._processed is None:
            return []
        return [("Processed GPR line, figure and recipe", lambda: self._save_results())]

    def _show_format_help(self) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("GPR data format")
        box.setTextFormat(Qt.RichText)
        box.setText(_FORMAT_HELP)
        box.exec()

    # -- automatic workflow display -------------------------------------------
    def show_run_inputs(self, inputs: Dict[str, Any]) -> str:
        """Open the run's GPR line here, for display; the run processes its own copy.

        A folder holding the MODFLOW example model is the simulated example line,
        as it is to the run (``runtime.catalog.read_gpr_input``). A line already
        open is left alone.
        """
        path = inputs.get("gpr_file")
        if self._raw is not None or not path or not Path(str(path)).exists():
            return ""
        source = Path(str(path))
        if source.is_dir():
            if not all((source / name).is_file()
                       for name in ("Watercontent.npy", "Porosity.npy", "bot.npy")):
                return ""
            self._simulate_example(source)
            return "the simulated example GPR line"
        self._load(str(source))
        return f"GPR line {source.name}"

    def show_run_stage(self, tool: str) -> str:
        if tool not in ("load_gpr_line", "process_gpr_line"):
            return ""
        raw = tool == "load_gpr_line"
        self._tabs.setCurrentWidget(self._data_view if raw else self._radargram)
        return DATA if raw else RADARGRAM

    # -- assistant ------------------------------------------------------------
    def agent_views(self) -> Dict[str, QWidget]:
        radargram = (self._data_view if self._tabs.currentWidget() is self._data_view
                     else self._radargram)
        return {"data": self._data_view, "radargram": radargram,
                "trace": self._trace_view, "spectrum": self._spectrum}

    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "load_file", "args": {"path": "str"},
                 "desc": "Read a GPR line: GSSI .DZT, pulseEKKO .DT1/.HD, MALA .rd3/.rd7/.rad, "
                         "SEG-Y or a saved .npz."},
                {"name": "use_example_data", "args": {},
                 "desc": "Simulate the example 100 MHz line over the MODFLOW catchment model."},
                {"name": "set_processing",
                 "args": {"dewow": "bool", "dewow_window_ns": "float", "time_zero": "bool",
                          "time_zero_ns": "float | null (null picks it)",
                          "bandpass": "bool", "low_mhz": "float", "high_mhz": "float",
                          "background": "bool", "background_traces": "int (0 = whole line)",
                          "migrate": "bool", "gain": "none | tpow | agc",
                          "gain_power": "float", "agc_window_ns": "float"},
                 "desc": "Set the clean-up, migration and gain; only the keys given change."},
                {"name": "set_velocity",
                 "args": {"velocity_m_per_ns": "float", "permittivity": "float"},
                 "desc": "Set the radar velocity (or the permittivity it stands for) used for "
                         "depth and migration."},
                {"name": "fit_hyperbola", "args": {"x_m": "float", "t_ns": "float"},
                 "desc": "Draw the diffraction curve with its apex at (x, t) at the current "
                         "velocity, to check the velocity on the radargram."},
                {"name": "show",
                 "args": {"view": "radargram | trace | spectrum",
                          "vertical": "time | depth | elevation", "data": "processed | raw"},
                 "desc": "Bring a view to the front (then capture_view it)."},
                {"name": "save_results", "args": {"folder": "str"},
                 "desc": "Write the processed line (.npz), figure (.png) and recipe (.json)."},
                {"name": "get_status", "args": {}, "desc": "What is loaded and how it is processed."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        try:
            if action == "load_file":
                path = str(args.get("path") or "")
                if not Path(path).is_file():
                    return {"status": "failed", "error": f"File not found: {path}"}
                self._set_raw(gpr.read_gpr(path))
                return self._agent_status()
            if action == "use_example_data":
                folder = self._example_dir()
                if folder is None:
                    return {"status": "failed", "error": io_utils.missing_example_message(
                        "The MODFLOW catchment example", f"{_EXAMPLE_DATA}/Watercontent.npy")}
                self._set_raw(gpr.example_profile(str(folder)))
                return self._agent_status()
            if action == "set_processing":
                self._agent_set_processing(args)
                return self._agent_status()
            if action == "set_velocity":
                if args.get("velocity_m_per_ns") is not None:
                    self._velocity.setValue(float(args["velocity_m_per_ns"]))
                elif args.get("permittivity") is not None:
                    self._permittivity.setValue(float(args["permittivity"]))
                return self._agent_status()
            if action == "fit_hyperbola":
                self._apex = (float(args["x_m"]), float(args["t_ns"]))
                self._tabs.setCurrentWidget(self._radargram)
                self._redraw()
                return self._agent_status()
            if action == "show":
                views = {"radargram": RADARGRAM, "trace": TRACE, "spectrum": SPECTRUM}
                view = str(args.get("view") or "radargram").lower()
                if view not in views:
                    return {"status": "failed", "error": f"view is one of {', '.join(views)}."}
                vertical = str(args.get("vertical") or "").lower()
                if vertical:
                    self._vertical.setCurrentText(vertical.title())
                data = str(args.get("data") or "").lower()
                target = {"radargram": self._radargram, "trace": self._trace_view,
                          "spectrum": self._spectrum}[view]
                if view == "radargram" and data == "raw":
                    target = self._data_view
                self._tabs.setCurrentWidget(target)
                return self._agent_status()
            if action == "save_results":
                out = self._save_results(str(args.get("folder") or "")
                                         or str(self.state.output_dir or Path.cwd()))
                return {"status": "ok" if out else "failed", "folder": str(out or "")}
            if action == "get_status":
                return self._agent_status()
        except Exception as exc:  # noqa: BLE001 - reported to the assistant, not raised
            return {"status": "failed", "error": str(exc)}
        return {"status": "failed", "error": f"Unknown action '{action}'.",
                "valid_actions": [a["name"] for a in self.agent_describe()["actions"]]}

    def _agent_set_processing(self, args: Dict[str, Any]) -> None:
        checks = {"dewow": self._dewow, "time_zero": self._time_zero,
                  "bandpass": self._bandpass, "background": self._background,
                  "migrate": self._migrate}
        for key, check in checks.items():
            if key in args:
                check.setChecked(bool(args[key]))
        numbers = {"dewow_window_ns": self._dewow_window, "low_mhz": self._low,
                   "high_mhz": self._high, "gain_power": self._power,
                   "agc_window_ns": self._agc_window}
        for key, box in numbers.items():
            if args.get(key) is not None:
                box.setValue(float(args[key]))
        if "background_traces" in args:
            self._bg_window.setValue(int(args["background_traces"] or 0))
        if "time_zero_ns" in args:
            manual = args["time_zero_ns"] is not None
            self._zero_mode.setCurrentIndex(1 if manual else 0)
            if manual:
                self._zero_time.setValue(float(args["time_zero_ns"]))
        if "gain" in args:
            index = self._gain.findData(str(args["gain"] or "none").lower())
            if index < 0:
                raise ValueError("gain is none, tpow or agc.")
            self._gain.setCurrentIndex(index)
        self._debounce.flush()

    def _agent_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "status": "ok",
            "loaded": self._raw is not None,
            "view": self._tabs.tabText(self._tabs.currentIndex()),
            "vertical": self._vertical.currentText().lower(),
            "steps": self.steps(),
            "velocity_m_per_ns": self._velocity.value(),
        }
        if self._raw is not None:
            status.update(source=self._raw.source, instrument=self._raw.instrument,
                          line=self._raw.describe(),
                          has_elevation=self._elevation() is not None)
        if self._processed is not None:
            status["applied"] = list(self._processed.history)
        if self._apex is not None:
            status["hyperbola_apex"] = {"x_m": self._apex[0], "t_ns": self._apex[1],
                                        "depth_m": gpr.time_to_depth(self._apex[1],
                                                                     self._velocity.value())}
        return status


def _mean_spectrum(profile: gpr.GPRProfile):
    """``(frequency MHz, amplitude dB)``: the traces' mean amplitude spectrum, peak 0 dB."""
    data = profile.data - np.mean(profile.data, axis=0, keepdims=True)
    spectrum = np.mean(np.abs(np.fft.rfft(data, axis=0)), axis=1)
    freq = np.fft.rfftfreq(profile.n_samples, d=profile.dt_ns) * 1000.0
    peak = float(np.max(spectrum)) if spectrum.size else 1.0
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(spectrum / max(peak, 1e-30), 1e-6))
    return freq, db
