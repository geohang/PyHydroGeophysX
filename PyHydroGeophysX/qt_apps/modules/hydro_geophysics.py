"""Hydro -> Geophysics: simulate geophysical surveys on a hydrologic model.

Laid out like the ERT page. The views are in the middle: the model's **Map**,
where the survey line is drawn with two clicks; the **Section** under that
line, with the electrodes, geophones and stations of the chosen methods on it;
and the **Results** of the run. The settings are on the right in the order the
work is done - the hydrologic model, the survey line, the methods, then each
chosen method's survey - with the Run button pinned under them and what still
stops it said above it. Expert settings (rock physics, error models, mesh) are
folded away under each panel's "Rock physics and errors" or "More options".

The numerics are reused from
``PyHydroGeophysX.Hydro_modular.hydro_to_geophysics``: the profile is sampled
here, and the forward runs go to a worker process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.transforms import offset_copy
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core.hydro_profile import inactive_cells, zone_boundaries
from PyHydroGeophysX.data_processing import run_inputs
from PyHydroGeophysX.Hydro_modular import hydro_to_geophysics as hydro_pipeline
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    Debouncer,
    ReproduceBar,
    make_spinbox,
    merged_row,
    select_directory,
    set_rows_visible,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.array_viewer import ArrayViewer
from PyHydroGeophysX.qt_apps.widgets.color_range import ColorRange
from PyHydroGeophysX.qt_apps.widgets.guided import (
    NextStepStrip,
    fold,
    hint as _hint,
    number_box as _number,
    start_card,
)
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row
from PyHydroGeophysX.qt_apps.widgets.run_controls import progress_with_stop
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker, TaskWorker
from PyHydroGeophysX.visualization.axis_units import set_section_axes
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_METHODS = ["ERT", "SRT", "TDEM", "FDEM", "Gravity"]
_METHOD_LABELS = {
    "ERT": "Electrical resistivity (ERT)",
    "SRT": "Seismic refraction (SRT)",
    "TDEM": "Time-domain EM (TDEM)",
    "FDEM": "Frequency-domain EM (FDEM)",
    "Gravity": "Gravity",
}
_METHOD_TIPS = {
    "ERT": "Electrodes along the line. Resistivity follows water content most directly.",
    "SRT": "Geophones along the line. Seismic velocity responds to saturation and to "
           "how stiff the rock is.",
    "TDEM": "Loop soundings at stations along the line: electrical conductivity with depth.",
    "FDEM": "Frequency soundings at stations along the line: shallow electrical conductivity.",
    "Gravity": "Gravity at stations along the line: the mass of the water in the ground.",
}
#: What the map can show, and the colour-bar label of each.
_MAP_VARS = {
    "Water content": "Volumetric water content",
    "Porosity": "Porosity",
    "Ground surface": "Ground surface elevation (m)",
    "Layer bottom": "Layer bottom elevation (m)",
}
_LAYERED_VARS = {"Water content", "Porosity", "Layer bottom"}
#: What the section can show.
_SECTION_VARS = ["Water content", "Porosity", "Saturation"]
#: The steps the assistant names. Each opens the view and the panel the step is
#: done in (see _go_to); the page itself no longer pages through them.
_STEPS = ["Data", "Profile", "Methods", "Parameters", "Run", "Results"]
_MAP_TAB, _SECTION_TAB, _RESULTS_TAB = range(3)
#: The ERT arrays offered: what the menu says, and the pyGIMLi scheme it makes.
_ERT_ARRAYS = [
    ("Wenner", "wa"),
    ("Wenner-beta", "wb"),
    ("Dipole-dipole", "dd"),
    ("Schlumberger", "slm"),
    ("Pole-dipole", "pd"),
]
#: The three zones the rock physics is set for, from the surface down
#: (core.hydro_profile.zone_boundaries splits the model into them).
_ZONES = ["Regolith", "Fractured rock", "Bedrock"]
#: Neutral plot ink: plot canvases stay light in both appearances.
_INK = "#1d1d1f"
_INK_MUTED = "#6e6e73"

#: The "Use example" hydrology grids, under the source repository's examples/:
#: the MODFLOW catchment model of Ex_ERT_workflow, whose arrays sit at the top
#: of examples/data.
_EXAMPLE_HYDRO = "data"
#: The section Ex_ERT_workflow takes through that catchment, as [col, row],
#: and the water-content snapshot it uses; the example opens on both.
_EXAMPLE_PROFILE = ([115.0, 70.0], [95.0, 180.0])
_EXAMPLE_SNAPSHOT = 5

_INPUT_FORMAT_FALLBACK = (
    "# Hydro input data format\n\n"
    "Place these four files in one folder:\n\n"
    "- `Watercontent.npy` - (n_time, n_layers, ny, nx) or (n_layers, ny, nx)\n"
    "- `Porosity.npy` - (n_layers, ny, nx)\n"
    "- `top.npy` - surface grid (ny, nx), NumPy\n"
    "- `bot.npy` - layer bottoms (n_layers, ny, nx)\n\n"
    "See hydro_input_format.md for details."
)


class HydroGeophysicsModule(BaseModule):
    module_key = "hydro_geophysics"
    module_title = "Hydro → Geophysics"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._manual_dir: Optional[Path] = None
        self._wc = self._por = self._top = self._bot = None
        #: Map cells outside the model domain, hidden on the Ground surface and
        #: Porosity maps.
        self._inactive: Optional[np.ndarray] = None
        #: Files of the four the loaded folder lacks.
        self._missing: List[str] = []
        self._point1: Optional[List[float]] = None
        self._point2: Optional[List[float]] = None
        self._profile: Optional[Dict[str, Any]] = None
        self._worker: Optional[ProcessWorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""
        #: Why the last run did not finish - its error, or that it was
        #: stopped - for the assistant, whose status otherwise still showed
        #: the result before it; "" while a run is under way or succeeded.
        self._run_problem = ""
        self._preview_worker: Optional[TaskWorker] = None
        self._preview_debounced = Debouncer(self._update_preview, 120)
        self._section_debounced = Debouncer(self._draw_section, 80)
        self._spins_debounced = Debouncer(self._apply_manual_points, 300)
        self._method_panels: Dict[str, QWidget] = {}
        self._result_paths: Dict[str, Dict[str, str]] = {}
        #: The first end of a new line is on the map and the second is awaited.
        self._line_started = False
        #: A run (or a saved configuration) has filled the Results tab.
        self._has_results = False
        self._current = 0
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        self._agent_parameters_confirmed = False
        self._agent_parameter_mode = ""

        # One colour map for the map and the section, which show the same
        # model: viridis until another is chosen.
        shared = cmaps.colormap_settings(self.state)
        self._hydro_colormap = cmaps.ColormapChooser(cmaps.HYDRO, "viridis", shared=shared)
        self._hydro_colormap.colormapChanged.connect(self._on_hydro_colormap_changed)

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_map_tab(shared), "Map")
        self._tabs.addTab(self._build_section_tab(), "Section")
        self._tabs.addTab(self._build_results_tab(), "Results")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._build_guide())
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        root.addWidget(self._build_controls())
        # View > Length Units: the model stays in metres; the section reticks.
        length_units.notifier().changed.connect(lambda _unit: self._section_debounced.trigger())

        self._reload()
        self._sync_method_panels()
        self._draw_section()

    # -- small widget helpers -----------------------------------------------
    @staticmethod
    def _zone_grid(rows: Sequence[Tuple[str, Sequence[QWidget]]]) -> QWidget:
        """One value per zone: the zones as columns, each setting named above its row."""
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        for column, zone in enumerate(_ZONES):
            grid.addWidget(QLabel(f"<b>{zone}</b>"), 0, column)
        row = 1
        for label, widgets in rows:
            grid.addWidget(_hint(label), row, 0, 1, len(_ZONES))
            for column, widget in enumerate(widgets):
                grid.addWidget(widget, row + 1, column)
            row += 2
        return holder

    # -- centre: what to do next, then map, section, results ------------------
    def _build_guide(self) -> QWidget:
        """The one line above the views that says what to do now (_next_step)."""
        self._guide = NextStepStrip()
        return self._guide

    def _build_start_card(self) -> QWidget:
        """What the Map tab shows before a model is loaded: what the page does, and how to start."""
        example_btn = self._example_button()
        example_btn.setProperty("primary", True)
        example_btn.setIcon(theme.icon("fa5s.box-open", color="#ffffff"))
        open_btn = QPushButton("Open folder…")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._select_folder)
        format_btn = QPushButton("Data format")
        format_btn.setIcon(theme.icon("fa5s.question-circle"))
        format_btn.clicked.connect(self._show_format_help)
        return start_card(
            "Simulate surveys on a hydrologic model",
            "See what ERT, seismic, EM and gravity surveys would measure over the "
            "water in a MODFLOW or ParFlow model.",
            ["Load a hydrologic model: its water content, porosity and layer surfaces.",
             "Draw a survey line on the model's map.",
             "Tick the surveys to simulate, and press Simulate surveys."],
            [example_btn, open_btn, format_btn],
            note=("The example is the MODFLOW catchment model of the ERT workflow "
                  "example, opened on its survey line."
                  if self.state.hydro_data_dir is None else ""))

    def _build_map_tab(self, shared: Dict[str, Any]) -> QWidget:
        # The start card until a model is loaded, then the map.
        self._map_stack = QStackedWidget()
        self._map_stack.addWidget(self._build_start_card())
        page = QWidget()
        self._map_stack.addWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self._variable = QComboBox()
        self._variable.addItems(list(_MAP_VARS))
        self._variable.currentTextChanged.connect(self._update_display)
        row.addWidget(self._variable)
        self._layer = QSpinBox()
        self._layer.setRange(0, 0)
        self._layer.setToolTip("The model layer the map shows, counted from the top (0).")
        self._layer.valueChanged.connect(self._update_display)
        self._layer_row = merged_row("Layer", self._layer)
        row.addWidget(self._layer_row)
        row.addStretch(1)
        row.addWidget(self._hydro_colormap)
        layout.addLayout(row)
        self._map = ArrayViewer(colormaps=shared, colormap_key=cmaps.HYDRO,
                                colormap_control=False)
        self._map.set_profile_mode(True)
        self._map.profileStarted.connect(self._on_profile_started)
        self._map.profileSelected.connect(self._on_profile_selected)
        self._map.profileCleared.connect(self._clear_profile)
        layout.addWidget(self._map, stretch=1)
        return self._map_stack

    def _build_section_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Show"))
        self._section_var = QComboBox()
        self._section_var.addItems(_SECTION_VARS)
        self._section_var.currentTextChanged.connect(self._on_section_var_changed)
        controls.addWidget(self._section_var)
        controls.addStretch(1)
        self._section_range = ColorRange(decimals=3, what="the section's colours")
        self._section_range.changed.connect(self._draw_section)
        controls.addWidget(self._section_range)
        layout.addLayout(controls)
        self._section_figure = Figure(figsize=(8, 4), layout="constrained")
        self._section_canvas = FigureCanvasQTAgg(self._section_figure)
        bar, self._section_toolbar = toolbar_row(self._section_canvas, page)
        layout.addWidget(bar)
        layout.addWidget(self._section_canvas, stretch=1)
        return page

    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("Method"))
        self._result_method = QComboBox()
        self._result_method.currentIndexChanged.connect(self._show_selected_result)
        header.addWidget(self._result_method)
        header.addStretch(1)
        open_btn = QPushButton("Open output folder")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._open_output_folder)
        header.addWidget(open_btn)
        layout.addLayout(header)
        self._result_summary = QLabel(
            "No results yet. Load a model, draw a line, choose the methods and press "
            "Simulate surveys.")
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)

        panels = QHBoxLayout()
        self._result_views: Dict[str, Tuple[ZoomableImageView, QLabel]] = {}
        for panel, title in (("model", "Model"), ("measurement", "Synthetic data")):
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            caption = QLabel("")
            caption.setWordWrap(True)
            view = ZoomableImageView()
            view.setMinimumHeight(360)
            box_layout.addWidget(caption)
            box_layout.addWidget(view, stretch=1)
            panels.addWidget(box, stretch=1)
            self._result_views[panel] = (view, caption)
        layout.addLayout(panels, stretch=1)
        self._result_files = _hint()
        layout.addWidget(self._result_files)
        return page

    # -- right: the settings, in the order the work is done ------------------
    def _build_controls(self) -> QWidget:
        scroll = ContentWidthScrollArea(minimum=430, maximum=480)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        self._data_group = self._build_data_group()
        self._line_group = self._build_line_group()
        self._methods_group = self._build_methods_group()
        self._more_group = self._build_more_group()
        #: Each numbered step's title; a finished step gets a tick after it.
        self._step_titles = {
            self._data_group: "1  Hydrologic model",
            self._line_group: "2  Survey line",
            self._methods_group: "3  Surveys",
        }
        for group in (self._data_group, self._line_group, self._methods_group,
                      self._more_group):
            layout.addWidget(group)
        layout.addStretch(1)
        self._controls_scroll = scroll

        # -- run: pinned under the column, as on the ERT page ---------------
        footer = QWidget()
        flayout = QVBoxLayout(footer)
        flayout.setContentsMargins(6, 4, 6, 6)
        flayout.setSpacing(6)
        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setFrameShadow(QFrame.Plain)
        flayout.addWidget(rule)
        self._run_reason = _hint()
        flayout.addWidget(self._run_reason)
        self._run_btn = QPushButton("Simulate surveys")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.play", color="#ffffff"))
        self._run_btn.setToolTip(
            "Turn the model under the line into resistivity, velocity, conductivity "
            "and density, and simulate each ticked survey on it.")
        self._run_btn.clicked.connect(self._run_forward)
        flayout.addWidget(self._run_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._stop = self.stop_button("The forward modeling")
        flayout.addWidget(progress_with_stop(self._progress, self._stop))
        self._run_status = QLabel("")
        self._run_status.setWordWrap(True)
        flayout.addWidget(self._run_status)

        column = QWidget()
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        column_layout.setSpacing(0)
        column_layout.addWidget(scroll, 1)
        column_layout.addWidget(footer)
        return column

    def _example_button(self) -> QPushButton:
        """"Use example", or "Use project data" when the studio was opened on a project's."""
        context = self.state.hydro_data_dir is not None
        button = QPushButton("Use project data" if context else "Use example")
        button.setIcon(theme.icon("fa5s.box-open"))
        button.setToolTip(
            "Load the hydrologic model of this project." if context else
            "Load the MODFLOW catchment model of the Ex_ERT_workflow example, on its "
            "time step and survey line.")
        button.clicked.connect(self._use_context_data)
        return button

    def _build_data_group(self) -> QGroupBox:
        box = QGroupBox("1  Hydrologic model")
        form = QFormLayout(box)
        open_btn = QPushButton("Open folder…")
        open_btn.setProperty("primary", True)
        open_btn.setIcon(theme.icon("fa5s.folder-open", color="#ffffff"))
        open_btn.setToolTip("Open a folder with Watercontent.npy, Porosity.npy, top.npy "
                            "and bot.npy.")
        open_btn.clicked.connect(self._select_folder)
        reload_btn = QPushButton()
        reload_btn.setIcon(theme.icon("fa5s.sync"))
        reload_btn.setToolTip("Read the folder again, after its files changed.")
        reload_btn.clicked.connect(self._reload)
        help_btn = QPushButton()
        help_btn.setIcon(theme.icon("fa5s.question-circle"))
        help_btn.setToolTip("What the folder must contain.")
        help_btn.clicked.connect(self._show_format_help)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (open_btn, self._example_button(), reload_btn, help_btn):
            row.addWidget(button)
        row.addStretch(1)
        form.addRow(row)
        self._data_status = QLabel("")
        self._data_status.setWordWrap(True)
        form.addRow(self._data_status)
        self._snapshot = QSpinBox()
        self._snapshot.setRange(0, 0)
        self._snapshot.setToolTip("The saved time of the model the surveys are simulated on.")
        self._snapshot.valueChanged.connect(self._on_snapshot_changed)
        self._snapshot_range = QLabel("")
        form.addRow("Time step", merged_row(self._snapshot, self._snapshot_range))
        return box

    def _build_line_group(self) -> QGroupBox:
        box = QGroupBox("2  Survey line")
        form = QFormLayout(box)
        self._line_hint = _hint()
        form.addRow(self._line_hint)
        self._p1x, self._p1y, self._p2x, self._p2y = (
            make_spinbox(0, 0, 100000) for _ in range(4))
        for spin in (self._p1x, self._p1y, self._p2x, self._p2y):
            spin.setKeyboardTracking(False)     # act on a typed number once it is in
            spin.valueChanged.connect(lambda *_: self._spins_debounced.trigger())
        form.addRow("Start", merged_row("column", self._p1x, "row", self._p1y))
        form.addRow("End", merged_row("column", self._p2x, "row", self._p2y))
        self._profile_label = QLabel("")
        self._profile_label.setWordWrap(True)
        form.addRow(self._profile_label)
        clear_btn = QPushButton("Clear line")
        clear_btn.setIcon(theme.icon("fa5s.eraser"))
        clear_btn.clicked.connect(self._clear_profile)
        form.addRow(merged_row(clear_btn))
        adv, aform = fold()
        self._num_samples = make_spinbox(
            220, 50, 1000, tooltip="How many points the model is read at along the "
                                   "line. More gives a finer section and a slower run.")
        self._num_samples.valueChanged.connect(lambda *_: self._preview_debounced.trigger())
        aform.addRow("Points along the line", self._num_samples)
        form.addRow(adv)
        self._show_line_hint()
        return box

    def _build_methods_group(self) -> QGroupBox:
        """The surveys: a tick each, and under a ticked one its settings in one line.

        The settings themselves are folded: every survey starts on values that
        run, so a first run needs nothing but the ticks.
        """
        box = QGroupBox("3  Surveys")
        layout = QVBoxLayout(box)
        layout.addWidget(_hint("Tick the surveys to simulate along the line. Each starts "
                               "on settings that work; unfold a line to change them."))
        builders = {
            "ERT": self._build_ert_panel,
            "SRT": self._build_srt_panel,
            "TDEM": self._build_tdem_panel,
            "FDEM": self._build_fdem_panel,
            "Gravity": self._build_gravity_panel,
        }
        self._method_boxes: Dict[str, QCheckBox] = {}
        self._method_panels = {}
        for name in _METHODS:
            check = QCheckBox(_METHOD_LABELS[name])
            check.setToolTip(_METHOD_TIPS[name])
            check.setChecked(name in ("ERT", "SRT"))
            check.toggled.connect(self._on_methods_changed)
            panel = builders[name]()
            layout.addWidget(check)
            layout.addWidget(panel)
            self._method_boxes[name] = check
            self._method_panels[name] = panel
        stations = QWidget()        # its own form, so the row hides with its label
        sform = QFormLayout(stations)
        sform.setContentsMargins(0, 4, 0, 0)
        self._em_stations = make_spinbox(
            24, 4, 200, tooltip="Stations spread evenly along the line, for TDEM, "
                                "FDEM and gravity.")
        self._em_stations.valueChanged.connect(self._on_survey_changed)
        sform.addRow("Stations (EM, gravity)", self._em_stations)
        layout.addWidget(stations)
        return box

    @staticmethod
    def _method_panel() -> Tuple[QWidget, QVBoxLayout, QLabel]:
        """What sits under a survey's tick: its one-line summary, then its folds."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 0, 0, 8)
        layout.setSpacing(0)
        summary = _hint()
        layout.addWidget(summary)
        return panel, layout, summary

    def _watch(self, *widgets: QWidget) -> None:
        """Redo the summaries and the section when any of ``widgets`` changes."""
        for widget in widgets:
            signal = (widget.currentIndexChanged if isinstance(widget, QComboBox)
                      else widget.valueChanged)
            signal.connect(self._on_survey_changed)

    def _build_ert_panel(self) -> QWidget:
        panel, layout, self._ert_summary = self._method_panel()
        self._ert_fit = _hint()
        layout.addWidget(self._ert_fit)
        survey, form = fold("Electrodes and array")
        self._ert_count = make_spinbox(72, 4, 512)
        self._ert_spacing = _number(1.0, 0.1, 50.0, 0.5, 2, " m")
        self._ert_start = _number(
            0.0, 0.0, 100000.0, 1.0, 1, " m",
            "Distance along the line of the first electrode. A spread that would run "
            "off the end of the line is moved back onto it.")
        self._ert_scheme = QComboBox()
        for label, code in _ERT_ARRAYS:
            self._ert_scheme.addItem(label, code)
        self._ert_noise = _number(3.0, 0.0, 100.0, 0.5, 2, " %",
                                  "Random noise added to each simulated reading.")
        form.addRow("Electrodes", self._ert_count)
        form.addRow("Spacing", self._ert_spacing)
        form.addRow("First electrode at", self._ert_start)
        form.addRow("Array", self._ert_scheme)
        form.addRow("Added noise", self._ert_noise)
        layout.addWidget(survey)

        adv, aform = fold("Rock physics and errors")
        self._rho = [_number(v, 1.0, 1e5, 10.0, 1) for v in (100.0, 500.0, 2400.0)]
        self._archie = [_number(v, 1.0, 3.0, 0.1, 2) for v in (2.2, 1.8, 2.5)]
        self._sigma_s = [_number(v, 0.0, 1.0, 0.001, 4) for v in (0.002, 0.0, 0.0)]
        aform.addRow(self._zone_grid([
            ("Resistivity when saturated (Ω·m)", self._rho),
            ("Saturation exponent n", self._archie),
            ("Surface conductivity (S/m)", self._sigma_s),
        ]))
        self._ert_rel = _number(3.0, 0.0, 100.0, 0.5, 2, " %")
        self._ert_abs = _number(0.0, 0.0, 100.0, 0.1, 3, " Ω")
        self._ert_rel.setToolTip("The error each simulated reading is given, as a "
                                 "share of the reading, plus the absolute part.")
        aform.addRow("Error model", merged_row(self._ert_rel, "+", self._ert_abs))
        layout.addWidget(adv)
        self._watch(self._ert_count, self._ert_spacing, self._ert_start, self._ert_scheme,
                    self._ert_noise)
        return panel

    def _build_srt_panel(self) -> QWidget:
        panel, layout, self._srt_summary = self._method_panel()
        self._srt_fit = _hint()
        layout.addWidget(self._srt_fit)
        survey, form = fold("Geophones and shots")
        self._srt_count = make_spinbox(72, 4, 512)
        self._srt_spacing = _number(1.0, 0.1, 50.0, 0.5, 2, " m")
        self._srt_start = _number(
            0.0, 0.0, 100000.0, 1.0, 1, " m",
            "Distance along the line of the first geophone. A spread that would run "
            "off the end of the line is moved back onto it.")
        self._srt_shot = make_spinbox(5, 1, 100)
        self._srt_noise = _number(1.0, 0.0, 100.0, 0.5, 2, " %",
                                  "Random noise added to each simulated travel time.")
        form.addRow("Geophones", self._srt_count)
        form.addRow("Spacing", self._srt_spacing)
        form.addRow("First geophone at", self._srt_start)
        form.addRow("A shot every", merged_row(self._srt_shot, "geophones"))
        form.addRow("Added noise", self._srt_noise)
        layout.addWidget(survey)

        adv, aform = fold("Rock physics and errors")
        self._vel_bulk = [_number(v, 1.0, 200.0, 1.0, 1) for v in (30.0, 50.0, 55.0)]
        self._vel_shear = [_number(v, 1.0, 200.0, 1.0, 1) for v in (20.0, 35.0, 50.0)]
        self._vel_rho = [_number(v, 1000.0, 4000.0, 10.0, 0) for v in (2650.0, 2670.0, 2680.0)]
        aform.addRow(self._zone_grid([
            ("Mineral bulk modulus (GPa)", self._vel_bulk),
            ("Mineral shear modulus (GPa)", self._vel_shear),
            ("Mineral density (kg/m³)", self._vel_rho),
        ]))
        self._vel_top_depth = _number(
            1.0, 0.1, 100.0, 0.5, 2, " m",
            "Depth used for the confining pressure of the regolith's grain contacts "
            "(Hertz-Mindlin).")
        self._vel_mid_aspect = _number(0.05, 0.001, 1.0, 0.01, 3,
                                       tip="Crack aspect ratio of the fractured rock.")
        self._vel_bot_aspect = _number(0.03, 0.001, 1.0, 0.01, 3,
                                       tip="Crack aspect ratio of the bedrock.")
        aform.addRow("Regolith pressure depth", self._vel_top_depth)
        aform.addRow("Crack aspect ratio",
                     merged_row("fractured", self._vel_mid_aspect, "bedrock", self._vel_bot_aspect))
        self._srt_noise_abs = _number(0.01, 0.0, 1000.0, 0.01, 4, " ms",
                                      "Absolute noise added to each travel time.")
        aform.addRow("Absolute noise", self._srt_noise_abs)
        layout.addWidget(adv)
        self._watch(self._srt_count, self._srt_spacing, self._srt_start, self._srt_shot,
                    self._srt_noise)
        return panel

    def _em_rock_physics(self, aform: QFormLayout) -> Tuple[QDoubleSpinBox, ...]:
        """The water and rock behind an EM method's conductivity (Archie)."""
        sigma_w = _number(0.05, 1e-4, 10.0, 0.01, 4, " S/m")
        m = _number(1.5, 1.0, 3.0, 0.1, 2)
        n = _number(2.0, 1.0, 3.0, 0.1, 2)
        sigma_s = _number(0.0, 0.0, 1.0, 0.001, 4, " S/m")
        aform.addRow("Pore-water conductivity", sigma_w)
        aform.addRow("Cementation exponent m", m)
        aform.addRow("Saturation exponent n", n)
        aform.addRow("Surface conductivity", sigma_s)
        return sigma_w, m, n, sigma_s

    def _build_tdem_panel(self) -> QWidget:
        panel, layout, self._tdem_summary = self._method_panel()
        survey, form = fold("Time gates")
        self._tdem_tmin = _number(0.01, 0.0001, 1000.0, 0.01, 4, " ms")
        self._tdem_tmax = _number(10.0, 0.01, 10000.0, 1.0, 3, " ms")
        self._tdem_ngates = make_spinbox(28, 4, 200)
        self._tdem_noise = _number(3.0, 0.0, 100.0, 0.5, 2, " %")
        form.addRow("Times", merged_row(self._tdem_tmin, "to", self._tdem_tmax))
        form.addRow("Time gates", self._tdem_ngates)
        form.addRow("Added noise", self._tdem_noise)
        layout.addWidget(survey)
        adv, aform = fold("Rock physics and loop")
        (self._tdem_sigma_w, self._tdem_m, self._tdem_n,
         self._tdem_sigma_s) = self._em_rock_physics(aform)
        self._tdem_radius = _number(10.0, 0.5, 200.0, 0.5, 1, " m")
        aform.addRow("Loop radius", self._tdem_radius)
        layout.addWidget(adv)
        self._watch(self._tdem_tmin, self._tdem_tmax, self._tdem_ngates, self._tdem_noise)
        return panel

    def _build_fdem_panel(self) -> QWidget:
        panel, layout, self._fdem_summary = self._method_panel()
        survey, form = fold("Frequencies")
        self._fdem_fmin = _number(10.0, 0.1, 1e6, 10.0, 2, " Hz")
        self._fdem_fmax = _number(1e4, 1.0, 1e7, 100.0, 1, " Hz")
        self._fdem_nfreq = make_spinbox(18, 4, 200)
        self._fdem_noise = _number(3.0, 0.0, 100.0, 0.5, 2, " %")
        form.addRow("Frequencies", merged_row(self._fdem_fmin, "to", self._fdem_fmax))
        form.addRow("How many", self._fdem_nfreq)
        form.addRow("Added noise", self._fdem_noise)
        layout.addWidget(survey)
        adv, aform = fold("Rock physics and instrument")
        (self._fdem_sigma_w, self._fdem_m, self._fdem_n,
         self._fdem_sigma_s) = self._em_rock_physics(aform)
        self._fdem_src_x = _number(0.0, -100.0, 1000.0, 1.0, 1, " m")
        self._fdem_rec_x = _number(12.0, -100.0, 1000.0, 1.0, 1, " m")
        self._fdem_radius = _number(10.0, 0.5, 200.0, 0.5, 1, " m")
        self._fdem_orient = QComboBox()
        self._fdem_orient.addItems(["z", "x", "y"])
        self._fdem_comp = QComboBox()
        self._fdem_comp.addItems(["secondary", "total"])
        self._fdem_wave = QComboBox()
        self._fdem_wave.addItems(["dipole", "loop"])
        aform.addRow("Transmitter at", self._fdem_src_x)
        aform.addRow("Receiver at", self._fdem_rec_x)
        aform.addRow("Loop radius", self._fdem_radius)
        aform.addRow("Receiver orientation", self._fdem_orient)
        aform.addRow("Receiver component", self._fdem_comp)
        aform.addRow("Source", self._fdem_wave)
        layout.addWidget(adv)
        self._watch(self._fdem_fmin, self._fdem_fmax, self._fdem_nfreq, self._fdem_noise)
        return panel

    def _build_gravity_panel(self) -> QWidget:
        panel, layout, self._grav_summary = self._method_panel()
        survey, form = fold("Sensor")
        self._grav_sensor_h = _number(1.0, 0.0, 100.0, 0.5, 2, " m")
        self._grav_noise = _number(2.0, 0.0, 100.0, 0.5, 2, " %")
        form.addRow("Height above ground", self._grav_sensor_h)
        form.addRow("Added noise", self._grav_noise)
        layout.addWidget(survey)
        adv, aform = fold("Densities and model grid")
        self._grav_rho_matrix = _number(2650.0, 1000.0, 4000.0, 10.0, 0, " kg/m³")
        self._grav_rho_water = _number(1000.0, 500.0, 1500.0, 10.0, 0, " kg/m³")
        self._grav_rho_air = _number(1.225, 0.0, 100.0, 0.1, 3, " kg/m³")
        self._grav_mesh_nx = make_spinbox(80, 10, 400)
        self._grav_mesh_nz = make_spinbox(60, 10, 400)
        self._grav_width_y = _number(12.0, 1.0, 200.0, 1.0, 1, " m")
        aform.addRow("Grain density", self._grav_rho_matrix)
        aform.addRow("Water density", self._grav_rho_water)
        aform.addRow("Air density", self._grav_rho_air)
        aform.addRow("Grid cells", merged_row(self._grav_mesh_nx, "across ×",
                                              self._grav_mesh_nz, "deep"))
        aform.addRow("Model width", self._grav_width_y)
        layout.addWidget(adv)
        self._watch(self._grav_sensor_h, self._grav_noise)
        return panel

    def _build_more_group(self) -> QGroupBox:
        box = QGroupBox("Simulation")
        outer = QFormLayout(box)
        outer.addRow(_hint("Rarely needs changing."))
        adv, form = fold("Noise seed, mesh and saved settings")
        outer.addRow(adv)
        self._seed = make_spinbox(
            7, 0, 10_000_000,
            tooltip="Seed of the random noise: the same seed gives the same noisy data.")
        self._mesh_quality = _number(
            32.0, 1.0, 40.0, 1.0, 0, "°",
            "Smallest angle of the mesh triangles. Higher gives better-shaped cells "
            "and more of them.")
        self._mesh_area = _number(1.0, 0.01, 100.0, 0.1, 2, " m²",
                                  "Largest area a mesh cell may have.")
        form.addRow("Noise seed", self._seed)
        form.addRow("Mesh quality", self._mesh_quality)
        form.addRow("Largest mesh cell", self._mesh_area)
        export_btn = QPushButton("Save survey settings")
        export_btn.setIcon(theme.icon("fa5s.file-export"))
        export_btn.setToolTip("Write the line, the surveys and their settings to "
                              "survey_config.json, without simulating anything.")
        export_btn.clicked.connect(self._export_config)
        form.addRow(merged_row(export_btn))
        return box

    # -- navigation (the assistant's step names) -----------------------------
    def _go_to(self, step: int) -> None:
        """Show the view and the panel the step is done in."""
        step = max(0, min(int(step), len(_STEPS) - 1))
        has_line = bool(self._point1 and self._point2)
        tab = {0: _MAP_TAB, 1: _MAP_TAB,
               2: _SECTION_TAB if has_line else _MAP_TAB,
               3: _SECTION_TAB if has_line else _MAP_TAB,
               4: _SECTION_TAB if has_line else _MAP_TAB,
               5: _RESULTS_TAB if self._has_results else _MAP_TAB}[step]
        self._tabs.setCurrentIndex(tab)
        self._current = step
        selected = [self._method_panels[m] for m in self._collect_methods()]
        group = {0: self._data_group, 1: self._line_group, 2: self._methods_group,
                 3: selected[0] if selected else self._methods_group}.get(step)
        if group is not None:
            self._controls_scroll.ensureWidgetVisible(group)

    def _on_tab_changed(self, index: int) -> None:
        if index == _MAP_TAB:
            self._current = 1 if self._data_loaded() else 0
        elif index == _SECTION_TAB:
            self._current = 3
        else:
            self._current = 5

    def _step_status(self) -> Dict[str, bool]:
        data_ok = self._data_loaded()
        methods_ok = len(self._collect_methods()) > 0
        profile_ok = bool(self._point1 and self._point2)
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"data": data_ok, "methods": methods_ok, "profile": profile_ok,
                "all": data_ok and methods_ok and profile_ok, "results": ran}

    def _data_loaded(self) -> bool:
        return self._top is not None or self._wc is not None

    # -- readiness and guidance ----------------------------------------------
    def _run_blocker(self) -> str:
        """What still stops a run, in a sentence; "" when it can run."""
        if not self._data_loaded():
            return "Load a hydrologic model (1) to simulate surveys."
        if self._missing:
            return f"The model folder has no {', '.join(self._missing)}."
        if not (self._point1 and self._point2):
            return "Draw a survey line on the map (2)."
        if not self._collect_methods():
            return "Tick at least one survey (3)."
        return ""

    def _next_step(self) -> str:
        """The one thing to do now, for the line above the views."""
        if self._run_busy is not None:
            return "Simulating the surveys. This takes a minute or two."
        if not self._data_loaded():
            return "Start by loading a hydrologic model: use the example or open a folder."
        if self._missing:
            return (f"This folder has no {', '.join(self._missing)}. Open another, or "
                    "press ? beside Open folder for what it must contain.")
        if not (self._point1 and self._point2):
            if self._line_started:
                return "Now click the end of the survey line on the map."
            return "Next: click the start and then the end of a survey line on the map."
        if not self._collect_methods():
            return "Next: tick the surveys to simulate (3, on the right)."
        if self._has_results:
            return ("Done: the results are on the Results tab. Change the line or the "
                    "surveys and simulate again to compare.")
        return ("Ready: the Section tab shows the ground under the line and where the "
                "sensors go. Press Simulate surveys.")

    def _sync_page(self) -> None:
        """Bring every cue on the page in line with where the work stands.

        The Simulate button and why it cannot run, the line that says what to
        do next, the start card in place of an empty map, the steps that cannot
        be done yet greyed out and the finished ones ticked, and the views that
        have nothing to show yet disabled.
        """
        if not hasattr(self, "_run_reason"):
            return
        reason = self._run_blocker()
        self._run_reason.setText(reason)
        self._run_reason.setVisible(bool(reason))
        if self._run_busy is None:
            self._run_btn.setEnabled(not reason)

        loaded = self._data_loaded()
        has_line = bool(self._point1 and self._point2)
        self._map_stack.setCurrentIndex(1 if loaded else 0)
        self._guide.setVisible(loaded or self._run_busy is not None)
        self._guide.setText(self._next_step())
        for group in (self._line_group, self._methods_group, self._more_group):
            group.setEnabled(loaded)
        done = {self._data_group: loaded and not self._missing,
                self._line_group: has_line,
                self._methods_group: has_line and bool(self._collect_methods())}
        for group, title in self._step_titles.items():
            group.setTitle(f"{title}  ✓" if done[group] else title)
        self._tabs.setTabEnabled(_SECTION_TAB, has_line)
        self._tabs.setTabEnabled(_RESULTS_TAB, self._has_results)

    # -- surveys -------------------------------------------------------------
    def _collect_methods(self) -> List[str]:
        if not hasattr(self, "_method_boxes"):
            return []
        return [name for name, box in self._method_boxes.items() if box.isChecked()]

    def _on_methods_changed(self, *_args) -> None:
        self._sync_method_panels()
        self._section_debounced.trigger()
        self._sync_page()

    def _sync_method_panels(self) -> None:
        """Show the settings of the ticked surveys only."""
        selected = set(self._collect_methods())
        for name, panel in self._method_panels.items():
            panel.setVisible(name in selected)
        set_rows_visible([self._em_stations], bool(selected & {"TDEM", "FDEM", "Gravity"}))
        self._update_summaries()

    def _on_survey_changed(self, *_args) -> None:
        self._update_summaries()
        self._section_debounced.trigger()

    def _line_length(self) -> Optional[float]:
        """The line's length in metres: the sampled one, or straight between the points."""
        if self._profile is not None:
            return float(np.asarray(self._profile["L_profile"], dtype=float)[-1])
        if self._point1 and self._point2:
            return float(np.hypot(self._point2[0] - self._point1[0],
                                  self._point2[1] - self._point1[1]))
        return None

    def _update_summaries(self) -> None:
        """Each survey in one line, and where each spread lies on the line."""
        self._ert_summary.setText(
            f"{self._ert_count.value()} electrodes {self._ert_spacing.value():g} m apart, "
            f"{self._ert_scheme.currentText()} array, {self._ert_noise.value():g} % noise.")
        self._srt_summary.setText(
            f"{self._srt_count.value()} geophones {self._srt_spacing.value():g} m apart, "
            f"a shot every {self._srt_shot.value()}, {self._srt_noise.value():g} % noise.")
        self._tdem_summary.setText(
            f"{self._tdem_ngates.value()} time gates from {self._tdem_tmin.value():g} to "
            f"{self._tdem_tmax.value():g} ms, {self._tdem_noise.value():g} % noise.")
        self._fdem_summary.setText(
            f"{self._fdem_nfreq.value()} frequencies from {self._fdem_fmin.value():g} to "
            f"{self._fdem_fmax.value():g} Hz, {self._fdem_noise.value():g} % noise.")
        self._grav_summary.setText(
            f"Sensor {self._grav_sensor_h.value():g} m above the ground, "
            f"{self._grav_noise.value():g} % noise.")
        length = self._line_length()
        for label, count_box, spacing_box, start_box, what in (
            (self._ert_fit, self._ert_count, self._ert_spacing, self._ert_start, "electrodes"),
            (self._srt_fit, self._srt_count, self._srt_spacing, self._srt_start, "geophones"),
        ):
            count, spacing = count_box.value(), spacing_box.value()
            span = (count - 1) * spacing
            label.setVisible(length is not None)
            if length is None:
                continue
            if span > length:
                label.setText(f"The spread is {span:.0f} m, longer than the {length:.0f} m "
                              f"line: use fewer or closer {what}.")
                theme.set_tone(label, "warn")
                continue
            first = float(hydro_pipeline.sensor_positions(
                start_box.value(), spacing, count, length)[0])
            text = f"Covers {first:.0f}–{first + span:.0f} m of the {length:.0f} m line"
            moved = abs(first - start_box.value()) > 1e-6
            label.setText(text + (", moved back to fit." if moved else "."))
            theme.set_tone(label, "hint")

    # -- data loading --------------------------------------------------------
    def _data_dir(self) -> Optional[Path]:
        if self._manual_dir is not None:
            return self._manual_dir
        if self.state.hydro_data_dir is not None:
            return Path(self.state.hydro_data_dir)
        return None

    def _use_context_data(self) -> None:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        self._manual_dir = None
        if self.state.hydro_data_dir is None:
            self._manual_dir = self._example_data_dir()
            if self._manual_dir is None:
                self.log(io_utils.missing_example_message(
                    "The MODFLOW catchment example", _EXAMPLE_HYDRO), "warn")
        self._reload()
        if self._data_loaded():
            self._agent_data_source_confirmed = True
            self._agent_data_source = (
                "project_context" if self.state.hydro_data_dir is not None else "example"
            )
            if self._agent_data_source == "example":
                self._apply_example_profile()

    def _example_data_dir(self) -> Optional[Path]:
        """Locate the bundled MODFLOW catchment grids for standalone launches.

        The arrays at the top of ``examples/data`` - the catchment model
        ``examples/Ex_ERT_workflow`` runs ERT on. The synthetic models that
        ``examples/generate_synthetic_examples.py`` writes under
        ``examples/data/synthetic`` can still be opened via "Open folder...".
        Both are in the source repository, not the pip package
        (``io_utils.example_dirs`` says where they are looked for).
        """
        def usable(folder: Path) -> bool:
            files = hydro_pipeline.find_hydro_files(folder)
            return files.get("top") is not None or files.get("water_content") is not None

        return io_utils.find_example(_EXAMPLE_HYDRO, self.state.project_root, usable)

    def _apply_example_profile(self) -> None:
        """Open the example on the snapshot and section Ex_ERT_workflow uses.

        Only a starting point: clicking the map or typing coordinates replaces it.
        """
        if self._top is None or np.ndim(self._top) != 2:
            return
        n_rows, n_cols = np.shape(self._top)
        if not all(0 <= p[0] < n_cols and 0 <= p[1] < n_rows for p in _EXAMPLE_PROFILE):
            return
        if self._wc is not None and self._wc.ndim == 4:
            self._snapshot.setValue(min(_EXAMPLE_SNAPSHOT, self._wc.shape[0] - 1))
        self._point1, self._point2 = (list(p) for p in _EXAMPLE_PROFILE)
        self._map.set_profile_points(self._point1, self._point2)
        self._sync_manual_spins()
        self._after_points_changed()

    def _show_format_help(self) -> None:
        """Pop a dialog rendering the input-data-format documentation."""
        doc_path = Path(__file__).with_name("hydro_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self)
        dlg.setWindowTitle("Hydro input data format")
        dlg.resize(760, 660)
        lay = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001 - very old Qt without setMarkdown
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn)
        dlg.exec()

    def _select_folder(self) -> None:
        start = str(self._data_dir() or Path.cwd())
        path = select_directory(self, "Select hydro data folder", start)
        if path:
            self._agent_data_source_confirmed = False
            self._agent_data_source = ""
            self._manual_dir = path
            self._reload()
            if self._data_loaded():
                self._agent_data_source_confirmed = True
                self._agent_data_source = "user_data"

    def _reload(self) -> None:
        data_dir = self._data_dir()
        self._wc = self._por = self._top = self._bot = None
        self._inactive = None
        self._missing = []
        if data_dir is None or not Path(data_dir).exists():
            self._data_status.setText(
                "No model loaded. Open a folder with Watercontent.npy, Porosity.npy, "
                "top.npy and bot.npy, or use the example.")
            theme.set_tone(self._data_status, None)
            self._snapshot_range.setText("")
            self._sync_page()
            return
        files = hydro_pipeline.find_hydro_files(Path(data_dir))
        # These stay memory-mapped for as long as the module holds them, which is
        # what keeps a multi-gigabyte water-content array off the heap. The cost
        # is that Windows will not let anything overwrite a mapped file, so a
        # regenerated folder cannot be reloaded until the old maps are dropped.
        # Releasing them here makes reloading the same folder work; the reference
        # count is what closes the map, so this has to happen before reassigning.
        self._release_hydro_arrays()
        self._wc = self._safe_load(files["water_content"], mmap=True)
        self._por = self._safe_load(files["porosity"], mmap=True)
        self._top = self._safe_load(files["top"], mmap=True)
        self._bot = self._safe_load(files["bot"], mmap=True)
        self._inactive = self._inactive_cells()
        self._missing = [hydro_pipeline.HYDRO_FILES[k] for k, v in files.items() if v is None]

        n_times = int(self._wc.shape[0]) if self._wc is not None and self._wc.ndim == 4 else 1
        self._snapshot.setRange(0, n_times - 1)
        self._snapshot.setEnabled(n_times > 1)
        self._snapshot_range.setText(f"of 0–{n_times - 1}" if n_times > 1 else "(one time)")
        layered = [arr for arr in (self._wc, self._por, self._bot)
                   if arr is not None and arr.ndim >= 3]
        n_layers = max((int(arr.shape[-3]) for arr in layered), default=1)
        self._layer.setRange(0, max(0, n_layers - 1))
        grid = next((np.shape(arr)[-2:] for arr in (self._top, self._wc, self._por, self._bot)
                     if arr is not None), None)
        if grid is not None:
            for spin, size in ((self._p1x, grid[1]), (self._p2x, grid[1]),
                               (self._p1y, grid[0]), (self._p2y, grid[0])):
                spin.blockSignals(True)
                spin.setMaximum(max(0, int(size) - 1))
                spin.blockSignals(False)

        example = self._example_data_dir()
        same = example is not None and Path(example).resolve() == Path(data_dir).resolve()
        name = "Example: MODFLOW catchment model" if same else f"Folder: {Path(data_dir).name}"
        lines = [f"<b>{name}</b>"]
        if grid is not None:
            lines.append(f"{n_times} time step{'s' if n_times != 1 else ''} · "
                         f"{n_layers} layers · {grid[0]} × {grid[1]} cells")
        if self._missing:
            lines.append(f"Missing: {', '.join(self._missing)} (see the ? button).")
        self._data_status.setText("<br>".join(lines))
        self._data_status.setToolTip(str(data_dir))
        theme.set_tone(self._data_status, "warn" if self._missing else None)
        self.log(f"Loaded hydro data from {data_dir}",
                 "success" if not self._missing else "warn")

        # A line from an earlier model is kept only if it still fits this one.
        if self._point1 and self._point2 and grid is not None:
            fits = all(0 <= p[0] < grid[1] and 0 <= p[1] < grid[0]
                       for p in (self._point1, self._point2))
            if fits:
                self._preview_debounced.trigger()
            else:
                self._clear_profile()
        self._update_display()
        self._sync_page()

    def _release_hydro_arrays(self) -> None:
        """Drop the memory-mapped hydro arrays so their file handles close.

        A numpy memmap closes when its last reference goes, so clearing the
        attributes is the release. gc.collect covers the case where a plot or a
        slice still holds one, which would otherwise keep the file locked.
        """
        import gc

        for name in ("_wc", "_por", "_top", "_bot"):
            if getattr(self, name, None) is not None:
                setattr(self, name, None)
        gc.collect()

    def _inactive_cells(self) -> Optional[np.ndarray]:
        """Map cells outside the model domain, for the Ground surface and Porosity maps.

        Without it those maps would fill the whole rectangle with the 0 tops
        and layer porosities a MODFLOW export writes there, and the zeros would
        squash the elevation colour scale.
        """
        if self._bot is None:
            return None
        try:
            return inactive_cells(self._bot)
        except Exception:  # noqa: BLE001 - masking is cosmetic
            return None

    def _safe_load(self, path, mmap: bool = False, text: bool = False):
        if path is None:
            return None
        try:
            if text:
                return np.loadtxt(path)
            return np.load(path, mmap_mode="r" if mmap else None)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to load {Path(path).name}: {exc}", "error")
            return None

    # -- map -----------------------------------------------------------------
    def _on_hydro_colormap_changed(self, name: str) -> None:
        """Recolour the map and the section; neither is re-extracted."""
        self._map.set_colormap(name)
        self._section_debounced.trigger()

    def _on_snapshot_changed(self, *_args) -> None:
        self._update_display()
        if self._point1 and self._point2:
            self._preview_debounced.trigger()

    def _update_display(self) -> None:
        var = self._variable.currentText()
        self._layer_row.setEnabled(var in _LAYERED_VARS)
        arr2d = None
        layer = self._layer.value()
        try:
            if var == "Water content" and self._wc is not None:
                w = self._wc[self._snapshot.value()] if self._wc.ndim == 4 else self._wc
                arr2d = np.asarray(w[min(layer, w.shape[0] - 1)], dtype=float)
            elif var == "Porosity" and self._por is not None:
                arr2d = np.asarray(self._por[min(layer, self._por.shape[0] - 1)], dtype=float)
            elif var == "Ground surface" and self._top is not None:
                arr2d = np.asarray(self._top, dtype=float)
            elif var == "Layer bottom" and self._bot is not None:
                arr2d = np.asarray(self._bot[min(layer, self._bot.shape[0] - 1)], dtype=float)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Display update failed: {exc}", "error")
            return
        if arr2d is None:
            return
        if (var in {"Ground surface", "Porosity"} and self._inactive is not None
                and self._inactive.shape == arr2d.shape):
            arr2d = np.where(self._inactive, np.nan, arr2d)
        title = var
        if var in _LAYERED_VARS:
            title += f" — layer {layer}"
        if var == "Water content" and self._wc is not None and self._wc.ndim == 4:
            title += f", time step {self._snapshot.value()}"
        self._map.set_array(arr2d, value_label=_MAP_VARS[var], title=title)
        if self._point1 and self._point2:
            self._map.set_profile_points(self._point1, self._point2)

    # -- survey line ---------------------------------------------------------
    def _show_line_hint(self, text: str = "") -> None:
        self._line_hint.setText(text or (
            "Click the start and then the end of the line on the map. You can also "
            "type the grid cells below."))

    def _on_profile_started(self, point: list) -> None:
        self._line_started = True
        self._show_line_hint(f"Start at column {point[0]:.0f}, row {point[1]:.0f}. "
                             "Now click the end of the line.")
        self._sync_page()

    def _on_profile_selected(self, p1: list, p2: list) -> None:
        self._point1 = [float(p1[0]), float(p1[1])]
        self._point2 = [float(p2[0]), float(p2[1])]
        self._line_started = False
        self._show_line_hint()
        self._sync_manual_spins()
        self._after_points_changed()

    def _apply_manual_points(self) -> None:
        p1 = [float(self._p1x.value()), float(self._p1y.value())]
        p2 = [float(self._p2x.value()), float(self._p2y.value())]
        if p1 == p2 or (p1 == self._point1 and p2 == self._point2):
            return
        self._point1, self._point2 = p1, p2
        self._map.set_profile_points(self._point1, self._point2)
        self._after_points_changed()

    def _sync_manual_spins(self) -> None:
        if self._point1 and self._point2:
            for spin, value in ((self._p1x, self._point1[0]), (self._p1y, self._point1[1]),
                                (self._p2x, self._point2[0]), (self._p2y, self._point2[1])):
                spin.blockSignals(True)
                spin.setValue(int(value))
                spin.blockSignals(False)

    def _after_points_changed(self) -> None:
        self._line_started = False
        self._show_line_hint()
        self.state.selected_points = [self._point1, self._point2]
        self._profile = None
        length = self._line_length() or 0.0
        self._profile_label.setText(
            f"Line from ({self._point1[0]:.0f}, {self._point1[1]:.0f}) to "
            f"({self._point2[0]:.0f}, {self._point2[1]:.0f}), about {length:.0f} m. "
            "The Section tab shows the ground under it.")
        self._update_summaries()
        self._preview_debounced.trigger()
        self._sync_page()

    def _update_preview(self) -> None:
        if not (self._point1 and self._point2):
            return
        # Capture params on the UI thread; extract the profile off-thread.
        params = self._collect_params()
        worker = TaskWorker(
            hydro_pipeline.extract_profile, self.state.context, params, self._point1, self._point2)
        worker.succeeded.connect(self._on_preview_ready)
        worker.failed.connect(lambda msg: self.log(f"Profile preview failed: {msg}", "warn"))
        self._preview_worker = self.register_worker(worker)
        worker.start()

    def _on_preview_ready(self, profile: dict) -> None:
        # A preview of a line or time step that has since changed: a newer one
        # is on its way.
        current = [[int(round(v)) for v in p] for p in (self._point1 or [], self._point2 or [])]
        if ([list(profile.get("point1", [])), list(profile.get("point2", []))] != current
                or profile.get("snapshot_index") != self._snapshot.value()):
            return
        self._profile = profile
        self._update_summaries()
        self._draw_section()

    def _clear_profile(self) -> None:
        if not (self._point1 or self._point2 or self._profile or self._line_started):
            return
        self._line_started = False
        self._point1 = self._point2 = self._profile = None
        self.state.selected_points = []
        self._map.clear_profile()
        self._profile_label.setText("")
        self._show_line_hint()
        self._update_summaries()
        self._draw_section()
        self._sync_page()

    # -- section -------------------------------------------------------------
    def _on_section_var_changed(self, *_args) -> None:
        # Another quantity starts on its own colour scale.
        self._section_range.blockSignals(True)
        self._section_range.unlock()
        self._section_range.blockSignals(False)
        self._draw_section()

    def _section_values(self) -> Tuple[np.ndarray, str]:
        wc = np.asarray(self._profile["water_content_profile"], dtype=float)
        por = np.asarray(self._profile["porosity_profile"], dtype=float)
        var = self._section_var.currentText()
        if var == "Porosity":
            return por, "Porosity"
        if var == "Saturation":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.clip(wc / np.where(por > 0, por, np.nan), 0.0, 1.0), "Saturation"
        return wc, "Volumetric water content"

    def _draw_section(self) -> None:
        """The model under the line, its three zones, and where each survey's sensors go."""
        figure = self._section_figure
        figure.clear()
        ax = figure.add_subplot(111)
        if self._profile is None:
            ax.set_axis_off()
            message = ("Draw a survey line on the Map to see the ground under it."
                       if not (self._point1 and self._point2) else "Reading the model under the line…")
            ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                    color=_INK_MUTED)
            self._section_canvas.draw_idle()
            self._section_toolbar.update()
            return
        distance = np.asarray(self._profile["L_profile"], dtype=float)
        bounds = np.asarray(self._profile["structure"], dtype=float)
        values, label = self._section_values()
        n_layers = min(values.shape[0], bounds.shape[0] - 1)
        cells = 0.5 * (values[:n_layers, :-1] + values[:n_layers, 1:])
        finite = cells[np.isfinite(cells)]
        lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
        if hi <= lo:
            hi = lo + 1e-3
        lo, hi = self._section_range.limits(float(lo), float(hi))
        image = ax.pcolormesh(np.tile(distance, (n_layers + 1, 1)), bounds[:n_layers + 1],
                              cells, shading="flat", cmap=self._hydro_colormap.colormap(),
                              vmin=lo, vmax=hi)
        figure.colorbar(image, ax=ax, pad=0.01).set_label(label)
        surface = bounds[0]
        ax.plot(distance, surface, color=_INK, lw=1.2)

        # The zones the rock physics is set for, named at the left edge.
        mid, bot = zone_boundaries(bounds.shape[0])
        for index in (mid, bot):
            ax.plot(distance, bounds[index], color=_INK, lw=0.9, ls="--")
        x_label = distance[0] + 0.01 * (distance[-1] - distance[0])
        depth_below = 0.15 * (surface[0] - bounds[bot, 0])
        for name, z in ((_ZONES[0], 0.5 * (surface[0] + bounds[mid, 0])),
                        (_ZONES[1], 0.5 * (bounds[mid, 0] + bounds[bot, 0])),
                        (_ZONES[2], bounds[bot, 0] - depth_below)):
            ax.text(x_label, z, name, fontsize=8, va="center", color=_INK,
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.75})

        # Where each chosen survey puts its sensors along the line: on the
        # ground, each kind a few points above the one before so none hides
        # another, at any zoom.
        methods = self._collect_methods()
        length = float(distance[-1])
        sensors = []
        if "ERT" in methods:
            x = hydro_pipeline.sensor_positions(self._ert_start.value(), self._ert_spacing.value(),
                                                self._ert_count.value(), length)
            sensors.append((x, {"marker": "v", "color": _INK}, f"{len(x)} electrodes"))
        if "SRT" in methods:
            x = hydro_pipeline.sensor_positions(self._srt_start.value(), self._srt_spacing.value(),
                                                self._srt_count.value(), length)
            sensors.append((x, {"marker": "^", "color": _INK_MUTED}, f"{len(x)} geophones"))
        if set(methods) & {"TDEM", "FDEM", "Gravity"}:
            x = distance[hydro_pipeline.station_indices(len(distance), self._em_stations.value())]
            sensors.append((x, {"marker": "s", "mfc": "white", "mec": _INK},
                            f"{len(x)} EM / gravity stations"))
        for row, (x, style, label_text) in enumerate(sensors):
            shifted = offset_copy(ax.transData, fig=figure, y=3.5 + 6.5 * row, units="points")
            ax.plot(x, np.interp(x, distance, surface), ls="none", ms=4, clip_on=False,
                    transform=shifted, label=label_text, **style)
        if methods:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ax.set_xlim(distance[0], distance[-1])
        # Room above the ground for the sensors.
        top, base = float(np.nanmax(surface)), float(np.nanmin(bounds[n_layers]))
        ax.set_ylim(base - 0.02 * (top - base), top + 0.08 * (top - base))
        set_section_axes(ax, surface=surface, unit=length_units.current())
        ax.set_title(f"{label} under the line, time step {self._profile.get('snapshot_index', 0)}")
        self._section_canvas.draw_idle()
        self._section_toolbar.update()

    # -- results -------------------------------------------------------------
    def _populate_results(self, result: Dict[str, Any]) -> None:
        self._has_results = True
        summary = result.get("data_summary") or {}
        counts = {"ERT": ("ert_count", "readings"), "SRT": ("srt_count", "travel times")}
        done = []
        for method in result.get("methods", []):
            key, unit = counts.get(method, ("", ""))
            if key and key in summary:
                done.append(f"{method} ({int(summary[key]):,} {unit})")
            else:
                done.append(str(method))
        if result.get("status") == "config_exported":
            self._result_summary.setText(
                "Nothing was simulated; the survey settings were saved instead.")
        else:
            self._result_summary.setText(
                f"Simulated {', '.join(done) or 'no method'} on the line. Each method's "
                "model and synthetic data are shown below.")
        paths = result.get("display_paths", {})
        self._result_paths = {
            str(method): {str(panel): str(path) for panel, path in panels.items()}
            for method, panels in paths.items() if isinstance(panels, dict)
        }
        # Backward-compatible fallback for old results written before separate
        # model/measurement figures were available.
        legacy = [str(path) for path in result.get("figure_paths", []) if Path(path).exists()]
        if not self._result_paths and legacy:
            self._result_paths = {
                method: {"model": legacy[0], "measurement": legacy[0]}
                for method in result.get("methods", [])
            }
        self._result_method.blockSignals(True)
        self._result_method.clear()
        for method in result.get("methods", []):
            if method in self._result_paths:
                self._result_method.addItem(_METHOD_LABELS.get(method, str(method)), method)
        self._result_method.blockSignals(False)
        if self._result_method.count():
            self._result_method.setCurrentIndex(0)
        self._show_selected_result()
        files = [Path(p).name for p in result.get("data_paths", [])]
        cfg = result.get("config_path")
        bits = [f"Synthetic data files: {', '.join(files)}" if files else ""]
        if cfg:
            bits.append(f"Survey settings: {cfg}")
        self._result_files.setText("<br>".join(bit for bit in bits if bit))
        self._sync_page()

    def _show_selected_result(self) -> None:
        """Load the selected method into the side-by-side model/data panels."""
        method = self._result_method.currentData()
        paths = self._result_paths.get(str(method), {}) if method else {}
        for panel, (view, caption) in self._result_views.items():
            path = paths.get(panel)
            if path and Path(path).exists() and view.set_image_file(path):
                caption.setText(Path(path).name)
            else:
                view.clear()
                caption.setText("" if not method else f"No {panel} figure for this method.")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.hydro_output_dir or self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- params / forward ----------------------------------------------------
    def _collect_params(self) -> Dict[str, Any]:
        """The run's parameters, in the units the pipeline takes (fractions, seconds)."""
        out_dir = self.state.hydro_output_dir or self.state.output_dir
        return {
            "snapshot_index": self._snapshot.value(),
            "num_samples": self._num_samples.value(),
            "hydro_data_dir": str(self._data_dir()) if self._data_dir() else "",
            "output_dir": str(out_dir) if out_dir else "",
            "seed": self._seed.value(),
            "mesh": {
                "quality": self._mesh_quality.value(),
                "area": self._mesh_area.value(),
            },
            "em_stations": self._em_stations.value(),
            "ert": {
                "num_electrodes": self._ert_count.value(),
                "electrode_spacing": self._ert_spacing.value(),
                "electrode_start": self._ert_start.value(),
                "scheme_name": str(self._ert_scheme.currentData()),
                "noise_level": self._ert_noise.value() / 100.0,
                "rel_error": self._ert_rel.value() / 100.0,
                "abs_error": self._ert_abs.value(),
            },
            "srt": {
                "num_sensors": self._srt_count.value(),
                "sensor_spacing": self._srt_spacing.value(),
                "shot_distance": self._srt_shot.value(),
                "sensor_start": self._srt_start.value(),
                "noise_level": self._srt_noise.value() / 100.0,
                "noise_abs": self._srt_noise_abs.value() / 1000.0,
            },
            "petro": {
                "rho_sat": [s.value() for s in self._rho],
                "archie_n": [s.value() for s in self._archie],
                "sigma_s": [s.value() for s in self._sigma_s],
            },
            "srt_vel": {
                "top": {
                    "bulk_modulus": self._vel_bulk[0].value(),
                    "shear_modulus": self._vel_shear[0].value(),
                    "mineral_density": self._vel_rho[0].value(),
                    "depth": self._vel_top_depth.value(),
                },
                "mid": {
                    "bulk_modulus": self._vel_bulk[1].value(),
                    "shear_modulus": self._vel_shear[1].value(),
                    "mineral_density": self._vel_rho[1].value(),
                    "aspect_ratio": self._vel_mid_aspect.value(),
                },
                "bot": {
                    "bulk_modulus": self._vel_bulk[2].value(),
                    "shear_modulus": self._vel_shear[2].value(),
                    "mineral_density": self._vel_rho[2].value(),
                    "aspect_ratio": self._vel_bot_aspect.value(),
                },
            },
            "tdem": {
                "t_min_s": self._tdem_tmin.value() / 1000.0,
                "t_max_s": self._tdem_tmax.value() / 1000.0,
                "n_gates": self._tdem_ngates.value(),
                "sigma_w": self._tdem_sigma_w.value(),
                "m": self._tdem_m.value(),
                "n": self._tdem_n.value(),
                "sigma_s": self._tdem_sigma_s.value(),
                "source_radius": self._tdem_radius.value(),
                "noise_level": self._tdem_noise.value() / 100.0,
            },
            "fdem": {
                "f_min_hz": self._fdem_fmin.value(),
                "f_max_hz": self._fdem_fmax.value(),
                "n_freqs": self._fdem_nfreq.value(),
                "sigma_w": self._fdem_sigma_w.value(),
                "m": self._fdem_m.value(),
                "n": self._fdem_n.value(),
                "sigma_s": self._fdem_sigma_s.value(),
                "source_x": self._fdem_src_x.value(),
                "receiver_x": self._fdem_rec_x.value(),
                "source_radius": self._fdem_radius.value(),
                "receiver_orientation": self._fdem_orient.currentText(),
                "receiver_component": self._fdem_comp.currentText(),
                "waveform_type": self._fdem_wave.currentText(),
                "noise_level": self._fdem_noise.value() / 100.0,
            },
            "gravity": {
                "rho_matrix": self._grav_rho_matrix.value(),
                "rho_water": self._grav_rho_water.value(),
                "rho_air": self._grav_rho_air.value(),
                "sensor_height": self._grav_sensor_h.value(),
                "noise_level": self._grav_noise.value() / 100.0,
                "mesh_nx": self._grav_mesh_nx.value(),
                "mesh_nz": self._grav_mesh_nz.value(),
                "model_width_y": self._grav_width_y.value(),
            },
        }

    def _export_config(self) -> str:
        params = self._collect_params()
        methods = self._collect_methods()
        config = hydro_pipeline.build_survey_config(
            self.state.context, params, methods, self._point1 or [0, 0], self._point2 or [0, 0],
            profile=self._profile)
        active = self.state.active_run(self.module_key)
        out_dir = active.outputs_dir if active else self.state.ensure_results_store().scratch_dir(self.module_key)
        config_path = out_dir / "survey_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported survey config to {config_path}", "success")
        self.report_result({
            "selected_points": [self._point1, self._point2], "methods": methods,
            "config_path": str(config_path), "status": "config_exported"})
        return str(config_path)

    def _run_forward(self) -> None:
        reason = self._run_blocker()
        if reason:
            self.log(reason, "warn")
            return
        params = self._collect_params()
        methods = self._collect_methods()
        seed = int(params.pop("seed"))
        data_dir = self._data_dir()
        params.pop("hydro_data_dir", None)
        params.pop("output_dir", None)
        try:
            run = self.begin_persisted_run(
                "hydro_geophysics.forward",
                workflow_id="hydro_geophysics.forward",
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        # One compressed bundle rather than a copy of each array: these are read
        # back only as a set, and the run is what has to stay small.
        inputs: Dict[str, Any] = {"context": {}}
        if data_dir is not None:
            present = {
                filename: data_dir / filename
                for filename in ("Watercontent.npy", "Porosity.npy", "top.npy", "bot.npy")
                if (data_dir / filename).is_file()
            }
            if present:
                try:
                    stored = run_inputs.save_file_bundle(
                        run.inputs_dir / "hydro_inputs",
                        present,
                        kind="hydrology_arrays",
                        meta={"source_dir": str(data_dir)},
                    )
                except OSError as exc:
                    self.fail_persisted_run(str(exc))
                    self.log(f"Could not persist the hydrology arrays: {exc}", "error")
                    return
                inputs["hydro_bundle"] = ArtifactRef.from_path(
                    stored,
                    artifact_id="hydro-input:bundle",
                    kind="hydrology_arrays",
                    base_dir=run.run_dir,
                    metadata={"files": sorted(present)},
                )
        spec = WorkflowSpec(
            workflow_id="hydro_geophysics.forward",
            inputs=inputs,
            parameters={
                **params,
                "methods": methods,
                "point1": list(self._point1 or []),
                "point2": list(self._point2 or []),
                # The run's figures are drawn in another process, which cannot
                # see the studio's View > Length Units choice.
                "length_unit": length_units.current(),
            },
            seed=seed,
            metadata={"profile_source": "qt_map_picker"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="hydro_forward"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController([self._run_btn])
        self._run_busy.start()
        self._run_btn.setText("Simulating…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy
        self._run_status.setText("Starting the simulation…")
        theme.set_tone(self._run_status, None)
        self._sync_page()
        self.log("Starting hydro → geophysics forward modeling…", "info")
        # In a process of its own, so the window keeps painting through the
        # forward runs; the results are files, and nothing else comes back.
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir,
                                       run.result_path)
        self._worker = self.register_worker(worker, activity="Running the forward models")
        self._run_problem = ""
        worker.logged.connect(lambda message: self._on_worker_log(message, "info"))
        worker.succeeded.connect(self._on_workflow_ok)
        worker.failed.connect(
            lambda message, w=worker: self._on_forward_failed(message, w.missing_backend))
        worker.finished.connect(self._reset_run_button)
        worker.start()
        self._stop.attach(worker, "hydro_geophysics.forward")

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "hydro_geophysics.forward",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_forward_ok(result.legacy_payload())

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message)
        self.log(message, level)

    def _on_forward_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"Forward modeling complete: {result.get('methods')}.", "success")
        self._run_status.setText("Finished. The results are on the Results tab.")
        theme.set_tone(self._run_status, "ok")
        result["selected_points"] = [self._point1, self._point2]
        self.report_result(result)
        self._populate_results(result)
        self._go_to(5)

    def _on_forward_failed(self, message: str, backend_unavailable: bool) -> None:
        self._run_problem = message
        self.fail_persisted_run(message, "hydro_geophysics.forward")
        self._progress.setRange(0, 1); self._progress.setValue(0)
        if not backend_unavailable:
            # The run itself failed - out of memory, arrays of the wrong shape -
            # and its own reason is what helps, beside the settings that can be
            # changed before trying again. Exporting the configuration in its
            # place blamed a backend that was there.
            self.log(f"Forward modeling failed: {message}", "error")
            self._run_status.setText(f"The simulation failed: {message}")
            theme.set_tone(self._run_status, "error")
            return
        self.log(f"Forward modeling could not run: {message}", "warn")
        config_path = self._export_config()
        note = ("A forward-modeling engine this run needs is not installed or could not be "
                "loaded (the log says which). The survey settings were saved to "
                f"{config_path}.")
        self._run_status.setText(note)
        theme.set_tone(self._run_status, "warn")
        self.log(note, "warn")
        self._populate_results({"status": "config_exported", "methods": self._collect_methods(),
                                "config_path": config_path, "figure_paths": []})
        self._go_to(5)

    def _reset_run_button(self) -> None:
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._run_btn.setText("Simulate surveys")
        self._progress.setVisible(False)
        if self._worker is not None and self._worker.is_cancelled():
            self._run_problem = "Stopped by user"
            self._run_status.setText("Stopped before it finished. Press Simulate surveys "
                                     "to start again.")
            theme.set_tone(self._run_status, "warn")
        self._sync_page()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        """Advertise the actions and current state to the AQUAH assistant."""
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example_data", "args": {},
                 "desc": "Load bundled example hydro data after the user explicitly chooses it."},
                {"name": "set_data_dir", "args": {"path": "str"},
                 "desc": "Load the user's hydro data after the user chooses it and supplies a folder."},
                {"name": "start_profile_pick", "args": {},
                 "desc": "Open the map and pause while the user clicks the two ends of the line."},
                {"name": "pick_profile", "args": {"p1": "[col, row]", "p2": "[col, row]"},
                 "desc": "Define the 2D cross-section by two grid points."},
                {"name": "select_methods", "args": {"methods": list(_METHODS)},
                 "desc": "Choose which forward methods to simulate."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                  "desc": ("Set parameters. Keys: snapshot_index, num_samples, seed, "
                           "ert.num_electrodes, ert.electrode_spacing, ert.electrode_start, "
                           "ert.scheme_name (one of wa/wb/dd/slm/pd), ert.noise_level, "
                           "ert.rel_error, ert.abs_error, srt.num_sensors, srt.sensor_spacing, "
                           "srt.sensor_start, srt.shot_distance, srt.noise_level, "
                           "srt.noise_abs, plus the TDEM/FDEM/Gravity keys returned in "
                           "parameter_defaults. Noise levels and errors are fractions "
                           "(0.03 = 3 %), times in seconds.")},
                {"name": "confirm_parameters", "args": {"mode": ["defaults", "custom"]},
                 "desc": "Confirm the user's explicit parameter choice before running; defaults keeps displayed values."},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Show the view and panel of a step, by name or index."},
                {"name": "run", "args": {},
                 "desc": "Run hydro -> geophysics forward modeling (needs data, profile, methods)."},
                {"name": "get_status", "args": {},
                 "desc": "Report readiness and the last result status."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one AQUAH action against this module (called on the UI thread)."""
        args = args or {}
        handlers = {
            "use_example_data": lambda: self._agent_use_example_data(),
            "set_data_dir": lambda: self._agent_set_data_dir(args.get("path")),
            "start_profile_pick": lambda: self._agent_start_profile_pick(),
            "pick_profile": lambda: self._agent_pick_profile(args),
            "select_methods": lambda: self._agent_select_methods(args.get("methods")),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "confirm_parameters": lambda: self._agent_confirm_parameters(args.get("mode")),
            "goto_step": lambda: self._agent_goto_step(args.get("step")),
            "run": lambda: self._agent_run(),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def show_run_inputs(self, inputs):
        """Open the automatic run's data here, so this panel is not blank.

        Part of the studio's contract: a module the run brings to the front has
        to show the user a result, not an empty tool with a banner over it.
        Display only - the workflow does its own work in its own process - and
        anything already loaded is left alone.
        """
        if self._agent_status().get("data_dir"):
            return ""
        folder = (inputs.get("modflow_dir") or inputs.get("parflow_dir")
                  or inputs.get("hydro_model"))
        if not folder or not Path(str(folder)).is_dir():
            return ""
        if not self._agent_set_data_dir(str(folder)).get("loaded"):
            return ""
        return Path(str(folder)).name

    def _agent_status(self) -> Dict[str, Any]:
        status = self._step_status()
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "steps": status,
            "current_step": _STEPS[self._current],
            "data_dir": str(self._data_dir() or ""),
            "data_source_confirmation_required": not self._agent_data_source_confirmed,
            "data_source": self._agent_data_source,
            "data_source_options": ["example", "user_data"],
            "methods": self._collect_methods(),
            "parameter_defaults": self._agent_parameter_defaults(),
            "parameters_confirmed": self._agent_parameters_confirmed,
            "parameter_mode": self._agent_parameter_mode,
            "profile": {"p1": self._point1, "p2": self._point2},
            "run_blocked_because": self._run_blocker(),
            "last_result_status": last.get("status"),
            "last_run_problem": self._run_problem,
        }

    def _agent_use_example_data(self) -> Dict[str, Any]:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        example_dir = self._example_data_dir()
        if example_dir is None:
            return {"status": "failed", "error": "Bundled example hydro data were not found."}
        self._manual_dir = example_dir
        self._reload()
        loaded = self._data_loaded()
        if loaded:
            self._agent_data_source_confirmed = True
            self._agent_data_source = "example"
            self._apply_example_profile()
        return {"status": "ok" if loaded else "failed",
                "data_dir": str(self._data_dir() or ""), "loaded": loaded,
                "data_source": self._agent_data_source}

    def _agent_set_data_dir(self, path: Any) -> Dict[str, Any]:
        self._agent_data_source_confirmed = False
        self._agent_data_source = ""
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a data folder."}
        data_dir = Path(str(path))
        if not data_dir.is_dir():
            return {"status": "failed", "error": f"Hydro data folder does not exist: {data_dir}"}
        self._manual_dir = data_dir
        self._reload()
        loaded = self._data_loaded()
        if loaded:
            self._agent_data_source_confirmed = True
            self._agent_data_source = "user_data"
        return {"status": "ok" if loaded else "failed",
                "data_dir": str(self._data_dir() or ""), "loaded": loaded,
                "data_source": self._agent_data_source,
                **({} if loaded else {"error": "No usable hydro arrays were found in the folder."})}

    def _agent_require_data_source(self) -> Optional[Dict[str, Any]]:
        """Return an agent-facing checkpoint until the user chooses the data source."""
        if self._agent_data_source_confirmed:
            return None
        return {
            "status": "failed",
            "error": "Ask the user to choose bundled example data or their own hydro data first.",
            "required_choice": ["example", "user_data"],
            "available_context_data_dir": str(self.state.hydro_data_dir or ""),
        }

    def _agent_parameter_defaults(self) -> Dict[str, Dict[str, Any]]:
        """Return the visible default acquisition settings for AQUAH to present."""
        params = self._collect_params()
        selected = set(self._collect_methods())
        defaults: Dict[str, Dict[str, Any]] = {}
        if "ERT" in selected:
            defaults["ERT"] = {
                "electrodes": params["ert"]["num_electrodes"],
                "spacing_m": params["ert"]["electrode_spacing"],
                "array": params["ert"]["scheme_name"],
                "noise_level": params["ert"]["noise_level"],
                "relative_error": params["ert"]["rel_error"],
                "absolute_error": params["ert"]["abs_error"],
            }
        if "SRT" in selected:
            defaults["SRT"] = {
                "sensors": params["srt"]["num_sensors"],
                "spacing_m": params["srt"]["sensor_spacing"],
                "shot_distance_sensors": params["srt"]["shot_distance"],
                "noise_level": params["srt"]["noise_level"],
                "absolute_noise_s": params["srt"]["noise_abs"],
            }
        if "TDEM" in selected:
            defaults["TDEM"] = {
                "time_range_s": [params["tdem"]["t_min_s"], params["tdem"]["t_max_s"]],
                "time_gates": params["tdem"]["n_gates"], "noise_level": params["tdem"]["noise_level"],
            }
        if "FDEM" in selected:
            defaults["FDEM"] = {
                "frequency_range_hz": [params["fdem"]["f_min_hz"], params["fdem"]["f_max_hz"]],
                "frequencies": params["fdem"]["n_freqs"], "noise_level": params["fdem"]["noise_level"],
            }
        if "Gravity" in selected:
            defaults["Gravity"] = {
                "sensor_height_m": params["gravity"]["sensor_height"],
                "noise_level": params["gravity"]["noise_level"],
                "stations": params["em_stations"],
            }
        return defaults

    def _agent_start_profile_pick(self) -> Dict[str, Any]:
        """Open the map and pause AQUAH for the user's two clicks."""
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        if not self._data_loaded():
            return {"status": "failed", "error": "Load hydrologic data before picking a profile.",
                    "hint": "Use example data or select a hydrologic model-output folder first."}
        self._go_to(1)
        self._map.set_profile_mode(True)
        return {
            "status": "awaiting_user",
            "step": _STEPS[self._current],
            "message": "The map is open. Click the start and the end of the line, then say ‘continue’.",
        }

    def _agent_pick_profile(self, args: Dict[str, Any]) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        p1, p2 = args.get("p1"), args.get("p2")
        if not (isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple))
                and len(p1) == 2 and len(p2) == 2):
            return {"status": "failed", "error": "Provide p1 and p2 as [col, row] pairs."}
        for spin, value in ((self._p1x, p1[0]), (self._p1y, p1[1]),
                            (self._p2x, p2[0]), (self._p2y, p2[1])):
            spin.blockSignals(True)
            spin.setValue(int(value))
            spin.blockSignals(False)
        self._apply_manual_points()
        return {"status": "ok", "profile": {"p1": self._point1, "p2": self._point2}}

    def _agent_select_methods(self, methods: Any) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        if not isinstance(methods, list) or not methods:
            return {"status": "failed", "error": "Provide 'methods' as a non-empty list.",
                    "valid": list(_METHODS)}
        invalid = [m for m in methods if m not in _METHODS]
        if invalid:
            return {"status": "failed", "error": f"Invalid methods {invalid}.",
                    "valid": list(_METHODS)}
        for name, box in self._method_boxes.items():
            box.blockSignals(True)
            box.setChecked(name in methods)
            box.blockSignals(False)
        self._on_methods_changed()
        self._agent_parameters_confirmed = False
        self._agent_parameter_mode = ""
        self._go_to(3)
        return {"status": "ok", "selected": self._collect_methods(),
                "parameter_defaults": self._agent_parameter_defaults(),
                "parameter_confirmation_required": True}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_scheme(value: Any) -> None:
            index = self._ert_scheme.findData(str(value))
            if index < 0:
                index = self._ert_scheme.findText(str(value))
            if index < 0:
                raise ValueError(f"must be one of {[code for _label, code in _ERT_ARRAYS]}")
            self._ert_scheme.setCurrentIndex(index)

        # The page shows noise and errors in %, and short times in ms; the
        # assistant speaks the pipeline's fractions and seconds.
        def percent(box):
            return lambda v: box.setValue(float(v) * 100.0)

        def millis(box):
            return lambda v: box.setValue(float(v) * 1000.0)

        handlers = {
            "snapshot_index": lambda v: self._snapshot.setValue(int(v)),
            "num_samples": lambda v: self._num_samples.setValue(int(v)),
            "seed": lambda v: self._seed.setValue(int(v)),
            "ert.num_electrodes": lambda v: self._ert_count.setValue(int(v)),
            "ert.electrode_spacing": lambda v: self._ert_spacing.setValue(float(v)),
            "ert.electrode_start": lambda v: self._ert_start.setValue(float(v)),
            "ert.scheme_name": set_scheme,
            "ert.noise_level": percent(self._ert_noise),
            "ert.rel_error": percent(self._ert_rel),
            "ert.abs_error": lambda v: self._ert_abs.setValue(float(v)),
            "srt.num_sensors": lambda v: self._srt_count.setValue(int(v)),
            "srt.sensor_spacing": lambda v: self._srt_spacing.setValue(float(v)),
            "srt.sensor_start": lambda v: self._srt_start.setValue(float(v)),
            "srt.shot_distance": lambda v: self._srt_shot.setValue(int(v)),
            "srt.noise_level": percent(self._srt_noise),
            "srt.noise_abs": millis(self._srt_noise_abs),
            "tdem.t_min_s": millis(self._tdem_tmin),
            "tdem.t_max_s": millis(self._tdem_tmax),
            "tdem.n_gates": lambda v: self._tdem_ngates.setValue(int(v)),
            "tdem.noise_level": percent(self._tdem_noise),
            "fdem.f_min_hz": lambda v: self._fdem_fmin.setValue(float(v)),
            "fdem.f_max_hz": lambda v: self._fdem_fmax.setValue(float(v)),
            "fdem.n_freqs": lambda v: self._fdem_nfreq.setValue(int(v)),
            "fdem.noise_level": percent(self._fdem_noise),
            "gravity.sensor_height": lambda v: self._grav_sensor_h.setValue(float(v)),
            "gravity.noise_level": percent(self._grav_noise),
            "em_stations": lambda v: self._em_stations.setValue(int(v)),
        }
        applied: Dict[str, Any] = {}
        ignored: Dict[str, str] = {}
        for key, value in params.items():
            handler = handlers.get(key)
            if handler is None:
                ignored[key] = "unknown parameter"
                continue
            try:
                handler(value)
                applied[key] = value
            except Exception as exc:  # noqa: BLE001
                ignored[key] = str(exc)
        if applied:
            self._agent_parameters_confirmed = False
            self._agent_parameter_mode = ""
        return {"status": "ok" if applied else "failed", "applied": applied, "ignored": ignored}

    def _agent_confirm_parameters(self, mode: Any) -> Dict[str, Any]:
        """Record the user's explicit default/custom parameter confirmation."""
        if mode not in ("defaults", "custom"):
            return {"status": "failed", "error": "mode must be 'defaults' or 'custom'."}
        self._agent_parameters_confirmed = True
        self._agent_parameter_mode = str(mode)
        return {"status": "ok", "mode": self._agent_parameter_mode,
                "methods": self._collect_methods(), "parameter_defaults": self._agent_parameter_defaults()}

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        source_required = self._agent_require_data_source()
        if source_required:
            return source_required
        status = self._step_status()
        if not status["all"]:
            missing = [k for k in ("data", "profile", "methods") if not status[k]]
            return {"status": "failed", "error": "Not ready to run.", "missing": missing,
                    "hint": "Load data, pick a profile, and select methods first."}
        if not self._agent_parameters_confirmed:
            return {"status": "failed",
                    "error": "Confirm the user's parameter choice before running.",
                    "required_action": "confirm_parameters",
                    "parameter_defaults": self._agent_parameter_defaults()}
        self._run_forward()
        return {"status": "started",
                "message": "Forward modeling started. Ask for status shortly.",
                "methods": self._collect_methods()}
