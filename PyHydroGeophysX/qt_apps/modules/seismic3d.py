"""Seismic → Structure: a 3D velocity model and bedrock surface from 2D seismic lines.

Laid out like Hydro → Geophysics and ERT → Water Content. The views are in
the middle: the **Map** of the lines, each placed by its endpoints; the
**Section** of one line, its velocity with the bedrock top drawn where the
velocity passes the threshold, redrawn as the threshold changes; and after a
build the **3D model** and its **Figures**. The settings are on the right in
the order the work is done - the lines, the bedrock interface, the 3D grid -
with the Build button pinned under them and what still stops it said above
it. Before a line is added the middle says what the page does and how to
start, and afterwards a line above the views says what to do next.

The numerics are reused from ``PyHydroGeophysX.Geophy_modular.structure_integration``
(which in turn reuses ``seismic_processor`` and ``core.kriging_3d``); the build
runs in a worker process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.data_processing import run_inputs
from PyHydroGeophysX.Geophy_modular import structure_integration as seismic3d_pipeline
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
from PyHydroGeophysX.qt_apps.widgets.color_range import ColorRange
from PyHydroGeophysX.qt_apps.widgets.guided import (
    NextStepStrip,
    fold,
    hint as _hint,
    number_box as _number,
    start_card,
)
from PyHydroGeophysX.qt_apps.widgets.image_view import ZoomableImageView
from PyHydroGeophysX.qt_apps.widgets.model3d_view import VTKVolumeView
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row
from PyHydroGeophysX.qt_apps.widgets.run_controls import progress_with_stop
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker
from PyHydroGeophysX.visualization.axis_units import set_length_axis, set_section_axes
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

#: The steps the assistant names. Each opens the view and the panel the step is
#: done in (see _go_to); the page itself no longer pages through them.
_STEPS = ["Lines", "Structure", "Grid", "Build", "Results"]
_MAP_TAB, _SECTION_TAB, _MODEL_TAB, _FIGURES_TAB = range(4)
#: The lines table: name, then the two endpoints in map coordinates.
_COLUMNS = ["Name", "Start x", "Start y", "End x", "End y"]
#: Where a line's mesh and velocity paths are kept, on its name cell.
_FILES_ROLE = Qt.UserRole
_INK = "#1d1d1f"
_INK_MUTED = "#6e6e73"

# Synthetic 3D seismic demo: three non-parallel velocity sections produced by
# ``examples/generate_synthetic_examples.py``. The folder names and map endpoints
# below must stay in sync with ``SEISMIC_LINES`` in that generator. A section's
# distance zero is that line's ``p2`` - profiles run from point2 toward point1
# (core.interpolation.setup_profile_coordinates) - so (x0, y0) here is ``p2``
# and (x1, y1) is ``p1``. Placed the other way round, each line lands mirrored
# and its interface misses the 3-D truth by 3 to 12 m instead of a few tenths.
_SYNTHETIC_LINES = [
    {"name": "L1", "folder": "line1", "x0": 105.0, "y0": 25.0, "x1": 15.0, "y1": 20.0},
    {"name": "L2", "folder": "line2", "x0": 100.0, "y0": 65.0, "x1": 20.0, "y1": 70.0},
    {"name": "L3", "folder": "line3", "x0": 35.0, "y0": 75.0, "x1": 30.0, "y1": 15.0},
]

_INPUT_FORMAT_FALLBACK = (
    "# Seismic → Structure input format\n\n"
    "Add one or more 2D seismic velocity lines. Each line needs:\n\n"
    "- a pyGIMLi mesh (`velmesh.bms`)\n"
    "- a per-cell velocity array (`Vinvmodel.npy`)\n"
    "- map coordinates of the line endpoints (x0, y0) → (x1, y1)\n\n"
    "See seismic3d_input_format.md for details."
)


class _LineMapView(QWidget):
    """The lines in map coordinates: each from its start (distance 0) to its end."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._figure = Figure(figsize=(7, 5), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        bar, self._toolbar = toolbar_row(self._canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(bar)
        layout.addWidget(self._canvas, stretch=1)
        self._lines: List[Dict[str, Any]] = []
        self._selected = -1
        length_units.notifier().changed.connect(lambda _unit: self.draw())

    def show_lines(self, lines: List[Dict[str, Any]], selected: int) -> None:
        self._lines, self._selected = list(lines), int(selected)
        self.draw()

    def draw(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        for i, line in enumerate(self._lines):
            x0, y0, x1, y1 = (float(line[k]) for k in ("x0", "y0", "x1", "y1"))
            chosen = i == self._selected
            colour = theme.PALETTE["primary"] if chosen else _INK
            width = 2.4 if chosen else 1.6
            # The line itself, so the axes take in both ends; an annotation
            # alone is left out of the limits and, its end outside them, clipped.
            ax.plot([x0, x1], [y0, y1], color=colour, lw=width)
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), annotation_clip=False,
                        arrowprops={"arrowstyle": "-|>", "color": colour, "lw": width,
                                    "mutation_scale": 18})
            ax.plot([x0], [y0], "o", color=colour, ms=7)
            ax.annotate(str(line.get("name", i + 1)), (0.5 * (x0 + x1), 0.5 * (y0 + y1)),
                        textcoords="offset points", xytext=(6, 6), color=colour,
                        fontsize=10, fontweight="bold" if chosen else "normal")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.12)
        ax.grid(True, alpha=0.3)
        unit = length_units.current()
        set_length_axis(ax, "x", "Easting", unit=unit)
        set_length_axis(ax, "y", "Northing", unit=unit)
        ax.set_title("Seismic lines: a dot marks distance 0, the arrow points along the line")
        self._canvas.draw_idle()
        self._toolbar.update()


class _VelocitySectionView(QWidget):
    """One line's velocity on its mesh, with the bedrock top at the threshold."""

    def __init__(self, colormaps: Any, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._colormap = cmaps.ColormapChooser(cmaps.VELOCITY, "turbo", shared=colormaps)
        self._colormap.colormapChanged.connect(lambda _name: self.draw())
        self._range = ColorRange(decimals=0, what="the velocity colours")
        self._range.changed.connect(self.draw)
        self._figure = Figure(figsize=(8, 4), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        bar, self._toolbar = toolbar_row(self._canvas, self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Line"))
        self.line_choice = QComboBox()
        controls.addWidget(self.line_choice)
        controls.addStretch(1)
        controls.addWidget(self._colormap)
        controls.addWidget(self._range)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(bar)
        layout.addWidget(self._canvas, stretch=1)
        self._data: Optional[Dict[str, Any]] = None
        self._message = "Add a seismic line to see its velocity section."
        length_units.notifier().changed.connect(lambda _unit: self.draw())

    def show_section(self, data: Optional[Dict[str, Any]], message: str = "") -> None:
        """``data``: name, polygons, velocity, interface (n, 2), threshold, mesh."""
        self._data, self._message = data, message
        self.draw()

    def draw(self) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        data = self._data
        if not data:
            ax.set_axis_off()
            ax.text(0.5, 0.5, self._message, ha="center", va="center",
                    transform=ax.transAxes, color=_INK_MUTED)
        else:
            velocity = np.asarray(data["velocity"], dtype=float)
            lo, hi = self._range.limits(float(np.nanpercentile(velocity, 2)),
                                        float(np.nanpercentile(velocity, 98)))
            cells = PolyCollection(data["polygons"], array=velocity,
                                   cmap=self._colormap.colormap(), edgecolors="none")
            cells.set_clim(lo, hi)
            ax.add_collection(cells)
            ax.autoscale_view()
            self._figure.colorbar(cells, ax=ax, pad=0.01).set_label("Velocity (m/s)")
            interface = data.get("interface")
            if interface is not None and len(interface):
                ax.plot(interface[:, 0], interface[:, 1], color="white", lw=3.2)
                ax.plot(interface[:, 0], interface[:, 1], color=_INK, lw=1.6,
                        label=f"Bedrock top ({data['threshold']:.0f} m/s)")
                ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
            set_section_axes(ax, mesh=data.get("mesh"), unit=length_units.current())
            ax.set_title(f"{data['name']}: velocity, and the bedrock top where it passes "
                         f"{data['threshold']:.0f} m/s")
        self._canvas.draw_idle()
        self._toolbar.update()


class Seismic3DModule(BaseModule):
    module_key = "seismic3d"
    module_title = "Seismic → Structure"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._lines: List[Dict[str, Any]] = []
        #: Per (mesh, velocity) path pair: the mesh, its velocity and cell outlines.
        self._line_cache: Dict[tuple, Dict[str, Any]] = {}
        self._worker: Optional[ProcessWorkflowWorker] = None
        self._run_busy: Optional[BusyStateController] = None
        self._workflow_recipe_path = ""
        #: Why the last run did not finish - its error, or that it was
        #: stopped - for the assistant, whose status otherwise still showed
        #: the result before it; "" while a run is under way or succeeded.
        self._run_problem = ""
        self._has_results = False
        self._results_tab = _MODEL_TAB
        self._current = 0
        self._section_debounced = Debouncer(self._show_section, 250)

        root = QHBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_map_tab(), "Map")
        self._section_view = _VelocitySectionView(cmaps.colormap_settings(self.state))
        self._section_view.line_choice.currentIndexChanged.connect(self._on_line_chosen)
        self._tabs.addTab(self._section_view, "Section")
        # A velocity volume, so it takes the colour map chosen for velocity
        # models - the seismic page's sections use the same one.
        self._volume_view = VTKVolumeView(colormaps=cmaps.colormap_settings(self.state),
                                          colormap_key=cmaps.VELOCITY)
        self._tabs.addTab(self._volume_view, "3D model")
        self._tabs.addTab(self._build_figures_tab(), "Figures")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._guide = NextStepStrip()
        self._reproduce = ReproduceBar()
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self._guide)
        center_layout.addWidget(self._tabs, stretch=1)
        center_layout.addWidget(self._reproduce)
        root.addWidget(center, stretch=1)
        root.addWidget(self._build_controls())
        self._on_lines_changed()

    # -- centre: the map, a section, the 3D model ----------------------------
    def _build_start_card(self) -> QWidget:
        example_btn = QPushButton("Use example")
        example_btn.setProperty("primary", True)
        example_btn.setIcon(theme.icon("fa5s.box-open", color="#ffffff"))
        example_btn.setToolTip("Add three synthetic velocity lines that cross one site.")
        example_btn.clicked.connect(self._use_example)
        add_btn = QPushButton("Add a line…")
        add_btn.setIcon(theme.icon("fa5s.folder-plus"))
        add_btn.clicked.connect(self._add_line_folder)
        format_btn = QPushButton("Data format")
        format_btn.setIcon(theme.icon("fa5s.question-circle"))
        format_btn.clicked.connect(self._show_format_help)
        return start_card(
            "Build a 3D model from seismic lines",
            "Join 2D seismic velocity sections into a 3D velocity model and a bedrock "
            "surface, for a hydrologic model or to split an ERT model into layers.",
            ["Add the seismic lines: each line's velocity section, and where it lies "
             "on the map.",
             "Choose the velocity that marks the top of bedrock, and check it on each "
             "line's section.",
             "Set the 3D grid, and press Build 3D model."],
            [example_btn, add_btn, format_btn],
            note="The example is three synthetic lines crossing one site.")

    def _build_map_tab(self) -> QWidget:
        # The start card until a line is added, then the map of the lines.
        self._map_stack = QStackedWidget()
        self._map_stack.addWidget(self._build_start_card())
        self._map_view = _LineMapView()
        self._map_stack.addWidget(self._map_view)
        return self._map_stack

    def _build_figures_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        header.addWidget(QLabel("Show"))
        self._figure_choice = QComboBox()
        self._figure_choice.currentIndexChanged.connect(self._show_selected_figure)
        header.addWidget(self._figure_choice, stretch=1)
        open_btn = QPushButton("Open output folder")
        open_btn.setIcon(theme.icon("fa5s.folder-open"))
        open_btn.clicked.connect(self._open_output_folder)
        header.addWidget(open_btn)
        layout.addLayout(header)
        self._result_summary = QLabel("")
        self._result_summary.setWordWrap(True)
        layout.addWidget(self._result_summary)
        self._figure_view = ZoomableImageView()
        self._figure_view.setMinimumHeight(380)
        layout.addWidget(self._figure_view, stretch=1)
        self._result_files = _hint()
        layout.addWidget(self._result_files)
        return page

    # -- right: the settings, in the order the work is done ------------------
    def _build_controls(self) -> QWidget:
        scroll = ContentWidthScrollArea(minimum=430, maximum=480)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        self._lines_group = self._build_lines_group()
        self._structure_group = self._build_structure_group()
        self._grid_group = self._build_grid_group()
        #: Each numbered step's title; a finished step gets a tick after it.
        self._step_titles = {
            self._lines_group: "1  Seismic lines",
            self._structure_group: "2  Bedrock interface",
            self._grid_group: "3  3D grid",
        }
        for group in self._step_titles:
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
        self._run_btn = QPushButton("Build 3D model")
        self._run_btn.setProperty("primary", True)
        self._run_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._run_btn.setToolTip("Interpolate the lines' velocity and their bedrock tops "
                                 "across the site, onto the 3D grid.")
        self._run_btn.clicked.connect(self._run_build)
        flayout.addWidget(self._run_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._stop = self.stop_button("The 3D model build")
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

    def _build_lines_group(self) -> QGroupBox:
        box = QGroupBox("1  Seismic lines")
        layout = QVBoxLayout(box)
        add_btn = QPushButton("Add a line…")
        add_btn.setProperty("primary", True)
        add_btn.setIcon(theme.icon("fa5s.folder-plus", color="#ffffff"))
        add_btn.setToolTip("Add a line from a folder with velmesh.bms and Vinvmodel.npy.")
        add_btn.clicked.connect(self._add_line_folder)
        example_btn = QPushButton("Use example")
        example_btn.setIcon(theme.icon("fa5s.box-open"))
        example_btn.setToolTip("Add three synthetic velocity lines that cross one site.")
        example_btn.clicked.connect(self._use_example)
        files_btn = QPushButton()
        files_btn.setIcon(theme.icon("fa5s.file-medical"))
        files_btn.setToolTip("Add a line from a mesh file and a velocity file chosen "
                             "one by one.")
        files_btn.clicked.connect(self._add_line_files)
        help_btn = QPushButton()
        help_btn.setIcon(theme.icon("fa5s.question-circle"))
        help_btn.setToolTip("What a line needs.")
        help_btn.clicked.connect(self._show_format_help)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        for button in (add_btn, example_btn, files_btn, help_btn):
            row.addWidget(button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(_hint("Place each line on the map by its two ends, in metres. "
                               "Start is where its section's distance 0 lies."))
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setMinimumHeight(140)
        self._table.itemChanged.connect(lambda *_: self._on_lines_changed())
        self._table.itemSelectionChanged.connect(self._on_table_selection)
        layout.addWidget(self._table)
        remove_btn = QPushButton("Remove selected")
        remove_btn.setIcon(theme.icon("fa5s.trash"))
        remove_btn.clicked.connect(self._remove_selected)
        layout.addWidget(merged_row(remove_btn))
        return box

    def _build_structure_group(self) -> QGroupBox:
        box = QGroupBox("2  Bedrock interface")
        form = QFormLayout(box)
        form.addRow(_hint("The top of bedrock is where the velocity first passes this "
                          "value going down. The Section tab draws it on each line."))
        self._threshold = _number(1200.0, 200.0, 8000.0, 50.0, 0, " m/s",
                                  "About 1200 m/s for the regolith / bedrock boundary.")
        self._interval = _number(4.0, 0.5, 50.0, 0.5, 1, " m",
                                 "How often along the line the interface is picked.")
        for box_ in (self._threshold, self._interval):
            box_.valueChanged.connect(lambda *_: self._section_debounced.trigger())
            box_.valueChanged.connect(lambda *_: self._sync_page())
        form.addRow("Bedrock top at", self._threshold)
        form.addRow("Pick every", self._interval)
        handoff = QPushButton("Send to ERT → Water Content")
        handoff.setIcon(theme.icon("fa5s.share"))
        handoff.setToolTip("Send the bedrock top of the line on the Section tab to ERT → "
                           "Water Content, to split its model into regolith and bedrock.")
        handoff.clicked.connect(self._send_structure_to_hydrology)
        form.addRow(merged_row(handoff))
        return box

    def _build_grid_group(self) -> QGroupBox:
        box = QGroupBox("3  3D grid")
        form = QFormLayout(box)
        self._grid_res = make_spinbox(50, 10, 200, tooltip="Cells along each map axis.")
        self._depth = _number(50.0, 5.0, 500.0, 5.0, 1, " m", "How deep the model goes.")
        self._n_layers = make_spinbox(19, 2, 100, tooltip="Layers of cells down to that depth.")
        self._interp = QComboBox()
        self._interp.addItem("Linear (griddata)", "griddata")
        self._interp.addItem("Kriging", "kriging")
        try:
            from PyHydroGeophysX.core.kriging_3d import GSTOOLS_AVAILABLE
        except Exception:  # noqa: BLE001
            GSTOOLS_AVAILABLE = False
        if not GSTOOLS_AVAILABLE:
            self._interp.model().item(self._interp.findData("kriging")).setEnabled(False)
            self._interp.setToolTip("Kriging needs the optional 'gstools' package; "
                                    "linear interpolation is used otherwise.")
        self._interp.currentIndexChanged.connect(self._on_interp_changed)
        form.addRow("Cells across", merged_row(self._grid_res, "× each way"))
        form.addRow("Depth", self._depth)
        form.addRow("Layers", self._n_layers)
        form.addRow("Between lines", self._interp)
        adv, aform = fold()
        self._z_scale = _number(1.0, 0.1, 50.0, 0.5, 1, tip=(
            "Vertical exaggeration during the 3D interpolation, so thin vertical "
            "structure is not smoothed away."))
        self._max_points = make_spinbox(
            40000, 2000, 500000, tooltip="Velocity samples are thinned to this many "
                                         "before the 3D interpolation, to keep it quick.")
        self._max_points.setSingleStep(5000)
        aform.addRow("Vertical scale", self._z_scale)
        aform.addRow("Velocity samples", self._max_points)
        d = seismic3d_pipeline.DEFAULT_KRIGING
        self._krig_model = QComboBox()
        self._krig_model.addItems(list(seismic3d_pipeline.VARIOGRAM_MODELS))
        self._krig_model.setCurrentText(d["model"])
        self._len_x = _number(d["len_scale_x"], 1.0, 100000.0, 10.0, 1, " m")
        self._len_y = _number(d["len_scale_y"], 1.0, 100000.0, 10.0, 1, " m")
        self._len_z = _number(d["len_scale_z"], 0.5, 10000.0, 1.0, 1, " m")
        self._krig_var = _number(d["var"], 0.001, 100.0, 0.1, 3)
        self._krig_nugget = _number(d["nugget"], 0.0, 100.0, 0.05, 3)
        self._krig_rows = [self._krig_model, self._len_x, self._len_y, self._len_z,
                           self._krig_var, self._krig_nugget]
        aform.addRow("Kriging model", self._krig_model)
        aform.addRow("Range east", self._len_x)
        aform.addRow("Range north", self._len_y)
        aform.addRow("Range down", self._len_z)
        aform.addRow("Variance (sill)", self._krig_var)
        aform.addRow("Nugget", self._krig_nugget)
        export_btn = QPushButton("Save 3D settings")
        export_btn.setIcon(theme.icon("fa5s.file-export"))
        export_btn.setToolTip("Write the lines and settings to seismic3d_config.json, "
                              "without building anything.")
        export_btn.clicked.connect(self._export_config)
        aform.addRow(merged_row(export_btn))
        form.addRow(adv)
        self._on_interp_changed()
        return box

    def _on_interp_changed(self, *_args) -> None:
        """The variogram rows only mean something for kriging."""
        set_rows_visible(self._krig_rows, self._interp.currentData() == "kriging")

    # -- navigation (the assistant's step names) -----------------------------
    def _go_to(self, step: int) -> None:
        """Show the view and the panel the step is done in."""
        step = max(0, min(int(step), len(_STEPS) - 1))
        lines = bool(self._lines)
        tab = {0: _MAP_TAB, 1: _SECTION_TAB if lines else _MAP_TAB, 2: _MAP_TAB,
               3: _SECTION_TAB if lines else _MAP_TAB,
               4: self._results_tab if self._has_results else _MAP_TAB}[step]
        self._tabs.setCurrentIndex(tab)
        self._current = step
        group = {0: self._lines_group, 1: self._structure_group,
                 2: self._grid_group}.get(step)
        if group is not None:
            self._controls_scroll.ensureWidgetVisible(group)

    def _on_tab_changed(self, index: int) -> None:
        self._current = {_MAP_TAB: 0, _SECTION_TAB: 1}.get(index, 4)

    def _step_status(self) -> Dict[str, bool]:
        self._sync_lines_from_table()
        lines_ok = len(self._lines) > 0
        ran = self.state.module_results.get(self.module_key, {}).get("status") == "ok"
        return {"lines": lines_ok, "all": lines_ok, "results": ran}

    # -- readiness and guidance ----------------------------------------------
    def _run_blocker(self) -> str:
        """What still stops a build, in a sentence; "" when it can run."""
        if not self._lines:
            return "Add at least one seismic line (1) to build a 3D model."
        return ""

    def _next_step(self) -> str:
        """The one thing to do now, for the line above the views."""
        if self._run_busy is not None:
            return ("Building the 3D model: interpolating the velocity between the lines. "
                    "This can take a few minutes.")
        if not self._lines:
            return "Start by adding seismic lines: use the example or add a line."
        if len(self._lines) == 1:
            return ("One line gives the 3D model no width: add the others (1), placing "
                    "each by its two ends, then press Build 3D model.")
        if self._has_results:
            return (f"Done: the model is on the {self._results_place()} tab. Send a line's "
                    "bedrock top to ERT → Water Content (2) to split its model into layers.")
        return ("Ready: check the bedrock top on each line's Section (2), then press Build "
                "3D model.")

    def _sync_page(self) -> None:
        """Bring every cue on the page in line with where the work stands."""
        if not hasattr(self, "_run_reason"):
            return
        reason = self._run_blocker()
        self._run_reason.setText(reason)
        self._run_reason.setVisible(bool(reason))
        if self._run_busy is None:
            self._run_btn.setEnabled(not reason)
        lines = bool(self._lines)
        self._map_stack.setCurrentIndex(1 if lines else 0)
        self._guide.setVisible(lines or self._run_busy is not None)
        self._guide.setText(self._next_step())
        for group in (self._structure_group, self._grid_group):
            group.setEnabled(lines)
        done = {self._lines_group: lines, self._structure_group: lines,
                self._grid_group: lines}
        for group, title in self._step_titles.items():
            group.setTitle(f"{title}  ✓" if done[group] else title)
        self._tabs.setTabEnabled(_SECTION_TAB, lines)
        self._tabs.setTabEnabled(_MODEL_TAB, self._has_results)
        self._tabs.setTabEnabled(_FIGURES_TAB, self._has_results)

    # -- lines ---------------------------------------------------------------
    def _add_row(self, line: Dict[str, Any]) -> None:
        r = self._table.rowCount()
        self._table.blockSignals(True)
        self._table.insertRow(r)
        values = [line.get("name", f"L{r + 1}"), line.get("x0", 0.0), line.get("y0", 0.0),
                  line.get("x1", 100.0), line.get("y1", 0.0)]
        for c, value in enumerate(values):
            item = QTableWidgetItem(f"{value:g}" if isinstance(value, float) else str(value))
            if c == 0:
                item.setData(_FILES_ROLE, (str(line.get("mesh", "")),
                                           str(line.get("velocity", ""))))
                item.setToolTip(f"{line.get('mesh', '')}\n{line.get('velocity', '')}")
            self._table.setItem(r, c, item)
        self._table.blockSignals(False)
        self._on_lines_changed()

    def _sync_lines_from_table(self) -> None:
        lines: List[Dict[str, Any]] = []
        for r in range(self._table.rowCount()):
            def cell(c, default=""):
                it = self._table.item(r, c)
                return it.text() if it is not None else default
            name_item = self._table.item(r, 0)
            mesh, velocity = (name_item.data(_FILES_ROLE) if name_item is not None
                              else ("", "")) or ("", "")
            try:
                line = {"name": cell(0, f"L{r + 1}"),
                        "x0": float(cell(1, 0)), "y0": float(cell(2, 0)),
                        "x1": float(cell(3, 0)), "y1": float(cell(4, 0)),
                        "mesh": mesh, "velocity": velocity}
            except ValueError:
                continue
            if line["mesh"] and line["velocity"]:
                lines.append(line)
        self._lines = lines

    def _on_lines_changed(self) -> None:
        """A line was added, removed, renamed or moved: redraw what shows them."""
        self._sync_lines_from_table()
        choice = self._section_view.line_choice
        current = choice.currentText()
        choice.blockSignals(True)
        choice.clear()
        for line in self._lines:
            choice.addItem(str(line["name"]))
        index = choice.findText(current)
        choice.setCurrentIndex(index if index >= 0 else (0 if self._lines else -1))
        choice.blockSignals(False)
        self._map_view.show_lines(self._lines, choice.currentIndex())
        self._section_debounced.trigger()
        self._sync_page()

    def _on_table_selection(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()})
        if rows and rows[0] < self._section_view.line_choice.count():
            self._section_view.line_choice.setCurrentIndex(rows[0])

    def _on_line_chosen(self, index: int) -> None:
        self._map_view.show_lines(self._lines, index)
        self._section_debounced.trigger()

    def _chosen_line(self) -> Optional[Dict[str, Any]]:
        index = self._section_view.line_choice.currentIndex()
        return self._lines[index] if 0 <= index < len(self._lines) else None

    def _line_data(self, line: Dict[str, Any]) -> Dict[str, Any]:
        """The line's mesh, velocity and cell outlines, read once."""
        key = (line["mesh"], line["velocity"])
        if key not in self._line_cache:
            from PyHydroGeophysX.core.mesh_serialization import read_bms

            mesh = read_bms(line["mesh"])
            velocity = seismic3d_pipeline._load_velocity(Path(line["velocity"]))
            if velocity.size != mesh.cellCount():
                raise ValueError(f"the velocity has {velocity.size} values for "
                                 f"{mesh.cellCount()} mesh cells")
            polygons = [np.array([[n.pos()[0], n.pos()[1]] for n in cell.nodes()])
                        for cell in mesh.cells()]
            self._line_cache[key] = {"mesh": mesh, "velocity": velocity, "polygons": polygons}
        return self._line_cache[key]

    def _show_section(self) -> None:
        """Draw the chosen line's velocity and its bedrock top at the threshold."""
        line = self._chosen_line()
        if line is None:
            self._section_view.show_section(None)
            return
        try:
            from PyHydroGeophysX.Geophy_modular.seismic_processor import (
                extract_velocity_structure,
            )

            data = self._line_data(line)
            x, z, _info = extract_velocity_structure(
                data["mesh"], data["velocity"], self._threshold.value(), self._interval.value())
            interface = np.column_stack([np.asarray(x, dtype=float), np.asarray(z, dtype=float)])
        except Exception as exc:  # noqa: BLE001 - the view says why
            self._section_view.show_section(None, f"{line['name']} could not be drawn: {exc}")
            return
        self._section_view.show_section({
            "name": line["name"], "polygons": data["polygons"], "velocity": data["velocity"],
            "interface": interface, "threshold": self._threshold.value(), "mesh": data["mesh"],
        })

    def _example_dir(self) -> Optional[Path]:
        """Locate the synthetic seismic demo root holding ``line1/``, ``line2/``, ...

        The dataset ``examples/generate_synthetic_examples.py`` writes in a clone
        of the source repository; the pip package carries no example data.
        """
        first = str(_SYNTHETIC_LINES[0]["folder"])

        def usable(folder: Path) -> bool:
            files = seismic3d_pipeline.find_velocity_files(folder / first)
            return files.get("mesh") is not None and files.get("velocity") is not None

        return io_utils.find_example("results/synthetic_seismic", self.state.project_root, usable)

    def _use_example(self) -> None:
        root = self._example_dir()
        if root is None:
            self.log(io_utils.missing_example_message(
                "The synthetic seismic lines example", "results/synthetic_seismic",
                generated=True), "warn")
            return
        added = 0
        for spec in _SYNTHETIC_LINES:
            files = seismic3d_pipeline.find_velocity_files(root / str(spec["folder"]))
            if files.get("mesh") is None or files.get("velocity") is None:
                continue
            self._add_row({"name": spec["name"], "mesh": str(files["mesh"]),
                           "velocity": str(files["velocity"]),
                           "x0": spec["x0"], "y0": spec["y0"],
                           "x1": spec["x1"], "y1": spec["y1"]})
            added += 1
        if added:
            self.log(f"Added {added} synthetic example line(s) from {root}", "success")
        else:
            self.log(f"No synthetic velocity lines found under {root}.", "warn")

    def _add_line_folder(self) -> None:
        path = select_directory(self, "Select velocity-model folder", Path.cwd())
        if not path:
            return
        files = seismic3d_pipeline.find_velocity_files(Path(path))
        if files.get("mesh") is None or files.get("velocity") is None:
            self.log(f"Folder must contain {seismic3d_pipeline.VELOCITY_FILES['mesh']} and "
                     f"{seismic3d_pipeline.VELOCITY_FILES['velocity']}.", "warn")
            return
        r = self._table.rowCount()
        self._add_row({"name": f"L{r + 1}", "mesh": str(files["mesh"]),
                       "velocity": str(files["velocity"]), "x0": 0.0, "y0": float(r * 50),
                       "x1": 100.0, "y1": float(r * 50)})
        self.log("Line added at a placeholder position: type its two ends in the table.",
                 "info")

    def _add_line_files(self) -> None:
        mesh, _ = QFileDialog.getOpenFileName(self, "Select mesh", "", "Mesh (*.bms);;All files (*)")
        if not mesh:
            return
        vel, _ = QFileDialog.getOpenFileName(self, "Select velocity array", "", "Array (*.npy);;All files (*)")
        if not vel:
            return
        r = self._table.rowCount()
        self._add_row({"name": f"L{r + 1}", "mesh": mesh, "velocity": vel,
                       "x0": 0.0, "y0": float(r * 50), "x1": 100.0, "y1": float(r * 50)})

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self._table.selectedIndexes()}, reverse=True)
        self._table.blockSignals(True)
        for r in rows:
            self._table.removeRow(r)
        self._table.blockSignals(False)
        self._on_lines_changed()

    # -- bedrock handoff -----------------------------------------------------
    def _send_structure_to_hydrology(self) -> None:
        self._sync_lines_from_table()
        line = self._chosen_line()
        if line is None:
            self.log("Add at least one seismic line first.", "warn")
            return
        try:
            res = seismic3d_pipeline.extract_line_structure(
                line, self._threshold.value(), self._interval.value(), log=lambda m: None)
            raw = res["raw_interface"]
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not extract bedrock interface: {exc}", "error")
            return
        self.state.shared_structure = {
            "interface_xz": [[float(x), float(z)] for x, z in raw],
            "source": f"seismic:{line.get('name', 'line')}",
            "threshold": float(self._threshold.value()),
        }
        if hasattr(self.state, "register_geophysical_resource"):
            self.state.register_geophysical_resource(
                "SRT", "interface", np.asarray(raw, dtype=float),
                label=f"Seismic interface · {line.get('name', 'line')}",
                metadata={"threshold": float(self._threshold.value())},
                resource_id="srt:interface:active",
            )
        self.log(f"Sent bedrock structure ({len(raw)} pts, {self._threshold.value():.0f} m/s) "
                 f"to ERT → Water Content. Use 'Layers from a seismic bedrock "
                 f"interface' under its Layers panel.", "success")
        self.navigateRequested.emit("geo_hydrology")

    # -- results -------------------------------------------------------------
    def _populate_results(self, result: Dict[str, Any]) -> None:
        # Settings saved in place of a build leave nothing to show; the line
        # under the Build button says so.
        built = result.get("status") != "config_exported"
        self._result_summary.setText(
            f"Built from {result.get('n_lines', '—')} lines on a "
            f"{result.get('grid_resolution', '—')} × {result.get('grid_resolution', '—')} "
            f"grid: velocity {self._fmt_range(result.get('velocity_range'))} m/s, "
            f"bedrock {self._fmt_range(result.get('bedrock_depth_range'))} m deep.")
        self._figure_choice.blockSignals(True)
        self._figure_choice.clear()
        for fig in result.get("figure_paths", []):
            if Path(fig).exists():
                self._figure_choice.addItem(Path(fig).stem.replace("_", " ").capitalize(),
                                            str(fig))
        self._figure_choice.blockSignals(False)
        if self._figure_choice.count():
            self._figure_choice.setCurrentIndex(0)
        self._show_selected_figure()
        bits = []
        vtk = result.get("vtk_path")
        rendered = False
        if vtk and Path(vtk).exists():
            rendered = self._volume_view.show_file(
                vtk, scalar_candidates=("Velocity", "velocity"), opacity=0.65)
            bits.append(f"3D volume: {vtk}")
        elif vtk:
            bits.append(f"3D volume file not found: {vtk}")
        if result.get("config_path"):
            bits.append(f"3D settings: {result['config_path']}")
        self._result_files.setText("<br>".join(bits))
        # Without a 3D view (no OpenGL, no VTK) the figures are the result.
        self._results_tab = _MODEL_TAB if rendered else _FIGURES_TAB
        self._has_results = built
        self._sync_page()
        if built:
            self._tabs.setCurrentIndex(self._results_tab)

    def _results_place(self) -> str:
        return "3D model" if self._results_tab == _MODEL_TAB else "Figures"

    def _show_selected_figure(self) -> None:
        path = self._figure_choice.currentData()
        if not (path and Path(path).exists() and self._figure_view.set_image_file(path)):
            self._figure_view.clear()

    @staticmethod
    def _fmt_range(rng) -> str:
        if not rng:
            return "—"
        return f"{rng[0]:.0f}–{rng[1]:.0f}"

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.output_dir or "")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(out))
        else:
            self.log("Output folder not available yet.", "warn")

    # -- format help ---------------------------------------------------------
    def _show_format_help(self) -> None:
        doc_path = Path(__file__).with_name("seismic3d_input_format.md")
        try:
            text = doc_path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            text = _INPUT_FORMAT_FALLBACK
        dlg = QDialog(self); dlg.setWindowTitle("Seismic → Structure input format")
        dlg.resize(760, 620); lay = QVBoxLayout(dlg)
        browser = QTextBrowser(); browser.setOpenExternalLinks(True)
        try:
            browser.setMarkdown(text)
        except Exception:  # noqa: BLE001
            browser.setPlainText(text)
        lay.addWidget(browser)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn); dlg.exec()

    # -- params / run --------------------------------------------------------
    def _context(self) -> Dict[str, Any]:
        ctx = dict(self.state.context or {})
        if self.state.output_dir:
            ctx.setdefault("output_dir", str(self.state.output_dir))
        return ctx

    def _collect_params(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        out_dir = self.state.output_dir
        return {
            "output_dir": str(out_dir) if out_dir else "",
            "lines": self._lines,
            "threshold": self._threshold.value(),
            "interval": self._interval.value(),
            "grid_resolution": self._grid_res.value(),
            "depth": self._depth.value(),
            "n_layers": self._n_layers.value(),
            "interp_method": str(self._interp.currentData()),
            "z_scale": self._z_scale.value(),
            "max_velocity_points": self._max_points.value(),
            "kriging": {
                "model": self._krig_model.currentText(),
                "len_scale_x": self._len_x.value(),
                "len_scale_y": self._len_y.value(),
                "len_scale_z": self._len_z.value(),
                "var": self._krig_var.value(),
                "nugget": self._krig_nugget.value(),
            },
        }

    def _export_config(self) -> str:
        params = self._collect_params()
        config = seismic3d_pipeline.build_seismic3d_config(self._context(), params)
        active = self.state.active_run(self.module_key)
        out_dir = active.outputs_dir if active else self.state.ensure_results_store().scratch_dir(self.module_key)
        config_path = out_dir / "seismic3d_config.json"
        io_utils.write_json(config_path, config)
        self.log(f"Exported config to {config_path}", "success")
        self.report_result({"n_lines": config["n_lines"], "config_path": str(config_path),
                            "status": "config_exported", "output_dir": str(out_dir)})
        return str(config_path)

    def _run_build(self) -> None:
        self._sync_lines_from_table()
        if not self._lines:
            self.log("Add at least one seismic line first.", "warn")
            return
        params = self._collect_params()
        try:
            run = self.begin_persisted_run(
                "seismic3d.build",
                workflow_id="seismic3d.build",
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        # One compressed bundle for every line's mesh and velocity, rather than
        # a folder of copies per line. The bundle member name carries which line
        # and which field a file belongs to, so the run expands back to the same
        # pairing.
        serialized_lines: List[Dict[str, Any]] = []
        members: Dict[str, Path] = {}
        for index, line in enumerate(params.pop("lines")):
            item = dict(line)
            for field in ("mesh", "velocity"):
                path = Path(str(item.get(field, "")))
                if not path.is_file():
                    self.fail_persisted_run(f"Missing input file: {path}")
                    self.log(
                        f"Seismic3D line {index + 1} {field} file does not exist: {path}",
                        "error",
                    )
                    return
                name = f"line_{index + 1}_{field}{path.suffix}"
                members[name] = path
                item[field] = name
            serialized_lines.append(item)
        try:
            stored = run_inputs.save_file_bundle(
                run.inputs_dir / "seismic3d_lines",
                members,
                kind="seismic3d_lines",
                meta={"line_count": len(serialized_lines)},
            )
        except OSError as exc:
            self.fail_persisted_run(str(exc))
            self.log(f"Could not persist the seismic3d line inputs: {exc}", "error")
            return
        params.pop("output_dir", None)
        # The run's figures are drawn in another process, which cannot see the
        # studio's View > Length Units choice.
        params["length_unit"] = length_units.current()
        spec = WorkflowSpec(
            workflow_id="seismic3d.build",
            inputs={
                "context": {},
                "lines": serialized_lines,
                "line_bundle": ArtifactRef.from_path(
                    stored,
                    artifact_id="seismic3d:lines",
                    kind="seismic3d_lines",
                    base_dir=run.run_dir,
                    metadata={"members": sorted(members)},
                ),
            },
            parameters=params,
            metadata={"source": "qt", "line_count": len(serialized_lines)},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="seismic3d"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._workflow_recipe_path = str(recipe_path)
        self._run_busy = BusyStateController([self._run_btn])
        self._run_busy.start()
        self._run_btn.setText("Building…")
        self._progress.setVisible(True); self._progress.setRange(0, 0)
        self._run_status.setText("Building the 3D model…")
        theme.set_tone(self._run_status, None)
        self._sync_page()
        self.log("Starting seismic → 3D model build…", "info")
        # In a process of its own, so the window keeps painting through the
        # kriging; the model comes back as the files it writes.
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir,
                                       run.result_path)
        self._worker = self.register_worker(worker, activity="Building the 3D model")
        self._run_problem = ""
        worker.logged.connect(lambda message: self._on_worker_log(message, "info"))
        worker.succeeded.connect(self._on_workflow_ok)
        worker.failed.connect(
            lambda message, w=worker: self._on_build_failed(message, w.missing_backend))
        worker.finished.connect(self._reset_run_button)
        worker.start()
        self._stop.attach(worker, "seismic3d.build")

    def _on_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "seismic3d.build",
                result.to_dict(),
                recipe_path=self._workflow_recipe_path,
            )
        self._on_build_ok(result.legacy_payload())

    def _on_worker_log(self, message: str, level: str) -> None:
        self._run_status.setText(message); self.log(message, level)

    def _on_build_ok(self, result: dict) -> None:
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self.log(f"3D model built: {result.get('n_grid_points')} grid points.", "success")
        self.report_result(result)
        self._populate_results(result)
        self._run_status.setText(f"Finished. The model is on the {self._results_place()} tab.")
        theme.set_tone(self._run_status, "ok")
        self._current = 4

    def _on_build_failed(self, message: str, backend_unavailable: bool) -> None:
        self._run_problem = message
        self.fail_persisted_run(message, "seismic3d.build")
        self._progress.setRange(0, 1); self._progress.setValue(0)
        if not backend_unavailable:
            # The build itself failed - out of memory, a grid too fine, a bad
            # line file - and its own reason is what helps, beside the settings
            # that can be changed before building again. Exporting the
            # configuration in its place blamed a backend that was there.
            self.log(f"3D model build failed: {message}", "error")
            self._run_status.setText(f"The build failed: {message}")
            theme.set_tone(self._run_status, "error")
            return
        self.log(f"3D model build could not run: {message}", "warn")
        config_path = self._export_config()
        note = ("The 3D engine (pygimli) is not installed or could not be loaded. The "
                f"3D settings were saved to {config_path}.")
        self._run_status.setText(note); self.log(note, "warn")
        theme.set_tone(self._run_status, "warn")
        self._populate_results({"status": "config_exported", "n_lines": len(self._lines),
                                "config_path": config_path, "figure_paths": []})

    def _reset_run_button(self) -> None:
        if self._run_busy is not None:
            self._run_busy.finish()
            self._run_busy = None
        self._run_btn.setText("Build 3D model")
        self._progress.setVisible(False)
        if self._worker is not None and self._worker.is_cancelled():
            self._run_problem = "Stopped by user"
            self._run_status.setText("Stopped before it finished. Press Build 3D model to "
                                     "start again.")
            theme.set_tone(self._run_status, "warn")
        self._sync_page()

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        return {
            "module": self.module_key,
            "title": self.module_title,
            "current_step": _STEPS[self._current],
            "state": self._agent_status(),
            "actions": [
                {"name": "use_example", "args": {},
                 "desc": "Add three bundled example velocity lines."},
                {"name": "add_line", "args": {"name": "str", "mesh": "str", "velocity": "str",
                                              "x0": "float", "y0": "float", "x1": "float", "y1": "float"},
                 "desc": "Add a 2D velocity line by its mesh + velocity files and map endpoints."},
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Keys: velocity_threshold, horizontal_sampling, "
                          "grid_resolution, depth, vertical_layers, interpolation (griddata/kriging), "
                          "z_scale, max_points, kriging_model, range_x, range_y, range_z, variance, nugget.")},
                {"name": "preview_interface", "args": {},
                 "desc": "Show the bedrock interface of the chosen line (the first by default)."},
                {"name": "send_structure_to_hydro", "args": {},
                 "desc": "Send the chosen line's bedrock interface to the ERT -> Water Content module."},
                {"name": "goto_step", "args": {"step": list(_STEPS)},
                 "desc": "Show the view and panel of a step, by name or index."},
                {"name": "run", "args": {},
                 "desc": "Build the 3D velocity model from the added lines."},
                {"name": "get_status", "args": {},
                 "desc": "Report the added lines and last result."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "use_example": lambda: self._agent_use_example(),
            "add_line": lambda: self._agent_add_line(args),
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "preview_interface": lambda: self._agent_preview_interface(),
            "send_structure_to_hydro": lambda: self._agent_send_structure(),
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
        """Add the velocity section the run has just recovered, as a line.

        Part of the studio's contract: a module the run brings to the front has
        to show the user a result. The seismic step writes its mesh and velocity
        model into the run folder, which is exactly what a line here is made of.
        """
        if self._agent_status().get("lines"):
            return ""
        folder = inputs.get("output_dir")
        if not folder:
            return ""
        seismic = Path(str(folder)) / "seismic"
        mesh, velocity = seismic / "seismic_mesh.bms", seismic / "velocity_model.npy"
        if not (mesh.is_file() and velocity.is_file()):
            return ""
        result = self._agent_add_line({"mesh": str(mesh), "velocity": str(velocity)})
        if result.get("status") == "failed":
            return ""
        return f"the velocity section from this run ({mesh.parent.name})"

    def show_run_stage(self, tool):
        """Follow the run through this page's steps."""
        if str(tool) != "derive_structure":
            return ""
        self._agent_goto_step("Structure")
        return "Structure"

    def _agent_status(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "current_step": _STEPS[self._current],
            "lines": len(self._lines),
            "line_names": [ln.get("name") for ln in self._lines],
            "run_blocked_because": self._run_blocker(),
            "last_result_status": last.get("status"),
            "last_run_problem": self._run_problem,
        }

    def _agent_use_example(self) -> Dict[str, Any]:
        self._use_example()
        self._sync_lines_from_table()
        return {"status": "ok", "lines": len(self._lines)}

    def _agent_add_line(self, args: Dict[str, Any]) -> Dict[str, Any]:
        mesh, velocity = args.get("mesh"), args.get("velocity")
        if not mesh or not velocity:
            return {"status": "failed", "error": "Provide 'mesh' and 'velocity' file paths."}
        for label, path in (("mesh", mesh), ("velocity", velocity)):
            if not Path(str(path)).exists():
                return {"status": "failed", "error": f"{label} file not found: {path}"}
        try:
            self._add_row({
                "name": args.get("name") or f"L{self._table.rowCount() + 1}",
                "mesh": str(mesh), "velocity": str(velocity),
                "x0": float(args.get("x0", 0.0)), "y0": float(args.get("y0", 0.0)),
                "x1": float(args.get("x1", 100.0)), "y1": float(args.get("y1", 0.0)),
            })
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        self._sync_lines_from_table()
        return {"status": "ok", "lines": len(self._lines)}

    def _agent_preview_interface(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one line first."}
        self._go_to(1)
        self._show_section()
        return {"status": "ok"}

    def _agent_send_structure(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one line first."}
        self._send_structure_to_hydrology()
        return {"status": "ok", "message": "Bedrock structure sent to ERT -> Water Content."}

    def _agent_goto_step(self, step: Any) -> Dict[str, Any]:
        names = {name.lower(): i for i, name in enumerate(_STEPS)}
        idx = step if isinstance(step, int) else names.get(str(step).lower())
        if idx is None or not (0 <= idx < len(_STEPS)):
            return {"status": "failed", "error": f"Unknown step '{step}'.", "valid": list(_STEPS)}
        self._go_to(idx)
        return {"status": "ok", "step": _STEPS[self._current]}

    def _agent_run(self) -> Dict[str, Any]:
        self._sync_lines_from_table()
        if not self._lines:
            return {"status": "failed", "error": "Add at least one seismic line first."}
        self._run_build()
        return {"status": "started", "message": "3D model build started. Ask for status shortly.",
                "lines": len(self._lines)}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            index = combo.findData(str(value))
            if index < 0:
                index = combo.findText(str(value))
            if index < 0:
                items = [combo.itemData(i) or combo.itemText(i) for i in range(combo.count())]
                raise ValueError(f"must be one of {items}")
            combo.setCurrentIndex(index)

        handlers = {
            "velocity_threshold": lambda v: self._threshold.setValue(float(v)),
            "horizontal_sampling": lambda v: self._interval.setValue(float(v)),
            "grid_resolution": lambda v: self._grid_res.setValue(int(v)),
            "depth": lambda v: self._depth.setValue(float(v)),
            "vertical_layers": lambda v: self._n_layers.setValue(int(v)),
            "interpolation": lambda v: set_combo(self._interp, v),
            "z_scale": lambda v: self._z_scale.setValue(float(v)),
            "max_points": lambda v: self._max_points.setValue(int(v)),
            "kriging_model": lambda v: set_combo(self._krig_model, v),
            "range_x": lambda v: self._len_x.setValue(float(v)),
            "range_y": lambda v: self._len_y.setValue(float(v)),
            "range_z": lambda v: self._len_z.setValue(float(v)),
            "variance": lambda v: self._krig_var.setValue(float(v)),
            "nugget": lambda v: self._krig_nugget.setValue(float(v)),
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
        return {"status": "ok" if applied else "failed", "applied": applied, "ignored": ignored}
