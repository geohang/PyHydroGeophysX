"""Mesh 3D module: an interactive 3D mesh builder.

Build sensor arrays (surface grid, single borehole, crosshole, surface-to-
borehole) for electrodes, geophones or other sensors, choose topography
(including loading x, y, z points from a file), generate a PyGIMLi 3D mesh
(surface-topography prism, Gmsh-free structured grid, Gmsh tetrahedra, or an
E4D-style mesh built the way E4D builds one, from a layout or an E4D ``.cfg``;
see ``core.e4d_mesh``), and view the sensors +
the generated mesh in an interactive PyVistaQt 3D viewer. Zones - boxes of known
or assumed resistivity (``widgets.zone_boxes``) - can shape the mesh on every
engine and give the 3D forward model its structure. Mesh generation runs on
a worker thread; the mesh can be exported to BMS / VTK / sensor CSV. The compute lives in
``qt_apps.mesh3d_builder`` (Qt-free); this module is the UI.

The page is laid out the way the ERT page is: the view in the middle and the
controls in a column on the right, one numbered panel per step in the order the
steps are taken - the sensor array, the mesh engine, the domain and its
topography, the mesh refinement, the zones, and building and saving the mesh.

If PyVistaQt is unavailable the builder still generates and exports meshes; only
the interactive 3D preview is replaced by a short note.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core import mesh_3d as mesh3d_builder
from PyHydroGeophysX.inversion.ert_zones import zone_colors
from PyHydroGeophysX.qt_apps import io_utils, theme
from PyHydroGeophysX.qt_apps.modules.base import BaseModule, LogFn
from PyHydroGeophysX.qt_apps.qt_utils import (
    BusyStateController,
    ContentWidthScrollArea,
    ReproduceBar,
    make_double_spinbox,
    merged_row,
    select_directory,
    set_rows_enabled,
    set_rows_visible,
)
from PyHydroGeophysX.qt_apps.widgets import colormaps as cmaps
from PyHydroGeophysX.qt_apps.widgets.zone_boxes import ZoneBoxEditor, clip_box
from PyHydroGeophysX.qt_apps.workers import ProcessWorkflowWorker, TaskWorker
from PyHydroGeophysX.workflows import (
    ArtifactRef,
    WorkflowRunResult,
    WorkflowSpec,
    export_workflow_bundle,
)

_INSTALL_MESSAGE = (
    "Interactive 3D preview needs <code>pyvista</code>, <code>pyvistaqt</code> and a "
    "working <code>vtk</code> build:<br>"
    "<code>pip install \"pyhydrogeophysx[desktop-3d]\"</code><br><br>"
    "Install with whichever tool already manages the packages here, which is not "
    "always the one that created the environment. Run <code>conda list numpy</code>: "
    "if it reports <code>pypi</code>, use the pip command above; if it names a conda "
    "channel, use <code>conda install -c conda-forge pyvista pyvistaqt</code> instead. "
    "Pulling VTK in through the other one leaves two builds in the same "
    "environment.<br><br>"
    "Mesh generation and export still work without it."
)
_MESH_FILTER = "Mesh (*.vtk *.vtu *.vtp *.stl *.ply *.bms);;All files (*)"
_ARRAY_TYPES = ["Surface grid", "Single borehole", "Crosshole", "Surface-to-borehole"]
_MESH_TYPES = ["Surface with topography", "Box mesh"]
_TOPO_TYPES = ["Flat", "Linear tilt", "Gaussian hill", "Custom expression", "From file (x, y, z)"]
_MESH_ENGINES = ["Auto", "Gmsh (tetrahedral)", "PyGIMLi prism", "Structured grid",
                 mesh3d_builder.E4D_ENGINE]
_E4D_CONFIG_FILTER = "E4D mesh configuration (*.cfg);;All files (*)"
_E4D_MESH_FILTER = "E4D / TetGen mesh (*.node *.ele);;All files (*)"
_ERT_FORWARD_SCHEMES = ("dd", "wa", "slm", "wb")
#: A note under a row, and one that warns.
_HINT_STYLE = "color:#5a6a7a; font-size:8pt;"
_WARN_STYLE = "color:#b42318; font-size:8pt;"
#: What the E4D engine lays its mesh out from while no .cfg is loaded.
_E4D_FROM_STEPS = "No configuration loaded: the layout is built from steps 1, 3 and 4."


# The offscreen guard, shared with the model viewer; it applies the VTK shim
# itself. Under the offscreen Qt platform (headless / CI / --self-test)
# constructing the live QtInteractor aborts the process at the VTK level, so the
# viewer is disabled there. Bound to the name this module always used, which
# tests patch.
from PyHydroGeophysX.visualization.pyvista_compat import (  # noqa: E402
    try_import_pyvista as _try_import_pyvista,
)


class Mesh3DModule(BaseModule):
    """The 3D mesh builder: the 3D view in the middle, the steps on the right.

    Each step is a panel of its own, and the panels always stand, so their
    numbers never shift; what changes is which rows a panel shows, which is
    what the mesher that will run - ``_engine_kind`` - reads.
    """

    module_key = "mesh3d"
    module_title = "3D Mesh Builder"

    def __init__(self, state: Any, log: LogFn, parent=None) -> None:
        super().__init__(state, log, parent)
        self._pv = None
        self._plotter = None
        self._sensors = None         # last previewed/generated sensor DataFrame
        self._mesh = None            # last generated pygimli mesh
        self._topo_points = None     # loaded (x, y, z) topography points
        self._gen_worker: Optional[ProcessWorkflowWorker] = None
        self._gen_busy: Optional[BusyStateController] = None
        self._ert_fwd_worker: Optional[ProcessWorkflowWorker] = None
        self._ert_fwd_result = None  # last agent-triggered 3D ERT forward result
        self._ert_forward_params = {"scheme": "dd", "background_res": 100.0, "noise": 0.03}
        self._mesh_zones: List[Dict[str, Any]] = []  # the zones the mesh shown was built with
        self._mesh_zone_key: Optional[tuple] = None  # what of them shaped it (see _zone_key)
        self._zone_actors: list = []                 # the zone boxes drawn over the view
        self._zone_view: Optional[Tuple[Dict[str, float], bool]] = None  # (region, filled)

        ok, pv, qt_interactor, err = _try_import_pyvista()
        self._pv = pv if ok else None

        # The view in the middle and the steps in a column on the right, the
        # way the ERT page and the other processing pages lay themselves out.
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        controls = self._build_controls()

        center = QVBoxLayout()
        center.setContentsMargins(0, 0, 0, 0)
        center.addLayout(self._build_toolbar())
        center.addLayout(self._build_colour_row())
        if ok:
            try:
                # Rendered when something changes, not on a timer: pyvistaqt
                # re-renders five times a second by default, on the UI thread
                # and whether the page is on screen or not.
                self._plotter = qt_interactor(self, auto_update=False)
                center.addWidget(self._plotter.interactor, stretch=1)
                self._plotter.set_background("white")
                self._plotter.add_axes()
                self.log("PyVistaQt 3D viewer ready.", "success")
            except Exception as exc:  # noqa: BLE001
                self._plotter = None
                self.log(f"3D viewer unavailable: {exc}", "warn")
                center.addWidget(self._viewer_placeholder(str(exc)), stretch=1)
        else:
            self.log(f"3D viewer unavailable: {err}", "warn")
            center.addWidget(self._viewer_placeholder(err), stretch=1)
        self._reproduce = ReproduceBar()
        center.addWidget(self._reproduce)
        view = QWidget()
        view.setLayout(center)
        root.addWidget(view, stretch=1)
        root.addWidget(controls)

        self._update_visibility()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Render once the page is on screen, which the timer used to see to."""
        super().showEvent(event)
        if self._plotter is not None:
            QTimer.singleShot(0, self._refresh_plotter)

    def _viewer_placeholder(self, err: str) -> QLabel:
        msg = QLabel(f"3D preview unavailable:<br><code>{err}</code><br><br>{_INSTALL_MESSAGE}")
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
        return msg

    # -- small widget helpers ------------------------------------------------
    @staticmethod
    def _dspin(lo, hi, val, step=1.0, dec=2, suffix="") -> QDoubleSpinBox:
        return make_double_spinbox(val, lo, hi, step, dec, suffix=suffix)

    @staticmethod
    def _ispin(lo, hi, val) -> QSpinBox:
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # -- toolbar (load/view tools) -------------------------------------------
    def _build_toolbar(self) -> QHBoxLayout:
        """What the view does: open a mesh built elsewhere, clip it, frame it.

        Opening a mesh only to look at it is no step of building one, so both
        ways of doing it sit here, over the view, rather than among the steps.
        """
        bar = QHBoxLayout()
        for label, slot, icon, tip in (
            ("Load mesh…", self._load_mesh, "fa5s.folder-open", ""),
            ("Open E4D mesh…", self._open_e4d_mesh_dialog, "fa5s.cube",
             "View a mesh E4D (or TetGen) built: the .1.node / .1.ele files, with the "
             ".trn beside them putting it back in survey coordinates."),
            ("Clip", self._toggle_clip, "fa5s.cut", ""),
            ("Axes", self._toggle_axes, "fa5s.location-arrow", ""),
            ("Reset view", self._reset_view, "fa5s.expand", ""),
        ):
            btn = QPushButton(label)
            btn.setIcon(theme.icon(icon))
            if tip:
                btn.setToolTip(tip)
            btn.clicked.connect(slot)
            bar.addWidget(btn)
        bar.addStretch(1)
        return bar

    def _build_colour_row(self) -> QHBoxLayout:
        """The colour map of whatever the viewer is colouring.

        Region markers, or the scalar of a volume loaded into it. Each opens on
        the map it always had, and a change swaps the lookup table in place, so
        the camera and a clip plane stay put. Off while nothing on screen is
        colour-mapped. A row of its own under the view tools, which are already
        the widest row on the page.
        """
        self._colormap = cmaps.ColormapChooser(
            cmaps.MESH_REGIONS, "coolwarm", shared=cmaps.colormap_settings(self.state))
        self._colormap.colormapChanged.connect(self._on_colormap_changed)
        self._colormap.setEnabled(False)
        self._scalar_actors: list = []
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(QLabel("Colour map"))
        row.addWidget(self._colormap)
        return row

    def _colour_actor(self, actor, key: str, default: str, scalars) -> None:
        """Hand ``actor`` to the chooser when it is drawn through a colour map."""
        self._scalar_actors = [actor] if (actor is not None and scalars) else []
        self._colormap.set_target(key, default)
        self._colormap.setEnabled(bool(self._scalar_actors))

    def _on_colormap_changed(self, name: str) -> None:
        """Recolour the actor on screen; nothing is re-read or re-added."""
        changed = False
        for actor in self._scalar_actors:
            try:
                actor.mapper.lookup_table.cmap = cmaps.to_pyvista(name)
                changed = True
            except Exception:  # noqa: BLE001 - an actor without a table is left alone
                pass
        if changed:
            self._refresh_plotter()

    # -- controls: one panel per step ----------------------------------------
    def _build_controls(self) -> ContentWidthScrollArea:
        """The steps, one panel each, top to bottom in the order they are taken.

        As on the ERT page: a column on the right that scrolls vertically and is
        never narrower than its widest row, which changes as rows come and go
        with the settings.
        """
        scroll = ContentWidthScrollArea(minimum=450, maximum=500)
        panel = QWidget()
        scroll.setWidget(panel)
        layout = QVBoxLayout(panel)
        layout.addWidget(self._build_array_group())
        layout.addWidget(self._build_engine_group())
        layout.addWidget(self._build_domain_group())
        layout.addWidget(self._build_refinement_group())
        layout.addWidget(self._build_zone_group())
        layout.addWidget(self._build_build_group())
        layout.addStretch(1)
        scroll.fit_to_content()
        return scroll

    @staticmethod
    def _note(text: str = "", style: str = _HINT_STYLE) -> QLabel:
        """A wrapped note under a row."""
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(style)
        return label

    @staticmethod
    def _add_rows(form: QFormLayout, rows: List[Tuple[Optional[str], QWidget]]) -> List[QWidget]:
        """Add ``(label, field)`` rows - a None label spans the row - and return
        the fields, which is what shows or hides a row (``set_rows_visible``)."""
        for label, field in rows:
            if label is None:
                form.addRow(field)
            else:
                form.addRow(label, field)
        return [field for _, field in rows]

    def _build_array_group(self) -> QGroupBox:
        """Step 1: where the sensors are, one set of rows per kind of array.

        One form holds every array's rows, so that their labels line up, and the
        rows of the arrays not chosen are hidden.
        """
        box = QGroupBox("1. Sensor array")
        self._array_group = box
        form = QFormLayout(box)
        self._array_type = QComboBox(); self._array_type.addItems(_ARRAY_TYPES)
        self._array_type.setToolTip(
            "Where the electrodes, geophones or other sensors are: a grid on the surface, "
            "one borehole, several boreholes, or a line on the surface beside a borehole.")
        self._array_type.currentTextChanged.connect(self._update_visibility)
        form.addRow("Array", self._array_type)

        # Surface grid
        self._nx = self._ispin(2, 200, 10); self._ny = self._ispin(2, 200, 6)
        self._dx = self._dspin(0.1, 1000.0, 5.0, 0.5, 2, " m")
        self._dy = self._dspin(0.1, 1000.0, 5.0, 0.5, 2, " m")
        self._x_offset = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._y_offset = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        grid = [("Sensors", merged_row(self._nx, "×", self._ny)),
                ("Spacing", merged_row(self._dx, "×", self._dy)),
                ("First sensor at", merged_row("x", self._x_offset, "y", self._y_offset))]

        # Single borehole
        self._bh_x = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_y = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_z_start = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._bh_z_end = self._dspin(-1e5, 1e5, -20.0, 1.0, 2, " m")
        self._bh_n = self._ispin(2, 300, 12)
        single = [("Borehole at", merged_row("x", self._bh_x, "y", self._bh_y)),
                  ("Sensors", self._bh_n),
                  ("Sensors from z", merged_row(self._bh_z_start, "to", self._bh_z_end))]

        # Crosshole
        self._cross_n = self._ispin(2, 8, 2)
        self._cross_n.valueChanged.connect(self._sync_borehole_table)
        self._cross_z_start = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._cross_z_end = self._dspin(-1e5, 1e5, -20.0, 1.0, 2, " m")
        self._cross_nelec = self._ispin(2, 300, 12)
        self._bh_table = QTableWidget(0, 2)
        self._bh_table.setHorizontalHeaderLabels(["x (m)", "y (m)"])
        self._bh_table.horizontalHeader().setStretchLastSection(True)
        self._bh_table.setMaximumHeight(160)
        self._sync_borehole_table()
        cross = [("Boreholes", self._cross_n),
                 ("Sensors / borehole", self._cross_nelec),
                 ("Sensors from z", merged_row(self._cross_z_start, "to", self._cross_z_end)),
                 ("Positions", self._bh_table)]

        # Surface-to-borehole
        self._s2b_n = self._ispin(2, 300, 24)
        self._s2b_dx = self._dspin(0.1, 1000.0, 2.0, 0.5, 2, " m")
        self._s2b_x0 = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_sy = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_sz = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_bhx = self._dspin(-1e5, 1e5, 20.0, 1.0, 2, " m")
        self._s2b_bhy = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_bhn = self._ispin(2, 300, 16)
        self._s2b_zs = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._s2b_ze = self._dspin(-1e5, 1e5, -30.0, 1.0, 2, " m")
        line = [("Surface sensors", merged_row(self._s2b_n, "every", self._s2b_dx)),
                ("First at", merged_row("x", self._s2b_x0, "y", self._s2b_sy)),
                ("Surface z", self._s2b_sz),
                ("Borehole at", merged_row("x", self._s2b_bhx, "y", self._s2b_bhy)),
                ("Borehole sensors", self._s2b_bhn),
                ("Sensors from z", merged_row(self._s2b_zs, "to", self._s2b_ze))]

        for spin in (self._bh_z_start, self._cross_z_start, self._s2b_zs):
            spin.setToolTip("Elevation of the top sensor; the rest are spaced evenly down "
                            "to the bottom one.")
        self._array_rows: Dict[str, List[QWidget]] = {
            name: self._add_rows(form, rows)
            for name, rows in zip(_ARRAY_TYPES, (grid, single, cross, line))}
        # Said here when a loaded E4D configuration sets the electrodes instead.
        self._array_note = self._note(style=_WARN_STYLE)
        form.addRow(self._array_note)
        return box

    def _build_engine_group(self) -> QGroupBox:
        """Step 2: which mesher builds the mesh, and the settings of its own.

        Straight after the sensors, because the steps after it depend on it: the
        E4D engine lays out its domain and sizes its elements in terms of its
        own, and each of the others reads a different part of the settings.
        """
        defaults = mesh3d_builder.E4D_DEFAULTS
        box = QGroupBox("2. Mesh engine")
        form = QFormLayout(box)
        self._mesh_engine = QComboBox(); self._mesh_engine.addItems(_MESH_ENGINES)
        self._mesh_engine.setToolTip(
            "Auto: prism for surface+topography, structured otherwise.\n"
            "Gmsh (tetrahedral): high-quality refined tet mesh (flat top; best for box / borehole / mild terrain).\n"
            "PyGIMLi prism: topography-conforming.\n"
            "Structured grid: fast regular grid.\n"
            "E4D (Triangle + TetGen): built the way E4D builds its meshes - a fine zone "
            "around the electrodes inside a coarse outer zone reaching far out - and "
            "written as an E4D .cfg that E4D itself runs on.")
        self._mesh_engine.currentTextChanged.connect(self._update_visibility)
        form.addRow("Engine", self._mesh_engine)
        # What will build the mesh, which is not always the engine picked.
        self._engine_note = self._note()
        form.addRow(self._engine_note)

        # An E4D configuration written elsewhere replaces steps 1, 3 and 4.
        self._e4d_config_path = ""
        load_btn = QPushButton("Load E4D .cfg…")
        load_btn.setIcon(theme.icon("fa5s.file-import"))
        load_btn.setToolTip("Build from an existing E4D mesh configuration file instead "
                            "of the sensor array, domain and refinement set here.")
        load_btn.clicked.connect(self._load_e4d_config_dialog)
        self._e4d_clear = QPushButton("✕")
        self._e4d_clear.setMaximumWidth(32)
        self._e4d_clear.setToolTip("Go back to building from steps 1, 3 and 4.")
        self._e4d_clear.setEnabled(False)
        self._e4d_clear.clicked.connect(self._clear_e4d_config)
        source = QWidget()
        source_row = QHBoxLayout(source)
        source_row.setContentsMargins(0, 0, 0, 0)
        source_row.addWidget(load_btn, 1)
        source_row.addWidget(self._e4d_clear)
        self._e4d_cfg_label = QLabel(_E4D_FROM_STEPS)
        self._e4d_cfg_label.setWordWrap(True)

        self._e4d_mesher = QComboBox()
        for label, value in (("Auto (TetGen, else Gmsh)", "auto"), ("TetGen", "tetgen"),
                             ("Gmsh", "gmsh")):
            self._e4d_mesher.addItem(label, value)
        self._e4d_mesher.setToolTip(
            "TetGen is what E4D runs: the program if it is on PATH, otherwise its "
            "Python package (pip install tetgen). Gmsh stands in when TetGen is not "
            "installed; it aims at the zone volumes rather than enforcing them, so the "
            "log reports each zone's largest element against its limit.")
        self._e4d_tetgen = QLineEdit()
        self._e4d_tetgen.setPlaceholderText("TetGen program (optional)")
        self._e4d_tetgen.setToolTip("Path to a TetGen executable, when it is neither on "
                                    "PATH nor installed as the Python package.")
        self._e4d_status = self._note(self._e4d_mesher_status())
        self._e4d_sigma = self._dspin(1e-6, 100.0, defaults["e4d_conductivity"], 0.01, 4, " S/m")
        self._e4d_sigma.setToolTip("The zones' starting conductivity, written into the "
                                   "E4D configuration.")
        self._e4d_engine_rows = self._add_rows(form, [
            (None, source), (None, self._e4d_cfg_label),
            ("Mesher", self._e4d_mesher), ("TetGen path", self._e4d_tetgen),
            (None, self._e4d_status), ("Starting conductivity", self._e4d_sigma)])

        # Every engine honours it.
        self._single_region = QCheckBox("Single region (one marker)")
        self._single_region.setToolTip(
            "Collapse the mesh to one marker instead of preserving the parameter-domain (2) / "
            "boundary (1) split.")
        form.addRow(self._single_region)
        return box

    def _build_domain_group(self) -> QGroupBox:
        """Step 3: the volume the mesh fills, and the ground on top of it."""
        defaults = mesh3d_builder.E4D_DEFAULTS
        box = QGroupBox("3. Domain and topography")
        self._domain_group = box
        form = QFormLayout(box)
        self._mesh_type = QComboBox(); self._mesh_type.addItems(_MESH_TYPES)
        self._mesh_type.setToolTip(
            "Surface with topography builds the domain around the sensors, under the ground "
            "set below. Box mesh builds a box of the size set below, from the origin.")
        self._mesh_type.currentTextChanged.connect(self._update_visibility)
        form.addRow("Domain", self._mesh_type)
        self._domain_note = self._note()
        form.addRow(self._domain_note)

        # Every engine but E4D: how deep the inverted region reaches, and how
        # far the domain reaches around the sensors.
        self._para_depth = self._dspin(1.0, 500.0, 20.0, 1.0, 2, " m")
        self._para_depth.setToolTip(
            "Depth of the inverted region (marker 2) below the top of the mesh; the cells "
            "below it are the outer region (marker 1), and zones act only above it. Under "
            "a surface grid it is also how deep the mesh reaches.")
        self._bound_ext = self._dspin(1.0, 3.0, 1.4, 0.1, 2)
        self._bound_ext.setToolTip(
            "How far the domain reaches beyond a surface grid, as a multiple of the grid's "
            "size: 1.4 adds a fifth of its length on each side.")
        self._box_length = self._dspin(1.0, 5000.0, 50.0, 1.0, 2, " m")
        self._box_width = self._dspin(1.0, 5000.0, 30.0, 1.0, 2, " m")
        self._box_height = self._dspin(1.0, 1000.0, 25.0, 1.0, 2, " m")
        self._bh_lat_pad = self._dspin(1.0, 500.0, 10.0, 1.0, 2, " m")
        self._bh_lat_pad.setToolTip("How far the domain reaches beyond the outermost "
                                    "boreholes, in x and y.")
        self._bh_top_pad = self._dspin(0.0, 100.0, 2.0, 0.5, 2, " m")
        self._bh_top_pad.setToolTip("How far the domain reaches above the highest sensor.")
        self._bh_bot_pad = self._dspin(0.0, 500.0, 5.0, 1.0, 2, " m")
        self._bh_bot_pad.setToolTip(
            "How far the domain reaches below the deepest sensor - for Gmsh, below the "
            "investigation depth under it.")
        self._box_rows = [self._box_length, self._box_width, self._box_height]
        self._add_rows(form, [
            ("Investigation depth", self._para_depth), ("Boundary extension", self._bound_ext),
            ("Length (x)", self._box_length), ("Width (y)", self._box_width),
            ("Depth (z)", self._box_height),
            ("Lateral padding", self._bh_lat_pad), ("Top padding", self._bh_top_pad),
            ("Bottom padding", self._bh_bot_pad)])

        # The E4D engine's domain: a fine zone around the electrodes inside an
        # outer zone. The defaults are those of a crosshole configuration built
        # for E4D (Van Nuys): a zone a metre beyond the electrodes, the outer
        # boundary 100 m beyond it and the bottom 150 m down.
        self._e4d_pad = self._dspin(0.0, 1e4, defaults["e4d_fine_padding"], 0.5, 2, " m")
        self._e4d_pad.setToolTip("How far the fine zone reaches beyond the outermost "
                                 "electrodes, in x and y.")
        self._e4d_depth_pad = self._dspin(0.0, 1e4, defaults["e4d_fine_depth_padding"], 0.5, 2, " m")
        self._e4d_depth_pad.setToolTip("How far the fine zone reaches below the deepest "
                                       "electrode.")
        self._e4d_outer = self._dspin(1.0, 1e6, defaults["e4d_outer_distance"], 10.0, 1, " m")
        self._e4d_outer.setToolTip(
            "How far beyond the fine zone the outer boundary sits. The potentials "
            "must have died away there; ten times the survey size is the usual rule.")
        self._e4d_bottom = self._dspin(1.0, 1e6, defaults["e4d_bottom_depth"], 10.0, 1, " m")
        self._e4d_bottom.setToolTip("Depth of the mesh bottom below the lowest point of "
                                    "the surface (E4D's m_bot).")
        self._e4d_domain_rows = self._add_rows(form, [
            ("Fine zone padding", self._e4d_pad),
            ("Below deepest electrode", self._e4d_depth_pad),
            ("Outer boundary distance", self._e4d_outer),
            ("Mesh bottom depth", self._e4d_bottom)])

        # The ground, one set of rows per kind of surface.
        self._topo_type = QComboBox(); self._topo_type.addItems(_TOPO_TYPES)
        self._topo_type.setToolTip("The ground surface: what the mesh is built under, and "
                                   "what the sensors of a surface grid are placed on.")
        self._topo_type.currentTextChanged.connect(self._update_visibility)
        form.addRow("Topography", self._topo_type)
        self._z_flat = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        flat = [("Surface elevation", self._z_flat)]
        self._z_base = self._dspin(-1e5, 1e5, 100.0, 1.0, 2, " m")
        self._tilt_x = self._dspin(-1.0, 1.0, 0.05, 0.01, 3)
        self._tilt_y = self._dspin(-1.0, 1.0, 0.0, 0.01, 3)
        tilt = [("Base elevation", self._z_base),
                ("Slope", merged_row("x", self._tilt_x, "y", self._tilt_y))]
        self._hill_base = self._dspin(-1e5, 1e5, 0.0, 1.0, 2, " m")
        self._hill_amp = self._dspin(-1e4, 1e4, 5.0, 0.5, 2, " m")
        self._hill_sigma = self._dspin(0.1, 1e4, 10.0, 1.0, 2, " m")
        self._hill_cx = self._dspin(-1e5, 1e5, 25.0, 1.0, 2, " m")
        self._hill_cy = self._dspin(-1e5, 1e5, 15.0, 1.0, 2, " m")
        hill = [("Base z", self._hill_base), ("Amplitude", self._hill_amp),
                ("Width sigma", self._hill_sigma),
                ("Centre at", merged_row("x", self._hill_cx, "y", self._hill_cy))]
        self._topo_expr = QLineEdit("0.1*x - 0.05*y + 100")
        self._topo_expr.setToolTip("z = f(x, y). Allowed: x, y, np, sin, cos, exp, sqrt, abs, pi.")
        custom = [("z = f(x, y)", self._topo_expr)]
        # From file (x, y, z): the surface is interpolated from the points.
        load_btn = QPushButton("Load topography file (x, y, z)…")
        load_btn.setIcon(theme.icon("fa5s.file-upload"))
        load_btn.clicked.connect(self._load_topo_file)
        self._topo_file_label = QLabel("No file loaded. The surface is interpolated from the points.")
        self._topo_file_label.setWordWrap(True)
        points = [(None, load_btn), (None, self._topo_file_label)]
        self._topo_rows: Dict[str, List[QWidget]] = {
            name: self._add_rows(form, rows)
            for name, rows in zip(_TOPO_TYPES, (flat, tilt, hill, custom, points))}
        return box

    def _load_topo_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load topography points (x, y, z)", "",
            "Points (*.csv *.txt *.dat *.xyz);;All files (*)")
        if not path:
            return
        try:
            table = io_utils.load_xyz_table(path, min_cols=3)
            self._topo_points = table[:, :3]
            self._topo_file_label.setText(
                f"{Path(path).name}: {len(self._topo_points)} points "
                f"(z {self._topo_points[:, 2].min():.1f} to {self._topo_points[:, 2].max():.1f} m)")
            self.log(f"Loaded topography: {len(self._topo_points)} points from {Path(path).name}", "success")
        except Exception as exc:  # noqa: BLE001
            self._topo_points = None
            self._topo_file_label.setText(f"Load failed: {exc}")
            self.log(f"Could not load topography file: {exc}", "error")

    def _build_refinement_group(self) -> QGroupBox:
        """Step 4: how fine the elements are, in the terms of the mesher that runs."""
        defaults = mesh3d_builder.E4D_DEFAULTS
        box = QGroupBox("4. Mesh refinement")
        self._refine_group = box
        form = QFormLayout(box)
        self._elec_refine = self._dspin(0.01, 50.0, 0.5, 0.1, 3, " m")
        self._elec_refine.setToolTip("Gmsh: the element size at the sensors.")
        self._attractor = self._dspin(0.1, 200.0, 5.0, 0.5, 2, " m")
        self._attractor.setToolTip("Gmsh: the distance from the sensors over which the "
                                   "elements grow from the sensor size to the boundary size.")
        self._bound_refine = self._dspin(0.1, 100.0, 2.0, 0.5, 2, " m")
        self._bound_refine.setToolTip("Gmsh: the element size away from the sensors. The "
                                      "structured grid under a surface grid: its horizontal "
                                      "cell size.")
        self._dz_fine = self._dspin(0.05, 10.0, 0.5, 0.1, 2, " m")
        self._dz_fine.setToolTip("The prism mesh: the thickness of its fine layers. The "
                                 "structured grid under a surface grid: its vertical cell size.")
        self._dz_coarse = self._dspin(0.5, 50.0, 2.0, 0.5, 2, " m")
        self._dz_coarse.setToolTip("The prism mesh: the thickness of its coarse layers.")
        self._bh_hcell = self._dspin(0.1, 100.0, 2.0, 0.5, 2, " m")
        self._bh_hcell.setToolTip("The structured grid around boreholes: its horizontal "
                                  "cell size.")
        self._bh_vcell = self._dspin(0.1, 100.0, 1.0, 0.5, 2, " m")
        self._bh_vcell.setToolTip("The structured grid around boreholes: its vertical "
                                  "cell size.")
        self._add_rows(form, [
            ("Sensor refinement", self._elec_refine), ("Attractor distance", self._attractor),
            ("Boundary refinement", self._bound_refine), ("Fine layer dz", self._dz_fine),
            ("Coarse layer dz", self._dz_coarse), ("Horizontal cell", self._bh_hcell),
            ("Vertical cell", self._bh_vcell)])

        # E4D sizes its elements by the largest volume a zone allows and by shape.
        self._e4d_fine_vol = self._dspin(1e-4, 1e6, defaults["e4d_fine_volume"], 0.1, 4, " m³")
        self._e4d_fine_vol.setToolTip(
            "The largest element allowed in the fine zone - E4D's zone volume. "
            "Halving it roughly doubles the elements there.")
        self._e4d_quality = self._dspin(1.05, 3.0, defaults["e4d_quality"], 0.02, 2)
        self._e4d_quality.setToolTip(
            "Largest radius-edge ratio of any element (TetGen -q). Lower is better "
            "shaped and costs more elements; the E4D guide recommends 1.3 to 1.5.")
        self._e4d_refine = self._dspin(0.0, 10.0, defaults["e4d_refine_offset"], 0.01, 3, " m")
        self._e4d_refine.setToolTip(
            "Offset of the refinement point placed beside each buried electrode (and "
            "under each borehole top), which forces small elements where the "
            "potential changes fastest. 0 adds none.")
        self._e4d_refine_rows = self._add_rows(form, [
            ("Fine zone element volume", self._e4d_fine_vol),
            ("Quality (radius-edge)", self._e4d_quality),
            ("Electrode refinement", self._e4d_refine)])
        return box

    @staticmethod
    def _e4d_mesher_status() -> str:
        """Which mesher the E4D engine will use here, said before it is used."""
        from PyHydroGeophysX.core import e4d_mesh

        program = e4d_mesh.find_tetgen()
        if program:
            return f"TetGen program: {program}"
        package = e4d_mesh._tetgen_module()
        if package is not None:
            return f"TetGen: Python package {getattr(package, '__version__', '')}"
        if e4d_mesh._find_gmsh():
            return ("TetGen is not installed, so Gmsh stands in "
                    "(pip install tetgen for E4D's own mesher).")
        return "Neither TetGen nor Gmsh is installed: pip install tetgen."

    def _build_zone_group(self) -> QGroupBox:
        """Step 5: boxes of known or assumed resistivity, and what they do to the mesh."""
        self._zone_editor = ZoneBoxEditor()
        self._zone_editor.setTitle("5. Zones (optional)")
        self._zone_editor.set_region_provider(self._zone_region)
        self._zone_editor.zonesChanged.connect(lambda _zones: self._on_zones_changed())
        self._zone_editor.optionsChanged.connect(self._on_zones_changed)
        # The note names the E4D fine zone the zones must lie in.
        for spin in (self._e4d_pad, self._e4d_depth_pad):
            spin.valueChanged.connect(lambda _value: self._update_zone_note())
        return self._zone_editor

    def _build_build_group(self) -> QGroupBox:
        """Step 6: look at the layout, build the mesh, and what the build saves.

        What to save comes before the button because the build writes it: there
        is no export afterwards to choose it at.
        """
        box = QGroupBox("6. Build and save")
        form = QFormLayout(box)
        self._mesh_name = QLineEdit("my_3d_mesh")
        self._mesh_name.setToolTip("The name of the saved files, before the extension.")
        form.addRow("Mesh name", self._mesh_name)
        self._fmt_bms = QCheckBox("BMS"); self._fmt_bms.setChecked(True)
        self._fmt_vtk = QCheckBox("VTK"); self._fmt_vtk.setChecked(True)
        self._fmt_csv = QCheckBox("Sensor CSV"); self._fmt_csv.setChecked(True)
        form.addRow("Save as", merged_row(self._fmt_bms, self._fmt_vtk, self._fmt_csv))
        self._out_dir = QLineEdit("One outputs/ folder per run")
        self._out_dir.setReadOnly(True)
        self._out_dir.setToolTip(
            "Mesh files are saved under the active Project's unique run directory."
        )
        open_btn = QPushButton("Open")
        open_btn.setIcon(theme.icon("fa5s.folder"))
        open_btn.setToolTip("Open the folder the last build saved to.")
        open_btn.clicked.connect(self._open_output_folder)
        folder = QWidget()
        folder_row = QHBoxLayout(folder)
        folder_row.setContentsMargins(0, 0, 0, 0)
        folder_row.addWidget(self._out_dir, 1)
        folder_row.addWidget(open_btn)
        form.addRow("Folder", folder)
        form.addRow(self._note(
            "Preview draws the layout without meshing it. Generate builds the mesh in a "
            "process of its own and saves the files ticked above."))

        buttons = QWidget()
        button_row = QHBoxLayout(buttons)
        button_row.setContentsMargins(0, 0, 0, 0)
        self._preview_btn = QPushButton("Preview sensors")
        self._preview_btn.setIcon(theme.icon("fa5s.eye"))
        self._preview_btn.setToolTip(
            "Draw the sensors, the ground and the zones - with the E4D engine its whole "
            "layout - before anything is meshed.")
        self._preview_btn.clicked.connect(self._preview_sensors)
        button_row.addWidget(self._preview_btn)
        self._gen_btn = QPushButton("Generate mesh")
        self._gen_btn.setProperty("primary", True)
        self._gen_btn.setIcon(theme.icon("fa5s.cubes", color="#ffffff"))
        self._gen_btn.clicked.connect(self._generate_mesh)
        button_row.addWidget(self._gen_btn)
        form.addRow(buttons)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        form.addRow(self._progress)
        self._info = QLabel("Work down the steps, preview the layout, then generate the mesh.")
        self._info.setWordWrap(True)
        form.addRow(self._info)
        return box

    # -- visibility / config -------------------------------------------------
    def _update_visibility(self) -> None:
        """Show, in each step, the rows that the mesher that will run reads.

        Which mesher that is depends on the array and the domain as well as on
        the engine picked (``_engine_kind``), and a row shows when that mesher
        reads it, not when some engine could. The steps themselves stay; a
        loaded E4D configuration, which is the electrodes, the domain and the
        element sizes at once, greys out the three steps it stands in for.
        """
        array_type = self._array_type.currentText()
        grid = array_type == "Surface grid"
        box = self._mesh_type.currentText() == "Box mesh"
        kind = self._engine_kind()
        e4d, gmsh = kind == "e4d", kind == "gmsh"
        prism, structured = kind == "prism", kind == "structured"
        loaded = e4d and bool(self._e4d_config_path)

        # 1. the chosen array's rows
        for name, rows in self._array_rows.items():
            set_rows_visible(rows, name == array_type)
        # 2. the E4D engine's own settings
        set_rows_visible(self._e4d_engine_rows, e4d)
        # 3. the domain: E4D's zones, a box, or a domain around the sensors
        set_rows_visible([self._para_depth], not e4d)
        set_rows_visible([self._bound_ext], not e4d and grid and not box)
        set_rows_visible(self._box_rows, not e4d and box)
        set_rows_visible([self._bh_lat_pad], not box and not grid and (structured or gmsh))
        set_rows_visible([self._bh_top_pad], not box and not grid and structured)
        set_rows_visible([self._bh_bot_pad], not box and (gmsh or (structured and not grid)))
        set_rows_visible(self._e4d_domain_rows, e4d)
        # The ground places a surface grid's sensors - and is the top of a prism
        # mesh - and the E4D engine meshes it under every array. E4D meshes no
        # box: it builds on flat ground at the flat surface's elevation.
        ground = not box and (grid or e4d)
        set_rows_visible([self._topo_type], ground)
        topo = self._topo_type.currentText()
        for name, rows in self._topo_rows.items():
            set_rows_visible(rows, ground and name == topo)
        if e4d and box:
            set_rows_visible([self._z_flat], True)
        # 4. how fine, in the terms of the mesher that runs
        set_rows_visible([self._elec_refine, self._attractor], gmsh)
        set_rows_visible([self._bound_refine], gmsh or (structured and grid))
        set_rows_visible([self._dz_fine], grid and (prism or structured))
        set_rows_visible([self._dz_coarse], prism)
        set_rows_visible([self._bh_hcell, self._bh_vcell], structured and not grid)
        set_rows_visible(self._e4d_refine_rows, e4d)

        for group in (self._array_group, self._domain_group, self._refine_group):
            group.setEnabled(not loaded)
        set_rows_enabled([self._e4d_sigma], not loaded)   # the file has its own
        self._array_note.setText(
            f"Not used: {Path(self._e4d_config_path).name} sets the electrodes. Clear it "
            "in step 2 (✕) to build from this array." if loaded else "")
        self._array_note.setVisible(loaded)
        self._engine_note.setText(self._engine_note_text(kind))
        domain = self._domain_note_text(kind)
        self._domain_note.setText(domain)
        self._domain_note.setVisible(bool(domain))
        self._preview_btn.setText("Preview E4D layout" if e4d else "Preview sensors")
        self._update_zone_note()

    def _engine_note_text(self, kind: str) -> str:
        """What will build the mesh, which is not always the engine picked: the
        prism engine builds a structured grid for anything but a surface grid
        with topography, and Gmsh hands over to it when Gmsh fails."""
        engine = self._mesh_engine.currentText()
        if kind == "e4d":
            if self._e4d_config_path:
                return (f"Builds the mesh {Path(self._e4d_config_path).name} describes, the "
                        "way E4D does; the mesher below is still the one used.")
            return ("Built the way E4D builds it: the ground surface triangulated, then TetGen "
                    "fills a fine zone around the electrodes, with a refinement point beside "
                    "each buried one, inside an outer zone reaching far enough for the "
                    "boundary conditions not to matter. The E4D .cfg and .poly are saved with "
                    "the mesh, so E4D can run on the same geometry.")
        if kind == "gmsh":
            return ("Gmsh tetrahedra, graded from the sensors outward, in a box with a flat "
                    "top; the structured grid stands in when Gmsh is missing or fails.")
        if kind == "prism":
            return ("Auto: " if engine == "Auto" else "") + (
                "PyGIMLi prisms, under a top that follows the ground.")
        if engine == "PyGIMLi prism":
            return ("The prism mesh needs a surface grid with topography, so a structured grid "
                    "is built instead.")
        if engine == "Auto":
            return ("Auto: a structured PyGIMLi grid, with a node at every sensor and a flat "
                    "top; the prism mesh is for a surface grid with topography.")
        return "A structured PyGIMLi grid, with a node at every sensor and a flat top."

    def _domain_note_text(self, kind: str) -> str:
        """What the domain is for the current settings, or "" when the rows say it."""
        grid = self._array_type.currentText() == "Surface grid"
        box = self._mesh_type.currentText() == "Box mesh"
        if kind == "e4d":
            if self._e4d_config_path:
                return (f"Not used: {Path(self._e4d_config_path).name} sets the domain and "
                        "the ground.")
            if box:
                return ("E4D meshes no box: it builds on flat ground at the elevation below. "
                        "Pick Surface with topography for any other ground.")
            return ("A fine zone the padding beyond the electrodes, inside an outer zone out "
                    "to the boundary distance and down to the mesh bottom.")
        if box:
            return "A box from the origin, of the size below, with a flat top."
        if not grid:
            return "A block with a flat top around the sensors, padded as below."
        if kind == "prism":
            return "The mesh top follows the ground below."
        if self._topo_type.currentText() != "Flat":
            return ("The sensors are placed on the ground below, but this mesh has a flat top "
                    "at the highest of them; the prism engine follows the ground.")
        return ""

    def _sync_borehole_table(self) -> None:
        n = self._cross_n.value()
        self._bh_table.setRowCount(n)
        for i in range(n):
            for j, default in ((0, i * 10.0), (1, 0.0)):
                if self._bh_table.item(i, j) is None:
                    self._bh_table.setItem(i, j, QTableWidgetItem(f"{default:g}"))

    def _read_boreholes(self):
        out = []
        for i in range(self._bh_table.rowCount()):
            try:
                x = float(self._bh_table.item(i, 0).text())
                y = float(self._bh_table.item(i, 1).text())
            except Exception:  # noqa: BLE001
                x, y = i * 10.0, 0.0
            out.append((x, y))
        return out

    def _default_output_dir(self) -> Path:
        base = self.state.output_dir or Path.cwd()
        return Path(base) / "mesh3d"

    def _browse_output_dir(self) -> None:
        path = select_directory(self, "Output directory", self._out_dir.text())
        if path:
            self._out_dir.setText(str(path))

    def _collect_config(self) -> dict:
        cfg = {
            "mesh_type": self._mesh_type.currentText(),
            "array_type": self._array_type.currentText(),
            "mesh_engine": self._mesh_engine.currentText(),
            "single_region": self._single_region.isChecked(),
            "electrode_refinement": self._elec_refine.value(),
            "attractor_distance": self._attractor.value(),
            "boundary_refinement": self._bound_refine.value(),
            "para_depth": self._para_depth.value(),
            "dz_fine": self._dz_fine.value(),
            "dz_coarse": self._dz_coarse.value(),
            "boundary_extension": self._bound_ext.value(),
            "borehole_lateral_padding": self._bh_lat_pad.value(),
            "borehole_bottom_padding": self._bh_bot_pad.value(),
            "borehole_top_padding": self._bh_top_pad.value(),
            "borehole_horizontal_cell": self._bh_hcell.value(),
            "borehole_vertical_cell": self._bh_vcell.value(),
            "output_dir": str(self._default_output_dir()),
        }
        if cfg["mesh_engine"] == mesh3d_builder.E4D_ENGINE:
            cfg.update(
                e4d_fine_padding=self._e4d_pad.value(),
                e4d_fine_depth_padding=self._e4d_depth_pad.value(),
                e4d_fine_volume=self._e4d_fine_vol.value(),
                e4d_outer_distance=self._e4d_outer.value(),
                e4d_bottom_depth=self._e4d_bottom.value(),
                e4d_quality=self._e4d_quality.value(),
                e4d_refine_offset=self._e4d_refine.value(),
                e4d_conductivity=self._e4d_sigma.value(),
                e4d_mesher=str(self._e4d_mesher.currentData()),
                e4d_tetgen=self._e4d_tetgen.text().strip(),
                e4d_config_path=self._e4d_config_path)
        array = cfg["array_type"]
        if array == "Surface grid":
            cfg.update(nx=self._nx.value(), ny=self._ny.value(), dx=self._dx.value(), dy=self._dy.value(),
                       x_offset=self._x_offset.value(), y_offset=self._y_offset.value())
        elif array == "Single borehole":
            cfg.update(bh_x=self._bh_x.value(), bh_y=self._bh_y.value(),
                       z_start=self._bh_z_start.value(), z_end=self._bh_z_end.value(), n_bh_elec=self._bh_n.value())
        elif array == "Crosshole":
            cfg.update(boreholes=self._read_boreholes(), z_start=self._cross_z_start.value(),
                       z_end=self._cross_z_end.value(), n_bh_elec=self._cross_nelec.value())
        else:
            cfg.update(n_surface_elec=self._s2b_n.value(), surface_dx=self._s2b_dx.value(),
                       surface_x0=self._s2b_x0.value(), surface_y=self._s2b_sy.value(), surface_z=self._s2b_sz.value(),
                       bh_x=self._s2b_bhx.value(), bh_y=self._s2b_bhy.value(), n_bh_elec=self._s2b_bhn.value(),
                       z_start=self._s2b_zs.value(), z_end=self._s2b_ze.value())
        # topography
        topo = self._topo_type.currentText()
        cfg["topography_type"] = topo if cfg["mesh_type"] == "Surface with topography" else "Flat"
        cfg.update(z_flat=self._z_flat.value(), z_base=self._z_base.value(),
                   tilt_x=self._tilt_x.value(), tilt_y=self._tilt_y.value(),
                   hill_base=self._hill_base.value(), hill_amp=self._hill_amp.value(),
                   hill_sigma=self._hill_sigma.value(), hill_cx=self._hill_cx.value(), hill_cy=self._hill_cy.value(),
                   topography_expr=self._topo_expr.text(), topography_points=self._topo_points)
        if cfg["mesh_type"] == "Box mesh":
            cfg.update(box_length=self._box_length.value(), box_width=self._box_width.value(),
                       box_height=self._box_height.value())
        # Zones only when there are any, so that a recipe without them reads as
        # it did before they existed.
        zones = self._zone_editor.zones()
        if zones:
            cfg["zones"] = zones
            if self._zone_editor.conform_to_zones():
                cfg["conform_to_zones"] = True
            if self._zone_editor.decouple_zones():
                cfg["decouple_zones"] = True
        return cfg

    def _selected_formats(self):
        fmts = []
        if self._fmt_bms.isChecked():
            fmts.append("BMS mesh (.bms)")
        if self._fmt_vtk.isChecked():
            fmts.append("VTK mesh (.vtk)")
        if self._fmt_csv.isChecked():
            fmts.append("Sensor CSV")
        return fmts

    # -- zones ---------------------------------------------------------------
    def _zone_region(self, cfg: Optional[dict] = None,
                     sensors: Any = None) -> Optional[Dict[str, float]]:
        """Where zones act in the mesh the current settings build (see
        ``core.mesh_3d.zone_region``), None while the settings make none."""
        try:
            return mesh3d_builder.zone_region(cfg or self._collect_config(), sensors)
        except Exception:  # noqa: BLE001 - settings mid-edit
            return None

    @staticmethod
    def _zone_key(zones: Any, conform: Any, separate: Any) -> tuple:
        """What of the zones shapes a mesh: the two options, and the boxes when
        the mesh follows them or makes each a region. Their values never do."""
        zones = list(zones or [])
        conform, separate = bool(conform) and bool(zones), bool(separate) and bool(zones)
        boxes = tuple(tuple(tuple(zone[axis]) for axis in "xyz") for zone in zones)
        return (boxes if conform or separate else ()), conform, separate

    def _engine_kind(self) -> str:
        """The mesher ``generate_mesh`` runs for the current settings."""
        engine = self._mesh_engine.currentText()
        if engine == mesh3d_builder.E4D_ENGINE:
            return "e4d"
        if engine == "Gmsh (tetrahedral)":
            return "gmsh"
        surface_topo = (self._array_type.currentText() == "Surface grid"
                        and self._mesh_type.currentText() == "Surface with topography")
        return "prism" if surface_topo and engine in ("Auto", "PyGIMLi prism") else "structured"

    def _zone_note(self) -> Tuple[str, bool]:
        """What the selected engine does with the zones, and whether that - or a
        mesh built before they changed - deserves a warning."""
        editor = self._zone_editor
        zones = editor.zones()
        stale = (self._mesh is not None and self._mesh_zone_key is not None
                 and self._mesh_zone_key != self._zone_key(
                     zones, editor.conform_to_zones(), editor.decouple_zones()))
        after = " The zones changed after the mesh was built: generate it again." if stale else ""
        if not zones:
            return after.strip(), stale
        kind = self._engine_kind()
        if kind == "e4d" and self._e4d_config_path:
            return ("The loaded E4D configuration defines its own zones, so these are not "
                    "used; clear it (✕) to build from the sensor array with them.", True)
        if not editor.conform_to_zones():
            text = ("Each zone takes the cells whose centre lies inside it, so its faces are "
                    "only as sharp as the cells there.")
        else:
            text = {
                "e4d": ("E4D meshes each zone as a zone of its own. One reaching the fine "
                        "zone's walls on every side is a layer across it; the others are "
                        "blocks, which may stand apart, share a whole face, or lie inside a "
                        "layer or a block listed before them."),
                "structured": "The grid gets a line on every zone face.",
                "prism": ("The zone outlines join the plan triangulation, and a layer of "
                          "cells ends at every zone top and bottom."),
                "gmsh": "Gmsh cuts the zone boxes into its domain.",
            }[kind]
        if kind == "e4d":
            region = self._zone_region()
            if region is not None:
                text += (f" They act inside the fine zone: x {region['x_min']:g} to "
                         f"{region['x_max']:g}, y {region['y_min']:g} to {region['y_max']:g}, "
                         f"down to {region['z_bottom']:g} m; the fine zone padding and the "
                         "depth below the deepest electrode widen it.")
        elif kind != "gmsh":
            text += (" Only the inverted region is zoned, down to the investigation depth "
                     f"({self._para_depth.value():g} m below the top of the mesh).")
        return text + after, stale

    def _update_zone_note(self) -> None:
        text, warn = self._zone_note()
        self._zone_editor.set_note(text, warn)

    def _on_zones_changed(self) -> None:
        """A zone or a zone option changed: say what it does, redraw the boxes."""
        self._update_zone_note()
        if self._zone_view is not None:
            region, filled = self._zone_view
            self._draw_zone_boxes(region, filled=filled)
            self._refresh_plotter()

    def _draw_zone_boxes(self, region: Optional[Dict[str, float]], *, filled: bool) -> bool:
        """Draw the zones over the view as boxes within ``region`` - where they
        act - in place of those drawn before; the rest of the scene and the
        camera stay. ``filled`` shades them, for a view with nothing inside them.
        Returns whether any box was drawn."""
        if self._plotter is None:
            return False
        for actor in self._zone_actors:
            try:
                self._plotter.remove_actor(actor, render=False)
            except Exception:  # noqa: BLE001 - gone with a cleared scene
                pass
        self._zone_actors = []
        self._zone_view = None if region is None else (region, filled)
        if region is None:
            return False
        zones = self._zone_editor.zones()
        try:
            for zone, colour in zip(zones, zone_colors(len(zones))):
                box = clip_box(zone, region)
                if box is None:
                    continue
                shape = self._pv.Box(bounds=box)
                if filled:
                    self._zone_actors.append(self._plotter.add_mesh(
                        shape, color=colour, opacity=0.25, reset_camera=False))
                self._zone_actors.append(self._plotter.add_mesh(
                    shape.outline(), color=colour, line_width=3, reset_camera=False))
                # At a top corner: boxes centred on the survey share a centre.
                self._zone_actors.append(self._plotter.add_point_labels(
                    [[box[0], box[2], box[5]]], [zone["name"]], font_size=10,
                    text_color="#222222", shape_opacity=0.0, show_points=False,
                    always_visible=True, reset_camera=False))
        except Exception as exc:  # noqa: BLE001 - the boxes are an overlay
            self.log(f"Zone boxes could not be drawn: {exc}", "warn")
        return bool(self._zone_actors)

    def _draw_zone_cells(self, mesh: Any, pv_mesh: Any) -> bool:
        """Draw the cells each zone took in its colour: on a mesh built to follow
        the zones they fill the box, otherwise they step along its faces. A zone
        a later one overlaps - a layer around a tank - is drawn see-through, so
        the later one shows inside it. Returns whether any were drawn."""
        zones = self._mesh_zones
        if not zones:
            return False
        # As generate_mesh zoned them: the inverted cells, by cell centre.
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        owner = np.full(markers.size, -1, dtype=int)
        inverted = markers > 1
        owner[inverted] = mesh3d_builder.box_zone_owner(
            np.asarray(mesh.cellCenters(), dtype=float)[inverted], zones)

        def overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
            return all(a[axis][0] < b[axis][1] and b[axis][0] < a[axis][1] for axis in "xyz")

        drawn = False
        for index, colour in enumerate(zone_colors(len(zones))):
            ids = np.flatnonzero(owner == index)
            if not ids.size:
                continue
            covered = any(overlap(zones[index], later) for later in zones[index + 1:])
            self._plotter.add_mesh(pv_mesh.extract_cells(ids), color=colour,
                                   opacity=0.3 if covered else 1.0, show_edges=True,
                                   edge_color="#333333", line_width=0.5, reset_camera=False)
            drawn = True
        if drawn:
            try:
                self._plotter.enable_depth_peeling()   # see-through surfaces in order
            except Exception:  # noqa: BLE001 - not every OpenGL has it; still drawn
                pass
        return drawn

    @staticmethod
    def _inverted_region(mesh: Any, pv_mesh: Any) -> Dict[str, float]:
        """The bounds of the mesh's inverted cells (marker above 1), where zones act."""
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        ids = np.flatnonzero(markers > 1)
        part = pv_mesh.extract_cells(ids) if 0 < ids.size < markers.size else pv_mesh
        x0, x1, y0, y1, z0, z1 = (float(value) for value in part.bounds)
        return {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1,
                "z_bottom": z0, "z_top": z1}

    # -- preview / generate --------------------------------------------------
    def _preview_sensors(self) -> None:
        try:
            cfg = self._collect_config()
            if cfg["mesh_engine"] == mesh3d_builder.E4D_ENGINE:
                self._preview_e4d(cfg)
                return
            _, sensors = mesh3d_builder.build_electrodes(cfg)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Sensor preview failed: {exc}", "error")
            self._info.setText(f"Preview failed: {exc}")
            return
        self._sensors = sensors
        self._render_sensors(sensors, cfg)
        x, y, z = sensors["x"], sensors["y"], sensors["z"]
        self._info.setText(
            f"{len(sensors)} sensors  ·  x [{x.min():.1f}, {x.max():.1f}]  "
            f"y [{y.min():.1f}, {y.max():.1f}]  z [{z.min():.2f}, {z.max():.2f}] m"
        )
        self.log(f"Previewed {len(sensors)} sensors.", "info")

    # -- the E4D layout -------------------------------------------------------
    def _preview_e4d(self, cfg: dict) -> None:
        """Show the E4D layout before meshing: every control point by its role,
        the fine zone's internal boundaries and the outer boundary."""
        layout, sensors = mesh3d_builder.e4d_configuration(cfg)
        problems = layout.validate()
        self._sensors = sensors
        flags = layout.flags
        counts = {flag: int((flags == flag).sum()) for flag in (0, 1, 2)}
        outer = layout.points[layout.outer_indices()]
        span = np.ptp(outer[:, :2], axis=0) if len(outer) else np.zeros(2)
        # How the zones were fitted into the fine zone, when they had to be.
        notes = list(getattr(layout, "notes", []))
        self._info.setText(
            f"E4D layout: {len(layout.points)} control points ({counts[1]} on the surface, "
            f"{counts[0]} internal, {counts[2]} on the outer boundary), "
            f"{len(layout.boundaries)} internal boundaries, {len(layout.zones)} zones; "
            f"domain {span[0]:.0f} × {span[1]:.0f} m to {layout.bottom:.1f} m"
            + "".join(f"<br><span style='color:#5a6a7a'>{html.escape(note)}</span>"
                      for note in notes)
            + (f"<br><span style='color:#b42318'>{' '.join(problems)}</span>" if problems else ""))
        self.log(f"Previewed the E4D layout: {len(layout.points)} control points, "
                 f"{len(layout.zones)} zones.", "warn" if problems else "info")
        for note in notes:
            self.log(f"  {note}", "info")
        self._update_zone_note()
        if self._plotter is None:
            return
        try:
            self._plotter.clear()
            self._colour_actor(None, cmaps.MESH_REGIONS, "coolwarm", None)
            colours = {0: "#1f78b4", 1: "#33a02c", 2: "#6a3d9a"}
            for flag, colour in colours.items():
                chosen = layout.points[flags == flag]
                if len(chosen):
                    self._plotter.add_mesh(self._pv.PolyData(chosen), color=colour,
                                           render_points_as_spheres=True, point_size=7)
            for _, indices in layout.boundaries:
                ring = layout.points[[int(i) - 1 for i in indices]]
                self._plotter.add_mesh(self._pv.lines_from_points(np.vstack([ring, ring[:1]])),
                                       color="#1f78b4", line_width=2)
            if len(outer):
                bottom = outer.copy()
                bottom[:, 2] = layout.bottom
                for ring in (outer, bottom):
                    self._plotter.add_mesh(self._pv.lines_from_points(np.vstack([ring, ring[:1]])),
                                           color="#6a3d9a", line_width=1)
                for top, low in zip(outer, bottom):
                    self._plotter.add_mesh(self._pv.lines_from_points(np.vstack([top, low])),
                                           color="#6a3d9a", line_width=1)
            self._overlay_sensors(sensors, labels=False)
            # A loaded configuration has zones of its own; the list is not used.
            # Zones are framed on the fine zone they lie in, a small part of the
            # domain.
            region = None if self._e4d_config_path else self._zone_region(cfg, sensors)
            self._plotter.add_axes()
            if self._draw_zone_boxes(region, filled=True):
                self._plotter.reset_camera(bounds=[region[key] for key in (
                    "x_min", "x_max", "y_min", "y_max", "z_bottom", "z_top")])
            else:
                self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"E4D layout render failed: {exc}", "warn")

    def _load_e4d_config_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load E4D mesh configuration", "",
                                              _E4D_CONFIG_FILTER)
        if path:
            self._load_e4d_config(path)

    def _load_e4d_config(self, path: str) -> Dict[str, Any]:
        """Use an E4D configuration file in place of the sensor array."""
        from PyHydroGeophysX.core import e4d_mesh

        try:
            layout = e4d_mesh.read_e4d_config(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not read the E4D configuration {Path(path).name}: {exc}", "error")
            return {"status": "failed", "error": str(exc)}
        problems = layout.validate()
        self._e4d_config_path = str(path)
        self._e4d_clear.setEnabled(True)
        limits = ", ".join(f"zone {z.index}: " + ("unconstrained" if z.max_volume >= 1e11
                                                   else f"{z.max_volume:g} m³")
                           for z in layout.zones)
        self._e4d_cfg_label.setText(
            f"<b>{Path(path).name}</b>: {len(layout.points)} control points, "
            f"{len(layout.boundaries)} internal boundaries, {len(layout.zones)} zones "
            f"({limits}); bottom at {layout.bottom:g} m. It sets the electrodes, the domain "
            "and the element sizes, so steps 1, 3 and 4 are not used."
            + (f"<br><span style='color:#b42318'>{' '.join(problems)}</span>" if problems else ""))
        self._mesh_engine.setCurrentText(mesh3d_builder.E4D_ENGINE)
        # Also when the engine already was E4D, which emits no change.
        self._update_visibility()
        self.log(f"Loaded the E4D mesh configuration {Path(path).name}.",
                 "warn" if problems else "success")
        return {"status": "ok", "control_points": len(layout.points),
                "internal_boundaries": len(layout.boundaries),
                "zones": [z.index for z in layout.zones], "problems": problems}

    def _clear_e4d_config(self) -> None:
        self._e4d_config_path = ""
        self._e4d_clear.setEnabled(False)
        self._e4d_cfg_label.setText(_E4D_FROM_STEPS)
        self._update_visibility()

    def _open_e4d_mesh_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open E4D mesh", "", _E4D_MESH_FILTER)
        if path:
            self._open_e4d_mesh(path)

    def _open_e4d_mesh(self, path: str) -> None:
        """Read an E4D mesh off the UI thread - a large one takes a few seconds -
        and show it by zone."""
        from PyHydroGeophysX.core import e4d_mesh

        self._info.setText(f"Reading {Path(path).name}…")
        worker = TaskWorker(e4d_mesh.read_e4d_mesh, path)
        worker.succeeded.connect(lambda mesh, p=path: self._on_e4d_mesh_read(p, mesh))
        worker.failed.connect(lambda message, p=path: self.log(
            f"Could not read the E4D mesh {Path(p).name}: {message}", "error"))
        self.register_worker(worker)
        worker.start()

    def _on_e4d_mesh_read(self, path: str, mesh: Any) -> None:
        self._mesh = mesh
        self._sensors = None
        self._mesh_zones, self._mesh_zone_key = [], None   # not built here
        self._render_mesh(mesh, None)
        self._update_zone_note()
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        zones = ", ".join(f"zone {int(z)}: {int((markers == z).sum())}" for z in np.unique(markers))
        extra = " · resistivity from the .sig" if mesh.haveData("resistivity") else ""
        self._info.setText(f"<b>{Path(path).name}</b>: {mesh.cellCount()} cells, "
                           f"{mesh.nodeCount()} nodes ({zones}){extra}")
        self.log(f"Opened the E4D mesh {Path(path).name}: {mesh.cellCount()} cells, "
                 f"{zones}.", "success")

    def _generate_mesh(self) -> None:
        cfg = self._collect_config()
        try:
            run = self.begin_persisted_run("mesh3d.build", "mesh3d.build")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        output_dir = run.outputs_dir
        cfg["output_dir"] = str(output_dir)
        topography_points = cfg.pop("topography_points", None)
        inputs = {}
        e4d_source = cfg.pop("e4d_config_path", "")
        if e4d_source:
            # The configuration travels with the run, as an input of its own,
            # so the recipe does not point at a file that may move.
            import shutil

            stored = run.inputs_dir / "e4d_mesh_configuration.cfg"
            shutil.copyfile(e4d_source, stored)
            inputs["e4d_config"] = ArtifactRef.from_path(
                stored, artifact_id="mesh3d:e4d_config", kind="e4d_mesh_configuration",
                base_dir=run.run_dir)
        if topography_points is not None:
            topo_path = run.inputs_dir / "mesh3d_topography_points.npy"
            np.save(topo_path, np.asarray(topography_points, dtype=float))
            inputs["topography_points"] = ArtifactRef.from_path(
                topo_path,
                artifact_id="mesh3d:topography_points",
                kind="topography_points",
                base_dir=run.run_dir,
            )
        cfg["output_formats"] = self._selected_formats()
        cfg["output_name"] = self._safe_name()
        spec = WorkflowSpec(
            workflow_id="mesh3d.build",
            inputs=inputs,
            parameters=cfg,
            metadata={"source": "qt"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, run.run_dir, stem="mesh3d"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._mesh_recipe_path = str(recipe_path)
        self._gen_busy = BusyStateController([self._gen_btn])
        self._gen_busy.start()
        self._gen_btn.setText("Generating…")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.log("Generating 3D mesh…", "info")
        # In a process of its own: Gmsh, TetGen and PyGIMLi hold the GIL for
        # seconds, which in a thread would stop the window painting meanwhile.
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir,
                                       run.result_path, objects=("mesh", "electrodes"))
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(lambda res: self._on_mesh_workflow_ok(cfg, res))
        worker.failed.connect(self._on_mesh_failed)
        worker.finished.connect(self._reset_gen_button)
        self._gen_worker = self.register_worker(worker)
        worker.start()

    def _on_mesh_workflow_ok(self, cfg: dict, result: WorkflowRunResult) -> None:
        try:
            self._on_mesh_generated(cfg, result.legacy_payload())
        finally:
            if hasattr(self.state, "update_workflow_result"):
                self.state.update_workflow_result(
                    self.module_key,
                    "mesh3d.build",
                    result.to_dict(),
                    recipe_path=self._mesh_recipe_path,
                )

    def _on_mesh_generated(self, cfg: dict, res: dict) -> None:
        mesh = res.get("mesh")
        sensors = res.get("electrodes")
        self._mesh = mesh
        self._sensors = sensors
        # The zones this mesh was built with: drawn by the cells they took, and
        # kept to tell when the list has moved on since.
        zone_report = list(res.get("zones") or [])
        self._mesh_zones = list(cfg.get("zones") or []) if zone_report else []
        self._mesh_zone_key = self._zone_key(self._mesh_zones, cfg.get("conform_to_zones"),
                                             cfg.get("decouple_zones"))
        # Counts only for the boxes they were counted on: a zone moved while
        # the mesh was building shows none rather than another box's.
        def boxes(zones):
            return [[zone[axis] for axis in "xyz"] for zone in zones]

        same = boxes(self._zone_editor.zones()) == boxes(cfg.get("zones") or [])
        self._zone_editor.set_cells(zone_report if same else [])
        summary = mesh3d_builder.mesh_summary(mesh)
        self._render_mesh(mesh, sensors)
        # save selected outputs
        outputs = dict(res.get("outputs") or {})
        if not outputs:
            try:
                outputs = mesh3d_builder.save_outputs(
                    mesh, sensors, Path(cfg["output_dir"]), self._safe_name(), self._selected_formats())
            except Exception as exc:  # noqa: BLE001
                self.log(f"Saving outputs failed: {exc}", "warn")
        cells = summary.get("Cells", "?")
        nodes = summary.get("Nodes", "?")
        # What each E4D zone came out as, against the limit it was given: Gmsh,
        # standing in for TetGen, aims at the limit rather than enforcing it.
        zones = ""
        for entry in res.get("e4d_zones") or []:
            limit = ("unconstrained" if entry["max_volume"] >= 1e11
                     else f"limit {entry['max_volume']:g} m³")
            zones += (f"<br>zone {entry['zone']}: {entry['cells']} cells, largest "
                      f"{entry['largest_volume']:.3g} m³ ({limit})")
        # Built from an E4D file, the points shown are its control points -
        # refinement points and zone corners among them - not sensors.
        points = ("control points" if cfg.get("mesh_engine") == mesh3d_builder.E4D_ENGINE
                  and self._e4d_config_path else "sensors")
        # What the zone list did - and a zone that did nothing, said, not dropped.
        zone_line = ""
        if zone_report:
            conform = bool(res.get("zones_conform"))
            how = "the mesh follows their faces" if conform else "by cell centre"
            # A zone the prism layers could not bend to on a slope is by cell centre.
            zone_line = f"<br>zones, {how}: " + ", ".join(
                f"{html.escape(entry['name'])} {entry['cells']} cells"
                + (f" (region {entry['marker']})" if cfg.get("decouple_zones") else "")
                + (" (by cell centre)" if conform and entry.get("follows") is False else "")
                for entry in zone_report)
            empty = [entry["name"] for entry in zone_report if not entry["cells"]]
            if empty:
                zone_line += (f"<br><span style='color:#b42318'>{html.escape(', '.join(empty))}: "
                              "no cell, so no effect</span>")
                self.log(f"Zone(s) {', '.join(empty)} took no cell of the mesh and have no "
                         "effect: they lie outside its inverted region, or zones later in "
                         "the list cover them.", "warn")
        elif cfg.get("zones"):
            zone_line = ("<br><span style='color:#b42318'>zones not used: the loaded E4D "
                         "configuration defines its own</span>")
            self.log("The zones were not used: the loaded E4D configuration defines its own.",
                     "warn")
        self._info.setText(
            f"<b>{res.get('generator')}</b><br>cells: {cells}  ·  nodes: {nodes}  ·  "
            f"{points}: {len(sensors)}{zones}{zone_line}"
        )
        self.log(f"Mesh generated: {cells} cells, {nodes} nodes ({res.get('generator')}).", "success")
        for key, path in outputs.items():
            self.log(f"Saved {key}: {path}", "info")
        # The mesh is in memory and on screen either way; only the file is missing.
        # Saying so beats a success line that quietly wrote nothing.
        if res.get("output_error"):
            self.log(
                f"The mesh is ready and displayed, but writing it to "
                f"{cfg.get('output_dir')} failed: {res['output_error']}. "
                "Use Export to save it elsewhere.", "warn")
        self.report_result({
            "generator": res.get("generator"), "n_sensors": int(len(sensors)),
            "cells": summary.get("Cells"), "nodes": summary.get("Nodes"),
            "outputs": outputs, "output_dir": cfg["output_dir"],
            "zones": zone_report, "zones_conform": bool(res.get("zones_conform")),
        })
        self._update_zone_note()

    def _on_mesh_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "mesh3d.build")
        self.log(f"Mesh generation failed: {message}", "error")
        self._info.setText(f"Generation failed: {message}")

    def _reset_gen_button(self) -> None:
        if self._gen_busy is not None:
            self._gen_busy.finish()
            self._gen_busy = None
        self._gen_btn.setText("Generate mesh")
        self._progress.setVisible(False)

    # -- hidden 3D ERT forward action (AQUAH only) -------------------------
    def _set_ert_forward_param(self, key: str, value: Any) -> None:
        """Validate and store an agent-only 3D ERT forward parameter."""
        if key == "scheme":
            scheme = str(value)
            if scheme not in _ERT_FORWARD_SCHEMES:
                raise ValueError(f"scheme must be one of {list(_ERT_FORWARD_SCHEMES)}")
            self._ert_forward_params[key] = scheme
            return
        numeric = float(value)
        limits = {"background_res": (1.0, 100000.0), "noise": (0.0, 0.5)}
        low, high = limits[key]
        if not low <= numeric <= high:
            raise ValueError(f"{key} must be between {low:g} and {high:g}")
        self._ert_forward_params[key] = numeric

    def _run_ert_forward(self, marker_res=None) -> None:
        """Run the hidden 3D ERT forward action requested through AQUAH."""
        if self._mesh is None or self._sensors is None:
            self.log("Generate a 3D mesh first (Generate mesh, step 6).", "warn")
            return
        try:
            run = self.begin_persisted_run("ert3d.forward", "ert3d.forward")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not prepare Project run: {exc}", "error")
            return
        bundle_dir = run.run_dir
        input_dir = run.inputs_dir
        mesh_path = input_dir / "forward_mesh.bms"
        mesh_structure_path = input_dir / "forward_mesh.bms.structure.json"
        sensors_path = input_dir / "forward_sensors.csv"
        from PyHydroGeophysX.core.mesh_serialization import save_mesh_artifact

        save_mesh_artifact(self._mesh, mesh_path, mesh_structure_path)
        self._sensors.to_csv(sensors_path, index=False)
        parameters = {
            "scheme": self._ert_forward_params["scheme"],
            "background_res": self._ert_forward_params["background_res"],
            "marker_res": {
                str(key): float(value) for key, value in (marker_res or {}).items()
            },
            "noise": self._ert_forward_params["noise"],
        }
        # Each zone its resistivity, by cell centre - the box exactly on a mesh
        # built to follow it.
        zones = self._zone_editor.zones()
        if zones:
            parameters["zones"] = zones
        spec = WorkflowSpec(
            workflow_id="ert3d.forward",
            inputs={
                "mesh": ArtifactRef.from_path(
                    mesh_path,
                    artifact_id="ert3d:mesh",
                    kind="pygimli_mesh",
                    base_dir=bundle_dir,
                ),
                "mesh_structure": ArtifactRef.from_path(
                    mesh_structure_path,
                    artifact_id="ert3d:mesh_structure",
                    kind="mesh_structure",
                    base_dir=bundle_dir,
                ),
                "sensors": ArtifactRef.from_path(
                    sensors_path,
                    artifact_id="ert3d:sensors",
                    kind="electrode_geometry",
                    base_dir=bundle_dir,
                ),
            },
            parameters=parameters,
            seed=42,
            metadata={"source": "qt", "mesh_roundtrip": "bms"},
        )
        recipe_path, script_path = export_workflow_bundle(
            spec, bundle_dir, stem="ert3d_forward"
        )
        self._reproduce.set_bundle(recipe_path, script_path)
        self._ert3d_recipe_path = str(recipe_path)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self.log("Running agent-requested 3D ERT forward modeling…", "info")
        # The result is shown from the VTK file it writes; no object comes back.
        worker = ProcessWorkflowWorker(recipe_path, run.run_dir, run.outputs_dir,
                                       run.result_path)
        worker.logged.connect(lambda m: self.log(m, "info"))
        worker.succeeded.connect(self._on_ert3d_workflow_ok)
        worker.failed.connect(self._on_ert_forward_failed)
        worker.finished.connect(self._reset_ert_forward_progress)
        self._ert_fwd_worker = self.register_worker(worker)
        worker.start()

    def _on_ert3d_workflow_ok(self, result: WorkflowRunResult) -> None:
        if hasattr(self.state, "update_workflow_result"):
            self.state.update_workflow_result(
                self.module_key,
                "ert3d.forward",
                result.to_dict(),
                recipe_path=self._ert3d_recipe_path,
            )
        self._on_ert_forward_ok(result.legacy_payload())

    def _on_ert_forward_ok(self, result: dict) -> None:
        self._ert_fwd_result = result
        rmin, rmax = result.get("rhoa_min"), result.get("rhoa_max")
        rng = f"{rmin:.1f}-{rmax:.1f} ohm-m" if rmin is not None and rmax is not None else "n/a"
        self.log(f"3D ERT forward complete: {result.get('n_measurements')} measurements, "
                 f"rhoa {rng}. Saved {result.get('data_file')}.", "success")
        self.report_result({"ert3d_forward": result})
        vtk = result.get("vtk")
        if vtk:
            self.load_view_file(vtk)

    def _on_ert_forward_failed(self, message: str) -> None:
        self.fail_persisted_run(message, "ert3d.forward")
        self.log(f"3D ERT forward failed: {message}", "error")
        self._info.setText(f"3D ERT forward failed: {message}")

    def _reset_ert_forward_progress(self) -> None:
        self._progress.setVisible(False)

    def _safe_name(self) -> str:
        import re

        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._mesh_name.text().strip())
        return name or "mesh3d"

    # -- rendering -----------------------------------------------------------
    def _overlay_sensors(self, sensors_df, labels: bool = False) -> None:
        import numpy as np

        pts = np.column_stack([
            np.asarray(sensors_df["x"], dtype=float),
            np.asarray(sensors_df["y"], dtype=float),
            np.asarray(sensors_df["z"], dtype=float),
        ])
        cloud = self._pv.PolyData(pts)
        self._plotter.add_mesh(cloud, color="#d7191c", render_points_as_spheres=True, point_size=12)
        if labels and len(pts) <= 60 and "n" in sensors_df:
            try:
                self._plotter.add_point_labels(
                    pts, [str(int(n)) for n in sensors_df["n"]],
                    font_size=11, point_size=1, text_color="#222222", always_visible=True)
            except Exception:  # noqa: BLE001
                pass

    def _render_sensors(self, sensors_df, cfg: dict) -> None:
        if self._plotter is None:
            return
        try:
            import numpy as np

            self._plotter.clear()
            # Electrodes and a translucent terrain: nothing drawn through a map.
            self._colour_actor(None, cmaps.MESH_REGIONS, "coolwarm", None)
            self._overlay_sensors(sensors_df, labels=True)
            if cfg["array_type"] == "Surface grid" and cfg["mesh_type"] == "Surface with topography":
                topo = mesh3d_builder.topography_function(cfg)
                xs = np.asarray(sensors_df["x"], dtype=float)
                ys = np.asarray(sensors_df["y"], dtype=float)
                margin = max(cfg["dx"], cfg["dy"], 1.0) * 2.0
                gx = np.linspace(xs.min() - margin, xs.max() + margin, 40)
                gy = np.linspace(ys.min() - margin, ys.max() + margin, 40)
                gxx, gyy = np.meshgrid(gx, gy)
                gzz = np.vectorize(topo)(gxx, gyy)
                surf = self._pv.StructuredGrid(gxx, gyy, gzz)
                self._plotter.add_mesh(surf, cmap="gist_earth", opacity=0.35, show_scalar_bar=False)
            self._draw_zone_boxes(self._zone_region(cfg, sensors_df), filled=True)
            self._plotter.add_axes()
            self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Electrode render failed: {exc}", "warn")

    def _render_mesh(self, mesh, sensors_df) -> None:
        if self._plotter is None:
            return
        try:
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "mesh3d_view.vtk"
            mesh.exportVTK(str(tmp))
            pv_mesh = self._pv.read(str(tmp))
            self._plotter.clear()
            scalars = None
            for key in ("Marker", "marker", "Attribute", "region"):
                if key in pv_mesh.cell_data:
                    scalars = key
                    break
            cmap = self._colormap.set_target(cmaps.MESH_REGIONS, "coolwarm")
            actor = self._plotter.add_mesh(
                pv_mesh, scalars=scalars, show_edges=True, edge_color="#555555",
                line_width=0.5, cmap=cmaps.to_pyvista(cmap), show_scalar_bar=bool(scalars))
            self._colour_actor(actor, cmaps.MESH_REGIONS, "coolwarm", scalars)
            # The cells each zone took, seen through the rest of the mesh, and
            # the zone list as outlines, which show it moving on after the build;
            # framed on where the zones act, which on an E4D mesh is a small part.
            region = self._inverted_region(mesh, pv_mesh)
            zoned = self._draw_zone_cells(mesh, pv_mesh)
            if zoned:
                actor.GetProperty().SetOpacity(0.2)
            self._draw_zone_boxes(region, filled=False)
            if sensors_df is not None:
                self._overlay_sensors(sensors_df, labels=False)
            self._plotter.add_axes()
            if zoned:
                self._plotter.reset_camera(bounds=[region[key] for key in (
                    "x_min", "x_max", "y_min", "y_max", "z_bottom", "z_top")])
            else:
                self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Mesh render failed: {exc}", "warn")

    # -- view tools ----------------------------------------------------------
    def _refresh_plotter(self) -> None:
        """Repaint the embedded VTK widget after a stacked-page transition."""
        if self._plotter is None:
            return
        try:
            interactor = getattr(self._plotter, "interactor", None)
            if interactor is not None:
                interactor.raise_()
                interactor.update()
            self._plotter.render()
        except Exception as exc:  # noqa: BLE001 - rendering backend specific
            self.log(f"3D viewer refresh failed: {exc}", "warn")

    def _load_mesh(self) -> None:
        if self._plotter is None:
            self.log("3D viewer unavailable.", "warn")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Load mesh", "", _MESH_FILTER)
        if not path:
            return
        try:
            mesh = self._pv.read(path)
            self._plotter.clear()
            self._zone_view = None          # an unrelated volume: no zones drawn over it
            cmap = self._colormap.set_target(cmaps.MODEL3D, "viridis")
            actor = self._plotter.add_mesh(mesh, cmap=cmaps.to_pyvista(cmap), show_edges=False)
            self._colour_actor(actor, cmaps.MODEL3D, "viridis",
                               getattr(mesh, "active_scalars_name", None))
            self._plotter.add_axes()
            self._plotter.reset_camera()
            self.log(f"Loaded mesh {Path(path).name} ({getattr(mesh, 'n_points', 0)} points)", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to load mesh '{Path(path).name}': {exc}", "error")

    def load_view_file(self, path: str) -> None:
        """Load and display a mesh / 3D volume file (e.g. a seismic 3D model).

        Called by the main window when another module asks to view its output in
        the 3D viewer. Picks a sensible cell/point scalar (Velocity, Marker, …).
        """
        p = Path(path)
        if not p.exists():
            self.log(f"File to view not found: {p}", "warn")
            return
        if self._plotter is None or self._pv is None:
            self.log(f"3D viewer unavailable; '{p.name}' was produced but cannot be "
                     f"shown here ({_INSTALL_MESSAGE}).", "warn")
            self._info.setText(f"3D viewer unavailable. File ready at:<br><code>{p}</code>")
            return
        try:
            mesh = self._pv.read(str(p))
            scalars = None
            for key in ("Velocity", "velocity", "Marker", "marker", "Elevation", "Attribute"):
                if key in getattr(mesh, "cell_data", {}) or key in getattr(mesh, "point_data", {}):
                    scalars = key
                    break
            self._plotter.clear()
            self._zone_view = None          # an unrelated volume: no zones drawn over it
            cmap = self._colormap.set_target(cmaps.MODEL3D, "turbo")
            actor = self._plotter.add_mesh(mesh, scalars=scalars, cmap=cmaps.to_pyvista(cmap),
                                           opacity=0.6 if scalars == "Velocity" else 1.0,
                                           show_edges=False, show_scalar_bar=bool(scalars))
            self._colour_actor(actor, cmaps.MODEL3D, "turbo", scalars)
            try:
                self._plotter.add_mesh(mesh.outline(), color="grey")
            except Exception:  # noqa: BLE001 - outline is cosmetic
                pass
            self._plotter.add_axes(); self._plotter.reset_camera()
            self._refresh_plotter()
            # A second repaint after pending resize/expose events prevents the
            # previous module's framebuffer from showing through QtInteractor.
            QTimer.singleShot(0, self._refresh_plotter)
            self._mesh = None  # this is a pyvista object, not a pygimli mesh
            self._info.setText(
                f"Viewing <b>{p.name}</b>  ·  {getattr(mesh, 'n_points', 0)} points"
                + (f"  ·  scalar: {scalars}" if scalars else ""))
            self.log(f"Loaded {p.name} into the 3D viewer (scalar: {scalars}).", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Failed to display '{p.name}': {exc}", "error")

    def _toggle_clip(self) -> None:
        if self._plotter is None or self._mesh is None:
            self.log("Generate or load a mesh first.", "debug")
            return
        try:
            import tempfile

            tmp = Path(tempfile.mkdtemp()) / "mesh3d_clip.vtk"
            self._mesh.exportVTK(str(tmp))
            pv_mesh = self._pv.read(str(tmp))
            self._plotter.clear()
            cmap = self._colormap.set_target(cmaps.MESH_REGIONS, "coolwarm")
            actor = self._plotter.add_mesh_clip_plane(pv_mesh, cmap=cmaps.to_pyvista(cmap))
            self._colour_actor(actor, cmaps.MESH_REGIONS, "coolwarm",
                               getattr(pv_mesh, "active_scalars_name", None))
            self._plotter.add_axes()
            self._plotter.reset_camera()
            self.log("Added interactive clipping plane.", "success")
        except Exception as exc:  # noqa: BLE001
            self.log(f"Clipping plane not available: {exc}", "warn")

    def _toggle_axes(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.add_axes()
            self._plotter.render()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Axes toggle failed: {exc}", "error")

    def _reset_view(self) -> None:
        if self._plotter is None:
            return
        try:
            self._plotter.reset_camera()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Reset view failed: {exc}", "error")

    def _open_output_folder(self) -> None:
        out = self.state.module_results.get(self.module_key, {}).get("output_dir")
        out = out or str(self.state.results_store_root or self._default_output_dir())
        path = Path(out)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.log(f"Output folder does not exist yet: {path}", "warn")

    # -- AQUAH agent interface ----------------------------------------------
    def agent_describe(self) -> Dict[str, Any]:
        return {
            "module": self.module_key,
            "title": self.module_title,
            "state": self._agent_status(),
            "actions": [
                {"name": "set_params", "args": {"params": {"<key>": "value"}},
                 "desc": ("Set parameters. Geometry: mesh_type, array_type, mesh_engine, topo_type. "
                          "Surface grid: nx, ny, dx, dy, x_offset, y_offset. Box: box_length, box_width, "
                          "box_height. Topography: z_flat, z_base, tilt_x, tilt_y, hill_base, hill_amp, "
                          "hill_sigma, hill_cx, hill_cy, topo_expr. Mesh: elec_refine, attractor, "
                          "bound_refine, para_depth, dz_fine, dz_coarse, bound_ext, single_region. "
                          "Borehole: bh_x, bh_y, bh_z_start, bh_z_end, bh_n. Output: mesh_name, "
                          "fmt_bms, fmt_vtk, fmt_csv. E4D engine (mesh_engine "
                          f"'{mesh3d_builder.E4D_ENGINE}'): e4d_fine_padding and "
                          "e4d_fine_depth_padding (m, the fine zone beyond the electrodes and "
                          "below the deepest one), e4d_fine_volume (m^3, largest element "
                          "there), e4d_outer_distance (m, outer boundary beyond the fine "
                          "zone), e4d_bottom_depth (m), e4d_quality (radius-edge ratio), "
                          "e4d_refine_offset (m, 0 for no refinement points), "
                          "e4d_conductivity (S/m), e4d_mesher (auto/tetgen/gmsh), e4d_tetgen "
                          "(path to the program). Zones, boxes of known or assumed "
                          "resistivity on every engine: zones (replaces the list; each "
                          "{name, x: [from, to], y: [from, to], z: [bottom, top] as "
                          "elevations - a top at or above the ground reaches up to it - "
                          "resistivity in ohm-m}; a layer spans the mesh sideways, so give "
                          "it x and y ranges far past the survey), conform_to_zones (bool: "
                          "build the mesh so the zone faces are cell faces), "
                          "decouple_zones (bool: each zone a region of its own, markers "
                          f"{mesh3d_builder.ZONE_MARKER_START}, "
                          f"{mesh3d_builder.ZONE_MARKER_START + 1}, ..., so an inversion "
                          "does not smooth across its faces). Only inverted cells are "
                          "zoned, and on the E4D engine zones must lie in its fine zone. "
                          "run_ert_forward gives each zone its resistivity.")},
                {"name": "load_e4d_config", "args": {"path": "str"},
                 "desc": ("Build from an existing E4D mesh configuration (.cfg) instead of "
                          "the sensor array; selects the E4D engine. 'generate' then builds "
                          "it.")},
                {"name": "open_e4d_mesh", "args": {"path": "str"},
                 "desc": ("View a mesh E4D or TetGen built (.1.node / .1.ele, with the .trn "
                          "beside it), coloured by zone.")},
                {"name": "preview_sensors", "args": {},
                 "desc": "Build and preview the sensor layout for the current config."},
                {"name": "generate", "args": {},
                 "desc": "Generate the 3D mesh and save the selected output formats."},
                {"name": "run_ert_forward",
                 "args": {"scheme": "dd/wa/slm/wb", "background_res": "float",
                          "noise": "float", "marker_res": "{marker: resistivity} (optional)"},
                 "desc": ("Hidden UI action: run 3D ERT forward modeling on the generated "
                          "mesh; the zones set with set_params take their resistivity.")},
                {"name": "load_mesh", "args": {"path": "str"},
                 "desc": "Load a mesh / 3D volume file into the viewer."},
                {"name": "get_status", "args": {},
                 "desc": "Report the geometry settings and whether a sensor layout / mesh exists."},
            ],
        }

    def agent_apply(self, action: str, args: Dict[str, Any]) -> Dict[str, Any]:
        args = args or {}
        handlers = {
            "set_params": lambda: self._agent_set_params(args.get("params", args)),
            "preview_sensors": lambda: self._agent_preview_sensors(),
            "generate": lambda: self._agent_generate(),
            "run_ert_forward": lambda: self._agent_run_ert_forward(args),
            "load_mesh": lambda: self._agent_load_mesh(args.get("path")),
            "load_e4d_config": lambda: self._agent_e4d_file(args.get("path"), "config"),
            "open_e4d_mesh": lambda: self._agent_e4d_file(args.get("path"), "mesh"),
            "get_status": lambda: self._agent_status(),
        }
        handler = handlers.get(action)
        if handler is None:
            return {"status": "failed", "error": f"Unknown action '{action}'.",
                    "valid_actions": list(handlers.keys())}
        return handler()

    def _agent_status(self) -> Dict[str, Any]:
        last = self.state.module_results.get(self.module_key, {})
        return {
            "status": "ok",
            "mesh_type": self._mesh_type.currentText(),
            "array_type": self._array_type.currentText(),
            "mesh_engine": self._mesh_engine.currentText(),
            "topo_type": self._topo_type.currentText(),
            "has_sensors": self._sensors is not None,
            "has_mesh": self._mesh is not None,
            "has_ert_forward": self._ert_fwd_result is not None,
            "e4d_config": self._e4d_config_path,
            "e4d_mesher": self._e4d_mesher_status(),
            # Each zone with the cells it took in the last mesh (None before one).
            "zones": self._zone_editor.zone_report(),
            "conform_to_zones": self._zone_editor.conform_to_zones(),
            "decouple_zones": self._zone_editor.decouple_zones(),
            "zone_note": self._zone_editor.note(),
            "output_dir": "Managed by active Project",
            "last_result_keys": sorted(last.keys()),
        }

    def _agent_e4d_file(self, path: Any, kind: str) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path'."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        if kind == "config":
            return self._load_e4d_config(str(p))
        self._open_e4d_mesh(str(p))
        return {"status": "started", "message": f"Reading {p.name}; ask for status shortly."}

    def _agent_preview_sensors(self) -> Dict[str, Any]:
        self._preview_sensors()
        return {"status": "ok",
                "sensors": int(len(self._sensors)) if self._sensors is not None else 0}

    def _agent_generate(self) -> Dict[str, Any]:
        self._generate_mesh()
        return {"status": "started", "message": "3D mesh generation started. Ask for status shortly."}

    def _agent_run_ert_forward(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if self._mesh is None or self._sensors is None:
            return {"status": "failed", "error": "Generate a 3D mesh first (action 'generate')."}
        try:
            for key in ("scheme", "background_res", "noise"):
                if key in args:
                    self._set_ert_forward_param(key, args[key])
        except Exception as exc:  # noqa: BLE001
            return {"status": "failed", "error": str(exc)}
        marker_res = args.get("marker_res")
        if marker_res is not None and not isinstance(marker_res, dict):
            marker_res = None
        self._run_ert_forward(marker_res=marker_res)
        return {"status": "started",
                "message": "3D ERT forward modeling started. Ask for status shortly.",
                "scheme": self._ert_forward_params["scheme"]}

    def _agent_load_mesh(self, path: Any) -> Dict[str, Any]:
        if not path:
            return {"status": "failed", "error": "Provide 'path' to a mesh / volume file."}
        p = Path(str(path))
        if not p.exists():
            return {"status": "failed", "error": f"File not found: {p}"}
        self.load_view_file(str(p))
        return {"status": "ok", "viewing": str(p)}

    def _agent_set_params(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            return {"status": "failed", "error": "Provide 'params' as a JSON object."}

        def set_combo(combo, value):
            items = [combo.itemText(i) for i in range(combo.count())]
            if str(value) not in items:
                raise ValueError(f"must be one of {items}")
            combo.setCurrentText(str(value))

        handlers = {
            "mesh_type": lambda v: set_combo(self._mesh_type, v),
            "array_type": lambda v: set_combo(self._array_type, v),
            "mesh_engine": lambda v: set_combo(self._mesh_engine, v),
            "topo_type": lambda v: set_combo(self._topo_type, v),
            "nx": lambda v: self._nx.setValue(int(v)),
            "ny": lambda v: self._ny.setValue(int(v)),
            "dx": lambda v: self._dx.setValue(float(v)),
            "dy": lambda v: self._dy.setValue(float(v)),
            "x_offset": lambda v: self._x_offset.setValue(float(v)),
            "y_offset": lambda v: self._y_offset.setValue(float(v)),
            "box_length": lambda v: self._box_length.setValue(float(v)),
            "box_width": lambda v: self._box_width.setValue(float(v)),
            "box_height": lambda v: self._box_height.setValue(float(v)),
            "z_flat": lambda v: self._z_flat.setValue(float(v)),
            "z_base": lambda v: self._z_base.setValue(float(v)),
            "tilt_x": lambda v: self._tilt_x.setValue(float(v)),
            "tilt_y": lambda v: self._tilt_y.setValue(float(v)),
            "hill_base": lambda v: self._hill_base.setValue(float(v)),
            "hill_amp": lambda v: self._hill_amp.setValue(float(v)),
            "hill_sigma": lambda v: self._hill_sigma.setValue(float(v)),
            "hill_cx": lambda v: self._hill_cx.setValue(float(v)),
            "hill_cy": lambda v: self._hill_cy.setValue(float(v)),
            "topo_expr": lambda v: self._topo_expr.setText(str(v)),
            "elec_refine": lambda v: self._elec_refine.setValue(float(v)),
            "attractor": lambda v: self._attractor.setValue(float(v)),
            "bound_refine": lambda v: self._bound_refine.setValue(float(v)),
            "para_depth": lambda v: self._para_depth.setValue(float(v)),
            "dz_fine": lambda v: self._dz_fine.setValue(float(v)),
            "dz_coarse": lambda v: self._dz_coarse.setValue(float(v)),
            "bound_ext": lambda v: self._bound_ext.setValue(float(v)),
            "single_region": lambda v: self._single_region.setChecked(bool(v)),
            "bh_x": lambda v: self._bh_x.setValue(float(v)),
            "bh_y": lambda v: self._bh_y.setValue(float(v)),
            "bh_z_start": lambda v: self._bh_z_start.setValue(float(v)),
            "bh_z_end": lambda v: self._bh_z_end.setValue(float(v)),
            "bh_n": lambda v: self._bh_n.setValue(int(v)),
            "mesh_name": lambda v: self._mesh_name.setText(str(v)),
            "fmt_bms": lambda v: self._fmt_bms.setChecked(bool(v)),
            "fmt_vtk": lambda v: self._fmt_vtk.setChecked(bool(v)),
            "fmt_csv": lambda v: self._fmt_csv.setChecked(bool(v)),
            "e4d_fine_padding": lambda v: self._e4d_pad.setValue(float(v)),
            "e4d_fine_depth_padding": lambda v: self._e4d_depth_pad.setValue(float(v)),
            "e4d_fine_volume": lambda v: self._e4d_fine_vol.setValue(float(v)),
            "e4d_outer_distance": lambda v: self._e4d_outer.setValue(float(v)),
            "e4d_bottom_depth": lambda v: self._e4d_bottom.setValue(float(v)),
            "e4d_quality": lambda v: self._e4d_quality.setValue(float(v)),
            "e4d_refine_offset": lambda v: self._e4d_refine.setValue(float(v)),
            "e4d_conductivity": lambda v: self._e4d_sigma.setValue(float(v)),
            "e4d_mesher": lambda v: self._e4d_mesher.setCurrentIndex(
                ["auto", "tetgen", "gmsh"].index(str(v).strip().lower())),
            "e4d_tetgen": lambda v: self._e4d_tetgen.setText(str(v)),
            "zones": lambda v: (self._zone_editor.set_zones(v), self._on_zones_changed()),
            "conform_to_zones": lambda v: self._zone_editor.set_conform_to_zones(bool(v)),
            "decouple_zones": lambda v: self._zone_editor.set_decouple_zones(bool(v)),
            "ert_scheme": lambda v: self._set_ert_forward_param("scheme", v),
            "ert_background_res": lambda v: self._set_ert_forward_param("background_res", v),
            "ert_noise": lambda v: self._set_ert_forward_param("noise", v),
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
