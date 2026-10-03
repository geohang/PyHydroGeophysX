"""The ERT page's Mesh tab: the inversion mesh before the run, and zones on it.

The mesh an inversion runs on decides what it can resolve, so it is worth seeing
before minutes go into a run - how deep the inverted domain reaches, how coarse
it gets away from the line, where the electrodes sit on it. The view draws the
mesh exactly as the inversion will build it (``ert_mesh.mesh_preview``):
the parameter domain, whose cells are inverted, and around it the outer region
pyGIMLi appends to carry the boundary condition far from the array - the
"infinite" part of the mesh, never inverted, and many times larger than the
section, so it is drawn only on request.

On top of the mesh the user draws zones: polygons carrying an a-priori
resistivity, optionally held fixed (see ``inversion.ert_zones``). The zone list
lives here; the page reads it when it builds a run, and so does the assistant.
Two choices go with the zones: whether the mesh is rebuilt along their
outlines, so no cell straddles one, and whether the smoothness constraint stops
at them, so the model may jump there. The cell edges it stops at are drawn.

The page puts its mesh settings above the zones (:meth:`set_settings_panel`),
so the mesh is set up where it is shown.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.inversion.ert_zones import normalize_zones, zone_colors, zone_prior
from PyHydroGeophysX.qt_apps import theme
from PyHydroGeophysX.qt_apps.widgets import length_units
from PyHydroGeophysX.qt_apps.widgets.readout import navigation_toolbar
from PyHydroGeophysX.visualization.axis_units import set_section_axes, to_display_length

__all__ = ["MeshPreviewView", "engine_zone_note"]

_OUTER_FACE = "#e5e5ea"
_OUTER_EDGE = "#a3acb7"
_PARA_EDGE = "#7b8794"
_ELECTRODE = "#007aff"
_CUT_EDGE = "#1f2933"

_NAME, _RHO, _FIXED, _CELLS = range(4)

_CUT_NOTE = (" The smoothness stops at the zone outlines, so the model may jump "
             "there.")


def engine_zone_note(engine: str, time_lapse: bool, zones: Sequence[Dict[str, Any]],
                     decouple: bool = False):
    """What ``engine`` will do with ``zones``, and whether that deserves a warning.

    Returns ``(text, warn)``. Every engine starts from the zones; only the
    in-house one regularizes toward them and holds a fixed zone, and the ADTLERT
    time-lapse backend takes no zone values. Every engine stops the smoothness
    at the outlines when ``decouple`` asks for it. The user should read that
    before the run, not discover it in the result.
    """
    engine = str(engine or "pyhydro").lower()
    any_fixed = any(bool(zone.get("fixed")) for zone in zones or [])
    cut = _CUT_NOTE if decouple else ""
    if engine == "pyhydro":
        return ("The in-house engine starts from the zone values, smooths the "
                "departure from them rather than the model itself, and does not "
                "invert a fixed zone." + cut, False)
    if time_lapse and engine == "adtlert":
        if decouple:
            return ("The ADTLERT time-lapse backend does not take zone values: "
                    "they set neither its start nor its reference model, but the "
                    "smoothness still stops at the zone outlines. Use the "
                    "in-house engine to invert with the values.", bool(zones))
        return ("The ADTLERT time-lapse backend does not take zones: they will not "
                "be applied. Use the in-house engine to invert with them.",
                bool(zones))
    if engine in ("r2", "r3t"):
        name = "R2" if engine == "r2" else "R3t"
        return (f"{name} starts from the zone values and holds a fixed zone at its "
                "value; the other cells are inverted." + cut, False)
    name = {"pygimli": "PyGIMLi", "e4d": "E4D"}.get(engine, "ADTLERT")
    if any_fixed:
        return (f"{name} cannot hold a zone fixed: the fixed zones only set its "
                "starting model and are inverted like any other cell. Use the "
                "in-house engine to keep them fixed." + cut, True)
    return f"{name} starts from the zone values and inverts every cell." + cut, False


class MeshPreviewView(QWidget):
    """The inversion mesh, the outer region on request, and the zone editor.

    ``zonesChanged`` carries the zone list after every edit made here; zones set
    from outside with :meth:`set_zones` do not emit it. ``rebuildRequested``
    asks the page to build the mesh again. ``conformChanged`` and
    ``decoupleChanged`` carry the two zone options whenever they change.
    """

    zonesChanged = Signal(list)
    rebuildRequested = Signal()
    conformChanged = Signal(bool)
    decoupleChanged = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._preview: Optional[Dict[str, Any]] = None
        self._source = ""
        self._message = "Load ERT data, or import a mesh, to see the inversion mesh."
        self._zones: List[Dict[str, Any]] = []
        self._owner: Optional[np.ndarray] = None  # zone of each parameter cell
        self._counts: List[Optional[int]] = []    # cells per zone on that mesh
        self._cut: Optional[np.ndarray] = None    # cell edges between zones
        self._default_rho = 100.0
        self._engine, self._time_lapse = "pyhydro", False
        self._selector = None

        self._fig = Figure(figsize=(7.5, 4.2), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setMinimumHeight(280)
        # The polygon tool restarts on Esc, which needs the keyboard.
        self._canvas.setFocusPolicy(Qt.StrongFocus)
        self._ax = self._fig.add_subplot(111)
        # The cursor position goes to a readout of its own: the toolbar's label
        # would re-lay out the page on every mouse move (see widgets.readout).
        self._toolbar, self._coords = navigation_toolbar(self._canvas, self)

        self._show_outer = QCheckBox("Outer region")
        self._show_outer.setToolTip(
            "Also draw the outer (\"infinite\") region: the coarse cells pyGIMLi "
            "appends around the parameter domain so the boundary condition sits far "
            "from the electrodes. They are part of the forward mesh but are never "
            "inverted, and they reach many times further than the section, so the "
            "view zooms out to show them.")
        self._show_outer.toggled.connect(self._redraw)
        self._show_edges = QCheckBox("Cell edges")
        self._show_edges.setChecked(True)
        self._show_edges.setToolTip("Draw the cell boundaries.")
        self._show_edges.toggled.connect(self._redraw)
        self._rebuild = QPushButton("Rebuild")
        self._rebuild.setIcon(theme.icon("fa5s.sync"))
        self._rebuild.setToolTip(
            "Build the mesh again from the current data and mesh settings. It is "
            "rebuilt on its own when this tab is opened after they change.")
        self._rebuild.clicked.connect(self.rebuildRequested.emit)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self._toolbar)
        bar.addWidget(self._coords, stretch=1)
        bar.addWidget(self._show_outer)
        bar.addWidget(self._show_edges)
        bar.addWidget(self._rebuild)

        self._info = QLabel()
        self._info.setWordWrap(True)
        self._info.setStyleSheet(f"color:{theme.PALETTE['muted']}; font-size:8pt;")

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._canvas)
        splitter.addWidget(self._build_side_panel())
        splitter.setCollapsible(0, False)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        # The section gets the room; the side column needs only its forms and
        # the table's four columns.
        splitter.setSizes([1000, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(bar)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._info)
        self._sync_zone_controls()
        self._redraw()
        # View > Length Units: the axes and the sizes quoted under them change;
        # the mesh and the zone outlines stay in metres.
        length_units.notifier().changed.connect(self._on_length_unit_changed)

    def _on_length_unit_changed(self, _unit: str) -> None:
        """Redraw the mesh in the studio's new length unit."""
        self._redraw()

    # -- side column ---------------------------------------------------------
    def _build_side_panel(self) -> QWidget:
        """The page's mesh settings (once set) above the zone editor, scrolling
        as one column when the two are taller than the tab."""
        column = QWidget()
        self._side = QVBoxLayout(column)
        self._side.setContentsMargins(0, 0, 4, 0)
        self._side.addWidget(self._build_zone_panel(), stretch=1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(column)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(520)
        return scroll

    # -- zone panel ----------------------------------------------------------
    def _build_zone_panel(self) -> QWidget:
        box = QGroupBox("A-priori zones")
        layout = QVBoxLayout(box)
        intro = QLabel(
            "Draw a polygon on the mesh and give it the resistivity known from a "
            "borehole, an excavation or another survey. The inversion starts from "
            "it; tick Fixed to keep the zone out of the inversion altogether.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        self._draw_btn = QPushButton("Draw zone")
        self._draw_btn.setIcon(theme.icon("fa5s.draw-polygon"))
        self._draw_btn.setCheckable(True)
        self._draw_btn.setToolTip(
            "Click on the mesh to place the vertices and click the first one again "
            "to close the polygon. Esc starts the polygon over; press the button "
            "again to stop without adding a zone.")
        self._draw_btn.toggled.connect(self._on_draw_toggled)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setIcon(theme.icon("fa5s.trash-alt"))
        self._remove_btn.setToolTip("Remove the selected zones.")
        self._remove_btn.clicked.connect(self._remove_selected)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Remove every zone.")
        self._clear_btn.clicked.connect(self._clear_zones)
        buttons.addWidget(self._draw_btn, 1)
        buttons.addWidget(self._remove_btn)
        buttons.addWidget(self._clear_btn)
        layout.addLayout(buttons)

        self._draw_hint = QLabel("Click to place vertices; click the first vertex to "
                                 "close. Esc starts over.")
        self._draw_hint.setWordWrap(True)
        self._draw_hint.setStyleSheet(f"color:{theme.PALETTE['accent']};")
        self._draw_hint.setVisible(False)
        layout.addWidget(self._draw_hint)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Zone", "ρ (Ω·m)", "Fixed", "Cells"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_NAME, QHeaderView.Stretch)
        for column in (_RHO, _FIXED, _CELLS):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self._table.setToolTip(
            "Double-click a name to rename the zone. Cells counts the cells of the "
            "parameter domain whose centre lies inside the polygon; where zones "
            "overlap, the one lower in the list takes the cell.")
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemSelectionChanged.connect(self._redraw)
        self._table.setMinimumHeight(120)
        layout.addWidget(self._table, stretch=1)

        # What the zones do to the mesh and to the regularization, beyond their
        # values. Both are off by default: a run with zones then does exactly
        # what it did before these existed.
        self._conform = QCheckBox("Mesh follows the zones")
        self._conform.setToolTip(
            "Build the mesh again so that the zone outlines become cell edges, "
            "the way E4D meshes its internal boundaries: no cell straddles an "
            "outline, so a zone's value - and a fixed zone - covers exactly the "
            "area drawn. The mesh is rebuilt after every zone you draw, move or "
            "remove; changing a zone's value leaves the mesh alone. Only the parts "
            "of an outline inside the inverted region are used. An outline meets "
            "the surface at the nearest surface node, and a vertex within a fifth "
            "of an electrode spacing of the surface or of a node is moved onto it, "
            "so no slivers of tiny cells form. Not available for an imported mesh.")
        self._conform_help = self._conform.toolTip()  # restored when it is available again
        self._conform.toggled.connect(self._on_conform_toggled)
        self._decouple = QCheckBox("Sharp zone edges")
        self._decouple.setToolTip(
            "Drop the smoothness constraint between cells on either side of a "
            "zone outline - between two zones, or a zone and the ground around "
            "it - so the inversion may put a sharp contrast there instead of "
            "smearing it out. Use it for a boundary known from a borehole, GPR "
            "or an excavation. The cell edges it cuts are drawn in black. Every "
            "engine honours it; rebuilding the mesh along the outlines makes the "
            "cut follow them exactly rather than the nearest cell edges.")
        self._decouple.toggled.connect(self._on_decouple_toggled)
        # A check box cannot wrap its text, so each says what it does on a
        # line of its own underneath.
        for check, text in (
                (self._conform, "Rebuild the mesh so its cell edges run along the "
                                "outlines; it is rebuilt after every zone edit."),
                (self._decouple, "No smoothing across the outlines: the model may "
                                 "jump there. Cut edges are drawn in black.")):
            hint = QLabel(text)
            hint.setWordWrap(True)
            hint.setContentsMargins(22, 0, 0, 2)
            hint.setStyleSheet(f"color:{theme.PALETTE['muted']}; font-size:8pt;")
            layout.addWidget(check)
            layout.addWidget(hint)

        self._engine_note = QLabel()
        self._engine_note.setWordWrap(True)
        layout.addWidget(self._engine_note)
        box.setMinimumWidth(260)
        return box

    # -- public API ----------------------------------------------------------
    def set_preview(self, preview: Optional[Dict[str, Any]], source: str = "") -> None:
        """Show a mesh from ``ert_mesh.mesh_preview``; None clears it."""
        self._preview = preview
        self._source = str(source or "")
        self._update_zone_cells()
        self._sync_zone_controls()
        self._redraw()

    def set_message(self, text: str) -> None:
        """Clear the view and say why there is nothing to draw."""
        self._message = str(text)
        self.set_preview(None)

    def set_status(self, text: str) -> None:
        """A line under the plot, such as "Building the mesh…"; "" restores it."""
        self._info.setText(str(text) or self._summary())

    def preview(self) -> Optional[Dict[str, Any]]:
        return self._preview

    def zones(self) -> List[Dict[str, Any]]:
        """The zones in their canonical form, for a recipe or the assistant."""
        return copy.deepcopy(self._zones)

    def set_zones(self, zones: Optional[Sequence[Dict[str, Any]]]) -> None:
        """Replace the zone list. Raises ``ValueError`` for an invalid zone."""
        self._zones = normalize_zones(zones)
        self._refresh_zones(emit=False)

    def zone_report(self) -> List[Dict[str, Any]]:
        """Each zone with the cells it covers on the mesh shown, or None before one."""
        report = [dict(zone) for zone in self._zones]
        counts = self._zone_counts()
        for entry, cells in zip(report, counts):
            entry["cells"] = cells
        return report

    def set_default_resistivity(self, value: float) -> None:
        """The resistivity a newly drawn zone starts at: the survey's median,
        to three figures - it is a starting point to be edited, not a datum."""
        if value and np.isfinite(value) and value > 0:
            self._default_rho = float(f"{float(value):.3g}")

    def set_engine(self, engine: str, time_lapse: bool) -> None:
        """Which engine will use the zones, for the note under the table."""
        self._engine, self._time_lapse = str(engine or "pyhydro"), bool(time_lapse)
        self._update_engine_note()

    def set_settings_panel(self, widget: QWidget) -> None:
        """Show the page's mesh settings above the zones."""
        self._side.insertWidget(0, widget)

    def conform_to_zones(self) -> bool:
        """Whether the mesh is to be rebuilt along the zone outlines."""
        return self._conform.isChecked()

    def set_conform_to_zones(self, on: bool) -> None:
        self._conform.setChecked(bool(on))

    def decouple_zones(self) -> bool:
        """Whether the smoothness is to stop at the zone outlines."""
        return self._decouple.isChecked()

    def set_decouple_zones(self, on: bool) -> None:
        self._decouple.setChecked(bool(on))

    def set_conform_available(self, available: bool, reason: str = "") -> None:
        """Only a generated mesh can be rebuilt; an imported one is used as is.

        The choice is kept while it is unavailable, so going back to a generated
        mesh restores it, and the page leaves it out of the run meanwhile.
        """
        self._conform.setEnabled(bool(available))
        self._conform.setToolTip(self._conform_help if available or not reason else reason)

    # -- drawing -------------------------------------------------------------
    def _summary(self) -> str:
        preview = self._preview
        if not preview:
            return ""
        parts = [f"{preview['cells']} cells ({preview['para_cells']} inverted, "
                 f"{preview['outer_cells']} outer), {preview['nodes']} nodes"]
        if preview.get("para_extent") is not None:
            # The size of what is inverted and of the whole domain, so the effect
            # of a setting is read here rather than guessed from the picture.
            unit = length_units.current()
            xmin, xmax, zmin, zmax = preview["para_extent"]
            inverted = (f"inverted region {to_display_length(xmax - xmin):.3g} × "
                        f"{to_display_length(zmax - zmin):.3g} {unit}")
            areas = self._para_areas()
            if areas.size:
                # An area takes the length factor twice.
                scale = to_display_length(1.0) ** 2
                inverted += (f", cells {areas.min() * scale:.2g}–"
                             f"{areas.max() * scale:.2g} {unit}²")
            parts.append(inverted)
            fxmin, fxmax, fzmin, fzmax = preview["full_extent"]
            parts.append(f"whole mesh {to_display_length(fxmax - fxmin):.0f} × "
                         f"{to_display_length(fzmax - fzmin):.0f} {unit}")
        edges = int((preview.get("build") or {}).get("zone_outline_edges", 0))
        if edges:
            parts.append(f"follows the zones ({edges} outline segments)")
        if len(preview.get("sensors", ())):
            parts.append(f"{len(preview['sensors'])} electrodes")
        if self._source:
            parts.append(self._source)
        return " · ".join(parts)

    def _para_areas(self) -> np.ndarray:
        """Area of every inverted cell, by the shoelace formula."""
        polygons = (self._preview or {}).get("para_polygons")
        if polygons is None or not len(polygons):
            return np.zeros(0)
        cells = polygons if isinstance(polygons, np.ndarray) else None
        if cells is not None:
            x, z = cells[..., 0], cells[..., 1]
            return 0.5 * np.abs((x * np.roll(z, -1, axis=1)
                                 - np.roll(x, -1, axis=1) * z).sum(axis=1))
        return np.asarray([0.5 * abs(float((c[:, 0] * np.roll(c[:, 1], -1)
                                            - np.roll(c[:, 0], -1) * c[:, 1]).sum()))
                           for c in (np.asarray(p, dtype=float) for p in polygons)])

    def _redraw(self, *_args: Any) -> None:
        from matplotlib.collections import LineCollection, PolyCollection
        from matplotlib.colors import to_rgba
        from matplotlib.patches import Polygon

        if self._selector is not None:
            return  # a redraw would wipe the polygon being drawn
        ax = self._ax
        ax.clear()
        preview = self._preview
        self._info.setText(self._summary())
        if not preview or preview.get("dim") != 2:
            text = self._message
            if preview:
                text = (f"{preview['dim']}-D mesh: {preview['cells']} cells, "
                        f"{preview['para_cells']} inverted. The preview draws 2-D "
                        "meshes only, and zones apply to 2-D meshes.")
            ax.set_axis_off()
            ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True,
                    color=theme.PALETTE["muted"], transform=ax.transAxes)
            self._canvas.draw_idle()
            return

        edges = self._show_edges.isChecked()
        width = 0.3 if edges else 0.0
        outer_on = self._show_outer.isChecked()
        if outer_on and len(preview["outer_polygons"]):
            ax.add_collection(PolyCollection(
                preview["outer_polygons"], facecolors=_OUTER_FACE,
                edgecolors=_OUTER_EDGE if edges else "none", linewidths=width))
        colors = zone_colors(len(self._zones))
        faces = np.tile(to_rgba("white"), (int(preview["para_cells"]), 1))
        if self._owner is not None:
            for index, color in enumerate(colors):
                faces[self._owner == index] = to_rgba(color, 0.45)
        ax.add_collection(PolyCollection(
            preview["para_polygons"], facecolors=faces,
            edgecolors=_PARA_EDGE if edges else "none", linewidths=width))
        if self._decouple.isChecked() and self._cut is not None and len(self._cut):
            # The cell edges the smoothness will not cross: exactly the outlines
            # on a mesh built along them, the nearest cell edges otherwise.
            ax.add_collection(LineCollection(self._cut, colors=_CUT_EDGE,
                                             linewidths=2.0, zorder=3.5))

        selected = {index.row() for index in self._table.selectionModel().selectedRows()}
        for index, (zone, color) in enumerate(zip(self._zones, colors)):
            polygon = np.asarray(zone["polygon"], dtype=float)
            ax.add_patch(Polygon(polygon, closed=True, fill=False, edgecolor=color,
                                 linewidth=3.0 if index in selected else 1.6,
                                 linestyle="-" if zone["fixed"] else "--", zorder=4))
            label = f"{zone['name']}\n{zone['resistivity']:g} Ω·m"
            if zone["fixed"]:
                label += " · fixed"
            centre = polygon.mean(axis=0)
            ax.text(centre[0], centre[1], label, ha="center", va="center", fontsize=8,
                    color=theme.PALETTE["text"], zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.85))

        sensors = np.asarray(preview.get("sensors", np.zeros((0, 2))), dtype=float)
        if len(sensors):
            ax.plot(sensors[:, 0], sensors[:, 1], linestyle="none", marker="v",
                    markersize=5, color=_ELECTRODE, zorder=5, label="Electrodes")

        xmin, xmax, zmin, zmax = preview["full_extent" if outer_on else "para_extent"]
        pad_x = 0.03 * max(xmax - xmin, 1e-9)
        pad_z = 0.05 * max(zmax - zmin, 1e-9)
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(zmin - pad_z, zmax + pad_z)
        ax.set_aspect("equal", adjustable="box")
        # The electrodes are the ground surface; a survey read without
        # elevations has them all at z = 0, and the mesh then reads as depth.
        if len(sensors):
            set_section_axes(ax, surface=sensors[:, 1], xlabel="x")
        else:
            set_section_axes(ax, z=[preview["para_extent"][3]], xlabel="x")
        title = "Inversion mesh"
        if not outer_on:
            title += " (parameter domain)"
        ax.set_title(title)
        self._canvas.draw_idle()

    # -- zone editing --------------------------------------------------------
    def _on_draw_toggled(self, on: bool) -> None:
        from matplotlib.widgets import PolygonSelector

        self._draw_hint.setVisible(on)
        if not on:
            self._stop_drawing()
            return
        # Pan and zoom hold the mouse, and the polygon tool yields to them.
        mode = str(getattr(self._toolbar, "mode", "") or "")
        if mode == "pan/zoom":
            self._toolbar.pan()
        elif mode == "zoom rect":
            self._toolbar.zoom()
        colour = zone_colors(len(self._zones) + 1)[-1]
        self._selector = PolygonSelector(
            self._ax, self._on_polygon, useblit=True,
            props=dict(color=colour, linewidth=2, alpha=0.9),
            handle_props=dict(markeredgecolor=colour, markerfacecolor="white",
                              markersize=6))
        self._canvas.setFocus()

    def _stop_drawing(self) -> None:
        selector, self._selector = self._selector, None
        if selector is not None:
            selector.disconnect_events()
            selector.set_visible(False)
        self._draw_btn.blockSignals(True)
        self._draw_btn.setChecked(False)
        self._draw_btn.blockSignals(False)
        self._draw_hint.setVisible(False)
        self._redraw()

    def _on_polygon(self, vertices) -> None:
        """A polygon was closed: it becomes a zone at the default resistivity.

        Vertices are kept to the millimetre; the rest of a mouse position is the
        screen's pixel grid, not anything the user meant.
        """
        vertices = [[round(float(x), 3), round(float(z), 3)] for x, z in vertices]
        self._stop_drawing()
        if len(vertices) < 3:
            return
        taken = {zone["name"] for zone in self._zones}
        number = len(self._zones) + 1
        while f"Zone {number}" in taken:
            number += 1
        self._zones.append({"name": f"Zone {number}", "polygon": vertices,
                            "resistivity": float(self._default_rho), "fixed": False})
        self._refresh_zones(emit=True)
        self._table.selectRow(len(self._zones) - 1)

    def _remove_selected(self) -> None:
        rows = sorted({index.row() for index in self._table.selectionModel().selectedRows()},
                      reverse=True)
        if not rows:
            return
        for row in rows:
            del self._zones[row]
        self._refresh_zones(emit=True)

    def _clear_zones(self) -> None:
        if self._zones:
            self._zones = []
            self._refresh_zones(emit=True)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if not 0 <= row < len(self._zones):
            return
        zone = self._zones[row]
        if item.column() == _NAME:
            name = item.text().strip()
            if not name:
                self._table.blockSignals(True)
                item.setText(zone["name"])
                self._table.blockSignals(False)
                return
            zone["name"] = name
        elif item.column() == _FIXED:
            zone["fixed"] = item.checkState() == Qt.Checked
            self._update_engine_note()
        else:
            return
        self._redraw()
        self.zonesChanged.emit(self.zones())

    def _on_rho_changed(self, row: int, value: float) -> None:
        if 0 <= row < len(self._zones) and value > 0:
            self._zones[row]["resistivity"] = float(value)
            self._redraw()
            self.zonesChanged.emit(self.zones())

    def _refresh_zones(self, *, emit: bool) -> None:
        """Rebuild the table and the overlay from ``self._zones``."""
        self._update_zone_cells()
        table = self._table
        table.blockSignals(True)
        # Emptied first: a replaced cell widget is only deleted later, and until
        # then it stays on screen where its column used to be. A removed row's
        # widgets are hidden at once.
        table.setRowCount(0)
        table.setRowCount(len(self._zones))
        counts = self._zone_counts()
        for row, zone in enumerate(self._zones):
            name = QTableWidgetItem(zone["name"])
            name.setToolTip("Vertices (x, elevation): " + ", ".join(
                f"({x:g}, {z:g})" for x, z in zone["polygon"]))
            table.setItem(row, _NAME, name)
            rho = QDoubleSpinBox()
            rho.setRange(0.01, 1e7)
            rho.setDecimals(2)
            rho.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            rho.setKeyboardTracking(False)
            rho.setFrame(False)
            rho.setValue(float(zone["resistivity"]))
            rho.valueChanged.connect(lambda value, r=row: self._on_rho_changed(r, value))
            table.setCellWidget(row, _RHO, rho)
            fixed = QTableWidgetItem()
            fixed.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            fixed.setCheckState(Qt.Checked if zone["fixed"] else Qt.Unchecked)
            table.setItem(row, _FIXED, fixed)
            cells = QTableWidgetItem("—" if counts[row] is None else str(counts[row]))
            cells.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            cells.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row, _CELLS, cells)
        table.blockSignals(False)
        self._sync_zone_controls()
        self._update_engine_note()
        self._redraw()
        if emit:
            self.zonesChanged.emit(self.zones())

    def _update_zone_cells(self) -> None:
        """Which zone each parameter cell of the mesh shown belongs to."""
        preview = self._preview
        self._owner = None
        self._cut = None
        self._counts = [None] * len(self._zones)
        if not preview or preview.get("dim") != 2 or not self._zones:
            return
        prior = zone_prior(preview["para_centers"], self._zones)
        self._owner = prior.owner
        self._cut = _owner_edges(preview["para_polygons"], prior.owner)
        self._counts = [int(entry["cells"]) for entry in prior.report]
        for row, count in enumerate(self._counts):
            item = self._table.item(row, _CELLS)
            if item is not None:
                item.setText(str(count))

    def _zone_counts(self) -> List[Optional[int]]:
        counts = list(self._counts)
        return counts + [None] * (len(self._zones) - len(counts))

    def _sync_zone_controls(self) -> None:
        drawable = bool(self._preview) and self._preview.get("dim") == 2
        self._draw_btn.setEnabled(drawable)
        self._remove_btn.setEnabled(bool(self._zones))
        self._clear_btn.setEnabled(bool(self._zones))

    def _update_engine_note(self) -> None:
        if not self._zones:
            self._engine_note.setText("")
            return
        text, warn = engine_zone_note(self._engine, self._time_lapse, self._zones,
                                      decouple=self._decouple.isChecked())
        colour = theme.PALETTE["red"] if warn else theme.PALETTE["muted"]
        self._engine_note.setStyleSheet(f"color:{colour}; font-size:8pt;")
        self._engine_note.setText(text)

    def _on_conform_toggled(self, on: bool) -> None:
        self.conformChanged.emit(bool(on))

    def _on_decouple_toggled(self, on: bool) -> None:
        self._update_engine_note()
        self._redraw()
        self.decoupleChanged.emit(bool(on))


def _owner_edges(polygons, owner: np.ndarray) -> np.ndarray:
    """``(n, 2, 2)`` segments: the cell edges shared by cells of different zones.

    Cells are matched by their corner coordinates, which neighbouring cells of
    one mesh share exactly; ``owner`` is the zone of each cell, -1 for none.
    """
    open_edges: Dict[tuple, int] = {}
    cut: List[np.ndarray] = []
    for index, corners in enumerate(polygons):
        corners = np.asarray(corners, dtype=float)
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            key = tuple(sorted((tuple(np.round(a, 6)), tuple(np.round(b, 6)))))
            other = open_edges.pop(key, None)
            if other is None:
                open_edges[key] = index
            elif owner[other] != owner[index]:
                cut.append(np.array([a, b]))
    return np.asarray(cut, dtype=float).reshape(-1, 2, 2)
