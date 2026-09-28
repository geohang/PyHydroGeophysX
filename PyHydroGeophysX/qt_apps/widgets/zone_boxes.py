"""The 3D Mesh Builder's zones: boxes of known or assumed resistivity.

A zone here is a box in the survey's coordinates - a clay layer, a tank, a
plume - with a resistivity (see ``core.mesh_3d.normalize_box_zones``). The
editor keeps the list: a table of the zones, the box of the one selected, and
the two choices that go with them, the same two the ERT page's Mesh tab offers
for its 2-D zones - whether the mesh is built along the zones' faces, and
whether each zone is a region of its own, which an inversion on the mesh does
not smooth across. The page reads the list when it builds a mesh or runs the 3D
forward model, and tells the editor how many cells each zone took.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.core._mesh_3d_builder import ZONE_MARKER_START, normalize_box_zones
from PyHydroGeophysX.inversion.ert_zones import zone_colors
from PyHydroGeophysX.qt_apps import theme

__all__ = ["ZoneBoxEditor", "clip_box"]

_NAME, _RHO, _CELLS = range(3)
_MUTED = theme.PALETTE["muted"]
#: How far a layer reaches past the survey sideways: beyond any domain the
#: builder makes, so that it spans whatever mesh it is used on.
_LAYER_REACH = 1.0e4
#: Where a new zone goes before the page has said where the survey is: the
#: default surface grid, down to the default investigation depth.
_DEFAULT_REGION = {"x_min": -9.0, "x_max": 54.0, "y_min": -5.0, "y_max": 30.0,
                   "z_bottom": -20.0, "z_top": 0.0}


def clip_box(zone: Mapping[str, Any], region: Mapping[str, float]) -> Optional[List[float]]:
    """``zone``'s box within ``region`` (as ``core.mesh_3d.zone_region`` gives
    it) as ``[x0, x1, y0, y1, z0, z1]``, or None when they do not meet - what a
    view draws, so that a layer reaching far past the survey does not zoom the
    view out to it."""
    limits = (("x", "x_min", "x_max"), ("y", "y_min", "y_max"), ("z", "z_bottom", "z_top"))
    box: List[float] = []
    for axis, low_key, high_key in limits:
        low = max(float(zone[axis][0]), float(region[low_key]))
        high = min(float(zone[axis][1]), float(region[high_key]))
        if high <= low:
            return None
        box += [low, high]
    return box


def _spin() -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-1.0e6, 1.0e6)
    box.setDecimals(2)
    box.setSingleStep(0.5)
    box.setSuffix(" m")
    box.setKeyboardTracking(False)
    return box


def _is_layer(zone: Mapping[str, Any]) -> bool:
    return min(zone["x"][1] - zone["x"][0], zone["y"][1] - zone["y"][0]) >= _LAYER_REACH


class ZoneBoxEditor(QGroupBox):
    """The zone list, the box of the selected zone, and the two zone options.

    ``zonesChanged`` carries the zone list after every edit made here;
    :meth:`set_zones` does not emit it. ``optionsChanged`` follows the two
    check boxes.
    """

    zonesChanged = Signal(list)
    optionsChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Zones", parent)
        self._zones: List[Dict[str, Any]] = []
        self._cells: List[Optional[int]] = []
        self._region_provider: Optional[Callable[[], Optional[Mapping[str, float]]]] = None
        self._editing = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Bodies known or assumed in the ground - a clay layer, a tank, a plume - "
            "as boxes. The mesh can follow their faces, each can be a region of its "
            "own, and the 3D forward model gives each its resistivity.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color:{_MUTED}; font-size:8pt;")
        layout.addWidget(intro)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Zone", "ρ (Ω·m)", "Cells"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_NAME, QHeaderView.Stretch)
        for column in (_RHO, _CELLS):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        self._table.setToolTip(
            "Double-click a name to rename the zone. Cells counts the inverted cells "
            "the zone took in the last mesh built; where zones overlap, the one lower "
            "in the list takes the cell.")
        self._table.setMaximumHeight(150)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.itemSelectionChanged.connect(self._show_selected)
        layout.addWidget(self._table)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        add = QPushButton("Add zone")
        add.setIcon(theme.icon("fa5s.cube"))
        add.setToolTip("A box under the middle of the survey, in the upper half of the "
                       "inverted region; set its extent below.")
        add.clicked.connect(self._add_zone)
        layer = QPushButton("Add layer")
        layer.setIcon(theme.icon("fa5s.layer-group"))
        layer.setToolTip(
            "A zone spanning the whole mesh sideways, from the ground (or the bottom "
            "of the layer added before it) down a third of the inverted region. It "
            "goes in the list after the layers and before the other zones, so that "
            "a zone inside it keeps its cells.")
        layer.clicked.connect(self._add_layer)
        self._remove = QPushButton("Remove")
        self._remove.setIcon(theme.icon("fa5s.trash-alt"))
        self._remove.setToolTip("Remove the selected zone.")
        self._remove.clicked.connect(self._remove_selected)
        self._clear = QPushButton("Clear")
        self._clear.setToolTip("Remove every zone.")
        self._clear.clicked.connect(self._clear_zones)
        for button in (add, layer, self._remove, self._clear):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        # The box of the selected zone.
        self._box = QWidget()
        form = QFormLayout(self._box)
        form.setContentsMargins(0, 0, 0, 0)
        self._ranges: Dict[str, Any] = {}
        for axis, label, tip in (
                ("x", "x from", "Extent along x (m)."),
                ("y", "y from", "Extent along y (m)."),
                ("z", "z from", "Elevation of the bottom and of the top (m). A top at "
                                "or above the ground takes the zone up to the ground.")):
            low, high = _spin(), _spin()
            for spin in (low, high):
                spin.setToolTip(tip)
                spin.valueChanged.connect(self._on_box_edited)
            row = QWidget()
            line = QHBoxLayout(row)
            line.setContentsMargins(0, 0, 0, 0)
            line.addWidget(low, 1)
            line.addWidget(QLabel("to"))
            line.addWidget(high, 1)
            form.addRow(label, row)
            self._ranges[axis] = (low, high)
        layout.addWidget(self._box)

        # What the zones do to the mesh, beyond their values. Both are off by
        # default: a mesh built with zones is then the mesh built without them.
        self._conform = QCheckBox("Mesh follows the zones")
        self._conform.setToolTip(
            "Build the mesh so that every zone face is made of cell faces, the way "
            "E4D meshes its zones: no cell straddles a face, so a zone's resistivity "
            "covers exactly the box. E4D meshes each zone as a zone of its own, the "
            "structured grid puts a grid line on every face, the prism mesh adds the "
            "outlines to its plan triangulation and a layer at every top and bottom, "
            "and Gmsh cuts the boxes into its domain. Generate the mesh again after "
            "moving a zone. Without it a zone takes the cells whose centre lies "
            "inside it.")
        self._decouple = QCheckBox("Sharp zone edges")
        self._decouple.setToolTip(
            f"Make each zone a region of its own - markers {ZONE_MARKER_START}, "
            f"{ZONE_MARKER_START + 1}, ... in the order of the list - so that an inversion "
            "of this mesh in PyGIMLi may put a sharp contrast at its faces: PyGIMLi puts "
            "no smoothness constraint between regions. Only inverted cells are zoned; "
            "the outer region keeps its marker. Off, the zones stay in the inverted "
            "region (marker 2) and are smoothed across like the rest.")
        # A check box cannot wrap its text, so each says what it does on a line
        # of its own underneath.
        for check, text in (
                (self._conform, "Cell faces along every zone face; generate the mesh "
                                "again after moving a zone."),
                (self._decouple, "Each zone a region of its own, so an inversion does "
                                 "not smooth across its faces.")):
            check.toggled.connect(lambda _on: self.optionsChanged.emit())
            hint = QLabel(text)
            hint.setWordWrap(True)
            hint.setContentsMargins(22, 0, 0, 2)
            hint.setStyleSheet(f"color:{_MUTED}; font-size:8pt;")
            layout.addWidget(check)
            layout.addWidget(hint)

        self._note = QLabel()
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self.set_note("")
        self._refresh()

    # -- public API ----------------------------------------------------------
    def zones(self) -> List[Dict[str, Any]]:
        """The zones in their canonical form, for a config or the assistant."""
        return copy.deepcopy(self._zones)

    def set_zones(self, zones: Optional[Sequence[Dict[str, Any]]]) -> None:
        """Replace the list. Raises ``ValueError`` for an invalid zone."""
        self._zones = normalize_box_zones(zones)
        self._cells = [None] * len(self._zones)
        self._refresh()

    def zone_report(self) -> List[Dict[str, Any]]:
        """Each zone with the cells it took in the last mesh, None before one."""
        return [dict(zone, cells=cells) for zone, cells in zip(self._zones, self._cells)]

    def conform_to_zones(self) -> bool:
        return self._conform.isChecked()

    def set_conform_to_zones(self, on: bool) -> None:
        self._conform.setChecked(bool(on))

    def decouple_zones(self) -> bool:
        return self._decouple.isChecked()

    def set_decouple_zones(self, on: bool) -> None:
        self._decouple.setChecked(bool(on))

    def set_region_provider(self, provider: Callable[[], Optional[Mapping[str, float]]]) -> None:
        """Where a new zone goes: ``provider`` returns the region zones act in
        (``core.mesh_3d.zone_region``) for the current settings, None when the
        settings do not make one."""
        self._region_provider = provider

    def set_cells(self, report: Optional[Sequence[Dict[str, Any]]]) -> None:
        """The cells each zone took in the mesh just built (``generate_mesh``'s
        ``zones``, in the order of the list)."""
        counts = [int(entry.get("cells", 0)) for entry in (report or [])]
        self._cells = (counts + [None] * len(self._zones))[:len(self._zones)]
        self._refresh(keep_selection=True)

    def set_note(self, text: str, warn: bool = False) -> None:
        """What the mesh engine does with the zones, under the options."""
        colour = theme.PALETTE["red"] if warn else _MUTED
        self._note.setStyleSheet(f"color:{colour}; font-size:8pt;")
        self._note.setText(text)
        self._note.setVisible(bool(text))

    def note(self) -> str:
        return self._note.text()

    # -- editing -------------------------------------------------------------
    def _region(self) -> Mapping[str, float]:
        region = None
        if self._region_provider is not None:
            try:
                region = self._region_provider()
            except Exception:  # noqa: BLE001 - settings mid-edit; place it anyway
                region = None
        return region or _DEFAULT_REGION

    def _next_name(self, stem: str) -> str:
        taken = {zone["name"] for zone in self._zones}
        number = 1
        while f"{stem} {number}" in taken:
            number += 1
        return f"{stem} {number}"

    def _insert(self, zone: Dict[str, Any], position: Optional[int] = None) -> None:
        position = len(self._zones) if position is None else position
        self._zones.insert(position, normalize_box_zones([zone])[0])
        self._cells.insert(position, None)
        self._refresh()
        self._table.selectRow(position)
        self.zonesChanged.emit(self.zones())

    def _add_zone(self) -> None:
        r = self._region()
        cx, cy = 0.5 * (r["x_min"] + r["x_max"]), 0.5 * (r["y_min"] + r["y_max"])
        hx = max(0.2 * (r["x_max"] - r["x_min"]), 0.5)
        hy = max(0.2 * (r["y_max"] - r["y_min"]), 0.5)
        top, bottom = r["z_top"], r["z_bottom"]
        depth = max(top - bottom, 1.0)
        z = [top - 0.6 * depth, top - 0.2 * depth]
        # With layers, in the middle of the thickest band between their faces:
        # a box cut by a layer face is one E4D cannot mesh.
        faces = sorted({min(max(value, bottom), top) for zone in self._zones
                        if _is_layer(zone) for value in zone["z"]} | {bottom, top})
        if len(faces) > 2:
            low, high = max(zip(faces, faces[1:]), key=lambda band: band[1] - band[0])
            z = [low + 0.25 * (high - low), high - 0.25 * (high - low)]
        self._insert({"name": self._next_name("Zone"), "x": [cx - hx, cx + hx],
                      "y": [cy - hy, cy + hy], "z": z, "resistivity": 20.0})

    def _add_layer(self) -> None:
        r = self._region()
        depth = max(r["z_top"] - r["z_bottom"], 1.0)
        layers = [index for index, zone in enumerate(self._zones) if _is_layer(zone)]
        # The first from above the ground, which takes it up to the ground;
        # the next under the lowest layer there is.
        top = (min(self._zones[index]["z"][0] for index in layers) if layers
               else r["z_top"] + 1.0)
        bottom = (top if layers else r["z_top"]) - depth / 3.0
        # Below any box its bottom would cut through, for the same reason.
        boxes = [zone["z"] for zone in self._zones if not _is_layer(zone)]
        for _ in boxes:
            bottom = min([bottom] + [low for low, high in boxes if low < bottom < high])
        # After the layers and before the other zones: where zones overlap the
        # later one takes the cells, so a zone inside the layer keeps its own.
        self._insert({"name": self._next_name("Layer"),
                      "x": [r["x_min"] - _LAYER_REACH, r["x_max"] + _LAYER_REACH],
                      "y": [r["y_min"] - _LAYER_REACH, r["y_max"] + _LAYER_REACH],
                      "z": [bottom, top], "resistivity": 100.0},
                     max(layers, default=-1) + 1)

    def _remove_selected(self) -> None:
        row = self._selected_row()
        if row is None:
            return
        del self._zones[row]
        del self._cells[row]
        self._refresh()
        if self._zones:
            self._table.selectRow(min(row, len(self._zones) - 1))
        self.zonesChanged.emit(self.zones())

    def _clear_zones(self) -> None:
        if self._zones:
            self._zones, self._cells = [], []
            self._refresh()
            self.zonesChanged.emit(self.zones())

    def _selected_row(self) -> Optional[int]:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if item.column() != _NAME or not 0 <= row < len(self._zones):
            return
        name = item.text().strip()
        if not name:
            self._table.blockSignals(True)
            item.setText(self._zones[row]["name"])
            self._table.blockSignals(False)
            return
        self._zones[row]["name"] = name
        self.zonesChanged.emit(self.zones())

    def _on_rho_changed(self, row: int, value: float) -> None:
        if 0 <= row < len(self._zones) and value > 0:
            self._zones[row]["resistivity"] = float(value)
            self.zonesChanged.emit(self.zones())

    def _on_box_edited(self, *_args: Any) -> None:
        row = self._selected_row()
        if self._editing or row is None:
            return
        zone = dict(self._zones[row])
        for axis, (low, high) in self._ranges.items():
            zone[axis] = [low.value(), high.value()]
        try:
            self._zones[row] = normalize_box_zones([zone])[0]
        except ValueError:
            return                                   # a range without extent, mid-edit
        self._cells[row] = None                      # counted on the old box
        self._refresh(keep_selection=True)
        self.zonesChanged.emit(self.zones())

    def _show_selected(self) -> None:
        row = self._selected_row()
        self._box.setEnabled(row is not None)
        self._remove.setEnabled(row is not None)
        if row is None:
            return
        self._editing = True
        try:
            for axis, (low, high) in self._ranges.items():
                low.setValue(self._zones[row][axis][0])
                high.setValue(self._zones[row][axis][1])
        finally:
            self._editing = False

    def _refresh(self, keep_selection: bool = False) -> None:
        row = self._selected_row() if keep_selection else None
        table = self._table
        table.blockSignals(True)
        # Emptied first: a replaced cell widget is only deleted later, and until
        # then it stays on screen where its column used to be.
        table.setRowCount(0)
        table.setRowCount(len(self._zones))
        for index, (zone, colour) in enumerate(zip(self._zones, zone_colors(len(self._zones)))):
            swatch = QPixmap(10, 10)
            swatch.fill(QColor(colour))
            name = QTableWidgetItem(QIcon(swatch), zone["name"])
            (x0, x1), (y0, y1), (z0, z1) = zone["x"], zone["y"], zone["z"]
            name.setToolTip(f"Across the whole mesh, z {z0:g} to {z1:g} m" if _is_layer(zone)
                            else f"x {x0:g} to {x1:g}, y {y0:g} to {y1:g}, z {z0:g} to {z1:g} m")
            table.setItem(index, _NAME, name)
            rho = QDoubleSpinBox()
            rho.setRange(0.01, 1e7)
            rho.setDecimals(2)
            rho.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
            rho.setKeyboardTracking(False)
            rho.setFrame(False)
            rho.setValue(float(zone["resistivity"]))
            rho.valueChanged.connect(lambda value, r=index: self._on_rho_changed(r, value))
            table.setCellWidget(index, _RHO, rho)
            count = self._cells[index] if index < len(self._cells) else None
            cells = QTableWidgetItem("—" if count is None else str(count))
            cells.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            cells.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if count == 0:
                cells.setForeground(QColor(theme.PALETTE["red"]))
                cells.setToolTip("This zone took no cell in the last mesh: it lies outside "
                                 "the inverted region, or zones lower in the list cover it.")
            table.setItem(index, _CELLS, cells)
        table.blockSignals(False)
        self._clear.setEnabled(bool(self._zones))
        if row is not None and row < len(self._zones):
            table.selectRow(row)
        self._show_selected()
