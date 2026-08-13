"""Left navigator tree that emits a module key when an item is selected."""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from PyHydroGeophysX.qt_apps import theme

_MODULE_ROLE = Qt.UserRole

_GROUP_ICONS = {
    "Geophysical Data Processing": "fa5s.wave-square",
    "Hydro → Geophysics": "fa5s.water",
    "Geophy → Hydrology": "fa5s.mountain",
    "Project": "fa5s.folder-open",
}
_CHILD_ICONS = {
    "Seismic": "fa5s.wave-square",
    "ERT": "fa5s.bolt",
    "3D Mesh Builder": "fa5s.cube",
    "EM": "fa5s.broadcast-tower",
    "Gravity / Magnetics": "fa5s.magnet",
    "Joint Inversion": "fa5s.project-diagram",
    "Hydro Data": "fa5s.database",
    "Profile Selection": "fa5s.route",
    "Survey Geometry": "fa5s.ruler-combined",
    "Forward Modeling": "fa5s.play",
    "Results": "fa5s.chart-area",
    "Inverted Model": "fa5s.database",
    "Layers": "fa5s.layer-group",
    "Target": "fa5s.bullseye",
    "Estimation": "fa5s.dice",
    "Seismic Lines": "fa5s.wave-square",
    "Interface": "fa5s.mountain",
    "3D Grid": "fa5s.th",
    "Build Model": "fa5s.cubes",
    "Seismic → Structure": "fa5s.cube",
    "ERT → Water Content": "fa5s.tint",
    "Saved Results": "fa5s.history",
}

# (group label, [(child label, module key), ...]). Several Hydro children map to
# the same module page; that page exposes the corresponding section internally.
TREE_STRUCTURE: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "Geophysical Data Processing",
        [
            ("Seismic", "seismic"),
            ("ERT", "ert"),
            ("3D Mesh Builder", "mesh3d"),
            ("EM", "em"),
            ("Gravity / Magnetics", "gravmag"),
            ("Joint Inversion", "joint_inversion"),
        ],
    ),
    (
        "Hydro → Geophysics",
        [
            ("Hydro Data", "hydro_geophysics"),
            ("Profile Selection", "hydro_geophysics"),
            ("Survey Geometry", "hydro_geophysics"),
            ("Forward Modeling", "hydro_geophysics"),
            ("Results", "hydro_geophysics"),
        ],
    ),
    (
        "Geophy → Hydrology",
        [
            ("Seismic → Structure", "seismic3d"),
            ("ERT → Water Content", "geo_hydrology"),
        ],
    ),
    # Every computation is recorded in the active Project, so the history needs a
    # permanent home in the navigator rather than only a Tools menu entry.
    (
        "Project",
        [
            ("Saved Results", "model_viewer"),
        ],
    ),
]


class ProjectTree(QTreeWidget):
    """A two-group navigator. Emits ``moduleSelected(module_key)`` on click."""

    moduleSelected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setMinimumWidth(220)
        self.setIconSize(QSize(16, 16))
        self.setIndentation(12)
        self._key_to_item = {}
        self._build()
        self.itemClicked.connect(self._on_item_clicked)

    def _build(self) -> None:
        for group_label, children in TREE_STRUCTURE:
            group = QTreeWidgetItem([group_label])
            group.setFlags(group.flags() & ~Qt.ItemIsSelectable)
            font = group.font(0)
            font.setBold(True)
            group.setFont(0, font)
            group.setIcon(0, theme.icon(_GROUP_ICONS.get(group_label, "fa5s.folder"), color=theme.PALETTE["primary"]))
            self.addTopLevelItem(group)
            for child_label, module_key in children:
                child = QTreeWidgetItem([child_label])
                child.setData(0, _MODULE_ROLE, module_key)
                child.setIcon(0, theme.icon(_CHILD_ICONS.get(child_label, "fa5s.circle"), color=theme.PALETTE["accent"]))
                group.addChild(child)
                self._key_to_item.setdefault(module_key, child)
            group.setExpanded(True)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        key = item.data(0, _MODULE_ROLE)
        if key:
            self.moduleSelected.emit(str(key))

    def select_module(self, module_key: str) -> None:
        """Programmatically highlight the first tree item for ``module_key``."""
        item = self._key_to_item.get(module_key)
        if item is not None:
            self.setCurrentItem(item)
