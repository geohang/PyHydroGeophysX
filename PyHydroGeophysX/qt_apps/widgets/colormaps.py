"""One colormap chooser for every colour-mapped view in the studio.

A section, a map or a volume is read through its colours, and the right map
depends on the quantity and on the reader: a diverging map for a signed change,
a perceptually uniform one for print, the colours a collaborator's figures
already use. So every view that draws a colorbar or a colour-mapped image
offers the same compact chooser beside its display controls: a gradient swatch
that opens a list of curated maps, and a toggle that reverses the one chosen.

The choice is remembered for the session under a key that names what is shown
- resistivity sections, time-lapse change, velocity, EM sections, and so on -
in a dict the studio state owns (``StudioState.colormap_settings``). Every view
of that quantity reads the same entry, so the ERT page and a result reopened in
Saved Results show a section in the same colours, and a change made on either
page is what the other shows. Nothing is stored until the user picks, and a view
with no stored choice draws exactly what it drew before this chooser existed:
each view passes its own former colormap as the default.

The helpers below turn a chosen name into what each plotting library takes -
matplotlib and PyGIMLi (a colormap name, or a Colormap object for the custom
seismic map), pyqtgraph (a ``ColorMap`` or a lookup table) and PyVista (a name,
or a Colormap object) - so a view never has to know where a map came from.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QWidget,
)

from PyHydroGeophysX.qt_apps import theme

__all__ = [
    "APPARENT_RESISTIVITY", "ARRAY", "COVERAGE", "DENSITY", "DIVERGING",
    "EM_SECTION", "EM_STATIONS", "GATHER", "GRAVMAG", "HYDRO", "MAP_LAYER",
    "MESH_REGIONS", "MODEL3D", "RESISTIVITY", "RESISTIVITY_CHANGE",
    "SEISMIC_GATHER", "SEQUENTIAL", "SUSCEPTIBILITY", "VELOCITY",
    "ColormapChooser", "colormap_entries", "colormap_settings", "gradient_icon",
    "is_reversed", "lookup_table", "matplotlib_colormap", "reversed_name",
    "to_matplotlib", "to_pyqtgraph", "to_pyvista", "apply_to_histogram",
]

# -- the curated maps -----------------------------------------------------------

#: Maps for a quantity read from low to high. ``Spectral_r`` is the ERT
#: convention and stays in this group, where the resistivity views look for it.
SEQUENTIAL: Tuple[str, ...] = (
    "viridis", "plasma", "inferno", "magma", "cividis", "turbo",
    "Spectral_r", "jet", "gist_earth", "terrain", "gray",
)
#: Maps for a signed quantity read either side of a centre value.
DIVERGING: Tuple[str, ...] = ("RdBu_r", "coolwarm", "seismic", "bwr", "PuOr", "BrBG")

#: The seismic gather's own blue-white-red: nearly white over the middle 4 % so
#: the zero crossings stay quiet and only the arrivals carry colour. It is not a
#: matplotlib map, so it is built here, and offered by the view that uses it.
GATHER = "phgx_gather"
GATHER_STOPS = np.array([0.0, 0.48, 0.5, 0.52, 1.0])
GATHER_COLORS = np.array([
    [8, 48, 107, 255],
    [247, 251, 255, 255],
    [255, 255, 255, 255],
    [255, 245, 240, 255],
    [153, 0, 13, 255],
], dtype=np.ubyte)

#: Maps defined here rather than by matplotlib: name -> (stops, RGBA bytes, label).
_CUSTOM: Dict[str, Tuple[np.ndarray, np.ndarray, str]] = {
    GATHER: (GATHER_STOPS, GATHER_COLORS, "gather blue–white–red"),
}

# -- what is shown: the keys a choice is remembered under -----------------------

RESISTIVITY = "resistivity"                  #: ERT resistivity sections
RESISTIVITY_CHANGE = "resistivity_change"    #: time-lapse % change from the baseline
COVERAGE = "coverage"                        #: ERT coverage (log10 sensitivity)
APPARENT_RESISTIVITY = "apparent_resistivity"  #: the ERT QC pseudosection
VELOCITY = "velocity"                        #: seismic velocity sections and volumes
SEISMIC_GATHER = "seismic_gather"            #: a shot gather's amplitude image
EM_SECTION = "em_section"                    #: EM resistivity: sections, volume, slices
EM_STATIONS = "em_stations"                  #: the EM survey QC station map
GRAVMAG = "gravmag"                          #: gravity / magnetic anomaly maps
DENSITY = "density"                          #: gravity density-contrast models
SUSCEPTIBILITY = "susceptibility"            #: magnetic susceptibility models
MAP_LAYER = "map_layer"                      #: Project Map plan layers
HYDRO = "hydro_model"                        #: hydrological model maps
ARRAY = "array"                              #: a saved 2-D array
MODEL3D = "model3d"                          #: a 3-D volume read from a file
MESH_REGIONS = "mesh_regions"                #: mesh region markers

#: How a tooltip names each key: "resistivity sections", ...
_WHAT: Dict[str, str] = {
    RESISTIVITY: "ERT resistivity sections",
    RESISTIVITY_CHANGE: "time-lapse change sections",
    COVERAGE: "coverage (sensitivity) sections",
    APPARENT_RESISTIVITY: "the apparent-resistivity pseudosection",
    VELOCITY: "seismic velocity models",
    SEISMIC_GATHER: "shot gathers",
    EM_SECTION: "EM resistivity sections, volumes and depth slices",
    EM_STATIONS: "the EM station map",
    GRAVMAG: "gravity and magnetic anomaly maps",
    DENSITY: "density-contrast models",
    SUSCEPTIBILITY: "susceptibility models",
    MAP_LAYER: "Project Map layers",
    HYDRO: "hydrological model maps",
    ARRAY: "saved arrays",
    MODEL3D: "3-D volumes",
    MESH_REGIONS: "mesh regions",
}


def colormap_settings(state: Any) -> Optional[MutableMapping[str, str]]:
    """The session's colormap choices held by ``state``, or None without one.

    What a page passes to the views it builds, the way it passes the
    temperature-correction settings: one dict for the whole studio.
    """
    return getattr(state, "colormap_settings", None)


# -- names ----------------------------------------------------------------------

def is_reversed(name: str) -> bool:
    return str(name).endswith("_r")


def reversed_name(name: str) -> str:
    """``viridis`` <-> ``viridis_r``; matplotlib's own convention, custom maps too."""
    name = str(name)
    return name[:-2] if name.endswith("_r") else f"{name}_r"


def _custom(name: str) -> Optional[Tuple[np.ndarray, np.ndarray, bool]]:
    """``(stops, colors, reversed)`` for a map defined here, else None."""
    name = str(name)
    if name in _CUSTOM:
        stops, colors, _label = _CUSTOM[name]
        return stops, colors, False
    base = reversed_name(name) if is_reversed(name) else None
    if base in _CUSTOM:
        stops, colors, _label = _CUSTOM[base]
        return stops, colors, True
    return None


def display_name(name: str) -> str:
    """What the list shows for ``name``: matplotlib's own name, or a custom label."""
    name = str(name)
    if name in _CUSTOM:
        return _CUSTOM[name][2]
    if is_reversed(name) and reversed_name(name) in _CUSTOM:
        return f"{_CUSTOM[reversed_name(name)][2]} (reversed)"
    return name


def colormap_entries(default: str = "") -> List[Tuple[str, Tuple[str, ...]]]:
    """The chooser's groups, ``[(heading, names), ...]``.

    The curated maps, plus the view's own default when it is not among them - so
    whatever a view drew before stays one click away.
    """
    groups: List[Tuple[str, Tuple[str, ...]]] = [
        ("Sequential", SEQUENTIAL), ("Diverging", DIVERGING)]
    curated = set(SEQUENTIAL) | set(DIVERGING)
    base = str(default or "")
    if base and base not in curated and reversed_name(base) in curated:
        return groups                      # a reversed curated map: the toggle reaches it
    if base and base not in curated:
        if is_reversed(base) and _custom(base) is not None:
            base = reversed_name(base)
        heading = "Seismic" if base == GATHER else "This view"
        groups.insert(0, (heading, (base,)))
    return groups


# -- conversions ----------------------------------------------------------------

def matplotlib_colormap(name: Union[str, Any]):
    """A matplotlib ``Colormap`` for ``name`` (or ``name`` itself if it is one)."""
    from matplotlib import colormaps
    from matplotlib.colors import Colormap, LinearSegmentedColormap

    if isinstance(name, Colormap):
        return name
    custom = _custom(str(name))
    if custom is not None:
        stops, colors, flip = custom
        base = LinearSegmentedColormap.from_list(
            reversed_name(str(name)) if flip else str(name),
            list(zip(stops.tolist(), (colors / 255.0).tolist())), N=256)
        return base.reversed(name=str(name)) if flip else base
    return colormaps[str(name)]


def to_matplotlib(name: str):
    """What matplotlib and PyGIMLi take as ``cmap``/``cMap``.

    The name itself whenever matplotlib knows it - so a view's default is drawn
    through exactly the call it always made - and a Colormap object otherwise.
    """
    from matplotlib import colormaps

    name = str(name)
    if name in colormaps:
        return name
    return matplotlib_colormap(name)


#: PyVista takes the same as matplotlib.
to_pyvista = to_matplotlib


def to_pyqtgraph(name: str):
    """A pyqtgraph ``ColorMap`` for ``name``.

    The exact stops for a map defined here, and matplotlib's definition for
    everything else - so a pyqtgraph view and a matplotlib view of the same map
    show the same colours. The one exception is ``viridis``: several views drew
    it with ``pg.colormap.get("viridis")`` before, which is matplotlib's map to
    the byte, and they keep getting exactly that object. (pyqtgraph's bundled
    ``cividis``, by contrast, is an older variant, so it is not used.)
    """
    import pyqtgraph as pg

    name = str(name)
    custom = _custom(name)
    if custom is not None:
        stops, colors, flip = custom
        if flip:
            return pg.ColorMap(1.0 - stops[::-1], colors[::-1])
        return pg.ColorMap(stops, colors)
    if name == "viridis":
        try:
            return pg.colormap.get("viridis")
        except Exception:  # noqa: BLE001 - fall through to matplotlib's definition
            pass
    try:
        cmap = pg.colormap.getFromMatplotlib(name)
        if cmap is not None:
            return cmap
    except Exception:  # noqa: BLE001 - sample it instead
        pass
    samples = np.linspace(0.0, 1.0, 256)
    rgba = np.round(matplotlib_colormap(name)(samples) * 255.0).astype(np.ubyte)
    return pg.ColorMap(samples, rgba)


def lookup_table(name: str, n: int = 256) -> np.ndarray:
    """``n`` colours of ``name`` as bytes, as a pyqtgraph ImageItem takes them.

    ``(n, 3)`` RGB for an opaque map - which all of these are - and ``(n, 4)``
    only for one with transparency, exactly as ``ColorMap.getLookupTable`` has
    always returned them to the views.
    """
    return to_pyqtgraph(name).getLookupTable(0.0, 1.0, int(n))


def apply_to_histogram(histogram: Any, name: str, *, preset: Optional[str] = None) -> None:
    """Colour a pyqtgraph ``HistogramLUTItem`` (and the image it drives).

    ``preset`` is the gradient preset the view used before this chooser: loaded
    as such, it looks exactly as it did, ticks and all. Any other map is set in
    full and its ticks hidden, as pyqtgraph's own colormap menu does - a 256-stop
    map would otherwise stripe the gradient bar with 256 handles.
    """
    gradient = histogram.gradient
    if preset:
        try:
            gradient.loadPreset(preset)
            gradient.showTicks(True)
            return
        except Exception:  # noqa: BLE001 - not a preset after all: set it in full
            pass
    gradient.setColorMap(to_pyqtgraph(name))
    gradient.showTicks(False)


# -- swatches -------------------------------------------------------------------

_ICONS: Dict[Tuple[str, int, int], QIcon] = {}


def gradient_pixmap(name: str, width: int = 56, height: int = 12) -> QPixmap:
    """A horizontal swatch of ``name``, framed, for the chooser and its list."""
    width, height = max(int(width), 2), max(int(height), 2)
    lut = lookup_table(name, width)[:, :3]
    rows = np.ascontiguousarray(np.repeat(lut[None, :, :], height, axis=0), dtype=np.ubyte)
    image = QImage(rows.data, width, height, 3 * width, QImage.Format_RGB888).copy()
    pixmap = QPixmap.fromImage(image)
    painter = QPainter(pixmap)
    painter.setPen(QColor("#8b949e"))
    painter.drawRect(0, 0, width - 1, height - 1)
    painter.end()
    return pixmap


def gradient_icon(name: str, width: int = 56, height: int = 12) -> QIcon:
    key = (str(name), int(width), int(height))
    icon = _ICONS.get(key)
    if icon is None:
        icon = QIcon(gradient_pixmap(name, width, height))
        _ICONS[key] = icon
    return icon


# -- the chooser ----------------------------------------------------------------

class _ChoiceHub(QObject):
    """Tells every chooser on the same shared dict that an entry changed.

    Two views of one quantity can be on screen at once - an EM section and the
    EM layer on the Project Map, a hydrological map and its profile preview - and
    the one that was not clicked should not wait to be hidden and shown again.
    """

    changed = Signal(object, str, str)   # shared dict, key, name


_HUB: Optional[_ChoiceHub] = None


def _hub() -> _ChoiceHub:
    global _HUB
    if _HUB is None:
        _HUB = _ChoiceHub()
    return _HUB


class _SwatchCombo(QComboBox):
    """A combo whose closed face is only the swatch of the map in use.

    A name beside the swatch would double the width of a control that sits in
    rows which are already full; the list it opens names every map.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._swatch: Optional[QIcon] = None

    def set_swatch(self, icon: QIcon) -> None:
        self._swatch = icon
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        # Room for the swatch and the arrow, measured through the style so the
        # studio's padding is counted; Qt's own hint drops the icon once no
        # characters are asked for.
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        icon = self.iconSize()
        contents = QSize(icon.width() + 4, max(icon.height(), self.fontMetrics().height()))
        return self.style().sizeFromContents(QStyle.CT_ComboBox, option, contents, self)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt override
        return self.sizeHint()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt override
        painter = QStylePainter(self)
        painter.setPen(self.palette().color(QPalette.Text))
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        painter.drawComplexControl(QStyle.CC_ComboBox, option)
        option.currentText = ""
        if self._swatch is not None:
            option.currentIcon = self._swatch
        painter.drawControl(QStyle.CE_ComboBoxLabel, option)


class ColormapChooser(QWidget):
    """A colormap picker: a gradient swatch with a list of maps, and a Reverse toggle.

    ``key`` names what the owning view shows and ``default`` is the map it drew
    before - shown until the user picks. ``shared`` is the dict every page
    passes, the studio state's, so the same quantity keeps one choice across
    pages; without it the chooser keeps its choices to itself.

    The owning view connects :attr:`colormapChanged` to its redraw and reads
    :meth:`colormap` when it draws. A view that shows different quantities in
    turn calls :meth:`set_target` whenever it switches, which also returns the
    map to draw the new quantity with.
    """

    #: The full map name - ``viridis``, ``viridis_r`` - after the user picked one,
    #: or after another page changed the shared choice while this one was hidden.
    colormapChanged = Signal(str)

    def __init__(self, key: str, default: str, *,
                 shared: Optional[MutableMapping[str, str]] = None,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._shared: MutableMapping[str, str] = shared if shared is not None else {}
        self._key = str(key)
        self._default = str(default)
        self._current = ""
        self._groups: List[Tuple[str, Tuple[str, ...]]] = []
        self._filling = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._combo = _SwatchCombo()
        self._combo.setIconSize(QSize(56, 12))
        self._combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._combo.setMinimumContentsLength(0)
        self._combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_user_change)
        layout.addWidget(self._combo)

        self._reverse = QToolButton()
        self._reverse.setCheckable(True)
        self._reverse.setAutoRaise(True)
        icon = theme.icon("fa5s.exchange-alt")
        if icon.isNull():
            self._reverse.setText("⇄")
        else:
            self._reverse.setIcon(icon)
            self._reverse.setIconSize(QSize(14, 14))
        self._reverse.setToolTip(
            "Reverse the colour map, so the colours run the other way along the "
            "colour bar.")
        self._reverse.toggled.connect(self._on_user_change)
        layout.addWidget(self._reverse)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        _hub().changed.connect(self._on_shared_changed)
        self._populate(self._default)
        self._select(self._wanted())

    # -- what the view asks --------------------------------------------------

    def key(self) -> str:
        return self._key

    def default(self) -> str:
        return self._default

    def colormap(self) -> str:
        """The map to draw with: the stored choice for the key, or the default."""
        return self._current

    def choice(self) -> Optional[str]:
        """What the user chose for the key, or None when nothing has been chosen.

        For a second view of the same quantity whose own default differs: it
        follows a choice, and keeps its default until there is one.
        """
        value = self._shared.get(self._key)
        return str(value) if value else None

    def set_target(self, key: str, default: str) -> str:
        """Point the chooser at what the view shows now; return the map to use.

        Emits nothing: the caller is about to draw anyway, with what this
        returns. The shared entry is read fresh, so a choice made on another
        page is what the next redraw here uses.
        """
        key, default = str(key), str(default)
        if (key, default) != (self._key, self._default):
            self._key, self._default = key, default
            self._populate(default)
            self._update_tooltip()
        self._select(self._wanted())
        return self._current

    def set_colormap(self, name: str, *, remember: bool = True) -> None:
        """Choose ``name`` as though the user had, telling the view and the page."""
        name = str(name)
        self._select(name)
        if remember:
            self._remember(name)
        self.colormapChanged.emit(self._current)

    def resync(self) -> bool:
        """Adopt the shared choice if it changed elsewhere; True when it did."""
        wanted = self._wanted()
        if wanted == self._current:
            return False
        self._select(wanted)
        self.colormapChanged.emit(self._current)
        return True

    # -- internals -----------------------------------------------------------

    def _wanted(self) -> str:
        value = self._shared.get(self._key)
        return str(value) if value else self._default

    def _names(self) -> List[str]:
        return [name for _heading, names in self._groups for name in names]

    def _populate(self, default: str) -> None:
        groups = colormap_entries(default)
        if groups == self._groups:
            return
        self._groups = groups
        self._filling = True
        try:
            self._combo.clear()
            model = self._combo.model()
            longest = 0
            metrics = self._combo.fontMetrics()
            for heading, names in groups:
                self._combo.addItem(heading)
                item = model.item(self._combo.count() - 1)
                item.setFlags(item.flags() & ~(Qt.ItemIsSelectable | Qt.ItemIsEnabled))
                font = QFont(self._combo.font())
                font.setBold(True)
                item.setFont(font)
                for name in names:
                    label = display_name(name)
                    self._combo.addItem(gradient_icon(name), label, name)
                    longest = max(longest, metrics.horizontalAdvance(label))
            # The closed combo is only as wide as a swatch; the list it opens
            # must still fit a swatch and a name side by side.
            self._combo.view().setMinimumWidth(56 + longest + 48)
        finally:
            self._filling = False

    def _split(self, name: str) -> Tuple[str, bool]:
        """``(entry, reversed)`` that together give ``name``."""
        names = self._names()
        if name in names:
            return name, False
        if reversed_name(name) in names:
            return reversed_name(name), True
        return "", False

    def _select(self, name: str) -> None:
        name = str(name)
        entry, flip = self._split(name)
        if not entry:
            # A name this list does not carry - an old choice, a typo in a call.
            # Fall back to what the view drew before rather than to nothing.
            entry, flip = self._split(self._default)
        if not entry:
            entry, flip = self._names()[0], False
        self._filling = True
        try:
            index = self._combo.findData(entry)
            if index >= 0:
                self._combo.setCurrentIndex(index)
            self._reverse.setChecked(flip)
        finally:
            self._filling = False
        self._current = reversed_name(entry) if flip else entry
        self._combo.set_swatch(gradient_icon(self._current))
        self._update_tooltip()

    def _compose(self) -> str:
        entry = self._combo.currentData()
        if not entry:
            return self._current or self._default
        return reversed_name(str(entry)) if self._reverse.isChecked() else str(entry)

    def _remember(self, name: str) -> None:
        self._shared[self._key] = name
        _hub().changed.emit(self._shared, self._key, name)

    def _on_user_change(self, *_args: Any) -> None:
        if self._filling:
            return
        name = self._compose()
        if name == self._current:
            return
        self._current = name
        self._combo.set_swatch(gradient_icon(name))
        self._update_tooltip()
        self._remember(name)
        self.colormapChanged.emit(name)

    def _on_shared_changed(self, shared: Any, key: str, _name: str) -> None:
        # Only the choosers on the same dict and key; a hidden one catches up
        # when it is shown, rather than redrawing a view nobody is looking at.
        if shared is not self._shared or key != self._key:
            return
        if self.isVisible():
            self.resync()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        # The other page may have changed the shared choice while this one was
        # hidden; coming back shows it rather than a stale copy.
        super().showEvent(event)
        self.resync()

    def _update_tooltip(self) -> None:
        what = _WHAT.get(self._key, "this view")
        shown = display_name(self._current)
        default = " (this view's default)" if self._current == self._default else ""
        self._combo.setToolTip(
            f"Colour map for {what}: {shown}{default}.\n"
            f"Remembered for this session and shared by every view of {what}; "
            f"a figure exported from the view uses it too.")
