"""One visual style for every figure a report contains.

The figure code grew one generator at a time and each made its own choices. In
a single five-survey report that produced three figure sizes ((12, 6), (20, 5)
and one scaled per panel), six font sizes between 9 and 16, four different ways
of titling the same survey - ``Baseline``, ``Baseline (t=0)``, ``Time Step 2``,
``Survey 2`` - and three colormaps including ``jet``. Nothing was wrong on its
own; together they read as figures from four different reports.

So the decisions live here instead, in one :class:`FigureStyle` that every
generator asks. Two consequences beyond consistency:

- **Survey 3 is called Survey 3 everywhere.** The tables already number surveys
  from the baseline and date them; :func:`survey_title` gives the figures the
  same labels, so a panel can be matched to a table row without counting.
- **The reader can change it.** A style is built from the workflow
  configuration, so "make the figures larger" or "don't use jet" is a setting
  rather than an edit to five functions. Values are validated against what
  matplotlib actually accepts, because a style that came from a language model
  reading a request will occasionally contain something invented.

``jet`` is not among the defaults on purpose: it is not perceptually uniform,
so it invents banding that readers see as structure in the subsurface.
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Colormaps offered for each quantity, first being the default. Restricted to
#: perceptually uniform maps for magnitudes and diverging maps for changes: a
#: change plot needs a colormap whose midpoint is zero, and a magnitude plot
#: needs one whose steps are visually equal.
COLORMAPS: Dict[str, List[str]] = {
    "resistivity": ["viridis", "cividis", "plasma", "magma", "turbo"],
    "change": ["RdBu_r", "coolwarm", "seismic", "PuOr_r", "BrBG_r"],
    "water_content": ["viridis", "cividis", "YlGnBu", "Blues"],
}

#: Named sizes, as inches per panel. A request saying "bigger figures" should
#: not have to name a number.
SIZES: Dict[str, Tuple[float, float]] = {
    "compact": (3.2, 3.6),
    "normal": (4.2, 4.6),
    "large": (5.6, 6.0),
}

#: No figure is allowed to exceed this width in inches, however many panels it
#: has. Five panels at "large" would otherwise be 28 inches across, which is
#: 8400 pixels at 300 dpi - the size that made the desktop preview unusable.
MAX_FIGURE_WIDTH_IN = 20.0


@dataclass(frozen=True)
class FigureStyle:
    """Every visual decision a report figure makes."""

    panel_width: float = SIZES["normal"][0]
    panel_height: float = SIZES["normal"][1]
    dpi: int = 300
    title_size: int = 12
    label_size: int = 11
    tick_size: int = 10
    font_family: str = "Arial"
    resistivity_cmap: str = COLORMAPS["resistivity"][0]
    change_cmap: str = COLORMAPS["change"][0]
    water_content_cmap: str = COLORMAPS["water_content"][0]
    colorbar_orientation: str = "vertical"

    def figure_size(self, n_panels: int, rows: int = 1) -> Tuple[float, float]:
        """Width and height in inches for a row of ``n_panels``, capped.

        Examples
        --------
        >>> FigureStyle().figure_size(1)
        (4.2, 4.6)
        >>> width, height = FigureStyle().figure_size(12)
        >>> width <= MAX_FIGURE_WIDTH_IN
        True
        """
        panels = max(1, int(n_panels))
        width = min(self.panel_width * panels, MAX_FIGURE_WIDTH_IN)
        return (round(width, 2), round(self.panel_height * max(1, int(rows)), 2))

    def cmap_for(self, quantity: str) -> str:
        """The colormap for ``'resistivity'``, ``'change'`` or ``'water_content'``.

        Examples
        --------
        >>> FigureStyle().cmap_for('change')
        'RdBu_r'
        >>> FigureStyle().cmap_for('something else')
        'viridis'
        """
        return {"resistivity": self.resistivity_cmap,
                "change": self.change_cmap,
                "water_content": self.water_content_cmap}.get(
                    quantity, self.resistivity_cmap)


def style_from_config(config: Optional[Dict[str, Any]] = None) -> FigureStyle:
    """A style from ``config['figure_style']``, ignoring anything invalid.

    Parameters
    ----------
    config : dict, optional
        Workflow configuration. ``figure_style`` may carry ``size`` (a key of
        :data:`SIZES`, or a number of inches per panel), ``dpi``, ``font``, and
        a colormap per quantity.

    Returns
    -------
    FigureStyle
        The defaults, with any valid override applied.

    Raises
    ------
    None
        A style is a preference. An unusable value is dropped rather than
        failing a run that has already done its computation - and a colormap
        that does not exist would otherwise raise inside matplotlib, several
        steps after the place that could explain it.

    Examples
    --------
    >>> style_from_config({'figure_style': {'size': 'large'}}).panel_width
    5.6
    >>> style_from_config({'figure_style': {'size': 3.0, 'dpi': 150}}).dpi
    150
    >>> style_from_config({'figure_style': {'resistivity_cmap': 'jet'}}).resistivity_cmap
    'viridis'
    >>> style_from_config(None) == FigureStyle()
    True
    """
    requested = ((config or {}).get("figure_style") or {})
    if not isinstance(requested, dict):
        return FigureStyle()
    style = FigureStyle()
    changes: Dict[str, Any] = {}

    size = requested.get("size")
    if isinstance(size, str) and size.lower() in SIZES:
        width, height = SIZES[size.lower()]
        changes.update(panel_width=width, panel_height=height)
    elif isinstance(size, (int, float)) and 1.5 <= float(size) <= 10.0:
        changes.update(panel_width=float(size), panel_height=float(size) * 1.1)

    dpi = requested.get("dpi")
    if isinstance(dpi, (int, float)) and 72 <= int(dpi) <= 600:
        changes["dpi"] = int(dpi)

    font = requested.get("font") or requested.get("font_family")
    if isinstance(font, str) and font.strip():
        changes["font_family"] = font.strip()

    for quantity, field in (("resistivity", "resistivity_cmap"),
                            ("change", "change_cmap"),
                            ("water_content", "water_content_cmap")):
        value = requested.get(field) or requested.get(f"{quantity}_colormap")
        if isinstance(value, str) and value in COLORMAPS[quantity]:
            changes[field] = value

    orientation = requested.get("colorbar")
    if orientation in ("vertical", "horizontal"):
        changes["colorbar_orientation"] = orientation

    return replace(style, **changes) if changes else style


def survey_title(index: int, dates: Optional[Sequence[str]] = None) -> str:
    """How every figure names one survey.

    Parameters
    ----------
    index : int
        Zero-based position in the series; 0 is the baseline.
    dates : sequence of str, optional
        Acquisition dates, in the same order.

    Returns
    -------
    str
        ``'Survey 1 (baseline)'`` with the date on a second line when known.
        The number matches the tables, which count from the baseline, so a
        panel and a row can be matched without counting panels.

    Raises
    ------
    None

    Examples
    --------
    >>> survey_title(0)
    'Survey 1 (baseline)'
    >>> survey_title(2, ['2017-11-05', '2017-11-06', '2017-11-07'])
    'Survey 3\\n2017-11-07'
    >>> survey_title(9, ['2017-11-05'])
    'Survey 10'
    """
    label = f"Survey {index + 1}" + (" (baseline)" if index == 0 else "")
    if dates and 0 <= index < len(dates) and str(dates[index]).strip():
        return f"{label}\n{dates[index]}"
    return label


def change_title(index: int, dates: Optional[Sequence[str]] = None) -> str:
    """How every figure names a change relative to the baseline.

    Examples
    --------
    >>> change_title(1, ['2017-11-05', '2017-11-06'])
    'Survey 2 - Survey 1\\n2017-11-06'
    >>> change_title(1)
    'Survey 2 - Survey 1'
    """
    label = f"Survey {index + 1} - Survey 1"
    if dates and 0 <= index < len(dates) and str(dates[index]).strip():
        return f"{label}\n{dates[index]}"
    return label


def apply(ax, style: FigureStyle, title: str = "", xlabel: str = "Distance (m)",
          ylabel: str = "Elevation (m)") -> None:
    """Give one panel the report's axis labels, title and type sizes.

    Parameters
    ----------
    ax : matplotlib axes
        The panel.
    style : FigureStyle
        The style in force.
    title : str, optional
        Panel title; omitted when blank.
    xlabel, ylabel : str
        Axis labels. Pass ``''`` to leave an axis unlabelled - worth doing for
        every panel but the first in a row, where repeating "Elevation (m)"
        five times costs the width that made the labels collide.
    """
    if title:
        ax.set_title(title, fontsize=style.title_size, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style.label_size)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style.label_size)
    ax.tick_params(labelsize=style.tick_size)


def panels(n_panels: int, style: FigureStyle, rows: int = 1):
    """A figure and its axes, sized by the style.

    Returns ``(fig, axes)`` with ``axes`` always a flat array, so a
    single-panel figure and a five-panel one are indexed the same way.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(rows, max(1, int(n_panels)),
                             figsize=style.figure_size(n_panels, rows))
    return fig, np.atleast_1d(axes).ravel()


def save(fig, path: str, style: FigureStyle) -> str:
    """Write ``fig`` at the style's resolution and close it."""
    import matplotlib.pyplot as plt

    fig.savefig(path, dpi=style.dpi, bbox_inches="tight")
    plt.close(fig)
    return path
