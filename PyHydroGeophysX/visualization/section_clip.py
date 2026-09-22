"""Clip a drawn 2-D section to the part of it the data resolve.

A coverage mask blanks whole cells, so the cut can only follow cell boundaries -
and an inversion mesh has metre-scale triangles at the surface and ten-metre ones
at depth, which is why a masked section ends in a saw-tooth edge with islands
hanging off it. Clipping the *drawing* instead, with a matplotlib clip path, cuts
exactly on the envelope: the clean shape traditional resistivity software produces,
independent of the mesh.

Use :func:`clip_section_to_coverage` right after ``pg.show`` on the same axes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from PyHydroGeophysX.core import section_geometry as geometry

__all__ = ["clip_axes_to_polygon", "clip_section_to_coverage"]


def clip_axes_to_polygon(ax: Any, polygon: Any, *, outline: bool = True,
                         outline_color: str = "0.25", outline_width: float = 0.8,
                         tighten: bool = False) -> bool:
    """Clip everything already drawn on ``ax`` to ``polygon``.

    Args:
        ax: matplotlib axes holding the section.
        polygon: closed ``(n, 2)`` array of ``(x, elevation)`` vertices.
        outline: draw the clipping boundary, so the edge reads as a deliberate cut
            rather than as where the model happens to stop.
        outline_color: colour of that boundary.
        outline_width: line width of that boundary.
        tighten: shrink the y-limits to the clipped shape, removing the empty band
            the blanked part of the mesh would otherwise leave.

    Returns:
        True when the clip was applied.
    """
    from matplotlib.patches import Polygon as MplPolygon

    vertices = np.asarray(polygon, dtype=float)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 2:
        return False

    # Snapshot first: the clip patch is itself an axes patch, and clipping it by
    # itself is at best a no-op and at worst a recursive transform.
    artists = list(ax.collections) + list(ax.images) + list(ax.patches)
    patch = MplPolygon(vertices[:, :2], closed=True, transform=ax.transData,
                       facecolor="none", edgecolor="none", linewidth=0.0)
    ax.add_patch(patch)
    for artist in artists:
        try:
            artist.set_clip_path(patch)
        except Exception:  # noqa: BLE001 - one artist refusing is not fatal
            continue
    if outline:
        ax.plot(vertices[:, 0], vertices[:, 1], color=outline_color,
                linewidth=float(outline_width), zorder=5, solid_joinstyle="round")
    if tighten:
        lo, hi = float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))
        if hi > lo:
            margin = 0.04 * (hi - lo)
            ax.set_ylim(lo - margin, hi + margin)
    return True


def clip_section_to_coverage(ax: Any, mesh: Any, coverage: Any,
                             threshold: float = -2.0, *,
                             sensors: Any = None,
                             outline: bool = True,
                             tighten: bool = False,
                             **envelope_kwargs: Any) -> Optional[Dict[str, Any]]:
    """Clip the section on ``ax`` to the coverage envelope of ``mesh``.

    Returns the envelope description (see
    :func:`PyHydroGeophysX.core.section_geometry.coverage_envelope`) when a clip was
    applied, or ``None`` when the cut keeps everything, keeps nothing, or the mesh
    geometry cannot be read - in which case the section is simply left unclipped.
    """
    envelope = geometry.coverage_envelope(
        mesh, coverage, threshold, sensors=sensors, **envelope_kwargs)
    polygon = geometry.envelope_polygon(envelope)
    if polygon is None:
        return None
    if not clip_axes_to_polygon(ax, polygon, outline=outline, tighten=tighten):
        return None
    envelope = dict(envelope)
    envelope["polygon"] = polygon
    return envelope
