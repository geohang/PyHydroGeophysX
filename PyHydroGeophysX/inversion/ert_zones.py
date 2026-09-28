"""A-priori resistivity zones for ERT inversion: polygons, each with a value.

A zone is a polygon drawn on the inversion mesh in its own coordinates - x along
the line and the mesh's second axis, elevation - carrying an a-priori
resistivity and, optionally, a flag that holds it fixed. It is how prior
knowledge enters the inversion: a clay lens known from a borehole, a concrete
foundation, the water in a tank.

The value is a priori in the regularized sense. The inversion starts from it,
and the smoothness constraint acts on the departure from the a-priori model
rather than on the model itself, so the contrast at a zone's edge costs nothing
unless the data argue against it. A fixed zone is not inverted at all.

Zones are geometry, not cell lists: the cells a zone covers are those whose
centres fall inside its polygon, worked out afresh on whatever mesh the run
builds, so a zone survives a change of mesh quality or depth. Where zones
overlap, the later one wins.

The same zone list reaches every consumer - the ERT page's mesh preview, the
single-survey and time-lapse inversions - as plain JSON-friendly dicts::

    {"name": "Clay", "polygon": [[x0, z0], [x1, z1], ...],
     "resistivity": 20.0, "fixed": False}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ZonePrior", "cell_centers_2d", "normalize_zones", "polygons_cover_nothing",
           "zone_colors", "zone_prior"]


@dataclass
class ZonePrior:
    """What a zone list means on one parameter mesh.

    ``values`` holds the a-priori resistivity of every cell a zone covers and
    NaN elsewhere; ``owner`` the index of the zone each cell belongs to, -1
    outside every zone; ``fixed`` the cells held fixed; ``report`` has one entry
    per zone, with the number of cells it ended up covering once later zones had
    taken their share.
    """

    values: np.ndarray
    owner: np.ndarray
    fixed: np.ndarray
    report: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def in_zone(self) -> np.ndarray:
        return self.owner >= 0

    @property
    def any_fixed(self) -> bool:
        return bool(np.any(self.fixed))

    def with_background(self, background: float) -> np.ndarray:
        """The a-priori model: zone values, and ``background`` everywhere else."""
        return np.where(self.in_zone, self.values, float(background))

    def summary(self) -> str:
        """One line per zone, for the run log."""
        lines = []
        for entry in self.report:
            state = "fixed" if entry["fixed"] else "a priori"
            note = f" (clipped from {entry['requested']:g})" if entry.get("clipped") else ""
            lines.append(f"{entry['name']}: {entry['cells']} cells at "
                         f"{entry['resistivity']:g} ohm-m{note}, {state}")
        return "; ".join(lines)


def normalize_zones(zones: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Validate a zone list and bring every entry to the canonical form.

    Raises ``ValueError`` naming the zone at fault: a polygon with fewer than
    three vertices, or a resistivity that is not a positive number, would
    otherwise surface as a silent no-op or a NaN deep in the inversion.

    >>> normalize_zones([{"polygon": [[0, 0], [4, 0], [4, -2]], "resistivity": 30}])
    [{'name': 'Zone 1', 'polygon': [[0.0, 0.0], [4.0, 0.0], [4.0, -2.0]], 'resistivity': 30.0, 'fixed': False}]
    """
    out: List[Dict[str, Any]] = []
    for index, zone in enumerate(zones or []):
        name = str((zone or {}).get("name") or f"Zone {index + 1}")
        try:
            polygon = [[float(x), float(y)] for x, y in (zone.get("polygon") or [])]
        except (TypeError, ValueError):
            raise ValueError(f"{name}: the polygon must be a list of [x, z] pairs.") from None
        if len(polygon) < 3:
            raise ValueError(f"{name}: a zone needs at least three vertices, "
                             f"not {len(polygon)}.")
        try:
            resistivity = float(zone.get("resistivity"))
        except (TypeError, ValueError):
            raise ValueError(f"{name}: the resistivity must be a number.") from None
        if not np.isfinite(resistivity) or resistivity <= 0:
            raise ValueError(f"{name}: the resistivity must be positive, "
                             f"not {resistivity:g}.")
        out.append({"name": name, "polygon": polygon, "resistivity": resistivity,
                    "fixed": bool(zone.get("fixed", False))})
    return out


def cell_centers_2d(mesh_or_centers: Any) -> np.ndarray:
    """``(n, 2)`` cell centres - x and the mesh's second axis - of a 2-D mesh.

    Accepts a pyGIMLi mesh or an array of centres. A 3-D mesh is refused: a
    polygon drawn on a section says nothing about a volume.
    """
    if hasattr(mesh_or_centers, "cellCenters"):
        if int(getattr(mesh_or_centers, "dim", lambda: 2)()) == 3:
            raise ValueError("Zones are drawn on a 2-D section and cannot be "
                             "applied to a 3-D mesh.")
        centres = np.asarray(mesh_or_centers.cellCenters(), dtype=float)
    else:
        centres = np.asarray(mesh_or_centers, dtype=float)
    centres = np.atleast_2d(centres)
    if centres.shape[1] < 2:
        raise ValueError("Cell centres need two coordinates.")
    return centres[:, :2]


def zone_prior(mesh_or_centers: Any, zones: Optional[Iterable[Any]], *,
               bounds: Optional[Tuple[float, float]] = None) -> ZonePrior:
    """Which cells each zone covers, and the a-priori model they make.

    Parameters
    ----------
    mesh_or_centers : pygimli.Mesh or array_like
        The parameter mesh (``paraDomain``), in the order of the model vector,
        or its ``(n, 2)`` cell centres.
    zones : iterable of dict
        See :func:`normalize_zones`. Later zones override earlier ones.
    bounds : (float, float), optional
        The inversion's resistivity limits. A value outside them is clipped to
        the limit, as the inversion would clip it on its first step, and the
        report says so.

    Returns
    -------
    ZonePrior

    Examples
    --------
    >>> centres = [[0.5, -0.5], [1.5, -0.5], [2.5, -0.5]]
    >>> zones = [{"polygon": [[0, 0], [2, 0], [2, -1], [0, -1]], "resistivity": 10},
    ...          {"polygon": [[1, 0], [3, 0], [3, -1], [1, -1]], "resistivity": 99,
    ...           "fixed": True}]
    >>> prior = zone_prior(centres, zones)
    >>> prior.values.tolist(), prior.owner.tolist(), prior.fixed.tolist()
    ([10.0, 99.0, 99.0], [0, 1, 1], [False, True, True])
    >>> [entry["cells"] for entry in prior.report]
    [1, 2]
    """
    from matplotlib.path import Path

    zones = normalize_zones(zones)
    centres = cell_centers_2d(mesh_or_centers)
    n = centres.shape[0]
    owner = np.full(n, -1, dtype=int)
    for index, zone in enumerate(zones):
        inside = Path(np.asarray(zone["polygon"], dtype=float)).contains_points(centres)
        owner[inside] = index                      # a later zone takes the cell

    values = np.full(n, np.nan)
    fixed = np.zeros(n, dtype=bool)
    report: List[Dict[str, Any]] = []
    low, high = (float(bounds[0]), float(bounds[1])) if bounds else (0.0, np.inf)
    for index, zone in enumerate(zones):
        requested = zone["resistivity"]
        value = float(np.clip(requested, low, high)) if bounds else requested
        cells = owner == index
        values[cells] = value
        fixed[cells] = zone["fixed"]
        report.append({"name": zone["name"], "resistivity": value,
                       "requested": requested, "clipped": value != requested,
                       "fixed": zone["fixed"], "cells": int(cells.sum())})
    return ZonePrior(values=values, owner=owner, fixed=fixed, report=report)


def polygons_cover_nothing(prior: ZonePrior) -> List[str]:
    """Names of zones that ended up with no cell at all, to warn about."""
    return [entry["name"] for entry in prior.report if entry["cells"] == 0]


def zone_colors(count: int) -> Sequence[str]:
    """Distinct colours for ``count`` zones, stable in their order."""
    palette = ("#e4572e", "#2e86ab", "#f3a712", "#76b041", "#a23b72",
               "#5c4d7d", "#17bebb", "#c17767", "#6b818c", "#ffc914")
    return [palette[i % len(palette)] for i in range(int(count))]
