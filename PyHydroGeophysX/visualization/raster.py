"""Rasterize a pygimli per-cell field onto a regular grid for image display.

Shared by the ERT and seismic modules so an inverted model can be shown in the
interactive :class:`ArrayViewer` (zoom, colorbar, value read-out) instead of a
static matplotlib PNG. Qt-free so it can run inside a worker thread.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def rasterize_cell_field(
    mesh: Any, values: Any, nx: int = 260, nz: int = 150
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Interpolate a per-cell ``values`` field onto a regular grid.

    Returns ``(grid, extent)`` where ``grid`` has shape ``(nz, nx)`` with row 0 at
    the top of the section and ``extent = (x0, x1, z0, z1)`` uses depth (0 at the
    top of the para-domain, increasing downward). With the ``ArrayViewer`` default
    inverted Y axis this renders surface-at-top. Cells outside the para-domain
    stay ``NaN`` (transparent), matching the masked matplotlib view.
    """
    from scipy.interpolate import griddata

    centers = np.asarray(mesh.cellCenters(), dtype=float)
    xc, yc = centers[:, 0], centers[:, 1]
    vals = np.asarray(values, dtype=float).ravel()
    depth = float(yc.max()) - yc  # 0 at the top, positive downward

    xi = np.linspace(float(xc.min()), float(xc.max()), int(nx))
    di = np.linspace(float(depth.min()), float(depth.max()), int(nz))
    grid_x, grid_d = np.meshgrid(xi, di)
    grid = griddata((xc, depth), vals, (grid_x, grid_d), method="linear")  # row 0 = top
    extent = (float(xc.min()), float(xc.max()), float(di.min()), float(di.max()))
    return np.asarray(grid, dtype=float), extent
