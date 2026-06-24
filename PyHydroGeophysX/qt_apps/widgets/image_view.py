"""Pan/zoom viewer for a pre-rendered RGB image (e.g. a matplotlib figure).

Used by the forward-model gallery so composite figures that are already rendered
become interactive (mouse-wheel zoom, drag to pan) and rescale with the window,
instead of being shown as a fixed-width ``QLabel`` pixmap.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class ZoomableImageView(QWidget):
    """Show an RGB(A) image file in a pan/zoom pyqtgraph view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)  # image row 0 at the top
        self._img = pg.ImageItem(axisOrder="row-major")
        self._vb.addItem(self._img)

    def set_image_file(self, path) -> bool:
        """Load and display an image file. Returns ``False`` if it cannot be read."""
        try:
            import matplotlib.image as mpimg

            arr = mpimg.imread(str(path))  # (H, W) or (H, W, 3/4)
        except Exception:  # noqa: BLE001
            return False
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        levels = (0.0, 1.0) if arr.dtype.kind == "f" else None
        self._img.setImage(arr, levels=levels)
        self._vb.autoRange()
        return True
