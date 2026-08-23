"""Render a live studio panel to PNG bytes so the agent can look at it.

The studio already tells AQUAH what happened in numbers; this module is how it
shows what happened. Panels are discovered generically by widget class, so a new
module page gets capture support without registering anything, and a page may
override the choice by defining ``agent_views() -> {name: widget}``.

Three rendering paths, in order of fidelity:

- a matplotlib panel is re-rendered through ``Figure.savefig`` at print DPI, which
  keeps axis ticks and colorbar numbers legible where a widget grab would blur
  them into noise;
- an embedded PyVista plotter uses its own ``screenshot``, because the GL surface
  behind the widget often grabs as a black rectangle;
- anything else falls back to ``QWidget.grab``.

Images are capped on the long edge (``PHGX_AGENT_CAPTURE_MAX_PX``, default 1400)
since every extra pixel is tokens on each later call and no current model reads
finer detail than that.
"""

from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QBuffer, QByteArray, Qt
from PySide6.QtWidgets import QWidget

#: Longest edge kept in a captured image, in pixels.
MAX_EDGE_PX = int(os.getenv("PHGX_AGENT_CAPTURE_MAX_PX", "1400"))

#: DPI used when a matplotlib figure is re-rendered instead of grabbed.
FIGURE_DPI = 110

#: Name reported for the whole module page, always available.
PAGE_VIEW = "page"

#: Viewer widget class name -> the short view name the agent sees. Matching by
#: class name avoids importing every viewer (several pull optional GL/VTK deps).
_VIEWER_NAMES: Dict[str, str] = {
    "MeshResultView": "section",
    "InversionQualityView": "quality",
    "SeismicViewer": "gather",
    "Model3DView": "model3d",
    "VTKVolumeView": "volume3d",
    "PlanSliceView": "plan_slice",
    "ZoomableImageView": "figure",
    "CurveViewer": "curves",
    "ArrayViewer": "array",
}


def _is_visible(widget: QWidget) -> bool:
    try:
        return bool(widget.isVisible() and widget.width() > 8 and widget.height() > 8)
    except Exception:  # noqa: BLE001 - a half-constructed widget is simply not capturable
        return False


def discover_views(page: QWidget) -> List[Dict[str, Any]]:
    """List the capturable panels of ``page``, visible ones first.

    Each entry is ``{"name", "widget", "visible"}``. The whole page is always the
    last entry, so a module with no recognised viewer can still be captured.
    """
    found: List[Dict[str, Any]] = []
    declared = getattr(page, "agent_views", None)
    if callable(declared):
        try:
            mapping = declared() or {}
        except Exception:  # noqa: BLE001
            mapping = {}
        if isinstance(mapping, dict):
            found = [{"name": str(k), "widget": w, "visible": _is_visible(w)}
                     for k, w in mapping.items() if isinstance(w, QWidget)]
    if not found:
        counts: Dict[str, int] = {}
        for child in page.findChildren(QWidget):
            base = _VIEWER_NAMES.get(type(child).__name__)
            if base is None:
                continue
            counts[base] = counts.get(base, 0) + 1
            name = base if counts[base] == 1 else f"{base}{counts[base]}"
            found.append({"name": name, "widget": child, "visible": _is_visible(child)})
    found.sort(key=lambda v: not v["visible"])  # stable: visible panels first
    found.append({"name": PAGE_VIEW, "widget": page, "visible": _is_visible(page)})
    return found


def view_names(page: QWidget) -> List[str]:
    """The view names ``capture`` accepts for ``page``, in preference order."""
    return [v["name"] for v in discover_views(page)]


def capture(page: QWidget, view: Optional[str] = None) -> Tuple[str, bytes]:
    """Render one view of ``page`` to PNG bytes.

    With no ``view``, the first visible panel is used, falling back to the whole
    page. Raises :class:`LookupError` when a named view does not exist and
    :class:`RuntimeError` when the panel renders empty.
    """
    views = discover_views(page)
    if view:
        match = next((v for v in views if v["name"] == view), None)
        if match is None:
            raise LookupError(view)
    else:
        match = next((v for v in views if v["visible"]), views[-1])
    data = _render(match["widget"])
    if not data:
        raise RuntimeError(f"View '{match['name']}' rendered empty.")
    return match["name"], data


# -- rendering ----------------------------------------------------------------
def _render(widget: QWidget) -> bytes:
    """PNG bytes for one widget, by the most faithful path it supports."""
    fig = getattr(widget, "_fig", None)
    if fig is not None and hasattr(fig, "savefig"):
        try:
            return _figure_png(fig)
        except Exception:  # noqa: BLE001 - fall through to the widget grab
            pass
    plotter = getattr(widget, "_plotter", None)
    if plotter is not None and hasattr(plotter, "screenshot"):
        try:
            return _plotter_png(plotter)
        except Exception:  # noqa: BLE001
            pass
    return _pixmap_png(widget.grab())


def _figure_png(fig: Any) -> bytes:
    buf = io.BytesIO()
    # Carry the figure's own face colour so a dark-theme panel is captured as the
    # user sees it rather than on the rcParams default white.
    fig.savefig(buf, format="png", dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    return _downscale(buf.getvalue())


def _plotter_png(plotter: Any) -> bytes:
    import matplotlib.image as mpimg
    import numpy as np

    arr = plotter.screenshot(return_img=True)
    buf = io.BytesIO()
    mpimg.imsave(buf, np.asarray(arr), format="png")
    return _downscale(buf.getvalue())


def _pixmap_png(pixmap: Any) -> bytes:
    if pixmap is None or pixmap.isNull():
        return b""
    if max(pixmap.width(), pixmap.height()) > MAX_EDGE_PX:
        pixmap = pixmap.scaled(MAX_EDGE_PX, MAX_EDGE_PX,
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QBuffer.WriteOnly)
    pixmap.save(buffer, "PNG")
    buffer.close()
    return bytes(data)


def _downscale(png: bytes) -> bytes:
    """Re-encode ``png`` under the long-edge cap, or return it unchanged."""
    from PySide6.QtGui import QPixmap

    pixmap = QPixmap()
    if not pixmap.loadFromData(png, "PNG"):
        return png
    if max(pixmap.width(), pixmap.height()) <= MAX_EDGE_PX:
        return png
    return _pixmap_png(pixmap)
