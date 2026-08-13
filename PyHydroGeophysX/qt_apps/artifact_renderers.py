"""Pure artifact-to-viewer routing used by Model Viewer and unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


_VTK_FORMATS = {"vtk", "vtu", "vtp", "stl", "ply", "obj"}
_IMAGE_FORMATS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}


def select_renderer(
    artifact: Mapping[str, Any], array_shape: Optional[Sequence[int]] = None
) -> str:
    """Return a stable renderer key without importing Qt or scientific engines."""
    kind = str(artifact.get("kind") or "").lower()
    fmt = str(artifact.get("format") or "").lower().lstrip(".")
    if not fmt:
        fmt = Path(str(artifact.get("path") or "")).suffix.lower().lstrip(".")
    if "model_bundle" in kind or kind in {"pygimli_bundle", "mesh_model_bundle"}:
        return "mesh_bundle"
    if fmt in _VTK_FORMATS or kind in {"velocity_model", "volume", "vtk"}:
        return "vtk"
    if fmt in _IMAGE_FORMATS or "figure" in kind or kind == "image":
        return "image"
    if fmt in {"npy", "npz"}:
        shape = tuple(array_shape or artifact.get("shape") or ())
        if len(shape) >= 3:
            return "array_stack"
        if len(shape) == 1 or any(word in kind for word in ("curve", "profile", "response")):
            return "curve"
        return "array"
    if fmt in {"csv", "tsv", "dat", "txt"}:
        if any(word in kind for word in ("curve", "profile", "response", "convergence")):
            return "curve"
        return "table"
    if fmt == "json":
        return "json"
    return "file"


__all__ = ["select_renderer"]
