"""Lossless sidecar support for PyGIMLi mesh workflow boundaries."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict


def _position(value: Any) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def ansi_safe(text: str) -> bool:
    """Whether Windows' ANSI codepage can represent this path.

    PyGIMLi's mesh readers and writers take a narrow ``std::string`` and hand it
    to the C++ runtime, which on Windows resolves the bytes through the ANSI
    codepage rather than UTF-8. A path containing characters that codepage cannot
    encode never reaches the filesystem, and the C++ layer reports the failure as
    "No such file or directory" even though the directory is present. A localized
    OneDrive folder is the usual way to hit this: `C:\\Users\\me\\OneDrive\\文档`
    is unrepresentable whenever the machine's codepage is a Western one.
    """
    try:
        text.encode("mbcs")          # Windows only; the ANSI codepage
    except UnicodeEncodeError:
        return False
    except LookupError:
        return True                  # not Windows, so narrow paths are UTF-8
    return True


def via_ascii_path(run: Callable[[str], None], target: Path, *, mode: str) -> Path:
    """Run a PyGIMLi reader/writer that cannot open a non-ANSI path.

    ``mode`` is ``"write"`` or ``"read"``. The call is attempted directly first,
    so nothing changes on paths that already work. Only when that fails does the
    file get staged through a temporary directory whose own path is ASCII, moved
    or copied with Python's wide-character filesystem calls.
    """
    target = Path(target)
    if mode == "write":
        target.parent.mkdir(parents=True, exist_ok=True)
    if ansi_safe(str(target)):
        run(str(target))
        return target

    with tempfile.TemporaryDirectory(prefix="phgx_mesh_") as tmp:
        if not ansi_safe(tmp):
            # Staging cannot help when the temp directory is unrepresentable too.
            raise RuntimeError(
                f"PyGIMLi cannot open '{target}' because Windows' ANSI codepage cannot "
                f"represent it, and the temporary directory '{tmp}' has the same problem. "
                "Set TEMP to an ASCII path, or choose an output folder without characters "
                "outside the system codepage.")
        staged = Path(tmp) / f"staged{target.suffix or '.tmp'}"
        if mode == "read":
            shutil.copy2(target, staged)
            run(str(staged))
            return staged
        run(str(staged))
        # PyGIMLi appends its own extension when the name lacks one, so trust the
        # directory over the name we asked for.
        produced = staged if staged.exists() else next(iter(sorted(Path(tmp).iterdir())), None)
        if produced is None:
            raise RuntimeError(f"Writing '{target.name}' produced no file.")
        shutil.move(str(produced), str(target))
        return target


def save_mesh_artifact(
    mesh: Any,
    mesh_path: str | Path,
    sidecar_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save BMS plus metadata BMS omits (regions, holes, secondary nodes)."""
    bms = Path(mesh_path)
    sidecar = (
        Path(sidecar_path)
        if sidecar_path is not None
        else bms.with_suffix(bms.suffix + ".structure.json")
    )
    bms.parent.mkdir(parents=True, exist_ok=True)
    via_ascii_path(mesh.save, bms, mode="write")
    secondary = []
    for index in range(int(mesh.secondaryNodeCount())):
        node = mesh.secondaryNode(index)
        secondary.append(
            {
                "position": _position(node.pos()),
                "cells": sorted(int(cell.id()) for cell in node.cellSet()),
                "boundaries": sorted(int(boundary.id()) for boundary in node.boundSet()),
            }
        )
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "node_count": int(mesh.nodeCount()),
        "cell_count": int(mesh.cellCount()),
        "cell_markers": [int(value) for value in mesh.cellMarkers()],
        "boundary_markers": [int(value) for value in mesh.boundaryMarkers()],
        "region_markers": [
            {"position": _position(marker), "marker": int(marker.marker())}
            for marker in mesh.regionMarkers()
        ],
        "hole_markers": [_position(marker) for marker in mesh.holeMarkers()],
        "secondary_nodes": secondary,
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return bms, sidecar


def load_mesh_artifact(
    mesh_path: str | Path,
    sidecar_path: str | Path,
) -> Any:
    """Load BMS and restore every structure recorded by its sidecar."""
    import pygimli as pg

    loaded: Dict[str, Any] = {}
    via_ascii_path(lambda p: loaded.setdefault("mesh", pg.load(p)), Path(mesh_path), mode="read")
    mesh = loaded["mesh"]
    payload = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    if int(mesh.nodeCount()) != int(payload["node_count"]):
        raise ValueError("Mesh node count changed during BMS round-trip.")
    if int(mesh.cellCount()) != int(payload["cell_count"]):
        raise ValueError("Mesh cell count changed during BMS round-trip.")
    if [int(value) for value in mesh.cellMarkers()] != payload["cell_markers"]:
        raise ValueError("Mesh cell markers changed during BMS round-trip.")
    if [int(value) for value in mesh.boundaryMarkers()] != payload["boundary_markers"]:
        raise ValueError("Mesh boundary markers changed during BMS round-trip.")

    for item in payload.get("region_markers") or []:
        mesh.addRegionMarker(pg.Pos(*item["position"]), int(item["marker"]))
    for position in payload.get("hole_markers") or []:
        mesh.addHoleMarker(pg.Pos(*position))
    for item in payload.get("secondary_nodes") or []:
        node = mesh.createSecondaryNode(pg.Pos(*item["position"]))
        for cell_id in item.get("cells") or []:
            mesh.cell(int(cell_id)).addSecondaryNode(node)
        for boundary_id in item.get("boundaries") or []:
            mesh.boundary(int(boundary_id)).addSecondaryNode(node)
    return mesh


__all__ = ["ansi_safe", "load_mesh_artifact", "save_mesh_artifact", "via_ascii_path"]
