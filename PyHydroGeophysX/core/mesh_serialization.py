"""Lossless sidecar support for PyGIMLi mesh workflow boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _position(value: Any) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


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
    mesh.save(str(bms))
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

    mesh = pg.load(str(mesh_path))
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


__all__ = ["load_mesh_artifact", "save_mesh_artifact"]
