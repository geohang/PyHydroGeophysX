"""Process-safe persistence for joint-inversion observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


def _pack(value: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise TypeError("Object-dtype arrays cannot cross a workflow boundary.")
        key = f"array_{len(arrays)}"
        arrays[key] = np.asarray(value)
        return {"type": "array", "key": key}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return {"type": "scalar", "value": value}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": {str(key): _pack(item, arrays) for key, item in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {"type": "list", "items": [_pack(item, arrays) for item in value]}
    raise TypeError(
        f"Unsupported joint observation value {type(value).__name__}; "
        "materialize it to arrays and JSON scalars first."
    )


def _unpack(node: Mapping[str, Any], archive: Any) -> Any:
    node_type = node["type"]
    if node_type == "array":
        return np.asarray(archive[str(node["key"])])
    if node_type == "scalar":
        return node.get("value")
    if node_type == "mapping":
        return {
            str(key): _unpack(value, archive)
            for key, value in dict(node.get("items") or {}).items()
        }
    if node_type == "list":
        return [_unpack(value, archive) for value in node.get("items") or []]
    raise ValueError(f"Unknown joint observation manifest node {node_type!r}.")


def save_joint_observations(
    method: str,
    payload: Any,
    destination: str | Path,
) -> Path:
    """Persist one method's observations without pickle or Qt dependencies."""
    normalized = str(method).upper()
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if normalized in {"ERT", "SRT"}:
        target = target.with_suffix(".dat")
        if not hasattr(payload, "save"):
            raise TypeError(f"{normalized} observations must provide DataContainer.save().")
        payload.save(str(target))
        return target

    target = target.with_suffix(".npz")
    arrays: Dict[str, np.ndarray] = {}
    manifest = {
        "schema_version": 1,
        "method": normalized,
        "payload": _pack(payload, arrays),
    }
    np.savez_compressed(
        target,
        __manifest__=np.asarray(json.dumps(manifest, sort_keys=True)),
        **arrays,
    )
    return target


def load_joint_observations(method: str, source: str | Path) -> Any:
    """Load a sidecar written by :func:`save_joint_observations`."""
    normalized = str(method).upper()
    path = Path(source)
    if normalized == "ERT":
        from pygimli.physics import ert

        return ert.load(str(path))
    if normalized == "SRT":
        import pygimli.physics.traveltime as tt

        return tt.load(str(path))
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive["__manifest__"].item()))
        if str(manifest.get("method", "")).upper() != normalized:
            raise ValueError(
                f"Joint artifact contains {manifest.get('method')!r}, expected {normalized!r}."
            )
        return _unpack(manifest["payload"], archive)


__all__ = ["load_joint_observations", "save_joint_observations"]
