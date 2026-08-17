"""Process-safe persistence for joint-inversion observations.

The array packing lives in :mod:`PyHydroGeophysX.data_processing.run_inputs`,
which every module now writes its run inputs through. The manifest keys below
stay as they were so joint runs saved before that move still load; only the
ERT/SRT branches, which hand off to PyGIMLi's own writers, are specific to this
module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from PyHydroGeophysX.data_processing.run_inputs import pack as _pack, unpack as _unpack


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
