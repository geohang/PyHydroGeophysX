"""A workflow's live objects across a process boundary.

A workflow run in a process of its own (the studio's ``ProcessWorkflowWorker``)
hands its result back as JSON, and ``WorkflowRunResult.objects`` - the meshes,
arrays, tables and fitted models a page shows - would stay behind in the child.
:func:`save_objects` writes them into a folder beside the result file and
returns a JSON manifest; :func:`load_objects` reads them back in the studio, so
a page gets the same ``objects`` whether the run was in a thread or a process.

Every value is written in the format made for it: arrays as ``.npy``, a
PyGIMLi mesh as ``.bms`` with the sidecar that keeps what BMS drops, a data
container in its own file format, a table with pandas, and a fitted PyGIMLi
manager - which holds solver state no file can - as the parts a page draws
(:class:`~PyHydroGeophysX.inversion.model_result.ModelResult`). Mappings,
sequences and dataclasses are taken apart and put back together. Only a plain
Python object none of these fit is pickled, and one that cannot be written at
all is named in the manifest's ``skipped``, not dropped silently.

Nothing here imports PyGIMLi or pandas until a value needs them.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np

#: Manifest layout; a studio reading a newer one says so instead of guessing.
OBJECTS_SCHEMA = 1

#: The one file every array of a result is written into.
ARRAYS_FILE = "arrays.npz"

_JSON_SCALARS = (str, int, float, bool, type(None))


def _pg_class(value: Any) -> str:
    """The PyGIMLi core (C++) class of ``value`` (``Mesh``, ``RVector``, ...), or ''.

    Only the compiled core: a method manager (``ERTManager``,
    ``TravelTimeManager``) is Python, lives in ``pygimli.physics`` and is
    written by what is drawn from it (:meth:`_Writer.manager`).
    """
    kind = type(value)
    return kind.__name__ if kind.__module__.startswith("pgcore") else ""


def _is_manager(value: Any) -> bool:
    """A fitted PyGIMLi method manager or its stand-in: a model on a parameter mesh."""
    return (not isinstance(value, type)
            and hasattr(value, "paraDomain") and hasattr(value, "model")
            and not _pg_class(value))


class _Writer:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.count = 0
        # A mesh or an array the result holds twice - as ``mesh`` and as a
        # manager's ``paraDomain`` - is written once; ``kept`` holds each so
        # its id is not reused while the result is written.
        self.written: Dict[int, Dict[str, Any]] = {}
        self.kept: List[Any] = []
        #: What could not be written, by where it was in the result
        #: (``domain_result.meta.fop``): the rest of it still is.
        self.skipped: Dict[str, str] = {}
        #: Every array, written together into one ``.npz``: a result can hold
        #: hundreds of small ones (a ray path per reading), and opening a file
        #: each costs more than the reading.
        self.arrays: Dict[str, np.ndarray] = {}

    def close(self) -> None:
        if self.arrays:
            self.folder.mkdir(parents=True, exist_ok=True)
            np.savez(self.folder / ARRAYS_FILE, **self.arrays)

    def path(self, stem: str, suffix: str) -> Tuple[Path, str]:
        self.count += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:40] or "object"
        name = f"{self.count:03d}_{safe}{suffix}"
        self.folder.mkdir(parents=True, exist_ok=True)
        return self.folder / name, name

    def encode(self, value: Any, stem: str) -> Dict[str, Any]:
        if isinstance(value, _JSON_SCALARS):
            if isinstance(value, float) and not np.isfinite(value):
                return {"kind": "float", "value": repr(value)}
            return {"kind": "json", "value": value}
        if isinstance(value, np.generic):
            return self.encode(value.item(), stem)
        if isinstance(value, (Mapping, list, tuple)) or dataclasses.is_dataclass(value):
            return self.encode_new(value, stem)
        known = self.written.get(id(value))
        if known is None:
            try:
                known = self.encode_new(value, stem)
            except Exception as exc:  # noqa: BLE001 - lose this part, not the object
                self.skipped[stem] = f"{type(value).__name__}: {exc}"
                return {"kind": "missing"}
            self.written[id(value)] = known
            self.kept.append(value)
        return known

    def encode_new(self, value: Any, stem: str) -> Dict[str, Any]:
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                return self.pickled(value, stem)
            key = f"a{len(self.arrays):05d}"
            self.arrays[key] = value
            return {"kind": "ndarray", "key": key}
        pg_class = _pg_class(value)
        if pg_class:
            return self.pygimli(value, pg_class, stem)
        if _is_manager(value):
            return self.manager(value, stem)
        if type(value).__module__.startswith("pandas"):
            path, name = self.path(stem, ".pkl")
            value.to_pickle(path)
            return {"kind": "pandas", "file": name}
        if isinstance(value, Mapping):
            if not all(isinstance(key, str) for key in value):
                return self.pickled(value, stem)
            return {"kind": "dict", "items": {key: self.encode(item, f"{stem}.{key}")
                                              for key, item in value.items()}}
        if isinstance(value, (list, tuple)):
            items = [self.encode(item, f"{stem}.{index}") for index, item in enumerate(value)]
            return {"kind": "tuple" if isinstance(value, tuple) else "list", "items": items}
        if dataclasses.is_dataclass(value):
            kind = type(value)
            return {"kind": "dataclass", "class": f"{kind.__module__}:{kind.__qualname__}",
                    "fields": {field.name: self.encode(getattr(value, field.name),
                                                       f"{stem}.{field.name}")
                               for field in dataclasses.fields(value)}}
        return self.pickled(value, stem)

    def pickled(self, value: Any, stem: str) -> Dict[str, Any]:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        path, name = self.path(stem, ".pkl")
        path.write_bytes(data)
        return {"kind": "pickle", "file": name}

    def pygimli(self, value: Any, pg_class: str, stem: str) -> Dict[str, Any]:
        if pg_class == "Mesh":
            from PyHydroGeophysX.core.mesh_serialization import save_mesh_artifact

            path, name = self.path(stem, ".bms")
            _, sidecar = save_mesh_artifact(value, path)
            return {"kind": "mesh", "file": name, "structure": sidecar.name}
        if pg_class in ("RVector", "IVector", "BVector", "IndexArray", "CVector"):
            described = self.encode(np.asarray(value), stem)
            return {"kind": "pg_vector", "class": pg_class, "array": described}
        if pg_class.startswith("DataContainer"):
            path, name = self.path(stem, ".dat")
            value.save(str(path))
            tokens = [token for token in value.dataMap().keys() if value.isSensorIndex(token)]
            return {"kind": "pg_data", "class": pg_class, "file": name,
                    "sensor_tokens": sorted(tokens)}
        raise TypeError(f"a PyGIMLi {pg_class} has no file format here")

    def manager(self, value: Any, stem: str) -> Dict[str, Any]:
        """A fitted manager as the parts a page draws from it; its solver stays
        behind. Coverage, the standardized coverage and the ray paths travel
        time has, and the data, come too when the manager can give them."""
        parts: Dict[str, Any] = {
            "mesh": self.encode(value.paraDomain, f"{stem}.mesh"),
            "model": self.encode(np.asarray(value.model, dtype=float), f"{stem}.model"),
        }

        def part(name: str) -> Any:
            try:                        # a property can raise, as velocity does
                item = getattr(value, name, None)
                return item() if callable(item) else item
            except Exception:  # noqa: BLE001 - a part the manager cannot give
                return None

        response = part("response")
        if response is None and getattr(value, "inv", None) is not None:
            response = getattr(value.inv, "response", None)
        arrays = {"response": response, "coverage": part("coverage"),
                  "standardized_coverage": part("standardizedCoverage"),
                  "velocity": part("velocity")}
        for key, item in arrays.items():
            try:
                if item is not None:
                    parts[key] = self.encode_new(np.asarray(item, dtype=float), f"{stem}.{key}")
            except Exception:  # noqa: BLE001 - left out; the page does without
                continue
        rays = part("getRayPaths")
        if rays:
            try:
                parts["ray_paths"] = self.encode(
                    [np.asarray(path, dtype=float) for path in rays], f"{stem}.rays")
            except Exception:  # noqa: BLE001
                pass
        data = getattr(value, "data", None)
        if _pg_class(data).startswith("DataContainer"):
            try:
                parts["data"] = self.encode_new(data, f"{stem}.data")
            except Exception:  # noqa: BLE001
                pass
        return {"kind": "manager", "class": type(value).__name__, "parts": parts}


def is_plain_data(value: Any, depth: int = 0) -> bool:
    """Numbers, text, lists and string-keyed mappings of them - what JSON holds,
    NaN and infinity included. Small, so a workflow process sends it back
    whether asked for or not (a result's summary cannot hold a NaN)."""
    if isinstance(value, (_JSON_SCALARS, np.generic)):
        return not isinstance(value, np.generic) or np.ndim(value) == 0
    if depth > 8:
        return False
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and is_plain_data(item, depth + 1)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return len(value) <= 100_000 and all(is_plain_data(item, depth + 1) for item in value)
    return False


def objects_folder(result_file: str | Path) -> Path:
    """Where the objects of the run whose result is ``result_file`` are written."""
    result_file = Path(result_file)
    return result_file.with_name(f"{result_file.stem}_objects")


def save_objects(objects: Mapping[str, Any], folder: str | Path) -> Dict[str, Any]:
    """Write ``objects`` into ``folder``; returns the JSON manifest.

    ``{"schema": 1, "entries": {name: description}, "skipped": {name: reason}}``,
    the descriptions naming files relative to ``folder``.
    """
    writer = _Writer(Path(folder))
    entries: Dict[str, Any] = {}
    for name, value in dict(objects or {}).items():
        try:
            entry = writer.encode(value, str(name))
        except Exception as exc:  # noqa: BLE001 - one object must not lose the others
            writer.skipped[str(name)] = f"{type(value).__name__}: {exc}"
            continue
        if entry.get("kind") != "missing":
            entries[str(name)] = entry
    writer.close()
    manifest = {"schema": OBJECTS_SCHEMA, "entries": entries, "skipped": dict(writer.skipped)}
    json.dumps(manifest, allow_nan=False)
    return manifest


class _Reader:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        # What was written once is read once, and shared as it was.
        self.read: Dict[str, Any] = {}
        self._arrays: Any = None

    def array(self, key: str) -> np.ndarray:
        if self._arrays is None:
            self._arrays = np.load(self.file(ARRAYS_FILE), allow_pickle=False)
        return self._arrays[key]

    def close(self) -> None:
        # The archive stays open while arrays are read from it; closed, the
        # folder it is in can be removed (Windows keeps an open file).
        if self._arrays is not None:
            self._arrays.close()
            self._arrays = None

    def file(self, name: str) -> Path:
        path = (self.folder / name).resolve()
        if self.folder.resolve() not in path.parents:
            raise ValueError(f"{name!r} lies outside the objects folder")
        return path

    def decode(self, entry: Mapping[str, Any]) -> Any:
        kind = entry.get("kind")
        if kind == "missing":
            return None                 # named in the manifest's ``skipped``
        if kind in ("json", "float", "dict", "list", "tuple", "dataclass"):
            return self.decode_new(entry)
        key = json.dumps(entry, sort_keys=True)
        if key not in self.read:
            self.read[key] = self.decode_new(entry)
        return self.read[key]

    def decode_new(self, entry: Mapping[str, Any]) -> Any:
        kind = entry.get("kind")
        if kind == "json":
            return entry.get("value")
        if kind == "float":
            return float(entry["value"])
        if kind == "ndarray":
            return self.array(str(entry["key"]))
        if kind == "dict":
            return {key: self.decode(item) for key, item in entry["items"].items()}
        if kind in ("list", "tuple"):
            items = [self.decode(item) for item in entry["items"]]
            return tuple(items) if kind == "tuple" else items
        if kind == "pandas":
            import pandas as pd

            return pd.read_pickle(self.file(entry["file"]))
        if kind == "mesh":
            from PyHydroGeophysX.core.mesh_serialization import load_mesh_artifact

            # Without the cell-neighbour table, which is most of a 3-D mesh's
            # load and is built again by whatever needs it: this runs in the
            # window's thread, and one C++ call there holds it still.
            return load_mesh_artifact(self.file(entry["file"]), self.file(entry["structure"]),
                                      neighbours=False)
        if kind == "pg_vector":
            import pygimli as pg

            array = self.decode(entry["array"])
            return getattr(pg, str(entry["class"]), pg.Vector)(array)
        if kind == "pg_data":
            import pygimli as pg

            cls = getattr(pg, str(entry["class"]), None) or pg.DataContainer
            path = str(self.file(entry["file"]))
            tokens = " ".join(entry.get("sensor_tokens") or [])
            if cls is pg.DataContainer and tokens:
                return cls(path, tokens)
            return cls(path)
        if kind == "manager":
            return _restored_manager({key: self.decode(item)
                                      for key, item in entry["parts"].items()})
        if kind == "dataclass":
            module, _, qualname = str(entry["class"]).partition(":")
            cls: Any = importlib.import_module(module)
            for part in qualname.split("."):
                cls = getattr(cls, part)
            value = cls.__new__(cls)
            for name, item in entry["fields"].items():
                object.__setattr__(value, name, self.decode(item))
            return value
        if kind == "pickle":
            # Written a moment ago by the workflow process this studio started.
            return pickle.loads(self.file(entry["file"]).read_bytes())
        raise ValueError(f"unknown object kind {kind!r}")


def _restored_manager(parts: Mapping[str, Any]) -> Any:
    """A fitted manager's parts as a :class:`ModelResult`, the shape the viewers take.

    ``getRayPaths`` is there only when ray paths came - the viewers offer the
    overlay when it is - and ``standardizedCoverage`` and ``data`` when those did.
    """
    # numpy only: the inversion code that made the result need not load here.
    from PyHydroGeophysX.inversion.model_result import ModelResult, RayPathModelResult

    common = {"response": parts.get("response"), "coverage": parts.get("coverage"),
              "velocity": parts.get("velocity")}
    if parts.get("ray_paths"):
        result = RayPathModelResult(parts["mesh"], parts["model"],
                                    ray_paths=list(parts["ray_paths"]), **common)
    else:
        result = ModelResult(parts["mesh"], parts["model"], **common)
    standardized = parts.get("standardized_coverage")
    if standardized is not None:
        result.standardizedCoverage = lambda: standardized
    if parts.get("data") is not None:
        result.data = parts["data"]
    return result


def load_objects(manifest: Mapping[str, Any],
                 folder: str | Path) -> Tuple[Dict[str, Any], List[str]]:
    """Read back what :func:`save_objects` wrote; returns ``(objects, problems)``.

    ``problems`` names every object that is missing: skipped by the workflow
    process, or unreadable here.
    """
    if int(manifest.get("schema", 0) or 0) > OBJECTS_SCHEMA:
        return {}, [f"the run's objects are in a newer format ({manifest.get('schema')}) "
                    "than this studio reads"]
    reader = _Reader(Path(folder))
    objects: Dict[str, Any] = {}
    problems = [f"{name} ({reason})" for name, reason in
                dict(manifest.get("skipped") or {}).items()]
    try:
        for name, entry in dict(manifest.get("entries") or {}).items():
            try:
                objects[name] = reader.decode(entry)
            except Exception as exc:  # noqa: BLE001 - keep the rest of the result usable
                problems.append(f"{name} (could not be read: {exc})")
    finally:
        reader.close()
    return objects, problems


__all__ = ["OBJECTS_SCHEMA", "is_plain_data", "load_objects", "objects_folder",
           "save_objects"]
