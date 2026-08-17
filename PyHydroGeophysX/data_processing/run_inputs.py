"""One compact on-disk format for what a run keeps about its inputs.

A run used to record its inputs by copying whatever the user had selected, so a
run cost as much disk as the survey it read: a TEMcompany project folder runs to
hundreds of megabytes and every inversion took another copy of it. What a run
needs to be reproducible is the data the page imported, which is arrays and a
little metadata, and not the acquisition those arrays were parsed out of.

The format is the one :mod:`PyHydroGeophysX.data_processing.joint_io` already
used for its non-PyGIMLi observations, lifted out so every module writes the
same thing: a single ``.npz`` holding the arrays, plus a JSON manifest stored
under ``__manifest__`` that describes the structure they came from. Nesting is
preserved, so a loader's return value can be handed straight to
:func:`save_container` and comes back the same shape.

``np.savez_compressed`` does the shrinking. Field data is mostly smooth float
arrays and compresses well, and no pickle is involved, so the file crosses a
process boundary into an isolated workflow run the same way a plain array does.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

#: Bumped only when an older reader could misread a newer file. Additive
#: metadata does not need it.
SCHEMA_VERSION = 1

_MANIFEST_KEY = "__manifest__"


def pack(value: Any, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Split ``value`` into a JSON-safe skeleton and a flat array table."""
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
            "items": {str(key): pack(item, arrays) for key, item in value.items()},
        }
    if isinstance(value, (list, tuple)):
        return {"type": "list", "items": [pack(item, arrays) for item in value]}
    raise TypeError(
        f"Unsupported run-input value {type(value).__name__}; "
        "materialize it to arrays and JSON scalars first."
    )


def unpack(node: Mapping[str, Any], archive: Any) -> Any:
    """Rebuild what :func:`pack` took apart."""
    node_type = node["type"]
    if node_type == "array":
        return np.asarray(archive[str(node["key"])])
    if node_type == "scalar":
        return node.get("value")
    if node_type == "mapping":
        return {
            str(key): unpack(value, archive)
            for key, value in dict(node.get("items") or {}).items()
        }
    if node_type == "list":
        return [unpack(value, archive) for value in node.get("items") or []]
    raise ValueError(f"Unknown run-input manifest node {node_type!r}.")


def save_container(
    destination: str | Path,
    payload: Any,
    *,
    kind: str,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write ``payload`` as one compressed container and return its path.

    ``kind`` names what the file holds (``"em_soundings"``,
    ``"hydrology_arrays"``, …) so a reader can refuse a file it was not meant to
    open. ``meta`` carries anything a reader needs before it unpacks the arrays,
    such as the method the data was imported as.
    """
    target = Path(destination).with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": str(kind),
        "meta": dict(meta or {}),
        "payload": pack(payload, arrays),
    }
    np.savez_compressed(
        target,
        **{_MANIFEST_KEY: np.asarray(json.dumps(manifest, sort_keys=True))},
        **arrays,
    )
    return target


def read_manifest(source: str | Path) -> Dict[str, Any]:
    """Return the manifest without unpacking any array."""
    with np.load(Path(source), allow_pickle=False) as archive:
        return json.loads(str(archive[_MANIFEST_KEY].item()))


def load_container(source: str | Path, *, kind: Optional[str] = None) -> Any:
    """Load a container written by :func:`save_container`.

    Passing ``kind`` checks the file is the one the caller expects, which turns
    a mismatched path into a clear error instead of a downstream KeyError.
    """
    path = Path(source)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive[_MANIFEST_KEY].item()))
        stored = str(manifest.get("kind", ""))
        if kind is not None and stored != str(kind):
            raise ValueError(
                f"{path.name} holds {stored or 'an unnamed kind'!r}, expected {kind!r}."
            )
        return unpack(manifest["payload"], archive)


def _same(left: Any, right: Any) -> bool:
    """Structural equality that also answers for arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_array, right_array = np.asarray(left), np.asarray(right)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        return bool(np.array_equal(left_array, right_array, equal_nan=
                                   left_array.dtype.kind == "f"))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _split(items: list) -> Dict[str, Any]:
    """Describe a list of like values as shared parts and varying parts.

    A survey's soundings agree on almost everything: the system geometry, the
    protocol, the gate times, the survey-wide position and line arrays. Storing
    each sounding whole repeats all of that once per station, which is how a
    container for two soundings came out larger than the raw files it replaced.
    This walks the values together so anything identical is stored once, and
    arrays that differ only in their values get stacked into one array rather
    than becoming one zip member each.
    """
    first = items[0]
    if all(_same(first, item) for item in items[1:]):
        return {"node": "shared", "value": first}
    if (isinstance(first, np.ndarray) and first.dtype != object
            and all(isinstance(item, np.ndarray) and item.shape == first.shape
                    and item.dtype == first.dtype for item in items)):
        return {"node": "stack", "values": items}
    if (isinstance(first, Mapping)
            and all(isinstance(item, Mapping) and set(item) == set(first)
                    for item in items)):
        return {
            "node": "mapping",
            "keys": {
                str(key): _split([item[key] for item in items]) for key in first
            },
        }
    return {"node": "items", "values": items}


def _pack_split(node: Mapping[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    kind = node["node"]
    if kind == "shared":
        return {"node": "shared", "value": pack(node["value"], arrays)}
    if kind == "stack":
        key = f"stack_{len(arrays)}"
        arrays[key] = np.stack([np.asarray(value) for value in node["values"]])
        return {"node": "stack", "key": key}
    if kind == "mapping":
        return {
            "node": "mapping",
            "keys": {k: _pack_split(v, arrays) for k, v in node["keys"].items()},
        }
    return {"node": "items", "values": [pack(v, arrays) for v in node["values"]]}


def _unpack_split(node: Mapping[str, Any], archive: Any, index: int) -> Any:
    kind = node["node"]
    if kind == "shared":
        return unpack(node["value"], archive)
    if kind == "stack":
        return np.asarray(archive[str(node["key"])][index])
    if kind == "mapping":
        return {
            key: _unpack_split(value, archive, index)
            for key, value in dict(node.get("keys") or {}).items()
        }
    return unpack(node["values"][index], archive)


def save_sequence_container(
    destination: str | Path,
    items: list,
    *,
    kind: str,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Write a list of like-shaped payloads, storing what they share once."""
    if not items:
        raise ValueError("A sequence container needs at least one item.")
    target = Path(destination).with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": str(kind),
        "meta": dict(meta or {}),
        "sequence": _pack_split(_split(list(items)), arrays),
        "count": len(items),
    }
    np.savez_compressed(
        target,
        **{_MANIFEST_KEY: np.asarray(json.dumps(manifest, sort_keys=True))},
        **arrays,
    )
    return target


def load_sequence_item(
    source: str | Path, index: int, *, kind: Optional[str] = None
) -> Any:
    """Return one item from a container written by :func:`save_sequence_container`."""
    path = Path(source)
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(str(archive[_MANIFEST_KEY].item()))
        stored = str(manifest.get("kind", ""))
        if kind is not None and stored != str(kind):
            raise ValueError(
                f"{path.name} holds {stored or 'an unnamed kind'!r}, expected {kind!r}."
            )
        if "sequence" not in manifest:
            raise ValueError(f"{path.name} is not a sequence container.")
        count = int(manifest.get("count", 0))
        position = max(0, min(int(index), count - 1))
        return _unpack_split(manifest["sequence"], archive, position)


def sequence_length(source: str | Path) -> int:
    """How many items :func:`load_sequence_item` can return."""
    return int(read_manifest(source).get("count", 0))


def save_file_bundle(
    destination: str | Path,
    files: Mapping[str, str | Path],
    *,
    kind: str,
    meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Store a set of files as one compressed container, keyed by file name.

    Some inputs cannot be reduced to arrays on the way in: a PyGIMLi mesh and a
    BERT data file are read back by PyGIMLi's own loaders, which want a real
    path. Their bytes are kept verbatim, so what a run saves here is the
    compression and the file count rather than a change of representation.
    ASCII data files shrink several-fold; ``.npy`` arrays shrink by whatever
    deflate finds in them.

    Pair with :func:`expand_file_bundle`, which puts the files back under their
    original names so the reader sees the directory it expects.
    """
    target = Path(destination).with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    entries = []
    for name, source in files.items():
        payload = Path(source).read_bytes()
        key = f"file_{len(arrays)}"
        arrays[key] = np.frombuffer(payload, dtype=np.uint8)
        entries.append({"name": str(name), "key": key, "size": len(payload)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": str(kind),
        "meta": dict(meta or {}),
        "files": entries,
    }
    np.savez_compressed(
        target,
        **{_MANIFEST_KEY: np.asarray(json.dumps(manifest, sort_keys=True))},
        **arrays,
    )
    return target


def expand_file_bundle(source: str | Path, destination: str | Path) -> Path:
    """Write a bundle back out as loose files and return the directory."""
    out_dir = Path(destination)
    out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(Path(source), allow_pickle=False) as archive:
        manifest = json.loads(str(archive[_MANIFEST_KEY].item()))
        entries = list(manifest.get("files") or [])
        if not entries:
            raise ValueError(f"{Path(source).name} is not a file bundle.")
        for entry in entries:
            # Names come from a manifest this package wrote, but a bundle is a
            # file like any other: refuse anything that would land outside.
            name = Path(str(entry["name"])).name
            (out_dir / name).write_bytes(
                np.asarray(archive[str(entry["key"])], dtype=np.uint8).tobytes()
            )
    return out_dir


def bundle_file_names(source: str | Path) -> list:
    """The names :func:`expand_file_bundle` would write."""
    return [
        Path(str(entry["name"])).name
        for entry in read_manifest(source).get("files") or []
    ]


def is_file_bundle(source: str | Path) -> bool:
    """True when ``source`` is a bundle written by :func:`save_file_bundle`."""
    if not is_container(source):
        return False
    try:
        return bool(read_manifest(source).get("files"))
    except (OSError, ValueError, KeyError):  # pragma: no cover - unreadable file
        return False


def is_container(source: str | Path) -> bool:
    """True when ``source`` is one of our containers.

    Reads the zip directory rather than the arrays, so this stays cheap enough
    to sit in a loader's dispatch chain in front of the file sniffers.
    """
    path = Path(source)
    if path.suffix.lower() != ".npz" or not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return f"{_MANIFEST_KEY}.npy" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


__all__ = [
    "SCHEMA_VERSION",
    "bundle_file_names",
    "expand_file_bundle",
    "is_container",
    "is_file_bundle",
    "save_file_bundle",
    "load_container",
    "load_sequence_item",
    "pack",
    "read_manifest",
    "save_container",
    "save_sequence_container",
    "sequence_length",
    "unpack",
]
