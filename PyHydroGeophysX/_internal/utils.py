"""Small dependency-free helpers shared by scientific and UI adapters."""

from __future__ import annotations

import datetime as _dt
import json as _json
import math as _math
from collections.abc import Mapping as _Mapping
from pathlib import Path as _Path


def noop(*_args, **_kwargs) -> None:
    return None


def json_safe(value, *, array_limit=None):
    """A view of ``value`` that ``json.dumps(..., allow_nan=False)`` accepts.

    Numbers stay numbers, non-finite floats become None (the bare ``NaN`` some
    encoders write is rejected by every strict parser), paths and unknown
    objects become strings, mappings get string keys, and arrays and NumPy
    scalars become lists and Python scalars. NumPy is never imported: arrays
    are recognised by their ``shape`` and ``dtype``.

    Parameters
    ----------
    value : object
        What to convert; nested containers are converted throughout.
    array_limit : int, optional
        Arrays with more elements are summarised as
        ``{"$array": {"shape", "dtype", "size"}}`` instead of listed. None lists
        every array.

    Returns
    -------
    object
        Built from dict, list, str, int, float, bool and None only.

    Examples
    --------
    >>> json_safe({1: float("nan"), "p": _Path("run.json"), "t": (1, 2.5)})
    {'1': None, 'p': 'run.json', 't': [1, 2.5]}
    """
    if isinstance(value, float):
        return value if _math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, _Path):
        return str(value)
    if isinstance(value, _Mapping):
        return {str(key): json_safe(item, array_limit=array_limit)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, array_limit=array_limit) for item in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = [int(part) for part in value.shape]
            size = int(value.size)
            if (array_limit is None or size <= array_limit) and hasattr(value, "tolist"):
                return json_safe(value.tolist(), array_limit=array_limit)
            return {"$array": {"shape": shape, "dtype": str(value.dtype), "size": size}}
        except Exception:  # noqa: BLE001 - an array-like that is not one
            pass
    if hasattr(value, "item"):
        try:
            return json_safe(value.item(), array_limit=array_limit)
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def parse_json_object(reply):
    """The JSON object in a model's reply, or None.

    Models wrap JSON in prose and in code fences however they are asked not
    to, so the text from the first ``{`` to the last ``}`` is parsed. The
    agents, the run controller, its recovery step and the reader adapters
    each carried a copy of this.

    Parameters
    ----------
    reply : str
        The model's reply. Anything else returns None.

    Returns
    -------
    dict or None
        The parsed object; None when there is none or it is not valid JSON.

    Examples
    --------
    >>> parse_json_object('Sure:\\n```json\\n{"tool": "invert_ert"}\\n```')
    {'tool': 'invert_ert'}
    >>> parse_json_object('no json here') is None
    True
    >>> parse_json_object(None) is None
    True
    """
    if not isinstance(reply, str):
        return None
    text = reply.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return _json.loads(text[start:end + 1])
    except ValueError:
        return None


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def velocity_of(manager):
    """Velocity from a travel-time result, whichever engine produced it.

    A PyGIMLi ``TravelTimeManager`` exposes ``velocity`` (and raises on some
    older versions); the in-house solver returns a manager-shaped shim whose
    ``model`` already holds velocity. Callers that only knew the first shape
    broke the moment the second became the default, so the lookup lives here
    rather than being repeated at each display and export site.
    """
    for name in ("velocity", "model"):
        try:
            value = getattr(manager, name, None)
        except Exception:  # noqa: BLE001 - older PyGIMLi raises from the property
            continue
        if value is not None:
            return value
    raise AttributeError(
        f"{type(manager).__name__} exposes neither velocity nor model.")


__all__ = ["json_safe", "noop", "parse_json_object", "utc_now", "velocity_of"]
