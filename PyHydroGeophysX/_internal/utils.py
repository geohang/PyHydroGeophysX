"""Small dependency-free helpers shared by scientific and UI adapters."""

from __future__ import annotations

import datetime as _dt


def noop(*_args, **_kwargs) -> None:
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


__all__ = ["noop", "utc_now", "velocity_of"]
