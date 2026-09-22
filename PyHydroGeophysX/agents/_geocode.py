"""Turning a place name into coordinates, without asking the model for them.

Climate retrieval needs a longitude and latitude: :mod:`pydaymet` samples a
gridded product at a point, so "somewhere in Wyoming" is not enough. The
temptation is to ask the LLM for the coordinates directly, and that is the one
thing not to do — models state plausible, wrong coordinates with complete
confidence, and a silently misplaced site produces a climate series for the
wrong valley that nothing downstream can catch.

So the work is split. The model does what it is good at: reading "this is from
the Medicine Bow site in Wyoming" out of a sentence, in any language. This
module does what a gazetteer is good at: turning that name into a position,
from a service whose answer can be checked and cited.

Nominatim (OpenStreetMap) is the default because it needs no API key. Its usage
policy requires an identifying User-Agent and at most one request per second,
both of which :func:`geocode_place` honours; it is fine for the occasional
lookup a workflow run makes, and not for bulk querying.
"""

import json
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

#: Nominatim asks that clients identify themselves, and blocks those that do not.
USER_AGENT = "PyHydroGeophysX/0.4 (https://github.com/geohang/PyHydroGeophysX)"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

#: Their policy is one request per second; a workflow makes one or two.
_MIN_INTERVAL_S = 1.0
_last_call = [0.0]


def _http_get_json(url: str, timeout: float = 10.0) -> Any:
    """GET ``url`` and parse JSON, or return None. Never raises."""
    try:
        import requests
    except ImportError:  # pragma: no cover - requests ships with the app
        return None
    try:
        elapsed = time.monotonic() - _last_call[0]
        if elapsed < _MIN_INTERVAL_S:
            time.sleep(_MIN_INTERVAL_S - elapsed)
        _last_call[0] = time.monotonic()
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:  # noqa: BLE001 - offline, blocked, rate-limited: all "no answer"
        return None


def geocode_place(place: str, fetch: Optional[Callable[[str], Any]] = None
                  ) -> Optional[Dict[str, Any]]:
    """Look up ``place`` and return its position and the name that was matched.

    Parameters
    ----------
    place : str
        A place name as a person would write it, e.g. ``"Medicine Bow,
        Wyoming"``. Coordinates are not accepted here; this is the step that
        produces them.
    fetch : callable, optional
        Takes a URL and returns parsed JSON. Defaults to an HTTP GET against
        Nominatim; injected in tests and available for sites behind a proxy.

    Returns
    -------
    dict or None
        ``{"coords": (lon, lat), "matched_name": str, "source": str}``, or None
        when nothing was found or the service could not be reached. The matched
        name is returned so the caller can show *what the service thought you
        meant* — the failure mode here is a confident match on the wrong place,
        and it is only catchable if the name is visible.

    Raises
    ------
    None

    Examples
    --------
    >>> reply = [{"lon": "-106.2", "lat": "41.4", "display_name": "Medicine Bow, WY"}]
    >>> geocode_place("Medicine Bow", fetch=lambda url: reply)
    {'coords': (-106.2, 41.4), 'matched_name': 'Medicine Bow, WY', 'source': 'nominatim'}
    >>> geocode_place("nowhere at all", fetch=lambda url: []) is None
    True
    """
    name = (place or "").strip()
    if not name:
        return None
    getter = fetch or _http_get_json
    url = f"{NOMINATIM_URL}?{urlencode({'q': name, 'format': 'json', 'limit': 1})}"
    payload = getter(url)
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, list) or not payload:
        return None
    first = payload[0]
    try:
        lon, lat = float(first["lon"]), float(first["lat"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return {"coords": (lon, lat),
            "matched_name": str(first.get("display_name", name)),
            "source": "nominatim"}


def coords_from_config(config: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Coordinates already present in ``config``, in ``(lon, lat)`` order.

    Checks the places a user or an earlier stage may have put them, so a
    hand-filled ``climate_config.coords`` is honoured without a lookup.

    Parameters
    ----------
    config : dict
        Workflow configuration.

    Returns
    -------
    tuple of float or None
        ``(lon, lat)``, or None when the configuration does not carry a position.

    Raises
    ------
    None

    Examples
    --------
    >>> coords_from_config({"climate_config": {"coords": [-106.2, 41.4]}})
    (-106.2, 41.4)
    >>> coords_from_config({"site_info": {"coordinates": [-106.2, 41.4]}})
    (-106.2, 41.4)
    >>> coords_from_config({"crs": "local"}) is None
    True
    """
    candidates = [
        (config.get("climate_config") or {}).get("coords"),
        (config.get("site_info") or {}).get("coords"),
        (config.get("site_info") or {}).get("coordinates"),
        config.get("coords"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            try:
                lon, lat = float(candidate[0]), float(candidate[1])
            except (TypeError, ValueError):
                continue
            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                return (lon, lat)
    return None
