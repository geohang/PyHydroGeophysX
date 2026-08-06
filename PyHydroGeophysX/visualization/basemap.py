"""Satellite / street basemaps for survey maps drawn in projected metres.

A survey map reads far better over imagery: the reader sees at once that a line
ran along a road, across a field, or beside a river. This fetches XYZ raster
tiles and warps them onto a matplotlib axes whose units are projected metres
(UTM, state plane, a local grid), so the map keeps honest distances instead of
being redrawn in Web Mercator, where a metre is not a metre.

Deliberately built on ``requests`` + ``Pillow`` alone. ``contextily`` is the
usual answer and does more (many providers, automatic attribution), but it
brings ``rasterio`` and ``pyproj`` with it; this package's other map views need
neither, and a field laptop benefits from an on-disk tile cache more than from
another projection stack.

No projection library is needed because the caller already knows both
coordinates of every station: its projected ``(x, y)`` and its ``(lon, lat)``.
UTM and Web Mercator are both conformal, so over one survey the map between them
is a similarity transform (one rotation, one scale, one shift) to well under a
metre. :func:`fit_local_transform` recovers it by least squares from those pairs.

Tile-server terms of use apply to whatever source is selected. Esri World
Imagery and OpenStreetMap both allow light non-commercial use with attribution,
which :func:`basemap_image` returns so the caller can place it on the figure.
Keep requests modest: ``max_tiles`` bounds every call, and cached tiles are
never re-fetched.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

#: Earth radius used by the Web Mercator tile scheme (EPSG:3857).
_R = 6378137.0

#: Tile providers, keyed by the label a UI can show. ``url`` takes ``z``, ``x``
#: and ``y``; Esri numbers its rows before its columns, hence the order there.
TILE_SOURCES: Dict[str, Dict[str, Any]] = {
    "Satellite (Esri World Imagery)": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        "attribution": "Imagery © Esri, Maxar, Earthstar Geographics",
        "max_zoom": 19,
    },
    "Street map (OpenStreetMap)": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors",
        "max_zoom": 19,
    },
    "Topographic (Esri)": {
        "url": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Topo_Map/MapServer/tile/{z}/{y}/{x}"),
        "attribution": "© Esri, HERE, Garmin, USGS",
        "max_zoom": 19,
    },
}

#: A polite identity. OpenStreetMap's tile policy rejects the default
#: ``python-requests`` agent outright.
_USER_AGENT = "PyHydroGeophysX/basemap (+https://github.com/geohang/PyHydroGeophysX)"

_TILE_PIXELS = 256


def default_cache_dir() -> Path:
    """Where tiles are kept between sessions.

    Honours ``PYHYDROGEOPHYSX_TILE_CACHE`` so a shared or read-only machine can
    redirect it.
    """
    override = os.environ.get("PYHYDROGEOPHYSX_TILE_CACHE")
    root = Path(override) if override else Path.home() / ".pyhydrogeophysx" / "tilecache"
    return root


def web_mercator(lon, lat) -> "Tuple[np.ndarray, np.ndarray]":
    """Longitude / latitude in degrees to Web Mercator metres (EPSG:3857)."""
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = _R * np.radians(lon)
    y = _R * np.log(np.tan(0.25 * np.pi + 0.5 * np.radians(lat)))
    return x, y


def fit_local_transform(x, y, lon, lat) -> Optional["Tuple[complex, complex]"]:
    """Least-squares similarity from projected metres to Web Mercator metres.

    Returns ``(a, b)`` such that ``mx + i*my = a * (x + i*y) + b``. Written over
    the complex plane because a complex multiply *is* a rotation plus a scale,
    which is exactly the freedom a pair of conformal projections leaves; the fit
    is then one two-column least squares.

    Returns ``None`` when fewer than three usable stations are given or the
    residual is too large to trust, so the caller can drop the basemap rather
    than draw imagery in the wrong place.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    n = min(x.size, y.size, lon.size, lat.size)
    if n < 3:
        return None
    good = (np.isfinite(x[:n]) & np.isfinite(y[:n])
            & np.isfinite(lon[:n]) & np.isfinite(lat[:n]))
    if good.sum() < 3:
        return None
    mx, my = web_mercator(lon[:n][good], lat[:n][good])
    source = x[:n][good] + 1j * y[:n][good]
    target = mx + 1j * my
    # Centre both clouds: a raw fit on UTM coordinates of order 1e6 loses
    # precision in the offset term.
    source_mid, target_mid = source.mean(), target.mean()
    design = np.column_stack([source - source_mid, np.ones(source.size)])
    solution, *_ = np.linalg.lstsq(design, target - target_mid, rcond=None)
    a, offset = complex(solution[0]), complex(solution[1])
    b = target_mid + offset - a * source_mid
    if not (np.isfinite([a.real, a.imag, b.real, b.imag]).all() and abs(a) > 0):
        return None
    residual = np.abs((a * source + b) - target)
    span = float(np.abs(source - source_mid).max()) or 1.0
    # Over one survey a similarity holds to a fraction of a metre, most of the
    # residual being how consistently the instrument wrote its two coordinate
    # pairs rather than the projections themselves. One part in a thousand of
    # the survey's own extent means the inputs do not describe one place.
    if float(residual.max()) > max(1.0, 1e-3 * span * abs(a)):
        return None
    return a, b


def _tile_range(west: float, east: float, south: float, north: float,
                zoom: int) -> "Tuple[int, int, int, int]":
    """Inclusive tile index box covering a Web Mercator bounding box."""
    count = 2 ** int(zoom)
    span = 2.0 * math.pi * _R

    def to_index(value: float, origin: float, sign: float) -> float:
        return (sign * (value - origin) / span) * count

    x0 = int(math.floor(to_index(west, -math.pi * _R, 1.0)))
    x1 = int(math.floor(to_index(east, -math.pi * _R, 1.0)))
    y0 = int(math.floor(to_index(north, math.pi * _R, -1.0)))
    y1 = int(math.floor(to_index(south, math.pi * _R, -1.0)))
    clamp = lambda v: max(0, min(count - 1, v))  # noqa: E731
    return clamp(x0), clamp(x1), clamp(y0), clamp(y1)


def _choose_zoom(west: float, east: float, source: Dict[str, Any],
                 target_pixels: int, max_tiles: int) -> int:
    """Deepest zoom that shows the extent at ``target_pixels`` within the budget."""
    span = max(abs(float(east) - float(west)), 1e-6)
    world = 2.0 * math.pi * _R
    # Resolution wanted, in metres per pixel, then the zoom that provides it.
    wanted = span / max(int(target_pixels), 1)
    zoom = int(round(math.log2(world / (_TILE_PIXELS * wanted))))
    zoom = max(1, min(int(source.get("max_zoom", 19)), zoom))
    while zoom > 1:
        x0, x1, y0, y1 = _tile_range(west, east, 0.0, 0.0, zoom)
        if (x1 - x0 + 1) <= max_tiles:
            break
        zoom -= 1
    return zoom


def _fetch_tile(url: str, path: Path, timeout: float):
    """Return one tile as a PIL image, using the on-disk copy when present."""
    from PIL import Image

    if path.is_file() and path.stat().st_size > 0:
        try:
            with Image.open(path) as handle:
                return handle.convert("RGB").copy()
        except Exception:  # noqa: BLE001 - a truncated cache entry is refetched
            pass
    import requests

    response = requests.get(url, timeout=timeout,
                            headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    from io import BytesIO
    with Image.open(BytesIO(response.content)) as handle:
        return handle.convert("RGB").copy()


def fetch_mosaic(west: float, south: float, east: float, north: float, *,
                 source: Dict[str, Any], zoom: int,
                 cache_dir: Optional[Path] = None,
                 timeout: float = 8.0) -> "Optional[Tuple[np.ndarray, Tuple[float, float, float, float]]]":
    """Stitch the tiles covering a Web Mercator box.

    Returns ``(rgb, (west, east, south, north))`` for the mosaic's own extent,
    which is a whole number of tiles and so slightly larger than the request.
    """
    from PIL import Image

    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    name = str(source.get("name") or source.get("url", ""))
    key = "".join(character if character.isalnum() else "_" for character in name)[-60:]
    x0, x1, y0, y1 = _tile_range(west, east, south, north, zoom)
    count = 2 ** int(zoom)
    span = 2.0 * math.pi * _R
    mosaic = Image.new("RGB", ((x1 - x0 + 1) * _TILE_PIXELS,
                              (y1 - y0 + 1) * _TILE_PIXELS))
    fetched = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            url = str(source["url"]).format(z=zoom, x=tx, y=ty)
            path = cache / key / str(zoom) / str(tx) / f"{ty}.png"
            try:
                tile = _fetch_tile(url, path, timeout)
            except Exception:  # noqa: BLE001 - a missing tile leaves a gap only
                continue
            mosaic.paste(tile, ((tx - x0) * _TILE_PIXELS, (ty - y0) * _TILE_PIXELS))
            fetched += 1
    if not fetched:
        return None
    extent = (
        -math.pi * _R + span * x0 / count,
        -math.pi * _R + span * (x1 + 1) / count,
        math.pi * _R - span * (y1 + 1) / count,
        math.pi * _R - span * y0 / count,
    )
    return np.asarray(mosaic), extent


def basemap_image(x_limits: Sequence[float], y_limits: Sequence[float], *,
                  transform: "Tuple[complex, complex]",
                  source: str = "Satellite (Esri World Imagery)",
                  target_pixels: int = 900, max_tiles: int = 24,
                  cache_dir: Optional[Path] = None,
                  timeout: float = 8.0) -> Optional[Dict[str, Any]]:
    """Imagery for a projected-metre axes, warped onto that axes' own grid.

    ``transform`` comes from :func:`fit_local_transform`. The returned dict holds
    ``image`` (an RGB array ready for ``imshow``), ``extent`` in the axes' own
    units, ``attribution``, and ``zoom``. Returns ``None`` when the source is
    unknown, nothing could be fetched, or a dependency is missing, so the caller
    simply draws no basemap.
    """
    spec = TILE_SOURCES.get(str(source))
    if spec is None:
        return None
    spec = {**spec, "name": str(source)}
    a, b = transform
    x_min, x_max = float(min(x_limits)), float(max(x_limits))
    y_min, y_max = float(min(y_limits)), float(max(y_limits))
    if not (x_max > x_min and y_max > y_min):
        return None

    corners = np.array([x_min + 1j * y_min, x_min + 1j * y_max,
                        x_max + 1j * y_min, x_max + 1j * y_max])
    projected = a * corners + b
    west, east = float(projected.real.min()), float(projected.real.max())
    south, north = float(projected.imag.min()), float(projected.imag.max())

    zoom = _choose_zoom(west, east, spec, target_pixels, max_tiles)
    try:
        mosaic = fetch_mosaic(west, south, east, north, source=spec, zoom=zoom,
                              cache_dir=cache_dir, timeout=timeout)
    except Exception:  # noqa: BLE001 - requests / Pillow absent, or no network
        return None
    if mosaic is None:
        return None
    rgb, (m_west, m_east, m_south, m_north) = mosaic
    rows, cols = rgb.shape[0], rgb.shape[1]

    # Warp: walk the destination grid in the axes' own metres, carry each pixel
    # to Web Mercator, and sample. Nearest neighbour is right here, because the
    # zoom was chosen so one source pixel is about one destination pixel.
    height = int(min(target_pixels, max(64, round(
        target_pixels * (y_max - y_min) / (x_max - x_min)))))
    width = int(target_pixels)
    xs = np.linspace(x_min, x_max, width)
    ys = np.linspace(y_max, y_min, height)          # top row first, for imshow
    grid = xs[None, :] + 1j * ys[:, None]
    target = a * grid + b
    col = (target.real - m_west) / (m_east - m_west) * cols
    row = (m_north - target.imag) / (m_north - m_south) * rows
    col = np.clip(col.astype(int), 0, cols - 1)
    row = np.clip(row.astype(int), 0, rows - 1)
    return {
        "image": rgb[row, col],
        "extent": (x_min, x_max, y_min, y_max),
        "attribution": str(spec.get("attribution", "")),
        "zoom": int(zoom),
        "source": str(source),
    }


__all__ = [
    "TILE_SOURCES",
    "basemap_image",
    "default_cache_dir",
    "fetch_mosaic",
    "fit_local_transform",
    "web_mercator",
]
