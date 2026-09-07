#!/usr/bin/env python3
"""Collect download, traffic, and visitor statistics for the documentation site.

The script writes ``docs/source/_static/stats/usage_stats.json``, which the
"Usage and Downloads" page renders client-side. Only the Python standard library
is required, so the docs runner needs no extra dependencies.

Data sources
------------
PyPI installs
    ClickHouse hosts a public, read-only mirror of the PyPI download logs
    (https://clickpy.clickhouse.com). The per-country tables carry an
    ``installer`` column, which matters here: most raw PyPI "downloads" are
    mirror bots such as bandersnatch that would place a package in whichever
    data center the mirror runs in. Counting only real installer clients
    (pip, uv, poetry, ...) is what pypistats.org reports and is the honest
    number to publish.

GitHub
    Stars, forks, and watchers come from the public repository endpoint. Views
    and clones come from the traffic endpoint, which needs a token with push
    access; in Actions the built-in ``GITHUB_TOKEN`` is enough. Without a token
    the traffic block is omitted and the page hides those tiles.

GoatCounter (optional)
    Docs-site visitors by country. Set ``GOATCOUNTER_SITE`` (the subdomain) and
    ``GOATCOUNTER_TOKEN``. Without them the visitor map is omitted and the page
    shows a short setup note instead.

Usage
-----
    python tools/fetch_usage_stats.py
    python tools/fetch_usage_stats.py --refresh-basemap   # also rebuild the world outline
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
STATS_DIR = REPO_ROOT / "docs" / "source" / "_static" / "stats"

PYPI_PROJECT = "pyhydrogeophysx"
GITHUB_REPO = "geohang/PyHydroGeophysX"

CLICKHOUSE_URL = "https://sql-clickhouse.clickhouse.com/"
CLICKHOUSE_USER = "demo"

# PyPI logs an "installer" string per download. Two of the three buckets below
# count as downloads and are reported as one number; the third is excluded.
#
# PACKAGE_MANAGERS is an install command: pip, uv, poetry, and friends.
#
# AGENT_CLIENTS is everything that fetches the distribution directly over HTTP
# without a package manager: LLM coding agents working in a sandbox, notebook
# environments, CI scripts, and clicks on the PyPI web page. The empty installer
# string dominates this bucket and is what most agent sandboxes report. Someone
# wanting the package is what a download measures, and that holds whether the
# install command came from a person or from an agent acting for one, so these
# are counted alongside the package managers rather than split out. Their
# geography (US, CA, GB, CN, DE, HK, FR, SE, JP, KR, NL, IE, ...) also looks like
# people rather than infrastructure.
#
# MIRROR_CLIENTS is index-mirroring and proxy-caching infrastructure. One
# bandersnatch mirror pulls every release regardless of whether anyone wants it,
# and it reports the location of its own data center, so counting it would both
# inflate the total and put the package on the wrong part of the map. This bucket
# is excluded and the excluded count is published alongside the total.
PACKAGE_MANAGERS = (
    "pip",
    "uv",
    "poetry",
    "pdm",
    "pipenv",
    "hatch",
    "rye",
    "conda",
    "pex",
    "bazel",
    "setuptools",
)

AGENT_CLIENTS = (
    "",
    "Browser",
    "requests",
    "httpx",
    "urllib",
    "urllib3",
    "aiohttp",
    "curl",
    "Wget",
    "OS",
)

MIRROR_CLIENTS = (
    "bandersnatch",
    "Nexus",
    "devpi",
    "Artifactory",
    "proxpi",
    "pulp",
)

BY_COUNTRY_TABLE = "pypi.pypi_downloads_per_day_by_version_by_installer_by_type_by_country"

# Territory codes folded into another entry before anything is counted or drawn.
# The map and the ranking then show one combined figure, and the folded
# territory's polygon is filled with the colour of the entry it belongs to.
# This mapping is written into the statistics JSON so that usage-stats.js reads
# the same table for its live query; keeping it in one place stops the snapshot
# and the live figures from disagreeing.
TERRITORY_MERGES = {"TW": "CN"}

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_110m_admin_0_countries.geojson"
)

USER_AGENT = "PyHydroGeophysX-docs-stats/1.0 (+https://github.com/geohang/PyHydroGeophysX)"


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def clickhouse(query: str) -> list[dict[str, Any]]:
    """Run one read-only query against the public ClickHouse PyPI mirror."""
    params = urllib.parse.urlencode(
        {"user": CLICKHOUSE_USER, "default_format": "JSON", "query": query}
    )
    payload = json.loads(http_get(f"{CLICKHOUSE_URL}?{params}").decode("utf-8"))
    return payload.get("data", [])


def bucket_of(installer: str) -> str:
    """Map a PyPI installer string onto one of the three traffic buckets."""
    if installer in PACKAGE_MANAGERS:
        return "installs"
    if installer in MIRROR_CLIENTS:
        return "mirrors"
    if installer in AGENT_CLIENTS:
        return "agents"
    # An installer we have not seen before is far more likely to be a new client
    # or agent runtime than a new mirror, so it counts rather than disappears.
    return "agents"


def _sorted_desc(counts: dict[str, int]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def fold_territories(counts: dict[str, int]) -> dict[str, int]:
    """Merge the TERRITORY_MERGES codes into the entry they belong to."""
    folded: dict[str, int] = {}
    for code, value in counts.items():
        target = TERRITORY_MERGES.get(code, code)
        folded[target] = folded.get(target, 0) + value
    return folded


# --------------------------------------------------------------------------- #
# PyPI downloads
# --------------------------------------------------------------------------- #


def collect_pypi() -> dict[str, Any]:
    """Pull PyPI download logs and split them into the three traffic buckets.

    Every query groups by ``installer`` so the split happens here rather than in
    SQL. That keeps the bucket definitions in one place and lets the page offer
    "all downloads" and "package-manager installs" as two layers of one dataset.
    """
    where = f"project = '{PYPI_PROJECT}'"

    rows_country = clickhouse(
        f"SELECT country_code AS k, installer, sum(count) AS v FROM {BY_COUNTRY_TABLE} "
        f"WHERE {where} AND country_code != '' GROUP BY k, installer"
    )
    rows_day = clickhouse(
        f"SELECT toString(date) AS k, installer, sum(count) AS v FROM {BY_COUNTRY_TABLE} "
        f"WHERE {where} GROUP BY k, installer"
    )
    rows_version = clickhouse(
        f"SELECT version AS k, installer, sum(count) AS v FROM {BY_COUNTRY_TABLE} "
        f"WHERE {where} GROUP BY k, installer"
    )
    rows_type = clickhouse(
        f"SELECT type AS k, installer, sum(count) AS v FROM {BY_COUNTRY_TABLE} "
        f"WHERE {where} GROUP BY k, installer"
    )
    def split(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        """Fold rows into {bucket: {key: count}} plus a counted "all" bucket."""
        out: dict[str, dict[str, int]] = {"installs": {}, "agents": {}, "mirrors": {}, "all": {}}
        for row in rows:
            key = row["k"]
            count = int(row["v"])
            bucket = bucket_of(row["installer"])
            out[bucket][key] = out[bucket].get(key, 0) + count
            if bucket != "mirrors":
                out["all"][key] = out["all"].get(key, 0) + count
        return out

    country = split(rows_country)
    day = split(rows_day)
    version = split(rows_version)
    file_type = split(rows_type)

    today = dt.date.today()

    def window(daily: dict[str, int], days: int) -> int:
        cutoff = today - dt.timedelta(days=days)
        return sum(n for d, n in daily.items() if dt.date.fromisoformat(d) >= cutoff)

    def to_month(daily: dict[str, int]) -> dict[str, int]:
        months: dict[str, int] = {}
        for d, n in daily.items():
            months[d[:7]] = months.get(d[:7], 0) + n
        return dict(sorted(months.items()))

    # A download is a download. Whether a person typed the install command or an
    # LLM agent ran it on their behalf does not change that someone wanted the
    # package, so the two are reported as one number. Only mirror and proxy
    # infrastructure is held out, and that exclusion is disclosed on the page.
    daily = day["all"]
    by_country = fold_territories(country["all"])

    return {
        "total": sum(daily.values()),
        "last_30d": window(daily, 30),
        "last_90d": window(daily, 90),
        "last_365d": window(daily, 365),
        "countries_reached": len(by_country),
        "first_day": min(daily) if daily else None,
        "last_day": max(daily) if daily else None,
        "excluded_mirror_downloads": sum(day["mirrors"].values()),
        "by_country": _sorted_desc(by_country),
        "by_day": dict(sorted(daily.items())),
        "by_month": to_month(daily),
        "by_version": _sorted_desc(version["all"]),
        "by_file_type": _sorted_desc(file_type["all"]),
    }


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


def collect_github() -> dict[str, Any]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repo = json.loads(http_get(f"https://api.github.com/repos/{GITHUB_REPO}", headers).decode())
    out: dict[str, Any] = {
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "watchers": repo.get("subscribers_count"),
        "open_issues": repo.get("open_issues_count"),
        "pushed_at": repo.get("pushed_at"),
    }

    if not token:
        return out

    # The traffic endpoints need push access and cover a rolling 14-day window.
    for name, key in (("views", "views_14d"), ("clones", "clones_14d")):
        try:
            data = json.loads(
                http_get(f"https://api.github.com/repos/{GITHUB_REPO}/traffic/{name}", headers).decode()
            )
        except urllib.error.HTTPError as exc:
            print(f"  note: GitHub traffic/{name} unavailable ({exc.code})", file=sys.stderr)
            continue
        out[key] = {
            "total": data.get("count"),
            "uniques": data.get("uniques"),
            "daily": {
                item["timestamp"][:10]: item["count"] for item in data.get(name, [])
            },
        }
    return out


# --------------------------------------------------------------------------- #
# GoatCounter (docs-site visitors)
# --------------------------------------------------------------------------- #


def collect_visitors() -> dict[str, Any] | None:
    site = os.environ.get("GOATCOUNTER_SITE")
    token = os.environ.get("GOATCOUNTER_TOKEN")
    if not site or not token:
        return None

    base = f"https://{site}.goatcounter.com/api/v0"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    window = f"start={start.isoformat()}&end={end.isoformat()}"

    def call(path: str) -> dict[str, Any]:
        return json.loads(http_get(f"{base}/{path}", headers).decode())

    out: dict[str, Any] = {"window_start": start.isoformat(), "window_end": end.isoformat()}

    # GoatCounter's headline metric is visitors rather than raw pageviews, so
    # /stats/total returns "total" (visitors, events included) and there is no
    # separate unique count to read. Per-path "count" is likewise visitors.
    try:
        totals = call(f"stats/total?{window}")
        out["visitors"] = totals.get("total")
        out["events"] = totals.get("total_events")
    except urllib.error.HTTPError as exc:
        print(f"  note: GoatCounter totals unavailable ({exc.code})", file=sys.stderr)

    try:
        locations = call(f"stats/locations?{window}&limit=250")
        by_country: dict[str, int] = {}
        names: dict[str, str] = {}
        for row in locations.get("stats", []):
            # Location ids are "US" or "US-CA"; fold any region into its country.
            code = str(row.get("id") or "")[:2].upper()
            if len(code) == 2 and code.isalpha():
                by_country[code] = by_country.get(code, 0) + int(row.get("count") or 0)
                names.setdefault(code, str(row.get("name") or code))
        folded = fold_territories(by_country)
        out["by_country"] = _sorted_desc(folded)
        out["countries_reached"] = len(folded)
    except urllib.error.HTTPError as exc:
        print(f"  note: GoatCounter locations unavailable ({exc.code})", file=sys.stderr)

    # daily=true collapses the per-hour arrays, which keeps the response small.
    try:
        pages = call(f"stats/hits?{window}&daily=true&limit=20")
        out["top_pages"] = [
            {
                "path": hit.get("path"),
                "title": hit.get("title") or "",
                "count": hit.get("count"),
                "event": bool(hit.get("event")),
            }
            for hit in pages.get("hits", [])
        ]
    except urllib.error.HTTPError as exc:
        print(f"  note: GoatCounter hits unavailable ({exc.code})", file=sys.stderr)

    return out


# --------------------------------------------------------------------------- #
# World outline for the maps
# --------------------------------------------------------------------------- #


def build_basemap(destination: Path) -> None:
    """Download Natural Earth 110m countries and write a compact GeoJSON.

    Natural Earth is public domain. Coordinates are rounded to two decimals
    (about 1 km, well below what a world-scale map resolves) and repeated points
    are dropped, which cuts the file to roughly a fifth of the source size.
    """
    print(f"  downloading {NATURAL_EARTH_URL}")
    source = json.loads(http_get(NATURAL_EARTH_URL, timeout=180).decode("utf-8"))

    def thin(ring: list[list[float]]) -> list[list[float]] | None:
        out: list[list[float]] = []
        for lon, lat in ring:
            point = [round(float(lon), 2), round(float(lat), 2)]
            if not out or out[-1] != point:
                out.append(point)
        if len(out) < 4:
            return None
        if out[0] != out[-1]:
            out.append(out[0])
        return out

    features = []
    for feature in source.get("features", []):
        props = feature.get("properties", {})
        iso = props.get("ISO_A2_EH") or props.get("ISO_A2") or ""
        name = props.get("NAME_EN") or props.get("NAME") or ""
        if iso in ("", "-99") or name == "Antarctica":
            continue

        geometry = feature.get("geometry") or {}
        kind = geometry.get("type")
        if kind == "Polygon":
            polygons = [geometry.get("coordinates", [])]
        elif kind == "MultiPolygon":
            polygons = geometry.get("coordinates", [])
        else:
            continue

        kept = []
        for polygon in polygons:
            rings = [thinned for ring in polygon if (thinned := thin(ring))]
            if rings:
                kept.append(rings)
        if not kept:
            continue

        features.append(
            {
                "type": "Feature",
                "properties": {"iso": iso.upper(), "name": name},
                "geometry": {"type": "MultiPolygon", "coordinates": kept},
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")),
        encoding="utf-8",
    )
    size_kb = destination.stat().st_size / 1024
    print(f"  wrote {destination.name}: {len(features)} countries, {size_kb:.0f} KB")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-basemap",
        action="store_true",
        help="also rebuild the world outline GeoJSON (rarely needed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=STATS_DIR / "usage_stats.json",
        help="path of the statistics JSON to write",
    )
    args = parser.parse_args()

    basemap = args.out.parent / "world-110m.geo.json"
    if args.refresh_basemap or not basemap.exists():
        print("Building the world basemap")
        build_basemap(basemap)

    stats: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "project": PYPI_PROJECT,
        "repo": GITHUB_REPO,
        "buckets": {
            "package_managers": list(PACKAGE_MANAGERS),
            "agent_clients": [name or "(unreported)" for name in AGENT_CLIENTS],
            "mirror_clients": list(MIRROR_CLIENTS),
        },
        "territory_merges": dict(TERRITORY_MERGES),
    }

    print("Querying PyPI download logs")
    try:
        stats["pypi"] = collect_pypi()
        pypi = stats["pypi"]
        print(
            f"  {pypi['total']} downloads from {pypi['countries_reached']} countries "
            f"({pypi['excluded_mirror_downloads']} mirror downloads excluded)"
        )
    except Exception as exc:  # noqa: BLE001 - one bad source must not break the build
        print(f"  PyPI stats unavailable: {exc}", file=sys.stderr)
        stats["pypi"] = None

    print("Querying GitHub")
    try:
        stats["github"] = collect_github()
        print(f"  {stats['github'].get('stars')} stars, {stats['github'].get('forks')} forks")
    except Exception as exc:  # noqa: BLE001
        print(f"  GitHub stats unavailable: {exc}", file=sys.stderr)
        stats["github"] = None

    print("Querying docs-site analytics")
    try:
        stats["visitors"] = collect_visitors()
        if stats["visitors"] is None:
            print("  skipped: GOATCOUNTER_SITE / GOATCOUNTER_TOKEN not set")
        else:
            print(
                f"  {stats['visitors'].get('visitors')} visitors from "
                f"{stats['visitors'].get('countries_reached')} countries"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  visitor stats unavailable: {exc}", file=sys.stderr)
        stats["visitors"] = None

    if stats["pypi"] is None and stats["github"] is None and stats["visitors"] is None:
        print("Every source failed; leaving the existing JSON in place.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
