"""Standalone Daymet fetch script used by ``ClimateDataAgent.fetch_climate_data_with_conda``.

This script runs inside a minimal helper conda environment (python + pandas +
pydaymet, optionally pyet), so it must not import PyHydroGeophysX. It reads a
JSON configuration and writes a CSV under ``./data/climate/`` in the working
directory, which is where the calling agent looks for the result.

Config keys:
    coords:      [lon, lat] pair in EPSG:4326 (Daymet expects lon/lat order)
    dates:       ["YYYY-MM-DD", "YYYY-MM-DD"] range, or a list of years
    variables:   optional list; default ["prcp", "tmin", "tmax", "srad", "vp", "dayl"]
    time_scale:  optional; "daily" (default), "monthly", or "annual"
    region:      optional; "na" (default), "hi", or "pr"
    pet:         optional PET method name (e.g. "penman_monteith"); omit to skip
    output_name: optional CSV file name; default "climate_data.csv"

Usage::

    python fetch_climate_data.py --config climate_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_VARIABLES = ["prcp", "tmin", "tmax", "srad", "vp", "dayl"]


def _parse_dates(raw):
    """Accept a [start, end] date-string pair or a list of integer years."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and any(
        isinstance(item, str) and "-" in item for item in raw
    ):
        return (str(raw[0]), str(raw[1]))
    return [int(item) for item in raw]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Daymet climate data to CSV.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: could not read config '{config_path}': {exc}", file=sys.stderr)
        return 2

    coords = config.get("coords")
    dates = config.get("dates")
    if not coords or not dates:
        print("ERROR: config must provide 'coords' ([lon, lat]) and 'dates'.", file=sys.stderr)
        return 2

    try:
        coords = tuple(float(v) for v in list(coords)[:2])
        dates = _parse_dates(dates)
    except Exception as exc:
        print(f"ERROR: invalid coords/dates in config: {exc}", file=sys.stderr)
        return 2

    variables = config.get("variables") or DEFAULT_VARIABLES
    time_scale = config.get("time_scale", "daily")
    region = config.get("region", "na")
    pet_method = config.get("pet")

    try:
        import pydaymet
    except ImportError as exc:
        print(f"ERROR: pydaymet is not installed in this environment: {exc}", file=sys.stderr)
        return 3

    print(f"Fetching Daymet data at {coords} for {dates} ({time_scale}, region={region})...")
    try:
        data = pydaymet.get_bycoords(
            coords,
            dates,
            variables=variables,
            crs=4326,
            time_scale=time_scale,
            region=region,
            pet=pet_method,
        )
    except Exception as exc:
        print(f"ERROR: Daymet fetch failed: {exc}", file=sys.stderr)
        return 4

    out_dir = Path("data") / "climate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / str(config.get("output_name") or "climate_data.csv")
    data.to_csv(out_path)
    print(f"Saved {len(data)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
