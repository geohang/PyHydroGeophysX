"""
Climate Data Agent: daily weather at a survey site, for comparison with ERT.

Daily precipitation, minimum and maximum air temperature and reference
evapotranspiration come from the Open-Meteo historical-weather API, which serves
the ERA5 reanalysis: one HTTPS request, no API key, global coverage, and data to
within about a week of today. Evapotranspiration is FAO-56 Penman-Monteith
reference ET0 as Open-Meteo computes it.

This replaced PyDaymet. That package pinned NumPy versions the geophysics stack
cannot use, so it was reachable only through a second conda environment and a
subprocess, and it was not installed in the project environment at all - every
retrieval failed. Daymet is also limited to North America and published about a
year behind. ERA5 is coarser (0.1-0.25 degrees against Daymet's 1 km), which is
why the source is carried in the result for the report to name.

Open-Meteo data are licensed CC BY 4.0: "Weather data by Open-Meteo.com".
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base_agent import AgentResult, BaseAgent


#: Open-Meteo's historical-weather endpoint.
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

#: The Open-Meteo daily variable behind each column the package reads, in the
#: units the reports and figures assume: mm/day for prcp and pet, deg C for
#: tmin and tmax.
DAILY_VARIABLES = {
    "prcp": "precipitation_sum",
    "tmin": "temperature_2m_min",
    "tmax": "temperature_2m_max",
    "pet": "et0_fao_evapotranspiration",
}

SOURCE = "ERA5 reanalysis via the Open-Meteo historical weather API"
PET_METHOD = "FAO-56 Penman-Monteith reference evapotranspiration (ET0)"
ATTRIBUTION = "Weather data by Open-Meteo.com (CC BY 4.0)"


class ClimateDataAgent(BaseAgent):
    """Daily weather for a site, and the antecedent-moisture features ERT is read against.

    ``execute`` returns ``climate_data`` (a daily DataFrame with ``prcp``,
    ``tmin``, ``tmax`` and ``pet``), ``derived_features`` (antecedent
    precipitation totals and P - PET), ``ert_alignment`` (the rows of the survey
    days) and ``metadata``, which names the source.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai", timeout: float = 60.0):
        super().__init__("climate_data", api_key, model, llm_provider)
        self.timeout = float(timeout)
        self.default_variables = list(DAILY_VARIABLES)
        self.results: Dict[str, Any] = {}

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve the daily series for a site and period.

        Args:
            input_data: Dictionary containing:
                - coords: (x, y) in ``crs`` - longitude, latitude by default - or
                  a list of pairs, which are averaged: an ERT line spans far less
                  than one reanalysis cell
                - dates: (start_date, end_date), or a list of years
                - crs: coordinate reference system of ``coords`` (default 4326)
                - ert_timestamps: optional survey times to align the series to
                - antecedent_days: windows for antecedent precipitation totals
                - variables: optional names asked for; any the series does not
                  hold is noted
                - pet_method: a method other than Penman-Monteith is noted, not
                  computed; the series is always ET0
                - output_dir: optional folder to save the series in
                - csv_file / metadata_file: a series saved earlier, read instead

        Returns:
            Dictionary containing the series, derived features and metadata, or a
            failed AgentResult saying what was missing or refused.
        """
        self._log("Starting climate data retrieval")

        csv_file = input_data.get('csv_file')
        if csv_file:
            validation_error = self.validate_input_file(
                csv_file,
                supported_extensions=[".csv"],
                field_name="csv_file",
                max_size_mb=input_data.get("max_file_size_mb"),
            )
            if validation_error:
                return validation_error
            return self._load_from_csv(csv_file, input_data)

        dates = input_data.get('dates')
        if dates is None:
            return AgentResult(
                status="failed",
                summary="Climate data request is missing the date range.",
                data={},
                error="dates parameter is required",
                error_fix_hint="Provide dates as (start_date, end_date), for example ('2022-03-01', '2022-06-30').",
            )
        coords = input_data.get('coords')
        if coords is None:
            return AgentResult(
                status="failed",
                summary="Climate data request is missing a location.",
                data={},
                error="coords must be provided",
                error_fix_hint="Provide coords as (longitude, latitude) for the site.",
            )

        try:
            lon, lat = self._site(coords, input_data.get('crs', 4326))
            start, end = self._period(dates)
            climate_data, cell = self._fetch(lat, lon, start, end)
        except Exception as exc:  # noqa: BLE001 - reported to the caller as a failure
            return AgentResult(
                status="failed",
                summary="Climate data could not be retrieved.",
                data={},
                error=str(exc),
                error_fix_hint=("Check the site coordinates and the period, and that "
                                "this machine can reach archive-api.open-meteo.com."),
            )

        notes = []
        requested = str(input_data.get('pet_method') or 'penman_monteith')
        if requested.lower().replace('-', '_') not in ('penman_monteith', 'fao56', 'et0'):
            notes.append(f"PET is {PET_METHOD}; the requested '{requested}' method "
                         f"is not computed.")
        if str(input_data.get('time_scale', 'daily')).lower() != 'daily':
            notes.append("The climate series is daily; the requested time scale "
                         "was not applied.")
        # A variable that was asked for and is not in the series is said, not
        # dropped: Daymet-era requests still name srad, vp and dayl.
        unavailable = [str(name) for name in (input_data.get('variables') or [])
                       if str(name) not in DAILY_VARIABLES]
        if unavailable:
            notes.append(f"Not in this source's series: {', '.join(unavailable)}. "
                         f"It provides {', '.join(DAILY_VARIABLES)}.")
        missing = int(climate_data.isna().any(axis=1).sum())
        if missing:
            notes.append(f"{missing} of {len(climate_data)} days have no value in the "
                         f"reanalysis yet (the most recent days are published last).")

        metadata = {
            'dates': (start, end),
            'variables': list(DAILY_VARIABLES),
            'pet_method': PET_METHOD,
            'time_scale': 'daily',
            'source': SOURCE,
            'attribution': ATTRIBUTION,
            'site': {'longitude': lon, 'latitude': lat},
            'grid_cell': cell,
            'crs': 4326,
        }
        self.results = self._package(climate_data, metadata, input_data, notes,
                                     data_source='open-meteo')
        self._save(input_data.get('output_dir'))
        self._log("Climate data retrieval completed")
        return self.results

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------
    @staticmethod
    def _site(coords: Any, crs: Any) -> Tuple[float, float]:
        """Longitude and latitude of the site, averaging several points."""
        points = np.atleast_2d(np.asarray(coords, dtype=float))
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"coords must be (x, y) pairs, got {coords!r}")
        x, y = points.mean(axis=0)
        if str(crs).upper().replace('EPSG:', '') != '4326':
            from pyproj import Transformer
            x, y = Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
        if not (-180.0 <= x <= 180.0 and -90.0 <= y <= 90.0):
            raise ValueError(f"({x:g}, {y:g}) is not a longitude and latitude; "
                             f"give crs for projected coordinates")
        return float(x), float(y)

    @staticmethod
    def _period(dates: Any) -> Tuple[str, str]:
        """(start, end) as ISO dates, from a pair of dates or a list of years."""
        values = [dates] if isinstance(dates, str) else list(dates)
        if values and all(isinstance(v, (int, np.integer)) for v in values):
            return f"{min(values)}-01-01", f"{max(values)}-12-31"
        if len(values) != 2:
            raise ValueError(f"dates must be (start, end) or a list of years, got {dates!r}")
        start, end = (pd.Timestamp(v).strftime('%Y-%m-%d') for v in values)
        if start > end:
            raise ValueError(f"the period starts after it ends: {start} > {end}")
        return start, end

    def _fetch(self, lat: float, lon: float, start: str, end: str
               ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """One request to Open-Meteo; the daily table and the grid cell it came from."""
        import requests

        response = requests.get(OPEN_METEO_ARCHIVE, params={
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "start_date": start,
            "end_date": end,
            "daily": ",".join(DAILY_VARIABLES.values()),
            # Local calendar days, which is what survey times are recorded in.
            "timezone": "auto",
        }, timeout=self.timeout)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.status_code != 200 or payload.get("error"):
            reason = payload.get("reason") or response.text[:200]
            raise ValueError(f"Open-Meteo refused the request ({response.status_code}): {reason}")

        daily = payload.get("daily") or {}
        frame = pd.DataFrame(
            {name: daily.get(variable) for name, variable in DAILY_VARIABLES.items()},
            index=pd.DatetimeIndex(pd.to_datetime(daily.get("time") or []), name="time"),
            dtype=float,
        )
        if frame.empty:
            raise ValueError(f"Open-Meteo returned no days for {start} to {end}.")
        cell = {"latitude": payload.get("latitude"), "longitude": payload.get("longitude"),
                "elevation_m": payload.get("elevation"), "timezone": payload.get("timezone")}
        self._log(f"Retrieved {len(frame)} days for ({lat:.4f}, {lon:.4f}); grid cell at "
                  f"({cell['latitude']}, {cell['longitude']}), {cell['elevation_m']} m")
        return frame, cell

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_derived_features(climate_data: pd.DataFrame,
                                  antecedent_days: Sequence[int] = (1, 3, 7)) -> Dict[str, Any]:
        """Antecedent precipitation totals and P - PET, as columns and as series."""
        enhanced = climate_data.copy()
        features: Dict[str, Any] = {}
        if 'prcp' in enhanced:
            for days in antecedent_days:
                name = f'prcp_antecedent_{int(days)}d'
                enhanced[name] = enhanced['prcp'].rolling(window=int(days), min_periods=1).sum()
                features[name] = enhanced[name]
            if 'pet' in enhanced:
                enhanced['p_minus_pet'] = enhanced['prcp'] - enhanced['pet']
                features['p_minus_pet'] = enhanced['p_minus_pet']
        features['enhanced_data'] = enhanced
        return features

    def _align_with_ert(self, climate_data: pd.DataFrame, derived_features: Dict[str, Any],
                        ert_timestamps: Sequence[Any]) -> Dict[str, Any]:
        """The row of each survey's calendar day, or NaN outside the series.

        By day, not by nearest timestamp: a survey at 14:00 is nearer the next
        midnight, and nearest matching paired it with the next day's rain.
        """
        self._log("Aligning climate data with ERT timestamps")
        frame = derived_features.get('enhanced_data', climate_data)
        times = pd.DatetimeIndex(pd.to_datetime(list(ert_timestamps)))
        if frame is None or frame.empty or times.empty:
            return {}
        by_day = frame.copy()
        by_day.index = pd.DatetimeIndex(by_day.index).normalize()
        aligned = by_day.reindex(times.normalize())
        aligned.index = times
        return {'ert_aligned_data': aligned, 'ert_timestamps': times}

    def _package(self, climate_data: pd.DataFrame, metadata: Dict[str, Any],
                 input_data: Dict[str, Any], notes: list, data_source: str) -> Dict[str, Any]:
        derived = self._compute_derived_features(
            climate_data, antecedent_days=input_data.get('antecedent_days', [1, 3, 7]))
        alignment = None
        if input_data.get('ert_timestamps') is not None:
            alignment = self._align_with_ert(climate_data, derived, input_data['ert_timestamps'])
        return {
            'climate_data': climate_data,
            'derived_features': derived,
            'ert_alignment': alignment,
            'pet_comparison': None,
            'metadata': metadata,
            'data_source': data_source,
            'notes': notes,
        }

    # ------------------------------------------------------------------
    # files
    # ------------------------------------------------------------------
    def _save(self, output_dir: Optional[str]) -> None:
        """The series and its metadata, readable back through ``csv_file``."""
        if not output_dir or not self.results:
            return
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / 'climate_data.csv'
        self.results['climate_data'].to_csv(csv_path)
        csv_path.with_suffix('.json').write_text(
            json.dumps(self.results['metadata'], indent=2, default=str), encoding='utf-8')
        self.results['csv_file'] = str(csv_path)

    def _load_from_csv(self, csv_file: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """A series saved earlier: this agent's own file, or PyDaymet's with units in the headers."""
        self._log(f"Loading pre-fetched climate data from CSV: {csv_file}")
        climate_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        # "prcp (mm/day)" -> "prcp"
        climate_data = climate_data.rename(columns=lambda c: str(c).split('(')[0].strip())

        metadata_file = Path(input_data.get('metadata_file') or Path(csv_file).with_suffix('.json'))
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        else:
            metadata = {
                'dates': (str(climate_data.index.min().date()), str(climate_data.index.max().date())),
                'variables': [c for c in climate_data.columns if c in DAILY_VARIABLES],
                'pet_method': 'unknown',
                'time_scale': 'daily',
                'source': f'pre-fetched file {Path(csv_file).name}',
            }
        self.results = self._package(climate_data, metadata, input_data, [],
                                     data_source='pre_fetched_csv')
        self.results['csv_file'] = str(csv_file)
        self._log("Climate data loaded from CSV successfully")
        return self.results

    def _log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        print(f"[{level}] ClimateDataAgent: {message}")

    def get_climate_summary(self) -> str:
        """A plain-text summary of the last series retrieved."""
        if not self.results:
            return "No climate data retrieved yet."
        metadata = self.results.get('metadata', {})
        lines = ["=" * 60, "Climate Data Summary", "=" * 60,
                 f"\nDate Range: {metadata.get('dates')}",
                 f"Variables: {', '.join(metadata.get('variables', []))}",
                 f"PET Method: {metadata.get('pet_method')}",
                 f"Source: {metadata.get('source')}"]
        features = [key for key in (self.results.get('derived_features') or {})
                    if key != 'enhanced_data']
        if features:
            lines.append("\nDerived Features:")
            lines.extend(f"  - {key}" for key in features)
        alignment = self.results.get('ert_alignment') or {}
        if 'ert_timestamps' in alignment:
            lines.append(f"\nERT Alignment: {len(alignment['ert_timestamps'])} timestamps matched")
        lines.append("=" * 60)
        return "\n".join(lines)
