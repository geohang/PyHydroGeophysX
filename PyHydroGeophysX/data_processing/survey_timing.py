"""Acquisition times of a monitoring sequence, read from the files themselves.

A time-lapse survey is a sequence of files whose only difference is *when* they
were recorded, and almost every field crew records that in the file name. If the
software cannot read it, the sequence silently degrades into "survey 1, 2, 3 ...",
the panels are headed by an index, and nobody can tell a one-hour gap from a
one-month gap on the figure.

This module reads the timestamps and says where each one came from. Filenames are
tried first, against a list of patterns; a pattern is accepted only when it parses
*every* file in the set and yields distinct, increasing times, which is what lets
ambiguous layouts (``YYMMDD`` against ``DDMMYY``) be resolved by the sequence
rather than by guesswork. A file header is read next, and the filesystem
modification time only if the caller opts in. When nothing parses, the fallback is
the old ``1..n`` index - reported as such, so it is visible rather than assumed.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "SurveyTiming",
    "format_duration",
    "parse_timestamp",
    "timestamp_from_header",
    "survey_timing",
]

_SECONDS_PER_DAY = 86400.0

#: Filename timestamp patterns, most specific first. Each entry is
#: ``(name, compiled regex, builder)``; the builder turns the match groups into a
#: datetime and raises ValueError on an impossible date.
_PATTERNS: List[Tuple[str, re.Pattern, Any]] = []


def _register(name: str, pattern: str, builder) -> None:
    _PATTERNS.append((name, re.compile(pattern), builder))


def _ymd_hms(year, month, day, hour=None, minute=None, second=None) -> _dt.datetime:
    return _dt.datetime(int(year), int(month), int(day), int(hour or 0),
                        int(minute or 0), int(second or 0))


_register(
    "iso_datetime",
    r"(?<!\d)(\d{4})[-_.](\d{2})[-_.](\d{2})[T_ -]+(\d{2})[:_.h-]?(\d{2})(?:[:_.-]?(\d{2}))?(?!\d)",
    lambda g: _ymd_hms(*g),
)
_register(
    "compact_datetime",
    r"(?<!\d)(\d{4})(\d{2})(\d{2})[T_ -]?(\d{2})(\d{2})(\d{2})?(?!\d)",
    lambda g: _ymd_hms(*g),
)
_register(
    "us_datetime",
    r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})[T_ -]+(\d{2})[:_.h-]?(\d{2})(?:[:_.-]?(\d{2}))?(?!\d)",
    lambda g: _ymd_hms(g[2], g[0], g[1], g[3], g[4], g[5]),
)
_register(
    "short_datetime",
    r"(?<!\d)(\d{2})(\d{2})(\d{2})[T_ -](\d{2})(\d{2})(\d{2})?(?!\d)",
    lambda g: _ymd_hms(2000 + int(g[0]), g[1], g[2], g[3], g[4], g[5]),
)
_register(
    "epoch_seconds",
    r"(?<!\d)(1[0-9]{9})(?!\d)",
    lambda g: _dt.datetime.fromtimestamp(int(g[0]), _dt.timezone.utc).replace(tzinfo=None),
)
_register(
    "iso_date",
    r"(?<!\d)(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})(?!\d)",
    lambda g: _ymd_hms(g[0], g[1], g[2]),
)
_register(
    "us_date",
    r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)",
    lambda g: _ymd_hms(g[2], g[0], g[1]),
)


def format_duration(seconds: Optional[float]) -> str:
    """A duration a reader can take in at a glance: ``45 s``, ``1 h 02 min``, ``3 d 06 h``.

    Reporting every interval in decimal days is what makes an hourly sequence look
    like a rounding error; this keeps the unit attached to the size of the gap.
    """
    if seconds is None:
        return ""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value != value:  # NaN
        return ""
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value < 60.0:
        return f"{sign}{value:.0f} s"
    if value < 3600.0:
        minutes = value / 60.0
        return f"{sign}{minutes:.0f} min" if minutes >= 10 else f"{sign}{minutes:.1f} min"
    if value < _SECONDS_PER_DAY:
        hours = int(value // 3600)
        minutes = int(round((value - hours * 3600) / 60.0))
        if minutes == 60:
            hours, minutes = hours + 1, 0
        return f"{sign}{hours} h {minutes:02d} min"
    days = int(value // _SECONDS_PER_DAY)
    hours = int(round((value - days * _SECONDS_PER_DAY) / 3600.0))
    if hours == 24:
        days, hours = days + 1, 0
    return f"{sign}{days} d {hours:02d} h"


def parse_timestamp(text: str, patterns: Optional[Sequence[str]] = None
                    ) -> Optional[Tuple[_dt.datetime, str]]:
    """First timestamp in ``text``, with the name of the pattern that read it.

    ``patterns`` restricts the search to named patterns, which is how a whole file
    set is forced to agree on one interpretation.
    """
    for name, regex, builder in _PATTERNS:
        if patterns is not None and name not in patterns:
            continue
        for match in regex.finditer(str(text)):
            try:
                return builder(match.groups()), name
            except (ValueError, TypeError, OverflowError, OSError):
                continue  # an impossible date: keep scanning this pattern
    return None


def _timestamps_by_pattern(stems: Sequence[str]) -> Tuple[List[Optional[_dt.datetime]], str]:
    """Parse every stem with one pattern, preferring one that reads the whole set.

    A set of files shares a naming convention, so the right pattern is the one that
    works on all of them. Falling back to per-file parsing would let one pattern
    read half the sequence as 2021 and another half as 2012.
    """
    for name, _regex, _builder in _PATTERNS:
        stamps = [parse_timestamp(stem, patterns=(name,)) for stem in stems]
        if any(item is None for item in stamps):
            continue
        values = [item[0] for item in stamps]
        if len(set(values)) == len(values):
            return values, name
    # No single pattern covers everything; take what each name gives on its own so
    # a partially dated set still reports the dates it has.
    mixed: List[Optional[_dt.datetime]] = []
    for stem in stems:
        found = parse_timestamp(stem)
        mixed.append(found[0] if found else None)
    return mixed, "mixed"


def timestamp_from_header(path: str, max_lines: int = 80) -> Optional[_dt.datetime]:
    """Best-effort acquisition time from the top of a data file.

    Most instrument exports carry the date in their header (Syscal, ABEM, Res2DInv
    and the E4D survey files all do, in their own layouts). Rather than a parser
    per vendor, the first timestamp-shaped text in the header wins - enough to
    recover a sequence whose filenames were renamed on download.
    """
    try:
        with open(path, "r", errors="ignore") as handle:
            head = [next(handle, "") for _ in range(int(max_lines))]
    except (OSError, ValueError):
        return None
    for line in head:
        if not line.strip():
            continue
        found = parse_timestamp(line)
        if found is not None:
            return found[0]
        slashed = re.search(
            r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?:[T ,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
            line)
        if slashed:
            a, b, year, hour, minute, second = slashed.groups()
            # Month first unless that is impossible, which is the only way to tell
            # 03/04/2024 apart without knowing the vendor's locale.
            month, day = (a, b) if int(a) <= 12 else (b, a)
            try:
                return _ymd_hms(year, month, day, hour, minute, second)
            except ValueError:
                continue
    return None


@dataclass
class SurveyTiming:
    """Acquisition times of an ordered monitoring sequence.

    ``times`` stays in elapsed days from the first survey - the unit the inversion
    and every existing export already use - while ``timestamps`` keeps the absolute
    times so durations can be reported in units a reader recognises.
    """

    files: List[str] = field(default_factory=list)
    timestamps: List[Optional[_dt.datetime]] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    source: str = "index"
    pattern: str = ""
    unit: str = ""

    @property
    def dated(self) -> bool:
        """True when every survey has a real acquisition time."""
        return bool(self.timestamps) and all(t is not None for t in self.timestamps)

    @property
    def intervals(self) -> List[float]:
        """Seconds between consecutive surveys; empty when they are not dated."""
        if not self.dated or len(self.timestamps) < 2:
            return []
        return [(b - a).total_seconds()
                for a, b in zip(self.timestamps[:-1], self.timestamps[1:])]

    @property
    def total_seconds(self) -> Optional[float]:
        if not self.dated or len(self.timestamps) < 2:
            return None
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()

    def summary(self) -> str:
        """One line naming the span, the typical gap and where the times came from."""
        n = len(self.files)
        if not self.dated:
            if self.source != "index" and self.times:
                return (f"{n} surveys with numeric times only "
                        f"({min(self.times):g} to {max(self.times):g}"
                        f"{' ' + self.unit if self.unit else ''}), from "
                        f"{self.source}. Without acquisition timestamps the real "
                        f"duration between surveys cannot be reported.")
            return (f"{n} surveys with no readable acquisition time; using a "
                    f"sequential 1..{n} index. Panels will be headed \"Time step N\" "
                    f"and every gap is treated as equal.")
        gaps = self.intervals
        span = format_duration(self.total_seconds)
        head = (f"{n} surveys, {self.timestamps[0]:%Y-%m-%d %H:%M} to "
                f"{self.timestamps[-1]:%Y-%m-%d %H:%M} (span {span}), "
                f"times from the {self.source}")
        if not gaps:
            return head + "."
        median = statistics.median(gaps)
        text = head + f". Interval: median {format_duration(median)}"
        if max(gaps) - min(gaps) > 0.02 * max(abs(median), 1.0):
            text += (f", {format_duration(min(gaps))} to {format_duration(max(gaps))}"
                     f" - the sampling is irregular")
        return text + "."

    def rows(self) -> List[Tuple[Any, ...]]:
        """Table rows for the times CSV: index, file, timestamp, elapsed, gap."""
        out: List[Tuple[Any, ...]] = []
        for index, path in enumerate(self.files):
            stamp = self.timestamps[index] if index < len(self.timestamps) else None
            previous = self.timestamps[index - 1] if index > 0 else None
            if stamp is not None and previous is not None:
                gap = (stamp - previous).total_seconds()
                gap_days: Any = gap / _SECONDS_PER_DAY
                gap_text = format_duration(gap)
            else:
                gap_days, gap_text = "", ""
            out.append((
                index,
                Path(path).name,
                stamp.isoformat(sep=" ") if stamp is not None else "",
                float(self.times[index]) if index < len(self.times) else "",
                gap_days,
                gap_text,
                self.source,
            ))
        return out

    @staticmethod
    def csv_header() -> List[str]:
        return ["index", "file", "timestamp", "elapsed[d]", "interval[d]",
                "interval", "time_source"]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable view for the run result and the report."""
        gaps = self.intervals
        return {
            "source": self.source,
            "pattern": self.pattern,
            "dated": self.dated,
            "unit": self.unit,
            "timestamps": [t.isoformat(sep=" ") if t else "" for t in self.timestamps],
            "times": [float(t) for t in self.times],
            "labels": list(self.labels),
            "interval_seconds": [float(g) for g in gaps],
            "interval_text": [format_duration(g) for g in gaps],
            "median_interval_seconds": (
                float(statistics.median(gaps)) if gaps else None),
            "median_interval": (
                format_duration(statistics.median(gaps)) if gaps else ""),
            "total_seconds": self.total_seconds,
            "total_duration": format_duration(self.total_seconds),
            "summary": self.summary(),
        }


def _labels_for(stamps: Sequence[Optional[_dt.datetime]]) -> List[str]:
    """Date labels, carrying the clock time only when the dates alone repeat."""
    dated = [s for s in stamps if s is not None]
    show_time = len({s.date() for s in dated}) != len(dated)
    out = []
    for index, stamp in enumerate(stamps):
        if stamp is None:
            out.append(str(index + 1))
        else:
            out.append(stamp.strftime("%Y-%m-%d %H:%M" if show_time else "%Y-%m-%d"))
    return out


def survey_timing(files: Sequence[str], *, allow_header: bool = True,
                  allow_mtime: bool = False,
                  timestamps: Optional[Sequence[Optional[_dt.datetime]]] = None
                  ) -> SurveyTiming:
    """Acquisition times for an ordered list of survey files.

    Args:
        files: data files in acquisition order.
        allow_header: read the file header when the name carries no timestamp.
        allow_mtime: fall back to the filesystem modification time. Off by
            default - a copied or re-exported file carries the time of the copy,
            so this is a choice the user has to make, not a silent default.
        timestamps: caller-supplied times that override everything else, e.g. a
            table the user edited in the interface.

    Returns:
        A :class:`SurveyTiming`. When no source yields a complete, strictly
        increasing set of times it falls back to the ``1..n`` index and says so in
        ``source``.
    """
    paths = [str(f) for f in files]
    if not paths:
        return SurveyTiming()

    source, pattern = "index", ""
    stamps: List[Optional[_dt.datetime]] = [None] * len(paths)

    if timestamps is not None and len(timestamps) == len(paths):
        stamps = [t if isinstance(t, _dt.datetime) else None for t in timestamps]
        source, pattern = "supplied times", "supplied"
    else:
        stamps, pattern = _timestamps_by_pattern([Path(p).stem for p in paths])
        if all(s is not None for s in stamps):
            source = "file names"
        if allow_header and any(s is None for s in stamps):
            from_names = sum(s is not None for s in stamps)
            filled = [s if s is not None else timestamp_from_header(p)
                      for s, p in zip(stamps, paths)]
            if all(s is not None for s in filled):
                stamps = filled
                # Naming which files needed the header, because a set that is half
                # named and half sniffed is worth a second look.
                source = "file headers" if from_names == 0 else "file names + headers"
        if allow_mtime and any(s is None for s in stamps):
            filled = []
            for s, p in zip(stamps, paths):
                if s is not None:
                    filled.append(s)
                    continue
                try:
                    filled.append(_dt.datetime.fromtimestamp(os.path.getmtime(p)))
                except OSError:
                    filled.append(None)
            if all(s is not None for s in filled):
                stamps = filled
                source = "file modification times"

    if all(s is not None for s in stamps):
        origin = min(s for s in stamps if s is not None)
        times = [(s - origin).total_seconds() / _SECONDS_PER_DAY for s in stamps]
        if len(set(times)) == len(times):
            return SurveyTiming(files=paths, timestamps=list(stamps), times=times,
                                labels=_labels_for(stamps), source=source,
                                pattern=pattern, unit="d")
        # Identical stamps cannot order a sequence; the index at least can.

    n = len(paths)
    return SurveyTiming(
        files=paths, timestamps=[None] * n,
        times=[float(i + 1) for i in range(n)],
        labels=[str(i + 1) for i in range(n)],
        source="index", pattern="", unit="")
