"""Acquisition times of a monitoring sequence, read from the files themselves.

A time-lapse survey is a sequence of files whose only difference is *when* they
were recorded, and almost every field crew records that in the file name. If the
software cannot read it, the sequence silently degrades into "survey 1, 2, 3 ...",
the panels are headed by an index, and nobody can tell a one-hour gap from a
one-month gap on the figure.

This module reads the timestamps and says where each one came from. Filenames are
tried first, against a list of patterns; a pattern is accepted only when it parses
*every* file in the set and yields distinct times, which is what lets ambiguous
layouts (``YYMMDD`` against ``DDMMYY``, month against day first) be resolved by
the sequence rather than by guesswork: of the readings that date every name, the
one whose times increase in file order, and then the most compact, is kept - and
the summary says when another reading was possible. A file header is read next,
and the filesystem modification time only if the caller opts in. When nothing
parses, the fallback is the old ``1..n`` index - reported as such, so it is
visible rather than assumed.

Every survey also keeps where its own time came from (``SurveyTiming.sources``),
because a set dated half by name and half by header is worth a second look, and
the one file whose name lacks the time is the one to rename.

A file whose format has no field for the time - a BERT / pyGIMLi unified data
file - can carry it as a comment on its first line, before the electrode count::

    # date: 2026-01-12 05:50:38

Every ERT reader the studio uses skips that line as a comment (tested on the
shipped BERT survey with pyGIMLi, ResIPy and this package's own reader). It has to
be the first line: between the count and the ``# x z`` line, pyGIMLi takes a
comment for the list of columns.
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
    "format_time",
    "parse_timestamp",
    "reciprocal_pairs",
    "timestamp_from_header",
    "survey_timing",
]

#: Where one survey's time came from, as ``SurveyTiming.sources`` records it.
FROM_NAME = "file name"
FROM_HEADER = "file header"
FROM_MTIME = "file modified time"
FROM_SUPPLIED = "supplied"

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
# Each ambiguous layout is registered once per reading, the day-first one right
# after its rival, so a lone string still reads the way it always did and only a
# whole set of names (see _timestamps_by_pattern) can choose between them.
_DASHED_DATETIME = (r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})[T_ -]+(\d{2})[:_.h-]?(\d{2})"
                    r"(?:[:_.-]?(\d{2}))?(?!\d)")
_SHORT_DATETIME = r"(?<!\d)(\d{2})(\d{2})(\d{2})[T_ -](\d{2})(\d{2})(\d{2})?(?!\d)"
_DASHED_DATE = r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](\d{4})(?!\d)"
_register(
    "us_datetime",
    _DASHED_DATETIME,
    lambda g: _ymd_hms(g[2], g[0], g[1], g[3], g[4], g[5]),
)
_register(
    "dmy_datetime",
    _DASHED_DATETIME,
    lambda g: _ymd_hms(g[2], g[1], g[0], g[3], g[4], g[5]),
)
_register(
    "short_datetime",
    _SHORT_DATETIME,
    lambda g: _ymd_hms(2000 + int(g[0]), g[1], g[2], g[3], g[4], g[5]),
)
_register(
    "short_dmy_datetime",
    _SHORT_DATETIME,
    lambda g: _ymd_hms(2000 + int(g[2]), g[1], g[0], g[3], g[4], g[5]),
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
    _DASHED_DATE,
    lambda g: _ymd_hms(g[2], g[0], g[1]),
)
_register(
    "dmy_date",
    _DASHED_DATE,
    lambda g: _ymd_hms(g[2], g[1], g[0]),
)

#: Readings of the same digits in a different order, grouped: a set of names
#: that more than one of them reads is the ambiguous case the sequence resolves.
_RIVALS: Dict[str, str] = {
    "us_datetime": "dashed_datetime", "dmy_datetime": "dashed_datetime",
    "short_datetime": "short_datetime", "short_dmy_datetime": "short_datetime",
    "us_date": "dashed_date", "dmy_date": "dashed_date",
}

#: How an ambiguous reading is named to the user.
_LAYOUTS: Dict[str, str] = {
    "us_datetime": "month first (MM-DD-YYYY)", "dmy_datetime": "day first (DD-MM-YYYY)",
    "us_date": "month first (MM-DD-YYYY)", "dmy_date": "day first (DD-MM-YYYY)",
    "short_datetime": "year first (YYMMDD)", "short_dmy_datetime": "day first (DDMMYY)",
}


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


def format_time(stamp: Any, seconds: bool = False) -> str:
    """An acquisition time as a heading shows it: ``2026-01-12 05:50``.

    ``stamp`` is a datetime or the ISO text a run records. A time of exactly
    midnight is what a date-only name parses to, so it is shown as the date alone
    rather than claiming a clock time nobody recorded.
    """
    if isinstance(stamp, str):
        try:
            stamp = _dt.datetime.fromisoformat(stamp.strip())
        except ValueError:
            return stamp.strip()
    if not isinstance(stamp, _dt.datetime):
        return ""
    if stamp.time() == _dt.time(0, 0):
        return f"{stamp:%Y-%m-%d}"
    return stamp.strftime("%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M")


def _listed(names: Sequence[str], limit: int = 3) -> str:
    """``a.dat, b.dat and c.dat``, or the first ``limit`` and how many more."""
    names = list(names)
    if len(names) > limit:
        return f"{', '.join(names[:limit])} and {len(names) - limit} more"
    if len(names) > 1:
        return f"{', '.join(names[:-1])} and {names[-1]}"
    return names[0] if names else ""


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


def _read_all(stems: Sequence[str], name: str) -> Optional[List[_dt.datetime]]:
    """Every stem read with one pattern; None if it misses one or repeats a time."""
    stamps = [parse_timestamp(stem, patterns=(name,)) for stem in stems]
    if any(item is None for item in stamps):
        return None
    values = [item[0] for item in stamps]
    return values if len(set(values)) == len(values) else None


def _choose_reading(readings: List[Tuple[str, List[_dt.datetime]]]
                    ) -> Tuple[str, List[_dt.datetime], str]:
    """The reading of an ambiguous set of names to keep, and what to say about it.

    ``readings`` holds ``(pattern, times)`` for each rival that reads every name.
    Files come in acquisition order, so a reading under which the times increase
    is preferred; of those left, the most compact wins, because daily surveys read
    the wrong way round become monthly ones (01-11 ... 09-11-2021: 8 d day first,
    243 d month first). Neither is proof, so whenever a second reading was
    possible the note says which, instead of letting the choice pass unseen.
    """
    distinct: List[Tuple[str, List[_dt.datetime]]] = []
    for name, values in readings:
        if all(values != kept for _, kept in distinct):
            distinct.append((name, values))    # 01-01 and 02-02 read alike either way
    if len(distinct) == 1:
        return distinct[0][0], distinct[0][1], ""
    if len(distinct[0][1]) == 1:
        # One name has no sequence to decide by: say both readings, and how a
        # name avoids the question.
        name, values = distinct[0]
        others = "; ".join(f"{_LAYOUTS.get(other, other)}, as {other_values[0]:%Y-%m-%d}"
                           for other, other_values in distinct[1:])
        return name, values, (
            f"Ambiguous date: the name also reads {others}. It was read "
            f"{_LAYOUTS.get(name, name)}; a name that starts with the year "
            f"(YYYY-MM-DD) can be read only one way.")

    def span(values: List[_dt.datetime]) -> float:
        return (max(values) - min(values)).total_seconds()

    increasing = [item for item in distinct
                  if all(b > a for a, b in zip(item[1], item[1][1:]))]
    pool = increasing or distinct
    name, values = min(pool, key=lambda item: span(item[1]))   # ties: listed first
    in_order = [item[1] for item in increasing]
    rivals = []
    for other, other_values in distinct:
        if other == name:
            continue
        if increasing and other_values not in in_order:
            rivals.append(f"{_LAYOUTS.get(other, other)}, which puts the files "
                          f"out of order")
        else:
            rivals.append(f"{_LAYOUTS.get(other, other)}, spanning "
                          f"{format_duration(span(other_values))}")
    why = ("the only one in acquisition order" if len(increasing) == 1
           else "the more compact sequence")
    note = (f"Ambiguous dates: every name also reads {'; '.join(rivals)}. The "
            f"{_LAYOUTS.get(name, name)} reading was kept as {why}. Names that "
            f"start with the year (YYYY-MM-DD) can be read only one way.")
    return name, values, note


def _timestamps_by_pattern(stems: Sequence[str]
                           ) -> Tuple[List[Optional[_dt.datetime]], str, str]:
    """Parse every stem with one pattern, preferring one that reads the whole set.

    A set of files shares a naming convention, so the right pattern is the one that
    works on all of them. Falling back to per-file parsing would let one pattern
    read half the sequence as 2021 and another half as 2012. Where rival readings
    of the same digits both work (month or day first), :func:`_choose_reading`
    picks one from the sequence. Returns ``(times, pattern, note)``.
    """
    tried = set()
    for name, _regex, _builder in _PATTERNS:
        family = _RIVALS.get(name, name)
        if family in tried:
            continue
        tried.add(family)
        readings = []
        for rival, _r, _b in _PATTERNS:
            if _RIVALS.get(rival, rival) == family:
                values = _read_all(stems, rival)
                if values is not None:
                    readings.append((rival, values))
        if readings:
            chosen, values, note = _choose_reading(readings)
            return values, chosen, note
    # No single pattern covers everything; take what each name gives on its own so
    # a partially dated set still reports the dates it has.
    mixed: List[Optional[_dt.datetime]] = []
    for stem in stems:
        found = parse_timestamp(stem)
        mixed.append(found[0] if found else None)
    return mixed, "mixed", ""


#: ``key,value`` header rows that name when the survey was acquired. They win over
#: the first date-shaped text, which in a Subsurface Insights export is the
#: firmware's build date - weeks before any survey the file records.
_ACQUISITION_KEYS = ("system_datetime", "gps_datetime")

#: Header rows whose date is not an acquisition time at all.
_NOT_ACQUISITION_KEYS = ("code_version_date",)

#: A header line that names the acquisition time, behind an optional comment
#: mark: ``# date: 2026-01-12 05:50:38``, ``#time=2026-01-12T05:50:38``,
#: ``% acquired 2026-01-12 05:50``. The documented way to date a file whose
#: format has no field for it (see the module docstring); taken before any other
#: date-shaped text in the header.
_ACQUISITION_LINE = re.compile(
    r"^\s*(?:#+|//|[;%!*]+)?\s*"
    r"(?:acquisition[ _-]?(?:date|time)|acquired|measured|recorded|"
    r"start[ _-]?(?:date|time)|date[ _-]?time|timestamp|date|time)"
    r"\s*[:=,]?\s*(?P<value>\S.*)$",
    re.IGNORECASE)

#: One plain number: what a measurement row is made of.
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

#: Years an ERT survey can have been recorded in. A date outside them in a
#: header is digits that happen to fit the pattern, not an acquisition time.
_PLAUSIBLE_YEARS = (1950, 2100)


def _header_key(line: str) -> str:
    return line.split(",", 1)[0].strip().lower()


def _is_measurement_row(line: str) -> bool:
    """True for a row of plain numbers with a decimal among them.

    Such a row is a measurement or an electrode, never a date, and its decimals
    fit the date patterns all too well: the electrode at x = 36.76100516 in the
    shipped BERT survey read as 7610-05-16, which dated every undated BERT file
    by its own coordinates.
    """
    tokens = [t for t in re.split(r"[\s,;]+", line.strip()) if t]
    return bool(tokens) and all(_NUMBER.fullmatch(t) for t in tokens) and any(
        "." in t or "e" in t.lower() for t in tokens)


def _stamp_in(text: str) -> Optional[_dt.datetime]:
    """The first plausible acquisition time in one header line, or None."""
    found = parse_timestamp(text)
    stamp = found[0] if found is not None else None
    if stamp is None:
        slashed = re.search(
            r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?:[T ,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
            text)
        if slashed:
            a, b, year, hour, minute, second = slashed.groups()
            # Month first unless that is impossible, which is the only way to tell
            # 03/04/2024 apart without knowing the vendor's locale.
            month, day = (a, b) if int(a) <= 12 else (b, a)
            try:
                stamp = _ymd_hms(year, month, day, hour, minute, second)
            except ValueError:
                stamp = None
    low, high = _PLAUSIBLE_YEARS
    return stamp if stamp is not None and low <= stamp.year <= high else None


def timestamp_from_header(path: str, max_lines: int = 80) -> Optional[_dt.datetime]:
    """Best-effort acquisition time from the top of a data file.

    Most instrument exports carry the date in their header (Syscal, ABEM, Res2DInv
    and the E4D survey files all do, in their own layouts). Rather than a parser
    per vendor, the first timestamp-shaped text in the header wins - enough to
    recover a sequence whose filenames were renamed on download - except where the
    header names its acquisition time outright (``system_datetime``, then a
    ``# date: ...`` line), which is taken first, and a software build date, which
    is never taken. Rows of plain numbers are measurements and are not read.
    """
    try:
        with open(path, "r", errors="ignore") as handle:
            head = [next(handle, "") for _ in range(int(max_lines))]
    except (OSError, ValueError):
        return None
    for line in head:
        if _header_key(line) in _ACQUISITION_KEYS:
            stamp = _stamp_in(line)
            if stamp is not None:
                return stamp
    for line in head:
        named = _ACQUISITION_LINE.match(line)
        if named:
            stamp = _stamp_in(named.group("value"))
            if stamp is not None:
                return stamp
    for line in head:
        if (not line.strip() or _header_key(line) in _NOT_ACQUISITION_KEYS
                or _is_measurement_row(line)):
            continue
        stamp = _stamp_in(line)
        if stamp is not None:
            return stamp
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
    #: Set when the names also read another way (month or day first) and the
    #: sequence, not the names, decided; ``summary()`` carries it.
    note: str = ""
    #: Where each file's own time was found: ``FROM_NAME``, ``FROM_HEADER``,
    #: ``FROM_MTIME``, ``FROM_SUPPLIED``, or "" for none. Kept when the set falls
    #: back to the index, so the files that lack a time can be named.
    sources: List[str] = field(default_factory=list)
    #: The time each file gave, kept - unlike ``timestamps`` - when the set falls
    #: back to the index, so files that share a time can be named.
    found: List[Optional[_dt.datetime]] = field(default_factory=list)

    def undated_files(self) -> List[str]:
        """Names of the files no time could be read for."""
        sources = self.sources or [""] * len(self.files)
        return [Path(f).name for f, s in zip(self.files, sources) if not s]

    def shared_times(self) -> List[Tuple[_dt.datetime, List[str]]]:
        """``(time, names)`` for every time two or more files gave."""
        groups: Dict[_dt.datetime, List[str]] = {}
        for path, stamp in zip(self.files, self.found):
            if stamp is not None:
                groups.setdefault(stamp, []).append(Path(path).name)
        return [(stamp, names) for stamp, names in groups.items() if len(names) > 1]

    def describe(self, index: int) -> str:
        """Where survey ``index`` got its time, as one or two plain sentences."""
        if not (0 <= index < len(self.files)):
            return ""
        origin = self.sources[index] if index < len(self.sources) else ""
        stamp = self.timestamps[index] if index < len(self.timestamps) else None
        if stamp is not None:
            when = format_time(stamp, seconds=True)
            if origin == FROM_HEADER:
                return f"Time {when}, read from the file header."
            if origin == FROM_MTIME:
                return (f"Time {when}, taken from the file's modified time: its name "
                        f"and header carry none.")
            if origin == FROM_SUPPLIED:
                return f"Time {when}, as entered."
            return f"Time {when}, read from the file name." + self._reading(index, stamp)
        if not origin:
            return "No time found in this file's name or header." + (
                " Until it has one, the whole list is numbered 1, 2, 3 … instead of "
                "dated." if len(self.files) > 1 else "")
        where = {FROM_HEADER: "file header", FROM_MTIME: "file's modified time"}.get(
            origin, "file name")
        found = self.found[index] if index < len(self.found) else None
        head = f"Time {format_time(found, seconds=True)}, read from the {where}"
        name = Path(self.files[index]).name
        twins = next((names for when, names in self.shared_times()
                      if when == found), [])
        if twins:
            return (f"{head}: the same time as {_listed([n for n in twins if n != name])}. "
                    f"Two surveys cannot share a time, so the list is numbered 1, 2, "
                    f"3 … until one of them is fixed.")
        missing = len(self.undated_files())
        if missing:
            return (f"{head}, but {missing} other file(s) have none, so the list is "
                    f"numbered 1, 2, 3 … until those are fixed.")
        return (f"{head}, but other files share a time (named under the list), so "
                f"the list is numbered 1, 2, 3 … until those are fixed.")

    def _reading(self, index: int, stamp: _dt.datetime) -> str:
        """Which way a name that also reads month or day first was read, if so."""
        stem = Path(self.files[index]).stem
        name = self.pattern
        if name not in _RIVALS:
            found = parse_timestamp(stem)
            name = found[1] if found is not None else ""
        if name not in _RIVALS:
            return ""
        for rival, _regex, _builder in _PATTERNS:
            if rival == name or _RIVALS.get(rival) != _RIVALS[name]:
                continue
            other = parse_timestamp(stem, patterns=(rival,))
            if other is not None and other[0] != stamp:
                return (f" It was read {_LAYOUTS[name]}; it could also be read "
                        f"{_LAYOUTS[rival]}, as {other[0]:%Y-%m-%d}. A name that "
                        f"starts with the year ({stamp:%Y-%m-%d_%H-%M-%S}) can be "
                        f"read only one way.")
        return ""

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
            # A series is dated only when every survey is: part dated and part
            # numbered would place surveys on two different axes. What blocks it
            # is named, file by file, so the names can be fixed.
            fallback = (f"Panels will be headed \"Time step N\" and every gap is "
                        f"treated as equal.")
            undated = self.undated_files()
            if undated and len(undated) < n:
                return (f"{n} surveys, numbered 1..{n}: no time could be read for "
                        f"{len(undated)} of them ({_listed(undated)}). The other "
                        f"{n - len(undated)} are dated, but a series is dated only "
                        f"when every file is. {fallback}")
            shared = self.shared_times()
            if shared and not undated:
                clash = "; ".join(f"{_listed(names)} share "
                                  f"{format_time(stamp, seconds=True)}"
                                  for stamp, names in shared[:3])
                more = f" (and {len(shared) - 3} more times)" if len(shared) > 3 else ""
                return (f"{n} surveys, numbered 1..{n}: {clash}{more}, and two "
                        f"surveys cannot share a time. {fallback}")
            return (f"{n} surveys with no readable acquisition time; using a "
                    f"sequential 1..{n} index. {fallback}")
        gaps = self.intervals
        span = format_duration(self.total_seconds)
        head = (f"{n} surveys, {self.timestamps[0]:%Y-%m-%d %H:%M} to "
                f"{self.timestamps[-1]:%Y-%m-%d %H:%M} (span {span}), "
                f"times from the {self.source}")
        note = f" {self.note}" if self.note else ""
        if not gaps:
            return head + "." + note
        median = statistics.median(gaps)
        text = head + f". Interval: median {format_duration(median)}"
        if max(gaps) - min(gaps) > 0.02 * max(abs(median), 1.0):
            text += (f", {format_duration(min(gaps))} to {format_duration(max(gaps))}"
                     f" - the sampling is irregular")
        return text + "." + note

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
            own = self.sources[index] if index < len(self.sources) else ""
            out.append((
                index,
                Path(path).name,
                stamp.isoformat(sep=" ") if stamp is not None else "",
                float(self.times[index]) if index < len(self.times) else "",
                gap_days,
                gap_text,
                own if stamp is not None and own else self.source,
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
            "note": self.note,
            "sources": list(self.sources),
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

    source, pattern, note = "index", "", ""
    stamps: List[Optional[_dt.datetime]] = [None] * len(paths)

    if timestamps is not None and len(timestamps) == len(paths):
        stamps = [t if isinstance(t, _dt.datetime) else None for t in timestamps]
        origins = [FROM_SUPPLIED if s is not None else "" for s in stamps]
        source, pattern = "supplied times", "supplied"
    else:
        stamps, pattern, note = _timestamps_by_pattern([Path(p).stem for p in paths])
        stamps = list(stamps)
        origins = [FROM_NAME if s is not None else "" for s in stamps]
        # Each file the names leave undated tries its header, then - only when the
        # user allowed it - its modification time. A header read is kept even when
        # some other file has none, so a modification time fills only what is left.
        for index, path in enumerate(paths):
            if stamps[index] is None and allow_header:
                found = timestamp_from_header(path)
                if found is not None:
                    stamps[index], origins[index] = found, FROM_HEADER
            if stamps[index] is None and allow_mtime:
                try:
                    stamps[index] = _dt.datetime.fromtimestamp(os.path.getmtime(path))
                    origins[index] = FROM_MTIME
                except OSError:
                    pass
        # Naming every source used, because a set that is half named and half
        # sniffed is worth a second look.
        source = _source_of(origins)

    if all(s is not None for s in stamps):
        origin = min(s for s in stamps if s is not None)
        times = [(s - origin).total_seconds() / _SECONDS_PER_DAY for s in stamps]
        if len(set(times)) == len(times):
            return SurveyTiming(files=paths, timestamps=list(stamps), times=times,
                                labels=_labels_for(stamps), source=source,
                                pattern=pattern, unit="d",
                                note=note if source == "file names" else "",
                                sources=origins, found=list(stamps))
        # Identical stamps cannot order a sequence; the index at least can.

    # Not part dated and part numbered: one undated or duplicated file numbers the
    # whole list. What each file did give is kept, so summary() and describe()
    # can name the files that stand in the way.
    n = len(paths)
    return SurveyTiming(
        files=paths, timestamps=[None] * n,
        times=[float(i + 1) for i in range(n)],
        labels=[str(i + 1) for i in range(n)],
        source="index", pattern="", unit="", sources=origins, found=list(stamps))


def _source_of(origins: Sequence[str]) -> str:
    """The set's ``source``: ``file names``, ``file names + headers``, ...

    Unchanged from the single-source wording it always had, so a set dated by
    one kind of source reads as it did.
    """
    whole = {FROM_NAME: "file names", FROM_HEADER: "file headers",
             FROM_MTIME: "file modification times"}
    short = {FROM_NAME: "names", FROM_HEADER: "headers",
             FROM_MTIME: "modification times"}
    used = [kind for kind in (FROM_NAME, FROM_HEADER, FROM_MTIME) if kind in origins]
    if not used:
        return "index"
    if len(used) == 1:
        return whole[used[0]]
    return "file " + " + ".join(short[kind] for kind in used)


#: Words in a file name that mark it as the reciprocal half of a survey, and the
#: words that mark the forward half; both are dropped to match the two names.
_RECIPROCAL_WORDS = frozenset({"recip", "recips", "reciprocal", "reciprocals", "rcp"})
_FORWARD_WORDS = frozenset({"fwd", "forward", "normal", "norm"})


def _name_without_time(stem: str) -> Tuple[Tuple[str, ...], bool]:
    """The words of a name with its timestamp taken out, and whether it says reciprocal."""
    for _name, regex, builder in _PATTERNS:
        for match in regex.finditer(stem):
            try:
                builder(match.groups())
            except (ValueError, TypeError, OverflowError, OSError):
                continue
            stem = stem[:match.start()] + " " + stem[match.end():]
            break
        else:
            continue
        break
    words = [w for w in re.split(r"[^0-9a-z]+", stem.lower()) if w]
    reciprocal = any(w in _RECIPROCAL_WORDS for w in words)
    return tuple(w for w in words
                 if w not in _RECIPROCAL_WORDS and w not in _FORWARD_WORDS), reciprocal


def reciprocal_pairs(files: Sequence[str],
                     timestamps: Optional[Sequence[Optional[_dt.datetime]]] = None
                     ) -> Dict[str, Any]:
    """Files that look like the forward and reciprocal halves of the same surveys.

    Some instruments write a survey as two files, a forward one and its
    reciprocal (``..._sorted_2026_01_12_05_50_38`` and
    ``..._recip_sorted_2026_01_12_06_12_23``). A time-lapse list makes every
    file a time step of its own, so such a set has each survey twice, a few
    minutes apart, and the two halves are never compared - reciprocal errors are
    formed only between readings in one file. This finds the pattern so the
    page can say so: a name that carries a reciprocal word and otherwise matches
    another file's name, its timestamp aside.

    Returns ``{"pairs": [(forward, reciprocal), ...], "reciprocal": [...],
    "orphans": [...], "gap_seconds": median gap or None}`` with indices into
    ``files``; empty lists when nothing looks paired. ``orphans`` are the files
    named as reciprocals that no forward file matches, which the ERT page leaves
    out of a paired series and names.
    """
    names = [_name_without_time(Path(str(f)).stem) for f in files]
    stamps = list(timestamps or [None] * len(names))
    reciprocal = [i for i, (_words, is_recip) in enumerate(names) if is_recip]
    pairs: List[Tuple[int, int]] = []
    taken: set = set()

    # Forward files by name, looked up rather than searched: a 420-survey
    # series is 840 files.
    forwards: Dict[Tuple[str, ...], List[int]] = {}
    for i, (words, is_recip) in enumerate(names):
        if not is_recip:
            forwards.setdefault(words, []).append(i)

    def candidates(r: int) -> List[int]:
        return forwards.get(names[r][0], [])

    def dated(r: int) -> bool:
        return stamps[r] is not None and all(stamps[i] is not None for i in candidates(r))

    # The forward half is the one measured last before its reciprocal, chosen
    # among every forward file, taken or not, and the reciprocal measured first
    # after it wins. A reciprocal whose own forward file is missing would
    # otherwise reach back to an earlier survey whose reciprocal is missing
    # too, hours apart, and merge two different surveys; it is left out.
    timed = sorted((r for r in reciprocal if dated(r)), key=lambda r: stamps[r])
    claimed = {max((i for i in candidates(r) if stamps[i] <= stamps[r]),
                   key=lambda i: stamps[i]) for r in timed
               if any(stamps[i] <= stamps[r] for i in candidates(r))}
    for r in timed:
        before = [i for i in candidates(r) if stamps[i] <= stamps[r]]
        if before:
            partner = max(before, key=lambda i: stamps[i])
        else:
            # Measured before every forward file: the first one after it, unless
            # a later reciprocal is that file's own.
            after = [i for i in candidates(r) if i not in claimed]
            if not after:
                continue
            partner = min(after, key=lambda i: stamps[i])
        if partner in taken:
            continue
        taken.add(partner)
        pairs.append((partner, r))
    # A reciprocal is measured minutes after its forward file. One that would
    # pair across many times the usual gap - its own forward file missing, the
    # one before it unpaired too - joins two surveys hours apart; it is left
    # out instead. Judged once three pairs give a usual gap to judge by.
    timed_gaps = [(stamps[r] - stamps[f]).total_seconds() for f, r in pairs]
    if len(timed_gaps) >= 3:
        usual = statistics.median(abs(g) for g in timed_gaps)
        if usual > 0:
            far = {r for (f, r), g in zip(pairs, timed_gaps) if abs(g) > 4.0 * usual}
            pairs = [(f, r) for f, r in pairs if r not in far]
            taken = {f for f, _r in pairs}
    for r in reciprocal:
        if dated(r):
            continue
        # Undated: the nearest forward file in the list still free.
        free = [i for i in candidates(r) if i not in taken]
        if free:
            partner = min(free, key=lambda i: abs(i - r))
            taken.add(partner)
            pairs.append((partner, r))
    pairs.sort(key=lambda pair: pair[1])
    gaps = [(stamps[r] - stamps[f]).total_seconds() for f, r in pairs
            if stamps[f] is not None and stamps[r] is not None]
    matched = {r for _f, r in pairs}
    return {"pairs": pairs, "reciprocal": [r for _f, r in pairs],
            "orphans": [r for r in reciprocal if r not in matched],
            "gap_seconds": float(statistics.median(gaps)) if gaps else None}
