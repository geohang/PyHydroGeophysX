"""The plain-text records a studio run leaves for the people who review it.

A run folder already held what a program needs to repeat the run: the recipe
and the rerun script. What a person checking the run asks for was missing or
scattered - the log as it scrolled past in the window (gone with the session),
the settings in one readable place, and what the data QC removed. This module
writes those as text and lists them, so the page that ran the run and Saved
Results offer the same files.

Qt-free on purpose. The page side - which lines belong to which run, and when a
run's log is complete - is in ``modules/base.py``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

#: Every line the page logged for the run, as the Log window showed it.
RUN_LOG_NAME = "run_log.txt"
#: Everything the workflow process printed, with its exit code at the end.
OUTPUT_LOG_NAME = "workflow_output.log"
#: The run's settings, at the top of the run folder where they are found first.
SETTINGS_NAME = "inversion_settings.txt"
#: What the data QC kept and removed, and the reciprocal error statistics.
QC_REPORT_NAME = "qc_report.txt"
#: The reciprocal error model's figure, as the ERT page's Reciprocal errors
#: tab draws it; drawing it never sets inversion weights.
ERROR_MODEL_FIGURE_NAME = "reciprocal_error_model.png"
#: The figure's pairs - R, dR, survey, kept by the filter - so Saved Results
#: draws the same view as the ERT page.
ERROR_PAIRS_NAME = "reciprocal_error_pairs.npz"
#: One QC log per survey of a time-lapse series.
QC_FOLDER = "qc"
#: The logs carry chi2, lambda and ohm as the window shows them (χ², λ, Ω).
#: UTF-8 with its byte-order mark, which every Windows editor and shell reads
#: as UTF-8; without it Notepad's older versions and PowerShell 5 showed
#: "Ï‡Â²". Appending writes the mark only into an empty file.
LOG_ENCODING = "utf-8-sig"

#: (path in the run folder, kind, label) of the records a run can hold, in the
#: order they are offered. The kinds route them to the text viewer.
_DOCUMENTS: Tuple[Tuple[str, str, str], ...] = (
    (SETTINGS_NAME, "run_settings", "Inversion settings"),
    (QC_REPORT_NAME, "qc_report", "Data QC report"),
    (f"logs/{RUN_LOG_NAME}", "run_log", "Run log"),
    (f"logs/{OUTPUT_LOG_NAME}", "workflow_output", "Workflow process output"),
)
TEXT_KINDS = frozenset(kind for _path, kind, _label in _DOCUMENTS) | {"qc_survey_log"}


def run_documents(run_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """The run's own records that exist on disk, as artifact entries.

    ``path`` is relative to ``run_dir``. The per-survey QC logs of a long
    series are many, so they carry ``listing_only``: a viewer lists them with
    the run's files rather than among the things it draws. The reciprocal
    error pairs are drawn, not opened (``viewer_only``); where they are kept,
    the PNG of the same figure is a file to open, not another view
    (``files_only``).
    """
    base = Path(run_dir)
    found: List[Dict[str, Any]] = []
    for relative, kind, label in _DOCUMENTS:
        if (base / relative).is_file():
            found.append({"artifact_id": f"record:{relative}", "kind": kind,
                          "format": Path(relative).suffix.lstrip("."),
                          "path": relative, "label": label})
    pairs = base / ERROR_PAIRS_NAME
    applied = False
    if pairs.is_file():
        found.append({"artifact_id": "record:reciprocal_error_pairs",
                      "kind": "reciprocal_error_pairs", "format": "npz",
                      "path": ERROR_PAIRS_NAME, "label": "Reciprocal errors",
                      "metadata": {"viewer_only": True}})
        try:
            import numpy as np

            with np.load(pairs, allow_pickle=False) as saved:
                applied = bool(saved["applied"])
        except Exception:  # noqa: BLE001 - a damaged file only loses the label
            pass
    if (base / ERROR_MODEL_FIGURE_NAME).is_file():
        found.append({"artifact_id": "record:reciprocal_error_model", "kind": "figure",
                      "format": "png", "path": ERROR_MODEL_FIGURE_NAME,
                      "label": "Reciprocal error model ("
                               + ("used as the data errors" if applied else "diagnostic only")
                               + ")",
                      **({"metadata": {"files_only": True}} if pairs.is_file() else {})})
    folder = base / QC_FOLDER
    if folder.is_dir():
        for path in sorted(folder.glob("*.txt")):
            relative = f"{QC_FOLDER}/{path.name}"
            found.append({"artifact_id": f"record:{relative}", "kind": "qc_survey_log",
                          "format": "txt", "path": relative,
                          "label": f"QC log · {path.stem}",
                          "metadata": {"listing_only": True}})
    return found


# -- the run log ---------------------------------------------------------------
def log_line(when: _dt.datetime, level: str, message: Any) -> str:
    """One Log-window line as plain text: ``[HH:MM:SS] LEVEL   message``.

    The same layout as ``widgets.log_panel.LogPanel``, the level padded to
    seven characters. A message of several lines keeps its lines, indented
    under the first, so every entry still starts with its time.
    """
    prefix = f"[{when:%H:%M:%S}] {(level or 'info').upper():7} "
    lines = str(message).splitlines() or [""]
    return "\n".join([prefix + lines[0], *(" " * len(prefix) + rest for rest in lines[1:])])


class RunLog:
    """One run's ``logs/run_log.txt``, appended to as its lines arrive.

    Lines are queued by :meth:`add` and written by :meth:`flush`, which the
    page calls once per event-loop turn: a burst of a few hundred lines from
    the workflow process is one write rather than a few hundred. A write that
    fails (OneDrive holding the file a moment) keeps its lines for the next.
    """

    def __init__(self, path: Union[str, Path], title: str) -> None:
        self.path = Path(path)
        self.title = str(title)
        self._pending: List[str] = []
        self._day: Optional[_dt.date] = None
        #: The warning lines among them, each once, in order: what the run's
        #: record keeps as its warnings.
        self.warnings: List[str] = []

    def add(self, when: _dt.datetime, level: str, message: Any) -> None:
        if str(level).lower() in ("warn", "warning") and str(message) not in self.warnings:
            self.warnings.append(str(message))
        if self._day is None:
            self._pending += [
                self.title,
                f"Lines from {when:%Y-%m-%d}; times are local, levels as the Log "
                "window shows them.",
                "",
            ]
        elif when.date() != self._day:
            # The lines carry the time of day only; a run through midnight says so.
            self._pending.append(f"--- {when:%Y-%m-%d} ---")
        self._day = when.date()
        self._pending.append(log_line(when, level, message))

    def flush(self) -> bool:
        """Append what is queued. False when the file could not be written."""
        if not self._pending:
            return True
        try:
            with open(self.path, "a", encoding=LOG_ENCODING, newline="\n") as handle:
                handle.write("\n".join(self._pending) + "\n")
        except OSError:
            return False
        self._pending.clear()
        return True


# -- settings files --------------------------------------------------------------
Row = Union[str, Tuple[str, Any]]
Section = Tuple[str, Sequence[Row]]


def plain_value(value: Any) -> str:
    """A setting as a person writes it: yes/no, plain numbers, no Python reprs."""
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "none"
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return ", ".join(plain_value(item) for item in value)
        return f"{len(value)} entries"
    if isinstance(value, dict):
        if not value:
            return "none"
        return "; ".join(f"{key} {plain_value(item)}" for key, item in value.items())
    text = str(value)
    return text if text else "(empty)"


def format_sections(title: str, sections: Iterable[Section], *, width: int = 34) -> str:
    """``title`` and its sections as aligned text: a label column, then values.

    A row is ``(label, value)``; a plain string is written as it is, for
    tables and notes that keep their own layout. The label column is as wide
    as the longest label, one column for the whole file, and a value of
    several lines continues under the value column.
    """
    sections = [(heading, list(rows)) for heading, rows in sections]
    # Wide enough for most labels, not for the odd long one, which takes a
    # line of its own instead of pushing every value off the page.
    labels = sorted(len(str(row[0])) for _h, rows in sections for row in rows
                    if not isinstance(row, str))
    usual = labels[int(0.9 * (len(labels) - 1))] if labels else 0
    width = min(max(width - 2, usual), 48) + 2
    out = [title, "=" * len(title), ""]
    for heading, rows in sections:
        if not rows:
            continue
        out += [heading, "-" * len(heading)]
        for row in rows:
            if isinstance(row, str):
                out.append(row)
                continue
            label, value = row
            text = plain_value(value)
            lines = text.splitlines() or [""]
            if len(str(label)) >= width:
                out.append(str(label))
                out.extend(" " * width + line for line in lines)
                continue
            out.append(f"{label:<{width}}{lines[0]}".rstrip())
            out.extend(" " * width + line for line in lines[1:])
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def table(header: Sequence[str], rows: Iterable[Sequence[Any]], *, indent: int = 2) -> List[str]:
    """Rows under ``header`` in columns as wide as their widest cell."""
    body = [[plain_value(cell) if not isinstance(cell, str) else cell for cell in row]
            for row in rows]
    widths = [len(name) for name in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    pad = " " * indent

    def line(cells: Sequence[str]) -> str:
        return (pad + "  ".join(f"{cell:<{widths[i]}}" for i, cell in enumerate(cells))).rstrip()

    return [line(header), pad + "  ".join("-" * w for w in widths), *(line(row) for row in body)]


def software_versions(extra: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """``(name, version)`` of the software a run depends on.

    PyHydroGeophysX gives the version it declares and the folder it was loaded
    from: an editable install's packaging metadata keeps the version it was
    installed at, and two checkouts can share a version number. The others come
    from their installed packages, without importing them into this process.
    """
    from importlib import metadata

    import PyHydroGeophysX

    rows = [("PyHydroGeophysX", f"{PyHydroGeophysX.__version__} "
                                f"({Path(PyHydroGeophysX.__file__).resolve().parent})"),
            ("Python", f"{platform.python_version()} ({sys.executable})")]
    for name in ("pygimli", "numpy", "scipy", *extra):
        try:
            rows.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            rows.append((name, "not installed"))
    return rows


def write_text(path: Union[str, Path], text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


__all__ = [
    "ERROR_MODEL_FIGURE_NAME",
    "ERROR_PAIRS_NAME",
    "OUTPUT_LOG_NAME",
    "QC_FOLDER",
    "QC_REPORT_NAME",
    "RUN_LOG_NAME",
    "RunLog",
    "SETTINGS_NAME",
    "TEXT_KINDS",
    "format_sections",
    "log_line",
    "plain_value",
    "run_documents",
    "software_versions",
    "table",
    "write_text",
]
