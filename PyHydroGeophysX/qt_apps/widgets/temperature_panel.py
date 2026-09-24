"""The temperature-correction panel, shared by every page that shows an ERT series.

Bulk resistivity falls about 2 % per degC, so over a monitoring season the
temperature signal is the same size as the moisture signal the survey is run to
see. Correcting every survey to one reference temperature is what leaves the
remaining change attributable to water - and the same choices have to be
available wherever a series is read, whether it was just inverted on the ERT page
or reopened in Saved Results months later.

Nothing is switched on in advance. The user sets the options and presses Apply,
and the page corrects the section it is showing, there and then; Remove puts the
inverted models back. The panel validates before it asks: a correction that was
set up wrongly and quietly skipped produces a section indistinguishable from a
corrected one, so an incomplete setup stops at the button, with the reason, rather
than at the figure. After every Apply the panel states what the section now is.

The functions below the panel are the part both pages share beyond the widget -
where each survey sits in time, and the correction itself - so one run corrected
on either page gives the same numbers.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from PyHydroGeophysX.petrophysics import temperature as temperature_model
from PyHydroGeophysX.qt_apps import theme

__all__ = [
    "TemperatureOptions",
    "correct_series",
    "run_correction",
    "series_survey_times",
    "title_suffix",
]

_STATUS_COLORS = {
    "muted": theme.PALETTE["muted"],
    "ok": theme.PALETTE["green"],
    "error": theme.PALETTE["red"],
}


def _format_time(value: Any) -> str:
    """A record or survey time, as a reader would write it."""
    if isinstance(value, _dt.datetime):
        if (value.hour, value.minute) == (0, 0):
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M")
    return f"day {float(value):g}"


class TemperatureOptions(QGroupBox):
    """Temperature-correction settings, applied with their own button.

    The owning page connects :attr:`applyRequested` (a validated spec) and
    :attr:`removeRequested`, tells the panel what it would be correcting with
    :meth:`set_context` and :meth:`set_available`, and answers every request
    with :meth:`show_applied` or :meth:`show_problem`.

    ``shared`` is a dict every page passes to the panels it builds - the studio
    state's - so the ERT page and Saved Results open with the same settings.
    """

    #: A validated correction spec, when the user presses Apply.
    applyRequested = Signal(dict)
    #: The user asked for the inverted models back.
    removeRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None,
                 title: str = "Temperature correction",
                 shared: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(title, parent)
        self._n_surveys = 0
        self._dated = False
        self._survey_days: Optional[list] = None
        self._survey_dates: Optional[list] = None
        self._available = False
        self._shared = shared
        self._restoring = False
        # The correction on screen and the settings it was made with, so an
        # edit afterwards is not mistaken for one that has been applied.
        self._applied_report: Optional[Mapping[str, Any]] = None
        self._applied_settings: Optional[Dict[str, Any]] = None
        self.setToolTip(
            "Report every time step at one reference temperature, so the change "
            "between surveys is moisture rather than the seasonal warming and "
            "cooling of the ground (about 2 % of resistivity per degC).")
        form = QFormLayout(self)
        self._form = form

        self._ref = QDoubleSpinBox()
        self._ref.setRange(-20.0, 60.0); self._ref.setDecimals(1)
        self._ref.setValue(25.0); self._ref.setSuffix(" °C")
        self._ref.setToolTip(
            "Temperature the corrected sections are reported at. 25 °C is the "
            "laboratory convention petrophysical relations are calibrated at; the "
            "site's annual mean is the other defensible choice, and keeps the "
            "correction small.")
        form.addRow("Reference", self._ref)

        self._model = QComboBox()
        self._model.addItem("Hayley et al. 2007 (≈2.0 %/°C)", "hayley")
        self._model.addItem("Linear (Campbell/Arps)", "linear")
        self._model.setToolTip(
            "Hayley et al. (2007) is fitted on soils and pore fluids from 0 to "
            "25 °C. The linear form is the classic 1 + α(T − Tref) with α set "
            "below.")
        self._model.currentIndexChanged.connect(self._sync_rows)
        form.addRow("Law", self._model)

        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.0, 0.2); self._alpha.setDecimals(4)
        self._alpha.setSingleStep(0.001); self._alpha.setValue(0.025)
        self._alpha.setToolTip(
            "Fractional resistivity change per °C in the linear law. 0.025 (2.5 %) "
            "is the standard value for pore water.")
        form.addRow("α (per °C)", self._alpha)

        self._mode = QComboBox()
        self._mode.addItem("Constant temperature", "constant")
        self._mode.addItem("Measured profile (depth × time)", "profile")
        self._mode.addItem("Surface record → 1-D conduction", "surface")
        self._mode.addItem("Seasonal model (depth + date)", "seasonal")
        self._mode.setToolTip(
            "Where the ground temperature comes from. A measured profile - a "
            "thermistor string - gives it with depth and through time, so nothing "
            "has to be modelled. Most sites only measure the surface, which is what "
            "the 1-D conduction mode is for: it diffuses that record into the ground "
            "and supplies the depth structure. One temperature per survey belongs "
            "there too - applied uniformly over depth it would correct the deep "
            "ground by what happened at the surface, which it never felt.")
        self._mode.currentIndexChanged.connect(self._sync_rows)
        form.addRow("Temperature from", self._mode)

        self._value = QDoubleSpinBox()
        self._value.setRange(-20.0, 60.0); self._value.setDecimals(1)
        self._value.setValue(15.0); self._value.setSuffix(" °C")
        self._value.setToolTip(
            "One temperature for the whole section and the whole sequence. This "
            "rescales every step by the same factor, so it changes the resistivity "
            "values but not the change between surveys.")
        form.addRow("Temperature", self._value)

        self._profile, profile_row = self._file_row(
            "CSV: date, depth_m, temperature_C",
            "A measured temperature record, in whichever layout the logger wrote:\n"
            "  • date, depth_m, temperature_C — one reading per row\n"
            "  • a header of depths and a row per date (one column per sensor)\n"
            "  • the same table transposed: a header of dates, a row per depth\n"
            "  • depth_m, temperature_C — one profile, held constant in time\n\n"
            "Dates, or days on the same clock as the surveys. Gaps are filled in "
            "time. Before and after the record, and above the shallowest sensor, "
            "the nearest reading is held; below the deepest sensor its seasonal "
            "swing is damped with depth towards its mean, at the annual damping "
            "depth of the diffusivity.")
        self._profile_row = profile_row
        form.addRow("Profile table", profile_row)
        self._profile_note = self._note_label()
        form.addRow("", self._profile_note)

        self._surface, surface_row = self._file_row(
            "CSV: date (or day), surface temperature_C",
            "A record of the ground-surface temperature: two columns, a date (or a "
            "number of days) and the temperature in °C. Daily values over the "
            "monitoring period are ample; an air-temperature series from a nearby "
            "station is a workable stand-in.")
        self._surface_row = surface_row
        form.addRow("Surface record", surface_row)
        self._surface_note = self._note_label()
        form.addRow("", self._surface_note)

        self._diffusivity = QDoubleSpinBox()
        self._diffusivity.setRange(0.005, 1.0); self._diffusivity.setDecimals(3)
        self._diffusivity.setSingleStep(0.01)
        self._diffusivity.setValue(float(temperature_model.DEFAULT_DIFFUSIVITY))
        self._diffusivity.setSuffix(" m²/day")
        self._diffusivity.setToolTip(
            "Thermal diffusivity of the ground. Soils run about 0.04 to 0.13 m²/day; "
            "wetter and denser ground is higher. The damping depth below says what "
            "the value means in metres.")
        self._diffusivity.valueChanged.connect(self._sync_damping_note)
        form.addRow("Diffusivity", self._diffusivity)

        self._damping_note = self._note_label()
        form.addRow("", self._damping_note)

        self._mean = QDoubleSpinBox()
        self._mean.setRange(-20.0, 40.0); self._mean.setDecimals(1)
        self._mean.setValue(12.0); self._mean.setSuffix(" °C")
        self._mean.setToolTip(
            "Annual mean ground temperature, which is what the ground settles to "
            "below a few damping depths. Close to the annual mean air temperature "
            "at most sites.")
        form.addRow("Annual mean", self._mean)

        self._amplitude = QDoubleSpinBox()
        self._amplitude.setRange(0.0, 40.0); self._amplitude.setDecimals(1)
        self._amplitude.setValue(10.0); self._amplitude.setSuffix(" °C")
        self._amplitude.setToolTip(
            "Half the peak-to-peak annual swing at the ground surface.")
        form.addRow("Surface amplitude", self._amplitude)

        self._damping = QDoubleSpinBox()
        self._damping.setRange(0.1, 20.0); self._damping.setDecimals(2)
        self._damping.setValue(2.5); self._damping.setSuffix(" m")
        self._damping.setToolTip(
            "Depth at which the annual swing has decayed to 1/e of its surface "
            "amplitude. Typically 2–3 m in soils, more in dry coarse ground. Below "
            "about three of these the correction stops varying with the season.")
        form.addRow("Damping depth", self._damping)

        self._peak = QDoubleSpinBox()
        self._peak.setRange(1.0, 365.0); self._peak.setDecimals(0)
        self._peak.setValue(200.0)
        self._peak.setToolTip(
            "Day of the year the ground surface is warmest: about 200 (mid-July) in "
            "the northern hemisphere, about 20 in the southern.")
        form.addRow("Warmest day", self._peak)

        buttons = QWidget()
        bar = QHBoxLayout(buttons)
        bar.setContentsMargins(0, 4, 0, 0)
        self._apply = QPushButton("Apply")
        self._apply.setProperty("primary", True)
        self._apply.setToolTip(
            "Correct the section on screen with these settings. Press again after "
            "changing them; nothing is corrected until you do.")
        self._apply.clicked.connect(self._on_apply)
        self._remove = QPushButton("Remove")
        self._remove.setToolTip("Show the resistivity as inverted again.")
        self._remove.clicked.connect(self.removeRequested.emit)
        bar.addWidget(self._apply, stretch=1)
        bar.addWidget(self._remove)
        form.addRow(buttons)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        form.addRow(self._status)

        for spin in (self._ref, self._alpha, self._value, self._diffusivity,
                     self._mean, self._amplitude, self._damping, self._peak):
            spin.valueChanged.connect(self._on_settings_changed)
        for combo in (self._model, self._mode):
            combo.currentIndexChanged.connect(self._on_settings_changed)
        for edit in (self._profile, self._surface):
            edit.textChanged.connect(self._on_settings_changed)
            edit.editingFinished.connect(self._describe_files)

        self._sync_damping_note()
        self._sync_rows()
        self._describe_files()
        self.set_available(False)
        if shared:
            self.restore_settings(shared)

    # -- construction helpers ------------------------------------------------

    @staticmethod
    def _note_label() -> QLabel:
        label = QLabel("")
        label.setWordWrap(True)
        label.setStyleSheet(f"color:{_STATUS_COLORS['muted']}; font-size:8pt;")
        return label

    def _file_row(self, placeholder: str, tooltip: str) -> Tuple[QLineEdit, QWidget]:
        row = QWidget()
        bar = QHBoxLayout(row)
        bar.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.setToolTip(tooltip)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._browse_into(edit))
        bar.addWidget(edit, stretch=1)
        bar.addWidget(browse)
        return edit, row

    def _browse_into(self, edit: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Temperature record", "",
            "Tables (*.csv *.txt *.dat);;All files (*)")
        if path:
            edit.setText(path)
            self._describe_files()

    def _row_label(self, widget: QWidget):
        return self._form.labelForField(widget)

    def _set_rows_visible(self, widgets, visible: bool) -> None:
        for widget in widgets:
            widget.setVisible(bool(visible))
            label = self._row_label(widget)
            if label is not None:
                label.setVisible(bool(visible))

    def _sync_damping_note(self, *_args: Any) -> None:
        depth = temperature_model.annual_damping_depth(float(self._diffusivity.value()))
        self._damping_note.setText(
            f"Annual swing falls to 1/e by about {depth:.1f} m, and is negligible "
            f"below roughly {3 * depth:.0f} m.")

    def _sync_rows(self, *_args: Any) -> None:
        """Show the inputs the selected law and temperature source actually read."""
        self._set_rows_visible((self._alpha,),
                               str(self._model.currentData()) == "linear")
        mode = str(self._mode.currentData() or "constant")
        self._set_rows_visible((self._value,), mode == "constant")
        self._set_rows_visible((self._profile_row, self._profile_note), mode == "profile")
        self._set_rows_visible((self._surface_row, self._surface_note), mode == "surface")
        # A logged profile reads the diffusivity too: it sets how fast the swing
        # of the deepest sensor dies away in the cells below it.
        self._set_rows_visible((self._diffusivity, self._damping_note),
                               mode in ("surface", "profile"))
        self._set_rows_visible(
            (self._mean, self._amplitude, self._damping, self._peak),
            mode == "seasonal")

    # -- what the chosen files hold ------------------------------------------

    def _describe_files(self, *_args: Any) -> None:
        """Say what each chosen file was read as - depths, times, span - before Apply.

        A layout read the wrong way round, or a record that stops before the
        surveys do, would otherwise only show up as a wrong section.

        It never raises. It runs whenever a panel is built from the shared
        settings, so a saved path whose file cannot be read stopped every later
        panel from being built - Saved Results then called each ERT result
        incomplete. Whatever went wrong is the note instead.
        """
        for note, describe, edit in ((self._profile_note, self._profile_summary, self._profile),
                                     (self._surface_note, self._surface_summary, self._surface)):
            try:
                text = describe()
            except Exception as exc:  # noqa: BLE001 - see above
                text = f"Could not read {Path(edit.text().strip()).name}: {exc}"
            note.setText(text)

    def _profile_summary(self) -> str:
        path = self._profile.text().strip()
        if not path:
            return ("Rows of date, depth and temperature, or one column per sensor "
                    "depth. Without a date column the profile is held constant in "
                    "time.")
        if not Path(path).is_file():
            return f"{Path(path).name}: no such file."
        try:
            depths, times, values, _dated = temperature_model.load_temperature_profiles(path)
        except Exception as exc:  # noqa: BLE001 - any failure to read is a note
            return f"Could not read {Path(path).name}: {exc}"
        z = np.asarray(depths, dtype=float)
        span = (f"{z.size} depths ({z.min():g}–{z.max():g} m)" if z.size > 1
                else f"1 depth ({z[0]:g} m)")
        temps = np.asarray(values, dtype=float)
        heat = f"{np.nanmin(temps):.1f} to {np.nanmax(temps):.1f} °C"
        if times is None:
            return (f"{span}, {heat}, no time axis: held constant in time. Add a "
                    f"date column to follow the ground through the season.")
        text = (f"{span} × {len(times)} times ({_format_time(min(times))} → "
                f"{_format_time(max(times))}), {heat}.")
        return text + self._coverage_warning(times)

    def _surface_summary(self) -> str:
        path = self._surface.text().strip()
        if not path:
            return "Rows of date (or day) and the surface temperature."
        if not Path(path).is_file():
            return f"{Path(path).name}: no such file."
        try:
            times, values, _dated = temperature_model.load_surface_temperature(path)
        except Exception as exc:  # noqa: BLE001 - any failure to read is a note
            return f"Could not read {Path(path).name}: {exc}"
        text = (f"{len(times)} readings ({_format_time(min(times))} → "
                f"{_format_time(max(times))}), {min(values):.1f} to "
                f"{max(values):.1f} °C at the surface.")
        return text + self._coverage_warning(times)

    def _coverage_warning(self, record_times: Sequence[Any]) -> str:
        """A warning when surveys fall outside the record, else ''.

        Outside its own span a record is held at its nearest reading, which is
        a guess the section would carry without saying so.
        """
        if not record_times:
            return ""
        first, last = min(record_times), max(record_times)
        if isinstance(first, _dt.datetime):
            surveys = [d for d in (self._survey_dates or [])
                       if isinstance(d, _dt.datetime)]
        else:
            surveys = [float(d) for d in (self._survey_days or [])]
        try:
            outside = sum(1 for when in surveys if when < first or when > last)
        except TypeError:
            return ""
        if not outside:
            return ""
        return (f" {outside} of {len(surveys)} surveys fall outside the record "
                f"and take its nearest reading.")

    # -- settings, shared between pages --------------------------------------

    def settings(self) -> Dict[str, Any]:
        """The panel's inputs as plain values, for another panel to restore."""
        return {
            "reference": float(self._ref.value()),
            "model": str(self._model.currentData() or "hayley"),
            "alpha": float(self._alpha.value()),
            "mode": str(self._mode.currentData() or "constant"),
            "value": float(self._value.value()),
            "profile_path": self._profile.text(),
            "surface_path": self._surface.text(),
            "diffusivity": float(self._diffusivity.value()),
            "mean_temperature": float(self._mean.value()),
            "amplitude": float(self._amplitude.value()),
            "damping_depth": float(self._damping.value()),
            "peak_day": float(self._peak.value()),
        }

    def restore_settings(self, values: Mapping[str, Any]) -> None:
        """Put back what :meth:`settings` returned; unknown keys are ignored."""
        if not values:
            return
        self._restoring = True
        try:
            for spin, key in ((self._ref, "reference"), (self._alpha, "alpha"),
                              (self._value, "value"), (self._diffusivity, "diffusivity"),
                              (self._mean, "mean_temperature"),
                              (self._amplitude, "amplitude"),
                              (self._damping, "damping_depth"), (self._peak, "peak_day")):
                if key in values:
                    try:
                        spin.setValue(float(values[key]))
                    except (TypeError, ValueError):
                        pass
            for combo, key in ((self._model, "model"), (self._mode, "mode")):
                index = combo.findData(values.get(key))
                if index >= 0:
                    combo.setCurrentIndex(index)
            for edit, key in ((self._profile, "profile_path"),
                              (self._surface, "surface_path")):
                if key in values:
                    edit.setText(str(values[key] or ""))
        finally:
            self._restoring = False
        self._sync_rows()
        self._describe_files()
        self._flag_unapplied_edits()

    def _on_settings_changed(self, *_args: Any) -> None:
        if self._restoring:
            return
        if self._shared is not None:
            self._shared.clear()
            self._shared.update(self.settings())
        self._flag_unapplied_edits()

    def _flag_unapplied_edits(self) -> None:
        """After an Apply, say when the settings no longer match what is on screen."""
        if self._applied_report is None or self._applied_settings is None:
            return
        if self.settings() == self._applied_settings:
            self.show_applied(self._applied_report)
            return
        reference = float(self._applied_report.get("reference_temperature_C", 25.0))
        self._set_status(
            f"Settings changed — press Apply to use them. The section still shows "
            f"the earlier correction, at {reference:g} °C.", "muted")

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        # The other page may have changed the shared settings while this one
        # was hidden; coming back shows them rather than a stale copy.
        if self._shared:
            self.restore_settings(self._shared)
        super().showEvent(event)

    # -- what the page tells the panel ----------------------------------------

    def set_context(self, n_surveys: int, dated: bool,
                    days: Optional[Sequence[float]] = None,
                    dates: Optional[Sequence[Any]] = None) -> None:
        """Tell the panel what it would be correcting.

        ``dated`` says whether the surveys carry acquisition times, which the
        seasonal mode and any dated record need: a damped annual wave has no
        meaning without a position in the year, and a dated record cannot be
        lined up with surveys that have no dates. ``days`` and ``dates`` are
        the survey times themselves, used to warn when a record does not cover
        them.
        """
        self._n_surveys = int(n_surveys)
        self._dated = bool(dated)
        self._survey_days = list(days) if days is not None else None
        self._survey_dates = list(dates) if dates is not None else None
        self._describe_files()

    def set_available(self, available: bool, hint: str = "") -> None:
        """Enable Apply only when there is a section to correct."""
        self._available = bool(available)
        self._apply.setEnabled(self._available)
        if not self._available:
            self._applied_report = self._applied_settings = None
            self._remove.setEnabled(False)
            self._set_status(hint or "There is no result to correct yet.", "muted")

    def show_applied(self, report: Optional[Mapping[str, Any]]) -> None:
        """State what the section on screen is, after an Apply or a Remove."""
        if not report:
            self._applied_report = self._applied_settings = None
            self._set_status(
                "Not applied — the section shows the resistivity as inverted.",
                "muted")
            self._remove.setEnabled(False)
            return
        if report is not self._applied_report:
            self._applied_report = report
            self._applied_settings = self.settings()
        text = (f"Applied — sections reported at "
                f"{float(report.get('reference_temperature_C', 25.0)):g} °C.")
        ground, scaled = report.get("temperature_range_C"), report.get("factor_range")
        if ground and scaled:
            text += (f" Ground {ground[0]:.1f} to {ground[1]:.1f} °C; resistivity "
                     f"× {scaled[0]:.2f} to {scaled[1]:.2f} (up to "
                     f"{float(report.get('max_abs_change_percent', 0.0)):.0f} %).")
        # The full account - law, source, record - is one hover away, and in the log.
        self._set_status(text, "ok",
                         tooltip=str(report.get("note", "")).replace("degC", "°C"))
        self._remove.setEnabled(True)

    def show_problem(self, message: str) -> None:
        """Say why the last request did not go through; the section is unchanged."""
        self._set_status(str(message), "error")

    def status_text(self) -> str:
        return self._status.text()

    def _set_status(self, text: str, level: str, tooltip: str = "") -> None:
        self._status.setText(text)
        self._status.setToolTip(tooltip)
        self._status.setStyleSheet(
            f"color:{_STATUS_COLORS.get(level, _STATUS_COLORS['muted'])};")

    # -- the request ------------------------------------------------------------

    def _on_apply(self) -> None:
        try:
            spec, problem = self.spec()
        except Exception as exc:  # noqa: BLE001 - Apply must answer, never do nothing
            spec, problem = None, f"Could not use the temperature record: {exc}"
        if problem:
            self.show_problem(problem)
            return
        self.applyRequested.emit(spec)

    def spec(self) -> Tuple[Optional[Dict[str, Any]], str]:
        """``(options, error)``: the correction these settings describe, or why not."""
        mode = str(self._mode.currentData() or "constant")
        spec: Dict[str, Any] = {
            "enabled": True,
            "model": str(self._model.currentData() or "hayley"),
            "alpha": float(self._alpha.value()),
            "reference": float(self._ref.value()),
            "mode": mode,
        }
        if mode == "constant":
            spec["value"] = float(self._value.value())
        elif mode == "profile":
            path = self._profile.text().strip()
            if not path or not Path(path).is_file():
                return None, ("Choose the measured temperature table: date, depth "
                              "and temperature per row, or one column per sensor.")
            try:
                depths, times, values, dated = \
                    temperature_model.load_temperature_profiles(path)
            except Exception as exc:  # noqa: BLE001 - any reader failure is the problem shown
                return None, f"Could not read {Path(path).name}: {exc}"
            if times is None:
                spec["profile"] = [[float(d), float(v)]
                                   for d, v in zip(depths, np.asarray(values).ravel())]
            else:
                if dated and not self._dated:
                    return None, ("The profile is logged against dates but the "
                                  "surveys are not, so the two cannot be lined up "
                                  "in time. Name the survey files with their "
                                  "acquisition date, or log the profile in days on "
                                  "the same clock as the surveys.")
                # Embedded rather than referenced by path, so the record of what
                # was applied carries everything needed to reproduce it.
                spec["profile_depths"] = [float(d) for d in depths]
                spec["profile_times"] = [
                    t.isoformat(sep=" ") if hasattr(t, "isoformat") else float(t)
                    for t in times]
                spec["profile_values"] = np.asarray(values, dtype=float).tolist()
                spec["diffusivity"] = float(self._diffusivity.value())
        elif mode == "surface":
            path = self._surface.text().strip()
            if not path or not Path(path).is_file():
                return None, ("Choose a CSV with the surface temperature record: a "
                              "date (or a number of days) and a temperature per row.")
            try:
                times, values, dated = temperature_model.load_surface_temperature(path)
            except Exception as exc:  # noqa: BLE001 - any reader failure is the problem shown
                return None, f"Could not read {Path(path).name}: {exc}"
            if dated and not self._dated:
                return None, ("The surface record is dated but the surveys are not, "
                              "so the two cannot be lined up in time. Name the survey "
                              "files with their acquisition date, or give the surface "
                              "record in days on the same clock as the surveys.")
            # Embedded in the spec rather than referenced by path, so the record
            # of what was applied carries everything needed to reproduce it.
            spec["surface_times"] = [
                t.isoformat(sep=" ") if hasattr(t, "isoformat") else float(t)
                for t in times]
            spec["surface_temperature"] = [float(v) for v in values]
            spec["diffusivity"] = float(self._diffusivity.value())
        elif mode == "seasonal":
            if not self._dated:
                return None, ("The seasonal model needs to know where in the year "
                              "each survey sits, and these surveys carry no "
                              "acquisition date. Use a constant, a measured profile "
                              "or a surface record in days instead.")
            spec.update({
                "mean_temperature": float(self._mean.value()),
                "amplitude": float(self._amplitude.value()),
                "damping_depth": float(self._damping.value()),
                "peak_day": float(self._peak.value()),
            })
        return spec, ""


# -- shared by both pages -----------------------------------------------------

def series_survey_times(summary: Optional[Mapping[str, Any]],
                        n_steps: int) -> Tuple[Optional[list], Optional[list]]:
    """``(days, dates)`` of a series' surveys, from what its run recorded.

    The ERT page reads the result it has just received and Saved Results the
    stored record; both carry the same keys, so one survey is placed at one time
    whichever page corrects it.
    """
    from PyHydroGeophysX.data_processing.survey_timing import parse_timestamp

    summary = summary or {}
    days = summary.get("measurement_times")
    days = ([float(value) for value in days]
            if isinstance(days, (list, tuple)) and len(days) == n_steps else None)
    timing = summary.get("survey_timing")
    stamps = timing.get("timestamps") if isinstance(timing, Mapping) else None
    dates = None
    if isinstance(stamps, (list, tuple)) and len(stamps) == n_steps and all(stamps):
        try:
            dates = [_dt.datetime.fromisoformat(str(value)) for value in stamps]
        except ValueError:
            dates = None
    if dates is None:
        # A run saved before the timing was recorded still has its labels, and
        # those are the dates the panels were headed with.
        labels = summary.get("time_labels") or summary.get("step_titles")
        if isinstance(labels, (list, tuple)) and len(labels) == n_steps:
            parsed = [parse_timestamp(str(label)) for label in labels]
            if all(item is not None for item in parsed):
                dates = [item[0] for item in parsed]
    return days, dates


def correct_series(mesh: Any, models: Any, spec: Mapping[str, Any], *,
                   days: Optional[Sequence[float]] = None,
                   dates: Optional[Sequence[Any]] = None
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Correct a displayed series to the reference temperature in ``spec``.

    ``models`` may be one model or a series in either orientation, as the series
    view accepts; the result comes back the same way round. Depth is measured
    below the surface of ``mesh`` alone, so a run gives the same correction on
    the ERT page and in Saved Results, where the electrodes are not to hand.
    """
    from PyHydroGeophysX.core import section_geometry

    array = np.asarray(models, dtype=float)
    depths = section_geometry.cell_depths(mesh)
    n_cells = int(depths.size)
    if array.ndim == 1:
        stack = array.reshape(-1, 1)
        restore: Callable[[np.ndarray], np.ndarray] = lambda out: out[:, 0]
    elif array.ndim == 2 and array.shape[0] == n_cells:
        stack, restore = array, (lambda out: out)
    elif array.ndim == 2 and array.shape[1] == n_cells:
        stack, restore = array.T, (lambda out: out.T)
    else:
        raise ValueError(
            f"the models {array.shape} do not match the mesh's {n_cells} cells")
    corrected, report = temperature_model.correct_time_lapse_models(
        stack, dict(spec), depths, days=days, dates=dates)
    return restore(corrected), report


def run_correction(summary: Optional[Mapping[str, Any]], models: Any,
                   resolve: Callable[[str], Path]
                   ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """``(inverted, report)``: a run's models as inverted, and any correction it made.

    A run made while the correction was still chosen before the inversion
    corrected its own series: its ``final_models.npy`` is the corrected one, with
    the raw series saved beside it. Correcting that again would count the
    temperature twice, so a new correction has to start from the raw models.
    ``inverted`` is None when the run corrected its models and the raw ones can
    no longer be found; ``report`` is None when the run made no correction.
    """
    report = (summary or {}).get("temperature_correction")
    if not isinstance(report, Mapping) or not report.get("applied"):
        return np.asarray(models, dtype=float), None
    try:
        raw = np.load(resolve(str(report.get("uncorrected_models") or "")),
                      allow_pickle=False)
    except Exception:  # noqa: BLE001 - missing or unreadable: say so, never guess
        return None, dict(report)
    return np.asarray(raw, dtype=float), dict(report)


def title_suffix(report: Optional[Mapping[str, Any]]) -> str:
    """What a corrected section's title adds, so the figure says it too."""
    if not report:
        return ""
    return f" · at {float(report.get('reference_temperature_C', 25.0)):g} °C"
