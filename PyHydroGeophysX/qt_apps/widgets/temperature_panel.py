"""The temperature-correction options, shared by every page that offers them.

Bulk resistivity falls about 2 % per degC, so over a monitoring season the
temperature signal is the same size as the moisture signal the survey is run to
see. Correcting every survey to one reference temperature is what leaves the
remaining change attributable to water - and the same set of choices has to be
available wherever a series is read, whether it was just inverted or opened from
the Result Store months later.

The panel validates before it hands anything back: a correction the user ticked
and the run quietly skipped produces a section indistinguishable from a corrected
one, so an incomplete setup stops the run at the button rather than at the figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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

__all__ = ["TemperatureOptions"]


class TemperatureOptions(QGroupBox):
    """Checkable group of temperature-correction settings.

    Callers tell it how many surveys there are and whether they are dated with
    :meth:`set_context`, then read :meth:`spec`.
    """

    def __init__(self, parent: Optional[QWidget] = None,
                 title: str = "Temperature correction") -> None:
        super().__init__(title, parent)
        self._n_surveys = 0
        self._dated = False
        self.setCheckable(True)
        self.setChecked(False)
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
        self._mode.addItem("Measured depth profile", "profile")
        self._mode.addItem("Surface record → 1-D conduction", "surface")
        self._mode.addItem("Seasonal model (depth + date)", "seasonal")
        self._mode.setToolTip(
            "Where the ground temperature comes from. Most sites only ever measure "
            "the surface, which is what the 1-D conduction mode is for: it diffuses "
            "that record into the ground and supplies the depth structure the "
            "correction needs. One temperature per survey belongs there too - "
            "applied uniformly over depth it would correct the deep ground by what "
            "happened at the surface, which it never felt.")
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
            "CSV with depth_m, temperature_C",
            "A measured temperature profile: two columns, depth below ground in "
            "metres and temperature in °C. Held constant in time and interpolated "
            "onto the mesh; outside the measured interval the nearest value is held "
            "rather than extrapolated.")
        self._profile_row = profile_row
        form.addRow("Profile CSV", profile_row)

        self._surface, surface_row = self._file_row(
            "CSV with date (or day), surface temperature_C",
            "A record of the ground-surface temperature: two columns, a date (or a "
            "number of days) and the temperature in °C. Daily values over the "
            "monitoring period are ample; an air-temperature series from a nearby "
            "station is a workable stand-in.")
        self._surface_row = surface_row
        form.addRow("Surface record", surface_row)

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

        self._damping_note = QLabel("")
        self._damping_note.setWordWrap(True)
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

        self._sync_damping_note()
        self._sync_rows()

    # -- construction helpers ------------------------------------------------

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
        self._set_rows_visible((self._profile_row,), mode == "profile")
        self._set_rows_visible((self._surface_row, self._diffusivity,
                                self._damping_note), mode == "surface")
        self._set_rows_visible(
            (self._mean, self._amplitude, self._damping, self._peak),
            mode == "seasonal")

    # -- context and result --------------------------------------------------

    def set_context(self, n_surveys: int, dated: bool) -> None:
        """Tell the panel what it is being asked to correct.

        ``dated`` says whether the surveys carry acquisition times, which the
        seasonal and surface modes both need: a damped annual wave has no meaning
        without a position in the year, and a surface record cannot be lined up
        with surveys that have no dates.
        """
        self._n_surveys = int(n_surveys)
        self._dated = bool(dated)

    def spec(self) -> Tuple[Optional[Dict[str, Any]], str]:
        """``(options, error)``: the correction to apply, or why it cannot be."""
        if not self.isChecked():
            return None, ""
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
                return None, "Choose a CSV with the measured depth/temperature profile."
            table = self._read_profile(path)
            if table is None:
                return None, (f"Could not read a depth/temperature table from "
                              f"{Path(path).name}. Expected two numeric columns: "
                              f"depth in metres, temperature in °C.")
            spec["profile"] = table
        elif mode == "surface":
            path = self._surface.text().strip()
            if not path or not Path(path).is_file():
                return None, ("Choose a CSV with the surface temperature record: a "
                              "date (or a number of days) and a temperature per row.")
            try:
                times, values, dated = temperature_model.load_surface_temperature(path)
            except (OSError, ValueError) as exc:
                return None, f"Could not read {Path(path).name}: {exc}"
            if dated and not self._dated:
                return None, ("The surface record is dated but the surveys are not, "
                              "so the two cannot be lined up in time. Name the survey "
                              "files with their acquisition date, or give the surface "
                              "record in days on the same clock as the surveys.")
            # Embedded in the spec rather than referenced by path: the run bundle
            # has to carry everything it needs to reproduce the correction.
            spec["surface_times"] = [
                t.isoformat(sep=" ") if hasattr(t, "isoformat") else float(t)
                for t in times]
            spec["surface_temperature"] = [float(v) for v in values]
            spec["diffusivity"] = float(self._diffusivity.value())
        elif mode == "seasonal":
            if not self._dated:
                return None, ("The seasonal model needs to know where in the year "
                              "each survey sits, and these surveys carry no "
                              "acquisition date. Use a constant, a per-survey list "
                              "or a measured profile instead.")
            spec.update({
                "mean_temperature": float(self._mean.value()),
                "amplitude": float(self._amplitude.value()),
                "damping_depth": float(self._damping.value()),
                "peak_day": float(self._peak.value()),
            })
        return spec, ""

    @staticmethod
    def _read_profile(path: str) -> Optional[List[List[float]]]:
        """Two numeric columns (depth, temperature) from a CSV, header or not."""
        for delimiter in (",", None, ";", "\t"):
            try:
                table = np.genfromtxt(path, delimiter=delimiter, dtype=float,
                                      usecols=(0, 1), invalid_raise=False)
            except Exception:  # noqa: BLE001 - try the next delimiter
                continue
            table = np.atleast_2d(table)
            table = table[np.isfinite(table).all(axis=1)]
            if table.shape[0] >= 1 and table.shape[1] == 2:
                return [[float(row[0]), float(row[1])] for row in table]
        return None
