"""The ERT page's Reciprocal errors tab: how far each reciprocal pair disagrees.

A reading and its reciprocal - current and potential electrodes exchanged -
measure the same resistance, so the difference between them is an error the
data measured themselves. The tab draws every pair at its mean resistance and
that difference, on log axes, with the error model fitted through them: one
survey's pairs, or a time-lapse series' pairs coloured first survey to last,
as Craig Ulrich draws his.

The plot is on the left and its options in a column on the right, as on the
Resistivity model and Mesh tabs. "Fit to" decides which pairs the model is
fitted to - all of them as read, or only those the Data QC filter kept, the
others drawn in grey. It is the ERT page's one setting for it: the data errors
taken from the model, the QC report, the settings file and the run's figure
all follow it. The "Show" options only change the picture. Saved Results shows
the same view from the pairs a run kept, its fit shown but not changeable.

The drawing itself is ``ert_records.draw_error_model``, shared with the PNG
the run writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps import ert_records, theme
from PyHydroGeophysX.qt_apps.qt_utils import ContentWidthScrollArea
from PyHydroGeophysX.qt_apps.widgets.readout import toolbar_row

_CHOICES = (("All pairs (before filtering)", ert_records.FIT_ALL),
            ("Pairs kept by the filter", ert_records.FIT_KEPT))

_TIP = ("Which reciprocal pairs the error model is fitted to.\n\n"
        "All pairs (before filtering): every pair, as read.\n"
        "Pairs kept by the filter: only the pairs whose readings pass the Data QC "
        "checks applied with Apply filter. The reciprocal-error limit under More "
        "checks is the check that removes outlier pairs. The pairs left out are drawn "
        "in grey.\n\n"
        "This choice holds wherever the model is fitted: the data errors when they are "
        "taken from the reciprocal error model, the QC report, the settings file and "
        "the run's figure.")


class ReciprocalErrorView(QWidget):
    """Reciprocal error against resistance, with the error model fitted to it.

    Parameters
    ----------
    chooser : bool
        Let the user change "Fit to" and offer "Use for the data errors" (the
        ERT page). Saved Results passes False: the run was fitted one way, which
        the panel shows, and the run's data errors are already decided.
    """

    #: The "Fit to" choice changed: ``ert_records.FIT_ALL`` or ``FIT_KEPT``.
    fitToChanged = Signal(str)
    #: "Use for the data errors" was pressed; the page sets its Data errors.
    useRequested = Signal()

    def __init__(self, parent=None, *, chooser: bool = True) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._pairs: Optional[Dict[str, Any]] = None
        self._series = False
        self._note = ""
        self._hints = True
        self._fit: Optional[Dict[str, Any]] = None
        self._editable = bool(chooser)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split)

        # -- the plot, or what to do instead -------------------------------------
        self._stack = QStackedLayout()
        self._placeholder = QLabel()
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setContentsMargins(24, 24, 24, 24)
        theme.set_tone(self._placeholder, "muted")
        self._stack.addWidget(self._placeholder)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(2)
        self._figure = Figure(constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._figure)
        # Zoom, pan and Save as on every plot. Its colour bar numbers the
        # surveys of a series, not a measured value, so it has no colour range.
        bar, self._toolbar = toolbar_row(self._canvas, page)
        page_layout.addWidget(bar)
        page_layout.addWidget(self._canvas, stretch=1)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setContentsMargins(8, 0, 8, 6)
        theme.set_tone(self._status, "muted")
        page_layout.addWidget(self._status)
        self._stack.addWidget(page)
        self._page = page
        holder = QWidget()
        holder.setLayout(self._stack)
        split.addWidget(holder)
        split.addWidget(self._build_side())
        split.setCollapsible(0, False)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, 340])

        # The canvas is light in both appearances, dimmed a little in Dark. A
        # bound method, so Saved Results' views, made one per selection, are
        # let go of when they are deleted.
        theme.notifier().changed.connect(self._on_appearance)
        self.show_message("Load ERT data to see how far each reading and its reciprocal "
                          "disagree.")

    def _build_side(self) -> QWidget:
        """The options column: the fit, how the plot looks, and the inversion."""
        column = QWidget()
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 4, 0)

        fit_box = QGroupBox("Error model")
        fit_form = QFormLayout(fit_box)
        self._fit_to = QComboBox()
        for label, value in _CHOICES:
            self._fit_to.addItem(label, value)
        self._fit_to.setToolTip(_TIP if self._editable else
                                "How this run fitted its error model; fixed once it ran.")
        self._fit_to.setEnabled(self._editable)
        self._fit_to.currentIndexChanged.connect(self._on_fit_to)
        fit_form.addRow("Fit to", self._fit_to)
        self._result = QLabel()
        self._result.setWordWrap(True)
        self._result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fit_form.addRow(self._result)
        col.addWidget(fit_box)

        show_box = QGroupBox("Show")
        show_form = QFormLayout(show_box)
        self._error_as = QComboBox()
        self._error_as.addItem("Ohms  (|dR|)", False)
        self._error_as.addItem("Percent of the resistance", True)
        self._error_as.setToolTip(
            "Draw each pair's disagreement in ohms, as the model is fitted, or as a "
            "percentage of its resistance - the number the reciprocal-error limit "
            "under More checks and a data error in percent are compared with. The "
            "model is the same either way.")
        self._error_as.currentIndexChanged.connect(lambda _i: self._render())
        show_form.addRow("Error as", self._error_as)
        self._show_left_out = QCheckBox("Pairs the filter left out")
        self._show_left_out.setChecked(True)
        self._show_left_out.setToolTip("Draw, in grey, the pairs the fit does not use.")
        self._show_left_out.toggled.connect(lambda _on: self._render())
        show_form.addRow(self._show_left_out)
        self._by_survey = QCheckBox("Colour by survey")
        self._by_survey.setChecked(True)
        self._by_survey.setToolTip("Colour a time-lapse series' pairs from the first "
                                   "survey to the last, or draw them in one colour.")
        self._by_survey.toggled.connect(lambda _on: self._render())
        show_form.addRow(self._by_survey)
        col.addWidget(show_box)

        use_box = QGroupBox("Inversion")
        use_layout = QVBoxLayout(use_box)
        self._use_note = QLabel()
        self._use_note.setWordWrap(True)
        use_layout.addWidget(self._use_note)
        self._use_button = QPushButton("Use for the data errors")
        self._use_button.setToolTip(
            "Take the next inversion's data errors from this model: under Data errors, "
            "the source becomes “From the reciprocal error model”.")
        self._use_button.clicked.connect(self.useRequested.emit)
        use_layout.addWidget(self._use_button)
        use_box.setVisible(self._editable)
        col.addWidget(use_box)

        self._save = QPushButton("Save figure and pairs…")
        self._save.setToolTip("Save the figure as it is drawn, and every pair beside it "
                              "as a CSV table with the model in its header.")
        self._save.clicked.connect(self._save_figure)
        col.addWidget(self._save)
        col.addStretch(1)

        self.set_use_state(False)
        self._sync_side()
        side = ContentWidthScrollArea(minimum=300, maximum=380)
        side.setWidget(column)
        return side

    # -- the choice ----------------------------------------------------------
    def fit_to(self) -> str:
        """``ert_records.FIT_ALL`` or ``ert_records.FIT_KEPT``."""
        return str(self._fit_to.currentData() or ert_records.FIT_ALL)

    def set_fit_to(self, value: Any) -> None:
        """Choose what the model is fitted to; emits ``fitToChanged`` on a change."""
        index = self._fit_to.findData(ert_records.fit_to_key(value))
        if index >= 0:
            self._fit_to.setCurrentIndex(index)

    def set_use_state(self, used: bool) -> None:
        """Say whether the next inversion takes its data errors from this model."""
        self._use_note.setText(
            "The next inversion takes its data errors from this model."
            if used else
            "The next inversion does not use this model: its data errors come from "
            "the source chosen under Data errors.")
        self._use_button.setVisible(not used)

    def _on_appearance(self, *_args: Any) -> None:
        self._render()

    def _on_fit_to(self, _index: int) -> None:
        self._render()
        self.fitToChanged.emit(self.fit_to())

    # -- what is shown -------------------------------------------------------
    def show_message(self, text: str) -> None:
        """Say ``text`` in place of the plot (no data, no pairs, still reading)."""
        self._pairs = None
        self._fit = None
        self._placeholder.setText(str(text))
        self._stack.setCurrentWidget(self._placeholder)
        self._sync_side()

    def show_pairs(self, pairs: Mapping[str, Any], *, series: bool = False,
                   note: str = "", hints: bool = True) -> None:
        """Draw ``pairs`` (``ert_records.error_model_pairs``) and the model fitted
        to the chosen ones. ``series`` colours them by survey; ``note`` is added
        to the line under the plot; ``hints`` lets that line point at the ERT
        page's controls."""
        self._pairs = dict(pairs)
        self._series = bool(series)
        self._note = str(note or "")
        self._hints = bool(hints)
        self._stack.setCurrentWidget(self._page)
        self._render()

    def show_saved(self, path: Path) -> None:
        """A run's view, from the pairs it kept (``ert_records.write_error_model_pairs``)."""
        saved = ert_records.read_error_model_pairs(Path(path))
        self._fit_to.blockSignals(True)
        self.set_fit_to(saved["fit_to"])
        self._fit_to.blockSignals(False)
        used = ("This run used the model as its data errors." if saved["applied"] else
                "For reference: this run took its data errors from elsewhere.")
        self.show_pairs(saved, series=len(saved["names"]) > 1, note=used, hints=False)

    def has_plot(self) -> bool:
        return self._pairs is not None

    def last_fit(self) -> Optional[Dict[str, Any]]:
        """The model drawn (``ert_records.fit_pairs``), or None."""
        return self._fit

    def status_text(self) -> str:
        return self._status.text() if self.has_plot() else self._placeholder.text()

    @staticmethod
    def _colors() -> Dict[str, str]:
        # The plot canvas is light in both appearances, so what is drawn on it
        # takes the Light palette's colours; only the ground follows the theme.
        light = theme.LIGHT
        return {"canvas": theme.PALETTE["canvas"], "text": light["canvas_text"],
                "muted": light["muted"], "data": theme.DATA_COLOR,
                "model": light["vivid_red"], "bins": light["text"],
                "left_out": light["disabled_text"], "grid": light["border"]}

    def _sync_side(self) -> None:
        """Enable only the options that change something in what is drawn."""
        pairs = self._pairs
        left_out = 0
        if pairs is not None:
            used = (np.asarray(pairs["kept"], dtype=bool)
                    if self.fit_to() == ert_records.FIT_KEPT
                    else np.ones(len(pairs["kept"]), dtype=bool))
            left_out = int((~used).sum())
        self._show_left_out.setEnabled(left_out > 0)
        self._show_left_out.setText(f"Pairs the filter left out ({left_out:,})"
                                    if left_out else "Pairs the filter left out")
        self._by_survey.setVisible(self._series)
        self._error_as.setEnabled(pairs is not None)
        self._save.setEnabled(pairs is not None)
        self._result.setText(self._result_text())

    def _result_text(self) -> str:
        """The fitted model in a few plain lines, for the panel."""
        if self._pairs is None:
            return "No pairs to fit yet."
        fit = self._fit
        if fit is None:
            return "Too few pairs for a model: at least 15 are needed."
        m, b = float(fit["m"]), float(fit["b"])
        R = np.asarray(self._pairs["R"], dtype=float)
        typical = float(np.median(R)) if R.size else float("nan")
        percent = 100.0 * 10.0 ** b * typical ** (m - 1.0)
        return (f"dR = 10^{b:.3f} · R^{m:.3f}\n"
                f"Fits the mean error of {int(fit['bins'])} groups of pairs "
                f"with R² {float(fit['r2']):.3f}.\n"
                f"At the median resistance, {typical:.3g} Ω, the error is "
                f"{percent:.2g} % of the reading.")

    def _render(self) -> None:
        if self._pairs is None:
            self._sync_side()
            return
        fit_to = self.fit_to()
        self._fit = ert_records.fit_pairs(self._pairs, fit_to)
        ert_records.draw_error_model(
            self._figure, self._pairs, self._fit, fit_to=fit_to, series=self._series,
            colors=self._colors(), relative=bool(self._error_as.currentData()),
            show_left_out=self._show_left_out.isChecked(),
            by_survey=self._by_survey.isChecked())
        text = ert_records.error_model_status(self._pairs, self._fit, fit_to,
                                              series=self._series, hints=self._hints)
        self._status.setText(" ".join(part for part in (text, self._note) if part))
        self._sync_side()
        self._canvas.draw_idle()
        self._toolbar.update()   # Home and Back belong to the new plot

    def _save_figure(self) -> None:
        """Write the figure as drawn, and the pairs beside it as a CSV table."""
        if self._pairs is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the reciprocal error figure", "reciprocal_errors.png",
            "PNG image (*.png)")
        if not path:
            return
        figure_path = Path(path).with_suffix(".png")
        self._figure.savefig(figure_path, dpi=200)
        table = figure_path.with_name(figure_path.stem + "_pairs.csv")
        R = np.asarray(self._pairs["R"], dtype=float)
        dR = np.asarray(self._pairs["dR"], dtype=float)
        survey = np.asarray(self._pairs["survey"], dtype=int) + 1
        used = (np.asarray(self._pairs["kept"], dtype=bool)
                if self.fit_to() == ert_records.FIT_KEPT else np.ones(R.size, dtype=bool))
        fit = self._fit
        model = (f"dR = 10^{float(fit['b']):.6f} * R^{float(fit['m']):.6f}, binned R2 "
                 f"{float(fit['r2']):.4f}" if fit is not None else "no model fitted")
        header = (f"Reciprocal pairs; error model {model}; fitted to "
                  f"{ert_records.fit_to_sentence(self.fit_to())}\n"
                  "survey,average_resistance_ohm,reciprocal_error_ohm,"
                  "relative_error_percent,used_in_fit")
        rows = np.column_stack([survey, R, dR, 100.0 * dR / R, used.astype(int)])
        np.savetxt(table, rows, delimiter=",", header=header, comments="# ",
                   fmt=["%d", "%.8g", "%.8g", "%.6g", "%d"])
        self._status.setText(f"Saved {figure_path.name} and {table.name} in "
                             f"{figure_path.parent}.")


__all__ = ["ReciprocalErrorView"]
