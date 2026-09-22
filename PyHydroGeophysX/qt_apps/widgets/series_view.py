"""A time-lapse model series with the controls reading one needs.

A monitoring result is a stack of models on one mesh, and looking at them one at a
time answers almost nothing: what a reader wants is the change from the baseline,
on a scale that does not move between steps. This widget wraps
:class:`~PyHydroGeophysX.qt_apps.widgets.mesh_view.MeshResultView` with the step
selector, the absolute/change switch and the shared colour scale, so every place
that shows a series - the ERT module after a run, the saved-results browser, a
result someone else inverted and handed over - shows it the same way.

The two functions below are the arithmetic, kept apart from the widget so the
modules that own their own layout can use the same definitions instead of
re-deriving percentage change slightly differently.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from PyHydroGeophysX.qt_apps.widgets.mesh_view import MeshResultView

__all__ = ["percent_change", "series_color_limits", "TimeLapseSeriesView"]


def percent_change(models: Any, index: int, baseline: int = 0) -> np.ndarray:
    """Percentage change of step ``index`` against the baseline survey.

    Cells whose baseline is zero or non-finite become NaN rather than a spike: a
    division artefact placed next to real change reads as change.
    """
    values = np.asarray(models, dtype=float)
    base = values[:, int(baseline)]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = 100.0 * (values[:, int(index)] - base) / np.abs(base)
    return np.where(np.isfinite(change), change, np.nan)


def series_color_limits(models: Any, mode: str = "model",
                        baseline: int = 0) -> Optional[Tuple[float, float]]:
    """Colour limits that hold over the whole series, for the mode on screen.

    A viewer sees one step at a time, so left alone it autoscales each one to its
    own extremes and the change between steps is rescaled away. The limits come
    from the whole stack at the 2-98 percentile, matching the exported summary
    figure, so a handful of poorly covered cells cannot flatten everything else.
    Change is put on a symmetric scale, because a signed quantity read off an
    off-centre scale has its sign decided by the colours rather than the numbers.
    """
    values = np.asarray(models, dtype=float)
    if values.ndim != 2 or values.shape[1] == 0:
        return None
    if str(mode) == "change":
        with np.errstate(divide="ignore", invalid="ignore"):
            change = 100.0 * (values - values[:, [int(baseline)]]) / np.abs(
                values[:, [int(baseline)]])
        finite = change[np.isfinite(change)]
        if finite.size == 0:
            return None
        span = float(np.nanpercentile(np.abs(finite), 98.0))
        return (-span, span) if span > 0.0 else None
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        return None
    low = float(np.nanpercentile(finite, 2.0))
    high = float(np.nanpercentile(finite, 98.0))
    return (low, max(high, low * 1.01))


class TimeLapseSeriesView(QWidget):
    """Step through a series of models on one mesh, absolute or as change."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._mesh = None
        self._models: Optional[np.ndarray] = None
        self._coverage: Optional[np.ndarray] = None
        self._titles: List[str] = []
        self._kind = "ert"

        self._view = MeshResultView()

        bar = QHBoxLayout()
        self._step = QComboBox()
        self._step.setToolTip("Which survey of the series is displayed.")
        self._step.currentIndexChanged.connect(self._show_current)
        bar.addWidget(self._step, stretch=1)
        self._prev = QPushButton("◀"); self._prev.setMaximumWidth(34)
        self._prev.setToolTip("Previous survey")
        self._prev.clicked.connect(lambda: self._step_by(-1))
        self._next = QPushButton("▶"); self._next.setMaximumWidth(34)
        self._next.setToolTip("Next survey")
        self._next.clicked.connect(lambda: self._step_by(1))
        bar.addWidget(self._prev); bar.addWidget(self._next)

        self._mode = QComboBox()
        self._mode.addItem("Model", "model")
        self._mode.addItem("% change from baseline", "change")
        self._mode.setToolTip(
            "Absolute model, or its change from the first survey. The change is "
            "what a monitoring survey is run to see: it cancels the static "
            "structure both surveys share and leaves what moved.")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self._mode)

        self._bar_row = QWidget()
        self._bar_row.setLayout(bar)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._bar_row)
        layout.addWidget(self._view, stretch=1)

    # -- data ---------------------------------------------------------------

    def set_series(self, mesh: Any, models: Any, coverage: Any = None,
                   titles: Optional[Sequence[str]] = None,
                   kind: str = "ert") -> None:
        """Show ``models`` on ``mesh``.

        ``models`` may be ``(cells, times)`` or ``(times, cells)``; whichever axis
        matches the mesh is taken as the cells. A one-dimensional array is shown
        as a single model with the series controls hidden.
        """
        values = np.asarray(models, dtype=float)
        n_cells = int(mesh.cellCount()) if mesh is not None else 0
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        elif values.shape[0] != n_cells and values.shape[1] == n_cells:
            values = values.T
        self._mesh, self._models, self._kind = mesh, values, str(kind)
        self._coverage = self._normalize_coverage(coverage, values.shape)
        n_steps = values.shape[1]
        self._titles = [str(t) for t in (titles or [])][:n_steps]
        while len(self._titles) < n_steps:
            self._titles.append(f"Time step {len(self._titles) + 1}")

        self._step.blockSignals(True)
        self._step.clear()
        for index in range(n_steps):
            self._step.addItem(f"{index + 1}/{n_steps}  ·  {self._titles[index]}", index)
        self._step.setCurrentIndex(0)
        self._step.blockSignals(False)
        self._mode.blockSignals(True)
        self._mode.setCurrentIndex(0)
        self._mode.blockSignals(False)
        # A single model has no baseline to change from and nothing to step to.
        self._bar_row.setVisible(n_steps > 1)
        self._seed_color_range()
        self._show_current()

    def _normalize_coverage(self, coverage: Any, shape: Tuple[int, int]):
        """Coverage as ``(cells, times)``, whichever way round it arrived."""
        if coverage is None:
            return None
        values = np.asarray(coverage, dtype=float)
        n_cells, n_times = shape
        if values.ndim == 1:
            return values.reshape(-1, 1) if values.size == n_cells else None
        if values.shape == (n_cells, n_times):
            return values
        if values.shape == (n_times, n_cells):
            return values.T
        return None

    # -- state a caller may need (map export, snapshots) ---------------------

    @property
    def mesh_view(self) -> MeshResultView:
        """The underlying section view, for callers that drive it directly."""
        return self._view

    def current_index(self) -> int:
        return max(0, self._step.currentIndex())

    def current_mode(self) -> str:
        return str(self._mode.currentData() or "model")

    def current_title(self) -> str:
        index = self.current_index()
        return self._titles[index] if index < len(self._titles) else ""

    def current_values(self) -> Optional[np.ndarray]:
        """Exactly what is on screen, so an export cannot disagree with it."""
        if self._models is None:
            return None
        index = self.current_index()
        if self.current_mode() == "change":
            return percent_change(self._models, index)
        return self._models[:, index]

    def current_coverage(self) -> Optional[np.ndarray]:
        if self._coverage is None:
            return None
        index = min(self.current_index(), self._coverage.shape[1] - 1)
        return self._coverage[:, index]

    # -- display ------------------------------------------------------------

    def _step_by(self, delta: int) -> None:
        count = self._step.count()
        if count:
            self._step.setCurrentIndex((self._step.currentIndex() + delta) % count)

    def _on_mode_changed(self, _index: int = 0) -> None:
        self._seed_color_range()
        self._show_current()

    def _seed_color_range(self) -> None:
        limits = series_color_limits(self._models, self.current_mode())
        if limits is not None:
            self._view.set_color_range(*limits)

    def _show_current(self, _index: int = 0) -> None:
        if self._mesh is None or self._models is None:
            return
        index = self.current_index()
        if index >= self._models.shape[1]:
            return
        title = self.current_title()
        if self.current_mode() == "change":
            baseline = self._titles[0] if self._titles else "the baseline"
            title = f"{title} − {baseline}" if index else f"{title} (baseline)"
            kind = "change"
        else:
            kind = self._kind
        self._view.show_field(self._mesh, self.current_values(), kind=kind,
                              coverage=self.current_coverage(), title=title)
