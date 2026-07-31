"""A compact inversion-quality panel shared by every inversion module.

After an inversion finishes, modules call :meth:`InversionQualityView.show_quality`
with a small metrics dict (chi-square, relative RMS, iterations, data count, the
regularization weight, ...) and, when available, a per-iteration convergence
history. The panel shows the headline numbers with a plain-language verdict and a
chi-square-vs-iteration curve, so the user can judge an inversion without reading
the log. :func:`metrics_from_manager` pulls the standard fields off a pyGIMLi
manager (ERT / travel-time) so those modules stay one-liners.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from PyHydroGeophysX.inversion.metrics import metrics_from_manager


class InversionQualityView(QWidget):
    """Metrics header + convergence curve for one inversion run."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._metrics = QLabel("No inversion run yet.")
        self._metrics.setWordWrap(True)
        self._metrics.setTextFormat(Qt.RichText)
        self._metrics.setContentsMargins(8, 6, 8, 2)
        self._fig = Figure(figsize=(5.5, 3.0), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._metrics)
        layout.addWidget(self._canvas, stretch=1)
        self.clear()

    @staticmethod
    def _verdict(chi2: Optional[float]) -> Tuple[str, str]:
        if chi2 is None or chi2 != chi2:
            return "#888888", "no chi-square reported"
        if chi2 <= 1.2:
            return "#2e7d32", "excellent fit (χ² ≈ 1)"
        if chi2 <= 2.0:
            return "#558b2f", "good fit"
        if chi2 <= 5.0:
            return "#f9a825", "fair fit — consider tuning λ or errors"
        return "#c62828", "high misfit — check data errors / λ / geometry"

    def show_quality(self, metrics: Optional[Dict[str, Any]] = None,
                     convergence: Optional[Sequence[float]] = None, title: str = "") -> None:
        m = dict(metrics or {})
        chi2 = m.get("chi2")
        color, verdict = self._verdict(chi2)

        parts: List[str] = []
        if chi2 is not None and chi2 == chi2:
            parts.append(f"<b>χ² = {chi2:.2f}</b>")
        if m.get("rrms") is not None:
            parts.append(f"RMS = {float(m['rrms']):.1f}%")
        if m.get("iterations") is not None:
            parts.append(f"{int(m['iterations'])} iterations")
        if m.get("n_data") is not None:
            parts.append(f"{int(m['n_data'])} data")
        if m.get("lambda") is not None:
            parts.append(f"λ = {float(m['lambda']):g}")
        if m.get("method"):
            parts.append(str(m["method"]))
        for key, val in (m.get("extra") or {}).items():
            parts.append(f"{key} = {val}")

        title_html = f"<b>{title}</b><br>" if title else ""
        head = "  ·  ".join(parts) if parts else "inversion finished"
        note = f"<br><span style='color:#888888'>{m['note']}</span>" if m.get("note") else ""
        self._metrics.setText(
            f"{title_html}<span style='font-size:13px'>{head}</span><br>"
            f"<span style='color:{color}'><b>{verdict}</b></span>{note}")

        self._draw_convergence(convergence, final_chi2=chi2,
                               track=m.get("convergence_track"))

    def _draw_track(self, ax, track) -> bool:
        """Plot every iteration the pipeline ran, segmented by lambda.

        The lambda sweep and the rejection passes each restart the misfit, so a
        single curve of the final run hides most of what happened. Here the
        segments are laid end to end on one iteration axis, with a boundary line
        and a label wherever lambda changed or data were dropped.
        """
        segments = []
        for seg in track or []:
            values = [float(c) for c in (seg.get("chi2") or [])
                      if float(c) == float(c)]
            if values:
                segments.append((seg, values))
        if not segments:
            return False

        colours = ["#1565ff", "#00897b", "#8e24aa", "#f4511e", "#3949ab", "#00838f"]
        start = 1
        ticks = []
        for index, (seg, values) in enumerate(segments):
            xs = list(range(start, start + len(values)))
            colour = colours[index % len(colours)]
            ax.plot(xs, values, "o-", color=colour, lw=1.6, ms=3.5, zorder=3)
            if index:
                ax.axvline(start - 0.5, color="#9e9e9e", ls=":", lw=1.0, zorder=1)
                # Join the segments so the jump at a boundary is visible as a
                # step rather than as a gap.
                previous = segments[index - 1][1][-1]
                ax.plot([start - 1, start], [previous, values[0]],
                        color="#bdbdbd", lw=1.0, ls="-", zorder=2)
            label = f"λ={float(seg.get('lambda', 0)):g}"
            if str(seg.get("stage", "")).startswith("reject"):
                label = f"{seg['stage']}\nn={seg.get('n_data')}"
            ticks.append((start + len(values) / 2.0 - 0.5, label))
            start += len(values)

        top = ax.get_ylim()[1]
        for x, label in ticks:
            ax.annotate(label, xy=(x, top), xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top", fontsize=7.5, color="#555555")
        ax.axhline(1.0, color="#c62828", ls="--", lw=1.0, label="target χ² = 1")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration (all stages, in order)")
        ax.set_ylabel("χ²")
        ax.set_title("Convergence")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="best")
        return True

    def _draw_convergence(self, convergence: Optional[Sequence[float]],
                          final_chi2: Optional[float] = None,
                          track: Optional[Sequence[dict]] = None) -> None:
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if self._draw_track(ax, track):
            self._canvas.draw()
            return
        conv: List[float] = []
        if convergence is not None:
            for c in convergence:
                try:
                    fc = float(c)
                    if fc == fc:
                        conv.append(fc)
                except (TypeError, ValueError):
                    continue
        if conv:
            iters = list(range(1, len(conv) + 1))
            ax.plot(iters, conv, "o-", color="#1565ff", lw=1.6, ms=4)
            ax.axhline(1.0, color="#c62828", ls="--", lw=1.0, label="target χ² = 1")
            ax.set_yscale("log")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("χ²")
            ax.set_title("Convergence")
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.legend(fontsize=8, loc="best")
            if len(iters) <= 15:
                ax.set_xticks(iters)
        elif final_chi2 is not None and final_chi2 == final_chi2:
            value = float(final_chi2)
            color = "#2e7d32" if value <= 1.2 else "#f9a825" if value <= 5.0 else "#c62828"
            ax.bar(["Final \u03c7\u00b2"], [value], color=color, width=0.5)
            ax.axhline(1.0, color="#1565ff", ls="--", lw=1.2, label="target \u03c7\u00b2 = 1")
            ax.set_ylabel("\u03c7\u00b2")
            ax.set_title("Final data misfit")
            ax.grid(True, axis="y", ls=":", alpha=0.4)
            ax.legend(fontsize=8, loc="best")
            ax.text(0, value, f"  {value:.2f}", va="bottom", ha="center", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No per-iteration history for this inversion.",
                    ha="center", va="center", transform=ax.transAxes, color="#888888")
            ax.axis("off")
        # This tab is usually hidden when an inversion worker completes; draw()
        # guarantees the finished figure is ready when the user opens it.
        self._canvas.draw()

    def clear(self) -> None:
        self._metrics.setText("<span style='color:#888888'>Run an inversion to see its "
                              "quality (χ², RMS, convergence) here.</span>")
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.axis("off")
        self._canvas.draw()
