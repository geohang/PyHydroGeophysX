"""The ERT page's run records: its settings file, its QC logs and error model.

Built from what the page hands the workflow - the ``WorkflowSpec`` and the
counts its own QC filter returned - never from a second pass over the data,
so a file cannot disagree with the run it describes. Defaults the page does
not set are read from the source of the code the workflow process runs,
without importing it: that code loads the inversion's native libraries, which
the window's process keeps out.

Qt-free; ``modules/ert_processing.py`` calls it. Files are ASCII apart from the
names of data files, so they read the same in any editor.
"""

from __future__ import annotations

import ast
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from PyHydroGeophysX.qt_apps.run_records import (
    ERROR_MODEL_FIGURE_NAME,
    ERROR_PAIRS_NAME,
    QC_FOLDER,
    QC_REPORT_NAME,
    SETTINGS_NAME,
    format_sections,
    plain_value,
    software_versions,
    table,
    write_text,
)

_RULE = "=" * 72

#: What the reciprocal error model is fitted to: every pair as read, or only
#: the pairs the QC filter kept. Chosen on the ERT page's Reciprocal errors
#: tab; the one setting every fit of the model follows.
FIT_ALL = "all"
FIT_KEPT = "kept"
FIT_TO_TEXT = {FIT_ALL: "all pairs before filtering", FIT_KEPT: "pairs kept by the filter"}

#: The figure's colours: those of ``qt_apps/theme.py`` on its plot canvas,
#: which is light in both appearances (repeated so this module stays Qt-free;
#: the page hands the live palette in).
PLOT_COLORS: Dict[str, str] = {
    "canvas": "#ffffff",       # theme PALETTE["canvas"]; a saved figure is white
    "text": "#1d1d1f",         # PALETTE["canvas_text"]
    "muted": "#6e6e73",        # PALETTE["muted"]
    "data": "#007aff",         # theme.DATA_COLOR, one survey's pairs
    "model": "#ff3b30",        # PALETTE["vivid_red"], the fitted model
    "bins": "#1d1d1f",         # the binned means, dark squares
    "left_out": "#aeaeb2",     # PALETTE["disabled_text"], pairs left out of the fit
    "grid": "#e5e5ea",         # the canvas grid of apply_matplotlib_style
    "grid_minor": "#f2f2f7",   # the faint minor grid of the log axes
    "axis": "#c7c7cc",         # axes.edgecolor of apply_matplotlib_style
}
#: The series' colour map, first survey to last: the ERT page's default.
SURVEY_COLORMAP = "viridis"


def fit_to_key(value: Any) -> str:
    """``FIT_KEPT`` for a value naming the filter's pairs, else ``FIT_ALL``."""
    return FIT_KEPT if str(value or "").strip().lower() in (FIT_KEPT, "filtered") else FIT_ALL


def fit_to_sentence(fit_to: str, filtered: Optional[bool] = None) -> str:
    """The choice as the records state it; ``filtered`` False says that no
    filter was applied, so the filter's pairs are every pair."""
    text = FIT_TO_TEXT[fit_to_key(fit_to)]
    if fit_to_key(fit_to) == FIT_KEPT and filtered is False:
        text += " (no filter was applied, so every pair)"
    return text


# -- reciprocal pairing and the error model ----------------------------------------
def reciprocal_pairing(scores: Any) -> Dict[str, Any]:
    """How a survey's readings pair, from the scored frame the QC filtered with.

    ``scores`` is ``ert_formats.reciprocal_errors``'s output over the survey
    (``a, b, m, n, resist, reciprocalErrRel, reciprocalMean``), or None when the
    survey has no resistances to pair. Pairs are counted with the library's own
    labelling, so a pair written in reverse order counts as the pair it is.
    ``R`` and ``dR`` are one value per scored pair - the pair's mean
    resistance and the difference between its two directions - for the error
    model fits.
    """
    if scores is None or not len(scores):
        return {"available": False, "readings": 0 if scores is None else int(len(scores))}
    import pandas as pd
    from PyHydroGeophysX.data_processing.ert_formats import (
        _reciprocal_direction,
        _reciprocal_key,
    )

    key, _sign = _reciprocal_key(scores)
    exchanged = pd.Series(_reciprocal_direction(scores, key), index=scores.index)
    paired = ((~exchanged).groupby(key).transform("any")
              & exchanged.groupby(key).transform("any"))
    error = scores["reciprocalErrRel"].to_numpy(dtype=float)
    scored = np.isfinite(error)
    per_pair = pd.DataFrame({"key": key, "error": error,
                             "mean": np.abs(scores["reciprocalMean"].to_numpy(dtype=float))}
                            )[scored].groupby("key", sort=False).first()
    pairs = int(key[paired].nunique())
    errors = per_pair["error"].to_numpy(dtype=float)
    mean = per_pair["mean"].to_numpy(dtype=float)
    return {
        "available": True,
        "readings": int(len(scores)),
        "normal": int((~exchanged).sum()),
        "reverse": int(exchanged.sum()),
        "pairs": pairs,
        "paired_readings": int(paired.sum()),
        "unpaired_readings": int((~paired).sum()),
        "scored_pairs": int(len(per_pair)),
        "error_median": float(np.median(errors)) if errors.size else None,
        "error_p90": float(np.percentile(errors, 90)) if errors.size else None,
        "error_max": float(errors.max()) if errors.size else None,
        "R": mean,
        "dR": errors * mean,
    }


def reciprocal_pair_kept(scores: Any, keep: Any, averaged: bool = False) -> Optional[np.ndarray]:
    """Whether the QC filter kept each scored pair, in :func:`reciprocal_pairing`'s order.

    ``scores`` is the survey's scored frame as read and ``keep`` the filter's
    verdict on each of its rows (``ERTProcessingModule._qc_survey``). A pair
    is kept when every reading of it survives - a check that drops one of the
    two drops the pair - or, when the filter averaged the pairs (``averaged``),
    when the one reading the pair became survives. None without scores.
    """
    if scores is None or not len(scores):
        return None
    import pandas as pd
    from PyHydroGeophysX.data_processing.ert_formats import _reciprocal_key

    keep = np.asarray(keep, dtype=bool)
    if keep.size != len(scores):
        return None
    key, _sign = _reciprocal_key(scores)
    frame = pd.DataFrame({"key": key, "keep": keep})
    survived = frame.groupby("key", sort=False)["keep"].transform("any" if averaged else "all")
    scored = np.isfinite(scores["reciprocalErrRel"].to_numpy(dtype=float))
    per_pair = pd.DataFrame({"key": key, "kept": survived.to_numpy(dtype=bool)}
                            )[scored].groupby("key", sort=False).first()
    return per_pair["kept"].to_numpy(dtype=bool)


def error_model_pairs(pairings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Every survey's pairs as the error model and its figure use them.

    ``pairings`` are :func:`reciprocal_pairing`'s results in survey order, a
    ``kept`` array among them where a filter was applied
    (:func:`reciprocal_pair_kept`). Only pairs with a positive, finite R and
    dR are kept - a pair whose two readings agree exactly has no logarithm -
    and counted in ``zero``. Returns ``R``, ``dR``, ``survey`` (the index of
    each pair's survey in ``pairings``), ``kept`` (all True for a survey no
    filter judged), ``filtered`` (whether any survey was filtered),
    ``surveys`` (how many were given) and ``zero``.
    """
    R_parts, dR_parts, survey_parts, kept_parts = [], [], [], []
    filtered, zero = False, 0
    for index, pairing in enumerate(pairings):
        if not (pairing and pairing.get("available") and pairing.get("scored_pairs")):
            continue
        R = np.asarray(pairing.get("R", []), dtype=float)
        dR = np.asarray(pairing.get("dR", []), dtype=float)
        flags = pairing.get("kept")
        if flags is not None and len(flags) == R.size:
            filtered = True
            flags = np.asarray(flags, dtype=bool)
        else:
            flags = np.ones(R.size, dtype=bool)
        ok = np.isfinite(R) & np.isfinite(dR) & (R > 0) & (dR > 0)
        zero += int((np.isfinite(R) & np.isfinite(dR) & (R > 0) & (dR == 0)).sum())
        R_parts.append(R[ok])
        dR_parts.append(dR[ok])
        survey_parts.append(np.full(int(ok.sum()), index, dtype=int))
        kept_parts.append(flags[ok])

    def joined(parts, dtype):
        return np.concatenate(parts) if parts else np.array([], dtype=dtype)

    return {"R": joined(R_parts, float), "dR": joined(dR_parts, float),
            "survey": joined(survey_parts, int), "kept": joined(kept_parts, bool),
            "filtered": filtered, "surveys": len(pairings), "zero": zero}


def _used(pairs: Mapping[str, Any], fit_to: str) -> np.ndarray:
    kept = np.asarray(pairs["kept"], dtype=bool)
    return kept if fit_to_key(fit_to) == FIT_KEPT else np.ones(kept.size, dtype=bool)


def fit_pairs(pairs: Mapping[str, Any], fit_to: str = FIT_ALL) -> Optional[Dict[str, Any]]:
    """The binned error model (:func:`fit_error_model_binned`) over the pairs
    ``fit_to`` names, with what it was fitted to.

    Adds ``fit_to``, ``surveys`` (surveys with a pair in the fit),
    ``left_out`` (pairs the filter removed and the fit leaves out) and
    ``filtered``. None with too few pairs.
    """
    use = _used(pairs, fit_to)
    R, dR = np.asarray(pairs["R"], dtype=float), np.asarray(pairs["dR"], dtype=float)
    binned = fit_error_model_binned(R[use], dR[use])
    if binned is None:
        return None
    return {**binned, "fit_to": fit_to_key(fit_to),
            "surveys": int(np.unique(np.asarray(pairs["survey"])[use]).size),
            "left_out": int((~use).sum()), "filtered": bool(pairs.get("filtered"))}


def _line_fit(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if x.size < 3 or float(np.ptp(x)) == 0.0:
        return None
    m, b = np.polyfit(x, y, 1)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - (m * x + b)) ** 2)) / total if total > 0 else float("nan")
    return float(m), float(b), r2


def _log_pairs(R: Any, dR: Any) -> Tuple[np.ndarray, np.ndarray]:
    R, dR = np.asarray(R, dtype=float), np.asarray(dR, dtype=float)
    ok = np.isfinite(R) & np.isfinite(dR) & (R > 0) & (dR > 0)
    return np.log10(R[ok]), np.log10(dR[ok])


def fit_error_model(R: Any, dR: Any) -> Optional[Dict[str, Any]]:
    """``log10(dR) = m log10(R) + b`` over every pair, with its R2.

    Pairs whose two directions agree exactly have no logarithm and are left
    out. None with fewer than three usable pairs.
    """
    x, y = _log_pairs(R, dR)
    fit = _line_fit(x, y)
    if fit is None:
        return None
    return {"m": fit[0], "b": fit[1], "r2": fit[2], "n": int(x.size)}


def _positive_pairs(R: Any, dR: Any) -> Tuple[np.ndarray, np.ndarray]:
    R, dR = np.asarray(R, dtype=float), np.asarray(dR, dtype=float)
    ok = np.isfinite(R) & np.isfinite(dR) & (R > 0) & (dR > 0)
    return R[ok], dR[ok]


def _bin_pairs(R: np.ndarray, dR: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    groups = min(int(bins), R.size // 5)
    if groups < 3:
        return np.array([]), np.array([])
    chunks = np.array_split(np.argsort(R), groups)
    return (np.array([R[c].mean() for c in chunks]),
            np.array([dR[c].mean() for c in chunks]))


def fit_error_model_binned(R: Any, dR: Any, bins: int = 20) -> Optional[Dict[str, Any]]:
    """The same model fitted to the pairs grouped by resistance.

    The pairs are sorted by R and cut into ``bins`` groups of equal count; the
    line goes through log10 of each group's mean R and mean dR. Single pairs
    scatter by an order of magnitude, so a fit to them all has a low R2 even
    when the trend is clear; the grouped fit is the usual way to report it.
    ``r2`` is over the groups, ``r2_raw`` the same line against every pair.
    """
    R, dR = _positive_pairs(R, dR)
    mean_R, mean_dR = _bin_pairs(R, dR, bins)
    if not mean_R.size:
        return None
    xb, yb = np.log10(mean_R), np.log10(mean_dR)
    fit = _line_fit(xb, yb)
    if fit is None:
        return None
    m, b, r2 = fit
    x, y = np.log10(R), np.log10(dR)
    total = float(np.sum((y - y.mean()) ** 2))
    raw = 1.0 - float(np.sum((y - (m * x + b)) ** 2)) / total if total > 0 else float("nan")
    return {"m": m, "b": b, "r2": r2, "r2_raw": raw, "bins": int(mean_R.size), "n": int(R.size)}


def error_model_status(pairs: Mapping[str, Any], fit: Optional[Mapping[str, Any]],
                       fit_to: str, *, series: bool = False, hints: bool = True) -> str:
    """What the figure fitted and what it left out, in a sentence or two;
    ``hints`` adds what to do on the ERT page to change it."""
    R = np.asarray(pairs["R"])
    use = _used(pairs, fit_to)
    surveys = int(np.unique(np.asarray(pairs["survey"])).size)
    where = f" from {surveys} surveys" if series and surveys > 1 else ""
    removed = int((~np.asarray(pairs["kept"], dtype=bool)).sum())
    parts = []
    if fit_to_key(fit_to) == FIT_KEPT:
        if pairs.get("filtered"):
            parts.append(f"Fitted to the {int(use.sum()):,} of {R.size:,} pairs{where} that the "
                         f"filter kept; the {removed:,} it removed are drawn in grey.")
        else:
            parts.append(f"Fitted to all {R.size:,} pairs{where}: no filter has been applied, "
                         "so the filter has kept every pair."
                         + (" Apply filter under Data QC to leave pairs out; the "
                            "reciprocal-error limit under More checks removes outlier pairs."
                            if hints else ""))
    else:
        parts.append(f"Fitted to all {R.size:,} pairs{where}, before any filter.")
        if removed:
            parts.append(f"The filter removes {removed:,} of them"
                         + ("; choose “Pairs kept by the filter” to leave those out."
                            if hints else "."))
    if pairs.get("zero"):
        parts.append(f"{int(pairs['zero']):,} pairs whose two readings agree exactly have "
                     "no place on log axes and are not used.")
    if fit is None and R.size:
        parts.append("At least 15 pairs with a difference are needed to fit the model.")
    return " ".join(parts)


def draw_error_model(figure: Any, pairs: Mapping[str, Any], fit: Optional[Mapping[str, Any]],
                     *, fit_to: str = FIT_ALL, series: bool = False,
                     colors: Optional[Mapping[str, str]] = None, relative: bool = False,
                     show_left_out: bool = True, by_survey: bool = True) -> Any:
    """Draw the reciprocal error figure on ``figure``; returns its axes.

    Every pair is a dot at its mean resistance and the difference between its
    two directions, on log axes - one colour for a survey, the colour map
    first survey to last for a series. The pairs the fit leaves out
    (``fit_to`` the filter's, and the filter removed them) are grey; the dark
    dots are the mean error of each group of pairs the line is fitted through,
    each with the spread of its group (16th to 84th percentile), and the line
    is ``fit`` (:func:`fit_pairs`). The title says what the fit found in
    words, the line under it the model. The ERT page's Reciprocal errors tab,
    Saved Results and the run's PNG all draw it here.

    The rest only change how it looks, never the fit: ``relative`` draws each
    error as a percentage of its resistance (the model then reads
    ``dR/R = 10^b R^(m-1)``), ``show_left_out`` False hides the grey pairs, and
    ``by_survey`` False draws a series in one colour.
    """
    from matplotlib import ticker

    colors = {**PLOT_COLORS, **dict(colors or {})}
    figure.clear()
    figure.set_facecolor(colors["canvas"])
    ax = figure.add_subplot(111)
    ax.set_facecolor(colors["canvas"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    R = np.asarray(pairs["R"], dtype=float)
    dR = np.asarray(pairs["dR"], dtype=float)
    survey = np.asarray(pairs["survey"], dtype=int)
    use = _used(pairs, fit_to)
    count = max(int(pairs.get("surveys") or 0), int(survey.max()) + 1 if survey.size else 1)

    # What is drawn up the axis: the error itself, or as a percentage of R.
    def shown(r, d):
        return 100.0 * d / r if relative else d

    # Many thousands of dots are drawn as one image in a saved figure.
    dense = int(use.sum()) > 5000
    colour_by_survey = series and by_survey
    left = ~use
    # Added in this order - pairs, group means, their spread, pairs left out -
    # so the collections are found where readers of the figure expect them;
    # zorder, not the order, decides what is drawn over what.
    if colour_by_survey:
        points = ax.scatter(R[use], shown(R[use], dR[use]), c=survey[use],
                            cmap=SURVEY_COLORMAP, vmin=0, vmax=max(count - 1, 1), s=8,
                            alpha=0.45, edgecolors="none", rasterized=dense, zorder=2,
                            label="Reciprocal pairs, coloured by survey")
    else:
        points = ax.scatter(R[use], shown(R[use], dR[use]), color=colors["data"], s=9,
                            alpha=0.35, edgecolors="none", rasterized=dense, zorder=2,
                            label="Reciprocal pairs")

    # The groups the line is fitted through (as _bin_pairs makes them), each
    # with how widely its pairs spread: the mean alone hides that a model
    # through a tight cloud and one through a scattered cloud look alike.
    Ru, dRu = R[use], dR[use]
    groups = min(20, Ru.size // 5)
    if groups >= 3:
        chunks = np.array_split(np.argsort(Ru), groups)
        mean_R = np.array([Ru[c].mean() for c in chunks])
        mean_dR = np.array([dRu[c].mean() for c in chunks])
        low = np.array([np.percentile(shown(Ru[c], dRu[c]), 16) for c in chunks])
        high = np.array([np.percentile(shown(Ru[c], dRu[c]), 84) for c in chunks])
        ax.scatter(mean_R, shown(mean_R, mean_dR), marker="o", s=34,
                   facecolor=colors["bins"], edgecolor=colors["canvas"], linewidth=1.2,
                   zorder=4, label="Mean of each group, with its spread")
        ax.vlines(mean_R, low, high, color=colors["bins"], alpha=0.35, linewidth=1.2,
                  zorder=3)
    if left.any() and show_left_out:
        ax.scatter(R[left], shown(R[left], dR[left]), color=colors["left_out"], s=9,
                   alpha=0.6, edgecolors="none", rasterized=dense, zorder=1,
                   label=f"Left out by the filter ({int(left.sum()):,})")
    if fit is not None:
        x = np.geomspace(Ru.min() if Ru.size else R.min(),
                         Ru.max() if Ru.size else R.max(), 200)
        ax.plot(x, shown(x, 10 ** fit["b"] * x ** fit["m"]), color=colors["model"],
                linewidth=2.2, solid_capstyle="round", zorder=5,
                label=f"{'Global model' if series else 'Model'} "
                      f"(binned R² = {float(fit['r2']):.3f})")
    else:
        ax.text(.02, .03, "Fit unavailable: at least 15 positive pairs, spread over a "
                "range of resistances, are needed.", transform=ax.transAxes, fontsize=9,
                color=colors["text"])

    # The studio's look: no box, a faint grid, plain numbers rather than
    # powers of ten.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(colors["axis"])
        ax.spines[side].set_linewidth(0.8)
    plain = ticker.FuncFormatter(_tick_label)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(colors=colors["muted"], which="both", labelsize=9, length=3)
    ax.tick_params(which="minor", length=1.5)
    ax.grid(True, which="major", color=colors["grid"], linewidth=0.7)
    # A minor grid helps read a short axis; over many decades it is a hatching.
    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        if lo > 0 and hi / lo <= 1e3:
            axis.grid(True, which="minor", color=colors["grid_minor"], linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average resistance |R|  (Ω)", color=colors["text"], fontsize=10)
    ax.set_ylabel("Reciprocal error |dR| / |R|  (%)" if relative
                  else "Reciprocal error |dR|  (Ω)", color=colors["text"], fontsize=10)

    title, subtitle = _figure_titles(fit, int(use.sum()), relative=relative, series=series)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold",
                 color=colors["text"], pad=20)
    ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5,
            color=colors["muted"], va="bottom", ha="left")

    # The key goes in the emptiest corner of what was drawn.
    drawn_x = [R[use]]
    drawn_y = [shown(R[use], dR[use])]
    if left.any() and show_left_out:
        drawn_x.append(R[left])
        drawn_y.append(shown(R[left], dR[left]))
    if fit is not None:
        drawn_x.append(x)
        drawn_y.append(shown(x, 10 ** fit["b"] * x ** fit["m"]))
    where = _emptiest_corner(ax, np.concatenate(drawn_x), np.concatenate(drawn_y))
    legend = ax.legend(loc=where, fontsize=8.5, frameon=True, framealpha=0.88,
                       facecolor=colors["canvas"], edgecolor="none", markerscale=1.6,
                       handletextpad=0.4, borderaxespad=0.6)
    for handle in getattr(legend, "legend_handles", []):
        handle.set_alpha(1.0)          # the faint dots, legible in the key
    if colour_by_survey and use.any():
        bar = figure.colorbar(points, ax=ax, pad=0.015, fraction=0.03, aspect=40)
        bar.solids.set_alpha(1.0)
        bar.outline.set_visible(False)
        bar.set_ticks([0, max(count - 1, 1)] if count > 1 else [0])
        bar.set_ticklabels(["1", str(count)] if count > 1 else ["1"])
        bar.set_label("Survey", color=colors["muted"], fontsize=9)
        bar.ax.tick_params(colors=colors["muted"], labelsize=9, length=0)
    return ax


def _emptiest_corner(ax: Any, x: np.ndarray, y: np.ndarray) -> str:
    """The corner of log-log ``ax`` holding the fewest of the points (x, y)."""
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ok = (x > 0) & (y > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fx = (np.log10(x[ok]) - np.log10(x0)) / (np.log10(x1) - np.log10(x0))
        fy = (np.log10(y[ok]) - np.log10(y0)) / (np.log10(y1) - np.log10(y0))
    corners = {"upper left": (fx < 0.45) & (fy > 0.7), "upper right": (fx > 0.55) & (fy > 0.7),
               "lower left": (fx < 0.45) & (fy < 0.3), "lower right": (fx > 0.55) & (fy < 0.3)}
    return min(corners, key=lambda name: int(np.count_nonzero(corners[name])))


def _tick_label(value: float, _pos: Any = None) -> str:
    """A log axis tick as a plain number, or a power of ten beyond 0.001-100000."""
    if value <= 0:
        return ""
    if 1e-3 <= value < 1e5:
        return f"{value:g}"
    return rf"$10^{{{int(round(np.log10(value)))}}}$"


def _figure_titles(fit: Optional[Mapping[str, Any]], used: int, *, relative: bool,
                   series: bool) -> Tuple[str, str]:
    """The figure's title - what the fit found, in words - and the line under it."""
    if fit is None:
        return "Too few reciprocal pairs to fit an error model", f"{used:,} pairs"
    m, b = float(fit["m"]), float(fit["b"])
    trend = ("Reciprocal error grows with resistance" if m > 0.1 else
             "Reciprocal error falls as resistance grows" if m < -0.1 else
             "Reciprocal error hardly changes with resistance")
    a = 10.0 ** b
    model = (rf"dR/|R| = {100.0 * a:.3g} % $\times$ $R^{{{m - 1.0:.2f}}}$" if relative
             else rf"dR = {a:.3g} $\times$ $R^{{{m:.2f}}}$")
    scope = "Global model over the series" if series else "Model"
    return trend, (f"{scope}:  {model}     R² {float(fit['r2']):.3f} over "
                   f"{int(fit['bins'])} groups     {used:,} pairs")


def write_error_model_figure(run_dir: Path,
                             surveys: Sequence[Tuple[str, Mapping[str, Any]]],
                             fit_to: str = FIT_ALL,
                             applied: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """Plot the report's reciprocal pairs and its exact binned fit as a PNG.

    The fit is over ``fit_to``'s pairs, as the report's; ``applied`` is the
    model the inversion used as its data errors, when it did. Writing the
    figure never changes data or error weights. An independent Agg canvas
    works in the QC worker without changing the Qt backend or creating pyplot
    windows. All positive finite pairs are plotted.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    pairs = error_model_pairs([pairing for _name, pairing in surveys])
    path = Path(run_dir) / ERROR_MODEL_FIGURE_NAME
    if not pairs["R"].size:
        # A refreshed report must not retain an earlier, now inapplicable plot.
        path.unlink(missing_ok=True)
        return None
    fit = fit_pairs(pairs, fit_to)
    used = _used(pairs, fit_to)
    figure = Figure(figsize=(10, 6.5), facecolor="white")
    FigureCanvasAgg(figure)
    try:
        draw_error_model(figure, pairs, fit, fit_to=fit_to, series=len(surveys) > 1)
        figure.text(.5, .045, "Used as the data errors of this inversion" if applied else
                    "Diagnostic only — not used in inversion weights", ha="center", fontsize=10)
        surveys_used = int(np.unique(pairs["survey"][used]).size)
        figure.text(.5, .015, f"Fitted to {fit_to_sentence(fit_to, pairs['filtered'])} · "
                    f"{int(used.sum()):,} positive finite pairs · {surveys_used} survey(s)"
                    + (f" · {int((~used).sum()):,} left out (grey)" if (~used).any() else ""),
                    ha="center", fontsize=9, color=PLOT_COLORS["muted"])
        figure.tight_layout(rect=(0, .075, 1, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=180)
    finally:
        figure.clear()
    return path


def write_error_model_pairs(run_dir: Path, surveys: Sequence[Tuple[str, Mapping[str, Any]]],
                            fit_to: str = FIT_ALL,
                            applied: Optional[Mapping[str, Any]] = None) -> Optional[Path]:
    """Keep the figure's pairs in the run folder (``ERROR_PAIRS_NAME``), so
    Saved Results draws the same view as the ERT page: each pair's R, dR,
    survey and whether the filter kept it, the survey names, the choice and
    whether the inversion used the model. None, and no file, without pairs."""
    pairs = error_model_pairs([pairing for _name, pairing in surveys])
    path = Path(run_dir) / ERROR_PAIRS_NAME
    if not pairs["R"].size:
        path.unlink(missing_ok=True)
        return None
    fit = fit_pairs(pairs, fit_to)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, R=pairs["R"], dR=pairs["dR"], survey=pairs["survey"], kept=pairs["kept"],
        filtered=np.array(bool(pairs["filtered"])), zero=np.array(int(pairs["zero"])),
        names=np.array([str(name) for name, _p in surveys], dtype=str),
        fit_to=np.array(fit_to_key(fit_to)), applied=np.array(bool(applied)),
        m=np.array(float(fit["m"]) if fit else np.nan),
        b=np.array(float(fit["b"]) if fit else np.nan))
    return path


def read_error_model_pairs(path: Path) -> Dict[str, Any]:
    """What :func:`write_error_model_pairs` kept: the pairs as
    :func:`error_model_pairs` returns them, with ``names``, ``fit_to`` and
    ``applied``."""
    with np.load(Path(path), allow_pickle=False) as saved:
        names = [str(name) for name in saved["names"]]
        return {"R": np.array(saved["R"], dtype=float), "dR": np.array(saved["dR"], dtype=float),
                "survey": np.array(saved["survey"], dtype=int),
                "kept": np.array(saved["kept"], dtype=bool),
                "filtered": bool(saved["filtered"]), "zero": int(saved["zero"]),
                "surveys": len(names), "names": names,
                "fit_to": fit_to_key(str(saved["fit_to"])), "applied": bool(saved["applied"])}


def _write_qc_report(run_dir: Path, text: str,
                     surveys: Sequence[Tuple[str, Mapping[str, Any]]],
                     fit_to: str = FIT_ALL,
                     applied: Optional[Mapping[str, Any]] = None) -> Path:
    """Keep the text report even when an optional diagnostic plot fails."""
    report = write_text(Path(run_dir) / QC_REPORT_NAME, text)
    try:
        figure = write_error_model_figure(run_dir, surveys, fit_to, applied)
        note = ((f"Reciprocal error figure: {figure.name} ("
                 + ("the model the inversion used" if applied else "diagnostic only") + ").")
                if figure else "No reciprocal error figure: no positive finite pairs to plot.")
    except Exception as exc:  # noqa: BLE001 - a diagnostic must not stop inversion
        note = f"Reciprocal error figure could not be written: {exc}"
    try:
        write_error_model_pairs(run_dir, surveys, fit_to, applied)
    except Exception as exc:  # noqa: BLE001 - Saved Results falls back to the PNG
        note += f"\nThe pairs for Saved Results could not be written: {exc}"
    return write_text(report, text + "\n" + note + "\n")


def _equation(fit: Mapping[str, Any]) -> str:
    sign = "-" if fit["b"] < 0 else "+"
    return f"log10(dR) = {fit['m']:.4f} * log10(R) {sign} {abs(fit['b']):.4f}"


def power_law(fit: Mapping[str, Any]) -> str:
    """The model as Craig Ulrich's QC logs write it: ``dR = 10^b * R^m``."""
    return f"dR = 10^{float(fit['b']):.4f} * R^{float(fit['m']):.4f}"


def fit_series_error_model(pairings: Sequence[Mapping[str, Any]],
                           fit_to: str = FIT_ALL) -> Optional[Dict[str, Any]]:
    """The reciprocal error model over every survey's pairs.

    ``pairings`` are :func:`reciprocal_pairing`'s results, one per survey;
    ``fit_to`` is :data:`FIT_ALL` (every pair as read) or :data:`FIT_KEPT`
    (the pairs the filter kept, where a ``kept`` array says which). The binned
    fit (:func:`fit_pairs`) of those pairs together; it is what
    :func:`error_model_lines` reports and the figure draws, so a run that
    applies it and the records that describe it hold the same numbers. Adds
    ``surveys``, the number of surveys with a pair in the fit. None when there
    are too few pairs.
    """
    return fit_pairs(error_model_pairs(pairings), fit_to)


def error_model_lines(surveys: Sequence[Tuple[str, Mapping[str, Any]]],
                      error_model: str = "",
                      applied: Optional[Mapping[str, Any]] = None,
                      fit_to: str = FIT_ALL) -> List[str]:
    """The reciprocal error model across ``surveys``: one fit over their
    pairs, then each survey's own.

    ``surveys`` is ``(name, pairing)`` in order, ``pairing`` as
    :func:`reciprocal_pairing` returns it. ``fit_to`` names the pairs fitted
    (:func:`fit_series_error_model`). ``applied`` is the model the inversion
    used as its data errors (``m``, ``b``, ``floor``), when it did; otherwise
    ``error_model`` says what the inversion used instead.
    """
    pairs = error_model_pairs([pairing for _name, pairing in surveys])
    kept_only = fit_to_key(fit_to) == FIT_KEPT and pairs["filtered"]
    out = ["Reciprocal error model",
           f"  Error model fitted to: {fit_to_sentence(fit_to, pairs['filtered'])}",
           "  Fitted to the scored normal/reciprocal pairs the QC filter kept; the pairs a "
           "check removed\n  (the reciprocal-error limit removes outlier pairs) are left out."
           if kept_only else
           "  Fitted to every scored normal/reciprocal pair, as read, before any filter."]
    if applied:
        out += ["  The inversion used it as the data errors: each reading's relative error is",
                f"  dR / |R| = 10^{float(applied['b']):.4f} * "
                f"|R|^({float(applied['m']):.4f} - 1), never below "
                f"{100.0 * float(applied.get('floor') or 0.0):g} %."]
    else:
        out.append("  For reference: the studio does not apply it to the data errors"
                   + (f";\n  the inversion used: {error_model}." if error_model else "."))
    if not pairs["R"].size:
        return out + ["  No survey has reciprocal pairs, so there is nothing to fit."]
    used = _used(pairs, fit_to)
    binned = fit_pairs(pairs, fit_to)
    which = "Global" if len(surveys) > 1 else "Survey"
    if binned is None:
        out.append(f"  {which} fit: too few pairs ({int(used.sum())}).")
    else:
        out += [
            f"  {which} model: {_equation(binned)}",
            f"  i.e. {power_law(binned)}  "
            f"(= {10 ** binned['b']:.4g} * R^{binned['m']:.4f}, R and dR in ohm)",
            f"  Parameters: m = {binned['m']:.6f}, b = {binned['b']:.6f}",
            f"  Binned R2 = {binned['r2']:.4f} ({binned['bins']} groups of equal count), "
            f"raw R2 = {binned['r2_raw']:.4f} over {binned['n']} pairs"
            + (f" from {binned['surveys']} surveys" if len(surveys) > 1 else "")
            + (f"; {binned['left_out']} pairs the filter removed left out"
               if binned["left_out"] else ""),
        ]
    rows = []
    for index, (name, pairing) in enumerate(surveys):
        mine = (pairs["survey"] == index) & used
        fit = fit_error_model(pairs["R"][mine], pairs["dR"][mine])
        if fit is None:
            rows.append([name, str(int(mine.sum()) if pairing.get("available") else 0),
                         "-", "-", "-"])
        else:
            rows.append([name, str(fit["n"]), f"{fit['m']:.4f}", f"{fit['b']:.4f}",
                         f"{fit['r2']:.4f}"])
    out += ["", "  Each survey's own fit, over its "
            + ("pairs the filter kept:" if kept_only else "pairs:")]
    out += table(["file", "pairs", "m", "b", "raw R2"], rows, indent=4)
    return out


# -- QC logs ------------------------------------------------------------------------
def _percent(value: Optional[float]) -> str:
    return "-" if value is None else f"{100.0 * value:.3g} %"


def qc_threshold_rows(qc: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The thresholds of a QC setting, every check named, off ones too."""
    more = bool(qc.get("more_checks"))

    def extra(on: bool, text: str) -> str:
        return text if more and on else "off"

    floor = float(qc.get("reciprocal_floor") or 0.0)
    return [
        # .get: QC settings saved before averaging existed do not name it.
        ("Average each reciprocal pair (check 0)",
         f"on; the pair's difference is its error, at least {100.0 * floor:g} %"
         if qc.get("average_reciprocals") else "off"),
        ("Apparent resistivity range", f"{qc['min_rhoa']:g} to {qc['max_rhoa']:g} ohm-m"),
        ("Relative error (err column)",
         f"<= {qc['max_error']:g} %" if qc["max_error"] > 0 else "off"),
        ("More checks", "on" if more else "off (the checks below do not run)"),
        ("Apparent resistivity above 0", extra(qc["drop_nonpositive"], "on")),
        ("Potential |V|", extra(qc["min_voltage"] > 0, f">= {qc['min_voltage']:g} V")),
        ("Current |I|", extra(qc["min_current"] > 0, f">= {qc['min_current']:g} A")),
        ("Geometric factor |k|", extra(qc["max_k"] > 0, f"<= {qc['max_k']:g} m")),
        ("Contact resistance", extra(qc["max_contact_r"] > 0,
                                     f"<= {qc['max_contact_r']:g} ohm")),
        ("Stacking spread", extra(qc.get("max_stack", 0.0) > 0,
                                  f"<= {qc.get('max_stack', 0.0):g} %")),
        ("Reciprocal error", extra(qc["max_reciprocal"] > 0,
                                   f"<= {qc['max_reciprocal']:g} % (unpaired kept)")),
    ]


def file_pair_lines(files: Mapping[str, Any], pairing: Mapping[str, Any],
                    indent: str = "") -> List[str]:
    """A survey's forward and reciprocal files, as Craig Ulrich's parsed
    summary lists them: each file, the status, and the readings of each.

    ``files`` is what the page recorded when it merged them: ``forward`` and
    ``reciprocal`` (names; no reciprocal for an unpaired forward file) with
    their ``forward_readings`` and ``reciprocal_readings``.
    """
    forward, reciprocal = files.get("forward", ""), files.get("reciprocal")
    n_forward = int(files.get("forward_readings", 0) or 0)
    if not reciprocal:
        return [f"{indent}Forward file: {forward}",
                f"{indent}  Status: UNPAIRED (no reciprocal file; inverted on its own)",
                f"{indent}  Readings: forward {n_forward}"]
    n_reciprocal = int(files.get("reciprocal_readings", 0) or 0)
    out = [f"{indent}Forward file:    {forward}",
           f"{indent}Reciprocal file: {reciprocal}",
           f"{indent}  Status: PAIRED (forward + reciprocal, merged into one survey)",
           f"{indent}  Readings: forward {n_forward} | reciprocal {n_reciprocal} | "
           f"total {n_forward + n_reciprocal}"]
    if pairing.get("available"):
        out.append(f"{indent}  Quadrupoles measured both ways: {pairing.get('pairs', 0)}; "
                   f"readings without a reciprocal: {pairing.get('unpaired_readings', 0)}")
    dropped = list(files.get("dropped_fields") or [])
    if dropped:
        out.append(f"{indent}  Columns only one file carried, left out: "
                   + ", ".join(dropped))
    return out


def _pairing_lines(pairing: Mapping[str, Any], averaged: bool = False,
                   fit_to: str = FIT_ALL) -> List[str]:
    if not pairing.get("available"):
        return ["Reciprocal pairing: not possible, the file has no resistances to compare."]
    if not pairing["pairs"]:
        return [f"Reciprocal pairing: none of the {pairing['readings']} readings has a "
                "reciprocal in this file."]
    out = [f"Reciprocal pairing: {pairing['pairs']} quadrupoles measured both ways "
           f"({pairing['paired_readings']} readings); {pairing['unpaired_readings']} "
           "readings have no reciprocal",
           f"  Normal / reciprocal direction: {pairing['normal']} / "
           f"{pairing['reverse']} readings"]
    unscored = pairing["pairs"] - pairing["scored_pairs"]
    if unscored:
        out.append(f"  {unscored} pair(s) straddle zero and are not scored")
    out += [f"  Reciprocal error of the scored pairs: median "
            f"{_percent(pairing['error_median'])}, 90 % of pairs under "
            f"{_percent(pairing['error_p90'])}, largest {_percent(pairing['error_max'])}",
            "  Each pair is averaged into one reading (check 0 below)." if averaged else
            "  Pairs are scored, not averaged: both readings of a pair stay in the data."]
    pairs = error_model_pairs([pairing])
    used = _used(pairs, fit_to)
    fit = fit_error_model(pairs["R"][used], pairs["dR"][used])
    if fit is not None:
        kept = fit_to_key(fit_to) == FIT_KEPT and pairs["filtered"]
        out.append(f"  Error model of this survey: {_equation(fit)}, raw R2 = "
                   f"{fit['r2']:.4f} ({fit['n']} pairs"
                   + (", those the filter kept)" if kept else ")"))
    return out


def survey_qc_log(name: str, report: Mapping[str, Any], *, heading: Sequence[str] = (),
                  fit_to: str = FIT_ALL) -> str:
    """One survey's QC as filtered: the pairing, then each check in the order
    applied with its threshold and what it kept, then the total kept.

    ``report`` is what ``ERTProcessingModule._qc_keep`` filled in while it
    filtered: ``readings``, ``pairing``, ``checks`` and ``kept``; for a survey
    written as two files, ``files`` (:func:`file_pair_lines`); and when its
    pairs were averaged, ``averaged`` (``pairs``, ``removed``), logged as
    check 0 the way Craig Ulrich's logs do.
    """
    rows = []
    number = 0
    for check in report.get("checks", []):
        if check.get("before") is None:
            rows.append(["", check["name"], check.get("threshold", ""), check.get("note", "")])
            continue
        number += 1
        before, after = int(check["before"]), int(check["after"])
        rows.append([f"F{number}", check["name"], check["threshold"],
                     f"Kept {after} / {before} (Dropped {before - after})"])
    pairing = report.get("pairing") or {}
    averaged = report.get("averaged") or {}
    files = report.get("files") or {}
    lines = [f"QC log for {name}", *heading, _RULE,
             *(file_pair_lines(files, pairing) + [""] if files else []),
             f"Readings in the {'files' if files.get('reciprocal') else 'file'}: "
             f"{report.get('readings', 0)}",
             *_pairing_lines(pairing, bool(averaged), fit_to), _RULE]
    if averaged:
        lines += [f"Check 0: {averaged.get('pairs', 0)} pairs averaged. "
                  f"{averaged.get('removed', 0)} redundant rows removed.",
                  f"  {averaged.get('kept', 0)} readings go on to the checks below."]
    if rows:
        lines += ["Checks in the order applied; each counts the readings left by the one "
                  "before.", *table(["", "check", "threshold", "result"], rows, indent=0)]
    else:
        lines.append("No QC filter applied: every reading is inverted.")
    lines += [_RULE,
              f"TOTAL DATA KEPT: {report.get('kept', 0)} / {report.get('readings', 0)}"]
    return "\n".join(lines) + "\n"


def step_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    """A survey's QC counts as plain numbers, for a run's JSON inputs."""
    pairing = dict(report.get("pairing") or {})
    files = dict(report.get("files") or {})
    return {
        "readings": int(report.get("readings", 0)),
        "kept": int(report.get("kept", 0)),
        "reciprocal_file": files.get("reciprocal") or "",
        "averaged_pairs": int((report.get("averaged") or {}).get("pairs", 0) or 0),
        "reciprocal_pairs": int(pairing.get("pairs", 0) or 0),
        "unpaired_readings": int(pairing.get("unpaired_readings", 0) or 0),
        "reciprocal_error_median": pairing.get("error_median"),
        "checks": [{key: value for key, value in check.items()}
                   for check in report.get("checks", [])],
    }


def _threshold_section(qc: Optional[Mapping[str, Any]]) -> Tuple[str, List[Any]]:
    if not qc:
        return ("Thresholds", [("Applied", "none; no QC filter was applied")])
    return ("Thresholds, as Apply filter applied them", qc_threshold_rows(qc))


def write_single_qc_report(run_dir: Path, *, source: str, acquired: str, reader: str,
                           qc: Optional[Mapping[str, Any]], report: Mapping[str, Any],
                           error_model: str,
                           applied_model: Optional[Mapping[str, Any]] = None,
                           fit_to: str = FIT_ALL) -> Path:
    """``qc_report.txt`` of a single inversion: thresholds, the survey's QC log
    and its reciprocal error model, fitted to ``fit_to``'s pairs
    (``applied_model`` when the inversion used it as the data errors)."""
    files = report.get("files") or {}
    survey = [("File", source)]
    if files.get("reciprocal"):
        survey.append(("Reciprocal file", f"{files['reciprocal']} (merged with it)"))
    head = format_sections("Data QC report", [
        ("Survey", [*survey, ("Acquired", acquired or "unknown"), ("Read as", reader)]),
        _threshold_section(qc),
    ])
    pairing = report.get("pairing") or {}
    model = error_model_lines([(source, pairing)], error_model, applied_model, fit_to)
    text = (head + "\n" + survey_qc_log(source, report, fit_to=fit_to) + "\n"
            + "\n".join(model) + "\n")
    return _write_qc_report(run_dir, text, [(source, pairing)], fit_to, applied_model)


def series_qc_log_name(index: int, source: str) -> str:
    """The name of survey ``index``'s QC log in ``qc/``."""
    return f"step_{index + 1:04d}_{Path(source).stem}_QC.txt"


def write_series_qc_logs(folder: Path, index: int, total: int, source: str,
                         report: Mapping[str, Any], *, acquired: str = "",
                         reader: str = "", model: str = "", fit_to: str = FIT_ALL) -> Path:
    """One survey's QC log of a time-lapse series, in ``qc/``. ``model`` is
    the reciprocal error model line put at its top, as Craig Ulrich's logs
    carry it; ``fit_to`` the pairs the survey's own fit is over."""
    heading = [f"Survey {index + 1} of {total}" + (f", acquired {acquired}" if acquired else "")]
    if reader:
        heading.append(f"Read as {reader}")
    if model:
        heading.append(f"Model: {model}")
    return write_text(Path(folder) / series_qc_log_name(index, source),
                      survey_qc_log(Path(source).name, report, heading=heading, fit_to=fit_to))


def write_series_qc_report(run_dir: Path, *, sources: Sequence[str], acquired: Sequence[str],
                           reader: str, qc: Optional[Mapping[str, Any]],
                           reports: Sequence[Mapping[str, Any]], error_model: str,
                           applied_model: Optional[Mapping[str, Any]] = None,
                           left_out: Sequence[str] = (), fit_to: str = FIT_ALL) -> Path:
    """``qc_report.txt`` of a time-lapse series: thresholds, one row per survey,
    the forward and reciprocal files of each, what each check removed over the
    series, and the reciprocal error model fitted to ``fit_to``'s pairs
    (``applied_model`` when the inversion used it). ``left_out`` names
    reciprocal files that had no forward file and were not inverted."""
    rows = []
    for index, (source, report) in enumerate(zip(sources, reports)):
        pairing = report.get("pairing") or {}
        rows.append([str(index + 1), acquired[index] if index < len(acquired) else "",
                     Path(source).name, str(report.get("readings", 0)),
                     str(pairing.get("pairs", 0) or 0),
                     str(pairing.get("unpaired_readings", report.get("readings", 0))),
                     _percent(pairing.get("error_median")),
                     str(report.get("kept", 0)),
                     str(int(report.get("readings", 0)) - int(report.get("kept", 0)))])
    totals: Dict[str, int] = {}
    for report in reports:
        averaged = report.get("averaged") or {}
        if averaged:
            totals["Check 0: rows removed by averaging pairs"] = (
                totals.get("Check 0: rows removed by averaging pairs", 0)
                + int(averaged.get("removed", 0)))
        for check in report.get("checks", []):
            if check.get("before") is not None:
                totals[check["name"]] = (totals.get(check["name"], 0)
                                         + int(check["before"]) - int(check["after"]))
    readings = sum(int(r.get("readings", 0)) for r in reports)
    kept = sum(int(r.get("kept", 0)) for r in reports)
    paired = sum(1 for r in reports if (r.get("files") or {}).get("reciprocal"))
    series = [("Surveys", f"{len(reports)}, each prepared on its own before the inversion"),
              ("Read as", reader),
              ("Per-survey QC logs", f"{QC_FOLDER}/ in this run folder"),
              ("Readings kept", f"{kept} of {readings}")]
    if paired or left_out:
        series.insert(1, ("Forward and reciprocal files",
                          f"{paired} of {len(reports)} surveys merged with their "
                          "reciprocal file" + (f"; {len(left_out)} reciprocal file(s) "
                                               "without a forward file left out"
                                               if left_out else "")))
    head = format_sections("Data QC report", [("Series", series), _threshold_section(qc)])
    lines = [head, "Per survey",
             *table(["step", "acquired", "file", "readings", "recip. pairs", "unpaired",
                     "median recip. error", "kept", "dropped"], rows), ""]
    if paired or left_out:
        lines.append("Forward and reciprocal files, by time step")
        for index, (source, report) in enumerate(zip(sources, reports)):
            files = report.get("files") or {"forward": Path(source).name,
                                             "forward_readings": report.get("readings", 0)}
            lines += [f"  Time step {index + 1}"
                      + (f" ({acquired[index]})" if index < len(acquired) and acquired[index]
                         else ""),
                      *file_pair_lines(files, report.get("pairing") or {}, indent="    "),
                      "  " + "-" * 50]
        for name in left_out:
            lines += [f"  Reciprocal file: {Path(name).name}",
                      "    Status: LEFT OUT (no forward file matches it; not inverted)",
                      "  " + "-" * 50]
        lines.append("")
    lines += ["Dropped by each check over the series",
              *table(["check", "dropped"], [[name, str(count)] for name, count in totals.items()]),
              "", *error_model_lines([(Path(s).name, r.get("pairing") or {})
                                      for s, r in zip(sources, reports)], error_model,
                                     applied_model, fit_to)]
    return _write_qc_report(run_dir, "\n".join(lines) + "\n",
                            [(Path(s).name, r.get("pairing") or {})
                             for s, r in zip(sources, reports)], fit_to, applied_model)


# -- the settings file ---------------------------------------------------------------
def _package_file(*parts: str) -> Path:
    import PyHydroGeophysX

    return Path(PyHydroGeophysX.__file__).resolve().parent.joinpath(*parts)


def _literal(node: ast.AST, names: Mapping[str, Any]) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        text = ast.unparse(node)
        return names.get(text, text)


def function_defaults(parts: Sequence[str], function: str,
                      names: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The keyword defaults of ``function`` in the package file ``parts``,
    read from its source. ``names`` resolves a default that is a constant's
    name. Empty when the file cannot be read."""
    try:
        tree = ast.parse(_package_file(*parts).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            args = node.args
            pairs = list(zip(args.args[len(args.args) - len(args.defaults):], args.defaults))
            pairs += [(arg, value) for arg, value in zip(args.kwonlyargs, args.kw_defaults)
                      if value is not None]
            return {arg.arg: _literal(value, names or {}) for arg, value in pairs
                    if arg.arg != "log"}
    return {}


def module_dict(parts: Sequence[str], name: str) -> Dict[str, Any]:
    """A module-level dict literal such as ``DEFAULT_TL``, read from source."""
    try:
        tree = ast.parse(_package_file(*parts).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return {}
    # The module's own plain constants, for an entry that names one
    # (DEFAULT_TL's figure_max_panels is OVERVIEW_MAX_PANELS).
    constants: Dict[str, Any] = {}
    for node in tree.body:
        target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                  else node.target if isinstance(node, ast.AnnAssign) else None)
        if isinstance(target, ast.Name) and node.value is not None:
            try:
                constants[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if not isinstance(node.value, ast.Dict):
                return {}
            # Entry by entry: one value naming a constant (DEFAULT_TL's
            # figure_max_panels) used to make the whole dict unreadable, and
            # the settings file then marked no time-lapse default at all.
            try:
                return {ast.literal_eval(key): _literal(value, constants)
                        for key, value in zip(node.value.keys, node.value.values)
                        if key is not None}
            except (ValueError, TypeError, SyntaxError):
                return {}
    return {}


#: Settings shown as fractions are also given as a percentage.
_FRACTIONS = {"relative_error", "relativeError", "plateau_tolerance", "min_data_fraction",
              "chi2_tolerance_fraction", "error_floor"}

#: (section, [(key, label)]) for the single inversion, in the order a person reads them.
_SINGLE_LAYOUT: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    ("Inversion", (
        ("engine", "Engine"),
        ("lam", "Regularisation weight lambda"),
        ("max_iterations", "Progress check every (iterations)"),
        ("max_total_iterations", "Max iterations (at each lambda)"),
        ("plateau_tolerance", "Stop when chi2 improves less than (per iteration)"),
        ("solver", "Linear solver"),
        ("model_constraints", "Resistivity bounds (ohm-m)"),
        ("geometric_factor_policy", "Geometric factor check"),
        ("geometric_factor_tolerance", "Geometric factor tolerance"),
    )),
    ("Fit assistance", (
        ("auto_lambda", "Auto-lambda"),
        ("target_chi2", "Target chi2"),
        ("chi2_tolerance", "Target chi2 tolerance"),
        ("max_lambda_trials", "Max lambda trials"),
        ("lambda_bounds", "Lambda search bounds"),
        ("lambda_warm_start", "Lambda trials start from last model"),
        ("lambda_cold_retry_chi2", "Cold restart when chi2 stalls above"),
        ("reject_outliers", "Reject outliers"),
        ("outlier_threshold", "Outlier cut (in data errors)"),
        ("outlier_passes", "Outlier passes"),
        ("min_data_fraction", "Keep at least (fraction of data)"),
    )),
    ("Data errors", (
        ("error_source", "Data errors taken from"),
        ("error_model", "Reciprocal error model"),
        ("relative_error", "Relative error"),
        ("absolute_error", "Absolute error (ohm, added as abs/|R|)"),
        ("error_floor", "Error floor"),
    )),
    ("Mesh", (
        ("mesh_file", "Imported mesh"),
        ("mesh_quality", "Mesh quality (min angle, deg)"),
        ("para_depth", "Depth of the inverted region (m, 0 = auto)"),
        ("para_max_cell_size", "Max cell size (0 = no limit)"),
        ("para_boundary", "Side margin (electrode spacings)"),
        ("surface_nodes", "Nodes between electrodes"),
        ("outer_width", "Outer region width (0 = auto)"),
        ("outer_max_cell_size", "Outer max cell size (0 = no limit)"),
        ("zones", "A-priori zones"),
        ("conform_to_zones", "Mesh follows the zone outlines"),
        ("decouple_zones", "Smoothness stops at zone outlines"),
    )),
)

_TIMELAPSE_LAYOUT: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...] = (
    ("Inversion", (
        ("engine", "Engine"),
        ("windowed", "Windowed (sliding window)"),
        ("window_size", "Window size (surveys)"),
        ("window_step", "Window step (surveys)"),
        ("lambda_val", "Spatial regularisation lambda"),
        ("alpha", "Temporal regularisation alpha"),
        ("temporal_weighting", "Temporal weighting"),
        ("temporal_weight_limit", "Temporal weight limit"),
        ("inversion_type", "Norm"),
        ("max_iterations", "Max iterations (per window when windowed)"),
        ("method", "Linear solver"),
        ("rho_min", "Minimum resistivity (ohm-m)"),
        ("rho_max", "Maximum resistivity (ohm-m)"),
        ("save_memory", "Low-memory (sparse) mode"),
        ("plateau_tolerance", "Stop when chi2 improves less than (per iteration; in-house and E4D)"),
    )),
    ("Fit assistance", (
        ("auto_lambda", "Auto-lambda"),
        ("target_chi2", "Target chi2"),
        ("chi2_tolerance", "Target chi2 tolerance"),
        ("max_lambda_trials", "Max lambda trials"),
        ("lambda_warm_start", "Lambda trials start from last model"),
    )),
    ("Data errors", (
        ("error_model", "Reciprocal error model"),
        ("relativeError", "Relative error"),
        ("absoluteUError", "Absolute potential error (V)"),
        ("max_error", "Drop readings with error above (in the run)"),
    )),
    ("Mesh", (
        ("mesh_file", "Imported mesh"),
        ("mesh_quality", "Mesh quality (min angle, deg)"),
        ("para_depth", "Depth of the inverted region (m, 0 = auto)"),
        ("para_max_cell_size", "Max cell size (0 = no limit)"),
        ("para_boundary", "Side margin (electrode spacings)"),
        ("surface_nodes", "Nodes between electrodes"),
        ("outer_width", "Outer region width (0 = auto)"),
        ("outer_max_cell_size", "Outer max cell size (0 = no limit)"),
        ("zones", "A-priori zones"),
        ("conform_to_zones", "Mesh follows the zone outlines"),
        ("decouple_zones", "Smoothness stops at zone outlines"),
    )),
    ("Results", (
        ("temperature_correction", "Temperature correction in the run"),
        ("figure_clip", "Figure clipping"),
        ("figure_clip_threshold", "Figure clipping threshold"),
        ("figure_max_panels", "Overview figure: most time steps drawn"),
    )),
)


def describe_error_model(model: Mapping[str, Any]) -> str:
    """A reciprocal error model as the settings file states it: the law, its
    fit, what it was fitted over and the floor."""
    parts = [f"{power_law(model)}, i.e. log10(dR) = {float(model['m']):.4f} * log10(R) "
             f"{'-' if float(model['b']) < 0 else '+'} {abs(float(model['b'])):.4f}",
             f"m = {float(model['m']):.6f}, b = {float(model['b']):.6f}"]
    fit = []
    if model.get("r2") is not None:
        fit.append(f"binned R2 = {float(model['r2']):.4f}")
    if model.get("r2_raw") is not None:
        fit.append(f"raw R2 = {float(model['r2_raw']):.4f}")
    if fit:
        parts.append(", ".join(fit))
    if model.get("fitted_over"):
        parts.append(f"fitted over {model['fitted_over']}")
    parts.append(f"each reading's error: dR / |R|, never below "
                 f"{100.0 * float(model.get('floor') or 0.0):g} %")
    return "\n".join(parts)


def _setting(key: str, value: Any) -> str:
    if key == "error_model":
        return (describe_error_model(value) if isinstance(value, Mapping) and value
                else "none")
    if key == "zones" and isinstance(value, list) and value:
        return "\n".join(
            f"{zone.get('name', 'zone')}: {float(zone.get('resistivity', 0)):g} ohm-m"
            + (", fixed" if zone.get("fixed") else "")
            + f", {len(zone.get('polygon') or [])} vertices" for zone in value)
    if key == "mesh_file":
        return str(value) if value else "none (generated from the electrodes)"
    if isinstance(value, dict):
        return "\n".join(f"{k}: {plain_value(v)}" for k, v in value.items()) or "none"
    if key in _FRACTIONS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:g} ({100.0 * value:g} %)"
    if value is None and key in ("max_error", "temperature_correction"):
        return "none"
    return plain_value(value)


def _parameter_sections(layout, values: Mapping[str, Any], set_by_page: Iterable[str],
                        extra: Mapping[str, Any]) -> List[Tuple[str, List[Any]]]:
    """Rows of every known setting, marking the ones the page left at the
    code's default; settings with no known place are listed after them."""
    page = set(set_by_page)
    placed = set()
    sections = []
    for title, keys in layout:
        rows = []
        for key, label in keys:
            if key not in values:
                continue
            placed.add(key)
            text = _setting(key, values[key])
            if key not in page:
                text += "   (default)"
            rows.append((f"{label} [{key}]", text))
        sections.append((title, rows))
    others = [(f"[{key}]", _setting(key, value) + ("" if key in page else "   (default)"))
              for key, value in values.items() if key not in placed]
    others += [(f"[{key}]", value) for key, value in extra.items()]
    if others:
        sections.append(("Other settings", others))
    return sections


def _software(engine: str, instrument: str = "") -> List[Tuple[str, str]]:
    extra = ["adtlert"] if engine == "adtlert" else []
    if instrument and instrument.upper() not in ("BERT", "NONE"):
        extra.append("resipy")
    return software_versions(extra)


def single_settings_text(spec: Any, *, run_id: str, source: str, acquired: str,
                         instrument: str, reader: str, electrode_file: str,
                         electrodes: int, measurements: Tuple[int, int],
                         qc: Optional[Mapping[str, Any]], mode: str,
                         reciprocal: Optional[Mapping[str, Any]] = None,
                         fit_to: Optional[str] = None) -> str:
    """The settings of a single ERT inversion, from the spec handed to it.

    ``measurements`` is (in the file, inverted). The parameters are the
    recipe's under the names the inversion takes (the page's ``lambda`` is its
    ``lam``); the rest of ``run_ert_manager_inversion``'s keywords are listed
    at their defaults. ``reciprocal`` describes the reciprocal file merged
    with the data file (``file``, ``forward_readings``,
    ``reciprocal_readings``), when there was one. ``fit_to``, for data with
    reciprocal pairs, is the pairs the reciprocal error model is fitted to.
    """
    from PyHydroGeophysX.inversion.lambda_search import LAMBDA_BOUNDS

    params = dict(spec.parameters)
    if "lambda" in params:
        params["lam"] = params.pop("lambda")
    defaults = function_defaults(("inversion", "ert_inversion.py"), "run_ert_manager_inversion",
                                 {"LAMBDA_BOUNDS": tuple(LAMBDA_BOUNDS)})
    data = spec.inputs.get("data")
    native = bool(getattr(data, "metadata", {}).get("qc_filtered", False))
    # What the workflow passes: the keywords the function takes, None left out
    # (workflows.domain._keywords_for); without the defaults the function's own.
    taken = {key: value for key, value in params.items()
             if value is not None and (key in defaults or not defaults)}
    extra: Dict[str, Any] = {}
    if native and "instrument" in taken:
        taken.pop("instrument")
        extra["instrument"] = (f"recorded as {params['instrument']}, not used: the run reads "
                               "the saved data file in PyGIMLi's format")
    for key in params:
        if key not in taken and key not in extra and defaults and key not in defaults:
            extra[key] = f"{plain_value(params[key])} (recorded in the recipe, not used)"
    values = {**{k: v for k, v in defaults.items() if k not in ("instrument",)}, **taken}
    kept, total = measurements[1], measurements[0]
    data_rows = [("Data file", source)]
    if reciprocal:
        data_rows.append(("Reciprocal file", f"{reciprocal.get('file', '')}, merged with the "
                          f"data file: forward {reciprocal.get('forward_readings', 0)} + "
                          f"reciprocal {reciprocal.get('reciprocal_readings', 0)} readings"))
    data_rows += [("Acquired", acquired or "unknown"),
                  ("Instrument / format", instrument), ("Read by", reader or "unknown"),
                  ("Electrode positions", electrode_file or "the data file's own header"),
                  ("Electrodes", electrodes),
                  ("Measurements inverted",
                   f"{kept} of {total} in the file{'s' if reciprocal else ''}"),
                  ("Saved for the run as", _input_path(data))]
    qc_rows: List[Any] = (
        [("Applied", f"yes; kept {kept} of {total} (see {QC_REPORT_NAME})"),
         *qc_threshold_rows(qc)] if qc else
        [("Applied", "no; every measurement is inverted as loaded"
                     + (" (electrode edits aside)" if kept != total else ""))])
    sections = [
        ("Run", [("Run", run_id), ("Kind", "ERT single inversion"),
                 ("Written", f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} (local time)"),
                 ("Mode", mode),
                 ("Recipe", "ert_recipe.json; run_ert.py reruns it exactly")]),
        ("Software", _software(str(values.get("engine", "")), instrument)),
        ("Data", data_rows),
        ("Data QC", qc_rows),
        *_parameter_sections(_SINGLE_LAYOUT, values, taken, extra),
    ]
    if fit_to is not None:
        _data_error_row(sections, ("Error model fitted to",
                                   fit_to_sentence(fit_to, bool(qc))))
    return format_sections("ERT inversion settings", sections)


def _data_error_row(sections: List[Tuple[str, List[Any]]], row: Tuple[str, Any],
                    first: bool = False) -> None:
    """Put ``row`` in the settings' Data errors section, making it if missing."""
    for title, rows in sections:
        if title == "Data errors":
            rows.insert(0, row) if first else rows.append(row)
            return
    sections.append(("Data errors", [row]))


def _input_path(ref: Any) -> str:
    return str(getattr(ref, "path", "") or "") or "-"


def timelapse_settings_text(spec: Any, *, run_id: str, sources: Sequence[str],
                            labels: Sequence[str], stamps: Sequence[str],
                            times: Optional[Sequence[float]], instrument: str,
                            electrode_file: str,
                            qc: Optional[Mapping[str, Any]],
                            partners: Optional[Sequence[Optional[str]]] = None,
                            left_out: Sequence[str] = (),
                            data_errors: str = "",
                            fit_to: Optional[str] = None) -> str:
    """The settings of a time-lapse ERT inversion, from the spec handed to it.

    The parameters are the recipe's over ``DEFAULT_TL`` (as the workflow
    merges them); an engine may still change a default it does not support,
    which the outcome section reports once the run says what it used.
    ``partners`` are each survey's reciprocal file (None where there is none),
    ``left_out`` the reciprocal files with no forward file,
    ``data_errors`` the page's sentence on where the data errors come from,
    and ``fit_to``, for a series with reciprocal pairs, the pairs the
    reciprocal error model is fitted to.
    """
    params = dict(spec.parameters)
    defaults = module_dict(("inversion", "_time_lapse_workflow.py"), "DEFAULT_TL")
    values = {**defaults, **params}
    if values.get("windowed"):
        values.setdefault("window_step", 1)
    extra: Dict[str, Any] = {}
    if qc and values.get("instrument") is None:
        # The QC wrote each survey back in PyGIMLi's format, so the run reads
        # those; the instrument they were read as is under Data.
        values.pop("instrument", None)
        extra["instrument"] = ("none in the run: it reads the prepared surveys (QC-filtered, "
                               "or merged with their reciprocal file) in PyGIMLi's format")
    steps = list((qc or {}).get("steps") or [])
    partners = list(partners or [])
    paired = any(partners)
    folders = {str(Path(source).parent) for source in sources}
    rows = []
    for index, source in enumerate(sources):
        stamp = stamps[index] if index < len(stamps) else ""
        elapsed = f"{float(times[index]):.4g}" if times is not None and index < len(times) else ""
        step = steps[index] if index < len(steps) else {}
        kept = (f"{step.get('kept')} of {step.get('total', step.get('readings'))}"
                if step else "as loaded")
        row = [str(index + 1), stamp or (labels[index] if index < len(labels) else ""),
               elapsed, Path(source).name if len(folders) == 1 else str(source)]
        if paired:
            partner = partners[index] if index < len(partners) else None
            row.append((Path(partner).name if len(folders) == 1 else str(partner))
                       if partner else "none (unpaired)")
        rows.append(row + [kept])
    files = table(["step", "acquired", "elapsed (d)", "file"]
                  + (["reciprocal file"] if paired else []) + ["measurements"], rows)
    if len(folders) == 1:
        files = [f"  All in {next(iter(folders))}", *files]
    thresholds = dict((qc or {}).get("thresholds") or {})
    qc_rows: List[Any] = (
        [("Applied", f"yes, to each survey before the inversion (see {QC_REPORT_NAME} "
                     f"and {QC_FOLDER}/)"), *qc_threshold_rows(thresholds)]
        if thresholds else [("Applied", "no; every survey is inverted as loaded"
                             + (" (merged with its reciprocal file)" if paired else ""))])
    data_rows: List[Any] = [
        ("Surveys", len(sources)), ("Instrument / format", instrument),
        ("Electrode positions", electrode_file or "each file's own header"),
        ("Times", "acquisition times parsed from the file names or headers"
         if any(stamps) else "no acquisition times; steps are one unit apart")]
    if paired or left_out:
        data_rows.append((
            "Reciprocal files",
            f"{sum(1 for p in partners if p)} survey(s) merged with their reciprocal "
            "file, the forward file's time being the survey's"
            + ("; left out, having no forward file: "
               + ", ".join(Path(name).name for name in left_out) if left_out else "")))
    engine = str(values.get("engine", ""))
    sections = [
        ("Run", [("Run", run_id), ("Kind", "ERT time-lapse inversion"),
                 ("Written", f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} (local time)"),
                 ("Recipe", "ert_timelapse_recipe.json; run_ert_timelapse.py reruns it "
                            "exactly")]),
        ("Software", _software(engine, instrument)),
        ("Data", [*data_rows, "", "Files, in inversion order:", *files]),
        ("Data QC", qc_rows),
        *_parameter_sections(_TIMELAPSE_LAYOUT, values, params, extra),
    ]
    if data_errors:
        _data_error_row(sections, ("Data errors taken from", data_errors), first=True)
    if fit_to is not None:
        _data_error_row(sections, ("Error model fitted to",
                                   fit_to_sentence(fit_to, bool(thresholds))))
    return format_sections("ERT time-lapse inversion settings", sections)


def write_settings(run_dir: Path, text: str) -> Path:
    return write_text(Path(run_dir) / SETTINGS_NAME, text)


def append_outcome(path: Path, rows: Sequence[Tuple[str, Any]],
                   notes: Sequence[str] = ()) -> None:
    """Add what the run reported - engine used, fit, lambda - to the settings."""
    path = Path(path)
    if not path.is_file():
        return
    rows = [("Finished", f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S} (local time)"), *rows]
    text = format_sections("Outcome", [("As the run reported it", rows)])
    if notes:
        text += "\n".join(notes) + "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n" + text)


def _number(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{digits}g}" if number == number else "-"


def _engine_row(summary: Mapping[str, Any]) -> Tuple[str, str]:
    used = str(summary.get("engine") or "")
    asked = str(summary.get("engine_requested") or used)
    if asked and used and asked != used:
        return ("Engine", f"{used} ({asked} was asked for and could not run)")
    return ("Engine", used or "-")


def single_outcome(summary: Mapping[str, Any],
                   metrics: Mapping[str, Any]) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """Rows of what a single inversion reports about itself: the engine that
    ran, the fit, the lambda it ended at and what it did to the data."""
    errors = dict(summary.get("data_error") or {})
    geometry = dict(summary.get("geometric_factors") or {})
    outliers = dict(summary.get("outliers") or {})
    rows: List[Tuple[str, Any]] = [
        _engine_row(summary),
        ("Final chi2", _number(metrics.get("chi2"))),
        ("RRMS (%)", _number(metrics.get("rrms"))),
        ("Iterations", metrics.get("iterations", "-")),
        ("Measurements inverted", metrics.get("n_data", "-")),
        ("Lambda used", f"{_number(summary.get('lambda_used'))} (asked: "
                        f"{_number(summary.get('lambda_requested'))}); auto-lambda "
                        f"{summary.get('auto_lambda_status', 'off')}"),
    ]
    if errors:
        rows.append(("Data errors used", f"{errors.get('source', '-')}, mean "
                     f"{100.0 * float(errors.get('mean', float('nan'))):.3g} %"))
    rows.append(("Geometric factors", "not checked" if not geometry.get("checked")
                 else "recomputed" if geometry.get("repaired")
                 else "checked, consistent" if geometry.get("ok", True)
                 else "checked, scale looks wrong"))
    if outliers.get("dropped"):
        rows.append(("Outliers rejected", f"{outliers.get('dropped')} "
                     f"({outliers.get('kept')} of {outliers.get('n_start')} kept)"))
    if summary.get("convergence_stop"):
        rows.append(("Stopped because", summary.get("convergence_stop")))
    return rows, []


def timelapse_outcome(payload: Mapping[str, Any], launched: Mapping[str, Any]
                      ) -> Tuple[List[Tuple[str, Any]], List[str]]:
    """Rows of what a time-lapse inversion reports, and the settings it ran
    with differently from the ones written at launch (read back from the
    ``timelapse_config.json`` the workflow writes)."""
    rows: List[Tuple[str, Any]] = [
        _engine_row(payload),
        ("Mode", payload.get("mode", "-")),
        ("Time steps", payload.get("n_times", "-")),
        ("Mesh cells", payload.get("mesh_cells", "-")),
        ("Final chi2", _number(payload.get("chi2"))),
        ("Lambda used", _number(payload.get("lambda_used"))),
        ("Measurements, all steps", payload.get("n_data", "-")),
        ("Low-memory mode", bool(payload.get("save_memory"))),
    ]
    if payload.get("linearized_solver"):
        rows.append(("Linearized solver", payload.get("linearized_solver")))
    errors = dict(payload.get("data_error") or {})
    if errors.get("median") is not None:
        text = (f"{errors.get('source', '-')}, median {100.0 * float(errors['median']):.3g} % "
                f"(from {100.0 * float(errors['min']):.3g} to "
                f"{100.0 * float(errors['max']):.3g} %)")
        moved = [f"{errors[key]} {word}" for key, word in (
            ("raised_to_engine_minimum", "raised to the engine's minimum"),
            ("lowered_to_engine_maximum", "lowered to the engine's maximum"),
            ("filled_at_100_percent", "readings a survey lacks filled in at 100 %"))
            if errors.get(key)]
        if moved:
            text += "; " + ", ".join(moved)
        rows.append(("Data errors used", text))
    elif errors.get("source"):
        rows.append(("Data errors used", errors["source"]))
    if payload.get("zones_not_applied"):
        rows.append(("Zones not applied", ", ".join(map(str, payload["zones_not_applied"]))))
    notes: List[str] = []
    config = payload.get("config_path")
    if config:
        import json

        try:
            ran = dict(json.loads(Path(str(config)).read_text(encoding="utf-8"))
                       .get("inversion") or {})
        except (OSError, ValueError, TypeError):
            ran = {}
        defaults = module_dict(("inversion", "_time_lapse_workflow.py"), "DEFAULT_TL")
        changed = changed_settings({**defaults, **launched}, ran)
        if changed:
            notes = ["Run with these settings changed from the ones above (the engine's "
                     "own choice):", *changed]
    return rows, notes


def changed_settings(launched: Mapping[str, Any], ran: Mapping[str, Any]) -> List[str]:
    """Settings the run used differently from those it was launched with -
    an engine replacing a default it does not support, say."""
    out = []
    for key, value in ran.items():
        # The zones come back in their checked form; they are listed above.
        if key == "zones":
            continue
        if key in launched and _comparable(launched[key]) != _comparable(value):
            out.append(f"  {key}: {_setting(key, launched[key])} -> {_setting(key, value)}")
    return out


def _comparable(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 12)
    if isinstance(value, (list, tuple)):
        return [_comparable(item) for item in value]
    return value


__all__ = [
    "FIT_ALL",
    "FIT_KEPT",
    "FIT_TO_TEXT",
    "append_outcome",
    "changed_settings",
    "describe_error_model",
    "draw_error_model",
    "error_model_pairs",
    "error_model_status",
    "file_pair_lines",
    "fit_pairs",
    "fit_series_error_model",
    "fit_to_key",
    "fit_to_sentence",
    "power_law",
    "read_error_model_pairs",
    "reciprocal_pair_kept",
    "write_error_model_figure",
    "write_error_model_pairs",
    "series_qc_log_name",
    "single_outcome",
    "timelapse_outcome",
    "error_model_lines",
    "fit_error_model",
    "fit_error_model_binned",
    "qc_threshold_rows",
    "reciprocal_pairing",
    "single_settings_text",
    "step_summary",
    "survey_qc_log",
    "timelapse_settings_text",
    "write_series_qc_logs",
    "write_series_qc_report",
    "write_settings",
    "write_single_qc_report",
]
