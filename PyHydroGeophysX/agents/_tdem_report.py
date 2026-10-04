"""The TDEM part of a workflow report: what was measured, how, and what it shows.

A run whose only data were TEM soundings ended without a report, because the
report step needed an ERT inversion, and a run with both described the ERT
survey alone. These functions write the TDEM sections from what
:class:`~PyHydroGeophysX.agents.tdem_agent.TDEMAgent` returns - an instrument
survey (TEM2Go, tTEM) inverted as laterally constrained 1D models, or one
sounding read from a table. :mod:`PyHydroGeophysX.agents._survey_report` lays
them out, as a report of their own or as a section of an ERT report. Every
number is computed from those results; no sentence here is written by a
language model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._document import bullets, facts, table
from ._report_text import band as _band, length as _length, relative as _relative, source_name
from ._uncertainty import water_content_reliability
from ..visualization.axis_units import to_display_length

KEY = "tdem"
ARTIFACT = "tdem_results"
TITLE = "Time-Domain Electromagnetic Survey"
FIGURE_TAG = "T"

#: Chi-squared above which a sounding is said to fit its data poorly.
POOR_FIT_CHI2 = 3.0

#: Depth band edges (m) the model is summarised over, cut to the model's depth.
DEPTH_BANDS_M = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)

#: How each lateral coupling is named in the method section.
LCI_LABELS = {
    "simultaneous": "Laterally constrained 1D inversion; every line solved as one "
                    "system (simultaneous LCI)",
    "sequential": "Laterally constrained 1D inversion by block-coordinate passes",
    "off": "Independent 1D inversion of each sounding",
}


# ---------------------------------------------------------------------------
# reading the results
# ---------------------------------------------------------------------------

def _array(tdem: Mapping[str, Any], key: str, dtype=float) -> np.ndarray:
    value = tdem.get(key)
    if value is None:
        return np.array([], dtype=dtype)
    return np.asarray(value, dtype=dtype).ravel()


def model(tdem: Mapping[str, Any]) -> np.ndarray:
    """Resistivity as ``(stations, layers)``, surface first, NaN where unresolved."""
    rho = tdem.get("recovered_resistivity")
    if rho is None and tdem.get("recovered_conductivity") is not None:
        rho = 1.0 / np.asarray(tdem["recovered_conductivity"], dtype=float)
    rho = np.asarray(rho if rho is not None else [], dtype=float)
    return rho.reshape(1, -1) if rho.ndim == 1 else rho


def _layer_depths(tdem: Mapping[str, Any], n_layers: int
                  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Top and bottom depth (m) of each layer; the last one is the half-space."""
    thickness = _array(tdem, "thicknesses")
    if n_layers < 2 or thickness.size != n_layers - 1:
        return None, None
    top = np.concatenate([[0.0], np.cumsum(thickness)])
    return top, np.concatenate([np.cumsum(thickness), [np.inf]])


def _resolved(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values) & (values > 0)]


def depth_bands(tdem: Mapping[str, Any], values: Optional[np.ndarray] = None
                ) -> List[Dict[str, Any]]:
    """The model summarised over :data:`DEPTH_BANDS_M`, shallowest first.

    Each band holds the layers whose centre (the top, for the half-space) falls
    in it. ``coverage`` is the fraction of stations with a resolved cell there,
    which is how deep the survey can be read; ``values`` - water content, say -
    are summarised on the same cells instead of the resistivity.
    """
    rho = model(tdem)
    top, bottom = _layer_depths(tdem, rho.shape[1] if rho.ndim == 2 else 0)
    if top is None:
        return []
    centre = np.where(np.isfinite(bottom), 0.5 * (top + bottom), top)
    data = rho if values is None else np.asarray(values, dtype=float).reshape(rho.shape)
    edges = [edge for edge in DEPTH_BANDS_M if edge < centre.max()] + [np.inf]
    bands = []
    for low, high in zip(edges[:-1], edges[1:]):
        layers = (centre >= low) & (centre < high)
        if not layers.any():
            continue
        usable = np.isfinite(rho[:, layers]) & (rho[:, layers] > 0)
        cells = data[:, layers][usable]
        cells = cells[np.isfinite(cells)]
        bands.append({"top": low, "bottom": high,
                      "coverage": float(usable.any(axis=1).mean()),
                      "cells": cells})
    return bands


def _station_label(tdem: Mapping[str, Any], index: int) -> str:
    lines = _array(tdem, "line_numbers", int)
    ids = tdem.get("station_ids")
    ids = list(np.asarray(ids, dtype=object).ravel()) if ids is not None else []
    station = ids[index] if index < len(ids) and str(ids[index]).strip() else index + 1
    return (f"line {lines[index]} station {station}" if index < lines.size
            else f"sounding {index + 1}")


def poor_fits(tdem: Mapping[str, Any]) -> List[int]:
    """Soundings fitting worse than :data:`POOR_FIT_CHI2`, worst first."""
    chi2 = _array(tdem, "chi2_list")
    order = np.argsort(-np.nan_to_num(chi2, nan=-np.inf))
    return [int(i) for i in order if np.isfinite(chi2[i]) and chi2[i] > POOR_FIT_CHI2]


def _fit_word(chi2: float) -> str:
    """What a median chi-squared says about the fit, in a phrase."""
    if not np.isfinite(chi2):
        return "with a misfit that could not be computed"
    if chi2 < 0.5:
        return ("more closely than their stated errors, so the errors may be "
                "overestimated or the models fit some noise")
    if chi2 <= 2.0:
        return "to within their stated errors"
    return "less well than their stated errors allow"


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def describe(tdem: Mapping[str, Any]) -> Dict[str, Any]:
    """What the report's front matter and scope say about this survey."""
    lines = tdem.get("lines") or []
    system = tdem.get("system") or {}
    return {
        "data": source_name(tdem.get("source_file")),
        "instrument": system.get("instrument"),
        "title": TITLE if tdem.get("survey") else "Time-Domain Electromagnetic Sounding",
        "survey": (f"{tdem['n_soundings']} TEM soundings on {len(lines)} "
                   f"line{'s' if len(lines) != 1 else ''}" if tdem.get("survey")
                   else "One TEM sounding"),
        "scope": (f"Time-domain electromagnetic (TEM) data from "
                  f"{source_name(tdem.get('source_file')) or 'the supplied file'} were "
                  "inverted for the electrical resistivity of the ground beneath each "
                  "sounding."),
    }


def findings(tdem: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
             unit: str = "m") -> List[str]:
    """The principal findings, each a computed statement."""
    items: List[str] = []
    rho = model(tdem)
    resolved = _resolved(rho)
    if tdem.get("survey"):
        chi2 = float(tdem.get("chi2_sounding_median", np.nan))
        poor = poor_fits(tdem)
        lines = tdem.get("lines") or []
        items.append(
            f"{tdem['n_soundings']} soundings on {len(lines)} line"
            f"{'s' if len(lines) != 1 else ''} were inverted as laterally constrained 1D "
            f"models. They fit their data {_fit_word(chi2)}: median chi-squared "
            f"{chi2:.2f} per sounding (global {float(tdem.get('chi2', np.nan)):.2f})"
            + (f"; {len(poor)} soundings fit worse than chi-squared {POOR_FIT_CHI2:g}."
               if poor else "."))
    elif tdem.get("chi2") is not None:
        items.append(f"The sounding was inverted for a {rho.shape[1]}-layer model, fitting "
                     f"its data to chi-squared {float(tdem['chi2']):.2f}.")
    if resolved.size:
        p25, p75 = np.percentile(resolved, [25, 75])
        items.append(f"Resolved resistivity spans {resolved.min():.3g} to "
                     f"{resolved.max():.3g} ohm-m; half of the resolved cells lie between "
                     f"{p25:.3g} and {p75:.3g} ohm-m.")
    bands = [band for band in depth_bands(tdem)
             if band["coverage"] >= 0.5 and band["cells"].size]
    if len(bands) >= 2:
        medians = [float(np.median(band["cells"])) for band in bands]
        low, high = bands[int(np.argmin(medians))], bands[int(np.argmax(medians))]
        items.append(
            f"The most conductive part of the model is {_band(low['top'], low['bottom'], unit)} "
            f"deep (median {min(medians):.3g} ohm-m) and the most resistive "
            f"{_band(high['top'], high['bottom'], unit)} (median {max(medians):.3g} ohm-m), "
            "counting only depths that at least half of the soundings resolve.")
    doi = _array(tdem, "doi")
    doi = doi[np.isfinite(doi) & (doi > 0)]
    if doi.size:
        items.append(f"The median depth of investigation is {_length(np.median(doi), unit)} "
                     f"(range {_length(doi.min(), unit)} to {_length(doi.max(), unit)}); "
                     "cells below it are blank in the figures and excluded from every "
                     "number in this report.")
    if tdem.get("water_content_mean") is not None:
        mean = _array(tdem, "water_content_mean")
        std = _array(tdem, "water_content_std")
        reliability = water_content_reliability(tdem, config or {})
        items.append(f"Converted to water content, the model gives {np.nanmin(mean):.3f} to "
                     f"{np.nanmax(mean):.3f} (mean uncertainty ±{np.nanmean(std):.3f})."
                     + (f" {reliability['sentence']}" if reliability else ""))
    return items


def confidence(tdem: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    """How far the findings can be trusted, in a paragraph."""
    parts = []
    if tdem.get("survey"):
        unresolved = int(tdem.get("unresolved_soundings") or 0)
        failed = int(tdem.get("failed_soundings") or 0)
        parts.append(f"The fit statistics describe {tdem['n_soundings'] - failed} "
                     "soundings with a model"
                     + (f"; {failed} could not be inverted" if failed else "")
                     + (f" and {unresolved} resolved no layer above their depth of "
                        "investigation" if unresolved else "") + ".")
    parts.append("Each model is one-dimensional: it is valid where the ground is close "
                 "to horizontally layered within the footprint of the loop and receiver, "
                 "and a lateral change between soundings is drawn as a smooth transition "
                 "rather than located.")
    parts.append("A layered TEM model is not unique: a thin conductive layer trades "
                 "thickness against resistivity, so interfaces are better read as "
                 "transitions than as boundaries.")
    if tdem.get("water_content_mean") is not None:
        relationship = tdem.get("petrophysical_relationship")
        parts.append("The water content is calibrated by the petrophysical parameters the "
                     "request gave." if relationship == "user" else
                     "The water content rests on petrophysical parameters that were not "
                     "calibrated for this site, so it is an indication, not a measurement.")
    return " ".join(parts)


def _data_rows(tdem: Mapping[str, Any], unit: str) -> List[Tuple[str, Any]]:
    """What was measured, for the method section."""
    system = tdem.get("system") or {}
    protocol = tdem.get("protocol") or {}
    source = source_name(tdem.get("source_file"))
    if not tdem.get("survey"):
        times = tdem.get("time_range") or []
        return [
            ("Data", source),
            ("Time gates", tdem.get("n_data")),
            ("Time range", f"{times[0] * 1e6:.3g} µs to {times[1] * 1e3:.3g} ms"
             if len(times) == 2 else None),
            ("Transmitter", f"circular loop of radius {_length(tdem['source_radius'], unit)}, "
                            "step-off waveform" if tdem.get("source_radius") else None),
        ]
    lines = tdem.get("lines") or []
    total = int(tdem.get("n_soundings_total") or tdem["n_soundings"])
    rows: List[Tuple[str, Any]] = [
        ("Data", f"{source} ({tdem['source_format']})" if tdem.get("source_format") else source),
        ("Instrument", system.get("instrument")),
        ("Acquisition protocol", protocol.get("protocol_file")),
        ("Soundings", f"{tdem['n_soundings']} on {len(lines)} line{'s' if len(lines) != 1 else ''}"
                      + (f" ({total} in the survey)" if total > tdem["n_soundings"] else "")),
        ("Moments", "low and high moment, inverted jointly" if tdem.get("tem_moment") == "LM+HM"
         else {"HM": "high moment", "LM": "low moment"}.get(str(tdem.get("tem_moment")))),
    ]
    area = system.get("loop_area")
    if area:
        side = float(to_display_length(float(np.sqrt(area)), unit))
        rows.append(("Transmitter loop", f"{side ** 2:.3g} {unit}² "
                     + (f"({int(system['loop_turns'])} turns)" if system.get("loop_turns") else "")))
    separation = _array(tdem, "tx_rx_distances")
    separation = separation[np.isfinite(separation) & (separation > 0)]
    if separation.size:
        rows.append(("Transmitter-receiver separation",
                     f"median {_length(np.median(separation), unit)}, "
                     f"{_length(separation.min(), unit)} to {_length(separation.max(), unit)} "
                     "as measured at each station"
                     + (f" (nominal {_length(system['tx_rx_sep_nominal'], unit)})"
                        if system.get("tx_rx_sep_nominal") else "")))
    counts = _array(tdem, "data_count_list")
    counts = counts[counts > 0]
    if counts.size:
        rows.append(("Gates per sounding",
                     f"{int(counts.min())} at every sounding" if counts.min() == counts.max()
                     else f"median {int(np.median(counts))}, "
                          f"{int(counts.min())} to {int(counts.max())}"))
    rel = (tdem.get("inversion_settings") or {}).get("rel_error")
    if rel is not None:
        rows.append(("Data errors", f"recorded stack errors, with a {100 * float(rel):.3g}% "
                                    "uniform error from the acquisition protocol"))
    return rows


def _inversion_rows(tdem: Mapping[str, Any], unit: str) -> List[Tuple[str, Any]]:
    """How the models were obtained, for the method section."""
    rho = model(tdem)
    n_layers = rho.shape[1] if rho.ndim == 2 else 0
    top, _ = _layer_depths(tdem, n_layers)
    layers = (f"{n_layers}, the first {_length(top[1], unit)} thick, over a half-space "
              f"from {_length(top[-1], unit)}" if top is not None else n_layers or None)
    if not tdem.get("survey"):
        return [("Scheme", "1D inversion of a single sounding"
                 + (", smooth (L2) then sparse (IRLS)" if tdem.get("use_irls") else "")),
                ("Forward model", "SimPEG 1D time-domain simulation (Cockett et al., 2015)"),
                ("Layers", layers)]
    settings = tdem.get("inversion_settings") or {}
    robust = tdem.get("robust") or {}
    start = ("the best-fitting half-space, chosen from the data at each station"
             if settings.get("auto_starting_model", True) else
             f"a {float(settings.get('starting_resistivity', 100.0)):.3g} ohm-m half-space")
    rows: List[Tuple[str, Any]] = [
        ("Scheme", LCI_LABELS.get(str(tdem.get("lci_mode")), tdem.get("lci_mode"))),
        ("Forward model", "SimPEG 1D time-domain simulation (Cockett et al., 2015) with "
                          "the waveform, gates and filters the data file records"),
        ("Layers", layers),
        ("Vertical smoothness", settings.get("smoothness")),
        ("Lateral smoothness", settings.get("lateral_smoothness")
         if tdem.get("lci_mode") != "off" else None),
        ("Starting model", start),
        ("Error weighting", "robust: a gate the model cannot explain gets a bounded increase "
                            "in its error instead of being removed"
                            + (f"; {robust['downweighted']} of {robust['n_start']} gates were "
                               "down-weighted" if robust.get("downweighted") is not None
                               and robust.get("n_start") else "")
         if settings.get("robust_errors") else "the stated data errors"),
        ("Settings from", ("the inversion settings stored in the project, over the package's "
                           "ground-TEM preset" if tdem.get("settings_source") == "project"
                           else "the package's ground-TEM preset")
                          + (f"; set by the request: {', '.join(tdem['overrides'])}"
                             if tdem.get("overrides") else "")),
        ("Depth of investigation", "estimated per sounding from the model sensitivity"),
    ]
    return rows


def method_section(tdem: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    """What was measured, how it was inverted, and what that assumes.

    The assumptions sit here, beside the method they belong to: the workflow
    audit adds the run's general "Limitations and Uncertainty" section before
    the recommendations, and a second limitations section beside it would
    read as two lists of the same thing.
    """
    sub = level + "#"
    return (f"{level} Data and Method\n\n{sub} Data\n\n"
            + facts([row for row in _data_rows(tdem, unit) if row[1] not in (None, "")],
                    "Item", "Value")
            + f"\n{sub} Inversion\n\n"
            + facts([row for row in _inversion_rows(tdem, unit) if row[1] not in (None, "")],
                    "Item", "Value")
            + f"\n{sub} Assumptions and Limits\n\n" + bullets(limitations(tdem)))


def results_section(tdem: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    """The fit, the lines and the model by depth, in tables."""
    sub = level + "#"
    section = f"{level} Results\n\n"
    rho = model(tdem)
    if tdem.get("survey"):
        poor = poor_fits(tdem)
        robust = tdem.get("robust") or {}
        fit = [
            ("Soundings inverted", tdem["n_soundings"]),
            ("Could not be inverted", tdem.get("failed_soundings") or 0),
            ("Resolved nothing above the depth of investigation",
             tdem.get("unresolved_soundings") or 0),
            ("Median chi-squared per sounding", f"{float(tdem['chi2_sounding_median']):.3g}"),
            ("Global chi-squared", f"{float(tdem['chi2']):.3g}"),
            (f"Soundings with chi-squared above {POOR_FIT_CHI2:g}", len(poor)),
        ]
        if robust.get("chi2_effective") is not None and robust.get("enabled"):
            fit.append(("Chi-squared with robust errors", f"{float(robust['chi2_effective']):.3g}"))
        section += f"{sub} Data Fit\n\n" + facts(fit, "Measure", "Value")
        if poor:
            shown = ", ".join(f"{_station_label(tdem, i)} ({_array(tdem, 'chi2_list')[i]:.3g})"
                              for i in poor[:8])
            section += (f"\nThe worst-fitting soundings are {shown}"
                        + (f", and {len(poor) - 8} more" if len(poor) > 8 else "") + ".\n")
        section += "\n" + _line_table(tdem, unit, sub)
    elif tdem.get("chi2") is not None:
        section += (f"{sub} Data Fit\n\nThe model fits the {tdem.get('n_data', '')} gates "
                    f"to chi-squared {float(tdem['chi2']):.3g}.\n\n")
    bands = depth_bands(tdem)
    if bands:
        rows = []
        for band in bands:
            cells = band["cells"]
            rows.append([_band(band["top"], band["bottom"], unit),
                         f"{100 * band['coverage']:.0f}%",
                         f"{np.median(cells):.3g}" if cells.size else None,
                         (lambda p: f"{p[0]:.3g} - {p[1]:.3g}")(np.percentile(cells, [10, 90]))
                         if cells.size else None])
        section += (f"\n{sub} Resistivity by Depth\n\n"
                    "Summarised over the resolved cells of every sounding; the coverage "
                    "column is the share of soundings that resolve that depth.\n\n"
                    + table(["Depth", "Coverage", "Median (ohm-m)", "10th-90th percentile"],
                            rows, align=["left", "right", "right", "right"]))
    elif rho.size:
        top, bottom = _layer_depths(tdem, rho.shape[1])
        if top is not None and rho.shape[0] == 1:
            section += (f"\n{sub} Layered Model\n\n"
                        + table(["Depth", "Resistivity (ohm-m)"],
                                [[_band(t, b, unit), f"{r:.3g}"]
                                 for t, b, r in zip(top, bottom, rho[0])],
                                align=["left", "right"]))
    section += _map_section(tdem, unit, sub)
    return section


def _map_section(tdem: Mapping[str, Any], unit: str, sub: str) -> str:
    """How the plan-view maps were made, and how far they can be trusted."""
    slices = tdem.get("depth_slices") or {}
    if not slices or not tdem.get("map_figure"):
        return ""
    depths = slices.get("depths", [])
    background = slices.get("basemap") or {}
    kriged = bool(slices.get("variograms"))
    rows = [("Depths mapped", ", ".join(_length(d, unit) for d in depths)),
            ("Interpolation",
             "Ordinary kriging of log10 resistivity (Matheron, 1963); a spherical, "
             "exponential or Gaussian variogram with a nugget fitted at each depth by "
             "pair-count weighted least squares, the best-fitting kept"
             if kriged else str(slices.get("method", ""))),
            ("Coverage", f"blank beyond {_length(float(slices['max_distance']), unit)} from the "
                         "nearest sounding resolving that depth, and outside the soundings' "
                         "outline" if slices.get("max_distance") else
                         "filled across the soundings' outline, gaps between lines included; "
                         "soundings below their depth of investigation are left out at that "
                         "depth"),
            ("Grid cell", _length(float(slices.get("cell", 0.0)), unit)),
            ("Coordinates", slices.get("coordinate_system") or "survey coordinates"),
            ("Basemap", str(background.get("attribution") or "").removeprefix("Basemap: ")
             or "none available")]
    section = (f"\n{sub} Resistivity in Plan View\n\n"
               "Resistivity at fixed depths below ground, kriged from the soundings that "
               "resolve each depth. The colours between survey lines, and wherever the "
               "soundings do not reach a depth, are an estimate from the soundings around "
               "them, not a measurement there; the kriging uncertainty maps among the figures "
               "show, cell by cell, how far that estimate can be off.\n\n"
               + facts(rows, "Item", "Value"))
    if not kriged:
        return section
    used = slices.get("soundings_used") or []
    spread = slices.get("std_decades") or []
    table_rows = []
    for k, depth in enumerate(depths):
        model = (slices["variograms"][k:k + 1] or [None])[0]
        std = (spread[k:k + 1] or [None])[0]
        if model is None:
            kind, reach, nugget = "one resistivity throughout", None, None
        else:
            sill = model["sill"]
            kind = str(model["model"]).capitalize() + ("" if model.get("fitted", True)
                                     else " (assumed: too few sounding pairs to fit)")
            reach = _length(float(model["range"]), unit)
            nugget = f"{100 * model['nugget'] / sill:.0f}%" if sill > 0 else None
        table_rows.append([
            _length(depth, unit), used[k] if k < len(used) else None, kind, reach, nugget,
            f"{std['median']:.2f} (factor {10 ** std['median']:.2f})" if std else None,
            f"{std['p90']:.2f} (factor {10 ** std['p90']:.2f})" if std else None])
    return (section + "\n"
            + table(["Depth", "Soundings", "Variogram", "Range", "Nugget share",
                     "Kriging std, median (decades)", "Kriging std, 90th percentile"],
                    table_rows,
                    align=["left", "right", "left", "right", "right", "right", "right"])
            + "\nThe range is the distance over which resistivity at that depth stops being "
              "related; the nugget share is the part of its variance that changes even between "
              "neighbouring soundings. A standard deviation of 0.1 decade means the mapped "
              "resistivity is good to about a factor of 1.26 either way. It is the "
              "interpolation's uncertainty only and adds to that of each sounding's model. The "
              "standard deviation of every cell is saved beside the resistivity grids "
              "(`_std.asc`).\n")


def _line_table(tdem: Mapping[str, Any], unit: str, sub: str) -> str:
    """One row per survey line."""
    lines = _array(tdem, "line_numbers", int)
    if not lines.size:
        return ""
    rho = model(tdem)
    positions = _array(tdem, "positions")
    chi2 = _array(tdem, "chi2_list")
    doi = _array(tdem, "doi")
    rows = []
    for line in dict.fromkeys(lines.tolist()):
        index = np.flatnonzero(lines == line)
        cells = _resolved(rho[index]) if rho.shape[0] == lines.size else np.array([])
        span = (positions[index].max() - positions[index].min()
                if positions.size == lines.size and index.size > 1 else np.nan)
        depth = doi[index] if doi.size == lines.size else np.array([])
        depth = depth[np.isfinite(depth) & (depth > 0)]
        fit = chi2[index] if chi2.size == lines.size else np.array([])
        fit = fit[np.isfinite(fit)]
        rows.append([
            line, index.size,
            _length(span, unit) if np.isfinite(span) else None,
            f"{np.median(fit):.3g}" if fit.size else None,
            _length(np.median(depth), unit) if depth.size else None,
            (lambda p: f"{p[0]:.3g} - {p[1]:.3g}")(np.percentile(cells, [10, 90]))
            if cells.size else "nothing resolved",
        ])
    return (f"{sub} Survey Lines\n\n"
            + table(["Line", "Soundings", "Length", "Median chi-squared",
                     "Median depth of investigation", "Resistivity 10th-90th (ohm-m)"],
                    rows, align=["right", "right", "right", "right", "right", "right"]))


def water_content_section(tdem: Mapping[str, Any], steps: Optional[Sequence[Any]] = None,
                          config: Optional[Mapping[str, Any]] = None, unit: str = "m",
                          output_dir: Optional[str] = None, level: str = "##") -> str:
    """The water content converted from the model, with its uncertainty and assumptions.

    Empty when there is none; a conversion that was asked for and failed is
    stated once for the run, not per method.
    """
    if tdem.get("water_content_mean") is None:
        return ""
    step = (list(steps or []) or [{}])[0] or {}
    section = f"{level} Water Content\n\n"
    if step.get("layering"):
        section += f"{step['layering']}\n\n"
    reliability = water_content_reliability(tdem, config or {})
    if reliability:
        section += f"{reliability['sentence']}\n\n"
    statement = tdem.get("prior_statement") or step.get("prior_statement")
    if statement:
        relationship = tdem.get("petrophysical_relationship") or step.get(
            "petrophysical_relationship")
        section += ("" if relationship == "user" else "**Warning:** ") + f"{statement}\n\n"
    mean = np.asarray(tdem["water_content_mean"], dtype=float)
    std = np.asarray(tdem.get("water_content_std"), dtype=float)
    rows = []
    for band_mean, band_std in zip(depth_bands(tdem, mean), depth_bands(tdem, std)):
        if not band_mean["cells"].size:
            continue
        low, high = np.percentile(band_mean["cells"], [10, 90])
        rows.append([_band(band_mean["top"], band_mean["bottom"], unit),
                     f"{np.mean(band_mean['cells']):.3f}", f"{low:.3f} - {high:.3f}",
                     f"±{np.mean(band_std['cells']):.3f}" if band_std["cells"].size else None])
    if rows:
        section += table(["Depth", "Mean", "10th-90th percentile", "Mean uncertainty"],
                         rows, align=["left", "right", "right", "right"])
    else:
        section += facts([("Mean", f"{np.nanmean(mean):.3f}"),
                          ("Range", f"{np.nanmin(mean):.3f} - {np.nanmax(mean):.3f}"),
                          ("Mean uncertainty", f"±{np.nanmean(std):.3f}")], "Measure", "Value")
    path = tdem.get("water_content_table")
    if path:
        section += f"\nEvery station and layer is listed in `{_relative(path, output_dir)}`.\n"
    return section + ("\nWater content is derived from the resistivity through a "
                      "petrophysical relationship, so it carries the uncertainty of that "
                      "relationship as well as that of the inversion.\n")


def figure_caption(tdem: Mapping[str, Any]) -> str:
    """The caption of the TDEM figure ``TDEMAgent`` drew."""
    if tdem.get("survey"):
        return ("Resistivity sections, one per survey line, from the laterally "
                "constrained inversion, on one colour scale. Cells below each sounding's "
                "depth of investigation are blank. The sections are drawn against ground "
                "elevation where the soundings carry one, otherwise against depth.")
    return ("Recovered resistivity model (left), observed and predicted data (centre) "
            "and normalised residuals (right).")


def map_caption(tdem: Mapping[str, Any]) -> str:
    """The caption of the plan-view maps."""
    slices = tdem.get("depth_slices") or {}
    background = (slices.get("basemap") or {}).get("attribution")
    coverage = ("left blank where none is near (the distance is stated under Resistivity in "
                "Plan View)." if slices.get("max_distance") else
                "filled across the survey outline; the next figure shows how firm each "
                "cell is.")
    return ("Resistivity in plan view at fixed depths below ground, on the colour scale of "
            "the sections"
            + (f", over {background.removeprefix('Basemap: ')}" if background else "")
            + f", kriged from the soundings (black dots) that resolve each depth and {coverage}")


def uncertainty_caption(tdem: Mapping[str, Any]) -> str:
    """The caption of the kriging-uncertainty maps."""
    return ("Kriging standard deviation of the plan-view maps, in decades of log10 "
            "resistivity, with the factor it stands for (0.1 decade is a factor of 1.26 "
            "either way). It is smallest beside the soundings (black dots) that resolve each "
            "depth and grows into the gaps between lines and where the soundings do not "
            "reach that depth; it measures the interpolation only, not the inversion.")


def figures(tdem: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``(path, caption)`` for each figure the inversion drew."""
    drawn = []
    path = tdem.get("visualization_file")
    if path and os.path.exists(str(path)):
        drawn.append((str(path), figure_caption(tdem)))
    maps = tdem.get("map_figure")
    if maps and os.path.exists(str(maps)):
        drawn.append((str(maps), map_caption(tdem)))
    spread = tdem.get("map_uncertainty_figure")
    if spread and os.path.exists(str(spread)):
        drawn.append((str(spread), uncertainty_caption(tdem)))
    return drawn


def interpretation(tdem: Mapping[str, Any]) -> Optional[str]:
    """The language model's reading of the results, when one was written."""
    text = tdem.get("interpretation")
    return str(text).strip() if isinstance(text, str) and text.strip() else None


def limitations(tdem: Mapping[str, Any]) -> List[str]:
    """What qualifies every number in the report."""
    items = [
        "**One-dimensional models.** Each sounding is explained by horizontal layers "
        "beneath it. Steep structure, buried infrastructure and lateral changes within "
        "the measurement footprint violate that assumption and can appear as false layers.",
        "**Depth of investigation.** It is an estimate from the model sensitivity, not a "
        "hard limit; resistivity near it is less certain than above it.",
        "**Equivalence.** Different layered models fit TEM data equally well; thin "
        "conductive layers in particular trade thickness for resistivity.",
    ]
    if tdem.get("survey") and tdem.get("lci_mode") != "off":
        items.append("**Lateral constraints.** Neighbouring soundings are tied together, "
                     "which suppresses noise but also smooths genuine lateral change "
                     "over a few station spacings.")
    if tdem.get("map_figure"):
        items.append("**Plan-view maps.** The maps are filled across the survey outline, so "
                     "between survey lines, and on a deep map wherever the soundings do not "
                     "reach that depth, the colours are kriged from soundings farther away; the "
                     "uncertainty maps say where. A feature narrower than the line spacing, "
                     "or lying between lines, can be missed or smeared across the gap. The "
                     "variogram is isotropic, so structure elongated along or across the lines "
                     "is not favoured, and with soundings far closer along a line than "
                     "between lines it is mostly fitted from along-line pairs.")
    if (tdem.get("robust") or {}).get("enabled"):
        items.append("**Robust error weighting.** Gates the model could not explain were "
                     "given larger errors; such gates may carry real structure the 1D model "
                     "cannot represent.")
    return items


def recommendations(tdem: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    """Next steps that follow from this run."""
    config = config or {}
    items = []
    poor = poor_fits(tdem)
    if poor:
        items.append(f"**Review the {len(poor)} poorly fitting soundings** "
                     f"(chi-squared above {POOR_FIT_CHI2:g}; see the Data Fit table) for "
                     "noise, cultural interference or 2D/3D effects before interpreting the "
                     "model beneath them.")
    unresolved = int(tdem.get("unresolved_soundings") or 0)
    if unresolved:
        items.append(f"**Check the {unresolved} sounding{'s' if unresolved > 1 else ''} "
                     f"that resolved nothing** above {'their' if unresolved > 1 else 'its'} "
                     "depth of investigation; the late gates may be too noisy to use, or "
                     "the station may need re-measuring.")
    lines = _array(tdem, "line_numbers", int)
    single = [int(v) for v in dict.fromkeys(lines.tolist()) if np.count_nonzero(lines == v) == 1]
    if single:
        if len(single) > 1:
            names = ", ".join(map(str, single[:-1])) + f" and {single[-1]}"
            items.append(f"**Lines {names} each hold a single sounding**, which no "
                         "lateral constraint supports; treat their models as stand-alone "
                         "1D estimates.")
        else:
            items.append(f"**Line {single[0]} holds a single sounding**, which no lateral "
                         "constraint supports; treat its model as a stand-alone 1D estimate.")
    if tdem.get("water_content_mean") is not None and tdem.get(
            "petrophysical_relationship") != "user":
        items.append("**Calibrate the petrophysical relationship** with samples, logs or "
                     "soil-moisture measurements from the site; the generated parameters "
                     "are the largest contributor to the water-content uncertainty.")
    return items


#: What each file the line inversion writes holds, for the appendix.
FILE_CONTENTS = {
    "resistivity_section.npz": "the section, its sensitivity and depth of investigation",
    "model_cells.csv": "one row per sounding and layer, with coordinates",
    "soundings.csv": "one row per sounding: position, fit and depth of investigation",
    "lci_report.json": "the laterally constrained solve: iterations and settings",
    "robust_errors.json": "the robust error weighting, per gate",
    "robust_gate_errors.csv": "each gate's original and effective error",
    "shallow_prior.json": "the resistive-background prior: settings and the stations it "
                          "acted on",
    "shallow_prior.csv": "the prior's signal and noise ratios, per sounding",
    "water_content_by_layer.csv": "water content and its uncertainty, per station and layer",
}


def output_files(tdem: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """``(path, contents)`` for each file the TDEM steps wrote."""
    paths = [str(path) for path in tdem.get("data_paths") or [] if path]
    if tdem.get("water_content_table"):
        paths.append(str(tdem["water_content_table"]))
    described = [(path, FILE_CONTENTS.get(os.path.basename(path))) for path in dict.fromkeys(paths)]
    grid_system = (tdem.get("depth_slices") or {}).get("coordinate_system") or "survey coordinates"
    described += [(str(path), ("kriging standard deviation of log10 resistivity (decades)"
                               if str(path).endswith("_std.asc") else "resistivity (ohm-m)")
                   + f" at one depth as an ESRI ASCII grid, for GIS ({grid_system})")
                  for path in tdem.get("map_grids") or []]
    return described
