"""The seismic refraction part of a workflow report.

Written from what :class:`~PyHydroGeophysX.agents.seismic_agent.SeismicAgent`
returns: a travel-time tomography on pyGIMLi's ``TravelTimeManager``, from picks
or from a SEG-Y file it picked itself, and the velocity contours it traced as
interfaces. :mod:`PyHydroGeophysX.agents._survey_report` lays the sections out.
Every number is computed from the results; no sentence is written by a model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._document import bullets, facts, table
from ._report_text import band, given, length, plural, source_name, spread

KEY = "seismic"
ARTIFACT = "seismic_results"
TITLE = "Seismic Refraction Survey"
FIGURE_TAG = "S"

#: Depth band edges (m) below the ground the velocities are summarised over.
DEPTH_BANDS_M = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)

#: Chi-squared above which the picks are said to be fitted poorly.
POOR_FIT_CHI2 = 2.0


def _covered_cells(results: Mapping[str, Any]
                   ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``(velocity, depth below ground, x)`` of the ray-covered cells, or None.

    pyGIMLi's standardised coverage is 1 where rays pass and 0 where none do;
    a cell no ray crosses holds the starting model, not a measurement.
    """
    mesh = results.get("mesh")
    velocity = np.asarray(results.get("velocity_model") if results.get("velocity_model")
                          is not None else [], dtype=float).ravel()
    if mesh is None or not velocity.size:
        return None
    try:
        from PyHydroGeophysX.core.section_geometry import surface_line

        centres = np.asarray([[c[0], c[1]] for c in mesh.cellCenters()], dtype=float)
        sx, sz = surface_line(mesh)
        order = np.argsort(sx)
        ground = np.interp(centres[:, 0], np.asarray(sx)[order], np.asarray(sz)[order])
    except Exception:  # noqa: BLE001 - the summary needs a readable mesh
        return None
    if centres.shape[0] != velocity.size:
        return None
    coverage = np.asarray(results.get("coverage") if results.get("coverage") is not None
                          else np.ones(velocity.size), dtype=float).ravel()
    covered = coverage > 0.5 if coverage.size == velocity.size else np.ones(velocity.size, bool)
    return velocity[covered], (ground - centres[:, 1])[covered], centres[covered, 0]


def _interfaces(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each traced velocity contour, with its extent and depth below ground."""
    traced = results.get("interfaces") or {}
    mesh = results.get("mesh")
    ground = None
    if mesh is not None:
        try:
            from PyHydroGeophysX.core.section_geometry import surface_line

            sx, sz = surface_line(mesh)
            order = np.argsort(sx)
            ground = (np.asarray(sx)[order], np.asarray(sz)[order])
        except Exception:  # noqa: BLE001
            ground = None
    rows = []
    for threshold, line in sorted(traced.items(), key=lambda item: float(item[0])):
        x = np.asarray(line.get("x", []), dtype=float).ravel()
        z = np.asarray(line.get("z", []), dtype=float).ravel()
        keep = np.isfinite(x) & np.isfinite(z)
        x, z = x[keep], z[keep]
        if not x.size:
            continue
        depth = np.interp(x, *ground) - z if ground is not None else None
        rows.append({"threshold": float(threshold), "x": x, "z": z, "depth": depth})
    return rows


def describe(results: Mapping[str, Any]) -> Dict[str, Any]:
    """What the report's front matter and scope say about this survey."""
    raw = results.get("raw_seismic_file")
    source = source_name(raw or results.get("source_file"))
    how = ("inverted by travel-time tomography for the P-wave velocity of the ground "
           "beneath the line.")
    return {
        "data": source,
        "instrument": None,
        "survey": (f"{plural(int(results.get('n_shots') or 0), 'shot')}, "
                   f"{plural(int(results.get('n_receivers') or 0), 'receiver')}, "
                   f"{plural(int(results.get('n_data') or 0), 'travel time')}"),
        "scope": (f"First breaks were picked from the SEG-Y file {source} and {how}" if raw
                  else f"Seismic refraction travel times from {source or 'the supplied file'} "
                       f"were {how}"),
    }


def _fit_word(chi2: Optional[float]) -> str:
    if chi2 is None or not np.isfinite(chi2):
        return "with a misfit that was not reported"
    if chi2 < 0.5:
        return "more closely than their stated errors, which may be overestimated"
    if chi2 <= POOR_FIT_CHI2:
        return "to within their stated errors"
    return "less well than their stated errors allow"


def findings(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
             unit: str = "m") -> List[str]:
    """The principal findings, each a computed statement."""
    items = []
    chi2, rrms = results.get("chi2"), results.get("rrms")
    items.append(f"The velocity model fits the {plural(int(results.get('n_data') or 0), 'travel time')} "
                 f"{_fit_word(chi2)}"
                 + (f": chi-squared {float(chi2):.2f}" if chi2 is not None else "")
                 + (f", relative RMS {float(rrms):.1f}%" if rrms is not None else "") + ".")
    cells = _covered_cells(results)
    if cells is not None and cells[0].size:
        velocity, depth, _ = cells
        items.append(f"Where rays pass, velocity rises from {np.percentile(velocity, 5):.0f} to "
                     f"{np.percentile(velocity, 95):.0f} m/s (5th-95th percentile); rays reach "
                     f"{length(np.max(depth), unit)} below ground.")
    for line in _interfaces(results):
        if line["depth"] is not None:
            items.append(f"The {line['threshold']:.0f} m/s contour lies "
                         f"{length(np.min(line['depth']), unit)} to "
                         f"{length(np.max(line['depth']), unit)} below ground (median "
                         f"{length(np.median(line['depth']), unit)}) along "
                         f"{length(np.ptp(line['x']), unit)} of the line.")
    return items


def confidence(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    """How far the findings can be trusted, in a paragraph."""
    parts = ["Refraction tomography sees only where velocity increases with depth: a slower "
             "layer beneath a faster one, or a layer too thin to carry a first arrival, does "
             "not appear.",
             "The smoothness constraint draws velocity changes as gradients, so a traced "
             "contour marks where the velocity passes a value, not a sharp boundary."]
    if results.get("geometry_warnings"):
        parts.append("The survey geometry raised warnings; they are listed with the run's "
                     "warnings.")
    return " ".join(parts)


def _interval(microseconds: float) -> str:
    """A sample interval as written: microseconds below one millisecond, all its digits.

    >>> _interval(31.25), _interval(250.0), _interval(1000.0)
    ('31.25 µs', '250 µs', '1 ms')
    """
    return (f"{microseconds:g} µs" if microseconds < 1000
            else f"{microseconds / 1000:g} ms")


def _left_out(results: Mapping[str, Any], unit: str) -> Optional[str]:
    """The shots the reciprocity check left out, and by how much each was off."""
    dropped = results.get("dropped_shots") or []
    if not dropped:
        return None
    return ("; ".join(f"shot at {length(float(shot['source_x']), unit)}, "
                      f"{abs(float(shot['median_difference_ms'])):.3g} ms "
                      f"{'early' if float(shot['median_difference_ms']) < 0 else 'late'} "
                      f"against {shot['pairs']} reciprocal times" for shot in dropped)
            + " - each a timing error of its own record, so its picks were not inverted")


def _corrected(results: Mapping[str, Any]) -> Optional[str]:
    """The picks made again by AIC (Maeda, 1985), and against what."""
    parts = []
    if results.get("repicked_picks"):
        parts.append(f"{results['repicked_picks']} that strayed from their shot's first-arrival "
                     "curve, along it")
    if results.get("neighbour_repicked"):
        parts.append(f"{results['neighbour_repicked']} that disagreed with the neighbouring "
                     "shots' picks at the same geophone, where those put them")
    if not parts:
        return None
    return "picked again by the AIC picker (Maeda, 1985): " + "; ".join(parts)


def _single_left_out(results: Mapping[str, Any]) -> Optional[str]:
    """The single picks left out, and for breaking what."""
    parts = []
    if results.get("rejected_picks"):
        parts.append(f"{results['rejected_picks']} that break their shot's first-arrival curve "
                     "- a first arrival cannot come earlier farther from the shot")
    if results.get("neighbour_rejected"):
        parts.append(f"{results['neighbour_rejected']} that still disagree with the neighbouring "
                     "shots at their geophone")
    if not parts:
        return None
    return "; ".join(parts) + " - marked in the picks figure"


def _data_rows(results: Mapping[str, Any], unit: str) -> List[Tuple[str, Any]]:
    raw = results.get("raw_seismic_file")
    meta = results.get("segy_metadata") or {}
    return given([
        ("Data", source_name(raw or results.get("source_file"))),
        ("Picks", ("first breaks picked automatically from the traces; written to "
                   f"`{os.path.basename(str(results.get('first_break_picks_file') or ''))}`")
         if raw else "travel times as supplied"),
        ("Picks corrected", _corrected(results)),
        ("Shots left out", _left_out(results, unit)),
        ("Picks left out", _single_left_out(results)),
        ("SEG-Y", (f"{meta.get('trace_count')} traces, {meta.get('samples_per_trace')} samples "
                   f"at {_interval(float(meta['sample_interval_us']))}")
         if raw and meta.get("sample_interval_us") else None),
        ("Shots", results.get("n_shots")),
        ("Receivers", results.get("n_receivers")),
        ("Travel times", results.get("n_data")),
    ])


def _velocity_limits(limits: Any) -> Optional[Tuple[float, float]]:
    """The velocity bounds, in m/s, however they were given.

    pyGIMLi takes them as velocity but the run's settings can carry slowness
    (s/m), which printed "0.000125 to 0.00333 m/s".

    >>> _velocity_limits([1 / 8000, 1 / 300])
    (300.0, 8000.0)
    """
    try:
        low, high = sorted(float(v) for v in limits)
    except (TypeError, ValueError):
        return None
    if high < 1.0:
        low, high = sorted((1.0 / low, 1.0 / high))
    return round(low, 6), round(high, 6)


def _inversion_rows(results: Mapping[str, Any], unit: str) -> List[Tuple[str, Any]]:
    params = results.get("inversion_params") or {}
    limits = _velocity_limits(params.get("limits"))
    return given([
        ("Scheme", "Smoothness-constrained travel-time tomography, pyGIMLi "
                   "TravelTimeManager (Rücker et al., 2017)"),
        ("Mesh", f"{results['n_cells']} triangular cells to "
                 f"{length(params['paraDepth'], unit)} depth" if results.get("n_cells")
         and params.get("paraDepth") else None),
        ("Smoothness (lambda)", params.get("lam")),
        ("Vertical-to-horizontal smoothness (zWeight)", params.get("zWeight")),
        ("Starting model", f"velocity increasing from {params['vTop']:g} m/s at the surface to "
                           f"{params['vBottom']:g} m/s at depth"
         if params.get("vTop") is not None and params.get("vBottom") is not None else None),
        ("Velocity limits", f"{limits[0]:g} to {limits[1]:g} m/s" if limits else None),
        ("Interfaces", ", ".join(f"{float(v):g} m/s" for v in results.get("velocity_thresholds")
                                 or []) + " contours of the velocity model"
         if results.get("velocity_thresholds") else None),
    ])


def limitations(results: Mapping[str, Any]) -> List[str]:
    return [
        "**Velocity inversions and hidden layers.** A layer slower than the one above it, or "
        "too thin to carry a first arrival, is invisible to refraction.",
        "**Ray coverage.** Below the deepest rays, and at the ends of the line, the model is "
        "the starting gradient; the figure shades those cells.",
        "**Smoothing.** Interfaces are contours of a smooth model and their depth depends on "
        "the velocity chosen to trace.",
    ]


def method_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    return (f"{level} Data and Method\n\n{sub} Data\n\n"
            + facts(_data_rows(results, unit), "Item", "Value")
            + f"\n{sub} Inversion\n\n" + facts(_inversion_rows(results, unit), "Item", "Value")
            + f"\n{sub} Assumptions and Limits\n\n" + bullets(limitations(results)))


def results_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    chi2, rrms = results.get("chi2"), results.get("rrms")
    span = results.get("velocity_range") or []
    section = (f"{level} Results\n\n{sub} Data Fit\n\n"
               + facts(given([
                   ("Travel times", results.get("n_data")),
                   ("Chi-squared", f"{float(chi2):.3g}" if chi2 is not None else None),
                   ("Relative RMS", f"{float(rrms):.3g}%" if rrms is not None else None),
                   ("Velocity range of the model",
                    f"{float(span[0]):.0f} to {float(span[1]):.0f} m/s" if len(span) == 2 else None),
               ]), "Measure", "Value"))
    cells = _covered_cells(results)
    if cells is not None and cells[0].size:
        velocity, depth, _ = cells
        edges = [edge for edge in DEPTH_BANDS_M if edge < depth.max()] + [np.inf]
        rows = []
        for top, bottom in zip(edges[:-1], edges[1:]):
            inside = (depth >= top) & (depth < bottom)
            if inside.any():
                rows.append([band(top, bottom, unit), int(inside.sum()),
                             f"{np.median(velocity[inside]):.0f}", spread(velocity[inside])])
        section += (f"\n{sub} Velocity by Depth\n\nOver the ray-covered cells, by depth below "
                    "the ground surface.\n\n"
                    + table(["Depth", "Cells", "Median (m/s)", "10th-90th percentile (m/s)"],
                            rows, align=["left", "right", "right", "right"]))
    lines = _interfaces(results)
    if lines:
        rows = [[f"{line['threshold']:.0f} m/s", length(np.ptp(line["x"]), unit),
                 (f"{float(np.min(line['depth'])):.3g} - {length(np.max(line['depth']), unit)}"
                  if line["depth"] is not None else None),
                 length(np.median(line["depth"]), unit) if line["depth"] is not None else None]
                for line in lines]
        section += (f"\n{sub} Velocity Interfaces\n\n"
                    + table(["Contour", "Traced along", "Depth below ground", "Median depth"],
                            rows, align=["left", "right", "right", "right"]))
    return section


def water_content_section(results: Mapping[str, Any], steps: Optional[Sequence[Any]] = None,
                          config: Optional[Mapping[str, Any]] = None, unit: str = "m",
                          output_dir: Optional[str] = None, level: str = "##") -> str:
    """Seismic velocity is not converted to water content by these workflows."""
    return ""


def figures(results: Mapping[str, Any]) -> List[Tuple[str, str]]:
    drawn = []
    gathers = results.get("gathers_figure")
    if gathers and os.path.exists(str(gathers)):
        drawn.append((str(gathers), "Shot gathers spread along the line, as wiggle traces at "
                                    "their receiver positions with automatic gain control, "
                                    "and the first-arrival picks on them: blue dots kept, "
                                    "orange rings picked again - along the shot's curve, or "
                                    "where the neighbouring shots' picks at the same geophone "
                                    "put them - red crosses left out"
                      + ("; the shots left out for times that disagree with their "
                         "reciprocals are included, with grey picks" if results.get("dropped_shots")
                         else "") + "."))
    picks = results.get("picks_figure")
    if picks and os.path.exists(str(picks)):
        drawn.append((str(picks), "First-arrival picks, travel time against receiver position, "
                                  "one curve per shot; triangles mark the shot positions"
                      + (", and dashed grey curves the shots left out for times that "
                         "disagree with their reciprocals" if results.get("dropped_shots")
                         else "") + "."))
    path = results.get("structure_figure") or results.get("visualization_file")
    if path and os.path.exists(str(path)):
        drawn.append((str(path), "P-wave velocity model from travel-time tomography, shaded "
                                 "where no ray passes, with the traced velocity contours."))
    return drawn


def interpretation(results: Mapping[str, Any]) -> Optional[str]:
    text = results.get("interpretation")
    return str(text).strip() if isinstance(text, str) and text.strip() else None


def recommendations(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    items = []
    chi2 = results.get("chi2")
    if chi2 is not None and float(chi2) > POOR_FIT_CHI2:
        items.append(f"**Review the first-break picks.** The model fits them at chi-squared "
                     f"{float(chi2):.2f}; outlying picks and mis-assigned shot positions are the "
                     "usual cause.")
    if results.get("raw_seismic_file"):
        items.append("**Check the automatic picks** against the traces before relying on "
                     "the model; the picker's threshold was not tuned to this survey.")
    items.append("**Tie the velocity interfaces to a borehole or test pit,** which fixes "
                 "which contour corresponds to the boundary of interest.")
    return items


FILE_CONTENTS = {
    "velocity_model.npy": "velocity of each mesh cell (m/s)",
    "coverage.npy": "ray coverage of each cell",
    "seismic_mesh.bms": "the inversion mesh (pyGIMLi)",
    "first_break_picks.csv": "the automatic first-break picks",
    "seismic_traveltime_from_segy.dat": "the travel-time file built from the picks",
}


def output_files(results: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    paths = [str(path) for path in results.get("data_paths") or [] if path]
    paths += [str(results[key]) for key in ("first_break_picks_file", "traveltime_file")
              if results.get(key)]
    return [(path, FILE_CONTENTS.get(os.path.basename(path))
             or ("a traced velocity contour (x, z)" if os.path.basename(path).startswith("interface_")
                 else None))
            for path in dict.fromkeys(paths) if os.path.exists(path)]
