"""The gravity / magnetics part of a workflow report.

Written from what :class:`~PyHydroGeophysX.agents.gravmag_agent.GravMagAgent`
returns: the station data and their regional-residual separation, and the 3D
density-contrast or susceptibility model inverted from the residual.
:mod:`PyHydroGeophysX.agents._survey_report` lays the sections out. Every number
is computed from the results; no sentence is written by a model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._document import bullets, facts, table
from ._report_text import given, length, number, source_name

KEY = "gravmag"
ARTIFACT = "gravmag_results"
TITLE = "Gravity and Magnetic Survey"
FIGURE_TAG = "G"

#: Chi-squared above which the model is said to explain the data poorly.
POOR_FIT_CHI2 = 2.0

#: How the polynomial regional trend is named.
TREND = {0: "none", 1: "linear (plane)", 2: "quadratic", 3: "cubic"}


def _name(results: Mapping[str, Any]) -> str:
    return "gravity" if results.get("kind") == "gravity" else "magnetic"


def title(results: Mapping[str, Any]) -> str:
    """The section heading for this survey's field."""
    return "Gravity Survey" if results.get("kind") == "gravity" else "Magnetic Survey"


def describe(results: Mapping[str, Any]) -> Dict[str, Any]:
    field = _name(results)
    inverted = results.get("inversion") is not None
    return {
        "data": source_name(results.get("source_file")),
        "instrument": None,
        "survey": f"{results.get('n_stations')} {field} stations",
        "title": title(results),
        "scope": (f"The {field} anomaly at {results.get('n_stations')} stations was separated "
                  "into a regional trend and a residual"
                  + (", and the residual inverted for a 3D "
                     + ("density-contrast" if field == "gravity" else "magnetic-susceptibility")
                     + " model of the ground beneath the survey." if inverted else ".")),
    }


def findings(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
             unit: str = "m") -> List[str]:
    items = []
    field_unit = results.get("unit", "")
    stats = results.get("stats") or {}
    observed, residual = stats.get("Observed") or {}, stats.get("Residual") or {}
    if observed:
        items.append(f"The observed {_name(results)} field ranges from {number(observed['min'])} "
                     f"to {number(observed['max'])} {field_unit} over the survey.")
    if residual and results.get("detrend"):
        items.append(f"After a {TREND.get(results['detrend'], 'polynomial')} regional trend is "
                     f"removed, the residual spans {number(residual['min'])} to "
                     f"{number(residual['max'])} {field_unit} (standard deviation "
                     f"{residual['std']:.3g}).")
    inversion = results.get("inversion")
    if inversion:
        quantity = "Density contrast" if results.get("kind") == "gravity" else "Susceptibility"
        low, high = inversion["model_range"]
        fit = ("explains the residual to within its errors" if inversion["chi2"] <= POOR_FIT_CHI2
               else "does not explain the residual to within its errors")
        items.append(f"The 3D model {fit} (chi-squared {inversion['chi2']:.3g} over "
                     f"{inversion['n_data']} stations). {quantity} ranges from {low:.3g} to "
                     f"{high:.3g} {inversion['label'].split('(')[-1].rstrip(')')}"
                     + ("; the model reaches its bound, so the contrast it needs is larger than "
                        "the inversion allowed." if inversion.get("at_bound") else "."))
        strongest = inversion["model_summary"]
        positive, negative = strongest["strongest_positive"], strongest["strongest_negative"]
        items.append(f"The strongest positive anomaly sits about "
                     f"{length(positive['depth'], unit)} deep and the strongest negative about "
                     f"{length(negative['depth'], unit)} deep; potential-field depths are "
                     "weakly constrained (see the limitations).")
    elif results.get("inversion_error"):
        items.append(f"The 3D inversion did not run: {results['inversion_error']}.")
    return items


def confidence(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    parts = ["A potential field does not determine the depth of its source: a smooth model "
             "places anomalies where the depth weighting puts them, and a deeper, stronger or "
             "shallower, weaker body fits equally well.",
             "The regional-residual separation is a choice; a different trend changes the "
             "residual and every model built on it."]
    if results.get("assumptions"):
        parts.append("The processing made assumptions the data do not settle; they are "
                     "listed with the run's warnings and in the method section.")
    return " ".join(parts)


def limitations(results: Mapping[str, Any]) -> List[str]:
    items = [
        "**Non-uniqueness.** The model is the smoothest one that fits, with sensitivity "
        "weighting to counter the decay with depth; depths are indicative.",
        "**Regional trend.** The polynomial trend removed before inversion is a choice, not a "
        "measurement.",
    ]
    inversion = results.get("inversion") or {}
    if inversion and inversion.get("n_data", 0) < int(results.get("n_stations") or 0):
        items.append(f"**Station subset.** {inversion['n_data']} of {results['n_stations']} "
                     "stations, chosen to cover the survey evenly, were inverted.")
    if results.get("kind") != "gravity":
        items.append("**Induced magnetization only.** The model assumes magnetization along "
                     "the inducing field; remanent magnetization is not represented.")
    items += [f"**Assumption.** {text}" for text in results.get("assumptions") or []]
    return items


def method_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    extent = results.get("extent") or {}
    span = (f"{length(np.ptp(extent['x']), unit)} by {length(np.ptp(extent['y']), unit)}"
            if extent.get("x") and extent.get("y") else None)
    data = given([
        ("Data", source_name(results.get("source_file"))),
        ("Field", f"{'gravity' if results.get('kind') == 'gravity' else 'total-field magnetic'} "
                  f"anomaly, {results.get('unit')}"),
        ("Columns read", ", ".join(map(str, results.get("columns") or [])) or None),
        ("Stations", results.get("n_stations")),
        ("Survey extent", span),
        ("Station elevation", results.get("elevation_source")),
    ])
    field = results.get("field")
    if field:
        data.append(("Inducing field", f"inclination {field['inclination']:g} deg, declination "
                                       f"{field['declination']:g} deg, {field['strength_nT']:g} nT"))
    settings = results.get("settings") or {}
    inversion = results.get("inversion") or {}
    beta = inversion.get("beta") or {}
    rows = given([
        ("Regional trend removed", TREND.get(int(results.get("detrend") or 0), "polynomial")),
        ("Gridding", "linear interpolation onto a 120 x 120 grid, for the maps"),
        ("3D inversion", "SimPEG potential-field integral simulation (Cockett et al., 2015), "
                         "smooth (Tikhonov) regularization with sensitivity weighting"
         if inversion else ("not run: " + str(results.get("inversion_error"))
                            if results.get("inversion_error") else "not run")),
        ("Mesh", f"{' x '.join(map(str, inversion['shape']))} cells of "
                 f"{length(inversion['cell_size'][0], unit)} x "
                 f"{length(inversion['cell_size'][1], unit)} x "
                 f"{length(inversion['cell_size'][2], unit)}, to "
                 f"{length(inversion['depth_extent'], unit)} depth" if inversion else None),
        ("Data errors", f"{100 * inversion['relative_error']:.3g}% of the value plus "
                        f"{inversion['noise_floor']:g} {results.get('unit')}" if inversion else None),
        ("Regularization weight", (f"chosen to reach chi-squared {settings.get('target_chi2', 1):g}"
                                   f" ({beta.get('status')})" if settings.get("auto_beta")
                                   else "fixed") if inversion else None),
        ("Model bounds", f"±{inversion['bound']:g} {inversion['label'].split('(')[-1].rstrip(')')}"
         if inversion else None),
    ])
    return (f"{level} Data and Method\n\n{sub} Data\n\n" + facts(data, "Item", "Value")
            + f"\n{sub} Processing and Inversion\n\n" + facts(rows, "Item", "Value")
            + f"\n{sub} Assumptions and Limits\n\n" + bullets(limitations(results)))


def results_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    field_unit = results.get("unit", "")
    stats = results.get("stats") or {}
    rows = [[name, number(s["min"]), number(s["max"]), number(s["mean"]), number(s["std"], 3)]
            for name, s in stats.items() if name != "Regional" or results.get("detrend")]
    section = (f"{level} Results\n\n{sub} Field Statistics\n\n"
               + table(["Field", f"Min ({field_unit})", f"Max ({field_unit})",
                        f"Mean ({field_unit})", f"Std ({field_unit})"],
                       rows, align=["left", "right", "right", "right", "right"]))
    inversion = results.get("inversion")
    if inversion:
        name, _, model_unit = inversion["label"].partition(" (")
        rows = [("Stations inverted", f"{inversion['n_data']} of {inversion['n_input']}"),
                ("Chi-squared", f"{inversion['chi2']:.3g}"),
                (f"{name.capitalize()} range ({model_unit.rstrip(')')})",
                 f"{inversion['model_range'][0]:.3g} to {inversion['model_range'][1]:.3g}"),
                ("Reaches the model bound", "yes" if inversion.get("at_bound") else "no")]
        for name, label in (("strongest_positive", "Strongest positive anomaly"),
                            ("strongest_negative", "Strongest negative anomaly")):
            spot = inversion["model_summary"][name]
            rows.append((label, f"{spot['value']:.3g} at easting {spot['x']:,.0f}, northing "
                                f"{spot['y']:,.0f}, about {length(spot['depth'], unit)} deep"))
        section += f"\n{sub} 3D Model\n\n" + facts(rows, "Measure", "Value")
    elif results.get("inversion_error"):
        section += (f"\n{sub} 3D Model\n\nThe inversion did not run: "
                    f"{results['inversion_error']}.\n")
    return section


def water_content_section(results: Mapping[str, Any], steps: Optional[Sequence[Any]] = None,
                          config: Optional[Mapping[str, Any]] = None, unit: str = "m",
                          output_dir: Optional[str] = None, level: str = "##") -> str:
    """Density and susceptibility are not converted to water content."""
    return ""


def figures(results: Mapping[str, Any]) -> List[Tuple[str, str]]:
    captions = {
        "gravmag_qc.png": f"Observed {_name(results)} field, the regional trend fitted to it "
                          "and the residual, with the stations.",
        "gravmag_model.png": "Depth slices of the 3D model and a section through its "
                             "strongest anomaly.",
    }
    return [(str(path), captions.get(os.path.basename(str(path)), ""))
            for path in results.get("figures") or [] if path and os.path.exists(str(path))]


def interpretation(results: Mapping[str, Any]) -> Optional[str]:
    return None


def recommendations(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    items = []
    inversion = results.get("inversion") or {}
    if inversion and inversion["chi2"] > POOR_FIT_CHI2:
        items.append(f"**Revisit the error model or the regional trend.** The model fits at "
                     f"chi-squared {inversion['chi2']:.3g}; a trend of another degree, or "
                     "errors matched to the survey's repeatability, changes what the model "
                     "has to explain.")
    if inversion.get("at_bound"):
        items.append("**Allow a larger contrast or a finer mesh** where the model reaches its "
                     "bound, or accept that the anomaly's source is more compact than this "
                     "mesh can represent.")
    if any("elevation" in text for text in results.get("assumptions") or []):
        items.append("**Supply station elevations** (a z column) so terrain is modelled.")
    if any("inducing magnetic field" in text for text in results.get("assumptions") or []):
        items.append("**Supply the IGRF field** for the survey site and date.")
    items.append("**Constrain the depth** of the main anomaly with a borehole, a seismic or "
                 "an electrical section; the potential field alone cannot.")
    return items


FILE_CONTENTS = {
    "observed_grid.csv": "the observed field on the map grid",
    "regional_grid.csv": "the regional trend on the map grid",
    "residual_grid.csv": "the residual field on the map grid",
    "model_grid.npz": "the 3D model and its cell edges",
    "model.vtr": "the 3D model for ParaView (VTK)",
}


def output_files(results: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    paths = [str(path) for path in results.get("data_paths") or [] if path]
    return [(path, FILE_CONTENTS.get(os.path.basename(path))) for path in dict.fromkeys(paths)
            if os.path.exists(path) and os.path.basename(path) in FILE_CONTENTS]
