"""The magnetotelluric part of a workflow report.

Written from what the runtime's ``invert_mt`` step returns: one Occam 1D model
per site with its static shift, a 2D TE/TM section when three or more located
sites form a line, and water content per site when it was converted.
:mod:`PyHydroGeophysX.agents._survey_report` lays the sections out. Every number
is computed from the results; no sentence is written by a model.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._document import bullets, facts, table
from ._report_text import band, given, length, number, plural, source_name, spread
from ._uncertainty import water_content_reliability

KEY = "mt"
ARTIFACT = "mt_results"
TITLE = "Magnetotelluric Survey"
FIGURE_TAG = "M"

#: Depth band edges (m) the site models are summarised over.
DEPTH_BANDS_M = (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)

#: RMS above which a site is said to be fitted poorly.
POOR_FIT_RMS = 2.0


def _sites(results: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(site) for site in results.get("sites") or []]


def _layers(site: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top, bottom (m) and resistivity of each layer; the last is the half-space."""
    rho = np.asarray(site.get("resistivity_ohm_m", []), dtype=float).ravel()
    top = np.asarray(site.get("depth_top_m", []), dtype=float).ravel()
    if top.size != rho.size:
        thickness = np.asarray(site.get("thicknesses", []), dtype=float).ravel()
        top = np.concatenate([[0.0], np.cumsum(thickness)])[:rho.size]
    bottom = np.concatenate([top[1:], [np.inf]]) if top.size else top
    return top, bottom, rho


def sensitivity_depth(site: Mapping[str, Any]) -> Optional[float]:
    """A rough depth of sensitivity: the skin depth at the longest period.

    Taken over the geometric mean of the model's resistivity above it, so it is
    a guide to how deep the model is worth reading, not a resolution estimate.
    """
    periods = site.get("periods_s") or []
    _, _, rho = _layers(site)
    rho = rho[np.isfinite(rho) & (rho > 0)]
    if len(periods) < 2 or not rho.size:
        return None
    return float(503.0 * np.sqrt(float(10 ** np.mean(np.log10(rho))) * float(max(periods))))


def _band_values(results: Mapping[str, Any], key: str = "resistivity_ohm_m"
                 ) -> List[Dict[str, Any]]:
    """Each site's model over :data:`DEPTH_BANDS_M`, above its depth of sensitivity."""
    bands = []
    edges = list(DEPTH_BANDS_M) + [np.inf]
    for top, bottom in zip(edges[:-1], edges[1:]):
        values, sites = [], 0
        for site in _sites(results):
            layer_top, layer_bottom, rho = _layers(site)
            data = (rho if key == "resistivity_ohm_m"
                    else np.asarray(site.get(key, []), dtype=float).ravel())
            if data.size != rho.size:
                continue
            limit = sensitivity_depth(site) or np.inf
            centre = np.where(np.isfinite(layer_bottom), 0.5 * (layer_top + layer_bottom),
                              layer_top)
            inside = (centre >= top) & (centre < bottom) & (centre <= limit)
            if inside.any():
                values.extend(data[inside][np.isfinite(data[inside])].tolist())
                sites += 1
        if values:
            bands.append({"top": top, "bottom": bottom, "values": np.asarray(values),
                          "sites": sites})
    return bands


def describe(results: Mapping[str, Any]) -> Dict[str, Any]:
    sites = _sites(results)
    names = [source_name(site.get("path")) for site in sites]
    return {
        "data": names[0] if len(names) == 1 else f"{len(names)} transfer-function files",
        "instrument": None,
        "survey": plural(len(sites), "MT site"),
        "scope": (f"Magnetotelluric transfer functions at {plural(len(sites), 'site')} were "
                  "inverted for the electrical resistivity of the ground beneath each site"
                  + (", and along the line of sites as a 2D section." if results.get("profile")
                     else ".")),
    }


def findings(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
             unit: str = "m") -> List[str]:
    items = []
    sites = _sites(results)
    rms = np.asarray([site.get("rms", np.nan) for site in sites], dtype=float)
    if rms.size:
        poor = int(np.sum(rms > POOR_FIT_RMS))
        items.append(f"{plural(len(sites), 'site')} {'was' if len(sites) == 1 else 'were'} "
                     "inverted in 1D by Occam's method; "
                     + ("the models fit their data" if len(sites) > 1 else "its model fits its data")
                     + f" to RMS {np.nanmin(rms):.2f}"
                     + (f" to {np.nanmax(rms):.2f} (median {np.nanmedian(rms):.2f})"
                        if rms.size > 1 else "")
                     + (f"; {plural(poor, 'site')} fit worse than RMS {POOR_FIT_RMS:g}." if poor
                        else "."))
    periods = [p for site in sites for p in (site.get("periods_s") or [])]
    if periods:
        items.append(f"The data span periods of {number(min(periods), 3)} to "
                     f"{number(max(periods), 3)} s.")
    bands = [b for b in _band_values(results) if b["sites"] >= max(1, len(sites) / 2)]
    if len(bands) >= 2:
        medians = [float(np.median(b["values"])) for b in bands]
        low, high = bands[int(np.argmin(medians))], bands[int(np.argmax(medians))]
        items.append(f"The most conductive depths are {band(low['top'], low['bottom'], unit)} "
                     f"(median {min(medians):.3g} ohm-m) and the most resistive "
                     f"{band(high['top'], high['bottom'], unit)} (median {max(medians):.3g} "
                     "ohm-m), within each site's depth of sensitivity.")
    profile = results.get("profile") or {}
    if profile.get("rms") is not None:
        items.append(f"The 2D TE/TM section along the sites fits to RMS {float(profile['rms']):.2f}.")
    with_water = [site for site in sites if site.get("water_content_mean") is not None]
    if with_water:
        mean = np.concatenate([np.asarray(s["water_content_mean"], float).ravel() for s in with_water])
        std = np.concatenate([np.asarray(s["water_content_std"], float).ravel() for s in with_water])
        reliability = water_content_reliability(
            {**results, "water_content_mean": mean, "water_content_std": std}, config or {})
        items.append(f"Converted to water content, the site models give {np.nanmin(mean):.3f} to "
                     f"{np.nanmax(mean):.3f} (mean uncertainty ±{np.nanmean(std):.3f})."
                     + (f" {reliability['sentence']}" if reliability else ""))
    return items


def confidence(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    parts = ["Each 1D model assumes the ground is layered beneath its site; where the data "
             "are two- or three-dimensional - a strong skew or a split between the xy and yx "
             "curves - the 1D model is an approximation.",
             "Occam's method returns the smoothest model that fits, so boundaries appear as "
             "gradients, and resolution falls with depth."]
    shifts = [value for site in _sites(results) for value in (site.get("static_shift") or {}).values()]
    if shifts and any(abs(np.log10(float(v))) > 0.1 for v in shifts if v):
        parts.append("Static shifts were estimated as part of each inversion; they trade "
                     "against the resistivity of the shallow layers.")
    return " ".join(parts)


def limitations(results: Mapping[str, Any]) -> List[str]:
    return [
        "**Dimensionality.** The 1D models ignore lateral structure; the phase tensor and "
        "skew of each site tell whether that is justified.",
        "**Depth of sensitivity.** Deeper than about one skin depth at the longest period the "
        "model is unconstrained; the tables stop there.",
        "**Static shift.** A galvanic distortion shifts apparent resistivity by a constant "
        "factor; it was estimated with the model, and it scales the shallow resistivity.",
    ]


def method_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    sites = _sites(results)
    periods = [p for site in sites for p in (site.get("periods_s") or [])]
    located = sum(1 for s in sites if np.isfinite([s.get("latitude", np.nan) or np.nan,
                                                   s.get("longitude", np.nan) or np.nan]).all())
    data = given([
        ("Sites", len(sites)),
        ("Files", ", ".join(filter(None, (source_name(s.get("path")) for s in sites[:6])))
         + (f" and {len(sites) - 6} more" if len(sites) > 6 else "")),
        ("Periods", f"{number(min(periods), 3)} to {number(max(periods), 3)} s"
         if periods else None),
        ("Located sites", f"{located} of {len(sites)}"),
    ])
    profile = results.get("profile") or {}
    inversion = given([
        ("1D scheme", "Occam's inversion (Constable et al., 1987), the static shift solved "
                      "as a parameter"),
        ("2D section", f"TE/TM inversion on SimPEG (Cockett et al., 2015), "
                       f"{profile.get('iterations')} iterations" if profile else None),
    ])
    return (f"{level} Data and Method\n\n{sub} Data\n\n" + facts(data, "Item", "Value")
            + f"\n{sub} Inversion\n\n" + facts(inversion, "Item", "Value")
            + f"\n{sub} Assumptions and Limits\n\n" + bullets(limitations(results)))


def results_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    rows = []
    for site in _sites(results):
        periods = site.get("periods_s") or []
        shifts = site.get("static_shift") or {}
        depth = sensitivity_depth(site)
        rows.append([
            site.get("station"),
            f"{number(periods[0], 3)} - {number(periods[1], 3)}" if len(periods) == 2 else None,
            f"{float(site.get('rms', np.nan)):.2f}", site.get("iterations"),
            ", ".join(f"{mode} {float(v):.2f}" for mode, v in shifts.items()) or None,
            spread(_layers(site)[2]),
            length(depth, unit) if depth else None,
        ])
    section = (f"{level} Results\n\n{sub} Sites\n\n"
               + table(["Site", "Periods (s)", "RMS", "Iterations", "Static shift",
                        "Resistivity 10th-90th (ohm-m)", "Approx. depth of sensitivity"],
                       rows, align=["left", "right", "right", "right", "left", "right", "right"]))
    bands = _band_values(results)
    if bands:
        section += (f"\n{sub} Resistivity by Depth\n\nOver every site's model, above each "
                    "site's approximate depth of sensitivity.\n\n"
                    + table(["Depth", "Sites", "Median (ohm-m)", "10th-90th percentile"],
                            [[band(b["top"], b["bottom"], unit), b["sites"],
                              f"{np.median(b['values']):.3g}", spread(b["values"])]
                             for b in bands],
                            align=["left", "right", "right", "right"]))
    return section


def water_content_section(results: Mapping[str, Any], steps: Optional[Sequence[Any]] = None,
                          config: Optional[Mapping[str, Any]] = None, unit: str = "m",
                          output_dir: Optional[str] = None, level: str = "##") -> str:
    sites = [site for site in _sites(results) if site.get("water_content_mean") is not None]
    if not sites:
        return ""
    step = (list(steps or []) or [{}])[0] or {}
    section = f"{level} Water Content\n\n"
    if step.get("layering"):
        section += ("Each site's layered model was converted as one unit with one "
                    "petrophysical parameter set; no interface divides its layers into "
                    "geological units.\n\n")
    statement = results.get("prior_statement") or step.get("prior_statement")
    if statement:
        relationship = results.get("petrophysical_relationship") or step.get(
            "petrophysical_relationship")
        section += ("" if relationship == "user" else "**Warning:** ") + f"{statement}\n\n"
    rows = []
    for site in sites:
        mean = np.asarray(site["water_content_mean"], dtype=float).ravel()
        std = np.asarray(site.get("water_content_std"), dtype=float).ravel()
        rows.append([site.get("station"), f"{np.nanmean(mean):.3f}",
                     f"{np.nanmin(mean):.3f} - {np.nanmax(mean):.3f}", f"±{np.nanmean(std):.3f}"])
    section += table(["Site", "Mean", "Range over the layers", "Mean uncertainty"], rows,
                     align=["left", "right", "right", "right"])
    return section + ("\nWater content is derived from the resistivity through a "
                      "petrophysical relationship, so it carries the uncertainty of that "
                      "relationship as well as that of the inversion.\n")


def figures(results: Mapping[str, Any]) -> List[Tuple[str, str]]:
    out = []
    for site in _sites(results):
        path = site.get("figure")
        if path and os.path.exists(str(path)):
            out.append((str(path), f"Site {site.get('station')}: apparent resistivity and "
                                   "phase with the Occam model's fit (left), and the 1D "
                                   "model (right)."))
    profile = (results.get("profile") or {}).get("figure")
    if profile and os.path.exists(str(profile)):
        out.append((str(profile), "2D resistivity section from the TE/TM inversion along "
                                  "the line of sites."))
    return out


def interpretation(results: Mapping[str, Any]) -> Optional[str]:
    return None


def recommendations(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    items = []
    poor = [site.get("station") for site in _sites(results)
            if float(site.get("rms", 0) or 0) > POOR_FIT_RMS]
    if poor:
        items.append(f"**Examine {', '.join(map(str, poor[:6]))}** (RMS above "
                     f"{POOR_FIT_RMS:g}) for noise or 2D/3D effects before reading their "
                     "models.")
    if len(_sites(results)) >= 3 and not results.get("profile"):
        items.append("**Invert the sites together as a 2D section** once their positions "
                     "along a line are known; the 1D models cannot place a lateral boundary.")
    if any(site.get("water_content_mean") is not None for site in _sites(results)) and \
            results.get("petrophysical_relationship") != "user":
        items.append("**Calibrate the petrophysical relationship** for the depths of "
                     "interest; the generated parameters dominate the water-content "
                     "uncertainty.")
    items.append("**Fix the static shift with a TEM sounding** at one or more sites, which "
                 "measures the shallow resistivity the shift trades against.")
    return items


def output_files(results: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    rows = []
    for site in _sites(results):
        station = site.get("station")
        rows += [(site.get("model_csv"), f"site {station}: the 1D model"),
                 (site.get("fit_csv"), f"site {station}: observed and predicted data"),
                 (site.get("water_content_table"), f"site {station}: water content by layer")]
    profile = results.get("profile") or {}
    rows.append((profile.get("section_npz"), "the 2D section"))
    return [(str(path), text) for path, text in rows if path and os.path.exists(str(path))]
