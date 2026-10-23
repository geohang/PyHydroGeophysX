"""The ground-penetrating radar part of a workflow report.

Written from what the run's ``process_gpr_line`` step returns: the line as
read, the processing applied to it, and the velocity its depths use.
:mod:`PyHydroGeophysX.agents._survey_report` lays the sections out. Every
number is computed from the results; no sentence is written by a model.

A depth on a radargram is ``v t / 2``, so the velocity is stated wherever a
depth is, with where it came from; a line processed without one is reported
in two-way time, and the report says that no depth is given and why.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ._document import bullets, facts
from ._report_text import given, length, source_name

KEY = "gpr"
ARTIFACT = "gpr_results"
TITLE = "Ground-Penetrating Radar"
FIGURE_TAG = "R"
#: The report's name when the line is the run's only method: nothing is inverted.
REPORT = "Processing Report"
#: How an ERT report says the line stands beside its model.
ALONE = "Its radargram is reported on its own and is not combined with the ERT model."


def _velocity(results: Mapping[str, Any]) -> Optional[float]:
    value = results.get("velocity_m_per_ns")
    return float(value) if value else None


def _line(results: Mapping[str, Any], unit: str) -> str:
    parts = [f"{results.get('n_traces')} traces"]
    if results.get("line_length_m"):
        parts.append(f"over {length(results['line_length_m'], unit)}")
    if results.get("frequency_mhz"):
        parts.append(f"{results['frequency_mhz']:.0f} MHz antenna")
    return ", ".join(parts)


def _depth_sentence(results: Mapping[str, Any], unit: str = "m") -> str:
    velocity = _velocity(results)
    end = float(results.get("time_window_ns") or 0.0)
    if velocity is None:
        return (f"No radar velocity was given, so the line is reported in two-way time (to "
                f"{end:.0f} ns) and this report gives no depth: a depth would rest on a velocity "
                "nobody supplied.")
    return (f"Depths use a radar velocity of {velocity:.3g} m/ns "
            f"({results.get('velocity_source')}), a relative permittivity of "
            f"{float(results.get('permittivity') or 0):.3g}; at that velocity the {end:.0f} ns "
            f"record reaches about {length(results.get('max_depth_m') or 0.0, unit)}.")


def describe(results: Mapping[str, Any]) -> Dict[str, Any]:
    simulated = bool(results.get("simulated"))
    return {
        "data": ("simulated example line" if simulated
                 else source_name(results.get("source_file"))),
        "instrument": results.get("instrument") or None,
        "survey": "1 line, " + _line(results, "m"),
        "title": TITLE,
        "scope": ("A ground-penetrating radar line "
                  + ("simulated over the hydrological model " if simulated else "")
                  + "was processed into a radargram"
                  + (" and its two-way time converted to depth with a stated velocity."
                     if _velocity(results) else
                     "; no velocity was given, so it is reported in two-way time.")),
    }


def findings(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
             unit: str = "m") -> List[str]:
    items = []
    if results.get("simulated"):
        items.append(f"The line is simulated, not measured: drawn over the {results.get('source')}, "
                     "its permittivity from the model's water content and porosity (CRIM).")
    items.append(f"The line holds {_line(results, unit)}; after time zero the record runs to "
                 f"{float(results.get('time_window_ns') or 0):.0f} ns.")
    applied = [line.split(":")[0].split(" (")[0] for line in results.get("applied") or []]
    if applied:
        items.append("Processing applied, in order: "
                     + ", ".join(a[:1].lower() + a[1:] for a in applied) + ".")
    items.append(_depth_sentence(results, unit))
    return items


def confidence(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    parts = ["A radargram shows where the permittivity changes - with water content, clay or "
             "the material - not what the ground is; naming the reflectors needs a borehole, a "
             "pit or another method."]
    if _velocity(results) is None:
        parts.append("With no velocity, the section is in two-way time only.")
    else:
        parts.append("Every depth scales with the velocity: a velocity 10% off puts every depth "
                     "10% off, and one velocity is used for the whole line.")
        if "header" in str(results.get("velocity_source") or ""):
            parts.append("The velocity is the one set on the instrument, not one measured here.")
    if any("gain" in str(line).lower() for line in results.get("applied") or []):
        parts.append("The gain applied for display makes amplitudes along a trace no longer "
                     "comparable.")
    return " ".join(parts)


def limitations(results: Mapping[str, Any]) -> List[str]:
    items = ["**One velocity.** Time is converted to depth with a single velocity for the "
             "whole line; where the ground is wetter, depths are overstated, and where it is "
             "drier, understated."
             if _velocity(results) else
             "**No depth.** No radar velocity was given, so nothing in this report is a depth."]
    if not any(step.get("name") == "migrate" for step in results.get("steps") or []):
        items.append("**Not migrated.** Diffractions still appear as hyperbolas and dipping "
                     "reflectors are drawn down-dip of where they are.")
    if not results.get("frequency_mhz"):
        items.append("**Antenna frequency unknown.** The dewow window is the 30 ns default "
                     "rather than three periods of the antenna.")
    items += [f"**Not done.** {text}" for text in results.get("not_applied") or []]
    items += [f"**Reader.** {text}" for text in results.get("warnings") or []]
    return items


def method_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    simulated = bool(results.get("simulated"))
    data = given([
        ("Data", "simulated from " + str(results.get("source")) if simulated
         else source_name(results.get("source_file"))),
        ("Instrument / format", results.get("instrument")),
        ("Traces", results.get("n_traces")),
        ("Samples per trace", results.get("n_samples")),
        ("Sample interval", f"{float(results['dt_ns']):.3g} ns" if results.get("dt_ns") else None),
        ("Time window as recorded", f"{float(results['raw_window_ns']):.0f} ns"
         if results.get("raw_window_ns") else None),
        ("Line length", length(results["line_length_m"], unit)
         if results.get("line_length_m") else None),
        ("Trace spacing", length(results["trace_spacing_m"], unit)
         if results.get("trace_spacing_m") else None),
        ("Antenna frequency", f"{results['frequency_mhz']:.0f} MHz"
         if results.get("frequency_mhz") else "not known"),
        ("Antenna separation", length(results["antenna_separation_m"], unit)
         if results.get("antenna_separation_m") else None),
        ("Ground elevation", f"{length(results['elevation_range'][0], unit)} to "
                             f"{length(results['elevation_range'][1], unit)}"
         if results.get("elevation_range") else None),
    ])
    steps = [(f"{n}", line) for n, line in enumerate(results.get("applied") or [], 1)]
    velocity = _velocity(results)
    depth = given([
        ("Velocity", f"{velocity:.3g} m/ns" if velocity else "not given"),
        ("Where it came from", results.get("velocity_source") if velocity else None),
        ("Relative permittivity", f"{float(results['permittivity']):.3g}"
         if results.get("permittivity") else None),
        ("Depth reached by the record", length(results["max_depth_m"], unit)
         if results.get("max_depth_m") else "not given: no velocity"),
    ])
    return (f"{level} Data and Method\n\n{sub} Data\n\n" + facts(data, "Item", "Value")
            + f"\n{sub} Processing\n\nThe steps use the standard processing sequence also used "
              "by GPRPy, readgssi, ImpDAR and RGPR, as written in "
              "`PyHydroGeophysX.data_processing.gpr`:\n\n"
            + facts(steps, "Step", "What was done")
            + f"\n{sub} Velocity and Depth\n\n" + facts(depth, "Item", "Value")
            + "\n" + _depth_sentence(results, unit) + "\n"
            + f"\n{sub} Assumptions and Limits\n\n" + bullets(limitations(results)))


def results_section(results: Mapping[str, Any], unit: str = "m", level: str = "##") -> str:
    sub = level + "#"
    velocity = _velocity(results)
    rows = given([
        ("Record after time zero", f"{float(results.get('time_window_ns') or 0):.0f} ns"),
        ("Depth reached", length(results["max_depth_m"], unit) if velocity
         else "not given (no velocity)"),
        ("Section in elevation", "drawn" if velocity and results.get("elevation_range")
         else ("not drawn: no velocity" if results.get("elevation_range")
               else "not drawn: no ground elevations")),
    ])
    return (f"{level} Results\n\n{sub} Processed Radargram\n\n" + facts(rows, "Measure", "Value")
            + "\nThe reflectors are not picked or interpreted automatically; the radargram "
              "is given for a geophysicist to read.\n")


def water_content_section(results: Mapping[str, Any], steps: Optional[Sequence[Any]] = None,
                          config: Optional[Mapping[str, Any]] = None, unit: str = "m",
                          output_dir: Optional[str] = None, level: str = "##") -> str:
    """No water content is derived from the radar line in a run."""
    return ""


def figures(results: Mapping[str, Any]) -> List[Tuple[str, str]]:
    return [(str(path), str(caption)) for path, caption in results.get("figures") or []
            if path and os.path.exists(str(path))]


def interpretation(results: Mapping[str, Any]) -> Optional[str]:
    return None


def recommendations(results: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None
                    ) -> List[str]:
    items = []
    source = str(results.get("velocity_source") or "")
    if _velocity(results) is None:
        items.append("**Measure the radar velocity** - fit a diffraction hyperbola on the GPR "
                     "page (Fit a hyperbola), or use a common-midpoint survey or a reflector of "
                     "known depth - and run again with it stated, to put the line in depth.")
    elif "header" in source:
        items.append("**Check the velocity** on a diffraction hyperbola: the one used is the "
                     "permittivity set on the instrument, not a measurement of this ground.")
    if not any(step.get("name") == "migrate" for step in results.get("steps") or []):
        items.append("**Migrate the line** once the velocity is checked, to collapse the "
                     "diffraction hyperbolas and put dipping reflectors in place.")
    if any("spacing" in str(text) for text in results.get("warnings") or []):
        items.append("**Set the trace spacing** (or a distance-triggered recording), so "
                     "distances along the line are true.")
    return items


FILE_CONTENTS = {
    "_processed.npz": "the processed line (PyHydroGeophysX GPR .npz; data, time and positions)",
    "_processing.json": "the processing recipe, the velocity and where it came from",
    "_radargram.png": "the processed radargram",
    "_elevation.png": "the processed line hung from the ground surface",
}


def output_files(results: Mapping[str, Any]) -> List[Tuple[str, Optional[str]]]:
    paths = [str(path) for path in results.get("data_paths") or [] if path]
    out = []
    for path in dict.fromkeys(paths):
        contents = next((text for end, text in FILE_CONTENTS.items() if path.endswith(end)), None)
        if contents and os.path.exists(path):
            out.append((path, contents))
    return out
