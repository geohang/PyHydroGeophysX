"""One report for the methods a run inverted, and their sections in an ERT report.

A run whose models came from TDEM, seismic refraction, MT or gravity and
magnetics, without ERT, ended "success" with no report, because the report step
needed an ERT inversion; and an ERT report described the ERT survey alone. Each
method now writes its own sections from its step's results, in a module with
one interface - :mod:`._tdem_report`, :mod:`._seismic_report`,
:mod:`._mt_report`, :mod:`._gravmag_report`::

    KEY, ARTIFACT, TITLE, FIGURE_TAG
    describe(results)        -> {"data", "instrument", "survey", "scope"[, "title"]}
    findings(results, config, unit)      -> [sentence]
    confidence(results, config)          -> paragraph
    method_section(results, unit, level), results_section(results, unit, level)
    water_content_section(results, steps, config, unit, output_dir, level)
    figures(results)         -> [(path, caption)]
    interpretation(results)  -> text or None
    recommendations(results, config)     -> [item]
    output_files(results)    -> [(path, contents)]

and this module lays them out: a report of its own, titled for one method or
for several, or a section per method in an ERT report.

Two things every method's results may carry are laid out here, the same way
for each: ``raw_figures``, the data as the run read them (drawn by the load
step, :mod:`._raw_data`), which open the method's figures; and
``evaluation``, the inversion's quality assessment
(:mod:`._method_evaluation`), which adds an Inversion Quality section, its
fit figure, its recommendations, and a sentence to the confidence paragraph.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from . import _gravmag_report, _mt_report, _seismic_report, _tdem_report
from ._document import control_block, facts, numbered, renumber, table
from ._report_text import length_unit, relative

#: In the order a report presents them.
METHODS = (_seismic_report, _tdem_report, _mt_report, _gravmag_report)

#: The artifact each method's results are kept under in a run.
ARTIFACTS = tuple(module.ARTIFACT for module in METHODS)

#: The one recommendation every method makes, made once.
INDEPENDENT_DATA = ("**Test the models against independent data** - a borehole log, a well "
                    "or another method over the same ground - which settles what the "
                    "geophysics alone leaves open.")


def present(artifacts: Mapping[str, Any]) -> List[Tuple[Any, Dict[str, Any]]]:
    """``(module, results)`` for each method the run produced results for."""
    return [(module, dict(artifacts[module.ARTIFACT])) for module in METHODS
            if isinstance(artifacts.get(module.ARTIFACT), Mapping)
            and artifacts.get(module.ARTIFACT)]


def _title(module: Any, results: Mapping[str, Any]) -> str:
    return module.describe(results).get("title") or module.TITLE


#: How each component of a quality score is named in a report.
COMPONENTS = {"data_fit": "Data fit", "ray_coverage": "Ray coverage",
              "physical_plausibility": "Physical plausibility", "convergence": "Convergence",
              "resolution": "Soundings resolved", "static_shift": "Static shifts"}

#: The caption of each method's fit figure.
FIT_CAPTIONS = {
    "seismic": ("Fit of the travel times: the picks (dots) and the kept model's times (lines), "
                "one colour per shot, and the residuals against the typical pick error "
                "(dashed)."),
    "tdem": ("Fit of each sounding: chi-squared along the line, the accepted range shaded and "
             "poorly fitted soundings above the dashed line, with each sounding's depth of "
             "investigation below."),
    "tdem_sounding": ("The decay and the model's response, and the residuals normalized by "
                      "the data errors (dashed: one error)."),
    "mt": ("Fit of each site: RMS against Occam's target of one (solid) and the line above "
           "which a site is fitted poorly (dashed)."),
    "gravmag": ("Fit of the residual anomaly at the inverted stations: the data, the model's "
                "anomaly on the same scale, and the misfit in errors."),
}


def _evaluation(results: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluation = results.get("evaluation")
    return evaluation if isinstance(evaluation, Mapping) and evaluation.get("quality_score") \
        is not None else {}


def method_figures(module: Any, results: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """The method's figures: its data as read, its own, and the fit of its model."""
    drawn = [(str(path), str(caption)) for path, caption in results.get("raw_figures") or []
             if path and os.path.exists(str(path))]
    drawn += list(module.figures(results))
    # The water content with its standard deviation, which no conversion is
    # reported without.
    drawn += [(str(path), str(caption)) for path, caption in
              results.get("water_content_figures") or [] if path and os.path.exists(str(path))]
    evaluation = _evaluation(results)
    fit = evaluation.get("figure")
    if fit and os.path.exists(str(fit)):
        key = ("tdem_sounding" if evaluation.get("method") == "tdem" and not results.get("survey")
               else str(evaluation.get("method")))
        drawn.append((str(fit), FIT_CAPTIONS.get(key, "Fit of the model to its data.")))
    return drawn


def quality_section(results: Mapping[str, Any], level: str = "###") -> str:
    """The inversion's quality assessment, as a subsection of the method's results."""
    evaluation = _evaluation(results)
    if not evaluation:
        return ""
    score, threshold = float(evaluation["quality_score"]), float(evaluation.get("threshold", 70))
    rows = [("Quality score", f"{score:.0f} / 100 ("
             + ("meets" if evaluation.get("status") == "success" else "below")
             + f" the threshold of {threshold:.0f})")]
    rows += [(COMPONENTS.get(name, name.replace("_", " ").capitalize()), f"{value:.0f} / 100")
             for name, value in (evaluation.get("component_scores") or {}).items()]
    section = (f"{level} Inversion Quality\n\n" + facts(rows, "Measure", "Value")
               + "\n" + str(evaluation.get("summary") or "").strip() + "\n")
    history = evaluation.get("evaluation_history") or []
    if len(history) > 1:
        kept = evaluation.get("adopted_attempt")
        section += ("\nThe fit was off, so the inversion was run again with the "
                    "regularization changed; the run keeps the best-scoring model.\n\n"
                    + table(["Attempt", "Lambda", "Chi-squared", "Score", "Kept"],
                            [[row["attempt"],
                              f"{float(row['lam']):g}" if row.get("lam") is not None else "",
                              f"{float(row['chi2']):.3g}" if row.get("chi2") is not None else "",
                              f"{float(row['quality_score']):.0f}",
                              "yes" if row["attempt"] == kept else "no"] for row in history],
                            align=["right", "right", "right", "right", "left"]))
    return section + ("\nThe score is a heuristic: it weighs the data fit, "
                      + ", ".join(COMPONENTS.get(name, name).lower() for name in
                                  (evaluation.get("component_scores") or {}) if name != "data_fit")
                      + " as the ERT evaluation does, and a passing score does not make the "
                        "model unique.\n")


def quality_sentence(results: Mapping[str, Any]) -> str:
    """One sentence for the confidence paragraph, or ''."""
    evaluation = _evaluation(results)
    if not evaluation:
        return ""
    verdict = ("meets the threshold set for this workflow"
               if evaluation.get("status") == "success"
               else "falls below the threshold set for this workflow and needs review")
    return (f"The automated quality assessment scored the inversion "
            f"{float(evaluation['quality_score']):.0f} out of 100, which {verdict}.")


def quality_recommendations(results: Mapping[str, Any]) -> List[str]:
    """What the evaluation recommends, when the result fell short of a criterion."""
    evaluation = _evaluation(results)
    items = [str(item) for item in evaluation.get("recommendations") or []]
    return [] if evaluation.get("status") == "success" and len(items) <= 1 else items


def _figure_block(figures: Sequence[Tuple[str, str]], output_dir: Optional[str],
                  first: int = 1, tag: str = "") -> str:
    return "\n".join(f"![Figure {tag}{n}]({relative(path, output_dir)})\n\n"
                     f"**Figure {tag}{n}.** {caption}\n"
                     for n, (path, caption) in enumerate(figures, first))


def survey_report(artifacts: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
                  output_dir: Optional[str] = None, *,
                  water_steps: Optional[Sequence[Any]] = None,
                  water_failure: Optional[str] = None, not_delivered: str = "",
                  site: Optional[Mapping[str, Any]] = None,
                  notices: Tuple[str, str] = ("", "")) -> Dict[str, Any]:
    """The report of a run with no ERT model, from every method it inverted.

    ``notices`` is the standing notice for a report a language model wrote part
    of, and for one it did not. Returns the Markdown, the executive summary,
    the figures, and the file name to write it under: ``<method>_report`` for
    one method, ``survey_report`` for several.
    """
    config = config or {}
    methods = present(artifacts)
    if not methods:
        raise ValueError("No geophysical results to report.")
    unit = length_unit(config)
    single = len(methods) == 1
    level = "##" if single else "###"
    info = [(module, results, module.describe(results)) for module, results in methods]

    # Front matter.
    site = site or {}
    rows: List[Tuple[str, Any]] = [("Site", site.get("name")), ("Location", site.get("location"))]
    for module, results, described in info:
        if single:
            rows += [("Data", described.get("data")), ("Instrument", described.get("instrument")),
                     ("Survey", described.get("survey"))]
        else:
            name = _title(module, results)
            survey = ", ".join(filter(None, (described.get("survey"), described.get("data"))))
            rows.append((name, survey + (f" ({described['instrument']})"
                                         if described.get("instrument") else "")))
    rows += [("Report date", datetime.now().strftime("%Y-%m-%d %H:%M")),
             ("Prepared by", "PyHydroGeophysX multi-agent workflow"),
             ("Status", "Automated output - technical review required")]
    narratives = [(module, results, module.interpretation(results)) for module, results in methods]
    written = any(text for _, _, text in narratives)
    title = (f"{_title(*methods[0])}: Inversion Report" if single
             else "Geophysical Survey: Inversion Report")
    front = control_block(title, [row for row in rows if row[1] not in (None, "", "N/A")],
                          notice=notices[0] if written else notices[1])

    # Executive summary.
    request = str(config.get("user_request") or "").strip()
    summary = "## Executive Summary\n\n### Scope\n\n"
    if request:
        summary += f"This report responds to the request: “{request}”\n\n"
    summary += " ".join(described["scope"] for _, _, described in info) + "\n\n"
    found = []
    for module, results, _ in info:
        items = module.findings(results, config, unit)
        if single:
            found += [f"- {item}" for item in items]
        elif items:
            found.append(f"- **{_title(module, results)}.**")
            found += [f"  - {item}" for item in items]
    if water_failure:
        found.append(f"- Water content was requested but not produced: {water_failure}.")
    if found:
        summary += "### Principal Findings\n\n" + "\n".join(found) + "\n\n"
    confidence = [(" ".join(filter(None, (module.confidence(results, config),
                                          quality_sentence(results)))), module, results)
                  for module, results, _ in info]
    summary += "### Confidence in These Findings\n\n" + "\n\n".join(
        text if single else f"**{_title(module, results)}.** {text}"
        for text, module, results in confidence) + "\n"

    # One block of sections per method.
    body: List[str] = [summary, not_delivered]
    produced_water = False
    for module, results, _ in info:
        water = module.water_content_section(results, water_steps, config, unit, output_dir,
                                             level)
        produced_water = produced_water or bool(water)
        parts = [module.method_section(results, unit, level),
                 module.results_section(results, unit, level),
                 quality_section(results, level + "#"), water]
        body.append(("" if single else f"## {_title(module, results)}\n\n")
                    + "\n\n".join(part.strip() for part in parts if part and part.strip()))
    if water_failure and not produced_water:
        body.append("## Water Content\n\n**Water content was requested, but the conversion "
                    f"did not complete:** {water_failure}\n")

    figures = [figure for module, results in methods for figure in method_figures(module, results)]
    if figures:
        body.append("## Figures\n\n" + _figure_block(figures, output_dir))
    texts = [(module, results, text) for module, results, text in narratives if text]
    if texts:
        body.append("## Interpretation\n\n**AI-generated interpretation - verify before "
                    "citing.**\n\n" + "\n\n".join(
                        text if single else f"**{_title(module, results)}.** {text}"
                        for module, results, text in texts))
    items = [item for module, results in methods
             for item in quality_recommendations(results) + module.recommendations(results, config)]
    body.append("## Recommendations\n\n" + numbered(list(dict.fromkeys(items + [INDEPENDENT_DATA]))))
    files = [(path, text) for module, results in methods for path, text in module.output_files(results)]
    if files:
        body.append("## Appendix: Output Files\n\n"
                    + table(["File", "Contents"],
                            [[f"`{relative(path, output_dir)}`", text] for path, text in
                             dict.fromkeys(files)], align=["left", "left"]))
    markdown = (front.rstrip() + "\n\n"
                + renumber("\n\n".join(part.strip() for part in body if part and part.strip()))
                + "\n")
    return {
        "markdown": markdown,
        "summary": summary,
        "figures": {f"{module.KEY}_{n}": path for module, results in methods
                    for n, (path, _) in enumerate(method_figures(module, results), 1)},
        "filename": f"{methods[0][0].KEY}_report" if single else "survey_report",
        "model_written": written,
    }


def ert_sections(artifacts: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None,
                 output_dir: Optional[str] = None,
                 water_steps: Optional[Sequence[Any]] = None,
                 combined: Optional[Mapping[str, str]] = None) -> str:
    """A section per method that ran beside an ERT survey, for the ERT report.

    ``combined`` says, per method key, how its model entered the ERT result -
    the seismic interface that constrained the inversion, say. A method not
    named there was inverted on its own, and the section says so.
    """
    config = config or {}
    combined = combined or {}
    unit = length_unit(config)
    sections = []
    for module, results in present(artifacts):
        items = "\n".join(f"- {item}" for item in module.findings(results, config, unit))
        parts = [f"## {_title(module, results)}",
                 module.describe(results)["scope"] + " "
                 + (combined.get(module.KEY) or "Its model is reported on its own and is not "
                                                "combined with the ERT model."),
                 items,
                 module.method_section(results, unit, "###"),
                 module.results_section(results, unit, "###"),
                 quality_section(results, "####"),
                 module.water_content_section(results, water_steps, config, unit, output_dir,
                                              "###"),
                 _figure_block(method_figures(module, results), output_dir,
                               tag=module.FIGURE_TAG)]
        sections.append("\n\n".join(part.strip() for part in parts if part and part.strip()))
    return "\n\n".join(sections) + ("\n" if sections else "")
