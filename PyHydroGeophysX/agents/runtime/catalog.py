"""The workflow's capabilities, registered as tools.

Each entry wraps an existing agent rather than reimplementing it: the agents
are the tested part of this codebase and their behaviour must not change here.
What changes is who decides to call them, and in what order.

Registration order is dependency order. That is not decoration - it is the
fallback policy: without a model to ask, the controller takes the first tool
whose requirements are met, which reproduces the pipeline these tools replaced.
A run with no API key therefore behaves as it always did.

Each handler returns ``(summary, outputs)``. The summary is the sentence the
next step and the final report will read, so it states what was *found*, not
that the step finished - "5 surveys, 812 measurements each, 2017-11-05 to
2017-11-09" rather than "loading complete". This is the shared memory the
agents did not previously have: ``BaseAgent.context`` is per-agent, so nothing
an agent concluded ever reached another one.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._chi2 import chi2_history, chi2_summary
from .._intent import climate_blocker, wants_climate, wants_water_content
from .._geocode import coords_from_config, geocode_place
from .._method import IMPLEMENTED_SCHEME
from .context import RunContext
from .tools import Tool, register


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def agent_kwargs(ctx: RunContext) -> Dict[str, Any]:
    """The constructor arguments every agent takes."""
    return {"api_key": ctx.settings.get("api_key"),
            "model": ctx.settings.get("model"),
            "llm_provider": ctx.settings.get("llm_provider", "openai")}


def resolve_path(path: Any, project_dir: Any = ".") -> str:
    """A data file's real location, trying the ways a request gets it wrong.

    Requests name files as the user sees them, which is often a bare file name,
    a path relative to the project directory, or one carrying a duplicated
    ``examples/`` prefix. This was written out three times in the branch this
    package replaces, once per file role, with the copies already drifting.

    Parameters
    ----------
    path : str or Path
        The path as configured.
    project_dir : str or Path, optional
        Directory the request's files are relative to.

    Returns
    -------
    str
        The first candidate that exists, or the original path unchanged so the
        caller reports a missing file by the name the user used.

    Raises
    ------
    None

    Examples
    --------
    >>> resolve_path('absent.dat')
    'absent.dat'
    >>> resolve_path('')
    ''
    >>> resolve_path(None)
    ''
    """
    if not path:
        return str(path or "")
    candidate = Path(str(path))
    if candidate.exists():
        return str(candidate)
    tries: List[Path] = []
    if project_dir and str(project_dir) != ".":
        tries.append(Path(str(project_dir)) / candidate.name)
    if candidate.parts and candidate.parts[0] == "examples":
        tries.append(Path(*candidate.parts[1:]))
    for attempt in list(tries):
        if attempt.parts and attempt.parts[0] == "examples":
            tries.append(Path(*attempt.parts[1:]))
    for attempt in tries:
        if attempt.exists():
            return str(attempt)
    return str(candidate)


def survey_files(config: Dict[str, Any]) -> List[str]:
    """Every ERT survey the configuration names, in order, without duplicates."""
    listed = (config.get("time_lapse_files") or config.get("timelapse_files") or [])
    if listed:
        return [str(p) for p in listed]
    single = config.get("ert_file") or config.get("data_file")
    return [str(single)] if single else []


def _describe_models(results: Dict[str, Any]) -> str:
    """One sentence about what an inversion recovered."""
    history = chi2_history(results.get("chi2_values"))
    models = results.get("final_models")
    parts = []
    if models is not None and getattr(models, "ndim", 0) == 2:
        parts.append(f"{models.shape[1]} model(s) over {models.shape[0]} cells")
    if history:
        parts.append(f"final chi-squared {history[-1]:.3f} after {len(history)} iterations")
    return "; ".join(parts) if parts else "an inversion result"


#: The ERT format assumed when the configuration names none. There is no format
#: auto-detect anywhere in this package - the device is chosen, not guessed - so
#: this default has to be readable from outside: the studio panel that opens a
#: run's data for display must open it the same way the run does, and a second
#: copy of the string here and there would drift.
DEFAULT_INSTRUMENT = "E4D"

#: Where a run writes its inverted-model bundle, under the run's output folder.
#: Named here because two things have to agree on it: the inversion step that
#: writes it, and the studio page that opens it while the run is still going.
MODEL_BUNDLE_DIR = "ert_model"


def _configured(*keys: str):
    """A gate that opens when the configuration names any of ``keys``."""
    def gate(ctx: RunContext) -> bool:
        return any(ctx.config.get(key) for key in keys)
    return gate


def export_model_bundle(ctx: RunContext, results: Dict[str, Any]) -> str:
    """Write the recovered model as the inverted-model bundle, and say where.

    The bundle - ``mesh_res.bms``, ``resmodel.npy``, ``index_marker.npy`` and
    optionally ``all_coverage.npy`` - is this package's own interchange format
    for an inverted model; :mod:`PyHydroGeophysX.Geophy_modular.ERT_to_WC` reads
    exactly these names, and the studio's ERT-to-water-content page asks for a
    folder containing them.

    An automatic run used to leave nothing in that format: it held the mesh and
    the models in memory, converted them in the same process, and wrote only
    figures. So a run that had just inverted five surveys still left every other
    part of the studio with nothing to open, and the water-content page asking
    the user to go and find a model folder that did not exist.

    Parameters
    ----------
    ctx : RunContext
        The run, for its output directory.
    results : dict
        The inversion agent's result, carrying ``mesh`` and either
        ``time_lapse_models`` or ``resistivity_model``.

    Returns
    -------
    str
        The folder written, or "" when there was nothing to write or the mesh
        could not be saved. Exporting is a convenience: a run must not fail
        because a file could not be written beside the result it already has.
    """
    models = results.get("time_lapse_models")
    if models is None:
        single = results.get("resistivity_model")
        models = [single] if single is not None else []
    models = [np.asarray(m, dtype=float).ravel() for m in models
              if m is not None and len(np.asarray(m).ravel())]
    mesh = results.get("mesh")
    if not models or mesh is None:
        return ""
    widths = {m.size for m in models}
    if len(widths) != 1:
        # Models of different lengths are not one bundle; saying so beats
        # writing an array whose rows mean different things.
        return ""
    folder = Path(ctx.output_dir) / MODEL_BUNDLE_DIR
    try:
        folder.mkdir(parents=True, exist_ok=True)
        # (n_cells, n_time), which is what the reader expects; a single survey
        # is one column rather than a special case.
        np.save(folder / "resmodel.npy", np.column_stack(models))
        # Layer markers, and only if that is what they are. An inversion
        # parameter mesh carries one marker per cell, which identifies cells and
        # says nothing about geology: writing those out as `index_marker.npy`
        # told the water-content page it had 2025 layers of one cell each.
        # `resolve_layers` already decides this question for the petrophysics,
        # and asking it here keeps one definition of what a layer is. The file
        # is optional, so omitting it leaves the page offering to derive layers
        # from a structural interface - which is where they actually come from.
        from ..petrophysics_agent import resolve_layers

        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        resolved, unique, why = resolve_layers(markers, models[0].size)
        if unique.size > 1:
            np.save(folder / "index_marker.npy", resolved)
        else:
            ctx.note(f"The exported model carries no layer markers: {why}")
        coverage = results.get("all_coverage")
        if coverage is None:
            coverage = results.get("coverage")
        if coverage is not None:
            np.save(folder / "all_coverage.npy", np.asarray(coverage, dtype=float))
        mesh.save(str(folder / "mesh_res.bms"))
    except Exception as exc:  # noqa: BLE001 - an export is not the result
        ctx.note(f"The recovered model could not be exported as a reusable "
                 f"bundle ({exc}); the numbers in this run are unaffected.")
        return ""
    return str(folder)


# ---------------------------------------------------------------------------
# 1. loading
# ---------------------------------------------------------------------------
def _load_ert(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..ert_loader_agent import ERTLoaderAgent

    config = ctx.config
    paths = survey_files(config)
    if not paths:
        raise ValueError("The configuration names no ERT data file.")
    project_dir = config.get("project_dir", ".")
    electrode = resolve_path(config.get("electrode_file"), project_dir) or None
    loader = ERTLoaderAgent(**agent_kwargs(ctx))

    loaded, failed = [], []
    for path in paths:
        resolved = resolve_path(path, project_dir)
        result = loader.execute({
            "data_file": resolved,
            "instrument": config.get("instrument", DEFAULT_INSTRUMENT),
            "project_dir": project_dir,
            "electrode_file": electrode,
            "crs": config.get("crs", "local"),
        })
        if result.get("status") != "success":
            failed.append(f"{Path(resolved).name}: {result.get('error')}")
            continue
        loaded.append(result["ert_data"])

    if not loaded:
        raise ValueError("No ERT survey could be loaded. " + "; ".join(failed))
    for message in failed:
        ctx.note(f"An ERT file could not be loaded and was left out: {message}")

    from .._intent import wants_water_content as _wwc  # noqa: F401 - documented below
    summary = f"Loaded {len(loaded)} ERT survey(s) from {len(paths)} configured file(s)."
    if failed:
        summary += f" {len(failed)} could not be read."
    return summary, {"ert_data": loaded, "n_surveys": len(loaded)}


register(Tool(
    name="load_ert_surveys",
    description="Read the ERT survey files named in the configuration, applying "
                "electrode geometry and topography. Run this before any ERT "
                "inversion.",
    handler=_load_ert,
    produces=("ert_data",),
    agent="ERTLoaderAgent",
    label="Load ERT data",
    module="ert",
    # Without this gate the loader is offered on every run, including one that
    # carries only a seismic file, and fails with "names no ERT data file" -
    # a recorded failure for a step nobody asked for.
    when=_configured("time_lapse_files", "timelapse_files", "ert_file", "data_file"),
))


# ---------------------------------------------------------------------------
# 2. climate, when the request asked for it and a position is known
# ---------------------------------------------------------------------------
def _climate_wanted(ctx: RunContext) -> bool:
    return wants_climate(ctx.config)


def _fetch_climate(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..climate_data_agent import ClimateDataAgent

    config = ctx.config
    coords = coords_from_config(config)
    if coords is None:
        # The parser extracts a place NAME; turning it into a position is a
        # gazetteer's job. A model-supplied coordinate is confidently wrong
        # often enough to put the series in the wrong valley.
        place = config.get("site_location")
        located = geocode_place(place) if place else None
        if located is None:
            raise ValueError(climate_blocker(config)
                             or "No coordinates and no place name to look up.")
        coords = located["coords"]
        climate_config = dict(config.get("climate_config") or {})
        climate_config["coords"] = list(coords)
        config["climate_config"] = climate_config
        config.setdefault("site_info", {})["location"] = located["matched_name"]
        ctx.note(f"Located '{place}' as {located['matched_name']} at {coords} "
                 f"via {located['source']}; verify this is the right site.")

    agent = ClimateDataAgent(**agent_kwargs(ctx))
    result = agent.execute({
        "coords": list(coords),
        "climate_config": config.get("climate_config", {}),
        "start_date": (config.get("climate_config") or {}).get("start_date"),
        "end_date": (config.get("climate_config") or {}).get("end_date"),
        "output_dir": str(Path(ctx.output_dir) / "climate"),
    })
    if result.get("status") not in (None, "success"):
        raise ValueError(str(result.get("error") or "Climate retrieval failed."))
    return (f"Retrieved meteorological data for {coords}.",
            {"climate_data": result.get("climate_data") or result})


register(Tool(
    name="fetch_climate",
    description="Retrieve precipitation, temperature and potential "
                "evapotranspiration for the survey dates at the site's "
                "coordinates. Only useful when the request asks to relate the "
                "geophysics to weather.",
    handler=_fetch_climate,
    produces=("climate_data",),
    agent="ClimateDataAgent",
    label="Fetch climate data",
    module="geo_hydrology",
    when=_climate_wanted,
))


# ---------------------------------------------------------------------------
# 3. inversion
# ---------------------------------------------------------------------------
def _is_time_lapse(ctx: RunContext) -> bool:
    return len(ctx.get("ert_data") or []) >= 2


def _is_single(ctx: RunContext) -> bool:
    return len(ctx.get("ert_data") or []) == 1


def _invert_time_lapse(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..ert_inversion_agent import ERTInversionAgent

    config = ctx.config
    agent = ERTInversionAgent(**agent_kwargs(ctx))
    results = agent.execute({
        "time_lapse_data": ctx.get("ert_data"),
        # The acquisition time lives in the file name, and the loaded containers
        # do not carry it; without these the sequence degrades to a 1..n index.
        "source_files": survey_files(config),
        "inversion_mode": "time-lapse",
        "time_lapse_method": config.get("time_lapse_method", IMPLEMENTED_SCHEME),
        "temporal_regularization": config.get("temporal_regularization", 10.0),
        "baseline_index": 0,
        "inversion_params": config.get("inversion_params",
                                       {"lambda": 15.0, "max_iterations": 10,
                                        "method": "cgls"}),
        "output_dir": str(Path(ctx.output_dir) / "inversion"),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Time-lapse inversion failed."))
    bundle = export_model_bundle(ctx, results)
    return (f"Inverted all surveys together: {_describe_models(results)}.",
            {"inversion_results": results,
             "model_directory": bundle or None})


register(Tool(
    name="invert_time_lapse",
    description="Recover a resistivity model for every survey at once, with a "
                "temporal constraint linking consecutive surveys. Use when two "
                "or more surveys are loaded and the question is how the "
                "subsurface changed.",
    handler=_invert_time_lapse,
    requires=("ert_data",),
    produces=("inversion_results",),
    agent="ERTInversionAgent",
    label="Run time-lapse inversion",
    module="ert",
    when=_is_time_lapse,
))


def _invert_single(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..ert_inversion_agent import ERTInversionAgent

    config = ctx.config
    agent = ERTInversionAgent(**agent_kwargs(ctx))
    results = agent.execute({
        "ert_data": (ctx.get("ert_data") or [None])[0],
        "inversion_mode": "standard",
        "inversion_params": config.get("inversion_params", {}),
        "output_dir": str(Path(ctx.output_dir) / "inversion"),
        "project_dir": config.get("project_dir", "."),
        "instrument": config.get("instrument", DEFAULT_INSTRUMENT),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Inversion failed."))
    history = chi2_history(results.get("chi2_values") or results.get("all_chi2"))
    detail = f" Final chi-squared {history[-1]:.3f}." if history else ""
    bundle = export_model_bundle(ctx, results)
    return f"Recovered a resistivity model from one survey.{detail}", {
        "inversion_results": results, "model_directory": bundle or None}


register(Tool(
    name="invert_ert",
    description="Recover a resistivity model from a single ERT survey. Use when "
                "exactly one survey is loaded.",
    handler=_invert_single,
    requires=("ert_data",),
    produces=("inversion_results",),
    agent="ERTInversionAgent",
    label="Run ERT inversion",
    module="ert",
    when=_is_single,
))


# ---------------------------------------------------------------------------
# 4. evaluation
# ---------------------------------------------------------------------------
def _evaluate(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..inversion_evaluation_agent import InversionEvaluationAgent

    config = ctx.config
    surveys = ctx.get("ert_data") or []
    results = ctx.get("inversion_results")
    agent = InversionEvaluationAgent(**agent_kwargs(ctx))
    evaluation = agent.execute({
        "inversion_results": results,
        "ert_data": surveys[0] if surveys else None,
        "time_lapse_data": surveys if len(surveys) > 1 else None,
        "inversion_mode": "time-lapse" if len(surveys) > 1 else "standard",
        "inversion_params": config.get("inversion_params", {}),
        "auto_adjust": config.get("auto_adjust", True),
        "output_dir": str(Path(ctx.output_dir) / "inversion"),
        "max_attempts": config.get("max_attempts", 3),
        "quality_threshold": config.get("quality_threshold", 70),
        "progress_callback": ctx.settings.get("progress_callback"),
        "project_dir": config.get("project_dir", "."),
        "instrument": config.get("instrument", DEFAULT_INSTRUMENT),
    })

    # A retry that improved the model replaces it, which is the one place the
    # old pipeline could already react to a finding.
    improved = (evaluation.get("status") == "success"
                and evaluation.get("attempts", 1) > 1
                and evaluation.get("final_results"))
    if improved:
        results = evaluation["final_results"]
    # Carried on the results as well as returned: the audit's quality block and
    # the desktop runner's "needs review" warning both read it from there.
    stripped = {k: v for k, v in evaluation.items() if k != "final_results"}
    if isinstance(results, dict):
        results["evaluation_results"] = stripped

    score = evaluation.get("quality_score")
    verdict = evaluation.get("summary") or evaluation.get("status", "evaluated")
    summary = (f"Quality {float(score):.1f}/100 after "
               f"{evaluation.get('attempts', 1)} attempt(s): {verdict}"
               if score is not None else f"Evaluation returned: {verdict}")
    if evaluation.get("status") != "success":
        ctx.note(str(verdict))
    return summary, {"inversion_results": results, "evaluation_results": stripped}


register(Tool(
    name="evaluate_inversion",
    description="Judge how well the recovered model fits the data and whether it "
                "is physically plausible, retrying with adjusted regularization "
                "if the configuration allows. Run after any inversion.",
    handler=_evaluate,
    requires=("inversion_results",),
    produces=("evaluation_results",),
    agent="InversionEvaluationAgent",
    label="Evaluate inversion quality",
    module="ert",
))


# ---------------------------------------------------------------------------
# 5. petrophysics
# ---------------------------------------------------------------------------
def _water_content_wanted(ctx: RunContext) -> bool:
    return wants_water_content(ctx.config)


def _convert_water_content(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..petrophysics_agent import PetrophysicsAgent

    config = ctx.config
    results = ctx.get("inversion_results") or {}
    mesh = results.get("mesh")
    models = results.get("time_lapse_models") or []
    if not models:
        single = results.get("resistivity_model")
        models = [single] if single is not None else []
    if not models:
        raise ValueError("The inversion returned no resistivity model to convert.")

    markers = (np.array(mesh.cellMarkers()) if mesh is not None
               else np.zeros(len(models[0])))
    agent = PetrophysicsAgent(**agent_kwargs(ctx))
    per_step: List[Dict[str, Any]] = []
    for index, model in enumerate(models):
        step = agent.execute({
            "resistivity_model": model,
            "mesh": mesh,
            "cell_markers": markers,
            "petrophysical_params": config.get("petrophysical_params", {}),
            "n_realizations": config.get("n_realizations", 100),
            "geological_context": config.get("geological_context", "generic watershed"),
            "output_dir": str(Path(ctx.output_dir) / "petrophysics"
                              / f"timestep_{index + 1}"),
        })
        if step.get("status") != "success":
            # One failed step must not discard the ones that worked.
            ctx.note(f"Water content failed at survey {index + 1}: {step.get('error')}")
            break
        per_step.append(step)

    if not per_step:
        raise ValueError("No time step could be converted to water content.")

    results["time_lapse_water_content"] = per_step
    results["water_content_mean"] = per_step[0].get("water_content_mean")
    results["water_content_std"] = per_step[0].get("water_content_std")
    results["petrophysical_params"] = config.get("petrophysical_params", {})

    layering = per_step[0].get("layering") or ""
    summary = (f"Converted {len(per_step)} of {len(models)} model(s) to water "
               f"content by Monte Carlo petrophysics.")
    if layering:
        summary += f" {layering}"
    return summary, {"inversion_results": results, "water_content": per_step}


register(Tool(
    name="convert_water_content",
    description="Convert the recovered resistivity to volumetric water content "
                "for every survey, propagating petrophysical uncertainty by "
                "Monte Carlo. Required whenever the request asks about water "
                "content, moisture or saturation.",
    handler=_convert_water_content,
    # Evaluation, not just the inversion. `evaluate_inversion` may retry with
    # adjusted regularization and *replace* the model; a conversion that ran
    # first would report water content derived from a model the run then threw
    # away. The controller does re-order these - on one run it chose to convert
    # before evaluating - so the constraint has to live in the dependency, not
    # in the order the tools happen to be registered.
    requires=("inversion_results", "evaluation_results"),
    produces=("water_content",),
    agent="PetrophysicsAgent",
    label="Convert to water content",
    module="geo_hydrology",
    when=_water_content_wanted,
))


# ---------------------------------------------------------------------------
# 6. other methods
# ---------------------------------------------------------------------------
def _invert_seismic(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..seismic_agent import SeismicAgent

    config = ctx.config
    agent = SeismicAgent(**agent_kwargs(ctx))
    raw = config.get("raw_seismic_file")
    inputs = {
        "seismic_file": resolve_path(config.get("seismic_file"),
                                     config.get("project_dir", ".")) or None,
        # SEG-Y needs first-break picking before tomography, which the agent
        # does itself when given the raw file under its own key.
        "raw_seismic_file": resolve_path(raw, config.get("project_dir", ".")) or None,
        "geophone_file": config.get("geophone_file"),
        "topography_file": config.get("topography_file"),
        "velocity_threshold": config.get("velocity_threshold", 1200.0),
        "inversion_params": config.get("seismic_inversion_params", {}),
        "output_dir": str(Path(ctx.output_dir) / "seismic"),
        "align_origin": config.get("align_origin"),
    }
    results = agent.execute(inputs)
    mismatch = results.get("origin_mismatch") if results.get("status") != "success" else None
    if mismatch and not inputs["align_origin"]:
        # Not a guess this code is entitled to make: the coordinate file and
        # the SEG-Y headers describe the same line from two different origins,
        # and only somebody who knows the survey can say which one to report.
        # Asking beats both alternatives - failing throws away the picking work
        # already done, and choosing silently puts an unlabelled shift into
        # coordinates that will later be laid beside an ERT line.
        choice = ctx.ask(
            f"The coordinate file and the SEG-Y headers place this line "
            f"{abs(mismatch.get('shift', 0.0)):g} m apart along x, so the shots "
            f"fall outside the elevation profile. The survey geometry is the "
            f"same either way; which origin should the results carry?",
            [{"id": "profile",
              "label": "Use the coordinate file's origin",
              "detail": "Shift the SEG-Y shot and receiver positions to match "
                        "the station/topography file. Usually right when the "
                        "file holds surveyed distances along the line."},
             {"id": "segy",
              "label": "Use the SEG-Y header origin",
              "detail": "Shift the coordinate file to match the SEG-Y headers. "
                        "Right when the acquisition geometry is authoritative "
                        "and the file was written from a different datum."},
             {"id": "stop",
              "label": "Do not process the seismic data",
              "detail": "Leave seismic out of this run so the two files can be "
                        "checked. Everything else still runs."}],
            default="profile")
        if choice == "stop":
            raise ValueError(
                "Seismic processing stopped: the coordinate file and the SEG-Y "
                "headers disagree about the origin of the line, and the choice "
                "was left open. " + str(results.get("error") or ""))
        inputs["align_origin"] = choice
        ctx.note(f"Seismic geometry was reconciled onto the "
                 f"{'coordinate file' if choice == 'profile' else 'SEG-Y header'} "
                 f"origin, a shift of {abs(mismatch.get('shift', 0.0)):g} m, "
                 f"chosen during the run. The velocity model is unaffected; the "
                 f"x coordinates it is reported against are not.")
        results = agent.execute(inputs)
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Seismic inversion failed."))
    for note in results.get("geometry_warnings") or []:
        ctx.note(str(note))
    span = results.get("velocity_range")
    detail = f" Velocity {span[0]:.0f} to {span[1]:.0f} m/s." if span else ""
    return (f"Recovered a seismic velocity model from {results.get('n_data', '?')} "
            f"travel times.{detail}", {"seismic_results": results})


register(Tool(
    name="invert_seismic",
    description="Invert seismic refraction travel times for a velocity model, "
                "and pick the interface depths it implies.",
    handler=_invert_seismic,
    produces=("seismic_results",),
    agent="SeismicAgent",
    label="Run seismic refraction inversion",
    module="seismic",
    when=_configured("seismic_file", "raw_seismic_file"),
))


def _invert_tdem(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..tdem_agent import TDEMAgent

    config = ctx.config
    agent = TDEMAgent(**agent_kwargs(ctx))
    # TDEMAgent takes its settings flat, under 'data_file' - not an
    # 'inversion_params' dict and not 'tdem_file'. Passing the names this
    # workflow uses elsewhere silently gave it no data file at all.
    payload = {
        "mode": "inversion",
        "data_file": resolve_path(config.get("tdem_file") or config.get("em_file"),
                                  config.get("project_dir", ".")),
        "output_dir": str(Path(ctx.output_dir) / "tdem"),
    }
    payload.update({k: v for k, v in (config.get("tdem_params") or {}).items()})
    results = agent.execute(payload)
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "TDEM inversion failed."))
    span = results.get("resistivity_range")
    detail = f" Resistivity {span[0]:.1f} to {span[1]:.1f} ohm-m." if span else ""
    return (f"Recovered a {results.get('n_layers', '?')}-layer resistivity model "
            f"from the TDEM sounding.{detail}", {"tdem_results": results})


register(Tool(
    name="invert_tdem",
    description="Invert a time-domain electromagnetic sounding for a layered "
                "resistivity model.",
    handler=_invert_tdem,
    produces=("tdem_results",),
    agent="TDEMAgent",
    label="Run TDEM inversion",
    module="em",
    when=_configured("tdem_file", "em_file"),
))


def _load_model_output(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..model_output_agent import ModelOutputAgent

    config = ctx.config
    agent = ModelOutputAgent(**agent_kwargs(ctx))
    # ModelOutputAgent picks the reader from 'hydro_model' and takes the
    # directory under the key for that model. 'model_type' and
    # 'model_directory' were names this workflow invented; only the latter is
    # read at all, and only as a MODFLOW fallback.
    results = agent.execute({
        "hydro_model": config.get("hydro_model") or config.get("model_type") or "auto",
        "modflow_dir": config.get("modflow_dir") or config.get("model_directory"),
        "parflow_dir": config.get("parflow_dir"),
        "user_request": config.get("user_request", ""),
        "petrophysical_params": config.get("petrophysical_params", {}),
        "output_dir": str(Path(ctx.output_dir) / "model_output"),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Model output could not be read."))
    return "Read hydrological model outputs.", {"model_output": results}


register(Tool(
    name="load_model_output",
    description="Read water content or saturation from a hydrological model such "
                "as MODFLOW or ParFlow, for comparison with the geophysics.",
    handler=_load_model_output,
    produces=("model_output",),
    agent="ModelOutputAgent",
    label="Load hydrological model output",
    module="hydro_geophysics",
    when=_configured("model_directory", "modflow_dir", "parflow_dir",
                     "hydro_model"),
))


def interface_coords(seismic: Dict[str, Any], threshold: Any
                     ) -> Optional[Tuple[Any, Any]]:
    """The ``(x, z)`` arrays of one velocity interface, or None.

    The two agents disagree about shape, and neither is wrong on its own.
    ``SeismicAgent`` may extract several interfaces at once and returns
    ``{threshold: {'x': ..., 'z': ...}}``; ``StructureConstraintAgent`` builds a
    mesh from exactly one and unpacks ``interface_x, interface_z =
    interface_coords``. Handing the dict straight across failed with "not
    enough values to unpack (expected 2, got 1)" - a message that says nothing
    about interfaces - so the translation belongs here, named.

    Parameters
    ----------
    seismic : dict
        A ``SeismicAgent`` result.
    threshold : float
        The velocity threshold whose interface is wanted. Falls back to the
        only interface present when there is exactly one, since a run that
        extracted one interface meant that one.

    Returns
    -------
    tuple or None
        ``(x, z)``, or None when no interface matches.

    Raises
    ------
    None

    Examples
    --------
    >>> result = {'interfaces': {1200.0: {'x': [0, 1], 'z': [-1, -2]}}}
    >>> interface_coords(result, 1200.0)
    ([0, 1], [-1, -2])
    >>> interface_coords(result, 900.0)        # the only one there is
    ([0, 1], [-1, -2])
    >>> interface_coords({'interfaces': {}}, 1200.0) is None
    True
    """
    interfaces = (seismic or {}).get("interfaces") or {}
    if not isinstance(interfaces, dict) or not interfaces:
        return None
    chosen = interfaces.get(threshold)
    if chosen is None:
        # Keys can arrive as ints, floats or strings depending on how the
        # threshold was configured.
        for key, value in interfaces.items():
            try:
                if float(key) == float(threshold):
                    chosen = value
                    break
            except (TypeError, ValueError):
                continue
    if chosen is None and len(interfaces) == 1:
        chosen = next(iter(interfaces.values()))
    if isinstance(chosen, dict) and "x" in chosen and "z" in chosen:
        return chosen["x"], chosen["z"]
    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        return chosen[0], chosen[1]
    return None


def as_pygimli_data(survey: Any, output_dir: Any, use_source_error: bool = True):
    """A loaded survey as the pyGIMLi container the inversion agents use.

    ``ERTLoaderAgent`` returns a ``StandardERT`` - the package's own record,
    carrying electrodes, observations, instrument and CRS. Several agents
    instead expect ``pygimli``'s ``DataContainerERT``, which they index
    (``data['k']``, ``data['err']``) and query (``sensorCount()``). Passing the
    first where the second is expected fails with "'StandardERT' object is not
    subscriptable", which names neither side of the mismatch.

    The conversion is the one the inversion agents already perform: export to a
    pyGIMLi unified data file, then load it back.

    Parameters
    ----------
    survey : StandardERT
        A loaded survey. Anything that is already a pyGIMLi container is
        returned unchanged.
    output_dir : str or Path
        Where the intermediate file is written.
    use_source_error : bool
        Carry the instrument's own error estimates into the export.

    Returns
    -------
    DataContainerERT
        The survey in pyGIMLi's form.

    Raises
    ------
    ValueError
        If ``survey`` is None.
    """
    if survey is None:
        raise ValueError("No ERT survey to convert.")
    if hasattr(survey, "sensorCount"):
        return survey
    from pygimli.physics import ert as _ert

    from ...data_processing.ert_data_agent import export_for_inversion

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    exported = export_for_inversion(survey, outdir=str(target), fmt="pgimli",
                                    use_source_error=bool(use_source_error))
    return _ert.load(exported)


def _derive_structure(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..structure_constraint_agent import StructureConstraintAgent

    agent = StructureConstraintAgent(**agent_kwargs(ctx))
    seismic = ctx.get("seismic_results") or {}
    surveys = ctx.get("ert_data") or []
    threshold = ctx.config.get("velocity_threshold", 1200.0)
    coords = interface_coords(seismic, threshold)
    if coords is None:
        raise ValueError(
            f"The seismic result carries no interface at {threshold} m/s. "
            f"Available: {sorted((seismic.get('interfaces') or {}))}.")
    output = Path(ctx.output_dir) / "structure"
    results = agent.execute({
        "ert_data": as_pygimli_data(surveys[0] if surveys else None, output),
        "interface_coords": coords,
        "velocity_threshold": threshold,
        "inversion_params": ctx.config.get("inversion_params", {}),
        "output_dir": str(output),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Structure extraction failed."))
    return "Derived layer interfaces from the velocity model.", {
        "structure_results": results}


register(Tool(
    name="derive_structure",
    description="Turn a seismic velocity model into layer interfaces that can "
                "constrain an ERT inversion or a petrophysical conversion.",
    handler=_derive_structure,
    requires=("seismic_results", "ert_data"),
    produces=("structure_results",),
    agent="StructureConstraintAgent",
    label="Extract structural constraints",
    module="seismic3d",
))


def available_methods(ctx: RunContext) -> List[str]:
    """The methods this run actually produced a result for.

    Read from the artifacts rather than from ``config['methods']``. The
    configured list says what the user hoped for; a fusion asked to combine
    three methods when two succeeded fails with "requires methods:
    ['petrophysics']" instead of combining the two it has.
    """
    present = []
    if ctx.has("seismic_results"):
        present.append("seismic")
    if ctx.has("inversion_results"):
        present.append("ert")
    if ctx.has("water_content"):
        present.append("petrophysics")
    if ctx.has("tdem_results"):
        present.append("tdem")
    return present


def fusion_pattern(methods: Sequence[str]) -> Optional[str]:
    """The richest fusion pattern ``methods`` can satisfy, or None.

    Parameters
    ----------
    methods : sequence of str
        Methods with a result, as :func:`available_methods` reports them.

    Returns
    -------
    str or None
        A pattern name, or None when no pattern's requirements are met.

    Raises
    ------
    None

    Examples
    --------
    >>> fusion_pattern(['seismic', 'ert', 'petrophysics'])
    'full_integration'
    >>> fusion_pattern(['seismic', 'ert'])
    'structure_constraint'
    >>> fusion_pattern(['ert', 'petrophysics'])
    'petrophysics_integration'
    >>> fusion_pattern(['ert']) is None
    True
    """
    have = set(methods or ())
    # Richest first: a run with all three should not settle for a pairwise one.
    for name, needed in (("full_integration", {"seismic", "ert", "petrophysics"}),
                         ("structure_constraint", {"seismic", "ert"}),
                         ("petrophysics_integration", {"ert", "petrophysics"})):
        if needed <= have:
            return name
    return None


def _fuse(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..data_fusion_agent import DataFusionAgent

    config = ctx.config
    methods = available_methods(ctx)
    pattern = config.get("fusion_pattern") or fusion_pattern(methods)
    if pattern is None:
        raise ValueError(f"No fusion pattern is satisfied by {methods or 'nothing'}.")
    agent = DataFusionAgent(**agent_kwargs(ctx))
    results = agent.execute({
        "fusion_pattern": pattern,
        "methods": methods,
        "workflow_config": config,
        "data": {"ert": ctx.get("inversion_results"),
                 "seismic": ctx.get("seismic_results"),
                 "structure": ctx.get("structure_results"),
                 "petrophysics": ctx.get("water_content"),
                 "model_output": ctx.get("model_output")},
        "output_dir": str(Path(ctx.output_dir) / "fusion"),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Data fusion failed."))
    return (f"Combined {', '.join(methods)} using the {pattern} pattern.",
            {"fusion_results": results})


register(Tool(
    name="fuse_methods",
    description="Combine two or more geophysical methods into a single "
                "interpretation. Only useful when more than one method has "
                "produced a result.",
    handler=_fuse,
    produces=("fusion_results",),
    agent="DataFusionAgent",
    label="Fuse methods",
    module="joint_inversion",
    # Offered only when a pattern is actually satisfiable, so the controller is
    # never shown a step that can only fail.
    when=lambda ctx: fusion_pattern(available_methods(ctx)) is not None,
))


# ---------------------------------------------------------------------------
# 7. the report, last
# ---------------------------------------------------------------------------
def _write_report(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..report_agent import ReportAgent
    from .._intent import unmet_requests
    from .._uncertainty import result_caveats

    config = ctx.config
    results = ctx.get("inversion_results") or {}
    agent = ReportAgent(**agent_kwargs(ctx))
    time_lapse = len(ctx.get("ert_data") or []) > 1 or bool(
        results.get("time_lapse_models"))

    site_info = build_site_info(ctx)
    payload = {
        "inversion_results": results,
        "climate_data": ctx.get("climate_data"),
        "site_info": site_info,
        "comparison_data": ctx.get("comparison_data"),
        "evaluation_results": ctx.get("evaluation_results"),
        "workflow_config": config,
        "time_lapse_method": config.get("time_lapse_method"),
        "output_dir": str(ctx.output_dir),
    }
    report = (agent.generate_timelapse_report(payload) if time_lapse
              else agent.execute({**payload, "workflow_data": results}))
    if report.get("status") == "failed":
        raise ValueError(str(report.get("error") or "Report generation failed."))

    for warning in report.get("warnings") or []:
        ctx.note(warning)
    # Products the request named and the run did not deliver, stated here rather
    # than left for the reader to notice.
    for warning in unmet_requests(config, results) + result_caveats(config, results):
        ctx.note(warning)

    files = {"report_markdown": report.get("report_file")}
    for name, path in (report.get("visualization_files") or {}).items():
        files[f"visualization_{name}"] = path
    return ("Wrote the report and its figures.",
            {"report_files": {k: v for k, v in files.items() if v},
             "interpretation": report.get("executive_summary")})


register(Tool(
    name="write_report",
    description="Write the final report from everything produced so far, with "
                "its figures. Run this last, once the products the request "
                "asked for exist.",
    handler=_write_report,
    requires=("inversion_results",),
    produces=("report_files",),
    agent="ReportAgent",
    label="Generate report",
    module="one_click",
))


def build_site_info(ctx: RunContext) -> Dict[str, Any]:
    """Site description for the report, from the configuration and file names."""
    from ..base_agent import _dates_from_filenames

    config = ctx.config
    configured = dict(config.get("site_info") or {})
    stamps = _dates_from_filenames(survey_files(config))
    coords = coords_from_config(config)
    period = configured.get("study_period")
    if not period and stamps:
        period = f"{min(stamps)} to {max(stamps)}"
    return {
        "name": str(configured.get("name", "Geophysical Monitoring Site")),
        "location": str(configured.get("location", "N/A")),
        "coordinates": str(coords) if coords else "N/A",
        "elevation": str(configured.get("elevation", "N/A")),
        "study_period": period or "N/A",
        "survey_dates": stamps,
        "description": str(configured.get(
            "description",
            "Time-lapse ERT monitoring with climate integration."
            if ctx.has("climate_data") else
            "Time-lapse ERT monitoring of subsurface resistivity change.")),
    }


def summarise_run(ctx: RunContext) -> str:
    """The interpretation text, written from the steps that actually ran.

    Replaces an f-string that restated the configuration - it reported the
    configured time-lapse method and the configured regularization whether or
    not the inversion had used them, because it was written beside the config
    rather than beside the result.
    """
    results = ctx.get("inversion_results") or {}
    lines = [f"Workflow completed in {ctx.elapsed():.0f} s.", "", "**Steps taken:**"]
    lines += [f"- {step.line()}" for step in ctx.steps]
    history = chi2_summary(results.get("chi2_values"))
    if history != "N/A":
        lines += ["", f"**Data fit:** chi-squared {history}."]
    if ctx.warnings:
        lines += ["", "**Raised during the run:**"]
        lines += [f"- {w}" for w in ctx.warnings]
    return "\n".join(lines)
