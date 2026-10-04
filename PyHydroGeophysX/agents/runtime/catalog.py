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

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._chi2 import chi2_history, chi2_summary
from .._intent import climate_blocker, wants_climate, wants_spatial_map, wants_water_content
from .._geocode import coords_from_config, geocode_place
from .._method import IMPLEMENTED_SCHEME
from .context import RunContext
from .tools import TOOLS, Tool, register


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

        # A structural constraint's layers, when the model has them; otherwise
        # whatever the mesh carries, which resolve_layers then judges.
        layers = results.get("cell_markers")
        markers = np.asarray(
            layers if layers is not None and np.asarray(layers).size == models[0].size
            else mesh.cellMarkers(), dtype=int)
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


def _reexport(ctx: RunContext, results: Dict[str, Any], what: str) -> Dict[str, Any]:
    """Rewrite the model bundle for a model that replaced the run's model.

    The studio opens ``ert_model/``, which the inversion step wrote; left alone
    it keeps showing the model the run has since replaced.
    """
    bundle = export_model_bundle(ctx, results)
    if not bundle and ctx.get("model_directory"):
        ctx.note(f"{ctx.get('model_directory')} still holds the model the run "
                 f"replaced: {what} could not be written over it.")
    return {"model_directory": bundle or None}


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
    declared = config.get("instrument")

    from .. import _raw_data as raw

    loaded, files, failed, read_as = [], [], [], []
    # What each survey holds, and a pseudosection of it drawn as it loads: the
    # Live tab shows each one the moment it is written, which is the only thing
    # a user has to look at while a long series loads. A long series is sampled.
    readings, figures = [], []
    drawn = set(raw.chosen(len(paths)))
    for index, path in enumerate(paths):
        resolved = resolve_path(path, project_dir)
        # A default instrument is a guess, and the loader refuses a file whose
        # header names another one - so a request that named no instrument had
        # every DAS-1 file refused as "Syscal". When nobody chose, the header
        # decides; the default is for a file that says nothing about itself.
        detected = None if declared else loader._detect_instrument_from_header(resolved)
        instrument = declared or detected or DEFAULT_INSTRUMENT
        result = loader.execute({
            "data_file": resolved,
            "instrument": instrument,
            "project_dir": project_dir,
            "electrode_file": electrode,
            "crs": config.get("crs", "local"),
        })
        if result.get("status") != "success":
            failed.append(f"{Path(resolved).name}: {_loader_failure(result)}")
            continue
        loaded.append(result["ert_data"])
        files.append(str(path))
        read_as.append((instrument, detected))
        readings.append(raw.ert_readings(result["ert_data"]))
        if index in drawn:
            figure = _draw_ert_survey(ctx, readings[-1], index, len(paths), path)
            if figure:
                figures.append(figure)

    if not loaded:
        raise ValueError("No ERT survey could be loaded. " + "; ".join(failed))
    for message in failed:
        ctx.note(f"An ERT file could not be loaded and was left out: {message}")
    readers = {instrument for instrument, _ in read_as}
    if not declared and len(readers) == 1:
        # Recorded, so the report and every later step name the reader that
        # was actually used rather than a default nobody chose.
        config["instrument"] = readers.pop()
        if all(found for _, found in read_as):
            ctx.note(f"No instrument was named, so the files were read as "
                     f"{config['instrument']}, as their headers say.")

    from .._intent import wants_water_content as _wwc  # noqa: F401 - documented below
    summary = _ert_load_summary(readings, files, len(paths))
    if failed:
        summary += f" {len(failed)} could not be read."
    # For the report, the surveys side by side on one colour scale.
    report_figure = _draw_ert_series(ctx, readings, files) if len(readings) > 1 else (
        figures[:1] or [None])[0]
    figures = [report_figure] if report_figure else []
    # The files that loaded, in order. The acquisition times are read from the
    # file names, so a list that still holds a file which failed to load no
    # longer lines up with the surveys, and every survey then loses its date.
    return summary, {"ert_data": loaded, "n_surveys": len(loaded), "ert_files": files,
                     "ert_raw_figures": figures}


def _draw_ert_survey(ctx: RunContext, readings: Mapping[str, Any], index: int, count: int,
                     path: Any) -> Optional[Tuple[str, str]]:
    """One loaded survey's apparent-resistivity pseudosection, and its caption."""
    from .. import _raw_data as raw
    from .._figstyle import style_from_config

    style = style_from_config({"figure_style": _figure_style(ctx)})
    name = Path(str(path)).name
    which = f"Survey {index + 1} of {count} ({name})" if count > 1 else name
    try:
        drawn = raw.ert_pseudosection(
            readings, raw.figure_path(ctx.output_dir, f"ert_apparent_resistivity_{index + 1:02d}"),
            title=f"{which}: apparent resistivity, {np.size(readings.get('rhoa'))} readings",
            unit=style.length_unit, cmap=style.cmap_for("resistivity"))
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        return None
    if not drawn:
        return None
    return drawn, (f"Apparent resistivity of {which} as read, each reading at the midpoint "
                   "of its four electrodes and a pseudo-depth of 0.19 of their spread"
                   + ("" if readings.get("converted") else
                      "; the values are the file's own, which may be resistances") + ".")


def _draw_ert_series(ctx: RunContext, readings: Sequence[Mapping[str, Any]],
                     files: Sequence[str]) -> Optional[Tuple[str, str]]:
    """The loaded surveys' pseudosections side by side on one scale, and the caption."""
    from .. import _raw_data as raw
    from .._figstyle import style_from_config

    style = style_from_config({"figure_style": _figure_style(ctx)})
    picked = raw.chosen(len(readings))
    try:
        drawn = raw.ert_pseudosections(
            [readings[k] for k in picked],
            raw.figure_path(ctx.output_dir, "ert_apparent_resistivity_series"),
            titles=[f"Survey {k + 1}: {Path(str(files[k])).name}" for k in picked],
            unit=style.length_unit, cmap=style.cmap_for("resistivity"))
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        return None
    if not drawn:
        return None
    which = (f"the {len(readings)} surveys" if len(picked) == len(readings)
             else f"{len(picked)} of the {len(readings)} surveys, the first and last among them")
    return drawn, (f"Apparent resistivity of {which} as read, on one colour scale, each "
                   "reading at the midpoint of its four electrodes and a pseudo-depth of 0.19 "
                   "of their spread: the data the inversion fitted.")


def _ert_load_summary(readings: Sequence[Mapping[str, Any]], files: Sequence[str],
                      configured: int) -> str:
    """What the load step read, in a sentence: counts and the apparent resistivity."""
    from .. import _raw_data as raw

    if len(readings) == 1:
        return f"Loaded {Path(str(files[0])).name}: {raw.describe_ert(readings[0])}."
    counts = [int(r.get("n_readings") or 0) for r in readings]
    electrodes = sorted({int(r.get("n_electrodes") or 0) for r in readings})
    values = np.concatenate([np.asarray(r.get("rhoa"), dtype=float) for r in readings]
                            + [np.zeros(0)])
    text = (f"Loaded {len(readings)} ERT surveys from {configured} configured files: "
            f"{'/'.join(map(str, electrodes))} electrodes, {min(counts)}"
            + ("" if min(counts) == max(counts) else f" to {max(counts)}")
            + " readings per survey")
    if values.size:
        text += (f"; apparent resistivity {values.min():.3g} to {values.max():.3g} Ω·m "
                 f"across the series (median {np.median(values):.3g})")
    return text + "."


def _loader_failure(result: Mapping[str, Any]) -> str:
    """Why a survey was not loaded, in the loader's own words.

    A refusal - the declared instrument contradicting the file header - comes
    back as ``needs_review`` with a summary and a fix hint but no ``error``, and
    reporting only the error printed "None", which neither the user nor a
    recovery could act on.
    """
    reason = (result.get("error") or result.get("summary")
              or f"the loader returned status {result.get('status')!r}")
    hint = result.get("error_fix_hint")
    return f"{reason} {hint}" if hint else str(reason)


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

    # ClimateDataAgent takes the period as ``dates`` = (start, end), which is the
    # key the request parser fills (inferring it from the survey times when the
    # request names none). Passing start_date/end_date instead failed every
    # retrieval with "dates parameter is required".
    climate_config = config.get("climate_config") or {}
    dates = climate_config.get("dates")
    if not dates and climate_config.get("start_date") and climate_config.get("end_date"):
        dates = (climate_config["start_date"], climate_config["end_date"])
    if not dates:
        raise ValueError("No period for the climate series: give "
                         "climate_config['dates'] as (start, end).")
    agent = ClimateDataAgent(**agent_kwargs(ctx))
    # The agent reads its options (pet_method, antecedent_days, ert_timestamps,
    # crs) at the top level; nested under "climate_config" they never arrived.
    result = agent.execute({
        **climate_config,
        "coords": list(coords),
        "dates": dates,
        "output_dir": str(Path(ctx.output_dir) / "climate"),
    })
    if result.get("status") not in (None, "success"):
        raise ValueError(str(result.get("error") or "Climate retrieval failed."))
    for note in result.get("notes") or []:
        ctx.note(note)
    # The whole result, not its DataFrame: the reports read metadata, derived
    # features and the survey alignment from it. (``frame or result`` also
    # raised, a DataFrame having no truth value.)
    source = (result.get("metadata") or {}).get("source", "the climate service")
    return (f"Retrieved daily meteorological data for {coords} from {source}.",
            {"climate_data": result})


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
def _survey_count(ctx: RunContext) -> int:
    """How many ERT surveys the run holds - or, on a projected route, will hold."""
    if ctx.projected("ert_data"):
        return len(survey_files(ctx.config))
    return len(ctx.get("ert_data") or [])


def _is_time_lapse(ctx: RunContext) -> bool:
    return _survey_count(ctx) >= 2


def _is_single(ctx: RunContext) -> bool:
    return _survey_count(ctx) == 1


def _invert_time_lapse(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..ert_inversion_agent import ERTInversionAgent

    config = ctx.config
    agent = ERTInversionAgent(**agent_kwargs(ctx))
    results = agent.execute({
        "time_lapse_data": ctx.get("ert_data"),
        # The acquisition time lives in the file name, and the loaded containers
        # do not carry it; without these the sequence degrades to a 1..n index.
        "source_files": ctx.get("ert_files") or survey_files(config),
        "inversion_mode": "time-lapse",
        "time_lapse_method": config.get("time_lapse_method", IMPLEMENTED_SCHEME),
        "temporal_regularization": config.get("temporal_regularization", 10.0),
        "baseline_index": 0,
        # No linear solver: the library's own default (spd_cholesky for the
        # time-lapse normal equations) applies unless the caller chose one.
        # 'cgls' here overrode it, and the solver then warned that a
        # least-squares method had been handed a normal matrix.
        "inversion_params": config.get("inversion_params",
                                       {"lambda": 15.0, "max_iterations": 10}),
        "output_dir": str(Path(ctx.output_dir) / "inversion"),
        # The evaluation step interprets the model the run keeps, once.
        "interpret": False,
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
        # The evaluation step interprets the model the run keeps, once.
        "interpret": False,
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
    first = results if isinstance(results, dict) else {}
    time_lapse = len(surveys) > 1
    # A retry re-runs the inversion from this input with lambda (and the
    # iteration cap) changed, so everything else has to be what the first
    # attempt used. The time-lapse settings were not passed at all: a retry ran
    # at the agent's default alpha of 10 on a 1..n index instead of the survey
    # dates, and the temperature correction then ran on those indices.
    used = (first.get("inversion_params")
            or (first.get("processing") or {}).get("inversion_params"))
    payload = {
        "inversion_results": results,
        "ert_data": surveys[0] if surveys else None,
        "time_lapse_data": surveys if time_lapse else None,
        "inversion_mode": "time-lapse" if time_lapse else "standard",
        "inversion_params": dict(used) if used else config.get("inversion_params", {}),
        "auto_adjust": config.get("auto_adjust", True),
        "output_dir": str(Path(ctx.output_dir) / "inversion"),
        "max_attempts": config.get("max_attempts", 3),
        "quality_threshold": config.get("quality_threshold", 70),
        "progress_callback": ctx.settings.get("progress_callback"),
        "project_dir": config.get("project_dir", "."),
        "instrument": config.get("instrument", DEFAULT_INSTRUMENT),
    }
    if time_lapse:
        payload.update({
            "source_files": ctx.get("ert_files") or survey_files(config),
            "temporal_regularization": first.get(
                "temporal_regularization", config.get("temporal_regularization", 10.0)),
            "time_lapse_method": (first.get("time_lapse_method_requested")
                                  or config.get("time_lapse_method", IMPLEMENTED_SCHEME)),
            "baseline_index": 0,
        })
    if first.get("structure_constrained"):
        # A retry re-inverts without the seismic interface, so a better score
        # would swap the constrained model for an unconstrained one. Score it,
        # and leave the model alone.
        payload["auto_adjust"] = False
    agent = InversionEvaluationAgent(**agent_kwargs(ctx))
    evaluation = agent.execute(payload)

    # The best-scoring attempt replaces the model whenever a retry beat the
    # first, whether or not it then cleared the threshold: keeping the first
    # while the report printed the retry's score and lambda described a model
    # the run had thrown away.
    best = evaluation.get("final_results")
    adopted = (isinstance(best, dict) and best is not results
               and best.get("status") == "success")
    outputs: Dict[str, Any] = {}
    if adopted:
        results = best
        outputs.update(_reexport(ctx, results, "the adopted retry"))
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
    if adopted:
        history = evaluation.get("evaluation_history") or []
        lam = (evaluation.get("adjusted_params") or {}).get("lambda")
        summary += (" The run now uses the best-scoring retry's model"
                    + (f" (lambda {float(lam):g})" if lam is not None else "")
                    + (f", up from {float(history[0]['quality_score']):.1f} on the first "
                       f"attempt." if history and history[0].get("quality_score") is not None
                       else "."))
    if evaluation.get("status") != "success":
        ctx.note(str(verdict))
    return summary, {"inversion_results": results, "evaluation_results": stripped,
                     **outputs}


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


#: The tools that make each product a request can name (``_intent.PRODUCTS``).
PRODUCERS = {"water_content": ("convert_water_content", "convert_tdem_water_content",
                               "convert_mt_water_content"),
             "climate": ("fetch_climate",),
             "spatial_map": ("map_tdem_plan_view",)}

#: The steps that recover a resistivity model a conversion can start from.
_MODEL_STEPS = ("load_ert_surveys", "invert_time_lapse", "invert_ert", "load_tdem_data",
                "invert_tdem", "load_mt_sites", "invert_mt")


def plain_error(error: Any) -> str:
    """A step's error for a reader, without the exception class a tool raised it as.

    The tools raise ``ValueError`` to explain themselves, and the step records
    ``"ValueError: <explanation>"``; the class name means nothing to a reader
    of the report. Any other class is kept, because there it is information.

    >>> plain_error("ValueError: no site coordinates")
    'no site coordinates'
    >>> plain_error("KeyError: 'rhoa'")
    "KeyError: 'rhoa'"
    """
    return re.sub(r"^(?:ValueError|RuntimeError):\s*", "", str(error or "")).strip()


def _last_failure(ctx: RunContext, tools: Sequence[str]) -> Any:
    """The latest failed step among ``tools`` that no later attempt recovered."""
    return next((step for step in reversed(ctx.steps)
                 if step.tool in tools and step.status == "failed" and not ctx.ran(step.tool)),
                None)


def _water_content_failure(ctx: RunContext, results: Mapping[str, Any]) -> Optional[str]:
    """Why water content the request asked for is missing, or None.

    The report states a failed conversion as a failure, with its reason; it
    used to read as "not requested", or leave the section out. With no
    resistivity model to convert, the reason is the step that failed to make
    one - a TDEM file that could not be read, say - rather than a bare "this
    run produced none".
    """
    if (not _water_content_wanted(ctx) or results.get("water_content_mean") is not None
            or results.get("time_lapse_water_content") or ctx.has("water_content")):
        return None
    failed = _last_failure(ctx, PRODUCERS["water_content"])
    if failed is not None:
        return plain_error(failed.error or failed.summary)
    if not (ctx.has("inversion_results") or ctx.has("tdem_results") or ctx.has("mt_results")):
        broken = _last_failure(ctx, _MODEL_STEPS)
        return ("no resistivity model was recovered to convert"
                + (f": {broken.description or broken.tool} failed "
                   f"({plain_error(broken.error)})" if broken is not None else ""))
    return "the conversion step did not run"


def shortfall_reasons(ctx: RunContext,
                      results: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Why each product the request can name is missing from this run, where it is.

    For :func:`~PyHydroGeophysX.agents._intent.unmet_requests`, which on its
    own can say only that a product is missing; the reason is what the user can
    act on.

    Examples
    --------
    >>> ctx = RunContext('x', {'user_request': 'estimate the water content'})
    >>> _ = ctx.begin('invert_tdem', description='Run TDEM inversion')
    >>> ctx.finish(status='failed', error='ValueError: no time column')
    >>> shortfall_reasons(ctx)['water_content']
    'no resistivity model was recovered to convert: Run TDEM inversion failed (no time column)'
    """
    reasons: Dict[str, str] = {}
    water = _water_content_failure(
        ctx, results if results is not None else (ctx.get("inversion_results") or {}))
    if water:
        reasons["water_content"] = water
    if wants_climate(ctx.config) and not ctx.has("climate_data"):
        failed = _last_failure(ctx, PRODUCERS["climate"])
        reasons["climate"] = (plain_error(failed.error) if failed is not None
                              else climate_blocker(ctx.config) or "the climate step did not run")
    if wants_spatial_map(ctx.config) and not ctx.has("plan_maps"):
        failed = _last_failure(ctx, PRODUCERS["spatial_map"])
        reasons["spatial_map"] = (plain_error(failed.error) if failed is not None
                                  else "the mapping step did not run")
    return reasons


def not_delivered_items(ctx: RunContext, delivered: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """``(what, why)`` for everything a report must say it does not contain.

    The products the request named and the run did not make, then each step
    that failed and was not recovered, once and with its reason. A run with a
    broken TDEM file beside good ERT data used to write a complete-looking
    report that never mentioned TDEM, and to call itself a success.

    Examples
    --------
    >>> ctx = RunContext('x', {'data_file': 'a.ohm'})
    >>> _ = ctx.begin('invert_tdem', description='Run TDEM inversion')
    >>> ctx.finish(status='failed', error='ValueError: TDEM data file not found: s.csv')
    >>> not_delivered_items(ctx, {})
    [('Run TDEM inversion', 'TDEM data file not found: s.csv')]
    """
    from .._intent import PRODUCTS, unmet_products

    items: List[Tuple[str, str]] = []
    covered: set = set()
    for product, reason in unmet_products(ctx.config, dict(delivered),
                                          shortfall_reasons(ctx, delivered)):
        items.append((PRODUCTS[product], str(reason or "it was not produced")))
        # The step that makes a missing product is stated through the product.
        covered.update(PRODUCERS.get(product, ()))
    last: Dict[str, Any] = {}
    for step in ctx.steps:
        if step.status == "failed" and not ctx.ran(step.tool) and step.tool not in covered:
            last[step.tool] = step
    items += [(step.description or step.tool, plain_error(step.error) or "the step failed")
              for step in last.values()]
    return items


def _structure_pending(ctx: RunContext) -> bool:
    """Whether a seismic structural constraint is still to come for the model.

    With a seismic line beside one ERT survey, ``derive_structure`` re-inverts
    the survey with the velocity interface built into the mesh, and that model
    replaces the unconstrained one. Registration order put the conversion
    first, so water content was computed from the model about to be replaced
    and the constrained one was never used. A conversion or a fusion therefore
    waits until the constraint has been tried, or can no longer be.
    """
    if _survey_count(ctx) != 1:
        return False
    if not any(ctx.config.get(key) for key in ("seismic_file", "raw_seismic_file")):
        return False
    if ctx.attempted("derive_structure"):
        return False
    # A seismic step that failed or was skipped leaves nothing to constrain with.
    return not (ctx.attempted("invert_seismic") and not ctx.has("seismic_results"))


def _state_prior(ctx: RunContext, step: Dict[str, Any]) -> str:
    """Tell the user what the conversion assumed; return the flag for the summary.

    Every conversion to water content says which petrophysical parameters it
    drew and over what ranges (:func:`.._uncertainty.describe_prior`). When any
    of them were defaults rather than the user's, that is a warning - the
    result exists but is not reliable - and the step's own summary says so too,
    so it is seen while the run is still going, not only in the report.
    """
    for note in step.get("realization_notes") or []:
        ctx.note(note)
    dropped = ctx.config.get("petrophysics_dropped") or []
    if dropped:
        ctx.note("Petrophysical values the request does not contain were not used ("
                 + "; ".join(map(str, dropped)) + "): they came from reading the "
                 "request, not from you, and only a relationship you give counts as yours.")
    relationship = step.get("petrophysical_relationship") or ""
    if relationship and relationship != "user" and step.get("prior_statement"):
        ctx.note(step["prior_statement"])
    if relationship == "default":
        return (" Not reliable: no petrophysical relationship was given, so it rests on "
                "generic default parameters (named, with their ranges, in the warnings).")
    if relationship == "partial":
        return (" Partly on default petrophysical parameters (named, with their ranges, "
                "in the warnings).")
    return ""


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

    layers = results.get("cell_markers")
    requested = config.get("layer_params") or None
    layer_params = None
    if layers is not None and np.asarray(layers).size == np.asarray(models[0]).size:
        # The layers a structural constraint drew (above and below the seismic
        # interface), one per model cell. The parameter mesh's own markers only
        # number its cells, and would make the conversion a single unit.
        markers = np.asarray(layers)
        # Parameters the request gave per named layer (regolith, fractured
        # bedrock) apply to these layers in order, as the fusion pipeline this
        # replaced applied them; without it they were dropped for defaults.
        layer_params = requested
    else:
        markers = (np.array(mesh.cellMarkers()) if mesh is not None
                   else np.zeros(len(models[0])))
        if requested:
            # Nothing to assign them to. Say so, here and in the report,
            # rather than convert with other parameters in silence.
            ctx.note("Layer parameters given for " + ", ".join(map(str, requested))
                     + " were not applied: this model has no structural layers "
                     "(a seismic interface draws them), so the conversion used "
                     + ("petrophysical_params." if config.get("petrophysical_params")
                        else "generated petrophysical parameters."))
    # What the report states about the parameters, from what was applied.
    results["layer_params_applied"] = layer_params or {}
    results["layer_params_not_applied"] = {} if layer_params else (requested or {})
    agent = PetrophysicsAgent(**agent_kwargs(ctx))
    per_step: List[Dict[str, Any]] = []
    for index, model in enumerate(models):
        step = agent.execute({
            "resistivity_model": model,
            "mesh": mesh,
            "cell_markers": markers,
            "petrophysical_params": config.get("petrophysical_params", {}),
            "layer_params": layer_params,
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
    # Where the conversion completed the request's per-layer parameters with
    # defaults, or two named layers fell on one; the same for every step.
    for note in per_step[0].get("layer_param_notes") or []:
        ctx.note(note)

    results["time_lapse_water_content"] = per_step
    results["water_content_mean"] = per_step[0].get("water_content_mean")
    results["water_content_std"] = per_step[0].get("water_content_std")
    results["petrophysical_params"] = config.get("petrophysical_params", {})
    first = per_step[0]
    for key in ("petrophysical_relationship", "prior_statement", "prior_ranges_text",
                "n_realizations"):
        results[key] = first.get(key)
    flag = _state_prior(ctx, first)

    figure = _draw_water_content(ctx, mesh, per_step)
    if figure:
        results["water_content_figure"] = figure

    layering = first.get("layering") or ""
    source = " the structure-constrained" if results.get("structure_constrained") else ""
    means = [float(np.nanmean(step.get("water_content_mean"))) for step in per_step]
    sigmas = [float(np.nanmean(step.get("water_content_std"))) for step in per_step]
    summary = (f"Converted {len(per_step)} of {len(models)}{source} model(s) to water "
               f"content by Monte Carlo petrophysics ({first.get('n_realizations', '')} "
               f"draws): mean {np.mean(means):.3f} ± {np.mean(sigmas):.3f} "
               f"(average standard deviation per cell).{flag}")
    if layering:
        summary += f" {layering}"
    return summary, {"inversion_results": results, "water_content": per_step}


def _draw_water_content(ctx: RunContext, mesh: Any, per_step: Sequence[Mapping[str, Any]]
                        ) -> Optional[str]:
    """The water content and its standard deviation, drawn as the conversion ends.

    Up to three surveys - the first, the middle and the last - so the Live tab
    shows the uncertainty the moment it exists, not only in the report.
    """
    from .. import _raw_data as raw
    from .._figstyle import style_from_config
    from .._uncertainty import draw_water_content

    picked = raw.chosen(len(per_step), 3)
    try:
        return draw_water_content(
            mesh, [per_step[k].get("water_content_mean") for k in picked],
            [per_step[k].get("water_content_std") for k in picked],
            Path(ctx.output_dir) / "petrophysics" / "water_content_mean_and_uncertainty.png",
            titles=[f"Survey {k + 1}" if len(per_step) > 1 else "" for k in picked],
            style=style_from_config({"figure_style": _figure_style(ctx)}))
    except Exception:  # noqa: BLE001 - the numbers stand without the picture
        return None


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
    when=lambda ctx: _water_content_wanted(ctx) and not _structure_pending(ctx),
))


# ---------------------------------------------------------------------------
# 6. other methods
# ---------------------------------------------------------------------------
#: The request parser's seismic stage writes its settings at the top level, in
#: snake case; SeismicAgent reads them by these names.
_SEISMIC_TOP_LEVEL = (("lam", "lam"), ("z_weight", "zWeight"), ("v_top", "vTop"),
                      ("v_bottom", "vBottom"), ("para_depth", "paraDepth"),
                      ("velocity_limits", "limits"))


def seismic_inversion_params(config: Mapping[str, Any]) -> Dict[str, Any]:
    """The seismic inversion settings a configuration gives, under any of its names.

    ``seismic_inversion_params`` is this runtime's own key and wins, setting by
    setting. The request parser writes ``seismic_params`` (its fusion stage)
    and, for a seismic-only request, top-level ``lam``, ``z_weight`` and the
    like; reading the first key alone sent SeismicAgent an empty dictionary, so
    every value the request stated was replaced by the agent's defaults.

    Examples
    --------
    >>> seismic_inversion_params({'seismic_params': {'lam': 5, 'zWeight': 1.0}})
    {'lam': 5, 'zWeight': 1.0}
    >>> seismic_inversion_params({'seismic_params': {'lam': 5},
    ...                           'seismic_inversion_params': {'lam': 30}, 'z_weight': 0.5})
    {'zWeight': 0.5, 'lam': 30}
    """
    merged: Dict[str, Any] = {name: config[key] for key, name in _SEISMIC_TOP_LEVEL
                              if config.get(key) is not None}
    for key in ("seismic_params", "seismic_inversion_params"):
        given = config.get(key)
        if isinstance(given, Mapping):
            merged.update({k: v for k, v in given.items() if v is not None})
    return merged


_SEGY_SUFFIXES = (".sgy", ".segy")


def _raw_seismic(config: Mapping[str, Any]) -> Optional[str]:
    """The SEG-Y file to pick, when the run starts from traces rather than travel times."""
    raw = config.get("raw_seismic_file")
    given = config.get("seismic_file")
    if not raw and given and Path(str(given)).suffix.lower() in _SEGY_SUFFIXES:
        raw = given
    return (resolve_path(raw, config.get("project_dir", ".")) or None) if raw else None


def _travel_time_file(config: Mapping[str, Any]) -> Optional[str]:
    """The travel-time file the configuration names, when it is not a SEG-Y file."""
    given = config.get("seismic_file")
    if given and Path(str(given)).suffix.lower() not in _SEGY_SUFFIXES:
        return resolve_path(given, config.get("project_dir", ".")) or None
    return None


def _pick_first_breaks(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..seismic_agent import SeismicAgent

    config = ctx.config
    agent = SeismicAgent(**agent_kwargs(ctx))
    inputs = {
        "raw_seismic_file": _raw_seismic(config),
        "geophone_file": config.get("geophone_file"),
        "topography_file": config.get("topography_file"),
        "first_break_params": dict(config.get("first_break_params") or {}),
        "output_dir": str(Path(ctx.output_dir) / "seismic"),
        "align_origin": config.get("align_origin"),
        "figure_style": _figure_style(ctx),
    }
    picks = agent.pick_travel_times(inputs)
    mismatch = picks.get("origin_mismatch") if picks.get("status") != "success" else None
    # Only a real translation is a choice to put to the user: with the two
    # origins 0 m apart, either answer re-ran the same failing geometry.
    if mismatch and abs(float(mismatch.get("shift") or 0.0)) > 1e-6 and not inputs["align_origin"]:
        # Not a guess this code is entitled to make: the coordinate file and
        # the SEG-Y headers describe the same line from two different origins,
        # and only somebody who knows the survey can say which one to report.
        # Asking beats both alternatives - failing throws away the picking work
        # already done, and choosing silently puts an unlabelled shift into
        # coordinates that will later be laid beside an ERT line.
        choice = ctx.ask(
            f"The coordinate file and the SEG-Y headers place this line "
            f"{abs(mismatch.get('shift', 0.0)):g} m apart along x. The survey "
            f"geometry is the same either way; which origin should the results carry?",
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
                "was left open. " + str(picks.get("error") or ""))
        inputs["align_origin"] = choice
        ctx.note(f"Seismic geometry was reconciled onto the "
                 f"{'coordinate file' if choice == 'profile' else 'SEG-Y header'} "
                 f"origin, a shift of {abs(mismatch.get('shift', 0.0)):g} m, "
                 f"chosen during the run. The velocity model is unaffected; the "
                 f"x coordinates it is reported against are not.")
        picks = agent.pick_travel_times(inputs)
    if picks.get("status") != "success":
        raise ValueError(str(picks.get("error") or "First arrivals could not be picked."))
    for note in (picks.get("geometry_warnings") or []) + (picks.get("pick_warnings") or []):
        ctx.note(str(note))
    dropped = picks.get("dropped_shots") or []
    positions = ", ".join(f"{float(shot['source_x']):g}" for shot in dropped)
    left_out = (f"; {len(dropped)} shot{'s' if len(dropped) != 1 else ''} left out "
                f"(x = {positions} m) for times that disagree with their reciprocals"
                if dropped else "")
    if picks.get("repicked_picks"):
        left_out += (f"; {picks['repicked_picks']} picks that strayed from their shot's "
                     "first-arrival curve picked again along it")
    if picks.get("rejected_picks"):
        left_out += (f"; {picks['rejected_picks']} single picks left out for breaking their "
                     "shot's first-arrival curve")
    if picks.get("neighbour_repicked") or picks.get("neighbour_rejected"):
        left_out += (f"; against the neighbouring shots at the same geophone, "
                     f"{picks.get('neighbour_repicked', 0)} picked again and "
                     f"{picks.get('neighbour_rejected', 0)} left out")
    return (f"Picked {picks.get('n_picks', '?')} first arrivals on "
            f"{picks.get('picked_shots', '?')} shots{left_out}.", {"seismic_picks": picks})


register(Tool(
    name="pick_first_breaks",
    description="Pick the first-arrival travel times in a raw SEG-Y file, place them on "
                "the survey geometry, and leave out shots whose times disagree with "
                "their reciprocals.",
    handler=_pick_first_breaks,
    produces=("seismic_picks",),
    agent="SeismicAgent",
    label="Pick first-arrival travel times",
    module="seismic",
    when=lambda ctx: bool(_raw_seismic(ctx.config)) and not _travel_time_file(ctx.config),
))


def _load_seismic_traveltimes(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from pygimli.physics import traveltime as tt

    from .. import _raw_data as raw
    from .._figstyle import style_from_config

    path = _travel_time_file(ctx.config)
    if not path or not Path(str(path)).exists():
        raise ValueError(f"Seismic travel-time file not found: {ctx.config.get('seismic_file')}")
    data = tt.load(str(path))
    if not data.size():
        raise ValueError(f"{Path(str(path)).name} holds no travel times.")
    table = raw.traveltime_table(data)
    times = table["time_s"]
    shots = int(np.unique(table["shot"]).size)
    geophones = int(np.unique(np.asarray(data["g"], dtype=int)).size)
    name = Path(str(path)).name
    try:
        figure = raw.traveltime_curves(
            table, raw.figure_path(ctx.output_dir, "seismic_traveltimes"),
            title=f"{name}: {data.size()} travel times from {shots} shots",
            unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit)
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        figure = None
    return (f"Read {data.size()} first-arrival travel times from {name}: {shots} shots into "
            f"{geophones} geophones, {times.min() * 1e3:.1f} to {times.max() * 1e3:.1f} ms.",
            {"seismic_traveltimes": {"file": str(path), "n_data": int(data.size()),
                                     "n_shots": shots, "n_geophones": geophones,
                                     "time_range_s": [float(times.min()), float(times.max())],
                                     "figure": figure}})


register(Tool(
    name="load_seismic_traveltimes",
    description="Read the first-arrival travel-time file the configuration names and draw "
                "its travel-time curves. Run before the seismic inversion.",
    handler=_load_seismic_traveltimes,
    produces=("seismic_traveltimes",),
    agent="SeismicAgent",
    label="Load seismic travel times",
    module="seismic",
    when=lambda ctx: bool(_travel_time_file(ctx.config)),
))


#: What the picking step found that the report of the inversion describes.
_PICK_KEYS = ("raw_seismic_file", "traveltime_file", "first_break_picks_file", "picks_figure",
              "gathers_figure", "n_picks", "picked_shots", "dropped_shots", "rejected_picks", "repicked_picks",
              "neighbour_rejected", "neighbour_repicked", "segy_metadata")


def _run_seismic(ctx: RunContext, params: Optional[Mapping[str, Any]] = None,
                 folder: Optional[Path] = None) -> Dict[str, Any]:
    """Invert the run's travel times; ``params`` replace the configured settings."""
    from ..seismic_agent import SeismicAgent

    config = ctx.config
    agent = SeismicAgent(**agent_kwargs(ctx))
    picks = dict(ctx.get("seismic_picks") or {})
    loaded = dict(ctx.get("seismic_traveltimes") or {})
    inputs = {
        "seismic_file": (picks.get("traveltime_file") or loaded.get("file")
                         or _travel_time_file(config)),
        "velocity_threshold": config.get("velocity_threshold", 1200.0),
        "inversion_params": (seismic_inversion_params(config) if params is None
                             else dict(params)),
        "output_dir": str(folder or Path(ctx.output_dir) / "seismic"),
        # Traced as a step of its own, after the model exists.
        "extract_interfaces": False,
        "figure_style": _figure_style(ctx),
    }
    results = agent.execute(inputs)
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Seismic inversion failed."))
    results.update({key: picks[key] for key in _PICK_KEYS if picks.get(key) is not None})
    results["geometry_warnings"] = list(picks.get("geometry_warnings") or []) + list(
        results.get("geometry_warnings") or [])
    results["pick_warnings"] = list(picks.get("pick_warnings") or [])
    if loaded.get("figure"):
        results["raw_figures"] = [(loaded["figure"], "First-arrival travel times as read, "
                                                     "against geophone position, one curve "
                                                     "per shot; triangles mark the shots.")]
    return results


def _invert_seismic(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    results = _run_seismic(ctx)
    span = results.get("velocity_range")
    detail = f" Velocity {span[0]:.0f} to {span[1]:.0f} m/s." if span else ""
    fit = (f" Relative RMS misfit {float(results['rrms']):.3g}%."
           if results.get("rrms") is not None else "")
    return (f"Recovered a seismic velocity model from {results.get('n_data', '?')} "
            f"travel times.{detail}{fit}", {"seismic_results": results})


register(Tool(
    name="invert_seismic",
    description="Invert first-arrival travel times for a P-wave velocity model by "
                "refraction tomography.",
    handler=_invert_seismic,
    produces=("seismic_results",),
    agent="SeismicAgent",
    label="Run seismic refraction inversion",
    module="seismic",
    # On travel times that were read, or picked: the loading step draws them
    # first, as the ERT loader draws its surveys.
    when=lambda ctx: ctx.has("seismic_picks") or ctx.has("seismic_traveltimes"),
))


def _attempt_row(attempt: int, results: Mapping[str, Any],
                 evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    return {"attempt": attempt, "lam": (results.get("inversion_params") or {}).get("lam"),
            "chi2": results.get("chi2"), "quality_score": evaluation.get("quality_score")}


def _quality_threshold(ctx: RunContext) -> float:
    from .._method_evaluation import QUALITY_THRESHOLD

    try:
        return float(ctx.config.get("quality_threshold", QUALITY_THRESHOLD))
    except (TypeError, ValueError):
        return QUALITY_THRESHOLD


def _needs_review(ctx: RunContext, what: str, evaluation: Mapping[str, Any]) -> None:
    """One short warning when an evaluation falls short; the report says why."""
    if evaluation.get("status") != "success":
        ctx.note(f"{what} quality needs review ({float(evaluation['quality_score']):.0f}/100); "
                 "the report's Inversion Quality section says why.")


def _evaluate_seismic(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Score the velocity model, and re-invert with lambda changed while the fit is off.

    As ``evaluate_inversion`` does for ERT: a fit below the target (errors
    overstated) doubles lambda, one above it halves it, up to
    ``max_attempts`` inversions in all, stopping once the score passes or the
    fit crosses the target. The best-scoring model is the one the run keeps.
    """
    from .. import _method_evaluation as quality
    from .._figstyle import style_from_config

    config = ctx.config
    first = dict(ctx.get("seismic_results") or {})
    unit = style_from_config({"figure_style": _figure_style(ctx)}).length_unit
    threshold = _quality_threshold(ctx)
    folder = Path(first.get("output_dir") or Path(ctx.output_dir) / "seismic")

    def score(results: Mapping[str, Any], attempt: int) -> Dict[str, Any]:
        name = ("seismic_data_fit.png" if attempt == 1
                else f"seismic_data_fit_attempt{attempt}.png")
        return quality.evaluate_seismic(results, figure_path=folder / name, unit=unit,
                                        threshold=threshold)

    current, evaluation = first, score(first, 1)
    best, best_score, adopted = first, evaluation, 1
    history = [_attempt_row(1, first, evaluation)]
    attempts, limit = 1, int(config.get("max_attempts", 3))
    while config.get("auto_adjust", True) and attempts < limit and best_score["status"] != "success":
        direction = evaluation["metrics"]["data_fit"].get("status")
        lam = (current.get("inversion_params") or {}).get("lam")
        if direction not in ("underfit", "overfit") or lam is None:
            break
        params = {**current["inversion_params"],
                  "lam": float(lam) * (0.5 if direction == "underfit" else 2.0)}
        attempts += 1
        try:
            retry = _run_seismic(ctx, params, Path(ctx.output_dir) / "seismic" / f"attempt_{attempts}")
        except ValueError as exc:
            ctx.note(f"Re-inverting the seismic data with lambda {params['lam']:g} failed "
                     f"({plain_error(exc)}); the earlier model is kept.")
            break
        current, evaluation = retry, score(retry, attempts)
        history.append(_attempt_row(attempts, retry, evaluation))
        if evaluation["quality_score"] > best_score["quality_score"]:
            best, best_score, adopted = retry, evaluation, attempts
        if evaluation["metrics"]["data_fit"].get("status") != direction:
            break  # the target lies between the two lambdas tried
    best_score = {**best_score, "attempts": attempts, "adopted_attempt": adopted,
                  "evaluation_history": history}
    if adopted > 1:
        best_score["adjusted_params"] = {"lam": best["inversion_params"]["lam"]}
    _needs_review(ctx, "Seismic inversion", best_score)
    return (quality.describe(best_score),
            {"seismic_results": {**best, "evaluation": best_score},
             "seismic_evaluation": best_score})


register(Tool(
    name="evaluate_seismic_inversion",
    description="Judge how well the velocity model fits the travel times, how much of it "
                "the rays cover and whether velocities sit at the inversion's bounds; "
                "re-invert with lambda changed while the fit is off. Run after the seismic "
                "inversion, before anything is built on the model.",
    handler=_evaluate_seismic,
    requires=("seismic_results",),
    produces=("seismic_evaluation",),
    agent="SeismicAgent",
    label="Evaluate seismic inversion",
    module="seismic",
))


def _velocity_thresholds(config: Mapping[str, Any]) -> List[float]:
    """The velocities to trace as interfaces: ``velocity_thresholds``, else the one threshold."""
    listed = config.get("velocity_thresholds")
    if isinstance(listed, (list, tuple)) and listed:
        return [float(v) for v in listed]
    return [float(config.get("velocity_threshold", 1200.0))]


def _extract_seismic_interfaces(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..seismic_agent import SeismicAgent
    from .._figstyle import style_from_config

    seismic = dict(ctx.get("seismic_results") or {})
    if seismic.get("mesh") is None or seismic.get("velocity_model") is None:
        raise ValueError("The seismic inversion returned no velocity model to trace "
                         "interfaces in.")
    thresholds = _velocity_thresholds(ctx.config)
    folder = str(seismic.get("output_dir") or Path(ctx.output_dir) / "seismic")
    agent = SeismicAgent(**agent_kwargs(ctx))
    velocity = np.asarray(seismic["velocity_model"], dtype=float)
    interfaces = agent.interfaces_from_model(seismic["mesh"], velocity, thresholds, folder)
    figure = agent._generate_velocity_plot(
        None, seismic["mesh"], velocity, seismic.get("coverage"),
        seismic.get("sensors") if seismic.get("sensors") is not None else [], interfaces,
        thresholds, folder,
        length_unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit,
        filename="seismic_structure.png",
        title="Seismic Refraction Tomography - Layer Interfaces")
    seismic.update({
        "interfaces": interfaces, "velocity_thresholds": thresholds,
        "structure_figure": figure,
        "data_paths": list(seismic.get("data_paths") or []) + [
            str(Path(folder) / f"interface_{threshold}ms.txt") for threshold in interfaces],
    })
    missing = [t for t in thresholds if t not in interfaces]
    if missing:
        # A velocity the model never reaches is a finding, not a failure: no
        # layer that fast lies within the depth the rays reach.
        ctx.note(f"No interface was traced at {', '.join(f'{t:g}' for t in missing)} m/s: "
                 f"the velocity model runs from {velocity.min():.0f} to {velocity.max():.0f} "
                 "m/s, and the contour does not cross the line.")
    traced = ", ".join(f"{t:g} m/s" for t in thresholds if t in interfaces)
    return ((f"Traced the {traced} interface{'s' if len(interfaces) > 1 else ''} through the "
             "velocity model." if interfaces else
             "No threshold velocity is crossed in the model, so no interface was traced."),
            {"seismic_results": seismic,
             "seismic_structure": {"thresholds": thresholds,
                                   "traced": [float(t) for t in interfaces]}})


register(Tool(
    name="extract_seismic_interfaces",
    description="Trace layer interfaces through the seismic velocity model at the "
                "threshold velocities (velocity_threshold, 1200 m/s by default), and draw "
                "the model with them.",
    handler=_extract_seismic_interfaces,
    # On the model the evaluation kept, which may be a retry's.
    requires=("seismic_results", "seismic_evaluation"),
    produces=("seismic_structure",),
    agent="SeismicAgent",
    label="Extract layer interfaces",
    module="seismic",
))


def _seismic_structure_pending(ctx: RunContext) -> bool:
    """A seismic model exists whose interfaces are still to be traced."""
    tool = TOOLS.get("extract_seismic_interfaces")
    return bool(ctx.has("seismic_results") and tool is not None and tool.available(ctx))


def _figure_style(ctx: RunContext) -> Optional[Dict[str, Any]]:
    """The figure style for a step that draws before any report is written.

    "In feet" in the request has to reach the figures now, and is recorded in
    the configuration so the report's tables agree with them.
    """
    from .._figures import length_unit_from_text

    style = dict(ctx.config.get("figure_style") or {})
    if not style.get("length_unit"):
        unit = length_unit_from_text(str(ctx.config.get("user_request") or ""))
        if unit:
            style["length_unit"] = unit
            ctx.config["figure_style"] = style
    return style or None


def _load_tdem(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Read the TDEM data the inversion will read, and draw a few soundings' decays."""
    from PyHydroGeophysX.data_processing.em1d import load_sounding

    from .. import _raw_data as raw
    from .._figstyle import style_from_config
    from ..tdem_agent import TDEMAgent

    config = ctx.config
    given = config.get("tdem_file") or config.get("em_file")
    path = resolve_path(given, config.get("project_dir", "."))
    if not path or not Path(str(path)).exists():
        raise ValueError(f"TDEM data not found: {given}")
    path, name = str(path), Path(str(path)).name
    if TDEMAgent._is_instrument_survey(path):
        # The moment the inversion reads: LM+HM, else HM for a one-moment survey.
        requested = config.get("tem_moment")
        moment = str(requested or "LM+HM")
        try:
            head = load_sounding(path, "TDEM", sounding=0, moment=moment)
        except ValueError:
            if requested:
                raise
            moment = "HM"
            head = load_sounding(path, "TDEM", sounding=0, moment=moment)
        total = int(head.get("n_soundings", 1))
        drawn = raw.chosen(total)
        soundings = [head if k == 0 else load_sounding(path, "TDEM", sounding=k, moment=moment)
                     for k in drawn]
        ids = list(np.asarray(head.get("station_ids", []), dtype=object).ravel())
        labels = [f"Station {ids[k]}" if k < len(ids) else f"Sounding {k + 1}" for k in drawn]
        layout = {"x": head.get("x", []), "y": head.get("y", []),
                  "line_numbers": head.get("line_numbers", []), "drawn": drawn}
        what = f"{total} soundings ({head.get('source_format') or 'TEM survey'}, moment {moment})"
    else:
        sounding = int((config.get("tdem_params") or {}).get("sounding", 0))
        times, observed, errors = TDEMAgent(**agent_kwargs(ctx))._load_tdem_data(path, sounding)
        relative = np.abs(errors) / np.maximum(np.abs(observed), 1e-30)
        soundings = [{"times": times, "response": observed, "relative_std": relative}]
        labels, layout, total, moment, drawn = [f"Sounding {sounding + 1}"], None, 1, None, [0]
        what = "one sounding"
    curves = [curve for sounding in soundings for curve in raw.decays(sounding)]
    if not curves:
        raise ValueError(f"{name} holds no TDEM decay to invert.")
    times = np.concatenate([curve[1] for curve in curves])
    try:
        figure = raw.tdem_decays(
            soundings, raw.figure_path(ctx.output_dir, "tdem_decays"),
            title=f"{name}: " + (f"{len(drawn)} of {total} soundings" if total > 1 else "decay"),
            labels=labels, layout=layout,
            unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit)
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        figure = None
    return (f"Read {what} from {name}; gates from {times.min() * 1e3:.3g} to "
            f"{times.max() * 1e3:.3g} ms" + (f", the decays of {len(drawn)} drawn"
                                             if total > 1 else "") + ".",
            {"tdem_data": {"source_file": path, "n_soundings": total, "moment": moment,
                           "figure": figure, "survey": layout is not None}})


register(Tool(
    name="load_tdem_data",
    description="Read the TDEM sounding or survey the configuration names and draw the "
                "decay curves of a few soundings. Run before the TDEM inversion.",
    handler=_load_tdem,
    produces=("tdem_data",),
    agent="TDEMAgent",
    label="Load TDEM data",
    module="em",
    when=_configured("tdem_file", "em_file"),
))


def _invert_tdem(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..tdem_agent import TDEMAgent

    config = ctx.config
    agent = TDEMAgent(**agent_kwargs(ctx))
    style = _figure_style(ctx)
    # TDEMAgent takes its settings flat, under 'data_file' - not an
    # 'inversion_params' dict and not 'tdem_file'. Passing the names this
    # workflow uses elsewhere silently gave it no data file at all.
    payload = {
        "mode": "inversion",
        "data_file": resolve_path(config.get("tdem_file") or config.get("em_file"),
                                  config.get("project_dir", ".")),
        "output_dir": str(Path(ctx.output_dir) / "tdem"),
        "figure_style": style,
    }
    if config.get("tem_moment"):
        payload["tem_moment"] = config["tem_moment"]
    payload.update({k: v for k, v in (config.get("tdem_params") or {}).items()})
    results = agent.execute(payload)
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "TDEM inversion failed."))
    loaded = ctx.get("tdem_data") or {}
    if loaded.get("figure"):
        results["raw_figures"] = [(loaded["figure"], (
            "TDEM decays as read, |response| against time after turn-off on log axes; "
            "shading is the stated error"
            + (", beside the station layout with the drawn soundings ringed"
               if loaded.get("survey") else "") + "."))]
    span = results.get("resistivity_range")
    detail = f" Resistivity {span[0]:.1f} to {span[1]:.1f} ohm-m." if span else ""
    if results.get("survey"):
        lines = results.get("lines") or []
        failed = int(results.get("failed_soundings") or 0)
        if failed:
            ctx.note(f"{failed} of {results['n_soundings']} TDEM soundings could not be "
                     "inverted and are blank in the section.")
        unresolved = int(results.get("unresolved_soundings") or 0)
        if unresolved:
            ctx.note(f"{unresolved} of {results['n_soundings']} TDEM soundings resolved no "
                     "layer above their depth of investigation and are blank in the section.")
        return (f"Inverted {results['n_soundings']} {results.get('instrument') or 'TEM'} "
                f"soundings on {len(lines)} line{'s' if len(lines) != 1 else ''} "
                f"({results.get('tem_moment')}, {results.get('lci_mode')} LCI) for a "
                f"{results.get('n_layers', '?')}-layer resistivity section; median sounding "
                f"chi-squared {results['chi2_sounding_median']:.2f}.{detail}",
                {"tdem_results": results})
    return (f"Recovered a {results.get('n_layers', '?')}-layer resistivity model "
            f"from the TDEM sounding.{detail}", {"tdem_results": results})


register(Tool(
    name="invert_tdem",
    description="Invert a time-domain electromagnetic sounding for a layered "
                "resistivity model.",
    handler=_invert_tdem,
    requires=("tdem_data",),
    produces=("tdem_results",),
    agent="TDEMAgent",
    label="Run TDEM inversion",
    module="em",
    when=_configured("tdem_file", "em_file"),
))


def _evaluate_tdem(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Score the TDEM model on its data fit, what it resolved, and its resistivities."""
    from .. import _method_evaluation as quality
    from .._figstyle import style_from_config

    tdem = dict(ctx.get("tdem_results") or {})
    folder = Path(tdem.get("output_dir") or Path(ctx.output_dir) / "tdem")
    evaluation = quality.evaluate_tdem(
        tdem, figure_path=folder / "tdem_data_fit.png",
        unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit,
        threshold=_quality_threshold(ctx))
    evaluation["attempts"] = 1
    _needs_review(ctx, "TDEM inversion", evaluation)
    return (quality.describe(evaluation),
            {"tdem_results": {**tdem, "evaluation": evaluation}, "tdem_evaluation": evaluation})


register(Tool(
    name="evaluate_tdem_inversion",
    description="Judge how well the TDEM model fits each sounding's decay, how many "
                "soundings resolved a model above their depth of investigation, and "
                "whether its resistivities are physical. Run after the TDEM inversion, "
                "before anything is built on the model.",
    handler=_evaluate_tdem,
    requires=("tdem_results",),
    produces=("tdem_evaluation",),
    agent="TDEMAgent",
    label="Evaluate TDEM inversion",
    module="em",
))


def _plan_maps_wanted(ctx: RunContext) -> bool:
    """The request asks where things are, and the TDEM result is a survey to map.

    A single sounding has no plan view. Before the inversion has run there is
    no result to look in, so on a projected route the step is shown whenever
    the request asks for the spatial distribution.
    """
    if not wants_spatial_map(ctx.config):
        return False
    if ctx.projected("tdem_results"):
        return True
    tdem = ctx.get("tdem_results") or {}
    return bool(tdem.get("survey")) and not tdem.get("map_figure")


def _map_tdem_plan_view(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from .._figstyle import style_from_config
    from ...visualization.axis_units import length_factor, normalize_length_unit
    from ...visualization.em_maps import draw_survey_plan_maps, survey_plan_grids

    config = ctx.config
    tdem = dict(ctx.get("tdem_results") or {})
    source = tdem.get("source_file") or resolve_path(
        config.get("tdem_file") or config.get("em_file"), config.get("project_dir", "."))
    # The basemap is the image the inputs name, else the georeferenced one the
    # survey folder keeps (a TEM2Go controller saves it under Maps), else tiles.
    maps = survey_plan_grids(tdem, source=source, basemap_file=config.get("basemap_file"),
                             basemap=str(config.get("basemap") or "auto"))
    style = style_from_config({"figure_style": _figure_style(ctx)})
    unit = normalize_length_unit(style.length_unit)
    drawn = draw_survey_plan_maps(maps, Path(ctx.output_dir) / "tdem", unit=unit,
                                  cmap=style.cmap_for("resistivity"),
                                  name=Path(str(source)).name if source else "")
    tdem.update(drawn)
    summary = drawn["depth_slices"]
    spread = [s["median"] for s in summary["std_decades"] if s]
    depths = ", ".join(f"{d * length_factor(unit):.3g}" for d in summary["depths"])
    return (f"Kriged resistivity at {len(summary['depths'])} depths ({depths} {unit}) across "
            "the survey outline, with a map of the kriging uncertainty for each"
            + (f"; median standard deviation {min(spread):.2f} to {max(spread):.2f} decade"
               if spread else "") + ".",
            {"tdem_results": tdem, "plan_maps": summary})


register(Tool(
    name="map_tdem_plan_view",
    description="Krige the TDEM survey's resistivity onto plan-view maps at several "
                "depths, filled across the survey outline, with a map of the kriging "
                "uncertainty for each, over the survey's basemap. Required when the "
                "request asks for the spatial distribution of a TEM survey.",
    handler=_map_tdem_plan_view,
    requires=("tdem_results", "tdem_evaluation"),
    produces=("plan_maps",),
    agent="TDEMAgent",
    label="Interpolate plan-view maps",
    module="em",
    when=_plan_maps_wanted,
))


def _tdem_water_content_wanted(ctx: RunContext) -> bool:
    """Water content is asked for, and the TDEM sounding is the run's resistivity model.

    With ERT data in the run the ERT model carries the conversion. A sounding
    on its own used to end the run "incomplete" with no reason given, although
    its layered model converts exactly as an ERT section does.
    """
    return _water_content_wanted(ctx) and not survey_files(ctx.config)


def _layered_water_content(ctx: RunContext, resistivity: Any, thicknesses: Any, *,
                           label: str, folder: Path, positions: Any = None,
                           max_depth: Optional[float] = None):
    """Water content of a layered resistivity model, layer by layer, by Monte Carlo.

    Shared by the TDEM sounding and the MT sites. Returns ``(mean, std, step,
    table)``: the per-layer mean and spread, the petrophysics step's result
    with the layering stated, and the CSV written (or None).

    A ``(n_stations, n_layers)`` resistivity is a section of 1D models on one
    layer grid - a TEM survey - and comes back in that shape, one table row per
    station and layer. Cells without a resistivity (NaN below a station's depth
    of investigation) are not converted and stay NaN.
    """
    from ..petrophysics_agent import PetrophysicsAgent

    config = ctx.config
    resistivity = np.asarray(resistivity, dtype=float)
    shape = resistivity.shape
    cells = resistivity.reshape(-1, shape[-1]) if resistivity.ndim == 2 else resistivity.reshape(1, -1)
    n_layers = cells.shape[1]
    usable = np.isfinite(cells) & (cells > 0)
    if not usable.any():
        raise ValueError(f"The {label} model has no resolved resistivity to convert.")
    thicknesses = np.asarray(thicknesses if thicknesses is not None else [], dtype=float).ravel()
    if thicknesses.size == n_layers - 1:
        # The last layer of a 1D model is the half-space below the others.
        top = np.concatenate([[0.0], np.cumsum(thicknesses)])
        bottom = np.concatenate([np.cumsum(thicknesses), [np.inf]])
    else:
        top = bottom = np.full(n_layers, np.nan)
    if config.get("layer_params"):
        ctx.note("Layer parameters given for " + ", ".join(map(str, config["layer_params"]))
                 + f" were not applied: the {label} model's layers are not divided into "
                   "geological units, so the conversion used "
                 + ("petrophysical_params." if config.get("petrophysical_params")
                    else "generated petrophysical parameters."))
    # The model's layers are one unit: a smooth 1D model draws no interface
    # between them, so there is no layer boundary to hang a second set on.
    step = PetrophysicsAgent(**agent_kwargs(ctx)).execute({
        "resistivity_model": cells[usable],
        "cell_markers": np.zeros(int(usable.sum()), dtype=int),
        "petrophysical_params": config.get("petrophysical_params", {}),
        "n_realizations": config.get("n_realizations", 100),
        "geological_context": config.get("geological_context", "generic watershed"),
        "output_dir": str(folder / "petrophysics"),
    })
    if step.get("status") != "success":
        raise ValueError(str(step.get("error") or f"The {label} water-content conversion failed."))
    step = {**step, "prior_flag": _state_prior(ctx, step)}
    mean_cells = np.full(cells.shape, np.nan)
    std_cells = np.full(cells.shape, np.nan)
    mean_cells[usable] = np.asarray(step.get("water_content_mean"), dtype=float).ravel()
    std_cells[usable] = np.asarray(step.get("water_content_std"), dtype=float).ravel()
    mean, std = mean_cells.reshape(shape), std_cells.reshape(shape)
    table: Optional[Path] = folder / "water_content_by_layer.csv"
    header = ("depth_top_m,depth_bottom_m,resistivity_ohm_m,"
              "water_content_mean,water_content_std")
    columns = [np.tile(top, len(cells)), np.tile(bottom, len(cells)), cells.ravel(),
               mean_cells.ravel(), std_cells.ravel()]
    if resistivity.ndim == 2:
        header = "station_index," + header
        columns.insert(0, np.repeat(np.arange(len(cells)), n_layers))
    try:
        table.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(table, np.column_stack(columns), delimiter=",", fmt="%.6g",
                   comments="", header=header)
    except Exception as exc:  # noqa: BLE001 - a table is not the result
        ctx.note(f"The {label} water-content table could not be written ({exc}).")
        table = None
    model = (f"{len(cells)}-station, {n_layers}-layer {label} section"
             if resistivity.ndim == 2 else f"{n_layers}-layer {label} model")
    try:
        from .._figstyle import style_from_config
        from .._uncertainty import draw_layered_water_content

        figure = draw_layered_water_content(
            mean, std, top, bottom, folder / "water_content_mean_and_uncertainty.png",
            title=f"{label} water content", positions=positions, max_depth=max_depth,
            unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit)
    except Exception:  # noqa: BLE001 - the numbers stand without the picture
        figure = None
    step = {**step, "figure": figure, "layering": (
        f"The {model} was converted as one unit with one "
        f"petrophysical parameter set; no interface divides its layers into "
        f"geological units."), "depth_top_m": top, "depth_bottom_m": bottom}
    return mean, std, step, table


def _layered_caption(section: bool) -> str:
    return ("Water content of each station's layered model (top) and its Monte Carlo "
            "standard deviation (bottom); blank below a station's depth of investigation."
            if section else
            "Water content of the layered model against depth, with bands of one and two "
            "Monte Carlo standard deviations; the last layer is the half-space.")


def _convert_tdem_water_content(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from .._uncertainty import result_caveats

    config = ctx.config
    tdem = dict(ctx.get("tdem_results") or {})
    resistivity = tdem.get("recovered_resistivity")
    if resistivity is None and tdem.get("recovered_conductivity") is not None:
        resistivity = 1.0 / np.asarray(tdem["recovered_conductivity"], dtype=float)
    if resistivity is None:
        raise ValueError("The TDEM inversion returned no layered resistivity model "
                         "to convert.")
    shape = np.shape(resistivity)
    model = (f"{shape[0]}-station, {shape[1]}-layer TDEM resistivity section"
             if len(shape) == 2 else f"{np.size(resistivity)}-layer TDEM resistivity model")
    mean, std, step, table = _layered_water_content(
        ctx, resistivity, tdem.get("thicknesses"), label="TDEM",
        folder=Path(ctx.output_dir) / "tdem", positions=tdem.get("positions"))
    tdem.update({"water_content_mean": mean, "water_content_std": std,
                 "water_content_table": str(table) if table else None})
    if step.get("figure"):
        tdem["water_content_figures"] = [(step["figure"], _layered_caption(len(shape) == 2))]
    for caveat in result_caveats(config, tdem):
        ctx.note(caveat)
    tdem.update({key: step.get(key) for key in ("petrophysical_relationship",
                                                 "prior_statement", "prior_ranges_text")})
    return (f"Converted the {model} to water content by "
            f"Monte Carlo petrophysics: {float(np.nanmin(mean)):.3f} to "
            f"{float(np.nanmax(mean)):.3f} (mean uncertainty ± {float(np.nanmean(std)):.3f})."
            f"{step.get('prior_flag', '')}",
            {"tdem_results": tdem, "water_content": [step]})


register(Tool(
    name="convert_tdem_water_content",
    description="Convert the TDEM sounding's layered resistivity model to volumetric "
                "water content, layer by layer, propagating petrophysical "
                "uncertainty by Monte Carlo. Required when the request asks about "
                "water content and the sounding is the run's only resistivity model.",
    handler=_convert_tdem_water_content,
    requires=("tdem_results", "tdem_evaluation"),
    produces=("water_content",),
    agent="PetrophysicsAgent",
    label="Convert TDEM model to water content",
    module="em",
    when=_tdem_water_content_wanted,
))


# ---------------------------------------------------------------------------
# magnetotellurics
# ---------------------------------------------------------------------------
def mt_files(config: Mapping[str, Any]) -> List[str]:
    """The MT sites the configuration names: transfer-function files or folders of them."""
    listed = config.get("mt_files") or config.get("mt_file") or []
    if isinstance(listed, (str, Path)):
        listed = [listed]
    return [str(item) for item in listed if item]


def _mt_sites(ctx: RunContext) -> List[Path]:
    """Every transfer-function file the configuration names, a folder standing for its files."""
    from PyHydroGeophysX.data_processing import mt

    sites: List[Path] = []
    for item in mt_files(ctx.config):
        path = Path(resolve_path(item, ctx.config.get("project_dir", ".")))
        if path.is_dir():
            sites += sorted({p for pattern in mt.TRANSFER_FUNCTION_PATTERNS for p in path.glob(pattern)
                             if mt.is_transfer_function_file(p)})
        elif path.is_file():
            sites.append(path)
        else:
            raise ValueError(f"MT site file not found: {item}")
    if not sites:
        raise ValueError("No MT transfer-function files (EDI, EMTF XML, Z- or J-files) were found in "
                         + ", ".join(mt_files(ctx.config)) + ".")
    return list(dict.fromkeys(sites))


def _mt_figure(tf: Any, model: Any, path: Path) -> str:
    """The site's sounding with the model's fit, beside the model, as one PNG."""
    from matplotlib.figure import Figure

    from PyHydroGeophysX.visualization import plot_mt_model_1d, plot_mt_sounding

    figure = Figure(figsize=(11, 6.5))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[3, 2])
    ax_rho = figure.add_subplot(grid[0, 0])
    ax_phase = figure.add_subplot(grid[1, 0], sharex=ax_rho)
    ax_model = figure.add_subplot(grid[:, 1])
    sounding = model.sounding
    mode = sounding.modes[0] if len(sounding.modes) == 1 else "xy"
    shift = model.static_shift.get(mode, 1.0)
    fit = {mode: (1 / sounding.frequency, 10 ** model.predicted["log_rho_a"] * shift,
                  model.predicted["phase"])}
    plot_mt_sounding(tf, components=("xy", "yx") + (("det",) if mode == "det" else ()),
                     axes=(ax_rho, ax_phase), predicted=fit, title=tf.station or path.stem)
    plot_mt_model_1d({f"Occam 1D (RMS {model.rms:.2f})": model}, ax=ax_model)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    return str(path)


def _site_folder(index: int, path: Path) -> str:
    return f"{index:02d}_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def _load_mt(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Read every MT site the configuration names, and draw their soundings."""
    from PyHydroGeophysX.data_processing.mt import read_transfer_function

    from .. import _raw_data as raw

    sites = _mt_sites(ctx)
    tfs = [read_transfer_function(path) for path in sites]
    names = [str(tf.station or path.stem) for tf, path in zip(tfs, sites)]
    drawn = raw.chosen(len(tfs))
    try:
        figure = raw.mt_soundings(
            [tfs[k] for k in drawn], raw.figure_path(ctx.output_dir, "mt_sites"),
            title=(f"{len(drawn)} of {len(tfs)} MT sites" if len(tfs) > len(drawn)
                   else "MT site" + ("s" if len(tfs) > 1 else "")) + ": apparent resistivity and phase")
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        figure = None
    periods = np.concatenate([np.asarray(tf.period, dtype=float).ravel() for tf in tfs])

    def located(tf: Any) -> bool:
        try:
            return bool(np.isfinite([float(tf.latitude), float(tf.longitude)]).all())
        except (TypeError, ValueError):
            return False

    where = sum(located(tf) for tf in tfs)
    return (f"Read {len(tfs)} MT site{'s' if len(tfs) != 1 else ''} ("
            + ", ".join(names[:6]) + (", ..." if len(names) > 6 else "")
            + f"); periods {periods.min():.3g} to {periods.max():.3g} s"
            + (f", {where} with coordinates" if len(tfs) > 1 else "") + ".",
            {"mt_data": {"sites": [str(path) for path in sites], "stations": names,
                         "figure": figure}})


register(Tool(
    name="load_mt_sites",
    description="Read the MT transfer functions the configuration names and draw each "
                "site's apparent resistivity and phase. Run before the MT inversion.",
    handler=_load_mt,
    produces=("mt_data",),
    agent="MT inversion (Occam 1D, SimPEG 2D)",
    label="Load MT sites",
    module="mt",
    when=_configured("mt_files", "mt_file"),
))


def _invert_mt(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from PyHydroGeophysX.workflows import ArtifactRef, WorkflowSpec, run_workflow
    from PyHydroGeophysX.workflows import RunContext as WorkflowContext

    sites = _mt_sites(ctx)
    parameters = dict(ctx.config.get("mt_params") or {})
    profile = parameters.pop("profile", None)
    out = Path(ctx.output_dir) / "mt"

    def site_ref(index: int, path: Path) -> ArtifactRef:
        return ArtifactRef.from_path(path, artifact_id=f"mt:site:{index}",
                                     kind="mt_transfer_function", base_dir=path.parent)

    entries: List[Dict[str, Any]] = []
    tfs: List[Any] = []
    for k, path in enumerate(sites):
        folder = out / _site_folder(k, path)
        spec = WorkflowSpec("mt.invert_1d", inputs={"transfer_function": site_ref(k, path)},
                            parameters=parameters)
        result = run_workflow(spec, WorkflowContext(project_root=path.parent, output_dir=folder))
        model, tf = result.objects["result"], result.objects["transfer_function"]
        tfs.append(tf)
        tops = np.asarray(model.depth, dtype=float)
        entries.append({
            "station": tf.station or path.stem, "path": str(path),
            "latitude": tf.latitude, "longitude": tf.longitude,
            "periods_s": [float(tf.period.min()), float(tf.period.max())],
            "rms": float(model.rms), "iterations": int(model.iterations),
            "static_shift": dict(model.static_shift),
            "depth_top_m": tops, "thicknesses": np.diff(tops),
            "resistivity_ohm_m": np.asarray(model.resistivity, dtype=float),
            "model_csv": str(folder / "mt1d_model.csv"), "fit_csv": str(folder / "mt1d_fit.csv"),
            "figure": _mt_figure(tf, model, folder / "mt1d.png"),
        })
    results: Dict[str, Any] = {"sites": entries, "output_dir": str(out),
                               "figures": [entry["figure"] for entry in entries]}
    loaded = ctx.get("mt_data") or {}
    if loaded.get("figure"):
        results["raw_figures"] = [(loaded["figure"], "Apparent resistivity and phase of the "
                                                     "MT sites as read, against period; the "
                                                     "Zyx phase is folded by 180 degrees.")]
    rms = [entry["rms"] for entry in entries]
    summary = (f"Inverted {len(entries)} MT site{'s' if len(entries) > 1 else ''} in 1D (Occam), "
               f"RMS {min(rms):.2f}" + (f" to {max(rms):.2f}" if len(rms) > 1 else "") + ".")
    located = all(np.isfinite([e["latitude"], e["longitude"]]).all() for e in entries)
    if profile or (profile is None and len(entries) >= 3 and located):
        from matplotlib.figure import Figure

        from PyHydroGeophysX.visualization import plot_mt_section

        # A section along a line of sites; few frequencies keep an unattended run bounded.
        frequency = np.sort(np.asarray(tfs[0].frequency, dtype=float))[::-1]
        index = np.unique(np.round(np.linspace(0, frequency.size - 1,
                                               min(12, frequency.size))).astype(int))
        spec = WorkflowSpec(
            "mt.invert_profile",
            inputs={"transfer_functions": [site_ref(k, path) for k, path in enumerate(sites)]},
            parameters={"frequencies": [float(f) for f in frequency[index]], "max_iterations": 10,
                        **dict(ctx.config.get("mt_profile_params") or {})},
            seed=0)
        try:
            section = run_workflow(spec, WorkflowContext(project_root=sites[0].parent,
                                                         output_dir=out / "profile")).objects["result"]
        except Exception as exc:  # noqa: BLE001 - the 1D models stand on their own
            ctx.note(f"The 2D MT profile could not be inverted ({plain_error(exc)}); "
                     "the sites are reported as 1D models.")
        else:
            figure = Figure(figsize=(10, 4.5))
            plot_mt_section(section, ax=figure.add_subplot(111),
                            title=f"2D MT section (RMS {section.rms:.2f})")
            figure.savefig(out / "profile" / "mt2d_section.png", dpi=150)
            results["profile"] = {"rms": float(section.rms), "iterations": len(section.history),
                                  "section_npz": str(out / "profile" / "mt2d_section.npz"),
                                  "figure": str(out / "profile" / "mt2d_section.png")}
            results["figures"].append(results["profile"]["figure"])
            summary += (f" A 2D TE/TM section along the {len(entries)} sites fits to "
                        f"RMS {section.rms:.2f}.")
    return summary, {"mt_results": results}


register(Tool(
    name="invert_mt",
    description="Invert magnetotelluric sites (EDI, EMTF XML, Z- or J-files) for "
                "layered resistivity by Occam 1D, and a line of three or more "
                "located sites for a 2D TE/TM section.",
    handler=_invert_mt,
    requires=("mt_data",),
    produces=("mt_results",),
    agent="MT inversion (Occam 1D, SimPEG 2D)",
    label="Run MT inversion",
    module="mt",
    when=_configured("mt_files", "mt_file"),
))


def _evaluate_mt(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Score the MT models on each site's fit, the static shifts and their resistivities."""
    from .. import _method_evaluation as quality

    mt_results = dict(ctx.get("mt_results") or {})
    folder = Path(mt_results.get("output_dir") or Path(ctx.output_dir) / "mt")
    evaluation = quality.evaluate_mt(mt_results, figure_path=folder / "mt_data_fit.png",
                                     threshold=_quality_threshold(ctx))
    evaluation["attempts"] = 1
    _needs_review(ctx, "MT inversion", evaluation)
    return (quality.describe(evaluation),
            {"mt_results": {**mt_results, "evaluation": evaluation}, "mt_evaluation": evaluation})


register(Tool(
    name="evaluate_mt_inversion",
    description="Judge how well each MT site's model fits its data against Occam's target "
                "RMS, how large the static shifts are, and whether the resistivities are "
                "physical. Run after the MT inversion, before anything is built on it.",
    handler=_evaluate_mt,
    requires=("mt_results",),
    produces=("mt_evaluation",),
    agent="MT inversion (Occam 1D, SimPEG 2D)",
    label="Evaluate MT inversion",
    module="mt",
))


def _mt_water_content_wanted(ctx: RunContext) -> bool:
    """Water content is asked for, and the MT sites hold the run's resistivity models."""
    config = ctx.config
    return (_water_content_wanted(ctx) and not survey_files(config)
            and not (config.get("tdem_file") or config.get("em_file")))


def _mt_sensitivity_depth(entry: Mapping[str, Any]) -> Optional[float]:
    """How deep a site's model is worth drawing: the MT report's sensitivity depth."""
    from .._mt_report import sensitivity_depth

    try:
        return sensitivity_depth(entry)
    except Exception:  # noqa: BLE001 - without it the whole model is drawn
        return None


def _convert_mt_water_content(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    mt_results = dict(ctx.get("mt_results") or {})
    entries = [dict(entry) for entry in mt_results.get("sites") or []]
    if not entries:
        raise ValueError("The MT inversion returned no layered model to convert.")
    steps, ranges = [], []
    for entry in entries:
        mean, std, step, table = _layered_water_content(
            ctx, entry["resistivity_ohm_m"], entry["thicknesses"],
            label=f"MT ({entry['station']})", folder=Path(entry["model_csv"]).parent,
            max_depth=_mt_sensitivity_depth(entry))
        entry.update({"water_content_mean": mean, "water_content_std": std,
                      "water_content_table": str(table) if table else None})
        if step.get("figure"):
            mt_results.setdefault("water_content_figures", []).append(
                (step["figure"], f"{entry['station']}: " + _layered_caption(False)))
        steps.append(step)
        ranges.append(f"{entry['station']} {float(np.nanmin(mean)):.3f}-{float(np.nanmax(mean)):.3f}"
                      f" (± {float(np.nanmean(std)):.3f})")
    mt_results["sites"] = entries
    mt_results.update({key: steps[0].get(key) for key in (
        "petrophysical_relationship", "prior_statement", "prior_ranges_text")})
    return ("Converted the MT sites' layered models to water content by Monte Carlo "
            "petrophysics: " + "; ".join(ranges) + "." + (steps[0].get("prior_flag") or ""),
            {"mt_results": mt_results, "water_content": steps})


register(Tool(
    name="convert_mt_water_content",
    description="Convert each MT site's layered resistivity model to volumetric water "
                "content, layer by layer, propagating petrophysical uncertainty by "
                "Monte Carlo. Required when the request asks about water content and "
                "the MT sites are the run's only resistivity models.",
    handler=_convert_mt_water_content,
    requires=("mt_results", "mt_evaluation"),
    produces=("water_content",),
    agent="PetrophysicsAgent",
    label="Convert MT models to water content",
    module="mt",
    when=_mt_water_content_wanted,
))


def _load_gravmag(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Read the station table, say which field it holds, and map the stations."""
    from PyHydroGeophysX.data_processing import table_io

    from .. import _raw_data as raw
    from .._figstyle import style_from_config
    from ..gravmag_agent import resolve_kind

    config = ctx.config
    path = resolve_path(config.get("gravmag_file"), config.get("project_dir", "."))
    if not path or not Path(str(path)).is_file():
        raise ValueError(f"Gravity / magnetic station file not found: {config.get('gravmag_file')}")
    table = table_io.load_xyz_table(str(path), min_cols=3)
    header = table_io.table_header(str(path)) or []
    kind = resolve_kind(path, header, config.get("gravmag_kind"),
                        str(config.get("user_request") or ""))
    unit = "mGal" if kind == "gravity" else "nT"
    x, y, value = (np.asarray(table[:, i], dtype=float) for i in range(3))
    name = Path(str(path)).name
    field = "gravity" if kind == "gravity" else "magnetic"
    try:
        figure = raw.gravmag_stations(
            x, y, value, raw.figure_path(ctx.output_dir, "gravmag_stations"),
            title=f"{name}: {x.size} {field} stations as read", label=f"Observed ({unit})",
            unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit)
    except Exception:  # noqa: BLE001 - a picture of the data must not fail the load
        figure = None
    return (f"Read {x.size} {field} stations from {name}: {np.nanmin(value):.4g} to "
            f"{np.nanmax(value):.4g} {unit} over {np.ptp(x):.0f} x {np.ptp(y):.0f} m"
            + ("" if table.shape[1] >= 4 else "; the table gives no station elevations") + ".",
            {"gravmag_data": {"file": str(path), "kind": kind, "unit": unit,
                              "n_stations": int(x.size), "figure": figure}})


register(Tool(
    name="load_gravmag_data",
    description="Read the gravity or magnetic station table the configuration names, say "
                "which field it holds, and map the stations' values. Run before the "
                "gravity / magnetic processing.",
    handler=_load_gravmag,
    produces=("gravmag_data",),
    agent="GravMagAgent",
    label="Load gravity / magnetic data",
    module="gravmag",
    when=_configured("gravmag_file"),
))


def _invert_gravmag(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..gravmag_agent import GravMagAgent

    config = ctx.config
    results = GravMagAgent(**agent_kwargs(ctx)).execute({
        "data_file": resolve_path(config.get("gravmag_file"), config.get("project_dir", ".")),
        "output_dir": str(Path(ctx.output_dir) / "gravmag"),
        "kind": config.get("gravmag_kind"),
        "field": config.get("magnetic_field"),
        "user_request": config.get("user_request", ""),
        "figure_style": _figure_style(ctx),
        **dict(config.get("gravmag_params") or {}),
    })
    if results.get("status") != "success":
        raise ValueError(str(results.get("error") or "Gravity / magnetic processing failed."))
    for note in results.get("assumptions") or []:
        ctx.note(note)
    loaded = ctx.get("gravmag_data") or {}
    if loaded.get("figure"):
        results["raw_figures"] = [(loaded["figure"], "The station values as read, before the "
                                                     "regional trend was removed.")]
    field = "gravity" if results["kind"] == "gravity" else "magnetic"
    summary = (f"Separated the regional trend from the {field} anomaly at "
               f"{results['n_stations']} stations")
    inversion = results.get("inversion")
    if inversion:
        quantity = "density contrast" if field == "gravity" else "susceptibility"
        low, high = inversion["model_range"]
        summary += (f" and inverted the residual for a 3D {quantity} model "
                    f"({low:.3g} to {high:.3g}; chi-squared {inversion['chi2']:.3g}).")
    else:
        ctx.note(f"The 3D {field} inversion did not run ({results.get('inversion_error')}); "
                 "the report gives the QC maps only.")
        summary += "; the 3D inversion did not run."
    return summary, {"gravmag_results": results}


register(Tool(
    name="invert_gravmag",
    description="Process gravity or magnetic station data: separate the regional trend "
                "from the residual anomaly, map both, and invert the residual for a 3D "
                "density-contrast or susceptibility model.",
    handler=_invert_gravmag,
    requires=("gravmag_data",),
    produces=("gravmag_results",),
    agent="GravMagAgent",
    label="Run gravity / magnetic inversion",
    module="gravmag",
    when=_configured("gravmag_file"),
))


def _gravmag_inverted(ctx: RunContext) -> bool:
    """A 3D model exists to evaluate (or, on a projected route, is coming)."""
    if ctx.projected("gravmag_results"):
        return True
    return bool((ctx.get("gravmag_results") or {}).get("inversion"))


def _evaluate_gravmag(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    """Score the 3D model on its fit, its bounds and the beta search."""
    from .. import _method_evaluation as quality
    from .._figstyle import style_from_config

    results = dict(ctx.get("gravmag_results") or {})
    folder = Path(results.get("output_dir") or Path(ctx.output_dir) / "gravmag")
    evaluation = quality.evaluate_gravmag(
        results, figure_path=folder / "gravmag_data_fit.png",
        unit=style_from_config({"figure_style": _figure_style(ctx)}).length_unit,
        threshold=_quality_threshold(ctx))
    evaluation["attempts"] = 1
    field = "Gravity" if results.get("kind") == "gravity" else "Magnetic"
    _needs_review(ctx, f"{field} inversion", evaluation)
    return (quality.describe(evaluation),
            {"gravmag_results": {**results, "evaluation": evaluation},
             "gravmag_evaluation": evaluation})


register(Tool(
    name="evaluate_gravmag_inversion",
    description="Judge how well the 3D density-contrast or susceptibility model explains "
                "the residual anomaly, whether it reaches the solver's bounds, and whether "
                "the beta search landed on its target. Run after the gravity / magnetic "
                "inversion.",
    handler=_evaluate_gravmag,
    requires=("gravmag_results",),
    produces=("gravmag_evaluation",),
    agent="GravMagAgent",
    label="Evaluate gravity / magnetic inversion",
    module="gravmag",
    when=_gravmag_inverted,
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
    summary = (f"Re-inverted the first ERT survey with the {_speed(threshold)}"
               f"velocity interface built into the mesh.")
    outputs: Dict[str, Any] = {"structure_results": results}
    if len(surveys) == 1:
        # The constrained model is the run's model from here on. Kept only as
        # structure_results, nothing read it: water content, the report and
        # the exported bundle all described the unconstrained model.
        adopted = _constrained_model(results, threshold)
        outputs["unconstrained_inversion_results"] = ctx.get("inversion_results")
        outputs["inversion_results"] = adopted
        outputs.update(_reexport(ctx, adopted, "the structure-constrained model"))
        if ctx.has("evaluation_results"):
            # The score on record is the unconstrained model's; the report would
            # print it under the constrained one.
            outputs["evaluation_results"] = _score_constrained(ctx, adopted)
        summary += (" That model replaces the unconstrained one for the water "
                    "content, the report and the exported model.")
    else:
        ctx.note("The structure-constrained model covers the first survey only: the "
                 "time-lapse models, and anything converted from them, are not "
                 "constrained by the seismic interface.")
    return summary, outputs


def _speed(threshold: Any) -> str:
    """``"1200 m/s "`` for a sentence, or nothing when no threshold is known.

    A parsed configuration carries ``null`` for a setting the request did not
    mention, and a summary must not fail after the inversion it describes ran.
    """
    try:
        return f"{float(threshold):g} m/s "
    except (TypeError, ValueError):
        return ""


def _constrained_model(structure: Dict[str, Any], threshold: Any) -> Dict[str, Any]:
    """A structure-constrained inversion in the shape the rest of the run reads."""
    return {
        "status": "success",
        "mesh": structure.get("mesh"),
        "resistivity_model": structure.get("resistivity_model"),
        "coverage": structure.get("coverage"),
        "chi2": structure.get("chi2"),
        # Layer markers on the parameter cells: above and below the interface.
        "cell_markers": structure.get("cell_markers"),
        "inversion_params": dict(structure.get("inversion_params") or {}),
        "interpretation": structure.get("interpretation"),
        "structure_constrained": True,
        "inversion_method": (f"Structure-constrained inversion: the {_speed(threshold)}"
                             f"seismic velocity interface is built into the mesh as a "
                             f"boundary, so the smoothing does not act across it"),
    }


def _score_constrained(ctx: RunContext, adopted: Dict[str, Any]) -> Dict[str, Any]:
    """The quality evaluation of ``adopted`` alone, without any retry."""
    from ..inversion_evaluation_agent import InversionEvaluationAgent

    evaluation = InversionEvaluationAgent(**agent_kwargs(ctx)).execute({
        "inversion_results": adopted,
        "inversion_params": adopted.get("inversion_params") or {},
        # A retry re-inverts without the interface; see _evaluate.
        "auto_adjust": False,
        "max_attempts": 1,
        "quality_threshold": ctx.config.get("quality_threshold", 70),
    })
    stripped = {k: v for k, v in evaluation.items() if k != "final_results"}
    adopted["evaluation_results"] = stripped
    return stripped


register(Tool(
    name="derive_structure",
    description="Turn a seismic velocity model into layer interfaces that can "
                "constrain an ERT inversion or a petrophysical conversion.",
    handler=_derive_structure,
    # After the unconstrained inversion, not only after the seismic one: the
    # constrained model replaces it, and an inversion run afterwards would
    # replace the constrained model in turn.
    requires=("seismic_results", "seismic_structure", "ert_data", "inversion_results"),
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
    # DataFusionAgent.execute only drafts an execution plan. The summary used to
    # announce "Combined seismic, ert, petrophysics" on the strength of that
    # plan, so it now reports what the other steps of this run combined.
    combined, apart = fusion_account(ctx)
    results = {**results, "combined": combined, "not_combined": apart,
               "planned_only": not combined}
    if combined:
        summary = f"Fused {', '.join(methods)} ({pattern}): {'; '.join(combined)}."
        if apart:
            summary += f" Not combined: {'; '.join(apart)}."
    else:
        summary = (f"Only planned the {pattern} pattern for {', '.join(methods)}; "
                   f"nothing was combined"
                   + (f": {'; '.join(apart)}." if apart else "."))
        ctx.note(f"Data fusion was planned ({pattern}) but no results were "
                 f"combined; each method is reported on its own.")
    return summary, {"fusion_results": results}


def fusion_account(ctx: RunContext) -> Tuple[List[str], List[str]]:
    """What this run actually combined across methods, and what it did not.

    Parameters
    ----------
    ctx : RunContext
        The run, read for the artifacts the other steps left.

    Returns
    -------
    tuple
        ``(combined, not_combined)``, one clause per item, for the fusion
        step's summary.

    Raises
    ------
    None

    Examples
    --------
    >>> ctx = RunContext('x')
    >>> ctx.put('seismic_results', {'status': 'success'})
    >>> ctx.put('inversion_results', {'status': 'success'})
    >>> fusion_account(ctx)
    ([], ['the seismic model did not constrain the ERT inversion (no structural constraint was derived)'])
    """
    combined: List[str] = []
    apart: List[str] = []
    inversion = ctx.get("inversion_results") or {}
    structure = ctx.get("structure_results") or {}
    constrained = bool(inversion.get("structure_constrained"))
    if constrained:
        threshold = structure.get("velocity_threshold",
                                  ctx.config.get("velocity_threshold", 1200.0))
        combined.append(f"the {_speed(threshold)}seismic velocity interface "
                        f"constrained the ERT inversion")
    elif structure:
        apart.append("a structure-constrained model was computed for the first "
                     "survey only; the time-lapse models were not constrained"
                     if len(ctx.get("ert_data") or []) > 1 else
                     "a structure-constrained model was computed but is not the "
                     "model this run reports")
    elif ctx.has("seismic_results") and ctx.has("inversion_results"):
        apart.append("the seismic model did not constrain the ERT inversion "
                     "(no structural constraint was derived)")
    steps = ctx.get("water_content") or []
    if steps:
        source = ("the structure-constrained model" if constrained
                  else "the resistivity model(s)")
        layering = str((steps[0] or {}).get("layering") or "").strip()
        combined.append(f"water content was converted from {source}"
                        + (f" ({layering.rstrip('.')})" if layering else ""))
    for key, clause in (("tdem_results", "the TDEM sounding was inverted on its own"),
                        ("mt_results", "the MT sites were inverted on their own"),
                        ("gravmag_results", "the gravity / magnetic survey was inverted on "
                                            "its own")):
        if ctx.has(key):
            apart.append(f"{clause} and not combined with the other methods")
    return combined, apart


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
    # never shown a step that can only fail - and not while a structural
    # constraint is still to come, since it reports what was combined.
    when=lambda ctx: (fusion_pattern(available_methods(ctx)) is not None
                      and not _structure_pending(ctx)
                      and not _seismic_structure_pending(ctx)),
))


# ---------------------------------------------------------------------------
# 7. the report, last
# ---------------------------------------------------------------------------
def _survey_report_input(ctx: RunContext, results: Dict[str, Any]) -> Dict[str, Any]:
    """What ``ReportAgent.execute`` reads for a single survey.

    It takes the configuration under ``config`` and the step outputs nested by
    step - ``inversion_results``, ``water_content``, ``ert_data`` - the shape the
    pre-controller pipeline assembled. Handing it the flat inversion dict and
    ``workflow_config`` instead gave a report with no request, no inversion
    statistics, no water content and no figures, while the step still said it
    had written them.
    """
    config = ctx.config
    failure = _water_content_failure(ctx, results)
    workflow_data: Dict[str, Any] = {
        "inversion_results": results,
        "evaluation_results": ctx.get("evaluation_results") or {},
        "raw_figures": ctx.get("ert_raw_figures") or [],
        # "Not requested" only when it was not: a conversion that was asked for
        # and failed used to be reported as never requested.
        "skip_petrophysics": (results.get("water_content_mean") is None
                              and not _water_content_wanted(ctx)),
    }
    if failure:
        workflow_data["water_content_failed"] = failure
    surveys = ctx.get("ert_data") or []
    if surveys:
        electrodes = len(getattr(surveys[0], "electrodes", None) or [])
        readings = len(getattr(surveys[0], "observations", None) or [])
        workflow_data["ert_data"] = {
            "n_electrodes": electrodes, "num_electrodes": electrodes,
            "n_measurements": readings, "num_measurements": readings,
            "instrument": config.get("instrument", DEFAULT_INSTRUMENT),
        }
    if results.get("water_content_mean") is not None:
        step = (ctx.get("water_content") or [{}])[0] or {}
        workflow_data["water_content"] = {
            "mesh": results.get("mesh"),
            "water_content_mean": results.get("water_content_mean"),
            "water_content_std": results.get("water_content_std"),
            "layer_params_used": step.get("layer_params_used", {}),
            "layer_params": step.get("layer_params", {}),
            "petrophysical_params": config.get("petrophysical_params", {}),
            "layer_params_applied": results.get("layer_params_applied") or {},
            "layer_params_not_applied": results.get("layer_params_not_applied") or {},
            "n_realizations": step.get("n_realizations") or config.get("n_realizations", 100),
            "interpretation": step.get("interpretation"),
            "petrophysical_relationship": step.get("petrophysical_relationship"),
            "prior_statement": step.get("prior_statement"),
            "prior_ranges_text": step.get("prior_ranges_text"),
        }
        workflow_data["petrophysics_results"] = step
        workflow_data["petrophysical_params"] = config.get("petrophysical_params", {})
    climate = ctx.get("climate_data")
    if isinstance(climate, Mapping):
        workflow_data["climate_data"] = climate
    if results.get("structure_constrained"):
        # The report's seismic section and its "Seismic Integration" line read
        # this; without it a constrained run was reported as ERT alone.
        structure = ctx.get("structure_results") or {}
        # An interpretation nobody wrote is left out, not printed as "N/A".
        workflow_data["seismic_structure"] = {
            "velocity_threshold": structure.get("velocity_threshold"),
            "interpretation": (ctx.get("seismic_results") or {}).get("interpretation"),
        }
    # The other methods of the run, each given its own section.
    for key in _SURVEY_RESULTS:
        if ctx.has(key):
            workflow_data[key] = ctx.get(key)
    delivered = {**results, "climate_data": ctx.get("climate_data")}
    workflow_data["not_delivered"] = not_delivered_items(ctx, delivered)
    return {"workflow_data": workflow_data, "config": config,
            "output_dir": str(ctx.output_dir)}


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
    failure = _water_content_failure(ctx, results)
    # Climate data are a step output of their own, not part of the inversion
    # results; without them here every run that retrieved them was told it had not.
    delivered = {**results, "climate_data": ctx.get("climate_data")}
    payload = {
        "inversion_results": ({**results, "water_content_failed": failure} if failure
                              else results),
        "not_delivered": not_delivered_items(ctx, delivered),
        "climate_data": ctx.get("climate_data"),
        "site_info": site_info,
        "comparison_data": ctx.get("comparison_data"),
        "raw_figures": ctx.get("ert_raw_figures") or [],
        "evaluation_results": ctx.get("evaluation_results"),
        "workflow_config": config,
        "time_lapse_method": config.get("time_lapse_method"),
        **{key: ctx.get(key) for key in _SURVEY_RESULTS if ctx.has(key)},
        "output_dir": str(ctx.output_dir),
    }
    report = (agent.generate_timelapse_report(payload) if time_lapse
              else agent.execute(_survey_report_input(ctx, results)))
    if report.get("status") == "failed":
        raise ValueError(str(report.get("error") or "Report generation failed."))

    for warning in report.get("warnings") or []:
        ctx.note(warning)
    # Products the request named and the run did not deliver, stated here rather
    # than left for the reader to notice, with the reason each is missing.
    for warning in (unmet_requests(config, delivered, shortfall_reasons(ctx, delivered))
                    + result_caveats(config, results)):
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


#: The ERT steps whose model, when it comes, makes ``write_report`` the run's report.
_ERT_MODEL_STEPS = ("load_ert_surveys", "invert_ert", "invert_time_lapse")


#: The artifacts of the methods a survey report covers, as the report modules name them.
_SURVEY_RESULTS = ("tdem_results", "seismic_results", "mt_results", "gravmag_results")

#: Steps whose products the survey report describes, so it waits for them.
_SURVEY_STEPS = ("load_tdem_data", "invert_tdem", "evaluate_tdem_inversion",
                 "map_tdem_plan_view", "pick_first_breaks", "load_seismic_traveltimes",
                 "invert_seismic", "evaluate_seismic_inversion", "extract_seismic_interfaces",
                 "load_mt_sites", "invert_mt", "evaluate_mt_inversion", "load_gravmag_data",
                 "invert_gravmag", "evaluate_gravmag_inversion",
                 "convert_tdem_water_content", "convert_mt_water_content", "fetch_climate")


def _survey_report_due(ctx: RunContext) -> bool:
    """The survey report is this run's report, and what it describes exists.

    A run whose models came only from TDEM, seismic refraction, MT or gravity
    and magnetics used to end without a report, "success" and all, because
    ``write_report`` needs an ERT inversion. This report takes its place when
    no ERT model exists or is still coming - an ERT step that failed leaves the
    other surveys to be reported - once every method step and water-content
    conversion the run can still take has been tried.
    """
    if ctx.has("inversion_results") or ctx.has("report_files"):
        return False
    if not any(ctx.has(key) for key in _SURVEY_RESULTS):
        return False
    return not any(TOOLS[name].available(ctx) for name in _ERT_MODEL_STEPS + _SURVEY_STEPS
                   if name in TOOLS)


def _write_survey_report(ctx: RunContext) -> Tuple[str, Dict[str, Any]]:
    from ..report_agent import ReportAgent
    from .._intent import unmet_requests
    from .._uncertainty import result_caveats

    config = ctx.config
    results = {key: dict(ctx.get(key)) for key in _SURVEY_RESULTS if ctx.get(key)}
    steps = ctx.get("water_content") or []
    delivered = {**(results.get("tdem_results") or {}),
                 "water_content": steps or None, "climate_data": ctx.get("climate_data")}
    report = ReportAgent(**agent_kwargs(ctx)).generate_survey_report({
        **results,
        "water_content": steps,
        "water_content_failed": _water_content_failure(ctx, delivered),
        "not_delivered": not_delivered_items(ctx, delivered),
        "site_info": dict(config.get("site_info") or {}),
        "workflow_config": config,
        "output_dir": str(ctx.output_dir),
    })
    if report.get("status") == "failed":
        raise ValueError(str(report.get("error") or "Survey report generation failed."))
    for warning in unmet_requests(config, delivered, shortfall_reasons(ctx, delivered)):
        ctx.note(warning)
    if results.get("tdem_results"):
        for warning in result_caveats(config, results["tdem_results"]):
            ctx.note(warning)
    files = {"report_markdown": report.get("report_file"), "report_pdf": report.get("pdf_file")}
    for name, path in (report.get("visualization_files") or {}).items():
        files[f"visualization_{name}"] = path
    return (f"Wrote the report ({Path(report['report_file']).name}) and its figures.",
            {"report_files": {k: v for k, v in files.items() if v},
             "interpretation": report.get("executive_summary")})


register(Tool(
    name="write_survey_report",
    description="Write the final report of a run without ERT, from the TDEM, seismic, MT "
                "and gravity/magnetic results it produced: data, method, fit, the models, "
                "water content if converted, figures, recommendations. Run this last.",
    handler=_write_survey_report,
    produces=("report_files",),
    agent="ReportAgent",
    label="Generate report",
    module="one_click",
    when=_survey_report_due,
))


def build_site_info(ctx: RunContext) -> Dict[str, Any]:
    """Site description for the report, from the configuration and file names."""
    from ..base_agent import _dates_from_filenames

    config = ctx.config
    configured = dict(config.get("site_info") or {})
    # The surveys that loaded, so each date labels the survey it belongs to.
    stamps = _dates_from_filenames(ctx.get("ert_files") or survey_files(config))
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


def summarise_run(ctx: RunContext, incomplete: Sequence[str] = ()) -> str:
    """The interpretation text, written from the steps that actually ran.

    Replaces an f-string that restated the configuration - it reported the
    configured time-lapse method and the configured regularization whether or
    not the inversion had used them, because it was written beside the config
    rather than beside the result.

    ``incomplete`` - why the run cannot be called complete, when it cannot -
    heads the text in place of "completed", so its first line never claims a
    run finished that did not.
    """
    results = ctx.get("inversion_results") or {}
    if incomplete:
        lines = [f"Workflow did not complete ({ctx.elapsed():.0f} s):"]
        lines += [f"- {reason}" for reason in incomplete] + [""]
    else:
        lines = [f"Workflow completed in {ctx.elapsed():.0f} s.", ""]
    lines.append("**Steps taken:**")
    lines += [f"- {step.line()}" for step in ctx.steps] or ["- none"]
    history = chi2_summary(results.get("chi2_values"))
    if history != "N/A":
        lines += ["", f"**Data fit:** chi-squared {history}."]
    raised = [w for w in ctx.warnings if w not in incomplete]
    if raised:
        lines += ["", "**Raised during the run:**"]
        lines += [f"- {w}" for w in raised]
    return "\n".join(lines)
