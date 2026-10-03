"""
Agent Coordinator for Multi-Agent Workflow

Coordinates the execution of multiple specialized agents to complete
the full geophysical processing workflow. Supports cross-modal geophysical
data processing (ERT, seismic, and more) with multiple LLM API providers
(GPT, Gemini, Claude).
"""

import json
import os
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import AgentResult, _dates_from_filenames
from ._intent import (PRODUCTS, names_tdem, names_unnegated, wants_climate,
                      wants_water_content)
from ._method import IMPLEMENTED_SCHEME
from ._pricing import estimate_llm_cost_usd, estimate_tokens


#: Inputs :meth:`AgentCoordinator.execute_workflow` has no step for, with the
#: workflow each one names. The runtime controller behind
#: ``BaseAgent.run_unified_agent_workflow()`` runs all of them.
_UNSUPPORTED_INPUTS = {
    "tdem_file": "TDEM",
    "em_file": "TDEM",
    "raw_seismic_file": "raw SEG-Y seismic processing",
    "seismic_file": "seismic travel-time inversion",
    "mt_files": "magnetotelluric inversion",
    "mt_file": "magnetotelluric inversion",
}

#: The keys that name the ERT data of a run.
_ERT_INPUTS = ("time_lapse_files", "timelapse_files", "data_file", "ert_file")

#: How a request asks for a time-lapse inversion.
_TIME_LAPSE_TERMS = ("time-lapse", "timelapse", "time lapse", "时移")


def _unsupported_by_coordinator(config: Dict[str, Any]) -> Optional[str]:
    """Why the coordinator cannot run *config*, or None when it can.

    Its pipeline is ERT (one survey or a time-lapse series) with optional
    climate data, a seismic structure constraint read from ``seismic_data``,
    and water content. The preview used to plan TDEM, SEG-Y and seismic-fusion
    workflows as well, which the run then skipped: it ran the ERT steps alone,
    or failed for want of an ERT file. The preview and the run now both refuse
    such a configuration. A request that rules TDEM out ("not TDEM", "不需要
    TDEM") does not count as asking for it; one that asks for a TEM sounding
    does.

    Examples
    --------
    >>> _unsupported_by_coordinator({"data_file": "a.ohm"}) is None
    True
    >>> print(_unsupported_by_coordinator({"tdem_file": "s.csv"}).split(";")[0])
    AgentCoordinator has no step for TDEM (tdem_file)
    >>> _unsupported_by_coordinator({"data_file": "a.ohm",
    ...                              "user_request": "ERT only, not TDEM"}) is None
    True
    >>> print(_unsupported_by_coordinator({"data_file": "a.ohm",
    ...       "user_request": "Invert a.ohm and the TEM sounding"}).split(";")[0])
    AgentCoordinator has no step for TDEM (requested)
    """
    named = [f"{label} ({key})" for key, label in _UNSUPPORTED_INPUTS.items()
             if config.get(key)]
    request = str(config.get("user_request") or config.get("request") or "")
    if not any(key in ("tdem_file", "em_file") for key in config if config.get(key)) and \
            names_tdem(request):
        named.append("TDEM (requested)")
    if not named:
        return None
    return ("AgentCoordinator has no step for " + ", ".join(named) + "; it runs the ERT "
            "pipeline only. Run this configuration with "
            "BaseAgent.run_unified_agent_workflow(), which does.")


def _input_problem(config: Dict[str, Any]) -> Optional[str]:
    """Why the ERT data *config* names cannot be run, or None.

    A request that says "time-lapse" of one file becomes a series of one,
    which the run refused only after loading it, while the preview planned the
    whole pipeline. Both now refuse it before any step.

    Examples
    --------
    >>> print(_input_problem({"time_lapse_files": ["a.ohm"]}))
    A time-lapse inversion needs at least two surveys; the configuration names one (a.ohm).
    >>> _input_problem({"time_lapse_files": ["a.ohm", "b.ohm"]}) is None
    True
    """
    files = _time_lapse_files(config)
    if len(files) == 1:
        return ("A time-lapse inversion needs at least two surveys; the configuration "
                f"names one ({files[0]}).")
    return None


def _load_failure(loaded: Any) -> str:
    """What the loader said about a survey it did not load, with its suggested fix."""
    reason = loaded.get('error') or loaded.get('summary') or 'no reason given'
    hint = loaded.get('error_fix_hint')
    return f"{reason} {hint}" if hint else str(reason)


def _time_lapse_files(config: Dict[str, Any]) -> List[str]:
    """The time-lapse survey files a configuration names, in acquisition order.

    ``ContextInputAgent`` writes them to ``time_lapse_files`` and also sets
    ``data_file`` to the first one as the baseline, which is why a pipeline
    that read only ``data_file`` inverted the baseline alone.
    ``timelapse_files`` is the older spelling.

    Examples
    --------
    >>> _time_lapse_files({"timelapse_files": ["a.ohm", "b.ohm"]})
    ['a.ohm', 'b.ohm']
    >>> _time_lapse_files({"data_file": "a.ohm"})
    []
    """
    return list(config.get("time_lapse_files") or config.get("timelapse_files") or [])


def _single_survey_file(config: Dict[str, Any]) -> Optional[str]:
    """The ERT file a single-survey run loads: ``data_file``, else ``ert_file``.

    The preview planned a run for either name, while the run read
    ``data_file`` alone and failed to load anything for an ``ert_file``.

    Examples
    --------
    >>> _single_survey_file({"ert_file": "a.ohm"})
    'a.ohm'
    >>> _single_survey_file({"data_file": "a.ohm", "ert_file": "b.ohm"})
    'a.ohm'
    """
    return config.get("data_file") or config.get("ert_file")


def _converts_water_content(config: Dict[str, Any]) -> bool:
    """Whether the coordinator's pipeline ends in a water-content step.

    The preview and the run both ask this, so the plan cannot list a step the
    run skips, or the reverse. A configuration that states an intent (the
    ``convert_to_water_content`` flag, petrophysical parameters, or the
    request's own words) is read by :func:`._intent.wants_water_content`, as
    the runtime reads it. The ``"water content" in request`` test this
    replaced missed a typo, a synonym and a request written in Chinese. A
    configuration that states nothing keeps the pipeline this coordinator has
    always run and documents, which converts.

    Examples
    --------
    >>> _converts_water_content({"data_file": "a.ohm"})
    True
    >>> _converts_water_content({"user_request": "estimate the water conent"})
    True
    >>> _converts_water_content({"user_request": "resistivity only, please"})
    False
    >>> _converts_water_content({"request": "帮我算含水量"})
    True
    """
    view = dict(config)
    if not str(view.get("user_request") or "").strip():
        view["user_request"] = str(view.get("request") or "")
    stated = (view.get("convert_to_water_content") is not None
              or bool(view.get("petrophysical_params"))
              or bool(view["user_request"].strip()))
    return wants_water_content(view) if stated else True


def _climate_gap(config: Dict[str, Any]) -> Optional[str]:
    """Why climate data *config* asks for will not be fetched, or None.

    The report says it with this reason: "No climate data was integrated"
    read as if none had been asked for.

    Examples
    --------
    >>> print(_climate_gap({"user_request": "compare with rainfall"}))
    no climate_config (site coordinates and dates) was given, so the climate step did not run
    >>> _climate_gap({"use_climate": True, "climate_config": {}}) is None
    True
    """
    if not wants_climate(config) or (config.get("use_climate") and "climate_config" in config):
        return None
    if "climate_config" not in config:
        return ("no climate_config (site coordinates and dates) was given, so the climate "
                "step did not run")
    return "a climate_config was given, but use_climate is not True, so the climate step did not run"


def _water_content_input(inversion_results: Dict[str, Any]) -> Dict[str, Any]:
    """A time-lapse result in the shape ``WaterContentAgent`` reads.

    That agent takes ``resistivity_model`` (cells, or cells x time steps) and
    one coverage value per cell. A time-lapse run returns ``final_models`` and
    a coverage per time step; the baseline's coverage masks every step, as
    the time-lapse report masks them. Other results pass through unchanged.
    """
    if inversion_results.get("inversion_mode") != "time-lapse":
        return inversion_results
    view = dict(inversion_results)
    view["resistivity_model"] = np.asarray(inversion_results.get("final_models"))
    coverage = inversion_results.get("coverage")
    if coverage is not None and len(coverage):
        coverage = np.asarray(coverage[0] if isinstance(coverage, (list, tuple))
                              else coverage, dtype=float)
        if coverage.ndim == 2:
            n_cells = view["resistivity_model"].shape[0]
            coverage = coverage[0] if coverage.shape[1] == n_cells else coverage[:, 0]
        view["coverage"] = coverage
    else:
        view["coverage"] = None
    return view


# ---------------------------------------------------------------------------
# Agent Coordinator
# ---------------------------------------------------------------------------
class AgentCoordinator:
    """
    Coordinates multiple agents to execute a complete workflow.
    
    The coordinator manages cross-modal geophysical workflows such as: 
    "load geophysical data → process → invert → convert to hydrologic parameters → report"
    with support for multiple data types (ERT, seismic, etc.) and LLM providers (GPT, Gemini, Claude).
    """
    
    def __init__(self, api_key: Optional[str] = None, output_dir: str = "results/agents",
                 llm_provider: str = "openai"):
        """
        Initialize the agent coordinator.
        
        Args:
            api_key: LLM API key for agents
            output_dir: Directory for saving results
            llm_provider: LLM provider to use ('openai', 'gemini', or 'claude')
        """
        self.api_key = api_key or self._get_default_api_key(llm_provider)
        self.output_dir = output_dir
        self.llm_provider = llm_provider.lower()
        self.agents = {}
        self.workflow_state = {
            'status': 'initialized',
            'current_step': None,
            'completed_steps': [],
            'data': {}
        }
        self.execution_log = []
        self.llm_usage_ledger: List[Dict[str, Any]] = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def preview_workflow(self, config: Dict[str, Any]) -> AgentResult:
        """Resolve and validate a workflow plan without running processing.

        Parameters
        ----------
        config : dict
            Workflow configuration. May include ``user_request`` or ``request``
            for deterministic preview parsing.

        Returns
        -------
        AgentResult
            Preview result with resolved config, validation warnings, plan, and
            approximate LLM cost.

        Raises
        ------
        None

        Examples
        --------
        >>> import tempfile
        >>> coordinator = AgentCoordinator(api_key=None, output_dir=tempfile.mkdtemp())
        >>> result = coordinator.preview_workflow({"data_file": "missing.ohm"})  # doctest: +ELLIPSIS
        <BLANKLINE>
        ===== PyHydroGeophysX Agent Preview =====
        ...
        >>> result["status"]
        'failed'
        """
        request = config.get("user_request") or config.get("request") or ""
        request_notes: List[str] = []
        resolved_config = self._resolve_config(config, request_notes)
        validation_errors, validation_warnings = self._validate_preview_files(resolved_config)
        unsupported = _unsupported_by_coordinator(resolved_config)
        problem = None if unsupported else _input_problem(resolved_config)
        plan = [] if unsupported or problem else self._build_preview_plan(resolved_config)
        for refusal in (unsupported, problem):
            if refusal:
                validation_errors.append(refusal)
        cost_estimate = self._estimate_preview_cost(resolved_config, plan)
        dep_warnings = self._check_dependencies(plan)
        validation_warnings = dep_warnings + validation_warnings + request_notes
        if (_time_lapse_files(resolved_config) and resolved_config.get("use_seismic")
                and "seismic_data" in resolved_config):
            validation_warnings.append(
                "use_seismic: the seismic structure constraint applies to a "
                "single-survey inversion; the time-lapse run skips it.")

        print("\n===== PyHydroGeophysX Agent Preview =====")
        if request:
            print(f"Request: {request}")
        print(f"Estimated LLM cost: ${cost_estimate:.4f} (approximate)")
        print("Resolved execution plan:")
        for idx, step in enumerate(plan, 1):
            print(f"  {idx}. {step['agent']}: {step['step']}")
        if validation_errors:
            print("Validation errors:")
            for err in validation_errors:
                print(f"  - {err}")
        if validation_warnings:
            print("Validation warnings:")
            for warn in validation_warnings:
                print(f"  - {warn}")

        if validation_errors:
            return AgentResult(
                status="failed",
                summary="Workflow preview found input problems before execution.",
                data={
                    "workflow_config": resolved_config,
                    "execution_plan": plan,
                    "validation_errors": validation_errors,
                },
                warnings=validation_warnings,
                cost_estimate_usd=cost_estimate,
                error="Input validation failed.",
                error_fix_hint=(
                    "Run this configuration with BaseAgent.run_unified_agent_workflow()."
                    if unsupported and len(validation_errors) == 1 else
                    "Name every survey of the series in time_lapse_files, or give the one "
                    "survey as data_file."
                    if problem and len(validation_errors) == 1 else
                    "Fix the listed file paths or unsupported extensions before running the workflow. See: "
                    "https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#data-file-not-found"
                ),
            )

        status = "success" if plan else "needs_review"
        summary = (
            "Workflow preview is ready to run."
            if plan
            else "Workflow preview needs more information before it can build a plan."
        )
        return AgentResult(
            status=status,
            summary=summary,
            data={
                "workflow_config": resolved_config,
                "execution_plan": plan,
                "validation_warnings": validation_warnings,
            },
            warnings=validation_warnings,
            next_suggested_action="Review the resolved configuration, then run without dry_run.",
            cost_estimate_usd=cost_estimate,
            error_fix_hint=None
            if plan
            else (
                "Provide a supported data file or a more specific workflow request. See: "
                "https://geohang.github.io/PyHydroGeophysX/agents/troubleshooting.html#ambiguous-natural-language-request"
            ),
        )

    def _check_dependencies(self, plan: List[Dict[str, Any]]) -> List[str]:
        """Check that required Python packages and CLI tools are available.

        Inspects the proposed *plan* and tests only the dependencies that will
        actually be used.  Returns a list of human-readable warning strings
        (empty list when everything is in order).

        Parameters
        ----------
        plan : list
            Execution plan as returned by :meth:`_build_preview_plan`.

        Returns
        -------
        list of str
            Warning messages for missing dependencies.
        """
        import importlib
        import subprocess

        agent_names = {step.get("agent", "") for step in plan}
        warnings_out: List[str] = []

        needs_pygimli = any(
            a in agent_names
            for a in (
                "ERTLoaderAgent", "ERTInversionAgent", "SeismicAgent",
                "StructureConstraintAgent", "InversionEvaluationAgent",
            )
        )
        if needs_pygimli:
            if importlib.util.find_spec("pygimli") is None:
                warnings_out.append(
                    "PyGIMLi is not installed or cannot be imported. "
                    "ERT/seismic steps will fail at runtime. "
                    "Install via: conda install -c gimli pygimli"
                )

        needs_gmsh = any("Mesh" in a for a in agent_names)
        if needs_gmsh:
            try:
                result = subprocess.run(
                    ["gmsh", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    warnings_out.append(
                        "GMSH binary returned non-zero exit code. "
                        "3D mesh generation may fail."
                    )
            except FileNotFoundError:
                warnings_out.append(
                    "GMSH is not on PATH. 3D mesh generation will fail at runtime. "
                    "Install GMSH and ensure it is accessible as 'gmsh'."
                )
            except Exception as exc:
                warnings_out.append(f"Could not verify GMSH installation: {exc}")

        needs_anthropic = False
        needs_google = False
        provider = getattr(self, "llm_provider", "openai")
        if provider == "claude":
            needs_anthropic = True
        elif provider in ("gemini", "google"):
            needs_google = True

        if needs_anthropic and importlib.util.find_spec("anthropic") is None:
            warnings_out.append(
                "anthropic package is not installed. LLM calls will fail. "
                "Install via: pip install anthropic"
            )
        def _importable(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except ImportError:            # its parent package is missing too
                return False

        # google-genai, or the older google-generativeai BaseAgent falls back to.
        if needs_google and not (_importable("google.genai")
                                 or _importable("google.generativeai")):
            warnings_out.append(
                "No Gemini SDK is installed. LLM calls will fail. "
                "Install via: pip install google-genai"
            )

        return warnings_out

    def _validate_preview_files(self, config: Dict[str, Any]) -> tuple:
        """Validate file references in a preview configuration."""
        extension_map = {
            "data_file": {".ohm", ".bin", ".dat", ".stg", ".txt", ".data"},
            "ert_file": {".ohm", ".bin", ".dat", ".stg", ".txt", ".data"},
            "electrode_file": {".dat", ".txt", ".csv"},
            "seismic_file": {".dat", ".txt"},
            "raw_seismic_file": {".sgy", ".segy"},
            "tdem_file": {".dat", ".txt", ".csv"},
            "mt_file": {".edi", ".xml", ".zmm", ".zrr", ".zss", ".j"},
            "csv_file": {".csv"},
            "metadata_file": {".json"},
        }
        errors: List[str] = []
        warnings_out: List[str] = []

        def _check_one(field: str, value: Any) -> None:
            if not value:
                return
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            absolute = path.resolve()
            if not absolute.exists():
                errors.append(f"{field}: file does not exist at {absolute}")
                return
            allowed = extension_map.get(field)
            if allowed and absolute.suffix.lower() not in allowed:
                errors.append(
                    f"{field}: unsupported extension '{absolute.suffix}'. "
                    f"Supported extensions: {', '.join(sorted(allowed))}."
                )
            size_mb = absolute.stat().st_size / (1024 * 1024)
            if size_mb > 500:
                warnings_out.append(
                    f"{field}: {absolute} is {size_mb:.1f} MB. Use local execution for large files."
                )

        for field in extension_map:
            _check_one(field, config.get(field))

        for list_field in ("time_lapse_files", "timelapse_files"):
            for value in config.get(list_field, []) or []:
                _check_one("ert_file", value)

        return errors, warnings_out

    def _resolve_config(self, config: Dict[str, Any],
                        notes: Optional[List[str]] = None) -> Dict[str, Any]:
        """The configuration the preview shows and the run executes.

        A ``user_request`` (or ``request``) fills in what the configuration
        leaves out, and only with what the request itself states, as read by
        :meth:`ContextInputAgent.request_inputs`:

        * its ERT files, when the configuration names no ERT data at all, or
          the time-lapse series it asks for when the configuration's one
          survey is part of it. A file mentioned beside the caller's
          ``data_file`` (an output name, say) no longer turns the run into a
          time-lapse series;
        * the instrument and the petrophysical parameters it names;
        * a seismic travel-time file, as the ``seismic_data`` of the seismic
          structure constraint when there is ERT data to constrain and the
          caller has not turned ``use_seismic`` off. Without ERT data it stays
          a ``seismic_file``, which this pipeline refuses, as it refuses the
          TDEM and SEG-Y files a request names.

        The parser's defaults - a regularisation, an iteration count, an
        uncertainty run - are not taken: they replaced the defaults this run
        has always used, the time-lapse ones among them, and a guess such as
        a seismic file refused the seismic constraint the caller had
        configured. A key the caller set keeps the caller's value. Wherever
        the request states something this run will not use, a sentence saying
        so is appended to ``notes``. The preview used to let the parsed values
        overwrite the caller's, and the run ignored the request, so a
        time-lapse series named only in the request was planned and never
        inverted; both now read this result. A configuration without a request
        is returned as given, apart from moving a SEG-Y ``seismic_file`` to
        ``raw_seismic_file``.
        """
        request = config.get("user_request") or config.get("request") or ""
        resolved = dict(config)
        notes = [] if notes is None else notes
        if request:
            from .context_input_agent import ContextInputAgent, _same_file

            caller_files = [str(f) for f in (config.get("data_file"), config.get("ert_file"),
                                             *_time_lapse_files(config)) if f]
            stated = ContextInputAgent(api_key=None, llm_provider=self.llm_provider) \
                .request_inputs(str(request), known_files=caller_files)

            named_files = stated.get("time_lapse_files") or (
                [stated["data_file"]] if stated.get("data_file") else [])
            # The configuration's one survey as the baseline of the series the
            # request asks for - the shape ContextInputAgent itself produces.
            # A second file alone ("save the model as model.bin") is no series.
            series = stated.get("time_lapse_files") or []
            baseline_of_series = (
                len(series) > 1 and bool(caller_files) and not _time_lapse_files(config)
                and all(any(_same_file(given, f) for f in series) for given in caller_files)
                and any(names_unnegated(str(request), term) for term in _TIME_LAPSE_TERMS))
            if baseline_of_series:
                resolved["time_lapse_files"] = series
            elif not any(config.get(key) for key in _ERT_INPUTS):
                for key in ("time_lapse_files", "data_file", "ert_file"):
                    if stated.get(key):
                        resolved[key] = stated[key]
            else:
                unused = [f for f in named_files
                          if not any(_same_file(f, given) for given in caller_files)]
                if unused:
                    used = _time_lapse_files(resolved) or [_single_survey_file(resolved)]
                    notes.append(f"The request names {', '.join(unused)}, but the run uses "
                                 f"the configuration's ERT data ({', '.join(map(str, used))}).")

            for key in ("instrument", "petrophysical_params"):
                if not stated.get(key):
                    continue
                if not resolved.get(key):
                    resolved[key] = stated[key]
                elif resolved[key] != stated[key]:
                    notes.append(f"The request gives {key} {stated[key]!r}, but the run uses "
                                 f"the configuration's {resolved[key]!r}.")

            if stated.get("tdem_file") and not resolved.get("tdem_file"):
                resolved["tdem_file"] = stated["tdem_file"]
            self._resolve_request_seismic(resolved, stated, notes, str(request))

            resolved["user_request"] = str(request)
            # The parser saw the request alone; the caller's time-lapse files
            # make this a time-lapse run, as the run itself decides.
            if _time_lapse_files(resolved) and config.get("inversion_mode") is None:
                resolved["inversion_mode"] = "time-lapse"
        self._normalize_raw_seismic_config(resolved)
        if wants_climate(resolved) and not (resolved.get("use_climate")
                                            and "climate_config" in resolved):
            notes.append(
                "Climate data were asked for, but the configuration has no climate_config, "
                "so the climate step does not run." if "climate_config" not in resolved else
                "A climate_config is given, but use_climate is not True, so the climate "
                "step does not run.")
        return resolved

    @staticmethod
    def _resolve_request_seismic(resolved: Dict[str, Any], stated: Dict[str, Any],
                                 notes: List[str], request: str) -> None:
        """Put the seismic data a request names where this pipeline reads it, in place.

        The pipeline's seismic step is the structure constraint of a
        single-survey ERT inversion, read from ``use_seismic`` and
        ``seismic_data``. A travel-time file the request names fills in
        ``seismic_data`` when there is ERT data and the caller has no seismic
        data of its own, and turns an unset ``use_seismic`` on. A raw SEG-Y
        file, or seismic data with no ERT to constrain, is recorded where
        :func:`_unsupported_by_coordinator` refuses it. Only a named file
        changes the run; a request that only mentions seismic data, and gets
        no constraint, is told what is missing.
        """
        from .context_input_agent import _same_file

        key = next((k for k in ("raw_seismic_file", "seismic_file") if stated.get(k)), None)
        named = stated.get(key) if key else None
        if not named and not (names_unnegated(request, "seismic", prefix=True)
                              or names_unnegated(request, "srt")):
            return
        refused = bool(resolved.get("seismic_file") or resolved.get("raw_seismic_file"))
        if named and resolved.get("seismic_data") is None and not refused:
            if key == "raw_seismic_file" or not (_time_lapse_files(resolved)
                                                 or _single_survey_file(resolved)):
                resolved[key] = named
                return
            if resolved.get("use_seismic") is None or resolved["use_seismic"]:
                resolved["seismic_data"] = named
        if named and resolved.get("use_seismic") is None and resolved.get("seismic_data") is not None:
            resolved["use_seismic"] = True
        given = resolved.get("seismic_data")
        if named and isinstance(given, (str, os.PathLike)) and not _same_file(named, given):
            notes.append(f"The request names seismic data ({named}), but the run uses the "
                         f"configuration's seismic_data ({given}).")
        if not (resolved.get("use_seismic") and "seismic_data" in resolved) and not refused:
            missing = ("use_seismic is off" if named or given is not None
                       else "the configuration gives no seismic_data")
            notes.append(f"The request asks for seismic data{f' ({named})' if named else ''}, "
                         f"but {missing}, so no seismic structure constraint is applied.")

    def _normalize_raw_seismic_config(self, config: Dict[str, Any]) -> None:
        """Separate raw SEG-Y paths from travel-time seismic paths in-place."""
        seismic_file = config.get("seismic_file")
        if seismic_file and Path(str(seismic_file)).suffix.lower() in {".sgy", ".segy"}:
            config["raw_seismic_file"] = seismic_file
            config.pop("seismic_file", None)
            config["raw_seismic_processing"] = True

    def _build_preview_plan(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a human-readable execution plan from a resolved config.

        Only a configuration :func:`_unsupported_by_coordinator` accepts gets
        here, so the plan is the ERT pipeline, or empty when no ERT data is
        named. The TDEM, SEG-Y and seismic-fusion plans this used to return
        described steps :meth:`execute_workflow` never ran.
        """
        if _time_lapse_files(config) or _single_survey_file(config):
            return self._ert_pipeline_plan(config)
        return []

    def _ert_pipeline_plan(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The steps :meth:`execute_workflow` runs for an ERT configuration.

        Built from the same tests the run makes, step for step, so the preview
        can no longer list a step the run skips (it listed an evaluation step
        the run never made) or omit one it runs (the water-content step, read
        from a substring, and every time step after the baseline).
        """
        time_lapse = bool(_time_lapse_files(config))
        seismic = (not time_lapse and bool(config.get("use_seismic", False))
                   and "seismic_data" in config)
        plan: List[Dict[str, Any]] = []
        if config.get("use_climate", False) and "climate_config" in config:
            plan.append({"agent": "ClimateDataAgent", "step": "Fetch climate data"})
        if time_lapse:
            count = len(_time_lapse_files(config))
            plan.append({"agent": "ERTLoaderAgent",
                         "step": f"Load each of the {count} time-lapse ERT datasets"})
            plan.append({"agent": "ERTInversionAgent", "step": "Run time-lapse inversion"})
        else:
            plan.append({"agent": "ERTLoaderAgent", "step": "Load and validate ERT data"})
            if seismic:
                plan.append({"agent": "SeismicAgent",
                             "step": "Process seismic data into a structure constraint"})
            plan.append({"agent": "ERTInversionAgent",
                         "step": "Run ERT inversion with the seismic structure constraint"
                         if seismic else "Run ERT inversion"})
        if _converts_water_content(config):
            plan.append({"agent": "WaterContentAgent",
                         "step": "Convert resistivity to water content"})
        plan.append({"agent": "ReportAgent",
                     "step": "Generate time-lapse report" if time_lapse
                     else "Generate workflow report"})
        return plan

    def _estimate_preview_cost(self, config: Dict[str, Any], plan: List[Dict[str, Any]]) -> float:
        """Estimate approximate LLM cost for the proposed workflow."""
        request = str(config.get("user_request", ""))
        prompt_tokens = estimate_tokens(request) + 400 * max(len(plan), 1)
        completion_tokens = 250 * max(len(plan), 1)
        return estimate_llm_cost_usd(
            self.llm_provider,
            config.get("llm_model", "gpt-4o-mini"),
            prompt_tokens,
            completion_tokens,
        )

    def _as_agent_result(self, result: Any, agent_name: str) -> AgentResult:
        """Normalize legacy agent dictionaries to AgentResult."""
        if isinstance(result, AgentResult):
            return result
        if isinstance(result, dict):
            warnings.warn(
                f"{agent_name} returned a legacy dictionary. "
                "This remains supported but will prefer AgentResult in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            return AgentResult.from_dict(
                result,
                default_summary=f"{agent_name} completed.",
            )
        return AgentResult(
            status="needs_review",
            summary=f"{agent_name} returned an unsupported result type.",
            data={"raw_result": result},
            error_fix_hint="Update the agent to return AgentResult or a legacy dictionary.",
        )

    def _get_default_api_key(self, provider: str) -> Optional[str]:
        """Get default API key based on provider."""
        provider_env_map = {
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GEMINI_API_KEY',
            'claude': 'ANTHROPIC_API_KEY'
        }
        env_var = provider_env_map.get(provider.lower())
        return os.getenv(env_var) if env_var else None
    
    def register_agent(self, agent_name: str, agent_instance):
        """
        Register an agent with the coordinator.
        
        Args:
            agent_name: Unique identifier for the agent
            agent_instance: Agent instance to register
        """
        self.agents[agent_name] = agent_instance
        self._log(f"Registered agent: {agent_name}")
    
    def _save_checkpoint(self, step_name: str, result: Any) -> bool:
        """Persist an intermediate result to disk for later resumption, if it can be.

        Checkpoints are stored as pickle files inside ``<output_dir>/checkpoints/``.
        Primitive/JSON-safe results are also saved as a ``*.json`` sidecar for
        quick inspection without unpickling.

        A checkpoint is a convenience for ``resume=True``, not part of the
        result, so writing one is best-effort. A result that cannot be pickled
        - an inversion result holds a pyGIMLi mesh, which cannot - or a disk
        that refuses the file leaves a warning in the log and no checkpoint,
        and the run goes on; a resumed run then repeats that step. It used to
        end the run: the example notebook's inversion finished and the workflow
        failed on pickling the mesh it had just computed, with no water content
        and no report. The pickle is written beside its final name and moved
        into place only when complete, so a failure leaves no partial file,
        and an older checkpoint of the same step is removed rather than left
        for a resumed run to mistake for this run's result.

        Parameters
        ----------
        step_name : str
            Unique workflow step identifier (e.g. ``'load_ert'``).
        result : Any
            The result object to persist.

        Returns
        -------
        bool
            Whether the checkpoint was written.
        """
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        pkl_path = os.path.join(ckpt_dir, f"{step_name}.pkl")
        partial = pkl_path + ".part"
        try:
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(partial, "wb") as fh:
                pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(partial, pkl_path)
        except Exception as exc:  # noqa: BLE001 - a checkpoint must not fail the run
            for stale in (partial, pkl_path):
                try:
                    os.remove(stale)
                except OSError:
                    pass
            self._log(f"Checkpoint for {step_name} not saved ({type(exc).__name__}: "
                      f"{exc}); the run continues, and a resumed run repeats this step.",
                      level="WARNING")
            return False

        # Best-effort JSON sidecar (skipped for non-serialisable objects)
        json_path = os.path.join(ckpt_dir, f"{step_name}.json")
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(result if isinstance(result, dict) else {"value": str(result)}, fh, indent=2, default=str)
        except Exception:
            try:
                os.remove(json_path)
            except OSError:
                pass

        self._log(f"Checkpoint saved: {step_name} → {pkl_path}")
        return True

    def _load_checkpoint(self, step_name: str) -> Optional[Any]:
        """Load a previously saved checkpoint.

        Parameters
        ----------
        step_name : str
            The step name used when calling :meth:`_save_checkpoint`.

        Returns
        -------
        Any or None
            The unpickled result, or ``None`` if no checkpoint exists or it
            cannot be read - a file truncated by an interrupted write, or one
            corrupted since. The step then runs again, and its new result
            replaces the file.
        """
        pkl_path = os.path.join(self.output_dir, "checkpoints", f"{step_name}.pkl")
        if not os.path.exists(pkl_path):
            return None
        try:
            with open(pkl_path, "rb") as fh:
                result = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - a bad checkpoint means "run the step"
            self._log(f"Checkpoint {step_name} could not be read ({type(exc).__name__}: "
                      f"{exc}); the step runs again.", level="WARNING")
            return None
        self._log(f"Checkpoint loaded: {step_name} ← {pkl_path}")
        return result

    def execute_workflow(self, config: Dict[str, Any], dry_run: bool = False,
                         resume: bool = False) -> Any:
        """
        Execute the complete workflow with registered agents.
        
        Args:
            config: Configuration dictionary containing:
                - user_request: Optional natural-language request. It is read
                  as the preview reads it: the ERT files it names are run
                  when this dictionary names none, the instrument,
                  petrophysical parameters and seismic data it names fill
                  what is left out, and keys set here win. What it states
                  that the run does not use is reported in ``warnings``
                - data_file: Path to ERT data file (``ert_file`` is accepted)
                - time_lapse_files: Survey files in acquisition order; when
                  given, every one is loaded and inverted together (a
                  time-lapse inversion) and data_file is ignored
                - instrument: Instrument type (E4D, Syscal, etc.)
                - inversion_params: Parameters for inversion
                - petrophysical_params: Parameters for water content conversion.
                  The water-content step runs unless the configuration says
                  otherwise (``convert_to_water_content``, or a request that
                  asks for resistivity only), read as the runtime reads it
                - use_seismic: Whether to include seismic processing (default: False)
                - seismic_data: Seismic travel-time data for the structure
                  constraint, loaded or as a file path
                - use_climate: Whether to include climate data (default: False)
                - climate_config: Climate data configuration (coords/geometry, dates, etc.)
                - ert_timestamps: Timestamps for ERT acquisitions (for climate alignment)
                A configuration naming ``tdem_file``, ``em_file``,
                ``seismic_file`` or ``raw_seismic_file``, or a request that
                asks for TDEM, is refused: this pipeline has no step for them,
                and ``BaseAgent.run_unified_agent_workflow()`` runs them.
            dry_run: If True, return a preview plan without executing.
            resume: If True, load checkpointed intermediate results and skip
                    already-completed steps.

        Returns:
            Dictionary containing workflow results, or ``AgentResult`` when
            ``dry_run=True``.
        """
        if dry_run:
            return self.preview_workflow(config)

        self.llm_usage_ledger = []
        run_warnings: List[str] = []
        # (what, why) for each product the request asked for, or step the run
        # attempted, that the report will not contain; the report says so.
        not_delivered: List[Tuple[str, str]] = []
        self._log("Starting workflow execution" + (" (resume mode)" if resume else ""))
        self.workflow_state['status'] = 'running'
        
        def _step_done(step_name: str) -> bool:
            """Return True if *step_name* has a valid checkpoint when resuming."""
            if not resume:
                return False
            return self._load_checkpoint(step_name) is not None

        def _maybe_load(step_name: str) -> Optional[Any]:
            """Load checkpoint when resuming; return None otherwise."""
            if not resume:
                return None
            return self._load_checkpoint(step_name)
        
        try:
            # Run the configuration the preview showed: the files and values a
            # request states fill what the caller left out, and a configuration
            # the preview refuses is refused here too, before any step runs.
            config = self._resolve_config(config, run_warnings)
            refusal = _unsupported_by_coordinator(config) or _input_problem(config)
            if refusal:
                raise ValueError(refusal)
            for note in run_warnings:
                self._log(note, level='WARNING')
            climate_gap = _climate_gap(config)
            if climate_gap:
                not_delivered.append((PRODUCTS['climate'], climate_gap))

            # Step 0 (Optional): Fetch climate data if requested
            if config.get('use_climate', False) and 'climate_config' in config:
                self._log("Step 0: Fetching climate data")
                self.workflow_state['current_step'] = 'fetch_climate'
                
                ckpt = _maybe_load('fetch_climate')
                if ckpt is not None:
                    climate_results = ckpt
                    self._log("  → Loaded from checkpoint")
                else:
                    climate_config = config['climate_config']
                    if 'ert_timestamps' in config:
                        climate_config['ert_timestamps'] = config['ert_timestamps']
                    climate_results = self._execute_agent('climate_data', climate_config)
                    self._save_checkpoint('fetch_climate', climate_results)
                self.workflow_state['data']['climate_data'] = climate_results
                self.workflow_state['completed_steps'].append('fetch_climate')
            
            # A time-lapse configuration names every survey; data_file is only
            # its baseline, so reading data_file alone inverted one survey.
            time_lapse_files = _time_lapse_files(config)

            # Step 1: Load ERT data
            self._log("Step 1: Loading ERT data")
            self.workflow_state['current_step'] = 'load_ert'
            ckpt = _maybe_load('load_ert')
            if ckpt is not None:
                ert_data = ckpt
                self._log("  → Loaded from checkpoint")
            elif time_lapse_files:
                ert_data = []
                for index, data_file in enumerate(time_lapse_files, 1):
                    self._log(f"  Loading survey {index}/{len(time_lapse_files)}: {data_file}")
                    loaded = self._execute_agent('ert_loader', {
                        'data_file': data_file,
                        'instrument': config.get('instrument', 'E4D'),
                        'project_dir': config.get('project_dir', '.'),
                        'crs': config.get('crs', 'local')
                    })
                    if loaded.get('status') != 'success':
                        raise ValueError(f"Time-lapse survey {data_file} could not be loaded: "
                                         f"{_load_failure(loaded)}")
                    ert_data.append(loaded)
                if len(ert_data) < 2:
                    raise ValueError("A time-lapse inversion needs at least two surveys; "
                                     f"the configuration names {len(ert_data)}.")
                self._save_checkpoint('load_ert', ert_data)
            else:
                ert_data = self._execute_agent('ert_loader', {
                    'data_file': _single_survey_file(config),
                    'instrument': config.get('instrument', 'E4D'),
                    'project_dir': config.get('project_dir', '.'),
                    'crs': config.get('crs', 'local')
                })
                # As for each survey of a series: a survey the loader did not
                # load (an instrument that contradicts the file header, say)
                # stops the run with the loader's reason, where it used to
                # fail further on with a bare KeyError 'ert_data'.
                if ert_data.get('status') != 'success':
                    raise ValueError(f"ERT survey {_single_survey_file(config)} could not be "
                                     f"loaded: {_load_failure(ert_data)}")
                self._save_checkpoint('load_ert', ert_data)
            self.workflow_state['data']['ert_data'] = ert_data
            self.workflow_state['completed_steps'].append('load_ert')

            # Step 1b (Optional): Process seismic data if available. The
            # structure constraint is applied to a single-survey inversion
            # only, so a time-lapse run skips it rather than compute it unused.
            if (time_lapse_files and config.get('use_seismic', False)
                    and 'seismic_data' in config):
                self._log("Step 1b: Seismic structure is not applied to a time-lapse "
                          "inversion; skipped", level='WARNING')
                run_warnings.append("use_seismic: the seismic structure constraint applies "
                                    "to a single-survey inversion; the time-lapse run "
                                    "skipped it.")
            elif config.get('use_seismic', False) and 'seismic_data' in config:
                self._log("Step 1b: Processing seismic data")
                self.workflow_state['current_step'] = 'process_seismic'
                ckpt = _maybe_load('process_seismic')
                if ckpt is not None:
                    seismic_results = ckpt
                    self._log("  → Loaded from checkpoint")
                else:
                    seismic_results = self._execute_agent('seismic_processor', {
                        'seismic_data': config['seismic_data'],
                        'velocity_threshold': config.get('velocity_threshold', 1200)
                    })
                    if seismic_results.get('status') == 'success':
                        self._save_checkpoint('process_seismic', seismic_results)
                if seismic_results.get('status') == 'success':
                    self.workflow_state['data']['seismic_structure'] = seismic_results
                    self.workflow_state['completed_steps'].append('process_seismic')
                else:
                    # A failed seismic step is not a structure. Stored as one,
                    # it made the inversion "structure-constrained" by a result
                    # that held no interface, and the step was listed as done.
                    reason = (seismic_results.get('error') or seismic_results.get('summary')
                              or f"status {seismic_results.get('status')!r}")
                    note = (f"The seismic step failed ({reason}), so no seismic structure "
                            f"constraint was applied: the ERT inversion is unconstrained.")
                    self._log(note, level='WARNING')
                    run_warnings.append(note)
                    not_delivered.append(('Seismic structure constraint',
                                          f"the seismic step failed: {reason}"))
            
            # Step 2: ERT Inversion
            self._log("Step 2: Running ERT inversion")
            self.workflow_state['current_step'] = 'invert_ert'
            ckpt = _maybe_load('invert_ert')
            if ckpt is not None:
                inversion_results = ckpt
                self._log("  → Loaded from checkpoint")
            elif time_lapse_files:
                # The keys the legacy time-lapse pipeline hands the same agent.
                inversion_results = self._execute_agent('ert_inversion', {
                    'time_lapse_data': [loaded['ert_data'] for loaded in ert_data],
                    # The loaded surveys no longer carry their file names, and
                    # the acquisition times are read from them.
                    'source_files': list(time_lapse_files),
                    'inversion_mode': 'time-lapse',
                    'time_lapse_method': config.get('time_lapse_method', IMPLEMENTED_SCHEME),
                    'temporal_regularization': config.get('temporal_regularization', 10.0),
                    'baseline_index': 0,
                    'inversion_params': config.get('inversion_params', {}),
                })
                self._save_checkpoint('invert_ert', inversion_results)
            else:
                inversion_input = {
                    'ert_data': ert_data['ert_data'],
                    'inversion_params': config.get('inversion_params', {}),
                }
                if 'seismic_structure' in self.workflow_state['data']:
                    inversion_input['seismic_structure'] = self.workflow_state['data']['seismic_structure']
                    inversion_input['use_structure_constraint'] = True
                inversion_results = self._execute_agent('ert_inversion', inversion_input)
                self._save_checkpoint('invert_ert', inversion_results)
            self.workflow_state['data']['inversion_results'] = inversion_results
            self.workflow_state['completed_steps'].append('invert_ert')

            # Step 3: Convert to water content, when this configuration asks for
            # it - decided by the same test the preview's plan is built from.
            if _converts_water_content(config):
                self._log("Step 3: Converting to water content")
                self.workflow_state['current_step'] = 'convert_to_wc'
                ckpt = _maybe_load('convert_to_wc')
                if ckpt is not None:
                    wc_results = ckpt
                    self._log("  → Loaded from checkpoint")
                else:
                    from ._uncertainty import realizations

                    # A water content always carries its uncertainty: a single
                    # draw reported a standard deviation of exactly zero, which
                    # reads as certainty about a conversion nobody calibrated.
                    wc_results = self._execute_agent('water_content', {
                        'inversion_results': _water_content_input(inversion_results),
                        'petrophysical_params': config.get('petrophysical_params', {}),
                        'uncertainty_analysis': True,
                        'n_realizations': realizations(config, note=self._log)
                    })
                    self._save_checkpoint('convert_to_wc', wc_results)
                self.workflow_state['data']['water_content'] = wc_results
                self.workflow_state['completed_steps'].append('convert_to_wc')
            else:
                self._log("Step 3: Water content was not requested; skipped")
                # The report says so instead of leaving the section empty.
                self.workflow_state['data']['skip_petrophysics'] = True

            # Step 4: Generate report
            self._log("Step 4: Generating report")
            self.workflow_state['current_step'] = 'generate_report'
            ckpt = _maybe_load('generate_report')
            if ckpt is not None:
                report = ckpt
                self._log("  → Loaded from checkpoint")
            elif time_lapse_files and hasattr(self.agents.get('report'),
                                              'generate_timelapse_report'):
                report = self._execute_agent(
                    'report', self._timelapse_report_input(config, time_lapse_files,
                                                           not_delivered),
                    method='generate_timelapse_report')
                if report.get('status') == 'failed':
                    raise RuntimeError(f"Time-lapse report failed: {report.get('error')}")
                self._save_checkpoint('generate_report', report)
            else:
                report = self._execute_agent('report', {
                    'workflow_data': {**self.workflow_state['data'],
                                      'not_delivered': list(not_delivered)},
                    'config': config,
                    'output_dir': self.output_dir
                })
                self._save_checkpoint('generate_report', report)
            self.workflow_state['data']['report'] = report
            self.workflow_state['completed_steps'].append('generate_report')
            
            # Mark workflow as complete
            self.workflow_state['status'] = 'completed'
            self.workflow_state['current_step'] = None
            self._log("Workflow completed successfully")
            
            # Save workflow state
            self._save_workflow_state()
            
            return {
                'status': 'success',
                'results': self.workflow_state['data'],
                'log': self.execution_log,
                'warnings': run_warnings,
                'llm_usage_ledger': self.llm_usage_ledger,
                'total_llm_cost_estimate_usd': sum(
                    float(item.get('cost_estimate_usd') or 0.0)
                    for item in self.llm_usage_ledger
                ),
            }
            
        except Exception as e:
            self.workflow_state['status'] = 'failed'
            self._log(f"Workflow failed: {str(e)}", level='ERROR')
            self._save_workflow_state()
            
            return {
                'status': 'failed',
                'error': str(e),
                'log': self.execution_log,
                'warnings': run_warnings,
                'partial_results': self.workflow_state['data'],
                'llm_usage_ledger': self.llm_usage_ledger,
                'total_llm_cost_estimate_usd': sum(
                    float(item.get('cost_estimate_usd') or 0.0)
                    for item in self.llm_usage_ledger
                ),
            }
    
    def _timelapse_report_input(self, config: Dict[str, Any],
                                time_lapse_files: List[str],
                                not_delivered: Optional[List[Tuple[str, str]]] = None
                                ) -> Dict[str, Any]:
        """What ``ReportAgent.generate_timelapse_report`` reads, from this run."""
        data = self.workflow_state['data']
        inversion = data.get('inversion_results')
        inversion = inversion.to_dict() if isinstance(inversion, AgentResult) else dict(inversion or {})
        water_content = data.get('water_content')
        if water_content is not None and water_content.get('water_content_mean') is not None:
            # The report's water-content section reads one entry per survey,
            # as the legacy pipeline stores them; the agent returns cells x
            # surveys.
            mean = np.asarray(water_content.get('water_content_mean'), dtype=float)
            mean = mean.reshape(mean.shape[0], -1)
            std = np.asarray(water_content.get('water_content_std'), dtype=float)
            std = std.reshape(mean.shape) if std.size == mean.size else np.zeros_like(mean)
            per_step = [{'water_content_mean': mean[:, i], 'water_content_std': std[:, i]}
                        for i in range(mean.shape[1])]
            inversion.setdefault('time_lapse_water_content', per_step)
            inversion.setdefault('water_content_mean', per_step[0]['water_content_mean'])
            inversion.setdefault('water_content_std', per_step[0]['water_content_std'])
            inversion.setdefault('petrophysical_params', config.get('petrophysical_params', {}))
        # The survey dates are in the file names; the report labels its rows
        # with them and states the study period.
        dates = _dates_from_filenames(time_lapse_files)
        site_info = {'survey_dates': dates,
                     'study_period': f"{min(dates)} to {max(dates)}" if dates else 'N/A'}
        site_info.update(config.get('site_info') or {})
        return {
            'inversion_results': inversion,
            'climate_data': data.get('climate_data'),
            'site_info': site_info,
            'workflow_config': config,
            'time_lapse_method': config.get('time_lapse_method', IMPLEMENTED_SCHEME),
            'inversion_mode': 'time-lapse',
            'output_dir': self.output_dir,
            'not_delivered': list(not_delivered or []),
        }

    def _execute_agent(self, agent_name: str, input_data: Dict[str, Any],
                       method: str = "execute") -> AgentResult:
        """
        Execute a specific agent.

        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            method: The agent method to call with ``input_data``

        Returns:
            Agent execution results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")

        agent = self.agents[agent_name]
        self._log(f"Executing agent: {agent_name}")

        try:
            result = self._as_agent_result(getattr(agent, method)(input_data), agent_name)
            ledger = getattr(agent, "llm_usage_ledger", [])
            seen_count = int(getattr(agent, "_coordinator_seen_ledger_count", 0) or 0)
            if len(ledger) > seen_count:
                self.llm_usage_ledger.extend(ledger[seen_count:])
                setattr(agent, "_coordinator_seen_ledger_count", len(ledger))
            self._log(f"Agent {agent_name} completed successfully")
            return result
        except Exception as e:
            self._log(f"Agent {agent_name} failed: {str(e)}", level='ERROR')
            raise
    
    def _log(self, message: str, level: str = 'INFO'):
        """Add entry to execution log."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.execution_log.append(log_entry)
        print(f"[{level}] {message}")
    
    def _save_workflow_state(self):
        """Save current workflow state to file."""
        state_file = os.path.join(self.output_dir, 'workflow_state.json')
        
        # Convert non-serializable objects to strings
        serializable_state = {
            'status': self.workflow_state['status'],
            'current_step': self.workflow_state['current_step'],
            'completed_steps': self.workflow_state['completed_steps'],
            'data_keys': list(self.workflow_state['data'].keys())
        }
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_state, f, indent=2)
        
        # Save execution log
        log_file = os.path.join(self.output_dir, 'execution_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.execution_log, f, indent=2)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution."""
        total_cost = sum(
            float(item.get('cost_estimate_usd') or 0.0)
            for item in self.llm_usage_ledger
        )
        total_tokens = sum(
            int(item.get('total_tokens') or 0)
            for item in self.llm_usage_ledger
        )
        return {
            'status': self.workflow_state['status'],
            'completed_steps': self.workflow_state['completed_steps'],
            'total_steps': len(self.workflow_state['completed_steps']),
            'current_step': self.workflow_state['current_step'],
            'available_results': list(self.workflow_state['data'].keys()),
            'total_llm_cost_estimate_usd': total_cost,
            'total_llm_tokens': total_tokens,
            'llm_calls': len(self.llm_usage_ledger),
        }
