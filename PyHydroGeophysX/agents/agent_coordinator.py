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
from typing import Any, Dict, List, Optional

from .base_agent import AgentResult
from ._pricing import estimate_llm_cost_usd, estimate_tokens


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
        >>> coordinator = AgentCoordinator(api_key=None)
        >>> result = coordinator.preview_workflow({"data_file": "missing.ohm"})
        >>> result["status"]
        'failed'
        """
        request = config.get("user_request") or config.get("request") or ""
        resolved_config = dict(config)

        if request:
            from .context_input_agent import ContextInputAgent

            context_agent = ContextInputAgent(api_key=None, llm_provider=self.llm_provider)
            preview = context_agent.preview_config(str(request))
            resolved_config.update(preview.get("workflow_config", preview.get("data", {}).get("workflow_config", {})))
            resolved_config["user_request"] = str(request)

        self._normalize_raw_seismic_config(resolved_config)
        validation_errors, validation_warnings = self._validate_preview_files(resolved_config)
        plan = self._build_preview_plan(resolved_config)
        cost_estimate = self._estimate_preview_cost(resolved_config, plan)
        dep_warnings = self._check_dependencies(plan)
        validation_warnings = dep_warnings + validation_warnings

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
        if needs_google and importlib.util.find_spec("google.generativeai") is None:
            warnings_out.append(
                "google-generativeai package is not installed. LLM calls will fail. "
                "Install via: pip install google-generativeai"
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

    def _normalize_raw_seismic_config(self, config: Dict[str, Any]) -> None:
        """Separate raw SEG-Y paths from travel-time seismic paths in-place."""
        seismic_file = config.get("seismic_file")
        if seismic_file and Path(str(seismic_file)).suffix.lower() in {".sgy", ".segy"}:
            config["raw_seismic_file"] = seismic_file
            config.pop("seismic_file", None)
            config["raw_seismic_processing"] = True

    def _build_preview_plan(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build a human-readable execution plan from a resolved config."""
        request = str(config.get("user_request", "")).lower()
        has_ert = bool(config.get("ert_file") or config.get("data_file"))
        has_seismic = bool(config.get("seismic_file"))
        has_raw_seismic = bool(config.get("raw_seismic_file"))
        has_tdem = bool(config.get("tdem_file") or "tdem" in request)
        time_lapse_files = config.get("time_lapse_files") or config.get("timelapse_files")
        wants_water_content = "water content" in request or bool(config.get("petrophysical_params"))

        if has_tdem:
            return [
                {"agent": "TDEMAgent", "step": "Validate and process TDEM data"},
                {"agent": "ReportAgent", "step": "Summarize TDEM results"},
            ]
        if time_lapse_files:
            return [
                {"agent": "ERTLoaderAgent", "step": "Load each time-lapse ERT dataset"},
                {"agent": "ERTInversionAgent", "step": "Run time-lapse inversion"},
                {"agent": "InversionEvaluationAgent", "step": "Evaluate quality and retry if needed"},
                {"agent": "ReportAgent", "step": "Generate time-lapse report"},
            ]
        if has_raw_seismic and has_ert:
            return [
                {"agent": "RawSeismicProcessor", "step": "Read raw SEG-Y and organize shot gathers"},
                {"agent": "RawSeismicProcessor", "step": "Apply preprocessing and first-break picking"},
                {"agent": "RawSeismicProcessor", "step": "Export PyGIMLi travel-time data"},
                {"agent": "SeismicAgent", "step": "Run seismic refraction workflow"},
                {"agent": "ERTLoaderAgent", "step": "Load ERT data"},
                {"agent": "StructureConstraintAgent", "step": "Apply seismic structure to ERT inversion"},
                {"agent": "PetrophysicsAgent", "step": "Convert resistivity to water content with uncertainty"},
                {"agent": "ReportAgent", "step": "Generate data-fusion report"},
            ]
        if has_raw_seismic:
            return [
                {"agent": "RawSeismicProcessor", "step": "Read raw SEG-Y and organize shot gathers"},
                {"agent": "RawSeismicProcessor", "step": "Apply preprocessing and first-break picking"},
                {"agent": "RawSeismicProcessor", "step": "Export PyGIMLi travel-time data"},
                {"agent": "SeismicAgent", "step": "Run seismic refraction workflow"},
                {"agent": "ReportAgent", "step": "Summarize seismic processing and SRT results"},
            ]
        if has_ert and has_seismic:
            return [
                {"agent": "SeismicAgent", "step": "Invert seismic data and extract interfaces"},
                {"agent": "ERTLoaderAgent", "step": "Load ERT data"},
                {"agent": "StructureConstraintAgent", "step": "Apply seismic structure to ERT inversion"},
                {"agent": "PetrophysicsAgent", "step": "Convert resistivity to water content with uncertainty"},
                {"agent": "ReportAgent", "step": "Generate data-fusion report"},
            ]
        if has_seismic:
            return [
                {"agent": "SeismicAgent", "step": "Run seismic refraction workflow"},
                {"agent": "ReportAgent", "step": "Summarize seismic results"},
            ]
        if has_ert:
            plan = [
                {"agent": "ERTLoaderAgent", "step": "Load and validate ERT data"},
                {"agent": "ERTInversionAgent", "step": "Run ERT inversion"},
                {"agent": "InversionEvaluationAgent", "step": "Evaluate inversion quality"},
            ]
            if wants_water_content:
                plan.append(
                    {"agent": "WaterContentAgent", "step": "Convert resistivity to water content"}
                )
            plan.append({"agent": "ReportAgent", "step": "Generate workflow report"})
            return plan
        return []

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
    
    def _save_checkpoint(self, step_name: str, result: Any) -> None:
        """Persist an intermediate result to disk for later resumption.

        Checkpoints are stored as pickle files inside ``<output_dir>/checkpoints/``.
        Primitive/JSON-safe results are also saved as a ``*.json`` sidecar for
        quick inspection without unpickling.

        Parameters
        ----------
        step_name : str
            Unique workflow step identifier (e.g. ``'load_ert'``).
        result : Any
            The result object to persist.
        """
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        pkl_path = os.path.join(ckpt_dir, f"{step_name}.pkl")
        with open(pkl_path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Best-effort JSON sidecar (skipped for non-serialisable objects)
        try:
            json_path = os.path.join(ckpt_dir, f"{step_name}.json")
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(result if isinstance(result, dict) else {"value": str(result)}, fh, indent=2, default=str)
        except Exception:
            pass

        self._log(f"Checkpoint saved: {step_name} → {pkl_path}")

    def _load_checkpoint(self, step_name: str) -> Optional[Any]:
        """Load a previously saved checkpoint.

        Parameters
        ----------
        step_name : str
            The step name used when calling :meth:`_save_checkpoint`.

        Returns
        -------
        Any or None
            The unpickled result, or ``None`` if no checkpoint exists.
        """
        pkl_path = os.path.join(self.output_dir, "checkpoints", f"{step_name}.pkl")
        if not os.path.exists(pkl_path):
            return None
        try:
            with open(pkl_path, "rb") as fh:
                result = pickle.load(fh)
            self._log(f"Checkpoint loaded: {step_name} ← {pkl_path}")
            return result
        except Exception as exc:
            self._log(f"Failed to load checkpoint {step_name}: {exc}", level="WARNING")
            return None

    def execute_workflow(self, config: Dict[str, Any], dry_run: bool = False,
                         resume: bool = False) -> Any:
        """
        Execute the complete workflow with registered agents.
        
        Args:
            config: Configuration dictionary containing:
                - data_file: Path to ERT data file
                - instrument: Instrument type (E4D, Syscal, etc.)
                - inversion_params: Parameters for inversion
                - petrophysical_params: Parameters for water content conversion
                - use_seismic: Whether to include seismic processing (default: False)
                - seismic_data: Optional seismic data file
                - use_climate: Whether to include climate data (default: False)
                - climate_config: Climate data configuration (coords/geometry, dates, etc.)
                - ert_timestamps: Timestamps for ERT acquisitions (for climate alignment)
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
            
            # Step 1: Load ERT data
            self._log("Step 1: Loading ERT data")
            self.workflow_state['current_step'] = 'load_ert'
            ckpt = _maybe_load('load_ert')
            if ckpt is not None:
                ert_data = ckpt
                self._log("  → Loaded from checkpoint")
            else:
                ert_data = self._execute_agent('ert_loader', {
                    'data_file': config.get('data_file'),
                    'instrument': config.get('instrument', 'E4D'),
                    'project_dir': config.get('project_dir', '.'),
                    'crs': config.get('crs', 'local')
                })
                self._save_checkpoint('load_ert', ert_data)
            self.workflow_state['data']['ert_data'] = ert_data
            self.workflow_state['completed_steps'].append('load_ert')
            
            # Step 1b (Optional): Process seismic data if available
            if config.get('use_seismic', False) and 'seismic_data' in config:
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
                    self._save_checkpoint('process_seismic', seismic_results)
                self.workflow_state['data']['seismic_structure'] = seismic_results
                self.workflow_state['completed_steps'].append('process_seismic')
            
            # Step 2: ERT Inversion
            self._log("Step 2: Running ERT inversion")
            self.workflow_state['current_step'] = 'invert_ert'
            ckpt = _maybe_load('invert_ert')
            if ckpt is not None:
                inversion_results = ckpt
                self._log("  → Loaded from checkpoint")
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
            
            # Step 3: Convert to water content
            self._log("Step 3: Converting to water content")
            self.workflow_state['current_step'] = 'convert_to_wc'
            ckpt = _maybe_load('convert_to_wc')
            if ckpt is not None:
                wc_results = ckpt
                self._log("  → Loaded from checkpoint")
            else:
                wc_results = self._execute_agent('water_content', {
                    'inversion_results': inversion_results,
                    'petrophysical_params': config.get('petrophysical_params', {}),
                    'uncertainty_analysis': config.get('run_uncertainty', False),
                    'n_realizations': config.get('n_realizations', 100)
                })
                self._save_checkpoint('convert_to_wc', wc_results)
            self.workflow_state['data']['water_content'] = wc_results
            self.workflow_state['completed_steps'].append('convert_to_wc')
            
            # Step 4: Generate report
            self._log("Step 4: Generating report")
            self.workflow_state['current_step'] = 'generate_report'
            ckpt = _maybe_load('generate_report')
            if ckpt is not None:
                report = ckpt
                self._log("  → Loaded from checkpoint")
            else:
                report = self._execute_agent('report', {
                    'workflow_data': self.workflow_state['data'],
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
                'partial_results': self.workflow_state['data'],
                'llm_usage_ledger': self.llm_usage_ledger,
                'total_llm_cost_estimate_usd': sum(
                    float(item.get('cost_estimate_usd') or 0.0)
                    for item in self.llm_usage_ledger
                ),
            }
    
    def _execute_agent(self, agent_name: str, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute a specific agent.
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            
        Returns:
            Agent execution results
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not registered")
        
        agent = self.agents[agent_name]
        self._log(f"Executing agent: {agent_name}")
        
        try:
            result = self._as_agent_result(agent.execute(input_data), agent_name)
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
