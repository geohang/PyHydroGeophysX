"""The pre-controller agent workflow, kept reachable on purpose.

``BaseAgent.run_unified_agent_workflow`` now builds a run context and lets the
controller in :mod:`PyHydroGeophysX.agents.runtime` choose each step. This
module holds the implementation it replaced: it classifies the request into a
workflow type (:func:`_detect_workflow_type`, formerly a method of the removed
``WorkflowOrchestratorAgent``) and runs that type's fixed sequence. A request
no type matches is refused; it no longer goes to the removed
``CodeGenerationAgent``. It covers
workflow types whose tools cannot be exercised on every machine (ParFlow
output, SEG-Y files, a GPU), and keeping it callable lets a run that behaves
differently be compared against the code it replaced.

It lived inside ``base_agent.py``, where every agent module imported it with
the base class. Reach it as before, through
``BaseAgent.run_legacy_agent_workflow`` or by setting
``PHGX_LEGACY_WORKFLOW=1``.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._geocode import coords_from_config, geocode_place
from ._intent import climate_blocker, wants_climate, wants_water_content
from ._method import IMPLEMENTED_SCHEME
from .base_agent import _dates_from_filenames


def _detect_workflow_type(config: Dict[str, Any]) -> str:
    """Classify a configuration into the legacy pipeline's workflow type.

    Moved here from ``WorkflowOrchestratorAgent``, which was removed: only this
    pipeline classifies requests, and the runtime controller does not.

    Args:
        config: Workflow configuration dictionary

    Returns:
        Workflow type string.  Possible values:
        'tdem', 'seismic', 'model_output', 'time_lapse', 'data_fusion',
        'ert_data_process', 'direct_ert', 'standard_ert', 'custom'
    """
    config_keys = set(config.keys())
    user_request_lower = config.get('user_request', '').lower()

    # ── Intent flags ────────────────────────────────────────────────────
    mentions_ert = ('ert' in user_request_lower) or bool(config.get('ert_file'))
    mentions_seismic = (
        ('seismic' in user_request_lower)
        or bool(config.get('seismic_file') or config.get('raw_seismic_file'))
    )
    mentions_tdem = ('tdem' in user_request_lower) or bool(config.get('tdem_file'))
    inversion_keywords = ['invert', 'inversion', 'tomography', 'forward']
    mentions_inversion = any(kw in user_request_lower for kw in inversion_keywords)

    hydro_keywords = [
        'modflow', 'parflow', 'par flow', 'hydrological model',
        'watercontent', 'saturation', 'porosity',
    ]
    mentions_hydro = (
        any(kw in user_request_lower for kw in hydro_keywords)
        or bool(config.get('hydro_model'))
    )
    hydro_only = mentions_hydro and not (
        mentions_ert or mentions_seismic or mentions_tdem or mentions_inversion
    )

    processing_keywords = [
        'data processing', 'quality control', 'qc', 'preprocess', 'export', 'resipy',
    ]
    ert_processing = bool(config.get('ert_data_processing')) or (
        bool(config.get('ert_file'))
        and any(kw in user_request_lower for kw in processing_keywords)
        and not any(kw in user_request_lower for kw in inversion_keywords)
    )

    # ── Priority order ───────────────────────────────────────────────────
    # 1. TDEM
    if (
        config.get('tdem_file')
        or config.get('tdem_data')
        or 'tdem' in user_request_lower
        or 'tem ' in user_request_lower
        or 'electromagnetic' in user_request_lower
    ):
        return 'tdem'

    # 2. Standalone seismic
    if (
        (config.get('raw_seismic_file') and not config.get('ert_file'))
        or (config.get('seismic_file') and not config.get('ert_file'))
        or config.get('seismic_only', False)
        or (
            'seismic' in user_request_lower
            and 'ert' not in user_request_lower
            and 'resistivity' not in user_request_lower
            and 'fusion' not in user_request_lower
        )
        or 'srt inversion' in user_request_lower
        or 'seismic refraction' in user_request_lower
        or 'travel time' in user_request_lower
    ):
        return 'seismic'

    # 3. Hydrological model output (MODFLOW / ParFlow)
    if config.get('hydro_model') in ['modflow', 'parflow', 'both'] or hydro_only:
        return 'model_output'

    # 4. Time-lapse ERT
    if not hydro_only and (
        'timelapse_files' in config_keys
        or 'time_lapse_files' in config_keys
        or 'timelapse_params' in config_keys
        or config.get('inversion_mode') == 'time-lapse'
    ):
        return 'time_lapse'

    # 5. Data fusion (ERT + seismic or explicit fusion config)
    if (
        config.get('velocity_threshold')
        or (config.get('ert_file') and (config.get('seismic_file') or config.get('raw_seismic_file')))
        or (
            config.get('fusion_pattern')
            and config.get('fusion_pattern') not in [None, 'None', '']
        )
        or (
            config.get('methods')
            and len(config.get('methods', [])) > 1
            and 'seismic' in config.get('methods', [])
        )
        or config.get('use_seismic')
    ):
        return 'data_fusion'

    # 6. ERT data processing (QC / export, no inversion)
    if ert_processing:
        return 'ert_data_process'

    # 7. Standard / direct ERT
    if config.get('ert_file'):
        return 'direct_ert'

    # 8. Unknown: no workflow type matches, and the pipeline refuses the request
    return 'custom'


def run_legacy_agent_workflow(workflow_config, api_key, llm_model, llm_provider,
                              output_dir, progress_callback=None):
    """Infer the workflow type from the configuration and run its pipeline.

    Supported: data fusion, time-lapse, direct ERT conversion and the other
    types :func:`_detect_workflow_type` returns.

    Parameters
    ----------
    workflow_config : dict
        Configuration from ``ContextInputAgent``. Normalised in place, as it
        always was (``data_file`` is copied to ``ert_file``, a raw SEG-Y
        ``seismic_file`` becomes ``raw_seismic_file``).
    api_key, llm_model, llm_provider : str
        Model access for the agents the pipeline creates.
    output_dir : str or Path
        Where results are written; created when missing.
    progress_callback : callable, optional
        ``(step, progress, details)``.

    Returns
    -------
    tuple
        ``(results, execution_plan, interpretation, report_files)``.
    """
    def update_progress(step: str, progress: float, details: str = ""):
        """Update progress if callback is available."""
        if progress_callback:
            progress_callback(step, progress, details)
        print(f"[Progress {progress*100:.0f}%] {step}: {details}")

    # Normalize output directory so path joins work regardless of caller type.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Normalise key names ------------------------------------------------
    config_keys = set(workflow_config.keys())
    print(f'\nDetecting workflow type from config keys: {config_keys}')

    if 'data_file' in workflow_config and 'ert_file' not in workflow_config:
        workflow_config['ert_file'] = workflow_config['data_file']
        print("  → Normalized 'data_file' to 'ert_file'")

    if (
        workflow_config.get("seismic_file")
        and Path(str(workflow_config.get("seismic_file"))).suffix.lower() in {".sgy", ".segy"}
    ):
        workflow_config["raw_seismic_file"] = workflow_config.pop("seismic_file")
        workflow_config["raw_seismic_processing"] = True
        print("  -> Normalized raw SEG-Y seismic_file to raw_seismic_file")

    # 2. Classify the request
    workflow_type = _detect_workflow_type(workflow_config)

    print(f'\n===== WORKFLOW TYPE: {workflow_type.upper()} =====')
    execution_plan = None
    interpretation = None
    results = {}
    report_files = {}

    if workflow_type == 'data_fusion':
        # Use DataFusionAgent for complete workflow execution
        from .data_fusion_agent import DataFusionAgent
        fusion_agent = DataFusionAgent(
            api_key=api_key,
            model=llm_model,
            llm_provider=llm_provider
        )

        fusion_input = {
            'fusion_pattern': workflow_config.get('fusion_pattern', 'full_integration'),
            'methods': workflow_config.get('methods', ['seismic', 'ert', 'petrophysics']),
            'workflow_config': workflow_config,
            'data': {
                'seismic': workflow_config.get('seismic_file'),
                'ert': workflow_config.get('ert_file')
            },
            'output_dir': workflow_config.get('output_dir', str(output_dir))
        }

        update_progress("Planning data fusion workflow", 0.15, "Analyzing multi-method integration")
        print('Getting data fusion execution plan...')
        plan_result = fusion_agent.execute(fusion_input)
        execution_plan = plan_result.get('execution_plan')
        interpretation = plan_result.get('interpretation')

        print('\n' + '='*70)
        print('DATA FUSION EXECUTION PLAN')
        print('='*70)
        print(f"Pattern: {plan_result.get('fusion_pattern')}")
        print(f"\nInterpretation: {interpretation}")
        print(f"\nExecution Steps ({len(execution_plan)} total):")
        for i, step in enumerate(execution_plan, 1):
            print(f"\n  Step {i}: {step['step']}")
            print(f"    Agent: {step['agent']}")
            print(f"    Description: {step['description']}")
            print(f"    Outputs: {', '.join(step['outputs'])}")

        update_progress("Executing data fusion workflow", 0.30, "Processing seismic and ERT data")
        print('\nExecuting complete data fusion workflow...')
        results = fusion_agent.execute_full_workflow(fusion_input)
        update_progress("Data fusion complete", 0.75, "Generating report")

        # Update interpretation with detailed results
        if results.get('status') == 'success':
            # Build layer parameters summary
            layer_params = workflow_config.get('layer_params', {})
            params_summary = ""
            if layer_params:
                for layer_name, params in layer_params.items():
                    params_summary += f"\n  - {layer_name}: "
                    if 'rho_sat_range' in params:
                        params_summary += f"ρ_sat={params['rho_sat_range']}, "
                    if 'porosity_range' in params:
                        params_summary += f"φ={params['porosity_range']}, "
                    if 'n_range' in params:
                        params_summary += f"n={params['n_range']}"

            # Get water content stats if available
            wc_stats = ""
            if 'water_content_mean' in results:
                wc_mean = results['water_content_mean']
                wc_std = results.get('water_content_std', np.zeros_like(wc_mean))
                wc_stats = f"\n- Mean water content: {np.nanmean(wc_mean):.3f} ± {np.nanmean(wc_std):.3f}"
                wc_stats += f"\n- Water content range: {np.nanmin(wc_mean):.3f} - {np.nanmax(wc_mean):.3f}"

            interpretation = f"""Data Fusion workflow completed successfully.

**Multi-Method Integration:**
- Seismic file: {workflow_config.get('seismic_file', 'N/A')}
- ERT file: {workflow_config.get('ert_file', 'N/A')}
- Velocity threshold: {workflow_config.get('velocity_threshold', 1200)} m/s
- Fusion pattern: {workflow_config.get('fusion_pattern', 'full_integration')}

**Petrophysical Parameters:**{params_summary if params_summary else ' Default Archie parameters applied'}

**Water Content Results:**{wc_stats if wc_stats else ' Not computed'}

**Key Benefits:**
- Seismic-derived structure constrains ERT inversion
- Layer-specific petrophysics improves accuracy
- Monte Carlo provides uncertainty quantification
"""

        # Generate comprehensive report with layer-specific statistics
        if results.get('status') == 'success':
            from .report_agent import ReportAgent
            report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
            workflow_config['user_request'] = workflow_config.get('user_request', 'Data fusion workflow')

            # Add layer-specific statistics if available
            if 'cell_markers' in results and 'water_content_mean' in results:
                cell_markers = results['cell_markers']
                water_content_mean = results['water_content_mean'].ravel()
                unique_markers = np.unique(cell_markers)

                layer_stats = {}
                for layer_id in unique_markers:
                    mask = cell_markers == layer_id
                    layer_wc = water_content_mean[mask]

                    # Map layer ID to name (from petrophysics: marker 2=regolith, 3=bedrock)
                    layer_name = f"Layer {layer_id}"
                    for name in workflow_config.get('layer_params', {}).keys():
                        if layer_id == 2 and 'regolith' in name.lower():
                            layer_name = "Regolith"
                        elif layer_id == 3 and ('bedrock' in name.lower() or 'fractured' in name.lower()):
                            layer_name = "Fractured Bedrock"

                    layer_stats[str(layer_id)] = {
                        'name': layer_name,
                        'mean_wc': float(np.nanmean(layer_wc)),
                        'std_wc': float(np.nanstd(layer_wc)),
                        'min_wc': float(np.nanmin(layer_wc)),
                        'max_wc': float(np.nanmax(layer_wc)),
                        'n_cells': int(np.sum(mask))
                    }

                results['layer_statistics'] = layer_stats

            report_input = {
                'structure_results': results,
                'petro_results': results if 'water_content_mean' in results else None,
                'workflow_config': workflow_config,
                'output_dir': str(output_dir)
            }
            report_results = report_agent.generate_data_fusion_report(report_input)
            if report_results.get('status') == 'success':
                # Collect all report artifacts (md/html/pdf + visualizations)
                report_files = {}
                if report_results.get('report_file'):
                    report_files['report_markdown'] = report_results['report_file']
                if report_results.get('html_file'):
                    report_files['report_html'] = report_results['html_file']
                if report_results.get('pdf_file'):
                    report_files['report_pdf'] = report_results['pdf_file']
                for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                    report_files[f'visualization_{vis_name}'] = vis_path

    elif workflow_type == 'time_lapse':
        # Use ERT agents for time-lapse workflow
        from .ert_inversion_agent import ERTInversionAgent
        from .ert_loader_agent import ERTLoaderAgent
        from .inversion_evaluation_agent import InversionEvaluationAgent
        ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        update_progress("Starting time-lapse workflow", 0.15, "Loading multiple ERT datasets")
        print('Running time-lapse ERT workflow...')

        # Read the request first, then design the plan from what it asked
        # for, then dispatch exactly the agents the plan names. These two
        # flags gate both the plan below and the execution further down, so
        # a step is listed if and only if it runs. Hardcoding the plan let
        # it drift: it advertised a ClimateDataAgent step that configuration
        # had disabled, and omitted petrophysics for a request that asked
        # for water content in so many words.
        plan_climate = wants_climate(workflow_config)
        if plan_climate and coords_from_config(workflow_config) is None:
            # The reanalysis is sampled at a point, so a position is required. The
            # request parser extracts a place NAME; resolving it to a
            # position is a gazetteer's job, not the model's - a remembered
            # coordinate is confidently wrong often enough to put the whole
            # climate series in the wrong valley.
            place = workflow_config.get('site_location')
            located = geocode_place(place) if place else None
            if located:
                climate_config = dict(workflow_config.get('climate_config') or {})
                climate_config['coords'] = list(located['coords'])
                workflow_config['climate_config'] = climate_config
                workflow_config.setdefault('site_info', {})
                workflow_config['site_info']['location'] = located['matched_name']
                print(f"  Located '{place}' -> {located['matched_name']} "
                      f"at {located['coords']} (via {located['source']})")
            else:
                blocked = climate_blocker(workflow_config)
                print(f"  Climate requested but cannot run: {blocked}")
                plan_climate = False
        plan_water_content = wants_water_content(workflow_config)
        print(f'Plan: climate={plan_climate}, water_content={plan_water_content}')

        execution_plan = [
            {'step': 'Load Time-Lapse ERT Data', 'agent': 'ERTLoaderAgent',
             'description': 'Load multiple ERT datasets for time-lapse monitoring',
             'outputs': ['ert_data_list']},
        ]
        if plan_climate:
            execution_plan.append(
                {'step': 'Fetch Climate Data', 'agent': 'ClimateDataAgent',
                 'description': 'Fetch meteorological data (precipitation, temperature, PET)',
                 'outputs': ['climate_data']})
        execution_plan += [
            {'step': 'Time-Lapse Inversion', 'agent': 'ERTInversionAgent',
             'description': 'Run time-lapse inversion with temporal regularization',
             'outputs': ['resistivity_changes', 'temporal_models']},
            {'step': 'Evaluate Inversion Quality', 'agent': 'InversionEvaluationAgent',
             'description': 'Assess inversion quality and optimize parameters if needed',
             'outputs': ['quality_metrics', 'optimized_results']},
        ]
        if plan_water_content:
            execution_plan.append(
                {'step': 'Convert to Water Content', 'agent': 'PetrophysicsAgent',
                 'description': 'Apply petrophysics with Monte Carlo to every time step',
                 'outputs': ['water_content', 'uncertainty']})
        execution_plan.append(
            {'step': 'Generate Time-Lapse Report', 'agent': 'ReportAgent',
             'description': ('Create report with water content and climate correlation'
                             if plan_water_content and plan_climate else
                             'Create report from the recovered models'),
             'outputs': ['report', 'visualizations']})

        # Initial interpretation - will be updated with actual results later
        interpretation = None  # Will be set after inversion completes

        # Load time-lapse data files (check both naming conventions)
        time_lapse_files = workflow_config.get('time_lapse_files') or workflow_config.get('timelapse_files', [])
        if not time_lapse_files:
            raise ValueError('No time-lapse files specified in configuration')

        print(f'  → Found {len(time_lapse_files)} time-lapse files')

        # Get electrode file for topography (if provided)
        electrode_file = workflow_config.get('electrode_file')
        if electrode_file:
            electrode_file_path = Path(electrode_file)
            project_dir = workflow_config.get('project_dir', '.')

            # Normalize electrode file path
            if not electrode_file_path.exists():
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / electrode_file_path.name
                    if combined_path.exists():
                        electrode_file_path = combined_path
                    else:
                        if len(combined_path.parts) > 0 and combined_path.parts[0] == 'examples':
                            combined_path = Path(*combined_path.parts[1:])
                            if combined_path.exists():
                                electrode_file_path = combined_path
                if not electrode_file_path.exists() and len(electrode_file_path.parts) > 0 and electrode_file_path.parts[0] == 'examples':
                    alt_path = Path(*electrode_file_path.parts[1:])
                    if alt_path.exists():
                        electrode_file_path = alt_path

            electrode_file = str(electrode_file_path)
            print(f'  → Using electrode file: {electrode_file_path.name}')

        time_lapse_data = []
        for i, data_file in enumerate(time_lapse_files):
            # Normalize each time-lapse file path
            data_file_path = Path(data_file)
            project_dir = workflow_config.get('project_dir', '.')

            if not data_file_path.exists():
                # Try combining with project_dir
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / data_file_path.name
                    if combined_path.exists():
                        data_file_path = combined_path
                        print(f'  → Resolved: {project_dir} + {data_file_path.name}')
                    else:
                        # Handle duplicate 'examples/' prefix
                        if len(combined_path.parts) > 0 and combined_path.parts[0] == 'examples':
                            combined_path = Path(*combined_path.parts[1:])
                            if combined_path.exists():
                                data_file_path = combined_path

                # Try removing 'examples/' prefix if present
                if not data_file_path.exists() and len(data_file_path.parts) > 0 and data_file_path.parts[0] == 'examples':
                    alt_path = Path(*data_file_path.parts[1:])
                    if alt_path.exists():
                        data_file_path = alt_path
                        print(f'  → Removed examples/ prefix')

            data_file = str(data_file_path)
            print(f'Loading dataset {i+1}/{len(time_lapse_files)}: {Path(data_file).name}')

            result = ert_loader.execute({
                'data_file': data_file,
                'instrument': workflow_config.get('instrument', 'E4D'),
                'project_dir': workflow_config.get('project_dir', '.'),
                'electrode_file': electrode_file,  # Pass electrode file for topography
                'crs': workflow_config.get('crs', 'local')
            })
            if result['status'] != 'success':
                print(f'Failed to load {data_file}: {result.get("error")}')
                continue
            time_lapse_data.append(result['ert_data'])

        if len(time_lapse_data) < 2:
            raise ValueError(f'Need at least 2 datasets for time-lapse, got {len(time_lapse_data)}')

        update_progress("Data loaded", 0.30, f"Loaded {len(time_lapse_data)} time-lapse datasets")

        # Fetch climate data if requested
        climate_results = None
        # The flag the plan was built from, so the listed step and the
        # executed one cannot disagree.
        if plan_climate:
            print('\nFetching climate data for correlation analysis...')
            from .climate_data_agent import ClimateDataAgent

            climate_agent = ClimateDataAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            climate_config = workflow_config.get('climate_config', {})
            if climate_config:
                import re
                from datetime import datetime, timedelta

                # Survey dates from the file names. The series spans them with
                # a month either side, for the figures.
                ert_dates = []
                for fname in time_lapse_files:
                    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(fname))
                    if match:
                        ert_dates.append(match.group(1))
                dates = climate_config.get('dates')
                if ert_dates:
                    first = datetime.strptime(min(ert_dates), '%Y-%m-%d')
                    last = datetime.strptime(max(ert_dates), '%Y-%m-%d')
                    dates = ((first - timedelta(days=30)).strftime('%Y-%m-%d'),
                             (last + timedelta(days=30)).strftime('%Y-%m-%d'))

                climate_results = climate_agent.execute({
                    'coords': climate_config.get('coords'),
                    'dates': dates,
                    'crs': climate_config.get('crs', 4326),
                    'pet_method': climate_config.get('pet_method', 'penman_monteith'),
                    'ert_timestamps': ert_dates or None,
                    'output_dir': str(output_dir / 'climate'),
                })
                if climate_results.get('status') == 'failed':
                    print(f"  Climate data could not be retrieved: {climate_results.get('error', 'unknown error')}")
                else:
                    workflow_config['climate_data'] = climate_results
                    print(f"  Climate data retrieved: {climate_results['metadata']['source']}")

        # Run time-lapse inversion
        update_progress("Running time-lapse inversion", 0.45, "This may take several minutes...")
        inversion_input = {
            'time_lapse_data': time_lapse_data,
            # The loaded containers no longer know where they came from, and
            # the acquisition time is in the file name: without these the run
            # falls back to a 1..n index and cannot report an interval.
            'source_files': list(time_lapse_files),
            'inversion_mode': 'time-lapse',
            'time_lapse_method': workflow_config.get('time_lapse_method', IMPLEMENTED_SCHEME),
            'temporal_regularization': workflow_config.get('temporal_regularization', 10.0),
            'baseline_index': 0,
            'inversion_params': workflow_config.get('inversion_params', {
                'lambda': 15.0,
                'max_iterations': 10
            }),
            'output_dir': str(output_dir / 'inversion')
        }

        print('\nRunning time-lapse inversion...')
        results = ert_inversion.execute(inversion_input)
        update_progress("Inversion complete", 0.65, f"Processed {results.get('n_timesteps', 'N/A')} time steps")

        print(f"  → Inversion status: {results.get('status')}")
        if results.get('status') == 'success':
            print(f"  → Number of timesteps: {results.get('n_timesteps', 'N/A')}")
            print(f"  → Chi² values: {results.get('chi2_values', 'N/A')}")

        # Evaluate and optimize inversion quality if successful
        evaluation_results = None
        if results.get('status') == 'success':
            print('Evaluating inversion quality and optimizing parameters...')
            eval_input = {
                'inversion_results': results,
                'ert_data': time_lapse_data[0],  # Use baseline data for evaluation
                'time_lapse_data': time_lapse_data,
                'inversion_mode': 'time-lapse',
                'inversion_params': workflow_config.get('inversion_params', {
                    'lambda': 15.0,
                    'max_iterations': 10
                }),
                'auto_adjust': workflow_config.get('auto_adjust', True),
                'output_dir': str(output_dir / 'inversion'),
                'max_attempts': workflow_config.get('max_attempts', 3),
                'quality_threshold': workflow_config.get('quality_threshold', 70),
                'progress_callback': progress_callback,
                'project_dir': workflow_config.get('project_dir', 'data/ERT/E4D'),
                'instrument': workflow_config.get('instrument', 'E4D')
            }

            evaluation_results = eval_agent.execute(eval_input)

            # Update results if optimization improved them
            if evaluation_results.get('status') == 'success' and evaluation_results.get('attempts', 1) > 1:
                print('✓ Inversion was optimized! Using improved results.')
                results = evaluation_results['final_results']
            # Carry the evaluation on the results, not only into the report:
            # the audit's quality block and the runner's "needs review"
            # warning both read it here, so without this a 54.8/100 run was
            # published as a clean success with no caveat anywhere.
            if isinstance(results, dict):
                # Without final_results: that key holds this same dict, and
                # the self-reference makes the results unserialisable - the
                # audit writer stops with "Circular reference detected" after
                # the whole workflow has already succeeded. The audit wants
                # the verdict, not a second copy of the models.
                results['evaluation_results'] = {
                    key: value for key, value in evaluation_results.items()
                    if key != 'final_results'}

        # Convert every time step to water content when the request asked
        # for it. Only the single-survey branch used to do this, so a
        # time-lapse request for water content returned a resistivity report
        # with the product silently missing.
        if results.get('status') == 'success' and plan_water_content:
            from .petrophysics_agent import PetrophysicsAgent
            update_progress("Converting to water content", 0.70,
                            "Running Monte Carlo petrophysics per time step")
            print('Converting resistivity to water content...')
            petro_agent = PetrophysicsAgent(api_key=api_key, model=llm_model,
                                            llm_provider=llm_provider)
            mesh = results.get('mesh')
            models = results.get('time_lapse_models') or []
            if mesh is not None:
                cell_markers = np.array(mesh.cellMarkers())
            else:
                cell_markers = np.zeros(len(models[0]) if models else 0)
            per_step = []
            for index, model in enumerate(models):
                petro = petro_agent.execute({
                    'resistivity_model': model,
                    'mesh': mesh,
                    'cell_markers': cell_markers,
                    'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                    'n_realizations': workflow_config.get('n_realizations', 100),
                    'geological_context': workflow_config.get('geological_context',
                                                              'generic watershed'),
                    'output_dir': str(output_dir / 'petrophysics' / f'timestep_{index + 1}'),
                })
                if petro.get('status') != 'success':
                    # One failed step must not discard the ones that worked,
                    # and must not be reported as if nothing was asked for.
                    print(f"  ⚠ Water content failed at time step {index + 1}: "
                          f"{petro.get('error')}")
                    break
                per_step.append(petro)
                print(f"  → Water content for time step {index + 1}/{len(models)}")
            if per_step:
                results['time_lapse_water_content'] = per_step
                results['water_content_mean'] = per_step[0].get('water_content_mean')
                results['water_content_std'] = per_step[0].get('water_content_std')
                results['petrophysical_params'] = workflow_config.get(
                    'petrophysical_params', {})
                update_progress("Water content complete", 0.78,
                                f"{len(per_step)} of {len(models)} time steps converted")

        # Build detailed interpretation after inversion
        if results.get('status') == 'success':
            n_timesteps = results.get('n_timesteps', len(time_lapse_data))
            from ._chi2 import chi2_summary as _chi2_summary
            # chi2_values is one row per iteration here, each holding the
            # three objective terms; min()/max() over the rows returns a row,
            # which no float format string accepts.
            chi2_summary = _chi2_summary(results.get('chi2_values', []))

            interpretation = f"""Time-lapse ERT monitoring workflow completed successfully.

**Survey Summary:**
- Number of time steps: {n_timesteps}
- Data files processed: {len(time_lapse_files)}

**Inversion Results:**
- Chi-squared (final): {chi2_summary}
- Temporal regularization: {workflow_config.get('temporal_regularization', 10.0)}
- Inversion method: {workflow_config.get('time_lapse_method', IMPLEMENTED_SCHEME)}

**Climate Integration:**
- Climate data: {'Available' if workflow_config.get('climate_data') else 'Not requested'}

**Key Findings:**
Time-lapse resistivity changes capture subsurface moisture dynamics over the monitoring period.
Decreasing resistivity indicates increased soil moisture (wetting events).
Increasing resistivity indicates soil drying (evapotranspiration or drainage).
"""

        # Generate comprehensive report with climate integration if available
        if results.get('status') == 'success':
            from .report_agent import ReportAgent
            report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Prepare comprehensive report input
            climate_config = workflow_config.get('climate_config', {})
            dates = climate_config.get('dates', ['N/A', 'N/A']) if climate_config else ['N/A', 'N/A']

            # Ensure dates is a list and extract start/end as strings
            if not isinstance(dates, list):
                dates = ['N/A', 'N/A']
            start_date = str(dates[0]) if len(dates) > 0 else 'N/A'
            end_date = str(dates[-1]) if len(dates) > 1 else 'N/A'

            # Get site coordinates
            site_info_config = workflow_config.get('site_info', {})
            coordinates_str = str(site_info_config.get('coordinates', 'N/A'))
            if coordinates_str == 'N/A':
                # Try to get from climate_config
                coords_list = climate_config.get('coords', [])
                if coords_list and len(coords_list) == 2:
                    coordinates_str = f"{coords_list[1]:.5f}°N, {coords_list[0]:.5f}°W"

            # Prepare comparison DataFrame for climate-resistivity correlation if climate data available
            comparison_df = None
            if workflow_config.get('climate_data'):
                import pandas as pd

                climate_results = workflow_config['climate_data']
                if climate_results.get('ert_alignment') and 'ert_aligned_data' in climate_results['ert_alignment']:
                    aligned_df = climate_results['ert_alignment']['ert_aligned_data']

                    # Calculate mean resistivity changes for each time step
                    final_models = results.get('final_models')
                    if final_models is not None:
                        baseline = final_models[:, 0]
                        resistivity_changes = []
                        for i in range(1, final_models.shape[1]):
                            change = final_models[:, i] - baseline
                            mean_change = np.mean(change)
                            resistivity_changes.append(mean_change)

                        # Create comparison dataframe
                        prcp_vals = aligned_df.get('prcp', [0] * (len(aligned_df) - 1))[1:]
                        tmin_vals = aligned_df.get('tmin', [0] * (len(aligned_df) - 1))[1:]
                        tmax_vals = aligned_df.get('tmax', [0] * (len(aligned_df) - 1))[1:]
                        pet_vals = aligned_df.get('pet', [0] * (len(aligned_df) - 1))[1:]

                        # Convert dates to strings for report compatibility
                        date_strings = [str(dt.date()) if hasattr(dt, 'date') else str(dt) for dt in aligned_df.index[1:]]

                        comparison_df = pd.DataFrame({
                            'Date': date_strings,
                            'Mean_Resistivity_Change_Ohm_m': resistivity_changes[:len(aligned_df)-1],
                            'Precipitation_mm': prcp_vals,
                            'Temp_Min_C': tmin_vals,
                            'Temp_Max_C': tmax_vals,
                            'Temp_Mean_C': (np.array(tmin_vals) + np.array(tmax_vals)) / 2,
                            'PET_mm': pet_vals
                        })

            # The survey dates are in the file names the run was given, so
            # reporting "N/A to N/A" for the study period threw away
            # something already in hand. Any date the caller supplied wins.
            stamps = _dates_from_filenames(time_lapse_files)
            if start_date == 'N/A' or end_date == 'N/A':
                if stamps:
                    # min/max, not first/last: the labels follow the file
                    # order, which is the survey order but need not be
                    # the order the caller listed them in.
                    start_date, end_date = min(stamps), max(stamps)

            site_info = {
                # Every survey date, not just the endpoints: the report
                # labels its rows with them, so "Survey 4" says when.
                'survey_dates': stamps,
                'name': str(site_info_config.get('name', 'Time-Lapse ERT Monitoring Site')),
                'location': str(site_info_config.get('location', coordinates_str)),
                'coordinates': str(coordinates_str),
                'elevation': str(site_info_config.get('elevation', 'N/A')),
                'study_period': f"{start_date} to {end_date}",
                # Says what this run did. The fixed wording advertised
                # climate integration on runs that had no climate data.
                'description': str(
                    'Time-lapse ERT monitoring with climate integration for '
                    'subsurface moisture dynamics.' if plan_climate else
                    'Time-lapse ERT monitoring of subsurface resistivity change.')
            }

            report_input = {
                'inversion_results': results,
                'climate_data': workflow_config.get('climate_data'),
                'site_info': site_info,
                'comparison_data': comparison_df,
                'evaluation_results': evaluation_results,
                'workflow_config': workflow_config,
                'time_lapse_method': workflow_config.get('time_lapse_method', IMPLEMENTED_SCHEME),
                'output_dir': str(output_dir)
            }

            update_progress("Generating report", 0.85, "Creating visualizations and analysis")
            print('\n' + '='*70)
            print('GENERATING TIME-LAPSE REPORT')
            print('='*70)
            print(f"  → Output directory: {output_dir}")
            print(f"  → Time-lapse method: {workflow_config.get('time_lapse_method', IMPLEMENTED_SCHEME)}")
            print(f"  → Climate data available: {workflow_config.get('climate_data') is not None}")

            report_results = report_agent.generate_timelapse_report(report_input)
            update_progress("Report complete", 0.95, "Saving files")

            if report_results.get('status') == 'success':
                report_files = {}
                if report_results.get('report_file'):
                    report_files['report_markdown'] = report_results['report_file']
                if report_results.get('html_file'):
                    report_files['report_html'] = report_results['html_file']
                if report_results.get('pdf_file'):
                    report_files['report_pdf'] = report_results['pdf_file']
                for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                    report_files[f'visualization_{vis_name}'] = vis_path
                print("\nReport generation completed successfully!")
                print(f"  Generated {len(report_results.get('visualization_files', {}))} visualization files")
                print(f"  Report file: {report_results.get('report_file')}")
                print(f"  HTML file: {report_results.get('html_file')}")
            else:
                print(f"\nReport generation failed: {report_results.get('error', 'Unknown error')}")
                print("  Check logs for details")

    elif workflow_type == 'ert_data_process':
        # ERT data processing workflow (QC + export)
        from PyHydroGeophysX.data_processing.ert_data_agent import export_for_inversion

        from .ert_loader_agent import ERTLoaderAgent

        ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        data_file = workflow_config.get('ert_file') or workflow_config.get('data_file')
        if not data_file:
            raise ValueError("No ERT data file specified for processing.")

        # Normalize file path
        data_file_path = Path(data_file)
        project_dir = workflow_config.get('project_dir', '.')
        if not data_file_path.exists():
            if project_dir and project_dir != '.':
                combined_path = Path(project_dir) / data_file_path.name
                if combined_path.exists():
                    data_file_path = combined_path
                elif combined_path.parts and combined_path.parts[0] == 'examples':
                    combined_path = Path(*combined_path.parts[1:])
                    if combined_path.exists():
                        data_file_path = combined_path
            if not data_file_path.exists() and data_file_path.parts and data_file_path.parts[0] == 'examples':
                alt_path = Path(*data_file_path.parts[1:])
                if alt_path.exists():
                    data_file_path = alt_path

        electrode_file = workflow_config.get('electrode_file')
        if electrode_file:
            electrode_file_path = Path(electrode_file)
            if not electrode_file_path.exists():
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / electrode_file_path.name
                    if combined_path.exists():
                        electrode_file_path = combined_path
                    elif combined_path.parts and combined_path.parts[0] == 'examples':
                        combined_path = Path(*combined_path.parts[1:])
                        if combined_path.exists():
                            electrode_file_path = combined_path
                if not electrode_file_path.exists() and electrode_file_path.parts and electrode_file_path.parts[0] == 'examples':
                    alt_path = Path(*electrode_file_path.parts[1:])
                    if alt_path.exists():
                        electrode_file_path = alt_path
            electrode_file = str(electrode_file_path)

        update_progress("Loading ERT data", 0.20, f"File: {data_file_path.name}")
        load_result = ert_loader.execute({
            'data_file': str(data_file_path),
            'instrument': workflow_config.get('instrument', 'E4D'),
            'project_dir': workflow_config.get('project_dir', str(data_file_path.parent)),
            'electrode_file': electrode_file,
            'crs': workflow_config.get('crs', 'local'),
            'quality_check': True,
            'output_dir': str(output_dir / 'ert_processing')
        })

        if load_result.get('status') != 'success':
            raise ValueError(f"ERT data processing failed: {load_result.get('error')}")

        ert_data = load_result.get('ert_data')
        qc_artifacts = ert_loader.get_context('qc_artifacts') or {}

        export_requested = workflow_config.get('export_for_inversion')
        if export_requested is None:
            # Never defined in this function before: every legacy ERT
            # processing run without export_for_inversion hit a NameError.
            user_request_lower = str(workflow_config.get('user_request', '')).lower()
            export_requested = 'export' in user_request_lower or 'bert' in user_request_lower or 'pgimli' in user_request_lower

        export_file = None
        if export_requested and ert_data is not None:
            update_progress("Exporting data", 0.40, "Preparing inversion-ready format")
            export_format = workflow_config.get('export_format', 'pgimli')
            export_file = export_for_inversion(ert_data, outdir=str(output_dir / 'ert_processing'), fmt=export_format)

        results = {
            'status': 'success',
            'ert_data': ert_data,
            'qc_results': load_result.get('qc_results'),
            'qc_artifacts': qc_artifacts,
            'export_file': export_file,
            'output_dir': str(output_dir / 'ert_processing')
        }

        execution_plan = [
            {'step': 'Load ERT Data', 'agent': 'ERTLoaderAgent',
             'description': 'Load and validate ERT data', 'outputs': ['ert_data']},
            {'step': 'QC Diagnostics', 'agent': 'ERTLoaderAgent',
             'description': 'Generate quality control artifacts', 'outputs': ['qc_plots']},
            {'step': 'Export', 'agent': 'ERTLoaderAgent',
             'description': 'Export data for inversion', 'outputs': ['bert_data.dat']},
        ]

        interpretation = "ERT data processing completed. QC artifacts and export files are ready."

        report_text = f"""# ERT Data Processing Report

**Data file:** `{data_file_path.name}`
**Instrument:** {workflow_config.get('instrument', 'E4D')}
**Electrodes:** {len(ert_data.electrodes) if ert_data else 'N/A'}
**Measurements:** {len(ert_data.observations) if ert_data else 'N/A'}

## QC Artifacts
{chr(10).join(f"- {k}: {v}" for k, v in (qc_artifacts or {}).items()) if qc_artifacts else "No QC artifacts generated."}

## Export
{f"Exported file: `{export_file}`" if export_file else "Export not requested."}
"""
        report_file = output_dir / 'ert_processing_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        report_files = {'report_markdown': str(report_file)}
        from .report_agent import ReportAgent
        report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        html_file = report_agent._save_html_report(report_text, str(output_dir), 'ert_processing_report')
        if html_file:
            report_files['report_html'] = html_file

    elif workflow_type == 'model_output':
        # Hydrological model output loading (MODFLOW / ParFlow)
        from .model_output_agent import ModelOutputAgent
        model_agent = ModelOutputAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        update_progress("Loading hydrological model outputs", 0.20, "Reading MODFLOW/ParFlow data")
        model_input = {**workflow_config, 'output_dir': str(output_dir / 'model_output')}
        results = model_agent.execute(model_input)

        execution_plan = []
        if results.get('modflow'):
            execution_plan.append({'step': 'Load MODFLOW Outputs', 'agent': 'ModelOutputAgent',
                                   'description': 'Load water content and porosity from MODFLOW',
                                   'outputs': ['water_content', 'porosity']})
        if results.get('parflow'):
            execution_plan.append({'step': 'Load ParFlow Outputs', 'agent': 'ModelOutputAgent',
                                   'description': 'Load saturation and porosity from ParFlow',
                                   'outputs': ['saturation', 'porosity']})

        interpretation = "Hydrological model outputs loaded successfully."
        if results.get('warnings'):
            interpretation += f" Warnings: {', '.join(results['warnings'])}"

        report_text = "# Hydrological Model Output Report\n\n"
        if results.get('modflow'):
            mod = results['modflow']
            report_text += "## MODFLOW\n"
            report_text += f"- Model directory: `{mod.get('model_directory')}`\n"
            report_text += f"- Water content file: `{mod.get('water_content_file')}`\n"
            if mod.get('porosity_file'):
                report_text += f"- Porosity file: `{mod.get('porosity_file')}`\n"
            if mod.get('resistivity_file'):
                report_text += f"- Resistivity file: `{mod.get('resistivity_file')}`\n"
            if mod.get('resistivity_stats'):
                rs = mod['resistivity_stats']
                report_text += f"- Resistivity stats: min={rs.get('min')}, max={rs.get('max')}, mean={rs.get('mean')}\n"
            report_text += "\n"
            if mod.get('plots'):
                for _, path in mod['plots'].items():
                    report_text += f"![MODFLOW Plot]({Path(path).name})\n\n"
        if results.get('parflow'):
            par = results['parflow']
            report_text += "## ParFlow\n"
            report_text += f"- Model directory: `{par.get('model_directory')}`\n"
            report_text += f"- Saturation file: `{par.get('saturation_file')}`\n"
            report_text += f"- Porosity file: `{par.get('porosity_file')}`\n"
            if par.get('mask_file'):
                report_text += f"- Mask file: `{par.get('mask_file')}`\n"
            if par.get('resistivity_file'):
                report_text += f"- Resistivity file: `{par.get('resistivity_file')}`\n"
            if par.get('resistivity_stats'):
                rs = par['resistivity_stats']
                report_text += f"- Resistivity stats: min={rs.get('min')}, max={rs.get('max')}, mean={rs.get('mean')}\n"
            report_text += "\n"
            if par.get('plots'):
                for _, path in par['plots'].items():
                    report_text += f"![ParFlow Plot]({Path(path).name})\n\n"

        report_file = output_dir / 'hydro_model_output_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        report_files = {'report_markdown': str(report_file)}
        from .report_agent import ReportAgent
        report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        html_file = report_agent._save_html_report(report_text, str(output_dir), 'hydro_model_output_report')
        if html_file:
            report_files['report_html'] = html_file

    elif workflow_type == 'direct_ert':
        # Use ERT agents for direct ERT workflow
        from .ert_inversion_agent import ERTInversionAgent
        from .ert_loader_agent import ERTLoaderAgent
        from .inversion_evaluation_agent import InversionEvaluationAgent
        ert_loader = ERTLoaderAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        ert_inversion = ERTInversionAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)
        eval_agent = InversionEvaluationAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        # Detect if user wants water content conversion
        # Check explicit flag or presence of petrophysical parameters
        # One intent decision, shared with the time-lapse branch, so the two
        # cannot disagree about what the same sentence meant. The keyword
        # lists this replaces were exact substring matches, and the typo in
        # "estimate the water conent" matched none of them.
        skip_petrophysics = not wants_water_content(workflow_config)

        if skip_petrophysics:
            print('Running ERT inversion workflow (no water content conversion)...')
            update_progress("Running ERT inversion", 0.20, "ERT-only mode detected")
        else:
            print('Running direct ERT to water content workflow...')
            update_progress("Running ERT to water content workflow", 0.20, "Full conversion mode")

        # Load ERT data
        data_file = workflow_config.get('ert_file')
        if not data_file:
            raise ValueError('No ERT file specified in configuration')

        # Normalize path: handle various scenarios
        data_file_path = Path(data_file)
        project_dir = workflow_config.get('project_dir', '.')

        # Try to find the file in different locations
        if not data_file_path.exists():
            # Scenario 1: Try combining project_dir + data_file
            if project_dir and project_dir != '.':
                combined_path = Path(project_dir) / data_file_path.name
                if combined_path.exists():
                    data_file_path = combined_path
                    print(f'  → Combined path: {project_dir} + {data_file_path.name} = {data_file_path}')
                else:
                    # Scenario 2: Maybe project_dir has 'examples/' prefix but we're in examples/
                    if combined_path.parts[0] == 'examples':
                        combined_path = Path(*combined_path.parts[1:])
                        if combined_path.exists():
                            data_file_path = combined_path
                            print(f'  → Fixed duplicate: {data_file_path}')

            # Scenario 3: Try removing 'examples/' prefix from original path
            if not data_file_path.exists() and data_file_path.parts[0] == 'examples':
                alt_path = Path(*data_file_path.parts[1:])
                if alt_path.exists():
                    data_file_path = alt_path
                    print(f'  → Removed examples/ prefix: {data_file_path}')

        data_file = str(data_file_path)

        # Update project_dir to match the data file location
        if not project_dir or project_dir == '.':
            workflow_config['project_dir'] = str(data_file_path.parent)
        else:
            # Make sure project_dir matches where the file actually is
            workflow_config['project_dir'] = str(data_file_path.parent)

        update_progress("Loading ERT data", 0.25, f"File: {data_file_path.name}")

        # Handle electrode file if provided
        electrode_file = workflow_config.get('electrode_file')
        if electrode_file:
            electrode_file_path = Path(electrode_file)
            # Normalize electrode file path similar to data_file
            if not electrode_file_path.exists():
                # Try combining with project_dir
                if project_dir and project_dir != '.':
                    combined_path = Path(project_dir) / electrode_file_path.name
                    if combined_path.exists():
                        electrode_file_path = combined_path
                    elif combined_path.parts[0] == 'examples':
                        combined_path = Path(*combined_path.parts[1:])
                        if combined_path.exists():
                            electrode_file_path = combined_path
                # Try removing 'examples/' prefix
                if not electrode_file_path.exists() and electrode_file_path.parts[0] == 'examples':
                    alt_path = Path(*electrode_file_path.parts[1:])
                    if alt_path.exists():
                        electrode_file_path = alt_path
            electrode_file = str(electrode_file_path)
            print(f'  → Electrode file: {electrode_file_path.name}')

        print(f'Loading ERT data: {Path(data_file).name}')
        load_result = ert_loader.execute({
            'data_file': data_file,
            'instrument': workflow_config.get('instrument', 'DAS-1'),
            'project_dir': workflow_config.get('project_dir', '.'),
            'electrode_file': electrode_file,
            'crs': workflow_config.get('crs', 'local')
        })

        if load_result['status'] != 'success':
            raise ValueError(f'Failed to load ERT data: {load_result.get("error")}')

        ert_data = load_result['ert_data']
        update_progress("Data loaded successfully", 0.35, f"{len(ert_data.electrodes)} electrodes, {len(ert_data.observations)} measurements")

        # Run inversion
        update_progress("Running ERT inversion", 0.40, "This may take a few minutes...")
        inversion_input = {
            'ert_data': ert_data,
            'instrument': workflow_config.get('instrument', 'DAS-1'),
            'project_dir': workflow_config.get('project_dir', '.'),
            'output_dir': str(output_dir / 'inversion'),
            'inversion_params': workflow_config.get('inversion_params', {
                'lambda': 20.0,
                'max_iterations': 12
            })
        }

        inversion_results = ert_inversion.execute(inversion_input)

        if inversion_results.get('status') != 'success':
            raise ValueError(f'Inversion failed: {inversion_results.get("error")}')

        chi2_value = inversion_results.get('chi2', 'N/A')
        update_progress("Inversion complete", 0.55, f"Chi² = {chi2_value}")

        # Evaluate inversion quality
        evaluation_results = None
        if inversion_results.get('status') == 'success':
            update_progress("Evaluating inversion quality", 0.60, "Checking convergence and data fit")
            eval_input = {
                **inversion_input,
                'inversion_results': inversion_results,
                'ert_data': ert_data,
                'auto_adjust': workflow_config.get('auto_adjust', True),
                'quality_threshold': workflow_config.get('quality_threshold', 70),
                'max_attempts': workflow_config.get('max_attempts', 3),
                'progress_callback': progress_callback,
            }
            evaluation_results = eval_agent.execute(eval_input)

            if evaluation_results.get('final_results'):
                inversion_results = evaluation_results['final_results']
            if evaluation_results.get('status') == 'failed':
                raise ValueError(f"Inversion evaluation failed: {evaluation_results.get('error')}")

        # If skipping petrophysics (ERT-only mode)
        if skip_petrophysics:
            update_progress("Preparing ERT results", 0.75, "Skipping water content conversion")
            results = {
                'status': 'success',
                'ert_data': ert_data,
                'inversion_results': inversion_results,
                'evaluation_results': evaluation_results,
                'skip_petrophysics': True
            }

            # Set execution plan for ERT-only workflow
            execution_plan = [
                {'step': 'Load ERT Data', 'agent': 'ERTLoaderAgent', 
                 'description': 'Load ERT data file', 'outputs': ['ert_data']},
                {'step': 'Run Inversion', 'agent': 'ERTInversionAgent', 
                 'description': 'Invert for resistivity model', 'outputs': ['resistivity_model', 'mesh']},
                {'step': 'Evaluate Quality', 'agent': 'InversionEvaluationAgent', 
                 'description': 'Assess inversion quality', 'outputs': ['quality_score']},
                {'step': 'Generate Report', 'agent': 'ReportAgent', 
                 'description': 'Create summary report', 'outputs': ['report']}
            ]
            interpretation = "ERT inversion completed. Resistivity model generated without water content conversion."
        else:
            from .petrophysics_agent import PetrophysicsAgent
            petrophysics_agent = PetrophysicsAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Convert to water content
            update_progress("Converting to water content", 0.65, "Running Monte Carlo petrophysics")

            # Get mesh and cell markers from inversion results
            mesh = inversion_results.get('mesh')
            cell_markers = np.array(mesh.cellMarkers()) if mesh else np.zeros(len(inversion_results.get('resistivity_model', [])))

            petro_input = {
                'resistivity_model': inversion_results.get('resistivity_model'),
                'mesh': mesh,
                'cell_markers': cell_markers,
                'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                'n_realizations': workflow_config.get('n_realizations', 100),
                'geological_context': workflow_config.get('geological_context', 'generic watershed'),
                'output_dir': str(output_dir / 'petrophysics')
            }

            petro_results = petrophysics_agent.execute(petro_input)

            if petro_results.get('status') != 'success':
                raise ValueError(f'Petrophysics conversion failed: {petro_results.get("error")}')

            update_progress("Petrophysics complete", 0.75, "Water content model generated")

            # Combine results
            results = {
                'status': 'success',
                'ert_data': ert_data,
                'inversion_results': inversion_results,
                'evaluation_results': evaluation_results,
                'petrophysics_results': petro_results,
                'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                'water_content_mean': petro_results.get('water_content_mean'),
                'water_content_std': petro_results.get('water_content_std'),
                'skip_petrophysics': False
            }

            # Set execution plan for full workflow
            execution_plan = [
                {'step': 'Load ERT Data', 'agent': 'ERTLoaderAgent', 
                 'description': 'Load ERT data file', 'outputs': ['ert_data']},
                {'step': 'Run Inversion', 'agent': 'ERTInversionAgent', 
                 'description': 'Invert for resistivity model', 'outputs': ['resistivity_model', 'mesh']},
                {'step': 'Evaluate Quality', 'agent': 'InversionEvaluationAgent', 
                 'description': 'Assess inversion quality', 'outputs': ['quality_score']},
                {'step': 'Convert to Water Content', 'agent': 'PetrophysicsAgent', 
                 'description': 'Apply petrophysics with Monte Carlo', 'outputs': ['water_content', 'uncertainty']},
                {'step': 'Generate Report', 'agent': 'ReportAgent', 
                 'description': 'Create comprehensive report', 'outputs': ['report']}
            ]

            # Build detailed interpretation including petrophysical parameters used
            layer_params_used = petro_results.get('layer_params_used', {})
            params_summary = ""
            if layer_params_used:
                for layer_name, params in layer_params_used.items():
                    params_summary += f"\n  - {layer_name}: "

                    # Helper function to extract value (handles both dict with 'mean' and plain float)
                    def get_param_value(param):
                        if isinstance(param, dict):
                            return param.get('mean', param.get('value', 0))
                        return param

                    if 'rho_sat' in params:
                        val = get_param_value(params['rho_sat'])
                        params_summary += f"ρ_sat={val:.1f}Ωm, "
                    if 'porosity' in params:
                        val = get_param_value(params['porosity'])
                        params_summary += f"φ={val:.2f}, "
                    if 'n' in params:
                        val = get_param_value(params['n'])
                        params_summary += f"n={val:.2f}, "
                    if 'm' in params:
                        val = get_param_value(params['m'])
                        params_summary += f"m={val:.2f}"

            # Format chi2 value safely
            chi2_val = inversion_results.get('chi2', None)
            chi2_str = f"{chi2_val:.3f}" if chi2_val is not None and isinstance(chi2_val, (int, float)) else 'N/A'

            interpretation = f"""ERT inversion and petrophysical conversion completed successfully.

**Inversion Results:**
- Chi-squared misfit: {chi2_str}
- Iterations: {inversion_results.get('iterations', 'N/A')}

**Petrophysical Conversion:**
- Monte Carlo realizations: {workflow_config.get('n_realizations', 100)}
- Parameters used:{params_summary if params_summary else ' Default Archie parameters'}

**Water Content Statistics:**
- Mean water content range: {np.nanmin(petro_results.get('water_content_mean', [0])):.3f} - {np.nanmax(petro_results.get('water_content_mean', [0])):.3f}
- Uncertainty (std): {np.nanmean(petro_results.get('water_content_std', [0])):.3f}
"""

        # Generate comprehensive report
        if results.get('status') == 'success':
            update_progress("Generating report", 0.85, "Creating visualizations and summary")
            from .report_agent import ReportAgent
            report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Build workflow_data based on whether petrophysics was run
            workflow_data = {
                'ert_data': {
                    'n_electrodes': len(ert_data.electrodes),
                    'num_electrodes': len(ert_data.electrodes),
                    'n_measurements': len(ert_data.observations),
                    'num_measurements': len(ert_data.observations),
                    'instrument': workflow_config.get('instrument', 'DAS-1')
                },
                'inversion_results': {
                    'chi2': inversion_results.get('chi2'),
                    'iterations': inversion_results.get('iterations'),
                    'resistivity_model': inversion_results.get('resistivity_model'),
                    'mesh': inversion_results.get('mesh'),
                    'coverage': inversion_results.get('coverage')
                },
                'evaluation_results': evaluation_results or {},
                'skip_petrophysics': skip_petrophysics
            }

            # Only include water content and petrophysics data if conversion was performed
            workflow_data['inversion_results']['processing'] = inversion_results.get('processing', {})
            if not skip_petrophysics:
                petro_results = results.get('petrophysics_results', {})
                workflow_data['water_content'] = {
                    'mesh': inversion_results['mesh'],
                    'water_content_mean': petro_results.get('water_content_mean'),
                    'water_content_std': petro_results.get('water_content_std'),
                    'layer_params_used': petro_results.get('layer_params_used', {}),
                    'layer_params': petro_results.get('layer_params', {}),
                    'petrophysical_params': workflow_config.get('petrophysical_params', {}),
                    'n_realizations': workflow_config.get('n_realizations', 200)
                }
                workflow_data['petrophysics_results'] = petro_results
                workflow_data['petrophysical_params'] = workflow_config.get('petrophysical_params', {})

            report_input = {
                'workflow_data': workflow_data,
                'config': workflow_config,
                'output_dir': str(output_dir)
            }
            report_results = report_agent.execute(report_input)

            update_progress("Report complete", 0.95, "Saving files")

            if report_results.get('status') == 'success':
                report_files = {}
                if report_results.get('report_file'):
                    report_files['report_markdown'] = report_results['report_file']
                if report_results.get('html_file'):
                    report_files['report_html'] = report_results['html_file']
                if report_results.get('pdf_file'):
                    report_files['report_pdf'] = report_results['pdf_file']
                for vis_name, vis_path in (report_results.get('visualization_files') or {}).items():
                    report_files[f'visualization_{vis_name}'] = vis_path

    elif workflow_type == 'tdem':
        # Use TDEMAgent for Time-Domain Electromagnetic workflow
        from .tdem_agent import TDEMAgent
        tdem_agent = TDEMAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        update_progress("Starting TDEM workflow", 0.15, "Configuring electromagnetic inversion")
        print('Running TDEM workflow...')

        # Set execution plan for TDEM workflow
        execution_plan = [
            {'step': 'Load TDEM Data', 'agent': 'TDEMAgent', 
             'description': 'Load time-domain electromagnetic sounding data', 
             'outputs': ['times', 'dobs', 'uncertainties']},
            {'step': 'Run TDEM Inversion', 'agent': 'TDEMAgent', 
             'description': 'Invert for 1D conductivity model using SimPEG', 
             'outputs': ['conductivity_model', 'chi2']},
            {'step': 'Generate Visualization', 'agent': 'TDEMAgent', 
             'description': 'Create result plots and interpretation', 
             'outputs': ['visualization_files', 'interpretation']},
        ]

        interpretation = (
            "TDEM (Time-Domain Electromagnetic) workflow processes electromagnetic sounding data "
            "to recover subsurface conductivity structure. Uses SimPEG for 1D layered Earth inversion."
        )

        # Get TDEM data file with path resolution
        tdem_file = workflow_config.get('tdem_file') or workflow_config.get('data_file')
        if tdem_file:
            tdem_file_path = Path(tdem_file)
            if not tdem_file_path.exists():
                # Try to find in uploaded_files
                uploaded_files = workflow_config.get('uploaded_files', {})
                if tdem_file_path.name in uploaded_files:
                    tdem_file = uploaded_files[tdem_file_path.name]
                    tdem_file_path = Path(tdem_file)
                    print(f'  → Found TDEM file in uploads: {tdem_file_path.name}')
                else:
                    # Try project_dir
                    project_dir = workflow_config.get('project_dir', '.')
                    if project_dir and project_dir != '.':
                        combined_path = Path(project_dir) / tdem_file_path.name
                        if combined_path.exists():
                            tdem_file_path = combined_path
                            tdem_file = str(tdem_file_path)

            if not tdem_file_path.exists():
                raise ValueError(f"TDEM data file not found: {tdem_file}")

            print(f'  → TDEM file: {tdem_file_path.name}')
            tdem_file = str(tdem_file_path)

        # Prepare TDEM input
        tdem_input = {
            'mode': workflow_config.get('tdem_mode', 'inversion'),
            'data_file': tdem_file,
            'source_radius': workflow_config.get('source_radius', 10.0),
            'n_layers': workflow_config.get('n_layers', 20),
            'min_thickness': workflow_config.get('min_thickness', 0.5),
            'max_thickness': workflow_config.get('max_thickness', 10.0),
            'starting_conductivity': workflow_config.get('starting_conductivity', 0.001),
            'use_irls': workflow_config.get('use_irls', True),
            'max_iterations': workflow_config.get('max_iterations', 50),
            'output_dir': str(output_dir / 'tdem'),
            'verbose': True
        }

        # Handle forward modeling mode
        if tdem_input['mode'] == 'forward':
            tdem_input['thicknesses'] = workflow_config.get('thicknesses')
            tdem_input['conductivity'] = workflow_config.get('conductivity')
            tdem_input['times'] = workflow_config.get('times')
            tdem_input['noise_level'] = workflow_config.get('noise_level', 0.05)

        # Handle hydro-to-tdem mode
        if tdem_input['mode'] == 'hydro_to_tdem':
            tdem_input['water_content'] = workflow_config.get('water_content')
            tdem_input['porosity'] = workflow_config.get('porosity')
            tdem_input['layer_thicknesses'] = workflow_config.get('layer_thicknesses')
            tdem_input['petrophysical_params'] = workflow_config.get('petrophysical_params', {})

        update_progress("Running TDEM processing", 0.30, "Loading data and running inversion")

        # Execute TDEM workflow
        results = tdem_agent.execute(tdem_input)

        if results.get('status') == 'success':
            update_progress("TDEM complete", 0.80, f"Chi² = {results.get('chi2', 'N/A')}")

            # Generate report
            update_progress("Generating report", 0.90, "Creating TDEM report")
            from .report_agent import ReportAgent
            report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Create TDEM-specific report
            # Format values safely
            chi2_val = results.get('chi2', None)
            chi2_str = f"{chi2_val:.3f}" if chi2_val is not None and isinstance(chi2_val, (int, float)) else 'N/A'

            cond_range = results.get('conductivity_range', None)
            if cond_range and len(cond_range) == 2:
                cond_str = f"{cond_range[0]:.4f} - {cond_range[1]:.4f} S/m"
            else:
                cond_str = "N/A"

            res_range = results.get('resistivity_range', None)
            if res_range and len(res_range) == 2:
                res_str = f"{res_range[0]:.1f} - {res_range[1]:.1f} Ωm"
            else:
                res_str = "N/A"

            tdem_report = f"""# TDEM Inversion Report

**Generated by:** PyHydroGeophysX TDEMAgent
**Mode:** {results.get('mode', 'inversion')}

## Executive Summary

{results.get('interpretation', 'TDEM processing completed successfully.')}

## Inversion Results

### Model Statistics
- **Number of Layers:** {results.get('n_layers', 'N/A')}
- **Chi-squared Misfit:** {chi2_str}
- **Conductivity Range:** {cond_str}
- **Resistivity Range:** {res_str}

## Visualization

![TDEM Result]({Path(results.get('visualization_file', '')).name})

## Output Files

- Output directory: `{results.get('output_dir', 'N/A')}`
- Recovered conductivity: `recovered_conductivity.npy`
- Layer thicknesses: `inv_thicknesses.npy`
- Predicted data: `predicted_data.npy`

---
*Report generated by PyHydroGeophysX TDEMAgent using SimPEG*
"""
            # Save report
            report_file = output_dir / 'tdem_report.md'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(tdem_report)

            report_files = {'report_markdown': str(report_file)}

            # Save HTML version
            html_file = report_agent._save_html_report(tdem_report, str(output_dir), 'tdem_report')
            if html_file:
                report_files['report_html'] = html_file

            # Save PDF version
            pdf_file = report_agent._save_pdf_report(tdem_report, str(output_dir), 
                                                     visualization_files={'tdem_result': results.get('visualization_file', '')},
                                                     filename='tdem_report')
            if pdf_file:
                report_files['report_pdf'] = pdf_file

            if results.get('visualization_file'):
                report_files['visualization_tdem'] = results['visualization_file']

            interpretation = results.get('interpretation', interpretation)
        else:
            update_progress("TDEM failed", 1.0, results.get('error', 'Unknown error'))
            raise ValueError(f"TDEM processing failed: {results.get('error')}")

    elif workflow_type == 'seismic':
        # Use SeismicAgent for standalone seismic refraction tomography
        from .seismic_agent import SeismicAgent
        seismic_agent = SeismicAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

        update_progress("Starting seismic workflow", 0.15, "Configuring seismic refraction tomography")
        print('Running seismic refraction tomography workflow...')

        raw_seismic_file = workflow_config.get('raw_seismic_file')

        # Set execution plan for seismic workflow
        if raw_seismic_file:
            execution_plan = [
                {'step': 'Read Raw SEG-Y', 'agent': 'RawSeismicProcessor',
                 'description': 'Read SEG-Y headers and organize shot gathers',
                 'outputs': ['segy_metadata', 'shot_gathers']},
                {'step': 'Pick First Breaks', 'agent': 'RawSeismicProcessor',
                 'description': 'Apply preprocessing and assisted first-break picking',
                 'outputs': ['first_break_picks']},
                {'step': 'Export Travel-Time Data', 'agent': 'RawSeismicProcessor',
                 'description': 'Convert first breaks to PyGIMLi travel-time format',
                 'outputs': ['traveltime_file']},
                {'step': 'Run SRT Inversion', 'agent': 'SeismicAgent',
                 'description': 'Invert for P-wave velocity model using PyGIMLI',
                 'outputs': ['velocity_model', 'mesh', 'coverage']},
                {'step': 'Generate Visualization', 'agent': 'SeismicAgent',
                 'description': 'Create velocity tomogram and interface plots',
                 'outputs': ['visualization_files', 'interpretation']},
            ]
        else:
            execution_plan = [
                {'step': 'Load Seismic Data', 'agent': 'SeismicAgent', 
                 'description': 'Load travel time data from .dat file', 
                 'outputs': ['seismic_data']},
                {'step': 'Run SRT Inversion', 'agent': 'SeismicAgent', 
                 'description': 'Invert for P-wave velocity model using PyGIMLI', 
                 'outputs': ['velocity_model', 'mesh', 'coverage']},
                {'step': 'Extract Interfaces', 'agent': 'SeismicAgent', 
                 'description': 'Extract geological interfaces from velocity thresholds', 
                 'outputs': ['interface_coords']},
                {'step': 'Generate Visualization', 'agent': 'SeismicAgent', 
                 'description': 'Create velocity tomogram and interface plots', 
                 'outputs': ['visualization_files', 'interpretation']},
            ]

        interpretation = (
            "Seismic refraction tomography (SRT) workflow inverts travel time data to "
            "recover subsurface P-wave velocity structure. Velocity interfaces are "
            "extracted for geological interpretation and hydrogeological modeling."
        )

        # Get seismic file
        seismic_file = raw_seismic_file or workflow_config.get('seismic_file')
        if not seismic_file:
            raise ValueError('No seismic file specified in configuration. '
                           'Please provide seismic_file or raw_seismic_file path.')

        # Normalize seismic file path
        seismic_file_path = Path(seismic_file)
        project_dir = workflow_config.get('project_dir', '.')

        if not seismic_file_path.exists():
            # Try combining with project_dir
            if project_dir and project_dir != '.':
                combined_path = Path(project_dir) / seismic_file_path.name
                if combined_path.exists():
                    seismic_file_path = combined_path
                elif combined_path.parts[0] == 'examples':
                    combined_path = Path(*combined_path.parts[1:])
                    if combined_path.exists():
                        seismic_file_path = combined_path

            # Try removing 'examples/' prefix
            if not seismic_file_path.exists() and len(seismic_file_path.parts) > 0:
                if seismic_file_path.parts[0] == 'examples':
                    alt_path = Path(*seismic_file_path.parts[1:])
                    if alt_path.exists():
                        seismic_file_path = alt_path

        seismic_file = str(seismic_file_path)
        print(f'  → Seismic file: {seismic_file_path.name}')

        # Get velocity thresholds for interface extraction
        velocity_thresholds = workflow_config.get('velocity_thresholds', [1200])
        if isinstance(velocity_thresholds, (int, float)):
            velocity_thresholds = [velocity_thresholds]

        # Add additional threshold from user request if specified
        velocity_threshold = workflow_config.get('velocity_threshold')
        if velocity_threshold and velocity_threshold not in velocity_thresholds:
            velocity_thresholds.append(velocity_threshold)

        print(f'  → Velocity thresholds: {velocity_thresholds} m/s')

        # Prepare inversion parameters
        inversion_params = workflow_config.get('inversion_params', {})
        if not inversion_params:
            inversion_params = {
                'lam': workflow_config.get('lambda', 50),
                'zWeight': workflow_config.get('z_weight', 0.2),
                'vTop': workflow_config.get('v_top', 500),
                'vBottom': workflow_config.get('v_bottom', 5000),
                'paraDepth': workflow_config.get('para_depth', 30.0),
                'limits': workflow_config.get('velocity_limits', [300., 8000.])
            }

        # Prepare seismic input
        seismic_input = {
            'seismic_file': seismic_file,
            'raw_seismic_file': str(seismic_file_path) if raw_seismic_file else None,
            'velocity_threshold': velocity_thresholds[0] if velocity_thresholds else 1200,
            'velocity_thresholds': velocity_thresholds,
            'inversion_params': inversion_params,
            'extract_interfaces': workflow_config.get('extract_interfaces', True),
            'output_dir': str(output_dir / 'seismic'),
            'raw_max_traces': workflow_config.get('raw_max_traces'),
            'geophone_file': workflow_config.get('geophone_file'),
            'topography_file': workflow_config.get('topography_file'),
            'first_break_params': workflow_config.get('first_break_params', {}),
        }

        if raw_seismic_file:
            update_progress("Processing raw SEG-Y", 0.30, "Picking first breaks and exporting travel-time data")
        else:
            update_progress("Running seismic inversion", 0.30, "Loading data and inverting velocity model")

        # Execute seismic workflow
        results = seismic_agent.execute(seismic_input)

        if results.get('status') == 'success':
            vel_range = results.get('velocity_range', [0, 0])
            update_progress("Seismic inversion complete", 0.80, 
                          f"Velocity: {vel_range[0]:.0f} - {vel_range[1]:.0f} m/s")

            # Generate report
            update_progress("Generating report", 0.90, "Creating seismic report")
            from .report_agent import ReportAgent
            report_agent = ReportAgent(api_key=api_key, model=llm_model, llm_provider=llm_provider)

            # Format interface information
            interfaces_info = ""
            for threshold, data in results.get('interfaces', {}).items():
                z_min = min(data['z']) if len(data['z']) > 0 else 'N/A'
                z_max = max(data['z']) if len(data['z']) > 0 else 'N/A'
                interfaces_info += f"- **{threshold} m/s interface:** Depth range {z_min:.1f} to {z_max:.1f} m\n"

            # Create seismic-specific report
            seismic_report = f"""# Seismic Refraction Tomography Report

**Generated by:** PyHydroGeophysX SeismicAgent

## Executive Summary

{results.get('interpretation', 'Seismic refraction tomography completed successfully.')}

## Survey Information

- **Data File:** `{seismic_file_path.name}`
- **Number of Shots:** {results.get('n_shots', 'N/A')}
- **Number of Receivers:** {results.get('n_receivers', 'N/A')}
- **Total Travel Times:** {results.get('n_data', 'N/A')}

## Inversion Results

### Velocity Model Statistics
- **Velocity Range:** {vel_range[0]:.0f} - {vel_range[1]:.0f} m/s
- **Mesh Cells:** {results.get('mesh').cellCount() if results.get('mesh') else 'N/A'}

### Inversion Parameters
- **Lambda (regularization):** {inversion_params.get('lam', 50)}
- **Z-Weight:** {inversion_params.get('zWeight', 0.2)}
- **Velocity Constraints:** {inversion_params.get('vTop', 500)} - {inversion_params.get('vBottom', 5000)} m/s

## Extracted Interfaces

{interfaces_info if interfaces_info else 'No interfaces extracted.'}

### Geological Interpretation

Based on typical velocity-depth relationships:
- **< 1200 m/s:** Weathered soil/regolith
- **1200-3000 m/s:** Fractured rock
- **> 3000 m/s:** Competent bedrock

## Visualization

![Seismic Velocity Model]({Path(results.get('visualization_file', '')).name})

## Output Files

- **Output directory:** `{results.get('output_dir', 'N/A')}`
- **Velocity model:** `velocity_model.npy`
- **Coverage:** `coverage.npy`
- **Mesh:** `seismic_mesh.bms`
- **Interface files:** `interface_*ms.txt`

---
*Report generated by PyHydroGeophysX SeismicAgent using PyGIMLI*
"""
            # Save report
            report_file = output_dir / 'seismic_report.md'
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(seismic_report)

            report_files = {'report_markdown': str(report_file)}

            # Save HTML version
            html_file = report_agent._save_html_report(seismic_report, str(output_dir), 'seismic_report')
            if html_file:
                report_files['report_html'] = html_file

            # Save PDF version
            pdf_file = report_agent._save_pdf_report(seismic_report, str(output_dir),
                                                     visualization_files={'velocity_model': results.get('visualization_file', '')},
                                                     filename='seismic_report')
            if pdf_file:
                report_files['report_pdf'] = pdf_file

            if results.get('visualization_file'):
                report_files['visualization_seismic'] = results['visualization_file']

            interpretation = results.get('interpretation', interpretation)
        else:
            update_progress("Seismic inversion failed", 1.0, results.get('error', 'Unknown error'))
            raise ValueError(f"Seismic processing failed: {results.get('error')}")

    elif workflow_type == 'custom':
        # A request no workflow type matches. Out-of-scope requests used to go
        # to CodeGenerationAgent, which wrote and ran model-generated code; that
        # agent was removed, so every such request gets the error an in-scope
        # request without data files always got.
        raise ValueError(
            'Could not infer workflow type from config! '
            'Please provide at least ert_file or data_file.'
        )

    else:
        raise ValueError('Unknown workflow type!')

    llm_usage_ledger = []
    seen_agent_ids = set()
    for value in list(locals().values()):
        if hasattr(value, "llm_usage_ledger") and id(value) not in seen_agent_ids:
            seen_agent_ids.add(id(value))
            llm_usage_ledger.extend(getattr(value, "llm_usage_ledger", []) or [])
    total_llm_cost = sum(
        float(item.get("cost_estimate_usd") or 0.0)
        for item in llm_usage_ledger
    )
    if isinstance(results, dict):
        results["llm_usage_ledger"] = llm_usage_ledger
        results["total_llm_cost_estimate_usd"] = total_llm_cost

    return results, execution_plan, interpretation, report_files
