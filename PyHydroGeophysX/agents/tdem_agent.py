"""
TDEM (Time-Domain Electromagnetic) Agent

Agent for processing Time-Domain Electromagnetic data using SimPEG.
Supports forward modeling, inversion, and integration with hydrological models.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import AgentResult, BaseAgent
from PyHydroGeophysX.visualization.axis_units import set_length_axis


# ---------------------------------------------------------------------------
# TDEMAgent
# ---------------------------------------------------------------------------
class TDEMAgent(BaseAgent):
    """
    Agent for Time-Domain Electromagnetic (TDEM) data processing.
    
    This agent provides functionality for:
    - Loading TDEM sounding data from text files
    - Forward modeling from hydrological models (MODFLOW, ParFlow)
    - 1D TDEM inversion with L2 and sparse (IRLS) regularization
    - Petrophysical conversion between water content and conductivity
    - Visualization and reporting
    
    Example (with your own sounding file):
        >>> agent = TDEMAgent()
        >>> result = agent.execute({  # doctest: +SKIP
        ...     'data_file': 'tdem_data.txt',
        ...     'source_radius': 10.0,
        ...     'n_layers': 20,
        ...     'output_dir': 'results/tdem'
        ... })
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize TDEM Agent."""
        super().__init__("tdem_agent", api_key, model, llm_provider)
        self.system_message = """You are an expert in electromagnetic geophysics, 
specializing in Time-Domain Electromagnetic (TDEM) methods for subsurface characterization.
You understand the physics of electromagnetic induction in layered Earth models
and can interpret conductivity structures in terms of geological and hydrological properties."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TDEM workflow based on input configuration.
        
        Args:
            input_data: Dictionary containing:
                - data_file: Path to TDEM data file (optional if forward modeling)
                - mode: 'inversion', 'forward', or 'hydro_to_tdem'
                - source_radius: Loop radius in meters (default: 10)
                - n_layers: Number of layers for inversion (default: 20)
                - output_dir: Output directory for results
                - use_irls: Use sparse regularization (default: True)
                
                For forward modeling:
                - thicknesses: Layer thicknesses (m)
                - conductivity: Layer conductivities (S/m)
                
                For hydro_to_tdem:
                - water_content: Water content array
                - porosity: Porosity array
                - layer_thicknesses: Layer thicknesses (m)
                - petrophysical_params: Petrophysical parameters
                
        Returns:
            Dictionary containing results based on mode
        """
        self._log_execution("Starting TDEM workflow")
        
        try:
            mode = input_data.get('mode', 'inversion')
            output_dir = input_data.get('output_dir', 'results/tdem')
            os.makedirs(output_dir, exist_ok=True)
            
            if mode == 'inversion':
                return self._run_inversion(input_data, output_dir)
            elif mode == 'forward':
                return self._run_forward(input_data, output_dir)
            elif mode == 'hydro_to_tdem':
                return self._run_hydro_to_tdem(input_data, output_dir)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'inversion', 'forward', or 'hydro_to_tdem'")
                
        except Exception as e:
            self._log_execution(f"Error in TDEM workflow: {str(e)}", level='ERROR')
            return AgentResult(
                status="failed",
                summary="TDEM workflow could not be completed.",
                data={},
                error=str(e),
                error_fix_hint="Check the TDEM mode, data_file path, and required inversion or forward-modeling inputs.",
            )
    
    def _run_inversion(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run TDEM inversion workflow."""
        self._log_execution("Running TDEM inversion")
        data_file = input_data.get('data_file')
        if data_file and self._is_instrument_survey(data_file):
            return self._run_survey_inversion(input_data, output_dir)

        from PyHydroGeophysX.inversion.tdem_inversion import TDEMInversion, TDEMInversionResult

        # Load data
        validation_error = self.validate_input_file(
            data_file,
            supported_extensions=[".dat", ".txt", ".csv"],
            field_name="data_file",
            max_size_mb=input_data.get("max_file_size_mb"),
        )
        if validation_error:
            return validation_error
        
        times, dobs, uncertainties = self._load_tdem_data(
            data_file, int(input_data.get('sounding', 0)))
        self._log_execution(f"Loaded {len(times)} time channels from {Path(data_file).name}")
        
        # Inversion parameters
        source_radius = input_data.get('source_radius', 10.0)
        n_layers = input_data.get('n_layers', 20)
        min_thickness = input_data.get('min_thickness', 0.5)
        max_thickness = input_data.get('max_thickness', 10.0)
        starting_conductivity = input_data.get('starting_conductivity', 0.001)
        use_irls = input_data.get('use_irls', True)
        max_iterations = input_data.get('max_iterations', 50)
        verbose = input_data.get('verbose', True)
        
        # Create inversion object
        self._log_execution(f"Setting up inversion with {n_layers} layers")
        tdem_inv = TDEMInversion(
            times=times,
            dobs=dobs,
            uncertainties=uncertainties,
            source_radius=source_radius,
            n_layers=n_layers,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            starting_conductivity=starting_conductivity,
            use_irls=use_irls,
            max_iterations=max_iterations,
            verbose=verbose
        )
        
        # Run inversion
        self._log_execution("Running inversion (this may take a few minutes)...")
        result = tdem_inv.run()
        
        self._log_execution(f"Inversion complete! Chi² = {result.chi2:.3f}")
        
        # Save results
        np.save(os.path.join(output_dir, "recovered_conductivity.npy"), result.recovered_conductivity)
        np.save(os.path.join(output_dir, "recovered_model_log.npy"), result.recovered_model)
        np.save(os.path.join(output_dir, "inv_thicknesses.npy"), result.thicknesses)
        np.save(os.path.join(output_dir, "predicted_data.npy"), result.predicted_data)
        
        if result.l2_conductivity is not None:
            np.save(os.path.join(output_dir, "l2_conductivity.npy"), result.l2_conductivity)
        
        # Generate visualization
        vis_file = self._generate_inversion_plots(
            tdem_inv, result, times, dobs, uncertainties, output_dir
        )
        
        # Generate interpretation
        interpretation = None
        if self.llm_enabled:
            interpretation = self._interpret_results(result, times)
        
        self.results = {
            'status': 'success',
            'mode': 'inversion',
            'chi2': result.chi2,
            'n_layers': len(result.recovered_conductivity),
            'conductivity_range': [
                float(result.recovered_conductivity.min()),
                float(result.recovered_conductivity.max())
            ],
            'resistivity_range': [
                float(1.0 / result.recovered_conductivity.max()),
                float(1.0 / result.recovered_conductivity.min())
            ],
            'recovered_conductivity': result.recovered_conductivity,
            'recovered_resistivity': 1.0 / result.recovered_conductivity,
            'thicknesses': result.thicknesses,
            'predicted_data': result.predicted_data,
            # The data the prediction is compared with, for the evaluation step.
            'times': np.asarray(times, dtype=float),
            'observed_data': np.asarray(dobs, dtype=float),
            'uncertainties': np.asarray(uncertainties, dtype=float),
            'l2_conductivity': result.l2_conductivity,
            'visualization_file': vis_file,
            'interpretation': interpretation,
            'output_dir': output_dir,
            # What the report's method section states.
            'source_file': str(data_file),
            'n_data': int(len(times)),
            'time_range': [float(np.min(times)), float(np.max(times))],
            'source_radius': float(source_radius),
            'use_irls': bool(use_irls),
        }

        return self.results

    @staticmethod
    def _is_instrument_survey(path: Any) -> bool:
        """Whether ``path`` is a survey the EM reader opens with its own system description.

        TEMcompany/TEM2Go projects (``project.tiw``/``project.db``, or the
        folder holding one), raw ``.stb`` acquisition folders,
        ``*_StationData.xyz`` exports, tTEM ``.skb`` acquisitions and stored
        sounding containers. Any folder goes there too: the reader's refusal
        says what a TEM folder lacks, where the text reader could only name an
        extension.
        """
        from PyHydroGeophysX.data_processing import run_inputs
        from PyHydroGeophysX.data_processing.em1d import is_temcompany_source, is_ttem_source

        source = str(path)
        return (Path(source).is_dir() or is_temcompany_source(source)
                or is_ttem_source(source) or run_inputs.is_container(source))

    def _run_survey_inversion(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Invert an instrument survey with the package's laterally constrained EM inversion.

        The text-sounding path models a 10 m circular loop with a step-off
        waveform and reads one column of one file, so it could neither open a
        TEM2Go project nor model one: the loop is 0.63 m square with four
        turns, the receiver sits about 15 m away, and the gates, ramps and
        filters are in the project. This reads the survey with
        :func:`~PyHydroGeophysX.workflows.em1d.load_sounding`, models every
        station with the system the file records, and inverts all of them
        together with :func:`~PyHydroGeophysX.workflows.em1d.invert_line`. Those
        are the settings the EM Processing page starts from for this data (the
        ``ground_tem`` preset under the project's stored inversion settings) and
        the pattern of ``examples/Ex_TEM_LMHM_LCI.py``.

        ``tem_moment`` (``'LM+HM'`` by default, ``'HM'`` for a single-moment
        survey), ``lines`` and ``max_soundings`` choose the data; any key of
        :data:`~PyHydroGeophysX.workflows.em1d.DEFAULT_INVERSION` given in
        ``input_data`` replaces the project's value.
        """
        from PyHydroGeophysX.workflows import em1d

        path = str(input_data['data_file'])
        requested = input_data.get('tem_moment')
        moment = str(requested or 'LM+HM')
        try:
            head = em1d.load_sounding(path, 'TDEM', sounding=0, moment=moment)
        except ValueError:
            if requested:
                raise
            # A single-moment survey has no LM+HM to read jointly.
            moment = 'HM'
            head = em1d.load_sounding(path, 'TDEM', sounding=0, moment=moment)
        system = dict(head.get('system') or {})
        # The recorded system is the geometry, as in Ex_TEM_LMHM_LCI; the
        # forward would complete it from each station's own record anyway.
        geometry = {**system, 'tem_moment': moment}
        inversion = {**em1d.preset_inversion('ground_tem'),
                     **dict(head.get('inversion_defaults') or {})}
        uniform = (head.get('protocol') or {}).get('uniform_std')
        if uniform is not None:
            inversion['rel_error'] = float(uniform)
        overrides = {key: input_data[key] for key in em1d.DEFAULT_INVERSION if key in input_data}
        if input_data.get('starting_conductivity') and 'starting_resistivity' not in overrides:
            overrides.update(starting_resistivity=1.0 / float(input_data['starting_conductivity']),
                             auto_starting_model=False)
        inversion.update(overrides)

        n_total = int(head.get('n_soundings', 1))
        lines = input_data.get('lines')
        lines = [int(value) for value in (lines if isinstance(lines, (list, tuple)) else [lines])] \
            if lines not in (None, '', []) else None
        self._log_execution(
            f"{Path(path).name}: {head.get('source_format') or 'TDEM survey'}, {n_total} "
            f"sounding(s), moment {moment}"
            + (f", {system['instrument']}" if system.get('instrument') else "")
            + (f"; request overrides {sorted(overrides)}" if overrides else ""))
        result = em1d.invert_line(
            path, 'TDEM', geometry, inversion,
            max_soundings=int(input_data.get('max_soundings') or n_total), lines=lines,
            out_dir=Path(output_dir), log=lambda message: self._log_execution(str(message)))

        # Surface first, NaN below each sounding's depth of investigation.
        model = np.asarray(result['model3d'], dtype=float)[:, 0, ::-1]
        resolved = model[np.isfinite(model) & (model > 0)]
        if not resolved.size:
            raise ValueError(f"The inversion of {Path(path).name} resolved no cell above the "
                             "depth of investigation at any station.")
        counts = np.asarray(result.get('data_count_list', []), dtype=int)
        # invert_line wrote the section, its tables and the LCI reports there.
        data_paths = list(result.get('saved') or [])
        vis_file = self._generate_survey_sections(result, input_data, output_dir, Path(path).name)
        self.results = {
            'status': 'success',
            'mode': 'inversion',
            'survey': True,
            'source_file': self._project_read(path),
            'source_format': str(head.get('source_format') or ''),
            'instrument': system.get('instrument'),
            'tem_moment': moment,
            'n_soundings': int(result['n_soundings']),
            'n_soundings_total': n_total,
            'failed_soundings': int(np.count_nonzero(counts == 0)),
            # Inverted, but with no cell above the depth of investigation.
            'unresolved_soundings': int(np.count_nonzero(
                ~np.isfinite(model).any(axis=1) & (counts > 0)))
            if counts.size == model.shape[0] else 0,
            'lines': sorted({int(v) for v in np.asarray(result['line_numbers']).ravel()}),
            'lci_mode': result.get('lci_mode'),
            'chi2': float(result['chi2_global']),
            'chi2_sounding_median': float(result['chi2_sounding_median']),
            'n_layers': int(result['n_layers']),
            'thicknesses': np.asarray(result['thickness'], dtype=float),
            'recovered_resistivity': model,
            'recovered_conductivity': 1.0 / model,
            'resistivity_range': [float(resolved.min()), float(resolved.max())],
            'conductivity_range': [float(1.0 / resolved.max()), float(1.0 / resolved.min())],
            'doi': np.asarray(result['doi'], dtype=float),
            'positions': np.asarray(result['positions'], dtype=float),
            'line_numbers': np.asarray(result['line_numbers'], dtype=int),
            'x': np.asarray(result.get('x', []), dtype=float),
            'y': np.asarray(result.get('y', []), dtype=float),
            # What places the survey on a basemap (em_maps.survey_plan_grids).
            'coordinate_system': str(result.get('coordinate_system') or ''),
            'longitude': np.asarray(result.get('longitude', []), dtype=float),
            'latitude': np.asarray(result.get('latitude', []), dtype=float),
            'station_ids': np.asarray(result.get('station_ids', []), dtype=object),
            'surface_elevation': np.asarray(result.get('surface_elevation', []), dtype=float),
            'depth_edges': np.asarray(result['depth_edges'], dtype=float),
            'chi2_list': np.asarray(result.get('chi2_list', []), dtype=float),
            'data_count_list': counts,
            # What the report's method section states: the system the file
            # records, the protocol it ran, and the settings the inversion used.
            'system': {key: system[key] for key in (
                'instrument', 'loop_area', 'loop_turns', 'tx_rx_sep_nominal',
                'receiver_type', 'waveform') if system.get(key) is not None},
            'tx_rx_distances': np.asarray(head.get('rx_tx_distances', []), dtype=float),
            'protocol': {key: value for key, value in (head.get('protocol') or {}).items()
                         if isinstance(value, (int, float, str))},
            'inversion_settings': {key: inversion[key] for key in (
                'n_layers', 'min_thickness', 'max_thickness', 'smoothness',
                'lateral_smoothness', 'rel_error', 'auto_starting_model',
                'starting_resistivity', 'robust_errors', 'robust_target_chi2',
                'max_iterations', 'auto_lambda') if key in inversion},
            'settings_source': ('project' if head.get('inversion_defaults')
                                else 'ground_tem preset'),
            'overrides': sorted(overrides),
            'robust': {key: (result.get('robust') or {}).get(key) for key in (
                'enabled', 'downweighted', 'n_start', 'chi2_effective')},
            'visualization_file': vis_file,
            'data_paths': data_paths,
            'output_dir': output_dir,
        }
        # Plan-view maps are a step of their own in a workflow run
        # (map_tdem_plan_view), drawn from these results.
        self.results['interpretation'] = (self._interpret_survey(self.results)
                                          if self.llm_enabled else None)
        return self.results

    @staticmethod
    def _project_read(path: str) -> str:
        """The project file a TEMcompany folder is read from, or ``path`` itself.

        The reader takes the standard name when a folder holds several
        (``project.tiw`` before ``project2.tiw``), and a report naming only
        the folder would not say which of them it describes.
        """
        from PyHydroGeophysX.data_processing import temcompany_project

        if Path(path).is_dir():
            try:
                return str(temcompany_project.project_file(path))
            except (OSError, ValueError):
                pass
        return path

    def _generate_survey_sections(self, result: Dict[str, Any], input_data: Dict[str, Any],
                                  output_dir: str, source_name: str) -> str:
        """One resistivity section per survey line, on one colour scale.

        Drawn against elevation where the stations carry one and against depth
        otherwise, in the unit ``input_data['figure_style']`` asks for. Cells
        below a station's depth of investigation are left blank.
        """
        import matplotlib.colors as mcolors

        from . import _figstyle as figstyle
        from PyHydroGeophysX.visualization.axis_units import set_section_axes

        style = figstyle.style_from_config(input_data)
        model = np.asarray(result['model3d'], dtype=float)[:, 0, ::-1]
        depth_edges = np.asarray(result['depth_edges'], dtype=float).ravel()[:model.shape[1] + 1]
        positions = np.asarray(result['positions'], dtype=float).ravel()
        lines = np.asarray(result['line_numbers'], dtype=int).ravel()
        ground = np.asarray(result.get('surface_elevation', []), dtype=float).ravel()
        if ground.size != positions.size:
            ground = np.full(positions.size, np.nan)
        resolved = model[np.isfinite(model) & (model > 0)]
        norm = mcolors.LogNorm(vmin=float(resolved.min()), vmax=float(resolved.max()))
        names = list(dict.fromkeys(lines.tolist()))
        cols = min(3, len(names))
        rows = int(np.ceil(len(names) / cols))
        fig = figstyle.detached_figure((
            min(style.panel_width * 1.5 * cols, figstyle.MAX_FIGURE_WIDTH_IN),
            0.6 * style.panel_height * rows + 0.8))
        fig.set_layout_engine('constrained')
        axes = np.atleast_1d(fig.subplots(rows, cols, squeeze=False)).ravel()
        image = None
        for ax, line in zip(axes, names):
            index = np.flatnonzero(lines == line)
            index = index[np.argsort(positions[index], kind='stable')]
            x = positions[index]
            if x.size > 1:
                middle = 0.5 * (x[:-1] + x[1:])
                edges = np.concatenate([[2 * x[0] - middle[0]], middle, [2 * x[-1] - middle[-1]]])
            else:
                edges = np.array([x[0] - 2.5, x[0] + 2.5])
            surface = ground[index]
            elevated = bool(np.all(np.isfinite(surface)) and np.any(np.abs(surface) > 1e-6))
            top = np.interp(edges, x, surface) if elevated else np.zeros_like(edges)
            depth_y = top[None, :] - depth_edges[:, None]
            image = ax.pcolormesh(np.broadcast_to(edges[None, :], depth_y.shape), depth_y,
                                  np.ma.masked_invalid(model[index].T), shading='flat',
                                  cmap=style.cmap_for('resistivity'), norm=norm)
            # The survey's own path distance, which runs on from line to line.
            set_section_axes(ax, surface=surface if elevated else np.zeros(1),
                             unit=style.length_unit, xlabel='Distance along survey path',
                             fontsize=style.label_size)
            blank = '' if np.isfinite(model[index]).any() else ', nothing resolved'
            ax.set_title(f"Line {line} ({index.size} sounding{'s' if index.size != 1 else ''}"
                         f"{blank})", fontsize=style.title_size)
            ax.tick_params(labelsize=style.tick_size)
        for ax in axes[len(names):]:
            ax.set_visible(False)
        fig.colorbar(image, ax=axes[:len(names)].tolist(), label='Resistivity (ohm-m)')
        fig.suptitle(f"{source_name}: TEM resistivity, {result.get('lci_mode', 'off')} LCI, "
                     f"median sounding chi-squared {float(result['chi2_sounding_median']):.2f}",
                     fontsize=style.title_size)
        vis_file = figstyle.save(fig, os.path.join(output_dir, 'tdem_sections.png'), style)
        self._log_execution(f"Saved visualization to {vis_file}")
        return vis_file

    def _interpret_survey(self, results: Dict[str, Any]) -> Optional[str]:
        """A short LLM reading of a survey inversion, from its numbers only."""
        thickness = np.asarray(results['thicknesses'], dtype=float)
        doi = np.asarray(results['doi'], dtype=float)
        prompt = f"""Interpret this ground TEM survey inversion for a geophysics report:

- Data: {results['source_format']} ({results.get('instrument') or 'TEM'}), moment {results['tem_moment']}
- {results['n_soundings']} soundings on {len(results['lines'])} line(s), {results['lci_mode']} laterally constrained inversion
- {results['n_layers']} layers, top layer {thickness[0]:.1f} m, layered to {thickness.sum():.0f} m
- Median depth of investigation: {np.nanmedian(doi):.0f} m
- Chi-squared: median per sounding {results['chi2_sounding_median']:.2f}, global {results['chi2']:.2f}
- Resolved resistivity range: {results['resistivity_range'][0]:.1f} - {results['resistivity_range'][1]:.1f} ohm-m

Provide a brief interpretation (3-4 sentences) covering the data fit (a global chi-squared
far above the median means a few soundings fit poorly), what the resistivity range suggests,
and where the model is reliable given the depth of investigation."""
        try:
            return self.query_llm(prompt, self.system_message, temperature=0.5, max_tokens=300)
        except Exception as exc:  # noqa: BLE001 - the inversion stands without it
            self._log_execution(f"Could not generate interpretation: {exc}", level='WARNING')
            return None

    def _run_forward(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run TDEM forward modeling."""
        from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling, TDEMSurveyConfig
        
        self._log_execution("Running TDEM forward modeling")
        
        # Required parameters
        thicknesses = np.asarray(input_data.get('thicknesses'))
        conductivity = np.asarray(input_data.get('conductivity'))
        
        if thicknesses is None or conductivity is None:
            raise ValueError("thicknesses and conductivity are required for forward modeling")
        
        # Survey parameters
        times = input_data.get('times')
        if times is None:
            times = np.logspace(-5, -2, 31)  # Default: 10µs to 10ms
        times = np.asarray(times)
        
        source_radius = input_data.get('source_radius', 10.0)
        noise_level = input_data.get('noise_level', 0.05)
        seed = input_data.get('seed', 42)
        
        # Create survey config
        survey_config = TDEMSurveyConfig(
            source_location=np.array([0.0, 0.0, 0.0]),
            source_radius=source_radius,
            times=times,
            waveform_type="step_off"
        )
        
        # Create forward modeler
        fwd = TDEMForwardModeling(
            thicknesses=thicknesses,
            survey_config=survey_config
        )
        
        # Compute forward response
        dobs, dpred_clean, uncertainties = fwd.forward_with_noise(
            conductivity,
            noise_level=noise_level,
            seed=seed
        )
        
        self._log_execution(f"Forward modeling complete: {len(dobs)} data points")
        
        # Save synthetic data
        data_file = os.path.join(output_dir, "tdem_synthetic_data.txt")
        np.savetxt(
            data_file, 
            np.c_[times, dobs, uncertainties],
            fmt="%.6e",
            header="TIME(s) BZ(T) UNCERTAINTY(T)"
        )
        
        # Generate plot
        vis_file = self._generate_forward_plot(times, dobs, dpred_clean, uncertainties, output_dir)
        
        self.results = {
            'status': 'success',
            'mode': 'forward',
            'n_data': len(dobs),
            'time_range': [float(times.min()), float(times.max())],
            'data_range': [float(np.abs(dobs).min()), float(np.abs(dobs).max())],
            'dobs': dobs,
            'dpred_clean': dpred_clean,
            'uncertainties': uncertainties,
            'times': times,
            'data_file': data_file,
            'visualization_file': vis_file,
            'output_dir': output_dir
        }
        
        return self.results
    
    def _run_hydro_to_tdem(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run hydrological model to TDEM conversion."""
        from PyHydroGeophysX.forward.tdem_forward import hydro_to_tdem
        from PyHydroGeophysX.petrophysics.resistivity_models import WS_Model
        
        self._log_execution("Converting hydrological model to TDEM response")
        
        # Required inputs
        water_content = np.asarray(input_data.get('water_content'))
        porosity = np.asarray(input_data.get('porosity'))
        layer_thicknesses = np.asarray(input_data.get('layer_thicknesses'))
        
        if water_content is None or porosity is None or layer_thicknesses is None:
            raise ValueError("water_content, porosity, and layer_thicknesses are required")
        
        # Petrophysical parameters
        petro_params = input_data.get('petrophysical_params', {})
        sigma_w = petro_params.get('sigma_w', 0.05)  # Pore water conductivity
        m = petro_params.get('m', 1.5)  # Cementation exponent
        n = petro_params.get('n', 2.0)  # Saturation exponent
        sigma_s = petro_params.get('sigma_s', 0.0)  # Surface conductivity
        
        # Survey parameters
        times = input_data.get('times')
        if times is None:
            times = np.logspace(-5, -2, 31)
        source_radius = input_data.get('source_radius', 10.0)
        noise_level = input_data.get('noise_level', 0.05)
        seed = input_data.get('seed', 42)
        
        # Run conversion
        dobs, dpred_clean, uncertainties, conductivity = hydro_to_tdem(
            water_content=water_content,
            porosity=porosity,
            layer_thicknesses=layer_thicknesses,
            sigma_w=sigma_w,
            m=m,
            n=n,
            sigma_s=sigma_s,
            times=times,
            source_radius=source_radius,
            noise_level=noise_level,
            seed=seed,
            verbose=True
        )
        
        self._log_execution(f"Conversion complete: conductivity range {conductivity.min():.4f} - {conductivity.max():.4f} S/m")
        
        # Save results
        data_file = os.path.join(output_dir, "tdem_from_hydro.txt")
        np.savetxt(
            data_file,
            np.c_[times, dobs, uncertainties],
            fmt="%.6e",
            header="TIME(s) BZ(T) UNCERTAINTY(T)"
        )
        
        np.save(os.path.join(output_dir, "conductivity_from_hydro.npy"), conductivity)
        
        # Generate plots
        vis_file = self._generate_hydro_plots(
            times, dobs, dpred_clean, uncertainties,
            water_content, porosity, conductivity, layer_thicknesses,
            output_dir
        )
        
        self.results = {
            'status': 'success',
            'mode': 'hydro_to_tdem',
            'n_layers': len(conductivity),
            'conductivity_range': [float(conductivity.min()), float(conductivity.max())],
            'water_content_range': [float(water_content.min()), float(water_content.max())],
            'dobs': dobs,
            'dpred_clean': dpred_clean,
            'uncertainties': uncertainties,
            'times': times,
            'conductivity': conductivity,
            'data_file': data_file,
            'visualization_file': vis_file,
            'output_dir': output_dir
        }
        
        return self.results
    
    def _generate_inversion_plots(self, tdem_inv, result, times, dobs, uncertainties, 
                                   output_dir: str) -> str:
        """Generate inversion result plots."""
        import matplotlib
        import matplotlib.pyplot as plt
        
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Conductivity model
        ax1 = axes[0]
        depths = np.cumsum(np.r_[0, result.thicknesses])
        
        # Plot L2 model if available
        if result.l2_conductivity is not None:
            for i in range(len(result.l2_conductivity)):
                if i < len(depths) - 1:
                    ax1.plot([result.l2_conductivity[i], result.l2_conductivity[i]],
                            [depths[i], depths[i+1]], 'b-', lw=2,
                            label='L2 Model' if i == 0 else '')
        
        # Plot sparse model
        for i in range(len(result.recovered_conductivity)):
            if i < len(depths) - 1:
                ax1.plot([result.recovered_conductivity[i], result.recovered_conductivity[i]],
                        [depths[i], depths[i+1]], 'r-', lw=2,
                        label='Sparse Model' if i == 0 else '')
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Conductivity (S/m)')
        set_length_axis(ax1, "y", "Depth")
        ax1.set_title('Recovered Conductivity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. Data fit
        ax2 = axes[1]
        ax2.loglog(times * 1e3, np.abs(dobs), 'ko', markersize=6, label='Observed')
        ax2.loglog(times * 1e3, np.abs(result.predicted_data), 'r-', lw=2, label='Predicted')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('|Bz| (T)')
        ax2.set_title(f'Data Fit (χ² = {result.chi2:.2f})')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.5)
        
        # 3. Residuals
        ax3 = axes[2]
        residual = (dobs - result.predicted_data) / uncertainties
        ax3.semilogx(times * 1e3, residual, 'ro-', lw=1.5, markersize=5)
        ax3.axhline(0, color='k', linestyle='-', lw=0.5)
        ax3.axhline(2, color='gray', linestyle='--', lw=0.5)
        ax3.axhline(-2, color='gray', linestyle='--', lw=0.5)
        ax3.fill_between(times * 1e3, -2, 2, alpha=0.1, color='green')
        ax3.set_xlabel('Time (ms)')
        ax3.set_ylabel('Normalized Residual')
        ax3.set_title('Data Residuals (±2σ shaded)')
        ax3.grid(True, alpha=0.5)
        
        plt.tight_layout()
        
        vis_file = os.path.join(output_dir, "tdem_inversion_result.png")
        fig.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self._log_execution(f"Saved visualization to {vis_file}")
        return vis_file
    
    def _generate_forward_plot(self, times, dobs, dpred_clean, uncertainties,
                               output_dir: str) -> str:
        """Generate forward modeling plot."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.loglog(times * 1e3, np.abs(dpred_clean), 'b-', lw=2, label='Clean data')
        ax.loglog(times * 1e3, np.abs(dobs), 'ko', markersize=6, label='Noisy data')
        ax.fill_between(times * 1e3,
                        np.abs(dobs) - uncertainties,
                        np.abs(dobs) + uncertainties,
                        alpha=0.3, color='gray', label='Uncertainty')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('|Bz| (T)')
        ax.set_title('TDEM Forward Modeling')
        ax.legend()
        ax.grid(True, which='both', alpha=0.5)
        
        plt.tight_layout()
        
        vis_file = os.path.join(output_dir, "tdem_forward_result.png")
        fig.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return vis_file
    
    def _generate_hydro_plots(self, times, dobs, dpred_clean, uncertainties,
                              water_content, porosity, conductivity, thicknesses,
                              output_dir: str) -> str:
        """Generate hydro-to-TDEM conversion plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        
        # Calculate depths for plotting
        n_layers = len(conductivity)
        last_thickness = 5.0  # For visualization
        depths = np.cumsum(np.r_[0, thicknesses, last_thickness])
        
        # 1. Water content
        ax1 = axes[0]
        for i in range(n_layers):
            ax1.fill_betweenx([depths[i], depths[i+1]], 0, water_content[i], 
                             alpha=0.7, color='dodgerblue')
        ax1.set_xlabel('Water Content (-)')
        set_length_axis(ax1, "y", "Depth")
        ax1.set_title('Water Content')
        ax1.set_xlim(0, 0.5)
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 2. Porosity
        ax2 = axes[1]
        for i in range(n_layers):
            ax2.fill_betweenx([depths[i], depths[i+1]], 0, porosity[i],
                             alpha=0.7, color='steelblue')
        ax2.set_xlabel('Porosity (-)')
        set_length_axis(ax2, "y", "Depth")
        ax2.set_title('Porosity')
        ax2.set_xlim(0, 0.5)
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)
        
        # 3. Conductivity
        ax3 = axes[2]
        for i in range(n_layers):
            ax3.fill_betweenx([depths[i], depths[i+1]], 1e-5, conductivity[i],
                             alpha=0.7, color='red')
        ax3.set_xlabel('Conductivity (S/m)')
        set_length_axis(ax3, "y", "Depth")
        ax3.set_title('Conductivity')
        ax3.set_xscale('log')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3)
        
        # 4. TDEM response
        ax4 = axes[3]
        ax4.loglog(times * 1e3, np.abs(dpred_clean), 'b-', lw=2, label='Clean')
        ax4.loglog(times * 1e3, np.abs(dobs), 'ko', markersize=5, label='With noise')
        ax4.set_xlabel('Time (ms)')
        ax4.set_ylabel('|Bz| (T)')
        ax4.set_title('TDEM Response')
        ax4.legend()
        ax4.grid(True, which='both', alpha=0.5)
        
        plt.tight_layout()
        
        vis_file = os.path.join(output_dir, "hydro_to_tdem_result.png")
        fig.savefig(vis_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return vis_file
    
    #: Header names that mark a column as an uncertainty rather than a sounding.
    UNCERTAINTY_NAMES = ('unc', 'err', 'std', 'noise', 'sigma')

    def _load_tdem_data(self, data_file: str, sounding: int = 0
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Times, observations and uncertainties from a TDEM data file.

        Handles the two shapes these files come in, which the previous version
        handled neither of for the data shipped with this package:

        - **Delimiter.** It read with ``np.loadtxt``'s whitespace default, so a
          comma-separated file failed on its first row. The delimiter is now
          taken from the header line.
        - **What the third column is.** It assumed column 2 was an uncertainty.
          In a multi-sounding file - ``time_s, dBdt_E603036_N6413476,
          dBdt_E611529_N6405000, ...``, which is what
          ``examples/data/EM/skytem_bhmar_tdem.csv`` is - every column after the
          time is a separate sounding, and using the second one as the error on
          the first weights the inversion by another site's data. A column is
          treated as an uncertainty only when its name says so.

        Parameters
        ----------
        data_file : str
            Path to the data file. A header line is used when present.
        sounding : int
            Which sounding to invert, when the file holds several. Zero-based.

        Returns
        -------
        tuple of ndarray
            ``(times, dobs, uncertainties)``.

        Raises
        ------
        FileNotFoundError
            If the file is absent.
        ValueError
            If it cannot be parsed, has fewer than two columns, or does not
            contain the requested sounding.
        """
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"TDEM data file not found: {data_file}")

        try:
            first_line = data_path.read_text(encoding='utf-8',
                                             errors='replace').splitlines()[0]
        except (OSError, IndexError) as exc:
            raise ValueError(f"Could not read {data_file}: {exc}")

        delimiter = ',' if first_line.count(',') >= 1 else (
            '\t' if first_line.count('\t') >= 1 else None)
        header: List[str] = []
        try:
            float(first_line.split(delimiter)[0] if delimiter else first_line.split()[0])
            skip = 0
        except ValueError:
            header = [name.strip().lower() for name in
                      (first_line.split(delimiter) if delimiter else first_line.split())]
            skip = 1

        try:
            data = np.loadtxt(data_path, delimiter=delimiter, skiprows=skip,
                              comments='#')
        except Exception as exc:  # noqa: BLE001 - reported with the file name
            raise ValueError(f"Failed to load TDEM data from {data_file}: {exc}")

        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError(f"{data_file} has {data.shape[1]} column(s); TDEM data "
                             f"needs at least a time and a measurement column.")

        times = data[:, 0]
        uncertainty_column = next(
            (i for i, name in enumerate(header)
             if i > 0 and any(mark in name for mark in self.UNCERTAINTY_NAMES)), None)
        sounding_columns = [i for i in range(1, data.shape[1])
                            if i != uncertainty_column]
        if sounding >= len(sounding_columns):
            raise ValueError(f"{data_file} holds {len(sounding_columns)} sounding(s); "
                             f"sounding {sounding} was requested.")
        column = sounding_columns[sounding]
        dobs = data[:, column]
        if len(sounding_columns) > 1:
            name = header[column] if column < len(header) else f"column {column}"
            self._log_execution(f"{data_file} holds {len(sounding_columns)} soundings; "
                                f"inverting '{name}'. Set 'sounding' to choose another.")

        if uncertainty_column is not None:
            uncertainties = data[:, uncertainty_column]
        else:
            # 5% of the measurement plus a noise floor, so late-time gates whose
            # signal has decayed to nothing do not dominate the misfit.
            uncertainties = np.abs(dobs) * 0.05 + 1e-15
            self._log_execution("No uncertainty column found, estimating 5% of the "
                                "measurement plus a noise floor")
        return times, dobs, uncertainties
    
    def _interpret_results(self, result, times: np.ndarray) -> str:
        """Generate LLM interpretation of TDEM results."""
        try:
            # Calculate resistivity statistics
            resistivity = 1.0 / result.recovered_conductivity
            
            prompt = f"""Interpret these TDEM inversion results for a geophysics report:

TDEM Inversion Results:
- Number of layers: {len(result.recovered_conductivity)}
- Chi-squared misfit: {result.chi2:.3f}
- Time range: {times.min()*1e6:.1f} µs to {times.max()*1e3:.1f} ms
- Conductivity range: {result.recovered_conductivity.min():.4f} - {result.recovered_conductivity.max():.4f} S/m
- Resistivity range: {resistivity.min():.1f} - {resistivity.max():.1f} Ωm

Provide a brief interpretation (3-4 sentences) covering:
1. Data fit quality (chi-squared indicates over/under-fitting if far from 1.0)
2. What the conductivity structure suggests about subsurface geology
3. Reliability of the recovered model at different depths
4. Any recommendations for further analysis"""

            interpretation = self.query_llm(prompt, self.system_message,
                                           temperature=0.5, max_tokens=300)
            return interpretation
        except Exception as e:
            self._log_execution(f"Could not generate interpretation: {e}", level='WARNING')
            return f"TDEM inversion completed with chi² = {result.chi2:.3f}"
