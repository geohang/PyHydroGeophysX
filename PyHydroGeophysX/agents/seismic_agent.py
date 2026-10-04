"""
Seismic Data Processing Agent

Specialized agent for processing seismic refraction data and extracting velocity structures.
Supports standalone seismic refraction tomography (SRT) inversion workflows.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import _figstyle as figstyle
from .base_agent import AgentResult, BaseAgent
from PyHydroGeophysX.data_processing.survey_geometry import OriginMismatch
from PyHydroGeophysX.visualization.axis_units import set_section_axes


# ---------------------------------------------------------------------------
# Seismic Agent
# ---------------------------------------------------------------------------
class SeismicAgent(BaseAgent):
    """
    Agent specialized in seismic refraction tomography (SRT) processing.
    
    Uses PyGIMLI and PyHydroGeophysX seismic processing modules to invert 
    seismic travel time data and extract velocity interfaces for structural constraints.
    
    Supports two modes:
    - 'inversion': Load seismic data file and run SRT inversion
    - 'interface': Extract velocity interfaces from existing velocity model
    
    Example (with your own travel-time file):
        >>> agent = SeismicAgent()
        >>> result = agent.execute({  # doctest: +SKIP
        ...     'seismic_file': 'seismic_data.dat',
        ...     'velocity_threshold': 1200,
        ...     'output_dir': 'results/seismic'
        ... })
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 llm_provider: str = "openai"):
        """Initialize Seismic Agent."""
        super().__init__("seismic_processor", api_key, model, llm_provider)
        self.system_message = """You are an expert in seismic refraction tomography (SRT). 
Your role is to process seismic travel time data, perform velocity inversions, and 
extract geological structure interfaces. You understand velocity-depth relationships 
and how to identify layer boundaries. You can interpret velocity models in terms of 
geological materials: weathered regolith (<1200 m/s), fractured bedrock (1200-3000 m/s), 
and fresh bedrock (>3000 m/s)."""
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process seismic data and extract velocity structure.
        
        Args:
            input_data: Dictionary containing ``seismic_file`` or
                pre-loaded ``seismic_data`` (a path given as ``seismic_data``
                is read as ``seismic_file``), optional velocity thresholds,
                inversion parameters, an output directory, and an
                ``extract_interfaces`` flag. Supported inversion parameters
                include ``lam``, ``zWeight``, ``vTop``, ``vBottom``,
                ``paraDepth``, ``paraMaxCellSize``, and ``limits``.

        Returns:
            Dictionary containing velocity model, mesh, interfaces, and visualizations
        """
        self._log_execution("Starting seismic data processing")
        self._geometry_warnings, self._pick_warnings = [], []

        try:
            seismic_file = input_data.get('seismic_file')
            raw_seismic_file = input_data.get('raw_seismic_file')
            seismic_data = input_data.get('seismic_data')
            # AgentCoordinator documents seismic_data as a file, and the
            # workflow guide passes one; used as a loaded container, a path
            # failed with "'str' object is not callable".
            if isinstance(seismic_data, (str, os.PathLike)):
                seismic_file = seismic_file or str(seismic_data)
                seismic_data = None
            velocity_threshold = input_data.get('velocity_threshold', 1200)
            velocity_thresholds = input_data.get('velocity_thresholds', [velocity_threshold])
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/seismic')
            extract_interfaces = input_data.get('extract_interfaces', True)

            if raw_seismic_file is None and seismic_file is not None:
                if Path(str(seismic_file)).suffix.lower() in {'.sgy', '.segy'}:
                    raw_seismic_file = seismic_file

            if raw_seismic_file is not None:
                validation_error = self.validate_input_file(
                    raw_seismic_file,
                    supported_extensions=[".sgy", ".segy"],
                    field_name="raw_seismic_file",
                    max_size_mb=input_data.get("max_file_size_mb"),
                )
                if validation_error:
                    return validation_error
            elif seismic_file is not None:
                validation_error = self.validate_input_file(
                    seismic_file,
                    supported_extensions=[".dat", ".txt"],
                    field_name="seismic_file",
                    max_size_mb=input_data.get("max_file_size_mb"),
                )
                if validation_error:
                    return validation_error

            import pygimli as pg
            import pygimli.physics.traveltime as tt
            from pygimli.physics import TravelTimeManager

            os.makedirs(output_dir, exist_ok=True)
            raw_artifacts: Dict[str, Any] = {}

            if raw_seismic_file is not None:
                self._log_execution(f"Processing raw SEG-Y data from {Path(str(raw_seismic_file)).name}")
                seismic_file, raw_artifacts = self._process_raw_segy_to_traveltime(
                    raw_seismic_file=str(raw_seismic_file),
                    output_dir=output_dir,
                    input_data=input_data,
                )
            
            # Load seismic data from file if provided
            if seismic_file is not None:
                seismic_file_path = Path(seismic_file)
                if not seismic_file_path.exists():
                    raise FileNotFoundError(f"Seismic file not found: {seismic_file}")
                self._log_execution(f"Loading seismic data from {seismic_file_path.name}")
                seismic_data = tt.load(str(seismic_file_path))
            
            if seismic_data is None:
                raise ValueError("seismic_file or seismic_data is required")
            
            self._log_execution("Processing seismic tomography")
            
            # Get number of shots and receivers
            n_shots = len(set(seismic_data('s')))
            n_receivers = len(seismic_data.sensors())
            n_data = seismic_data.size()
            self._log_execution(f"Data: {n_shots} shots, {n_receivers} sensors, {n_data} travel times")
            
            # The model's regularisation may come from the LLM; the mesh and the
            # velocities come from the survey itself, which a generic
            # recommendation ("vTop 300-800 m/s for regolith") knows nothing of.
            if self.llm_enabled and not inversion_params:
                self._log_execution("Requesting LLM recommendations for seismic inversion")
                recommended = self._get_recommended_params(seismic_data)
                inversion_params = {key: recommended[key] for key in ('lam', 'zWeight')
                                    if key in recommended}
            scaled = self.survey_scaled_defaults(seismic_data)
            self._log_execution(
                "Settings scaled to the survey where not given: "
                + ", ".join(f"{key}={value}" for key, value in scaled.items()
                            if key not in inversion_params))

            lam = inversion_params.get('lam', 50)
            zWeight = inversion_params.get('zWeight', 0.2)
            vTop = inversion_params.get('vTop', scaled['vTop'])
            vBottom = inversion_params.get('vBottom', scaled['vBottom'])
            paraDepth = inversion_params.get('paraDepth', scaled['paraDepth'])
            paraMaxCellSize = inversion_params.get('paraMaxCellSize', scaled['paraMaxCellSize'])
            quality = inversion_params.get('quality', 32)
            limits = inversion_params.get('limits', scaled['limits'])

            self._log_execution(f"Inversion parameters: lam={lam}, zWeight={zWeight}, vTop={vTop}, "
                                f"vBottom={vBottom}, limits={limits}, paraDepth={paraDepth}, "
                                f"paraMaxCellSize={paraMaxCellSize}")
            
            # Create travel time manager and mesh
            self._log_execution("Creating inversion mesh...")
            TT = TravelTimeManager()
            mesh_inv = TT.createMesh(seismic_data, paraMaxCellSize=paraMaxCellSize, 
                                     quality=quality, paraDepth=paraDepth)
            self._log_execution(f"Mesh created: {mesh_inv.cellCount()} cells")
            
            # Run inversion
            self._log_execution("Running seismic inversion (this may take a few minutes)...")
            # A copy: pyGIMLi turns the list it is given into slowness in place,
            # so the settings recorded below read [1/8000, 1/86] instead of the
            # velocities - and a re-inversion from them bounded the model by those.
            TT.invert(seismic_data, mesh=mesh_inv, lam=lam, zWeight=zWeight,
                     vTop=vTop, vBottom=vBottom, verbose=1,
                     limits=list(limits) if limits is not None else limits)
            
            self._log_execution("Seismic inversion completed")
            
            # Get velocity model and coverage
            velocity_model = TT.model.array()
            try:
                coverage = TT.standardizedCoverage()
            except Exception:
                coverage = np.ones(mesh_inv.cellCount())
            
            velocity_range = [np.min(velocity_model), np.max(velocity_model)]
            self._log_execution(f"Velocity range: {velocity_range[0]:.0f} - {velocity_range[1]:.0f} m/s")
            # The fit, for the report: it stated a velocity model with no word
            # on how well it explains the picks.
            try:
                chi2, rrms = float(TT.inv.chi2()), float(TT.inv.relrms())
            except Exception:  # noqa: BLE001 - a missing statistic is reported as missing
                chi2 = rrms = None
            # The picks beside the model's times, and the errors they were
            # weighted by (relative, as pyGIMLi holds them), for the step that
            # evaluates the fit and draws it.
            fit = {}
            try:
                from ._raw_data import traveltime_table

                fit = {'traveltimes': traveltime_table(seismic_data),
                       'predicted_times': np.asarray(TT.inv.response, dtype=float),
                       'relative_errors': np.asarray(TT.inv.dataErrs, dtype=float),
                       'chi2_history': [float(v) for v in
                                        (getattr(TT.inv, 'chi2History', None) or [])]}
            except Exception:  # noqa: BLE001 - the evaluation falls back to chi2 alone
                fit = {}
            
            # Save velocity model
            np.save(os.path.join(output_dir, 'velocity_model.npy'), velocity_model)
            np.save(os.path.join(output_dir, 'coverage.npy'), coverage)
            mesh_inv.save(os.path.join(output_dir, 'seismic_mesh.bms'))
            
            # Extract velocity interfaces if requested
            interfaces = {}
            if extract_interfaces:
                self._log_execution("Extracting velocity interfaces...")
                interfaces = self.interfaces_from_model(mesh_inv, velocity_model,
                                                        velocity_thresholds, output_dir)

            # Generate visualization
            vis_file = self._generate_velocity_plot(
                TT, mesh_inv, velocity_model, coverage, seismic_data, 
                interfaces, velocity_thresholds, output_dir,
                length_unit=figstyle.style_from_config(input_data).length_unit,
            )
            
            # Get LLM interpretation
            interpretation = None
            if self.llm_enabled:
                self._log_execution("Generating interpretation of seismic results")
                interpretation = self._interpret_velocity_results(
                    velocity_model, velocity_range, interfaces, n_shots, n_receivers
                )
            
            self.results = {
                'status': 'success',
                'velocity_model': velocity_model,
                'mesh': mesh_inv,
                'coverage': coverage,
                'velocity_range': velocity_range,
                'interfaces': interfaces,
                'velocity_thresholds': velocity_thresholds,
                'n_shots': n_shots,
                'n_receivers': n_receivers,
                'n_data': n_data,
                'interpretation': interpretation,
                'visualization_file': vis_file,
                'output_dir': output_dir,
                # Off-end shots and the like: the work succeeded, and the caller
                # still has to be able to say what the geometry rests on.
                'geometry_warnings': list(getattr(self, '_geometry_warnings', [])),
                'pick_warnings': list(getattr(self, '_pick_warnings', [])),
                # Where the shots and geophones stand, to draw them again later.
                'sensors': np.asarray(seismic_data.sensors(), dtype=float),
                'extract_interfaces': bool(extract_interfaces),
                # What the report's method and results sections state.
                'source_file': str(seismic_file) if seismic_file is not None else None,
                'chi2': chi2,
                'rrms': rrms,
                **fit,
                'n_cells': int(mesh_inv.cellCount()),
                'inversion_params': {'lam': lam, 'zWeight': zWeight, 'vTop': vTop,
                                     'vBottom': vBottom, 'paraDepth': paraDepth,
                                     'paraMaxCellSize': paraMaxCellSize, 'limits': limits},
                'data_paths': [os.path.join(output_dir, name) for name in (
                    'velocity_model.npy', 'coverage.npy', 'seismic_mesh.bms')]
                + [os.path.join(output_dir, f'interface_{threshold}ms.txt')
                   for threshold in interfaces],
                **raw_artifacts,
            }

            return self.results
            
        except OriginMismatch as e:
            self.results = self._origin_failure(e)
            return self.results

        except Exception as e:
            self._log_execution(f"Error during seismic processing: {str(e)}", level='ERROR')
            self.results = AgentResult(
                status="failed",
                summary="Seismic data could not be processed.",
                data={},
                error=str(e),
                error_fix_hint="Check that seismic_file exists, has a supported extension, and matches the expected travel-time format.",
            )
            return self.results

    def _origin_failure(self, error: OriginMismatch) -> AgentResult:
        """The failed result for a coordinate file and SEG-Y headers in two frames.

        Recoverable, and only by someone who knows the survey: the two frames
        differ by a rigid translation, so the work can be repeated once told
        which origin to report. Carried out as structured data rather than a
        sentence so the caller can offer that choice instead of re-parsing the
        message.
        """
        self._log_execution(f"Geometry origin mismatch: {str(error)}", level='ERROR')
        return AgentResult(
            status="failed",
            summary="The coordinate file and the SEG-Y headers do not share an origin.",
            data={'origin_mismatch': {'shift': error.shift}},
            error=str(error),
            error_fix_hint="Say which origin to report - the coordinate file's or the SEG-Y headers' - or correct one of the two files.",
        )

    def pick_travel_times(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """First-arrival travel times picked from a raw SEG-Y file, for the inversion.

        The picking half of :meth:`execute`, on its own so a workflow can show
        it as a step and stop before the inversion when the picks are wrong.
        Takes ``raw_seismic_file``, ``output_dir``, ``geophone_file`` and
        ``topography_file``, ``align_origin`` and ``first_break_params`` as
        :meth:`execute` does; ``first_break_params['reciprocity_check']``
        (default True) leaves out shots whose times disagree with their
        reciprocals (:func:`~PyHydroGeophysX.data_processing.seismic.reciprocal_shot_check`),
        and ``first_break_params['neighbour_check']`` (default True) picks again
        the picks that disagree with the neighbouring shots' at the same geophone
        (:func:`~PyHydroGeophysX.data_processing.seismic.neighbour_shot_check`).

        Returns the travel-time file and the picks (``traveltime_file``,
        ``first_break_picks_file``, ``picks_figure``, ``gathers_figure`` - shot
        gathers as wiggles with their picks - ``n_picks``, ``dropped_shots``,
        ``segy_metadata``) with the geometry and picking
        warnings, or a failed result - with ``origin_mismatch`` when the
        coordinate file and the headers sit in two frames.
        """
        raw = input_data.get('raw_seismic_file')
        validation_error = self.validate_input_file(
            raw, supported_extensions=[".sgy", ".segy"], field_name="raw_seismic_file",
            max_size_mb=input_data.get("max_file_size_mb"))
        if validation_error:
            return validation_error
        output_dir = input_data.get('output_dir', 'results/seismic')
        os.makedirs(output_dir, exist_ok=True)
        self._geometry_warnings, self._pick_warnings = [], []
        try:
            self._log_execution(f"Picking first arrivals in {Path(str(raw)).name}")
            _, artifacts = self._process_raw_segy_to_traveltime(str(raw), output_dir, input_data)
        except OriginMismatch as e:
            return self._origin_failure(e)
        except Exception as e:  # noqa: BLE001 - reported as the step's failure
            self._log_execution(f"First-break picking failed: {e}", level='ERROR')
            return AgentResult(status="failed", summary="First arrivals could not be picked.",
                               data={}, error=str(e),
                               error_fix_hint="Check that the SEG-Y file holds shot gathers "
                                              "with source and receiver positions.")
        return {'status': 'success', **artifacts, 'output_dir': output_dir,
                'geometry_warnings': list(self._geometry_warnings),
                'pick_warnings': list(self._pick_warnings)}

    def _picks_figure(self, picks, dropped, output_dir: str,
                      input_data: Dict[str, Any], rejected=(), repicked=()) -> Optional[str]:
        """Travel time against receiver position, one curve per shot.

        Shots left out are dashed grey; single picks left out are red crosses;
        picks made again - along their shot's curve, or where the neighbouring
        shots put them - are ringed in orange.
        """
        from PyHydroGeophysX.visualization.axis_units import length_factor, normalize_length_unit

        try:
            unit = normalize_length_unit(figstyle.style_from_config(input_data).length_unit)
            factor = length_factor(unit)
            left_out = {round(float(shot['source_x']), 3) for shot in dropped}
            fig = figstyle.detached_figure((9, 5.5))
            ax = fig.add_subplot(1, 1, 1)
            import matplotlib

            shots = sorted({round(float(p.source_x), 3) for p in picks})
            cmap = matplotlib.colormaps['viridis']
            out = {(round(float(p.source_x), 3), round(float(p.receiver_x), 3)) for p in rejected}
            if rejected:
                ax.plot([float(p.receiver_x) * factor for p in rejected],
                        [1e3 * float(p.time_s) for p in rejected], 'x', color='tab:red', ms=6,
                        mew=1.5, zorder=5, label="pick left out")
            if repicked:
                ax.plot([float(p.receiver_x) * factor for p in repicked],
                        [1e3 * float(p.time_s) for p in repicked], 'o', mfc='none',
                        mec='tab:orange', ms=7, mew=1.3, zorder=6, label="picked again")
            for i, shot in enumerate(shots):
                mine = sorted((float(p.receiver_x), 1e3 * float(p.time_s)) for p in picks
                              if round(float(p.source_x), 3) == shot and p.time_s > 0
                              and (shot, round(float(p.receiver_x), 3)) not in out)
                if not mine:
                    continue
                x, t = np.array(mine).T
                bad = shot in left_out
                ax.plot(x * factor, t, '--' if bad else '-o', ms=2.5, lw=1.0,
                        color='0.6' if bad else cmap(i / max(len(shots) - 1, 1)),
                        label=f"{shot * factor:g} {unit}" + (" (left out)" if bad else ""))
                ax.plot(shot * factor, 0, 'v', color='0.6' if bad else 'k', ms=5)
            ax.invert_yaxis()
            ax.set_xlabel(f"Receiver position ({unit})")
            ax.set_ylabel("First-arrival time (ms)")
            ax.set_title("First-arrival picks, one curve per shot (triangles: shot positions)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, ncol=2, title="Shot at", title_fontsize=8,
                      loc='lower right', frameon=False)
            path = os.path.join(output_dir, 'seismic_first_breaks.png')
            fig.savefig(path, dpi=200, bbox_inches='tight')
            return path
        except Exception as e:  # noqa: BLE001 - a figure is not the picks
            self._log_execution(f"Could not draw the picks: {e}", level='WARNING')
            return None

    def _gathers_figure(self, dataset, picks, dropped, output_dir: str,
                        input_data: Dict[str, Any], rejected=(), repicked=(),
                        agc_window: float = 0.05, panels: int = 6) -> Optional[str]:
        """Shot gathers as wiggle traces with their picks: the evidence for the picks.

        Up to ``panels`` shots, spread evenly along the line, and every shot
        left out besides (to eight panels in all). Each trace is drawn at its
        receiver's position as variable-area wiggles, with the same automatic
        gain control as the picker saw and scaled to its own peak, down to
        one and a half times the latest pick. Picks kept are blue dots, picks
        made again (along their shot's curve, or against the neighbouring
        shots) are ringed in orange, single picks
        left out are red crosses; a shot left out has grey picks.
        """
        from PyHydroGeophysX.data_processing.seismic import apply_agc
        from PyHydroGeophysX.visualization.axis_units import length_factor, normalize_length_unit

        try:
            unit = normalize_length_unit(figstyle.style_from_config(input_data).length_unit)
            factor = length_factor(unit)
            dt = float(dataset.metadata.sample_interval_s)
            key = lambda p: (int(p.field_record), int(p.trace_index))  # noqa: E731
            out = {key(p) for p in rejected}
            again = {key(p) for p in repicked}
            left_out = {round(float(s['source_x']), 3): s for s in dropped}
            by_record: Dict[int, list] = {}
            for p in picks:
                if 0 <= int(p.trace_index) < dataset.traces.shape[1]:
                    by_record.setdefault(int(p.field_record), []).append(p)
            if not by_record:
                return None
            shot_x = {r: float(np.median([p.source_x for p in ps])) for r, ps in by_record.items()}
            order = sorted(by_record, key=lambda r: shot_x[r])
            chosen = {order[int(i)] for i in
                      np.unique(np.linspace(0, len(order) - 1, min(panels, len(order))).round())}
            chosen |= {r for r in order if round(shot_x[r], 3) in left_out}
            chosen = sorted(chosen, key=lambda r: shot_x[r])[:8]

            # A pick without a positive time is no arrival, and is not drawn as one.
            timed = lambda p: bool(np.isfinite(p.time_s) and p.time_s > 0)  # noqa: E731
            kept_times = [p.time_s for r in chosen for p in by_record[r]
                          if key(p) not in out and timed(p)]
            t_max = min(float(dataset.time[-1]),
                        max(1.5 * float(np.percentile(kept_times, 95)) if kept_times else 0.0,
                            20 * dt, 0.005))
            n_samples = min(dataset.traces.shape[0], int(t_max / dt) + 1)
            t_ms = 1e3 * dt * np.arange(n_samples)

            ncols = min(3, len(chosen))
            nrows = int(np.ceil(len(chosen) / ncols))
            fig = figstyle.detached_figure((4.3 * ncols, 3.9 * nrows + 0.6))
            axes = fig.subplots(nrows, ncols, squeeze=False)
            for ax in axes.flat[len(chosen):]:
                ax.set_visible(False)
            for ax, record in zip(axes.flat, chosen):
                mine = sorted(by_record[record], key=lambda p: p.receiver_x)
                columns = [int(p.trace_index) for p in mine]
                xs = np.array([float(p.receiver_x) for p in mine]) * factor
                traces = np.asarray(dataset.traces[:, columns], dtype=float)
                if agc_window and agc_window > 0:
                    traces = apply_agc(traces, dt=dt, window=float(agc_window))
                traces = traces[:n_samples]
                traces = traces / np.maximum(np.nanmax(np.abs(traces), axis=0), 1e-30)
                width = 0.5 * float(np.median(np.diff(xs))) if xs.size > 1 else 0.5 * factor
                bad_shot = round(shot_x[record], 3) in left_out
                for x, trace in zip(xs, traces.T):
                    wiggle = np.nan_to_num(np.clip(trace, -1.0, 1.0)) * width
                    ax.fill_betweenx(t_ms, x, x + wiggle, where=wiggle > 0, color='k', lw=0)
                    ax.plot(x + wiggle, t_ms, color='k', lw=0.35)
                good = [(x, p) for x, p in zip(xs, mine) if key(p) not in out and timed(p)]
                if good:
                    gx, gp = zip(*good)
                    ax.plot(gx, [1e3 * p.time_s for p in gp], 'o', ms=3.2,
                            color='0.55' if bad_shot else 'tab:blue', zorder=5)
                ring = [(x, p) for x, p in good if key(p) in again]
                if ring:
                    rx, rp = zip(*ring)
                    ax.plot(rx, [1e3 * p.time_s for p in rp], 'o', mfc='none', mec='tab:orange',
                            ms=7, mew=1.3, zorder=6)
                cross = [(x, p) for x, p in zip(xs, mine) if key(p) in out and timed(p)]
                if cross:
                    cx, cp = zip(*cross)
                    ax.plot(cx, [1e3 * p.time_s for p in cp], 'x', color='tab:red', ms=6,
                            mew=1.5, zorder=6)
                sx = shot_x[record] * factor
                lo, hi = xs.min() - width * 2, xs.max() + width * 2
                ax.set_xlim(min(lo, sx - width), max(hi, sx + width))
                ax.plot(sx, 0, 'v', color='0.6' if bad_shot else 'k', ms=7, clip_on=False, zorder=7)
                ax.set_ylim(t_ms[-1], 0)
                title = f"Shot at {sx:g} {unit}"
                if bad_shot:
                    shot = left_out[round(shot_x[record], 3)]
                    title += f" - left out ({shot['median_difference_ms']:g} ms off its reciprocals)"
                ax.set_title(title, fontsize=9)
                ax.tick_params(labelsize=8)
            for ax in axes[-1]:
                ax.set_xlabel(f"Receiver position ({unit})", fontsize=8)
            for ax in axes[:, 0]:
                ax.set_ylabel("Time (ms)", fontsize=8)
            from matplotlib.lines import Line2D
            handles = [Line2D([], [], ls='', marker='o', ms=4, color='tab:blue', label="pick kept"),
                       Line2D([], [], ls='', marker='o', mfc='none', mec='tab:orange', ms=7,
                              mew=1.3, label="picked again (shot's curve or neighbouring shots)"),
                       Line2D([], [], ls='', marker='x', color='tab:red', ms=6, mew=1.5,
                              label="pick left out"),
                       Line2D([], [], ls='', marker='v', color='k', ms=6, label="shot position")]
            fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8, frameon=False)
            gain = f"AGC {agc_window * 1e3:g} ms, " if agc_window and agc_window > 0 else ""
            fig.suptitle(f"Shot gathers with their first-arrival picks ({gain}each trace "
                         f"scaled to its peak)", fontsize=10)
            fig.tight_layout(rect=(0, 0.05, 1, 0.97))
            path = os.path.join(output_dir, 'seismic_shot_gathers.png')
            fig.savefig(path, dpi=170, bbox_inches='tight')
            return path
        except Exception as e:  # noqa: BLE001 - a figure is not the picks
            self._log_execution(f"Could not draw the shot gathers: {e}", level='WARNING')
            return None

    @staticmethod
    def survey_scaled_defaults(seismic_data) -> Dict[str, Any]:
        """Inversion settings scaled to this survey, for any the caller does not give.

        The fixed defaults (a 30 m deep mesh of 2 m cells, velocities from
        500 m/s at the surface to 5000 m/s, bounded below at 300 m/s) describe a
        line hundreds of metres long over regolith. On an 11.5 m spread of
        0.5 m geophones over soil with a 120 m/s surface layer they left the
        picks a median 4.9 ms from the model; scaled, 1.4 ms. Each setting is
        scaled only where the survey calls for it, so a long line keeps the
        fixed values:

        - ``paraDepth``: 0.4 of the line's length, at most 30 m;
        - ``paraMaxCellSize``: two sensor spacings, at most 2 m;
        - ``vTop``: the direct wave's velocity (offsets of two spacings or
          less), at most 500 m/s;
        - ``vBottom``: one and a half times the apparent velocity at the far
          offsets - about the refractor's velocity over flat layers - at least
          twice ``vTop`` and at most 5000 m/s;
        - ``limits``: from half the direct-wave velocity, at most 300 m/s, to
          8000 m/s.
        """
        sensors = np.asarray(seismic_data.sensors(), dtype=float)
        x = np.unique(np.round(sensors[:, 0], 6))
        span = float(x.max() - x.min()) if x.size > 1 else 0.0
        spacing = float(np.median(np.diff(x))) if x.size > 1 else 1.0
        s = np.asarray(seismic_data('s'), dtype=int)
        g = np.asarray(seismic_data('g'), dtype=int)
        t = np.asarray(seismic_data('t'), dtype=float)
        offset = np.abs(sensors[g, 0] - sensors[s, 0])
        usable = (t > 0) & (offset > 0)
        near = usable & (offset <= 2 * spacing + 1e-9)
        direct = float(np.median(offset[near] / t[near])) if near.any() else 500.0
        far = usable & (offset >= np.percentile(offset[usable], 67)) if usable.any() else usable
        apparent = None
        if far.sum() >= 3 and np.ptp(offset[far]) > 0:
            slope = np.polyfit(offset[far], t[far], 1)[0]
            apparent = 1.0 / slope if slope > 0 else None
        v_top = float(min(500.0, direct))
        v_bottom = float(min(5000.0, max(1.5 * apparent if apparent else 5000.0, 2.0 * v_top)))
        return {
            'paraDepth': round(min(30.0, 0.4 * span), 2) if span > 0 else 30.0,
            'paraMaxCellSize': round(min(2.0, 2.0 * spacing), 3),
            'vTop': round(v_top), 'vBottom': round(v_bottom),
            'limits': [round(min(300.0, 0.5 * direct)), 8000.0],
        }

    def interfaces_from_model(self, mesh, velocity_model, thresholds,
                              output_dir: str) -> Dict[float, Dict[str, Any]]:
        """Trace each threshold velocity through the model, and save it as x, z."""
        from PyHydroGeophysX.core.mesh_utils import extract_velocity_interface

        interfaces = {}
        for threshold in thresholds:
            self._log_execution(f"  Extracting interface at {threshold} m/s")
            try:
                smooth_x, smooth_z = extract_velocity_interface(
                    mesh, velocity_model, threshold=threshold, interval=5)
                interfaces[threshold] = {'x': smooth_x, 'z': smooth_z}
                interface_file = os.path.join(output_dir, f'interface_{threshold}ms.txt')
                np.savetxt(interface_file, np.c_[smooth_x, smooth_z],
                           header=f"X(m) Z(m) - Velocity interface at {threshold} m/s")
                self._log_execution(f"  Interface saved with {len(smooth_x)} points")
            except Exception as e:  # noqa: BLE001 - one threshold may not be crossed
                self._log_execution(f"  Could not extract interface at {threshold} m/s: {e}",
                                    level='WARNING')
        return interfaces

    def _process_raw_segy_to_traveltime(
        self,
        raw_seismic_file: str,
        output_dir: str,
        input_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert raw SEG-Y traces into PyGIMLi travel-time input."""

        from PyHydroGeophysX.data_processing.seismic import (
            apply_record_interval,
            export_first_breaks,
            first_breaks_to_traveltime,
            pick_and_correct,
            read_segy,
            screen_picks,
        )

        first_break_params = dict(input_data.get("first_break_params") or {})
        max_traces = input_data.get("raw_max_traces", first_break_params.get("max_traces"))
        if max_traces in ("", None):
            max_traces = None
        elif max_traces is not None:
            max_traces = int(max_traces)

        dataset = read_segy(raw_seismic_file, max_traces=max_traces)
        timing_notes = []
        if first_break_params.get("use_record_interval", True):
            note = apply_record_interval(dataset, raw_seismic_file)
            if note:
                timing_notes.append(note)

        place = None
        if input_data.get('geophone_file') or input_data.get('topography_file'):
            from PyHydroGeophysX.data_processing.survey_geometry import apply_pick_geometry
            geometry_notes: list = []

            def place(picks):
                return apply_pick_geometry(picks, input_data.get('geophone_file'),
                                           input_data.get('topography_file'),
                                           align_origin=input_data.get('align_origin'),
                                           warn=geometry_notes.append)
        # The picking the studio's Seismic page shares: threshold picks on the
        # traces with gain, placed on the survey's geometry, the stray ones
        # picked again along their shot's curve on the traces without gain.
        picks, repicked = pick_and_correct(
            dataset,
            agc_window=float(first_break_params.get("agc_window", 0.05)),
            bandpass=first_break_params.get("bandpass"),
            threshold=float(first_break_params.get("threshold", 0.2)),
            noise_multiplier=float(first_break_params.get("noise_multiplier", 5.0)),
            min_time=float(first_break_params.get("min_time", 0.0)),
            max_time=first_break_params.get("max_time", 0.15),
            polarity=float(first_break_params.get("polarity", 1.0)),
            place=place,
            repick=bool(first_break_params.get("repick", True)),
        )

        output = Path(output_dir)
        if place is not None:
            self._log_execution('Applied explicit receiver coordinates / x-z topographic profile before travel-time export')
            for note in geometry_notes:
                self._log_execution(note, level='WARNING')
            self._geometry_warnings = list(geometry_notes)
            if input_data.get('align_origin'):
                self._log_execution(
                    f"Coordinate file and SEG-Y headers were reconciled onto the "
                    f"{input_data['align_origin']} origin, as chosen by the user")
        # A pick that jumped to noise, and a shot with a timing error of its own
        # (wrong at every pick): the inversion can only spread either through
        # the model. What still strays after the re-pick is left out; then the
        # shot check sees clean curves; then each pick is held to the
        # neighbouring shots' at its geophone, and picked again there.
        picked = picks
        screen = screen_picks(
            picks, monotonic_check=bool(first_break_params.get("monotonic_check", True)),
            reciprocity_check=bool(first_break_params.get("reciprocity_check", True)),
            neighbour_check=bool(first_break_params.get("neighbour_check", True)),
            traces=dataset.traces, dt=dataset.metadata.sample_interval_s)
        picks, rejected, dropped = screen.kept, screen.rejected, screen.dropped
        nb_rejected, nb_repicked = screen.neighbour_rejected, screen.neighbour_repicked
        # The figures show every pick as it finally stands.
        trace_key = lambda p: (int(p.field_record), int(p.trace_index))  # noqa: E731
        final = {trace_key(p): p for p in list(picks) + list(nb_rejected)}
        shown = [final.get(trace_key(p), p) for p in picked]
        again = {trace_key(p): p for p in list(repicked) + list(nb_repicked)}
        shown_repicked = [final.get(key, p) for key, p in again.items()]
        shown_rejected = list(rejected) + list(nb_rejected)
        # In the order the work was done: timing, picks corrected, picks and
        # shots left out, the neighbouring shots.
        self._pick_warnings = list(timing_notes)
        if repicked:
            self._pick_warnings.append(
                f"{len(repicked)} of {len(picked)} automatic picks strayed from their shot's "
                f"first-arrival curve and were picked again along it (AIC picker, Maeda 1985).")
        if rejected:
            self._pick_warnings.append(
                f"{len(rejected)} of {len(picked)} automatic picks were left out for breaking "
                f"their shot's first-arrival curve (a first arrival cannot come earlier "
                f"farther from the shot); they are marked in the picks figure.")
        self._pick_warnings += [
            f"The shot at x={shot['source_x']:g} m was left out: its travel times differ "
            f"from their reciprocals by a median of {shot['median_difference_ms']:g} ms over "
            f"{shot['pairs']} pairs, a timing error of its own." for shot in dropped]
        if nb_repicked:
            one = len(nb_repicked) == 1
            self._pick_warnings.append(
                f"{len(nb_repicked)} {'pick' if one else 'picks'} disagreed with the neighbouring "
                f"shots' picks at the same geophone - one geophone's picks across the shots form "
                f"a first-arrival curve too, by reciprocity - and lay out of line in "
                f"{'its' if one else 'their'} own shot's; {'it was' if one else 'they were'} "
                f"picked again where the neighbours put {'it' if one else 'them'}.")
        if nb_rejected:
            one = len(nb_rejected) == 1
            self._pick_warnings.append(
                f"{len(nb_rejected)} {'pick' if one else 'picks'} "
                f"{'still ' if nb_repicked else ''}disagreed with the neighbouring shots' picks "
                f"at the same geophone and {'was' if one else 'were'} left out; "
                f"{'it is' if one else 'they are'} marked in the picks figure.")
        # A travel time is a positive time between two places; the export
        # drops the rest, so they are neither counted nor passed over in silence.
        def placed(p, role):
            return (round(float(getattr(p, f'{role}_x')), 6), round(float(getattr(p, f'{role}_z')), 6))
        usable = [p for p in picks if np.isfinite(p.time_s) and p.time_s > 0
                  and placed(p, 'source') != placed(p, 'receiver')]
        untimed = [p for p in picks if not (np.isfinite(p.time_s) and p.time_s > 0)
                   and placed(p, 'source') != placed(p, 'receiver')]
        if untimed:
            self._pick_warnings.append(
                f"{len(untimed)} traces gave no first arrival that the picker or the re-pick "
                f"could time, and are not in the travel times.")
        for note in self._pick_warnings:
            self._log_execution(note, level='WARNING')
        picks_csv = export_first_breaks(picks, str(output / "first_break_picks.csv"))
        traveltime_file = first_breaks_to_traveltime(
            picks,
            str(output / "seismic_traveltime_from_segy.dat"),
            receiver_spacing=float(first_break_params.get("receiver_spacing", 1.0)),
            shot_spacing=first_break_params.get("shot_spacing"),
        )
        picks_figure = self._picks_figure(shown, dropped, output_dir, input_data,
                                          rejected=shown_rejected, repicked=shown_repicked)
        gathers_figure = self._gathers_figure(
            dataset, shown, dropped, output_dir, input_data, rejected=shown_rejected,
            repicked=shown_repicked, agc_window=float(first_break_params.get("agc_window", 0.05)))

        self._log_execution(
            f"Raw SEG-Y processing exported {len(usable)} picks to {Path(traveltime_file).name}"
        )
        return traveltime_file, {
            "raw_seismic_file": raw_seismic_file,
            "traveltime_file": traveltime_file,
            "first_break_picks_file": picks_csv,
            "picks_figure": picks_figure,
            "gathers_figure": gathers_figure,
            "n_picks": len(usable),
            "picked_shots": len({round(float(p.source_x), 3) for p in usable}),
            "dropped_shots": dropped,
            "rejected_picks": len(rejected),
            "repicked_picks": len(repicked),
            "neighbour_rejected": len(nb_rejected),
            "neighbour_repicked": len(nb_repicked),
            "segy_metadata": {
                "sample_interval_us": dataset.metadata.sample_interval_us,
                "samples_per_trace": dataset.metadata.samples_per_trace,
                "format_code": dataset.metadata.format_code,
                "trace_count": dataset.metadata.trace_count,
            },
        }
    
    def _get_recommended_params(self, seismic_data) -> Dict[str, Any]:
        """
        Get LLM recommendations for seismic inversion parameters.
        
        Args:
            seismic_data: Seismic travel time data
            
        Returns:
            Recommended parameters dictionary
        """
        try:
            data_info = f"""
            Seismic Data Characteristics:
            - Data type: Travel time data
            - Expected geology: Regolith over bedrock
            """
            
            prompt = f"""Based on typical seismic refraction surveys over regolith-bedrock 
systems, recommend inversion parameters:

{data_info}

Provide recommendations for:
1. Lambda (regularization): typical range 20-100
2. zWeight (vertical regularization): typical range 0.1-0.5
3. vTop (top velocity): typical 300-800 m/s for soil/regolith
4. vBottom (bottom velocity): typical 3000-6000 m/s for bedrock

Return as: lam=XX, zWeight=XX, vTop=XX, vBottom=XX"""
            
            response = self.query_llm(prompt, self.system_message, temperature=0.3, max_tokens=200)
            
            # Parse response with robust error handling
            params = {'lam': 50, 'zWeight': 0.2, 'vTop': 500, 'vBottom': 5000}  # defaults
            
            try:
                import re
                for key in ['lam', 'zWeight', 'vTop', 'vBottom']:
                    pattern = rf'{key}[=:\s]+(\d+\.?\d*)'
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        params[key] = float(match.group(1))
            except (ValueError, AttributeError) as e:
                self._log_execution(f"Could not parse LLM response: {e}, using defaults")
            
            self._log_execution(f"LLM recommended parameters: {params}")
            
            return params
        except Exception as e:
            self._log_execution(f"Could not get LLM recommendations: {e}, using defaults")
            return {'lam': 50, 'zWeight': 0.2, 'vTop': 500, 'vBottom': 5000}
    
    def _generate_velocity_plot(self, TT, mesh_inv, velocity_model, coverage, seismic_data,
                                  interfaces: Dict, thresholds: list, output_dir: str,
                                  length_unit: Optional[str] = None,
                                  filename: str = 'seismic_velocity_model.png',
                                  title: str = 'Seismic Refraction Tomography - Velocity Model'
                                  ) -> str:
        """
        Generate publication-quality velocity tomogram visualization.
        
        Args:
            TT: TravelTimeManager with inversion results
            mesh_inv: Inversion mesh
            velocity_model: Velocity values array
            coverage: Coverage array
            seismic_data: Seismic data container
            interfaces: Dict of extracted interfaces {threshold: {'x': [...], 'z': [...]}}
            thresholds: List of velocity thresholds
            output_dir: Output directory
            
        Returns:
            Path to saved visualization file
        """
        import matplotlib
        import matplotlib.pyplot as plt
        import pygimli as pg

        from PyHydroGeophysX.core.mesh_utils import createTriangles, fill_holes_2d
        
        matplotlib.rcParams['font.family'] = 'Arial'
        matplotlib.rcParams['font.size'] = 12
        
        try:
            # Try to use BlueDarkRed colormap if available
            from palettable.lightbartlein.diverging import BlueDarkRed18_18
            cmap = BlueDarkRed18_18.mpl_colormap
        except ImportError:
            cmap = 'viridis'
        
        # Calculate dynamic colormap limits
        vel_min = np.percentile(velocity_model, 2)
        vel_max = np.percentile(velocity_model, 98)
        # A soil's surface layer can be slower than 300 m/s; a floor there drew
        # it in the same colour as everything up to the floor.
        cMin = max(50, vel_min * 0.9)
        cMax = min(8000, vel_max * 1.1)
        
        # Fill holes in coverage for better visualization
        pos = np.array(mesh_inv.cellCenters())
        try:
            filled_cov = fill_holes_2d(pos, coverage)
        except Exception:
            filled_cov = coverage
        
        # Detached from pyplot: this is often the run's first pg.show, and
        # pyGIMLi's first pg.plt use calls plt.show(), which blocks on an open
        # pyplot figure under an interactive backend - the seismic path hung a
        # fresh process under offscreen Qt this way.
        fig = figstyle.detached_figure((10, 8))
        ax = fig.add_subplot(1, 1, 1)

        # Plot velocity model
        figstyle.pg_show(mesh_inv, velocity_model, cMap=cmap, coverage=filled_cov, ax=ax,
                label='Velocity (m/s)', pad=0.3, cMin=cMin, cMax=cMax,
                orientation='vertical')
        
        # Add contour lines for velocity thresholds
        try:
            x, y, triangles, _, _ = createTriangles(mesh_inv)
            z = pg.meshtools.cellDataToNodeData(mesh_inv, velocity_model)
            
            linestyles = ['--', '-', '-.', ':']
            for i, threshold in enumerate(thresholds):
                ls = linestyles[i % len(linestyles)]
                ax.tricontour(x, y, triangles, z, levels=[threshold], 
                             linewidths=1.5, colors='k', linestyles=ls)
        except Exception as e:
            self._log_execution(f"Could not add contours: {e}", level='WARNING')
        
        # Plot extracted interfaces
        if interfaces:
            colors = ['white', 'cyan', 'yellow', 'magenta']
            for i, (threshold, data) in enumerate(interfaces.items()):
                color = colors[i % len(colors)]
                ax.plot(data['x'], data['z'], color=color, linewidth=2.5,
                       label=f'{threshold} m/s interface')
        
        # Draw sensors: a travel-time container, or the positions saved from one.
        try:
            positions = (seismic_data.sensors() if hasattr(seismic_data, 'sensors')
                         else [pg.Pos(*row) for row in np.asarray(seismic_data, dtype=float)])
            # A third of the sensor spacing, at most 0.8 m: a fixed 0.8 m covered
            # a 0.5 m spread with overlapping discs.
            xs = np.unique(np.round([float(p[0]) for p in positions], 6))
            diam = min(0.8, 0.3 * float(np.median(np.diff(xs)))) if xs.size > 1 else 0.8
            pg.viewer.mpl.drawSensors(ax, positions, diam=diam,
                                      facecolor='black', edgecolor='white')
        except Exception:
            pass

        # Depth, positive down, for a line surveyed without elevations.
        set_section_axes(ax, mesh=mesh_inv, unit=length_unit, fontsize=14)
        ax.set_title(title, fontsize=16)

        if interfaces:
            ax.legend(loc='lower right', fontsize=10)

        vis_file = os.path.join(output_dir, filename)
        fig.savefig(vis_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self._log_execution(f"Velocity plot saved: {vis_file}")
        return vis_file
    
    def _interpret_velocity_results(self, velocity_model: np.ndarray, velocity_range: list,
                                    interfaces: Dict, n_shots: int, n_receivers: int) -> str:
        """
        Get LLM interpretation of seismic velocity results.
        
        Args:
            velocity_model: Array of velocity values
            velocity_range: [min, max] velocity
            interfaces: Dict of extracted interfaces
            n_shots: Number of shot points
            n_receivers: Number of receiver positions
            
        Returns:
            Interpretation string
        """
        try:
            interface_summary = ""
            for threshold, data in interfaces.items():
                z_min = np.min(data['z']) if len(data['z']) > 0 else 'N/A'
                z_max = np.max(data['z']) if len(data['z']) > 0 else 'N/A'
                interface_summary += f"\n  - {threshold} m/s interface: depth range {z_min:.1f} to {z_max:.1f} m"
            
            results_summary = f"""
Seismic Refraction Inversion Results:
- Survey: {n_shots} shots, {n_receivers} sensors
- Velocity range: {velocity_range[0]:.0f} to {velocity_range[1]:.0f} m/s
- Extracted interfaces: {interface_summary if interface_summary else 'None extracted'}
"""
            
            prompt = f"""Interpret these seismic refraction tomography results:

{results_summary}

Geological context:
- Velocities < 1200 m/s typically indicate weathered soil/regolith
- Velocities 1200-3000 m/s suggest fractured rock
- Velocities > 3000 m/s indicate competent bedrock

Provide a concise interpretation (2-3 sentences) covering:
1. The subsurface geological structure revealed by the velocity model
2. The significance of the extracted interfaces for hydrogeological applications
3. Data quality assessment based on velocity range and interface extraction"""
            
            interpretation = self.query_llm(prompt, self.system_message, 
                                          temperature=0.5, max_tokens=300)
            return interpretation
        except Exception as e:
            self._log_execution(f"Could not generate interpretation: {e}", level='WARNING')
            return f"Seismic inversion completed. Velocity range: {velocity_range[0]:.0f} - {velocity_range[1]:.0f} m/s. " \
                   f"Extracted {len(interfaces)} velocity interfaces for structural analysis."
    
    def _interpret_results(self, TT_manager, interface_data) -> str:
        """
        Legacy method for backward compatibility.
        """
        try:
            velocity_model = TT_manager.model.array()
            velocity_range = [np.min(velocity_model), np.max(velocity_model)]
            return self._interpret_velocity_results(velocity_model, velocity_range, {}, 0, 0)
        except Exception:
            return "Could not generate interpretation"
