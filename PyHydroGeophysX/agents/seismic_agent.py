"""
Seismic Data Processing Agent

Specialized agent for processing seismic refraction data and extracting velocity structures.
Supports standalone seismic refraction tomography (SRT) inversion workflows.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_agent import AgentResult, BaseAgent


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
    
    Example:
        >>> agent = SeismicAgent()
        >>> result = agent.execute({
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
            input_data: Dictionary containing:
                - seismic_file: Path to seismic travel time data file (.dat format)
                - seismic_data: Pre-loaded seismic data (alternative to seismic_file)
                - velocity_threshold: Threshold for interface detection (default: 1200 m/s)
                - velocity_thresholds: List of thresholds for multiple interfaces
                - inversion_params: Parameters for seismic inversion:
                    * lam: Regularization (default: 50)
                    * zWeight: Vertical smoothing (default: 0.2)
                    * vTop: Top velocity constraint (default: 500)
                    * vBottom: Bottom velocity constraint (default: 5000)
                    * paraDepth: Parametric depth (default: 30)
                    * paraMaxCellSize: Max cell size (default: 2)
                    * limits: [min_vel, max_vel] (default: [300, 8000])
                - output_dir: Directory for saving results
                - extract_interfaces: Whether to extract velocity interfaces (default: True)
                
        Returns:
            Dictionary containing velocity model, mesh, interfaces, and visualizations
        """
        self._log_execution("Starting seismic data processing")
        
        try:
            seismic_file = input_data.get('seismic_file')
            raw_seismic_file = input_data.get('raw_seismic_file')
            seismic_data = input_data.get('seismic_data')
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

            from PyHydroGeophysX.core.mesh_utils import (
                createTriangles,
                extract_velocity_interface,
                fill_holes_2d,
            )

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
            
            # Get LLM recommendations for inversion parameters if needed
            if self.api_key and not inversion_params:
                self._log_execution("Requesting LLM recommendations for seismic inversion")
                inversion_params = self._get_recommended_params(seismic_data)
            
            # Set default parameters
            lam = inversion_params.get('lam', 50)
            zWeight = inversion_params.get('zWeight', 0.2)
            vTop = inversion_params.get('vTop', 500)
            vBottom = inversion_params.get('vBottom', 5000)
            paraDepth = inversion_params.get('paraDepth', 30.0)
            paraMaxCellSize = inversion_params.get('paraMaxCellSize', 2)
            quality = inversion_params.get('quality', 32)
            limits = inversion_params.get('limits', [300., 8000.])
            
            self._log_execution(f"Inversion parameters: lam={lam}, zWeight={zWeight}, vTop={vTop}, vBottom={vBottom}")
            
            # Create travel time manager and mesh
            self._log_execution("Creating inversion mesh...")
            TT = TravelTimeManager()
            mesh_inv = TT.createMesh(seismic_data, paraMaxCellSize=paraMaxCellSize, 
                                     quality=quality, paraDepth=paraDepth)
            self._log_execution(f"Mesh created: {mesh_inv.cellCount()} cells")
            
            # Run inversion
            self._log_execution("Running seismic inversion (this may take a few minutes)...")
            TT.invert(seismic_data, mesh=mesh_inv, lam=lam, zWeight=zWeight,
                     vTop=vTop, vBottom=vBottom, verbose=1, limits=limits)
            
            self._log_execution("Seismic inversion completed")
            
            # Get velocity model and coverage
            velocity_model = TT.model.array()
            try:
                coverage = TT.standardizedCoverage()
            except Exception:
                coverage = np.ones(mesh_inv.cellCount())
            
            velocity_range = [np.min(velocity_model), np.max(velocity_model)]
            self._log_execution(f"Velocity range: {velocity_range[0]:.0f} - {velocity_range[1]:.0f} m/s")
            
            # Save velocity model
            np.save(os.path.join(output_dir, 'velocity_model.npy'), velocity_model)
            np.save(os.path.join(output_dir, 'coverage.npy'), coverage)
            mesh_inv.save(os.path.join(output_dir, 'seismic_mesh.bms'))
            
            # Extract velocity interfaces if requested
            interfaces = {}
            if extract_interfaces:
                self._log_execution("Extracting velocity interfaces...")
                for threshold in velocity_thresholds:
                    self._log_execution(f"  Extracting interface at {threshold} m/s")
                    try:
                        smooth_x, smooth_z = extract_velocity_interface(
                            mesh_inv, velocity_model, threshold=threshold, interval=5
                        )
                        interfaces[threshold] = {'x': smooth_x, 'z': smooth_z}
                        # Save interface
                        interface_file = os.path.join(output_dir, f'interface_{threshold}ms.txt')
                        np.savetxt(interface_file, np.c_[smooth_x, smooth_z], 
                                  header=f"X(m) Z(m) - Velocity interface at {threshold} m/s")
                        self._log_execution(f"  Interface saved with {len(smooth_x)} points")
                    except Exception as e:
                        self._log_execution(f"  Could not extract interface at {threshold} m/s: {e}", level='WARNING')
            
            # Generate visualization
            vis_file = self._generate_velocity_plot(
                TT, mesh_inv, velocity_model, coverage, seismic_data, 
                interfaces, velocity_thresholds, output_dir
            )
            
            # Get LLM interpretation
            interpretation = None
            if self.api_key:
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
                **raw_artifacts,
            }
            
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

    def _process_raw_segy_to_traveltime(
        self,
        raw_seismic_file: str,
        output_dir: str,
        input_data: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Convert raw SEG-Y traces into PyGIMLi travel-time input."""

        from PyHydroGeophysX.data_processing.seismic import (
            SeismicDataset,
            apply_agc,
            bandpass_filter,
            export_first_breaks,
            first_breaks_to_traveltime,
            pick_first_breaks,
            read_segy,
        )

        first_break_params = dict(input_data.get("first_break_params") or {})
        max_traces = input_data.get("raw_max_traces", first_break_params.get("max_traces"))
        if max_traces in ("", None):
            max_traces = None
        elif max_traces is not None:
            max_traces = int(max_traces)

        dataset = read_segy(raw_seismic_file, max_traces=max_traces)
        traces = dataset.traces
        dt = dataset.metadata.sample_interval_s

        agc_window = float(first_break_params.get("agc_window", 0.05))
        if agc_window > 0:
            traces = apply_agc(traces, dt=dt, window=agc_window)

        bandpass = first_break_params.get("bandpass")
        if bandpass:
            traces = bandpass_filter(
                traces,
                dt=dt,
                f1=float(bandpass[0]),
                f2=float(bandpass[1]),
                f3=float(bandpass[2]),
                f4=float(bandpass[3]),
            )

        processed = SeismicDataset(
            path=dataset.path,
            traces=traces,
            time=dataset.time,
            headers=dataset.headers,
            metadata=dataset.metadata,
        )
        picks = pick_first_breaks(
            processed,
            threshold=float(first_break_params.get("threshold", 0.2)),
            noise_multiplier=float(first_break_params.get("noise_multiplier", 5.0)),
            min_time=float(first_break_params.get("min_time", 0.0)),
            max_time=first_break_params.get("max_time", 0.15),
            polarity=float(first_break_params.get("polarity", 1.0)),
        )

        output = Path(output_dir)
        picks_csv = export_first_breaks(picks, str(output / "first_break_picks.csv"))
        traveltime_file = first_breaks_to_traveltime(
            picks,
            str(output / "seismic_traveltime_from_segy.dat"),
            receiver_spacing=float(first_break_params.get("receiver_spacing", 1.0)),
            shot_spacing=first_break_params.get("shot_spacing"),
        )

        self._log_execution(
            f"Raw SEG-Y processing exported {len(picks)} picks to {Path(traveltime_file).name}"
        )
        return traveltime_file, {
            "raw_seismic_file": raw_seismic_file,
            "traveltime_file": traveltime_file,
            "first_break_picks_file": picks_csv,
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
                                  interfaces: Dict, thresholds: list, output_dir: str) -> str:
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
        cMin = max(300, vel_min * 0.9)
        cMax = min(8000, vel_max * 1.1)
        
        # Fill holes in coverage for better visualization
        pos = np.array(mesh_inv.cellCenters())
        try:
            filled_cov = fill_holes_2d(pos, coverage)
        except Exception:
            filled_cov = coverage
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        
        # Plot velocity model
        pg.show(mesh_inv, velocity_model, cMap=cmap, coverage=filled_cov, ax=ax,
                label='Velocity (m/s)', xlabel='Distance (m)', ylabel='Elevation (m)',
                pad=0.3, cMin=cMin, cMax=cMax, orientation='vertical')
        
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
        
        # Draw sensors
        try:
            pg.viewer.mpl.drawSensors(ax, seismic_data.sensors(), diam=0.8,
                                      facecolor='black', edgecolor='white')
        except Exception:
            pass
        
        ax.set_xlabel('Distance (m)', fontsize=14)
        ax.set_ylabel('Elevation (m)', fontsize=14)
        ax.set_title('Seismic Refraction Tomography - Velocity Model', fontsize=16)
        
        if interfaces:
            ax.legend(loc='lower right', fontsize=10)
        
        vis_file = os.path.join(output_dir, 'seismic_velocity_model.png')
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
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
