"""
Seismic Data Processing Agent

Specialized agent for processing seismic refraction data and extracting velocity structures.
Supports standalone seismic refraction tomography (SRT) inversion workflows.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent


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
            import pygimli as pg
            import pygimli.physics.traveltime as tt
            from pygimli.physics import TravelTimeManager

            from PyHydroGeophysX.core.mesh_utils import (
                createTriangles,
                extract_velocity_interface,
                fill_holes_2d,
            )

            # Extract parameters
            seismic_file = input_data.get('seismic_file')
            seismic_data = input_data.get('seismic_data')
            velocity_threshold = input_data.get('velocity_threshold', 1200)
            velocity_thresholds = input_data.get('velocity_thresholds', [velocity_threshold])
            inversion_params = input_data.get('inversion_params', {})
            output_dir = input_data.get('output_dir', 'results/seismic')
            extract_interfaces = input_data.get('extract_interfaces', True)
            
            os.makedirs(output_dir, exist_ok=True)
            
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
                'output_dir': output_dir
            }
            
            return self.results
            
        except Exception as e:
            self._log_execution(f"Error during seismic processing: {str(e)}", level='ERROR')
            self.results = {
                'status': 'failed',
                'error': str(e)
            }
            raise
    
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
