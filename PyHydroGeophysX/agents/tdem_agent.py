"""
TDEM (Time-Domain Electromagnetic) Agent

Agent for processing Time-Domain Electromagnetic data using SimPEG.
Supports forward modeling, inversion, and integration with hydrological models.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base_agent import BaseAgent


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
    
    Example:
        >>> agent = TDEMAgent()
        >>> result = agent.execute({
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
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _run_inversion(self, input_data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
        """Run TDEM inversion workflow."""
        from PyHydroGeophysX.inversion.tdem_inversion import TDEMInversion, TDEMInversionResult
        
        self._log_execution("Running TDEM inversion")
        
        # Load data
        data_file = input_data.get('data_file')
        if data_file is None:
            raise ValueError("data_file is required for inversion mode")
        
        times, dobs, uncertainties = self._load_tdem_data(data_file)
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
        if self.api_key:
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
            'l2_conductivity': result.l2_conductivity,
            'visualization_file': vis_file,
            'interpretation': interpretation,
            'output_dir': output_dir
        }
        
        return self.results
    
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
    
    def _load_tdem_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load TDEM data from text file.
        
        Expected format: TIME(s) BZ(T) UNCERTAINTY(T)
        Can also handle formats with only time and data columns.
        """
        data_path = Path(data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"TDEM data file not found: {data_file}")
        
        # Load data, skipping header lines starting with #
        data = np.loadtxt(data_file, comments='#')
        
        if data.ndim == 1:
            raise ValueError("Data file must have at least 2 columns (time, data)")
        
        times = data[:, 0]
        dobs = data[:, 1]
        
        # Handle uncertainty column
        if data.shape[1] >= 3:
            uncertainties = data[:, 2]
        else:
            # Estimate uncertainties as 5% of data
            uncertainties = 0.05 * np.abs(dobs)
            self._log_execution("No uncertainty column found, using 5% of data magnitude")
        
        return times, dobs, uncertainties
    
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
        ax1.set_ylabel('Depth (m)')
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
        ax1.set_ylabel('Depth (m)')
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
        ax2.set_ylabel('Depth (m)')
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
        ax3.set_ylabel('Depth (m)')
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
    
    def _load_tdem_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load TDEM data from text file.
        
        Expects file format with columns: TIME(s) BZ(T) UNCERTAINTY(T)
        Header line is skipped.
        
        Args:
            data_file: Path to data file
            
        Returns:
            Tuple of (times, dobs, uncertainties) arrays
        """
        data_path = Path(data_file)
        
        if not data_path.exists():
            raise FileNotFoundError(f"TDEM data file not found: {data_file}")
        
        try:
            # Load data, skipping header
            data = np.loadtxt(data_path, skiprows=1)
            
            if data.ndim == 1:
                raise ValueError("Data file must have at least 3 columns")
            
            times = data[:, 0]
            dobs = data[:, 1]
            
            # Uncertainties can be column 3 or estimated
            if data.shape[1] >= 3:
                uncertainties = data[:, 2]
            else:
                # Estimate uncertainties as 5% of data + noise floor
                uncertainties = np.abs(dobs) * 0.05 + 1e-15
                self._log_execution("No uncertainty column found, estimating 5% + noise floor")
            
            return times, dobs, uncertainties
            
        except Exception as e:
            raise ValueError(f"Failed to load TDEM data from {data_file}: {e}")
    
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
    
    def _log_execution(self, message: str, level: str = 'INFO'):
        """Log execution message."""
        print(f"[{self.name}] [{level}] {message}")
