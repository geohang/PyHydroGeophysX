"""
Inversion utilities for Time-Domain Electromagnetic (TDEM) data.

This module provides classes for 1D TDEM inversion using SimPEG,
with support for both L2 and sparse (IRLS) regularization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import simpeg.electromagnetics.time_domain as tdem

# SimPEG imports
from discretize import TensorMesh
from simpeg import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
from simpeg.utils import mkvc


# ---------------------------------------------------------------------------
# TDEMInversion Result
# ---------------------------------------------------------------------------
@dataclass
class TDEMInversionResult:
    """Container for TDEM inversion results.
    
    Attributes:
        recovered_model: Final recovered model (log-conductivity)
        recovered_conductivity: Recovered conductivity (S/m)
        l2_model: L2 model before IRLS (log-conductivity)
        l2_conductivity: L2 conductivity (S/m)
        predicted_data: Predicted data from recovered model
        mesh: TensorMesh used for inversion
        thicknesses: Layer thicknesses used in inversion
        chi2: Final chi-squared misfit
        iterations: Number of iterations
        convergence_history: History of data misfit per iteration
    """
    recovered_model: np.ndarray = None
    recovered_conductivity: np.ndarray = None
    l2_model: np.ndarray = None
    l2_conductivity: np.ndarray = None
    predicted_data: np.ndarray = None
    mesh: TensorMesh = None
    thicknesses: np.ndarray = None
    chi2: float = None
    iterations: int = 0
    convergence_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TDEMInversion
# ---------------------------------------------------------------------------
class TDEMInversion:
    """Class for 1D TDEM inversion using SimPEG.
    
    This class provides functionality for inverting TDEM sounding data
    to recover 1D layered Earth conductivity models.
    
    Example:
        >>> # Load or create data
        >>> times = np.logspace(-5, -2, 31)
        >>> dobs = ...  # observed data
        >>> uncertainties = ...  # data uncertainties
        >>> 
        >>> # Create inversion
        >>> inv = TDEMInversion(
        ...     times=times,
        ...     dobs=dobs,
        ...     uncertainties=uncertainties,
        ...     source_radius=10.0
        ... )
        >>> 
        >>> # Run inversion
        >>> result = inv.run()
    """
    
    def __init__(
        self,
        times: np.ndarray,
        dobs: np.ndarray,
        uncertainties: np.ndarray,
        source_radius: float = 10.0,
        source_location: Optional[np.ndarray] = None,
        receiver_location: Optional[np.ndarray] = None,
        n_layers: int = 25,
        min_thickness: float = 1.0,
        max_thickness: float = 30.0,
        **kwargs
    ):
        """
        Initialize TDEM inversion.
        
        Args:
            times: Time channels (s)
            dobs: Observed data (T)
            uncertainties: Data uncertainties (T)
            source_radius: Loop radius (m)
            source_location: Source center [x, y, z] (m)
            receiver_location: Receiver position [x, y, z] (m)
            n_layers: Number of layers for inversion mesh
            min_thickness: Minimum layer thickness (m)
            max_thickness: Maximum layer thickness (m)
            **kwargs: Additional parameters:
                - starting_conductivity: Starting model conductivity (S/m)
                - alpha_s: Smallness regularization weight
                - alpha_x: Smoothness regularization weight
                - max_iterations: Maximum iterations
                - use_irls: Whether to use IRLS for sparse inversion
                - irls_norms: [p, q] norms for IRLS (default [1, 0])
                - beta0_ratio: Initial beta ratio
                - verbose: Print progress
        """
        self.times = np.asarray(times)
        self.dobs = np.asarray(dobs)
        self.uncertainties = np.asarray(uncertainties)
        self.source_radius = source_radius
        
        if source_location is None:
            source_location = np.array([0.0, 0.0, 0.0])
        if receiver_location is None:
            receiver_location = np.array([0.0, 0.0, 0.0])
        
        self.source_location = source_location
        self.receiver_location = receiver_location
        
        # Inversion mesh parameters
        self.n_layers = n_layers
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness
        
        # Default parameters
        self.parameters = {
            'starting_conductivity': 0.01,  # S/m
            'alpha_s': 0.01,
            'alpha_x': 1.0,
            'max_iterations': 100,
            'use_irls': True,
            'irls_norms': [1, 0],
            'beta0_ratio': 1e2,
            'max_irls_iterations': 30,
            'cg_maxiter': 30,
            'verbose': True
        }
        self.parameters.update(kwargs)
        
        # Initialize components
        self.survey = None
        self.simulation = None
        self.mesh = None
        self.inv_thicknesses = None
        self._setup_complete = False
    
    def setup(self) -> None:
        """Set up inversion components (survey, mesh, simulation)."""
        # Create inversion layer thicknesses
        self.inv_thicknesses = np.logspace(
            np.log10(self.min_thickness),
            np.log10(self.max_thickness),
            self.n_layers
        )
        
        # Create mesh for regularization
        self.mesh = TensorMesh(
            [np.r_[self.inv_thicknesses, self.inv_thicknesses[-1]]], "0"
        )
        
        # Create receiver
        receiver_list = [
            tdem.receivers.PointMagneticFluxDensity(
                self.receiver_location,
                self.times,
                orientation="z"
            )
        ]
        
        # Create waveform and source
        waveform = tdem.sources.StepOffWaveform()
        source_list = [
            tdem.sources.CircularLoop(
                receiver_list=receiver_list,
                location=self.source_location,
                waveform=waveform,
                current=1.0,
                radius=self.source_radius,
            )
        ]
        
        # Create survey
        self.survey = tdem.Survey(source_list)
        
        # Create simulation with exponential mapping (model is log-conductivity)
        self.model_mapping = maps.ExpMap()
        self.simulation = tdem.Simulation1DLayered(
            survey=self.survey,
            thicknesses=self.inv_thicknesses,
            sigmaMap=self.model_mapping
        )
        
        self._setup_complete = True
        
        if self.parameters['verbose']:
            print(f"Inversion setup complete:")
            print(f"  - {self.n_layers} layers")
            print(f"  - Thickness range: {self.inv_thicknesses[0]:.2f} - {self.inv_thicknesses[-1]:.2f} m")
            print(f"  - {len(self.dobs)} data points")
    
    def run(
        self,
        starting_model: Optional[np.ndarray] = None
    ) -> TDEMInversionResult:
        """
        Run TDEM inversion.
        
        Args:
            starting_model: Initial log-conductivity model (optional)
            
        Returns:
            TDEMInversionResult with inversion results
        """
        if not self._setup_complete:
            self.setup()
        
        verbose = self.parameters['verbose']
        
        # Create starting model if not provided
        if starting_model is None:
            sigma_0 = self.parameters['starting_conductivity']
            starting_model = np.log(sigma_0) * np.ones(self.mesh.nC)
        
        if verbose:
            print(f"Starting model: {np.exp(starting_model[0]):.4f} S/m")
        
        # Create data object
        data_object = data.Data(
            self.survey,
            dobs=self.dobs,
            standard_deviation=self.uncertainties
        )
        
        # Data misfit
        dmis = data_misfit.L2DataMisfit(
            simulation=self.simulation,
            data=data_object
        )
        dmis.W = 1.0 / self.uncertainties
        
        # Regularization
        reg_map = maps.IdentityMap(nP=self.mesh.nC)
        reg = regularization.Sparse(
            self.mesh,
            mapping=reg_map,
            alpha_s=self.parameters['alpha_s'],
            alpha_x=self.parameters['alpha_x']
        )
        reg.reference_model = starting_model
        
        if self.parameters['use_irls']:
            reg.norms = self.parameters['irls_norms']
        
        # Optimization
        opt = optimization.ProjectedGNCG(
            maxIter=self.parameters['max_iterations'],
            maxIterLS=20,
            maxIterCG=self.parameters['cg_maxiter'],
            tolCG=1e-3
        )
        
        # Inverse problem
        inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
        
        # Directives
        directives_list = [
            directives.UpdateSensitivityWeights(),
            directives.BetaEstimate_ByEig(beta0_ratio=self.parameters['beta0_ratio']),
            directives.SaveOutputEveryIteration(save_txt=False),
        ]
        
        if self.parameters['use_irls']:
            directives_list.append(
                directives.UpdateIRLS(
                    max_irls_iterations=self.parameters['max_irls_iterations'],
                    irls_cooling_factor=1.5
                )
            )
        
        directives_list.append(directives.UpdatePreconditioner())
        
        # Create and run inversion
        inv = inversion.BaseInversion(inv_prob, directives_list)
        
        if verbose:
            print("\nRunning TDEM inversion...")
            print("=" * 60)
        
        recovered_model = inv.run(starting_model)
        
        if verbose:
            print("=" * 60)
            print("Inversion complete!")
        
        # Create result object
        result = TDEMInversionResult()
        result.recovered_model = recovered_model
        result.recovered_conductivity = self.model_mapping * recovered_model
        result.mesh = self.mesh
        result.thicknesses = self.inv_thicknesses
        
        # Get L2 model if available
        if hasattr(inv_prob, 'l2model') and inv_prob.l2model is not None:
            result.l2_model = inv_prob.l2model
            result.l2_conductivity = self.model_mapping * inv_prob.l2model
        
        # Predicted data
        result.predicted_data = self.simulation.dpred(recovered_model)
        
        # Compute chi-squared
        residual = (self.dobs - result.predicted_data) / self.uncertainties
        result.chi2 = np.sum(residual**2) / len(self.dobs)
        
        if verbose:
            print(f"\nFinal chi-squared: {result.chi2:.3f}")
        
        return result
    
    def plot_result(
        self,
        result: TDEMInversionResult,
        true_model: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot inversion results.
        
        Args:
            result: TDEMInversionResult from run()
            true_model: Tuple of (thicknesses, conductivity) for true model
            save_path: Path to save figure (optional)
        """
        import matplotlib.pyplot as plt
        from simpeg.utils import plot_1d_layer_model
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Model comparison
        ax1 = axes[0]
        
        # Plot true model if provided
        if true_model is not None:
            true_thick, true_cond = true_model
            plot_thick = np.r_[true_thick, 50.0]
            plot_1d_layer_model(plot_thick, true_cond, ax=ax1, show_layers=False, 
                              color='k', linewidth=3)
        
        # Plot L2 model if available
        if result.l2_conductivity is not None:
            plot_1d_layer_model(result.mesh.h[0], result.l2_conductivity, 
                              ax=ax1, show_layers=False, color='b', linewidth=2)
        
        # Plot sparse model
        plot_1d_layer_model(result.mesh.h[0], result.recovered_conductivity,
                          ax=ax1, show_layers=False, color='r', linewidth=2)
        
        ax1.set_xscale('log')
        ax1.set_xlabel('Conductivity (S/m)')
        ax1.set_ylabel('Depth (m)')
        ax1.set_title('Recovered Conductivity Model')
        
        legend_labels = []
        if true_model is not None:
            legend_labels.append('True Model')
        if result.l2_conductivity is not None:
            legend_labels.append('L2 Model')
        legend_labels.append('Sparse Model')
        ax1.legend(legend_labels)
        ax1.grid(True)
        ax1.invert_yaxis()
        
        # Data comparison
        ax2 = axes[1]
        ax2.loglog(self.times * 1e3, np.abs(self.dobs), 'ko', markersize=6, label='Observed')
        ax2.loglog(self.times * 1e3, np.abs(result.predicted_data), 'r-', lw=2, label='Predicted')
        
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('|Bz| (T)')
        ax2.set_title(f'Data Fit (χ² = {result.chi2:.2f})')
        ax2.legend()
        ax2.grid(True, which='both', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
        
        plt.show()


# ---------------------------------------------------------------------------
# run tdem inversion
# ---------------------------------------------------------------------------
def run_tdem_inversion(
    times: np.ndarray,
    dobs: np.ndarray,
    uncertainties: np.ndarray,
    source_radius: float = 10.0,
    n_layers: int = 25,
    use_irls: bool = True,
    verbose: bool = True,
    **kwargs
) -> TDEMInversionResult:
    """
    Convenience function to run TDEM inversion.
    
    Args:
        times: Time channels (s)
        dobs: Observed data (T)
        uncertainties: Data uncertainties (T)
        source_radius: Loop radius (m)
        n_layers: Number of inversion layers
        use_irls: Use IRLS for sparse inversion
        verbose: Print progress
        **kwargs: Additional parameters for TDEMInversion
        
    Returns:
        TDEMInversionResult with inversion results
    """
    inv = TDEMInversion(
        times=times,
        dobs=dobs,
        uncertainties=uncertainties,
        source_radius=source_radius,
        n_layers=n_layers,
        use_irls=use_irls,
        verbose=verbose,
        **kwargs
    )
    
    return inv.run()
