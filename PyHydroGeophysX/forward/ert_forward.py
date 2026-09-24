"""
Forward modeling utilities for Electrical Resistivity Tomography (ERT).
"""
from typing import Any, Optional, Tuple, Union

import numpy as np
import pygimli as pg
from pygimli.physics import ert


# ---------------------------------------------------------------------------
# ERTForward Modeling
# ---------------------------------------------------------------------------
class ERTForwardModeling:
    """Class for forward modeling of Electrical Resistivity Tomography (ERT) data."""
    
    def __init__(self, mesh: pg.Mesh, data: Optional[pg.DataContainer] = None):
        """
        Initialize ERT forward modeling.
        
        Args:
            mesh: PyGIMLI mesh for forward modeling
            data: ERT data container
        """
        self.mesh = mesh
        self.data = data
        self.fwd_operator = ert.ERTModelling()
        
        if data is not None:
            self.fwd_operator.setData(data)
        
        self.fwd_operator.setMesh(mesh)
    
    def set_data(self, data: pg.DataContainer) -> None:
        """
        Set ERT data for forward modeling.
        
        Args:
            data: ERT data container
        """
        self.data = data
        self.fwd_operator.setData(data)
    
    def set_mesh(self, mesh: pg.Mesh) -> None:
        """
        Set mesh for forward modeling.
        
        Args:
            mesh: PyGIMLI mesh
        """
        self.mesh = mesh
        self.fwd_operator.setMesh(mesh)
    
    def forward(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute forward response for a given resistivity model.

        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model contains natural-log
                resistivity; when True the response is also returned in natural-log space.

        Returns:
            Apparent resistivity (ohm-m), or its natural logarithm when
            log_transform=True.
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model

        # Apply exponentiation if log-transformed input
        if log_transform:
            # Validate before exponentiation
            model_array = np.array(model)
            if not np.all(np.isfinite(model_array)):
                raise ValueError(f"Non-finite values in log-resistivity model")

            model = pg.Vector(np.exp(model))

        # Validate resistivity values before PyGIMLi call
        model_array = np.array(model)
        if np.any(model_array <= 0):
            raise ValueError(f"Forward modeling received invalid resistivity values: min={np.min(model_array):.2e}, max={np.max(model_array):.2e}")

        # Calculate response
        response = self.fwd_operator.response(model)

        # Log-transform response if requested
        if log_transform:
            return np.log(response.array())

        return response.array()
    
    def forward_and_jacobian(self, resistivity_model: np.ndarray, log_transform: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward response and Jacobian matrix.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Tuple of (response, Jacobian). With log_transform=True these
            are ln(apparent resistivity) and its derivative with respect to
            ln(model resistivity); otherwise both use linear resistivity.
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model
            
        # Apply exponentiation if log-transformed input
        if log_transform:
            model = pg.Vector(np.exp(model))
        
        # Calculate response
        response = self.fwd_operator.response(model)
        
        # Create Jacobian matrix
        self.fwd_operator.createJacobian(model)
        jacobian = self.fwd_operator.jacobian()
        J = pg.utils.gmat2numpy(jacobian)
        
        # Process Jacobian according to log transformations
        if log_transform:
            # For log-transformed model and response
            # With m = ln(rho), J_log = d ln(d) / dm = J * rho / d.
            J = np.exp(resistivity_model.ravel()) * J
            response_array = response.array()
            J = J / response_array.reshape(response_array.shape[0], 1)
            
            return np.log(response.array()), J
        
        return response.array(), J
    
    def get_coverage(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute a sensitivity-based coverage proxy for each model cell.
        
        Args:
            resistivity_model: Resistivity model values
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Coverage values for each cell
        """
        # Convert to PyGIMLI RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model = pg.Vector(resistivity_model.ravel())
        else:
            model = resistivity_model
            
        # Apply exponentiation if log-transformed input
        if log_transform:
            model = pg.Vector(np.exp(model))
        
        # Calculate response and Jacobian
        response = self.fwd_operator.response(model)
        self.fwd_operator.createJacobian(model)
        
        # Calculate coverage
        covTrans = pg.core.coverageDCtrans(
            self.fwd_operator.jacobian(), 
            1.0 / response, 
            1.0 / model
        )
        
        # Weight by cell sizes
        paramSizes = np.zeros(len(model))
        mesh = self.fwd_operator.paraDomain
        
        for c in mesh.cells():
            paramSizes[c.marker()] += c.size()
            
        FinalJ = np.log10(covTrans / paramSizes)
        
        return FinalJ
    
    @classmethod
    def create_synthetic_data(cls, xpos: np.ndarray, 
                            ypos: Optional[np.ndarray] = None, 
                            mesh: Optional[pg.Mesh] = None, 
                            res_models: Optional[np.ndarray] = None, 
                            schemeName: str = 'wa', 
                            noise_level: float = 0.05, 
                            absolute_error: float = 0.0, 
                            relative_error: float = 0.05,
                            save_path: Optional[str] = None, 
                            show_data: bool = False, 
                            seed: Optional[int] = None,
                            xbound: float = 100, 
                            ybound: float = 100) -> Tuple[pg.DataContainer, pg.Mesh]:
        """
        Create synthetic ERT data using forward modeling.
        
        This method simulates an ERT survey by placing electrodes, creating a measurement 
        scheme, performing forward modeling to generate synthetic data, and adding noise.
        
        Args:
            xpos: X-coordinates of electrodes
            ypos: Y-coordinates of electrodes (if None, uses flat surface)
            mesh: Mesh for forward modeling
            res_models: Resistivity model values
            schemeName: Name of measurement scheme ('wa', 'dd', etc.)
            noise_level: Level of Gaussian noise to add
            absolute_error: Absolute error for data estimation
            relative_error: Relative error for data estimation
            save_path: Path to save synthetic data (if None, does not save)
            show_data: Whether to display data after creation
            seed: Random seed for noise generation
            xbound: X boundary extension for mesh
            ybound: Y boundary extension for mesh
            
        Returns:
            Tuple of (synthetic ERT data container, simulation mesh)
        """
        # Create electrode positions
        if ypos is None:
            # Create flat surface if no y-coordinates provided
            ypos = np.zeros_like(xpos)
        
        pos = np.hstack((xpos.reshape(-1, 1), ypos.reshape(-1, 1)))
        
        # Create ERT survey scheme
        scheme = ert.createData(elecs=pos, schemeName=schemeName)
        
        # Prepare mesh for forward modeling
        if mesh is not None:
            # Set all cells to same marker, on a copy: this used to relabel
            # the caller's own mesh, wiping whatever layer markers it carried.
            mesh = pg.Mesh(mesh)
            mesh.setCellMarkers(np.ones(mesh.cellCount()) * 2)
            
            # Append triangle boundary for forward modeling
            grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1,
                                                        xbound=xbound, ybound=ybound)
        else:
            # Create a simple mesh if none provided
            grid = pg.createGrid(
                x=np.linspace(np.min(xpos) - 10, np.max(xpos) + 10, 50),
                y=np.linspace(np.min(ypos) - 20, 0, 20)
            )
            grid = pg.meshtools.appendTriangleBoundary(grid, marker=1,
                                                        xbound=xbound, ybound=ybound)
            
            # Create homogeneous resistivity model if none provided
            if res_models is None:
                res_models = np.ones(grid.cellCount()) * 100
        
        # Create synthetic data
        synth_data = scheme.copy()
        
        # Forward response
        fob = ert.ERTModelling()
        fob.setData(scheme)
        fob.setMesh(grid)
        dr = fob.response(res_models)
        
        # Add noise. pg.randn draws from NumPy's global legacy RNG and reseeds
        # it when given a seed; the pg.rrng.randpin.seed call used before does
        # not exist, so any seed raised AttributeError. A private RandomState
        # makes the same draws as pg.randn(n, seed=seed) without resetting the
        # caller's global NumPy RNG.
        if seed is not None:
            noise = np.random.RandomState(seed).randn(dr.size())
        else:
            noise = pg.randn(dr.size())
        dr *= 1. + noise * noise_level
        
        # Set data and error values
        synth_data['rhoa'] = dr
        
        # Estimate error
        ert_manager = ert.ERTManager(synth_data)
        synth_data['err'] = ert_manager.estimateError(
            synth_data, absoluteUError=absolute_error, relativeError=relative_error
        )
        
        # Display data if requested
        if show_data:
            ert.showData(synth_data, logscale=True)
        
        # Save data if a path is provided
        if save_path is not None:
            synth_data.save(save_path)
        
        return synth_data, grid

# ---------------------------------------------------------------------------
# ertforward
# ---------------------------------------------------------------------------
def ertforward(
    fob: Any,
    mesh: Any,
    rhomodel: Any,
    xr: Any,
) -> Any:
    """
    Forward model for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        mesh (pg.Mesh): Mesh for the forward model.
        rhomodel (pg.RVector): Resistivity model vector.
        xr (np.ndarray): Log-transformed model parameter (resistivity).

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        rhomodel (pg.RVector): Updated resistivity model.
    """
    xr1 = np.log(rhomodel.array())
    xr1[mesh.cellMarkers() == 2] = np.exp(xr)
    rhomodel = pg.matrix.RVector(xr1)
    dr = fob.response(rhomodel)
    return np.log(dr.array()), rhomodel


# ---------------------------------------------------------------------------
# ertforward2
# ---------------------------------------------------------------------------
def ertforward2(
    fob: Any,
    xr: Any,
    mesh: Any,
    *,
    with_response: bool = False,
) -> Any:
    """
    Simplified ERT forward model.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.
        with_response: Also return the linear response the solve gave, which
            ``ertforandjac2(..., response=)`` can build the Jacobian from.

    Returns:
        dr (np.ndarray): Log-transformed forward response, and with
        ``with_response`` the linear response as a second value.
    """
    # Validate input before exponentiation
    if not np.all(np.isfinite(xr)):
        raise ValueError(f"Non-finite values in log-resistivity model: NaN={np.sum(np.isnan(xr))}, Inf={np.sum(np.isinf(xr))}")

    # Check bounds to prevent exp overflow/underflow
    if np.any(xr > 20) or np.any(xr < -20):
        print(f'WARNING: Extreme log-resistivity values detected: min={np.min(xr):.2f}, max={np.max(xr):.2f}')
        xr = np.clip(xr, -20, 20)

    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1

    # Validate resistivity model before PyGIMLi call
    if not np.all(rhomodel > 0):
        raise ValueError(f"Forward modeling received non-positive resistivity values: min={np.min(rhomodel):.2e}, max={np.max(rhomodel):.2e}")

    # Clip to physically reasonable range
    rhomodel = np.clip(rhomodel, 0.001, 1e6)

    dr = fob.response(rhomodel)
    if with_response:
        linear = np.array(dr, dtype=float)
        return np.log(linear), linear
    dr = np.log(dr)
    return dr


# ---------------------------------------------------------------------------
# ertforandjac
# ---------------------------------------------------------------------------
def ertforandjac(
    fob: Any,
    rhomodel: Any,
    xr: Any,
) -> Any:
    """
    Forward model and Jacobian for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        rhomodel (pg.RVector): Resistivity model.
        xr (np.ndarray): Log-transformed model parameter.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        J (np.ndarray): Jacobian matrix.
    """
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J


# ---------------------------------------------------------------------------
# ertforandjac2
# ---------------------------------------------------------------------------
def ertforandjac2(
    fob: Any,
    xr: Any,
    mesh: Any,
    *,
    response: Any = None,
    with_response: bool = False,
) -> Any:
    """
    Alternative ERT forward model and Jacobian using log-resistivity values.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.
        response: The linear response of ``fob``'s last forward solve, when
            that solve was for this very model and ``fob`` has solved nothing
            since - as ``ertforward2(..., with_response=True)`` just returned it.
            The solve is then not repeated. pyGIMLi's createJacobian reads the
            potentials of the operator's last response() call and does not
            check which model they belong to: given another model's, it returns
            a Jacobian of neither. So pass it only on that guarantee.
        with_response: Also return the linear response.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        J (np.ndarray): Jacobian matrix.
        With ``with_response``, the linear response as a third value.
    """
    # Validate input before exponentiation
    if not np.all(np.isfinite(xr)):
        raise ValueError(f"Non-finite values in log-resistivity model: NaN={np.sum(np.isnan(xr))}, Inf={np.sum(np.isinf(xr))}")

    # Check bounds to prevent exp overflow/underflow
    if np.any(xr > 20) or np.any(xr < -20):
        print(f'WARNING: Extreme log-resistivity values detected: min={np.min(xr):.2f}, max={np.max(xr):.2f}')
        xr = np.clip(xr, -20, 20)

    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1

    # Validate resistivity model before PyGIMLi call
    if not np.all(rhomodel > 0):
        raise ValueError(f"Forward modeling received non-positive resistivity values: min={np.min(rhomodel):.2e}, max={np.max(rhomodel):.2e}")

    # Clip to physically reasonable range
    rhomodel = np.clip(rhomodel, 0.001, 1e6)

    # The pyGIMLi vector is held until it is copied: an array taken from a
    # temporary one can outlive the memory it views.
    solved = fob.response(rhomodel) if response is None else response
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr.T)*J
    dr = np.array(solved, dtype=float)
    J = J/dr.reshape(dr.shape[0],1)
    if with_response:
        return np.log(dr), J, dr
    dr = np.log(dr)
    return dr, J


