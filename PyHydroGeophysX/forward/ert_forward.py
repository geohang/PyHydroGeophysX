"""
Forward modeling utilities for Electrical Resistivity Tomography (ERT).
"""
import numpy as np
import pygimli as pg
from pygimli.physics import ert
from typing import Tuple, Optional, Union


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
            log_transform: Whether resistivity_model is log-transformed
            
        Returns:
            Forward response (apparent resistivity)
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
            Tuple of (forward response, Jacobian matrix)
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
            # J_log = J * exp(m) / d = d(log(d))/d(log(m))
            J = np.exp(resistivity_model.ravel()) * J
            response_array = response.array()
            J = J / response_array.reshape(response_array.shape[0], 1)
            
            return np.log(response.array()), J
        
        return response.array(), J
    
    def get_coverage(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute coverage (resolution) for a given resistivity model.
        
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
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            pg.rrng.randpin.seed(seed)
        
        # Create electrode positions
        if ypos is None:
            # Create flat surface if no y-coordinates provided
            ypos = np.zeros_like(xpos)
        
        pos = np.hstack((xpos.reshape(-1, 1), ypos.reshape(-1, 1)))
        
        # Create ERT survey scheme
        scheme = ert.createData(elecs=pos, schemeName=schemeName)
        
        # Prepare mesh for forward modeling
        if mesh is not None:
            # Set all cells to same marker
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
        
        # Initialize a forward operator
        fwd_operator = cls(mesh=grid, data=scheme)
        
        # Forward response
        fob = ert.ERTModelling()
        fob.setData(scheme)
        fob.setMesh(grid)
        dr = fob.response(res_models)
        
        # Add noise
        dr *= 1. + pg.randn(dr.size()) * noise_level
        
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

def ertforward(fob, mesh, rhomodel, xr):
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


def ertforward2(fob, xr, mesh):
    """
    Simplified ERT forward model.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
    """
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1

    dr = fob.response(rhomodel)
    dr = np.log(dr)
    return dr


def ertforandjac(fob, rhomodel, xr):
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


def ertforandjac2(fob, xr, mesh):
    """
    Alternative ERT forward model and Jacobian using log-resistivity values.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        dr (np.ndarray): Log-transformed forward response.
        J (np.ndarray): Jacobian matrix.
    """
    xr1 = xr.copy()
    xr1 = np.exp(xr)
    rhomodel = xr1
    dr = fob.response(rhomodel)
    fob.createJacobian(rhomodel)
    J = fob.jacobian()
    J = pg.utils.gmat2numpy(J)
    J = np.exp(xr.T)*J
    dr = dr.array()
    J = J/dr.reshape(dr.shape[0],1)
    dr = np.log(dr)
    return dr, J


