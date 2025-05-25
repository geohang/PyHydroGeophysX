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
            resistivity_model (np.ndarray or pg.RVector): Resistivity model values.
                                                        Can be a NumPy array or a PyGIMLi RVector.
            log_transform (bool, optional): If True, assumes `resistivity_model` is log-transformed
                                          (natural log) and the returned response and Jacobian
                                          will also be for log-transformed data and model.
                                          Defaults to True.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Forward response (apparent resistivity). If `log_transform` is True, this is log(apparent resistivity).
                - Jacobian matrix. If `log_transform` is True, this is the Jacobian for
                  d(log(data)) / d(log(model)).
        """
        # Convert to PyGIMLi RVector if needed
        if isinstance(resistivity_model, np.ndarray):
            model_input_for_jacobian = resistivity_model.ravel() # Keep original scale for Jacobian transformation if log_transform
            model_for_fwd = pg.Vector(model_input_for_jacobian)
        else: # Assuming pg.RVector
            model_input_for_jacobian = resistivity_model.array() # Keep original scale
            model_for_fwd = resistivity_model
            
        # If input model is log-transformed, exponentiate for forward calculation
        if log_transform:
            model_physical = pg.Vector(np.exp(model_input_for_jacobian))
        else:
            model_physical = model_for_fwd # Already in physical scale
        
        # Calculate forward response using the physical model
        response_physical = self.fwd_operator.response(model_physical)
        
        # Create Jacobian matrix based on the physical model
        self.fwd_operator.createJacobian(model_physical)
        jacobian_physical = self.fwd_operator.jacobian()
        J_physical_np = pg.utils.gmat2numpy(jacobian_physical) # Convert GIMLi matrix to NumPy array
        
        # Process response and Jacobian based on log_transform flag
        if log_transform:
            # Jacobian for log-data / log-model sensitivity: J_log = d(log(data))/d(log(model))
            # This is derived from J_physical = d(data)/d(model) using chain rule:
            # J_log = (d(data)/data) / (d(model)/model) * J_physical
            # J_log = (model / data) * J_physical
            # Here, `model_input_for_jacobian` is log(model_physical), so exp(model_input_for_jacobian) is model_physical.
            # `response_physical.array()` is data_physical.
            
            # Multiply by model_physical values (element-wise for columns of J)
            J_transformed = J_physical_np * model_physical.array() # J_physical_np * exp(log_model)
            
            # Divide by data_physical values (element-wise for rows of J)
            response_physical_arr = response_physical.array()
            # Avoid division by zero or very small response values, which can cause instability
            response_physical_arr[response_physical_arr == 0] = 1e-12 # Replace zeros with a tiny number
            J_final = J_transformed / response_physical_arr.reshape(-1, 1) # Reshape for broadcasting over columns
            
            return np.log(response_physical_arr), J_final
        else:
            return response_physical.array(), J_physical_np
    
    def get_coverage(self, resistivity_model: np.ndarray, log_transform: bool = True) -> np.ndarray:
        """
        Compute coverage (resolution) for a given resistivity model.
        
        Args:
            resistivity_model (np.ndarray or pg.RVector): Resistivity model values for which to compute coverage.
                                                        Assumed to be in the physical domain (not log-transformed)
                                                        if `log_transform` is True, as it will be transformed.
            log_transform (bool, optional): Whether the input `resistivity_model` is log-transformed.
                                          The coverage calculation itself often uses log-transformed
                                          sensitivities and model parameters. Defaults to True.
            
        Returns:
            np.ndarray: Array of coverage values (log10 of weighted sensitivity density) for each cell in the mesh.
                        Coverage, often related to the diagonal of the resolution matrix, indicates the
                        sensitivity density or how well each model parameter (cell resistivity) is resolved
                        by the data. Higher values suggest better resolution.
        """
        # Convert to PyGIMLi RVector if needed, and handle log transformation
        if isinstance(resistivity_model, np.ndarray):
            model_input_physical = np.exp(resistivity_model.ravel()) if log_transform else resistivity_model.ravel()
        else: # Assuming pg.RVector
            model_input_physical = np.exp(resistivity_model.array()) if log_transform else resistivity_model.array()
        
        model_pgvector = pg.Vector(model_input_physical)
        
        # Calculate forward response and Jacobian using the physical model
        response_physical = self.fwd_operator.response(model_pgvector)
        self.fwd_operator.createJacobian(model_pgvector)
        jacobian_physical = self.fwd_operator.jacobian() # This is J = d(data)/d(model_physical)
        
        # Calculate coverage (sensitivity density)
        # pg.core.coverageDCtrans typically expects J, 1/data, 1/model (all physical scale)
        # or can handle log versions depending on its internal implementation.
        # Given the typical use of log-transforms in inversion for ERT:
        # If log_transform is True, it implies we are interested in d(log data)/d(log model) sensitivities.
        # The `coverageDCtrans` function might be designed for d(ln U)/d(ln rho).
        # If `model` was log(rho_physical) and `response` was log(data_physical)
        # then 1.0/response -> 1/log(data_physical) which is not standard.
        # It's more common to use 1.0/data_physical and 1.0/model_physical for weighting.
        
        # Assuming coverageDCtrans expects physical values for response and model for weighting:
        cov_trans = pg.core.coverageDCtrans(
            jacobian_physical, 
            1.0 / response_physical, # Weight by inverse of physical data
            1.0 / model_pgvector     # Weight by inverse of physical model parameters
        )
        
        # Weight by cell sizes. `covTrans` should be a vector of sensitivities per cell.
        # `paramSizes` should correspond to the size of each model parameter (cell).
        # The model is cell-based, so `len(model_pgvector)` is the number of cells.
        mesh_para_domain = self.fwd_operator.paraDomain() # The mesh used for parameters
        num_cells = mesh_para_domain.cellCount()
        
        if len(model_pgvector) != num_cells:
            raise ValueError("Resistivity model size does not match the number of cells in the parameter domain mesh.")

        param_sizes = np.zeros(num_cells)
        for i, cell in enumerate(mesh_para_domain.cells()):
            # Assuming the model parameters directly correspond to mesh cells in order.
            # If model parameters are region-based, this needs adjustment.
            # For cell-based models, covTrans should have one entry per cell.
            param_sizes[i] = cell.size() 
            # Original code: paramSizes[c.marker()] += c.size() summed sizes for cells with the same marker.
            # This is appropriate if the model is region-based (one parameter per marker).
            # If cell-based (one parameter per cell), then each cell's size is used individually.
            # The division `covTrans / paramSizes` implies paramSizes should be aligned with covTrans.
            # `pg.core.coverageDCtrans` typically returns a vector of length num_cells.

        if cov_trans.size() != num_cells:
             raise ValueError(f"Size of covTrans ({cov_trans.size()}) does not match number of cells ({num_cells}). Check model parameterization (cell/region).")

        # Perform division carefully if param_sizes can be zero (though cell sizes should be > 0)
        param_sizes[param_sizes == 0] = 1e-12 # Avoid division by zero for safety
        
        # Final coverage value, often presented in log scale.
        # This represents a weighted sensitivity or illumination for each cell.
        final_coverage = np.log10(cov_trans.array() / param_sizes) 
        
        return final_coverage
    
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
    dr (np.ndarray): Log-transformed forward response (log(apparent resistivity)).
    rhomodel (pg.RVector): The resistivity model vector used for the forward calculation
                           (after modification based on `xr`).

    Note:
        This function appears to be a specialized helper, possibly for an inversion routine
        where only a part of the model (cells with marker 2) is updated based on `xr`.
        `xr` is expected to be log-transformed resistivity values for these specific cells.
        The rest of `rhomodel` is converted to log, updated, then converted back to physical
        scale for the forward modeling, and the response is log-transformed.
        Consider if this function should be internal (e.g., `_ertforward`) or if its
        complexity warrants simplification or integration into the class structure.
    """
    # TODO: Evaluate for deprecation or making internal (e.g. _ertforward).

    # Take the natural logarithm of the input physical resistivity model (`rhomodel`).
    log_rhomodel_array = np.log(rhomodel.array())
    
    # Update the log-transformed resistivity values for cells with marker == 2.
    # `xr` is assumed to contain the new log-resistivity values for these specific cells.
    # It implies that `xr` should be of a size corresponding to the number of cells with marker 2.
    # The `np.exp(xr)` suggests `xr` itself is log-transformed, but then it's assigned to
    # `log_rhomodel_array`, which is confusing. If `xr` is already log(rho) for marker 2 cells,
    # then it should be `log_rhomodel_array[mesh.cellMarkers() == 2] = xr`.
    # If `xr` is physical resistivity for marker 2 cells, then it should be `np.log(xr)`.
    # Assuming `xr` contains new log-resistivity values for the subset of cells:
    # This line seems to intend to update a portion of the model.
    # However, `xr1[mesh.cellMarkers() == 2] = np.exp(xr)` implies xr is log(log(rho)) or similar.
    # Let's assume the original intent was: `xr` contains new *log-transformed* resistivity values
    # for the cells marked with 2.
    # If `xr` are the new log-resistivity values for the cells with marker 2:
    log_rhomodel_array[mesh.cellMarkers() == 2] = xr # Assuming xr corresponds to cells with marker 2
    
    # Convert the entire updated log-resistivity model back to physical resistivity.
    rhomodel_updated_physical = pg.matrix.RVector(np.exp(log_rhomodel_array))
    
    # Compute the forward response using the updated physical resistivity model.
    response_physical = fob.response(rhomodel_updated_physical)
    
    # Return the natural logarithm of the response and the updated physical model.
    return np.log(response_physical.array()), rhomodel_updated_physical


def ertforward2(fob, xr, mesh):
    """
    Simplified ERT forward model.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        np.ndarray: Log-transformed forward response (log(apparent resistivity)).

    Note:
        This function calculates the ERT forward response assuming the input `xr`
        is a complete log-transformed resistivity model for all cells in the mesh
        managed by `fob`.
        Consider if this function should be internal (e.g., `_ertforward2`) or if its
        functionality is sufficiently covered by the `ERTForwardModeling.forward` method
        with `log_transform=True` for the input model and obtaining the log of the output.
    """
    # TODO: Evaluate for deprecation or making internal (e.g. _ertforward2).

    # `xr` is assumed to be a log-transformed resistivity model (ln(rho)).
    # Convert log-transformed model `xr` to physical resistivity values.
    rhomodel_physical = np.exp(xr) # Element-wise exponentiation

    # Compute the forward response using the physical resistivity model.
    # `fob.response()` expects physical resistivity values.
    response_physical = fob.response(rhomodel_physical)
    
    # Return the natural logarithm of the response.
    log_response = np.log(response_physical.array())
    return log_response


def ertforandjac(fob, rhomodel, xr):
    """
    Forward model and Jacobian for ERT.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        rhomodel (pg.RVector): Resistivity model.
        xr (np.ndarray): Log-transformed model parameter.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dr (np.ndarray): Log-transformed forward response (log(apparent resistivity)).
            - J (np.ndarray): Jacobian matrix, transformed for log-data and log-model
                              sensitivity (d(log(data))/d(log(model))).

    Note:
        This function appears to be a specialized helper, possibly for an inversion routine.
        - `rhomodel` is an input physical resistivity model.
        - `xr` seems to be the log-transformed version of the part of `rhomodel` that is being perturbed
          or is of interest for the Jacobian calculation (e.g., cells with marker 2, if consistent with `ertforward`).
        The Jacobian transformation `J = np.exp(xr)*J / dr` suggests it's converting a physical
        Jacobian `d(data)/d(model_physical)` to `d(log(data))/d(log(model_perturbed_part))`.
        This is very similar to `ERTForwardModeling.forward_and_jacobian` with `log_transform=True`.
        Consider if this function should be internal or if its functionality is covered by the class method.
    """
    # TODO: Evaluate for deprecation or making internal.

    # `rhomodel` is assumed to be the current physical resistivity model.
    # `xr` is assumed to be the log-transformed version of the model parameters
    # for which the Jacobian sensitivities are scaled. Often, `xr` would be log(rhomodel).

    # Compute the forward response using the physical resistivity model.
    response_physical = fob.response(rhomodel)
    response_physical_arr = response_physical.array()

    # Compute the Jacobian matrix with respect to the physical model parameters.
    # J_physical = d(data_physical) / d(model_physical)
    fob.createJacobian(rhomodel)
    jacobian_gimli = fob.jacobian()
    J_physical_np = pg.utils.gmat2numpy(jacobian_gimli)

    # Transform the Jacobian for log-data and log-model sensitivities.
    # J_log_log = d(log(data)) / d(log(model))
    # J_log_log = (model_physical / data_physical) * J_physical
    # `np.exp(xr)` effectively converts the log-model `xr` to physical model values.
    # This assumes `xr` corresponds to the columns of `J_physical_np`.
    if J_physical_np.shape[1] == len(xr):
        model_physical_for_jacobian_scaling = np.exp(xr)
    else:
        # This case is problematic if xr doesn't match J's columns.
        # It might imply xr is only for a part of the model, but J is for the whole model.
        # Or, rhomodel should be used: model_physical_for_jacobian_scaling = rhomodel.array()
        # Original code used `np.exp(xr)*J`. If `xr` is log of the full model, this works.
        print(f"Warning in ertforandjac: Shape mismatch between Jacobian columns ({J_physical_np.shape[1]}) and xr ({len(xr)}). Using rhomodel for scaling.")
        model_physical_for_jacobian_scaling = rhomodel.array()


    # Avoid division by zero for response array
    response_physical_arr_safe = response_physical_arr.copy()
    response_physical_arr_safe[response_physical_arr_safe == 0] = 1e-12

    J_log_log = J_physical_np * model_physical_for_jacobian_scaling # Multiply by model_physical for each column
    J_log_log = J_log_log / response_physical_arr_safe.reshape(-1, 1)    # Divide by data_physical for each row
    
    # Return log-transformed response and the transformed Jacobian.
    log_response = np.log(response_physical_arr_safe)
    return log_response, J_log_log


def ertforandjac2(fob, xr, mesh):
    """
    Alternative ERT forward model and Jacobian using log-resistivity values.

    Args:
        fob (pygimli.ERTModelling): ERT forward operator.
        xr (np.ndarray): Log-transformed model parameter.
        mesh (pg.Mesh): Mesh for the forward model.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - dr (np.ndarray): Log-transformed forward response (log(apparent resistivity)).
            - J (np.ndarray): Jacobian matrix, transformed for log-data and log-model
                              sensitivity (d(log(data))/d(log(model))).

    Note:
        This function assumes `xr` is a 1D array representing the complete log-transformed
        resistivity model for all cells in the `mesh`.
        The operation `xr.T` on a 1D NumPy array has no effect; it returns the same 1D array.
        If `xr` were intended to be a column or row vector for matrix operations,
        it should be explicitly reshaped (e.g., `xr[:, np.newaxis]` or `xr[np.newaxis, :]`).
        The Jacobian transformation is consistent with calculating d(log(data))/d(log(model)).
        This function is very similar to `ERTForwardModeling.forward_and_jacobian`
        with `log_transform=True`. Consider for deprecation or internal use.
    """
    # TODO: Evaluate for deprecation or making internal.

    # `xr` is assumed to be a log-transformed resistivity model (ln(rho)) for all cells.
    # Convert log-transformed model `xr` to physical resistivity values.
    rhomodel_physical = np.exp(xr) # Element-wise exponentiation

    # Compute the forward response using the physical resistivity model.
    response_physical = fob.response(rhomodel_physical)
    response_physical_arr = response_physical.array()

    # Compute the Jacobian matrix with respect to the physical model parameters.
    # J_physical = d(data_physical) / d(model_physical)
    fob.createJacobian(rhomodel_physical)
    jacobian_gimli = fob.jacobian()
    J_physical_np = pg.utils.gmat2numpy(jacobian_gimli)

    # Transform the Jacobian for log-data and log-model sensitivities.
    # J_log_log = d(log(data)) / d(log(model))
    # J_log_log = (model_physical / data_physical) * J_physical
    
    # `xr.T` on a 1D NumPy array `xr` has no effect. It returns `xr`.
    # `np.exp(xr)` gives the physical model values.
    model_physical_values = np.exp(xr) 

    # Avoid division by zero for response array
    response_physical_arr_safe = response_physical_arr.copy()
    response_physical_arr_safe[response_physical_arr_safe == 0] = 1e-12

    J_log_log = J_physical_np * model_physical_values # Multiply by model_physical for each column
    J_log_log = J_log_log / response_physical_arr_safe.reshape(-1, 1) # Divide by data_physical for each row
    
    # Return log-transformed response and the transformed Jacobian.
    log_response = np.log(response_physical_arr_safe)
    return log_response, J_log_log


