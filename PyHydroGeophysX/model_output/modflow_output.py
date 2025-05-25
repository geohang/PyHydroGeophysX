"""
Module for processing MODFLOW model outputs.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict

from .base import HydroModelOutput


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses NumPy to read from a binary file. This method is generally faster
    than using the `struct` module for reading binary data.

    Args:
        file (file object): Open file object in binary read mode (`'rb'`).
        vartype (type or list or np.dtype): Variable type to read.
            Can be a standard Python type (e.g., `str`, `int`, `float`),
            a NumPy dtype (e.g., `np.float32`, `np.int64`), or a list of tuples
            if reading structured arrays (e.g., `[('kstp', '<i4'), ('kper', '<i4')]`).
        shape (tuple, optional): Shape of the data to read if reading an array.
                                 Defaults to `(1,)`.
        charlen (int, optional): Length of character strings if `vartype` is `str`.
                                 Defaults to 16.

    Returns:
        np.ndarray or bytes or any: The read data.
            - If `vartype` is `str`, returns bytes.
            - If `nval` (total elements from `shape`) is 1 and `vartype` is not `str`,
              it returns a NumPy array containing a single element (e.g., `np.array([value])`).
              The original code had `result = result # [0]`, which implies it might have
              intended to return the scalar value directly, but `np.fromfile` with `nval=1`
              still returns an array. For consistency and to avoid potential issues if a scalar
              is expected by the caller, this behavior should be noted or adjusted if necessary.
            - Otherwise, returns a NumPy array reshaped to `shape`.
    """
    # Read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1) # Returns bytes
    else:
        # Calculate the total number of values to read based on the shape
        nval = np.prod(shape)
        # Read data from file using NumPy
        result = np.fromfile(file, vartype, nval) # Returns a 1D array or structured array
        
        # If only one value was read and it's not a structured array (vartype is simple dtype)
        if nval == 1 and isinstance(vartype, type): # Or check if vartype is a simple dtype
            # result is already a single-element NumPy array, e.g., array([value]).
            # Callers might expect a scalar if shape was (1,).
            # For now, it returns the array as is. Example: result[0] would give scalar.
            pass # result remains as np.array([value])
        elif nval > 1 : # If multiple values, reshape to the specified shape
            result = np.reshape(result, shape)
        # If nval == 1 and vartype is for a structured array, result is a single-element structured array.
        # If nval == 0, result is an empty array.
    return result


class MODFLOWWaterContent(HydroModelOutput):
    """Class for processing water content data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.
        
        Args:
            model_directory: Path to simulation workspace
            idomain: Domain array indicating active cells
        """
        super().__init__(model_directory)
        self.idomain = idomain # Active model domain array (typically 2D for UZF package)
        self.nrows, self.ncols = idomain.shape
        
        # Build a reverse lookup dictionary for UZF cell numbers (iuzno).
        # `iuzno` is the cell number used in the UZF package output files.
        # This dictionary maps the linear `iuzno` to (row, col) grid indices.
        # The comment "(only for first layer as in original code)" implies that `idomain`
        # represents the active cells in a way that is consistent across layers for UZF,
        # or that the UZF cell numbering itself is effectively 2D and applied per layer.
        # UZF cells are typically numbered sequentially for active cells in the grid.
        self.iuzno_dict_rev = {}
        iuzno_counter = 0 # UZF cell number counter
        for i in range(self.nrows):
            for j in range(self.ncols):
                if idomain[i, j] != 0: # Active cell in the idomain
                    self.iuzno_dict_rev[iuzno_counter] = (i, j)
                    iuzno_counter += 1
        
        # Store the number of active UZ flow cells in a single layer.
        self.nuzfcells = len(self.iuzno_dict_rev)
    
    def load_timestep(self, timestep_idx: int, nlay: int = 3) -> np.ndarray:
        """
        Load water content for a specific timestep.
        
        Args:
            timestep_idx: Index of the timestep to load
            nlay: Number of layers in the model
            
        Returns:
            Water content array with shape (nlay, nrows, ncols)
        """
        return self.load_time_range(timestep_idx, timestep_idx + 1, nlay)[0]
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                      nlay: int = 3) -> np.ndarray:
        """
        Load water content for a range of timesteps.
        
        Args:
            start_idx: Starting timestep index (default: 0)
            end_idx: Ending timestep index (exclusive, default: None loads all)
            nlay: Number of layers in the model (default: 3)
            
        Returns:
            Water content array with shape (timesteps, nlay, nrows, ncols)
        """
        # Calculate total UZ flow cells
        nuzfcells = self.nuzfcells * nlay
        
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")
        
        WC_tot = []
        
        # Skip to starting timestep
        for _ in range(start_idx):
            try:
                # Read header
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                binaryread(file, vartype)
                
                # Skip data for this timestep
                vartype = [("data", "<f8")]
                for _ in range(nuzfcells):
                    binaryread(file, vartype)
            except Exception:
                print(f"Error skipping to timestep {start_idx}")
                file.close()
                return np.array(WC_tot)
        
        # Read timesteps
        timestep = 0
        while True:
            # Break if we've read the requested number of timesteps
            if end_idx is not None and timestep >= (end_idx - start_idx):
                break
                
            try:
                # Read header information
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                header = binaryread(file, vartype)
                
                # Initialize water content array for this timestep with NaNs
                WC_arr = np.full((nlay, self.nrows, self.ncols), np.nan)
                
                # Define the data type for reading water content values (double precision float)
                data_vartype = [("data", "<f8")] # Structured array type for binaryread
                
                # Read water content data for each layer and each active UZF cell
                for k_layer in range(nlay): # Iterate through layers
                    for n_uzf_cell_idx in range(self.nuzfcells): # Iterate through active UZF cells in a layer
                        row_idx, col_idx = self.iuzno_dict_rev[n_uzf_cell_idx]
                        # binaryread with this vartype returns a structured numpy array, e.g., array([(value,)], dtype=[('data', '<f8')])
                        # To get the scalar value, access by field name 'data' and then the first element.
                        read_value_struct_array = binaryread(file, data_vartype)
                        WC_arr[k_layer, row_idx, col_idx] = read_value_struct_array['data'][0]
                
                WC_tot.append(WC_arr)
                timestep += 1
                
            except Exception as e:
                print(f"Reached end of file or error at timestep {timestep}: {str(e)}")
                break
        
        file.close()
        
        return np.array(WC_tot)
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Get information about each timestep in the WaterContent file.
        
        Returns:
            List[Tuple[int, int, float, float]]: List of tuples, where each tuple contains
                                                 (kstp, kper, pertim, totim) for a timestep.
                                                 kstp: Timestep number within the stress period.
                                                 kper: Stress period number.
                                                 pertim: Time within the current stress period.
                                                 totim: Total simulation time.
        """
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")
        
        timestep_info = []
        # Number of data points to skip per timestep. This assumes that the number of layers (nlay)
        # for which data is written in the WaterContent file is known or can be assumed.
        # The original code used a fixed nlay=3 here. This should ideally be consistent
        # with how load_time_range interprets nlay, or made a parameter if it can vary.
        # For now, let's assume nlay=3 as per original implicit assumption for skipping.
        # TODO: Parameterize nlay or fetch it dynamically if possible and file structure varies.
        nlay_assumed_for_skip = 3 
        nuzfcells_to_skip = self.nuzfcells * nlay_assumed_for_skip
        
        while True:
            try:
                # Read header information
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"), 
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                header = binaryread(file, vartype)
                
                # Extract timestep info
                kstp = header[0][0]
                kper = header[0][1]
                pertim = header[0][2]
                totim = header[0][3]
                
                timestep_info.append((kstp, kper, pertim, totim))
                
                # Skip data records for this timestep to get to the next header
                data_record_vartype = [("data", "<f8")]
                for _ in range(nuzfcells_to_skip):
                    binaryread(file, data_record_vartype)
                    
            except Exception: # Typically EOF or read error
                break
        
        file.close()
        return timestep_info


class MODFLOWPorosity(HydroModelOutput):
    """Class for processing porosity data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, model_name: str):
        """
        Initialize MODFLOWPorosity processor.
        
        Args:
            model_directory (str): Path to the MODFLOW simulation workspace (directory containing model files).
            model_name (str): Name of the MODFLOW model (e.g., the base name of the .nam file).
        """
        super().__init__(model_directory)
        # self.model_directory = model_directory # Already set in super().__init__
        self.model_name = model_name
        self.nlay = 1
        self.nrow = 1
        self.ncol = 1
        
        try:
            import flopy
            self.flopy_available = True
        except ImportError:
            self.flopy_available = False
            raise ImportError("flopy is required to load MODFLOW porosity data. Please install flopy.")
    
    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from a MODFLOW model.
        This method attempts to load porosity (specific yield, SY) or, as a fallback,
        specific storage (SS) from common MODFLOW packages using Flopy.
        It supports both MODFLOW 6 and earlier versions (MODFLOW-2005, NWT, etc.).

        The order of preference for obtaining porosity/SY is generally:
        1. For MODFLOW 6: Storage (STO) package's specific yield (`sy.array`).
        2. For legacy MODFLOW:
           a. Unstructured Unsaturated Flow (UPW) package's specific yield (`upw.sy.array`).
           b. Layer Property Flow (LPF) package's specific yield (`lpf.sy.array`).
        If specific yield is not found, it falls back to specific storage:
        3. For legacy MODFLOW:
           a. UPW package's specific storage (`upw.ss.array`).
           b. LPF package's specific storage (`lpf.ss.array`).
        If none of these are found, a warning is printed, and a default porosity of 0.3 is returned.
        
        Returns:
            np.ndarray: A 3D array of porosity values with shape (nlay, nrow, ncol).
                        If porosity cannot be loaded, returns an array filled with a default value (0.3).

        Raises:
            ImportError: If Flopy is not installed.
            ValueError: If there's an error during the Flopy model loading process.
        """
        if not self.flopy_available:
            # This check is also in __init__, but kept here for robustness as it's a public method.
            raise ImportError("flopy is required to load MODFLOW porosity data.")
            
        try:
            import flopy # Flopy import
            
            # Determine if the model is MODFLOW 6 by checking for common MF6 simulation files
            mf6_indicator_files = ["mfsim.nam", f"{self.model_name}.sim"]
            is_mf6 = any(os.path.exists(os.path.join(self.model_directory, f)) for f in mf6_indicator_files)
            
            if is_mf6:
                # MODFLOW 6 approach
                try:
                    # Load the MODFLOW 6 simulation
                    sim = flopy.mf6.MFSimulation.load(
                        sim_name=self.model_name,
                        sim_ws=self.model_directory,
                        exe_name="mf6",
                    )
                    
                    # Get the groundwater flow model
                    gwf = sim.get_model(self.model_name)
                    
                    # Try to get dimensions from DIS (Discretization) package
                    dis = gwf.get_package("DIS") # Common for structured grids
                    if dis is None: # Try DISV for unstructured grids if DIS is not found
                        dis = gwf.get_package("DISV") 
                    
                    if dis is not None:
                        if hasattr(dis, 'nlay'): self.nlay = dis.nlay.data
                        if hasattr(dis, 'nrow'): self.nrow = dis.nrow.data # For DIS package
                        if hasattr(dis, 'ncol'): self.ncol = dis.ncol.data # For DIS package
                        # For DISV, dimensions might be trickier (e.g. ncpl - number of cells per layer)
                        # For now, assuming DIS or similar structured grid info.
                    else:
                        print("Warning: DIS or DISV package not found in MF6 model. Cannot determine dimensions accurately.")
                        # Fallback or error needed if dimensions are crucial and not found
                    
                    # Try to get specific yield (sy) from the STO (Storage) package
                    sto = gwf.get_package("STO")
                    if sto is not None and hasattr(sto, 'sy') and sto.sy is not None:
                        # Ensure sy.array is accessed correctly, it might be a MultiArray instance
                        if hasattr(sto.sy, 'array'):
                            return sto.sy.array
                        elif isinstance(sto.sy.get_data(), np.ndarray) : # If it's a scalar MFArray
                             return np.full((self.nlay, self.nrow, self.ncol), sto.sy.get_data())

                    print("Warning: Specific Yield (SY) not found in STO package for MODFLOW 6 model.")
                    # Note: MODFLOW 6 STO package might also have SS (specific storage).
                    # If SY is primary, this path might need to check SS as a fallback if SY is None.
                        
                except Exception as e:
                    print(f"Error loading MODFLOW 6 model or its packages: {str(e)}")
                    
            else:
                # Legacy MODFLOW models (2005 or earlier)
                try:
                    # Load the model
                    model = flopy.modflow.Modflow.load(
                        f"{self.model_name}.nam",
                        model_ws=self.model_directory,
                        load_only=["UPW", "LPF", "DIS"],  # Load packages with porosity and dimensions
                        check=False
                    )
                    
                    # Get dimensions
                    self.nlay = model.nlay
                    self.nrow = model.nrow
                    self.ncol = model.ncol
                    
                    # Try to get porosity from UPW package first
                    if hasattr(model, 'upw') and model.upw is not None:
                        if hasattr(model.upw, 'sy'):
                            return model.upw.sy.array
                    
                    # Then try LPF package
                    if hasattr(model, 'lpf') and model.lpf is not None:
                        if hasattr(model.lpf, 'sy'):
                            return model.lpf.sy.array
                    
                    # If specific yield not found, try specific storage
                    if hasattr(model, 'upw') and model.upw is not None:
                        if hasattr(model.upw, 'ss'):
                            print("WARNING: Using specific storage as substitute for porosity")
                            return model.upw.ss.array
                    
                    if hasattr(model, 'lpf') and model.lpf is not None:
                        if hasattr(model.lpf, 'ss'):
                            print("WARNING: Using specific storage as substitute for porosity")
                            return model.lpf.ss.array
                            
                except Exception as e:
                    print(f"Error loading legacy MODFLOW model: {str(e)}")
            
            # If nothing found, use default value
            print("WARNING: No porosity data found in model. Using default value of 0.3")
            return np.ones((self.nlay, self.nrow, self.ncol)) * 0.3
                
        except Exception as e:
            raise ValueError(f"Error loading porosity data: {str(e)}")
    
    # Implement required abstract methods
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity for a specific timestep.
        Note: For MODFLOW, porosity is typically constant over time,
        so this returns the same array regardless of timestep.
        
        Args:
            timestep_idx: Index of the timestep (unused)
            
        Returns:
            3D array of porosity values
        """
        return self.load_porosity()
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load porosity for a range of timesteps.
        Since porosity is typically constant, this returns a stack of identical arrays.
        
        Args:
            start_idx: Starting timestep index (unused)
            end_idx: Ending timestep index (unused)
            
        Returns:
            4D array of porosity values (nt, nlay, nrow, ncol) where all timesteps are identical
        """
        porosity = self.load_porosity()
        
        # Determine number of timesteps
        nt = 1 if end_idx is None else (end_idx - start_idx)
        
        # Stack porosity array for each timestep
        return np.tile(porosity[np.newaxis, :, :, :], (nt, 1, 1, 1))
    
    def get_timestep_info(self) -> List[Tuple]:
        """
        Get information about each timestep in the model.
        Returns a minimal placeholder since porosity doesn't vary with time.
        
        Returns:
            List[Tuple[int, float, float]]: A list containing a single tuple with placeholder values,
                                            as porosity is considered static.
                                            Example: `[(0, 0.0, 0.0)]` (stress_period, time_in_period, total_time)
        """
        # Since porosity is static, return a single placeholder entry.
        # The values (0, 0, 0.0) are arbitrary and indicate a single "static" time state.
        # (kstp, kper, pertim, totim) -> (timestep_in_stress_period, stress_period, time_in_stress_period, total_simulation_time)
        # A more common representation might be (stress_period, time_within_period, total_time)
        return [(0, 0.0, 0.0)] 