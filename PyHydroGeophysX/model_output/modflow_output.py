"""
Module for processing MODFLOW model outputs.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict

from .base import HydroModelOutput


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file. This was found to be faster than the
    struct approach and is used as the default.

    Args:
        file: Open file object in binary read mode
        vartype: Variable type to read
        shape: Shape of the data to read (default: (1,))
        charlen: Length of character strings (default: 16)

    Returns:
        The read data
    """
    # Read a string variable of length charlen
    if vartype == str:
        result = file.read(charlen * 1)
    else:
        # Find the number of values
        nval = np.prod(shape)
        result = np.fromfile(file, vartype, nval)
        if nval == 1:
            result = result  # [0]
        else:
            result = np.reshape(result, shape)
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
        self.idomain = idomain
        self.nrows, self.ncols = idomain.shape
        
        # Build reverse lookup dictionary (only for first layer as in original code)
        self.iuzno_dict_rev = {}
        iuzno = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                if idomain[i, j] != 0:
                    self.iuzno_dict_rev[iuzno] = (i, j)
                    iuzno += 1
        
        # Store number of UZ flow cells
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
                
                # Initialize water content array for this timestep
                WC_arr = np.zeros((nlay, self.nrows, self.ncols)) * np.nan
                
                # Read water content data
                vartype = [("data", "<f8")]
                
                # Read data for each layer and cell
                for k in range(nlay):
                    for n in range(self.nuzfcells):
                        i, j = self.iuzno_dict_rev[n]
                        WC_arr[k, i, j] = np.array(binaryread(file, vartype).tolist())
                
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
            List of tuples (kstp, kper, pertim, totim) for each timestep
        """
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")
        
        timestep_info = []
        nuzfcells = self.nuzfcells * 3  # Assuming 3 layers by default
        
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
                
                # Skip data for this timestep
                vartype = [("data", "<f8")]
                for _ in range(nuzfcells):
                    binaryread(file, vartype)
                    
            except Exception:
                break
        
        file.close()
        return timestep_info


class MODFLOWPorosity(HydroModelOutput):
    """Class for processing porosity data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, model_name: str):
        """
        Initialize MODFLOWPorosity processor.
        
        Args:
            model_directory: Path to simulation workspace
            model_name: Name of the MODFLOW model
        """
        super().__init__(model_directory)
        self.model_directory = model_directory
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
        Load porosity data from MODFLOW model (supports both MODFLOW 6 and earlier versions).
        
        Returns:
            3D array of porosity values (nlay, nrow, ncol)
        """
        if not self.flopy_available:
            raise ImportError("flopy is required to load MODFLOW porosity data.")
            
        try:
            import flopy
            
            # Check if this is a MODFLOW 6 model
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
                    
                    # Try to get dimensions from DIS package
                    
                    dis = gwf.get_package("DIS")
                    self.nlay = dis.nlay.data
                    self.nrow = dis.nrow.data
                    self.ncol = dis.ncol.data
                    
                    # Try to get porosity from the STO (Storage) package
                    
                    sto = gwf.get_package("STO")
                    
                    return sto.sy.array
                        
                except Exception as e:
                    print(f"Error loading MODFLOW 6 model: {str(e)}")
                    
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
            List with single dummy timestep info
        """
        return [(0, 0, 0.0)]  # (stress_period, timestep, time)