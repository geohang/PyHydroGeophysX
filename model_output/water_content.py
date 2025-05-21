"""
Module for handling MODFLOW water content data.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List


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


class MODFLOWWaterContent:
    """Class for processing water content data from MODFLOW simulations."""
    
    def __init__(self, sim_ws: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.
        
        Args:
            sim_ws: Path to simulation workspace
            idomain: Domain array indicating active cells
        """
        self.sim_ws = sim_ws
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
        fpth = os.path.join(self.sim_ws, "WaterContent")
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
                binaryread(file, vartype)
                
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
                print(f"Reached end of file or error at timestep {timestep}")
                break
        
        file.close()
        
        return np.array(WC_tot)
    
    def calculate_saturation(self, water_content: np.ndarray, 
                           porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate saturation from water content and porosity.
        
        Args:
            water_content: Water content array
            porosity: Porosity value(s)
            
        Returns:
            Saturation array with same shape as water_content
        """
        # Handle scalar porosity
        if isinstance(porosity, (int, float)):
            saturation = water_content / porosity
        else:
            # Make sure porosity has compatible dimensions
            if porosity.ndim != water_content.ndim:
                if porosity.ndim == water_content.ndim - 1:
                    # Expand porosity for multiple timesteps
                    porosity = np.repeat(
                        porosity[np.newaxis, ...], 
                        water_content.shape[0], 
                        axis=0
                    )
                else:
                    raise ValueError("Porosity dimensions not compatible with water content")
            
            saturation = water_content / porosity
        
        # Ensure saturation is between 0 and 1
        saturation = np.clip(saturation, 0.0, 1.0)
        
        return saturation
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Get information about each timestep in the WaterContent file.
        
        Returns:
            List of tuples (kstp, kper, pertim, totim) for each timestep
        """
        # Open water content file
        fpth = os.path.join(self.sim_ws, "WaterContent")
        file = open(fpth, "rb")
        
        timestep_info = []
        nuzfcells = self.nuzfcells * 3  # Assuming 3 layers
        
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
