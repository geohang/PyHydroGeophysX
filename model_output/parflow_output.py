"""
Module for processing ParFlow model outputs.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any

from .base import HydroModelOutput


class ParflowOutput(HydroModelOutput):
    """Base class for processing ParFlow outputs."""
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow output processor.
        
        Args:
            model_directory: Path to ParFlow simulation directory
            run_name: Name of the ParFlow run
        """
        super().__init__(model_directory)
        self.run_name = run_name
        
        try:
            import parflow
            from parflow.tools.io import read_pfb
            self.parflow_available = True
            self.read_pfb = read_pfb
        except ImportError:
            self.parflow_available = False
            raise ImportError("parflow is not available. Please install parflow with: pip install parflow")
        
        # Get available timesteps
        self.available_timesteps = self._get_available_timesteps()
    
    def _get_available_timesteps(self) -> List[int]:
        """
        Get list of available timesteps from ParFlow outputs.
        
        Returns:
            List of timestep indices
        """
        timesteps = []
        
        # Check for pressure, saturation, or other output files to determine timesteps
        satur_pattern = f"{self.run_name}.out.satur."
        press_pattern = f"{self.run_name}.out.press."
        
        for file in os.listdir(self.model_directory):
            try:
                if file.startswith(satur_pattern):
                    timestep = int(file.replace(satur_pattern, "").split(".")[0])
                    timesteps.append(timestep)
                elif file.startswith(press_pattern) and len(timesteps) == 0:
                    timestep = int(file.replace(press_pattern, "").split(".")[0])
                    timesteps.append(timestep)
            except (ValueError, IndexError):
                continue
        
        return sorted(timesteps)
    
    def get_pfb_dimensions(self, pfb_file: str) -> Tuple[int, int, int]:
        """
        Get dimensions of a PFB file.
        
        Args:
            pfb_file: Path to PFB file
            
        Returns:
            Tuple of (nz, ny, nx)
        """
        data = self.read_pfb(pfb_file)
        return data.shape


class ParflowSaturation(ParflowOutput):
    """Class for processing saturation data from ParFlow simulations."""
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow saturation processor.
        
        Args:
            model_directory: Path to ParFlow simulation directory
            run_name: Name of the ParFlow run
        """
        super().__init__(model_directory, run_name)
    
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load saturation data for a specific timestep.
        
        Args:
            timestep_idx: Index of the timestep to load
            
        Returns:
            3D array of saturation values (nz, ny, nx)
        """
        if not self.available_timesteps:
            raise ValueError("No timesteps available in the ParFlow output directory")
            
        if timestep_idx >= len(self.available_timesteps):
            raise ValueError(f"Timestep index {timestep_idx} out of range. Max: {len(self.available_timesteps)-1}")
        
        timestep = self.available_timesteps[timestep_idx]
        return self._load_saturation(timestep)
    
    def _load_saturation(self, timestep: int) -> np.ndarray:
        """
        Load saturation data for a specific timestep number.
        
        Args:
            timestep: Actual timestep number
            
        Returns:
            3D array of saturation values (nz, ny, nx)
        """
        # Construct the saturation file path
        satur_file = os.path.join(
            self.model_directory, 
            f"{self.run_name}.out.satur.{timestep:05d}.pfb"
        )
        
        # If file doesn't exist with 5-digit formatting, try other formats
        if not os.path.exists(satur_file):
            satur_file = os.path.join(
                self.model_directory, 
                f"{self.run_name}.out.satur.{timestep}.pfb"
            )
        
        # Read the saturation data
        try:
            saturation = self.read_pfb(satur_file)
            
            # Replace very small values with NaN (based on your example)
            saturation[saturation < -1e38] = np.nan
            
            return saturation
        except Exception as e:
            raise ValueError(f"Error loading saturation data: {str(e)}")
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load saturation data for a range of timesteps.
        
        Args:
            start_idx: Starting timestep index
            end_idx: Ending timestep index (exclusive)
            
        Returns:
            4D array of saturation values (nt, nz, ny, nx)
        """
        if not self.available_timesteps:
            raise ValueError("No timesteps available in the ParFlow output directory")
        
        if start_idx < 0:
            start_idx = 0
        
        if end_idx is None:
            end_idx = len(self.available_timesteps)
        else:
            end_idx = min(end_idx, len(self.available_timesteps))
        
        # List of timesteps to load
        timesteps_to_load = self.available_timesteps[start_idx:end_idx]
        
        if not timesteps_to_load:
            raise ValueError(f"No valid timesteps in range [{start_idx}, {end_idx})")
        
        # Load first timestep to get dimensions
        first_data = self._load_saturation(timesteps_to_load[0])
        
        # Initialize array to store all timesteps
        saturation_data = np.zeros((len(timesteps_to_load), *first_data.shape))
        
        # Store first timestep
        saturation_data[0] = first_data
        
        # Load remaining timesteps
        for i, timestep in enumerate(timesteps_to_load[1:], 1):
            saturation_data[i] = self._load_saturation(timestep)
        
        return saturation_data
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Get information about each timestep.
        
        Returns:
            List of tuples (timestep, time)
        """
        # For ParFlow, the timestep number corresponds to simulation time in most cases
        return [(t, float(t)) for t in self.available_timesteps]


class ParflowPorosity(ParflowOutput):
    """Class for processing porosity data from ParFlow simulations."""
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow porosity processor.
        
        Args:
            model_directory: Path to ParFlow simulation directory
            run_name: Name of the ParFlow run
        """
        super().__init__(model_directory, run_name)
    
    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from ParFlow model.
        
        Returns:
            3D array of porosity values (nz, ny, nx)
        """
        # Try different possible file names for porosity
        porosity_file_patterns = [
            os.path.join(self.model_directory, f"{self.run_name}.out.porosity.pfb"),
            os.path.join(self.model_directory, f"{self.run_name}.out.porosity"),
            os.path.join(self.model_directory, f"{self.run_name}.pf.porosity.pfb"),
            os.path.join(self.model_directory, f"{self.run_name}.pf.porosity")
        ]
        
        for file_pattern in porosity_file_patterns:
            if os.path.exists(file_pattern):
                try:
                    porosity = self.read_pfb(file_pattern)
                    porosity[porosity < -1e38] = np.nan
                    return porosity
                except Exception as e:
                    print(f"Warning: Error reading {file_pattern}: {str(e)}")
        
        # If we couldn't find the porosity file, raise an error
        raise ValueError(f"Could not find porosity file for run {self.run_name} in {self.model_directory}")
    
    def load_mask(self) -> np.ndarray:
        """
        Load porosity data from ParFlow model.
        
        Returns:
            3D array of porosity values (nz, ny, nx)
        """
        # Try different possible file names for porosity
        porosity_file_patterns = [
            os.path.join(self.model_directory, f"{self.run_name}.out.mask.pfb"),
            os.path.join(self.model_directory, f"{self.run_name}.out.mask"),
            os.path.join(self.model_directory, f"{self.run_name}.pf.mask.pfb"),
            os.path.join(self.model_directory, f"{self.run_name}.pf.mask")
        ]
        
        for file_pattern in porosity_file_patterns:
            if os.path.exists(file_pattern):
                try:
                    porosity = self.read_pfb(file_pattern)
                    porosity[porosity < -1e38] = np.nan
                    return porosity
                except Exception as e:
                    print(f"Warning: Error reading {file_pattern}: {str(e)}")
        
        # If we couldn't find the porosity file, raise an error
        raise ValueError(f"Could not find porosity file for run {self.run_name} in {self.model_directory}")
        
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity for a specific timestep.
        Note: For ParFlow, porosity is typically constant over time,
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
            4D array of porosity values (nt, nz, ny, nx) where all timesteps are identical
        """
        porosity = self.load_porosity()
        
        # Determine number of timesteps
        if end_idx is None:
            nt = len(self.available_timesteps)
        else:
            nt = min(end_idx - start_idx, len(self.available_timesteps))
        
        # Cap at 1 if no timesteps available
        nt = max(1, nt)
        
        # Stack porosity array for each timestep
        return np.tile(porosity[np.newaxis, :, :, :], (nt, 1, 1, 1))
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Get information about each timestep.
        
        Returns:
            List of tuples (timestep, time)
        """
        # Use the same timestep info as saturation
        return [(t, float(t)) for t in self.available_timesteps]