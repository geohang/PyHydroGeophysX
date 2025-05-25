"""
Base classes for model output processing.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any


class HydroModelOutput(ABC):
    """Base class for all hydrological model outputs."""
    
    def __init__(self, model_directory: str):
        """
        Initialize model output processor.
        
        Args:
            model_directory: Path to model output directory
        """
        self.model_directory = model_directory
    
    @abstractmethod
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load data for a specific timestep.
        
        Args:
            timestep_idx: Index of the timestep to load
            **kwargs: Additional parameters specific to the model type
            
        Returns:
            Data array for the specified timestep
        """
        pass
    
    @abstractmethod
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load data for a range of timesteps.
        
        Args:
            start_idx: Starting timestep index
            end_idx: Ending timestep index (exclusive)
            **kwargs: Additional parameters specific to the model type
            
        Returns:
            Data array for the specified timestep range
        """
        pass
    
    @abstractmethod
    def get_timestep_info(self) -> List[Tuple]:
        """
        Get information about each timestep.
        
        Returns:
            List of timestep information tuples
        """
        pass
    
    def calculate_saturation(self, water_content: np.ndarray, 
                           porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate saturation from water content and porosity.
        
        Args:
            water_content (np.ndarray): Array of water content values. Can be N-dimensional,
                                        where the first dimension is typically time if multiple
                                        timesteps are present.
            porosity (Union[float, np.ndarray]): Porosity value(s). Can be a scalar float
                                                 or a NumPy array. If an array, its dimensions
                                                 should be compatible with `water_content`.
                                                 A common case is for `porosity` to be spatial
                                                 (e.g., (nz, ny, nx)) while `water_content` might be
                                                 spatio-temporal (e.g., (nt, nz, ny, nx)).
            
        Returns:
            np.ndarray: Saturation array, with values clipped between 0.0 and 1.0.

        Raises:
            ValueError: If `porosity` is an array and its dimensions are not compatible
                        with `water_content` dimensions for broadcasting.
                        Specifically, if `porosity.ndim == water_content.ndim - 1`,
                        it's assumed that `porosity` is spatial (e.g., for a 3D grid) and
                        `water_content` includes a time dimension as the first axis.
                        The porosity array is then repeated (broadcast) across the time dimension.
                        Other dimension mismatches will raise an error.
        """
        # Handle scalar porosity: directly divide.
        if isinstance(porosity, (int, float)):
            saturation = water_content / porosity
        else:
            # Make sure porosity has compatible dimensions
            if porosity.ndim != water_content.ndim:
                if porosity.ndim == water_content.ndim - 1:
                    # Assumed case: water_content is (time, z, y, x) and porosity is (z, y, x).
                    # Expand porosity to (1, z, y, x) and then repeat along the time axis (axis=0)
                    # to match water_content's shape for element-wise division.
                    porosity_expanded = np.repeat(
                        porosity[np.newaxis, ...],  # Add a new axis at the beginning: (1, z, y, x)
                        water_content.shape[0],     # Number of timesteps from water_content
                        axis=0                      # Repeat along the new time axis
                    )
                    porosity = porosity_expanded # Use the expanded porosity
                else:
                    raise ValueError(
                        f"Porosity dimensions (shape: {porosity.shape}) are not directly compatible "
                        f"with water content dimensions (shape: {water_content.shape}), "
                        "and the case for porosity.ndim == water_content.ndim - 1 (spatial porosity for temporal water content) "
                        "also does not match."
                    )
            
            # Element-wise division for saturation
            saturation = water_content / porosity
        
        # Ensure saturation is between 0 and 1
        saturation = np.clip(saturation, 0.0, 1.0)
        
        return saturation