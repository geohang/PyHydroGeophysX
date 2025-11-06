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
            water_content: Water content array
            porosity: Porosity value(s)
            
        Returns:
            Saturation array
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