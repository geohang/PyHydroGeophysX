"""
Base classes for hydrological model output processing.

This module defines an abstract base class `HydroModelOutput` that outlines
the common interface for interacting with outputs from various hydrological models.
It provides a structure for loading time-series data and includes a utility
function for calculating saturation.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional, Union, List, Any # Removed Dict as it's not used


class HydroModelOutput(ABC):
    """
    Abstract base class for all hydrological model output processors.

    This class defines a common interface that all specific model output
    processor classes (e.g., for MODFLOW, ParFlow) should implement.
    It ensures consistency in how data is loaded and accessed across different
    hydrological models.
    """

    def __init__(self, model_directory: str):
        """
        Initialize the model output processor.

        Args:
            model_directory (str): The path to the directory containing the
                                   hydrological model's output files.
        """
        if not isinstance(model_directory, str):
            # Potential Issue: Consider checking if the directory exists.
            # However, this might be premature if the directory is created later
            # or if the path is to a remote resource not yet accessible.
            raise TypeError("model_directory must be a string path.")
        self.model_directory = model_directory

    @abstractmethod
    def load_timestep(self, timestep_idx: int, **kwargs: Any) -> np.ndarray:
        """
        Load and return data for a specific timestep.

        This method must be implemented by subclasses to handle the specific
        file formats and data structures of their respective hydrological models.

        Args:
            timestep_idx (int): The zero-based index of the timestep to load.
            **kwargs (Any): Additional keyword arguments that might be required by
                            the specific model output loader (e.g., layer number,
                            variable name).

        Returns:
            np.ndarray: A NumPy array containing the data for the specified timestep.
                        The shape and content of the array will depend on the model
                        and the type of data being loaded.
        """
        pass

    @abstractmethod
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """
        Load and return data for a specified range of timesteps.

        This method must be implemented by subclasses. It should efficiently
        load multiple timesteps, potentially optimizing for memory and speed.

        Args:
            start_idx (int, optional): The zero-based starting timestep index.
                                       Defaults to 0 (the first timestep).
            end_idx (Optional[int], optional): The zero-based ending timestep index (exclusive).
                                               If None, data should be loaded up to the last
                                               available timestep. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the specific model loader.

        Returns:
            np.ndarray: A NumPy array containing the data for the specified timestep range.
                        Typically, the first dimension of this array will represent time,
                        followed by spatial dimensions (e.g., [time, z, y, x]).
        """
        pass

    @abstractmethod
    def get_timestep_info(self) -> List[Tuple[Any, ...]]: # Changed from List[Tuple] to be more specific if possible
        """
        Retrieve information about each available timestep.

        This method should be implemented by subclasses to parse metadata or
        scan output files to determine characteristics of each timestep, such as
        simulation time, date, or specific event flags.

        Returns:
            List[Tuple[Any, ...]]: A list of tuples, where each tuple contains
                                   information pertaining to a single timestep.
                                   The structure of the tuple can vary depending on
                                   the model (e.g., (timestep_number, simulation_time_days)
                                   or (date_str, stress_period, time_step_in_period)).
        """
        pass

    def calculate_saturation(self, water_content: np.ndarray,
                           porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate volumetric saturation from water content and porosity.

        Saturation (S) is calculated as: S = water_content / porosity.
        The resulting saturation values are clipped to the range [0.0, 1.0].

        Args:
            water_content (np.ndarray): A NumPy array representing volumetric water content.
                                        This can be for a single timestep or multiple timesteps
                                        (e.g., shape [time, z, y, x] or [z, y, x]).
            porosity (Union[float, np.ndarray]): The porosity of the medium.
                                                 This can be a single scalar value (uniform porosity)
                                                 or a NumPy array with dimensions compatible for
                                                 broadcasting with `water_content`.
                                                 If `porosity` is an array, its dimensions should
                                                 match `water_content` or be expandable to match
                                                 (e.g., porosity [z,y,x] for water_content [t,z,y,x]).

        Returns:
            np.ndarray: A NumPy array of the same shape as `water_content`, containing
                        the calculated saturation values (clipped between 0 and 1).

        Raises:
            ValueError: If `porosity` is an array and its dimensions are not compatible
                        with `water_content` for element-wise division.
            TypeError: If `water_content` or `porosity` are not of the expected types.
        """
        if not isinstance(water_content, np.ndarray):
            raise TypeError("water_content must be a NumPy array.")
        if not isinstance(porosity, (int, float, np.ndarray)):
            raise TypeError("porosity must be a float, int, or NumPy array.")
        if np.any(porosity <= 0) and not np.all(porosity == 0): # Allow all zero for specific cases, but not mixed or negative.
             # Potential Issue: Division by zero if porosity is zero.
             # Depending on context, could return NaN, 0, or raise error.
             # Current behavior will result in `inf` or `nan` from division, then clipped.
             # Consider adding a small epsilon to porosity or specific handling for porosity == 0.
             print("Warning: Porosity contains zero or negative values. This may lead to division by zero or invalid saturation.")


        saturation: np.ndarray
        if isinstance(porosity, (int, float)):
            if porosity == 0:
                # Handle division by zero for scalar porosity explicitly
                # Return NaN where water content is non-zero, 0 where water content is zero (0/0 = NaN by default, but 0 could be argued)
                # Or, based on physical meaning, if porosity is 0, saturation is undefined or 0 if no water.
                # Let's assume if porosity is 0, saturation should be 0 if WC is 0, otherwise NaN (or error).
                # For simplicity, current numpy behavior (inf/nan) then clip is kept, but this is a critical point.
                if water_content.any() and porosity == 0.0 : # if any water content but no porosity
                    # This case is physically problematic. Saturation would be infinite.
                    # Clipping to 1.0 might hide this issue.
                    # Consider raising an error or returning NaNs more explicitly.
                    pass # Will result in inf/nan then clip.
            saturation = water_content / float(porosity) # Ensure float division
        else: # porosity is a NumPy array
            # Ensure porosity has dimensions compatible with water_content for broadcasting.
            if porosity.ndim != water_content.ndim:
                # If water_content has a time dimension and porosity does not,
                # try to align them by adding a new axis to porosity.
                if porosity.ndim == water_content.ndim - 1 and water_content.shape[0] > 0: # Assuming time is the first dimension
                    # Check if spatial dimensions match
                    if water_content.shape[1:] == porosity.shape:
                        # Expand porosity to match timesteps of water_content
                        # This assumes porosity is constant over time.
                        porosity_expanded = np.repeat(
                            porosity[np.newaxis, ...], # Add new axis at the beginning for time
                            water_content.shape[0],    # Repeat for number of timesteps
                            axis=0
                        )
                        saturation = water_content / porosity_expanded
                    else:
                        raise ValueError(
                            "Spatial dimensions of porosity array do not match water_content array. "
                            f"Water content shape: {water_content.shape}, Porosity shape: {porosity.shape}"
                        )
                else:
                    raise ValueError(
                        "Porosity array dimensions are not compatible with water_content array dimensions. "
                        f"Water content ndim: {water_content.ndim}, Porosity ndim: {porosity.ndim}"
                    )
            else: # Dimensions are the same, direct division should work if shapes match or broadcast.
                if water_content.shape != porosity.shape:
                    # Allow broadcasting if shapes are compatible (e.g. one is scalar-like for some dims)
                    try:
                        saturation = water_content / porosity
                    except ValueError as e:
                        raise ValueError(
                            "Water content and porosity shapes are not directly divisible or broadcastable. "
                            f"Water content shape: {water_content.shape}, Porosity shape: {porosity.shape}. Error: {e}"
                        )
                else: # Shapes are identical
                    saturation = water_content / porosity
        
        # Clip saturation values to be within the physically meaningful range [0, 1].
        # This handles cases where water_content might slightly exceed porosity due to numerical errors,
        # or where porosity is zero leading to `inf` values from division.
        saturation = np.clip(saturation, 0.0, 1.0)
        # Potential Issue: Clipping `inf` (from division by zero porosity) to 1.0 might mask
        # problematic areas. Consider if NaN or raising an error is more appropriate when porosity is zero
        # and water_content is non-zero. Current np.clip(np.inf, 0, 1) results in 1.

        return saturation