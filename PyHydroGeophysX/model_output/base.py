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
            # Type check for the model_directory path.
            # SUGGESTION: Consider checking if the directory exists using os.path.exists().
            # However, this might be premature if the directory is created later by the user
            # or if the path is to a remote resource not yet accessible.
            raise TypeError("model_directory must be a string path.")
        self.model_directory = model_directory # Store the path to the model output directory.

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
        # This is an abstract method, so it has no implementation in the base class.
        # Subclasses must provide their own implementation.
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
        # Abstract method: Subclasses need to implement this to load a series of timesteps.
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
        # Abstract method: Subclasses must implement this to provide metadata about timesteps.
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
        # --- Input Type Validation ---
        if not isinstance(water_content, np.ndarray):
            raise TypeError("water_content must be a NumPy array.")
        if not isinstance(porosity, (int, float, np.ndarray)): # Porosity can be scalar or array
            raise TypeError("porosity must be a float, int, or NumPy array.")

        # --- Porosity Value Validation ---
        # Check for non-positive porosity values, which can lead to division by zero or meaningless saturation.
        if isinstance(porosity, np.ndarray):
            # If porosity is an array, check if any values are problematic.
            if np.any(porosity <= 0):
                # SUGGESTION: Distinguish between porosity == 0 and porosity < 0.
                # Porosity < 0 is physically impossible. Porosity == 0 might have specific meaning (e.g., impermeable).
                print("Warning: Porosity array contains zero or negative values. This may lead to division by zero or invalid saturation.")
        elif porosity <= 0: # Scalar porosity
             print("Warning: Scalar porosity is zero or negative. This may lead to division by zero or invalid saturation.")
             # If scalar porosity is 0 and any water content exists, division will yield inf.
             # If water content is also 0, 0/0 yields nan.

        saturation: np.ndarray # Initialize type hint for the output

        # --- Saturation Calculation ---
        if isinstance(porosity, (int, float)): # Scalar porosity
            porosity_float = float(porosity) # Ensure float for division
            if porosity_float == 0.0:
                # Handle scalar porosity being zero.
                # If water_content is also 0, saturation is undefined (NaN) or could be considered 0.
                # If water_content is > 0, saturation is infinite.
                # Numpy's default behavior for x/0 is inf (for x!=0) or nan (for 0/0).
                # The subsequent clipping will handle these (inf -> 1.0, nan -> 0.0 or remains nan if clip is careful).
                # Explicitly printing a warning here is good. The earlier warning also covers this.
                pass # Proceed with division, np.clip will handle inf/nan.
            saturation = water_content / porosity_float
        else: # Porosity is a NumPy array
            # Check if porosity dimensions are compatible with water_content dimensions for broadcasting.
            if porosity.ndim != water_content.ndim:
                # Attempt to align dimensions if water_content has a time dimension and porosity does not.
                # This assumes porosity is spatially variable but constant over time.
                if (water_content.ndim > 0 and porosity.ndim == water_content.ndim - 1 and
                        water_content.shape[1:] == porosity.shape): # Time is often the first dimension
                    # Expand porosity array by adding a new axis for time and repeating values.
                    # e.g., water_content [t, z, y, x], porosity [z, y, x] -> porosity_expanded [t, z, y, x]
                    porosity_expanded = np.repeat(
                        porosity[np.newaxis, ...], # Add new axis at the beginning (for time)
                        water_content.shape[0],    # Repeat for the number of timesteps in water_content
                        axis=0
                    )
                    # Perform element-wise division with the expanded porosity array.
                    # Handle division by zero within the array: result will be inf/nan where porosity_expanded is 0.
                    saturation = water_content / porosity_expanded
                else:
                    # If dimensions don't match in a way that can be automatically handled by expansion.
                    raise ValueError(
                        "Porosity array dimensions are not compatible with water_content array dimensions for typical time-series expansion. "
                        f"Water content ndim: {water_content.ndim} (shape {water_content.shape}), "
                        f"Porosity ndim: {porosity.ndim} (shape {porosity.shape})"
                    )
            else: # Dimensions are the same (porosity.ndim == water_content.ndim)
                # If shapes are not identical, NumPy will attempt broadcasting.
                # If shapes are identical, direct element-wise division occurs.
                try:
                    # This handles both identical shapes and compatible shapes for broadcasting.
                    # Division by zero within the array will result in inf/nan.
                    saturation = water_content / porosity
                except ValueError as e:
                    # Broadcasting failed.
                    raise ValueError(
                        "Water content and porosity shapes are not directly divisible or broadcastable. "
                        f"Water content shape: {water_content.shape}, Porosity shape: {porosity.shape}. Original error: {e}"
                    )
        
        # --- Clipping Saturation Values ---
        # Clip saturation to the physically meaningful range [0.0, 1.0].
        # This handles:
        #   1. water_content > porosity (due to numerical errors, S > 1) -> S = 1.0
        #   2. water_content / 0 (where porosity is 0, water_content > 0, S = inf) -> S = 1.0
        #   3. 0 / 0 (where porosity is 0, water_content = 0, S = nan) -> S = 0.0 (np.clip(np.nan,0,1) is 0)
        # SUGGESTION: The case of porosity == 0 and water_content > 0 resulting in S=1.0 after clipping
        # might mask critical modeling issues (e.g., water appearing in an impermeable zone).
        # Consider if returning NaN or raising a more specific error would be better for such physical inconsistencies.
        # For example, `saturation[porosity == 0] = np.nan` before clipping, or a separate check.
        saturation = np.clip(saturation, 0.0, 1.0)

        return saturation