"""
ParFlow Model Output Processing.

This module provides classes for reading and processing outputs from ParFlow,
a parallel, integrated watershed model. It focuses on extracting common
hydrological variables such as saturation and porosity from ParFlow's PFB
(ParFlow Binary File) format.

Classes:
    ParflowOutput: Base class for handling common ParFlow output functionalities.
    ParflowSaturation: For loading and processing saturation data.
    ParflowPorosity: For loading and processing porosity data (and potentially mask data).

Requires the `parflow` Python package with its PFB reading tools.
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
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run (e.g., used as a prefix
                            in output filenames like `run_name.out.satur.00000.pfb`).
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
        Get a sorted list of available numerical timesteps from ParFlow output filenames.
        
        This method scans the `model_directory` for files matching common ParFlow
        output patterns (e.g., `run_name.out.satur.NNNNN.pfb` or `run_name.out.press.NNNNN.pfb`)
        and extracts the timestep numbers (NNNNN). It prioritizes saturation files
        for determining timesteps; if none are found, it attempts to use pressure files.
        
        Returns:
            List[int]: A sorted list of unique integer timestep indices found in the directory.
        """
        timesteps = set() # Use a set to store unique timesteps
        
        # Define common ParFlow output filename patterns.
        # Example: {self.run_name}.out.satur.00001.pfb
        #          {self.run_name}.out.press.00001.pfb
        # The timestep is typically the integer part after "satur." or "press.".
        satur_prefix = f"{self.run_name}.out.satur."
        press_prefix = f"{self.run_name}.out.press."
        
        # Scan files in the model directory
        for filename in os.listdir(self.model_directory):
            try:
                if filename.startswith(satur_prefix) and filename.endswith(".pfb"):
                    # Extract timestep number: run_name.out.satur. (timestep) .pfb
                    timestep_str = filename[len(satur_prefix):-len(".pfb")]
                    timesteps.add(int(timestep_str))
                # If no saturation timesteps found yet, try pressure files
                elif filename.startswith(press_prefix) and filename.endswith(".pfb") and not timesteps:
                    timestep_str = filename[len(press_prefix):-len(".pfb")]
                    timesteps.add(int(timestep_str))
            except (ValueError, IndexError): # Handle cases where parsing fails
                continue
        
        return sorted(timesteps)
    
    def get_pfb_dimensions(self, pfb_file: str) -> Tuple[int, int, int]:
        """
        Get the dimensions (nz, ny, nx) of a ParFlow Binary File (PFB).
        
        Args:
            pfb_file (str): Path to the PFB file.
            
        Returns:
            Tuple[int, int, int]: A tuple containing the dimensions of the data
                                  in the PFB file, typically in the order (nz, ny, nx),
                                  representing layers, rows, and columns respectively.
        """
        data = self.read_pfb(pfb_file) # parflow.tools.io.read_pfb returns a NumPy array
        return data.shape


class ParflowSaturation(ParflowOutput):
    """Class for processing saturation data from ParFlow simulations."""
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow saturation processor.
        
        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run, used to identify output files.
        """
        super().__init__(model_directory, run_name)
    
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load saturation data for a specific timestep, identified by its index
        in the list of available timesteps.
        
        Args:
            timestep_idx (int): Index of the timestep to load from the sorted list
                                of available timesteps (obtained via `_get_available_timesteps`).
            **kwargs: Additional keyword arguments (currently unused by this method).
            
        Returns:
            np.ndarray: A 3D NumPy array of saturation values with shape (nz, ny, nx).

        Raises:
            ValueError: If no timesteps are available or if `timestep_idx` is out of range.
        """
        if not self.available_timesteps:
            raise ValueError("No ParFlow output timesteps found in the specified directory.")
            
        if not (0 <= timestep_idx < len(self.available_timesteps)):
            raise ValueError(
                f"Timestep index {timestep_idx} is out of range. "
                f"Available indices: 0 to {len(self.available_timesteps)-1}."
            )
        
        # Get the actual timestep number (e.g., 00005) from the index
        timestep = self.available_timesteps[timestep_idx]
        return self._load_saturation(timestep)
    
    def _load_saturation(self, timestep: int) -> np.ndarray:
        """
        Load saturation data for a specific ParFlow timestep number.
        
        Handles potential variations in timestep formatting in filenames (e.g., with or
        without zero-padding). Replaces ParFlow's common no-data value (-1e38 or smaller)
        with NaN.
        
        Args:
            timestep (int): The actual timestep number (e.g., 5 for the 5th output).
            
        Returns:
            np.ndarray: A 3D NumPy array of saturation values (nz, ny, nx).

        Raises:
            ValueError: If the saturation file for the given timestep cannot be found
                        or if there's an error during reading.
        """
        # Attempt to construct the saturation file path with 5-digit zero-padding for timestep.
        # Example: run_name.out.satur.00005.pfb
        satur_file_padded = os.path.join(
            self.model_directory, 
            f"{self.run_name}.out.satur.{timestep:05d}.pfb"
        )
        
        # Fallback: if padded version doesn't exist, try without padding.
        # Example: run_name.out.satur.5.pfb
        satur_file_unpadded = os.path.join(
            self.model_directory, 
            f"{self.run_name}.out.satur.{timestep}.pfb"
        )
        
        if os.path.exists(satur_file_padded):
            satur_file_to_load = satur_file_padded
        elif os.path.exists(satur_file_unpadded):
            satur_file_to_load = satur_file_unpadded
        else:
            raise ValueError(
                f"Saturation file for timestep {timestep} not found. "
                f"Checked: '{satur_file_padded}' and '{satur_file_unpadded}'."
            )
        
        # Read the saturation data using the ParFlow library tools
        try:
            saturation_data = self.read_pfb(satur_file_to_load)
            
            # ParFlow often uses very small negative numbers (e.g., around -1.00000000e+38 or -2e+38)
            # to represent no-data or inactive cells (especially for values like saturation or pressure head).
            # Replace these with NaN for more standard handling in NumPy/plotting.
            saturation_data[saturation_data < -1e30] = np.nan # A common threshold for ParFlow no-data
            
            return saturation_data
        except Exception as e:
            raise ValueError(f"Error loading ParFlow saturation data from '{satur_file_to_load}': {str(e)}")
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load saturation data for a range of timesteps.
        
        Args:
            start_idx (int, optional): Starting index of the timesteps to load from the
                                     list of available timesteps. Defaults to 0.
            end_idx (Optional[int], optional): Ending index (exclusive) of the timesteps to load.
                                               If None, loads until the end of available timesteps.
                                               Defaults to None.
            **kwargs: Additional keyword arguments (currently unused).
            
        Returns:
            np.ndarray: A 4D NumPy array of saturation values with shape (nt, nz, ny, nx),
                        where `nt` is the number of loaded timesteps.

        Raises:
            ValueError: If no timesteps are available or if the specified range is invalid.
        """
        if not self.available_timesteps:
            raise ValueError("No ParFlow output timesteps found in the specified directory.")
        
        # Validate and adjust start and end indices
        actual_start_idx = max(0, start_idx)
        actual_end_idx = len(self.available_timesteps) if end_idx is None else min(end_idx, len(self.available_timesteps))
        
        if actual_start_idx >= actual_end_idx:
            raise ValueError(
                f"Invalid timestep range: start_idx ({start_idx} -> {actual_start_idx}) "
                f"must be less than end_idx ({end_idx} -> {actual_end_idx})."
            )
        
        # Get the list of actual timestep numbers to load
        timesteps_to_load_numbers = self.available_timesteps[actual_start_idx:actual_end_idx]
        
        if not timesteps_to_load_numbers: # Should be caught by previous check, but as a safeguard
            raise ValueError(f"No valid timesteps in the processed range [{actual_start_idx}, {actual_end_idx}).")
        
        # Load the first timestep to determine the shape (nz, ny, nx) for pre-allocation
        first_timestep_data = self._load_saturation(timesteps_to_load_numbers[0])
        nz, ny, nx = first_timestep_data.shape
        num_timesteps_to_load = len(timesteps_to_load_numbers)
        
        # Initialize a 4D array to store all saturation data for the time range
        all_saturation_data = np.empty((num_timesteps_to_load, nz, ny, nx), dtype=first_timestep_data.dtype)
        
        # Store the already loaded first timestep's data
        all_saturation_data[0] = first_timestep_data
        
        # Load the remaining timesteps
        for i in range(1, num_timesteps_to_load):
            timestep_number = timesteps_to_load_numbers[i]
            all_saturation_data[i] = self._load_saturation(timestep_number)
        
        return saturation_data
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Get information about each available ParFlow output timestep.
        
        For ParFlow, the output timestep number (e.g., from `filename....00005.pfb`)
        often directly corresponds to the simulation time in some consistent unit
        (e.g., hours or days, depending on simulation setup). This method assumes
        the timestep number itself can be used as a proxy for simulation time.
        
        Returns:
            List[Tuple[int, float]]: A list of tuples, where each tuple is (timestep_number, simulation_time).
                                     Currently, `simulation_time` is simply `float(timestep_number)`.
        """
        # Assumes the ParFlow timestep number can be directly interpreted as a time value.
        # This might need adjustment if ParFlow run uses a separate time output or specific time units.
        return [(timestep_number, float(timestep_number)) for timestep_number in self.available_timesteps]


class ParflowPorosity(ParflowOutput):
    """Class for processing porosity data from ParFlow simulations."""
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow porosity processor.
        
        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run, used to identify output files.
        """
        super().__init__(model_directory, run_name)
    
    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from a ParFlow model run.
        
        This method searches for common ParFlow porosity output filenames (e.g.,
        `run_name.out.porosity.pfb`, `run_name.pf.porosity.pfb`). It reads the first
        one found and replaces ParFlow's no-data values with NaN.
        
        Returns:
            np.ndarray: A 3D NumPy array of porosity values (nz, ny, nx).

        Raises:
            ValueError: If no suitable porosity file is found in the model directory,
                        or if there is an error reading the file.
        """
        # Define a list of common ParFlow porosity filename patterns to search for.
        # ParFlow filenames can vary based on version and user configuration.
        possible_porosity_filenames = [
            f"{self.run_name}.out.porosity.pfb", # Common modern ParFlow output
            f"{self.run_name}.out.porosity",     # Older or alternative extension
            f"{self.run_name}.pf.porosity.pfb",  # Alternative prefixing
            f"{self.run_name}.pf.porosity"
        ]
        
        porosity_filepath_to_load = None
        for filename_pattern in possible_porosity_filenames:
            potential_path = os.path.join(self.model_directory, filename_pattern)
            if os.path.exists(potential_path):
                porosity_filepath_to_load = potential_path
                break # Found a porosity file
        
        if porosity_filepath_to_load is None:
            raise ValueError(
                f"Could not find a porosity file for run '{self.run_name}' in directory "
                f"'{self.model_directory}'. Checked patterns: {possible_porosity_filenames}"
            )
            
        try:
            porosity_data = self.read_pfb(porosity_filepath_to_load)
            # Replace ParFlow's no-data values with NaN.
            # ParFlow often uses very small negative numbers for no-data.
            porosity_data[porosity_data < -1e30] = np.nan 
            return porosity_data
        except Exception as e:
            raise ValueError(f"Error loading ParFlow porosity data from '{porosity_filepath_to_load}': {str(e)}")
    
    def load_mask(self) -> np.ndarray:
        """
        Load domain mask data from a ParFlow model run.
        
        ParFlow can output a domain mask file (e.g., `run_name.out.mask.pfb`),
        which typically indicates active (1) and inactive (0) cells in the model domain.
        This method searches for common mask filenames and loads the data.
        No-data values (e.g., very small numbers if present) are replaced with NaN.
        
        Returns:
            np.ndarray: A 3D NumPy array representing the domain mask (nz, ny, nx).

        Raises:
            ValueError: If no suitable mask file is found or if there's an error reading it.
        
        Note:
            The interpretation of mask values (0 vs 1, or other values) depends on the
            specific ParFlow simulation setup. This method provides the raw mask data.
        """
        # Define a list of common ParFlow mask filename patterns.
        possible_mask_filenames = [
            f"{self.run_name}.out.mask.pfb",
            f"{self.run_name}.out.mask",
            f"{self.run_name}.pf.mask.pfb",
            f"{self.run_name}.pf.mask"
        ]
        
        mask_filepath_to_load = None
        for filename_pattern in possible_mask_filenames:
            potential_path = os.path.join(self.model_directory, filename_pattern)
            if os.path.exists(potential_path):
                mask_filepath_to_load = potential_path
                break # Found a mask file
                
        if mask_filepath_to_load is None:
            raise ValueError(
                f"Could not find a mask file for run '{self.run_name}' in directory "
                f"'{self.model_directory}'. Checked patterns: {possible_mask_filenames}"
            )
            
        try:
            mask_data = self.read_pfb(mask_filepath_to_load)
            # Replace potential ParFlow no-data values with NaN, though masks are often 0s and 1s.
            mask_data[mask_data < -1e30] = np.nan 
            return mask_data
        except Exception as e:
            raise ValueError(f"Error loading ParFlow mask data from '{mask_filepath_to_load}': {str(e)}")
        
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity data. As porosity is typically static (time-invariant) for ParFlow models,
        this method effectively returns the single porosity field, ignoring the `timestep_idx`.
        
        Args:
            timestep_idx (int): Index of the timestep (unused, as porosity is static).
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            np.ndarray: A 3D NumPy array of porosity values (nz, ny, nx).
        """
        return self.load_porosity()
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load porosity data for a conceptual range of timesteps.
        
        Since porosity is typically static for ParFlow models, this method returns
        a 4D array where the porosity field (3D) is repeated along the time dimension (`nt` times).
        The number of repetitions (`nt`) is determined by the range `start_idx` to `end_idx`
        and the number of available timesteps (from saturation/pressure files), or defaults to 1
        if no dynamic timesteps are found.
        
        Args:
            start_idx (int, optional): Starting index for determining the number of time repetitions.
                                     Defaults to 0.
            end_idx (Optional[int], optional): Ending index (exclusive) for determining time repetitions.
                                               If None, uses the total number of available timesteps.
                                               Defaults to None.
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            np.ndarray: A 4D array of porosity values, shape (nt, nz, ny, nx), where
                        each slice along the first (time) axis is identical.
        """
        porosity_3d = self.load_porosity() # Shape (nz, ny, nx)
        
        # Determine the number of timesteps (nt) to repeat the porosity array.
        # This is based on the provided start/end indices and available dynamic data timesteps.
        if not self.available_timesteps:
            num_repetitions = 1 # If no dynamic timesteps, assume a single static representation
        elif end_idx is None:
            num_repetitions = len(self.available_timesteps) - start_idx
        else:
            num_repetitions = end_idx - start_idx
        
        num_repetitions = max(1, num_repetitions) # Ensure at least one repetition
        
        # Tile the 3D porosity array `num_repetitions` times along a new first axis (time).
        # np.newaxis adds a new dimension: (1, nz, ny, nx)
        # np.tile repeats it along the first axis.
        porosity_4d = np.tile(porosity_3d[np.newaxis, :, :, :], (num_repetitions, 1, 1, 1))
        
        return porosity_4d
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Get information about each available ParFlow output timestep.
        
        As porosity is static, this method returns the same timestep information
        as derived from dynamic outputs (like saturation), effectively providing a
        list of timesteps for which this static porosity field is relevant.
        
        Returns:
            List[Tuple[int, float]]: A list of tuples, where each tuple is (timestep_number, simulation_time).
                                     Currently, `simulation_time` is `float(timestep_number)`.
        """
        # Porosity is static but applies to all dynamic timesteps.
        return [(timestep_number, float(timestep_number)) for timestep_number in self.available_timesteps]