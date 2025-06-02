"""
Module for processing ParFlow model outputs.

This module provides classes to handle specific types of ParFlow outputs,
such as saturation and porosity, by reading ParFlow Binary Files (PFB).
It relies on the `parflow` Python package for PFB reading capabilities.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict, Any # Dict and Union not directly used here, but Any is.

from .base import HydroModelOutput # Assuming base.py is in the same directory or package


class ParflowOutput(HydroModelOutput):
    """
    Base class for processing ParFlow model outputs.

    This class handles common ParFlow output functionalities, such as
    identifying available timesteps and interfacing with the `parflow`
    Python package for reading PFB files. Specific data types (like
    saturation, porosity) should be handled by subclasses.
    """
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow output processor.

        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run (e.g., 'my_run' if
                            output files are like 'my_run.out.satur.00001.pfb').
                            This is used to construct filenames.

        Raises:
            ImportError: If the `parflow` Python package is not installed.
            FileNotFoundError: If `model_directory` does not exist or is not a directory.
        """
        super().__init__(model_directory) # Call base class constructor
        # Check if the provided model_directory is actually a directory.
        if not os.path.isdir(self.model_directory):
            raise FileNotFoundError(f"Model directory not found or is not a directory: {self.model_directory}")

        self.run_name = run_name # Store the base name for ParFlow output files.
        
        # --- ParFlow Package Availability Check ---
        try:
            # Attempt to import the ParFlow IO tools.
            # This makes `parflow` a "soft" dependency at the package level,
            # but it's a hard dependency for this specific class.
            import parflow.tools.io as pftools
            self.parflow_available = True
            self.read_pfb = pftools.read_pfb # Store the read_pfb function handle for later use.
        except ImportError:
            self.parflow_available = False
            # If parflow.tools.io cannot be imported, this class cannot function.
            raise ImportError("The 'parflow' Python package (specifically parflow.tools.io) is required "
                              "to process ParFlow outputs. Please install it (e.g., 'pip install parflow').")
        
        # --- Discover Available Timesteps ---
        # Scan the output directory to find all available timestep numbers based on common file patterns.
        # SUGGESTION: This is done at initialization. If output files are added to the directory
        # after this object is created, this list will be stale. Consider a refresh mechanism if needed.
        self.available_timesteps = self._get_available_timesteps() # Store the sorted list of timestep numbers.
        if not self.available_timesteps:
            # Print a warning if no timestep files are found, as this might indicate an issue
            # with the provided run_name, directory, or the simulation outputs themselves.
            print(f"Warning: No ParFlow output timesteps found for run '{self.run_name}' in '{self.model_directory}'. "
                  "Check run_name and file patterns (e.g., looking for files like '{self.run_name}.out.satur.*.pfb').")
    
    def _get_available_timesteps(self) -> List[int]:
        """
        Scans the model directory to find available ParFlow output timesteps.

        It looks for files matching common ParFlow output patterns (e.g., for
        saturation or pressure) and extracts the timestep numbers from their names.
        Timestep numbers are assumed to be integers.

        Returns:
            List[int]: A sorted list of unique integer timestep numbers found.
                       Returns an empty list if no matching files are found.
        """
        timesteps_set = set() # Use a set to store found timestep numbers, automatically handling duplicates.

        # --- Define File Patterns for Timestep Discovery ---
        # ParFlow output files commonly follow the pattern: <run_name>.out.<variable>.<timestep_number>.pfb
        # The timestep number is typically an integer, often zero-padded (e.g., 00001, 00002).
        # This method tries common variable names like 'satur' (saturation) and 'press' (pressure)
        # to discover the range of available timesteps.
        # SUGGESTION: Make these patterns configurable or extendable if users have different naming conventions.
        patterns_to_check = [
            f"{self.run_name}.out.satur.", # Primary pattern: saturation files
            f"{self.run_name}.out.press."  # Secondary pattern: pressure files (used if no saturation files found for discovery)
            # Add other common patterns if needed, e.g., f"{self.run_name}.out.evaptrans."
        ]

        found_files_for_primary_pattern = False
        for pattern_prefix in patterns_to_check:
            # If timesteps were already found using the primary pattern (e.g., saturation),
            # don't use secondary patterns for discovery to avoid mixing timestep types if they differ.
            if found_files_for_primary_pattern and pattern_prefix == patterns_to_check[1]:
                break

            # Iterate over all files in the model directory.
            for filename in os.listdir(self.model_directory):
                # Check if the filename starts with the current pattern and ends with '.pfb'.
                if filename.startswith(pattern_prefix) and filename.lower().endswith(".pfb"):
                    try:
                        # Attempt to extract the timestep number from the filename.
                        # Example: filename = "myrun.out.satur.00001.pfb", pattern_prefix = "myrun.out.satur."
                        # timestep_str_part will be "00001.pfb"
                        timestep_str_part = filename[len(pattern_prefix):]

                        # The timestep number is expected before the first '.' in timestep_str_part.
                        # Example: "00001.pfb" -> "00001"
                        # Example: "00001.clm.pfb" -> "00001" (if CLM outputs are named this way)
                        timestep_str = timestep_str_part.split('.')[0]

                        timestep = int(timestep_str) # Convert the extracted string to an integer.
                        timesteps_set.add(timestep) # Add to set.

                        # If this was a primary pattern, mark that we found files with it.
                        if pattern_prefix == patterns_to_check[0]:
                            found_files_for_primary_pattern = True
                    except ValueError:
                        # If int() conversion fails (e.g., filename is "my_run.out.satur.final.pfb").
                        # print(f"Warning: Could not parse integer timestep from filename: {filename}") # Optional debug
                        continue # Skip this file.
                    except IndexError:
                        # If split('.') results in an empty list (e.g., if filename was just "prefix.pfb").
                        # print(f"Warning: Could not parse timestep due to unexpected filename structure: {filename}") # Optional debug
                        continue # Skip this file.

        return sorted(list(timesteps_set)) # Convert set to list and sort for ordered timestep numbers.
    
    def get_pfb_dimensions(self, pfb_file_path: str) -> Tuple[int, int, int]:
        """
        Reads a PFB file and returns its data dimensions (nz, ny, nx).

        Args:
            pfb_file_path (str): The full path to the PFB file.

        Returns:
            Tuple[int, int, int]: The dimensions of the data in the PFB file,
                                  typically in (nz, ny, nx) order for ParFlow.

        Raises:
            FileNotFoundError: If the `pfb_file_path` does not exist.
            Exception: If `self.read_pfb` (from parflow.tools.io) fails to read the file or returns unexpected type.
        """
        # Check if the specified PFB file exists.
        if not os.path.exists(pfb_file_path):
            raise FileNotFoundError(f"PFB file not found: {pfb_file_path}")

        # Use the stored read_pfb function (parflow.tools.io.read_pfb) to load the PFB file.
        # This function is expected to return a NumPy array containing the data.
        data_array = self.read_pfb(pfb_file_path)

        # Validate that the loaded data is a NumPy array.
        if not isinstance(data_array, np.ndarray):
            # This would be unexpected if parflow.tools.io.read_pfb behaves as documented.
            raise TypeError(f"Expected NumPy array from ParFlow PFB reader, but got {type(data_array)} for file {pfb_file_path}")

        # --- Determine Dimensions ---
        # ParFlow PFB files typically store 3D data in (nz, ny, nx) order (layers, rows, columns).
        # This method attempts to handle cases where data might be 1D or 2D as well.
        if data_array.ndim == 3:
            return data_array.shape # Returns (nz, ny, nx) directly.
        elif data_array.ndim == 2:
            # If data is 2D, assume it's (ny, nx) and nz=1.
            # print(f"Warning: PFB file '{pfb_file_path}' data is 2-dimensional (shape: {data_array.shape}). Assuming nz=1.") # Optional debug
            return (1, data_array.shape[0], data_array.shape[1]) # (nz=1, ny, nx)
        elif data_array.ndim == 1:
            # If data is 1D, assume it's (nx) and nz=1, ny=1.
            # print(f"Warning: PFB file '{pfb_file_path}' data is 1-dimensional (shape: {data_array.shape}). Assuming nz=1, ny=1.") # Optional debug
            return (1, 1, data_array.shape[0]) # (nz=1, ny=1, nx)
        else:
            # If data has more than 3 dimensions or 0 dimensions (scalar), it's unsupported here.
            raise ValueError(f"PFB file '{pfb_file_path}' has unsupported data dimensionality: {data_array.ndim}. Expected 1D, 2D, or 3D.")

        # Redundant return, as all paths in if/elif/else should return. Kept from original structure.
        # return data_array.shape


class ParflowSaturation(ParflowOutput):
    """
    Processes saturation data from ParFlow simulations (.out.satur.*.pfb files).
    """
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow saturation processor.

        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run (used for file naming).
        """
        super().__init__(model_directory, run_name) # Initialize base ParflowOutput class
        # No saturation-specific initialization beyond what the base class does (like finding timesteps).
        # One could add a check here to ensure that the timesteps found by _get_available_timesteps
        # were indeed from saturation files if that's a strict requirement for this class.
        # e.g., check if f"{self.run_name}.out.satur." was the pattern that yielded timesteps.
    
    def load_timestep(self, timestep_idx: int, **kwargs: Any) -> np.ndarray:
        """
        Load saturation data for a specific, zero-based timestep index.

        Args:
            timestep_idx (int): The zero-based index of the timestep to load from the
                                list of available timesteps discovered during initialization.
            **kwargs (Any): Additional keyword arguments (not used by this method).

        Returns:
            np.ndarray: A 3D NumPy array of saturation values (nz, ny, nx).

        Raises:
            ValueError: If no timesteps are available or if `timestep_idx` is out of range.
        """
        # Check if any timesteps were discovered during initialization.
        if not self.available_timesteps:
            raise ValueError("No ParFlow timesteps were found during initialization. Cannot load saturation data.")
            
        # Validate that the requested timestep_idx is within the bounds of available timesteps.
        if not (0 <= timestep_idx < len(self.available_timesteps)):
            raise ValueError(f"Requested timestep_idx {timestep_idx} is out of range. "
                             f"Available indices are 0 to {len(self.available_timesteps)-1}.")
        
        # Get the actual ParFlow timestep number (from filename) corresponding to the requested index.
        actual_timestep_number = self.available_timesteps[timestep_idx]
        # Call the internal method to load data for this specific ParFlow timestep number.
        return self._load_saturation_for_timestep_num(actual_timestep_number)
    
    def _load_saturation_for_timestep_num(self, timestep_number: int) -> np.ndarray:
        """
        Internal helper to load saturation data for a given ParFlow timestep number.

        Args:
            timestep_number (int): The actual timestep number as found in the ParFlow
                                   output filename (e.g., 1 for *.00001.pfb).

        Returns:
            np.ndarray: 3D array of saturation values (nz, ny, nx).

        Raises:
            FileNotFoundError: If the specific saturation PFB file cannot be found.
            ValueError: If there's an error reading or processing the PFB file.
        """
        # --- Construct Filename and Check Existence ---
        # ParFlow often uses 5-digit zero-padding for timestep numbers in filenames.
        # This attempts to find the file with padding first, then without if that fails.
        # SUGGESTION: The padding length (e.g., 5) could be a class or method parameter if it varies.
        satur_filename_padded = f"{self.run_name}.out.satur.{timestep_number:05d}.pfb"
        satur_file_path_padded = os.path.join(self.model_directory, satur_filename_padded)

        satur_filename_unpadded = f"{self.run_name}.out.satur.{timestep_number}.pfb" # No padding
        satur_file_path_unpadded = os.path.join(self.model_directory, satur_filename_unpadded)

        chosen_file_path = ""
        if os.path.exists(satur_file_path_padded):
            chosen_file_path = satur_file_path_padded
        elif os.path.exists(satur_file_path_unpadded):
            chosen_file_path = satur_file_path_unpadded
        else:
            # If neither padded nor unpadded version is found, raise an error.
            raise FileNotFoundError(
                f"ParFlow saturation file for timestep {timestep_number} not found. "
                f"Attempted paths: '{satur_file_path_padded}' and '{satur_file_path_unpadded}'.")

        # --- Load and Process PFB Data ---
        try:
            # Read the PFB file using the method from parflow.tools.io.
            saturation_data = self.read_pfb(chosen_file_path)
            
            # Handle No-Data Values:
            # ParFlow often uses very large negative numbers (e.g., around -1.0E+38 or -2.0E+38, specific value can vary)
            # to represent inactive cells or no-data regions (e.g., cells outside the active domain mask).
            # These are replaced with np.nan for easier handling in Python/NumPy based analysis and plotting.
            # SUGGESTION: The exact no-data value can sometimes be specified in ParFlow runscripts or vary.
            # A more robust solution might involve getting this value from run metadata if possible,
            # or making the threshold (-1e38) a configurable parameter.
            no_data_threshold = -1e38 # Define a threshold for identifying no-data values.
            saturation_data[saturation_data < no_data_threshold] = np.nan # Replace no-data values with NaN.
            
            return saturation_data
        except Exception as e: # Catch potential errors from self.read_pfb or NumPy operations.
            raise ValueError(f"Error loading or processing ParFlow saturation data from '{chosen_file_path}': {str(e)}")
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """
        Load saturation data for a specified range of zero-based timestep indices.

        Args:
            start_idx (int, optional): Starting zero-based timestep index. Defaults to 0.
            end_idx (Optional[int], optional): Ending zero-based timestep index (exclusive).
                                               If None, loads up to the last available timestep.
                                               Defaults to None.
            **kwargs (Any): Additional keyword arguments (not used).

        Returns:
            np.ndarray: A 4D NumPy array of saturation values (num_timesteps, nz, ny, nx).
                        Returns an empty 4D array if the range is invalid or no data is found.

        Raises:
            ValueError: If no timesteps are available, or if the specified range results in no timesteps to load.
        """
        if not self.available_timesteps:
            raise ValueError("No ParFlow timesteps available to load from.")
        
        # --- Determine Actual Start and End Indices for Slicing ---
        # Ensure start_idx is valid.
        if not (0 <= start_idx < len(self.available_timesteps)):
             raise ValueError(f"start_idx {start_idx} is out of range. Available indices: 0 to {len(self.available_timesteps)-1}.")

        # Determine the effective end_idx for slicing self.available_timesteps.
        # If end_idx is None, load all timesteps from start_idx to the end.
        # If end_idx is provided, use it, but ensure it doesn't exceed available length.
        # Python slice `[start:end]` naturally handles `end` being larger than list length.
        effective_end_idx: Optional[int]
        if end_idx is None:
            effective_end_idx = len(self.available_timesteps)
        elif end_idx < 0: # Handle negative end_idx (counts from the end)
            effective_end_idx = len(self.available_timesteps) + end_idx
        else: # Positive or zero end_idx
            effective_end_idx = end_idx

        # Ensure effective_end_idx is not less than start_idx if positive.
        # For slicing, if effective_end_idx <= start_idx, it results in an empty list, which is handled.

        # Get the list of actual ParFlow timestep numbers to load based on the calculated indices.
        timesteps_to_load_numbers = self.available_timesteps[start_idx:effective_end_idx]

        if not timesteps_to_load_numbers:
            # If the calculated range results in no timesteps to load (e.g., end_idx <= start_idx).
            # Determine spatial shape from the first *available* timestep to return a correctly shaped empty array.
            # This avoids error if an empty range is requested but we still need to know output shape.
            # print(f"Warning: Requested time range (start_idx={start_idx}, end_idx={end_idx} -> effective_end_idx={effective_end_idx}) is empty.") # Optional debug
            if self.available_timesteps: # Check again, though outer check should catch no timesteps at all.
                 first_available_timestep_num = self.available_timesteps[0]
                 first_data_sample = self._load_saturation_for_timestep_num(first_available_timestep_num)
                 return np.empty((0, *first_data_sample.shape), dtype=first_data_sample.dtype) # 0 timesteps, but correct spatial shape and dtype
            else: # Should be caught by the very first check.
                 return np.empty((0,1,1,1), dtype=float) # Fallback empty array.

        # --- Load Data for the Selected Timestep Numbers ---
        # Load the first timestep in the selected range to determine spatial dimensions (nz, ny, nx).
        # This assumes all PFB files for this variable within the run have consistent dimensions.
        first_timestep_to_load_num = timesteps_to_load_numbers[0]
        first_timestep_data = self._load_saturation_for_timestep_num(first_timestep_to_load_num)
        nz, ny, nx = first_timestep_data.shape # Get spatial dimensions.
        
        # Initialize a 4D NumPy array to store all loaded saturation data.
        # Shape: (number_of_timesteps_in_range, nz, ny, nx).
        # Use the dtype from the first loaded data for consistency.
        all_saturation_data = np.zeros((len(timesteps_to_load_numbers), nz, ny, nx), dtype=first_timestep_data.dtype)
        
        all_saturation_data[0] = first_timestep_data # Store the already loaded first timestep's data.
        
        # Load the data for the remaining timesteps in the selected range.
        # Start from the second timestep in the list (index 1), as the first (index 0) is already loaded.
        for i, timestep_num in enumerate(timesteps_to_load_numbers[1:], start=1):
            all_saturation_data[i] = self._load_saturation_for_timestep_num(timestep_num)
        
        return all_saturation_data
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Provides information about available ParFlow timesteps.

        For ParFlow, the timestep number from the filename often directly corresponds
        to the simulation time (e.g., if output is every 1 hour, timestep 24 is 24 hours).
        This method returns a list of tuples: (timestep_number, simulation_time).
        Currently, simulation_time is simply cast from timestep_number.
        More accurate time mapping would require parsing ParFlow timing files if complex.

        Returns:
            List[Tuple[int, float]]: A list where each tuple is (timestep_number, time_value).
                                     Time_value is float representation of timestep_number.
        """
        # Assumes timestep number from filename can be directly used as a proxy for simulation time (e.g., hours).
        # For more accurate or complex timing (e.g., variable timestep lengths, specific dates),
        # ParFlow's run script, .tcl files, or specific timing output files (.timing) would need to be parsed.
        # This is beyond the scope of this method which focuses on PFB file data.
        if not self.available_timesteps:
            return [] # Return empty list if no timesteps were found.
        # Create a list of tuples, where each tuple is (timestep_number_from_file, float_representation_of_time).
        return [(ts_num, float(ts_num)) for ts_num in self.available_timesteps]


class ParflowPorosity(ParflowOutput):
    """
    Processes porosity data from ParFlow simulations.
    Porosity in ParFlow is typically static (time-invariant) and stored in a
    single PFB file (e.g., <run_name>.out.porosity.pfb or similar).
    """
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow porosity processor.

        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run.
        """
        # Initialize the base ParflowOutput class.
        # The base class's _get_available_timesteps will run, which typically looks for
        # time-varying files like saturation or pressure. For static porosity, this list
        # might not be directly relevant for loading porosity itself, but is part of the
        # HydroModelOutput interface. The get_timestep_info for this class might reflect that.
        super().__init__(model_directory, run_name)
        # No porosity-specific initialization needed beyond base class.
        # One could add a check here to see if a porosity file actually exists,
        # but load_porosity() will handle that.
    
    def load_porosity(self) -> np.ndarray:
        """
        Load the static porosity data from the ParFlow model.

        It searches for common ParFlow porosity filename patterns within the
        model directory.

        Returns:
            np.ndarray: A 3D NumPy array of porosity values (nz, ny, nx).

        Raises:
            FileNotFoundError: If no standard porosity PFB file can be found.
            ValueError: If there's an error reading or processing the PFB file.
        """
        # --- Define Potential Porosity Filenames ---
        # ParFlow porosity is often a static field, possibly an input or a direct output.
        # Filenames can vary based on user convention or ParFlow version.
        # This list includes common patterns.
        # SUGGESTION: Make these patterns configurable if more flexibility is needed.
        porosity_filename_candidates = [
            f"{self.run_name}.out.porosity.pfb",  # Standard output pattern
            f"{self.run_name}.porosity.pfb",      # Common for input or static fields
            f"{self.run_name}.pf.porosity.pfb",   # Another possible prefixing style
            f"{self.run_name}.indicator.pfb",     # Sometimes indicator files might represent porosity zones
            # Fallbacks for files that might not have the .pfb extension explicitly written by older tools/scripts
            f"{self.run_name}.out.porosity",
            f"{self.run_name}.porosity",
            f"{self.run_name}.pf.porosity"
        ]
        
        # --- Search for and Load the Porosity File ---
        found_file_path = None
        for filename_candidate in porosity_filename_candidates:
            potential_path = os.path.join(self.model_directory, filename_candidate)
            if os.path.exists(potential_path):
                found_file_path = potential_path
                break # Use the first candidate found.

        # If no porosity file was found after checking all candidates.
        if not found_file_path:
            raise FileNotFoundError(
                f"Could not find a ParFlow porosity file for run '{self.run_name}' in directory '{self.model_directory}'. "
                f"Checked candidates like '{porosity_filename_candidates[0]}', '{porosity_filename_candidates[1]}', etc.")

        # --- Load and Process PFB Data ---
        try:
            porosity_data = self.read_pfb(found_file_path) # Load the PFB file.

            # Handle ParFlow's no-data values (large negative numbers) by converting them to NaN.
            no_data_threshold = -1e38 # Same threshold as used for saturation.
            porosity_data[porosity_data < no_data_threshold] = np.nan

            # SUGGESTION: Porosity values should physically be between 0 and 1 (or slightly >0 for active cells).
            # Consider adding validation here to check if values fall outside this range (excluding NaNs).
            # For example: `if np.any((porosity_data < 0) | (porosity_data > 1.0)) and not np.all(np.isnan(porosity_data))`, print warning.
            # Clipping `np.clip(porosity_data, 0.0, 1.0)` could be done, but might mask issues if
            # no-data NaNs are inadvertently clipped to 0. Better to ensure NaNs are handled correctly first.
            # If only active cells are expected to have porosity > 0, this is fine.

            return porosity_data
        except Exception as e: # Catch errors from PFB reading or NumPy operations.
            raise ValueError(f"Error loading or processing ParFlow porosity data from '{found_file_path}': {str(e)}")
    
    def load_mask(self) -> np.ndarray: # Original name was load_porosity, but seems to load mask
        """
        Load the domain mask data from a ParFlow model.
        The mask file (.out.mask.pfb) indicates active (1) and inactive (0) cells.

        Returns:
            np.ndarray: A 3D NumPy array representing the domain mask (nz, ny, nx).
                        Values are typically 0 or 1.

        Raises:
            FileNotFoundError: If no standard mask PFB file can be found.
            ValueError: If there's an error reading or processing the PFB file.
        """
        # --- Define Potential Mask Filenames ---
        # ParFlow mask files define the active (1) and inactive (0) regions of the model domain.
        mask_filename_candidates = [
            f"{self.run_name}.out.mask.pfb",  # Common output pattern
            f"{self.run_name}.mask.pfb",      # Often used for input or static mask
            f"{self.run_name}.pf.mask.pfb",   # Alternative prefixing
            f"{self.run_name}.out.mask",      # Fallback for files without .pfb extension
            f"{self.run_name}.mask"
        ]
        
        # --- Search for and Load the Mask File ---
        found_file_path = None
        for filename_candidate in mask_filename_candidates:
            potential_path = os.path.join(self.model_directory, filename_candidate)
            if os.path.exists(potential_path):
                found_file_path = potential_path
                break # Use the first candidate found
        
        if not found_file_path:
            # If no mask file is found after checking all candidates.
            raise FileNotFoundError(
                f"Could not find a ParFlow mask file for run '{self.run_name}' in directory '{self.model_directory}'. "
                f"Checked candidates like '{mask_filename_candidates[0]}'.")
        
        # --- Load and Process PFB Data for Mask ---
        try:
            mask_data = self.read_pfb(found_file_path) # Load the PFB file.

            # Handle ParFlow's no-data values (large negative numbers) if they exist in a mask file,
            # though typically mask files should contain 0s and 1s.
            # Replacing with NaN allows identification of potentially problematic no-data values in a mask.
            # Alternatively, one might replace them with 0 (inactive) if that's more appropriate for a mask.
            no_data_threshold = -1e38
            mask_data[mask_data < no_data_threshold] = np.nan # Or potentially 0, e.g., mask_data[mask_data < 0] = 0

            # SUGGESTION: Validate that mask_data contains expected values (e.g., 0 and 1, or NaNs).
            # For instance, `unique_values = np.unique(mask_data[~np.isnan(mask_data)])`.
            # If values other than 0 or 1 are present, it might indicate an issue.

            return mask_data
        except Exception as e: # Catch errors from PFB reading or NumPy operations.
            raise ValueError(f"Error loading or processing ParFlow mask data from '{found_file_path}': {str(e)}")

    def load_timestep(self, timestep_idx: int, **kwargs: Any) -> np.ndarray:
        """
        Load porosity data. For ParFlow, porosity is typically time-invariant.
        This method returns the static porosity array, ignoring `timestep_idx`.
        
        Args:
            timestep_idx (int): Index of the timestep (ignored, as porosity is static).
            **kwargs (Any): Additional keyword arguments (not used).
            
        Returns:
            np.ndarray: A 3D NumPy array of porosity values (nz, ny, nx).
        """
        # Porosity is considered static for ParFlow models (defined once, not changing per timestep).
        # Therefore, this method ignores timestep_idx and always returns the single, static porosity field.
        return self.load_porosity()
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """
        Load porosity data for a conceptual range of timesteps.
        Since porosity is time-invariant, this method returns a 4D array where
        the static 3D porosity data is repeated along the time axis.

        The number of repetitions along the time axis (`nt`) is determined by
        the length of `self.available_timesteps` (discovered from saturation/pressure files)
        if `end_idx` is None, or by `min(end_idx - start_idx, len(available_timesteps))`.
        A minimum of 1 repetition is ensured if any timesteps are notionally available.
        
        Args:
            start_idx (int, optional): Starting timestep index (used to determine `nt`). Defaults to 0.
            end_idx (Optional[int], optional): Ending timestep index (exclusive, used for `nt`).
                                               Defaults to None (use all available timesteps).
            **kwargs (Any): Additional keyword arguments (not used).
            
        Returns:
            np.ndarray: A 4D NumPy array of porosity values (nt, nz, ny, nx).
                        All slices along the time dimension are identical.
        """
        porosity_3d = self.load_porosity() # (nz, ny, nx)

        porosity_3d = self.load_porosity() # Load the static 3D porosity data (nz, ny, nx).

        # --- Determine Number of Timesteps for Replication ---
        # The number of times the static porosity array should be replicated along the time axis.
        # This is determined by the range [start_idx, end_idx) applied to `self.available_timesteps`,
        # which are discovered from other time-varying outputs (like saturation).
        # If no such time-varying outputs were found, it defaults to 1 repetition.

        num_replications: int
        if not self.available_timesteps:
            # If no time-dependent files were found by the base class to define a time axis length.
            num_replications = 1 # Default to a single "timestep" for static data.
        else:
            # Calculate effective start and end indices for slicing available_timesteps.
            effective_start_idx = max(0, start_idx) # Ensure start_idx is not negative.

            effective_end_idx: int
            if end_idx is None:
                effective_end_idx = len(self.available_timesteps)
            elif end_idx < 0: # Negative indexing from the end.
                effective_end_idx = len(self.available_timesteps) + end_idx
            else: # Positive or zero.
                effective_end_idx = end_idx

            # Ensure effective_end_idx is not beyond the list length or less than start.
            effective_end_idx = min(effective_end_idx, len(self.available_timesteps))
            effective_end_idx = max(effective_start_idx, effective_end_idx) # Handles cases where end < start.

            num_replications = effective_end_idx - effective_start_idx

            # If the range is empty but timesteps are available (e.g. start_idx = end_idx, or out of bounds start)
            # default to 1 replication, assuming user wants at least one instance of the static data.
            if num_replications == 0 and self.available_timesteps:
                 # print("Warning: Time range resulted in zero timesteps for static data replication. Defaulting to 1.") # Optional debug
                 num_replications = 1
            elif num_replications == 0 and not self.available_timesteps: # Should be covered by initial check, but for safety.
                 return np.empty((0, *porosity_3d.shape), dtype=porosity_3d.dtype) # Return empty, correctly shaped array.


        # --- Create 4D Array ---
        # Add a new axis for time at the beginning of the 3D porosity array.
        porosity_4d_single_timestep = porosity_3d[np.newaxis, ...] # Shape: (1, nz, ny, nx)
        
        # Replicate (tile) the porosity data along this new time axis.
        # Results in a 4D array where each slice along the time dimension is identical.
        replicated_porosity_4d = np.tile(porosity_4d_single_timestep, (num_replications, 1, 1, 1))
        
        return replicated_porosity_4d
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Returns timestep information, typically based on other ParFlow outputs
        (like saturation) as porosity itself is static.
        
        Returns:
            List[Tuple[int, float]]: A list of (timestep_number, time_value) tuples,
                                     derived from `self.available_timesteps`.
        """
        # Porosity is static. However, to maintain a consistent interface with time-varying
        # data types, this method returns timestep information based on the `self.available_timesteps`
        # list, which is populated by the base class (ParflowOutput) by scanning for
        # time-stamped files (like saturation or pressure).
        if not self.available_timesteps:
            # If the base class found no time-stamped files (e.g., directory only contains static files),
            # then provide a single, placeholder entry for the static porosity data.
            # This indicates one "state" or "timestep" (index 0, time 0.0).
            return [(0, 0.0)] # (timestep_number_identifier, conceptual_time_value)

        # If available_timesteps were found (e.g., from .satur files), use them to provide
        # a time context for the static porosity, implying it's valid for all those times.
        # Each tuple is (timestep_number_from_file, float_representation_of_time).
        return [(ts_num, float(ts_num)) for ts_num in self.available_timesteps]