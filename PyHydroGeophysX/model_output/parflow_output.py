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

        Raises:
            ImportError: If the `parflow` Python package is not installed.
            FileNotFoundError: If `model_directory` does not exist.
        """
        super().__init__(model_directory)
        if not os.path.isdir(self.model_directory):
            raise FileNotFoundError(f"Model directory not found: {self.model_directory}")
            
        self.run_name = run_name
        
        try:
            # Dynamically import parflow and its PFB reading tool.
            # This makes `parflow` an optional dependency if ParflowOutput classes are not used.
            import parflow.tools.io as pftools # Changed from `import parflow`
            self.parflow_available = True
            self.read_pfb = pftools.read_pfb # Assign the function directly
        except ImportError:
            self.parflow_available = False
            # Critical dependency for this class to function.
            raise ImportError("The 'parflow' Python package is required to process ParFlow outputs. "
                              "Please install it (e.g., 'pip install parflow').")
        
        # Discover available timesteps upon initialization.
        # Potential Issue: If new output files are written after initialization,
        # this list won't update unless refreshed.
        self.available_timesteps = self._get_available_timesteps()
        if not self.available_timesteps:
            print(f"Warning: No ParFlow output timesteps found for run '{self.run_name}' in '{self.model_directory}'. "
                  "Check run_name and file patterns (e.g., *.out.satur.*.pfb).")
    
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
        timesteps_set = set() # Use a set to automatically handle duplicates if patterns overlap
        
        # Common ParFlow output file patterns.
        # Files are typically named <run_name>.out.<variable_name>.<timestep_number>.pfb
        # The timestep number is often zero-padded (e.g., 00001, 00002, ...).
        # This pattern tries to capture that.
        # Example: my_run.out.satur.00001.pfb -> timestep 1
        # Example: my_run.out.press.00010.pfb -> timestep 10
        
        # Define patterns to search for. Prioritize saturation files, then pressure if no saturation found.
        # This is heuristic; ParFlow output naming can be configured.
        patterns_to_check = [
            f"{self.run_name}.out.satur.", # Saturation files
            f"{self.run_name}.out.press."  # Pressure files (as fallback for timestep discovery)
        ]
        
        found_primary_pattern = False
        for pattern_prefix in patterns_to_check:
            if found_primary_pattern and pattern_prefix == patterns_to_check[1]: # If already found satur, skip press for discovery
                break

            for filename in os.listdir(self.model_directory):
                if filename.startswith(pattern_prefix) and filename.lower().endswith(".pfb"):
                    try:
                        # Extract the part after the prefix: e.g., "00001.pfb"
                        timestep_str_part = filename[len(pattern_prefix):]
                        # Remove ".pfb" and any other potential suffixes if complex naming (e.g. .clm.).
                        # Simplest is to split by '.' and take the first part.
                        timestep_str = timestep_str_part.split('.')[0]
                        timestep = int(timestep_str) # Convert to integer
                        timesteps_set.add(timestep)
                        if pattern_prefix == patterns_to_check[0]: # Found saturation files
                            found_primary_pattern = True
                    except ValueError: # If int conversion fails (e.g., filename "my_run.out.satur.final.pfb")
                        # print(f"Warning: Could not parse timestep from filename: {filename}")
                        continue # Skip this file
                    except IndexError: # If split by '.' results in empty list (shouldn't happen with .pfb)
                        # print(f"Warning: Could not parse timestep due to unexpected filename structure: {filename}")
                        continue
        
        return sorted(list(timesteps_set)) # Return sorted list of unique timesteps
    
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
            Exception: If `self.read_pfb` (from parflow.tools.io) fails to read the file.
        """
        if not os.path.exists(pfb_file_path):
            raise FileNotFoundError(f"PFB file not found: {pfb_file_path}")
        
        # self.read_pfb is `parflow.tools.io.read_pfb`
        # This function typically returns a NumPy array.
        data_array = self.read_pfb(pfb_file_path)
        if not isinstance(data_array, np.ndarray):
            # Should not happen if read_pfb works as expected.
            raise TypeError(f"Expected NumPy array from read_pfb, got {type(data_array)} for file {pfb_file_path}")
        
        # ParFlow PFB files usually store data in (nz, ny, nx) order.
        if data_array.ndim != 3:
            # Potential Issue: If data is not 3D (e.g., 2D slice, or 1D output).
            # This method assumes 3D output. Adapt if other dimensionalities are common.
            print(f"Warning: PFB file '{pfb_file_path}' data is not 3-dimensional (shape: {data_array.shape}). Assuming (nz=1, ny=1, nx=shape[0]) or similar if 1D.")
            # Handle common cases for non-3D data to still return a 3-tuple.
            if data_array.ndim == 1: return (1, 1, data_array.shape[0])
            if data_array.ndim == 2: return (1, data_array.shape[0], data_array.shape[1])
            # If > 3D, this is unexpected for standard ParFlow scalar outputs.
            raise ValueError(f"PFB file '{pfb_file_path}' has unsupported data dimensionality: {data_array.ndim}")
            
        return data_array.shape # Returns (nz, ny, nx)


class ParflowSaturation(ParflowOutput):
    """
    Processes saturation data from ParFlow simulations (.out.satur.*.pfb files).
    """
    
    def __init__(self, model_directory: str, run_name: str):
        """
        Initialize ParFlow saturation processor.

        Args:
            model_directory (str): Path to the ParFlow simulation output directory.
            run_name (str): The base name of the ParFlow run.
        """
        super().__init__(model_directory, run_name)
        # Additional check: Ensure saturation files were indeed the source of timesteps if specific.
        # For now, _get_available_timesteps is generic.
    
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
        if not self.available_timesteps:
            raise ValueError("No ParFlow timesteps were found. Cannot load saturation data.")
            
        if not (0 <= timestep_idx < len(self.available_timesteps)):
            raise ValueError(f"Timestep index {timestep_idx} is out of range. "
                             f"Available indices: 0 to {len(self.available_timesteps)-1}.")
        
        actual_timestep_number = self.available_timesteps[timestep_idx]
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
        # Construct the expected saturation PFB filename.
        # ParFlow typically zero-pads timestep numbers to 5 digits.
        # Potential Issue: Padding might vary based on ParFlow version or settings.
        # The original code tries 5-digit padding first, then without if not found.
        # This seems like a reasonable fallback.
        satur_filename_padded = f"{self.run_name}.out.satur.{timestep_number:05d}.pfb"
        satur_file_path_padded = os.path.join(self.model_directory, satur_filename_padded)
        
        satur_filename_unpadded = f"{self.run_name}.out.satur.{timestep_number}.pfb"
        satur_file_path_unpadded = os.path.join(self.model_directory, satur_filename_unpadded)

        chosen_path = ""
        if os.path.exists(satur_file_path_padded):
            chosen_path = satur_file_path_padded
        elif os.path.exists(satur_file_path_unpadded):
            chosen_path = satur_file_path_unpadded
        else:
            raise FileNotFoundError(
                f"Saturation file for timestep {timestep_number} not found. "
                f"Checked: '{satur_file_path_padded}' and '{satur_file_path_unpadded}'.")
        
        try:
            saturation_data = self.read_pfb(chosen_path) # Use the instance's PFB reader
            
            # ParFlow uses large negative numbers (e.g., -1.0E+39, -2.0E+39) to denote no-data or inactive cells.
            # Replace these with NaN for more standard handling in NumPy/plotting.
            # The threshold -1e38 is from the original code.
            # Potential Improvement: This threshold might need to be more robust or configurable
            # if ParFlow's no-data value representation varies.
            saturation_data[saturation_data < -1e38] = np.nan 
            
            return saturation_data
        except Exception as e: # Catch errors from read_pfb or subsequent numpy operations
            raise ValueError(f"Error loading or processing saturation data from '{chosen_path}': {str(e)}")
    
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
            ValueError: If no timesteps are available, or if the specified range is invalid
                        (e.g., `start_idx` out of bounds, `end_idx` <= `start_idx` leading to empty range).
        """
        if not self.available_timesteps:
            # This check is also in load_timestep, but good for direct calls to load_time_range.
            raise ValueError("No ParFlow timesteps available. Cannot load saturation data.")
        
        # Validate and adjust start_idx and end_idx
        if not (0 <= start_idx < len(self.available_timesteps)):
             raise ValueError(f"start_idx {start_idx} is out of range for available timesteps (0 to {len(self.available_timesteps)-1}).")

        if end_idx is None:
            actual_end_idx = len(self.available_timesteps) # Go up to the last available timestep
        else:
            if end_idx < 0 : # Interpret negative end_idx relative to end, like list slicing
                actual_end_idx = len(self.available_timesteps) + end_idx
            else: # Positive or zero
                actual_end_idx = min(end_idx, len(self.available_timesteps))
        
        if actual_end_idx <= start_idx:
            # print(f"Warning: Requested time range (start_idx={start_idx}, end_idx={end_idx} -> actual_end_idx={actual_end_idx}) is empty or invalid.")
            # Return empty array with expected dimensions if possible (need to know spatial dims first)
            # For now, let timesteps_to_load handle this; if it's empty, ValueError will be raised.
            # Or, get spatial dims from first available timestep if start_idx is valid.
            if self.available_timesteps:
                 first_data_sample = self._load_saturation_for_timestep_num(self.available_timesteps[0])
                 return np.empty((0, *first_data_sample.shape)) # 0 timesteps, but correct spatial shape
            else: # Should not happen if initial check passed.
                 return np.empty((0,1,1,1)) # Fallback if no timesteps at all.


        # Get the list of actual ParFlow timestep numbers to load based on indices
        timesteps_to_load_numbers = self.available_timesteps[start_idx:actual_end_idx]
        
        if not timesteps_to_load_numbers:
            # This case should ideally be caught by actual_end_idx <= start_idx logic.
            # However, if slicing results in empty list for other reasons:
            # print(f"Warning: No timesteps selected for index range [{start_idx}, {actual_end_idx}).")
            # Determine spatial shape from the first available timestep for empty array structure
            first_data_sample = self._load_saturation_for_timestep_num(self.available_timesteps[0])
            return np.empty((0, *first_data_sample.shape))

        # Load the first timestep in the range to determine spatial dimensions (nz, ny, nx)
        # This assumes all PFB files for this variable have consistent dimensions.
        first_timestep_data = self._load_saturation_for_timestep_num(timesteps_to_load_numbers[0])
        nz, ny, nx = first_timestep_data.shape
        
        # Initialize a 4D NumPy array to store all loaded saturation data
        # Shape: (number_of_timesteps_in_range, nz, ny, nx)
        all_saturation_data = np.zeros((len(timesteps_to_load_numbers), nz, ny, nx))
        
        all_saturation_data[0] = first_timestep_data # Store the already loaded first timestep
        
        # Load the remaining timesteps in the range
        for i, timestep_num in enumerate(timesteps_to_load_numbers[1:], start=1): # Start from 1 as 0 is filled
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
        # Assumes timestep number can be directly used as a proxy for simulation time.
        # For more complex timing, ParFlow's run script or timing output files (.tcl, .timing)
        # would need to be parsed, which is beyond the scope of simple PFB reading.
        if not self.available_timesteps:
            return []
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
        # Call super().__init__ but note that _get_available_timesteps might not be
        # relevant if porosity is truly static and doesn't have timed output files.
        # However, the base class structure expects it.
        super().__init__(model_directory, run_name)
        # Porosity doesn't usually have multiple timesteps, so self.available_timesteps
        # from base class (derived from saturation/pressure files) might be misleading if used here.
    
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
        # Common filename patterns for ParFlow porosity files.
        # Order can matter if multiple potentially exist, though usually only one is standard.
        porosity_filename_candidates = [
            f"{self.run_name}.out.porosity.pfb", # Common for outputs
            f"{self.run_name}.porosity.pfb",     # Sometimes used if input or static field
            f"{self.run_name}.pf.porosity.pfb",  # Alternative prefixing
            # Add non-pfb extensions if ParFlow might write them as plain binary without .pfb
            f"{self.run_name}.out.porosity", 
            f"{self.run_name}.pf.porosity"
        ]
        
        found_file_path = None
        for filename in porosity_filename_candidates:
            file_path = os.path.join(self.model_directory, filename)
            if os.path.exists(file_path):
                found_file_path = file_path
                break # Found a candidate
        
        if not found_file_path:
            raise FileNotFoundError(
                f"Could not find a porosity file for run '{self.run_name}' in directory '{self.model_directory}'. "
                f"Checked patterns like '{self.run_name}.out.porosity.pfb'.")
        
        try:
            porosity_data = self.read_pfb(found_file_path)
            # Handle ParFlow's no-data values, similar to saturation.
            porosity_data[porosity_data < -1e38] = np.nan 
            # Potential Issue: Porosity should ideally be between 0 and 1.
            # Add validation or clipping if ParFlow might output other values for active cells.
            # e.g., np.clip(porosity_data, 0.0, 1.0) after NaNs are set.
            # However, if -1e38 are truly no-data, they should remain NaN, not clipped to 0.
            return porosity_data
        except Exception as e:
            raise ValueError(f"Error loading or processing porosity data from '{found_file_path}': {str(e)}")
    
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
        # Common filename patterns for ParFlow mask files.
        mask_filename_candidates = [
            f"{self.run_name}.out.mask.pfb",
            f"{self.run_name}.mask.pfb",
            f"{self.run_name}.pf.mask.pfb",
            f"{self.run_name}.out.mask", # Fallback for non-.pfb extension
            f"{self.run_name}.pf.mask"
        ]
        
        found_file_path = None
        for filename in mask_filename_candidates:
            file_path = os.path.join(self.model_directory, filename)
            if os.path.exists(file_path):
                found_file_path = file_path
                break
        
        if not found_file_path:
            raise FileNotFoundError(
                f"Could not find a mask file for run '{self.run_name}' in directory '{self.model_directory}'. "
                f"Checked patterns like '{self.run_name}.out.mask.pfb'.")
        
        try:
            mask_data = self.read_pfb(found_file_path)
            # ParFlow mask values are typically 0 (inactive) or 1 (active).
            # No-data value handling might not be strictly necessary if format is clean 0s and 1s.
            # However, applying it consistently with other PFB reads doesn't hurt.
            mask_data[mask_data < -1e38] = np.nan # Or set to 0 if NaN is not desired for mask
            return mask_data
        except Exception as e:
            raise ValueError(f"Error loading or processing mask data from '{found_file_path}': {str(e)}")
        
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
        # Porosity is static, so any timestep_idx request returns the same data.
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
        
        num_timesteps_in_output: int
        if not self.available_timesteps: # No timed output files found by base class
            num_timesteps_in_output = 1 # Assume at least one "time" for static data
        elif end_idx is None:
            # If end_idx is not specified, use timesteps from start_idx to end of available list
            if start_idx < 0 : start_idx = 0 # Ensure start_idx is not negative
            num_timesteps_in_output = len(self.available_timesteps[start_idx:])
        else:
            # Adjust end_idx if it's negative or too large
            actual_end_idx = end_idx
            if end_idx < 0: actual_end_idx = len(self.available_timesteps) + end_idx
            actual_end_idx = min(actual_end_idx, len(self.available_timesteps))
            
            # Ensure start_idx is valid
            if not (0 <= start_idx < len(self.available_timesteps)): start_idx = 0

            num_timesteps_in_output = actual_end_idx - start_idx
        
        num_timesteps_in_output = max(0, num_timesteps_in_output) # Ensure not negative
        if num_timesteps_in_output == 0 and self.available_timesteps: # If range was invalid but timesteps exist
            num_timesteps_in_output = 1 # Default to 1 repetition if a valid range wasn't specified but data exists
        elif num_timesteps_in_output == 0 and not self.available_timesteps: # No range, no data
             return np.empty((0, *porosity_3d.shape))


        # Add a new time axis and tile the 3D porosity array
        # Shape becomes (1, nz, ny, nx), then tiled to (num_timesteps_in_output, nz, ny, nx)
        porosity_4d = np.tile(porosity_3d[np.newaxis, ...], (num_timesteps_in_output, 1, 1, 1))
        
        return porosity_4d
    
    def get_timestep_info(self) -> List[Tuple[int, float]]:
        """
        Returns timestep information, typically based on other ParFlow outputs
        (like saturation) as porosity itself is static.
        
        Returns:
            List[Tuple[int, float]]: A list of (timestep_number, time_value) tuples,
                                     derived from `self.available_timesteps`.
        """
        # Porosity is static, but to align with the interface, return timestep info
        # based on what other timed outputs (e.g., saturation) suggest.
        if not self.available_timesteps:
            # If no other timed outputs were found to define timesteps,
            # provide a single dummy entry for the static porosity data.
            return [(0, 0.0)] # (timestep_number, conceptual_time)
        return [(ts_num, float(ts_num)) for ts_num in self.available_timesteps]