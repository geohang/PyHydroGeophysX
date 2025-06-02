"""
Module for handling MODFLOW Unsaturated-Zone Flow (UZF) package water content data.

This module provides a class `MODFLOWWaterContent` (which seems to be a duplicate
or very similar to the one in `modflow_output.py` but focused here) for reading
binary 'WaterContent' files produced by MODFLOW's UZF package. It also includes
a utility for calculating saturation.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Any # Added Any for file_obj in binaryread


# This binaryread function is identical to the one in modflow_output.py.
# To avoid duplication, it would typically be in a shared utility module.
# For this exercise, it's documented here as per the file context.
# SUGGESTION: Consolidate this function with the one in modflow_output.py into a common utility module
# to adhere to the DRY (Don't Repeat Yourself) principle.
def binaryread(file_obj: Any, # Type hint should ideally be BinaryIO from `typing`
               vartype: Union[type, List[Tuple[str, str]]], # Can be a basic type like `str` or `np.float64`, or a structured dtype list
               shape: Tuple[int, ...] = (1,), # Default shape for array data
               charlen: int = 16) -> Union[bytes, np.ndarray, np.void]: # Return type varies
    """
    Reads data from an open binary file using numpy.fromfile or file.read.

    Designed for MODFLOW binary output files, handling various data types including
    standard numpy dtypes, strings, and structured arrays (records).

    Args:
        file_obj: Open file object in binary read mode (e.g., from `open(path, 'rb')`).
        vartype: The variable type to read.
                 - For strings: `str`. Result will be raw bytes.
                 - For standard numerical data: A numpy dtype like `np.float64`, `np.int32`.
                 - For structured data (records): A list of tuples defining the structure,
                   e.g., `[('kstp', '<i4'), ('kper', '<i4')]`.
        shape (Tuple[int, ...], optional): The desired shape of the output NumPy array if
                                           `vartype` is a standard numerical dtype.
                                           Defaults to `(1,)`, meaning a single value or 1D array of one element.
        charlen (int, optional): The length of the string (number of bytes) to read if `vartype` is `str`.
                                 Defaults to 16, a common length for text identifiers in MODFLOW files.

    Returns:
        Union[bytes, np.ndarray, np.void]:
            - `bytes`: If `vartype` is `str`, returns the raw bytes read from the file.
            - `np.ndarray`: If `vartype` is a standard numerical dtype, returns a NumPy array
                            of the specified `shape`.
            - `np.void`: If `vartype` is a structured dtype list, returns a single NumPy void object
                         (structured scalar) representing one record.

    Raises:
        EOFError: If the end of the file is reached unexpectedly while trying to read
                  the requested number of bytes or data elements.
    """
    if vartype == str:
        # Read `charlen` bytes directly from the file for string types.
        # Note: This returns raw bytes. Decoding (e.g., to UTF-8 or ASCII) and stripping
        # null characters would be an additional step if a Python string is needed.
        # SUGGESTION: Consider adding optional decoding with error handling, e.g., file_obj.read(charlen).decode('ascii', 'ignore').
        data_bytes = file_obj.read(charlen)
        if len(data_bytes) < charlen: # Check if enough bytes were read
            raise EOFError(f"Attempted to read {charlen} bytes for string, but reached EOF.")
        return data_bytes
    elif isinstance(vartype, list): # Indicates a structured dtype, e.g., for a header record.
        dt = np.dtype(vartype) # Create a numpy dtype object from the list definition.
        # Read one item (record) of this structured type from the file.
        # `count=1` ensures numpy attempts to read exactly one full record.
        record = np.fromfile(file_obj, dtype=dt, count=1)
        if record.size == 0: # Check if np.fromfile returned an empty array (EOF).
            raise EOFError("Attempted to read a structured record but reached EOF before reading any data.")
        return record[0] # Return the single structured element (np.void or structured scalar).
    else: # Assumed to be a standard numpy dtype (e.g., np.float64, np.int32).
        num_values_to_read = int(np.prod(shape)) # Calculate total number of elements to read.
        # Read data into a flat NumPy array.
        data_array = np.fromfile(file_obj, dtype=vartype, count=num_values_to_read)

        # Check if the expected number of values were read.
        if data_array.size < num_values_to_read:
            raise EOFError(f"Attempted to read {num_values_to_read} values of type {vartype}, "
                           f"but only found {data_array.size} values (likely EOF reached prematurely).")

        # Reshape the flat array to the desired output shape.
        # If shape is (1,), it will be a 1D array with one element.
        # If num_values_to_read was 1, this still returns a 1-element array.
        # Callers expecting a scalar for shape (1,) should do `result[0]`.
        return np.reshape(data_array, shape)


class MODFLOWWaterContent: # This class definition seems to be a focal point of this file.
                           # If it's meant to be distinct from a similar class in `modflow_output.py`,
                           # its unique aspects or context should be clear.
                           # For this exercise, comments are added based on its current state here.
    """
    Processes water content data from MODFLOW's UZF (Unsaturated-Zone Flow) package.

    This class reads the binary 'WaterContent' file output by MODFLOW when the UZF
    package is active and output is requested. It maps the 1D array of UZF cell
    water contents back to a 2D or 3D grid based on the provided `idomain`.
    It also includes a method to calculate saturation from water content and porosity.

    Attributes:
        sim_ws (str): Path to the simulation workspace.
        idomain (np.ndarray): The 2D idomain array used for mapping UZF cells.
        nrows (int): Number of rows in the model grid.
        ncols (int): Number of columns in the model grid.
        iuzno_dict_rev (Dict[int, Tuple[int,int]]): Reverse lookup dictionary mapping
                                                    sequential UZF cell number to (row, col) index.
        nuzfcells_2d (int): Number of active UZF cells in the 2D plane (derived from idomain).
    """
    
    def __init__(self, sim_ws: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.

        Args:
            sim_ws (str): Path to the MODFLOW simulation workspace (directory containing
                          the 'WaterContent' output file).
            idomain (np.ndarray): A 2D or 3D integer NumPy array indicating active model cells
                                  (typically, >0 for active, 0 for inactive). If 3D, the
                                  first layer (idomain[0,:,:]) is used for UZF cell mapping,
                                  as UZF is typically associated with the top model layer.

        Raises:
            TypeError: If `idomain` is not a NumPy array.
            ValueError: If the effective `idomain` (after potentially taking the first slice from 3D) is not 2D.
            FileNotFoundError: If `sim_ws` directory does not exist or is not a directory.
        """
        # Validate simulation workspace path
        if not os.path.isdir(sim_ws):
            raise FileNotFoundError(f"Simulation workspace directory not found: {sim_ws}")
        self.sim_ws = sim_ws # Store simulation workspace path

        # Validate idomain type
        if not isinstance(idomain, np.ndarray):
            raise TypeError("idomain must be a NumPy array.")

        # --- Determine the relevant 2D idomain slice for UZF mapping ---
        # The UZF package typically operates on a 2D footprint of cells, often related to the top model layer.
        # If a 3D idomain is provided, this logic assumes the first layer (index 0) is representative.
        # SUGGESTION: If UZF can be associated with layers other than the top, this logic might need adjustment
        # or a parameter to specify which layer of idomain to use.
        current_idomain_slice: np.ndarray
        if idomain.ndim == 3:
            current_idomain_slice = idomain[0, :, :] # Use the first layer for UZF mapping
        elif idomain.ndim == 2:
            current_idomain_slice = idomain # Use the provided 2D idomain directly
        else:
            raise ValueError(f"idomain must be a 2D or 3D array; got {idomain.ndim}D.")

        # Store the effective 2D idomain slice and its dimensions.
        self.idomain = current_idomain_slice
        self.nrows, self.ncols = self.idomain.shape
        
        # --- Create a reverse lookup for UZF cell numbering ---
        # UZF data in 'WaterContent' files is often a 1D array for active UZF cells.
        # This dictionary maps the sequential UZF cell number (iuzno) back to its (row, column) grid index.
        self.iuzno_dict_rev: Dict[int, Tuple[int, int]] = {} # {iuzno: (row, col)}
        iuzno_counter = 0 # Counter for active UZF cells
        for r_idx in range(self.nrows): # Iterate through rows
            for c_idx in range(self.ncols): # Iterate through columns
                # Cells with idomain > 0 are considered active for UZF.
                if self.idomain[r_idx, c_idx] > 0:
                    self.iuzno_dict_rev[iuzno_counter] = (r_idx, c_idx) # Map counter to (row, col)
                    iuzno_counter += 1
        
        # Store the total number of active UZF cells in the 2D grid plane.
        self.nuzfcells_2d = len(self.iuzno_dict_rev)
        if self.nuzfcells_2d == 0:
            # Warn if no active cells are found, as subsequent data loading will likely fail or return empty.
            print("Warning: No active UZF cells found based on the provided idomain (all idomain values <= 0 in the relevant slice).")
    
    def load_timestep(self, timestep_idx: int, nlay_uzf: int = 3) -> np.ndarray:
        """
        Load water content data for a single, specific timestep.

        Args:
            timestep_idx (int): The zero-based index of the timestep to load from the 'WaterContent' file.
            nlay_uzf (int, optional): The number of unsaturated zone layers simulated in UZF,
                                   which determines how many values are stored per (row, col) UZF cell.
                                   Defaults to 3. This must match the UZF package configuration.

        Returns:
            np.ndarray: A 3D NumPy array of water content values with shape (nlay_uzf, nrows, ncols).
                        Values for inactive grid cells (where idomain <= 0) will be NaN.

        Raises:
            IndexError: If `timestep_idx` results in no data being loaded (e.g., it's out of range for the file).
            RuntimeError: If `load_time_range` returns an unexpected shape.
        """
        # This method provides a simpler interface to load a single timestep
        # by calling the more general `load_time_range` method for a range of one.
        # SUGGESTION: For critical performance with many single timestep loads, this could be optimized
        # to avoid the overhead of creating a 4D array in `load_time_range` just to extract one slice.
        # However, for typical use, this delegation is often acceptable.

        # Load a time range covering only the requested timestep_idx.
        data_4d = self.load_time_range(start_idx=timestep_idx, end_idx=timestep_idx + 1, nlay_uzf=nlay_uzf)

        # Validate the result from load_time_range.
        if data_4d.shape[0] == 1:
            return data_4d[0] # Return the first (and only) 3D array from the 4D result.
        elif data_4d.shape[0] == 0:
            # This means load_time_range found no data for this specific index.
             raise IndexError(f"Timestep index {timestep_idx} resulted in no data being loaded. "
                              "It might be out of range for the 'WaterContent' file, or the file could be empty/corrupt.")
        else:
            # This case should ideally not be reached if load_time_range works correctly for end_idx = start_idx + 1.
            raise RuntimeError(f"Unexpected data shape {data_4d.shape} returned when loading single timestep {timestep_idx}. "
                               "Expected 1 timestep, got {data_4d.shape[0]}.")

    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                      nlay_uzf: int = 3) -> np.ndarray:
        """
        Load water content data for a specified range of timesteps from the 'WaterContent' file.

        Args:
            start_idx (int, optional): Zero-based starting timestep index. Defaults to 0.
            end_idx (Optional[int], optional): Zero-based ending timestep index (exclusive).
                                               If None, loads all timesteps from `start_idx` to
                                               the end of the file. Defaults to None.
            nlay_uzf (int, optional): The number of unsaturated zone layers in the UZF model.
                                   This dictates how many data values are read per active UZF cell
                                   at each timestep. Defaults to 3.

        Returns:
            np.ndarray: A 4D NumPy array of water content values, with shape
                        (num_timesteps_loaded, nlay_uzf, nrows, ncols).
                        Returns an empty 4D array (shape (0, nlay_uzf, nrows, ncols))
                        if no timesteps are loaded or if an error occurs during initial file access.
        """
        # If there are no active UZF cells based on idomain, no data can be loaded.
        if self.nuzfcells_2d == 0:
            # This warning was already issued at __init__, but good to be explicit here too.
            print("Warning: No active UZF cells (nuzfcells_2d is 0). Cannot load water content data. Returning empty array.")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols)) # Return empty array with correct rank

        # Calculate the total number of UZF data points (values) expected per full timestep record in the binary file.
        # This is the number of active 2D UZF cells multiplied by the number of UZF layers.
        total_uzf_data_points_per_record = self.nuzfcells_2d * nlay_uzf
        
        # Construct the full path to the MODFLOW 'WaterContent' file.
        wc_file_path = os.path.join(self.sim_ws, "WaterContent")
        if not os.path.exists(wc_file_path): # Check if the file exists before attempting to open.
            raise FileNotFoundError(f"MODFLOW 'WaterContent' file not found in the specified simulation workspace: {wc_file_path}")

        all_timesteps_data_list: List[np.ndarray] = [] # List to accumulate 3D arrays for each timestep.
        
        # Define the structured dtype for reading the header of each record in the MODFLOW binary file.
        # This matches the typical format: kstp, kper, pertim, totim, text, maxbound, and two auxiliary integers.
        header_dtype = np.dtype([
            ("kstp", "<i4"),      # Timestep number in current stress period
            ("kper", "<i4"),      # Stress period number
            ("pertim", "<f8"),    # Time in current stress period
            ("totim", "<f8"),     # Total simulation time
            ("text", "S16"),     # Text description (e.g., "VOLUMETRIC WC")
            ("maxbound", "<i4"),  # Number of UZF cells in this record (should match total_uzf_data_points_per_record)
            ("aux1", "<i4"),      # Auxiliary field 1 (often layer number or constant) - name changed for clarity
            ("aux2", "<i4"),      # Auxiliary field 2 (often constant) - name changed for clarity
        ])
        # Define the dtype for reading a single water content data point (float64).
        data_point_dtype = np.dtype([("data", "<f8")]) # Reading as a structured type with one field 'data'

        try:
            # Open the 'WaterContent' file in binary read mode.
            with open(wc_file_path, "rb") as file:
                # --- Skip records to reach the start_idx ---
                for _ in range(start_idx):
                    try:
                        # Read and discard the header of the timestep to skip.
                        _ = binaryread(file, header_dtype)
                        # Seek past the data block for this timestep.
                        # This is faster than reading all data points individually.
                        bytes_to_skip = total_uzf_data_points_per_record * data_point_dtype.itemsize
                        file.seek(bytes_to_skip, os.SEEK_CUR) # Move file cursor forward
                    except EOFError:
                        # If EOF is reached while trying to skip to start_idx.
                        print(f"Warning: EOF reached while skipping to start_idx {start_idx} in 'WaterContent' file. No data will be loaded.")
                        return np.empty((0, nlay_uzf, self.nrows, self.ncols)) # Return empty array
                    except Exception as e: # Catch other potential errors during skipping
                        print(f"Error occurred while skipping to timestep {start_idx} in 'WaterContent' file: {e}")
                        return np.empty((0, nlay_uzf, self.nrows, self.ncols))
                
                # --- Read the requested range of timesteps ---
                timesteps_read_count = 0
                while True:
                    # Stop reading if end_idx is specified and reached.
                    if end_idx is not None and timesteps_read_count >= (end_idx - start_idx):
                        break

                    try:
                        # Read the header for the current timestep.
                        header_data = binaryread(file, header_dtype)

                        # Optional: Validate 'maxbound' from header against expected points.
                        # maxbound_from_header = int(header_data['maxbound'])
                        # if maxbound_from_header != total_uzf_data_points_per_record:
                        #     print(f"Warning: 'maxbound' in header ({maxbound_from_header}) does not match expected "
                        #           f"UZF data points ({total_uzf_data_points_per_record}) at timestep count {timesteps_read_count}. "
                        #           "This may indicate incorrect nlay_uzf or a corrupt file.")
                        #     break # Stop further processing due to this inconsistency.

                        # Initialize a 3D array (nlay_uzf, nrows, ncols) with NaNs for the current timestep.
                        current_wc_3d_array = np.full((nlay_uzf, self.nrows, self.ncols), np.nan)

                        # Read water content data for each UZF layer and each active 2D UZF cell.
                        # Data is typically ordered by UZF cell number (iuzno), then by UZF layer.
                        # Or, all cells for layer 1, then all for layer 2, etc. This code assumes the latter.
                        for k_layer_idx in range(nlay_uzf): # Iterate through UZF layers
                            for iuzno_2d_idx in range(self.nuzfcells_2d): # Iterate through active 2D UZF cells
                                # Read a single data point (water content value).
                                wc_value_struct = binaryread(file, data_point_dtype)
                                wc_value = wc_value_struct['data'] # Extract the float value from the structured scalar.

                                # Get the (row, col) grid indices for this UZF cell.
                                r_idx, c_idx = self.iuzno_dict_rev[iuzno_2d_idx]
                                # Assign the water content value to the corresponding position in the 3D array.
                                current_wc_3d_array[k_layer_idx, r_idx, c_idx] = wc_value

                        all_timesteps_data_list.append(current_wc_3d_array) # Add the current timestep's data.
                        timesteps_read_count += 1

                    except EOFError:
                        # This is the normal way to detect the end of the file if end_idx is None.
                        break
                    except Exception as e:
                        # Handle other errors during data reading for a specific timestep.
                        print(f"Error reading data at loaded timestep count {timesteps_read_count} "
                              f"(effective file index {start_idx + timesteps_read_count}): {e}")
                        break # Stop processing further timesteps on error.
        
        except FileNotFoundError: # This should have been caught by the os.path.exists check earlier.
            # Defensive coding: re-raise if it somehow occurs here.
            raise
        except Exception as e: # Catch other potential errors like permission issues for open().
            print(f"Failed to open or process 'WaterContent' file at '{wc_file_path}': {e}")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols)) # Return empty on failure.

        # If no data was successfully loaded into the list (e.g., range was empty or error on first read).
        if not all_timesteps_data_list:
            # A warning/info message might be useful here if the range was valid but file ended early.
            # print("Warning: No data loaded. The specified range might be empty or past EOF.")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols))

        # Convert the list of 3D arrays into a single 4D NumPy array.
        return np.array(all_timesteps_data_list)
    
    def calculate_saturation(self, water_content: np.ndarray, 
                           porosity: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculate volumetric saturation from water content and porosity.

        Saturation (S) is computed as S = water_content / porosity.
        The result is clipped to the range [0.0, 1.0].

        Args:
            water_content (np.ndarray): NumPy array of water content values. Can be
                                        for a single timestep (e.g., [nlay, nrow, ncol])
                                        or multiple timesteps (e.g., [time, nlay, nrow, ncol]).
            porosity (Union[float, np.ndarray]): Porosity of the medium. Can be a scalar
                                                 (uniform porosity) or a NumPy array. If an array,
                                                 its dimensions must be compatible with `water_content`
                                                 (e.g., matching spatial dimensions for broadcasting
                                                 across time if needed).

        Returns:
            np.ndarray: NumPy array of calculated saturation values, same shape as `water_content`,
                        with values clipped between 0 and 1.

        Raises:
            ValueError: If `porosity` is an array and its dimensions are incompatible
                        with `water_content` for element-wise division.
            TypeError: If inputs are not of expected types (NumPy arrays, float/int for scalar porosity).
        """
        # This method calculates volumetric saturation. It's similar to a method that might
        # exist in a base class (like HydroModelOutput from base.py).
        # If this MODFLOWWaterContent class is intended to inherit from such a base and
        # the calculation is identical, `super().calculate_saturation(...)` could be used.
        # For this exercise, commenting the implementation as it stands.

        # --- Input Validation ---
        if not isinstance(water_content, np.ndarray):
            raise TypeError("water_content must be a NumPy array.")
        if not isinstance(porosity, (int, float, np.ndarray)): # Allow int for scalar porosity
            raise TypeError("porosity must be a float, int, or NumPy array.")

        # --- Warning for Non-Positive Porosity ---
        # Non-positive porosity values can lead to division by zero or physically meaningless saturation.
        if isinstance(porosity, np.ndarray):
            if np.any(porosity <= 0): # Check if any element in porosity array is <= 0
                print("Warning: Porosity array contains zero or negative values. Saturation calculation may result in NaNs, Infs, or physically unrealistic values.")
        elif porosity <= 0: # Check for scalar porosity
            print("Warning: Scalar porosity is zero or negative. Saturation calculation may result in NaNs, Infs, or physically unrealistic values.")

        # --- Saturation Calculation ---
        saturation_intermediate: np.ndarray # To hold result before clipping
        if isinstance(porosity, (int, float)): # Scalar porosity
            porosity_float = float(porosity) # Ensure float for division
            # Use np.divide for safer division, especially to handle porosity_float == 0.0.
            # np.divide by zero gives inf (for non-zero numerator) or nan (for 0/0) without raising error immediately.
            saturation_intermediate = np.divide(water_content, porosity_float)
        else: # Porosity is a NumPy array
            # Check for dimension compatibility for broadcasting, especially if water_content has a time dimension
            # and porosity is static (spatial only).
            if porosity.ndim != water_content.ndim:
                # Common case: water_content is 4D [time, nlay, nrow, ncol] and porosity is 3D [nlay, nrow, ncol].
                if water_content.ndim == porosity.ndim + 1 and water_content.shape[1:] == porosity.shape:
                    # Add a new axis to porosity to align for broadcasting with time dimension of water_content.
                    # e.g., porosity shape (nlay,nrow,ncol) -> (1,nlay,nrow,ncol)
                    porosity_expanded = porosity[np.newaxis, ...]
                    saturation_intermediate = np.divide(water_content, porosity_expanded)
                else:
                    # If dimensions are not compatible in this specific way, raise an error.
                    raise ValueError(
                        f"Porosity array dimensions ({porosity.ndim}, shape {porosity.shape}) are not compatible "
                        f"with water_content dimensions ({water_content.ndim}, shape {water_content.shape}) for standard broadcasting."
                    )
            else: # Dimensions are the same, direct element-wise division (or broadcasting if shapes differ but are compatible).
                saturation_intermediate = np.divide(water_content, porosity)
        
        # --- Clipping to Physical Range [0, 1] ---
        # Saturation values should be between 0 and 1. Clipping handles:
        # - Numerical inaccuracies where water_content might slightly exceed porosity (S > 1).
        # - Division by zero (porosity=0, water_content>0 leads to inf, clipped to 1).
        #   This might mask issues; if porosity is 0, saturation is arguably undefined or should be NaN.
        # - Division of 0/0 (porosity=0, water_content=0 leads to NaN, clipped to 0 by np.clip).
        # SUGGESTION: For physical realism, if porosity is zero where water_content is non-zero,
        # saturation should ideally be NaN rather than 1.0. This requires more specific handling
        # before clipping, e.g., `saturation_intermediate[np.isinf(saturation_intermediate)] = np.nan`.
        saturation_clipped = np.clip(saturation_intermediate, 0.0, 1.0)
        
        return saturation_clipped
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Reads the 'WaterContent' file to extract header information for each timestep.

        Returns:
            List[Tuple[int, int, float, float]]: A list of tuples, where each tuple
                                                 contains (kstp, kper, pertim, totim)
                                                 for a timestep:
                                                 - kstp (int): Timestep number within the stress period.
                                                 - kper (int): Stress period number.
                                                 - pertim (float): Time within the current stress period (relative to start of stress period).
                                                 - totim (float): Total simulation time (absolute).
        """
        # This method reads through the 'WaterContent' file, parsing only the header of each record
        # to extract time information, then skips the actual data block.
        
        wc_file_path = os.path.join(self.sim_ws, "WaterContent") # Path to the data file.
        if not os.path.exists(wc_file_path):
            print(f"Warning: 'WaterContent' file not found at {wc_file_path}. Cannot retrieve timestep info.")
            return [] # Return empty list if file doesn't exist.

        timestep_info_list: List[Tuple[int, int, float, float]] = [] # To store (kstp, kper, pertim, totim) tuples.

        # Define the structured dtype for the header, matching the one in load_time_range.
        header_dtype = np.dtype([
            ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
            ("text", "S16"), ("maxbound", "<i4"), ("aux1", "<i4"), ("aux2", "<i4"),
        ])
        # Item size for a single data point (float64 for water content) used for calculating skip bytes.
        data_point_itemsize = np.dtype("<f8").itemsize

        try:
            with open(wc_file_path, "rb") as file:
                while True: # Loop until EOF or error
                    try:
                        # Read the header of the current record.
                        header_data_record = binaryread(file, header_dtype)

                        # Extract time information from the header record.
                        # Assumes binaryread returns a structured np.void object for structured dtypes.
                        kstp = int(header_data_record['kstp'])
                        kper = int(header_data_record['kper'])
                        pertim = float(header_data_record['pertim'])
                        totim = float(header_data_record['totim'])
                        maxbound_in_header = int(header_data_record['maxbound']) # Number of data values in the subsequent block.

                        timestep_info_list.append((kstp, kper, pertim, totim)) # Add to list.

                        # Skip the data block associated with this header.
                        # The number of bytes to skip is maxbound_in_header * size_of_each_data_point.
                        bytes_to_skip = maxbound_in_header * data_point_itemsize
                        file.seek(bytes_to_skip, os.SEEK_CUR) # Move file cursor forward.

                    except EOFError:
                        # Expected way to terminate the loop when the end of the file is reached.
                        break
                    except Exception as e:
                        # Handle other potential errors during reading a record or skipping data.
                        print(f"Error reading timestep info or skipping data within 'WaterContent' file: {e}")
                        break # Stop processing on error.
        except Exception as e:
            # Handle errors related to opening or initial processing of the file.
            print(f"Failed to open or process 'WaterContent' file for timestep info: {e}")
            return [] # Return empty list on failure.

        return timestep_info_list
