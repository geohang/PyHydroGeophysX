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
def binaryread(file_obj: Any, # Should be BinaryIO
               vartype: Union[type, List[Tuple[str, str]]],
               shape: Tuple[int, ...] = (1,),
               charlen: int = 16) -> Union[bytes, np.ndarray, np.void]:
    """
    Reads data from an open binary file using numpy.fromfile or file.read.

    Designed for MODFLOW binary output files, handling various data types.

    Args:
        file_obj: Open file object in binary read mode.
        vartype: Variable type to read (e.g., np.float64, str, or structured dtype list).
        shape (Tuple[int, ...], optional): Desired output shape for standard numpy dtypes. Defaults to (1,).
        charlen (int, optional): Length for string types if `vartype` is `str`. Defaults to 16.

    Returns:
        Union[bytes, np.ndarray, np.void]: Data read from file. `bytes` for `str` type,
                                           `np.ndarray` for standard dtypes, `np.void` for structured.

    Raises:
        EOFError: If EOF is reached unexpectedly while reading data for standard dtypes.
    """
    if vartype == str:
        # Reads `charlen` bytes. Does not decode.
        # Potential Improvement: Add optional decoding and stripping of null chars.
        return file_obj.read(charlen)
    elif isinstance(vartype, list): # Assuming structured dtype list
        dt = np.dtype(vartype)
        # Reads one item of the structured type.
        # `count=1` ensures it tries to read exactly one full record.
        record = np.fromfile(file_obj, dtype=dt, count=1)
        if record.size == 0: # Check if anything was read
            raise EOFError("Attempted to read a structured record but reached EOF.")
        return record[0] # Return the single structured void/scalar
    else: # Standard numpy dtype
        num_values_to_read = int(np.prod(shape)) # Ensure integer for count
        data_array = np.fromfile(file_obj, dtype=vartype, count=num_values_to_read)

        if data_array.size < num_values_to_read:
            raise EOFError(f"Attempted to read {num_values_to_read} values of type {vartype}, "
                           f"but only found {data_array.size} values (EOF reached).")

        # Original code had `result = result` for nval == 1.
        # `np.fromfile` already returns an array. Reshaping is appropriate.
        # If a scalar is specifically desired for shape (1,), the caller should extract it via data_array[0].
        return np.reshape(data_array, shape)


class MODFLOWWaterContent: # Renamed to avoid direct conflict if imported alongside the other one.
                           # Or, this is the primary one if water_content.py is the intended module.
                           # Assuming this is the version to be documented from water_content.py.
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
                                  first layer (index 0) is used for UZF cell mapping, as UZF
                                  is typically associated with the top model layer.

        Raises:
            TypeError: If `idomain` is not a NumPy array.
            ValueError: If the effective `idomain` (after potentially taking the first slice) is not 2D.
            FileNotFoundError: If `sim_ws` directory does not exist.
        """
        if not os.path.isdir(sim_ws):
            raise FileNotFoundError(f"Simulation workspace directory not found: {sim_ws}")
        self.sim_ws = sim_ws

        if not isinstance(idomain, np.ndarray):
            raise TypeError("idomain must be a NumPy array.")

        # UZF cells are typically related to the top active layer of the model.
        # If a 3D idomain is provided, use its first layer (slice).
        current_idomain_slice = idomain[0, :, :] if idomain.ndim == 3 else idomain
        if current_idomain_slice.ndim != 2:
            raise ValueError("The effective idomain for UZF mapping must be a 2D array.")

        self.idomain = current_idomain_slice
        self.nrows, self.ncols = self.idomain.shape
        
        self.iuzno_dict_rev: Dict[int, Tuple[int, int]] = {}
        iuzno_counter = 0
        for r_idx in range(self.nrows):
            for c_idx in range(self.ncols):
                if self.idomain[r_idx, c_idx] > 0:  # Active cells where UZF is applied
                    self.iuzno_dict_rev[iuzno_counter] = (r_idx, c_idx)
                    iuzno_counter += 1
        
        self.nuzfcells_2d = len(self.iuzno_dict_rev) # Number of active UZF cells in the 2D grid plane
        if self.nuzfcells_2d == 0:
            print("Warning: No active UZF cells found based on the provided idomain (all idomain values <= 0 in the mapping slice).")
    
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
            IndexError: If `timestep_idx` results in no data being loaded (e.g., out of bounds).
            RuntimeError: If data loading for the specific timestep fails unexpectedly.
        """
        # This is a convenience method that calls load_time_range for a single timestep.
        # Potential Improvement: Could be optimized to avoid creating a 4D array if performance is critical.
        data_4d = self.load_time_range(start_idx=timestep_idx, end_idx=timestep_idx + 1, nlay_uzf=nlay_uzf)
        if data_4d.shape[0] == 1:
            return data_4d[0] # Extract the single 3D array
        elif data_4d.shape[0] == 0:
             raise IndexError(f"Timestep index {timestep_idx} resulted in no data being loaded. It might be out of range or the file might be empty/corrupt.")
        else: # Should not happen if end_idx = start_idx + 1
            raise RuntimeError(f"Unexpected data shape {data_4d.shape} when loading single timestep {timestep_idx}. Expected 1 timestep.")

    
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
        if self.nuzfcells_2d == 0:
            print("Warning: No active UZF cells defined by idomain; returning empty array for water content.")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols))

        # Total number of data points per full timestep record in the binary file
        total_uzf_data_points_per_record = self.nuzfcells_2d * nlay_uzf
        
        wc_file_path = os.path.join(self.sim_ws, "WaterContent")
        if not os.path.exists(wc_file_path):
            raise FileNotFoundError(f"'WaterContent' file not found in simulation workspace: {wc_file_path}")

        all_timesteps_data_list: List[np.ndarray] = []
        
        # Define the structured dtype for reading the MODFLOW binary file header.
        header_dtype = np.dtype([
            ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
            ("text", "S16"), ("maxbound", "<i4"), ("aux1", "<i4"), ("aux2", "<i4"),
        ])
        # Define dtype for reading a single water content data point.
        data_point_dtype = np.dtype([("data", "<f8")])

        try:
            with open(wc_file_path, "rb") as file:
                # Skip records to reach the start_idx
                for _ in range(start_idx):
                    try:
                        _ = binaryread(file, header_dtype) # Read and discard header
                        # Skip the data block based on total_uzf_data_points_per_record
                        file.seek(total_uzf_data_points_per_record * data_point_dtype.itemsize, os.SEEK_CUR)
                    except EOFError:
                        print(f"Warning: EOF reached while skipping to start_idx {start_idx}. No data will be loaded.")
                        return np.empty((0, nlay_uzf, self.nrows, self.ncols))
                    except Exception as e:
                        print(f"Error while skipping to timestep {start_idx} in 'WaterContent' file: {e}")
                        return np.empty((0, nlay_uzf, self.nrows, self.ncols))
                
                # Read the requested range of timesteps
                timesteps_read_count = 0
                while True:
                    # Check if end_idx is met
                    if end_idx is not None and timesteps_read_count >= (end_idx - start_idx):
                        break

                    try:
                        header_data = binaryread(file, header_dtype) # Read one header record

                        # Validate 'maxbound' from header if necessary.
                        # maxbound_from_header = header_data['maxbound']
                        # if maxbound_from_header != total_uzf_data_points_per_record:
                        #     print(f"Warning: Mismatch in expected UZF data points. Header: {maxbound_from_header}, Expected: {total_uzf_data_points_per_record}. May indicate incorrect nlay_uzf.")
                        #     break # Stop further processing due to inconsistency

                        # Initialize a 3D array for the current timestep's water content data
                        current_wc_3d_array = np.full((nlay_uzf, self.nrows, self.ncols), np.nan)

                        # Read water content data for each UZF layer and each active 2D UZF cell
                        for k_layer_idx in range(nlay_uzf):
                            for iuzno_2d_idx in range(self.nuzfcells_2d):
                                wc_value_struct = binaryread(file, data_point_dtype) # Read one data point
                                wc_value = wc_value_struct['data'] # Extract float from the structured scalar

                                r_idx, c_idx = self.iuzno_dict_rev[iuzno_2d_idx] # Map to grid cell
                                current_wc_3d_array[k_layer_idx, r_idx, c_idx] = wc_value

                        all_timesteps_data_list.append(current_wc_3d_array)
                        timesteps_read_count += 1

                    except EOFError: # Expected way to finish if end_idx is None (read till end)
                        # print("Info: Reached end of 'WaterContent' file.")
                        break
                    except Exception as e:
                        print(f"Error reading data at loaded timestep count {timesteps_read_count} (file index {start_idx + timesteps_read_count}): {e}")
                        break
        
        except FileNotFoundError: # Should be caught by initial check, but defense-in-depth
            raise
        except Exception as e: # Catch other errors like permission issues for open()
            print(f"Failed to open or process 'WaterContent' file at '{wc_file_path}': {e}")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols))

        if not all_timesteps_data_list:
            # print("Warning: No data loaded. The specified range might be empty or past EOF.")
            return np.empty((0, nlay_uzf, self.nrows, self.ncols))

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
            TypeError: If inputs are not of expected types (NumPy arrays, float).
        """
        # This method is identical to the one in HydroModelOutput base class.
        # It's included here if this class might be used standalone or if specific
        # MODFLOW context for saturation calculation becomes necessary later.
        # For now, it could delegate to super().calculate_saturation if part of a class hierarchy.

        if not isinstance(water_content, np.ndarray):
            raise TypeError("water_content must be a NumPy array.")
        if not isinstance(porosity, (int, float, np.ndarray)): # Allow int as scalar porosity
            raise TypeError("porosity must be a float, int, or NumPy array.")

        # Warning for non-positive porosity values
        if isinstance(porosity, np.ndarray) and np.any(porosity <= 0):
            print("Warning: Porosity array contains zero or negative values. Saturation calculation may result in NaNs or Infs.")
        elif isinstance(porosity, (int, float)) and porosity <= 0:
            print("Warning: Scalar porosity is zero or negative. Saturation calculation may result in NaNs or Infs.")

        saturation_result: np.ndarray
        if isinstance(porosity, (int, float)):
            # Ensure float division, handle porosity = 0 to avoid runtime warning for 0/0 or x/0 if possible
            if float(porosity) == 0.0:
                # If porosity is zero, saturation is undefined (NaN) unless water content is also zero (then 0).
                # np.divide handles 0/0 as nan, x/0 as inf correctly by default with warnings.
                # Let result be NaN/inf and then clip.
                saturation_result = np.divide(water_content, float(porosity))
            else:
                saturation_result = water_content / float(porosity)
        else: # porosity is a NumPy array
            if porosity.ndim != water_content.ndim:
                # Attempt to broadcast if porosity is static (e.g., 3D) and water_content is timed (e.g., 4D)
                if porosity.ndim == water_content.ndim - 1 and water_content.shape[1:] == porosity.shape:
                    # Add time axis to porosity for broadcasting: (1, nlay, nrow, ncol)
                    porosity_expanded = porosity[np.newaxis, ...]
                    saturation_result = np.divide(water_content, porosity_expanded) # Handles porosity_expanded possibly containing 0
                else:
                    raise ValueError(f"Porosity array dimensions ({porosity.ndim}) are not directly compatible "
                                     f"with water_content dimensions ({water_content.ndim}) for broadcasting. "
                                     f"WC shape: {water_content.shape}, Porosity shape: {porosity.shape}")
            else: # Dimensions are the same
                 # Element-wise division, handles cases where porosity might have zeros.
                saturation_result = np.divide(water_content, porosity)
        
        # Clip saturation to the physical range [0, 1].
        # np.nan_to_num can be used if NaNs/Infs from division by zero need to be specific values before clipping.
        # For example, np.nan_to_num(saturation_result, nan=0.0, posinf=1.0, neginf=0.0) could be an option.
        # However, simple clipping is often sufficient if subsequent use handles NaNs from 0/0 correctly.
        # If porosity is 0 and WC > 0, result is inf, clipped to 1. This might mask issues.
        # If porosity is 0 and WC = 0, result is nan, clipped to 0. (np.divide(0,0)=nan)
        saturation_final = np.clip(saturation_result, 0.0, 1.0)
        
        return saturation_final
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Reads the 'WaterContent' file to extract header information for each timestep.

        Returns:
            List[Tuple[int, int, float, float]]: A list of tuples, where each tuple
                                                 contains (kstp, kper, pertim, totim)
                                                 for a timestep:
                                                 - kstp (int): Timestep number within the stress period.
                                                 - kper (int): Stress period number.
                                                 - pertim (float): Time within the current stress period.
                                                 - totim (float): Total simulation time.
        """
        # This method is identical to the one in the other MODFLOWWaterContent class.
        # Assuming it's correctly implemented there, it's replicated here.
        # A default nlay_uzf might be needed for skipping if not stored or passed.
        default_nlay_uzf_for_skip = 3 # Must match how data blocks are sized.
                                     # More robust: use maxbound from header to skip.
        
        wc_file_path = os.path.join(self.sim_ws, "WaterContent")
        if not os.path.exists(wc_file_path):
            print(f"Warning: 'WaterContent' file not found at {wc_file_path}. Cannot get timestep info.")
            return []

        timestep_info_list: List[Tuple[int, int, float, float]] = []
        header_dtype = np.dtype([
            ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
            ("text", "S16"), ("maxbound", "<i4"), ("aux1", "<i4"), ("aux2", "<i4"),
        ])
        # Assuming data points are float64 (8 bytes) for skipping.
        data_point_itemsize = np.dtype("<f8").itemsize

        try:
            with open(wc_file_path, "rb") as file:
                while True:
                    try:
                        header_data = binaryread(file, header_dtype) # Read one header record

                        kstp = int(header_data['kstp'])
                        kper = int(header_data['kper'])
                        pertim = float(header_data['pertim'])
                        totim = float(header_data['totim'])
                        maxbound_in_header = int(header_data['maxbound']) # Number of data points in the following block

                        timestep_info_list.append((kstp, kper, pertim, totim))

                        # Skip the data block using maxbound_in_header for accuracy
                        bytes_to_skip = maxbound_in_header * data_point_itemsize
                        file.seek(bytes_to_skip, os.SEEK_CUR)

                    except EOFError:
                        break # End of file reached cleanly
                    except Exception as e:
                        print(f"Error reading timestep info or skipping data in 'WaterContent': {e}")
                        break
        except Exception as e:
            print(f"Failed to open or process 'WaterContent' for timestep info: {e}")
            return []

        return timestep_info_list
