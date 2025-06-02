"""
Module for processing MODFLOW model outputs.
"""
import os
import numpy as np
from typing import Tuple, Optional, Union, List, Dict

from .base import HydroModelOutput


def binaryread(file, vartype, shape=(1,), charlen=16):
    """
    Uses numpy to read from binary file. This was found to be faster than the
    struct approach and is used as the default.

    Args:
        file: Open file object in binary read mode
        vartype: Variable type to read
        shape: Shape of the data to read (default: (1,))
        charlen: Length of character strings (default: 16)

    Returns:
        The read data
    """
    # Read a string variable of specified character length
    if vartype == str:
        # Read 'charlen' bytes and decode as string (assuming default encoding, e.g., ASCII or Latin-1 for typical MODFLOW text)
        # SUGGESTION: Specify encoding if known, e.g., result = file.read(charlen).decode('ascii', errors='replace')
        result = file.read(charlen * 1) # Reads charlen bytes
    else:
        # For numerical types, calculate the total number of values to read based on the given shape.
        nval = np.prod(shape) # Total number of elements
        # Read 'nval' items of 'vartype' from the binary file directly into a NumPy array.
        result = np.fromfile(file, vartype, nval)

        # Post-processing the result
        if nval == 1:
            # If only one value was read, it's returned as is (potentially still in an array of 1 element).
            # The original code had `result = result # [0]`. If a scalar is desired, `result = result[0]` would be needed.
            # Keeping as is, as it might be intentional for consistency if other parts expect an array.
            result = result
        else:
            # If multiple values were read, reshape the flat array to the specified 'shape'.
            result = np.reshape(result, shape)
    return result


class MODFLOWWaterContent(HydroModelOutput):
    """Class for processing water content data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.
        
        Args:
            model_directory: Path to simulation workspace
            idomain: Domain array indicating active cells
        """
        super().__init__(model_directory)
        self.idomain = idomain # Active model domain (1 for active, 0 for inactive)
        self.nrows, self.ncols = idomain.shape # Number of rows and columns from idomain shape
        
        # Build a reverse lookup dictionary for UZF cell numbers (iuzno) to (row, col) indices.
        # This dictionary maps a linear UZF cell index to its 2D grid coordinates.
        # The original code implies this is for the first layer or a 2D context of UZF cells.
        # UZF (Unsaturated-Zone Flow) package cells are typically numbered sequentially for active cells.
        self.iuzno_dict_rev = {} # Dictionary to store mapping from iuzno to (row, col)
        iuzno = 0 # Initialize UZF cell counter
        for i in range(self.nrows): # Iterate over rows
            for j in range(self.ncols): # Iterate over columns
                if idomain[i, j] != 0: # Check if the cell is active based on idomain
                    self.iuzno_dict_rev[iuzno] = (i, j) # Map the sequential UZF index to (row, col)
                    iuzno += 1 # Increment UZF cell counter
        
        # Store the total number of active UZF cells (in a single conceptual layer or 2D footprint).
        self.nuzfcells = len(self.iuzno_dict_rev)
    
    def load_timestep(self, timestep_idx: int, nlay: int = 3) -> np.ndarray:
        """
        Load water content for a specific timestep.
        
        Args:
            timestep_idx: Index of the timestep to load
            nlay: Number of layers in the model
            
        Returns:
            Water content array with shape (nlay, nrows, ncols)
        """
        # This method loads a single timestep by calling load_time_range for a range of one.
        # It then returns the first (and only) timestep from the result.
        return self.load_time_range(timestep_idx, timestep_idx + 1, nlay)[0]
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, 
                      nlay: int = 3) -> np.ndarray:
        """
        Load water content for a range of timesteps from MODFLOW's binary "WaterContent" file.
        
        Args:
            start_idx: Starting timestep index (0-based) (default: 0).
            end_idx: Ending timestep index (0-based, exclusive). If None, reads to end of file (default: None).
            nlay: Number of layers in the UZF model (default: 3). This implies a conceptual layering within UZF package,
                  not necessarily number of MODFLOW model layers.
            
        Returns:
            Water content array with shape (num_timesteps_read, nlay, nrows, ncols).
        """
        # Calculate the total number of UZF cell data points to read per timestep.
        # This is the number of active 2D cells multiplied by the number of UZF layers.
        # SUGGESTION: `nlay` here seems to refer to UZF conceptual layers. Clarify if it's related to MODFLOW model layers.
        nuzfcells_per_timestep = self.nuzfcells * nlay
        
        # Construct the full path to the "WaterContent" binary file.
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb") # Open the file in binary read mode.
        
        WC_tot = [] # List to store water content arrays for each read timestep.
        
        # --- Skip to the starting timestep ---
        # This loop reads and discards headers and data for timesteps before start_idx.
        for skip_count in range(start_idx): # Use skip_count for clarity in print messages
            try:
                # Define the data types and names for the header structure of each timestep.
                # These typically include timestep number, stress period, simulation times, and text description.
                # '<i4' is little-endian 4-byte integer, '<f8' is little-endian 8-byte float (double).
                # 'S16' is a 16-byte string.
                header_vartype = [
                    ("kstp", "<i4"),      # Timestep number within the stress period
                    ("kper", "<i4"),      # Stress period number
                    ("pertim", "<f8"),    # Time in the current stress period
                    ("totim", "<f8"),     # Total simulation time
                    ("text", "S16"),     # Text description (e.g., "VOLUMETRIC WC")
                    ("maxbound", "<i4"),  # Number of cells for which data is written (should match nuzfcells_per_timestep)
                    ("1", "<i4"),         # Often a constant or layer index, meaning can vary. Here named "1".
                    ("11", "<i4"),        # Often a constant, here named "11".
                ]
                # Read the header for the current timestep to be skipped.
                header_content = binaryread(file, header_vartype)
                if not header_content.size > 0: # Check if read was successful
                    raise EOFError(f"EOF encountered while trying to read header during skip at timestep index {skip_count}")

                # Skip the actual water content data for this timestep.
                # Data is assumed to be a series of float64 values.
                data_vartype = [("data", "<f8")] # Define type for reading data points
                for _ in range(nuzfcells_per_timestep):
                    data_point = binaryread(file, data_vartype) # Read and discard each data point
                    if not data_point.size > 0: # Check if read was successful
                        raise EOFError(f"EOF encountered while skipping data at timestep index {skip_count}")
            except EOFError as e_eof: # Catch EOF specifically
                print(f"EOF reached while skipping to start_idx {start_idx}. Processed {skip_count} timesteps before EOF: {e_eof}")
                file.close()
                return np.array(WC_tot) # Return whatever was collected (likely empty)
            except Exception as e:
                # If any other error occurs (e.g., file format issue), print message and return.
                print(f"Error encountered while skipping to start_idx {start_idx} at skip index {skip_count}: {e}")
                file.close()
                return np.array(WC_tot)
        
        # --- Read the desired range of timesteps ---
        timesteps_read_count = 0
        while True:
            # If end_idx is specified, stop if the desired number of timesteps has been read.
            if end_idx is not None and timesteps_read_count >= (end_idx - start_idx):
                break
                
            try:
                # Read header information for the current timestep.
                header_vartype = [ # Same structure as defined before for skipping
                    ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
                    ("text", "S16"), ("maxbound", "<i4"), ("1", "<i4"), ("11", "<i4"),
                ]
                header_data = binaryread(file, header_vartype) # Read the header
                if not header_data.size > 0: # Check if header read was successful (binaryread might return empty on EOF)
                    # This is a common way to detect EOF if no more headers can be read.
                    # print(f"EOF suspected when trying to read header for effective timestep {start_idx + timesteps_read_count}.") # Debug line
                    break

                # Initialize a NumPy array for the current timestep's water content data.
                # Array shape is (nlay, nrows, ncols), initialized with NaNs.
                WC_arr_timestep = np.full((nlay, self.nrows, self.ncols), np.nan)
                
                # Define the data type for reading water content values.
                data_vartype = [("data", "<f8")]
                
                # Read data for each UZF layer and each active UZF cell.
                # The data in the file is typically ordered by cell number (iuzno), then by UZF layer,
                # or all cells for layer 1, then all cells for layer 2, etc.
                # The loop structure `for k_layer_idx in range(nlay): for iuzno_idx in range(self.nuzfcells):`
                # implies data is grouped by layer first, then by iuzno within that layer.
                for k_layer_idx in range(nlay): # Iterate over UZF layers
                    for iuzno_idx in range(self.nuzfcells): # Iterate over active 2D cells
                        # Get the (row, col) for the current UZF cell index (iuzno_idx).
                        row_idx, col_idx = self.iuzno_dict_rev[iuzno_idx]

                        # Read the water content value.
                        wc_value_array = binaryread(file, data_vartype)
                        if wc_value_array.size == 1: # Ensure a single value was read
                             WC_arr_timestep[k_layer_idx, row_idx, col_idx] = wc_value_array[0] # Extract scalar
                        elif wc_value_array.size == 0: # Handle unexpected EOF during data read
                            raise EOFError(f"EOF encountered while reading data for layer {k_layer_idx}, cell {iuzno_idx}")
                        else:
                            # This case should ideally not happen if reading one value at a time per cell.
                            print(f"Warning: Expected single value for WC, got array: {wc_value_array} at layer {k_layer_idx}, cell {iuzno_idx}")
                            WC_arr_timestep[k_layer_idx, row_idx, col_idx] = wc_value_array.flat[0] # Take first element if array, or handle error

                WC_tot.append(WC_arr_timestep) # Add the populated array for this timestep to the list.
                timesteps_read_count += 1 # Increment counter for timesteps read.
                
            except EOFError: # Specifically catch EOFError if np.fromfile raises it or if explicitly raised above
                # print(f"Reached end of file while reading data for effective timestep {start_idx + timesteps_read_count}.") # Debug line
                break
            except Exception as e:
                # Catch any other errors during reading (e.g., file corruption, unexpected format).
                print(f"Error reading data at effective timestep {start_idx + timesteps_read_count}: {str(e)}")
                break
        
        file.close() # Close the binary file.
        
        # Convert the list of timestep arrays into a single NumPy array.
        # Expected shape: (timesteps_read_count, nlay, nrows, ncols).
        return np.array(WC_tot)
    
    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Get information about each timestep in the WaterContent file.
        
        Returns:
            List of tuples (kstp, kper, pertim, totim) for each timestep
        """
        # Construct the full path to the "WaterContent" binary file.
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb") # Open the file in binary read mode.
        
        timestep_info_list = [] # List to store header information for each timestep.
        # Assuming nlay=3 by default for UZF as in load_time_range and docs.
        # This is used to calculate how many data records to skip after reading each header.
        # SUGGESTION: `nlay` should ideally be an argument or class attribute if it can vary for UZF.
        # Hardcoding to 3 here might be inconsistent if other calls use a different nlay.
        # For get_timestep_info, if nlay varies per record (unlikely for standard files but possible if file structure is complex),
        # this would need to parse 'maxbound' or other header fields to determine records to skip.
        # Assuming 'maxbound' in the header refers to total records for *that* timestep (nuzfcells * nlay_for_that_record).
        # For simplicity, using a fixed nlay for skipping, as in load_time_range.
        # A more robust approach might be to read `maxbound` from header and use that to skip.
        nuzf_layers_for_skip = 3 # Default number of UZF layers assumed for calculating data record size.
        num_data_records_to_skip_per_timestep = self.nuzfcells * nuzf_layers_for_skip
        
        while True:
            try:
                # Define the header structure (same as in load_time_range).
                header_vartype = [
                    ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
                    ("text", "S16"), ("maxbound", "<i4"), ("1", "<i4"), ("11", "<i4"),
                ]
                header_data = binaryread(file, header_vartype) # Read the header.

                # Check if the header was read successfully (e.g. not EOF)
                if not header_data.size > 0 or not hasattr(header_data, '__getitem__') or len(header_data) == 0 or len(header_data[0]) < 4 :
                    # This condition implies header_data is not structured as expected or is empty, likely EOF.
                    break

                # Extract specific information from the header.
                # header_data is typically a structured array (if numpy.fromfile was used with a dtype)
                # or a simple array if binaryread processes it into a plain ndarray of basic types.
                # Given the vartype, `binaryread` with np.fromfile will produce a structured array if vartype is a dtype.
                # If `binaryread` returns a list/tuple of tuples for structured types (as original code might imply by header[0][0]):
                # Accessing elements assuming structured array or similar tuple structure:
                kstp = header_data['kstp'][0] if isinstance(header_data, np.ndarray) and header_data.dtype.names else header_data[0][0]
                kper = header_data['kper'][0] if isinstance(header_data, np.ndarray) and header_data.dtype.names else header_data[0][1]
                pertim = header_data['pertim'][0] if isinstance(header_data, np.ndarray) and header_data.dtype.names else header_data[0][2]
                totim = header_data['totim'][0] if isinstance(header_data, np.ndarray) and header_data.dtype.names else header_data[0][3]
                # maxbound_from_header = header_data['maxbound'][0] if ... else header_data[0][5] # Example if needed for dynamic skip
                # num_data_records_to_skip_per_timestep = maxbound_from_header # If using dynamic skip

                timestep_info_list.append((kstp, kper, pertim, totim)) # Store the extracted info.
                
                # Skip the actual water content data for this timestep.
                data_vartype = [("data", "<f8")] # Type of data records to skip
                for _ in range(num_data_records_to_skip_per_timestep): # Use the determined number of records
                    data_read_result = binaryread(file, data_vartype) # Read and discard.
                    if not data_read_result.size > 0 : # Check if read was successful
                        raise EOFError("EOF during data skipping after a successful header read.")

            except EOFError: # Catch EOF explicitly if raised by checks
                break # Expected way to end the loop when file end is reached.
            except Exception as e:
                # Catch any other errors (e.g., file ends unexpectedly mid-record, format issues).
                # print(f"Error while reading timestep info: {str(e)}") # Optional: for debugging
                break
        
        file.close() # Close the file.
        return timestep_info


class MODFLOWPorosity(HydroModelOutput):
    """Class for processing porosity data from MODFLOW simulations."""
    
    def __init__(self, model_directory: str, model_name: str):
        """
        Initialize MODFLOWPorosity processor.
        
        Args:
            model_directory: Path to simulation workspace
            model_name: Name of the MODFLOW model
        """
        super().__init__(model_directory)
        self.model_directory = model_directory # Path to the simulation workspace
        self.model_name = model_name           # Name of the MODFLOW model (e.g., used for .nam file)
        self.nlay = 1 # Initialize default model dimensions, will be updated when model is loaded
        self.nrow = 1
        self.ncol = 1
        
        # Check for flopy availability at initialization.
        try:
            import flopy # Attempt to import flopy
            self.flopy_available = True
        except ImportError:
            self.flopy_available = False
            # Raise an error immediately if flopy is not available, as it's essential for this class.
            raise ImportError("flopy is required to use MODFLOWPorosity. Please install flopy.")
    
    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from MODFLOW model (supports both MODFLOW 6 and earlier versions).
        
        Returns:
            3D array of porosity values (nlay, nrow, ncol)
        """
        if not self.flopy_available:
            # This check is somewhat redundant if constructor already raised error, but good for safety.
            raise ImportError("flopy is required to load MODFLOW porosity data.")
            
        try:
            import flopy # Flopy should be available due to the check in __init__
            
            # --- Determine MODFLOW version (MF6 or earlier) ---
            # Check for common MODFLOW 6 simulation files.
            # SUGGESTION: A more robust way might be to try loading as MF6 and fall back,
            # or have the user specify the version if ambiguous.
            mf6_sim_file = os.path.join(self.model_directory, "mfsim.nam")
            # Some MF6 setups might use modelname.sim as the primary simulation file name.
            mf6_model_sim_file = os.path.join(self.model_directory, f"{self.model_name}.sim") # Use f-string
            is_mf6 = os.path.exists(mf6_sim_file) or os.path.exists(mf6_model_sim_file)
            
            if is_mf6:
                # --- MODFLOW 6 specific loading logic ---
                try:
                    # Load the MODFLOW 6 simulation.
                    # sim_name for MFSimulation.load should be the name of the simulation defined in mfsim.nam,
                    # or can be None if there's only one simulation.
                    # Using self.model_name if it's intended to be the simulation name, or adjust as needed.
                    # If mfsim.nam exists, it's safer to parse it or use a known sim name.
                    # For now, assuming self.model_name might be the GWF model name, not necessarily sim name.
                    # Trying with sim_name=None first, then specific if needed and known.
                    sim = flopy.mf6.MFSimulation.load(
                        sim_name=None, # Try to auto-detect simulation name from mfsim.nam
                        sim_ws=self.model_directory,
                        exe_name="mf6", # Not strictly needed for loading data.
                    )
                    
                    # Get the groundwater flow (GWF) model from the simulation.
                    # The GWF model name might be self.model_name or needs to be discovered from the simulation object.
                    gwf_model = None
                    if self.model_name in sim.model_names:
                        gwf_model = sim.get_model(self.model_name)
                    elif sim.gwf_models: # If there are GWF models, try taking the first one
                        gwf_model = sim.gwf_models[0]
                        if len(sim.gwf_models) > 1:
                            print(f"Warning: Multiple GWF models found. Using first one: {gwf_model.name}. Specify model_name if this is not correct.")
                    else:
                        raise ValueError("No GWF model found in the MODFLOW 6 simulation.")

                    # Get model dimensions from the Discretization (DIS) package.
                    dis = gwf_model.dis # Access DIS package directly from the model object
                    self.nlay = dis.nlay.data # Number of layers
                    self.nrow = dis.nrow.data # Number of rows
                    self.ncol = dis.ncol.data # Number of columns
                    
                    # Try to get porosity (Specific Yield, SY) from the Storage (STO) package.
                    sto = gwf_model.sto # Access STO package
                    if sto is not None and hasattr(sto, 'sy') and sto.sy.data is not None:
                        porosity_data = sto.sy.array
                        return np.asarray(porosity_data) # Ensure it's a standard ndarray
                    elif sto is not None and hasattr(sto, 'ss') and sto.ss.data is not None:
                        # Fallback to Specific Storage if SY is not available, with a warning.
                        print("WARNING: Specific Yield (SY) not found in MODFLOW 6 STO package. Using Specific Storage (SS) as a substitute. This may not be appropriate for porosity.")
                        ss_data = sto.ss.array
                        return np.asarray(ss_data)
                    else:
                        print("Warning: Neither SY nor SS found in MODFLOW 6 STO package.")
                        # Fall through to default value at the end of the function.
                        
                except Exception as e:
                    print(f"Error loading porosity from MODFLOW 6 model '{self.model_name}': {str(e)}")
                    # Fall through to default value if MF6 loading fails.
                    
            # --- Legacy MODFLOW (MODFLOW-2000, MODFLOW-2005) specific loading logic ---
            if not is_mf6: # Executed if not identified as MF6
                try:
                    # Load the legacy MODFLOW model name file.
                    model = flopy.modflow.Modflow.load( # Use the general Modflow class for older versions
                        f"{self.model_name}.nam", # Construct path to .nam file using f-string
                        model_ws=self.model_directory,
                        load_only=["UPW", "LPF", "BCF6", "DIS"],  # Relevant packages for porosity (SY/SS) & dimensions
                        check=False # Disable model check for faster loading
                    )
                    
                    # Get model dimensions from the loaded model object.
                    self.nlay = model.nlay
                    self.nrow = model.nrow
                    self.ncol = model.ncol
                    
                    # Attempt to get Specific Yield (SY)
                    porosity_array = None
                    if hasattr(model, 'upw') and model.upw is not None and hasattr(model.upw, 'sy') and model.upw.sy is not None:
                        porosity_array = model.upw.sy.array
                    elif hasattr(model, 'lpf') and model.lpf is not None and hasattr(model.lpf, 'sy') and model.lpf.sy is not None:
                        porosity_array = model.lpf.sy.array
                    elif hasattr(model, 'bcf') and model.bcf is not None and hasattr(model.bcf, 'sy') and model.bcf.sy is not None: # For BCF6 it's .sy
                        porosity_array = model.bcf.sy.array
                    
                    if porosity_array is not None:
                        return np.asarray(porosity_array)

                    # If SY is not found, try Specific Storage (SS) as a fallback
                    ss_array = None
                    if hasattr(model, 'upw') and model.upw is not None and hasattr(model.upw, 'ss') and model.upw.ss is not None:
                        ss_array = model.upw.ss.array
                    elif hasattr(model, 'lpf') and model.lpf is not None and hasattr(model.lpf, 'ss') and model.lpf.ss is not None:
                        ss_array = model.lpf.ss.array
                    elif hasattr(model, 'bcf') and model.bcf is not None: # BCF stores SS as sf2 (confined)
                        if hasattr(model.bcf, 'sf2') and model.bcf.sf2 is not None: # sf2 is specific storage for confined layers
                           ss_array = model.bcf.sf2.array
                           # Note: sf1 in BCF is SY for unconfined layers. If SY check above missed it for BCF, this is another place.
                           # However, the primary SY check should cover BCF's SY if flopy structures it under .sy attribute.

                    if ss_array is not None:
                        print("WARNING: Specific Yield (SY) not found. Using Specific Storage (SS) as a substitute for porosity. This may not be appropriate.")
                        return np.asarray(ss_array)
                            
                except Exception as e:
                    # Error during legacy model loading.
                    print(f"Error loading porosity from legacy MODFLOW model '{self.model_name}': {str(e)}")
                    # Fall through to default value if legacy loading also fails.
            
            # --- Default Fallback ---
            # This part is reached if:
            #   - Not MF6, and legacy loading failed or found no data.
            #   - Is MF6, but MF6 loading failed or found no data.
            # Dimensions (nlay, nrow, ncol) should be set if DIS package was read in any attempt.
            # If still default (1,1,1), the default array will be small.
            print(f"WARNING: Porosity data (SY or SS) could not be loaded from model '{self.model_name}'. "
                  f"Using a default uniform porosity of 0.3 for current dimensions ({self.nlay}, {self.nrow}, {self.ncol}).")
            return np.ones((self.nlay, self.nrow, self.ncol)) * 0.3 # Default porosity array
                
        except Exception as e:
            # Catch-all for any other unexpected errors during the process (e.g., flopy import itself failing despite check).
            raise ValueError(f"An unexpected error occurred while trying to load porosity data: {str(e)}")
    
    # Implement required abstract methods
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity for a specific timestep.
        Note: For MODFLOW, porosity is typically constant over time,
        so this returns the same array regardless of timestep.
        
        Args:
            timestep_idx: Index of the timestep (unused)
            
        Returns:
            3D array of porosity values (nlay, nrow, ncol)
        """
        # For MODFLOW, porosity is typically static (defined once in packages like LPF, UPW, STO, or BCF).
        # Therefore, loading for a specific timestep just means loading the single defined porosity array.
        return self.load_porosity()
    
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load porosity for a range of timesteps.
        Since porosity is typically constant, this returns a stack of identical arrays.
        
        Args:
            start_idx: Starting timestep index (unused)
            end_idx: Ending timestep index (unused)
            
        Returns:
            4D array of porosity values (num_timesteps, nlay, nrow, ncol), where all num_timesteps slices are identical.
        """
        porosity_3d = self.load_porosity() # Load the single 3D porosity array.

        # Determine the number of "timesteps" to replicate this array for.
        # If end_idx is None, it implies a single representation or perhaps based on model's total timesteps.
        # Here, it defaults to 1 if end_idx is None.
        # SUGGESTION: Clarify behavior if end_idx is None. Should it try to get actual model num_timesteps?
        # For now, it creates 'nt' copies.
        num_timesteps_to_replicate = 1
        if end_idx is not None:
            if end_idx > start_idx:
                num_timesteps_to_replicate = end_idx - start_idx
            else:
                # Handle invalid range, perhaps return single or raise error.
                # Defaulting to 1 for now if range is non-positive.
                print(f"Warning: end_idx ({end_idx}) not greater than start_idx ({start_idx}). Returning porosity for 1 effective timestep.")
                num_timesteps_to_replicate = 1
        
        # Add a new axis for time at the beginning of the 3D porosity array.
        porosity_4d_single_timestep = porosity_3d[np.newaxis, :, :, :]
        
        # Replicate the 3D porosity array along the new time axis.
        # np.tile repeats the array. (num_timesteps_to_replicate, 1, 1, 1) means repeat N times along axis 0, 1 time along others.
        return np.tile(porosity_4d_single_timestep, (num_timesteps_to_replicate, 1, 1, 1))
    
    def get_timestep_info(self) -> List[Tuple]:
        """
        Get information about each timestep in the model.
        Returns a minimal placeholder since porosity doesn't vary with time.
        
        Returns:
            List with a single tuple representing placeholder timestep information,
            as porosity is constant. (stress_period_idx, time_step_in_period_idx, total_time)
        """
        # Since porosity is considered static for MODFLOW models,
        # this method returns a placeholder list with a single entry.
        # This signifies that the porosity data is not time-dependent in the same way
        # other outputs like water content or head might be.
        # The values (0, 0, 0.0) are arbitrary placeholders.
        return [(0, 0, 0.0)]  # Represents (example: stress_period=0, timestep_in_period=0, time=0.0)