"""
Module for processing MODFLOW model outputs.
"""
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from PyHydroGeophysX._internal.deprecations import renamed_keywords, warn_legacy_path
from .base import HydroModelOutput

# In 0.3.0 this module held its own copy of these readers, which named three
# keywords differently (file_obj, sim_ws and nlay_uzf) and kept the folder as
# the attribute ``sim_ws``. Those names still work, with a warning, until 0.6.0.


# ---------------------------------------------------------------------------
# binaryread
# ---------------------------------------------------------------------------
@renamed_keywords("binaryread", file_obj="file")
def binaryread(
    file: Any,
    vartype: Union[type, List[Tuple[str, str]]],
    shape: Tuple[int, ...] = (1,),
    charlen: int = 16,
) -> Union[bytes, np.ndarray, np.void]:
    """
    Read one item from an open MODFLOW binary output file.

    Args:
        file: Open file object in binary read mode (``file_obj`` before 0.5.0).
            Any object with ``read`` works, an ``io.BytesIO`` included.
        vartype: ``str``, a NumPy dtype, or a structured dtype given as a list
            of ``(name, format)`` pairs.
        shape: Output shape for a plain dtype (default: (1,)).
        charlen: Number of bytes read for ``str`` (default: 16).

    Returns:
        ``bytes`` for ``str``; for a structured dtype, the one record read, a
        ``numpy.void`` whose fields are read by name; otherwise an array of
        ``shape``. These are the types 0.3.0 returned.

    Raises:
        EOFError: When the file ends before the whole item is read.

    Examples
    --------
    >>> import io
    >>> header = [("kstp", "<i4"), ("kper", "<i4")]
    >>> record = binaryread(io.BytesIO(np.array([(3, 7)], dtype=header).tobytes()), header)
    >>> int(record["kstp"]), int(record["kper"])
    (3, 7)
    >>> binaryread(io.BytesIO(np.arange(4.0).tobytes()), np.float64, shape=(2, 2)).tolist()
    [[0.0, 1.0], [2.0, 3.0]]
    """
    if vartype == str:
        return file.read(charlen)
    structured = isinstance(vartype, list)
    dtype = np.dtype(vartype)
    count = 1 if structured else int(np.prod(shape))
    # read() and frombuffer rather than fromfile, which needs a real file
    # descriptor; a short read is the end of the file, not an empty result.
    raw = file.read(dtype.itemsize * count)
    if len(raw) < dtype.itemsize * count:
        raise EOFError(f"Expected {count} value(s) of {dtype}, but the file ended.")
    values = np.frombuffer(raw, dtype=dtype, count=count).copy()
    return values[0] if structured else values.reshape(shape)


# ---------------------------------------------------------------------------
# MODFLOWWater Content
# ---------------------------------------------------------------------------
class MODFLOWWaterContent(HydroModelOutput):
    """Class for processing water content data from MODFLOW simulations."""

    @renamed_keywords("MODFLOWWaterContent", sim_ws="model_directory")
    def __init__(self, model_directory: str, idomain: np.ndarray):
        """
        Initialize MODFLOWWaterContent processor.

        Args:
            model_directory: Path to simulation workspace
            idomain: Two-dimensional (rows, columns) domain array; nonzero cells
                receive consecutive UZF indices in row-major order.
        """
        super().__init__(model_directory)
        self.idomain = idomain
        self.nrows, self.ncols = idomain.shape

        # Map sequential UZF indices back to the supplied 2D active-cell grid.
        self.iuzno_dict_rev = {}
        iuzno = 0
        for i in range(self.nrows):
            for j in range(self.ncols):
                if idomain[i, j] != 0:
                    self.iuzno_dict_rev[iuzno] = (i, j)
                    iuzno += 1

        # Store number of UZ flow cells
        self.nuzfcells = len(self.iuzno_dict_rev)

    @property
    def sim_ws(self) -> str:
        """The simulation folder under its 0.3.0 name; use ``model_directory``."""
        warn_legacy_path("MODFLOWWaterContent.sim_ws", "MODFLOWWaterContent.model_directory")
        return self.model_directory

    @renamed_keywords("MODFLOWWaterContent.load_timestep", nlay_uzf="nlay")
    def load_timestep(self, timestep_idx: int, nlay: int = 3) -> np.ndarray:
        """
        Load water content for a specific timestep.

        Args:
            timestep_idx: Index of the timestep to load
            nlay: Number of layers in the model

        Returns:
            Water content array with shape (nlay, nrows, ncols)
        """
        return self.load_time_range(timestep_idx, timestep_idx + 1, nlay)[0]

    #: One record header of the UZF WaterContent file. ``maxbound`` times the
    #: next field (1) is the number of values that follow: UZF cells x layers.
    _RECORD_HEADER = np.dtype([
        ("kstp", "<i4"), ("kper", "<i4"), ("pertim", "<f8"), ("totim", "<f8"),
        ("text", "S16"), ("maxbound", "<i4"), ("1", "<i4"), ("11", "<i4"),
    ])

    @renamed_keywords("MODFLOWWaterContent.load_time_range", nlay_uzf="nlay")
    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None,
                      nlay: int = 3) -> np.ndarray:
        """
        Load water content for a range of timesteps.

        Args:
            start_idx: Starting timestep index (default: 0)
            end_idx: Ending timestep index (exclusive, default: None loads all)
            nlay: Number of layers in the model (default: 3). Each record's
                header states how many values it holds; a count other than
                nlay x the UZF cells raises ValueError instead of being read
                across the record boundary.

        Returns:
            Water content array with shape (timesteps, nlay, nrows, ncols)
        """
        nlay = int(nlay)
        expected = self.nuzfcells * nlay
        # UZF cell n of every layer sits at (rows[n], cols[n]) of the idomain grid.
        rows = np.array([self.iuzno_dict_rev[n][0] for n in range(self.nuzfcells)], dtype=int)
        cols = np.array([self.iuzno_dict_rev[n][1] for n in range(self.nuzfcells)], dtype=int)

        fpth = os.path.join(self.model_directory, "WaterContent")
        WC_tot = []
        index = 0
        with open(fpth, "rb") as file:
            while end_idx is None or index < end_idx:
                header = np.fromfile(file, self._RECORD_HEADER, 1)
                if header.size != 1:
                    break  # end of file
                n_values = int(header["maxbound"][0]) * int(header["1"][0])
                if n_values != expected:
                    layers = (f"; the file has {n_values // self.nuzfcells} layers"
                              if self.nuzfcells and n_values % self.nuzfcells == 0 else "")
                    raise ValueError(
                        f"WaterContent record {index} holds {n_values} values, but "
                        f"nlay={nlay} with {self.nuzfcells} UZF cells per layer needs "
                        f"{expected}{layers}. Pass the model's layer count as nlay.")
                if index < start_idx:
                    file.seek(8 * n_values, os.SEEK_CUR)
                else:
                    # A whole record in one read. Reading it a value at a time
                    # (one fromfile call per cell and layer) was the cost of the
                    # load; the values are the same float64 either way.
                    values = np.fromfile(file, "<f8", n_values)
                    if values.size != n_values:
                        print(f"Reached end of file or error at timestep {index - start_idx}: "
                              "WaterContent ended before the current timestep was complete.")
                        break
                    WC_arr = np.full((nlay, self.nrows, self.ncols), np.nan)
                    WC_arr[:, rows, cols] = values.reshape(nlay, self.nuzfcells)
                    WC_tot.append(WC_arr)
                index += 1

        return np.array(WC_tot)

    def get_timestep_info(self) -> List[Tuple[int, int, float, float]]:
        """
        Get information about each timestep in the WaterContent file.

        Returns:
            List of tuples (kstp, kper, pertim, totim) for each timestep
        """
        # Open water content file
        fpth = os.path.join(self.model_directory, "WaterContent")
        file = open(fpth, "rb")

        timestep_info = []

        while True:
            try:
                # Read header information
                vartype = [
                    ("kstp", "<i4"),
                    ("kper", "<i4"),
                    ("pertim", "<f8"),
                    ("totim", "<f8"),
                    ("text", "S16"),
                    ("maxbound", "<i4"),
                    ("1", "<i4"),
                    ("11", "<i4"),
                ]
                header = binaryread(file, vartype)

                # Extract timestep info
                kstp = header["kstp"]
                kper = header["kper"]
                pertim = header["pertim"]
                totim = header["totim"]

                timestep_info.append((kstp, kper, pertim, totim))

                # Skip data for this timestep. The header states how many
                # values follow (ncol * nrow = UZF cells x layers, 1866 = 622 x 3
                # in examples/data/modflow). This used to assume
                # nuzfcells * 3, which lands mid-record, and then reads data as
                # headers, for a model with any other layer count.
                n_values = int(header["maxbound"]) * int(header["1"])
                if n_values <= 0:
                    break
                np.fromfile(file, dtype="<f8", count=n_values)

            except Exception:
                break

        file.close()
        return timestep_info


# ---------------------------------------------------------------------------
# MODFLOWPorosity
# ---------------------------------------------------------------------------
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
        self.model_directory = model_directory
        self.model_name = model_name
        self.nlay = 1
        self.nrow = 1
        self.ncol = 1

        try:
            import flopy
            self.flopy_available = True
        except ImportError:
            self.flopy_available = False
            raise ImportError("flopy is required to load MODFLOW porosity data. Please install flopy.")

    def load_porosity(self) -> np.ndarray:
        """
        Load porosity data from MODFLOW model (supports both MODFLOW 6 and earlier versions).

        Returns:
            3D array of porosity values (nlay, nrow, ncol)
        """
        if not self.flopy_available:
            raise ImportError("flopy is required to load MODFLOW porosity data.")

        try:
            import flopy

            # Check if this is a MODFLOW 6 model
            mf6_indicator_files = ["mfsim.nam", f"{self.model_name}.sim"]
            is_mf6 = any(os.path.exists(os.path.join(self.model_directory, f)) for f in mf6_indicator_files)

            if is_mf6:
                # MODFLOW 6 approach
                try:
                    # Load only DIS and STO packages to avoid failures from
                    # optional packages (e.g. UZF) whose files may be absent
                    sim = flopy.mf6.MFSimulation.load(
                        sim_name=self.model_name,
                        sim_ws=self.model_directory,
                        exe_name="mf6",
                        verbosity_level=0,
                        load_only=["dis", "sto"],
                        strict=False,
                    )

                    # Get the groundwater flow model
                    gwf = sim.get_model(self.model_name)

                    # Try to get dimensions from DIS package

                    dis = gwf.get_package("DIS")
                    self.nlay = int(dis.nlay.data)
                    self.nrow = int(dis.nrow.data)
                    self.ncol = int(dis.ncol.data)

                    # Try to get porosity from the STO (Storage) package

                    sto = gwf.get_package("STO")

                    return sto.sy.array

                except Exception as e:
                    print(f"Error loading MODFLOW 6 model: {str(e)}")

            else:
                # Legacy MODFLOW models (2005 or earlier)
                try:
                    # Load the model
                    model = flopy.modflow.Modflow.load(
                        f"{self.model_name}.nam",
                        model_ws=self.model_directory,
                        load_only=["UPW", "LPF", "DIS"],  # Load packages with porosity and dimensions
                        check=False
                    )

                    # Get dimensions
                    self.nlay = model.nlay
                    self.nrow = model.nrow
                    self.ncol = model.ncol

                    # Try to get porosity from UPW package first
                    if hasattr(model, 'upw') and model.upw is not None:
                        if hasattr(model.upw, 'sy'):
                            return model.upw.sy.array

                    # Then try LPF package
                    if hasattr(model, 'lpf') and model.lpf is not None:
                        if hasattr(model.lpf, 'sy'):
                            return model.lpf.sy.array

                    # If specific yield not found, try specific storage
                    if hasattr(model, 'upw') and model.upw is not None:
                        if hasattr(model.upw, 'ss'):
                            print("WARNING: Using specific storage as substitute for porosity")
                            return model.upw.ss.array

                    if hasattr(model, 'lpf') and model.lpf is not None:
                        if hasattr(model.lpf, 'ss'):
                            print("WARNING: Using specific storage as substitute for porosity")
                            return model.lpf.ss.array

                except Exception as e:
                    print(f"Error loading legacy MODFLOW model: {str(e)}")

            # If nothing found, use default value
            print("WARNING: No porosity data found in model. Using default value of 0.3")
            return np.ones((self.nlay, self.nrow, self.ncol)) * 0.3

        except Exception as e:
            raise ValueError(f"Error loading porosity data: {str(e)}")

    # Implement required abstract methods
    def load_timestep(self, timestep_idx: int, **kwargs) -> np.ndarray:
        """
        Load porosity for a specific timestep.
        Note: For MODFLOW, porosity is typically constant over time,
        so this returns the same array regardless of timestep.

        Args:
            timestep_idx: Index of the timestep (unused)

        Returns:
            3D array of porosity values
        """
        return self.load_porosity()

    def load_time_range(self, start_idx: int = 0, end_idx: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Load porosity for a range of timesteps.
        Since porosity is typically constant, this returns a stack of identical arrays.

        Args:
            start_idx: Starting timestep index (unused)
            end_idx: Ending timestep index (unused)

        Returns:
            4D array of porosity values (nt, nlay, nrow, ncol) where all timesteps are identical
        """
        porosity = self.load_porosity()

        # Determine number of timesteps
        nt = 1 if end_idx is None else (end_idx - start_idx)

        # Stack porosity array for each timestep
        return np.tile(porosity[np.newaxis, :, :, :], (nt, 1, 1, 1))

    def get_timestep_info(self) -> List[Tuple]:
        """
        Get information about each timestep in the model.
        Returns a minimal placeholder since porosity doesn't vary with time.

        Returns:
            List with single dummy timestep info
        """
        return [(0, 0, 0.0)]  # (stress_period, timestep, time)
