"""Read PFLOTRAN snapshot HDF5 output with Time: <value> <unit> groups."""
from pathlib import Path
import re

import numpy as np

from ._hdf5_output import HDF5ModelOutput, resolve_field, select_axes, xdmf_mixed_centroids


class PFLOTRANOutput(HDF5ModelOutput):
    """Read single-file or multiple-file snapshots in numerical time order.

    Select a run using ``run_name`` (matches ``run_name.h5`` and the numbered
    files ``run_name-NNN.h5`` of PFLOTRAN's multiple-file output, and no other
    run's files), or provide ``filename`` explicitly. Unstructured arrays and
    structured arrays retain their HDF5 shape and axes - ``(nx, ny, nz)`` for a
    structured grid, as PFLOTRAN writes it - and :meth:`output_cell_centers`
    gives the cell centres in their C-order flattening. Mixed time units and
    duplicate snapshot times are rejected to avoid silently joining unrelated
    runs or restarts.
    """

    aliases = {
        "saturation": ("Liquid_Saturation", "Liquid_Saturation [ ]", "Liquid_Saturation [-]"),
        "porosity": ("Porosity", "Porosity [ ]", "Porosity [-]"),
        "pressure": ("Liquid_Pressure", "Liquid_Pressure [Pa]"),
        "temperature": ("Temperature", "Temperature [C]"),
    }
    _time_pattern = re.compile(r"^Time:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?)\s+(\S+)\s*$")

    def __init__(self, model_directory, run_name=None, *, filename=None,
                 variable_map=None, cell_centers=None):
        super().__init__(model_directory, variable_map=variable_map, cell_centers=cell_centers)
        folder = Path(model_directory)
        if filename is not None and run_name is not None:
            raise ValueError("Choose filename or run_name, not both")
        if filename is not None:
            files = [folder / filename]
        elif run_name is not None:
            # Only the run's own numbered files: "run-*.h5" would also take in
            # another run named "run-hires", joined into one series unnoticed.
            numbered = re.compile(rf"{re.escape(run_name)}-\d+\.h5")
            files = sorted(path for path in folder.iterdir()
                           if path.is_file() and numbered.fullmatch(path.name))
            if (folder / f"{run_name}.h5").is_file():
                files.append(folder / f"{run_name}.h5")
        else:
            raise ValueError("Provide run_name or filename to select a PFLOTRAN run")
        if not files:
            raise FileNotFoundError(f"No PFLOTRAN HDF5 output found for {run_name!r} "
                                    f"({run_name}.h5 or {run_name}-NNN.h5) in {folder}")
        self._files = list(files)
        self._grid_centres = {}
        snapshots = []
        units = set()
        for path in files:
            with self._h5py.File(path, "r") as handle:
                for key in handle:
                    match = self._time_pattern.fullmatch(key)
                    if match and isinstance(handle[key], self._h5py.Group):
                        time = float(match[1].replace("D", "E").replace("d", "e"))
                        if not np.isfinite(time):
                            raise ValueError(f"Nonfinite snapshot time in {path}: {key}")
                        units.add(match[2])
                        snapshots.append((time, path, key))
        if not snapshots:
            raise ValueError("No PFLOTRAN 'Time: <value> <unit>' snapshot groups found")
        if len(units) != 1:
            raise ValueError(f"Mixed PFLOTRAN time units: {sorted(units)}")
        snapshots.sort(key=lambda entry: entry[0])
        self.times = np.array([entry[0] for entry in snapshots])
        if len(np.unique(self.times)) != len(self.times):
            raise ValueError("Duplicate PFLOTRAN snapshot times; select a single run/restart")
        self._snapshots = snapshots
        self.available_timesteps = [entry[2] for entry in snapshots]
        self.time_unit = units.pop()

    def read_field(self, variable, timestep_idx):
        self._check_index(timestep_idx)
        _, path, group = self._snapshots[timestep_idx]
        with self._h5py.File(path, "r") as handle:
            field = resolve_field(handle[group], variable, self.variable_map, self.aliases)
            dataset = handle[group][field]
            if not isinstance(dataset, self._h5py.Dataset):
                raise ValueError(f"PFLOTRAN field {field!r} must be a dataset")
            return np.asarray(dataset, dtype=float)

    def output_cell_centers(self, timestep_idx=0, *, axes=None):
        """Cell centres from the run's own grid, in the C-order flattening of
        ``load_timestep(timestep_idx)``.

        A structured grid's ``Coordinates`` group holds the cell edges along
        x, y and z, and its fields are ``(nx, ny, nz)``; an unstructured
        grid's ``Domain`` group holds the vertices and the XDMF cell
        connectivity. The snapshot's own file is looked in first, then the
        run's other files.
        """
        self._check_index(timestep_idx)
        _, path, group = self._snapshots[timestep_idx]
        shape = self.read_field("saturation" if self.variable == "water_content" else self.variable,
                                timestep_idx).shape
        for candidate in [path] + [other for other in self._files if other != path]:
            with self._h5py.File(candidate, "r") as handle:
                coords = handle.get("Coordinates")
                if isinstance(coords, self._h5py.Group) and all(
                        f"{axis} [m]" in coords for axis in "XYZ"):
                    edges = [np.asarray(coords[f"{axis} [m]"], dtype=float).ravel() for axis in "XYZ"]
                    cells = tuple(len(edge) - 1 for edge in edges)
                    if shape != cells:
                        raise ValueError(f"PFLOTRAN fields are {shape} but the Coordinates give "
                                         f"{cells} cells (nx, ny, nz)")
                    middles = [(edge[:-1] + edge[1:]) / 2.0 for edge in edges]
                    grid = np.meshgrid(*middles, indexing="ij")
                    centres = np.column_stack([axis.ravel() for axis in grid])
                    return select_axes(centres, axes)
                domain = handle.get("Domain")
                if isinstance(domain, self._h5py.Group) and "Cells" in domain and "Vertices" in domain:
                    key = str(candidate)
                    if key not in self._grid_centres:     # the decoding is the slow part
                        self._grid_centres[key] = xdmf_mixed_centroids(domain["Cells"][...],
                                                                       domain["Vertices"][...])
                    centres = self._grid_centres[key]
                    if len(centres) != int(np.prod(shape)):
                        raise ValueError(f"PFLOTRAN Domain has {len(centres)} cells, the fields "
                                         f"{int(np.prod(shape))}")
                    return select_axes(centres, axes)
        raise ValueError("This PFLOTRAN output records no grid (neither a Coordinates nor a "
                         "Domain group); pass cell_centers")


class PFLOTRANSaturation(PFLOTRANOutput):
    """Read liquid saturation without changing HDF5 axis order."""
    variable = "saturation"


class PFLOTRANPorosity(PFLOTRANOutput):
    """Read snapshot porosity."""
    variable = "porosity"


class PFLOTRANWaterContent(PFLOTRANOutput):
    """Read volumetric liquid water content calculated as porosity * saturation."""
    variable = "water_content"
