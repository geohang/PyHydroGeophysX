"""Read ATS visualization HDF5 files (field / cycle / scalar cell values).

Supports current ``ats_vis_*_data.h5`` and older ``visdump_data.h5`` files.
The domain is read from the filename (``ats_vis_data.h5`` is the subsurface
domain, ``ats_vis_surface_data.h5`` the surface), and a variable is found
under its plain name or with that domain's prefix (``domain-saturation_liquid``),
as ATS names them; other names are selected with ``variable_map``. Cell
centres come from the companion ``*_mesh.h5`` file ATS writes beside the data.
"""
from pathlib import Path
import re

import numpy as np

from ._hdf5_output import (HDF5ModelOutput, resolve_field, select_axes, text_attribute,
                           xdmf_mixed_centroids)


class ATSOutput(HDF5ModelOutput):
    """ATS subsurface scalar output; returns one-dimensional cell arrays.

    ``filename`` is relative to model_directory or absolute. When omitted,
    exactly one of the standard subsurface filenames must exist. ``time_unit``
    overrides missing file metadata; otherwise an unknown unit remains None.
    """

    base_aliases = {
        "saturation": ("saturation_liquid", "saturation_liquid.cell.0"),
        "porosity": ("porosity", "porosity.cell.0"),
        "pressure": ("pressure", "pressure.cell.0"),
        "temperature": ("temperature", "temperature.cell.0"),
    }

    def __init__(self, model_directory, filename=None, *, variable_map=None,
                 cell_centers=None, time_unit=None):
        super().__init__(model_directory, variable_map=variable_map, cell_centers=cell_centers)
        if filename is None:
            candidates = [Path(model_directory) / name for name in
                          ("ats_vis_data.h5", "ats_vis_domain_data.h5", "visdump_data.h5")]
            candidates = [p for p in candidates if p.is_file()]
            if len(candidates) != 1:
                raise ValueError("Specify filename: expected exactly one standard ATS subsurface data file")
            self.filename = candidates[0]
        else:
            self.filename = Path(model_directory) / filename
        # ATS names a domain's variables "<domain>-<name>", the subsurface's
        # "domain-<name>" in recent versions and "<name>" in older ones.
        match = re.fullmatch(r"ats_vis_(.+)_data\.h5", self.filename.name)
        self.domain = match.group(1) if match else "domain"
        self.aliases = {key: tuple(names) + tuple(f"{self.domain}-{name}" for name in names)
                        for key, names in self.base_aliases.items()}
        self._mesh_centres = {}
        with self._h5py.File(self.filename, "r") as handle:
            field = resolve_field(handle, "saturation" if self.variable == "water_content" else self.variable,
                                  self.variable_map, self.aliases)
            group = handle[field]
            if not isinstance(group, self._h5py.Group):
                raise ValueError(f"ATS field {field!r} must contain cycle datasets")
            self._cycles = sorted([key for key in group if key.isdigit()], key=int)
            if not self._cycles:
                raise ValueError(f"No ATS cycles found for {field!r}")
            self.available_timesteps = [int(key) for key in self._cycles]
            self.times = np.array([float(group[key].attrs.get("Time", np.nan)) for key in self._cycles])
            unit = handle.attrs.get("time unit")
            self.time_unit = time_unit if time_unit is not None else (None if unit is None else text_attribute(unit))

    def read_field(self, variable, timestep_idx):
        self._check_index(timestep_idx)
        cycle = self._cycles[timestep_idx]
        with self._h5py.File(self.filename, "r") as handle:
            field = resolve_field(handle, variable, self.variable_map, self.aliases)
            if cycle not in handle[field]:
                raise KeyError(f"ATS field {field!r} missing cycle {cycle}")
            dataset = handle[field][cycle]
            time = float(dataset.attrs.get("Time", np.nan))
            if np.isfinite(time) and np.isfinite(self.times[timestep_idx]) and time != self.times[timestep_idx]:
                raise ValueError(f"ATS field {field!r} has an inconsistent Time for cycle {cycle}")
            values = np.asarray(dataset, dtype=float)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(f"ATS scalar cell field expected, got shape {values.shape}")
        return values

    @property
    def mesh_filename(self):
        """The mesh file ATS writes beside the data file (``*_mesh.h5``)."""
        name = self.filename.name
        if not name.endswith("_data.h5"):
            raise ValueError(f"Cannot name the mesh file of {name!r}; pass cell_centers")
        return self.filename.with_name(name[:-len("_data.h5")] + "_mesh.h5")

    def output_cell_centers(self, timestep_idx=0, *, axes=None):
        """Element centres from ATS's mesh file, in the cell order of the data.

        The mesh of this timestep's cycle is used when ATS wrote one (a
        deforming mesh), else the latest mesh written before it, else the only
        one (a fixed mesh).
        """
        self._check_index(timestep_idx)
        cycle = self._cycles[timestep_idx]
        path = self.mesh_filename
        if not path.is_file():
            raise FileNotFoundError(f"ATS mesh file {path} not found; pass cell_centers")
        with self._h5py.File(path, "r") as handle:
            keys = [key for key in handle
                    if isinstance(handle[key], self._h5py.Group) and "Mesh" in handle[key]]
            if cycle in keys:
                key = cycle
            else:
                earlier = [key for key in keys if key.isdigit() and int(key) <= int(cycle)]
                if earlier:
                    key = max(earlier, key=int)
                elif len(keys) == 1:
                    key = keys[0]
                else:
                    raise KeyError(f"No mesh in {path.name} for cycle {cycle}; found {keys}")
            if key not in self._mesh_centres:
                mesh = handle[key]["Mesh"]
                self._mesh_centres[key] = xdmf_mixed_centroids(mesh["MixedElements"][...],
                                                               mesh["Nodes"][...])
        return select_axes(self._mesh_centres[key], axes)


class ATSSaturation(ATSOutput):
    """Read liquid saturation in native ATS cell order."""
    variable = "saturation"


class ATSPorosity(ATSOutput):
    """Read porosity with its own available cycles."""
    variable = "porosity"


class ATSWaterContent(ATSOutput):
    """Read volumetric liquid water content calculated as porosity * saturation."""
    variable = "water_content"
