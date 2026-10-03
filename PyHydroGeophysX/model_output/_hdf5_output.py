"""Shared, import-light helpers for HDF5 hydrological output readers."""
from pathlib import Path

import numpy as np

from .base import HydroModelOutput


def require_h5py():
    """Load the optional dependency only when a reader is constructed."""
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            'ATS/PFLOTRAN readers require h5py; install "pyhydrogeophysx[hydrology]".'
        ) from exc
    return h5py


def text_attribute(value):
    return value.decode() if isinstance(value, bytes) else str(value)


#: Nodes per element of the fixed-size XDMF topology types: triangle,
#: quadrilateral, tetrahedron, pyramid, wedge and hexahedron.
_XDMF_NODES = {4: 3, 5: 4, 6: 4, 7: 5, 8: 6, 9: 8}
_XDMF_POLYGON, _XDMF_POLYHEDRON = 3, 16


def xdmf_mixed_centroids(connectivity, nodes):
    """Element centres of an XDMF mixed topology, as ATS and PFLOTRAN write it.

    ``connectivity`` is the flat array of ATS's ``Mesh/MixedElements`` or
    PFLOTRAN's ``Domain/Cells``: for each element its XDMF type code, then its
    zero-based node numbers (a polygon gives its node count first, a
    polyhedron its face count and then each face as a polygon). The centre is
    the mean of the element's distinct nodes, as ATS's own ``ats_xdmf`` tools
    compute it. Returns ``(n_elements, dimension)`` in element order, which is
    the order of the cell values.
    """
    conn = np.asarray(connectivity, dtype=np.int64).ravel()
    nodes = np.asarray(nodes, dtype=float)
    if conn.size == 0:
        raise ValueError("The mesh has no elements")
    code = int(conn[0])
    width = _XDMF_NODES.get(code, 0) + 1
    if width > 1 and conn.size % width == 0 and np.all(conn[::width] == code):
        # One fixed-size element type throughout: read it as a table.
        return nodes[conn.reshape(-1, width)[:, 1:]].mean(axis=1)
    centres, i = [], 0
    while i < conn.size:
        code = int(conn[i])
        if code in _XDMF_NODES:
            ids = conn[i + 1:i + 1 + _XDMF_NODES[code]]
            i += 1 + _XDMF_NODES[code]
        elif code == _XDMF_POLYGON:
            count = int(conn[i + 1])
            ids = conn[i + 2:i + 2 + count]
            i += 2 + count
        elif code == _XDMF_POLYHEDRON:
            faces, i, ids = int(conn[i + 1]), i + 2, []
            for _ in range(faces):
                count = int(conn[i])
                ids.extend(conn[i + 1:i + 1 + count].tolist())
                i += 1 + count
            ids = np.unique(ids)
        else:
            raise ValueError(f"Unsupported XDMF element type {code} in the mesh")
        centres.append(nodes[ids].mean(axis=0))
    return np.asarray(centres)


def select_axes(centres, axes):
    """Columns ``axes`` (e.g. ``"xz"``) of x, y, z cell centres."""
    if axes is None:
        return centres
    picks = ["xyz".index(axis) for axis in str(axes).lower()]
    if not picks or max(picks) >= centres.shape[1]:
        raise ValueError(f"axes {axes!r} do not name coordinates of {centres.shape[1]}-D centres")
    return centres[:, picks]


class HDF5ModelOutput(HydroModelOutput):
    """Common time-series and spatial mapping API; subclasses index the files.

    ``variable_map`` maps canonical field names to exact HDF5 variable names.
    Arrays retain the simulator's cell/axis order. No clipping, grid reshaping,
    coordinate conversion or time conversion is performed.
    """

    variable = "saturation"

    def __init__(self, model_directory, *, variable_map=None, cell_centers=None):
        super().__init__(str(model_directory))
        if not Path(model_directory).is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_directory}")
        self._h5py = require_h5py()
        self.variable_map = dict(variable_map or {})
        self.cell_centers = None if cell_centers is None else np.asarray(cell_centers, dtype=float)
        self.available_timesteps = []
        self.times = np.empty(0)
        self.time_unit = None

    def _check_index(self, timestep_idx):
        if not isinstance(timestep_idx, (int, np.integer)) or not 0 <= timestep_idx < len(self.available_timesteps):
            raise ValueError(f"Timestep index {timestep_idx!r} out of range for {len(self.available_timesteps)} timesteps")

    def read_field(self, variable, timestep_idx):
        """Read a canonical field or an exact HDF5 variable name."""
        raise NotImplementedError

    def load_timestep(self, timestep_idx, **kwargs):
        """Read one zero-based timestep; ``variable`` may override the field."""
        variable = kwargs.pop("variable", self.variable)
        porosity = kwargs.pop("porosity", None)
        if kwargs:
            raise TypeError(f"Unexpected arguments: {', '.join(kwargs)}")
        self._check_index(timestep_idx)
        if variable == "water_content":
            return self.get_water_content(timestep_idx, porosity=porosity)
        return self.read_field(variable, timestep_idx)

    def get_water_content(self, timestep_idx, porosity=None):
        """Volumetric liquid water content (m3/m3) = liquid saturation * porosity.

        ATS's native ``water_content`` is an extensive quantity and is deliberately
        not used. Pass a scalar or an exactly aligned array if porosity was not
        saved in the output. Ice is not included.
        """
        saturation = self.read_field("saturation", timestep_idx)
        if porosity is None:
            porosity = self.read_field("porosity", timestep_idx)
        porosity = np.asarray(porosity, dtype=float)
        if porosity.ndim != 0 and porosity.shape != saturation.shape:
            raise ValueError(f"Porosity shape {porosity.shape} must match saturation {saturation.shape}")
        if np.any(np.isinf(porosity) | (porosity < 0) | (porosity > 1)):
            raise ValueError("Porosity must be between 0 and 1 (NaN is allowed for missing cells)")
        return saturation * porosity

    def load_time_range(self, start_idx=0, end_idx=None, **kwargs):
        """Stack timesteps with an exclusive end index, following Python slicing."""
        indices = list(range(len(self.available_timesteps)))[start_idx:end_idx]
        if not indices:
            sample = self.load_timestep(0, **kwargs)
            return np.empty((0, *sample.shape), dtype=float)
        arrays = [self.load_timestep(idx, **kwargs) for idx in indices]
        if any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("Spatial shape changes across timesteps; map each timestep separately")
        return np.stack(arrays)

    def get_timestep_info(self):
        """Return (cycle/group name, time) pairs in the output's native time unit.

        Missing ATS Time attributes are represented by NaN, never by cycle number.
        """
        return list(zip(self.available_timesteps, self.times.tolist()))

    def output_cell_centers(self, timestep_idx=0, *, axes=None):
        """Cell centres the output's own mesh gives, in the order of
        ``load_timestep(timestep_idx).ravel()``; ``axes`` picks coordinates,
        e.g. ``"xz"`` for a vertical section. Readers implement this."""
        raise NotImplementedError(f"{type(self).__name__} does not read its mesh; pass cell_centers")

    def interpolate_timestep(self, timestep_idx, target_centers, *, cell_centers=None,
                             method="linear", axes=None, **kwargs):
        """Interpolate a scalar field onto geophysical cell centers using SciPy.

        Source centers must follow ``load_timestep(...).ravel()`` exactly; source
        and target must share coordinates, dimensions (2 or 3), datum and units.
        Without ``cell_centers`` (here or on the reader) the output's own mesh
        gives them (:meth:`output_cell_centers`), for this timestep, so a moving
        mesh is followed; ``axes`` then picks the coordinates to use, e.g.
        ``"xz"`` for a 2-D section of a run one cell wide in y. Linear
        interpolation returns NaN outside the source convex hull. Nearest
        interpolation extrapolates.
        """
        from scipy.interpolate import griddata

        source = self.cell_centers if cell_centers is None else np.asarray(cell_centers, dtype=float)
        target = np.asarray(target_centers, dtype=float)
        if source is None:
            source = self.output_cell_centers(timestep_idx, axes=axes)
        elif axes is not None:
            source = select_axes(source, axes)
        if source.ndim != 2 or source.shape[1] not in (2, 3):
            raise ValueError("cell_centers must have shape (n_cells, 2 or 3)")
        if target.ndim != 2 or target.shape[1] != source.shape[1]:
            raise ValueError(
                f"target_centers are {target.shape[-1]}-D but the cell centers are "
                f"{source.shape[1]}-D; give axes (e.g. axes='xz') to choose the source "
                "coordinates")
        if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
            raise ValueError("Cell coordinates must be finite")
        if method not in ("linear", "nearest"):
            raise ValueError("method must be 'linear' or 'nearest'")
        values = self.load_timestep(timestep_idx, **kwargs).ravel()
        if len(values) != len(source):
            raise ValueError("One source center is required per output cell")
        valid = np.isfinite(values)
        if not np.any(valid):
            raise ValueError("No finite source values to interpolate")
        return griddata(source[valid], values[valid], target, method=method)


def resolve_field(container, variable, variable_map, aliases):
    """Resolve known aliases, refusing ambiguous matches instead of guessing."""
    if variable in variable_map:
        name = variable_map[variable]
        if name not in container:
            raise KeyError(f"Configured field {name!r} missing; available: {list(container)}")
        return name
    if variable in container:
        return variable
    matches = [name for name in aliases.get(variable, ()) if name in container]
    if len(matches) != 1:
        raise KeyError(f"Cannot uniquely resolve {variable!r}; matches: {matches}; available: {list(container)}. Set variable_map.")
    return matches[0]
