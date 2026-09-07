"""Explicit petrophysics and distance-limited transfer to hydrological grids."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree


def _parameter(value, shape, name, low=0, high=None, inclusive=False):
    result = np.broadcast_to(np.asarray(value, dtype=float), shape)
    bad = result < low if inclusive else result <= low
    if not np.isfinite(result).all() or np.any(bad) or (high is not None and np.any(result > high)):
        raise ValueError(f'Invalid {name}; check physical bounds and finite values.')
    return result


def interpret_resistivity(resistivity, *, rho_fluid, m, n, a=1., sigma_sur=0.,
                         porosity=None, saturation=None):
    """Infer S and water content given porosity, or porosity given known S.

    Uses the package's unscaled surface-conductivity convention:
    1/rho = phi**m/(a*rho_fluid)*S**n + sigma_sur*S**(n-1).
    Supply exactly one of porosity or saturation. Parameters may be scalar or
    broadcast to rho.shape. n must exceed 1. No clipping or fallback is applied
    to physically incompatible observations. rho uses ohm-m; sigma_sur uses S/m.
    """
    rho = np.asarray(resistivity, dtype=float)
    if not rho.size or not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError('Resistivity must contain finite positive values.')
    if (porosity is None) == (saturation is None):
        raise ValueError('Supply exactly one of known porosity or known saturation.')
    fluid = _parameter(rho_fluid, rho.shape, 'rho_fluid')
    cement = _parameter(m, rho.shape, 'm')
    exponent = _parameter(n, rho.shape, 'n', low=1)
    factor = _parameter(a, rho.shape, 'a')
    surface = _parameter(sigma_sur, rho.shape, 'sigma_sur', inclusive=True)
    observed = 1. / rho
    if porosity is not None:
        phi = _parameter(porosity, rho.shape, 'porosity', high=1)
        coefficient = phi**cement / (factor * fluid)
        if np.any(observed > (coefficient + surface) * (1 + 1e-12)):
            raise ValueError('Resistivity implies saturation above 1 under the supplied petrophysics.')
        sat = np.array(np.minimum((observed / coefficient)**(1/exponent), 1.))
        for index in np.ndindex(rho.shape):
            if surface[index] > 0:
                c, b, e, obs = coefficient[index], surface[index], exponent[index], observed[index]
                sat[index] = brentq(lambda s: c*s**e + b*s**(e-1) - min(obs, c+b), 0., 1., xtol=1e-12)
    else:
        sat = _parameter(saturation, rho.shape, 'saturation', high=1)
        bulk = observed - surface * sat**(exponent-1)
        if np.any(bulk <= 0):
            raise ValueError('Surface conductivity exceeds the observed conductivity.')
        phi = (bulk * factor * fluid / sat**exponent)**(1/cement)
        if np.any(phi > 1):
            raise ValueError('Resistivity implies porosity above 1.')
    return {'porosity': np.array(phi), 'saturation': np.array(sat),
            'water_content': np.array(phi * sat)}


def saturation_to_pressure(saturation, *, alpha, n, residual_saturation,
                           saturated_pressure=None):
    """Invert van Genuchten for pressure head (negative in unsaturated cells).

    alpha is inverse length; output uses that length unit. m=1-1/n. Fully
    saturated cells need an explicit pressure (scalar or matching array), because
    saturation alone cannot determine positive pressure. S at/below residual is
    rejected because its inverse pressure is unbounded.
    """
    sat = np.asarray(saturation, dtype=float)
    _parameter(sat, sat.shape, 'saturation', high=1)
    alpha = _parameter(alpha, sat.shape, 'alpha')
    n = _parameter(n, sat.shape, 'van Genuchten n', low=1)
    residual = _parameter(residual_saturation, sat.shape, 'residual saturation', high=1, inclusive=True)
    if np.any(sat <= residual):
        raise ValueError('Saturation must exceed residual saturation.')
    se = (sat-residual)/(1-residual)
    full = sat >= 1
    if np.any(full) and saturated_pressure is None:
        raise ValueError('Fully saturated cells require explicit saturated_pressure.')
    with np.errstate(over='raise', invalid='raise'):
        pressure = -np.expm1(-np.log(se)/(1-1/n))**(1/n)/alpha
    if saturated_pressure is not None:
        supplied = _parameter(saturated_pressure, sat.shape, 'saturated pressure', inclusive=True)
        pressure = np.where(full, supplied, pressure)
    return pressure


@dataclass
class HydroGrid:
    """Cell centres in a shared metric XYZ frame; Z is elevation, increasing up."""
    centers: np.ndarray
    active: np.ndarray

    def __post_init__(self):
        self.centers = np.array(self.centers, dtype=float, copy=True)
        self.active = np.array(self.active, dtype=bool, copy=True)
        if self.centers.shape != self.active.shape + (3,) or not np.isfinite(self.centers).all():
            raise ValueError('centers must be finite with shape active.shape + (3,).')

    @classmethod
    def from_modflow(cls, model):
        """Use FloPy cell centres, including model offsets and rotation."""
        grid = model.modelgrid
        z = np.asarray(grid.zcellcenters)
        centers = np.stack([np.broadcast_to(grid.xcellcenters, z.shape),
                            np.broadcast_to(grid.ycellcenters, z.shape), z], axis=-1)
        active = np.ones(z.shape, dtype=bool) if grid.idomain is None else np.asarray(grid.idomain) > 0
        return cls(centers, active)

    @classmethod
    def from_parflow(cls, config, *, active):
        """Build uniform-grid PFB centres, bottom-up. Supply an explicit domain mask.

        Variable-dz / terrain-following grids need explicitly constructed centres.
        """
        config = config if isinstance(config, dict) else config.to_dict()
        if config.get('Solver.Nonlinear.VariableDz', False) or config.get('Solver.TerrainFollowingGrid', False):
            raise ValueError('Supply explicit HydroGrid centres for variable-dz or terrain-following grids.')
        coordinates = []
        for axis in ('X', 'Y', 'Z'):
            count = int(config['ComputationalGrid.N' + axis])
            spacing = float(config['ComputationalGrid.D' + axis])
            if count <= 0 or not np.isfinite(spacing) or spacing <= 0:
                raise ValueError('Grid counts and spacing must be positive.')
            coordinates.append(float(config['ComputationalGrid.Lower.' + axis]) + (np.arange(count)+.5)*spacing)
        z, y, x = np.meshgrid(coordinates[2], coordinates[1], coordinates[0], indexing='ij')
        return cls(np.stack([x,y,z], axis=-1), active)


@dataclass
class MappedField:
    """Mapped values and an audit of which target cells were updated."""
    values: np.ndarray
    updated: np.ndarray
    distance: np.ndarray
    source_index: np.ndarray


def map_to_hydro_grid(source_xyz, values, grid, *, baseline, max_distance,
                      source_valid=None):
    """Transfer nearest valid samples within an explicit distance; keep other cells.

    source_xyz must be N×3 in the same CRS, metre units and vertical datum as
    grid.centers. No CRS guessing, depth-to-elevation conversion or extrapolation
    beyond max_distance is performed. This is point transfer, not conservative
    cell-volume averaging. Pass DOI/coverage exclusions through source_valid.
    """
    xyz = np.asarray(source_xyz, dtype=float)
    values = np.asarray(values, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or values.shape != (len(xyz),):
        raise ValueError('source_xyz must be (N,3) and values must be (N,).')
    if not np.isfinite(xyz).all() or not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError('Coordinates must be finite and max_distance positive.')
    valid = np.ones(len(values), dtype=bool) if source_valid is None else np.asarray(source_valid, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError('source_valid must match values.')
    valid = valid & np.isfinite(values)
    if not valid.any():
        raise ValueError('No valid source samples remain.')
    # Conflicting values at identical positions otherwise depend on tree ordering.
    if len(np.unique(xyz[valid], axis=0)) != int(valid.sum()):
        raise ValueError('Aggregate duplicate source positions explicitly before mapping.')
    output = np.array(baseline, dtype=float, copy=True)
    if output.shape != grid.active.shape or not np.isfinite(output).all():
        raise ValueError('A finite full-grid baseline is required.')
    distances, indices = cKDTree(xyz[valid]).query(grid.centers.reshape(-1,3))
    distances = distances.reshape(output.shape)
    indices = np.flatnonzero(valid)[indices].reshape(output.shape)
    updated = grid.active & (distances <= max_distance)
    output[updated] = values[indices[updated]]
    return MappedField(output, updated, distances, np.where(updated, indices, -1))


def prepare_hydro_updates(source_xyz, resistivity, grid, *, petrophysics,
                          field_transforms, baselines, max_distance, source_valid=None):
    """Convert source cells, derive requested fields and map to writer-ready arrays.

    field_transforms maps writer field names to either an interpreted field name
    (e.g. 'porosity') or a callable accepting the interpreted source dictionary.
    Use a calibrated callable for conductivity or a retention relation for pressure.
    Returns {field: MappedField}; pass each .values to the input writer.
    """
    rho = np.asarray(resistivity, dtype=float)
    xyz = np.asarray(source_xyz, dtype=float)
    if rho.ndim != 1 or xyz.shape != (len(rho),3):
        raise ValueError('Provide one resistivity per XYZ source cell.')
    valid = np.ones(rho.shape, dtype=bool) if source_valid is None else np.asarray(source_valid, dtype=bool)
    if valid.shape != rho.shape or not valid.any():
        raise ValueError('Provide at least one valid source cell.')
    params = {k: (None if v is None else np.broadcast_to(v, rho.shape)[valid]) for k,v in petrophysics.items()}
    interpreted = interpret_resistivity(rho[valid], **params)
    if not field_transforms or set(field_transforms) - baselines.keys():
        raise ValueError('Every requested field needs a baseline.')
    results = {}
    for name, transform in field_transforms.items():
        source = interpreted[transform] if isinstance(transform, str) else transform(interpreted)
        source = np.asarray(source, dtype=float)
        if source.shape != (int(valid.sum()),):
            raise ValueError(f'{name} transform must return one value per valid source cell.')
        values = np.full(rho.shape, np.nan)
        values[valid] = source
        if not np.isfinite(values[valid]).all():
            raise ValueError(f'{name} transform returned nonfinite values.')
        results[name] = map_to_hydro_grid(xyz, values, grid, baseline=baselines[name],
                                          max_distance=max_distance, source_valid=valid)
    return results
