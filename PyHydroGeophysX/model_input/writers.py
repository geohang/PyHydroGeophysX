"""Model-input adapters. Values must already be in the target grid and units."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import time

import numpy as np


def _array(value, shape, name, bounds):
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite with shape {tuple(shape)}; got {array.shape}.')
    low, high = bounds
    inclusive = name in ('specific_storage', 'specific_yield')
    if low is not None and np.any(array < low if inclusive else array <= low):
        relation = 'at least' if inclusive else 'greater than'
        raise ValueError(f'{name} must be {relation} {low}.')
    if high is not None and np.any(array > high):
        raise ValueError(f'{name} must be at most {high}.')
    return array.copy()


def _destination(path):
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f'Use a new output directory: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _relative(value):
    path = Path(str(value))
    if path.is_absolute() or '..' in path.parts or ':' in str(value):
        raise ValueError(f'Input references must be relative and remain in the workspace: {value}')
    return path


def _record(folder, engine, fields, convention):
    (folder / 'hydro_update.json').write_text(json.dumps({
        'engine': engine, 'updated_fields': list(fields), 'array_convention': convention,
        'simulation_executed': False,
    }, indent=2), encoding='utf-8')


def _publish(folder, target):
    # Windows scanners can briefly hold freshly written PFB files open.
    for attempt in range(5):
        if target.exists():
            raise FileExistsError(target)
        try:
            folder.rename(target)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.05 * (attempt + 1))


def write_modflow6_inputs(simulation, output_dir, updates, *, model_name=None):
    """Write a copy of a FloPy MF6 simulation with mapped GWF array updates.

    Supported keys are ``hydraulic_conductivity`` (NPF k), ``vertical_conductivity``
    (NPF k33), ``specific_storage`` (STO ss), ``specific_yield`` (STO sy), and
    ``initial_head`` (IC strt). ``bottom_elevation`` updates structured DIS layer
    bottoms and requires strictly positive layer thickness in every column.
    Arrays must match the model grid exactly, in model
    length/time units. Structured grids use (layer, row, column), top layer first.
    Existing packages are required; stress periods and boundary conditions remain
    as configured. This writes inputs only and never runs MODFLOW.
    """
    rules = {
        'hydraulic_conductivity': ('npf', 'k', (0, None)),
        'vertical_conductivity': ('npf', 'k33', (0, None)),
        'specific_storage': ('sto', 'ss', (0, None)),
        'specific_yield': ('sto', 'sy', (0, 1)),
        'initial_head': ('ic', 'strt', (None, None)),
        'bottom_elevation': ('dis', 'botm', (None, None)),
    }
    if not updates or set(updates) - rules.keys():
        raise ValueError(f'Choose one or more supported fields: {list(rules)}')
    names = simulation.model_names
    if model_name is None:
        if len(names) != 1:
            raise ValueError('Select model_name explicitly for a multi-model simulation.')
        model_name = names[0]
    model = simulation.get_model(model_name)
    if model is None or model.model_type != 'gwf6':
        raise ValueError('A MODFLOW 6 groundwater-flow (GWF) model is required.')
    arrays = {}
    for key, value in updates.items():
        package, field, bounds = rules[key]
        if model.get_package(package) is None:
            raise ValueError(f'{key} requires an existing {package.upper()} package.')
        # With ratio flags, k33 is dimensionless rather than conductivity.
        if key == 'vertical_conductivity' and model.npf.k33overk.get_data():
            raise ValueError('Disable k33overk before supplying absolute vertical conductivity.')
        arrays[key] = _array(value, model.modelgrid.shape, key, bounds)
    if 'bottom_elevation' in arrays:
        bottoms = arrays['bottom_elevation']
        interfaces = np.concatenate([np.asarray(model.dis.top.array)[None], bottoms], axis=0)
        if np.any(np.diff(interfaces, axis=0) >= 0):
            raise ValueError('Layer bottoms must decrease strictly below top in every column.')
    clone = deepcopy(simulation)
    # Resolve external arrays in the original workspace before relocating the copy.
    clone.set_all_data_internal()
    for name in clone.model_names:
        m = clone.get_model(name)
        _relative(clone.simulation_data.mfpath.model_relative_path.get(name, '.') or '.')
        for package in m.packagelist:
            _relative(package.filename)
    for package in clone.sim_package_list:
        _relative(package.filename)
    for key, value in arrays.items():
        package, field, _ = rules[key]
        getattr(clone.get_model(model_name).get_package(package), field).set_data(value)
    target = _destination(output_dir)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.hydro-update-') as temp:
        folder = Path(temp) / 'model'
        folder.mkdir()
        clone.set_sim_path(folder)
        clone.write_simulation(silent=True)
        _record(folder, 'MODFLOW6', arrays, 'native model grid; model length/time units')
        _publish(folder, target)
    return target


def write_parflow_inputs(run, source_dir, output_dir, updates, *, domain='domain'):
    """Copy a ParFlow workspace and write PFB updates plus a PFIDB definition.

    Keys: ``permeability``, ``porosity``, ``initial_pressure``. Values use the
    existing run's units and (z, y, x) grid order, bottom layer first. Permeability
    retains the run's tensor settings. Initial pressure is pressure head, not
    saturation. Other input files must be inside source_dir with relative paths.
    The source workspace is copied, including forcing files; no solver is run.
    ``run`` may be a ParFlow Run or a flat key/value dictionary from read_pfidb.
    Only the full-domain field is replaced; this is not a partial-facies update.
    """
    from parflow.tools.io import write_pfb, write_dict
    rules = {'permeability': (0, None), 'porosity': (0, 1),
             'initial_pressure': (None, None)}
    if not updates or set(updates) - rules.keys():
        raise ValueError(f'Choose one or more supported fields: {list(rules)}')
    if not domain.isidentifier():
        raise ValueError('domain must be a simple geometry name.')
    config = deepcopy(run if isinstance(run, dict) else run.to_dict())
    if config.get('Domain.GeomName') != domain:
        raise ValueError('Select the existing full-domain geometry; partial geometry updates are unsupported.')
    def grid(key):
        return config['ComputationalGrid.' + key]
    shape = tuple(int(grid(key)) for key in ('NZ', 'NY', 'NX'))
    spacing = np.asarray([grid(key) for key in ('DX', 'DY', 'DZ')], dtype=float)
    origin = np.asarray([grid('Lower.' + key) for key in ('X', 'Y', 'Z')], dtype=float)
    topology = tuple(int(config.get('Process.Topology.' + key, 1)) for key in ('P', 'Q', 'R'))
    if not np.isfinite(spacing).all() or np.any(spacing <= 0) or not np.isfinite(origin).all():
        raise ValueError('Grid origin must be finite and spacing must be positive.')
    if any(n <= 0 or n > extent for n, extent in zip(topology, shape[::-1])):
        raise ValueError('Process topology must fit the grid dimensions.')
    arrays = {k: _array(v, shape, k, rules[k]) for k, v in updates.items()}
    source = Path(source_dir).resolve()
    target = _destination(output_dir)
    if not source.is_dir() or source == target or source in target.parents:
        raise ValueError('Use an existing source workspace and a separate output directory.')
    if any(p.is_symlink() for p in source.rglob('*')):
        raise ValueError('Source workspace must not contain symbolic links.')
    # Keep ancillary file references portable when copying the workspace.
    for key, value in config.items():
        if key.endswith('.FileName') and value:
            path = _relative(value)
            if not (source / path).is_file():
                raise FileNotFoundError(source / path)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.hydro-update-') as temp:
        folder = Path(temp) / 'model'
        shutil.copytree(source, folder)
        data_dir = folder / 'hydro_updates'
        if data_dir.exists():
            raise FileExistsError('Source already contains hydro_updates; use a clean source workspace.')
        data_dir.mkdir()
        for key, array in arrays.items():
            filename = f'hydro_updates/{key}.pfb'
            write_pfb(str(folder / filename), array,
                      p=int(config.get('Process.Topology.P', 1)),
                      q=int(config.get('Process.Topology.Q', 1)),
                      r=int(config.get('Process.Topology.R', 1)), x=float(grid('Lower.X')),
                      y=float(grid('Lower.Y')), z=float(grid('Lower.Z')),
                      dx=float(grid('DX')), dy=float(grid('DY')), dz=float(grid('DZ')), dist=True)
            if key == 'initial_pressure':
                settings = {'ICPressure.Type': 'PFBFile', 'ICPressure.GeomNames': domain,
                            f'Geom.{domain}.ICPressure.FileName': filename}
            else:
                prop = 'Perm' if key == 'permeability' else 'Porosity'
                settings = {f'Geom.{prop}.Names': domain,
                            f'Geom.{domain}.{prop}.Type': 'PFBFile',
                            f'Geom.{domain}.{prop}.FileName': filename}
                if prop == 'Porosity':
                    settings.pop('Geom.Porosity.Names')
                    settings['Geom.Porosity.GeomNames'] = domain
            config.update(settings)
        # The official flat-dictionary writer avoids Run.pfset's Windows path handling
        # and leaves existing CLM forcing files exactly as copied from source_dir.
        write_dict(config, str(folder / 'updated_model.pfidb'))
        _record(folder, 'ParFlow', arrays, 'z,y,x; bottom first; native run units')
        _publish(folder, target)
    return target
