"""Qt-free, project-owned survey snapshots and coordinate placement.

Besides the surveys, the map holds wells (kind ``'wells'``): the Boreholes
page's data, saved with the project, so a well can be compared with every survey
around it. :func:`profile_at` reads any survey snapshot down the vertical at a
map point - a section's cells, the nearest sounding, a grid column - which is
what that comparison draws.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .results_store import _atomic_write_json, _utc_now
from PyHydroGeophysX.inversion.em1d_lci import DOI_SENSITIVITY_THRESHOLD


def map_coordinates(xy, crs):
    """Convert explicit survey XY to the shared map frame; never infer a CRS."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('Survey coordinates must be a nonempty finite (n, 2) array.')
    crs = str(crs).strip().upper()
    if crs == 'LOCAL':
        return xy.copy(), 'local'
    if crs in ('EPSG:4326', '4326'):
        if np.any(np.abs(xy[:, 0]) > 180) or np.any(np.abs(xy[:, 1]) > 85.05112878):
            raise ValueError('Longitude/latitude is outside the supported map extent.')
        from PyHydroGeophysX.visualization.basemap import web_mercator
        x, y = web_mercator(xy[:, 0], xy[:, 1])
    else:
        try:
            from pyproj import CRS, Transformer
        except ImportError as exc:
            raise ValueError('Install pyproj to place projected coordinates, or supply longitude/latitude.') from exc
        source = CRS.from_user_input(crs)
        x, y = Transformer.from_crs(source, 'EPSG:3857', always_xy=True).transform(
            xy[:, 0], xy[:, 1], errcheck=True)
    mapped = np.column_stack([x, y])
    if not np.isfinite(mapped).all() or np.any(np.abs(mapped[:, 1]) > 20037509):
        raise ValueError('Coordinates cannot be placed within the supported map extent.')
    return mapped, 'geographic'


def place_profile(distance, start, bearing, controls=None):
    """Place metre chainages on a straight line or an explicit distance/x/y table."""
    distance = np.asarray(distance, dtype=float).ravel()
    if not distance.size or not np.isfinite(distance).all():
        raise ValueError('Profile distances must be finite.')
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if (controls.ndim != 2 or controls.shape[1] != 3 or len(controls) < 2
                or not np.isfinite(controls).all() or np.any(np.diff(controls[:, 0]) <= 0)):
            raise ValueError('Control CSV needs increasing distance,x,y rows (at least two).')
        if distance.min() < controls[0, 0] or distance.max() > controls[-1, 0]:
            raise ValueError('Control distances must span the full profile distance range.')
        return np.column_stack([np.interp(distance, controls[:, 0], controls[:, i]) for i in (1, 2)])
    start = np.asarray(start, dtype=float)
    if start.shape != (2,) or not np.isfinite(start).all() or not np.isfinite(bearing):
        raise ValueError('Start coordinate and bearing must be finite.')
    angle = np.deg2rad(bearing)
    return start + (distance - distance.min())[:, None] * np.array([np.sin(angle), np.cos(angle)])


def mesh_snapshot(mesh, values, method, title='', coverage=None, units=None):
    """Preserve original section cell polygons without needing PyGIMLi to reopen."""
    if mesh.dim() != 2:
        raise ValueError('Add to Map currently accepts 2D ERT/seismic sections.')
    values = np.asarray(values, dtype=float).ravel()
    if values.size != mesh.cellCount() or not np.isfinite(values).any():
        raise ValueError('The model must contain one value per mesh cell.')
    vertices, offsets = [], [0]
    for cell in mesh.cells():
        vertices.extend([[node.pos().x(), node.pos().y()] for node in cell.nodes()])
        offsets.append(len(vertices))
    x = np.asarray(vertices)[:, 0]
    arrays = {'vertices': np.asarray(vertices), 'offsets': np.asarray(offsets, dtype=int),
              'values': values, 'distance': np.linspace(x.min(), x.max(), 100)}
    if coverage is not None and np.size(coverage) == values.size:
        arrays['coverage'] = np.asarray(coverage, dtype=float).ravel()
    # A derived quantity - percentage change, water content - is not in the
    # method's own units, so the caller can say what it is rather than have the
    # map label a change map in ohm-metres.
    return {'kind': 'mesh', 'method': method, 'label': title or method,
            'units': units or ('m/s' if method == 'Seismic' else 'Ω·m')}, arrays


def em_snapshot(result):
    """Copy the inversion's own selected/reordered coordinates, not loader prefixes."""
    arrays = {}
    for key in ('model3d', 'positions', 'depth_edges', 'line_numbers', 'sensitivity',
                'doi', 'surface_elevation', 'chi2_list', 'chi2_effective_list'):
        if key in result:
            arrays[key] = np.asarray(result[key], dtype=float)
    model = arrays.get('model3d')
    if model is None or model.ndim != 3 or model.shape[1] != 1:
        raise ValueError('An EM line/sounding model is required.')
    n = model.shape[0]
    if arrays.get('positions', np.array([])).size != n:
        raise ValueError('EM positions do not match the recovered soundings.')
    arrays['distance'] = arrays['positions'].copy()
    if arrays.get('line_numbers', np.array([])).size != n:
        arrays['line_numbers'] = np.zeros(n)
    if arrays.get('depth_edges', np.array([])).size != model.shape[2] + 1:
        raise ValueError('EM depth boundaries do not match the model.')
    meta = {'kind': 'em', 'method': 'EM', 'label': str(result.get('method', 'EM')),
            'units': 'Ω·m', 'doi_threshold': float(result.get('doi_threshold', DOI_SENSITIVITY_THRESHOLD)),
            'log_scale': bool(result.get('log_scale', True)), 'cmap': result.get('cmap', 'turbo')}
    for names, crs in [(('longitude', 'latitude'), 'EPSG:4326'), (('x', 'y'), '')]:
        columns = [np.asarray(result.get(k, []), dtype=float).ravel() for k in names]
        if any(c.size != n for c in columns):
            continue
        xy = np.column_stack(columns)
        if xy.shape == (n, 2) and np.isfinite(xy).all():
            arrays['survey_xy'] = xy
            meta['suggested_crs'] = crs
            break
    return meta, arrays


def grid_snapshot(edges, model, method, units, label=None):
    """Preserve voxel values and grid geometry for map slices and 3D inspection."""
    edges = [np.asarray(e, dtype=float).ravel() for e in edges]
    model = np.asarray(model, dtype=float)
    if (len(edges) != 3 or model.shape != tuple(len(e) - 1 for e in edges)
            or any(not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in edges)):
        raise ValueError('Grid edges must increase and match the model shape (nx, ny, nz).')
    x, y = np.meshgrid((edges[0][:-1]+edges[0][1:])/2,
                       (edges[1][:-1]+edges[1][1:])/2, indexing='ij')
    xy = np.column_stack([x.ravel(), y.ravel()])
    return ({'kind': 'grid', 'method': method, 'units': units, 'label': label or method,
             'log_scale': False},
            {'model3d': model, 'x_edges': edges[0], 'y_edges': edges[1], 'z_edges': edges[2],
             'survey_xy': xy, 'distance': np.arange(len(xy), dtype=float)})


def point_snapshot(xy, values, method, units, label=None):
    """A scalar station/grid product for any geophysical method, without regridding."""
    xy, values = np.asarray(xy, dtype=float), np.asarray(values, dtype=float).ravel()
    if xy.shape != (len(values), 2) or not len(values) or not np.isfinite(xy).all():
        raise ValueError('Point products need one x,y position for every value.')
    if not np.isfinite(values).any():
        raise ValueError('No finite values in this product.')
    return ({'kind': 'points', 'method': str(method), 'units': str(units),
             'label': label or str(method)},
            {'survey_xy': xy, 'values': values, 'distance': np.arange(len(values), dtype=float)})


def section_snapshot(x_edges, depth_edges, values, method, units, label=None):
    """A section on a regular grid - a stacked reflection section in depth, say.

    ``values`` is ``(n_depth, n_x)``; each value becomes a rectangle cell, so the
    map reads it like any inverted section (kind ``'mesh'``), its top at z = 0.
    """
    x_edges = np.asarray(x_edges, dtype=float).ravel()
    z_edges = np.asarray(depth_edges, dtype=float).ravel()
    values = np.asarray(values, dtype=float)
    if values.shape != (z_edges.size - 1, x_edges.size - 1):
        raise ValueError('Section values must be (depth cells, x cells).')
    if not np.isfinite(values).any():
        raise ValueError('The section has no values.')
    vertices, offsets, flat = [], [0], []
    for i in range(z_edges.size - 1):
        top, bottom = -z_edges[i], -z_edges[i + 1]
        for j in range(x_edges.size - 1):
            a, b = x_edges[j], x_edges[j + 1]
            vertices.extend([[a, top], [b, top], [b, bottom], [a, bottom]])
            offsets.append(len(vertices))
            flat.append(values[i, j])
    arrays = {'vertices': np.asarray(vertices), 'offsets': np.asarray(offsets, dtype=int),
              'values': np.asarray(flat, dtype=float),
              'distance': np.linspace(x_edges.min(), x_edges.max(), 100)}
    return {'kind': 'mesh', 'method': method, 'label': label or method, 'units': units}, arrays


# ---------------------------------------------------------------------------
# Wells
# ---------------------------------------------------------------------------

def _clean(values):
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def wells_snapshot(data, label='Wells'):
    """The Boreholes page's data as a map layer: one point per well with a location.

    The logs, water levels and legend travel with the points as JSON (stored as
    bytes, since a snapshot holds numeric arrays only). The CRS the wells give
    is offered as the layer's; wells in more than one CRS are refused.
    """
    import json

    wells = [w for w in data.wells.values() if np.isfinite(w.easting) and np.isfinite(w.northing)]
    if not wells:
        raise ValueError('No well has a location yet: load the wells (easting, northing) first.')
    systems = sorted({w.crs.strip() for w in wells if w.crs.strip()})
    if len(systems) > 1:
        raise ValueError(f"The wells are in more than one coordinate system ({', '.join(systems)}); "
                         'give them one before adding them to the map.')
    payload = {
        'wells': [dict(well_id=w.well_id, easting=w.easting, northing=w.northing,
                       ground_elevation=_clean([w.ground_elevation])[0], crs=w.crs,
                       datum_offset=w.datum_offset, source=w.source) for w in wells],
        'lithology': [dict(well_id=i.well_id, top=i.top, bottom=i.bottom, unit=i.unit,
                           description=i.description) for i in data.lithology],
        'water_levels': [dict(well_id=w.well_id, time=w.time.isoformat() if w.time else None,
                              depth=_clean([w.depth])[0], head=_clean([w.head])[0],
                              reference=w.reference) for w in data.water_levels],
        'logs': [dict(well_id=g.well_id, name=g.name, unit=g.unit, depth=_clean(g.depth),
                      values=_clean(g.values), source=g.source) for g in data.logs],
        'legend': {unit: list(entry) for unit, entry in data.legend.items()},
    }
    raw = json.dumps(payload, sort_keys=True).encode('utf8')
    xy = np.array([(w.easting, w.northing) for w in wells], dtype=float)
    meta = {'kind': 'wells', 'method': 'Boreholes', 'label': label, 'units': '',
            'suggested_crs': systems[0] if systems else 'LOCAL'}
    return meta, {'survey_xy': xy, 'distance': np.arange(len(wells), dtype=float),
                  'boreholes_json': np.frombuffer(raw, dtype=np.uint8).copy()}


def wells_data(arrays):
    """The :class:`~PyHydroGeophysX.data_processing.boreholes.BoreholeData` of a wells layer,
    and its well ids in the order of the layer's points."""
    import datetime as dt
    import json

    from PyHydroGeophysX.data_processing import boreholes as bh

    payload = json.loads(bytes(np.asarray(arrays['boreholes_json'], dtype=np.uint8)).decode('utf8'))

    def number(value):
        return float('nan') if value is None else float(value)

    data = bh.BoreholeData()
    order = []
    for w in payload['wells']:
        data.wells[w['well_id']] = bh.Well(w['well_id'], w['easting'], w['northing'],
                                           number(w['ground_elevation']), w['crs'],
                                           w['datum_offset'], w['source'])
        order.append(w['well_id'])
    data.lithology = [bh.LithologyInterval(**i) for i in payload['lithology']]
    data.water_levels = [bh.WaterLevel(w['well_id'],
                                       dt.datetime.fromisoformat(w['time']) if w['time'] else None,
                                       number(w['depth']), number(w['head']), w['reference'])
                         for w in payload['water_levels']]
    data.logs = [bh.GeophysicalLog(g['well_id'], g['name'], g['unit'],
                                   [number(v) for v in g['depth']],
                                   [number(v) for v in g['values']], g['source'])
                 for g in payload['logs']]
    data.legend = {unit: tuple(entry) for unit, entry in payload['legend'].items()}
    return data, order


# ---------------------------------------------------------------------------
# Reading a survey down the vertical at a map point
# ---------------------------------------------------------------------------

def ground_distance(map_distance, map_y, frame):
    """A distance measured in the map frame, in ground metres.

    Web Mercator stretches lengths by 1/cos(latitude); a local frame is metres.
    """
    if frame != 'geographic':
        return float(map_distance)
    latitude = np.arctan(np.sinh(float(map_y) / 6378137.0))
    return float(map_distance) * float(np.cos(latitude))


def _cut(polygon, x):
    """The vertical extent of ``polygon`` along the line x = ``x``, or None."""
    ys = []
    p = np.asarray(polygon, dtype=float)
    for (x0, y0), (x1, y1) in zip(p, np.roll(p, -1, axis=0)):
        if (x0 - x) * (x1 - x) <= 0:
            if x1 == x0:
                ys.extend([y0, y1])
            else:
                ys.append(y0 + (x - x0) * (y1 - y0) / (x1 - x0))
    return (min(ys), max(ys)) if len(ys) >= 2 else None


def profile_at(entry, arrays, point, frame):
    """A survey read down the vertical at map point ``point``.

    Returns ``{'top', 'bottom', 'values'}`` (depth intervals in metres below the
    survey's surface, positive down), ``'faded'`` (below the depth of
    investigation, where the survey has one), ``'distance'`` (ground metres
    from ``point`` to where the survey was read), ``'where'`` (a short
    description of that place), or a single ``'value'`` for a point product;
    None for a kind with no values here.

    * section (``'mesh'``): the point is projected onto the section's trace and
      the cells the vertical crosses at that distance are read; depth is below
      the section's top there.
    * soundings (``'em'``): the nearest sounding's layers.
    * grid: the nearest column; ``Model Z`` read as depth when it is all of
      one sign.
    * points: the nearest station's value.
    """
    point = np.asarray(point, dtype=float)
    xy = np.asarray(arrays['map_xy'], dtype=float)
    kind = entry['kind']
    out = {'faded': None}
    if kind == 'mesh':
        from PyHydroGeophysX.data_processing.boreholes import ProfileLine

        try:
            line = ProfileLine(xy)
        except ValueError:
            return None
        geo = line.project([point[0]], [point[1]])
        along, off = float(geo['along'][0]), float(geo['distance'][0])
        lengths = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        distance = np.asarray(arrays['distance'], dtype=float)
        if along < 0 or along > lengths[-1]:
            return None
        d = float(np.interp(along, lengths, distance))
        vertices = np.asarray(arrays['vertices'], dtype=float)
        offsets = np.asarray(arrays['offsets'], dtype=int)
        values = np.asarray(arrays['values'], dtype=float)
        cells = []
        for k, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
            polygon = vertices[a:b]
            if polygon[:, 0].min() <= d <= polygon[:, 0].max():
                cut = _cut(polygon, d)
                if cut is not None and cut[1] > cut[0]:
                    cells.append((cut[0], cut[1], values[k]))
        if not cells:
            return None
        surface = max(c[1] for c in cells)
        cells.sort(key=lambda c: -c[1])
        out.update(top=np.array([surface - c[1] for c in cells]),
                   bottom=np.array([surface - c[0] for c in cells]),
                   values=np.array([c[2] for c in cells]),
                   distance=ground_distance(off, point[1], frame),
                   where=f'{d:.1f} m along the section')
        return out
    nearest = int(np.argmin(np.hypot(*(xy - point).T)))
    gap = ground_distance(float(np.hypot(*(xy[nearest] - point))), point[1], frame)
    if kind == 'em':
        edges = np.asarray(arrays['depth_edges'], dtype=float)
        rho = np.asarray(arrays['model3d'], dtype=float)[nearest, 0, ::-1]
        out.update(top=edges[:-1], bottom=edges[1:], values=rho, distance=gap,
                   where=f'sounding {nearest + 1}')
        sensitivity = arrays.get('sensitivity')
        if sensitivity is not None and np.shape(sensitivity)[:1] == (xy.shape[0],):
            out['faded'] = (np.asarray(sensitivity, dtype=float)[nearest]
                            < entry.get('doi_threshold', DOI_SENSITIVITY_THRESHOLD))
        return out
    if kind == 'grid':
        model = np.asarray(arrays['model3d'], dtype=float)
        z = np.asarray(arrays['z_edges'], dtype=float)
        if np.all(z <= 0):
            depth = -z
        elif np.all(z >= 0):
            depth = z
        else:
            return None
        i, j = divmod(nearest, model.shape[1])
        column = model[i, j, :]
        order = np.argsort(depth[:-1])
        top, bottom = np.minimum(depth[:-1], depth[1:]), np.maximum(depth[:-1], depth[1:])
        out.update(top=top[order], bottom=bottom[order], values=column[order], distance=gap,
                   where='nearest grid column')
        return out
    if kind == 'points':
        out.update(value=float(np.asarray(arrays['values'], dtype=float)[nearest]), distance=gap,
                   where='nearest station')
        return out
    return None


def em_result(meta, arrays):
    result = dict(arrays)
    result.update(label='Resistivity (Ω·m)', cmap=meta.get('cmap', 'turbo'),
                  log_scale=meta.get('log_scale', True), doi_threshold=meta.get('doi_threshold', DOI_SENSITIVITY_THRESHOLD))
    return result


class ProjectMapStore:
    """Map membership is saved immediately, independent of unsaved run history."""
    def __init__(self, root, read_only=False):
        self.root = Path(root).resolve()
        self.directory = self.root / 'project_map'
        self.index = self.directory / 'index.json'
        self.read_only = read_only

    def entries(self):
        if not self.index.exists():
            return []
        payload = json.loads(self.index.read_text(encoding='utf8'))
        if payload.get('schema_version') != 1 or not isinstance(payload.get('surveys'), list):
            raise ValueError('Unsupported or invalid project map index.')
        for entry in payload['surveys']:
            if (not isinstance(entry, dict) or not all(k in entry for k in (
                    'id', 'name', 'method', 'kind', 'frame', 'data', 'crs', 'created_at', 'fingerprint'))
                    or entry['kind'] not in ('mesh', 'em', 'grid', 'points', 'wells')
                    or entry['frame'] not in ('local', 'geographic')):
                raise ValueError('Invalid survey entry in the project map index.')
        return payload['surveys']

    def _write(self, entries):
        if self.read_only:
            raise PermissionError('This project is read-only.')
        _atomic_write_json(self.index, {'schema_version': 1, 'surveys': entries})

    def add(self, meta, arrays, xy, crs, name):
        if self.read_only:
            raise PermissionError('This project is read-only.')
        if not str(name).strip():
            raise ValueError('Enter a survey name.')
        mapped, frame = map_coordinates(xy, crs)
        if len(mapped) != np.size(arrays['distance']):
            raise ValueError('Map coordinates must match the profile samples.')
        copied = {k: np.asarray(v) for k, v in arrays.items()}
        copied.update(map_xy=mapped, source_xy=np.asarray(xy, dtype=float))
        if any(v.dtype.kind not in 'biuf' for v in copied.values()):
            raise ValueError('Map snapshots support numeric arrays only.')
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
        digest.update(frame.encode())
        for key, value in sorted(copied.items()):
            digest.update(key.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        identity = digest.hexdigest()
        entries = self.entries()
        existing = next((e for e in entries if e['fingerprint'] == identity), None)
        if existing:
            return existing
        survey_id = uuid.uuid4().hex
        entry = {**meta, 'id': survey_id, 'name': str(name).strip(), 'crs': str(crs),
                 'frame': frame, 'visible': True, 'created_at': _utc_now(),
                 'fingerprint': identity, 'data': f'{survey_id}.npz'}
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / entry['data']
        temporary = target.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **copied)
            os.replace(temporary, target)
            self._write(entries + [entry])
        finally:
            temporary.unlink(missing_ok=True)
        return entry

    def load(self, entry):
        path = (self.directory / entry['data']).resolve()
        if path.parent != self.directory.resolve() or path.suffix != '.npz':
            raise ValueError('Invalid map snapshot path.')
        with np.load(path, allow_pickle=False) as payload:
            return {k: payload[k] for k in payload.files}

    def update(self, survey_id, **changes):
        if set(changes) - {'name', 'visible'}:
            raise ValueError('Only name and visibility can be changed.')
        if 'name' in changes:
            changes['name'] = str(changes['name']).strip()
            if not changes['name']:
                raise ValueError('Survey name cannot be empty.')
        entries = self.entries()
        found = next(e for e in entries if e['id'] == survey_id)
        found.update(changes)
        self._write(entries)

    def remove(self, survey_id):
        # Detach from the map; source runs and the snapshot stay recoverable.
        self._write([e for e in self.entries() if e['id'] != survey_id])
