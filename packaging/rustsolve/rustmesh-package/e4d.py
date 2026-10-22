"""Write E4D's TetGen-style mesh files from a :class:`rustmesh.Mesh3D`.

File specification: PNNL E4D Electrical Methods Mesh Files. No meshing
executable is called; the module only writes the mesh files E4D reads.
"""
import json
import re
from pathlib import Path
import numpy as np


def write_e4d(mesh, directory, name='mesh', *, surface_markers=(-1,),
              dirichlet_markers=(-2,), translation=None, conductivity=None,
              allow_incomplete=False, overwrite=False):
    """Export a Mesh3D to .1.node/.ele/.face/.neigh, .trn and provenance JSON.

    Exterior facet markers in ``surface_markers`` become no-flux (1); those in ``dirichlet_markers``
    become zero-potential (2). At their intersection, 2 takes precedence.
    Unknown exterior markers are errors. Original markers are retained in
    the JSON sidecar. Zones are contiguous 1-based material-marker IDs.
    Optional conductivity is positive S/m, scalar or one value per cell.
    Coordinates are shifted by ``translation`` (default: mean node position).
    Files preserve cell order. Existing output files require overwrite=True.
    """
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,25}', name):
        raise ValueError('E4D name must start with a letter and contain at most 26 ASCII letters, digits, underscores or hyphens')
    if not allow_incomplete and not mesh.report.get('complete', False):
        raise ValueError('mesh quality limits are unmet; inspect the report or pass allow_incomplete=True')
    nodes = np.asarray(mesh.nodes, dtype=float)
    tets = np.asarray(mesh.tetrahedra)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or not np.all(np.isfinite(nodes)):
        raise ValueError('nodes must be a finite (n,3) array')
    if tets.ndim != 2 or tets.shape[1] != 4 or len(tets)==0 or not np.issubdtype(tets.dtype, np.integer):
        raise ValueError('tetrahedra must be a nonempty integer (n,4) array')
    if np.any(tets<0) or np.any(tets>=len(nodes)) or np.any(np.diff(np.sort(tets,axis=1),axis=1)==0):
        raise ValueError('invalid tetrahedron vertex indices')
    if len(np.unique(np.sort(tets,axis=1),axis=0))!=len(tets):
        raise ValueError('duplicate tetrahedra')
    shift = nodes.mean(axis=0) if translation is None else np.asarray(translation, dtype=float)
    if shift.shape!=(3,) or not np.all(np.isfinite(shift)):
        raise ValueError('translation must contain three finite coordinates')
    local = nodes-shift
    p=local[tets]
    volume=np.einsum('ij,ij->i',np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6.
    if np.any(~np.isfinite(volume)) or np.any(volume<=0):
        raise ValueError('translated mesh contains nonpositive or nonfinite cells')
    # Reconstruct incidence independently; never write stale neighbor metadata.
    opposite=((1,2,3),(0,3,2),(0,1,3),(0,2,1))
    incidence={}
    for i,t in enumerate(tets):
        for k,face in enumerate(opposite):
            key=tuple(sorted(int(t[j]) for j in face))
            incidence.setdefault(key,[]).append((i,k))
    neighbors=np.full((len(tets),4),-1,dtype=np.int64)
    for entries in incidence.values():
        if len(entries)>2:
            raise ValueError('non-manifold tetrahedral face')
        if len(entries)==2:
            (a,ka),(b,kb)=entries
            neighbors[a,ka]=b
            neighbors[b,kb]=a
    if not np.array_equal(neighbors,mesh.neighbors):
        raise ValueError('neighbor metadata does not match tetrahedra')
    def marker_set(values):
        raw=np.asarray(list(values))
        if raw.size==0:
            return set()
        if raw.ndim!=1 or not np.issubdtype(raw.dtype,np.integer):
            raise ValueError('boundary marker sets must contain integers')
        return set(raw.tolist())
    surface=marker_set(surface_markers)
    fixed=marker_set(dirichlet_markers)
    if surface & fixed:
        raise ValueError('surface and Dirichlet marker sets must be disjoint')
    records={}
    for i,f in enumerate(mesh.faces):
        key=tuple(sorted(int(v) for v in f))
        if key in records or key not in incidence:
            raise ValueError('invalid or duplicate marked face')
        records[key]=i
    exterior={f for f,cells in incidence.items() if len(cells)==1}
    if not exterior.issubset(records):
        raise ValueError('exterior face missing its boundary marker')
    flags=np.zeros(len(nodes),dtype=int)
    face_flags=[]
    inner_markers=sorted({int(mesh.face_markers[i]) for key,i in records.items() if key not in exterior})
    interface_ids={marker:i+3 for i,marker in enumerate(inner_markers)}
    for f,marker in zip(mesh.faces,mesh.face_markers):
        key=tuple(sorted(int(v) for v in f))
        if key in exterior:
            if marker in fixed:
                flag=2
            elif marker in surface:
                flag=1
            else:
                raise ValueError(f'exterior marker {marker} has no E4D boundary condition')
            flags[f]=np.maximum(flags[f],flag)
        else:
            flag=interface_ids[int(marker)]
        face_flags.append(flag)
    if not np.any(flags==2):
        raise ValueError('at least one exterior Dirichlet boundary is required')
    markers=np.asarray(mesh.cell_markers)
    if markers.shape!=(len(tets),) or not np.issubdtype(markers.dtype,np.integer):
        raise ValueError('cell_markers must contain one integer per tetrahedron')
    levels,zones=np.unique(markers,return_inverse=True)
    zones=zones+1
    sigma=None
    if conductivity is not None:
        sigma=np.asarray(conductivity,dtype=float)
        if sigma.ndim==0:
            sigma=np.full(len(tets),float(sigma))
        if sigma.shape!=(len(tets),) or np.any(~np.isfinite(sigma)) or np.any(sigma<=0):
            raise ValueError('conductivity must be positive finite S/m, scalar or one value per cell')
    directory=Path(directory)
    base=directory/f'{name}.1'
    paths={ext:Path(f'{base}.{ext}') for ext in ('node','ele','face','neigh')}
    paths.update(trn=directory/f'{name}.trn',metadata=directory/f'{name}.rustmesh.json')
    if sigma is not None:
        paths['sig']=directory/f'{name}.sig'
    if not overwrite:
        occupied=[str(p) for p in paths.values() if p.exists()]
        if occupied:
            raise FileExistsError('E4D export would overwrite: '+', '.join(occupied))
    lines={}
    lines['node']=[f'{len(nodes)} 3 1 1']+[
        f'{i+1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} 1 {flags[i]}' for i,p in enumerate(local)]
    lines['ele']=[f'{len(tets)} 4 1']+[
        f'{i+1} '+ ' '.join(str(int(v)+1) for v in t)+f' {zones[i]}' for i,t in enumerate(tets)]
    lines['neigh']=[f'{len(tets)} 4']+[
        f'{i+1} '+' '.join(str(int(v)+1 if v>=0 else -1) for v in row) for i,row in enumerate(neighbors)]
    lines['face']=[f'{len(mesh.faces)} 1']+[
        f'{i+1} '+' '.join(str(int(v)+1) for v in face)+f' {face_flags[i]}' for i,face in enumerate(mesh.faces)]
    lines['trn']=[' '.join(f'{x:.17g}' for x in shift)]
    if sigma is not None:
        lines['sig']=[f'{len(sigma)} 1']+[f'{x:.17g}' for x in sigma]
    metadata=dict(format_version=1,translation=shift.tolist(),
                  zone_of_marker={str(int(m)):i+1 for i,m in enumerate(levels)},
                  interface_id_of_marker=interface_ids,
                  node_markers=np.asarray(mesh.node_markers).tolist(),
                  input_vertex_nodes=np.asarray(mesh.input_vertex_nodes).tolist(),
                  cell_regions=np.asarray(mesh.cell_regions).tolist(),
                  face_markers=np.asarray(mesh.face_markers).tolist(),
                  face_facets=np.asarray(mesh.face_facets).tolist(),
                  surface_markers=sorted(surface),dirichlet_markers=sorted(fixed),
                  report=mesh.report)
    # Prepare every payload before creating files, so argument errors leave no
    # partial output. Filesystem I/O errors still propagate to the caller.
    payload={key:'\n'.join(value)+'\n' for key,value in lines.items()}
    payload['metadata']=json.dumps(metadata,indent=2)+'\n'
    directory.mkdir(parents=True,exist_ok=True)
    for key,value in payload.items():
        paths[key].write_text(value,encoding='utf-8')
    return dict(files=paths,node_file=paths['node'].name,translation=shift,zones=zones,
                zone_of_marker={int(m):i+1 for i,m in enumerate(levels)})
