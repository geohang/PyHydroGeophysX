# Seismic → 3D Model input data format

This module builds a 3D subsurface model (velocity volume + bedrock-interface
surface) from one or more **2D seismic velocity sections**. It does not invert the
travel times itself; use the **Seismic Processing** module (or your own workflow)
to produce each inverted velocity section first.

## Per-line inputs

Each seismic line needs three things:

| Item | Format | Meaning |
|------|--------|---------|
| Mesh | pyGIMLi mesh (`velmesh.bms`) | The 2D inversion mesh of the section. |
| Velocity | NumPy (`Vinvmodel.npy`) | Per-cell P-wave velocity (m/s), length = mesh cell count. A 2D `(n_cells, n_time)` array uses column 0. |
| Line geometry | four numbers | Map coordinates of the line endpoints: `(x0, y0)` → `(x1, y1)`. |

The **Add line (folder)…** button expects a folder holding `velmesh.bms` and
`Vinvmodel.npy` (the convention used by the bundled `seismic_example`).
**Add line (files)…** lets you pick the mesh and velocity array separately. The
endpoint columns in the table are editable so you can position each section in
map coordinates.

## How the 3D model is built

1. For each line, the **bedrock interface** is extracted where velocity crosses
   the threshold (default ~1200 m/s) using
   `Geophy_modular.seismic_processor.extract_velocity_structure`.
2. The profile distance of each section is mapped to map coordinates along its
   endpoints, giving 3D surface points, interface points, and velocity samples.
3. A 3D structured grid is built from the combined topography
   (`core.kriging_3d.create_3d_structured_grid`), and velocity is interpolated
   into the volume (scipy `griddata`, or `gstools` ordinary kriging when that
   optional package is installed).
4. Outputs: a depth-to-bedrock map, a 3D structure view, a VTK velocity volume
   (openable in ParaView or the Mesh 3D module), and `.npy` surfaces.

## Example (bundled)

`examples/results/seismic_example/` holds one inverted section
(`velmesh.bms`, `Vinvmodel.npy`). **Use example** adds it three times at map
`y = 0, 60, 120 m` so the 3D interpolation has multiple lines to work with.

## Tips

- Two or more non-parallel lines give a much better 3D interface than a single
  line. With one line the model is effectively a 2D section extruded in map view.
- Increase **Grid resolution** for a finer model (slower). Use the **Vertical
  scale** (advanced) to avoid over-smoothing thin vertical structure.
