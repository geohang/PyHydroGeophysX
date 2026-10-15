# Seismic → Structure input format

Provide inverted **2D velocity sections**; use Seismic Processing to invert
travel times first.

| Per-line input | Format / units |
|---|---|
| Mesh | pyGIMLi `velmesh.bms` |
| Velocity | `Vinvmodel.npy`: `(n_cells,)`, in **m/s**; a 2D array uses column 0 |
| Ends | Start `(x0, y0)` → end `(x1, y1)` in a shared map coordinate system; the start is the section's distance 0 |

- **Add a line…** loads `velmesh.bms` and `Vinvmodel.npy` from one folder.
  The file button beside it selects them one by one.
- Velocity values must match the mesh's cell count and order.
- Type each line's start and end in the table, in metres; the **Map** tab
  draws the lines where they are.
- Set the velocity at the top of bedrock and check it on each line's
  **Section**, then **Build 3D model**. Two or more non-parallel lines
  constrain the volume better than one line.
- More **Cells across** take longer to compute.

Outputs include a bedrock map, 3D structure, VTK velocity volume and NumPy
surfaces. **Use example** loads the available synthetic lines.
