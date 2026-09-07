# Seismic → Structure input format

Provide inverted **2D velocity sections**; use Seismic Processing to invert
travel times first.

| Per-line input | Format / units |
|---|---|
| Mesh | pyGIMLi `velmesh.bms` |
| Velocity | `Vinvmodel.npy`: `(n_cells,)`, in **m/s**; a 2D array uses column 0 |
| Endpoints | `(x0, y0)` → `(x1, y1)` in a shared map coordinate system |

- **Add line (folder)…** loads `velmesh.bms` and `Vinvmodel.npy` together.
  **Add line (files)…** selects them separately.
- Velocity values must match the mesh's cell count and order.
- Edit endpoints in the table to locate each section. Use consistent metre units.
- Set the bedrock velocity threshold, then **Build 3D model**. Two or more
  non-parallel lines constrain the volume better than one line.
- Finer **Grid resolution** increases computation time.

Outputs include a bedrock map, 3D structure, VTK velocity volume and NumPy
surfaces. **Use example** loads the available synthetic lines.
