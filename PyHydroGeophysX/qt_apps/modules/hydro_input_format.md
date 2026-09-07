# Hydro → Geophysics input format

Select a folder containing these four NumPy arrays (`np.save`):

| File | Shape | Meaning |
|---|---|---|
| `Watercontent.npy` | `(n_time, n_layers, ny, nx)` or `(n_layers, ny, nx)` | Volumetric water content, fraction |
| `Porosity.npy` | `(n_layers, ny, nx)` | Porosity, fraction |
| `top.npy` | `(ny, nx)` | Ground elevation, metres |
| `bot.npy` | `(n_layers, ny, nx)` | Layer-bottom elevations, metres |

- Use matching grid dimensions and layer counts across files.
- Layers run from the surface downward; elevations increase upward.
  Interfaces are `top`, `bot[0]`, `bot[1]`, and so on.
- Water content and porosity are fractions, not percentages.
- Horizontal spacing is one unit per cell; profile distances use those units.
- For time-dependent water content, choose **Snapshot** in the Profile step.

Example: `Watercontent.npy` `(10, 14, 194, 157)`, `Porosity.npy` and `bot.npy`
`(14, 194, 157)`, and `top.npy` `(194, 157)`.
Use **Use example / context data** to load the available demo.
