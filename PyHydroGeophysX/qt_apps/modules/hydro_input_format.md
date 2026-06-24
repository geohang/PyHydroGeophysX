# Hydro → Geophysics input data format

The **Data** step expects a single folder holding four NumPy files that describe a
gridded hydrologic model. The grid is a regular map of `ny` rows by `nx`
columns with `n_layers` model layers.

## Required files

| File | Format | Shape | Meaning |
|------|--------|-------|---------|
| `Watercontent.npy` | NumPy (`np.load`) | `(n_time, n_layers, ny, nx)` or `(n_layers, ny, nx)` | Volumetric water content θ. A 4D file holds one map per time step. |
| `Porosity.npy` | NumPy (`np.load`) | `(n_layers, ny, nx)` | Porosity φ of each layer. |
| `top.npy` | NumPy (`np.load`) | `(ny, nx)` | Ground-surface elevation at each grid cell. |
| `bot.npy` | NumPy (`np.load`) | `(n_layers, ny, nx)` | Bottom elevation of each model layer. |

All four files use the same NumPy binary format (`np.save` / `np.load`).

## Conventions

- **Layer order** runs from the surface downward. The interfaces used for
  meshing are `[top, bot[0], bot[1], ...]`, so `top` is the surface and each
  `bot[i]` is the bottom of layer `i`.
- **Elevations** are in metres, increasing upward.
- **Grid spacing** is treated as one unit per cell (`pixel_width = 1`,
  `pixel_height = -1`); profile distances are reported in those units.
- **Water content** is a fraction (0–1). For a 4D `Watercontent.npy`, pick the
  time step with the **Snapshot** control in the Profile step.
- All four files share the same `ny`, `nx`, and `n_layers`.

## Example (bundled Treeline demo)

```
Watercontent.npy  (10, 14, 194, 157)   # 10 time steps, 14 layers
Porosity.npy      (14, 194, 157)
top.npy           (194, 157)
bot.npy           (14, 194, 157)
```

Click **Use example / context data** in the Data step to load this dataset.

## Preparing your own data

```python
import numpy as np

np.save("Watercontent.npy", water_content)   # (n_time, n_layers, ny, nx)
np.save("Porosity.npy", porosity)            # (n_layers, ny, nx)
np.save("top.npy", top)                      # (ny, nx)
np.save("bot.npy", bot)                       # (n_layers, ny, nx)
```
