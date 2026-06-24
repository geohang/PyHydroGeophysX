# Geophysics → Hydro input data format

The **Data** step expects a single folder holding an **inverted ERT model
bundle** — the output of a (time-lapse) ERT inversion. The module does not run the
inversion itself; it converts an existing resistivity model to water content and
porosity. The bundled Treeline demo lives in `examples/results/Structure_WC`.

## Required files

| File | Format | Shape | Meaning |
|------|--------|-------|---------|
| `mesh_res.bms` | pyGIMLi mesh (`pg.load`) | `n_cells` | The inversion mesh the model lives on. |
| `resmodel.npy` | NumPy (`np.load`) | `(n_cells, n_time)` | Inverted resistivity (Ω·m) per cell, one column per time step. A 1D `(n_cells,)` file is treated as a single time step. |
| `index_marker.npy` | NumPy int (`np.load`) | `(n_cells,)` | Geological layer id per cell. Required for per-layer petrophysics; if absent, all cells are treated as one layer. |
| `all_coverage.npy` | NumPy (`np.load`) | `(n_time, n_cells)` or `(n_cells,)` | Sensitivity / coverage used to mask low-confidence regions. Optional. |

## Conventions

- **Layer ids** come from `index_marker.npy`, **not** from the mesh's intrinsic
  cell markers (those are often unique per cell and cannot identify layers). In
  the demo the markers are `3` (regolith) and `2` (fractured bedrock).
- **Resistivity** is in ohm-metres; columns are ordered by acquisition time.
- **Petrophysics** uses the Waxman-Smits / Archie relation. Each layer is given
  distributions (mean ± std) for cementation `m`, fluid resistivity `rho_fluid`,
  saturation exponent `n`, surface conductivity `sigma_sur`, and porosity `φ`.
- **Water content** θ = S·φ, where saturation S is solved from resistivity per
  realization. **Porosity** mode instead assumes a known saturation (1.0 in the
  saturated zone) and solves for φ.
- All arrays share the same `n_cells`; `resmodel` and `all_coverage` share the
  same `n_time`.

## Example (bundled Treeline demo)

```
mesh_res.bms        4515 cells
resmodel.npy        (4515, 12)     # 12 monitoring times
index_marker.npy    (4515,)        # markers {2, 3}
all_coverage.npy    (12, 4515)
```

Click **Use example / context data** in the Data step to load this dataset.

## Preparing your own data

```python
import numpy as np
import pygimli as pg

mesh.save("mesh_res.bms")                 # pyGIMLi inversion mesh
np.save("resmodel.npy", resistivity)      # (n_cells, n_time)
np.save("index_marker.npy", markers)      # (n_cells,) integer layer ids
np.save("all_coverage.npy", coverage)     # (n_time, n_cells), optional
```
