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
- For time-dependent water content, choose the **Time step** in the
  Hydrologic model panel.

- Mark cells outside the model domain with NaN in `bot.npy` (and in
  `Watercontent.npy`), as MODFLOW's inactive cells are. The maps leave them
  blank, and their `top` is ignored (MODFLOW writes 0 there): a profile that
  crosses them takes its ground surface from the active cells beside them.

Example: `Watercontent.npy` `(10, 14, 194, 157)`, `Porosity.npy` and `bot.npy`
`(14, 194, 157)`, and `top.npy` `(194, 157)`.
**Use example / context data** loads this MODFLOW catchment model, the one
`examples/Ex_ERT_workflow` uses, on its snapshot 5 and its profile
(115, 70) → (95, 180).
