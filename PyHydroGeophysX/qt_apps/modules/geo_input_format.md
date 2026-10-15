# ERT → Water Content input format

Select a folder containing an **already inverted ERT model**.

| File | Shape | Required / meaning |
|---|---|---|
| `mesh_res.bms` | pyGIMLi mesh | Required inversion mesh |
| `resmodel.npy` | `(n_cells, n_time)` or `(n_cells,)` | Required resistivity in Ω·m |
| `index_marker.npy` | `(n_cells,)` | Optional integer geological layer IDs; omitted = one layer. Read as layers only when each holds at least 10 cells; one marker per cell (an inversion mesh's cell numbers) is one layer |
| `all_coverage.npy` | `(n_time, n_cells)` or `(n_cells,)` | Optional sensitivity / coverage mask |

- An ERT inversion's output folder opens as it is: `final_models.npy` with
  `timelapse_mesh.bms` (time-lapse) or `resistivity_model.npy` with
  `resistivity_mesh.bms` (one survey) stand in for the two required files.
- Model rows must match mesh cell order; time columns follow acquisition order.
- Layer-specific petrophysics uses `index_marker.npy`, not mesh cell markers.
- Save arrays with `np.save`; all files must agree on cell and time counts.
- Set each layer's petrophysical parameters in the panel. Water content is
  θ = saturation × porosity; porosity mode assumes a specified saturation.

Example: a 4,515-cell model with 12 times has `resmodel.npy` shape `(4515, 12)`
and `all_coverage.npy` shape `(12, 4515)`.
**Use example** loads the Treeline time-lapse model of `examples/Ex_MC_Hydro`
(`examples/results/Structure_WC`): 12 surveys over regolith (marker 3) and
bedrock (marker 2).
