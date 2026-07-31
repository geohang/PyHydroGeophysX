# Packaged Qt Example Data

Compact datasets shipped inside the wheel so the workbench "Use example" buttons
work in an installed environment, without a source checkout. Each module prefers
a full-resolution dataset under `examples/results/` when one is present and falls
back to the copies here.

`examples/results/` is git-ignored, so these copies are the only example data
that reaches an installed package.

## Contents

| Path | Used by | Files |
|---|---|---|
| `geo_hydrology/` | `qt_apps/modules/geo_hydrology.py` (ERT to Water Content) | `mesh_res.bms`, `resmodel.npy`, `index_marker.npy`, `all_coverage.npy` |
| `seismic3d/line1..3/` | `qt_apps/modules/seismic3d.py` (Seismic to Structure) | `velmesh.bms`, `Vinvmodel.npy` |

`geo_hydrology/` holds an inverted ERT bundle of 3822 cells over 3 timesteps,
with cell markers 3 (regolith) and 2 (bedrock). `seismic3d/` holds three 2D
velocity sections whose values straddle the 1200 m/s regolith/bedrock threshold
the seismic module uses by default.

## How These Files Were Produced

Both datasets come from one run of the `timelapse_infiltration` scenario in
`examples/generate_synthetic_examples.py`, which drives the real
`Hydro_modular.hydro_to_ert` and `Hydro_modular.hydro_to_srt` forward workflows.
The ERT bundle and the seismic sections are therefore consistent slices of the
same synthetic hydrology model.

To regenerate the full-resolution sources and refresh these copies:

```bash
python examples/generate_synthetic_examples.py
cp examples/results/synthetic_Structure_WC/* PyHydroGeophysX/data/qt_examples/geo_hydrology/
for i in 1 2 3; do cp examples/results/synthetic_seismic/line$i/* PyHydroGeophysX/data/qt_examples/seismic3d/line$i/; done
```

Regenerating changes the cell count, so update the expected values in
`tests/test_coupled_example_data.py` at the same time.

## Packaging

The globs that ship these files live in `pyproject.toml` under
`[tool.setuptools.package-data]`. Adding a new example directory here requires a
matching glob there, because setuptools does not recurse into subdirectories.
