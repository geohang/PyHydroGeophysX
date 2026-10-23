# Temperature correction of ERT models

Bulk resistivity falls by about 2 % per degC, so over a monitoring season the
temperature signal is as large as the moisture signal. Correcting every survey
to one reference temperature leaves the remaining change attributable to water.
It is post-processing: it never needs the inversion to be re-run.

## Settings (`DEFAULT_TEMPERATURE_SPEC`, `petrophysics/temperature.py`)

| Key | Default | Meaning |
|---|---|---|
| `enabled` | False | nothing is corrected unless asked |
| `model` | `hayley` | Hayley et al. (2007); or `linear` |
| `alpha` | 0.025 per degC | `linear` only (Campbell / Arps) |
| `reference` | 25.0 degC | temperature the corrected models are reported at |
| `mode` | `constant` | `constant`, `profile`, `surface`, `seasonal` |
| `value` | 15.0 degC | `constant` |
| `diffusivity` | 0.06 m2/day | `surface`; annual damping depth about 2.64 m |
| `mean_temperature`, `amplitude`, `damping_depth`, `peak_day` | 12, 10, 2.5 m, 200 | `seasonal` |

## Temperature sources

- **Profile**: a thermistor string, depth x time (`load_temperature_profiles`
  reads long, wide or transposed tables); interpolated in time then depth;
  below the deepest sensor its seasonal swing is damped with depth.
- **Surface**: a surface logger, or a nearby station's air temperature, driven
  through 1-D heat conduction (`simulate_temperature_1d`); only the
  diffusivity is chosen (soils about 0.04-0.13 m2/day).
- **Seasonal**: the analytical damped annual wave.
- **Constant**: one value for the whole model.
- There is no mode applying one temperature uniformly over depth.
- The record and the surveys must be on one clock: dates on both, or days on
  both. A dated record against undated surveys is refused.

## Where it is applied

- Studio: the Temperature correction panel beside the section (ERT page,
  Resistivity model tab, and Project -> Saved Results), Apply / Remove.
  Exports write the corrected model beside the inverted one
  (`final_models_temperature_corrected.npy`,
  `timelapse_resistivity_temperature_corrected.vtk`,
  `resistivity_model_temperature_corrected.npy`, `temperature_correction.json`).
- Script / automatic run: `inversion_params["temperature_correction"] = {...}`.
  The corrected models become the series; the raw ones are kept as
  `final_models_uncorrected.npy`. A correction that cannot be built is logged
  as a warning and the raw models are returned (`applied: False`).
- Every run states which it produced, because a corrected section looks the
  same as an uncorrected one.

## Sources

- `PyHydroGeophysX/petrophysics/temperature.py`
- `PyHydroGeophysX/agents/ert_inversion_agent.py` (`_correct_for_temperature`)
- `PyHydroGeophysX/inversion/_time_lapse_workflow.py` (`_apply_temperature_correction`)
- `docs/source/tutorials/time_lapse_monitoring.rst`
