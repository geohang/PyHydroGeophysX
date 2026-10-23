# TDEM inversion: details

## Readers (`docs/source/methods/em.rst`, `data_and_processing.rst`)

| Source | Notes |
|---|---|
| TEM2Go project `.tiw` or `.db` | station stacks, gate flags, errors, geometry, saved inversion settings |
| TEM2Go acquisition folder `.stb` | straight off the instrument; gates, calibration and loop from the raw stream |
| `_StationData.xyz`, `_RawData.xyz` | self-describing; station stacks preferred |
| tTEM raw `.skb` with `.sps` | system GEX for geometry, waveform, gates, calibration; TFI import filter when present |
| `.csv` column tables | airborne and synthetic, with a geometry file |

`TDEMAgent._is_instrument_survey` sends to the survey path: a TEMcompany
source, a tTEM source, a stored sounding container, or any folder. A folder
holding several projects is read from the standard name (`project.tiw` before
`project2.tiw`); `source_file` names the file actually read.

## Survey path (`_run_survey_inversion` -> `em1d.invert_line`)

Settings, later ones winning:
1. `preset_inversion("ground_tem")` = `INVERSION_PRESETS["ground_tem"]` over
   `DEFAULT_INVERSION`;
2. the project's stored inversion settings (`inversion_defaults`);
3. a uniform protocol error (`protocol.uniform_std`) as `rel_error`;
4. any `DEFAULT_INVERSION` key given in `tdem_params` (`overrides`);
   `starting_conductivity` sets `starting_resistivity` and turns off
   `auto_starting_model`.

`lines` and `max_soundings` choose the data. `settings_source` says
`project` or `ground_tem preset`.

### `ground_tem` values (with the reason the code records)

- `n_layers` 20, `min_thickness` 1.0, `max_thickness` 14.74 m: a ground
  system's gates span about a hundred metres of diffusion depth.
- `smoothness` 1.5, `lateral_smoothness` 1.3, `auto_lambda` false,
  `scale_bounds` (0.5, 2.0): a median of four or five gates per sounding
  against twenty layers; chasing chi2 = 1 fits noise.
- `robust_errors` true, `robust_threshold` 3.0, `robust_passes` 3,
  `robust_max_error_factor` 10, `robust_min_unchanged_fraction` 0.70,
  `robust_target_chi2` 1.75; `reject_outliers` false: large residuals get a
  larger effective error, no gate is deleted.
- `shallow_prior_enabled` true, mode `signal_threshold`, factor 2.0.
- `min_gates_per_sounding` 3, `outlier_passes` 2, `min_data_fraction` 0.8.
- `min_rel_error` 0.03, `rel_error` 0.06.
- `rho_max` 1e4 ohm-m (`rho_min` 1.0 from the defaults): a deep layer the data
  cannot resolve walks to the bound, so a cell on it reports the bound.
- `doi_threshold` 6.0 (cumulative sensitivity), `lci_ftol` 1e-3.

From `DEFAULT_INVERSION` unchanged: `lci_mode` `simultaneous` (acts only when
`lateral_smoothness` > 0), `model_damping` 0.4, `max_iterations` 30,
`target_chi2` 1.0, `chi2_tolerance` 0.2.

Result keys: `n_soundings`, `failed_soundings` (no data inverted),
`unresolved_soundings` (no cell above the depth of investigation), `lines`,
`lci_mode`, `chi2` (global), `chi2_sounding_median`, `chi2_list`, `doi`,
`recovered_resistivity` (surface first, NaN below each DOI), `thicknesses`,
`system`, `protocol`, `inversion_settings`, `robust` (`enabled`,
`downweighted`, `n_start`, `chi2_effective`).

## Text sounding (`_run_inversion`)

`TDEMInversion` on `.dat`, `.txt`, `.csv`: `source_radius` 10 m (circular
loop), step-off, `n_layers` 20, `min_thickness` 0.5, `max_thickness` 10.0,
`starting_conductivity` 0.001 S/m, `use_irls` true, `max_iterations` 50;
`sounding` index 0. A TEM2Go system (0.63 m square loop, four turns, receiver
about 15 m away) cannot be modelled this way.

## Gate QC (`data_processing/em1d.py`)

`GATE_STATUS`: `kept`; `dummy` (non-finite, non-positive time, or a fill
value); `flagged out` (the file's InUseFlags); `noisy` (relative stack error
above `max_relative_std`; `None` disables only that test); `after a noisy one`
(carried out by truncation); `reversed sign` (negative, with the sign test on).
`GATE_REJECTION_MODES`: `truncate` (default; the first noisy gate and all later
ones) or `individual`. `gate_report(path, sounding, moment=...)` returns every
gate of a station with its verdict.

## Evaluation (`evaluate_tdem`)

| Component | Weight | Rule |
|---|---|---|
| Data fit | 0.55 | 0.6 x `chi2_fit_score`(median sounding chi2) + 40 x share of soundings with chi2 <= 2.0 |
| Resolution | 0.25 | share of soundings neither failed nor unresolved |
| Plausibility | 0.2 | share outside 0.1-1e5 ohm-m: 100 - 200 x share |

Threshold 70; data fit at least 60. Advice when under 80 % of soundings are
within chi2 2.0, when soundings are blank, and when more than 2 % of the model
is outside the range. No retry: the line inversion has its own smoothness
search, which `ground_tem` turns off on purpose; to search anyway set
`auto_lambda` true in `tdem_params`.

## After the inversion

- `map_tdem_plan_view`: only when the request asks for a spatial distribution
  of a survey; kriged resistivity at several depths with a kriging-uncertainty
  map for each, over the survey's basemap.
- `convert_tdem_water_content`: only when the run has no ERT survey; the
  layered model is one unit; `water_content_by_layer.csv`.

## Sources

- `PyHydroGeophysX/agents/tdem_agent.py`
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_load_tdem`, `_invert_tdem`, `_evaluate_tdem`, `_map_tdem_plan_view`, `_convert_tdem_water_content`)
- `PyHydroGeophysX/inversion/em1d.py`
- `PyHydroGeophysX/data_processing/em1d.py`
- `PyHydroGeophysX/agents/_method_evaluation.py` (`evaluate_tdem`)
- `examples/Ex_TEM_LMHM_LCI.py`
- `docs/source/methods/em.rst`, `docs/source/data_and_processing.rst`
