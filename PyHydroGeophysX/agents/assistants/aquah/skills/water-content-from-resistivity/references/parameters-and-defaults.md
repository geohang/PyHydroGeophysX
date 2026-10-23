# Water content from resistivity: parameters and defaults

## The relationship (`run_petrophysics_monte_carlo`, `resistivity_models.py`)

Per layer, one parameter draw per realisation, shared by the layer's cells and
time steps:

- Archie route: `rho_sat = a * rho_fluid * porosity^(-m)`, `a` = 1
  (`tortuosity_a`).
- Direct route (`use_rho_sat`): a saturated resistivity `rho_sat` replaces m and
  rho_fluid.
- Waxman-Smits: `1/rho = S^n / rho_sat + sigma_sur * S^(n-1)`, solved for S;
  water content = S x porosity.

Outputs: `water_content_mean`, `water_content_std`, `water_content_p10`,
`_p50`, `_p90`, `saturation_mean`, `saturation_std`, plus `params_used`,
`petrophysical_relationship`, `prior_ranges`, `prior_statement`, `layering`.

## Distributions (`PetrophysicsAgent`)

With `petrophysical_params` (information level `high`):

| Given as | Distribution |
|---|---|
| a value | mean = value, std = 5 % of it (`rho_sat`: 0.5 %) |
| `[low, high]` | mean = midpoint, std = (high - low) / 4 |
| not given, beside given ones | default mean, std = max(0.5 x mean, floor); floors porosity 0.08, n 0.3, m 0.3 |
| `rho_fluid` not given | fixed at 20 ohm-m |

Default means then (`DEFAULT_LAYER_PARAMS`): first layer `regolith` m 1.3,
n 2.1, porosity 0.42, sigma_sur 1/200 +/- 1/200 S/m; deeper layers `bedrock`
m 1.9, n 1.7, porosity 0.25, sigma_sur 0.

With `layer_params` per named layer: `rho_sat_range`, `n_range`,
`porosity_range` (mean = midpoint, std = range/4); sigma_sur 0 +/- 0.001 when
not given; anything missing takes the layer's default, each substitution
noted. Names map to mesh markers: `regolith` 2, `fractured_bedrock` and
`bedrock` 3, `background` 1 (from `add_velocity_interface`).

With nothing given (`_guess_params_from_geology`, `geological_context` default
"generic watershed"):

- first layer m 1.5, n 2.0, porosity 0.40, sigma_sur 1/200 S/m, rho_fluid 20;
  deeper layers m 1.8, n 2.0, porosity 0.22, sigma_sur 1/400;
- keywords replace these: sand/sandstone; clay/shale/mud; carbonate/limestone/
  dolomite; fractured/bedrock/granite/basalt; soil/regolith/weathered;
- `low` information (the default; any context containing "generic", or 50
  characters or fewer): std = max(mean, 0.7) for m and n, max(porosity, 0.15),
  sigma_sur std max(sigma_sur, 1/150);
- `medium` (a context over 50 characters without "generic"): std =
  max(0.5 x mean, 0.3), porosity floor 0.08, sigma_sur std max(sigma_sur, 1/250).

Sampler bounds (`_sample_layer`): m and n clipped to `EXPONENT_BOUNDS`
(1.0, 4.0); porosity to 0.01-0.9; rho_fluid and rho_sat at least 1e-6;
sigma_sur at least 0.

## The statement (`describe_prior`)

- Core parameters: `rho_sat`, `m`, `rho_fluid`, `n`, `porosity`
  (`sigma_sur` is not core).
- `user`: every core parameter each layer's route uses was supplied (the
  direct route uses `rho_sat`, `n`, `porosity`; the Archie route `m`,
  `rho_fluid`, `n`, `porosity`). `partial`: some. `default`: none.
- Ranges are the 5th-95th percentiles of the draws actually made.
- `default`: "Water content was produced, but it is not reliable: no
  petrophysical relationship was given ..." with the ranges, and a request
  for the site's Archie m, n and porosity or a saturated resistivity.
- `partial`: names the parameters that took generic defaults.
- The run adds the statement to the warnings and the step summary
  (`_state_prior`): "Not reliable: ..." or "Partly on default ...".
- `petrophysics_dropped` lists request values removed by
  `drop_unstated_petrophysics` (a value counts as stated only when its number,
  or the percentage it is a fraction of, appears in the request).

## Realisations (`realizations`)

`n_realizations` default 100; below `MIN_REALIZATIONS` 50 it is raised, with a
note. Seed 7. The agents guide's best practice: `n_realizations` >= 1000 to
quantify uncertainty (`docs/source/agents/workflows.rst`).

## Layers (`resolve_layers`)

Markers are layers only with at least two values, each covering at least
`MIN_CELLS_PER_LAYER` (10) cells; otherwise one unit. The `layering` sentence
says which. TDEM and MT layered models are converted as one unit
(`_layered_water_content`); cells without a resistivity (below a station's
depth of investigation) stay NaN.

## Reliability (`water_content_reliability`)

ratio = mean std / (max - min of the mean values). >= 0.5: not usable;
0.25-0.5: qualitative only; < 0.25: contrasts resolved. Not calibrated
(relationship not `user`): the sentence adds that the parameters were not
calibrated for the site. All std zero: "No uncertainty was computed".
`result_caveats` turns an unusable ratio into a run warning.

## Failure modes

- Non-finite or non-positive resistivity: `run_petrophysics_monte_carlo`
  raises `ValueError`.
- A time-lapse step that fails ends the loop; the steps before it are kept and
  the failure noted.
- Layer parameters for a model without structural layers: not applied, said so.
- Porosity (not water content) draws with no Waxman-Smits root, when
  sigma_sur alone explains more conduction than measured: NaN, warned.

## Sources

- `PyHydroGeophysX/agents/petrophysics_agent.py`
- `PyHydroGeophysX/agents/_uncertainty.py`
- `PyHydroGeophysX/agents/context_input_agent.py` (`drop_unstated_petrophysics`)
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_state_prior`, `_convert_water_content`, `_layered_water_content`)
- `PyHydroGeophysX/petrophysics/monte_carlo.py`, `PyHydroGeophysX/petrophysics/resistivity_models.py`
- `docs/source/agents/workflows.rst`, `docs/source/methods/petrophysics.rst`
