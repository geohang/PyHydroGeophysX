---
name: water-content-from-resistivity
description: >
  Use whenever a request asks for water content, moisture or saturation from
  ERT, TDEM or MT resistivity, or when reporting one. Covers the Archie /
  Waxman-Smits Monte Carlo conversion, its parameters and defaults, and the
  rule: always with uncertainty; not reliable without the user's relationship.
methods: [ert, tdem, mt]
---

# Water content from resistivity

**The rule.** Every water content carries its uncertainty (the Monte
Carlo standard deviation). Without a petrophysical relationship from the user
the result is produced but flagged **not reliable**, naming the parameters used
and their ranges; with a partial one, the parameters that took defaults are
named. The code writes this (`describe_prior`; the step summary says "Not
reliable: ..."). Never drop, soften or reword away that warning.

1. **Which model.** Convert only an evaluated model (`convert_water_content`
   needs `evaluation_results`). With a seismic line beside one ERT survey, wait
   for the structure-constrained model. TDEM (`convert_tdem_water_content`, only
   when the run has no ERT survey) and MT models are converted as one unit, and
   cells below the depth of investigation stay NaN.
2. **Relationship.** Archie gives rho_sat = rho_fluid x porosity^-m (or a given
   `rho_sat` is used directly); Waxman-Smits solves 1/rho = S^n / rho_sat +
   sigma_sur x S^(n-1); water content = S x porosity. Core parameters: `rho_sat`,
   `m`, `rho_fluid`, `n`, `porosity`.
3. **Ask.** If none was given, ask for Archie m, n and porosity, or a saturated
   resistivity, from cores, logs or published values for the formation.
   Proceeding on defaults is the user's decision, and the flag stays.
4. **Inputs.** `petrophysical_params` (a value or `[low, high]`) applies to the
   whole model. `layer_params` (`rho_sat_range`, `n_range`, `porosity_range`)
   applies only when the model has structural layers (at least 2 markers of at
   least 10 cells each, as a seismic interface draws); otherwise it is not
   applied and the run says so. A value parsed from a request counts only when
   its number appears in the request (`drop_unstated_petrophysics`). Never
   present a value you chose as the user's.
5. **Draws.** `n_realizations` default 100, never below `MIN_REALIZATIONS` 50;
   the agents guide recommends 1000 or more to quantify uncertainty. State the
   number used.
6. **Report** mean +/- std (p10, p50, p90 where useful), the prior statement
   with its 5-95 % ranges, the layering sentence and the reliability sentence.
7. **Reliability** (`water_content_reliability`; ratio = mean std / range of
   the means): 0.5 or more, no moisture pattern is established; 0.25 to 0.5,
   read only differences much larger than the error bar and quote no single
   value; below 0.25, contrasts can be compared. A std of exactly zero means no
   uncertainty was computed, not certainty. A change between surveys smaller
   than its std is not resolved.

Details (distributions, defaults with no input, sampler bounds, the statement,
failure modes): references/parameters-and-defaults.md

## Sources

Under `PyHydroGeophysX/`: `agents/petrophysics_agent.py`,
`agents/_uncertainty.py`, `agents/context_input_agent.py`,
`agents/runtime/catalog.py`, `petrophysics/monte_carlo.py`,
`petrophysics/resistivity_models.py`. Docs:
`docs/source/agents/workflows.rst`, `docs/source/methods/petrophysics.rst`.
