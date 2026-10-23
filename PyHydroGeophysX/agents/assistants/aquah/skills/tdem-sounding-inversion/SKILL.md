---
name: tdem-sounding-inversion
description: >
  Use when a request names TDEM, TEM, TEM2Go, TEMcompany, tTEM or a
  .tiw/.db/.stb/.skb file, or when reviewing a TEM section. Covers the reader
  and system model, LM and HM moments, gate QC, the ground_tem laterally
  constrained (LCI) settings, depth of investigation and evaluation.
methods: [tdem]
---

# TDEM soundings and laterally constrained sections

Tools: `load_tdem_data` -> `invert_tdem` -> `evaluate_tdem_inversion`, then
`map_tdem_plan_view` and `convert_tdem_water_content` when asked. Config keys:
`tdem_file` (or `em_file`), `tem_moment`, `tdem_params`.

1. **Path.** A TEM2Go/TEMcompany project (`.tiw`, `.db` or its folder), a raw
   `.stb` folder, `*_StationData.xyz`, a tTEM `.skb` with `.sps`, or any folder:
   every station is modelled with the system the file records and all are
   inverted together (`invert_line`). A `.dat`, `.txt` or `.csv` is one
   sounding with a 10 m circular loop and a step-off waveform; never send
   instrument data that way.
2. **Moment.** `LM+HM` by default, fitted jointly; a one-moment survey falls
   back to `HM` unless the user named a moment. State the moment used.
3. **Gates.** Each gate is `kept`, `dummy`, `flagged out` (the file's own
   flags), `noisy` (`max_relative_std`, off unless set), `after a noisy one` or
   `reversed sign`. `gate_rejection` `truncate` (default) ends a decay at the
   first bad gate; `individual` drops only the bad ones. Look at the decay
   figure first; `gate_report` explains why a station kept few gates.
4. **Settings.** The `ground_tem` preset, then the project's stored settings,
   then `tdem_params` overrides (`settings_source` and `overrides` say which
   applied). Key values: 20 layers, smoothness 1.5, lateral smoothness 1.3,
   `auto_lambda` false, `rel_error` 0.06, robust errors on, `rho_max` 1e4
   ohm-m. The smoothness is fixed because four or five gates against twenty
   layers, chasing chi2 = 1, fit noise. Turning `auto_lambda` on is the user's
   decision; explain that trade first.
5. **Evaluate** (threshold 70; no automatic retry). Data fit (weight 0.55):
   median sounding chi2 (target 1, 0.8-1.5) and the share of soundings at chi2
   2.0 or below; under 80 %, check the poorly fitted decays for noisy late
   gates or coupling and re-invert without those gates. Resolution (0.25):
   failed and unresolved soundings are blank. Plausibility (0.2): more than 2 %
   of the model outside 0.1-1e5 ohm-m is set by the regularisation, not the
   data. Report the robust weighting's `downweighted` gates.
6. **Read the section.** Cells below each station's depth of investigation are
   NaN: do not interpret or convert them. Deep cells sitting on `rho_max`
   report the bound, not the ground. The models are 1D and stitched along the
   line; strong lateral change within a sounding's footprint is for ERT to
   resolve.
7. **Products.** Plan-view maps only when a spatial distribution is asked
   for; show each depth with its kriging-uncertainty map. Water content only
   when the run has no ERT survey: one unit, with uncertainty (skill
   `water-content-from-resistivity`).

Failure: "resolved no cell above the depth of investigation at any station"
means no usable model; report it rather than invert again blindly.

Details (readers, all ground_tem values and reasons, result keys, gate QC,
scoring): references/tdem-details.md

## Sources

Under `PyHydroGeophysX/`: `agents/tdem_agent.py`, `agents/runtime/catalog.py`,
`inversion/em1d.py`, `data_processing/em1d.py`, `agents/_method_evaluation.py`.
Also `examples/Ex_TEM_LMHM_LCI.py`, `docs/source/methods/em.rst`,
`docs/source/data_and_processing.rst`.
