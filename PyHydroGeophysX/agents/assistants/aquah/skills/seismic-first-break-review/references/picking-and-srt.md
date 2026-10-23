# First-break picking and SRT: details

## Studio Seismic page actions (`agent_describe`)

- `load_data`: SEG-Y `.sgy`/`.segy`, Geometrics `.dat`, SEG-2, or a
  `.npy`/`.csv` matrix.
- `load_geometry`: per-geophone positions and topography, columns
  `station distance_m elevation_m` or `x z`; applies receiver x and elevation,
  re-stamps existing picks, and the SRT inversion honours topography.
- `set_geometry`: `spacing`, `geophone_start`, `first_shot_x`,
  `shot_spacing`, `shot_x` (per-record override, may be negative).
- `pick_next_shot`: selects the next un-picked record, auto-picks it and
  returns `review_picks`' answer: `status` `awaiting_user`, `current_record`,
  `picks`, `auto`, `manual`, `suspect_traces`, `records_total`,
  `records_picked`, `records_remaining`, `next_record`, and `resume`
  (`pick_next_shot`, or `run_srt` when none remain). It turns Manual pick mode
  on so the user can click or Ctrl+drag picks.
- `set_pick` (`trace`, `time_s` in seconds), `delete_pick` (`trace`),
  `list_picks` (with suspect traces), `clear_picks`.
- `auto_pick_all`: picks every record at once and screens them; its own
  description says to follow with `review_picks`. The persona's procedure is
  `pick_next_shot`.
- `load_traveltime`: pyGIMLi/BERT `.sgt`/`.dat`, or CSV/text with
  `source_x, receiver_x, time` (or with `source_z`, `receiver_z`); then
  `run_srt`. `clear_traveltime` goes back to the picks.
- `run_srt`: inverts uploaded travel times if any, otherwise the picks; needs
  at least 8 picks.
- `set_params` picking keys: `pick_threshold` (fraction of each trace's peak,
  0.05-0.95, default 0.2), `pick_max_time_ms` (latest first arrival, default
  150), `screen_picks` (default on). Inversion keys: `engine`
  (pyhydro/pygimli), `lam`, `max_iterations`, `max_total_iterations`,
  `plateau_tolerance`, mesh keys, and `auto_lambda`, `target_chi2`,
  `chi2_tolerance`, `max_lambda_trials` (in-house engine only).

## Suspect traces (`_suspect_pick_traces`)

A straight line of pick time against offset is fitted to the record's picks;
a pick whose residual is more than 3 x 1.4826 x MAD from the median residual is
flagged. Needs at least 4 picks.

## Picker and screening (`pick_and_correct`, `screen_picks`)

- Threshold picks on traces with AGC: `threshold` 0.2, `agc_window` 0.05 s,
  `noise_multiplier` 5.0, `max_time` 0.15 s, `polarity` 1.0; `repick` on.
- Picks that stray from their shot's first-arrival curve are picked again
  along it (AIC picker, Maeda 1985); what still breaks the curve (a first
  arrival earlier farther from the shot) is left out (`monotonic_check`).
- `reciprocal_shot_check` (`reciprocity_check`, default on): for each shot the
  median difference between its times and their reciprocals is compared with
  the survey's typical difference; beyond `factor` 3.0 times that and at least
  `floor_s` 2 ms, the whole shot is dropped (a trigger that fired early or
  late). Shots with no reciprocal pair (off the end of the spread) are kept.
- `neighbour_shot_check` (`neighbour_check`, default on): a pick out of line
  with the neighbouring shots' picks at the same geophone (`tolerance_s`
  1.5 ms or `relative` 10 %) is picked again where the neighbours put it, or
  left out.
- Traces with no timeable arrival are not in the travel times; the warning
  counts them.

## Automatic run (`pick_first_breaks`)

Runs when the configuration has a SEG-Y file (`raw_seismic_file`, or a
`.sgy`/`.segy` `seismic_file`) and no travel-time file. Outputs:
`first_break_picks.csv`, `seismic_traveltime_from_segy.dat`, the picks figure
and the shot-gathers figure (wiggles with their picks), `dropped_shots`,
`rejected_picks`, `repicked_picks`, `neighbour_rejected`,
`neighbour_repicked`, and the pick and geometry warnings. When the coordinate
file and the SEG-Y headers place the line apart, the run asks:
`profile` (coordinate file's origin), `segy` (header origin) or `stop`.
`first_break_params` passes the picker settings above.

## Inversion (`SeismicAgent.execute`)

`lam` 50, `zWeight` 0.2, mesh `quality` 32. Where not given, scaled to the
survey (`survey_scaled_defaults`): `paraDepth` 0.4 x line length (at most
30 m), `paraMaxCellSize` two sensor spacings (at most 2 m), `vTop` the
direct-wave velocity from offsets of two spacings or less (at most 500 m/s),
`vBottom` 1.5 x the far-offset apparent velocity (at least 2 x vTop, at most
5000 m/s), `limits` from min(300, half the direct-wave velocity) to 8000 m/s.

## Evaluation (`evaluate_seismic`, `_evaluate_seismic`)

| Component | Weight | Rule |
|---|---|---|
| Data fit | 0.45 | `chi2_fit_score` (target 1, 0.8-1.5 accepted) |
| Ray coverage | 0.2 | share of cells with coverage, scored against 0.5 |
| Physical plausibility | 0.2 | 100 - 300 x share at a bound - 300 x share outside 100-7000 m/s, over covered cells |
| Convergence | 0.15 | `convergence_score`; a history ending at chi2 1 counts as converged |

- Threshold 70 (`QUALITY_THRESHOLD`), and data fit at least 60.
- Advice: more than 5 % of picks beyond three errors; coverage under 40 %;
  more than 2 % of covered cells within 2 % of a bound; chi2 below 1 (pyGIMLi
  assumes 3 % error when the file gives none).
- Retry while `auto_adjust` and below threshold: underfit halves `lam`,
  overfit doubles it, at most `max_attempts` 3 inversions; stops when the fit
  crosses the target; the best-scoring model is kept.
- Interfaces: `velocity_threshold` 1200 m/s by default (`velocity_thresholds`
  for several). A threshold the model never reaches is noted as a finding.

## Sources

- `PyHydroGeophysX/agents/assistants/aquah/__init__.py` (`PERSONA`)
- `PyHydroGeophysX/qt_apps/modules/seismic_processing.py`
- `PyHydroGeophysX/agents/seismic_agent.py`
- `PyHydroGeophysX/data_processing/seismic.py`
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_pick_first_breaks`, `_evaluate_seismic`, `_extract_seismic_interfaces`)
- `PyHydroGeophysX/agents/_method_evaluation.py`
- `docs/source/methods/seismic.rst`
