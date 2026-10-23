# Time-lapse ERT: engines, defaults and alignment

## What each path runs

- **Automatic run** (`invert_time_lapse`, two or more surveys):
  ERTInversionAgent `_execute_time_lapse` -> `TimeLapseERTInversion`. One 4D
  problem: a data-misfit term per survey, a spatial smoothness term (lambda)
  within each time step, and a temporal term (alpha) on the first differences
  between the model blocks of adjacent time steps (`W_t`), optionally weighted
  by the interval. Label: `SCHEME_LABEL` "Simultaneous inversion with temporal
  constraints (4D)"; id `IMPLEMENTED_SCHEME` `temporal_constraint`.
- `time_lapse_method` is written to the log only. `resolve_scheme` turns a
  request for `difference`, `ratio` or `joint` into a note saying it was not
  implemented and what ran instead (`_method.ALTERNATIVE_SCHEMES`).
- **Studio ERT page** (`add_timelapse_files`, `run_timelapse`; settings via
  `set_params`: `tl_alpha`, `tl_norm`, `tl_windowed`, `tl_window_size`,
  `tl_low_memory`, `engine`):
  - `pyhydro`: `TimeLapseERTInversion` through `run_timelapse_ert`; windowed
    mode (`WindowedTimeLapseERTInversion`) for series too long to invert at
    once.
  - `e4d`: E4D's own time-lapse inversion. The first survey is the baseline,
    each later one starts from the solution before it and the smoothness acts
    on the change. Only measurements common to every survey are inverted.
    Alpha, the interval weighting and auto-lambda are not used; the log says so.
  - `r2`, `r3t`: their difference inversion (LaBrecque and Yang, 2001): the
    first survey is inverted, every later one from that model with smoothness
    on the change, using the readings it shares with the first.
  - `adtlert`: requires windowed mode and a common survey geometry; aligns the
    union of readings and fills those missing from a survey at 100 % relative
    error; builds its own temporal operator and constrains every pair equally.

## Defaults

| Setting | Automatic run | Studio (`DEFAULT_TL`) | `TimeLapseERTInversion` class |
|---|---|---|---|
| lambda | 15 (`inversion_params`) | 50 | 100 |
| alpha | 10 (`temporal_regularization`) | 10 | 10 |
| max iterations | 10 | 15 | - |
| relativeError | 0.05 | 0.05 | 0.05 |
| model bounds | (0.001, 1e4) | rho 1 to 1e4 | (0.0001, 10000) |
| norm | L2 | L2 | L2 |
| solver | library default | `spd_cholesky` | `spd_cholesky` |

- The automatic run builds its mesh from the first survey with
  `quality=31, paraMaxCellSize=25, paraDepth=28`.
- Class stopping: chi2 below `target_chi_squared` 1.0, or a relative chi2
  change below `convergence_tolerance` 0.01 after `min_iterations` 5.
- Errors from the file are clipped to 0.01-0.50; estimated ones are
  `relativeError + absoluteError/abs(r)` (plus `absoluteUError`/abs(U) when
  set), clipped the same way.
- Studio auto-lambda (`auto_lambda`, off): target 1.0 +/- 0.2, at most 4 trials,
  lambda only divided by 4 each step (never raised), warm-started, never
  returns a worse fit than the first run.
- `tl_norm`: L2 smooth, L1 blocky, L1L2 hybrid. `tl_window_size` 3.
- `temporal_weighting` `interval` (default): each adjacent pair weighted by
  1/dt, normalised by the median interval, capped at
  `temporal_weight_limit` 10 either way; `uniform` reproduces older runs. What
  was applied is recorded under `temporal_weighting` in the result.

## Survey timing (`survey_timing`)

File names first (a pattern is accepted only if it parses every file and gives
distinct times; ambiguous layouts are resolved by the sequence), then the file
header, then the file modification time only on opt-in (studio: "Date undated
files by when they were last saved", off by default). Otherwise a 1..n index,
stated in the log and the report. A BERT/pyGIMLi file can carry
`# date: 2026-01-12 05:50:38` as its first line. The run reads times from the
files that loaded, so a failed file does not shift the dates.

## Keeping surveys consistent

- One mesh for all surveys, from the first.
- `normalize_for_timelapse` (studio): each survey filtered independently,
  at least 4 valid readings per step; a reciprocal `error_model` replaces every
  survey's `err` column with one fit over all surveys.
- `pyhydro` keeps each survey's own readings and ordering; E4D the common
  ones; R2/R3t those shared with the first; ADTLERT refuses differing
  electrode positions.

## Evaluation

`evaluate_inversion` scores chi2 from the history (column 0 of the last row)
and plausibility and smoothness from the first survey's model
(`baseline_model`). A retry keeps the run's alpha, survey files and dates.

## Sources

- `PyHydroGeophysX/agents/_method.py`
- `PyHydroGeophysX/agents/ert_inversion_agent.py`
- `PyHydroGeophysX/agents/inversion_evaluation_agent.py`
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_invert_time_lapse`, `_evaluate`)
- `PyHydroGeophysX/inversion/time_lapse.py`, `PyHydroGeophysX/inversion/_time_lapse_workflow.py`, `PyHydroGeophysX/inversion/temporal_weights.py`
- `PyHydroGeophysX/data_processing/survey_timing.py`, `PyHydroGeophysX/data_processing/ert_io.py`
- `PyHydroGeophysX/qt_apps/modules/ert_processing.py`
- `docs/source/methods/ert.rst`, `docs/source/tutorials/time_lapse_monitoring.rst`
