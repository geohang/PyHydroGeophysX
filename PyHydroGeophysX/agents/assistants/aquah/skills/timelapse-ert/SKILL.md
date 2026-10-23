---
name: timelapse-ert
description: >
  Use when two or more ERT surveys of one line are compared through time
  (monitoring, a time-lapse series), or to invert them or correct them for
  temperature. Says what each engine runs (in-house: all surveys at once with a
  temporal constraint, not a difference inversion), dating, alpha.
methods: [ert]
---

# Time-lapse ERT

1. **Name the scheme correctly.** The automatic run (`invert_time_lapse`) and
   the studio's `pyhydro` engine invert all surveys at once: a misfit per
   survey, spatial smoothness (lambda) and a temporal term (alpha) on the
   differences between consecutive surveys - "Simultaneous inversion with
   temporal constraints (4D)", **not** a difference inversion.
   `time_lapse_method` is only recorded: a request for a difference, ratio or
   joint inversion is not honoured and the run says so; tell the user before
   running. Studio `e4d`: baseline first, each survey from the one before,
   common readings only. Studio `r2`/`r3t`: their difference inversion against
   the first survey. Studio `adtlert`: windowed mode, common geometry.
2. **Defaults.** Automatic run: lambda 15, 10 iterations, alpha
   (`temporal_regularization`) 10, solver `spd_cholesky`; its mesh is built from
   the first survey with `paraDepth=28`: check that depth suits the line.
   Studio: lambda 50, `tl_alpha` 10, L2, 15 iterations, auto-lambda off.
3. **Date and order the surveys.** Times come from the file names (a pattern
   must read every file), then the headers; modification time only if the user
   opts in; otherwise a 1..n index, which the log states. Read the
   `survey_timing` summary. Undated surveys: ask for times or renamed files;
   never assume an interval. A file that fails to load is left out: check the
   remaining list still matches the dates.
4. **Keep surveys comparable.** One mesh comes from the first survey, so the
   electrodes must match; only ADTLERT checks this, so compare the electrode
   tables yourself. One QC filter and one error model for all surveys (a
   reciprocal error model is one fit over all of them); at least 4 valid
   readings per survey.
5. **Alpha.** A larger alpha damps genuine abrupt change; a recovered change
   depends on alpha as well as the data, and the steps are not independent.
   No value is recommended beyond the default 10: first reproduce one
   time step, then judge whether the temporal smoothing hides real change, and
   compare runs only with the same data and error model. `temporal_weighting`
   `interval` (default) penalises the rate of change, capped at 10x either way.
6. **Fit.** chi2 is judged over all surveys together; the evaluation scores
   that chi2 and the first survey's model. A retry keeps alpha and the dates.
7. **Temperature correction** (about 2 % per degC, as large as a season's
   moisture signal). Post-processing; `hayley` (default) or `linear` (0.025 per
   degC), reference 25 degC. The temperature source (`constant`, `profile`,
   `surface`, `seasonal`) is the user's choice; never pick one silently. A
   dated record against undated surveys is refused. Raw models are kept beside
   the corrected ones, and every result must say which it shows.
8. **Downstream.** Water content: convert every step with the same parameters
   (skill `water-content-from-resistivity`).

Details: references/engines-and-defaults.md (engines, defaults, timing,
alignment) and references/temperature-correction.md.

## Sources

Under `PyHydroGeophysX/`: `agents/_method.py`, `agents/ert_inversion_agent.py`,
`agents/runtime/catalog.py`, `inversion/time_lapse.py`,
`inversion/_time_lapse_workflow.py`, `inversion/temporal_weights.py`,
`data_processing/survey_timing.py`, `data_processing/ert_io.py`,
`petrophysics/temperature.py`, `qt_apps/modules/ert_processing.py`. Docs:
`docs/source/methods/ert.rst`, `docs/source/tutorials/time_lapse_monitoring.rst`.
