# ERT inversion settings and scoring: details

## Defaults

**Automatic run** (`invert_ert` -> ERTInversionAgent `_execute_standard` ->
`ERTInversion`), read from `inversion_params`: `lambda` 20, `max_iterations`
10, `relativeError` 0.05 (used only where the exported data carry no valid
`err`), `model_constraints` (1e-6, 1e7), `lambda_rate` 1.0 (no cooling),
`lambda_min` 1.0. The linear solver is the library's (`cgls` in
`ERTInversion`) unless `method` is given. The agent passes `quality`,
`paraDX`, `paraMaxCellSize` and `paraDepth` only when the request gives them.
Without a supplied mesh, `ERTInversion.setup()` applies positive values to
mesh creation; 0 leaves the default (`quality=34`, otherwise pyGIMLi's choice).
Default runs still use `quality=34`. The agent also passes `absoluteUError`,
which `ERTInversion` does not read; it reads `absoluteError` (0.0001 ohm).

`ERTInversion` stopping: chi2 below `target_chi_squared` 1.0 (`target`); chi2
changing by less than `convergence_tolerance` 0.005 after `min_iterations` 5
(`plateau`); otherwise the iteration cap (`iteration_cap`). The stop reason is
in `meta['stop_reason']`.

**Studio ERT page** (`run_inversion` -> `run_ert_manager_inversion`), keys of
`set_params`: `lambda` 50, `max_iterations` 15, `plateau_tolerance` 0.005,
`max_total_iterations` 60, `relative_error` 0.05, `absolute_error` 0,
`engine` pyhydro, `inversion_mode` quick, `target_chi2` 1.0, `chi2_tolerance`
0.2, `max_lambda_trials` 6, `reject_outliers` false, `outlier_threshold` 3.0,
`outlier_passes` 2, `min_data_fraction` 0.5. `LAMBDA_BOUNDS` = (1e-3, 1e5).

`quick` sets `geometric_factor_policy` off and `auto_lambda` false; `full`
sets `fix` and true. It is a preset: keys sent after `inversion_mode` in the
same object override it.

## The studio pipeline, in order (`run_ert_manager_inversion`)

0. Geometric factors (`ensure_geometric_factors`, tolerance 0.05).
1. Error model (`_prepare_ert_data`).
2. The requested lambda iterated to a plateau (`_fit_to_plateau`); a run that
   uses up `max_iterations` is continued from its own model up to
   `max_total_iterations`. Always kept as `fixed_lambda`. A run whose chi2 ends
   above where it began is marked `diverged`.
3. Outlier rejection (`_reject_outliers`): readings whose weighted residual
   exceeds the threshold are dropped and the inversion repeated at the same
   lambda; never below `min_data_fraction` of the data.
4. Lambda search (`search_lambda_for_chi2`): multiply or divide lambda by 4
   until the target is bracketed, then bisect in log lambda; the best trial by
   closeness to the target wins. Warm-started from the nearest solved lambda;
   if the warm sweep ends above chi2 15 and did not converge it is repeated cold
   and the closer one kept. Start lambda on the smooth side, so the sweep
   relaxes toward the data.

Result keys to read: `engine`, `engine_requested`, `lambda_requested`,
`lambda_used`, `auto_lambda_status`, `auto_lambda_note`, `lambda_trials`,
`outliers`, `convergence_stop`, `geometric_factors`, `data_error`
(with `implied_from_residuals`), `convergence_track`, `fixed_lambda`.

## Engines

- `pyhydro`: in-house Gauss-Newton (`ERTInversion`). `pygimli`: ERTManager.
- `adtlert`: differentiable 2.5D GPU backend; falls back to `pyhydro` without
  CUDA 12 or cuDSS, and for remote electrodes encoded with negative ABMN.
- `e4d`: PNNL's E4D, external; at least two MPI processes; never falls back;
  when it cannot run, its run folder is written and the run stops.
- `r2`, `r3t`: Binley's programs; Occam inversions choosing their own alpha
  every iteration, so `lambda` and `auto_lambda` do not apply; they stop at an
  RMS of sqrt(`target_chi2`), when the misfit stops improving, or at the
  iteration limit. R2 is the engine for profiles.
- Fixed a-priori zones are held fixed only by `pyhydro`, `r2` and `r3t`.

## Quality score (InversionEvaluationAgent)

| Component | Weight | Rule |
|---|---|---|
| Data fit | 0.40 | `chi2_fit_score`: inside 0.8-1.5, 100 - 20 x abs(chi2 - 1); below 0.8 (`overfit`) 40 + 20 x chi2/0.8; above 1.5 (`underfit`) 60 - 10 x (chi2 - 1.5), not below 0 |
| Smoothness | 0.25 | gradients of consecutive model values against `max_gradient` 100 |
| Physical plausibility | 0.25 | share of cells outside 1-10000 ohm-m: 100 - 200 x share; x 0.8 when max/min > 1000 |
| Convergence | 0.10 | mean relative chi2 drop over the last two steps: < 0.001 -> 100, < 0.01 -> 90, < 0.05 -> 70, else 50; fewer than 3 values -> 60 |

Accepted when score >= `quality_threshold` (70), data fit >= 60 and
plausibility >= 70; otherwise `needs_review`. A time-lapse history row is
`[chi2, phi_m, phi_t]`; the misfit is column 0.

Recommendations the agent gives: chi2 above target -> check errors, geometry
and convergence, a smaller lambda may help; below -> check for overestimated
errors or overfitting, a larger lambda may be appropriate.

## Automatic retry (`auto_adjust`, `max_attempts` 3 including the first)

- underfit: lambda x 0.5; overfit: lambda x 2.0; otherwise x 1.2 when chi2 < 1
  and x 0.8 when above; clipped to `LAMBDA_BOUNDS`.
- `still_improving`: `max_iterations` + 5, at most 30.
- Stops when accepted, or when the last three scores differ by less than 2.0.
- The best-scoring attempt replaces the model even if below the threshold.
- No retry (`auto_adjust` False) for a structure-constrained model or one taken
  over from the studio.
- A failed retry keeps the best model before it (`retry_errors`).

## Troubleshooting (docs)

- Quality loop never reaches threshold: review the attempt log, adjust damping
  or data errors manually, treat the result as preliminary.
- Inversion diverges: increase lambda, check electrode positions, inspect noisy
  data points before another full inversion.

## Sources

- `PyHydroGeophysX/agents/ert_inversion_agent.py`
- `PyHydroGeophysX/agents/inversion_evaluation_agent.py`
- `PyHydroGeophysX/agents/_method_evaluation.py` (`chi2_fit_score`)
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_invert_single`, `_evaluate`)
- `PyHydroGeophysX/inversion/ert_inversion.py`, `PyHydroGeophysX/inversion/lambda_search.py`
- `PyHydroGeophysX/qt_apps/modules/ert_processing.py`
- `docs/source/methods/ert.rst`, `docs/source/agents/troubleshooting.rst`
