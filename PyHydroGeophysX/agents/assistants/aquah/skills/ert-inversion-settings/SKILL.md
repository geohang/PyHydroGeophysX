---
name: ert-inversion-settings
description: >
  Use when setting up or reviewing a single ERT inversion, choosing lambda,
  reading its chi2 or quality score, or deciding whether to invert again.
  Covers lambda and its search, the chi2 target, stopping, engines, the
  evaluation's score and when to re-run.
methods: [ert]
---

# ERT inversion settings and judging the fit

Check the data first (skill `ert-data-quality`). Say which path you use:
automatic run (`invert_ert`: lambda 20, 10 iterations, relativeError 0.05) or
studio ERT page (`set_params`: lambda 50, max_iterations 15, relative_error
0.05, engine pyhydro, inversion_mode quick). Lambda stays within 1e-3 to 1e5.

1. **Mode.** Studio `quick` skips the geometric-factor check and the lambda
   search; `full` runs both. Send `inversion_mode` first, exceptions after it.
2. **Engine** is the user's choice. `adtlert` falls back to pyhydro without
   CUDA 12 or cuDSS; `e4d` never falls back; `r2` and `r3t` choose their own
   smoothing, so lambda and `auto_lambda` do not apply. Report the engine that
   ran (`engine` against `engine_requested`).
3. **Fixed lambda first.** The requested lambda is iterated until chi2
   flattens (`plateau_tolerance` 0.005, up to `max_total_iterations` 60) and is
   always kept as `fixed_lambda`. Start lambda on the smooth side.
4. **Outliers** (`reject_outliers`, off by default): beyond 3.0 sigma, 2
   passes, never below half the data.
5. **Lambda search** (`auto_lambda`): chi2 target 1.0 +/- 0.2, at most 6
   trials, bracketing by a factor of 4, then bisection in log lambda.
6. **Judge** (`evaluate_inversion`). chi2 accepted 0.8-1.5, target 1. Score =
   0.40 data fit + 0.25 smoothness + 0.25 plausibility (cells outside 1-10000
   ohm-m) + 0.10 convergence. Accepted only if score >= 70, data fit >= 60 and
   plausibility >= 70; otherwise `needs_review`, and the result is
   preliminary. Call the scores heuristics.
7. **What chi2 hides.** A wrong geometric factor (only the k check finds it).
   `implied_from_residuals` above 1.5 x the assumed error: the data are noisier
   than the error model says. Misfit flat at every lambda: the limit is the
   data or the error model, not lambda. `diverged`: not a result; raise lambda,
   check electrode positions, inspect noisy readings. chi2 below 0.8: errors
   overstated or noise fitted.
8. **Automatic retry** (`max_attempts` 3, the first included): underfit halves
   lambda, overfit doubles it, a still-improving run gets 5 more iterations (at
   most 30). The best-scoring attempt is kept even below the threshold; name
   its lambda. No retry for a structure-constrained or studio-made model.
9. **Re-run decisions.** You may change lambda and iterations. Error model,
   removing readings or electrodes, engine, mesh or zones: propose with the
   numbers behind it; the user decides. After `needs_review` from the loop,
   report the attempt log and stop rather than guess again.

Report: lambda requested and used, engine, error model and mean error, chi2
history and stop reason, readings inverted, score and its components.

Details (defaults, pipeline stages, engines, scoring formulas, retry rules):
references/settings-and-scoring.md

## Sources

Under `PyHydroGeophysX/`: `agents/ert_inversion_agent.py`,
`agents/inversion_evaluation_agent.py`, `agents/_method_evaluation.py`,
`agents/runtime/catalog.py`, `inversion/ert_inversion.py`,
`inversion/lambda_search.py`, `qt_apps/modules/ert_processing.py`. Docs:
`docs/source/methods/ert.rst`, `docs/source/agents/troubleshooting.rst`.
