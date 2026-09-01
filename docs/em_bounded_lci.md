# Bound-aware simultaneous LCI

The `invert_line` workflow now supports a sparse trust-region reflective solver
through `inv["lci_solver"] = "trf"`. TRF is the formal framework and Qt default;
the legacy `gauss_newton` path remains selectable for fast previews and exact
reproduction of older results. The balanced default (`lci_max_nfev=90`,
`lci_ftol=1e-4`) completed the full 882-sounding trailcreek robust workflow.
The stricter `1e-6` tolerance exhausted that budget before robust reweighting.

## Parameters

```python
INVERSION_OPTIONS = {
    "lci_mode": "simultaneous",   # requires positive lateral_smoothness and >=2 stations
    "lci_solver": "trf",         # or "gauss_newton" for legacy comparisons
    "lci_max_nfev": 90,           # framework and Qt default, per fixed-error stage
    "lci_ftol": 1e-4,
    "lci_xtol": 1e-6,
    "lci_gtol": 1e-5,
    "auto_lambda": False,        # fix regularization for the initial comparison
    "robust_errors": False,      # do not increase any imported gate's uncertainty
    "reject_outliers": False,    # do not remove gates by fitted residual
    "shallow_prior_enabled": False,
}
```

Keep the project's layer thicknesses, geometry, vertical and lateral weights,
and resistivity bounds unchanged for an algorithm comparison. Import flags and
reader QC remain independent of `robust_errors` and `reject_outliers`.

TRF's budget counts forward evaluations, including rejected trials; it is not
the legacy `max_iterations`. `ftol` concerns the complete objective, `xtol` the
model step, and `gtol` SciPy's bound-scaled gradient. TRF does not stop early on
data chi-squared: it minimizes the complete fixed-weight objective, while
`target_chi2` guides the optional outer regularization search. The legacy path
retains its existing target-band/plateau behavior.

## Objective and implementation

The unknown is `log10(rho)`. The residual vector stacks the data residual divided
by its original error, vertical first differences, lateral differences and any
enabled one-sided shallow-prior penalty. The regularization multipliers have
exactly the same squared-penalty meaning as before. TRF solves within the bounds;
it does not clip an unconstrained Gauss-Newton step and then assume descent.

The implementation reuses the analytic SimPEG Jacobian, sparse matrices, the
existing per-sounding thread pool and cached responses. The sparse linear solver
is LSMR. It is compatible with SciPy >=1.8 without the newer callback interface.
Progress history contains accepted Jacobian-evaluation models, not every trial.

Both solvers additionally record `chi2_median_history`: the equal-sounding
median of the per-sounding mean squared normalized gate residual at every
accepted model, excluding empty soundings. This is **not** the median over gates
and is not reconstructed from the global history. The optimizer still minimizes
the original full objective. Qt shows median histories prominently and global
histories as a dotted reference; robust runs use effective-error medians with
original-error medians only at recorded solve endpoints (dashed). Old caches
without median histories retain their explicitly global display.

## Diagnostics and limitations

`result["lci_report"]["diagnostics"]`, also exported in `lci_report.json`, includes
objective history, data/regularization contributions, evaluation counts,
termination status, bound fractions, scaled optimality and projected gradient.

- `solver_converged`: a SciPy tolerance was met; **not** a global optimum claim.
- `stationary`: the projected gradient meets the requested gradient tolerance.
- `max_nfev`: budget exhaustion, **not** convergence.
- `target_band_reached`: the actual data chi-squared lies in the requested band;
  unrelated to whether the optimizer is stationary.

For TRF, an incomplete initial inner solve prevents auto-lambda exploration or
subsequent robust error inflation. An incomplete reweighted solve stops further
inflation and prevents a target-success claim; its actually used errors remain
in the report. The legacy solver has no new convergence certificate and retains
its historical behavior. Reweighting cannot establish geological correctness.

The aapl run improves the fit without requiring resemblance to vendor models.
Deep low resistivity can remain. Examine late-gate residuals, sensitivities and
initial-model/regularization dependence before geological interpretation. DOI is
a sensitivity heuristic, not a confidence interval or a resolution proof.

## Tx/Rx height contract

Project geometry is applied per station, with separate `tx_height` and
`rx_height`. Qt no longer passes the project's Rx-only `heights` array as a
common-height override. An explicitly supplied `invert_line(..., heights=...)`
still means **set both** Tx and Rx heights, preserving the existing API contract
for imported common-height geometry. When using recorded project geometry,
omit that argument. This also keeps inversion and forward-audit geometry equal.

## Scope of validation

Unit/regression tests cover a constrained optimum that clipping cannot reach,
one-sided priors, full-objective progress with increasing data misfit, analytic
Jacobian finite differences, serial/threaded equivalence, budget guards and Qt
height propagation. Local experiments compare all 145 aapl soundings and
24-sounding prefixes of three other surveys. Prefix tests are not a claim about
entire surveys or all possible acquisition geometries. Existing project
databases and previous inversion results are preserved.
