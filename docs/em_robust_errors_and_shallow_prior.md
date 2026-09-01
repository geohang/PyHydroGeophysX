# TEM fitting weights and empirical resistive-background prior

The `ground_tem` preset and the trailcreek notebook enable the following options.
Other survey presets keep their previous defaults. Existing saved runs keep their
recorded settings; re-import a project and start a new run to use these changes.

```python
robust_errors = True
reject_outliers = False
robust_threshold = 3.0
robust_passes = 3
robust_max_error_factor = 10.0
robust_min_unchanged_fraction = 0.70
robust_target_chi2 = 1.75
robust_target_tolerance = 0.25

shallow_prior_enabled = True
shallow_prior_mode = "signal_threshold"  # or "quality_trend" / "manual"
shallow_prior_signal_threshold = 0.0     # auto instrument forward; positive = raw-unit override
shallow_prior_signal_reference_resistivity = 1000.0
shallow_prior_min_resistivity = 0.0       # 0 = automatic target
shallow_prior_resistivity_factor = 2.0    # target = effective initial model × factor
shallow_prior_weight = 1.0
shallow_prior_window = 11
shallow_prior_snr_ratio = 0.60
shallow_prior_reference_gate = 2      # zero-based raw LM gate index
shallow_prior_signal_ratio = 0.80
shallow_prior_noise_ratio_max = 2.0
shallow_prior_lines = []             # all selected lines
shallow_prior_distance_min_m = 0.0
shallow_prior_distance_max_m = 0.0    # unlimited
```

## Effective errors

Every **imported** gate stays in the objective; project flags and invalid-data
rules still apply during import. At each reweighting pass, at least
`ceil(0.70 * N)` gates retain exactly their original fitting uncertainty. The
smallest original-error normalized residuals are protected; their identities can
change between passes. The quota applies across one simultaneous LCI run, or
separately to each independent sounding. It is not a per-moment guarantee.

Only unprotected residuals exceeding the sigma threshold may receive larger
effective errors, bounded by the maximum factor. A scalar inflation strength is
searched before refitting the model; it does not multiply all data errors. The
actual final effective chi-square, not the pre-refit estimate, determines whether
the 1.50–2.00 band was reached. The band is a preference, not a guarantee: the
protected fraction and error cap always take precedence. The report records the
minimum attainable chi-square **at the current model** under these limits.

Set `robust_target_chi2 = 0.0` for Huber-style reweighting without target calibration.
Calibration is not a measured noise estimate, nor independent proof of a better
physical fit. Original-error chi-square, all observations, predictions, errors and
weights remain in `robust_errors.json` and the gate audit CSV. Reweighted LCI
refits use change in the full objective for plateau detection; chi-square alone
can plateau or increase while regularisation improves.

## Empirical prior

This is the user's survey-specific assumption, **not** a physical implication
that every low-SNR observation proves resistive ground.

The default `signal_threshold` mode compares absolute raw LM amplitude with a
homogeneous-half-space forward response at the fixed signal-reference resistivity.
Actual station geometry, waveform, filtering and gate averaging are included.
Automatic thresholds are in stored project-response units, before inversion
`data_scale`. A positive manual threshold overrides the forward calculation.
At least 80% of a full rolling window, including the current station, must fall
below their thresholds. Missing raw gates are unavailable, not zero signal. A
whole quiet line can trigger without first establishing a high-SNR baseline.

For trailcreek at 8.95 µs, three tested geometries give approximately 1.67e-8,
2.88e-9, and 5.66e-10 for homogeneous 300, 1000, and 3000 Ω·m respectively.
But 10 m of 1000 Ω·m over 100 Ω·m gives about 3.75e-8: an amplitude trigger
does not identify the depth of high resistivity. Consequently the code no longer
turns this trigger into a shallow-layer constraint. It represents only a weak
resistive-background tendency across the 1-D model.

In the alternative `quality_trend` mode, the first full window of
each selected line establishes its own baseline. A subsequent full trailing
window must show reduced SNR and reduced signal at the same raw LM gate time,
without excessive growth in the stored absolute uncertainty proxy. A single bad
station, noise growth alone, or a change of gate time does not activate the raw
diagnostic trigger. Geometry and interference can still cause ambiguity.

Raw LM diagnostics travel with new saved Qt input containers so Qt and direct
project reads use the same trigger. Old containers and non-database sources that
lack them cannot activate the absolute-signal prior. In quality-trend mode they
use the explicitly reported imported-LM-quality fallback, which cannot
separate lost signal from increased noise. Re-import TEMcompany data to avoid
that weaker fallback. No usable initial LM baseline means no automatic prior.

At each activated station the automatic target is twice the effective starting
half-space (or the geometric-mean resistivity of a supplied warm-start model),
capped by `rho_max`. A positive explicit `shallow_prior_min_resistivity` overrides
that rule. The factor of two is deliberately modest: it records a direction of
tendency without manufacturing an order-of-magnitude contrast. The historical
`shallow_prior_*` prefix remains for saved-file compatibility.

The penalty is

`sum_j [(weight * score / sqrt(n_layers)) * max(log10(rho_target) - log10(rho_j), 0)]²`.

Every layer, including the basal half-space, participates; normalisation keeps a
uniform-model penalty comparable when the layer grid changes. It is a soft,
one-sided penalty: models may stay below the target when data and other constraints
favor that, and there is no penalty above it. It does not add
observations, enter data chi-square, or provide sensitivity/DOI evidence.
Activation scores and their diagnostic inputs are saved in `shallow_prior.json`
and `shallow_prior.csv`.

For a known segment, use `mode="manual"`, specify its line and local distance
range, and retain a modest strength. This explicitly imposed prior bypasses the
automatic quality trigger. Set `shallow_prior_enabled=False` or weight zero for a
sensitivity comparison. The notebook keeps its prior layer grid, forward model,
and existing regularisation settings unchanged.

In Qt these controls are inside a collapsed **Advanced: empirical
resistive-background prior** group because this is a survey-specific assumption,
not a routine inversion setting.

## Qt quality display

For robust runs, the headline and per-sounding solid curve use effective-error
(weighted) chi-square. Original-error chi-square appears only as a dashed
reference, while exports retain both values. The raw convergence reference uses
the recorded solve endpoints; it is not an invented raw value at each iteration.
No robust weighting means the existing ordinary chi-square display is unchanged.
