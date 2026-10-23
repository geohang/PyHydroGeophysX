# ERT data quality: details

The detail behind the steps of `SKILL.md`, with the code each rests on.

## Readers and formats

- Formats read: DAS-1 `.Data`, `.tx0` (Lippmann, GeoTest), Res2DInv `.dat`,
  AGI SuperSting `.stg`, Subsurface Insights `results_processed_*.csv`, BGS
  PRIME `.tab`, E4D and pyGIMLi/BERT `.ohm`/`.dat`; ResIPy adds formats when
  installed (`docs/source/methods/ert.rst`).
- The automatic run's loader (`ERTLoaderAgent.execute`) accepts the extensions
  `.ohm .bin .dat .stg .txt .data .tab`, and electrode files `.dat .txt .csv`.
- With no instrument named, `_load_ert` takes it from the header
  (`_detect_instrument_from_header` -> `guess_ert_format`); `DEFAULT_INSTRUMENT`
  `E4D` is used only for a file that names nothing.
- Declared vs detected mismatch: `status="needs_review"` with
  `declared_instrument` and `detected_instrument`, unless the declared reader
  also reads that layout (`_ALSO_READ_BY`). Fix: troubleshooting
  `#wrong-ert-instrument`.
- `implausible_survey`: fewer than 4 electrodes, or no measurements, means the
  file was read with the wrong format.
- The run draws a pseudosection per survey: each reading at the midpoint of its
  four electrodes, pseudo-depth 0.19 of their spread; the caption warns when
  the values are "the file's own, which may be resistances".

## Electrode positions

- BGS PRIME `.tab`: electrodes numbered by channel, no positions: a 1 m
  placeholder until the surveyed electrode file is loaded.
- Subsurface Insights: positions rebuilt from the instrument's geometric
  factors only for a straight, evenly spaced line; otherwise the surveyed file.
- AGI `.stg`: the electrode table is rebuilt from the records; the file's own
  geometric factor is kept, which makes a nominal-versus-surveyed mismatch
  visible.
- Mesh generation fails on missing, duplicated or non-monotonic positions
  (troubleshooting `#mesh-generation-fails`).

## Reciprocal errors

- `ert_formats.reciprocal_errors`: `e = |R_reciprocal - R_normal| / |mean|`, on
  transfer resistance (the geometric factor cancels). Repeats in one direction
  are stacked before the two directions are compared; a reversed pair order is
  sign-corrected. Unpaired readings get NaN; pairs straddling zero are left
  unscored.
- `ert_data_agent.calculate_reciprocal_errors` reports the same idea as a
  percentage, `200 * |R_n - R_r| / (|R_n| + |R_r|)`.
- Two files (forward and reciprocal, `recip` in the name):
  `ert_io.merge_reciprocal` appends the reciprocal readings on the forward
  file's electrodes, matched within `snap` 1e-3 m; electrodes elsewhere are
  refused as another layout.
- Check 0, `average_reciprocal_pairs` (off by default): each pair becomes one
  reading with the mean resistance and the pair's difference as its error,
  never below `reciprocal_error_floor`.

## Error sources (`ERROR_SOURCES` in `inversion/ert_inversion.py`)

| `error_source` | Error per reading |
|---|---|
| `file` (default) | the file's `err`; estimated where missing |
| `estimate` | `relative_error + absolute_error / abs(R)` |
| `max` | the larger of file and estimate |
| `stack` | stacking spread and estimate in quadrature; only for data that record a spread; usually overstates, so chi2 comes out low |
| `reciprocal` | `dR = 10^b * abs(R)^m` fitted to the pairs, never below `reciprocal_error_floor` (default 0.01); `error_model_fit` `all` (default) or `kept` |

- Studio panel: `relative_error` 0.05, `absolute_error` 0.
- `run_ert_manager_inversion` clips errors to `error_floor` 0.005 .. 1.0 and
  logs "error model: <source>, mean x %".
- `ERTInversion` without a usable `err` column: `relativeError` 0.05 plus
  `absoluteError` 0.0001 ohm / abs(r), clipped to `min_relative_error` 0.01 ..
  `max_relative_error` 2.00.

## Automatic run's export (`export_for_inversion`)

- Default strategy, when ResIPy is installed and the raw file is found: ResIPy
  rebuilds the survey; with complete reciprocal pairing it exports the pair
  mean `recipMean` with `abs((relative * resist + absolute) / recipMean)`
  errors, relative 0.01 and absolute 0.001.
- Otherwise the legacy path. ERTInversionAgent passes `use_source_error=True`
  by default: file errors normalised (percent converted), missing ones 0.08,
  clipped to 0.01-0.50. With `use_source_error=False`: reciprocal estimate,
  default 0.05, clipped to 0.01-0.10.
- The log line "Export strategy:" says which path ran.

## Data QC filter (studio `apply_filter`, `_QC_DEFAULTS`)

| Check | Key | Starting value |
|---|---|---|
| Apparent resistivity range | `min_rhoa`, `max_rhoa` | 0 to 100000 ohm-m |
| Relative error | `max_error` | 20 (percent; 0 = off) |
| rhoa above 0 | `drop_nonpositive_rhoa` | true |
| Potential | `min_voltage` | 1e-5 V |
| Current | `min_current` | 1e-4 A |
| Geometric factor | `max_geometric_factor` | 20 x the file's median abs(k) |
| Contact resistance | `max_contact_resistance` | 30000 ohm |
| Stacking spread | `max_stacking_spread` | 0.5 (fraction) |
| Reciprocal error | `max_reciprocal_error` | 0.05 (fraction; unpaired kept) |

Naming any of the last six turns "More checks" on. A check whose field the
file lacks is skipped and logged "not applied". `_qc_keep` records each check
with its threshold and the counts before and after it.

## The run loader's statistics (`_analyze_data_quality`)

Mean, std, min and max of each reading's apparent resistivity (`app_res`, or
`dV / I * K` when absent and all three values are present with nonzero I),
stored under `mean_rhoa`, `std_rhoa`, ... . Only finite values are counted in
`num_readings`; no usable values returns an error. Also `num_negative` and
`num_outliers` (more than 3 std from the mean). `qc_and_visualize` writes `electrodes.png`,
`rhoa_hist.png`, the observations table (A, B, M, N, app_res, dV, I, rel_err,
fid; parquet, or CSV without a parquet engine), `electrodes.csv` and
`ert_standard.json`.

## Geometric factors (`ensure_geometric_factors`)

A homogeneous `rho0` = 100 ohm-m forward run must return 100 ohm-m for every
configuration; the ratio is `k / k_true`. Off by more than the tolerance
(0.05) in its median, spread, or against the file's own `k_file`: `fix`
recomputes k numerically on the inversion mesh and rebuilds rhoa from the
measured resistance, kept only if the check then passes. Ratios near 2 or 0.5:
half-space and full-space conventions mixed; near 2*pi: a missing factor.

## Sources

- `PyHydroGeophysX/agents/ert_loader_agent.py`
- `PyHydroGeophysX/agents/runtime/catalog.py` (`_load_ert`, `_draw_ert_survey`)
- `PyHydroGeophysX/data_processing/ert_data_agent.py`
- `PyHydroGeophysX/data_processing/ert_formats.py`
- `PyHydroGeophysX/data_processing/ert_io.py`
- `PyHydroGeophysX/inversion/ert_inversion.py`
- `PyHydroGeophysX/qt_apps/modules/ert_processing.py`
- `docs/source/methods/ert.rst`, `docs/source/data_and_processing.rst`, `docs/source/agents/troubleshooting.rst`
