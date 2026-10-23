---
name: ert-data-quality
description: >
  Use when loading ERT surveys, when the loader returns needs_review, or before
  any ERT inversion. Checks the reader, electrode positions, reciprocal errors,
  the data-error model, the Data QC filter, suspect electrodes and geometric
  factors, in order, and what each must report.
methods: [ert]
---

# ERT data quality before inversion

A wrong reader, wrong electrode positions or wrong geometric factors give a
section that fits its data and is still wrong. Check in this order and report
what each check found. Automatic run: `load_ert_surveys`. Studio ERT page:
`load_data`, `load_electrodes`, `apply_filter` (no QC acts until it is called).

1. **Reader.** With no instrument named, the run reads the file's layout. A
   declared instrument that contradicts the header returns `needs_review`: tell
   the user and offer the detected one; do not override it. Fewer than 4
   electrodes or no readings means the wrong reader. The studio's `load_data`
   has no auto-detect: pick the matching format.
2. **Electrodes.** BGS PRIME `.tab` positions are a 1 m placeholder, and
   Subsurface Insights positions are rebuilt only for a straight, evenly spaced
   line: ask for the surveyed electrode file (`.dat`, `.txt`, `.csv`). Check x
   range, spacing and elevation (`electrodes.png`, `list_electrodes`);
   duplicated or non-monotonic positions break the mesh.
3. **Reciprocals.** Error = |R_reciprocal - R_normal| / |mean|, on transfer
   resistance; unpaired readings are kept. A survey in two files: keep
   `pair_reciprocal_files` true; the electrodes must coincide within 1e-3 m.
4. **Error model** (`error_source`): `file` (default), `estimate`, `max`,
   `stack` (only with a recorded stacking spread; chi2 tends low), `reciprocal`
   (power law fitted to the pairs, floor `reciprocal_error_floor` 0.01). The
   automatic run logs its "Export strategy:"; report it. If the file's errors
   and the reciprocal estimate disagree, show both; the user chooses.
5. **QC filter** (`apply_filter`): rhoa 0-100000 ohm-m and `max_error` 20
   (percent). "More checks": rhoa > 0, |V| >= 1e-5 V, |I| >= 1e-4 A, |k| <= 20 x
   the file's median, contact resistance <= 30000 ohm, stacking spread <= 0.5
   and reciprocal error <= 0.05 (fractions; unpaired kept). A check without its
   field is "not applied". Report how many readings each check removed. Good:
   few removed, mostly failing several checks. Bad: one check removes a large
   share; report it, do not loosen it. No acceptable share is set: give
   the counts and ask.
6. **Suspect electrodes.** There is no per-electrode screen. Look for failed
   readings (contact resistance, stacking spread, reciprocal error) that share
   one electrode in A, B, M or N. Removing it (`delete_electrode` drops every
   reading that uses it) is the user's decision: propose it with the counts.
7. **Values.** rhoa <= 0 stops `ERTInversion`, which inverts log rhoa: filter
   those readings, or supply `r`, or `u` and `i`. The run's loader also reports
   `num_negative` and `num_outliers` (more than 3 std from the mean).
8. **Geometric factors.** `geometric_factor_policy` `fix`, `check` or `off`,
   tolerance 0.05. The studio's `quick` mode (default) skips this check; use
   `full` (or `check`) for a new instrument, a new format or topography. A
   uniformly wrong k scales the whole section and leaves chi2 unchanged.

Hand-over before inverting: reader, electrodes, readings loaded and removed per
check, reciprocal pairs, error model and its mean, k-check result.

Details (formats, error sources, export path, QC table, k ratios):
references/qc-details.md

## Sources

Under `PyHydroGeophysX/`: `agents/ert_loader_agent.py`,
`agents/runtime/catalog.py`, `data_processing/ert_data_agent.py`,
`data_processing/ert_formats.py`, `data_processing/ert_io.py`,
`inversion/ert_inversion.py`, `qt_apps/modules/ert_processing.py`. Docs:
`docs/source/methods/ert.rst`, `docs/source/data_and_processing.rst`,
`docs/source/tutorials/field_ert_data_qc.rst`,
`docs/source/agents/troubleshooting.rst`.
