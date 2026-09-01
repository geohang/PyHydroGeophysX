# EM input format

The EM module inverts **soundings** (response vs frequency or time). The survey
geometry (loop radius, height, waveform, etc.) is set in the panel on the right,
**not** in the file. Pick the **Method** (FDEM / TDEM) before loading so the
columns are read correctly. A header line is optional; a non-numeric first row is
ignored.

In the **Inversion** group, **Initial model ρ (Ω·m)** sets the homogeneous
starting resistivity assigned to every layer. It controls only the optimizer's
initial point; the recovered layers remain free to move to different values.

## TEMcompany / TEM2Go projects

Set **Data format** to **TEMcompany / TEM2Go**, click **Load data…**, and select
the project directory (the folder containing `project.db`). The studio reads
the station-stacked TDEM data
directly from the project database and automatically imports:

- HM and LM gate centre times and measured `dB/dt`;
- TEMcompany in-use flags and per-gate relative standard deviations;
- UTM easting/northing, elevation, line number, and station ID;
- loop area/turns, equivalent circular-loop radius, Tx–Rx separation, and
  instrument height.

Choose **LM+HM** (the default) to fit all available early- and late-time gates to
one shared layered model. **HM** and **LM** remain available for moment-specific
diagnostics. Disabled/dummy gates are omitted separately for every sounding.
For a project line, the Studio also reads the saved layer grid and L2
vertical/lateral smoothness settings. Lateral constraints connect adjacent
stations only within the same survey-line number.

The self-describing `project_StationData.xyz` and `project_RawData.xyz` exports
can also be opened with **Load sounding(s)…**. `StationData` is preferred for
inversion because it contains station stacks. A standalone XYZ file has
latitude/longitude but not the database UTM fields, so the studio derives
local metric map coordinates. A complete project folder preserves the original
UTM coordinates and enabled-gate flags.

## TEMcompany tTEM raw (SKB/SPS)

Set **Data format** to **TEMcompany tTEM raw**, click **Load data**, and select
the survey directory containing `tTEMLog`. The importer finds every
`*_tTEM_Rawdata.skb` below it and pairs each file with its GPS and transmitter
current `.sps` logs. The **System GEX** and **Import filter TFI** selectors
auto-detect a single `.gex`/`.tfi` in that directory, or let you browse to files
stored elsewhere. It then:

- decodes the LM/HM alternating-polarity records and stacks consecutive cycles
  into approximately 2-second soundings;
- applies each gate's acquisition scale factor and normalizes by its measured
  LM/HM transmitter current;
- interpolates GPS position/elevation to each sounding and keeps acquisition
  runs as separate survey-line numbers;
- sends the resulting LM+HM observations into the same joint 1D/LCI inversion
  used for TEM2Go projects.

The GEX supplies loop area/turns, Tx-Rx geometry, full moment waveforms, gate
windows/time shifts, gate factors, first/last usable gates, and the uniform data
error. The TFI FIR coefficients are convolved with each sign-corrected LM/HM
transient sequence before stacking. The calibration status is shown below the
geometry controls and its file paths are saved in the workflow recipe.
The GEX receiver-coil two-pole low-pass and the TiB low-pass cascade are applied
to the SimPEG transient on a dense early-time grid before gate-window averaging.
The identical linear response operator is applied to the analytic
Jacobian, so predicted data and inversion sensitivities use the same calibration.

The raw directory is referenced in place instead of copied into every Project
run, because a survey can be hundreds of MB. Keep the original directory when
you need to reproduce a saved run. The initial line-inversion cap is 200
soundings; raise it deliberately for a larger section.

For raw tTEM, **Tx loop area**, **Tx-Rx separation**, and **Height** in Survey
geometry are editable calculation inputs. Changing the loop area re-normalizes
the raw response by the selected area; it is not merely a plot label. The area
also defines the equivalent circular-loop radius. The separation and height are
passed to every forward operator on the line. Values loaded from this reader are
fallbacks and may be replaced by field measurements.

When no GEX is selected or uniquely auto-detected, the importer reports the
limitation and falls back to an 8 m2 one-turn loop, 9.28 m Tx-Rx separation,
0.43 m height, and waveform/gate information from the SKB header. When no TFI
is selected, it uses polarity-pair stacking without the import FIR.

## TDEM (time domain) — one sounding

Two columns:

| Column | Meaning |
|--------|---------|
| `time` | Gate time in seconds (one row per time channel, usually log-spaced). |
| `response` | Measured response (dB/dt or H) at that time. |

```
time,response
1.0e-05,4.1e-07
1.6e-05,2.7e-07
...
```

## FDEM (frequency domain) — one sounding

Three columns:

| Column | Meaning |
|--------|---------|
| `frequency` | Frequency in Hz (one row per frequency, usually log-spaced). |
| `real` | Real (in-phase) part of the secondary field. |
| `imag` | Imaginary (quadrature) part of the secondary field. |

A two-column file (`frequency, value`) is accepted; `imag` is set to zero.

## Several soundings in one file (a survey line)

Stack soundings **side by side** so one file carries a whole line. The section is
then built by inverting each sounding and placing the recovered models next to
each other.

- **TDEM**: column 0 is the gate time; each following column is one sounding's
  response. `time, snd1, snd2, snd3, …`
- **FDEM**: column 0 is the frequency; the responses come in `(real, imag)` pairs,
  one pair per sounding. `frequency, real1, imag1, real2, imag2, …`

```
time,snd01,snd02,snd03
6.944e-06,1.29e-11,1.62e-11,2.32e-11
8.102e-06,1.22e-11,1.48e-11,2.17e-11
...
```

## Geometry file (optional, for a line) — "Load geometry…"

By default the section's x-axis uses a uniform **Sounding spacing**. To use the
real along-line distance (and, optionally, the true sensor height per sounding),
load a geometry file. It has **one row per sounding, in the same order as the
data file's soundings**. Columns are matched by header name; extra columns (e.g. a
sounding index or UTM coordinates) are ignored.

| Recognized header | Meaning |
|-------------------|---------|
| `dist_m` / `distance` / `position` / `x` | Along-line distance (m). |
| `sensor_alt_m` / `alt` / `height` | Sensor height above ground (m), optional. |
| `E_UTM13N`+`N_UTM13N` / `easting`+`northing` | Used to derive distance if no distance column is present. |

```
sounding,dist_m,sensor_alt_m
1,0,90
2,150,87
3,286.5,80
...
```

A header-less file is read by column order: one column = position; two columns =
`position, height`. Positions are shifted so the first sounding is at 0 m.

## Notes

- A line inversion runs the per-sounding work on threads: reading the stations,
  building their forward operators, and every forward and Jacobian evaluation
  inside the coupled solve. The thread count comes from the machine; set
  `parallel_workers` through `set_params` to pin it. Each sounding owns its
  forward operator and the workers only read the shared model, so the models
  come back bit for bit identical to a serial run, which the test suite checks
  over repeated solves. The coupled solve itself measures 9.5 times faster on 20
  threads; end to end the gain is smaller, because building the forward
  operators is a fixed cost that threads do not remove (roughly 1.4 times on a
  short run, 3.9 times on one that iterates 25 times).
- **Auto-calibrate** (checkbox, on by default) estimates the data-scale
  calibration from the data before inverting. Leave it on for normalized airborne
  data (e.g. moment-normalized dB/dt); it returns ~1 for data already in the
  forward's units.
- **Lateral smoothness** and **LCI passes** control the line constraint. Set
  lateral smoothness to zero to recover the former independent-1D workflow.
- A line inversion opens the **Resistivity model** tab on "View: Overview
  (map + section)": the survey map with every sounding in black and the sectioned
  line picked out, above that line's distance vs depth resistivity section. The
  two panels get the whole figure; one small caption underneath carries the few
  numbers the drawing does not already hold, being what was run, how many
  soundings and layers, the median χ², and the depth of investigation. The colour
  range is on the colourbar and the depth range on the axis, so neither is
  repeated, and how the misfit varies along the survey is on the **Inversion
  quality** tab. Scroll over either panel to zoom it about the cursor, drag to
  pan, and double-click to go back to the whole survey; with a basemap on, tiles
  are fetched again for the new extent once the wheel stops, so zooming in gives
  sharper imagery rather than a magnified one. The **Map** slider sets how much
  of the height goes to the map against the section. A
  survey holding several line numbers is sectioned one line at a time (pick it
  with "Survey line"), because chaining the lines would put an artificial jump in
  the middle of the section. Columns hatched in white are soundings that
  contributed no data of their own; their model came from their neighbours
  through the lateral constraint. **Hide below DOI** blanks the cells the data do
  not constrain, at the sensitivity threshold beside it, and both act on the
  drawing rather than on the result, so the cut can be moved without inverting
  again. **Depth / Elevation** picks what the vertical axis measures: depth hangs
  every sounding from a flat top, elevation hangs each from its own recorded
  ground level so the section follows the topography and a flat-lying unit reads
  as flat. A TEM project records an elevation per station, so the section opens on
  elevation; it falls back to depth, and the control switches itself off, when
  the survey carries no elevations or is flat to within 5 cm. The inversion is
  the same either way, because each 1D model is solved under its own station. The figure is written to
  `em_results/em_line_overview.png`. How the misfit varies along the survey is on
  the **Inversion quality** tab, next to the convergence history.
- A finished line inversion writes two tables next to `resistivity_section.npz`,
  and **Export recovered model (csv)…** puts a copy wherever you want one.
  `model_cells.csv` is one row per layer per sounding: `line`, `station`, `x`,
  `y`, `longitude`, `latitude`, `surface_elevation`, `distance_m`,
  `depth_top_m`, `depth_bottom_m`, `depth_center_m`, `z` (the cell centre's
  elevation), `resistivity_ohm_m`, `sensitivity`, `below_doi` and `chi2`. Every
  row carries its own coordinate, so the section reconstructs in a GIS or a
  gridding package without knowing anything about the layer grid.
  `soundings.csv` is the per-station summary: the same location columns plus
  `chi2`, `n_data` and `doi_m`. Cells below the depth of investigation are
  written with their resistivity and flagged rather than dropped, so the reader
  decides what to do with them.
- Tick **Basemap** to draw satellite imagery, a street map, or a topographic map
  under the soundings. Tiles are fetched once and cached under
  `~/.pyhydrogeophysx/tilecache` (override with `PYHYDROGEOPHYSX_TILE_CACHE`), so
  a survey already looked at still draws its basemap offline. This needs
  longitude/latitude per sounding, which a TEMcompany project carries; the axes
  stay in projected metres, so distances on the map are unaffected. Tile-server
  terms of use apply, and the attribution shown on the map has to stay with any
  figure that is published.

## How deep the section is allowed to go

The depth of investigation follows Christiansen and Auken (2012), *A global
measure for depth of investigation*, GEOPHYSICS 77(4), WB171–WB177:

1. Take the Jacobian of the final model in logarithmic data and model space,
   `G_ij = d log(data_i) / d log(rho_j)`. Logarithms on both sides are what make
   the number comparable between data types, so one absolute threshold serves
   every system.
2. Normalize by each datum's own standard deviation and sum over all the data:
   `s_j = sum_i |G_ij| / sigma_i`.
3. Cumulate from the bottom layer upward: `S_j = sum_{k >= j} s_k`. Entry `j` is
   the total information the data carry about layer `j` and everything below it.
4. The depth of investigation is where `S` falls through the threshold.

Their published threshold is **0.8**, fine-tuned across ground conductivity
meters, DC soundings and airborne TEM, with 0.6 to 1.2 the range they considered.
It is the default here. Only the data part of the Jacobian takes part, so a depth
that clears the threshold is one the measurements reach and not one the lateral
constraint filled in. Their step 2, sub-discretizing a few-layer model before
differentiating, is skipped as the paper allows for smooth models: these are
solved on a fixed grid of a dozen layers or more.

Summing the per-layer sensitivities, rather than normalizing them by thickness
(the paper's equation 4, which it uses only for plotting), is what makes the cut
independent of the layer grid: split a layer in two and its sensitivity splits
with it, so the depth does not move when the grid is refined (measured at 0.2 to
2.5 % across a 2x refinement).

The threshold is a judgement, so it sits on the plot next to "Below DOI" rather
than inside the inversion, and it can be moved without inverting again. On sparse
ground TDEM it can saturate: with a handful of gates per station and a model of
twenty layers, the deepest layer often clears 0.8 on its own, because the measure
cumulates from the bottom up and that layer is thick. The reported depth then
collapses onto the bottom of the parameterisation for much of the survey, which
says more about a coarse deep grid than about resolution. Two symptoms identify
it: a large share of stations reporting exactly the model bottom, and stations
holding three gates reporting the same depth as stations holding ten. Values in
the 6 to 8 range keep the reported depth inside the model on such data.

Earlier releases cut instead at a diffusion depth taken from the latest gate
time, which is the first approach the paper lists and criticizes. That rule also
read the gate time once, from the first sounding on the line, and gave every
other sounding the same reach. On a ground TDEM survey where the late gates
survive at some stations and not others, that single borrowed gate time made
every station claim the reach of the best one, several times deeper than the
stations holding only early gates can support.

## Answering a high χ²

Fit assistance offers distinct choices; a large residual alone does not establish
that a measurement is bad (geometry or a 1D earth assumption can also be wrong):

- **Auto-λ** re-solves the line at other smoothness weights to reach the target
  χ². Use it when the model is too stiff for the data. When no weight reaches the
  target, the smoothest of the weights that fit equally well is kept rather than
  the roughest, so a fraction of a percent of misfit does not buy a railed model.
- **Robust errors (keep all gates)** is the Ground TEM default. It enlarges
  effective fitting errors for large residuals without removing any imported gate.
  The original recorded errors are unchanged; see the formula and audit below.
- **Hard rejection (legacy)** drops the gates the converged model cannot explain (beyond
  the σ cut, over the given passes) and solves again at the same smoothness. Use
  it when a minority of gates are simply wrong. Cutting is per gate rather than
  per sounding, because a TDEM station may carry only a handful of gates. "Keep
  at least" stops it before it would gut the survey; if it stops there, gates
  beyond the cut remain and the Inversion quality page says so.

Before these, check **Relative error**. It is the uniform part of the error
budget, the part that applies to every gate alike: system calibration and the
error in representing the ground as 1D layers. It joins each gate's recorded
stack error in quadrature, so a noisy gate stays relatively noisy. A stack error
measures repeatability only, so on ground TDEM it is usually far smaller than the
1D representation error; leaving the budget at the recorded few percent reports a
χ² in the tens or hundreds that no model can reach, and both settings above will
then work hard for nothing.

It is the size of that uniform term, not an amount added on top of whatever is
already there. TEMcompany station stacks store an error that is the stack scatter
already combined with a uniform term, recorded as `UniStd` in the acquisition
protocol and 3% on the systems seen so far. The reader reports that, and only the
shortfall is added: **Relative error** at 0.03 reproduces the stored error
exactly, and at 0.05 gives the same answer as 5% on a bare stack error. Setting
it below what is already folded in cannot shrink the stored value.

Three controls decide which gates arrive at all.

- **Tail cut (σ)** condemns a gate whose relative stack error exceeds it. The
  new panel default is 0 (off), retaining noisy project-enabled gates for weighting.
- **Cut removes** decides whether a failed stack-error test ends the sounding
  (truncate) or costs only its own gate (individual). Truncation argues that the
  decay has reached the noise floor and a later clean-looking gate is a
  fluctuation above it.
- **Keep sign reversals** decides whether a negative gate is judged on its stack
  error alone. On by default, which is what the TEMcompany inversion does:
  measured over 1,503 station-moment datasets of one project, its gate selection
  is exactly the stored in-use flags, and it keeps a non-positive gate in 87
  low-moment and 251 high-moment datasets. Either way a sign reversal costs its
  own gate and no other, because truncation is the noise-floor argument and an
  early-time reversal makes no claim about the gates after it.

An offset-loop system does genuinely reverse sign at early time, and whether the
reversal reaches the gates being inverted has a number attached: the crossing
sits near an induction number of one, so a gate at time `t` sees it only once the
ground is more conductive than `mu0*r^2/(4t)`. On a 15 m offset that is about
6 Ω·m at 12 µs and 1 Ω·m at 61 µs. Put the site's resistivity into that
expression before keeping reversals: where it says the crossing falls far earlier
than the first gate, a negative gate is something no layered earth the site could
have will produce, and keeping it lets the fit trade the rest of the sounding
against a value it can never reach.

**Min HM gates** drops the deep moment from a station that survives selection
with fewer gates than that; LM is never dropped. Off by default, because the deep
moment is the only thing that sees deep.

### Robust error settings and audit

The editable inversion parameters are `robust_errors=True`, `robust_threshold=3.0`,
`robust_passes=3`, `robust_max_error_factor=10.0`, and `reject_outliers=False` in
the Ground TEM preset. Generic settings retain the previous least-squares default.
When both switches are supplied programmatically, robust weighting takes precedence.

For each gate let `r = (prediction - observation) / original_error`. Each reweighting
pass sets `factor = min(max_error_factor, sqrt(max(1, abs(r)/threshold)))` and
`effective_error = original_error * factor`. The inverse-variance multiplier is
`1/factor²`: at the default cap it cannot fall below 0.01, and never reaches zero.
For example, at a cut of 3, a 12-sigma residual gets twice the original error and
one quarter of the original weight. This is bounded Huber-style IRLS on the data
term only, not on vertical or lateral regularisation. It is not an independent
measurement of noise or a reason to force the ground to become more resistive.

Every update uses the ORIGINAL error budget (recorded errors plus the configured
error model), never the previous inflated errors. Gates may regain weight. The
initial fit can use the configured lambda search; later passes freeze the chosen
regularisation and warm-start from the previous model. At most three reweighting
passes run by default, with early stopping when factors change by at most 1%.
This bounded iteration is not a guarantee of convergence to an exact Huber optimum.

The main χ² always uses original errors and every imported gate. `chi2_effective`
and the per-pass convergence curves use the errors actually used for fitting;
they are not directly comparable to unweighted or rejection-only χ². Sensitivity
and DOI for line inversion use the effective errors. `robust_gate_errors.csv` and
`robust_errors.json` retain per-gate original/effective errors, weights, predictions,
and residuals for auditing. The 80% retention setting is only for legacy rejection;
robust fitting retains 100% of the gates that reached the solver.

Import selection is separate: project in-use flags and nonfinite/dummy-value
checks still apply. If a caller explicitly enables tail cuts or minimum-moment
gate cuts, those gates are still removed BEFORE fitting. Robust weighting does
not restore them or turn flagged-out gates into below-detection constraints.

- A geometry file next to the data file is loaded automatically when its name is
  `<data>_geometry.csv` or it is the only `*geom*.csv` in that folder; otherwise
  use "Load geometry...". With easting/northing in it, the **Resistivity model**
  tab's plan-view map ("View: Plan slice") uses real **UTM easting/northing** axes,
  and you pick the depth with the slider. Soundings spread in 2D draw a filled map;
  a single flight line draws points coloured by resistivity. "View: Section" shows
  the along-line distance vs depth section on its own.
- See `examples/data/EM/EastRiver_VTEM/` for a paired data + geometry example.
- TEMcompany exports use moment-normalized `dB/dt` units. The imported system
  geometry and unit scale are applied automatically, so **Auto-calibrate** is
  disabled for this format. Use a different scale only when the export metadata
  or an independent reference survey supports it.
