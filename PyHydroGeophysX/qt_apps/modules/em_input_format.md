# EM input format

The EM module inverts **soundings** (response vs frequency or time). The survey
geometry (loop radius, height, waveform, etc.) is set in the panel on the right,
**not** in the file. Pick the **Method** (FDEM / TDEM) before loading so the
columns are read correctly. A header line is optional; a non-numeric first row is
ignored.

## TEMcompany / TEM2Go projects

Set **Data format** to **TEMcompany / TEM2Go**, click **Load data…**, and select
the project directory (the folder containing `project.db`). The workbench reads
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
For a project line, the Workbench also reads the saved layer grid and L2
vertical/lateral smoothness settings. Lateral constraints connect adjacent
stations only within the same survey-line number.

The self-describing `project_StationData.xyz` and `project_RawData.xyz` exports
can also be opened with **Load sounding(s)…**. `StationData` is preferred for
inversion because it contains station stacks. A standalone XYZ file has
latitude/longitude but not the database UTM fields, so the workbench derives
local metric map coordinates. A complete project folder preserves the original
UTM coordinates and enabled-gate flags.

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

- **Auto-calibrate** (checkbox, on by default) estimates the data-scale
  calibration from the data before inverting. Leave it on for normalized airborne
  data (e.g. moment-normalized dB/dt); it returns ~1 for data already in the
  forward's units.
- **Lateral smoothness** and **LCI passes** control the line constraint. Set
  lateral smoothness to zero to recover the former independent-1D workflow.
- A line inversion opens the **Resistivity model** tab on "View: Overview
  (map + section)": the survey map with every sounding in black and the sectioned
  line picked out, above that line's distance vs depth resistivity section.
  Beside the map are the few numbers the picture does not already carry: what was
  run, how many soundings and layers, the median χ², and the depth of
  investigation. The colour range is on the colourbar and the depth range on the
  axis, so neither is repeated, and how the misfit varies along the survey is on
  the **Inversion quality** tab. A
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
than inside the inversion, and it can be moved without inverting again. Vendors
do not agree on how conservative to be: on a 71-station ground TDEM survey the
TEMcompany software reported depths about 2.6 times shallower than 0.8 gives
(median 12 m against 37 m), and its numbers come back at roughly 8. Raise it to
line up with an acquisition package's own sections.

Earlier releases cut instead at a diffusion depth taken from the latest gate
time, which is the first approach the paper lists and criticizes. That rule also
read the gate time once, from the first sounding on the line, and gave every
other sounding the same reach; on a ground TDEM survey where the late gates
survive at some stations and not others it ran about five times deeper than the
acquisition software's own depths of investigation.

## Answering a high χ²

Two settings, the same pair the ERT module offers, and they address different
causes:

- **Auto-λ** re-solves the line at other smoothness weights to reach the target
  χ². Use it when the model is too stiff for the data. When no weight reaches the
  target, the smoothest of the weights that fit equally well is kept rather than
  the roughest, so a fraction of a percent of misfit does not buy a railed model.
- **Reject outliers** drops the gates the converged model cannot explain (beyond
  the σ cut, over the given passes) and solves again at the same smoothness. Use
  it when a minority of gates are simply wrong. Cutting is per gate rather than
  per sounding, because a TDEM station may carry only a handful of gates. "Keep
  at least" stops it before it would gut the survey; if it stops there, gates
  beyond the cut remain and the Inversion quality page says so.

Before either, check **Relative error**. For TEMcompany data it is a floor on the
per-gate stack error the instrument recorded. A stack error measures repeatability
only, so on ground TDEM it is usually far smaller than the error in representing
the ground as 1D layers; leaving it at the recorded few percent reports a χ² in
the tens or hundreds that no model can reach, and both settings above will then
work hard for nothing.
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
