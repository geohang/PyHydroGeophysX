# EM input format

Choose **Method** (TDEM / FDEM) and **Data format** before loading.
For generic files, set survey geometry and response units in the panel.

## Generic sounding files

Numeric text with an optional header:

| Method | One sounding | Multiple soundings, side by side |
|---|---|---|
| TDEM | `time,response` | `time,response1,response2,…` |
| FDEM | `frequency,real,imag` | `frequency,real1,imag1,real2,imag2,…` |

- Time is in **seconds**; frequency is in **Hz**.
- TDEM responses must match the selected response type and scale.
- FDEM columns are in-phase and quadrature responses. A two-column single
  sounding is accepted with the imaginary component set to zero.

```csv
time,response1,response2
1.0e-05,4.1e-07,3.8e-07
1.6e-05,2.7e-07,2.5e-07
```

## TEMcompany / TEM2Go

Select **TEMcompany / TEM2Go → Load data…** and choose the project folder.
The reader supports `.tiw` and `.db` projects; if several are available, choose
from the project list. Check the displayed filename after loading.

The project imports HM/LM gates, enabled-gate flags, errors, station coordinates,
elevations, line IDs and instrument geometry. Available saved inversion settings
are also imported. **LM+HM** fits both moments to a shared model; select LM or HM
to inspect a single moment.

`project_StationData.xyz` and `project_RawData.xyz` exports are also supported.
Prefer **StationData** for inversion; a complete project preserves more metadata,
including original UTM coordinates and enabled-gate flags. The imported unit scale
is applied automatically; **Auto-calibrate** is disabled for this format.

A folder copied straight off the instrument also loads, with no project and no
export. Point the same **Load data…** at it and keep the `Data` subfolder, the
`.sts` protocol and the `.lin` line file together. Gate times, calibration and
loop geometry are read from the raw stream itself; records outside every survey
line are skipped, and station boundaries are placed by distance travelled, so
they can differ by one record from an imported copy of the same folder. Prefer a
project where one exists.

## Raw tTEM (SKB/SPS)

Select **TEMcompany tTEM raw → Load data** and choose the survey folder containing
`tTEMLog`. Keep the `*_tTEM_Rawdata.skb` files and matching GPS/current `.sps` logs
together. Select the **System GEX** and **Import filter TFI** files, or use
automatic detection when there is one of each in the folder.

- GEX supplies system geometry, waveform, gates and calibration; TFI supplies
  the import filter. Check the calibration status before inversion.
- Without GEX, fallback geometry is **8 m²**, one turn, **9.28 m** Tx–Rx separation
  and **0.43 m** height, with SKB waveform/gate information. Without TFI, stacking
  runs without its FIR filter.
- Check editable loop area, separation and height: these affect calculations.
- Keep the original raw folder; saved runs reference it rather than copying it.
  The initial line limit is **200 soundings**; increase it for larger surveys.

## Optional line geometry

Use **Load geometry…** for one row per sounding, in data-column order.
Without geometry, the section uses **Sounding spacing**.

| Header aliases | Meaning |
|---|---|
| `dist_m`, `distance`, `position` | Along-line distance (m) |
| `sensor_alt_m`, `alt`, `height` | Optional sensor height above ground (m) |
| `easting` + `northing`, `x` + `y`, or `E_UTM13N` + `N_UTM13N` | Derive distance when no distance column exists |

```csv
dist_m,sensor_alt_m
0,90
150,87
286.5,80
```

Without headers, use `position` or `position,height`. The minimum position is
shifted to zero. A neighbouring `<data>_geometry.csv`, or the only `*geom*.csv`,
can be loaded automatically.

## Before inversion and export

- Check geometry, units and **Relative error** before adjusting the fit.
  **Initial model ρ** is the starting resistivity, not a fixed recovered value.
- **Robust errors** downweights large residuals; **Hard rejection** removes gates.
  Neither restores gates excluded during import. Review tail cuts, sign reversals
  and minimum HM gates before running.
- Use the line selector, **Depth / Elevation**, and DOI controls to inspect the
  section. DOI display changes do not require another inversion.
- Choose **Add to Map…** to locate and save the result in **Project → Map**.
  There, pick a depth under **Slice** and an interpolator under **Surface** to
  read that layer as a plan-view resistivity image (ordinary kriging with a
  fitted variogram, inverse distance, triangulation or a thin-plate spline)
  rather than as coloured soundings. Soundings on a single line have no plan
  view; the interpolation needs lines that spread in two dimensions.
- **Export recovered model (csv)…** exports layer values and station summaries.
  Below-DOI values are retained and flagged.
