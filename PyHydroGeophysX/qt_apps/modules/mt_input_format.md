# Magnetotelluric input formats

## Time series (step 1)

Open a recording **folder** or a single file. The format is recognised from
the files themselves:

| Instrument | What to open | Calibration |
|---|---|---|
| Phoenix MTU-5C / MTU-5P / MTU-8A | the recording folder (with `recmeta.json`), or one `.bin` / `.td_*` file | `rxcal.json` and `scal.json` in the folder or a parent are found automatically |
| Phoenix MTU-5A / V8 (legacy) | the site folder with `.TBL` and `.TS2`–`.TS5` files, or one `.TBL` | the TBL's flat gains only: Phoenix's CLB/CLC files are not decoded, so the long periods are wrong without a coil table given through the Python API |
| Metronix ADU | the folder of `.ats` files (searched recursively) | the measurement `.xml`, or coil calibration `.txt` files in a `cal` folder |
| Zonge ZEN | the folder of `.Z3D` files | the coil table stored in each file (`CAL.ANT`) |
| LEMI-424 | the folder of daily `.TXT` files | the fluxgate is flat; set the two dipole lengths (m) in step 1 |

The **Calibration** field (Phoenix and Metronix) points at a calibration file
or a folder of them, overriding what the reader would find on its own; leave it
empty otherwise. A channel shown as *uncalibrated* stays in its recorded units,
and processing refuses it unless "Allow uncalibrated channels" is ticked.

A **remote reference** is a second recording made at the same time at a quiet
site. Its magnetic fields are used as instruments in the regression, which
removes the bias that noise local to the site puts on the impedance.

## Transfer functions (step 3)

Add sites that were processed elsewhere:

* **EDI** (SEG standard, impedance or spectra sections)
* **EMTF XML** (IRIS / USMTArray)
* **Z-files** from EMTF (`.zss`, `.zrr`, `.zmm`)
* **J-files** (`.j`, BIRRP)

Sites are listed in the order they are added. The ticked sites form the 2D
profile, so tick them in their order along the line; their distances come
from the latitudes and longitudes in the files, or from the site spacing in
step 5 when the files have none.

## TEM sounding for the 1D joint inversion (step 4)

A central-loop TDEM sounding at the same place: a CSV or text file whose first
two numeric columns are gate time (s) and the normalised response, or a `.npz`
with `times` and `response` arrays. Header lines are skipped.
