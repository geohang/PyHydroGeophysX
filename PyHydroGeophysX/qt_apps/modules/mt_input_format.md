# Magnetotelluric input formats

## Time series (step 1)

Open a recording **folder** or a single file. The format is recognised from
the files themselves:

| Instrument | What to open | Calibration |
|---|---|---|
| Phoenix MTU-5C / MTU-5P / MTU-8A | the recording folder (with `recmeta.json`), or one `.bin` / `.td_*` file | `rxcal.json`, `scal.json` and binary `.scal` coil files are found in the folder, the two above it, or a folder there named for calibrations (`Calibration_files`...) |
| Phoenix MTU-5A / V8 (legacy) | the site folder with `.TBL` and `.TS2`–`.TS5` files, or one `.TBL` | the TBL's flat gains only: Phoenix's CLB/CLC files are not decoded, so the long periods are wrong without a coil table given through the Python API |
| Metronix ADU | the folder of `.ats` files (searched recursively) | the measurement `.xml`, or coil calibration `.txt` files in a `cal` folder |
| Zonge ZEN | the folder of `.Z3D` files | the coil table stored in each file (`CAL.ANT`) |
| LEMI-424 | the folder of daily `.TXT` files | the fluxgate is flat; set the two dipole lengths (m) in step 1 |

The **Calibration** field (Phoenix and Metronix) points at a calibration file
or a folder of them (searched through), overriding what the reader would find
on its own; leave it empty otherwise. A channel shown as *uncalibrated* stays
in its recorded units, and processing refuses it unless "Allow uncalibrated
channels" is ticked.

A Phoenix receiver records which coil is on each connector only when it
detects the coil; otherwise its serial is 0 and the coil's calibration cannot
be matched. Choose the coil on H1 (Hx), H2 (Hy) and H3 (Hz) from the coils
listed below **Calibration** - the calibrations found, by serial - and the
recording is read again with them. Changing the calibration does the same.

**Directions.** Which way each channel points, in degrees clockwise from
north: 0 north, 90 east, 180 south, 270 west. Leave them *as recorded* unless
the layout differs from what the receiver was told - at Manchester, Iowa, Ex
was laid out to the south (180) and Ey and Hy to the west (270), and read as
recorded the Zyx phase came out of its quadrant. The recording is read again
with them, and the run records them. A remote reference keeps its own
directions; they do not change the impedance.

**Periodic noise.** Interference that repeats exactly on the receiver's GPS
clock - a pulse every 4 s, or a disturbance with each 30 s burst, as Phoenix
MTU-5Cs carried at Manchester, Iowa - is looked for when a recording is read.
When found, its periods fill in **Periodic noise (s)** under Processing and
its waveform is subtracted before processing; that removes only the narrow
lines at its harmonics. It matters most at 1-10 s, where the natural field is
weakest: there the interference can pass for a tight, coherent impedance, and
it is coherent between two receivers, so a remote reference does not remove it.
Clear the field to process the data as recorded. **Repeats averaged** (31 by
default) is how many repeats make up the waveform taken off each one; when the
interference changes shape within minutes, as at Manchester, 5 removes far more
of it.

**Power line.** Hum from the grid (50 or 60 Hz) is looked for when a recording
is read and, when found, set here; its frequency and harmonics are then notched
out of every run before processing. It matters where the hum lies inside the
record, such as 60 Hz in 150 Hz data: there it leaks into bands far from it and
looks like a phase near 0° with resistivity rising as the period, and a remote
site on the same grid does not remove it.

**Frequencies / decade** (10 by default) sets overlapping bands. Each band
spans about a factor of 2 and shares data with its neighbours, so the
apparent resistivity and phase are continuous, as in commercial processing.
**EMTF bands** do not overlap; their points are independent, which is what a
1D or 2D inversion assumes.

**Estimator.** The other Processing options are for noisy sites:
- **Tapers** of 3 or 5 (multitaper) give smaller errors where windows are
  few.
- **Limit the pull of noisy Hx, Hy** (bounded influence) resists bursts of
  noise in the magnetic channels.
- **Min coherence** leaves out incoherent stretches. It is meant for a
  single site.
- **Stack windows of several lengths** (multi-window) adds the
  neighbouring levels' windows to each band.
- **Noise classes** weight noise that grows with the signal.
- **Error bars from the data's spread** (jackknife) replaces the
  parametric errors.

Each band's residual stable index (2 is Gaussian) is in the result's
metadata.

A smooth curve is not proof of a good estimate. Coherent noise in E and H
gives a phase near 0° or 180° and a resistivity rising in step with period;
treat such stretches as noise, whatever their error bars.

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

### Sites recorded at different frequencies

The 2D profile is inverted at one set of frequencies, taken from the first
ticked site. A site whose own frequencies differ (another instrument, another
processing run) has its impedance interpolated to those frequencies, between
its two nearest frequencies on either side - never beyond its own band, and
only where those two are less than a factor of 2.25 apart (three frequencies
per decade still qualifies). It is interpolated as `Z / sqrt(f)` against
`ln f`, which follows how an impedance changes with frequency: on layered test
earths it is off by 0.1-0.3 % at 7-10 frequencies per decade and by under 2 %
at three, well inside the 5 % error floor. A site frequency within 5 % of a
profile frequency is used as it is. Interpolating every site onto common
periods before a 2D inversion is the usual preparation (MTpy's ModEM data
files are written the same way). The run's warnings name every site that
gives fewer values than the profile asks for, and how many of its values were
interpolated.

## TEM sounding for the 1D joint inversion (step 4)

A central-loop TDEM sounding at the same place: a CSV or text file whose first
two numeric columns are gate time (s) and the normalised response, or a `.npz`
with `times` and `response` arrays. Header lines are skipped.
