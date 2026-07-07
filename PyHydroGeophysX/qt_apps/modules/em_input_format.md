# EM input format

The EM module inverts **soundings** (response vs frequency or time). The survey
geometry (loop radius, height, waveform, etc.) is set in the panel on the right,
**not** in the file. Pick the **Method** (FDEM / TDEM) before loading so the
columns are read correctly. A header line is optional; a non-numeric first row is
ignored.

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
- Sensor height mainly changes the signal amplitude (absorbed by the
  calibration), not the decay shape, so loading geometry is mostly about the
  along-line distance and the map coordinates, not the fit.
- A geometry file next to the data file is loaded automatically when its name is
  `<data>_geometry.csv` or it is the only `*geom*.csv` in that folder; otherwise
  use "Load geometry...". With easting/northing in it, the **Resistivity model**
  tab's plan-view map ("View: Plan slice") uses real **UTM easting/northing** axes,
  and you pick the depth with the slider. Soundings spread in 2D draw a filled map;
  a single flight line draws points coloured by resistivity. "View: Section" shows
  the along-line distance vs depth section.
- See `examples/data/EM/EastRiver_VTEM/` for a paired data + geometry example.
