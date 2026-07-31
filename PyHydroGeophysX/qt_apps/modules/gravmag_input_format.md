# Gravity / Magnetics station input format

The module loads **scattered station data**: a point cloud of map positions and a
measured anomaly value. A whitespace- or comma-delimited text file
(`.csv` / `.txt` / `.dat`) with (at least) three columns:

| Column | Meaning |
|--------|---------|
| `x` | Easting / map x coordinate (m). |
| `y` | Northing / map y coordinate (m). |
| `value` | The anomaly at the station. |

```
# x        y        value
-98.2    12.4     0.213
-71.5   -33.1     0.147
...
```

The meaning of `value` depends on the **Field type** you pick:

- **gravity** — gravity anomaly in **mGal** (e.g. a Bouguer or free-air anomaly).
- **magnetics** — total-field magnetic anomaly in **nT**.

## Workflow

1. Load the station file and choose the field type.
2. **Regional / residual**: fit a polynomial regional trend (degree 1–3) and
   subtract it to isolate the residual.
3. **Gridding**: interpolate the displayed field to a regular grid (export to
   `.npy` / CSV / VTK).
4. **Profile**: sample the grid along a line and export the profile.
5. **Forward body**: add a buried body (gravity: sphere or prism; magnetics:
   dipole / magnetized sphere) and compare its response with the data.

## Notes

- A header line is optional; non-numeric leading rows are ignored.
- Extra columns after the third are ignored.
- The body parameters and (for magnetics) the ambient field inclination /
  declination / strength are set in the panel, not in the file.
