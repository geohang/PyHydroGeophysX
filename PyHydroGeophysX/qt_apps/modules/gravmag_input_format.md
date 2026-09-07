# Gravity / Magnetics input format

Load a comma- or whitespace-delimited `.csv`, `.txt` or `.dat` station file.
The first three columns are read in this order:

| Column | Meaning / unit |
|---|---|
| `x` | Easting / map X, metres |
| `y` | Northing / map Y, metres |
| `value` | Gravity anomaly in **mGal**, or total-field magnetic anomaly in **nT** |

```csv
x,y,value
100,200,0.213
125,210,0.147
```

- Choose the matching **Field type**. Use projected metre coordinates.
- A header is optional; non-numeric leading rows and columns after the third
  are ignored.
- Set body parameters and magnetic field inclination, declination and strength
  in the panel.
- After loading, inspect the field and optionally remove a regional trend
  before gridding or inversion.
