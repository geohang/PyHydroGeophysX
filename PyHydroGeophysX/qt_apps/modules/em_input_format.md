# EM sounding input format

The EM module loads a **1D sounding** (one curve of response vs frequency or
time). The survey geometry (source/receiver, loop radius, height, etc.) is set in
the panel on the right, **not** in the file.

## FDEM (frequency domain)

A whitespace- or comma-delimited text file (`.csv` / `.txt` / `.dat`) with three
columns:

| Column | Meaning |
|--------|---------|
| `frequency` | Frequency in Hz (one row per frequency, usually log-spaced). |
| `real` | Real (in-phase) part of the secondary magnetic field. |
| `imag` | Imaginary (quadrature) part of the secondary magnetic field. |

```
# frequency   real        imag
100        5.57e-09    1.83e-09
215        7.42e-09    2.55e-09
...
```

A two-column file (`frequency, value`) is accepted too; the second column is
treated as the real part and the imaginary part is set to zero.

## TDEM (time domain)

Two columns:

| Column | Meaning |
|--------|---------|
| `time` | Gate time in seconds (one row per time channel, usually log-spaced). |
| `response` | Measured response (dB/dt or H) at that time. |

```
# time        response
1.0e-05     4.1e-07
1.6e-05     2.7e-07
...
```

## Notes

- A header line is optional; non-numeric leading rows are ignored.
- Pick the **Method** (FDEM / TDEM) before loading so the columns are read
  correctly.
- The same geometry you set for forward modeling is used for the inversion, so
  match it to the field system before inverting.
- The layered model in the **Layered model** table is `thickness` (one per layer)
  plus `resistivity` (one extra value for the half-space at the bottom).
