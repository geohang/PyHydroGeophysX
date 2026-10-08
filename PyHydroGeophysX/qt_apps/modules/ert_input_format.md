# ERT input formats

## Data files

Pick the instrument or file format under **Instrument / format**, then
**Add files…**. The format is not guessed: a file read with the wrong format can
still show a plausible number of electrodes and measurements.

Click a file row to preview it. For time-lapse, add one file per survey and keep
the list in acquisition order, earliest at the top; **Sort by time** orders it
using the times in the file names or headers. Hover over the format selector
to see which readers are available.

| Instrument / format | What to add |
|---|---|
| BERT / Unified (.ohm/.dat) | pyGIMLi / BERT unified data file (layout below) |
| E4D | E4D survey file (`.srv`, `.ohm`) |
| DAS-1 | DAS-1 export (`.Data`) |
| Syscal | Prosys text export |
| ABEM-Lund | Terrameter LS text export |
| Res2DInv | Res2DInv `.dat` (general array or a standard array) |
| Protocol DC / Protocol IP | R2 / cR2 `protocol.dat` |
| Sting / SuperSting | `.stg` |
| ARES, Lippmann (`.tx0`), Electra | the instrument's export |
| Subsurface Insights | `.csv` export |

BERT / unified, E4D, DAS-1, Res2DInv, Sting, ABEM-Lund, Lippmann, ARES and
Subsurface Insights files load without ResIPy; Syscal, Protocol and Electra
files need ResIPy installed. The line under the format names the reader that
loaded the file.

A BERT / unified file, with the optional date line on the first line:

```
# date: 2026-01-12 05:50:38
64
# x z
0.0  0.0
1.0  0.0
...
649
# a b m n r err
1 2 3 4  12.31  0.03
...
```

The data block can give `rhoa`, `r` (resistance) or `u` and `i`; `err` is a
fraction (0.03 = 3 %).

## Electrode file (optional)

A text table of electrode positions in metres, one row per electrode, in the
order the data number them. A header row decides the columns: `x` and `z` (or
`elevation`), or `x y z`. Without a header, `x z` or `x y z` is read, and an
electrode-number column (1, 2, 3 …) is set aside. The table must list as many
electrodes as the data use. Every file in a time-lapse list is placed on it.

## Time-lapse: one file per survey

- Each file in the list is one time step. The top file is the baseline, and
  each survey is compared with the one before it, so the order matters:
  **Sort by time** orders the list by the survey times below.
- Use one format and the same electrode layout for every file.
- **Forward and reciprocal in two files.** Some instruments write each survey
  as a forward file and, a few minutes later, a reciprocal file with the same
  name plus "recip" (`wennerv2_64_sorted_2026_01_12_05_50_38.dat` and
  `wennerv2_64_recip_sorted_2026_01_12_06_12_23.dat`). Add both. With **Pair
  reciprocal files with their forward file** ticked (it appears, ticked, when
  the list holds such names), each reciprocal file joins the forward file
  measured last before it: one row, one time step at the forward file's time,
  and one data set holding both files' readings on the forward file's
  electrodes, so every reading is compared with its reciprocal and **Max
  recip. error** under Data QC → More checks can use the pairs. A reciprocal
  that would join a forward file far more than the usual few minutes earlier
  is not joined. A forward file with no reciprocal is inverted on its own
  (marked "unpaired"); a reciprocal file with no forward file is left out and
  named. Untick the box if a file was joined to the wrong partner. The run's
  `qc_report.txt` lists each time step's files and readings, and each
  survey's QC log in `qc/` begins with them.

## Reciprocal pairs: averaging and the error model

A reading and its reciprocal - current and potential electrodes exchanged -
measure the same transfer resistance, so their difference is a measured error
(Slater et al. 2000; Binley & Slater 2020).

- **Average each reading with its reciprocal** (Data QC, off by default) is
  check 0: each pair becomes one reading with the mean resistance, the pair's
  relative difference as its error (never below **Minimum** under Data errors), and the
  other checks run on what is left. The QC log says
  `Check 0: N pairs averaged. N redundant rows removed.`
- **Data errors → From the reciprocal error model** fits
  `log10(dR) = m·log10(R) + b` to the pairs - R the pair's mean resistance, dR
  the difference between its directions - over groups of pairs of equal count
  (Koestel et al. 2008), and gives every reading the
  relative error `dR / R = 10^b · R^(m-1)`, never below **Minimum** (1 %
  by default, the smallest error the in-house and ADTLERT time-lapse
  inversions take). A single inversion fits the survey's own pairs; a
  time-lapse run fits one model to the pairs of every survey and gives it to
  all of them. The settings file and `qc_report.txt` state m, b and R².
- The **Reciprocal errors** tab, beside Pseudosection, draws every pair - its
  reciprocal error against its mean resistance, a series' surveys in colour
  from first to last - with the model fitted through them. Its **Fit to**
  chooses the pairs the model is fitted to: all of them as read (the
  default), or only those the Data QC filter kept when Apply filter was
  pressed - the reciprocal-error limit under More checks is the check that
  removes outlier pairs, and the pairs left out are drawn in grey. The data
  errors, `qc_report.txt`, the settings file and the run's
  `reciprocal_error_model.png` all use that choice, and say which it was.

Slater, L., Binley, A., Daily, W., & Johnson, R. (2000). Cross-hole electrical
imaging of a controlled saline tracer injection. *Journal of Applied
Geophysics*, 44, 85-102. Koestel, J., Kemna, A., Javaux, M., Binley, A., &
Vereecken, H. (2008). Quantitative imaging of solute transport in an
unsaturated and undisturbed soil monolith with 3-D ERT and TDR. *Water
Resources Research*, 44, W12411. Binley, A., & Slater, L. (2020). *Resistivity
and Induced Polarization*. Cambridge University Press.

## Survey times

**Time tag format:** `YYYY_MM_DD_HH_MM_SS`, using a 24-hour clock.
For example: `site_2026_01_12_05_50_38.dat` means 12 January 2026 at 05:50:38.

Each survey's time heads its time step, sets the gaps between surveys, and dates
a single inversion. It is read from the first of these that has one:

1. **The file name.** Put the year first:
   - `site_2026-01-12_05-50-38.dat` (recommended)
   - `site_2026_01_12_05_50_38.dat`
   - `site_20260112_055038.dat`
   - `site_2026-01-12T05-50-38.dat`

   Seconds can be left out (`site_2026-01-12_05-50.dat`), and a date alone
   (`site_2026-01-12.dat`) is read as that day. Windows does not allow colons in
   file names, so use `-` or `_` in the time.
2. **A date line at the top of the file**, when the name has no time:
   `# date: 2026-01-12 05:50:38`. `# time`, `# datetime` and `# acquired`
   work too (`#time=2026-01-12T05:50:38`, `# acquired 2026-01-12 05:50`). In a
   BERT / unified file put it on the very first line, before the electrode
   count: pyGIMLi, ResIPy and the studio's own reader all skip it there, but
   between the electrode count and the `# x z` line pyGIMLi reads it as the
   column list. Some instruments (DAS-1, Subsurface Insights) already write a
   date in their header, and it is found the same way.
3. **The file's modified time**, only when **Use file times** is ticked. A
   copied or edited file carries the time of the copy, so this is off by default.

**Avoid month-first or day-first dates** such as `01-12-2026`: the same name is
also 1 December 2026. For a list, the reading that keeps the files in time
order is chosen and the summary says so; a single file cannot be checked.
A year-first name can be read only one way.

Times are used as written, without time-zone conversion. Hover over a file in
the list to see the time read for it and where it came from; hover over the
summary line under the list for these forms.
