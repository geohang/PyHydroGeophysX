.. _gpr:

Ground-penetrating radar (GPR)
==============================

A GPR antenna sends a short radar pulse into the ground and records its echoes
from changes in relative permittivity, which in a soil is set mostly by the
water it holds. A common-offset line, one trace every few centimetres, gives a
radargram: reflectors and diffraction hyperbolas in two-way travel time, a few
metres deep at 100-500 MHz.

The GPR tools are a first, simple framework: one line at a time, a single
velocity for the whole line, no picking yet. They are written in the package
(``PyHydroGeophysX.data_processing.gpr``) and need only NumPy and SciPy. The
processing steps use the standard sequence also used by the open-source GPR
codes GPRPy, readgssi, ImpDAR and RGPR; see :doc:`/citation`.

What you can do
---------------

- Read a line from GSSI ``.DZT``, Sensors & Software pulseEKKO ``.DT1`` with
  its ``.HD``, MALA ``.rd3``/``.rd7`` with its ``.rad``, SEG-Y, or a ``.npz``
  saved by the package.
- Clean it up, in this order:

  - remove the slow drift under the first arrivals (dewow);
  - move time zero onto the first arrival;
  - band-pass filter (zero-phase Butterworth);
  - remove the horizontal bands that antenna ringing leaves (background
    removal, over the whole line or a running window of traces);
  - migrate with a constant velocity (F-K, Stolt 1978);
  - gain the late part for display (power of time, or automatic gain control).
- Fit the radar velocity by drawing a diffraction hyperbola over the
  radargram, or set it from a relative permittivity (``v = c / sqrt(eps)``).
- Show the line in two-way time, in depth (``d = v t / 2``) or, with ground
  elevations along the line, in elevation.

Required inputs
---------------

One GPR line in a format listed above. A line recorded by time rather than by
distance (a GSSI file without scans per metre) has no trace spacing; give one,
and the traces are spaced evenly. For the elevation view, a CSV of
``distance, elevation`` in metres along the line (a pulseEKKO file's own
topography is used when it has one).

Typical outputs
---------------

The processed line as ``.npz`` (read back with ``gpr.read_gpr``), the
radargram as drawn (``.png``) and the processing recipe (``.json``), which
lists every step with its settings and what it did.

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has a GPR page under Geophysical
Data Processing, laid out like the ERT page. **Data** shows the original
recording: open or drop a GPR file, or choose **Use example**, then set the
signal filters and gain on the right. Every change updates the processed line
automatically. **Processed radargram** shows the result, with velocity, depth
and migration settings beside it. **Trace** compares one trace before and after
processing; **Spectrum** compares their amplitude spectra with the band-pass
marked. The filters remain available beside both diagnostic views.

To fit the velocity, open **Processed radargram**, press **Fit a hyperbola**,
click its top and change the velocity until the dashed curve lies on it. Choose
**Depth** or **Elevation** below the plot for the vertical axis. The result
and save buttons stay below the settings while those settings scroll.
**Save results** exports the processed line, its radargram and the recipe.

**Use example** simulates a 100 MHz line over the
MODFLOW catchment model in ``examples/data``: the model's water content and
porosity become permittivity through the CRIM mixing law, and each layer
boundary reflects.

.. _gpr-aquah:

In an AQUAH run
---------------

Give the AQUAH assistant a line under the **GPR line** role (``gpr_file``), or
name it in the request ("process LINE01.DZT at 0.08 m/ns"). A ``.DZT``,
``.DT1``/``.HD`` or ``.rd3``/``.rd7``/``.rad`` file is a GPR line whatever the
request calls it; a SEG-Y file is one when the request is about radar and not
seismic. A folder holding the MODFLOW model of ``examples/data`` stands for the
simulated example line. The run makes two steps before its report:

- ``load_gpr_line`` reads the line and draws it as recorded under
  ``raw_data/``;
- ``process_gpr_line`` runs the page's default recipe - dewow, time zero on
  the first arrival, the mean trace removed, automatic gain over 25 ns - and
  writes ``gpr/<line>_processed.npz``, the radargram and
  ``gpr/<line>_processing.json``. ``gpr_params`` adds a band-pass
  (``bandpass``), migration (``migrate``) or another gain, or replaces the
  recipe (``steps``).

A depth needs a velocity, and the run uses only one somebody gave: the
request's ``gpr_velocity`` (m/ns) or ``gpr_permittivity``, or the permittivity
in a GSSI file's header, which the report says is the value set on the
instrument. Without any, the radargram stays in two-way time, migration is
left out, and the report says that no depth is given and how to measure the
velocity. A velocity or permittivity the request does not contain is never
taken from the language model's reading of it.

.. _gpr-python-examples:

Python examples
---------------

:doc:`/auto_examples/Ex_GPR_workflow` runs the whole chain on the line
simulated over the MODFLOW model. It processes the line, then checks time zero
and the velocity against the CRIM permittivity that made it, so that a wrong
choice shows as nanoseconds and metres rather than as a plausible picture. It
ends with depth and elevation sections and the saved line.

On a line of your own:

.. code-block:: python

   from PyHydroGeophysX.data_processing import gpr

   line = gpr.read_gpr("LINE01.DZT")
   steps = [
       {"name": "dewow"},
       {"name": "time_zero"},
       {"name": "bandpass", "low_mhz": 100, "high_mhz": 1000},
       {"name": "background"},
       {"name": "migrate", "velocity": 0.1},
       {"name": "gain", "kind": "agc", "window_ns": 20},
   ]
   processed = gpr.process(line, steps)
   print("\n".join(processed.history))
   depth = gpr.time_to_depth(processed.time_ns, 0.1)
   gpr.save_profile(processed, "LINE01_processed.npz")

Optional dependencies
---------------------

None beyond NumPy and SciPy. SEG-Y is read with ``segyio`` or ObsPy when
installed, and with the package's own reader otherwise.
