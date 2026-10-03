Time-Lapse Monitoring
=====================

Workflow at a glance
--------------------

**Prepare:** A sequence of ERT surveys with compatible acquisition geometry and time ordering.

**Produce:** A resistivity model for each time and their temporal changes.

**Check / next step:** First reproduce one time step; then assess data fit and whether temporal smoothing suppresses real changes.

This workflow builds synthetic or field time-lapse ERT datasets and inverts temporal changes.

Steps
-----

1. Generate or import multi-time ERT measurements.
2. Configure temporal regularization.
3. Compare inversion strategies (full, windowed, L1/L2).

Run the examples in this order: generate measurements, invert the sequence,
then add structure or compare memory-oriented strategies. Keep the same input
data and error model when comparing strategies so changes in the results can
be attributed to the inversion choices.

Acquisition times
-----------------

A time-lapse sequence is only as readable as its time axis. The acquisition time
of each survey is read from the file name - ``site_2024-06-12_1430.dat``,
``20171105_1418.Data``, ``06-12-2024 14-30.dat`` and the other common instrument
layouts are all recognised - and from the file header when the name carries
nothing. A layout is accepted only when it reads every file in the set, so an
ambiguous one is resolved by the sequence rather than by guesswork.

What this buys you:

- panels titled with the acquisition date instead of ``Time step 3``;
- the gap since the previous survey, shown next to each file in the Studio and
  written to ``survey_times.csv`` beside the models, in hours or days rather than
  decimal days;
- the position in the year that the seasonal temperature model needs.

When nothing can be read, the run falls back to a sequential ``1..n`` index and
says so in the log and in the report - it does not quietly assume an interval.
In the Studio, *Use file times when the names carry none* opts into the
filesystem modification time as a last resort; it is off by default because a
copied or re-exported file carries the time of the copy.

.. code-block:: python

   from PyHydroGeophysX.data_processing.survey_timing import survey_timing

   timing = survey_timing(sorted(glob.glob("data/*.ohm")))
   print(timing.summary())
   # 10 surveys, 2021-10-08 14:00 to 2022-06-26 00:30 (span 260 d 10 h), times
   # from the file names. Interval: median 30 d 22 h, 17 d 12 h to 31 d 00 h -
   # the sampling is irregular.

Temperature correction
----------------------

Bulk resistivity falls by about 2 % per °C, so across a monitoring season the
temperature signal is the same size as the moisture signal the survey is run to
see: an uncorrected series shows the ground "drying" as it cools. Correcting each
survey to one reference temperature leaves the remaining change attributable to
water.

In the Studio, run the inversion, then set the **Temperature correction** panel
beside the section on the **Resistivity model** tab and press **Apply** (see
`Correcting a result that is already inverted`_). In a script, pass
``temperature_correction`` in the inversion parameters:

.. code-block:: python

   params["temperature_correction"] = {
       "enabled": True,
       "model": "hayley",     # or "linear", with alpha (default 0.025 per degC)
       "reference": 25.0,     # degC the corrected sections are reported at
       "mode": "surface",     # constant | profile | surface | seasonal
       "surface_times": [...],        # dates or days of the surface record
       "surface_temperature": [...],  # degC at the ground surface
       "diffusivity": 0.06,           # m2/day
   }

The temperature comes from one of four sources: a constant, a **measured profile**
(with depth, and through time when it was logged that way), a **surface record
diffused into the ground**, or the analytical damped annual wave. In a scripted
run the corrected models become the series every panel, change plot and export is
built from; the raw ones are kept beside them as ``final_models_uncorrected.npy``.
Every run states which it produced - in the log, in the figure title and in the
report - because a corrected section is indistinguishable from an uncorrected one
on the page.

Measured profile, depth × time
``````````````````````````````

A thermistor string is the best temperature data a site can have: it measures the
depth structure and how it evolves, so nothing has to be modelled.
``load_temperature_profiles`` - and the panel's **Profile table** - read it in
whichever layout the logger wrote, told apart by the file's own shape:

``date, depth_m, temperature_C``
    one reading per row, the long form a database exports;
a header of depths, then a row per date
    one column per sensor, the wide form a logger exports;
a header of dates, then a row per depth
    the same table transposed, as it is often typed by hand;
``depth_m, temperature_C``
    a single profile, held constant in time.

Sensor headers may carry a unit or a prefix (``0.5m``, ``50 cm``, ``T_0.5``), the
time may span two columns (``date, time``), and gaps - empty cells, ``NaN``,
``-9999`` - are filled in time for each sensor.

The record is interpolated linearly in time and then in depth onto every cell and
survey. Before and after its span, and above its shallowest sensor, the nearest
reading is held. Below its deepest sensor the seasonal swing that sensor recorded
is damped with depth towards its mean, at the annual damping depth of the
``diffusivity`` option (2.6 m by default), because conduction damps it; holding the
deepest reading instead would carry a shallow string's full season to every
deeper cell. The panel says what it read - how many depths over which interval,
how many times over which dates - and warns when some surveys fall outside the
record, before anything is applied.

Surface record, 1-D conduction
``````````````````````````````

The mode most sites can actually use. Almost nobody has a thermistor string, and
almost everybody has a surface logger - or an air-temperature series from a
nearby station, which is a workable stand-in. ``mode: "surface"`` solves 1-D heat
conduction, :math:`\partial T/\partial t = \alpha\, \partial^2 T/\partial z^2`,
driven by that record and insulated at depth, and samples the result at every
cell depth and survey date.

The ground's own physics then supplies the depth structure: the annual swing
damps as ``exp(-z/d)`` and lags by ``z/d`` radians, so a cold snap that matters at
0.3 m has all but vanished at 3 m. Only the thermal diffusivity has to be chosen -
soils run about 0.04 to 0.13 m²/day - and the panel reports the annual damping
depth it implies, which is a number with an intuition attached:

.. code-block:: python

   from PyHydroGeophysX.petrophysics.temperature import (
       annual_damping_depth, simulate_temperature_1d)

   annual_damping_depth(0.06)      # 2.64 m
   field = simulate_temperature_1d(days, surface_degC, cell_depths, survey_days,
                                   diffusivity=0.06)

A two-column CSV (a date or a number of days, then the temperature) is read by
``load_surface_temperature`` and by the **Browse…** button in the panel. Driven by
a clean annual sinusoid the simulation reproduces the analytical damped wave to
better than 0.1 °C, which is the check in the test suite.

One temperature per survey belongs here too, as a coarse surface record. There is
deliberately no mode that applies such a value uniformly over depth: it would
correct a cell ten metres down by what happened at the surface, which that cell
never felt, and so manufacture a change at depth that the ground never had.

The record and the surveys have to be on one clock: dates on both sides, or plain
days on both sides. A dated record against undated surveys is refused rather than
guessed, because an offset of a few months in the forcing is a sign error in the
correction, not a small one.

Correcting a result that is already inverted
````````````````````````````````````````````

The correction is post-processing, so it never needs the inversion to be re-run,
and in the Studio it is always applied this way - to a single inversion as much as
to a time-lapse series. The same panel sits beside the section on the ERT page's
**Resistivity model** tab and beside a reopened section in **Project → Saved
Results**. Choose the settings and press **Apply**: the model on screen is
corrected there and then, its title and the panel say what temperature it is now
reported at, and **Remove** puts the inverted model back. Nothing is switched on in
advance, and the settings are shared between the two pages. A single survey is
placed in time by the date in its file name or header, which a dated record or the
seasonal model needs.

On the ERT page the exports then write the corrected model beside the inverted
one, never in place of it: ``final_models_temperature_corrected.npy`` and
``timelapse_resistivity_temperature_corrected.vtk`` for a series,
``resistivity_model_temperature_corrected.npy`` and ``.vtk`` for a single model,
the cell CSV holding what is on screen, and ``temperature_correction.json``
recording what was applied. ``final_models.npy``, ``resistivity_model.npy``, the
per-step VTKs and the figure remain the models as inverted.

Weighting the temporal constraint by the interval
-------------------------------------------------

The temporal regularization penalizes the difference between adjacent surveys.
On an evenly sampled series that is the right question; on a campaign that samples
hourly for a week and then monthly it is not, because a one-hour change and a
one-month change are held equally still, and real change across the wide gaps is
pushed into the data misfit instead.

Each adjacent pair is therefore weighted by ``1/Δt``, which turns the penalty into
one on the *rate* of change. The weights are normalized by the median interval, so
an evenly sampled series is bit-for-bit unchanged and ``alpha`` keeps the meaning
it always had; only irregular sampling is affected. A cap (10× either way by
default) keeps two surveys minutes apart inside a monthly series from being
effectively frozen together.

.. code-block:: python

   params["temporal_weighting"] = "interval"   # the default; "uniform" is the old behaviour
   params["temporal_weight_limit"] = 10.0      # None removes the cap

The run logs what it applied and records it under ``temporal_weighting`` in the
result. In the Studio it is the *Weight by the interval between surveys* tick in
the time-lapse panel. The GPU (ADTLERT) backend builds its own temporal operator
and constrains every pair equally; the run says so rather than letting the setting
look as though it applied.

Reading a result someone else inverted
--------------------------------------

Saved runs open in **Project → Saved Results** with the same section controls the
processing modules have: step through the surveys by acquisition date, switch to
**% change from baseline** on a symmetric scale, clip, smooth, and add the
section - in whichever of the two views is on screen - to the Project Map. A
result that has already been inverted does not need to be inverted again to be
read, compared or mapped.

Clipping the sections
---------------------

An inverted model is drawn on the whole parameter mesh, so the outer edges of the
picture are regularization rather than data. The model view's **Hide below**
blanks cells under a coverage cut; **Clean cut** next to it reduces that same cut
to a smooth clipping depth and clips the drawing to it, which gives the
traditional clean-edged resistivity image instead of a saw-tooth of surviving
triangles. *Clip sections to the coverage envelope* applies the same cut to the
exported panels, with one envelope for the whole sequence so the steps stay
comparable.

How smoothly the section is drawn
---------------------------------

**Smooth** next to *Show mesh* sets how the model is rendered:

``Off``
    the inversion's own cells, which is honest about where the model's degrees of
    freedom are;
``Cells ×1`` / ``Cells ×2``
    the model interpolated onto a once- or twice-subdivided mesh, still drawn as
    cells;
``Contour``
    filled contours on a regular grid, blanked above the ground surface - the
    continuous image traditional resistivity software produces. The **levels**
    box next to it sets the number of bands: a high count reads as a continuous
    image, a low one bands the section so a value can be read off it.

All three are display choices. No resolution is added and the exported model is
unchanged.

Related Examples
----------------

- :doc:`/auto_examples/Ex_Time_lapse_measurement`
- :doc:`/auto_examples/Ex_TL_inversion`
- :doc:`/auto_examples/Ex_TL_inversion_memory`
- :doc:`/auto_examples/Ex_structure_TLresinv`

