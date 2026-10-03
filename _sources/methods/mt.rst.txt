Magnetotellurics (MT and AMT)
=============================

Natural variations of the Earth's magnetic field induce electric currents in
the ground; the ratio of the electric to the magnetic field at the surface,
frequency by frequency, is the impedance, and it carries the resistivity from
a few metres (audio-magnetotellurics, kilohertz) to hundreds of kilometres
(long-period MT, hours). MT responds to the same property as ERT and TDEM, with
no source to lay out and far more depth, at the cost of resolution near the
surface and of a static shift that only a second method can remove.

The processing and the 1D inversion are written in the package itself and need
only NumPy and SciPy; the 2D inversion runs on SimPEG.

What you can do
---------------

- Read an instrument's time series with its calibrations: Phoenix MTU-5C,
  MTU-5P and MTU-8A, legacy Phoenix MTU-5A and V8, Metronix ADU (ATS), Zonge
  ZEN (Z3D) and LEMI-424.
- Estimate the impedance and tipper with robust, remote-referenced processing
  in the manner of EMTF: cascade decimation, Huber then redescending weights,
  optional leverage weights, and an error model from the regression's own
  residuals.
- Read and write transfer functions as EDI, EMTF XML, EMTF Z-files and
  J-files.
- Judge dimensionality and strike from the phase tensor, the Swift and Bahr
  skews and the induction arrows.
- Invert a site in 1D by Occam's method, with a static-shift multiplier per
  mode, and jointly with a central-loop TEM sounding that fixes the shift.
- Turn a layered model into a water-content profile through Waxman-Smits.
- Invert a profile's TE and TM impedances for a 2D section.

Required inputs
---------------

Either a recording, read with ``mt.read_timeseries``, or sites processed
elsewhere, read with ``mt.read_transfer_function``.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Format
     - Notes
   * - Phoenix MTU-5C / 5P / 8A, a recording folder or ``.bin`` / ``.td_*``
     - Native and decimated streams; times in UTC; ``rxcal.json`` and
       ``scal.json`` found beside the recording.
   * - Phoenix MTU-5A / V8, ``.TBL`` with ``.TS2`` to ``.TS5``
     - The table's gains and dipoles; Phoenix's CLB/CLC coil files are not
       decoded, so give coil tables through ``calibration``.
   * - Metronix ADU, ``.ats``
     - Header geometry and LSB; coil calibrations from the measurement XML or
       a ``cal`` folder, with the chopper setting of each channel.
   * - Zonge ZEN, ``.Z3D``
     - GPS-stamped blocks and the coil table stored in each file.
   * - LEMI-424, daily ``.TXT``
     - The 1 Hz fluxgate record; give the two dipole lengths.
   * - EDI, EMTF XML, ``.zss`` / ``.zrr`` / ``.zmm``, ``.j``
     - Transfer functions with their errors; EMTF XML and Z-files keep the
       full error covariance.

Typical outputs
---------------

Per site, the impedance and tipper with errors, written as EDI and EMTF XML,
and a table of apparent resistivity and phase. Per site inverted in 1D, a
layered resistivity model with its fit and, when asked, the static shift of
each mode and a water-content profile. Per profile, a 2D resistivity section.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - The whole chain on the MODFLOW model, and a field station
     - :doc:`/auto_examples/Ex_MT_workflow`
   * - Remove a static shift with a TEM sounding
     - Step 4 of :doc:`/auto_examples/Ex_MT_workflow`
   * - Compare MT against ERT and TDEM over the same subsurface
     - :doc:`/auto_examples/Ex_hydro_to_multigeophys` for ERT and TDEM, and
       :doc:`/auto_examples/Ex_MT_workflow`, which starts from the same
       MODFLOW model

The three registered workflows are ``mt.process`` (time series to transfer
functions), ``mt.invert_1d`` and ``mt.invert_profile``. Each runs through
``run_workflow`` and exports a recipe, a runner and a readable walkthrough like
every other workflow.

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has a Magnetotellurics page:
open a recording or add EDI / XML sites, process them, check the phase tensors,
and run the 1D (with TEM and water content) and 2D inversions, with every run
recorded in the Project. The AQUAH assistant inverts the MT sites a request
names (``mt_files``).

Python examples
---------------

- :doc:`/auto_examples/Ex_MT_workflow`: a MODFLOW column to AMT time series,
  processed back to an EDI file; the static shift fixed by TEM; water content;
  a 2D profile along the model; and the USMTArray station NMX20.

Process a recording and invert it in 1D:

.. code-block:: python

   from PyHydroGeophysX.data_processing import mt

   runs = mt.read_timeseries("survey/site01")          # Phoenix, Metronix, Zonge or LEMI
   remote = mt.read_timeseries("survey/remote")       # recorded at the same time
   tf = mt.process_mt(runs, remote=remote)
   mt.write_edi(tf, "site01.edi")

   model = mt.occam1d(tf, mode="det", static_shift=True)
   print(model.rms, model.static_shift)

A line of sites in 2D (needs SimPEG):

.. code-block:: python

   sites = mt.read_transfer_functions("survey/edi")
   distance, azimuth = mt.station_distances(sites)
   section = mt.invert_profile(sites, distance, strike=(azimuth + 90) % 180)

Relevant API
------------

- :doc:`/api/data_processing` for the readers, ``process_mt``,
  ``ProcessingConfig``, the analysis functions, ``occam1d``,
  ``water_content_profile`` and ``invert_profile``.
- :doc:`/api/visualization` for ``plot_mt_sounding``, ``plot_mt_model_1d``,
  ``plot_mt_dimensionality``, ``plot_phase_tensor_pseudosection`` and
  ``plot_mt_section``.

Limitations and optional dependencies
-------------------------------------

The 2D inversion and the joint TEM inversion use SimPEG, installed by the
``geophysics`` extra:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

MT alone fixes only the relative static shift between the two modes, not its
absolute level: that takes a TEM sounding at the site, or a known resistivity.
The 2D inversion assumes one strike for the whole profile; where the phase
tensor's skew exceeds about 3 degrees the ground is 3D and a 2D section is an
approximation. Legacy Phoenix coil calibrations (CLB/CLC) are not decoded.
