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

See :ref:`mt-references` for the method references and complete BibTeX entries.

What you can do
---------------

- Read an instrument's time series with its calibrations: Phoenix MTU-5C,
  MTU-5P and MTU-8A, legacy Phoenix MTU-5A and V8, Metronix ADU (ATS), Zonge
  ZEN (Z3D) and LEMI-424.
- Estimate the impedance and tipper with robust, remote-referenced processing
  in the manner of EMTF: cascade decimation, Huber then redescending weights,
  and an error model from the regression's own residuals. For noisy
  recordings there are optional extras (see :ref:`mt-processing-options`):

  - removal of power-line hum;
  - removal of receiver interference;
  - bounded-influence weights;
  - coherence presorting;
  - multitaper spectra;
  - multi-window stacking;
  - noise classes;
  - jackknife errors.
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
     - Native and decimated streams; times in UTC; ``rxcal.json``,
       ``scal.json`` and binary ``.scal`` coil files found near the recording
       (its folder, the two above, a folder there named for calibrations).
       A receiver that did not detect its coils records serial 0: say which
       coil was on each connector with ``coils={"H1": "4684", ...}``.
       Interference repeating exactly on the receiver's GPS clock (a pulse
       every 4 s on some MTU-5Cs) is found with ``mt.periodic_interference``
       and subtracted with ``ProcessingConfig(periodic_noise=(4.0,))``.
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

A layout that differs from what the receiver recorded is given at reading:
``mt.read_timeseries(path, azimuths={"ex": 180, "ey": 270, "hy": 270})``
(degrees clockwise from north) replaces those channels' directions in every
run. Processing expresses every channel in the frame of Hx, so a dipole laid
out to the south or a coil pointing west needs nothing more. Both Manchester,
Iowa sites (October 2026) were laid out this way; read as recorded, their Zyx
came out of its quadrant. On the Desktop Studio page the same setting is
**Directions**, under Recording, and the run records it with the other reader
options.

A dipole that changed during a recording is split there:
``run.window(None, at)`` and ``run.window(at, None).with_dipole_length("ey", 50)``
give the part after the change its own length, and the pieces are processed
together. When one electrode of a line laid out about a ground electrode breaks
off, the receiver goes on measuring the other arm against the ground
electrode, half the line. At Osborne, Iowa (October 2026) the Ey line's
response then fell to half, in phase, until its noise grew.

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

Where Zxy and Zyx split, their phases too, a layered earth whose layers conduct
differently along two horizontal axes may explain both. ``occam1d_anisotropic``
fits one resistivity profile along the phase tensor's strike (or a strike you
give) and one across it, on the same layers; a layer is made anisotropic only
where the data need it (Pek and Santos 2002, for axes that share one azimuth).
One site cannot tell this from a 2D structure with the same strike, or from
galvanic distortion; with ``static_shift=True`` the modes' difference in level
goes to the shifts and the anisotropy comes from the phases alone.

.. code-block:: python

   model = mt.occam1d_anisotropic(tf, static_shift=True)
   print(model.strike, model.rms)
   print(model.depth, model.anisotropy)      # across / along, layer by layer
   along = model.axis(0)                     # an Occam1DResult, for the 1D plots

A line of sites in 2D (needs SimPEG):

.. code-block:: python

   sites = mt.read_transfer_functions("survey/edi")
   distance, azimuth = mt.station_distances(sites)
   section = mt.invert_profile(sites, distance, strike=(azimuth + 90) % 180)

.. _mt-processing-options:

Processing noisy recordings
---------------------------

``process_mt`` defaults to EMTF's estimate. ``ProcessingConfig`` (and the
Processing group of the studio's Magnetotellurics page) adds the following.
All of them are off unless asked for.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - What it does, and when it helps
   * - ``powerline=60.0``
     - Notches the grid's hum (50 or 60 Hz) and its harmonics out of every
       run, each line as wide as it actually is. ``mt.powerline_frequency``
       finds it. It matters most where the hum lies inside the record, such
       as 60 Hz in 150 Hz data. There it leaks through the taper into bands far
       from it, where it looks like a phase near 0° or 180° with resistivity
       rising in step with period. A remote site on the same grid does not
       remove it. A run shorter than a minute, such as a 2 s burst, has bins
       too wide for a notch: hum off its bins leaks across the spectrum. There
       the hum's harmonics of one fitted grid frequency are fitted and
       subtracted instead.
   * - ``drop_saturated=True``
     - Leaves out the windows holding a sample at the recorder's full scale,
       where the reader knows it (Phoenix). On by default: a clipped stretch
       gives an impedance that looks plausible but is wrong, with small
       errors.
   * - ``periodic_noise=(4.0, 30.0)``
     - Subtracts interference that repeats exactly on the receiver's clock.
       ``mt.periodic_interference`` finds it. Each channel is high-passed by
       its moving mean over one period, folded on the period and averaged, and
       that waveform subtracted: one time-invariant comb filter, the same on
       every channel, so the impedance between the harmonics is kept. A burst
       too short to fold on itself gets the mean of the other bursts that
       start at the same phase of the period subtracted instead.
   * - ``periodic_where_found=True``
     - On by default. A period is taken out of a long run only if
       ``mt.periodic_interference`` finds it in that run (when the run holds
       30 cycles or more, enough to tell). The comb also takes the field at
       every harmonic of the period, so removing interference a run does not
       carry only costs data, and where a level's windows cannot resolve the
       notches it biases the band. At Manchester the 30 s disturbance is on
       Baileys Ford's Ey only: removed from Coffins Grove as well, it pushed
       Coffins Grove Zxy's phase out of its quadrant at 0.8-2.5 s. Bursts too
       short to fold are cleaned for every period, as before; ``False`` takes
       every period out of every run.
   * - ``periodic_cycles=5``
     - How many repeats of that interference are averaged into the waveform
       subtracted from each one (31 by default). Fewer follow interference
       whose shape drifts; each notch is then a little wider.
   * - ``skip_notched=True``
     - On by default with ``periodic_noise``. It leaves out of the bands the
       harmonics inside those notches, where the windows are long enough to
       resolve them (twice the period or more). The notch takes the field
       there too, and what is left is the interference's drift. At Manchester
       that drift gave the harmonics near 0.25 Hz a phase near 0°, even after
       removal.
   * - ``widen_above=0.5``
     - A band whose Zxy and Zyx errors average above this share of :math:`|Z|`
       is estimated again from a band about twice as wide (Q 1.5 instead of
       3). The new band is at the same level and centre. The wider estimate is
       kept if its errors are smaller. This gives more data where the field is
       weakest, at the price of resolution. At Coffins Grove it filled the
       5-6.4 s bands, whose errors had been over 100 %.
   * - ``min_windows=4``
     - The fewest windows a decimation level needs (5 by default), counted
       after the trimmed ends and the dropped first and last windows. A night
       of 5-6 h gives the 200-600 s level only 4 windows of an hour.
   * - ``bounded_influence=True``
     - Also weights every point by its leverage (Chave and Thomson 2004), so
       bursts of noise in Hx or Hy cannot pull the estimate. The leverage
       weights accumulate over the iterations, so a point once downweighted
       stays so.
   * - ``min_coherence=0.5``
     - Coherence presorting (Egbert and Livelybrooks 1996): leaves out
       stretches of 10 windows where E and H are less coherent than this. It
       helps a single site; with a remote reference it can drop good data.
   * - ``min_reference_coherence=0.1``
     - With a remote reference, each band reports how coherent its Hx and Hy
       are with the remote's (``metadata["reference_coherence"]``). With this
       set, the elements estimated from an input the remote shares less than
       this (Hx: Zxx, Zyx; Hy: Zxy, Zyy, and the tipper's) get relative errors
       of 100 % or more. Off by default: low reference coherence alone does
       not make a band wrong when it has many points. USArray IAK40's 12 days
       give EMTF's impedance to 2-5 % at 5-12 s with a reference coherence of
       0.01-0.06. A biased band, though, can be steady through the record, so
       its jackknife errors look small; check such bands for physical
       consistency (phase inside its quadrant, apparent resistivity and phase
       agreeing).
   * - ``tapers=3``
     - Multitaper spectra (Thomson 1982): each window is tapered by 3 (or 5)
       Slepian sequences instead of one Hamming window. This gives smaller
       errors where windows are few, at the cost of smoothing over a few
       harmonics. At a band's lowest harmonics that smoothing reaches far down
       in frequency; where the field's power rises steeply towards long
       periods, as it does on either side of the dead band, the estimate is
       drawn towards the longer periods. The errors do not show this.
   * - ``multitaper_above=0.05``
     - Tapers only where they can pay: a band whose Zxy (or Zyx, or tipper)
       relative error is above this with one taper is estimated again with
       ``multitaper_tapers`` (3) tapers, output by output, and that estimate
       kept whatever its errors. Which of the two estimates is nearer the
       truth is not known, so their difference is added to the errors in
       quadrature: the curve is the steadier one, and the one-taper estimate
       lies within its errors. ``metadata["tapered"]`` lists the outputs
       estimated again in each band. Bands whose errors are a few per cent
       keep one taper. ``multitaper_periods=(0.1, 1000)`` keeps the tapers
       to a period range; in Manchester's 2 s bursts at 24 kHz they made the
       curves rougher, not steadier. Off by default; see the Manchester note
       below.
   * - ``multi_window=True``
     - Multi-window stacking (Ernst et al. 2001): each band also takes the
       windows of the neighbouring decimation levels, 4 times longer or
       shorter, at its frequencies. Errors then come from the jackknife.
   * - ``noise_classes=10``
     - Handles noise that grows with the signal and comes from several
       sources. Points are put into noise classes, each class is normalised
       by its own noise level, and the M-estimate is made again. The number
       of classes rises until the residuals' stable index reaches 1.8.
   * - ``jackknife=True``
     - Error bars from the delete-a-group jackknife (Chave and Thomson 1989)
       instead of the parametric model. Each group holds whole windows. A band
       with too few windows for it keeps the parametric errors; the others
       keep their own.
   * - ``bands_per_decade=10``
     - Overlapping bands (after Chen 2020): 10 frequencies per decade, each
       band holding the harmonics within a third of its centre. Neighbours
       share most of their data, so the curves are continuous, as in
       commercial processing; their errors are correlated, so invert EMTF's
       independent bands. The studio's page uses 10 by default.

Where two sample rates cover the same frequencies, the rate whose Zxy and Zyx
errors are smaller over that whole stretch takes it, so the curve does not hop
between rates. Each rate is estimated on its own and the curves are joined, as
in most processing codes. Within a rate, ``multi_window`` pools the
neighbouring window lengths at a frequency.

Impedance errors are one standard error of the real part and of the imaginary
part alike (the EDI convention). This is the complex error divided by
:math:`\sqrt{2}`.

Every band records its residuals' stable index (``metadata["residual_alpha"]``).
A value of 2 means Gaussian residuals; below about 1.8, the residuals are
heavy-tailed and the parametric errors are optimistic.

Judge an option by whether it makes the data agree with themselves, not by
whether it smooths the curve. Two checks do this:

- Process the two halves of a night separately and compare them.
- Look at the phase. Coherent noise in E and H gives a real impedance that
  does not change with frequency: phase near 0° or 180°, and resistivity
  rising in step with period. Such noise yields smooth curves with small
  errors. Where two sample rates overlap, the estimate with the smaller
  errors is kept, so it can win there.

At the Manchester, Iowa sites (Phoenix MTU-5C), splitting the night in two
and using a remote reference, 3 tapers halved the disagreement between the
halves at 1-10 s. 3 tapers with bounded influence gave the steadiest
10-100 s. Steadier is not shown to be right, though: 5 tapers above 5 %
error (``multitaper_above=0.05, multitaper_tapers=5``) cut the halves'
disagreement at 0.2-10 s from 24-67 % to 14-46 %, and also lowered
:math:`|Z|` there by 4-23 % (Coffins Grove Zyx by 10 %, more than the
one-taper errors), and 3 tapers everywhere lowered Baileys Ford's
well-determined 10-150 s by 2-4 %. On a synthetic layered earth, tapers raised
the bias of :math:`|Z|` from 0.3 % to 1-5 %. The halves share such a shift, so
their agreement cannot show it. The other tests are split: with the tapers
the two sites' dead-band Zyx phases agree better (the scatter of their
difference falls from 10° to 5°), while a 1D model fitted on either side of
the dead band predicts the one-taper level slightly better at Baileys Ford
and neither clearly better at Coffins Grove. Neither estimate is known to be
right, so ``multitaper_above`` keeps the steadier one and puts the difference
in its errors; at Manchester that keeps the dead band in the 1D inversions,
weighted by how little is known there. The 60 Hz hum in the 150 Hz records made 0.03-0.15 s look like
that coherent noise at both sites, remote reference or not. With the hum
notched out the phases there came back to 30-50°. The receivers' 4 s
interference changed shape within minutes. Averaging it over 5 repeats
instead of 31 raised the coherence between the two sites' Hx at 0.3-4 s
from about 0.2 to 0.8.

An earlier form of that subtraction took each cycle's own mean off. That
turned the field's slow drift into a staircase, with steps at every 4 s. The
steps gave both sites a 0.2-5 s "plateau" with a phase near 0° and
resistivity rising in step with period. Its ratio between the sites matched
their static shift, so it looked like the response of a conductive cover over
an insulator. Two checks gave it away:

- The apparent resistivity fell sixfold between 0.14 s and 0.2 s. A 1D earth
  cannot do that: :math:`|d\log\rho_a / d\log T|` stays at or under 1.
- The same period computed from two neighbouring decimation levels
  disagreed fivefold.

Compute a period from two levels whenever a curve steps where one level
hands over to the next.

Relevant API
------------

- :doc:`/api/data_processing` for the readers, ``process_mt``,
  ``ProcessingConfig``, the analysis functions, ``occam1d``, ``occam1d_anisotropic``,
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
