TDEM and FDEM
=============

Electromagnetic induction, in the time domain and in the frequency domain. Both
respond to electrical conductivity, the same property ERT responds to, but they
cover ground far faster and without electrodes, at the cost of resolving a
layered model rather than a full section.

What you can do
---------------

- Forward modelling of layered-earth soundings, time domain and frequency
  domain.
- Single-sounding inversion.
- Laterally constrained inversion along a line, so neighbouring soundings are
  tied to one another.
- Low-moment and high-moment TDEM fitted together to a single layered model.
- Joint FDEM and TDEM inversion.
- Predict soundings directly from hydrological states.

Required inputs
---------------

A sounding or a line of soundings with the system geometry, waveform and gate
times. The readers below cover what the common instruments write.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Format
     - Notes
   * - TEM2Go project, ``.tiw`` or ``.db``
     - Station stacks, gate flags, errors, geometry and saved inversion
       settings. Both project layouts are detected automatically.
   * - TEM2Go acquisition folder, ``.stb``
     - A folder copied straight off the instrument, with no project and no
       export. Gate times, calibration and loop geometry come from the raw
       stream.
   * - ``_StationData.xyz``, ``_RawData.xyz``
     - Self-describing exports. Station stacks are preferred for inversion.
   * - tTEM raw, ``.skb`` with ``.sps``
     - Uses the system GEX for geometry, waveform, gates and calibration, and
       the TFI import filter when both are present.
   * - Column tables, ``.csv``
     - Airborne and synthetic datasets with a geometry file alongside.

Gate-level quality control applies the acquisition protocol's own thresholds,
so a survey is filtered on the terms it was recorded under.

Typical outputs
---------------

A layered conductivity model per sounding, with the fit to the observed decay
or the observed in-phase and quadrature response. Along a line, the laterally
constrained result is a stitched conductivity section.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - Predict TDEM or FDEM data from a hydrological model
     - :doc:`/tutorials/hydrology_to_tdem`
   * - Invert a line of airborne soundings
     - :doc:`/auto_examples/Ex_EM_line_section`
   * - Fit low and high moment together
     - :doc:`/auto_examples/Ex_TEM_LMHM_LCI`
   * - Run a frequency-domain workflow end to end
     - :doc:`/auto_examples/Ex_FDEM_workflow`
   * - Compare EM against ERT over the same subsurface
     - :doc:`/auto_examples/Ex_hydro_to_multigeophys`

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has an EM processing module
covering import, gate quality control, inversion settings and the recovered
section.

Python examples
---------------

- :doc:`/auto_examples/Ex_TDEM_workflow`: hydrology to a layered model to a
  simulated sounding, then back by inversion.
- :doc:`/auto_examples/Ex_FDEM_workflow`: the frequency-domain equivalent.
- :doc:`/auto_examples/Ex_EM_line_section`: calibrate and invert a line of
  airborne soundings.
- :doc:`/auto_examples/Ex_TEM_LMHM_LCI`: both moments with line constraints,
  using the bundled nine-station project.

A self-contained frequency-domain round trip, forward then inverse, with no
data file involved:

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.forward import FDEMForwardModeling, FDEMSurveyConfig
   from PyHydroGeophysX.inversion import FDEMInversion

   thicknesses = np.array([5.0, 10.0, 20.0])      # 4 layers -> 3 thicknesses
   sigma_true = np.array([0.01, 0.02, 0.05, 0.08])
   cfg = FDEMSurveyConfig(frequencies=np.logspace(2, 4, 12))

   fwd = FDEMForwardModeling(thicknesses=thicknesses, survey_config=cfg)
   dobs = fwd.forward(sigma_true)
   uncert = 0.05 * np.maximum(np.abs(dobs), 1e-12)

   inv = FDEMInversion(
       frequencies=cfg.frequencies,
       dobs=dobs,
       uncertainties=uncert,
       thicknesses=thicknesses,
       receiver_component="secondary",
   )
   result = inv.run()
   print("FDEM chi2:", result.chi2)

Relevant API
------------

- :doc:`/api/forward` for ``TDEMForwardModeling``, ``FDEMForwardModeling``,
  the survey configuration classes and ``hydro_to_tdem``.
- :doc:`/api/inversion` for ``TDEMInversion``, ``FDEMInversion``,
  ``tdem_joint_invert``, ``JointFDEMTDEMInversion`` and the laterally
  constrained routines.

Limitations and optional dependencies
-------------------------------------

The EM forward and inverse solvers are SimPEG, installed by the ``geophysics``
extra:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

The models are 1D layered earths, stitched laterally rather than solved as a
single 2D or 3D system. Where the subsurface varies strongly across a sounding
footprint, ERT resolves it and EM does not.
