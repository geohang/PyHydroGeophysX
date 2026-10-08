Seismic refraction
==================

Travel-time tomography for P-wave velocity. Velocity responds strongly to
weathering and to the transition into competent bedrock, which makes it the
usual source of structural information for the electrical workflows.

What you can do
---------------

- Forward modelling of first-arrival travel times from a velocity model.
- Single-survey travel-time tomography.
- Time-lapse seismic inversion.
- Supply a recovered interface as a structural constraint on an ERT inversion.
- Joint ERT and seismic inversion with cross-gradient or geostatistical
  coupling.
- Predict velocity from hydrological states through rock-physics models.

Required inputs
---------------

First-arrival travel times with source and receiver positions. SEG-2 records
are read through ObsPy when it is installed, with a text fallback for arrays
that were already exported; pyGIMLi ``.dat`` and ``.sgt`` travel-time files are
read directly.

For the hydrology-driven direction you supply water content and porosity, and a
rock-physics model produces the velocity field.

Typical outputs
---------------

A velocity model on the inversion mesh with its data fit, and the interfaces
picked from it. Those interfaces are what the structure-constrained and joint
workflows consume.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - Generate travel-time data from a velocity model
     - :doc:`/auto_examples/EX_SRT_forward`
   * - Invert a refraction survey
     - :doc:`/auto_examples/Ex_SRT_inv`
   * - Use seismic structure to guide an ERT inversion
     - :doc:`/tutorials/seismic_structure_constraints`
   * - Invert ERT and seismic together
     - :doc:`/tutorials/joint_inversion`
   * - Predict velocity from a hydrological model
     - :doc:`/auto_examples/Ex_hydro_to_multigeophys`

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has a seismic processing module
and a 3D seismic module, alongside the joint-inversion module that pairs
seismic with ERT.

Python examples
---------------

- :doc:`/auto_examples/EX_SRT_forward`: generate travel times.
- :doc:`/auto_examples/Ex_SRT_inv`: recover a velocity section, compared
  against a direct pyGIMLi run.
- :doc:`/auto_examples/Ex_Structure_resinv`: structure from seismic applied to
  resistivity.
- :doc:`/auto_examples/Ex_joint_inversion`: both datasets inverted together.

A single inversion:

.. code-block:: python

   from PyHydroGeophysX.inversion import SRTInversion

   inv = SRTInversion(
       data_file="examples/data/Seismic/srtfieldline2.dat",
       lambda_val=50.0,
       max_iterations=20,
   )
   result = inv.run()
   print("velocity model cells:", result.final_model.size)

Joint inversion with ERT, which is where the two methods pay for each other:

.. code-block:: python

   from PyHydroGeophysX.inversion import JointERTSRTInversion

   joint = JointERTSRTInversion(
       ert_data="examples/data/ERT/Bert/fielddataline2.dat",
       srt_data="examples/data/Seismic/srtfieldline2.dat",
       regularization_mode="geostat",   # or "smoothness"
       cross_gradient_mode="direct",    # or "spatial"
       lambda_cg_ert=120.0,
       lambda_cg_srt=80.0,
   )
   result = joint.run()
   print(result.chi2_ert, result.chi2_srt)

Relevant API
------------

- :doc:`/api/forward` for ``SeismicForwardModeling``.
- :doc:`/api/inversion` for ``SRTInversion``, ``TimeLapseSRTInversion`` and
  ``JointERTSRTInversion``.
- :doc:`/api/petrophysics` for the velocity models, including Voigt-Reuss-Hill,
  differential effective medium, Hertz-Mindlin and Brie.

Limitations and optional dependencies
-------------------------------------

Forward modelling and inversion run on pyGIMLi, installed by the ``geophysics``
extra:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

Reading SEG-2 field records needs ObsPy. Travel times are first arrivals: the
package inverts picks, it does not pick them for you.
