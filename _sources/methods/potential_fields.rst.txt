Gravity and magnetics
=====================

Potential-field methods. They respond to density and magnetic susceptibility
rather than to water, so they sit further from the hydrological question than
the other methods here; what they contribute is the geometry the water moves
through, over areas too large to cover with a ground survey.

What you can do
---------------

- Inspect an anomaly and separate the regional field from the residual.
- Forward model the response of buried bodies, including prisms, spheres and
  dipoles.
- Compact 3D inversion for density contrast or susceptibility.
- Joint gravity and magnetics inversion on a shared mesh.

Required inputs
---------------

A column table of station coordinates with the observed gravity disturbance or
magnetic anomaly, and, for magnetics, the inducing field parameters. Two field
datasets ship with the package: a Bushveld gravity disturbance and a British
aeromagnetic anomaly, both under ``examples/data/Gravity_Magnetics``.

Typical outputs
---------------

A separated regional and residual field, and a 3D property model on a tensor
mesh with the data it reproduces.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 46 54

   * - Your goal
     - Start here
   * - Invert a gravity or magnetic survey
     - :doc:`/auto_examples/Ex_gravity_magnetics_inversion`
   * - See what these methods add beside ERT, seismic and EM
     - :doc:`/auto_examples/Ex_hydro_to_multigeophys`

Desktop support
---------------

:doc:`Desktop Studio </agents/desktop_studio>` has a gravity and magnetics
processing module.

Python examples
---------------

- :doc:`/auto_examples/Ex_gravity_magnetics_inversion`: inspect the anomalies,
  forward model and run compact 3D inversion on both bundled datasets.
- :doc:`/auto_examples/Ex_hydro_to_multigeophys`: the potential-field response
  of a hydrological profile, beside the responses of the other methods.

.. code-block:: python

   from PyHydroGeophysX.inversion import invert_gravmag

   result = invert_gravmag(x, y, value, kind="gravity", detrend=1)

Relevant API
------------

- :doc:`/api/forward` for ``gravity_prism``, ``gravity_sphere``,
  ``magnetic_dipole`` and ``forward_bodies``.
- :doc:`/api/inversion` for ``invert_gravmag`` and
  ``JointGravityMagneticsInversion``.
- :doc:`/api/data_processing` for the table readers and the
  regional-residual separation.

Limitations and optional dependencies
-------------------------------------

Inversion runs on SimPEG and discretize with a usable sparse solver, all
installed by the ``geophysics`` extra:

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

When that stack is missing, ``invert_gravmag`` raises
``InversionBackendUnavailable`` rather than falling back silently; the forward
routines still run, since they need only NumPy.

Potential-field data constrain density and susceptibility, not saturation. Use
these methods for structure, and read the water content from ERT, EM or
seismic.
