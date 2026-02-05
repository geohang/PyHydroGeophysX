Seismic Structure Constraints
=============================

Use seismic velocity structure to constrain ERT inversion meshes and improve geological realism.

Steps
-----

1. Build or load seismic travel-time inversion outputs.
2. Extract velocity interfaces for structural boundaries.
3. Apply interfaces in resistivity inversion meshes.

.. code-block:: python

   from PyHydroGeophysX.core.mesh_utils import extract_velocity_interface

Related Examples
----------------

- :doc:`/auto_examples/EX_SRT_forward`
- :doc:`/auto_examples/Ex_SRT_inv`
- :doc:`/auto_examples/Ex_Structure_resinv`

