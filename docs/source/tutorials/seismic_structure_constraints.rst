Seismic Structure Constraints
=============================

Use seismic velocity structure to constrain ERT inversion meshes and improve geological realism.

Steps
-----

1. Build or load seismic travel-time inversion outputs.
2. Extract velocity interfaces or boundary weights for structural boundaries.
3. Apply interfaces in resistivity inversion meshes or joint inversion constraints.

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.inversion.cross_constraints import StructuralConstraint

   velocity = np.array([800.0, 900.0, 2000.0, 2200.0])
   boundary_weights = StructuralConstraint.from_velocity_model(
       velocity_model=velocity,
       mesh=mesh,
       gradient_threshold=0.3,
   )

Related Examples
----------------

- :doc:`/auto_examples/EX_SRT_forward`
- :doc:`/auto_examples/Ex_SRT_inv`
- :doc:`/auto_examples/Ex_Structure_resinv`
- :doc:`/auto_examples/Ex_joint_inversion`
