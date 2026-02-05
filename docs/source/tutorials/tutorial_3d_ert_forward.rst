3D ERT Forward Modeling
=======================

This tutorial shows how to build a small 3D ERT mesh that honors topography
and compute a forward response. It is based on `examples/Ex_3D_ERT_forward.py`
and the rendered example :doc:`/auto_examples/Ex_3D_ERT_forward`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_3D_ERT_forward_thumb.png
   :alt: 3D ERT forward example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- PyGIMLi installed.
- Optional: PyVista for 3D visualization.

Steps
-----

1. Create a small electrode grid and a topography-aware 3D mesh.
2. Build a 3D ERT data container for the survey.
3. Compute a forward response for a simple resistivity model.

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator, create_3d_ert_data_container
   from PyHydroGeophysX.forward.ert_forward import ERTForwardModeling

   creator = Mesh3DCreator(
       mesh_directory=".",
       elec_refinement=0.5,
       node_refinement=1.0,
       attractor_distance=3.0
   )

   elec = creator.create_surface_electrode_array(
       nx=2, ny=2, dx=5.0, dy=5.0, z=0.0
   )
   elec["z"] = 0.2 * elec["x"] + 0.1 * elec["y"]

   mesh = creator.create_3d_mesh_with_topography(
       electrode_positions=elec,
       topography_func=lambda x, y: 0.2 * x + 0.1 * y,
       para_depth=8.0,
       boundary_depth=2.0,
       para_max_cell_size=8.0,
       dz_fine=2.0,
       dz_coarse=4.0,
       use_prism_mesh=True
   )

   data = create_3d_ert_data_container(elec, scheme="wa", dimension=3)

   rho = np.full(mesh.cellCount(), 100.0)
   fwd = ERTForwardModeling(mesh, data)
   response = fwd.forward(rho, log_transform=False)

   print(response[:5])

Outputs
-------

- A 3D mesh with embedded electrodes.
- Synthetic apparent resistivity response for the survey.

Next
----

- Full example: :doc:`/auto_examples/Ex_3D_ERT_forward`.
