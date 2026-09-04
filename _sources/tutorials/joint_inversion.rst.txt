Joint ERT + SRT Inversion
=========================

This workflow jointly inverts ERT and SRT data with structural coupling.
It supports both smoothness regularization and geostatistical regularization.

When To Use
-----------

- You have collocated ERT and SRT surveys on the same profile.
- You want structural consistency between resistivity and velocity models.
- You want to test cross-gradient constraints with geostatistical priors.

Core API
--------

.. code-block:: python

   from PyHydroGeophysX.inversion import JointERTSRTInversion

   inv = JointERTSRTInversion(
       ert_data="examples/data/ert/field_line.dat",
       srt_data="examples/data/srt/field_line.sgt",
       regularization_mode="geostat",      # or "smoothness"
       cross_gradient_mode="direct",       # or "spatial"
       cross_gradient_source="smoothness", # or "covariance"
       lambda_ert=10.0,
       lambda_srt=10.0,
       lambda_cg_ert=120.0,
       lambda_cg_srt=80.0,
       max_iterations=50,
   )
   result = inv.run()

   print("ERT chi2:", result.chi2_ert)
   print("SRT chi2:", result.chi2_srt)

Factory Dispatch (Unified Interface)
------------------------------------

.. code-block:: python

   from PyHydroGeophysX.inversion import GeophysicalInversion

   joint = GeophysicalInversion(
       "joint_ert_srt",
       ert_data="examples/data/ert/field_line.dat",
       srt_data="examples/data/srt/field_line.sgt",
   )
   result = joint.run()

Related Examples
----------------

- :doc:`/auto_examples/Ex_joint_inversion`
- :doc:`/auto_examples/Ex_SRT_inv`
- :doc:`/auto_examples/Ex_hydro_to_multigeophys`
