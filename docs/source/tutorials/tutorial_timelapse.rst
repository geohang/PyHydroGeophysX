Time-Lapse Inversion
====================

This tutorial shows how to run time-lapse ERT inversion and when to apply
structural constraints. It uses `examples/Ex_TL_inversion.py` and
:doc:`/auto_examples/Ex_TL_inversion`.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_TL_inversion_thumb.png
   :alt: Time-lapse inversion thumbnail
   :align: center
   :width: 70%

Steps
-----

1. Prepare a list of time-ordered datasets.

.. code-block:: python

   ert_files = ["t0.dat", "t1.dat", "t2.dat"]
   measurement_times = [0, 30, 60]

2. Configure the time-lapse inversion.

.. code-block:: python

   from PyHydroGeophysX.inversion import TimeLapseERTInversion

   inversion = TimeLapseERTInversion(
       data_files=ert_files,
       measurement_times=measurement_times,
       lambda_val=50.0,
       alpha=10.0
   )
   result = inversion.run()

3. Add structural constraints when available.

- Use SRT interfaces to constrain ERT meshes.
- See :doc:`/auto_examples/Ex_structure_TLresinv` for a full example.

Outputs
-------

- Time-lapse resistivity models
- Change maps suitable for monitoring studies

Next
----

- Structure constraints: :doc:`/auto_examples/Ex_Structure_resinv`
- Synthetic time-lapse surveys: :doc:`/auto_examples/Ex_Time_lapse_measurement`
