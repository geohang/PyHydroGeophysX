Windowed Time-Lapse Inversion Setup
===================================

This tutorial shows how to configure windowed time-lapse inversion for large
monitoring datasets. It complements :doc:`/auto_examples/Ex_TL_inversion`
and the paper’s monitoring workflow.

.. figure:: /auto_examples/images/thumb/sphx_glr_Ex_TL_inversion_thumb.png
   :alt: Time-lapse inversion example thumbnail
   :align: center
   :width: 70%

Prerequisites
-------------

- PyGIMLi installed.

Steps
-----

1. Define the ordered list of ERT files and their measurement times.
2. Configure the window size and inspect the window indices.
3. Run the inversion on real data (optional).

.. code-block:: python

   from PyHydroGeophysX.inversion.windowed import WindowedTimeLapseERTInversion

   ert_files = [f"t{i}.dat" for i in range(5)]
   measurement_times = [0, 10, 20, 30, 40]

   windowed = WindowedTimeLapseERTInversion(
       data_dir=".",
       ert_files=ert_files,
       measurement_times=measurement_times,
       window_size=3
   )

   print(windowed.window_indices)
   print(windowed.mid_idx)

   # To run on real data:
   # result = windowed.run(window_parallel=False)

Outputs
-------

- Window indices and the middle index used for stitching results.

Next
----

- Full time-lapse example: :doc:`/auto_examples/Ex_TL_inversion`.
