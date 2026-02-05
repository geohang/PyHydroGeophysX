Time-Lapse Monitoring
=====================

This workflow builds synthetic or field time-lapse ERT datasets and inverts temporal changes.

Steps
-----

1. Generate or import multi-time ERT measurements.
2. Configure temporal regularization.
3. Compare inversion strategies (full, windowed, L1/L2).

.. code-block:: python

   from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion
   from PyHydroGeophysX.inversion.windowed import WindowedTimeLapseERTInversion

Related Examples
----------------

- :doc:`/auto_examples/Ex_Time_lapse_measurement`
- :doc:`/auto_examples/Ex_TL_inversion`
- :doc:`/auto_examples/Ex_structure_TLresinv`

