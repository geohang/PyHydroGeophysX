Time-Lapse Monitoring
=====================

Workflow at a glance
--------------------

**Prepare:** A sequence of ERT surveys with compatible acquisition geometry and time ordering.

**Produce:** A resistivity model for each time and their temporal changes.

**Check / next step:** First reproduce one time step; then assess data fit and whether temporal smoothing suppresses real changes.

This workflow builds synthetic or field time-lapse ERT datasets and inverts temporal changes.

Steps
-----

1. Generate or import multi-time ERT measurements.
2. Configure temporal regularization.
3. Compare inversion strategies (full, windowed, L1/L2).

Run the examples in this order: generate measurements, invert the sequence,
then add structure or compare memory-oriented strategies. Keep the same input
data and error model when comparing strategies so changes in the results can
be attributed to the inversion choices.

Related Examples
----------------

- :doc:`/auto_examples/Ex_Time_lapse_measurement`
- :doc:`/auto_examples/Ex_TL_inversion`
- :doc:`/auto_examples/Ex_TL_inversion_memory`
- :doc:`/auto_examples/Ex_structure_TLresinv`

