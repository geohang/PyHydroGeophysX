Examples Gallery
================

For categories, outcomes and a recommended starting point, open
:doc:`Examples by task </examples/index>`.

For a compact model–data interaction with actual hydrological solver execution,
start with :doc:`/tutorials/modflow_feedback` and
:doc:`/auto_examples/Ex_MODFLOW_geophysics_feedback`.

The central theme is the interaction between geophysical data and hydrological
models: predicting observations from model states and interpreting data as
hydrological information. See :doc:`/tutorials/hydro_geophysical_interaction`.

Open the notebooks in JupyterLab or VS Code and run the cells in order: inspect
the data, edit parameters, run the model, then view the results. Every scientific
example has a companion ``.ipynb``; the ``.py`` versions use ``# %%`` cells for
editors and website generation.

This gallery contains Python scripts and notebook downloads. It spans
hydrological model access, petrophysical conversion, multi-method responses,
field inversion, time-lapse monitoring, joint inversion and uncertainty analysis.

Start with :doc:`/auto_examples/Ex_model_output`, then
:doc:`/auto_examples/Ex_ERT_workflow`. Use
:doc:`/auto_examples/Ex_hydro_to_multigeophys` to see how one hydrological profile
connects to several geophysical methods. Review each script’s input paths and
dependencies before running it; the site uses pre-generated figures.

Run order, run time and memory
------------------------------

Most examples read only the shipped files under ``examples/data``. Three read
what another example writes under ``examples/results`` and stop with
``FileNotFoundError`` when it is missing:

* ``Ex_SRT_inv`` reads ``results/SRT_forward/synthetic_seismic_data_long.dat``.
  Run ``EX_SRT_forward`` first; it takes about a minute.
* ``Ex_structure_TLresinv`` reads ``results/Structure_WC/mesh_with_interface.bms``.
  Run ``Ex_Structure_resinv`` first.
* ``Ex_MC_Hydro`` reads the models that ``Ex_structure_TLresinv`` writes to
  ``results/Structure_WC``. Run ``Ex_Structure_resinv`` and then
  ``Ex_structure_TLresinv`` first, which takes about 3 hours.

The heavier examples, as measured on the maintainer's machine:

=============================  ============  =============
Example                        Run time      Memory peak
=============================  ============  =============
``Ex_ERT_workflow``            about 12 min  about 13.5 GB
``Ex_TL_inversion``            about 42 min  about 15 GB
``Ex_structure_TLresinv``      about 3 h
``Ex_joint_inversion``         about 44 min
``Ex_hydro_to_multigeophys``                 about 8 GB
=============================  ============  =============

Survey diagnostics and data assimilation on the MODFLOW model
-------------------------------------------------------------

These three examples start from the Treeline catchment MODFLOW model under
``examples/data`` and follow the hillslope transect of ``Ex_ERT_workflow``.
They run the package's own profile interpolation, petrophysics, ERT forward
model and inversion, so they need PyHydroGeophysX with pyGIMLi. Each takes
two to three minutes on the maintainer's machine, uses fixed random seeds,
writes its arrays and figure under ``examples/results/`` and has a companion
notebook.

* ``Ex_sensitivity_analysis.py``: predicts the ERT survey of a MODFLOW state
  with ``hydro_to_ert``, inverts it, and maps cumulative sensitivity, model
  resolution and the Oldenburg-Li depth of investigation over the MODFLOW
  units.
* ``Ex_posterior_uncertainty.py``: inverts the same survey, reads the inversion
  as a Gaussian posterior and propagates its covariance into water-content
  intervals, which it checks against the MODFLOW water content, at two
  strengths of the prior.
* ``Ex_ensemble_assimilation.py``: updates an ensemble of MODFLOW states with
  the day-210 survey in ``data/TL_measurements``, by EnKF and ES-MDA.

To use a ParFlow run instead, read saturation and porosity with
``ParflowSaturation`` and ``ParflowPorosity``; their product is the water
content. ParFlow numbers its layers from the bottom up, so reverse the layer
axis, and give the layer elevations of the ParFlow grid in place of ``top.txt``
and ``bot.npy``.

From the repository root, for example::

    python -m pip install -e ".[geophysics]"
    python examples/Ex_ensemble_assimilation.py
    python -m pytest -q tests/test_core.py -k "posterior or sensitivity or resolution or depth_of_investigation"
