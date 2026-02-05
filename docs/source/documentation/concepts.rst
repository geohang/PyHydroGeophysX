Concepts
========

PyHydroGeophysX treats hydrogeophysics as a staged pipeline. Each stage produces
artifacts used by the next stage, and each has its own assumptions.

Workflow Stages
---------------

- Hydrology output: water content, saturation, or porosity fields from MODFLOW or ParFlow.
- Petrophysics: convert hydrologic properties to resistivity or velocity using Archie,
  Waxman-Smits, DEM, or Hertz-Mindlin models.
- Geophysics: simulate ERT, SRT, or TDEM data and design surveys.
- Inversion: recover resistivity or velocity models, optionally time-lapse or
  structure-constrained.

Typical Inputs and Outputs
--------------------------

- Inputs: model grids, time series, survey geometry, electrode or receiver layouts.
- Outputs: synthetic data, inverted models, quality metrics, and uncertainty estimates.

Related pages
-------------

- :doc:`hydrology_io`
- :doc:`petrophysics`
- :doc:`geophysics_ert`
- :doc:`timelapse_constraints`
