PyHydroGeophysX Documentation
==============================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

Welcome to PyHydroGeophysX
---------------------------

A comprehensive Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in electrical resistivity tomography (ERT) and seismic refraction tomography (SRT) for watershed monitoring applications.

🌟 Key Features
---------------

* **Hydrological Model Integration**: Seamless loading and processing of MODFLOW and ParFlow outputs
* **Petrophysical Relationships**: Advanced models for converting between water content, saturation, resistivity, and seismic velocity
* **Forward Modeling**: Complete ERT and SRT forward modeling capabilities
* **Time-Lapse Inversion**: Sophisticated algorithms for time-lapse ERT inversion with temporal regularization
* **Structure-Constrained Inversion**: Integration of seismic velocity interfaces for constrained ERT inversion
* **Uncertainty Quantification**: Monte Carlo methods for parameter uncertainty assessment

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from PyHydroGeophysX import (
       MODFLOWWaterContent, 
       ProfileInterpolator,
       water_content_to_resistivity,
       TimeLapseERTInversion
   )

   # Load hydrological model output
   processor = MODFLOWWaterContent("path/to/modflow", idomain)
   water_content = processor.load_timestep(0)

   # Convert to resistivity
   resistivity = water_content_to_resistivity(
       water_content, rhos=100, n=2.2, porosity=0.3
   )

   # Run time-lapse inversion
   inversion = TimeLapseERTInversion(
       data_files=data_files,
       measurement_times=times,
       lambda_val=50.0,
       alpha=10.0
   )
   result = inversion.run()

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`