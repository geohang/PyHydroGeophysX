Quickstart
==========

Welcome to **PyHydroGeophysX**—a Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in ERT and SRT for watershed monitoring.

Key Features
------------
- Hydrological model integration (MODFLOW, ParFlow)
- Advanced petrophysical relationships (Archie, Waxman-Smits, DEM, Hertz-Mindlin)
- Forward modeling and time-lapse inversion for ERT & SRT
- Structure-constrained inversion and uncertainty quantification
- Optional GPU acceleration and parallel processing

Installation
------------
PyHydroGeophysX requires **Python 3.8 or higher**.

**From PyPI** (when available):
.. code-block:: bash

   pip install PyHydroGeophysX

**From source** (recommended for latest features):
.. code-block:: bash

   git clone https://github.com/geohang/PyHydroGeophysX.git
   cd PyHydroGeophysX
   pip install -e .

Install core dependencies if needed:
.. code-block:: bash

   pip install numpy scipy matplotlib pygimli tqdm

For optional GPU and parallel processing:
.. code-block:: bash

   pip install cupy-cuda11x  # Replace '11x' with your CUDA version
   pip install joblib

Basic Example: Hydrological Model Integration
---------------------------------------------
Load and process MODFLOW or ParFlow outputs:

.. code-block:: python

   from PyHydroGeophysX.model_output import MODFLOWWaterContent, ParflowSaturation

   # For MODFLOW
   processor = MODFLOWWaterContent("path/to/modflow_workspace", idomain)  # idomain: array for active model domain
   water_content = processor.load_time_range(start_idx=0, end_idx=10)

   # For ParFlow
   saturation_proc = ParflowSaturation("path/to/parflow_model_dir", "parflow_run_name")
   saturation = saturation_proc.load_timestep(100)

Petrophysical Modeling
----------------------
Convert water content to resistivity, or water content to seismic velocity:

.. code-block:: python

   from PyHydroGeophysX.petrophysics import water_content_to_resistivity, HertzMindlinModel

   resistivity = water_content_to_resistivity(
       water_content=wc, rho_saturated=100.0, saturation_exponent_n=2.0, 
       porosity=porosity, sigma_surface=0.002
   )

   hm_model = HertzMindlinModel()
   vp_high, vp_low = hm_model.calculate_velocity(
       porosity=porosity, saturation=saturation,
       matrix_bulk_modulus=30.0, matrix_shear_modulus=20.0, matrix_density=2650.0
   )

Forward Modeling (ERT/SRT)
--------------------------
Generate synthetic geophysical data (requires mesh/data creation as in examples):

.. code-block:: python

   from PyHydroGeophysX.forward import ERTForwardModeling

   ert_fwd = ERTForwardModeling(mesh=mesh, data_scheme=data)
   synthetic_data_ert = ert_fwd.forward(resistivity_model=resistivity_model)

Time-Lapse Inversion (ERT)
--------------------------
.. code-block:: python

   from PyHydroGeophysX.inversion import TimeLapseERTInversion

   # inversion = TimeLapseERTInversion(
   #     data_files=ert_files,
   #     measurement_times=times,
   #     lambda_val=50.0,
   #     alpha=10.0,
   #     inversion_type="L2"
   # )
   # result = inversion.run()

More Examples and Documentation
------------------------------
- See the :ref:`api` for detailed API documentation.
- Explore the :ref:`auto_examples` for more workflows.
- Example scripts are in the `examples/` directory on [GitHub](https://github.com/geohang/PyHydroGeophysX).

For troubleshooting, updates, and community support, please visit our [GitHub Issues](https://github.com/geohang/PyHydroGeophysX/issues).

----

*PyHydroGeophysX — Bridging the gap between hydrological models and geophysical monitoring.*

