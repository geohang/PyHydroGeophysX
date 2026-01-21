Quickstart
==========

Welcome to **PyHydroGeophysX**—a Python package for integrating hydrological model outputs with geophysical forward modeling and inversion, specializing in ERT and SRT for watershed monitoring.

Key Features
------------
- **ERT data processing**: Load, QC, and export field ERT data with RESIPY integration
- **Multi-Agent AI System**: Automated cross-modal geophysics workflows with LLM support **NEW**
- **3D ERT Modeling**: Complete 3D mesh creation, forward modeling, and PyVista visualization **NEW**
- **TDEM Forward & Inversion**: Time-Domain Electromagnetic modeling using SimPEG **NEW**
- Hydrological model integration (MODFLOW, ParFlow)
- Advanced petrophysical relationships (Archie, Waxman-Smits, DEM, Hertz-Mindlin)
- Forward modeling and time-lapse inversion for ERT & SRT
- Structure-constrained inversion and uncertainty quantification
- Optional GPU acceleration and parallel processing

Installation
------------
PyHydroGeophysX requires **Python 3.8 or higher**.

**From PyPI** (available now!):
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

For ERT data processing (field data):
.. code-block:: bash

   pip install resipy

ERT Field Data Processing
--------------------------
Load, quality control, and export field ERT data from commercial instruments:

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
   )

   # Load ERT field data (supports E4D, Syscal, ABEM, Sting, ARES, and more)
   ert = load_ert_resipy(
       project_dir="data/ERT/E4D",
       data_file="data/ERT/E4D/2021-10-08_1400.ohm",
       instrument="E4D",
       crs="local",
       local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
   )

   # Generate QC plots (histogram, pseudosection, summary stats)
   artifacts = qc_and_visualize(ert, outdir="results/qc")

   # Export to pyGIMLi/BERT format for inversion
   bert_path = export_for_inversion(ert, outdir="results/inversion", fmt="pgimli")

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

3D ERT Forward Modeling (NEW)
-----------------------------
Create complete 3D meshes with electrode arrays and forward model with MODFLOW integration:

.. code-block:: python

   from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator
   from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent
   import pygimli as pg

   # Define domain and electrode array
   domain_x, domain_y, domain_z = 20.0, 20.0, 14.0  # meters
   
   # Create mesh with electrodes
   mesh_creator = Mesh3DCreator(domain_x, domain_y, domain_z, 
                                max_element_size=2.0, quality=1.2)
   electrodes = mesh_creator.create_electrode_grid(
       nx=5, ny=5, spacing=4.0, z_offset=0.0
   )
   mesh = mesh_creator.create_3d_mesh_with_topography(
       electrodes=electrodes, topography=None
   )

   # Load MODFLOW data and interpolate to mesh
   wc_processor = MODFLOWWaterContent(model_dir, idomain, cell_size=1.0)
   water_content = wc_processor.load_timestep(0)

   # Convert to resistivity and run 3D forward modeling
   from PyHydroGeophysX.petrophysics import water_content_to_resistivity
   resistivity = water_content_to_resistivity(water_content, rhos=100, n=2.2, porosity=0.3)

TDEM Forward Modeling (NEW)
---------------------------
Time-Domain Electromagnetic forward modeling using SimPEG:

.. code-block:: python

   from PyHydroGeophysX.forward.tdem_forward import TDEMForwardModeling, TDEMSurveyConfig
   import numpy as np

   # Define survey configuration
   config = TDEMSurveyConfig(
       source_type='VMD',              # Vertical magnetic dipole
       source_location=[0, 0, 1],      # Source at 1m height
       source_radius=5.0,              # 5m loop radius
       receiver_location=[0, 0, 1],    # Co-located receiver
       receiver_type='dBdt',           # Measure dB/dt
       times=np.logspace(-5, -2, 31)   # Time gates
   )

   # Define 1D layered Earth model
   layer_thicknesses = [0.5, 1.0, 2.0, 5.0]  # meters
   conductivities = [0.01, 0.1, 0.05, 0.02]  # S/m

   # Create forward model and compute response
   tdem_fwd = TDEMForwardModeling(config)
   response = tdem_fwd.forward(layer_thicknesses, conductivities)

More Examples and Documentation
------------------------------
- See the :ref:`api` for detailed API documentation.
- Explore the :ref:`auto_examples` for more workflows.
- Example scripts are in the `examples/` directory on [GitHub](https://github.com/geohang/PyHydroGeophysX).

For troubleshooting, updates, and community support, please visit our [GitHub Issues](https://github.com/geohang/PyHydroGeophysX/issues).

----

*PyHydroGeophysX — Bridging the gap between hydrological models and geophysical monitoring.*

