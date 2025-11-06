Data Processing Module
======================

The ``data_processing`` module provides tools for loading, quality control, and exporting geophysical field data.

ERT Data Agent
--------------

.. automodule:: PyHydroGeophysX.data_processing.ert_data_agent
   :members:
   :undoc-members:
   :show-inheritance:

Overview
~~~~~~~~

The ``ert_data_agent`` module provides a standardized interface for working with electrical resistivity tomography (ERT) field data. It integrates with RESIPY to support data loading from multiple commercial instruments and provides quality control visualization and export functionality.

**Key Features:**

* Load ERT data from 14+ commercial instruments (E4D, Syscal, ABEM, Sting, ARES, etc.)
* Automatic coordinate reference system handling (local, projected, geographic)
* Quality control visualizations (histograms, pseudosections)
* Export to pyGIMLi/BERT format for inversion
* Support for time-lapse ERT surveys

Data Structures
~~~~~~~~~~~~~~~

LocalRef
^^^^^^^^

.. py:class:: LocalRef

   Named tuple for local coordinate reference system parameters.

   :param origin_x: X-coordinate of profile origin in world coordinates (default: 0.0)
   :type origin_x: float
   :param origin_y: Y-coordinate of profile origin in world coordinates (default: 0.0)
   :type origin_y: float
   :param azimuth_deg: Profile azimuth in degrees clockwise from north (default: 0.0)
   :type azimuth_deg: float

Electrode
^^^^^^^^^

.. py:class:: Electrode

   Dataclass representing a single electrode.

   :param id: Electrode identifier
   :type id: int
   :param x: X-coordinate
   :type x: float
   :param y: Y-coordinate (default: 0.0)
   :type y: float
   :param z: Z-coordinate/elevation (default: 0.0)
   :type z: float

Quadruplet
^^^^^^^^^^

.. py:class:: Quadruplet

   Dataclass representing a 4-electrode measurement configuration.

   :param A: Current injection electrode A
   :type A: int
   :param B: Current injection electrode B
   :type B: int
   :param M: Potential measurement electrode M
   :type M: int
   :param N: Potential measurement electrode N
   :type N: int

Observation
^^^^^^^^^^^

.. py:class:: Observation

   Dataclass representing a single ERT measurement.

   :param quad: 4-electrode configuration
   :type quad: Quadruplet
   :param app_res: Apparent resistivity in Ω·m (optional)
   :type app_res: float | None
   :param dV: Measured potential difference in V (optional)
   :type dV: float | None
   :param I: Injected current in A (optional)
   :type I: float | None
   :param resist: Measured resistance in Ω (optional)
   :type resist: float | None
   :param K: Geometric factor (optional)
   :type K: float | None
   :param err: Measurement error/uncertainty (optional)
   :type err: float | None
   :param valid: Validity flag (optional)
   :type valid: bool | None

ERTDataset
^^^^^^^^^^

.. py:class:: ERTDataset

   Dataclass representing a complete ERT survey dataset.

   :param electrodes: List of electrode positions
   :type electrodes: List[Electrode]
   :param observations: List of measurements
   :type observations: List[Observation]
   :param crs: Coordinate reference system ('local', 'EPSG:XXXX', or 'WGS84')
   :type crs: str
   :param local_ref: Local coordinate reference (optional)
   :type local_ref: LocalRef | None
   :param epsg: EPSG code for projected coordinates (optional)
   :type epsg: int | None
   :param metadata: Additional survey metadata
   :type metadata: Dict[str, Any]

Functions
~~~~~~~~~

load_ert_resipy
^^^^^^^^^^^^^^^

.. py:function:: load_ert_resipy(project_dir: str, data_file: str, instrument: str, crs: str = "local", local_ref: LocalRef | None = None, epsg: int | None = None) -> ERTDataset

   Load ERT field data using RESIPY library with support for multiple instruments.

   :param project_dir: Directory for RESIPY project (working directory)
   :type project_dir: str
   :param data_file: Path to ERT data file (relative or absolute)
   :type data_file: str
   :param instrument: Instrument type (see Supported Instruments below)
   :type instrument: str
   :param crs: Coordinate reference system ('local', 'EPSG:XXXX', or 'WGS84')
   :type crs: str
   :param local_ref: Local coordinate reference parameters (required if crs='local')
   :type local_ref: LocalRef | None
   :param epsg: EPSG code for projected coordinates (required if crs starts with 'EPSG:')
   :type epsg: int | None
   :return: Complete ERT dataset with electrodes, measurements, and metadata
   :rtype: ERTDataset
   :raises ImportError: If RESIPY is not installed
   :raises FileNotFoundError: If data_file does not exist
   :raises ValueError: If instrument type is not supported or CRS parameters are invalid

   **Supported Instruments:**
   
   * **Protocol DC** - Iris Instruments Protocol DC systems
   * **Syscal** - Iris Instruments Syscal systems
   * **Protocol IP** - Iris Instruments Protocol IP systems
   * **ResInv** - ResInv format files
   * **PRIME/RESIMGR** - Prime/Resimgr format
   * **Sting** - AGI Sting systems
   * **ABEM-Lund** - ABEM/Lund systems
   * **Lippmann** - Lippmann systems
   * **ARES** - GF Instruments ARES systems
   * **BERT** - pyGIMLi/BERT format files
   * **E4D** - E4D format (common in watershed monitoring)
   * **DAS-1** - DAS-1 systems
   * **Electra** - Electra systems
   * **Custom** - Custom data formats
   * **Merged** - Merged datasets

   **Example:**

   .. code-block:: python

      from PyHydroGeophysX.data_processing.ert_data_agent import (
          load_ert_resipy, LocalRef
      )

      # Load E4D data in local coordinates
      ert = load_ert_resipy(
          project_dir="data/ERT/E4D",
          data_file="data/ERT/E4D/2021-10-08_1400.ohm",
          instrument="E4D",
          crs="local",
          local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
      )

      # Load Syscal data in UTM coordinates
      ert = load_ert_resipy(
          project_dir="data/ERT/Syscal",
          data_file="data/ERT/Syscal/survey.txt",
          instrument="Syscal",
          crs="EPSG:32615",  # UTM Zone 15N
          epsg=32615
      )

   **Notes:**
   
   * Function handles Windows/OneDrive permission issues automatically
   * Supports Unix-style paths on Windows
   * Flexible column name detection (app/rhoa/Rho, resError/magErr)
   * Automatically converts between resistance and apparent resistivity

qc_and_visualize
^^^^^^^^^^^^^^^^

.. py:function:: qc_and_visualize(ert: ERTDataset, outdir: str = "results") -> Dict[str, str]

   Generate quality control plots and summary statistics for ERT dataset.

   :param ert: ERT dataset from load_ert_resipy
   :type ert: ERTDataset
   :param outdir: Output directory for plots and reports
   :type outdir: str
   :return: Dictionary mapping artifact types to file paths
   :rtype: Dict[str, str]

   **Generated Artifacts:**
   
   * ``rhoa_hist.png``: Histogram of log10 apparent resistivity values
   * ``pseudosection.png``: Pseudosection plot (if supported by instrument)
   * ``data_summary.json``: Statistical summary (count, mean, std, min, max, percentiles)

   **Example:**

   .. code-block:: python

      from PyHydroGeophysX.data_processing.ert_data_agent import (
          load_ert_resipy, qc_and_visualize
      )

      ert = load_ert_resipy(
          project_dir="data/ERT/E4D",
          data_file="data/ERT/E4D/2021-10-08_1400.ohm",
          instrument="E4D"
      )

      artifacts = qc_and_visualize(ert, outdir="results/qc")
      print(f"Histogram: {artifacts['histogram']}")
      print(f"Summary: {artifacts['summary']}")

export_for_inversion
^^^^^^^^^^^^^^^^^^^^

.. py:function:: export_for_inversion(ert: ERTDataset, outdir: str = "results", fmt: str = "pgimli", filename: str = "bert_data.dat") -> str

   Export ERT dataset to format suitable for inversion codes.

   :param ert: ERT dataset from load_ert_resipy
   :type ert: ERTDataset
   :param outdir: Output directory
   :type outdir: str
   :param fmt: Export format ('pgimli' or 'bert')
   :type fmt: str
   :param filename: Output filename (default: 'bert_data.dat')
   :type filename: str
   :return: Path to exported file
   :rtype: str

   **Supported Formats:**
   
   * **pgimli/bert**: Unified data format for pyGIMLi/BERT inversion codes

   **File Structure (pyGIMLi/BERT):**
   
   .. code-block:: text

      112                                    # Number of electrodes
      # x y z                                # Electrode coordinate header
      0.0    0.0    3213.46                  # Electrode 1 coordinates
      3.0    0.0    3211.65                  # Electrode 2 coordinates
      ...
      237.0  0.0    3134.49                  # Electrode 112 coordinates
      3647                                   # Number of measurements
      # a b m n err i ip iperr k r rhoa u valid
      1  2  3  4  0.05  0.1  0  0  1.23  45.6  56.1  1  1
      ...

   **Columns in measurement data:**
   
   * **a**: Current injection electrode A (1-indexed)
   * **b**: Current injection electrode B (1-indexed)
   * **m**: Potential measurement electrode M (1-indexed)
   * **n**: Potential measurement electrode N (1-indexed)
   * **err**: Relative error (default: 0.05 = 5%)
   * **i**: Injected current in A
   * **ip**: Induced polarization (0 for DC-only)
   * **iperr**: IP error (0 for DC-only)
   * **k**: Geometric factor
   * **r**: Measured resistance in Ω
   * **rhoa**: Apparent resistivity in Ω·m
   * **u**: Voltage/potential difference in V
   * **valid**: Validity flag (1=valid, 0=invalid)

   **Example:**

   .. code-block:: python

      from PyHydroGeophysX.data_processing.ert_data_agent import (
          load_ert_resipy, export_for_inversion
      )

      ert = load_ert_resipy(
          project_dir="data/ERT/E4D",
          data_file="data/ERT/E4D/2021-10-08_1400.ohm",
          instrument="E4D"
      )

      # Export to pyGIMLi format
      bert_path = export_for_inversion(
          ert,
          outdir="results/inversion",
          fmt="pgimli",
          filename="survey_2021-10-08.dat"
      )
      print(f"Exported to: {bert_path}")

Workflow Example
~~~~~~~~~~~~~~~~

Complete workflow from field data to inversion-ready format:

.. code-block:: python

   from PyHydroGeophysX.data_processing.ert_data_agent import (
       load_ert_resipy, qc_and_visualize, export_for_inversion, LocalRef
   )

   # 1. Load field data
   ert = load_ert_resipy(
       project_dir="data/ERT/E4D",
       data_file="data/ERT/E4D/2021-10-08_1400.ohm",
       instrument="E4D",
       crs="local",
       local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
   )

   # 2. Quality control
   artifacts = qc_and_visualize(ert, outdir="results/qc")
   print(f"Generated QC plots: {artifacts}")

   # 3. Export for inversion
   bert_path = export_for_inversion(
       ert,
       outdir="results/inversion",
       fmt="pgimli"
   )
   print(f"Ready for inversion: {bert_path}")

   # 4. Inspect dataset
   print(f"Survey has {len(ert.electrodes)} electrodes")
   print(f"Survey has {len(ert.observations)} measurements")
   print(f"CRS: {ert.crs}")

Time-Lapse Surveys
~~~~~~~~~~~~~~~~~~

For time-lapse monitoring, process each timestep separately:

.. code-block:: python

   from pathlib import Path
   from datetime import datetime

   # Time-lapse data files
   data_files = [
       "2021-10-08_1400.ohm",
       "2021-10-09_1400.ohm",
       "2021-10-10_1400.ohm",
   ]

   # Process all timesteps
   bert_files = []
   for data_file in data_files:
       # Extract timestamp from filename
       timestamp = datetime.strptime(
           Path(data_file).stem, "%Y-%m-%d_%H%M"
       )
       
       # Load and process
       ert = load_ert_resipy(
           project_dir="data/ERT/E4D",
           data_file=f"data/ERT/E4D/{data_file}",
           instrument="E4D",
           crs="local",
           local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0)
       )
       
       # Export with timestamp
       bert_path = export_for_inversion(
           ert,
           outdir="results/time_lapse",
           fmt="pgimli",
           filename=f"survey_{timestamp.strftime('%Y%m%d_%H%M')}.dat"
       )
       bert_files.append(bert_path)

   print(f"Processed {len(bert_files)} time-lapse surveys")

See Also
~~~~~~~~

* :doc:`../quickstart`: Getting started guide
* :doc:`inversion`: ERT inversion module
* :doc:`../auto_examples/Ex_ERT_data_process`: Complete example notebook

Acknowledgments
~~~~~~~~~~~~~~~

The ERT data processing module is built on `RESIPY <https://gitlab.com/hkex/resipy>`_, an intuitive open-source software for complex geoelectrical inversion/modeling developed by Guillaume Blanchy, Jimmy Boyd, and contributors.

**Citation:**

Blanchy, G., Saneiyan, S., Boyd, J., McLachlan, P., & Binley, A. (2020). ResIPy, an intuitive open source software for complex geoelectrical inversion/modeling. *Computers & Geosciences*, 137, 104423. https://doi.org/10.1016/j.cageo.2020.104423
