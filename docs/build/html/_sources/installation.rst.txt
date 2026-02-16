Installation
============

PyHydroGeophysX requires Python 3.8 or higher.

Install from PyPI
-----------------

.. code-block:: bash

   pip install pyhydrogeophysx

Install with geophysics engines (recommended)
---------------------------------------------

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/geohang/PyHydroGeophysX.git
   cd PyHydroGeophysX
   pip install -e .

Core Dependencies
-----------------

.. code-block:: bash

   pip install numpy scipy matplotlib tqdm

Optional Dependencies
---------------------

- PyGIMLi for ERT/SRT forward and inversion
- SimPEG for TDEM/FDEM workflows
- RESIPY for field ERT data processing
- CuPy for GPU acceleration
- joblib for parallel CPU workflows

.. code-block:: bash

   pip install pygimli simpeg resipy joblib
   pip install cupy-cuda11x  # Replace with your CUDA version

Verification
------------

.. code-block:: python

   import PyHydroGeophysX as phg
   print("PyHydroGeophysX version:", phg.__version__)
