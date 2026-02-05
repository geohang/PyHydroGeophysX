Installation
============

PyHydroGeophysX requires Python 3.8 or higher.

Install from PyPI
-----------------

.. code-block:: bash

   pip install pyhydrogeophysx

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/geohang/PyHydroGeophysX.git
   cd PyHydroGeophysX
   pip install -e .

Core Dependencies
-----------------

.. code-block:: bash

   pip install numpy scipy matplotlib pygimli joblib tqdm

Optional Dependencies
---------------------

- RESIPY for field ERT data processing
- SimPEG for TDEM workflows
- CuPy for GPU acceleration

.. code-block:: bash

   pip install resipy simpeg
   pip install cupy-cuda11x  # Replace with your CUDA version

Verification
------------

.. code-block:: python

   import PyHydroGeophysX as phg
   print("PyHydroGeophysX version:", phg.__version__)
