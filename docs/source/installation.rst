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

Install the optional ADTLERT ERT backend
----------------------------------------

ADTLERT provides a differentiable 2.5D ERT engine for the existing
``run_ert_manager_inversion`` pipeline. It requires Python 3.11 or newer and
is intentionally separate from the general ``geophysics`` extra because it
also installs PyTorch.

.. code-block:: bash

   pip install "pyhydrogeophysx[adtlert]"

Select it explicitly; the default ERT engine remains unchanged.

.. code-block:: python

   from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

   result = run_ert_manager_inversion(
       "survey.dat",
       "output",
       engine="adtlert",
   )

On Linux, this extra installs the GPU-enabled PyPI Torch distribution and
ADTLERT's CUDA 12 CuPy/cuDSS solver stack. ADTLERT automatically falls back to
SciPy when CUDA is unavailable.

Do not combine this with ``pyhydrogeophysx[gpu]``. That extra currently uses
``cupy-cuda11x``, and two CuPy CUDA variants cannot share one environment.

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
- ADTLERT for differentiable 2.5D ERT inversion
- SimPEG for TDEM/FDEM workflows
- RESIPY for field ERT data processing
- CuPy for GPU acceleration
- joblib for parallel CPU workflows

.. code-block:: bash

   pip install pygimli simpeg resipy joblib
   pip install cupy-cuda11x  # Replace with your CUDA version

Desktop App (Qt Workbench)
--------------------------

Prebuilt Windows and macOS bundles are published on `GitHub Releases
<https://github.com/geohang/PyHydroGeophysX/releases/latest>`_ and need no Python
environment. To run the workbench from a Python install instead:

.. code-block:: bash

   pip install "pyhydrogeophysx[desktop]"
   pyhydrogeophysx-workbench

See :doc:`agents/desktop_workbench` for the full usage guide.

Verification
------------

.. code-block:: python

   import PyHydroGeophysX as phg
   print("PyHydroGeophysX version:", phg.__version__)
