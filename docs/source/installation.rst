Installation
============

PyHydroGeophysX requires Python 3.8 or higher.

Installation video
------------------

Follow the manual installation walkthrough, using the commands below for the
current package. For guided setup, see :doc:`agent_install`.

.. raw:: html

   <div class="setup-video">
     <iframe src="https://www.youtube-nocookie.com/embed/jaqfjqq7SN0"
       title="PyHydroGeophysX manual installation walkthrough"
       loading="lazy" allow="fullscreen; picture-in-picture" allowfullscreen></iframe>
   </div>

`Watch on YouTube <https://www.youtube.com/watch?v=jaqfjqq7SN0>`_.

Install from PyPI
-----------------

.. code-block:: bash

   pip install pyhydrogeophysx

Install with geophysics engines (recommended)
---------------------------------------------

.. code-block:: bash

   pip install "pyhydrogeophysx[geophysics]"

.. _adert-speed-comparison:

ADERT speed comparison
----------------------

Watch the speed comparison, then see the ADTLERT backend setup below.
Runtime depends on the hardware, dataset and solver settings.

.. raw:: html

   <div class="setup-video">
     <iframe src="https://www.youtube-nocookie.com/embed/25FvGltrHpE"
       title="ADERT speed comparison"
       loading="lazy" allow="fullscreen; picture-in-picture" allowfullscreen></iframe>
   </div>

`Watch on YouTube <https://www.youtube.com/watch?v=25FvGltrHpE>`_.

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

The extra installs CuPy CUDA 12 and cuDSS on both Linux and Windows. Both
platforms use CUDA-enabled Torch, CuPy GPU CGLS and the cuDSS GPU forward
solver. The slower SciPy forward solver is intentionally disabled so ADTLERT
is never reported while running an unaccelerated forward path. Linux remains
the recommended, most thoroughly tested and generally fastest platform. When
Torch, CuPy CUDA 12 or cuDSS is unavailable, the Python API can fall back to
the original PyHydro ERT engine. Studio requires a passing GPU check before
starting a GPU mode and offers manual selection of PyHydro CPU on failure.
ADTLERT 0.1 also cannot represent remote electrodes encoded as negative ABMN
indices; those surveys safely use the original engine without changing data.

On Windows, install the CUDA-enabled Torch wheel before the extra, for example:

.. code-block:: powershell

   python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
   python -m pip install "pyhydrogeophysx[adtlert]"

For long monitoring sequences, select ADTLERT together with windowed
time-lapse inversion. The backend reuses one forward operator and Jacobian
cache across overlapping windows. Every timestep must have the
same electrode positions and ABMN ordering; process-level window parallelism
is disabled for ADTLERT to avoid duplicating GPU memory. The default ``cgls``
method selects CuPy CGLS on the CUDA-backed ADTLERT path.

The ``adtlert`` and ``gpu`` extras share ``cupy-cuda12x``. Do not install a
second CuPy package such as ``cupy-cuda11x`` in the same environment.

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
   pip install "cupy-cuda12x[ctk]"

Desktop App (Qt Studio)
--------------------------

Prebuilt Windows and macOS bundles are published on `GitHub Releases
<https://github.com/geohang/PyHydroGeophysX/releases/latest>`_ and need no Python
environment. To run the studio from a Python install instead:

.. code-block:: bash

   pip install "pyhydrogeophysx[desktop]"
   pyhydrogeophysx-studio

See :doc:`agents/desktop_studio` for the full usage guide.

Verification
------------

.. code-block:: python

   import PyHydroGeophysX as phg
   print("PyHydroGeophysX version:", phg.__version__)


Verify GPU time-lapse compatibility
-------------------------------------

Use the same Python interpreter as Studio::

   python -m PyHydroGeophysX.inversion.adtlert_diagnostics --report gpu-check.json

The command checks CUDA runtime availability, a single-survey GPU inversion,
and a two-iteration windowed GPU inversion in separate processes. The normal
line-search configuration remains enabled. Each stage has a 120-second timeout;
use ``--timeout`` to change it. Exit code 0 means all three passed. Inspect the
JSON report for individual results, commands, complete logs, exit codes,
package versions and interpreter/source paths. A small synthetic test does
not establish convergence for every field survey.

Studio runs these checks when ADTLERT is selected and displays single-survey
and time-lapse availability separately. A GPU mode waits for its own successful
check; switch to PyHydro for CPU operation or reselect ADTLERT to rerun checks.
Neither GPU detection nor the Studio UI self-test proves time-lapse compatibility.

Conflicting OpenMP runtimes on Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies may initialize both ``libomp.dll`` and ``libiomp5md.dll``. This is
a software-runtime issue, not specific to Intel CPUs. Keep the diagnostic
report and verify the CPU path separately if the GPU time-lapse check fails.
Do not set ``KMP_DUPLICATE_LIB_OK``, remove DLLs, or disable line search to
turn a failing installation check into a pass. A process-isolation or numerical
compatibility patch requires its own review and validation.

After all required checks pass, save ``python -m pip freeze`` and, when using
Conda, ``conda env export``. Record the GPU/driver and package source commit as
well. Recheck after dependency changes. The repository's ``environment.yml``
is a starting configuration, not a universal validated GPU lockfile.

For editable installs, the source checkout and environment's launcher normally
live in different directories. Verify the actual imported checkout before
updating; do not create another checkout merely because these paths differ::

   python -c "import sys, PyHydroGeophysX; print(sys.executable); print(PyHydroGeophysX.__file__)"
