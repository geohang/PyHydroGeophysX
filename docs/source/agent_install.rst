Install with an AI agent
========================

A coding agent with terminal access can do the setup for you. Two decisions trip
people up: whether pip or conda owns the environment, and whether the machine can
use the CUDA build. Both package managers are correct in different environments,
and picking the wrong one leaves two builds of VTK or Qt on the path. The GPU
question is worse, because a CPU-only Torch wheel installs without complaint and
the CUDA engine then quietly falls back.

The prompt below checks both, installs the matching build, and runs a small
example before reporting success. Review the commands it proposes before letting
it change anything.

Installation video
------------------

.. raw:: html

   <div class="setup-video">
     <iframe src="https://www.youtube-nocookie.com/embed/W9SpHMyN9Fc"
       title="Installing PyHydroGeophysX with an AI coding agent"
       loading="lazy" allow="fullscreen; picture-in-picture" allowfullscreen></iframe>
   </div>

`Watch on YouTube <https://www.youtube.com/watch?v=W9SpHMyN9Fc>`_ if the embedded
player does not load.

Copy this prompt
----------------

Paste it into Claude Code, Codex or another agent that can run commands on your
machine. It is the same prompt the project README carries, so the two stay in
step.

.. code-block:: text

   Install PyHydroGeophysX from this repository into my current Python environment.
   Match the build to my hardware, then prove it works on a small example before you
   tell me it is done.

   STEP 1 - which package manager owns this environment
   Run `conda list numpy`. If the channel column says `pypi`, use pip for
   everything. If it names a conda channel such as conda-forge, use conda for the
   binary packages. Do not mix the two: an environment created by conda can still be
   pip-managed, so go by that check rather than by how the environment was created.

   STEP 2 - is there a usable CUDA GPU
   Run `nvidia-smi` and `python -c "import sys; print(sys.version_info[:2])"`.
   Report the GPU name and the "CUDA Version" in the nvidia-smi header, which is the
   highest CUDA the driver supports rather than what is installed.

     - No nvidia-smi, no NVIDIA GPU, or Python older than 3.11 -> CPU path.
       Install `geophysics` only, skip every GPU package, and say plainly that you
       chose CPU and why.
     - NVIDIA GPU with driver CUDA 12 or newer, and Python 3.11 or newer -> CUDA path.

   STEP 3 - install
   CPU path:
     pip install -e ".[geophysics]"

   CUDA path, in this order. Torch goes first because the extra does not pull a
   CUDA build of it:
     python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
     python -c "import torch; print(torch.cuda.is_available())"
     pip install -e ".[geophysics,adtlert]"

   That torch check must print True before you continue. On Windows the default
   PyPI torch wheel is CPU-only, so a False there means you skipped the index-url.
   Never install cupy-cuda11x next to cupy-cuda12x; the `adtlert` and `gpu` extras
   both pin cupy-cuda12x. If PyGIMLi will not build under pip, run
   `conda install -c gimli pygimli` first, then repeat the pip line.

   STEP 4 - fetch one small example
   Do not clone examples/data, it is about 180 MB. Download these two files only,
   about 175 KB total:
     curl -L -o line2.dat https://raw.githubusercontent.com/geohang/PyHydroGeophysX/main/examples/data/ERT/Bert/fielddataline2.dat
     curl -L -o e4d.ohm   https://raw.githubusercontent.com/geohang/PyHydroGeophysX/main/examples/data/ERT/E4D/2021-10-08_1400.ohm

   STEP 5 - prove it runs, about 10 seconds on CPU
     python -c "from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion as r; d=r('line2.dat','out_cpu',max_iterations=4); print('engine',d['engine'],'chi2 %.3f'%d['chi2'])"
   Expect `engine pyhydro` and chi2 near 0.3.

   STEP 6 - CUDA path only: prove the GPU engine actually engages
     python -c "from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion as r; d=r('e4d.ohm','out_gpu',max_iterations=4,engine='adtlert'); print('requested',d['engine_requested'],'-> engine',d['engine'])"
   `requested adtlert -> engine adtlert` means the CUDA path is live. If it prints
   `-> engine pyhydro` the GPU engine fell back, so report which of Torch CUDA,
   CuPy CUDA 12 or cuDSS is missing instead of calling the install finished. Use
   e4d.ohm and not line2.dat for this check: line2.dat has remote electrodes with
   negative ABMN indices, which ADTLERT 0.1 cannot represent, so it falls back on
   that file even when the GPU stack is perfectly healthy.

   Rules: show me a dry run and what would change before you modify my environment.
   Do not accept any channel Terms of Service for me; if a package manager asks,
   stop and give me the exact command to run myself.

   Optional, for the desktop studio: also install the `desktop` and `desktop-3d`
   groups, then verify with
     python -c "import PyHydroGeophysX, pygimli, PySide6, pyvista; print('ok')"
     python -m PyHydroGeophysX.qt_apps.launcher --self-test

What it checks
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Step
     - What it settles
   * - Package manager
     - Reads the channel of an installed package rather than assuming, because a
       conda-created environment can still be pip-managed.
   * - GPU
     - Reports the driver's highest supported CUDA, then takes the CPU path when
       there is no NVIDIA GPU or Python is older than 3.11.
   * - Install
     - Installs Torch from the CUDA index before the extras on the GPU path,
       since the extra does not pull a CUDA build of it.
   * - Proof
     - Inverts a 175 KB field dataset and reports the engine that actually ran,
       so a silent fallback to the CPU engine is visible rather than hidden.

Manual setup is in :doc:`installation`, and the application guide is in
:doc:`agents/desktop_studio`. An installation agent is separate from the in-app
AQUAH assistant; see :doc:`agents/quick_start` to configure that.
