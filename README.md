[![tests](https://github.com/geohang/PyHydroGeophysX/actions/workflows/tests.yml/badge.svg)](https://github.com/geohang/PyHydroGeophysX/actions/workflows/tests.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17025139.svg)](https://doi.org/10.5281/zenodo.17025139)

<div align="center">
  <img src="logo.png" alt="PyHydroGeophysX Logo" width="400">
</div>

# PyHydroGeophysX

A Python package for integrating hydrological model outputs (MODFLOW, ParFlow) with geophysical forward modeling and inversion — ERT, SRT, TDEM, FDEM, MT — for watershed monitoring and critical zone science. Includes a multi-agent AI system for automated geophysical workflows.

<div align="center">
  <img src="frame.png" alt="HydroGeophysX Framework" width="600">
</div>

**Links:** [Documentation](https://geohang.github.io/PyHydroGeophysX/) · [Examples gallery](https://geohang.github.io/PyHydroGeophysX/auto_examples/index.html) · [Live demo app](https://pyhydrogeophysx.streamlit.app/) · [Issues](https://github.com/geohang/PyHydroGeophysX/issues)

---

## Features

- **Hydrological model integration** — load MODFLOW, ParFlow, ATS and PFLOTRAN outputs
- **ERT data processing** — field data QC, export, and RESIPY integration
- **Forward modeling** — 2D/3D ERT, SRT, TDEM, FDEM synthetic data generation
- **E4D-style 3D meshes** — borehole and surface ERT meshes built the way [E4D](https://e4d-userguide.pnnl.gov) builds them (control points, a fine zone inside a far-reaching outer zone, Triangle surface + TetGen volume); E4D `.cfg` configurations read and written, E4D meshes imported for inversion
- **E4D as an inversion engine** — PNNL's parallel 3D ERT code run from the same pipeline and Studio page as the other engines, for single surveys and time-lapse series (E4D's own time-lapse mode); a profile is inverted in 3D around the line and shown as its section. E4D is an external program: see [where it runs](#e4d-an-external-3d-engine)
- **R2 and R3t as inversion engines** — Andrew Binley's 2D and 3D codes, the ones ResIPy runs, for single surveys and time-lapse series (their difference inversion), holding fixed zones fixed; native on Windows, through Wine on Linux and macOS: see [where they run](#r2-and-r3t-binleys-ert-codes)
- **Inversion** — single-time, time-lapse, windowed, structure-constrained, joint ERT+SRT, TDEM, FDEM
- **Magnetotellurics (MT/AMT)** — time series read from Phoenix MTU-5C/5P/8A and legacy MTU-5A, Metronix ATS, Zonge Z3D and LEMI-424 recordings with their calibrations; robust remote-reference processing in the manner of EMTF (Egbert 1997); EDI, EMTF XML, Z- and J-files read and written; phase tensor, skew and strike; Occam 1D with the static shift as a parameter, fixed by a joint TEM sounding (Meju 1996), and a water-content profile; 2D TE/TM profile inversion on SimPEG. Only NumPy/SciPy for the processing and 1D, SimPEG for 2D
- **Prior knowledge in ERT** — a-priori resistivity zones drawn on the inversion mesh (starting and reference values, optionally held fixed), a mesh rebuilt so its cell edges follow the zone outlines, and the smoothness dropped across those outlines so the model may jump there, with every engine; in 3D, zones as boxes (layers, tanks, plumes) that every 3D mesh engine — structured grid, prism, Gmsh, E4D — can mesh along and make regions of their own, and that the 3D forward model takes
- **Monitoring** — acquisition times read from the survey files, temperature correction to a reference temperature, sections clipped to what the data resolve
- **Petrophysics** — water content ↔ resistivity (Waxman-Smits/Archie), seismic velocity (Hertz-Mindlin, DEM)
- **Uncertainty quantification** — Monte Carlo for petrophysical parameter uncertainty
- **Multi-agent AI system** — automated workflows via GPT, Gemini, or Claude APIs
- **GPU acceleration** — optional CuPy/CUDA support for large-scale inversions

---

## Installation

### Recommended (conda — handles binary deps for PyGIMLi)

```bash
conda env create -f environment.yml
conda activate pyhydrogeophysx
```

### From PyPI

```bash
# Core only (petrophysics, model I/O, solvers)
pip install pyhydrogeophysx

# With geophysics engines (ERT/SRT/TDEM/FDEM inversion and forward modeling)
pip install "pyhydrogeophysx[geophysics]"

# With the optional ADTLERT differentiable 2.5D ERT backend (Python 3.11+)
pip install "pyhydrogeophysx[adtlert]"

# With AI agent support
pip install "pyhydrogeophysx[geophysics,agents]"

# With web app
pip install "pyhydrogeophysx[geophysics,webapp]"

# Everything
pip install "pyhydrogeophysx[all]"

# Optional: TetGen, for E4D-style 3D meshes (TetGen is AGPL-licensed, so it is
# not installed with the package; without it, Gmsh stands in)
pip install tetgen
```

> **What the pip package includes:** the library and the Qt desktop studio
> (`pyhydrogeophysx-studio`, with the `desktop` extra). The example scripts and
> notebooks, their data in `examples/data/`, and the Streamlit web apps in
> `examples/` come with the source repository, not the pip package. Clone the
> repository ([From Source](#from-source)) to run them, or use the
> [hosted web app](https://pyhydrogeophysx.streamlit.app/). The launchers
> `pyhydrogeophysx-gui` and `python -m PyHydroGeophysX.gui_mesh3d` take an app's
> path as their first argument; after a pip install they print where the apps are.

> **Note on PyGIMLi:** PyGIMLi links against C++ libraries. If `pip install` fails, install it first via conda:
> ```bash
> conda install -c gimli pygimli
> pip install "pyhydrogeophysx[agents]"  # then add other extras
> ```

The ADTLERT backend plugs into the existing single-time ERT pipeline without
changing its default engine:

```python
from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

result = run_ert_manager_inversion(
    "survey.dat",
    "output",
    engine="adtlert",
)
```

The `adtlert` extra installs CuPy CUDA 12 and cuDSS on both Linux and Windows.
Both platforms use CUDA-enabled Torch, CuPy GPU CGLS and the cuDSS GPU forward
solver. The slower SciPy forward solver is intentionally disabled so ADTLERT
is never reported while running an unaccelerated forward path. Linux remains
the recommended, most thoroughly tested and generally fastest platform. When
Torch, CuPy CUDA 12 or cuDSS is unavailable, the Python API can fall back to
the original PyHydro ERT engine. Studio requires a passing GPU check before
starting a GPU mode and offers manual selection of PyHydro CPU on failure.
ADTLERT 0.1 also cannot represent remote electrodes encoded as negative ABMN
indices; those surveys safely use the original engine without changing data.

On Windows, install the CUDA-enabled Torch wheel before the extra, for example:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install "pyhydrogeophysx[adtlert]"
```

ADTLERT can also run the windowed time-lapse workflow on one shared GPU forward
operator:

```python
from PyHydroGeophysX.inversion.time_lapse import run_timelapse_ert

result = run_timelapse_ert(
    ["survey_0.dat", "survey_1.dat", "survey_2.dat"],
    [0.0, 1.0, 2.0],
    {"engine": "adtlert", "windowed": True, "window_size": 3},
    "output",
)
```

ADTLERT timesteps must share electrode positions and numbering. Measurements
are aligned by their ABMN union; missing rows receive 100% relative error and
an apparent-resistivity placeholder from the available timesteps. This reduces
their weight; it does not exclude them from fitting. Native PyHydro time-lapse
keeps each survey's own measurement count and ordering. ADTLERT
processes overlapping windows sequentially on the GPU so solver state and
Jacobian caches are reused without duplicating GPU memory across processes.
The default `cgls` method selects CuPy CGLS on the CUDA-backed ADTLERT path.

The `adtlert` and `gpu` extras now share `cupy-cuda12x`; do not install a second
CuPy package such as `cupy-cuda11x` in the same environment.

### E4D, an external 3D engine

[E4D](https://github.com/pnnl/E4D) (PNNL; BSD-style licence) is a Fortran/MPI
program built from source with gfortran, PETSc and MPI. It is **not installed
with PyHydroGeophysX**: the `e4d` engine writes the files E4D reads, runs E4D,
and reads its results back, so QC, the error model, the geometric-factor check,
outlier rejection, the λ search and every viewer and export work as with the
other engines.

| Where | Can E4D run? | How PyHydroGeophysX reaches it |
|---|---|---|
| Linux workstation or cluster node | Yes — E4D's own platform | `e4d` and `mpirun` on PATH, `$PYHYDRO_E4D` / `$PYHYDRO_MPIRUN`, or paths in the settings |
| Windows | Only inside **WSL 2** (E4D has no Windows build) | Build E4D in the WSL Linux; choose the `wsl` launcher. PyHydroGeophysX stays on Windows |
| macOS | When built from source (not in E4D's build instructions; untested here) | As on Linux |
| Cluster with a scheduler, or no E4D at hand | Elsewhere | `files` launcher: the complete E4D run folder is written; run `mpirun -np N e4d` in it, then read it back with `read_e4d_run` / `read_e4d_time_lapse` |

E4D needs at least two MPI processes (one master, one or more workers) and no
more workers than electrodes. `$PYHYDRO_E4D_COMMAND` replaces `mpirun -np N e4d`
with any command line (`{processes}` is filled in), for `srun` or a container.
To see what this machine offers:

```bash
python -m PyHydroGeophysX.inversion.e4d
```

```python
from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

result = run_ert_manager_inversion(
    "survey.dat", "output", engine="e4d", lam=20,
    e4d={"launcher": "auto", "processes": 8},   # or "local", "wsl", "files"
)
print(result["e4d"]["run_dir"])   # E4D's e4d.log, sigma.N and resistivity_3d.vtk
```

E4D models in 3D. A 3D survey on an imported tetrahedral mesh is inverted on
that mesh. A 2D profile is inverted on a 3D mesh built around the line the way
E4D users build one (a fine zone under the electrodes, reaching half the
inverted depth to either side, inside an outer zone reaching four electrode
spreads beyond it; the ground along the line carried across it), and the section along the line is
read back onto the profile's own mesh. Each run is E4D's mode ERT3 at a fixed
λ (E4D's β): nearest-neighbour smoothness, update option 3 so it stops once the
objective stops falling, and stopped after the iteration limit, since E4D has
none of its own. E4D's χ² is that of the transfer resistances, with a standard
deviation of `err·|R|`.

A time-lapse series runs as E4D's own time-lapse inversion (ERT4) through
`run_timelapse_ert(..., {"engine": "e4d", ...})` or the Time-lapse panel: the
first survey is the baseline, each later one starts from the solution before
it, and the smoothness acts on the change from it. The measurements common to
every survey are inverted, as E4D requires the same ones in each. The temporal
weight α, the interval weighting and auto-λ belong to the in-house joint
inversion and are not used; the log says so.

### R2 and R3t, Binley's ERT codes

[R2 and R3t](http://www.es.lancs.ac.uk/people/amb/Freeware/R2/R2.htm) are
Andrew Binley's finite-element inversion programs (Lancaster University), R2
for 2D profiles and R3t for 3D surveys — the programs ResIPy runs. They are
**not installed with PyHydroGeophysX**: the `r2` and `r3t` engines write the
files they read, run them, and read their results back through the same
pipeline as the other engines. Both are provided for non-commercial use;
commercial use needs the author's permission.

| Where | Can R2/R3t run? | How PyHydroGeophysX reaches it |
|---|---|---|
| Windows | Yes, natively (`R2.exe`, `R3t.exe`) | Found in an installed ResIPy (`pip install resipy`) without importing it; or named in the settings, `$PYHYDRO_R2` / `$PYHYDRO_R3T`, or on PATH |
| Linux | Through Wine, as their manuals and ResIPy run them | `wine` (or `wine64`) on PATH, e.g. `sudo apt install wine`; the same `.exe` files |
| macOS | Through Wine (untested here) | Homebrew's `wine-stable` |
| Nowhere at hand | Elsewhere | `files` launcher: the complete run folder is written; run `R2.exe` / `R3t.exe` in it, then read it back with `read_r2_run` |

```bash
python -m PyHydroGeophysX.inversion.r2 --program r3t
```

```python
result = run_ert_manager_inversion(
    "survey.dat", "output", engine="r2",          # or "r3t"
    r2={"launcher": "auto"},                       # or {"executable": r"C:\R2\R2.exe"}
)
print(result["r2"]["run_dir"], result["r2"]["alpha"])   # R2.out, f001_res.dat; its alpha
```

R2 inverts a profile on the profile's own mesh, every element a parameter.
R3t inverts a 3D survey on its tetrahedral mesh, and a profile on a 3D mesh
built around the line as for E4D, read back as the section along it — far
slower than R2 (the 72-electrode BERT example took about 16 minutes for ten
iterations on 44 000 elements, R2 seconds) and with the ground off the line held
only by the smoothness, so R2 is the engine for profiles. R3t leaves out of its
fit any reading it cannot use (no signal over uniform ground, with M and N
symmetric about A and B, or a polarity its model does not reproduce); the log
says how many, χ² is over the readings it fitted, and a forward run gives the
others' predictions. Both are
Occam inversions that search their smoothing weight α at every iteration
themselves, so **λ and auto-λ do not apply**: the run reports the α it settled
on instead. They stop at an RMS misfit of `sqrt(target χ²)`, when the misfit
stops improving, or at the iteration limit. The data are transfer resistances
with their own standard deviations `err·|R|`, inverted as logarithms, so their
RMS² is the same χ² the other engines report; their own reweighting is off
because the pipeline rejects outliers itself. A fixed a-priori zone is held at
its value (R2's `param = 0`), the zone outlines become R2/R3t zones that the
smoothness does not cross, and a remote electrode is placed on the mesh node
farthest from the array.

A time-lapse series is their difference inversion (LaBrecque and Yang, 2001):
the first survey is inverted, then every later one from that model with the
smoothness on the change from it, using the readings it shares with the first.
R2 forms the difference data `d − d0 + f(m0)` itself; for R3t they are written
from the baseline's predicted data, as its manual describes.

### From Source

```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
pip install -e ".[geophysics]"
```

### Verify Windows/Linux GPU time-lapse before using field data

Run with the same interpreter used to launch Studio:

```bash
python -m PyHydroGeophysX.inversion.adtlert_diagnostics --report gpu-check.json
```

This runs CUDA-runtime, single-survey, and two-iteration windowed time-lapse
checks in separate processes, preserving the default line search. Each stage
has a 120-second timeout (override with `--timeout`). Exit code 0 means all
three checks passed. The JSON report includes versions, source/interpreter
paths, exact commands, full logs and per-stage results. It is a smoke test,
not a guarantee of convergence for every field dataset.

Studio performs these checks when ADTLERT is selected and reports single and
time-lapse readiness separately. A mode cannot start until its check passes.
Select PyHydro to use the CPU path; switch engines and reselect ADTLERT to retry.
GPU visibility or a passing Studio self-test alone does not verify time-lapse.

On Windows, different dependencies can load conflicting OpenMP runtimes
(`libomp.dll` / `libiomp5md.dll`), including on AMD CPUs. Keep the error report
and use a separately verified CPU setup while investigating. Do not force
`KMP_DUPLICATE_LIB_OK`, remove DLLs, or disable line search to claim success.
Keep an environment export from a passing installation and rerun the checks
after package updates. `environment.yml` is a starting environment, not a
validated CUDA lockfile for every driver and computer.

### With a coding agent (Claude Code, Codex)

The install has two decisions that trip people up: whether to reach for pip or
conda, and whether this machine can use the CUDA build. Both pip and conda are
correct in different environments, and picking the wrong one leaves two builds of
VTK or Qt on the path. The GPU question is worse, because a CPU-only Torch wheel
installs without complaint and the CUDA engine then quietly falls back.

Paste the block below into Claude Code or Codex. It checks the machine, installs
the matching build, and runs a small example before reporting success.

```text
Help me install PyHydroGeophysX and verify it on this computer.
Repository: https://github.com/geohang/PyHydroGeophysX
Official instructions: https://geohang.github.io/PyHydroGeophysX/installation.html
Desktop guide: https://geohang.github.io/PyHydroGeophysX/agents/desktop_studio.html

1. Inspect my OS, Python executable/version, available environment managers,
   and existing NumPy, Qt, VTK and geophysical packages. Ask whether I want
   Desktop Studio, Python workflows, or both, and which methods I need.
   Do not assume Conda exists or infer all package origins from NumPy alone.

2. Read the current official installation instructions and package metadata.
   Prefer a dedicated environment with a supported Python version. If I ask
   to reuse an environment, inspect its binary dependencies first. Use its
   existing package manager for those dependencies; avoid duplicate pip/Conda
   Qt and VTK installations. Show the plan before changing the environment.

3. Choose a released package or source checkout explicitly. For a release,
   use python -m pip install with the appropriate extras from the official
   instructions. For source, locate an existing checkout or download the
   official repository into a new directory, then enter the directory that
   contains pyproject.toml before using an editable install. Do not overwrite
   an existing checkout. Every pip command must use the chosen interpreter.
   For updates, locate the checkout currently imported by that interpreter;
   do not create another clone or worktree unless I request it. Record:
     python -c "import sys, PyHydroGeophysX; print(sys.executable); print(PyHydroGeophysX.__file__)"
   An editable source folder and an environment's Scripts launcher are normally
   in different locations; that alone does not indicate a broken installation.

4. Start with the CPU setup unless GPU acceleration is needed and supported.
   For CUDA, inspect the NVIDIA GPU, driver, OS, Python and dependency support.
   nvidia-smi reports driver capability, not the installed CUDA toolkit.
   Select a compatible PyTorch/CuPy/cuDSS combination using current official
   instructions; do not assume any CUDA 12 driver supports every CUDA wheel.
   Never install competing CuPy variants. If torch.cuda.is_available() is
   False, inspect the wheel, driver and device availability before diagnosing
   the cause. Report an unsupported GPU configuration and offer the CPU path.

5. Verify imports, the installed version and required engines. Download only
   the small example files needed for a check, using a downloader suitable
   for this OS, into a new test directory. Run a short example and report
   finite results, any warnings and the engine that actually executed.
   If testing ADTLERT, use a compatible dataset without remote electrodes;
   report a fallback rather than claiming that GPU execution succeeded.
   For ADTLERT, also run the three isolated checks (runtime, single, time-lapse):
     python -m PyHydroGeophysX.inversion.adtlert_diagnostics --report gpu-check.json
   Wait for the command to finish and inspect every stage's exit code and JSON
   result. CUDA availability, a passing single-survey test, or Studio self-test
   alone does not verify GPU time-lapse. A traceback line alone does not prove
   that a process crashed; preserve the complete log and final exit status.
   If this command is unavailable in an older release, report time-lapse as
   unverified and use the matching release's documented test, or offer an update.
   On failure, retain gpu-check.json and report each capability separately.
   OpenMP conflicts can affect Intel or AMD CPUs. Do not set KMP_DUPLICATE_LIB_OK,
   delete/rename DLLs, disable line search, or patch NumPy to make a check pass.
   Do not modify source algorithms as part of installation unless I explicitly
   request a code fix. Offer the CPU engine and verify it independently;
   do not reinstall the same package combination repeatedly.
   After successful verification, save python -m pip freeze and, for Conda,
   conda env export. These record this tested machine; they are not a universal
   GPU compatibility guarantee. Re-run the checks after dependency updates.

6. For Desktop Studio, install the desktop extras and the engines needed for
   my methods, then run:
     python -m PyHydroGeophysX.qt_apps.launcher --self-test
   Launch the application and give me the exact command to reopen it.
   Report the environment name, installation path and any unverified features.

   For AQUAH Auto to report, keep provider/model/API key settings in the right
   assistant panel. The OpenAI default is gpt-5.6-luna with medium reasoning.
   Folder classification sends filenames and short previews to that provider;
   report whether a real API call was verified or only offline tests passed.
   RAG uses local reference text. If I request MCP integration, install the
   optional mcp extra and test the stdio server's list_workflows tool. Leave
   numerical recipe execution disabled unless I request --allow-run.
   Never claim that a generic DEM/XYZ file was used by an inversion merely
   because the classifier recognized it; verify the supported geometry adapter.

Do not delete environments, overwrite projects, or accept third-party Terms
of Service on my behalf. If a step needs my action, explain what to do.
Do not report success while required verification is failing.
```

### Optional extras

| Extra | Packages installed |
|---|---|
| `geophysics` | pygimli, simpeg, pymatsolver, flopy, pftools |
| `hydrology` | h5py (the ATS and PFLOTRAN output readers) |
| `adtlert` | adtlert, pygimli, Torch, CuPy CUDA 12 and cuDSS acceleration on Windows/Linux; Linux recommended (Python 3.11+) |
| `desktop` | PySide6, pyqtgraph, qtawesome, numpy, pandas |
| `desktop-3d` | pyvista, pyvistaqt, vtk (the Mesh 3D and volume viewers) |
| `agents` | openai, google-genai, anthropic |
| `climate` | pandas, requests |
| `webapp` | streamlit, plotly, streamlit-plotly-events, pyarrow |
| `seismic-raw` | obspy |
| `gpu` | cupy-cuda12x with CUDA Toolkit components |
| `docs` | sphinx, sphinx-gallery, sphinx_rtd_theme |
| `dev` | pytest, pytest-cov, black, flake8 |
| `all` | all general-purpose groups above; ADTLERT remains opt-in |

`desktop-3d` is separate because `vtk` is a large binary wheel. Without it the
studio runs and exports meshes as usual, and the 3D panels show an install
message instead of a viewer.

---

## First run (light example)

A full clone of `examples/data/` is about 180 MB, which is far more than you need
to confirm the install works. These two files come to about 175 KB together and
cover both the CPU and the CUDA paths.

```bash
curl -L -o line2.dat https://raw.githubusercontent.com/geohang/PyHydroGeophysX/main/examples/data/ERT/Bert/fielddataline2.dat
curl -L -o e4d.ohm   https://raw.githubusercontent.com/geohang/PyHydroGeophysX/main/examples/data/ERT/E4D/2021-10-08_1400.ohm
```

**Does it work?** One 37 KB field line, 936 measurements, roughly 10 seconds on a
laptop CPU:

```python
from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion

result = run_ert_manager_inversion("line2.dat", "out_cpu", max_iterations=4)
print(result["engine"], result["chi2"])      # pyhydro, chi2 near 0.31
```

`out_cpu/` then holds `resistivity_model.npy`, `resistivity_mesh.bms`,
`resistivity_model.vtk`, and the coverage and forward-response arrays.

**Is the GPU actually being used?** Ask for the CUDA engine, then compare what you
requested against what ran:

```python
result = run_ert_manager_inversion("e4d.ohm", "out_gpu", max_iterations=4, engine="adtlert")
print(result["engine_requested"], "->", result["engine"])   # adtlert -> adtlert
```

`adtlert -> adtlert` means the CUDA path is live. `adtlert -> pyhydro` means it
fell back, normally because Torch is a CPU-only wheel or because CuPy CUDA 12 or
cuDSS is missing. Run this check on `e4d.ohm` rather than `line2.dat`:
`line2.dat` carries remote electrodes encoded as negative ABMN indices, which
ADTLERT 0.1 cannot represent, so it falls back on that file even when the GPU
stack is healthy.

[Yang et al. (2026)](https://arxiv.org/abs/2608.14661) measured the CUDA engine
against pyGIMLi and report an approximately 51-fold speedup under their tested
configuration. Forward responses, gradients, and recovered resistivity models
agreed closely in that comparison. The engine is aimed at larger lines and
time-lapse windows.

---

## Running the apps

**Web app (Streamlit):**

```bash
streamlit run examples/app_geophysics_workflow.py
```

On Windows, users who downloaded the source package can instead double-click
`PyHydroGeophysX\start_webapp.bat` (`PyHydroGeophysX/start_webapp.sh` on macOS
and Linux). The launcher finds a compatible Python or conda
environment, opens the browser automatically, and installs the web-app
dependencies into a local `.venv-webapp` environment when needed.

Or hand the whole thing to Claude Code or Codex:

```text
Set up and run the PyHydroGeophysX Streamlit app from this repository at
http://localhost:8501.

Check `conda list numpy` first: `pypi` in the channel column means install with
pip, a conda channel means install with conda. Install the `webapp` extra, and
`geophysics` as well if pygimli is missing. Show me a dry run before you change
my environment.

Then run:
  streamlit run examples/app_geophysics_workflow.py

Leave it running, tell me the URL, and report any error from the first page load
rather than only that the server started. If port 8501 is busy, use the next
free port and tell me which one.
```

**Desktop studio (Qt):**

### Agent workflow: folders, reasoning, references and tools

- Select a data folder and let AQUAH classify supported files from names and
  short previews. Review/edit roles, then continue; unknown files are not used
  automatically. Individual files can still be added, including geometry and references.
- Default OpenAI model: **GPT-5.6 Luna**, with selectable reasoning effort
  (default **medium**). Explicit environment model settings retain precedence.
- **RAG** retrieves local documentation/reference excerpts and records citations.
- **MCP** exposes the existing workflow registry, folder scan, documentation
  search, recipe validation, and optional recipe execution to other MCP clients.
  Install with `pip install "pyhydrogeophysx[mcp]"`; start with
  `python -m PyHydroGeophysX.mcp_server --root /path/to/project`.
- Real backend events and elapsed time stay visible. Detailed reports include
  filenames/hashes, parameters, observed software calls/versions, references,
  processing warnings and sources of uncertainty.

See the [Agent Workflow guide](https://geohang.github.io/PyHydroGeophysX/agents/agent_workbench.html)
for coordinate formats, current terrain limitations, MCP configuration and audit scope.

Use the right-hand **AQUAH** assistant as the single AI entry point. Choose
**Step-by-step assistance** to review actions, or **Auto to report** to submit a
complete analysis goal. Both use the provider/model and session key configured
in AQUAH. The central **Workflow** page contains data selection, survey ordering,
progress, report preview, output files, and activity; it has no separate AI settings.
If data are missing, AQUAH keeps your goal and asks you to add files and send
**continue**. Completion or failure is reported back in the same conversation.
Follow-up instructions refine the retained goal; **New chat** starts a new goal
while keeping selected data. Set `max_attempts: 1` or request no automatic
optimization to assess the first inversion without retries. Completion with
**Needs review** means the computation finished but quality criteria were not met.
Use **Stop** in Workflow to cancel. Each attempt has its own output folder;
use **File → Save Runs to Project** to retain it in project history.
Auto to report currently supports OpenAI and Claude, reuses the Streamlit unified
workflow, and may incur API charges. Session keys are passed through stdin.

```bash
python -m PyHydroGeophysX.qt_apps.launcher
# or, after (re)installing the package:
pyhydrogeophysx-studio
```

On Windows, `examples\start_studio.bat` opens the studio from a
double-click, the desktop counterpart of `start_webapp.bat`: no activated
environment and no `PATH` entry needed, and it creates a local
`.venv-studio` with the desktop dependencies when it finds none.
`examples/start_studio.sh` is the macOS and Linux version.

Desktop dependencies come from the `desktop` extra (`pip install "pyhydrogeophysx[desktop]"`) or `requirements-desktop.txt`, plus `desktop-3d` for the 3D viewers. Prebuilt Windows/macOS bundles (light and full variants) are on [GitHub Releases](https://github.com/geohang/PyHydroGeophysX/releases/latest); the usage guide is at [Desktop Studio documentation](https://geohang.github.io/PyHydroGeophysX/agents/desktop_studio.html).

---

## Package Structure

```
PyHydroGeophysX/
├── core/               # Interpolation, 2D/3D mesh utilities
├── agents/             # Multi-agent AI orchestration
├── data_processing/    # ERT field data loading, QC, export; mt/ for magnetotellurics
├── model_output/       # MODFLOW, ParFlow, ATS and PFLOTRAN output interfaces
├── petrophysics/       # Resistivity and velocity rock-physics models
├── forward/            # ERT, SRT, TDEM, FDEM forward modeling
├── inversion/          # ERT, SRT, TDEM, FDEM, joint, time-lapse inversion
├── solvers/            # CGLS, LSQR, RRLS linear solvers (optional GPU)
├── Hydro_modular/      # Hydro-to-geophysics conversion utilities
└── Geophy_modular/     # Geophysical data processing tools
```

---

## Examples

All examples have paired `.ipynb` notebooks and `.py` scripts under `examples/`. Data is in `examples/data/`, outputs go to `examples/results/`.

| Example | Description |
|---|---|
| `Ex_model_output` | MODFLOW/ParFlow output loading |
| `Ex_ERT_workflow` | End-to-end ERT forward + inversion |
| `Ex_Time_lapse_measurement` | Synthetic time-lapse ERT schedules |
| `Ex_TL_inversion` | Time-lapse ERT inversion |
| `Ex_TL_inversion_memory` | Memory-optimized versus standard time-lapse ERT inversion |
| `Ex_Structure_resinv` | Structure-constrained resistivity inversion |
| `Ex_structure_TLresinv` | Structure-constrained time-lapse inversion |
| `EX_SRT_forward` | SRT forward modeling |
| `Ex_SRT_inv` | SRT inversion (PyGIMLi + packaged `SRTInversion`) |
| `Ex_joint_inversion` | Joint ERT+SRT inversion (direct and geostatistical cross-gradient coupling) |
| `Ex_3D_ERT_forward` | 3D ERT forward with MODFLOW integration |
| `Ex_TDEM_workflow` | TDEM forward + inversion (SimPEG) |
| `Ex_FDEM_workflow` | FDEM forward + inversion (SimPEG) |
| `Ex_MT_workflow` | Magnetotellurics over the MODFLOW model: time-series processing, static shift fixed by TEM, water content, a 2D profile, and a USMTArray site |
| `Ex_hydro_to_multigeophys` | Hydro → petrophysics → multi-method forward |
| `Ex_MC_Hydro` | Monte Carlo uncertainty quantification |
| `Ex_sensitivity_analysis` | Sensitivity, resolution and depth of investigation of an ERT survey over the MODFLOW model |
| `Ex_ensemble_assimilation` | EnKF and ES-MDA updates of MODFLOW water content with an ERT survey |
| `Ex_posterior_uncertainty` | Posterior ERT uncertainty propagated to water content, checked against MODFLOW |
| `Ex_multi_agent_workflow` | Automated multi-agent ERT+seismic workflow |

Three examples read another example's output: run `EX_SRT_forward` before
`Ex_SRT_inv`, `Ex_Structure_resinv` before `Ex_structure_TLresinv`, and both of
those before `Ex_MC_Hydro`. Run times and memory for the heavy examples are
listed in the [examples gallery](https://geohang.github.io/PyHydroGeophysX/auto_examples/index.html).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Standard fork → feature branch → PR workflow.

---

## Acknowledgments

We gratefully acknowledge **Craig Ulrich** (Lawrence Berkeley National Laboratory)
for valuable feedback and continued support during the development of
PyHydroGeophysX.

---

## Citation

If you use PyHydroGeophysX, please cite:

```bibtex
@article{chen2026pyhydrogeophysx,
  author  = {Chen, Hang and Niu, Qifei and Wu, Yuxin},
  title   = {PyHydroGeophysX: An Extensible Open-Source Platform for Integrating
             Hydrological Models with Geophysical Measurements},
  journal = {SoftwareX},
  year    = {2026},
  note    = {In press},
  url     = {https://github.com/geohang/PyHydroGeophysX}
}
```

```bibtex
@article{chen2026agentworkflow,
  author  = {Chen, Hang},
  title   = {A Generalizable Automated Geophysical Agent Workflow for
             Accessible Subsurface Hydrology Analysis},
  journal = {Big Data and Earth System},
  pages   = {100042},
  year    = {2026}
}
```

Please also cite the underlying libraries you use:

**Differentiable time-lapse ERT (ADTLERT):**
```bibtex
@article{yang2026adtlert,
  author  = {Yang, Pu and Fang, Zhengyang and Liu, Yuxin and Su, Xuan and
             Feng, Deshan and Chen, Hang},
  title   = {An automatic-differentiation framework for time-lapse electrical
             resistivity tomography inversion of hydrologic dynamics},
  journal = {arXiv preprint arXiv:2608.14661},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.14661}
}
```

**ERT data processing (ResIPy):**
```bibtex
@article{blanchy2020resipy,
  title   = {ResIPy, an intuitive open source software for complex geoelectrical inversion/modeling},
  author  = {Blanchy, Guillaume and Saneiyan, Sina and Boyd, Jimmy and McLachlan, Paul and Binley, Andrew},
  journal = {Computers \& Geosciences},
  volume  = {137},
  pages   = {104423},
  year    = {2020},
  doi     = {10.1016/j.cageo.2020.104423}
}
```

**R2 and R3t inversions (the `r2` and `r3t` engines) and their difference inversion for time-lapse series:**
```bibtex
@book{binley2020resistivity,
  author    = {Binley, Andrew and Slater, Lee},
  title     = {Resistivity and Induced Polarization: Theory and Applications to
               the Near-Surface Earth},
  publisher = {Cambridge University Press},
  year      = {2020},
  doi       = {10.1017/9781108685955}
}

@incollection{binley2015tools,
  author    = {Binley, A.},
  title     = {Tools and Techniques: Electrical Methods},
  booktitle = {Treatise on Geophysics},
  edition   = {2},
  publisher = {Elsevier},
  pages     = {233--259},
  year      = {2015},
  doi       = {10.1016/B978-0-444-53802-4.00192-5}
}

@incollection{binley2005dc,
  author    = {Binley, Andrew and Kemna, Andreas},
  title     = {{DC} Resistivity and Induced Polarization Methods},
  booktitle = {Hydrogeophysics},
  series    = {Water Science and Technology Library},
  publisher = {Springer},
  pages     = {129--156},
  year      = {2005},
  doi       = {10.1007/1-4020-3102-5_5}
}

@article{labrecque2001difference,
  author  = {LaBrecque, Douglas J. and Yang, Xianjin},
  title   = {Difference Inversion of {ERT} Data: a Fast Inversion Method for
             {3-D} In Situ Monitoring},
  journal = {Journal of Environmental and Engineering Geophysics},
  volume  = {6},
  number  = {2},
  pages   = {83--89},
  year    = {2001},
  doi     = {10.4133/JEEG6.2.83}
}
```

**E4D inversions (the `e4d` engine) and E4D-style 3D meshes (E4D, TetGen):**
```bibtex
@article{johnson2010e4d,
  author  = {Johnson, Timothy C. and Versteeg, Roelof J. and Ward, Andy and
             Day-Lewis, Frederick D. and Revil, Andr{\'e}},
  title   = {Improved hydrogeophysical characterization and monitoring through
             parallel modeling and inversion of time-domain resistivity and
             induced-polarization data},
  journal = {Geophysics},
  volume  = {75},
  number  = {4},
  pages   = {WA27--WA41},
  year    = {2010},
  doi     = {10.1190/1.3475513}
}

@manual{johnson2020e4duserguide,
  author       = {Johnson, T. C. and Robinson, J. R. and White, S. K. and
                  Zue, Y. and Jaysaval, P.},
  title        = {{E4D} User Guide},
  organization = {Pacific Northwest National Laboratory},
  year         = {2020},
  url          = {https://e4d-userguide.pnnl.gov}
}

@article{si2015tetgen,
  author  = {Si, Hang},
  title   = {{TetGen}, a {Delaunay}-Based Quality Tetrahedral Mesh Generator},
  journal = {ACM Transactions on Mathematical Software},
  volume  = {41},
  number  = {2},
  pages   = {1--36},
  year    = {2015},
  doi     = {10.1145/2629697}
}
```

**3D meshes built with Gmsh (the Gmsh engine and its zones, or Gmsh standing in for TetGen):**
```bibtex
@article{geuzaine2009gmsh,
  author  = {Geuzaine, Christophe and Remacle, Jean-Fran{\c{c}}ois},
  title   = {{Gmsh}: A 3-{D} finite element mesh generator with built-in
             pre- and post-processing facilities},
  journal = {International Journal for Numerical Methods in Engineering},
  volume  = {79},
  number  = {11},
  pages   = {1309--1331},
  year    = {2009},
  doi     = {10.1002/nme.2579}
}
```

**Geophysical modeling (PyGIMLi):**
```bibtex
@article{rucker2017pygimli,
  title   = {pyGIMLi: An open-source library for modelling and inversion in geophysics},
  author  = {R{\"u}cker, Carsten and G{\"u}nther, Thomas and Wagner, Florian M},
  journal = {Computers \& Geosciences},
  volume  = {109},
  pages   = {106--123},
  year    = {2017},
  doi     = {10.1016/j.cageo.2017.07.011}
}
```

**EM modeling (SimPEG):**
```bibtex
@article{cockett2015simpeg,
  title   = {SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications},
  author  = {Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J and Pidlisecky, Adam and Oldenburg, Douglas W},
  journal = {Computers \& Geosciences},
  volume  = {85},
  pages   = {142--154},
  year    = {2015},
  doi     = {10.1016/j.cageo.2015.09.015}
}
```

**Magnetotellurics (`data_processing.mt`):** the processing follows EMTF
(Egbert & Booker 1986; Egbert 1997) with a remote reference (Gamble et al. 1979)
and bounded-influence weights (Chave & Thomson 2004); the 1D inversion is Occam's
(Constable et al. 1987) on Wait's (1954) recursion, jointly with TEM after Meju
(1996) for the static shift (Sternberg et al. 1988); the phase tensor is Caldwell
et al.'s (2004); EMTF XML is Kelbert's (2020) format; the 2D inversion runs on
SimPEG (above). The example site NMX20 is from the USMTArray (IRIS SPUD, CC BY 4.0).
```bibtex
@article{egbert1986robust,
  author  = {Egbert, Gary D. and Booker, John R.},
  title   = {Robust estimation of geomagnetic transfer functions},
  journal = {Geophysical Journal International},
  volume  = {87},
  number  = {1},
  pages   = {173--194},
  year    = {1986},
  doi     = {10.1111/j.1365-246X.1986.tb04552.x}
}

@article{egbert1997robust,
  author  = {Egbert, Gary D.},
  title   = {Robust multiple-station magnetotelluric data processing},
  journal = {Geophysical Journal International},
  volume  = {130},
  number  = {2},
  pages   = {475--496},
  year    = {1997},
  doi     = {10.1111/j.1365-246X.1997.tb05663.x}
}

@article{gamble1979remote,
  author  = {Gamble, T. D. and Goubau, W. M. and Clarke, J.},
  title   = {Magnetotellurics with a remote magnetic reference},
  journal = {Geophysics},
  volume  = {44},
  number  = {1},
  pages   = {53--68},
  year    = {1979},
  doi     = {10.1190/1.1440923}
}

@article{chave2004bounded,
  author  = {Chave, Alan D. and Thomson, David J.},
  title   = {Bounded influence magnetotelluric response function estimation},
  journal = {Geophysical Journal International},
  volume  = {157},
  number  = {3},
  pages   = {988--1006},
  year    = {2004},
  doi     = {10.1111/j.1365-246X.2004.02203.x}
}

@article{constable1987occam,
  author  = {Constable, Steven C. and Parker, Robert L. and Constable, Catherine G.},
  title   = {Occam's inversion: A practical algorithm for generating smooth models
             from electromagnetic sounding data},
  journal = {Geophysics},
  volume  = {52},
  number  = {3},
  pages   = {289--300},
  year    = {1987},
  doi     = {10.1190/1.1442303}
}

@article{wait1954relation,
  author  = {Wait, James R.},
  title   = {On the relation between telluric currents and the Earth's magnetic field},
  journal = {Geophysics},
  volume  = {19},
  number  = {2},
  pages   = {281--289},
  year    = {1954},
  doi     = {10.1190/1.1437994}
}

@article{meju1996joint,
  author  = {Meju, Maxwell A.},
  title   = {Joint inversion of {TEM} and distorted {MT} soundings: Some effective
             practical considerations},
  journal = {Geophysics},
  volume  = {61},
  number  = {1},
  pages   = {56--65},
  year    = {1996},
  doi     = {10.1190/1.1443956}
}

@article{sternberg1988static,
  author  = {Sternberg, Ben K. and Washburne, James C. and Pellerin, Louise},
  title   = {Correction for the static shift in magnetotellurics using transient
             electromagnetic soundings},
  journal = {Geophysics},
  volume  = {53},
  number  = {11},
  pages   = {1459--1468},
  year    = {1988},
  doi     = {10.1190/1.1442426}
}

@article{caldwell2004phase,
  author  = {Caldwell, T. Grant and Bibby, Hugh M. and Brown, Colin},
  title   = {The magnetotelluric phase tensor},
  journal = {Geophysical Journal International},
  volume  = {158},
  number  = {2},
  pages   = {457--469},
  year    = {2004},
  doi     = {10.1111/j.1365-246X.2004.02281.x}
}

@article{kelbert2020emtf,
  author  = {Kelbert, Anna},
  title   = {{EMTF XML}: New data interchange format and conversion tools for
             electromagnetic transfer functions},
  journal = {Geophysics},
  volume  = {85},
  number  = {1},
  pages   = {F1--F17},
  year    = {2020},
  doi     = {10.1190/geo2018-0679.1}
}

@misc{schultz2020usmtarray,
  author    = {Schultz, A. and Pellerin, L. and Bedrosian, P. and Kelbert, A. and
               Crosbie, J.},
  title     = {{USMTArray} South Magnetotelluric Transfer Functions},
  publisher = {Seismological Facility for the Advancement of Geoscience},
  year      = {2020},
  note      = {2020--2023},
  doi       = {10.17611/DP/EMTF/USMTARRAY/SOUTH}
}

@misc{iris2011emtf,
  author    = {{Incorporated Research Institutions for Seismology}},
  title     = {Data Services Products: {EMTF}, The Magnetotelluric Transfer
               Functions},
  publisher = {Seismological Facility for the Advancement of Geoscience},
  year      = {2011},
  doi       = {10.17611/DP/EMTF.1}
}
```

**Shallow seismic processing (`data_processing.seismic_shallow`):** air-wave
onsets are picked with Maeda's (1985) AIC; traces are stacked by common midpoint (Mayne
1962) after NMO with Dix (1955) interval velocities; velocity analysis uses
semblance (Neidell & Taner 1971); the trace QC, mutes and the
flat-event test follow the pitfalls listed by Steeples & Miller (1998).
```bibtex
@article{maeda1985method,
  author  = {Maeda, Naoki},
  title   = {A method for reading and checking phase time in auto-processing
             system of seismic wave data},
  journal = {Zisin (Journal of the Seismological Society of Japan. 2nd ser.)},
  volume  = {38},
  number  = {3},
  pages   = {365--379},
  year    = {1985},
  doi     = {10.4294/zisin1948.38.3_365}
}

@article{dix1955seismic,
  author  = {Dix, C. Hewitt},
  title   = {Seismic velocities from surface measurements},
  journal = {Geophysics},
  volume  = {20},
  number  = {1},
  pages   = {68--86},
  year    = {1955},
  doi     = {10.1190/1.1438126}
}

@article{mayne1962common,
  author  = {Mayne, W. Harry},
  title   = {Common reflection point horizontal data stacking techniques},
  journal = {Geophysics},
  volume  = {27},
  number  = {6},
  pages   = {927--938},
  year    = {1962},
  doi     = {10.1190/1.1439118}
}

@article{neidell1971semblance,
  author  = {Neidell, N. S. and Taner, M. Turhan},
  title   = {Semblance and other coherency measures for multichannel data},
  journal = {Geophysics},
  volume  = {36},
  number  = {3},
  pages   = {482--497},
  year    = {1971},
  doi     = {10.1190/1.1440186}
}

@article{steeples1998avoiding,
  author  = {Steeples, Don W. and Miller, Richard D.},
  title   = {Avoiding pitfalls in shallow seismic reflection surveys},
  journal = {Geophysics},
  volume  = {63},
  number  = {4},
  pages   = {1213--1224},
  year    = {1998},
  doi     = {10.1190/1.1444422}
}
```

**Hydrological modeling (FloPy / MODFLOW):**
```bibtex
@article{bakker2016flopy,
  title   = {Scripting MODFLOW Model Development Using Python and FloPy},
  author  = {Bakker, Mark and Post, Vincent and Langevin, Christian D and Hughes, Joseph D and White, Jeremy T and Starn, J Jeffrey and Fienen, Michael N},
  journal = {Groundwater},
  volume  = {54},
  number  = {5},
  pages   = {733--739},
  year    = {2016},
  doi     = {10.1111/gwat.12413}
}

@misc{langevin2017modflow6,
  author       = {Langevin, Christian D. and Hughes, Joseph D. and Banta, Edward R. and
                  Provost, Alden M. and Niswonger, Richard G. and Panday, Sorab},
  title        = {{MODFLOW} 6 Modular Hydrologic Model},
  howpublished = {U.S. Geological Survey Software},
  year         = {2017},
  doi          = {10.5066/F76Q1VQV}
}

@techreport{langevin2017modflow6gwf,
  author      = {Langevin, Christian D. and Hughes, Joseph D. and Banta, Edward R. and
                 Niswonger, Richard G. and Panday, Sorab and Provost, Alden M.},
  title       = {Documentation for the {MODFLOW} 6 Groundwater Flow Model},
  institution = {U.S. Geological Survey},
  type        = {Techniques and Methods},
  number      = {6-A55},
  year        = {2017},
  doi         = {10.3133/tm6A55}
}
```

**Hydrological modeling (ParFlow):** the four papers ParFlow asks users to cite.
Cite the release you ran as well; https://doi.org/10.5281/zenodo.4816884 resolves
to the latest one.
```bibtex
@article{ashby1996parflow,
  author  = {Ashby, Steven F. and Falgout, Robert D.},
  title   = {A Parallel Multigrid Preconditioned Conjugate Gradient Algorithm for
             Groundwater Flow Simulations},
  journal = {Nuclear Science and Engineering},
  volume  = {124},
  number  = {1},
  pages   = {145--159},
  year    = {1996},
  doi     = {10.13182/NSE96-A24230}
}

@article{jones2001parflow,
  author  = {Jones, Jim E. and Woodward, Carol S.},
  title   = {Newton--{Krylov}-multigrid solvers for large-scale, highly
             heterogeneous, variably saturated flow problems},
  journal = {Advances in Water Resources},
  volume  = {24},
  number  = {7},
  pages   = {763--774},
  year    = {2001},
  doi     = {10.1016/S0309-1708(00)00075-0}
}

@article{kollet2006parflow,
  author  = {Kollet, Stefan J. and Maxwell, Reed M.},
  title   = {Integrated surface--groundwater flow modeling: A free-surface overland
             flow boundary condition in a parallel groundwater flow model},
  journal = {Advances in Water Resources},
  volume  = {29},
  number  = {7},
  pages   = {945--958},
  year    = {2006},
  doi     = {10.1016/j.advwatres.2005.08.006}
}

@article{maxwell2013parflow,
  author  = {Maxwell, Reed M.},
  title   = {A terrain-following grid transform and preconditioner for parallel,
             large-scale, integrated hydrologic modeling},
  journal = {Advances in Water Resources},
  volume  = {53},
  pages   = {109--117},
  year    = {2013},
  doi     = {10.1016/j.advwatres.2012.10.001}
}
```

**Hydrological modeling (ATS):** the code, as ATS asks in all works, and its
watershed-hydrology paper.
```bibtex
@misc{coon2020ats,
  author    = {Coon, E. T. and Berndt, M. and Jan, A. and Svyatsky, D. and
               Atchley, A. L. and Kikinzon, E. and Harp, D. R. and Manzini, G. and
               Shelef, E. and Lipnikov, K. and Garimella, R. and Xu, C. and
               Moulton, J. D. and Karra, S. and Painter, S. L. and Jafarov, E. and
               Molins, S.},
  title     = {Advanced Terrestrial Simulator},
  publisher = {U.S. Department of Energy},
  note      = {Version 1.0},
  year      = {2020},
  doi       = {10.11578/dc.20190911.1}
}

@article{coon2020coupling,
  author  = {Coon, Ethan T. and Moulton, J. David and Kikinzon, Evgeny and
             Berndt, Markus and Manzini, Gianmarco and Garimella, Rao and
             Lipnikov, Konstantin and Painter, Scott L.},
  title   = {Coupling surface flow and subsurface flow in complex soil structures
             using mimetic finite differences},
  journal = {Advances in Water Resources},
  volume  = {144},
  pages   = {103701},
  year    = {2020},
  doi     = {10.1016/j.advwatres.2020.103701}
}
```

**Hydrological modeling (PFLOTRAN):** the three references PFLOTRAN asks users to cite.
```bibtex
@article{hammond2014pflotran,
  author  = {Hammond, Glenn E. and Lichtner, Peter C. and Mills, Richard T.},
  title   = {Evaluating the performance of parallel subsurface simulators: An
             illustrative example with {PFLOTRAN}},
  journal = {Water Resources Research},
  volume  = {50},
  number  = {1},
  pages   = {208--228},
  year    = {2014},
  doi     = {10.1002/2012WR013483}
}

@misc{pflotran-web-page,
  author = {Lichtner, Peter C. and Hammond, Glenn E. and Lu, Chuan and Karra, Satish
            and Bisht, Gautam and Andre, Benjamin and Mills, Richard T. and
            Kumar, Jitendra and Frederick, Jennifer M.},
  title  = {{PFLOTRAN} Web page},
  note   = {http://www.pflotran.org},
  year   = {2020}
}

@techreport{pflotran-user-ref,
  author = {Lichtner, Peter C. and Hammond, Glenn E. and Lu, Chuan and Karra, Satish
            and Bisht, Gautam and Andre, Benjamin and Mills, Richard T. and
            Kumar, Jitendra and Frederick, Jennifer M.},
  title  = {{PFLOTRAN} User Manual},
  note   = {http://documentation.pflotran.org},
  year   = {2020}
}
```

**Related geophysics-hydrology study:**

- Chen, H., Niu, Q., Mendieta, A., Bradford, J., & McNamara, J. (2023). Geophysics-informed hydrologic modeling of a mountain headwater catchment for studying hydrological partitioning in the critical zone. *Water Resources Research, 59*(12), e2023WR035280. https://doi.org/10.1029/2023WR035280
