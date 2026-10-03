"""E4D, PNNL's parallel 3-D ERT code, as an inversion engine.

E4D (Johnson et al., 2010; https://github.com/pnnl/E4D) is a Fortran program
run under MPI. PyHydroGeophysX does not bundle it. This module writes the
files E4D reads, runs E4D on them, and reads back what it writes, so an E4D
inversion goes through the same pipeline as the other engines - QC, error
model, geometric-factor check, outlier rejection, the lambda search - and ends
in the same viewers and exports.

Where E4D runs
--------------
E4D is built from source with gfortran, PETSc and MPI, following the
``Installation.txt`` of its repository, which is written for Linux.

* **Linux** (a workstation or a cluster node): E4D's own platform. ``e4d``
  and ``mpirun`` are found on PATH, through ``$PYHYDRO_E4D`` and
  ``$PYHYDRO_MPIRUN``, or from the paths given in the settings.
* **Windows**: E4D has no Windows build. It runs inside WSL 2, a Linux
  distribution on Windows: build E4D there and choose the ``wsl`` launcher;
  PyHydroGeophysX itself stays on Windows and calls into WSL.
* **macOS**: not covered by E4D's build instructions. A source build with
  gfortran, PETSc and Open MPI is the same procedure and is then used like a
  Linux one, but it is not tested here.
* **Anywhere else** - no E4D at hand, or a cluster whose jobs go through a
  scheduler: the ``files`` launcher writes a complete E4D run folder and
  stops. Run ``mpirun -np N e4d`` in that folder on a machine with E4D, then
  read the result back with :func:`read_e4d_run` (or
  :func:`read_e4d_time_lapse` for a series).

``python -m PyHydroGeophysX.inversion.e4d`` says which of these applies to the
machine it runs on.

E4D needs at least two MPI processes (one master, one or more workers) and no
more workers than electrodes. ``$PYHYDRO_E4D_COMMAND`` (or ``command`` in the
settings) replaces ``mpirun -np N e4d`` with any command line, ``{processes}``
filled in - an ``srun`` line, or a container.

How a run maps onto E4D
-----------------------
E4D models in 3-D. A 3-D survey on an imported tetrahedral mesh is handed to
E4D on that mesh. A 2-D profile is inverted in 3-D on a mesh built around the
line the way E4D users build one (a fine zone under the electrodes inside a
far-reaching outer zone, the topography carried across the line; see
``core.e4d_mesh``), and the section along the line is read back onto the
profile's own parameter mesh, so the result looks and exports like any other.
The full 3-D model is written next to it.

Each fit is E4D's mode ERT3 at a fixed regularization weight: lambda is E4D's
beta, the smoothness is its nearest-neighbour constraint (structural metric 2,
never relaxed), zone 1 (the outer zone) is smoothed into the fine zone as in
E4D's examples, and update option 3 stops E4D once the objective stops
decreasing at that beta - the fixed-lambda run to a plateau the pipeline
expects. E4D has no cap on its outer iterations, so it is stopped here after
``max_iterations`` inverse updates. Data go to E4D as transfer resistances with
a standard deviation of ``err * |R|``; E4D's chi-squared is the mean squared
weighted residual of those resistances, and its own data culling is off
because the pipeline rejects outliers itself.

A time-lapse series is E4D's own time-lapse mode, ERT4
(:func:`invert_e4d_time_lapse`): the first survey is inverted as the baseline,
then each later one starting from the solution before it (E4D's "previous
solution" option), and the smoothness acts on the change from that solution
(structural metric 8 against ``pref``). E4D requires the same measurements in
every survey, so those common to all are inverted. It runs each survey to its
plateau or target in one E4D run, which cannot be stopped part-way through the
series, so the iteration limit does not apply there.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from PyHydroGeophysX._internal.utils import noop as _noop_log

__all__ = [
    "E4DEngine",
    "E4DLauncher",
    "E4DNotRun",
    "E4DSettings",
    "LAUNCHERS",
    "find_e4d",
    "invert_e4d_time_lapse",
    "read_e4d_dpd",
    "read_e4d_log",
    "read_e4d_run",
    "read_e4d_sigma",
    "read_e4d_survey",
    "read_e4d_time_lapse",
    "run_e4d",
    "write_e4d_conductivity",
    "write_e4d_inp",
    "write_e4d_inversion_options",
    "write_e4d_output_options",
    "write_e4d_survey",
    "write_e4d_time_lapse_list",
]

LogFn = Callable[[str], None]

#: Environment variables a site sets once instead of in every run.
ENV_EXECUTABLE = "PYHYDRO_E4D"
ENV_MPIRUN = "PYHYDRO_MPIRUN"
ENV_COMMAND = "PYHYDRO_E4D_COMMAND"

#: ``auto`` tries this computer, then WSL on Windows; ``local`` and ``wsl``
#: insist on one; ``files`` writes the run folder and does not run E4D.
LAUNCHERS = ("auto", "local", "wsl", "files")

#: Names of the files in a run folder. E4D reads names of at most 40
#: characters, cuts the mesh name at its first dot to find the ``.trn``, and
#: reads an unquoted name only up to a slash, so they are short, flat and
#: relative.
MESH_NAME = "e4d"
SURVEY_FILE = "survey.srv"
START_FILE = "start.sig"
OUTPUT_FILE = "e4d.out"
INVERSION_FILE = "e4d.inv"
PREDICTED_FILE = "survey.dpd"
TIME_LAPSE_FILE = "steps.txt"
INFO_FILE = "pyhydro_e4d.json"

_WHERE = ("E4D runs on Linux, on Windows only inside WSL 2, and on macOS when built from "
          "source; elsewhere choose 'Write files only' and run the folder where E4D is.")


class E4DNotRun(RuntimeError):
    """E4D could not be run here; the run folder was written for elsewhere."""

    def __init__(self, message: str, run_dir: Union[str, Path]):
        super().__init__(message)
        self.run_dir = str(run_dir)


# ---------------------------------------------------------------------------
# Where E4D is, and how it is started
# ---------------------------------------------------------------------------
@dataclass
class E4DSettings:
    """How to reach E4D, and the few choices E4D itself needs.

    ``executable`` and ``mpirun`` are program names or paths - Linux paths for
    the ``wsl`` launcher - empty for ``$PYHYDRO_E4D`` / ``$PYHYDRO_MPIRUN`` and
    then PATH. ``element_volume`` is the largest element (m^3) of the fine
    zone of a profile's 3-D mesh, 0 to size it from the electrode spacing.
    ``timeout`` (seconds, 0 for none) bounds one E4D run. ``workdir`` is where
    run folders go.
    """

    launcher: str = "auto"
    executable: str = ""
    mpirun: str = ""
    processes: int = 4
    command: str = ""
    timeout: float = 0.0
    element_volume: float = 0.0
    workdir: str = ""

    @classmethod
    def from_value(cls, value: Any = None) -> "E4DSettings":
        if isinstance(value, cls):
            return value
        known = {item.name for item in fields(cls)}
        settings = cls(**{k: v for k, v in dict(value or {}).items() if k in known})
        settings.launcher = str(settings.launcher or "auto").lower()
        if settings.launcher not in LAUNCHERS:
            raise ValueError(f"E4D launcher {settings.launcher!r} is not one of "
                             f"{', '.join(LAUNCHERS)}.")
        settings.processes = max(2, int(settings.processes or 2))
        return settings


@dataclass
class E4DLauncher:
    """How E4D will be started: ``kind`` is ``local``, ``wsl`` or ``command``
    when it can run, ``files`` or ``missing`` when it cannot (``reason`` says
    why). ``shown`` is the command as a person would type it."""

    kind: str
    argv: List[str] = field(default_factory=list)
    shown: str = ""
    reason: str = ""

    @property
    def runs(self) -> bool:
        return self.kind in ("local", "wsl", "command")

    def describe(self) -> str:
        if self.runs:
            where = {"local": "on this computer", "wsl": "in WSL",
                     "command": "with the configured command"}[self.kind]
            return f"E4D {where}: {self.shown}"
        return self.reason


def _split(command: str) -> List[str]:
    if os.name != "nt":
        return shlex.split(command)
    return [part.strip('"') for part in shlex.split(command, posix=False)]


def _local_program(given: str, *names: str) -> str:
    for candidate in ([given] if given else []) + list(names):
        candidate = str(candidate).strip().strip("'\"")
        if not candidate:
            continue
        if Path(candidate).is_file():
            return str(Path(candidate))
        found = shutil.which(candidate)
        if found:
            return found
        if given:                      # a named program that is not there
            return ""
    return ""


def _decode(raw: bytes) -> str:
    # wsl.exe's own messages are UTF-16; what Linux prints through it is not.
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw.count(b"\x00") > len(raw) // 4:
        return raw.decode("utf-16-le", errors="replace").replace("\x00", "")
    return raw.decode("utf-8", errors="replace")


def _wsl_find(e4d: str, mpirun: str, timeout: float = 60.0) -> Tuple[Optional[Tuple[str, str]], str]:
    """``((e4d, mpirun), "")`` as WSL resolves them, or ``(None, why not)``."""
    wsl = shutil.which("wsl")
    if not wsl:
        return None, "WSL is not installed (wsl.exe was not found)."
    script = f"command -v {shlex.quote(e4d)} && command -v {shlex.quote(mpirun)}"
    try:
        done = subprocess.run([wsl, "-e", "bash", "-lc", script], capture_output=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"WSL did not answer ({exc})."
    found = [line.strip() for line in _decode(done.stdout).splitlines() if line.strip()]
    if done.returncode == 0 and len(found) >= 2:
        return (found[0], found[1]), ""
    detail = " ".join((_decode(done.stderr) + " " + _decode(done.stdout)).split()).lower()
    if "subsystem for linux is not installed" in detail or "wsl --install" in detail:
        return None, ("WSL is not installed (an administrator can install it with "
                      "`wsl --install`, then build E4D in the Linux it provides).")
    if "no installed distributions" in detail or "not installed" in detail:
        return None, "WSL has no Linux distribution installed."
    missing = e4d if not found else mpirun
    return None, f"WSL does not find {missing!r} on its PATH."


def find_e4d(settings: Any = None, *, probe_wsl: bool = True) -> E4DLauncher:
    """How E4D would be started with ``settings`` (see :class:`E4DSettings`).

    Never raises: when E4D cannot run, the launcher's ``reason`` says why and
    which environments it runs in. ``probe_wsl=False`` skips asking WSL, which
    can take a few seconds while its virtual machine starts.
    """
    s = E4DSettings.from_value(settings)
    n = int(s.processes)
    typed = f"mpirun -np {n} e4d"
    if s.launcher == "files":
        return E4DLauncher("files", shown=typed, reason=(
            "Write files only: the E4D run folder is written, and E4D is run on it "
            "elsewhere."))
    command = s.command or os.environ.get(ENV_COMMAND, "")
    if command:
        text = command.replace("{processes}", str(n))
        return E4DLauncher("command", _split(text), text)
    e4d_name = s.executable or os.environ.get(ENV_EXECUTABLE, "")
    mpirun_name = s.mpirun or os.environ.get(ENV_MPIRUN, "")
    notes = []
    if s.launcher in ("auto", "local"):
        e4d = _local_program(e4d_name, "e4d")
        mpirun = _local_program(mpirun_name, "mpirun", "mpiexec")
        if e4d and mpirun:
            return E4DLauncher("local", [mpirun, "-np", str(n), e4d], f"{mpirun} -np {n} {e4d}")
        notes.append("this computer has no " + " and no ".join(
            name for name, ok in (("e4d", e4d), ("mpirun", mpirun)) if not ok)
            + " on PATH")
    if s.launcher in ("auto", "wsl"):
        if os.name != "nt":
            if s.launcher == "wsl":
                notes.append("WSL exists only on Windows")
        elif probe_wsl:
            found, why = _wsl_find(e4d_name or "e4d", mpirun_name or "mpirun")
            if found:
                e4d, mpirun = found
                line = f"{shlex.quote(mpirun)} -np {n} {shlex.quote(e4d)}"
                # The process id is kept so that a run stopped from Windows
                # stops in Linux too: ending wsl.exe does not end its children.
                script = f"{line} & echo $! > .e4d_pid; wait $!"
                return E4DLauncher("wsl", [shutil.which("wsl") or "wsl", "-e", "bash", "-lc",
                                           script], f"wsl {line}")
            notes.append(why.rstrip("."))
        else:
            notes.append("WSL was not checked")
    reason = "E4D was not found: " + "; ".join(notes) + ". " + _WHERE
    return E4DLauncher("missing", shown=typed, reason=reason)


# ---------------------------------------------------------------------------
# The files E4D reads
# ---------------------------------------------------------------------------
def write_e4d_survey(path: Union[str, Path], electrodes: Any, abmn: Any, resistance: Any,
                     std: Any, *, flags: Optional[Sequence[int]] = None) -> Path:
    """Write an E4D survey file (``.srv``).

    ``electrodes`` are x, y, z in survey coordinates (E4D subtracts the
    mesh's ``.trn``); ``abmn`` are 0-based electrode indices with -1 for an
    electrode not used (a pole), written 1-based with 0 as E4D numbers them;
    ``resistance`` the transfer resistances (V/I, Ohm) and ``std`` their
    standard deviations, in Ohm. ``flags`` default to 1, as E4D's examples
    give surface electrodes; E4D reads the flag only for infrastructure.
    """
    xyz = np.asarray(electrodes, dtype=float).reshape(-1, 3)
    abmn = np.asarray(abmn, dtype=np.int64).reshape(-1, 4)
    resistance = np.asarray(resistance, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    if not (len(abmn) == len(resistance) == len(std)):
        raise ValueError("abmn, resistance and std must have one row per measurement.")
    if np.any(abmn >= len(xyz)) or np.any(abmn < -1):
        raise ValueError("abmn refers to electrodes the survey does not have.")
    if np.any(~np.isfinite(std) | (std <= 0)):
        raise ValueError("Every standard deviation must be positive; E4D would "
                         "otherwise give the datum no weight without saying so.")
    flags = np.ones(len(xyz), dtype=int) if flags is None else np.asarray(flags, dtype=int)
    lines = [f"{len(xyz)}        number of electrodes: index x y z flag"]
    lines += [f"{i} {x:.6f} {y:.6f} {z:.6f} {int(f)}"
              for i, ((x, y, z), f) in enumerate(zip(xyz, flags), start=1)]
    lines += ["", f"{len(abmn)}        number of measurements: index a b m n "
                  "resistance(Ohm) sd(Ohm)"]
    one_based = np.where(abmn >= 0, abmn + 1, 0)
    lines += [f"{i} {a} {b} {m} {n} {r:.8e} {s:.8e}"
              for i, ((a, b, m, n), r, s) in enumerate(zip(one_based, resistance, std), start=1)]
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def write_e4d_conductivity(path: Union[str, Path], sigma: Any) -> Path:
    """Write an E4D conductivity file: the count, then one S/m per element."""
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if np.any(~np.isfinite(sigma) | (sigma <= 0)):
        raise ValueError("E4D conductivities must be finite and positive.")
    target = Path(path)
    target.write_text(f"{len(sigma)} 1\n" + "\n".join(f"{v:.8e}" for v in sigma) + "\n",
                      encoding="ascii")
    return target


def write_e4d_inversion_options(path: Union[str, Path], blocks: Sequence[Tuple[int, Sequence[int]]],
                                *, beta: float, plateau: float, target_chi2: float,
                                min_sigma: float, max_sigma: float,
                                change: bool = False) -> Path:
    """Write E4D's inversion options for a smoothness-regularized run.

    ``blocks`` are ``(zone, linked zones)``: each zone is smoothed between
    neighbouring elements (structural metric 2, with a weighting function that
    never relaxes the constraint, the "Occam" setting of E4D's examples), and
    across its boundary into each zone it is linked to. Lambda is E4D's beta,
    held fixed: update option 3 ends the run once the objective falls by less
    than ``plateau`` (a fraction) in an iteration, or the data reach
    ``target_chi2``. Data culling is off.

    With ``change`` (a time-lapse run) it is the change from the previous
    solution that is smoothed - structural metric 8 against ``pref``, E4D's
    difference regularization. For the baseline the previous solution is the
    starting model, so a uniform start smooths the model itself, as metric 2
    does.
    """
    metric = ("8 1.0 1.0 1.0        structural metric 8: smoothness of the change from "
              "the previous solution, x y z weights" if change else
              "2 1.0 1.0 1.0        structural metric 2 (smoothness), x y z weights")
    reference = ("pref        the previous solution" if change else
                 "0.0        reference value (not used by metric 2)")
    lines = [f"{len(blocks)}        number of constraint blocks", ""]
    for zone, links in blocks:
        links = [int(link) for link in links]
        lines += [f"{int(zone)}        zone this block constrains", metric,
                  "1 10.0 0.01        weighting function 1, mean 10, sd 0.01: never relaxed",
                  " ".join(str(v) for v in [len(links)] + links)
                  + "        number of zones linked, and the zones",
                  reference, "1.0        relative weight", ""]
    lines += [f"{float(beta):.8g} {float(plateau):.8g} 0.5        beta (lambda), minimum "
              "relative decrease of the objective, beta reduction (unused)",
              f"{max(float(target_chi2), 1e-6):.8g}        target chi-squared",
              "30 50        minimum and maximum inner iterations",
              f"{float(min_sigma):.8g} {float(max_sigma):.8g}        minimum and maximum "
              "conductivity (S/m)",
              "3        update option 3: stop when the objective stops decreasing",
              "0 3.0        no data culling"]
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def write_e4d_output_options(path: Union[str, Path], predicted: str = PREDICTED_FILE) -> Path:
    """Write E4D's output options: the simulated data, no potentials or Jacobian."""
    target = Path(path)
    target.write_text(f"1        write the simulated data\n{predicted}\n"
                      "0        number of potential fields to write\n"
                      "0        no Jacobian\n", encoding="ascii")
    return target


def _plain_name(name: str) -> str:
    if len(name) > 40 or "/" in name or "\\" in name or " " in name:
        raise ValueError(f"E4D cannot read the file name {name!r}: it must be a plain "
                         "name of at most 40 characters, without spaces or slashes.")
    return name


def write_e4d_inp(folder: Union[str, Path], *, mesh: str, survey: str = SURVEY_FILE,
                  conductivity: str = START_FILE, output: str = OUTPUT_FILE,
                  inversion: str = INVERSION_FILE, reference: str = "none",
                  time_lapse: str = "") -> Path:
    """Write ``e4d.inp`` for an inversion of one survey (mode 3, ERT3), or,
    with a ``time_lapse`` list file, of a baseline and the surveys the list
    names after it (mode 4, ERT4), each starting from the solution before it
    (E4D's baseline mode 2, "use previous solution")."""
    for name in (mesh, survey, conductivity, output, inversion, reference):
        _plain_name(name)
    lines = ["4        run mode: time-lapse inversion (ERT4)" if time_lapse else
             "3        run mode: inversion of one survey (ERT3)",
             f"{mesh}        mesh", f"{survey}        survey",
             f"{conductivity}        starting conductivity",
             f"{output}        output options", f"{inversion}        inversion options",
             f"{reference}        reference model"]
    if time_lapse:
        lines.append(f"{_plain_name(time_lapse)} 2        time-lapse survey list; 2: each "
                     "survey starts from the previous solution")
    target = Path(folder) / "e4d.inp"
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


def write_e4d_time_lapse_list(path: Union[str, Path], surveys: Sequence[str],
                              times: Sequence[float]) -> Path:
    """Write E4D's time-lapse survey list: the count, then a survey file and
    its time per line. E4D names each step's model ``tl_sig<time>`` with the
    time written to three decimals in eight characters, so times stay below
    10,000."""
    if len(surveys) != len(times):
        raise ValueError("One time per time-lapse survey.")
    if any(not 0 <= float(t) < 1.0e4 for t in times):
        raise ValueError("E4D writes time-lapse times in eight characters; keep them "
                         "between 0 and 10,000.")
    lines = [f"{len(surveys)}        number of time-lapse surveys: file time"]
    lines += [f"{_plain_name(name)} {float(t):.3f}" for name, t in zip(surveys, times)]
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="ascii")
    return target


# ---------------------------------------------------------------------------
# What E4D writes
# ---------------------------------------------------------------------------
def read_e4d_sigma(path: Union[str, Path]) -> Tuple[np.ndarray, Optional[float]]:
    """An E4D conductivity file: the values (S/m), and the chi-squared E4D
    records in the header of a ``sigma.N`` it wrote (None in an input file)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().split()
        count = int(float(header[0]))
        values = np.loadtxt(handle, max_rows=count, ndmin=2)[:, 0]
    if len(values) != count:
        raise ValueError(f"{Path(path).name} declares {count} values but holds {len(values)}.")
    chi2 = float(header[2]) if len(header) > 2 else None
    return values, chi2


def read_e4d_survey(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """An E4D survey file: ``electrodes`` (x, y, z, flag), ``abmn`` (1-based,
    0 for none), ``resistance`` and ``std``. Read record by record, as E4D
    reads it, so blank lines and trailing comments do not matter."""
    records = [line.split() for line in
               Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
               if line.strip()]
    n_electrodes = int(float(records[0][0]))
    electrodes = np.array([[float(v) for v in row[1:5]]
                           for row in records[1:n_electrodes + 1]]).reshape(-1, 4)
    count = int(float(records[n_electrodes + 1][0]))
    rows = records[n_electrodes + 2:n_electrodes + 2 + count]
    if len(rows) != count:
        raise ValueError(f"{Path(path).name} declares {count} measurements but holds "
                         f"{len(rows)}.")
    return {"electrodes": electrodes,
            "abmn": np.array([[int(float(v)) for v in row[1:5]] for row in rows]).reshape(-1, 4),
            "resistance": np.array([float(row[5]) for row in rows]),
            "std": np.array([float(row[6]) for row in rows])}


def read_e4d_dpd(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """E4D's simulated data: ``abmn`` (1-based, 0 for none), ``observed`` and
    ``predicted`` resistances, in survey order."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        count = int(float(handle.readline().split()[0]))
        rows = np.loadtxt(handle, max_rows=count, ndmin=2)
    return {"abmn": rows[:, 1:5].astype(int), "observed": rows[:, 5], "predicted": rows[:, 6]}


def read_e4d_log(path: Union[str, Path]) -> Dict[str, Any]:
    """The convergence E4D reports in ``e4d.log``.

    ``chi2`` is every chi-squared it reported, in order: the starting model,
    each inverse update, and the final adjustment; ``updates`` counts the
    inverse updates reported so far; ``converged`` whether E4D said so;
    ``started`` whether it got as far as the starting model's misfit, which an
    input error stops it short of. ``segments`` splits ``chi2`` at each
    starting model: one list for a single survey, and in a time-lapse run one
    for the baseline and one for each survey after it.
    """
    out: Dict[str, Any] = {"chi2": [], "kinds": [], "updates": 0, "converged": False,
                           "started": False, "segments": []}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    pending = None
    for line in text.splitlines():
        if "CONVERGENCE STATISTICS AT STARTING MODEL" in line or "DATA FIT AT STARTING" in line:
            pending = "start"
        elif "AFTER INVERSE UPDATE" in line:
            pending = "update"
        elif "FINAL SOLUTION ADJUSTMENT" in line:
            pending = "final"
        elif "SOLUTION CONVERGED" in line:
            out["converged"] = True
        elif "Chi2 is currently" in line and pending:
            try:
                value = float(line.split("currently", 1)[1].split()[0])
            except (IndexError, ValueError):
                continue                 # written half-way; read again later
            out["chi2"].append(value)
            out["kinds"].append(pending)
            if pending == "start" or not out["segments"]:
                out["segments"].append([])
            out["segments"][-1].append(value)
            if pending == "start":
                out["started"] = True
            elif pending == "update":
                out["updates"] += 1
            pending = None
    return out


def _latest_sigma(folder: Path) -> Optional[Path]:
    numbered = []
    for path in folder.glob("sigma.*"):
        suffix = path.name.split(".", 1)[1]
        if suffix.isdigit():
            numbered.append((int(suffix), path))
    return max(numbered)[1] if numbered else None


def _tail(path: Path, lines: int = 12) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(line.strip() for line in text[-lines:] if line.strip())


# ---------------------------------------------------------------------------
# Running E4D
# ---------------------------------------------------------------------------
def _stop(process: subprocess.Popen, launcher: E4DLauncher, run_dir: Path) -> None:
    """End an E4D run and every MPI process it started."""
    if launcher.kind == "wsl":
        try:
            subprocess.run([launcher.argv[0], "-e", "bash", "-lc",
                            "kill -TERM $(cat .e4d_pid) 2>/dev/null"],
                           cwd=str(run_dir), capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        process.kill()
        process.wait(timeout=20)


def run_e4d(run_dir: Union[str, Path], launcher: E4DLauncher, *, max_iterations: int = 0,
            timeout: float = 0.0, log: LogFn = _noop_log, poll: float = 0.5,
            watch: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Run E4D in ``run_dir`` and wait for it.

    E4D runs until its data reach the target or its objective stops falling;
    it has no limit on its iterations of its own, so it is stopped once it has
    reported ``max_iterations`` inverse updates (0 for no limit), with the
    model and the simulated data of the last of them on disk. Its console
    output goes to ``e4d_console.txt``. ``watch`` is handed the parsed log at
    every poll and once more at the end, for files E4D overwrites as it goes.
    Returns ``returncode``, ``capped`` (stopped at the limit), ``seconds`` and
    the parsed ``e4d.log``.
    """
    run_dir = Path(run_dir)
    if not launcher.runs:
        raise E4DNotRun(launcher.reason, run_dir)
    log_path = run_dir / "e4d.log"
    log_path.unlink(missing_ok=True)
    started = time.monotonic()
    kwargs: Dict[str, Any] = {}
    if os.name != "nt":
        kwargs["start_new_session"] = True        # so the MPI ranks can be stopped too
    with open(run_dir / "e4d_console.txt", "wb") as console:
        try:
            process = subprocess.Popen(launcher.argv, cwd=str(run_dir), stdout=console,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       **kwargs)
        except OSError as exc:
            raise RuntimeError(f"E4D could not be started ({launcher.shown}): {exc}") from exc
        reported, updates, surveys, capped = 0, 0, 0, False
        while process.poll() is None:
            time.sleep(poll)
            history = read_e4d_log(log_path)
            for kind, value in list(zip(history["kinds"], history["chi2"]))[reported:]:
                if kind == "start":
                    surveys, updates = surveys + 1, 0
                updates += kind == "update"
                label = {"start": "starting model", "final": "final adjustment"}.get(
                    kind, f"update {updates}")
                # A time-lapse run reports every survey after the baseline anew.
                where = "" if surveys <= 1 else f" survey {surveys}"
                log(f"    E4D{where} {label}: chi2 = {value:.3f}")
                reported += 1
            if watch is not None:
                watch(history)
            if max_iterations and history["updates"] >= int(max_iterations):
                capped = True
                _stop(process, launcher, run_dir)
                break
            if timeout and time.monotonic() - started > float(timeout):
                _stop(process, launcher, run_dir)
                raise RuntimeError(f"E4D did not finish within {float(timeout):g} s; it was "
                                   f"stopped. Its files are in {run_dir}.")
        returncode = process.wait()
    history = read_e4d_log(log_path)
    if watch is not None:
        watch(history)
    if not history["started"]:
        detail = _tail(log_path) or _tail(run_dir / "e4d_console.txt")
        raise RuntimeError(f"E4D stopped before it fitted the starting model (exit "
                           f"{returncode}): {detail or 'no output'}. The run folder is "
                           f"{run_dir}.")
    return {"returncode": returncode, "capped": capped, "history": history,
            "seconds": time.monotonic() - started}


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
def _find_cells(mesh, points: np.ndarray, dim: int) -> np.ndarray:
    """The cell of ``mesh`` holding each point, the nearest cell for any
    point on or just outside its edge, -1 for none."""
    import pygimli as pg
    from scipy.spatial import cKDTree

    found = np.full(len(points), -1, dtype=np.int64)
    for index, point in enumerate(points):
        cell = mesh.findCell(pg.Pos(*point[:dim]))
        if cell is not None:
            found[index] = int(cell.id())
    missing = found < 0
    if np.any(missing):
        centres = np.asarray(mesh.cellCenters(), dtype=float)[:, :dim]
        _, nearest = cKDTree(centres).query(points[missing, :dim])
        found[missing] = nearest
    return found


def _inside_2d(mesh, points: np.ndarray) -> np.ndarray:
    """The 2-D cell holding each (x, z), -1 outside the mesh."""
    import pygimli as pg

    found = np.full(len(points), -1, dtype=np.int64)
    nodes = np.asarray(mesh.positions(), dtype=float)[:, :2]
    low, high = nodes.min(axis=0), nodes.max(axis=0)
    near = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
    cells = [cell.ids() for cell in mesh.cells()]
    if all(len(ids) == 3 for ids in cells):
        import matplotlib.tri as mtri

        finder = mtri.Triangulation(nodes[:, 0], nodes[:, 1],
                                    np.asarray(cells, dtype=int)).get_trifinder()
        found[near] = finder(points[near, 0], points[near, 1])
        return found
    for index in near:
        cell = mesh.findCell(pg.Pos(float(points[index, 0]), float(points[index, 1])))
        if cell is not None:
            found[index] = int(cell.id())
    return found


def _survey_data(container, relative_error: float = 0.03, error_floor: float = 0.0,
                 program: str = "E4D") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(resistance, std, k)``: what E4D (or R2, R3t) fits, from a loaded
    container.

    The resistance is the measured transfer resistance ``r`` where the
    container has it, else ``rhoa / k``; ``k`` is what turns E4D's simulated
    resistances back into apparent resistivities. The standard deviation is
    ``err * |R|`` - a relative error in rhoa is the same in R - with
    ``relative_error`` where the container has no usable ``err``, and never
    below ``error_floor``.
    """
    n = int(container.size())
    rhoa = np.asarray(container["rhoa"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(container["r"], dtype=float) if container.haveData("r") else None
        k = np.asarray(container["k"], dtype=float) if container.haveData("k") else None
        if r is not None and np.all(np.isfinite(r) & (r != 0)):
            resistance = r
            if k is None:
                k = rhoa / r
        elif k is not None:
            resistance = rhoa / np.where(np.abs(k) > 1e-12, k, np.nan)
        else:
            resistance = np.full(n, np.nan)
    if np.any(~np.isfinite(resistance)) or k is None:
        raise ValueError(f"{program} needs each reading's transfer resistance: the data "
                         "carry neither a usable resistance (r) nor geometric factors (k).")
    err = np.asarray(container["err"], dtype=float) if container.haveData("err") else None
    if err is None or not np.all(np.isfinite(err) & (err > 0)):
        err = np.full(n, float(relative_error))
    return resistance, np.abs(resistance) * np.maximum(err, float(error_floor)), k


def _profile_mesh(container, para, settings: E4DSettings, workdir: Path, *,
                  outer_width: float, log: LogFn, label: str = "E4D"):
    """The 3-D mesh E4D (or R3t, ``label``) inverts a profile on, built
    around the line.

    The line runs along x at y = 0. The fine zone covers what the profile's
    parameter mesh covers - its ends and its depth - and reaches half that
    depth to either side of the line; the outer zone reaches ``outer_width``
    electrode spreads beyond it. The ground along the line is carried across
    it unchanged. Unless ``element_volume`` is set, the fine zone's largest
    element is sized to hold about twenty thousand of them, and never less than
    a tetrahedron of twice the electrode spacing: TetGen refines toward the
    electrodes on its own, and the quality bound multiplies the count several
    times over.
    """
    from PyHydroGeophysX.core import e4d_mesh as e4d

    sensors = np.asarray(container.sensorPositions(), dtype=float)
    x, z = sensors[:, 0], sensors[:, 1]
    order = np.argsort(x)
    nodes = np.asarray(para.positions(), dtype=float)
    spread = max(float(np.ptp(x)), 1.0)
    unique = np.unique(np.round(x, 6))
    spacing = float(np.median(np.diff(unique))) if len(unique) > 1 else spread
    depth = max(float(z.min() - nodes[:, 1].min()), 2.0 * spacing)
    pad_x = max(float(x.min() - nodes[:, 0].min()), float(nodes[:, 0].max() - x.max()), spacing)
    fine = (float(np.ptp(x)) + 2.0 * pad_x) * depth * depth
    volume = float(settings.element_volume) or max(
        (2.0 * spacing) ** 3 / (6.0 * math.sqrt(2.0)), fine / 20000.0)
    outer = max(float(outer_width) if outer_width > 0 else 4.0, 1.0) * spread
    topography = float(np.ptp(z)) > 1e-6

    def ground(px: float, _py: float) -> float:
        return float(np.interp(px, x[order], z[order]))

    config, _ = e4d.e4d_config_from_electrodes(
        np.column_stack([x, np.zeros_like(x), z]), surface=ground if topography else float(z[0]),
        fine_padding=(pad_x, 0.5 * depth), fine_depth_padding=depth,
        outer_distance=outer, bottom_depth=depth + outer, fine_volume=volume,
        quality=1.4, refine_offset=0.0, topography_points=8 if topography else 0)
    log(f"  {label} mesh around the line: fine zone {np.ptp(x) + 2 * pad_x:.0f} x {depth:.0f} m "
        f"wide x {depth:.0f} m deep, elements up to {volume:.3g} m^3 there; outer zone "
        f"{outer:.0f} m beyond it")
    built = e4d.build_e4d_mesh(config, workdir / "mesh", name="profile", log=log)
    return built["mesh"], np.column_stack([x, np.zeros_like(x), z])


class E4DEngine:
    """E4D behind the common ERT engine contract (see the module docstring).

    ``options`` are :class:`E4DSettings`. The result mesh is the profile's
    parameter mesh for a 2-D survey, and the parameter domain (cells marked 2
    and above) of the imported mesh for a 3-D one; the whole 3-D model of
    every run is written to its run folder as ``resistivity_3d.vtk``.
    """

    name = "e4d"

    def __init__(self, container, mesh, *, model_constraints=(1e-2, 1e5), method=None,
                 zones=None, log: LogFn = _noop_log, options: Any = None, outer_width: float = 0.0):
        import pygimli as pg
        from pygimli.physics import ert

        from .ert_mesh import _interface_regions
        from .ert_zones import zone_prior

        self.settings = E4DSettings.from_value(options)
        self.container = container
        self._log = log
        self.workdir = Path(self.settings.workdir or tempfile.mkdtemp(prefix="e4d_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.launcher = find_e4d(self.settings)
        log("  " + self.launcher.describe())
        self._counter = [0]                   # run folders, shared with with_data copies
        self._bounds = tuple(float(v) for v in model_constraints)
        self._take_data(container)
        n_electrodes = int(container.sensorCount())
        if self.settings.processes > n_electrodes + 1:
            log(f"  E4D: {self.settings.processes} processes is more than {n_electrodes} "
                f"electrodes allow; using {n_electrodes + 1}")
            self.settings.processes = n_electrodes + 1
            self.launcher = find_e4d(self.settings)

        self.zone_prior = None
        self.profile = int(mesh.dim()) == 2
        if self.profile:
            fop = ert.ERTModelling()
            fop.setData(container)
            fop.setMesh(mesh)
            self.mesh = pg.Mesh(fop.paraDomain)
            mesh3d, self._electrodes = _profile_mesh(
                container, self.mesh, self.settings, self.workdir,
                outer_width=float(outer_width), log=log)
            # Section cells read the element on the line at their centre; the
            # elements take the section's value at their (x, z), across the line.
            centres = np.asarray(self.mesh.cellCenters(), dtype=float)
            self._element_of_cell = _find_cells(
                mesh3d, np.column_stack([centres[:, 0], np.zeros(len(centres)), centres[:, 1]]), 3)
            elements = np.asarray(mesh3d.cellCenters(), dtype=float)
            self._cell_of_element = _inside_2d(self.mesh, elements[:, [0, 2]])
            if zones:
                self.zone_prior = zone_prior(self.mesh, zones, bounds=model_constraints)
            units = _interface_regions(mesh, [int(c.id()) for c in mesh.cells()
                                              if int(c.marker()) > 1])
            markers = np.asarray(mesh3d.cellMarkers(), dtype=int)   # 1 outer, 2 fine
            if units is not None:
                # Zone outlines the smoothness must not cross: each part of the
                # section is an E4D zone of its own, linked to none.
                inside = self._cell_of_element >= 0
                markers = markers.copy()
                markers[inside] = 3 + units[self._cell_of_element[inside]]
                log(f"  E4D: smoothness cut along the zone outlines ({int(units.max()) + 1} "
                    "zones of their own)")
            self._cut = units is not None
        else:
            mesh3d = mesh
            sensors = np.asarray(container.sensorPositions(), dtype=float)
            self._electrodes = sensors[:, :3]
            markers = np.asarray(mesh3d.cellMarkers(), dtype=int)
            active = np.flatnonzero(markers > 1)
            if not active.size:
                raise ValueError("The mesh has no cells marked 2 or above to invert.")
            self._active = active
            self.mesh = mesh3d.createMeshByCellIdx(pg.IVector(active.tolist()))
            self._cut = False
            if zones:
                log("  Note: a-priori zones are drawn on 2-D sections and do not apply "
                    "to a 3-D mesh; E4D starts from a uniform model.")
        self.mesh3d = mesh3d
        self._markers = markers
        written = self._write_mesh(markers)
        self._zones = written["zones"]
        self._zone_of_marker = written["zone_of_marker"]
        self._translation = written["translation"]
        log(f"  E4D mesh: {mesh3d.cellCount()} elements, {mesh3d.nodeCount()} nodes, "
            f"zones {', '.join(str(z) for z in np.unique(self._zones))}")

    def _take_data(self, container) -> None:
        """The transfer resistances E4D fits, and their standard deviations."""
        self.container = container
        self._resistance, self._std, self._k = _survey_data(container)
        rhoa = np.asarray(container["rhoa"], dtype=float)
        good = rhoa[np.isfinite(rhoa) & (rhoa > 0)]
        self._background = float(np.median(good)) if good.size else 100.0

    def with_data(self, container) -> "E4DEngine":
        """This engine on other data from the same survey - what is left after
        an outlier pass - on the same E4D mesh."""
        import copy

        twin = copy.copy(self)
        twin._take_data(container)
        return twin

    # -- files ------------------------------------------------------------
    def _write_mesh(self, markers) -> Dict[str, Any]:
        from PyHydroGeophysX.core.e4d_mesh import write_e4d_mesh

        written = write_e4d_mesh(self.mesh3d, self.workdir / "mesh", MESH_NAME,
                                 translation=self._electrodes.mean(axis=0), zones=markers)
        self._mesh_files = [Path(p) for p in written["files"].values()]
        return written

    def _blocks(self) -> List[Tuple[int, List[int]]]:
        """Which zones are smoothed into which. For a profile, the outer zone
        is smoothed into the fine zone, as E4D's examples link them, and both
        into every part the zone outlines cut the section into, while those
        parts are not smoothed into one another: the smoothness stops at the
        outlines and nowhere else. For a 3-D mesh, every zone is smoothed into
        every other. A link is written once, from the lower-numbered zone."""
        zones = [int(z) for z in np.unique(self._zones)]
        if self.profile:
            # Markers 1 and 2 are the outer and the fine zone; E4D numbers
            # zones 1, 2, ... in order, so look them up rather than assume.
            joined = {self._zone_of_marker[m] for m in (1, 2) if m in self._zone_of_marker}
            return [(z, [o for o in zones if o != z and not (o in joined and o < z)]
                     if z in joined else []) for z in zones]
        return [(z, [other for other in zones if other > z]) for z in zones]

    def _sigma(self, model) -> np.ndarray:
        """Element conductivities for a model on the result mesh (ohm-m), or
        the starting model when ``model`` is None."""
        n = int(self.mesh3d.cellCount())
        rho = np.full(n, self._background)
        if model is None and self.zone_prior is not None:
            model = self.zone_prior.with_background(self._background)
        if model is not None:
            model = np.asarray(model, dtype=float).reshape(-1)
            if self.profile:
                inside = self._cell_of_element >= 0
                rho[inside] = model[self._cell_of_element[inside]]
            else:
                rho[self._active] = model
        low, high = self._bounds
        return 1.0 / np.clip(rho, low, high)

    def _prepare(self, *, lam, plateau_tolerance, target_chi2, start_model,
                 steps: Sequence[Any] = ()) -> Path:
        """A run folder holding everything E4D reads; with ``steps`` - the
        surveys after this engine's own, as ``(resistance, std, k)`` in its
        measurement order - a time-lapse run."""
        self._counter[0] += 1
        self._run = self._counter[0]
        run_dir = self.workdir / f"run{self._run:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        for source in self._mesh_files:
            target = run_dir / source.name
            try:
                os.link(source, target)
            except OSError:
                shutil.copyfile(source, target)
        sensors = self._electrodes
        abmn = np.column_stack([np.asarray(self.container[key], dtype=np.int64)
                                for key in ("a", "b", "m", "n")])
        write_e4d_survey(run_dir / SURVEY_FILE, sensors, abmn, self._resistance, self._std)
        names, ks = [], [self._k]
        for number, (resistance, std, k) in enumerate(steps, start=1):
            names.append(f"step{number:03d}.srv")
            write_e4d_survey(run_dir / names[-1], sensors, abmn, resistance, std)
            ks.append(k)
        if names:
            write_e4d_time_lapse_list(run_dir / TIME_LAPSE_FILE, names,
                                      [float(n) for n in range(1, len(names) + 1)])
        write_e4d_conductivity(run_dir / START_FILE, self._sigma(start_model))
        low, high = self._bounds
        write_e4d_inversion_options(run_dir / INVERSION_FILE, self._blocks(), beta=lam,
                                    plateau=plateau_tolerance, target_chi2=target_chi2,
                                    min_sigma=1.0 / high, max_sigma=1.0 / low,
                                    change=bool(names))
        write_e4d_output_options(run_dir / OUTPUT_FILE)
        write_e4d_inp(run_dir, mesh=f"{MESH_NAME}.1.node",
                      time_lapse=TIME_LAPSE_FILE if names else "")
        info = {"written_by": "PyHydroGeophysX", "engine": "e4d", "lambda": float(lam),
                "processes": int(self.settings.processes), "profile": bool(self.profile),
                "command": self.launcher.shown, "n_data": int(len(self._resistance)),
                "time_lapse_steps": len(names)}
        np.save(run_dir / "section_k.npy", self._k)
        if names:
            np.save(run_dir / "steps_k.npy", np.vstack(ks))
        if self.profile:
            from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

            via_ascii_path(self.mesh.save, run_dir / "section.bms", mode="write")
            np.save(run_dir / "section_elements.npy", self._element_of_cell)
        else:
            np.save(run_dir / "active_elements.npy", self._active)
        (run_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return run_dir

    # -- the contract -----------------------------------------------------
    def reference_model(self):
        """E4D's smoothness acts on the model itself; there is no reference to pin."""
        return None

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None):
        from .ert_inversion import ERTRun

        run_dir = self._prepare(lam=lam, plateau_tolerance=plateau_tolerance,
                                target_chi2=target_chi2, start_model=start_model)
        if not self.launcher.runs:
            raise E4DNotRun(
                f"{self.launcher.reason} The E4D run folder is {run_dir}: run "
                f"`mpirun -np {self.settings.processes} e4d` in it on a machine with E4D, "
                "then read the result with PyHydroGeophysX.inversion.e4d.read_e4d_run.",
                run_dir)
        self._log(f"    E4D run {self._run} at beta = {float(lam):g} in {run_dir.name}")
        outcome = run_e4d(run_dir, self.launcher, max_iterations=int(max_iterations),
                          timeout=float(self.settings.timeout), log=self._log)
        result = read_e4d_run(run_dir, mesh=self.mesh3d)
        history = list(outcome["history"]["chi2"])
        target = float(target_chi2)
        if outcome["capped"]:
            stop = "iteration_cap"
        elif target > 0 and result["chi2"] <= target:
            stop = "target"
        else:
            stop = "plateau"
        if not history or abs(history[-1] - result["chi2"]) > 1e-6 * max(1.0, result["chi2"]):
            history.append(result["chi2"])
        model = result["model"]
        return ERTRun(
            lam=float(lam), chi2=float(result["chi2"]),
            iterations=int(outcome["history"]["updates"]), stop=stop,
            convergence=history, model=np.asarray(model, dtype=float),
            response=np.asarray(result["response"], dtype=float), mesh=self.mesh,
            coverage=None,
            metrics={"backend": "e4d", "chi2": float(result["chi2"]), "lambda": float(lam),
                     "iterations": int(outcome["history"]["updates"]),
                     "n_data": int(len(self._resistance)),
                     "e4d_run": str(run_dir), "e4d_command": self.launcher.shown,
                     "e4d_seconds": float(outcome["seconds"]),
                     "e4d_elements": int(self.mesh3d.cellCount()),
                     "chi2_definition": "E4D: mean of ((R_obs - R_pred) / sd)^2, "
                                        "transfer resistances, sd = err * |R_obs|"})

    def fit_time_lapse(self, steps: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], *,
                       lam: float, plateau_tolerance: float, target_chi2: float,
                       start_model=None) -> Dict[str, Any]:
        """E4D's own time-lapse inversion (ERT4) of this engine's survey as the
        baseline and ``steps`` after it, each ``(resistance, std, k)`` in this
        survey's measurement order. See :func:`read_e4d_time_lapse` for what
        is returned."""
        run_dir = self._prepare(lam=lam, plateau_tolerance=plateau_tolerance,
                                target_chi2=target_chi2, start_model=start_model, steps=steps)
        if not self.launcher.runs:
            raise E4DNotRun(
                f"{self.launcher.reason} The E4D time-lapse run folder is {run_dir}: run "
                f"`mpirun -np {self.settings.processes} e4d` in it on a machine with E4D, "
                "then read the result with "
                "PyHydroGeophysX.inversion.e4d.read_e4d_time_lapse.", run_dir)
        # E4D rewrites its simulated-data file at every forward run. Each
        # survey's last one is still on disk when the next survey's starting
        # misfit is reported, so it is kept then, recognised by its data.
        observed = [self._resistance] + [step[0] for step in steps]
        kept: Dict[int, bool] = {}

        def watch(history: Dict[str, Any]) -> None:
            for step in range(len(history["segments"]) - 1):
                if step not in kept:
                    done = _keep_prediction(run_dir, step, observed)
                    if done is not None:
                        kept[step] = done

        self._log(f"    E4D time-lapse run at beta = {float(lam):g}: the baseline and "
                  f"{len(steps)} survey(s) after it, in {run_dir.name}")
        outcome = run_e4d(run_dir, self.launcher, timeout=float(self.settings.timeout),
                          log=self._log, watch=watch)
        if len(steps) not in kept:
            kept[len(steps)] = bool(_keep_prediction(run_dir, len(steps), observed))
        result = read_e4d_time_lapse(run_dir, mesh=self.mesh3d)
        result["seconds"] = float(outcome["seconds"])
        return result


def _keep_prediction(run_dir: Path, step: int, observed: Sequence[np.ndarray]) -> Optional[bool]:
    """Copy E4D's simulated data to ``step<NNN>.dpd`` when they are survey
    ``step``'s; False once another survey's have replaced them, None when the
    file cannot be read just now."""
    try:
        predicted = read_e4d_dpd(run_dir / PREDICTED_FILE)
    except (OSError, ValueError, IndexError):
        return None
    mine = observed[step]
    if (len(predicted["observed"]) == len(mine)
            and np.allclose(predicted["observed"], mine, rtol=1e-4, atol=0.0)):
        shutil.copyfile(run_dir / PREDICTED_FILE, run_dir / f"step{step:03d}.dpd")
        return True
    return False


def read_e4d_time_lapse(run_dir: Union[str, Path], *, mesh=None) -> Dict[str, Any]:
    """Read back an E4D time-lapse run (ERT4) in ``run_dir``.

    Returns ``steps``, one entry per survey, the baseline first, each like
    :func:`read_e4d_run`'s result (``model``, ``response``, ``chi2``, ...)
    with that survey's ``convergence`` from ``e4d.log``; ``result_mesh``;
    ``mesh3d`` with ``resistivity_t<i>`` cell data, also written as
    ``resistivity_3d_timelapse.vtk``; and ``missing_fit``, the surveys whose
    simulated data E4D overwrote before they could be kept (their model is
    read, their response and chi2 are NaN).
    """
    from PyHydroGeophysX.core.e4d_mesh import read_e4d_mesh
    from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

    run_dir = Path(run_dir)
    info = json.loads((run_dir / INFO_FILE).read_text(encoding="utf-8"))
    count = int(info.get("time_lapse_steps", 0))
    if mesh is None:
        mesh = read_e4d_mesh(run_dir / f"{MESH_NAME}.1.node", conductivity=False)
    models = {}
    for path in run_dir.glob("tl_sig*"):
        try:
            models[int(round(float(path.name[len("tl_sig"):])))] = path.name
        except ValueError:
            continue
    segments = read_e4d_log(run_dir / "e4d.log")["segments"]
    steps, missing = [], []
    for step in range(count + 1):
        if step and step not in models:
            raise FileNotFoundError(f"E4D wrote no model for time-lapse survey {step} "
                                    f"(tl_sig{step}.000) in {run_dir}; see its e4d.log.")
        predicted = f"step{step:03d}.dpd"
        if not (run_dir / predicted).is_file():
            missing.append(step)
        entry = read_e4d_run(run_dir, mesh=mesh, sigma_file=models.get(step, "") if step else "",
                             predicted_file=predicted,
                             survey_file=SURVEY_FILE if step == 0 else f"step{step:03d}.srv",
                             k_row=step, export=False)
        entry["convergence"] = list(segments[step]) if step < len(segments) else []
        steps.append(entry)
        mesh[f"resistivity_t{step}"] = 1.0 / entry["sigma"]
    out: Dict[str, Any] = {"run_dir": str(run_dir), "steps": steps, "mesh3d": mesh,
                           "missing_fit": missing,
                           "result_mesh": steps[0].get("result_mesh")}
    try:
        via_ascii_path(mesh.exportVTK, run_dir / "resistivity_3d_timelapse.vtk", mode="write")
        out["vtk_3d"] = str(run_dir / "resistivity_3d_timelapse.vtk")
    except Exception:  # noqa: BLE001 - the models are still returned
        out["vtk_3d"] = ""
    return out


def invert_e4d_time_lapse(containers: Sequence[Any], mesh, *, lam: float,
                          plateau_tolerance: float = 0.005, target_chi2: float = 1.0,
                          model_constraints=(1e-2, 1e5), zones=None, options: Any = None,
                          outer_width: float = 0.0, relative_error: float = 0.03,
                          error_floor: float = 0.005, log: LogFn = _noop_log):
    """A time-lapse series inverted by E4D, shaped like the in-house result.

    E4D inverts the first survey as the baseline, then each later one starting
    from the solution before it and smoothing the change from it (see
    :func:`write_e4d_inversion_options`). Every survey must hold the same
    measurements in the same order, as E4D checks; the measurements common to
    all of them are used, in the baseline's order. Returns an object with
    ``final_models`` (cells x steps on ``mesh``'s parameter domain, or the
    profile's section), ``mesh``, ``iteration_chi2`` (each survey's final
    chi2), ``convergence`` (per survey) and ``meta``. A survey without a usable
    error column is given ``relative_error``; no error is taken below
    ``error_floor``.
    """
    import types

    import pygimli as pg

    if len(containers) < 2:
        raise ValueError("A time-lapse inversion needs at least two surveys.")
    keys = []
    for container in containers:
        abmn = np.column_stack([np.asarray(container[key], dtype=np.int64)
                                for key in ("a", "b", "m", "n")])
        keys.append({tuple(row): index for index, row in enumerate(abmn)})
    common = [key for key in keys[0] if all(key in other for other in keys[1:])]
    if not common:
        raise ValueError("The surveys share no measurement (the same A, B, M and N), "
                         "which an E4D time-lapse inversion needs.")
    dropped = max(len(k) for k in keys) - len(common)
    if dropped:
        log(f"  E4D time-lapse: {len(common)} measurements are common to all "
            f"{len(containers)} surveys and are inverted; up to {dropped} others are "
            "left out, since E4D needs the same measurements in every survey.")
    keep = np.zeros(int(containers[0].size()), dtype=bool)
    keep[[keys[0][key] for key in common]] = True
    baseline = pg.DataContainerERT(containers[0])
    # BVector from the numpy array: a Python list of bools marks nothing.
    baseline.markInvalid(pg.core.BVector(~keep))
    baseline.removeInvalid()
    resistance, std, _ = _survey_data(baseline, relative_error, error_floor)
    baseline["err"] = std / np.abs(resistance)     # the errors the steps are given too
    order = [keys[0][key] for key in common]          # the baseline's own order
    rows = [tuple(int(v) for v in row) for row in np.column_stack(
        [np.asarray(containers[0][key], dtype=np.int64)[np.sort(order)]
         for key in ("a", "b", "m", "n")])]
    steps = []
    for container, index in zip(containers[1:], keys[1:]):
        resistance, std, k = _survey_data(container, relative_error, error_floor)
        pick = np.asarray([index[row] for row in rows])
        steps.append((resistance[pick], std[pick], k[pick]))

    engine = E4DEngine(baseline, mesh, model_constraints=model_constraints, zones=zones,
                       log=log, options=options, outer_width=outer_width)
    out = engine.fit_time_lapse(steps, lam=lam, plateau_tolerance=plateau_tolerance,
                                target_chi2=target_chi2)
    if out["missing_fit"]:
        log(f"  Note: E4D replaced the simulated data of survey(s) "
            f"{', '.join(str(s + 1) for s in out['missing_fit'])} before they could be "
            "kept; their models are read, their fit is not known.")
    models = np.column_stack([np.asarray(step["model"], dtype=float) for step in out["steps"]])
    finals = [float(step["chi2"]) for step in out["steps"]]
    log("  E4D time-lapse chi2 per survey: " + ", ".join(f"{c:.2f}" for c in finals))
    note = ("E4D time-lapse: each survey starts from the previous solution and the "
            "smoothness acts on the change from it (E4D's difference regularization); "
            "the temporal weight alpha and the interval weighting belong to the "
            "in-house joint inversion and are not used.")
    return types.SimpleNamespace(
        final_models=models, mesh=engine.mesh, iteration_chi2=finals, all_chi2=finals,
        convergence=[step["convergence"] for step in out["steps"]],
        responses=[step.get("response") for step in out["steps"]],
        meta={"backend_version": "E4D", "linearized_solver": "E4D PCGLS",
              "temporal_weighting": {"mode": "e4d_previous_solution", "note": note},
              "e4d": {"run_dir": out["run_dir"], "vtk_3d": out["vtk_3d"],
                      "command": engine.launcher.shown, "measurements": len(common),
                      "missing_fit": list(out["missing_fit"]),
                      "seconds": out.get("seconds", 0.0)}})


def read_e4d_run(run_dir: Union[str, Path], *, mesh=None, sigma_file: str = "",
                 predicted_file: str = PREDICTED_FILE, survey_file: str = SURVEY_FILE,
                 k_row: int = -1, export: bool = True) -> Dict[str, Any]:
    """Read back an E4D inversion run in ``run_dir``, here or on a cluster.

    Returns ``mesh3d`` (the E4D mesh with ``resistivity`` cell data, in survey
    coordinates), ``sigma`` (S/m per element), ``chi2`` (E4D's, recomputed
    from the simulated data), ``convergence`` (from ``e4d.log``),
    ``predicted`` resistances and, when the folder was written by
    :class:`E4DEngine`, ``model`` and ``result_mesh`` - the section along a
    profile, or the parameter domain of a 3-D mesh - and the predicted
    apparent resistivities ``response``. Writes ``resistivity_3d.vtk``.

    The model is the last ``sigma.N`` unless ``sigma_file`` names another (a
    time-lapse step's ``tl_sig``); ``predicted_file`` and ``survey_file`` are
    the simulated data and the survey they belong to, and ``k_row`` picks
    that survey's geometric factors in a time-lapse run. For a time-lapse run
    as a whole, see :func:`read_e4d_time_lapse`.
    """
    from PyHydroGeophysX.core.e4d_mesh import read_e4d_mesh
    from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

    run_dir = Path(run_dir)
    if mesh is None:
        mesh = read_e4d_mesh(run_dir / f"{MESH_NAME}.1.node", conductivity=False)
    latest = run_dir / sigma_file if sigma_file else _latest_sigma(run_dir)
    sigma, _ = read_e4d_sigma(latest if latest is not None else run_dir / START_FILE)
    if len(sigma) != int(mesh.cellCount()):
        raise ValueError(f"{(latest or run_dir / START_FILE).name} has {len(sigma)} values "
                         f"for a mesh of {mesh.cellCount()} elements.")
    predicted_path = run_dir / predicted_file
    survey = read_e4d_survey(run_dir / survey_file)
    std = survey["std"]
    if not predicted_path.is_file():
        if predicted_file == PREDICTED_FILE:
            raise FileNotFoundError(f"E4D wrote no simulated data ({predicted_file}) in "
                                    f"{run_dir}.")
        # A time-lapse step whose simulated data E4D overwrote before they
        # could be kept: the model is known, the fit is not.
        nan = np.full(len(std), np.nan)
        predicted = {"observed": survey["resistance"], "predicted": nan}
    else:
        predicted = read_e4d_dpd(predicted_path)
    if len(std) != len(predicted["observed"]):
        raise ValueError(f"{PREDICTED_FILE} has {len(predicted['observed'])} data, the "
                         f"survey {len(std)}.")
    residual = (predicted["observed"] - predicted["predicted"]) / std
    out: Dict[str, Any] = {
        "run_dir": str(run_dir), "sigma": sigma, "predicted": predicted["predicted"],
        "observed": predicted["observed"], "chi2": float(np.mean(residual ** 2)),
        "convergence": read_e4d_log(run_dir / "e4d.log")["chi2"],
        "sigma_file": str(latest) if latest is not None else ""}
    rho = 1.0 / sigma
    mesh["resistivity"] = rho
    out["mesh3d"] = mesh
    out["vtk_3d"] = ""
    if export:
        try:
            via_ascii_path(mesh.exportVTK, run_dir / "resistivity_3d.vtk", mode="write")
            out["vtk_3d"] = str(run_dir / "resistivity_3d.vtk")
        except Exception:  # noqa: BLE001 - the model is still returned
            pass
    k_path = run_dir / ("steps_k.npy" if k_row >= 0 else "section_k.npy")
    if k_path.is_file():
        k = np.load(k_path)
        out["response"] = predicted["predicted"] * (k[k_row] if k_row >= 0 else k)
    if (run_dir / "section_elements.npy").is_file():
        out["model"] = rho[np.load(run_dir / "section_elements.npy")]
        section = run_dir / "section.bms"
        if section.is_file():
            from PyHydroGeophysX.core.mesh_serialization import read_bms

            out["result_mesh"] = read_bms(section)
    elif (run_dir / "active_elements.npy").is_file():
        active = np.load(run_dir / "active_elements.npy")
        out["model"] = rho[active]
    return out


# ---------------------------------------------------------------------------
# Command line: where is E4D?
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report whether, and how, PyHydroGeophysX can run E4D. " + _WHERE)
    parser.add_argument("--launcher", choices=LAUNCHERS, default="auto")
    parser.add_argument("--executable", default="", help="the e4d program")
    parser.add_argument("--mpirun", default="", help="mpirun or mpiexec")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--command", default="", help="a full command line to run instead")
    parser.add_argument("--json", action="store_true", help="print one JSON object (Studio)")
    args = parser.parse_args(argv)
    launcher = find_e4d({"launcher": args.launcher, "executable": args.executable,
                         "mpirun": args.mpirun, "processes": args.processes,
                         "command": args.command})
    result = {"runs": launcher.runs, "kind": launcher.kind, "command": launcher.shown,
              "message": launcher.describe(), "platform": sys.platform}
    if args.json:
        print(json.dumps({"ok": True, "result": result}), flush=True)
    else:
        print(launcher.describe())
    return 0 if launcher.runs or args.json else 1


if __name__ == "__main__":
    raise SystemExit(main())
