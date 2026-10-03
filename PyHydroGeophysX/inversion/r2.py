"""R2 and R3t, Andrew Binley's ERT inversion codes, as inversion engines.

R2 (2-D) and R3t (3-D) are the finite-element resistivity inversion programs
of Andrew Binley (Lancaster University; Binley and Slater, 2020) that ResIPy
drives. PyHydroGeophysX does not bundle them. This module writes the files
they read, runs them, and reads back what they write, so an R2 or R3t
inversion goes through the same pipeline as the other engines - QC, error
model, geometric-factor check, outlier rejection - and ends in the same
viewers and exports.

Where R2 and R3t run
--------------------
Both are distributed as Windows programs, ``R2.exe`` and ``R3t.exe``, from
Binley's web page and inside every ResIPy installation (its ``exe`` folder).

* **Windows**: natively. The program is the one named in the settings, else
  ``$PYHYDRO_R2`` / ``$PYHYDRO_R3T``, else an installed ResIPy's copy, else
  the first ``R2.exe`` / ``R3t.exe`` on PATH.
* **Linux and macOS**: through Wine (``wine R2.exe``), as the manuals and
  ResIPy run them. A native build named in the settings is run directly.
* **Anywhere else**: the ``files`` launcher writes the run folder and stops.
  Run the program in that folder where it is, then read the result back with
  :func:`read_r2_run`.

Both are provided for non-commercial use; commercial use needs the author's
permission (see their manuals).

``python -m PyHydroGeophysX.inversion.r2`` says which of these applies to the
machine it runs on.

How a run maps onto R2 and R3t
------------------------------
R2 inverts a profile on the profile's own 2-D mesh. Every element is a
parameter, the outer region's too, as R2 meshes are used; the result is read
on the parameter domain. R3t inverts a 3-D survey on its tetrahedral mesh,
and a profile on a 3-D mesh built around the line as for E4D (see
``inversion.e4d``), read back as the section along the line - far slower
than R2, and with the ground off the line held only by the smoothness, so R2
is the engine for profiles.

R3t leaves out of its fit, and of its error file, any reading it cannot use:
one with no signal over uniform ground (M and N symmetric about A and B) or
one whose polarity its model does not reproduce. chi2 is then over the
readings it fitted, as its RMS is, and the others' predictions come from a
forward run on its model (:func:`forward_r2`).

Both are Occam inversions: at every iteration they search the smoothing weight
alpha for the lowest misfit themselves, and stop when the RMS misfit reaches
their tolerance, when it stops improving, or after ``max_iterations``. There
is no fixed regularization weight, so the pipeline's lambda and its lambda
search do not apply; the alpha of the last iteration is reported instead. The
data go in as transfer resistances with their own standard deviations
``err * |R|`` (``a_wgt = b_wgt = 0``) and are inverted as logarithms, so the
RMS misfit is the root mean square of ``(log R_pred - log R_obs) / err`` -
chi2 = RMS^2, as for the other engines, and the tolerance is
``sqrt(target_chi2)``. The data weights are left alone (``error_mod = 0``)
because the pipeline rejects outliers itself.

A-priori zones give the starting model, and a fixed zone is held at its value
(``param = 0``), which both programs support. Zone outlines the smoothness
must not cross become R2/R3t zones, between which no smoothness is applied;
the outer region joins the zone nearest to it.

A time-lapse series (:func:`invert_r2_time_lapse`) is the difference
inversion of LaBrecque and Yang (2001): the first survey is inverted, then
every later one starting from that model, with the smoothness acting on the
change from it (``reg_mode = 2``) and the data replaced by
``d - d0 + f(m0)``. R2 forms that dataset itself from a second data column;
for R3t it is written from the baseline's simulated data, as its manual
describes.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
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
    "ENGINES",
    "LAUNCHERS",
    "R2Engine",
    "R2Launcher",
    "R2NotRun",
    "R2Settings",
    "complete_prediction",
    "find_r2",
    "forward_r2",
    "invert_r2_time_lapse",
    "read_r2_errors",
    "read_r2_out",
    "read_r2_run",
    "read_r2_values",
    "run_r2",
    "write_r2_in",
    "write_r2_mesh",
    "write_r2_protocol",
    "write_r2_start",
    "write_r3t_in",
    "write_r3t_mesh",
]

LogFn = Callable[[str], None]

#: The engine names, and the program each runs.
ENGINES = {"r2": "R2", "r3t": "R3t"}

#: Environment variables a site sets once instead of in every run.
ENV_EXECUTABLE = {"r2": "PYHYDRO_R2", "r3t": "PYHYDRO_R3T"}

#: ``auto`` finds and runs the program; ``files`` writes the run folder only.
LAUNCHERS = ("auto", "files")

#: Names of the files in a run folder: R2 reads a start file name of at most
#: 15 characters, R3t one of at most 20.
PROTOCOL_FILE = "protocol.dat"
START_FILE = "start_res.dat"
INFO_FILE = "pyhydro_r2.json"
MESH_FILE = {"r2": "mesh.dat", "r3t": "mesh3d.dat"}

#: R2's and R3t's own flag for an apparent resistivity it could not compute.
_NO_VALUE = -99999.0

_WHERE = ("R2 and R3t are Windows programs: they run natively on Windows and through Wine "
          "on Linux and macOS; elsewhere choose 'Write files only' and run the folder "
          "where they are.")


def _engine_name(program: str) -> str:
    name = str(program).lower()
    if name not in ENGINES:
        raise ValueError(f"Unknown program {program!r}; choose one of {', '.join(ENGINES)}.")
    return name


class R2NotRun(RuntimeError):
    """R2/R3t could not be run here; the run folder was written for elsewhere."""

    def __init__(self, message: str, run_dir: Union[str, Path]):
        super().__init__(message)
        self.run_dir = str(run_dir)


# ---------------------------------------------------------------------------
# Where R2 and R3t are, and how they are started
# ---------------------------------------------------------------------------
@dataclass
class R2Settings:
    """How to reach R2 or R3t, and the few choices they need.

    ``executable`` is the program (a path or a name on PATH), empty for
    ``$PYHYDRO_R2`` / ``$PYHYDRO_R3T``, then ResIPy's copy, then PATH;
    ``wine`` is the Wine program on Linux and macOS, empty for ``wine`` or
    ``wine64`` on PATH. ``element_volume`` is the largest element (m^3) of the
    fine zone of the 3-D mesh R3t inverts a profile on, 0 to size it from the
    electrode spacing. ``timeout`` (seconds, 0 for none) bounds one run.
    ``workdir`` is where run folders go.
    """

    launcher: str = "auto"
    executable: str = ""
    wine: str = ""
    timeout: float = 0.0
    element_volume: float = 0.0
    workdir: str = ""

    @classmethod
    def from_value(cls, value: Any = None) -> "R2Settings":
        if isinstance(value, cls):
            return value
        known = {item.name for item in fields(cls)}
        settings = cls(**{k: v for k, v in dict(value or {}).items() if k in known})
        settings.launcher = str(settings.launcher or "auto").lower()
        if settings.launcher not in LAUNCHERS:
            raise ValueError(f"R2/R3t launcher {settings.launcher!r} is not one of "
                             f"{', '.join(LAUNCHERS)}.")
        return settings


@dataclass
class R2Launcher:
    """How R2 or R3t will be started: ``kind`` is ``native`` or ``wine`` when
    it can run, ``files`` or ``missing`` when it cannot (``reason`` says why).
    ``path`` is the program and ``source`` where it was found."""

    program: str
    kind: str
    argv: List[str] = field(default_factory=list)
    shown: str = ""
    reason: str = ""
    path: str = ""
    source: str = ""

    @property
    def runs(self) -> bool:
        return self.kind in ("native", "wine")

    def describe(self) -> str:
        if self.runs:
            how = "through Wine" if self.kind == "wine" else "on this computer"
            where = f" ({self.source})" if self.source else ""
            return f"{self.program} {how}: {self.path}{where}"
        return self.reason


def resipy_executable(program: str) -> str:
    """The R2/R3t program inside an installed ResIPy, or ''. ResIPy is
    located without being imported."""
    name = ENGINES[_engine_name(program)] + ".exe"
    try:
        spec = importlib.util.find_spec("resipy")
    except (ImportError, ValueError):
        return ""
    if spec is None:
        return ""
    roots = list(spec.submodule_search_locations or [])
    if not roots and spec.origin:
        roots = [str(Path(spec.origin).parent)]
    for root in roots:
        candidate = Path(root) / "exe" / name
        if candidate.is_file():
            return str(candidate)
    return ""


def find_r2(program: str = "r2", settings: Any = None) -> R2Launcher:
    """How ``program`` (``r2`` or ``r3t``) would be started with ``settings``
    (see :class:`R2Settings`). Never raises: when it cannot run, the
    launcher's ``reason`` says why and where it runs."""
    name = _engine_name(program)
    exe = ENGINES[name]
    s = R2Settings.from_value(settings)
    if s.launcher == "files":
        return R2Launcher(exe, "files", shown=f"{exe}.exe", reason=(
            f"Write files only: the {exe} run folder is written, and {exe} is run on it "
            "elsewhere."))
    given = str(s.executable or "").strip().strip("'\"")
    path, source = "", ""
    if given:
        found = given if Path(given).is_file() else shutil.which(given)
        if not found:
            return R2Launcher(exe, "missing", shown=f"{exe}.exe", reason=(
                f"{exe} was not found at {given!r}. {_WHERE}"))
        path, source = str(found), "settings"
    if not path:
        env = os.environ.get(ENV_EXECUTABLE[name], "").strip().strip("'\"")
        if env:
            found = env if Path(env).is_file() else shutil.which(env)
            if not found:
                return R2Launcher(exe, "missing", shown=f"{exe}.exe", reason=(
                    f"${ENV_EXECUTABLE[name]} names {env!r}, which does not exist. {_WHERE}"))
            path, source = str(found), f"${ENV_EXECUTABLE[name]}"
    if not path:
        path = resipy_executable(name)
        source = "ResIPy" if path else ""
    if not path:
        for candidate in (f"{exe}.exe", exe):
            found = shutil.which(candidate)
            if found:
                path, source = found, "PATH"
                break
    if not path:
        return R2Launcher(exe, "missing", shown=f"{exe}.exe", reason=(
            f"{exe} was not found: name {exe}.exe in the settings, set "
            f"${ENV_EXECUTABLE[name]}, or install ResIPy, which carries it. {_WHERE}"))
    if os.name != "nt" and path.lower().endswith(".exe"):
        wine = s.wine or shutil.which("wine") or shutil.which("wine64") or ""
        if s.wine and not (Path(s.wine).is_file() or shutil.which(s.wine)):
            wine = ""
        if not wine:
            return R2Launcher(exe, "missing", shown=f"wine {path}", path=path, source=source,
                              reason=(f"{exe}.exe ({path}) is a Windows program and needs Wine "
                                      "here, which was not found (on Debian or Ubuntu: "
                                      "`sudo apt install wine`; on macOS: Homebrew's "
                                      "wine-stable)."))
        wine = shutil.which(wine) or wine
        return R2Launcher(exe, "wine", [wine, path], f"{wine} {path}", path=path, source=source)
    return R2Launcher(exe, "native", [path], path, path=path, source=source)


# ---------------------------------------------------------------------------
# The files R2 and R3t read
# ---------------------------------------------------------------------------
def _fmt(value: float) -> str:
    return f"{float(value):.8e}"


def write_r2_mesh(path: Union[str, Path], nodes: Any, cells: Any, params: Any, zones: Any, *,
                  dirichlet: int = 0) -> Path:
    """Write R2's ``mesh.dat``: ``nodes`` (n, 2) x, z; ``cells`` (m, 3) or
    (m, 4) 0-based node indices, counter-clockwise; ``params`` and ``zones``
    per element (``param`` 0 holds an element at its starting value, and such
    elements must come last); ``dirichlet`` a 1-based node, 0 for R2 to pick."""
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells, dtype=np.int64) + 1
    lines = [f"{len(cells)} {len(nodes)} {int(dirichlet)}"]
    lines += [" ".join(str(int(v)) for v in (index, *cell, param, zone))
              for index, (cell, param, zone) in enumerate(zip(cells, params, zones), start=1)]
    lines += [f"{index} {x:.6f} {z:.6f}" for index, (x, z) in enumerate(nodes[:, :2], start=1)]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_r3t_mesh(path: Union[str, Path], nodes: Any, cells: Any, params: Any, zones: Any, *,
                   dirichlet: int, datum: float = 0.0) -> Path:
    """Write R3t's ``mesh3d.dat`` for a tetrahedral mesh: ``nodes`` (n, 3),
    ``cells`` (m, 4) 0-based, ``params`` and ``zones`` per element,
    ``dirichlet`` the 1-based fixed-potential node, far from the electrodes.
    With more than one zone, every zone smooths with weight 1."""
    nodes = np.asarray(nodes, dtype=float)
    cells = np.asarray(cells, dtype=np.int64) + 1
    zones = np.asarray(zones, dtype=np.int64)
    lines = [f"{len(cells)} {len(nodes)} 1 {float(datum):.6f} 4 0"]
    lines += [" ".join(str(int(v)) for v in (index, *cell, param, zone))
              for index, (cell, param, zone) in enumerate(zip(cells, params, zones), start=1)]
    lines += [f"{index} {x:.6f} {y:.6f} {z:.6f}"
              for index, (x, y, z) in enumerate(nodes[:, :3], start=1)]
    lines.append(str(int(dirichlet)))
    count = int(zones.max()) if zones.size else 1
    if count > 1:
        lines += [f"{zone} 1.0" for zone in range(1, count + 1)]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_r2_protocol(path: Union[str, Path], datasets: Sequence[Dict[str, Any]], *,
                      three_d: bool = False) -> Path:
    """Write ``protocol.dat``, one block per dataset (R2 and R3t invert every
    block in turn with the same settings). A dataset has ``abmn`` (1-based
    electrode numbers), ``data`` (transfer resistances, with their sign),
    ``std`` and, for R2's difference inversion, ``reference`` (the baseline's
    resistances). R3t numbers every electrode on string 1."""
    lines: List[str] = []
    for dataset in datasets:
        abmn = np.asarray(dataset["abmn"], dtype=np.int64)
        data = np.asarray(dataset["data"], dtype=float)
        std = np.asarray(dataset["std"], dtype=float)
        reference = dataset.get("reference")
        lines.append(str(len(data)))
        for index, row in enumerate(abmn):
            numbers = (" ".join(f"1 {int(e)}" for e in row) if three_d
                       else " ".join(str(int(e)) for e in row))
            values = [_fmt(data[index])]
            if reference is not None:
                values.append(_fmt(reference[index]))
            values.append(_fmt(std[index]))
            lines.append(f"{index + 1} {numbers} {' '.join(values)}")
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_r2_start(path: Union[str, Path], centres: Any, resistivity: Any) -> Path:
    """Write a starting model, one line per element in element order: the
    centroid (x, z or x, y, z; not used by the programs) and the resistivity,
    then its log10, as R2's and R3t's own ``_res.dat`` files are laid out."""
    centres = np.asarray(centres, dtype=float)
    rho = np.asarray(resistivity, dtype=float)
    lines = [" ".join(f"{v:.6f}" for v in centre) + f" {r:.8e} {math.log10(r):.6f}"
             for centre, r in zip(centres, rho)]
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_r2_in(folder: Union[str, Path], *, electrode_nodes: Sequence[int], tolerance: float,
                max_iterations: int, reg_mode: int = 0, quadrilateral: bool = False,
                header: str = "PyHydroGeophysX R2 inversion") -> Path:
    """Write ``R2.in`` for an inversion on ``mesh.dat`` from ``start_res.dat``:
    log data (R2 turns this off itself for ``reg_mode`` 2), individual data
    errors, unmodified weights, the sensitivity map, the whole mesh output.
    ``electrode_nodes`` are 1-based mesh nodes in electrode order."""
    lines = [header[:80],
             f"1 {6 if quadrilateral else 3} 3.0 0 1"]     # inverse, mesh.dat, 2.5-D, no singularity removal
    if not quadrilateral:
        lines.append("1.0")                               # coordinate scale
    lines += ["0", START_FILE,                            # start from a file
              "1 0.0",                                    # linear-filter regularization, full step
              f"1 {int(reg_mode)}",                       # log data
              f"{float(tolerance):.6g} {int(max_iterations)} 0 1.0",
              "0.0 0.0 -1e10 1e10",                       # errors from protocol.dat, keep all data
              "0",                                        # no output polygon
              str(len(electrode_nodes))]
    lines += [f"{number} {int(node)}" for number, node in enumerate(electrode_nodes, start=1)]
    path = Path(folder) / "R2.in"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_r3t_in(folder: Union[str, Path], *, electrode_nodes: Sequence[int], tolerance: float,
                 max_iterations: int, z_range: Tuple[float, float], reg_mode: int = 0,
                 header: str = "PyHydroGeophysX R3t inversion") -> Path:
    """Write ``R3t.in`` for an inversion on ``mesh3d.dat`` from
    ``start_res.dat``, set up as :func:`write_r2_in` is; ``z_range`` covers
    the mesh so every element is written out."""
    lines = [header[:80],
             "1 0 1",                                     # inverse, no singularity removal, sensitivity
             "0", START_FILE,
             "1 0.0",
             f"1 {int(reg_mode)}",
             f"{float(tolerance):.6g} {int(max_iterations)} 0 1.0",
             "0.0 0.0 -1e10 1e10",
             f"{float(z_range[0]):.6f} {float(z_range[1]):.6f}",
             "0",
             str(len(electrode_nodes))]
    lines += [f"1 {number} {int(node)}" for number, node in enumerate(electrode_nodes, start=1)]
    path = Path(folder) / "R3t.in"
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


# ---------------------------------------------------------------------------
# The files R2 and R3t write
# ---------------------------------------------------------------------------
_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
_ALPHA = re.compile(rf"Alpha:\s*({_FLOAT})\s+RMS Misfit:\s*({_FLOAT})")


def _number(text: str) -> float:
    match = re.search(_FLOAT, text)
    return float(match.group().replace("D", "E").replace("d", "e")) if match else float("nan")


def read_r2_out(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse ``R2.out`` / ``R3t.out``.

    Returns ``datasets``, one entry per dataset inverted, each with ``rms``
    (the starting RMS misfit, then the misfit after every iteration),
    ``alpha`` (the smoothing weight each iteration settled on), ``stop``
    (``target``, ``plateau``, ``iteration_cap``, ``failed`` or '' while
    running) and ``message``; and ``fatal``, the program's error message if
    it gave up, else ''.
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"datasets": [], "fatal": ""}
    datasets: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    trials: List[Tuple[float, float]] = []
    fatal = ""
    lines = text.splitlines()
    for number, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("Iteration"):
            parts = line.split()
            if current is None or (len(parts) > 1 and parts[1] == "1") or current["stop"]:
                current = {"rms": [], "alpha": [], "stop": "", "message": ""}
                datasets.append(current)
            trials = []
        elif current is None:
            if "FATAL" in line:
                follow = " ".join(part.strip() for part in lines[number + 1:number + 3])
                fatal = " ".join(f"{line} {follow}".replace("**", "").split())
            continue
        elif line.startswith("Initial RMS Misfit:"):
            if not current["rms"]:
                current["rms"].append(_number(line.split(":", 1)[1]))
        elif line.startswith("Alpha:"):
            match = _ALPHA.search(line)
            if match:
                trials.append((float(match.group(1)), float(match.group(2))))
        elif line.startswith("Final RMS Misfit:"):
            current["rms"].append(_number(line.split(":", 1)[1]))
            current["alpha"].append(min(trials, key=lambda t: t[1])[0] if trials else float("nan"))
        elif "Solution converged" in line:
            current.update(stop="target", message=line)
        elif "Solution not improved" in line or "Step length too small" in line:
            current.update(stop="plateau", message=line)
        elif "Solution not converged in" in line:
            current.update(stop="iteration_cap", message=line)
        elif "failure" in line.lower() and "Outputing" in line:
            current.update(stop="failed", message=line)
        if "FATAL" in line and not fatal:
            follow = " ".join(part.strip() for part in lines[number + 1:number + 3])
            fatal = " ".join(f"{line} {follow}".replace("**", "").split())
    return {"datasets": datasets, "fatal": fatal}


def read_r2_values(path: Union[str, Path], *, three_d: bool = False) -> np.ndarray:
    """The values of an R2/R3t ``_res.dat`` (resistivity) or ``_sen.dat``
    file, one row per element. Resistivity is the column after the
    coordinates; the whole array is returned."""
    table = np.loadtxt(path, ndmin=2)
    return table[:, (3 if three_d else 2):]


def read_r2_errors(path: Union[str, Path], *, three_d: bool = False) -> Dict[str, np.ndarray]:
    """Read ``f00N_err.dat``: the electrodes as written to protocol.dat, the
    ``normalised`` misfit, and the ``observed`` and ``calculated`` apparent
    resistivities R2 or R3t computed for each reading (NaN where they flagged
    a geometric factor they could not compute)."""
    table = np.loadtxt(path, skiprows=1, ndmin=2)
    first = 8 if three_d else 4
    electrodes = table[:, 1:first:2] if three_d else table[:, :4]
    observed, calculated = table[:, first + 1].copy(), table[:, first + 2].copy()
    for column in (observed, calculated):
        column[column <= _NO_VALUE] = np.nan
    return {"electrodes": electrodes.astype(np.int64), "normalised": table[:, first],
            "observed": observed, "calculated": calculated}


def _tail(path: Path, lines: int = 8) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(line.strip() for line in text[-lines:] if line.strip())


# ---------------------------------------------------------------------------
# Running R2 and R3t
# ---------------------------------------------------------------------------
def _stop(process: subprocess.Popen) -> None:
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
        process.kill()
        process.wait(timeout=20)


def _launch(launcher: R2Launcher, folder: Path, *, timeout: float = 0.0, poll: float = 0.5,
            on_poll: Optional[Callable[[], None]] = None) -> int:
    """Run the program in ``folder`` until it ends; its console output goes to
    ``console.txt``. Returns its exit code (0 even when it gave up: the
    ``.out`` file says that)."""
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kwargs["start_new_session"] = True
    started = time.monotonic()
    with open(folder / "console.txt", "wb") as console:
        try:
            process = subprocess.Popen(launcher.argv, cwd=str(folder), stdout=console,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                       **kwargs)
        except OSError as exc:
            raise RuntimeError(f"{launcher.program} could not be started "
                               f"({launcher.shown}): {exc}") from exc
        while process.poll() is None:
            time.sleep(poll)
            if on_poll is not None:
                on_poll()
            if timeout and time.monotonic() - started > float(timeout):
                _stop(process)
                raise RuntimeError(f"{launcher.program} did not finish within "
                                   f"{float(timeout):g} s; it was stopped. Its files are in "
                                   f"{folder}.")
        return process.wait()


def run_r2(run_dir: Union[str, Path], launcher: R2Launcher, *, timeout: float = 0.0,
           log: LogFn = _noop_log, poll: float = 0.5) -> Dict[str, Any]:
    """Run R2 or R3t in ``run_dir`` and wait for it; its console output goes
    to ``console.txt``. Returns ``returncode``, ``seconds`` and the parsed
    ``.out`` file (see :func:`read_r2_out`)."""
    run_dir = Path(run_dir)
    if not launcher.runs:
        raise R2NotRun(launcher.reason, run_dir)
    out_path = run_dir / f"{launcher.program}.out"
    out_path.unlink(missing_ok=True)
    started = time.monotonic()
    reported = [0, 0]                          # datasets, iterations of the last
    try:
        info = json.loads((run_dir / INFO_FILE).read_text(encoding="utf-8"))
        many = int(info.get("datasets", 1)) > 1
    except (OSError, ValueError):
        many = False

    def report(parsed: Dict[str, Any]) -> None:
        sets = parsed["datasets"]
        for index in range(reported[0], len(sets)):
            rms = sets[index]["rms"]
            done = reported[1] if index == reported[0] else 0
            where = f" dataset {index + 1}" if many else ""
            for step in range(done, len(rms)):
                label = "starting model" if step == 0 else f"iteration {step}"
                log(f"    {launcher.program}{where} {label}: RMS {rms[step]:.3f} "
                    f"(chi2 {rms[step] ** 2:.3f})")
            reported[0], reported[1] = index, len(rms)

    returncode = _launch(launcher, run_dir, timeout=timeout, poll=poll,
                         on_poll=lambda: report(read_r2_out(out_path)))
    parsed = read_r2_out(out_path)
    report(parsed)
    if parsed["fatal"]:
        hint = ""
        if "Initial RMS shows solution" in parsed["fatal"]:
            hint = (" The starting model already fits the data to the target misfit: the "
                    "data errors are larger than the misfit, so lower the error or the "
                    "target chi2.")
        raise RuntimeError(f"{launcher.program} stopped: {parsed['fatal']}.{hint} The run "
                           f"folder is {run_dir}.")
    if not parsed["datasets"]:
        detail = _tail(out_path) or _tail(run_dir / "console.txt")
        raise RuntimeError(f"{launcher.program} stopped before it inverted anything (exit "
                           f"{returncode}): {detail or 'no output'}. The run folder is "
                           f"{run_dir}.")
    return {"returncode": returncode, "seconds": time.monotonic() - started, "out": parsed}


def forward_r2(run_dir: Union[str, Path], launcher: R2Launcher, dataset: int = 1, *,
               timeout: float = 0.0) -> np.ndarray:
    """The transfer resistances the model of dataset ``dataset`` in
    ``run_dir`` predicts for every one of its readings, from a forward run of
    the program on that model (in ``forward_NNN`` beside it).

    R3t leaves out of its fit, and of its ``_err.dat``, a reading it cannot
    use: one whose geometric factor over uniform ground is infinite (M and N
    symmetric about A and B, so no signal over a uniform half-space), or one
    whose polarity its model does not reproduce. That reading's prediction is
    still wanted, and a forward run gives it.
    """
    run_dir = Path(run_dir)
    if not launcher.runs:
        raise R2NotRun(launcher.reason, run_dir)
    info = json.loads((run_dir / INFO_FILE).read_text(encoding="utf-8"))
    three_d, program = bool(info["three_d"]), str(info["program"])
    folder = run_dir / f"forward_{int(dataset):03d}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir()
    mesh_name = MESH_FILE["r3t" if three_d else "r2"]
    shutil.copyfile(run_dir / mesh_name, folder / mesh_name)
    model = "model_res.dat"
    shutil.copyfile(run_dir / f"f{int(dataset):03d}_res.dat", folder / model)
    data = np.load(run_dir / f"data_{int(dataset):03d}.npz")
    # In a forward run the programs read only the electrodes of each reading.
    write_r2_protocol(folder / PROTOCOL_FILE, [{"abmn": data["numbers"], "data": data["fitted"],
                                                "std": data["std"]}], three_d=three_d)
    nodes = [int(node) for node in info["electrode_nodes"]]
    if three_d:
        lines = ["PyHydroGeophysX R3t forward", "0 0 1", "0", model, str(len(nodes))]
        lines += [f"1 {number} {node}" for number, node in enumerate(nodes, start=1)]
    else:
        quadrilateral = bool(info.get("quadrilateral", False))
        lines = ["PyHydroGeophysX R2 forward", f"0 {6 if quadrilateral else 3} 3.0 0 1"]
        lines += ([] if quadrilateral else ["1.0"]) + ["0", model, "0", str(len(nodes))]
        lines += [f"{number} {node}" for number, node in enumerate(nodes, start=1)]
    (folder / f"{program}.in").write_text("\n".join(lines) + "\n", encoding="ascii")
    _launch(launcher, folder, timeout=timeout)
    result = folder / f"{program}_forward.dat"
    if not result.is_file():
        detail = _tail(folder / f"{program}.out") or _tail(folder / "console.txt")
        raise RuntimeError(f"The {program} forward run wrote no {result.name}: "
                           f"{detail or 'no output'}. Its folder is {folder}.")
    table = np.loadtxt(result, skiprows=1, ndmin=2)
    if len(table) != len(data["fitted"]):
        raise ValueError(f"{result.name} has {len(table)} readings, the protocol "
                         f"{len(data['fitted'])}.")
    return table[:, 9 if three_d else 5]


# ---------------------------------------------------------------------------
# Reading a run back
# ---------------------------------------------------------------------------
def _match_rows(written: np.ndarray, reported: np.ndarray) -> np.ndarray:
    """For every row of ``reported`` (electrode numbers), the row of
    ``written`` it is, in order, so repeated readings pair up one by one;
    -1 for none."""
    from collections import defaultdict, deque

    slots: Dict[Tuple[int, ...], Any] = defaultdict(deque)
    for index, row in enumerate(np.asarray(written, dtype=np.int64)):
        slots[tuple(row)].append(index)
    return np.asarray([slots[tuple(row)].popleft() if slots[tuple(row)] else -1
                       for row in np.asarray(reported, dtype=np.int64)], dtype=np.int64)


def _normalised(predicted_fit: np.ndarray, fitted: np.ndarray, std: np.ndarray,
                log_data: bool) -> np.ndarray:
    """The programs' normalised misfit: (log f - log d) / (sd / |d|) for log
    data, (f - d) / sd otherwise."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_data:
            return (np.log(np.abs(predicted_fit)) - np.log(np.abs(fitted))) / (std / np.abs(fitted))
        return (predicted_fit - fitted) / std


def complete_prediction(result: Dict[str, Any], predicted: np.ndarray) -> Dict[str, Any]:
    """``result`` (from :func:`read_r2_run`) with the prediction for every
    reading, from :func:`forward_r2`, and so a response for every reading.
    ``chi2`` stays that of the readings the program fitted, as its RMS is;
    ``chi2_all`` counts the ones it left out too."""
    run_dir = Path(result["run_dir"])
    info = json.loads((run_dir / INFO_FILE).read_text(encoding="utf-8"))
    data = np.load(run_dir / f"data_{int(result['dataset']):03d}.npz")
    predicted = np.asarray(predicted, dtype=float)
    offset = data["offset"] if "offset" in data else 0.0
    misfit = _normalised(predicted - offset, data["fitted"], data["std"],
                         bool(info.get("log_data", True)))
    result = dict(result)
    result.update(predicted=predicted, chi2_all=float(np.mean(misfit ** 2)))
    if "k" in data:
        result["response"] = predicted * data["k"]
    return result

def read_r2_run(run_dir: Union[str, Path], dataset: int = 1) -> Dict[str, Any]:
    """Read back dataset ``dataset`` (1 for the first) of an R2 or R3t run in
    ``run_dir``, here or wherever it was run.

    Returns ``element_resistivity`` (every mesh element, in the mesh's own
    order), ``chi2`` (the mean squared normalised misfit, R2's RMS squared),
    ``convergence`` (chi2 at the start and after every iteration), ``stop``,
    ``alpha`` (the smoothing weight of the last iteration), ``iterations``,
    ``predicted`` and ``observed`` transfer resistances, and, when the folder
    was written by :class:`R2Engine`, ``model`` on the result mesh,
    ``response`` (predicted apparent resistivities), ``coverage`` (log10 of
    the sensitivity, when the program wrote it) and ``result_mesh``.

    ``unreported`` lists the readings the program left out of its fit and of
    its error file; their prediction is NaN here, and chi2, like the
    program's RMS, is over the others. :func:`forward_r2` and
    :func:`complete_prediction` fill their predictions in where the program
    can run.
    """
    run_dir = Path(run_dir)
    info = json.loads((run_dir / INFO_FILE).read_text(encoding="utf-8"))
    program = str(info["program"])
    three_d = bool(info["three_d"])
    stem = f"f{int(dataset):03d}"
    result_path = run_dir / f"{stem}_res.dat"
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise FileNotFoundError(f"{program} wrote no model for dataset {dataset} "
                                f"({result_path.name}) in {run_dir}; see {program}.out.")
    values = read_r2_values(result_path, three_d=three_d)
    order = np.load(run_dir / "order.npy")
    if len(values) != len(order):
        raise ValueError(f"{result_path.name} has {len(values)} values for a mesh of "
                         f"{len(order)} elements.")
    rho = np.empty(len(order))
    rho[order] = values[:, 0]
    out_parsed = read_r2_out(run_dir / f"{program}.out")
    sets = out_parsed["datasets"]
    entry = sets[dataset - 1] if len(sets) >= dataset else {"rms": [], "alpha": [], "stop": ""}
    data = np.load(run_dir / f"data_{int(dataset):03d}.npz")
    fitted, std = data["fitted"], data["std"]
    errors = read_r2_errors(run_dir / f"{stem}_err.dat", three_d=three_d)
    # The error file lists the readings in protocol order but can leave some
    # out (see forward_r2), so its rows are matched by their electrodes.
    rows = _match_rows(data["numbers"], errors["electrodes"])
    if np.any(rows < 0):
        raise ValueError(f"{stem}_err.dat lists readings that are not in the protocol "
                         f"written to {run_dir}.")
    normalised = np.full(len(fitted), np.nan)
    normalised[rows] = errors["normalised"]
    unreported = np.flatnonzero(~np.isfinite(normalised))
    # The prediction, from the normalised misfit the program reports for what it
    # fitted: (log f - log d) / (sd / |d|) for log data, (f - d) / sd otherwise.
    # Unlike the apparent resistivities beside it this needs no geometric
    # factor, and it stays exact where a difference d - d0 is close to zero.
    with np.errstate(over="ignore", invalid="ignore"):
        if bool(info.get("log_data", True)):
            predicted = fitted * np.exp(normalised * std / np.abs(fitted))
        else:
            predicted = fitted + normalised * std
    if "offset" in data:
        predicted = predicted + data["offset"]   # R2's difference data: f(m0) back on
    measured = data["measured"] if "measured" in data else fitted
    rms = [float(v) for v in entry["rms"]]
    out: Dict[str, Any] = {
        "run_dir": str(run_dir), "program": program, "dataset": int(dataset),
        "element_resistivity": rho, "predicted": predicted, "observed": measured,
        # Over the readings the program reported; see complete_prediction.
        "chi2": float(np.mean(errors["normalised"] ** 2)),
        "unreported": unreported.tolist(),
        "convergence": [v * v for v in rms], "stop": entry["stop"] or "plateau",
        "message": entry.get("message", ""),
        "alpha": float(entry["alpha"][-1]) if entry["alpha"] else float("nan"),
        "iterations": max(len(rms) - 1, 0)}
    if "k" in data:
        out["response"] = predicted * data["k"]
    cells = run_dir / "cell_elements.npy"
    if cells.is_file():
        out["model"] = rho[np.load(cells)]
    sensitivity = run_dir / f"{stem}_sen.dat"
    out["coverage"] = None
    if sensitivity.is_file() and sensitivity.stat().st_size > 0 and cells.is_file():
        table = read_r2_values(sensitivity, three_d=three_d)
        if len(table) == len(order):
            log_sens = np.empty(len(order))
            log_sens[order] = table[:, -1]
            log_sens[log_sens < -90] = np.nan          # elements held fixed
            out["coverage"] = log_sens[np.load(cells)]
    section = run_dir / "section.bms"
    if section.is_file():
        from PyHydroGeophysX.core.mesh_serialization import read_bms

        out["result_mesh"] = read_bms(section)
    for name in (f"{stem}_res.vtk", f"{stem}.vtk"):
        if (run_dir / name).is_file():
            out["vtk"] = str(run_dir / name)
            break
    return out


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
_FACES = {(2, 3): ((0, 1), (1, 2), (2, 0)), (2, 4): ((0, 1), (1, 2), (2, 3), (3, 0)),
          (3, 4): ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))}


def _neighbour_pairs(cells: np.ndarray, dim: int) -> np.ndarray:
    """Pairs of elements sharing an edge (2-D) or a face (3-D)."""
    faces = _FACES[(dim, cells.shape[1])]
    keys = np.sort(np.concatenate([cells[:, list(face)] for face in faces]), axis=1)
    owner = np.tile(np.arange(len(cells)), len(faces))
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    order = np.argsort(inverse, kind="stable")
    shared = counts[inverse[order]] == 2
    return owner[order][shared].reshape(-1, 2)


def _connect_zones(zone: np.ndarray, pairs: np.ndarray, anchored: np.ndarray) -> np.ndarray:
    """Zones in one piece, as R2 and R3t require: a piece of a zone that does
    not hold any of its ``anchored`` elements joins the zone it borders most."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    zone = zone.copy()
    n = len(zone)
    for _ in range(20):
        same = zone[pairs[:, 0]] == zone[pairs[:, 1]]
        link = pairs[same]
        graph = coo_matrix((np.ones(len(link)), (link[:, 0], link[:, 1])), shape=(n, n))
        _, piece = connected_components(graph, directed=False)
        kept = set(np.unique(piece[anchored]).tolist())
        stray = np.flatnonzero(~np.isin(piece, list(kept)))
        if not stray.size:
            break
        across = pairs[~same]
        moved = False
        for label in np.unique(piece[stray]):
            members = piece == label
            touch = np.concatenate([zone[across[members[across[:, 0]], 1]],
                                    zone[across[members[across[:, 1]], 0]]])
            if touch.size:
                zone[members] = np.bincount(touch).argmax()
                moved = True
        if not moved:
            break
    return zone


def _orient(nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Cells with counter-clockwise (2-D) or positive-volume (3-D) node order."""
    cells = cells.copy()
    a = nodes[cells[:, 0]]
    if nodes.shape[1] == 2:
        # The shoelace sum works for triangles and quadrilaterals alike.
        x, z = nodes[cells][..., 0], nodes[cells][..., 1]
        area = np.sum(x * np.roll(z, -1, axis=1) - np.roll(x, -1, axis=1) * z, axis=1)
        flip = area < 0
        cells[flip] = cells[flip][:, ::-1]
    else:
        b, c, d = nodes[cells[:, 1]], nodes[cells[:, 2]], nodes[cells[:, 3]]
        volume = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
        flip = volume < 0
        cells[flip] = cells[flip][:, [0, 2, 1, 3]]
    return cells


class R2Engine:
    """R2 or R3t (``program``) behind the common ERT engine contract (see the
    module docstring).

    ``options`` are :class:`R2Settings`. The result mesh is the parameter
    domain (cells marked 2 and above) of a profile's mesh or of an imported
    3-D mesh; for R3t on a profile, the section of its 3-D mesh along the line
    is read onto it. Each run folder also holds the program's own VTK model.
    """

    holds_fixed_zones = True

    def __init__(self, container, mesh, *, program: str = "r2", model_constraints=(1e-2, 1e5),
                 method=None, zones=None, log: LogFn = _noop_log, options: Any = None,
                 outer_width: float = 0.0):
        import pygimli as pg

        from .ert_mesh import _interface_regions
        from .ert_zones import zone_prior

        self.name = _engine_name(program)
        self.program = ENGINES[self.name]
        self.settings = R2Settings.from_value(options)
        self._log = log
        self.workdir = Path(self.settings.workdir or tempfile.mkdtemp(prefix=f"{self.name}_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.launcher = find_r2(self.name, self.settings)
        log("  " + self.launcher.describe())
        self._counter = [0]
        self._bounds = tuple(float(v) for v in model_constraints)
        self._last: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self._take_data(container)

        dim = int(mesh.dim())
        if self.name == "r2" and dim != 2:
            raise ValueError("R2 inverts 2-D profiles; choose R3t for a 3-D survey.")
        markers = np.asarray(mesh.cellMarkers(), dtype=int)
        active = np.flatnonzero(markers > 1)
        if not active.size:
            raise ValueError("The mesh has no cells marked 2 or above to invert.")
        self.mesh = mesh.createMeshByCellIdx(pg.IVector(active.tolist()))
        self.profile = dim == 2
        self.three_d = self.name == "r3t"
        self.zone_prior = None
        units = None
        if self.profile:
            if zones:
                self.zone_prior = zone_prior(self.mesh, zones, bounds=model_constraints)
            units = _interface_regions(mesh, active.tolist())
        elif zones:
            log("  Note: a-priori zones are drawn on 2-D sections and do not apply to a 3-D "
                f"mesh; {self.program} starts from a uniform model.")
        sensors = np.asarray(container.sensorPositions(), dtype=float)
        n_cells = int(self.mesh.cellCount())

        if self.name == "r2" or not self.profile:
            # The program inverts this very mesh.
            forward = mesh
            self._dim = dim
            nodes = np.asarray(forward.positions(), dtype=float)[:, :dim]
            electrodes = sensors[:, :dim]
            self._cell_of_element = np.full(int(forward.cellCount()), -1, dtype=np.int64)
            self._cell_of_element[active] = np.arange(len(active))
            self._element_of_cell = active.astype(np.int64)
            centres = np.asarray(forward.cellCenters(), dtype=float)[:, :dim]
        else:
            from .e4d import _find_cells, _inside_2d, _profile_mesh

            forward, electrodes = _profile_mesh(container, self.mesh, self.settings,
                                                self.workdir, outer_width=float(outer_width),
                                                log=log, label="R3t")
            log(f"  Note: R3t inverts this profile in 3-D, on {forward.cellCount()} elements "
                "around the line: far slower than R2, and with the ground off the line held "
                "only by the smoothness. R2 inverts a profile in 2-D.")
            self._dim = 3
            nodes = np.asarray(forward.positions(), dtype=float)
            centres = np.asarray(forward.cellCenters(), dtype=float)
            section = np.asarray(self.mesh.cellCenters(), dtype=float)
            self._element_of_cell = _find_cells(
                forward, np.column_stack([section[:, 0], np.zeros(n_cells), section[:, 1]]), 3)
            self._cell_of_element = _inside_2d(self.mesh, centres[:, [0, 2]])
        cells = np.asarray([cell.ids() for cell in forward.cells()], dtype=np.int64)
        if len({len(c) for c in cells}) != 1 or cells.shape[1] not in (
                (3, 4) if self._dim == 2 else (4,)):
            raise ValueError(f"{self.program} needs a mesh of "
                             + ("triangles or quadrilaterals." if self._dim == 2
                                else "tetrahedra."))
        self.forward_mesh = forward
        self._centres = centres

        # Zones the smoothness is not applied across: each part the zone
        # outlines cut the section into, and the outer elements with the part
        # nearest them, every zone in one piece.
        zone = np.ones(len(cells), dtype=np.int64)
        if units is not None:
            from scipy.spatial import cKDTree

            inside = self._cell_of_element >= 0
            zone[inside] = 1 + units[self._cell_of_element[inside]]
            plane = (lambda xyz: xyz[:, [0, 2]]) if self._dim == 3 else (lambda xy: xy[:, :2])
            section = np.asarray(self.mesh.cellCenters(), dtype=float)[:, :2]
            if np.any(~inside):
                _, nearest = cKDTree(section).query(plane(centres[~inside]))
                zone[~inside] = 1 + units[nearest]
            zone = _connect_zones(zone, _neighbour_pairs(cells, self._dim), inside)
            _, zone = np.unique(zone, return_inverse=True)
            zone = zone.astype(np.int64) + 1
            log(f"  {self.program}: smoothness cut along the zone outlines "
                f"({int(zone.max())} zones)")
        self._zone = zone

        # Elements held at their starting value come last, as R2 requires.
        fixed = np.zeros(len(cells), dtype=bool)
        if self.zone_prior is not None and self.zone_prior.any_fixed:
            inside = self._cell_of_element >= 0
            fixed[inside] = self.zone_prior.fixed[self._cell_of_element[inside]]
            log(f"  {self.program}: {int(fixed.sum())} elements of the fixed zone(s) are held "
                "at their value (param 0)")
        self._fixed = fixed
        self._order = np.concatenate([np.flatnonzero(~fixed), np.flatnonzero(fixed)])
        params = np.zeros(len(cells), dtype=np.int64)
        params[: int((~fixed).sum())] = np.arange(1, int((~fixed).sum()) + 1)

        # Coordinates relative to the electrodes, which the programs' single
        # precision needs for survey coordinates; the vertical is kept.
        shift = np.zeros(nodes.shape[1])
        shift[0] = electrodes[:, 0].mean()
        if self._dim == 3:
            shift[1] = electrodes[:, 1].mean()
        self._shift = shift
        local = nodes - shift
        self._local_centres = centres - shift
        ordered = _orient(local, cells)[self._order]
        self._electrode_nodes, self._remote = self._place_electrodes(nodes, electrodes)
        far = self._far_node(nodes, electrodes)
        mesh_dir = self.workdir / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = mesh_dir / MESH_FILE[self.name]
        if self._dim == 2:
            write_r2_mesh(mesh_path, local, ordered, params, zone[self._order], dirichlet=far)
        else:
            write_r3t_mesh(mesh_path, local, ordered, params, zone[self._order], dirichlet=far,
                           datum=float(np.max(electrodes[:, -1])))
        self._mesh_path = mesh_path
        self._quadrilateral = self._dim == 2 and cells.shape[1] == 4
        self._z_range = (float(nodes[:, -1].min()) - 1.0, float(nodes[:, -1].max()) + 1.0)
        log(f"  {self.program} mesh: {len(cells)} elements, {len(nodes)} nodes"
            + (f", {int(zone.max())} zones" if zone.max() > 1 else ""))

    def _take_data(self, container) -> None:
        """The transfer resistances the program fits, and their deviations."""
        from .e4d import _survey_data

        self.container = container
        self._resistance, self._std, self._k = _survey_data(container, program=self.program)
        self._abmn = np.column_stack([np.asarray(container[key], dtype=np.int64)
                                      for key in ("a", "b", "m", "n")])
        rhoa = np.asarray(container["rhoa"], dtype=float)
        good = rhoa[np.isfinite(rhoa) & (rhoa > 0)]
        self._background = float(np.median(good)) if good.size else 100.0

    def with_data(self, container) -> "R2Engine":
        """This engine on other data from the same survey - what is left after
        an outlier pass - on the same mesh."""
        import copy

        twin = copy.copy(self)
        twin._take_data(container)
        twin._last = None
        return twin

    def _place_electrodes(self, nodes: np.ndarray, electrodes: np.ndarray):
        """The 1-based mesh node of every electrode, and of the remote one
        that stands for an index of -1 (0 when there is none)."""
        from scipy.spatial import cKDTree

        distance, index = cKDTree(nodes).query(electrodes)
        if len(set(index.tolist())) < len(index):
            raise ValueError(f"Two electrodes fall on one mesh node; {self.program} needs a "
                             "node for each electrode.")
        unique = np.unique(np.round(electrodes[:, 0], 6))
        spacing = float(np.median(np.diff(unique))) if len(unique) > 1 else 1.0
        if float(distance.max()) > 0.05 * spacing:
            self._log(f"  Note: an electrode is {float(distance.max()):.3g} m from the nearest "
                      f"mesh node, where {self.program} places it.")
        remote = 0
        if np.any(self._abmn < 0):
            # A remote electrode is put on the node farthest from the array.
            gap = cKDTree(electrodes).query(nodes)[0]
            remote = int(np.argmax(gap)) + 1
            self._log(f"  {self.program}: the remote electrode is placed {float(gap.max()):.0f} m "
                      "from the array, on the mesh's farthest node.")
        return [int(i) + 1 for i in index], remote

    def _far_node(self, nodes: np.ndarray, electrodes: np.ndarray) -> int:
        """The fixed-potential node: the one farthest from every electrode,
        the remote one included."""
        from scipy.spatial import cKDTree

        points = electrodes if not self._remote else np.vstack([electrodes,
                                                                nodes[self._remote - 1]])
        gap = cKDTree(points).query(nodes)[0]
        if self._remote:
            gap[self._remote - 1] = -1.0
        return int(np.argmax(gap)) + 1

    def _numbers(self, abmn: np.ndarray) -> np.ndarray:
        """Electrode numbers as the programs read them; -1 is the remote one."""
        numbers = abmn + 1
        numbers[abmn < 0] = len(self._electrode_nodes) + 1
        return numbers

    def _element_rho(self, model) -> np.ndarray:
        """Element resistivities for a model on the result mesh, or the
        starting model when ``model`` is None. The model this engine last
        returned continues with its whole mesh, the outer region included."""
        if model is not None and self._last is not None:
            last_model, last_elements, _ = self._last
            candidate = np.asarray(model, dtype=float).reshape(-1)
            if candidate.shape == last_model.shape and np.allclose(candidate, last_model):
                return last_elements.copy()
        rho = np.full(len(self._cell_of_element), self._background)
        if model is None and self.zone_prior is not None:
            model = self.zone_prior.with_background(self._background)
        if model is not None:
            model = np.asarray(model, dtype=float).reshape(-1)
            inside = self._cell_of_element >= 0
            rho[inside] = model[self._cell_of_element[inside]]
        low, high = self._bounds
        return np.clip(rho, low, high)

    def _prepare(self, datasets: Sequence[Dict[str, Any]], *, start: np.ndarray,
                 tolerance: float, max_iterations: int, reg_mode: int = 0) -> Path:
        """A run folder holding everything the program reads."""
        from PyHydroGeophysX.core.mesh_serialization import via_ascii_path

        self._counter[0] += 1
        self._run = self._counter[0]
        run_dir = self.workdir / f"run{self._run:02d}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        target = run_dir / self._mesh_path.name
        try:
            os.link(self._mesh_path, target)
        except OSError:
            shutil.copyfile(self._mesh_path, target)
        write_r2_protocol(run_dir / PROTOCOL_FILE, [
            {"abmn": self._numbers(d["abmn"]), "data": d["data"], "std": d["std"],
             "reference": d.get("reference")} for d in datasets], three_d=self.three_d)
        write_r2_start(run_dir / START_FILE, self._local_centres[self._order],
                       start[self._order])
        nodes = list(self._electrode_nodes) + ([self._remote] if self._remote else [])
        if self.three_d:
            write_r3t_in(run_dir, electrode_nodes=nodes, tolerance=tolerance,
                         max_iterations=max_iterations, z_range=self._z_range,
                         reg_mode=reg_mode)
        else:
            write_r2_in(run_dir, electrode_nodes=nodes, tolerance=tolerance,
                        max_iterations=max_iterations, reg_mode=reg_mode,
                        quadrilateral=self._quadrilateral)
        for number, dataset in enumerate(datasets, start=1):
            # ``fitted`` is what the program reports a misfit for, ``offset``
            # what turns its prediction of that into the survey's own, and
            # ``measured`` the survey's resistances themselves.
            extra = {key: np.asarray(dataset[key]) for key in ("offset", "measured")
                     if dataset.get(key) is not None}
            np.savez(run_dir / f"data_{number:03d}.npz", fitted=np.asarray(dataset["fitted"]),
                     std=np.asarray(dataset["std"]), k=np.asarray(dataset["k"]),
                     numbers=self._numbers(dataset["abmn"]), **extra)
        np.save(run_dir / "order.npy", self._order)
        np.save(run_dir / "cell_elements.npy", self._element_of_cell)
        via_ascii_path(self.mesh.save, run_dir / "section.bms", mode="write")
        info = {"written_by": "PyHydroGeophysX", "engine": self.name, "program": self.program,
                "three_d": bool(self.three_d), "profile": bool(self.profile),
                "elements": int(len(self._order)), "datasets": len(datasets),
                "reg_mode": int(reg_mode),
                # R2's difference inversion fits the data themselves, not logs.
                "log_data": not (reg_mode == 2 and not self.three_d),
                "tolerance": float(tolerance), "max_iterations": int(max_iterations),
                "electrode_nodes": [int(node) for node in nodes],
                "quadrilateral": bool(self._quadrilateral),
                "command": self.launcher.shown}
        (run_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return run_dir

    def _not_run(self, run_dir: Path) -> R2NotRun:
        return R2NotRun(f"{self.launcher.reason} The {self.program} run folder is {run_dir}: "
                        f"run {self.program}.exe in it, then read the result with "
                        "PyHydroGeophysX.inversion.r2.read_r2_run.", run_dir)

    def _read(self, run_dir: Path, dataset: int = 1) -> Dict[str, Any]:
        """A dataset of a run read back, with the prediction of every reading:
        those the program left out of its error file come from a forward run
        on the model it found."""
        result = read_r2_run(run_dir, dataset)
        missing = result["unreported"]
        if missing:
            result = complete_prediction(
                result, forward_r2(run_dir, self.launcher, dataset,
                                   timeout=float(self.settings.timeout)))
            result["unreported_filled"] = len(missing)
            self._log(f"    {self.program} left {len(missing)} reading(s) out of its fit and "
                      "its error file (no apparent resistivity over uniform ground, with M "
                      "and N symmetric about A and B, or a polarity it could not fit): chi2 "
                      f"is over the {len(result['predicted']) - len(missing)} it fitted "
                      f"({result['chi2_all']:.2f} with them); their predictions come from "
                      "a forward run on its model")
        return result

    @staticmethod
    def _tolerance(target_chi2: float) -> float:
        # The programs stop at an RMS misfit; chi2 is its square.
        return math.sqrt(max(float(target_chi2), 1e-4))

    # -- the contract -----------------------------------------------------
    def reference_model(self):
        """R2's and R3t's smoothness acts on the model itself; there is no
        reference to pin."""
        return None

    def fit(self, *, lam, max_iterations, plateau_tolerance, target_chi2,
            start_model=None, reference_model=None):
        from .ert_inversion import ERTRun

        start = self._element_rho(start_model)
        run_dir = self._prepare(
            [{"abmn": self._abmn, "data": self._resistance, "std": self._std,
              "fitted": self._resistance, "k": self._k}],
            start=start, tolerance=self._tolerance(target_chi2),
            max_iterations=max(int(max_iterations), 1))
        if not self.launcher.runs:
            raise self._not_run(run_dir)
        self._log(f"    {self.program} run {self._run} in {run_dir.name} (it chooses its own "
                  "smoothing weight at every iteration)")
        outcome = run_r2(run_dir, self.launcher, timeout=float(self.settings.timeout),
                         log=self._log)
        result = self._read(run_dir)
        stop = {"failed": "stalled"}.get(result["stop"], result["stop"])
        model = np.asarray(result["model"], dtype=float)
        self._last = (model, result["element_resistivity"], result["predicted"])
        history = list(result["convergence"]) or [result["chi2"]]
        return ERTRun(
            lam=float(lam), chi2=float(result["chi2"]), iterations=int(result["iterations"]),
            stop=stop, convergence=history, model=model,
            response=np.asarray(result["response"], dtype=float), mesh=self.mesh,
            coverage=result["coverage"],
            metrics={"backend": self.name, "chi2": float(result["chi2"]),
                     "iterations": int(result["iterations"]),
                     "n_data": int(len(self._resistance)),
                     f"{self.name}_alpha": float(result["alpha"]),
                     f"{self.name}_run": str(run_dir),
                     f"{self.name}_command": self.launcher.shown,
                     f"{self.name}_seconds": float(outcome["seconds"]),
                     f"{self.name}_elements": int(len(self._order)),
                     f"{self.name}_stop_message": result["message"],
                     # Readings left out of the error file, predicted by a forward run.
                     f"{self.name}_unreported": int(result.get("unreported_filled", 0)),
                     "vtk_3d": result.get("vtk", "") if self.three_d else "",
                     "chi2_definition": f"{self.program}: RMS misfit squared, the mean of "
                                        "((log R_pred - log R_obs) / err)^2"})

    def fit_time_lapse(self, steps: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray]], *,
                       max_iterations: int, target_chi2: float) -> Dict[str, Any]:
        """The difference inversion of this engine's survey as the baseline
        and ``steps`` after it, each ``(rows, resistance, std, k)``: ``rows``
        the baseline readings it repeats, then that survey's values for them.
        Returns ``runs`` (one :func:`read_r2_run` result per survey, the
        baseline first) and ``run_dirs``."""
        baseline = self.fit(lam=0.0, max_iterations=max_iterations, plateau_tolerance=0.0,
                            target_chi2=target_chi2)
        base_dir = Path(baseline.metrics[f"{self.name}_run"])
        _, base_elements, base_predicted = self._last
        datasets = []
        for rows, resistance, std, k in steps:
            rows = np.asarray(rows, dtype=np.int64)
            d = np.asarray(resistance, dtype=float)
            d0, f0 = self._resistance[rows], base_predicted[rows]
            dataset = {"abmn": self._abmn[rows], "std": std, "k": k, "measured": d}
            if self.three_d:
                # The difference inversion's data, d - d0 + f(m0), written out.
                dataset.update(data=d - d0 + f0, fitted=d - d0 + f0)
            else:
                # R2 forms that dataset itself from d and d0, and reports its
                # misfit on the difference d - d0 against f(m) - f(m0).
                dataset.update(data=d, reference=d0, fitted=d - d0, offset=f0)
            datasets.append(dataset)
        run_dir = self._prepare(datasets, start=base_elements,
                                tolerance=self._tolerance(target_chi2),
                                max_iterations=max(int(max_iterations), 1), reg_mode=2)
        if not self.launcher.runs:
            raise self._not_run(run_dir)
        self._log(f"    {self.program} difference inversion of {len(steps)} survey(s) against "
                  f"the baseline in {run_dir.name}")
        outcome = run_r2(run_dir, self.launcher, timeout=float(self.settings.timeout),
                         log=self._log)
        runs = [self._read(base_dir)]
        for number in range(1, len(steps) + 1):
            runs.append(self._read(run_dir, number))
        return {"runs": runs, "run_dirs": [str(base_dir), str(run_dir)],
                "seconds": float(baseline.metrics[f"{self.name}_seconds"]) + outcome["seconds"]}


def invert_r2_time_lapse(containers: Sequence[Any], mesh, *, program: str = "r2",
                         max_iterations: int = 20, target_chi2: float = 1.0,
                         model_constraints=(1e-2, 1e5), zones=None, options: Any = None,
                         outer_width: float = 0.0, relative_error: float = 0.03,
                         error_floor: float = 0.005, log: LogFn = _noop_log):
    """A time-lapse series inverted by R2 or R3t, shaped like the in-house
    result.

    The first survey is inverted as the baseline; every later one is a
    difference inversion against it (see the module docstring), using the
    readings it shares with the baseline. Returns an object with
    ``final_models`` (cells x surveys on the parameter domain), ``mesh``,
    ``iteration_chi2`` (each survey's final chi2), ``convergence`` (per
    survey), ``responses`` and ``meta``. A survey without a usable error
    column is given ``relative_error``; no error is taken below
    ``error_floor``.
    """
    import types

    import pygimli as pg

    from .e4d import _survey_data

    name = _engine_name(program)
    exe = ENGINES[name]
    if len(containers) < 2:
        raise ValueError("A time-lapse inversion needs at least two surveys.")
    baseline = pg.DataContainerERT(containers[0])
    resistance, std, _ = _survey_data(baseline, relative_error, error_floor, program=exe)
    baseline["err"] = std / np.abs(resistance)     # the errors the steps are given too
    keys = [tuple(int(v) for v in row) for row in np.column_stack(
        [np.asarray(baseline[key], dtype=np.int64) for key in ("a", "b", "m", "n")])]
    steps = []
    for number, container in enumerate(containers[1:], start=2):
        values, deviation, k = _survey_data(container, relative_error, error_floor, program=exe)
        index = {tuple(int(v) for v in row): i for i, row in enumerate(np.column_stack(
            [np.asarray(container[key], dtype=np.int64) for key in ("a", "b", "m", "n")]))}
        rows = np.asarray([i for i, key in enumerate(keys) if key in index], dtype=np.int64)
        if not rows.size:
            raise ValueError(f"Survey {number} shares no reading (the same A, B, M and N) "
                             f"with the baseline, which a {exe} difference inversion needs.")
        if rows.size < len(keys) or rows.size < len(index):
            log(f"  {exe} time-lapse: survey {number} repeats {rows.size} of the baseline's "
                f"{len(keys)} readings, which are inverted; its other "
                f"{len(index) - rows.size} reading(s) are left out.")
        pick = np.asarray([index[keys[i]] for i in rows])
        steps.append((rows, values[pick], deviation[pick], k[pick]))

    engine = R2Engine(baseline, mesh, program=name, model_constraints=model_constraints,
                      zones=zones, log=log, options=options, outer_width=outer_width)
    out = engine.fit_time_lapse(steps, max_iterations=max_iterations, target_chi2=target_chi2)
    runs = out["runs"]
    models = np.column_stack([np.asarray(run["model"], dtype=float) for run in runs])
    finals = [float(run["chi2"]) for run in runs]
    log(f"  {exe} time-lapse chi2 per survey: " + ", ".join(f"{c:.2f}" for c in finals))
    note = (f"{exe} difference inversion (LaBrecque and Yang, 2001): every survey after the "
            "first starts from the baseline model and is smoothed on its change from it; the "
            "temporal weight alpha and the interval weighting belong to the in-house joint "
            "inversion and are not used.")
    extra = {}
    if all(run.get("coverage") is not None for run in runs):
        # The programs write a sensitivity map only for a dataset they took
        # to its target; the series has one only when every survey did.
        extra["all_coverage"] = np.vstack([run["coverage"] for run in runs])
    return types.SimpleNamespace(
        final_models=models, mesh=engine.mesh, iteration_chi2=finals, all_chi2=finals,
        convergence=[run["convergence"] for run in runs],
        responses=[run.get("response") for run in runs], **extra,
        meta={"backend_version": exe, "linearized_solver": f"{exe} Occam",
              "temporal_weighting": {"mode": f"{name}_difference", "note": note},
              name: {"run_dir": out["run_dirs"][-1], "baseline_run_dir": out["run_dirs"][0],
                     "command": engine.launcher.shown,
                     "readings": [int(len(step[0])) for step in steps],
                     "alpha": [float(run["alpha"]) for run in runs],
                     "stop": [run["stop"] for run in runs],
                     "elements": int(len(engine._order)), "seconds": out["seconds"]}})


# ---------------------------------------------------------------------------
# Command line: where are R2 and R3t?
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Report whether, and how, PyHydroGeophysX can run R2 or R3t. " + _WHERE)
    parser.add_argument("--program", choices=sorted(ENGINES), default="r2")
    parser.add_argument("--launcher", choices=LAUNCHERS, default="auto")
    parser.add_argument("--executable", default="", help="the R2/R3t program")
    parser.add_argument("--wine", default="", help="the Wine program (Linux, macOS)")
    parser.add_argument("--json", action="store_true", help="print one JSON object (Studio)")
    args = parser.parse_args(argv)
    launcher = find_r2(args.program, {"launcher": args.launcher, "executable": args.executable,
                                      "wine": args.wine})
    result = {"runs": launcher.runs, "kind": launcher.kind, "command": launcher.shown,
              "path": launcher.path, "source": launcher.source,
              "message": launcher.describe(), "platform": sys.platform}
    if args.json:
        print(json.dumps({"ok": True, "result": result}))
    else:
        print(result["message"])
        if not launcher.runs:
            print(_WHERE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
