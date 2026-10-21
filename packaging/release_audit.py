"""Inventory, licences, source records and checks for a desktop archive.

Run from the repository root, in the environment that built the archive:

``stage``
    After PyInstaller and the smoke test. Traces every file PyInstaller
    collected (its COLLECT, PYZ and PKG tables of contents) to the pip
    distribution, conda package, Python installation or repository file it
    came from (the Full build's rustsolve files by their SHA-256, from
    ``packaging/rustsolve/manifest.json``), and writes into the staging
    folder whose contents become the archive root: ``BUILD_INFO.txt``
    (build facts and the component
    inventory), ``SOURCE_CODE.md`` and ``LICENSES/``. Also writes a review
    report (not shipped) with every check.
``verify``
    On the final ``.zip``: its root, its licence files, the paths
    ``BUILD_INFO.txt`` names and the links in ``SOURCE_CODE.md``.
``replace``
    On the final ``.zip``: unpacks it, replaces its Qt, PySide6 and Shiboken6
    libraries with separately obtained copies of the same release, alters
    one of them, re-signs on macOS and runs ``--self-test``.

Each exits 1 when a blocking check fails (``stage`` only with ``--strict``).
Network access is needed: the licence of each bundled Qt module and the
notices of the third-party code in it are read from that Qt version's
sources, and source links are checked.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/geohang/PyHydroGeophysX"
APP = "PyHydroGeophysX-Studio"
ROOT_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "BUILD_INFO.txt", "SOURCE_CODE.md")
GNU_TEXTS = {  # file name -> SHA-256 of the text published at gnu.org
    "LGPL-3.0-only.txt": "e3a994d82e644b03a792a930f574002658412f62407f5fee083f2555c5f23118",
    "GPL-3.0-only.txt": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}
LICENCE_NAME = re.compile(r"^(licen[cs]e|copying|notice|copyright|authors|third[-_]?party|patents)"
                          r"([-_. ][^/\\]*)?$", re.I)
CODE_SUFFIX = (".py", ".pyc", ".pyi", ".pyd", ".so", ".dll", ".dylib", ".exe", ".c", ".h", ".cpp")
NATIVE_MAGIC = (b"MZ", b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe",
                b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe")
QT_PACKAGES = ("pyside6", "pyside6-essentials", "pyside6-addons", "shiboken6", "qt6-main", "qt-main")

# ---------------------------------------------------------------- Qt rules

#: Qt modules offered under the GPL v3 (or a commercial licence) only: the
#: "Modules available under GNU General Public License v3" on
#: https://doc.qt.io/qt-6/licensing.html, plus Qt Charts and Qt Data
#: Visualization, whose own documentation gives the same terms. Prefixes of
#: the module name. Never bundled; the per-file licence check below also
#: catches any module this list misses.
QT_GPL_ONLY = ("CanvasPainter", "Coap", "Graphs", "Grpc", "HttpServer", "Lottie", "Mqtt",
               "NetworkAuth", "QmlCompiler", "Quick3D", "QuickTimeline", "VirtualKeyboard",
               "WaylandCompositor", "Charts", "DataVisualization")
#: Qt and PySide6 programs that must not be bundled (development tools).
QT_TOOLS = {"designer", "linguist", "assistant", "lupdate", "lrelease", "lconvert", "uic", "rcc",
            "moc", "qmlls", "qmllint", "qmlformat", "qmlcachegen", "qmltyperegistrar",
            "qmlimportscanner", "qsb", "balsam", "balsamui", "qmlsc", "qmltestrunner", "qtdiag",
            "qtpaths", "qtplugininfo", "windeployqt", "macdeployqt", "androiddeployqt", "qmlscene",
            "qml", "qdbus", "qdbusviewer", "pyside6-designer", "pyside6-uic", "pyside6-rcc",
            "pyside6-lupdate", "shiboken6", "shiboken6-genpyi", "QtWebEngineProcess"}
#: Library name -> folder under <Qt repository>/src.
QT_SRC_DIR = {"Core": "corelib", "Gui": "gui", "Widgets": "widgets", "Network": "network",
              "OpenGL": "opengl", "OpenGLWidgets": "openglwidgets", "Test": "testlib",
              "DBus": "dbus", "PrintSupport": "printsupport", "Concurrent": "concurrent",
              "Xml": "xml", "Sql": "sql", "Qml": "qml", "QmlMeta": "qmlmeta",
              "QmlModels": "qmlmodels", "QmlWorkerScript": "qmlworkerscript", "Quick": "quick",
              "QuickWidgets": "quickwidgets", "Svg": "svg", "SvgWidgets": "svgwidgets"}
QT_REPOS = ("qtbase", "qtdeclarative", "qtsvg", "qtimageformats")
PERMISSIVE_SPDX = re.compile(r"^(BSD|MIT|Apache|Zlib|BSL|Unicode|ISC|CC0|Unlicense|Libpng|IJG|FTL|AFL|"
                             r"curl|X11|Bitstream|OFL|Public|PSF|HPND|blessing)", re.I)

# ------------------------------------------------- excluded and flagged code

#: File or folder names that must never be in an official archive.
EXCLUDED_FILES = {
    "Gmsh": re.compile(r"^(lib)?gmsh([-_.\d][^/]*)?\.(dll|so|dylib|pyd|exe)$|^gmsh(\.py)?$|^gmsh-[\d.]+\.dist-info$",
                       re.I),
    "TetGen": re.compile(r"^(lib)?tet(gen)?([-_.\d][^/]*)?\.(dll|so|dylib|pyd|exe)$|^tetgen(\.exe)?$", re.I),
    "ResIPy": re.compile(r"^resipy$|^resipy-[\d.]+\.dist-info$", re.I),
    "E4D": re.compile(r"^e4d([-_][^/]*)?(\.exe)?$", re.I),
    "R2/R3t": re.compile(r"^c?r(2|3t)(\.exe)?$", re.I),
    "Triangle program": re.compile(r"^(triangle|showme)(\.exe)?$", re.I),
}
#: Byte strings that identify code compiled into a binary, whatever its name:
#: (label, needles, all needed, blocking). Triangle and TetGen are recognised
#: by their authors' bug-report addresses, which their library builds keep.
SIGNATURES = [
    ("Gmsh", (b"Gmsh", b"Geuzaine"), True, True),
    ("TetGen", (b"Hang.Si@wias-berlin.de", b"TetGen version"), False, True),
    ("Triangle", (b"jrs@cs.berkeley.edu", b"Jonathan Richard Shewchuk"), False, False),
    ("UMFPACK (GPL-2.0-or-later)", (b"umfpack_di_numeric", b"umfpack_dl_numeric"), False, True),
    ("CHOLMOD GPL modules (Supernodal, Modify, MatrixOps)",
     (b"cholmod_super_numeric", b"cholmod_updown", b"cholmod_ssmult"), False, True),
    ("x264 (GPL-2.0-or-later)", (b"x264 - core",), True, True),
    ("x265 (GPL-2.0-or-later)", (b"x265 (build",), True, True),
    ("FFmpeg", (b"FFmpeg developers",), True, False),
]
#: Microsoft's C/C++ runtime and Universal CRT files.
MS_RUNTIME = re.compile(r"^(msvcp140.*|vcruntime140.*|concrt140|vcomp140|vccorlib140|vcamp140|"
                        r"ucrtbase|api-ms-win-.*)\.dll$", re.I)

#: Packages whose licence file gives different terms to different files: the
#: terms of each bundled file, by its source path, and phrases the licence
#: file must still contain for that split to hold. A bundled file outside
#: every listed scope fails the check.
LICENCE_SCOPES = {
    "pyinstaller": (
        [(re.compile(r"[/\\](bootloader|PyInstaller[/\\]loader)[/\\]|[/\\]localpycs[/\\]pyi(mod|boot)\d"
                     r"|[/\\]PyHydroGeophysX-Studio(\.exe)?$"),
          "GPL-2.0-or-later WITH Bootloader-exception (bootloader and loader)"),
         (re.compile(r"[/\\]PyInstaller[/\\](hooks[/\\]rthooks|fake-modules)[/\\]"),
          "Apache-2.0 (run-time hooks and modules)"),
         (re.compile(r"(^|[/\\])(Info\.plist|PkgInfo|CodeResources)$"),
          "no third-party code (bundle metadata written by PyInstaller)")],
        ("Bootloader Exception", "./PyInstaller/loader", "./PyInstaller/hooks/rthooks", "./PyInstaller/fake-modules"),
    ),
    "pyinstaller-hooks-contrib": (
        [(re.compile(r"[/\\]_pyinstaller_hooks_contrib[/\\]rthooks[/\\]"), "Apache-2.0 (run-time hooks)")],
        ("_pyinstaller_hooks_contrib/rthooks", "Apache-2.0"),
    ),
}

#: Native libraries a package carries besides its own (a wheel's .libs or
#: .dylibs folder; DLLs inside a conda package's Python folder): file-name
#: pattern; project; words that show a licence text covers it; texts in
#: packaging/licenses/vendored/ to add when the package's own files do not;
#: licence class; where its source is (for LGPL and GPL code).
VENDORED_LIBS = [
    (r"(lib)?proj([-_.]|\d)", "PROJ", ("proj",), (), "permissive", ""),
    (r"(lib)?curl", "curl", ("curl",), ("curl-COPYING",), "permissive", ""),
    (r"(lib)?(jpeg|turbojpeg)", "libjpeg-turbo", ("libjpeg",),
     ("libjpeg-turbo-LICENSE.md", "libjpeg-turbo-README.ijg"), "permissive", ""),
    (r"(lib)?lzma", "XZ Utils (liblzma)", ("liblzma",), ("xz-COPYING", "xz-COPYING.0BSD"), "permissive", ""),
    (r"(lib)?sqlite3", "SQLite", ("sqlite",), ("sqlite-LICENSE.md",), "permissive", ""),
    (r"(lib)?tiff", "LibTIFF", ("libtiff",), ("libtiff-LICENSE.md",), "permissive", ""),
    (r"(lib)?z(lib)?(1|-ng)?[-.]|zlib", "zlib", ("zlib",), ("zlib-LICENSE",), "permissive", ""),
    (r"(lib)?zstd", "Zstandard", ("zstandard", "zstd"), ("zstd-LICENSE",), "permissive", ""),
    (r"(lib)?brotli", "Brotli", ("brotli",), (), "permissive", ""),
    (r"(lib)?lcms2", "Little CMS", ("lcms",), (), "permissive", ""),
    (r"(lib)?xcb", "libxcb", ("xcb",), (), "permissive", ""),
    (r"(lib)?xau", "libXau", ("xau",), (), "permissive", ""),
    (r"(lib)?freetype", "FreeType", ("freetype",), (), "permissive", ""),
    (r"(lib)?harfbuzz", "HarfBuzz", ("harfbuzz",), (), "permissive", ""),
    (r"(lib)?(webp|sharpyuv)", "libwebp", ("webp",), (), "permissive", ""),
    (r"(lib)?openjp2", "OpenJPEG", ("openjpeg",), (), "permissive", ""),
    (r"(lib)?png", "libpng", ("libpng",), (), "permissive", ""),
    (r"(lib)?avif", "libavif", ("avif",), (), "permissive", ""),
    (r"(lib)?quadmath", "libquadmath (GCC)", ("quadmath",), ("gcc-libquadmath-COPYING.LIB",), "LGPL-2.1-or-later",
     "https://ftp.gnu.org/gnu/gcc/"),
    (r"(lib)?(gfortran|gcc_s|gomp|stdc\+\+)", "GCC runtime libraries", ("gfortran", "runtime library exception"),
     ("gcc-COPYING3", "gcc-COPYING.RUNTIME"), "exception", ""),
    (r"(lib)?winpthread", "MinGW-w64 winpthreads", ("winpthread",), ("winpthreads-COPYING",), "permissive", ""),
    (r"(lib)?(scipy_)?openblas", "OpenBLAS", ("openblas",), ("openblas-LICENSE",), "permissive", ""),
    (r"(lib)?(crypto|ssl)", "OpenSSL", ("openssl",), (), "permissive", ""),
    (r"(lib)?boost", "Boost", ("boost software license",), ("boost-LICENSE_1_0.txt",), "permissive", ""),
    (r"(lib)?(amd|camd|ccolamd|colamd|suitesparseconfig)[-_.]", "SuiteSparse (AMD, CAMD, CCOLAMD, COLAMD, config)",
     ("suitesparse",), ("suitesparse-LICENSE.txt", "suitesparse-AMD-License.txt", "suitesparse-CAMD-License.txt",
                        "suitesparse-CCOLAMD-License.txt", "suitesparse-COLAMD-License.txt",
                        "suitesparse-SuiteSparse_config-README.txt"), "permissive", ""),
    (r"(lib)?cholmod[-_.]", "SuiteSparse CHOLMOD", ("cholmod",),
     ("suitesparse-LICENSE.txt", "suitesparse-CHOLMOD-License.txt"),
     "LGPL-2.1-or-later; its Supernodal, Modify and MatrixOps modules GPL-2.0-or-later",
     "https://github.com/DrTimothyAldenDavis/SuiteSparse/releases"),
    (r"(lib)?umfpack[-_.]", "SuiteSparse UMFPACK", ("umfpack",),
     ("suitesparse-LICENSE.txt", "suitesparse-UMFPACK-License.txt"), "GPL-2.0-or-later",
     "https://github.com/DrTimothyAldenDavis/SuiteSparse/releases"),
]
#: Libraries a package builds as its own besides those named like it.
OWN_LIBS = {"pgcore": ("gimli",), "pyside6": ("shiboken", "pyside"), "scipy": ("sf_error_state",),
            "vtk-base": ("vtk",)}

#: Header-only libraries a conda recipe compiles into its package (read from
#: the recipe's host requirements): name -> (project, texts, licence class,
#: source). Their code is in the package's binaries, so their notices are too.
HEADER_ONLY = {
    "pybind11": ("pybind11", ("pybind11-LICENSE",), "permissive", ""),
    "cppy": ("cppy", ("cppy-LICENSE",), "permissive", ""),
    "eigen": ("Eigen", ("eigen-COPYING.MPL2", "eigen-COPYING.README", "eigen-COPYING.BSD", "eigen-COPYING.APACHE",
                        "eigen-COPYING.MINPACK"), "MPL-2.0", "https://gitlab.com/libeigen/eigen/-/archive/{v}/eigen-{v}.tar.gz"),
    "libboost-headers": ("Boost", ("boost-LICENSE_1_0.txt",), "permissive", ""),
    "glad2": ("glad (generated OpenGL loader)", ("glad-LICENSE",), "permissive", ""),
    "libvulkan-headers": ("Vulkan-Headers", ("vulkan-headers-LICENSE.md",), "permissive", ""),
    "utfcpp": ("UTF8-CPP", ("utfcpp-LICENSE",), "permissive", ""),
    "nlohmann_json": ("JSON for Modern C++", ("nlohmann_json-LICENSE.MIT",), "permissive", ""),
    "cli11": ("CLI11", ("cli11-LICENSE",), "permissive", ""),
    "pythran": ("Pythran", ("pythran-LICENSE",), "permissive", ""),
}
VENDORED_DIR = REPO / "packaging" / "licenses" / "vendored"
#: Prebuilt files the Full build puts in place of pgcore's SuiteSparse solvers
#: (packaging/rustsolve/apply.py): their SHA-256, the pgcore build they belong
#: to, the files they make redundant and the components they are listed as.
RUSTSOLVE_MANIFEST = REPO / "packaging" / "rustsolve" / "manifest.json"
#: conda packages that ship no licence file, nor does any package from their
#: feedstock: the upstream texts in packaging/licenses/vendored/ to use.
_GCC_TEXTS = ("gcc-COPYING3", "gcc-COPYING.RUNTIME")
CONDA_LICENCE_TEXTS = {"libsqlite": ("sqlite-LICENSE.md",), "sqlite": ("sqlite-LICENSE.md",),
                       "libgcc": _GCC_TEXTS, "libgcc-ng": _GCC_TEXTS, "libgomp": _GCC_TEXTS,
                       "libgfortran": _GCC_TEXTS, "libgfortran5": _GCC_TEXTS + ("gcc-libquadmath-COPYING.LIB",),
                       "libstdcxx": _GCC_TEXTS, "libstdcxx-ng": _GCC_TEXTS}

#: Custom licences read and found permissive, and what each asks of a
#: redistributor beyond carrying its text in LICENSES/ (written to
#: SOURCE_CODE.md).
KNOWN_LICENCE_REFS = {
    "LicenseRef-Qhull": "Qhull: its original source code may be obtained from http://www.qhull.org "
                        "(condition 4 of its licence, `{notices}COPYING.txt`).",
}

#: Qt plugins built from more than their own source folder.
QT_PLUGIN_ALSO = {"direct2d": ("src/plugins/platforms/windows",)}

DATA_README = REPO / "examples" / "data" / "README.md"


# ===================================================================== model

@dataclass
class Finding:
    level: str      # FAIL (blocks), DECIDE (needs the owner), INFO
    check: str
    detail: str


@dataclass
class Component:
    key: str
    name: str
    version: str = ""
    kind: str = ""              # pip, conda, python, pyinstaller, project, data, rustsolve
    origin: str = ""
    licence: str = ""
    licence_files: list = field(default_factory=list)   # (source path, name in LICENSES/<key>/)
    files: list = field(default_factory=list)           # archive paths
    sources: list = field(default_factory=list)         # the path each file was collected from
    notes: list = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def notice_dir(self):
        return f"LICENSES/{self.key}"

    @property
    def canon(self):
        return re.sub(r"[-_.]+", "-", self.name).lower()


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def safe_key(text):
    return re.sub(r"[^A-Za-z0-9._+-]+", "-", text).strip("-") or "component"


def norm(path):
    return os.path.normcase(os.path.realpath(str(path)))


def iter_files(root):
    """Regular files under ``root``, not following or listing symlinks."""
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(folder) / name
            if not path.is_symlink():
                yield path


LICENCE_ORDER = ("permissive", "weak", "exception", "review", "strong", "unknown")


def classify_licence(text):
    """permissive; weak (LGPL/MPL/EPL: source to be offered); exception (GPL
    with a runtime or linking exception); strong (GPL/AGPL); review; unknown.
    An SPDX choice (``A OR B``) takes its least restrictive option."""
    t = " ".join((text or "").split()).upper()
    if not t or t in {"UNKNOWN", "NONE", "OTHER", "N/A"}:
        return "unknown"
    options = [o.strip(" ()") for o in re.split(r"\s+OR\s+", t)]
    if len(options) > 1 and all(re.fullmatch(r"[A-Z0-9.+-]+( WITH [A-Z0-9.+-]+)?", o) for o in options):
        return min((classify_licence(o) for o in options), key=LICENCE_ORDER.index)
    if re.fullmatch(r"LICENSEREF-PUBLIC-DOMAIN|PUBLIC[- ]DOMAIN|BLESSING", t):
        return "permissive"
    if "AGPL" in t or "AFFERO" in t:
        return "strong"
    stripped = t.replace("LESSER GENERAL PUBLIC", "").replace("LIBRARY GENERAL PUBLIC", "")
    if re.search(r"(?<![L])GPL|GENERAL PUBLIC LICENSE", stripped):
        if re.search(r"EXCEPTION|RUNTIME LIBRARY|BOOTLOADER", t):
            return "exception"
        if re.search(r"LGPL|LESSER", t) and " OR " in t:
            return "weak"
        return "strong"
    if re.search(r"LGPL|LESSER GENERAL|LIBRARY GENERAL|\bMPL\b|MOZILLA|\bEPL\b|ECLIPSE|\bCDDL\b", t):
        return "weak"
    if re.search(r"LICENSEREF|PROPRIETARY|INTEL|MICROSOFT|NVIDIA|COMMERCIAL", t):
        return "review"
    return "permissive"


def spdx_ok(expr):
    """Does a Qt source file's SPDX expression offer LGPL or a permissive licence?"""
    if expr == "(no SPDX header)":
        return True
    terms = [t.strip(" ()") for t in re.split(r"\bOR\b", expr)]
    return any(t.upper().startswith("LGPL") or PERMISSIVE_SPDX.match(t) for t in terms)


def fetch(url, dest=None, timeout=120, tries=3):
    """GET ``url`` into ``dest`` (or return the bytes)."""
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "PyHydroGeophysX-release-audit"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if dest is None:
                    return resp.read()
                with open(dest, "wb") as out:
                    shutil.copyfileobj(resp, out, 1 << 20)
                return dest
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def url_ok(url, timeout=60, tries=2):
    """Does the URL answer? A server that times out gets a second try."""
    if not re.match(r"(https?|ftp)://", url or ""):
        return False
    for attempt in range(tries):
        for method in ("HEAD", "GET"):
            headers = {"User-Agent": "PyHydroGeophysX-release-audit"}
            if method == "GET":
                headers["Range"] = "bytes=0-0"
            try:
                req = urllib.request.Request(url, method=method, headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return resp.status < 400
            except urllib.error.HTTPError as exc:
                if exc.code in (404, 410):
                    return False
                if method == "GET" and exc.code < 500:
                    return False
            except (urllib.error.URLError, OSError):
                pass
        time.sleep(5 * (attempt + 1))
    return False


def is_native(path):
    try:
        with open(path, "rb") as f:
            head = f.read(4)
    except OSError:
        return False
    return any(head.startswith(m) for m in NATIVE_MAGIC)


def symbol_role(path, needles):
    """Whether a binary defines (exports) a signature's symbols or only
    imports them from another library; 'contains' when that cannot be told
    (the signature is a string, or the binary format is not read)."""
    names = {n.decode() for n in needles if re.fullmatch(rb"[A-Za-z_]\w+", n)}
    if not names:
        return "contains"
    try:
        if os.name == "nt":
            import pefile
            pe = pefile.PE(str(path), fast_load=True)
            pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                                                   pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
            exported = {s.name.decode() for s in getattr(pe, "DIRECTORY_ENTRY_EXPORT", None).symbols
                        if s.name} if hasattr(pe, "DIRECTORY_ENTRY_EXPORT") else set()
            if names & exported:
                return "defines it"
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []):
                if names & {i.name.decode() for i in entry.imports if i.name}:
                    return f"links to {entry.dll.decode()}"
        elif sys.platform == "darwin":
            defined = subprocess.run(["nm", "-gU", str(path)], capture_output=True, text=True).stdout
            if any(re.search(rf"\b_?{n}\b", defined) for n in names):
                return "defines it"
            undefined = subprocess.run(["nm", "-u", str(path)], capture_output=True, text=True).stdout
            if any(re.search(rf"\b_?{n}\b", undefined) for n in names):
                return "links to it"
    except Exception:  # noqa: BLE001 - only refines the report
        pass
    return "contains"


def gcc_build(path):
    """(GCC version, who built it) from the compiler notes in a GCC runtime
    library: ('12.2.0', 'Rev4, Built by MSYS2 project; Rev6, ...')."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return "", ""
    found = re.findall(rb"GCC: \(([^)]*)\) (\d+\.\d+\.\d+)", data)
    if not found:
        # Mach-O keeps no compiler note, but its debug paths name the source
        # folder; Homebrew builds in /private/tmp/<formula, @ as A>-<date>-...
        m = re.search(rb"/tmp/(gcc[A-Za-z0-9]*)-\d{8}-\d+-\w+/gcc-(\d+\.\d+\.\d+)/", data)
        if m:
            formula = m.group(1).decode().replace("A", "@", 1)
            return m.group(2).decode(), f"Homebrew (its {formula} build folder)"
        m = re.search(rb"/gcc-(\d+\.\d+\.\d+)/", data)
        return (m.group(1).decode(), "") if m else ("", "")
    versions = sorted({v.decode() for _, v in found})
    builders = sorted({b.decode("latin-1") for b, _ in found})
    return (versions[-1] if versions else ""), "; ".join(builders)


def suitesparse_version(path):
    """SuiteSparse release of the libsuitesparseconfig beside a SuiteSparse
    library, from its SuiteSparse_version(); '' when it cannot be loaded."""
    import ctypes
    folder = Path(path).parent
    lib = next((p for p in folder.iterdir() if re.match(r"(lib)?suitesparseconfig[-_.]", p.name.lower())), None)
    if lib is None:
        return ""
    try:
        if os.name == "nt":
            with os.add_dll_directory(str(folder)):
                handle = ctypes.CDLL(str(lib))
        else:
            handle = ctypes.CDLL(str(lib))
        out = (ctypes.c_int * 3)()
        handle.SuiteSparse_version(out)
        return ".".join(str(v) for v in out)
    except (OSError, AttributeError):
        return ""


def pe_version(path):
    """Windows file version of a DLL or EXE, or ''."""
    if os.name != "nt":
        return ""
    import ctypes
    from ctypes import wintypes
    ver = ctypes.windll.version
    size = ver.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return ""
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(str(path), 0, size, buf):
        return ""
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)):
        return ""
    info = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint32 * 13)).contents
    ms, ls = info[2], info[3]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def project_version():
    text = (REPO / "PyHydroGeophysX" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1)


_DATA_ROWS = None


def data_rows():
    """Rows of the table in examples/data/README.md: paths (relative to
    examples/data), contents, source and terms."""
    global _DATA_ROWS
    if _DATA_ROWS is None:
        _DATA_ROWS = []
        for line in DATA_README.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if line.startswith("| `") and len(cells) >= 4:
                _DATA_ROWS.append({"paths": re.findall(r"`([^`]+)`", cells[0]), "contents": cells[1],
                                   "source": cells[2], "terms": cells[3]})
    return _DATA_ROWS


def data_path_matches(rel, listed):
    if listed.endswith("/"):
        return rel.startswith(listed)
    return fnmatch.fnmatchcase(rel, listed)


def data_row_folder(row):
    """The folder a row's bare file names (acknowledgement.txt) sit in."""
    first = row["paths"][0]
    return first if first.endswith("/") else (first.rsplit("/", 1)[0] + "/" if "/" in first else "")


# ============================================================ the bundle

def load_toc(path):
    return ast.literal_eval(Path(path).read_text(encoding="utf-8"))


def bundle_roots(bundle):
    """Folders a TOC destination may land in (macOS .app or one-folder build)."""
    bundle = Path(bundle)
    if bundle.suffix == ".app":
        return [bundle / "Contents" / d for d in ("MacOS", "Frameworks", "Resources")]
    return [bundle, bundle / "_internal"]


def collected(workpath, bundle, stage):
    """[(archive path, source path)] for every file PyInstaller put in the
    bundle, and for the pure modules embedded in the executable."""
    # Real paths throughout: a macOS .app reaches many files through
    # symlinks (Resources -> Frameworks), and the archive path recorded is
    # the file's own.
    work, stage = Path(workpath), Path(stage).resolve()
    roots = bundle_roots(Path(bundle).resolve())
    entries, missing, seen = [], [], set()
    for dest, src, typecode in load_toc(work / "COLLECT-00.toc")[0]:
        if typecode == "SYMLINK":       # src is the link's target, not a file
            continue
        dest = dest.replace("\\", "/")
        found = next((root / dest for root in roots
                      if (root / dest).is_file() and not (root / dest).is_symlink()), None)
        if found is None:
            found = next((root / dest for root in roots if (root / dest).exists()), None)
        if found is None:
            missing.append(dest)
            continue
        apath = Path(found).resolve().relative_to(stage).as_posix()
        if apath not in seen:           # two table entries may reach one file
            seen.add(apath)
            entries.append((apath, src))
    exe_dir = roots[0]
    exe = next((p for p in exe_dir.iterdir() if p.is_file() and p.name.startswith(APP)), None)
    holder = exe.relative_to(stage).as_posix() if exe else f"{Path(bundle).name}/{APP}"
    # A source of "-" marks a namespace package, which has no file.
    pure = [(f"{holder} (embedded archive: {name})", src)
            for name, src, _t in load_toc(work / "PYZ-00.toc")[1] if src and src != "-"]
    pure += [(f"{holder} (embedded archive: {name})", src)
             for name, src, t in load_toc(work / "PKG-00.toc")[2]
             if t in ("PYMODULE", "PYSOURCE") and src and src != "-"]
    return entries, pure, missing


# ===================================================== where files came from

class Origins:
    """Maps an installed file to the pip distribution, conda package or Python
    installation it belongs to."""

    def __init__(self, conda_prefix=None, pip_report=None, workpath=None):
        self.by_path, self.dists, self.conda, self.pip = {}, {}, {}, {}
        self.workpath = norm(workpath) if workpath else None
        if pip_report and Path(pip_report).is_file():
            report = json.loads(Path(pip_report).read_text(encoding="utf-8"))
            for item in report.get("install", []):
                name = self.canon(item.get("metadata", {}).get("name", ""))
                info = item.get("download_info", {}) or {}
                hashes = (info.get("archive_info") or {}).get("hashes") or {}
                self.pip[name] = {"url": info.get("url", ""), "sha256": hashes.get("sha256", "")}
        for dist in importlib.metadata.distributions():
            key = self.canon(dist.metadata.get("Name") or "")
            if not key or key in self.dists or key == "pyhydrogeophysx":
                continue
            self.dists[key] = dist
            for rel in dist.files or []:
                self.by_path.setdefault(norm(dist.locate_file(rel)), ("pip", key))
        if conda_prefix:
            prefix = Path(conda_prefix)
            for meta_file in sorted((prefix / "conda-meta").glob("*.json")):
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                self.conda[meta["name"]] = meta
                for rel in meta.get("files", []):
                    # Files a conda-installed python package also lists in RECORD
                    # belong to the conda package (its build and channel).
                    self.by_path[norm(prefix / rel)] = ("conda", meta["name"])
        # The rustsolve files by SHA-256: they sit where pgcore's own files
        # were (or beside them), so their path names the wrong package.
        self.rustsolve, self.rustsolve_names = {}, set()
        if RUSTSOLVE_MANIFEST.is_file():
            manifest = json.loads(RUSTSOLVE_MANIFEST.read_text(encoding="utf-8"))
            for plat, spec in manifest["platforms"].items():
                for entry in spec["files"]:
                    self.rustsolve[entry["sha256"]] = (plat, entry["component"])
                    self.rustsolve_names.add(Path(entry["install"]).name.lower())
        self.base = norm(sys.base_prefix)
        self.site = [norm(p) for p in sys.path if p.endswith("site-packages")]
        import PyInstaller
        self.pyinstaller = norm(Path(PyInstaller.__file__).parent)
        self.repo = norm(REPO)

    @staticmethod
    def canon(name):
        return re.sub(r"[-_.]+", "-", name or "").lower()

    def owner(self, src):
        path = norm(src)
        if self.workpath and path.startswith(self.workpath + os.sep):
            # Files PyInstaller wrote: the executable and its loader, or the
            # standard library compiled (base_library.zip, localpycs/struct.pyc).
            name = Path(path).name
            if name == "base_library.zip" or ("localpycs" in Path(path).parts and not name.startswith("pyi")):
                return ("python", "")
            return ("pyinstaller", "")
        if path.startswith(self.repo + os.sep):
            rel = Path(os.path.realpath(str(src))).relative_to(REPO).as_posix()    # keeps the case
            return ("data", rel) if rel.startswith("examples/") else ("project", rel)
        if Path(path).name.lower() in self.rustsolve_names and os.path.isfile(src):
            key = self.rustsolve.get(sha256_of(src))
            if key:
                return ("rustsolve", key)
        if path in self.by_path:
            return self.by_path[path]
        if path.startswith(self.pyinstaller + os.sep):
            return ("pyinstaller", "")
        if path.startswith(self.base + os.sep) and not any(path.startswith(s) for s in self.site):
            return ("python", "")
        return (None, None)


def dist_licence(dist):
    meta = dist.metadata
    if meta.get("License-Expression"):
        return meta.get("License-Expression")
    text = (meta.get("License") or "").strip()
    first = text.splitlines()[0].strip() if text else ""
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or []
                   if c.startswith("License ::")]
    if first and len(first) < 120 and len(text.splitlines()) <= 3:
        return first + (f" ({'; '.join(classifiers)})" if classifiers else "")
    if classifiers:
        return "; ".join(classifiers)
    return first[:120]


def dist_licence_files(dist):
    out = []
    declared = [f.replace("\\", "/") for f in dist.metadata.get_all("License-File") or []]
    for rel in dist.files or []:
        posix = str(rel).replace("\\", "/")
        name = posix.rsplit("/", 1)[-1]
        in_info = ".dist-info/" in posix
        if in_info:
            wanted = "/licenses/" in posix or LICENCE_NAME.match(name) or any(posix.endswith(d) for d in declared)
        else:
            wanted = LICENCE_NAME.match(name) and not name.lower().endswith(CODE_SUFFIX)
        if wanted:
            path = Path(dist.locate_file(rel))
            if path.is_file():
                out.append((path, posix.split(".dist-info/", 1)[-1] if in_info else posix))
    return out


def conda_details(meta):
    """Licence files, recipe source and feedstock commit of a conda package."""
    pkg_dir = Path(meta.get("extracted_package_dir") or "")
    files, source = [], {}
    if not pkg_dir.is_dir():
        return files, source
    lic_root = pkg_dir / "info" / "licenses"
    if lic_root.is_dir():
        files = [(p, p.relative_to(lic_root).as_posix()) for p in sorted(lic_root.rglob("*")) if p.is_file()]
    about = pkg_dir / "info" / "about.json"
    if about.is_file():
        data = json.loads(about.read_text(encoding="utf-8"))
        extra = data.get("extra") or {}
        source["home"] = data.get("home", "")
        source["feedstock"] = extra.get("remote_url") or ""
        source["feedstock_sha"] = extra.get("sha", "")
    recipe = pkg_dir / "info" / "recipe"
    if recipe.is_dir():
        # rattler-build keeps the rendered recipe beside the template;
        # conda-build's meta.yaml there is already rendered.
        text = ""
        for name in ("rendered_recipe.yaml", "meta.yaml", "recipe.yaml"):
            if (recipe / name).is_file():
                text = (recipe / name).read_text(encoding="utf-8", errors="replace")
                break
        block = recipe_source_block(text)
        urls, in_list = [], False
        for line in block.splitlines():      # "url: X", or "url:" with "- X" items below
            m = re.match(r"^\s*-?\s*url:[ \t]*(\S*)", line)
            if m:
                in_list = not m.group(1)
                if m.group(1):
                    urls.append(m.group(1))
            elif in_list and re.match(r"^\s*-\s+\S", line):
                urls.append(line.split("-", 1)[1].strip())
            else:
                in_list = False
        shas = re.findall(r"^\s*-?\s*sha256:\s*([0-9a-fA-F]{64})", block, re.M)
        urls = [u.strip("'\"") for u in urls]
        source["upstream"] = [(u, shas[i] if i < len(shas) else "") for i, u in enumerate(urls)
                              if "{{" not in u and re.match(r"(https?|ftp)://", u)]
        source["patches"] = sorted({p.name for p in recipe.rglob("*") if p.suffix in (".patch", ".diff")})
        host = re.search(r"^(\s*)host:\s*$((?:\n\1\s+-.*|\n\1-.*)+)", text, re.M)
        names = set(re.findall(r"-\s+([A-Za-z0-9_.+-]+)", host.group(2))) if host else set()
        source["header_only"] = [(n, *resolved_version(text, n)) for n in sorted(names & set(HEADER_ONLY))]
    return files, source


def resolved_version(text, name):
    """(version, package URL) a rendered recipe resolved a host requirement
    to: conda-build pins ``- name 1.2.3 build``; rattler-build lists
    ``name:``, ``version:`` and ``url:`` under finalized_dependencies."""
    m = re.search(rf"^\s*-\s+{re.escape(name)}\s+(\d[^\s]*)\s+\S+\s*$", text, re.M)
    if m:
        return m.group(1), ""
    m = re.search(rf"^\s*name:\s*{re.escape(name)}\s*\n(?:.*\n){{0,14}}?\s*version:\s*'?([^\s']+)'?\s*\n"
                  rf"(?:.*\n){{0,3}}?\s*url:\s*(\S+)", text, re.M)
    return (m.group(1), m.group(2)) if m else ("", "")


def recipe_source_block(text):
    """The lines of a recipe's ``source:`` section (its nested lines and
    list items), so the URLs of build dependencies elsewhere are not read."""
    lines, out, indent = text.splitlines(), [], None
    for line in lines:
        stripped = line.lstrip()
        depth = len(line) - len(stripped)
        if indent is None:
            if re.match(r"source:\s*$", stripped):
                indent = depth
            continue
        if not stripped or depth > indent or (depth == indent and stripped.startswith("- ")):
            out.append(line)
            continue
        break
    return "\n".join(out)


# ===================================================== Qt sources and notices

def qt_source(repo, version, cache):
    """<repo>-everywhere-src-<version>.tar.xz from download.qt.io, MD5 checked
    against the published list: (path, url, sha256), or (None, '', '')."""
    mm = ".".join(version.split(".")[:2])
    name = f"{repo}-everywhere-src-{version}.tar.xz"
    dest = Path(cache) / name
    for base in (f"https://download.qt.io/official_releases/qt/{mm}/{version}/submodules/",
                 f"https://download.qt.io/archive/qt/{mm}/{version}/submodules/"):
        url = base + name
        if not url_ok(url):
            continue
        if not dest.is_file():
            fetch(url, dest, timeout=900)
        try:
            sums = fetch(base + "md5sums.txt", timeout=60).decode("utf-8", "replace")
            listed = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(name)), "")
        except RuntimeError:
            listed = ""
        if listed and hashlib.md5(dest.read_bytes()).hexdigest() != listed:
            dest.unlink()
            raise RuntimeError(f"{name}: MD5 does not match the published {listed}")
        return dest, url, sha256_of(dest)
    return None, "", ""


SPDX_LINE = re.compile(rb"SPDX-License-Identifier:\s*([^\r\n*]+)")


def tar_members(path):
    with tarfile.open(path, "r:xz") as tar:
        for member in tar:
            if member.isfile():
                rel = member.name.split("/", 1)[1] if "/" in member.name else member.name
                yield tar, member, rel


def scan_qt_tarball(path, wanted_dirs):
    """SPDX expressions per wanted src/ folder and per plugin folder
    (src/plugins/<type>/<name>), the files whose expression offers neither
    LGPL nor a permissive licence, and every qt_attribution.json."""
    spdx, odd = defaultdict(Counter), defaultdict(list)
    attributions = []
    for tar, member, rel in tar_members(path):
        keys = [d for d in wanted_dirs if rel.startswith(d + "/")]
        parts = rel.split("/")
        if rel.startswith("src/plugins/") and len(parts) > 4:
            keys.append("/".join(parts[:4]))
        low = rel.lower()
        if keys and low.endswith((".cpp", ".h", ".c", ".mm", ".qml", ".js", ".mjs")) and "/doc/" not in low:
            m = SPDX_LINE.search(tar.extractfile(member).read(4096))
            expr = m.group(1).decode("utf-8", "replace").strip() if m else "(no SPDX header)"
            for key in keys:
                spdx[key][expr] += 1
                if not spdx_ok(expr):
                    odd[key].append((rel, expr))
        if rel.startswith("src/") and rel.endswith("qt_attribution.json"):
            attributions.append((rel, tar.extractfile(member).read()))
    return spdx, odd, attributions


def extract_from_tarball(path, wanted):
    """{relative path: bytes} for the wanted files and everything in LICENSES/."""
    out = {}
    for tar, member, rel in tar_members(path):
        if rel in wanted or rel.startswith("LICENSES/"):
            out[rel] = tar.extractfile(member).read()
    return out


def attribution_entries(raw):
    """Entries of a qt_attribution.json. Some hold raw line breaks inside
    strings or are not UTF-8, which Qt's own tooling accepts."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    data = json.loads(text, strict=False)
    return data if isinstance(data, list) else [data]


def plugin_stem(name):
    """qwindows.dll -> windows, qtuiotouchplugin.dll -> tuiotouch,
    libqsvgicon.dylib -> svg, qschannelbackend.dll -> schannel."""
    stem = re.sub(r"\.(dll|dylib|so)$", "", name, flags=re.I)
    stem = re.sub(r"^lib", "", stem).lower()
    stem = re.sub(r"^q", "", stem)
    return re.sub(r"(plugin|backend|style|icon)$", "", stem)


def as_list(value):
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


# ================================================================ the audit

class Audit:
    def __init__(self, args):
        self.args = args
        self.stage = Path(args.stage).resolve()
        self.bundle = Path(args.bundle).resolve()
        self.findings, self.components = [], {}
        self.cache = Path(args.cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.signature_hits = {}
        self.ms_runtime = []
        self.qt = {"modules": {}, "plugins": [], "qml": [], "bindings": [], "sources": [],
                   "version": "", "notices": []}
        self.pyside_tar = None

    def add(self, level, check, detail):
        self.findings.append(Finding(level, check, detail))

    def component(self, key, name, version=""):
        if key not in self.components:
            self.components[key] = Component(key=key, name=name, version=version)
        return self.components[key]

    def qt_components(self):
        return [c for c in self.components.values() if c.canon in QT_PACKAGES]

    # ---------------------------------------------------------------- trace
    def trace(self):
        args = self.args
        entries, pure, missing = collected(args.workpath, self.bundle, self.stage)
        for dest in missing:
            self.add("FAIL", "collected files", f"listed by PyInstaller but not in the bundle: {dest}")
        self.origins = Origins(args.conda_prefix, args.pip_report, args.workpath)
        listed, unattributed = set(), []
        for apath, src in entries + pure:
            listed.add(apath)
            kind, key = self.origins.owner(src)
            if kind is None:
                unattributed.append(f"{apath}  <- {src}")
            else:
                self.attach(kind, key, apath, src)
        # A framework's own Info.plist and code signature belong to the
        # package whose framework it is.
        frameworks = {}
        for comp in self.components.values():
            for apath in comp.files:
                m = re.match(r"(.*?\.framework)/", apath)
                if m:
                    frameworks.setdefault(m.group(1), comp)
        app_meta = re.compile(rf"^{re.escape(self.bundle.name)}/Contents/(Info\.plist|PkgInfo|_CodeSignature/.*)$")
        for path in sorted(iter_files(self.bundle)):
            apath = path.relative_to(self.stage).as_posix()
            if apath in listed:
                continue
            fw = re.match(r"(.*?\.framework)/", apath)
            if path.name == "base_library.zip":
                self.attach("python", "", apath, apath)      # the standard library, compiled
            elif fw and fw.group(1) in frameworks:
                owner = frameworks[fw.group(1)]
                owner.files.append(apath)
                owner.sources.append(apath)
            elif path.name.startswith(APP) or app_meta.match(apath):
                self.attach("pyinstaller", "", apath, apath)
            elif path.suffix in (".icns", ".ico"):
                self.attach("project", "packaging/app icon (made from logo.png)", apath, apath)
            else:
                unattributed.append(f"{apath}  <- not in PyInstaller's tables")
        for line in unattributed:
            self.add("FAIL", "attribution", f"file not traced to a source: {line}")
        self.add("INFO", "attribution", f"{len(listed)} collected entries traced to {len(self.components)} components")

    def attach(self, kind, key, apath, src=None):
        o = self.origins
        if kind == "pip" and key == "pyinstaller":
            kind = "pyinstaller"
        if kind == "pip":
            dist = o.dists[key]
            name = dist.metadata.get("Name")
            comp = self.component(f"{safe_key(name)}-{dist.version}", name, dist.version)
            if not comp.kind:
                comp.kind = "pip"
                comp.licence = dist_licence(dist)
                comp.licence_files = dist_licence_files(dist)
                if not comp.licence_files and comp.canon in QT_PACKAGES:
                    # The macOS PySide6 wheels carry no licence file; their
                    # metadata names the LGPL-3.0 option this archive uses.
                    comp.licence_files = [(REPO / "packaging" / "licenses" / n, n) for n in GNU_TEXTS]
                    comp.notes.append("the wheel ships no licence file; LGPL-3.0 and GPL-3.0 texts from gnu.org")
                rep = o.pip.get(key)
                installer = (dist.read_text("INSTALLER") or "").strip() or "an unknown installer"
                if rep and rep["url"]:
                    comp.origin = rep["url"].rsplit("/", 1)[-1] + \
                        (f" (SHA-256 {rep['sha256']})" if rep["sha256"] else "") + " from " + \
                        rep["url"].split("/")[2]
                    comp.source.update(wheel_url=rep["url"], wheel_sha256=rep["sha256"])
                else:
                    comp.origin = f"installed by {installer}; no download record"
        elif kind == "conda":
            meta = o.conda[key]
            comp = self.component(f"{safe_key(meta['name'])}-{meta['version']}", meta["name"],
                                  f"{meta['version']} (build {meta.get('build', '')})")
            if not comp.kind:
                comp.kind = "conda"
                comp.licence = meta.get("license", "")
                files, source = conda_details(meta)
                if not files:
                    files = self.conda_sibling_licences(meta, source, comp)
                if not files and meta["name"] in CONDA_LICENCE_TEXTS:
                    files = [(VENDORED_DIR / t, t) for t in CONDA_LICENCE_TEXTS[meta["name"]]]
                    files.append((VENDORED_DIR / "SOURCES.txt", "SOURCES.txt"))
                    comp.notes.append("the package ships no licence file; the upstream text is from "
                                      "packaging/licenses/vendored/")
                comp.licence_files = files
                comp.source.update(source)
                channel = (meta.get("channel") or "").rstrip("/").rsplit("/", 1)[-1]
                digest = f"SHA-256 {meta['sha256']}" if meta.get("sha256") else f"MD5 {meta.get('md5', '')}"
                comp.origin = f"{channel}::{meta['name']}-{meta['version']}-{meta.get('build', '')} ({digest})"
                comp.source.update(package_url=meta.get("url", ""))
        elif kind == "python" and "python" in o.conda:
            # A conda environment: the standard library PyInstaller compiled
            # belongs to the conda python package.
            return self.attach("conda", "python", apath, src)
        elif kind == "python":
            version = platform.python_version()
            comp = self.component(f"CPython-{version}", "CPython", version)
            if not comp.kind:
                comp.kind = "python"
                comp.licence = "PSF-2.0, with the third-party terms in its LICENSE.txt"
                comp.origin = f"the Python installation at {sys.base_prefix} ({platform.python_compiler()})"
                for cand in (Path(sys.base_prefix) / "LICENSE.txt", Path(sys.base_prefix) / "LICENSE_PYTHON.txt",
                             Path(sys.base_prefix) / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}"
                             / "LICENSE.txt", Path(sys.base_prefix) / "Lib" / "LICENSE.txt"):
                    if cand.is_file():
                        comp.licence_files = [(cand, "LICENSE.txt")]
                        break
        elif kind == "pyinstaller":
            dist = o.dists.get("pyinstaller")
            version = dist.version if dist else ""
            comp = self.component(f"pyinstaller-{version}", "PyInstaller", version)
            if not comp.kind:
                comp.kind = "pyinstaller"
                comp.licence = dist_licence(dist) if dist else ""
                comp.licence_files = dist_licence_files(dist) if dist else []
                rep = o.pip.get("pyinstaller") or {}
                comp.origin = (rep.get("url", "").rsplit("/", 1)[-1] +
                               (f" (SHA-256 {rep['sha256']})" if rep.get("sha256") else "")) or "build tool"
                comp.notes.append("its bootloader starts the program; the rest are generated start-up files")
        elif kind == "project":
            version = project_version()
            comp = self.component(f"PyHydroGeophysX-{version}", "PyHydroGeophysX", version)
            if not comp.kind:
                comp.kind = "project"
                comp.licence = "Apache-2.0"
                comp.origin = f"{REPO_URL} at commit {self.args.commit}"
        elif kind == "data":
            # One component per row of examples/data/README.md, which records
            # each dataset's source and terms.
            rel = key[len("examples/data/"):]
            row = next((r for r in data_rows() if any(data_path_matches(rel, t) for t in r["paths"])), None)
            if row is None:
                comp = self.component("example-data-unlisted", "Example data not listed in examples/data/README.md")
                comp.kind = "data"
                comp.notes.append(rel)
            else:
                comp = self.component(f"example-data-{safe_key(row['paths'][0])}", f"Example data: {row['contents']}")
                if not comp.kind:
                    comp.kind = "data"
                    comp.source["row"] = row
                    comp.origin = (f"{row['source']}; {REPO_URL}/tree/{self.args.commit}/examples/data "
                                   f"(provenance: examples/data/README.md)")
                    own = re.match(r"repository licen[cs]e", row["terms"], re.I)
                    comp.licence = "Apache-2.0 (the repository licence, LICENSE at the archive root)" if own \
                        else row["terms"].replace("`", "")
                    comp.licence_files = [(DATA_README, "README.md")]
                    for name in re.findall(r"`([^`]+)`", row["terms"]):
                        path = REPO / "examples" / "data" / (name if "/" in name else
                                                             f"{data_row_folder(row)}{name}")
                        if path.is_file():
                            comp.licence_files.append((path, path.name))
                        else:
                            comp.source.setdefault("missing", []).append(path.relative_to(REPO).as_posix())
        elif kind == "rustsolve":
            plat, name = key
            spec = json.loads(RUSTSOLVE_MANIFEST.read_text(encoding="utf-8"))["platforms"][plat]["components"][name]
            comp = self.component(spec["key"], spec["name"], spec["version"])
            if not comp.kind:
                comp.kind = "rustsolve"
                comp.licence = spec["licence"]
                comp.origin = spec["origin"]
                comp.notes += spec.get("notes", [])
                # Notices are named relative to packaging/; a folder stands for
                # every file in it. In LICENSES/<component>/ they keep their
                # path below packaging/rustsolve/licenses/ or packaging/licenses/.
                pack = REPO / "packaging"
                for entry in spec["notices"]:
                    path = pack / entry
                    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
                    if not files or not all(p.is_file() for p in files):
                        self.add("FAIL", "licence files", f"{comp.name}: packaging/{entry} is missing")
                    for p in files:
                        if not p.is_file():
                            continue
                        base = pack / "rustsolve" / "licenses"
                        label = p.relative_to(base if base in p.parents else pack / "licenses").as_posix()
                        comp.licence_files.append((p, label))
        else:
            raise ValueError(kind)
        comp.files.append(apath)
        comp.sources.append(str(src or apath))

    def conda_sibling_licences(self, meta, source, comp):
        """Licence files of another package built by the same conda-forge
        feedstock (libfreetype6 ships none; its sibling freetype does)."""
        feedstock = source.get("feedstock")
        if not feedstock:
            return []
        for other in self.origins.conda.values():
            if other["name"] == meta["name"]:
                continue
            files, other_source = conda_details(other)
            if files and other_source.get("feedstock") == feedstock:
                comp.notes.append(f"the package ships no licence file; these are from {other['name']} "
                                  f"{other['version']}, built by the same feedstock")
                return files
        return []

    # -------------------------------------------------------------- licences
    def licence_checks(self):
        for comp in self.components.values():
            if comp.kind == "data":
                self.data_check(comp)
                continue
            if comp.canon in LICENCE_SCOPES:
                self.scope_check(comp)
            cls = classify_licence(comp.licence)
            if comp.licence in KNOWN_LICENCE_REFS:
                cls = "permissive"
                comp.source["source_note"] = KNOWN_LICENCE_REFS[comp.licence].format(notices=comp.notice_dir + "/")
            comp.source["class"] = cls
            if comp.kind == "project":
                continue
            if not comp.licence_files:
                self.add("FAIL", "licence files", f"{comp.name} {comp.version}: no licence or notice file found")
            if cls == "strong":
                self.add("FAIL", "copyleft", f"{comp.name} {comp.version}: '{comp.licence}' is GPL or AGPL "
                                             f"with no LGPL option or exception")
            elif cls == "unknown":
                self.add("FAIL", "licence", f"{comp.name} {comp.version}: licence not declared")
            elif cls == "review" and all(MS_RUNTIME.match(f.rsplit("/", 1)[-1]) for f in comp.files):
                pass    # Microsoft's runtime: one decision for all of it, under "Microsoft runtime"
            elif cls == "review":
                self.add("DECIDE", "licence terms", f"{comp.name} {comp.version}: '{comp.licence}'")
            elif cls == "exception":
                self.add("INFO", "licence", f"{comp.name} {comp.version}: '{comp.licence}' (GPL with an exception)")
            if comp.kind == "conda" or (comp.kind == "pip" and comp.canon not in QT_PACKAGES):
                self.vendored_check(comp)
        self.embedded_checks()

    def data_check(self, comp):
        comp.source["class"] = "data"
        if "row" not in comp.source:
            self.add("FAIL", "example data", f"not listed in examples/data/README.md, so its source and terms are "
                                             f"not recorded: {', '.join(comp.notes[:6])}")
            return
        for path in comp.source.get("missing", []):
            self.add("FAIL", "example data", f"{comp.name}: examples/data/README.md names {path}, which does not exist")
        self.add("INFO", "example data", f"{comp.name} ({len(comp.files)} files): {comp.origin.split(';')[0]}; "
                                         f"{comp.licence}")

    def scope_check(self, comp):
        """Terms per bundled file for packages whose licence splits them."""
        scopes, phrases = LICENCE_SCOPES[comp.canon]
        text = " ".join(Path(p).read_text(encoding="utf-8", errors="replace") for p, _ in comp.licence_files)
        for phrase in phrases:
            if phrase not in text:
                self.add("FAIL", "licence scope", f"{comp.name} {comp.version}: its licence file no longer says "
                                                  f"'{phrase}'; the terms of its bundled files need a new reading")
        used, outside = Counter(), []
        for apath, src in zip(comp.files, comp.sources):
            label = next((lab for pattern, lab in scopes if pattern.search(src)), None)
            if label:
                used[label] += 1
            else:
                outside.append(f"{apath} <- {src}")
        for line in outside:
            self.add("FAIL", "licence scope", f"{comp.name} {comp.version}: bundled file outside the scopes its "
                                              f"licence names: {line}")
        comp.licence = "; ".join(f"{lab}: {n} files" for lab, n in sorted(used.items()))

    def vendored_check(self, comp):
        """Native libraries a package carries besides its own: each must be a
        known project whose licence text the package includes or
        packaging/licenses/vendored/ supplies. For conda packages, also the
        header-only libraries their recipe compiled in."""
        text = " ".join(Path(p).read_text(encoding="utf-8", errors="replace").lower()
                        for p, _ in comp.licence_files)
        own = set(re.findall(r"[a-z]{4,}", comp.canon)) | set(OWN_LIBS.get(comp.canon, ()))
        added = set()
        for apath, src in zip(comp.files, comp.sources):
            name = apath.rsplit("/", 1)[-1]
            folder = apath.rsplit("/", 2)[-2] if apath.count("/") >= 2 else ""
            low = name.lower()
            native_lib = low.endswith((".dll", ".dylib")) or low.endswith(".so") or ".so." in low
            if MS_RUNTIME.match(name) or not native_lib or re.search(r"\.(cp\d+|abi3|cpython)", low):
                continue
            entry = next((e for e in VENDORED_LIBS if re.match(e[0], low)), None)
            if comp.kind == "conda" and "site-packages" not in src.replace("\\", "/"):
                # A conda package's own libraries (Library/bin, lib): only
                # LGPL or GPL code it carries under another project's name
                # (libquadmath in libgfortran5) is looked at.
                if entry is None or entry[4] in ("permissive", "exception") or \
                        any(w in comp.canon for w in entry[2]):
                    continue
            elif not (folder.endswith((".libs", ".dylibs")) or not any(w in low for w in own)):
                continue
            if entry is None:
                self.add("FAIL", "vendored library", f"{comp.name} {comp.version} carries {name}, a library this "
                                                     f"check does not know; identify it and its licence")
                continue
            _, project, words, texts, terms, source = entry
            if terms not in ("permissive", "exception"):     # copyleft: its licence is named
                version, url = "", ""
                compiler, builder = gcc_build(self.stage / apath)
                upstream = [u for u, _ in comp.source.get("upstream") or []]
                if comp.kind == "conda" and upstream and "site-packages" not in src.replace("\\", "/"):
                    # Built by the package's own recipe: its source and patches.
                    version, url, builder = comp.version.split()[0], upstream[0], ""
                elif "GCC" in project and compiler:
                    version, url = compiler, f"{source}gcc-{compiler}/gcc-{compiler}.tar.xz"
                elif project.startswith("SuiteSparse"):
                    version = suitesparse_version(self.stage / apath)
                    url = f"https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v{version}.tar.gz" \
                        if version else ""
                label = f"{project} (SuiteSparse {version})" if project.startswith("SuiteSparse") and version \
                    else f"{project} {version}".strip()
                comp.source.setdefault("embedded", []).append(
                    (label, terms, apath, url, f"built by {builder}" if builder else ""))
            if any(w in text for w in words):
                continue
            if not texts:
                self.add("FAIL", "vendored library", f"{comp.name} {comp.version} carries {name} ({project}); its "
                                                     f"licence files do not cover it and no text is on record")
                continue
            self.add_texts(comp, texts, added)
            comp.notes.append(f"carries {project} ({name}); its licence text is added under vendored/")
            self.add("INFO", "vendored library", f"{comp.name} {comp.version} carries {name} ({project}); licence "
                                                 f"text added from packaging/licenses/vendored/")
        if comp.kind == "conda":
            for lib, version, pkg_url in comp.source.get("header_only", []):
                project, texts, terms, source = HEADER_ONLY[lib]
                self.add_texts(comp, texts, added)
                label = f"{project} {version}".strip()
                if terms not in ("permissive", "exception"):
                    url = source.format(v=version) if version and "{v}" in source else ""
                    comp.source.setdefault("embedded", []).append(
                        (label, terms, "compiled in from its headers", url,
                         f"the headers as the build used them: {pkg_url}" if pkg_url else ""))
                comp.notes.append(f"compiled with {label} (header-only, per its recipe); its licence is under vendored/")
                self.add("INFO", "compiled-in headers", f"{comp.name} {comp.version} is compiled with {label}; "
                                                        f"licence text added")

    def embedded_checks(self):
        """LGPL, MPL and GPL code inside another component needs its own
        release-specific source, as a component of its own would."""
        for comp in self.components.values():
            for label, terms, where, url, note in comp.source.get("embedded", []):
                kind = terms
                if not url:
                    self.add("DECIDE", "embedded source", f"{label} ({kind}), {where} in {comp.name} {comp.version}: "
                                                          f"no release-specific source identified")
                elif note.startswith("built by"):
                    self.add("DECIDE", "embedded source", f"{label} ({kind}), {where} in {comp.name} "
                                                          f"{comp.version}: upstream source {url}; {note}, whose "
                                                          f"patches for this build are not pinned")
                else:
                    self.add("INFO", "embedded source", f"{label} ({kind}) in {comp.name} {comp.version}: {url}")

    def add_texts(self, comp, texts, added):
        for t in texts:
            if t not in added:
                comp.licence_files.append((VENDORED_DIR / t, f"vendored/{t}"))
                added.add(t)
        if "SOURCES.txt" not in added:
            comp.licence_files.append((VENDORED_DIR / "SOURCES.txt", "vendored/SOURCES.txt"))
            added.add("SOURCES.txt")

    # -------------------------------------------------------------- rustsolve
    def rustsolve_checks(self):
        """The Full build's rustsolve files (packaging/rustsolve/): each one is
        in the archive as the verified build, beside the pgcore build it was
        verified with, and none of the libraries it makes redundant is."""
        spec = json.loads(RUSTSOLVE_MANIFEST.read_text(encoding="utf-8"))["platforms"].get(self.args.os)
        if spec is None:
            self.add("FAIL", "rustsolve", f"packaging/rustsolve/manifest.json has no {self.args.os} build")
            return
        traced = {a.rsplit("/", 1)[-1].lower(): (c, a) for c in self.components.values() if c.kind == "rustsolve"
                  for a in c.files}
        for entry in spec["files"]:
            name = Path(entry["install"]).name
            if name.lower() in traced:
                comp, apath = traced[name.lower()]
                self.add("INFO", "rustsolve", f"{apath}: packaging/rustsolve/{entry['file']} (SHA-256 "
                                              f"{entry['sha256']}), listed as {comp.name} {comp.version}")
            else:
                self.add("FAIL", "rustsolve", f"{name} with the SHA-256 of packaging/rustsolve/{entry['file']} is "
                                              f"not in the archive")
        want = spec["pgcore"]
        pgcore = [c.version for c in self.components.values() if c.canon == "pgcore"]
        if f"{want['version']} (build {want['build']})" not in pgcore:
            self.add("FAIL", "rustsolve", f"the archive's pgcore is {', '.join(pgcore) or 'missing'}; the rustsolve "
                                          f"files were verified with {want['version']} (build {want['build']})")
        present = defaultdict(list)
        for path in iter_files(self.bundle):
            present[path.name.lower()].append(path.relative_to(self.stage).as_posix())
        removed = [Path(r).name for r in spec["remove"]]
        left = [p for n in removed for p in present.get(n.lower(), [])]
        if left:
            self.add("FAIL", "rustsolve", f"libraries rustsolve replaces are still in the archive: {', '.join(left)}")
        elif removed:
            self.add("INFO", "rustsolve", f"not in the archive, as pgcore's: {', '.join(removed)}")

    # ------------------------------------------------------------- exclusions
    def exclusion_checks(self):
        parts = defaultdict(set)
        for comp in self.components.values():
            for apath in comp.files:
                head = apath.split(" (embedded archive: ")
                for part in head[0].split("/"):
                    parts[part].add(head[0])
                if len(head) > 1:     # an embedded module: its top-level package
                    parts[head[1].rstrip(")").split(".")[0]].add(apath)
        for label, pattern in EXCLUDED_FILES.items():
            hits = sorted({p for part, paths in parts.items() if pattern.match(part) for p in paths})
            self.add("FAIL" if hits else "INFO", "excluded components",
                     f"{label}: {', '.join(hits[:5])}" if hits else f"{label}: no file or module by name")
        hits = defaultdict(list)
        for path in iter_files(self.bundle):
            if not is_native(path):
                continue
            data = path.read_bytes()
            for label, needles, all_needed, _ in SIGNATURES:
                found = [n in data for n in needles]
                if all(found) if all_needed else any(found):
                    hits[label].append(path.relative_to(self.stage).as_posix())
        for label, needles, _, blocking in SIGNATURES:
            if hits[label]:
                shown = [f"{p} ({symbol_role(self.stage / p, needles)})" for p in hits[label][:6]]
                self.add("FAIL" if blocking else "INFO", "compiled-in code",
                         f"{label}: {', '.join(shown)}" + (" ..." if len(hits[label]) > 6 else ""))
            else:
                self.add("INFO", "compiled-in code", f"{label}: not in any binary")
        self.signature_hits = hits

    # -------------------------------------------------------------------- Qt
    def qt_checks(self):
        lib_re = re.compile(r"^(?:lib)?Qt6([A-Z][A-Za-z0-9]*?)(?:\.\d+)*\.(?:dll|dylib|so(?:\.\d+)*)$")
        fw_re = re.compile(r"^Qt([A-Z][A-Za-z0-9]*)\.framework$")
        bind_re = re.compile(r"^Qt([A-Z][A-Za-z0-9]*)(?:\.[^./]+)?\.(?:pyd|so)$")
        modules, plugins, qml, bindings, tools = defaultdict(set), set(), set(), set(), set()
        plugin_dirs, runtime_libs, binding_dirs = set(), set(), set()
        core_owners = defaultdict(list)
        for comp in self.components.values():
            for apath in comp.files:
                if " (embedded archive: " in apath:
                    continue
                parts = apath.split("/")
                name = parts[-1]
                m = lib_re.match(name)
                if m:
                    modules[m.group(1)].add(apath)
                    if m.group(1) == "Core":
                        core_owners[f"{comp.name} {comp.version}"].append(apath)
                for i, part in enumerate(parts):
                    m = fw_re.match(part)
                    if m:
                        fw = "/".join(parts[:i + 1])
                        modules[m.group(1)].add(fw)
                        if m.group(1) == "Core" and name == "QtCore":     # the framework's library
                            core_owners[f"{comp.name} {comp.version}"].append(fw)
                m = bind_re.match(name)
                if m and "PySide6" in parts:
                    bindings.add(m.group(1))
                if "plugins" in parts:
                    i = len(parts) - 1 - parts[::-1].index("plugins")
                    if len(parts) > i + 2:
                        plugins.add((parts[i + 1], parts[-1]))
                        plugin_dirs.add("/".join(parts[:i + 1]))
                if re.match(r"^(lib)?(pyside6|shiboken6)[^/]*\.(dll|dylib|so[.\d]*)$", name, re.I):
                    runtime_libs.add(apath)
                if bind_re.match(name) and "PySide6" in parts:
                    binding_dirs.add("/".join(parts[:-1]))
                if "qml" in parts:
                    i = parts.index("qml")
                    if len(parts) > i + 2:
                        qml.add(parts[i + 1])
                stem = name[:-4] if name.lower().endswith(".exe") else name
                if stem in QT_TOOLS and (name.lower().endswith(".exe") or "." not in name):
                    tools.add(apath)
        self.qt.update(modules={k: sorted(v) for k, v in modules.items()}, plugins=sorted(plugins),
                       qml=sorted(qml), bindings=sorted(bindings),
                       lib_dirs=sorted({p.rsplit("/", 1)[0] for ps in modules.values() for p in ps}),
                       plugin_dirs=sorted(plugin_dirs), runtime_libs=sorted(runtime_libs),
                       binding_dirs=sorted(binding_dirs))
        if not modules:
            self.add("INFO", "Qt", "no Qt libraries in the archive")
            return
        if len(core_owners) > 1:
            self.add("FAIL", "Qt", "more than one Qt build is bundled: " +
                     "; ".join(f"{o}: {', '.join(p)}" for o, p in sorted(core_owners.items())))
        for path in sorted(tools):
            self.add("FAIL", "Qt tools", f"development tool bundled: {path}")
        if not tools:
            self.add("INFO", "Qt tools", "no Qt or PySide6 development tools bundled")
        gpl = [m for m in sorted(set(modules) | bindings) if m.startswith(QT_GPL_ONLY)]
        for m in gpl:
            self.add("FAIL", "Qt GPL-only modules", f"Qt {m} (GPL-3.0 only): {modules.get(m, ['binding'])[:2]}")
        if not gpl:
            self.add("INFO", "Qt GPL-only modules", "none of " + ", ".join(QT_GPL_ONLY) + " is bundled")
        version = self.qt_version()
        self.qt["version"] = version
        if not version:
            self.add("FAIL", "Qt licence", "Qt version unknown; module licences not checked")
            return
        wanted = {f"Qt {m}": [f"src/{QT_SRC_DIR.get(m, m.lower())}"] for m in modules}
        wanted.update({f"Qt {b} binding's library": [f"src/{QT_SRC_DIR.get(b, b.lower())}"] for b in bindings})
        for top in qml:
            wanted[f"QML {top}"] = [{"QtQml": "src/qml", "QtQuick": "src/quick"}.get(top, f"src/{top.lower()}")]
        spdx_all, odd_all = defaultdict(Counter), defaultdict(list)
        self.qt_attr = []
        for repo in QT_REPOS:
            try:
                path, url, digest = qt_source(repo, version, self.cache)
            except RuntimeError as exc:
                self.add("FAIL", "Qt sources", str(exc))
                continue
            if not path:
                self.add("FAIL", "Qt sources", f"{repo} {version}: no source archive on download.qt.io")
                continue
            self.qt["sources"].append((repo, url, digest, path))
            spdx, odd, attrs = scan_qt_tarball(path, sorted({d for ds in wanted.values() for d in ds}))
            for d, counts in spdx.items():
                spdx_all[d].update(counts)
                odd_all[d] += odd.get(d, [])
            self.qt_attr += [(repo, rel, raw) for rel, raw in attrs]
        # Each plugin is checked against its own source folder.
        for ptype, pname in plugins:
            label, stem = f"plugin {ptype}/{pname}", plugin_stem(pname)
            folders = sorted(d for d in spdx_all if d.startswith(f"src/plugins/{ptype}/"))
            names = {d: d.rsplit("/", 1)[1].lower() for d in folders}
            match = [d for d in folders if names[d] == stem] or \
                [d for d in folders if names[d].startswith(stem) or stem.startswith(names[d])]
            if len(match) != 1:
                self.add("FAIL", "Qt licence", f"{label}: source folder not identified under src/plugins/{ptype} "
                                               f"in Qt {version} (candidates: {match or folders})")
                continue
            wanted[label] = match + list(QT_PLUGIN_ALSO.get(names[match[0]], ()))
        for label, folders in sorted(wanted.items()):
            counts = Counter()
            for folder in folders:
                counts.update(spdx_all.get(folder, Counter()))
            where = " and ".join(folders)
            if not counts:
                self.add("FAIL", "Qt licence", f"{label}: no sources under {where} in Qt {version}")
                continue
            odd = [o for folder in folders for o in odd_all.get(folder, [])]
            # Linux kernel headers (WITH Linux-syscall-note) that Qt compiles
            # only for Linux, under `#if defined(Q_OS_LINUX)`.
            uapi = [(rel, e) for rel, e in odd if "Linux-syscall-note" in e and "/3rdparty/linux/" in rel]
            rest = [(rel, e) for rel, e in odd if (rel, e) not in uapi or self.args.os == "linux"]
            if rest:
                shown = "; ".join(f"{rel} ({e})" for rel, e in rest[:4])
                self.add("FAIL", "Qt licence", f"{label}: {len(rest)} source files under {where} offer no LGPL or "
                                               f"permissive licence: {shown}")
                continue
            self.add("INFO", "Qt licence", f"{label}: all {sum(counts.values()) - len(uapi)} source files under "
                                           f"{where} offer LGPL-3.0 or a permissive licence")
            for rel, e in uapi:
                self.add("INFO", "Qt licence", f"{label}: {rel} is a Linux kernel header ({e}) that Qt compiles "
                                               f"for Linux only; not in this {self.args.os} build")

    def qt_version(self):
        for comp in self.qt_components():
            if comp.canon in ("pyside6-essentials", "qt6-main", "qt-main"):
                return comp.version.split()[0]
        try:
            from PySide6 import QtCore
            return QtCore.qVersion()
        except ImportError:
            return ""

    def qt_notices(self):
        """Copy the notices of the third-party code Qt's sources document for
        the bundled modules and plugins."""
        if not self.qt["modules"]:
            return
        docs = {f"qt{m.lower()}" for m in self.qt["modules"]}
        if any(t == "imageformats" for t, _ in self.qt["plugins"]):
            docs.add("qtimageformats")
        chosen = []
        for repo, rel, raw in self.qt_attr:
            try:
                entries = attribution_entries(raw)
            except ValueError:
                self.add("FAIL", "Qt notices", f"{repo}/{rel}: unreadable qt_attribution.json")
                continue
            for entry in entries:
                if (entry.get("QDocModule") or "").lower() not in docs:
                    continue
                usage = entry.get("QtUsage") or ""
                if re.search(r"\b(only )?used (only )?(in|by) (the )?(examples?|tests?|tools?|demos?|documentation)\b",
                             usage, re.I):
                    continue
                chosen.append((repo, rel, entry))
        needed = defaultdict(set)
        for repo, rel, entry in chosen:
            base = rel.rsplit("/", 1)[0]
            for lf in as_list(entry.get("LicenseFiles")) + as_list(entry.get("LicenseFile")):
                needed[repo].add(os.path.normpath(f"{base}/{lf}").replace(os.sep, "/"))
        blobs = {}
        for repo, url, digest, path in self.qt["sources"]:
            blobs.update({f"{repo}/{k}": v for k, v in extract_from_tarball(path, needed[repo]).items()})
        out = self.stage / "LICENSES" / "Qt-third-party"
        rows = []
        for repo, rel, entry in chosen:
            ident = safe_key(f"{entry.get('QDocModule', '')}-{entry.get('Id') or entry.get('Name') or rel}")
            folder = out / ident
            folder.mkdir(parents=True, exist_ok=True)
            base = rel.rsplit("/", 1)[0]
            copied = []
            for lf in as_list(entry.get("LicenseFiles")) + as_list(entry.get("LicenseFile")):
                key = f"{repo}/" + os.path.normpath(f"{base}/{lf}").replace(os.sep, "/")
                if key in blobs:
                    (folder / Path(lf).name).write_bytes(blobs[key])
                    copied.append(Path(lf).name)
            if not copied:
                for spdx_id in re.findall(r"[A-Za-z0-9.+-]+", entry.get("LicenseId") or ""):
                    blob = blobs.get(f"{repo}/LICENSES/{spdx_id}.txt")
                    if blob:
                        (folder / f"{spdx_id}.txt").write_bytes(blob)
                        copied.append(f"{spdx_id}.txt")
            notice = "\n".join(as_list(entry.get("Copyright")))
            (folder / "NOTICE.txt").write_text(
                f"{entry.get('Name', ident)} {entry.get('Version', '')}\n"
                f"In Qt {self.qt['version']} ({entry.get('QDocModule')}): {entry.get('QtUsage', '')}\n"
                f"Licence: {entry.get('License', '')} ({entry.get('LicenseId', '')})\n"
                f"Recorded in {repo}/{rel}\n\n{notice}\n", encoding="utf-8")
            if not copied:
                self.add("FAIL", "Qt notices", f"{entry.get('Name')}: licence text not found in {repo}")
            rows.append((entry.get("Name", ident), entry.get("Version", ""), entry.get("LicenseId", ""),
                         entry.get("QDocModule", ""), f"LICENSES/Qt-third-party/{ident}/"))
        rows.sort(key=lambda r: r[0].lower())
        sources = ", ".join(f"{u.rsplit('/', 1)[-1]} (SHA-256 {d})" for _, u, d, _ in self.qt["sources"])
        lines = [f"Third-party code that Qt {self.qt['version']}'s sources (qt_attribution.json in {sources}) "
                 f"document for the bundled modules ({', '.join(sorted(docs))}). A given Qt build may use a "
                 f"system copy of some of them instead.", ""]
        lines += [f"{n} {v} | {l} | {m} | {p}" for n, v, l, m, p in rows]
        out.mkdir(parents=True, exist_ok=True)
        (out / "INDEX.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.qt["notices"] = rows
        self.add("INFO", "Qt notices", f"{len(rows)} third-party notices from Qt's sources")

    def pyside_checks(self):
        """PySide6/Shiboken6 source archive, and the licence of the two runtime
        libraries from it."""
        version = self.version_of("shiboken6")
        if not version or not self.qt_components():
            return
        name = f"pyside-setup-everywhere-src-{version}.tar.xz"
        for base in (f"https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-{version}-src/",
                     f"https://download.qt.io/archive/qt-for-python/pyside6/PySide6-{version}-src/"):
            if url_ok(base + name):
                dest = self.cache / name
                if not dest.is_file():
                    fetch(base + name, dest, timeout=900)
                self.pyside_tar = (base + name, sha256_of(dest), dest)
                break
        if not self.pyside_tar:
            self.add("FAIL", "PySide6 source", f"{name} not found on download.qt.io")
            return
        counts = defaultdict(Counter)
        dirs = ("sources/pyside6/libpyside", "sources/shiboken6/libshiboken")
        for tar, member, rel in tar_members(self.pyside_tar[2]):
            d = next((d for d in dirs if rel.startswith(d + "/")), None)
            if d and rel.endswith((".cpp", ".h")):
                m = SPDX_LINE.search(tar.extractfile(member).read(4096))
                counts[d][m.group(1).decode().strip() if m else "(no SPDX header)"] += 1
        for d in dirs:
            bad = {e: n for e, n in counts[d].items() if not spdx_ok(e)}
            if not counts[d] or bad:
                self.add("FAIL", "PySide6 licence", f"{d}: {bad or 'no sources found'}")
            else:
                self.add("INFO", "PySide6 licence", f"{d}: all {sum(counts[d].values())} source files offer LGPL-3.0")

    # ------------------------------------------------------------ Windows CRT
    def runtime_checks(self):
        for comp in self.components.values():
            for apath in comp.files:
                name = apath.rsplit("/", 1)[-1]
                if MS_RUNTIME.match(name):
                    self.ms_runtime.append((name, pe_version(self.stage / apath), f"{comp.name} {comp.version}",
                                            apath))
        self.ms_runtime.sort()
        if self.ms_runtime:
            self.add("DECIDE", "Microsoft runtime",
                     f"{len(self.ms_runtime)} Microsoft C/C++ runtime files are bundled (BUILD_INFO.txt lists each "
                     f"with its version and the package that carried it); their redistribution terms need an "
                     f"assessment")

    # ------------------------------------------------------ dependency check
    def dependency_checks(self):
        from PyInstaller.depend import bindepend
        present = set()      # every name in the bundle, symlinks included (libz.1.dylib -> libz.1.3.1.dylib)
        for _folder, dirs, files in os.walk(self.bundle, followlinks=False):
            present |= {n.lower() for n in dirs + files}
        natives = [p for p in iter_files(self.bundle) if is_native(p)]
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        missing = defaultdict(set)
        alias = {}          # libz.1.dylib -> libz.1.3.1.dylib
        for folder, _dirs, files in os.walk(self.bundle, followlinks=False):
            for n in files:
                p = Path(folder) / n
                if p.is_symlink():
                    alias[n.lower()] = p.resolve().name.lower()
        loaded_by = defaultdict(set)
        for path in natives:
            try:
                imports = bindepend.get_imports(str(path))
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                self.add("INFO", "dependencies", f"imports of {path.name} unreadable: {exc}")
                continue
            for name, _resolved in imports:
                base = Path(name).name.lower()
                loaded_by[alias.get(base, base)].add(path.relative_to(self.stage).as_posix())
                if base in present or base.startswith(("api-ms-win-", "ext-ms-")):
                    continue
                if name.startswith(("/usr/lib/", "/System/", "/lib/", "/lib64/")):
                    continue
                if "/" in name and any(f"{p}.framework" in present for p in [Path(name).name.lower()]):
                    continue
                if os.name == "nt" and (system32 / base).is_file():
                    continue
                missing[name].add(path.relative_to(self.stage).as_posix())
        for name, users in sorted(missing.items()):
            self.add("FAIL", "dependencies", f"{name} is needed by {', '.join(sorted(users)[:3])} but not bundled")
        # What brings each GPL component in: the decision about it starts here.
        for comp in self.components.values():
            if comp.source.get("class") != "strong":
                continue
            own = {a.rsplit("/", 1)[-1].lower() for a in comp.files}
            users = sorted({u for n in own for u in loaded_by.get(n, ())} - set(comp.files))
            self.add("INFO", "dependencies", f"{comp.name} {comp.version} ({comp.licence}) is loaded by: "
                                             f"{', '.join(users[:8]) or 'no bundled binary'}")
        if not missing:
            self.add("INFO", "dependencies", f"every import of the {len(natives)} native files resolves inside the "
                                             f"archive or to the operating system")

    # ------------------------------------------------------------- writing
    def write_licences(self):
        lic = self.stage / "LICENSES"
        lic.mkdir(parents=True, exist_ok=True)
        for name in GNU_TEXTS:
            shutil.copyfile(REPO / "packaging" / "licenses" / name, lic / name)
        for comp in self.components.values():
            for src, label in comp.licence_files:
                target = self.stage / comp.notice_dir / label
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, target)
        if self.signature_hits.get("Triangle"):
            self.triangle_notice(lic)

    def triangle_notice(self, lic):
        """Triangle's README (its terms) and source header, from the author's
        release at netlib."""
        folder = lic / "Triangle"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            raw = fetch("https://www.netlib.org/voronoi/triangle.zip", timeout=120)
        except RuntimeError as exc:
            self.add("FAIL", "Triangle", f"Triangle's notice could not be fetched: {exc}")
            return
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            (folder / "README").write_bytes(z.read("README"))
            source = z.read("triangle.c").decode("latin-1")
        header = source.split("*/", 1)[0] + "*/\n"
        (folder / "triangle.c-header.txt").write_text(header, encoding="utf-8")
        self.triangle = hashlib.sha256(raw).hexdigest()
        self.add("INFO", "Triangle", f"README and triangle.c header copied to LICENSES/Triangle/ from netlib's "
                                     f"triangle.zip (SHA-256 {self.triangle}); compiled into "
                                     f"{', '.join(self.signature_hits['Triangle'][:4])}")
        self.add("DECIDE", "Triangle", "Triangle's terms forbid selling it or including it in a commercial product "
                                       "without the author's permission; confirm whether pyGIMLi's copy is modified")

    def build_info(self):
        commit = self.args.commit
        lines = [
            "PyHydroGeophysX Studio desktop archive",
            f"Variant: {self.args.variant}",
            f"Platform: {self.args.os}; {platform.platform()}; architecture {platform.machine()}",
            f"Application: PyHydroGeophysX {project_version()}, built from commit {commit}",
            f"Source: {REPO_URL}/tree/{commit}",
            f"Built: {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC",
            f"Python: {platform.python_version()} ({platform.python_implementation()}, {platform.python_compiler()})",
            f"PyInstaller: {self.version_of('pyinstaller')}",
            f"Qt: {self.qt['version'] or 'not bundled'}",
            f"PySide6: {self.version_of('PySide6')}",
            f"Shiboken6: {self.version_of('shiboken6')}",
            "",
            "At the root of this archive: LICENSE (PyHydroGeophysX, Apache-2.0), THIRD_PARTY_NOTICES.md, "
            "SOURCE_CODE.md, this file, and LICENSES/ with LGPL-3.0-only.txt, GPL-3.0-only.txt, "
            "Qt-third-party/ and one folder of licence and notice files per component below.",
            "",
            "Components",
            "==========",
            "Name and version or build; licence as the package declares it [class]; origin; notice files; the "
            "files it supplies. 'embedded' modules are pure Python inside the executable.",
            "",
        ]
        for comp in sorted(self.components.values(), key=lambda c: c.name.lower()):
            if comp.licence_files:
                notices = comp.notice_dir + "/"
            else:
                notices = "LICENSE at the archive root" if comp.kind == "project" else "(none)"
            paths = [p for p in comp.files if " (embedded archive: " not in p]
            embedded = len(comp.files) - len(paths)
            lines += [f"- {comp.name} {comp.version}".rstrip(),
                      f"    licence: {comp.licence or '(not declared)'} [{comp.source.get('class', '')}]",
                      f"    origin: {comp.origin}",
                      f"    notices: {notices}"]
            if comp.notes:
                lines.append(f"    note: {'; '.join(sorted(set(comp.notes)))}")
            lines.append(f"    files: {len(paths)}" + (f", plus {embedded} embedded modules" if embedded else ""))
            lines += [f"      {p}" for p in summarise_paths(paths)]
        if self.qt["modules"]:
            lines += ["", "Qt", "==",
                      f"Qt {self.qt['version']}. The licence of every source file of these modules and plugins was "
                      f"checked in the Qt sources named in SOURCE_CODE.md.",
                      "  Libraries: " + ", ".join(f"Qt {m}" for m in sorted(self.qt["modules"])),
                      "  PySide6 bindings: " + ", ".join(f"Qt{b}" for b in self.qt["bindings"]),
                      "  Plugins: " + ", ".join(f"{t}/{n}" for t, n in self.qt["plugins"]),
                      "  QML folders: " + (", ".join(self.qt["qml"]) or "none"),
                      "  Third-party notices: LICENSES/Qt-third-party/ (INDEX.txt lists them)"]
        if self.signature_hits.get("Triangle"):
            lines += ["", "Triangle", "========",
                      "Compiled into " + ", ".join(self.signature_hits["Triangle"][:4]) +
                      "; its terms are in LICENSES/Triangle/."]
        if self.ms_runtime:
            lines += ["", "Microsoft C/C++ runtime files", "============================="]
            lines += [f"  {n} {v or '(no version resource)'}, carried by {o}: {p}" for n, v, o, p in self.ms_runtime]
        (self.stage / "BUILD_INFO.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def version_of(self, name):
        try:
            return (importlib.metadata.version(name) or "").strip()   # conda's shiboken6 records "6.11.2 "
        except importlib.metadata.PackageNotFoundError:
            canon = re.sub(r"[-_.]+", "-", name).lower()
            return next((c.version.split()[0] for c in self.components.values() if c.canon == canon), "")

    def source_code(self):
        commit = self.args.commit
        lines = [
            "# Source code for this archive",
            "",
            f"This archive ({self.args.os} {self.args.variant}) was built from PyHydroGeophysX commit `{commit}`:",
            "",
            f"- browse: {REPO_URL}/tree/{commit}",
            f"- download: {REPO_URL}/archive/{commit}.zip",
            "",
            "The build recipe is `.github/workflows/build-desktop.yml` and `packaging/pyinstaller_studio.spec` at "
            "that commit. `BUILD_INFO.txt` lists every bundled component with its version, origin and licence.",
            "",
        ]
        if self.args.tag_note:
            lines += [self.args.tag_note, ""]
        qt_comps = self.qt_components()
        if self.qt["modules"]:
            shib = self.version_of("shiboken6")
            lines += [
                "## Qt, PySide6 and Shiboken6 (GNU LGPL v3)",
                "",
                f"This archive uses Qt {self.qt['version']} and PySide6/Shiboken6 {shib} under the GNU Lesser "
                "General Public License v3 (`LICENSES/LGPL-3.0-only.txt`, which builds on "
                "`LICENSES/GPL-3.0-only.txt`). Their corresponding source is available at no charge from these "
                "release-specific downloads. The SHA-256 values are those of the files read when this archive was "
                "checked.",
                "",
            ]
            for repo, url, digest, _ in self.qt["sources"]:
                lines.append(f"- Qt {repo} {self.qt['version']}: {url} (SHA-256 `{digest}`)")
            lines.append(f"- Qt {self.qt['version']}, all modules in one archive: {qt_everywhere_url(self.qt['version'])}")
            if self.pyside_tar:
                lines.append(f"- PySide6 and Shiboken6 {shib}: {self.pyside_tar[0]} (SHA-256 `{self.pyside_tar[1]}`)")
            lines += ["", "The bundled Qt, PySide6 and Shiboken6 binaries come from:", ""]
            for comp in sorted(qt_comps, key=lambda c: c.name.lower()):
                lines.append(f"- {comp.name} {comp.version}: {comp.origin}")
                if comp.kind == "conda":
                    lines += self.conda_source_lines(comp, indent="  ")
                elif comp.kind == "pip":
                    lines.append("  - built and published by The Qt Company from the release sources above")
            lines += ["", f"Qt's own report of its build: `{qt_build_string()}`", ""]
            lines += replacement_instructions(self.args.os, self.qt)
        weak = [c for c in self.components.values()
                if c.source.get("class") in ("weak", "strong") and c not in qt_comps and c.kind != "project"]
        if weak:
            lines += ["", "## Other components under the LGPL, MPL or GPL", ""]
            for comp in sorted(weak, key=lambda c: c.name.lower()):
                lines.append(f"- {comp.name} {comp.version} ({comp.licence}):")
                lines += self.conda_source_lines(comp, "  ") if comp.kind == "conda" else \
                    [f"  - source: {pypi_sdist(comp.name, comp.version) or 'no source distribution on PyPI'}"]
        inside = [(c, e) for c in self.components.values() for e in c.source.get("embedded", [])]
        if inside:
            lines += ["", "## LGPL, MPL and GPL code inside other components", ""]
            for comp, (label, terms, where, url, note) in sorted(inside, key=lambda x: (x[0].name.lower(), x[1][0])):
                kind = terms
                lines.append(f"- {label} ({kind}), {where} in {comp.name} {comp.version}: "
                             f"{url or 'release-specific source not identified'}" + (f"; {note}" if note else ""))
        noted = [c for c in self.components.values() if c.source.get("source_note")]
        if noted:
            lines += ["", "## Source notices other licences ask for", ""]
            for comp in sorted(noted, key=lambda c: c.name.lower()):
                lines.append(f"- {comp.source['source_note']} Bundled: {comp.name} {comp.version}.")
                lines += self.conda_source_lines(comp, "  ") if comp.kind == "conda" else []
        lines += ["", "## Your rights", "",
                  "Neither PyHydroGeophysX's licence (Apache-2.0) nor this archive restricts modifying the "
                  "LGPL-covered libraries, replacing them, or reverse engineering the combined work to debug such "
                  "modifications."]
        (self.stage / "SOURCE_CODE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def conda_source_lines(self, comp, indent):
        src, out = comp.source, []
        if src.get("package_url"):
            out.append(f"{indent}- package: {src['package_url']}")
        answered = set()
        for u, s in sorted(src.get("upstream") or [], key=lambda x: not url_ok(x[0])):
            digest = f" (SHA-256 `{s}`)" if s else ""
            name = u.rstrip("/").rsplit("/", 1)[-1]
            if name in answered:
                continue        # another mirror the recipe lists for the same file
            if url_ok(u):
                out.append(f"{indent}- upstream source: {u}{digest}")
                answered.add(name)
                continue
            # The recipe's download mirror has gone: the same file where its
            # publisher keeps it, if it is there.
            alt = next((a for a in mirror_alternatives(u) if url_ok(a)), "")
            host = u.split("/")[2] if "//" in u else u
            if alt:
                out.append(f"{indent}- upstream source: {alt}{digest} (the recipe downloaded the same file name "
                           f"from {host}, which no longer answers)")
            else:
                out.append(f"{indent}- upstream source: the recipe's download from {host} no longer answers; "
                           f"the recipe below names it")
                self.add("DECIDE", "source", f"{comp.name} {comp.version}: its recipe's source download ({host}) "
                                             f"no longer answers and no other copy was found")
        if src.get("feedstock"):
            out.append(f"{indent}- recipe and patches: {src['feedstock'].removesuffix('.git')}"
                       + (f"/tree/{src['feedstock_sha']}" if src.get("feedstock_sha") else ""))
        if src.get("patches"):
            out.append(f"{indent}- patches in that recipe: {', '.join(src['patches'])}")
        return out or [f"{indent}- (the package records no source)"]

    def report(self, path):
        counts = Counter(f.level for f in self.findings)
        lines = [f"# Audit: {self.args.os} {self.args.variant}", "",
                 f"Commit {self.args.commit}; {len(self.components)} components; "
                 f"FAIL {counts['FAIL']}, DECIDE {counts['DECIDE']}, INFO {counts['INFO']}.", ""]
        for level in ("FAIL", "DECIDE", "INFO"):
            items = [f for f in self.findings if f.level == level]
            if items:
                lines += [f"## {level}", ""] + [f"- [{f.check}] {f.detail}" for f in items] + [""]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines[:3]))
        for f in self.findings:
            if f.level != "INFO":
                print(f"{f.level}: [{f.check}] {f.detail}")
        return counts["FAIL"]


def qt_build_string():
    try:
        from PySide6 import QtCore
        return QtCore.QLibraryInfo.build()
    except ImportError:
        return "PySide6 not importable where the archive was checked"


def summarise_paths(paths, limit=12):
    """Group archive paths by folder so long lists stay readable."""
    by_dir = defaultdict(list)
    for p in paths:
        folder, _, name = p.rpartition("/")
        by_dir[folder].append(name)
    out = []
    for folder, names in sorted(by_dir.items()):
        if len(names) <= 3:
            out += [f"{folder}/{n}" for n in sorted(names)]
        else:
            out.append(f"{folder}/ ({len(names)} files: {', '.join(sorted(names)[:3])}, ...)")
    if len(out) > limit:
        out = out[:limit] + [f"... and {len(out) - limit} more folders"]
    return out


def qt_everywhere_url(version):
    """Qt's single source archive: official_releases while current, archive after."""
    mm = ".".join(version.split(".")[:2])
    urls = [f"https://download.qt.io/{area}/qt/{mm}/{version}/single/qt-everywhere-src-{version}.tar.xz"
            for area in ("official_releases", "archive")]
    return next((u for u in urls if url_ok(u)), urls[0])


def mirror_alternatives(url):
    """The publisher's own address for a file a recipe took from a mirror
    (download.qt.io for Qt mirrors)."""
    m = re.search(r"/(?:archive|official_releases)/(qt/.+|qt-for-python/.+|QtForPython/.+)$", url)
    if not m:
        return []
    return [f"https://download.qt.io/{area}/{m.group(1)}" for area in ("official_releases", "archive")]


def pypi_sdist(name, version):
    version = (version.split() or [""])[0]
    if not version:
        return ""
    try:
        data = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60))
    except (RuntimeError, ValueError):
        return ""
    for f in data.get("urls", []):
        if f.get("packagetype") == "sdist":
            return f"{f['url']} (SHA-256 `{f['digests']['sha256']}`)"
    return ""


def replacement_instructions(osname, qt):
    """Where this archive's Qt, PySide6 and Shiboken6 files are, and how to
    replace them (the paths are those of the files in the archive)."""
    def paths(items, suffix=""):
        return ", ".join(f"`{p}{suffix}`" for p in items) or "(none)"
    libs = "Qt frameworks" if any(".framework" in p for ps in qt["modules"].values() for p in ps) else "Qt libraries"
    lines = ["## Replacing the Qt, PySide6 and Shiboken6 libraries", "",
             "The studio is a one-folder build: these libraries are separate files loaded when it starts, not "
             "compiled into the program. In this archive they are:", "",
             f"- {libs}: in {paths(qt['lib_dirs'], '/')}",
             f"- Qt plugins: in {paths(qt['plugin_dirs'], '/')}",
             f"- PySide6 and Shiboken6 runtime libraries: {paths(qt['runtime_libs'])}",
             f"- PySide6's Qt bindings, built against this Qt: in {paths(qt['binding_dirs'], '/')}", ""]
    if osname == "windows":
        lines.append("Copy your builds over the files of the same name there, keeping the folder layout, and start "
                     f"`{APP}/{APP}.exe`; the program checks no signature or checksum of these files.")
    else:
        lines.append("Copy your builds over the files of the same name there, keeping the layout. A build made for "
                     "another layout looks for its libraries elsewhere: add to each copied file the rpath entries of "
                     "the file it replaces that it lacks (`otool -l` lists them as LC_RPATH; "
                     "`install_name_tool -add_rpath` adds one). Then re-sign the application so macOS loads it: "
                     "`codesign --force --deep --sign - PyHydroGeophysX-Studio.app`. The archive's own check does "
                     "exactly this with copies of the same release.")
    lines += ["", "Build replacements from the sources above: Qt's build instructions are in its source archive "
              "(README and `configure --help`), PySide6's in `README.pyside6.md` of pyside-setup. A replacement "
              "must be interface-compatible with the Qt and PySide6 versions in BUILD_INFO.txt."]
    return lines


# ================================================================ verify

def cmd_verify(args):
    failures = []
    archive = Path(args.archive)
    digest = sha256_of(archive)
    with zipfile.ZipFile(archive) as z:
        infos = {i.filename.replace("\\", "/"): i for i in z.infolist()}
        names = list(infos)
        roots = {n.split("/")[0] for n in names}
        for name in ROOT_FILES + ("LICENSES",):
            if name not in roots:
                failures.append(f"missing at the archive root: {name}")
        text = {}
        for name in ROOT_FILES:
            if name in infos:
                data = z.read(infos[name])
                text[name] = data.decode("utf-8", "replace")
                if len(data.strip()) < 40:
                    failures.append(f"{name} is (nearly) empty")
                if re.search(rb"\b(TODO|TBD|PLACEHOLDER|FIXME|XXX)\b", data):
                    failures.append(f"{name} holds a placeholder")
        for lic, expected in GNU_TEXTS.items():
            key = f"LICENSES/{lic}"
            if key not in infos:
                failures.append(f"missing {key}")
            elif hashlib.sha256(z.read(infos[key])).hexdigest() != expected:
                failures.append(f"{key} differs from the text published at gnu.org")
        for n, i in infos.items():
            if n.startswith("LICENSES/") and not n.endswith("/") and i.file_size == 0:
                failures.append(f"empty licence file: {n}")
        info = text.get("BUILD_INFO.txt", "")
        for ref in sorted(set(re.findall(r"(LICENSES/[^\s,;)]+)", info))):
            ref = ref.rstrip("/.")
            if not any(n == ref or n.startswith(ref + "/") for n in names):
                failures.append(f"BUILD_INFO.txt names {ref}, which is not in the archive")
        for ref in sorted(set(re.findall(r"^      (\S+/\S+)$", info, re.M))):
            if ref not in infos:
                failures.append(f"BUILD_INFO.txt lists {ref}, which is not in the archive")
        for ref in sorted(set(re.findall(rf"`((?:LICENSES|{APP})[^`]*)`", text.get("SOURCE_CODE.md", "")))):
            if " " in ref:      # a command, not a path
                continue
            if not any(n == ref or n.startswith(ref.rstrip("/") + "/") for n in names):
                failures.append(f"SOURCE_CODE.md names {ref}, which is not in the archive")
    links = sorted({u.rstrip(".,;") for u in re.findall(r"https?://[^\s`)>\]]+", text.get("SOURCE_CODE.md", ""))})
    dead = [u for u in links if not url_ok(u)]
    failures += [f"SOURCE_CODE.md link does not answer: {u}" for u in dead]
    lines = [f"# Verify: {archive.name}", "", f"SHA-256 `{digest}`; {len(names)} entries; {len(links)} source links "
             f"checked.", ""]
    lines += [f"- FAIL: {f}" for f in failures] or ["- all checks passed"]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(str(archive) + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print("\n".join(lines))
    return 1 if failures else 0


# ================================================================ replace

def cmd_replace(args):
    """Replace the Qt, PySide6 and Shiboken6 libraries in an unpacked copy of
    the archive with separately obtained copies of the same release, alter
    one, re-sign on macOS, and run --self-test."""
    archive = Path(args.archive)
    work = Path(tempfile.mkdtemp(prefix="phgx-replace-"))
    if sys.platform == "darwin":
        subprocess.run(["ditto", "-x", "-k", str(archive), str(work)], check=True)   # keeps symlinks
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(work)
    app = next(p for p in work.iterdir() if p.name.startswith(APP))
    fresh = Path(tempfile.mkdtemp(prefix="phgx-fresh-"))
    origin = obtain_fresh_qt(args, fresh)
    lines = [f"# Library replacement: {archive.name}", "", f"Replacement copies: {origin}", ""]
    in_app = defaultdict(list)
    for p in iter_files(app):
        in_app[p.name].append(p)
    # Every name a library goes by in the fresh copy, its symlinks included:
    # the archive holds libQt6Core.6.dylib, the package the file
    # libQt6Core.6.9.2.dylib with libQt6Core.6.dylib linking to it.
    fresh_by_name = {}
    for folder, _dirs, files in os.walk(fresh):
        for name in files:
            real = (Path(folder) / name).resolve()
            if real.is_file() and re.match(r"^(lib)?(Qt6|Qt[A-Z]|pyside6|shiboken6)", name, re.I) and is_native(real):
                fresh_by_name.setdefault(name, real)
    swapped = 0
    for name, src in sorted(fresh_by_name.items()):
        for target in in_app.get(name, []):
            before = sha256_of(target)
            rpaths = macho_rpaths(target) if sys.platform == "darwin" else []
            shutil.copyfile(src, target)
            added = []
            if sys.platform == "darwin":
                # As a user would: the replacement also looks for its
                # libraries where the file it replaces did.
                added = [r for r in rpaths if r not in macho_rpaths(target)]
                for r in added:     # a failure shows in --self-test below
                    subprocess.run(["install_name_tool", "-add_rpath", r, str(target)], check=False)
            swapped += 1
            lines.append(f"- {target.relative_to(work).as_posix()}: {before[:16]} -> {sha256_of(target)[:16]}"
                         + (f" (rpaths added from the replaced file: {', '.join(added)})" if added else ""))
    if sys.platform != "darwin":
        core = next((p for name in ("Qt6Core.dll", "libQt6Core.so.6") for p in in_app.get(name, [])), None)
        if core:
            with open(core, "ab") as f:    # PE and ELF loaders ignore data after the image
                f.write(b"\0" * 4096 + b"PyHydroGeophysX library-replacement test")
            lines.append(f"- then altered (4 KiB appended): {core.relative_to(work).as_posix()} -> "
                         f"{sha256_of(core)[:16]}")
    else:
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)], check=True)
        lines.append("- re-signed: codesign --force --deep --sign - (ad hoc)")
    exe = app / "Contents" / "MacOS" / APP if app.suffix == ".app" else app / f"{APP}.exe"
    if not exe.exists():
        exe = app / APP
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([str(exe), "--self-test"], env=env, capture_output=True, text=True, timeout=900)
    ok = proc.returncode == 0 and swapped > 0
    lines += ["", f"{swapped} library files replaced; --self-test exit code {proc.returncode}.", "",
              "```", (proc.stdout + proc.stderr)[-3000:].strip(), "```", "", "Result: " + ("PASS" if ok else "FAIL")]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(fresh, ignore_errors=True)
    return 0 if ok else 1


def macho_rpaths(path):
    """A Mach-O file's LC_RPATH entries, in order, each once (otool lists a
    universal binary's for every architecture; install_name_tool changes
    them in all at once)."""
    out = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True).stdout
    return list(dict.fromkeys(re.findall(r"cmd LC_RPATH\s+cmdsize \d+\s+path (.+?) \(offset \d+\)", out)))


def obtain_fresh_qt(args, dest):
    """The same Qt/PySide6/Shiboken6 release, obtained again from its publisher
    (PyPI) or, for conda builds, from the conda package cache."""
    try:
        dist = importlib.metadata.distribution("PySide6_Essentials")
        installer = (dist.read_text("INSTALLER") or "").strip()
    except importlib.metadata.PackageNotFoundError:
        installer = ""
    if installer in ("pip", "uv"):
        wheels = dest / "wheels"
        pkgs = [f"{n}=={importlib.metadata.version(n)}" for n in ("PySide6_Essentials", "shiboken6")]
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                        "-d", str(wheels), *pkgs], check=True)
        for whl in wheels.glob("*.whl"):
            with zipfile.ZipFile(whl) as z:
                z.extractall(dest / whl.stem)
        return "wheels downloaded again from PyPI: " + ", ".join(p.name for p in wheels.glob("*.whl"))
    prefix = Path(args.conda_prefix or sys.prefix)
    used = []
    for meta_file in (prefix / "conda-meta").glob("*.json"):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta["name"] in ("qt6-main", "pyside6"):
            pkg_dir = Path(meta.get("extracted_package_dir") or "")
            if pkg_dir.is_dir():
                shutil.copytree(pkg_dir, dest / meta["name"], symlinks=True, dirs_exist_ok=True)
                used.append(pkg_dir.name)
    return "conda package cache: " + ", ".join(used)


# ================================================================== main

def cmd_stage(args):
    audit = Audit(args)
    audit.trace()
    audit.licence_checks()
    if args.rustsolve:
        audit.rustsolve_checks()
    audit.exclusion_checks()
    audit.qt_checks()
    audit.runtime_checks()
    audit.dependency_checks()
    audit.write_licences()
    audit.qt_notices()
    audit.pyside_checks()
    audit.build_info()
    audit.source_code()
    fails = audit.report(args.report)
    return 1 if fails and args.strict else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage")
    st.add_argument("--os", required=True)
    st.add_argument("--variant", required=True)
    st.add_argument("--bundle", required=True, help="the studio folder or .app inside the stage folder")
    st.add_argument("--stage", required=True)
    st.add_argument("--workpath", required=True, help="PyInstaller's build folder for the spec")
    st.add_argument("--pip-report")
    st.add_argument("--conda-prefix")
    st.add_argument("--commit", required=True)
    st.add_argument("--tag-note", default="")
    st.add_argument("--cache", default="build/audit-cache")
    st.add_argument("--report", required=True)
    st.add_argument("--strict", action="store_true", help="exit 1 when a check fails")
    st.add_argument("--rustsolve", action="store_true",
                    help="the build put packaging/rustsolve/ in place of pgcore's SuiteSparse solvers: check it did")
    ve = sub.add_parser("verify")
    ve.add_argument("--archive", required=True)
    ve.add_argument("--report", required=True)
    rp = sub.add_parser("replace")
    rp.add_argument("--archive", required=True)
    rp.add_argument("--conda-prefix")
    rp.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    return {"stage": cmd_stage, "verify": cmd_verify, "replace": cmd_replace}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
