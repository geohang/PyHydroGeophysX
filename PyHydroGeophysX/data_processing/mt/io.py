"""One entry point for MT transfer-function files and one for instrument time series."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from .edi import is_edi_file, read_edi
from .emtf_xml import is_emtf_xml_file, read_emtf_xml
from .jfile import is_jfile, read_jfile
from .lemi import is_lemi424_file, read_lemi424
from .metronix import is_ats_file, read_metronix
from .phoenix import is_phoenix_recording, read_phoenix
from .phoenix_legacy import is_phoenix_legacy_file, read_phoenix_legacy
from .timeseries import TimeSeriesRun
from .transfer_function import TransferFunction
from .zfile import is_zfile, read_zfile
from .zonge import is_z3d_file, read_zonge

#: The instrument formats :func:`read_timeseries` recognizes.
TIMESERIES_FORMATS = ("phoenix", "phoenix_legacy", "metronix", "zonge", "lemi424")

#: File patterns :func:`read_transfer_functions` collects from a folder.
TRANSFER_FUNCTION_PATTERNS = ("*.edi", "*.EDI", "*.xml", "*.zmm", "*.zrr", "*.zss", "*.j")


def is_transfer_function_file(path: Any) -> bool:
    return any(check(path) for check in (is_edi_file, is_emtf_xml_file, is_zfile, is_jfile))


def read_transfer_function(path: Any, *, sign_convention: str = "auto",
                           z_units: str = "auto") -> TransferFunction:
    """Read one site's transfer functions from an EDI, EMTF XML, Z- or J-file.

    ``sign_convention`` applies to EDI and J-files, which do not state it (see
    :func:`read_edi`), and ``z_units`` to J-files (see :func:`read_jfile`); EMTF
    XML and Z-files state both.
    """
    source = Path(path)
    if is_edi_file(source):
        return read_edi(source, sign_convention=sign_convention)
    if is_emtf_xml_file(source):
        return read_emtf_xml(source)
    if is_zfile(source):
        return read_zfile(source)
    if is_jfile(source):
        return read_jfile(source, z_units=z_units, sign_convention=sign_convention)
    raise ValueError(f"{source.name} is not a transfer-function file this package reads "
                     "(EDI, EMTF XML, .zmm/.zrr/.zss or .j)")


def read_transfer_functions(sources: Union[str, Path, Iterable[Any]],
                            **options: Any) -> List[TransferFunction]:
    """Read every transfer-function file of a folder, or of a list of paths, in name order."""
    if isinstance(sources, (str, Path)) and Path(sources).is_dir():
        folder = Path(sources)
        paths = sorted({p for pattern in TRANSFER_FUNCTION_PATTERNS for p in folder.glob(pattern)
                        if is_transfer_function_file(p)})
    elif isinstance(sources, (str, Path)):
        paths = [Path(sources)]
    else:
        paths = [Path(p) for p in sources]
    if not paths:
        raise ValueError(f"no transfer-function files found in {sources}")
    return [read_transfer_function(p, **options) for p in paths]


def timeseries_format(path: Any) -> Optional[str]:
    """Which instrument wrote ``path`` (a file or a folder), or None."""
    source = Path(path)
    if source.is_file():
        if is_phoenix_recording(source):
            return "phoenix"
        if is_phoenix_legacy_file(source):
            return "phoenix_legacy"
        if is_ats_file(source):
            return "metronix"
        if is_z3d_file(source):
            return "zonge"
        if is_lemi424_file(source):
            return "lemi424"
        return None
    if not source.is_dir():
        return None
    if is_phoenix_recording(source):
        return "phoenix"
    names = [p for p in source.iterdir() if p.is_file()]
    if any(p.suffix.upper() == ".TBL" for p in names):
        return "phoenix_legacy"
    if any(is_z3d_file(p) for p in names):
        return "zonge"
    if any(is_lemi424_file(p) for p in names[:20]):
        return "lemi424"
    if next(source.rglob("*.ats"), None) is not None:
        return "metronix"
    return None


def read_timeseries(path: Any, **options: Any) -> List[TimeSeriesRun]:
    """Read an instrument's MT time series - a file or a recording folder - into runs.

    The format is recognized from the files: Phoenix MTU-5C/5P/8A recordings
    (:func:`read_phoenix`), legacy Phoenix MTU-5A TS/TBL sites
    (:func:`read_phoenix_legacy`), Metronix ATS (:func:`read_metronix`),
    Zonge Z3D (:func:`read_zonge`) and LEMI-424 text files
    (:func:`read_lemi424`). ``options`` go to that reader; ``format`` forces one.
    """
    source = Path(path)
    kind = options.pop("format", None) or timeseries_format(source)
    if kind == "phoenix":
        return read_phoenix(source, **options)
    if kind == "phoenix_legacy":
        if source.is_dir():
            runs: List[TimeSeriesRun] = []
            for table in sorted({p for p in source.iterdir() if p.suffix.upper() == ".TBL"}):
                runs += read_phoenix_legacy(table, **options)
            return runs
        return read_phoenix_legacy(source, **options)
    if kind == "metronix":
        return read_metronix(source, **options)
    if kind == "zonge":
        return read_zonge(source, **options)
    if kind == "lemi424":
        return read_lemi424(source, **options)
    raise ValueError(f"{source} is not a time-series recording this package reads "
                     f"({', '.join(TIMESERIES_FORMATS)})")
