# PyHydroGeophysX/core/data_processing/ert_data_agent.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, NamedTuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import warnings

# =============================================================================
# ERT Data Loading with Fallback Parsers
# =============================================================================
# Primary: RESIPY (if available and working)
# Fallback: Embedded parsers (modified from ResIPy) + PyGIMLi
#
# ACKNOWLEDGEMENT & LICENSE
# -------------------------
# The embedded parser functions below are MODIFIED from the ResIPy project:
#   - Original Source: https://gitlab.com/hkex/resipy
#   - Original Authors: Guillaume Blanchy, Jimmy Boyd, Sina Saneiyan, Pedro Concha,
#                       and the ResIPy development team
#   - Original License: GPL-3.0 (GNU General Public License v3.0)
#
# These parsers are embedded here as a fallback when the full ResIPy package
# cannot be installed (e.g., due to C extension compilation issues on cloud
# platforms). All credit for the original parsing algorithms goes to the
# ResIPy developers. Modifications made for PyHydroGeophysX integration.
#
# For the full ResIPy package with meshing, inversion, and visualization:
#   pip install resipy
#   Website: https://gitlab.com/hkex/resipy
#   Documentation: https://hkex.gitlab.io/resipy/
# =============================================================================

import re
import struct

_RESIPY_ERROR = None
_HAS_RESIPY = False
_HAS_EMBEDDED_PARSERS = True  # Always available since embedded
_HAS_PYGIMLI = False

# Try full resipy package first
try:
    from resipy import Project
    _HAS_RESIPY = True
    print("[PyHydroGeophysX] RESIPY loaded successfully")
except Exception as e:
    _HAS_RESIPY = False
    _RESIPY_ERROR = str(e)
    print(f"[PyHydroGeophysX] RESIPY import failed: {e}, using embedded parsers (modified from ResIPy)")

# Try pygimli as additional fallback
try:
    import pygimli as pg
    from pygimli.physics import ert as pgert
    _HAS_PYGIMLI = True
    print("[PyHydroGeophysX] PyGIMLi loaded successfully")
except Exception as e:
    print(f"[PyHydroGeophysX] PyGIMLi import failed: {e}")


# =============================================================================
# EMBEDDED PARSERS - Modified from ResIPy (GPL-3.0)
# Original authors: Guillaume Blanchy, Jimmy Boyd, Sina Saneiyan, Pedro Concha
# =============================================================================

def _geom_fac(C1, C2, P1, P2):
    """
    Compute geometric factor for apparent resistivity calculation.
    Modified from ResIPy project (GPL-3.0).
    """
    Rc1p1 = np.abs(C1 - P1)
    Rc2p1 = np.abs(C2 - P1)
    Rc1p2 = np.abs(C1 - P2)
    Rc2p2 = np.abs(C2 - P2)
    
    # Avoid division by zero
    Rc1p1 = np.where(Rc1p1 == 0, 1e-10, Rc1p1)
    Rc2p1 = np.where(Rc2p1 == 0, 1e-10, Rc2p1)
    Rc1p2 = np.where(Rc1p2 == 0, 1e-10, Rc1p2)
    Rc2p2 = np.where(Rc2p2 == 0, 1e-10, Rc2p2)
    
    denom = (1/Rc1p1) - (1/Rc2p1) - (1/Rc1p2) + (1/Rc2p2)
    denom = np.where(denom == 0, 1e-10, denom)
    k = (2*np.pi)/denom
    return k


def _bertParser(fname):
    """
    Parse BERT/Unified Data Format (.ohm, .dat files).
    Modified from ResIPy project (GPL-3.0).
    Original authors: Guillaume Blanchy, Jimmy Boyd, et al.
    """
    with open(fname, "r") as f:
        dump = f.readlines()
    
    line = 0
    
    # Skip comment lines
    while line < len(dump) and dump[line].strip().startswith('#'):
        line += 1
    
    if line >= len(dump):
        raise ValueError("File appears to be empty or only contains comments")
    
    numStr = r'[-+]?\d*\.\d*[eE]?[-+]?\d+|\d+'
    
    # Check for number of electrodes line
    numElec = re.findall(numStr, dump[line])
    if len(numElec) == 1:
        line += 1
    
    # Check for electrode headers
    if line < len(dump):
        elecHeaders = re.findall(r'#|x|y|z', dump[line])
        if len(elecHeaders) != 0:
            line += 1
    
    # Read electrode positions
    elec_list = []
    if line < len(dump):
        elecLocs0 = re.findall(numStr, dump[line].split('#')[0])
        elecLocs_line = elecLocs0.copy()
        while len(elecLocs_line) == len(elecLocs0) and len(elecLocs_line) > 0:
            elecLine_input_raw = dump[line].split('#')[0]
            elecLocs_line = re.findall(numStr, elecLine_input_raw)
            if len(elecLocs_line) == len(elecLocs0):
                elec_list.append(elecLocs_line)
            line += 1
            if line >= len(dump):
                break
    
    if not elec_list:
        raise ValueError("Could not parse electrode positions")
    
    elec = np.array(elec_list).astype(float)
    
    # Ensure 3D coordinates (x, y, z)
    if elec.shape[1] < 3:
        if elec.shape[1] == 2:
            elec = np.c_[elec[:, 0], np.zeros(len(elec)), elec[:, 1]]
        else:
            elec = np.c_[elec[:, 0], np.zeros(len(elec)), np.zeros(len(elec))]
    
    # Find data section
    while line < len(dump):
        vals = re.findall(numStr, dump[line].split('#')[0])
        if len(vals) >= 4:
            break
        line += 1
    
    if line >= len(dump):
        raise ValueError("Could not find data section")
    
    # Get headers from line before data
    headers = re.findall(r'[A-Za-z\/]+', dump[line-1]) if line > 0 else ['a', 'b', 'm', 'n', 'r']
    
    # Read data
    df_list = []
    for val_line in dump[line:]:
        vals = re.findall(numStr, val_line.split('#')[0])
        if len(vals) < 4:
            break
        df_list.append([float(v) for v in vals])
    
    if not df_list:
        raise ValueError("Could not parse measurement data")
    
    # Create DataFrame
    df = pd.DataFrame(df_list)
    
    # Assign column names
    if len(headers) >= len(df.columns):
        df.columns = headers[:len(df.columns)]
    else:
        default_headers = ['a', 'b', 'm', 'n', 'r', 'err', 'ip'][:len(df.columns)]
        df.columns = default_headers
    
    # Standardize column names
    col_map = {
        'r': 'resist', 'R': 'resist', 'rho': 'resist', 'Rho': 'resist',
        'rhoa': 'app', 'Rhoa': 'app', 'Ra': 'app',
        'err': 'dev', 'ERR': 'dev', 'Err': 'dev',
        'ip': 'ip', 'IP': 'ip', 'M': 'ip'
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    
    # Ensure ABMN are integers
    for col in ['a', 'b', 'm', 'n']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Add IP column if missing
    if 'ip' not in df.columns:
        df['ip'] = np.nan
    
    return elec, df


def _syscalParser(fname):
    """
    Parse Syscal format (CSV from IRIS Instruments).
    Modified from ResIPy project (GPL-3.0).
    Original authors: Guillaume Blanchy, Jimmy Boyd, et al.
    """
    df = pd.read_csv(fname, skipinitialspace=True, engine='python', encoding_errors='ignore')
    headers = df.columns
    
    # Standardize column names based on format version
    if 'Spa.1' in headers:
        df = df.rename(columns={
            'Spa.1': 'a', 'Spa.2': 'b', 'Spa.3': 'm', 'Spa.4': 'n',
            'In': 'i', 'Vp': 'vp', 'Dev.': 'dev', 'M': 'ip', 'Sp': 'sp'
        })
    elif 'xA(m)' in headers or 'xA (m)' in headers:
        rename_map = {
            'xA(m)': 'a', 'xB(m)': 'b', 'xM(m)': 'm', 'xN(m)': 'n',
            'xA (m)': 'a', 'xB (m)': 'b', 'xM (m)': 'm', 'xN (m)': 'n',
            'Dev.': 'dev', 'Dev. Rho (%)': 'dev',
            'M (mV/V)': 'ip', 'SP (mV)': 'sp',
            'VMN (mV)': 'vp', 'IAB (mA)': 'i',
            'yA (m)': 'ya', 'yB (m)': 'yb', 'yM (m)': 'ym', 'yN (m)': 'yn',
            'zA (m)': 'za', 'zB (m)': 'zb', 'zM (m)': 'zm', 'zN (m)': 'zn'
        }
        df = df.rename(columns=rename_map)
    
    # Calculate resistance
    if 'vp' in df.columns and 'i' in df.columns:
        df['resist'] = df['vp'] / df['i']
    
    # Process electrode positions
    if 'ya' in df.columns:  # 3D format
        xarray = df[['a', 'b', 'm', 'n']].values.flatten() if 'xa' not in df.columns else df[['xa', 'xb', 'xm', 'xn']].values.flatten()
        yarray = df[['ya', 'yb', 'ym', 'yn']].values.flatten()
        zarray = df[['za', 'zb', 'zm', 'zn']].values.flatten() if 'za' in df.columns else np.zeros_like(xarray)
        arrayFull = np.c_[xarray, yarray, zarray]
        elec = np.unique(arrayFull, axis=0)
    else:  # 2D format
        array = df[['a', 'b', 'm', 'n']].values
        val = np.sort(np.unique(array.flatten()))
        elecLabel = 1 + np.arange(len(val))
        searchsortedArr = np.searchsorted(val, array)
        newval = elecLabel[searchsortedArr]
        df[['a', 'b', 'm', 'n']] = newval
        
        zval = np.zeros_like(val)
        if 'za' in df.columns:
            zarray = df[['za', 'zb', 'zm', 'zn']].values
            zvalflat = np.c_[searchsortedArr.flatten(), zarray.flatten()]
            zval = np.unique(zvalflat[zvalflat[:, 0].argsort()], axis=0)[:, 1]
        
        elec = np.c_[val, np.zeros_like(val), zval]
    
    if 'ip' not in df.columns:
        df['ip'] = np.nan
    
    return elec, df


def _protocolParser(fname, ip=False):
    """
    Parse Protocol format (from Lund Imaging System).
    Modified from ResIPy project (GPL-3.0).
    Original authors: Guillaume Blanchy, Jimmy Boyd, et al.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith('#') and not line.startswith('*'):
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 4:
                    data_start = i
                    break
            except ValueError:
                continue
    
    # Parse data
    data_list = []
    for line in lines[data_start:]:
        if line.strip() and not line.startswith('#'):
            try:
                vals = [float(x) for x in line.split()]
                if len(vals) >= 4:
                    data_list.append(vals)
            except ValueError:
                continue
    
    if not data_list:
        raise ValueError("Could not parse data from file")
    
    # Create DataFrame
    df = pd.DataFrame(data_list)
    
    # Assign columns based on number of values
    ncols = df.shape[1]
    if ncols >= 6:
        df.columns = ['a', 'b', 'm', 'n', 'resist', 'dev'][:ncols]
    elif ncols >= 5:
        df.columns = ['a', 'b', 'm', 'n', 'resist'][:ncols]
    else:
        df.columns = ['a', 'b', 'm', 'n'][:ncols]
    
    # Convert ABMN to int
    for col in ['a', 'b', 'm', 'n']:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    # Build electrode array
    array = df[['a', 'b', 'm', 'n']].values
    unique_elec = np.sort(np.unique(array.flatten()))
    n_elec = len(unique_elec)
    
    # Assume regular spacing
    if n_elec > 1:
        spacing = np.min(np.diff(unique_elec))
        elec = np.c_[unique_elec, np.zeros(n_elec), np.zeros(n_elec)]
    else:
        elec = np.array([[0, 0, 0]])
    
    if 'ip' not in df.columns:
        df['ip'] = np.nan
    
    return elec, df


# ---------------------------
# Types and schemas
# ---------------------------
Instrument = Literal[
    "Protocol DC", "Syscal", "Protocol IP", "ResInv", "PRIME/RESIMGR",
    "Sting", "ABEM-Lund", "Lippmann", "ARES", "BERT", "E4D",
    "DAS-1", "Electra", "Custom", "Merged"
]

class LocalRef(NamedTuple):
    origin_x: float = 0.0   # optional world X of profile start
    origin_y: float = 0.0   # optional world Y of profile start
    azimuth_deg: float = 0.0  # profile direction (deg clockwise from north)

@dataclass
class Electrode:
    id: int
    x: float
    y: float = 0.0
    z: float = 0.0

@dataclass
class Quadruplet:
    A: int; B: int; M: int; N: int

@dataclass
class Observation:
    quad: Quadruplet
    app_res: float | None = None   # apparent resistivity (ohm·m)
    dV: float | None = None        # potential difference (V)
    I: float | None = None         # injected current (A)
    rel_err: float | None = 0.03   # relative error fraction (e.g., 0.03)
    K: float | None = None         # geometric factor
    fid: str | None = None         # field id/record id

@dataclass
class StandardERT:
    # "local" or "EPSG:xxxx"
    crs: str = "local"
    instrument: str = "Syscal"
    electrodes: List[Electrode] = None
    observations: List[Observation] = None
    metadata: Dict[str, Any] = None  # may include epsg:int, local_ref:dict

    def to_json(self, path: str | Path):
        """Export ERT dataset to a standardized JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "crs": self.crs,
            "instrument": self.instrument,
            "electrodes": [asdict(e) for e in self.electrodes],
            "observations": [{
                "A": o.quad.A, "B": o.quad.B, "M": o.quad.M, "N": o.quad.N,
                "app_res": o.app_res, "dV": o.dV, "I": o.I,
                "rel_err": o.rel_err, "fid": o.fid
            } for o in self.observations],
            "metadata": self.metadata or {}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)


# ---------------------------
# Instrument mapping
# ---------------------------
_FTYPE_MAP = {
    "Protocol DC": "Protocol DC",
    "Syscal": "Syscal",
    "Protocol IP": "Protocol IP",
    "ResInv": "ResInv (2D/3D)",
    "PRIME/RESIMGR": "PRIME/RESIMGR",
    "Sting": "Sting",
    "ABEM-Lund": "ABEM-Lund",
    "Lippmann": "Lippmann",
    "ARES": "ARES (beta)",
    "BERT": "BERT",
    "E4D": "E4D",
    "DAS-1": "DAS-1",
    "Electra": "Electra",
    "Custom": "Custom",
    "Merged": "Merged",
}

def _to_ftype(instrument: Instrument) -> str:
    if instrument not in _FTYPE_MAP:
        raise ValueError(f"Unsupported instrument: {instrument}")
    return _FTYPE_MAP[instrument]


# Parser mapping for embedded parsers (modified from ResIPy)
_EMBEDDED_PARSER_MAP = {
    "Syscal": _syscalParser,
    "Protocol DC": _protocolParser,
    "Protocol IP": _protocolParser,
    "BERT": _bertParser,
    "E4D": _bertParser,
    "DAS-1": _protocolParser,  # DAS-1 uses protocol format
    "ABEM-Lund": _bertParser,  # Use BERT parser as fallback
    "Lippmann": _bertParser,   # Use BERT parser as fallback
    "ARES": _bertParser,       # Use BERT parser as fallback
}


# ---------------------------
# Local Parser Fallback Loader (adapted from ResIPy)
# ---------------------------
def _load_ert_embedded_parsers(
    data_file: str,
    electrode_file: Optional[str] = None,
    project_dir: str = ".",
    instrument: Instrument = "BERT",
    crs: str = "local",
    epsg: Optional[int] = None,
    local_ref: Optional[LocalRef] = None
) -> "StandardERT":
    """
    ERT data loader using embedded parsers (modified from ResIPy).
    Used when the full ResIPy package cannot be installed.
    
    ACKNOWLEDGEMENT & LICENSE
    -------------------------
    Parser functions are MODIFIED from the ResIPy project:
    - Original Source: https://gitlab.com/hkex/resipy
    - Original License: GPL-3.0 (GNU General Public License v3.0)
    - Original Authors: Guillaume Blanchy, Jimmy Boyd, Sina Saneiyan, Pedro Concha
    
    All credit for original parsing algorithms goes to the ResIPy developers.
    """
    data_file_path = Path(data_file)
    if not data_file_path.is_absolute():
        data_file_path = Path.cwd() / data_file_path
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    # Select appropriate parser based on instrument
    parser_func = _EMBEDDED_PARSER_MAP.get(instrument, _bertParser)
    
    if parser_func is None:
        parser_func = _bertParser  # Default fallback
    
    # Parse the data file
    try:
        elec_array, df = parser_func(str(data_file_path))
    except Exception as e:
        raise ValueError(f"Failed to parse ERT data with {parser_name}: {e}")
    
    # Build electrodes dataframe
    if isinstance(elec_array, np.ndarray):
        if elec_array.ndim == 1:
            elec_array = elec_array.reshape(-1, 1)
        n_cols = elec_array.shape[1]
        electrodes = pd.DataFrame({
            'x': elec_array[:, 0],
            'y': elec_array[:, 1] if n_cols > 1 else 0.0,
            'z': elec_array[:, 2] if n_cols > 2 else 0.0,
        })
    else:
        # Already a DataFrame
        electrodes = pd.DataFrame({
            'x': elec_array['x'] if 'x' in elec_array.columns else elec_array.iloc[:, 0],
            'y': elec_array['y'] if 'y' in elec_array.columns else 0.0,
            'z': elec_array['z'] if 'z' in elec_array.columns else 0.0,
        })
    
    # Build observations dataframe with standardized columns
    observations = pd.DataFrame()
    observations['a'] = df['a'].astype(int) if 'a' in df.columns else 1
    observations['b'] = df['b'].astype(int) if 'b' in df.columns else 2
    observations['m'] = df['m'].astype(int) if 'm' in df.columns else 3
    observations['n'] = df['n'].astype(int) if 'n' in df.columns else 4
    
    # Get apparent resistivity
    if 'app' in df.columns:
        observations['rhoa'] = df['app']
    elif 'resist' in df.columns:
        # Calculate geometric factor and convert
        observations['rhoa'] = df['resist'] * 1.0  # Simplified
    else:
        observations['rhoa'] = 100.0  # Default
    
    # Get error/dev
    if 'dev' in df.columns:
        observations['error'] = df['dev'] / 100.0  # Convert percentage to fraction
    else:
        observations['error'] = 0.05  # Default 5%
    
    observations['valid'] = True
    
    # Build metadata
    metadata = {
        'source_file': str(data_file_path),
        'loader': 'local_parsers_resipy_fallback',
        'parser_used': parser_name,
        'instrument': instrument,
        'n_electrodes': len(electrodes),
        'n_measurements': len(observations),
        'acknowledgement': 'Parsing logic adapted from ResIPy (https://gitlab.com/hkex/resipy) under GPL-3.0',
    }
    
    if local_ref is not None:
        metadata['local_origin_x'] = local_ref.origin_x
        metadata['local_origin_y'] = local_ref.origin_y
        metadata['azimuth_deg'] = local_ref.azimuth_deg
    
    if epsg is not None:
        metadata['epsg'] = epsg
    
    return StandardERT(
        electrodes=electrodes,
        observations=observations,
        crs=crs,
        instrument=instrument,
        metadata=metadata
    )


# ---------------------------
# PyGIMLi Fallback Loader
# ---------------------------
def _load_ert_pygimli(
    data_file: str,
    electrode_file: Optional[str] = None,
    project_dir: str = ".",
    instrument: Instrument = "BERT",
    crs: str = "local",
    epsg: Optional[int] = None,
    local_ref: Optional[LocalRef] = None
) -> "StandardERT":
    """
    Fallback ERT data loader using PyGIMLi when RESIPY is unavailable.
    Supports common formats: .ohm, .dat, .data files.
    """
    import pygimli as pg
    
    data_file_path = Path(data_file)
    if not data_file_path.is_absolute():
        data_file_path = Path.cwd() / data_file_path
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")
    
    # Load data with pygimli
    try:
        data = pg.load(str(data_file_path))
    except Exception as e:
        # Try loading as unified data format
        try:
            from pygimli.physics import ert as pgert
            data = pgert.load(str(data_file_path))
        except Exception:
            raise ValueError(f"Could not load ERT data file with PyGIMLi: {e}")
    
    # Extract electrode positions
    if hasattr(data, 'sensorPositions') and callable(data.sensorPositions):
        sensors = np.array(data.sensorPositions())
    elif hasattr(data, 'sensors'):
        sensors = np.array(data.sensors())
    else:
        # Try to get from electrodes
        sensors = np.array([[i, 0, 0] for i in range(data.size())])
    
    # Ensure 3D coordinates
    if sensors.ndim == 1:
        sensors = sensors.reshape(-1, 1)
    if sensors.shape[1] == 1:
        sensors = np.hstack([sensors, np.zeros((len(sensors), 2))])
    elif sensors.shape[1] == 2:
        sensors = np.hstack([sensors, np.zeros((len(sensors), 1))])
    
    # Build electrodes dataframe
    electrodes = pd.DataFrame({
        'x': sensors[:, 0],
        'y': sensors[:, 1] if sensors.shape[1] > 1 else 0.0,
        'z': sensors[:, 2] if sensors.shape[1] > 2 else 0.0,
    })
    
    # Extract measurements
    n_data = data.size()
    
    # Get electrode indices (a, b, m, n)
    a = np.array(data('a')) if 'a' in data.dataMap() else np.zeros(n_data, dtype=int)
    b = np.array(data('b')) if 'b' in data.dataMap() else np.zeros(n_data, dtype=int)
    m = np.array(data('m')) if 'm' in data.dataMap() else np.zeros(n_data, dtype=int)
    n = np.array(data('n')) if 'n' in data.dataMap() else np.zeros(n_data, dtype=int)
    
    # Get apparent resistivity or resistance
    if 'rhoa' in data.dataMap():
        rhoa = np.array(data('rhoa'))
    elif 'r' in data.dataMap():
        # Convert resistance to apparent resistivity using geometric factor
        r = np.array(data('r'))
        if 'k' in data.dataMap():
            k = np.array(data('k'))
            rhoa = r * k
        else:
            rhoa = r  # Use resistance as proxy
    else:
        rhoa = np.ones(n_data) * 100  # Default value
    
    # Get error if available
    if 'err' in data.dataMap():
        error = np.array(data('err'))
    elif 'error' in data.dataMap():
        error = np.array(data('error'))
    else:
        error = np.ones(n_data) * 0.05  # Default 5% error
    
    # Build observations dataframe
    observations = pd.DataFrame({
        'a': a.astype(int),
        'b': b.astype(int),
        'm': m.astype(int),
        'n': n.astype(int),
        'rhoa': rhoa,
        'error': error,
        'valid': np.ones(n_data, dtype=bool)
    })
    
    # Build metadata
    metadata = {
        'source_file': str(data_file_path),
        'loader': 'pygimli_fallback',
        'instrument': instrument,
        'n_electrodes': len(electrodes),
        'n_measurements': len(observations),
    }
    
    if local_ref is not None:
        metadata['local_origin_x'] = local_ref.origin_x
        metadata['local_origin_y'] = local_ref.origin_y
        metadata['azimuth_deg'] = local_ref.azimuth_deg
    
    if epsg is not None:
        metadata['epsg'] = epsg
    
    return StandardERT(
        electrodes=electrodes,
        observations=observations,
        crs=crs,
        instrument=instrument,
        metadata=metadata
    )


# ---------------------------
# Loader
# ---------------------------
def load_ert_resipy(
    project_dir: str,
    data_file: str,
    instrument: Instrument,
    spacing: Optional[float] = None,
    electrode_file: Optional[str] = None,
    crs: str = "local",
    epsg: Optional[int] = None,
    local_ref: Optional[LocalRef] = None
) -> StandardERT:
    """
    Load ERT data using RESIPY with an explicit instrument type, apply light QC,
    and return a standardized dataset.

    Parameters
    ----------
    project_dir : str
        RESIPY project folder (created if not exists).
    data_file : str
        Path to raw ERT data file exported from the instrument/software.
    instrument : Instrument
        One of the supported instrument types (see Instrument Literal).
    spacing : float, optional
        If electrodes are missing, create an evenly spaced line with this spacing (meters).
        Use only for quick demos; prefer real surveyed coordinates when available.
    electrode_file : str, optional
        Path to external electrode coordinate file. If provided, electrode positions from this
        file will be used instead of those in the data file. Format: space-separated x y z columns.
    crs : str
        "local" (default) for profile coordinates, or "EPSG:xxxx" for projected coords.
    epsg : int, optional
        EPSG code (e.g., 32615). If provided with crs != "local", metadata will include it.
    local_ref : LocalRef, optional
        Optional origin and azimuth metadata for local profiles.

    Returns
    -------
    StandardERT
        Standardized dataset with electrodes, observations, CRS, instrument, and metadata.
    """
    # Try resipy first, then local parsers, then pygimli
    if not _HAS_RESIPY:
        # Fallback 1: Embedded parsers (modified from ResIPy, GPL-3.0)
        if _HAS_EMBEDDED_PARSERS:
            print(f"[PyHydroGeophysX] Using embedded parsers (modified from ResIPy) - RESIPY unavailable: {_RESIPY_ERROR}")
            return _load_ert_embedded_parsers(
                data_file=data_file,
                electrode_file=electrode_file,
                project_dir=project_dir,
                instrument=instrument,
                crs=crs,
                epsg=epsg,
                local_ref=local_ref
            )
        # Fallback 2: PyGIMLi
        elif _HAS_PYGIMLI:
            print(f"[PyHydroGeophysX] Using PyGIMLi fallback for ERT data loading (RESIPY unavailable: {_RESIPY_ERROR})")
            return _load_ert_pygimli(
                data_file=data_file,
                electrode_file=electrode_file,
                project_dir=project_dir,
                instrument=instrument,
                crs=crs,
                epsg=epsg,
                local_ref=local_ref
            )
        else:
            error_msg = f"RESIPY import failed: {_RESIPY_ERROR}" if _RESIPY_ERROR else "RESIPY not installed. Please `pip install resipy`."
            raise ImportError(error_msg + " Local parsers and PyGIMLi fallbacks also unavailable.")
    ftype = _to_ftype(instrument)

    # Resolve relative paths to absolute paths based on current working directory
    data_file_path = Path(data_file)
    if not data_file_path.is_absolute():
        data_file_path = Path.cwd() / data_file_path
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    # Prefer to use the requested project_dir, but RESIPY may attempt to
    # remove/recreate the directory (calling shutil.rmtree) which can fail
    # on Windows (OneDrive or open handles). Try to instantiate Project and
    # if a PermissionError (or OSError with permission) occurs, fall back to
    # a temporary directory and warn the user.
    chosen_dir = project_dir
    try:
        prj = Project(chosen_dir)
    except PermissionError:
        warnings.warn(
            f"RESIPY cannot prepare project directory '{project_dir}' (PermissionError). "
            "Falling back to a temporary project directory.",
            UserWarning
        )
        chosen_dir = tempfile.mkdtemp(prefix="resipy_")
        prj = Project(chosen_dir)
    except OSError as e:
        # On some platforms, an OSError may be raised for permission issues
        if getattr(e, 'winerror', None) == 5 or e.errno in (13,):
            warnings.warn(
                f"RESIPY cannot prepare project directory '{project_dir}' ({e}). "
                "Falling back to a temporary project directory.",
                UserWarning
            )
            chosen_dir = tempfile.mkdtemp(prefix="resipy_")
            prj = Project(chosen_dir)
        else:
            raise
    # Ensure the chosen folder exists on disk first
    Path(chosen_dir).mkdir(parents=True, exist_ok=True)
    # RESIPY's Project API has changed over time; some releases expose
    # `createFolder`, others `create_folder` or similar. Try common names
    # and call the first that exists. If none exists, that's fine because
    # the filesystem folder was already created above.
    for _fn in ("createFolder", "create_folder", "createProject", "create_project", "create"):
        if hasattr(prj, _fn):
            try:
                getattr(prj, _fn)(chosen_dir)
            except TypeError:
                # Some implementations may not accept an argument; call without args
                try:
                    getattr(prj, _fn)()
                except Exception:
                    # If it still fails, ignore and continue — folder is present.
                    pass
            except Exception:
                # Be tolerant of any other runtime errors from RESIPY initialization
                # so that our loader remains usable across versions.
                pass
            break

    # Use explicit ftype for robust parsing. RESIPY's Project class uses 
    # method name 'createSurvey' to load data files and create a survey object.
    try:
        prj.createSurvey(fname=str(data_file_path), ftype=ftype)
    except ValueError as e:
        # Check for NumPy compatibility issues with pyGIMLi
        if "Buffer dtype mismatch" in str(e) or "dtype mismatch" in str(e).lower():
            raise RuntimeError(
                f"NumPy compatibility error with pyGIMLi/RESIPY: {str(e)}\n\n"
                "This is a known issue with NumPy 2.x and pyGIMLi. Try:\n"
                "  conda install numpy=1.26.4\n"
                "or:\n"
                "  pip install 'numpy<2.0'\n\n"
                "If the error persists, you may also need to reinstall pygimli:\n"
                "  conda install -c gimli -c conda-forge pygimli"
            ) from e
        else:
            raise
    
    # After createSurvey, data is stored in prj.surveys[0] (the first/only survey)
    if not prj.surveys:
        raise RuntimeError("No survey was created. Check that the data file format matches the specified instrument.")
    
    survey = prj.surveys[0]  # Get the first survey object

    # Step 1: Load raw resistance values from DAS data (before any K computation)
    # Get the raw dataframe with resistance values (R = V/I)
    df = survey.df.copy().dropna(subset=['a','b','m','n'])
    
    # Detect resistance column (raw V/I values from instrument)
    resist_col = next((c for c in ['resist', 'R', 'r', 'resistance'] if c in df.columns), None)
    if resist_col is None:
        raise RuntimeError("Cannot find resistance column in data. DAS-1 data should have 'resist' column.")
    
    print(f"   Loaded {len(df)} raw resistance measurements from {instrument}")

    # Step 2: Update electrode positions from external file
    if electrode_file is not None:
        electrode_file_path = Path(electrode_file)
        if not electrode_file_path.is_absolute():
            electrode_file_path = Path.cwd() / electrode_file_path
        
        if not electrode_file_path.exists():
            raise FileNotFoundError(f"Electrode file not found: {electrode_file_path}")
        
        # Load electrode coordinates from file
        try:
            elec_data = np.loadtxt(str(electrode_file_path))
            if elec_data.ndim == 1:
                elec_data = elec_data.reshape(-1, 3)
            
            # Set electrode positions in survey
            if hasattr(survey, 'setElec'):
                survey.setElec(elec_data)
            else:
                survey.elec = elec_data
            print(f"   Updated electrode positions from {electrode_file_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load electrode file '{electrode_file_path}': {e}") from e
    
    # If no electrode coordinates, generate a simple line for quick testing
    elif spacing is not None and (survey.elec is None or len(survey.elec) == 0):
        n_elec = int(np.max(df[['a','b','m','n']].values)) + 1
        elec = np.zeros((n_elec, 3))
        elec[:, 0] = np.arange(n_elec) * spacing
        if hasattr(survey, 'setElec'):
            survey.setElec(elec)
        else:
            survey.elec = elec

    # Step 3: Basic QC filters on raw data
    # Skip i/u filters for BERT format - these columns contain zeros in BERT/PyGIMLi format
    if instrument != 'BERT':
        if 'i' in df.columns:
            df = df[df['i'].abs() > 0]
        if 'u' in df.columns:
            df = df[df['u'].abs() > 0]
    df = df.drop_duplicates(subset=['a','b','m','n'])
    
    initial_count = len(df)
    print(f"   After basic QC filters: {initial_count} measurements")
    
    # Step 3b: Apply DAS-specific quality filters
    if instrument == 'DAS-1':
        # DAS-1 quality thresholds
        rec_threshold = 5      # max reciprocal error, %
        ctc_threshold = 30000  # max contact resistance, ohm
        stk_threshold = 20     # max stacking error, %
        v_threshold = 1E-5     # min voltage, V
        
        # Detect DAS column names (RESIPY creates standardized names)
        # reciprocalErrRel is in decimal form (0.05 = 5%), so convert to percentage
        rec_col = 'reciprocalErrRel' if 'reciprocalErrRel' in df.columns else None
        ctc_col = next((c for c in ['ContactR', 'ctc', 'contact_resistance'] if c in df.columns), None)
        stk_col = next((c for c in ['stk', 'stack_err', 'stacking_error'] if c in df.columns), None)
        v_col = next((c for c in ['u', 'U', 'v', 'V', 'voltage', 'dV'] if c in df.columns), None)
        
        # Apply reciprocal error filter (RESIPY provides this as decimal, e.g., 0.05 = 5%)
        if rec_col and rec_col in df.columns:
            before_rec = len(df)
            # Filter: keep only measurements where reciprocal error < threshold
            # reciprocalErrRel is in decimal form, so divide threshold by 100
            df = df[df[rec_col] < (rec_threshold / 100.0)]
            rec_filtered = before_rec - len(df)
            if rec_filtered > 0:
                print(f"   Applied reciprocal error filter (< {rec_threshold}%): removed {rec_filtered} measurements")
        
        # Apply other quality filters
        if ctc_col and ctc_col in df.columns:
            before_ctc = len(df)
            df = df[df[ctc_col] < ctc_threshold]
            ctc_filtered = before_ctc - len(df)
            if ctc_filtered > 0:
                print(f"   Applied contact resistance filter (< {ctc_threshold} Ω): removed {ctc_filtered} measurements")
        
        if stk_col and stk_col in df.columns:
            before_stk = len(df)
            df = df[df[stk_col] < stk_threshold]
            stk_filtered = before_stk - len(df)
            if stk_filtered > 0:
                print(f"   Applied stacking error filter (< {stk_threshold}%): removed {stk_filtered} measurements")
        
        if v_col and v_col in df.columns:
            before_v = len(df)
            df = df[df[v_col].abs() > v_threshold]
            v_filtered = before_v - len(df)
            if v_filtered > 0:
                print(f"   Applied voltage filter (> {v_threshold} V): removed {v_filtered} measurements")
        
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"   Total filtered with DAS quality thresholds: {filtered_count} measurements")
    
    print(f"   After all QC filters: {len(df)} measurements")

    # Simple relative error model - default 5%
    rel_err = np.full(len(df), 0.05)
    
    # Check for error columns (different names for different instruments)
    # Note: E4D format stores absolute errors in Ohms, need to convert to relative
    for err_col in ['resError', 'magErr', 'err', 'error', 'dev']:
        if err_col in df.columns:
            err_vals = df[err_col].values
            
            # Check if this is E4D format (absolute errors in Ohms)
            # E4D always stores absolute errors (in Ohms), need to convert to relative
            # Detection: Only apply to E4D instrument explicitly
            resist_col = next((c for c in ['resist', 'R', 'resistance'] if c in df.columns), None)
            
            if instrument == 'E4D' and resist_col and resist_col in df.columns:
                # Convert absolute error to relative error
                # For E4D: error column is absolute resistance error [Ohms]
                # Need: relative error = abs_error / resistance
                resist_vals = df[resist_col].values
                # Calculate relative error, avoiding division by zero
                rel_err_calc = np.where(
                    (np.isfinite(err_vals)) & (np.isfinite(resist_vals)) & (resist_vals != 0),
                    np.abs(err_vals / resist_vals),
                    0.05
                )
                # Clamp to reasonable range (0.5% to 50%)
                rel_err = np.clip(rel_err_calc, 0.005, 0.5)
                print(f"   Converted E4D absolute errors to relative (mean: {np.mean(rel_err):.4f})")
            else:
                # Assume already relative error (e.g., Syscal, ABEM formats)
                # Use the error if it's reasonable (between 0.5% and 50%)
                valid_err = np.where(
                    np.isfinite(err_vals) & (err_vals > 0.005) & (err_vals < 0.5), 
                    err_vals, 
                    0.05
                )
                rel_err = np.maximum(rel_err, valid_err)
            break

    # Electrodes - access from survey object
    elec_arr = np.array(survey.elec) if survey.elec is not None else \
               np.zeros((int(df[['a','b','m','n']].values.max())+1, 3))
    electrodes = [Electrode(i+1, float(elec_arr[i,0]), float(elec_arr[i,1]), float(elec_arr[i,2]))
                  for i in range(elec_arr.shape[0])]

    # Observations
    idx_to_pos = {idx: k for k, idx in enumerate(df.index)}
    observations: List[Observation] = []
    
    # Detect column names
    i_col = next((c for c in ['i', 'I', 'current'] if c in df.columns), None)
    v_col = next((c for c in ['u', 'U', 'v', 'V', 'voltage', 'dV'] if c in df.columns), None)
    r_col = next((c for c in ['resist', 'R', 'r', 'resistance'] if c in df.columns), None)
    # BERT/PyGIMLi format uses 'rhoa' for apparent resistivity directly
    rhoa_col = next((c for c in ['rhoa', 'rhoA', 'Rhoa', 'app_res'] if c in df.columns), None)
    
    for idx, row in df.iterrows():
        app_resistivity = None
        k_value = 1.0
        
        # For BERT format, rhoa column contains apparent resistivity directly
        if rhoa_col and rhoa_col in row and np.isfinite(row[rhoa_col]):
            app_resistivity = float(row[rhoa_col])
            # If K column exists, use it
            if 'k' in row and np.isfinite(row['k']):
                k_value = float(row['k'])
        # Get raw resistance (V/I from instrument) for other formats
        elif r_col and r_col in row and np.isfinite(row[r_col]):
            app_resistivity = float(row[r_col])
        elif v_col and i_col and v_col in row and i_col in row:
            # Calculate from V and I if resistance column not available
            if row[i_col] != 0:
                app_resistivity = float(row[v_col] / row[i_col])
            else:
                continue  # Skip if current is zero
        
        if app_resistivity is None:
            continue  # Skip if no resistance/rhoa data
        
        observations.append(Observation(
            quad=Quadruplet(int(row['a']), int(row['b']), int(row['m']), int(row['n'])),
            app_res=app_resistivity,  # Apparent resistivity (or raw resistance with k=1)
            dV=float(row[v_col]) if v_col and v_col in row and np.isfinite(row[v_col]) else None,
            I=float(row[i_col]) if i_col and i_col in row and np.isfinite(row[i_col]) else None,
            rel_err=float(rel_err[idx_to_pos[idx]]),
            K=k_value,  # Geometric factor if available
            fid=str(row['id']) if 'id' in row else str(idx)
        ))

    crs_out = ("EPSG:%d" % epsg) if (crs != "local" and epsg) else crs
    meta = {
        "loader": "RESIPY",
        "ftype": ftype,
        "project_dir": str(Path(chosen_dir).resolve()),
        "data_file": str(data_file_path.resolve()),
        "electrode_file": str(Path(electrode_file).resolve()) if electrode_file else None,
        "epsg": epsg,
        "local_ref": (local_ref._asdict() if isinstance(local_ref, LocalRef) else None)
    }

    return StandardERT(
        crs=crs_out,
        instrument=instrument,
        electrodes=electrodes,
        observations=observations,
        metadata=meta
    )


# ---------------------------
# Diagnostics and export
# ---------------------------
def qc_and_visualize(ert: StandardERT, outdir: str = "examples/results/ert") -> Dict[str, str]:
    """
    Create basic diagnostics and export normalized artifacts:
    - electrodes plot
    - histogram of log10 apparent resistivity
    - observations parquet, electrodes CSV, standardized JSON
    """
    # Handle paths starting with / on Windows by converting to relative path
    outdir_path = Path(outdir)
    if outdir.startswith('/') and not outdir_path.is_absolute():
        # Remove leading / and treat as relative to cwd
        outdir_path = Path.cwd() / outdir.lstrip('/')
    
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Electrodes plot
    ex = [e.x for e in ert.electrodes]
    ez = [e.z for e in ert.electrodes]
    plt.figure(figsize=(6, 2))
    plt.plot(ex, ez, 'k.-')
    plt.xlabel('x (m)'); plt.ylabel('z (m)')
    p1 = str(outdir_path / "electrodes.png")
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Apparent resistivity histogram
    vals = [o.app_res for o in ert.observations if o.app_res is not None and np.isfinite(o.app_res)]
    plt.figure(figsize=(4, 3))
    plt.hist(np.log10(vals), bins=40, color="#4C72B0")
    plt.xlabel('log10 apparent resistivity'); plt.ylabel('count')
    p2 = str(outdir_path / "rhoa_hist.png")
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    # Flat tables
    obs_rows = [{
        "A": o.quad.A, "B": o.quad.B, "M": o.quad.M, "N": o.quad.N,
        "app_res": o.app_res, "dV": o.dV, "I": o.I,
        "rel_err": o.rel_err, "fid": o.fid
    } for o in ert.observations]
    pd.DataFrame(obs_rows).to_parquet(outdir_path / "observations.parquet", index=False)
    pd.DataFrame([asdict(e) for e in ert.electrodes]).to_csv(outdir_path / "electrodes.csv", index=False)
    ert.to_json(outdir_path / "ert_standard.json")

    return {
        "electrodes_png": str(outdir_path/"electrodes.png"),
        "rhoa_hist_png": str(outdir_path/"rhoa_hist.png"),
        "observations_parquet": str(outdir_path/"observations.parquet"),
        "electrodes_csv": str(outdir_path/"electrodes.csv"),
        "standard_json": str(outdir_path/"ert_standard.json"),
    }

def export_for_inversion(ert: StandardERT, outdir: str = "examples/results/ert", fmt: str = "pgimli") -> str:
    """
    Export to inversion-ready formats:
    - fmt='pgimli': Unified data format for pyGIMLi/BERT with electrode coordinates and measurements
    - fmt='resipy': return the RESIPY project directory for running prj.start().
    """
    # Handle paths starting with / on Windows by converting to relative path
    outdir_str = str(outdir)  # Convert to string first
    outdir_path = Path(outdir)
    if outdir_str.startswith('/') and not outdir_path.is_absolute():
        # Remove leading / and treat as relative to cwd
        outdir_path = Path.cwd() / outdir_str.lstrip('/')
    
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    if fmt == "pgimli":
        path = outdir_path / "bert_data.dat"
        
        with open(path, 'w', encoding='utf-8') as f:
            # Write number of electrodes
            f.write(f"{len(ert.electrodes)}\n")
            
            # Write electrode coordinates header and data
            f.write("# x y z\n")
            for elec in ert.electrodes:
                f.write(f"{elec.x} {elec.y} {elec.z}\n")
            
            # Write number of measurements
            f.write(f"{len(ert.observations)}\n")
            
            # Check if error data exists (use rel_err attribute)
            has_error_data = any(obs.rel_err is not None and obs.rel_err > 0 for obs in ert.observations)
            
            # Write measurement data header (matching reference format)
            if has_error_data:
                f.write("# a b m n r rhoa k err\n")
            else:
                f.write("# a b m n r rhoa k\n")
            
            # Write measurement data
            # Format: a b m n r rhoa k [err]
            # r = raw resistance (V/I from instrument)
            # rhoa = apparent resistivity (r * K)
            # k = geometric factor
            # err = relative error (optional)
            for obs in ert.observations:
                # Determine if we have true apparent resistivity or raw resistance
                # For BERT/PyGIMLi format: obs.K > 1 means app_res is already apparent resistivity
                # For DAS-1 and others: obs.K == 1 means app_res is raw resistance
                
                # Extract raw resistance from voltage and current if available
                if obs.I is not None and obs.dV is not None and obs.I != 0:
                    R = abs(obs.dV / obs.I)
                    rhoa = R  # Will be recomputed below
                    k = 1  # PyGIMLi will compute K
                elif obs.app_res is not None:
                    # Check if K was stored (BERT format has K > 1)
                    if obs.K is not None and obs.K > 1:
                        # app_res is already apparent resistivity (rhoa = R * K)
                        # Compute R by dividing by K
                        R = obs.app_res / obs.K
                        rhoa = obs.app_res  # Keep original rhoa
                        k = obs.K  # Keep original K
                    else:
                        # app_res is raw resistance (K was 1 or not stored)
                        R = obs.app_res
                        rhoa = R  # Will be recomputed by PyGIMLi
                        k = 1  # PyGIMLi will compute K
                else:
                    # Skip measurements without valid data
                    continue
                
                # Format: a b m n r rhoa k [err]
                # Note: obs.quad uses 1-based electrode indices (matching PyGIMLi format)
                f.write(f"{obs.quad.A} {obs.quad.B} {obs.quad.M} {obs.quad.N} ")
                if has_error_data:
                    err_val = obs.rel_err if (obs.rel_err is not None and obs.rel_err > 0) else 0.05  # Default 5% error
                    f.write(f"{R} {rhoa} {k} {err_val}\n")
                else:
                    f.write(f"{R} {rhoa} {k}\n")
        
        print(f"   Exported data to {path}")
        
        # Step 4: Load and validate the exported data
        # Only recompute K if it was not already provided (k=1 placeholder)
        print("   Validating geometric factors...")
        try:
            import pygimli as pg
            import pygimli.physics.ert as ert_pg
            
            # Load the file we just created
            data = ert_pg.load(str(path))
            
            # Check if K was already provided (not all k=1)
            k_vals = np.array(data['k'])
            has_valid_k = np.any(k_vals > 1.5)  # If any K > 1.5, assume K was provided
            
            if has_valid_k:
                print(f"   K factors already provided (range: [{k_vals.min():.1f}, {k_vals.max():.1f}])")
                # No need to recompute - just validate
            else:
                # Compute geometric factors with topography using PyGIMLi
                print("   Recomputing geometric factors with PyGIMLi...")
                data['k'] = ert_pg.createGeometricFactors(data, numerical=True)
                
                # Recompute apparent resistivity with correct K
                # rhoa = R * K
                data['rhoa'] = data['r'] * data['k']
                k_vals = np.array(data['k'])
                print(f"   Computed K range: [{k_vals.min():.1f}, {k_vals.max():.1f}]")
            
            # Filter by geometric factor threshold (for DAS data)
            k_threshold = 1000  # max geometric factor, m
            k_vals = np.array(data['k'])
            k_valid = np.abs(k_vals) < k_threshold
            n_k_filtered = np.sum(~k_valid)
            if n_k_filtered > 0:
                print(f"   Filtered {n_k_filtered} measurements with |K| >= {k_threshold} m")
                remove_indices = np.where(~k_valid)[0]
                for idx in sorted(remove_indices, reverse=True):
                    data.remove(int(idx))
            
            # Recompute apparent resistivity with correct K
            # rhoa = R * K
            data['rhoa'] = data['r'] * data['k']
            
            # Filter extreme apparent resistivity values
            rhoa_vals = np.array(data['rhoa'])
            valid_mask = (np.isfinite(rhoa_vals)) & (rhoa_vals > 0.1) & (rhoa_vals < 1e6)
            
            # Calculate statistics for outlier detection
            valid_rhoa = rhoa_vals[valid_mask]
            if len(valid_rhoa) > 0:
                rhoa_median = np.median(valid_rhoa)
                rhoa_std = np.std(valid_rhoa)
                # Filter outliers: keep values within median ± 3*std
                lower_bound = max(0.1, rhoa_median - 3 * rhoa_std)
                upper_bound = min(1e6, rhoa_median + 3 * rhoa_std)
                valid_mask = valid_mask & (rhoa_vals >= lower_bound) & (rhoa_vals <= upper_bound)
                
                n_total = data.size()
                n_filtered = n_total - np.sum(valid_mask)
                if n_filtered > 0:
                    print(f"   Filtered {n_filtered} measurements with extreme apparent resistivity")
                    # Apply filter using PyGIMLi's remove method
                    # Get indices to remove (where valid_mask is False)
                    remove_indices = np.where(~valid_mask)[0]
                    for idx in sorted(remove_indices, reverse=True):
                        data.remove(int(idx))
            
            # Save the updated file with correct K values
            print(f"   Final dataset: {data.size()} measurements with computed K")
            
            # Rewrite the file with updated K and rhoa values
            temp_path = path.parent / (path.name + '.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                # Write number of electrodes
                f.write(f"{len(ert.electrodes)}\n")
                
                # Write electrode coordinates
                f.write("# x y z\n")
                for elec in ert.electrodes:
                    f.write(f"{elec.x} {elec.y} {elec.z}\n")
                
                # Write number of measurements
                f.write(f"{data.size()}\n")
                
                # Write measurement data with computed K and error
                # Check if error data exists
                has_err = 'err' in data.dataMap() and len(data['err']) == data.size()
                if has_err:
                    f.write("# a b m n r rhoa k err\n")
                else:
                    f.write("# a b m n r rhoa k\n")
                    
                for i in range(data.size()):
                    # PyGIMLi uses 0-based indices internally, but file format uses 1-based
                    f.write(f"{int(data['a'][i]) + 1} {int(data['b'][i]) + 1} {int(data['m'][i]) + 1} {int(data['n'][i]) + 1} ")
                    if has_err:
                        f.write(f"{data['r'][i]} {data['rhoa'][i]} {data['k'][i]} {data['err'][i]}\n")
                    else:
                        f.write(f"{data['r'][i]} {data['rhoa'][i]} {data['k'][i]}\n")
            
            # Replace original file with updated version
            temp_path.replace(path)
            print(f"   Saved data with computed K to {path}")
            
        except Exception as e:
            import traceback
            print(f"   Warning: Could not recompute K with PyGIMLi: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            print(f"   File kept with k=1 placeholder values")
        
        return str(path)
    elif fmt == "resipy":
        return ert.metadata.get("project_dir", "")
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
