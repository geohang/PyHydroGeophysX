# PyHydroGeophysX/core/data_processing/ert_data_agent.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import io
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
    try:
        with open(fname, "r") as f:
            dump = f.readlines()
    except Exception as e:
        raise ValueError(f"Could not read file {fname}: {e}")

    if not dump:
        raise ValueError(f"File {fname} is empty")

    numStr = r'[-+]?\d*\.\d*[eE]?[-+]?\d+|\d+'

    # Skip comment lines and find data start
    data_start_line = None
    for i, line_content in enumerate(dump):
        clean_line = line_content.strip()
        if clean_line and not clean_line.startswith('#') and not clean_line.startswith('*') and not clean_line.startswith('!'):
            try:
                # Try to parse as numbers
                vals = re.findall(numStr, clean_line)
                if len(vals) >= 4:  # A B M N minimum
                    data_start_line = i
                    break
            except:
                continue

    if data_start_line is None:
        raise ValueError("Could not find data section in file")

    # Try to read as CSV first (more common format)
    try:
        df = pd.read_csv(fname, comment='#', sep=r'\s+', engine='python')
        if len(df.columns) >= 4:
            # Assume first 4 columns are A B M N
            col_names = ['a', 'b', 'm', 'n'] + [f'col_{i}' for i in range(4, len(df.columns))]
            df.columns = col_names[:len(df.columns)]

            # Convert ABMN to int
            for col in ['a', 'b', 'm', 'n']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

            # Build electrode positions from unique values
            array = df[['a', 'b', 'm', 'n']].values
            unique_elec = np.sort(np.unique(array.flatten()))
            n_elec = len(unique_elec)

            # Assume regular spacing starting from 0
            spacing = 1.0 if n_elec <= 1 else np.min(np.diff(unique_elec))
            elec = np.c_[np.arange(n_elec) * spacing, np.zeros(n_elec), np.zeros(n_elec)]

            # Add IP column if missing
            if 'ip' not in df.columns:
                df['ip'] = np.nan

            return elec, df

    except Exception as csv_error:
        # Fall back to line-by-line parsing
        pass

    # Manual parsing for more complex formats
    df_list = []
    for line_content in dump[data_start_line:]:
        clean_line = line_content.strip().split('#')[0]  # Remove comments
        if not clean_line:
            continue

        vals = re.findall(numStr, clean_line)
        if len(vals) >= 4:  # Need at least A B M N
            df_list.append([float(v) for v in vals])

    if not df_list:
        raise ValueError("Could not parse any measurement data")

    # Create DataFrame
    df = pd.DataFrame(df_list)

    # Assign default column names
    default_headers = ['a', 'b', 'm', 'n', 'resist', 'dev', 'ip']
    col_count = df.shape[1]
    if col_count <= len(default_headers):
        df.columns = default_headers[:col_count]
    else:
        extras = [f'extra_{i}' for i in range(col_count - len(default_headers))]
        df.columns = default_headers + extras

    # Convert ABMN to int
    for col in ['a', 'b', 'm', 'n']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

    # Build electrode positions
    array = df[['a', 'b', 'm', 'n']].values
    unique_elec = np.sort(np.unique(array.flatten()))
    n_elec = len(unique_elec)

    # Assume regular spacing
    if n_elec > 1:
        spacing = np.min(np.diff(unique_elec))
        elec = np.c_[unique_elec, np.zeros(n_elec), np.zeros(n_elec)]
    else:
        elec = np.array([[0, 0, 0]])

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
    try:
        # Try reading as CSV first
        df = pd.read_csv(fname, skipinitialspace=True, engine='python', encoding_errors='ignore')
    except Exception:
        raise ValueError(f"Could not read {fname} as CSV")

    if df.empty:
        raise ValueError(f"No data found in {fname}")

    headers = df.columns

    # Standardize column names based on format version
    rename_map = {}

    if 'Spa.1' in headers:
        rename_map.update({
            'Spa.1': 'a', 'Spa.2': 'b', 'Spa.3': 'm', 'Spa.4': 'n',
            'In': 'i', 'Vp': 'vp', 'Dev.': 'dev', 'M': 'ip', 'Sp': 'sp'
        })
    elif any('xA' in h for h in headers):
        rename_map.update({
            'xA(m)': 'a', 'xB(m)': 'b', 'xM(m)': 'm', 'xN(m)': 'n',
            'xA (m)': 'a', 'xB (m)': 'b', 'xM (m)': 'm', 'xN (m)': 'n',
            'Dev.': 'dev', 'Dev. Rho (%)': 'dev',
            'M (mV/V)': 'ip', 'SP (mV)': 'sp',
            'VMN (mV)': 'vp', 'IAB (mA)': 'i',
            'yA (m)': 'ya', 'yB (m)': 'yb', 'yM (m)': 'ym', 'yN (m)': 'yn',
            'zA (m)': 'za', 'zB (m)': 'zb', 'zM (m)': 'zm', 'zN (m)': 'zn'
        })

    # Apply renaming
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in headers})

    # Calculate resistance if needed
    if 'vp' in df.columns and 'i' in df.columns and 'resist' not in df.columns:
        df['resist'] = df['vp'] / df['i']

    # Ensure we have the basic columns
    required_cols = ['a', 'b', 'm', 'n']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # Assume first 4 columns are A B M N
        if len(df.columns) >= 4:
            df.columns = ['a', 'b', 'm', 'n'] + list(df.columns[4:])
        else:
            raise ValueError(f"Missing required columns: {missing_cols}")

    # Convert ABMN to int
    for col in ['a', 'b', 'm', 'n']:
        if col in df.columns:
            try:
                df[col] = df[col].astype(int)
            except:
                # If conversion fails, keep as is
                pass

    # Process electrode positions
    try:
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
    except Exception as e:
        # Fallback: simple electrode array
        array = df[['a', 'b', 'm', 'n']].values
        val = np.sort(np.unique(array.flatten()))
        n_elec = len(val)
        if n_elec > 0:
            spacing = 1.0
            elec = np.c_[np.arange(n_elec) * spacing, np.zeros(n_elec), np.zeros(n_elec)]
        else:
            elec = np.array([[0, 0, 0]])

    if 'ip' not in df.columns:
        df['ip'] = np.nan

    return elec, df


def _protocolParser(fname, ip=False):
    """
    Parse Protocol format (from Lund Imaging System).
    Modified from ResIPy project (GPL-3.0).
    Original authors: Guillaume Blanchy, Jimmy Boyd, et al.
    """
    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Could not read file {fname}: {e}")

    # Find data start
    data_start = 0
    for i, line in enumerate(lines):
        clean_line = line.strip()
        if clean_line and not clean_line.startswith('#') and not clean_line.startswith('*'):
            try:
                vals = clean_line.split()
                if len(vals) >= 4:
                    # Try to convert to numbers
                    float_vals = [float(x) for x in vals[:4]]  # Check first 4 values
                    data_start = i
                    break
            except ValueError:
                continue

    if data_start == 0:
        raise ValueError("Could not find data section")

    # Parse data
    data_list = []
    for line in lines[data_start:]:
        clean_line = line.strip()
        if clean_line and not clean_line.startswith('#') and not clean_line.startswith('*'):
            try:
                vals = clean_line.split()
                if len(vals) >= 4:
                    num_vals = [float(x) for x in vals]
                    data_list.append(num_vals)
            except ValueError:
                continue

    if not data_list:
        raise ValueError("Could not parse any measurement data")

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
            try:
                df[col] = df[col].astype(int)
            except:
                pass  # Keep as float if conversion fails

    # Build electrode array
    try:
        array = df[['a', 'b', 'm', 'n']].values
        unique_elec = np.sort(np.unique(array.flatten()))
        n_elec = len(unique_elec)

        # Assume regular spacing
        if n_elec > 1:
            spacing = np.min(np.diff(unique_elec))
            elec = np.c_[unique_elec, np.zeros(n_elec), np.zeros(n_elec)]
        else:
            elec = np.array([[0, 0, 0]])
    except Exception:
        # Fallback electrode array
        n_measurements = len(df)
        n_elec = int(np.max(df[['a', 'b', 'm', 'n']].values.flatten()) if 'a' in df.columns else 4)
        spacing = 1.0
        elec = np.c_[np.arange(1, n_elec + 1) * spacing, np.zeros(n_elec), np.zeros(n_elec)]

    if 'ip' not in df.columns:
        df['ip'] = np.nan

    return elec, df


def _dasParser(fname):
    """
    Parse DAS-1 format (ERTLab DACQ).
    Modified from ResIPy project (GPL-3.0).
    Handles electrode blocks and mixed separators from DAS-1 exports.
    """
    try:
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            dump_raw = f.readlines()
    except Exception as e:
        raise ValueError(f"Could not read file {fname}: {e}")

    numStr = r'[-+]?\d*\.\d*[eE]?[-+]?\d+|\d+'

    # Remove known bad rows (e.g., out-of-range records)
    dump = [val for val in dump_raw if 'out of range' not in val]
    cleanData = ''.join(dump)

    # Electrode section
    try:
        elec_lineNum_s = next(i + 2 for i in range(len(dump)) if '#elec_start' in dump[i])
        elec_lineNum_e = next(i for i in range(len(dump)) if '#elec_end' in dump[i])
    except StopIteration:
        raise ValueError("Could not locate electrode section in DAS-1 file")

    nrows = elec_lineNum_e - elec_lineNum_s
    try:
        dfElec_raw = pd.read_csv(
            io.StringIO(cleanData),
            sep=r'\s+',
            skiprows=elec_lineNum_s,
            nrows=nrows,
            index_col=False,
            header=None,
            dtype=str,
            engine='python'
        )
    except Exception:
        # Fallback to detected encoding (slow path)
        enc = None
        try:
            import chardet
            with open(fname, 'rb') as f:
                enc = chardet.detect(f.read()).get('encoding', None)
        except Exception:
            enc = None

        dfElec_raw = pd.read_csv(
            io.StringIO(cleanData),
            sep=r'\s+',
            skiprows=elec_lineNum_s,
            nrows=nrows,
            index_col=False,
            header=None,
            dtype=str,
            engine='python',
            encoding=enc
        )

    elecNum = dfElec_raw.iloc[:, 0].str.split(',', expand=True)
    elecNum = elecNum.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).astype(str)
    elecLabel = elecNum[0].str.strip().str.cat(elecNum[1].str.strip(), sep=' ')

    dfelec = pd.DataFrame()
    dfelec['label'] = elecLabel.copy()
    dfelec[['x', 'y', 'z']] = dfElec_raw.iloc[:, 1:4].apply(pd.to_numeric, errors='coerce')
    dfelec['buried'] = False
    dfelec['remote'] = False

    # Remote electrodes flags (copied from ResIPy logic)
    remote_flags = [-9999999, -999999, -99999, -9999, -999,
                    9999999, 999999, 99999]
    iremote = np.isin(dfelec['x'].values, remote_flags)
    iremote = np.isinf(dfelec[['x', 'y', 'z']].values).any(1) | iremote
    dfelec['remote'] = iremote

    # Data section
    try:
        df_lineNum_s = next(i + 3 for i in range(len(dump)) if '#data_start' in dump[i])
        df_lineNum_e = next(i for i in range(len(dump)) if '#data_end' in dump[i])
    except StopIteration:
        raise ValueError("Could not locate data section in DAS-1 file")

    df_list = []
    for val in dump[df_lineNum_s:df_lineNum_e]:
        vals = re.findall(numStr, val)
        if vals:
            df_list.append(vals)

    if not df_list:
        raise ValueError("No measurement rows found in DAS-1 file")

    max_len = max(len(row) for row in df_list)
    normalized = [row + [np.nan] * (max_len - len(row)) for row in df_list]
    df_raw = pd.DataFrame(np.array(normalized, dtype=float))

    # Determine 2D vs 3D (line numbers vary for 3D)
    flagD = '3D' if np.mean(df_raw.iloc[:, 1]) != df_raw.iloc[0, 1] else '2D'

    def _header_col(keyword: str, default: int | None = None) -> int | None:
        for line in dump:
            if keyword in line:
                try:
                    return int(line.split()[-1]) - 1
                except Exception:
                    try:
                        # Fallback: last numeric token
                        tokens = re.findall(numStr, line)
                        if tokens:
                            return int(tokens[-1]) - 1
                    except Exception:
                        return default
        return default

    resCol = _header_col('data_res_col', default=df_raw.shape[1] - 1)
    devCol = _header_col('data_std_res_col', default=-1)
    ipCol = _header_col('data_ip_wind_col', default=-1)

    df = pd.DataFrame()
    arrHeader = ['a', 'b', 'm', 'n']

    # 2D array
    if flagD == '2D':
        lineNumber = int(df_raw.iloc[0, 1])
        elecLNum = elecNum[0].astype(int)
        selectElecs = elecLNum[elecLNum == lineNumber].index.values
        dfelec_sel = dfelec.iloc[selectElecs, :].reset_index(drop=True)

        # Convert 3D XY to 2D profile if needed
        if np.isfinite(dfelec_sel['x']).all() and np.mean(dfelec_sel['x']) == dfelec_sel['x'][0]:
            dfelec_sel['x'] = dfelec_sel['y'].values.copy()
            dfelec_sel['y'] = 0

        dfelec_sel = dfelec_sel.sort_values('x').reset_index(drop=True)

        for idx, name in enumerate(arrHeader):
            df[name] = pd.to_numeric(df_raw.iloc[:, (idx + 1) * 2], errors='coerce').astype(int)

        elec_out = dfelec_sel

    # 3D array
    else:
        lines = np.unique(df_raw.iloc[:, 1].values)
        elecLNum = elecNum[0].astype(int)
        dfelec_selected = []
        for lineNumber in lines:
            selectElecs = elecLNum[elecLNum == lineNumber].index.values
            dfelec_selected.append(dfelec.iloc[selectElecs, :])

        elec_out = pd.concat(dfelec_selected).reset_index(drop=True)

        for idx, name in enumerate(arrHeader):
            left = pd.to_numeric(df_raw.iloc[:, (idx * 2) + 1], errors='coerce').fillna(0).astype(int).astype(str)
            right = pd.to_numeric(df_raw.iloc[:, (idx + 1) * 2], errors='coerce').fillna(0).astype(int).astype(str)
            df[name] = left.str.cat(right, sep=' ')

    # Data columns
    if resCol is not None and 0 <= resCol < df_raw.shape[1]:
        df['resist'] = df_raw.iloc[:, resCol].values
    if devCol is not None and 0 <= devCol < df_raw.shape[1]:
        df['dev'] = df_raw.iloc[:, devCol].values
    if ipCol is not None and ipCol > 1 and ipCol < df_raw.shape[1]:
        df['ip'] = df_raw.iloc[:, ipCol].values
    else:
        df['ip'] = df.get('ip', 0)

    return elec_out, df


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
    "DAS-1": _dasParser,
    "ABEM-Lund": _bertParser,  # Use BERT parser as fallback
    "Lippmann": _bertParser,   # Use BERT parser as fallback
    "ARES": _bertParser,       # Use BERT parser as fallback
}


# ---------------------------
# Local Parser Fallback Loader (adapted from ResIPy)
# ---------------------------
def _compute_reciprocal_errors(df: pd.DataFrame, max_reciprocal_error: float = 0.05) -> pd.DataFrame:
    """
    Compute reciprocal errors and filter measurements based on reciprocal error threshold.

    ACKNOWLEDGEMENT & LICENSE
    -------------------------
    This function is MODIFIED from ResIPy's computeReciprocalP() method:
    - Original Source: https://gitlab.com/hkex/resipy (Survey.py, lines 787-900)
    - Original License: GPL-3.0 (GNU General Public License v3.0)
    - Original Authors: Guillaume Blanchy, Jimmy Boyd, Sina Saneiyan, Pedro Concha
    - Original Function: Survey.computeReciprocalP()

    Algorithm:
    1. Sort quadrupoles (A,B) and (M,N) to create standardized array
    2. Use pandas merge to match normal and reciprocal measurements
    3. Compute reciprocal error on resistance values: err = (R_recip - R_normal) / R_mean
    4. Filter measurements with reciprocal error > threshold

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: a, b, m, n, resist (resistance values)
    max_reciprocal_error : float
        Maximum allowed reciprocal error (default: 0.05 = 5%)

    Returns
    -------
    pd.DataFrame
        Filtered dataframe with added columns: reciprocalErrRel, recipMean
    """
    if 'resist' not in df.columns:
        print("   Warning: No 'resist' column found, skipping reciprocal error computation")
        return df

    resist = df['resist'].values
    n_data = len(df)

    # Get quadrupole array (0-indexed for Python)
    array = df[['a', 'b', 'm', 'n']].values.astype(int)

    # Initialize output arrays
    reciprocalErr = np.zeros(n_data) * np.nan
    reciprocalErrRel = np.zeros(n_data) * np.nan
    reciprocalMean = np.zeros(n_data) * np.nan

    # Sort quadrupoles: (min(A,B), max(A,B), min(M,N), max(M,N))
    # This creates a canonical form so normal and reciprocal match
    sortedArray = np.c_[np.sort(array[:, :2], axis=1), np.sort(array[:, 2:], axis=1)]

    # Build dataframe of normal and reciprocal and merge them
    # Normal: (A, B, M, N)
    df1 = pd.DataFrame(sortedArray, columns=['a', 'b', 'm', 'n'])
    df1['index1'] = np.arange(df1.shape[0])

    # Reciprocal: (M, N, A, B) - swap current and potential
    df2 = pd.DataFrame(sortedArray, columns=['m', 'n', 'a', 'b'])
    df2['index2'] = np.arange(df2.shape[0])

    # Merge on (a,b,m,n) to find matches
    dfm = pd.merge(df1, df2, on=['a', 'b', 'm', 'n'], how='outer')
    dfm = dfm.dropna()  # Keep only measurements that have both normal and reciprocal

    if len(dfm) == 0:
        print("   Warning: No reciprocal pairs found in data")
        df['reciprocalErrRel'] = np.nan
        df['recipMean'] = df['resist']  # Use resist for measurements without reciprocals
        return df

    # Sort and keep only half (avoid counting each pair twice)
    indexArray = np.sort(dfm[['index1', 'index2']].values.astype(int), axis=1)
    indexArrayUnique = np.unique(indexArray, axis=0)
    inormal = indexArrayUnique[:, 0]
    irecip = indexArrayUnique[:, 1]

    print(f"   Found {len(inormal)} reciprocal pairs ({100*len(inormal)/n_data:.1f}% of measurements)")

    # Compute reciprocal error on resistance values
    # err = R_recip - R_normal
    reciprocalErr[inormal] = resist[irecip] - resist[inormal]
    reciprocalErr[irecip] = resist[irecip] - resist[inormal]

    # Compute reciprocal mean with valid values only
    ok1 = ~(np.isnan(resist[inormal]) | np.isinf(resist[inormal]))
    ok2 = ~(np.isnan(resist[irecip]) | np.isinf(resist[irecip]))

    ie = ok1 & ok2  # Both normal and recip are valid
    reciprocalMean[inormal[ie]] = np.mean(np.c_[np.abs(resist[inormal[ie]]), np.abs(resist[irecip[ie]])], axis=1)
    reciprocalMean[irecip[ie]] = np.mean(np.c_[np.abs(resist[inormal[ie]]), np.abs(resist[irecip[ie]])], axis=1)

    ie = ok1 & ~ok2  # Only use normal
    reciprocalMean[inormal[ie]] = np.abs(resist[inormal[ie]])

    ie = ~ok1 & ok2  # Only use reciprocal
    reciprocalMean[inormal[ie]] = np.abs(resist[irecip[ie]])

    # Compute relative reciprocal error
    # Avoid division by zero: if reciprocalMean is too small, set error to NaN (will be filtered/replaced later)
    reciprocalErrRel = np.where(
        np.abs(reciprocalMean) > 1e-10,  # Only compute if mean is not near zero
        reciprocalErr / reciprocalMean,
        np.nan  # Mark as NaN if division would be invalid
    )

    # Preserve sign in reciprocalMean
    reciprocalMean = np.sign(resist) * reciprocalMean

    # For measurements without reciprocals, use original resistance
    inotRecip = np.isnan(reciprocalErrRel)
    reciprocalMean[inotRecip] = resist[inotRecip]

    # Add columns to dataframe
    df['reciprocalErrRel'] = reciprocalErrRel
    df['recipMean'] = reciprocalMean

    # Filter measurements with high reciprocal error
    before_filter = len(df)
    df = df[np.abs(df['reciprocalErrRel']) < max_reciprocal_error].copy()
    n_filtered = before_filter - len(df)

    if n_filtered > 0:
        print(f"   Filtered {n_filtered} measurements with reciprocal error > {max_reciprocal_error*100:.0f}%")

    return df


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
    parser_name = instrument  # Use instrument name for error message

    if parser_func is None:
        parser_func = _bertParser  # Default fallback
        parser_name = "BERT (fallback)"

    # Parse the data file
    try:
        print(f"[PyHydroGeophysX] Attempting to parse ERT data with {parser_name} parser...")
        elec_array, df = parser_func(str(data_file_path))
        print(f"[PyHydroGeophysX] Successfully parsed {len(df)} measurements and {len(elec_array)} electrodes")
    except Exception as e:
        print(f"[PyHydroGeophysX] Parser {parser_name} failed: {e}")
        # Try fallback parsers if this one fails
        if parser_func != _bertParser:
            print(f"[PyHydroGeophysX] Trying BERT parser as fallback...")
            try:
                elec_array, df = _bertParser(str(data_file_path))
                print(f"[PyHydroGeophysX] BERT parser fallback succeeded")
                parser_name = "BERT (fallback)"
            except Exception as e2:
                raise ValueError(f"Failed to parse ERT data with {parser_name} (primary) and BERT (fallback): {e} // {e2}")
        else:
            raise ValueError(f"Failed to parse ERT data with {parser_name}: {e}")
    
    # Build electrodes dataframe and optional label map for non-numeric electrode IDs
    label_map = None
    if isinstance(elec_array, np.ndarray):
        if elec_array.ndim == 1:
            elec_array = elec_array.reshape(-1, 1)
        n_cols = elec_array.shape[1]
        electrodes_df = pd.DataFrame({
            'x': elec_array[:, 0],
            'y': elec_array[:, 1] if n_cols > 1 else 0.0,
            'z': elec_array[:, 2] if n_cols > 2 else 0.0,
        })
    else:
        electrodes_df = pd.DataFrame({
            'x': elec_array['x'] if 'x' in elec_array.columns else elec_array.iloc[:, 0],
            'y': elec_array['y'] if 'y' in elec_array.columns else 0.0,
            'z': elec_array['z'] if 'z' in elec_array.columns else 0.0,
        })
        if 'label' in elec_array.columns:
            electrodes_df['label'] = elec_array['label']
            label_map = {str(lbl).strip(): idx + 1 for idx, lbl in enumerate(pd.unique(elec_array['label']))}

    # Override electrode positions from external file if provided
    # This matches ResIPy behavior: external electrode file takes priority
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

            # Update electrode positions in dataframe
            electrodes_df['x'] = elec_data[:, 0]
            electrodes_df['y'] = elec_data[:, 1] if elec_data.shape[1] > 1 else 0.0
            electrodes_df['z'] = elec_data[:, 2] if elec_data.shape[1] > 2 else 0.0
            print(f"   Updated electrode positions from {electrode_file_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load electrode file '{electrode_file_path}': {e}") from e
    
    # Build observations dataframe with standardized columns
    observations = pd.DataFrame()
    def _coerce_indices(col_name: str, fallback_value: int) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(np.full(len(df), fallback_value, dtype=int))
        series = df[col_name]
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            return numeric.astype(int)
        labels = series.astype(str).str.strip()
        if label_map:
            mapped = labels.map(label_map)
        else:
            unique_labels = pd.unique(labels)
            tmp_map = {lab: idx + 1 for idx, lab in enumerate(unique_labels)}
            mapped = labels.map(tmp_map)
        return mapped.fillna(fallback_value).astype(int)
    
    observations['a'] = _coerce_indices('a', 1)
    observations['b'] = _coerce_indices('b', 2)
    observations['m'] = _coerce_indices('m', 3)
    observations['n'] = _coerce_indices('n', 4)

    # Get resistance values (needed for reciprocal error computation)
    resist_cols = ['resist', 'R', 'r', 'resistance']
    resist_col = next((c for c in resist_cols if c in df.columns), None)
    if resist_col:
        observations['resist'] = pd.to_numeric(df[resist_col], errors='coerce')

    # Compute reciprocal errors BEFORE any K computation or rhoa calculation
    # This matches ResIPy behavior: reciprocal processing on resistance values
    if 'resist' in observations.columns:
        print("   Computing reciprocal errors on resistance values (ResIPy algorithm)...")
        observations = _compute_reciprocal_errors(observations, max_reciprocal_error=0.05)

    # Get apparent resistivity / resistance
    rho_cols = ['app', 'rhoa', 'rhoA', 'Rhoa', 'app_res']
    if any(c in df.columns for c in rho_cols):
        rho_col = next(c for c in rho_cols if c in df.columns)
        observations['rhoa'] = pd.to_numeric(df[rho_col], errors='coerce')
    elif 'resist' in observations.columns:
        # Use resistance values (will be converted to rhoa with K later)
        observations['rhoa'] = observations['resist']
    else:
        observations['rhoa'] = 100.0  # Default
    observations['rhoa'] = pd.to_numeric(observations['rhoa'], errors='coerce')
    if observations['rhoa'].isna().all():
        observations['rhoa'] = 100.0

    # Compute error estimates based on reciprocal data or standard deviation
    # Priority: 1) reciprocalErrRel from reciprocal processing, 2) dev/std column, 3) default 5%
    if 'reciprocalErrRel' in observations.columns:
        # Use reciprocal-based error estimates (best option)
        # For measurements without reciprocals or with zero error, use 5% default
        # IMPORTANT: fillna only handles NaN, need to also replace zeros and infinities
        observations['error'] = observations['reciprocalErrRel'].fillna(0.05)
        # Replace zeros and infinities with 5% default
        observations['error'] = observations['error'].replace([0, np.inf, -np.inf], 0.05)
        # Ensure all values are positive and at least 1% (avoid division by zero in inversion)
        observations['error'] = np.maximum(np.abs(observations['error']), 0.01)
        print(f"   Using reciprocal-based error estimates (mean: {observations['error'].mean():.4f})")
    else:
        # Fallback to dev/std column or default
        err_col = next((c for c in ['error', 'err', 'dev', 'std', 'std_res'] if c in df.columns), None)
        if err_col:
            err_vals = pd.to_numeric(df[err_col], errors='coerce')
            rho_vals = pd.to_numeric(observations['rhoa'], errors='coerce')
            rel_err = np.where(
                (np.isfinite(err_vals)) & (np.isfinite(rho_vals)) & (rho_vals != 0),
                np.abs(err_vals) / np.abs(rho_vals),
                np.nan
            )
            median_err = np.nanmedian(err_vals)
            if np.isfinite(median_err) and median_err > 1 and np.nanmedian(rel_err) > 1:
                rel_err = rel_err / 100.0
            observations['error'] = np.where(np.isfinite(rel_err), rel_err, 0.05)
            print(f"   Using standard deviation-based error estimates (mean: {observations['error'].mean():.4f})")
        else:
            observations['error'] = 0.05  # Default 5%
            print("   Using default 5% error estimates")

    observations['valid'] = True

    # Convert to dataclass lists for downstream compatibility
    electrodes_list = [
        Electrode(i + 1, float(row['x']), float(row['y']), float(row['z']))
        for i, row in electrodes_df.iterrows()
    ]

    obs_list: List[Observation] = []
    for idx, row in observations.iterrows():
        app_res_val = float(row['rhoa']) if np.isfinite(row['rhoa']) else None
        rel_err_val = float(row['error']) if np.isfinite(row['error']) else 0.05
        obs_list.append(Observation(
            quad=Quadruplet(int(row['a']), int(row['b']), int(row['m']), int(row['n'])),
            app_res=app_res_val,
            dV=None,
            I=None,
            rel_err=rel_err_val,
            K=1.0,
            fid=str(idx)
        ))
    
    # Build metadata
    metadata = {
        'source_file': str(data_file_path),
        'loader': 'local_parsers_resipy_fallback',
        'parser_used': parser_name,
        'instrument': instrument,
        'n_electrodes': len(electrodes_list),
        'n_measurements': len(obs_list),
        'acknowledgement': 'Parsing logic adapted from ResIPy (https://gitlab.com/hkex/resipy) under GPL-3.0',
    }
    if label_map is not None:
        metadata['electrode_label_map'] = label_map
    
    if local_ref is not None:
        metadata['local_origin_x'] = local_ref.origin_x
        metadata['local_origin_y'] = local_ref.origin_y
        metadata['azimuth_deg'] = local_ref.azimuth_deg
    
    if epsg is not None:
        metadata['epsg'] = epsg
    
    return StandardERT(
        electrodes=electrodes_list,
        observations=obs_list,
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
    electrodes_df = pd.DataFrame({
        'x': sensors[:, 0],
        'y': sensors[:, 1] if sensors.shape[1] > 1 else 0.0,
        'z': sensors[:, 2] if sensors.shape[1] > 2 else 0.0,
    })

    # Override electrode positions from external file if provided
    # This matches ResIPy behavior: external electrode file takes priority
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

            # Update electrode positions in dataframe
            electrodes_df['x'] = elec_data[:, 0]
            electrodes_df['y'] = elec_data[:, 1] if elec_data.shape[1] > 1 else 0.0
            electrodes_df['z'] = elec_data[:, 2] if elec_data.shape[1] > 2 else 0.0
            print(f"   Updated electrode positions from {electrode_file_path.name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load electrode file '{electrode_file_path}': {e}") from e

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
    observations_df = pd.DataFrame({
        'a': a.astype(int),
        'b': b.astype(int),
        'm': m.astype(int),
        'n': n.astype(int),
        'rhoa': rhoa,
        'error': error,
        'valid': np.ones(n_data, dtype=bool)
    })

    # Convert to dataclass lists for consistency
    electrodes_list = [
        Electrode(i + 1, float(row['x']), float(row['y']), float(row['z']))
        for i, row in electrodes_df.iterrows()
    ]
    observations_list = [
        Observation(
            quad=Quadruplet(int(row['a']), int(row['b']), int(row['m']), int(row['n'])),
            app_res=float(row['rhoa']) if np.isfinite(row['rhoa']) else None,
            dV=None,
            I=None,
            rel_err=float(row['error']) if np.isfinite(row['error']) else 0.05,
            K=1.0,
            fid=str(idx)
        )
        for idx, row in observations_df.iterrows()
    ]
    
    # Build metadata
    metadata = {
        'source_file': str(data_file_path),
        'loader': 'pygimli_fallback',
        'instrument': instrument,
        'n_electrodes': len(electrodes_list),
        'n_measurements': len(observations_list),
    }
    
    if local_ref is not None:
        metadata['local_origin_x'] = local_ref.origin_x
        metadata['local_origin_y'] = local_ref.origin_y
        metadata['azimuth_deg'] = local_ref.azimuth_deg
    
    if epsg is not None:
        metadata['epsg'] = epsg
    
    return StandardERT(
        electrodes=electrodes_list,
        observations=observations_list,
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

    # Electrodes plot (accept dataclasses or dict-like)
    if ert.electrodes and hasattr(ert.electrodes[0], "x"):
        ex = [e.x for e in ert.electrodes]
        ez = [e.z for e in ert.electrodes]
    else:
        # Fallback if electrodes are dict-like/Series
        elec_df = pd.DataFrame(ert.electrodes)
        ex = elec_df['x'].tolist()
        ez = elec_df.get('z', pd.Series(np.zeros(len(elec_df)))).tolist()
    plt.figure(figsize=(6, 2))
    plt.plot(ex, ez, 'k.-')
    plt.xlabel('x (m)'); plt.ylabel('z (m)')
    p1 = str(outdir_path / "electrodes.png")
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Apparent resistivity histogram
    vals = [o.app_res for o in ert.observations if o.app_res is not None and np.isfinite(o.app_res) and o.app_res > 0]
    plt.figure(figsize=(4, 3))
    if len(vals) > 0:
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

            # DEBUG: Print electrode information being written
            print(f"   DEBUG: Writing {len(ert.electrodes)} electrodes to file")
            if len(ert.electrodes) > 0:
                y_vals = [e.y for e in ert.electrodes]
                print(f"   DEBUG: First electrode: x={ert.electrodes[0].x:.3f}, y={ert.electrodes[0].y:.3f}, z={ert.electrodes[0].z:.3f}")
                print(f"   DEBUG: Last electrode: x={ert.electrodes[-1].x:.3f}, y={ert.electrodes[-1].y:.3f}, z={ert.electrodes[-1].z:.3f}")
                print(f"   DEBUG: Y-range (elevation) being written: [{min(y_vals):.3f}, {max(y_vals):.3f}]")

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
            rho_min = 0.1
            rho_max = 1e6
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

                # Enforce positivity and reasonable bounds before writing
                if not np.isfinite(R) or not np.isfinite(rhoa):
                    continue
                R = abs(R)
                rhoa = abs(rhoa)
                if R <= 0 or rhoa <= 0:
                    continue
                rhoa = float(np.clip(rhoa, rho_min, rho_max))
                R = float(max(R, rho_min))
                
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
        print("   Validating geometric factors... [CODE-VERSION: 2026-01-20-v3-RECIPROCAL-AFTER-K]")
        try:
            import pygimli as pg
            import pygimli.physics.ert as ert_pg
            
            # Load the file we just created
            data = ert_pg.load(str(path))

            # DEBUG: Print electrode positions loaded by PyGIMLi
            sensors = data.sensorPositions()
            print(f"   DEBUG: PyGIMLi loaded {len(sensors)} electrode positions")
            if len(sensors) > 0:
                print(f"   DEBUG: First electrode: x={sensors[0].x():.3f}, y={sensors[0].y():.3f}, z={sensors[0].z():.3f}")
                print(f"   DEBUG: Last electrode: x={sensors[-1].x():.3f}, y={sensors[-1].y():.3f}, z={sensors[-1].z():.3f}")
                y_vals = [s.y() for s in sensors]
                print(f"   DEBUG: Y-range (elevation): [{min(y_vals):.3f}, {max(y_vals):.3f}]")

            # Check if K was already provided (not all k=1)
            k_vals = np.array(data['k'])
            has_valid_k = np.any(k_vals > 1.5)  # If any K > 1.5, assume K was provided

            if has_valid_k:
                print(f"   K factors already provided (range: [{k_vals.min():.1f}, {k_vals.max():.1f}])")
                # No need to recompute - just validate
            else:
                # Compute geometric factors with topography using PyGIMLi
                print("   Computing geometric factors with PyGIMLi...")
                data['k'] = ert_pg.createGeometricFactors(data, numerical=True)
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

            # NOTE: Reciprocal filtering should have already been done during data loading:
            # - With ResIPy: reciprocalProcessing() is called automatically
            # - With embedded parser: reciprocal error filter is applied (line 1243-1246)
            # We do NOT call reciprocalProcessing() here because:
            # 1. It would be redundant (filtering already done)
            # 2. After computing K, reciprocal pairs have different rhoa even with identical R
            #    because K depends on electrode geometry, making reciprocal error artificially high

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
                # Check if error data exists in PyGIMLi data object
                has_err = 'err' in data.dataMap()
                if has_err:
                    err_vals = np.array(data['err'])
                    # Validate error data: check if non-zero and correct size
                    has_valid_err = (len(err_vals) == data.size()) and np.any(err_vals > 0)
                else:
                    has_valid_err = False

                # Always write error column (default to 5% if not available)
                f.write("# a b m n r rhoa k err\n")

                for i in range(data.size()):
                    # PyGIMLi uses 0-based indices internally, but file format uses 1-based
                    f.write(f"{int(data['a'][i]) + 1} {int(data['b'][i]) + 1} {int(data['m'][i]) + 1} {int(data['n'][i]) + 1} ")

                    # Write error value (use data['err'] if available, otherwise default to 5%)
                    if has_valid_err:
                        err_val = data['err'][i]
                    else:
                        err_val = 0.05  # Default 5% relative error

                    f.write(f"{data['r'][i]} {data['rhoa'][i]} {data['k'][i]} {err_val}\n")
            
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
