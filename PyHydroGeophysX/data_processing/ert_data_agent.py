"""ERT field-data loading, fallback parsing, QC, and export helpers."""

from __future__ import annotations

import io
import json
import re
import struct
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def _notify(message: str) -> None:
    """Write a package status line to stderr.

    These lines are diagnostics, not results. On stdout they corrupted the
    output of generated workflow scripts, whose contract is a single JSON
    document on stdout, so ``python run_<workflow>.py > result.json`` produced a
    file that no JSON parser would accept.
    """
    print(message, file=sys.stderr)


_RESIPY_ERROR = None
_HAS_RESIPY = False
_HAS_EMBEDDED_PARSERS = True  # Always available since embedded
_HAS_PYGIMLI = False

# Try full resipy package first
try:
    from resipy import Project
    _HAS_RESIPY = True
    _notify("[PyHydroGeophysX] RESIPY loaded successfully")
except Exception as e:
    _HAS_RESIPY = False
    _RESIPY_ERROR = str(e)
    _notify(f"[PyHydroGeophysX] RESIPY import failed: {e}, using embedded parsers (modified from ResIPy)")

# Try pygimli as additional fallback
try:
    import pygimli as pg
    from pygimli.physics import ert as pgert
    _HAS_PYGIMLI = True
    _notify("[PyHydroGeophysX] PyGIMLi loaded successfully")
except Exception as e:
    _notify(f"[PyHydroGeophysX] PyGIMLi import failed: {e}")

# Check SimPEG availability
try:
    import simpeg as _simpeg
    _notify(f"[PyHydroGeophysX] SimPEG loaded successfully (version {_simpeg.__version__})")
except Exception as e:
    try:
        import SimPEG as _simpeg
        _notify(f"[PyHydroGeophysX] SimPEG loaded successfully (version {_simpeg.__version__})")
    except Exception as fallback_error:
        _notify(f"[PyHydroGeophysX] SimPEG not available: {fallback_error or e}")


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


def _bertParser_legacy(fname):
    """
    Legacy BERT/Unified parser (CSV / positional fallback for plain files).
    ``_bertParser`` prefers ``_unified_ert_parser`` and only falls back here when a
    file is not the count-prefixed unified/E4D format.
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
            except Exception:
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


def _is_index_column(col0) -> bool:
    """True if ``col0`` is a 1..n or 0..n-1 integer sequence (a row-index column)."""
    a = np.asarray(col0, dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return False
    if not np.allclose(a, np.round(a)):
        return False
    r = np.round(a).astype(int)
    n = a.size
    return np.array_equal(r, np.arange(1, n + 1)) or np.array_equal(r, np.arange(0, n))


def _parse_elec_rows(rows, names):
    """Return an (N, 3) x/y/z electrode array from unified-format electrode rows."""
    w = min(len(r) for r in rows)
    arr = np.array([r[:w] for r in rows], dtype=float)
    lownames = [str(s).lower() for s in names] if names else None
    if lownames and 'x' in lownames:
        # Header names the columns explicitly (no leading index column).
        def pick(key, default_idx):
            if key in lownames and lownames.index(key) < w:
                return arr[:, lownames.index(key)]
            return arr[:, default_idx] if default_idx < w else np.zeros(len(arr))
        x = pick('x', 0)
        y = pick('y', 1)
        z = pick('z', 2) if 'z' in lownames else np.zeros(len(arr))
        return np.column_stack([x, y, z]).astype(float)
    # No usable header: detect an optional leading electrode-index column.
    offset = 1 if (w >= 4 and _is_index_column(arr[:, 0])) else 0
    body = arr[:, offset:]
    bw = body.shape[1]
    x = body[:, 0]
    y = body[:, 1] if bw > 1 else np.zeros(len(arr))
    z = body[:, 2] if bw > 2 else np.zeros(len(arr))
    return np.column_stack([x, y, z]).astype(float)


def _parse_data_rows(rows, names):
    """Return a DataFrame (a,b,m,n + rhoa/resist/dev/...) from unified data rows."""
    w = min(len(r) for r in rows)
    arr = np.array([r[:w] for r in rows], dtype=float)
    lownames = [str(s).lower() for s in names] if names else None
    if lownames and all(k in lownames for k in ('a', 'b', 'm', 'n')):
        col = {k: arr[:, lownames.index(k)] for k in lownames if lownames.index(k) < w}
        out = {'a': col['a'], 'b': col['b'], 'm': col['m'], 'n': col['n']}
        for k in ('rhoa', 'app', 'app_res'):
            if k in col:
                out['rhoa'] = col[k]
                break
        for k in ('r', 'resist', 'resistance'):
            if k in col:
                out['resist'] = col[k]
                break
        for k in ('err', 'dev', 'std', 'error'):
            if k in col:
                out['dev'] = col[k]
                break
        # Keep the file's geometric factors. When the file also reports rhoa,
        # these are the factors it was built with, and the inversion needs them
        # to check that its own k describes the same measurement.
        for k in ('k', 'geom', 'geom_factor'):
            if k in col:
                out['k'] = col[k]
                break
        for k in ('u', 'i', 'ip'):
            if k in col:
                out[k] = col[k]
        if 'rhoa' in out or 'resist' in out:
            df = pd.DataFrame(out)
            if 'ip' not in df.columns:
                df['ip'] = np.nan
            return df
    # Positional layout: detect an optional leading measurement-index column.
    offset = 1 if (w >= 6 and _is_index_column(arr[:, 0])) else 0
    body = arr[:, offset:]
    bw = body.shape[1]
    if bw < 4:
        raise ValueError("unified ERT parser: data rows need at least a, b, m, n")
    out = {'a': body[:, 0], 'b': body[:, 1], 'm': body[:, 2], 'n': body[:, 3]}
    if bw >= 5:
        out['resist'] = body[:, 4]
    if bw >= 6:
        out['dev'] = body[:, 5]
    df = pd.DataFrame(out)
    df['ip'] = np.nan
    return df


def _unified_ert_parser(fname):
    """Parse the pyGIMLi / BERT / E4D unified ERT format (no resipy required).

    Handles count-prefixed blocks (n_electrodes / electrode rows / n_data / data
    rows), optional ``# token`` header lines, optional leading index columns, and
    electrode rows of 2-5 columns. Returns ``(elec, df)`` like the other embedded
    parsers, with real electrode coordinates (including topography) and correctly
    named value columns, so apparent-resistivity vs resistance is detected right.
    """
    with open(fname, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read().splitlines()

    num_re = re.compile(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+')
    seq = []  # ("header", [names]) | ("nums", [floats])
    for ln in raw:
        s = ln.strip()
        if not s or s[0] in "!*":
            continue
        if s[0] == '#':
            names = [t for t in re.split(r'[\s,]+', s[1:].strip()) if t]
            if names:
                seq.append(("header", names))
            continue
        body = s.split('#')[0].split('!')[0]
        nums = num_re.findall(body)
        if nums:
            seq.append(("nums", [float(x) for x in nums]))
    if not seq:
        raise ValueError("unified ERT parser: no numeric content found")

    idx = 0

    def next_count():
        nonlocal idx
        while idx < len(seq) and seq[idx][0] != "nums":
            idx += 1
        if idx >= len(seq):
            return None
        val = int(round(seq[idx][1][0]))
        idx += 1
        return val

    def maybe_header():
        nonlocal idx
        if idx < len(seq) and seq[idx][0] == "header":
            h = seq[idx][1]
            idx += 1
            return h
        return None

    def take_rows(count):
        nonlocal idx
        rows = []
        limit = count if (count and count > 0) else len(seq)
        while idx < len(seq) and len(rows) < limit:
            if seq[idx][0] == "nums":
                rows.append(seq[idx][1])
            idx += 1
        return rows

    n_elec = next_count()
    if not n_elec or n_elec <= 0:
        raise ValueError("unified ERT parser: missing/invalid electrode count")
    elec_header = maybe_header()
    elec_rows = take_rows(n_elec)
    if len(elec_rows) < n_elec:
        raise ValueError(f"unified ERT parser: found {len(elec_rows)}/{n_elec} electrode rows")
    elec = _parse_elec_rows(elec_rows, elec_header)

    n_data = next_count()
    if n_data is None:
        raise ValueError("unified ERT parser: missing data block")
    data_header = maybe_header()
    data_rows = take_rows(n_data)
    if not data_rows:
        raise ValueError("unified ERT parser: no measurement rows")
    df = _parse_data_rows(data_rows, data_header)
    return elec, df


def _bertParser(fname):
    """BERT / E4D / unified ERT parser.

    Tries the robust :func:`_unified_ert_parser` first (count-prefixed blocks with
    topography and header-aware columns) and falls back to the legacy CSV /
    positional reader for plain files.
    """
    try:
        return _unified_ert_parser(fname)
    except Exception:
        return _bertParser_legacy(fname)


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
            except Exception:
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
            except Exception:
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


def _abem_lund_parser(fname):
    """
    Parse ABEM/Lund Terameter LS style .dat files.

    Common data row structure (12+ numeric columns) is:
      type xA yA xB yB xM yM xN yN <meta...>
    We extract ABMN from x-coordinates (xA, xB, xM, xN), and use the last
    two trailing numeric columns as resistivity-like values.
    """
    num_str = r'[-+]?\d*\.\d*[eE]?[-+]?\d+|\d+'
    rows = []
    try:
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                vals = re.findall(num_str, line.strip())
                if len(vals) < 11:
                    continue
                nums = [float(v) for v in vals]
                kind = int(round(nums[0]))
                # ABEM LS measurement rows are typically flagged as type=4.
                if kind != 4:
                    continue

                a = nums[1]
                b = nums[3]
                m = nums[5]
                n = nums[7]

                tail = nums[9:]
                if len(tail) >= 3:
                    dev = tail[0]
                    resist = tail[-2]
                    app = tail[-1]
                elif len(tail) == 2:
                    dev = np.nan
                    resist = tail[0]
                    app = tail[1]
                else:
                    dev = np.nan
                    resist = tail[0]
                    app = np.nan

                rows.append((a, b, m, n, resist, app, dev))
    except Exception as e:
        raise ValueError(f"Could not parse ABEM-Lund file {fname}: {e}")

    if not rows:
        raise ValueError("No ABEM-Lund measurement rows detected")

    df = pd.DataFrame(rows, columns=["a", "b", "m", "n", "resist", "app", "dev"])
    for col in ["a", "b", "m", "n"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[["a", "b", "m", "n"]].isna().all().any():
        raise ValueError("ABEM-Lund parser failed to extract ABMN coordinates.")

    unique_elec = np.sort(np.unique(df[["a", "b", "m", "n"]].values.astype(float).flatten()))
    elec = np.c_[unique_elec, np.zeros_like(unique_elec), np.zeros_like(unique_elec)]

    if "ip" not in df.columns:
        df["ip"] = np.nan

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
    """Local coordinate reference information for profile-based ERT data."""

    origin_x: float = 0.0   # optional world X of profile start
    origin_y: float = 0.0   # optional world Y of profile start
    azimuth_deg: float = 0.0  # profile direction (deg clockwise from north)


@dataclass
class Electrode:
    """Electrode position and identifier for standardized ERT datasets."""

    id: int
    x: float
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quadruplet:
    """Current and potential electrode indices for one ERT measurement."""

    A: int; B: int; M: int; N: int


@dataclass
class Observation:
    """Single apparent-resistivity observation and its supporting metadata."""

    quad: Quadruplet
    app_res: float | None = None   # apparent resistivity (ohm·m)
    dV: float | None = None        # potential difference (V)
    I: float | None = None         # injected current (A)
    rel_err: float | None = 0.03   # relative error fraction (e.g., 0.03)
    K: float | None = None         # geometric factor
    fid: str | None = None         # field id/record id


@dataclass
class StandardERT:
    """Standardized ERT container with coordinates, observations, and metadata."""

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


def _normalize_elevation_axis(electrodes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize 2D electrode coordinates so elevation is carried in `y`.

    Some loaders provide 2D profile coordinates as (x, 0, z_elev).
    PyGIMLi ERT workflows in this project treat `y` as elevation.
    This helper preserves the original elevation values and only remaps
    axes when it is clearly a 2D case (flat y, varying z).
    """
    if electrodes_df is None or len(electrodes_df) == 0:
        return electrodes_df

    out = electrodes_df.copy()
    if 'y' not in out.columns:
        out['y'] = 0.0
    if 'z' not in out.columns:
        out['z'] = 0.0

    y_vals = pd.to_numeric(out['y'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    z_vals = pd.to_numeric(out['z'], errors='coerce').fillna(0.0).to_numpy(dtype=float)

    y_span = float(np.nanmax(y_vals) - np.nanmin(y_vals)) if len(y_vals) else 0.0
    z_span = float(np.nanmax(z_vals) - np.nanmin(z_vals)) if len(z_vals) else 0.0

    # Only remap when elevation is clearly carried by z and y is effectively flat.
    if y_span < 1e-8 and z_span > 1e-8:
        out['y'] = z_vals
        out['z'] = 0.0

    return out


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

_INSTRUMENT_ALIAS_MAP = {
    "abem": "ABEM-Lund",
    "abem lund": "ABEM-Lund",
    "abem-lund": "ABEM-Lund",
    "abem terameter": "ABEM-Lund",
    "abem terameter ls": "ABEM-Lund",
    "terameter": "ABEM-Lund",
    "terameter ls": "ABEM-Lund",
    "das": "DAS-1",
    "das 1": "DAS-1",
    "das-1": "DAS-1",
    "syscal": "Syscal",
    "syscal pro": "Syscal",
    "bert": "BERT",
    "e4d": "E4D",
}

def _normalize_instrument_name(instrument: str) -> str:
    """
    Normalize user- or LLM-provided instrument aliases to supported canonical names.
    """
    if instrument in _FTYPE_MAP:
        return instrument
    token = re.sub(r'[^a-z0-9]+', ' ', str(instrument).strip().lower()).strip()
    if token in _INSTRUMENT_ALIAS_MAP:
        return _INSTRUMENT_ALIAS_MAP[token]
    if "abem" in token or ("terameter" in token and "ls" in token):
        return "ABEM-Lund"
    if token.startswith("das"):
        return "DAS-1"
    if "syscal" in token:
        return "Syscal"
    return instrument


def _looks_like_abem_lund_file(data_file_path: Path) -> bool:
    """
    Heuristically detect ABEM Terameter/Lund-style exports.
    """
    try:
        with open(data_file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline().strip() for _ in range(60)]
    except Exception:
        return False

    lines = [ln for ln in lines if ln]
    if not lines:
        return False

    header = "\n".join(lines[:25]).lower()
    has_abem_tag = "type of measurement" in header

    type4_rows = 0
    for ln in lines:
        parts = ln.split()
        if parts and parts[0] == "4":
            type4_rows += 1

    # ABEM exports usually contain the explicit "Type of measurement" line
    # and many data rows starting with "4".
    return bool(has_abem_tag and type4_rows >= 3)

def _to_ftype(instrument: Instrument) -> str:
    normalized = _normalize_instrument_name(str(instrument))
    if normalized not in _FTYPE_MAP:
        raise ValueError(f"Unsupported instrument: {instrument}")
    return _FTYPE_MAP[normalized]


# Parser mapping for embedded parsers (modified from ResIPy)
_EMBEDDED_PARSER_MAP = {
    "Syscal": _syscalParser,
    "Protocol DC": _protocolParser,
    "Protocol IP": _protocolParser,
    "BERT": _bertParser,
    "E4D": _bertParser,
    "DAS-1": _dasParser,
    "ABEM-Lund": _abem_lund_parser,
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
    requested_instrument = str(instrument)
    instrument = _normalize_instrument_name(requested_instrument)
    if instrument != requested_instrument:
        _notify(f"[PyHydroGeophysX] Normalized instrument '{requested_instrument}' -> '{instrument}'")

    data_file_path = Path(data_file)
    if not data_file_path.is_absolute():
        data_file_path = Path.cwd() / data_file_path
    
    if not data_file_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file_path}")

    # Guardrail: if the file clearly looks like ABEM-Lund but the instrument was
    # specified differently (common in quick mode), override to the safer parser.
    if instrument != "ABEM-Lund" and _looks_like_abem_lund_file(data_file_path):
        print(
            f"[PyHydroGeophysX] Detected ABEM-Lund style file; "
            f"overriding instrument '{instrument}' -> 'ABEM-Lund'"
        )
        instrument = "ABEM-Lund"
    
    # Select appropriate parser based on instrument
    parser_func = _EMBEDDED_PARSER_MAP.get(instrument, _bertParser)
    parser_name = instrument  # Use instrument name for error message

    if parser_func is None:
        parser_func = _bertParser  # Default fallback
        parser_name = "BERT (fallback)"

    # Parse the data file
    try:
        _notify(f"[PyHydroGeophysX] Attempting to parse ERT data with {parser_name} parser...")
        elec_array, df = parser_func(str(data_file_path))
        _notify(f"[PyHydroGeophysX] Successfully parsed {len(df)} measurements and {len(elec_array)} electrodes")
    except Exception as e:
        _notify(f"[PyHydroGeophysX] Parser {parser_name} failed: {e}")
        # Try fallback parsers if this one fails
        if parser_func != _bertParser:
            _notify(f"[PyHydroGeophysX] Trying BERT parser as fallback...")
            try:
                elec_array, df = _bertParser(str(data_file_path))
                _notify(f"[PyHydroGeophysX] BERT parser fallback succeeded")
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

    # Keep original elevation from data when no external electrode file is provided.
    if electrode_file is None:
        electrodes_df = _normalize_elevation_axis(electrodes_df)

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
    n_elec = int(len(electrodes_df))
    elec_x = pd.to_numeric(electrodes_df['x'], errors='coerce').to_numpy(dtype=float)
    elec_valid = np.isfinite(elec_x)
    elec_x_valid = elec_x[elec_valid]
    elec_idx_valid = np.arange(1, n_elec + 1, dtype=int)[elec_valid]

    def _coerce_indices(col_name: str, fallback_value: int) -> pd.Series:
        if col_name not in df.columns:
            return pd.Series(np.full(len(df), fallback_value, dtype=int))
        series = df[col_name]
        numeric = pd.to_numeric(series, errors='coerce')
        if numeric.notna().all():
            values = numeric.to_numpy(dtype=float)
            rounded = np.rint(values)

            # Case 1: already 1-based electrode indices
            if np.all(np.abs(values - rounded) < 1e-8):
                ints = rounded.astype(int)
                if ints.min() >= 1 and ints.max() <= n_elec:
                    return pd.Series(ints, index=series.index, dtype=int)
                # Case 2: 0-based indices
                if ints.min() >= 0 and ints.max() <= (n_elec - 1) and np.any(ints == 0):
                    return pd.Series(ints + 1, index=series.index, dtype=int)

            # Case 3: ABMN are coordinate-like values (e.g., ABEM Terameter exports)
            if len(elec_x_valid) > 0:
                # Derive matching tolerance from electrode spacing.
                tol = 1e-6
                uniq = np.unique(np.sort(elec_x_valid))
                if len(uniq) > 1:
                    diffs = np.diff(uniq)
                    diffs = diffs[diffs > 1e-8]
                    if len(diffs) > 0:
                        tol = max(tol, float(np.min(diffs)) * 0.25)

                dist = np.abs(values[:, None] - elec_x_valid[None, :])
                nearest_pos = np.argmin(dist, axis=1)
                nearest_dist = dist[np.arange(len(values)), nearest_pos]
                if np.all(np.isfinite(nearest_dist)) and np.max(nearest_dist) <= tol:
                    mapped = elec_idx_valid[nearest_pos]
                    return pd.Series(mapped.astype(int), index=series.index, dtype=int)

            # Case 4: numeric labels that are neither direct indices nor coordinates.
            # Map sorted unique values to sequential electrode IDs.
            uniq_vals = np.unique(values)
            if len(uniq_vals) <= n_elec:
                rank_map = {v: i + 1 for i, v in enumerate(np.sort(uniq_vals))}
                mapped = np.array([rank_map[v] for v in values], dtype=int)
                return pd.Series(mapped, index=series.index, dtype=int)

            # Last fallback: preserve previous behavior.
            return pd.Series(rounded.astype(int), index=series.index, dtype=int)
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
    app_res_source = "unknown"
    rho_cols = ['app', 'rhoa', 'rhoA', 'Rhoa', 'app_res']
    if any(c in df.columns for c in rho_cols):
        rho_col = next(c for c in rho_cols if c in df.columns)
        observations['rhoa'] = pd.to_numeric(df[rho_col], errors='coerce')
        app_res_source = "rhoa"
    elif 'resist' in observations.columns:
        # Use resistance values (will be converted to rhoa with K later)
        observations['rhoa'] = observations['resist']
        app_res_source = "resistance"
    else:
        observations['rhoa'] = 100.0  # Default
    observations['rhoa'] = pd.to_numeric(observations['rhoa'], errors='coerce')
    if observations['rhoa'].isna().all():
        observations['rhoa'] = 100.0

    # Compute error estimates using reciprocal pairs when available and robustly
    # fallback to source error columns (e.g., ABEM "dev") when reciprocal pairs
    # are missing.
    err_col = next((c for c in ['error', 'err', 'dev', 'std', 'std_res'] if c in df.columns), None)

    def _normalize_source_error(err_series: pd.Series, rho_series: pd.Series) -> np.ndarray:
        err_raw = pd.to_numeric(err_series, errors='coerce').to_numpy(dtype=float)
        rho_raw = pd.to_numeric(rho_series, errors='coerce').to_numpy(dtype=float)
        rel = np.full(err_raw.shape, np.nan, dtype=float)

        finite_pos = err_raw[np.isfinite(err_raw) & (err_raw > 0)]
        if finite_pos.size == 0:
            return rel

        med = float(np.nanmedian(finite_pos))
        q95 = float(np.nanpercentile(finite_pos, 95))

        # Heuristic:
        #  - <=2   : likely already relative fraction (e.g., 0.12)
        #  - <=200 : likely percent (e.g., 12.0 -> 0.12)
        #  - else  : likely absolute error in data units -> convert with rho
        if q95 <= 2.0:
            rel = np.abs(err_raw)
        elif q95 <= 200.0:
            rel = np.abs(err_raw) / 100.0
        else:
            rel = np.where(
                np.isfinite(rho_raw) & (np.abs(rho_raw) > 1e-12),
                np.abs(err_raw) / np.abs(rho_raw),
                np.nan,
            )

        rel = np.where(np.isfinite(rel) & (rel > 0), rel, np.nan)
        return np.clip(rel, 0.01, 0.50)

    # Reciprocal filtering preserves the original parser row index but can
    # remove rows. Align source error columns to the retained observations so
    # fallback error estimates have the same length as reciprocal estimates.
    source_error_series = df[err_col].reindex(observations.index) if err_col else None
    source_rel = (
        _normalize_source_error(source_error_series, observations['rhoa'])
        if source_error_series is not None
        else None
    )

    if 'reciprocalErrRel' in observations.columns:
        recip_rel = pd.to_numeric(observations['reciprocalErrRel'], errors='coerce').to_numpy(dtype=float)
        recip_rel = np.where(np.isfinite(recip_rel) & (np.abs(recip_rel) > 0), np.abs(recip_rel), np.nan)
        n_recip_finite = int(np.sum(np.isfinite(recip_rel)))

        if n_recip_finite > 0:
            merged_err = recip_rel
            if source_rel is not None:
                merged_err = np.where(np.isfinite(merged_err), merged_err, source_rel)
            merged_err = np.where(np.isfinite(merged_err), merged_err, 0.05)
            observations['error'] = np.clip(merged_err, 0.01, 0.50)
            print(
                f"   Using reciprocal-based error estimates "
                f"(finite={n_recip_finite}/{len(recip_rel)}, mean: {observations['error'].mean():.4f})"
            )
        elif source_rel is not None and np.any(np.isfinite(source_rel)):
            observations['error'] = np.where(np.isfinite(source_rel), source_rel, 0.05)
            observations['error'] = np.clip(observations['error'], 0.01, 0.50)
            print(
                f"   No reciprocal pairs found; using source '{err_col}' error estimates "
                f"(mean: {observations['error'].mean():.4f})"
            )
        else:
            observations['error'] = 0.05
            print("   No reciprocal/source error available; using default 5% error estimates")
    else:
        if source_rel is not None and np.any(np.isfinite(source_rel)):
            observations['error'] = np.where(np.isfinite(source_rel), source_rel, 0.05)
            observations['error'] = np.clip(observations['error'], 0.01, 0.50)
            print(f"   Using source '{err_col}' error estimates (mean: {observations['error'].mean():.4f})")
        else:
            observations['error'] = 0.05
            print("   Using default 5% error estimates")

    observations['valid'] = True

    # Convert to dataclass lists for downstream compatibility
    electrodes_list = [
        Electrode(i + 1, float(row['x']), float(row['y']), float(row['z']))
        for i, row in electrodes_df.iterrows()
    ]

    # Carry the file's own geometric factors. When the file reported apparent
    # resistivity, those are the factors rhoa was built with, and the inversion
    # needs them to tell "k disagrees with the geometry" apart from "rhoa was
    # formed under a different convention". Both scale the section; neither
    # shows up in chi2.
    k_col = next((c for c in ['k', 'K', 'geom', 'geom_factor'] if c in df.columns), None)
    file_k = (pd.to_numeric(df[k_col], errors='coerce').to_numpy(dtype=float)
              if k_col is not None else None)

    obs_list: List[Observation] = []
    for pos, (idx, row) in enumerate(observations.iterrows()):
        app_res_val = float(row['rhoa']) if np.isfinite(row['rhoa']) else None
        rel_err_val = float(row['error']) if np.isfinite(row['error']) else 0.05
        k_val = 1.0
        if file_k is not None and pos < file_k.size and np.isfinite(file_k[pos]) \
                and file_k[pos] != 0.0:
            k_val = float(file_k[pos])
        obs_list.append(Observation(
            quad=Quadruplet(int(row['a']), int(row['b']), int(row['m']), int(row['n'])),
            app_res=app_res_val,
            dV=None,
            I=None,
            rel_err=rel_err_val,
            K=k_val,
            fid=str(idx)
        ))
    
    # Build metadata
    metadata = {
        'source_file': str(data_file_path),
        'loader': 'local_parsers_resipy_fallback',
        'parser_used': parser_name,
        'instrument': instrument,
        'requested_instrument': requested_instrument,
        'app_res_source': app_res_source,
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
    requested_instrument = str(instrument)
    instrument = _normalize_instrument_name(requested_instrument)
    if instrument != requested_instrument:
        _notify(f"[PyHydroGeophysX] Normalized instrument '{requested_instrument}' -> '{instrument}'")

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

    # Keep original elevation from data when no external electrode file is provided.
    if electrode_file is None:
        electrodes_df = _normalize_elevation_axis(electrodes_df)

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

    # Some BERT-style files load through PyGIMLi with indices shifted by -1
    # (e.g., [0..N-1] becomes [-1..N-2]). Normalize back to non-negative
    # indexing so downstream export and validation keep all valid measurements.
    abmn_all = np.concatenate([a, b, m, n]).astype(float)
    finite_abmn = abmn_all[np.isfinite(abmn_all)]
    if finite_abmn.size > 0:
        min_idx = int(np.min(finite_abmn))
        max_idx = int(np.max(finite_abmn))
        n_elec = len(electrodes_df)
        if min_idx < 0 and max_idx <= max(n_elec - 2, 0):
            a = a + 1
            b = b + 1
            m = m + 1
            n = n + 1
            print("   Detected shifted ABMN indices from PyGIMLi loader; applied +1 correction.")
    
    # Get apparent resistivity or resistance
    app_res_source = "unknown"
    if 'rhoa' in data.dataMap():
        rhoa = np.array(data('rhoa'))
        app_res_source = "rhoa"
    elif 'r' in data.dataMap():
        # Convert resistance to apparent resistivity using geometric factor
        r = np.array(data('r'))
        app_res_source = "resistance"
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
        'requested_instrument': requested_instrument,
        'app_res_source': app_res_source,
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
    requested_instrument = str(instrument)
    instrument = _normalize_instrument_name(requested_instrument)
    if instrument != requested_instrument:
        _notify(f"[PyHydroGeophysX] Normalized instrument '{requested_instrument}' -> '{instrument}'")

    # Try resipy first, then local parsers, then pygimli
    if not _HAS_RESIPY:
        # Fallback 1: Embedded parsers (modified from ResIPy, GPL-3.0)
        if _HAS_EMBEDDED_PARSERS:
            _notify(f"[PyHydroGeophysX] Using embedded parsers (modified from ResIPy) - RESIPY unavailable: {_RESIPY_ERROR}")
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
            _notify(f"[PyHydroGeophysX] Using PyGIMLi fallback for ERT data loading (RESIPY unavailable: {_RESIPY_ERROR})")
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

    # Guardrail: if file content clearly indicates ABEM/Lund export, switch
    # parser target even when user/LLM selected a different instrument.
    if instrument != "ABEM-Lund" and _looks_like_abem_lund_file(data_file_path):
        print(
            f"[PyHydroGeophysX] Detected ABEM-Lund style file; "
            f"overriding instrument '{instrument}' -> 'ABEM-Lund'"
        )
        instrument = "ABEM-Lund"
        ftype = _to_ftype(instrument)

    # Prefer to use the requested project_dir, but RESIPY may attempt to
    # remove/recreate the directory (calling shutil.rmtree) which can fail
    # on Windows (OneDrive or open handles). Try to instantiate Project and
    # if a PermissionError (or OSError with permission) occurs, fall back to
    # a temporary directory and warn the user.
    chosen_dir = project_dir
    Path(chosen_dir).mkdir(parents=True, exist_ok=True)
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
    except Exception as e:
        # Check for NumPy compatibility issues with pyGIMLi
        if isinstance(e, ValueError) and (
            "Buffer dtype mismatch" in str(e) or "dtype mismatch" in str(e).lower()
        ):
            raise RuntimeError(
                f"NumPy compatibility error with pyGIMLi/RESIPY: {str(e)}\n\n"
                "This is a known issue with NumPy 2.x and pyGIMLi. Try:\n"
                "  conda install numpy=1.26.4\n"
                "or:\n"
                "  pip install 'numpy<2.0'\n\n"
                "If the error persists, you may also need to reinstall pygimli:\n"
                "  conda install -c gimli -c conda-forge pygimli"
            ) from e

        warnings.warn(
            f"RESIPY failed to parse '{data_file_path.name}' ({e}). "
            "Falling back to embedded parsers/PyGIMLi loader.",
            UserWarning
        )

        instrument_upper = str(instrument).strip().upper()
        # For BERT/unified files, prefer PyGIMLi fallback first because it preserves
        # native electrode coordinates/topography more reliably than the lightweight parser.
        if instrument_upper == "BERT":
            if _HAS_PYGIMLI:
                try:
                    return _load_ert_pygimli(
                        data_file=data_file,
                        electrode_file=electrode_file,
                        project_dir=project_dir,
                        instrument=instrument,
                        crs=crs,
                        epsg=epsg,
                        local_ref=local_ref
                    )
                except Exception as pygimli_err:
                    warnings.warn(
                        f"PyGIMLi fallback failed after RESIPY parse failure: {pygimli_err}. "
                        "Trying embedded parser fallback.",
                        UserWarning
                    )
            if _HAS_EMBEDDED_PARSERS:
                return _load_ert_embedded_parsers(
                    data_file=data_file,
                    electrode_file=electrode_file,
                    project_dir=project_dir,
                    instrument=instrument,
                    crs=crs,
                    epsg=epsg,
                    local_ref=local_ref
                )
            raise

        # Default fallback order for non-BERT instruments:
        # 1) Embedded parsers (ResIPy-derived) -> 2) PyGIMLi
        if _HAS_EMBEDDED_PARSERS:
            try:
                return _load_ert_embedded_parsers(
                    data_file=data_file,
                    electrode_file=electrode_file,
                    project_dir=project_dir,
                    instrument=instrument,
                    crs=crs,
                    epsg=epsg,
                    local_ref=local_ref
                )
            except Exception as embedded_err:
                if _HAS_PYGIMLI:
                    warnings.warn(
                        f"Embedded parser fallback failed: {embedded_err}. Trying PyGIMLi fallback.",
                        UserWarning
                    )
                else:
                    raise
        if _HAS_PYGIMLI:
            return _load_ert_pygimli(
                data_file=data_file,
                electrode_file=electrode_file,
                project_dir=project_dir,
                instrument=instrument,
                crs=crs,
                epsg=epsg,
                local_ref=local_ref
            )
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
    if elec_arr.ndim == 1:
        elec_arr = elec_arr.reshape(-1, 1)
    if elec_arr.shape[1] < 3:
        pad = np.zeros((elec_arr.shape[0], 3 - elec_arr.shape[1]))
        elec_arr = np.hstack([elec_arr, pad])

    electrodes_df = pd.DataFrame({
        'x': elec_arr[:, 0],
        'y': elec_arr[:, 1],
        'z': elec_arr[:, 2],
    })
    # If no external electrode file is provided, preserve source elevation.
    if electrode_file is None:
        electrodes_df = _normalize_elevation_axis(electrodes_df)

    electrodes = [
        Electrode(i + 1, float(row['x']), float(row['y']), float(row['z']))
        for i, row in electrodes_df.iterrows()
    ]

    # Observations
    idx_to_pos = {idx: k for k, idx in enumerate(df.index)}
    observations: List[Observation] = []
    
    # Detect column names
    i_col = next((c for c in ['i', 'I', 'current'] if c in df.columns), None)
    v_col = next((c for c in ['u', 'U', 'v', 'V', 'voltage', 'dV'] if c in df.columns), None)
    r_col = next((c for c in ['resist', 'R', 'r', 'resistance'] if c in df.columns), None)
    # BERT/PyGIMLi format uses 'rhoa' for apparent resistivity directly
    rhoa_col = next((c for c in ['rhoa', 'rhoA', 'Rhoa', 'app_res'] if c in df.columns), None)
    if rhoa_col is not None:
        app_res_source = "rhoa"
    elif r_col is not None or (v_col is not None and i_col is not None):
        app_res_source = "resistance"
    else:
        app_res_source = "unknown"
    
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
        "app_res_source": app_res_source,
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


def calculate_reciprocal_errors(ert: StandardERT) -> pd.DataFrame:
    """
    Estimate reciprocal error for each standardized ERT observation.

    Reciprocal pairs are matched by comparing the current pair ``(A, B)`` and
    potential pair ``(M, N)`` after sorting within each pair. Measurements where
    those two pair roles are swapped are treated as reciprocal observations.
    The returned error is a percentage,
    ``200 * |R_normal - R_recip| / (|R_normal| + |R_recip|)``.

    Parameters
    ----------
    ert : StandardERT
        Standardized ERT dataset.

    Returns
    -------
    pandas.DataFrame
        One row per observation with reciprocal metadata and error estimates.
        Unmatched observations have ``NaN`` reciprocal errors.
    """

    if ert is None or not ert.observations:
        return pd.DataFrame(
            columns=[
                "observation_index",
                "reciprocal_group",
                "reciprocal_pair_count",
                "reciprocal_error_percent",
                "reciprocal_mean_value",
                "reciprocal_partner_value",
            ]
        )

    records: List[Dict[str, Any]] = []
    groups: Dict[tuple[tuple[int, int], tuple[int, int]], List[int]] = {}
    directions: List[tuple[tuple[int, int], tuple[int, int]]] = []
    values: List[float] = []

    for idx, obs in enumerate(ert.observations):
        a = int(obs.quad.A)
        b = int(obs.quad.B)
        m = int(obs.quad.M)
        n = int(obs.quad.N)
        ab = tuple(sorted((a, b)))
        mn = tuple(sorted((m, n)))
        direction = (ab, mn)
        group_key = tuple(sorted((ab, mn)))
        value = np.nan if obs.app_res is None else float(obs.app_res)
        directions.append(direction)
        values.append(value)
        groups.setdefault(group_key, []).append(idx)
        records.append(
            {
                "observation_index": idx,
                "A": a,
                "B": b,
                "M": m,
                "N": n,
                "value": value,
                "reciprocal_group": f"{group_key[0][0]}-{group_key[0][1]}|{group_key[1][0]}-{group_key[1][1]}",
                "reciprocal_pair_count": 0,
                "reciprocal_error_percent": np.nan,
                "reciprocal_mean_value": np.nan,
                "reciprocal_partner_value": np.nan,
            }
        )

    for group_indices in groups.values():
        direction_to_indices: Dict[tuple[tuple[int, int], tuple[int, int]], List[int]] = {}
        for idx in group_indices:
            direction_to_indices.setdefault(directions[idx], []).append(idx)

        if len(direction_to_indices) < 2:
            continue

        direction_medians: Dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for direction, indices in direction_to_indices.items():
            direction_values = np.asarray([values[i] for i in indices], dtype=float)
            direction_values = direction_values[np.isfinite(direction_values)]
            if direction_values.size:
                direction_medians[direction] = float(np.nanmedian(np.abs(direction_values)))

        for idx in group_indices:
            ab, mn = directions[idx]
            opposite_direction = (mn, ab)
            if opposite_direction not in direction_medians:
                continue
            value = abs(float(values[idx]))
            partner_value = float(direction_medians[opposite_direction])
            if not (np.isfinite(value) and np.isfinite(partner_value)):
                continue
            denom = value + abs(partner_value)
            if denom <= 1e-12:
                continue
            reciprocal_error = 200.0 * abs(value - abs(partner_value)) / denom
            reciprocal_mean = 0.5 * (value + abs(partner_value))
            partner_count = len(direction_to_indices.get(opposite_direction, []))
            records[idx]["reciprocal_pair_count"] = int(partner_count)
            records[idx]["reciprocal_error_percent"] = float(reciprocal_error)
            records[idx]["reciprocal_mean_value"] = float(reciprocal_mean)
            records[idx]["reciprocal_partner_value"] = float(partner_value)

    return pd.DataFrame.from_records(records)

def export_for_inversion(
    ert: StandardERT,
    outdir: str = "examples/results/ert",
    fmt: str = "pgimli",
    use_source_error: bool = False,
    export_strategy: str = "default",
    default_relative_error: float = 0.01,
    default_absolute_error: float = 0.001,
    default_rhoa_limits: tuple[float, float] = (0.1, 10000.0),
    default_reciprocal_percent: float = 10.0,
    default_fit_error_lin: bool = True,
) -> str:
    """
    Export to inversion-ready formats:
    - fmt='pgimli': Unified data format for pyGIMLi/BERT with electrode coordinates and measurements
    - fmt='resipy': return the RESIPY project directory for running prj.start().
    - export_strategy='default' (default): rebuild the raw survey in ResIPy,
      keep only reciprocal-paired measurements, export recipMean as resistance,
      and use abs((relative * resist + absolute) / recipMean) for the error.
      If that path is unavailable, the function falls back to the legacy export.
    - export_strategy='legacy': use the older full-dataset export path.
    - use_source_error only affects the legacy export path.
    """
    def _coord_stats(x_vals: np.ndarray, y_vals: np.ndarray, z_vals: np.ndarray) -> str:
        return (
            f"x=[{np.nanmin(x_vals):.3f}, {np.nanmax(x_vals):.3f}], "
            f"y=[{np.nanmin(y_vals):.3f}, {np.nanmax(y_vals):.3f}], "
            f"z=[{np.nanmin(z_vals):.3f}, {np.nanmax(z_vals):.3f}]"
        )

    def _infer_elevation_axis(y_vals: np.ndarray, z_vals: np.ndarray) -> str:
        y_span = float(np.nanmax(y_vals) - np.nanmin(y_vals))
        z_span = float(np.nanmax(z_vals) - np.nanmin(z_vals))
        eps = 1e-8
        if y_span > eps and z_span <= eps:
            return "y"
        if z_span > eps and y_span <= eps:
            return "z"
        if y_span > 2.0 * max(z_span, eps):
            return "y"
        if z_span > 2.0 * max(y_span, eps):
            return "z"
        return "ambiguous"

    def _estimate_reciprocal_error(
        abmn: np.ndarray,
        resist: np.ndarray,
        default_err: float = 0.05,
        min_err: float = 0.01,
        max_err: float = 0.10,
    ) -> tuple[np.ndarray, int]:
        """
        Estimate relative data errors from reciprocal pairs using resistance values.
        For unmatched measurements, use a conservative default error.
        """
        n_obs = int(len(resist))
        if n_obs == 0:
            return np.zeros(0, dtype=float), 0

        errs = np.full(n_obs, float(default_err), dtype=float)
        if abmn.shape[0] != n_obs:
            errs = np.clip(errs, min_err, max_err)
            return errs, 0

        groups: Dict[tuple[int, int, int, int], List[int]] = {}
        for idx in range(n_obs):
            a, b, m, n = [int(v) for v in abmn[idx]]
            key = (min(a, b), max(a, b), min(m, n), max(m, n))
            groups.setdefault(key, []).append(idx)

        paired_measurements = 0
        visited: set[tuple[int, int, int, int]] = set()
        for key, left_idx in groups.items():
            if key in visited:
                continue
            recip = (key[2], key[3], key[0], key[1])
            right_idx = groups.get(recip, [])
            visited.add(key)
            visited.add(recip)

            if recip == key:
                continue
            if len(left_idx) == 0 or len(right_idx) == 0:
                continue

            n_pair = min(len(left_idx), len(right_idx))
            for li, ri in zip(left_idx[:n_pair], right_idx[:n_pair]):
                r_left = abs(float(resist[li]))
                r_right = abs(float(resist[ri]))
                denom = 0.5 * (r_left + r_right)
                if not (np.isfinite(r_left) and np.isfinite(r_right) and denom > 1e-12):
                    continue
                recip_err = abs(r_left - r_right) / denom
                if not np.isfinite(recip_err):
                    continue
                recip_err = float(np.clip(recip_err, min_err, max_err))
                errs[li] = recip_err
                errs[ri] = recip_err
                paired_measurements += 2

        errs = np.where(np.isfinite(errs) & (errs > 0), errs, default_err)
        errs = np.clip(errs, min_err, max_err)
        return errs.astype(float), int(paired_measurements)

    def _sanitize_source_error(
        err_vals: np.ndarray,
        default_err: float,
        min_err: float,
        max_err: float,
    ) -> np.ndarray:
        """
        Normalize source error values to a robust relative-error fraction.
        If values appear to be percentages, convert by dividing by 100.
        """
        arr = np.array(err_vals, dtype=float)
        finite_pos = arr[np.isfinite(arr) & (arr > 0)]
        if finite_pos.size > 0:
            med = float(np.nanmedian(finite_pos))
            q95 = float(np.nanpercentile(finite_pos, 95))
            if med > 1.0 or q95 > 2.0:
                arr = arr / 100.0
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, default_err)
        return np.clip(arr, min_err, max_err).astype(float)

    # Handle paths starting with / on Windows by converting to relative path
    outdir_str = str(outdir)
    outdir_path = Path(outdir)
    if outdir_str.startswith('/') and not outdir_path.is_absolute():
        outdir_path = Path.cwd() / outdir_str.lstrip('/')

    try:
        outdir_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Unable to create export directory '{outdir_path}': {e}") from e

    if fmt == "pgimli":
        if ert is None:
            raise ValueError("ERT dataset is None")
        if not ert.electrodes:
            raise ValueError("ERT dataset has no electrodes; cannot export.")
        if not ert.observations:
            raise ValueError("ERT dataset has no observations; cannot export.")

        strategy_raw = str(export_strategy).strip().lower()
        if strategy_raw in {"", "default", "native", "recommended"}:
            requested_export_mode = "default"
        elif strategy_raw in {"legacy", "standard", "classic"}:
            requested_export_mode = "legacy"
        else:
            raise ValueError(
                f"Unsupported export_strategy: {export_strategy}. "
                "Expected 'default' or 'legacy'."
            )

        path_name = "bert_data.dat" if requested_export_mode == "default" else "bert_data_legacy.dat"
        path = outdir_path / path_name

        x_vals = np.array([float(e.x) for e in ert.electrodes], dtype=float)
        y_vals = np.array([float(e.y) for e in ert.electrodes], dtype=float)
        z_vals = np.array([float(e.z) for e in ert.electrodes], dtype=float)
        if not (np.all(np.isfinite(x_vals)) and np.all(np.isfinite(y_vals)) and np.all(np.isfinite(z_vals))):
            raise ValueError("Non-finite electrode coordinates found. Please clean input data first.")

        n_elec = len(ert.electrodes)
        if n_elec < 2:
            raise ValueError(f"Need at least 2 electrodes to export, got {n_elec}.")
        inferred_axis = _infer_elevation_axis(y_vals, z_vals)
        print(f"   DEBUG: Writing {n_elec} electrodes to file")
        print(f"   DEBUG: First electrode: x={x_vals[0]:.3f}, y={y_vals[0]:.3f}, z={z_vals[0]:.3f}")
        print(f"   DEBUG: Last electrode: x={x_vals[-1]:.3f}, y={y_vals[-1]:.3f}, z={z_vals[-1]:.3f}")
        print(f"   DEBUG: Coordinate ranges: {_coord_stats(x_vals, y_vals, z_vals)}")
        print(f"   DEBUG: Inferred elevation axis: {inferred_axis}")
        app_res_source = str((ert.metadata or {}).get("app_res_source", "")).strip().lower()
        app_res_is_apparent = app_res_source in {"rhoa", "apparent", "apparent_resistivity"}
        if app_res_source:
            print(f"   DEBUG: app_res source metadata: {app_res_source}")
        if app_res_is_apparent:
            print("   DEBUG: Treating app_res as apparent resistivity (rhoa) for K-handling.")

        rho_min = 0.1
        rho_max = 1e6
        reciprocal_default_err = 0.05
        reciprocal_min_err = 0.01
        reciprocal_max_err = 0.10
        source_default_err = 0.08
        source_min_err = 0.01
        source_max_err = 0.50
        metadata = ert.metadata or {}
        source_file = metadata.get("data_file") or metadata.get("source_file")
        source_path = None
        if source_file:
            source_path = Path(source_file)
            if not source_path.is_absolute():
                source_path = Path.cwd() / source_path

        can_use_default_export = bool(
            _HAS_RESIPY and source_path is not None and source_path.exists()
        )
        use_default_export = requested_export_mode == "default" and can_use_default_export
        if requested_export_mode == "default" and not use_default_export:
            reason = []
            if not _HAS_RESIPY:
                reason.append("ResIPy unavailable")
            if source_path is None:
                reason.append("raw source file missing from metadata")
            elif not source_path.exists():
                reason.append(f"raw source file not found: {source_path}")
            reason_text = "; ".join(reason) if reason else "default export unavailable"
            print(f"   DEBUG: Default reciprocal-pair export unavailable ({reason_text}); falling back to legacy export.")

        if use_default_export:
            print("   DEBUG: Export strategy: default reciprocal-pair path")
            print(
                "   DEBUG: Error mode: abs((relative * resist + absolute) / recipMean) "
                f"with relative={default_relative_error:.4f}, absolute={default_absolute_error:.4f}"
            )
        else:
            print("   DEBUG: Export strategy: legacy full-dataset path")
            print(
                "   DEBUG: Error mode: "
                + ("source (dataset-provided)" if use_source_error else "reciprocal (auto-estimated)")
            )
        valid_rows = []
        skipped_counts = {
            "invalid_abmn": 0,
            "index_out_of_range": 0,
            "degenerate_quad": 0,
            "missing_resistivity": 0,
            "non_finite_values": 0,
        }
        _fallback_to_legacy = False
        if use_default_export:
            ftype = str(metadata.get("ftype") or _to_ftype(ert.instrument))
            elec_xyz = np.column_stack((x_vals, y_vals, z_vals))
            if elec_xyz.shape[1] < 3:
                elec_xyz = np.pad(elec_xyz, ((0, 0), (0, 3 - elec_xyz.shape[1])), constant_values=0.0)

            try:
                with tempfile.TemporaryDirectory(prefix="phgx_default_export_") as tmp_project_dir:
                    prj = Project(tmp_project_dir)
                    prj.createSurvey(fname=str(source_path), ftype=ftype)
                    if not prj.surveys:
                        raise RuntimeError("ResIPy did not create a survey from the raw data file.")

                    survey = prj.surveys[0]
                    if hasattr(survey, "setElec"):
                        survey.setElec(elec_xyz)
                    else:
                        survey.elec = elec_xyz

                    rhoa_min, rhoa_max = default_rhoa_limits
                    filter_app_resist = getattr(prj, "filterAppResist", None)
                    if callable(filter_app_resist):
                        filter_app_resist(vmin=float(rhoa_min), vmax=float(rhoa_max))

                    filter_recip = getattr(prj, "filterRecip", None)
                    if not callable(filter_recip):
                        raise RuntimeError("ResIPy Project.filterRecip is not available in this environment.")
                    filter_recip(percent=float(default_reciprocal_percent))

                    if default_fit_error_lin:
                        fit_error_lin = getattr(prj, "fitErrorLin", None)
                        if callable(fit_error_lin):
                            fit_error_lin()

                    df = prj.surveys[0].df.copy()
            except Exception as e:
                if "no reciprocal" in str(e).lower():
                    print(
                        f"   Warning: No reciprocal measurements detected in survey data; "
                        f"falling back to legacy export. ({e})"
                    )
                    _fallback_to_legacy = True
                    use_default_export = False
                else:
                    raise RuntimeError(
                        f"Failed to rebuild the default reciprocal-pair export with ResIPy: {e}"
                    ) from e

            if use_default_export:
                required_cols = {"a", "b", "m", "n", "irecip", "recipMean", "resist"}
                missing_cols = sorted(required_cols.difference(df.columns))
                if missing_cols:
                    raise ValueError(
                        "Default reciprocal-pair export requires ResIPy columns "
                        f"{missing_cols}, but they were not found in the rebuilt survey."
                    )

                recip_mask = pd.to_numeric(df["irecip"], errors="coerce").fillna(0).to_numpy(dtype=float) > 0
                default_df = df.loc[recip_mask].copy()
                if len(default_df) == 0:
                    raise ValueError("Default reciprocal-pair export found no reciprocal-paired measurements after filtering.")

                print(f"   DEBUG: Default export retained {len(default_df)} reciprocal-paired measurements")
                skipped_counts["missing_resistivity"] = 0
                for _, row in default_df.iterrows():
                    try:
                        a = int(row["a"])
                        b = int(row["b"])
                        m = int(row["m"])
                        n = int(row["n"])
                    except Exception:
                        skipped_counts["invalid_abmn"] += 1
                        continue

                    if min(a, b, m, n) < 0 or max(a, b, m, n) > n_elec:
                        skipped_counts["index_out_of_range"] += 1
                        continue
                    if a == b or m == n:
                        skipped_counts["degenerate_quad"] += 1
                        continue

                    recip_mean = float(row["recipMean"])
                    resist_val = float(row["resist"])
                    if not (np.isfinite(recip_mean) and np.isfinite(resist_val)):
                        skipped_counts["non_finite_values"] += 1
                        continue

                    R = abs(recip_mean)
                    if R <= 1e-12:
                        skipped_counts["missing_resistivity"] += 1
                        continue

                    err_val = abs(((float(default_relative_error) * resist_val) + float(default_absolute_error)) / recip_mean)
                    if not (np.isfinite(err_val) and err_val > 0):
                        skipped_counts["non_finite_values"] += 1
                        continue

                    valid_rows.append((a, b, m, n, R, R, 1.0, float(err_val)))
        if not use_default_export:
            for obs in ert.observations:
                try:
                    a = int(obs.quad.A)
                    b = int(obs.quad.B)
                    m = int(obs.quad.M)
                    n = int(obs.quad.N)
                except Exception:
                    skipped_counts["invalid_abmn"] += 1
                    continue

                # Allow 0 for remote electrodes, but disallow negative or > n_elec indices.
                if min(a, b, m, n) < 0 or max(a, b, m, n) > n_elec:
                    skipped_counts["index_out_of_range"] += 1
                    continue
                if a == b or m == n:
                    skipped_counts["degenerate_quad"] += 1
                    continue

                R = None
                rhoa = None
                k = 1.0
                if (
                    obs.I is not None
                    and obs.dV is not None
                    and np.isfinite(obs.I)
                    and np.isfinite(obs.dV)
                    and obs.I != 0
                ):
                    R = abs(float(obs.dV) / float(obs.I))
                    rhoa = R
                    k = 1.0
                elif obs.app_res is not None and np.isfinite(obs.app_res):
                    if obs.K is not None and np.isfinite(obs.K) and float(obs.K) > 1:
                        k = float(obs.K)
                        R = float(obs.app_res) / k
                        rhoa = float(obs.app_res)
                    else:
                        R = float(obs.app_res)
                        rhoa = float(obs.app_res)
                        k = 1.0
                else:
                    skipped_counts["missing_resistivity"] += 1
                    continue

                if not (np.isfinite(R) and np.isfinite(rhoa) and np.isfinite(k)):
                    skipped_counts["non_finite_values"] += 1
                    continue

                R = abs(float(R))
                rhoa = abs(float(rhoa))
                k = abs(float(k)) if k != 0 else 1.0
                if R <= 0 or rhoa <= 0:
                    skipped_counts["missing_resistivity"] += 1
                    continue

                rhoa = float(np.clip(rhoa, rho_min, rho_max))
                R = float(max(R, rho_min))

                src_err_val = np.nan
                if obs.rel_err is not None and np.isfinite(obs.rel_err) and obs.rel_err > 0:
                    src_err_val = float(obs.rel_err)

                valid_rows.append((a, b, m, n, R, rhoa, k, src_err_val))

        if len(valid_rows) == 0:
            total_obs = len(ert.observations)
            details = ", ".join(f"{k}={v}" for k, v in skipped_counts.items())
            hint = ""
            if (
                total_obs > 0
                and skipped_counts.get("missing_resistivity", 0) >= int(0.8 * total_obs)
            ):
                hint = " Likely instrument/format mismatch (e.g., wrong instrument type selected)."
            raise ValueError(
                "No valid observations remained after validation; export aborted. "
                f"Diagnostics: {details}.{hint}"
            )

        if use_default_export:
            source_err_rows = np.array([r[7] for r in valid_rows], dtype=float)
            err_rows = np.where(
                np.isfinite(source_err_rows) & (source_err_rows > 0),
                source_err_rows,
                source_default_err,
            ).astype(float)
            n_source_valid = int(np.sum(np.isfinite(source_err_rows) & (source_err_rows > 0)))
            print(
                f"   DEBUG: Preserved default export errors on rows: "
                f"valid={n_source_valid}/{len(valid_rows)}, "
                f"err_range=[{np.nanmin(err_rows):.3f}, {np.nanmax(err_rows):.3f}]"
            )
        elif use_source_error:
            source_err_rows = np.array([r[7] for r in valid_rows], dtype=float)
            err_rows = _sanitize_source_error(
                source_err_rows,
                default_err=source_default_err,
                min_err=source_min_err,
                max_err=source_max_err,
            )
            n_source_valid = int(np.sum(np.isfinite(source_err_rows) & (source_err_rows > 0)))
            print(
                f"   DEBUG: Source-error use on export rows: "
                f"valid={n_source_valid}/{len(valid_rows)}, "
                f"err_range=[{np.nanmin(err_rows):.3f}, {np.nanmax(err_rows):.3f}]"
            )
        else:
            abmn_rows = np.array([(r[0], r[1], r[2], r[3]) for r in valid_rows], dtype=int)
            resist_rows = np.array([r[4] for r in valid_rows], dtype=float)
            err_rows, paired_meas = _estimate_reciprocal_error(
                abmn=abmn_rows,
                resist=resist_rows,
                default_err=reciprocal_default_err,
                min_err=reciprocal_min_err,
                max_err=reciprocal_max_err,
            )
            print(
                f"   DEBUG: Reciprocal-error estimate on export rows: "
                f"paired={paired_meas}/{len(valid_rows)}, "
                f"err_range=[{np.nanmin(err_rows):.3f}, {np.nanmax(err_rows):.3f}]"
            )

        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"{n_elec}\n")
            f.write("# x y z\n")
            for elec in ert.electrodes:
                f.write(f"{float(elec.x)} {float(elec.y)} {float(elec.z)}\n")

            f.write(f"{len(valid_rows)}\n")
            f.write("# a b m n r rhoa k err\n")

            for idx_row, row in enumerate(valid_rows):
                a, b, m, n, R, rhoa, k, _src_err = row
                err_val = float(err_rows[idx_row])
                f.write(f"{a} {b} {m} {n} {R} {rhoa} {k} {err_val}\n")

        if any(v > 0 for v in skipped_counts.values()):
            print(f"   DEBUG: Skipped invalid observations: {skipped_counts}")
        print(f"   DEBUG: Wrote {len(valid_rows)} valid observations")
        print(f"   Exported data to {path}")

        print("   Validating geometric factors... [CODE-VERSION: 2026-02-20-v4-safe-export]")
        try:
            import pygimli as pg
            import pygimli.physics.ert as ert_pg

            data = ert_pg.load(str(path))
            if data.size() == 0:
                raise RuntimeError("Exported file loaded but contains 0 measurements.")
            required_fields = ("a", "b", "m", "n", "r")
            missing_fields = [k for k in required_fields if k not in data.dataMap()]
            if missing_fields:
                raise RuntimeError(f"Missing required fields after load: {missing_fields}")

            sensors = data.sensorPositions()
            print(f"   DEBUG: PyGIMLi loaded {len(sensors)} electrode positions")
            if len(sensors) > 0:
                sx = np.array([s.x() for s in sensors], dtype=float)
                sy = np.array([s.y() for s in sensors], dtype=float)
                sz = np.array([s.z() for s in sensors], dtype=float)
                inferred_loaded_axis = _infer_elevation_axis(sy, sz)
                print(f"   DEBUG: Loaded coordinate ranges: {_coord_stats(sx, sy, sz)}")
                print(f"   DEBUG: Inferred elevation axis (loaded): {inferred_loaded_axis}")

            has_k = 'k' in data.dataMap() and np.array(data['k']).size == data.size()
            if has_k:
                k_vals = np.array(data['k'], dtype=float)
                finite_k = np.isfinite(k_vals)
                has_valid_k = np.any(finite_k & (np.abs(k_vals) > 1.5))
            else:
                k_vals = np.array([])
                has_valid_k = False

            if has_valid_k:
                print(f"   K factors already provided (range: [{np.nanmin(k_vals):.1f}, {np.nanmax(k_vals):.1f}])")
            else:
                print("   Computing geometric factors with PyGIMLi...")
                data['k'] = ert_pg.createGeometricFactors(data, numerical=True)
                k_vals = np.array(data['k'], dtype=float)
                print(f"   Computed K range: [{np.nanmin(k_vals):.1f}, {np.nanmax(k_vals):.1f}]")

            keep_mask = np.ones(data.size(), dtype=bool)
            k_vals = np.array(data['k'], dtype=float)
            if use_default_export:
                k_valid = np.isfinite(k_vals) & (np.abs(k_vals) > 1e-12)
                n_k_filtered = int(np.sum(~k_valid))
                if n_k_filtered > 0:
                    print(f"   Filtered {n_k_filtered} measurements with invalid K")
            else:
                k_threshold = 1000
                k_valid = np.isfinite(k_vals) & (np.abs(k_vals) < k_threshold)
                n_k_filtered = int(np.sum(~k_valid))
                if n_k_filtered > 0:
                    print(f"   Filtered {n_k_filtered} measurements with invalid/large |K| (threshold={k_threshold})")
            keep_mask = keep_mask & k_valid

            if not np.any(keep_mask):
                raise RuntimeError("All measurements removed after K filtering.")

            if app_res_is_apparent and not has_valid_k:
                # Source observations already represent apparent resistivity.
                # Avoid rhoa <- r * k double-application when K is newly computed.
                print("   DEBUG: Preserving source rhoa and back-calculating r from computed K.")
                rhoa_raw = np.array(data['rhoa'], dtype=float) if 'rhoa' in data.dataMap() else np.array(data['r'], dtype=float)
                r_raw = np.array(data['r'], dtype=float)
                k_raw = np.array(data['k'], dtype=float)
                rhoa_fallback = r_raw * k_raw
                rhoa_vals_preserved = np.where(
                    np.isfinite(rhoa_raw) & (rhoa_raw > 0),
                    rhoa_raw,
                    rhoa_fallback
                )
                k_safe = np.where(np.isfinite(k_raw) & (np.abs(k_raw) > 1e-12), k_raw, np.nan)
                r_backcalc = np.divide(
                    rhoa_vals_preserved,
                    k_safe,
                    out=np.full_like(rhoa_vals_preserved, np.nan),
                    where=np.isfinite(k_safe)
                )
                r_backcalc = np.where(np.isfinite(r_backcalc) & (r_backcalc > 0), r_backcalc, r_raw)
                data['r'] = r_backcalc
                data['rhoa'] = rhoa_vals_preserved
            else:
                data['rhoa'] = data['r'] * data['k']

            rhoa_vals = np.array(data['rhoa'], dtype=float)
            if use_default_export:
                rhoa_valid = np.isfinite(rhoa_vals) & (rhoa_vals > 0)
                n_filtered = int(np.sum(keep_mask & ~rhoa_valid))
                if n_filtered > 0:
                    print(f"   Filtered {n_filtered} measurements with invalid apparent resistivity")
            else:
                rhoa_valid = np.isfinite(rhoa_vals) & (rhoa_vals > 0.1) & (rhoa_vals < 1e6)
                stat_mask = keep_mask & rhoa_valid
                valid_rhoa = rhoa_vals[stat_mask]
                if len(valid_rhoa) > 0:
                    rhoa_median = float(np.median(valid_rhoa))
                    rhoa_std = float(np.std(valid_rhoa))
                    lower_bound = max(0.1, rhoa_median - 3 * rhoa_std)
                    upper_bound = min(1e6, rhoa_median + 3 * rhoa_std)
                    rhoa_valid = rhoa_valid & (rhoa_vals >= lower_bound) & (rhoa_vals <= upper_bound)

                n_filtered = int(np.sum(keep_mask & ~rhoa_valid))
                if n_filtered > 0:
                    print(f"   Filtered {n_filtered} measurements with extreme apparent resistivity")
            keep_mask = keep_mask & rhoa_valid

            if not np.any(keep_mask):
                raise RuntimeError("All measurements removed after rhoa filtering.")

            n_kept = int(np.sum(keep_mask))
            print(f"   Final dataset: {n_kept} measurements with computed K")

            temp_path = path.parent / (path.name + '.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(f"{n_elec}\n")
                f.write("# x y z\n")
                for elec in ert.electrodes:
                    f.write(f"{float(elec.x)} {float(elec.y)} {float(elec.z)}\n")

                rows_written = 0
                skipped_rewrite = 0
                row_records = []
                has_err_data = 'err' in data.dataMap() and np.array(data['err']).size == data.size()
                err_data_all = np.array(data['err'], dtype=float) if has_err_data else np.array([])
                for i in range(data.size()):
                    if not keep_mask[i]:
                        skipped_rewrite += 1
                        continue
                    a = int(data['a'][i]) + 1
                    b = int(data['b'][i]) + 1
                    m = int(data['m'][i]) + 1
                    n = int(data['n'][i]) + 1
                    if min(a, b, m, n) < 0 or max(a, b, m, n) > n_elec:
                        skipped_rewrite += 1
                        continue
                    if a == b or m == n:
                        skipped_rewrite += 1
                        continue
                    r_val = float(data['r'][i])
                    rhoa_val = float(data['rhoa'][i])
                    k_val = float(data['k'][i])
                    if not (np.isfinite(r_val) and np.isfinite(rhoa_val) and np.isfinite(k_val)):
                        skipped_rewrite += 1
                        continue
                    if r_val <= 0 or rhoa_val <= 0:
                        skipped_rewrite += 1
                        continue
                    src_err_val = float(err_data_all[i]) if has_err_data else np.nan
                    row_records.append((a, b, m, n, r_val, rhoa_val, k_val, src_err_val))
                    rows_written += 1

                if rows_written == 0:
                    raise RuntimeError("No valid measurements remained for final rewrite.")

                if use_default_export:
                    source_err_final = np.array([r[7] for r in row_records], dtype=float)
                    err_final = np.where(
                        np.isfinite(source_err_final) & (source_err_final > 0),
                        source_err_final,
                        source_default_err,
                    ).astype(float)
                    n_source_valid_final = int(np.sum(np.isfinite(source_err_final) & (source_err_final > 0)))
                    print(
                        f"   DEBUG: Preserved default export errors on final rows: "
                        f"valid={n_source_valid_final}/{rows_written}, "
                        f"err_range=[{np.nanmin(err_final):.3f}, {np.nanmax(err_final):.3f}]"
                    )
                elif use_source_error:
                    source_err_final = np.array([r[7] for r in row_records], dtype=float)
                    err_final = _sanitize_source_error(
                        source_err_final,
                        default_err=source_default_err,
                        min_err=source_min_err,
                        max_err=source_max_err,
                    )
                    n_source_valid_final = int(np.sum(np.isfinite(source_err_final) & (source_err_final > 0)))
                    print(
                        f"   DEBUG: Source-error use on final rows: "
                        f"valid={n_source_valid_final}/{rows_written}, "
                        f"err_range=[{np.nanmin(err_final):.3f}, {np.nanmax(err_final):.3f}]"
                    )
                else:
                    abmn_final = np.array([(r[0], r[1], r[2], r[3]) for r in row_records], dtype=int)
                    resist_final = np.array([r[4] for r in row_records], dtype=float)
                    err_final, paired_final = _estimate_reciprocal_error(
                        abmn=abmn_final,
                        resist=resist_final,
                        default_err=reciprocal_default_err,
                        min_err=reciprocal_min_err,
                        max_err=reciprocal_max_err,
                    )
                    print(
                        f"   DEBUG: Reciprocal-error estimate on final rows: "
                        f"paired={paired_final}/{rows_written}, "
                        f"err_range=[{np.nanmin(err_final):.3f}, {np.nanmax(err_final):.3f}]"
                    )

                f.write(f"{rows_written}\n")
                f.write("# a b m n r rhoa k err\n")
                for idx_row, row in enumerate(row_records):
                    a, b, m, n, r_val, rhoa_val, k_val, _src_err = row
                    err_val = float(err_final[idx_row])
                    f.write(f"{a} {b} {m} {n} {r_val} {rhoa_val} {k_val} {err_val}\n")

            temp_path.replace(path)
            if skipped_rewrite > 0:
                print(f"   DEBUG: Skipped {skipped_rewrite} non-finite rows during final rewrite")
            print(f"   Saved data with computed K to {path}")

        except Exception as e:
            import traceback
            print(f"   Warning: Could not recompute K with PyGIMLi: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            print(f"   File kept with original exported values")

        return str(path)
    elif fmt == "resipy":
        return ert.metadata.get("project_dir", "")
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")


def export_ert_dataset(
    ert: StandardERT,
    outdir: str = "examples/results/ert",
    formats: Iterable[str] = ("standard_json", "observations_csv", "electrodes_csv"),
    export_strategy: str = "legacy",
) -> Dict[str, str]:
    """
    Export standardized ERT data to one or more user-facing formats.

    Parameters
    ----------
    ert : StandardERT
        Standardized ERT dataset to export.
    outdir : str
        Destination directory.
    formats : iterable of str
        Requested formats. Supported values are ``standard_json``,
        ``observations_csv``, ``electrodes_csv``, ``observations_parquet``,
        ``reciprocal_csv``, ``pygimli_bert``, and ``resipy_project``.
    export_strategy : str
        Strategy passed to :func:`export_for_inversion` for ``pygimli_bert``.

    Returns
    -------
    dict
        Mapping from format key to exported file path or project directory.
    """

    if ert is None:
        raise ValueError("ERT dataset is None")

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    requested = {str(fmt).strip().lower() for fmt in formats}
    outputs: Dict[str, str] = {}

    electrode_rows = [asdict(e) for e in (ert.electrodes or [])]
    observation_rows = []
    for index, obs in enumerate(ert.observations or []):
        observation_rows.append(
            {
                "observation_index": index,
                "A": obs.quad.A,
                "B": obs.quad.B,
                "M": obs.quad.M,
                "N": obs.quad.N,
                "value": obs.app_res,
                "dV": obs.dV,
                "I": obs.I,
                "rel_err": obs.rel_err,
                "K": obs.K,
                "fid": obs.fid,
            }
        )

    electrodes_df = pd.DataFrame(electrode_rows)
    observations_df = pd.DataFrame(observation_rows)
    reciprocal_df = calculate_reciprocal_errors(ert)

    if not observations_df.empty and not reciprocal_df.empty:
        reciprocal_cols = [
            "observation_index",
            "reciprocal_group",
            "reciprocal_pair_count",
            "reciprocal_error_percent",
            "reciprocal_mean_value",
            "reciprocal_partner_value",
        ]
        observations_df = observations_df.merge(
            reciprocal_df[reciprocal_cols],
            on="observation_index",
            how="left",
        )

    if "standard_json" in requested:
        path = outdir_path / "ert_standard.json"
        ert.to_json(path)
        outputs["standard_json"] = str(path)

    if "observations_csv" in requested:
        path = outdir_path / "observations.csv"
        observations_df.to_csv(path, index=False)
        outputs["observations_csv"] = str(path)

    if "electrodes_csv" in requested:
        path = outdir_path / "electrodes.csv"
        electrodes_df.to_csv(path, index=False)
        outputs["electrodes_csv"] = str(path)

    if "observations_parquet" in requested:
        path = outdir_path / "observations.parquet"
        observations_df.to_parquet(path, index=False)
        outputs["observations_parquet"] = str(path)

    if "reciprocal_csv" in requested:
        path = outdir_path / "reciprocal_qc.csv"
        reciprocal_df.to_csv(path, index=False)
        outputs["reciprocal_csv"] = str(path)

    if "pygimli_bert" in requested:
        outputs["pygimli_bert"] = export_for_inversion(
            ert,
            outdir=str(outdir_path),
            fmt="pgimli",
            export_strategy=export_strategy,
        )

    if "resipy_project" in requested:
        outputs["resipy_project"] = export_for_inversion(
            ert,
            outdir=str(outdir_path),
            fmt="resipy",
        )

    unsupported = requested.difference(
        {
            "standard_json",
            "observations_csv",
            "electrodes_csv",
            "observations_parquet",
            "reciprocal_csv",
            "pygimli_bert",
            "resipy_project",
        }
    )
    if unsupported:
        raise ValueError(f"Unsupported ERT export format(s): {sorted(unsupported)}")

    return outputs

