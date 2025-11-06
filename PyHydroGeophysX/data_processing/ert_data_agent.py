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

# Optional dependency: RESIPY
try:
    from resipy import Project
    _HAS_RESIPY = True
except Exception:
    _HAS_RESIPY = False


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


# ---------------------------
# Loader
# ---------------------------
def load_ert_resipy(
    project_dir: str,
    data_file: str,
    instrument: Instrument,
    spacing: Optional[float] = None,
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
    assert _HAS_RESIPY, "RESIPY not installed. Please `pip install resipy`."
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
    prj.createSurvey(fname=str(data_file_path), ftype=ftype)
    
    # After createSurvey, data is stored in prj.surveys[0] (the first/only survey)
    if not prj.surveys:
        raise RuntimeError("No survey was created. Check that the data file format matches the specified instrument.")
    
    survey = prj.surveys[0]  # Get the first survey object

    # If no electrode coordinates, generate a simple line for quick testing
    if spacing is not None and (survey.elec is None or len(survey.elec) == 0):
        n_elec = int(np.max(survey.df[['a','b','m','n']].values)) + 1
        elec = np.zeros((n_elec, 3))
        elec[:, 0] = np.arange(n_elec) * spacing
        survey.setElec(elec)

    # Minimal QC - access data from survey.df (the dataframe in the Survey object)
    df = survey.df.copy().dropna(subset=['a','b','m','n'])
    if 'i' in df.columns:
        df = df[df['i'].abs() > 0]
    if 'u' in df.columns:
        df = df[df['u'].abs() > 0]
    df = df.drop_duplicates(subset=['a','b','m','n'])

    # Apparent resistivity - RESIPY may use different column names
    # Common names: 'app', 'rhoa', 'Rho', 'resist' (for measured resistance)
    rhoa_col = None
    for col_name in ['app', 'rhoa', 'Rho']:
        if col_name in df.columns:
            rhoa_col = col_name
            break
    
    if rhoa_col is None:
        # Try to compute from resistance and geometric factor
        try:
            survey.computeK()  # Compute geometric factors first
            if hasattr(survey, 'computeRhoa'):
                survey.computeRhoa()
                df = survey.df.loc[df.index].copy()
                # Check again for apparent resistivity columns
                for col_name in ['app', 'rhoa', 'Rho']:
                    if col_name in df.columns:
                        rhoa_col = col_name
                        break
        except Exception:
            pass
    
    # Use the found column or set to NaN
    if rhoa_col:
        df['rhoa'] = df[rhoa_col]
    else:
        df['rhoa'] = np.nan

    # Simple relative error model
    rel_err = np.full(len(df), 0.03)
    # Check for error columns (different names for different instruments)
    for err_col in ['resError', 'magErr', 'err', 'error', 'dev']:
        if err_col in df.columns:
            err_vals = df[err_col].values
            # Use the error if it's reasonable (between 0.5% and 50%)
            valid_err = np.where(np.isfinite(err_vals) & (err_vals > 0.005) & (err_vals < 0.5), 
                                 err_vals, 0.03)
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
    
    # Detect current and voltage column names (vary by instrument)
    i_col = next((c for c in ['i', 'I', 'current'] if c in df.columns), None)
    v_col = next((c for c in ['u', 'U', 'v', 'V', 'voltage', 'dV'] if c in df.columns), None)
    
    for idx, r in df.iterrows():
        observations.append(Observation(
            quad=Quadruplet(int(r['a']), int(r['b']), int(r['m']), int(r['n'])),
            app_res=float(r['rhoa']) if 'rhoa' in r and np.isfinite(r['rhoa']) else None,
            dV=float(r[v_col]) if v_col and v_col in r and np.isfinite(r[v_col]) else None,
            I=float(r[i_col]) if i_col and i_col in r and np.isfinite(r[i_col]) else None,
            rel_err=float(rel_err[idx_to_pos[idx]]),
            fid=str(r['id']) if 'id' in r else str(idx)
        ))

    crs_out = ("EPSG:%d" % epsg) if (crs != "local" and epsg) else crs
    meta = {
        "loader": "RESIPY",
        "ftype": ftype,
        "project_dir": str(Path(chosen_dir).resolve()),
        "data_file": str(data_file_path.resolve()),
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
    outdir_path = Path(outdir)
    if outdir.startswith('/') and not outdir_path.is_absolute():
        # Remove leading / and treat as relative to cwd
        outdir_path = Path.cwd() / outdir.lstrip('/')
    
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    if fmt == "pgimli":
        path = outdir_path / "bert_data.dat"
        
        with open(path, 'w') as f:
            # Write number of electrodes
            f.write(f"{len(ert.electrodes)}\n")
            
            # Write electrode coordinates header and data
            f.write("# x y z\n")
            for elec in ert.electrodes:
                f.write(f"{elec.x:.2f}\t{elec.y:.2f}\t{elec.z:.2f}\n")
            
            # Write number of measurements
            f.write(f"{len(ert.observations)}\n")
            
            # Write measurement data header
            f.write("# a b m n err i ip iperr k r rhoa u valid\n")
            
            # Write measurement data
            for obs in ert.observations:
                rhoa = obs.app_res if obs.app_res is not None else 0.0
                err = obs.rel_err if obs.rel_err is not None else 0.03
                i_val = obs.I if obs.I is not None else 0.0
                u_val = obs.dV if obs.dV is not None else 0.0
                
                # Calculate geometric factor K from apparent resistivity and resistance
                # K = rhoa / R, where R = dV / I
                if i_val != 0 and u_val != 0:
                    R = u_val / i_val if i_val != 0 else 0.0
                    K = rhoa / R if R != 0 else 0.0
                else:
                    K = 0.0
                    R = 0.0
                
                # Format: a b m n err i ip iperr k r rhoa u valid
                # Note: IP parameters (ip, iperr) are set to 0 for DC-only data
                f.write(f"{obs.quad.A}\t{obs.quad.B}\t{obs.quad.M}\t{obs.quad.N}\t")
                f.write(f"{err:.14e}\t{i_val:.14e}\t0.00000000000000e+00\t0.00000000000000e+00\t")
                f.write(f"{K:.14e}\t{R:.14e}\t{rhoa:.14e}\t{u_val:.14e}\t1\n")
        
        return str(path)
    elif fmt == "resipy":
        return ert.metadata.get("project_dir", "")
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
