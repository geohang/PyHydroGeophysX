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
_RESIPY_ERROR = None
try:
    from resipy import Project
    _HAS_RESIPY = True
except Exception as e:
    _HAS_RESIPY = False
    _RESIPY_ERROR = str(e)


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
    if not _HAS_RESIPY:
        error_msg = f"RESIPY import failed: {_RESIPY_ERROR}" if _RESIPY_ERROR else "RESIPY not installed. Please `pip install resipy`."
        raise ImportError(error_msg)
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
