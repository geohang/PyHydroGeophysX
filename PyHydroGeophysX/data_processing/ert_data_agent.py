# PyHydroGeophysX/core/data_processing/ert_data_agent.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, NamedTuple
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    prj = Project(project_dir)
    Path(project_dir).mkdir(parents=True, exist_ok=True)
    prj.createFolder(project_dir)

    # Use explicit ftype for robust parsing
    prj.importData(data_file, ftype=ftype)

    # If no electrode coordinates, generate a simple line for quick testing
    if spacing is not None and (prj.elec is None):
        n_elec = int(np.max(prj.data[['a','b','m','n']].values)) + 1
        elec = np.zeros((n_elec, 3))
        elec[:, 0] = np.arange(n_elec) * spacing
        prj.setElec(elec)

    # Minimal QC
    df = prj.data.copy().dropna(subset=['a','b','m','n'])
    if 'i' in df.columns:
        df = df[df['i'].abs() > 0]
    if 'u' in df.columns:
        df = df[df['u'].abs() > 0]
    df = df.drop_duplicates(subset=['a','b','m','n'])

    # Apparent resistivity (compute if missing)
    if 'rhoa' not in df.columns:
        try:
            prj.computeRhoa()
            df['rhoa'] = prj.data.loc[df.index, 'rhoa']
        except Exception:
            df['rhoa'] = np.nan

    # Simple relative error model
    rel_err = np.full(len(df), 0.03)
    if 'err' in df.columns:
        rel_err = np.maximum(rel_err, df['err'].clip(0.005, 0.2).values)

    # Electrodes
    elec_arr = np.array(prj.elec) if prj.elec is not None else \
               np.zeros((int(df[['a','b','m','n']].values.max())+1, 3))
    electrodes = [Electrode(i+1, float(elec_arr[i,0]), float(elec_arr[i,1]), float(elec_arr[i,2]))
                  for i in range(elec_arr.shape[0])]

    # Observations
    idx_to_pos = {idx: k for k, idx in enumerate(df.index)}
    observations: List[Observation] = []
    for idx, r in df.iterrows():
        observations.append(Observation(
            quad=Quadruplet(int(r['a']), int(r['b']), int(r['m']), int(r['n'])),
            app_res=float(r['rhoa']) if np.isfinite(r['rhoa']) else None,
            dV=float(r['u']) if 'u' in r and np.isfinite(r['u']) else None,
            I=float(r['i']) if 'i' in r and np.isfinite(r['i']) else None,
            rel_err=float(rel_err[idx_to_pos[idx]]),
            fid=str(r['id']) if 'id' in r else None
        ))

    crs_out = ("EPSG:%d" % epsg) if (crs != "local" and epsg) else crs
    meta = {
        "loader": "RESIPY",
        "ftype": ftype,
        "project_dir": str(Path(project_dir).resolve()),
        "data_file": str(Path(data_file).resolve()),
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
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Electrodes plot
    ex = [e.x for e in ert.electrodes]
    ez = [e.z for e in ert.electrodes]
    plt.figure(figsize=(6, 2))
    plt.plot(ex, ez, 'k.-')
    plt.xlabel('x (m)'); plt.ylabel('z (m)')
    p1 = str(Path(outdir) / "electrodes.png")
    plt.tight_layout(); plt.savefig(p1, dpi=200); plt.close()

    # Apparent resistivity histogram
    vals = [o.app_res for o in ert.observations if o.app_res is not None and np.isfinite(o.app_res)]
    plt.figure(figsize=(4, 3))
    plt.hist(np.log10(vals), bins=40, color="#4C72B0")
    plt.xlabel('log10 apparent resistivity'); plt.ylabel('count')
    p2 = str(Path(outdir) / "rhoa_hist.png")
    plt.tight_layout(); plt.savefig(p2, dpi=200); plt.close()

    # Flat tables
    obs_rows = [{
        "A": o.quad.A, "B": o.quad.B, "M": o.quad.M, "N": o.quad.N,
        "app_res": o.app_res, "dV": o.dV, "I": o.I,
        "rel_err": o.rel_err, "fid": o.fid
    } for o in ert.observations]
    pd.DataFrame(obs_rows).to_parquet(Path(outdir) / "observations.parquet", index=False)
    pd.DataFrame([asdict(e) for e in ert.electrodes]).to_csv(Path(outdir) / "electrodes.csv", index=False)
    ert.to_json(Path(outdir) / "ert_standard.json")

    return {
        "electrodes_png": str(Path(outdir)/"electrodes.png"),
        "rhoa_hist_png": str(Path(outdir)/"rhoa_hist.png"),
        "observations_parquet": str(Path(outdir)/"observations.parquet"),
        "electrodes_csv": str(Path(outdir)/"electrodes.csv"),
        "standard_json": str(Path(outdir)/"ert_standard.json"),
    }

def export_for_inversion(ert: StandardERT, outdir: str = "examples/results/ert", fmt: str = "pgimli") -> str:
    """
    Export to inversion-ready formats:
    - fmt='pgimli': ASCII columns [A B M N rhoa rel_err] for BERT/pyGIMLi.
    - fmt='resipy': return the RESIPY project directory for running prj.start().
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    if fmt == "pgimli":
        rows = [[o.quad.A, o.quad.B, o.quad.M, o.quad.N,
                 (o.app_res if o.app_res is not None else np.nan),
                 (o.rel_err if o.rel_err is not None else 0.03)]
                for o in ert.observations]
        path = Path(outdir) / "bert_data.dat"
        np.savetxt(path, np.array(rows, dtype=float), fmt="%.6f")
        return str(path)
    elif fmt == "resipy":
        return ert.metadata.get("project_dir", "")
    else:
        raise ValueError(f"Unsupported fmt: {fmt}")
