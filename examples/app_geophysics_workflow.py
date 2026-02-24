"""
PyHydroGeophysX Streamlit Web Application
=========================================

Natural-language interface for geophysical workflows.
Usage: streamlit run app_geophysics_workflow.py
"""

import os
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

# Add parent directory to path so local package can be imported when run from examples/
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

IMPORT_ERROR = ""
try:
    from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent
    AGENTS_AVAILABLE = True
except ImportError as e:
    AGENTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    BaseAgent = None
    ContextInputAgent = None

# Check for pygimli availability
try:
    import pygimli
    PYGIMLI_AVAILABLE = True
except ImportError:
    PYGIMLI_AVAILABLE = False

# Data access abstraction — import directly from the module file to avoid
# triggering the heavy PyHydroGeophysX.__init__ (SimPEG, pygimli, etc.)
DATA_ACCESS_AVAILABLE = False
_DATA_ACCESS_ERROR = ""
try:
    import importlib.util as _ilu
    _acc_path = str(PARENT_DIR / "PyHydroGeophysX" / "data_access" / "accessors.py")
    _spec = _ilu.spec_from_file_location("_phgx_accessors", _acc_path)
    if _spec and _spec.loader:
        _acc_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_acc_mod)
        LocalHydroAccessor = _acc_mod.LocalHydroAccessor
        HttpHydroAccessor = _acc_mod.HttpHydroAccessor
        load_manifest = _acc_mod.load_manifest
        get_manifest_entry = _acc_mod.get_manifest_entry
        REQUIRED_FILES = _acc_mod.REQUIRED_FILES
        DATA_ACCESS_AVAILABLE = True
    else:
        _DATA_ACCESS_ERROR = f"spec_from_file_location returned None for {_acc_path}"
except Exception as _exc:  # noqa: BLE001
    _DATA_ACCESS_ERROR = f"File-based import: {type(_exc).__name__}: {_exc}"

# Fallback: try a normal package import if file-based loading failed
if not DATA_ACCESS_AVAILABLE:
    try:
        from PyHydroGeophysX.data_access.accessors import (
            LocalHydroAccessor, HttpHydroAccessor, load_manifest, get_manifest_entry,
            REQUIRED_FILES,
        )
        DATA_ACCESS_AVAILABLE = True
        _DATA_ACCESS_ERROR = ""
    except Exception as _exc2:  # noqa: BLE001
        _DATA_ACCESS_ERROR += f" | Package import: {type(_exc2).__name__}: {_exc2}"

# Ensure REQUIRED_FILES is always defined even if data_access import failed
if "REQUIRED_FILES" not in dir():
    REQUIRED_FILES = ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"]


def _is_streamlit_cloud() -> bool:
    """Heuristic: detect if running on Streamlit Community Cloud."""
    cloud_indicators = [
        os.environ.get("STREAMLIT_SHARING_MODE"),
        os.environ.get("IS_STREAMLIT_CLOUD"),
        os.environ.get("HOSTNAME", "").startswith("streamlit"),
    ]
    # Streamlit Cloud typically sets HOME to /home/appuser
    home = os.environ.get("HOME", "")
    if "/home/appuser" in home:
        return True
    return any(cloud_indicators)

st.set_page_config(
    page_title="PyHydroGeophysX - Geophysical Workflows",
    page_icon="PHGX",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
:root {
    --phgx-blue: #0f4c75;
    --phgx-green: #2d9c5b;
    --phgx-gray: #f5f7fb;
    --phgx-dark: #1b262c;
    --phgx-accent: #3d6cb9;
}

section.main > div {
    padding-top: 1rem;
}

.phgx-header {
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--phgx-dark);
    letter-spacing: 0.04em;
}

.phgx-subtitle-main {
    background: linear-gradient(90deg, var(--phgx-blue) 0%, var(--phgx-accent) 50%, var(--phgx-green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-top: 0.1rem;
    margin-bottom: 0.3rem;
}

.phgx-author-line {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
    flex-wrap: wrap;
}

.phgx-version-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.6rem;
    border-radius: 0.4rem;
    background: linear-gradient(135deg, #e8f4f8 0%, #f0f7ff 100%);
    border: 1px solid #c8dce8;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--phgx-blue);
}

.phgx-author-text {
    color: #5a6a7a;
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

.phgx-author-text a {
    color: var(--phgx-accent);
    text-decoration: none;
    border-bottom: 1px dotted var(--phgx-accent);
}

.phgx-author-text a:hover {
    color: var(--phgx-blue);
    border-bottom-style: solid;
}

.phgx-subtitle {
    color: #2f3b4a;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.01em;
    margin-top: -0.05rem;
    margin-bottom: 0.75rem;
}

.phgx-card {
    padding: 1.1rem 1.2rem;
    border-radius: 0.6rem;
    background: var(--phgx-gray);
    border: 1px solid #e1e5ec;
}

.phgx-pill {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    background: #e2f0ff;
    color: #174ea6;
    font-weight: 600;
    font-size: 0.85rem;
    margin-right: 0.35rem;
    margin-bottom: 0.3rem;
}

.phgx-mono {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.9rem;
}

.phgx-support-card {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border: 1px solid #cbd5e1;
    border-radius: 0.8rem;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
    text-align: center;
}

.phgx-support-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #334155;
    margin-bottom: 0.5rem;
}

.phgx-support-text {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 0.8rem;
    line-height: 1.5;
}

.phgx-venmo-btn {
    display: inline-block;
    background: linear-gradient(135deg, #008cff 0%, #0066cc 100%);
    color: white !important;
    padding: 0.5rem 1.2rem;
    border-radius: 2rem;
    font-weight: 600;
    font-size: 0.9rem;
    text-decoration: none;
    margin: 0.3rem;
    transition: all 0.2s ease;
}

.phgx-venmo-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 140, 255, 0.3);
}

.phgx-email-link {
    color: #0f4c75;
    text-decoration: none;
    font-weight: 500;
    border-bottom: 1px dotted #0f4c75;
}

.phgx-email-link:hover {
    color: #3d6cb9;
    border-bottom-style: solid;
}

.phgx-free-badge {
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    padding: 0.2rem 0.6rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

/* Make tabs larger and more prominent */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f0f4f8;
    padding: 0.5rem;
    border-radius: 0.6rem;
}

.stTabs [data-baseweb="tab"] {
    height: 60px;
    padding: 0 24px;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--phgx-dark);
    background-color: white;
    border-radius: 0.5rem;
    border: 1px solid #e1e5ec;
    white-space: pre-wrap;
}

.stTabs [data-baseweb="tab"]:hover {
    background-color: #e8f4f8;
    border-color: var(--phgx-accent);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--phgx-blue) 0%, var(--phgx-accent) 100%) !important;
    color: white !important;
    border-color: var(--phgx-blue) !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Keep metric cards readable with long values (e.g., grid summary strings). */
div[data-testid="stMetricLabel"] {
    font-size: 0.86rem !important;
    line-height: 1.2 !important;
}

div[data-testid="stMetricValue"] {
    font-size: 1.02rem !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

div[data-testid="stMetricValue"] > div {
    font-size: 1.02rem !important;
    line-height: 1.25 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* ===== Hydro workflow stepper & status strip ===== */

.hydro-status-strip {
    display: flex;
    gap: 0;
    margin-bottom: 1.2rem;
    border-radius: 0.7rem;
    overflow: hidden;
    border: 1px solid #e1e5ec;
    background: #f8fafc;
}

.hydro-step-indicator {
    flex: 1;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #94a3b8;
    border-right: 1px solid #e1e5ec;
    cursor: pointer;
    transition: all 0.2s ease;
    background: #f8fafc;
}

.hydro-step-indicator:last-child { border-right: none; }

.hydro-step-indicator:hover {
    background: #f0f4f8;
    color: #475569;
}

.hydro-step-indicator.active {
    background: linear-gradient(135deg, var(--phgx-blue) 0%, var(--phgx-accent) 100%);
    color: white;
}

.hydro-step-indicator.done {
    background: #ecfdf5;
    color: #059669;
}

.hydro-step-indicator.error {
    background: #fef2f2;
    color: #dc2626;
}

.hydro-step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: 700;
    background: #e2e8f0;
    color: #64748b;
    flex-shrink: 0;
}

.hydro-step-indicator.active .hydro-step-num {
    background: rgba(255,255,255,0.25);
    color: white;
}

.hydro-step-indicator.done .hydro-step-num {
    background: #059669;
    color: white;
}

/* ===== Data source cards ===== */

.hydro-source-cards {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}

.hydro-source-card {
    flex: 1;
    padding: 1.2rem;
    border-radius: 0.7rem;
    border: 2px solid #e2e8f0;
    background: white;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}

.hydro-source-card:hover {
    border-color: var(--phgx-accent);
    box-shadow: 0 2px 8px rgba(61, 108, 185, 0.1);
    transform: translateY(-1px);
}

.hydro-source-card.selected {
    border-color: var(--phgx-blue);
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4f8 100%);
    box-shadow: 0 2px 12px rgba(15, 76, 117, 0.15);
}

.hydro-source-icon {
    font-size: 2rem;
    margin-bottom: 0.4rem;
}

.hydro-source-title {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--phgx-dark);
    margin-bottom: 0.25rem;
}

.hydro-source-desc {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.4;
}

/* ===== Method pills / chips ===== */

.hydro-method-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0;
}

.hydro-method-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.45rem 1rem;
    border-radius: 2rem;
    border: 2px solid #e2e8f0;
    background: white;
    font-weight: 600;
    font-size: 0.88rem;
    color: #475569;
    cursor: pointer;
    transition: all 0.15s ease;
}

.hydro-method-chip:hover {
    border-color: var(--phgx-accent);
    background: #f0f4ff;
}

.hydro-method-chip.selected {
    border-color: var(--phgx-blue);
    background: linear-gradient(135deg, var(--phgx-blue) 0%, var(--phgx-accent) 100%);
    color: white;
}

.hydro-method-chip .chip-icon {
    font-size: 1rem;
}

/* ===== Results gallery ===== */

.hydro-result-card {
    border-radius: 0.7rem;
    border: 1px solid #e2e8f0;
    overflow: hidden;
    background: white;
    margin-bottom: 1rem;
}

.hydro-result-card-header {
    padding: 0.6rem 1rem;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--phgx-dark);
}

.hydro-result-card-body {
    padding: 0.8rem;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

EXAMPLE_REQUESTS: Dict[str, str] = {
    "ParFlow": """Load ParFlow outputs. I uploaded a saturation .pfb file.
Also uploaded porosity .pfb file.
Convert saturation to water content, then to resistivity with rho_sat=541 and n=1.24.""",
    "MODFLOW": """Load MODFLOW outputs. I uploaded id.txt (idomain file).
Model name: TLnewtest2sfb2. Timestep: 1. Number of layers: 3.
Convert water content to resistivity using rho_sat=541 and n=1.24.""",
    "Standard ERT": """Run a standard ERT inversion using the DAS-1 instrument.
Data file: 20171105_1418.Data
Electrode file: electrodes.dat
Petrophysics: rho_sat=541, porosity=0.37, n=1.24
Regularization lambda: 15""",
    "Time-Lapse ERT": """Run a time-lapse ERT inversion on four E4D files:
- 2022-03-26_0030.ohm (baseline)
- 2022-04-26_0030.ohm
- 2022-05-26_0030.ohm
- 2022-06-26_0030.ohm
Temporal regularization: 10
Include climate data for Mt. Snodgrass at 38.92584N, -106.97998W""",
    "Data Fusion": """Perform structure-constrained inversion using seismic + ERT.
Seismic: srtfieldline2.dat with velocity threshold 1000 m/s
ERT: fielddataline2.dat
Petrophysics:
- Regolith: rho_sat 50-250, n 1.3-2.2, porosity 0.25-0.50
- Fractured bedrock: rho_sat 165-350, n 2.0-2.2, porosity 0.2-0.3
Monte Carlo realizations: 100""",
    "Seismic Refraction": """Run a seismic refraction tomography (SRT) inversion.
Data file: synthetic_seismic_data_long.dat
Regularization lambda: 50
Vertical weight: 0.2
Velocity constraints: 500-5000 m/s
Parametric depth: 60 m
Extract velocity interfaces at: 1200 m/s (regolith-bedrock), 5000 m/s (fractured-fresh)"""
}

DATA_LINKS: Dict[str, str] = {
    "Example data folder (all)": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data",
}

# Example-specific data links organized by workflow type
EXAMPLE_DATA_LINKS: Dict[str, Dict[str, str]] = {
    "ParFlow Example": {
        "description": "Load ParFlow saturation outputs and convert to resistivity",
        "notebook": "https://github.com/geohang/PyHydroGeophysX/blob/main/examples/Ex_model_output.ipynb",
        "parflow_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/parflow/test2",
        "files": ["test2.out.satur.00005.pfb", "test2.out.porosity.pfb", "test2.out.mask.pfb"],
    },
    "MODFLOW Example": {
        "description": "Load MODFLOW water content outputs and convert to resistivity",
        "notebook": "https://github.com/geohang/PyHydroGeophysX/blob/main/examples/Ex_model_output.ipynb",
        "modflow_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/modflow",
        "files": ["id.txt", "WaterContent"],
    },
    "ERT Example (Ex1)": {
        "description": "Standard ERT inversion with DAS-1 instrument data from Snowy Range, Wyoming",
        "notebook": "https://github.com/geohang/PyHydroGeophysX/blob/main/examples/Ex_Unified_Workflow_ex1.ipynb",
        "ert_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/ERT/DAS",
        "files": ["20171105_1418.Data", "electrodes.dat"],
    },
    "Time-Lapse Example (Ex2)": {
        "description": "Time-lapse ERT monitoring with climate integration from Mt. Snodgrass, Colorado",
        "notebook": "https://github.com/geohang/PyHydroGeophysX/blob/main/examples/Ex_Unified_Workflow_ex2.ipynb",
        "ert_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/ERT/E4D",
        "files": ["2022-03-26_0030.ohm", "2022-04-26_0030.ohm", "2022-05-26_0030.ohm", "2022-06-26_0030.ohm"],
    },
    "Data Fusion Example (Ex3)": {
        "description": "Multi-method integration: Seismic + ERT with structure constraints",
        "notebook": "https://github.com/geohang/PyHydroGeophysX/blob/main/examples/Ex_Unified_Workflow_ex3.ipynb",
        "seismic_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/Seismic",
        "ert_data": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data/ERT/Bert",
        "files": ["srtfieldline2.dat (seismic)", "fielddataline2.dat (ERT)"],
    },
}

AUTHOR_LINK = "https://sites.google.com/view/hangchen"
AQUAH_PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S3050740526000024"
AQUAH_PAPER_TITLE = "A Generalizable Automated Geophysical Agent Workflow for Accessible Subsurface Hydrology Analysis"

STANDARD_ERT_TUTORIAL_IMAGES = [
    ("Step 1", "step1.png"),
    ("Step 2", "Step2.png"),
    ("Final result 1", "Final_result_1.png"),
    ("Final result 2", "Final_result_2.png"),
    ("Resistivity model", "resistivity_model (5).png"),
    ("Water content", "water_content.png"),
]

HYDRO_RESPONSE_METHODS = ["Profile", "ERT", "SRT", "TDEM", "FDEM", "Gravity"]
QUICK_RUN_MODES = ["Auto (LLM)", "ERT Only", "Time-Lapse ERT", "Seismic SRT"]
HYDRO_DEFAULTS_VERSION = 20260221
HYDRO_NOTEBOOK_DEFAULTS: Dict[str, Any] = {
    # Keep these aligned with examples/Ex_hydro_to_multigeophys.ipynb
    "hydro_data_dir": "data",
    "hydro_output_dir": "results/hydro_to_multigeophys",
    "hydro_methods": [],
    "hydro_run_style": "Batch methods",
    "hydro_single_method": "ERT",
    "hydro_snapshot_index": 5,
    "hydro_point1_x": 115.0,
    "hydro_point1_y": 70.0,
    "hydro_point2_x": 95.0,
    "hydro_point2_y": 180.0,
    "hydro_num_points": 220,
    "hydro_station_count": 24,
    "hydro_ert_scheme": "wa",
    "hydro_ert_electrode_spacing": 1.0,
    "hydro_ert_electrode_start": 15.0,
    "hydro_ert_num_electrodes": 72,
    "hydro_srt_sensor_spacing": 1.0,
    "hydro_srt_sensor_start": 15.0,
    "hydro_srt_num_sensors": 72,
    "hydro_srt_shot_distance": 5,
    "hydro_rho_sat_top": 100.0,
    "hydro_rho_sat_mid": 500.0,
    "hydro_rho_sat_bot": 2400.0,
    "hydro_archie_n_top": 2.2,
    "hydro_archie_n_mid": 1.8,
    "hydro_archie_n_bot": 2.5,
    "hydro_sigma_s_top": 1.0 / 500.0,
    "hydro_sigma_s_mid": 0.0,
    "hydro_sigma_s_bot": 0.0,
    "hydro_top_bulk_modulus": 30.0,
    "hydro_top_shear_modulus": 20.0,
    "hydro_top_mineral_density": 2650.0,
    "hydro_top_depth": 1.0,
    "hydro_mid_bulk_modulus": 50.0,
    "hydro_mid_shear_modulus": 35.0,
    "hydro_mid_mineral_density": 2670.0,
    "hydro_mid_aspect_ratio": 0.05,
    "hydro_bot_bulk_modulus": 55.0,
    "hydro_bot_shear_modulus": 50.0,
    "hydro_bot_mineral_density": 2680.0,
    "hydro_bot_aspect_ratio": 0.03,
    "hydro_seed": 7,
    "hydro_srt_noise_level": 0.01,
    "hydro_srt_noise_abs": 1e-5,
    "hydro_ert_noise_level": 0.03,
    "hydro_ert_abs_error": 0.0,
    "hydro_ert_rel_error": 0.03,
    "hydro_tdem_noise_level": 0.03,
    "hydro_fdem_noise_level": 0.03,
    "hydro_gravity_noise_level": 0.02,
}

README_REFERENCE_ENTRIES: List[Dict[str, str]] = [
    {
        "label": "PyHydroGeophysX platform paper (SSRN, 2026)",
        "url": "https://ssrn.com/abstract=6238293",
        "bibtex": """@misc{chen2026pyhydrogeophysx,
  author = {Chen, Hang and Niu, Qifei and Wu, Yuxin},
  title = {PyHydroGeophysX: An Extensible Open-Source Platform for Integrating Hydrological Models with Geophysical Measurements},
  year = {2026},
  howpublished = {SSRN},
  doi = {10.2139/ssrn.6238293},
  url = {https://ssrn.com/abstract=6238293}
}""",
    },
    {
        "label": "AQUAH workflow paper (Big Data and Earth System, 2026)",
        "url": AQUAH_PAPER_URL,
        "bibtex": """@article{chen2026agentworkflow,
  author = {Chen, Hang},
  title = {A Generalizable Automated Geophysical Agent Workflow for Accessible Subsurface Hydrology Analysis},
  journal = {Big Data and Earth System},
  pages = {100042},
  year = {2026}
}""",
    },
    {
        "label": "RESIPY reference",
        "url": "https://doi.org/10.1016/j.cageo.2020.104423",
        "bibtex": """@article{blanchy2020resipy,
  title={ResIPy, an intuitive open source software for complex geoelectrical inversion/modeling},
  author={Blanchy, Guillaume and Saneiyan, Sina and Boyd, Jimmy and McLachlan, Paul and Binley, Andrew},
  journal={Computers \\& Geosciences},
  volume={137},
  pages={104423},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2020.104423}
}""",
    },
    {
        "label": "pyGIMLi reference",
        "url": "https://doi.org/10.1016/j.cageo.2017.07.011",
        "bibtex": """@article{rucker2017pygimli,
  title={pyGIMLi: An open-source library for modelling and inversion in geophysics},
  author={Rucker, Carsten and Gunther, Thomas and Wagner, Florian M},
  journal={Computers \\& Geosciences},
  volume={109},
  pages={106--123},
  year={2017},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2017.07.011}
}""",
    },
    {
        "label": "ParFlow reference",
        "url": "https://doi.org/10.5194/gmd-8-923-2015",
        "bibtex": """@article{maxwell2015parflow,
  title={A high-resolution simulation of groundwater and surface water over most of the continental US with the integrated hydrologic model ParFlow v3},
  author={Maxwell, Reed M and Condon, Laura E and Kollet, Stefan J},
  journal={Geoscientific Model Development},
  volume={8},
  number={3},
  pages={923--937},
  year={2015},
  publisher={Copernicus GmbH},
  doi={10.5194/gmd-8-923-2015}
}""",
    },
    {
        "label": "MODFLOW/FloPy reference",
        "url": "https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12413",
        "bibtex": """@article{bakker2016flopy,
  author={Bakker, Mark and Post, Vincent and Langevin, Christian D and Hughes, Joseph D and White, Jeremy T and Starn, Jeffrey J and Fienen, Michael N},
  title={Scripting MODFLOW model development using Python and FloPy},
  journal={Groundwater},
  volume={54},
  number={5},
  pages={733--739},
  year={2016},
  doi={10.1111/gwat.12413},
  url={https://ngwa.onlinelibrary.wiley.com/doi/full/10.1111/gwat.12413}
}""",
    },
    {
        "label": "SimPEG reference",
        "url": "https://doi.org/10.1016/j.cageo.2015.09.015",
        "bibtex": """@article{cockett2015simpeg,
  title={SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications},
  author={Cockett, Rowan and Kang, Seogi and Heagy, Lindsey J and Pidlisecky, Adam and Oldenburg, Douglas W},
  journal={Computers \\& Geosciences},
  volume={85},
  pages={142--154},
  year={2015},
  publisher={Elsevier},
  doi={10.1016/j.cageo.2015.09.015}
}""",
    },
]


def _clone_default_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _apply_hydro_notebook_defaults(force: bool = False) -> None:
    for key, value in HYDRO_NOTEBOOK_DEFAULTS.items():
        if force or key not in st.session_state:
            st.session_state[key] = _clone_default_value(value)


def init_session_state() -> None:
    defaults = {
        "context_agent": None,
        "workflow_result": None,
        "api_key": "",
        "llm_model": "",
        "llm_provider": "openai",
        "output_dir": "results/streamlit_workflow",
        "user_request": "",
        "upload_dir": None,
        "workflow_config": None,
        "hydro_data_dir": "data",
        "hydro_output_dir": "results/hydro_to_multigeophys",
        "hydro_methods": [],
        "hydro_methods_persisted": [],
        "hydro_run_style": "Batch methods",
        "hydro_single_method": "ERT",
        "hydro_snapshot_index": 5,
        "hydro_point1_x": 115.0,
        "hydro_point1_y": 70.0,
        "hydro_point2_x": 95.0,
        "hydro_point2_y": 180.0,
        "hydro_num_points": 220,
        "hydro_station_count": 24,
        "hydro_ert_scheme": "wa",
        "hydro_ert_electrode_spacing": 1.0,
        "hydro_ert_electrode_start": 15.0,
        "hydro_ert_num_electrodes": 72,
        "hydro_srt_sensor_spacing": 1.0,
        "hydro_srt_sensor_start": 15.0,
        "hydro_srt_num_sensors": 72,
        "hydro_srt_shot_distance": 5,
        "hydro_rho_sat_top": 100.0,
        "hydro_rho_sat_mid": 500.0,
        "hydro_rho_sat_bot": 2400.0,
        "hydro_archie_n_top": 2.2,
        "hydro_archie_n_mid": 1.8,
        "hydro_archie_n_bot": 2.5,
        "hydro_sigma_s_top": 1.0 / 500.0,
        "hydro_sigma_s_mid": 0.0,
        "hydro_sigma_s_bot": 0.0,
        "hydro_top_bulk_modulus": 30.0,
        "hydro_top_shear_modulus": 20.0,
        "hydro_top_mineral_density": 2650.0,
        "hydro_top_depth": 1.0,
        "hydro_mid_bulk_modulus": 50.0,
        "hydro_mid_shear_modulus": 35.0,
        "hydro_mid_mineral_density": 2670.0,
        "hydro_mid_aspect_ratio": 0.05,
        "hydro_bot_bulk_modulus": 55.0,
        "hydro_bot_shear_modulus": 50.0,
        "hydro_bot_mineral_density": 2680.0,
        "hydro_bot_aspect_ratio": 0.03,
        "hydro_seed": 7,
        "hydro_srt_noise_level": 0.01,
        "hydro_srt_noise_abs": 1e-5,
        "hydro_ert_noise_level": 0.03,
        "hydro_ert_abs_error": 0.0,
        "hydro_ert_rel_error": 0.03,
        "hydro_tdem_noise_level": 0.03,
        "hydro_fdem_noise_level": 0.03,
        "hydro_gravity_noise_level": 0.02,
        "hydro_dialog_text": "",
        "hydro_chat_history": [],
        "hydro_last_run": None,
        "hydro_surface_selected_points": [],
        "quick_run_mode": "Auto (LLM)",
        # --- Step-based workflow keys ---
        "hydro_root": None,
        "hydro_data_summary": None,
        "hydro_data_source": None,  # "example", "http", "local"
        "hydro_data_source_label": "",
        "hydro_data_format_choice": "Auto-detect",  # "Auto-detect", "Pre-processed (.npy)", "MODFLOW", "ParFlow"
        "hydro_accessor": None,
        "hydro_http_dataset_id": None,
        "hydro_run_clicked": False,
        "hydro_results_paths": {},
        "hydro_defaults_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Migrate persisted sessions to current notebook-aligned defaults once.
    if int(st.session_state.get("hydro_defaults_version", 0)) != HYDRO_DEFAULTS_VERSION:
        _apply_hydro_notebook_defaults(force=True)
        st.session_state.hydro_defaults_version = HYDRO_DEFAULTS_VERSION
    else:
        _apply_hydro_notebook_defaults(force=False)


def render_header() -> None:
    st.markdown('<div class="phgx-header">PyHydroGeophysX Workflows</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="phgx-subtitle-main">AQUAH: Autonomous Query-driven Understanding Agent for Hydrogeophysics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-author-text" style="margin-top:-0.2rem; margin-bottom:0.45rem;">'
        f'Paper reference: <a href="{AQUAH_PAPER_URL}" target="_blank">{AQUAH_PAPER_TITLE}</a>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-author-line">'
        '<span class="phgx-version-badge">v1.0</span>'
        '<span class="phgx-author-text">Developed by <a href="https://sites.google.com/view/hangchen" target="_blank">Hang Chen</a> · University of Iowa</span>'
        '<a href="https://www.youtube.com/watch?v=d4lgs_hQqDo" target="_blank" style="margin-left: 1rem; background: #ff0000; color: white; padding: 0.3rem 0.8rem; border-radius: 0.4rem; font-size: 0.85rem; font-weight: 600; text-decoration: none;">▶ Video Tutorial</a>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-pill">Unified Workflow</div>'
        '<div class="phgx-pill">MODFLOW/ParFlow</div>'
        '<div class="phgx-pill">ERT</div>'
        '<div class="phgx-pill">TDEM</div>'
        '<div class="phgx-pill">Seismic</div>'
        '<div class="phgx-pill">Data Fusion</div>',
        unsafe_allow_html=True,
    )


def render_example_buttons() -> None:
    st.subheader("Example workflows")
    cols = st.columns(len(EXAMPLE_REQUESTS))
    for idx, (label, text) in enumerate(EXAMPLE_REQUESTS.items()):
        if cols[idx].button(label):
            st.session_state.user_request = text
            st.rerun()
    st.caption("Click any example to auto-fill the request box.")


def render_tutorial_tab() -> None:
    st.subheader("Tutorial")

    # Video Tutorial
    st.markdown("### Video Tutorials")
    vid_col1, vid_col2 = st.columns(2)
    with vid_col1:
        st.markdown("**Getting Started**")
        st.video("https://www.youtube.com/watch?v=d4lgs_hQqDo")
    with vid_col2:
        st.markdown("**Hydro-to-Geophysics Workflow**")
        st.video("https://www.youtube.com/watch?v=1SuqL_JhiwI")

    st.markdown("---")
    st.markdown(
        """
<div class="phgx-card">
    <div class="phgx-subtitle">Run a workflow in six steps</div>
    <ol>
        <li>Initialize the context agent in the sidebar (provider, model, API key).</li>
        <li>Pick sample files from GitHub or upload your own measurements.</li>
        <li>Describe the workflow in plain language with file names and parameters.</li>
        <li>Use the example buttons to auto-fill, then edit the request to match your data.</li>
        <li>Click "Run workflow" and watch the progress and execution plan.</li>
        <li>Download the report files and review the interpretation summary.</li>
    </ol>
</div>
""",
        unsafe_allow_html=True,
    )

    # API Key Setup Section
    st.markdown("### How to Get an API Key")
    with st.expander("Step-by-step guide to obtain LLM API keys", expanded=False):
        st.markdown("""
PyHydroGeophysX requires an LLM (Large Language Model) API key to power its natural language processing capabilities.
You can use any of the following providers:

#### Option 1: OpenAI (Recommended for beginners)
1. Go to [OpenAI Platform](https://platform.openai.com/signup)
2. Create an account or sign in with Google/Microsoft
3. Navigate to **API Keys** in the left sidebar (or go to [API Keys page](https://platform.openai.com/api-keys))
4. Click **"Create new secret key"**
5. Give it a name (e.g., "PyHydroGeophysX") and click **Create**
6. **Copy the key immediately** - you won't be able to see it again!
7. Add billing information at [Billing](https://platform.openai.com/account/billing) (required for API access)

**Recommended models:** `gpt-4o-mini` (fast & cheap), `gpt-4o` (more capable)

#### Option 2: Anthropic (Claude)
1. Go to [Anthropic Console](https://console.anthropic.com/)
2. Create an account and verify your email
3. Navigate to **API Keys** in the settings
4. Click **"Create Key"**
5. Copy and save your API key securely
6. Add billing information in the Billing section

**Recommended models:** `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307` (faster)

#### Option 3: Google (Gemini)
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** in the top right
4. Select or create a Google Cloud project
5. Copy your API key

**Recommended models:** `gemini-1.5-flash` (fast), `gemini-1.5-pro` (more capable)

---

**Important Tips:**
- Keep your API key **secret** - never share it publicly or commit it to GitHub
- API usage is **pay-per-use** - typical workflow costs $0.01-0.10 per run
- Start with cheaper models (`gpt-4o-mini`, `claude-3-haiku`, `gemini-1.5-flash`) for testing
- Set up **usage limits** in your provider's dashboard to avoid unexpected charges
        """)

        st.info("💡 **Tip:** OpenAI's `gpt-4o-mini` offers the best balance of cost and performance for most hydrogeophysics workflows.")

    st.markdown("---")

    st.markdown("### Example Data from GitHub")
    for label, link in DATA_LINKS.items():
        st.markdown(f"- [{label}]({link})")

    st.markdown("---")

    # Hydrological Model Output Tutorial
    st.markdown("### Example 0: Hydrological Model Outputs (MODFLOW / ParFlow)")
    with st.expander("Step-by-step tutorial for MODFLOW/ParFlow outputs", expanded=False):
        st.markdown("""
This tutorial shows how to **load MODFLOW or ParFlow outputs** and **convert them into geophysical properties** (resistivity or velocity).

**Supported Models:**
- **MODFLOW**: Water content + porosity arrays (requires `flopy` for porosity)
- **ParFlow**: Saturation + porosity + mask arrays (now works **without** `parflow` package!)

**Workflow:**
1. Upload hydrological model output file(s) - including porosity if available
2. Describe the conversion using natural language
3. Get resistivity or velocity outputs for comparison with geophysical data
""")

        st.markdown("#### A) ParFlow Workflow (Recommended - No extra packages needed)")
        st.markdown("""
**Step 1:** Upload ParFlow `.pfb` files:
- Saturation file (e.g., `test2.out.satur.00005.pfb`)
- Porosity file (e.g., `test2.out.porosity.pfb`) - optional but recommended
- Mask file (e.g., `test2.out.mask.pfb`) - optional

**Step 2:** Use this natural language request:
""")
        st.code("""Load ParFlow outputs. I uploaded a saturation .pfb file.
Also uploaded porosity .pfb file.
Convert saturation to water content, then to resistivity with rho_sat=541 and n=1.24.""", language="text")
        
        st.markdown("""
**What happens:**
- The system reads the PFB files directly (no `parflow` package needed!)
- Saturation × Porosity → Water content
- Water content → Resistivity (or velocity) using petrophysical models
- Generates plots and reports
""")

        st.markdown("#### B) MODFLOW Workflow")
        st.markdown("""
**Step 1:** Upload MODFLOW files:
- `id.txt` (idomain file)
- The `WaterContent` binary file should be in the same directory

**Step 2:** Use this natural language request:
""")
        st.code("""Load MODFLOW outputs. I uploaded id.txt (idomain file).
Model name: TLnewtest2sfb2. Timestep: 1. Number of layers: 3.
Convert water content to resistivity using rho_sat=541 and n=1.24.""", language="text")

        st.markdown("""
**Petrophysical Parameters:**
- `rho_sat`: Saturated resistivity (Ω·m) - typically 50-1000 depending on pore water salinity
- `n`: Saturation exponent (dimensionless) - typically 1.2-2.5
- `porosity`: Can be uploaded as a file or specified as a constant value
""")

        st.success("✅ **New Feature:** ParFlow .pfb files can now be read without installing the `parflow` Python package!")

        st.info("💡 **Tip:** Use the **'ParFlow'** or **'MODFLOW'** example buttons above to auto-fill a working request.")

    # ERT Example Tutorial
    st.markdown("### Example 1: Standard ERT Inversion")
    with st.expander("Step-by-step tutorial for ERT workflow", expanded=False):
        ex1 = EXAMPLE_DATA_LINKS["ERT Example (Ex1)"]
        st.markdown(f"**Description:** {ex1['description']}")
        st.markdown(f"**Jupyter Notebook:** [Ex_Unified_Workflow_ex1.ipynb]({ex1['notebook']})")
        st.markdown(f"**Data Files:** [ERT/DAS folder]({ex1['ert_data']})")
        st.markdown(f"- Files needed: `{', '.join(ex1['files'])}`")

        st.markdown("#### Step-by-Step Instructions")
        st.markdown("""
1. **Download the data files** from the GitHub link above or upload your own ERT data
2. **Initialize the system** in the sidebar with your LLM API key
3. **Describe your workflow** in the text area. Example request:
        """)
        st.code("""We have ERT data from DAS-1 instrument at examples/data/ERT/DAS/20171105_1418.Data
and electrode file in examples/data/ERT/DAS/electrodes.dat
in the Snowy Range in southeastern Wyoming. The bedrock consists of foliated gneiss in the Cheyenne Belt.
Use specific petrophysical parameters: rho_sat = 541, porosity = 0.37, n = 1.24""", language="text")
        st.markdown("""
4. **Click "Run workflow"** - the system will:
   - Parse your natural language request
   - Load ERT data and electrode positions
   - Run resistivity inversion
   - Convert to water content using petrophysical parameters
5. **Review results** - download the generated report with resistivity and water content models
        """)

        st.markdown("#### Standard ERT Inversion Screenshots")
        image_dir = CURRENT_DIR / "images"
        for caption, filename in STANDARD_ERT_TUTORIAL_IMAGES:
            image_path = image_dir / filename
            if image_path.exists():
                st.image(str(image_path), caption=caption, width="stretch")
            else:
                st.warning(f"Missing tutorial image: {image_path}")

    # Time-Lapse Example Tutorial
    st.markdown("### Example 2: Time-Lapse ERT with Climate Integration")
    with st.expander("Step-by-step tutorial for Time-Lapse workflow", expanded=False):
        ex2 = EXAMPLE_DATA_LINKS["Time-Lapse Example (Ex2)"]
        st.markdown(f"**Description:** {ex2['description']}")
        st.markdown(f"**Jupyter Notebook:** [Ex_Unified_Workflow_ex2.ipynb]({ex2['notebook']})")
        st.markdown(f"**Data Files:** [ERT/E4D folder]({ex2['ert_data']})")
        st.markdown(f"- Files needed: `{', '.join(ex2['files'])}`")

        st.markdown("#### Step-by-Step Instructions")
        st.markdown("""
1. **Download all 4 time-lapse files** from the GitHub link above
2. **Initialize the system** in the sidebar with your LLM API key
3. **Describe your workflow** including all timestep files and climate parameters:
        """)
        st.code("""I need to run a TIME-LAPSE ERT inversion to monitor moisture infiltration.

DATA FILES FOR TIME-LAPSE INVERSION:
Please use these 4 E4D format data files located in folder data/ERT/E4D:
- 2022-03-26_0030.ohm (BASELINE)
- 2022-04-26_0030.ohm
- 2022-05-26_0030.ohm
- 2022-06-26_0030.ohm

INVERSION SETTINGS:
- Temporal Regularization Parameter: 10
- Spatial Regularization (lambda): 15

CLIMATE DATA INTEGRATION:
- Site Coordinates: 38.92584°N, -106.97998°W
- Date Range: March 2022 to June 2022
- Variables: precipitation, temperature, solar radiation""", language="text")
        st.markdown("""
4. **Click "Run workflow"** - the system will:
   - Detect time-lapse mode from multiple files
   - Run temporal inversion with regularization
   - Fetch climate data from DayMet API
   - Correlate resistivity changes with precipitation and temperature
5. **Review temporal results** - see how subsurface moisture responds to climate events
        """)

    # Data Fusion Example Tutorial
    st.markdown("### Example 3: Data Fusion (Seismic + ERT)")
    with st.expander("Step-by-step tutorial for Data Fusion workflow", expanded=False):
        ex3 = EXAMPLE_DATA_LINKS["Data Fusion Example (Ex3)"]
        st.markdown(f"**Description:** {ex3['description']}")
        st.markdown(f"**Jupyter Notebook:** [Ex_Unified_Workflow_ex3.ipynb]({ex3['notebook']})")
        st.markdown(f"**Data Files:**")
        st.markdown(f"- Seismic: [Seismic folder]({ex3['seismic_data']})")
        st.markdown(f"- ERT: [ERT/Bert folder]({ex3['ert_data']})")
        st.markdown(f"- Files needed: `{', '.join(ex3['files'])}`")

        st.markdown("#### Step-by-Step Instructions")
        st.markdown("""
1. **Download both seismic and ERT data files** from the GitHub links above
2. **Initialize the system** in the sidebar with your LLM API key
3. **Describe your multi-method workflow** with layer-specific parameters:
        """)
        st.code("""I need to characterize subsurface water content using a multi-method approach:

1. First, use field seismic refraction data to identify the boundary between regolith and fractured bedrock.
   The seismic data is in 'data/Seismic/srtfieldline2.dat' (BERT format)
   Use a velocity threshold of 1000 m/s to extract the interface.

2. Then, use this seismic structure to constrain ERT inversion.
   The ERT data is in 'data/ERT/Bert/fielddataline2.dat' (BERT format).
   Apply moderate regularization (lambda=20).

3. Finally, convert to water content using layer-specific petrophysical parameters.
   Use Monte Carlo uncertainty analysis with 100 realizations.
   - Regolith layer: rho_sat (50-250 Ωm), n (1.3-2.2), porosity (0.25-0.5)
   - Fractured bedrock layer: rho_sat (165-350 Ωm), n (2.0-2.2), porosity (0.2-0.3)""", language="text")
        st.markdown("""
4. **Click "Run workflow"** - the system will:
   - Run seismic velocity inversion
   - Extract layer interface at velocity threshold
   - Use seismic structure to constrain ERT inversion
   - Apply layer-specific petrophysics with uncertainty quantification
5. **Review integrated results** - get water content with Monte Carlo uncertainty bounds
        """)

    # Hydro -> Geophysics tutorial (new tab)
    st.markdown("### Example 4: Hydro -> Geophysics (Profile + Multi-Geophysics)")
    with st.expander("Step-by-step tutorial for Hydro -> Geophysics tab", expanded=False):
        st.markdown(
            """
This tutorial uses the **Hydro -> Geophysics** tab to convert hydrologic outputs into synthetic geophysical responses.

**Step-by-step:**
1. Open the **Hydro -> Geophysics** tab.
2. Set your hydro data folder with files: `Watercontent.npy`, `Porosity.npy`, `top.txt`, `bot.npy`.
3. Use the surface map to pick **Point 1** and **Point 2** (or type coordinates manually).
4. Choose methods (single method or batch methods).
5. Keep defaults, or open **Customize settings** to adjust acquisition and rock-physics parameters.
6. (Optional) Use **Optional Dialog Control** to set parameters with natural language when LLM is active.
7. Click **Run ... forward modeling** and download generated figures.
"""
        )
        st.code(
            """Set ERT array to dd, electrode count 96, spacing 1.5, start 10.
Set SRT source start 20, shot distance 2, sensor count 80.
Use methods ERT and SRT only, snapshot index 8.
Set rho_sat=[120,600,2200], archie_n=[2.1,1.9,2.4], sigma_s=[0.002,0,0].""",
            language="text",
        )

    st.markdown("---")
    st.markdown("### Request Template (Quick Reference)")
    st.code(
        """Run a standard ERT inversion using the DAS-1 instrument.
Data file: 20171105_1418.Data
Electrode file: electrodes.dat
Petrophysics: rho_sat=541, porosity=0.37, n=1.24
Regularization lambda: 15""",
        language="text",
    )
    st.markdown(
        """
**Tips**
- Use the upload area if your local filenames differ from the examples.
- Time-lapse data can be listed as multiple files or uploaded together.
- Add geology hints, water content targets, or climate context for richer interpretations.
"""
    )

    st.markdown("---")
    st.markdown("### References (from README.md)")
    st.markdown("The references below mirror the `README.md` citation section.")
    for ref in README_REFERENCE_ENTRIES:
        if ref.get("url"):
            st.markdown(f"- [{ref['label']}]({ref['url']})")
        else:
            st.markdown(f"- {ref['label']}")

    with st.expander("Show BibTeX references", expanded=False):
        for ref in README_REFERENCE_ENTRIES:
            st.markdown(f"**{ref['label']}**")
            st.code(ref["bibtex"], language="bibtex")


def render_concepts_tab() -> None:
    st.subheader("Hydrogeophysics Concepts")
    st.markdown(
        """
Hydrogeophysics links geophysical measurements to subsurface water, structure, and flow.
The workflows in this app focus on the methods below.
"""
    )

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown(
            """
<div class="phgx-card" style="margin-bottom: 0.9rem;">
    <div class="phgx-subtitle">Electrical Resistivity Tomography (ERT)</div>
    <ul>
        <li>Injects current through electrodes and measures voltage to map resistivity.</li>
        <li>Sensitive to water content, salinity, and lithology contrasts.</li>
        <li>Time-lapse ERT tracks changes such as recharge, pumping, or snowmelt.</li>
    </ul>
</div>
<div class="phgx-card" style="margin-bottom: 0.9rem;">
    <div class="phgx-subtitle">Seismic Refraction Tomography (SRT)</div>
    <ul>
        <li>Uses first-arrival travel times to estimate P-wave velocity structure.</li>
        <li>Highlights layer boundaries and depth to bedrock or weathered zones.</li>
    </ul>
</div>
<div class="phgx-card" style="margin-bottom: 0.9rem;">
    <div class="phgx-subtitle">Time-Domain Electromagnetics (TDEM)</div>
    <ul>
        <li>Induces eddy currents with a transmitter loop and measures decay response.</li>
        <li>Well suited for depth sounding of conductivity and salinity variations.</li>
    </ul>
</div>
<div class="phgx-card">
    <div class="phgx-subtitle">Data Fusion + Petrophysics</div>
    <ul>
        <li>Combines ERT and seismic data to reduce ambiguity in subsurface models.</li>
        <li>Petrophysical transforms connect resistivity to water content or porosity.</li>
    </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown("#### Interactive Survey Visualization")
        html_sim = """
<!DOCTYPE html>
<html>
<head>
<style>
  :root { color-scheme: light; }
  body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; background: #f8fafc; color: #1f2937; }
  #phgx-sim-container { border: 1px solid #d6dde6; border-radius: 12px; padding: 14px; background: linear-gradient(180deg, #f9fbff 0%, #eef4fb 100%); }
  .phgx-sim-header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }
  .phgx-sim-title { font-size: 15px; font-weight: 700; color: #1b262c; }
  .phgx-sim-buttons { display: flex; gap: 6px; flex-wrap: wrap; }
  .phgx-sim-btn { border: 1px solid #cbd5e1; background: #ffffff; color: #0f4c75; padding: 6px 14px; font-size: 12px; font-weight: 600; border-radius: 999px; cursor: pointer; transition: all 0.2s; }
  .phgx-sim-btn:hover { background: #e8f4f8; }
  .phgx-sim-btn.active { background: linear-gradient(135deg, #0f4c75 0%, #3d6cb9 100%); color: #fff; border-color: #0f4c75; }
  #phgx-sim-canvas { width: 100%; height: auto; border-radius: 8px; background: #fff; border: 1px solid #e1e5ec; }
  #phgx-sim-legend { margin-top: 10px; font-size: 11px; color: #475569; line-height: 1.4; padding: 8px 10px; background: #f8fafc; border-radius: 6px; border: 1px solid #e1e5ec; }
  .legend-title { font-weight: 600; color: #1b262c; margin-bottom: 4px; }
  #phgx-sim-info { margin-top: 8px; font-size: 11px; color: #64748b; }
</style>
</head>
<body>
<div id="phgx-sim-container">
  <div class="phgx-sim-header">
    <div class="phgx-sim-title">Geophysical Survey Simulator</div>
    <div class="phgx-sim-buttons">
      <button class="phgx-sim-btn active" data-mode="ert">ERT</button>
      <button class="phgx-sim-btn" data-mode="seismic">Seismic</button>
      <button class="phgx-sim-btn" data-mode="tdem">TDEM</button>
    </div>
  </div>
  <canvas id="phgx-sim-canvas"></canvas>
  <div id="phgx-sim-legend"></div>
  <div id="phgx-sim-info"></div>
</div>

<script>
(() => {
  const container = document.getElementById("phgx-sim-container");
  const canvas = document.getElementById("phgx-sim-canvas");
  const ctx = canvas.getContext("2d");
  const legend = document.getElementById("phgx-sim-legend");
  const info = document.getElementById("phgx-sim-info");
  const buttons = container.querySelectorAll(".phgx-sim-btn");

  const legendData = {
    ert: {
      title: "Electrical Resistivity Tomography (ERT)",
      text: "Current injected at A flows to B through the subsurface. The sensitivity pattern (red shading) shows where the measurement is most sensitive - forming a 'banana' shape between electrodes. Potential difference measured at M-N relates to subsurface resistivity.",
      layers: ["Soil (200 Ωm)", "Saturated (80 Ωm)", "Bedrock (1000 Ωm)"]
    },
    seismic: {
      title: "Seismic Refraction Tomography (SRT)",
      text: "P-waves expand as wavefronts from source. At layer interfaces, waves refract according to Snell's law. Head waves travel along faster layers and return to surface. At crossover distance, refracted wave arrives before direct wave.",
      layers: ["Layer 1: 500 m/s", "Layer 2: 1500 m/s", "Layer 3: 3000 m/s"]
    },
    tdem: {
      title: "Time-Domain Electromagnetics (TDEM)",
      text: "After Tx current shutoff, the decaying magnetic field induces eddy currents that form 'smoke rings' expanding outward and downward. Conductive layers (low resistivity) sustain currents longer, producing stronger late-time response.",
      layers: ["Resistive (500 Ωm)", "Conductive (20 Ωm)", "Resistive (800 Ωm)"]
    }
  };

  let mode = "ert";
  let t = 0;
  let lastWidth = 0;

  // Layer model with resistivity/velocity
  const geoModel = {
    layers: [
      { y: 0.18, h: 0.15, color: "#d4a574", rho: 200, vel: 500, name: "Dry soil" },
      { y: 0.33, h: 0.22, color: "#6ba3c7", rho: 80, vel: 1500, name: "Saturated" },
      { y: 0.55, h: 0.45, color: "#8b7355", rho: 1000, vel: 3000, name: "Bedrock" }
    ]
  };

  function resize() {
    const width = Math.max(320, container.clientWidth - 28);
    const height = 300;
    if (width !== lastWidth) {
      canvas.width = width;
      canvas.height = height;
      lastWidth = width;
    }
  }

  function setMode(next) {
    mode = next;
    t = 0;
    buttons.forEach(btn => btn.classList.toggle("active", btn.dataset.mode === next));
    const data = legendData[next];
    legend.innerHTML = `<div class="legend-title">${data.title}</div>${data.text}<br><small><b>Model:</b> ${data.layers.join(" | ")}</small>`;
  }

  buttons.forEach(btn => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

  function drawGround(groundY) {
    const w = canvas.width, h = canvas.height;

    // Sky
    ctx.fillStyle = "#e3f2fd";
    ctx.fillRect(0, 0, w, groundY);

    // Draw geological layers
    geoModel.layers.forEach((layer, i) => {
      const ly = h * layer.y;
      const lh = h * layer.h;
      ctx.fillStyle = layer.color;
      ctx.fillRect(0, ly, w, lh);

      // Layer texture
      ctx.globalAlpha = 0.3;
      for (let j = 0; j < 20; j++) {
        ctx.fillStyle = i === 1 ? "#4a90a4" : "#5d4e37";
        const px = (j * 47 + i * 13) % w;
        const py = ly + 5 + (j * 17) % (lh - 10);
        ctx.beginPath();
        ctx.arc(px, py, 2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    });

    // Ground surface
    ctx.strokeStyle = "#2d5016";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(w, groundY);
    ctx.stroke();

    // Layer boundaries with labels
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#00000044";
    ctx.lineWidth = 1;
    ctx.font = "9px sans-serif";
    ctx.fillStyle = "#555";
    ctx.textAlign = "right";

    [0.33, 0.55].forEach((y, i) => {
      ctx.beginPath();
      ctx.moveTo(0, h * y);
      ctx.lineTo(w, h * y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Depth scale
    ctx.textAlign = "left";
    ctx.fillStyle = "#333";
    ctx.fillText("0m", 4, groundY + 12);
    ctx.fillText("5m", 4, h * 0.42);
    ctx.fillText("15m", 4, h * 0.65);
    ctx.fillText("30m", 4, h * 0.90);
  }

  function drawERT(groundY) {
    const w = canvas.width, h = canvas.height;
    const electrodeY = groundY;

    // Animated dipole-dipole array - electrodes move through different n-levels
    const cycleTime = 6; // seconds per full cycle
    const animPhase = (t % cycleTime) / cycleTime;

    // Dipole spacing (a) and n-level animation (1 to 5)
    const dipoleSpacing = 35; // electrode spacing within dipole
    const nLevel = 1 + Math.floor(animPhase * 5); // n = 1, 2, 3, 4, 5
    const nProgress = (animPhase * 5) % 1; // smooth transition within each n

    // Fixed current dipole A-B position
    const aX = w * 0.15;
    const bX = aX + dipoleSpacing;

    // Moving potential dipole M-N based on n-level
    const separation = nLevel * dipoleSpacing;
    const mX = bX + separation;
    const nX = mX + dipoleSpacing;

    // Calculate geometric factor K for dipole-dipole: K = π * n * (n+1) * (n+2) * a
    const geoFactor = Math.PI * nLevel * (nLevel + 1) * (nLevel + 2) * (dipoleSpacing / 100);

    // Subsurface resistivity (two-layer model)
    const rho1 = 100; // top layer resistivity (Ohm-m)
    const rho2 = 500; // bottom layer resistivity (Ohm-m)

    // Apparent resistivity changes with depth of investigation
    const depthFactor = Math.min(1, nLevel / 3);
    const apparentRho = rho1 * (1 - depthFactor) + rho2 * depthFactor;

    // Calculate voltage: V = I * rho / K (I = 1A assumed)
    const current = 100; // mA
    const voltage = (current * apparentRho / geoFactor / 10);

    // Draw sensitivity pattern (banana-shaped) - size depends on n-level
    const midAB = (aX + bX) / 2;
    const midMN = (mX + nX) / 2;
    const centerX = (midAB + midMN) / 2;
    const sepDist = midMN - midAB;
    const maxDepth = sepDist * 0.5;

    // Sensitivity contours
    for (let level = 5; level >= 1; level--) {
      const alpha = 0.06 + level * 0.035;
      const depth = maxDepth * (level / 5);
      const curveWidth = sepDist * (0.25 + level * 0.12);

      ctx.fillStyle = `rgba(220, 50, 50, ${alpha})`;
      ctx.beginPath();
      ctx.moveTo(midAB, electrodeY);
      ctx.quadraticCurveTo(centerX - curveWidth/3, electrodeY + depth * 0.7, centerX, electrodeY + depth);
      ctx.quadraticCurveTo(centerX + curveWidth/3, electrodeY + depth * 0.7, midMN, electrodeY);
      ctx.closePath();
      ctx.fill();
    }

    // Equipotential lines near current electrodes
    ctx.strokeStyle = "rgba(37, 99, 235, 0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (let i = -2; i <= 2; i++) {
      if (i === 0) continue;
      const px = (aX + bX) / 2 + i * 12;
      ctx.beginPath();
      ctx.moveTo(px - i * 4, electrodeY);
      ctx.quadraticCurveTo(px, electrodeY + 50, px + i * 4, electrodeY + 90);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw electrodes
    const drawElectrode = (x, label, isSource) => {
      ctx.fillStyle = isSource ? "#dc2626" : "#2563eb";
      ctx.fillRect(x - 2, electrodeY - 14, 4, 16);
      ctx.beginPath();
      ctx.arc(x, electrodeY - 16, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#1f2937";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x, electrodeY - 24);
    };

    drawElectrode(aX, "A", true);
    drawElectrode(bX, "B", true);
    drawElectrode(mX, "M", false);
    drawElectrode(nX, "N", false);

    // Connection line between M and N (voltmeter wires)
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(mX, electrodeY - 16);
    ctx.lineTo(mX, electrodeY - 38);
    ctx.lineTo(nX, electrodeY - 38);
    ctx.lineTo(nX, electrodeY - 16);
    ctx.stroke();

    // Voltmeter display box
    const vmX = (mX + nX) / 2;
    ctx.fillStyle = "#dbeafe";
    ctx.strokeStyle = "#2563eb";
    ctx.beginPath();
    ctx.roundRect(vmX - 28, electrodeY - 54, 56, 22, 4);
    ctx.fill();
    ctx.stroke();

    // Display voltage value (changes with electrode positions)
    ctx.fillStyle = "#1e40af";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("ΔV=" + voltage.toFixed(1) + "mV", vmX, electrodeY - 39);

    // n-level indicator
    ctx.fillStyle = "#7c3aed";
    ctx.font = "bold 11px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("n = " + nLevel, w - 75, electrodeY - 55);

    // Apparent resistivity indicator
    ctx.fillStyle = "#065f46";
    ctx.font = "10px sans-serif";
    ctx.fillText("ρₐ = " + apparentRho.toFixed(0) + " Ωm", w - 75, electrodeY - 40);

    // Layer resistivities label
    ctx.fillStyle = "#666";
    ctx.font = "9px sans-serif";
    ctx.fillText("ρ₁=" + rho1 + "Ωm", 5, h * 0.26);
    ctx.fillText("ρ₂=" + rho2 + "Ωm", 5, h * 0.48);

    info.textContent = "Dipole-Dipole array (n=" + nLevel + "): As n increases, M-N moves away from A-B, probing deeper. Voltage decreases with distance.";
  }

  function drawSeismic(groundY) {
    const w = canvas.width, h = canvas.height;
    const sourceX = 40;
    const nGeophones = 12;

    // Layer interfaces
    const interface1 = h * 0.33;
    const interface2 = h * 0.55;

    // Velocities
    const v1 = 500, v2 = 1500, v3 = 3000;

    // Critical angles
    const ic1 = Math.asin(v1 / v2);
    const ic2 = Math.asin(v1 / v3);

    // Animation timing
    const cycleTime = 5;
    const waveT = t % cycleTime;
    const waveRadius = waveT * 60;

    // Draw source
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(sourceX, groundY - 8, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.font = "bold 8px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("S", sourceX, groundY - 5);

    // Draw geophones
    for (let i = 0; i < nGeophones; i++) {
      const gx = 80 + i * ((w - 100) / (nGeophones - 1));
      ctx.fillStyle = "#059669";
      ctx.beginPath();
      ctx.moveTo(gx, groundY - 2);
      ctx.lineTo(gx - 4, groundY - 10);
      ctx.lineTo(gx + 4, groundY - 10);
      ctx.closePath();
      ctx.fill();
    }

    // Direct wave (circular wavefront in layer 1)
    if (waveRadius > 0 && waveRadius < w) {
      ctx.strokeStyle = "#dc262699";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(sourceX, groundY, waveRadius, -0.3, 0.3);
      ctx.stroke();

      // Wavefront label
      if (waveRadius > 40 && waveRadius < 150) {
        ctx.fillStyle = "#dc2626";
        ctx.font = "9px sans-serif";
        ctx.fillText("Direct (V₁)", sourceX + waveRadius * 0.7, groundY - 10);
      }
    }

    // Refracted wave at interface 1
    const timeToInterface1 = (interface1 - groundY) / (v1 * 0.08);
    if (waveT > timeToInterface1 * 0.3) {
      const refractT = waveT - timeToInterface1 * 0.3;
      const headwaveDist = refractT * v2 * 0.08;

      // Down-going ray at critical angle
      ctx.strokeStyle = "#f59e0b99";
      ctx.lineWidth = 2;
      const critX = sourceX + (interface1 - groundY) * Math.tan(ic1);
      ctx.beginPath();
      ctx.moveTo(sourceX, groundY);
      ctx.lineTo(critX, interface1);
      ctx.stroke();

      // Head wave along interface
      if (headwaveDist > 0) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(critX, interface1);
        ctx.lineTo(Math.min(critX + headwaveDist, w - 20), interface1);
        ctx.stroke();

        // Up-going rays
        ctx.strokeStyle = "#f59e0b66";
        ctx.lineWidth = 1;
        for (let i = 2; i < nGeophones; i++) {
          const gx = 80 + i * ((w - 100) / (nGeophones - 1));
          if (gx > critX && gx < critX + headwaveDist) {
            ctx.beginPath();
            ctx.moveTo(gx, interface1);
            ctx.lineTo(gx, groundY);
            ctx.stroke();
          }
        }

        if (headwaveDist > 50 && headwaveDist < 180) {
          ctx.fillStyle = "#f59e0b";
          ctx.font = "9px sans-serif";
          ctx.fillText("Head wave (V₂)", critX + headwaveDist * 0.5, interface1 - 5);
        }
      }
    }

    // Refracted wave at interface 2
    const timeToInterface2 = (interface2 - groundY) / (v1 * 0.06);
    if (waveT > timeToInterface2 * 0.4) {
      const refractT2 = waveT - timeToInterface2 * 0.4;
      const headwaveDist2 = refractT2 * v3 * 0.1;

      ctx.strokeStyle = "#8b5cf699";
      ctx.lineWidth = 2;
      const critX2 = sourceX + (interface2 - groundY) * Math.tan(ic2) * 0.8;
      ctx.beginPath();
      ctx.moveTo(sourceX, groundY);
      ctx.lineTo(critX2, interface2);
      ctx.stroke();

      if (headwaveDist2 > 0) {
        ctx.strokeStyle = "#8b5cf6";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(critX2, interface2);
        ctx.lineTo(Math.min(critX2 + headwaveDist2, w - 20), interface2);
        ctx.stroke();

        if (headwaveDist2 > 40 && headwaveDist2 < 150) {
          ctx.fillStyle = "#8b5cf6";
          ctx.font = "9px sans-serif";
          ctx.fillText("Head wave (V₃)", critX2 + headwaveDist2 * 0.4, interface2 - 5);
        }
      }
    }

    // Travel-time plot
    const plotX = w - 100, plotY = groundY + 15, plotW = 90, plotH = 70;
    ctx.fillStyle = "#ffffffee";
    ctx.strokeStyle = "#94a3b8";
    ctx.lineWidth = 1;
    ctx.fillRect(plotX, plotY, plotW, plotH);
    ctx.strokeRect(plotX, plotY, plotW, plotH);

    // Axes
    ctx.fillStyle = "#334155";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("t (ms)", plotX + 2, plotY + 10);
    ctx.fillText("x (m)", plotX + plotW - 22, plotY + plotH - 3);

    // Travel-time curves
    const xScale = (plotW - 10) / 100;
    const tScale = (plotH - 15) / 80;

    // Direct wave: t = x/v1
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x <= 100; x += 2) {
      const tt = x / 0.5; // t in ms
      const px = plotX + 5 + x * xScale;
      const py = plotY + plotH - 5 - tt * tScale;
      if (py > plotY + 10) {
        x === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    // Refracted V2: t = 2*z1*cos(ic)/v1 + x/v2
    ctx.strokeStyle = "#f59e0b";
    ctx.beginPath();
    const ti1 = 15; // intercept time
    const crossover1 = 35;
    for (let x = crossover1; x <= 100; x += 2) {
      const tt = ti1 + x / 1.5;
      const px = plotX + 5 + x * xScale;
      const py = plotY + plotH - 5 - tt * tScale;
      if (py > plotY + 10) {
        x === crossover1 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    // Refracted V3
    ctx.strokeStyle = "#8b5cf6";
    ctx.beginPath();
    const ti2 = 25;
    const crossover2 = 60;
    for (let x = crossover2; x <= 100; x += 2) {
      const tt = ti2 + x / 3;
      const px = plotX + 5 + x * xScale;
      const py = plotY + plotH - 5 - tt * tScale;
      if (py > plotY + 10) {
        x === crossover2 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      }
    }
    ctx.stroke();

    // Crossover distance marker
    ctx.fillStyle = "#065f46";
    ctx.font = "7px sans-serif";
    ctx.fillText("Xc", plotX + 5 + crossover1 * xScale - 5, plotY + plotH - 8);

    info.textContent = "Seismic refraction: Snell's law at interfaces. Head waves travel at V₂, V₃. Crossover distance Xc where refracted arrives first.";
  }

  function drawTDEM(groundY) {
    const w = canvas.width, h = canvas.height;
    const loopCenterX = w * 0.4;
    const loopRadius = 50;

    // Conductive layer (low resistivity)
    const condTop = h * 0.33;
    const condBot = h * 0.55;

    // Animation phase
    const cycleTime = 4;
    const phase = (t % cycleTime) / cycleTime;

    // Transmitter loop
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.ellipse(loopCenterX, groundY - 5, loopRadius, 8, 0, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = "#0f766e";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Tx Loop", loopCenterX, groundY - 18);

    // Receiver
    const rxX = loopCenterX;
    ctx.fillStyle = "#7c3aed";
    ctx.beginPath();
    ctx.arc(rxX, groundY - 5, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "8px sans-serif";
    ctx.fillText("Rx", rxX + 12, groundY - 2);

    // Current waveform indicator
    ctx.fillStyle = "#f8fafc";
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    const cwX = 15, cwY = groundY - 50, cwW = 50, cwH = 35;
    ctx.fillRect(cwX, cwY, cwW, cwH);
    ctx.strokeRect(cwX, cwY, cwW, cwH);

    ctx.fillStyle = "#334155";
    ctx.font = "7px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Tx Current", cwX + 2, cwY + 8);

    // Draw current waveform
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cwX + 5, cwY + 25);
    ctx.lineTo(cwX + 15, cwY + 25);
    ctx.lineTo(cwX + 15, cwY + 12);
    ctx.lineTo(cwX + 35, cwY + 12);
    ctx.lineTo(cwX + 35, cwY + 25);
    ctx.lineTo(cwX + 45, cwY + 25);
    ctx.stroke();

    // Time indicator on waveform
    const timeX = cwX + 5 + phase * 40;
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(timeX, phase < 0.25 ? cwY + 25 : (phase < 0.75 ? cwY + 12 : cwY + 25), 3, 0, Math.PI * 2);
    ctx.fill();

    // Phase 1: Current ON - Primary field
    if (phase < 0.25) {
      ctx.strokeStyle = "rgba(15, 118, 110, 0.5)";
      ctx.lineWidth = 2;

      // Primary magnetic field lines
      for (let i = 1; i <= 4; i++) {
        const fieldDepth = groundY + i * 25;
        const fieldWidth = loopRadius + i * 30;
        ctx.beginPath();
        ctx.moveTo(loopCenterX - fieldWidth, fieldDepth);
        ctx.quadraticCurveTo(loopCenterX, fieldDepth + i * 10, loopCenterX + fieldWidth, fieldDepth);
        ctx.stroke();

        // Field direction arrows
        ctx.fillStyle = "rgba(15, 118, 110, 0.6)";
        ctx.beginPath();
        ctx.moveTo(loopCenterX + fieldWidth - 10, fieldDepth - 3);
        ctx.lineTo(loopCenterX + fieldWidth, fieldDepth);
        ctx.lineTo(loopCenterX + fieldWidth - 10, fieldDepth + 3);
        ctx.fill();
      }

      ctx.fillStyle = "#0f766e";
      ctx.font = "9px sans-serif";
      ctx.fillText("Primary B-field (Tx ON)", loopCenterX, h * 0.85);
    }

    // Phase 2: Current OFF - Eddy currents (smoke rings)
    if (phase >= 0.25) {
      const diffusePhase = (phase - 0.25) / 0.75;

      // "Smoke ring" eddy currents expanding and diffusing down
      const nRings = 5;
      for (let ring = 0; ring < nRings; ring++) {
        const ringAge = diffusePhase - ring * 0.12;
        if (ringAge < 0 || ringAge > 1) continue;

        // Ring expands outward and moves down with sqrt(t) behavior
        const ringDepth = groundY + 20 + Math.sqrt(ringAge) * (h - groundY - 40);
        const ringRadius = loopRadius * (0.8 + ringAge * 1.5);

        // Decay is faster in resistive layers, slower in conductive
        let decay;
        if (ringDepth < condTop) {
          decay = Math.exp(-ringAge * 4); // Fast decay in resistive
        } else if (ringDepth < condBot) {
          decay = Math.exp(-ringAge * 1.5); // Slow decay in conductive
        } else {
          decay = Math.exp(-ringAge * 5); // Fast decay in resistive
        }

        // Draw eddy current ring
        ctx.strokeStyle = `rgba(234, 88, 12, ${decay * 0.8})`;
        ctx.lineWidth = 2.5 * decay + 0.5;
        ctx.beginPath();
        ctx.ellipse(loopCenterX, ringDepth, ringRadius, 6 + ring * 2, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Current direction indicators
        if (decay > 0.3) {
          const arrowAngle = t * 4 + ring;
          for (let a = 0; a < 4; a++) {
            const angle = arrowAngle + a * Math.PI / 2;
            const ax = loopCenterX + Math.cos(angle) * ringRadius;
            const ay = ringDepth + Math.sin(angle) * (6 + ring * 2);
            ctx.fillStyle = `rgba(234, 88, 12, ${decay})`;
            ctx.beginPath();
            ctx.arc(ax, ay, 2, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Secondary field going up to receiver
      const secIntensity = Math.exp(-diffusePhase * 2) * 0.6;
      if (secIntensity > 0.1) {
        ctx.strokeStyle = `rgba(124, 58, 237, ${secIntensity})`;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        for (let i = 1; i <= 3; i++) {
          ctx.beginPath();
          ctx.moveTo(loopCenterX - 20 - i * 8, groundY - 5 - i * 8);
          ctx.quadraticCurveTo(loopCenterX, groundY - 5 - i * 12, loopCenterX + 20 + i * 8, groundY - 5 - i * 8);
          ctx.stroke();
        }
        ctx.setLineDash([]);
      }

      ctx.fillStyle = "#ea580c";
      ctx.font = "9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Eddy currents diffuse (smoke rings)", loopCenterX, h * 0.85);
    }

    // Decay curve plot
    const dcX = w - 95, dcY = groundY + 20, dcW = 85, dcH = 65;
    ctx.fillStyle = "#ffffffee";
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    ctx.fillRect(dcX, dcY, dcW, dcH);
    ctx.strokeRect(dcX, dcY, dcW, dcH);

    ctx.fillStyle = "#334155";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Voltage Decay", dcX + 2, dcY + 10);
    ctx.fillText("log(V)", dcX + 2, dcY + 22);
    ctx.fillText("log(t)", dcX + dcW - 22, dcY + dcH - 2);

    // Draw decay curve with slope change at conductive layer
    ctx.strokeStyle = "#7c3aed";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < dcW - 12; i++) {
      const tNorm = i / (dcW - 12);
      let v;
      if (tNorm < 0.3) {
        v = Math.exp(-tNorm * 4) * 0.9; // Early time - steep
      } else if (tNorm < 0.7) {
        v = Math.exp(-0.3 * 4) * Math.exp(-(tNorm - 0.3) * 1.5) * 0.9; // Conductive layer - slower decay
      } else {
        v = Math.exp(-0.3 * 4) * Math.exp(-0.4 * 1.5) * Math.exp(-(tNorm - 0.7) * 5) * 0.9;
      }
      const px = dcX + 6 + i;
      const py = dcY + 28 + (1 - v) * (dcH - 35);
      i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Slope annotation
    ctx.fillStyle = "#6b7280";
    ctx.font = "7px sans-serif";
    ctx.fillText("slow", dcX + dcW * 0.4, dcY + 45);
    ctx.fillText("(cond.)", dcX + dcW * 0.38, dcY + 52);

    // Current time marker
    if (phase >= 0.25) {
      const tPos = (phase - 0.25) / 0.75;
      const cursorX = dcX + 6 + tPos * (dcW - 12);
      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      ctx.arc(cursorX, dcY + dcH - 10, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    info.textContent = "TDEM: After Tx off, eddy currents diffuse as 'smoke rings'. Conductive layers slow decay rate → detectable in late-time response.";
  }

  function animate() {
    resize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const groundY = canvas.height * 0.18;
    drawGround(groundY);

    if (mode === "ert") drawERT(groundY);
    else if (mode === "seismic") drawSeismic(groundY);
    else if (mode === "tdem") drawTDEM(groundY);

    t += 0.025;
    requestAnimationFrame(animate);
  }

  setMode(mode);
  animate();
})();
</script>
</body>
</html>
"""
        components.html(html_sim, height=450)
        st.caption("Interactive visualization showing geophysical survey physics. Click tabs to explore different methods.")

    # LLM-powered explanation section
    st.markdown("---")
    st.markdown("### 🤖 Ask AI About Hydrogeophysics")
    st.markdown(
        """
<div class="phgx-card">
    <div class="phgx-subtitle">Get AI-Powered Explanations</div>
    <p style="margin-bottom: 0.5rem; color: #475569;">
        Use the initialized LLM to ask questions about hydrogeophysics concepts,
        get help with Python code, or understand your geophysical data.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Initialize session state for AI chat
    if "concept_chat_history" not in st.session_state:
        st.session_state.concept_chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    # Check if there's a pending question from button click
    initial_value = ""
    if st.session_state.pending_question:
        initial_value = st.session_state.pending_question
        st.session_state.pending_question = ""  # Clear it after using

    # Example questions as buttons
    st.markdown("**Quick questions:**")
    example_questions = [
        "What is Archie's Law and how is it used in hydrogeophysics?",
        "How do I choose regularization parameters for ERT inversion?",
        "What is the difference between Wenner and Dipole-Dipole arrays?",
        "Show me Python code with PyHydroGeophysX to run ERT inversion and plot results",
    ]

    cols = st.columns(2)
    for idx, question in enumerate(example_questions):
        if cols[idx % 2].button(question, key=f"example_q_{idx}", width="stretch"):
            st.session_state.pending_question = question
            st.rerun()

    # Text input for custom questions
    user_question = st.text_area(
        "Or type your own question:",
        value=initial_value,
        height=100,
        placeholder="Ask about ERT, seismic, petrophysics, Python code, or any hydrogeophysics concept...",
    )

    col_ask, col_clear = st.columns([3, 1])
    ask_clicked = col_ask.button("🔍 Ask AI", type="primary", width="stretch")
    clear_clicked = col_clear.button("Clear History", width="stretch")

    if clear_clicked:
        st.session_state.concept_chat_history = []
        st.session_state.pending_question = ""
        st.rerun()

    if ask_clicked and user_question.strip():
        if not st.session_state.context_agent:
            st.warning("Please initialize the system in the sidebar first (set your API key).")
        else:
            with st.spinner("Thinking..."):
                try:
                    # Build context-aware prompt for hydrogeophysics
                    system_context = """You are a helpful hydrogeophysics expert assistant for PyHydroGeophysX.
You help users understand geophysical concepts, Python code for geophysical analysis,
and best practices for ERT, seismic, TDEM, and petrophysical workflows.

## PyHydroGeophysX Library Overview
PyHydroGeophysX is an AI-powered hydrogeophysics workflow system. When users ask for code examples,
ALWAYS show how to use PyHydroGeophysX agents first, then optionally show lower-level PyGIMLi code.

### Key PyHydroGeophysX Components:
1. **ContextInputAgent** - Parses natural language requests into workflow configurations
2. **BaseAgent.run_unified_agent_workflow()** - Main entry point for all workflows
3. **ERTAgent** - Handles ERT inversion using ResIPy/PyGIMLi
4. **SeismicAgent** - Handles seismic refraction tomography
5. **PetrophysicsAgent** - Converts resistivity to water content using Archie's Law
6. **TimeLapseAgent** - Handles multi-timestep ERT with temporal regularization
7. **DataFusionAgent** - Integrates seismic + ERT with structure constraints
8. **ClimateAgent** - Fetches and integrates meteorological data from DayMet

### Example PyHydroGeophysX Usage Patterns:

**Standard ERT Workflow:**
```python
from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent

# Initialize context agent
context_agent = ContextInputAgent(api_key=api_key, model='gpt-4o-mini', llm_provider='openai')

# Define workflow in natural language
user_request = '''Run ERT inversion on data.ohm with electrode file electrodes.dat.
Use regularization lambda=20 and convert to water content with rho_sat=500, porosity=0.35, n=1.5'''

# Parse and execute
config = context_agent.parse_request(user_request)
results, plan, interpretation, files = BaseAgent.run_unified_agent_workflow(
    config, api_key, 'gpt-4o-mini', 'openai', output_dir
)
```

**Time-Lapse ERT:**
```python
user_request = '''Run time-lapse ERT on files: baseline.ohm, time1.ohm, time2.ohm
Temporal regularization: 10, Spatial lambda: 15
Fetch climate data for coordinates 38.9N, -107.0W from March to June 2022'''
```

**Data Fusion (Seismic + ERT):**
```python
user_request = '''Use seismic data srt_data.dat with velocity threshold 1000 m/s
to constrain ERT inversion of ert_data.dat.
Layer petrophysics: regolith (rho_sat 50-250), bedrock (rho_sat 200-500)'''
```

### Key Parameters:
- **lambda (regularization)**: Controls smoothness (typical: 10-50, higher=smoother)
- **rho_sat**: Saturated resistivity in Archie's Law (Ωm)
- **porosity**: Rock/soil porosity (0-1)
- **n**: Archie's saturation exponent (typically 1.3-2.5)
- **velocity_threshold**: For seismic layer extraction (m/s)

When providing code examples:
1. FIRST show PyHydroGeophysX natural language approach
2. THEN optionally show equivalent PyGIMLi/low-level code if relevant
3. Use NumPy and matplotlib for data manipulation and plotting
4. Be concise but thorough. Use bullet points for clarity when appropriate.
5. If asked about specific parameters, provide typical ranges and explain the physical meaning."""

                    full_prompt = f"{system_context}\n\nUser question: {user_question}"

                    # Use the context agent's LLM to get a response
                    response = st.session_state.context_agent.query_llm(full_prompt)

                    # Add to chat history
                    st.session_state.concept_chat_history.append({
                        "question": user_question,
                        "answer": response
                    })

                    # Rerun to show the response (text area will be empty on next run)
                    st.rerun()

                except Exception as e:
                    st.error(f"Error getting AI response: {e}")

    # Display chat history
    if st.session_state.concept_chat_history:
        st.markdown("---")
        st.markdown("### Conversation History")
        for i, chat in enumerate(reversed(st.session_state.concept_chat_history)):
            with st.expander(f"Q: {chat['question'][:60]}...", expanded=(i == 0)):
                st.markdown(f"**Question:** {chat['question']}")
                st.markdown("---")
                st.markdown(f"**Answer:**\n\n{chat['answer']}")


def render_author_tab() -> None:
    # Custom CSS for author page
    st.markdown("""
    <style>
    .ship-header {
        background: linear-gradient(135deg, #0f4c75 0%, #3d6cb9 50%, #2d9c5b 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .ship-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .ship-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.6;
    }
    .profile-card {
        background: #f8fafc;
        border: 1px solid #e1e5ec;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .profile-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1b262c;
    }
    .profile-title {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 0.5rem;
    }
    .profile-contact {
        font-size: 0.9rem;
        color: #64748b;
    }
    .news-item {
        padding: 1rem;
        border-left: 4px solid #3d6cb9;
        background: #f8fafc;
        margin-bottom: 0.8rem;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .news-date {
        font-size: 0.8rem;
        color: #64748b;
        font-weight: 600;
    }
    .news-content {
        font-size: 0.95rem;
        color: #334155;
        margin-top: 0.3rem;
    }
    .link-card {
        display: block;
        padding: 1.2rem;
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 0.75rem;
        text-decoration: none;
        color: inherit;
        margin-bottom: 0.8rem;
        transition: all 0.2s ease;
    }
    .link-card:hover {
        border-color: #3d6cb9;
        box-shadow: 0 4px 12px rgba(61, 108, 185, 0.15);
        transform: translateY(-2px);
    }
    .link-card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f4c75;
        margin-bottom: 0.3rem;
    }
    .link-card-desc {
        font-size: 0.85rem;
        color: #64748b;
    }
    .member-card {
        background: white;
        border: 1px solid #e1e5ec;
        border-radius: 0.75rem;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .member-role {
        font-size: 0.75rem;
        color: white;
        background: #3d6cb9;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .member-name {
        font-weight: 600;
        color: #1b262c;
    }
    .member-info {
        font-size: 0.85rem;
        color: #64748b;
    }
    </style>
    """, unsafe_allow_html=True)

    # SHIP Lab Header
    st.markdown("""
    <div class="ship-header">
        <div class="ship-title">SHIP Lab</div>
        <div class="ship-subtitle">
            <strong>S</strong>ustainability, <strong>H</strong>ydrogeophysics, <strong>I</strong>maging, & <strong>P</strong>rediction<br>
            Advancing Earth systems understanding through integrated research approaches
        </div>
    </div>
    """, unsafe_allow_html=True)

    # PI Profile Card
    col_profile, col_contact = st.columns([2, 1])

    with col_profile:
        st.markdown("""
        <div class="profile-card">
            <div class="profile-name">Hang Chen, Ph.D.</div>
            <div class="profile-title">
                Assistant Professor, School of Earth, Environment, and Sustainability<br>
                University of Iowa | Affiliated Faculty, Lawrence Berkeley National Laboratory
            </div>
            <div class="profile-contact">
                📧 hchen117@uiowa.edu &nbsp;|&nbsp; 📍 23 Trowbridge Hall, Iowa City, IA
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_contact:
        st.markdown("[🌐 **Visit Full Website**](https://sites.google.com/view/hangchen)")
        st.markdown("[💻 **GitHub**](https://github.com/geohang)")

    st.markdown("---")

    # Sub-tabs inside the Author tab
    sub_tab3, sub_tab1, sub_tab4, sub_tab5, sub_tab6 = st.tabs([
        "🔬 Research",
        "🏠 Lab & People",
        "📄 Publications",
        "📚 Teaching",
        "💻 Open Source"
    ])

    # --- Lab & People Tab ---
    with sub_tab1:
        st.markdown("### SHIP Lab Members")
        st.markdown("*For full details, visit [SHIP Lab & People](https://sites.google.com/view/hangchen)*")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="member-card">
                <div class="member-role">Principal Investigator</div>
                <div class="member-name">Hang Chen</div>
                <div class="member-info">Assistant Professor, University of Iowa</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="member-card">
                <div class="member-role">PhD Student</div>
                <div class="member-name">Chen Xiong</div>
                <div class="member-info">Hydrogeophysics Research</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="member-card">
                <div class="member-role" style="background: #f59e0b;">Undergraduate</div>
                <div class="member-name">Cameron Roach</div>
                <div class="member-info">Gravity and Magnetic data joint inversion for geological hydrogen exploration</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="member-card">
                <div class="member-role">Master's Student</div>
                <div class="member-name">Weiyu Guo</div>
                <div class="member-info">Geophysical Modeling</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="member-card">
                <div class="member-role" style="background: #f59e0b;">Undergraduate</div>
                <div class="member-name">Jax Waller</div>
                <div class="member-info">Processing airborne electromagnetic data</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="member-card">
                <div class="member-role" style="background: #2d9c5b;">Open Position</div>
                <div class="member-name">Postdoc Opening</div>
                <div class="member-info">Contact for opportunities!</div>
            </div>
            """, unsafe_allow_html=True)

        st.info("🎓 **Interested in joining?** Visit [Opportunities](https://sites.google.com/view/hangchen/opportunities) for current openings.")

    # --- Research Tab ---
    with sub_tab3:
        st.markdown("### Research")
        st.markdown("*For detailed descriptions, visit [Research](https://sites.google.com/view/hangchen/research_1)*")

        # Research Methods
        st.markdown("#### Research Methods")
        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">📡 1. Advanced 4D Geophysical Imaging</div>
            <div class="link-card-desc">Joint inversion, structurally-constrained inversion, temporal constraint integration</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">💧 2. Integrated Hydrological Modeling</div>
            <div class="link-card-desc">Geophysics-informed hydrologic modeling, subsurface parameterization</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">🤖 3. AI-driven Solutions</div>
            <div class="link-card-desc">Deep learning inversion, AI agents, pattern recognition, automated workflows</div>
        </a>
        """, unsafe_allow_html=True)

        # Research Applications
        st.markdown("#### Research Applications")
        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">🌊 1. Integrated Watershed Analysis & Management</div>
            <div class="link-card-desc">Surface-groundwater interactions, climate impacts, sustainable water management</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">🏔️ 2. Critical Zone Ecosystem Dynamics</div>
            <div class="link-card-desc">Rock-soil-plant-atmosphere interactions, ecosystem resilience, bedrock hydrology</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">☢️ 3. Environmental Monitoring & Protection</div>
            <div class="link-card-desc">Nuclear waste disposal monitoring, 4D THM process characterization</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">⛏️ 4. Critical Mineral & Resource Characterization</div>
            <div class="link-card-desc">Electromagnetic methods for mineral deposits, geothermal resource assessment</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/research_1" target="_blank" class="link-card">
            <div class="link-card-title">⚡ 5. Geological Hydrogen Exploration</div>
            <div class="link-card-desc">Natural hydrogen accumulation identification using integrated geophysical methods</div>
        </a>
        """, unsafe_allow_html=True)

    # --- Publications Tab ---
    with sub_tab4:
        st.markdown("### Selected Publications")
        st.markdown("*For complete list, visit [Publications](https://sites.google.com/view/hangchen/publications)*")

        pubs_preview = [
            ("2025", "Development of an ERT-based framework for bentonite buffers monitoring - Part I & II", "JGR: Solid Earth"),
            ("2024", "Electrical resistivity changes during heating experiments in salt formations", "Geophysical Research Letters"),
            ("2024", "Influence of subsurface critical zone structure on hydrological partitioning", "Geophysical Research Letters"),
            ("2023", "Geophysics-informed hydrologic modeling of a mountain headwater catchment", "Water Resources Research"),
        ]

        for year, title, journal in pubs_preview:
            st.markdown(f"**{year}** | {title} - *{journal}*")

        st.markdown("---")
        st.markdown("[📄 **View All Publications →**](https://sites.google.com/view/hangchen/publications)")

    # --- Teaching Tab ---
    with sub_tab5:
        st.markdown("### Teaching")
        st.markdown("*For course materials, visit [Teaching](https://sites.google.com/view/hangchen/teaching)*")

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/teaching" target="_blank" class="link-card">
            <div class="link-card-title">📖 Courses at University of Iowa</div>
            <div class="link-card-desc">Hydrogeophysics, Environmental Geophysics, Data Analysis in Geosciences</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("""
        <a href="https://sites.google.com/view/hangchen/teaching" target="_blank" class="link-card">
            <div class="link-card-title">🎓 Student Mentoring</div>
            <div class="link-card-desc">Graduate and undergraduate research opportunities available</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("[📚 **View Teaching Page →**](https://sites.google.com/view/hangchen/teaching)")

    # --- Open Source Tab ---
    with sub_tab6:
        st.markdown("### Open Source Projects")
        st.markdown("*For all codes, visit [Open Source Codes](https://sites.google.com/view/hangchen/open-source-codes)*")

        st.markdown("""
        <a href="https://github.com/geohang/PyHydroGeophysX" target="_blank" class="link-card">
            <div class="link-card-title">🐍 PyHydroGeophysX</div>
            <div class="link-card-desc">AI-powered hydrogeophysics workflow platform - ERT, Seismic, TDEM inversion with LLM agents</div>
        </a>
        """, unsafe_allow_html=True)

        st.markdown("[💻 **View All Open Source Projects →**](https://sites.google.com/view/hangchen/open-source-codes)")

        st.markdown("---")
        st.markdown("### Acknowledgments")
        st.markdown("""
        PyHydroGeophysX development is supported by:
        - University of Iowa
        - Lawrence Berkeley National Laboratory

        Special thanks to **ResIPy**, **PyGIMLi**, and **SimPEG** for their excellent geophysical libraries.
        """)


def render_local_deployment_tab() -> None:
    st.subheader("Local Deployment")
    st.markdown(
        """
# PyHydroGeophysX - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 0: Download the GitHub Repository
```bash
git clone https://github.com/geohang/PyHydroGeophysX.git
cd PyHydroGeophysX
```

### Step 1: Launch the Web App
```bash
cd examples
streamlit run app_geophysics_workflow.py
```
Or use the launcher scripts:
- **Windows**: `start_webapp.bat`
- **Linux/Mac**: `./start_webapp.sh`

### Step 2: Configure API Key
In the sidebar:
1. Select LLM provider (OpenAI recommended)
2. Enter your API key
3. Click "🚀 Initialize System"

### Step 3: Run Your First Workflow
1. Choose a workflow example or describe your data
2. Upload files if needed
3. Click **Run workflow** and review the report outputs
"""
    )


def render_workflow_tab(sidebar_state: Dict[str, str]) -> None:
    st.markdown("---")
    st.info(
        "Cloud resources are limited. For big datasets, use the Local Deployment tab so you can run the same web interface with local compute."
    )
    st.subheader("Describe your workflow")
    request_text = st.text_area(
        "Describe what you want to do (files, parameters, outputs)",
        value=st.session_state.user_request,
        height=180,
        placeholder="Example: Run a time-lapse ERT inversion on four surveys...",
    )
    st.session_state.user_request = request_text

    render_example_buttons()

    if not st.session_state.context_agent:
        st.info("LLM is not initialized. You can still run with the No-LLM quick buttons beside upload.")

    st.markdown("---")
    col_upload, col_quick = st.columns([2, 1])
    with col_upload:
        st.subheader("Upload data (optional)")
        st.caption("Upload files here; the app maps them by filename. This panel is intentionally compact.")
        uploaded_files = st.file_uploader(
            "Data files",
            accept_multiple_files=True,
            type=["ohm", "dat", "data", "txt", "sgy", "segy", "pfb", "nam"],
            help="Single upload area for ERT, seismic, electrodes, etc.",
            label_visibility="collapsed",
        )

    with col_quick:
        st.subheader("No-LLM Quick")
        q0, q1, q2, q3 = st.columns(4)
        if q0.button("Auto", key="quick_mode_auto", width="stretch"):
            st.session_state.quick_run_mode = "Auto (LLM)"
        if q1.button("ERT", key="quick_mode_ert", width="stretch"):
            st.session_state.quick_run_mode = "ERT Only"
        if q2.button("TL", key="quick_mode_tl", width="stretch"):
            st.session_state.quick_run_mode = "Time-Lapse ERT"
        if q3.button("SRT", key="quick_mode_srt", width="stretch"):
            st.session_state.quick_run_mode = "Seismic SRT"

        st.caption(f"Mode: `{st.session_state.quick_run_mode}`")
        st.caption(_quick_mode_description(st.session_state.quick_run_mode))

    st.markdown("---")
    run_clicked = st.button("Run workflow", type="primary", width="stretch")

    if run_clicked:
        quick_mode = st.session_state.quick_run_mode
        if quick_mode == "Auto (LLM)":
            if not request_text.strip():
                st.error("Please describe your workflow.")
                return
            if not st.session_state.context_agent:
                st.error("Initialize the system in the sidebar, or switch to a No-LLM quick mode.")
                return

        output_path = Path(sidebar_state["output_dir"]).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        # Parse uploads
        upload_overrides: Dict[str, Any] = {}
        saved_paths = handle_uploads(output_path, uploaded_files, upload_overrides)

        # Merge uploaded workflow overrides into the text-derived config during run_workflow
        st.session_state.workflow_config = upload_overrides

        if quick_mode == "Auto (LLM)":
            run_workflow(request_text, upload_overrides, saved_paths, output_path)
        else:
            try:
                quick_cfg = _build_no_llm_workflow_config(
                    quick_mode=quick_mode,
                    upload_overrides=upload_overrides,
                    user_request=request_text,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Quick mode configuration error: {exc}")
                return

            run_workflow(
                request_text or f"Quick mode run: {quick_mode}",
                upload_overrides,
                saved_paths,
                output_path,
                direct_config=quick_cfg,
            )

    if st.session_state.workflow_result:
        st.markdown("---")
        render_results()


def _resolve_user_path(path_text: str) -> Path:
    raw = (path_text or "").strip().strip('"').strip("'")
    if not raw:
        return CURRENT_DIR

    path_obj = Path(raw).expanduser()
    candidates: List[Path] = []
    if path_obj.is_absolute():
        candidates.append(path_obj)
    else:
        candidates.extend([Path.cwd() / path_obj, CURRENT_DIR / path_obj, PARENT_DIR / path_obj])

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return candidates[0].resolve() if candidates else path_obj.resolve()


def _inspect_hydro_input_dir(data_dir: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(data_dir),
        "ok": False,
        "missing": [],
        "snapshot_count": None,
        "water_shape": None,
        "porosity_shape": None,
        "bot_shape": None,
        "error": "",
    }

    required = ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"]
    missing = [name for name in required if not (data_dir / name).exists()]
    info["missing"] = missing
    if missing:
        return info

    try:
        import numpy as np

        water = np.load(data_dir / "Watercontent.npy", mmap_mode="r")
        porosity = np.load(data_dir / "Porosity.npy", mmap_mode="r")
        bot = np.load(data_dir / "bot.npy", mmap_mode="r")

        info["water_shape"] = tuple(int(v) for v in water.shape)
        info["porosity_shape"] = tuple(int(v) for v in porosity.shape)
        info["bot_shape"] = tuple(int(v) for v in bot.shape)
        info["snapshot_count"] = int(water.shape[0]) if water.ndim >= 1 else 0
        info["ok"] = bool(water.ndim >= 3 and porosity.ndim >= 3 and bot.ndim >= 2)
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)

    return info


def _has_hydro_required_files(path: Path) -> bool:
    """Check if *path* contains the standard pre-processed .npy files."""
    required = ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"]
    return path.exists() and path.is_dir() and all((path / name).exists() for name in required)


def _detect_hydro_data_format(path: Path) -> str:
    """Auto-detect the hydro data format inside *path*.

    Returns one of ``"npy"``, ``"modflow"``, ``"parflow"``, or ``"unknown"``.
    """
    if not path.exists() or not path.is_dir():
        return "unknown"

    # Pre-processed .npy check (highest priority – explicit standard format)
    if _has_hydro_required_files(path):
        return "npy"

    entries = {e.name for e in path.iterdir()}

    # MODFLOW indicators
    if "WaterContent" in entries or "mfsim.nam" in entries:
        return "modflow"

    # ParFlow indicators – look for *.pfb files
    pfb_files = list(path.glob("*.pfb"))
    if pfb_files:
        return "parflow"

    # Check immediate sub-directories (e.g. data/modflow/, data/parflow/test2/)
    for child in path.iterdir():
        if child.is_dir():
            sub_entries = {e.name for e in child.iterdir()}
            if "WaterContent" in sub_entries or "mfsim.nam" in sub_entries:
                return "modflow"
            if list(child.glob("*.pfb")):
                return "parflow"

    return "unknown"


def _detect_modflow_info(path: Path) -> Dict[str, Any]:
    """Scan a MODFLOW directory and return auto-detected settings."""
    info: Dict[str, Any] = {
        "has_watercontent": (path / "WaterContent").exists(),
        "has_mfsim": (path / "mfsim.nam").exists(),
        "idomain_file": "",
        "model_name": "",
    }
    # Auto-detect idomain / id file
    for candidate in ["id.txt", "idomain.txt", "idomain.npy"]:
        if (path / candidate).exists():
            info["idomain_file"] = candidate
            break
    # Auto-detect model name from .nam files (exclude mfsim.nam)
    for f in path.iterdir():
        if f.suffix == ".nam" and f.name != "mfsim.nam":
            info["model_name"] = f.stem
            break
    # Fallback: look for .dis or other common extensions
    if not info["model_name"]:
        for f in path.iterdir():
            if f.suffix in (".dis", ".dis6"):
                info["model_name"] = f.stem
                break
    return info


def _detect_parflow_info(path: Path) -> Dict[str, Any]:
    """Scan a ParFlow directory and return auto-detected settings."""
    info: Dict[str, Any] = {
        "run_name": "",
        "has_porosity": False,
        "has_mask": False,
        "num_satur_files": 0,
    }
    pfb_files = sorted(path.glob("*.pfb"))
    # Infer run_name from saturation files
    for f in pfb_files:
        if ".out.satur." in f.name:
            info["run_name"] = f.name.split(".out.satur.")[0]
            break
    if not info["run_name"]:
        # Try any pfb
        for f in pfb_files:
            info["run_name"] = f.stem.split(".")[0]
            break
    rn = info["run_name"]
    info["has_porosity"] = any(
        (path / f"{rn}{suf}").exists()
        for suf in [".out.porosity.pfb", ".porosity.pfb", ".pf.porosity.pfb"]
    )
    info["has_mask"] = any(
        (path / f"{rn}{suf}").exists()
        for suf in [".out.mask.pfb", ".mask.pfb", ".pf.mask.pfb"]
    )
    info["num_satur_files"] = len(list(path.glob(f"{rn}.out.satur.*.pfb")))
    return info


def _convert_modflow_to_npy(
    modflow_dir: str,
    idomain_file: str,
    model_name: str,
    nlay: int = 3,
    out_dir: Optional[str] = None,
) -> str:
    """Load MODFLOW outputs and write standard .npy/.txt files.

    Returns the directory containing the converted files.
    """
    import numpy as np
    from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent

    modflow_path = Path(modflow_dir)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="phgx_mf_")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load idomain
    id_path = modflow_path / idomain_file
    if id_path.suffix == ".npy":
        idomain = np.load(str(id_path))
    else:
        idomain = np.loadtxt(str(id_path))

    # Water content → 4D (nt, nlay, nrows, ncols)
    wc_proc = MODFLOWWaterContent(model_directory=str(modflow_path), idomain=idomain)
    water_content = wc_proc.load_time_range(start_idx=0, end_idx=None, nlay=nlay)

    # Porosity → 3D (nlay, nrows, ncols)
    # flopy may not be installed; handle gracefully
    porosity = None
    try:
        from PyHydroGeophysX.model_output.modflow_output import MODFLOWPorosity
        por_proc = MODFLOWPorosity(model_directory=str(modflow_path), model_name=model_name)
        porosity = por_proc.load_porosity()
    except ImportError:
        pass  # flopy not available

    if porosity is None:
        # Fallback: use a uniform default porosity of 0.3
        _, n_lay, n_rows, n_cols = water_content.shape
        porosity = np.full((n_lay, n_rows, n_cols), 0.3)

    # Generate simple top / bot from layer indices (metres from 0)
    _, n_lay, n_rows, n_cols = water_content.shape
    top = np.zeros((n_rows, n_cols))  # surface at 0
    bot = np.zeros((n_lay, n_rows, n_cols))
    for k in range(n_lay):
        bot[k, :, :] = -(k + 1)  # each layer 1 m thick going downward

    np.save(str(out_path / "Watercontent.npy"), water_content)
    np.save(str(out_path / "Porosity.npy"), porosity)
    np.savetxt(str(out_path / "top.txt"), top)
    np.save(str(out_path / "bot.npy"), bot)

    return str(out_path)


def _convert_parflow_to_npy(
    parflow_dir: str,
    run_name: str,
    out_dir: Optional[str] = None,
) -> str:
    """Load ParFlow outputs and write standard .npy/.txt files.

    Returns the directory containing the converted files.
    """
    import numpy as np
    from PyHydroGeophysX.model_output.parflow_output import ParflowSaturation, ParflowPorosity

    pf_path = Path(parflow_dir)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="phgx_pf_")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Saturation → 4D (nt, nz, ny, nx)  — treated as water content proxy
    sat_proc = ParflowSaturation(model_directory=str(pf_path), run_name=run_name)
    saturation = sat_proc.load_time_range(start_idx=0, end_idx=None)

    # Porosity → 3D (nz, ny, nx)
    por_proc = ParflowPorosity(model_directory=str(pf_path), run_name=run_name)
    porosity = por_proc.load_porosity()

    # Mask for inactive cells
    try:
        mask = por_proc.load_mask()
        porosity[mask == 0] = np.nan
        for t in range(saturation.shape[0]):
            saturation[t][mask == 0] = np.nan
    except FileNotFoundError:
        pass  # no mask file, proceed without

    # Water content = saturation × porosity
    water_content = saturation * porosity[np.newaxis, :, :, :]

    # Top/bot: uniform layers based on grid shape
    _, n_lay, n_rows, n_cols = water_content.shape
    top = np.zeros((n_rows, n_cols))
    bot = np.zeros((n_lay, n_rows, n_cols))
    for k in range(n_lay):
        bot[k, :, :] = -(k + 1)

    np.save(str(out_path / "Watercontent.npy"), water_content)
    np.save(str(out_path / "Porosity.npy"), porosity)
    np.savetxt(str(out_path / "top.txt"), top)
    np.save(str(out_path / "bot.npy"), bot)

    return str(out_path)


def _discover_hydro_data_dirs(current_value: str) -> List[Path]:
    """Discover directories that contain *any* supported hydro data format."""
    seeds: List[Path] = [
        _resolve_user_path(current_value),
        CURRENT_DIR / "data",
        PARENT_DIR / "examples" / "data",
        PARENT_DIR / "data",
        Path.cwd() / "data",
        Path.cwd(),
    ]

    candidates: List[Path] = []
    seen = set()

    def _is_hydro_dir(p: Path) -> bool:
        """Return True if *p* contains any recognised hydro data format."""
        if _has_hydro_required_files(p):
            return True
        fmt = _detect_hydro_data_format(p)
        return fmt in ("modflow", "parflow")

    def _add_candidate(path_obj: Path) -> None:
        try:
            resolved = path_obj.resolve()
        except Exception:  # noqa: BLE001
            return
        key = str(resolved).lower()
        if key in seen:
            return
        seen.add(key)
        if _is_hydro_dir(resolved):
            candidates.append(resolved)

    for seed in seeds:
        _add_candidate(seed)
        parent = seed.parent if seed.parent.exists() else None
        if parent and parent.is_dir():
            try:
                for child in parent.iterdir():
                    if child.is_dir():
                        _add_candidate(child)
            except Exception:  # noqa: BLE001
                pass

    return candidates


def _extract_plotly_selected_points(
    event_data: Any,
    x_lookup: Optional[List[float]] = None,
    y_lookup: Optional[List[float]] = None,
) -> List[List[float]]:
    if event_data is None:
        return []

    payload: Any = event_data
    if hasattr(payload, "selection"):
        payload = getattr(payload, "selection")

    points_raw = None
    if isinstance(payload, dict):
        if isinstance(payload.get("selection"), dict):
            points_raw = payload.get("selection", {}).get("points")
        if points_raw is None:
            points_raw = payload.get("points")
    elif hasattr(payload, "points"):
        points_raw = getattr(payload, "points")
    elif hasattr(payload, "get"):
        try:
            points_raw = payload.get("points")
        except Exception:  # noqa: BLE001
            points_raw = None

    def _resolve_from_index(idx_val: Any) -> Optional[List[float]]:
        if idx_val is None or x_lookup is None or y_lookup is None:
            return None
        try:
            idx = int(idx_val)
        except Exception:  # noqa: BLE001
            return None
        if idx < 0 or idx >= len(x_lookup) or idx >= len(y_lookup):
            return None
        try:
            return [float(x_lookup[idx]), float(y_lookup[idx])]
        except Exception:  # noqa: BLE001
            return None

    if not isinstance(points_raw, list):
        # Some payloads may only provide point indices
        if isinstance(payload, dict):
            idxs = payload.get("point_indices")
            if not isinstance(idxs, list) and isinstance(payload.get("selection"), dict):
                idxs = payload.get("selection", {}).get("point_indices")
            if isinstance(idxs, list):
                out: List[List[float]] = []
                for idx_val in idxs:
                    mapped = _resolve_from_index(idx_val)
                    if mapped is not None:
                        out.append(mapped)
                return out
        return []

    selected: List[List[float]] = []
    for point in points_raw:
        x_val = None
        y_val = None
        if isinstance(point, dict):
            x_val = point.get("x")
            y_val = point.get("y")
            if x_val is None or y_val is None:
                idx_val = point.get("point_index", point.get("pointNumber"))
                mapped = _resolve_from_index(idx_val)
                if mapped is not None:
                    selected.append(mapped)
                    continue
            # Some payload variants include a list of point indices per selection
            if (x_val is None or y_val is None) and isinstance(point.get("point_indices"), list):
                for idx_val in point.get("point_indices", []):
                    mapped = _resolve_from_index(idx_val)
                    if mapped is not None:
                        selected.append(mapped)
                continue
        else:
            if hasattr(point, "x"):
                x_val = getattr(point, "x")
            if hasattr(point, "y"):
                y_val = getattr(point, "y")
            if x_val is None or y_val is None:
                idx_val = getattr(point, "point_index", None)
                if idx_val is None:
                    idx_val = getattr(point, "pointNumber", None)
                mapped = _resolve_from_index(idx_val)
                if mapped is not None:
                    selected.append(mapped)
                    continue
        if x_val is None or y_val is None:
            continue
        try:
            selected.append([float(x_val), float(y_val)])
        except Exception:  # noqa: BLE001
            continue
    return selected


def _mask_anomalous_zero_elevation_for_plot(top_array):
    """
    For plotting only: if zero-elevation cells are strongly inconsistent with the
    non-zero elevation distribution, mask those zeros as NaN.
    """
    import numpy as np

    top = np.asarray(top_array, dtype=float).copy()
    finite_mask = np.isfinite(top)
    zero_mask = finite_mask & np.isclose(top, 0.0)
    if not np.any(zero_mask):
        return top, False, ""

    nonzero = top[finite_mask & ~np.isclose(top, 0.0)]
    if nonzero.size < 20:
        return top, False, ""

    median_nonzero = float(np.nanmedian(nonzero))
    mad = float(np.nanmedian(np.abs(nonzero - median_nonzero)))
    robust_sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(nonzero))
    robust_sigma = max(robust_sigma, 1.0)

    # If 0 is many robust sigmas away from the central elevation level,
    # treat zeros as invalid placeholders for visualization.
    delta = abs(median_nonzero)
    threshold = max(6.0 * robust_sigma, 50.0)
    if delta > threshold:
        converted_count = int(np.count_nonzero(zero_mask))
        top[zero_mask] = np.nan
        note = (
            f"Detected anomalous zero elevations for plotting "
            f"(median={median_nonzero:.2f}, threshold={threshold:.2f}). "
            f"Converted {converted_count} values to NaN in surface map."
        )
        return top, True, note

    return top, False, ""


def _render_surface_picker(data_dir: Path) -> None:
    st.markdown("### Surface Map and Profile Point Picking")

    try:
        import numpy as np
        top = np.loadtxt(data_dir / "top.txt")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load `top.txt` for surface map: {exc}")
        return

    if top.ndim != 2:
        st.warning(f"Expected 2D surface map in `top.txt`, got shape {top.shape}.")
        return

    top_plot, zero_masked, zero_note = _mask_anomalous_zero_elevation_for_plot(top)
    if zero_masked and zero_note:
        st.info(zero_note)

    # Pick counter: 0 → next pick = P1, 1 → next pick = P2
    if "hydro_pick_next" not in st.session_state:
        st.session_state.hydro_pick_next = 0

    # Track whether the user has explicitly picked each point (via map or manual input).
    # Default values are NOT shown on the map until the user confirms them.
    if "hydro_user_picked_p1" not in st.session_state:
        st.session_state.hydro_user_picked_p1 = False
    if "hydro_user_picked_p2" not in st.session_state:
        st.session_state.hydro_user_picked_p2 = False

    import numpy as np
    max_x = max(0, top.shape[1] - 1)
    max_y = max(0, top.shape[0] - 1)

    # Read current points — only treat as set if user explicitly picked them
    has_p1 = st.session_state.hydro_user_picked_p1
    has_p2 = st.session_state.hydro_user_picked_p2

    p1x = float(np.clip(st.session_state.get("hydro_point1_x", 0), 0, max_x))
    p1y = float(np.clip(st.session_state.get("hydro_point1_y", 0), 0, max_y))
    p2x = float(np.clip(st.session_state.get("hydro_point2_x", 0), 0, max_x))
    p2y = float(np.clip(st.session_state.get("hydro_point2_y", 0), 0, max_y))

    # Status message — guide the user step by step
    next_pick = st.session_state.hydro_pick_next
    if not has_p1 and not has_p2:
        st.info("🖱️ **Step 1/2:** Use *box select* or *lasso* on the map to pick **P1** (start of profile).")
    elif has_p1 and not has_p2:
        st.info(f"✅ P1 = ({p1x:.0f}, {p1y:.0f}).  🖱️ **Step 2/2:** Now pick **P2** (end of profile).")
    else:
        st.success(f"✅ Profile set:  **P1** ({p1x:.0f}, {p1y:.0f})  →  **P2** ({p2x:.0f}, {p2y:.0f})")

    # --- Build plotly figure ---
    picked_points: List[List[float]] = []
    render_backend = "unknown"
    try:
        import inspect
        import plotly.graph_objects as go
        try:
            from streamlit_plotly_events import plotly_events
            plotly_events_available = True
        except Exception:
            plotly_events_available = False

        y_idx = list(range(top.shape[0]))
        x_idx = list(range(top.shape[1]))
        finite_vals = top_plot[np.isfinite(top_plot)]
        zmin = None
        zmax = None
        if finite_vals.size > 0:
            nonzero_vals = finite_vals[~np.isclose(finite_vals, 0.0)]
            ref_vals = nonzero_vals if nonzero_vals.size >= 20 else finite_vals
            try:
                zmin = float(np.nanpercentile(ref_vals, 2.0))
                zmax = float(np.nanpercentile(ref_vals, 98.0))
                if (not np.isfinite(zmin)) or (not np.isfinite(zmax)) or (zmax <= zmin):
                    zmin = None
                    zmax = None
            except Exception:  # noqa: BLE001
                zmin = None
                zmax = None

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=top_plot, x=x_idx, y=y_idx,
                    colorscale="Viridis",
                    zmin=zmin,
                    zmax=zmax,
                    colorbar={"title": "Surface elev."},
                )
            ]
        )

        x_lookup: Optional[List[float]] = None
        y_lookup: Optional[List[float]] = None
        # Add invisible selection grid for native Streamlit on_select backend
        # (needs discrete points) and as fallback when plotly_events is absent.
        _use_native_on_select = "on_select" in inspect.signature(st.plotly_chart).parameters
        if _use_native_on_select or not plotly_events_available:
            yy, xx = np.indices(top_plot.shape)
            x_lookup = xx.ravel().astype(float).tolist()
            y_lookup = yy.ravel().astype(float).tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_lookup, y=y_lookup,
                    mode="markers",
                    marker={"size": 4, "color": "rgba(0,0,0,0)"},
                    hovertemplate="x=%{x}<br>y=%{y}<extra></extra>",
                    showlegend=False, name="grid",
                )
            )

        # Only draw line + markers for points that have been picked
        if has_p1 and has_p2:
            fig.add_trace(go.Scatter(
                x=[p1x, p2x], y=[p1y, p2y], mode="lines",
                line={"color": "#ffffff", "width": 2, "dash": "dash"},
                showlegend=False, hoverinfo="skip", name="Profile line",
            ))
            fig.add_trace(go.Scatter(
                x=[p1x, p2x], y=[p1y, p2y],
                mode="markers+text", text=["P1", "P2"],
                textposition="top center",
                textfont={"color": "white", "size": 13},
                marker={"size": 12, "color": ["#ef4444", "#0ea5e9"], "symbol": "x"},
                name="Profile points",
            ))
        elif has_p1:
            fig.add_trace(go.Scatter(
                x=[p1x], y=[p1y],
                mode="markers+text", text=["P1"],
                textposition="top center",
                textfont={"color": "white", "size": 13},
                marker={"size": 12, "color": "#ef4444", "symbol": "x"},
                name="P1",
            ))

        fig.update_layout(
            title="Surface map (top.txt)",
            xaxis_title="X index", yaxis_title="Y index",
            dragmode="select", clickmode="event+select",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
            height=460,
        )
        # Keep full-map extent stable across reruns; prevents zoom-lock on picked points.
        fig.update_xaxes(range=[-0.5, max_x + 0.5])
        fig.update_yaxes(range=[max_y + 0.5, -0.5], autorange=False)

        # Render chart + capture selection
        # Prefer native Streamlit on_select (Streamlit ≥ 1.35) over the
        # third-party streamlit-plotly-events which is broken with
        # Streamlit ≥ 1.37 / Plotly ≥ 6.x (renders a blank iframe).
        if "on_select" in inspect.signature(st.plotly_chart).parameters:
            render_backend = "streamlit_plotly_on_select"
            event = st.plotly_chart(
                fig, use_container_width=True,
                key="hydro_surface_picker_plotly",
                on_select="rerun",
                # Box/lasso/points selection works reliably in native Streamlit Plotly.
                selection_mode=("box", "lasso", "points"),
            )
            st.caption("Tip: use box/lasso (or points mode) on the map to pick P1/P2.")
            picked_points = _extract_plotly_selected_points(event, x_lookup=x_lookup, y_lookup=y_lookup)
        elif plotly_events_available:
            render_backend = "streamlit_plotly_events"
            events = plotly_events(
                fig, click_event=True, select_event=True,
                hover_event=False, override_height=460,
                key="hydro_surface_picker_events",
            )
            if isinstance(events, list):
                for pt in events:
                    if isinstance(pt, dict) and pt.get("x") is not None and pt.get("y") is not None:
                        picked_points.append([float(pt["x"]), float(pt["y"])])
            st.caption("Tip: click on map to pick points (P1 then P2).")
        else:
            render_backend = "streamlit_plotly_static"
            st.plotly_chart(fig, use_container_width=True, key="hydro_surface_picker_plotly_static")
            st.info("Interactive point selection requires Streamlit >= 1.35 or `streamlit-plotly-events`.")

    except Exception:
        render_backend = "matplotlib_fallback"
        import matplotlib.pyplot as plt

        mpl_fig, ax = plt.subplots(figsize=(8.5, 4.6))
        im = ax.imshow(np.ma.masked_invalid(top_plot), cmap="viridis", origin="upper", aspect="auto")
        if has_p1 and has_p2:
            ax.plot([p1x, p2x], [p1y, p2y], color="white", linewidth=1.5, linestyle="--")
            ax.scatter([p1x, p2x], [p1y, p2y], c=["#ef4444", "#0ea5e9"], marker="x", s=80, zorder=5)
            ax.annotate("P1", (p1x, p1y), textcoords="offset points", xytext=(0, -12), ha="center", color="white", fontweight="bold")
            ax.annotate("P2", (p2x, p2y), textcoords="offset points", xytext=(0, -12), ha="center", color="white", fontweight="bold")
        elif has_p1:
            ax.scatter([p1x], [p1y], c=["#ef4444"], marker="x", s=80, zorder=5)
            ax.annotate("P1", (p1x, p1y), textcoords="offset points", xytext=(0, -12), ha="center", color="white", fontweight="bold")
        ax.set_title("Surface map (top.txt)")
        ax.set_xlabel("X index"); ax.set_ylabel("Y index")
        plt.colorbar(im, ax=ax, label="Surface elev.")
        st.pyplot(mpl_fig, use_container_width=True)
        plt.close(mpl_fig)
        st.info("Install `plotly` for interactive point picking.")

    # --- Process picked points: P1 then P2 ---
    if picked_points:
        xs = [p[0] for p in picked_points]
        ys = [p[1] for p in picked_points]
        new_x = float(round(np.median(xs)))
        new_y = float(round(np.median(ys)))

        if st.session_state.hydro_pick_next == 0:
            st.session_state.hydro_point1_x = new_x
            st.session_state.hydro_point1_y = new_y
            st.session_state.hydro_user_picked_p1 = True
            st.session_state.hydro_pick_next = 1
        else:
            st.session_state.hydro_point2_x = new_x
            st.session_state.hydro_point2_y = new_y
            st.session_state.hydro_user_picked_p2 = True
            st.session_state.hydro_pick_next = 0
        st.rerun()

    # Clear button — only shown after user has picked at least one point
    if has_p1 or has_p2:
        if st.button("🗑️ Clear picks & restart", key="hydro_clear_picked_points"):
            st.session_state.hydro_pick_next = 0
            st.session_state.hydro_user_picked_p1 = False
            st.session_state.hydro_user_picked_p2 = False
            # Reset coordinates to defaults but keep them hidden until re-picked
            for key in ("hydro_point1_x", "hydro_point1_y", "hydro_point2_x", "hydro_point2_y"):
                st.session_state.pop(key, None)
            st.rerun()

def _fill_profile_nans(values):
    import numpy as np

    arr = np.asarray(values, dtype=float).copy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}.")

    x = np.arange(arr.shape[1], dtype=float)
    valid_row_idx = []

    for i in range(arr.shape[0]):
        row = arr[i, :]
        valid = np.isfinite(row)
        if np.any(valid):
            if np.count_nonzero(valid) == 1:
                row[~valid] = row[valid][0]
            else:
                row[~valid] = np.interp(x[~valid], x[valid], row[valid])
            valid_row_idx.append(i)
        else:
            # Defer fully empty rows and fill them after we process valid rows.
            continue
        arr[i, :] = row

    if not valid_row_idx:
        raise RuntimeError("Profile interpolation failed: all layers are NaN along this profile.")

    valid_row_idx = np.asarray(valid_row_idx, dtype=int)
    for i in range(arr.shape[0]):
        if np.any(np.isfinite(arr[i, :])):
            continue

        lower = valid_row_idx[valid_row_idx < i]
        upper = valid_row_idx[valid_row_idx > i]

        if lower.size and upper.size:
            lo = int(lower[-1])
            hi = int(upper[0])
            weight = (i - lo) / float(hi - lo)
            arr[i, :] = (1.0 - weight) * arr[lo, :] + weight * arr[hi, :]
        elif lower.size:
            arr[i, :] = arr[int(lower[-1]), :]
        else:
            arr[i, :] = arr[int(upper[0]), :]

    return arr


def _get_mesh_xy(mesh):
    import numpy as np

    centers = np.asarray(mesh.cellCenters(), dtype=float)
    if centers.ndim == 2 and centers.shape[1] >= 2:
        return centers[:, 0], centers[:, 1]

    x = np.array([float(c[0]) for c in mesh.cellCenters()], dtype=float)
    y = np.array([float(c[1]) for c in mesh.cellCenters()], dtype=float)
    return x, y


def _assign_three_layer_markers(mesh, line1, line2, top_marker=0, mid_marker=3, bot_marker=2):
    import numpy as np

    x_cell, y_cell = _get_mesh_xy(mesh)
    y_line1 = np.interp(x_cell, line1[:, 0], line1[:, 1])
    y_line2 = np.interp(x_cell, line2[:, 0], line2[:, 1])

    markers = np.full(mesh.cellCount(), bot_marker, dtype=int)
    markers[y_cell >= y_line2] = mid_marker
    markers[y_cell >= y_line1] = top_marker
    mesh.setCellMarkers(markers)
    return markers


def _interpolate_profile_to_mesh(profile_values, layer_boundaries, x_profile, mesh):
    import numpy as np
    from scipy.interpolate import griddata

    values = np.asarray(profile_values, dtype=float)
    bounds = np.asarray(layer_boundaries, dtype=float)

    n_layers, n_profile = values.shape
    if bounds.shape != (n_layers + 1, n_profile):
        raise ValueError(
            f"layer_boundaries shape must be {(n_layers + 1, n_profile)}, got {bounds.shape}."
        )

    layer_centers = 0.5 * (bounds[:-1, :] + bounds[1:, :])
    x2d = np.repeat(np.asarray(x_profile, dtype=float)[np.newaxis, :], n_layers, axis=0)

    points = np.column_stack((x2d.ravel(), layer_centers.ravel()))
    vals = values.ravel()

    x_cell, y_cell = _get_mesh_xy(mesh)
    query = np.column_stack((x_cell, y_cell))

    interp_linear = griddata(points, vals, query, method="linear")
    interp_nearest = griddata(points, vals, query, method="nearest")
    out = np.asarray(interp_linear, dtype=float)
    nan_mask = ~np.isfinite(out)
    out[nan_mask] = interp_nearest[nan_mask]
    return out


def _relative_l2(noisy, clean) -> float:
    import numpy as np

    noisy_arr = np.asarray(noisy)
    clean_arr = np.asarray(clean)
    denom = np.linalg.norm(clean_arr)
    if denom <= 0:
        return float("nan")
    return float(np.linalg.norm(noisy_arr - clean_arr) / denom)


def _ordered_unique_methods(methods: List[str]) -> List[str]:
    selected: List[str] = []
    for method in HYDRO_RESPONSE_METHODS:
        if method in methods and method not in selected:
            selected.append(method)
    return selected


def _quick_mode_description(mode: str) -> str:
    mapping = {
        "Auto (LLM)": "Use natural language parsing via ContextInputAgent.",
        "ERT Only": "Run direct ERT inversion from one uploaded ERT file.",
        "Time-Lapse ERT": "Run time-lapse inversion from multiple uploaded ERT files.",
        "Seismic SRT": "Run seismic refraction inversion from one seismic file.",
    }
    return mapping.get(mode, mode)


def _infer_instrument_from_request(user_request: str) -> Optional[str]:
    """
    Infer likely instrument from free-text user request for No-LLM quick mode.
    """
    text = (user_request or "").strip().lower()
    if not text:
        return None

    if "abem" in text or "terameter" in text:
        return "ABEM-Lund"
    if "syscal" in text:
        return "Syscal"
    if re.search(r"\be4d\b", text):
        return "E4D"
    if re.search(r"\bdas\b", text) or "das-1" in text or "das 1" in text:
        return "DAS-1"
    if "bert" in text:
        return "BERT"
    if "sting" in text:
        return "Sting"
    if "ares" in text:
        return "ARES"
    if "protocol dc" in text:
        return "Protocol DC"
    return None


def _build_no_llm_workflow_config(
    quick_mode: str,
    upload_overrides: Dict[str, Any],
    user_request: str,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = dict(upload_overrides)
    cfg["user_request"] = user_request.strip() or f"Quick run mode: {quick_mode}"
    cfg["project_dir"] = str(Path.cwd())
    inferred_instrument = _infer_instrument_from_request(user_request)

    if quick_mode == "ERT Only":
        ert_file = cfg.get("ert_file") or cfg.get("data_file")
        if not ert_file:
            raise ValueError("No ERT file found. Upload one `.ohm/.data/.dat/.txt` ERT file.")
        cfg.update(
            {
                "ert_file": ert_file,
                "data_file": ert_file,
                "instrument": cfg.get("instrument") or inferred_instrument or "DAS-1",
                "convert_to_water_content": False,
                "inversion_params": {"lambda": 20.0, "max_iterations": 12, "method": "cgls"},
            }
        )

    elif quick_mode == "Time-Lapse ERT":
        tl_files = cfg.get("time_lapse_files") or cfg.get("timelapse_files")
        if not tl_files or len(tl_files) < 2:
            raise ValueError("Time-lapse mode requires at least two uploaded ERT files.")
        cfg.update(
            {
                "time_lapse_files": list(tl_files),
                "timelapse_files": list(tl_files),
                "instrument": cfg.get("instrument") or inferred_instrument or "E4D",
                "inversion_mode": "time-lapse",
                "time_lapse_method": "difference",
                "temporal_regularization": 10.0,
                "inversion_params": {"lambda": 15.0, "max_iterations": 10, "method": "cgls"},
            }
        )

    elif quick_mode == "Seismic SRT":
        seismic_file = cfg.get("seismic_file")
        if not seismic_file:
            raise ValueError("No seismic file found. Upload one seismic `.dat/.sgy/.segy` file.")
        cfg.update(
            {
                "seismic_file": seismic_file,
                "seismic_only": True,
                "extract_interfaces": True,
                "velocity_threshold": 1200,
                "inversion_params": {
                    "lam": 50,
                    "zWeight": 0.2,
                    "vTop": 500,
                    "vBottom": 5000,
                    "paraDepth": 30.0,
                    "limits": [300.0, 8000.0],
                },
            }
        )
    else:
        raise ValueError(f"Unsupported quick mode: {quick_mode}")

    return cfg


def _parse_hydro_command_rule_based(command: str) -> Dict[str, Any]:
    text = command.strip()
    lower = text.lower()
    updates: Dict[str, Any] = {}

    alias_map = {
        "profile": "Profile",
        "hydro profile": "Profile",
        "ert": "ERT",
        "resistivity": "ERT",
        "srt": "SRT",
        "seismic": "SRT",
        "tdem": "TDEM",
        "tem ": "TDEM",
        "fdem": "FDEM",
        "frequency-domain": "FDEM",
        "gravity": "Gravity",
    }

    mentioned: List[str] = []
    for alias, target in alias_map.items():
        if alias in lower:
            mentioned.append(target)

    if "all methods" in lower or "all responses" in lower or re.search(r"\ball\b", lower):
        mentioned = HYDRO_RESPONSE_METHODS.copy()

    excluded: List[str] = []
    for alias, target in alias_map.items():
        if re.search(rf"(without|except|exclude)\s+[^\n]*{re.escape(alias.strip())}", lower):
            excluded.append(target)

    selected_methods = [m for m in _ordered_unique_methods(mentioned) if m not in set(excluded)]
    if selected_methods:
        updates["hydro_methods"] = selected_methods

    m_snapshot = re.search(r"(snapshot|time\s*step|timestep|index)\s*[:=]?\s*(-?\d+)", lower)
    if m_snapshot:
        updates["hydro_snapshot_index"] = max(0, int(m_snapshot.group(2)))

    m_num_points = re.search(r"(profile\s*points|num\s*points|points)\s*[:=]?\s*(\d+)", lower)
    if m_num_points:
        updates["hydro_num_points"] = max(50, int(m_num_points.group(2)))

    m_station_count = re.search(r"(stations|station\s*count|soundings)\s*[:=]?\s*(\d+)", lower)
    if m_station_count:
        updates["hydro_station_count"] = max(4, int(m_station_count.group(2)))

    m_ert_scheme = re.search(
        r"(ert\s*(array|scheme)|array|scheme)\s*[:=]?\s*(wa|dd|wenner(?:-alpha)?|dipole-?dipole)",
        lower,
    )
    if m_ert_scheme:
        token = m_ert_scheme.group(3)
        updates["hydro_ert_scheme"] = "dd" if ("dd" in token or "dipole" in token) else "wa"

    m_ert_count = re.search(r"(ert\s*(electrode\s*)?count|num\s*electrodes)\s*[:=]?\s*(\d+)", lower)
    if m_ert_count:
        updates["hydro_ert_num_electrodes"] = max(4, int(m_ert_count.group(3)))

    m_ert_spacing = re.search(r"(ert\s*(electrode\s*)?spacing)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", lower)
    if m_ert_spacing:
        updates["hydro_ert_electrode_spacing"] = max(0.2, float(m_ert_spacing.group(3)))

    m_ert_start = re.search(r"(ert\s*(line\s*)?start|ert\s*start)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", lower)
    if m_ert_start:
        updates["hydro_ert_electrode_start"] = float(m_ert_start.group(3))

    m_srt_count = re.search(r"(srt\s*(sensor|receiver|geophone)\s*count|num\s*(sensors|geophones))\s*[:=]?\s*(\d+)", lower)
    if m_srt_count:
        updates["hydro_srt_num_sensors"] = max(4, int(m_srt_count.group(4)))

    m_srt_spacing = re.search(r"(srt\s*(sensor|receiver|geophone)\s*spacing)\s*[:=]?\s*(-?\d+(?:\.\d+)?)", lower)
    if m_srt_spacing:
        updates["hydro_srt_sensor_spacing"] = max(0.2, float(m_srt_spacing.group(3)))

    m_srt_start = re.search(
        r"(srt\s*(source|shot|line)\s*(start|location|x)|source\s*start)\s*[:=]?\s*(-?\d+(?:\.\d+)?)",
        lower,
    )
    if m_srt_start:
        updates["hydro_srt_sensor_start"] = float(m_srt_start.group(4))

    m_shot_distance = re.search(r"(srt\s*shot\s*distance|shot\s*distance)\s*[:=]?\s*(\d+)", lower)
    if m_shot_distance:
        updates["hydro_srt_shot_distance"] = max(1, int(m_shot_distance.group(2)))

    point_patterns = [
        ("point1", "hydro_point1_x", "hydro_point1_y"),
        ("point 1", "hydro_point1_x", "hydro_point1_y"),
        ("point2", "hydro_point2_x", "hydro_point2_y"),
        ("point 2", "hydro_point2_x", "hydro_point2_y"),
    ]
    for marker, x_key, y_key in point_patterns:
        m_point = re.search(
            rf"{re.escape(marker)}\s*[:=]?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)",
            lower,
        )
        if m_point:
            updates[x_key] = float(m_point.group(1))
            updates[y_key] = float(m_point.group(2))

    m_data = re.search(r"(data\s*(dir|directory|folder|path)|input\s*dir)\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m_data:
        updates["hydro_data_dir"] = m_data.group(3).strip().strip('"').strip("'")

    m_output = re.search(r"(output\s*(dir|directory|folder|path))\s*[:=]\s*([^\n]+)", text, re.IGNORECASE)
    if m_output:
        updates["hydro_output_dir"] = m_output.group(3).strip().strip('"').strip("'")

    return updates


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    candidate = text.strip()
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    start = 0
    while start < len(text):
        open_idx = text.find("{", start)
        if open_idx < 0:
            break

        depth = 0
        for idx in range(open_idx, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    block = text[open_idx : idx + 1]
                    try:
                        obj = json.loads(block)
                        return obj if isinstance(obj, dict) else None
                    except Exception:
                        break
        start = open_idx + 1

    return None


def _parse_hydro_command_llm(command: str):
    if not st.session_state.context_agent:
        return {}, ""

    parser_prompt = f"""
You parse control messages for a hydrogeophysics app.
Return ONLY JSON with optional keys:
- methods: array of values from ["Profile","ERT","SRT","TDEM","FDEM","Gravity"]
- snapshot_index: integer >= 0
- point1: [x, y]
- point2: [x, y]
- num_points: integer >= 50
- station_count: integer >= 4
- ert_scheme: "wa" or "dd"
- ert_num_electrodes: integer >= 4
- ert_electrode_spacing: float > 0
- ert_electrode_start: float
- srt_num_sensors: integer >= 4
- srt_sensor_spacing: float > 0
- srt_source_start: float
- srt_shot_distance: integer >= 1
- rho_sat: [top, mid, bot]
- archie_n: [top, mid, bot]
- sigma_s: [top, mid, bot]
- top_bulk_modulus: float > 0
- top_shear_modulus: float > 0
- top_mineral_density: float > 0
- top_depth: float > 0
- mid_bulk_modulus: float > 0
- mid_shear_modulus: float > 0
- mid_mineral_density: float > 0
- mid_aspect_ratio: float in (0,1]
- bot_bulk_modulus: float > 0
- bot_shear_modulus: float > 0
- bot_mineral_density: float > 0
- bot_aspect_ratio: float in (0,1]
- data_dir: string
- output_dir: string

If a field is not provided by user intent, omit it.
User message:
{command}
"""
    response = st.session_state.context_agent.query_llm(parser_prompt)
    parsed = _extract_json_object(response)
    if not parsed:
        return {}, response

    updates: Dict[str, Any] = {}

    methods = parsed.get("methods")
    if isinstance(methods, list):
        cleaned = [str(m).strip() for m in methods]
        chosen = _ordered_unique_methods(cleaned)
        if chosen:
            updates["hydro_methods"] = chosen

    if parsed.get("snapshot_index") is not None:
        updates["hydro_snapshot_index"] = max(0, int(parsed["snapshot_index"]))

    if parsed.get("num_points") is not None:
        updates["hydro_num_points"] = max(50, int(parsed["num_points"]))

    if parsed.get("station_count") is not None:
        updates["hydro_station_count"] = max(4, int(parsed["station_count"]))

    if parsed.get("ert_scheme"):
        token = str(parsed["ert_scheme"]).strip().lower()
        updates["hydro_ert_scheme"] = "dd" if token.startswith("d") else "wa"

    if parsed.get("ert_num_electrodes") is not None:
        updates["hydro_ert_num_electrodes"] = max(4, int(parsed["ert_num_electrodes"]))

    if parsed.get("ert_electrode_spacing") is not None:
        updates["hydro_ert_electrode_spacing"] = max(0.2, float(parsed["ert_electrode_spacing"]))

    if parsed.get("ert_electrode_start") is not None:
        updates["hydro_ert_electrode_start"] = float(parsed["ert_electrode_start"])

    if parsed.get("srt_num_sensors") is not None:
        updates["hydro_srt_num_sensors"] = max(4, int(parsed["srt_num_sensors"]))

    if parsed.get("srt_sensor_spacing") is not None:
        updates["hydro_srt_sensor_spacing"] = max(0.2, float(parsed["srt_sensor_spacing"]))

    if parsed.get("srt_source_start") is not None:
        updates["hydro_srt_sensor_start"] = float(parsed["srt_source_start"])

    if parsed.get("srt_shot_distance") is not None:
        updates["hydro_srt_shot_distance"] = max(1, int(parsed["srt_shot_distance"]))

    rho_sat = parsed.get("rho_sat")
    if isinstance(rho_sat, (list, tuple)) and len(rho_sat) >= 3:
        updates["hydro_rho_sat_top"] = max(1.0, float(rho_sat[0]))
        updates["hydro_rho_sat_mid"] = max(1.0, float(rho_sat[1]))
        updates["hydro_rho_sat_bot"] = max(1.0, float(rho_sat[2]))

    archie_n = parsed.get("archie_n")
    if isinstance(archie_n, (list, tuple)) and len(archie_n) >= 3:
        updates["hydro_archie_n_top"] = max(0.1, float(archie_n[0]))
        updates["hydro_archie_n_mid"] = max(0.1, float(archie_n[1]))
        updates["hydro_archie_n_bot"] = max(0.1, float(archie_n[2]))

    sigma_s = parsed.get("sigma_s")
    if isinstance(sigma_s, (list, tuple)) and len(sigma_s) >= 3:
        updates["hydro_sigma_s_top"] = max(0.0, float(sigma_s[0]))
        updates["hydro_sigma_s_mid"] = max(0.0, float(sigma_s[1]))
        updates["hydro_sigma_s_bot"] = max(0.0, float(sigma_s[2]))

    if parsed.get("top_bulk_modulus") is not None:
        updates["hydro_top_bulk_modulus"] = max(1.0, float(parsed["top_bulk_modulus"]))
    if parsed.get("top_shear_modulus") is not None:
        updates["hydro_top_shear_modulus"] = max(1.0, float(parsed["top_shear_modulus"]))
    if parsed.get("top_mineral_density") is not None:
        updates["hydro_top_mineral_density"] = max(500.0, float(parsed["top_mineral_density"]))
    if parsed.get("top_depth") is not None:
        updates["hydro_top_depth"] = max(0.1, float(parsed["top_depth"]))

    if parsed.get("mid_bulk_modulus") is not None:
        updates["hydro_mid_bulk_modulus"] = max(1.0, float(parsed["mid_bulk_modulus"]))
    if parsed.get("mid_shear_modulus") is not None:
        updates["hydro_mid_shear_modulus"] = max(1.0, float(parsed["mid_shear_modulus"]))
    if parsed.get("mid_mineral_density") is not None:
        updates["hydro_mid_mineral_density"] = max(500.0, float(parsed["mid_mineral_density"]))
    if parsed.get("mid_aspect_ratio") is not None:
        updates["hydro_mid_aspect_ratio"] = float(min(1.0, max(0.001, float(parsed["mid_aspect_ratio"]))))

    if parsed.get("bot_bulk_modulus") is not None:
        updates["hydro_bot_bulk_modulus"] = max(1.0, float(parsed["bot_bulk_modulus"]))
    if parsed.get("bot_shear_modulus") is not None:
        updates["hydro_bot_shear_modulus"] = max(1.0, float(parsed["bot_shear_modulus"]))
    if parsed.get("bot_mineral_density") is not None:
        updates["hydro_bot_mineral_density"] = max(500.0, float(parsed["bot_mineral_density"]))
    if parsed.get("bot_aspect_ratio") is not None:
        updates["hydro_bot_aspect_ratio"] = float(min(1.0, max(0.001, float(parsed["bot_aspect_ratio"]))))

    point1 = parsed.get("point1")
    if isinstance(point1, (list, tuple)) and len(point1) == 2:
        updates["hydro_point1_x"] = float(point1[0])
        updates["hydro_point1_y"] = float(point1[1])

    point2 = parsed.get("point2")
    if isinstance(point2, (list, tuple)) and len(point2) == 2:
        updates["hydro_point2_x"] = float(point2[0])
        updates["hydro_point2_y"] = float(point2[1])

    if parsed.get("data_dir"):
        updates["hydro_data_dir"] = str(parsed["data_dir"]).strip()

    if parsed.get("output_dir"):
        updates["hydro_output_dir"] = str(parsed["output_dir"]).strip()

    return updates, response


def _apply_hydro_updates(updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        st.session_state[key] = value
        # Keep persisted copy in sync for keys subject to widget cleanup.
        if key == "hydro_methods":
            st.session_state["hydro_methods_persisted"] = list(value)


def _build_hydro_profile(
    data_dir: Path,
    snapshot_index: int,
    point1: List[float],
    point2: List[float],
    num_points: int,
) -> Dict[str, Any]:
    import numpy as np
    from PyHydroGeophysX.core.interpolation import ProfileInterpolator

    water_content_4d = np.load(data_dir / "Watercontent.npy")
    porosity_3d = np.load(data_dir / "Porosity.npy")
    top = np.loadtxt(data_dir / "top.txt")
    bot = np.load(data_dir / "bot.npy")

    if top.ndim != 2:
        raise ValueError(f"top.txt must be 2D. Got shape {top.shape}.")

    n_rows, n_cols = top.shape
    p1_col = int(np.clip(round(float(point1[0])), 0, n_cols - 1))
    p1_row = int(np.clip(round(float(point1[1])), 0, n_rows - 1))
    p2_col = int(np.clip(round(float(point2[0])), 0, n_cols - 1))
    p2_row = int(np.clip(round(float(point2[1])), 0, n_rows - 1))

    # Avoid degenerate zero-length profile.
    if p1_col == p2_col and p1_row == p2_row:
        if p2_col < n_cols - 1:
            p2_col += 1
        elif p2_row < n_rows - 1:
            p2_row += 1
        elif p1_col > 0:
            p1_col -= 1
        else:
            p1_row = max(0, p1_row - 1)

    if water_content_4d.ndim < 4:
        raise ValueError(f"Watercontent.npy must be 4D. Got shape {water_content_4d.shape}.")

    max_snapshot = int(water_content_4d.shape[0] - 1)
    if snapshot_index < 0 or snapshot_index > max_snapshot:
        raise ValueError(f"Snapshot index {snapshot_index} is out of range [0, {max_snapshot}].")

    water_content_3d = np.asarray(water_content_4d[snapshot_index], dtype=float)

    def _sample_profile(p1c: int, p1r: int, p2c: int, p2r: int):
        interp = ProfileInterpolator(
            point1=[p1c, p1r],
            point2=[p2c, p2r],
            surface_data=top,
            origin_x=0.0,
            origin_y=0.0,
            pixel_width=1.0,
            pixel_height=-1.0,
            num_points=int(num_points),
        )
        sampled_structure = interp.interpolate_layer_data([top] + [bot[i] for i in range(bot.shape[0])])
        sampled_wc = interp.interpolate_3d_data(water_content_3d)
        sampled_por = interp.interpolate_3d_data(porosity_3d)
        return interp, sampled_structure, sampled_wc, sampled_por

    interpolator, structure, water_content_profile, porosity_profile = _sample_profile(
        p1_col, p1_row, p2_col, p2_row
    )

    if not np.isfinite(water_content_profile).any() or not np.isfinite(porosity_profile).any():
        # Auto-fallback for datasets where default endpoints miss the active domain.
        active_mask = np.any(np.isfinite(water_content_3d) & np.isfinite(porosity_3d), axis=0)
        if np.any(active_mask):
            rows, cols = np.where(active_mask)
            min_col, max_col = int(np.min(cols)), int(np.max(cols))
            min_row, max_row = int(np.min(rows)), int(np.max(rows))
            med_col, med_row = int(np.median(cols)), int(np.median(rows))

            fallback_lines = [
                (min_col, med_row, max_col, med_row),
                (med_col, min_row, med_col, max_row),
                (min_col, min_row, max_col, max_row),
                (min_col, max_row, max_col, min_row),
            ]
            for fp1_col, fp1_row, fp2_col, fp2_row in fallback_lines:
                interp_try, struct_try, wc_try, por_try = _sample_profile(
                    fp1_col, fp1_row, fp2_col, fp2_row
                )
                if np.isfinite(wc_try).any() and np.isfinite(por_try).any():
                    interpolator = interp_try
                    structure = struct_try
                    water_content_profile = wc_try
                    porosity_profile = por_try
                    p1_col, p1_row, p2_col, p2_row = fp1_col, fp1_row, fp2_col, fp2_row
                    break

    structure = _fill_profile_nans(structure)
    water_content_profile = np.clip(_fill_profile_nans(water_content_profile), 0.0, 0.8)
    porosity_profile = np.clip(_fill_profile_nans(porosity_profile), 0.01, 0.6)

    n_layers, n_profile = water_content_profile.shape
    L_profile = np.asarray(interpolator.L_profile, dtype=float)

    return {
        "interpolator": interpolator,
        "structure": structure,
        "water_content_profile": water_content_profile,
        "porosity_profile": porosity_profile,
        "L_profile": L_profile,
        "n_layers": int(n_layers),
        "n_profile": int(n_profile),
        "snapshot_index": int(snapshot_index),
        "water_shape": tuple(int(v) for v in water_content_3d.shape),
        "porosity_shape": tuple(int(v) for v in porosity_3d.shape),
        "profile_points_used": [(int(p1_col), int(p1_row)), (int(p2_col), int(p2_row))],
    }


def _run_hydro_multigeophys_methods(config: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    import matplotlib.pyplot as plt

    methods = _ordered_unique_methods(config.get("hydro_methods", HYDRO_RESPONSE_METHODS))
    output_dir = _resolve_user_path(config.get("hydro_output_dir", "results/hydro_to_multigeophys"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "methods_requested": methods,
        "methods_completed": [],
        "files": {},
        "stats": {},
        "errors": [],
    }

    data_dir = _resolve_user_path(config.get("hydro_data_dir", "data"))
    profile = _build_hydro_profile(
        data_dir=data_dir,
        snapshot_index=int(config.get("hydro_snapshot_index", 5)),
        point1=[float(config.get("hydro_point1_x", 115)), float(config.get("hydro_point1_y", 70))],
        point2=[float(config.get("hydro_point2_x", 95)), float(config.get("hydro_point2_y", 180))],
        num_points=int(config.get("hydro_num_points", 220)),
    )
    result["stats"]["water_shape"] = profile["water_shape"]
    result["stats"]["porosity_shape"] = profile["porosity_shape"]
    result["stats"]["profile_points"] = profile["n_profile"]
    result["stats"]["snapshot_index"] = profile["snapshot_index"]

    if "Profile" in methods:
        layer_centers = 0.5 * (profile["structure"][:-1, :] + profile["structure"][1:, :])
        x2d = np.repeat(profile["L_profile"][np.newaxis, :], profile["n_layers"], axis=0)

        fig, ax = plt.subplots(figsize=(10, 4.5))
        cf = ax.contourf(x2d, layer_centers, profile["water_content_profile"], levels=25, cmap="YlGnBu")
        ax.plot(profile["L_profile"], profile["structure"][0, :], "k-", lw=1.2)
        ax.set_title("Hydrologic 2D profile (water content)")
        ax.set_xlabel("Distance along profile (m)")
        ax.set_ylabel("Elevation (m)")
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label("Water content (-)")
        plt.tight_layout()

        profile_path = output_dir / "hydro_profile_water_content.png"
        fig.savefig(profile_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        result["files"]["profile"] = str(profile_path)
        result["methods_completed"].append("Profile")

    needs_ert = "ERT" in methods
    needs_srt = "SRT" in methods
    if needs_ert or needs_srt:
        if not PYGIMLI_AVAILABLE:
            result["errors"].append("ERT/SRT requested but PyGIMLI is not installed.")
        else:
            try:
                import pygimli as pg
                from pygimli.physics import ert as pg_ert
                import pygimli.physics.traveltime as tt
                from PyHydroGeophysX.Hydro_modular import hydro_to_ert, hydro_to_srt
                from PyHydroGeophysX.core.interpolation import create_surface_lines
                from PyHydroGeophysX.core.mesh_utils import MeshCreator

                n_bounds = profile["structure"].shape[0]
                mid_idx = max(1, min(4, n_bounds // 3))
                bot_idx = max(mid_idx + 1, min(12, n_bounds - 2))
                surface, line1, line2 = create_surface_lines(
                    L_profile=profile["L_profile"],
                    structure=profile["structure"],
                    top_idx=0,
                    mid_idx=mid_idx,
                    bot_idx=bot_idx,
                )

                mesh_creator = MeshCreator(quality=32, area=1.0)
                mesh, _ = mesh_creator.create_from_layers(
                    surface=surface,
                    layers=[line1, line2],
                    bottom_depth=float(np.min(line2[:, 1]) - 10.0),
                )
                mesh_markers = _assign_three_layer_markers(mesh, line1, line2, top_marker=0, mid_marker=3, bot_marker=2)

                wc_mesh = _interpolate_profile_to_mesh(
                    profile["water_content_profile"],
                    profile["structure"],
                    profile["L_profile"],
                    mesh,
                )
                porosity_mesh = _interpolate_profile_to_mesh(
                    profile["porosity_profile"],
                    profile["structure"],
                    profile["L_profile"],
                    mesh,
                )

                rho_parameters = {
                    "rho_sat": [
                        max(1.0, float(config.get("hydro_rho_sat_top", 100.0))),
                        max(1.0, float(config.get("hydro_rho_sat_mid", 500.0))),
                        max(1.0, float(config.get("hydro_rho_sat_bot", 2400.0))),
                    ],
                    "n": [
                        max(0.1, float(config.get("hydro_archie_n_top", 2.2))),
                        max(0.1, float(config.get("hydro_archie_n_mid", 1.8))),
                        max(0.1, float(config.get("hydro_archie_n_bot", 2.5))),
                    ],
                    "sigma_s": [
                        max(0.0, float(config.get("hydro_sigma_s_top", 1.0 / 500.0))),
                        max(0.0, float(config.get("hydro_sigma_s_mid", 0.0))),
                        max(0.0, float(config.get("hydro_sigma_s_bot", 0.0))),
                    ],
                }
                vel_parameters = {
                    "top": {
                        "bulk_modulus": max(1.0, float(config.get("hydro_top_bulk_modulus", 30.0))),
                        "shear_modulus": max(1.0, float(config.get("hydro_top_shear_modulus", 20.0))),
                        "mineral_density": max(500.0, float(config.get("hydro_top_mineral_density", 2650.0))),
                        "depth": max(0.1, float(config.get("hydro_top_depth", 1.0))),
                    },
                    "mid": {
                        "bulk_modulus": max(1.0, float(config.get("hydro_mid_bulk_modulus", 50.0))),
                        "shear_modulus": max(1.0, float(config.get("hydro_mid_shear_modulus", 35.0))),
                        "mineral_density": max(500.0, float(config.get("hydro_mid_mineral_density", 2670.0))),
                        "aspect_ratio": float(np.clip(config.get("hydro_mid_aspect_ratio", 0.05), 0.001, 1.0)),
                    },
                    "bot": {
                        "bulk_modulus": max(1.0, float(config.get("hydro_bot_bulk_modulus", 55.0))),
                        "shear_modulus": max(1.0, float(config.get("hydro_bot_shear_modulus", 50.0))),
                        "mineral_density": max(500.0, float(config.get("hydro_bot_mineral_density", 2680.0))),
                        "aspect_ratio": float(np.clip(config.get("hydro_bot_aspect_ratio", 0.03), 0.001, 1.0)),
                    },
                }

                ert_scheme_token = str(config.get("hydro_ert_scheme", "wa")).strip().lower()
                ert_scheme = "dd" if ("dd" in ert_scheme_token or "dipole" in ert_scheme_token) else "wa"
                ert_num_electrodes = max(4, min(int(config.get("hydro_ert_num_electrodes", 72)), int(profile["n_profile"])))
                ert_electrode_spacing = max(0.2, float(config.get("hydro_ert_electrode_spacing", 1.0)))
                ert_electrode_start = float(config.get("hydro_ert_electrode_start", 15.0))

                srt_num_sensors = max(4, min(int(config.get("hydro_srt_num_sensors", 72)), int(profile["n_profile"])))
                srt_sensor_spacing = max(0.2, float(config.get("hydro_srt_sensor_spacing", 1.0)))
                srt_sensor_start = float(config.get("hydro_srt_sensor_start", 15.0))
                srt_shot_distance = max(1, int(config.get("hydro_srt_shot_distance", 5)))

                result["stats"]["ert_scheme"] = ert_scheme.upper()
                result["stats"]["srt_source_start"] = srt_sensor_start
                result["stats"]["srt_shot_distance"] = srt_shot_distance
                layer_markers = [0, 3, 2]
                seed = int(config.get("hydro_seed", 7))

                ert_data = None
                srt_data = None
                resistivity_model = None
                velocity_model = None

                if needs_srt:
                    srt_data, velocity_model = hydro_to_srt(
                        water_content=wc_mesh,
                        porosity=porosity_mesh,
                        mesh=mesh,
                        profile_interpolator=profile["interpolator"],
                        layer_idx=[0, mid_idx, bot_idx],
                        structure=profile["structure"],
                        marker_labels=layer_markers,
                        vel_parameters=vel_parameters,
                        sensor_spacing=srt_sensor_spacing,
                        sensor_start=srt_sensor_start,
                        num_sensors=srt_num_sensors,
                        shot_distance=srt_shot_distance,
                        noise_level=float(config.get("hydro_srt_noise_level", 0.01)),
                        noise_abs=float(config.get("hydro_srt_noise_abs", 1e-5)),
                        mesh_markers=mesh_markers,
                        verbose=False,
                        seed=seed,
                    )
                    result["stats"]["srt_data_count"] = int(srt_data.size())
                    result["methods_completed"].append("SRT")

                if needs_ert:
                    ert_data, resistivity_model = hydro_to_ert(
                        water_content=wc_mesh,
                        porosity=porosity_mesh,
                        mesh=mesh,
                        profile_interpolator=profile["interpolator"],
                        layer_idx=[0, mid_idx, bot_idx],
                        structure=profile["structure"],
                        marker_labels=layer_markers,
                        rho_parameters=rho_parameters,
                        electrode_spacing=ert_electrode_spacing,
                        electrode_start=ert_electrode_start,
                        num_electrodes=ert_num_electrodes,
                        scheme_name=ert_scheme,
                        noise_level=float(config.get("hydro_ert_noise_level", 0.03)),
                        abs_error=float(config.get("hydro_ert_abs_error", 0.0)),
                        rel_error=float(config.get("hydro_ert_rel_error", 0.03)),
                        mesh_markers=mesh_markers,
                        verbose=False,
                        seed=seed,
                    )
                    result["stats"]["ert_data_count"] = int(ert_data.size())
                    result["methods_completed"].append("ERT")

                if needs_ert and needs_srt:
                    fig = plt.figure(figsize=(14, 8))
                    ax1 = fig.add_subplot(2, 2, 1)
                    pg.show(mesh, resistivity_model, ax=ax1, cMap="Spectral_r", label="Resistivity (ohm m)")
                    ax1.set_title("2D resistivity model")

                    ax2 = fig.add_subplot(2, 2, 2)
                    pg.show(mesh, velocity_model, ax=ax2, cMap="turbo", label="Velocity (m/s)")
                    ax2.set_title("2D velocity model")

                    ax3 = fig.add_subplot(2, 2, 3)
                    pg_ert.show(ert_data, ax=ax3)
                    ax3.set_title("Synthetic ERT response")

                    ax4 = fig.add_subplot(2, 2, 4)
                    tt.drawFirstPicks(ax4, srt_data)
                    ax4.set_title("Synthetic SRT first arrivals")
                elif needs_ert:
                    fig = plt.figure(figsize=(12, 5))
                    ax1 = fig.add_subplot(1, 2, 1)
                    pg.show(mesh, resistivity_model, ax=ax1, cMap="Spectral_r", label="Resistivity (ohm m)")
                    ax1.set_title("2D resistivity model")

                    ax2 = fig.add_subplot(1, 2, 2)
                    pg_ert.show(ert_data, ax=ax2)
                    ax2.set_title("Synthetic ERT response")
                else:
                    fig = plt.figure(figsize=(12, 5))
                    ax1 = fig.add_subplot(1, 2, 1)
                    pg.show(mesh, velocity_model, ax=ax1, cMap="turbo", label="Velocity (m/s)")
                    ax1.set_title("2D velocity model")

                    ax2 = fig.add_subplot(1, 2, 2)
                    tt.drawFirstPicks(ax2, srt_data)
                    ax2.set_title("Synthetic SRT first arrivals")

                plt.tight_layout()
                ert_srt_path = output_dir / "ert_srt_responses.png"
                fig.savefig(ert_srt_path, dpi=220, bbox_inches="tight")
                plt.close(fig)
                result["files"]["ert_srt"] = str(ert_srt_path)
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"ERT/SRT generation failed: {exc}")

    em_methods = [m for m in ["TDEM", "FDEM", "Gravity"] if m in methods]
    if em_methods:
        n_profile = int(profile["n_profile"])
        # Match Ex_hydro_to_multigeophys notebook station sampling.
        station_divisor = max(1, int(config.get("hydro_station_count", 24)))
        step = max(1, n_profile // station_divisor)
        station_idx = np.r_[np.arange(0, n_profile, step, dtype=int), n_profile - 1]
        station_idx = np.unique(station_idx)
        if station_idx.size < 4:
            station_idx = np.unique(np.linspace(0, n_profile - 1, 4, dtype=int))

        x_station = profile["L_profile"][station_idx]
        wc_station = profile["water_content_profile"][:, station_idx]
        por_station = profile["porosity_profile"][:, station_idx]
        structure_station = profile["structure"][:, station_idx]

        plot_payloads: List[Dict[str, Any]] = []
        seed = int(config.get("hydro_seed", 7))

        if "TDEM" in em_methods:
            try:
                from PyHydroGeophysX.Hydro_modular import hydro_to_tdem

                times = np.logspace(-5, -2, 28)
                _, tdem_clean, _, _ = hydro_to_tdem(
                    water_content=wc_station,
                    porosity=por_station,
                    layer_boundaries=structure_station,
                    times=times,
                    sigma_w=0.05,
                    m=1.5,
                    n=2.0,
                    sigma_s=0.0,
                    source_radius=10.0,
                    noise_level=float(config.get("hydro_tdem_noise_level", 0.03)),
                    seed=seed,
                    verbose=False,
                )
                result["stats"]["tdem_shape"] = tuple(int(v) for v in tdem_clean.shape)
                result["methods_completed"].append("TDEM")
                plot_payloads.append(
                    {
                        "method": "TDEM",
                        "x": x_station,
                        "y": times,
                        "z": np.abs(tdem_clean).T,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"TDEM generation failed: {exc}")

        if "FDEM" in em_methods:
            try:
                from PyHydroGeophysX.Hydro_modular import hydro_to_fdem

                frequencies = np.logspace(1, 4, 18)
                _, fdem_clean, _, _ = hydro_to_fdem(
                    water_content=wc_station,
                    porosity=por_station,
                    layer_boundaries=structure_station,
                    frequencies=frequencies,
                    sigma_w=0.05,
                    m=1.5,
                    n=2.0,
                    sigma_s=0.0,
                    source_location=np.array([0.0, 0.0, 0.0]),
                    receiver_location=np.array([12.0, 0.0, 0.0]),
                    receiver_component="secondary",
                    waveform_type="dipole",
                    noise_level=float(config.get("hydro_fdem_noise_level", 0.03)),
                    seed=seed,
                    verbose=False,
                )
                result["stats"]["fdem_shape"] = tuple(int(v) for v in fdem_clean.shape)
                result["methods_completed"].append("FDEM")
                plot_payloads.append(
                    {
                        "method": "FDEM",
                        "x": x_station,
                        "y": frequencies,
                        "z": np.abs(np.imag(fdem_clean)).T,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"FDEM generation failed: {exc}")

        if "Gravity" in em_methods:
            try:
                from PyHydroGeophysX.Hydro_modular import hydro_to_gravity

                _, grav_clean, _, _ = hydro_to_gravity(
                    water_content=wc_station,
                    porosity=por_station,
                    layer_boundaries=structure_station,
                    station_positions=x_station,
                    rho_matrix=2650.0,
                    rho_water=1000.0,
                    rho_air=1.225,
                    sensor_height=1.0,
                    noise_level=float(config.get("hydro_gravity_noise_level", 0.02)),
                    seed=seed,
                    verbose=False,
                )
                result["stats"]["gravity_range_mgal"] = [float(np.min(grav_clean)), float(np.max(grav_clean))]
                result["methods_completed"].append("Gravity")
                plot_payloads.append(
                    {
                        "method": "Gravity",
                        "x": x_station,
                        "clean": grav_clean,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                result["errors"].append(f"Gravity generation failed: {exc}")

        if plot_payloads:
            fig, axes = plt.subplots(1, len(plot_payloads), figsize=(5.4 * len(plot_payloads), 4.8))
            if len(plot_payloads) == 1:
                axes = [axes]

            for ax, payload in zip(axes, plot_payloads):
                method = payload["method"]
                if method == "TDEM":
                    im = ax.pcolormesh(payload["x"], payload["y"], payload["z"], shading="auto", cmap="magma")
                    ax.set_yscale("log")
                    ax.set_xlabel("Distance along profile (m)")
                    ax.set_ylabel("Time (s)")
                    ax.set_title("Pseudo-2D TDEM |response|")
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("|dB/dt| (arb.)")
                elif method == "FDEM":
                    im = ax.pcolormesh(payload["x"], payload["y"], payload["z"], shading="auto", cmap="viridis")
                    ax.set_yscale("log")
                    ax.set_xlabel("Distance along profile (m)")
                    ax.set_ylabel("Frequency (Hz)")
                    ax.set_title("Pseudo-2D FDEM |imag|")
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label("|H_imag| (arb.)")
                elif method == "Gravity":
                    ax.plot(payload["x"], payload["clean"], "k-", lw=1.8, label="Gravity")
                    ax.set_xlabel("Distance along profile (m)")
                    ax.set_ylabel("Gravity anomaly (mGal)")
                    ax.set_title("Pseudo-2D gravity profile")
                    ax.grid(True, alpha=0.25)
                    ax.legend(loc="best")

            plt.tight_layout()
            em_path = output_dir / "em_gravity_responses.png"
            fig.savefig(em_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            result["files"]["em_gravity"] = str(em_path)

    result["methods_completed"] = _ordered_unique_methods(result["methods_completed"])
    return result


# ---------------------------------------------------------------------------
# Step-based workflow helpers
# ---------------------------------------------------------------------------

def _load_zip_to_tempdir(uploaded_zip) -> str:
    """Extract an uploaded .zip file into a temporary directory. Return the extracted root path."""
    import zipfile
    tmp_dir = tempfile.mkdtemp(prefix="phgx_hydro_")
    with zipfile.ZipFile(uploaded_zip, "r") as zf:
        zf.extractall(tmp_dir)
    # If the zip contains a single top-level directory, return that instead
    entries = [e for e in Path(tmp_dir).iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return tmp_dir


def _detect_hydro_structure(root_path: str) -> Dict[str, Any]:
    """Inspect a hydro model directory and return a summary dict with validity flags.

    The function auto-detects the data format (npy / modflow / parflow) and stores
    it under the ``data_format`` key so that downstream code can branch accordingly.
    """
    data_dir = Path(root_path)
    summary: Dict[str, Any] = {
        "valid": False,
        "path": str(data_dir),
        "data_format": "unknown",
        "missing_files": [],
        "snapshot_count": None,
        "water_shape": None,
        "porosity_shape": None,
        "bot_shape": None,
        "grid_info": "",
        "error": "",
    }

    fmt = _detect_hydro_data_format(data_dir)
    summary["data_format"] = fmt

    if fmt == "npy":
        # Standard pre-processed check
        required = ["Watercontent.npy", "Porosity.npy", "top.txt", "bot.npy"]
        missing = [name for name in required if not (data_dir / name).exists()]
        summary["missing_files"] = missing
        if missing:
            return summary
        try:
            import numpy as np
            water = np.load(data_dir / "Watercontent.npy", mmap_mode="r")
            porosity = np.load(data_dir / "Porosity.npy", mmap_mode="r")
            bot = np.load(data_dir / "bot.npy", mmap_mode="r")
            summary["water_shape"] = tuple(int(v) for v in water.shape)
            summary["porosity_shape"] = tuple(int(v) for v in porosity.shape)
            summary["bot_shape"] = tuple(int(v) for v in bot.shape)
            summary["snapshot_count"] = int(water.shape[0]) if water.ndim >= 1 else 0
            summary["valid"] = bool(water.ndim >= 3 and porosity.ndim >= 3 and bot.ndim >= 2)
            if water.ndim >= 3:
                summary["grid_info"] = f"{water.shape[1]} x {water.shape[2]} cells, {water.shape[0]} timesteps"
        except Exception as exc:  # noqa: BLE001
            summary["error"] = str(exc)

    elif fmt == "modflow":
        mf_info = _detect_modflow_info(data_dir)
        summary["modflow_info"] = mf_info
        summary["valid"] = mf_info["has_watercontent"]
        if mf_info["has_watercontent"]:
            summary["grid_info"] = "MODFLOW binary WaterContent detected"
        else:
            summary["error"] = "WaterContent binary file not found in MODFLOW directory."

    elif fmt == "parflow":
        pf_info = _detect_parflow_info(data_dir)
        summary["parflow_info"] = pf_info
        summary["valid"] = pf_info["num_satur_files"] > 0 and pf_info["has_porosity"]
        if summary["valid"]:
            summary["grid_info"] = (
                f"ParFlow run '{pf_info['run_name']}' – "
                f"{pf_info['num_satur_files']} saturation files"
            )
        else:
            missing_items = []
            if pf_info["num_satur_files"] == 0:
                missing_items.append("saturation .pfb files")
            if not pf_info["has_porosity"]:
                missing_items.append("porosity .pfb file")
            summary["error"] = "Missing: " + ", ".join(missing_items)
    else:
        summary["error"] = (
            "Could not detect data format. Expected one of:\n"
            "• Pre-processed: Watercontent.npy, Porosity.npy, top.txt, bot.npy\n"
            "• MODFLOW: WaterContent binary + mfsim.nam / .nam files\n"
            "• ParFlow: *.out.satur.*.pfb + porosity.pfb files"
        )

    return summary


def _get_planned_methods() -> List[str]:
    """Return the list of methods the user has selected.

    Reads from the persisted key which survives widget cleanup when
    the multiselect on Step 2 is not rendered (user on another step).
    """
    return list(st.session_state.get("hydro_methods_persisted",
                                     st.session_state.get("hydro_methods", [])))


def _hydro_step_status() -> Dict[str, Any]:
    """Compute validation status for each workflow step."""
    summary = st.session_state.get("hydro_data_summary")
    data_ok = summary is not None and summary.get("valid", False)
    methods = _get_planned_methods()
    methods_ok = len(methods) > 0
    root = st.session_state.get("hydro_root")
    has_both_points = (
        st.session_state.get("hydro_user_picked_p1", False)
        and st.session_state.get("hydro_user_picked_p2", False)
        and st.session_state.get("hydro_point1_x") is not None
        and st.session_state.get("hydro_point1_y") is not None
        and st.session_state.get("hydro_point2_x") is not None
        and st.session_state.get("hydro_point2_y") is not None
    )
    profile_ok = root is not None and Path(root).exists() and has_both_points
    has_results = st.session_state.hydro_last_run is not None
    return {
        "data_ok": data_ok,
        "methods_ok": methods_ok,
        "profile_ok": profile_ok,
        "all_valid": data_ok and methods_ok and profile_ok,
        "has_results": has_results,
        "methods": methods,
    }


def _render_status_strip() -> None:
    """Render the persistent status strip at the top of the workflow."""
    status = _hydro_step_status()
    current = st.session_state.get("hydro_active_step", 1)

    step_defs = [
        (1, "Data", status["data_ok"]),
        (2, "Method", status["methods_ok"]),
        (3, "Profile", status["profile_ok"]),
        (4, "Run", status["all_valid"]),
        (5, "Results", status["has_results"]),
    ]

    html_parts = []
    for num, label, ok in step_defs:
        if num == current:
            cls = "active"
        elif ok:
            cls = "done"
        else:
            cls = ""
        check = "&#10003;" if ok and num != current else ""
        html_parts.append(
            f'<div class="hydro-step-indicator {cls}">'
            f'<span class="hydro-step-num">{check if check else num}</span>'
            f'{label}</div>'
        )

    st.markdown(
        f'<div class="hydro-status-strip">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )

    # Step navigation buttons (real Streamlit buttons for interactivity)
    cols = st.columns(5)
    step_labels = ["1. Data", "2. Method", "3. Profile", "4. Run", "5. Results"]
    for i, (col, label) in enumerate(zip(cols, step_labels), 1):
        if col.button(
            label,
            key=f"hydro_nav_step_{i}",
            use_container_width=True,
            type="primary" if i == current else "secondary",
        ):
            st.session_state.hydro_active_step = i
            st.rerun()


# ---------------------------------------------------------------------------
# Step 1: Data (card-based source picker)
# ---------------------------------------------------------------------------

def _set_data_from_accessor(acc, resolved: str, source: str, label: str) -> None:
    """Shared helper: validate via accessor and store results in session_state."""
    ok, summary, errors = acc.validate()
    st.session_state.hydro_accessor = acc
    st.session_state.hydro_data_summary = {
        "valid": ok, "path": resolved, **summary,
        "missing_files": [], "error": "; ".join(errors) if errors else "",
    }
    st.session_state.hydro_root = resolved
    st.session_state.hydro_data_source = source
    st.session_state.hydro_data_source_label = label
    st.session_state.hydro_data_dir = resolved


def _render_data_step() -> None:
    """Step 1 - Data: card-based source picker with multi-format support."""
    st.markdown("##### Choose your data source")

    # --- Data format selector ---
    fmt_options = ["Auto-detect", "Pre-processed (.npy)", "MODFLOW", "ParFlow"]
    fmt_idx = fmt_options.index(st.session_state.get("hydro_data_format_choice", "Auto-detect"))
    chosen_fmt = st.radio(
        "Data format",
        fmt_options,
        index=fmt_idx,
        key="hydro_fmt_radio",
        horizontal=True,
        help=(
            "**Auto-detect** inspects the directory automatically.  \n"
            "**Pre-processed** expects Watercontent.npy, Porosity.npy, top.txt, bot.npy.  \n"
            "**MODFLOW** reads binary WaterContent + flopy-based porosity.  \n"
            "**ParFlow** reads .pfb saturation/porosity files."
        ),
    )
    st.session_state.hydro_data_format_choice = chosen_fmt

    on_cloud = _is_streamlit_cloud()
    current_source = st.session_state.get("hydro_data_source", None)

    # Selection via columns of buttons
    btn_cols = st.columns(3 if not on_cloud else 2)

    with btn_cols[0]:
        st.markdown(
            '<div style="text-align:center;font-size:2rem;">&#128230;</div>'
            '<div style="text-align:center;font-size:0.8rem;color:#64748b;margin-bottom:0.3rem;">'
            'Treeline catchment demo dataset included in the repository</div>',
            unsafe_allow_html=True,
        )
        if st.button("Example (bundled)", key="hydro_src_example",
                      use_container_width=True,
                      type="primary" if current_source == "example" else "secondary"):
            example_path = CURRENT_DIR / "data"
            resolved = str(example_path.resolve())
            if _has_hydro_required_files(example_path):
                if DATA_ACCESS_AVAILABLE:
                    _set_data_from_accessor(LocalHydroAccessor(resolved), resolved, "example", "Example (bundled)")
                else:
                    st.session_state.hydro_data_summary = _detect_hydro_structure(resolved)
                    st.session_state.hydro_root = resolved
                    st.session_state.hydro_data_source = "example"
                    st.session_state.hydro_data_source_label = "Example (bundled)"
                    st.session_state.hydro_data_dir = resolved
                st.rerun()
            else:
                st.warning("Example data not found. Try GitHub/HTTP instead.")

    with btn_cols[1]:
        st.markdown(
            '<div style="text-align:center;font-size:2rem;">&#127760;</div>'
            '<div style="text-align:center;font-size:0.8rem;color:#64748b;margin-bottom:0.3rem;">'
            'Download from a remote URL on demand; cached locally</div>',
            unsafe_allow_html=True,
        )
        if st.button("GitHub / HTTP", key="hydro_src_http",
                      use_container_width=True,
                      type="primary" if current_source == "http" else "secondary"):
            st.session_state.hydro_data_source = "http"
            st.rerun()

    if not on_cloud and len(btn_cols) > 2:
        with btn_cols[2]:
            st.markdown(
                '<div style="text-align:center;font-size:2rem;">&#128193;</div>'
                '<div style="text-align:center;font-size:0.8rem;color:#64748b;margin-bottom:0.3rem;">'
                'Read directly from a folder on your machine</div>',
                unsafe_allow_html=True,
            )
            if st.button("Local path", key="hydro_src_local",
                          use_container_width=True,
                          type="primary" if current_source == "local" else "secondary"):
                st.session_state.hydro_data_source = "local"
                st.rerun()

    # --- Source-specific controls ---
    if current_source == "http":
        st.markdown("---")
        if not DATA_ACCESS_AVAILABLE:
            st.error(f"The `data_access` module could not be imported.  \n**Reason:** {_DATA_ACCESS_ERROR}")
        else:
            custom_url = st.text_input(
                "Base URL (folder containing the required .npy / .txt files)",
                value=st.session_state.get("hydro_custom_url", ""),
                key="hydro_custom_url_input",
                placeholder="https://raw.githubusercontent.com/user/repo/branch/data",
                help=(
                    "Enter the URL of a directory that contains "
                    "Watercontent.npy, Porosity.npy, top.txt, and bot.npy. "
                    "Each file will be downloaded as `<base_url>/<filename>`."
                ),
            )
            col_dl, col_clr = st.columns([3, 1])
            if col_dl.button("Download & validate", key="hydro_http_custom_load",
                             use_container_width=True, type="primary"):
                url = custom_url.strip().rstrip("/")
                if not url:
                    st.error("Please enter a URL.")
                else:
                    custom_entry = {
                        "id": "_custom_url",
                        "name": "Custom URL",
                        "base_url": url,
                        "files": list(REQUIRED_FILES),
                    }
                    acc = HttpHydroAccessor(custom_entry)
                    with st.spinner("Downloading files (cached after first fetch)..."):
                        ok, summary, errors = acc.validate()
                    if ok:
                        work_dir = tempfile.mkdtemp(prefix="phgx_http_")
                        local_dir = acc.materialize(list(REQUIRED_FILES), work_dir)
                        _set_data_from_accessor(acc, local_dir, "http", f"Custom ({url})")
                    else:
                        st.session_state.hydro_accessor = acc
                        st.session_state.hydro_data_summary = {
                            "valid": False, "path": url, **summary,
                            "missing_files": [], "error": "; ".join(errors),
                        }
                    st.session_state.hydro_custom_url = url
                    st.rerun()
            if col_clr.button("Clear cache", key="hydro_http_custom_clear", use_container_width=True):
                url = custom_url.strip().rstrip("/")
                if url:
                    custom_entry = {
                        "id": "_custom_url", "name": "Custom URL",
                        "base_url": url, "files": list(REQUIRED_FILES),
                    }
                    n = HttpHydroAccessor(custom_entry).clear_cache()
                    st.success(f"Cleared {n} cached file(s).")

    elif current_source == "local":
        st.markdown("---")
        if on_cloud:
            st.error("Local paths are not available on Streamlit Cloud.")
        else:
            manual_path = st.text_input(
                "Hydro data directory", value=st.session_state.get("hydro_data_dir", "data"),
                key="hydro_manual_path_input",
                help="Folder with model data (auto-detected or specify format above)",
            )
            resolved_path = _resolve_user_path(manual_path)

            # Determine effective format
            if chosen_fmt == "Auto-detect":
                eff_fmt = _detect_hydro_data_format(resolved_path)
                if eff_fmt != "unknown":
                    st.info(f"Auto-detected format: **{eff_fmt.upper()}**")
            else:
                eff_fmt = {"Pre-processed (.npy)": "npy", "MODFLOW": "modflow", "ParFlow": "parflow"}[chosen_fmt]

            # --- Format-specific configuration ---
            if eff_fmt == "modflow":
                _render_modflow_config(resolved_path)
            elif eff_fmt == "parflow":
                _render_parflow_config(resolved_path)
            else:
                # npy or unknown — original discovery flow
                candidates = _discover_hydro_data_dirs(manual_path)
                if candidates:
                    detected_pick = st.selectbox(
                        "Detected folders", [str(p) for p in candidates], index=0,
                        key="hydro_detected_folders",
                    )
                else:
                    detected_pick = manual_path

                if st.button("Use this path", key="hydro_use_local_path",
                             use_container_width=True, type="primary"):
                    resolved = str(_resolve_user_path(detected_pick))
                    if DATA_ACCESS_AVAILABLE:
                        _set_data_from_accessor(LocalHydroAccessor(resolved), resolved, "local", "Local path")
                    else:
                        st.session_state.hydro_data_summary = _detect_hydro_structure(resolved)
                        st.session_state.hydro_root = resolved
                        st.session_state.hydro_data_source = "local"
                        st.session_state.hydro_data_source_label = "Local path"
                        st.session_state.hydro_data_dir = resolved
                    st.rerun()

    # --- Data summary ---
    _render_data_summary()


def _render_modflow_config(data_dir: Path) -> None:
    """Render MODFLOW-specific configuration controls and Load button."""
    mf_info = _detect_modflow_info(data_dir)
    st.markdown("**MODFLOW configuration**")

    if not mf_info["has_watercontent"]:
        st.warning("No `WaterContent` binary file found in the selected directory.")

    c1, c2, c3 = st.columns(3)
    idomain_file = c1.text_input(
        "Idomain file",
        value=st.session_state.get("hydro_mf_idomain", mf_info.get("idomain_file", "id.txt")),
        key="hydro_mf_idomain_input",
        help="Text or .npy file defining active cells (e.g. id.txt)",
    )
    model_name = c2.text_input(
        "Model name",
        value=st.session_state.get("hydro_mf_model_name", mf_info.get("model_name", "")),
        key="hydro_mf_model_name_input",
        help="MODFLOW model name used by flopy (e.g. TLnewtest2sfb2)",
    )
    nlay = c3.number_input(
        "Number of layers",
        min_value=1, max_value=100,
        value=int(st.session_state.get("hydro_mf_nlay", 3)),
        key="hydro_mf_nlay_input",
        help="Number of vertical layers in the UZF water-content output",
    )

    if st.button("Load & convert MODFLOW data", key="hydro_load_modflow",
                  use_container_width=True, type="primary"):
        st.session_state.hydro_mf_idomain = idomain_file
        st.session_state.hydro_mf_model_name = model_name
        st.session_state.hydro_mf_nlay = nlay
        with st.spinner("Reading MODFLOW binary files and converting to .npy …"):
            try:
                out_dir = _convert_modflow_to_npy(
                    modflow_dir=str(data_dir),
                    idomain_file=idomain_file,
                    model_name=model_name,
                    nlay=int(nlay),
                )
                # Store as if it were a standard .npy directory
                summary = _detect_hydro_structure(out_dir)
                summary["data_format"] = "modflow"
                summary["original_path"] = str(data_dir)
                st.session_state.hydro_data_summary = summary
                st.session_state.hydro_root = out_dir
                st.session_state.hydro_data_source = "local"
                st.session_state.hydro_data_source_label = "MODFLOW (converted)"
                st.session_state.hydro_data_dir = out_dir
                st.rerun()
            except Exception as exc:
                st.error(f"MODFLOW conversion failed: {exc}")


def _render_parflow_config(data_dir: Path) -> None:
    """Render ParFlow-specific configuration controls and Load button."""
    pf_info = _detect_parflow_info(data_dir)
    st.markdown("**ParFlow configuration**")

    if pf_info["num_satur_files"] == 0:
        st.warning("No saturation `.pfb` files found in the selected directory.")

    run_name = st.text_input(
        "Run name",
        value=st.session_state.get("hydro_pf_run_name", pf_info.get("run_name", "")),
        key="hydro_pf_run_name_input",
        help="The ParFlow run name prefix (e.g. test2)",
    )
    mc = st.columns(3)
    mc[0].metric("Saturation files", pf_info["num_satur_files"])
    mc[1].metric("Porosity", "found" if pf_info["has_porosity"] else "missing")
    mc[2].metric("Mask", "found" if pf_info["has_mask"] else "not found")

    if st.button("Load & convert ParFlow data", key="hydro_load_parflow",
                  use_container_width=True, type="primary"):
        st.session_state.hydro_pf_run_name = run_name
        with st.spinner("Reading ParFlow .pfb files and converting to .npy …"):
            try:
                out_dir = _convert_parflow_to_npy(
                    parflow_dir=str(data_dir),
                    run_name=run_name,
                )
                summary = _detect_hydro_structure(out_dir)
                summary["data_format"] = "parflow"
                summary["original_path"] = str(data_dir)
                st.session_state.hydro_data_summary = summary
                st.session_state.hydro_root = out_dir
                st.session_state.hydro_data_source = "local"
                st.session_state.hydro_data_source_label = "ParFlow (converted)"
                st.session_state.hydro_data_dir = out_dir
                st.rerun()
            except Exception as exc:
                st.error(f"ParFlow conversion failed: {exc}")


def _render_data_summary() -> None:
    """Show validated data summary."""
    summary = st.session_state.get("hydro_data_summary")
    if not summary:
        if st.session_state.get("hydro_root") is None:
            st.info("No data loaded yet. Select a source above and load it.")
        return

    if summary.get("valid"):
        st.markdown("---")
        data_fmt = summary.get("data_format", "npy")
        fmt_label = {"npy": "Pre-processed .npy", "modflow": "MODFLOW", "parflow": "ParFlow"}.get(data_fmt, data_fmt)
        st.markdown(
            '<div class="phgx-card">'
            '<strong>Detected data summary</strong></div>',
            unsafe_allow_html=True,
        )
        mc = st.columns(5)
        mc[0].metric("Format", fmt_label)
        mc[1].metric("Snapshots", summary.get("snapshot_count", "?"))
        mc[2].metric("Grid", summary.get("grid_info", "?"))
        mc[3].metric("Water shape", str(summary.get("water_shape", "?")))
        mc[4].metric("Source", st.session_state.get("hydro_data_source_label", "?"))
        if summary.get("original_path"):
            st.caption(f"Original model directory: `{summary['original_path']}`")
    else:
        err = summary.get("error", "")
        missing = summary.get("missing_files", [])
        if missing:
            st.error(f"Missing required files: {', '.join(missing)}")
        elif err:
            st.error(f"Error: {err}")
        else:
            st.error("Data not valid. Expected water/porosity >=3D arrays and bot >=2D.")


# ---------------------------------------------------------------------------
# Step 2: Method (pill toggles)
# ---------------------------------------------------------------------------

_METHOD_ICONS: Dict[str, str] = {
    "Profile": "&#9776;",
    "ERT": "&#9889;",
    "SRT": "&#127925;",
    "TDEM": "&#128225;",
    "FDEM": "&#128246;",
    "Gravity": "&#127758;",
}


def _render_method_step() -> None:
    """Step 2 - Select methods, then show per-method setup + rock physics."""
    st.markdown("##### Select Geophysical Methods")

    # Quick-preset buttons — write to persisted key so selection survives step navigation
    p1, p2, p3, p4 = st.columns(4)
    if p1.button("Select All", key="hydro_btn_all3", use_container_width=True):
        st.session_state.hydro_methods_persisted = HYDRO_RESPONSE_METHODS.copy()
        st.rerun()
    if p2.button("ERT + SRT", key="hydro_btn_ertsrt3", use_container_width=True):
        st.session_state.hydro_methods_persisted = ["Profile", "ERT", "SRT"]
        st.rerun()
    if p3.button("EM + Gravity", key="hydro_btn_emg3", use_container_width=True):
        st.session_state.hydro_methods_persisted = ["Profile", "TDEM", "FDEM", "Gravity"]
        st.rerun()
    if p4.button("Clear All", key="hydro_btn_clear3", use_container_width=True):
        st.session_state.hydro_methods_persisted = []
        st.rerun()

    # Use default= from persisted key; capture return value and sync back.
    # Do NOT rely on key= alone — Streamlit cleans up widget keys when the
    # widget is not rendered (e.g. user navigates to another step).
    _current_default = list(st.session_state.get("hydro_methods_persisted", HYDRO_RESPONSE_METHODS))
    selected_methods = st.multiselect(
        "Methods",
        options=HYDRO_RESPONSE_METHODS,
        default=_current_default,
        help="Choose one or more geophysical response types to compute.",
    )
    st.session_state.hydro_methods_persisted = list(selected_methods)

    methods = list(selected_methods)
    if not methods:
        st.warning("Select at least one method to proceed.")
        return

    method_set = set(methods)

    # Sanitise stored session values
    scheme_value = str(st.session_state.get("hydro_ert_scheme", "wa")).strip().lower()
    st.session_state.hydro_ert_scheme = scheme_value if scheme_value in {"wa", "dd"} else "wa"
    for key_name, min_value in [
        ("hydro_ert_num_electrodes", 4),
        ("hydro_srt_num_sensors", 4),
        ("hydro_srt_shot_distance", 1),
    ]:
        try:
            st.session_state[key_name] = max(min_value, int(st.session_state.get(key_name, min_value)))
        except Exception:
            st.session_state[key_name] = min_value

    st.markdown("---")

    # Build one tab per selected method (skip Profile – it's just sampling)
    tab_methods = [m for m in methods if m != "Profile"]
    if not tab_methods:
        # Only Profile selected – show sampling settings inline
        st.number_input("Profile points", min_value=50, max_value=2000, step=10, key="hydro_num_points")
        st.number_input("Random seed", min_value=0, step=1, key="hydro_seed")
        return

    method_tabs = st.tabs(tab_methods)

    # Track which rock-physics group has already been rendered (avoid duplicate keys)
    resistivity_rendered = False

    for tab, method in zip(method_tabs, tab_methods):
        with tab:
            if method == "ERT":
                _render_ert_tab()
                resistivity_rendered = True
            elif method == "SRT":
                _render_srt_tab()
            elif method == "TDEM":
                _render_tdem_tab(show_rock_physics=not resistivity_rendered)
                resistivity_rendered = True
            elif method == "FDEM":
                _render_fdem_tab(show_rock_physics=not resistivity_rendered)
                resistivity_rendered = True
            elif method == "Gravity":
                _render_gravity_tab()

    # Shared settings
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Profile points", min_value=50, max_value=2000, step=10, key="hydro_num_points")
    with c2:
        st.number_input("Random seed", min_value=0, step=1, key="hydro_seed")
    if method_set & {"TDEM", "FDEM", "Gravity"}:
        st.number_input("Station count (TDEM/FDEM/Gravity)", min_value=4, step=1, key="hydro_station_count")

    # Natural language dialog (collapsed)
    with st.expander("Natural Language Parameter Control", expanded=False):
        _render_dialog_control()


# ---------------------------------------------------------------------------
# Per-method tabs for Step 2
# ---------------------------------------------------------------------------

def _hd(key: str, fallback: Any = 0.0) -> Any:
    """Read a hydro default: session_state first, then HYDRO_NOTEBOOK_DEFAULTS, then fallback."""
    if key in st.session_state:
        return st.session_state[key]
    return HYDRO_NOTEBOOK_DEFAULTS.get(key, fallback)


def _render_resistivity_rock_physics() -> None:
    """Shared resistivity rock-physics inputs (Archie's Law), used by ERT/TDEM/FDEM."""
    st.markdown("**Rock Physics — Resistivity Model (Archie's Law)**")
    st.caption("Layer-wise: rho_sat (saturated resistivity), n (cementation exponent), sigma_s (surface conductivity).")
    rp1, rp2, rp3 = st.columns(3)
    with rp1:
        st.markdown("*Top layer*")
        st.number_input("rho_sat", min_value=1.0, step=10.0, value=float(_hd("hydro_rho_sat_top", 100.0)), key="hydro_rho_sat_top")
        st.number_input("Archie n", min_value=0.1, step=0.1, value=float(_hd("hydro_archie_n_top", 2.2)), key="hydro_archie_n_top")
        st.number_input("sigma_s", min_value=0.0, step=0.0005, format="%.4f", value=float(_hd("hydro_sigma_s_top", 0.002)), key="hydro_sigma_s_top")
    with rp2:
        st.markdown("*Middle layer*")
        st.number_input("rho_sat", min_value=1.0, step=10.0, value=float(_hd("hydro_rho_sat_mid", 500.0)), key="hydro_rho_sat_mid")
        st.number_input("Archie n", min_value=0.1, step=0.1, value=float(_hd("hydro_archie_n_mid", 1.8)), key="hydro_archie_n_mid")
        st.number_input("sigma_s", min_value=0.0, step=0.0005, format="%.4f", value=float(_hd("hydro_sigma_s_mid", 0.0)), key="hydro_sigma_s_mid")
    with rp3:
        st.markdown("*Bottom layer*")
        st.number_input("rho_sat", min_value=1.0, step=10.0, value=float(_hd("hydro_rho_sat_bot", 2400.0)), key="hydro_rho_sat_bot")
        st.number_input("Archie n", min_value=0.1, step=0.1, value=float(_hd("hydro_archie_n_bot", 2.5)), key="hydro_archie_n_bot")
        st.number_input("sigma_s", min_value=0.0, step=0.0005, format="%.4f", value=float(_hd("hydro_sigma_s_bot", 0.0)), key="hydro_sigma_s_bot")


def _render_ert_tab() -> None:
    """ERT acquisition setup + resistivity rock physics."""
    st.markdown("**Acquisition**")
    st.selectbox(
        "Array type", options=["wa", "dd"], key="hydro_ert_scheme",
        format_func=lambda v: "Wenner-Alpha" if v == "wa" else "Dipole-Dipole",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Electrodes", min_value=4, step=1, value=int(_hd("hydro_ert_num_electrodes", 72)), key="hydro_ert_num_electrodes")
    with c2:
        st.number_input("Spacing (m)", min_value=0.1, step=0.1, value=float(_hd("hydro_ert_electrode_spacing", 1.0)), key="hydro_ert_electrode_spacing")
    with c3:
        st.number_input("Start X (m)", min_value=0.0, step=0.5, value=float(_hd("hydro_ert_electrode_start", 15.0)), key="hydro_ert_electrode_start")

    st.markdown("**Noise**")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.number_input("Noise level", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_ert_noise_level", 0.03)), key="hydro_ert_noise_level")
    with n2:
        st.number_input("Relative error", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_ert_rel_error", 0.03)), key="hydro_ert_rel_error")
    with n3:
        st.number_input("Absolute error", min_value=0.0, step=0.0001, format="%.4f", value=float(_hd("hydro_ert_abs_error", 0.0)), key="hydro_ert_abs_error")

    _render_resistivity_rock_physics()


def _render_srt_tab() -> None:
    """SRT acquisition setup + velocity rock physics."""
    st.markdown("**Acquisition**")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Sensors", min_value=4, step=1, value=int(_hd("hydro_srt_num_sensors", 72)), key="hydro_srt_num_sensors")
    with c2:
        st.number_input("Spacing (m)", min_value=0.1, step=0.1, value=float(_hd("hydro_srt_sensor_spacing", 1.0)), key="hydro_srt_sensor_spacing")
    with c3:
        st.number_input("Start X (m)", min_value=0.0, step=0.5, value=float(_hd("hydro_srt_sensor_start", 15.0)), key="hydro_srt_sensor_start")
    st.number_input("Shot interval", min_value=1, step=1, value=int(_hd("hydro_srt_shot_distance", 5)), key="hydro_srt_shot_distance")

    st.markdown("**Noise**")
    s1, s2 = st.columns(2)
    with s1:
        st.number_input("Noise level", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_srt_noise_level", 0.01)), key="hydro_srt_noise_level")
    with s2:
        st.number_input("Absolute noise", min_value=0.0, step=0.0001, format="%.6f", value=float(_hd("hydro_srt_noise_abs", 1e-5)), key="hydro_srt_noise_abs")

    st.markdown("**Rock Physics — Velocity Model**")
    st.caption("Layer-wise elastic parameters: bulk/shear modulus (GPa), mineral density (kg/m³), depth factor or aspect ratio.")
    sv1, sv2, sv3 = st.columns(3)
    with sv1:
        st.markdown("*Top layer*")
        st.number_input("Bulk mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_top_bulk_modulus", 30.0)), key="hydro_top_bulk_modulus")
        st.number_input("Shear mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_top_shear_modulus", 20.0)), key="hydro_top_shear_modulus")
        st.number_input("Density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_top_mineral_density", 2650.0)), key="hydro_top_mineral_density")
        st.number_input("Depth factor", min_value=0.1, step=0.1, value=float(_hd("hydro_top_depth", 1.0)), key="hydro_top_depth")
    with sv2:
        st.markdown("*Middle layer*")
        st.number_input("Bulk mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_mid_bulk_modulus", 50.0)), key="hydro_mid_bulk_modulus")
        st.number_input("Shear mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_mid_shear_modulus", 35.0)), key="hydro_mid_shear_modulus")
        st.number_input("Density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_mid_mineral_density", 2670.0)), key="hydro_mid_mineral_density")
        st.number_input("Aspect ratio", min_value=0.001, max_value=1.0, step=0.01, value=float(_hd("hydro_mid_aspect_ratio", 0.05)), key="hydro_mid_aspect_ratio")
    with sv3:
        st.markdown("*Bottom layer*")
        st.number_input("Bulk mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_bot_bulk_modulus", 55.0)), key="hydro_bot_bulk_modulus")
        st.number_input("Shear mod. (GPa)", min_value=1.0, step=1.0, value=float(_hd("hydro_bot_shear_modulus", 50.0)), key="hydro_bot_shear_modulus")
        st.number_input("Density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_bot_mineral_density", 2680.0)), key="hydro_bot_mineral_density")
        st.number_input("Aspect ratio", min_value=0.001, max_value=1.0, step=0.01, value=float(_hd("hydro_bot_aspect_ratio", 0.03)), key="hydro_bot_aspect_ratio")


def _render_tdem_tab(show_rock_physics: bool = True) -> None:
    """TDEM noise + resistivity rock physics (shared with ERT)."""
    st.markdown("**Noise**")
    st.number_input("TDEM noise level", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_tdem_noise_level", 0.03)), key="hydro_tdem_noise_level")

    if show_rock_physics:
        _render_resistivity_rock_physics()
    else:
        st.info("Resistivity model parameters are configured in the **ERT** tab (shared).")


def _render_fdem_tab(show_rock_physics: bool = True) -> None:
    """FDEM noise + resistivity rock physics (shared with ERT)."""
    st.markdown("**Noise**")
    st.number_input("FDEM noise level", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_fdem_noise_level", 0.03)), key="hydro_fdem_noise_level")

    if show_rock_physics:
        _render_resistivity_rock_physics()
    else:
        st.info("Resistivity model parameters are configured in the **ERT** tab (shared).")


def _render_gravity_tab() -> None:
    """Gravity noise + density rock physics."""
    st.markdown("**Noise**")
    st.number_input("Gravity noise level", min_value=0.0, step=0.005, format="%.4f", value=float(_hd("hydro_gravity_noise_level", 0.02)), key="hydro_gravity_noise_level")

    st.markdown("**Rock Physics — Density Model**")
    st.caption("Layer-wise density parameters for gravity forward modeling.")
    gd1, gd2, gd3 = st.columns(3)
    with gd1:
        st.markdown("*Top layer*")
        st.number_input("Grain density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_density_top", 2650.0)), key="hydro_grav_density_top")
        st.number_input("Pore-fluid density", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_fluid_density_top", 1000.0)), key="hydro_grav_fluid_density_top")
    with gd2:
        st.markdown("*Middle layer*")
        st.number_input("Grain density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_density_mid", 2670.0)), key="hydro_grav_density_mid")
        st.number_input("Pore-fluid density", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_fluid_density_mid", 1000.0)), key="hydro_grav_fluid_density_mid")
    with gd3:
        st.markdown("*Bottom layer*")
        st.number_input("Grain density (kg/m³)", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_density_bot", 2680.0)), key="hydro_grav_density_bot")
        st.number_input("Pore-fluid density", min_value=500.0, step=10.0, value=float(_hd("hydro_grav_fluid_density_bot", 1000.0)), key="hydro_grav_fluid_density_bot")


def _render_advanced_settings(methods: Optional[List[str]] = None) -> None:
    """Legacy wrapper — kept for any remaining callers."""
    pass


def _render_dialog_control() -> None:
    """Natural-language dialog control for settings."""
    if st.session_state.context_agent:
        st.info("LLM is active. Describe settings in natural language.")
    else:
        st.caption("LLM not initialized. Rule-based parsing available.")

    dialog_examples = [
        ("Acquisition",
         "Set ERT array to dd, electrode count 96, spacing 1.5, start 10. "
         "Set SRT source start 20, shot distance 2, sensor count 80."),
        ("Methods + RP",
         "Use methods ERT and SRT only, snapshot index 8, profile points 320, "
         "point1=(110,70), point2=(90,180). "
         "Set rho_sat=[120,600,2200], archie_n=[2.1,1.9,2.4]."),
        ("Velocity",
         "Set velocity model: top bulk 28 shear 18 density 2620 depth 1.2; "
         "mid bulk 48 shear 33 density 2660 aspect 0.06; "
         "bot bulk 58 shear 52 density 2690 aspect 0.025."),
    ]
    ex_cols = st.columns(len(dialog_examples))
    for idx, (label, text) in enumerate(dialog_examples):
        if ex_cols[idx].button(label, key=f"hydro_dialog_ex3_{idx}", use_container_width=True):
            st.session_state.hydro_dialog_text = text
            st.rerun()

    st.text_area("Command", key="hydro_dialog_text", height=80,
                 placeholder="e.g. set rho_sat=[150,600,2000], archie_n=[2.0,1.8,2.5], sigma_s=[0.002,0,0]")

    d1, d2 = st.columns([3, 1])
    apply_dialog = d1.button("Apply", key="hydro_apply_dialog3", use_container_width=True, type="primary")
    if d2.button("Clear", key="hydro_clear_dialog3", use_container_width=True):
        st.session_state.hydro_chat_history = []
        st.session_state.hydro_dialog_text = ""
        st.rerun()

    if apply_dialog and st.session_state.hydro_dialog_text.strip():
        command = st.session_state.hydro_dialog_text.strip()
        parser_used = "rule-based"
        updates: Dict[str, Any] = {}
        llm_raw = ""
        if st.session_state.context_agent:
            try:
                llm_updates, llm_raw = _parse_hydro_command_llm(command)
                if llm_updates:
                    updates = llm_updates
                    parser_used = "llm"
            except Exception:  # noqa: BLE001
                updates = {}
        if not updates:
            updates = _parse_hydro_command_rule_based(command)
        if updates:
            _apply_hydro_updates(updates)
            st.session_state.hydro_dialog_text = ""
            st.session_state.hydro_chat_history.append(
                {"command": command, "parser": parser_used, "updates": updates, "llm_raw": llm_raw}
            )
            st.rerun()
        else:
            st.session_state.hydro_chat_history.append(
                {"command": command, "parser": "none", "updates": {}, "llm_raw": llm_raw}
            )
            st.warning("No changes detected.")

    if st.session_state.hydro_chat_history:
        with st.expander("Dialog history", expanded=False):
            for item in reversed(st.session_state.hydro_chat_history[-6:]):
                st.markdown(f"**{item.get('parser', '?')}**: `{item.get('command', '')}`")
                if item.get("updates"):
                    st.json(item["updates"])


# ---------------------------------------------------------------------------
# Step 3: Profile (map-first layout)
# ---------------------------------------------------------------------------

def _render_profile_step() -> None:
    """Step 3 - Map-first profile point selection."""
    st.markdown("##### Pick profile endpoints")

    def _as_float_or(value: Any, fallback: float) -> float:
        try:
            if value is None:
                raise TypeError
            return float(value)
        except Exception:  # noqa: BLE001
            return float(fallback)

    # Surface map (interactive picking)
    root = st.session_state.get("hydro_root")
    if root and Path(root).exists():
        _render_surface_picker(Path(root))
    else:
        st.markdown(
            '<div class="phgx-card" style="text-align:center; padding:3rem;">'
            '<div style="font-size:2.5rem; margin-bottom:0.5rem;">&#128506;</div>'
            '<strong>Surface map will appear here</strong><br>'
            '<span style="color:#64748b;">Load data in Step 1 first</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Or enter coordinates manually:")

    # Manual coordinate inputs — use notebook defaults when points are missing
    default_p1x = _as_float_or(HYDRO_NOTEBOOK_DEFAULTS.get("hydro_point1_x"), 115.0)
    default_p1y = _as_float_or(HYDRO_NOTEBOOK_DEFAULTS.get("hydro_point1_y"), 70.0)
    default_p2x = _as_float_or(HYDRO_NOTEBOOK_DEFAULTS.get("hydro_point2_x"), 95.0)
    default_p2y = _as_float_or(HYDRO_NOTEBOOK_DEFAULTS.get("hydro_point2_y"), 180.0)
    cur_p1x = _as_float_or(st.session_state.get("hydro_point1_x"), default_p1x)
    cur_p1y = _as_float_or(st.session_state.get("hydro_point1_y"), default_p1y)
    cur_p2x = _as_float_or(st.session_state.get("hydro_point2_x"), default_p2x)
    cur_p2y = _as_float_or(st.session_state.get("hydro_point2_y"), default_p2y)

    coord_col, snap_col, reset_col = st.columns([4, 2, 1])
    with coord_col:
        pc1, pc2, pc3, pc4 = st.columns(4)
        with pc1:
            p1x = st.number_input(
                "P1 X", value=float(cur_p1x),
                step=1.0, format="%.1f",
            )
        with pc2:
            p1y = st.number_input(
                "P1 Y", value=float(cur_p1y),
                step=1.0, format="%.1f",
            )
        with pc3:
            p2x = st.number_input(
                "P2 X", value=float(cur_p2x),
                step=1.0, format="%.1f",
            )
        with pc4:
            p2y = st.number_input(
                "P2 Y", value=float(cur_p2y),
                step=1.0, format="%.1f",
            )
    with snap_col:
        snapshot_idx = st.number_input(
            "Snapshot", min_value=0,
            value=int(st.session_state.get("hydro_snapshot_index", 5)),
            step=1, format="%d",
            help="Timestep index from the hydro model",
        )
    with reset_col:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear", key="hydro_reset_points", use_container_width=True):
            st.session_state.hydro_pick_next = 0
            st.session_state.hydro_user_picked_p1 = False
            st.session_state.hydro_user_picked_p2 = False
            for key in ("hydro_point1_x", "hydro_point1_y", "hydro_point2_x", "hydro_point2_y"):
                st.session_state.pop(key, None)
            st.session_state.hydro_snapshot_index = int(HYDRO_NOTEBOOK_DEFAULTS.get("hydro_snapshot_index", 5))
            st.rerun()

    # Persist manual inputs → session_state.
    # Only mark as user-picked if the number_input value was actually changed
    # from the default (to avoid auto-marking on every rerun).
    _prev_p1x = st.session_state.get("hydro_point1_x")
    _prev_p1y = st.session_state.get("hydro_point1_y")
    _prev_p2x = st.session_state.get("hydro_point2_x")
    _prev_p2y = st.session_state.get("hydro_point2_y")

    st.session_state.hydro_point1_x = float(p1x)
    st.session_state.hydro_point1_y = float(p1y)
    st.session_state.hydro_point2_x = float(p2x)
    st.session_state.hydro_point2_y = float(p2y)
    st.session_state.hydro_snapshot_index = int(snapshot_idx)

    # Detect if the user explicitly changed any manual coordinate input
    if _prev_p1x is not None and (float(p1x) != float(_prev_p1x) or float(p1y) != float(_prev_p1y)):
        st.session_state.hydro_user_picked_p1 = True
    if _prev_p2x is not None and (float(p2x) != float(_prev_p2x) or float(p2y) != float(_prev_p2y)):
        st.session_state.hydro_user_picked_p2 = True


# ---------------------------------------------------------------------------
# Step 4: Run (styled status + validation)
# ---------------------------------------------------------------------------

def _render_run_step() -> None:
    """Step 4 - Validate and execute."""
    st.markdown("##### Review & Run")

    status = _hydro_step_status()

    # Styled checklist
    checks = [
        ("Data loaded", status["data_ok"],
         st.session_state.get("hydro_data_source_label", "")),
        ("Methods selected", status["methods_ok"],
         ", ".join(status["methods"]) if status["methods"] else "none"),
        ("Profile configured", status["profile_ok"],
         (f"P1=({st.session_state.hydro_point1_x}, {st.session_state.hydro_point1_y}) "
          f"P2=({st.session_state.hydro_point2_x}, {st.session_state.hydro_point2_y})")
         if status["profile_ok"] else "Pick P1 and P2 on the map"),
    ]

    for label, ok, detail in checks:
        icon = "&#9989;" if ok else "&#10060;"
        color = "#059669" if ok else "#dc2626"
        bg = "#ecfdf5" if ok else "#fef2f2"
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:0.5rem; '
            f'padding:0.5rem 0.8rem; margin-bottom:0.4rem; '
            f'border-radius:0.5rem; background:{bg};">'
            f'<span style="font-size:1.1rem;">{icon}</span>'
            f'<strong style="color:{color};">{label}</strong>'
            f'<span style="color:#64748b; font-size:0.85rem; margin-left:auto;">{detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if not status["all_valid"]:
        st.caption("Complete all steps above before running.")

    st.text_input("Output directory", key="hydro_output_dir")

    with st.expander("Full configuration", expanded=False):
        st.json({
            "data_source": st.session_state.get("hydro_data_source_label", ""),
            "data_dir": st.session_state.get("hydro_root", ""),
            "output_dir": st.session_state.hydro_output_dir,
            "methods": status["methods"],
            "snapshot_index": st.session_state.hydro_snapshot_index,
            "profile_p1": [st.session_state.hydro_point1_x, st.session_state.hydro_point1_y],
            "profile_p2": [st.session_state.hydro_point2_x, st.session_state.hydro_point2_y],
            "num_points": st.session_state.hydro_num_points,
            "ert_scheme": st.session_state.hydro_ert_scheme,
            "noise": {
                "seed": st.session_state.hydro_seed,
                "ert_level": st.session_state.hydro_ert_noise_level,
                "ert_rel_error": st.session_state.hydro_ert_rel_error,
                "srt_level": st.session_state.hydro_srt_noise_level,
                "tdem_level": st.session_state.hydro_tdem_noise_level,
                "fdem_level": st.session_state.hydro_fdem_noise_level,
                "gravity_level": st.session_state.hydro_gravity_noise_level,
            },
        })

    methods = status["methods"]
    run_label = (
        f"Run {methods[0]} forward modeling" if len(methods) == 1
        else "Run selected geophysical methods"
    ) if methods else "Run (select methods first)"

    run_clicked = st.button(
        run_label, type="primary",
        disabled=not status["all_valid"],
        use_container_width=True,
    )

    if run_clicked:
        # Materialize data via accessor if available
        accessor = st.session_state.get("hydro_accessor")
        if accessor is not None:
            try:
                from PyHydroGeophysX.data_access.accessors import REQUIRED_FILES as _REQ
                work_dir = st.session_state.get("hydro_root") or tempfile.mkdtemp(prefix="phgx_run_")
                local_dir = accessor.materialize(list(_REQ), work_dir)
                st.session_state.hydro_data_dir = local_dir
                st.session_state.hydro_root = local_dir
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to materialize data: {exc}")
                return

        run_config = {
            "hydro_data_dir": st.session_state.hydro_data_dir,
            "hydro_output_dir": st.session_state.hydro_output_dir,
            "hydro_methods": methods,
            "hydro_snapshot_index": st.session_state.hydro_snapshot_index,
            "hydro_point1_x": st.session_state.hydro_point1_x,
            "hydro_point1_y": st.session_state.hydro_point1_y,
            "hydro_point2_x": st.session_state.hydro_point2_x,
            "hydro_point2_y": st.session_state.hydro_point2_y,
            "hydro_num_points": st.session_state.hydro_num_points,
            "hydro_station_count": st.session_state.hydro_station_count,
            "hydro_ert_scheme": st.session_state.hydro_ert_scheme,
            "hydro_ert_num_electrodes": st.session_state.hydro_ert_num_electrodes,
            "hydro_ert_electrode_spacing": st.session_state.hydro_ert_electrode_spacing,
            "hydro_ert_electrode_start": st.session_state.hydro_ert_electrode_start,
            "hydro_srt_num_sensors": st.session_state.hydro_srt_num_sensors,
            "hydro_srt_sensor_spacing": st.session_state.hydro_srt_sensor_spacing,
            "hydro_srt_sensor_start": st.session_state.hydro_srt_sensor_start,
            "hydro_srt_shot_distance": st.session_state.hydro_srt_shot_distance,
            "hydro_rho_sat_top": st.session_state.hydro_rho_sat_top,
            "hydro_rho_sat_mid": st.session_state.hydro_rho_sat_mid,
            "hydro_rho_sat_bot": st.session_state.hydro_rho_sat_bot,
            "hydro_archie_n_top": st.session_state.hydro_archie_n_top,
            "hydro_archie_n_mid": st.session_state.hydro_archie_n_mid,
            "hydro_archie_n_bot": st.session_state.hydro_archie_n_bot,
            "hydro_sigma_s_top": st.session_state.hydro_sigma_s_top,
            "hydro_sigma_s_mid": st.session_state.hydro_sigma_s_mid,
            "hydro_sigma_s_bot": st.session_state.hydro_sigma_s_bot,
            "hydro_top_bulk_modulus": st.session_state.hydro_top_bulk_modulus,
            "hydro_top_shear_modulus": st.session_state.hydro_top_shear_modulus,
            "hydro_top_mineral_density": st.session_state.hydro_top_mineral_density,
            "hydro_top_depth": st.session_state.hydro_top_depth,
            "hydro_mid_bulk_modulus": st.session_state.hydro_mid_bulk_modulus,
            "hydro_mid_shear_modulus": st.session_state.hydro_mid_shear_modulus,
            "hydro_mid_mineral_density": st.session_state.hydro_mid_mineral_density,
            "hydro_mid_aspect_ratio": st.session_state.hydro_mid_aspect_ratio,
            "hydro_bot_bulk_modulus": st.session_state.hydro_bot_bulk_modulus,
            "hydro_bot_shear_modulus": st.session_state.hydro_bot_shear_modulus,
            "hydro_bot_mineral_density": st.session_state.hydro_bot_mineral_density,
            "hydro_bot_aspect_ratio": st.session_state.hydro_bot_aspect_ratio,
            "hydro_seed": st.session_state.hydro_seed,
            "hydro_srt_noise_level": st.session_state.hydro_srt_noise_level,
            "hydro_srt_noise_abs": st.session_state.hydro_srt_noise_abs,
            "hydro_ert_noise_level": st.session_state.hydro_ert_noise_level,
            "hydro_ert_abs_error": st.session_state.hydro_ert_abs_error,
            "hydro_ert_rel_error": st.session_state.hydro_ert_rel_error,
            "hydro_tdem_noise_level": st.session_state.hydro_tdem_noise_level,
            "hydro_fdem_noise_level": st.session_state.hydro_fdem_noise_level,
            "hydro_gravity_noise_level": st.session_state.hydro_gravity_noise_level,
        }

        with st.spinner("Generating geophysical responses..."):
            try:
                st.session_state.hydro_last_run = _run_hydro_multigeophys_methods(run_config)
            except Exception as exc:  # noqa: BLE001
                st.session_state.hydro_last_run = {
                    "methods_requested": methods,
                    "methods_completed": [],
                    "files": {},
                    "stats": {},
                    "errors": [str(exc)],
                    "output_dir": st.session_state.hydro_output_dir,
                }
        st.session_state.hydro_run_clicked = True
        st.session_state.hydro_active_step = 5  # auto-navigate to results
        st.rerun()


# ---------------------------------------------------------------------------
# Step 5: Results (tabbed gallery)
# ---------------------------------------------------------------------------

def _render_results_step() -> None:
    """Step 5 - Tabbed results gallery."""
    st.markdown("##### Results")

    run_data = st.session_state.hydro_last_run
    if not run_data:
        st.markdown(
            '<div class="phgx-card" style="text-align:center; padding:3rem;">'
            '<div style="font-size:2.5rem; margin-bottom:0.5rem;">&#128202;</div>'
            '<strong>No results yet</strong><br>'
            '<span style="color:#64748b;">Complete Steps 1-4 and run forward modeling</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    completed = run_data.get("methods_completed", [])
    requested = run_data.get("methods_requested", [])
    if completed:
        st.success(f"Completed: {', '.join(completed)}")
    else:
        st.warning(f"No methods completed. Requested: {', '.join(requested)}")

    if run_data.get("errors"):
        for err in run_data["errors"]:
            st.warning(err)

    files = run_data.get("files", {})
    stats = run_data.get("stats", {})

    # Build available result tabs
    result_tab_names = []
    result_tab_keys = []
    caption_map = {
        "profile": "Hydrologic Profile",
        "ert_srt": "ERT / SRT Responses",
        "em_gravity": "TDEM / FDEM / Gravity",
    }
    for key in ["profile", "ert_srt", "em_gravity"]:
        if files.get(key) and Path(files[key]).exists():
            result_tab_names.append(caption_map[key])
            result_tab_keys.append(key)

    if stats:
        result_tab_names.append("Statistics")
        result_tab_keys.append("_stats")

    if result_tab_names:
        result_tabs = st.tabs(result_tab_names)
        for tab, key in zip(result_tabs, result_tab_keys):
            with tab:
                if key == "_stats":
                    st.json(stats)
                else:
                    path_obj = Path(files[key])
                    st.image(str(path_obj), use_container_width=True)
                    with open(path_obj, "rb") as fh:
                        st.download_button(
                            f"Download {path_obj.name}",
                            data=fh, file_name=path_obj.name,
                            mime="image/png", use_container_width=True,
                        )

    # Download all as ZIP
    existing_files = [
        Path(files[k]) for k in ["profile", "ert_srt", "em_gravity"]
        if files.get(k) and Path(files[k]).exists()
    ]
    if len(existing_files) > 1:
        import io
        import zipfile as _zf
        buf = io.BytesIO()
        with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
            for fp in existing_files:
                zf.write(str(fp), fp.name)
        buf.seek(0)
        st.download_button(
            "Download all results (.zip)",
            data=buf, file_name="hydro_geophys_results.zip",
            mime="application/zip", use_container_width=True,
        )

    st.caption(f"Output directory: `{run_data.get('output_dir', st.session_state.hydro_output_dir)}`")


# ---------------------------------------------------------------------------
# Main tab orchestrator (stepper-based)
# ---------------------------------------------------------------------------

def render_hydro_multigeophys_tab() -> None:
    st.subheader("Hydro to Geophysics")
    st.caption(
        "Convert hydro-model outputs into synthetic geophysical responses. "
        "Navigate the steps using the bar below."
    )

    # Initialize active step
    if "hydro_active_step" not in st.session_state:
        st.session_state.hydro_active_step = 1

    # Persistent status strip + navigation
    _render_status_strip()

    st.markdown("---")

    # Render only the active step
    step = st.session_state.hydro_active_step
    if step == 1:
        _render_data_step()
    elif step == 2:
        _render_method_step()
    elif step == 3:
        _render_profile_step()
    elif step == 4:
        _render_run_step()
    elif step == 5:
        _render_results_step()

    # Bottom navigation
    st.markdown("---")
    nav_prev, nav_spacer, nav_next = st.columns([1, 3, 1])
    if step > 1:
        if nav_prev.button("Previous", key="hydro_nav_prev", use_container_width=True):
            st.session_state.hydro_active_step = step - 1
            st.rerun()
    if step < 5:
        if nav_next.button("Next", key="hydro_nav_next",
                           use_container_width=True, type="primary"):
            st.session_state.hydro_active_step = step + 1
            st.rerun()


def render_sidebar() -> Dict[str, str]:
    st.sidebar.header("Configuration")

    provider = st.sidebar.selectbox(
        "LLM provider",
        options=["openai", "gemini", "claude"],
        index=["openai", "gemini", "claude"].index(st.session_state.llm_provider)
        if st.session_state.llm_provider in ["openai", "gemini", "claude"]
        else 0,
        help="Used by the context agent to parse your natural-language request.",
    )

    default_models = {"openai": "gpt-4o-mini", "gemini": "gemini-pro", "claude": "claude-3-opus-20240229"}
    model_default = st.session_state.llm_model or default_models.get(provider, "gpt-4o-mini")
    model = st.sidebar.text_input("Model name", value=model_default)

    env_map = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "claude": "ANTHROPIC_API_KEY"}
    preset_key = st.session_state.api_key or os.getenv(env_map[provider], "")
    api_key = st.sidebar.text_input(
        "API key",
        type="password",
        value=preset_key,
        help=f"Read from environment if set: {env_map[provider]}",
    )

    output_dir = st.sidebar.text_input("Output directory", value=st.session_state.output_dir)

    col_a, col_b = st.sidebar.columns(2)
    init_clicked = col_a.button("Initialize", type="primary", width="stretch")
    reset_clicked = col_b.button("Reset state", width="stretch")

    if reset_clicked:
        for key in ["context_agent", "workflow_result", "workflow_config", "upload_dir"]:
            st.session_state[key] = None
        st.session_state.user_request = ""
        for key in [
            "hydro_last_run",
            "hydro_data_summary",
            "hydro_root",
            "hydro_accessor",
            "hydro_data_source",
            "hydro_data_source_label",
            "hydro_surface_selected_points",
            "hydro_http_dataset_id",
        ]:
            st.session_state[key] = None if key != "hydro_surface_selected_points" else []
        _apply_hydro_notebook_defaults(force=True)
        st.session_state.hydro_defaults_version = HYDRO_DEFAULTS_VERSION
        st.session_state.hydro_active_step = 1
        st.sidebar.info("Session cleared and Hydro defaults restored from notebook.")

    if init_clicked:
        if not api_key.strip():
            st.sidebar.error("Please provide an API key or set the environment variable first.")
        else:
            try:
                st.session_state.context_agent = ContextInputAgent(
                    api_key=api_key.strip(), model=model.strip(), llm_provider=provider
                )
                st.session_state.api_key = api_key.strip()
                st.session_state.llm_model = model.strip()
                st.session_state.llm_provider = provider
                st.sidebar.success("Context agent ready.")
            except Exception as exc:  # noqa: BLE001
                st.sidebar.error(f"Initialization failed: {exc}")
                st.sidebar.exception(exc)

    st.sidebar.markdown("---")
    if st.session_state.context_agent:
        st.sidebar.success("System status: ready")
    else:
        st.sidebar.warning("System status: not initialized")

    return {"provider": provider, "model": model, "api_key": api_key, "output_dir": output_dir}


def save_upload(file_obj, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / file_obj.name
    dest.write_bytes(file_obj.read())
    return dest


def handle_uploads(
    output_dir: Path,
    uploaded_files: Optional[List],
    workflow_config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Single upload entrypoint. Saves all files and applies lightweight heuristics:
    - Electrode files detected by name (electrode/elec) kept as electrode_file
    - First .ohm/.data/.dat (excluding electrode files) becomes ert_file
    - If multiple data files remain, treat as time-lapse list
    - First file with 'seis' in name or '.seis/.sgy/.segy' becomes seismic_file
    - All files exposed in uploaded_files map
    """
    saved_paths: Dict[str, str] = {}
    if not uploaded_files:
        return saved_paths

    temp_dir = Path(tempfile.mkdtemp(prefix="phgx_uploads_"))
    st.session_state.upload_dir = str(temp_dir)
    st.info(f"Uploaded files stored in: {temp_dir}")

    all_paths = []
    for f in uploaded_files:
        dest = save_upload(f, temp_dir)
        all_paths.append(dest)
        saved_paths[f.name] = str(dest)

    # Heuristics for convenience
    def is_electrode(p: Path) -> bool:
        name = p.name.lower()
        return ("electrode" in name or "electrodes" in name or "elec" in name) and p.suffix.lower() in [".dat", ".txt", ".csv"]

    electrode_files = [p for p in all_paths if is_electrode(p)]
    
    # Detect TDEM files (typically contain 'tdem', 'tem', or 'electromagnetic' in name)
    def is_tdem(p: Path) -> bool:
        name = p.name.lower()
        return ("tdem" in name or "tem_" in name or "electromagnetic" in name) and p.suffix.lower() in [".txt", ".dat", ".csv"]
    
    tdem_candidates = [p for p in all_paths if is_tdem(p)]

    # Detect hydrological model files (MODFLOW / ParFlow)
    def is_modflow_idomain(p: Path) -> bool:
        return p.name.lower() in ["id.txt", "idomain.txt", "idomain.dat"]

    def is_modflow_watercontent(p: Path) -> bool:
        return p.name.lower() == "watercontent"

    def is_modflow_nam(p: Path) -> bool:
        return p.suffix.lower() == ".nam"

    def is_parflow_pfb(p: Path) -> bool:
        return p.suffix.lower() == ".pfb"

    def infer_parflow_run_name(p: Path) -> Optional[str]:
        name_lower = p.name.lower()
        patterns = [".out.satur.", ".out.press.", ".out.porosity", ".out.mask"]
        for pattern in patterns:
            if pattern in name_lower:
                idx = name_lower.index(pattern)
                return p.name[:idx]
        return None

    modflow_idomain_files = [p for p in all_paths if is_modflow_idomain(p)]
    modflow_wc_files = [p for p in all_paths if is_modflow_watercontent(p)]
    modflow_nam_files = [p for p in all_paths if is_modflow_nam(p)]
    parflow_pfb_files = [p for p in all_paths if is_parflow_pfb(p)]
    has_modflow = bool(modflow_idomain_files or modflow_wc_files or modflow_nam_files)
    has_parflow = bool(parflow_pfb_files)
    
    def is_hydro(p: Path) -> bool:
        return is_modflow_idomain(p) or is_modflow_watercontent(p) or is_modflow_nam(p) or is_parflow_pfb(p)

    # Data candidates exclude electrode files, seismic files, TDEM files, and hydro model files
    data_candidates = [
        p for p in all_paths
        if p.suffix.lower() in [".ohm", ".data", ".dat", ".txt"]
        and not is_electrode(p)
        and "seis" not in p.name.lower()
        and not is_tdem(p)
        and not is_hydro(p)
    ]
    seismic_candidates = [p for p in all_paths if "seis" in p.name.lower() or p.suffix.lower() in [".sgy", ".segy"]]

    if electrode_files:
        workflow_config["electrode_file"] = str(electrode_files[0])

    if len(data_candidates) == 1:
        workflow_config["data_file"] = str(data_candidates[0])
        workflow_config["ert_file"] = str(data_candidates[0])
    elif len(data_candidates) > 1:
        workflow_config["time_lapse_files"] = [str(p) for p in data_candidates]
        workflow_config["timelapse_files"] = [str(p) for p in data_candidates]

    if seismic_candidates:
        workflow_config["seismic_file"] = str(seismic_candidates[0])
    
    if tdem_candidates:
        workflow_config["tdem_file"] = str(tdem_candidates[0])

    # Hydrological model upload handling (single-file friendly)
    if has_modflow or has_parflow:
        if has_modflow and has_parflow:
            workflow_config["hydro_model"] = "both"
        elif has_modflow:
            workflow_config["hydro_model"] = "modflow"
        else:
            workflow_config["hydro_model"] = "parflow"

    if modflow_idomain_files:
        workflow_config["idomain_file"] = str(modflow_idomain_files[0])
        workflow_config["modflow_dir"] = str(modflow_idomain_files[0].parent)
    if modflow_wc_files:
        workflow_config["modflow_dir"] = str(modflow_wc_files[0].parent)
        workflow_config["load_water_content"] = True
    if modflow_nam_files and "modflow_dir" not in workflow_config:
        workflow_config["modflow_dir"] = str(modflow_nam_files[0].parent)

    if parflow_pfb_files:
        # Use the first PFB file to infer run_name and set directory
        first_pfb = parflow_pfb_files[0]
        workflow_config["parflow_dir"] = str(first_pfb.parent)
        run_name = infer_parflow_run_name(first_pfb)
        if run_name:
            workflow_config["run_name"] = run_name
        # Set load flags based on filename patterns
        name_lower = first_pfb.name.lower()
        if ".out.satur." in name_lower:
            workflow_config["load_saturation"] = True
        if ".out.porosity" in name_lower:
            workflow_config["load_porosity"] = True
        if ".out.mask" in name_lower:
            workflow_config["load_mask"] = True

    # Expose all uploads for downstream agents
    workflow_config["uploaded_files"] = {p.name: str(p) for p in all_paths}
    workflow_config["output_dir"] = str(output_dir)
    return saved_paths


def run_workflow(
    user_request: str,
    upload_overrides: Dict[str, Any],
    saved_paths: Dict[str, str],
    output_dir: Path,
    direct_config: Optional[Dict[str, Any]] = None,
) -> None:
    # Create progress container for real-time updates
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0.0, text="Initializing workflow...")
        status_text = st.empty()
        step_expander = st.expander("Workflow Steps", expanded=True)
    
    def update_progress(step: str, progress: float, details: str = ""):
        """Callback to update progress in the UI."""
        progress_bar.progress(progress, text=step)
        if details:
            status_text.info(details)
    
    workflow_config: Dict[str, Any] = {}
    try:
        if direct_config is not None:
            workflow_config = dict(upload_overrides)
            workflow_config.update(direct_config)
            workflow_config["user_request"] = user_request.strip() or workflow_config.get("user_request", "")
            workflow_config["output_dir"] = str(output_dir)
            st.session_state.workflow_config = workflow_config
            update_progress(
                "Using quick configuration",
                0.10,
                f"Detected workflow type: {_detect_workflow_type(workflow_config)}",
            )
        else:
            update_progress("Parsing request with LLM...", 0.05, "Analyzing your natural language request")
            workflow_config = st.session_state.context_agent.parse_request(user_request.strip())
            # Merge upload-driven overrides on top of parsed configuration
            workflow_config.update(upload_overrides)
            workflow_config["user_request"] = user_request.strip()
            workflow_config["output_dir"] = str(output_dir)
            st.session_state.workflow_config = workflow_config

            update_progress(
                "Request parsed successfully",
                0.10,
                f"Detected workflow type: {_detect_workflow_type(workflow_config)}",
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to prepare workflow config: {exc}")
        st.exception(exc)
        return

    try:
        # Show execution plan before running
        with step_expander:
            st.markdown("**Execution Plan:**")
        
        update_progress("Starting workflow execution...", 0.15, "Loading data and preparing inversion")
        
        # Run workflow with progress callback
        results, execution_plan, interpretation, report_files = BaseAgent.run_unified_agent_workflow(
            workflow_config,
            st.session_state.api_key,
            st.session_state.llm_model,
            st.session_state.llm_provider,
            output_dir,
            progress_callback=update_progress,
        )
        
        # Display execution steps
        if execution_plan:
            with step_expander:
                for i, step in enumerate(execution_plan, 1):
                    st.markdown(f"{i}. **{step.get('step', '')}** - {step.get('agent', '')}")
        
        update_progress("Workflow complete!", 1.0, "All steps completed successfully")
        
        st.session_state.workflow_result = {
            "results": results,
            "execution_plan": execution_plan,
            "interpretation": interpretation,
            "report_files": report_files,
            "workflow_config": workflow_config,
            "uploads": saved_paths,
        }
    except Exception as exc:  # noqa: BLE001
        update_progress("Workflow failed", 1.0)
        st.error(f"Workflow failed: {exc}")
        # Try LLM to suggest root cause if context agent available
        suggestion = None
        if st.session_state.context_agent:
            try:
                prompt = (
                    "You are debugging a geophysics workflow error. "
                    f"User request: {user_request}\n"
                    f"Workflow config: {workflow_config}\n"
                    f"Error: {exc}\n"
                    "Suggest concise steps the user should check (file paths, electrode files, instrument type). "
                    "Keep it under 5 bullets."
                )
                suggestion = st.session_state.context_agent.query_llm(prompt)
            except Exception:
                suggestion = None
        if suggestion:
            st.info(f"Suggested checks:\n{suggestion}")
        st.exception(exc)


def _detect_workflow_type(config: Dict) -> str:
    """Detect workflow type from configuration."""
    config_keys = set(config.keys())
    user_request = config.get('user_request', '').lower()
    
    # ERT data processing detection (QC/export)
    processing_keywords = ['data processing', 'quality control', 'qc', 'preprocess', 'export', 'resipy']
    inversion_keywords = ['invert', 'inversion', 'tomography', 'time-lapse', 'timelapse']
    if (config.get('ert_data_processing') or
        (any(kw in user_request for kw in processing_keywords) and not any(kw in user_request for kw in inversion_keywords))):
        return "ERT Data Processing"
    
    # Hydrological model output detection (MODFLOW / ParFlow)
    hydro_keywords = ['modflow', 'parflow', 'par flow', 'hydrological model', 'watercontent', 'saturation', 'porosity']
    mentions_hydro = config.get('hydro_model') or any(kw in user_request for kw in hydro_keywords)
    mentions_geophysics = any(kw in user_request for kw in ['ert', 'seismic', 'tdem', 'inversion', 'forward'])
    if mentions_hydro and not mentions_geophysics:
        return "Hydrological Model Output"
    
    # TDEM detection
    if (config.get('tdem_file') or config.get('tdem_mode') or
        'tdem' in user_request or 'tem ' in user_request or
        'electromagnetic' in user_request):
        return "TDEM Inversion"
    # Seismic-only detection
    elif (config.get('seismic_file') and not config.get('ert_file') or
          config.get('seismic_only') or
          'seismic refraction' in user_request or 'srt inversion' in user_request):
        return "Seismic Refraction Tomography"
    elif 'timelapse_files' in config_keys or 'time_lapse_files' in config_keys:
        return "Time-Lapse ERT"
    elif config.get('velocity_threshold') or (config.get('ert_file') and config.get('seismic_file')):
        return "Data Fusion (Seismic + ERT)"
    elif config.get('ert_file') or config.get('data_file'):
        # Check if water content is requested
        if 'water content' in user_request or 'petrophysic' in user_request or 'moisture' in user_request:
            return "ERT Inversion + Petrophysics"
        return "Direct ERT Inversion"
    return "Unknown"


def render_results() -> None:
    data = st.session_state.workflow_result
    if not data:
        return

    st.success("Workflow complete.")

    interpretation = data.get("interpretation")
    if interpretation:
        st.markdown("### Interpretation")
        st.info(interpretation)

    execution_plan = data.get("execution_plan") or []
    if execution_plan:
        st.markdown("### Execution plan")
        for idx, step in enumerate(execution_plan, 1):
            st.markdown(f"{idx}. **{step.get('step','')}** - {step.get('agent','')}")

    results = data.get("results") or {}
    if results.get("status") == "success":
        st.markdown("### Results summary")
        stats = results.get("statistics", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            if stats.get("resistivity_range"):
                rng = stats["resistivity_range"]
                st.metric("Resistivity range (ohm-m)", f"{rng[0]:.1f} to {rng[1]:.1f}")
        with col2:
            if stats.get("wc_range"):
                rng = stats["wc_range"]
                st.metric("Water content range", f"{rng[0]:.4f} to {rng[1]:.4f}")
        with col3:
            if stats.get("num_cells"):
                st.metric("Mesh cells", stats["num_cells"])
            elif stats.get("n_timesteps"):
                st.metric("Time steps", stats["n_timesteps"])
    elif results:
        st.error(f"Workflow reported an error: {results.get('error','Unknown error')}")

    report_files = data.get("report_files") or {}
    if report_files:
        st.markdown("### Generated files")
        for file_type, file_path in report_files.items():
            path_obj = Path(str(file_path))
            label_map = {
                "report_markdown": "Download report (Markdown)",
                "report_html": "Download report (HTML)",
                "report_pdf": "Download report (PDF)",
            }
            default_label = f"Download {file_type.replace('_', ' ').title()}"
            label = label_map.get(file_type, default_label)
            if path_obj.exists():
                with open(path_obj, "rb") as f:
                    st.download_button(
                        label=label,
                        data=f,
                        file_name=path_obj.name,
                        mime="application/octet-stream",
                    )
            else:
                st.markdown(f"- {file_type}: {path_obj}")

    if data.get("workflow_config"):
        with st.expander("View workflow configuration"):
            st.json(data["workflow_config"])

    if data.get("uploads"):
        with st.expander("Uploaded file locations"):
            st.json(data["uploads"])


def render_cloud_tips() -> None:
    st.markdown("---")
    st.markdown("### Run in the cloud")
    st.markdown(
        """
- Use `streamlit run examples/app_geophysics_workflow.py` inside a container or VM with Python 3.10+ and required libs installed (`pip install -r requirements.txt streamlit`).
- Set API keys as environment variables (`OPENAI_API_KEY`, `GEMINI_API_KEY`, or `ANTHROPIC_API_KEY`) in your cloud platform secrets.
- Persist `results/` by mounting a volume or cloud storage (e.g., S3, Azure Files, GCS) to avoid losing generated reports.
- On Streamlit Community Cloud, add `requirements.txt` and set the working directory to `examples/`; entry point: `streamlit run app_geophysics_workflow.py`.
"""
    )


def render_support_section() -> None:
    """Render the support/donate section."""
    st.markdown(
        """
<div class="phgx-support-card">
    <div class="phgx-free-badge">🎉 FREE & OPEN SOURCE</div>
    <div class="phgx-support-title">Support PyHydroGeophysX Development</div>
    <div class="phgx-support-text">
        This app is developed for <strong>free usage</strong> by the research community.<br>
        If you find it useful, consider supporting better Cloud Services!
    </div>
    <a href="https://venmo.com/Hang-Chen-35" target="_blank" class="phgx-venmo-btn">
        💙 Donate via Venmo @Hang-Chen-35
    </a>
    <div class="phgx-support-text" style="margin-top: 1rem; font-size: 0.85rem;">
        <strong>Need a GPT API key to try?</strong><br>
        Email me at <a href="mailto:hang-chen-1@uiowa.edu" class="phgx-email-link">hang-chen-1@uiowa.edu</a>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_session_state()
    render_header()
    
    # Check for missing dependencies
    if not AGENTS_AVAILABLE:
        st.error(f"""
        ⚠️ **Missing Dependencies**
        
        Some required packages are not installed: `{IMPORT_ERROR}`
        
        Please install the required dependencies:
        ```bash
        pip install pygimli SimPEG openai
        ```
        
        Or use conda for pygimli:
        ```bash
        conda install -c gimli pygimli
        ```
        """)
        st.stop()
    
    if not PYGIMLI_AVAILABLE:
        st.warning("""
        ⚠️ **PyGIMLi Not Available**
        
        ERT inversion and some geophysical functions require PyGIMLi.
        Install with: `conda install -c gimli pygimli` or `pip install pygimli`
        
        You can still use TDEM workflows with SimPEG if available.
        """)
    
    sidebar_state = render_sidebar()

    tab_workflow, tab_hydro_multi, tab_tutorial, tab_concepts, tab_local, tab_author = st.tabs([
        "🚀 Run Workflow",
        "🌊 Hydro → Geophysics",
        "📖 Step-by-Step Tutorials",
        "🔬 Learn Hydrogeophysics & Ask AI",
        "💻 Local Deployment",
        "👤 About Author",
    ])

    with tab_workflow:
        render_workflow_tab(sidebar_state)

    with tab_hydro_multi:
        render_hydro_multigeophys_tab()

    with tab_tutorial:
        render_tutorial_tab()

    with tab_concepts:
        render_concepts_tab()

    with tab_local:
        render_local_deployment_tab()

    with tab_author:
        render_author_tab()
    
    # Render support section in sidebar
    with st.sidebar:
        render_support_section()

if __name__ == "__main__":
    main()
