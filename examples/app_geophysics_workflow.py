"""
PyHydroGeophysX Streamlit Web Application

Natural-language interface for geophysical workflows.
Usage: streamlit run app_geophysics_workflow.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

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
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

EXAMPLE_REQUESTS: Dict[str, str] = {
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
Extract velocity interfaces at: 1200 m/s (regolith-bedrock), 5000 m/s (fractured-fresh)""",
    "TDEM Inversion": """Run a TDEM (Time-Domain Electromagnetic) inversion.
Data file: tdem_synthetic_data.txt
Loop source radius: 10 meters
Number of inversion layers: 20
Use sparse regularization (IRLS): yes
Maximum iterations: 50"""
}

DATA_LINKS: Dict[str, str] = {
    "Example data folder (all)": "https://github.com/geohang/PyHydroGeophysX/tree/main/examples/data",
}

# Example-specific data links organized by workflow type
EXAMPLE_DATA_LINKS: Dict[str, Dict[str, str]] = {
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

STANDARD_ERT_TUTORIAL_IMAGES = [
    ("Step 1", "step1.png"),
    ("Step 2", "Step2.png"),
    ("Final result 1", "Final_result_1.png"),
    ("Final result 2", "Final_result_2.png"),
    ("Resistivity model", "resistivity_model (5).png"),
    ("Water content", "water_content.png"),
]


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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_header() -> None:
    st.markdown('<div class="phgx-header">PyHydroGeophysX Workflows</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="phgx-subtitle-main">AQUAH: Autonomous Query-driven Understanding Agent for Hydrogeophysics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-author-line">'
        '<span class="phgx-version-badge">v1.0</span>'
        '<span class="phgx-author-text">Developed by <a href="https://sites.google.com/view/hangchen" target="_blank">Hang Chen</a> · University of Iowa</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-pill">Unified Workflow</div>'
        '<div class="phgx-pill">ERT</div>'
        '<div class="phgx-pill">TDEM</div>'
        '<div class="phgx-pill">Seismic</div>'
        '<div class="phgx-pill">Data Fusion</div>'
        '<div class="phgx-pill">Climate</div>',
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
                st.image(str(image_path), caption=caption, use_container_width=True)
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
  #phgx-sim-legend { margin-top: 10px; font-size: 12px; color: #475569; line-height: 1.5; padding: 8px 10px; background: #f8fafc; border-radius: 6px; border: 1px solid #e1e5ec; }
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
      text: "Current flows between electrodes A and B. Potential electrodes M and N measure voltage. Current paths curve through subsurface, with deeper penetration at larger electrode spacings.",
      layers: ["Soil (100-500 Ωm)", "Saturated zone (50-150 Ωm)", "Bedrock (500-2000 Ωm)"]
    },
    seismic: {
      title: "Seismic Refraction Tomography (SRT)",
      text: "Seismic waves travel from source to geophones. Direct waves travel along surface; refracted waves travel faster along layer interfaces and arrive first at distant receivers.",
      layers: ["Soil (300-800 m/s)", "Weathered rock (1000-2000 m/s)", "Fresh bedrock (3000-5000 m/s)"]
    },
    tdem: {
      title: "Time-Domain Electromagnetics (TDEM)",
      text: "Transmitter loop creates primary magnetic field. When current is switched off, eddy currents diffuse downward through conductive layers, inducing secondary field measured by receiver.",
      layers: ["Dry soil (high ρ)", "Clay/saturated (low ρ)", "Bedrock (high ρ)"]
    }
  };

  let mode = "ert";
  let t = 0;
  let lastWidth = 0;
  let animFrame = null;

  // Geological model
  const layers = [
    { name: "Air", color: "#e8f4fc", y: 0, h: 0.18 },
    { name: "Topsoil", color: "#c9a66b", y: 0.18, h: 0.12, pattern: "dots" },
    { name: "Saturated Zone", color: "#7cb5d4", y: 0.30, h: 0.22, pattern: "water" },
    { name: "Weathered Bedrock", color: "#a08060", y: 0.52, h: 0.18, pattern: "cracks" },
    { name: "Fresh Bedrock", color: "#706050", y: 0.70, h: 0.30, pattern: "solid" }
  ];

  function resize() {
    const width = Math.max(300, container.clientWidth - 28);
    const height = 280;
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
    legend.innerHTML = `<div class="legend-title">${data.title}</div>${data.text}<br><small><b>Layers:</b> ${data.layers.join(" → ")}</small>`;
  }

  buttons.forEach(btn => btn.addEventListener("click", () => setMode(btn.dataset.mode)));

  function drawLayers() {
    const w = canvas.width, h = canvas.height;
    const groundY = h * 0.18;

    // Draw each layer
    layers.forEach(layer => {
      const ly = h * layer.y;
      const lh = h * layer.h;
      ctx.fillStyle = layer.color;
      ctx.fillRect(0, ly, w, lh);

      // Add texture patterns
      if (layer.pattern === "dots") {
        ctx.fillStyle = "#b8956055";
        for (let i = 0; i < 40; i++) {
          const px = Math.random() * w;
          const py = ly + Math.random() * lh;
          ctx.beginPath();
          ctx.arc(px, py, 1.5, 0, Math.PI * 2);
          ctx.fill();
        }
      } else if (layer.pattern === "water") {
        ctx.strokeStyle = "#5a9fc466";
        ctx.lineWidth = 1;
        for (let i = 0; i < 8; i++) {
          const wy = ly + 8 + i * (lh / 8);
          ctx.beginPath();
          for (let x = 0; x < w; x += 4) {
            const y = wy + Math.sin(x * 0.03 + t * 0.5 + i) * 2;
            x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      } else if (layer.pattern === "cracks") {
        ctx.strokeStyle = "#60503055";
        ctx.lineWidth = 1;
        for (let i = 0; i < 12; i++) {
          const cx = (i * w / 12) + 20;
          ctx.beginPath();
          ctx.moveTo(cx, ly);
          ctx.lineTo(cx + (Math.random() - 0.5) * 15, ly + lh);
          ctx.stroke();
        }
      }
    });

    // Ground surface with grass
    ctx.strokeStyle = "#2d5016";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    for (let x = 0; x < w; x += 2) {
      ctx.lineTo(x, groundY + Math.sin(x * 0.1) * 1.5);
    }
    ctx.stroke();

    // Layer boundaries
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#00000033";
    ctx.lineWidth = 1;
    [0.30, 0.52, 0.70].forEach(y => {
      ctx.beginPath();
      ctx.moveTo(0, h * y);
      ctx.lineTo(w, h * y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    // Depth scale
    ctx.fillStyle = "#475569";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("0m", 5, groundY + 12);
    ctx.fillText("5m", 5, h * 0.40);
    ctx.fillText("15m", 5, h * 0.60);
    ctx.fillText("30m", 5, h * 0.85);

    return groundY;
  }

  function drawERT(groundY) {
    const w = canvas.width, h = canvas.height;
    const nElec = 12;
    const spacing = (w - 60) / (nElec - 1);
    const startX = 30;

    // Electrode array
    for (let i = 0; i < nElec; i++) {
      const x = startX + i * spacing;
      const isActive = (i === 2 || i === 9); // A and B
      const isMeasure = (i === 4 || i === 7); // M and N

      // Electrode stake
      ctx.fillStyle = isActive ? "#dc2626" : (isMeasure ? "#2563eb" : "#64748b");
      ctx.fillRect(x - 2, groundY - 12, 4, 14);
      ctx.beginPath();
      ctx.arc(x, groundY - 14, 4, 0, Math.PI * 2);
      ctx.fill();

      // Labels
      ctx.fillStyle = "#1f2937";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      if (i === 2) ctx.fillText("A", x, groundY - 22);
      if (i === 9) ctx.fillText("B", x, groundY - 22);
      if (i === 4) ctx.fillText("M", x, groundY - 22);
      if (i === 7) ctx.fillText("N", x, groundY - 22);
    }

    // Current flow lines (realistic curved paths)
    const aX = startX + 2 * spacing;
    const bX = startX + 9 * spacing;
    const midX = (aX + bX) / 2;

    ctx.strokeStyle = "rgba(220, 38, 38, 0.5)";
    ctx.lineWidth = 2;

    for (let d = 1; d <= 5; d++) {
      const depth = 20 + d * 35;
      const phase = t * 2 + d * 0.3;
      const intensity = Math.sin(phase) * 0.3 + 0.7;

      ctx.globalAlpha = intensity * 0.6;
      ctx.beginPath();
      ctx.moveTo(aX, groundY);

      // Bezier curve for current path
      const cp1x = aX + (midX - aX) * 0.3;
      const cp1y = groundY + depth * 0.6;
      const cp2x = midX - (midX - aX) * 0.3;
      const cp2y = groundY + depth;
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, midX, groundY + depth);

      const cp3x = midX + (bX - midX) * 0.3;
      const cp3y = groundY + depth;
      const cp4x = bX - (bX - midX) * 0.3;
      const cp4y = groundY + depth * 0.6;
      ctx.bezierCurveTo(cp3x, cp3y, cp4x, cp4y, bX, groundY);
      ctx.stroke();

      // Arrows on current lines
      if (d === 3) {
        const arrowX = midX;
        const arrowY = groundY + depth - 5;
        ctx.beginPath();
        ctx.moveTo(arrowX - 6, arrowY - 4);
        ctx.lineTo(arrowX, arrowY + 4);
        ctx.lineTo(arrowX + 6, arrowY - 4);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;

    // Equipotential lines
    ctx.strokeStyle = "rgba(37, 99, 235, 0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    for (let i = -2; i <= 2; i++) {
      if (i === 0) continue;
      const px = midX + i * 25;
      ctx.beginPath();
      ctx.moveTo(px, groundY);
      ctx.quadraticCurveTo(px + i * 10, groundY + 60, px, groundY + 100);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Voltage measurement indicator
    const mX = startX + 4 * spacing;
    const nX = startX + 7 * spacing;
    const voltY = groundY - 35;
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(mX, groundY - 14);
    ctx.lineTo(mX, voltY);
    ctx.lineTo(nX, voltY);
    ctx.lineTo(nX, groundY - 14);
    ctx.stroke();

    // Voltmeter
    ctx.fillStyle = "#dbeafe";
    ctx.strokeStyle = "#2563eb";
    ctx.lineWidth = 2;
    const vmX = (mX + nX) / 2;
    ctx.beginPath();
    ctx.roundRect(vmX - 18, voltY - 12, 36, 18, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#1e40af";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    const voltage = (Math.sin(t * 2) * 50 + 150).toFixed(0);
    ctx.fillText(voltage + "mV", vmX, voltY + 1);

    info.textContent = "Dipole-Dipole array: Current injection A→B, Voltage measurement M-N";
  }

  function drawSeismic(groundY) {
    const w = canvas.width, h = canvas.height;
    const sourceX = 50;
    const nGeophones = 10;
    const geoSpacing = (w - 100) / (nGeophones - 1);

    // Source (hammer/shot)
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.moveTo(sourceX, groundY - 25);
    ctx.lineTo(sourceX - 8, groundY - 5);
    ctx.lineTo(sourceX + 8, groundY - 5);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#1f2937";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Source", sourceX, groundY - 30);

    // Geophones
    for (let i = 0; i < nGeophones; i++) {
      const gx = 80 + i * geoSpacing;
      ctx.fillStyle = "#059669";
      ctx.beginPath();
      ctx.moveTo(gx, groundY - 2);
      ctx.lineTo(gx - 5, groundY - 12);
      ctx.lineTo(gx + 5, groundY - 12);
      ctx.closePath();
      ctx.fill();
      ctx.fillRect(gx - 1, groundY - 2, 2, 6);
    }

    // Wave propagation
    const waveSpeed = 80;
    const cycleTime = 4;
    const waveT = (t % cycleTime);

    // Direct wave (along surface)
    const directDist = waveT * waveSpeed;
    if (directDist > 0 && directDist < w) {
      ctx.strokeStyle = "#dc262688";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(sourceX, groundY, directDist, -0.15, 0.15);
      ctx.stroke();

      // Wave label
      if (directDist > 50 && directDist < 200) {
        ctx.fillStyle = "#dc2626";
        ctx.font = "9px sans-serif";
        ctx.fillText("Direct", sourceX + directDist * 0.7, groundY - 8);
      }
    }

    // Refracted waves at layer boundaries
    const layer1Y = h * 0.30; // First interface
    const layer2Y = h * 0.52; // Second interface

    // Refracted wave at first interface (faster in saturated zone)
    const refractDelay1 = (layer1Y - groundY) / 40;
    const refractT1 = waveT - refractDelay1;
    if (refractT1 > 0) {
      const refractDist1 = refractT1 * waveSpeed * 1.5;

      // Down-going ray
      ctx.strokeStyle = "#f59e0b88";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sourceX, groundY);
      ctx.lineTo(sourceX + 30, layer1Y);
      ctx.stroke();

      // Critically refracted ray along interface
      if (refractDist1 < w - sourceX) {
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(sourceX + 30, layer1Y);
        ctx.lineTo(sourceX + 30 + refractDist1, layer1Y);
        ctx.stroke();

        // Up-going rays to geophones
        ctx.strokeStyle = "#f59e0b55";
        ctx.lineWidth = 1;
        for (let i = 0; i < nGeophones; i++) {
          const gx = 80 + i * geoSpacing;
          if (gx > sourceX + 30 && gx < sourceX + 30 + refractDist1) {
            ctx.beginPath();
            ctx.moveTo(gx, layer1Y);
            ctx.lineTo(gx, groundY);
            ctx.stroke();
          }
        }

        if (refractDist1 > 80 && refractDist1 < 180) {
          ctx.fillStyle = "#f59e0b";
          ctx.font = "9px sans-serif";
          ctx.fillText("Refracted (V₁)", sourceX + 30 + refractDist1 * 0.5, layer1Y - 5);
        }
      }
    }

    // Refracted wave at second interface (faster in bedrock)
    const refractDelay2 = (layer2Y - groundY) / 35;
    const refractT2 = waveT - refractDelay2;
    if (refractT2 > 0) {
      const refractDist2 = refractT2 * waveSpeed * 2.2;

      ctx.strokeStyle = "#8b5cf688";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(sourceX, groundY);
      ctx.lineTo(sourceX + 50, layer2Y);
      ctx.stroke();

      if (refractDist2 < w - sourceX) {
        ctx.strokeStyle = "#8b5cf6";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(sourceX + 50, layer2Y);
        ctx.lineTo(sourceX + 50 + refractDist2, layer2Y);
        ctx.stroke();

        if (refractDist2 > 60 && refractDist2 < 150) {
          ctx.fillStyle = "#8b5cf6";
          ctx.font = "9px sans-serif";
          ctx.fillText("Refracted (V₂)", sourceX + 50 + refractDist2 * 0.5, layer2Y - 5);
        }
      }
    }

    // Seismogram preview (small)
    const sgX = w - 90, sgY = groundY + 20, sgW = 80, sgH = 60;
    ctx.fillStyle = "#ffffffdd";
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    ctx.fillRect(sgX, sgY, sgW, sgH);
    ctx.strokeRect(sgX, sgY, sgW, sgH);

    ctx.fillStyle = "#1f2937";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Travel-time curve", sgX + 2, sgY + 10);

    // Draw travel time curves
    ctx.strokeStyle = "#dc2626";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(sgX + 5, sgY + 55);
    ctx.lineTo(sgX + sgW - 5, sgY + 25);
    ctx.stroke();

    ctx.strokeStyle = "#f59e0b";
    ctx.beginPath();
    ctx.moveTo(sgX + 25, sgY + 50);
    ctx.lineTo(sgX + sgW - 5, sgY + 30);
    ctx.stroke();

    ctx.strokeStyle = "#8b5cf6";
    ctx.beginPath();
    ctx.moveTo(sgX + 40, sgY + 48);
    ctx.lineTo(sgX + sgW - 5, sgY + 38);
    ctx.stroke();

    info.textContent = "First arrivals: Direct wave (red), Refracted V₁ (orange), Refracted V₂ (purple)";
  }

  function drawTDEM(groundY) {
    const w = canvas.width, h = canvas.height;
    const loopCenterX = w * 0.5;
    const loopWidth = 100;

    // Transmitter loop
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 4;
    ctx.fillStyle = "#0f766e11";
    ctx.beginPath();
    ctx.rect(loopCenterX - loopWidth/2, groundY - 20, loopWidth, 15);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#0f766e";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Tx Loop", loopCenterX, groundY - 25);

    // Receiver coil
    const rxX = loopCenterX;
    ctx.fillStyle = "#7c3aed";
    ctx.beginPath();
    ctx.arc(rxX, groundY - 5, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#7c3aed";
    ctx.font = "9px sans-serif";
    ctx.fillText("Rx", rxX + 15, groundY - 2);

    // EM field propagation (eddy current diffusion)
    const cycleTime = 5;
    const phase = (t % cycleTime) / cycleTime;

    // Primary field (at t=0, during current on)
    if (phase < 0.15) {
      const intensity = 1 - phase / 0.15;
      ctx.strokeStyle = `rgba(15, 118, 110, ${intensity * 0.6})`;
      ctx.lineWidth = 2;

      for (let i = 1; i <= 4; i++) {
        const fieldY = groundY + i * 15;
        ctx.beginPath();
        ctx.moveTo(loopCenterX - loopWidth/2 - i*20, fieldY);
        ctx.quadraticCurveTo(loopCenterX, fieldY + i*8, loopCenterX + loopWidth/2 + i*20, fieldY);
        ctx.stroke();
      }

      ctx.fillStyle = "#0f766e";
      ctx.font = "9px sans-serif";
      ctx.fillText("Primary Field (Tx ON)", loopCenterX, groundY + 85);
    }

    // Eddy currents diffusing downward (after current shutoff)
    if (phase >= 0.15) {
      const diffusePhase = (phase - 0.15) / 0.85;
      const maxDepth = h - groundY - 20;
      const currentDepth = diffusePhase * maxDepth;

      // Draw eddy current rings at different depths
      for (let d = 0; d < 5; d++) {
        const ringDepth = currentDepth - d * 25;
        if (ringDepth < 10 || ringDepth > maxDepth) continue;

        const ringY = groundY + ringDepth;
        const decay = Math.exp(-d * 0.4) * Math.exp(-diffusePhase * 2);
        const ringWidth = 60 + d * 15;

        // Eddy current loops
        ctx.strokeStyle = `rgba(234, 88, 12, ${decay * 0.8})`;
        ctx.lineWidth = 2.5 - d * 0.3;
        ctx.beginPath();
        ctx.ellipse(loopCenterX, ringY, ringWidth, 8, 0, 0, Math.PI * 2);
        ctx.stroke();

        // Current direction arrows
        if (d < 3) {
          const arrowAngle = t * 3 + d;
          const ax = loopCenterX + Math.cos(arrowAngle) * ringWidth;
          const ay = ringY + Math.sin(arrowAngle) * 8;
          ctx.fillStyle = `rgba(234, 88, 12, ${decay})`;
          ctx.beginPath();
          ctx.arc(ax, ay, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Secondary field response (going back up)
      const secIntensity = Math.exp(-diffusePhase * 3) * 0.5;
      ctx.strokeStyle = `rgba(124, 58, 237, ${secIntensity})`;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      for (let i = 1; i <= 3; i++) {
        const fy = groundY - 10 - i * 12;
        ctx.beginPath();
        ctx.moveTo(loopCenterX - 30, fy);
        ctx.quadraticCurveTo(loopCenterX, fy - 8, loopCenterX + 30, fy);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      ctx.fillStyle = "#ea580c";
      ctx.font = "9px sans-serif";
      ctx.fillText("Eddy Currents Diffusing", loopCenterX, groundY + 85);
    }

    // Decay curve display
    const dcX = w - 95, dcY = groundY + 15, dcW = 85, dcH = 55;
    ctx.fillStyle = "#ffffffdd";
    ctx.strokeStyle = "#64748b";
    ctx.lineWidth = 1;
    ctx.fillRect(dcX, dcY, dcW, dcH);
    ctx.strokeRect(dcX, dcY, dcW, dcH);

    ctx.fillStyle = "#1f2937";
    ctx.font = "8px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Decay Curve", dcX + 3, dcY + 10);
    ctx.fillText("log(V)", dcX + 3, dcY + 25);
    ctx.fillText("log(t)", dcX + dcW - 25, dcY + dcH - 3);

    // Decay curve
    ctx.strokeStyle = "#7c3aed";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let x = 0; x < dcW - 15; x++) {
      const tx = x / (dcW - 15);
      const v = Math.exp(-tx * 3) * (dcH - 25);
      const px = dcX + 8 + x;
      const py = dcY + 15 + (dcH - 25) - v;
      x === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Current time indicator on curve
    const cursorX = dcX + 8 + phase * (dcW - 15);
    const cursorV = Math.exp(-phase * 3) * (dcH - 25);
    const cursorY = dcY + 15 + (dcH - 25) - cursorV;
    ctx.fillStyle = "#dc2626";
    ctx.beginPath();
    ctx.arc(cursorX, cursorY, 4, 0, Math.PI * 2);
    ctx.fill();

    info.textContent = "TDEM: Tx current off → Eddy currents diffuse → Secondary field measured at Rx";
  }

  function animate() {
    resize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const groundY = drawLayers();

    if (mode === "ert") drawERT(groundY);
    else if (mode === "seismic") drawSeismic(groundY);
    else if (mode === "tdem") drawTDEM(groundY);

    t += 0.03;
    animFrame = requestAnimationFrame(animate);
  }

  setMode(mode);
  animate();
})();
</script>
</body>
</html>
"""
        components.html(html_sim, height=420)
        st.caption("Interactive visualization showing realistic survey physics. Click tabs to explore different methods.")

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
        if cols[idx % 2].button(question, key=f"example_q_{idx}", use_container_width=True):
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
    ask_clicked = col_ask.button("🔍 Ask AI", type="primary", use_container_width=True)
    clear_clicked = col_clear.button("Clear History", use_container_width=True)

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
        st.warning("Initialize the system from the sidebar before running. You can still draft your request below.")

    st.markdown("---")
    st.subheader("Upload data (optional)")
    st.caption("Upload any data files here; the app will map them by filename. Otherwise, just reference paths in your description.")
    uploaded_files = st.file_uploader(
        "Data files",
        accept_multiple_files=True,
        type=["ohm", "dat", "data", "txt", "sgy", "segy"],
        help="Single upload area for ERT, seismic, electrodes, etc.",
    )

    st.markdown("---")
    run_clicked = st.button("Run workflow", type="primary", use_container_width=True)

    if run_clicked:
        if not request_text.strip():
            st.error("Please describe your workflow.")
            return
        if not st.session_state.context_agent:
            st.error("Initialize the system in the sidebar before running.")
            return

        output_path = Path(sidebar_state["output_dir"]).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        # Parse uploads
        upload_overrides: Dict[str, str] = {}
        saved_paths = handle_uploads(output_path, uploaded_files, upload_overrides)

        # Merge uploaded workflow overrides into the text-derived config during run_workflow
        st.session_state.workflow_config = upload_overrides

        run_workflow(request_text, upload_overrides, saved_paths, output_path)

    if st.session_state.workflow_result:
        st.markdown("---")
        render_results()


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
    init_clicked = col_a.button("Initialize", type="primary", use_container_width=True)
    reset_clicked = col_b.button("Reset state", use_container_width=True)

    if reset_clicked:
        for key in ["context_agent", "workflow_result", "workflow_config", "upload_dir"]:
            st.session_state[key] = None
        st.session_state.user_request = ""
        st.sidebar.info("Session cleared.")

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
    workflow_config: Dict[str, str],
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
    
    # Data candidates exclude electrode files, seismic files, and TDEM files
    data_candidates = [
        p for p in all_paths
        if p.suffix.lower() in [".ohm", ".data", ".dat", ".txt"]
        and not is_electrode(p)
        and "seis" not in p.name.lower()
        and not is_tdem(p)
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

    # Expose all uploads for downstream agents
    workflow_config["uploaded_files"] = {p.name: str(p) for p in all_paths}
    workflow_config["output_dir"] = str(output_dir)
    return saved_paths


def run_workflow(
    user_request: str,
    upload_overrides: Dict[str, str],
    saved_paths: Dict[str, str],
    output_dir: Path,
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
    
    try:
        update_progress("Parsing request with LLM...", 0.05, "Analyzing your natural language request")
        workflow_config = st.session_state.context_agent.parse_request(user_request.strip())
        # Merge upload-driven overrides on top of parsed configuration
        workflow_config.update(upload_overrides)
        workflow_config["user_request"] = user_request.strip()
        workflow_config["output_dir"] = str(output_dir)
        st.session_state.workflow_config = workflow_config
        
        update_progress("Request parsed successfully", 0.10, f"Detected workflow type: {_detect_workflow_type(workflow_config)}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to parse request: {exc}")
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

    tab_workflow, tab_tutorial, tab_concepts, tab_local, tab_author = st.tabs([
        "🚀 Run Workflow",
        "📖 Step-by-Step Tutorials",
        "🔬 Learn Hydrogeophysics & Ask AI",
        "💻 Local Deployment",
        "👤 About Author",
    ])

    with tab_workflow:
        render_workflow_tab(sidebar_state)

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


