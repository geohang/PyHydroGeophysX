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

# Add parent directory to path so local package can be imported when run from examples/
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent

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
}

.phgx-mono {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.9rem;
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
Monte Carlo realizations: 100"""
}


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
        '<div class="phgx-subtitle">AQUAH: Autonomous Query-driven Understanding Agent for Hydrogeophysics</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="phgx-pill">Unified Workflow</div>'
        '<div class="phgx-pill">ERT</div>'
        '<div class="phgx-pill">Seismic</div>'
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
    data_candidates = [
        p for p in all_paths
        if p.suffix.lower() in [".ohm", ".data", ".dat", ".txt"]
        and not is_electrode(p)
        and "seis" not in p.name.lower()
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
    if 'timelapse_files' in config_keys or 'time_lapse_files' in config_keys:
        return "Time-Lapse ERT"
    elif config.get('velocity_threshold') or (config.get('ert_file') and config.get('seismic_file')):
        return "Data Fusion (Seismic + ERT)"
    elif config.get('ert_file') or config.get('data_file'):
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


def main() -> None:
    init_session_state()
    render_header()
    sidebar_state = render_sidebar()

    st.markdown("---")
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

if __name__ == "__main__":
    main()
