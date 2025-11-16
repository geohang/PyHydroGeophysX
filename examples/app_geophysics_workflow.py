"""
PyHydroGeophysX Streamlit Web Application

Natural language interface for geophysical workflows.
Simply describe your workflow and upload data to get results automatically!

Usage: streamlit run app_geophysics_workflow.py
"""

import streamlit as st
import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from PyHydroGeophysX.agents import BaseAgent, ContextInputAgent

# Page configuration
st.set_page_config(
    page_title="PyHydroGeophysX - Geophysical Workflows",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .workflow-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'context_agent' not in st.session_state:
    st.session_state.context_agent = None
if 'workflow_result' not in st.session_state:
    st.session_state.workflow_result = None
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = None
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = None

# Header
st.markdown('<div class="main-header">🌍 PyHydroGeophysX</div>', unsafe_allow_html=True)
st.markdown("### AQUAH (Autonomous Query-driven Understanding Agent for Hydrogeophysics)")

# Sidebar - Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # LLM Provider selection
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini", "claude"],
        index=0,
        help="Select your LLM provider"
    )
    
    llm_model = st.text_input(
        "Model Name",
        value="gpt-4o-mini" if llm_provider == "openai" else "gemini-pro",
        help="Specify the model to use"
    )
    
    # API Key input
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your API key for the selected provider"
    )
    
    # Output directory
    output_dir = st.text_input(
        "Output Directory",
        value="results/streamlit_workflow",
        help="Where to save results"
    )
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        if not api_key:
            st.error("Please enter an API key!")
        else:
            try:
                with st.spinner("Initializing system..."):
                    # Initialize context agent for natural language parsing
                    st.session_state.context_agent = ContextInputAgent(
                        api_key=api_key,
                        model=llm_model,
                        llm_provider=llm_provider
                    )
                    
                    # Store configuration
                    st.session_state.api_key = api_key
                    st.session_state.llm_model = llm_model
                    st.session_state.llm_provider = llm_provider
                
                st.success("✅ System initialized successfully!")
                st.info("✓ Context agent ready\n✓ Using BaseAgent.run_unified_agent_workflow()\n✓ Workflow type will be auto-detected!")
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
    
    # System status
    st.markdown("---")
    st.header("📊 System Status")
    if st.session_state.context_agent:
        st.success("✅ Ready")
    else:
        st.warning("⚠️ Not initialized")

# Main content
if not st.session_state.context_agent:
    st.info("👈 Please configure and initialize the system using the sidebar")
    
    # Show example workflows
    st.markdown("---")
    st.header("📚 Example Workflows")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Standard ERT**
        ```
        We have ERT data from DAS-1 at
        examples/data/ERT/DAS/20171105_1418.Data
        with electrode file at
        examples/data/ERT/DAS/electrodes.dat
        Use petrophysical parameters:
        rho_sat=541, porosity=0.37, n=1.24
        ```
        """)
    
    with col2:
        st.markdown("""
        **Time-Lapse ERT**
        ```
        Run TIME-LAPSE ERT on 4 E4D files:
        2022-03-26_0030.ohm (baseline)
        2022-04-26_0030.ohm
        2022-05-26_0030.ohm
        2022-06-26_0030.ohm
        Temporal regularization: 10
        Fetch climate data for Mt. Snodgrass
        (38.92584°N, -106.97998°W)
        ```
        """)
    
    with col3:
        st.markdown("""
        **Data Fusion**
        ```
        Use seismic at data/Seismic/srtfieldline2.dat
        with threshold 1000 m/s to constrain
        ERT at data/ERT/Bert/fielddataline2.dat
        Layer-specific petrophysics:
        - Regolith: rho_sat 50-250, n 1.3-2.2
        - Bedrock: rho_sat 165-350, n 2.0-2.2
        Monte Carlo: 100 realizations
        ```
        """)

else:
    # Main workflow interface
    st.header("📝 Describe Your Workflow")
    
    # Natural language input
    user_request = st.text_area(
        "Describe what you want to do in plain English:",
        height=150,
        placeholder="Example: I want to run a time-lapse ERT inversion on 5 datasets...",
        help="Be specific about data files, parameters, and desired outputs"
    )
    
    # File upload section
    st.markdown("---")
    st.header("📁 Upload Data (Optional)")
    st.info("💡 You can either specify file paths in your description OR upload files here. Uploaded files will override paths in your description.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ert_files = st.file_uploader(
            "ERT Data Files",
            accept_multiple_files=True,
            type=['ohm', 'dat', 'data', 'Data'],
            help="Upload ERT measurement files. Multiple files = time-lapse workflow"
        )
    
    with col2:
        seismic_files = st.file_uploader(
            "Seismic Data (for data fusion)",
            accept_multiple_files=False,
            type=['dat', 'txt'],
            help="Upload seismic travel time data for structure-constrained ERT"
        )
    
    with col3:
        electrode_file = st.file_uploader(
            "Electrode File (optional)",
            accept_multiple_files=False,
            type=['dat', 'txt'],
            help="Upload electrode positions for topography support"
        )
    
    # Process button
    st.markdown("---")
    if st.button("🚀 Run Workflow", type="primary", use_container_width=True):
        if not user_request:
            st.error("Please describe your workflow!")
        else:
            try:
                with st.spinner("🔍 Analyzing your request..."):
                    # Parse natural language
                    workflow_config = st.session_state.context_agent.parse_request(user_request)
                    
                    # Handle uploaded files
                    temp_dir = Path(tempfile.mkdtemp()) if (ert_files or seismic_files or electrode_file) else None
                    
                    if ert_files:
                        ert_file_paths = []
                        for ert_file in ert_files:
                            file_path = temp_dir / ert_file.name
                            with open(file_path, 'wb') as f:
                                f.write(ert_file.read())
                            ert_file_paths.append(str(file_path))
                        
                        if len(ert_file_paths) == 1:
                            workflow_config['data_file'] = ert_file_paths[0]
                            workflow_config['ert_file'] = ert_file_paths[0]  # For data fusion
                        else:
                            workflow_config['time_lapse_files'] = ert_file_paths
                            workflow_config['timelapse_files'] = ert_file_paths  # Alternative key
                    
                    if seismic_files:
                        seismic_path = temp_dir / seismic_files.name
                        with open(seismic_path, 'wb') as f:
                            f.write(seismic_files.read())
                        workflow_config['seismic_file'] = str(seismic_path)
                    
                    if electrode_file:
                        electrode_path = temp_dir / electrode_file.name
                        with open(electrode_path, 'wb') as f:
                            f.write(electrode_file.read())
                        workflow_config['electrode_file'] = str(electrode_path)
                    
                    # Run unified workflow - automatically detects type and executes!
                    with st.spinner("🚀 Running unified workflow (auto-detecting type)..."):
                        results, execution_plan, interpretation, report_files = BaseAgent.run_unified_agent_workflow(
                            workflow_config,
                            st.session_state.api_key,
                            st.session_state.llm_model,
                            st.session_state.llm_provider,
                            Path(output_dir)
                        )
                    
                    # Store results
                    st.session_state.workflow_result = {
                        'results': results,
                        'execution_plan': execution_plan,
                        'interpretation': interpretation,
                        'report_files': report_files
                    }
                
                # Display results
                if st.session_state.workflow_result:
                    result = st.session_state.workflow_result
                    
                    st.success("✅ Workflow Complete!")
                    
                    # Interpretation
                    if result.get('interpretation'):
                        st.markdown("### 💡 Interpretation")
                        st.info(result['interpretation'])
                    
                    # Execution plan
                    if result.get('execution_plan'):
                        st.markdown("### 📋 Execution Plan")
                        for i, step in enumerate(result['execution_plan'], 1):
                            st.markdown(f"{i}. **{step['step']}** → {step['agent']}")
                    
                    # Execution results
                    results = result.get('results', {})
                    
                    if results.get('status') == 'success':
                        st.markdown("---")
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown("### ✅ Workflow Executed Successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display workflow-specific results
                        if 'statistics' in results:
                            st.markdown("### 📊 Results Summary")
                            stats = results['statistics']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if 'resistivity_range' in stats:
                                    st.metric(
                                        "Resistivity Range (Ωm)",
                                        f"{stats['resistivity_range'][0]:.1f} - {stats['resistivity_range'][1]:.1f}"
                                    )
                            
                            with col2:
                                if 'wc_range' in stats:
                                    st.metric(
                                        "Water Content Range",
                                        f"{stats['wc_range'][0]:.4f} - {stats['wc_range'][1]:.4f}"
                                    )
                            
                            with col3:
                                if 'num_cells' in stats:
                                    st.metric("Mesh Cells", stats['num_cells'])
                                elif 'n_timesteps' in stats:
                                    st.metric("Time Steps", stats['n_timesteps'])
                        
                        # Display generated files
                        if result.get('report_files'):
                            st.markdown("### 📁 Generated Files")
                            for file_type, file_path in result['report_files'].items():
                                file_path_str = str(file_path)
                                if Path(file_path_str).exists():
                                    # Create download button for files
                                    with open(file_path_str, 'rb') as f:
                                        st.download_button(
                                            label=f"📥 Download {file_type}",
                                            data=f,
                                            file_name=Path(file_path_str).name,
                                            mime="application/octet-stream"
                                        )
                                else:
                                    st.markdown(f"- **{file_type}**: `{file_path}`")
                    else:
                        st.error(f"❌ Workflow failed: {results.get('error', 'Unknown error')}")
                    
                    # Show configuration
                    with st.expander("🔧 View Generated Configuration"):
                        st.json(workflow_config)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p>PyHydroGeophysX - Automated Geophysical Data Processing</p>
    <p>Powered by Multi-Agent System with LLM Intelligence</p>
</div>
""", unsafe_allow_html=True)

