"""
3D Mesh Builder – Interactive Streamlit App
============================================
An interactive GUI for creating 3D meshes for ERT forward modeling and inversion.

Usage
-----
    streamlit run examples/app_mesh3d.py
or via the launcher:
    python -m PyHydroGeophysX.gui_mesh3d
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import streamlit as st

# ---------------------------------------------------------------------------
# Path setup – allow running directly from the examples/ folder
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# ---------------------------------------------------------------------------
# Optional heavy imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    from PyHydroGeophysX.core.mesh_3d import Mesh3DCreator
    MESH3D_AVAILABLE = True
except Exception as _e:
    MESH3D_AVAILABLE = False
    _MESH3D_ERROR = str(_e)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="3D Mesh Builder | PyHydroGeophysX",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS tweaks
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .metric-label { font-size: 0.85rem; }
    .block-container { padding-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔷 3D Mesh Builder")
st.markdown(
    "**PyHydroGeophysX** — Interactive tool for creating 3D meshes "
    "for ERT forward modeling and inversion. "
    "Configure the mesh in the sidebar, preview in **Electrode View**, "
    "generate in **Generate Mesh**, and export in **Export**."
)

if not MESH3D_AVAILABLE:
    st.error(
        "⚠️ `PyHydroGeophysX.core.mesh_3d` could not be imported. "
        "Mesh generation is disabled, but the electrode preview is still active."
    )

# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # ------------------------------------------------------------------ mesh type
    st.subheader("Mesh Type")
    mesh_type = st.radio(
        "Type",
        ["Surface with Topography (Prism)", "Box Mesh (Simple)"],
        help="Prism mesh follows terrain; Box mesh is a flat rectangular domain.",
    )

    st.divider()

    # ------------------------------------------------------------------ electrode array
    st.subheader("Electrode Array")
    array_type = st.selectbox(
        "Array Type",
        ["Surface Grid", "Borehole", "Crosshole"],
    )

    if array_type == "Surface Grid":
        c1, c2 = st.columns(2)
        with c1:
            nx = int(st.number_input("nx (X count)", 2, 100, 10, help="Number of electrodes in X direction"))
            dx = st.number_input("dx (X spacing, m)", 0.1, 1000.0, 5.0, step=0.5)
            x_offset = st.number_input("X offset (m)", value=0.0, step=1.0)
        with c2:
            ny = int(st.number_input("ny (Y count)", 2, 100, 6, help="Number of electrodes in Y direction"))
            dy = st.number_input("dy (Y spacing, m)", 0.1, 1000.0, 5.0, step=0.5)
            y_offset = st.number_input("Y offset (m)", value=0.0, step=1.0)
        # borehole vars not used
        bh_x_single = bh_y_single = 0.0
        z_start = z_end = 0.0
        n_bh_elec = 10
        bh_positions: list[tuple[float, float]] = []

    elif array_type == "Borehole":
        c1, c2 = st.columns(2)
        with c1:
            bh_x_single = st.number_input("Borehole X (m)", value=0.0)
        with c2:
            bh_y_single = st.number_input("Borehole Y (m)", value=0.0)
        c3, c4 = st.columns(2)
        with c3:
            z_start = st.number_input("Z top (m)", value=0.0)
        with c4:
            z_end = st.number_input("Z bottom (m)", value=-20.0)
        n_bh_elec = int(st.number_input("# Electrodes", 2, 100, 10))
        nx = ny = dx = dy = x_offset = y_offset = 0
        bh_positions = []

    else:  # Crosshole
        n_boreholes = int(st.number_input("# Boreholes", 2, 10, 2))
        bh_positions = []
        for i in range(n_boreholes):
            c1, c2 = st.columns(2)
            with c1:
                _x = st.number_input(f"BH {i+1} X (m)", value=float(i * 10), key=f"bhx_{i}")
            with c2:
                _y = st.number_input(f"BH {i+1} Y (m)", value=0.0, key=f"bhy_{i}")
            bh_positions.append((_x, _y))
        c3, c4 = st.columns(2)
        with c3:
            z_start = st.number_input("Z top (m)", value=0.0)
        with c4:
            z_end = st.number_input("Z bottom (m)", value=-20.0)
        n_bh_elec = int(st.number_input("# Electrodes per borehole", 2, 100, 10))
        nx = ny = dx = dy = x_offset = y_offset = 0
        bh_x_single = bh_y_single = 0.0

    st.divider()

    # ------------------------------------------------------------------ topography
    if mesh_type == "Surface with Topography (Prism)":
        st.subheader("Topography")
        topo_type = st.selectbox(
            "Topography Type",
            ["Flat", "Linear Tilt", "Gaussian Hill", "Custom Expression"],
        )

        if topo_type == "Flat":
            z_flat = st.number_input("Surface elevation (m)", value=0.0, step=1.0)
            topo_params: dict = {"z_flat": z_flat}

        elif topo_type == "Linear Tilt":
            z_base = st.number_input("Base elevation (m)", value=100.0, step=1.0)
            c1, c2 = st.columns(2)
            with c1:
                tilt_x = st.slider("X slope (m/m)", -1.0, 1.0, 0.05, 0.01)
            with c2:
                tilt_y = st.slider("Y slope (m/m)", -1.0, 1.0, 0.0, 0.01)
            topo_params = {"z_base": z_base, "tilt_x": tilt_x, "tilt_y": tilt_y}

        elif topo_type == "Gaussian Hill":
            c1, c2 = st.columns(2)
            with c1:
                hill_base = st.number_input("Base elevation (m)", value=0.0, step=1.0)
                hill_amp = st.number_input("Amplitude (m)", value=5.0, step=0.5)
                hill_sigma = st.number_input("Width σ (m)", value=10.0, step=1.0)
            with c2:
                hill_cx = st.number_input("Center X (m)", value=25.0, step=1.0)
                hill_cy = st.number_input("Center Y (m)", value=15.0, step=1.0)
            topo_params = {
                "hill_base": hill_base, "hill_amp": hill_amp,
                "hill_sigma": hill_sigma, "hill_cx": hill_cx, "hill_cy": hill_cy,
            }

        else:  # Custom Expression
            st.caption(
                "Enter a Python expression using `x`, `y`, and `np` (numpy). "
                "Example: `0.1 * x - 0.05 * y + 100`"
            )
            topo_expr = st.text_input("f(x, y) =", value="0.1*x - 0.05*y + 100")
            topo_params = {"expr": topo_expr}

    else:  # Box mesh
        st.subheader("Box Dimensions")
        box_length = st.number_input("Length X (m)", 1.0, 5000.0, 50.0, step=1.0)
        box_width  = st.number_input("Width  Y (m)", 1.0, 5000.0, 30.0, step=1.0)
        box_height = st.number_input("Depth  Z (m)", 1.0, 1000.0, 25.0, step=1.0)
        topo_type = "Flat"
        topo_params = {"z_flat": 0.0}

    st.divider()

    # ------------------------------------------------------------------ mesh parameters
    st.subheader("Mesh Parameters")
    elec_refine = st.number_input(
        "Electrode refinement (m)", 0.01, 50.0, 0.5, step=0.1,
        help="Target cell size at electrode positions.",
    )
    node_refine = st.number_input(
        "Boundary refinement (m)", 0.1, 100.0, 2.0, step=0.5,
        help="Target cell size at domain boundaries.",
    )
    attractor_dist = st.number_input(
        "Attractor distance (m)", 0.1, 200.0, 5.0, step=0.5,
        help="Distance over which electrode refinement fades to boundary size.",
    )

    if mesh_type == "Surface with Topography (Prism)":
        para_depth   = st.number_input("Investigation depth (m)", 1.0, 500.0, 20.0, step=1.0)
        dz_fine      = st.number_input("Fine layer Δz (m)",  0.05, 10.0, 0.5,  step=0.1)
        dz_coarse    = st.number_input("Coarse layer Δz (m)", 0.5, 50.0, 2.0,  step=0.5)
        boundary_ext = st.slider("Boundary extension factor", 1.0, 3.0, 1.4, 0.1)
    else:
        para_depth = box_height
        dz_fine = dz_coarse = boundary_ext = 0.0

    st.divider()

    # ------------------------------------------------------------------ output
    st.subheader("Output")
    output_dir = st.text_input("Output directory", value="./mesh_output")
    mesh_name  = st.text_input("Mesh name",        value="my_3d_mesh")
    export_bms = st.checkbox("Export .bms (PyGIMLi native)", value=True)
    export_vtk = st.checkbox("Export .vtk (ParaView)", value=True)


# ===========================================================================
# Helper functions
# ===========================================================================

def _build_topo_func() -> callable | None:
    """Construct a topography callable from the sidebar settings."""
    if topo_type == "Flat":
        z0 = topo_params["z_flat"]
        return lambda x, y: float(z0)

    if topo_type == "Linear Tilt":
        zb = topo_params["z_base"]
        tx = topo_params["tilt_x"]
        ty = topo_params["tilt_y"]
        return lambda x, y: zb + tx * x + ty * y

    if topo_type == "Gaussian Hill":
        zb  = topo_params["hill_base"]
        amp = topo_params["hill_amp"]
        sig = topo_params["hill_sigma"]
        cx  = topo_params["hill_cx"]
        cy  = topo_params["hill_cy"]
        return lambda x, y: zb + amp * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sig ** 2))

    # Custom Expression – restricted eval for safety
    expr = topo_params["expr"]
    _allowed = {"np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp,
                "sqrt": np.sqrt, "abs": abs, "pi": np.pi}

    def _custom(x, y):
        try:
            return float(eval(expr, {"__builtins__": {}}, {**_allowed, "x": x, "y": y}))  # noqa: S307
        except Exception:
            return 0.0

    return _custom


def _build_electrodes() -> tuple[Mesh3DCreator, pd.DataFrame]:
    """Instantiate Mesh3DCreator and compute electrode positions."""
    creator = Mesh3DCreator(
        mesh_directory=output_dir,
        elec_refinement=elec_refine,
        node_refinement=node_refine,
        attractor_distance=attractor_dist,
    )

    if array_type == "Surface Grid":
        elec = creator.create_surface_electrode_array(
            nx=nx, ny=ny, dx=dx, dy=dy,
            x_offset=x_offset, y_offset=y_offset, z=0.0,
        )
        if mesh_type == "Surface with Topography (Prism)":
            tf = _build_topo_func()
            if tf is not None:
                elec["z"] = [tf(xi, yi) for xi, yi in zip(elec["x"], elec["y"])]

    elif array_type == "Borehole":
        z_arr = np.linspace(z_start, z_end, n_bh_elec)
        elec = creator.create_borehole_electrode_array(bh_x_single, bh_y_single, z_arr)

    else:  # Crosshole
        z_arr = np.linspace(z_start, z_end, n_bh_elec)
        elec = creator.create_crosshole_electrode_array(bh_positions, z_arr)

    return creator, elec


def _electrode_plotly(elec: pd.DataFrame) -> go.Figure:
    """Build an interactive 3-D scatter of electrode positions."""
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=elec["x"], y=elec["y"], z=elec["z"],
        mode="markers+text",
        marker=dict(size=5, color="red", symbol="circle"),
        text=[str(n) for n in elec["n"]],
        textposition="top center",
        name="Electrodes",
    ))

    # Optionally show topography surface
    if mesh_type == "Surface with Topography (Prism)" and array_type == "Surface Grid":
        tf = _build_topo_func()
        if tf is not None:
            margin = max(dx, dy) * 2
            xg = np.linspace(elec["x"].min() - margin, elec["x"].max() + margin, 40)
            yg = np.linspace(elec["y"].min() - margin, elec["y"].max() + margin, 40)
            XG, YG = np.meshgrid(xg, yg)
            ZG = np.vectorize(tf)(XG, YG)
            fig.add_trace(go.Surface(
                x=XG, y=YG, z=ZG,
                colorscale="earth", opacity=0.35,
                showscale=False, name="Topography",
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        title=f"{len(elec)} electrode(s)",
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _mesh_summary(mesh) -> dict:
    """Extract basic mesh statistics from a PyGIMLi mesh."""
    try:
        return {
            "Cells": mesh.cellCount(),
            "Nodes": mesh.nodeCount(),
            "Boundaries": mesh.boundaryCount(),
            "Dimension": mesh.dim(),
        }
    except Exception:
        return {}


# ===========================================================================
# TABS
# ===========================================================================
tab_elec, tab_gen, tab_export = st.tabs(
    ["📍 Electrode View", "🔲 Generate Mesh", "💾 Export"]
)

# ---------------------------------------------------------------------------
# TAB 1 – Electrode Preview
# ---------------------------------------------------------------------------
with tab_elec:
    st.subheader("Electrode Configuration Preview")

    if not MESH3D_AVAILABLE:
        st.warning("Electrode preview requires PyHydroGeophysX (mesh_3d module).")
    elif not PLOTLY_AVAILABLE:
        st.warning("Install `plotly` for interactive 3D visualization: `pip install plotly`")
    else:
        try:
            _, elec_df = _build_electrodes()

            col_plot, col_info = st.columns([3, 1])

            with col_plot:
                st.plotly_chart(_electrode_plotly(elec_df), use_container_width=True)

            with col_info:
                st.metric("Total Electrodes", len(elec_df))
                st.metric("X range (m)", f"{elec_df['x'].min():.1f} – {elec_df['x'].max():.1f}")
                st.metric("Y range (m)", f"{elec_df['y'].min():.1f} – {elec_df['y'].max():.1f}")
                st.metric("Z range (m)", f"{elec_df['z'].min():.2f} – {elec_df['z'].max():.2f}")

            with st.expander("Electrode positions (full table)"):
                st.dataframe(elec_df, use_container_width=True, height=300)

            # Download electrode CSV
            csv = elec_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download electrodes as CSV",
                data=csv,
                file_name=f"{mesh_name}_electrodes.csv",
                mime="text/csv",
            )

        except Exception as exc:
            st.error(f"Could not generate electrode preview: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# TAB 2 – Mesh Generation
# ---------------------------------------------------------------------------
with tab_gen:
    st.subheader("Mesh Generation")

    if not MESH3D_AVAILABLE:
        st.error("PyHydroGeophysX is required for mesh generation.")
    else:
        st.info(
            "Configure your electrode array and mesh parameters in the sidebar, "
            "then press **Generate Mesh**. "
            "Mesh generation calls PyGIMLi and may take from a few seconds to "
            "several minutes depending on the number of electrodes and refinement settings."
        )

        # Parameter summary
        with st.expander("Parameter summary", expanded=False):
            cfg = {
                "Mesh type": mesh_type,
                "Array type": array_type,
                "Electrode refinement (m)": elec_refine,
                "Boundary refinement (m)": node_refine,
                "Attractor distance (m)": attractor_dist,
                "Output directory": output_dir,
                "Mesh name": mesh_name,
            }
            if mesh_type == "Surface with Topography (Prism)":
                cfg.update({
                    "Investigation depth (m)": para_depth,
                    "Fine Δz (m)": dz_fine,
                    "Coarse Δz (m)": dz_coarse,
                    "Boundary extension": boundary_ext,
                    "Topography": topo_type,
                })
            else:
                cfg.update({
                    "Box length X (m)": box_length,
                    "Box width Y (m)": box_width,
                    "Box depth Z (m)": box_height,
                })
            st.table(pd.DataFrame(cfg.items(), columns=["Parameter", "Value"]))

        if st.button("🚀 Generate Mesh", type="primary"):
            with st.spinner("Running mesh generation …"):
                try:
                    creator, elec_df = _build_electrodes()

                    save_fmts = (
                        (["bms"] if export_bms else []) +
                        (["vtk"] if export_vtk else [])
                    )

                    if mesh_type == "Box Mesh (Simple)":
                        mesh = creator.create_box_mesh(
                            length=box_length,
                            width=box_width,
                            height=box_height,
                            electrode_positions=elec_df,
                            output_name=mesh_name,
                        )
                    else:
                        tf = _build_topo_func()
                        mesh = creator.create_3d_mesh_with_topography(
                            electrode_positions=elec_df,
                            topography_func=tf,
                            para_depth=para_depth,
                            dz_fine=dz_fine,
                            dz_coarse=dz_coarse,
                            boundary_extension=boundary_ext,
                            use_prism_mesh=True,
                        )

                    st.session_state["mesh"] = mesh
                    st.session_state["mesh_name"] = mesh_name
                    st.session_state["output_dir"] = output_dir
                    st.session_state["save_fmts"] = save_fmts

                    st.success("✅ Mesh generated successfully!")

                except Exception as exc:
                    st.error(f"Mesh generation failed: {exc}")
                    with st.expander("Full traceback"):
                        st.code(traceback.format_exc())

        # Show results if mesh exists in session state
        if "mesh" in st.session_state:
            mesh = st.session_state["mesh"]
            summary = _mesh_summary(mesh)

            st.divider()
            st.subheader("Mesh Statistics")
            cols = st.columns(len(summary))
            for col, (key, val) in zip(cols, summary.items()):
                col.metric(key, val)

            # Simple node scatter plot (subsample for performance)
            if PLOTLY_AVAILABLE:
                try:
                    import pygimli as pg
                    nodes = np.array(mesh.positions())
                    step = max(1, len(nodes) // 3000)
                    sub = nodes[::step]
                    fig_m = go.Figure(go.Scatter3d(
                        x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color=sub[:, 2], colorscale="Viridis"),
                        name="Nodes (subsampled)",
                    ))
                    fig_m.update_layout(
                        scene=dict(
                            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
                            aspectmode="data",
                        ),
                        title="Mesh nodes (subsampled for display)",
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )
                    st.plotly_chart(fig_m, use_container_width=True)
                except Exception:
                    st.info("3D mesh node preview unavailable (requires pygimli).")


# ---------------------------------------------------------------------------
# TAB 3 – Export
# ---------------------------------------------------------------------------
with tab_export:
    st.subheader("Export")

    if "mesh" not in st.session_state:
        st.info("Generate a mesh first (in the **Generate Mesh** tab) to enable export.")
    else:
        mesh        = st.session_state["mesh"]
        _mesh_name  = st.session_state.get("mesh_name", mesh_name)
        _out_dir    = st.session_state.get("output_dir", output_dir)
        _save_fmts  = st.session_state.get("save_fmts", [])

        abs_out_dir = Path(_out_dir).resolve()
        st.success(f"Mesh saved to: `{abs_out_dir}`")

        col1, col2 = st.columns(2)

        # .bms file
        bms_path = abs_out_dir / f"{_mesh_name}.bms"
        with col1:
            st.markdown("**PyGIMLi native (.bms)**")
            if bms_path.exists():
                bms_bytes = bms_path.read_bytes()
                st.download_button(
                    "⬇️ Download .bms",
                    data=bms_bytes,
                    file_name=bms_path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                if st.button("Save .bms now", use_container_width=True):
                    try:
                        abs_out_dir.mkdir(parents=True, exist_ok=True)
                        mesh.save(str(bms_path))
                        st.success(f"Saved: {bms_path}")
                    except Exception as exc:
                        st.error(str(exc))

        # .vtk file
        vtk_path = abs_out_dir / f"{_mesh_name}.vtk"
        with col2:
            st.markdown("**ParaView / VTK (.vtk)**")
            if vtk_path.exists():
                vtk_bytes = vtk_path.read_bytes()
                st.download_button(
                    "⬇️ Download .vtk",
                    data=vtk_bytes,
                    file_name=vtk_path.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            else:
                if st.button("Save .vtk now", use_container_width=True):
                    try:
                        abs_out_dir.mkdir(parents=True, exist_ok=True)
                        mesh.exportVTK(str(vtk_path))
                        st.success(f"Saved: {vtk_path}")
                    except Exception as exc:
                        st.error(str(exc))

        st.divider()
        st.markdown("**Electrode positions (.csv)**")
        try:
            _, elec_df = _build_electrodes()
            csv_bytes = elec_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download electrode CSV",
                data=csv_bytes,
                file_name=f"{_mesh_name}_electrodes.csv",
                mime="text/csv",
            )
        except Exception:
            pass

        st.divider()
        st.caption(
            "All generated files are also available in the output directory on disk. "
            "Load `.vtk` files in **ParaView** for full 3D visualization. "
            "Load `.bms` files in PyGIMLi with `pg.load('mesh.bms')`."
        )
