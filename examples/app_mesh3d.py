"""
3D Mesh Builder – Interactive Streamlit App
============================================
An interactive GUI for creating 3D meshes for ERT forward modeling and inversion.

The mesh is built by :func:`PyHydroGeophysX.core.mesh_3d.generate_mesh`, the
builder the desktop studio's 3D mesh page uses, so both offer the same engines
(PyGIMLi prism, structured grid, Gmsh tetrahedra, and meshes laid out the way
E4D lays them out), the same zones - boxes of known or assumed resistivity that
the mesh can follow and treat as regions of their own - and the same files.

Usage
-----
    streamlit run examples/app_mesh3d.py
or via the launcher:
    python -m PyHydroGeophysX.gui_mesh3d
"""

from __future__ import annotations

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
    from PyHydroGeophysX.core.mesh_3d import (
        E4D_DEFAULTS,
        E4D_ENGINE,
        build_electrodes,
        find_gmsh_binary,
        generate_mesh,
        mesh_summary,
        normalize_box_zones,
        save_outputs,
        topography_function,
        zone_region,
    )
    MESH3D_AVAILABLE = True
except Exception as _e:
    MESH3D_AVAILABLE = False
    _MESH3D_ERROR = str(_e)
    E4D_ENGINE = "E4D (Triangle + TetGen)"
    E4D_DEFAULTS = {
        "e4d_fine_padding": 1.0, "e4d_fine_depth_padding": 1.0, "e4d_fine_volume": 1.0,
        "e4d_outer_distance": 100.0, "e4d_bottom_depth": 150.0, "e4d_quality": 1.28,
        "e4d_refine_offset": 0.01, "e4d_conductivity": 0.1,
    }

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from PyHydroGeophysX.visualization.axis_units import length_factor, length_label

#: The engines of the desktop studio's 3D mesh page, in its order.
MESH_ENGINES = ["Auto", "Gmsh (tetrahedral)", "PyGIMLi prism", "Structured grid", E4D_ENGINE]

#: One row per zone in the zone table.
ZONE_COLUMNS = ["name", "x_min", "x_max", "y_min", "y_max", "z_bottom", "z_top", "resistivity"]

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


def _e4d_mesher_status() -> str:
    """Which mesher the E4D engine will use here, said before it is used."""
    try:
        from PyHydroGeophysX.core import e4d_mesh
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return f"The E4D engine is unavailable: {exc}"
    program = e4d_mesh.find_tetgen()
    if program:
        return f"TetGen program: {program}"
    package = e4d_mesh._tetgen_module()
    if package is not None:
        return f"TetGen: Python package {getattr(package, '__version__', '')}"
    if e4d_mesh._find_gmsh():
        return "TetGen is not installed, so Gmsh stands in (pip install tetgen for E4D's own mesher)."
    return "Neither TetGen nor Gmsh is installed, so this engine cannot mesh here: pip install tetgen."


def _zones_from_table(table: pd.DataFrame) -> list[dict]:
    """The zone table's rows as zones; raises ValueError naming the zone at fault.

    A row with no numbers at all is one being filled in and is skipped.
    """
    zones = []
    for _, row in table.iterrows():
        if all(pd.isna(row.get(column)) for column in ZONE_COLUMNS[1:]):
            continue
        name = row.get("name")
        name = "" if name is None or (not isinstance(name, str) and pd.isna(name)) else str(name).strip()
        zones.append({
            "name": name or f"Zone {len(zones) + 1}",
            "x": [row.get("x_min"), row.get("x_max")],
            "y": [row.get("y_min"), row.get("y_max")],
            "z": [row.get("z_bottom"), row.get("z_top")],
            "resistivity": row.get("resistivity"),
        })
    return normalize_box_zones(zones)


# ===========================================================================
# SIDEBAR
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    # ------------------------------------------------------------------ mesh type
    st.subheader("Mesh Type")
    mesh_type = st.radio(
        "Type",
        ["Surface with topography", "Box mesh"],
        help="The domain: a block under the ground surface, or a box with a flat top.",
    )

    # ------------------------------------------------------------------ engine
    mesh_engine = st.selectbox(
        "Mesh engine",
        MESH_ENGINES,
        help=(
            "Auto: prism for a surface grid with topography, structured grid otherwise.\n"
            "Gmsh (tetrahedral): refined tetrahedra, flat top; falls back to the structured "
            "grid when Gmsh is missing or fails.\n"
            "PyGIMLi prism: follows the topography (surface grids only).\n"
            "Structured grid: fast regular grid with a node at every sensor.\n"
            "E4D (Triangle + TetGen): a fine zone around the electrodes inside a far-reaching "
            "outer zone, written as an E4D .cfg that E4D itself runs on."
        ),
    )
    if mesh_engine == "Gmsh (tetrahedral)" and MESH3D_AVAILABLE and not find_gmsh_binary():
        st.caption("Gmsh was not found (install Gmsh, or resipy, which bundles it), so the "
                   "structured grid will stand in.")
    single_region = st.checkbox(
        "Single region (one marker)", value=False,
        help="Collapse the mesh to one marker instead of the parameter (2) / boundary (1) split.",
    )

    e4d_options: dict = {}
    if mesh_engine == E4D_ENGINE:
        with st.expander("E4D engine", expanded=True):
            if mesh_type == "Box mesh":
                st.caption("E4D meshes no box: it builds on flat ground at elevation 0.")
            e4d_config_path = st.text_input(
                "E4D .cfg to build from (optional)", value="",
                help="An existing E4D mesh configuration; it replaces the sensor array, "
                     "domain and refinement set here.",
            )
            c1, c2 = st.columns(2)
            with c1:
                e4d_pad = st.number_input("Fine-zone padding (m)", 0.0, 1000.0,
                                          float(E4D_DEFAULTS["e4d_fine_padding"]), step=0.5)
                e4d_fine_volume = st.number_input("Fine element volume (m³)", 0.001, 1.0e6,
                                                  float(E4D_DEFAULTS["e4d_fine_volume"]), step=0.1)
                e4d_bottom = st.number_input("Mesh bottom depth (m)", 1.0, 1.0e6,
                                             float(E4D_DEFAULTS["e4d_bottom_depth"]), step=10.0)
                e4d_refine = st.number_input("Refinement offset (m)", 0.0, 10.0,
                                             float(E4D_DEFAULTS["e4d_refine_offset"]), step=0.01,
                                             format="%.3f")
            with c2:
                e4d_depth_pad = st.number_input("Fine-zone depth padding (m)", 0.0, 1000.0,
                                                float(E4D_DEFAULTS["e4d_fine_depth_padding"]),
                                                step=0.5)
                e4d_outer = st.number_input("Outer boundary distance (m)", 1.0, 1.0e6,
                                            float(E4D_DEFAULTS["e4d_outer_distance"]), step=10.0)
                e4d_quality = st.number_input("TetGen quality", 1.0, 3.0,
                                              float(E4D_DEFAULTS["e4d_quality"]), step=0.01)
                e4d_sigma = st.number_input("Starting conductivity (S/m)", 1.0e-6, 100.0,
                                            float(E4D_DEFAULTS["e4d_conductivity"]), step=0.01,
                                            format="%.4f")
            e4d_mesher = st.selectbox(
                "Mesher", ["auto", "tetgen", "gmsh"],
                format_func={"auto": "Auto (TetGen, else Gmsh)", "tetgen": "TetGen",
                             "gmsh": "Gmsh"}.get,
            )
            e4d_tetgen = st.text_input("TetGen program (optional)", value="")
            st.caption(_e4d_mesher_status())
        e4d_options = {
            "e4d_fine_padding": e4d_pad, "e4d_fine_depth_padding": e4d_depth_pad,
            "e4d_fine_volume": e4d_fine_volume, "e4d_outer_distance": e4d_outer,
            "e4d_bottom_depth": e4d_bottom, "e4d_quality": e4d_quality,
            "e4d_refine_offset": e4d_refine, "e4d_conductivity": e4d_sigma,
            "e4d_mesher": e4d_mesher, "e4d_tetgen": e4d_tetgen.strip(),
            "e4d_config_path": e4d_config_path.strip(),
        }

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
    if mesh_type == "Surface with topography":
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

    if mesh_type == "Surface with topography":
        para_depth   = st.number_input("Investigation depth (m)", 1.0, 500.0, 20.0, step=1.0)
        dz_fine      = st.number_input("Fine layer Δz (m)",  0.05, 10.0, 0.5,  step=0.1,
                                       help="Also the vertical cell size of a structured grid "
                                            "under a surface grid.")
        dz_coarse    = st.number_input("Coarse layer Δz (m)", 0.5, 50.0, 2.0,  step=0.5)
        boundary_ext = st.slider("Boundary extension factor", 1.0, 3.0, 1.4, 0.1)
    else:
        para_depth = box_height
        dz_fine = st.number_input("Vertical cell size (m)", 0.05, 10.0, 0.5, step=0.1,
                                  help="The structured grid's vertical cell size under a "
                                       "surface grid.")
        dz_coarse, boundary_ext = 2.0, 1.4

    with st.expander("Boreholes (structured grid and Gmsh)"):
        bh_lateral_pad = st.number_input("Lateral padding (m)", 0.0, 1000.0, 10.0, step=1.0)
        bh_top_pad = st.number_input("Top padding (m)", 0.0, 1000.0, 2.0, step=0.5)
        bh_bottom_pad = st.number_input("Bottom padding (m)", 0.0, 1000.0, 5.0, step=0.5)
        bh_hcell = st.number_input("Horizontal cell size (m)", 0.05, 100.0, 2.0, step=0.5)
        bh_vcell = st.number_input("Vertical cell size (m)", 0.05, 100.0, 1.0, step=0.5)

    st.divider()

    # ------------------------------------------------------------------ zones
    st.subheader("Zones (optional)")
    st.caption(
        "Boxes of known or assumed resistivity - a clay layer, a tank, a plume - in the "
        "mesh's coordinates, z being elevation. A zone takes the inverted cells inside it."
    )
    _editor = getattr(st, "data_editor", None) or getattr(st, "experimental_data_editor", None)
    zone_error = None
    zones: list[dict] = []
    if _editor is None:
        st.caption("Editing zones needs Streamlit 1.23 or newer.")
    else:
        empty_zones = pd.DataFrame({
            column: pd.Series(dtype=str if column == "name" else float) for column in ZONE_COLUMNS
        })
        zone_table = _editor(empty_zones, num_rows="dynamic", key="zone_table")
        if MESH3D_AVAILABLE:
            try:
                zones = _zones_from_table(pd.DataFrame(zone_table))
            except ValueError as exc:
                zone_error = str(exc)
                st.error(zone_error)
    conform_to_zones = st.checkbox(
        "Mesh follows the zone faces", value=True,
        help="Build the mesh so that each zone's faces are cell faces; otherwise a zone takes "
             "the cells whose centre lies inside it.",
    )
    decouple_zones = st.checkbox(
        "Each zone is a region of its own", value=False,
        help="An inversion on the mesh does not smooth across a region boundary.",
    )

    st.divider()

    # ------------------------------------------------------------------ output
    st.subheader("Output")
    output_dir = st.text_input("Output directory", value="./mesh_output")
    mesh_name  = st.text_input("Mesh name",        value="my_3d_mesh")
    export_bms = st.checkbox("Export .bms (PyGIMLi native)", value=True)
    export_vtk = st.checkbox("Export .vtk (ParaView)", value=True)
    export_csv = st.checkbox("Export sensor positions (.csv)", value=True)

    st.divider()

    # ------------------------------------------------------------------ display
    st.subheader("Display")
    # Per session: the package's set_length_unit() is process-wide and would
    # reach every other session this server runs.
    display_unit = st.radio(
        "Length units", ["m", "ft"], horizontal=True, key="length_unit",
        help="Units of the X, Y and Z axes of the 3-D views. Only the plot axes change; "
             "the inputs above, the tables and every exported file stay in metres.",
    )


# ===========================================================================
# Helper functions
# ===========================================================================

def _builder_config() -> dict:
    """The sidebar's sensor and topography settings as the mesh builder's config.

    Topography and electrodes are built by the same functions the desktop
    studio's 3D mesh page uses; this app only names the choices differently.
    """
    topo_names = {"Flat": "Flat", "Linear Tilt": "Linear tilt",
                  "Gaussian Hill": "Gaussian hill", "Custom Expression": "Custom expression"}
    array_names = {"Surface Grid": "Surface grid", "Borehole": "Single borehole",
                   "Crosshole": "Crosshole"}
    config = {
        "output_dir": output_dir,
        "electrode_refinement": elec_refine,
        "boundary_refinement": node_refine,
        "attractor_distance": attractor_dist,
        "mesh_type": mesh_type,
        "array_type": array_names[array_type],
        "topography_type": topo_names[topo_type],
        "nx": nx, "ny": ny, "dx": dx, "dy": dy, "x_offset": x_offset, "y_offset": y_offset,
        "bh_x": bh_x_single, "bh_y": bh_y_single, "boreholes": bh_positions,
        "z_start": z_start, "z_end": z_end, "n_bh_elec": n_bh_elec,
    }
    config.update({key: value for key, value in topo_params.items() if key != "expr"})
    if "expr" in topo_params:
        config["topography_expr"] = topo_params["expr"]
    return config


def _mesh_config() -> dict:
    """Everything ``generate_mesh`` reads, as the desktop page collects it."""
    config = _builder_config()
    config.update({
        "mesh_engine": mesh_engine,
        "single_region": single_region,
        "para_depth": para_depth,
        "dz_fine": dz_fine,
        "dz_coarse": dz_coarse,
        "boundary_extension": boundary_ext,
        "borehole_lateral_padding": bh_lateral_pad,
        "borehole_top_padding": bh_top_pad,
        "borehole_bottom_padding": bh_bottom_pad,
        "borehole_horizontal_cell": bh_hcell,
        "borehole_vertical_cell": bh_vcell,
        # The E4D engine writes its .cfg, .poly and mesh files under this name.
        "e4d_basename": mesh_name,
    })
    if mesh_type == "Box mesh":
        config.update(box_length=box_length, box_width=box_width, box_height=box_height)
    config.update(e4d_options)
    # Zones only when there are any, so a config without them reads as before.
    if zones:
        config["zones"] = zones
        if conform_to_zones:
            config["conform_to_zones"] = True
        if decouple_zones:
            config["decouple_zones"] = True
    return config


def _selected_formats() -> list[str]:
    """The export formats ticked in the sidebar, as ``save_outputs`` names them."""
    return ((["BMS mesh (.bms)"] if export_bms else [])
            + (["VTK mesh (.vtk)"] if export_vtk else [])
            + (["Sensor positions (.csv)"] if export_csv else []))


def _build_topo_func() -> callable | None:
    """Construct a topography callable from the sidebar settings."""
    return topography_function(_builder_config())


def _build_electrodes() -> pd.DataFrame:
    """Compute electrode positions without creating the output directory."""
    return build_electrodes(_builder_config(), create_directory=False)[1]


def _zone_box_traces(zone_list: list[dict], factor: float = 1.0) -> list:
    """The twelve edges of each zone box, for the electrode view, scaled by ``factor``."""
    traces = []
    for zone in zone_list:
        (x0, x1), (y0, y1), (z0, z1) = ([float(v) * factor for v in zone[axis]]
                                        for axis in ("x", "y", "z"))
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        xs, ys, zs = [], [], []
        for z in (z0, z1):                              # bottom and top rings
            xs += [c[0] for c in corners] + [None]
            ys += [c[1] for c in corners] + [None]
            zs += [z] * len(corners) + [None]
        for cx, cy in corners[:4]:                      # vertical edges
            xs += [cx, cx, None]
            ys += [cy, cy, None]
            zs += [z0, z1, None]
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color="royalblue", width=4),
                                   name=f"{zone['name']} ({zone['resistivity']:g} Ωm)"))
    return traces


def _electrode_plotly(elec: pd.DataFrame) -> go.Figure:
    """Build an interactive 3-D scatter of electrode positions, in the chosen length unit."""
    # Plotly has no tick transform, so coordinates are scaled for display only.
    factor = length_factor(display_unit)
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=elec["x"] * factor, y=elec["y"] * factor, z=elec["z"] * factor,
        mode="markers+text",
        marker=dict(size=5, color="red", symbol="circle"),
        text=[str(n) for n in elec["n"]],
        textposition="top center",
        name="Electrodes",
    ))

    # Optionally show topography surface
    if mesh_type == "Surface with topography" and array_type == "Surface Grid":
        tf = _build_topo_func()
        if tf is not None:
            margin = max(dx, dy) * 2
            xg = np.linspace(elec["x"].min() - margin, elec["x"].max() + margin, 40)
            yg = np.linspace(elec["y"].min() - margin, elec["y"].max() + margin, 40)
            XG, YG = np.meshgrid(xg, yg)
            ZG = np.vectorize(tf)(XG, YG)
            fig.add_trace(go.Surface(
                x=XG * factor, y=YG * factor, z=ZG * factor,
                colorscale="earth", opacity=0.35,
                showscale=False, name="Topography",
            ))

    for trace in _zone_box_traces(zones, factor):
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis_title=length_label("X", display_unit),
            yaxis_title=length_label("Y", display_unit),
            zaxis_title=length_label("Z", display_unit),
            aspectmode="data",
        ),
        title=f"{len(elec)} electrode(s)",
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def _download(label: str, path: Path, mime: str = "application/octet-stream") -> None:
    """A download button for a file on disk, or a note that it is missing."""
    if path.exists():
        st.download_button(label, data=path.read_bytes(), file_name=path.name, mime=mime,
                           use_container_width=True)
    else:
        st.caption(f"{path.name} was not written.")


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
            elec_df = _build_electrodes()

            col_plot, col_info = st.columns([3, 1])

            with col_plot:
                st.plotly_chart(_electrode_plotly(elec_df), use_container_width=True)

            with col_info:
                st.metric("Total Electrodes", len(elec_df))
                st.metric("X range (m)", f"{elec_df['x'].min():.1f} – {elec_df['x'].max():.1f}")
                st.metric("Y range (m)", f"{elec_df['y'].min():.1f} – {elec_df['y'].max():.1f}")
                st.metric("Z range (m)", f"{elec_df['z'].min():.2f} – {elec_df['z'].max():.2f}")

            if zones:
                region = zone_region(_mesh_config(), elec_df)
                st.caption(
                    "Zones act on inverted cells inside x {x_min:.1f} to {x_max:.1f}, "
                    "y {y_min:.1f} to {y_max:.1f}, z {z_bottom:.1f} to {z_top:.1f} m."
                    .format(**region)
                )

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
            "Configure your electrode array, engine, zones and mesh parameters in the "
            "sidebar, then press **Generate Mesh**. "
            "Mesh generation may take from a few seconds to several minutes depending on "
            "the engine, the number of electrodes and the refinement settings."
        )

        # Parameter summary
        with st.expander("Parameter summary", expanded=False):
            cfg = {
                "Mesh type": mesh_type,
                "Mesh engine": mesh_engine,
                "Array type": array_type,
                "Electrode refinement (m)": elec_refine,
                "Boundary refinement (m)": node_refine,
                "Attractor distance (m)": attractor_dist,
                "Zones": len(zones),
                "Output directory": output_dir,
                "Mesh name": mesh_name,
            }
            if mesh_type == "Surface with topography":
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
            st.table(pd.DataFrame([(k, str(v)) for k, v in cfg.items()],
                                  columns=["Parameter", "Value"]))

        if st.button("🚀 Generate Mesh", type="primary"):
            if zone_error:
                st.error(f"Fix the zone table first: {zone_error}")
            else:
                log_lines: list[str] = []
                with st.spinner("Running mesh generation …"):
                    try:
                        result = generate_mesh(_mesh_config(), log=log_lines.append)
                        outputs: dict = {}
                        formats = _selected_formats()
                        if formats:
                            # A mesh that generated is kept even when writing it out fails.
                            try:
                                outputs = save_outputs(result["mesh"], result["electrodes"],
                                                       Path(output_dir), mesh_name, formats)
                            except Exception as exc:
                                log_lines.append(f"The mesh generated, but saving it to "
                                                 f"{output_dir} failed: {exc}")
                        # The E4D engine has already written what E4D runs on.
                        for key, path in dict(result.get("e4d_files") or {}).items():
                            outputs[f"e4d_{key}"] = path
                        st.session_state["mesh_result"] = {
                            "mesh": result["mesh"],
                            "electrodes": result["electrodes"],
                            "generator": result["generator"],
                            "zones": list(result.get("zones") or []),
                            "zones_conform": bool(result.get("zones_conform")),
                            "e4d_zones": list(result.get("e4d_zones") or []),
                            "outputs": outputs,
                            "output_dir": output_dir,
                            "mesh_name": mesh_name,
                            "log": log_lines,
                        }
                        st.success(f"✅ Mesh generated: {result['generator']}")
                    except Exception as exc:
                        st.error(f"Mesh generation failed: {exc}")
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc())
                if log_lines:
                    with st.expander("Mesh log", expanded=False):
                        st.code("\n".join(log_lines))

        # Show results if a mesh exists in session state
        if "mesh_result" in st.session_state:
            built = st.session_state["mesh_result"]
            mesh = built["mesh"]
            summary = mesh_summary(mesh)

            st.divider()
            st.subheader("Mesh Statistics")
            st.caption(f"Built by: {built['generator']}")
            cols = st.columns(len(summary))
            for col, (key, val) in zip(cols, summary.items()):
                col.metric(key, val)

            if built["zones"]:
                st.markdown("**Zones** "
                            + ("(the mesh follows their faces)" if built["zones_conform"]
                               else "(by cell centre)"))
                st.dataframe(pd.DataFrame(built["zones"]), use_container_width=True)
                empty = [z["name"] for z in built["zones"] if not z["cells"]]
                if empty:
                    st.warning(f"{', '.join(empty)} took no cell - outside the inverted region, "
                               "or covered by zones later in the list - and has no effect.")
            if built["e4d_zones"]:
                st.markdown("**E4D zones**")
                st.dataframe(pd.DataFrame(built["e4d_zones"]), use_container_width=True)

            # Simple node scatter plot (subsample for performance)
            if PLOTLY_AVAILABLE:
                try:
                    nodes = np.array(mesh.positions())
                    step = max(1, len(nodes) // 3000)
                    # Scaled for display only; the saved mesh stays in metres.
                    sub = nodes[::step] * length_factor(display_unit)
                    fig_m = go.Figure(go.Scatter3d(
                        x=sub[:, 0], y=sub[:, 1], z=sub[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color=sub[:, 2], colorscale="Viridis"),
                        name="Nodes (subsampled)",
                    ))
                    fig_m.update_layout(
                        scene=dict(
                            xaxis_title=length_label("X", display_unit),
                            yaxis_title=length_label("Y", display_unit),
                            zaxis_title=length_label("Z", display_unit),
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

    if "mesh_result" not in st.session_state:
        st.info("Generate a mesh first (in the **Generate Mesh** tab) to enable export.")
    else:
        built = st.session_state["mesh_result"]
        mesh = built["mesh"]
        outputs = built["outputs"]
        abs_out_dir = Path(built["output_dir"]).resolve()
        st.success(f"Files are written to: `{abs_out_dir}`")

        col1, col2 = st.columns(2)

        # .bms file (with its structure sidecar)
        with col1:
            st.markdown("**PyGIMLi native (.bms)**")
            if "bms" in outputs:
                _download("⬇️ Download .bms", Path(outputs["bms"]))
                if outputs.get("mesh_structure"):
                    _download("⬇️ Download mesh structure sidecar", Path(outputs["mesh_structure"]),
                              "application/json")
            elif st.button("Save .bms now", use_container_width=True):
                try:
                    outputs.update(save_outputs(mesh, built["electrodes"], abs_out_dir,
                                                built["mesh_name"], ["BMS mesh (.bms)"]))
                    st.success(f"Saved: {outputs['bms']}")
                except Exception as exc:
                    st.error(str(exc))

        # .vtk file
        with col2:
            st.markdown("**ParaView / VTK (.vtk)**")
            if "vtk" in outputs:
                _download("⬇️ Download .vtk", Path(outputs["vtk"]))
            elif st.button("Save .vtk now", use_container_width=True):
                try:
                    outputs.update(save_outputs(mesh, built["electrodes"], abs_out_dir,
                                                built["mesh_name"], ["VTK mesh (.vtk)"]))
                    st.success(f"Saved: {outputs['vtk']}")
                except Exception as exc:
                    st.error(str(exc))

        e4d_files = {key: path for key, path in outputs.items() if key.startswith("e4d_")}
        if e4d_files:
            st.divider()
            st.markdown("**E4D files** (what E4D itself runs on)")
            for key, path in e4d_files.items():
                _download(f"⬇️ {Path(path).name}", Path(path))

        st.divider()
        st.markdown("**Electrode positions (.csv)**")
        csv_bytes = built["electrodes"].to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download electrode CSV",
            data=csv_bytes,
            file_name=f"{built['mesh_name']}_electrodes.csv",
            mime="text/csv",
        )

        st.divider()
        st.caption(
            "All generated files are also available in the output directory on disk. "
            "Load `.vtk` files in **ParaView** for full 3D visualization. "
            "Load `.bms` files in PyGIMLi with `pg.load('mesh.bms')`."
        )
