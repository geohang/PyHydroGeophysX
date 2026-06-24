import sys, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\hchen117\OneDrive - University of Iowa\Documents\GitHub\PyHydroGeophysX")
from PyHydroGeophysX.qt_apps import mesh3d_builder as B

print("gmsh binary:", B.find_gmsh_binary())
tmp = tempfile.mkdtemp()
mp = dict(electrode_refinement=1.0, boundary_refinement=2.0, attractor_distance=5.0,
          para_depth=20.0, dz_fine=0.8, dz_coarse=2.0, boundary_extension=1.4,
          borehole_lateral_padding=10.0, borehole_bottom_padding=5.0, borehole_top_padding=2.0,
          borehole_horizontal_cell=3.0, borehole_vertical_cell=2.0, output_dir=tmp)
sg = dict(mesh_type="Surface with topography", array_type="Surface grid",
          nx=6, ny=4, dx=5.0, dy=5.0, x_offset=0.0, y_offset=0.0,
          topography_type="Flat", z_flat=0.0, **mp)


def run(label, **over):
    cfg = dict(sg)
    cfg.update(over)
    res = B.generate_mesh(cfg)
    mk = sorted(set(int(c.marker()) for c in res["mesh"].cells()))
    print(f"  {label:32s} cells={res['mesh'].cellCount():>6} markers={mk}  [{res['generator']}]")


run("Auto (prism)", mesh_engine="Auto")
run("PyGIMLi prism", mesh_engine="PyGIMLi prism")
run("Structured grid", mesh_engine="Structured grid")
run("Gmsh (tetrahedral)", mesh_engine="Gmsh (tetrahedral)")
run("Gmsh + single region", mesh_engine="Gmsh (tetrahedral)", single_region=True)
run("Prism + single region", mesh_engine="PyGIMLi prism", single_region=True)
run("Gmsh box", mesh_type="Box mesh", mesh_engine="Gmsh (tetrahedral)",
    box_length=40.0, box_width=25.0, box_height=20.0)
print("ENGINE + SINGLE-REGION OK")
