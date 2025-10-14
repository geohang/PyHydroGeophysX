# %%
"""
Ex . Seismic Refraction Tomography (SRT) Inversion
====================================================
Ex. Seismic Refraction Tomography (SRT) Inversion and Interface Delineation
This example demonstrates how to perform a 2D seismic refraction tomography (SRT) 
inversion and interpret the results to define subsurface structures.

The script focuses on the inversion and post-processing stages of a geophysical workflow. 
It begins by loading pre-existing synthetic travel time data and then uses tomographic inversion 
to reconstruct the subsurface P-wave velocity distribution. A key feature demonstrated is the extraction of geological interfaces based on velocity thresholds.

The workflow includes:
Loading Synthetic Data: The script loads pre-generated synthetic seismic travel time data for both a long and a short survey profile.
Tomographic Inversion: It creates an unstructured mesh and performs a travel time inversion to estimate the subsurface P-wave velocity model for each dataset.
Visualization: The resulting velocity tomograms are plotted, showing the recovered subsurface structure.
Interface Extraction: For the long profile, the script uses the extract_velocity_interface function to automatically delineate boundaries between different geological layers (e.g., regolith, fractured bedrock, and fresh bedrock) based on velocity thresholds.
Exporting Results: The coordinates of the extracted interfaces are saved to text files, making them available for constraining other models, such as hydrogeological simulations.

"""



# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.physics import ert
from pygimli.physics import TravelTimeManager
import pygimli.physics.traveltime as tt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pygimli.meshtools as mt

# Setup package path for development
try:
    # For regular Python scripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # For Jupyter notebooks
    current_dir = os.getcwd()

# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import PyHydroGeophysX modules
from PyHydroGeophysX.model_output.modflow_output import MODFLOWWaterContent
from PyHydroGeophysX.core.interpolation import ProfileInterpolator, create_surface_lines
from PyHydroGeophysX.core.mesh_utils import MeshCreator,fill_holes_2d,createTriangles,extract_velocity_interface
from PyHydroGeophysX.petrophysics.velocity_models import HertzMindlinModel, DEMModel

# %%
output_dir = "results/seismic_example"
os.makedirs(output_dir, exist_ok=True)

# %% [markdown]
# ## Long seismic profile

# %% [markdown]
# ### Load seismic data and inversion

# %%
datasrt = tt.load("./results/SRT_forward/synthetic_seismic_data_long.dat")
TT = pg.physics.traveltime.TravelTimeManager()
mesh_inv = TT.createMesh(datasrt, paraMaxCellSize=2, quality=32, paraDepth = 60.0)
TT.invert(datasrt, mesh = mesh_inv,lam=50,
          zWeight=0.2,vTop=500, vBottom=8000,
          verbose=1, limits=[300., 10000.])

# %% [markdown]
# ### Get parameters for plotting layers

# %%
# Get coverage and cell positions
cov = TT.standardizedCoverage()
pos = np.array(mesh_inv.cellCenters())
# For layered model visualization
x, y, triangles, _, dataIndex = createTriangles(mesh_inv)
z = pg.meshtools.cellDataToNodeData(mesh_inv,TT.model.array())


# %%
params = {'legend.fontsize': 15,
          #'figure.figsize': (15, 5),
         'axes.labelsize': 15,
         'axes.titlesize':16,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
import matplotlib.pylab as pylab
pylab.rcParams.update(params)

plt.rcParams["font.family"] = "Arial"

from palettable.lightbartlein.diverging import BlueDarkRed18_18
fixed_cmap = BlueDarkRed18_18.mpl_colormap

fig = plt.figure(figsize=[8,9])
ax1 = fig.add_subplot(1,1,1)
pg.show(mesh_inv,TT.model.array(),cMap=fixed_cmap,coverage = cov,ax = ax1,label='Velocity (m s$^{-1}$)',
        xlabel="Distance (m)", ylabel="Elevation (m)",pad=0.3,cMin =500, cMax=5000
       ,orientation="vertical")
ax1.set_xlabel("Distance (m)", fontsize=15)
ax1.set_ylabel("Elevation (m)", fontsize=15)
ax1.tricontour(x, y, triangles, z, levels=[1200], linewidths=1.0, colors='k', linestyles='dashed')
ax1.tricontour(x, y, triangles, z, levels=[5000], linewidths=1.0, colors='k', linestyles='-')


pg.viewer.mpl.drawSensors(ax1, datasrt.sensors(), diam=0.9,
                         facecolor='black', edgecolor='black')
fig.savefig(os.path.join(output_dir, 'seismic_velocity_long.tiff'), dpi=300, bbox_inches='tight')

# %% [markdown]
# ### Get subsurface structure for hydrologic modeling

# %%
# Assuming TT.model.array() gives you the velocity values
velocity_data = TT.model.array()

# Call the function with velocity data
# Get subsurface structure (Regolith) for hydrologic modeling
smooth_x1, smooth_z1 = extract_velocity_interface(mesh_inv, velocity_data, threshold=1200,interval = 5)

# Get subsurface structure (Fractured bedrock) for hydrologic modeling
# Here we limit the x range to extract the structure area within the coverage as shown in the figure
smooth_x2, smooth_z2 = extract_velocity_interface(mesh_inv, velocity_data, threshold=5000,interval = 5, x_min=22, x_max=84)


# %%
## plot the extracted interfaces withe filled velocity images
filled_cov1 = fill_holes_2d(pos, TT.standardizedCoverage())

fig = plt.figure(figsize=[8,9])
ax1 = fig.add_subplot(1,1,1)
pg.show(mesh_inv,TT.model.array(),cMap=fixed_cmap,coverage = filled_cov1,ax = ax1,label='Velocity (m s$^{-1}$)',
        xlabel="Distance (m)", ylabel="Elevation (m)",pad=0.3,cMin =500, cMax=5000
       ,orientation="vertical")
ax1.set_xlabel("Distance (m)", fontsize=15)
ax1.set_ylabel("Elevation (m)", fontsize=15)
ax1.plot(smooth_x1, smooth_z1, 'k--', linewidth=2, label='Regolith-Fractured Bedrock Interface')
ax1.plot(smooth_x2, smooth_z2, 'k-', linewidth=2, label='Fractured Bedrock- Fresh Bedrock Interface')
ax1.legend(fontsize=12)
np.savetxt(os.path.join(output_dir, 'regolith_interface.txt'), np.c_[smooth_x1, smooth_z1])
np.savetxt(os.path.join(output_dir, 'fractured_bedrock_interface.txt'), np.c_[smooth_x2, smooth_z2])

# %% [markdown]
# ## Short seismic profiles

# %%
ttData = tt.load("./results/workflow_example/synthetic_seismic_data.dat")
TT_short = pg.physics.traveltime.TravelTimeManager()
mesh_inv1 = TT_short.createMesh(ttData , paraMaxCellSize=2, quality=32, paraDepth = 30.0)
TT_short.invert(ttData , mesh = mesh_inv,lam=50,
          zWeight=0.2,vTop=500, vBottom=5500,
          verbose=1, limits=[300., 8000.])

# %%
x1, y1, triangles1, _, dataIndex1 = createTriangles(mesh_inv)
z1 = pg.meshtools.cellDataToNodeData(mesh_inv,np.array(TT_short.model))
pos = np.array(mesh_inv.cellCenters())
filled_cov1 = fill_holes_2d(pos, TT_short.standardizedCoverage())

# %%
params = {'legend.fontsize': 15,
          #'figure.figsize': (15, 5),
         'axes.labelsize': 15,
         'axes.titlesize':16,
         'xtick.labelsize':15,
         'ytick.labelsize':15}
import matplotlib.pylab as pylab
pylab.rcParams.update(params)

plt.rcParams["font.family"] = "Arial"

from palettable.lightbartlein.diverging import BlueDarkRed18_18
fixed_cmap = BlueDarkRed18_18.mpl_colormap

fig = plt.figure(figsize=[8,9])
ax1 = fig.add_subplot(1,1,1)
pg.show(mesh_inv,TT_short.model.array(),cMap=fixed_cmap,coverage = TT_short.standardizedCoverage(),ax = ax1,label='Velocity (m s$^{-1}$)',
        xlabel="Distance (m)", ylabel="Elevation (m)",pad=0.3,cMin =500, cMax=5000
       ,orientation="vertical")
ax1.set_xlabel("Distance (m)", fontsize=15)
ax1.set_ylabel("Elevation (m)", fontsize=15)

ax1.tricontour(x1, y1, triangles1, z1, levels=[1200], linewidths=1.0, colors='k', linestyles='dashed')



pg.viewer.mpl.drawSensors(ax1, ttData.sensors(), diam=0.8,
                         facecolor='black', edgecolor='black')
fig.savefig(os.path.join(output_dir, 'seismic_velocity_short.tiff'), dpi=300, bbox_inches='tight')


