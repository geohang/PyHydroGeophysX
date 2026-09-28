"""Ex. Sensitivity, Resolution and Reference-Model Dependence
===========================================================

Inspect a small linear resistivity inverse problem before interpreting it.
The synthetic averaging kernel is a teaching surrogate, not an ERT solver.
Only NumPy, SciPy and Matplotlib are needed; no input files are required.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_sensitivity_analysis_fig_01.png'

# %%
import os
import sys
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
    if os.path.isdir(os.path.join(current_dir, "examples")):
        current_dir = os.path.join(current_dir, "examples")
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from PyHydroGeophysX.analysis import (
    compute_cumulative_sensitivity, compute_depth_of_investigation,
    compute_resolution_matrix, plot_sensitivity_map,
)
from PyHydroGeophysX.uncertainty import model_resolution_spread

# %% [markdown]
# Define a synthetic resistivity profile and survey kernel
# ------------------------------------------------------
# Each datum averages nearby cells. Sensitivity decreases with depth.

# %%
depth = np.arange(1., 13.)
centres = np.linspace(1., 9., 8)
J = np.exp(-.5 * ((depth[None, :] - centres[:, None]) / 1.5) ** 2)
J /= J.sum(axis=1, keepdims=True)
J *= np.exp(-depth[None, :] / 12.)
true_resistivity = 100. + 50. * np.exp(-.5 * ((depth - 6.) / 1.5) ** 2)
data_std = 2.
data = J @ true_resistivity + np.random.default_rng(2026).normal(0., data_std, len(centres))
Wd = np.full(len(data), 1. / data_std)  # inverse standard deviations, not variances
lam = .002

# %% [markdown]
# Compute sensitivity and regularized model resolution
# ----------------------------------------------------
# Resolution depends on both the survey and regularization strength.

# %%
sensitivity = compute_cumulative_sensitivity(J)
resolution = compute_resolution_matrix(J, Wd, 1., lam)
strong_resolution = compute_resolution_matrix(J, Wd, 1., 10. * lam)
metrics = model_resolution_spread(resolution)
print(f"Mean model resolution: {metrics['mean_resolution']:.3f}")

# %% [markdown]
# Repeat inversion with two reference models
# -----------------------------------------
# The adapter implements the run(initial_model=...) protocol used by DOI.
# Here initial_model is also the regularization reference, so changing it
# changes the solution. Merely changing a converged solver's starting guess
# would not measure reference dependence. The returned DOI is a normalized
# model-difference index, not a physical investigation depth in metres.

# %%
class LinearReferenceInversion:
    def run(self, initial_model):
        normal = J.T @ J / data_std ** 2 + lam * np.eye(len(depth))
        rhs = J.T @ data / data_std ** 2 + lam * initial_model
        return SimpleNamespace(final_model=np.linalg.solve(normal, rhs))


doi, models = compute_depth_of_investigation(
    LinearReferenceInversion(), {"rhoa": data},
    SimpleNamespace(cellCount=lambda: len(depth)), reference_resistivity=100.)

# %% [markdown]
# Save diagnostics and compare the recovered profiles
# ---------------------------------------------------

# %%
output_dir = os.path.join(current_dir, "results", "sensitivity_analysis")
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(os.path.join(output_dir, "diagnostics.npz"), depth=depth,
                    sensitivity=sensitivity, resolution=resolution, doi=doi, **models)
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
plot_sensitivity_map(sensitivity, mesh=None, ax=axes[0])
axes[1].plot(depth, np.diag(resolution), "o-", label="Regularization")
axes[1].plot(depth, np.diag(strong_resolution), "s-", label="10x regularization")
axes[1].plot(depth, doi, "--", label="Reference dependence")
axes[1].set(xlabel="Depth (m)", ylabel="Dimensionless index", title="Resolution and DOI")
axes[1].legend()
axes[2].plot(true_resistivity, depth, "k--", label="Truth")
axes[2].plot(models["model_low"], depth, label="80 ohm m reference")
axes[2].plot(models["model_high"], depth, label="120 ohm m reference")
axes[2].set(xlabel="Resistivity (ohm m)", ylabel="Depth (m)", title="Reference-model comparison")
axes[2].invert_yaxis()
axes[2].legend()
fig.savefig(os.path.join(output_dir, "sensitivity_analysis.png"), dpi=150)
plt.show()

# %% [markdown]
# Cumulative sensitivity (left) falls off with depth. The resolution diagonal
# (middle) drops further under ten times stronger regularization, and the
# reference-dependence index rises where the data stop constraining the model.
# There the two profiles recovered with different reference models (right)
# separate.
#
# .. image:: /auto_examples/images/Ex_sensitivity_analysis_fig_01.png
#    :align: center
#    :width: 900px
