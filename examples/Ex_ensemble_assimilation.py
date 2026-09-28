"""Ex. Hydro-Geophysical Ensemble Data Assimilation
==================================================

Update a synthetic water-content profile using EnKF and ES-MDA. A simple
wetting step supplies the forecast; Archie-type petrophysics and a spatial
averaging kernel supply observations. These are teaching surrogates, not
MODFLOW or ERT simulations. No external data or optional engines are needed.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_ensemble_assimilation_fig_01.png'

# %%
import os
import sys

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

from PyHydroGeophysX.assimilation import ESMDA, EnsembleKalmanFilter, HydroGeophysObsOperator
from PyHydroGeophysX.petrophysics.resistivity_models import water_content_to_resistivity

# %% [markdown]
# Build the observation operator and initial ensemble
# ---------------------------------------------------
# Ensemble axes are (state cells, members); observation covariance is in
# squared data units. This example observes averaged electrical conductivity.

# %%
rng = np.random.default_rng(2026)
depth = np.arange(1., 7.)
porosity = .4
kernel = .8 * np.eye(len(depth)) + .2 / len(depth)
operator = HydroGeophysObsOperator(
    petro_transform=lambda theta: water_content_to_resistivity(theta, rhos=50., n=2., porosity=porosity),
    forward_operator=lambda rho: kernel @ (1. / rho))
truth = .24 + .025 * np.exp(-depth / 3.)
ensemble = rng.normal(.20, .025, size=(len(depth), 600))
data_std = .00015  # S/m
observations = operator(truth) + rng.normal(0., data_std, len(depth))
obs_cov = np.eye(len(depth)) * data_std ** 2

# %% [markdown]
# Forecast water content, then assimilate the same observations two ways
# ---------------------------------------------------------------------
# ES-MDA starts from the forecast, not from the EnKF posterior; otherwise
# these same observations would be counted twice. Four alpha=4 updates obey
# sum(1/alpha)=1. Both methods are unconstrained Gaussian ensemble algorithms.

# %%
enkf = EnsembleKalmanFilter(operator, obs_cov)


def wetting_step(theta, recharge):
    return theta + recharge * np.exp(-depth / 3.)


forecast = enkf.forecast(ensemble, wetting_step, recharge=.012)
posterior_enkf = enkf.update(forecast, observations, rng=np.random.default_rng(10))
smoother = ESMDA(operator, obs_cov, n_steps=4)
smoothed = smoother.update(forecast, observations, rng=np.random.default_rng(11))
posterior_esmda = smoothed["ensemble"]
for name, members in [("Forecast", forecast), ("EnKF", posterior_enkf), ("ES-MDA", posterior_esmda)]:
    rmse = np.sqrt(np.mean((members.mean(axis=1) - truth) ** 2))
    print(f"{name}: water-content RMSE = {rmse:.5f}")
    if np.any((members < 0.) | (members > porosity)):
        raise ValueError("Ensemble left physical water-content bounds; revise the state parameterization.")

# %% [markdown]
# Save ensembles and plot water-content uncertainty
# -------------------------------------------------
# Shading shows the 5th to 95th ensemble percentiles, conditional on the
# chosen petrophysical parameters and synthetic observation operator.

# %%
output_dir = os.path.join(current_dir, "results", "ensemble_assimilation")
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(os.path.join(output_dir, "ensembles.npz"), depth=depth, truth=truth,
                    forecast=forecast, enkf=posterior_enkf, esmda=posterior_esmda,
                    history=smoothed["history"], observations=observations)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
for ax, posterior, title in zip(axes, [posterior_enkf, posterior_esmda], ["EnKF", "ES-MDA"]):
    for label, members, color in [("Forecast", forecast, "C0"), (title, posterior, "C1")]:
        low, high = np.quantile(members, [.05, .95], axis=1)
        ax.fill_betweenx(depth, low, high, color=color, alpha=.2)
        ax.plot(members.mean(axis=1), depth, color=color, label=label)
    ax.plot(truth, depth, "k--", label="Truth")
    ax.set(xlabel="Water content (m3/m3)", ylabel="Depth (m)", title=title)
    ax.invert_yaxis()
    ax.legend()
fig.savefig(os.path.join(output_dir, "ensemble_assimilation.png"), dpi=150)
plt.show()

# %% [markdown]
# Both updates move the forecast ensemble mean towards the true profile and
# narrow its 5th to 95th percentile band, EnKF in one step (left) and ES-MDA
# in four inflated steps (right).
#
# .. image:: /auto_examples/images/Ex_ensemble_assimilation_fig_01.png
#    :align: center
#    :width: 800px
