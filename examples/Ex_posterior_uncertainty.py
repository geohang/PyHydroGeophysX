"""Ex. Posterior Resistivity Uncertainty to Water Content
========================================================

Compute Gaussian posterior covariance for a synthetic linear survey and
propagate it through an Archie-type water-content transform. Unlike
Ex_MC_Hydro, this example samples resistivity uncertainty while holding
petrophysical parameters fixed. No previous inversion output is required.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_posterior_uncertainty_fig_01.png'

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

from PyHydroGeophysX.uncertainty import linearized_posterior, propagate_petro_uncertainty

# %% [markdown]
# Define survey errors and a correlated resistivity prior
# -------------------------------------------------------
# This averaging kernel is a linear teaching surrogate, not an ERT solver.
# Cd contains variances (ohm m)^2, unlike inverse standard-deviation weights.

# %%
depth = np.arange(1., 7.)
J = .75 * np.eye(len(depth)) + .25 / len(depth)
prior_mean = np.full(len(depth), 180.)
prior_cov = 20. ** 2 * np.exp(-np.abs(depth[:, None] - depth[None, :]) / 2.)
data_variance = np.full(len(depth), 5. ** 2)
truth = 180. + 25. * np.sin(depth / 2.)
observations = J @ truth + np.random.default_rng(2026).normal(0., 5., len(depth))
posterior_cov = linearized_posterior(J, data_variance, prior_cov)
posterior_mean = prior_mean + posterior_cov @ J.T @ ((observations - J @ prior_mean) / data_variance)

# %% [markdown]
# Propagate the full posterior covariance into water content
# ---------------------------------------------------------
# For fixed porosity=.4, saturated resistivity=50 ohm m and exponent n=2,
# theta = porosity * sqrt(rhos/rho). Samples must remain above rhos to stay
# unsaturated. This narrow Gaussian example checks that assumption explicitly;
# broader uncertainties may require a bounded or log-resistivity model.

# %%
def resistivity_to_water_content(rho):
    if np.any(rho < 50.):
        raise ValueError("A Gaussian sample left the assumed unsaturated domain.")
    return .4 * np.sqrt(50. / rho)


water_content = propagate_petro_uncertainty(
    posterior_mean, posterior_cov, resistivity_to_water_content,
    n_samples=5000, seed=2026)
low, high = np.quantile(water_content["samples"], [.025, .975], axis=0)
print("Posterior resistivity standard deviations:", np.sqrt(np.diag(posterior_cov)))
print("Water-content means:", water_content["mean"])

# %% [markdown]
# Save covariance, samples and interpretation plots
# ------------------------------------------------
# The interval is conditional on fixed petrophysics; it does not include
# uncertainty in porosity, salinity or the forward-model assumptions.

# %%
output_dir = os.path.join(current_dir, "results", "posterior_uncertainty")
os.makedirs(output_dir, exist_ok=True)
np.savez_compressed(os.path.join(output_dir, "posterior.npz"), depth=depth,
                    resistivity_mean=posterior_mean, resistivity_cov=posterior_cov,
                    water_content_mean=water_content["mean"],
                    water_content_samples=water_content["samples"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
axes[0].plot(np.sqrt(np.diag(prior_cov)), depth, label="Prior")
axes[0].plot(np.sqrt(np.diag(posterior_cov)), depth, label="Posterior")
axes[0].set(xlabel="Resistivity std (ohm m)", ylabel="Depth (m)", title="Uncertainty reduction")
axes[0].invert_yaxis()
axes[0].legend()
image = axes[1].imshow(posterior_cov, cmap="viridis")
axes[1].set(xlabel="Cell index", ylabel="Cell index", title="Posterior covariance")
fig.colorbar(image, ax=axes[1], label="(ohm m)^2")
axes[2].fill_betweenx(depth, low, high, alpha=.25, label="95% sample interval")
axes[2].plot(water_content["mean"], depth, label="Posterior mean")
axes[2].plot(resistivity_to_water_content(truth), depth, "k--", label="Truth")
axes[2].set(xlabel="Water content (m3/m3)", ylabel="Depth (m)", title="Hydrological interpretation")
axes[2].invert_yaxis()
axes[2].legend()
fig.savefig(os.path.join(output_dir, "posterior_uncertainty.png"), dpi=150)
plt.show()

# %% [markdown]
# The survey reduces the resistivity standard deviation in every cell from
# 20 to about 6 ohm m (left). The posterior covariance (middle) is almost
# diagonal: the data remove nearly all of the prior correlation between cells.
# The water-content interval (right) is that posterior resistivity uncertainty
# mapped through the fixed Archie-type transform.
#
# .. image:: /auto_examples/images/Ex_posterior_uncertainty_fig_01.png
#    :align: center
#    :width: 900px
