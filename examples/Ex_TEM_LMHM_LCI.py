"""Ex. Synthetic LM+HM Line Inversion with Lateral Constraints
===========================================================

This example loads the bundled nine-station SQLite project through the same
TEMcompany/TEM2Go reader used by the Qt Studio. It jointly fits the LM and HM
gates, applies same-line L2 lateral constraints, and compares the recovered
section with the known synthetic resistivity model.

Because the truth model is known, the run doubles as an accuracy check: it
reports a log10 RMSE and a correlation against the truth, so a regression in the
1D forward model or in the lateral constraint shows up as a number rather than
as a section that merely looks plausible.
"""

# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_TEM_LMHM_LCI_fig_01.png'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from PyHydroGeophysX.workflows import em1d

###############################################################################
# 1. Load the bundled synthetic survey
# ------------------------------------

# %%
spec = em1d.example_catalog()["synthetic_tem_lci"]
project = spec["path"]
truth = np.load(project / "truth_model.npy")
head = em1d.load_sounding(str(project), "TDEM", moment="LM+HM")


###############################################################################
# 2. Choose geometry and inversion settings
# -----------------------------------------

# %%
geometry = {**head["system"], "tem_moment": "LM+HM"}
inversion = {
    **em1d.DEFAULT_INVERSION,
    **head["inversion_defaults"],
    **spec["params"],
}


###############################################################################
# 3. Invert the LM and HM gates together
# --------------------------------------

# %%
result = em1d.invert_line(
    str(project),
    "TDEM",
    geometry,
    inversion,
    positions=np.asarray(head["positions"], dtype=float),
    heights=np.asarray(head["heights"], dtype=float),
    max_soundings=truth.shape[0],
    doi_blank=False,
)


###############################################################################
# 4. Compare the recovered model with the truth
# ---------------------------------------------

# %%
recovered = np.asarray(result["model3d"][:, 0, ::-1], dtype=float)
log_truth = np.log10(truth)
log_recovered = np.log10(recovered)
log_rmse = float(np.sqrt(np.mean((log_recovered - log_truth) ** 2)))
correlation = float(np.corrcoef(
    log_truth.ravel(), log_recovered.ravel())[0, 1])

print(f"Mean data chi2: {result['chi2']:.3f}")
print(f"Model log10 RMSE: {log_rmse:.3f}")
print(f"Model log10 correlation: {correlation:.3f}")


###############################################################################
# 5. Plot the true and recovered sections
# ---------------------------------------

# %%
thickness = np.asarray(result["thickness"], dtype=float)
depth_edges = np.concatenate([
    [0.0], np.cumsum(thickness),
    [float(np.sum(thickness) + thickness[-1])],
])
positions = np.asarray(result["positions"], dtype=float)
position_edges = np.concatenate([
    [positions[0] - 5.0],
    0.5 * (positions[:-1] + positions[1:]),
    [positions[-1] + 5.0],
])
norm = LogNorm(vmin=float(np.min(truth)), vmax=float(np.max(truth)))

# Constrained layout, because a colorbar spanning both axes is one of the
# cases tight_layout cannot solve: it warns and then draws the bar on top of
# the right-hand panel.
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for axis, values, title in zip(
    axes, (truth, recovered), ("Synthetic truth", "LM+HM LCI recovery")
):
    image = axis.pcolormesh(
        position_edges, depth_edges, values.T,
        shading="flat", cmap="turbo", norm=norm)
    axis.invert_yaxis()
    axis.set_title(title)
    axis.set_xlabel("Distance (m)")
axes[0].set_ylabel("Depth (m)")
fig.colorbar(image, ax=axes, label="Resistivity (ohm m)")
fig.suptitle(
    f"log10 RMSE={log_rmse:.3f}; log10 correlation={correlation:.3f}")
plt.show()
