"""Memory-Optimized Time-Lapse ERT Inversion
==========================================

This example compares the sparse lower-RAM solver path
(``save_memory=True``) with the standard time-lapse inversion path
(``save_memory=False``). Both runs use the same measurements, mesh, and
inversion parameters so runtime, process memory, and recovered resistivity
distributions can be compared directly.
"""

# %%
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_TL_inversion_memory_fig_01.png'

import os
import sys
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from pygimli.physics import ert

# Setup package path for development
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
    if (
        not os.path.exists(os.path.join(current_dir, "data"))
        and os.path.exists(os.path.join(current_dir, "examples", "data"))
    ):
        current_dir = os.path.join(current_dir, "examples")

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from PyHydroGeophysX.inversion.time_lapse import TimeLapseERTInversion

try:
    import psutil
    process = psutil.Process()
except Exception as exc:
    process = None
    print("psutil not available; RSS memory numbers will be skipped.", exc)

# %% [markdown]
# Load the time-lapse measurements and create the inversion mesh
# --------------------------------------------------------------

# %%
ert_files = [
    "synthetic_data30.dat",
    "synthetic_data60.dat",
    "synthetic_data90.dat",
    "synthetic_data120.dat",
]

generated_data_dir = os.path.join(
    current_dir,
    "results",
    "TL_measurements",
    "appres",
)
bundled_data_dir = os.path.join(
    current_dir,
    "data",
    "TL_measurements",
    "appres",
)

generated_files = [
    os.path.join(generated_data_dir, filename)
    for filename in ert_files
]
data_dir = (
    generated_data_dir
    if all(os.path.exists(path) for path in generated_files)
    else bundled_data_dir
)

data_files = [
    os.path.join(data_dir, filename)
    for filename in ert_files
]
measurement_times = list(range(1, len(data_files) + 1))

missing_files = [
    path for path in data_files
    if not os.path.exists(path)
]
if missing_files:
    raise FileNotFoundError(
        "Missing time-lapse measurements:\n"
        + "\n".join(missing_files)
    )

data = ert.load(data_files[0])
mesh = ert.ERTManager(data).createMesh(data=data, quality=34)

print(f"Data directory: {data_dir}")
print(f"Time steps: {len(data_files)}")
print(f"Measurements per step: {data.size()}")
print(f"Inversion mesh cells: {mesh.cellCount()}")

# %% [markdown]
# Define the settings shared by both runs
# ---------------------------------------

# %%
base_params = {
    "lambda_val": 10.0,
    "alpha": 10.0,
    "decay_rate": 0.0,
    "method": "cgls",
    "model_constraints": (0.001, 1e4),
    "max_iterations": 8,
    "absoluteUError": 0.0,
    "relativeError": 0.05,
    "lambda_rate": 1.0,
    "lambda_min": 1.0,
    "inversion_type": "L2",
}

# %% [markdown]
# Run the memory-optimized inversion
# ----------------------------------
#
# The first run activates the sparse solver path with ``save_memory=True``.

# %%
print("Running save_memory=True ...")

save_params = {
    **base_params,
    "save_memory": True,
}
save_inversion = TimeLapseERTInversion(
    data_files=data_files,
    measurement_times=measurement_times,
    mesh=mesh,
    **save_params,
)

save_rss_before = (
    process.memory_info().rss / 1e9
    if process is not None
    else None
)
save_start = time.time()
result_save = save_inversion.run()
save_duration = time.time() - save_start

gc.collect()
save_rss_after = (
    process.memory_info().rss / 1e9
    if process is not None
    else None
)

stats_save = {
    "save_memory": True,
    "rss_before_gb": save_rss_before,
    "rss_after_gb": save_rss_after,
    "delta_gb": (
        save_rss_after - save_rss_before
        if save_rss_before is not None and save_rss_after is not None
        else None
    ),
    "duration_s": save_duration,
}

# %% [markdown]
# Run the standard inversion
# --------------------------
#
# The second run uses the same data, mesh, and inversion settings with ``save_memory=False``.

# %%
print("Running save_memory=False ...")

dense_params = {
    **base_params,
    "save_memory": False,
}
dense_inversion = TimeLapseERTInversion(
    data_files=data_files,
    measurement_times=measurement_times,
    mesh=mesh,
    **dense_params,
)

dense_rss_before = (
    process.memory_info().rss / 1e9
    if process is not None
    else None
)
dense_start = time.time()
result_dense = dense_inversion.run()
dense_duration = time.time() - dense_start

gc.collect()
dense_rss_after = (
    process.memory_info().rss / 1e9
    if process is not None
    else None
)

stats_dense = {
    "save_memory": False,
    "rss_before_gb": dense_rss_before,
    "rss_after_gb": dense_rss_after,
    "delta_gb": (
        dense_rss_after - dense_rss_before
        if dense_rss_before is not None and dense_rss_after is not None
        else None
    ),
    "duration_s": dense_duration,
}

# %% [markdown]
# Compare runtime and process memory
# ----------------------------------

# %%
for label, stats in [
    ("save_memory=True ", stats_save),
    ("save_memory=False", stats_dense),
]:
    if stats["rss_before_gb"] is None:
        memory_text = "RSS tracking unavailable"
    else:
        memory_text = (
            f"before: {stats['rss_before_gb']:.2f} GB; "
            f"after: {stats['rss_after_gb']:.2f} GB; "
            f"delta: {stats['delta_gb']:.2f} GB"
        )

    print(
        f"{label} -> {memory_text}; "
        f"duration: {stats['duration_s']:.1f} s"
    )

# %% [markdown]
# Compare the recovered resistivity distributions
# -----------------------------------------------

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

if result_save.final_models is not None:
    axes[0].hist(
        np.log10(result_save.final_models[:, 0] + 1e-6),
        bins=40,
    )
    axes[0].set_title("save_memory=True")
    axes[0].set_xlabel("log10 resistivity")
    axes[0].set_ylabel("Count")
else:
    axes[0].text(0.5, 0.5, "No data", ha="center")
    axes[0].set_axis_off()

if result_dense.final_models is not None:
    axes[1].hist(
        np.log10(result_dense.final_models[:, 0] + 1e-6),
        bins=40,
        color="orange",
    )
    axes[1].set_title("save_memory=False")
    axes[1].set_xlabel("log10 resistivity")
    axes[1].set_ylabel("Count")
else:
    axes[1].text(0.5, 0.5, "No data", ha="center")
    axes[1].set_axis_off()

plt.tight_layout()
plt.show()

# %% [markdown]
# The histograms provide a quick consistency check for the recovered baseline
# resistivity distributions from the two solver paths.
#
# .. image:: /auto_examples/images/Ex_TL_inversion_memory_fig_01.png
#    :width: 900px
#    :align: center
