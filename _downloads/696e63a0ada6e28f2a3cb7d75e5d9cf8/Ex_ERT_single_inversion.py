"""
Ex. Single ERT File Inversion (No Time-Lapse)
==============================================

This example shows a minimal, robust workflow for one ERT survey:

1. Accept either a folder path or a single ERT file path.
2. Use ``ert_data_agent`` functions to load/QC/export data.
3. Run one ERT inversion with ``ERTInversion``.
4. Save inversion artifacts (model, convergence, summary) to one folder.
"""
# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_ERT_single_inversion_fig_01.png'

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pygimli as pg

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

from PyHydroGeophysX.data_processing.ert_data_agent import (
    LocalRef,
    export_for_inversion,
    load_ert_resipy,
    qc_and_visualize,
)
from PyHydroGeophysX.inversion.ert_inversion import ERTInversion

SUPPORTED_ERT_EXTENSIONS = (
    ".ohm",
    ".data",
    ".dat",
    ".stg",
    ".ares",
    ".pro",
    ".inv",
    ".txt",
    ".csv",
)

_EXTENSION_TO_INSTRUMENT = {
    ".ohm": "E4D",
    ".data": "DAS-1",
    ".stg": "Sting",
    ".ares": "ARES",
    ".pro": "Protocol DC",
    ".inv": "ResInv",
}


###############################################################################
# Resolve the Input Dataset
# -------------------------

# %%
def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    return resolved


# %%
def _find_ert_data_file(input_path: str | Path) -> Path:
    resolved = _resolve_path(input_path)
    if resolved.is_file():
        return resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")

    ignored_dirs = {"invdir", "results", "res", "__pycache__", ".git"}
    ignored_name_tokens = ("acknow", "readme", "license")
    ext_priority = {ext.lower(): i for i, ext in enumerate(SUPPORTED_ERT_EXTENSIONS)}

    candidates = []
    for candidate in sorted(resolved.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in ext_priority:
            continue
        if any(token in candidate.name.lower() for token in ignored_name_tokens):
            continue
        if any(part.lower() in ignored_dirs for part in candidate.parts):
            continue
        candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError(f"No supported ERT files found in: {resolved}")

    candidates = sorted(
        candidates,
        key=lambda p: (ext_priority.get(p.suffix.lower(), 999), str(p)),
    )
    if len(candidates) > 1:
        print(f"Multiple ERT files found. Using: {candidates[0]}")
    return candidates[0]


# %%
def _detect_instrument(data_file: Path) -> str:
    suffix = data_file.suffix.lower()
    if suffix in _EXTENSION_TO_INSTRUMENT:
        return _EXTENSION_TO_INSTRUMENT[suffix]
    if suffix in (".txt", ".csv"):
        return "Syscal"
    if suffix == ".dat":
        return "BERT"
    return "BERT"


###############################################################################
# Define a Reusable Data-Processing Helper
# ----------------------------------------

# %%
def _process_with_data_agent(
    input_path: str | Path,
    instrument: Optional[str],
    outdir: Path,
    project_dir: Optional[str | Path],
    crs: str,
    use_source_error: bool = False,
    use_electrode_file: bool = False,
    electrode_file: Optional[str | Path] = None,
) -> Dict[str, object]:
    data_file = _find_ert_data_file(input_path)
    resolved_instrument = instrument or _detect_instrument(data_file)
    project_dir_path = _resolve_path(project_dir) if project_dir else data_file.parent
    electrode_file_path = None
    if use_electrode_file:
        if electrode_file is None:
            raise ValueError(
                "use_electrode_file=True but no electrode_file path was provided."
            )
        electrode_file_path = _resolve_path(electrode_file)
        if not electrode_file_path.exists():
            raise FileNotFoundError(f"Electrode file not found: {electrode_file_path}")
        print(f"Using electrode file: {electrode_file_path}")

    ert = load_ert_resipy(
        project_dir=str(project_dir_path),
        data_file=str(data_file),
        instrument=resolved_instrument,
        electrode_file=str(electrode_file_path) if electrode_file_path else None,
        crs=crs,
        local_ref=LocalRef(origin_x=0.0, origin_y=0.0, azimuth_deg=90.0),
    )
    artifacts = qc_and_visualize(ert, outdir=str(outdir))
    bert_path = export_for_inversion(
        ert,
        outdir=str(outdir),
        fmt="pgimli",
        use_source_error=use_source_error,
    )

    return {
        "ert": ert,
        "data_file": str(data_file),
        "instrument": resolved_instrument,
        "project_dir": str(project_dir_path),
        "use_electrode_file": bool(use_electrode_file),
        "electrode_file": str(electrode_file_path) if electrode_file_path else None,
        "artifacts": artifacts,
        "bert_path": bert_path,
    }


###############################################################################
# Configure the Single-Survey Inversion
# -------------------------------------
#
# Edit these values directly when using the downloaded Python script or
# notebook with another survey.

# %%
input_path = Path(current_dir) / "data" / "ERT" / "E4D" / "2021-10-08_1400.ohm"
instrument = "E4D"
output_dir_path = Path(current_dir) / "results" / "ert_single_inversion"
project_dir = None
crs = "local"
lambda_val = 10.0
max_iterations = 10
method = "cgls"
use_gpu = False
use_source_error = False
use_electrode_file = False
electrode_file = None

output_dir_path.mkdir(parents=True, exist_ok=True)

###############################################################################
# Load, Quality-Control, and Export the ERT Data
# ----------------------------------------------

# %%
process_result = _process_with_data_agent(
    input_path=input_path,
    instrument=instrument,
    outdir=output_dir_path,
    project_dir=project_dir,
    crs=crs,
    use_source_error=use_source_error,
    use_electrode_file=use_electrode_file,
    electrode_file=electrode_file,
)

bert_path = process_result["bert_path"]
print(f"Input file: {process_result['data_file']}")
print(f"Instrument: {process_result['instrument']}")
print(f"BERT file: {bert_path}")

###############################################################################
# Run the ERT Inversion
# ---------------------

# %%
inversion = ERTInversion(
    data_file=str(bert_path),
    lambda_val=lambda_val,
    method=method,
    max_iterations=max_iterations,
    lambda_rate=1.0,
    use_gpu=use_gpu,
)
inversion_result = inversion.run()

###############################################################################
# Save Numerical Results
# ----------------------

# %%
result_prefix = output_dir_path / "single_ert_inversion"
inversion_result.save(str(result_prefix))

final_model_path = output_dir_path / "final_model.npy"
predicted_data_path = output_dir_path / "predicted_data.npy"
coverage_path = output_dir_path / "coverage.npy"
np.save(final_model_path, inversion_result.final_model)
np.save(predicted_data_path, inversion_result.predicted_data)
np.save(coverage_path, inversion_result.coverage)

###############################################################################
# Plot the Inverted Resistivity Model
# -----------------------------------

# %%
model_plot_path = output_dir_path / "single_ert_model.png"
fig, ax = plt.subplots(figsize=(10, 4))
coverage_mask = None
if inversion_result.coverage is not None:
    coverage_mask = np.asarray(inversion_result.coverage) > -1.0
pg.show(
    inversion_result.mesh,
    inversion_result.final_model,
    ax=ax,
    cMap="Spectral_r",
    cMin=float(np.percentile(inversion_result.final_model, 2)),
    cMax=float(np.percentile(inversion_result.final_model, 98)),
    logScale=True,
    label="Resistivity [Ohm-m]",
    coverage=coverage_mask,
    show=False,
)
ax.set_title("Single ERT Inversion Result")
fig.tight_layout()
fig.savefig(model_plot_path, dpi=200)
plt.show()

###############################################################################
# The inverted section shows the recovered resistivity distribution for the
# selected field survey.
#
# .. image:: /auto_examples/images/Ex_ERT_single_inversion_fig_01.png
#    :align: center
#    :width: 800px

###############################################################################
# Plot Inversion Convergence
# --------------------------

# %%
chi2_plot_path = output_dir_path / "single_ert_convergence.png"
if inversion_result.iteration_chi2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(inversion_result.iteration_chi2, "o-", color="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Chi2")
    ax.set_yscale("log")
    ax.set_title("Inversion Convergence")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(chi2_plot_path, dpi=200)
    plt.show()
else:
    chi2_plot_path = None

###############################################################################
# The chi-squared history documents convergence and provides a direct check on
# whether additional iterations materially improve the data fit.
#
# .. image:: /auto_examples/images/Ex_ERT_single_inversion_fig_02.png
#    :align: center
#    :width: 600px

###############################################################################
# Write a Reproducible Run Summary
# --------------------------------

# %%
mesh_path = str(result_prefix) + ".bms"
if not Path(mesh_path).exists():
    mesh_path = None

summary = {
    "input_path": str(input_path),
    "resolved_data_file": process_result["data_file"],
    "instrument_used": process_result["instrument"],
    "use_source_error": bool(use_source_error),
    "use_electrode_file": bool(use_electrode_file),
    "electrode_file": process_result.get("electrode_file"),
    "bert_file": str(bert_path),
    "qc_artifacts": process_result["artifacts"],
    "result_pickle": str(result_prefix) + ".pkl",
    "result_mesh": mesh_path,
    "final_model_npy": str(final_model_path),
    "predicted_data_npy": str(predicted_data_path),
    "coverage_npy": str(coverage_path),
    "model_plot": str(model_plot_path),
    "convergence_plot": str(chi2_plot_path) if chi2_plot_path else None,
    "final_chi2": (
        float(inversion_result.iteration_chi2[-1])
        if inversion_result.iteration_chi2
        else None
    ),
}

summary_path = output_dir_path / "single_ert_summary.json"
with open(summary_path, "w", encoding="utf-8") as summary_file:
    json.dump(summary, summary_file, indent=2)

print("Single-file inversion finished.")
print(f"Summary: {summary_path}")
print(f"Model plot: {model_plot_path}")
