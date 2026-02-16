"""
FDEM Forward + Inversion Workflow
=================================

This example demonstrates a complete 1D FDEM workflow:
1. Build synthetic FDEM data from hydrological properties.
2. Invert the synthetic data with `FDEMInversion`.
"""

# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_FDEM_workflow_fig_01.png'

import numpy as np

from PyHydroGeophysX.forward.fdem_forward import FDEMForwardModeling
from PyHydroGeophysX.inversion.fdem_inversion import FDEMInversion


def run_example():
    water_content = np.array([0.12, 0.16, 0.22, 0.28])
    porosity = np.array([0.30, 0.32, 0.35, 0.38])
    thicknesses = np.array([5.0, 10.0, 15.0])
    frequencies = np.logspace(1, 4, 12)

    noisy, clean, uncertainty, conductivity = FDEMForwardModeling.hydro_to_fdem(
        water_content=water_content,
        porosity=porosity,
        layer_thicknesses=thicknesses,
        frequencies=frequencies,
        receiver_component="secondary",
        waveform_type="dipole",
        noise_level=0.03,
        seed=42,
        sigma_w=0.05,
        m=1.5,
        n=2.0,
        sigma_s=0.0,
    )

    inversion = FDEMInversion(
        frequencies=frequencies,
        dobs=noisy,
        uncertainties=uncertainty,
        thicknesses=thicknesses,
        receiver_component="secondary",
        waveform_type="dipole",
        max_iterations=40,
        use_irls=True,
    )
    result = inversion.run()

    print("FDEM workflow complete")
    print(f"  true conductivity: {conductivity}")
    print(f"  recovered conductivity: {result.recovered_conductivity}")
    print(f"  chi2: {result.chi2:.3f}")

    return result, clean


if __name__ == "__main__":
    run_example()
