"""
Hydrology to Multi-Geophysics Pipeline
======================================

This example demonstrates a compact end-to-end bridge:
1. Hydrological water content + porosity.
2. Petrophysical conversion to resistivity, velocity, conductivity.
3. Setup points for ERT/SRT/TDEM/FDEM workflows.
"""

# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_hydro_to_multigeophys_fig_01.png'

import numpy as np

from PyHydroGeophysX.inversion.cross_constraints import PetrophysicalCoupling


def run_example():
    # In practice, these arrays come from MODFLOW or ParFlow output extraction.
    water_content = np.array([0.10, 0.14, 0.18, 0.21, 0.24, 0.27])
    porosity = np.array([0.28, 0.30, 0.32, 0.34, 0.35, 0.36])

    geophys = PetrophysicalCoupling.water_content_to_all_geophysics(
        water_content=water_content,
        porosity=porosity,
        rhos=120.0,
        n=2.0,
        sigma_sur=0.0,
        velocity_model="hertz_mindlin",
        bulk_modulus=36.0,
        shear_modulus=45.0,
        mineral_density=2650.0,
        depth=8.0,
    )

    print("Hydro -> multi-geophysics conversion complete")
    print(f"  resistivity range: {geophys['resistivity'].min():.3f} - {geophys['resistivity'].max():.3f} ohm-m")
    print(f"  velocity range: {geophys['velocity'].min():.3f} - {geophys['velocity'].max():.3f} m/s")
    print(f"  conductivity range: {geophys['conductivity'].min():.6f} - {geophys['conductivity'].max():.6f} S/m")

    # Next-step hooks for method-specific forward/inversion classes.
    # - ERT: PyHydroGeophysX.inversion.ERTInversion
    # - SRT: PyHydroGeophysX.inversion.SRTInversion
    # - TDEM: PyHydroGeophysX.inversion.TDEMInversion
    # - FDEM: PyHydroGeophysX.inversion.FDEMInversion

    return geophys


if __name__ == "__main__":
    run_example()
