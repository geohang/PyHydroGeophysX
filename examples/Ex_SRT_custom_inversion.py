"""
Custom SRT Inversion with `SRTInversion`
========================================

This example shows how to use the package-level `SRTInversion` class instead
of calling `TravelTimeManager.invert()` directly.
"""

# sphinx_gallery_thumbnail_path = 'auto_examples/images/Ex_SRT_custom_inversion_fig_01.png'

from pathlib import Path

from PyHydroGeophysX.inversion.srt_inversion import SRTInversion


def run_example(data_file: str):
    inversion = SRTInversion(
        data_file=data_file,
        lambda_val=50.0,
        zWeight=0.2,
        vTop=500.0,
        vBottom=5000.0,
        model_constraints=(100.0, 10000.0),
        max_iterations=20,
        method="cgls",
    )
    result = inversion.run()

    print("SRT inversion finished")
    print(f"  cells: {len(result.final_model)}")
    print(f"  velocity range: {result.final_model.min():.2f} - {result.final_model.max():.2f} m/s")
    if result.iteration_chi2:
        print(f"  final chi2: {result.iteration_chi2[-1]:.3f}")

    return result


if __name__ == "__main__":
    default_file = Path("results") / "SRT_forward" / "synthetic_seismic_data_long.dat"
    if not default_file.exists():
        raise FileNotFoundError(
            "Provide a valid .sgt/.dat travel-time file before running this example."
        )
    run_example(str(default_file))
