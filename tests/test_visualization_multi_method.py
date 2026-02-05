import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PyHydroGeophysX.visualization import (
    plot_cross_section_with_wells,
    plot_em_data_fit,
    plot_em_fit_and_residuals,
    plot_em_residuals,
    plot_hydro_vs_geophys,
    plot_layered_profiles,
    plot_multi_method_panel,
    plot_petrophysical_scatter,
    plot_time_lapse_panel,
)


class _Result:
    def __init__(self, final_model=None, recovered_conductivity=None):
        self.final_model = final_model
        self.recovered_conductivity = recovered_conductivity


def test_multi_method_basic_plots():
    ert = _Result(final_model=np.linspace(10, 50, 20))
    srt = _Result(final_model=np.linspace(1200, 3500, 20))
    em = _Result(recovered_conductivity=np.linspace(0.01, 0.1, 20))

    fig1, axes1 = plot_multi_method_panel(ert, srt, em, mesh=None)
    assert len(axes1) == 3

    fig2, axes2 = plot_hydro_vs_geophys(np.linspace(0.1, 0.3, 20), np.linspace(0.12, 0.28, 20))
    assert len(axes2) == 2

    fig3, ax3 = plot_cross_section_with_wells(
        ert,
        mesh=None,
        well_data={"x": np.array([2.0, 6.0]), "z": np.array([1.0, 1.5]), "labels": ["A", "B"]},
    )
    assert ax3 is not None

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


def test_em_fit_residual_and_scatter_plots():
    times = np.logspace(-5, -3, 12)
    observed = np.exp(-times * 1e4)
    predicted = observed * 1.03
    uncertainties = np.full_like(times, 0.02)

    fig1, ax1 = plot_em_data_fit(times, observed, predicted, uncertainties=uncertainties, chi2=1.2)
    fig2, ax2 = plot_em_residuals(times, observed, predicted, uncertainties=uncertainties)
    fig3, axes3 = plot_em_fit_and_residuals(times, observed, predicted, uncertainties=uncertainties)
    fig4, ax4 = plot_petrophysical_scatter(
        x=np.linspace(0.2, 0.4, 20),
        y=np.linspace(100, 500, 20),
        color=np.linspace(0.3, 0.9, 20),
    )

    assert ax1 is not None
    assert ax2 is not None
    assert len(axes3) == 2
    assert ax4 is not None

    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


def test_time_lapse_and_layered_profile_plots():
    snapshots = [np.linspace(i, i + 1, 15) for i in range(4)]
    fig1, axes1 = plot_time_lapse_panel(snapshots, ncols=2)
    assert axes1.shape == (2, 2)

    depth_edges = np.array([0.0, 2.0, 5.0, 9.0])
    profiles = {
        "Porosity (-)": np.array([0.35, 0.30, 0.25]),
        "Conductivity (S/m)": np.array([0.08, 0.03, 0.01]),
    }
    fig2, axes2 = plot_layered_profiles(depth_edges, profiles, xscale="log")
    assert len(axes2) == 2

    plt.close(fig1)
    plt.close(fig2)
