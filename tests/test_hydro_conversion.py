"""Analytical petrophysics, coordinate and input-writer integration checks."""
import numpy as np
import pytest
from PyHydroGeophysX.model_input import (
    HydroGrid, interpret_resistivity, saturation_to_pressure,
    map_to_hydro_grid, prepare_hydro_updates, write_parflow_inputs,
)


@pytest.mark.parametrize('surface', [0., .002])
def test_petrophysics_inverse(surface):
    phi = np.array([.2,.3,.4])
    sat = np.array([.35,.7,1.])
    rho = 1/(phi**2/10*sat**2+surface*sat)
    result = interpret_resistivity(rho, porosity=phi, rho_fluid=10, m=2, n=2, sigma_sur=surface)
    np.testing.assert_allclose(result['saturation'], sat, atol=1e-10)
    np.testing.assert_allclose(result['water_content'], phi*sat)
    reverse = interpret_resistivity(rho, saturation=sat, rho_fluid=10, m=2,n=2,sigma_sur=surface)
    np.testing.assert_allclose(reverse['porosity'],phi)
    scalar = interpret_resistivity(float(rho[0]), porosity=.2,rho_fluid=10,m=2,n=2,sigma_sur=surface)
    assert scalar['saturation'] == pytest.approx(.35)


def test_pressure_retention_and_saturated_ambiguity():
    pressure = np.array([-10., -1., -.1])
    sat = .1+.9*(1+(.5*np.abs(pressure))**2)**(-.5)
    actual = saturation_to_pressure(sat,alpha=.5,n=2,residual_saturation=.1)
    np.testing.assert_allclose(actual,pressure)
    with pytest.raises(ValueError,match='explicit'):
        saturation_to_pressure([1.],alpha=.5,n=2,residual_saturation=.1)
    assert saturation_to_pressure([1.],alpha=.5,n=2,residual_saturation=.1,saturated_pressure=3)[0] == 3
    with pytest.raises(ValueError):
        interpret_resistivity([1.],porosity=.3,rho_fluid=10,m=2,n=2)


def test_mapping_mask_range_order_and_baseline():
    grid = HydroGrid(np.array([[[[0.,0.,-2.],[0.,0.,-1.],[50.,0.,0.]]]]), np.array([[[1,0,1]]]))
    baseline = np.array([[[9.,8.,7.]]])
    mapped = map_to_hydro_grid([[0,0,-1],[0,0,-2]], [1.,2.], grid,
                                baseline=baseline,max_distance=.1)
    np.testing.assert_array_equal(mapped.values,[[[2,8,7]]])
    np.testing.assert_array_equal(mapped.source_index,[[[1,-1,-1]]])
    np.testing.assert_array_equal(baseline,[[[9,8,7]]])
    with pytest.raises(ValueError,match='duplicate'):
        map_to_hydro_grid([[0,0,0],[0,0,0]],[1,2],grid,baseline=baseline,max_distance=2)


def test_conversion_to_parflow_export(tmp_path):
    pytest.importorskip('parflow')
    from parflow.tools.io import read_pfb
    config = {'Domain.GeomName':'domain', 'ComputationalGrid.NX':2,
              'ComputationalGrid.NY':1,'ComputationalGrid.NZ':2,
              'ComputationalGrid.DX':1.,'ComputationalGrid.DY':1.,'ComputationalGrid.DZ':1.,
              'ComputationalGrid.Lower.X':0.,'ComputationalGrid.Lower.Y':0.,'ComputationalGrid.Lower.Z':-2.}
    grid = HydroGrid.from_parflow(config,active=np.ones((2,1,2),dtype=bool))
    xyz = grid.centers.reshape(-1,3)[::-1]
    phi = np.array([.2,.25,.3,.35])
    rho = 10/phi**2  # Known fully saturated formation, no surface conductivity.
    result = prepare_hydro_updates(xyz,rho,grid,petrophysics={
        'saturation':1.,'rho_fluid':10.,'m':2.,'n':2.},
        field_transforms={'porosity':'porosity'},baselines={'porosity':np.full((2,1,2),.1)},
        max_distance=.1)
    source = tmp_path/'source'
    source.mkdir()
    target = write_parflow_inputs(config,source,tmp_path/'updated',{'porosity':result['porosity'].values})
    np.testing.assert_allclose(read_pfb(str(target/'hydro_updates/porosity.pfb')),phi[::-1].reshape(2,1,2))
    assert result['porosity'].updated.all()


def test_modflow_coordinates_follow_native_top_down_grid():
    flopy = pytest.importorskip('flopy')
    sim = flopy.mf6.MFSimulation()
    model = flopy.mf6.ModflowGwf(sim)
    flopy.mf6.ModflowGwfdis(model,nlay=2,nrow=1,ncol=2,top=10,botm=[5,0],idomain=[[[1,0]],[[1,1]]])
    model.modelgrid.set_coord_info(xoff=100,yoff=200,angrot=30)
    grid = HydroGrid.from_modflow(model)
    np.testing.assert_array_equal(grid.centers[:,0,0,2],[7.5,2.5])
    assert not grid.active[0,0,1]
    assert grid.centers[...,0].min() > 99


def test_scalar_transform_is_not_silently_broadcast():
    grid = HydroGrid(np.array([[[[0.,0.,0.],[1.,0.,0.]]]]),np.ones((1,1,2),dtype=bool))
    with pytest.raises(ValueError,match='one value per valid'):
        prepare_hydro_updates([[0,0,0],[1,0,0]],[1000,1000],grid,
            petrophysics={'porosity':.3,'rho_fluid':10,'m':2,'n':2},
            field_transforms={'hydraulic_conductivity':lambda state:1.},
            baselines={'hydraulic_conductivity':np.ones((1,1,2))},max_distance=1)
