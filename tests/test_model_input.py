"""Round-trip exported solver inputs without requiring solver executables."""
import numpy as np
import pytest

from PyHydroGeophysX.model_input import write_modflow6_inputs, write_parflow_inputs


def test_modflow_roundtrip_and_source_unchanged(tmp_path):
    flopy = pytest.importorskip('flopy')
    sim = flopy.mf6.MFSimulation(sim_ws=str(tmp_path/'source'))
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim)
    model = flopy.mf6.ModflowGwf(sim, modelname='flow')
    flopy.mf6.ModflowGwfdis(model, nlay=2, nrow=2, ncol=3, top=10, botm=[5, 0])
    flopy.mf6.ModflowGwfnpf(model, k=1.)
    flopy.mf6.ModflowGwfic(model, strt=8.)
    flopy.mf6.ModflowGwfsto(model, ss=1e-5, sy=.1)
    sim.write_simulation(silent=True)
    original = (tmp_path/'source'/'flow.npf').read_bytes()
    values = np.arange(12.).reshape(2,2,3)+1
    target = write_modflow6_inputs(sim, tmp_path/'updated', {'hydraulic_conductivity':values,
                                   'initial_head': np.full_like(values, 7.)})
    loaded = flopy.mf6.MFSimulation.load(sim_ws=str(target), verbosity_level=0).get_model()
    np.testing.assert_array_equal(loaded.npf.k.array, values)
    np.testing.assert_array_equal(loaded.ic.strt.array, 7.)
    np.testing.assert_array_equal(model.npf.k.array, 1.)
    assert (tmp_path/'source'/'flow.npf').read_bytes() == original
    with pytest.raises(FileExistsError):
        write_modflow6_inputs(sim, target, {'initial_head': values})
    with pytest.raises(ValueError):
        write_modflow6_inputs(sim, tmp_path/'bad', {'hydraulic_conductivity': -values})
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError, match='shape'):
        write_modflow6_inputs(sim, tmp_path/'bad_shape', {'initial_head': values.ravel()})
    model.npf.k33overk.set_data(True)
    with pytest.raises(ValueError, match='k33overk'):
        write_modflow6_inputs(sim, tmp_path/'bad_ratio', {'vertical_conductivity':values})
    bottoms = np.broadcast_to(np.array([4.,-1.])[:,None,None],values.shape)
    moved = write_modflow6_inputs(sim,tmp_path/'geometry',{'bottom_elevation':bottoms})
    reloaded = flopy.mf6.MFSimulation.load(sim_ws=str(moved),verbosity_level=0).get_model()
    np.testing.assert_array_equal(reloaded.dis.botm.array,bottoms)
    np.testing.assert_array_equal(model.dis.botm.array[:,0,0],[5,0])
    with pytest.raises(ValueError,match='decrease'):
        write_modflow6_inputs(sim,tmp_path/'crossed',{'bottom_elevation':bottoms[::-1]})


def test_parflow_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr('sys.argv', ['parflow-test'])
    parflow = pytest.importorskip('parflow')
    from parflow.tools.io import read_pfb
    settings = {'ComputationalGrid.NX':3, 'ComputationalGrid.NY':2, 'ComputationalGrid.NZ':2,
                'ComputationalGrid.DX':2., 'ComputationalGrid.DY':3., 'ComputationalGrid.DZ':4.,
                'ComputationalGrid.Lower.X':10., 'ComputationalGrid.Lower.Y':20.,
                'ComputationalGrid.Lower.Z':-8., 'Process.Topology.P':1,
                'Process.Topology.Q':1, 'Process.Topology.R':1,
                'GeomInput.Names':'box', 'GeomInput.box.InputType':'Box',
                'GeomInput.box.GeomName':'domain', 'Domain.GeomName':'domain'}
    run = settings
    source = tmp_path/'source'
    source.mkdir()
    (source/'forcing.txt').write_text('preserve me')
    values = np.arange(12.).reshape(2,2,3)+1
    target = write_parflow_inputs(run, source, tmp_path/'updated',
        {'permeability':values, 'porosity':np.full_like(values,.3), 'initial_pressure':-values})
    np.testing.assert_array_equal(read_pfb(str(target/'hydro_updates/permeability.pfb')),values)
    np.testing.assert_array_equal(read_pfb(str(target/'hydro_updates/initial_pressure.pfb')),-values)
    from parflow.tools.io import read_pfidb
    loaded = read_pfidb(str(target/'updated_model.pfidb'))
    assert loaded['Geom.domain.Perm.Type'] == 'PFBFile'
    assert loaded['Geom.domain.Porosity.FileName'] == 'hydro_updates/porosity.pfb'
    assert loaded['ICPressure.Type'] == 'PFBFile'
    assert (target/'forcing.txt').read_text() == 'preserve me'
    assert not (source/'hydro_updates').exists()
    assert 'Geom.domain.Perm.Type' not in run
    import struct
    with (target/'hydro_updates/permeability.pfb').open('rb') as stream:
        assert struct.unpack('>3d', stream.read(24)) == (10.,20.,-8.)
        assert struct.unpack('>3i', stream.read(12)) == (3,2,2)
        assert struct.unpack('>3d', stream.read(24)) == (2.,3.,4.)
    with pytest.raises(ValueError):
        write_parflow_inputs(run, source, tmp_path/'bad', {'porosity':values})
    with pytest.raises(FileExistsError):
        write_parflow_inputs(run, source, target, {'permeability':values})
    invalid = dict(run, **{'Geom.domain.Perm.FileName':'../outside.pfb'})
    with pytest.raises(ValueError, match='relative'):
        write_parflow_inputs(invalid, source, tmp_path/'escape', {'permeability':values})
    assert not (tmp_path/'escape').exists()
