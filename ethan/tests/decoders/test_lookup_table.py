from itertools import product
import numpy as np
import pytest
import stim

from iqm_surface_code.decoders.lookup_table import build_lookup_table, fwht_inplace, LookupTableDecoder
from iqm_surface_code.decoding import decode_syndromes
from iqm_surface_code.noise_models import NoiseModel, add_noise
from iqm_surface_code.surface_code import make_stim_circuit


def brute_force(dem):
    n=dem.num_detectors
    faults=[]
    for inst in dem.flattened():
        if inst.type!='error':
            continue
        mask=0
        for t in inst.targets_copy():
            if t.is_relative_detector_id(): mask ^= 1<<t.val
            elif t.is_logical_observable_id(): mask ^= 1<<(n+t.val)
        faults.append((inst.args_copy()[0],mask))
    joint=np.zeros(1<<(n+1))
    for chosen in product((0,1),repeat=len(faults)):
        p,mask=1.,0
        for take,(rate,event) in zip(chosen,faults):
            p*=rate if take else 1-rate
            if take: mask^=event
        joint[mask]+=p
    return joint


@pytest.mark.parametrize('probabilities',[(.1,.2,.07),(.5,.2,1.),(.8,0.,.9)])
def test_exact_full_table_matches_fault_enumeration(probabilities):
    p,q,r=probabilities
    dem=stim.DetectorErrorModel(f'error({p}) D0 L0\nerror({q}) D1\nerror({r}) D0 ^ D1 L0')
    joint=brute_force(dem)
    decoder=build_lookup_table(dem)
    half=1<<dem.num_detectors
    expected=(joint[half:]>joint[:half]).astype(np.uint8)
    assert np.array_equal(decoder.table,expected)
    assert decoder.metadata['model_bayes_failure']==pytest.approx(np.minimum(joint[:half],joint[half:]).sum())
    assert decoder.metadata['training_shots']==0
    assert not decoder.table.flags.writeable


def test_separated_components_remain_one_correlated_fault():
    dem=stim.DetectorErrorModel('error(0.1) D0 ^ D1 L0')
    decoder=build_lookup_table(dem)
    assert decoder.decode_batch(np.array([[0,0],[1,1]])).tolist()==[[False],[True]]
    assert decoder.metadata['model_bayes_failure']==pytest.approx(0,abs=1e-14)


def test_repeat_offsets_and_save_load(tmp_path):
    dem=stim.DetectorErrorModel('''repeat 2 {
        error(0.1) D0 L0
        shift_detectors 1
    }''')
    decoder=build_lookup_table(dem)
    decoder.save(tmp_path/'lut.npz')
    restored=LookupTableDecoder.load(tmp_path/'lut.npz')
    dets=np.array(list(product((0,1),repeat=2)),dtype=bool)
    assert np.array_equal(decoder.decode_batch(dets),restored.decode_batch(dets))
    assert decoder.decode_batch(dets[::-1]).shape==(4,1)


def test_transform_involution():
    a=np.random.default_rng(4).normal(size=64)
    assert np.allclose(fwht_inplace(fwht_inplace(a.copy()))/64,a)
    with pytest.raises(ValueError): fwht_inplace(np.zeros(3))


def test_binary_validation_empty_and_unsupported_regimes():
    decoder=build_lookup_table(stim.DetectorErrorModel('error(.1) D0 L0'))
    assert decoder.decode_batch(np.empty((0,1))).shape==(0,1)
    for value in (np.array([1]),np.zeros((2,2)),np.array([[2]]),np.array([[float('nan')]])):
        with pytest.raises(ValueError): decoder.decode_batch(value)
    with pytest.raises(ValueError,match='correlations'): decoder.decode_batch([[0]],enable_correlations=False)
    with pytest.raises(ValueError,match='one logical'): build_lookup_table(stim.DetectorErrorModel('error(.1) D0 L1'))
    with pytest.raises(ValueError,match='limit'): build_lookup_table(stim.DetectorErrorModel('error(.1) D24 L0'))


@pytest.mark.parametrize('basis',['X','Z'])
@pytest.mark.parametrize('no_reset',[True,False])
def test_surface_predictions_noiseless_and_matched_simulation(basis,no_reset):
    ideal=make_stim_circuit(3,1,basis,no_reset=no_reset,gate_basis='cz')
    noisy=add_noise(ideal,NoiseModel.uniform(.003))
    lut=build_lookup_table(noisy)
    dets,obs=noisy.compile_detector_sampler(seed=23).sample(40000,separate_observables=True)
    result=decode_syndromes(dict(det_events=dets,obs_flips=obs,stim_circuit=ideal),noisy,matcher=lut)
    assert result.probability==pytest.approx(lut.metadata['model_bayes_failure'],abs=.006)
    assert not lut.decode_batch(np.zeros((4,ideal.num_detectors),dtype=bool)).any()
    altered=obs.copy();altered[:]=~altered
    # Predictions depend on dets alone. Truth changes scoring only.
    result2=decode_syndromes(dict(det_events=dets,obs_flips=altered),noisy,matcher=lut)
    assert result.errors+result2.errors==len(obs)
