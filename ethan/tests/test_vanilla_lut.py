from collections import Counter
import json
import numpy as np
import pytest

from iqm_surface_code.benchmark_vanilla_lut import vanilla_circuit,memory_syndromes,fitted_results,fit_postfirst_ler
from iqm_surface_code.decoders.lookup_table import build_lookup_table
from iqm_surface_code.noise_models import NoiseModel,add_noise
from iqm_surface_code.surface_code import make_stim_circuit


@pytest.mark.parametrize('basis',['X','Z'])
@pytest.mark.parametrize('rounds',[1,2,3])
def test_canonical_no_reset_native_schedule(basis,rounds):
    c=vanilla_circuit(rounds,basis)
    assert c.num_detectors==8*rounds
    assert c.num_measurements==8*rounds+9
    assert sum(len(i.targets_copy())//2 for i in c if i.name=='CZ')==24*rounds
    assert not any(i.name in {'CX','MR','MRX'} for i in c)
    samples=c.compile_sampler(seed=7).sample(32)
    dets,obs=c.compile_m2d_converter().convert(measurements=samples,separate_observables=True)
    assert not dets.any() and not obs.any()
    optimized=make_stim_circuit(3,rounds,basis,gate_basis='cz')
    h=lambda a:sum(len(i.targets_copy()) for i in a if i.name=='H')
    assert h(c)>h(optimized)


def test_original_memory_order_and_lut_scoring():
    c=vanilla_circuit(1,'X');samples=c.compile_sampler(seed=31).sample(32)
    records=[''.join(str(int(v)) for v in row[::-1]) for row in samples]
    counts=Counter(records)
    raw,s=memory_syndromes(counts,records,c,32)
    assert np.array_equal(raw,samples)
    lut=build_lookup_table(add_noise(c,NoiseModel.uniform(.003)))
    assert not lut.decode_batch(s['det_events']).any()
    with pytest.raises(ValueError,match='memory'): memory_syndromes(counts,records[:-1],c,32)


def test_primary_challenge_and_secondary_spam_fits_are_distinct():
    cases=[dict(basis='Z',rounds=r,shots=10000,analysis=dict(errors=k)) for r,k in [(1,1000),(2,1800),(3,2440)]]
    fits=fitted_results(cases)
    assert len(fits)==2
    assert fits[0]['fit_role']=='primary challenge equation'
    assert fits[0]['epsilon_per_round']==pytest.approx(.1,abs=1e-6)
    assert fits[1]['fit_spam']
    assert fits[0]['deviance']<1e-7


def test_first_round_transient_is_explicit_and_profiled():
    cases=[dict(basis='X',rounds=r,shots=10000,analysis=dict(errors=k)) for r,k in [(1,1400),(2,3200),(3,4100)]]
    primary,diagnostic=fitted_results(cases)
    assert primary['deviance']>10
    assert diagnostic['epsilon_per_round']==pytest.approx(.25,abs=1e-6)
    assert diagnostic['first_round_visibility']==pytest.approx(.72,abs=1e-6)
    assert diagnostic['deviance']<1e-7
    assert diagnostic['ci95_profile'][0]<.25<diagnostic['ci95_profile'][1]
    assert diagnostic['round_offset']==1
    with pytest.raises(ValueError): fit_postfirst_ler([1,2],[1,2],[100,100])
    with pytest.raises(ValueError): fit_postfirst_ler([1,2,3],[1,2,101],[100,100,100])
