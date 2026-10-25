import numpy as np
import pytest
import stim
from challenge.decoders.hybrid import build_decoder, save_decoder, load_decoder


def tiny():
    return stim.Circuit("R 0 1\nX_ERROR(0.1) 0\nX_ERROR(0.08) 1\nM 0 1\nDETECTOR rec[-2]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-2]")


@pytest.mark.parametrize("family", ["mwpm","mwpm_fit","ensemble","ensemble_fit","ml","ml_fit","empirical_1","empirical_10","posterior_1","posterior_10","posterior_100","coherence"])
def test_roundtrip_and_order_on_held_out_shots(tmp_path, family):
    if family in {'ml','ml_fit'} or family.startswith('posterior_'):
        pytest.importorskip('iqm_surface_code')
    model=tiny()
    td,to=model.compile_detector_sampler(seed=9).sample(400,separate_observables=True)
    d,o=model.compile_detector_sampler(seed=13).sample(80,separate_observables=True)
    options=dict(coherence_models={300:model,800:model}) if family=='coherence' else {}
    decoder,dem=build_decoder(model,family,train_dets=td,train_obs=to,**options)
    prediction=decoder.decode_batch(d)
    assert prediction.shape==o.shape and np.array_equal(prediction,o)
    assert np.array_equal(decoder.decode_batch(d[::-1]),prediction[::-1])
    assert decoder.decode_batch(d[:0]).shape==(0,1)
    save_decoder(decoder,dem,tmp_path/'frozen')
    assert np.array_equal(load_decoder(tmp_path/'frozen').decode_batch(d),prediction)


def test_invalid_inputs_and_unsupported_correlations():
    decoder,_=build_decoder(tiny(),'mwpm')
    for d in [np.zeros((2,3)),np.array([[0,2]]),np.zeros(2)]:
        with pytest.raises(ValueError):decoder.decode_batch(d)
    with pytest.raises(ValueError,match='correlation'):
        decoder.decode_batch(np.zeros((1,2)),enable_correlations=False)
    with pytest.raises(ValueError,match='independent training'):
        build_decoder(tiny(),'empirical_10')


def test_empirical_misses_use_explicit_fallback_without_dropping_shots():
    decoder,_=build_decoder(tiny(),'empirical_30',train_dets=np.zeros((10,2)),train_obs=np.zeros((10,1)))
    d=np.array([[0,0],[1,0],[0,1],[1,1]],bool)
    assert np.array_equal(decoder.decode_batch(d)[:,0],d[:,0])
    assert decoder.decoder.coverage==0


@pytest.mark.parametrize('half_fault',[False,True])
def test_exact_posterior_keeps_joint_separators_and_deterministic_faults(half_fault):
    pytest.importorskip('iqm_surface_code')
    from itertools import product
    from challenge.decoders.hybrid import model_posterior
    # The observable appears twice in the first joint fault and cancels.
    dem=stim.DetectorErrorModel('error(0.13) D0 L0 ^ D1 L0\nerror(0.08) D0 L0\nerror(1) D1'+('\nerror(0.5) D0' if half_fault else ''))
    probabilities=[.13,.08,1.]+([.5] if half_fault else [])
    masks=[3,5,2]+([1] if half_fault else [])
    joint=np.zeros(8)
    for flags in product((0,1),repeat=len(probabilities)):
        mask=0;probability=1.
        for flag,p,m in zip(flags,probabilities,masks):
            if flag:mask^=m
            probability*=p if flag else 1-p
        joint[mask]+=probability
    expected=joint[4:]/(joint[:4]+joint[4:])
    assert np.allclose(model_posterior(str(dem)),expected,atol=1e-12)


@pytest.mark.parametrize('family',['mlp','conv'])
def test_neural_inference_state_roundtrip_without_pickle(tmp_path,family):
    torch=pytest.importorskip('torch')
    from dataclasses import asdict
    from challenge.decoders.hybrid import FrozenAdapter
    from challenge.surface_code import make_stim_circuit
    from challenge.noise_models import add_noise,NoiseModel
    model=add_noise(make_stim_circuit(3,1,'Z',True,gate_basis='cz'),NoiseModel.uniform(.003))
    if family=='mlp':
        from iyanu.neural_gpu import NeuralConfig,NeuralDecoder,_mlp
        config=NeuralConfig(hidden=16,layers=1,steps=2,batch=8,device='cpu')
        decoder=NeuralDecoder(_mlp(model.num_detectors,1,16,1),model.num_detectors,1,torch.device('cpu'),config,{})
    else:
        from iyanu.conv_decoders import ConvConfig,ConvDecoder
        config=ConvConfig(channels=4,layers=1,head=16,steps=2,batch=8,device='cpu')
        decoder=ConvDecoder(model,config)
        decoder._headline_circuit=model
    adapter=FrozenAdapter(decoder,model.num_detectors,metadata=dict(family=family,config=asdict(config)))
    d,_=model.compile_detector_sampler(seed=19).sample(40,separate_observables=True)
    prediction=adapter.decode_batch(d)
    assert np.array_equal(adapter.decode_batch(d,enable_correlations=False),prediction)
    with pytest.raises(ValueError,match='does not apply'):
        adapter.decode_batch(d,enable_correlations=True)
    save_decoder(adapter,model.detector_error_model(approximate_disjoint_errors=True),tmp_path/'frozen')
    assert np.array_equal(load_decoder(tmp_path/'frozen').decode_batch(d),prediction)
