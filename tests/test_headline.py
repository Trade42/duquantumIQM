import numpy as np
import pytest
from challenge.benchmark_headline import challenge_fit,fit_many,ler_difference,choose
from challenge.decoding import fit_ler


def test_primary_is_exact_challenge_equation_even_when_fit_bad():
    f=challenge_fit([1,2,3],[20,150,260],[1000]*3)
    assert not f['fit_spam'] and f['amplitude']==1
    assert f['stationary_model_rejected_5pct']
    assert f['epsilon_per_round']==fit_ler([1,2,3],[20,150,260],[1000]*3)['epsilon_per_round']


def test_bootstrap_mle_matches_repository_mle_and_zero_boundary():
    counts=np.array([[0,0,0],[30,80,140],[100,190,250],[900,900,900]])
    actual=fit_many(counts,np.full(3,1000))
    expected=[fit_ler([1,2,3],k,[1000]*3)['epsilon_per_round'] for k in counts]
    assert np.allclose(actual,expected,atol=1e-7)


def test_same_shot_identical_decoders_have_zero_difference_interval():
    failures={r:{0:np.arange(1000)%11==0,1:np.arange(1000)%13==0} for r in (1,2,3)}
    result=ler_difference(failures,failures,paired=True,draws=100)
    assert result['absolute_ler_reduction']==0
    assert result['ci']==[0,0]


def test_no_forced_physical_combination():
    rows=[dict(basis='X',profile=p,family='mwpm',fit=dict(epsilon_per_round=e))
          for p,e in [('ethan',.03),('dara',.05),('combined',.04)]]
    assert choose(rows,'X')['profile']=='ethan'
    assert choose(rows,'X',('dara','combined'))['profile']=='combined'


def test_decoder_seal_rejects_changed_weights_before_replay(tmp_path):
    from challenge.benchmark_headline import seal_inference,write_json
    selection=dict(profile='ethan',family='mwpm')
    write_json(tmp_path/'frozen_selection.json',dict(arms={'X':{'candidate':selection}}))
    manifest=dict(cases=[])
    for r in (1,2,3):
        directory=tmp_path/f'decoder{r}'
        directory.mkdir();(directory/'model.dem').write_text('error(0.1) D0 L0\n')
        manifest['cases'].append(dict(phase='pilot',basis='X',rounds=r,profile='ethan',
                                     validation={'mwpm':dict(decoder_dir=directory.name)}))
    seal_inference(tmp_path,manifest)
    seal_inference(tmp_path,manifest,verify_only=True)
    (tmp_path/'decoder2/model.dem').write_text('error(0.2) D0 L0\n')
    with pytest.raises(ValueError,match='Frozen decoder state changed'):
        seal_inference(tmp_path,manifest,verify_only=True)


def test_neural_expansion_cannot_consume_confirmation_data(tmp_path):
    from challenge.benchmark_headline import extend_neural
    with pytest.raises(ValueError,match='precede every confirmation'):
        extend_neural(tmp_path,dict(cases=[dict(phase='confirmation')]))


def test_policy_rejects_changed_calibration_before_creating_output(tmp_path):
    pytest.importorskip('iqm.qiskit_iqm')
    from types import SimpleNamespace
    from challenge.benchmark_headline import execute_policy
    args=SimpleNamespace(policy_output=tmp_path/'new-run',policy_shots=2)
    with pytest.raises(ValueError,match='calibration changed'):
        execute_policy(args,tmp_path,dict(calibration_set_id='previous'),None,None,None,None,None,'unused',
                       active_calibration_set_id='changed')
    assert not args.policy_output.exists()


@pytest.mark.parametrize('profile',['ethan','dara','iyanu','combined'])
@pytest.mark.parametrize('basis',['X','Z'])
def test_physical_preparation_preserves_no_reset_frames(tmp_path,monkeypatch,profile,basis):
    pytest.importorskip('iqm_surface_code')
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm_surface_code.benchmark_garnet import select_native_layout
    from iqm_surface_code.surface_code import make_stim_circuit
    from iqm_surface_code.calibration import CalibrationSnapshot,QubitCalibration
    import iqm_surface_code.hardware_optimization as optimization
    from challenge.benchmark_headline import prepare_case
    import stim
    backend=IQMFakeGarnet()
    edges=[(backend.index_to_qubit_name(a),backend.index_to_qubit_name(b)) for a,b in backend.coupling_map.get_edges()]
    mapping,_=select_native_layout(make_stim_circuit(3,2,basis,gate_basis='cz'),edges,backend.physical_qubits)
    cal=CalibrationSnapshot({q:QubitCalibration(.001,.02,.002,50000,20000) for q in backend.physical_qubits},
        {CalibrationSnapshot.edge_key(a,b):.005 for a,b in edges},{'1q':16,'2q':40,'measure':500,'reset':300000})
    # Model construction is independently tested in Ethan's suite; stub only
    # the pulse-specific observation source absent from IQMFakeGarnet.
    monkeypatch.setattr(optimization,'native_noise',lambda circuit,*a,**kw:circuit)
    class OfflineExecutor:
        def compile(self,*a,**kw):return None,None,dict(measurement_operations=[])
    case=dict(id='offline',basis=basis,rounds=2,profile=profile,shots=16)
    qc,_,_=prepare_case(tmp_path,case,backend,OfflineExecutor(),cal,{},mapping)
    source=stim.Circuit.from_file(tmp_path/'offline.source.stim')
    native=stim.Circuit.from_file(tmp_path/'offline.native.stim')
    raw=np.random.default_rng(31).integers(0,2,(64,source.num_measurements),dtype=np.uint8).astype(bool)
    a=source.compile_m2d_converter().convert(measurements=raw,separate_observables=True)
    b=native.compile_m2d_converter().convert(measurements=raw[:,case['native_measurement_columns']],separate_observables=True)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    dets,obs=native.compile_m2d_converter().convert(measurements=native.compile_sampler(seed=11).sample(64),separate_observables=True)
    assert not dets.any() and not obs.any()
    assert not qc.count_ops().get('reset') and not qc.count_ops().get('swap')
