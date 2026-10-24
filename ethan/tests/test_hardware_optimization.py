import math
import numpy as np
import pytest
import stim

from iqm_surface_code.hardware_optimization import r_clifford,native_stim,optimize_native,native_noise,dd_strategy


@pytest.mark.parametrize('angle',[-math.pi,-math.pi/2,0,math.pi/2,math.pi,2*math.pi])
@pytest.mark.parametrize('phase',[0,math.pi/2,math.pi,3*math.pi/2,5*math.pi/2])
def test_native_r_clifford_matrix(angle,phase):
    qiskit=pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator
    qc=qiskit.QuantumCircuit(1);qc.r(angle,phase,0)
    actual=Operator(qc).data
    expected=stim.gate_data(r_clifford(angle,phase)).unitary_matrix
    phase_factor=np.vdot(expected,actual)/2
    assert np.allclose(actual,phase_factor*expected,atol=1e-7)


@pytest.mark.parametrize('phase',[math.pi/4,3*math.pi/4,5*math.pi/4,7*math.pi/4])
def test_native_diagonal_half_turn(phase):
    pytest.importorskip('qiskit')
    from qiskit.quantum_info import Operator
    from qiskit import QuantumCircuit
    q=QuantumCircuit(1);q.r(math.pi,phase,0)
    u=Operator(q).data;v=stim.gate_data(r_clifford(math.pi,phase)).unitary_matrix
    assert np.allclose(u,np.vdot(v,u)/2*v,atol=1e-7)


def test_nonclifford_rejected_and_identity_axis_irrelevant():
    assert r_clifford(0,.123)=='I'
    with pytest.raises(ValueError):r_clifford(math.pi/4,0)


def offline_setup(basis):
    pytest.importorskip('iqm.qiskit_iqm');pytest.importorskip('networkx')
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    from iqm.qiskit_iqm.iqm_provider import IQMBackend
    from iqm_surface_code.benchmark_garnet import select_native_layout
    from iqm_surface_code.run_on_hardware import prepare_qiskit
    from iqm_surface_code.surface_code import make_stim_circuit
    from iqm_surface_code.calibration import CalibrationSnapshot,QubitCalibration
    b=IQMFakeGarnet();b.serialize_circuit=IQMBackend.serialize_circuit.__get__(b)
    edges=[(b.index_to_qubit_name(a),b.index_to_qubit_name(z)) for a,z in b.coupling_map.get_edges()]
    mapping,_=select_native_layout(make_stim_circuit(3,2,basis,gate_basis='cz'),edges,b.physical_qubits)
    cal=CalibrationSnapshot({q:QubitCalibration(.001,.02,.002,50000,20000) for q in b.physical_qubits},
        {CalibrationSnapshot.edge_key(a,z):.005 for a,z in edges},{'1q':16,'2q':40,'measure':500,'reset':300000})
    observations=[]
    for q in b.physical_qubits:
        for key,value in ((f'metrics.ssro.measure_fidelity.constant.{q}.fidelity',.99),
                          (f'metrics.rrc.measure.constant.{q}.qndness',.95),
                          (f'metrics.rrc.measure.constant.{q}.leakage',.01)):
            observations.append(dict(dut_field=key,value=value,invalid=False))
    prepared=prepare_qiskit(b,rounds=2,basis=basis,qubit_map=mapping,calibration=cal)
    return b,mapping,cal,dict(observations=observations),prepared


@pytest.mark.parametrize('basis',['X','Z'])
def test_native_remap_model_and_source_frame_match_on_arbitrary_records(basis):
    b,mapping,cal,quality,p=offline_setup(basis)
    qc,report=optimize_native(p['circuit'])
    assert report['after'].get('r',0)<=report['before']['r']
    assert report['after'].get('barrier',0)==0
    native,columns=native_stim(qc,p['stim_circuit'],mapping,b)
    raw=np.random.default_rng(3).integers(0,2,size=(128,native.num_measurements),dtype=np.uint8).astype(bool)
    src=p['stim_circuit'].compile_m2d_converter().convert(measurements=raw,separate_observables=True)
    dst=native.compile_m2d_converter().convert(measurements=raw[:,columns],separate_observables=True)
    assert np.array_equal(src[0],dst[0]) and np.array_equal(src[1],dst[1])
    model=native_noise(native,b,cal,quality,phase_round=.15,backaction_scale=1)
    assert model.without_noise()==native
    assert model.detector_error_model(approximate_disjoint_errors=True).num_detectors==16
    assert model.num_measurements==p['stim_circuit'].num_measurements
    # Full native ideal execution independently reproduces the source record frame.
    actual=native.compile_sampler(seed=9).sample(64)
    det,obs=native.compile_m2d_converter().convert(measurements=actual,separate_observables=True)
    assert not det.any() and not obs.any()


def test_adaptive_strategy_is_explicit_and_bounded():
    pytest.importorskip('iqm.iqm_client')
    s=dd_strategy('adaptive',['QB1','QB4'])
    assert s.target_qubits==frozenset(['QB1','QB4'])
    assert len(s.gate_sequences)==3 and s.skip_leading_wait
    assert dd_strategy('off',[]) is None
    with pytest.raises(ValueError):dd_strategy('unknown',[])


def test_decoder_readout_prior_follows_compiler_trace_and_excludes_unused_probes():
    b,mapping,cal,quality,p=offline_setup('Z')
    native,_=native_stim(p['circuit'],p['stim_circuit'],mapping,b)
    for name in b.physical_qubits:
        quality['observations'].append(dict(dut_field=f'metrics.ssro.measure.constant.{name}.fidelity',value=.98))
    trace=[dict(gate='measure',implementation='constant',locus=[b.index_to_qubit_name(t.value)],key='returned')
        for i in native.flattened() if i.name=='M' for t in i.targets_copy()]
    trace.append(dict(gate='measure_fidelity',implementation='constant',locus=list(b.physical_qubits),key='__ONLY_PROBE'))
    model=native_noise(native,b,cal,quality,measurement_operations=trace)
    assert all(i.gate_args_copy()==pytest.approx([.02]) for i in model.flattened() if i.name=='M')
    with pytest.raises(ValueError,match='does not cover'):
        native_noise(native,b,cal,quality,measurement_operations=trace[1:])


@pytest.mark.parametrize('basis',['X','Z'])
@pytest.mark.parametrize('mode',['retain','fenced','fused'])
def test_protective_compilation_preserves_exact_detector_frame(basis,mode):
    b,mapping,cal,quality,p=offline_setup(basis)
    qc,report=optimize_native(p['circuit'],mode=mode)
    assert qc.count_ops()['cz']==48
    assert qc.count_ops().get('barrier',0)>0
    native,columns=native_stim(qc,p['stim_circuit'],mapping,b)
    raw=np.random.default_rng(47).integers(0,2,(32,native.num_measurements),dtype=np.uint8).astype(bool)
    a=p['stim_circuit'].compile_m2d_converter().convert(measurements=raw,separate_observables=True)
    z=native.compile_m2d_converter().convert(measurements=raw[:,columns],separate_observables=True)
    assert np.array_equal(a[0],z[0]) and np.array_equal(a[1],z[1])
    actual=native.compile_sampler(seed=12).sample(64)
    det,obs=native.compile_m2d_converter().convert(measurements=actual,separate_observables=True)
    assert not det.any() and not obs.any()


def test_terminal_fusion_drops_dead_uncompute_but_rejects_repeated_measurement():
    pytest.importorskip('iqm.qiskit_iqm')
    from qiskit import QuantumCircuit
    q=QuantumCircuit(2,2);q.cz(0,1);q.measure(0,0);q.r(math.pi/2,0,0);q.measure(1,1)
    optimized,_=optimize_native(q,mode='fused')
    assert not optimized.count_ops().get('r',0) and optimized.count_ops()['measure']==2
    q.measure(0,0)
    with pytest.raises(ValueError,match='repeated measurement'):
        optimize_native(q,mode='fused')
