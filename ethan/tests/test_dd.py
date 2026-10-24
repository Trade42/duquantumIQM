from dataclasses import dataclass
import sys
import types

import numpy as np
import pytest
import stim

from iqm_surface_code.calibration import CalibrationSnapshot,QubitCalibration,synthetic_snapshot
from iqm_surface_code.dynamical_decoupling import (DDConfig,plan_dd,insert_dd,iqm_compilation_options,
    normalize_dd,is_dd_pulse)
from iqm_surface_code.noise_models import NoiseModel,add_noise
from iqm_surface_code.surface_code import make_stim_circuit
from iqm_surface_code.experiment import simulate_memory


def fixture_model(*,gate_error=0.,t1=1e12,t2=1e12):
    q = QubitCalibration(gate_error,0.,0.,t1,t2)
    c = CalibrationSnapshot({"QB1":q,"QB2":q},{"QB1|QB2":0.},
        {"1q":20.,"2q":80.,"measure":700.,"reset":1000.})
    return NoiseModel.uniform(0).with_changes(calibration=c,coherence=True)


def echo_circuit():
    return stim.Circuit("QUBIT_COORDS(1,1) 0\nQUBIT_COORDS(2,2) 1\nR 0 1\nTICK\nH 0\nTICK\nM 1\nTICK\nH 0\nTICK\nM 0\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]")


@pytest.mark.parametrize("sequence",["xx","xy4","xy8"])
def test_dd_is_identity_within_existing_idle_window(sequence):
    ideal = echo_circuit()
    plan = plan_dd(ideal,fixture_model(),sequence,qubit_map={0:"QB1",1:"QB2"})
    assert len(plan.windows) == 1 and plan.windows[0].qubit == 0
    assert plan.windows[0].duration_ns == 700
    assert abs(plan.windows[0].signed_time_ns) < 1e-10
    unitary = stim.Circuit()
    for g in plan.windows[0].pattern:
        unitary.append(g,[0])
    assert unitary.to_tableau() == stim.Circuit("I 0").to_tableau()
    dd_ideal = insert_dd(ideal,plan)
    assert dd_ideal.num_ticks == ideal.num_ticks
    assert dd_ideal.num_measurements == ideal.num_measurements
    dets,obs = dd_ideal.compile_detector_sampler(seed=1).sample(100,separate_observables=True)
    assert not dets.any() and not obs.any()


def test_dd_fit_short_windows_and_no_active_or_measured_qubit_pulses():
    model = fixture_model()
    ideal = echo_circuit()
    plan = plan_dd(ideal,model,DDConfig(pulse_duration_ns=400),qubit_map={0:"QB1",1:"QB2"})
    assert not plan.windows
    plan = plan_dd(ideal,model,DDConfig(target="all"),qubit_map={0:"QB1",1:"QB2"})
    assert all(w.qubit == 0 for w in plan.windows)  # Leading/trailing ancilla waits skipped.
    with pytest.raises(ValueError,match="does not match"):
        insert_dd(ideal+stim.Circuit("I 0"),plan)
    with pytest.raises(ValueError,match="requires calibration"):
        plan_dd(ideal,NoiseModel.uniform(),"xx")
    assert normalize_dd(False) is None and normalize_dd("off") is None
    with pytest.raises(ValueError):
        DDConfig(sequence="x")


def test_markov_decay_not_suppressed_and_pulse_errors_counted():
    ideal,model = echo_circuit(),fixture_model(gate_error=.001,t1=30000,t2=40000)
    mapping = {0:"QB1",1:"QB2"}
    before = add_noise(ideal,model,qubit_map=mapping)
    after = add_noise(ideal,model,qubit_map=mapping,dd="xx")
    channel = list(model.coherence_channel(0,700,mapping))
    assert any(i.name == "PAULI_CHANNEL_1" and np.allclose(i.gate_args_copy(),channel) for i in before)
    assert any(i.name == "PAULI_CHANNEL_1" and np.allclose(i.gate_args_copy(),channel) for i in after)
    assert sum(is_dd_pulse(i) for i in after) == 2
    assert sum(i.name == "PAULI_CHANNEL_1" for i in after) == sum(i.name == "PAULI_CHANNEL_1" for i in before)+2


def test_slow_detuning_refocused_without_removing_background():
    ideal = echo_circuit()
    model = fixture_model().with_changes(idle_detuning_std_rad_per_ns=.003)
    before = add_noise(ideal,model,qubit_map={0:"QB1",1:"QB2"})
    after = add_noise(ideal,model,qubit_map={0:"QB1",1:"QB2"},dd="xx")
    assert any(i.name == "Z_ERROR" and i.gate_args_copy()[0] > .4 for i in before)
    assert not any(i.name == "Z_ERROR" and i.targets_copy()[0].value == 0 for i in after)


@pytest.mark.parametrize("distance",[3,5,7])
@pytest.mark.parametrize("basis",["X","Z"])
def test_dd_surface_code_distance_and_noiseless_parities(distance,basis):
    ideal = make_stim_circuit(distance,distance,basis,gate_basis="cz")
    # Uniform model plus explicit timing fixture avoids pretending d=5/7 fits Emerald.
    config = DDConfig(durations_ns={"1q":20,"2q":80,"measure":700,"reset":1000})
    plan = plan_dd(ideal,NoiseModel.uniform(),config)
    noiseless = insert_dd(ideal,plan)
    dets,obs = noiseless.compile_detector_sampler(seed=1).sample(32,separate_observables=True)
    assert not dets.any() and not obs.any()
    noisy = add_noise(ideal,NoiseModel.uniform(),dd=plan)
    assert len(noisy.shortest_graphlike_error()) == distance


def test_reused_dd_schedule_with_different_decoder_weights():
    snapshot = synthetic_snapshot()
    from iqm_surface_code.build_emerald_qubit_rotated import rank_placements
    ideal = make_stim_circuit(3,3,gate_basis="cz")
    mapping = rank_placements(ideal,calibration=snapshot)[0].stim_to_physical
    model = NoiseModel(calibration=snapshot,coherence=True,idle=0)
    result,physical,decoder = simulate_memory(3,3,model=model,decoder_model=NoiseModel.uniform(),
        shots=100,qubit_map=mapping,dd="xx")
    assert result["dd"]["pulse_count"] > 0
    assert result["circuit_info"]["dd_pulses"] == result["dd"]["pulse_count"]
    assert physical.without_noise() == decoder.without_noise()
    pytest.importorskip("qiskit")
    from iqm_surface_code.stim_to_qiskit import stim_to_qiskit
    with pytest.raises(ValueError,match="timed pulse"):
        stim_to_qiskit(physical)


def fake_iqm(monkeypatch):
    @dataclass
    class Options:
        dd_mode: str = "disabled"
        dd_strategy: object = None
        heralding_mode: str = "preserve"
        active_reset_cycles: int = 3
    class Strategy:
        def __init__(self,**values):
            self.__dict__.update(values)
    client = types.ModuleType("iqm.iqm_client")
    client.CircuitCompilationOptions = Options
    client.DDStrategy = Strategy
    client.DDMode = types.SimpleNamespace(ENABLED="enabled",DISABLED="disabled")
    parent = types.ModuleType("iqm")
    parent.iqm_client = client
    monkeypatch.setitem(sys.modules,"iqm",parent)
    monkeypatch.setitem(sys.modules,"iqm.iqm_client",client)
    return Options


def test_iqm_options_preserve_unrelated_fields_and_explicit_off(monkeypatch):
    Options = fake_iqm(monkeypatch)
    original = Options()
    options = iqm_compilation_options("xy4",options=original,target_names=["QB2","QB9"])
    assert options.dd_mode == "enabled" and original.dd_mode == "disabled"
    assert options.heralding_mode == "preserve" and options.active_reset_cycles == 3
    assert options.dd_strategy.gate_sequences == [(9,"XYXY","center")]
    assert options.dd_strategy.target_qubits == frozenset({"QB2","QB9"})
    assert iqm_compilation_options(None,options=options) is options
    disabled = iqm_compilation_options(False,options=options)
    assert disabled.dd_mode == "disabled" and disabled.dd_strategy is None


@pytest.mark.parametrize("sequence",["xx","xy4","xy8"])
def test_exact_echo_cancels_static_detuning(sequence):
    pytest.importorskip("qiskit_aer")
    from iqm_surface_code.aer_simulation import simulate_nonpauli
    model,ideal,mapping = fixture_model(),echo_circuit(),{0:"QB1",1:"QB2"}
    before = simulate_nonpauli(ideal,model,shots=1000,qubit_map=mapping,idle_detuning_rad_per_ns=.004,seed=1)
    after = simulate_nonpauli(ideal,model,shots=1000,qubit_map=mapping,idle_detuning_rad_per_ns=.004,dd=sequence,seed=2)
    assert before["obs_flips"].mean() > .9
    assert after["obs_flips"].mean() < .01


def test_exact_dd_does_not_cancel_markov_ramsey_decay():
    pytest.importorskip("qiskit_aer")
    from iqm_surface_code.aer_simulation import simulate_nonpauli
    model,ideal,mapping = fixture_model(t1=500,t2=1000),echo_circuit(),{0:"QB1",1:"QB2"}
    probabilities = []
    for dd in (None,"xx"):
        result = simulate_nonpauli(ideal,model,shots=4000,qubit_map=mapping,dd=dd,seed=3)
        probabilities.append(result["obs_flips"].mean())
    expected = .5*(1-np.exp(-.7))
    assert all(abs(p-expected) < .03 for p in probabilities)


def test_hardware_dd_options_reach_submission_without_gate_level_fake_dd(monkeypatch):
    pytest.importorskip("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2
    from iqm_surface_code import run_on_hardware as hardware
    Options = fake_iqm(monkeypatch)
    edges = [p for r in range(8) for c in range(8) for rr,cc in ((r+1,c),(r,c+1))
             if rr<8 and cc<8 for p in ((r*8+c,rr*8+cc),(rr*8+cc,r*8+c))]
    offline = GenericBackendV2(num_qubits=64,basis_gates=["rz","sx","x","cz"],coupling_map=edges)
    offline.index_to_qubit_name = lambda i:f"QB{i+1}"
    offline.qubit_name_to_index = lambda q:int(q[2:])-1
    prepared = hardware.prepare_qiskit(offline,rounds=3,dd="xy4")
    assert len(prepared["dd_target_qubits"]) == 9
    assert prepared["report"]["two_qubit_gate_count"] == 72
    assert prepared["report"]["dd"]["actual_pulse_count"] is None
    raw = prepared["stim_circuit"].compile_sampler(seed=2).sample(8)
    from collections import Counter
    counts = Counter("".join("1" if b else "0" for b in row[::-1]) for row in raw)
    captured = {}
    def submit(circuit,**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(result=lambda:types.SimpleNamespace(get_counts=lambda:counts),job_id=lambda:"offline-dd-test")
    backend = types.SimpleNamespace(run=submit)
    monkeypatch.setattr(hardware,"prepare_qiskit",lambda *args,**kwargs:prepared)
    result = hardware.run_qiskit(rounds=3,shots=8,backend=backend,dd="xy4",circuit_compilation_options=Options())
    options = captured["circuit_compilation_options"]
    assert options.dd_mode == "enabled" and options.heralding_mode == "preserve"
    assert options.dd_strategy.gate_sequences == [(9,"XYXY","center")]
    assert not result["det_events"].any()
    assert result["hardware_report"]["dd"]["decoder_model_mismatch"] is True
