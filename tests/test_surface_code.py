import importlib
import json
import numpy as np
import pytest
import stim

from challenge.surface_code import make_stim_circuit, get_circuit_info, remove_ancilla_resets
from challenge.noise_models import NoiseModel, noise_suite, add_noise
from challenge.calibration import synthetic_snapshot, CalibrationSnapshot
from challenge.build_emerald_qubit_rotated import rank_placements, validate_layout, build_emerald_qubit_map
from challenge.decoding import fit_ler, decode_syndromes, wilson_interval
from challenge.internal_helpers import counts_to_measurement_array, get_meas_order
from challenge.extract_syndromes import extract_syndromes
from challenge.leakage import sample_leakage
from challenge.experiment import simulate_memory, confidence_sequence


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("rounds", [1, 2, 5])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("no_reset", [False, True])
@pytest.mark.parametrize("gate_basis", ["cx", "cz"])
def test_ideal_determinism(distance, rounds, basis, no_reset, gate_basis):
    circuit = make_stim_circuit(distance, rounds, basis, no_reset, gate_basis=gate_basis)
    circuit.detector_error_model()  # Reject non-deterministic/gauge parities.
    dets, obs = circuit.compile_detector_sampler(seed=21).sample(64, separate_observables=True)
    assert not dets.any() and not obs.any()
    info = get_circuit_info(circuit)
    assert info["num_qubits"] == 2*distance**2-1
    assert circuit.num_detectors == rounds*(distance**2-1)
    assert circuit.num_measurements == rounds*(distance**2-1)+distance**2
    assert len(get_meas_order(circuit)) == circuit.num_measurements
    assert info["reset_measurements"] == (0 if no_reset else rounds*(distance**2-1))


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("no_reset", [False, True])
@pytest.mark.parametrize("gate_basis", ["cx", "cz"])
def test_circuit_distance(distance, basis, no_reset, gate_basis):
    circuit = make_stim_circuit(distance, distance, basis, no_reset,
                                noise=NoiseModel.uniform(), gate_basis=gate_basis)
    assert len(circuit.shortest_graphlike_error()) == distance


def test_no_reset_record_equations_and_readout_is_classical():
    circ = make_stim_circuit(3, 3)
    # Bulk round 3 detectors are m3 XOR m1, not m3 XOR m2.
    records, bulk = 0, []
    for inst in circ:
        if inst.name in {"M", "MX"}:
            records += len(inst.targets_copy())
        if inst.name == "DETECTOR" and records == 24:
            bulk.append([t.value for t in inst.targets_copy()])
    assert len(bulk) == 8
    assert all(len(v) == 2 and abs(v[0]-v[1]) == 16 for v in bulk)
    noisy = add_noise(circ, NoiseModel.uniform(0).with_changes(readout=.1))
    assert any(i.name == "M" and i.gate_args_copy() == [.1] for i in noisy)
    assert not any(i.name == "X_ERROR" for i in noisy)
    with pytest.raises(ValueError):
        remove_ancilla_resets(noisy)


def test_four_disjoint_interaction_layers():
    circuit = make_stim_circuit(5, 1)
    layers = []
    for inst in circuit:
        if inst.name == "CX":
            qs = [t.value for t in inst.targets_copy()]
            assert len(qs) == len(set(qs))
            layers.append(qs)
    assert len(layers) == 4


@pytest.mark.parametrize("distance,rounds", [(2, 3), (3, 0), (3.5, 2), (True, 2)])
def test_invalid_arguments(distance, rounds):
    with pytest.raises(ValueError):
        make_stim_circuit(distance, rounds)


def test_calibration_roundtrip_and_all_noise_models(tmp_path):
    snapshot = synthetic_snapshot()
    snapshot.save(tmp_path/"cal.json")
    assert CalibrationSnapshot.load(tmp_path/"cal.json") == snapshot
    ideal = make_stim_circuit(3, 3, gate_basis="cz")
    placement = rank_placements(ideal, calibration=snapshot)[0]
    for model in noise_suite(calibration=snapshot).values():
        noisy = add_noise(ideal, model, qubit_map=placement.stim_to_physical)
        assert noisy.without_noise() == ideal
        noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    with pytest.raises(ValueError, match="complete"):
        add_noise(ideal, noise_suite(calibration=snapshot)["calibrated_pauli"])


def test_iqm_normalization():
    metric = {"SQG":{"QB1":.999,"QB2":.998}, "READOUT":{"QB1":.98,"QB2":.97},
              "T1":{"QB1":40.,"QB2":50.}, "T2":{"QB1":30.,"QB2":40.},
              "CZ":{("QB1","QB2"):.995,("QB2","QB1"):.994}}
    snapshot = CalibrationSnapshot.from_iqm_metrics(metric, coherence_unit="us",
        durations_ns={"1q":20,"2q":80,"measure":700,"reset":1000}, reset_error=.002)
    assert snapshot.qubits["QB1"].t1_ns == 40000
    assert snapshot.edge_error("QB1","QB2") == pytest.approx(.006)
    model = NoiseModel(calibration=snapshot)
    assert model.qubit_rates(0,{0:"QB1"})[0] == pytest.approx(.0015)
    assert model.edge_rate(0,1,{0:"QB1",1:"QB2"}) == pytest.approx(.0075)


def test_layout_native_and_calibration_optimum():
    ideal = make_stim_circuit(3, 3, gate_basis="cz")
    snapshot = synthetic_snapshot()
    candidates = rank_placements(ideal, calibration=snapshot)
    assert len(candidates) == 128
    assert candidates[0].score <= candidates[-1].score
    for p in candidates:
        assert validate_layout(ideal,p.stim_to_physical,[e.split("|") for e in snapshot.edges])["routing_swaps"] == 0
    assert len(build_emerald_qubit_map(ideal)) == 17
    with pytest.raises(ValueError, match="No native"):
        rank_placements(ideal, couplings=[])
    with pytest.raises(ValueError, match="No native"):
        rank_placements(make_stim_circuit(5, 3))


def test_persistent_leakage_survives_measurement():
    ideal = stim.Circuit("QUBIT_COORDS(1,1) 0\nQUBIT_COORDS(2,2) 1\nR 0 1\nCX 0 1\nTICK\nM 0 1\nTICK\nM 0 1\nTICK")
    model = NoiseModel.uniform(0).with_changes(leakage=1, seepage=0)
    raw, diag = sample_leakage(ideal,model,1000,seed=22)
    assert diag["leaked_measurements"] == 4000
    assert diag["final_leaked_fraction"] == 1
    assert .4 < raw.mean() < .6
    reset = ideal + stim.Circuit("R 0 1\nTICK")
    assert sample_leakage(reset,model,20,seed=1)[1]["final_leaked_fraction"] == 0
    assert sample_leakage(reset,model.with_changes(reset_clears_leakage=False),20,seed=1)[1]["final_leaked_fraction"] == 1


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_leakage_sampler_pauli_limit_matches_stim(basis):
    model = noise_suite(.003)["correlated_zz"]
    circuit = make_stim_circuit(3,3,basis,noise=model,gate_basis="cz")
    raw, diag = sample_leakage(circuit,model,20000,seed=4)
    actual = extract_syndromes(raw,circuit,False)["det_events"].mean(axis=0)
    expected = circuit.compile_detector_sampler(seed=5).sample(20000).mean(axis=0)
    sigma = np.sqrt(np.maximum(.001,expected*(1-expected))*2/20000)
    assert np.all(np.abs(actual-expected) < 6*sigma)
    assert diag["leakage_entries"] == 0


def test_simulate_decode_and_mismatched_circuit_rejected():
    result, circuit, _ = simulate_memory(3,3,shots=2000,seed=8,model=NoiseModel.uniform(.003))
    assert 0 < result["errors"] < 1000
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(50,separate_observables=True)
    syndromes = dict(det_events=dets,obs_flips=obs,stim_circuit=circuit)
    assert decode_syndromes(syndromes,circuit).shots == 50
    other = make_stim_circuit(3,3,no_reset=False,noise=NoiseModel.uniform())
    with pytest.raises(ValueError,match="differs"):
        decode_syndromes(syndromes,other)


def test_statistics_and_zero_failure_limits():
    rounds = np.array([1,3,5,9])
    eps = .012
    n = np.full(4,1000000)
    k = np.rint(n*.5*(1-(1-2*eps)**rounds))
    fit = fit_ler(rounds,k,n)
    assert abs(fit["epsilon_per_round"]-eps) < .00001
    zero = fit_ler(rounds,[0]*4,[1000]*4)
    assert zero["epsilon_per_round"] == 0
    assert zero["ci95_profile"][1] > 0
    assert wilson_interval(0,100)[1] > 0
    assert confidence_sequence(0,100)[1] > wilson_interval(0,100)[1]
    assert confidence_sequence(100,100)[0] < 1
    with pytest.raises(ValueError):
        fit_ler([1],[0],[100])


def test_counts_order_spaces_and_invalid_totals():
    assert counts_to_measurement_array({"10 01":2},4,2).tolist() == [[True,False,False,True]]*2
    with pytest.raises(ValueError):
        counts_to_measurement_array({"01":1},2,2)
    with pytest.raises(ValueError):
        counts_to_measurement_array({"012":1},3,1)


def test_exchange_identity():
    from challenge.exchange import verify_exchange_identity
    assert verify_exchange_identity()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_qiskit_translation_end_to_end(basis):
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator
    from challenge.stim_to_qiskit import stim_to_qiskit
    circuit = make_stim_circuit(3,3,basis,gate_basis="cz")
    qc, dense, order = stim_to_qiskit(circuit)
    assert len(dense) == 17 and order == get_meas_order(circuit)
    assert not qc.count_ops().get("reset",0)
    counts = AerSimulator(method="stabilizer",max_parallel_threads=1).run(qc,shots=128,seed_simulator=7).result().get_counts()
    raw = counts_to_measurement_array(counts,circuit.num_measurements,128)
    syndromes = extract_syndromes(raw,circuit,False)
    assert not syndromes["det_events"].any() and not syndromes["obs_flips"].any()


def test_imports_have_no_hardware_side_effects():
    importlib.import_module("challenge.run_on_hardware")
    importlib.import_module("challenge.iqm_utils.get_calibration_data")


def test_legacy_round_data_noise_occurs_each_round():
    model = {"before_round_data_depolarization":.01}
    for gate_basis in ("cx", "cz"):
        circuit = make_stim_circuit(3,5,noise=model,gate_basis=gate_basis)
        count = sum(len(i.targets_copy()) for i in circuit if i.name == "PAULI_CHANNEL_1")
        assert count == 9*5


def test_zero_noise_simulation_and_stopping():
    result, _, _ = simulate_memory(3,3,model=NoiseModel.uniform(0),shots=30)
    assert result["errors"] == 0 and result["graphlike_distance"] is None
    assert result["ci95"][1] > 0
    result, _, _ = simulate_memory(3,3,model=NoiseModel.uniform(.02),shots=2000,
                                    batch_size=100,target_errors=1,seed=9)
    assert result["shots"] < 2000 and result["errors"] >= 1
    assert "anytime" in result["interval_method"]


def test_spam_fit_recovers_synthetic_parameters():
    rounds = np.array([1,3,5,9,15])
    n = np.full(len(rounds),1000000)
    k = np.rint(n*.5*(1-.96*(1-2*.015)**rounds))
    fit = fit_ler(rounds,k,n,fit_spam=True)
    assert fit["epsilon_per_round"] == pytest.approx(.015,abs=1e-5)
    assert fit["amplitude"] == pytest.approx(.96,abs=1e-5)


def test_hardware_preparation_on_offline_backend():
    pytest.importorskip("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2
    from challenge.run_on_hardware import prepare_qiskit
    edges = [p for r in range(8) for c in range(8) for rr,cc in ((r+1,c),(r,c+1))
             if rr < 8 and cc < 8 for p in ((r*8+c,rr*8+cc),(rr*8+cc,r*8+c))]
    backend = GenericBackendV2(num_qubits=64,basis_gates=["rz","sx","x","cz"],coupling_map=edges)
    backend.index_to_qubit_name = lambda i: f"QB{i+1}"
    backend.qubit_name_to_index = lambda q: int(q[2:])-1
    prepared = prepare_qiskit(backend,rounds=3,basis="X")
    assert prepared["report"]["routing_swaps"] == 0
    assert prepared["report"]["two_qubit_gate_count"] == 24*3
    assert prepared["circuit"].count_ops()["measure"] == 8*3+9
