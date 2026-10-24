from challenge.decoding import make_matcher
from darawork.garnet_dataset import features, garnet_placements, layout_info
from challenge.noise_models import NoiseModel, add_noise
from challenge.surface_code import make_stim_circuit


def run(model, rounds=4, basis="Z", shots=2000):
    ideal = make_stim_circuit(3, rounds, basis, True, gate_basis="cz")
    circuit = add_noise(ideal, model)
    raw = circuit.compile_sampler(seed=1).sample(shots)
    dets, obs = ideal.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    return features(raw, dets, obs, ideal, basis, rounds, make_matcher(circuit), make_matcher(circuit, correlated=False))[0]


def test_zero_swap_garnet_placement():
    ideal = make_stim_circuit(3, 2, "Z", True, gate_basis="cz")
    placements = garnet_placements(ideal)
    assert len(placements) >= 1 and all(len(set(p.values())) == 17 for p in placements)


def test_stabilizer_kinds():
    for basis in "XZ":
        stabs, _, _ = layout_info(make_stim_circuit(3, 3, basis, True, gate_basis="cz"), basis)
        kinds = [s["kind"] for s in stabs]
        assert kinds.count("X") == kinds.count("Z") == 4


def test_noiseless_features_are_zero():
    cols = run(NoiseModel(p1=0, p2=0, readout=0, reset=0, idle=0), shots=50)
    assert not any(cols[k].any() for k in cols if k.startswith(("det__", "lag", "space", "diag", "w__")))
    assert not cols["obs_flip"].any() and cols["pm_correct"].all()


def test_readout_errors_show_lag2_signature():
    # Readout error on m_k flips detectors k and k+2; only the final round's pairs
    # with the terminal detector at lag 1. Expect 7:1 at 8 rounds.
    cols = run(NoiseModel(p1=0, p2=0, readout=0.02, reset=0, idle=0), rounds=8)
    assert cols["lag2_total"].mean() > 4 * cols["lag1_total"].mean()


def test_memory_to_raw_reverses_qiskit_order():
    from darawork.garnet_dataset import memory_to_raw
    assert memory_to_raw(["001", "110"], 3).tolist() == [[True, False, False], [False, True, True]]


def test_shot_cap_refuses_before_connecting():
    import pytest
    from darawork.garnet_dataset import run_garnet
    with pytest.raises(ValueError, match="exceeds --max-shots"):
        run_garnet(rounds_list=[1, 2], bases=["Z"], shots=1000, max_shots=1000, submit=True)


def test_sqrt_y_form_keeps_detectors_and_halves_h():
    from darawork.garnet_dataset import sqrt_y_form
    for basis in "ZX":
        ideal = make_stim_circuit(3, 4, basis, True, gate_basis="cz")
        alt = sqrt_y_form(ideal)
        alt.detector_error_model()  # Raises if any detector or observable became non-deterministic.
        assert alt.num_measurements == ideal.num_measurements and alt.num_detectors == ideal.num_detectors
        noisy = add_noise(alt, NoiseModel.uniform(0.002))
        assert len(noisy.shortest_graphlike_error()) == 3
        dets, obs = alt.compile_detector_sampler(seed=0).sample(200, separate_observables=True)
        assert not dets.any() and not obs.any()
