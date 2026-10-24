import numpy as np
import pytest

torch = pytest.importorskip("torch")

from challenge.surface_code import make_stim_circuit
from challenge.noise_models import noise_suite, add_noise
from challenge.decoding import make_matcher
from iyanu.neural_gpu import NeuralConfig, train_neural_decoder, paired_comparison

SMALL = NeuralConfig(hidden=128, layers=3, steps=1500, batch=2048, lr=3e-3, seed=7, device="cpu")


@pytest.fixture(scope="module")
def setup():
    circuit = add_noise(make_stim_circuit(3, 2, "Z", no_reset=True, gate_basis="cz"), noise_suite(0.005)["uniform"])
    return circuit, train_neural_decoder(circuit, SMALL)


def test_shapes_and_order(setup):
    circuit, dec = setup
    dets, obs = circuit.compile_detector_sampler(seed=1).sample(50, separate_observables=True)
    out = dec.decode_batch(dets)
    assert out.shape == (50, circuit.num_observables) and out.dtype == np.bool_
    assert dec.decode_batch(dets[:1]).shape == (1, circuit.num_observables)
    assert np.array_equal(dec.decode_batch(dets[::-1]), out[::-1])


def test_noiseless_syndrome_predicts_no_flip(setup):
    circuit, dec = setup
    assert not dec.decode_batch(np.zeros((3, circuit.num_detectors), dtype=bool)).any()


def test_invalid_inputs(setup):
    circuit, dec = setup
    with pytest.raises(ValueError):
        dec.decode_batch(np.zeros(circuit.num_detectors, dtype=bool))
    with pytest.raises(ValueError):
        dec.decode_batch(np.zeros((2, circuit.num_detectors+1), dtype=bool))
    with pytest.raises(ValueError):
        dec.decode_batch(np.full((2, circuit.num_detectors), 2))
    with pytest.raises(ValueError):
        dec.decode_batch(np.zeros((2, circuit.num_detectors), dtype=bool), enable_correlations=True)


def test_seeded_training_reproduces(setup):
    circuit, dec = setup
    again = train_neural_decoder(circuit, SMALL)
    dets, _ = circuit.compile_detector_sampler(seed=3).sample(2000, separate_observables=True)
    assert np.array_equal(dec.decode_batch(dets), again.decode_batch(dets))


def test_learns_better_than_no_correction_and_near_matching(setup):
    circuit, dec = setup
    dets, obs = circuit.compile_detector_sampler(seed=11).sample(20000, separate_observables=True)
    matcher = make_matcher(circuit)
    cmp = paired_comparison(dec.decode_batch(dets), matcher.decode_batch(dets, enable_correlations=True), obs)
    raw = int(np.any(obs, axis=1).sum())
    assert cmp["a_errors"] < raw/2
    assert cmp["a_errors"] < 2*cmp["b_errors"]+50
