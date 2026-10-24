import numpy as np
import pytest

torch = pytest.importorskip("torch")

from challenge.surface_code import make_stim_circuit
from challenge.noise_models import noise_suite, add_noise
from challenge.decoding import make_matcher
from iyanu.conv_decoders import ConvConfig, ConvDecoder, ConvPredecoder, SpaceTimeLayout, dem_edges

TINY = ConvConfig(channels=8, layers=2, head=32, steps=60, batch=256, seed=3, device="cpu")


@pytest.fixture(scope="module")
def circuit():
    return add_noise(make_stim_circuit(3, 2, "Z", no_reset=True, gate_basis="cz"), noise_suite(0.005)["uniform"])


@pytest.fixture(scope="module")
def shots(circuit):
    return circuit.compile_detector_sampler(seed=5).sample(3000, separate_observables=True)


@pytest.fixture(scope="module")
def predecoder(circuit):
    return ConvPredecoder(circuit, TINY).train()


def test_layout_places_each_detector_once(circuit):
    lay = SpaceTimeLayout(circuit)
    assert lay.shape == (3, 4, 4) and lay.mask.sum() == circuit.num_detectors
    x = torch.eye(circuit.num_detectors)
    grid = lay.to_lattice(x, torch)
    assert grid.shape == (circuit.num_detectors, 2, 3, 4, 4)
    assert torch.all(grid[:, 0].flatten(1).sum(1) == 1)


def test_dem_edges_reproduce_every_error(circuit):
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    edges, error_to_edges, _ = dem_edges(dem)
    errors = [i for i in dem.flattened() if i.type == "error"]
    for inst, flipped in zip(errors, error_to_edges):
        dets = set()
        for e in flipped:
            dets ^= set(edges[e]["detectors"])
        assert dets == {t.val for t in inst.targets_copy() if t.is_relative_detector_id()}
    assert all(1 <= len(e["detectors"]) <= 2 and 0 < e["prior"] < 1 for e in edges)


def test_conv_full_shapes_and_validation(circuit, shots):
    dec = ConvDecoder(circuit, TINY).train(circuit)
    dets, _ = shots
    out = dec.decode_batch(dets[:40])
    assert out.shape == (40, circuit.num_observables) and out.dtype == np.bool_
    assert np.array_equal(dec.decode_batch(dets[:40][::-1]), out[::-1])
    with pytest.raises(ValueError):
        dec.decode_batch(dets[0])
    with pytest.raises(ValueError):
        dec.decode_batch(dets[:2], enable_correlations=True)


def test_predecoder_never_applying_equals_pymatching(circuit, shots):
    import dataclasses
    dets, _ = shots
    pre = ConvPredecoder(circuit, dataclasses.replace(TINY, threshold=1.01))
    pre.train = None  # untrained weights are fine: nothing passes the threshold
    m = make_matcher(circuit, correlated=True)
    pred, stats = pre.decode_batch(dets, return_stats=True)
    assert stats["edges_applied"] == 0
    assert np.array_equal(pred, m.decode_batch(dets, enable_correlations=True).astype(bool))


def test_predecoder_applies_disjoint_lit_edges_only(predecoder, shots):
    dets, _ = shots
    x = torch.from_numpy(dets.astype(np.float32))
    applied = predecoder.select_edges(x).numpy()
    inc = predecoder.incidence.toarray()
    per_detector = applied.astype(int) @ inc
    assert per_detector.max() <= 1
    assert not (per_detector & ~dets).any()  # only lit detectors are cleared


def test_predecoder_is_never_worse_than_no_correction(predecoder, shots):
    dets, obs = shots
    pred, stats = predecoder.decode_batch(dets, return_stats=True)
    assert stats["events_left"] <= stats["events_in"]
    assert np.any(pred != obs, axis=1).sum() < np.any(obs, axis=1).sum()
    assert not predecoder.decode_batch(np.zeros((2, dets.shape[1]), dtype=bool)).any()
