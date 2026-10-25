import numpy as np
import pytest
import stim

from challenge.decoders.history import HistoryDecoder, encode, layout, load, network, save


def circuit(rounds=3):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=rounds,
                                  after_clifford_depolarization=.005)


def test_coordinates_and_masks_cover_long_history_without_zero_padding_as_data():
    c = circuit(7)
    cells = layout(c)
    d = np.zeros((2, c.num_detectors), bool)
    d[1, -1] = True
    x = encode(d, cells, np.array([[0], [1]], bool))
    assert x.shape == (2, 8, 20)
    assert x[0, :, 8:16].sum() == c.num_detectors
    assert x[0, 0, 8:16].sum() == 4 and x[0, -1, 8:16].sum() == 4
    t, site = cells[-1]
    assert x[1, t, site] == 1 and x[0, t, site] == 0
    assert np.all(x[1, :, 19] == 1)


def test_recurrent_state_roundtrip_preserves_order_shape_and_correlated_prior(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    torch.manual_seed(73)
    c = circuit()
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    d, _ = c.compile_detector_sampler(seed=19).sample(32, separate_observables=True)
    module = network(8)
    dec = HistoryDecoder(module, c, dem)
    pred = dec.decode_batch(d)
    save(module, dict(hidden=8), tmp_path/"state")
    frozen, metadata = load(tmp_path/"state")
    restored = HistoryDecoder(frozen, c, dem, metadata=metadata)
    order = np.random.default_rng(4).permutation(len(d))
    assert np.array_equal(restored.decode_batch(d[order]), pred[order])
    assert restored.decode_batch(d[:0]).shape == (0, 1)
    assert restored.decode_batch(d[:1]).shape == (1, 1)
    with pytest.raises(ValueError, match="correlated MWPM"):
        restored.decode_batch(d, enable_correlations=False)
    with pytest.raises(ValueError):
        restored.decode_batch(np.full(d.shape, 2))


def test_unrecognized_layout_fails_loudly():
    with pytest.raises(ValueError, match="distance-three"):
        layout(stim.Circuit("M 0\nDETECTOR(99,99,0) rec[-1]"))


def test_training_accepts_variable_history_lengths_and_separate_development():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from challenge.decoders.history import train
    records, validation = [], []
    for r in (1, 7):
        c = circuit(r)
        dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        d, o = c.compile_detector_sampler(seed=73+r).sample(192, separate_observables=True)
        records.append((c, dem, d[:128], o[:128]))
        validation.append((c, dem, d[128:], o[128:]))
    module, metadata = train(records, validation, steps=100, hidden=8)
    assert metadata["train_shots"] == 256 and metadata["tuning_shots"] == 128
    assert not metadata["hidden_leakage_inputs"]
    assert HistoryDecoder(module, c, dem).decode_batch(d).shape == (192, 1)
