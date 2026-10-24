import itertools

import numpy as np
import pytest
import stim

from darawork.decoders.ensemble_decoder import EnsembleDecoder, MatchingDecoder
from darawork.decoders.tn_decoder import TNDecoder, mechanisms
from challenge.noise_models import NoiseModel, add_noise
from challenge.surface_code import make_stim_circuit

TINY = stim.DetectorErrorModel("""
error(0.1) D0 L0
error(0.2) D0 D1
error(0.05) D1 D2 L0
error(0.15) D2
error(0.08) D0 D2
error(0.12) L0
error(0.03) D1
""")


def brute_force(dem):
    mechs, table = mechanisms(dem), np.zeros((2 ** dem.num_detectors, 2))
    for e in itertools.product([0, 1], repeat=len(mechs)):
        w, s, L = 1.0, 0, 0
        for x, (p, dets, obs) in zip(e, mechs):
            w *= p if x else 1 - p
            if x:
                s ^= sum(1 << d for d in dets)
                L ^= len(obs) % 2
        table[s, L] += w
    return table


@pytest.mark.parametrize("chi", [None, 64])
def test_tn_matches_brute_force(chi):
    exact = brute_force(TINY)
    dets = np.array([[(i >> k) & 1 for k in range(3)] for i in range(8)], bool)
    P = TNDecoder(TINY, chi=chi, device="cpu").probabilities(dets)
    np.testing.assert_allclose(P[:, 1] / P.sum(1), exact[:, 1] / exact.sum(1), atol=1e-12)


def test_tn_not_worse_than_matching_d3():
    c = add_noise(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"), NoiseModel.uniform(0.01))
    dets, obs = c.compile_detector_sampler(seed=3).sample(3000, separate_observables=True)
    tn = (TNDecoder.from_circuit(c, device="cpu").decode(dets) != obs).any(1)
    pm = (MatchingDecoder(c).decode(dets) != obs).any(1)
    # Paired: ML may lose single shots, never many more than it wins.
    assert (tn & ~pm).sum() <= (pm & ~tn).sum() + 3 * np.sqrt((tn ^ pm).sum() + 1)


def test_ensemble_member_zero_is_baseline_and_votes_are_odd():
    c = add_noise(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"), NoiseModel.uniform(0.01))
    dets, _ = c.compile_detector_sampler(seed=4).sample(500, separate_observables=True)
    ens = EnsembleDecoder(c, members=5, sigma=0.3)
    assert (ens.members[0].decode(dets) == MatchingDecoder(c).decode(dets)).all()
    assert ens.decode(dets).shape == (500, 1)
    with pytest.raises(ValueError):
        EnsembleDecoder(c, members=4)


def test_nn_grid_layout_and_untrained_decode():
    pytest.importorskip("torch")
    from darawork.decoders.nn_decoder import NNDecoder, build_model, grid_layout, to_grid
    for basis in "ZX":
        ideal = make_stim_circuit(3, 3, basis, True, gate_basis="cz")
        layout = grid_layout(ideal)
        assert layout["shape"] == (4, 4, 4)
        x = to_grid(np.ones((2, ideal.num_detectors), bool), layout)
        assert x[:, :, :2].sum() == 2 * ideal.num_detectors  # Every detector lands on one cell.
        assert (x[:, :, :2] <= x[:, :, 2:4]).all()  # Events only where a detector exists.
    out = NNDecoder(build_model(8), ideal).decode(np.zeros((3, ideal.num_detectors), bool))
    assert out.shape == (3, 1) and out.dtype == bool


def test_lut_majority_and_fallback():
    from darawork.decoders.lut_decoder import LUTDecoder

    class Const:
        name = "const"

        def __init__(self):
            self.calls = 0

        def decode(self, dets):
            self.calls += len(dets)
            return np.ones((len(dets), 1), bool)

    train = np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], bool)
    obs = np.array([[1], [1], [0], [0], [1]], bool)  # [1,0]: 2 of 3 flipped; [0,1]: tie.
    fb = Const()
    lut = LUTDecoder(train, obs, fb, min_count=1)
    pred = lut.decode(np.array([[1, 0], [0, 1], [1, 1]], bool))
    assert pred[:, 0].tolist() == [True, True, True] and fb.calls == 2  # Tie and unseen go to fallback.
    assert lut.coverage == 1 / 3


def test_dem_fit_recovers_true_rates():
    from darawork.benchmark import noisy
    from darawork.decoders.dem_fit import fit_dem
    ideal, truth, _, _ = noisy("garnet", 3, 4, "Z")  # Calibrated, non-uniform truth.
    wrong = add_noise(ideal, NoiseModel.uniform(0.002))
    train, _ = truth.compile_detector_sampler(seed=11).sample(20000, separate_observables=True)
    dets, obs = truth.compile_detector_sampler(seed=12).sample(20000, separate_observables=True)
    err = {name: int((MatchingDecoder(src).decode(dets) != obs).any(1).sum())
           for name, src in (("wrong", wrong), ("fit", fit_dem(wrong, train)), ("true", truth))}
    noise = 3 * np.sqrt(err["true"])
    assert err["fit"] < err["wrong"] - noise and abs(err["fit"] - err["true"]) <= noise, err


def test_cached_lut_matches_model_and_counts_hits():
    from darawork.decoders.lut_decoder import CachedLUT
    c = add_noise(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"), NoiseModel.uniform(0.01))
    build, _ = c.compile_detector_sampler(seed=7).sample(4000, separate_observables=True)
    dets, _ = c.compile_detector_sampler(seed=8).sample(1000, separate_observables=True)
    tn = TNDecoder.from_circuit(c, device="cpu")
    cache = CachedLUT(build, tn, MatchingDecoder(c), capacity=64)
    assert len(cache.table) == 64 and cache.memory_bytes > 0
    pred = cache.decode(dets)
    hit = np.array([k in cache.table for k in __import__("darawork.decoders.lut_decoder", fromlist=["_keys"])._keys(dets)])
    assert np.isclose(cache.hit_rate, hit.mean())
    assert (pred[hit] == tn.decode(dets[hit])).all()  # Hits reproduce the model's decision exactly.
    online = CachedLUT(build, tn, MatchingDecoder(c), capacity=64, online=True)
    assert (online.decode(dets) == tn.decode(dets)).all()  # Online cache == the model itself.


def test_protocol_slices_are_disjoint_and_ordered():
    from darawork.protocol import TEST, VAL, slices
    tr, va, te = slices(20000)
    idx = [set(range(20000)[s]) for s in (tr, va, te)]
    assert len(idx[1]) == VAL and len(idx[2]) == TEST and len(idx[0]) == 20000 - VAL - TEST
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert max(idx[0]) < min(idx[1]) and max(idx[1]) < min(idx[2])


def test_lut_min_count_chosen_on_training_shots_only():
    from darawork.decoders.lut_decoder import LUTDecoder
    c = add_noise(make_stim_circuit(3, 2, "Z", True, gate_basis="cz"), NoiseModel.uniform(0.01))
    dets, obs = c.compile_detector_sampler(seed=9).sample(4000, separate_observables=True)
    lut = LUTDecoder(dets, obs, MatchingDecoder(c))
    assert set(lut.cv_errors) == {1, 3, 10, 30} and lut.min_count in lut.cv_errors
