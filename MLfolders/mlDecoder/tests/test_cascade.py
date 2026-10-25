import numpy as np
import pytest

from mlDecoder import data
from mlDecoder.data import DatasetSpec

torch = pytest.importorskip("torch")
from mlDecoder import cascade  # noqa: E402
from mlDecoder.evaluate import Matcher  # noqa: E402
from mlDecoder.models import rnn_lstm  # noqa: E402
from mlDecoder.relay import RelayBP, dem_matrices  # noqa: E402
from mlDecoder.sequence import Component, Mixture  # noqa: E402

SPEC = DatasetSpec(distance=3, rounds=3, basis="Z", noise="uniform", p=0.003)
PRIOR = dict(noise="uniform", p=0.003, calibration=None)


def test_dem_matrices_and_relay_bp_against_mwpm():
    pytest.importorskip("relay_bp")
    _, noisy = data.make_circuits(SPEC)
    H, L, priors = dem_matrices(noisy)
    assert H.shape[0] == noisy.num_detectors and L.shape[0] == 1 and H.shape[1] == L.shape[1] == len(priors)
    assert (np.asarray(H.sum(axis=0)) > 0).all() and ((priors > 0) & (priors < 0.5)).all()
    ds = data.make_dataset(SPEC, "val", 3000, seed=2)
    relay = RelayBP(noisy)
    out = relay.decode_detailed(ds.dets)
    assert out["predictions"].shape == (3000, 1) and out["converged"].dtype == bool
    assert not out["predictions"][~ds.dets.any(axis=1)].any()            # Empty syndromes: no flip.
    fail = lambda p: np.any(p != ds.obs.astype(bool), axis=1).mean()
    mwpm = fail(Matcher(noisy, True).decode_batch(ds.dets))
    assert abs(fail(relay.decode_batch(ds.dets)) - mwpm) < 0.01          # Relay-BP ~ matching on the surface code.
    with pytest.raises(ValueError):
        relay.decode_batch(ds.dets[:, 1:])


def _layer1():
    _, noisy = data.make_circuits(SPEC)
    decoder = rnn_lstm.build(noisy, dict(model=dict(hidden=16, head=16),
                                         train=dict(batch_size=256, max_train_shots=256 * 20, eval_every_steps=10,
                                                    eval_every_seconds=1e9, anneal_by_time=False, cpu_threads=2)))
    mix = Mixture(3, "Z", (Component("uniform", 0.003, None, 1.0),), (2, 3), curriculum_fraction=0.0)
    decoder.fit(mix, [data.make_dataset(SPEC, "val", 500, seed=1)], device="cpu", seed=0, log=lambda *_: None)
    return decoder


def test_cascade_layers_tuning_and_curve():
    pytest.importorskip("relay_bp")
    layer1 = _layer1()
    val = [data.make_dataset(SPEC, "val", 1500, seed=3), data.make_dataset(SPEC.__class__(**{**SPEC.__dict__, "rounds": 2}),
                                                                            "val", 1500, seed=3)]
    test = data.make_dataset(SPEC, "test", 1500)
    c0 = cascade.Cascade(layer1, PRIOR, 0.0)
    assert np.array_equal(c0.decode_batch(test.dets), layer1.decode_batch(test.dets))   # tau 0: layer 1 alone.
    c1 = cascade.Cascade(layer1, PRIOR, 1.0 + 1e-9)
    assert np.array_equal(c1.decode_batch(test.dets), c1.relay(3).decode_batch(test.dets))  # tau > 1: layer 2 alone.
    p = layer1.probabilities(test.dets)
    assert ((p >= 0) & (p <= 1)).all() and (p[~test.dets.any(axis=1)] == 0).all()
    assert np.array_equal(p > 0.5, layer1.decode_batch(test.dets)[:, 0])
    parts = cascade.components(layer1, c0.relay, val)
    tuned = cascade.tune(parts)
    rows = cascade.curve(parts)
    fewest = min(r["failures"] for r in rows)
    assert tuned["argmin"]["validation"]["failures"] == fewest and tuned["tau"] <= tuned["argmin"]["tau"]
    assert tuned["validation"]["failures"] - fewest <= tuned["paired_vs_argmin"]["se"]
    assert rows[0]["deferred_fraction"] == 0 and rows[-1]["deferred_fraction"] == 1
    c = cascade.Cascade(layer1, PRIOR, tuned["tau"])
    for part, ds in zip(cascade.components(layer1, c.relay, val), val):
        f, n, d, _ = cascade.failures_at([part], c.tau)
        detail = c.decode_detailed(ds.dets)
        assert f == int(np.any(detail["predictions"] != ds.obs.astype(bool), axis=1).sum())  # The curve is the decoder.
        assert d == int((detail["layer"] >= 1).sum()) and n == ds.shots
    cal = cascade.calibration(parts)
    assert sum(b["n"] for b in cal["bins"]) == cal["shots"] and 0 <= cal["ece"] <= 1


def _parts(doubtful: int, layer1_only: int, relay_only: int, confident: int = 300) -> list[dict]:
    """Known answers: ``doubtful`` shots at confidence 0.02 (layer 1 says flip), ``confident`` ones layer 1 gets right.
    Layer 1 alone fails ``layer1_only`` doubtful shots, Relay-BP alone ``relay_only`` others; none fail both."""
    p = np.r_[np.full(doubtful, 0.51), np.full(confident, 0.01)]
    obs = np.r_[np.ones(doubtful, bool), np.zeros(confident, bool)]
    obs[:layer1_only] = False
    relay = obs.copy()
    relay[layer1_only:layer1_only + relay_only] ^= True
    n = len(p)
    return [dict(rounds=3, p=p, relay=relay[:, None], converged=np.ones(n, bool), mwpm=np.zeros((n, 1), bool),
                 obs=obs[:, None], relay_us_per_shot=0.0, active=np.ones(n, bool))]


def test_tune_one_standard_error_rule():
    noise = cascade.tune(_parts(100, 50, 48))        # Deferring saves 2 failures; 1 SE = sqrt(50 + 48).
    assert noise["argmin"]["tau"] > 0.02 and noise["argmin"]["validation"]["failures"] == 48
    assert noise["tau"] == 0.0 and noise["validation"]["failures"] == 50
    assert (noise["paired_vs_argmin"]["only_tau_fails"], noise["paired_vs_argmin"]["only_argmin_fails"]) == (50, 48)
    assert noise["paired_vs_argmin"]["se"] == pytest.approx(98 ** 0.5)
    real = cascade.tune(_parts(100, 50, 10))         # Saves 40; 1 SE = sqrt(60): the deferral stays.
    assert real["tau"] == real["argmin"]["tau"] > 0.02 and real["validation"]["failures"] == 10
    assert real["paired_vs_argmin"]["se"] == 0.0


def test_choose_prior_one_standard_error():
    cal = np.zeros(1000, bool)
    uni = np.zeros(1000, bool)
    cal[:60], uni[60:110] = True, True                # Uniform fails 50, calibrated 60: 10 < sqrt(110).
    keep = cascade.choose_prior(cal, uni)
    assert keep["chosen"] == "calibrated" and (keep["only_uniform_fails"], keep["only_calibrated_fails"]) == (50, 60)
    assert keep["se"] == pytest.approx(110 ** 0.5)
    cal[110:200] = True                               # Calibrated now fails 150: 100 > sqrt(200).
    switch = cascade.choose_prior(cal, uni)
    assert switch["chosen"] == "uniform" and switch["calibrated_failures"] == 150 and switch["uniform_failures"] == 50
    assert cascade.choose_prior(uni, uni)["chosen"] == "calibrated"   # A tie keeps the incumbent.


def test_uncertain_shots_and_nonconvergence_preserve_order_without_dropping_shots():
    class Learned:
        distance, basis, device = 3, "Z", "cpu"
        def probabilities(self, dets, raw_meas=None):
            return np.array([.9, .51, .49, .1])

    class Fallback:
        def decode_batch(self, dets):
            assert dets.shape == (1, 8) and dets[0, 2] == 1
            return np.array([[False]])

    class Relay:
        fallback = Fallback()
        def decode_detailed(self, dets):
            assert dets.shape == (2, 8)
            return dict(predictions=np.array([[True], [True]]), converged=np.array([True, False]))

    dets = np.eye(8, dtype=np.uint8)[:4]
    decoder = cascade.Cascade(Learned(), PRIOR, .2)
    decoder._relays[1] = Relay()
    out = decoder.decode_detailed(dets)
    assert out["predictions"].ravel().tolist() == [True, True, False, False]
    assert out["layer"].tolist() == [0, 1, 2, 0]
    with pytest.raises(ValueError, match="binary"):
        decoder.decode_batch(dets.astype(float) + .5)
