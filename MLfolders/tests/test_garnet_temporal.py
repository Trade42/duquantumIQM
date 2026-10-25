"""The temporal features against known injected errors."""

from __future__ import annotations

import numpy as np

from dqiqm import surface as S
from dqiqm import temporal as T
from dqiqm.experiments import readout as RO


def _det_meta(c):
    coords = c.get_detector_coordinates()
    t = np.array([int(coords[i][2]) for i in range(c.num_detectors)])
    xy = [tuple(coords[i][:2]) for i in range(c.num_detectors)]
    labels = sorted(set(xy))
    return t, np.array([labels.index(p) for p in xy])


def test_pij_finds_a_planted_pair():
    rng = np.random.default_rng(1)
    n = 200_000
    x = rng.random((n, 4)) < 0.02
    pair = rng.random(n) < 0.03
    x[:, 1] ^= pair
    x[:, 3] ^= pair
    p = T.pij(x.astype(np.uint8), chunk=50_000)
    assert abs(p[1, 3] - 0.03) < 0.003
    assert abs(p[0, 2]) < 0.003 and abs(p[0, 1]) < 0.003


def test_record_flips_show_at_lag_two_and_state_flips_at_lag_one():
    """No reset: a misread bit fires D_k and D_(k+2); a flipped ancilla fires D_(k+1) and D_(k+2)."""
    clean = S.memory_circuit(3, 6, "Z")
    t, stab = _det_meta(clean)
    out = {}
    for name, nm in (("record", S.NoiseModel(record_flip_probability=0.03)),
                     ("state", S.NoiseModel(qubit_rates={a: {"idle_during_measurement": 0.03} for a in S.geometry(3).ancillas}))):
        noisy = S.memory_circuit(3, 6, "Z", noise=nm)
        raw = noisy.compile_sampler(seed=5).sample(30_000)
        dets, _ = clean.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
        out[name] = T.lag_summary(T.pij(dets.astype(np.uint8)), t, stab)
    assert out["record"]["same_stab_lag2"] > 0.02 > 0.005 > abs(out["record"]["same_stab_lag1"])
    assert out["state"]["same_stab_lag1"] > 0.02 > 0.005 > abs(out["state"]["same_stab_lag2"])


def test_readout_flips_separate_record_from_state():
    coords = [(0, q) for q in range(3)]
    clean = RO.circuit(coords, [1, 1, 1], 12)
    noisy = RO.circuit(coords, [1, 1, 1], 12, record_flip=0.03, state_flip=0.01)
    raw = noisy.compile_sampler(seed=9).sample(40_000)
    dets, _ = clean.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
    rows = T.readout_flips(dets.astype(np.uint8), raw.astype(np.uint8), [1, 1, 1], 3, 12)
    for r in rows:
        assert abs(r["record_flip"] - 0.03) < 0.004
        assert abs(r["state_flip"] - 0.01) < 0.004


def test_streaks_and_weights():
    # 2 stabilizers x 4 rounds; shot 0 fires stab 0 at t = 0, 1, 2 and stab 1 at t = 3
    t = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    stab = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dets = np.array([[1, 1, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    longest, rows = T.streaks(dets, t, stab)
    assert longest.tolist() == [3, 0]
    assert {(r["stab"], r["length"]): r["count"] for r in rows} == {(0, 3): 1, (1, 1): 1}
    w, rounds = T.syndrome_weight(dets, t)
    assert rounds == [0, 1, 2, 3] and w[0].tolist() == [1, 1, 1, 1]
