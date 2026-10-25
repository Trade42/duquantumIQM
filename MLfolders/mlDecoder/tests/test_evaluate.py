import numpy as np
import pytest

from mlDecoder import data, evaluate
from mlDecoder.data import DatasetSpec

SMALL = DatasetSpec(distance=3, rounds=2, basis="Z", noise="uniform", p=0.003)


def test_paired_counts_and_interval():
    rng = np.random.default_rng(1)
    a = rng.random(20000) < 0.02
    b = a.copy()
    b[np.flatnonzero(a)[:100]] = False      # B right where A is wrong: 100 shots.
    b[np.flatnonzero(~a)[:30]] = True       # B wrong where A is right: 30 shots.
    out = evaluate.paired(a, b)
    assert (out["a_wrong_b_right"], out["b_wrong_a_right"]) == (100, 30)
    assert out["delta_p_l"] == pytest.approx(70 / 20000)
    lo, hi = out["ci95_paired_bootstrap"]
    assert 0 < lo < out["delta_p_l"] < hi and out["verdict"] == "B better"
    same = evaluate.paired(a, a)
    assert same["delta_p_l"] == 0 and same["verdict"].startswith("tie")


def test_fits_both_forms():
    rounds, eps = np.array([1, 2, 3, 5, 8, 10]), 0.004
    shots = np.full(6, 2_000_000)
    failures = np.rint(shots * .5 * (1 - (1 - 2*eps)**rounds))
    out = evaluate.fits(rounds, failures, shots)
    assert out["no_spam"]["epsilon"] == pytest.approx(eps, rel=1e-3)
    assert out["spam"]["epsilon"] == pytest.approx(eps, rel=2e-2) and 0 < out["spam"]["amplitude"] <= 1


def test_reference_is_cached_and_order_preserving(isolated_results):
    ds = data.frozen_test_set(SMALL, 3000)
    first = evaluate.reference(ds, True)
    assert (isolated_results / "datasets" / ds.manifest["dataset_id"] / "ref_mwpm_corr.npz").exists()
    again = evaluate.reference(ds, True)
    assert first["failures"] == again["failures"] and np.array_equal(first["predictions"], again["predictions"])
    direct = evaluate.Matcher(ds.circuit, True).decode_batch(ds.dets)
    assert np.array_equal(direct, first["predictions"])
    with pytest.raises(ValueError):
        evaluate.Matcher(ds.circuit, True).decode_batch(ds.dets[:, 1:])

    class FirstBit:  # A fake decoder: predicts the first detection event, so order is checkable.
        def decode_batch(self, dets):
            return dets[:, :1].astype(bool)
    pred, _ = evaluate.decode_dataset(FirstBit(), ds, chunk=700)
    assert np.array_equal(pred[:, 0], ds.dets[:, 0].astype(bool))


def test_benchmark_rows():
    dets = np.zeros((100, 8), dtype=np.uint8)
    rows = evaluate.benchmark(lambda x: x[:, :1], dets, batch_sizes=(1, 64, 4096), min_seconds=.01, min_calls_b1=5)
    assert [r["batch"] for r in rows] == [1, 64, 4096]
    assert rows[0]["shots_per_second"] > 0 and "skipped" in rows[2]


def test_eps_mle_matches_fit_ler():
    from challenge.decoding import fit_ler
    rounds, shots = [1, 2, 3, 5, 8, 10], np.full(6, 100_000)
    rng = np.random.default_rng(4)
    for eps in (0.0005, 0.004, 0.03):
        k = rng.binomial(shots, .5 * (1 - (1 - 2*eps)**np.array(rounds)), size=(5, 6))
        fast = evaluate.eps_mle(rounds, k, shots)
        slow = [fit_ler(rounds, row, shots)["epsilon_per_round"] for row in k]
        assert np.allclose(fast, slow, rtol=1e-6, atol=1e-12)
    assert evaluate.eps_mle(rounds, np.zeros((1, 6)), shots)[0] == 0


def _fake_run():
    """A run-shaped result: two distances, five round counts, MWPM slightly worse than correlated."""
    rng, points = np.random.default_rng(0), []
    for d, eps in ((3, 0.004), (5, 0.001)):
        for r in (1, 2, 3, 5, 8):
            n = 200_000
            p = .5 * (1 - (1 - 2*eps)**r)
            corr = int(rng.binomial(n, p))
            only_u, only_c = int(rng.binomial(n, p * .2)), int(rng.binomial(n, p * .1))
            u = corr - only_c + only_u
            ref = lambda k: dict(point=dict(failures=k, shots=n, p_l=k/n, ci95_wilson=[0, 1]))
            points.append(dict(spec=dict(p=0.001, noise="uniform", distance=d, basis="Z", rounds=r),
                               references=dict(mwpm_corr=ref(corr), mwpm=ref(u)),
                               paired_mwpm_vs_mwpm_corr=dict(a_wrong_b_right=only_u, b_wrong_a_right=only_c)))
    return dict(points=points)


def test_ler_record_and_files(tmp_path):
    result = _fake_run()
    record = evaluate.write_ler(tmp_path, result, spam_bootstrap=40)
    names = {p.name for p in (tmp_path / "ler").iterdir()}
    assert {"ler_points.csv", "ler_fits.json", "ler_3_Z_uniform_p0.001.png", "ler_5_Z_uniform_p0.001.png"} <= names
    header = (tmp_path / "ler" / "ler_points.csv").read_text().splitlines()[0]
    assert header == ",".join(evaluate.LER_COLUMNS)
    fit = next(f for f in record["fits"] if f["decoder"] == "mwpm" and f["d"] == 3)
    assert fit["enough_rounds"] and fit["no_spam"]["goodness_of_fit"]["dof"] == 4
    ratio = fit["ratio_to_mwpm_corr"]
    lo, hi = ratio["no_spam_ci95_paired_bootstrap"]
    assert lo < ratio["no_spam"] < hi and ratio["no_spam"] > 1
    slo, shi = ratio["spam_ci95_paired_bootstrap"]
    assert slo < shi and ratio["spam_bootstrap_samples"] == 40
    lam = [l for l in record["lambdas"] if l["decoder"] == "mwpm_corr"]
    assert len(lam) == 1 and 3 < lam[0]["lambda_no_spam"] < 5


def test_sensitivity_fits_recover_a_round_offset():
    """Data whose curve starts half a round late: A=1 fits badly, free A finds A > 1 and R0 = 0.5."""
    rounds, eps, offset = np.array([1, 2, 3, 5, 8, 10]), 0.002, 0.5
    shots = np.full(6, 20_000_000)
    failures = np.rint(shots * .5 * (1 - (1 - 2*eps)**(rounds - offset)))
    rows = [(int(r), int(n), int(k)) for r, n, k in zip(rounds, shots, failures)]
    sens = evaluate.sensitivity_fits(rows)
    free = sens["free_amplitude"]
    assert free["amplitude"] > 1 and free["round_offset"] == pytest.approx(offset, abs=0.02)
    assert free["epsilon"] == pytest.approx(eps, rel=1e-3) and free["lr_test_a_above_1"]["p_value"] < 1e-6
    assert free["goodness_of_fit"]["pearson_p"] > 0.5
    entry = dict(no_spam=evaluate._fit_both([(r, n, k) for r, n, k in rows])["no_spam"], sensitivity=sens)
    # With A forced to 1, dropping early rounds shrinks the offset's bias but does not remove it.
    all_r, r3, r5 = entry["no_spam"]["epsilon"], sens["a1_r_min_3"]["epsilon"], sens["a1_r_min_5"]["epsilon"]
    assert all_r < r3 < r5 < eps and sens["a1_r_min_5"]["rounds"] == [5, 8, 10]
    assert entry["no_spam"]["goodness_of_fit"]["pearson_p"] < 1e-6
    assert evaluate.trusted_fit(entry) == "free A, all R"
