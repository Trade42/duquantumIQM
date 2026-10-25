"""dqiqm.stats: Wilson intervals, single-count ε, and the fit of F = A(1 − 2ε)^n on small synthetic sets."""

import numpy as np
import pytest

from dqiqm import stats
from dqiqm.stats import eps_from_p, eps_single, fit_eps, p_model, wilson

SEED = 2026
ROUNDS = np.arange(1, 11)


def test_wilson_known_values_and_edges():
    assert wilson(10, 100) == pytest.approx((0.05523, 0.17437), abs=1e-5)
    assert wilson(0, 100) == pytest.approx((0.0, 0.03699), abs=1e-5)
    assert wilson(100, 100) == pytest.approx((0.96301, 1.0), abs=1e-5)
    lo, hi = wilson([0, 10, 100], [100, 100, 100])
    assert lo.shape == (3,) and lo[0] == 0.0 and hi[2] == 1.0
    with pytest.raises(ValueError):
        wilson(5, 4)


def test_single_count_eps_inverts_the_model():
    for eps, n in ((0.0, 3), (0.004, 1), (0.02, 7), (0.2, 10)):
        assert eps_from_p(p_model(eps, n), n) == pytest.approx(eps, abs=1e-12)
    assert np.isnan(eps_from_p(0.6, 2))
    e, lo, hi = eps_single([50], [1000], [5])
    assert lo[0] < e[0] < hi[0]
    assert (lo[0], hi[0]) == pytest.approx(tuple(eps_from_p(np.array(wilson(50, 1000)), 5)), abs=1e-15)


@pytest.mark.parametrize("amplitude", [1.0, 0.9])
def test_fit_recovers_eps_and_amplitude_on_exact_data(amplitude):
    n = np.full(10, 10**6)
    k = np.round(n * p_model(0.02, ROUNDS, amplitude))
    free = fit_eps(ROUNDS, k, n, free_amplitude=True, bootstrap=0)
    assert free.eps == pytest.approx(0.02, abs=1e-6) and free.amplitude == pytest.approx(amplitude, abs=1e-5)
    assert free.chi2 < 1e-3 and free.dof == 8
    fixed = fit_eps(ROUNDS, k, n, bootstrap=0)
    assert fixed.amplitude == 1.0 and fixed.dof == 9
    if amplitude == 1.0:
        assert fixed.eps == pytest.approx(0.02, abs=1e-6) and fixed.chi2 < 1e-3
    else:
        # A = 1 cannot describe a decay with A = 0.9: the fit is biased, and its residuals say so
        assert fixed.eps > 0.025 and fixed.chi2_per_dof > 100


def test_fit_intervals_cover_the_truth_and_the_bootstrap_is_seeded():
    rng = np.random.default_rng(SEED)
    n = np.full(10, 20_000)
    k = rng.binomial(n, p_model(0.01, ROUNDS, 0.95))
    fit = fit_eps(ROUNDS, k, n, free_amplitude=True, bootstrap=300, seed=7)
    for lo, hi in (fit.eps_ci_curvature, fit.eps_ci_bootstrap):
        assert lo < 0.01 < hi
    assert fit.amplitude_ci_bootstrap[0] < 0.95 < fit.amplitude_ci_bootstrap[1]
    # the two intervals have about the same width
    width = lambda ci: ci[1] - ci[0]  # noqa: E731
    assert 0.7 < width(fit.eps_ci_bootstrap) / width(fit.eps_ci_curvature) < 1.3
    assert fit.chi2_per_dof < 3
    again = fit_eps(ROUNDS, k, n, free_amplitude=True, bootstrap=300, seed=7)
    assert again.eps_ci_bootstrap == fit.eps_ci_bootstrap
    other = fit_eps(ROUNDS, k, n, free_amplitude=True, bootstrap=300, seed=8)
    assert other.eps_ci_bootstrap != fit.eps_ci_bootstrap
    d = fit.to_dict()
    assert d["form"] == "F = A(1-2eps)^n" and len(d["pearson_residuals"]) == 10


def test_fit_handles_zero_failures():
    fit = fit_eps([1, 2, 3], [0, 0, 1], [1000, 1000, 1000], bootstrap=50)
    assert 0 <= fit.eps < 1e-3 and fit.eps_ci_curvature[0] == 0.0


def test_fit_refuses_bad_input():
    with pytest.raises(ValueError):
        fit_eps([1, 2], [1, 2, 3], [10, 10, 10])
    with pytest.raises(ValueError):
        fit_eps([0, 1], [1, 2], [10, 10])
    with pytest.raises(ValueError):
        fit_eps([3, 3], [1, 2], [10, 10], free_amplitude=True)


@pytest.mark.parametrize("quantity", ["P", "F"])
def test_plot_fidelity_decay_writes_a_png(tmp_path, quantity):
    rng = np.random.default_rng(SEED)
    n = np.full(10, 5000)
    series = {}
    for label, eps in (("Z basis", 0.01), ("X basis", 0.012)):
        k = rng.binomial(n, p_model(eps, ROUNDS, 0.97))
        series[label] = [fit_eps(ROUNDS, k, n, free_amplitude=True, bootstrap=0), fit_eps(ROUNDS, k, n, bootstrap=0)]
    out = tmp_path / f"decay-{quantity}.png"
    fig = stats.plot_fidelity_decay(series, quantity=quantity, title="test", source="synthetic", path=out)
    assert out.stat().st_size > 10_000 and len(fig.axes) == 2
