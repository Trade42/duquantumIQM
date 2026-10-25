"""The logical error rate per round, ε_L, from the logical error probability at several round counts.

A memory experiment of n rounds fails with probability P_L(n). The logical fidelity
F = 1 − 2·P_L decays as F = A·(1 − 2ε_L)^n (Google Quantum AI, Nature 2021 and 2023, Fig. 3):

* **fixed A = 1**: 2·P_L = 1 − (1 − 2ε_L)^n, the challenge's formula;
* **free A**: the amplitude A ≤ 1 absorbs the error of preparation and of the final readout,
  which happen once per experiment rather than once per round.

:func:`fit_eps` fits either form by maximum likelihood on the binomial counts (k failures in n
shots at each round count), so a point with k = 0 or with P_L near ½ carries its proper weight.
It reports two 95% intervals for ε_L: one from the curvature of the log-likelihood (the Fisher
information at the fit), one from a seeded bootstrap over shots. For a pass/fail outcome per
shot, resampling a point's n shots with replacement gives exactly k* ~ Binomial(n, k/n), which
is what the bootstrap draws.

Every interval for a single probability is Wilson's (:func:`wilson`), at 95%.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import xlogy

Z95 = 1.959963984540054  # two-sided 95% quantile of the standard normal
EPS_MAX = 0.5 - 1e-9
_P_FLOOR = 1e-15


def wilson(k: Any, n: Any, z: float = Z95) -> tuple[Any, Any]:
    """Wilson score interval for k successes in n trials (95% by default). Vectorised."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    if np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n and n > 0")
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo = np.where(k == 0, 0.0, np.clip(centre - half, 0.0, 1.0))
    hi = np.where(k == n, 1.0, np.clip(centre + half, 0.0, 1.0))
    if lo.ndim == 0:
        return float(lo), float(hi)
    return lo, hi


def p_model(eps: float, rounds: Any, amplitude: float = 1.0) -> np.ndarray:
    """P_L after ``rounds`` rounds: (1 − A·(1 − 2ε)^rounds) / 2."""
    return 0.5 * (1 - amplitude * (1 - 2 * eps) ** np.asarray(rounds, dtype=float))


def eps_from_p(p: Any, rounds: Any) -> np.ndarray:
    """ε_L from one round count: ½·[1 − (1 − 2P)^(1/rounds)]; NaN where P > ½."""
    p = np.asarray(p, dtype=float)
    rounds = np.asarray(rounds, dtype=float)
    f = 1 - 2 * p
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(f >= 0, 0.5 * (1 - np.abs(f) ** (1 / rounds)), np.nan)


def eps_single(k: Any, n: Any, rounds: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ε_L from each round count alone, with the Wilson 95% interval of P mapped through.

    The map P -> ε is increasing on [0, ½], so the interval's ends map to the ends. An upper
    end at P ≥ ½ maps to ε = ½.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    lo, hi = wilson(k, n)
    return (
        eps_from_p(k / n, rounds),
        eps_from_p(np.minimum(lo, 0.5), rounds),
        eps_from_p(np.minimum(hi, 0.5), rounds),
    )


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EpsFit:
    """A maximum-likelihood fit of F = A·(1 − 2ε)^n to k failures in n shots per round count."""

    free_amplitude: bool
    eps: float
    amplitude: float
    eps_se: float  # from the curvature (Fisher information)
    amplitude_se: float  # NaN when A is fixed
    eps_ci_curvature: tuple[float, float]  # ε ± 1.96 σ, clipped to [0, ½]
    eps_ci_bootstrap: tuple[float, float]  # percentile 2.5 / 97.5; NaN without a bootstrap
    amplitude_ci_bootstrap: tuple[float, float]
    rounds: np.ndarray
    k: np.ndarray
    n: np.ndarray
    p_fit: np.ndarray  # the model's P_L at each round count
    residuals: np.ndarray  # Pearson: (k − n·P) / sqrt(n·P·(1 − P))
    chi2: float
    dof: int
    bootstrap: int
    seed: int | None

    @property
    def form(self) -> str:
        return "F = A(1-2eps)^n" if self.free_amplitude else "F = (1-2eps)^n"

    @property
    def chi2_per_dof(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "free_amplitude": self.free_amplitude,
            "eps": self.eps,
            "eps_se_curvature": self.eps_se,
            "eps_ci_curvature": list(self.eps_ci_curvature),
            "eps_ci_bootstrap": list(self.eps_ci_bootstrap),
            "amplitude": self.amplitude,
            "amplitude_se_curvature": self.amplitude_se,
            "amplitude_ci_bootstrap": list(self.amplitude_ci_bootstrap),
            "rounds": self.rounds.tolist(),
            "k": self.k.tolist(),
            "n": self.n.tolist(),
            "p_fit": self.p_fit.tolist(),
            "pearson_residuals": self.residuals.tolist(),
            "chi2": self.chi2,
            "dof": self.dof,
            "bootstrap": self.bootstrap,
            "seed": self.seed,
        }


def _unpack(theta: np.ndarray, free: bool) -> tuple[float, float]:
    return float(theta[0]), float(theta[1]) if free else 1.0


def _nll_and_grad(theta: np.ndarray, r: np.ndarray, k: np.ndarray, n: np.ndarray, free: bool):
    eps, a = _unpack(theta, free)
    x = 1 - 2 * eps
    xr = x**r
    p = np.clip(0.5 * (1 - a * xr), _P_FLOOR, 1 - _P_FLOOR)
    nll = -float(np.sum(xlogy(k, p) + xlogy(n - k, 1 - p)))
    dl_dp = k / p - (n - k) / (1 - p)  # d(log-likelihood)/dP
    grad = [-float(np.sum(dl_dp * a * r * x ** (r - 1)))]
    if free:
        grad.append(-float(np.sum(dl_dp * (-0.5 * xr))))
    return nll, np.array(grad)


def _fisher(eps: float, a: float, r: np.ndarray, n: np.ndarray, free: bool) -> np.ndarray:
    x = 1 - 2 * eps
    p = np.clip(0.5 * (1 - a * x**r), _P_FLOOR, 1 - _P_FLOOR)
    g = [a * r * x ** (r - 1)]
    if free:
        g.append(-0.5 * x**r)
    g = np.array(g)  # (params, points)
    w = n / (p * (1 - p))
    return (g * w) @ g.T


def _bounds(free: bool) -> list[tuple[float, float]]:
    return [(0.0, EPS_MAX), (1e-9, 1.0)] if free else [(0.0, EPS_MAX)]


def _mle(r, k, n, free: bool, starts: Sequence[Sequence[float]]) -> np.ndarray:
    best = None
    for x0 in starts:
        res = minimize(
            _nll_and_grad,
            np.array(x0, dtype=float),
            args=(r, k, n, free),
            jac=True,
            method="L-BFGS-B",
            bounds=_bounds(free),
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def _starts(r: np.ndarray, k: np.ndarray, n: np.ndarray, free: bool) -> list[list[float]]:
    """Starting points: the median single-count ε, and a weighted line through log F."""
    singles = eps_from_p(k / n, r)
    singles = singles[np.isfinite(singles)]
    e0 = float(np.clip(np.median(singles) if singles.size else 0.01, 1e-6, 0.49))
    if not free:
        return [[e0]]
    starts = [[e0, 1.0]]
    p = np.maximum(k, 0.5) / n
    f = 1 - 2 * p
    ok = f > 0
    if ok.sum() >= 2:
        w = n[ok] * f[ok] ** 2 / (4 * p[ok] * (1 - p[ok]))  # 1 / Var(log F)
        slope, intercept = np.polyfit(r[ok], np.log(f[ok]), 1, w=np.sqrt(w))
        starts.append(
            [float(np.clip((1 - np.exp(slope)) / 2, 1e-6, 0.49)), float(np.clip(np.exp(intercept), 1e-3, 1.0))]
        )
    return starts


def fit_eps(
    rounds: Sequence[int],
    k: Sequence[int],
    n: Sequence[int],
    *,
    free_amplitude: bool = False,
    bootstrap: int = 2000,
    seed: int | None = 2026,
) -> EpsFit:
    """Fit ε_L (and A, with ``free_amplitude``) to k failures in n shots at each round count.

    ``bootstrap`` replicates redraw every point's failures as Binomial(n, k/n), the same as
    resampling its shots with replacement, and refit; ``seed`` seeds that draw. ``bootstrap=0``
    skips it. With A free and at its bound A = 1, the curvature interval is only approximate.
    """
    r = np.asarray(rounds, dtype=float)
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    if not (r.shape == k.shape == n.shape) or r.ndim != 1:
        raise ValueError("rounds, k and n must be 1-D and the same length")
    if np.any(r < 1) or np.any(n <= 0) or np.any(k < 0) or np.any(k > n):
        raise ValueError("need rounds >= 1, n > 0 and 0 <= k <= n")
    n_params = 2 if free_amplitude else 1
    if len(np.unique(r)) < n_params:
        raise ValueError(f"need at least {n_params} distinct round counts")

    theta = _mle(r, k, n, free_amplitude, _starts(r, k, n, free_amplitude))
    eps, a = _unpack(theta, free_amplitude)

    cov = np.linalg.pinv(_fisher(eps, a, r, n, free_amplitude))
    eps_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    a_se = float(np.sqrt(max(cov[1, 1], 0.0))) if free_amplitude else float("nan")
    ci_curv = (max(eps - Z95 * eps_se, 0.0), min(eps + Z95 * eps_se, 0.5))

    ci_boot = (float("nan"), float("nan"))
    a_ci_boot = (float("nan"), float("nan"))
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        draws = np.empty((bootstrap, n_params))
        for b in range(bootstrap):
            kb = rng.binomial(n.astype(np.int64), k / n).astype(float)
            draws[b] = _mle(r, kb, n, free_amplitude, [theta])
        lo, hi = np.percentile(draws[:, 0], [2.5, 97.5])
        ci_boot = (float(lo), float(hi))
        if free_amplitude:
            alo, ahi = np.percentile(draws[:, 1], [2.5, 97.5])
            a_ci_boot = (float(alo), float(ahi))

    p_fit = p_model(eps, r, a)
    resid = (k - n * p_fit) / np.sqrt(n * np.clip(p_fit * (1 - p_fit), _P_FLOOR, None))
    return EpsFit(
        free_amplitude=free_amplitude,
        eps=eps,
        amplitude=a,
        eps_se=eps_se,
        amplitude_se=a_se,
        eps_ci_curvature=(float(ci_curv[0]), float(ci_curv[1])),
        eps_ci_bootstrap=ci_boot,
        amplitude_ci_bootstrap=a_ci_boot,
        rounds=r.astype(int),
        k=k.astype(int),
        n=n.astype(int),
        p_fit=p_fit,
        residuals=resid,
        chi2=float(np.sum(resid**2)),
        dof=len(r) - n_params,
        bootstrap=bootstrap,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# The plot: P_L (or F) against rounds, with the fits, as in Google 2023, Fig. 3.
# ---------------------------------------------------------------------------

# Categorical slots in fixed order (one per data series), and neutral inks for text and grid.
_SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3df"
_SURFACE = "#fcfcfb"


def plot_fidelity_decay(
    series: Mapping[str, Sequence[EpsFit]],
    *,
    quantity: str = "P",
    title: str | None = None,
    source: str | None = None,
    path: str | Path | None = None,
):
    """Plot P_L (``quantity="P"``) or F = 1 − 2·P_L (``"F"``, log scale) against rounds.

    ``series`` maps a label (``"Z basis"``, say) to the fits made to one data set; the points and
    their Wilson 95% bars come from the first fit, and every fit is drawn as a curve (solid for
    free A, dashed for A = 1). A lower panel shows each fit's Pearson residuals. ``source`` names
    the data file, printed under the plot. Saves to ``path`` when given; returns the figure.
    """
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter

    if quantity not in ("P", "F"):
        raise ValueError('quantity must be "P" or "F"')
    if len(series) > len(_SERIES):
        raise ValueError(f"at most {len(_SERIES)} data series per plot")

    fig = Figure(figsize=(7.2, 6.4), facecolor=_SURFACE, layout="constrained")
    ax, rax = fig.subplots(2, 1, sharex=True, height_ratios=(3, 1))
    for a in (ax, rax):
        a.set_facecolor(_SURFACE)
        a.grid(True, color=_GRID, linewidth=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=_INK_2, labelcolor=_INK_2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(_INK_2)

    n_fits = sum(len(f) for f in series.values())
    offsets = np.linspace(-0.12, 0.12, n_fits) if n_fits > 1 else np.zeros(1)
    i_fit = 0
    r_max = 1
    for color, (label, fits) in zip(_SERIES, series.items()):
        if not fits:
            continue
        first = fits[0]
        r, k, n = first.rounds, first.k, first.n
        r_max = max(r_max, int(r.max()))
        p = k / n
        lo, hi = wilson(k, n)
        if quantity == "P":
            y, ylo, yhi = p, lo, hi
        else:
            y, ylo, yhi = 1 - 2 * p, 1 - 2 * hi, 1 - 2 * lo
        yerr = np.vstack([y - ylo, yhi - y])
        shown = np.ones_like(y, dtype=bool)
        if quantity == "F":
            shown = y > 0  # a log axis cannot show F <= 0
            yerr[0] = np.minimum(yerr[0], y * (1 - 1e-3))
        ax.errorbar(
            r[shown],
            y[shown],
            yerr=yerr[:, shown],
            fmt="o",
            ms=6,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=2.5,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.0,
            label=f"{label}: data, Wilson 95%",
            zorder=3,
        )
        grid = np.linspace(0, r.max() + 0.5, 200)
        for fit in fits:
            curve = p_model(fit.eps, grid, fit.amplitude)
            if quantity == "F":
                curve = 1 - 2 * curve
            boot = bool(np.all(np.isfinite(fit.eps_ci_bootstrap)))
            ci = fit.eps_ci_bootstrap if boot else fit.eps_ci_curvature
            amp = f"free A = {fit.amplitude:.4f}" if fit.free_amplitude else "A = 1"
            ax.plot(
                grid,
                curve,
                color=color,
                linewidth=2.0,
                linestyle="-" if fit.free_amplitude else "--",
                label=(
                    f"{label}, {amp}: $\\varepsilon_L$ = {fit.eps:.5f} [{ci[0]:.5f}, {ci[1]:.5f}] "
                    f"({'bootstrap' if boot else 'curvature'})"
                ),
                zorder=2,
            )
            rax.plot(
                fit.rounds + offsets[i_fit],
                fit.residuals,
                linestyle="none",
                marker="o",
                ms=6,
                color=color,
                markerfacecolor=color if fit.free_amplitude else _SURFACE,
                markeredgewidth=1.5,
                label=f"{label}, {'free A' if fit.free_amplitude else 'A = 1'}",
            )
            i_fit += 1

    if quantity == "P":
        ax.set_ylabel("logical error probability $P_L$")
        ax.set_ylim(bottom=0)
    else:
        ax.set_ylabel("logical fidelity $F = 1 - 2P_L$ (log scale)")
        ax.set_yscale("log")
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
    ax.yaxis.label.set_color(_INK)
    ax.legend(
        frameon=False, fontsize=8, labelcolor=_INK, loc="upper left" if quantity == "P" else "lower left",
        title="fits: $\\varepsilon_L$ per round [95% interval]", title_fontsize=8,
    )
    for yline, style in ((0, "-"), (-2, ":"), (2, ":")):
        rax.axhline(yline, color=_INK_2, linewidth=0.8, linestyle=style, zorder=1)
    rax.set_ylabel("Pearson residual\n(σ)", color=_INK)
    rax.set_xlabel("QEC rounds $n$", color=_INK)
    rax.set_xlim(-0.3, r_max + 0.5)
    rax.set_xticks(range(0, r_max + 1))
    rax.legend(frameon=False, fontsize=7, labelcolor=_INK, ncol=2, loc="best")
    if title:
        ax.set_title(title, color=_INK, fontsize=11, loc="left")
    if source:
        fig.supxlabel(f"data: {source}", x=0.01, ha="left", color=_INK_2, fontsize=7)
    if path is not None:
        fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig
