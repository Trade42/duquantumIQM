"""Challenge LER estimation and fixed-shot comparison statistics."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2
from .decoding import fit_ler, wilson_interval
ROUNDS = (1, 2, 3)

def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str)+"\n", encoding="utf8")
    temporary.replace(path)


def challenge_fit(rounds, errors, shots):
    """Unchanged challenge estimator plus an explicit lack-of-fit diagnostic."""
    result = fit_ler(rounds, errors, shots, fit_spam=False)
    k, n, p = np.asarray(errors), np.asarray(shots), np.array(result["fitted_probabilities"])
    result["deviance"] = max(0., float(2*np.sum(xlogy(k, k/n)+xlogy(n-k, 1-k/n)
                                                -xlogy(k, p)-xlogy(n-k, 1-p))))
    result["nominal_df"] = len(k)-1
    result["goodness_of_fit_p"] = float(chi2.sf(result["deviance"], len(k)-1))
    result["stationary_model_rejected_5pct"] = result["goodness_of_fit_p"] < .05
    result["rounds"] = list(rounds)
    result["errors"], result["shots"] = list(map(int, k)), list(map(int, n))
    result["probabilities"] = (k/n).tolist()
    result["wilson_ci95"] = [wilson_interval(int(a), int(b)) for a, b in zip(k, n)]
    return result


def fit_many(k, n, rounds=ROUNDS):
    """Vectorized binomial MLE for a nonparametric stratified bootstrap."""
    k, n, r = np.asarray(k), np.asarray(n), np.asarray(rounds)
    lo, hi = np.zeros(k.shape[0]), np.full(k.shape[0], .5)
    def objective(e):
        p = np.clip((1-(1-2*e[:, None])**r)/2, 1e-15, 1-1e-15)
        return -np.sum(xlogy(k, p)+xlogy(n-k, 1-p), axis=1)
    for _ in range(64):
        left, right = (2*lo+hi)/3, (lo+2*hi)/3
        choose = objective(left) < objective(right)
        hi, lo = np.where(choose, right, hi), np.where(choose, lo, left)
    result = (lo+hi)/2
    result[np.all(k == 0, axis=1)] = 0
    return result


def ler_difference(first, second, *, paired, draws=10000, alpha=.05, seed=2203):
    """first minus second; resample within round and acquisition block.

    Paired samples use the joint four-cell failure table. Independent physical
    arms use separate empirical binomials. Model fit is recomputed each draw.
    """
    rng = np.random.default_rng(seed)
    a = np.zeros((draws, len(ROUNDS)), int)
    b = np.zeros_like(a)
    na, nb = np.zeros(len(ROUNDS), int), np.zeros(len(ROUNDS), int)
    for j, r in enumerate(ROUNDS):
        for block in sorted(first[r]):
            fa, fb = np.asarray(first[r][block], bool), np.asarray(second[r][block], bool)
            na[j] += len(fa); nb[j] += len(fb)
            if paired:
                if len(fa) != len(fb):
                    raise ValueError("Paired failure vectors must have identical shots.")
                cells = np.bincount(2*fa.astype(int)+fb, minlength=4)
                sample = rng.multinomial(len(fa), cells/len(fa), size=draws)
                a[:, j] += sample[:, 2]+sample[:, 3]
                b[:, j] += sample[:, 1]+sample[:, 3]
            else:
                a[:, j] += rng.binomial(len(fa), fa.mean(), size=draws)
                b[:, j] += rng.binomial(len(fb), fb.mean(), size=draws)
    difference = fit_many(a, na)-fit_many(b, nb)
    observed = challenge_fit(ROUNDS, [sum(v.sum() for v in first[r].values()) for r in ROUNDS], na)["epsilon_per_round"] \
        -challenge_fit(ROUNDS, [sum(v.sum() for v in second[r].values()) for r in ROUNDS], nb)["epsilon_per_round"]
    return dict(absolute_ler_reduction=observed, ci=np.quantile(difference, [alpha/2, 1-alpha/2]).tolist(),
                confidence=1-alpha, paired=paired, draws=draws, seed=seed,
                method="stratified nonparametric bootstrap of fixed-shot failures; refit challenge equation",
                excludes="calibration drift between blocks and model-selection uncertainty")
