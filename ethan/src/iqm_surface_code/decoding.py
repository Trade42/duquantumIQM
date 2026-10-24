"""Correlated MWPM, binomial intervals, multi-round maximum-likelihood LER fit."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
import pymatching
from scipy.optimize import minimize_scalar, brentq
from functools import lru_cache
from scipy.stats import chi2


def wilson_interval(errors, shots, z=1.959963984540054):
    if shots < 1 or not 0 <= errors <= shots:
        raise ValueError("Require 0 <= errors <= shots and shots > 0.")
    p, z2 = errors/shots, z*z
    center = (p+z2/(2*shots))/(1+z2/shots)
    half = z*math.sqrt(p*(1-p)/shots+z2/(4*shots*shots))/(1+z2/shots)
    return max(0., center-half), min(1., center+half)


@dataclass(frozen=True)
class DecodeResult:
    shots: int
    errors: int
    probability: float
    standard_error: float
    ci95: tuple[float, float]
    raw_logical_flip_probability: float

    def to_dict(self):
        return asdict(self)


def make_matcher(circuit, *, correlated=True):
    # No gauge detectors or ignored decomposition failures: fail loudly.
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem, enable_correlations=correlated)
    matcher.ensure_num_fault_ids(circuit.num_observables)
    return matcher


def decode_syndromes(syndromes, circuit, *, correlated=True, matcher=None):
    dets, obs = np.asarray(syndromes["det_events"], dtype=bool), np.asarray(syndromes["obs_flips"], dtype=bool)
    if dets.ndim != 2 or obs.ndim != 2 or len(dets) != len(obs) or len(obs) == 0:
        raise ValueError("Expected nonempty 2D detector/observable arrays with equal shots.")
    if dets.shape[1] != circuit.num_detectors or obs.shape[1] != circuit.num_observables:
        raise ValueError("Decoder circuit and syndrome dimensions do not match.")
    source = syndromes.get("stim_circuit")
    if source is not None and source.without_noise() != circuit.without_noise():
        raise ValueError("Decoder circuit differs from executed circuit, including parity records.")
    matcher = matcher if matcher is not None else make_matcher(circuit, correlated=correlated)
    predictions = matcher.decode_batch(dets, enable_correlations=correlated)
    fail = np.any(predictions != obs, axis=1)
    errors, shots = int(fail.sum()), len(fail)
    p = errors/shots
    return DecodeResult(shots, errors, p, math.sqrt(p*(1-p)/shots), wilson_interval(errors, shots),
                        float(np.any(obs, axis=1).mean()))


def fit_ler(rounds, errors, shots, *, fit_spam=False):
    """Fit P(r)=[1-A(1-2*epsilon)^r]/2 by binomial likelihood.

    Default A=1 reproduces the challenge equation. Optional SPAM amplitude A
    requires >=3 distinct counts. Profile likelihood gives epsilon CI, including
    zero-error results. No log-fit clipping or fabricated positive failures.
    """
    r, k, n = (np.asarray(v, dtype=float) for v in (rounds, errors, shots))
    if r.ndim != 1 or r.shape != k.shape or r.shape != n.shape or not len(r):
        raise ValueError("Round/error/shot arrays must have equal 1D shape.")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(n)):
        raise ValueError("Fit inputs must be finite.")
    if np.any(r < 1) or np.any(r != np.floor(r)) or np.any(n < 1) or np.any(k < 0) or np.any(k > n) or np.any(k != np.floor(k)) or np.any(n != np.floor(n)):
        raise ValueError("Invalid integer rounds or binomial counts.")
    if len(set(r)) < (3 if fit_spam else 2):
        raise ValueError("Need >=3 distinct rounds for SPAM fit, >=2 otherwise.")

    def objective(eps, amplitude=1.):
        p = .5*(1-amplitude*np.power(1-2*eps, r))
        # Endpoint likelihood is exact when counts at that endpoint agree.
        if np.any((p == 0) & (k > 0)) or np.any((p == 1) & (k < n)):
            return float("inf")
        from scipy.special import xlogy
        return float(-np.sum(xlogy(k, p)+xlogy(n-k, 1-p)))

    @lru_cache(maxsize=1024)
    def best_amplitude(e):
        fit = minimize_scalar(lambda a: objective(e,a),bounds=(0,1),method="bounded",
                              options={"xatol":1e-12})
        amplitude = min((0.,float(fit.x),1.),key=lambda a:objective(e,a))
        return amplitude,objective(e,amplitude)

    if fit_spam:
        # Profile the convex amplitude subproblem. A clipped 2D simplex can
        # stick at A=1 and report a non-optimal fit near the boundary.
        outer = minimize_scalar(lambda e:best_amplitude(float(e))[1],bounds=(0,.5),
                                method="bounded",options={"xatol":1e-12})
        if not outer.success:
            raise RuntimeError(f"LER fit failed: {outer.message}")
        eps = min((0.,float(outer.x),.5),key=lambda e:best_amplitude(e)[1])
        amp = best_amplitude(eps)[0]
    else:
        fit = minimize_scalar(objective, bounds=(0, .5), method="bounded", options={"xatol": 1e-12})
        eps = min((0., float(fit.x), .5), key=objective)
        amp = 1.
    best = objective(eps, amp)
    threshold = chi2.ppf(.95, 1)/2

    def profile(e):
        if not fit_spam:
            return objective(e)-best-threshold
        value = best_amplitude(float(e))[1]
        return value-best-threshold
    lo = 0. if eps == 0 or profile(0) <= 0 else brentq(profile, 0, eps)
    hi = .5 if eps == .5 or profile(.5) <= 0 else brentq(profile, eps, .5)
    fitted = .5*(1-amp*np.power(1-2*eps, r))
    return dict(epsilon_per_round=eps, ci95_profile=[lo, hi], amplitude=amp,
        fit_spam=fit_spam, fitted_probabilities=fitted.tolist(), negative_log_likelihood=best,
        boundary_fit=bool(eps < 1e-8 or eps > .5-1e-8),
        uncertainty_method="asymptotic profile likelihood; fixed-shot binomial observations")
