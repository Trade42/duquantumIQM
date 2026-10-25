"""The fits that fill the table: C1 per qubit, C2 per coupler, each with a 95% interval.

C1: maximum likelihood per qubit
    The data are, per C1 circuit, the number of shots in which the qubit read 1. The model is
    :mod:`.model`'s exact Bloch-vector evolution with five free parameters: eps and eta (the
    coherent errors), lam (depolarizing per gate) and the readout a0 = P(1|0), a1 = P(0|1). T1, T2
    and the gate time are fixed from the calibration. The binomial likelihood is maximised from a
    grid of starts (so an aliased eps cannot win), and the interval is Wald's from the observed
    information (the numerical Hessian of the negative log-likelihood), widened by
    sqrt(max(1, chi2/dof)) when the data scatter more than binomially (drift, model gaps). A
    parameter that ends on a bound (lam = 0 or a readout error of 0 on a clean simulation) is held
    there for the Hessian.

C2: phases, then a straight line in n
    Per (control state s, CZ count n), the target's four analysis phases give
    sin(Phi) = P(psi = pi) - P(0) and cos(Phi) = P(pi/2) - P(3pi/2), so Phi = atan2(sin, cos). The
    differences cancel every readout offset, and the readout and decay only scale the contrast,
    never the phase. Its variance is the delta-method one. Then, after taking n pi out of s = 1,

        Phi_0(n) = A_0 + n phi_t,        Phi_1(n) - n pi = A_1 + n (phi_t + delta),

    fitted by weighted least squares on phases unwrapped point by point in increasing n (each one
    onto the branch the fit so far predicts, as in robust phase estimation). The intercepts are
    separate per s, so nothing that is constant in n (the preparation, the analysis pulse, a
    crosstalk of the control's pulses) can bias the slopes. The interval is the WLS covariance,
    widened by sqrt(max(1, chi2/dof)). Both orientations measure delta; the table gives their
    inverse-variance mean and, separately, each local phase.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize

from dqiqm.characterization import model as M

Z95 = 1.959963984540054
C1_PARAMS = ("eps", "eta", "lam", "a0", "a1")
C1_BOUNDS = {"eps": (-0.4, 0.4), "eta": (-0.4, 0.4), "lam": (0.0, 0.2), "a0": (0.0, 0.5), "a1": (0.0, 0.5)}
P_CLIP = 1e-12


def wrap(x: float) -> float:
    """Into (-pi, pi]."""
    y = (x + math.pi) % (2 * math.pi) - math.pi
    return math.pi if y == -math.pi else y


# ---------------------------------------------------------------------------
# C1.
# ---------------------------------------------------------------------------


def _rot(phase: float, eps: float, eta: float) -> np.ndarray:
    """The SO(3) rotation of G(phase) (Rodrigues), equal to ``model.rotation(model.prx_actual(...))``."""
    a = M.HALF_PI + eps
    v = np.array([a * math.cos(phase), a * math.sin(phase), eta])
    th = float(np.linalg.norm(v))
    if th == 0.0:
        return np.eye(3)
    n = v / th
    k = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


@dataclass(frozen=True)
class C1Data:
    """One qubit's C1 data: per circuit its sequence (runs of (phase, repeat)), shots and ones."""

    qubit: str
    sequences: tuple[tuple[tuple[float, int], ...], ...]
    shots: np.ndarray
    ones: np.ndarray
    names: tuple[str, ...] = ()
    t1_ns: float = math.inf
    t2_ns: float = math.inf
    gate_ns: float = 16.0


def c1_probs(d: C1Data, theta: Sequence[float]) -> np.ndarray:
    """P(read 1) for every circuit of ``d`` under theta = (eps, eta, lam, a0, a1)."""
    eps, eta, lam, a0, a1 = theta
    relax = M.relaxation(d.gate_ns, d.t1_ns, d.t2_ns) if math.isfinite(d.t1_ns) else np.eye(4)
    tail = M.depolarize(lam) @ relax
    steps: dict[float, np.ndarray] = {}
    powers: dict[tuple[float, int], np.ndarray] = {}
    out = np.empty(len(d.sequences))
    for i, seq in enumerate(d.sequences):
        v = np.array([0.0, 0.0, 1.0, 1.0])
        for phase, k in seq:
            if k <= 0:
                continue
            if phase not in steps:
                steps[phase] = tail @ M.affine(_rot(phase, eps, eta))
            key = (phase, k)
            if key not in powers:
                powers[key] = np.linalg.matrix_power(steps[phase], k)
            v = powers[key] @ v
        p1 = (1 - v[2]) / 2
        out[i] = a0 + (1 - a0 - a1) * p1
    return out


def _nll(theta: Sequence[float], d: C1Data) -> float:
    p = np.clip(c1_probs(d, theta), P_CLIP, 1 - P_CLIP)
    return float(-np.sum(d.ones * np.log(p) + (d.shots - d.ones) * np.log(1 - p)))


def _hessian(f, x: np.ndarray, steps: np.ndarray) -> np.ndarray:
    n = len(x)
    h = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = steps[i]
        h[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / steps[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = steps[j]
            h[i, j] = h[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * steps[i] * steps[j])
    return h


@dataclass
class Fit:
    """A fit's estimates, standard errors and 95% intervals (already widened by ``inflation``)."""

    locus: str
    values: dict[str, float]
    se: dict[str, float]
    ci: dict[str, tuple[float, float]]
    chi2: float
    dof: int
    inflation: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def chi2_per_dof(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def to_json(self) -> dict[str, Any]:
        return {"locus": self.locus, "values": self.values, "se": self.se,
                "ci95": {k: list(v) for k, v in self.ci.items()}, "chi2": self.chi2, "dof": self.dof,
                "chi2_per_dof": self.chi2_per_dof, "inflation": self.inflation, **self.extra}


def _readout_start(d: C1Data) -> tuple[float, float]:
    """a0, a1 from the N = 0 and N = 2 circuits, if present (an X ends in |1>)."""
    a0 = a1 = 0.02
    for seq, s, k in zip(d.sequences, d.shots, d.ones):
        if seq == ():
            a0 = float(k / s)
        elif seq == ((0.0, 2),):
            a1 = float(1 - k / s)
    return min(max(a0, 1e-4), 0.4), min(max(a1, 1e-4), 0.4)


def fit_c1(d: C1Data, *, lam0: float = 0.002, fixed: Mapping[str, float] | None = None) -> Fit:
    """The C1 maximum-likelihood fit of one qubit. ``fixed`` holds parameters at given values
    (C4's reduced sets fix the readout and lam at the parallel fit's)."""
    fixed = dict(fixed or {})
    free = [p for p in C1_PARAMS if p not in fixed]
    a0, a1 = _readout_start(d)
    base = {"eps": 0.0, "eta": 0.0, "lam": lam0, "a0": a0, "a1": a1}

    def full(x: np.ndarray) -> list[float]:
        vals = dict(base)
        vals.update(fixed)
        vals.update(dict(zip(free, x)))
        return [vals[p] for p in C1_PARAMS]

    def f(x: np.ndarray) -> float:
        return _nll(full(x), d)

    bounds = [C1_BOUNDS[p] for p in free]
    # a coarse grid over the coherent errors first, so that an aliased eps (N eps past pi) cannot win
    grid = []
    eps_grid = np.linspace(-0.12, 0.12, 25) if "eps" in free else [fixed.get("eps", 0.0)]
    eta_grid = np.linspace(-0.12, 0.12, 13) if "eta" in free else [fixed.get("eta", 0.0)]
    for e in eps_grid:
        for h in eta_grid:
            x0 = np.array([{"eps": e, "eta": h}.get(p, base[p]) for p in free])
            grid.append((f(x0), x0))
    grid.sort(key=lambda t: t[0])
    best = None
    for _, x0 in grid[:4]:
        res = minimize(f, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    x = np.asarray(best.x, dtype=float)
    # the Hessian over the parameters off their bounds
    inner = [i for i, p in enumerate(free) if bounds[i][0] + 1e-7 < x[i] < bounds[i][1] - 1e-7]
    steps = np.array([1e-4 if free[i] in ("eps", "eta") else 2e-5 for i in inner])

    def f_inner(y: np.ndarray) -> float:
        z = x.copy()
        z[inner] = y
        return f(z)

    cov = np.full((len(inner), len(inner)), np.nan)
    if inner:
        h = _hessian(f_inner, x[inner], steps)
        try:
            cov = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            pass
    theta = full(x)
    p = np.clip(c1_probs(d, theta), P_CLIP, 1 - P_CLIP)
    chi2 = float(np.sum((d.ones - d.shots * p) ** 2 / (d.shots * p * (1 - p))))
    dof = len(d.shots) - len(free)
    infl = math.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0
    values = dict(zip(C1_PARAMS, (float(v) for v in theta)))
    se: dict[str, float] = {}
    ci: dict[str, tuple[float, float]] = {}
    for name in C1_PARAMS:
        if name in fixed:
            se[name] = float("nan")
        elif free.index(name) in inner:
            j = inner.index(free.index(name))
            se[name] = float(math.sqrt(cov[j, j])) if np.isfinite(cov[j, j]) and cov[j, j] > 0 else float("nan")
        else:
            se[name] = 0.0  # on a bound
        w = Z95 * se[name] * infl if np.isfinite(se[name]) else float("nan")
        ci[name] = (values[name] - w, values[name] + w)
    return Fit(d.qubit, values, se, ci, chi2, dof, infl,
               {"nll": float(best.fun), "converged": bool(best.success), "fixed": fixed,
                "at_bound": [p for i, p in enumerate(free) if i not in inner],
                "t1_ns": d.t1_ns, "t2_ns": d.t2_ns, "gate_ns": d.gate_ns, "circuits": len(d.shots),
                "shots": int(np.sum(d.shots))})


# ---------------------------------------------------------------------------
# C2.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhasePoint:
    s: int
    n: int
    phi: float  # rad, wrapped
    var: float
    contrast: float
    shots: int


def phase_point(s: int, n: int, ones: Sequence[int], shots: Sequence[int]) -> PhasePoint:
    """Phi from the four analysis phases psi = 0, pi/2, pi, 3pi/2 (ones and shots in that order)."""
    p = np.asarray(ones, float) / np.asarray(shots, float)
    v = np.maximum(p * (1 - p), 1.0 / np.asarray(shots, float)) / np.asarray(shots, float)
    sin_, cos_ = p[2] - p[0], p[1] - p[3]
    var_s, var_c = v[2] + v[0], v[1] + v[3]
    phi = math.atan2(sin_, cos_)
    c2 = sin_ ** 2 + cos_ ** 2
    var = (math.cos(phi) ** 2 * var_s + math.sin(phi) ** 2 * var_c) / max(c2, 1e-12)
    return PhasePoint(s, n, phi, var, math.sqrt(c2), int(sum(shots)))


def _wls(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    xtw = x.T * w
    cov = np.linalg.inv(xtw @ x)
    beta = cov @ (xtw @ y)
    r = y - x @ beta
    return beta, cov, float(np.sum(w * r * r))


def _unwrap(points: Sequence[PhasePoint]) -> list[float]:
    """Each point's phase (with n pi removed for s = 1) on the branch that a line through the
    earlier points predicts, in increasing n."""
    pts = sorted(points, key=lambda p: p.n)
    out: list[float] = []
    for i, p in enumerate(pts):
        y = p.phi - (p.n * math.pi if p.s == 1 else 0.0)
        if i == 0:
            out.append(wrap(y))
            continue
        if i == 1:  # the first step is taken to be within +-pi
            pred = out[0]
        else:
            ns = np.array([q.n for q in pts[:i]], float)
            w = np.array([1 / q.var for q in pts[:i]])
            beta, _, _ = _wls(np.column_stack([np.ones(i), ns]), np.array(out), w)
            pred = beta[0] + beta[1] * p.n
        out.append(pred + wrap(y - pred))
    return out


def fit_c2(points: Sequence[PhasePoint], locus: str) -> Fit:
    """delta and phi_t (with A_0, A_1) from one orientation's phase points."""
    rows, ys, ws, meta = [], [], [], []
    for s in (0, 1):
        pts = sorted([p for p in points if p.s == s], key=lambda p: p.n)
        for p, y in zip(pts, _unwrap(pts)):
            rows.append([1.0 - s, float(s), float(p.n), float(s * p.n)])
            ys.append(y)
            ws.append(1.0 / p.var)
            meta.append({"s": s, "n": p.n, "phi_unwrapped": y, "se": math.sqrt(p.var), "contrast": p.contrast,
                         "shots": p.shots})
    x, y, w = np.array(rows), np.array(ys), np.array(ws)
    beta, cov, chi2 = _wls(x, y, w)
    dof = len(y) - 4
    infl = math.sqrt(max(1.0, chi2 / dof)) if dof > 0 else 1.0
    names = ("A0", "A1", "phi_t", "delta")
    values = {k: float(v) for k, v in zip(names, beta)}
    values["delta"] = wrap(values["delta"])
    values["phi_t"] = wrap(values["phi_t"])
    se = {k: float(math.sqrt(cov[i, i])) for i, k in enumerate(names)}
    ci = {k: (values[k] - Z95 * se[k] * infl, values[k] + Z95 * se[k] * infl) for k in names}
    for m, r in zip(meta, y - x @ beta):
        m["residual"] = float(r)
    return Fit(locus, values, se, ci, chi2, dof, infl, {"points": meta})


def combine(values: Sequence[float], ses: Sequence[float]) -> tuple[float, float, float]:
    """Inverse-variance mean, its standard error, and the chi2 of the inputs against it."""
    v, s = np.asarray(values, float), np.asarray(ses, float)
    w = 1 / s ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    return mean, float(math.sqrt(1 / np.sum(w))), float(np.sum(w * (v - mean) ** 2))
