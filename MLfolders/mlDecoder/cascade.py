"""A two-layer decoder: the LSTM decides the shots it is confident about, Relay-BP the rest.

Layer 1, the recurrent decoder (rnn_lstm), decodes every shot and gives P(flip) = p. Its confidence
is c = |2p − 1| in [0, 1]. Shots with c < τ go to layer 2, Relay-BP (``relay.RelayBP``) on the decoder
prior's detector error model. Layer 3, correlated MWPM on the same prior, catches the shots Relay-BP
does not converge on. Every shot gets a decision, and ``decode_detailed`` says which layer made it.

τ is tuned on validation shots only (``tune``), by the one-standard-error rule (dq-orch-3, 2026-10-25, fixed
before the test batch was opened): τ* has the fewest validation failures, and τ is the smallest τ ≤ τ* whose
failures exceed τ*'s by at most one standard error of the paired difference, sqrt(b + c), with b and c the
shots on which only one of the two fails. A deferral is kept only where it beats paired noise; τ* is recorded
beside it. ``curve`` is the whole trade-off: τ against the deferred fraction, the failures, and the mean
latency the measured per-layer costs imply.

Layers 2 and 3 share one decoder prior, chosen per (device patch, basis) on validation before τ
(``choose_prior``, the same test): the calibrated prior, unless uniform weights beat it beyond one
standard error.
"""
from __future__ import annotations
import time
import numpy as np

from . import data
from .data import DatasetSpec


class Cascade:
    name = "cascade"

    def __init__(self, layer1, prior: dict, tau: float, relay_config: dict | None = None):
        """``prior``: dict(noise, p, calibration) naming the decoder prior's noise model for the device patch."""
        self.layer1, self.prior, self.tau = layer1, dict(prior), float(tau)
        self.relay_config = dict(relay_config or {})
        self.distance, self.basis = layer1.distance, layer1.basis
        self._relays = {}
        self.device = layer1.device

    def relay(self, rounds: int):
        """Relay-BP (with its MWPM fallback) for R rounds, built once from the prior's noisy circuit."""
        from .relay import RelayBP
        if rounds not in self._relays:
            spec = DatasetSpec(distance=self.distance, rounds=rounds, basis=self.basis, noise=self.prior["noise"],
                               p=float(self.prior["p"]), calibration=self.prior["calibration"])
            self._relays[rounds] = RelayBP(data.make_circuits(spec)[1], self.relay_config)
        return self._relays[rounds]

    def to(self, device: str) -> "Cascade":
        self.layer1.to(device)
        self.device = device
        return self

    def decode_detailed(self, det_events, *, raw_meas=None) -> dict:
        from .sequence import rounds_from_width
        x = np.ascontiguousarray(np.asarray(det_events), dtype=np.uint8)
        p = self.layer1.probabilities(x, raw_meas=raw_meas)
        confidence = np.abs(2 * p - 1)
        pred = (p > 0.5)[:, None].copy()
        layer = np.zeros(len(x), dtype=np.int8)
        defer = np.flatnonzero(confidence < self.tau)
        if len(defer):
            relay = self.relay(rounds_from_width(x.shape[1], self.distance))
            out = relay.decode_detailed(x[defer])
            pred[defer] = out["predictions"]
            layer[defer] = 1
            bad = defer[~out["converged"]]
            if len(bad):
                pred[bad] = relay.fallback.decode_batch(x[bad])
                layer[bad] = 2
        return dict(predictions=pred, layer=layer, p=p, confidence=confidence)

    def decode_batch(self, det_events, *, enable_correlations=None, raw_meas=None) -> np.ndarray:
        return self.decode_detailed(det_events, raw_meas=raw_meas)["predictions"]

    def decode_dataset(self, dataset) -> np.ndarray:
        return self.decode_batch(dataset.dets, raw_meas=dataset.raw if self.layer1.uses_raw else None)

    def profile(self, det_events, reps: int = 20) -> dict:
        """Layer 1's host/device split (layer 2 runs on the CPU)."""
        return dict(self.layer1.profile(det_events, reps), note="layer 1 only; deferred shots add Relay-BP on the CPU")

    def parameter_count(self) -> int:
        return self.layer1.parameter_count()

    def metadata(self) -> dict:
        return dict(name=self.name, tau=self.tau, parameters=self.layer1.parameter_count(),
                    architecture=f"cascade: {self.layer1.describe_architecture()} -> Relay-BP -> correlated MWPM", confidence="|2p - 1| of layer 1's P(flip)", prior=self.prior,
                    layer1=self.layer1.metadata(), layer2="Relay-BP (relay-bp) on the prior's DEM",
                    layer3="correlated MWPM on the prior's DEM, for unconverged Relay-BP shots",
                    relay_config=self.relay_config)


def components(layer1, relay_for, datasets) -> list[dict]:
    """Per dataset: layer 1's p, Relay-BP's prediction and convergence, MWPM's prediction, and the truth (for curves)."""
    out = []
    for ds in datasets:
        dets = ds.dets
        p = layer1.probabilities(dets, raw_meas=ds.raw if layer1.uses_raw else None)
        relay = relay_for(ds.spec.rounds)
        started = time.perf_counter()
        r = relay.decode_detailed(dets)
        relay_seconds = time.perf_counter() - started
        mwpm = relay.fallback.decode_batch(dets)
        out.append(dict(rounds=ds.spec.rounds, p=p, relay=r["predictions"], converged=r["converged"], mwpm=mwpm,
                        obs=ds.obs.astype(bool), relay_us_per_shot=1e6 * relay_seconds / max(1, len(dets)),
                        active=dets.any(axis=1)))
    return out


def _decide(c: dict, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Per shot of one dataset's components: whether the cascade at τ fails, and whether the shot is deferred."""
    defer = np.abs(2 * c["p"] - 1) < tau
    pred = (c["p"] > 0.5)[:, None].copy()
    relay_or_mwpm = np.where(c["converged"][:, None], c["relay"], c["mwpm"])
    pred[defer] = relay_or_mwpm[defer]
    return np.any(pred != c["obs"], axis=1), defer


def failure_mask(parts: list[dict], tau: float) -> np.ndarray:
    """Per-shot failures of the cascade at τ, concatenated in the order of ``parts``."""
    return np.concatenate([_decide(c, tau)[0] for c in parts])


def failures_at(parts: list[dict], tau: float) -> tuple[int, int, int, int]:
    """(failures, shots, deferred, unconverged among deferred) of the cascade at τ."""
    fails = shots = deferred = unconv = 0
    for c in parts:
        fail, defer = _decide(c, tau)
        fails += int(fail.sum())
        shots += len(fail)
        deferred += int(defer.sum())
        unconv += int((defer & ~c["converged"]).sum())
    return fails, shots, deferred, unconv


def curve(parts: list[dict], taus=None, *, layer1_us: float | None = None, relay_us: float | None = None,
          mwpm_us: float | None = None) -> list[dict]:
    """τ against deferred fraction, failures, P_L and (if per-shot costs are given) the implied mean latency."""
    taus = np.unique(np.concatenate([np.linspace(0, 1, 101), [1.0 + 1e-9]])) if taus is None else taus
    rows = []
    for tau in taus:
        f, n, d, u = failures_at(parts, float(tau))
        row = dict(tau=float(tau), failures=f, shots=n, p_l=f / n, deferred_fraction=d / n, unconverged=u)
        if layer1_us is not None and relay_us is not None:
            row["mean_latency_us"] = layer1_us + (d / n) * relay_us + (u / n) * (mwpm_us or 0.0)
        rows.append(row)
    return rows


def tune(parts: list[dict]) -> dict:
    """τ by the one-standard-error rule on validation shots, with the argmin τ* beside it.

    τ* has the fewest failures (ties: the smallest τ). τ is the smallest τ ≤ τ* with F(τ) − F(τ*) ≤ sqrt(b + c),
    where b counts the shots only τ fails and c those only τ* fails: F(τ) − F(τ*) = b − c, and sqrt(b + c) is the
    standard error of that paired difference. τ* itself qualifies (b = c = 0).
    """
    rows = curve(parts)
    best = min(rows, key=lambda r: (r["failures"], r["tau"]))
    ref = failure_mask(parts, best["tau"])
    for row in rows:                                   # Ascending τ.
        fail = failure_mask(parts, row["tau"])
        b, c = int((fail & ~ref).sum()), int((ref & ~fail).sum())
        if row["failures"] - best["failures"] <= np.sqrt(b + c):
            break
    return dict(tau=row["tau"], validation=row, argmin=dict(tau=best["tau"], validation=best),
                paired_vs_argmin=dict(only_tau_fails=b, only_argmin_fails=c, se=float(np.sqrt(b + c))),
                rule="one standard error: the smallest tau whose validation failures exceed the argmin tau's by at most "
                     "sqrt(b + c), b and c the shots on which only one of the two fails (the argmin tau is recorded)")


def choose_prior(calibrated_fail: np.ndarray, uniform_fail: np.ndarray) -> dict:
    """Layers 2 and 3's decoder prior from validation shots, by the same one-standard-error test as ``tune``.

    The calibrated prior stays unless correlated MWPM with uniform weights fails fewer shots by more than
    sqrt(b + c), where b counts the shots only the uniform weights fail and c those only the calibrated fail.
    """
    b = int((uniform_fail & ~calibrated_fail).sum())
    c = int((calibrated_fail & ~uniform_fail).sum())
    se = float(np.sqrt(b + c))
    return dict(chosen="uniform" if c - b > se else "calibrated", shots=int(len(calibrated_fail)),
                calibrated_failures=int(calibrated_fail.sum()), uniform_failures=int(uniform_fail.sum()),
                only_uniform_fails=b, only_calibrated_fails=c, se=se,
                rule="calibrated unless uniform-weight correlated MWPM fails fewer validation shots by more than "
                     "sqrt(b + c), b and c the shots on which only one of the two fails")


def calibration(parts: list[dict], bins: int = 10) -> dict:
    """Reliability of layer 1's P(flip) on non-trivial shots: per-bin mean p against the observed flip rate, and ECE."""
    p = np.concatenate([c["p"][c["active"]] for c in parts])
    y = np.concatenate([c["obs"][c["active"], 0] for c in parts]).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            rows.append(dict(lo=float(lo), hi=float(hi), n=int(m.sum()), mean_p=float(p[m].mean()), observed=float(y[m].mean())))
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return dict(bins=rows, ece=float(ece), shots=int(len(p)))
