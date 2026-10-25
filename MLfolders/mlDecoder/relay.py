"""Relay-BP on a circuit's detector error model, with correlated MWPM for the shots it does not converge on.

Relay-BP: T. Müller, T. Alexander, M. E. Beverland et al. (IBM), "Improved belief propagation is
sufficient for real-time decoding of quantum memory", arXiv:2506.01779 (2025). It chains memory-BP
runs ("legs") whose per-error memory strengths are drawn from a disordered distribution (negative
values included), relays each leg's marginals into the next, and stops after ``stop_nconv`` legs
have found a valid correction, keeping the most likely one. We run IBM's ``relay-bp`` package (Rust,
Apache-2.0, pinned in requirements-ml.txt), on the same noise model as correlated MWPM: the decoder
prior's noisy circuit, here with hyperedges kept whole (``decompose_errors=False``) since BP does not
need a graph.

The disordered memory strengths are drawn once per decoder from a seeded generator (``gamma_seed``),
so each shot's decision is deterministic and independent of the batch it is decoded in.
``RelayBP.decode_detailed`` reports per shot whether Relay-BP converged. ``decode_batch`` gives every
shot a decision: Relay-BP's where it converged, correlated MWPM's elsewhere (the cascade's safety
net). ``raw=True`` keeps Relay-BP's own unconverged output instead ("Relay-BP alone").
"""
from __future__ import annotations
import time
import numpy as np
import stim

DEFAULTS = dict(gamma0=0.1, pre_iter=80, num_sets=60, set_max_iter=60, gamma_dist_interval=(-0.24, 0.66),
                stop_nconv=5, alpha=None, parallel=False, gamma_seed=20261025)


def dem_matrices(circuit: stim.Circuit):
    """(H: detectors x errors, L: observables x errors, priors) from the circuit's undecomposed DEM.

    Errors with identical detector and observable sets are merged (p = p1 + p2 - 2 p1 p2); errors that
    flip no detector are dropped (no decoder can see them).
    """
    from scipy.sparse import csc_matrix
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    merged = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        if not dets:
            continue
        key = (tuple(sorted(dets)), tuple(sorted(obs)))
        q = merged.get(key, 0.0)
        merged[key] = q + p - 2 * q * p
    keys = list(merged)
    rows, cols, orows, ocols = [], [], [], []
    for j, (dets, obs) in enumerate(keys):
        rows += dets
        cols += [j] * len(dets)
        orows += obs
        ocols += [j] * len(obs)
    H = csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(dem.num_detectors, len(keys)))
    L = csc_matrix((np.ones(len(orows), dtype=np.uint8), (orows, ocols)), shape=(dem.num_observables, len(keys)))
    return H, L, np.array([merged[k] for k in keys], dtype=np.float64)


class RelayBP:
    """Relay-BP observable decoder for one circuit, with a correlated-MWPM fallback."""

    name = "relay_bp"
    device = "cpu"

    def __init__(self, circuit: stim.Circuit, config: dict | None = None):
        import relay_bp
        from .evaluate import Matcher
        self.config = {**DEFAULTS, **(config or {})}
        started = time.perf_counter()
        H, L, priors = dem_matrices(circuit)
        c = self.config
        # The disordered memory strengths, drawn once from a seeded generator: relay-bp would otherwise redraw
        # them from its own advancing RNG, so a shot's decision would depend on the shots decoded before it.
        lo, hi = c["gamma_dist_interval"]
        gammas = np.random.default_rng(int(c["gamma_seed"])).uniform(lo, hi, size=(int(c["num_sets"]), H.shape[1]))
        self.decoder = relay_bp.RelayDecoderF64(H, error_priors=priors, alpha=c["alpha"], gamma0=c["gamma0"],
                                                pre_iter=c["pre_iter"], num_sets=c["num_sets"],
                                                set_max_iter=c["set_max_iter"],
                                                gamma_dist_interval=tuple(c["gamma_dist_interval"]),
                                                explicit_gammas=gammas, stop_nconv=c["stop_nconv"])
        self.runner = relay_bp.ObservableDecoderRunner(self.decoder, L, include_decode_result=True)
        self.num_detectors, self.num_observables, self.num_errors = circuit.num_detectors, circuit.num_observables, H.shape[1]
        self.fallback = Matcher(circuit, True)
        self.build_seconds = time.perf_counter() - started

    def _check(self, det_events):
        x = np.ascontiguousarray(np.asarray(det_events), dtype=np.uint8)
        if x.ndim != 2 or x.shape[1] != self.num_detectors:
            raise ValueError(f"Expected (shots, {self.num_detectors}) detection events, got {x.shape}")
        return x

    def decode_detailed(self, det_events) -> dict:
        """predictions (Relay-BP's own, converged or not), converged (N,) bool, iterations (N,)."""
        x = self._check(det_events)
        pred = np.zeros((len(x), self.num_observables), dtype=bool)
        converged = np.ones(len(x), dtype=bool)
        iterations = np.zeros(len(x), dtype=np.int64)
        active = np.flatnonzero(x.any(axis=1))      # An empty syndrome needs no decoding: no flip.
        if len(active):
            results = self.runner.decode_observables_detailed_batch(x[active], parallel=bool(self.config["parallel"]),
                                                                    progress_bar=False)
            pred[active] = np.array([r.observables for r in results], dtype=bool).reshape(len(active), -1)
            converged[active] = [r.converged for r in results]
            iterations[active] = [r.iterations for r in results]
        return dict(predictions=pred, converged=converged, iterations=iterations)

    def decode_batch(self, det_events, *, enable_correlations=None, raw: bool = False) -> np.ndarray:
        """(N, num_obs) bool: Relay-BP where it converged, correlated MWPM elsewhere (``raw``: Relay-BP's own)."""
        x = self._check(det_events)
        out = self.decode_detailed(x)
        pred = out["predictions"]
        if not raw and not out["converged"].all():
            bad = ~out["converged"]
            pred[bad] = self.fallback.decode_batch(x[bad])
        return pred

    def metadata(self) -> dict:
        import importlib.metadata
        return dict(name=self.name, package=f"relay-bp {importlib.metadata.version('relay-bp')}", config=self.config,
                    errors=self.num_errors, build_seconds=self.build_seconds,
                    dem="decompose_errors=False, approximate_disjoint_errors=True; identical columns merged",
                    fallback="correlated MWPM (challenge.decoding.make_matcher) on unconverged shots")
