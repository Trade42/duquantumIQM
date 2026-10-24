"""Empirical lookup-table decoder: syndrome -> majority logical flip seen in training.

Fit on training shots only (simulated or real hardware); no noise model, so on
hardware it absorbs whatever the device really does. Syndromes seen fewer than
min_count times, never seen, or tied go to the fallback decoder, which only
decodes those shots. min_count="cv" picks it from `grid` on held-back training
shots (rarer entries are mostly noise and can lose to the fallback).
"""
from __future__ import annotations
import numpy as np


def _keys(dets):
    packed = np.packbits(np.asarray(dets, bool), axis=1)
    return [row.tobytes() for row in packed]


class LUTDecoder:
    def __init__(self, train_dets, train_obs, fallback, *, min_count="cv", grid=(1, 3, 10, 30)):
        if np.asarray(train_obs).shape[1] != 1:
            raise ValueError("LUT handles a single observable.")
        if min_count == "cv":  # Pick on held-back training shots, never on test shots.
            half = len(train_dets) // 2
            held_d, held_o = np.asarray(train_dets)[half:], np.asarray(train_obs)[half:]
            errors = {mc: int((LUTDecoder(train_dets[:half], train_obs[:half], fallback, min_count=mc)
                               .decode(held_d) != held_o).any(1).sum()) for mc in grid}
            min_count = min(errors, key=lambda mc: (errors[mc], -mc))  # Ties: the more conservative table.
            self.cv_errors = errors
        self.table = {}
        for k, flip in zip(_keys(train_dets), np.asarray(train_obs)[:, 0]):
            n, f = self.table.get(k, (0, 0))
            self.table[k] = (n + 1, f + int(flip))
        self.fallback, self.min_count = fallback, min_count
        self.name = f"LUT + {fallback.name}"
        self.coverage = None

    def decode(self, det_events):
        dets = np.asarray(det_events, bool)
        nf = np.array([self.table.get(k, (0, 0)) for k in _keys(dets)]).reshape(-1, 2)
        n, f = nf[:, 0], nf[:, 1]
        decided = (n >= self.min_count) & (2 * f != n)
        out = 2 * f > n
        if (~decided).any():
            out[~decided] = self.fallback.decode(dets[~decided])[:, 0]
        self.coverage = float(decided.mean())
        return out[:, None]


class CachedLUT:
    """Weighted cache of model posteriors (LILLIPUT-style, Das et al., arXiv:2108.06569).

    Built offline: the `capacity` most frequent syndromes in `build_dets` are decoded
    once by `model` (anything with .probabilities, e.g. TNDecoder) and stored as
    P(flip | syndrome). With `train_obs`, each entry also keeps empirical counts and
    the decision uses (flips + alpha * p_model) / (count + alpha): the model is a
    prior the data can overrule. Misses go to `fallback`, or with online=True to the
    model, and are then cached.
    """

    def __init__(self, build_dets, model, fallback, *, capacity=1 << 16, train_obs=None, alpha=10.0, online=False):
        keys = _keys(build_dets)
        first, counts, flips = {}, {}, {}
        obs = None if train_obs is None else np.asarray(train_obs)[:, 0]
        for i, k in enumerate(keys):
            first.setdefault(k, i)
            counts[k] = counts.get(k, 0) + 1
            if obs is not None:
                flips[k] = flips.get(k, 0) + int(obs[i])
        top = sorted(counts, key=counts.get, reverse=True)[:capacity]
        p = self._posterior(model, np.asarray(build_dets, bool)[[first[k] for k in top]])
        empirical = obs is not None
        self.table = {k: (pk, counts[k] if empirical else 0, flips.get(k, 0)) for k, pk in zip(top, p)}
        self.model, self.fallback, self.alpha, self.online = model, fallback, alpha, online
        self.name = f"cached {getattr(model, 'name', 'model')}" + (" + data" if empirical else "")
        self.hit_rate = None

    @staticmethod
    def _posterior(model, dets):
        if not len(dets):
            return np.zeros(0)
        P = model.probabilities(dets)
        return P[:, 1] / np.maximum(P.sum(1), 1e-300)

    @property
    def memory_bytes(self):
        """Keys plus three numbers per entry; Python dict overhead not counted."""
        return sum(len(k) + 24 for k in self.table)

    def decode(self, det_events):
        dets = np.asarray(det_events, bool)
        keys = _keys(dets)
        hit = np.array([k in self.table for k in keys])
        self.hit_rate = float(hit.mean()) if len(hit) else 0.0
        if self.online and (~hit).any():
            miss_rows = np.flatnonzero(~hit)
            uniq = {}
            for i in miss_rows:
                uniq.setdefault(keys[i], i)
            for k, pk in zip(uniq, self._posterior(self.model, dets[list(uniq.values())])):
                self.table[k] = (pk, 0, 0)
            hit[:] = True
        out = np.zeros(len(dets), bool)
        for i in np.flatnonzero(hit):
            pk, n, f = self.table[keys[i]]
            out[i] = (f + self.alpha * pk) / (n + self.alpha) > 0.5
        if (~hit).any():
            out[~hit] = self.fallback.decode(dets[~hit])[:, 0]
        return out[:, None]
