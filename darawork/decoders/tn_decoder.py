"""Maximum-likelihood decoding by tensor-network contraction of a Stim DEM.

P(L, s) = sum_e prod_j p_j^e_j (1-p_j)^(1-e_j) [H e = s] [A e = L]: one weight
tensor per error mechanism, XOR parity per detector/observable, all entries
nonnegative (no sign cancellation). Detector values only enter as projections,
so the contraction schedule is built once per DEM and reused for every batch.
Optimal for the DEM it is given: a ceiling only when shots come from that DEM.

TNDecoder: time-ordered boundary sweep. The boundary holds the open detector
parities plus L; at d=3 it is 17 bits for any round count (exact, dense).
Larger boundaries (d=5: 49 bits) keep the chi most likely configurations per
shot, the bond-dimension truncation; d=7 (97 bits) exceeds the 63-bit packing.
CotengraTNDecoder: general hyper-optimized contraction, a cross-check only:
log2 width per shot 17 at d=3 r=3, 27 at r=5, 45 at d=5 r=3.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import stim

XOR = np.zeros((2, 2, 2))
for _a in range(2):
    for _b in range(2):
        XOR[_a, _b, _a ^ _b] = 1.0


def mechanisms(dem: stim.DetectorErrorModel, prune: float = 0.0):
    """(p, detectors, observables) per error instruction; drops p < prune."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        ts = inst.targets_copy()
        dets = sorted({t.val for t in ts if t.is_relative_detector_id()})
        obs = sorted({t.val for t in ts if t.is_logical_observable_id()})
        if p >= prune and p > 0 and (dets or obs):
            out.append((p, dets, obs))
    return out


class CotengraTNDecoder:
    name = "TN ML (cotengra)"

    def __init__(self, dem: stim.DetectorErrorModel, *, prune: float = 0.0, batch_size: int = 1024,
                 max_repeats: int = 64, cache_dir: str | Path | None = "results/tn_cache"):
        import cotengra as ctg
        self.num_detectors, self.num_observables = dem.num_detectors, dem.num_observables
        self.batch_size, self.prune = batch_size, prune
        mechs = mechanisms(dem, prune)
        self.num_mechanisms = len(mechs)
        sym = iter(ctg.get_symbol(i) for i in range(10**7))
        inputs, arrays, size = [], [], {}
        err = []
        for p, _, _ in mechs:
            e = next(sym)
            err.append(e)
            inputs.append((e,)); arrays.append(np.array([1 - p, p])); size[e] = 2
        members_d = [[] for _ in range(self.num_detectors)]
        members_o = [[] for _ in range(self.num_observables)]
        for j, (_, dets, obs) in enumerate(mechs):
            for d in dets:
                members_d[d].append(err[j])
            for o in obs:
                members_o[o].append(err[j])

        def chain(vars_):
            a = vars_[0]
            for v in vars_[1:]:
                c = next(sym)
                inputs.append((a, v, c)); arrays.append(XOR); size[c] = 2
                a = c
            return a

        self.batch_index = b = next(sym)
        size[b] = batch_size
        self.syndrome_slots = []  # (array position, detector) for per-batch one-hot vectors.
        for d, vars_ in enumerate(members_d):
            if not vars_:
                continue  # No mechanism (after pruning) reaches it: carries no evidence.
            end = chain(vars_)
            self.syndrome_slots.append((len(arrays), d))
            inputs.append((end, b)); arrays.append(None)
        outputs = []
        for o, vars_ in enumerate(members_o):
            if not vars_:
                raise ValueError(f"No error mechanism flips observable {o}.")
            outputs.append(chain(vars_))
        self.inputs, self.arrays, self.output = inputs, arrays, (*outputs, b)
        opt = ctg.ReusableHyperOptimizer(directory=None if cache_dir is None else str(cache_dir),
                                         methods=["greedy"], max_repeats=max_repeats, minimize="combo",
                                         parallel=False)
        self.tree = ctg.array_contract_tree(inputs, self.output, size_dict=size, optimize=opt)

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kw):
        return cls(circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True), **kw)

    @property
    def width(self):
        """log2 of the largest intermediate tensor, batch index included."""
        return float(self.tree.contraction_width())

    def probabilities(self, det_events) -> np.ndarray:
        """Unnormalized P(L, s) per shot, shape (shots, 2**num_observables)."""
        dets = np.asarray(det_events, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"Expected (shots, {self.num_detectors}) detector array.")
        out, B = [], self.batch_size
        arrays = list(self.arrays)
        for start in range(0, len(dets), B):
            chunk = dets[start:start + B]
            pad = np.zeros((B, self.num_detectors), bool)
            pad[:len(chunk)] = chunk
            for pos, d in self.syndrome_slots:
                s = pad[:, d]
                arrays[pos] = np.stack([~s, s]).astype(float)
            r = self.tree.contract(arrays)  # (2,)*num_observables + (B,)
            out.append(np.moveaxis(r, -1, 0).reshape(B, -1)[:len(chunk)])
        return np.concatenate(out)

    def decode(self, det_events) -> np.ndarray:
        best = self.probabilities(det_events).argmax(1)
        bits = (best[:, None] >> np.arange(self.num_observables)[::-1]) & 1
        return bits.astype(bool)


class TNDecoder:
    name = "TN ML"

    def __init__(self, dem: stim.DetectorErrorModel, *, chi: int | None = None, prune: float = 0.0,
                 batch_size: int = 256, device: str | None = None, max_dense_bits: int = 20):
        import torch
        self.torch = torch
        self.num_detectors, self.num_observables = dem.num_detectors, dem.num_observables
        self.batch_size, self.prune = batch_size, prune
        mechs = sorted(mechanisms(dem, prune), key=lambda m: max(m[1]) if m[1] else -1)
        self.num_mechanisms = len(mechs)
        last = {}
        for i, (_, dets, _) in enumerate(mechs):
            for d in dets:
                last[d] = i
        # Schedule: (mask, p) to apply, then (slot, detector) to close. Built once.
        free, slot, steps, width = [], {}, [], 0
        next_slot = self.num_observables
        for i, (p, dets, obs) in enumerate(mechs):
            for d in dets:
                if d not in slot:
                    slot[d] = free.pop() if free else next_slot
                    next_slot = max(next_slot, slot[d] + 1)
            width = max(width, len(slot))
            mask = sum(1 << slot[d] for d in dets) + sum(1 << o for o in obs)
            closes = [(slot[d], d) for d in dets if last[d] == i]
            for s_, d in closes:
                del slot[d]
                free.append(s_)
            steps.append((mask, p, closes, sorted(slot.items(), key=lambda kv: kv[1])))
        # Beam lookahead: P(remaining mechanisms flip detector d an odd number of times).
        remaining_even = np.ones(self.num_detectors)
        flip_later = [None] * len(mechs)
        for i in range(len(mechs) - 1, -1, -1):
            flip_later[i] = (1 - remaining_even) / 2
            for d in mechs[i][1]:
                remaining_even[d] *= 1 - 2 * mechs[i][0]
        self.steps = [(m, p, c, [(s_, d, flip_later[i][d]) for d, s_ in o]) for i, (m, p, c, o) in enumerate(steps)]
        self.bits = next_slot
        self.frontier = width
        self.dense = chi is None
        if self.dense and self.bits > max_dense_bits:
            raise ValueError(f"Boundary is {self.bits} bits; exact dense sweep capped at {max_dense_bits}. Pass chi.")
        if not self.dense and self.bits > 63:
            raise ValueError(f"Boundary is {self.bits} bits; beam packing supports at most 63.")
        self.chi = chi
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float32 if self.device.type == "mps" else torch.float64

    @classmethod
    def from_circuit(cls, circuit: stim.Circuit, **kw):
        return cls(circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True), **kw)

    def _dense(self, s):
        t, B, N = self.torch, len(s), 1 << self.bits
        ar = t.arange(N, device=self.device)
        S = t.zeros((B, N), dtype=self.dtype, device=self.device)
        S[:, 0] = 1
        for mask, p, closes, _ in self.steps:
            flipped = S.index_select(1, ar ^ mask)
            S.mul_(1 - p).add_(flipped, alpha=p)
            for j, d in closes:  # Project bit j onto s_d, then move it back to 0.
                V = S.view(B, N >> (j + 1), 2, 1 << j)
                keep = t.where(s[:, d].view(B, 1, 1), V[:, :, 1], V[:, :, 0])
                S = t.stack([keep, t.zeros_like(keep)], 2).view(B, N)
                S = S / S.amax(1, keepdim=True).clamp_min(self.torch.finfo(self.dtype).tiny)
        return S[:, :1 << self.num_observables]

    def _beam(self, s):
        t, B = self.torch, len(s)
        C = t.zeros((B, 1), dtype=t.int64, device=self.device)
        W = t.ones((B, 1), dtype=self.dtype, device=self.device)
        tiny = t.finfo(self.dtype).tiny
        for mask, p, closes, open_ in self.steps:
            C2, W2 = t.cat([C, C ^ mask], 1), t.cat([W * (1 - p), W * p], 1)
            Cs, perm = t.sort(C2, 1)
            Ws = t.gather(W2, 1, perm)
            new = t.ones_like(Cs, dtype=t.bool)
            new[:, 1:] = Cs[:, 1:] != Cs[:, :-1]
            gid = t.cumsum(new.long(), 1) - 1
            W = t.zeros_like(Ws).scatter_add_(1, gid, Ws)
            C = t.zeros_like(Cs).scatter_(1, gid, Cs)
            if C.shape[1] > self.chi:
                # Rank by prior x chance the rest of the sweep makes each open bit match s_d.
                score = W.clamp_min(tiny).log()
                if open_:
                    slots = t.tensor([s_ for s_, _, _ in open_], device=self.device)
                    dets = [d for _, d, _ in open_]
                    q = t.tensor([f for _, _, f in open_], dtype=self.dtype, device=self.device).clamp(1e-12, 1 - 1e-12)
                    miss = ((C.unsqueeze(2) >> slots) & 1).bool() != s[:, dets].unsqueeze(1)
                    score = score + t.where(miss, q.log(), (1 - q).log()).sum(2)
                idx = t.topk(score, self.chi, 1).indices
                W, C = t.gather(W, 1, idx), t.gather(C, 1, idx)
            for j, d in closes:
                W = t.where(((C >> j) & 1).bool() == s[:, d:d + 1], W, 0)
                C = C & ~(1 << j)
                W = W / W.amax(1, keepdim=True).clamp_min(self.torch.finfo(self.dtype).tiny)
        L = t.arange(1 << self.num_observables, device=self.device)
        return (W.unsqueeze(2) * (C.unsqueeze(2) == L)).sum(1)

    def probabilities(self, det_events) -> np.ndarray:
        """P(L, s) per shot up to a per-shot scale, shape (shots, 2**num_observables)."""
        dets = np.asarray(det_events, dtype=bool)
        if dets.ndim != 2 or dets.shape[1] != self.num_detectors:
            raise ValueError(f"Expected (shots, {self.num_detectors}) detector array.")
        uniq, inverse = np.unique(dets, axis=0, return_inverse=True)  # Decode each syndrome once.
        run = self._dense if self.dense else self._beam
        # Dense state is batch x 2**bits: keep it near 2**23 elements (64 MB float64), or it swaps.
        B = max(1, min(self.batch_size, (1 << 23) >> self.bits)) if self.dense else self.batch_size
        out = []
        with self.torch.no_grad():
            for start in range(0, len(uniq), B):
                s = self.torch.as_tensor(uniq[start:start + B], device=self.device)
                out.append(run(s).cpu().double().numpy())
        return np.concatenate(out)[inverse.reshape(-1)]

    def decode(self, det_events) -> np.ndarray:
        best = self.probabilities(det_events).argmax(1)
        return ((best[:, None] >> np.arange(self.num_observables)) & 1).astype(bool)
