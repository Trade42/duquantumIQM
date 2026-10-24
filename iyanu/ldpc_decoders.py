"""Union-Find, BP+OSD and BP+Union-Find (ldpc 2.x) built from a Stim detector error model.

The DEM is not decomposed: belief propagation and these decoders handle hyperedges, so Y and
correlated errors keep their true detector sets. H is detectors x error mechanisms, L is
observables x error mechanisms, priors are the DEM probabilities. A shot's prediction is
L @ e mod 2 for the decoded error vector e. Every shot gets a prediction (OSD always returns
a solution consistent with the syndrome; UF likewise), so nothing is dropped.
"""
from __future__ import annotations
import numpy as np


def dem_matrices(circuit):
    import scipy.sparse as sp
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    rows, cols, lrows, lcols, priors = [], [], [], [], []
    for j, inst in enumerate(i for i in dem.flattened() if i.type == "error"):
        priors.append(inst.args_copy()[0])
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                rows.append(t.val); cols.append(j)
            elif t.is_logical_observable_id():
                lrows.append(t.val); lcols.append(j)
    n = len(priors)
    H = sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(dem.num_detectors, n))
    L = sp.csr_matrix((np.ones(len(lrows), np.uint8), (lrows, lcols)), shape=(dem.num_observables, n))
    return H, L, np.asarray(priors)


class LdpcDecoder:
    """`decode_batch(det_events) -> (shots, observables)` bool for kind in uf, bposd, belief_find."""

    def __init__(self, circuit, kind="bposd", *, max_iter=30, osd_order=7):
        import ldpc
        self.H, self.L, self.priors = dem_matrices(circuit)
        self.kind = kind
        if kind == "bposd":
            self.dec = ldpc.BpOsdDecoder(self.H, error_channel=list(self.priors), max_iter=max_iter,
                                         bp_method="minimum_sum", osd_method="osd_cs", osd_order=osd_order)
        elif kind == "belief_find":
            self.dec = ldpc.BeliefFindDecoder(self.H, error_channel=list(self.priors), max_iter=max_iter,
                                              bp_method="minimum_sum", uf_method="inversion")
        elif kind == "uf":
            self.dec = ldpc.UnionFindDecoder(self.H, uf_method="inversion")
        else:
            raise ValueError(f"unknown kind {kind}")
        self.config = dict(kind=kind, max_iter=max_iter, osd_order=osd_order, errors=len(self.priors))

    def decode_batch(self, det_events, enable_correlations=None):
        dets = np.asarray(det_events, dtype=np.uint8)
        if dets.ndim != 2 or dets.shape[1] != self.H.shape[0]:
            raise ValueError(f"Expected 2D detection events with {self.H.shape[0]} columns, got {dets.shape}.")
        # Identical syndromes decode identically: decode each distinct one once.
        uniq, inv = np.unique(dets, axis=0, return_inverse=True)
        sols = np.array([self.dec.decode(s) for s in uniq], dtype=np.uint8)
        obs = (self.L @ sols.T).T % 2
        return obs[inv.ravel()].astype(bool)
