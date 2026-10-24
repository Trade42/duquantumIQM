"""Fit a detector error model's probabilities to measured detection events.

Edge probabilities come from pairwise detector correlations (Google Quantum AI,
Nature 2021, Supplementary "p_ij"): for detectors i, j joined by an edge,
  p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) / (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>)),
and boundary edges from the remaining single-detector rate. Each mechanism of the
model's decomposed DEM is then rescaled by the geometric mean of its components'
fitted/model ratios, so hyperedges (Y-type correlations) survive. Fit on training
shots only.
"""
from __future__ import annotations
import numpy as np
import stim


def _components(inst):
    """Graphlike pieces of one decomposed error: tuples of 1 or 2 detector ids."""
    parts, cur = [], []
    for t in inst.targets_copy():
        if t.is_separator():
            parts.append(tuple(sorted(cur))); cur = []
        elif t.is_relative_detector_id():
            cur.append(t.val)
    parts.append(tuple(sorted(cur)))
    return [p for p in parts if p]


def fit_dem(circuit_or_dem, det_events, *, clip_ratio=(0.1, 10.0), p_max=0.4) -> stim.DetectorErrorModel:
    dem = circuit_or_dem if isinstance(circuit_or_dem, stim.DetectorErrorModel) else \
        circuit_or_dem.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    dem = dem.flattened()
    x = np.asarray(det_events, dtype=np.float64)
    m = x.mean(0)
    mm = x.T @ x / len(x)
    model_even = {}  # Edge -> prod(1 - 2p) over model mechanisms containing it.
    for inst in dem:
        if inst.type == "error":
            for c in _components(inst):
                model_even[c] = model_even.get(c, 1.0) * (1 - 2 * inst.args_copy()[0])
    model_p = {c: (1 - v) / 2 for c, v in model_even.items()}
    fitted = {}
    for c in model_p:
        if len(c) == 2:
            i, j = c
            den = 1 - 2 * m[i] - 2 * m[j] + 4 * mm[i, j]
            val = 1 - 4 * (mm[i, j] - m[i] * m[j]) / den if den > 0 else 1.0
            fitted[c] = 0.5 - 0.5 * np.sqrt(min(max(val, 0.0), 1.0))
    for c in model_p:
        if len(c) == 1:
            (i,) = c
            even = np.prod([1 - 2 * fitted[e] for e in fitted if i in e])
            fitted[c] = 0.5 - 0.5 * (1 - 2 * m[i]) / even if even > 0 else model_p[c]
    ratio = {c: float(np.clip(fitted[c] / model_p[c], *clip_ratio)) if model_p[c] > 0 else 1.0 for c in model_p}
    out = stim.DetectorErrorModel()
    for inst in dem:
        if inst.type == "error":
            comps = _components(inst)
            r = float(np.exp(np.mean([np.log(ratio[c]) for c in comps]))) if comps else 1.0
            out.append("error", [min(p_max, inst.args_copy()[0] * r)], inst.targets_copy())
        else:
            out.append(inst)
    return out


def choose_dem(circuit, train_dets, train_obs):
    """Fitted DEM only if it beats the model on held-back training shots; else the model's.

    Fit on the first half of the training split, compare correlated MWPM on the
    second half (paired, same shots), then refit on all training shots if it won.
    Small or saturated samples make p_ij noisy, so the fit is not trusted blindly.
    """
    from .ensemble_decoder import MatchingDecoder
    model = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    half = len(train_dets) // 2
    held_d, held_o = train_dets[half:], train_obs[half:]
    errors = {name: int((MatchingDecoder(d).decode(held_d) != held_o).any(1).sum())
              for name, d in (("model", model), ("fit", fit_dem(model, train_dets[:half])))}
    if errors["fit"] < errors["model"]:
        return fit_dem(model, train_dets), "fit", errors
    return model, "model", errors
