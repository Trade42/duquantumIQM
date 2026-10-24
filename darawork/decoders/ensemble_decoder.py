"""Correlated MWPM baseline and an ensemble of weight-perturbed matchers.

Ensemble: members decode the same shot with log-normally perturbed error
probabilities (member 0 unperturbed); the majority vote is the prediction.
Idea from Shutty et al., arXiv:2401.12434; perturbation scale is a parameter.
"""
from __future__ import annotations
import numpy as np
import pymatching
import stim


def _dem(circuit_or_dem):
    if isinstance(circuit_or_dem, stim.DetectorErrorModel):
        return circuit_or_dem
    return circuit_or_dem.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


class MatchingDecoder:
    name = "correlated MWPM"

    def __init__(self, circuit_or_dem, *, correlated=True):
        self.correlated = correlated
        self.matcher = pymatching.Matching.from_detector_error_model(_dem(circuit_or_dem), enable_correlations=correlated)
        if not correlated:
            self.name = "MWPM"

    @classmethod
    def from_circuit(cls, circuit, **kw):
        return cls(circuit, **kw)

    def decode(self, det_events):
        return self.matcher.decode_batch(np.asarray(det_events, bool), enable_correlations=self.correlated).astype(bool)


def perturbed(dem: stim.DetectorErrorModel, sigma: float, rng) -> stim.DetectorErrorModel:
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            p = min(0.5 - 1e-9, inst.args_copy()[0] * float(np.exp(sigma * rng.standard_normal())))
            out.append("error", [p], inst.targets_copy())
        else:
            out.append(inst)
    return out


class EnsembleDecoder:
    def __init__(self, circuit_or_dem, *, members=21, sigma=0.5, seed=0, correlated=True):
        if members % 2 == 0:
            raise ValueError("Use an odd number of members so the vote cannot tie.")
        dem, rng = _dem(circuit_or_dem), np.random.default_rng(seed)
        self.members = [MatchingDecoder(dem if i == 0 else perturbed(dem, sigma, rng), correlated=correlated)
                        for i in range(members)]
        self.name = f"MWPM ensemble x{members}"

    @classmethod
    def from_circuit(cls, circuit, **kw):
        return cls(circuit, **kw)

    def decode(self, det_events):
        votes = sum(m.decode(det_events).astype(np.int32) for m in self.members)
        return votes * 2 > len(self.members)
