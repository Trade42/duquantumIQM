"""Correlated minimum-weight perfect matching for Garnet memory."""
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
