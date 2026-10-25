"""Reference backend: the team's PyMatching (``challenge.decoding.make_matcher``).

``python -m mlDecoder.run --backend mwpm`` runs no learned decoder: it builds the
frozen test sets and the baseline table (correlated and uncorrelated MWPM). This
module puts the same matcher in the backend shape, so it can be loaded like any
other backend.
"""
from __future__ import annotations
import json
import stim

from ..data import circuit_sha256
from ..evaluate import Matcher


def build(circuit: stim.Circuit, config: dict) -> "MWPMDecoder":
    return MWPMDecoder(circuit, bool(config.get("correlated", True)))


class MWPMDecoder(Matcher):
    """decode_batch follows the build's correlation setting; ``enable_correlations`` is checked against it."""

    def __init__(self, circuit: stim.Circuit, correlated: bool = True):
        super().__init__(circuit, correlated)
        self.circuit_sha256 = circuit_sha256(circuit)

    def fit(self, train_spec_or_data, val, *, device: str = "cpu", seed: int = 0, log=print) -> dict:
        return dict(trained=False, build_seconds=self.build_seconds, device="cpu")

    def decode_batch(self, det_events, *, enable_correlations=None):
        if enable_correlations is not None and bool(enable_correlations) != self.correlated:
            raise ValueError(f"Matcher built with correlated={self.correlated}; got enable_correlations={enable_correlations}")
        return super().decode_batch(det_events)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(dict(correlated=self.correlated, circuit_sha256=self.circuit_sha256), stream)

    @classmethod
    def load(cls, path, circuit: stim.Circuit, device: str = "cpu") -> "MWPMDecoder":
        with open(path, encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved["circuit_sha256"] != circuit_sha256(circuit):
            raise ValueError(f"{path}: saved for a different circuit")
        return cls(circuit, saved["correlated"])

    def metadata(self) -> dict:
        import pymatching
        return dict(name=self.name, architecture="PyMatching 2 sparse blossom" + (", correlated" if self.correlated else ""),
                    pymatching=pymatching.__version__, build_seconds=self.build_seconds, threads=1,
                    dem="decompose_errors=True, approximate_disjoint_errors=True")
