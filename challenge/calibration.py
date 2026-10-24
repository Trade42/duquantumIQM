"""Versioned calibration snapshots. No networking or credentials on import."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class QubitCalibration:
    single_qubit_infidelity: float
    readout_error: float
    reset_error: float
    t1_ns: float
    t2_ns: float
    leakage_per_gate: float = 0.0

    def __post_init__(self):
        for name in ("single_qubit_infidelity", "readout_error", "reset_error", "leakage_per_gate"):
            p = getattr(self, name)
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid {name}: {p}")
        if not (math.isfinite(self.t1_ns) and math.isfinite(self.t2_ns)
                and self.t1_ns > 0 and 0 < self.t2_ns <= 2 * self.t1_ns):
            raise ValueError("Coherence requires 0 < T2 <= 2*T1 (nanoseconds).")


@dataclass
class CalibrationSnapshot:
    qubits: dict[str, QubitCalibration]
    edges: dict[str, float]  # Unordered QB labels joined with '|'; average infidelity.
    durations_ns: dict[str, float]
    metadata: dict = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1 or not self.qubits:
            raise ValueError("Unsupported or empty calibration snapshot.")
        for key, value in self.edges.items():
            if len(key.split("|")) != 2 or any(q not in self.qubits for q in key.split("|")):
                raise ValueError(f"Invalid edge: {key}")
            if not math.isfinite(value) or not 0 <= value <= 0.8:
                raise ValueError(f"Invalid two-qubit infidelity: {key}")
        for gate in ("1q", "2q", "measure", "reset"):
            if gate not in self.durations_ns or not math.isfinite(self.durations_ns[gate]) or self.durations_ns[gate] <= 0:
                raise ValueError(f"Missing/invalid {gate} duration in ns.")

    @staticmethod
    def edge_key(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def edge_error(self, a: str, b: str) -> float:
        return self.edges[self.edge_key(a, b)]

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["qubits"] = {q: QubitCalibration(**v) for q, v in data["qubits"].items()}
        return cls(**data)

    @classmethod
    def from_iqm_metrics(cls, metrics, *, durations_ns, coherence_unit: str,
                         reset_error: float, metadata=None):
        """Normalize CalibrationDataManager maps using explicit units/defaults.

        Metric keys: SQG/1Q, CZ, READOUT, T1, T2 (enum names accepted).
        Pair keys may be tuples or strings containing two QB labels.
        IQM does not supply leakage or gate durations here; never infer them.
        """
        scales = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1}
        if coherence_unit not in scales:
            raise ValueError("Explicit coherence_unit must be s, ms, us or ns.")
        normalized = {}
        for key, values in metrics.items():
            key = str(getattr(key, "name", key)).upper().split(".")[-1]
            normalized[key] = values
        def lookup(*names):
            for name in names:
                if name in normalized:
                    return normalized[name]
            raise ValueError(f"Missing calibration metric: {names}")
        sq, ro, t1, t2 = lookup("SQG", "1Q"), lookup("READOUT"), lookup("T1"), lookup("T2")
        qubits = {}
        for q in sq.keys() & ro.keys() & t1.keys() & t2.keys():
            qubits[q] = QubitCalibration(1 - float(sq[q]), 1 - float(ro[q]), reset_error,
                                        float(t1[q]) * scales[coherence_unit],
                                        float(t2[q]) * scales[coherence_unit])
        edges = {}
        for key, fidelity in lookup("CZ").items():
            pair = list(key) if isinstance(key, tuple) else re.findall(r"QB\d+", str(key))
            if len(pair) != 2:
                raise ValueError(f"Cannot parse CZ edge: {key}")
            if any(q not in qubits for q in pair):
                continue
            edge = cls.edge_key(*pair)
            # Reverse-direction metrics may differ: use the conservative one.
            edges[edge] = max(edges.get(edge, 0), 1 - float(fidelity))
        meta = dict(metadata or {})
        meta.update(coherence_input_unit=coherence_unit, assumed_reset_error=reset_error,
                    missing_leakage="not calibrated; zero unless stress-tested")
        return cls(qubits, edges, dict(durations_ns), meta)


def synthetic_snapshot(rows: int = 8, cols: int = 8, seed: int = 7):
    """Heterogeneous fixture, explicitly NOT a measured Emerald calibration."""
    import numpy as np
    rng = np.random.default_rng(seed)
    qubits = {}
    for i in range(rows * cols):
        t1 = float(rng.uniform(35e3, 70e3))
        qubits[f"QB{i+1}"] = QubitCalibration(float(rng.uniform(0.0002, 0.001)),
            float(rng.uniform(0.005, 0.025)), 0.002, t1, float(rng.uniform(25e3, 50e3)))
    edges = {}
    for r in range(rows):
        for c in range(cols):
            for rr, cc in ((r+1, c), (r, c+1)):
                if rr < rows and cc < cols:
                    a, b = f"QB{r*cols+c+1}", f"QB{rr*cols+cc+1}"
                    edges[CalibrationSnapshot.edge_key(a, b)] = float(rng.uniform(0.002, 0.008))
    return CalibrationSnapshot(qubits, edges, dict(**{"1q": 20, "2q": 80}, measure=700, reset=1000),
        dict(source="SYNTHETIC example; not IQM measurements", seed=seed, rows=rows, cols=cols,
             created_utc=datetime.now(timezone.utc).isoformat()))
