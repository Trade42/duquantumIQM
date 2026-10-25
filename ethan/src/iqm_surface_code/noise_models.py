"""Uniform baseline and calibrated Pauli priors for surface-code memory."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import stim
from .calibration import CalibrationSnapshot

def probability(p, name="probability"):
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError(f"Invalid {name}: {p}")
    return p


@dataclass(frozen=True)
class NoiseModel:
    name: str = "uniform"
    p1: float = 0.001
    p2: float = 0.001
    readout: float = 0.001
    reset: float = 0.001
    idle: float = 0.001
    calibration: CalibrationSnapshot | None = None
    legacy_round_idle: bool = False

    def __post_init__(self):
        for key in ("p1", "p2", "readout", "reset", "idle"):
            probability(getattr(self, key), key)

    @classmethod
    def uniform(cls, p=0.001):
        """Paper Table 7: probability p at gates, prep, readout and idle moments."""
        return cls(p1=p, p2=p, readout=p, reset=p, idle=p)

    @classmethod
    def from_legacy(cls, rates):
        allowed = {"after_clifford_depolarization", "after_reset_flip_probability",
                   "before_measure_flip_probability", "before_round_data_depolarization"}
        if set(rates) - allowed:
            raise ValueError(f"Unknown legacy noise keys: {set(rates) - allowed}")
        return cls(name="legacy", p1=rates.get("after_clifford_depolarization", 0),
                   p2=rates.get("after_clifford_depolarization", 0),
                   readout=rates.get("before_measure_flip_probability", 0),
                   reset=rates.get("after_reset_flip_probability", 0),
                   idle=rates.get("before_round_data_depolarization", 0), legacy_round_idle=True)

    def with_changes(self, **changes):
        return replace(self, **changes)

    def qubit_rates(self, q, mapping):
        if self.calibration is None:
            return self.p1, self.readout, self.reset
        cal = self.calibration.qubits[mapping[q]]
        # Average infidelity r -> nonidentity Pauli probability (D+1)/D * r.
        return probability(1.5 * cal.single_qubit_infidelity), cal.readout_error, cal.reset_error

    def edge_rate(self, a, b, mapping):
        if self.calibration is None:
            return self.p2
        return probability(1.25 * self.calibration.edge_error(mapping[a], mapping[b]))

    def duration(self, name):
        key = ("2q" if name in {"CX", "CZ"} else "measure" if name in {"M", "MX", "MR", "MRX"}
               else "reset" if name in {"R", "RX"} else "1q")
        return self.calibration.durations_ns[key] if self.calibration else 1


def noise_suite(p=0.001, calibration=None):
    models = {"uniform": NoiseModel.uniform(p)}
    if calibration is not None:
        models["calibrated_pauli"] = NoiseModel(name="calibrated_pauli", calibration=calibration, idle=0)
    return models

def add_noise(circuit: stim.Circuit, model: NoiseModel, *, qubit_map=None) -> stim.Circuit:
    """Insert the baseline or calibrated gate/readout Pauli channels."""
    if circuit != circuit.without_noise():
        raise ValueError("Noise insertion requires an ideal circuit.")
    mapping = qubit_map or {}
    coords = circuit.get_final_qubit_coordinates()
    allq = sorted(coords)
    if model.calibration:
        if set(allq) - set(mapping):
            raise ValueError("Calibration noise requires a complete sparse-Stim -> QB map.")
        if len(set(mapping[q] for q in allq)) != len(allq):
            raise ValueError("Calibration qubit map must be injective.")
    out, active, duration = stim.Circuit(), set(), 0.0
    awaiting_round = True
    data_qubits = {q for q, xy in coords.items() if int(xy[0]) % 2 and int(xy[1]) % 2}

    def pauli(q, p):
        if p:
            out.append("PAULI_CHANNEL_1", [q], [p/3, p/3, p/3])

    def finish():
        if duration:
            for q in allq:
                if q not in active and not model.legacy_round_idle:
                    pauli(q, model.idle)
        active.clear()

    for inst in circuit.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "TICK":
            finish()
            duration = 0.0
            out.append(inst)
            continue
        if not qs or name == "QUBIT_COORDS":
            out.append(inst)
            continue
        active.update(qs)
        duration = max(duration, model.duration(name))
        if model.legacy_round_idle and awaiting_round and name in {"CX", "CZ"}:
            # Uniform depolarization commutes with the preceding data H gates.
            # First entangler identifies a round in both compiled gate bases.
            for q in sorted(data_qubits):
                pauli(q, model.idle)
            awaiting_round = False
        if name in {"M", "MX", "MR", "MRX"}:
            # Measurement argument changes the reported result, not the state.
            for q in qs:
                _, ro, rst = model.qubit_rates(q, mapping)
                out.append(name, [q], ro)
                if name.startswith("MR") and rst:
                    out.append("Z_ERROR" if name == "MRX" else "X_ERROR", [q], rst)
            if any(q not in data_qubits for q in qs):
                awaiting_round = True
        else:
            out.append(inst)
            if name in {"R", "RX"}:
                for q in qs:
                    rst = model.qubit_rates(q, mapping)[2]
                    if rst:
                        out.append("Z_ERROR" if name == "RX" else "X_ERROR", [q], rst)
            elif name in {"CX", "CZ"}:
                for a, b in zip(qs[::2], qs[1::2]):
                    p = model.edge_rate(a, b, mapping)
                    if p:
                        out.append("DEPOLARIZE2", [a, b], p)
            else:
                for q in qs:
                    pauli(q, model.qubit_rates(q, mapping)[0])
    finish()
    return out
