"""Circuit-level Pauli models and explicit assumptions for calibration/stress tests."""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import stim

try:
    from .calibration import CalibrationSnapshot
except ImportError:
    from calibration import CalibrationSnapshot


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
    bias: float = 1.0  # pZ / pX = pZ / pY; fixed total single-qubit error.
    zz_crosstalk: float = 0.0
    coherent_z_angle: float = 0.0  # Pauli-twirled surrogate, not coherent simulation.
    calibration: CalibrationSnapshot | None = None
    coherence: bool = False
    leakage: float = 0.0  # Per participant per two-qubit gate.
    seepage: float = 0.05  # Recovery per active moment, NOT per round.
    leakage_partner_error: float = 0.75
    leakage_spread: float = 0.0
    reset_clears_leakage: bool = True
    legacy_round_idle: bool = False
    idle_detuning_std_rad_per_ns: float = 0.0

    def __post_init__(self):
        for key in ("p1", "p2", "readout", "reset", "idle", "zz_crosstalk", "leakage", "seepage",
                    "leakage_partner_error", "leakage_spread"):
            probability(getattr(self, key), key)
        if not math.isfinite(self.bias) or self.bias <= 0 or not math.isfinite(self.coherent_z_angle):
            raise ValueError("Bias must be positive; angle must be finite.")
        if self.calibration is None and self.coherence:
            raise ValueError("Coherence model requires calibration.")
        if not math.isfinite(self.idle_detuning_std_rad_per_ns) or self.idle_detuning_std_rad_per_ns < 0:
            raise ValueError("Idle detuning standard deviation must be nonnegative finite rad/ns.")
        if self.idle_detuning_std_rad_per_ns and self.calibration is None:
            raise ValueError("Slow idle-detuning noise requires explicit calibration timings.")

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

    def coherence_channel(self, q, duration, mapping):
        c = self.calibration.qubits[mapping[q]]
        # Pauli twirl of amplitude + phase damping: lambda_X/Y=e^-t/T2,
        # lambda_Z=e^-t/T1. Non-unital relaxation to |0> is not represented.
        px = -math.expm1(-duration / c.t1_ns) / 4
        pz = (1 + math.exp(-duration / c.t1_ns) - 2 * math.exp(-duration / c.t2_ns)) / 4
        return px, px, max(0.0, pz)


def noise_suite(p=0.001, calibration=None) -> dict[str, NoiseModel]:
    base = NoiseModel.uniform(p)
    models = {"uniform": base,
              "biased_z": replace(base, name="biased_z", bias=30),
              "readout_heavy": replace(base, name="readout_heavy", readout=min(1, 5*p)),
              "correlated_zz": replace(base, name="correlated_zz", zz_crosstalk=p),
              "coherent_twirl": replace(base, name="coherent_twirl", coherent_z_angle=0.04),
              "leakage": replace(base, name="leakage", leakage=p, seepage=0.02, leakage_spread=0.01),
              "leakage_stress": replace(base, name="leakage_stress", leakage=min(1, 5*p), seepage=0.002,
                                         leakage_spread=0.05)}
    if calibration:
        calibrated = NoiseModel(name="calibrated_pauli", calibration=calibration, idle=0)
        models.update(calibrated_pauli=calibrated,
            calibrated_coherence=replace(calibrated, name="calibrated_coherence", coherence=True),
            calibrated_leakage=replace(calibrated, name="calibrated_leakage", coherence=True,
                                       leakage=p, seepage=0.01, leakage_spread=0.02))
    return models


def add_noise(circuit: stim.Circuit, model: NoiseModel, *, qubit_map=None, dd=None) -> stim.Circuit:
    """Insert static channels; persistent leakage is sampled separately.

    Coherence applies to idle qubits only: calibrated gate/readout errors already
    include active-operation decoherence. Idle duration = longest operation in
    each TICK-delimited moment. Correlated ZZ is residual error on active pairs.
    """
    if circuit != circuit.without_noise():
        raise ValueError("Noise insertion requires an ideal circuit.")
    mapping = qubit_map or {}
    try:
        from .dynamical_decoupling import plan_dd,insert_dd,is_dd_pulse,pulse_error
    except ImportError:
        from dynamical_decoupling import plan_dd,insert_dd,is_dd_pulse,pulse_error
    if any(is_dd_pulse(i) for i in circuit.flattened()):
        raise ValueError("Pass the base ideal circuit and a DD plan; noise cannot be inserted into untimed DD tags alone.")
    dd_plan = plan_dd(circuit,model,dd,qubit_map=mapping)
    circuit = insert_dd(circuit,dd_plan)
    window_by_location = {(w.moment,w.qubit):w for w in dd_plan.windows} if dd_plan else {}
    coords = circuit.get_final_qubit_coordinates()
    allq = sorted(coords)
    if model.calibration:
        if set(allq) - set(mapping):
            raise ValueError("Calibration noise requires a complete sparse-Stim -> QB map.")
        if len(set(mapping[q] for q in allq)) != len(allq):
            raise ValueError("Calibration qubit map must be injective.")
    out, active, duration = stim.Circuit(), set(), 0.0
    awaiting_round,moment_index = True,0
    data_qubits = {q for q, xy in coords.items() if int(xy[0]) % 2 and int(xy[1]) % 2}

    def pauli(q, p):
        if p:
            out.append("PAULI_CHANNEL_1", [q], [p/(model.bias+2), p/(model.bias+2), p*model.bias/(model.bias+2)])

    def finish():
        if duration:
            for q in allq:
                if q not in active and not model.legacy_round_idle:
                    if model.coherence:
                        channel = model.coherence_channel(q, duration, mapping)
                        if sum(channel):
                            out.append("PAULI_CHANNEL_1", [q], channel)
                    else:
                        pauli(q, model.idle)
                    if model.idle_detuning_std_rad_per_ns:
                        window = window_by_location.get((moment_index,q))
                        signed_time = window.signed_time_ns if window else duration
                        angle = model.idle_detuning_std_rad_per_ns*signed_time
                        variance = angle*angle
                        pz = -.5*math.expm1(-variance/2)
                        if pz:
                            out.append("Z_ERROR",[q],pz)
        active.clear()

    for inst in circuit.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "TICK":
            finish()
            duration = 0.0
            out.append(inst)
            moment_index += 1
            continue
        if is_dd_pulse(inst):
            out.append(inst)
            for q in qs:
                pauli(q,pulse_error(dd_plan.config,model,q,mapping))
            # DD occupies an existing idle window. Do not mark the qubit
            # active and accidentally erase its Markov background/T1 noise.
            continue
        if not qs or name == "QUBIT_COORDS":
            out.append(inst)
            continue
        active.update(qs)
        duration = max(duration, model.duration(name))
        if dd_plan:
            duration = dd_plan.moment_durations_ns[moment_index]
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
                    if model.zz_crosstalk:
                        out.append("CORRELATED_ERROR", [stim.target_z(a), stim.target_z(b)], model.zz_crosstalk)
            else:
                for q in qs:
                    pauli(q, model.qubit_rates(q, mapping)[0])
                    if model.coherent_z_angle:
                        out.append("Z_ERROR", [q], math.sin(model.coherent_z_angle / 2)**2)
    finish()
    return out
