"""Idle-window DD planning and IQM pulse-compiler configuration.

Local Stim pulses carry timing-plan tags; they are not an untimed hardware
implementation. Real IQM execution uses server-side pulse compilation instead.
Background Markov noise/T1 is never reduced by a fitted DD improvement factor.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, is_dataclass, replace
import hashlib
import math
from numbers import Integral

import stim

PATTERNS = {"xx":"XX", "xy4":"XYXY", "xy8":"XYXYYXYX"}
ANNOTATIONS = {"QUBIT_COORDS","SHIFT_COORDS","DETECTOR","OBSERVABLE_INCLUDE"}


@dataclass(frozen=True)
class DDConfig:
    sequence: str = "xx"
    target: str = "data"
    min_idle_ratio: int | None = None
    pulse_duration_ns: float | None = None
    pulse_error_probability: float | None = None
    max_pulse_error: float | None = None
    skip_leading_wait: bool = True
    skip_trailing_wait: bool = True
    durations_ns: dict | None = None

    def __post_init__(self):
        if self.sequence not in PATTERNS or self.target not in {"data","all"}:
            raise ValueError("DD sequence must be xx/xy4/xy8; target must be data/all.")
        ratio = self.min_idle_ratio
        if ratio is not None and (isinstance(ratio,bool) or not isinstance(ratio,Integral) or ratio < len(self.pattern)):
            raise ValueError("DD min_idle_ratio must be an integer at least the pulse count.")
        if self.pulse_duration_ns is not None and (not math.isfinite(self.pulse_duration_ns) or self.pulse_duration_ns <= 0):
            raise ValueError("DD pulse duration must be positive finite ns.")
        for p in (self.pulse_error_probability,self.max_pulse_error):
            if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
                raise ValueError("DD pulse error probabilities must be in [0,1].")
        if self.durations_ns is not None:
            for key in ("1q","2q","measure","reset"):
                value = self.durations_ns.get(key,0)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("DD timing overrides require positive 1q/2q/measure/reset ns.")

    @property
    def pattern(self):
        return PATTERNS[self.sequence]

    @property
    def ratio(self):
        return self.min_idle_ratio if self.min_idle_ratio is not None else 2*len(self.pattern)+1


def normalize_dd(dd):
    if dd is None or dd is False or dd == "off":
        return None
    if dd is True:
        return DDConfig()
    if isinstance(dd,DDConfig):
        return dd
    if isinstance(dd,str):
        return DDConfig(sequence=dd.lower())
    raise TypeError("dd must be None/off, xx/xy4/xy8, DDConfig or a precomputed DDPlan.")


def is_dd_pulse(inst):
    return inst.tag.startswith("dd:")


def moments(circuit):
    """Yield instructions and their terminating TICK (or None at the end)."""
    group = []
    for inst in circuit.flattened():
        if inst.name == "TICK":
            yield tuple(group),inst
            group = []
        else:
            group.append(inst)
    if group:
        yield tuple(group),None


def operation_key(name):
    return ("2q" if name in {"CX","CZ"} else "measure" if name in {"M","MX","MR","MRX"}
            else "reset" if name in {"R","RX"} else "1q")


def fingerprint(circuit):
    return hashlib.sha256(str(circuit).encode()).hexdigest()


def pulse_error(config,model,q,mapping):
    return (config.pulse_error_probability if config.pulse_error_probability is not None
            else model.qubit_rates(q,mapping)[0])


def target_qubits(circuit,config,*,model=None,qubit_map=None):
    mapping = qubit_map or {}
    result = []
    for q,xy in circuit.get_final_qubit_coordinates().items():
        if config.target == "data" and not (int(xy[0])%2 and int(xy[1])%2):
            continue
        if config.max_pulse_error is not None:
            if model is None:
                raise ValueError("DD fidelity filtering requires a noise/calibration model.")
            if pulse_error(config,model,q,mapping) > config.max_pulse_error:
                continue
        result.append(q)
    return tuple(sorted(result))


@dataclass(frozen=True)
class DDWindow:
    moment: int
    qubit: int
    duration_ns: float
    pulse_duration_ns: float
    pattern: str

    @property
    def centers_ns(self):
        n = len(self.pattern)
        return tuple((k+.5)*self.duration_ns/n for k in range(n))

    @property
    def signed_time_ns(self):
        # Instantaneous pi pulses at centers; finite-pulse effects are a
        # separate imperfection, not an assumed extension of calibrated T2.
        sign,total,previous = 1.,0.,0.
        for center in self.centers_ns:
            total += sign*(center-previous)
            sign,previous = -sign,center
        return total+sign*(self.duration_ns-previous)


@dataclass(frozen=True)
class DDPlan:
    circuit_sha256: str
    config: DDConfig
    windows: tuple[DDWindow,...]
    moment_durations_ns: tuple[float,...]

    def metadata(self):
        return dict(enabled=True,config=asdict(self.config),
            idle_windows=len(self.windows),pulse_count=sum(len(w.pattern) for w in self.windows),
            protected_qubit_time_ns=sum(w.duration_ns for w in self.windows),
            wall_time_extension_ns=0.,timing_model="TICK moments; pi pulses at symmetric centers",
            markov_noise_suppressed=False,leakage_cleared=False,
            pulse_error_model="additive independent Pauli pulse error; full-window background decay retained",
            circuit_sha256=self.circuit_sha256)


def plan_dd(circuit,model,dd,*,qubit_map=None):
    if isinstance(dd,DDPlan):
        if dd.circuit_sha256 != fingerprint(circuit):
            raise ValueError("DD plan does not match the ideal circuit.")
        return dd
    config = normalize_dd(dd)
    if config is None:
        return None
    if circuit != circuit.without_noise() or any(is_dd_pulse(i) for i in circuit.flattened()):
        raise ValueError("DD planning needs an ideal circuit without existing DD pulses.")
    timing = config.durations_ns or (model.calibration.durations_ns if model.calibration else None)
    if timing is None:
        raise ValueError("Simulation DD requires calibration durations or DDConfig(durations_ns=...).")
    pulse_ns = config.pulse_duration_ns or timing["1q"]
    groups = list(moments(circuit))
    active,durations,first,last = [],[],{},{}
    for index,(group,_) in enumerate(groups):
        touched,duration = set(),0.
        for inst in group:
            if inst.name in ANNOTATIONS:
                continue
            qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
            if qs:
                touched.update(qs)
                duration = max(duration,timing[operation_key(inst.name)])
        for q in touched:
            first.setdefault(q,index)
            last[q] = index
        active.append(touched);durations.append(float(duration))
    targets = target_qubits(circuit,config,model=model,qubit_map=qubit_map)
    windows = []
    for index,duration in enumerate(durations):
        if duration < config.ratio*pulse_ns:
            continue
        for q in targets:
            if q in active[index] or q not in first:
                continue
            if config.skip_leading_wait and index < first[q]:
                continue
            if config.skip_trailing_wait and index > last[q]:
                continue
            windows.append(DDWindow(index,q,duration,float(pulse_ns),config.pattern))
    return DDPlan(fingerprint(circuit),config,tuple(windows),tuple(durations))


def insert_dd(circuit,plan):
    """Tag identity pulse sequences without adding time or measurements.

    Timing is in DDPlan, not encoded as Stim TICK duration. Hardware converters
    must not treat these annotations as a physically timed DD implementation.
    """
    if plan is None:
        return circuit
    if fingerprint(circuit) != plan.circuit_sha256:
        raise ValueError("DD plan does not match the ideal circuit.")
    out = stim.Circuit()
    by_moment = {}
    for window in plan.windows:
        by_moment.setdefault(window.moment,[]).append(window)
    for index,(group,tick) in enumerate(moments(circuit)):
        for inst in group:
            out.append(inst)
        for window in by_moment.get(index,[]):
            for k,gate in enumerate(window.pattern):
                out.append(gate,[window.qubit],tag=f"dd:{index}:{k}:{plan.config.sequence}")
        if tick:
            out.append(tick)
    return out


def iqm_compilation_options(dd,*,options=None,target_names=None):
    """Build IQM client 35.x options lazily; preserve unrelated caller fields.

    None leaves caller options untouched. Explicit off clears existing DD.
    Pulse timing is delegated to the server; no untimed gate identities are sent.
    """
    if dd is None:
        return options
    config = normalize_dd(dd)
    try:
        from iqm.iqm_client import CircuitCompilationOptions,DDMode,DDStrategy
    except ImportError as error:
        raise ImportError("IQM DD execution requires the optional hardware SDK: install .[hardware] in its isolated environment.") from error
    strategy = None if config is None else DDStrategy(
        gate_sequences=[(config.ratio,config.pattern,"center")],
        target_qubits=frozenset(target_names) if target_names is not None else None,
        skip_leading_wait=config.skip_leading_wait,skip_trailing_wait=config.skip_trailing_wait,
        merge_contiguous_waits=True)
    changes = dict(dd_mode=DDMode.DISABLED if config is None else DDMode.ENABLED,dd_strategy=strategy)
    if options is None:
        return CircuitCompilationOptions(**changes)
    if not is_dataclass(options):
        raise TypeError("Expected IQM CircuitCompilationOptions dataclass; refusing to discard caller fields.")
    return replace(options,**changes)
