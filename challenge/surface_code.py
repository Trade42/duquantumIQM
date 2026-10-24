"""Hook-safe rotated memories: four interaction layers, optional IQM CZ lowering."""
from __future__ import annotations
from numbers import Integral
from typing import Mapping
import stim

try:
    from .internal_helpers import get_qubit_lists, get_meas_order
except ImportError:
    from internal_helpers import get_qubit_lists, get_meas_order

DEFAULT_NOISE = dict(after_clifford_depolarization=0.003,
                     after_reset_flip_probability=0.003,
                     before_measure_flip_probability=0.003,
                     before_round_data_depolarization=0.003)


def remove_ancilla_resets(circuit: stim.Circuit) -> stim.Circuit:
    """Substitute s_j = m_j XOR m_prev(j), including final data boundaries.

    Apply only to an ideal reset-based template. Reset errors must disappear
    with removed resets. Symmetric difference cancels repeated record targets.
    """
    if circuit != circuit.without_noise():
        raise ValueError("Remove resets before inserting noise.")
    out, history, substitutions = stim.Circuit(), {}, []
    for inst in circuit.flattened():
        name, targets = inst.name, inst.targets_copy()
        if name in {"M", "MR", "MX", "MRX", "MY", "MRY"}:
            for t in targets:
                j = len(substitutions)
                refs = {j}
                if t.value in history:
                    refs.add(history[t.value])
                substitutions.append(refs)
                if name.startswith("MR"):
                    history[t.value] = j
            out.append({"MR": "M", "MRX": "MX", "MRY": "MY"}.get(name, name),
                       targets, inst.gate_args_copy())
        elif name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            refs, n = set(), len(substitutions)
            for t in targets:
                if not t.is_measurement_record_target:
                    raise ValueError("Expected measurement-record parity target.")
                refs ^= substitutions[n + t.value]
            out.append(name, [stim.target_rec(j - n) for j in sorted(refs)],
                       inst.gate_args_copy())
        else:
            out.append(inst)
    return out


def lower_to_cz(circuit: stim.Circuit) -> stim.Circuit:
    """Lower CX to H-CZ-H, cancel local H pairs, schedule disjoint gates ASAP.

    Measurement/reset/annotation boundaries preserve record order. Noise must
    be added afterwards. IQM CZ is targeted; native iSWAP is not assumed.
    """
    out, pending, segment = stim.Circuit(), set(), []

    def flush_h(q):
        if q in pending:
            segment.append(("H", [q]))
            pending.remove(q)

    def flush_segment():
        for q in sorted(pending):
            segment.append(("H", [q]))
        pending.clear()
        last, layers = {}, []
        for gate, qs in segment:
            level = max((last.get(q, -1) for q in qs), default=-1) + 1
            while len(layers) <= level:
                layers.append([])
            layers[level].append((gate, qs))
            for q in qs:
                last[q] = level
        for layer in layers:
            for gate, qs in layer:
                out.append(gate, qs)
            out.append("TICK")
        segment.clear()

    for inst in circuit.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "TICK":
            if inst.tag == "logical_step":
                flush_segment()
                out.append(inst)
            continue
        if name == "H":
            pending.symmetric_difference_update(qs)
        elif name == "CX":
            for a, b in zip(qs[::2], qs[1::2]):
                pending.symmetric_difference_update({b})
                flush_h(a)
                flush_h(b)
                segment.append(("CZ", [a, b]))
                pending.symmetric_difference_update({b})
        elif name in {"X", "Y", "Z", "S", "S_DAG", "CZ"}:
            for q in qs:
                flush_h(q)
            size = 2 if name == "CZ" else 1
            segment.extend((name, qs[i:i + size]) for i in range(0, len(qs), size))
        else:
            flush_segment()
            out.append(inst)
            if name in {"R", "RX", "M", "MX", "MR", "MRX"}:
                out.append("TICK")
    flush_segment()
    return out


def make_stim_circuit(distance: int, rounds: int, basis: str = "Z",
                      no_reset: bool = True, noise=None, *,
                      gate_basis: str = "cx", qubit_map: Mapping[int, str] | None = None,
                      transversal_gate_set="pauli_h", logical_operations=(),
                      logical_after_round: int = 1, optimize_logical_gates: bool = False
                      ) -> stim.Circuit:
    """Odd-distance >=3 X/Z memory. None = ideal; dict = legacy Stim rates.

    transversal_gate_set selects none/pauli/pauli_h/css or an explicit subset.
    Logical operations are inserted between QEC rounds; H tracks dual boundaries.
    Optimized Pauli readout metadata is available via make_logical_algorithm.
    NoiseModel supports calibration; CZ lowering precedes noise insertion.
    """
    if isinstance(distance, bool) or not isinstance(distance, Integral) or distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer >= 3")
    if isinstance(rounds, bool) or not isinstance(rounds, Integral) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    basis = basis.upper()
    if basis not in {"X", "Z"}:
        raise ValueError("basis must be X or Z")
    if gate_basis not in {"cx", "cz"}:
        raise ValueError("gate_basis must be cx or cz")
    try:
        from .logical_gates import normalize_gate_set, insert_logical_operations
    except ImportError:
        from logical_gates import normalize_gate_set, insert_logical_operations
    normalize_gate_set(transversal_gate_set)
    circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}",
                                    distance=int(distance), rounds=int(rounds))
    if no_reset:
        circuit = remove_ancilla_resets(circuit)
    if logical_operations:
        circuit, _ = insert_logical_operations(circuit,distance,rounds,logical_operations,
            transversal_gate_set=transversal_gate_set,after_round=logical_after_round,
            optimize=optimize_logical_gates)
    if gate_basis == "cz":
        circuit = lower_to_cz(circuit)
    if noise is not None:
        try:
            from .noise_models import NoiseModel, add_noise
        except ImportError:
            from noise_models import NoiseModel, add_noise
        model = NoiseModel.from_legacy(noise) if isinstance(noise, dict) else noise
        circuit = add_noise(circuit, model, qubit_map=qubit_map)
    return circuit


def get_circuit_info(circuit: stim.Circuit) -> dict:
    data, anc = get_qubit_lists(circuit)
    twoq = sum(len(i.targets_copy()) // 2 for i in circuit.flattened() if i.name in {"CX", "CZ"})
    return dict(num_qubits=len(data) + len(anc), stim_index_capacity=circuit.num_qubits,
                num_data_qubits=len(data), num_anc_qubits=len(anc),
                num_measurements=circuit.num_measurements, num_detectors=circuit.num_detectors,
                num_observables=circuit.num_observables, data_stim_indices=data,
                anc_stim_indices=anc, meas_order=get_meas_order(circuit),
                two_qubit_gates=twoq, ticks=circuit.num_ticks,
                reset_measurements=sum(len(i.targets_copy()) for i in circuit.flattened()
                                       if i.name in {"MR", "MRX"}))


def make_qiskit_circuit(distance: int, rounds: int, basis: str = "Z",
                        no_reset: bool = True, *, gate_basis: str = "cz",
                        transversal_gate_set="pauli_h", logical_operations=(),
                        logical_after_round=1, optimize_logical_gates=False):
    try:
        from .stim_to_qiskit import stim_to_qiskit
    except ImportError:
        from stim_to_qiskit import stim_to_qiskit
    return stim_to_qiskit(make_stim_circuit(distance, rounds, basis, no_reset,
        gate_basis=gate_basis,transversal_gate_set=transversal_gate_set,
        logical_operations=logical_operations,logical_after_round=logical_after_round,
        optimize_logical_gates=optimize_logical_gates))[0]


def make_qrisp_experiment(distance: int, rounds: int, basis: str = "Z",
                         no_reset: bool = True, p: float | None = None, *,
                         transversal_gate_set="pauli_h",logical_operations=(),
                         logical_after_round=1,optimize_logical_gates=False):
    try:
        from .stim_to_qrisp import stim_to_detector_experiment
        from .noise_models import NoiseModel
    except ImportError:
        from stim_to_qrisp import stim_to_detector_experiment
        from noise_models import NoiseModel
    circuit = make_stim_circuit(distance, rounds, basis, no_reset,
        None if p is None else NoiseModel.uniform(p),transversal_gate_set=transversal_gate_set,
        logical_operations=logical_operations,logical_after_round=logical_after_round,
        optimize_logical_gates=optimize_logical_gates)
    return stim_to_detector_experiment(circuit, no_reset=False)
