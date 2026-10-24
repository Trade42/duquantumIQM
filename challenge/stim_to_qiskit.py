"""Stim -> Qiskit with exact measurement order, both bases, no-reset hardware."""
import stim
from qiskit import QuantumCircuit


def stim_to_qiskit(stim_circuit: stim.Circuit, *, skip_initial_resets=True,
                    preserve_ticks=True) -> tuple[QuantumCircuit, dict, list]:
    """Strip simulation noise, retain parity metadata on the Stim side.

    Hardware initializes |0>; only untouched initial resets are omitted. RX
    still emits H. Any subsequent reset remains visible for hardware validation.
    """
    flat = stim_circuit.without_noise().flattened()
    qubits = sorted({t.value for i in flat if i.name != "QUBIT_COORDS"
                     for t in i.targets_copy() if t.is_qubit_target})
    mapping = {q: i for i, q in enumerate(qubits)}
    qc = QuantumCircuit(len(qubits), flat.num_measurements)
    touched, order = set(), []
    single = {"H": qc.h, "X": qc.x, "Y": qc.y, "Z": qc.z, "S": qc.s, "S_DAG": qc.sdg}
    for inst in flat:
        name = inst.name
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        if name == "TICK":
            if preserve_ticks:
                qc.barrier()
            continue
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        ds = [mapping[q] for q in qs]
        if name in {"R", "RX"}:
            for q, d in zip(qs, ds):
                if not skip_initial_resets or q in touched:
                    qc.reset(d)
                if name == "RX":
                    qc.h(d)
        elif name in single:
            for d in ds:
                single[name](d)
        elif name in {"CX", "CZ", "SWAP", "ISWAP"}:
            gate = {"CX": qc.cx, "CZ": qc.cz, "SWAP": qc.swap, "ISWAP": qc.iswap}[name]
            for a, b in zip(ds[::2], ds[1::2]):
                gate(a, b)
        elif name in {"M", "MX", "MR", "MRX"}:
            for q, d in zip(qs, ds):
                if name in {"MX", "MRX"}:
                    qc.h(d)
                qc.measure(d, len(order))
                order.append(q)
                if name.startswith("MR"):
                    qc.reset(d)
                    if name == "MRX":
                        qc.h(d)
                elif name == "MX":
                    qc.h(d)  # Preserve postmeasurement state if used again.
        else:
            raise NotImplementedError(f"Unhandled Stim gate: {name}")
        touched.update(qs)
    qc.metadata = {"stim_to_dense": mapping, "meas_order": order}
    return qc, mapping, order
