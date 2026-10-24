"""Validated transversal Pauli/H gates and Pauli-frame compilation.

H changes the patch's X/Z boundary orientation. Subsequent extraction and final
readout are conjugated accordingly; no physical patch-rotation SWAPs are hidden.
S/T are not transversal gates of this static rotated patch. A separate CNOT
primitive requires two identical CSS blocks and verified cross-block coupling.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from collections.abc import Mapping
from numbers import Integral
import stim

PROFILES = {
    "none": frozenset(),
    "pauli": frozenset({"X", "Y", "Z"}),
    "pauli_h": frozenset({"X", "Y", "Z", "H"}),
    "css": frozenset({"X", "Y", "Z", "H", "CX"}),
}


def normalize_gate_set(transversal_gate_set="pauli_h"):
    if isinstance(transversal_gate_set, str):
        if transversal_gate_set not in PROFILES:
            raise ValueError(f"Unknown transversal_gate_set; choose {tuple(PROFILES)} or an explicit subset.")
        return PROFILES[transversal_gate_set]
    allowed = frozenset(str(g).upper().replace("CNOT", "CX") for g in transversal_gate_set)
    if allowed - PROFILES["css"]:
        raise ValueError("Static rotated surface code supports Pauli/H profiles and a two-block CX primitive; S/T need a different protocol.")
    return allowed


def normalize_operations(operations, allowed):
    ops = tuple(str(g).upper().replace("CNOT", "CX") for g in operations)
    if set(ops) - allowed:
        raise ValueError(f"Logical operation(s) {sorted(set(ops)-allowed)} not in transversal_gate_set.")
    if "CX" in ops:
        raise ValueError("CX needs two identical CSS blocks; use transversal_cnot, not the single-block memory generator.")
    return ops


@dataclass(frozen=True)
class PauliHPlan:
    """U = X^x Z^z H^h up to global phase, in the evolving logical frame."""
    source_operations: tuple[str, ...]
    hadamard: bool
    pauli_x: bool
    pauli_z: bool

    def physical_operations(self, optimize=True):
        return (("H",) if self.hadamard else ()) if optimize else self.source_operations

    def final_basis(self, input_basis):
        return ({"X":"Z", "Z":"X"}[input_basis] if self.hadamard else input_basis)

    def readout_xor(self, input_basis):
        return int(self.pauli_x if self.final_basis(input_basis) == "Z" else self.pauli_z)


def plan_pauli_h(operations, transversal_gate_set="pauli_h"):
    ops = normalize_operations(operations, normalize_gate_set(transversal_gate_set))
    x = z = h = False
    for gate in ops:
        if gate == "H":
            x, z, h = z, x, not h
        else:
            x ^= gate in {"X", "Y"}
            z ^= gate in {"Z", "Y"}
    return PauliHPlan(ops, h, x, z)


def normalize_program(operations, after_round, transversal_gate_set):
    """A sequence is one block; a mapping schedules separate blocks by round."""
    raw = operations if isinstance(operations, Mapping) else {after_round:operations}
    allowed = normalize_gate_set(transversal_gate_set)
    program = {}
    for round_number,ops in raw.items():
        if isinstance(round_number,bool) or not isinstance(round_number,Integral):
            raise ValueError("Logical schedule keys must be integer round numbers.")
        block = normalize_operations(ops,allowed)
        if block:
            program[int(round_number)] = block
    return dict(sorted(program.items()))


def logical_supports(distance):
    """Extract canonical logical strings from Stim's own observable records."""
    result = {}
    for basis in ("X", "Z"):
        circuit = stim.Circuit.generated(f"surface_code:rotated_memory_{basis.lower()}", distance=distance, rounds=1)
        measured, support = [], set()
        for inst in circuit:
            if inst.name in {"M", "MX", "MR", "MRX"}:
                measured.extend(t.value for t in inst.targets_copy())
            if inst.name == "OBSERVABLE_INCLUDE":
                support ^= {measured[len(measured)+t.value] for t in inst.targets_copy()}
        result[basis] = tuple(sorted(support))
    return result


def insert_logical_operations(circuit, distance, rounds, operations, *,
                               transversal_gate_set="pauli_h", after_round=1,
                               optimize=False):
    """Insert a unitary block between QEC rounds; preserve mandatory QEC rounds.

    Optimized output omits known Paulis, recorded by plan.readout_xor(). It is
    equivalent only with that classical frame applied. make_logical_algorithm
    returns this metadata; bare Stim circuits expose observable *flips* only.
    """
    program = normalize_program(operations,after_round,transversal_gate_set)
    plan = plan_pauli_h([g for block in program.values() for g in block],transversal_gate_set)
    if not plan.source_operations:
        return circuit, plan
    if any(not 1 <= r < rounds for r in program):
        raise ValueError("Logical block requires 1 <= logical_after_round < rounds.")
    if circuit != circuit.without_noise():
        raise ValueError("Insert logical operations before noise.")
    coords = circuit.get_final_qubit_coordinates()
    data = {q for q, xy in coords.items() if int(xy[0])%2 and int(xy[1])%2}
    support = logical_supports(distance)
    out, completed, dual = stim.Circuit(), 0, False

    def emit_block(block):
        nonlocal dual
        out.append("TICK", tag="logical_step")
        for gate in plan_pauli_h(block,transversal_gate_set).physical_operations(optimize):
            if gate == "H":
                out.append("H", sorted(data), tag="logical_gate")
                dual = not dual
            else:
                # After H, logical X/Z retain their Pauli types but swap strings.
                xs = set(support["Z" if dual else "X"]) if gate in {"X", "Y"} else set()
                zs = set(support["X" if dual else "Z"]) if gate in {"Z", "Y"} else set()
                for q in sorted(xs | zs):
                    out.append("Y" if q in xs & zs else "X" if q in xs else "Z", [q], tag="logical_gate")
            out.append("TICK", tag="logical_step")

    for inst in circuit.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if dual and name == "CX":
            # H conjugation on data wires changes the measured check type.
            control_data = [a for a,b in zip(qs[::2],qs[1::2]) if a in data]
            if control_data:
                out.append("H", control_data)
                out.append("TICK")
            for a,b in zip(qs[::2],qs[1::2]):
                out.append("CZ" if b in data else "CX", [a,b])
            if control_data:
                out.append("TICK")
                out.append("H", control_data)
        elif dual and name in {"M", "MX"} and qs and all(q in data for q in qs):
            out.append("MX" if name == "M" else "M", inst.targets_copy(), inst.gate_args_copy())
        else:
            out.append(inst)
        if name in {"M", "MR"} and qs and all(q not in data for q in qs):
            completed += 1
            if completed in program:
                emit_block(program[completed])
    if completed != rounds:
        raise ValueError("Unexpected ancilla measurement structure.")
    return out, plan


def transversal_cnot(control_data, target_data, *, available_pairs):
    """One parallel CX layer between matching CSS data wires.

    Does not add syndrome extraction or an independent-block decoder. Callers
    must supply verified couplings and a joint decoding protocol before running
    a fault-tolerant two-block computation. No automatic long-range routing.
    """
    control, target = list(control_data), list(target_data)
    if not control or len(control) != len(target) or len(set(control+target)) != 2*len(control):
        raise ValueError("Need disjoint equal-sized data blocks with unique indices.")
    edges = {frozenset(pair) for pair in available_pairs}
    pairs = list(zip(control,target))
    if any(frozenset(pair) not in edges for pair in pairs):
        raise ValueError("Missing physical coupling for transversal CNOT; no hidden routing allowed.")
    circuit = stim.Circuit()
    circuit.append("CX", [q for pair in pairs for q in pair])
    return circuit


@dataclass
class LogicalAlgorithm:
    circuit: stim.Circuit
    plan: PauliHPlan
    optimized: bool
    input_basis: str
    rounds: int
    virtual_readout_xor: int
    expected_logical_bit: int
    schedule: dict

    def corrected_logical_bits(self, measurements):
        """Physical output parity XOR tracked Pauli; no reference subtraction."""
        import numpy as np
        raw = np.asarray(measurements,dtype=bool)
        if raw.ndim != 2 or raw.shape[1] != self.circuit.num_measurements:
            raise ValueError("Wrong measurement array dimensions.")
        n, indices = 0, set()
        for inst in self.circuit.flattened():
            if inst.name in {"M","MX","MR","MRX"}:
                n += len(inst.targets_copy())
            elif inst.name == "OBSERVABLE_INCLUDE":
                indices ^= {n+t.value for t in inst.targets_copy()}
        return np.logical_xor.reduce(raw[:,sorted(indices)],axis=1) ^ bool(self.virtual_readout_xor)

    def metadata(self):
        return dict(plan=asdict(self.plan), optimized=self.optimized, rounds=self.rounds,
            input_basis=self.input_basis, output_measurement_basis=self.plan.final_basis(self.input_basis),
            virtual_readout_xor=self.virtual_readout_xor, expected_logical_bit=self.expected_logical_bit,
            boundary_orientation="dual" if self.plan.hadamard else "canonical",schedule=self.schedule)


def make_logical_algorithm(distance, rounds, operations, *, basis="Z",
                            transversal_gate_set="pauli_h", optimize=True,
                            logical_after_round=1, gate_basis="cz", no_reset=True,
                            noise=None, qubit_map=None,dd=None):
    try:
        from .surface_code import make_stim_circuit
    except ImportError:
        from surface_code import make_stim_circuit
    program = normalize_program(operations,logical_after_round,transversal_gate_set)
    plan = plan_pauli_h([g for block in program.values() for g in block],transversal_gate_set)
    circuit = make_stim_circuit(distance,rounds,basis,no_reset,noise,
        gate_basis=gate_basis,qubit_map=qubit_map,transversal_gate_set=transversal_gate_set,
        logical_operations=program,logical_after_round=logical_after_round,
        optimize_logical_gates=optimize,dd=dd)
    basis = basis.upper()
    return LogicalAlgorithm(circuit,plan,optimize,basis,rounds,
        plan.readout_xor(basis) if optimize else 0, plan.readout_xor(basis),program)
