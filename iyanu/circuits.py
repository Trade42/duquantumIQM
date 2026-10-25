"""Hardware circuit variants for the d=3 no-reset memory, all verified on the serialized request.

  ours      repo circuit (make_logical_algorithm, CZ basis, custom H cancellation/ASAP),
            transpiled at optimization level 1 with Qiskit's default scheduling stage
            (what challenge.run_on_hardware.prepare_qiskit does, with the iyanu backend).
  ours-iqm  the same circuit, transpiled with IQM's own scheduling stage (iqm_default_scheduling).
  vanilla   the canonical Stim CX schedule with each CX converted literally to H-CZ-H, optimization
            level 0, IQM scheduling: copied from ethan/src/iqm_surface_code/benchmark_vanilla_lut.py
            (vanilla_circuit / prepare_vanilla), the team's lookup-table baseline circuit.

IQM's scheduling stage rebuilds the circuit with its qubits reordered, so the compiled circuit's
bit indices are not physical indices and prepare_qiskit's coupler check cannot be used on it.
Correctness is instead checked where it matters, on the serialized run request
(iyanu.qpu.check_request): every CZ on a coupler and equal, as a multiset, to the placement.
"""
from __future__ import annotations
import stim

VARIANTS = ("ours", "ours-iqm", "vanilla")


def vanilla_circuit(rounds, basis, distance=3):
    """Canonical Stim schedule with literal H-CZ-H conversion of each CX layer (from ethan/)."""
    from challenge.surface_code import make_stim_circuit
    original = make_stim_circuit(distance, rounds, basis, no_reset=True, gate_basis="cx",
                                 transversal_gate_set="none", logical_operations=())
    out = stim.Circuit()
    for inst in original.flattened():
        if inst.name == "CX":
            targets = inst.targets_copy()
            out.append("H", targets[1::2])
            out.append("TICK")
            out.append("CZ", targets)
            out.append("TICK")
            out.append("H", targets[1::2])
        else:
            out.append(inst)
    return out


def ideal_circuit(variant, distance, rounds, basis):
    if variant == "vanilla":
        return vanilla_circuit(rounds, basis, distance)
    from challenge.logical_gates import make_logical_algorithm
    return make_logical_algorithm(distance, rounds, (), basis=basis, gate_basis="cz", no_reset=True).circuit


def prepare(backend, variant, distance, rounds, basis, *, calibration, grid):
    """Place, compile and model one circuit; same dict shape as prepare_qiskit plus rounds/basis/variant."""
    from qiskit import transpile
    from challenge.build_emerald_qubit_rotated import rank_placements, validate_layout
    from challenge.stim_to_qiskit import stim_to_qiskit
    from challenge.noise_models import NoiseModel, add_noise
    from iyanu import qpu
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    ideal = ideal_circuit(variant, distance, rounds, basis)
    couplings = qpu.couplings(backend)
    ranked = rank_placements(ideal, calibration=calibration, grid=grid, rows=len(grid), cols=len(grid[0]),
                             couplings=couplings)
    placement = ranked[0]
    qc, dense, _ = stim_to_qiskit(ideal)
    layout = [backend.qubit_name_to_index(placement.stim_to_physical[q]) for q in sorted(dense)]
    level = 1 if variant == "ours" else 0
    scheduling = "default" if variant == "ours" else "iqm_default_scheduling"
    compiled = transpile(qc, backend=backend, initial_layout=layout, routing_method="none",
                         optimization_level=level, scheduling_method=scheduling, seed_transpiler=7)
    ops = compiled.count_ops()
    if ops.get("reset", 0):
        raise qpu.QPUGuardError("Compiled circuit contains a mid-circuit reset.")
    if ops.get("measure", 0) != ideal.num_measurements:
        raise qpu.QPUGuardError("Compilation changed the measurement record size.")
    if any(i.operation.num_qubits == 2 and i.operation.name not in {"cz", "barrier"} for i in compiled.data):
        raise qpu.QPUGuardError("Unexpected two-qubit gate after compilation.")
    report = validate_layout(ideal, placement.stim_to_physical, couplings)
    report.update(depth=compiled.depth(), placement_candidates=len(ranked), placement_score=placement.score,
                  optimization_level=level, scheduling=scheduling, variant=variant)
    model = NoiseModel(name="calibrated_pauli", calibration=calibration, idle=0)
    return dict(circuit=compiled, stim_circuit=ideal, placement=placement, report=report,
                decoder_circuit=add_noise(ideal, model, qubit_map=placement.stim_to_physical),
                rounds=rounds, basis=basis, variant=variant)
