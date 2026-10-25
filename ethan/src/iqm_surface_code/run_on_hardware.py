"""Explicit Garnet memory preparation and audited final-policy execution."""
from __future__ import annotations
import os
from .surface_code import make_stim_circuit
from .build_emerald_qubit_rotated import Placement, rank_placements, validate_layout
from .decoding import decode_syndromes
from .noise_models import NoiseModel, add_noise
from .stim_to_qiskit import stim_to_qiskit
SERVER_URL = "https://resonance.iqm.tech"

def connect_backend(token=None, quantum_computer="garnet"):
    from iqm.qiskit_iqm import IQMProvider
    token = token or os.environ.get("IQM_TOKEN")
    if not token:
        raise ValueError("Set IQM_TOKEN or supply token explicitly.")
    return IQMProvider(SERVER_URL, quantum_computer=quantum_computer, token=token).get_backend()


def prepare_qiskit(backend, distance=3, rounds=3, basis="Z", *, calibration=None,
                    grid=None, rows=8, cols=8, noise=None, qubit_map=None):
    """Read-only preparation. Refuse non-native gates or routing overhead.

    Backend's actual index/name conversion is used instead of QB number minus
    one. Supply grid explicitly if the device does not use row-major numbering.
    """
    from qiskit import transpile
    ideal = make_stim_circuit(distance, rounds, basis, gate_basis="cz", no_reset=True)
    coupling_indices = list(backend.coupling_map.get_edges())
    couplings = [(backend.index_to_qubit_name(a), backend.index_to_qubit_name(b))
                 for a, b in coupling_indices]
    if qubit_map is None:
        ranked = rank_placements(ideal, calibration=calibration, grid=grid,
                                 rows=rows, cols=cols, couplings=couplings)
    else:
        validate_layout(ideal, qubit_map, couplings)
        if set(qubit_map) != set(ideal.get_final_qubit_coordinates()):
            raise ValueError("Explicit layout must contain exactly the circuit qubits.")
        ranked = [Placement(dict(qubit_map), 0., -1, (0, 0))]
    placement = ranked[0]
    qc, dense, _ = stim_to_qiskit(ideal)
    layout = [backend.qubit_name_to_index(placement.stim_to_physical[q]) for q in sorted(dense)]
    compiled = transpile(qc, backend=backend, initial_layout=layout,
                         routing_method="none", optimization_level=1, seed_transpiler=7)
    if compiled.count_ops().get("reset", 0):
        raise ValueError("Emerald circuit contains unsupported mid-circuit reset.")
    expected = sum(len(i.targets_copy())//2 for i in ideal if i.name == "CZ")
    twoq = [i for i in compiled.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"]
    if len(twoq) > expected or any(i.operation.name != "cz" for i in twoq):
        raise ValueError("Transpilation inserted routing or unexpected two-qubit gates.")
    # IQM 35 may retain virtual registers and serialize through initial_layout.
    # Inspect the actual wire names sent to IQM, rather than assuming that a
    # circuit's bit-list indices are physical indices after every SDK pass.
    if callable(getattr(backend, "serialize_circuit", None)):
        edges = {frozenset(e) for e in couplings}
        serialized = backend.serialize_circuit(compiled)
        for inst in serialized.instructions:
            if inst.name == "cz" and frozenset(inst.locus) not in edges:
                raise ValueError("Serialized interaction is not in live coupling map.")
    else:
        edges = {frozenset(e) for e in coupling_indices}
        initial = compiled.layout.initial_layout if compiled.layout is not None else None
        use_layout = initial is not None and initial.get_registers() == set(compiled.qregs)
        for inst in twoq:
            pair = [initial[q] if use_layout else compiled.find_bit(q).index for q in inst.qubits]
            if frozenset(pair) not in edges:
                raise ValueError("Compiled interaction is not in live coupling map.")
    if compiled.num_clbits != ideal.num_measurements or compiled.count_ops().get("measure", 0) != ideal.num_measurements:
        raise ValueError("Compilation changed measurement record size.")
    model = noise or (NoiseModel(name="calibrated_pauli", calibration=calibration, idle=0)
                      if calibration else NoiseModel.uniform())
    decoder_circuit = add_noise(ideal, model, qubit_map=placement.stim_to_physical)
    report = validate_layout(ideal, placement.stim_to_physical, couplings)
    report.update(depth=compiled.depth(), two_qubit_gate_count=len(twoq),
                  placement_candidates=len(ranked), placement_score=placement.score,
                  noise_schedule="logical CZ/H schedule; native r decomposition and pulse timing may differ")
    return dict(circuit=compiled, stim_circuit=ideal, decoder_circuit=decoder_circuit,
                placement=placement, report=report)


def run_qiskit(distance=3, rounds=1, shots=5000, token=None,
                quantum_computer="garnet", *, basis="Z", backend=None,
                calibration=None, qubit_map=None, artifact_dir=None,
                train_dets=None, train_obs=None, hardware_profile="validated"):
    """Explicitly execute the final physical policy; never run on import.

    Z posterior construction requires independent training records matching the
    circuit's detector frame. A new calibration requires fresh validation.
    """
    if hardware_profile != "validated":
        raise ValueError("Only the final validated Garnet memory profile is supported.")
    if distance != 3 or rounds not in (1, 2, 3) or basis.upper() not in {"X", "Z"}:
        raise ValueError("The retained policy supports distance-three X/Z memory at R1..3 only.")
    if shots < 1:
        raise ValueError("shots must be positive")
    if basis.upper() == "Z" and (train_dets is None or train_obs is None):
        raise ValueError("Z memory requires independent training shots for posterior_10.")
    from .hardware_optimization import run_integrated_qiskit
    backend = backend if backend is not None else connect_backend(token, quantum_computer)
    return run_integrated_qiskit(backend, token=token, distance=distance, rounds=rounds,
        shots=shots, basis=basis.upper(), calibration=calibration, qubit_map=qubit_map,
        artifact_dir=artifact_dir, train_dets=train_dets, train_obs=train_obs)


def run_optimized_memory(*args, **kwargs):
    """Execute the retained Garnet X/Z policy with audited XY4 pulses."""
    kwargs.setdefault("quantum_computer", "garnet")
    kwargs.setdefault("hardware_profile", "validated")
    return run_qiskit(*args, **kwargs)


def decode_hardware_results(syndromes, circuit=None, *, correlated=True):
    """Return logical probability and 1-sigma SE for one round count.

    Use decoding.fit_ler across round counts for an actual per-round LER.
    The full DecodeResult contains a nonzero Wilson upper bound at zero errors.
    """
    circuit = circuit if circuit is not None else syndromes.get("decoder_circuit")
    if circuit is None:
        raise ValueError("Supply matching noisy decoder circuit or syndromes['decoder_circuit'].")
    result = decode_syndromes(syndromes, circuit, correlated=correlated,matcher=syndromes.get("decoder"))
    return result.probability, result.standard_error
