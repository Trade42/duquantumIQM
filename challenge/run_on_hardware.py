"""Qiskit IQM path: prepare and inspect first, submit only when called to run."""
from __future__ import annotations
import os

try:
    from .surface_code import make_stim_circuit
    from .stim_to_qiskit import stim_to_qiskit
    from .build_emerald_qubit_rotated import rank_placements, validate_layout
    from .internal_helpers import counts_to_measurement_array
    from .extract_syndromes import extract_syndromes
    from .decoding import decode_syndromes
    from .noise_models import NoiseModel, add_noise
except ImportError:
    from surface_code import make_stim_circuit
    from stim_to_qiskit import stim_to_qiskit
    from build_emerald_qubit_rotated import rank_placements, validate_layout
    from internal_helpers import counts_to_measurement_array
    from extract_syndromes import extract_syndromes
    from decoding import decode_syndromes
    from noise_models import NoiseModel, add_noise

SERVER_URL = "https://resonance.iqm.tech"


def connect_backend(token=None, quantum_computer="emerald"):
    from iqm.qiskit_iqm import IQMProvider
    token = token or os.environ.get("IQM_TOKEN")
    if not token:
        raise ValueError("Set IQM_TOKEN or supply token explicitly.")
    return IQMProvider(SERVER_URL, quantum_computer=quantum_computer, token=token).get_backend()


def prepare_qiskit(backend, distance=3, rounds=3, basis="Z", *, calibration=None,
                    grid=None, rows=8, cols=8, noise=None, transversal_gate_set="pauli_h",
                    logical_operations=(),logical_after_round=1,optimize_logical_gates=False):
    """Read-only preparation. Refuse non-native gates or routing overhead.

    Backend's actual index/name conversion is used instead of QB number minus
    one. Supply grid explicitly if the device does not use row-major numbering.
    """
    from qiskit import transpile
    try:
        from .logical_gates import make_logical_algorithm
    except ImportError:
        from logical_gates import make_logical_algorithm
    algorithm = make_logical_algorithm(distance,rounds,logical_operations,basis=basis,
        transversal_gate_set=transversal_gate_set,logical_after_round=logical_after_round,
        optimize=optimize_logical_gates,gate_basis="cz",no_reset=True)
    ideal = algorithm.circuit
    coupling_indices = list(backend.coupling_map.get_edges())
    couplings = [(backend.index_to_qubit_name(a), backend.index_to_qubit_name(b))
                 for a, b in coupling_indices]
    ranked = rank_placements(ideal, calibration=calibration, grid=grid,
                             rows=rows, cols=cols, couplings=couplings)
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
    edges = {frozenset(e) for e in coupling_indices}
    for inst in twoq:
        pair = [compiled.find_bit(q).index for q in inst.qubits]
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
                placement=placement, report=report,logical_algorithm=algorithm.metadata())


def run_qiskit(distance=3, rounds=1, shots=5000, token=None,
                quantum_computer="emerald", no_reset=True, *, basis="Z", backend=None,
                calibration=None, grid=None, rows=8, cols=8, noise=None,
                circuit_compilation_options=None,transversal_gate_set="pauli_h",
                logical_operations=(),logical_after_round=1,optimize_logical_gates=False):
    """Submit one no-reset memory job. Pass IQM compilation options for DD.

    No hardware calls occur on import. Calibration and compiled schedule are
    attached to results so the decoder uses the same parity records.
    """
    if not no_reset:
        raise ValueError("Emerald requires no_reset=True.")
    if shots < 1:
        raise ValueError("shots must be positive")
    backend = backend if backend is not None else connect_backend(token, quantum_computer)
    prepared = prepare_qiskit(backend, distance, rounds, basis, calibration=calibration,
        grid=grid,rows=rows,cols=cols,noise=noise,transversal_gate_set=transversal_gate_set,
        logical_operations=logical_operations,logical_after_round=logical_after_round,
        optimize_logical_gates=optimize_logical_gates)
    options = {} if circuit_compilation_options is None else {
        "circuit_compilation_options": circuit_compilation_options}
    job = backend.run(prepared["circuit"], shots=shots, **options)
    counts = job.result().get_counts()
    raw = counts_to_measurement_array(counts, prepared["stim_circuit"].num_measurements, shots)
    syndromes = extract_syndromes(raw, prepared["stim_circuit"], print_summary=False)
    syndromes.update(stim_circuit=prepared["stim_circuit"], decoder_circuit=prepared["decoder_circuit"],
        job_id=job.job_id(), hardware_report=prepared["report"],rounds=rounds,basis=basis,
        logical_algorithm=prepared["logical_algorithm"])
    return syndromes


def decode_hardware_results(syndromes, circuit=None, *, correlated=True):
    """Return logical probability and 1-sigma SE for one round count.

    Use decoding.fit_ler across round counts for an actual per-round LER.
    The full DecodeResult contains a nonzero Wilson upper bound at zero errors.
    """
    circuit = circuit if circuit is not None else syndromes.get("decoder_circuit")
    if circuit is None:
        raise ValueError("Supply matching noisy decoder circuit or syndromes['decoder_circuit'].")
    result = decode_syndromes(syndromes, circuit, correlated=correlated)
    return result.probability, result.standard_error
