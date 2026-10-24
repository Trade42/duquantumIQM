"""Optional exact non-Pauli trajectory simulations (small patches, costly).

Retains coherent Z/ZZ rotations and non-unital T1/T2 relaxation. Does not model
qutrit leakage. Decode with a separately identified Pauli DEM of identical
parity structure; this is a deliberate decoder-model mismatch experiment.
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error, depolarizing_error, thermal_relaxation_error

from .internal_helpers import counts_to_measurement_array
from .extract_syndromes import extract_syndromes


def simulate_nonpauli(ideal, model, *, shots=1000, seed=7, qubit_map=None,
                       coherent_z_angle=0., coherent_zz_angle=0., method="matrix_product_state",
                       dd=None,idle_detuning_rad_per_ns=0.):
    """Execute exact Kraus/noisy trajectories, with duration-aware idle channels.

    Pass an ideal circuit. Calibration average infidelity becomes Pauli channel
    on active gates; non-unital thermal relaxation acts only during idles to
    avoid double counting active gate decoherence. Readout flips are classical.
    """
    if ideal != ideal.without_noise():
        raise ValueError("Expected ideal circuit; do not add static noise twice.")
    if any(i.tag.startswith("dd:") for i in ideal.flattened()):
        raise ValueError("Pass the base ideal circuit and dd=...; DD tags alone do not encode timing.")
    if model.idle_detuning_std_rad_per_ns:
        raise ValueError("Exact idle DD path uses deterministic idle_detuning_rad_per_ns; Gaussian window noise is in the Stim model.")
    if model.leakage:
        raise ValueError("Use leakage.sample_leakage for persistent leakage.")
    if shots < 1:
        raise ValueError("shots must be positive")
    qubits = sorted(ideal.get_final_qubit_coordinates())
    dense = {q:i for i,q in enumerate(qubits)}
    mapping = qubit_map or {}
    from .dynamical_decoupling import plan_dd,pulse_error
    dd_plan = plan_dd(ideal,model,dd,qubit_map=mapping)
    windows = {(w.moment,w.qubit):w for w in dd_plan.windows} if dd_plan else {}
    qc = QuantumCircuit(len(qubits), ideal.num_measurements)
    active, duration, mi, readout_rates,moment_index = set(), 0., 0, [],0
    def channel(error, qs):
        qc.append(error.to_instruction(), [dense[q] for q in qs])
    def finish():
        if duration:
            for q in qubits:
                if q in active:
                    continue
                window = windows.get((moment_index,q))
                def evolve(dt):
                    if model.calibration and model.coherence:
                        cal = model.calibration.qubits[mapping[q]]
                        channel(thermal_relaxation_error(cal.t1_ns,cal.t2_ns,dt),[q])
                    if idle_detuning_rad_per_ns:
                        qc.rz(idle_detuning_rad_per_ns*dt,dense[q])
                previous = 0.
                if window:
                    for center,gate in zip(window.centers_ns,window.pattern):
                        evolve(center-previous)
                        (qc.x if gate == "X" else qc.y)(dense[q])
                        p = pulse_error(dd_plan.config,model,q,mapping)
                        if p:
                            channel(pauli_error([("X",p/(model.bias+2)),("Y",p/(model.bias+2)),
                                ("Z",p*model.bias/(model.bias+2)),("I",1-p)]),[q])
                        previous = center
                evolve(duration-previous)
                if not (model.calibration and model.coherence) and model.idle:
                    p = model.idle
                    channel(pauli_error([("X",p/(model.bias+2)),("Y",p/(model.bias+2)),
                        ("Z",p*model.bias/(model.bias+2)),("I",1-p)]), [q])
        active.clear()
    for inst in ideal.flattened():
        name = inst.name
        qs = [t.value for t in inst.targets_copy() if t.is_qubit_target]
        if name == "TICK":
            finish()
            duration = 0.
            moment_index += 1
            continue
        if name in {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"}:
            continue
        active.update(qs)
        duration = max(duration, model.duration(name))
        if dd_plan:
            duration = dd_plan.moment_durations_ns[moment_index]
        if name in {"R", "RX"}:
            for q in qs:
                qc.reset(dense[q])
                if name == "RX":
                    qc.h(dense[q])
                p = model.qubit_rates(q, mapping)[2]
                if p:
                    channel(pauli_error([("Z" if name == "RX" else "X",p),("I",1-p)]), [q])
        elif name in {"M", "MX", "MR", "MRX"}:
            for q in qs:
                is_x = name in {"MX", "MRX"}
                if is_x:
                    qc.h(dense[q])
                qc.measure(dense[q], mi)
                readout_rates.append(model.qubit_rates(q, mapping)[1])
                mi += 1
                if name.startswith("MR"):
                    qc.reset(dense[q])
                    if is_x:
                        qc.h(dense[q])
                    p = model.qubit_rates(q, mapping)[2]
                    if p:
                        channel(pauli_error([("Z" if is_x else "X",p),("I",1-p)]), [q])
                elif is_x:
                    qc.h(dense[q])
        elif name in {"CX", "CZ"}:
            for a, b in zip(qs[::2], qs[1::2]):
                (qc.cx if name == "CX" else qc.cz)(dense[a], dense[b])
                p = model.edge_rate(a, b, mapping)
                if p:
                    channel(depolarizing_error(16*p/15, 2), [a,b])
                if model.zz_crosstalk:
                    pzz = model.zz_crosstalk
                    channel(pauli_error([("ZZ",pzz),("II",1-pzz)]), [a,b])
                if coherent_zz_angle:
                    qc.rzz(coherent_zz_angle, dense[a], dense[b])
        elif name in {"H", "X", "Y", "Z", "S", "S_DAG"}:
            gate = {"H":qc.h,"X":qc.x,"Y":qc.y,"Z":qc.z,"S":qc.s,"S_DAG":qc.sdg}[name]
            for q in qs:
                gate(dense[q])
                p = model.qubit_rates(q, mapping)[0]
                if p:
                    channel(pauli_error([("X",p/(model.bias+2)),("Y",p/(model.bias+2)),
                        ("Z",p*model.bias/(model.bias+2)),("I",1-p)]), [q])
                angle = coherent_z_angle or model.coherent_z_angle
                if angle:
                    qc.rz(angle, dense[q])
        else:
            raise NotImplementedError(name)
    finish()
    # Run exact channels; sampling budget controls trajectory cost.
    backend = AerSimulator(method=method, seed_simulator=seed, max_parallel_threads=1)
    counts = backend.run(qc, shots=shots).result().get_counts()
    raw = counts_to_measurement_array(counts, ideal.num_measurements, shots)
    rng = np.random.default_rng(seed+1)
    raw ^= rng.random(raw.shape) < np.array(readout_rates)
    result = extract_syndromes(raw, ideal, print_summary=False)
    result.update(stim_circuit=ideal, simulator="Aer exact Kraus/coherent trajectories",
        method=method, seed=seed, coherent_z_angle=coherent_z_angle or model.coherent_z_angle,
        coherent_zz_angle=coherent_zz_angle,idle_detuning_rad_per_ns=idle_detuning_rad_per_ns,
        dd=dd_plan.metadata() if dd_plan else dict(enabled=False))
    return result
