"""A compensated E1 twin: the measured local CZ phases cancelled by phase tracking, at circuit level.

Of the table's four errors, only phi (the residual local Z phase per CZ) can be undone without
pulse access. A Z rotation RZ(-phi_q) after a CZ on q commutes with every later CZ and with the
Z readout, and moves through a later prx as

    (RZ(a), then prx(theta, psi)) = (prx(theta, psi - a), then RZ(a)),

so cancelling phi_q after each CZ is the same as adding the running sum of phi_q to the phase of
every later prx on q. No gate is added and no pulse changes; only phases do.

The others cannot be compensated at circuit level:
* eps: a prx at an angle other than pi/2 is two pulses with ``drag_crf_sx``, so changing the
  angle doubles the pulse count (Track B's ``amplitude_i`` is the fix);
* eta (an axis tilt) and delta (a two-qubit phase) are not virtual Z rotations: Track B's
  ``rz_before``/``rz_after`` and ``coupler.amplitude``.

Validation (:func:`detection_check`): the compensated circuit, run with the errors it compensates
injected after every CZ, gives zero detection events on every shot, as the ideal circuit does,
while the uncompensated one does not. That is noiseless equivalence in the presence of the errors.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RGate, RZGate

from dqiqm.experiments import device as dev

#: phis[(a, b) sorted pair][q] = phi_q per CZ on that coupler (rad)
Phis = Mapping[tuple[str, str], Mapping[str, float]]


def compensate(native: QuantumCircuit, names: Mapping[int, str], phis: Phis) -> QuantumCircuit:
    """``native`` (an E1 native circuit: prx, cz, measure, barrier) with every prx phase advanced by
    the running sum of its qubit's local CZ phases. ``names`` maps logical qubit -> device qubit."""
    frame = [0.0] * native.num_qubits
    out = QuantumCircuit(*native.qregs, *native.cregs, name=f"{native.name}_comp")
    for ci in native.data:
        idx = [native.find_bit(q).index for q in ci.qubits]
        qs = [out.qubits[i] for i in idx]
        cs = [out.clbits[native.find_bit(c).index] for c in ci.clbits]
        op = ci.operation
        if op.name == "r":
            theta, psi = (float(x) for x in op.params)
            out.append(RGate(theta, psi + frame[idx[0]]), qs)
            continue
        out.append(op, qs, cs)
        if op.name == "cz":
            pair = dev.pair_key(names[idx[0]], names[idx[1]])
            for i in idx:
                frame[i] += float(phis.get(pair, {}).get(names[i], 0.0))
        elif op.name not in ("measure", "barrier"):
            raise ValueError(f"{op.name} is not expected in an E1 native circuit")
    return out


def with_cz_phases(native: QuantumCircuit, names: Mapping[int, str], phis: Phis) -> QuantumCircuit:
    """``native`` with the device's local phases injected: RZ(phi_q) on both qubits after every CZ."""
    out = QuantumCircuit(*native.qregs, *native.cregs, name=f"{native.name}_phi")
    for ci in native.data:
        idx = [native.find_bit(q).index for q in ci.qubits]
        qs = [out.qubits[i] for i in idx]
        out.append(ci.operation, qs, [out.clbits[native.find_bit(c).index] for c in ci.clbits])
        if ci.operation.name == "cz":
            pair = dev.pair_key(names[idx[0]], names[idx[1]])
            for i, q in zip(idx, qs):
                out.append(RZGate(float(phis.get(pair, {}).get(names[i], 0.0))), [q])
    return out


def detection_check(spec: Any, phis_true: Phis, phis_used: Phis, *, shots: int = 400, seed: int = 2026) -> dict[str, Any]:
    """Noise-free Aer runs of the E1 spec's native circuit with ``phis_true`` injected: plain, and
    compensated with ``phis_used``. Returns each one's detection-event fraction and observable flips."""
    from qiskit_aer import AerSimulator

    from dqiqm import decode as D

    native = spec.hardware.logical
    names = {i: spec.physical[s] for i, s in enumerate(spec.hardware.stim_qubits)}
    sim = AerSimulator(method="statevector", seed_simulator=seed)
    out = {}
    for label, circ in (("ideal", native), ("plain", with_cz_phases(native, names, phis_true)),
                        ("compensated", with_cz_phases(compensate(native, names, phis_used), names, phis_true))):
        mem = sim.run(circ, shots=shots, memory=True).result().get_memory()
        rec = np.array([[int(b) for b in m.replace(" ", "")[::-1]] for m in mem], dtype=np.uint8)
        dets, obs = D.detection_events(spec.stim, rec)
        out[label] = {"shots": shots, "detection_fraction": float(dets.mean()),
                      "shots_with_events": int(dets.any(axis=1).sum()), "observable_flips": int(obs.sum())}
    return out
