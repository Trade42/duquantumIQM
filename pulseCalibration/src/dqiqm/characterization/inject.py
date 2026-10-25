"""The closure test's twin of a placed circuit: the same circuit with known coherent errors added.

The placed circuit (checked by :func:`.circuits.check`) is copied instruction by instruction:

* before each ``r`` (``prx(pi/2, phase)``) on a qubit with an error (eps, eta), the exact unitary
  E = prx(pi/2, phase)^dag . G(phase) (:func:`.model.prx_error`), so that E then the ideal gate is
  the modelled device gate G;
* after each ``cz`` on a coupler with an error, ``rz(phi_a)`` on a, ``rz(phi_b)`` on b and
  ``cp(delta)`` on (a, b): together diag(1, e^{i phi_b}, e^{i phi_a}, e^{i(phi_a + phi_b + delta)}),
  which commutes with the CZ.

The added gates are ``unitary``, ``rz`` and ``cp``. The stand-in's Aer noise model attaches noise only to
``r``, ``cz`` and readout, so the injected errors are exact and instantaneous and the calibrated
noise stays where it was. The twin is for Aer only: it is never placed, planned or submitted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, RZGate, UnitaryGate

from dqiqm.characterization import model as M
from dqiqm.characterization.circuits import Placed
from dqiqm.experiments import device as dev


@dataclass(frozen=True)
class Errors:
    """Injected coherent errors: ``prx[q] = (eps, eta)``; ``cz[(a, b)] = (delta, phi_a, phi_b)`` on sorted pairs."""

    prx: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    cz: Mapping[tuple[str, str], tuple[float, float, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "prx": {q: {"eps_rad": e, "eta_rad": h} for q, (e, h) in self.prx.items()},
            "cz": {f"{a}-{b}": {"delta_rad": d, f"phi_{a}_rad": pa, f"phi_{b}_rad": pb}
                   for (a, b), (d, pa, pb) in self.cz.items()},
        }

    def phi(self, pair: tuple[str, str], q: str) -> float:
        d, pa, pb = self.cz.get(dev.pair_key(*pair), (0.0, 0.0, 0.0))
        return pa if q == dev.pair_key(*pair)[0] else pb


def random_errors(qubits, couplers, *, seed: int, eps_max: float = 0.03, eta_max: float = 0.03,
                  delta_max: float = 0.08, phi_max: float = 0.08) -> Errors:
    """Seeded uniform errors, a few times what C1/C2 resolve and inside what a calibrated device has."""
    rng = np.random.default_rng(seed)
    prx = {q: (float(rng.uniform(-eps_max, eps_max)), float(rng.uniform(-eta_max, eta_max))) for q in qubits}
    cz = {dev.pair_key(*p): (float(rng.uniform(-delta_max, delta_max)), float(rng.uniform(-phi_max, phi_max)),
                              float(rng.uniform(-phi_max, phi_max))) for p in couplers}
    return Errors(prx, cz)


def physical(p: Placed, num_qubits: int, name_to_index: Mapping[str, int]) -> QuantumCircuit:
    """The circuit the server would receive, on the backend's physical indices, for Aer.

    Built from the serialized instructions (:func:`dqiqm.layout.sent_instructions`), so the twin
    is exactly what the server would receive. IQM's level-0 transpilation returns the circuit in
    virtual qubit order (its qubit 3 is QB1) with a layout; Aer reads that layout correctly (the
    same seed gives identical samples either way, checked on the N = 0 circuit), but the twin
    does not depend on it. Clbit j is the measurement key's clbit.
    """

    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    from dqiqm import layout as L

    out = QuantumCircuit(num_qubits, len(p.qubits), name=p.design.name)
    for inst in L.sent_instructions(p.circuit, p.qubit_index_to_name):
        idx = [name_to_index[q] for q in inst.locus]
        if inst.name == "prx":
            out.r(float(inst.args["angle"]), float(inst.args["phase"]), idx[0])
        elif inst.name == "cz":
            out.cz(*idx)
        elif inst.name == "barrier":
            out.barrier(idx)
        elif inst.name == "measure":
            out.measure(idx[0], MeasurementKey.from_string(inst.args["key"]).clbit_idx)
        else:
            raise ValueError(f"{inst.name} is not expected in a characterization circuit")
    return out


def twin(p: Placed, errors: Errors, backend: Any) -> QuantumCircuit:
    """:func:`physical` of ``p`` with ``errors`` added as exact gates (see the module docstring)."""
    src = physical(p, backend.num_qubits, {q: backend.qubit_name_to_index(q) for q in p.qubits})
    name_of = {backend.qubit_name_to_index(q): q for q in p.qubits}
    out = QuantumCircuit(*src.qregs, *src.cregs, name=src.name)
    for ci in src.data:
        idx = [src.find_bit(q).index for q in ci.qubits]
        qs = [out.qubits[i] for i in idx]
        name = ci.operation.name
        if name == "r":
            q = name_of[idx[0]]
            if q in errors.prx:
                angle, phase = (float(x) for x in ci.operation.params)
                if abs(angle - math.pi / 2) > 1e-12:
                    raise ValueError(f"the PRX error model is for prx(pi/2); got angle {angle}")
                eps, eta = errors.prx[q]
                out.append(UnitaryGate(M.prx_error(phase, eps, eta), label=f"err_{q}"), qs)
        out.append(ci.operation, qs, [out.clbits[src.find_bit(c).index] for c in ci.clbits])
        if name == "cz":
            a, b = name_of[idx[0]], name_of[idx[1]]
            key = dev.pair_key(a, b)
            if key in errors.cz:
                delta, _, _ = errors.cz[key]
                out.append(RZGate(errors.phi(key, a)), [qs[0]])
                out.append(RZGate(errors.phi(key, b)), [qs[1]])
                out.append(CPhaseGate(delta), qs)
    return out


def aer(backend: Any, *, seed: int, noisy: bool = True, threads: int = 1, method: str = "matrix_product_state"):
    """The stand-in's simulator (``scripts/garnet_dryrun.py``'s settings): Aer MPS with the backend's noise."""
    from qiskit_aer import AerSimulator

    return AerSimulator(method=method, noise_model=backend.noise_model if noisy else None, seed_simulator=seed,
                        max_parallel_threads=threads)
