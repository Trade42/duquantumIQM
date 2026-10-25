"""The coherent-error model of Garnet's two gates, and the exact forward model of the C1 sequences.

One ``prx(pi/2, phi)``
    The device's gate is modelled as

        G(phi) = RZ(phi) . exp(-i/2 [(pi/2 + eps) X + eta Z]) . RZ(-phi)
               = exp(-i/2 [(pi/2 + eps)(cos(phi) X + sin(phi) Y) + eta Z]),

    with RZ(a) = diag(exp(-ia/2), exp(ia/2)). ``eps`` is the signed amplitude (rotation-angle)
    error: eps > 0 over-rotates. ``eta`` is the Z component of the rotation vector, in rad: the
    axis tilt out of the equator, from a detuning, a DRAG error or a miscalibrated ``rz_before`` /
    ``rz_after``. It is what the table calls the axis (phase) error. IQM applies ``phi`` as a
    frame change, which is exact, so G(phi) is G(0) conjugated by RZ(phi).

    **The in-plane axis offset is a gauge.** A constant offset of every gate's phase on a qubit is
    a frame change. It commutes with CZ and with a Z measurement, so no circuit that starts in |0>
    and ends in a Z readout can see it. The model has no such parameter, on purpose.

One CZ on (a, b)
    diag(1, e^{i phi_b}, e^{i phi_a}, -e^{i(phi_a + phi_b + delta)}) in the basis |ab>: ``delta`` is
    the conditional-phase error (the entangling phase is pi + delta), ``phi_a`` / ``phi_b`` the
    residual local Z phases left after IQM's own virtual-RZ corrections. A phase ``phi`` here is
    the one ``RZ(phi) ~ diag(1, e^{i phi})`` adds to |1>. Leakage and residual swap terms are not
    in it (C3 is the leakage design).

The C1 forward model
    A qubit's state is its Bloch vector, so every step is a 4x4 affine map in homogeneous
    coordinates: the rotation of G(phi), then the relaxation over the gate's duration (T1 toward
    |0>, coherences decaying at T2), then a depolarizing shrink by (1 - lam), in the order of the
    stand-in's Aer noise (``thermal_relaxation.compose(depolarizing)``). A sequence of N
    identical gates is a matrix power. The readout is classical: P(read 1) = a0 + (1 - a0 - a1) p1,
    with a0 = P(1|0) and a1 = P(0|1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
PAULIS = (X, Y, Z)
HALF_PI = math.pi / 2


def prx(angle: float, phase: float) -> np.ndarray:
    """IQM's ``prx(angle, phase)`` = exp(-i angle/2 (cos(phase) X + sin(phase) Y)); qiskit's ``RGate``."""
    return expm(-0.5j * angle * (math.cos(phase) * X + math.sin(phase) * Y))


def prx_actual(phase: float, eps: float = 0.0, eta: float = 0.0, angle: float = HALF_PI) -> np.ndarray:
    """The modelled device gate G(phase) for a nominal ``prx(angle, phase)``."""
    return expm(-0.5j * ((angle + eps) * (math.cos(phase) * X + math.sin(phase) * Y) + eta * Z))


def prx_error(phase: float, eps: float = 0.0, eta: float = 0.0, angle: float = HALF_PI) -> np.ndarray:
    """E with G(phase) = prx(angle, phase) . E: the error unitary applied just *before* the ideal gate."""
    return prx(angle, phase).conj().T @ prx_actual(phase, eps, eta, angle)


def cz_actual(phi_a: float = 0.0, phi_b: float = 0.0, delta: float = 0.0) -> np.ndarray:
    """The modelled CZ in the basis |ab> = |00>, |01>, |10>, |11> (a is the left factor)."""
    return np.diag([1, np.exp(1j * phi_b), np.exp(1j * phi_a), -np.exp(1j * (phi_a + phi_b + delta))])


# ---------------------------------------------------------------------------
# Bloch-vector maps.
# ---------------------------------------------------------------------------


def rotation(u: np.ndarray) -> np.ndarray:
    """The SO(3) matrix of a 2x2 unitary: R_ij = tr(s_i U s_j U^dag) / 2."""
    ud = u.conj().T
    return np.array([[0.5 * np.real(np.trace(si @ u @ sj @ ud)) for sj in PAULIS] for si in PAULIS])


def affine(r3: np.ndarray, offset: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r3
    m[:3, 3] = offset
    return m


def relaxation(t_ns: float, t1_ns: float, t2_ns: float) -> np.ndarray:
    """T1/T2 decay over ``t_ns`` (zero temperature, as Aer's ``thermal_relaxation_error``)."""
    if t_ns <= 0:
        return np.eye(4)
    e1 = math.exp(-t_ns / t1_ns)
    e2 = math.exp(-t_ns / min(t2_ns, 2 * t1_ns))
    return affine(np.diag([e2, e2, e1]), (0.0, 0.0, 1 - e1))


def depolarize(lam: float) -> np.ndarray:
    return affine(np.eye(3) * (1 - lam))


@dataclass(frozen=True)
class QubitModel:
    """One qubit's C1 parameters. ``lam`` is the depolarizing shrink per gate (an Aer depolarizing
    parameter p; 2x the RB infidelity for a pure depolarizing gate); T1, T2 and the gate time are
    fixed from the calibration in a fit."""

    eps: float = 0.0
    eta: float = 0.0
    lam: float = 0.0
    a0: float = 0.0  # P(read 1 | 0)
    a1: float = 0.0  # P(read 0 | 1)
    t1_ns: float = math.inf
    t2_ns: float = math.inf
    gate_ns: float = 16.0

    def step(self, phase: float) -> np.ndarray:
        """The affine map of one gate: rotation, then relaxation, then depolarizing."""
        rot = affine(rotation(prx_actual(phase, self.eps, self.eta)))
        relax = relaxation(self.gate_ns, self.t1_ns, self.t2_ns) if math.isfinite(self.t1_ns) else np.eye(4)
        return depolarize(self.lam) @ relax @ rot


#: A C1 sequence: ((phase, repeat), ...) in time order; ((0.0, 5),) is five prx(pi/2, 0).
Sequence1Q = tuple[tuple[float, int], ...]


def p1_ideal_bloch(seq: Sequence1Q, m: QubitModel) -> float:
    """P(qubit in |1>) after ``seq`` from |0>, before readout."""
    v = np.array([0.0, 0.0, 1.0, 1.0])
    cache: dict[float, np.ndarray] = {}
    for phase, k in seq:
        if k <= 0:
            continue
        if phase not in cache:
            cache[phase] = m.step(phase)
        v = np.linalg.matrix_power(cache[phase], k) @ v
    return float((1 - v[2]) / 2)


def p_read1(seq: Sequence1Q, m: QubitModel) -> float:
    """P(read 1) after ``seq``: the Bloch evolution and the classical readout."""
    p1 = p1_ideal_bloch(seq, m)
    return m.a0 + (1 - m.a0 - m.a1) * p1


def expand(seq: Sequence1Q) -> list[float]:
    """The phase of every gate, in order."""
    return [phase for phase, k in seq for _ in range(k)]


# ---------------------------------------------------------------------------
# Two-qubit ideal reference (for tests and the C2 design): exact state vectors.
# ---------------------------------------------------------------------------


def apply1(state: np.ndarray, u: np.ndarray, q: int, n: int) -> np.ndarray:
    """``u`` on qubit ``q`` of an ``n``-qubit state vector (qubit 0 is the leftmost tensor factor)."""
    psi = state.reshape([2] * n)
    psi = np.moveaxis(np.tensordot(u, psi, axes=([1], [q])), 0, q)
    return psi.reshape(-1)


def apply2(state: np.ndarray, u: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    psi = state.reshape([2] * n)
    u4 = u.reshape(2, 2, 2, 2)
    psi = np.tensordot(u4, psi, axes=([2, 3], [q0, q1]))
    psi = np.moveaxis(psi, [0, 1], [q0, q1])
    return psi.reshape(-1)


def p1_of(state: np.ndarray, q: int, n: int) -> float:
    psi = state.reshape([2] * n)
    return float(np.sum(np.abs(np.take(psi, 1, axis=q)) ** 2))


def simulate_design(layers, qubits: Sequence[str], prx_err=None, cz_err=None) -> dict[str, float]:
    """P(1) of each of ``qubits`` after a design's layers, noise-free, with coherent errors.

    ``layers`` are :class:`dqiqm.characterization.circuits.Design` layers; gates entirely outside
    ``qubits`` are skipped (C2's couplers are independent), a gate across the boundary is an
    error. ``prx_err[q] = (eps, eta)``; ``cz_err[(a, b)] = (delta, phi_a, phi_b)`` on sorted pairs,
    sorted as ``dqiqm.experiments.device.pair_key`` sorts.
    """
    from dqiqm.experiments.device import pair_key

    prx_err, cz_err = prx_err or {}, cz_err or {}
    idx = {q: i for i, q in enumerate(qubits)}
    n = len(qubits)
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1
    for layer in layers:
        for g in layer:
            inside = [q in idx for q in g.qubits]
            if not any(inside):
                continue
            if not all(inside):
                raise ValueError(f"{g} crosses the simulated set {qubits}")
            if g.op == "r":
                eps, eta = prx_err.get(g.qubits[0], (0.0, 0.0))
                state = apply1(state, prx_actual(g.phase, eps, eta, g.angle), idx[g.qubits[0]], n)
            else:
                a, b = pair_key(*g.qubits)
                delta, pa, pb = cz_err.get((a, b), (0.0, 0.0, 0.0))
                state = apply2(state, cz_actual(pa, pb, delta), idx[a], idx[b], n)
    return {q: p1_of(state, idx[q], n) for q in qubits}
