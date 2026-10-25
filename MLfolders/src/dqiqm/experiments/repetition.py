"""E2: no-reset repetition codes, bit-flip and phase-flip, on the longest line Garnet allows.

Garnet's coupler graph has no Hamiltonian cycle (QB9 can only be reached through couplers that
the degree-2 corners QB3, QB8, QB13 and QB4 already use up), so the longest loop is shorter than
the longest line. The longest line has 19 qubits: 10 data and 9 ancillas alternating, distance 10.
:func:`best_path` enumerates every 19-qubit line and ranks it by calibration (the same score form
as the resonance-check task's placement score, per round).

The circuit, along the line (Stim qubit i = position i; data at even i, ancillas at odd i):

* bit-flip (``kind="bit"``): Z_i Z_(i+2) checks; each round CX(data -> ancilla) from the left
  data, then from the right data, then measure the ancillas. Prepared in |0...0> or |1...1>
  (an X on every data qubit, prepended natively), read out in Z; the observable is data 0;
* phase-flip (``kind="phase"``): X_i X_(i+2) checks; H on the ancillas, CX(ancilla -> data) left
  then right, H, measure. Prepared in |+...+> (R then H), read out in X (H then M); the
  observable is data 0's X.

No reset: the detectors are ``dqiqm.surface``'s (round 1: m_1; round 2: m_2; round k >= 3:
m_k + m_(k-2); final: data parity + m_R + m_(R-1)), every check being of the basis type. Long
runs (up to 50 rounds) give the temporal correlations, leakage streaks and the k -> k+2 record-flip
signature of the no-reset readout, on one line of 9 ancillas.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Any

import stim

from dqiqm import layout as L
from dqiqm.experiments import device as dev
from dqiqm.experiments.circuits import CircuitSpec, Stabilizer, place, prepend_x
from dqiqm.surface import NoiseModel, _as_noise_model

E2_ROUNDS = (1, 2, 3, 5, 8, 12, 20, 30, 50)
E2_PREPS = (("bit", "0"), ("bit", "1"), ("phase", "+"))
PATH_LENGTH = 19


def paths(arch: L.Architecture, length: int = PATH_LENGTH) -> list[tuple[str, ...]]:
    """Every simple path of ``length`` qubits along couplers, once each (the lexicographically smaller direction)."""
    adj: dict[str, list[str]] = {q: [] for q in arch.qubits}
    for c in arch.couplers:
        a, b = sorted(c)
        adj[a].append(b)
        adj[b].append(a)
    for q in adj:
        adj[q].sort(key=dev.qubit_sort_key)
    found: set[tuple[str, ...]] = set()

    def extend(path: list[str], seen: set[str]) -> None:
        if len(path) == length:
            t = tuple(path)
            r = t[::-1]
            found.add(min(t, r, key=lambda p: [dev.qubit_sort_key(q) for q in p]))
            return
        for n in adj[path[-1]]:
            if n not in seen:
                seen.add(n)
                path.append(n)
                extend(path, seen)
                path.pop()
                seen.remove(n)

    for q in arch.qubits:
        extend([q], {q})
    return sorted(found, key=lambda p: [dev.qubit_sort_key(q) for q in p])


def path_score(path: Sequence[str], cal: dev.Calibration, dur: dev.Durations) -> float:
    """Per round: Σ over the 2(n−1) CZs of −ln F_CZ + Σ over ancillas of −ln(1 − e_readout) + Σ over qubits t_round/T1."""
    t_round = 2 * statistics.median(dur.cz_ns.values()) + 4 * statistics.median(dur.prx_ns.values()) + max(
        dur.measure_ns[q] for q in path[1::2]
    )
    s = sum(-math.log(1 - cal.cz_error(a, b)) for a, b in zip(path, path[1:]))
    s += sum(-math.log(1 - cal.readout_error(q)) for q in path[1::2])
    s += sum(t_round / float(cal.qubits[q]["t1_ns"]) for q in path)
    return s


def best_path(arch: L.Architecture, cal: dev.Calibration, dur: dev.Durations, length: int = PATH_LENGTH) -> dict:
    """The best-scoring line of ``length`` qubits, the number of lines, and the runner-up scores."""
    scored = sorted(((path_score(p, cal, dur), p) for p in paths(arch, length)), key=lambda sp: (sp[0], sp[1]))
    if not scored:
        raise ValueError(f"no line of {length} qubits on this architecture")
    # the line is the same either way round; put the end with the better readout first as data 0
    score, p = scored[0]
    if cal.readout_error(p[-1]) < cal.readout_error(p[0]):
        p = p[::-1]
    return {"path": p, "score": score, "n_paths": len(scored), "next_scores": [s for s, _ in scored[1:4]]}


def circuit(n_data: int, rounds: int, kind: str = "bit", prep: str = "0", noise=None, *, with_x: bool = True) -> stim.Circuit:
    """The no-reset repetition code on a line of 2·n_data − 1 qubits (see the module docstring).

    ``with_x=False`` leaves out the X gates of a |1...1> preparation: that is the source the
    converter takes, before the X gates are prepended natively.
    """
    if kind not in ("bit", "phase"):
        raise ValueError(f"kind must be 'bit' or 'phase', not {kind!r}")
    if (kind, prep) not in (("bit", "0"), ("bit", "1"), ("phase", "+")):
        raise ValueError(f"prep {prep!r} is not supported for kind {kind!r}")
    nm: NoiseModel = _as_noise_model(noise)
    n = 2 * n_data - 1
    data = list(range(0, n, 2))
    anc = list(range(1, n, 2))
    c = stim.Circuit()
    for q in range(n):
        c.append("QUBIT_COORDS", [q], [q, 0])
    c.append("R", data)
    nm.after_reset(c, data)
    c.append("R", anc)
    nm.after_reset(c, anc)
    if prep == "1" and with_x:
        c.append("X", data)
        nm.after_single_qubit_gate(c, data)
    record: list[int] = []
    last: dict[int, list[int]] = {a: [] for a in anc}

    def rec(indices):
        return [stim.target_rec(i - len(record)) for i in sorted(indices, reverse=True)]

    for k in range(1, rounds + 1):
        c.append("TICK")
        if k == 1 and kind == "phase":
            c.append("H", data)  # |0> -> |+>: the preparation, no gate noise
        nm.round_start_data(c, data)
        if kind == "phase":
            c.append("H", anc)
            nm.after_single_qubit_gate(c, anc)
            c.append("TICK")
        for side in (-1, 1):
            pairs = [(a + side, a) if kind == "bit" else (a, a + side) for a in anc]
            c.append("CX", [q for p in pairs for q in p])
            nm.after_cx(c, pairs)
            c.append("TICK")
        if kind == "phase":
            c.append("H", anc)
            nm.after_single_qubit_gate(c, anc)
        if k == rounds and kind == "phase":
            c.append("H", data)  # the X readout: no gate noise
        c.append("TICK")
        nm.before_measure(c, anc)
        nm.measurement(c, "M", anc)
        for a in anc:
            last[a].append(len(record))
            record.append(a)
        nm.idle_live_ancillas(c, anc)
        nm.idle_data(c, data)
        for a in anc:
            m = last[a]
            refs = [m[k - 1]] + ([m[k - 3]] if k >= 3 else [])
            c.append("DETECTOR", rec(refs), [a, 0, k - 1])
    nm.before_measure(c, data)
    nm.measurement(c, "M", data)
    data_at = {}
    for q in data:
        data_at[q] = len(record)
        record.append(q)
    for a in anc:
        m = last[a]
        refs = [data_at[a - 1], data_at[a + 1], m[rounds - 1]] + ([m[rounds - 2]] if rounds >= 2 else [])
        c.append("DETECTOR", rec(refs), [a, 0, rounds])
    c.append("OBSERVABLE_INCLUDE", rec([data_at[0]]), 0)
    return c


def spec(path: Sequence[str], rounds: int, kind: str, prep: str, backend: Any) -> CircuitSpec:
    n = len(path)
    n_data = (n + 1) // 2
    source = circuit(n_data, rounds, kind, prep)
    name = f"E2_rep{n_data}_{kind}_{'p' if prep == '+' else prep}_r{rounds:02d}"
    native, stim_qubits = L.native_circuit(circuit(n_data, rounds, kind, prep, with_x=False), target="hardware", name=name)
    data = list(range(0, n, 2))
    if prep == "1":
        native = prepend_x(native, [stim_qubits.index(q) for q in data])
    placement = L.Placement(origin=(0, 0), orientation=0, mapping=tuple((i, q) for i, q in enumerate(path)))
    hc = place(source, native, stim_qubits, backend, placement)
    pauli = "Z" if kind == "bit" else "X"
    anc = list(range(1, n, 2))
    pairs = tuple((a - 1, a) for a in anc) + tuple((a + 1, a) for a in anc)
    return CircuitSpec(
        experiment="E2",
        name=name,
        stim=source,
        hardware=hc,
        params={
            "family": "repetition",
            "kind": kind,
            "distance": n_data,
            "rounds": rounds,
            "basis": "Z" if kind == "bit" else "X",
            "prep": prep,
            "patch": "line19",
            "delay_ns": 0.0,
            "barriers": True,
        },
        roles={q: ("data" if q % 2 == 0 else "ancilla") for q in range(n)},
        stabilizers={a: Stabilizer(f"{pauli}{pauli}({a})", pauli, (a - 1, a + 1)) for a in anc},
        meas_round=tuple([k for k in range(1, rounds + 1) for _ in anc] + [rounds + 1] * len(data)),
        noisy=partial(circuit, n_data, rounds, kind, prep),
        x_ancillas=tuple(anc) if kind == "phase" else (),
        cx_pairs=pairs,
        checks={"swaps": int(hc.circuit.count_ops().get("swap", 0))},
    )


def e2_specs(path: Sequence[str], backend: Any, rounds: Iterable[int] = E2_ROUNDS, preps=E2_PREPS) -> list[CircuitSpec]:
    return [spec(path, r, kind, prep, backend) for kind, prep in preps for r in rounds]
