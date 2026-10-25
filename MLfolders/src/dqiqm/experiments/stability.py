"""E5: the smallest surface-code stability experiment (C. Gidney, "Stability experiments: the overlooked dual
of memory experiments", Quantum 6, 786 (2022)), without reset, on a 3x3 block of Garnet.

The patch is 4 data qubits around one weight-4 check, with a weight-2 check of the other type on
each of the 4 edges. In Stim's rotated coordinates: data at (1,1), (3,1), (1,3), (3,3); the
centre check at (2,2); the edge checks at (2,0), (0,2), (4,2), (2,4). On the device grid it is a
3x3 block: the centre check in the middle, the data on the edge midpoints, the edge checks on
the corners. Garnet has 4 such blocks.

``kind="Z"``: the edge checks are Z_a Z_b, the centre is X^4. Every data qubit is in exactly two
edge checks, so the product of the four edge checks is the identity: their outcomes multiply to
+1 in every round, whatever the state, and with the data in |+...+> each one alone is random.
The observable is that product in round 1. Data errors flip two edge checks at once and cannot
change it; only time-like errors (misread or flipped ancillas) can, and the decoder must find
them through the detectors, which compare rounds (no reset: m_k + m_(k−2)). There is no logical
qubit and no spatial boundary for those detectors; the time boundaries are the start and the end
(no first-round and no final detectors for the edge checks). So the protection grows with the
number of rounds: this isolates exactly the time-like errors a memory experiment mixes with
space-like ones. ``kind="X"`` is the dual (edge checks X_a X_b, centre Z^4, data in |0...0>).

Stim's graphlike distance is R under circuit noise without record flips. With record flips (every
real readout) it is ⌈R/2⌉: without reset a misread m_k fires D_k and D_(k+2), so one record flip
spans two rounds of the time direction.

The 4-layer CX schedule is found by :func:`schedule`: every data qubit is in three checks, so it is
busy in three of the four layers, and each edge check shares two data qubits with the centre, so
the two must meet them in the same order on both (or the two checks would not be measured as
commuting operators). The noiseless Stim circuit proves the schedule: every detector and the
observable are deterministic (``layout.check_equivalent`` in ``circuits.place``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from functools import cache, partial
from typing import Any

import stim

from dqiqm import layout as L
from dqiqm.experiments import device as dev
from dqiqm.experiments.circuits import CircuitSpec, Stabilizer, place
from dqiqm.surface import NoiseModel, _as_noise_model

E5_ROUNDS = (2, 3, 4, 6, 8, 12)
KINDS = ("Z", "X")

DATA = ((1, 1), (3, 1), (1, 3), (3, 3))
CENTRE = (2, 2)
EDGES = ((2, 0), (0, 2), (4, 2), (2, 4))


def _index(x: int, y: int) -> int:
    return x + 5 * (y // 2)  # Stim's rotated-code numbering for a 4x4 coordinate box


def _support(a: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = a
    return tuple(p for p in ((x + dx, y + dy) for dx in (-1, 1) for dy in (-1, 1)) if p in DATA)


@cache
def schedule() -> tuple[tuple[tuple[tuple[int, int], tuple[int, int]], ...], ...]:
    """Four CX layers of (check, data) pairs, the first valid one in a fixed search order."""
    for centre_order in itertools.permutations(DATA):
        busy = {(d, i) for i, d in enumerate(centre_order)}
        options = []
        for e in EDGES:
            a, b = _support(e)
            opts = []
            for la, lb in itertools.permutations(range(4), 2):
                if (a, la) in busy or (b, lb) in busy:
                    continue
                # the centre meets a and b in the same order relative to this edge check
                if (centre_order.index(a) < la) != (centre_order.index(b) < lb):
                    continue
                opts.append(((a, la), (b, lb)))
            options.append(opts)
        for choice in itertools.product(*options):
            used = set(busy)
            ok = True
            for pair in choice:
                for dl in pair:
                    if dl in used:
                        ok = False
                    used.add(dl)
            if ok:
                layers: list[list] = [[] for _ in range(4)]
                for i, d in enumerate(centre_order):
                    layers[i].append((CENTRE, d))
                for e, pair in zip(EDGES, choice):
                    for d, layer in pair:
                        layers[layer].append((e, d))
                return tuple(tuple(layer) for layer in layers)
    raise RuntimeError("no valid stability schedule")


def circuit(rounds: int, kind: str = "Z", noise=None) -> stim.Circuit:
    """The no-reset stability experiment of ``rounds`` rounds (see the module docstring)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be 'Z' or 'X', not {kind!r}")
    if rounds < 2:
        raise ValueError("a stability experiment needs at least 2 rounds")
    nm: NoiseModel = _as_noise_model(noise)
    data = [_index(*p) for p in DATA]
    centre = _index(*CENTRE)
    edges = [_index(*p) for p in EDGES]
    x_checks = [centre] if kind == "Z" else edges  # the checks measured with H around them
    anc = sorted([centre, *edges])
    data_x_basis = kind == "Z"
    c = stim.Circuit()
    for p in sorted([*DATA, CENTRE, *EDGES], key=lambda p: _index(*p)):
        c.append("QUBIT_COORDS", [_index(*p)], list(p))
    c.append("R", data)
    nm.after_reset(c, data)
    c.append("R", anc)
    nm.after_reset(c, anc)
    record: list[int] = []
    last: dict[int, list[int]] = {a: [] for a in anc}

    def rec(indices):
        return [stim.target_rec(i - len(record)) for i in sorted(indices, reverse=True)]

    layers = schedule()
    for k in range(1, rounds + 1):
        c.append("TICK")
        if k == 1 and data_x_basis:
            c.append("H", data)
        nm.round_start_data(c, data)
        c.append("H", sorted(x_checks))
        nm.after_single_qubit_gate(c, sorted(x_checks))
        c.append("TICK")
        for layer in layers:
            pairs = []
            for check, d in layer:
                a, q = _index(*check), _index(*d)
                pairs.append((a, q) if a in x_checks else (q, a))
            c.append("CX", [t for p in pairs for t in p])
            nm.after_cx(c, pairs)
            c.append("TICK")
        c.append("H", sorted(x_checks))
        nm.after_single_qubit_gate(c, sorted(x_checks))
        if k == rounds and data_x_basis:
            c.append("H", data)
        c.append("TICK")
        nm.before_measure(c, anc)
        nm.measurement(c, "M", anc)
        for a in anc:
            last[a].append(len(record))
            record.append(a)
        nm.idle_live_ancillas(c, anc)
        nm.idle_data(c, data)
        t = k - 1
        for a in anc:
            m = last[a]
            deterministic_from_start = a == centre  # the centre check is of the data's basis
            if k == 1:
                if deterministic_from_start:
                    c.append("DETECTOR", rec([m[0]]), [*_coords(a), t])
                continue
            refs = [m[k - 1]] + ([m[k - 3]] if k >= 3 else [])
            c.append("DETECTOR", rec(refs), [*_coords(a), t])
    nm.before_measure(c, data)
    nm.measurement(c, "M", data)
    data_at = {}
    for q in data:
        data_at[q] = len(record)
        record.append(q)
    m = last[centre]
    c.append("DETECTOR", rec([*(data_at[q] for q in data), m[rounds - 1], m[rounds - 2]]), [*_coords(centre), rounds])
    c.append("OBSERVABLE_INCLUDE", rec(last[e][0] for e in edges), 0)
    return c


def _coords(q: int) -> tuple[int, int]:
    for p in (*DATA, CENTRE, *EDGES):
        if _index(*p) == q:
            return p
    raise KeyError(q)


def block_score(placement: L.Placement, cal: dev.Calibration, dur: dev.Durations) -> float:
    """Per round: Σ over the 12 CZs of −ln F_CZ + Σ over the 5 checks of −ln(1 − e_readout) + Σ over qubits t_round/T1."""
    phys = placement.physical
    anc = [_index(*p) for p in (CENTRE, *EDGES)]
    pairs = [(_index(*a), _index(*d)) for layer in schedule() for a, d in layer]
    t_round = 4 * max(dur.cz_ns.values()) + 4 * max(dur.prx_ns.values()) + max(dur.measure_ns[phys[a]] for a in anc)
    s = sum(-math.log(1 - cal.cz_error(phys[a], phys[d])) for a, d in pairs)
    s += sum(-math.log(1 - cal.readout_error(phys[a])) for a in anc)
    s += sum(t_round / float(cal.qubits[q]["t1_ns"]) for q in phys.values())
    return s


def best_block(arch: L.Architecture, cal: dev.Calibration, dur: dev.Durations) -> dict:
    found = L.placements(circuit(2, "Z"), arch)
    by_set: dict[frozenset, L.Placement] = {}
    for p in found:  # all 8 orientations of a block score the same: keep the first
        by_set.setdefault(p.qubit_set, p)
    scored = sorted(((block_score(p, cal, dur), p) for p in by_set.values()), key=lambda sp: (sp[0], sp[1].origin))
    best_score, best = scored[0]
    return {"placement": best, "score": best_score, "blocks": len(by_set), "placements": len(found),
            "scores": [round(s, 4) for s, _ in scored]}


def spec(rounds: int, kind: str, placement: L.Placement, backend: Any) -> CircuitSpec:
    source = circuit(rounds, kind)
    name = f"E5_stab_{kind}_r{rounds:02d}"
    native, stim_qubits = L.native_circuit(source, target="hardware", name=name)
    hc = place(source, native, stim_qubits, backend, placement)
    data = [_index(*p) for p in DATA]
    centre = _index(*CENTRE)
    edges = [_index(*p) for p in EDGES]
    anc = sorted([centre, *edges])
    edge_pauli = kind
    centre_pauli = "X" if kind == "Z" else "Z"
    stabs = {centre: Stabilizer(f"{centre_pauli}(2,2)", centre_pauli, tuple(sorted(data)))}
    for e, p in zip(edges, EDGES):
        stabs[e] = Stabilizer(f"{edge_pauli}{p}".replace(" ", ""), edge_pauli, tuple(sorted(_index(*d) for d in _support(p))))
    x_checks = [centre] if kind == "Z" else edges
    pairs = tuple(
        ((_index(*a), _index(*d)) if _index(*a) in x_checks else (_index(*d), _index(*a)))
        for layer in schedule() for a, d in layer
    )
    return CircuitSpec(
        experiment="E5",
        name=name,
        stim=source,
        hardware=hc,
        params={
            "family": "stability",
            "kind": kind,
            "rounds": rounds,
            "basis": "X" if kind == "Z" else "Z",  # the data's preparation and readout basis
            "prep": "+" if kind == "Z" else "0",
            "patch": "block3x3",
            "delay_ns": 0.0,
            "barriers": True,
        },
        roles={q: ("data" if q in data else "ancilla") for q in [*data, *anc]},
        stabilizers=stabs,
        meas_round=tuple([k for k in range(1, rounds + 1) for _ in anc] + [rounds + 1] * len(data)),
        noisy=partial(circuit, rounds, kind),
        x_ancillas=tuple(x_checks),
        cx_pairs=pairs,
        checks={"swaps": int(hc.circuit.count_ops().get("swap", 0))},
    )


def e5_specs(placement: L.Placement, backend: Any, rounds: Iterable[int] = E5_ROUNDS) -> list[CircuitSpec]:
    return [spec(r, k, placement, backend) for k in KINDS for r in rounds]
