"""The characterization circuits: designs as layers, placement on Garnet, and the checks.

A :class:`Design` is a list of layers. Each layer holds gates on disjoint qubits and is followed by
a barrier on all 17 patch qubits; the last layer is followed by a barrier and the measurement of
all 17 qubits, qubit i into clbit i (logical qubit i is ``patch.qubits[i]``). Every gate is
``prx(pi/2, phase)``, one 16 ns pulse with ``drag_crf_sx``, or ``cz``. A ``prx(pi)`` is two pulses,
so an X is written as two ``prx(pi/2)`` in two layers.

**No gate merging.** Qiskit's transpiler merges repeated gates at optimization level 1 and above
(``docs/pulla-check.md``), so every repeated gate here has a barrier between it and the next, and
placement is at level 0. :func:`check` proves it on the exact circuit that would go to
``hw.submit``: ``count_ops()`` equals the design's counts, and in the serialized instructions every
qubit's ops are the designed ones, with a barrier between every two of them.

The experiments
---------------
**C1, PRX amplification** (all 17 qubits in parallel, the same sequence on each):

* ``amp_x``: N x ``prx(pi/2, 0)`` from |0>, N in :data:`C1_AMP_N`. N = 4k+1 and 4k+3 flip the sign
  of the amplitude signal (P1 ~ (1 +- sin(N eps)) / 2), so an offset or a drift toward |0> cannot
  mimic it. Even N end on a pole: N = 0 and 2 anchor the readout; N = 40 and 42 show the decay
  over 40 pulses, which a merged sequence would not have (the pilot's no-merging check).
* ``amp_y``: the same at phase pi/2, the phase the surface code's gates use (a model check: one
  eps for both phases).
* ``tilt``: ``prx(pi/2, pi/2)``, then n pairs [``prx(pi/2, 0)``, ``prx(pi/2, pi)``], then an
  analysis ``prx(pi/2, 0)`` (``pos``) or ``prx(pi/2, pi)`` (``neg``). The pair cancels eps to first
  order; an axis tilt eta turns each pair into a small rotation about (Y+Z)/sqrt(2) (about 1.8 eta),
  which accumulates linearly, and the two analysis signs read it with opposite signs.

**C2, conditional Ramsey with repeated CZ** (one surface-code CZ layer at a time: its 6 couplers
in parallel, as the round runs them). For each coupler the target gets ``prx(pi/2, pi/2)``; the
control gets two pulses, X90.X90 (s = 1, an X) or X90.X-90 (s = 0, an identity of the same length);
then n CZs, n in :data:`C2_N`, a barrier after each; then the target's analysis ``prx(pi/2, psi)``,
psi in {0, pi/2, pi, 3pi/2}. The target's phase is Phi_s(n) = A_s + n (phi_t + s (pi + delta)).
Both orientations run: the ancilla as target (``tanc``), then the data qubit (``tdat``).

**C4, isolated twins** (Should): C1 on one qubit at a time and C2 on one coupler at a time, the
other patch qubits idle in |0>, on a reduced set of points. The differences from C1/C2 are the
drive crosstalk and the spectator phases of the parallel context.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RGate

from dqiqm import layout as L
from dqiqm.characterization.patch import Patch
from dqiqm.experiments import device as dev

SEED = L.SEED
HALF_PI = math.pi / 2

C1_AMP_N = (0, 1, 2, 3, 5, 7, 9, 11, 17, 19, 33, 35, 40, 41, 42, 43)
C1_AMPY_N = (17, 19, 41, 43)
C1_TILT_PAIRS = (0, 2, 4, 8, 12, 16)
C2_N = (0, 1, 3, 9)
#: The pilot's longer CZ chain (layer 1, target = ancilla, s = 1): 16 CZs between it and n = 1.
C2_PILOT_N = 17
C2_PSI = (0.0, HALF_PI, math.pi, 3 * HALF_PI)
ORIENTATIONS = ("tanc", "tdat")  # target = the ancilla / the data qubit of each pair

#: C4's reduced point sets.
C4_C1_AMP_N = (2, 41, 43)
C4_C1_TILT = ((16, "pos"), (16, "neg"))
C4_C2_N = (1, 9)


@dataclass(frozen=True)
class Gate:
    op: str  # "r" (prx) or "cz"
    qubits: tuple[str, ...]
    angle: float = HALF_PI
    phase: float = 0.0


@dataclass(frozen=True)
class Design:
    """One circuit: layers of gates on disjoint qubits, a barrier after each, then all 17 measured."""

    experiment: str  # "C1", "C2", "C4"
    name: str
    layers: tuple[tuple[Gate, ...], ...]
    params: Mapping[str, Any] = field(default_factory=dict)

    def per_qubit(self, qubits: Sequence[str]) -> dict[str, list[tuple]]:
        """Each qubit's designed ops: ("prx", angle, phase) or ("cz", sorted pair), in order."""
        out: dict[str, list[tuple]] = {q: [] for q in qubits}
        for layer in self.layers:
            for g in layer:
                item = ("prx", round(g.angle, 12), round(g.phase % (2 * math.pi), 12)) if g.op == "r" else (
                    "cz", tuple(sorted(g.qubits, key=dev.qubit_sort_key)))
                for q in g.qubits:
                    out[q].append(item)
        return out

    def counts(self, n_qubits: int) -> dict[str, int]:
        """The circuit's ``count_ops()``: one barrier per layer (the last layer's is the one before the readout)."""
        c = {"r": 0, "cz": 0}
        for layer in self.layers:
            for g in layer:
                c[g.op] += 1
        out = {k: v for k, v in c.items() if v}
        if self.layers:
            out["barrier"] = len(self.layers)
        out["measure"] = n_qubits
        return out


def _r(q: str, phase: float) -> Gate:
    return Gate("r", (q,), HALF_PI, phase)


def _layer(qubits: Iterable[str], phase: float) -> tuple[Gate, ...]:
    return tuple(_r(q, phase) for q in qubits)


# ---------------------------------------------------------------------------
# C1: PRX amplification.
# ---------------------------------------------------------------------------


def c1_sequence(kind: str, n: int, sign: str = "pos") -> tuple[tuple[float, int], ...]:
    """A C1 sequence as runs of (phase, repeat), for :mod:`.model`."""
    if kind == "amp_x":
        return ((0.0, n),) if n else ()
    if kind == "amp_y":
        return ((HALF_PI, n),) if n else ()
    if kind == "tilt":
        pairs = ((0.0, 1), (math.pi, 1)) * n
        return ((HALF_PI, 1), *pairs, (0.0 if sign == "pos" else math.pi, 1))
    raise ValueError(kind)


def c1_design(kind: str, n: int, qubits: Sequence[str], *, sign: str = "pos", experiment: str = "C1",
              suffix: str = "") -> Design:
    seq = c1_sequence(kind, n, sign)
    layers = tuple(_layer(qubits, phase) for phase, k in seq for _ in range(k))
    name = f"{experiment}_{kind}_" + (f"p{n:02d}_{sign}" if kind == "tilt" else f"n{n:02d}") + suffix
    params = {"family": "prx_amplification", "kind": kind, "n": n, "sign": sign if kind == "tilt" else None,
              "sequence": [[phase, k] for phase, k in seq], "driven": list(qubits), "pulses": sum(k for _, k in seq)}
    return Design(experiment, name, layers, params)


def c1_designs(patch: Patch) -> list[Design]:
    q = patch.qubits
    out = [c1_design("amp_x", n, q) for n in C1_AMP_N]
    out += [c1_design("amp_y", n, q) for n in C1_AMPY_N]
    out += [c1_design("tilt", n, q, sign=s) for n in C1_TILT_PAIRS for s in ("pos", "neg")]
    return out


# ---------------------------------------------------------------------------
# C2: conditional Ramsey with repeated CZ.
# ---------------------------------------------------------------------------


def c2_roles(pairs: Sequence[tuple[str, str]], orientation: str) -> list[tuple[str, str]]:
    """(target, control) per (data, ancilla) pair."""
    if orientation == "tanc":
        return [(a, d) for d, a in pairs]
    if orientation == "tdat":
        return [(d, a) for d, a in pairs]
    raise ValueError(orientation)


def c2_design(pairs: Sequence[tuple[str, str]], orientation: str, s: int, n: int, k_psi: int, *,
              experiment: str = "C2", tag: str = "") -> Design:
    roles = c2_roles(pairs, orientation)
    psi = C2_PSI[k_psi]
    first = tuple(_r(t, HALF_PI) for t, _ in roles) + tuple(_r(c, 0.0) for _, c in roles)
    second = tuple(_r(c, 0.0 if s else math.pi) for _, c in roles)
    czs = tuple(Gate("cz", (c, t), 0.0, 0.0) for t, c in roles)
    analysis = tuple(_r(t, psi) for t, _ in roles)
    layers = (first, second, *([czs] * n), analysis)
    name = f"{experiment}_{tag}_t{orientation[1:]}_s{s}_n{n:02d}_q{k_psi}"
    params = {"family": "conditional_ramsey", "orientation": orientation, "s": s, "n": n, "psi_index": k_psi,
              "psi": psi, "pairs": [list(p) for p in pairs], "targets": [t for t, _ in roles],
              "controls": [c for _, c in roles], "tag": tag}
    return Design(experiment, name, layers, params)


def c2_designs(patch: Patch) -> list[Design]:
    out = []
    for li, layer in enumerate(patch.layers, 1):
        for o in ORIENTATIONS:
            for s in (0, 1):
                for n in C2_N:
                    for k in range(len(C2_PSI)):
                        d = c2_design(layer, o, s, n, k, tag=f"L{li}")
                        out.append(Design(d.experiment, d.name, d.layers, {**d.params, "layer": li}))
    return out


def c2_pilot_designs(patch: Patch) -> list[Design]:
    """The pilot's n = 17 chain on layer 1 (target = ancilla, s = 1): not part of the C2 jobs."""
    out = []
    for k in range(len(C2_PSI)):
        d = c2_design(patch.layers[0], "tanc", 1, C2_PILOT_N, k, tag="L1")
        out.append(Design(d.experiment, d.name, d.layers, {**d.params, "layer": 1, "pilot_only": True}))
    return out


# ---------------------------------------------------------------------------
# C4: isolated twins (Should).
# ---------------------------------------------------------------------------


def c4_designs(patch: Patch) -> list[Design]:
    out = []
    for q in patch.qubits:
        for n in C4_C1_AMP_N:
            out.append(c1_design("amp_x", n, (q,), experiment="C4", suffix=f"_{q}"))
        for n, sign in C4_C1_TILT:
            out.append(c1_design("tilt", n, (q,), sign=sign, experiment="C4", suffix=f"_{q}"))
    for li, layer in enumerate(patch.layers, 1):
        for pair in layer:
            for s in (0, 1):
                for n in C4_C2_N:
                    for k in range(len(C2_PSI)):
                        d = c2_design((pair,), "tanc", s, n, k, experiment="C4", tag=f"{pair[0]}-{pair[1]}")
                        out.append(Design(d.experiment, d.name, d.layers, {**d.params, "layer": li, "isolated": True}))
    return out


# ---------------------------------------------------------------------------
# Native circuits, placement and checks.
# ---------------------------------------------------------------------------


def native(design: Design, patch: Patch) -> QuantumCircuit:
    """The logical circuit: qubit i is ``patch.qubits[i]``, clbit i its readout."""
    n = len(patch.qubits)
    idx = {q: i for i, q in enumerate(patch.qubits)}
    qc = QuantumCircuit(n, n, name=design.name)
    for layer in design.layers:
        used = [q for g in layer for q in g.qubits]
        if len(set(used)) != len(used):
            raise ValueError(f"{design.name}: a layer acts twice on a qubit")
        for g in layer:
            if g.op == "r":
                qc.append(RGate(g.angle, g.phase), [idx[g.qubits[0]]])
            elif g.op == "cz":
                qc.cz(idx[g.qubits[0]], idx[g.qubits[1]])
            else:
                raise ValueError(f"{design.name}: {g.op} is not a native gate here")
        qc.barrier()  # after the last layer, this is the barrier before the readout
    for i in range(n):
        qc.measure(i, i)
    return qc


@dataclass(frozen=True)
class Placed:
    """A design placed on Garnet: ``hw.submit(placed.circuit, qubit_index_to_name=placed.qubit_index_to_name, ...)``."""

    design: Design
    circuit: QuantumCircuit  # transpiled at level 0 onto the patch's physical qubits
    logical: QuantumCircuit
    qubit_index_to_name: dict[int, str]
    initial_layout: tuple[int, ...]
    qubits: tuple[str, ...]  # logical qubit i -> device qubit; clbit i is qubit i's readout
    target: str = "hardware"


def place(design: Design, patch: Patch, backend: Any) -> Placed:
    logical = native(design, patch)
    initial = [backend.qubit_name_to_index(q) for q in patch.qubits]
    placed = transpile(logical, backend=backend, initial_layout=initial, optimization_level=0, seed_transpiler=SEED)
    return Placed(design, placed, logical, {p: backend.index_to_qubit_name(p) for p in initial}, tuple(initial),
                  tuple(patch.qubits))


def check(p: Placed, arch: L.Architecture, couplers: Iterable[tuple[str, str]] | None = None) -> dict[str, int]:
    """Prove a placed circuit from what IQM's serializer would send, or raise ``AssertionError``.

    * the layout is the one asked for, and there is no SWAP;
    * ``count_ops()`` of the exact circuit equals the design's counts (nothing merged or dropped);
    * only ``prx``, ``cz``, ``measure`` and ``barrier`` are sent, every ``cz`` on a device coupler
      (and on one of ``couplers``, when given);
    * every qubit's ops are the designed ones, in order, with the designed angles and phases;
    * **between every two consecutive ops on a qubit there is a barrier on that qubit**;
    * clbit j is measured once, on ``qubits[j]``, as the qubit's last op.

    Returns the sent instructions' counts.
    """
    from iqm.qiskit_iqm.qiskit_to_iqm import MeasurementKey

    layout = p.circuit.layout
    assert layout is not None, "the transpiled circuit has no layout"
    assert list(layout.initial_index_layout(filter_ancillas=True)) == list(p.initial_layout), "the transpiler moved the qubits"
    ops = dict(p.circuit.count_ops())
    assert "swap" not in ops, "a SWAP was inserted"
    want = p.design.counts(len(p.qubits))
    assert ops == want, f"{p.design.name}: count_ops {ops} != designed {want}"

    sent = L.sent_instructions(p.circuit, p.qubit_index_to_name)
    allowed_cz = None if couplers is None else {frozenset(c) for c in couplers}
    counts: dict[str, int] = {}
    per_qubit: dict[str, list[tuple]] = {q: [] for q in p.qubits}
    measured: dict[int, str] = {}
    for inst in sent:
        assert inst.name in ("prx", "cz", "measure", "barrier"), f"{inst.name} is not a native operation"
        counts[inst.name] = counts.get(inst.name, 0) + 1
        if inst.name == "cz":
            assert arch.coupled(*inst.locus), f"cz on {inst.locus} is not on a coupler"
            assert allowed_cz is None or frozenset(inst.locus) in allowed_cz, f"cz on {inst.locus} is not a patch coupler"
        for q in inst.locus:
            assert q in per_qubit, f"{inst.name} on {q}, outside the patch"
            if inst.name == "prx":
                item = ("prx", round(float(inst.args["angle"]), 12), round(float(inst.args["phase"]) % (2 * math.pi), 12))
            elif inst.name == "cz":
                item = ("cz", tuple(sorted(inst.locus, key=dev.qubit_sort_key)))
            elif inst.name == "measure":
                item = ("measure",)
                j = MeasurementKey.from_string(inst.args["key"]).clbit_idx
                assert j not in measured, f"clbit {j} is written twice"
                measured[j] = q
            else:
                item = ("barrier",)
            per_qubit[q].append(item)
    assert measured == dict(enumerate(p.qubits)), "clbit j is not the readout of qubits[j]"
    designed = p.design.per_qubit(p.qubits)
    for q, seq in per_qubit.items():
        ops_q = [x for x in seq if x[0] != "barrier"]
        assert ops_q[-1] == ("measure",), f"{q}: the readout is not the last op"
        got = [(x[0], round(x[1], 9), round(x[2], 9)) if x[0] == "prx" else x for x in ops_q[:-1]]
        exp = [(x[0], round(x[1], 9), round(x[2], 9)) if x[0] == "prx" else x for x in designed[q]]
        assert got == exp, f"{q}: sent ops differ from the design"
        last = None
        for x in seq:
            if x[0] == "barrier":
                last = None
                continue
            assert last is None, f"{q}: {last} and {x} have no barrier between them"
            last = x
    return counts


def validate(p: Placed, backend: Any) -> None:
    """IQM's own check: every instruction native on the device's loci (``validate_circuit``)."""
    from iqm.qiskit_iqm.iqm_circuit_validation import validate_circuit

    validate_circuit(p.circuit, backend, qubit_index_to_name=dict(p.qubit_index_to_name))


def native_json(p: Placed) -> list[dict[str, Any]]:
    return [{"name": i.name, "locus": list(i.locus), "args": {k: (float(v) if isinstance(v, float) else v)
                                                                 for k, v in i.args.items()}}
            for i in L.sent_instructions(p.circuit, p.qubit_index_to_name)]


def native_sha256(p: Placed) -> str:
    return hashlib.sha256(json.dumps(native_json(p), sort_keys=True).encode()).hexdigest()


def qasm(p: Placed) -> str:
    from qiskit import qasm3

    return qasm3.dumps(p.circuit)


def duration_ns(design: Design, durations: Mapping[str, float], terminal_window_ns: float) -> float:
    """The circuit's length: per layer its longest gate (barriers synchronise), plus the terminal
    readout window. ``durations`` has ``prx`` and ``cz`` in ns."""
    gates = sum(max(durations["prx" if g.op == "r" else "cz"] for g in layer) for layer in design.layers)
    return gates + terminal_window_ns
